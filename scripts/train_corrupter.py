#!/usr/bin/env python
"""LoRA fine-tune the base model to *make* German errors, not to fix them.

    python scripts/train_corrupter.py --limit 64 --epochs 1 --dry-run
    python scripts/train_corrupter.py

The generator learns from Falko-MERLIN train read backwards: the annotator's
corrected sentence in, the learner's original out, conditioned on how many
mistakes to make. `langlm.train.corrupter` has the schema and
`reports/phase3/headroom.md` the argument for why this is the lever worth
pulling. `scripts/generate_backtranslation.py` is what runs the result.

Model and LoRA settings come from `configs/phase3.yaml`, so the generator is the
same 1.7B base as the forward model -- there is no second checkpoint to manage,
and nothing here is shipped, so its size is a throughput question rather than a
footprint one.
"""

from __future__ import annotations

import argparse
import json
import random

from langlm.config import PROJECT_ROOT, load_config
from langlm.data import backtranslate
from langlm.train import corrupter
from langlm.train.format import IGNORE

#: Where the generator's adapter goes. Not `checkpoints/phase3-de*`: that prefix
#: is the shipped model's, and a tool that globs it should not find this.
DEFAULT_ADAPTER_DIR = "checkpoints/corrupter-de"

#: Held out from the reversed corpus, to watch for overfitting. Small, because
#: the corpus is small and every sentence held out here is one the generator
#: does not learn a mistake from.
VALIDATION = 300


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Use only the first N examples")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check the mask and one forward pass, then stop without training",
    )
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    config = load_config("phase3")
    model_cfg, lora_cfg, train_cfg = config["model"], config["lora"], config["training"]
    device = device_name()
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["repo_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = list(backtranslate.reversed_records(limit=args.limit))
    random.Random(train_cfg["seed"]).shuffle(records)
    encoded, dropped = [], 0
    for record in records:
        item = corrupter.encode(record, tokenizer, max_length=args.max_length)
        if item is None:
            dropped += 1
            continue
        encoded.append(item)
    held = min(VALIDATION, max(len(encoded) // 10, 1))
    train_set, val_set = encoded[held:], encoded[:held]
    print(f"train {len(train_set):,} | validation {len(val_set):,} | dropped {dropped:,}")
    report_mask(train_set[0], tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["repo_id"], dtype=getattr(torch, model_cfg["dtype"])
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    if args.dry_run:
        dry_run(model, train_set[:2], tokenizer, device)
        return

    output = PROJECT_ROOT / args.adapter_dir
    if output.exists() and any(output.iterdir()):
        raise SystemExit(
            f"{output} already holds an adapter. Training would overwrite it. Move it "
            f"aside, or pass --adapter-dir to write somewhere else."
        )
    print(
        f"batch {args.batch_size} x {args.gradient_accumulation} accumulation = "
        f"{args.batch_size * args.gradient_accumulation} effective | adapter -> {output}"
    )
    arguments = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_steps=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        bf16=(device == "cuda"),
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=2,
        seed=train_cfg["seed"],
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    trainer.train()

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    (output / "training_summary.json").write_text(
        json.dumps(
            {
                "task": "error generation (Falko-MERLIN train, reversed)",
                "device": device,
                "train_examples": len(train_set),
                "validation_examples": len(val_set),
                "dropped": dropped,
                "effective": {
                    "batch_size": args.batch_size,
                    "gradient_accumulation": args.gradient_accumulation,
                    "effective_batch": args.batch_size * args.gradient_accumulation,
                    "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "max_length": args.max_length,
                },
                "model": model_cfg,
                "lora": lora_cfg,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nAdapter written to {output}")


def device_name() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def collate(batch, pad_token_id: int) -> dict:
    """Pad a batch to its own longest member, masking the padding out of the loss."""
    import torch

    width = max(len(item) for item in batch)
    input_ids, labels, attention = [], [], []
    for item in batch:
        padding = width - len(item)
        input_ids.append(item.input_ids + [pad_token_id] * padding)
        labels.append(item.labels + [IGNORE] * padding)
        attention.append([1] * len(item) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention),
    }


def report_mask(example, tokenizer) -> None:
    """Print what the loss is computed over.

    The same check the forward trainer prints, and it matters more here, not
    less: a generator that trained on its prompts would learn to emit correct
    German, which is the one thing this model must not do.
    """
    supervised = [
        token
        for token, label in zip(example.input_ids, example.labels, strict=True)
        if label != IGNORE
    ]
    ignored = [
        token
        for token, label in zip(example.input_ids, example.labels, strict=True)
        if label == IGNORE
    ]
    print("\nloss mask, on one example:")
    print(f"  ignored   ({len(ignored):>3} tokens): {tokenizer.decode(ignored)!r}")
    print(f"  supervised({len(supervised):>3} tokens): {tokenizer.decode(supervised)!r}\n")


def dry_run(model, examples, tokenizer, device: str) -> None:
    """One forward pass, to prove the batch is shaped the way the model expects."""
    import torch

    model.to(device)
    batch = collate(examples, tokenizer.pad_token_id)
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        loss = model(**batch).loss
    print(f"dry run: loss {loss.item():.4f} over {batch['input_ids'].shape} -- nothing trained")


if __name__ == "__main__":
    main()
