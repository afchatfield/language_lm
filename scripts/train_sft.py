#!/usr/bin/env python
"""LoRA fine-tune the base model on the Phase 2 training set.

    python scripts/train_sft.py                 # the real run
    python scripts/train_sft.py --limit 64 --epochs 1 --dry-run

Device is picked automatically: CUDA if there is one, else Apple MPS, else CPU.
The intended path is an H200; the M1 is for `--dry-run`, which checks the loss
mask, the schema and one forward pass without training anything.

Everything else comes from `configs/phase3.yaml`.
"""

from __future__ import annotations

import argparse
import json

from langlm.config import PROJECT_ROOT, load_config
from langlm.train import data as train_data
from langlm.train.format import IGNORE


def device_name() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Use only the first N examples")
    parser.add_argument("--epochs", type=float, help="Override the configured epochs")
    parser.add_argument("--batch-size", type=int, help="Override the per-device batch size")
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

    train_set, val_set, sizes = train_data.build(
        tokenizer,
        max_length=train_cfg["max_length"],
        validation=train_cfg["validation_size"],
        seed=train_cfg["seed"],
        limit=args.limit,
    )
    print(f"train {sizes.train:,} | validation {sizes.validation:,} | dropped {sizes.dropped:,}")
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

    output = PROJECT_ROOT / config["output"]["adapter_dir"]
    arguments = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=args.epochs or train_cfg["epochs"],
        per_device_train_batch_size=args.batch_size or train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_ratio=train_cfg["warmup_ratio"],
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
        data_collator=lambda batch: train_data.collate(batch, tokenizer.pad_token_id),
    )
    trainer.train()

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    (output / "training_summary.json").write_text(
        json.dumps(
            {
                "device": device,
                "train_examples": sizes.train,
                "validation_examples": sizes.validation,
                "dropped": sizes.dropped,
                "config": config,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nAdapter written to {output}")


def report_mask(example, tokenizer) -> None:
    """Print what the loss is actually computed over.

    The roadmap says to verify the mask before launching anything long, and the
    cheapest way to keep that honest is to make every run show it. A run that
    silently trains on its prompts looks exactly like one that does not, until
    the model starts inventing learner sentences instead of correcting them.
    """
    pairs = zip(example.input_ids, example.labels, strict=True)
    supervised = [i for i, label in pairs if label != IGNORE]
    masked = len(example) - len(supervised)
    print(
        f"loss mask: {masked} prompt tokens ignored, {len(supervised)} answer tokens "
        f"supervised ({len(supervised) / len(example):.0%} of the sequence)"
    )
    print(f"  supervised text: {tokenizer.decode(supervised)[:110]}")


def dry_run(model, examples, tokenizer, device: str) -> None:
    """One forward pass, to prove the shapes and the loss are real."""
    import torch

    model.to(device).train()
    batch = train_data.collate(examples, tokenizer.pad_token_id)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        loss = model(**batch).loss
    print(f"dry run: batch {tuple(batch['input_ids'].shape)}, loss {loss.item():.3f}")
    print("Mask, schema and forward pass all fine. Nothing was trained.")


if __name__ == "__main__":
    main()
