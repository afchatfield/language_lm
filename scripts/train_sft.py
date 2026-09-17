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
from functools import cache

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
        "--gradient-accumulation",
        type=int,
        help="Override the accumulation steps. Pair it with --batch-size to keep the "
        "effective batch the same when the per-device one has to shrink to fit.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Recompute activations instead of storing them: much less memory, ~30%% slower",
    )
    parser.add_argument(
        "--adapter-dir",
        help="Write the adapter here instead of the configured directory, so a run "
        "cannot overwrite one that is still the best model",
    )
    parser.add_argument(
        "--changes-loss-weight",
        type=float,
        help="Multiplier on the loss over the `changes` list. 1.0 is uniform "
        "cross-entropy over the answer, which is what every run before this one "
        "used -- and which puts 70%% of the gradient on a field F0.5 does not read.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        help="Override the configured LoRA dropout",
    )
    parser.add_argument(
        "--resume-from",
        help="Resume from a checkpoint directory written by an interrupted run. "
        "Optimiser, scheduler and RNG state come back with it, so the result is "
        "the run that was interrupted rather than a fresh one warm-started. Pass "
        "the same --batch-size and --gradient-accumulation the checkpoint was "
        "written with, or the data order will not line up.",
    )
    parser.add_argument(
        "--base",
        help="Load base weights and tokenizer from here instead of the configured repo. "
        "For a compressed checkpoint the tokenizer must travel with it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check the mask and one forward pass, then stop without training",
    )
    parser.add_argument(
        "--language",
        default="de",
        choices=["de", "es"],
        help="Which language's config, prompt instruction and training file to use. "
        "German kept as the default so every existing invocation is unchanged.",
    )
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

    # `configs/phase3.yaml` stays unsuffixed -- German was here first, and the
    # documented H200 runbook (`docs/h200.md`) and `make phase3` both name it
    # directly. A second language gets its own `phase3_<lang>.yaml`.
    config = load_config("phase3" if args.language == "de" else f"phase3_{args.language}")
    model_cfg, lora_cfg, train_cfg = config["model"], config["lora"], config["training"]
    device = device_name()
    print(f"device: {device}")

    base_model = str(PROJECT_ROOT / args.base) if args.base else model_cfg["repo_id"]
    if args.base:
        print(f"base: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    changes_weight = (
        args.changes_loss_weight
        if args.changes_loss_weight is not None
        else train_cfg.get("changes_loss_weight", 1.0)
    )
    dropout = args.lora_dropout if args.lora_dropout is not None else lora_cfg["dropout"]

    train_set, val_set, sizes = train_data.build(
        tokenizer,
        max_length=train_cfg["max_length"],
        validation=train_cfg["validation_size"],
        seed=train_cfg["seed"],
        limit=args.limit,
        changes_weight=changes_weight,
        language=args.language,
    )
    print(f"train {sizes.train:,} | validation {sizes.validation:,} | dropped {sizes.dropped:,}")
    report_mask(train_set[0], tokenizer)
    report_weights(train_set[:500], changes_weight)

    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=getattr(torch, model_cfg["dtype"])
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=dropout,
            target_modules=lora_cfg["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    if args.dry_run:
        dry_run(model, train_set[:2], tokenizer, device)
        return

    output = PROJECT_ROOT / (args.adapter_dir or config["output"]["adapter_dir"])
    # A resume writes back into the directory it is continuing, so the
    # directory being non-empty is the normal case rather than the accident
    # this guard exists to catch.
    if output.exists() and any(output.iterdir()) and not args.resume_from:
        raise SystemExit(
            f"{output} already holds an adapter. Training would overwrite it. Move it "
            f"aside, or pass --adapter-dir to write somewhere else."
        )
    batch_size = args.batch_size or train_cfg["batch_size"]
    accumulation = args.gradient_accumulation or train_cfg["gradient_accumulation"]
    print(
        f"batch {batch_size} x {accumulation} accumulation = {batch_size * accumulation} "
        f"effective | adapter -> {output}"
    )
    arguments = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=args.epochs or train_cfg["epochs"],
        per_device_train_batch_size=batch_size,
        # Evaluation gets the same batch as training. The default is 8 whatever
        # the training batch is, which on a shared card means the run survives
        # training and then dies at the first evaluation step.
        per_device_eval_batch_size=batch_size,
        # Nothing here reads anything but `eval_loss`, and without this the
        # trainer still gathers the logits to hand to a `compute_metrics` that
        # does not exist -- then converts them to fp32. At this vocabulary that
        # is batch x length x 128k x 4 bytes, about 8GB, allocated on top of a
        # training step that has already peaked. The run survives training and
        # dies at the first evaluation, which is the same failure the comment
        # above describes and the same fix does not cover.
        prediction_loss_only=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=accumulation,
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        # transformers 5 dropped `warmup_ratio`; `warmup_steps` takes the ratio
        # directly when it is a float below 1.
        warmup_steps=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        bf16=(device == "cuda"),
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg.get("save_total_limit", 2),
        seed=train_cfg["seed"],
        report_to=[],
    )
    trainer = weighted_trainer()(
        model=model,
        args=arguments,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=lambda batch: train_data.collate(batch, tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=args.resume_from)

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
                # What ran, not what was configured. The two differ whenever a
                # run had to be squeezed onto a card that was already busy, and
                # a summary that reported the config would quietly attribute the
                # result to settings nothing used.
                "effective": {
                    "batch_size": batch_size,
                    "gradient_accumulation": accumulation,
                    "effective_batch": batch_size * accumulation,
                    "epochs": args.epochs or train_cfg["epochs"],
                    "gradient_checkpointing": args.gradient_checkpointing,
                    "changes_loss_weight": changes_weight,
                    "lora_dropout": dropout,
                },
                "config": config,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nAdapter written to {output}")


@cache
def weighted_trainer():
    """The `Trainer` subclass that respects the per-token weights in the batch.

    Hugging Face computes the answer's cross-entropy as a flat mean over every
    unmasked token. That is a choice, not a neutral default: the `changes` list
    is 70% of the supervised tokens, so a flat mean spends most of the gradient
    on the field the F0.5 score never reads. This takes a weighted mean instead,
    with the weights `langlm.train.format.encode` attached.

    With the default weight of 1.0 everywhere this is the flat mean again, so a
    control run and a weighted run go through the same code and differ only in
    the number they were given.

    Built behind a function because `transformers` is imported lazily here, so
    that `--help` and a bad argument answer immediately rather than after the
    library loads.
    """
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
            import torch

            # Pop it before the forward pass: the model signature has no such
            # argument, and passing it through would raise rather than be ignored.
            weights = inputs.pop("loss_weights", None)
            labels = inputs["labels"]
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})

            # Next-token prediction: position i predicts token i+1.
            logits = outputs.logits[:, :-1, :]
            targets = labels[:, 1:]
            weights = (targets != IGNORE).to(logits.dtype) if weights is None else weights[:, 1:]

            flat_targets = targets.reshape(-1)
            per_token = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                flat_targets.masked_fill(flat_targets == IGNORE, 0),
                reduction="none",
            )
            flat = weights.reshape(-1).to(per_token.dtype)
            # Masked positions carry weight 0, so they drop out of both sums and
            # need no separate mask here.
            loss = (per_token * flat).sum() / flat.sum().clamp(min=1e-8)
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def report_weights(examples, changes_weight: float) -> None:
    """Print what fraction of the gradient each field of the answer gets.

    The point of the dial is invisible in a loss curve, so it is printed where
    the mask report is printed, for the same reason: a run that silently trains
    on the wrong thing looks exactly like one that does not.
    """
    supervised = sum(sum(1 for w in e.weights if w > 0) for e in examples)
    total = sum(sum(e.weights) for e in examples)
    full = sum(sum(1 for w in e.weights if w == 1.0) for e in examples)
    if not supervised:
        return
    print(
        f"loss weights: changes x{changes_weight:g} | {full / supervised:.1%} of supervised "
        f"tokens at full weight, carrying {full / total:.1%} of the gradient"
    )


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
