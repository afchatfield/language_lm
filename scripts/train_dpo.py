#!/usr/bin/env python
"""Preference-train the SFT model on pairs drawn from its own beam.

    python scripts/build_preference_pairs.py --adapter checkpoints/phase3-de-iter4-cw025 \
        --name iter4-cw025
    python scripts/train_dpo.py --pairs iter4-cw025 \
        --adapter checkpoints/phase3-de-iter4-cw025 \
        --adapter-dir checkpoints/phase3-de-iter4-cw025-dpo

The argument for doing this at all is in `src/langlm/train/preference.py` and
`reports/phase3/nbest.md`: the beam holds a much better correction than the model
ranks first, and no selector reading features from outside the model finds it.

**The reference model is the SFT adapter, not the base.** DPO's KL term anchors
the policy to a reference, and the thing worth staying near is the model that
scores 0.7072, not the model that scores 0.2571 zero-shot. With a LoRA policy the
reference comes free: disabling the adapter *is* the reference forward pass, so
nothing needs a second copy of the weights in memory.

Configuration is `configs/phase3_dpo.yaml`. The defaults are deliberately timid --
one epoch, a tenth of the SFT learning rate, beta 0.1 -- because this starts from
a model that is already the best one and the failure mode of preference training
is not underfitting.
"""

from __future__ import annotations

import argparse
import json

from langlm.config import INTERIM_DIR, PROJECT_ROOT, load_config

PAIRS_DIR = INTERIM_DIR / "phase3" / "preference"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        default="iter4-cw025",
        help="Name of the pair file under data/interim/phase3/preference",
    )
    parser.add_argument(
        "--adapter",
        default="checkpoints/phase3-de-iter4-cw025",
        help="The SFT adapter to start from and to use as the DPO reference",
    )
    parser.add_argument("--adapter-dir", help="Where to write the trained adapter")
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--beta", type=float, help="KL strength. Higher stays nearer the reference."
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--min-gain", type=int, default=1, help="Keep pairs gaining at least this")
    parser.add_argument("--limit", type=int, help="Use only the first N pairs")
    parser.add_argument("--dry-run", action="store_true", help="Report the data and stop")
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    config = load_config("phase3_dpo")
    train_cfg = config["training"]
    base = load_config("phase3")["model"]

    path = PAIRS_DIR / f"{args.pairs}.pairs.jsonl"
    if not path.exists():
        raise SystemExit(f"No pairs at {path}. Run scripts/build_preference_pairs.py first.")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row["gain"] >= args.min_gain]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No pairs left after filtering.")

    # A small pair set -- a smoke run, or a hard --min-gain -- must not lose most
    # of itself to a validation split sized for the full one.
    validation = min(train_cfg["validation_size"], max(1, len(rows) // 10))
    dataset = Dataset.from_list(
        [{k: row[k] for k in ("prompt", "chosen", "rejected")} for row in rows]
    )
    split = dataset.train_test_split(test_size=validation, seed=train_cfg["seed"])
    print(
        f"{len(rows):,} pairs | train {len(split['train']):,} | validation {len(split['test']):,}"
    )
    print(
        f"mean gain {sum(r['gain'] for r in rows) / len(rows):.2f} tp-fp, "
        f"mean chosen rank {sum(r['chosen_rank'] for r in rows) / len(rows):.2f}"
    )

    if args.dry_run:
        for row in rows[:2]:
            print(f"\n  chosen  : {row['chosen'][:160]}")
            print(f"  rejected: {row['rejected'][:160]}")
        return

    adapter = PROJECT_ROOT / args.adapter
    if not adapter.exists():
        raise SystemExit(f"No adapter at {adapter}")

    tokenizer = AutoTokenizer.from_pretrained(str(adapter))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base["repo_id"], dtype=getattr(torch, base["dtype"])
    )
    # Loaded trainable, and left as the only adapter on the model. TRL detects a
    # PEFT model and takes the reference log-probabilities with the adapter
    # disabled, which is the SFT model exactly -- no second copy, no drift.
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
    model.print_trainable_parameters()

    output = PROJECT_ROOT / (args.adapter_dir or config["output"]["adapter_dir"])
    if output.exists() and any(output.iterdir()):
        raise SystemExit(
            f"{output} already holds an adapter. Training would overwrite it. Move it "
            f"aside, or pass --adapter-dir to write somewhere else."
        )
    if output == adapter:
        raise SystemExit("Refusing to overwrite the reference adapter with its own DPO run.")

    batch_size = args.batch_size or train_cfg["batch_size"]
    accumulation = args.gradient_accumulation or train_cfg["gradient_accumulation"]
    beta = args.beta if args.beta is not None else train_cfg["beta"]
    learning_rate = args.learning_rate or float(train_cfg["learning_rate"])
    epochs = args.epochs or train_cfg["epochs"]
    print(f"beta {beta} | lr {learning_rate:g} | {epochs} epoch(s) | adapter -> {output}")

    arguments = DPOConfig(
        output_dir=str(output),
        beta=beta,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_steps=train_cfg["warmup_ratio"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accumulation,
        max_length=train_cfg["max_length"],
        # trl 1.12 dropped the separate prompt cap; `truncation_mode` decides which
        # end goes when a pair exceeds `max_length`, and losing the start of the
        # prompt is better than losing the completion the preference is over.
        truncation_mode=train_cfg["truncation_mode"],
        bf16=torch.cuda.is_available(),
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg.get("save_total_limit", 4),
        seed=train_cfg["seed"],
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=arguments,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
    )
    trainer.train()

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    (output / "training_summary.json").write_text(
        json.dumps(
            {
                "reference_adapter": args.adapter,
                "pairs": str(path),
                "pairs_used": len(rows),
                "effective": {
                    "beta": beta,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "gradient_accumulation": accumulation,
                    "effective_batch": batch_size * accumulation,
                    "min_gain": args.min_gain,
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


if __name__ == "__main__":
    main()
