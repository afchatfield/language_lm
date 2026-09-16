#!/usr/bin/env python
"""Heal a vocabulary-trimmed base model with a short LM-objective pass on German.

    python scripts/heal_trimmed.py --base checkpoints/base-trimmed

**What healing is actually for.** Trimming removes rows; it does not touch the
rows it keeps. So the damage is not in the weights, it is in the *tokenization*:
with the rare rows gone, 4.25% of German sentences segment differently, and the
model meets pieces in combinations it was never trained on. That is the whole
injury, and it is why healing is a short pass on ordinary German rather than a
retraining.

The exit condition is the same one the rest of the project uses -- what it does
to F0.5 on dev, not what it does to the LM loss. `reports/phase4/gate.md` has the
untrained trimmed model's task score, which is what this has to beat to justify
its cost.

This is a full-parameter pass, not LoRA. A LoRA cannot repair an embedding table
it does not target, and the embedding table is the thing the trim disturbed.
"""

from __future__ import annotations

import argparse
import json

from langlm.config import PROCESSED_DIR, PROJECT_ROOT, load_config
from langlm.data.m2 import parse_m2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="checkpoints/base-trimmed")
    parser.add_argument("--out", default="checkpoints/base-trimmed-healed")
    parser.add_argument("--sentences", type=int, default=60_000)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true", help="Report the data and stop")
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    base = PROJECT_ROOT / args.base
    if not base.exists():
        raise SystemExit(f"No trimmed checkpoint at {base}. Run scripts/trim_vocab.py first.")

    # The tokenizer must come from the trimmed checkpoint. Reaching for the
    # original would pair 128,000 ids with 39,399 rows, which indexes out of
    # range on a good day and silently mislabels every row on a bad one.
    tokenizer = AutoTokenizer.from_pretrained(str(base))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    corpus = PROCESSED_DIR / "clean_de" / "de.m2"
    if not corpus.exists():
        raise SystemExit(f"No clean German corpus at {corpus}. Run `make clean-corpus`.")
    texts = [s.source for s in parse_m2(corpus)][: args.sentences]
    print(f"{len(texts):,} clean German sentences from {corpus.name}")

    # The overcorrection set is drawn from this same clean pool by construction,
    # so healing on all of it would train on the sentences the overcorrection
    # metric is measured over. `build_clean_corpus.py` already excludes them --
    # asserted here because the cost of being wrong is a silently flattering
    # overcorrection number in the Phase 4 ablation table.
    from langlm.data import overcorrection as overcorrection_data
    from langlm.data.splits import load_split

    held = {s.source for s in load_split(overcorrection_data.CORPUS, "all")}
    leaked = held & set(texts)
    if leaked:
        raise SystemExit(f"{len(leaked)} overcorrection sentences are in the healing corpus.")
    print(f"overcorrection set: {len(held):,} sentences, none present")

    if args.dry_run:
        for text in texts[:3]:
            print(f"  {text[:90]}")
        return

    def encode(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length, padding=False)

    dataset = Dataset.from_dict({"text": texts}).map(
        encode, batched=True, remove_columns=["text"], desc="Tokenizing"
    )

    model = AutoModelForCausalLM.from_pretrained(str(base), dtype=torch.bfloat16)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"vocab_size {model.config.vocab_size:,} | {parameters:,} params")

    output = PROJECT_ROOT / args.out
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"{output} is not empty. Move it aside or pass --out.")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output),
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=0.03,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            bf16=torch.cuda.is_available(),
            logging_steps=25,
            save_strategy="no",
            seed=load_config("phase3")["training"]["seed"],
            report_to=[],
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    result = trainer.train()

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))

    # save_pretrained writes the tokenizer the library holds in memory, which for
    # a trimmed checkpoint is already the trimmed one -- but the proto is copied
    # from wherever the tokenizer was loaded, so it comes across correctly only
    # because that was the trimmed directory. Verified rather than assumed.
    import json as _json

    written = _json.loads((output / "config.json").read_text())["vocab_size"]
    if written != model.config.vocab_size:
        raise SystemExit(
            f"config.json says vocab_size {written}, model says {model.config.vocab_size}"
        )

    (output / "healing_summary.json").write_text(
        json.dumps(
            {
                "base": args.base,
                "sentences": len(texts),
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "effective_batch": args.batch_size * args.gradient_accumulation,
                "max_length": args.max_length,
                "train_loss": result.training_loss,
                "vocab_size": model.config.vocab_size,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nHealed checkpoint written to {output} (train loss {result.training_loss:.4f})")


if __name__ == "__main__":
    main()
