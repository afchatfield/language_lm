#!/usr/bin/env python
"""Merge a LoRA adapter into its base and write a standalone checkpoint.

    python scripts/merge_adapter.py \
        --base checkpoints/base-trimmed-es \
        --adapter checkpoints/phase3-es-cw025 \
        --out checkpoints/ship-es

Nothing here changes what the model computes: `merge_and_unload` folds `B@A`
into the frozen weights it was trained against, which is arithmetic, not
retraining. What it changes is what the artefact *is* -- a single set of
weights that `convert_hf_to_gguf.py` can read, with no PEFT at load time.
Serving is the only reason to do it, which is why it lives here and not in
the training script.

**The tokenizer comes from the base, not the adapter.** On a vocabulary-trimmed
base this is the whole game: the adapter directory still carries the *untrimmed*
tokenizer it was trained with, because training never touched the vocabulary.
Saving that one beside trimmed weights produces a checkpoint whose tokenizer
emits ids the embedding matrix no longer has, and the failure is a wrong answer
rather than a crash. `langlm.compress.vocab` has the longer version of why the
four artefacts have to agree.

The adapter is *valid* on a trimmed base for the reason Phase 4 measured: the
Phase 3 LoRA targets no vocabulary-dimension module, so trimming rows out of
`embed_tokens` and `lm_head` leaves every tensor it adapts untouched. That is
not true of a depth-pruned base, where the adapter does not survive at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langlm.config import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base checkpoint, trimmed or not")
    parser.add_argument("--adapter", required=True, help="LoRA adapter directory")
    parser.add_argument("--out", required=True, help="Where to write the merged checkpoint")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # A base is either a checkpoint in this repo or a hub id. The uncompressed
    # row of the compression table is the second kind -- the adapter merged onto
    # the stock base model -- so refusing one would mean the table could not
    # measure its own starting point.
    local_base = PROJECT_ROOT / args.base
    base = local_base if local_base.exists() else args.base
    adapter = PROJECT_ROOT / args.adapter
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if not adapter.exists():
        raise SystemExit(f"No adapter at {adapter}")

    print(f"base:    {base}")
    print(f"adapter: {adapter}")

    model = AutoModelForCausalLM.from_pretrained(str(base), dtype=torch.bfloat16)
    before = model.get_input_embeddings().weight.shape[0]
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    after = model.get_input_embeddings().weight.shape[0]
    # A merge that changed the vocabulary dimension means the adapter carried
    # resized embeddings, and the checkpoint about to be written is not the one
    # that was measured. Cheap to assert, silent and wrong if it is not.
    if before != after:
        raise SystemExit(f"vocabulary changed during merge: {before:,} -> {after:,}")

    tokenizer = AutoTokenizer.from_pretrained(str(base))
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    (out / "merge.json").write_text(
        json.dumps(
            {
                "base": args.base,
                "adapter": str(adapter.relative_to(PROJECT_ROOT)),
                "vocab_size": after,
                "parameters": sum(p.numel() for p in model.parameters()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out} ({after:,} vocabulary rows)")


if __name__ == "__main__":
    main()
