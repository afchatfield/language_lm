"""Loading the training file into batches the trainer can consume."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langlm.config import PROCESSED_DIR
from langlm.train.format import IGNORE, Encoded, encode

#: Where `scripts/build_training_set.py` puts its output.
TRAINING_FILE = PROCESSED_DIR / "training" / "de.jsonl"


def read_records(path: Path = TRAINING_FILE) -> Iterator[dict[str, Any]]:
    """Yield the training records.

    Raises:
        FileNotFoundError: with the command that builds the file, because the
            most likely reason to hit this is a fresh clone.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No training data at {path}. Build it with `make training-set` "
            f"(which needs `make data corpora dictionary clean-corpus` first)."
        )
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


@dataclass
class SplitSizes:
    """How the file was divided."""

    train: int
    validation: int
    dropped: int


def build(
    tokenizer,
    path: Path = TRAINING_FILE,
    max_length: int = 1024,
    validation: int = 500,
    seed: int = 20260909,
    limit: int | None = None,
    changes_weight: float = 1.0,
) -> tuple[list[Encoded], list[Encoded], SplitSizes]:
    """Encode the file and hold out a validation slice.

    The validation slice comes from the same generated corpus, so it measures
    whether the model learned the task rather than whether it generalises to
    real learner German. That second question is what Falko-MERLIN dev is for,
    and it stays the thing the model is actually judged on.
    """
    records = list(read_records(path))
    if limit:
        records = records[:limit]
    random.Random(seed).shuffle(records)

    encoded, dropped = [], 0
    for record in records:
        item = encode(record, tokenizer, max_length=max_length, changes_weight=changes_weight)
        if item is None:
            dropped += 1
            continue
        encoded.append(item)

    held = min(validation, max(len(encoded) // 10, 1))
    return (
        encoded[held:],
        encoded[:held],
        SplitSizes(train=len(encoded) - held, validation=held, dropped=dropped),
    )


def collate(batch: list[Encoded], pad_token_id: int) -> dict:
    """Pad a batch to its own longest member.

    Right padding, and the padding is masked out of the loss as well as out of
    attention. Padding to the batch maximum rather than to `max_length` is most
    of the throughput on short data: the median example here is 138 tokens and
    the longest is 339.

    `loss_weights` rides along beside `labels`, padded with zero. It is only
    read by `WeightedTrainer` in `scripts/train_sft.py`, and it is all ones over
    the answer unless a run asked for something else, so a batch from here is
    still a plain causal-LM batch to anything that ignores the extra key.
    """
    import torch

    width = max(len(item) for item in batch)
    input_ids, labels, attention, weights = [], [], [], []
    for item in batch:
        padding = width - len(item)
        input_ids.append(item.input_ids + [pad_token_id] * padding)
        labels.append(item.labels + [IGNORE] * padding)
        attention.append([1] * len(item) + [0] * padding)
        weights.append(item.weights + [0.0] * padding)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention),
        "loss_weights": torch.tensor(weights, dtype=torch.float32),
    }
