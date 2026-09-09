"""Supervised fine-tuning: the format, the data, and the loss mask."""

from __future__ import annotations

from langlm.train.format import IGNORE, build_prompt, build_target, encode, parse_answer

__all__ = ["IGNORE", "build_prompt", "build_target", "encode", "parse_answer"]
