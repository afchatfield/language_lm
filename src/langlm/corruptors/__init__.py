"""Error injection: turning correct text into the training data's input side.

`base` holds the framework -- what a corruption is, and how a set of them turns
a correct sentence into an M2 block whose edits repair it. `de` holds the German
rules. Importing this package registers them.
"""

from __future__ import annotations

from langlm.corruptors import de  # noqa: F401  (registers the German rules)
from langlm.corruptors.base import (
    NEEDS_PARSE,
    REGISTRY,
    Corruption,
    apply,
    choose,
    corruptor,
    parse_corruptor,
    propose,
)

__all__ = [
    "NEEDS_PARSE",
    "REGISTRY",
    "Corruption",
    "apply",
    "choose",
    "corruptor",
    "parse_corruptor",
    "propose",
]
