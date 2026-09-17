"""Error injection: turning correct text into the training data's input side.

`base` holds the framework -- what a corruption is, and how a set of them turns
a correct sentence into an M2 block whose edits repair it. `de` and `es` hold
each language's rules, in their own registries (`langlm.corruptors.es.REGISTRY`,
`.NEEDS_PARSE`) rather than the module-level `REGISTRY`/`NEEDS_PARSE` exported
below, which stay German's: the two files reuse rule names for rules that mean
different things in each language, and one shared namespace would raise on the
first collision at import time. Importing this package registers both.
"""

from __future__ import annotations

from langlm.corruptors import (
    de,  # noqa: F401  (registers the German rules)
    es,  # noqa: F401  (registers the Spanish rules, in their own tables)
)
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
