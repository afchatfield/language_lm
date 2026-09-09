"""A small, compressed language model that finds and explains grammar errors.

The package is organised by pipeline stage, mirroring the phases in `roadmap.md`:

* `langlm.data`       corpus loading, the M2 annotation format, split freezing
* `langlm.lt`         LanguageTool client (the training-signal source)
* `langlm.tokenizers` tokenizer fertility and vocabulary-trim analysis
* `langlm.corruptors` synthetic error injection            (Phase 2)
* `langlm.eval`       M2 scorer, ERRANT, overcorrection     (Phase 1)
* `langlm.compress`   vocabulary trimming, depth pruning    (Phase 4)
"""

__version__ = "0.1.0"
