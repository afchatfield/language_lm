"""Evaluation: the M2 (MaxMatch) scorer, German ERRANT, and overcorrection.

Phase 1 exists so that every later claim is falsifiable. Three things live here:

* :mod:`langlm.eval.m2_scorer`   MaxMatch F0.5 over raw corrected sentences.
* :mod:`langlm.eval.errant_de`   German ERRANT: edit extraction and typing.
* :mod:`langlm.eval.overcorrection`  what a system does to text that was already correct.

The scorers take *corrected sentences*, not edits, because that is what every
system under test actually produces -- LanguageTool, a prompted base model, and
a fine-tuned one alike.
"""
