# Phase 2: the injected error distribution

Natural rates are Falko-MERLIN train + dev. Recall is the Phase 1 few-shot baseline, recomputed from its saved output. A type the baseline recovers at 0.10 or less is weighted up 2x.

| Error type | Natural | Few-shot recall | Boost | Injected |
|---|---:|---:|---:|---:|
| `R:SPELL` | 13.3% | 0.49 | 1x | **20.8%** |
| `R:DET:FORM` | 10.8% | 0.24 | 1x | **16.8%** |
| `M:PUNCT` | 9.6% | 0.37 | 1x | **14.9%** |
| `R:ORTH` | 7.7% | 0.42 | 1x | **12.0%** |
| `R:ADP` | 3.4% | 0.04 | **x2** | **10.7%** |
| `M:DET` | 3.1% | 0.05 | **x2** | **9.8%** |
| `R:WO` | 2.6% | 0.09 | **x2** | **8.0%** |
| `R:ADJ:FORM` | 4.6% | 0.44 | 1x | **7.1%** |

Boosted: `M:DET`, `R:ADP`, `R:WO`. These are the types the prompted baseline barely touches, so an example of one buys more than an example of a type it already handles.

Rates are renormalised over the types the corruptors can produce, so they do not match the corpus-wide histogram directly: types no rule injects -- verb morphology, word order -- are absent here and their share is redistributed.

## The block for `configs/phase2.yaml`

```yaml
injection:
  errors_per_sentence: [1, 2]
  weights:
    R:SPELL: 0.2079
    R:DET:FORM: 0.1677
    M:PUNCT: 0.1489
    R:ORTH: 0.1201
    R:ADP: 0.1065
    M:DET: 0.0979
    R:WO: 0.0797
    R:ADJ:FORM: 0.0712
```
