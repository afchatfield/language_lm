# Phase 2: the injected error distribution

Natural rates are Falko-MERLIN train + dev. Recall is the **fine-tuned** system, recomputed from its saved dev output. A type it recovers at 0.10 or less is weighted up 2x.

| Error type | Natural | fine-tuned recall | Boost | Injected |
|---|---:|---:|---:|---:|
| `R:SPELL` | 13.3% | 0.64 | 1x | **16.7%** |
| `R:DET:FORM` | 10.8% | 0.79 | 1x | **13.4%** |
| `M:PUNCT` | 9.6% | 0.70 | 1x | **11.9%** |
| `R:OTHER` | 8.0% | 0.30 | 1x | **10.0%** |
| `R:ORTH` | 7.7% | 0.70 | 1x | **9.6%** |
| `R:ADJ:FORM` | 4.6% | 0.79 | 1x | **5.7%** |
| `R:NOUN:FORM` | 4.0% | 0.72 | 1x | **4.9%** |
| `U:PUNCT` | 3.6% | 0.58 | 1x | **4.5%** |
| `R:ADP` | 3.4% | 0.38 | 1x | **4.3%** |
| `M:DET` | 3.1% | 0.39 | 1x | **3.9%** |
| `R:WO` | 2.6% | 0.57 | 1x | **3.2%** |
| `M:PRON` | 2.4% | 0.29 | 1x | **3.0%** |
| `R:VERB:FORM` | 1.9% | 0.58 | 1x | **2.4%** |
| `M:AUX` | 1.5% | 0.51 | 1x | **1.9%** |
| `R:AUX:FORM` | 1.4% | 0.50 | 1x | **1.7%** |
| `M:ADP` | 1.2% | 0.26 | 1x | **1.5%** |
| `R:PRON:FORM` | 1.0% | 0.62 | 1x | **1.3%** |

Boosted: none. Nothing the corruptors produce falls under the 0.10 recall the boost is for -- the fine-tuned system's weakest injectable type is at 0.26. The boost was calibrated against a prompted baseline that scored 0.04 on prepositions; a model that is no longer blind anywhere should be trained on the distribution learners actually produce, which is what these weights are.

Rates are renormalised over the types the corruptors can produce, so they do not match the corpus-wide histogram directly: types no rule injects are absent here and their share is redistributed.

## The block for `configs/phase2.yaml`

```yaml
injection:
  errors_per_sentence: [1, 4]
  weights:
    R:SPELL: 0.1665
    R:DET:FORM: 0.1343
    M:PUNCT: 0.1193
    R:OTHER: 0.0997
    R:ORTH: 0.0962
    R:ADJ:FORM: 0.0570
    R:NOUN:FORM: 0.0493
    U:PUNCT: 0.0445
    R:ADP: 0.0427
    M:DET: 0.0392
    R:WO: 0.0319
    M:PRON: 0.0301
    R:VERB:FORM: 0.0243
    M:AUX: 0.0193
    R:AUX:FORM: 0.0172
    M:ADP: 0.0153
    R:PRON:FORM: 0.0131
```
