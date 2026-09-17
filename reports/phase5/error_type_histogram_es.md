# COWS-L2H error-type distribution

Computed over the **train + dev** splits only (72,472 edits), typed by `errant_es`. The test split is held out and was not read.

Mirrors `error_type_histogram.md` from Phase 0, which Phase 2 used to weight German synthetic error injection to match real learner errors. This is the same table for `corruptors/es.py`.

| Error type | Count | Share |
|---|---:|---:|
| `R:OTHER` | 12,057 | 16.64% |
| `R:ORTH` | 8,357 | 11.53% |
| `U:PRON` | 5,908 | 8.15% |
| `R:SPELL` | 3,401 | 4.69% |
| `R:VERB:FORM` | 3,203 | 4.42% |
| `R:DET:FORM` | 3,158 | 4.36% |
| `M:ADP` | 2,960 | 4.08% |
| `R:ADP` | 2,884 | 3.98% |
| `M:PUNCT` | 2,461 | 3.40% |
| `M:DET` | 2,454 | 3.39% |
| `U:DET` | 2,227 | 3.07% |
| `R:VERB` | 2,097 | 2.89% |
| `R:NOUN` | 1,944 | 2.68% |
| `R:ADJ:FORM` | 1,840 | 2.54% |
| `U:OTHER` | 1,349 | 1.86% |
| `R:AUX:FORM` | 1,312 | 1.81% |
| `M:OTHER` | 1,176 | 1.62% |
| `U:ADP` | 1,169 | 1.61% |
| `U:PUNCT` | 1,113 | 1.54% |
| `M:PRON` | 984 | 1.36% |
| `R:AUX` | 864 | 1.19% |
| `R:PRON:FORM` | 847 | 1.17% |
| `R:DET` | 822 | 1.13% |
| `R:NOUN:FORM` | 713 | 0.98% |
| `R:MORPH` | 696 | 0.96% |
| `R:WO` | 657 | 0.91% |
| `R:ADJ` | 613 | 0.85% |
| `R:PNOUN` | 530 | 0.73% |
| `R:PRON` | 421 | 0.58% |
| `R:CONJ` | 335 | 0.46% |
| `R:ADV` | 310 | 0.43% |
| `M:AUX` | 299 | 0.41% |
| `M:SCONJ` | 290 | 0.40% |
| `M:PNOUN` | 274 | 0.38% |
| `U:ADV` | 261 | 0.36% |
| `U:PNOUN` | 260 | 0.36% |
| `U:NOUN` | 241 | 0.33% |
| `M:ADJ` | 239 | 0.33% |
| `U:ADJ` | 233 | 0.32% |
| `M:VERB` | 215 | 0.30% |
| `R:PUNCT` | 182 | 0.25% |
| `M:NOUN` | 180 | 0.25% |
| `U:SCONJ` | 173 | 0.24% |
| `M:ADV` | 162 | 0.22% |
| `U:VERB` | 126 | 0.17% |
| `U:AUX` | 116 | 0.16% |
| `R:SCONJ` | 107 | 0.15% |
| `M:CONJ` | 103 | 0.14% |
| `U:CONJ` | 90 | 0.12% |
| `R:ADV:FORM` | 40 | 0.06% |
| `R:NUM` | 19 | 0.03% |
