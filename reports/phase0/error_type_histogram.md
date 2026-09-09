# Falko-MERLIN error-type distribution

Computed over the **train + dev** splits only (55,443 edits). The test split is held out and was not read.

Phase 2 weights synthetic error injection to match this distribution, so that the training data resembles the mistakes real learners make rather than the mistakes that are easiest to generate.

| Error type | Count | Share |
|---|---:|---:|
| `R:SPELL` | 7,392 | 13.33% |
| `R:DET:FORM` | 5,963 | 10.76% |
| `M:PUNCT` | 5,295 | 9.55% |
| `R:OTHER` | 4,424 | 7.98% |
| `R:ORTH` | 4,271 | 7.70% |
| `R:ADJ:FORM` | 2,532 | 4.57% |
| `R:NOUN:FORM` | 2,190 | 3.95% |
| `U:PUNCT` | 1,976 | 3.56% |
| `R:ADP` | 1,894 | 3.42% |
| `M:DET` | 1,741 | 3.14% |
| `R:MORPH` | 1,544 | 2.78% |
| `R:WO` | 1,416 | 2.55% |
| `M:PRON` | 1,334 | 2.41% |
| `R:VERB:FORM` | 1,080 | 1.95% |
| `R:PUNCT` | 1,012 | 1.83% |
| `M:AUX` | 858 | 1.55% |
| `R:AUX:FORM` | 764 | 1.38% |
| `M:ADP` | 677 | 1.22% |
| `U:AUX` | 664 | 1.20% |
| `U:PRON` | 621 | 1.12% |
| `R:NOUN` | 601 | 1.08% |
| `R:PRON:FORM` | 582 | 1.05% |
| `M:OTHER` | 547 | 0.99% |
| `R:VERB` | 525 | 0.95% |
| `U:VERB` | 470 | 0.85% |
| `U:ADP` | 463 | 0.84% |
| `M:VERB` | 456 | 0.82% |
| `U:OTHER` | 451 | 0.81% |
| `R:PRON` | 425 | 0.77% |
| `M:PART` | 344 | 0.62% |
| `M:ADV` | 303 | 0.55% |
| `U:DET` | 301 | 0.54% |
| `U:ADV` | 250 | 0.45% |
| `R:AUX` | 243 | 0.44% |
| `U:PART` | 242 | 0.44% |
| `R:DET` | 234 | 0.42% |
| `R:ADV` | 203 | 0.37% |
| `M:SCONJ` | 186 | 0.34% |
| `R:ADJ` | 161 | 0.29% |
| `M:ADJ` | 131 | 0.24% |
| `U:ADJ` | 120 | 0.22% |
| `U:SCONJ` | 83 | 0.15% |
| `U:NOUN` | 76 | 0.14% |
| `R:SCONJ` | 74 | 0.13% |
| `M:CONJ` | 67 | 0.12% |
| `M:NOUN` | 59 | 0.11% |
| `U:PNOUN` | 48 | 0.09% |
| `R:CONJ` | 37 | 0.07% |
| `R:PNOUN` | 36 | 0.06% |
| `U:CONJ` | 32 | 0.06% |
| `R:ADV:FORM` | 19 | 0.03% |
| `R:*` | 11 | 0.02% |
| `M:PNOUN` | 11 | 0.02% |
| `R:PART` | 3 | 0.01% |
| `M:*` | 1 | 0.00% |
