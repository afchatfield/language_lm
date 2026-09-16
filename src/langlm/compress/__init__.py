"""Phase 4: making the model smaller without making it worse.

Two independent experiments. :mod:`langlm.compress.vocab` removes vocabulary
rows the target languages never touch, which on EuroLLM is 21.4% of the
parameters because its embeddings are untied and every removed row pays twice.
The depth prune is the second and lives beside it.
"""
