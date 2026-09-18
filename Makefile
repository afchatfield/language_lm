# Phase 0 workflow. See README.md for the full story.
#
#   make setup       one-off: create the conda env
#   make test        lint + offline test suite
#   make phase0      everything below, in order
#
.PHONY: help setup test lint format \
        ngrams lt-up lt-down lt-logs lt-check \
        corpora data dictionary fertility phase0 \
        overcorrection errant-check baselines phase1 \
        clean-corpus lt-survey weights training-set phase2 thesaurus \
        check-corruptors check-citations corrupter backtranslation \
        setup-cuda train-data dry-run train evaluate phase3 \
        preference-pairs train-dpo evaluate-test trim-vocab baselines-test \
        clean-ngrams clean-data \
        ngrams-es corpora-es data-es dictionary-es overcorrection-es \
        baselines-es clean-corpus-es training-set-es train-data-es \
        dry-run-es train-es evaluate-es phase1-es phase5 \
        baselines-es-test trim-vocab-es compress-es

PYTHON ?= python
COMPOSE = docker compose -f docker/docker-compose.yml

help:  ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- environment ------------------------------------------------------------

setup:  ## Create the `langlm` conda environment
	conda env create -f environment.yml

lint:  ## Run ruff
	ruff check .
	ruff format --check .

format:  ## Auto-fix formatting and lint issues
	ruff check --fix .
	ruff format .

test:  ## Lint plus the offline test suite
	ruff check .
	ruff format --check .
	pytest

# --- LanguageTool -----------------------------------------------------------

ngrams:  ## Download + extract the German n-gram data (~1.7GB download, ~6GB on disk)
	$(PYTHON) scripts/download_ngrams.py --lang de

lt-up:  ## Start the LanguageTool server (requires Docker Desktop to be running)
	$(COMPOSE) up -d
	@echo "LanguageTool starting on http://localhost:8010 - first boot with n-grams takes a minute."

lt-down:  ## Stop the LanguageTool server
	$(COMPOSE) down

lt-logs:  ## Tail the LanguageTool server logs
	$(COMPOSE) logs -f

lt-check:  ## Verify LanguageTool is up AND that the n-grams actually loaded
	$(PYTHON) scripts/lt_sanity_check.py

# --- data -------------------------------------------------------------------

corpora:  ## Download the Leipzig corpora used for tokenizer fertility
	$(PYTHON) scripts/download_leipzig.py

data:  ## Download Falko-MERLIN and freeze the train/dev/test splits
	$(PYTHON) scripts/download_falko_merlin.py
	$(PYTHON) scripts/freeze_splits.py

dictionary:  ## Fetch the German Hunspell dictionary used for error typing (~4MB)
	$(PYTHON) -c "from langlm.eval.errant_de.spelling import ensure_dictionary; ensure_dictionary()"

thesaurus:  ## Fetch OpenThesaurus, used only by the lenient scorer (~2MB)
	$(PYTHON) scripts/download_thesaurus.py

fertility:  ## Run the tokenizer fertility + vocab-trim analysis
	$(PYTHON) scripts/run_fertility.py

phase0: data corpora fertility  ## Run the whole of Phase 0 (after `make ngrams lt-up lt-check`)
	@echo "Phase 0 artefacts are in reports/phase0/"

# --- Phase 1: eval harness and baselines ------------------------------------

overcorrection:  ## Build and freeze the set of already-correct German sentences
	$(PYTHON) scripts/build_overcorrection_set.py

errant-check:  ## Measure the German ERRANT annotator against the corpus annotation
	$(PYTHON) scripts/check_errant_de.py

baselines:  ## Run every Phase 1 baseline and write the results table
	$(PYTHON) scripts/run_baselines.py

baselines-test:  ## The same baselines on the held-out split, so a test score has a test bar
	$(PYTHON) scripts/run_baselines.py --split test

phase1: dictionary overcorrection errant-check baselines  ## Run the whole of Phase 1
	@echo "Phase 1 artefacts are in reports/phase1/"

# --- Phase 2: data pipeline -------------------------------------------------

clean-corpus:  ## Filter and freeze the clean German text the corruptors damage
	$(PYTHON) scripts/build_clean_corpus.py

lt-survey:  ## Ask LanguageTool what it calls our injected errors (needs lt-up)
	$(PYTHON) scripts/survey_lt_rules.py

check-corruptors:  ## Verify each corruptor produces the error type it claims
	$(PYTHON) scripts/check_corruptors.py

check-citations:  ## Verify every cited Regelwerk paragraph against the official PDF
	$(PYTHON) scripts/check_citations.py

weights:  ## Re-derive the injected error distribution from Falko + the baselines
	$(PYTHON) scripts/derive_injection_weights.py

corrupter:  ## Train the error generator on Falko-MERLIN read backwards (needs a GPU)
	$(PYTHON) scripts/train_corrupter.py

backtranslation:  ## Run the error generator over clean German and filter the output
	$(PYTHON) scripts/generate_backtranslation.py

training-set:  ## Corrupt, explain, and write the training set
	$(PYTHON) scripts/build_training_set.py

phase2: clean-corpus lt-survey check-corruptors check-citations weights training-set  ## Run the whole of Phase 2
	@echo "Phase 2 artefacts are in reports/phase2/"

# --- Phase 3: supervised fine-tuning ----------------------------------------

setup-cuda:  ## Create the `langlm` environment on a CUDA box (no MLX)
	conda env create -f environment-cuda.yml

train-data:  ## Everything the trainer and the evaluation need, from a fresh clone
	$(PYTHON) scripts/download_falko_merlin.py
	$(PYTHON) scripts/freeze_splits.py
	$(PYTHON) scripts/download_leipzig.py
	$(PYTHON) -c "from langlm.eval.errant_de.spelling import ensure_dictionary; ensure_dictionary()"
	$(PYTHON) scripts/build_overcorrection_set.py
	$(PYTHON) scripts/build_clean_corpus.py
	$(PYTHON) scripts/build_training_set.py

dry-run:  ## Check the loss mask, the schema and one forward pass. Trains nothing.
	$(PYTHON) scripts/train_sft.py --limit 64 --dry-run

train:  ## LoRA fine-tune on the Phase 2 training set
	$(PYTHON) scripts/train_sft.py

evaluate:  ## Score the fine-tuned adapter against the Phase 1 baselines
	$(PYTHON) scripts/eval_finetuned.py

phase3: train-data dry-run train evaluate  ## Everything Phase 3 needs, in order
	@echo "Adapter in checkpoints/phase3-de/, results in reports/phase3/"

# --- Phase 3, iteration 6: preference training ------------------------------
#
# ADAPTER and NAME pick which SFT checkpoint is being improved, since these run
# once per candidate model rather than once for the project.

ADAPTER ?= checkpoints/phase3-de-iter4-cw025
NAME ?= iter4-cw025

preference-pairs:  ## Beam over Falko train, paired best-against-first (~1h on an H200)
	$(PYTHON) scripts/build_preference_pairs.py --adapter $(ADAPTER) --name $(NAME)

train-dpo:  ## Preference-train $(ADAPTER) on its own beam
	$(PYTHON) scripts/train_dpo.py --pairs $(NAME) --adapter $(ADAPTER) \
		--adapter-dir $(ADAPTER)-dpo

evaluate-test:  ## Unseal the held-out split. Once per model, at the very end.
	$(PYTHON) scripts/eval_finetuned.py --adapter $(ADAPTER) --name $(NAME)-test --split test

# --- Phase 4: compression -----------------------------------------------------

trim-vocab:  ## Trim the base model's vocabulary to the rows German and English use
	$(PYTHON) scripts/trim_vocab.py --threshold 0.999

# The adapter `evaluate-es` scores, and the row label it writes under.
# Overridable so a second recipe does not overwrite the first one's report:
#   make evaluate-es ADAPTER_ES=checkpoints/phase3-es-cw025 NAME_ES=fine-tuned-es-cw025
ADAPTER_ES ?= checkpoints/phase3-es
NAME_ES ?= fine-tuned-es

# --- Phase 5: Spanish port ---------------------------------------------------
#
# Mirrors Phase 0/1/2/3's targets above, `-es` suffixed. `eval_finetuned.py` is
# language-parameterised now, so `evaluate-es` and a full `phase5` chain exist
# alongside the rest.

ngrams-es:  ## Download + extract the Spanish n-gram data (~1.7GB download, ~6GB on disk)
	$(PYTHON) scripts/download_ngrams.py --lang es

corpora-es:  ## Download the Leipzig corpora the Spanish clean corpus is built from
	$(PYTHON) scripts/download_leipzig.py --corpus spa_news_2023_100K spa_wikipedia_2021_100K

data-es:  ## Download COWS-L2H, type it with errant_es, and freeze the splits
	$(PYTHON) scripts/download_cowsl2h.py
	$(PYTHON) scripts/build_cowsl2h_m2.py
	$(PYTHON) scripts/freeze_splits_es.py

dictionary-es:  ## Fetch the Spanish Hunspell dictionary used for error typing
	$(PYTHON) -c "from langlm.eval.errant_es.spelling import ensure_dictionary; ensure_dictionary()"

overcorrection-es:  ## Build and freeze the set of already-correct Spanish sentences
	$(PYTHON) scripts/build_overcorrection_set_es.py

baselines-es:  ## Run every Spanish baseline and write the results table (needs lt-up with ngrams-es)
	$(PYTHON) scripts/run_baselines.py --language es

phase1-es: dictionary-es overcorrection-es baselines-es  ## The Spanish counterpart of `phase1`
	@echo "Phase 5 baseline artefacts are in reports/phase5/"

clean-corpus-es:  ## Filter and freeze the clean Spanish text the corruptors damage
	$(PYTHON) scripts/build_clean_corpus_es.py

training-set-es:  ## Corrupt, explain, and write the Spanish training set
	$(PYTHON) scripts/build_training_set_es.py

train-data-es:  ## Everything the trainer needs for Spanish, from a fresh clone
	$(PYTHON) scripts/download_cowsl2h.py
	$(PYTHON) scripts/build_cowsl2h_m2.py
	$(PYTHON) scripts/freeze_splits_es.py
	$(PYTHON) scripts/download_leipzig.py --corpus spa_news_2023_100K spa_wikipedia_2021_100K
	$(PYTHON) -c "from langlm.eval.errant_es.spelling import ensure_dictionary; ensure_dictionary()"
	$(PYTHON) scripts/build_overcorrection_set_es.py
	$(PYTHON) scripts/build_clean_corpus_es.py
	$(PYTHON) scripts/build_training_set_es.py

dry-run-es:  ## Check the Spanish loss mask, the schema and one forward pass. Trains nothing.
	$(PYTHON) scripts/train_sft.py --language es --limit 64 --dry-run

train-es:  ## LoRA fine-tune on the Spanish training set
	$(PYTHON) scripts/train_sft.py --language es

evaluate-es:  ## Score the Spanish adapter against the Phase 5 baselines
	$(PYTHON) scripts/eval_finetuned.py --language es --adapter $(ADAPTER_ES) --name $(NAME_ES)

# The adapter that actually ships, which is not `ADAPTER_ES`: that one is the
# knob for scoring whichever recipe is being tried, and defaults to the first
# run. `cw025` is the one Phase 5 chose, the Spanish counterpart of German's
# `iter4-cw025`, so compression measures that one unless told otherwise.
SHIP_ADAPTER_ES ?= checkpoints/phase3-es-cw025

baselines-es-test:  ## Spanish baselines on the held-out split (needs lt-up with ngrams-es)
	$(PYTHON) scripts/run_baselines.py --language es --split test

trim-vocab-es:  ## Trim to the rows Spanish and English use, task corpus included
	$(PYTHON) scripts/trim_vocab.py --threshold 0.999 --language es --languages es en \
		--out checkpoints/base-trimmed-es

# `--language es` counts the Spanish task corpus, which is what keeps the JSON
# schema tokens; `--languages es en` picks the prose corpora. Phase 4's gate
# failed by counting only the second, so they stay separate flags.
compress-es: trim-vocab-es  ## Trim, merge the adapter, and score the trimmed Spanish model
	$(PYTHON) scripts/merge_adapter.py --base checkpoints/base-trimmed-es \
		--adapter $(SHIP_ADAPTER_ES) --out checkpoints/ship-es
	$(PYTHON) scripts/eval_finetuned.py --language es --base checkpoints/base-trimmed-es \
		--adapter $(SHIP_ADAPTER_ES) --name trimmed-nohealing-es

phase5: train-data-es dry-run-es train-es evaluate-es  ## Everything Phase 5 needs, in order
	@echo "Adapter in checkpoints/phase3-es/, results in reports/phase5/"

# --- cleanup ----------------------------------------------------------------

clean-ngrams:  ## Delete the extracted n-gram data (frees ~6GB)
	rm -rf data/raw/ngrams

clean-data:  ## Delete every downloaded and derived artefact (keeps the split manifest)
	rm -rf data/raw/* data/interim/* data/processed/*
