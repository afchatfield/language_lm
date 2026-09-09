# Phase 0 workflow. See README.md for the full story.
#
#   make setup       one-off: create the conda env
#   make test        lint + offline test suite
#   make phase0      everything below, in order
#
.PHONY: help setup test lint format \
        ngrams lt-up lt-down lt-logs lt-check \
        corpora data fertility phase0 \
        clean-ngrams clean-data

PYTHON ?= python
COMPOSE = docker compose -f docker/docker-compose.yml

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
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

fertility:  ## Run the tokenizer fertility + vocab-trim analysis
	$(PYTHON) scripts/run_fertility.py

phase0: data corpora fertility  ## Run the whole of Phase 0 (after `make ngrams lt-up lt-check`)
	@echo "Phase 0 artefacts are in reports/phase0/"

# --- cleanup ----------------------------------------------------------------

clean-ngrams:  ## Delete the extracted n-gram data (frees ~6GB)
	rm -rf data/raw/ngrams

clean-data:  ## Delete every downloaded and derived artefact (keeps the split manifest)
	rm -rf data/raw/* data/interim/* data/processed/*
