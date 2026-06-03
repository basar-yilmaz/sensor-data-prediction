SHELL := /bin/bash

.DEFAULT_GOAL := help

ARGS ?=
ENV_FILE ?= .env

define run_with_env
	@if [ -f "$(ENV_FILE)" ]; then \
		set -a; \
		source "$(ENV_FILE)"; \
		set +a; \
	fi; \
	$(1)
endef

.PHONY: help install env services train train_baseline fetch download repro prepare splits pull_model pull_baseline clean_data clean_dvc_cache

help:
	@printf "Available targets:\n"
	@printf "  make install                 Install dependencies with uv\n"
	@printf "  make env                     Create .env from .env.example if missing\n"
	@printf "  make services                Start local MLflow service\n"
	@printf "  make train ARGS='...'        Train neural model; loads .env when present\n"
	@printf "  make train_baseline ARGS='...' Train XGBoost baseline; loads .env when present\n"
	@printf "  make fetch                   Ensure raw/prepared data is available\n"
	@printf "  make download                Pull data through DVC remote\n"
	@printf "  make repro                   Reproduce DVC pipeline\n"
	@printf "  make prepare                 Run prepare stage directly\n"
	@printf "  make splits                  Run splits stage directly\n"
	@printf "  make pull_model              Pull neural model artifact\n"
	@printf "  make pull_baseline           Pull baseline model artifact\n"
	@printf "  make clean_data              Remove local generated data/artifacts and prune DVC cache\n"
	@printf "  make clean_dvc_cache         Remove the entire local DVC cache\n"

install:
	uv sync

env:
	@if [ ! -f "$(ENV_FILE)" ]; then \
		cp .env.example "$(ENV_FILE)"; \
	else \
		printf "%s already exists\n" "$(ENV_FILE)"; \
	fi

services:
	docker compose up -d mlflow

train:
	$(call run_with_env,uv run bfrb train $(ARGS))

train_baseline:
	$(call run_with_env,uv run bfrb train_baseline $(ARGS))

fetch:
	$(call run_with_env,uv run bfrb fetch $(ARGS))

download:
	uv run bfrb download $(ARGS)

repro:
	uv run dvc repro $(ARGS)

prepare:
	$(call run_with_env,uv run bfrb prepare $(ARGS))

splits:
	$(call run_with_env,uv run bfrb splits $(ARGS))

pull_model:
	uv run dvc pull models/temporal_conv_gru_tof.ckpt.dvc -r bfrb-models $(ARGS)

pull_baseline:
	uv run dvc pull models/xgboost_baseline.joblib.dvc -r bfrb-models $(ARGS)

clean_data:
	rm -f data/raw/train.csv
	rm -rf data/prepared/* artifacts plots outputs multirun mlruns mlruns-server mlartifacts
	rm -f models/*.joblib models/*.ckpt
	uv run dvc gc -w -f

clean_dvc_cache:
	rm -rf .dvc/cache .dvc/tmp
