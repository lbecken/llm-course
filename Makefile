# Modern LLM Engineering — common tasks.  Run `make help`.
PY ?= python
NANOCHAT_DIR ?= external/nanochat
RUN_DIR ?= $(CURDIR)/results/runs/nanochat-local

.PHONY: help install test test-all test-nanochat nanochat env-cpu env-gpu env-spark doctor check-docs \
        nanochat-local-data nanochat-local-train nanochat-local-eval spark-probe bench clean

help:
	@echo "Setup"
	@echo "  make install          install the course package (CPU labs and tests only need torch + numpy)"
	@echo "  make nanochat         clone karpathy/nanochat at the pinned commit into external/nanochat"
	@echo "  make env-cpu          nanochat's venv (CPU PyTorch) + course package         [tested]"
	@echo "  make env-gpu          nanochat's venv (CUDA 12.8 wheels, x86_64) + course package"
	@echo "  make env-spark        DGX Spark route, see docs/dgx-spark-setup.md           [awaiting validation]"
	@echo "  make doctor           environment diagnostics (GPU, precision, torch.compile, nanochat)"
	@echo "Checks"
	@echo "  make test             fast CPU test-suite (what CI runs)"
	@echo "  make test-all         including slow tests"
	@echo "  make test-nanochat    parity tests against the pinned nanochat checkout"
	@echo "  make check-docs       verify every link, file path and source reference in the Markdown"
	@echo "nanochat on a miniature local dataset (CPU-friendly, Tier A)"
	@echo "  make nanochat-local-data | nanochat-local-train | nanochat-local-eval"
	@echo "Measurement"
	@echo "  make spark-probe      20-step throughput/memory probe before committing to a long run"
	@echo "  make bench            course benchmark suite (KV cache, attention, precision) -> results/runs/"

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -m "not slow and not nanochat and not gpu"

test-all:
	$(PY) -m pytest -m "not gpu"

test-nanochat:
	NANOCHAT_DIR=$(NANOCHAT_DIR) $(PY) -m pytest tests/test_nanochat_parity.py -m nanochat

nanochat:
	NANOCHAT_DIR=$(NANOCHAT_DIR) scripts/setup/fetch_nanochat.sh

env-cpu:
	NANOCHAT_DIR=$(NANOCHAT_DIR) scripts/setup/setup_env.sh cpu

env-gpu:
	NANOCHAT_DIR=$(NANOCHAT_DIR) scripts/setup/setup_env.sh gpu

env-spark:
	NANOCHAT_DIR=$(NANOCHAT_DIR) scripts/setup/setup_env.sh spark

doctor:
	NANOCHAT_DIR=$(NANOCHAT_DIR) $(PY) scripts/diagnostics/check_env.py

check-docs:
	NANOCHAT_DIR=$(NANOCHAT_DIR) $(PY) scripts/diagnostics/check_links.py

# --- nanochat on a miniature local dataset -----------------------------------
nanochat-local-data:
	NANOCHAT_BASE_DIR=$(RUN_DIR) $(PY) scripts/experiments/make_local_dataset.py
	cd $(NANOCHAT_DIR) && NANOCHAT_BASE_DIR=$(RUN_DIR) $(PY) -m scripts.tok_train --vocab-size=512 --max-chars=2000000

nanochat-local-train:
	NANOCHAT_BASE_DIR=$(RUN_DIR) scripts/experiments/nanochat_local.sh train

nanochat-local-eval:
	NANOCHAT_BASE_DIR=$(RUN_DIR) scripts/experiments/nanochat_local.sh eval

spark-probe:
	scripts/experiments/spark_probe.sh

bench:
	$(PY) scripts/benchmarks/run_all.py

clean:
	rm -rf .pytest_cache build *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
