PYTHON ?= $(shell if [ -x .venv/bin/python ]; then printf %s .venv/bin/python; elif command -v python3 >/dev/null 2>&1; then command -v python3; else command -v python; fi)
RUFF_TARGETS := src tests examples benchmarks

.PHONY: install-dev install-examples lint test pip-check perf-check build check-fast check release-check

install-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

install-examples:
	$(PYTHON) -m pip install -e ".[examples]"

lint:
	$(PYTHON) -m ruff check $(RUFF_TARGETS)

test:
	$(PYTHON) -m pytest tests/ -v

pip-check:
	$(PYTHON) -m pip check

perf-check:
	$(PYTHON) benchmarks/profile_gate_smoke.py
	$(PYTHON) benchmarks/cache_profile_smoke.py
	$(PYTHON) benchmarks/score_fusion_profile_smoke.py
	$(PYTHON) benchmarks/inside_sampling_profile_smoke.py
	$(PYTHON) benchmarks/cache_worker_sweep_smoke.py
	$(PYTHON) benchmarks/registry_baseline_smoke.py
	$(PYTHON) benchmarks/concept_registry_smoke.py
	$(PYTHON) benchmarks/triple_extraction_smoke.py
	$(PYTHON) benchmarks/performance_baseline_smoke.py
	$(PYTHON) benchmarks/product_promotion_contract_smoke.py
	$(PYTHON) benchmarks/frontier_status_smoke.py
	$(PYTHON) benchmarks/frontier_release_evidence_smoke.py
	$(PYTHON) benchmarks/frontier_artifact_reference_smoke.py
	$(PYTHON) benchmarks/frontier_queue_execution_smoke.py
	$(PYTHON) benchmarks/product_trace_replay_smoke.py
	$(PYTHON) benchmarks/release_candidate_registry_smoke.py

build:
	$(PYTHON) -m build

check-fast: lint test pip-check

check: check-fast perf-check

release-check: check build
