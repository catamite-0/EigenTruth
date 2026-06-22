PYTHON ?= $(shell if [ -x .venv/bin/python ]; then printf %s .venv/bin/python; elif command -v python3 >/dev/null 2>&1; then command -v python3; else command -v python; fi)
RUFF_TARGETS := src tests examples benchmarks

.PHONY: install-dev install-examples lint test pip-check perf-check build check release-check

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

build:
	$(PYTHON) -m build

check: lint test pip-check perf-check

release-check: check build
