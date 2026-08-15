PYTHON ?= python3
PYTHONPATH := src

.PHONY: help setup test bench quick figures examples clean

help:
	@echo "setup     install the package in editable mode"
	@echo "test      run the unittest suite"
	@echo "bench     regenerate benchmarks/results.md and docs/*.png"
	@echo "quick     fast benchmark run (smaller panel, 2 windows)"
	@echo "figures   regenerate the figures only"
	@echo "examples  run every example script"
	@echo "clean     remove caches and build artefacts"

setup:
	$(PYTHON) -m pip install -e .

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

bench:
	$(PYTHON) benchmarks/run_benchmarks.py

quick:
	$(PYTHON) benchmarks/run_benchmarks.py --quick

figures:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m dflab figures --outdir docs

examples:
	@for f in examples/*.py; do \
		echo "--- $$f"; \
		$(PYTHON) $$f || exit 1; \
	done

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
