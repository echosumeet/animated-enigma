PYTHON ?= python3
PYTHONPATH := src

.PHONY: help setup test bench quick figures examples clean

help:
	@echo "setup     install the package in editable mode"
	@echo "test      run the unittest suite"
	@echo "bench     regenerate benchmarks/results.md and docs/calibration.png"
	@echo "quick     fast benchmark run (smaller panel, fewer trees)"
	@echo "figures   regenerate the figure only"
	@echo "examples  run every example script"
	@echo "clean     remove caches and build artefacts"

setup:
	$(PYTHON) -m pip install -e .

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

bench:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) benchmarks/run_benchmarks.py

quick:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) benchmarks/run_benchmarks.py --quick

figures:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m etarisk figures --outdir docs

examples:
	@for f in examples/*.py; do \
		echo "--- $$f"; \
		PYTHONPATH=$(PYTHONPATH) $(PYTHON) $$f || exit 1; \
	done

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
