PYTHON ?= python3
PYTHONPATH := src

.PHONY: help setup test bench quick figures examples corpus clean

help:
	@echo "setup     install the package in editable mode"
	@echo "test      run the unittest suite"
	@echo "bench     regenerate benchmarks/results.md and docs/stp_curve.png"
	@echo "quick     fast benchmark run (smaller corpus)"
	@echo "figures   regenerate the figure only"
	@echo "examples  run every example script"
	@echo "corpus    write a document corpus with PDFs and ground truth to ./corpus"
	@echo "clean     remove caches and generated corpora"

setup:
	$(PYTHON) -m pip install -e .

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

bench:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) benchmarks/run_benchmarks.py

quick:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) benchmarks/run_benchmarks.py --quick

figures:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m tradedoc figures --outdir docs

corpus:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m tradedoc generate --outdir corpus --pdf

examples:
	@for f in examples/*.py; do \
		echo "--- $$f"; \
		PYTHONPATH=$(PYTHONPATH) $(PYTHON) $$f || exit 1; \
	done

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache corpus example_out
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
