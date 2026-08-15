PY ?= python
export PYTHONPATH := src

.PHONY: help setup test bench figures examples lint clean all

help:
	@echo "setup     install the package in editable mode with figure extras"
	@echo "test      run the unittest suite"
	@echo "bench     regenerate benchmarks/results.md"
	@echo "figures   regenerate the figures in docs/"
	@echo "examples  run every example script"
	@echo "all       test + bench + figures + examples"

setup:
	$(PY) -m pip install -e ".[figures]"

test:
	$(PY) -m unittest discover -s tests -v

bench:
	$(PY) benchmarks/run_benchmarks.py

figures:
	$(PY) benchmarks/make_figures.py

examples:
	@for f in examples/*.py; do echo "--- $$f"; $(PY) $$f > /dev/null || exit 1; done
	@echo "all examples ran clean"

lint:
	$(PY) -m compileall -q src tests examples benchmarks

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf build dist *.egg-info src/*.egg-info

all: test bench figures examples
