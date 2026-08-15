PY ?= python
export PYTHONPATH := src

.PHONY: help setup test bench figures example all clean

help:
	@echo "setup    install the package and its dependencies (editable)"
	@echo "test     run the unittest suite"
	@echo "bench    regenerate benchmarks/results.md"
	@echo "figures  regenerate docs/pick_path.png"
	@echo "example  run the end-to-end example"
	@echo "all      test + bench + figures + example"

setup:
	$(PY) -m pip install -e ".[figures]"

test:
	$(PY) -m unittest discover -s tests -v

bench:
	$(PY) benchmarks/run_benchmarks.py

figures:
	$(PY) benchmarks/make_figure.py

example:
	$(PY) examples/slot_and_route.py

all: test bench figures example

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist *.egg-info src/*.egg-info
