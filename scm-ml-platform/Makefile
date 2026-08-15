PY ?= python
export PYTHONPATH := src

.PHONY: setup test bench figures example cli clean

setup:
	$(PY) -m pip install -e .

test:
	$(PY) -m unittest discover -s tests -v

bench:
	$(PY) benchmarks/run_benchmarks.py

figures:
	$(PY) benchmarks/run_benchmarks.py --quick

example:
	$(PY) examples/01_release_pipeline.py

cli:
	$(PY) -m scmplatform validate
	$(PY) -m scmplatform audit
	$(PY) -m scmplatform backtest

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
