.PHONY: setup test run

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

test:
	pytest -q

run:
	python -m alphafactory.run --config configs/base.yaml
