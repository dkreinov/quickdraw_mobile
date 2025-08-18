# Simple helpers (augment as the project grows)
.PHONY: setup sanity

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

sanity:
	python -c "import torch, transformers, datasets; print('OK')"
