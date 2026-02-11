# Adult Income Classification

Minimal scaffold for the adult-income-classification project.

Files added:
- `app.py` — simple entrypoint that loads model and predicts on `data/sample.csv` if present.
- `models.py` — placeholder load/save/predict helpers.
- `utils.py` — minimal data loading and preprocessing.
- `requirements.txt` — common dependencies.
- `data/` — place dataset files here (e.g. `sample.csv`).

Quick start

1. Create a virtual environment and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Put a CSV sample at `data/sample.csv` and run:

```bash
python app.py
```
