PY := .venv/bin/python3.14

.PHONY: setup part1 large synthetic part3 readme viewer all

all: part1 large part3 readme

setup:                ## venv + pinned requirements
	python3.14 -m venv .venv
	$(PY) -m pip install -r requirements.txt

part1:                ## flat vs beam, hybrid search (vehicle dataset)
	$(PY) -m search hybrid
	$(PY) results/compare_part1.py

large:                ## flat vs beam and hybrid speed on a 10^4 sample of the large dataset (needs the zip; ~15 min to embed)
	$(PY) experiment-big-data/sample.py --n 10000
	$(PY) experiment-big-data/run.py --n 10000
	$(PY) results/compare_part1.py --sample 10000

synthetic:            ## regenerate the synthetic queries (optional: they are committed; needs ANTHROPIC_API_KEY, ~$1.5)
	$(PY) synthetic/generate_synthetic.py --full

part3:                ## embed the synthetic queries, train the towers, evaluate
	$(PY) training/prepare.py --synthetic outputs/synthetic --out training/bundles/full.npz
	OUTPUT_DIR=training/runs/final $(PY) training/train_towers.py --bundle training/bundles/full.npz --epochs 20
	$(PY) results/compare_part3.py

readme:               ## rebuild the tables in results/README.md from the CSVs
	$(PY) results/make_readme.py

viewer:               ## browse, search and inspect everything at http://localhost:8501
	$(PY) -m viewer
