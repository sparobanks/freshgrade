.PHONY: data eval test demo serve clean

# render and cache the synthetic datasets
data:
	python scripts/make_data.py

# train the grader and write all metrics and charts
eval:
	python scripts/run_eval.py

# unit tests
test:
	pytest -q

# annotated demo panel for the README
demo:
	python scripts/make_demo.py

# operator dashboard at http://127.0.0.1:8000  (sign in: packer / freshgrade)
serve:
	uvicorn fg.api:app --host 127.0.0.1 --port 8000

clean:
	rm -rf data/*.npz models/*.pt reports/*.png reports/*.json __pycache__ */__pycache__
