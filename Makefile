
zero-shot:
	python scripts/run_zero_shot.py --language $(LANGUAGE) --model $(MODEL)
test:
	pytest

lint:
	ruff check .

list-languages:
	python scripts/list_fleurs_languages.py

inspect:
	python scripts/inspect_dataset.py --language $(LANGUAGE)
