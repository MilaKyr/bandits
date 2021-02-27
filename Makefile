.PHONY: run test test-cov
PYTHONPATH=.	pytest

run:
	@python play.py

test:
	@python	-m	pytest

test-cov:
	@pytest --cov-report term-missing --cov=src tests/