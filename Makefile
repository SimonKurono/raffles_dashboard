PYTHON ?= ./venv/bin/python

.PHONY: test
test:
	$(PYTHON) -m unittest discover -s tests -v
