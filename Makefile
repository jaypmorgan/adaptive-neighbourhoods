.PHONY: docs
docs:
	sphinx-build -b html docs build-docs

.PHONY: build
build:
	python3 -m build

.PHONY: pypi
pypi:
	python3 -m twine upload dist/*
