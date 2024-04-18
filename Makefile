.PHONY: docs
docs:
	sphinx-build -b html docs build-docs

.PHONY: install
install:
	python3 -m pip install -e .[dev]

.PHONY: build
build:
	python3 -m build --sdist

.PHONY: upload
upload:
	python3 -m twine upload dist/* --skip-existing

.PHONY: upload-test
upload-test: venv
	python3 -m twine upload --repository testpypi dist/* --skip-existing

.PHONY: test
test:
	python3 -m pytest
