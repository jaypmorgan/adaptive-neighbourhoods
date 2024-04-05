.PHONY: docs
docs:
	sphinx-build -b html docs build-docs

.PHONY: install
install:
	python3 -m pip install -e .

.PHONY: build
build:
	python3 -m build --sdist

.PHONY: upload
upload:
	python3 -m twine upload dist/*

.PHONY: upload-test
upload-test:
	python3 -m twine upload --repository testpypi dist/*