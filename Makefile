
all: install test

english:
	python -c 'import vectoria; vectoria.CharacterTrigramEmbedding("en")'
	python -c 'import vectoria; vectoria.WordEmbedding("en")'

test:
	pytest --doctest-modules vectoria
clean:
	rm -rf build
.PHONY: clean

buildext:
	python setup.py build_ext --inplace
.PHONY: buildext

install:
	pip install -r requirements.txt
	python setup.py install
.PHONY: install


upload: 
	python setup.py sdist upload

upload-to-pypitest: README.rst
	python setup.py sdist upload -r pypitest
.PHONY: upload-to-pypitest

install-from-pypitest::
	pip install -U --no-cache-dir -i https://testpypi.python.org/pypi vectoria
.PHONY: install-from-pypitest

install-dev:
	python setup.py develop
.PHONY: install-dev


