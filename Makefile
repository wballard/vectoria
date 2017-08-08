
all: install test

test: install-dev
	python -c "import vecoder; v = vecoder.loadFastTextModel('test/test.bin'); print(v['import'])"

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

# Install the pandoc(1) first to run this command
# sudo apt-get install pandoc
README.rst: README.md
	pandoc --from=markdown --to=rst --output=README.rst README.md

upload: README.rst
	python setup.py sdist upload

upload-to-pypitest: README.rst
	python setup.py sdist upload -r pypitest
.PHONY: upload-to-pypitest

install-from-pypitest::
	pip install -U --no-cache-dir -i https://testpypi.python.org/pypi vecoder
.PHONY: install-from-pypitest

install-dev: README.rst
	python setup.py develop
.PHONY: install-dev
