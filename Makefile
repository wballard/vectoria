

test: install-dev
	pytest --doctest-modules vectoria
.PHONY: test

english:
	python -c 'import vectoria; vectoria.CharacterTrigramEmbedding("en")'
	python -c 'import vectoria; vectoria.WordEmbedding("en")'

install:
	python setup.py install
.PHONY: install

upload: 
	python setup.py sdist upload
.PHONY: upload


install-dev:
	python setup.py develop
.PHONY: install-dev


