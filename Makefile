VENV := ./venv

.ONESHELL:
init:
	# Create virtual envirnoment
	if [ ! -d $(VENV) ]; then
		python -m venv $(VENV)
	fi

	source $(VENV)/bin/activate

	# Install dependencies
	pip install -U pip
	pip install -r ./requirements.txt

.PHONY: init
