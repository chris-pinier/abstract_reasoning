
# Manual Installation
## Python version management
conda / pyenv (mac, linux) / pyenv-win (windows)

## pipx
...

## poetry
...

## Python version
python = ">=3.8, <3.9"

## Psychopy (library that provides GUI for experiments)
psychopy = "2024.1.4"

## Pylink (library that provides support for communication with Eye Tracker (EyeLink))
In order to run the experiment with Psychopy and Eye Tracking, you will need to install the pylink library from SR Research:
https://psychopy.org/api/hardware/pylink.html

1. open the project's directory in the command line, 
2. activate the poetry environment: `poetry shell`
3. install pylink with pip:  `pip install --index-url=https://pypi.sr-support.com sr-research-pylink`

# Run the experiment
1. open the project's directory ("experiment-Lab") in the command line
2. run the following line: `poetry run python experiment.py`

