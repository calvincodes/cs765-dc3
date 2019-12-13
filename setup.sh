#!/bin/bash
# Usage: source run.sh
# Note: sh run.sh would not create a venv as after running the script, you will return to base shell.

# Install venv
python3 -m pip install --user virtualenv

# Creating and Activating venv
python3 -m venv env
source env/bin/activate

# Installing requirements
pip install -r requirements.txt
