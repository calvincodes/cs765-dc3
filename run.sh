#!/bin/sh

# Install venv
python3 -m pip install --user virtualenv

# Creating and Activating venv
python3 -m venv env
source env/bin/activate

# Installing requirements
pip install requirements.txt

# Running application
python3 scripts/app.py
