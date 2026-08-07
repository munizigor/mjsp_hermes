#!/bin/bash
pip install -r requirements_linux.txt
pre-commit install --hook-type pre-commit
pre-commit install --hook-type pre-push