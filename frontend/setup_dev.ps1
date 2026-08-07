#!/bin/bash
pip install -r requirements.txt
pre-commit install --hook-type pre-commit
pre-commit install --hook-type pre-push