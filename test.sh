#!/bin/bash
# Abort on Error
set -e
python -m pytest -s -q tests/test_$1.py
