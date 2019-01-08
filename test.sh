#!/bin/bash
# Abort on Error
set -e

python tests/gen_test.py -o tests/
if [ $# -eq 3 ]; then
    test_file=tests/test_$1_$2_$3.py
    if [ ! -f $test_file ]; then
        echo "File $test_file does not exist."
        exit 1
    fi
elif [ $# -eq 2 ]; then
    test_file=tests/test_$1_*_$2.py
else
	echo "Wrong test arguments."
	exit 1
fi
python -m pytest -s -q $test_file
