#!/bin/bash
# Abort on Error
set -e

export PING_SLEEP=60s
export WORKDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export BUILD_OUTPUT=$WORKDIR/build.out

touch $BUILD_OUTPUT

dump_output() {
   echo Tailing the last 100 lines of output:
   tail -100 $BUILD_OUTPUT
}
error_handler() {
  echo ERROR: An error was encountered with the build.
  dump_output
  exit 1
}

# If an error occurs, run our error handler to output a tail of the build
trap 'error_handler' ERR

# Set up a repeating loop to send some output to Travis.

bash -c "while true; do echo \$(date) - building ...; sleep $PING_SLEEP; done" &
PING_LOOP_PID=$!

# My build is using maven, but you could build anything with this, E.g.
python tests/gen_test.py -o tests/
test_file=tests/test_$1_$2_$3.py
if [ -f $test_file]; then
    python -m pytest -s -q $test_file >> $BUILD_OUTPUT 2>&1
else
    echo "File $test_file does not exist."
fi

# The build finished without returning an error so dump a tail of the output
dump_output

# nicely terminate the ping output loop
kill $PING_LOOP_PID