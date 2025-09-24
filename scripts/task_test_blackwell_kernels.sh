#!/bin/bash

set -eo pipefail
set -x
: ${JUNIT_DIR:=$(realpath ./junit)}


EXIT_CODE=0

pytest tests/ || EXIT_CODE=1

exit $EXIT_CODE
