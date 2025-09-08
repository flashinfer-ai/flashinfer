#!/bin/bash

set -eo pipefail
set -x

EXIT_CODE=0

test_scripts=(
  test_fp4_quantize.py
)

for test_file in "${test_scripts[@]}"; do
  xml_name="${test_file%.py}.xml"
  pytest -s "tests/${test_file}" --junit-xml="${CI_WORKSPACE}/$xml_name" || EXIT_CODE=1
done

exit $EXIT_CODE
