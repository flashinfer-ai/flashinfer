#!/bin/bash
set -eo pipefail
set -x
echo "Building FlashInfer documentation..."

# Install flashinfer package first
echo "Installing FlashInfer package..."
pip install -e ..
# PrimTS autodoc imports require the source-test CUTLASS DSL version rather
# than the published package default.
# shellcheck disable=SC1091
source ../scripts/setup_ci_test_env.sh

make clean
make SPHINXOPTS='-T -v' html

# Add RunLLM widget to generated HTML files
echo "Adding RunLLM widget to documentation..."
python3 wrap_run_llm.py

echo "Documentation build complete!"
