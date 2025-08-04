#!/bin/bash
set -eo pipefail
set -x
echo "Building FlashInfer documentation..."

make clean
make html

# Add RunLLM widget to generated HTML files
echo "Adding RunLLM widget to documentation..."
python3 wrap_run_llm.py

echo "Documentation build complete!"
