#!/bin/bash
set -eo pipefail
set -x
echo "Building FlashInfer documentation..."

if ! python3 -c "import flashinfer" &> /dev/null; then
  cd ..
  pip install -e . -v --no-deps --no-build-isolation
  cd docs
fi

make clean
make html

# Add RunLLM widget to generated HTML files
echo "Adding RunLLM widget to documentation..."
python3 wrap_run_llm.py

echo "Documentation build complete!"
