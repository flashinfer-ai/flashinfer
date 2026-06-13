#!/bin/bash
# Normalize the Python command used by CI scripts.

# Some CI images can expose /opt/conda/envs/py312/bin/python first while pip
# resolves to base conda. Prefer the Python that has the CI packages installed.
if ! python -c "import torch" >/dev/null 2>&1 || ! python -m pip --version >/dev/null 2>&1; then
  if [ -x /opt/conda/bin/python ] \
    && /opt/conda/bin/python -c "import torch" >/dev/null 2>&1 \
    && /opt/conda/bin/python -m pip --version >/dev/null 2>&1; then
    export PATH="/opt/conda/bin:${PATH}"
    hash -r
    echo "Using /opt/conda/bin/python for CI package environment"
  fi
fi
