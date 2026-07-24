#!/bin/bash

# Check CUDA version
CUDA_VERSION=$(nvcc --version | grep -oP '(?<=: )[^ ]+')

# Check if SM121 is supported
if [ "$CUDA_VERSION" == "12.9" ]; then
  echo "DGX Spark (SM121) support is available for CUDA version $CUDA_VERSION"
else
  echo "DGX Spark (SM121) support is not available for CUDA version $CUDA_VERSION"
fi