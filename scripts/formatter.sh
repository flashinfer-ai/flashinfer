#!/bin/bash
echo "Formatting CUDA files"
find include/ -regex '.*\.\(h\|cuh\|cu\|cc\)' | xargs clang-format -i
find src/ -regex '.*\.\(h\|cuh\|cu\|cc\)' -not -path './src/generated' | xargs clang-format -i
find csrc/ -regex '.*\.\(h\|cuh\|cu\|cc\)' -not -path './csrc/generated' | xargs clang-format -i
echo "Formatting Python files"
find flashinfer/ -regex '.*\.\(py\)' | xargs black
