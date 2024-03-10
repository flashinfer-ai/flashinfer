#!/bin/bash
echo "Formatting CUDA files"
find include/ -regex '.*\.\(h\|cuh\|cu\|cc\)' | xargs clang-format-19 -i
find src/ -regex '.*\.\(h\|cuh\|cu\|cc\)' | xargs clang-format-19 -i
find python/ -regex '.*\.\(h\|cuh\|cu\|cc\)' | xargs clang-format-19 -i
echo "Formatting Python files"
find python/ -regex '.*\.\(py\)' | xargs black

