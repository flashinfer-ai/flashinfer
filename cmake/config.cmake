# Whether to compile fp8 kernels or not.
set(FLASHINFER_ENABLE_FP8 ON)
# Whether to compile tvm bindings or not.
set(FLASHINFER_TVM_BINDING ON)
# Whether to compile prefill kernel tests/benchmarks or not.
set(FLASHINFER_PREFILL ON)
# Whether to compile decode kernel tests/benchmarks or not.
set(FLASHINFER_DECODE ON)
# Whether to compile page kernel tests/benchmarks or not.
set(FLASHINFER_PAGE ON)
# Set target cuda architectures for tests/benchmarks, defaults to native (which is new in CMake 3.24, you can configure it manually if you are using older CMake).
# Example: "80;86;90;90"
set(FLASHINFER_CUDA_ARCHITECTURES native)
