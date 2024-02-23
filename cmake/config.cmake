# Whether to compile fp8 kernels or not.
set(FLASHINFER_ENABLE_FP8 ON)
# Whether to compile bf16 kernels or not.
set(FLASHINFER_ENABLE_BF16 ON)
# Whether to compile tvm bindings or not.
set(FLASHINFER_TVM_BINDING ON)
# Whether to compile prefill kernel tests/benchmarks or not.
set(FLASHINFER_PREFILL ON)
# Whether to compile decode kernel tests/benchmarks or not.
set(FLASHINFER_DECODE ON)
# Whether to compile page kernel tests/benchmarks or not.
set(FLASHINFER_PAGE ON)
# Whether to compile cascade kernel tests/benchmarks or not.
set(FLASHINFER_CASCADE ON)
# Set target cuda architectures for tests/benchmarks, defaults to native.
# "native" is a special value for CMAKE_CUDA_ARCHITECTURES which means use the architectures of the host's GPU.
# it's new in CMake 3.24, if you are using an older of CMake or you want to use a different value, you can
# set its value here. Supported CUDA architctures include 80;86;89;90
# Example:
# set(FLASHINFER_CUDA_ARCHITECTURES 80)
set(FLASHINFER_CUDA_ARCHITECTURES native)
