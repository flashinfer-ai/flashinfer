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
# Whether to compile sampling kernel tests/benchmarks or not.
set(FLASHINFER_SAMPLING ON)
# Whether to compile normalization kernel tests/benchmarks or not.
set(FLASHINFER_NORMALIZATION ON)
# The following configurations can impact the binary
# size of the generated library
set(FLASHINFER_GEN_GROUP_SIZES 1 4 6 8)
set(FLASHINFER_GEN_PAGE_SIZES 1 16 32)
set(FLASHINFER_GEN_HEAD_DIMS 64 128 256)
set(FLASHINFER_GEN_KV_LAYOUTS 0 1)
set(FLASHINFER_GEN_POS_ENCODING_MODES 0 1 2)
set(FLASHINFER_GEN_ALLOW_FP16_QK_REDUCTIONS "false" "true")
set(FLASHINFER_GEN_MASK_MODES 0 1)

# Set target cuda architectures for tests/benchmarks, defaults to native.
# "native" is a special value for CMAKE_CUDA_ARCHITECTURES which means use the architectures of the host's GPU.
# it's new in CMake 3.24, if you are using an older of CMake or you want to use a different value, you can
# set its value here. Supported CUDA architctures include 80;86;89;90
# Example:
# set(FLASHINFER_CUDA_ARCHITECTURES 80)
set(FLASHINFER_CUDA_ARCHITECTURES native)
