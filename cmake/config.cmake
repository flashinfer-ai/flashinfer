# Whether to compile fp8 kernels or not.
set(FLASHINFER_ENABLE_FP8_E4M3 ON)
set(FLASHINFER_ENABLE_FP8_E5M2 ON)
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
set(FLASHINFER_NORM ON)
# Whether to compile fastdiv tests
set(FLASHINFER_FASTDIV_TEST ON)
# Whether to compile fastdequant tests
set(FLASHINFER_FASTDEQUANT_TEST ON)
# Whether to compile distributed tests
set(FLASHINFER_DISTRIBUTED ON)
# The following configurations can impact the binary size of the generated
# library
set(FLASHINFER_GEN_HEAD_DIMS 64 128 256 512)
set(FLASHINFER_GEN_KV_LAYOUTS 0 1)
set(FLASHINFER_GEN_POS_ENCODING_MODES 0 1 2)
set(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS "false" "true")
set(FLASHINFER_GEN_MASK_MODES 0 1 2)

# Set target cuda architectures for tests/benchmarks, defaults to native.
# "native" is a special value for CMAKE_CUDA_ARCHITECTURES which means use the
# architectures of the host's GPU. it's new in CMake 3.24, if you are using an
# older of CMake or you want to use a different value, you can set its value
# here. Supported CUDA architectures include 80;86;89;90
# NOTE(Zihao): using "native" might be slow because whenever compile a cuda file
# with `-arch=native`, nvcc will spawn a `__nvcc_device_query` process to get
# the architecture of the host's GPU, which could stall the compilation process.
# So it's recommended to set it to a specific value if you know the architecture
# of the target GPU. Example: set(FLASHINFER_CUDA_ARCHITECTURES 80)
set(FLASHINFER_CUDA_ARCHITECTURES native)
