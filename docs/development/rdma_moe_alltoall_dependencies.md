# RDMA MoE AlltoAll dependencies and build contract

RDMA MoE AlltoAll uses NCCL's GPU device API and GDAKI transport for non-self
messages. It requires an SM120-capable CUDA toolchain, matching NCCL host/device
headers and runtime, and a working GPUDirect RDMA provider. Self contributions
remain local. An NVSHMEM runtime is not required.

## Verified version policy

The currently supported NCCL version is **2.30.7**, with GDAKI required. This
exact-version check records the implementation
and validation boundary; it is not a claim that other versions lack the APIs or
that 2.30.7 was the first NCCL release to provide them.

The protocol relies on registered-window puts, strong-signal receive ordering,
and `flushAsync` followed by `wait` for source-buffer completion on shared
contexts. Construction checks device API support, an available GIN transport,
the selected provider, and equality of the header and actually loaded runtime
versions. New versions require checking the relevant completion semantics and
validating the current source, including the NVFP4 rank-packet path, with
matching headers and runtime; merely lowering a guard is insufficient.

## Installation and runtime selection

`NCCL_HOME` selects an installation first; otherwise the loader uses the
`nvidia.nccl` Python package. `NCCL_INCLUDE_DIR` and `NCCL_LIB_DIR` can override
its include and library directories. The selected include directory must
provide `nccl.h` and `nccl_device.h`, and the library directory must provide
`libnccl.so.2`.

The host framework may load NCCL before the communication object is created.
Select the intended runtime before starting Python, using the application's
normal loader configuration or `LD_PRELOAD` when needed. The constructor
rejects a header/runtime mismatch rather than silently using another library.
NIC selection and InfiniBand/RoCE configuration remain deployment concerns;
an import alone does not validate the network setup.

## JIT and package layout

The generator lives in `flashinfer/jit/rdma_moe_alltoall.py` and uses
FlashInfer's normal `gen_jit_spec` and `build_and_load` infrastructure. Native
translation units and TVM-FFI helpers are under `csrc/`; framework-independent
headers are under `include/flashinfer/comm/rdma_moe_alltoall/`. Standard
`flashinfer.data` package-data rules include these sources in source and wheel
installations; the experimental-source exclusion does not apply.

This feature is intentionally **JIT-only**. The GIN binary compiles against an
external NCCL device SDK and links its runtime. Its cache key includes the
resolved dependency locations and hashes of the NCCL headers and library, in
addition to the module's own sources. A generic prebuilt wheel cannot assume
that exact external installation or safely substitute a binary linked against
another NCCL version. Consequently, its loaders are not registered in
`flashinfer/aot.py`, following the external-dependency precedent used by
Ulysses. This is a build/dependency choice, not experimental API status.

Quantization is a separate JIT module. It needs no NCCL headers or RDMA
provider, and loading it does not initialize GIN. Both modules retain the
validated `sm_120a` target and explicit arithmetic without fast-math changes.

Source attribution and the retained MIT license are described in
[VENDOR.md](../../include/flashinfer/comm/rdma_moe_alltoall/VENDOR.md).
