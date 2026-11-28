# Source provenance and integration boundary

- Original project: [DeepEP](https://github.com/deepseek-ai/DeepEP), MIT,
  Copyright (c) 2025 DeepSeek.
- NVFP4 SM120 adaptation starting point: fork revision
  `be2582ffe69b5e7d61c3bc9bf7a5316bc48261f9`.
- Subsequent adaptations add BF16/FP8 transport,
  independent fixed-capacity RDMA dispatch/combine launch selection,
  compact typed arenas and BF16-input NVFP4 dispatch.
- FlashInfer integration base: [072c706e](https://github.com/flashinfer-ai/flashinfer/commit/072c706e4233597e8ec7365112195bc422b1f05e).

The adapted native code lives under `include/flashinfer/comm/rdma_moe_alltoall/`
and in `csrc/rdma_moe_alltoall_*.cu`. Kernel helpers retain
the DeepEP quantization, expert routing and weighted-combine behavior.
`csrc/rdma_moe_alltoall_gin_runtime.cu` adapts the original host runtime to
FlashInfer TVM-FFI: Python allocates output tensors and owns handle/hook
metadata, while native runtime objects own communication resources. JIT
builds independent quantization and GIN modules. No NVSHMEM runtime is vendored.

This is an adapted port, not a byte-identical upstream source snapshot.
Framework headers and bindings were replaced; the FlashInfer Python entry
point and JIT integration are new. The correctness reference under
the RDMA MoE AlltoAll tests is also adapted from the
SM120 adaptation’s transport tests.

The original MIT license text is preserved in
[`LICENSE.deepep.txt`](../../../../LICENSE.deepep.txt), with additive attribution
in the repository `NOTICE`. Native adapted source retains the DeepEP copyright
and MIT attribution. This attribution does not change the licenses of other
FlashInfer components. NCCL itself is an external prerequisite, not a vendored
source dependency.

The native relocation retains the original internal `deep_ep` namespace and
algorithm identifiers for attribution and mechanical source comparison. They
do not require an installed DeepEP package. The supported public interface is
FlashInfer's RDMA MoE AlltoAll interface, not the upstream DeepEP API.

See the [dependency and build contract](../../../../docs/development/rdma_moe_alltoall_dependencies.md)
for the verified NCCL version policy and why this backend is JIT-only.
