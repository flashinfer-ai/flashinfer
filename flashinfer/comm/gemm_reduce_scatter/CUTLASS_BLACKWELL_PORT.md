# CUTLASS CuTeDSL Blackwell GEMM+Reduce-Scatter Port Notes

This note tracks the FlashInfer port of the upstream CUTLASS CuTe DSL
`distributed_gemm_reduce_scatter_blackwell.py` example.

## Current State

The Blackwell-only package exports the direct entry point
`gemm_reduce_scatter_blackwell_cutlass` (also aliased as
`gemm_reduce_scatter`) and requires an explicit `BlackwellGemmRSWorkspace`.

It consists of:

- `cutlass_blackwell_gemm_rs.py`: the ported CUTLASS CuTe DSL persistent
  Blackwell GEMM+RS kernel with example runner code removed;
- `gemm_reduce_scatter_blackwell.py`: FlashInfer tensor adapters, NVSHMEM
  workspace, compiled-kernel cache, and backend bridge;
- `bench_flashinfer_blackwell_compare.py`: same-harness comparison against
  naive `torch.mm + reduce_scatter_tensor` and vLLM fused.


## Shape Contract

FlashInfer public contract:

```text
X_local: [M, K_local]
W_local: [K_local, N]
output:  [M / world_size, N]
semantic: output_rank = sum_r(X_local_r @ W_local_r)[rank_M_slice]
```

CUTLASS example contract:

```text
A: [M, K, L]
B: [N, K, L]
C: [M, N, L]
```

For FlashInfer GEMM+RS:

```text
K in CUTLASS example = K_local in FlashInfer
L = 1
A = X_local viewed as [M, K_local, 1]
B = W_local transposed/viewed as [N, K_local, 1]
C = symmetric full [M, N, 1] staging/output tensor
return C[rank * M_local : (rank + 1) * M_local, :, 0]
```

The fair comparison harness already uses this mapping by running CUTLASS with:

```text
mnkl = (M, N, K_total / world_size, 1)
```

## Required Port Work

1. Split the upstream example into reusable pieces.

   Keep:

   - `PersistentDenseGemmKernel`;
   - NVSHMEM multicast tensor setup for C and barrier flags;
   - `two_shot` reduce-scatter path;
   - `torchrun_uid_init_bcast` equivalent only for benchmark scripts, not the
     library entry point.

   Remove from library path:

   - internal random tensor creation;
   - CLI parser;
   - benchmark-only `testing.benchmark`;
   - CPU reference creation.

2. Build tensor adapters for user-provided tensors.

   Required adapters:

   - `X_local` contiguous `[M, K_local]` -> CuTe tensor `[M, K_local, 1]`;
   - `W_local` contiguous `[K_local, N]` -> CuTe tensor compatible with example
     B layout `[N, K_local, 1]`;
   - symmetric output `C_full` -> local CuTe tensor + multicast CuTe tensor +
     peer CuTe tensors.

   The B tensor is the main layout risk. The example expects B indexed as
   `[N, K, L]`; FlashInfer stores W as `[K, N]`. Either:

   - require/construct a contiguous transposed staging tensor `[N, K_local]`;
   - or change the CUTLASS layout path to accept `[K_local, N]` without a copy.

   `b_layout="nocopy"` is the default and creates a logical `[N, K_local, 1]`
   view over native contiguous `W_local [K_local, N]` storage with stride
   `(1, N, 1)`. Misaligned or non-native-stride weights automatically use the
   lazy `b_layout="staged"` fallback unless fallback is explicitly disabled.

3. Manage symmetric output/barrier workspace.

   Minimum workspace fields:

   - full symmetric `C_full: [M, N]` or `[M, N, 1]`;
   - multicast alias for `C_full`;
   - peer tensors for all ranks;
   - barrier flag tensor sized like the example:
     `num_tiles + num_sms`;
   - multicast alias for barrier flags.

   `BlackwellGemmRSWorkspace` owns these collective NVSHMEM allocations. The
   caller constructs and passes it explicitly; FlashInfer does not hide
   collective allocation or lifetime behind an implicit cache.

4. Compile/cache by static kernel parameters.

   Compilation key should include:

   - `M`, `N`, `K_local`;
   - dtype;
   - world size;
   - `mma_tiler_mn`;
   - `cluster_shape_mn`;
   - `use_2cta_instrs`;
   - `use_tma_store`;
   - `reduce_scatter="two_shot"`;
   - tensor layout choices.

   The compiled function is cached at workspace level and keyed by shape, dtype,
   device, world size, kernel configuration, effective B layout, and captured
   tensor pointers/strides.

5. Return FlashInfer-compatible output.

   The CUTLASS example writes a full C tensor but only each rank's M chunk is
   the reduce-scatter result. The library entry point must return:

   ```python
   C_full[rank * M_local : (rank + 1) * M_local, :]
   ```

   The returned view aliases workspace `C_full`. It is valid until the next call
   using that workspace or until `workspace.destroy()`; destruction synchronizes
   the device before collectively releasing NVSHMEM storage.

## Correctness Gates

Before claiming implementation correctness:

- world size 4 on GB200;
- dtype bf16 and fp16;
- M sweep: 2048, 4096, 8192, 16384, 32768, 65536;
- compare against same-harness naive `torch.mm + reduce_scatter_tensor`;
- compare against vLLM fused where available;
- all-M CUTLASS reference checks enabled in the harness;
- rank-distinct input data.

## Performance Gates

Before claiming performance:

- same node/allocation for all contenders;
- same shape contract: `K_total=8192`, `K_local=K_total/world_size`;
- report max-rank mean latency;
- hot-L2 and cold-L2 CUTLASS runs reported separately;
- repeat at least 3 runs;
- include raw per-rank JSON details;
- compare against both vLLM fused and cuBLAS naive.

## Known Risks

- CUTLASS example uses NVSHMEM tensors directly, not PyTorch `symm_mem`.
- vLLM fused works in this environment only when leaving symmetric memory
  backend as PyTorch default; explicit `NVSHMEM` or `NCCL` backend fails with
  `*SymmetricMemoryAllocator::alloc must not be called with a group_name`.
- The CUTLASS example is `two_shot`; make sure the semantics remain acceptable
  for FlashInfer/vLLM integration.
- `gemm_reduce_scatter_blackwell_cutlass` currently supports only `dist.group.WORLD`.
  Subgroups are rejected because the ported CuTe DSL kernel reads global
  `torch.distributed` rank/world state during construction.
- The library supports both B layouts:
  - `b_layout="staged"` stages `W_local.T` into `workspace.w_staging`;
  - `b_layout="nocopy"` maps logical CUTLASS B `[N, K, 1]` directly onto
    FlashInfer-native `W_local [K, N]`.
  No-copy is the default for eligible contiguous weights after passing
  workspace-reuse/pointer-rotation stress and the full ws=2/4/8 matrix.
  Staged remains an explicit override and automatic fallback.
- The reduce-scatter epilogue uses 128-bit packed multimem reduction and peer
  stores. Runtime alignment checks and explicit system-scope release/acquire
  synchronization now guard this path; future opcode changes require re-audit.
- Do not resurrect the old scalar `gemm_reduce_scatter_cutile.py` path for
  performance claims.

## Immediate Validation Added

- The Blackwell backend now fails fast unless `group is dist.group.WORLD`.
- The same-harness benchmark now has opt-in stress mode:

```text
--stress-loops N
--stress-pointer-pool P
```

`--stress-loops` reuses one `BlackwellGemmRSWorkspace` across repeated
correctness calls to exercise barrier-flag reuse. `--stress-pointer-pool > 1`
rotates distinct `X/W` tensor allocations to exercise compiled-cache
invalidation when activation or weight pointers change.

## Production Readiness Completed

- No-copy is the default B layout. Native `[K,N]` stride, dtype, pointer, and
  16-byte row alignment are checked; ineligible weights lazily stage a
  transpose. Explicit staged mode and fail-closed no-fallback mode remain.
- Packed bf16/fp16 reduction uses the CUTLASS helpers
  `multimem.ld_reduce...v4.f16x2` and `...v4.bf16x2`. The four 32-bit result
  registers are written with a bit-preserving 128-bit `v4.f32` peer store; this
  does not numerically convert the packed values to fp32.
- Per-tile readiness is ordered as TMA completion, then
  `multimem.red.release.sys`; reduction observes the flag with an acquire-system
  UC load before the weak/relaxed multimem reduce. Because the counter is
  updated through its MC alias and polled through its UC alias, a
  `fence.proxy.alias` separates those accesses. The MC C reduction and UC owner-peer store are separated by another
  alias-proxy fence. Peer stores are followed by a
  CTA barrier, an elected-thread release-system MC completion signal, another
  alias-proxy fence, and an acquire-system UC CAS before kernel return.
- Local, multicast, and peer C base pointers are checked for 16-byte alignment;
  N and K are required to be multiples of eight bf16/fp16 elements.
- Automated local GB200 tests cover ws=2/4, bf16/fp16, repeated workspace
  reuse, N values 1024/1536/2048/3072, non-contiguous-W staged fallback,
  and an alternate `(256,128)` MMA tile with `(4,1)` cluster. ws=8 cases are
  parametrized but skip on four-GPU hosts; pre-tyche ws=8 remains covered by
  the two-node canonical correctness matrix and stress harness. Unit tests cover default selection,
  fallback disabling, invalid tilers/clusters, alignment, M tails, and ws=16.
- The caller must construct `BlackwellGemmRSWorkspace`. It owns collective
  NVSHMEM allocations and the output. The returned output aliases workspace
  storage and is valid only until the next call on that workspace or destroy.
- Missing CuTe DSL/CUDA Python/NVSHMEM, uninitialized NVSHMEM, and unsupported
  device errors are reported before kernel compilation.

## Remaining Work

1. Add world-size 16+ kernel support.

   Current validation deliberately accepts only `[2,4,8]`; staged and no-copy
   share this kernel and are both rejected before launch at ws=16. Replacing the
   guard requires algorithm work, not a B-layout change:

   - require `m_tiles_in_total >= world_size`;
   - require `m_tiles_in_total % world_size == 0` so every output tile has
     exactly one owner;
   - require `cta_mma_tile_m % world_size == 0`;
   - validate that `m_local_rank = cta_mma_tile_m / world_size` is compatible
     with the four reduction warps and 128-bit copy partition;
   - audit `chunk_id`, rank-owned M ranges, peer stores, and all
     `world_size`/`2 * world_size` signal counts;
   - validate `TEAM_WORLD` multicast and peer access on four-node NVL36;
   - add bf16/fp16 correctness, barrier-reuse stress, and timeout diagnostics.

   The current `M=2048`, tile-M=256 configuration has only eight M tiles, so
   ws=16 would compute `m_tiles_per_rank=0`; adding 16 to an allowlist is unsafe.

2. Implement M-tail tiles.

   M values not divisible by MMA tile M are rejected with a clear error because
   the two-shot rank-ownership mapping has no masked tail path. N/K alignment
   violations are also rejected before launch.

3. Support process-group subgroups.

   The kernel reads global distributed rank/world state and therefore accepts
   only `dist.group.WORLD`.

4. Decide integration-level dispatch.

   The direct Blackwell entry point and explicit workspace ownership are
   the production contract for now. Auto dispatch or a higher-level cache should
   be considered only when dependency provisioning and vLLM integration define
   collective creation/destruction order.

5. Expand performance and long-duration stress coverage.

   Add more N/tile/cluster combinations to the full benchmark matrix and longer
   barrier-reuse runs. The 10-warmup/100-iteration study found tight medians/p75
   but isolated ws=8 maxima in every contender; add per-rank phase timing before
   making max-latency claims.
