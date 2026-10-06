# Cake Kimi-K3 AttnRes (experimental, SM100 / SM103)

Generated-program backend for the Kimi-K3 attention-residual mixing operator
(`_apply_attn_res` of the model; the vLLM `attn_res` op): on caller-owned BF16
tensors with hidden size 7168 and `K = num_blocks` in `[0, 8]`, one launch
performs the BF16 residual add `prefix += delta`, the optional exact snapshot
write `blocks[:, block_write_idx, :] = prefix`, the FP32 RMS-normalized score
of the `K + 1` candidate rows (`K` snapshot rows and the updated prefix), a
stable FP32 softmax over the candidates, the FP32 probability-weighted mix, and
the fused output RMSNorm with a single BF16 cast into `out`.

Public entry points: `flashinfer.kimi_k3_attn_res.prepare_kimi_k3_attn_res`
(bind one call; `launch()` allocates nothing and is CUDA-graph capturable) and
`flashinfer.kimi_k3_attn_res.kimi_k3_attn_res` (prepare + launch).

## Routes

`cake_backend.plan_route(arch, sm_count, M, num_blocks, enable_pdl, ...)` selects
the generated program from host-known facts only (the Cake production policy,
ported and checked row by row by the export):

| kind | program | when |
| --- | --- | --- |
| `small_m_direct:k{K}` | one token per 256-thread CTA, every source register-resident (the first three unpacked to f32 once), no TMA / mbarrier / TMEM; one cross-warp exchange for all K+1 statistics pairs | `M <= _SMALL_M_DIRECT_MAX_M[arch][K]` (256 / 512 for K0-K3, 128 for K4-K7, 64 for K8); selected ahead of every other dense program |
| `small_m_cluster{2,4}:k{K}` | the same program split over a cluster of 2 or 4 CTAs per token (128 / 64 threads each); each warp pushes its statistics pairs into every CTA's table over DSM, one cluster barrier per exchange, identical reduction order | the `(K, M band)` entries of the per-architecture table `_SMALL_M_CLUSTER` (K4-K8 at the smallest `M`; absent `K` or `M` above the last band -> one CTA per token) |
| `native_{k5,k6,k7,k8,m128}` | installed native ports, 148 CTAs x 288 threads, three sources per chunk, depth 2 | the measured cells of the 148-SM parts (`M = 1` for K5 / K6 / K7, `M <= 256` cells for K4, small `M` / `M = 256` for K8) |
| `k0_tma` | exact `K = 0` persistent path (bulk-copy fed, no TMEM); grid 2x/3x the SM count at `M` 256-1024 | `num_blocks == 0` |
| `persistent` | persistent TMEM common path, 288 threads, `nc` sources per chunk (1..5), depth 2 or 3, retraced per cell | every other dense call with `delta` and `output_norm_weight` and no snapshot write |
| `direct` | one 256-thread CTA per token | no `delta`, snapshot write, no output norm, or row-padded layouts |

`KERNELS[arch][kernel_key]` in `cake_jit.py` names the registered module of a
plan.  The checkout registers the programs measured on the Kimi-K3 evaluation
grid: `M` in {1, 2, 4, ..., 16384} x `K` in {0, 1, 4, 8}, every `K` at `M` = 1 /
4096, and the semantic variants at `M` = 1 / 3 / 7 / 17, with and without
programmatic dependent launch, on `sm_100a` and `sm_103a` (148 SMs).  The route
tables above describe every dense call; when they name a schedule variant the
checkout does not register (a token count off the grid, `K` 5-7 at mid `M`, ...),
`plan_route` substitutes the closest registered variant of the same program
family and block count (`small_m`: the same cluster size without the chunk
suffix, then cluster 2, cluster 4, direct; `persistent`: the same-`K` variant
with the same `nc` / depth, then the same `nc`, then any, in a fixed order) and
records the table key in `RoutePlan.fallback_from`; the substituted plan's
`route_id` ends in `.registered_fallback`.  Measured cells always run their
exact program.  `generated_program_available` answers whether the resolved
program is registered without raising; a call whose resolved program is not
registered raises `NotImplementedError`.

## Correctness contract

Output within `atol 8e-2 / rtol 3e-2` of the independent FP32 reference
(`cake_backend.reference_kimi_k3_attn_res`); `prefix` and `blocks` bit-exact
(including every unwritten byte); read-only inputs untouched.

## Generated sources

`csrc/cake_kimi_k3_attn_res/<arch>/*.cu` (device + TVM-FFI binding per module)
and the `MODULES` / `KERNELS` literals of `cake_jit.py` are receipt-bound
exporter output: do not edit by hand.
