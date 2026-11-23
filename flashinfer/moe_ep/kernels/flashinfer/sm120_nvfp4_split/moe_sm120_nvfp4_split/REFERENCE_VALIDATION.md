# Combine numerical contracts and validation

`combine_reference.py` is a PyTorch-only module usable on CPU or CUDA. It
does not import CuTeDSL/NVSHMEM or execute the production reducer.

## Two arithmetic contracts

Inputs to both functions are **BF16 per-expert FC2 outputs**, including whatever
top-k weighting is specified by the selected upstream compute graph. The combine
reference must not apply top-k weights a second time.

- `flat_topk_bf16_reference`: widen each slot to FP32, add in ascending top-k
  slot order, round the final sum to BF16.
- `owner_grouped_bf16_reference`: for each owner rank, add its slots in ascending
  slot order in FP32; round the **owner partial** to BF16; widen and add owner
  partials in ascending rank order; round the final output to BF16.

The second function deliberately models the extra owner-partial rounding in
RLC. It does not claim numerical equivalence to the first. Empty owners
contribute zero. Neither function uses a parallel reduction tree.

For arithmetic validation, `MegaMoETester.compute_reference()` first obtains
per-expert outputs from the independent NVFP4 reference. In contrast,
`rank_local_staging_report()` starts from **actual kernel staging** and checks
mapping, metadata, group/contributor counters, sideband plan, peer partials,
ready epoch and final K3 output. Passing that transport oracle cannot certify
FC1/FC2 arithmetic.

## Three independent results

The runner emits a JSON `COMBINE_VALIDATION` record on every rank:

| Field | Meaning | Enforcement |
|---|---|---|
| `grouped_implementation` | RLC output vs independent grouped BF16 reference, byte-exact and finite | Failures always fatal |
| `ordinary_topk` | Output vs ordinary slot-ordered reference; element counts, max/mean error, nonfinite count | Fatal when `MEGA_STRICT_TORCH_REF=1`, otherwise explicitly reported as FAIL but advisory |
| `staging_transport` | Actual-staging reconstruction and completion-state audit | Failures always fatal |

Tolerance is unchanged: `abs(actual-ref) <= 0.05 + 0.05*abs(ref)`.
Matching NaN/Inf are not passes. Verdicts are gathered over the EP group before
any rank raises, including when only a different rank fails.

`NOT_CHECKED` means not executed/unsupported, not a pass. In particular, RLC-off
direct peer stores have no independent pre-store staging buffer.
`NOT_APPLICABLE` for grouped arithmetic means RLC is disabled.
There is no aggregate "all correctness passed" flag. Exit success with strict
mode disabled does **not** certify ordinary top-k numerical equivalence.

Checks run after completed launches, outside captured/timed graph regions.
`--skip_ref_check` retains its existing meaning. Transport-only stress can use
`--transport_replay_validation`; it does not implicitly run arithmetic reference.
The generic runner validates the final replay, not every dynamic replay.

## Regressions

CPU:

```sh
python moe_sm120_nvfp4_split/test_combine_reference.py -v
```

Includes the RLC-42 cancellation example (flat=0, grouped=-512), count 0/1/2/3,
ordered FP32 additions, signed-zero byte comparison, invalid routes/mapping,
stale partials/epochs, nonfinite values, 100 changing-route CPU reconstructions,
and the actual runner strict/collective entry methods.

EP GPU regression:

```sh
# Prefix with the usual CUDA_VISIBLE_DEVICES/NVSHMEM/environment settings.
# Run from the repository root with the usual MegaMoE shape/launch arguments.
MEGA_STRICT_TORCH_REF=0 MEGA_CONTRACT_DYNAMIC=1 \
  torchrun --standalone --nproc_per_node=4 \
  moe_sm120_nvfp4_split/check_combine_contract_gpu.py \
  --num_tokens_per_rank 257 --hidden 4096 --intermediate 4096 \
  --num_topk 6 --num_total_experts 256 --gate_up_clamp 10 \
  --enable_static_expert_shape --comm_backend p2p_direct \
  --split_launch green_graph --rank_local_combine on --dispatch_warps 1 \
  --perf_run --use_cuda_events --perf_warmup 0 --perf_iters 100 \
  --dynamic_route_replays 100
```

This **diagnostic** wrapper audits all 100 actual graph replays, checks epoch
advancement and changing route hashes, and recomputes the independent expert
arithmetic reference on replays 0/49/99. Reported timings are perturbed and must
not be used for performance claims. Set `MEGA_CONTRACT_OUTPUT_DIR` to save JSON.
For the known RLC-42 static fixture T1537, strict mode plus
`MEGA_EXPECT_ORDINARY_FAILURE=1` is a negative regression: it requires ordinary
top-k to fail while grouped arithmetic and staging transport both pass.
