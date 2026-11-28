# SM90 WINT4 × AFP8 MegaMoE

The explicit INT4 frontend reuses the MXFP4 dispatch, scheduling, FP8 WGMMA,
SwiGLU and BF16 combine pipeline. It accepts E4M3 activations and signed
INT4 weights with one positive finite BF16 scale per row and K128 group.
Dequantization is `RN_E4M3_satfinite(RN_BF16(q * scale))`, with no zero point.
Activations use per-token E4M3 scaling, accumulation uses `1xacc`, and
combine/output uses BF16. The gate/up clamp defaults to 10; pass
`gate_up_clamp=None` to disable it when constructing the configuration.

## Weight contract

For `E` local experts, hidden width `H`, and intermediate width `I`:

| Input | Dtype | Shape |
| --- | --- | --- |
| `w13` | int8 or uint8 | `[E, 2*I, H/2]`, gate rows followed by up rows |
| `w2` | int8 or uint8 | `[E, H, I/2]` |
| `w13_scale` | BF16 | `[E, 2*I, H/128]` |
| `w2_scale` | BF16 | `[E, H, I/128]` |

Inputs are contiguous on one device. Packed bytes contain two's-complement
values in `[-8, 7]`, with the lower K element in the low nibble. `H` and `I`
must be multiples of 128; a K256 tactic requires multiples of 256.

Preprocessing applies gate8/up8 ordering to FC1 and the SM90 INT4 M16/K64
payload interleave. Prepared nibbles store the sign in bit 2 and the
`abs(q)-1` index in bits 0, 1 and 3; zero uses code 11. Each scale becomes an
eight-byte table for `+1..+4` and `-5..-8`. Two PRMT lookups use sign replication
and XOR to recover four signed FP8 bytes, reusing each table across the K32
fragments in a K128 group. Paired rows share a 128-bit table load. Zero-filled
payload and metadata decode to positive zero.

Each prepared leg contains `(payload, tables, unit_activation, unit_expert)`:
`payload` is uint8 `[E,K/2,M]` with storage-K stride 1; `tables` is contiguous
uint64 `[E,M/64,K/128,8,8]`. Within each M64 block, row `m` has table index
`p = 16*((m%64)//16) + 2*(m%8) + (m//8)%2`, stored at
`[e,m//64,k//128,p//8,p%8]`. This pairs rows `u` and `u+8` inside each M16 block.
The two FP32 unit tensors preserve the shared launcher ABI. Prepared weights
are specific to this INT4 frontend.

## Public API

Initialize the torch distributed process group and select each rank's CUDA
device before constructing the layer. Given packed weights and a
`MoEEpTensors` input bundle:

```python
from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    MegaConfig,
    MoEEpLayer,
    MoEWeightPack,
    Sm90_Fp8_Int4_Bf16_PullCutedsl_MegaMoeConfig,
    preprocess_sm90_pull_int4_mega_weights,
)

weights = MoEWeightPack(
    w13=w13, w2=w2, w13_scale=w13_scale, w2_scale=w2_scale,
)
prepared = preprocess_sm90_pull_int4_mega_weights(
    weights, hidden_size=H, intermediate_size=I,
)
layer = MoEEpLayer(
    bootstrap=BootstrapConfig(world_size=world_size, rank=rank, process_group=pg),
    fleet_params=FleetParams(
        num_experts=E_total, max_tokens_per_rank=capacity, token_hidden_size=H,
    ),
    weights=weights,
    backend=MegaConfig(
        megakernel=Sm90_Fp8_Int4_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=I, top_k=topk,
        ),
        preprocess_weights=False,
        transformed_weights=prepared,
    ),
)
try:
    layer.warmup(inputs)
    layer.stage_inputs(inputs)
    layer.compute_staged(output=preallocated_bf16_output)
finally:
    layer.destroy()
```

Alternatively, leave `MegaConfig.preprocess_weights=True` and let the layer
prepare weights once. `layer.forward(inputs)` includes input staging. All
ranks allocate, launch and destroy in the same order.

The low-level `flashinfer.moe_ep.shims.sm90.cutedsl_megamoe` package exposes
`get_symm_buffer_for_hopper_int4_mega_moe`, `hopper_int4_mega_moe`, and
`hopper_int4_mega_launch_thunk`. These use the same prepared weights and
MXFP4 activation staging. Returned output aliases the symmetric buffer unless
copied. Construct a launch thunk on the capture stream before CUDA Graph
capture, and rebuild it after changing the configuration, pointers or stream.

## Tactic selection and tuning

`get_symm_buffer_for_hopper_int4_mega_moe(..., knobs=None)` looks up the
INT4 tuning cache, then uses a shape-compatible fallback. A `knobs` dictionary
selects a complete explicit tactic. INT4 measurements have their own cache
identity, including the GPU, model shape, routing profile and candidate domain.
The weight group size remains 128 for every tactic, including K256 tiles.

The high-level backend is
`Sm90_Fp8_Int4_Bf16_PullCutedsl_MegaMoeConfig(intermediate_size=I, top_k=topk,
knobs="auto")`. Its first compute collectively measures candidates on all EP
ranks and retains the winner; rank 0 writes the cache. Complete this first
compute with a caller output buffer before CUDA Graph capture. For serving,
prepare the cache offline and use `knobs=None`:

```bash
torchrun --standalone --nproc_per_node=2 -m flashinfer.moe_ep.tune \
    --dtype sm90_int4 --hidden 3072 --intermediate 1024 \
    --num-experts 256 --topk 8 --max-tokens 128 512 2048 8192
```

The low-level `autotune_hopper_int4_mega_moe(y, fc1, fc2, buffer)` is also
available. Rebuild any CUDA Graph launch thunk after tuning.

Set `FLASHINFER_MOE_EP_KNOB_CACHE` to choose the cache file. Tune with the
deployment's GPU, EP size, routing profile and token capacities. The supported
candidate domain currently requires `fc1_ready_mode="tile"` and
`fc2_tail_n8=False`.

## Tests

Run the Hopper conversion, correctness and multi-rank suites:

```bash
bash tests/moe_ep/run_tests.sh oracle_sm90_int4
NPROC_MULTIRANK=2 bash tests/moe_ep/run_tests.sh mega_sm90_int4
```

The multi-rank suite also supports 4 and 8 ranks. It checks the public raw
and prepared weight paths, cross-rank and masked routing, workspace reuse,
and collective CUDA Graph replay against an independent reference.
