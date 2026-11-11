# SM90 WINT4 × AFP8 MegaMoE

The explicit INT4 frontend reuses the MXFP4 dispatch, scheduling, FP8 WGMMA,
SwiGLU and BF16 combine pipeline. It accepts E4M3 activations and signed
INT4 weights with one positive finite BF16 scale per row and K128 group.
Dequantization is `RN_E4M3_satfinite(RN_BF16(q * scale))`, with no zero point.

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
eight-byte table for `1` through `8`. Two PRMT lookups use sign replication
and XOR to recover four signed FP8 bytes, reusing each table across the K32
fragments in a K128 group. Zero-filled metadata decodes to signed zeros.

Each prepared leg contains `(payload, tables, unit_activation, unit_expert)`:
`payload` is uint8 `[E,K/2,M]` with storage-K stride 1; `tables` is contiguous
uint64 `[E,M/64,K/128,8,8]`, with logical row `m` at
`[e,m//64,k//128,(m%64)//8,m%8]`. The two FP32 unit tensors preserve the shared
launcher ABI. Prepared weights are specific to this INT4 frontend.

## Explicit launch

```python
from flashinfer.moe_ep.kernel_src.sm90 import pull_style_cutedsl_megakernel as mega
from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl.staging import (
    stage_mega_moe_inputs,
)

weights = mega.Int4WeightPack(w13, w13_scale, w2, w2_scale)
fc1, fc2 = mega.preprocess_int4_mega_weights(
    weights, hidden_size=H, intermediate_size=I,
)
buffer = mega.get_symm_buffer_for_hopper_int4_mega_moe(
    E_total, capacity, topk, H, I, rank, world_size,
    mma_tiler_mnk=(256, 64, 128),
)
try:
    stage_mega_moe_inputs(
        hidden_states, topk_weights, topk_ids,
        buffer.x, buffer.x_sf, buffer.topk_idx, buffer.topk_weights,
        quantize_input=True,
    )
    y = mega.hopper_int4_mega_moe(
        None, fc1, fc2, buffer, num_tokens=hidden_states.shape[0], sync=True,
    ).clone()
finally:
    buffer.destroy()
```

Initialize distributed/NVSHMEM state as for the other SM90 MegaMoE frontends;
all ranks allocate and launch in the same order. Staging clears inactive rows
when token counts shrink. The returned output aliases the buffer unless copied.
For CUDA Graph, construct `hopper_int4_mega_launch_thunk(fc1, fc2, buffer)` on
the capture stream before capture, then capture calls to that thunk.

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

The high-level backend accepts prequantized
`MoEWeightPack(w13, w2, w13_scale, w2_scale)` with BF16 K128 scales;
note its argument order differs from `Int4WeightPack` above.
The low-level `autotune_hopper_int4_mega_moe(y, fc1, fc2, buffer)` is also
available. Rebuild any CUDA Graph launch thunk after tuning.

Set `FLASHINFER_MOE_EP_KNOB_CACHE` to choose the cache file. Tune with the
deployment's GPU, EP size, routing profile and token capacities. The supported
candidate domain currently requires `fc1_ready_mode="tile"` and
`fc2_tail_n8=False`.
