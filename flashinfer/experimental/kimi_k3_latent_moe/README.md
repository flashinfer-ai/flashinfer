# Experimental Kimi-K3 Stable LatentMoE front / tail projections (generated programs)

Both this API and its generated-program backend are experimental and may
change without backward compatibility. Calling the API explicitly opts into
the experimental feature and emits FlashInfer's experimental API warning.
There is no automatic backend selection.

`flashinfer.kimi_k3_latent_moe` computes, per rank and layer, the two dense
projection groups that surround the routed experts of `nvidia/Kimi-K3-NVFP4`
(`KimiSparseMoeBlock`, latent width 3584; flashinfer-ai/flashinfer#4568):

```
x [T, 7168] bf16
front:  logits     = FP32(x @ gate_weight.T)                       [T, 896]      router logits
        latent     = BF16(x @ down_weight.T)                       [T, 3584]     routed-expert input
        shared_act = SiTU(x @ shared_gate.T, x @ shared_up.T)      [T, 6144/TP]  shared-expert intermediate
tail:   y   = KimiRMSNorm(sum of the P routed partials)            [T, 3584]     FP32 normalise -> BF16 -> * bf16 weight, eps 1e-5
        out = BF16(y[:, cols] @ up_weight[:, cols].T + shared_act @ shared_down.T)   [T, 7168], one rounding of the fp32 sum
SiTU(g, u) = 4 tanh(g / 4) sigmoid(g) * 25 tanh(u / 25)   (fp32 math, one bf16 rounding)
```

`cols` is this rank's `3584 / TP` latent slice (row-parallel up-projection);
with TP > 1 the caller all-reduces `out` afterwards.  Weights are the checkpoint's
`nn.Linear` `[out, in]` BF16 tensors in the serving layout (vLLM `KimiMoE` /
SGLang `KimiK3MoE`): `gate_weight`, `down_weight`, `norm_weight`, `up_weight`
replicated; the shared experts `MergedColumnParallel` (gate/up rows sharded,
passed as one `[2 * 6144/TP, 7168]` concatenation) + `RowParallel`
(`shared_down_weight` `[7168, 6144/TP]`).  Nothing is copied or re-packed.

## Route (mirrors the Cake production launcher)

| stage | tokens | program | launches |
| --- | --- | --- | --- |
| front | `T <= 128` | weight-streaming swapped-AB tcgen05 kernel (`decode:*`): 128-row weight tiles are the MMA A operand, the tokens the padded B operand (N in {8, 16, 32, 64, 128}), router / latent / shared SiTU tiles in one unified tile space; aligned 2-CTA cluster pairs when `2 x tiles` fits one wave (TP8), one CTA per tile otherwise (TP1) | 1 |
| front | `T > 128` | persistent 2-CTA tcgen05 GEMM (`front:i<6144/TP>`), 256x256 pair tiles over the K-concatenated `[gate; down; shared_gate; shared_up]` rows with class-specific epilogues | 1 |
| tail | `T <= 128` | the same streaming kernel with the KimiRMSNorm fused into the launch (`decode:*_f*`): the epilogue warps normalise the routed rows while the TMA warp streams the shared-down segment; for the smallest T the normalised rows are written straight into a resident smem B operand (`_sb`, staged rows `_rs`), larger T use the global `y_workspace` protocol | 1 |
| tail | `T > 128` | one-pass RMSNorm kernel (`tail_norm:e<0|1>`, late / early PDL trigger chosen from the GEMM grid) + persistent 2-CTA GEMM (`tail_gemm:tp<1|8>`) over `[up slice | shared down]` with trailing-wave stream-K, launched programmatic-dependent | 2 |

The host planner (`cake_backend.decode_front_plan`, `decode_tail_plan`,
`prefill_tail_plan`, `split_plan`) is the Cake planner re-implemented; every
plan names its physical kernel through a logical key registered in
`cake_jit.KERNELS[arch]`.  The plans were frozen for 148-SM devices (B200 /
B300); `prepare_*` refuses other SM counts.  Nothing is planned per launch
and nothing is allocated at launch (per-device scratch buffers are created at
preparation), so a prepared runner (or a CUDA Graph capturing it) replays for
new values written into the bound buffers.

```python
import torch
from flashinfer.kimi_k3_latent_moe import (
    kimi_k3_latent_moe_front,
    kimi_k3_latent_moe_tail,
    prepare_kimi_k3_latent_moe_front,
    prepare_kimi_k3_latent_moe_tail,
)

T, tp, rank = 16, 8, 0
i_local = 6144 // tp
dev = torch.device("cuda")
x = torch.randn(T, 7168, device=dev, dtype=torch.bfloat16)
logits = torch.empty(T, 896, device=dev, dtype=torch.float32)
latent = torch.empty(T, 3584, device=dev, dtype=torch.bfloat16)
shared_act = torch.empty(T, i_local, device=dev, dtype=torch.bfloat16)
# gate_weight [896, 7168], down_weight [3584, 7168], shared_gate_up [2 * i_local, 7168] (gate rows then up rows)
runner = prepare_kimi_k3_latent_moe_front(x, gate_weight, down_weight, shared_gate_up, logits, latent, shared_act)
runner()  # or capture into a CUDA Graph

# tail: routed [P, T, 3584] partials, norm_weight [3584], up_weight [7168, 3584], shared_down [7168, i_local]
out = torch.empty(T, 7168, device=dev, dtype=torch.bfloat16)
y = torch.empty(T, 3584, device=dev, dtype=torch.bfloat16)
kimi_k3_latent_moe_tail(routed, norm_weight, up_weight, shared_act, shared_down, out, tp=tp, rank=rank, y_workspace=y)
```

## Limits

- SM100 (B200) and SM103 (B300) only: tcgen05 / TMEM / TMA / 2-CTA MMA programs.
- TP 1 and TP 8 (no expert parallelism); TP 12 (7168 / 12 is not a multiple of
  64 for the tail slice) is out of scope (flashinfer-ai/flashinfer#4542).
- Model-layout weights only; the chunk-major packed-weight variant of the
  Cake kernels (+3-8 % on decode rows, one extra weight copy per rank) is not
  exported.
- Exactly the shapes of the validated row set are supported: `T` in
  `{1, ..., 128}` for the decode programs (any count, padded to the next N)
  and `T > 128` for the prefill programs (any count; validated up to 16384).

## Files

- `cake_backend.py`: host planner, operand validation, launch binding, runners.
- `cake_jit.py`: `MODULES` / `KERNELS` registry populated by the Cake export;
  JIT specs of the generated programs (`gen_jit_spec`, per-arch nvcc flags).
- `csrc/cake_kimi_k3_latent_moe/<arch>/*.cu`: generated device + host binding
  translation units (do not edit; `csrc/.clang-format` disables formatting).
- `tests/experimental/test_cake_kimi_k3_latent_moe.py`: host-plan unit tests
  and GPU correctness against the FP32/BF16 torch reference for both stages,
  TP 1 / 8, `T in {1, 8, 16, 128, 256, 4096}` (bit-identical re-launch and
  CUDA-graph replay; `y` byte-exact).
