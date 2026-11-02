# Cake packed-varlen GQA training attention backend (head dim 256, SM100 / SM103 / SM107)

Experimental backend for the forward and first-order backward of causal GQA
attention over packed variable-length segments, head dimension 256: BF16
`q [T_q, H_q, 256]`, `k`, `v [T_k, H_kv, 256]` (`H_q / H_kv` even), segments
packed by `cu_seqlens_q` / `cu_seqlens_k` (int32 on the device, never read on
the host).  Query row `i` of a segment with `q_len` rows and `k_len` keys sees
the keys `j <= i + k_len - q_len` (FlashAttention's bottom-right alignment);
a row that sees no key gets `out = 0`, `lse = -inf` and zero gradients.
`T_q`, `T_k`, the segment count and every length are run-time values: no
shape list, no recompilation, no device-to-host read, no device-wide
synchronization.

Both the API and the backend are experimental: no compatibility guarantee.
SM100 (B200 / GB200), SM103 (B300 / GB300) and SM107 (R200) are the
supported architectures; another device raises `RuntimeError`.  The feature does not
participate in automatic backend selection, autotuning or trace-apply.  Its
libraries are loaded prebuilt from `aot/` when a builder compiled them and
JIT-built by FlashInfer from the sources and flags `build.json` records
otherwise.  The package reads no environment variable.

## Opt-in

```python
from flashinfer.gqa_train import varlen_gqa_attention, gqa_train_workspace

ws = gqa_train_workspace(q.device)               # optional; reused across calls and shapes
out = varlen_gqa_attention(
    q, k, v, cu_seqlens_q, cu_seqlens_k,
    softmax_scale=0.0625,                        # default 256 ** -0.5
    max_seqlen_k=max_k,                          # optional host bound of the longest key segment
    workspace=ws,
)
out.backward(dout)                               # BF16 dq, dk, dv

# FP32 dK / dV added into caller-owned FP32 destinations [N_dst, H_kv, 256]
# through an int32 row map [T_k] (repeated rows accumulate); the callable runs
# in backward, so nothing is held between the forward and the backward:
out = varlen_gqa_attention(q, k, v, cu_seqlens_q, cu_seqlens_k,
                           dkv_acc=lambda: (dk_dst, dv_dst, row_map))
# FP32 window gradients [T_k, H_kv, 256] stored into caller-owned tensors:
out = varlen_gqa_attention(q, k, v, cu_seqlens_q, cu_seqlens_k,
                           dkv_acc=lambda: (dk_win, dv_win))
```

`varlen_gqa_attention_fwd` / `varlen_gqa_attention_bwd` are the two halves
without autograd (`dkv_dtype=torch.float32` returns FP32 window gradients;
`dk_acc`, `dv_acc`, `row_map` add into destinations).  `max_seqlen_k` (a
host int, never read from the device) must be at least every segment's key
count, as FlashAttention's `max_seqlen_k`: a smaller value leaves key blocks
out.  It sizes the backward grid (one CTA pair per 128 keys of the longest
segment, per key head and segment); without it the bound is `T_k`, which is
always correct but launches CTAs that retire at once: on a 12-segment layout of
257K keys the backward call took 8 to 15 % longer without it on SM100, SM103
and SM107 GPUs.  The backend package
exposes the same entry points (`varlen_gqa_attention`, `gqa_attention_fwd`,
`gqa_attention_bwd`, `Workspace`).

## Kernels

| stage | kernel |
|---|---|
| `fwd` | O and the natural-log LSE `[H_q, T_q]`: a CTA pair per 256 query rows of one (segment, query head), persistent tile tickets from a self-resetting counter |
| `bwd_pre` | `dPsum = sum(O * dO)`, `LSE * log2(e)` and the zeroed FP32 dQ accumulator, two query heads per scratch head |
| `bwd_fp32` / `bwd_rowmap` / `bwd_bf16` | the 5-GEMM backward: a CTA pair per 128 keys of one (segment, key head), looping the key head's query heads two at a time, dK / dV summed in FP32 in tensor memory; FP32 window stores, FP32 row-map adds or BF16 stores |
| `bwd_post` | `dQ = bf16(acc * softmax_scale)` |

Numerics: FP32 accumulation for the scores, `dP`, dK, dV, the dQ partials and
the row sums; P and dS are rounded to BF16 only as MMA operands.  dQ and the
row-map adds use atomic reductions, so they are not bitwise reproducible run
to run.

## Checkpointing and memory

Autograd saves `(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)` through
`ctx.save_for_backward` only; `ctx` holds scalars, the `dkv_acc` callable and
a weak reference to the workspace.  Under
`torch.utils.checkpoint(..., use_reentrant=False)` a forward retains nothing.
The backward scratch is FP32 `[H_q / 2, rows_pad, 256]` plus two
`[H_q / 2, rows_pad]` with `rows_pad = round_up(2 T_q + 128 B, 128)`
(`gqa_train_workspace_bytes`); a workspace keeps the largest one seen
(`.nbytes`, `.release()`).  One workspace serves one stream at a time;
without one, each device stream gets its own default workspace.

## Limitations

- Head dimension 256, BF16 inputs, causal only (bottom-right aligned), no
  dropout, no sliding window.
- `H_q / H_kv` must be even.
- `dout` of the low-level backward must be contiguous (autograd makes it so).
- `sm_100a`, `sm_103a` and `sm_107a` only.
