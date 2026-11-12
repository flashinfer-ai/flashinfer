# Cake non-deterministic packed-varlen GQA training attention backend (head dim 256, SM100 / SM103 / SM107)

Backend of `flashinfer.cake_gqa_nondet_train`: the forward and first-order backward of causal GQA
attention over packed variable-length segments, head dimension 256: BF16
`q [T_q, H_q, 256]`, `k`, `v [T_k, H_kv, 256]` (`H_q / H_kv` even), segments
packed by `cu_seqlens_q` / `cu_seqlens_k` (int32 on the device, never read on
the host).  Query row `i` of a segment with `q_len` rows and `k_len` keys sees
the keys `j <= i + k_len - q_len` (FlashAttention's bottom-right alignment);
a row that sees no key gets `out = 0`, `lse = -inf` and zero gradients.
`T_q`, `T_k`, the segment count and every length are run-time values: no
shape list, no recompilation, no device-to-host read, no device-wide
synchronization.

The facade `flashinfer.cake_gqa_nondet_train` is a stable API; this backend
package is experimental (no compatibility guarantee for its internal names).
SM100 (B200 / GB200), SM103 (B300 / GB300) and SM107 (R200) are the
supported architectures; another device raises `RuntimeError`.  The feature does not
participate in automatic backend selection, autotuning or trace-apply.  Its
libraries are loaded prebuilt from `aot/` when a builder compiled them and
JIT-built by FlashInfer from the sources and flags `build.json` records
otherwise.  The package reads no environment variable.

## Opt-in

```python
from flashinfer.cake_gqa_nondet_train import cake_gqa_nondet, cake_gqa_nondet_train_workspace

ws = cake_gqa_nondet_train_workspace(q.device)   # optional; reused across calls and shapes
out = cake_gqa_nondet(
    q, k, v, cu_seqlens_q, cu_seqlens_k,
    softmax_scale=0.0625,                        # default 256 ** -0.5
    max_seqlen_k=max_k,                          # optional host bound of the longest key segment
    workspace=ws,
)
out.backward(dout)                               # BF16 dq, dk, dv

# FP32 dK / dV added into caller-owned FP32 destinations [N_dst, H_kv, 256]
# through an int32 row map [T_k] (repeated rows accumulate in window order); the callable runs
# in backward, so nothing is held between the forward and the backward:
out = cake_gqa_nondet(q, k, v, cu_seqlens_q, cu_seqlens_k,
                      dkv_acc=lambda: (dk_dst, dv_dst, row_map))
# FP32 window gradients [T_k, H_kv, 256] stored into caller-owned tensors:
out = cake_gqa_nondet(q, k, v, cu_seqlens_q, cu_seqlens_k,
                      dkv_acc=lambda: (dk_win, dv_win))
```

`cake_gqa_nondet_train_forward` / `cake_gqa_nondet_train_backward` are the two halves
without autograd (`dkv_dtype=torch.float32` returns FP32 window gradients;
`dk_acc`, `dv_acc`, `row_map` add into destinations).  `max_seqlen_k` (a
host int, never read from the device) must be at least every segment's key
count, as FlashAttention's `max_seqlen_k`: a smaller value leaves key blocks
out (their gradients are not computed, and with a row map the rows they
would add are undefined).  It sizes the backward grids and the dS^T workspace
chunks; without it the bound is `T_k`, which is always correct but plans
more, mostly empty, key blocks: on a GB200, the backward of 12 segments with
30K query and 257K key tokens took 31.3 ms with the hint and 34.7 ms
without.  The backend package exposes the same entry points (`cake_gqa_nondet`,
`cake_gqa_nondet_train_forward`, `cake_gqa_nondet_train_backward`,
`CakeGQANondetWorkspace`, `CakeGQANondetFunction`).

## Kernels

| stage | kernel |
|---|---|
| `fwd` | O and the natural-log LSE `[H_q, T_q]`: a CTA pair per 128 query rows of two query heads of one key head, persistent tile tickets from a self-resetting counter |
| `bwd_pre` | `dPsum = sum(O * dO)` and `LSE * log2(e)`, two query heads per scratch head |
| `bwd_plan` | one CTA: the workspace tile numbering and the dQ work order from `cu_seqlens_q` / `cu_seqlens_k` (PyTorch ops instead past 16384 pairs of 128 query rows) |
| `bwd_main_fp32` / `bwd_main_bf16` | per workspace chunk: a CTA pair per 128 keys of one (segment, key head), looping the key head's query heads two at a time: S^T, dP^T, dK and dV (summed in FP32 in tensor memory, stored once as FP32 window gradients or as BF16), dS^T stored to the workspace |
| `bwd_dq` | per chunk: dQ = dS K over the workspace (persistent over the Q-tile pairs, the pairs with the most key blocks first): partial sums into the FP32 accumulator; a Q tile's last chunk writes `dQ = bf16(acc * softmax_scale)` |
| `bwd_fold` | row-map form only: the FP32 window rows added into their destination rows in ascending window order (a stable sort of the row map) |

Numerics: FP32 accumulation for the scores, `dP`, dK, dV, dQ and the row
sums; P and dS are rounded to BF16 only as MMA operands.  Every launch is a
plain stream-ordered launch: nothing cooperative, no PDL.  The backward builds
the tile tables (`bwd_plan`) and the row-map fold order (a stable sort of the
row map) on a high-priority side stream of the device, forked from the
current stream and joined back by CUDA events before their first read, so
they overlap `bwd_pre` and the main kernels; they hold the same integers as
on the current stream.

## Workspace and chunking

The main kernel stores dS^T as one 32 KiB BF16 tile (128 keys x 128 packed
rows) per (head pair, key block, Q tile of 64 rows).  A chunk covers `h` KV
heads and `n` key blocks and holds
`h * (H_q / H_kv / 2) * n * floor((T_q + 63 B) / 64)` tiles plus one zero
tile.  The workspace's `budget_bytes` (default 16 GiB,
`cake_gqa_nondet_train_workspace(device, budget_bytes=...)`) bounds a chunk:
all KV heads and key blocks in one chunk when they fit, else as many KV heads
as fit, else one KV head and the widest even key-block range that fits; at
least one KV head and two key blocks, even past the budget.  Each chunk runs
the main kernel and the dQ GEMM; each extra key-block chunk re-reads the FP32
dQ accumulator once and rewrites it for the Q tiles it does not finish (a Q
tile's last chunk writes its BF16 dQ instead).  The chunks come from host values only
(shapes, the segment count, `max_seqlen_k`, the budget); the tile numbering
inside a chunk is computed on the device from `cu_seqlens` (`bwd_plan`).  The budget moves
the rounding of dQ (its last bits, not its accuracy); `out`, `lse`, dK and dV
do not depend on it.

A caller that holds the boundaries on the host can pass them as
`cu_seqlens_q_host` / `cu_seqlens_k_host` (lists or 1-D CPU tensors, both or
neither; to `cake_gqa_nondet`, the backward and
`cake_gqa_nondet_train_workspace_bytes`).  The plan then counts each key
block's exact tiles (the numbering `bwd_plan` computes) instead of the bound
above, takes the key blocks from the longest key segment, and cuts them into
the widest even ranges that fit the budget; every range uses one stride, the
tile count of the fullest range.  The long layouts run in fewer chunks
(main-kernel and dQ launches with 24 query and 4 KV heads at the default
budget: 12 to 8 on a CP8 zigzag rank of 300k tokens with 16 segments, 16 to 8
on issue row 3, 8 to 2 on issue row 1) and with a dS^T workspace of the exact
size.  The kernels still read the device boundaries.  Before the main kernel a
device-side assert checks that every device segment's keys lie within the key
blocks and that every range holds at most its stride of tiles in the device
numbering.  Host boundaries within both bounds give the device boundaries'
gradients (only the chunks, and with them the dQ rounding, may differ); past
them the assert fires instead of leaving key blocks out or writing past a
chunk.  PyTorch reports a device-side assert at the next synchronization, and
the process's CUDA context is unusable after it.  The dQ bits depend on
whether the host boundaries are given (and on their values).

## Repeatability per output

Every sum of the forward and the backward has one fixed order; nothing adds
with atomics.  For one input, run to run:

| output | run to run | error vs FP64, rel-L2 / max abs err over max abs ref (`sm_100a`, `sm_103a`, `sm_107a`) |
|---|---|---|
| `out` (BF16) | repeats bit for bit | 2.35e-03 / 4.09e-03, 2.34e-03 / 3.91e-03, 2.34e-03 / 4.89e-03 |
| `lse` (FP32) | repeats bit for bit | 7.86e-08 / 3.17e-07, 7.85e-08 / 3.16e-07, 7.85e-08 / 3.03e-07 |
| `dq` (BF16) | repeats bit for bit | 2.46e-03 / 5.32e-03, 2.46e-03 / 5.55e-03, 2.46e-03 / 6.37e-03 |
| `dk` / `dv`, BF16 | repeats bit for bit | 2.45e-03 / 4.39e-03, 2.44e-03 / 4.22e-03, 2.45e-03 / 4.41e-03 |
| `dk` / `dv`, FP32 window | repeats bit for bit | 1.80e-03 / 3.09e-03, 1.79e-03 / 2.99e-03, 1.81e-03 / 3.04e-03 |
| row-map adds into `dk_acc` / `dv_acc` | repeats bit for bit: destination row `r` becomes `((dst + w_j1) + w_j2) + ...` over its window rows `j1 < j2 < ...` | 1.80e-03 / 3.09e-03, 1.79e-03 / 2.93e-03, 1.81e-03 / 3.04e-03 |

The errors are the worst of 5 runs over nine packed causal layouts (15K to
37K query and 25K to 433K key tokens, 4 to 16 segments) with random BF16
inputs, 24 query and 4 KV heads, against an FP64 reference from the same
inputs: `sm_100a` on a GB200, `sm_103a` on a B300, `sm_107a` on an R200.  The
`dk` / `dv` rows give the larger of the two; `dq` has the same error in every
form.

On one architecture the bits depend only on the shapes (`T_q`, `T_k`, `H_q`,
`H_kv`, the segment lengths), `max_seqlen_k`, the workspace budget and, for dQ,
the host boundaries (given or not): not on the stream, the workspace's earlier
calls or kernels running beside it.

## Determinism

This is the `nondet` package of the family; its names carry the mode word and
no entry takes a `deterministic=` keyword.  Its outputs nonetheless repeat
bit for bit (above): under `torch.use_deterministic_algorithms(True)`, strict
or with `warn_only=True`, nothing alerts or warns and the bits do not change.

## Checkpointing and memory

Autograd saves `(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)` through
`ctx.save_for_backward` only; `ctx` holds scalars, the host boundaries as
lists, the `dkv_acc` callable and
a weak reference to the workspace.  Under
`torch.utils.checkpoint(..., use_reentrant=False)` a forward retains nothing,
and the replay gives the gradients of the run without checkpointing bit for
bit.

A workspace keeps the forward's tile counter and the backward scratch, FP32
`[H_q / 2, rows_pad, 256]` plus two `[H_q / 2, rows_pad]` with
`rows_pad = round_up(2 T_q + 128 B, 128)`, grown to the largest shape seen
(`.nbytes`, `.release()`).  One backward call also allocates, from PyTorch's
caching allocator on the current stream, its dS^T chunk (32 KiB times the
chunk's tiles plus one) and, in the row-map form, two FP32 window gradients
`[T_k, H_kv, 256]`; it frees them when it returns, so nothing beyond the
scratch stays allocated between calls.  The per-call peak is the dS^T chunk
plus, in the row-map form, 2 KiB of windows per key and KV head.  The chunk
holds every tile when they fit in the budget and stays within the budget
otherwise.  With 24 query heads and 4 KV heads, 16461 query and 24693 key
tokens take 6.9 GB + 0.2 GB in one chunk; 32388 query and 275373 key tokens
take 17.1 GB + 2.3 GB in 16 chunks under the 16 GiB default.  A smaller
`budget_bytes` lowers the peak; each extra key-block chunk then re-reads the
FP32 dQ accumulator once and rewrites it for the Q tiles it does not finish.
`cake_gqa_nondet_train_workspace_bytes(T_q, H_q, B, total_k=, num_heads_kv=,
max_seqlen_k=, row_map=, budget_bytes=)` returns the scratch plus, when the
key sizes are given, the per-call buffers.  With torch's deterministic mode
and `torch.utils.deterministic.fill_uninitialized_memory` enabled, PyTorch
fills each per-call buffer when it is allocated: on a B300, a row-map backward
call with 7.1 GB, 9.8 GB or 19.4 GB of per-call buffers took 1.9 ms, 3.4 ms or
4.5 ms longer.  One workspace serves one stream at a time; without one, each
device stream gets its own default workspace.

## Limitations

- Head dimension 256, BF16 inputs, causal only (bottom-right aligned), no
  dropout, no sliding window.
- `H_q / H_kv` must be even.
- `dout` of the low-level backward must be contiguous (autograd makes it so).
- `sm_100a`, `sm_103a` and `sm_107a` only.
- Tiny calls can be slower than the previous version of this package: a
  training step (forward and backward, row-map form, 24 query and 4 KV heads)
  with up to about 2K query and 8K key tokens, 0.1 to 0.5 ms long, takes up
  to about 0.1 ms more for the extra launches (`bwd_plan`, the dQ GEMM, the
  fold and the row-map sort).  From 16K query and 25K key tokens up, a step
  takes at least 2 % less time on a GB200, 6 % less on a B300 and 5 % less
  on an R200.
