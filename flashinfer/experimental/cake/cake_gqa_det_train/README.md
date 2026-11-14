# Cake deterministic packed-varlen GQA training attention backend (head dim 256, SM100 / SM103 / SM107)

Backend of `flashinfer.cake_gqa_det_train`: the forward and first-order backward of causal GQA
attention over packed variable-length segments, head dimension 256: BF16
`q [T_q, H_q, 256]`, `k`, `v [T_k, H_kv, 256]` (`H_q / H_kv` even), segments
packed by `cu_seqlens_q` / `cu_seqlens_k` (int32 on the device, never read on
the host).  Query row `i` of a segment with `q_len` rows and `k_len` keys sees
the keys `j <= i + k_len - q_len` (FlashAttention's bottom-right alignment);
a row that sees no key gets `out = 0`, `lse = -inf` and zero gradients.
`T_q`, `T_k`, the segment count and every length are run-time values: no
shape list, no recompilation, no device-to-host read, no device-wide
synchronization.  Every output repeats bit for bit run to run.

The facade `flashinfer.cake_gqa_det_train` is a stable API; this backend
package is experimental (no compatibility guarantee for its internal names).
SM100 (B200 / GB200), SM103 (B300 / GB300) and SM107 (R200) are the
supported architectures; another device raises `RuntimeError`.  The feature does not
participate in automatic backend selection, autotuning or trace-apply.  Its
libraries are loaded prebuilt from `aot/` when a builder compiled them and
JIT-built by FlashInfer from the sources and flags `build.json` records
otherwise.  The package reads no environment variable.  The interface matches
the non-deterministic package `flashinfer.cake_gqa_nondet_train`.

## Opt-in

```python
from flashinfer.cake_gqa_det_train import cake_gqa_det, cake_gqa_det_train_workspace

ws = cake_gqa_det_train_workspace(q.device)      # optional; reused across calls and shapes
# ws = cake_gqa_det_train_workspace(q.device, budget_bytes=2 << 30)  # bound the dS^T workspace (default 16 GiB)
out = cake_gqa_det(
    q, k, v, cu_seqlens_q, cu_seqlens_k,
    softmax_scale=0.0625,                        # default 256 ** -0.5
    max_seqlen_k=max_k,                          # optional host bound of the longest key segment
    workspace=ws,
)
out.backward(dout)                               # BF16 dq, dk, dv

# FP32 dK / dV added into caller-owned FP32 destinations [N_dst, H_kv, 256]
# through an int32 row map [T_k] (repeated rows accumulate in window order);
# the callable runs in backward, so nothing is held between the forward and the backward:
out = cake_gqa_det(q, k, v, cu_seqlens_q, cu_seqlens_k,
                   dkv_acc=lambda: (dk_dst, dv_dst, row_map))
# FP32 window gradients [T_k, H_kv, 256] stored into caller-owned tensors:
out = cake_gqa_det(q, k, v, cu_seqlens_q, cu_seqlens_k,
                   dkv_acc=lambda: (dk_win, dv_win))
```

`cake_gqa_det_train_forward` / `cake_gqa_det_train_backward` are the two halves
without autograd (`dkv_dtype=torch.float32` returns FP32 window gradients;
`dk_acc`, `dv_acc`, `row_map` add into destinations).  `max_seqlen_k` (a
host int, never read from the device) must be at least every segment's key
count, as FlashAttention's `max_seqlen_k`: a smaller value leaves key blocks
out.  It sizes the backward grids and the dS^T workspace; without it the bound
is `T_k`, which is always correct but plans more (mostly empty) key blocks.
The backend package exposes the same entry points (`cake_gqa_det`,
`cake_gqa_det_train_forward`, `cake_gqa_det_train_backward`,
`CakeGQADetWorkspace`, `CakeGQADetFunction`).

## Determinism

Every sum of the forward and the backward has one fixed order; nothing adds
with atomics.  Run to run, for one input:

| output | run to run |
|---|---|
| `out` (BF16) | repeats bit for bit |
| `lse` (FP32) | repeats bit for bit |
| `dq` (BF16) | repeats bit for bit |
| `dk` / `dv`, BF16 | repeats bit for bit |
| `dk` / `dv`, FP32 window | repeats bit for bit |
| row-map adds into `dk_acc` / `dv_acc` | repeats bit for bit: destination row `r` becomes `((dst + w_j1) + w_j2) + ...` over its window rows `j1 < j2 < ...` |

The bits depend only on the shapes (`T_q`, `T_k`, `H_q`, `H_kv`, the segment
lengths), `max_seqlen_k`, the workspace budget and, for `dq`, the host
boundaries (given or not): not on the stream, the workspace's earlier calls or
kernels running beside it.  A different budget or other host boundaries can
move the last bits of `dq` (not its accuracy); `out`, `lse`, dK and dV depend
on neither.  Under `torch.use_deterministic_algorithms(True)`, strict or
with `warn_only=True`, nothing alerts or warns and the bits do not change.
This is the `det` package of the family; its names carry the mode word and no
entry takes a `deterministic=` keyword.

## Kernels

| stage | kernel |
|---|---|
| `fwd` | O and the natural-log LSE `[H_q, T_q]`: a CTA pair per 128 query rows of two query heads of one key head, persistent tile tickets from a self-resetting counter |
| `bwd_pre` | `dPsum = sum(O * dO)` and `LSE * log2(e)`, two query heads per scratch head |
| `bwd_plan` | one CTA: the workspace tile numbering and the dQ work order from `cu_seqlens_q` / `cu_seqlens_k` (PyTorch ops instead past 1024 pairs of 128 query rows) |
| `bwd_main_fp32` / `bwd_main_bf16` | per workspace chunk: a CTA pair per 128 keys of one (segment, key head): S^T, dP^T, dK and dV, dK and dV summed in tensor memory over every query head of the key head and stored once (FP32 window or BF16), dS^T stored to the workspace |
| `bwd_dq` | per chunk: dQ = dS K over the workspace (persistent over the Q-tile pairs): partial sums into the FP32 accumulator; a Q tile's last chunk writes `dQ = bf16(acc * softmax_scale)` |
| `bwd_fold` | row-map form only: the FP32 window rows added into their destination rows in ascending window order (a stable sort of the row map) |

Numerics: FP32 accumulation for the scores, `dP`, dK, dV, dQ and the row
sums; P and dS are rounded to BF16 only as MMA operands.  Every launch is a
plain stream-ordered launch: nothing cooperative, no PDL.  The backward builds
the tile tables (`bwd_plan`) and the row-map fold order (a stable sort of the
row map) on a high-priority side stream of the device, forked from the
current stream and joined back by CUDA events before their first read, so
they overlap `bwd_pre` and the main kernels; they hold the same integers as
on the current stream.

## Workspace budget and memory

The main kernel stores dS^T as one 32 KiB BF16 tile (128 keys x 128 packed
rows) per (head pair, key block, Q tile of 64 rows).  A chunk covers `h` KV
heads and `n` key blocks and holds
`h * (H_q / H_kv / 2) * n * floor((T_q + 63 B) / 64)` tiles plus one zero
tile.  The workspace's `budget_bytes` (default 16 GiB) bounds a chunk: all KV
heads and key blocks in one chunk when they fit, else as many KV heads as fit,
else one KV head and the widest even key-block range that fits; at least one
KV head and two key blocks, even past the budget.  Each extra key-block chunk
re-reads the FP32 dQ accumulator once and rewrites it for the Q tiles it does
not finish (a Q tile's last chunk writes its BF16 dQ instead).

A caller that holds the boundaries on the host can pass them as
`cu_seqlens_q_host` / `cu_seqlens_k_host` (lists or 1-D CPU tensors, both or
neither; to `cake_gqa_det`, the backward and
`cake_gqa_det_train_workspace_bytes`).  The plan then counts each key
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

Beside the dS^T chunk, the backward scratch is FP32 `[H_q / 2, rows_pad, 256]`
plus two `[H_q / 2, rows_pad]` with `rows_pad = round_up(2 T_q + 128 B, 128)`,
and the row-map form adds two FP32 window gradients `[T_k, H_kv, 256]`.
`cake_gqa_det_train_workspace_bytes(T_q, H_q, B, total_k=, num_heads_kv=,
max_seqlen_k=, row_map=, budget_bytes=, cu_seqlens_q_host=,
cu_seqlens_k_host=)` returns the total (exact with the host boundaries).  A workspace keeps
only the scratch, at the largest size seen (`.nbytes`, `.release()`); the dS^T
chunk and the window gradients come from PyTorch's caching allocator for each
backward call and are freed when it returns; they are never filled under
`torch.use_deterministic_algorithms(True)` (the kernels write every element
they read).  One workspace serves one stream at a time; without one, each
device stream gets its own default workspace.  Autograd keeps only a weak
reference to the workspace: a backward whose workspace was dropped after the
forward uses a new one with the same `budget_bytes`, and the same bits.

## Checkpointing

Autograd saves `(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)` through
`ctx.save_for_backward` only; `ctx` holds scalars, the host boundaries as
lists, the `dkv_acc` callable and
a weak reference to the workspace.  Under
`torch.utils.checkpoint(..., use_reentrant=False)` a forward retains nothing,
and the replay gives the gradients of the run without checkpointing bit for
bit.

## CUDA graphs

The forward and the backward can be captured in a CUDA graph: they read
nothing on the host and never synchronize.  Give each graph its own explicit
workspace, used by an eager call of the captured shape before the capture,
and keep that workspace alive and away from other shapes while the graph
exists: the graph records the addresses of its scratch and tile counter, and
a larger shape would move the scratch.  The dS^T chunk, the window gradients
and the plan tables come from the graph's memory pool.  Graphs that may
replay at the same time need separate workspaces.

## Build and check

```bash
python scripts/check_cake_build_manifests.py --package flashinfer.experimental.cake.cake_gqa_det_train
pytest tests/experimental/test_cake_gqa_det_train.py \
       tests/experimental/test_cake_gqa_det_train_api.py \
       tests/experimental/test_cake_gqa_det_train_names.py
python benchmarks/bench_cake_gqa_det_train.py --layouts issue1,issue2,issue3 --budget-gib 8
```

## Limitations

- Head dimension 256, BF16 inputs, causal only (bottom-right aligned), no
  dropout, no sliding window.
- `H_q / H_kv` must be even.
- `dout` of the low-level backward must be contiguous (autograd makes it so).
- `sm_100a`, `sm_103a` and `sm_107a` only.
