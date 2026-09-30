# Cake chunked LM-head + loss training backend (SM100 / SM103)

Experimental backend for the chunked large-vocabulary LM-head projection with
a fused log-sum-exp / cross-entropy / policy-gradient loss and a
memory-bounded backward (GLM-5.2 geometry: hidden size 6144, vocabulary
154880, irregular token counts such as 16231 or 16172 per step, default token
chunk 4096).  Tracking issue: flashinfer-ai/flashinfer#5680.

Both the API and the backend are experimental: no compatibility guarantee,
SM100 (B200 / GB200) is the acceptance architecture and SM103 (B300 / GB300)
is compiled from the same sources.  The feature is JIT-only and does not
participate in automatic backend selection, autotuning or trace-apply.

## Public entry points (`flashinfer/chunked_lm_head.py`)

```python
from flashinfer.chunked_lm_head import chunked_lm_head_loss, chunked_lm_head_logprob

loss = chunked_lm_head_loss(X, W, labels, objective="ce", loss_div=loss_div, chunk_size=4096)
loss, logp = chunked_lm_head_loss(X, W, labels, objective="policy", infer_logp=infer_logp,
                                  loss_weights=loss_weights, return_logp=True)
logp = chunked_lm_head_logprob(X, W, labels, chunk_size=4096)   # differentiable [T] FP32
```

* `X [T, H]` BF16 hidden states, row-major with any leading stride (`H` a
  multiple of 256; a row pitch that is not a multiple of 16 bytes is copied
  to a contiguous tensor first); `W [V, H]` BF16 contiguous output weight
  (`V` a multiple of 256); `labels [T]` int64 with `-100` marking an ignored
  row (zero loss, zero gradient, `logp = 0`).  A registered program is built
  for one `(H, V)` geometry, which its registry record pins; other geometries
  are rejected at validation.
* Per row `logp_t = z[t, y_t] - logsumexp_v z[t, v]` with `z = X @ W^T`
  computed as a BF16 GEMM (BF16 output of an FP32 accumulation) and promoted
  to FP32 for the max / log-sum-exp / loss arithmetic.
* `objective="ce"`: `loss = -sum(logp[valid]) / loss_div` with the
  caller-supplied positive divisor `loss_div` (never replaced by a local
  token mean).  `objective="policy"`:
  `loss = -sum(loss_weights * min(exp(logp - infer_logp), 2))` over valid rows
  with FP32 `infer_logp [T]` and signed FP32 `loss_weights [T]` (already
  masked and normalized; 0 on ignored rows); the logit-gradient scale is
  `-loss_weights_t * ratio_t` when `ratio_t <= 2` and `0` above the clipping
  boundary.
* Backward: `dz = d_t * (1[v = y_t] - softmax(z_t))` in BF16 (the `dlogits`
  boundary), `dX = dz @ W` and `dW = dz^T @ X` accumulated in FP32 across the
  chunks in a fixed order (no atomics) and cast once at the output: `dX` BF16,
  `dW` BF16 by default or FP32 with `grad_weight_dtype=torch.float32` (the
  FP32 form is served by the explicit `cake_backend.forward_loss` /
  `backward_loss` pair: the autograd entry casts every gradient to its leaf's
  dtype and therefore requires `grad_weight_dtype == W.dtype`).
  Gradients are produced only for inputs that require them (a frozen `X` or
  `W` skips its GEMM); the upstream scalar gradient is applied once, in the
  cast.  The saved FP32 accumulators are re-scaled, never mutated, so a
  retained graph may run the backward repeatedly.
* `chunked_lm_head_logprob` returns the differentiable `logp [T]` for
  arbitrary downstream losses; its forward saves only the FP32 row statistics
  (`lse` and the selected logit) and the backward recomputes each chunk's
  logits from the saved inputs (four GEMMs per chunk).
* Memory rule: no logits, probability or `dlogits` buffer ever spans more than
  `chunk_size` tokens; a batch smaller than `chunk_size` is one chunk, a tail
  `T % chunk_size` is neither dropped nor padded, and `T == 0` returns loss 0,
  an empty `logp` and zero gradients without binding or launching.
  `cake_backend.memory_report` reports the peak temporary bytes separately
  from the weights, the outputs and the FP32 accumulators.
* `deterministic=True` only: fixed sequential chunk order, bitwise
  reproducible `loss`, `logp`, `dX` and `dW` run to run.

Explicit forward / backward entry points without autograd (`forward_loss`,
`backward_loss`, `forward_logprob`, `backward_logprob`), a prepared
allocation-free runner (`prepare_lm_head_loss`, CUDA-graph capturable) and the
workspace sizing helper (`lm_head_loss_workspace_size`) live in
`cake_backend.py`.  The eager entry points (and the autograd wrappers behind
the public API) validate and bind once per input binding -- `(data_ptr, shape,
stride, dtype)` of every input plus the options -- and launch later calls from
the remembered argument plans with freshly allocated outputs and per-call
scratch (`cake_backend.BINDING_CACHE`).  A remembered binding pins no caller
tensor and holds no problem-sized scratch: the chunk workspace, the FP32
accumulators and the outputs come from the caching allocator on every call.
The cache keeps up to 64 bindings
(`FLASHINFER_CAKE_LM_HEAD_LOSS_BINDING_CACHE_CAPACITY` sets the capacity) and
evicts the least recently used one; `FLASHINFER_CAKE_LM_HEAD_LOSS_BINDING_CACHE=0`
disables it.  `backend="reference"` on the `cake_backend` entry points runs
the same chunk loop with PyTorch operators at the same rounding boundaries on
any device (the host-layer tests use it); the public API accepts
`backend="cake"` only.

## Kernel structure of one token chunk

Per chunk of `rows_c <= chunk_size` rows the host launches, in this order:

* `gemm_logits`: `z_c = bf16(X_c @ W^T)` (2-CTA tensor-core GEMM, 128-row
  tiles, 256 vocabulary columns per accumulator) together with the
  per-(row, 256-column tile) online `(max, sum-exp)` partials of the
  BF16-rounded logits, written from the epilogue.  The log-probability
  backward's recompute uses `gemm_logits_nostats`, the same GEMM without the
  statistics.
* `row_finalize`: one warp per row merges the partials into `lse`, gathers
  the selected logit, writes `logp` (0 on ignored rows) and, per objective,
  the per-row logit-gradient scale `d_t` and the loss term; the
  log-probability backward takes `d_t` from the incoming `dlogp` instead.
* `loss_reduce`: one CTA sums the chunk's loss terms in a fixed order into
  the loss accumulator; the last chunk writes the finished loss.
* `row_grad`: `dz_c = d_t * (1[v = y_t] - exp(z - lse_t))` in BF16, in place
  over `z_c`; ignored rows become zero.
* `gemm_dx`: `dX_acc[rows] = fp32(dz_c @ W)`.  A record may also register
  `gemm_dx_s2` / `gemm_dx_s3` / `gemm_dx_s4`, the same GEMM as 2 / 3 / 4
  K-slice work items per output tile (the persistent grid fills its last
  wave; a record registers a contiguous prefix of them): slice 0 writes
  `dX_acc`, slices `>= 1` write FP32 workspace slabs (`dx_ws`, temporary
  bucket) that the host adds into `dX_acc` in fixed slab order (one RN add per
  element per slab, no atomics).  The slice count of a chunk follows from its
  row count and the SM count (`cake_backend.recommended_k_slices`), so it is
  deterministic in the shapes.
* `gemm_dw_acc`: `dW_acc = fp32(dz_c^T @ X_c)` on the first chunk,
  `dW_acc += ...` afterwards (the chunk order is the reduction order).
* `scale_cast_bf16` / `scale_cast_f32`: `out = g * acc` over the flat FP32
  accumulators -- the single output cast of `dX` and `dW` in the backward,
  with the upstream gradient `g` read from a device scalar.

Grid rules live in the registry record (e.g. `rows_c/8` CTAs for
`row_finalize`, a persistent CTA-pair grid bounded by the SM count for the
GEMMs, `V/8/2048 x rows_c` for `row_grad`) and are evaluated by the host from
the chunk scalars; the GEMM operands are described by TMA descriptors the
host prepares in a per-runner descriptor workspace.

## Layout of this package

* `cake_jit.py` -- `MODULES` registry (one record per architecture, filled by
  the generated-program export), stage names and the JIT specs.
* `cake_backend.py` -- validation, chunk plans, workspace layout, memory
  accounting, argument-plan binding, the prepared runner, the torch reference
  path, the autograd `Function`s and the eager entry points.
* `csrc/cake_lm_head_loss/<arch>/` -- generated kernel and binding
  translation units (`.clang-format` disables formatting: the sources are
  identity-checked by the registry's closure digests).

## Status

The registry holds one record per architecture (`sm_100a`, `sm_103a`) with
the nine stages above (`abi = lm_head_loss_v1`), exported from the kernel
snapshot named in the pull request, pinned to hidden size 6144 and vocabulary
154880.

Tests: `tests/experimental/test_cake_lm_head_loss.py` (the host-layer tests
run on any device through the reference path; the device tests skip without a
registered program or a compute capability 10.0 / 10.3 device).  Benchmark:
`benchmarks/bench_cake_lm_head_loss.py`.
