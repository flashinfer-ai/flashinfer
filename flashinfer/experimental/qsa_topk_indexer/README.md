# Experimental QSA stage-1 token top-k block indexer (SM107)

`flashinfer.qsa_topk_indexer` scores every visible compressed key block for
each packed query token (`sum_h relu(qi . ck)` over the 4 index heads),
selects the top 512 blocks with a deterministic tie-break (the lower block
index wins), expands the selection to token ids plus the partial causal tail,
and writes the packed int32 `[T, 2052]` layout the SM107 QSA stage-2
sparse-attention path consumes (ascending request-local ids, `-1` padding,
valid count in the last column).

The **API is experimental** (`@flashinfer_experimental_api` on
`flashinfer.qsa.qsa_topk_indexer`; calling it is the opt-in and warns once)
and the **backend is experimental**: this package holds the kernels, the JIT
spec (`jit.py`), the loader cache and custom-op registration (`backend.py`),
and the backend support check. The core entry point in `flashinfer/qsa.py`
keeps only the public signature, shared validation, backend selection and a
deferred handoff to this package.

- Supported device: SM107 only. `"specialized"` is the only implementation of
  this contract; a call it cannot serve raises `NotImplementedError` with the
  reason recorded in `flashinfer.qsa.qsa_topk_indexer_fallback_reasons` — it
  never silently falls back.
- Kill switch: `FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1` guarantees the
  specialized kernel is never dispatched (the call then raises, since no
  composable fallback exists).
- The kernel is bitwise deterministic, allocation-free at run time and
  CUDA-graph safe (warm up once before capture to build the JIT module).
- JIT-only: no AOT registration, so it never ships in pre-built packages.
- Tests: `tests/experimental/test_qsa_topk_indexer.py` (exact PyTorch
  reference, validated on SM107).

The experimental lifecycle owner and the tracking issue are linked from the
pull request that adds the package.
