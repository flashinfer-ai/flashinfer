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
- JIT-only: no AOT registration or precompiled kernels. Source-enabled wheels
  include the CUDA sources and headers needed for JIT compilation. Release
  wheels built with `FLASHINFER_WHEEL_EXCLUDE_EXPERIMENTAL_SOURCES=1` omit
  these sources; use a source checkout or source-enabled wheel.
- Tests: `tests/experimental/test_qsa_topk_indexer.py` (exact PyTorch
  reference, validated on SM107).

Owner: [Brandon Zhang (@brandonfzhang)](https://github.com/brandonfzhang).
Use-case tracking: [QSA indexer coverage #8086](https://github.com/flashinfer-ai/flashinfer/issues/8086).
Study and implementation tracking: [Rubin kernel study #9593](https://github.com/flashinfer-ai/flashinfer/issues/9593).

Graduation requires settling the packed request-local index/count contract
with its serving consumer, validating that integration on SM107, and
completing the stable API/build checklist. This follows the experimental
policy's four-week review target after introduction; it is not a promise
of a stable release or automatic backend admission.

## Runnable example

Run on SM107 with a CUDA toolkit supporting that architecture. Calling the
API explicitly opts into the experimental feature and warns once.

```python
import torch
from flashinfer import qsa_topk_indexer

# One request of 2600 tokens, with 650 compressed blocks in 11 pages.
assert torch.cuda.get_device_capability() == (10, 7)
qi = torch.zeros((2600, 4, 128), device="cuda").to(torch.float8_e4m3fn)
ck = torch.ones((11, 64, 1, 128), device="cuda").to(torch.float8_e4m3fn)
cu = torch.tensor([0, 2600], dtype=torch.int32, device="cuda")
base = torch.tensor([0], dtype=torch.int32, device="cuda")
packed = qsa_topk_indexer(qi, ck, cu, base)
# All scores tie, so the last row retains the first 512 four-token blocks.
assert packed.shape == (2600, 2052)
assert packed[-1, -1].item() == 2048
assert torch.equal(packed[-1, :2048], torch.arange(2048, device="cuda"))
assert (packed[-1, 2048:-1] == -1).all()
```

The result contains request-local token IDs and a trailing count. It is not
an input to qsa_indexed_forward, which uses a different index layout and
device contract. A consumer must implement this packed SM107 layout.
