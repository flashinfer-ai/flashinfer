# ResServe page32 attention (experimental)

Owner: @ShawRong. Tracking issue: flashinfer-ai/flashinfer#6886.
Ported from [ResServe a46e2a5](https://github.com/ShawRong/resserve/commit/a46e2a5).
Original copyright and license notices are retained.

```python
from flashinfer.resserve import paged_decode_residue
```

Explicit opt-in, SM100, BF16 original Q/output, NVFP4 page32 KV, D128/D256,
GQA4/6/8. K scales are linear within a page; V scales use four-token
interleave. K/V data may use HND or NHD storage without copying. Page-strided
cache views and padded block-table rows are accepted.
The caller owns KV writing, layout, reconstruction scales and metadata.
No cache reordering or host synchronization occurs in decode.

The existing ResServe kernel and static split/reduction policy are retained.
The FlashInfer port replaces the external Q-quantization extension with JIT/TVM-FFI
and uses invocation-local scratch rather than a shared mutable buffer cache.
The API and backend support BF16 output only. See the entry point for exact
preconditions.
Warm up outside CUDA Graph capture. Requires CUTLASS DSL 4.8 with CUDA 13.4 PTX.

```sh
python examples/resserve/resattn.py
pytest -q tests/experimental/test_resserve_resattn.py
```

The example checks uniform attention. Tests cover decoded numerical references,
GQA/D variants, page/tile boundaries, strided cache views, output placement and
Graph replay with changing Q/KV, page IDs and lengths. They do not establish
full-model quality or end-to-end serving performance.
