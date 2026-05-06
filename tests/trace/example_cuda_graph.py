"""
fi_trace + CUDA graph example.

Demonstrates that @flashinfer_api(trace=...) auto-dump is compatible with
`torch.cuda.graph` capture:

  * The schema extraction path reads only CPU-side tensor metadata (shape,
    dtype) and writes a JSON file on the host thread — no CUDA stream ops,
    so nothing gets baked into the captured graph.
  * On graph *replay*, Python code does not run at all, so auto-dump cannot
    fire again. The _DUMPED_NAMES dedup in flashinfer/trace/template.py
    already prevents re-writes even when Python does run.

Run:
    python tests/trace/example_cuda_graph.py

Produces one file in ./fi_trace_out_cudagraph/:
    gqa_paged_decode_h32_kv8_d128_ps16.json
"""

import os
from pathlib import Path

# Must be set before any flashinfer import: template.py reads these at import time.
SAVE_DIR = Path(__file__).parent / "fi_trace_out_cudagraph"
os.environ.setdefault("FLASHINFER_TRACE_DUMP_DIR", str(SAVE_DIR))
os.environ.setdefault("FLASHINFER_TRACE_DUMP", "1")

import torch

from flashinfer.decode import CUDAGraphBatchDecodeWithPagedKVCacheWrapper


def main() -> None:
    device = "cuda"
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this example.")

    # Llama-3.1-8B paged decode: 32 qo heads / 8 kv heads / head_dim=128, 32 seqs
    batch_size, num_qo, num_kv, head_dim, page_size = 32, 32, 8, 128, 16
    num_pages_per_seq = 8
    total_pages = batch_size * num_pages_per_seq
    workspace = 128 * 1024 * 1024  # 128 MB

    # Static buffers the wrapper reuses across captures.
    kv_indptr_buf = torch.empty(batch_size + 1, dtype=torch.int32, device=device)
    kv_indices_buf = torch.empty(total_pages, dtype=torch.int32, device=device)
    kv_last_buf = torch.empty(batch_size, dtype=torch.int32, device=device)
    ws = torch.empty(workspace, dtype=torch.uint8, device=device)

    wrapper = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
        ws, kv_indptr_buf, kv_indices_buf, kv_last_buf, "NHD"
    )

    # Fill the static buffers with the layout we will replay against.
    kv_indptr_buf.copy_(
        torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        * num_pages_per_seq
    )
    kv_indices_buf.copy_(torch.arange(total_pages, dtype=torch.int32, device=device))
    kv_last_buf.copy_(
        torch.full((batch_size,), page_size, dtype=torch.int32, device=device)
    )

    # Plan runs on the CPU — never captured.
    wrapper.plan(
        kv_indptr_buf,
        kv_indices_buf,
        kv_last_buf,
        num_qo,
        num_kv,
        head_dim,
        page_size,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    q = torch.randn(batch_size, num_qo, head_dim, dtype=torch.bfloat16, device=device)
    kc = torch.randn(
        total_pages, page_size, num_kv, head_dim, dtype=torch.bfloat16, device=device
    )
    vc = torch.randn(
        total_pages, page_size, num_kv, head_dim, dtype=torch.bfloat16, device=device
    )

    expected = SAVE_DIR / "gqa_paged_decode_h32_kv8_d128_ps16.json"
    if expected.exists():
        expected.unlink()  # Start clean so we can observe the first dump.

    # Warmup on a side stream so the first captured iteration is well-behaved.
    # The first wrapper.run() triggers auto-dump on the host thread (schema
    # extraction is CPU-only: .shape / .dtype / json.dumps). Subsequent calls
    # hit the _DUMPED_NAMES dedup and skip file I/O.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _ = wrapper.run(q, (kc, vc))
    torch.cuda.current_stream().wait_stream(s)

    assert expected.exists(), (
        f"Expected trace JSON at {expected} to be written on the first call."
    )
    size_after_warmup = expected.stat().st_size
    mtime_after_warmup = expected.stat().st_mtime_ns
    print(f"[warmup]  wrote {expected.name} ({size_after_warmup} bytes)")

    # Capture: the @flashinfer_api(trace=...) wrapper's Python code still
    # runs once inside the capture block, but dedup skips the write. Kernel
    # launches are captured into the graph; host-side file I/O is never a
    # captured CUDA op, so it cannot corrupt the graph even when it does fire.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_captured = wrapper.run(q, (kc, vc))

    assert expected.stat().st_mtime_ns == mtime_after_warmup, (
        "Trace file was rewritten during capture — dedup failed."
    )
    print("[capture] graph captured; trace file untouched (dedup skipped re-write)")

    # Replay: Python doesn't run at all, so auto-dump definitely cannot fire.
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    assert expected.stat().st_mtime_ns == mtime_after_warmup, (
        "Trace file was rewritten during replay — auto-dump is not replay-idempotent."
    )
    print("[replay]  5 replays completed; trace file still untouched")

    # Correctness: eager call should match the graph output (same inputs,
    # same plan). Use the bound method's own fi_trace to confirm the schema
    # was generated even without file dump.
    eager_out = wrapper.run(q, (kc, vc))
    torch.testing.assert_close(out_captured, eager_out, rtol=1e-3, atol=1e-3)
    print("[verify]  captured output matches eager reference")

    # fi_trace() is still directly callable on the bound method for ad-hoc use.
    # Takes kwargs; positional tensor args are not supported.
    schema = wrapper.run.fi_trace(q=q, paged_kv_cache=(kc, vc))
    print(
        f"[fi_trace] {schema['name']} op_type={schema['op_type']} axes={schema['axes']}"
    )


if __name__ == "__main__":
    main()
