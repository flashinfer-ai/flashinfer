"""Run only native FP8 MLA under Compute Sanitizer, with a CPU FP32 oracle.

Copyright (c) 2026 by FlashInfer team.
Licensed under the Apache License, Version 2.0.
"""

import argparse
import json
import math
from pathlib import Path
import runpy

import torch

from flashinfer.mla import BatchMLAPagedAttentionWrapper


def check(q, qp, c, p, metadata, lengths, out, lse, scale):
    # Avoid cuBLAS initialization so sanitizer reports concern the target kernel.
    q, qp, c, p, out, lse = [x.cpu().float() for x in (q, qp, c, p, out, lse)]
    for i, n in enumerate(lengths):
        if n == 0:
            torch.testing.assert_close(out[i], torch.zeros_like(out[i]), rtol=0, atol=0)
            assert bool(torch.isneginf(lse[i]).all())
            continue
        start, end = metadata.kv_indptr[i : i + 2].cpu().tolist()
        pages = metadata.kv_indices[start:end].cpu().long()
        ck = c[pages].reshape(-1, 512)[:n]
        pk = p[pages].reshape(-1, 64)[:n]
        scores = (q[i] @ ck.T + qp[i] @ pk.T) * scale
        expected = scores.softmax(-1) @ ck
        torch.testing.assert_close(out[i], expected, rtol=0.03, atol=0.004)
        torch.testing.assert_close(
            lse[i], scores.logsumexp(-1) / math.log(2), rtol=0.001, atol=0.003
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (8, 9):
        parser.error("requires SM89")
    directory = Path(__file__).resolve().parent
    inputs = runpy.run_path(str(directory.parent / "bench_mla_sm89_fp8.py"))[
        "case_inputs"
    ]
    cases = json.loads((directory / "cases.json").read_text())["development"]
    result = {"status": "running", "oracle": "CPU FP32", "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        if case["id"] not in ("tail257", "h64_8k", "b7_h128_mixed", "empty_mix"):
            continue
        torch.manual_seed(731)
        lengths = case.get("lengths", [case.get("length", 1)] * case.get("batch", 1))
        q, qp, c8, p8, c16, p16, convert, kw = inputs(
            lengths, case["heads"], case["page"], case["scales"]
        )
        convert()
        w = BatchMLAPagedAttentionWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
            backend="fa2",
        )
        w.plan(**kw, kv_data_type=torch.float8_e4m3fn, scale_mode="kv-per-tensor")
        out = torch.empty_like(q)
        lse = torch.empty(q.shape[:2], device="cuda")

        def run():
            w.run(
                query=(q, qp),
                kv_cache=(c8, p8),
                out=out,
                lse=lse,
                return_lse=True,
                ckv_scale=case["scales"][0],
                kpe_scale=case["scales"][1],
            )

        run()
        check(q, qp, c16, p16, kw["metadata"], lengths, out, lse, kw["sm_scale"])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for _ in range(3):
            graph.replay()
        check(q, qp, c16, p16, kw["metadata"], lengths, out, lse, kw["sm_scale"])
        result["cases"].append({"id": case["id"], "reference_and_replay": "passed"})
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert len(result["cases"]) == 4
    result["status"] = "passed"
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
