# Copyright (c) 2026 Francesco Parisio
# SPDX-License-Identifier: Apache-2.0
"""Weight-free GLM53 NoPE precision screening on SM120/121.

Run from a CUDA development environment with FlashInfer installed:
    python benchmarks/repro_glm53_sm120_precision.py --route wrapper
    python benchmarks/repro_glm53_sm120_precision.py --route fp8-sg

The wrapper run measures ordinary dispatch. The fp8-sg control explicitly
resolves the existing FP8 SG kernel using FlashInfer's internal execution API,
so a future BF16-QK SG comparison can hold dispatch constant. It does not
modify FlashInfer sources or policy. Each route tests 13 cases at both table
widths. A failed screening limit exits 1 after collecting every finite result;
non-finite output or a CUDA exception aborts immediately. These thresholds are
local screening criteria, not an upstream accuracy guarantee.

Cache generation uses only PyTorch. The oracle reads the actual packed bytes
and retains FP32 dequantization, matmul and softmax. Neither model weights nor
SGLang are required. Kernel graph replay is not a model-level graph test.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

PAGE_SIZE = 64
CACHE_BYTES = 528
HEADS = 16
TOPK = 2051
SCALE = 1 / math.sqrt(256)
MAX_RELATIVE_RMS = 0.02
MAX_ABSOLUTE_ERROR = 0.03
SEED = 20260918


def reference(q, cache, indices, lengths):
    """Compute FP32 attention from packed cache bytes and length-masked indices.

    Negative indices are holes; rows with no live slots return zeros. Query
    values are not requantized, so this remains independent of kernel QK mode.
    """
    raw = cache.view(torch.uint8).reshape(-1, CACHE_BYTES)
    values = raw[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    scales = raw[:, 512:528].contiguous().view(torch.float32)
    dequantized = (values.reshape(-1, 4, 128) * scales.unsqueeze(-1)).reshape(-1, 512)
    output = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    for row in range(q.shape[0]):
        slots = indices[row, : int(lengths[row])]
        slots = slots[slots >= 0].long()
        if slots.numel() == 0:
            continue
        kv = dequantized[slots]
        probability = torch.softmax((q[row].float() @ kv.T) * SCALE, dim=-1)
        output[row] = probability @ kv
    return output


def error_metrics(output, expected):
    """Return case/row errors and the worst coordinate, rejecting nonfinite data."""
    if not torch.isfinite(output).all() or not torch.isfinite(expected).all():
        raise AssertionError("Non-finite output or reference; aborting diagnostic")
    difference = output.float() - expected.float()
    flat = difference.abs().reshape(-1)
    worst = int(flat.argmax())
    row = worst // (output.shape[1] * output.shape[2])
    head = (worst // output.shape[2]) % output.shape[1]
    channel = worst % output.shape[2]
    return {
        "relative_rms": (
            difference.square().mean().sqrt()
            / expected.float().square().mean().sqrt().clamp_min(1e-6)
        ).item(),
        "max_absolute": flat[worst].item(),
        "row_max_absolute": difference.abs().flatten(1).amax(dim=1).tolist(),
        "row_relative_rms": (
            difference.square().flatten(1).mean(dim=1).sqrt()
            / expected.float().square().flatten(1).mean(dim=1).sqrt().clamp_min(1e-6)
        ).tolist(),
        "worst_coordinate": [row, head, channel],
        "worst_actual": output[row, head, channel].item(),
        "worst_expected": expected[row, head, channel].item(),
    }


def quantized_query(q, *, power_of_two):
    """Round Q through E4M3 for diagnostics, optionally rounding scales to powers of two."""
    # Diagnostic approximation to FlashInfer's query quantization, not a
    # replacement correctness oracle. Decode/MG round scales up to powers
    # of two; swapAB retains arbitrary FP32 scales for GLM's QK operation.
    blocks = q.float().reshape(*q.shape[:-1], 4, 128)
    scales = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4) / 448.0
    if power_of_two:
        scales = torch.exp2(torch.ceil(torch.log2(scales)))
    fp8 = (blocks / scales).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (fp8.float() * scales).reshape(q.shape)


def pack_cache(values):
    """Pack four arbitrary FP32 scales after 512 E4M3 bytes per token.

    Same quantization convention as FlashInfer's quantize_kv_glm53_nope test
    helper, using the compact 528-byte layout introduced by PR #5075.
    """
    blocks = values.float().reshape(-1, 4, 128)
    scales = blocks.abs().amax(dim=-1).clamp_min(1e-4) / 448.0
    encoded = (blocks / scales.unsqueeze(-1)).clamp(-448, 448)
    encoded = encoded.to(torch.float8_e4m3fn)
    packed = torch.empty(
        len(blocks), CACHE_BYTES, dtype=torch.uint8, device=values.device
    )
    packed[:, :512] = encoded.reshape(-1, 512).view(torch.uint8)
    packed[:, 512:] = scales.contiguous().view(torch.uint8).reshape(-1, 16)
    return packed


def make_cache(device):
    """Return a randomized compact cache and live slots, with NaN-poisoned slot zero."""
    gains = torch.tensor([0.37, 0.71, 1.13, 1.91], device=device)
    values = torch.randn(4095, 1, 4, 128, device=device)
    values = (values * gains[None, None, :, None]).reshape(-1, 512).to(torch.bfloat16)
    packed = pack_cache(values)
    locations = torch.randperm(4095, device=device).to(torch.int64) + 1
    cache = torch.empty(4096, CACHE_BYTES, dtype=torch.uint8, device=device)
    cache[locations] = packed
    cache[0].fill_(0x7F)  # NaN payload and scales: masked reads must never use it.
    return cache.reshape(-1, PAGE_SIZE, CACHE_BYTES), locations


def make_inputs(rows, locations, width, *, all_masked=False):
    """Build BF16 queries and padded indices with poison just beyond each row's length."""
    device = locations.device
    q = torch.randn(rows, HEADS, 512, device=device, dtype=torch.bfloat16) * 0.7
    lengths = torch.tensor(
        [
            0 if all_masked else (1, 63, 64, 65, 2047, 2048, 2049, 2050, 2051)[r % 9]
            for r in range(rows)
        ],
        dtype=torch.int32,
        device=device,
    )
    indices = torch.full((rows, width), -1, dtype=torch.int32, device=device)
    for row in range(rows):
        length = int(lengths[row])
        indices[row, :length] = locations[:length].to(torch.int32)
        indices[row, length] = 0  # Positive poisoned slot, masked only by length.
    return q, indices, lengths


def make_runner(cache, route, precision):
    """Return attention and plan-inspection closures for wrapper or direct FP8 SG."""
    from flashinfer.mla import SparseMLASm120Wrapper

    if route == "wrapper":
        wrapper = SparseMLASm120Wrapper(
            kv_scale_format="arbitrary_fp32",
            compute_precision=precision,
        )

        def run(q, indices, lengths, output):
            """Write attention output through the wrapper using per-row lengths."""
            wrapper.run(q, cache, indices, output, SCALE, topk_length=lengths)

        def plans():
            """Report descriptors prepared by the wrapper for exercised shapes."""
            return [
                dict(call.plan.inspect()) for call in wrapper._prepared_calls.values()
            ]

        return run, plans

    # This control forces SG even for T<=64. No planner monkeypatch or source
    # overlay is used; resolve_attention validates metadata and workspace.
    from flashinfer.mla._sparse_mla_sm120._execution import get_sparse_mla_sm120_module

    module = get_sparse_mla_sm120_module()
    props = torch.cuda.get_device_properties(cache.device)
    prepared = {}

    def run(q, indices, lengths, output):
        """Resolve and reuse a direct FP8 SG plan, requiring warmup before capture."""
        key = (q.shape[0], indices.shape[1])
        if key not in prepared:
            if torch.cuda.is_current_stream_capturing():
                raise AssertionError("warm up before capture")
            lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
            metadata = list(
                module.inspect_metadata(
                    q,
                    cache,
                    indices,
                    output,
                    lengths,
                    None,
                    None,
                    None,
                    None,
                    lse,
                    None,
                    None,
                    3,
                    False,
                    512,
                )
            )
            metadata[-1] = 1  # PREFILL_SG
            plan = module.resolve_attention(
                metadata,
                0,
                1,
                props.multi_processor_count,
                props.shared_memory_per_block_optin,
            )
            scratch = [
                torch.empty(
                    tuple(shape), dtype=getattr(torch, str(dtype)), device=q.device
                )
                for shape, dtype, _, _ in plan.workspace()[:2]
            ]
            assert plan.inspect()["numeric_route"] == "fp8"
            prepared[key] = (plan, *scratch, lse)
        plan, mid, mlse, lse = prepared[key]
        module.execute_attention(
            plan,
            q,
            cache,
            indices,
            mid,
            mlse,
            output,
            lse,
            SCALE,
            lengths,
            None,
            None,
            None,
            None,
            1.0,
        )

    def plans():
        """Report descriptors cached by the direct FP8 SG control."""
        return [dict(item[0].inspect()) for item in prepared.values()]

    return run, plans


def measure(run, q, indices, lengths, output):
    """Time 30 CUDA-event intervals after 10 warmups, including host launch gaps."""
    for _ in range(10):
        run(q, indices, lengths, output)
    samples = []
    for _ in range(30):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        run(q, indices, lengths, output)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "timer": "cuda_event",
        "warmups": 10,
        "samples": 30,
        "latency_ms_median": statistics.median(samples),
    }


def run_suite(width, route="wrapper", precision="default", *, timing=False):
    """Return 13 seeded screening records for one width, including two graph replays.

    Finite tolerance failures are recorded without stopping later cases;
    nonfinite values and runtime errors abort. Timing covers eager cases only.
    """
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cache, locations = make_cache(torch.device("cuda"))
    run, plans = make_runner(cache, route, precision)
    records = []

    def compare(name, q, indices, lengths, output):
        """Record and print oracle errors, fixed-limit verdicts and optional timing."""
        expected = reference(q, cache, indices, lengths)
        metrics = error_metrics(output, expected)
        passed = (
            metrics["relative_rms"] <= MAX_RELATIVE_RMS
            and metrics["max_absolute"] <= MAX_ABSOLUTE_ERROR
        )
        record = {
            "case": name,
            "width": width,
            "route": route,
            "precision": precision,
            "passed": passed,
            "lengths": lengths.tolist(),
            **metrics,
        }
        if timing and not name.startswith("graph"):
            record.update(measure(run, q, indices, lengths, output))
        records.append(record)
        print(json.dumps(record), flush=True)

    for rows, name in ((1, "decode"), (7, "verify-shaped"), (65, "prefill")):
        for all_masked in (False, True):
            q, indices, lengths = make_inputs(
                rows, locations, width, all_masked=all_masked
            )
            output = torch.empty_like(q)
            run(q, indices, lengths, output)
            compare(
                name + ("-all-masked" if all_masked else ""),
                q,
                indices,
                lengths,
                output,
            )

    q, indices, lengths = make_inputs(1, locations, width)
    output = torch.empty_like(q)
    for length in (2048, 2049, 2050, 2051):
        lengths.fill_(length)
        indices.fill_(-1)
        indices[0, :length] = locations[:length].to(torch.int32)
        run(q, indices, lengths, output)
        compare(f"tail-{length - 2048}", q, indices, lengths, output)

    q, indices, lengths = make_inputs(1, locations, width, all_masked=True)
    indices.fill_(-1)
    indices[0, :4] = locations[:4].to(torch.int32)
    indices[0, 2048:2051] = locations[4:7].to(torch.int32)
    lengths.fill_(2051)  # Logical bound includes holes and the live tail.
    output = torch.empty_like(q)
    run(q, indices, lengths, output)
    compare("tail-after-holes", q, indices, lengths, output)

    q, indices, lengths = make_inputs(7, locations, width)
    output = torch.empty_like(q)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run(q, indices, lengths, output)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(q, indices, lengths, output)
    for replay in range(2):
        new_q, new_indices, new_lengths = make_inputs(7, locations, width)
        if replay:
            new_lengths.fill_(2051)
            new_indices.fill_(-1)
            new_indices[:, :2051] = locations[:2051].to(torch.int32).expand(7, -1)
        q.copy_(new_q)
        indices.copy_(new_indices)
        lengths.copy_(new_lengths)
        graph.replay()
        compare(f"graph-replay-{replay}", q, indices, lengths, output)
    assert len(records) == 13
    print(
        json.dumps(
            {
                "event": "plans",
                "width": width,
                "route": route,
                "precision": precision,
                "plans": plans(),
            }
        ),
        flush=True,
    )
    return records


def main():
    """Run the requested GPU screening suite, returning one for finite failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("wrapper", "fp8-sg"), default="wrapper")
    parser.add_argument("--precision", default="default")
    parser.add_argument(
        "--widths", type=int, nargs="+", choices=(2112, 2176), default=[2112, 2176]
    )
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    if args.route == "fp8-sg" and args.precision != "default":
        parser.error("fp8-sg is a fixed FP8 control; omit --precision")
    assert torch.cuda.is_available(), "A GPU is required; CPU execution is not a pass"
    assert torch.cuda.get_device_capability() in ((12, 0), (12, 1)), (
        "Run only on SM120/121"
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    import flashinfer

    print(
        json.dumps(
            {
                "event": "start",
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "device": torch.cuda.get_device_name(),
                "capability": torch.cuda.get_device_capability(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "flashinfer": flashinfer.__version__,
                "flashinfer_source": str(Path(flashinfer.__file__).resolve()),
                "flashinfer_commit": getattr(flashinfer, "__git_commit__", None),
                "seed": SEED,
                "limits": {
                    "relative_rms": MAX_RELATIVE_RMS,
                    "max_absolute": MAX_ABSOLUTE_ERROR,
                },
                "cache_bytes": CACHE_BYTES,
                "heads": HEADS,
                "sm_scale": SCALE,
            }
        ),
        flush=True,
    )
    records = []
    for width in args.widths:
        records.extend(run_suite(width, args.route, args.precision, timing=args.timing))
    failures = [
        {"case": r["case"], "width": r["width"]} for r in records if not r["passed"]
    ]
    print(
        json.dumps(
            {
                "event": "summary",
                "status": "FAIL" if failures else "PASS",
                "cases": len(records),
                "failed_cases": failures,
            }
        ),
        flush=True,
    )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
