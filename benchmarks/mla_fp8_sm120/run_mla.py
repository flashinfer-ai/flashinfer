"""Exercise FlashInfer's FA2 BatchMLAPagedAttentionKernel on GLM-4.7-Flash shapes.

Uses synthetic, already-absorbed MLA inputs; no model weights are loaded.
Compare to independent CPU FP64 attention, including ragged/causal page layouts.
Run: python run_mla.py
"""

import hashlib
import importlib.metadata
import json
import math
import statistics
import time
from pathlib import Path

import torch
import flashinfer
from flashinfer.mla import BatchMLAPagedAttentionWrapper
from flashinfer.utils import determine_mla_backend


ROOT = Path(__file__).resolve().parent
SCALE = 1 / math.sqrt(192 + 64)  # Original QK dimension, BEFORE absorption.
torch.manual_seed(47)
torch.set_num_threads(4)


def reference(q_nope, q_pe, cache, qo_indptr, kv_indptr, indices, lengths, causal):
    """Materialize logical pages and evaluate full attention in CPU FP64."""
    q_nope, q_pe, cache = [x.cpu().double() for x in (q_nope, q_pe, cache)]
    outputs, lses = [], []
    for b, length in enumerate(lengths):
        start, end = qo_indptr[b : b + 2].tolist()
        ps, pe = kv_indptr[b : b + 2].tolist()
        kv = cache[indices[ps:pe].long()].reshape(-1, 576)[:length]
        logits = (
            torch.einsum("qhd,kd->qhk", q_nope[start:end], kv[:, :512])
            + torch.einsum("qhd,kd->qhk", q_pe[start:end], kv[:, 512:])
        ) * SCALE
        if causal:
            visible = torch.arange(length)[None, :] <= (
                length - (end - start) + torch.arange(end - start)[:, None]
            )
            logits.masked_fill_(~visible[:, None, :], -torch.inf)
        lses.append(torch.logsumexp(logits, dim=-1))
        outputs.append(torch.einsum("qhk,kd->qhd", logits.softmax(-1), kv[:, :512]))
    return torch.cat(outputs), torch.cat(lses)


def run_case(
    workspace, name, dtype, heads, qlens, lengths, page_size, causal, profile=False
):
    qo_indptr = torch.tensor(
        [0] + list(torch.tensor(qlens).cumsum(0).tolist()), dtype=torch.int32
    )
    pages = [(length + page_size - 1) // page_size for length in lengths]
    kv_indptr = torch.tensor(
        [0] + list(torch.tensor(pages).cumsum(0).tolist()), dtype=torch.int32
    )
    # A permutation tests actual paged addressing, rather than only contiguous KV.
    indices = torch.randperm(sum(pages), dtype=torch.int32)
    q = torch.randn(sum(qlens), heads, 576, dtype=dtype, device="cuda")
    cache = torch.randn(sum(pages), page_size, 576, dtype=dtype, device="cuda")
    q_nope, q_pe = q[..., :512], q[..., 512:]
    ckv, kpe = cache[..., :512], cache[..., 512:]
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
    t0 = time.perf_counter()
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        indices,
        torch.tensor(lengths, dtype=torch.int32),
        heads,
        512,
        64,
        page_size,
        causal,
        SCALE,
        dtype,
        dtype,
    )
    out, lse = wrapper.run(
        q_nope, q_pe, ckv, kpe, return_lse=True, return_lse_base_on_e=True
    )
    torch.cuda.synchronize()
    plan_and_first_run_ms = (time.perf_counter() - t0) * 1000
    expected, expected_lse = reference(
        q_nope, q_pe, cache, qo_indptr, kv_indptr, indices, lengths, causal
    )
    actual = out.cpu().double()
    actual_lse = lse.cpu().double()
    atol, rtol = (0.008, 0.02) if dtype == torch.bfloat16 else (0.002, 0.005)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_lse, expected_lse, atol=0.005, rtol=0.001)

    # Time only repeated GPU run(), without JIT, planning or reference computation.
    for _ in range(5):
        wrapper.run(q_nope, q_pe, ckv, kpe, out=out)
    torch.cuda.synchronize()
    timings = []
    for _ in range(20):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        wrapper.run(q_nope, q_pe, ckv, kpe, out=out)
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000)

    result = dict(
        name=name,
        dtype=str(dtype),
        heads=heads,
        qlens=qlens,
        kv_lens=lengths,
        page_size=page_size,
        causal=causal,
        backend=wrapper._backend,
        out_shape=list(out.shape),
        max_abs_error=(actual - expected).abs().max().item(),
        relative_l2_error=((actual - expected).norm() / expected.norm()).item(),
        lse_max_abs_error=(actual_lse - expected_lse).abs().max().item(),
        median_run_us=statistics.median(timings),
        plan_and_first_run_ms=plan_and_first_run_ms,
        plan_info=list(wrapper._plan_info),
        atol=atol,
        rtol=rtol,
        passed=True,
    )
    if profile:
        try:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                wrapper.run(q_nope, q_pe, ckv, kpe, out=out)
                torch.cuda.synchronize()
            prof.export_chrome_trace(
                str(ROOT / f"trace_{str(dtype).split('.')[-1]}.json")
            )
            result["cuda_kernel_names"] = sorted(
                set(
                    e.name
                    for e in prof.events()
                    if e.device_type == torch.autograd.DeviceType.CUDA
                )
            )
        except Exception as exc:
            result["profile_error"] = repr(exc)
    return result


def rejected_dtype_cases(workspace):
    """Probe public API validation; never bypass validation or launch invalid types."""
    results = []
    for backend, qdtype, kvdtype in [
        ("fa2", torch.bfloat16, torch.float8_e4m3fn),
        ("fa3", torch.bfloat16, torch.float8_e4m3fn),
        ("fa2", torch.float8_e4m3fn, torch.float8_e4m3fn),
        ("fa2", torch.bfloat16, torch.float8_e5m2),
        ("fa2", torch.float32, torch.float32),
        ("fa2", torch.bfloat16, torch.uint8),
        ("fa2", torch.bfloat16, torch.int8),
    ]:
        wrapper = BatchMLAPagedAttentionWrapper(workspace, backend=backend)
        result = dict(backend=backend, qdtype=str(qdtype), kvdtype=str(kvdtype))
        try:
            wrapper.plan(
                torch.tensor([0, 1], dtype=torch.int32),
                torch.tensor([0, 1], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32),
                20,
                512,
                64,
                1,
                False,
                SCALE,
                qdtype,
                kvdtype,
            )
            result["unexpected_plan_success"] = True
        except Exception as exc:
            result.update(error_type=type(exc).__name__, error=str(exc))
        results.append(result)
    return results


def main():
    p = torch.cuda.get_device_properties(0)
    package = Path(flashinfer.__file__).resolve().parent
    sources = [
        "data/include/flashinfer/attention/mla.cuh",
        "mla/_core.py",
        "data/csrc/batch_mla_run.cu",
        "jit/attention/modules.py",
    ]
    report = dict(
        environment=dict(
            gpu=p.name,
            capability=[p.major, p.minor],
            sms=p.multi_processor_count,
            shared_memory_per_sm=p.shared_memory_per_multiprocessor,
            shared_memory_per_block_optin=p.shared_memory_per_block_optin,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            flashinfer=importlib.metadata.version("flashinfer-python"),
            auto_backend=determine_mla_backend(torch.device("cuda")),
            seed=47,
            free_vram_at_start=torch.cuda.mem_get_info()[0],
            source_sha256={
                s: hashlib.sha256((package / s).read_bytes()).hexdigest()
                for s in sources
            },
        ),
        cases=[],
        rejected_dtypes=[],
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    scenarios = [
        ("decode_single", 20, [1], [1024], 1, False),
        ("decode_ragged", 20, [1, 1, 1, 1], [63, 127, 1024, 4097], 16, False),
        ("extend_causal", 20, [3, 5], [129, 257], 16, True),
        ("decode_tp2_shape", 10, [1], [1024], 16, False),
        ("decode_tp4_shape", 5, [1], [1024], 16, False),
    ]
    for dtype in [torch.bfloat16, torch.float16]:
        for name, heads, qlens, lengths, page_size, causal in scenarios:
            result = run_case(
                workspace,
                name,
                dtype,
                heads,
                qlens,
                lengths,
                page_size,
                causal,
                profile=name == "decode_single",
            )
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
            (ROOT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    report["rejected_dtypes"] = rejected_dtype_cases(workspace)
    (ROOT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["rejected_dtypes"], indent=2), flush=True)
    print("All 10 numerical cases passed. Results:", ROOT / "results.json", flush=True)


if __name__ == "__main__":
    main()
