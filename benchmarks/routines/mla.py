import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
import random
import time

import numpy as np
import torch

import flashinfer
from flashinfer._backend import _BackendPlanUnsupportedError
from flashinfer.testing.utils import bench_gpu_time

from mla.reference import (
    MLAReferenceContract,
    mla_paged_attention_reference,
)

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    get_device,
    is_close_stats,
    sample_actual_seq_lens,
)


_MLA_WRAPPER_BACKENDS = (
    "fa2",
    "fa3",
    "cutlass",
    "trtllm-gen",
    "cute-dsl-monolithic",
    "cute-dsl-modular",
    "xqa",
    "auto",
    "prims-ts",
)


def _get_prims_ts_module():
    """Load PrimTS only when its benchmark backend is selected."""
    from flashinfer.attention import prims_ts

    return prims_ts


def parse_mla_args(line, parser, routine):
    """Route each MLA benchmark routine to its contract-specific parser."""
    if routine == "batch_mla_paged_attention":
        return parse_functional_mla_args(line, parser)

    from .attention import parse_attention_args

    return parse_attention_args(line, parser)


def run_mla_test(args):
    """Run the stateful wrapper or one-shot functional MLA benchmark."""
    if args.routine == "batch_mla_paged_attention":
        return run_functional_mla_test(args)
    if args.routine == "BatchMLAPagedAttentionWrapper":
        return testBatchMLAPagedAttentionWrapper(args)
    raise ValueError(f"Unsupported MLA routine: {args.routine}")


def _mla_timing_summary(samples):
    """Summarize one explicitly named MLA measurement phase in milliseconds."""
    if not samples:
        return {"median": None, "p90": None, "mad": None, "repetitions": 0}
    values = np.asarray(samples, dtype=float)
    median = float(np.median(values))
    return {
        "median": median,
        "p90": float(np.percentile(values, 90)),
        "mad": float(np.median(np.abs(values - median))),
        "repetitions": len(samples),
    }


def _mla_empty_timing():
    return _mla_timing_summary([])


def _mla_clone_output(output):
    if isinstance(output, tuple):
        return tuple(value.detach().clone() for value in output)
    return output.detach().clone()


def _mla_clear_reported_timings(outcome):
    outcome["plan_latency_ms"] = None
    outcome["first_run"] = _mla_empty_timing()
    outcome["warm"] = _mla_empty_timing()
    outcome["cold_l2"] = _mla_empty_timing()
    outcome["cuda_graph"] = _mla_empty_timing()


def _capture_mla_backend(
    *,
    wrapper_factory,
    backend,
    constructor_kwargs,
    plan_kwargs,
    run_kwargs,
    synchronize=None,
):
    """Construct, plan, and clone the first output without admitting evidence.

    Backends decide whether a request is supported during ``plan``.  This
    capture phase never performs warm, cold-L2, or graph timing. Correctness is
    decided later across all captured candidates, independently of attempt
    order.
    """

    def failed_outcome(status, reason, *, resolved_backend=None, rejections=()):
        outcome = {
            "requested_backend": backend,
            "resolved_backend": resolved_backend,
            "status": status,
            "correctness_status": status,
            "reason": str(reason),
            "rejections": rejections,
        }
        _mla_clear_reported_timings(outcome)
        return outcome

    try:
        wrapper = wrapper_factory(**constructor_kwargs)
    except Exception as exc:
        return failed_outcome("error", exc)

    try:
        if synchronize is not None:
            synchronize()
    except Exception as exc:
        return failed_outcome("error", exc)

    # Construction and the pre-plan synchronization are intentionally not part
    # of plan latency. Scope the unsupported exception to the planner call
    # itself: the same exception from any other lifecycle phase is an error.
    started = time.perf_counter()
    try:
        wrapper.plan(**plan_kwargs)
    except _BackendPlanUnsupportedError as exc:
        return failed_outcome("unsupported", exc)
    except Exception as exc:
        return failed_outcome("error", exc)

    try:
        resolved_backend = getattr(wrapper, "resolved_backend", None) or backend
    except Exception as exc:
        return failed_outcome("error", exc)

    try:
        trace = getattr(wrapper, "auto_selection_trace", None)
        rejections = () if trace is None else trace.rejections
    except Exception as exc:
        return failed_outcome("error", exc, resolved_backend=resolved_backend)

    try:
        if synchronize is not None:
            synchronize()
        plan_latency_ms = (time.perf_counter() - started) * 1e3
    except Exception as exc:
        return failed_outcome(
            "error",
            exc,
            resolved_backend=resolved_backend,
            rejections=rejections,
        )

    try:
        # Correctness is the actual first post-plan wrapper run. Do not hide it
        # behind a timing helper that may warm up before the first sample.
        if synchronize is not None:
            synchronize()
        first_started = time.perf_counter()
        output = wrapper.run(**run_kwargs)
        if synchronize is not None:
            synchronize()
        first_run = [(time.perf_counter() - first_started) * 1e3]
    except Exception as exc:
        return failed_outcome(
            "error",
            exc,
            resolved_backend=resolved_backend,
            rejections=rejections,
        )

    try:
        captured_output = _mla_clone_output(output)
    except Exception as exc:
        return failed_outcome(
            "error",
            exc,
            resolved_backend=resolved_backend,
            rejections=rejections,
        )

    outcome = {
        "requested_backend": backend,
        "resolved_backend": resolved_backend,
        "status": "captured",
        "correctness_status": "pending",
        "reason": "",
        "rejections": rejections,
        "_captured_output": captured_output,
        "_captured_plan_latency_ms": plan_latency_ms,
        "_captured_first_run": _mla_timing_summary(first_run),
        "_wrapper": wrapper,
        "_run_kwargs": run_kwargs,
    }
    _mla_clear_reported_timings(outcome)
    return outcome


def _mla_reference_tolerance(output):
    dtype = output.dtype if isinstance(output, torch.Tensor) else None
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return 0.15, 0.15
    return 2e-2, 2e-2


def _mla_reference_outputs_close(left, reference, *, require_lse):
    left_output, left_lse = left if isinstance(left, tuple) else (left, None)
    reference_output, reference_lse = (
        reference if isinstance(reference, tuple) else (reference, None)
    )
    if left_output.shape != reference_output.shape:
        return False
    output_rtol, output_atol = _mla_reference_tolerance(reference_output)
    for left_value, right_value in (
        (left_output, reference_output),
        (reference_output, left_output),
    ):
        different, _, _ = is_close_stats(
            left_value.float(),
            right_value.float(),
            rtol=output_rtol,
            atol=output_atol,
        )
        if different:
            return False
    if not require_lse:
        return True
    if (
        left_lse is None
        or reference_lse is None
        or left_lse.shape != reference_lse.shape
    ):
        return False
    for left_value, right_value in (
        (left_lse, reference_lse),
        (reference_lse, left_lse),
    ):
        different, _, _ = is_close_stats(
            left_value.float(),
            right_value.float(),
            rtol=2e-2,
            atol=2e-2,
        )
        if different:
            return False
    return True


def _mla_independent_reference_output(
    args,
    *,
    q_nope,
    q_pe,
    qo_indptr,
    block_tables,
    kv_lens,
    ckv_cache,
    kpe_cache,
    kv_cache,
    run_kwargs,
    sm_scale,
    out_dtype,
):
    contract = MLAReferenceContract(
        lse_mode=args.mla_lse_mode,
        kv_layout=args.mla_kv_layout,
        output_dtype=out_dtype,
        output_scale=args.mla_output_scale,
        scale_mode=args.mla_scale_mode,
        skip_softmax=args.mla_skip_softmax,
    )
    return mla_paged_attention_reference(
        q_nope=q_nope,
        q_pe=q_pe,
        qo_indptr=qo_indptr,
        block_tables=block_tables,
        seq_lens=kv_lens,
        page_size=args.page_size,
        contract=contract,
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
        kv_cache=kv_cache,
        sm_scale=sm_scale,
        ckv_scale=run_kwargs.get("ckv_scale"),
        kpe_scale=run_kwargs.get("kpe_scale"),
        bmm1_scale=run_kwargs.get("bmm1_scale"),
        bmm2_scale=run_kwargs.get("bmm2_scale"),
        o_scale=run_kwargs.get("o_scale"),
        causal=args.causal,
        sinks=run_kwargs.get("sinks"),
    )


def _apply_mla_correctness_consensus(outcomes, *, require_lse, reference_output=None):
    """Admit captured MLA outputs only through the independent reference."""
    captured = [
        index
        for index, outcome in enumerate(outcomes)
        if outcome["status"] == "captured"
    ]
    if reference_output is None:
        for index in captured:
            outcome = outcomes[index]
            outcome["status"] = "error"
            outcome["correctness_status"] = "error"
            outcome["reason"] = (
                "correctness certification unavailable: independent reference "
                "output is required before timing"
            )
            _mla_clear_reported_timings(outcome)
        return

    for index in captured:
        outcome = outcomes[index]
        if _mla_reference_outputs_close(
            outcome["_captured_output"],
            reference_output,
            require_lse=require_lse,
        ):
            outcome["status"] = "ok"
            outcome["correctness_status"] = "correct"
            outcome["reason"] = ""
            outcome["plan_latency_ms"] = outcome["_captured_plan_latency_ms"]
            outcome["first_run"] = outcome["_captured_first_run"]
        else:
            outcome["status"] = "error"
            outcome["correctness_status"] = "incorrect"
            outcome["reason"] = "output or LSE does not match the independent reference"
            _mla_clear_reported_timings(outcome)


def _measure_mla_backend(outcome, *, measure, include_cuda_graph):
    """Measure only a candidate already admitted by correctness consensus."""
    if outcome["correctness_status"] != "correct":
        return
    wrapper = outcome["_wrapper"]
    run_kwargs = outcome["_run_kwargs"]
    try:
        outcome["warm"] = _mla_timing_summary(
            measure(lambda: wrapper.run(**run_kwargs), "warm")
        )
        outcome["cold_l2"] = _mla_timing_summary(
            measure(lambda: wrapper.run(**run_kwargs), "cold_l2")
        )
        outcome["cuda_graph"] = _mla_timing_summary(
            measure(lambda: wrapper.run(**run_kwargs), "cuda_graph")
            if include_cuda_graph
            else []
        )
    except Exception as exc:
        outcome["status"] = "error"
        outcome["correctness_status"] = "error"
        outcome["reason"] = f"timing failed after correctness admission: {exc}"
        _mla_clear_reported_timings(outcome)


def _mla_result_row(
    args,
    outcome,
    *,
    workspace_bytes,
    peak_memory_delta_bytes,
):
    """Translate the MLA-only lifecycle result into the extended CSV schema."""
    row = defaultdict(str)
    row.update(
        routine=args.routine,
        backend=outcome["requested_backend"],
        requested_backend=outcome["requested_backend"],
        resolved_backend=outcome["resolved_backend"] or "",
        mla_qk_nope_head_dim=(
            "" if args.qk_nope_head_dim is None else args.qk_nope_head_dim
        ),
        mla_metadata_form=args.mla_metadata_form,
        mla_enable_pdl=(
            "default"
            if args.enable_pdl is None
            else "true"
            if args.enable_pdl
            else "false"
        ),
        mla_status=outcome["status"],
        mla_correctness_status=outcome["correctness_status"],
        mla_reason=outcome["reason"],
        mla_rejections=json.dumps(outcome["rejections"]),
        mla_lse_mode=args.mla_lse_mode,
        mla_kv_layout=args.mla_kv_layout,
        mla_output_scale=args.mla_output_scale,
        mla_scale_mode=args.mla_scale_mode,
        mla_skip_softmax=_bool_text(args.mla_skip_softmax),
        mla_measurement_seed=args.random_seed,
        plan_latency_ms=outcome["plan_latency_ms"] or "",
        workspace_bytes=workspace_bytes,
        peak_memory_delta_bytes=(
            peak_memory_delta_bytes if peak_memory_delta_bytes is not None else ""
        ),
        page_size=args.page_size,
        batch_size=args.batch_size,
        s_qo=args.s_qo,
        s_kv=args.s_kv,
        num_qo_heads=args.num_qo_heads,
        head_dim_ckv=args.head_dim_ckv,
        head_dim_kpe=args.head_dim_kpe,
        causal=args.causal,
        q_dtype=args.q_dtype,
        kv_dtype=args.kv_dtype,
        out_dtype=args.out_dtype or args.q_dtype,
        random_actual_seq_len=args.random_actual_seq_len,
        enable_pdl=args.enable_pdl,
        no_cuda_graph=args.no_cuda_graph,
        case_tag=args.case_tag,
    )
    for prefix, summary in (
        ("first_run", outcome["first_run"]),
        ("warm", outcome["warm"]),
        ("cold_l2", outcome["cold_l2"]),
        ("cuda_graph", outcome["cuda_graph"]),
    ):
        for statistic in ("median", "p90", "mad"):
            value = summary[statistic]
            row[f"{prefix}_{statistic}_ms"] = "" if value is None else value
    row.update(
        first_run_repetitions=outcome["first_run"]["repetitions"],
        warm_repetitions=outcome["warm"]["repetitions"],
        cold_l2_repetitions=outcome["cold_l2"]["repetitions"],
        graph_replay_repetitions=outcome["cuda_graph"]["repetitions"],
    )
    return row


def testBatchMLAPagedAttentionWrapper(args):
    """Benchmark the stateful MLA wrapper through one normalized lifecycle.

    This intentionally has no functional-TensorRT-LLM, functional-CuTe, or
    functional-auto dispatch.  Each requested backend reaches its own planner;
    its response is authoritative and becomes structured data in the result.
    """
    if args.verbose:
        print("[INFO] Running normalized BatchMLAPagedAttentionWrapper benchmark")

    device = get_device(args)
    input_generator = torch.Generator(device=device)
    input_generator.manual_seed(args.random_seed)
    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    out_dtype = dtype_str_to_torch_dtype(args.out_dtype) if args.out_dtype else q_dtype
    allowed_dtypes = (
        torch.bfloat16,
        torch.float16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if (
        q_dtype not in allowed_dtypes
        or kv_dtype not in allowed_dtypes
        or out_dtype not in allowed_dtypes
    ):
        raise ValueError("MLA benchmark supports BF16, FP16, and supported FP8 dtypes.")
    if args.mla_output_scale == "per-tensor" and out_dtype not in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        raise ValueError("--mla-output-scale per-tensor requires an FP8 --out_dtype.")

    requested = list(args.backends)
    invalid = sorted(set(requested) - set(_MLA_WRAPPER_BACKENDS))
    if invalid:
        raise ValueError("unsupported MLA wrapper backend(s): " + ", ".join(invalid))
    random.Random(args.random_seed).shuffle(requested)

    if args.mla_q_lengths is not None:
        q_lens = torch.tensor(args.mla_q_lengths, dtype=torch.int32, device=device)
        kv_lens = torch.tensor(args.mla_kv_lengths, dtype=torch.int32, device=device)
    else:
        q_lens = sample_actual_seq_lens(
            args.s_qo,
            args.batch_size,
            device,
            args.random_actual_seq_len,
            generator=input_generator,
        ).flatten()
        kv_lens = sample_actual_seq_lens(
            args.s_kv,
            args.batch_size,
            device,
            args.random_actual_seq_len,
            generator=input_generator,
        ).flatten()
    qo_indptr = torch.cat(
        (torch.zeros(1, device=device, dtype=torch.int32), q_lens.cumsum(0))
    ).int()
    pages_per_request = (kv_lens + args.page_size - 1) // args.page_size
    kv_indptr = torch.cat(
        (torch.zeros(1, device=device, dtype=torch.int32), pages_per_request.cumsum(0))
    ).int()
    total_pages = int(kv_indptr[-1].item())
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    block_tables = torch.full(
        (args.batch_size, int(pages_per_request.max().item())),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for index, pages in enumerate(pages_per_request.tolist()):
        start = int(kv_indptr[index].item())
        block_tables[index, :pages] = kv_indices[start : start + pages]

    total_q = int(q_lens.sum().item())
    q_pe = torch.randn(
        (total_q, args.num_qo_heads, args.head_dim_kpe),
        dtype=torch.bfloat16,
        device=device,
        generator=input_generator,
    ).to(q_dtype)
    q_nope = torch.randn(
        (total_q, args.num_qo_heads, args.head_dim_ckv),
        dtype=torch.bfloat16,
        device=device,
        generator=input_generator,
    ).to(q_dtype)
    combined_kv = torch.randn(
        (total_pages, args.page_size, args.head_dim_ckv + args.head_dim_kpe),
        dtype=torch.bfloat16,
        device=device,
        generator=input_generator,
    ).to(kv_dtype)
    combined_q = None
    if "prims-ts" in requested:
        combined_q = torch.cat((q_nope, q_pe), dim=-1)
    ckv_view, kpe_view = combined_kv.split(
        (args.head_dim_ckv, args.head_dim_kpe), dim=-1
    )
    if args.mla_kv_layout == "combined":
        kv_cache, ckv_cache, kpe_cache = combined_kv, None, None
    elif args.mla_kv_layout == "adjacent-split":
        kv_cache, ckv_cache, kpe_cache = None, ckv_view, kpe_view
    else:
        kv_cache, ckv_cache, kpe_cache = None, ckv_view.clone(), kpe_view.clone()

    run_kwargs = {
        "q_pe": q_pe,
        "return_lse": args.mla_lse_mode != "none",
        "return_lse_base_on_e": args.mla_lse_mode == "basee",
    }
    if kv_cache is not None:
        run_kwargs["kv_cache"] = kv_cache
    else:
        run_kwargs.update(ckv_cache=ckv_cache, kpe_cache=kpe_cache)
    if out_dtype != q_dtype or args.mla_output_scale == "per-tensor":
        run_kwargs["out"] = torch.empty(
            (total_q, args.num_qo_heads, args.head_dim_ckv),
            device=device,
            dtype=out_dtype,
        )
    if args.mla_output_scale == "per-tensor":
        run_kwargs["o_scale"] = 1.0
    if args.mla_scale_mode == "kv-per-tensor":
        run_kwargs.update(ckv_scale=1.0, kpe_scale=1.0)
    elif args.mla_scale_mode == "bmm-scalar":
        run_kwargs.update(bmm1_scale=1.0, bmm2_scale=1.0)
    elif args.mla_scale_mode == "bmm-tensor":
        run_kwargs.update(
            bmm1_scale=torch.ones((), device=device),
            bmm2_scale=torch.ones((), device=device),
        )
    if args.mla_use_sinks:
        run_kwargs["sinks"] = torch.zeros(
            args.num_qo_heads, device=device, dtype=torch.float32
        )
    if args.mla_skip_softmax:
        run_kwargs["skip_softmax_threshold_scale_factor"] = 1.0

    plan_kwargs = dict(
        num_heads=args.num_qo_heads,
        head_dim_ckv=args.head_dim_ckv,
        head_dim_kpe=args.head_dim_kpe,
        page_size=args.page_size,
        causal=args.causal,
        q_data_type=q_dtype,
        kv_data_type=kv_dtype,
        enable_pdl=args.enable_pdl,
        use_sinks=args.mla_use_sinks,
        lse_mode=args.mla_lse_mode,
        query_layout="split",
        kv_cache_layout=(
            "packed" if args.mla_kv_layout != "independent-split" else "split"
        ),
        output_dtype=out_dtype,
        output_scale=args.mla_output_scale,
        scale_mode=args.mla_scale_mode,
        skip_softmax=args.mla_skip_softmax,
    )
    if args.mla_metadata_form in ("csr", "dual"):
        plan_kwargs.update(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_len_arr=kv_lens,
        )
    if args.mla_metadata_form in ("dense", "dual"):
        plan_kwargs.update(
            cum_seq_lens_q=qo_indptr,
            block_tables=block_tables,
            seq_lens=kv_lens,
            max_q_len=int(q_lens.max().item()),
        )

    def measure(run, phase):
        return bench_gpu_time(
            fn=run,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=False,
            enable_cupti=args.use_cupti,
            use_cuda_graph=phase == "cuda_graph",
            cold_l2_cache=phase == "cold_l2",
        )

    qk_nope_head_dim = args.qk_nope_head_dim
    scale_qk_nope_head_dim = args.mla_softmax_scale_qk_nope_head_dim
    if scale_qk_nope_head_dim is None:
        scale_qk_nope_head_dim = (
            args.head_dim_ckv if qk_nope_head_dim is None else qk_nope_head_dim
        )
    sm_scale = 1.0 / ((scale_qk_nope_head_dim + args.head_dim_kpe) ** 0.5)
    reference_output = None
    try:
        reference_output = _mla_independent_reference_output(
            args,
            q_nope=q_nope,
            q_pe=q_pe,
            qo_indptr=qo_indptr,
            block_tables=block_tables,
            kv_lens=kv_lens,
            ckv_cache=ckv_cache,
            kpe_cache=kpe_cache,
            kv_cache=kv_cache,
            run_kwargs=run_kwargs,
            sm_scale=sm_scale,
            out_dtype=out_dtype,
        )
    except Exception as exc:
        if args.verbose:
            print(f"[INFO] MLA independent reference unavailable: {exc}")

    outcomes = []
    for backend in requested:
        # The physical q_nope tensor remains canonical and full-width for every
        # candidate. The logical QK profile is declared identically to every
        # concrete backend and auto; each planner remains authoritative about
        # support and unsupported responses are retained as evidence rows.
        backend_plan_kwargs = {
            **plan_kwargs,
            "qk_nope_head_dim": qk_nope_head_dim,
            "sm_scale": sm_scale,
        }
        backend_run_kwargs = {**run_kwargs, "q_nope": q_nope}
        torch.cuda.reset_peak_memory_stats(device)
        capture_baseline = torch.cuda.memory_allocated(device)
        workspace = torch.empty(
            0 if backend == "prims-ts" else 128 * 1024 * 1024,
            dtype=torch.int8,
            device=device,
        )
        constructor_kwargs = {
            "float_workspace_buffer": workspace,
            "use_cuda_graph": not args.no_cuda_graph,
            "backend": backend,
        }
        if args.mla_metadata_form in ("csr", "dual"):
            constructor_kwargs.update(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                kv_len_arr=kv_lens,
            )
        wrapper_factory = flashinfer.mla.BatchMLAPagedAttentionWrapper
        if backend == "prims-ts":
            prims_ts = _get_prims_ts_module()

            class PrimsTSMLAAdapter:
                resolved_backend = "prims-ts"
                auto_selection_trace = None

                def __init__(self, **_kwargs):
                    self._wrapper = prims_ts.BatchMLADecodePagedTSWrapper()
                    self._out = torch.empty(
                        total_q,
                        args.num_qo_heads,
                        args.head_dim_ckv,
                        device=device,
                        dtype=torch.bfloat16,
                    )

                def plan(self, **_kwargs):
                    if q_dtype != kv_dtype:
                        raise _BackendPlanUnsupportedError(
                            "prims-ts requires matching query and cache dtypes"
                        )
                    if q_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
                        raise _BackendPlanUnsupportedError(
                            "prims-ts MLA supports bfloat16 and float8_e4m3fn inputs"
                        )
                    if (args.head_dim_ckv, args.head_dim_kpe) != (512, 64):
                        raise _BackendPlanUnsupportedError(
                            "prims-ts requires head_dim_ckv=512 and head_dim_kpe=64"
                        )
                    if args.page_size not in (16, 32, 64, 128):
                        raise _BackendPlanUnsupportedError(
                            "prims-ts requires page_size in {16, 32, 64, 128}"
                        )
                    if out_dtype != torch.bfloat16:
                        raise _BackendPlanUnsupportedError(
                            "prims-ts MLA produces bfloat16 output"
                        )
                    if args.mla_kv_layout != "combined":
                        raise _BackendPlanUnsupportedError(
                            "prims-ts MLA benchmark requires --mla-kv-layout combined"
                        )
                    if args.mla_lse_mode != "none":
                        raise _BackendPlanUnsupportedError(
                            "prims-ts MLA does not expose LSE"
                        )
                    if args.mla_use_sinks or args.mla_skip_softmax:
                        raise _BackendPlanUnsupportedError(
                            "prims-ts MLA does not support sinks or skip-softmax"
                        )
                    if args.enable_pdl:
                        raise _BackendPlanUnsupportedError(
                            "prims-ts MLA does not expose PDL"
                        )
                    if args.mla_output_scale != "none" or args.mla_scale_mode not in (
                        "default",
                        "bmm-scalar",
                    ):
                        raise _BackendPlanUnsupportedError(
                            "prims-ts MLA supports unscaled BF16 output and scalar BMM scales"
                        )
                    self._wrapper.plan(
                        block_tables,
                        kv_lens,
                        args.num_qo_heads,
                        args.head_dim_ckv,
                        args.head_dim_kpe,
                        args.page_size,
                        qo_indptr=qo_indptr,
                        max_seq_len_q=int(q_lens.max().item()),
                        q_data_type=q_dtype,
                        kv_data_type=kv_dtype,
                        o_data_type=out_dtype,
                        mask_type="causal" if args.causal else "dense",
                        max_kv_len=args.s_kv,
                    )

                def run(self, **run_kwargs):
                    return self._wrapper.run(
                        combined_q,
                        combined_kv,
                        bmm1_scale=run_kwargs.get("bmm1_scale", sm_scale),
                        bmm2_scale=run_kwargs.get("bmm2_scale", 1.0),
                        out=self._out,
                    )

            wrapper_factory = PrimsTSMLAAdapter

        outcome = _capture_mla_backend(
            wrapper_factory=wrapper_factory,
            backend=backend,
            constructor_kwargs=constructor_kwargs,
            plan_kwargs=backend_plan_kwargs,
            run_kwargs=backend_run_kwargs,
            synchronize=lambda: torch.cuda.synchronize(device),
        )
        outcome["_workspace_bytes"] = workspace.numel() * workspace.element_size()
        outcome["_capture_peak_memory_delta_bytes"] = (
            max(
                0,
                torch.cuda.max_memory_allocated(device) - capture_baseline,
            )
            if outcome["status"] == "captured"
            else None
        )
        outcomes.append(outcome)

    _apply_mla_correctness_consensus(
        outcomes,
        require_lse=args.mla_lse_mode != "none",
        reference_output=reference_output,
    )

    results = []
    for outcome in outcomes:
        peak_delta = outcome["_capture_peak_memory_delta_bytes"]
        if outcome["correctness_status"] == "correct":
            torch.cuda.reset_peak_memory_stats(device)
            timing_baseline = torch.cuda.memory_allocated(device)
            _measure_mla_backend(
                outcome,
                measure=measure,
                include_cuda_graph=not args.no_cuda_graph,
            )
            if outcome["status"] == "ok":
                timing_delta = max(
                    0,
                    torch.cuda.max_memory_allocated(device) - timing_baseline,
                )
                peak_delta = max(
                    peak_delta,
                    timing_delta,
                )
        results.append(
            _mla_result_row(
                args,
                outcome,
                workspace_bytes=outcome["_workspace_bytes"],
                peak_memory_delta_bytes=peak_delta,
            )
        )
    return results


FUNCTIONAL_MLA_ROUTINE = "batch_mla_paged_attention"
FUNCTIONAL_MLA_BACKENDS = (
    "auto",
    "xqa",
    "trtllm-gen",
    "cute-dsl",
    "fa2",
    "fa3",
    "cutlass",
)
_DTYPE_NAMES = ("bfloat16", "float16", "float32", "fp8_e4m3", "fp8_e5m2")


@dataclass(frozen=True)
class _FunctionalMLAInputs:
    query: torch.Tensor
    packed_query: torch.Tensor
    kv_cache: torch.Tensor
    block_tables: torch.Tensor
    reference_block_tables: torch.Tensor
    seq_lens: torch.Tensor
    q_lengths: tuple[int, ...]
    qo_indptr: torch.Tensor
    cum_seq_lens_q: torch.Tensor | None
    max_q_len: int | None
    reference_sinks: torch.Tensor | None
    bmm1_scale: float
    bmm2_scale: float
    output_dtype: torch.dtype
    call_kwargs: dict[str, object]


def _parse_bool(value):
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected 'true' or 'false'")


def _parse_lengths(value):
    if not value:
        raise argparse.ArgumentTypeError("expected a nonempty comma-separated list")
    try:
        lengths = tuple(int(entry) for entry in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of integers"
        ) from exc
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("length entries must be positive")
    return lengths


def _validate_functional_args(args, parser):
    if len(set(args.backends)) != len(args.backends):
        parser.error("functional MLA backend requests must be unique")

    positive_fields = (
        "batch_size",
        "s_qo",
        "s_kv",
        "page_size",
        "num_qo_heads",
        "head_dim_ckv",
        "head_dim_kpe",
        "qk_nope_head_dim",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            parser.error(f"--{field} must be positive")

    q_lengths = args.mla_q_lengths
    kv_lengths = args.mla_kv_lengths
    if (q_lengths is None) != (kv_lengths is None):
        parser.error("--mla-q-lengths and --mla-kv-lengths must be supplied together")
    if q_lengths is not None:
        if len(q_lengths) != args.batch_size or len(kv_lengths) != args.batch_size:
            parser.error(
                "--mla-q-lengths and --mla-kv-lengths must each contain "
                "batch_size entries"
            )
        if any(length > args.s_qo for length in q_lengths):
            parser.error("--mla-q-lengths entries must not exceed --s_qo")
        if any(length > args.s_kv for length in kv_lengths):
            parser.error("--mla-kv-lengths entries must not exceed --s_kv")

    if args.mla_skip_softmax and args.mla_lse_mode != "none":
        parser.error("--mla-skip-softmax cannot be combined with requested LSE")
    if args.autotune and "auto" not in args.backends:
        parser.error("--autotune requires requested backend auto")
    if args.cuda_graph and args.no_cuda_graph:
        parser.error("--cuda-graph conflicts with --no_cuda_graph")
    args.no_cuda_graph = not args.cuda_graph
    del args.cuda_graph

    explicit_pdl = args.mla_enable_pdl
    if args.enable_pdl and explicit_pdl == "false":
        parser.error("--enable_pdl conflicts with --mla-enable-pdl false")
    if explicit_pdl == "true" or args.enable_pdl:
        args.enable_pdl = True
    elif explicit_pdl == "false":
        args.enable_pdl = False
    else:
        args.enable_pdl = None
    del args.mla_enable_pdl
    return args


def parse_functional_mla_args(line, parser):
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=FUNCTIONAL_MLA_BACKENDS,
        default=["auto"],
    )
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--s_qo", type=int, default=1)
    parser.add_argument("--s_kv", type=int, required=True)
    parser.add_argument("--page_size", type=int, required=True)
    parser.add_argument("--num_qo_heads", type=int, required=True)
    parser.add_argument("--head_dim_ckv", type=int, required=True)
    parser.add_argument("--head_dim_kpe", type=int, required=True)
    parser.add_argument(
        "--mla-qk-nope-head-dim",
        dest="qk_nope_head_dim",
        type=int,
        required=True,
    )
    parser.add_argument("--q_dtype", choices=_DTYPE_NAMES, default="bfloat16")
    parser.add_argument("--kv_dtype", choices=_DTYPE_NAMES, default="bfloat16")
    parser.add_argument("--out_dtype", choices=_DTYPE_NAMES, default=None)
    parser.add_argument("--mla-q-lengths", type=_parse_lengths, default=None)
    parser.add_argument("--mla-kv-lengths", type=_parse_lengths, default=None)
    parser.add_argument("--mla-is-var-seq", type=_parse_bool, default=True)
    parser.add_argument(
        "--mla-cute-dsl-impl",
        choices=("auto", "monolithic", "modular"),
        default="auto",
    )
    parser.add_argument(
        "--mla-uses-shared-paged-kv-idx",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--mla-enable-pdl",
        choices=("default", "true", "false"),
        default="default",
    )
    parser.add_argument("--mla-lse-mode", choices=("none", "basee"), default="none")
    parser.add_argument("--mla-use-sinks", action="store_true")
    parser.add_argument("--mla-skip-softmax", action="store_true")
    parser.add_argument("--mla-bmm1-scale", type=float, default=None)
    parser.add_argument("--mla-bmm2-scale", type=float, default=1.0)
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Explicitly request the optional functional CUDA-graph phase.",
    )
    return _validate_functional_args(parser.parse_args(line), parser)


def _random_tensor(shape, *, dtype, device, generator):
    seed_dtype = (
        dtype
        if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        else torch.float16
    )
    return torch.randn(
        shape,
        dtype=seed_dtype,
        device=device,
        generator=generator,
    ).to(dtype)


def _build_functional_mla_inputs(args, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(args.random_seed)

    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    output_dtype = (
        q_dtype if args.out_dtype is None else dtype_str_to_torch_dtype(args.out_dtype)
    )
    query_width = args.head_dim_ckv + args.head_dim_kpe
    q_lengths = (
        tuple(args.mla_q_lengths)
        if args.mla_q_lengths is not None
        else (args.s_qo,) * args.batch_size
    )
    kv_lengths = (
        tuple(args.mla_kv_lengths)
        if args.mla_kv_lengths is not None
        else (args.s_kv,) * args.batch_size
    )

    if len(set(q_lengths)) == 1:
        query = _random_tensor(
            (args.batch_size, q_lengths[0], args.num_qo_heads, query_width),
            dtype=q_dtype,
            device=device,
            generator=generator,
        )
        packed_query = query.reshape(sum(q_lengths), args.num_qo_heads, query_width)
        cum_seq_lens_q = None
        max_q_len = None
    else:
        query_parts = [
            _random_tensor(
                (length, args.num_qo_heads, query_width),
                dtype=q_dtype,
                device=device,
                generator=generator,
            )
            for length in q_lengths
        ]
        query = torch.cat(query_parts, dim=0)
        packed_query = query
        cum_seq_lens_q = torch.tensor(
            (0, *torch.tensor(q_lengths).cumsum(0).tolist()),
            dtype=torch.int32,
            device=device,
        )
        max_q_len = max(q_lengths)

    qo_indptr = torch.tensor(
        (0, *torch.tensor(q_lengths).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    max_pages = math.ceil(args.s_kv / args.page_size)
    total_pages = args.batch_size * max_pages
    reference_block_tables = torch.arange(
        total_pages, dtype=torch.int32, device=device
    ).reshape(args.batch_size, max_pages)
    block_tables = (
        reference_block_tables
        if args.mla_uses_shared_paged_kv_idx
        else torch.stack(
            (reference_block_tables, reference_block_tables),
            dim=1,
        )
    )
    kv_cache = _random_tensor(
        (total_pages, args.page_size, query_width),
        dtype=kv_dtype,
        device=device,
        generator=generator,
    )
    seq_lens = torch.tensor(kv_lengths, dtype=torch.int32, device=device)
    reference_sinks = (
        _random_tensor(
            (args.num_qo_heads,),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        if args.mla_use_sinks
        else None
    )
    bmm1_scale = (
        args.mla_bmm1_scale
        if args.mla_bmm1_scale is not None
        else 1 / math.sqrt(args.qk_nope_head_dim + args.head_dim_kpe)
    )

    call_kwargs = {
        "query": query,
        "kv_cache": kv_cache,
        "qk_nope_head_dim": args.qk_nope_head_dim,
        "kv_lora_rank": args.head_dim_ckv,
        "qk_rope_head_dim": args.head_dim_kpe,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "max_seq_len": args.s_kv,
        "bmm1_scale": bmm1_scale,
        "bmm2_scale": args.mla_bmm2_scale,
        "sinks": None if reference_sinks is None else [reference_sinks],
        "skip_softmax_threshold_scale_factor": (1.0 if args.mla_skip_softmax else None),
        "enable_pdl": args.enable_pdl,
        "is_var_seq": args.mla_is_var_seq,
        "uses_shared_paged_kv_idx": args.mla_uses_shared_paged_kv_idx,
        "return_lse": args.mla_lse_mode != "none",
        "cute_dsl_impl": args.mla_cute_dsl_impl,
        "cum_seq_lens_q": cum_seq_lens_q,
        "max_q_len": max_q_len,
    }
    return _FunctionalMLAInputs(
        query=query,
        packed_query=packed_query,
        kv_cache=kv_cache,
        block_tables=block_tables,
        reference_block_tables=reference_block_tables,
        seq_lens=seq_lens,
        q_lengths=q_lengths,
        qo_indptr=qo_indptr,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        reference_sinks=reference_sinks,
        bmm1_scale=bmm1_scale,
        bmm2_scale=args.mla_bmm2_scale,
        output_dtype=output_dtype,
        call_kwargs=call_kwargs,
    )


def _clone_functional_output(output):
    if isinstance(output, tuple):
        return tuple(
            value.detach().clone() if value is not None else None for value in output
        )
    return output.detach().clone()


def _functional_timing_summary(samples):
    if not samples:
        return {
            "median": None,
            "p90": None,
            "mad": None,
            "std": None,
            "repetitions": 0,
        }
    values = np.asarray(samples, dtype=float)
    median = float(np.median(values))
    return {
        "median": median,
        "p90": float(np.percentile(values, 90)),
        "mad": float(np.median(np.abs(values - median))),
        "std": float(np.std(values)),
        "repetitions": len(samples),
    }


def _functional_empty_timing():
    return _functional_timing_summary([])


def _clear_functional_timings(outcome):
    outcome["first_run"] = _functional_empty_timing()
    outcome["warm"] = _functional_empty_timing()
    outcome["cold_l2"] = _functional_empty_timing()
    outcome["cuda_graph"] = _functional_empty_timing()


def _failed_functional_outcome(backend, status, reason):
    outcome = {
        "requested_backend": backend,
        "resolved_backend": "" if backend == "auto" else backend,
        "status": status,
        "correctness_status": status,
        "reason": str(reason),
    }
    _clear_functional_timings(outcome)
    return outcome


def _capture_functional_backend(
    api,
    backend,
    call_kwargs,
    synchronize=None,
):
    kwargs = dict(call_kwargs)
    kwargs["backend"] = backend
    try:
        if synchronize is not None:
            synchronize()
        started = time.perf_counter()
        output = api(**kwargs)
        if synchronize is not None:
            synchronize()
        first_run_ms = (time.perf_counter() - started) * 1e3
        captured = _clone_functional_output(output)
    except (_BackendPlanUnsupportedError, NotImplementedError) as exc:
        return _failed_functional_outcome(backend, "unsupported", exc)
    except Exception as exc:
        return _failed_functional_outcome(backend, "error", exc)
    outcome = {
        "requested_backend": backend,
        "resolved_backend": "" if backend == "auto" else backend,
        "status": "captured",
        "correctness_status": "pending",
        "reason": "",
        "_captured_output": captured,
        "_first_run_ms": first_run_ms,
        "_call_kwargs": kwargs,
    }
    _clear_functional_timings(outcome)
    return outcome


def _normalize_functional_output(output):
    value, lse = output if isinstance(output, tuple) else (output, None)
    if value.ndim == 4:
        value = value.flatten(0, 1)
    if lse is not None and lse.ndim == 3:
        lse = lse.flatten(0, 1)
    return value, lse


def _functional_reference_outputs_close(candidate, reference, *, require_lse):
    candidate_output, candidate_lse = _normalize_functional_output(candidate)
    reference_output, reference_lse = _normalize_functional_output(reference)
    if candidate_output.shape != reference_output.shape:
        return False
    tolerance = (
        0.15
        if reference_output.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        else 2e-2
    )
    if not (
        torch.allclose(
            candidate_output.float(),
            reference_output.float(),
            rtol=tolerance,
            atol=tolerance,
        )
        and torch.allclose(
            reference_output.float(),
            candidate_output.float(),
            rtol=tolerance,
            atol=tolerance,
        )
    ):
        return False
    if not require_lse:
        return True
    if (
        candidate_lse is None
        or reference_lse is None
        or candidate_lse.shape != reference_lse.shape
    ):
        return False
    return torch.allclose(
        candidate_lse.float(),
        reference_lse.float(),
        rtol=2e-2,
        atol=2e-2,
    ) and torch.allclose(
        reference_lse.float(),
        candidate_lse.float(),
        rtol=2e-2,
        atol=2e-2,
    )


def _apply_functional_reference_gate(outcomes, reference_output, require_lse):
    for outcome in outcomes:
        if outcome["status"] != "captured":
            continue
        if reference_output is None:
            outcome["status"] = "error"
            outcome["correctness_status"] = "error"
            outcome["reason"] = (
                "correctness certification unavailable: independent reference "
                "output is required before timing"
            )
            _clear_functional_timings(outcome)
        elif _functional_reference_outputs_close(
            outcome["_captured_output"],
            reference_output,
            require_lse=require_lse,
        ):
            outcome["status"] = "ok"
            outcome["correctness_status"] = "correct"
            outcome["reason"] = ""
            outcome["first_run"] = _functional_timing_summary(
                [outcome["_first_run_ms"]]
            )
        else:
            outcome["status"] = "error"
            outcome["correctness_status"] = "incorrect"
            outcome["reason"] = "output does not match the independent reference"
            _clear_functional_timings(outcome)


def _functional_reference_output(args, inputs):
    return mla_paged_attention_reference(
        q_nope=inputs.packed_query[..., : args.head_dim_ckv],
        q_pe=inputs.packed_query[..., args.head_dim_ckv :],
        kv_cache=inputs.kv_cache,
        qo_indptr=inputs.qo_indptr,
        block_tables=inputs.reference_block_tables,
        seq_lens=inputs.seq_lens,
        page_size=args.page_size,
        sm_scale=inputs.bmm1_scale,
        bmm1_scale=inputs.bmm1_scale,
        bmm2_scale=inputs.bmm2_scale,
        sinks=inputs.reference_sinks,
        contract=MLAReferenceContract(
            lse_mode=args.mla_lse_mode,
            kv_layout="combined",
            scale_mode="bmm-scalar",
            output_dtype=inputs.output_dtype,
            skip_softmax=args.mla_skip_softmax,
        ),
    )


def _measure_functional_backend(outcome, measure, include_cuda_graph):
    if outcome["correctness_status"] != "correct":
        return
    try:
        outcome["warm"] = _functional_timing_summary(measure(outcome["_call"], "warm"))
        outcome["cold_l2"] = _functional_timing_summary(
            measure(outcome["_call"], "cold_l2")
        )
    except Exception as exc:
        outcome["status"] = "error"
        outcome["correctness_status"] = "error"
        outcome["reason"] = f"eager timing failed after correctness admission: {exc}"
        _clear_functional_timings(outcome)
        return

    if not include_cuda_graph:
        outcome["cuda_graph"] = _functional_empty_timing()
        return
    try:
        outcome["cuda_graph"] = _functional_timing_summary(
            measure(outcome["_call"], "cuda_graph")
        )
    except Exception as exc:
        outcome["cuda_graph"] = _functional_empty_timing()
        outcome["reason"] = f"cuda graph timing unavailable: {exc}"


def _run_with_autotune(call, *, enabled, cache):
    if enabled:
        with flashinfer.autotune(True, cache=cache):
            call()
    with flashinfer.autotune(False, cache=cache):
        return call()


def _capture_functional_backend_after_autotune(
    api,
    backend,
    call_kwargs,
    *,
    enabled,
    cache,
    synchronize=None,
):
    if enabled:
        profile_kwargs = dict(call_kwargs)
        profile_kwargs["backend"] = backend
        try:
            with flashinfer.autotune(True, cache=cache):
                api(**profile_kwargs)
        except (_BackendPlanUnsupportedError, NotImplementedError) as exc:
            return _failed_functional_outcome(backend, "unsupported", exc)
        except Exception as exc:
            return _failed_functional_outcome(backend, "error", exc)

    def replay_api(**kwargs):
        return _run_with_autotune(
            lambda: api(**kwargs),
            enabled=False,
            cache=cache,
        )

    return _capture_functional_backend(
        replay_api,
        backend,
        call_kwargs,
        synchronize=synchronize,
    )


def _bool_text(value):
    return "true" if value else "false"


def _functional_result_row(
    args,
    outcome,
    *,
    workspace_bytes,
    peak_memory_delta_bytes,
):
    row = defaultdict(str)
    row.update(
        routine=args.routine,
        backend=outcome["requested_backend"],
        requested_backend=outcome["requested_backend"],
        resolved_backend=outcome["resolved_backend"],
        mla_qk_nope_head_dim=args.qk_nope_head_dim,
        mla_enable_pdl=(
            "default" if args.enable_pdl is None else _bool_text(args.enable_pdl)
        ),
        mla_status=outcome["status"],
        mla_correctness_status=outcome["correctness_status"],
        mla_reason=outcome["reason"],
        mla_rejections="[]",
        mla_lse_mode=args.mla_lse_mode,
        mla_kv_layout="combined",
        mla_output_scale="none",
        mla_scale_mode="bmm-scalar",
        mla_skip_softmax=_bool_text(args.mla_skip_softmax),
        mla_measurement_seed=args.random_seed,
        mla_is_var_seq=_bool_text(args.mla_is_var_seq),
        mla_cute_dsl_impl=args.mla_cute_dsl_impl,
        mla_uses_shared_paged_kv_idx=_bool_text(args.mla_uses_shared_paged_kv_idx),
        mla_autotune=_bool_text(args.autotune),
        workspace_bytes=workspace_bytes,
        peak_memory_delta_bytes=(
            "" if peak_memory_delta_bytes is None else peak_memory_delta_bytes
        ),
        page_size=args.page_size,
        batch_size=args.batch_size,
        s_qo=args.s_qo,
        s_kv=args.s_kv,
        num_qo_heads=args.num_qo_heads,
        head_dim_ckv=args.head_dim_ckv,
        head_dim_kpe=args.head_dim_kpe,
        q_dtype=args.q_dtype,
        kv_dtype=args.kv_dtype,
        out_dtype=args.out_dtype or args.q_dtype,
        enable_pdl=args.enable_pdl,
        no_cuda_graph=args.no_cuda_graph,
        random_seed=args.random_seed,
        case_tag=args.case_tag,
    )
    for prefix in ("first_run", "warm", "cold_l2", "cuda_graph"):
        summary = outcome[prefix]
        for statistic in ("median", "p90", "mad"):
            value = summary[statistic]
            row[f"{prefix}_{statistic}_ms"] = "" if value is None else value
    row["median_time"] = (
        "" if outcome["warm"]["median"] is None else outcome["warm"]["median"]
    )
    row["std_time"] = "" if outcome["warm"]["std"] is None else outcome["warm"]["std"]
    row.update(
        first_run_repetitions=outcome["first_run"]["repetitions"],
        warm_repetitions=outcome["warm"]["repetitions"],
        cold_l2_repetitions=outcome["cold_l2"]["repetitions"],
        graph_replay_repetitions=outcome["cuda_graph"]["repetitions"],
    )
    return row


def _allocate_functional_workspace(device, backend):
    allocator = torch.zeros if backend in ("xqa", "auto") else torch.empty
    return allocator(
        256 * 1024 * 1024,
        dtype=torch.int8,
        device=device,
    )


def run_functional_mla_test(args):
    device = get_device(args)
    inputs = _build_functional_mla_inputs(args, device)
    requested = list(args.backends)
    random.Random(args.random_seed).shuffle(requested)

    reference_output = None
    try:
        reference_output = _functional_reference_output(args, inputs)
    except Exception as exc:
        if args.verbose:
            print(f"[INFO] Functional MLA independent reference unavailable: {exc}")

    api = flashinfer.mla.batch_mla_paged_attention
    is_cuda = device.type == "cuda"
    synchronize = (lambda: torch.cuda.synchronize(device)) if is_cuda else None
    outcomes = []
    for backend in requested:
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            capture_baseline = torch.cuda.memory_allocated(device)
        else:
            capture_baseline = 0

        workspace = _allocate_functional_workspace(device, backend)
        call_kwargs = {
            **inputs.call_kwargs,
            "workspace_buffer": workspace,
            "out": torch.empty(
                (*inputs.query.shape[:-1], args.head_dim_ckv),
                dtype=inputs.output_dtype,
                device=device,
            ),
        }
        if args.mla_lse_mode != "none":
            call_kwargs["lse"] = torch.empty(
                inputs.query.shape[:-1],
                dtype=torch.float32,
                device=device,
            )

        outcome = _capture_functional_backend_after_autotune(
            api,
            backend,
            call_kwargs,
            enabled=args.autotune and backend == "auto",
            cache=args.autotune_cache,
            synchronize=synchronize,
        )
        replay_kwargs = dict(call_kwargs)
        replay_kwargs["backend"] = backend

        def replay(*, _kwargs=replay_kwargs):
            return api(**_kwargs)

        if outcome["status"] == "captured":
            outcome["_call"] = replay
            outcome["_workspace"] = workspace
        outcome["_workspace_bytes"] = workspace.numel() * workspace.element_size()
        outcome["_capture_peak_memory_delta_bytes"] = (
            max(
                0,
                torch.cuda.max_memory_allocated(device) - capture_baseline,
            )
            if is_cuda and outcome["status"] == "captured"
            else None
        )
        outcomes.append(outcome)

    _apply_functional_reference_gate(
        outcomes,
        reference_output,
        require_lse=args.mla_lse_mode != "none",
    )
    for outcome in outcomes:
        outcome.pop("_captured_output", None)
        outcome.pop("_call_kwargs", None)
        outcome.pop("_first_run_ms", None)
        if outcome["correctness_status"] != "correct":
            outcome.pop("_call", None)
            outcome.pop("_workspace", None)

    def measure(call, phase):
        with flashinfer.autotune(False, cache=args.autotune_cache):
            return bench_gpu_time(
                fn=call,
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                sleep_after_run=False,
                enable_cupti=args.use_cupti,
                use_cuda_graph=phase == "cuda_graph",
                cold_l2_cache=phase == "cold_l2",
            )

    results = []
    for outcome in outcomes:
        peak_delta = outcome["_capture_peak_memory_delta_bytes"]
        if outcome["correctness_status"] == "correct":
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(device)
                timing_baseline = torch.cuda.memory_allocated(device)
            else:
                timing_baseline = 0
            _measure_functional_backend(
                outcome,
                measure,
                include_cuda_graph=not args.no_cuda_graph,
            )
            if is_cuda and outcome["status"] == "ok":
                peak_delta = max(
                    peak_delta or 0,
                    max(
                        0,
                        torch.cuda.max_memory_allocated(device) - timing_baseline,
                    ),
                )
        results.append(
            _functional_result_row(
                args,
                outcome,
                workspace_bytes=outcome["_workspace_bytes"],
                peak_memory_delta_bytes=peak_delta,
            )
        )
    return results
