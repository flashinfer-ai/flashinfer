from collections import defaultdict
import functools
import math

import numpy as np
import torch

import flashinfer
import flashinfer.decode

# Try to import cudnn for version checking
CUDNN_AVAILABLE = False
CUDNN_BACKEND_VERSION = 0
try:
    import cudnn

    CUDNN_AVAILABLE = True
    CUDNN_BACKEND_VERSION = cudnn.backend_version()
except ImportError:
    pass
except OSError as e:
    error_msg = str(e).lower()
    is_lib_missing = any(ext in error_msg for ext in [".so", ".dll"])
    if not is_lib_missing:
        raise
from flashinfer import autotune
from flashinfer.fp4_quantization import nvfp4_quantize_paged_kv_cache
from flashinfer.prefill import trtllm_fmha_v2_prefill
from flashinfer.utils import (
    get_compute_capability,
    get_device_sm_count,
    get_trtllm_gen_multi_ctas_kv_counter_bytes,
    is_sm12x_supported,
)
from flashinfer.testing.utils import (
    attention_tb_per_sec_with_actual_seq_lens,
    attention_tflops_per_sec_with_actual_seq_lens,
    bench_gpu_time,
)

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    get_device,
    print_perf_metrics,
    is_close_stats,
    filter_backends_by_compute_capability,
    to_float8,
)

TRTLLM_RAGGED_ROW_ACTIVITY_MODES = (
    "assumed_active",
    "device_check",
    "cpu_mirror",
)
TRTLLM_RAGGED_TIMING_METRIC = "cuda_event_eager_per_call_v1"


def normalize_backends(backends):
    """
    Normalize backend names planned for deprecation and print warnings.
    Currently:
    - Replaces deprecated 'trtllm-gen-native' with 'trtllm-native'.
    - Canonicalizes the Python-module spelling 'prims_ts' to 'prims-ts'.

    Args:
        backends: List of backend names

    Returns:
        List of normalized backend names
    """
    normalized = []
    for backend in backends:
        if backend == "trtllm-gen-native":
            print(
                "[WARNING] Backend name 'trtllm-gen-native' has been renamed to 'trtllm-native' and will be removed in a future release. "
            )
            normalized.append("trtllm-native")
        elif backend == "prims_ts":
            normalized.append("prims-ts")
        else:
            normalized.append(backend)
    return normalized


def _trtllm_ragged_row_activity_kwargs(mode, q_seq_lens_cpu, kv_seq_lens_cpu):
    """Build row-activity arguments for TRTLLM ragged prefill."""
    if mode == "assumed_active":
        return {"skip_all_rows_active_check": True}
    if mode == "device_check":
        return {}
    if mode == "cpu_mirror":
        return {
            "q_seq_lens_cpu": q_seq_lens_cpu,
            "kv_seq_lens_cpu": kv_seq_lens_cpu,
        }
    raise ValueError(f"Unsupported TRTLLM ragged row activity mode: {mode}")


def _drop_backend(backends, backend, reason):
    """Remove an unsupported backend while keeping CLI behavior non-fatal."""
    if backend in backends:
        print(f"[INFO] {backend} backend {reason}. Skipping.")
        backends.remove(backend)


def _drop_backends_without_mode(backends, args, window_ok, soft_cap_ok, sinks_ok):
    """Drop backends that cannot serve the requested sliding-window / soft-cap /
    sink mode, so no backend silently runs without the feature."""
    for backend in list(backends):
        if args.window_left >= 0 and backend not in window_ok:
            _drop_backend(backends, backend, "does not support --window_left here")
        elif args.logits_soft_cap > 0 and backend not in soft_cap_ok:
            _drop_backend(backends, backend, "does not support --logits_soft_cap")
        elif args.sinks and backend not in sinks_ok:
            _drop_backend(backends, backend, "does not support --sinks here")


def _drop_backends_without_trtllm_knobs(backends, args, device, knob_ok):
    """Keep only trtllm-gen backends when a trtllm-gen-only knob is requested.

    use_fp16_softmax / uses_spcompress select SM107-only cubin variants; the
    library raises on any other device (flashinfer/utils.py
    check_trtllm_gen_sm107_only_feature).
    """
    sm107_only = args.use_fp16_softmax or args.uses_spcompress
    if not (sm107_only or args.skip_softmax_threshold_scale_factor is not None):
        return
    for backend in list(backends):
        if backend not in knob_ok:
            _drop_backend(backends, backend, "does not take the trtllm-gen-only knobs")
        elif sm107_only and get_compute_capability(device) != (10, 7):
            _drop_backend(
                backends,
                backend,
                "--use_fp16_softmax/--uses_spcompress cubins exist only for SM107",
            )


def _record_mode_columns(cur_res, args):
    """CSV columns for the optional dense-attention modes; empty when unused so
    rows from lines that do not request a mode keep their historical values."""
    if args.window_left >= 0:
        cur_res["window_left"] = args.window_left
    if args.logits_soft_cap > 0:
        cur_res["logits_soft_cap"] = args.logits_soft_cap
    if args.sinks:
        cur_res["sinks"] = True
    if getattr(args, "use_token_head_sf", False):
        cur_res["use_token_head_sf"] = True
    if args.use_fp16_softmax:
        cur_res["use_fp16_softmax"] = True
    if args.uses_spcompress:
        cur_res["uses_spcompress"] = True
    if args.skip_softmax_threshold_scale_factor is not None:
        cur_res["skip_softmax_threshold_scale_factor"] = (
            args.skip_softmax_threshold_scale_factor
        )


def _cudnn_frontend_at_least(major, minor):
    """Whether the installed cudnn-frontend package is at least major.minor."""
    if not CUDNN_AVAILABLE:
        return False
    try:
        installed = tuple(int(x) for x in cudnn.__version__.split(".")[:2])
    except (AttributeError, ValueError):
        return False
    return installed >= (major, minor)


def _windowed_metric_seq_lens(q_lens, kv_lens, window_left, causal):
    """Effective KV lengths for tflops / TB/s under a left sliding window.

    Query row j of a request sits at position p = kv_len - q_len + j (bottom-right
    aligned) and attends keys [max(0, p - window_left), p] (causal) or
    [max(0, p - window_left), kv_len) (non-causal). Returns
    ``(flops_kv_lens, bytes_kv_lens)``: the first, used with causal=False, makes
    q_len * kv equal the attended (query, key) pairs; the second is the number of
    distinct keys any row reads, min(kv_len, q_len + window_left).
    """
    flops_kv = []
    for q_len, kv_len in zip(q_lens.tolist(), kv_lens.tolist(), strict=False):
        pos = kv_len - q_len + torch.arange(q_len, dtype=torch.float64)
        lo = torch.clamp(pos - window_left, min=0)
        hi = pos + 1 if causal else torch.full_like(pos, kv_len)
        flops_kv.append(float((hi - lo).sum()) / max(q_len, 1))
    bytes_kv = torch.minimum(kv_lens, q_lens + window_left)
    return torch.tensor(flops_kv, dtype=torch.float64), bytes_kv


def _metric_seq_lens(args, q_lens, kv_lens, causal):
    """(q, kv for flops, causal for flops, kv for bytes); identity without a window."""
    if args.window_left < 0:
        return q_lens, kv_lens, causal, kv_lens
    flops_kv, bytes_kv = _windowed_metric_seq_lens(
        q_lens, kv_lens, args.window_left, causal
    )
    return q_lens, flops_kv, False, bytes_kv


def _make_sinks(args, num_qo_heads, device):
    """Per-head learned sink logits (float32 [num_qo_heads]) or None."""
    if not args.sinks:
        return None
    return torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5


def _probe_backend(backend, fn):
    """Run one call; on failure report and return False (probe-and-drop).

    Used for backends whose support is decided by a runtime route or cubin
    lookup (xqa, cutile, cute-dsl) rather than a Python guard the
    harness can mirror.
    """
    try:
        fn()
        torch.cuda.synchronize()
        return True
    except Exception as e:
        msg = str(e).splitlines()[0][:200] if str(e) else ""
        print(
            f"[INFO] {backend} backend does not support this configuration: "
            f"{type(e).__name__}: {msg}. Skipping."
        )
        return False


def _get_prims_ts_module():
    """Import the experimental backend only when a benchmark requests it."""
    from flashinfer.attention import prims_ts

    return prims_ts


def _prims_ts_decode_supports_dtypes(q_dtype, kv_dtype, out_dtype):
    """Helper for checking if PrimTS supports the given dtypes."""
    return (
        (
            q_dtype == torch.float16
            and kv_dtype == torch.float16
            and out_dtype == torch.float16
        )
        or (
            q_dtype == torch.bfloat16
            and kv_dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.uint8)
            and out_dtype == torch.bfloat16
        )
        or (
            q_dtype == torch.float8_e4m3fn
            and kv_dtype in (torch.float8_e4m3fn, torch.uint8)
            and out_dtype in (torch.float16, torch.float8_e4m3fn)
        )
    )


def _select_reference_output(outputs, priority):
    """Return the first available reference backend and output."""
    for backend in priority:
        if backend in outputs:
            return backend, outputs[backend]
    return None, None


def _context_reference_sample_points(qo_indptr_host, num_qo_heads, limit=8):
    """Choose deterministic context samples across requests, rows, and heads."""
    batch_size = len(qo_indptr_host) - 1
    templates = (
        (0, 0.0, 0),
        (batch_size - 1, 1.0, num_qo_heads - 1),
        (batch_size // 4, 0.5, num_qo_heads // 2),
        (batch_size // 2, 1.0, 0),
        (batch_size - 1, 0.0, num_qo_heads // 2),
        (batch_size // 4, 1.0 / 3.0, num_qo_heads // 4),
        (0, 2.0 / 3.0, num_qo_heads - 1),
        (batch_size // 2, 0.5, 0),
    )
    points = []
    for batch_idx, fraction, query_head in templates:
        q_len = qo_indptr_host[batch_idx + 1] - qo_indptr_host[batch_idx]
        query_idx = int(round(fraction * (q_len - 1)))
        point = (batch_idx, query_idx, query_head)
        if point not in points:
            points.append(point)

    # Very small shapes can collapse template points. Fill from a deterministic
    # grid so a requested check still covers as many distinct values as possible.
    for batch_idx in range(batch_size):
        q_len = qo_indptr_host[batch_idx + 1] - qo_indptr_host[batch_idx]
        for query_idx in sorted({0, q_len // 2, q_len - 1}):
            for query_head in sorted({0, num_qo_heads // 2, num_qo_heads - 1}):
                point = (batch_idx, query_idx, query_head)
                if point not in points:
                    points.append(point)
                if len(points) >= limit:
                    return points
    return points[:limit]


@torch.inference_mode()
def _validate_prims_ts_context_samples(
    *,
    q,
    k,
    v,
    out,
    qo_indptr,
    kv_indptr,
    num_qo_heads,
    num_kv_heads,
    sm_scale,
    output_scale,
    causal,
    paged_kv_indices=None,
    kv_lens=None,
):
    """Validate PrimTS context output with exact sampled FP32 attention.

    This keeps the standalone context benchmark's independent accuracy oracle
    after its performance cases move into the unified runner.  ``kv_indptr`` is
    token based for ragged input and page based when ``paged_kv_indices`` is
    supplied.
    """
    flat_out = out.reshape(-1)
    chunk_elements = 8 * 1024 * 1024
    for begin in range(0, flat_out.numel(), chunk_elements):
        if not torch.isfinite(flat_out[begin : begin + chunk_elements].float()).all():
            raise AssertionError("prims-ts context output contains nonfinite values")

    qo_indptr_host = qo_indptr.to("cpu").tolist()
    kv_indptr_host = kv_indptr.to("cpu").tolist()
    kv_lens_host = kv_lens.to("cpu").flatten().tolist() if kv_lens is not None else None
    head_ratio = num_qo_heads // num_kv_heads
    total_error_sq = 0.0
    total_expected_sq = 0.0
    max_abs_error = 0.0
    if out.dtype == torch.float8_e4m3fn:
        rtol, atol, relative_l2_limit = 5e-2, 1.3e-1, 1e-1
    else:
        rtol, atol, relative_l2_limit = 1e-1, 3e-2, 5e-2
    sample_points = _context_reference_sample_points(qo_indptr_host, num_qo_heads)

    for batch_idx, query_idx, query_head in sample_points:
        q_begin, q_end = qo_indptr_host[batch_idx : batch_idx + 2]
        q_len = q_end - q_begin
        kv_head = query_head // head_ratio

        if paged_kv_indices is None:
            kv_begin, kv_end = kv_indptr_host[batch_idx : batch_idx + 2]
            k_matrix = k[kv_begin:kv_end, kv_head].float()
            v_matrix = v[kv_begin:kv_end, kv_head].float()
        else:
            page_begin, page_end = kv_indptr_host[batch_idx : batch_idx + 2]
            physical_ids = paged_kv_indices[page_begin:page_end].to(torch.int64)
            kv_len = kv_lens_host[batch_idx]
            k_matrix = (
                k[:, kv_head]
                .index_select(0, physical_ids)
                .reshape(-1, k.shape[-1])[:kv_len]
                .float()
            )
            v_matrix = (
                v[:, kv_head]
                .index_select(0, physical_ids)
                .reshape(-1, v.shape[-1])[:kv_len]
                .float()
            )

        if causal:
            # Bottom-right causal alignment: row i sees the prefix ending at
            # Skv - Sq + i, inclusive.
            visible_kv = k_matrix.shape[0] - q_len + query_idx + 1
            k_matrix = k_matrix[:visible_kv]
            v_matrix = v_matrix[:visible_kv]

        q_vector = q[q_begin + query_idx, query_head].float()
        probabilities = torch.softmax(torch.mv(k_matrix, q_vector) * sm_scale, dim=0)
        expected = torch.matmul(probabilities, v_matrix) * output_scale
        actual = out[q_begin + query_idx, query_head].float()
        difference = actual - expected
        sample_max_abs = float(difference.abs().max().item())
        max_abs_error = max(max_abs_error, sample_max_abs)

        allowed = atol + rtol * float(expected.abs().max().item())
        if sample_max_abs > allowed:
            raise AssertionError(
                "prims-ts sampled FP32 context reference mismatch at "
                f"batch={batch_idx}, query={query_idx}, head={query_head}: "
                f"max_abs={sample_max_abs:.6g} > allowed={allowed:.6g}"
            )
        total_error_sq += float(torch.sum(difference * difference).item())
        total_expected_sq += float(torch.sum(expected * expected).item())

    relative_l2 = total_error_sq**0.5 / max(total_expected_sq**0.5, 1e-6)
    if relative_l2 > relative_l2_limit:
        raise AssertionError(
            f"prims-ts sampled context relative L2 {relative_l2:.6g} "
            f"> {relative_l2_limit:.6g}"
        )
    return len(sample_points), max_abs_error


@torch.inference_mode()
def _replay_cuda_graph_once(fn, out):
    """Capture one allocation-free launch, poison output, and replay it once."""
    torch.cuda.synchronize(out.device)
    warmup_stream = torch.cuda.Stream(device=out.device)
    warmup_stream.wait_stream(torch.cuda.current_stream(out.device))
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream(out.device).wait_stream(warmup_stream)
    torch.cuda.synchronize(out.device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize(out.device)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize(out.device)
    return out.detach().clone()


def _validate_graph_output(graph_output, eager_output, rtol, atol):
    """Check that an explicitly replayed graph writes the expected output."""
    graph_output = graph_output.reshape_as(eager_output)
    if not torch.isfinite(graph_output.float()).all():
        raise AssertionError("prims-ts CUDA-graph output contains nonfinite values")
    torch.testing.assert_close(
        graph_output.float(), eager_output.float(), rtol=rtol, atol=atol
    )


def run_attention_test(args):
    """
    Run an attention test.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.routine == "BatchDecodeWithPagedKVCacheWrapper":
        return testBatchDecodeWithPagedKVCacheWrapper(args)
    elif args.routine == "BatchPrefillWithPagedKVCacheWrapper":
        return testBatchPrefillWithPagedKVCacheWrapper(args)
    elif args.routine == "BatchPrefillWithRaggedKVCacheWrapper":
        return testBatchPrefillWithRaggedKVCacheWrapper(args)
    elif args.routine == "BatchMLAPagedAttentionWrapper":
        return testBatchMLAPagedAttentionWrapper(args)
    elif args.routine == "trtllm_batch_decode_sparse_mla_dsv4":
        return testTrtllmBatchDecodeSparseMlaDsv4(args)
    elif args.routine == "concat_mla_k":
        return testConcatMlaK(args)
    elif args.routine == "concat_mla_kv_quant_fp8":
        return testConcatMlaKvQuantFp8(args)
    elif args.routine == "append_paged_kv_cache":
        return testAppendPagedKVCache(args)
    elif args.routine in (
        "single_decode_with_kv_cache",
        "single_prefill_with_kv_cache",
    ):
        return testSingleRequestAttention(args)
    elif args.routine == "merge_states":
        return testMergeStates(args)
    else:
        print(f"[ERROR] Unsupported routine: {args.routine}")
        return []


def parse_attention_args(line, parser):
    """
    Parse command line arguments for attention test configuration.

    Args:
        line: Command line arguments
        parser: ArgumentParser object already populated with shared arguments

    Returns:
        Parsed argument namespace
    """
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=None,
        choices=[
            "fa2",
            "fa2_tc",
            "fa3",
            "auto",
            "cudnn",
            "cudnn-native",
            "cutlass",
            "trtllm-gen",
            "trtllm-native",
            "trtllm-fmha-v2",
            "trtllm-gen-native",  # Deprecated, will be removed in future
            "cute-dsl",
            "prims-ts",
            "prims_ts",  # Accepted alias for the Python module spelling.
            "cute-dsl-prims",
            "xqa",
            "cutile",
            "cuda",
            "batch-attention",
            "specialized",
            "cute-dsl-rubin-mtp",
        ],
        help="Kernel backends to test. Default: fa2, except DSV4 sparse MLA defaults to trtllm-gen. prims-ts selects the experimental task-scheduled Blackwell backend for the wrapper attention routines. backend=auto is supported for BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper, and BatchMLAPagedAttentionWrapper (where it pairs with --autotune to select between trtllm-gen and cute-dsl).",
    )
    parser.add_argument(
        "--page_size",
        type=int,
        required=False,
        default=0,
        help="Page size for paged attention. Required for paged attention. Ignored for non-paged attention.",
    )
    parser.add_argument(
        "--batch_size", type=int, required=True, help="Batch size of test case."
    )
    parser.add_argument(
        "--s_qo",
        type=int,
        required=False,
        default=1,
        help="Max sequence length of the query. For decode, 1 is standard decode and >1 enables speculative decode on supported backends.",
    )
    parser.add_argument(
        "--s_kv",
        type=int,
        required=True,
        help="Max sequence length of the key and value.",
    )
    parser.add_argument(
        "--num_qo_heads", type=int, required=True, help="Number of query heads."
    )
    parser.add_argument(
        "--num_kv_heads", type=int, required=True, help="Number of key and value heads."
    )
    parser.add_argument(
        "--head_dim_qk",
        type=int,
        required=False,
        help="Head dimension of the query and key for prefill and decode MHA/GQA/MQA.",
    )
    parser.add_argument(
        "--head_dim_vo",
        type=int,
        required=False,
        help="Head dimension of the value and output for prefill and decode MHA/GQA/MQ.",
    )
    parser.add_argument(
        "--head_dim_ckv",
        type=int,
        required=False,
        help="Head dimension of compressed kv-cache tensor (without rope).",
    )
    parser.add_argument(
        "--head_dim_kpe",
        type=int,
        required=False,
        help="Head dimension of the rope part of the kv-cache tensor.",
    )
    parser.add_argument(
        "--q_dtype",
        type=str,
        required=False,
        default="bfloat16",
        help="Query data type; supported values depend on the selected backend.",
    )
    parser.add_argument(
        "--kv_dtype",
        type=str,
        required=False,
        default="bfloat16",
        help="Key/value data type; supported values depend on the selected backend.",
    )
    parser.add_argument(
        "--v_dtype",
        type=str,
        required=False,
        default=None,
        help="Value data type. If not specified, defaults to kv_dtype. Only the "
        "prims-ts backend accepts a V dtype that differs from kv_dtype "
        "(QK-BF16/PV-FP8).",
    )
    parser.add_argument(
        "--out_dtype",
        type=str,
        required=False,
        default=None,
        help="Data type of the output. If not specified, defaults to q_dtype.",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        default=False,
        help="Enable bottom-right causal masking for backends that support it.",
    )
    parser.add_argument(
        "--spec_dec_mask",
        type=str,
        choices=["causal", "full"],
        default="causal",
        help=(
            "Draft-block mask for speculative decode (--s_qo > 1) in decode tests. "
            "'causal': draft token i attends to draft tokens j <= i (standard "
            "speculative decoding, the default). 'full': every draft token attends "
            "to all draft tokens (e.g. DFlash-style drafters). The KV prefix is "
            "always fully visible."
        ),
    )
    parser.add_argument(
        "--random_actual_seq_len",
        action="store_true",
        default=False,
        help="Use random actual sequence lengths for the query and key and value. Random values are generated between 1 and maximum sequence length. If False, use maximum sequence length.",
    )
    parser.add_argument(
        "--row_activity_mode",
        choices=TRTLLM_RAGGED_ROW_ACTIVITY_MODES,
        default=None,
        help=(
            "TRTLLM native ragged prefill only: choose the all-active fast path, "
            "device-side row check, or CPU sequence-length mirrors."
        ),
    )
    parser.add_argument(
        "--calls_per_sample",
        type=int,
        default=1,
        help="TRTLLM native ragged prefill only: calls grouped into each timing sample.",
    )
    parser.add_argument(
        "--autotune",
        action="store_true",
        default=False,
        help=(
            "Enable autotuner warmup for supported attention routines "
            "(BatchMLAPagedAttentionWrapper with trtllm-native / cute-dsl). "
            "Pre-tunes the kernel configuration before timing so the steady-state "
            "measurement reflects the autotuned tactic."
        ),
    )
    parser.add_argument(
        "--mla_is_var_seq",
        choices=["true", "false", "auto"],
        default=None,
        help=(
            "MLA-only: control the is_var_seq argument passed to "
            "trtllm_batch_decode_with_kv_cache_mla, which selects the var-seq vs. "
            "persistent scheduler (is_persistent = not is_var_seq). "
            "'true'/'false' force the value; 'auto' resolves to --random_actual_seq_len. "
            "If unset (default), is_var_seq is not passed and the API default (True) "
            "is used, preserving existing behavior and perf baselines."
        ),
    )
    parser.add_argument(
        "--mla_cute_dsl_impl",
        choices=["auto", "modular", "monolithic"],
        default=None,
        help=(
            "MLA-only: control the cute_dsl_impl argument passed to "
            "trtllm_batch_decode_with_kv_cache_mla, selecting the CuTe DSL "
            "decode implementation. 'auto' (API default) runs monolithic and "
            "only promotes to modular for modular-only features (e.g. sinks); "
            "'modular'/'monolithic' force that impl. If unset (default), "
            "cute_dsl_impl is not passed and the API default ('auto') is used, "
            "preserving existing behavior and perf baselines."
        ),
    )
    parser.add_argument(
        "--mla_softmax_rescale_threshold_log2",
        type=float,
        default=0.0,
        help=(
            "BatchMLAPagedAttentionWrapper routine with the prims-ts backend only "
            "(ignored by other routines): passed to BatchMLADecodePagedTSWrapper."
            "plan(tuning=MLAAttentionTuning(softmax_rescale_threshold_log2=...)). "
            "A positive value skips the "
            "O rescale while a K tile raises a row maximum by at most this many "
            "binades (bound: 8 for FP8, 32 for BF16); FP8 P is then quantized at "
            "448 * 2**-T instead of 448. 0 (default) keeps the exact rescale."
        ),
    )
    parser.add_argument(
        "--swa_topk",
        type=int,
        default=128,
        help="DSV4 sparse MLA only: fixed sliding-window segment width.",
    )
    parser.add_argument(
        "--compressed_topk",
        type=int,
        default=1920,
        help="DSV4 sparse MLA only: maximum selected compressed-cache rows per query.",
    )
    parser.add_argument(
        "--compressed_kv_len",
        type=int,
        default=None,
        help=(
            "DSV4 sparse MLA only: logical compressed-cache rows per request "
            "(default: ceil(s_kv / 4))."
        ),
    )
    parser.add_argument(
        "--compressed_page_size",
        type=int,
        default=64,
        help="DSV4 sparse MLA only: compressed-cache page size.",
    )
    parser.add_argument(
        "--kv_layout",
        choices=["HND", "NHD"],
        default="HND",
        help="DSV4 sparse MLA and append_paged_kv_cache only: layout of the KV-cache pools.",
    )
    parser.add_argument(
        "--window_left",
        type=int,
        default=-1,
        help=(
            "Dense decode/prefill: left sliding-window size (-1 disables). Backends "
            "without sliding-window support are skipped. tflops and TB/s count only "
            "the in-window keys (min(kv_len, window_left + 1) per query row)."
        ),
    )
    parser.add_argument(
        "--logits_soft_cap",
        type=float,
        default=0.0,
        help="Dense decode/prefill: logits soft-cap (0 disables; fa2/fa3/auto, cute-dsl prefill).",
    )
    parser.add_argument(
        "--sinks",
        action="store_true",
        default=False,
        help=(
            "Dense decode/prefill: add a learned per-head attention-sink logit "
            "(GPT-OSS style). fa2/fa3 run through BatchAttentionWithAttentionSinkWrapper."
        ),
    )
    parser.add_argument(
        "--use_token_head_sf",
        action="store_true",
        default=False,
        help=(
            "BatchDecodeWithPagedKVCacheWrapper fa2_tc only: FP8 KV cache with "
            "one float32 scale per (token, KV head) passed as kv_cache_sf "
            "(plan(use_token_head_sf=True)). Needs an fp8 --kv_dtype and fp16/bf16 Q."
        ),
    )
    parser.add_argument(
        "--use_fp16_softmax",
        action="store_true",
        default=False,
        help=(
            "Paged/ragged prefill trtllm-gen/trtllm-native only: FP16-softmax cubins "
            "(SM107, BF16 only). Also BatchMLAPagedAttentionWrapper trtllm-native "
            "(use_fp16_softmax of trtllm_batch_decode_with_kv_cache_mla, SM107 only)."
        ),
    )
    parser.add_argument(
        "--uses_spcompress",
        action="store_true",
        default=False,
        help="Paged/ragged prefill trtllm-gen/trtllm-native only: Spcomp cubins (SM107, FP8 Q).",
    )
    parser.add_argument(
        "--skip_softmax_threshold_scale_factor",
        type=float,
        default=None,
        help="Dense decode/prefill trtllm-gen/trtllm-native only: skip-softmax threshold scale factor.",
    )
    parser.add_argument(
        "--kv_cache_kind",
        choices=["mha", "mla"],
        default="mha",
        help=(
            "append_paged_kv_cache only: 'mha' appends K/V (append_paged_kv_cache), "
            "'mla' appends ckv/kpe (append_paged_mla_kv_cache). --kv_dtype nvfp4 "
            "selects the NVFP4 quantize-append variant of either."
        ),
    )
    parser.add_argument(
        "--slot_mapping",
        action="store_true",
        default=False,
        help=(
            "append_paged_kv_cache only (mha, --kv_dtype nvfp4): use "
            "nvfp4_quantize_append_paged_kv_cache_with_slot_mapping."
        ),
    )
    parser.add_argument(
        "--num_states",
        type=int,
        default=2,
        help="merge_states only: attention states merged per token (2 = merge_state).",
    )

    parser.add_argument(
        "--sparse_mla_top_k",
        type=int,
        default=0,
        help=(
            "BatchMLAPagedAttentionWrapper only: token-sparse (DSA / DeepSeek-V3.2 "
            "style) MLA decode attending K selected KV tokens per query row "
            "(sparse_mla_top_k of trtllm_batch_decode_with_kv_cache_mla). "
            "trtllm-native/auto only. 0 (default) = dense."
        ),
    )
    parser.add_argument(
        "--dsv4_layer",
        choices=["csa", "hca"],
        default="csa",
        help=(
            "DSV4 sparse MLA only: 'csa' = compressed-sparse layer (compressed ratio "
            "4, --compressed_topk selected rows); 'hca' = heavily-compressed layer "
            "(compressed ratio 128, every compressed row attended; ignores "
            "--compressed_topk/--compressed_kv_len; enables the cute-dsl backend)."
        ),
    )

    args = parser.parse_args(line)

    if args.backends is None:
        args.backends = (
            ["trtllm-gen"]
            if args.routine == "trtllm_batch_decode_sparse_mla_dsv4"
            else ["cuda"]
            if args.routine in ("append_paged_kv_cache", "concat_mla_k", "merge_states")
            else ["specialized"]
            if args.routine == "concat_mla_kv_quant_fp8"
            else ["fa2"]
        )

    if args.routine == "trtllm_batch_decode_sparse_mla_dsv4" and args.page_size == 0:
        args.page_size = 256
    if args.routine == "trtllm_batch_decode_sparse_mla_dsv4":
        # The sparse tables encode causal visibility; there is no non-causal
        # mode in this DSV4 generation-form API.
        args.causal = True

    # Normalize backend names (handle deprecated names)
    args.backends = normalize_backends(args.backends)
    if args.calls_per_sample < 1:
        raise ValueError("--calls_per_sample must be positive")
    if args.row_activity_mode is None and args.calls_per_sample != 1:
        raise ValueError("--calls_per_sample requires --row_activity_mode")
    if args.row_activity_mode is not None:
        if args.routine != "BatchPrefillWithRaggedKVCacheWrapper":
            raise ValueError("--row_activity_mode only supports ragged prefill")
        if args.backends != ["trtllm-native"]:
            raise ValueError("--row_activity_mode requires only trtllm-native")
        if not args.no_cuda_graph or not args.use_cuda_events:
            raise ValueError(
                "--row_activity_mode requires --no_cuda_graph and --use_cuda_events"
            )
    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


def sample_actual_seq_lens(max_seqlen, batch_size, device, random_actual_seq_len):
    """
    Get an array of actual sequence lengths for given batch size and max sequence length.
    If random_actual_seq_len is True, sample actual sequence lengths randomly.
    Otherwise, set all actual sequence lengths to max_seqlen.

    Args:
        max_seqlen: Maximum sequence length.
        batch_size: Batch size.
        device: Device to sample on.
        random_actual_seq_len: Whether to sample actual sequence lengths randomly.

    Returns:
        actual_seq_lens: Actual sequence lengths for each batch.
    """
    if random_actual_seq_len:
        actual_seq_lens = torch.randint(
            1, max_seqlen + 1, (batch_size, 1, 1, 1), device=device, dtype=torch.int32
        )
    else:
        actual_seq_lens = torch.full(
            (batch_size, 1, 1, 1), max_seqlen, device=device, dtype=torch.int32
        )
    return actual_seq_lens


def generate_speculative_mask(batch_size, q_seq_len, device, mask_mode="causal"):
    """Packed draft-block mask for speculative decode (q_len > 1).

    mask_mode "causal": draft token i attends to draft tokens j <= i;
    "full": every draft token attends to all draft tokens. The KV prefix is
    always fully visible. Returns [batch_size, q_seq_len,
    ceil(q_seq_len / 32) * 2] uint16 (bit-packed, as decode APIs expect).
    """
    num_packed_masks_per_token = (q_seq_len + 31) // 32
    q_indices = torch.arange(q_seq_len, device=device, dtype=torch.int32).unsqueeze(1)
    kv_indices = torch.arange(q_seq_len, device=device, dtype=torch.int32).unsqueeze(0)
    if mask_mode == "causal":
        causal_bool_mask = kv_indices <= q_indices
    elif mask_mode == "full":
        causal_bool_mask = torch.ones(
            q_seq_len, q_seq_len, device=device, dtype=torch.bool
        )
    else:
        raise ValueError(f"Unsupported spec-decode mask mode: {mask_mode}")

    padded_seq_len = num_packed_masks_per_token * 32
    if padded_seq_len > q_seq_len:
        padding = torch.zeros(
            q_seq_len, padded_seq_len - q_seq_len, device=device, dtype=torch.bool
        )
        causal_bool_mask = torch.cat([causal_bool_mask, padding], dim=1)

    causal_bool_mask = causal_bool_mask.view(q_seq_len, num_packed_masks_per_token, 32)
    bit_positions = torch.tensor(
        [1 << i for i in range(32)], device=device, dtype=torch.int64
    )
    mask_uint32 = (
        (causal_bool_mask.to(torch.int64) * bit_positions).sum(dim=-1).to(torch.uint32)
    )
    mask_uint32 = (
        mask_uint32.unsqueeze(0)
        .expand(batch_size, q_seq_len, num_packed_masks_per_token)
        .contiguous()
    )
    return mask_uint32.view(torch.uint16)


def testBatchDecodeWithPagedKVCacheWrapper(args):
    """
    Test BatchDecodeWithPagedKVCacheWrapper API and equivalent cuDNN API.
    Supports fa2, fa2_tc, auto, cudnn, trtllm-gen, trtllm-native, and
    prims-ts backends.

    This test:
    1. Creates paged KV cache and query tensors
    2. Runs decode attention with different backends
    3. Verifies outputs match between backends
    4. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testBatchDecodeWithPagedKVCacheWrapper")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    # Basic setup
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    q_init_dtype = torch.bfloat16
    kv_init_dtype = torch.bfloat16
    rtol = 2e-1
    atol = 1e-2
    res = []

    # Handle different query data types.
    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    if q_dtype not in [torch.float16, torch.bfloat16, torch.float8_e4m3fn]:
        print(f"[ERROR] Unsupported q_dtype: {args.q_dtype}")
        return res
    q_init_dtype = torch.float16 if q_dtype == torch.float16 else torch.bfloat16

    # Handle different KV cache data types.
    is_nvfp4_kv = args.kv_dtype == "nvfp4"
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    if kv_dtype not in [
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.uint8,
    ]:
        print(f"[ERROR] Unsupported kv_dtype: {args.kv_dtype}")
        return res
    kv_init_dtype = torch.float16 if kv_dtype == torch.float16 else torch.bfloat16

    v_dtype = dtype_str_to_torch_dtype(args.v_dtype) if args.v_dtype else kv_dtype

    o_data_type = (
        dtype_str_to_torch_dtype(args.out_dtype) if args.out_dtype else q_dtype
    )
    if o_data_type not in [torch.bfloat16, torch.float16, torch.float8_e4m3fn]:
        print(f"[ERROR] Unsupported out_dtype: {args.out_dtype}")
        return res

    # Parse and validate backend configurations
    backends = args.backends
    page_size = args.page_size
    batch_size = args.batch_size
    s_qo = args.s_qo
    speculative_decode = s_qo > 1
    spec_dec_mask_mode = args.spec_dec_mask
    effective_causal = speculative_decode and spec_dec_mask_mode == "causal"
    s_kv = args.s_kv
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim_qk = args.head_dim_qk
    head_dim_vo = args.head_dim_vo if args.head_dim_vo is not None else head_dim_qk
    is_cuda_graph_compatible = not args.no_cuda_graph
    # return_lse = not args.no_lse # TO-DO: Add support for this
    run_refcheck = args.refcheck

    if s_qo > s_kv:
        print("[ERROR] Causal decode requires s_qo <= s_kv. Exiting.")
        return res

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    # Check for backend-specific constraints
    if "fa2" in backends:
        remove_fa2 = False
        if speculative_decode:
            print("[INFO] FA2 backend does not support speculative decode. Skipping.")
            remove_fa2 = True
        head_grp_size = (
            num_qo_heads // num_kv_heads
        )  # If 5, FA2 backend is not supported.
        if head_grp_size == 5:
            print(
                "[INFO] FA2 backend is not supported for this configuration. Skipping."
            )
            remove_fa2 = True
        if remove_fa2:
            backends.remove("fa2")

    if "fa2_tc" in backends:
        remove_fa2_tc = False
        if speculative_decode:
            print(
                "[INFO] FA2_TC backend does not support speculative decode. Skipping."
            )
            remove_fa2_tc = True
        if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            print("[INFO] FA2_TC backend does not support FP8 query. Skipping.")
            remove_fa2_tc = True
        if o_data_type in [torch.float8_e4m3fn, torch.float8_e5m2]:
            print("[INFO] FA2_TC backend does not support FP8 output. Skipping.")
            remove_fa2_tc = True
        if remove_fa2_tc:
            backends.remove("fa2_tc")

    for cudnn_backend in ("cudnn", "cudnn-native"):
        if cudnn_backend not in backends:
            continue
        remove_cudnn = False
        if speculative_decode and cudnn_backend == "cudnn-native":
            # The wrapper backend forwards q_len_per_req > 1 (bottom-right
            # causal rows); the standalone call here is shaped for one row.
            print(
                f"[INFO] {cudnn_backend} backend does not support speculative decode. Skipping."
            )
            remove_cudnn = True
        elif speculative_decode and spec_dec_mask_mode != "causal":
            # The cudnn graph applies the bottom-right causal diagonal to
            # multi-token rows; it takes no explicit draft-block mask.
            print(
                f"[INFO] {cudnn_backend} backend applies the causal mask to multi-token "
                f"decode; spec_dec_mask={spec_dec_mask_mode!r} is not supported. Skipping."
            )
            remove_cudnn = True
        if not (q_dtype == kv_dtype == o_data_type) or q_dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            print(
                f"[INFO] {cudnn_backend} decode requires one fp16/bf16 dtype for q, kv and out. Skipping."
            )
            remove_cudnn = True
        elif q_dtype == torch.float16 and not CUDNN_AVAILABLE:
            print(
                f"[INFO] {cudnn_backend} fp16 decode requires the cudnn-frontend python package "
                "(the fallback cubin path is bf16-only). Skipping."
            )
            remove_cudnn = True
        if remove_cudnn:
            backends.remove(cudnn_backend)

    if "auto" in backends:
        remove_auto = False
        if speculative_decode:
            print("[INFO] auto backend is disabled for speculative decode. Skipping.")
            remove_auto = True
        if o_data_type in [torch.float8_e4m3fn, torch.float8_e5m2]:
            print(
                "[INFO] auto backend may select an implementation without FP8 output support. Skipping."
            )
            remove_auto = True
        if remove_auto:
            backends.remove("auto")

    if "prims-ts" in backends:
        if not _prims_ts_decode_supports_dtypes(q_dtype, kv_dtype, o_data_type):
            _drop_backend(
                backends,
                "prims-ts",
                "does not support the requested Q/KV/output dtype combination",
            )
        elif v_dtype != kv_dtype and not (
            q_dtype == kv_dtype == torch.bfloat16 and v_dtype == torch.float8_e4m3fn
        ):
            _drop_backend(
                backends,
                "prims-ts",
                "requires matching K/V dtypes, except QK-BF16/PV-FP8",
            )
        elif head_dim_qk != head_dim_vo or head_dim_qk not in (64, 128, 192, 256):
            _drop_backend(
                backends,
                "prims-ts",
                "requires equal QK/VO head dimensions in {64, 128, 192, 256}",
            )
        elif page_size not in (4, 16, 32, 64, 128):
            _drop_backend(
                backends,
                "prims-ts",
                "requires page_size in {4, 16, 32, 64, 128}",
            )
        elif q_dtype != kv_dtype and page_size == 16:
            # fmha_decode_config.py: transformed (mixed-dtype) KV rejects page 16
            _drop_backend(
                backends,
                "prims-ts",
                "with mixed Q/KV dtypes requires page_size 32, 64, or 128",
            )
        elif num_qo_heads % num_kv_heads != 0 or not (
            1 <= num_qo_heads // num_kv_heads <= 128
        ):
            _drop_backend(
                backends,
                "prims-ts",
                "requires an integral Q/KV head ratio between 1 and 128",
            )
        elif args.enable_pdl:
            print("[WARNING] prims-ts does not expose PDL; ignoring --enable_pdl.")

    if is_nvfp4_kv:
        # NVFP4 packs V at half width; the CUDA-core decode template requires
        # equal QK/VO head dims (fa2_tc / trtllm serve NVFP4 KV).
        _drop_backend(backends, "fa2", "does not support NVFP4 KV (use fa2_tc)")
    if "cute-dsl" in backends:
        # BatchDecodePagedCuteDSLWrapper.plan (cute_dsl/attention/wrappers/batch_decode.py)
        if q_dtype != kv_dtype or is_nvfp4_kv:
            _drop_backend(backends, "cute-dsl", "requires kv_dtype == q_dtype")
        elif head_dim_qk != head_dim_vo or head_dim_qk % 64 != 0:
            _drop_backend(
                backends,
                "cute-dsl",
                "requires equal head dims that are multiples of 64",
            )
    if "xqa" in backends:
        # xqa.py: out dtype == q dtype unless fp8 out over fp8 KV; NVFP4 KV is SM12x only
        if o_data_type != q_dtype and not (
            o_data_type == torch.float8_e4m3fn and kv_dtype == torch.float8_e4m3fn
        ):
            _drop_backend(backends, "xqa", "requires out_dtype == q_dtype")
        elif is_nvfp4_kv and not is_sm12x_supported(device):
            _drop_backend(backends, "xqa", "supports NVFP4 KV only on SM12x")
    _drop_backends_without_mode(
        backends,
        args,
        window_ok=(
            "fa2",
            "fa2_tc",
            "auto",
            "trtllm-gen",
            "trtllm-native",
            "xqa",
            "cudnn",
            "cudnn-native",
            "cute-dsl",
            "prims-ts",
        ),
        soft_cap_ok=("fa2", "fa2_tc", "auto"),
        sinks_ok=(
            "trtllm-gen",
            "trtllm-native",
            "xqa",
            "cudnn",
            "cudnn-native",
            "cute-dsl",
        ),
    )
    if args.sinks and not speculative_decode and not _cudnn_frontend_at_least(1, 30):
        # BatchDecodeWithPagedKVCacheWrapper docstring: a sink at q_len_per_req == 1
        # needs cudnn-frontend 1.30+ (older stacks raise at the first run).
        for cudnn_backend in ("cudnn", "cudnn-native"):
            _drop_backend(
                backends,
                cudnn_backend,
                "needs cudnn-frontend >= 1.30 for sinks at q_len 1",
            )
    _drop_backends_without_trtllm_knobs(
        backends, args, device, ("trtllm-gen", "trtllm-native", "cute-dsl")
    )
    if args.use_fp16_softmax or args.uses_spcompress:
        print("[ERROR] --use_fp16_softmax/--uses_spcompress are prefill-only.")
        return res

    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    if speculative_decode and spec_dec_mask_mode == "full" and "trtllm-gen" in backends:
        print(
            "[WARNING] trtllm-gen wrapper backend applies implicit causal masking to "
            "the draft block and ignores the non-causal (DFlash) mask; refcheck "
            "against trtllm-native may mismatch."
        )

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}

    # Sample sequence lengths and create tensors
    actual_seq_lens_kv = sample_actual_seq_lens(
        s_kv, batch_size, device, args.random_actual_seq_len
    ).clamp_min(s_qo)
    sum_seq_kv = torch.sum(actual_seq_lens_kv).item()
    avg_seq_len_kv = sum_seq_kv // batch_size

    if args.verbose >= 1:
        print(f"[VERBOSE] Average actual seq len: {avg_seq_len_kv}")
    if args.verbose >= 2:
        print(f"[VVERBOSE] {actual_seq_lens_kv.flatten() = }")

    # Create query tensor
    q = torch.rand(
        batch_size * s_qo,
        num_qo_heads,
        head_dim_qk,
        device=device,
        dtype=q_init_dtype,
    )
    if args.verbose >= 2:
        print(f"[VVERBOSE] {q.shape = }")

    # Create KV cache
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    if args.verbose >= 2:
        print(f"[VVERBOSE] {num_pages_per_seq = }")
        print(f"[VVERBOSE] {total_num_pages = }")

    # Initialize KV cache with appropriate shape and stride
    kv_cache_shape = (
        total_num_pages,
        2,  # 2 for key and value
        num_kv_heads,
        page_size,
        head_dim_qk,
    )
    kv_cache = torch.randn(size=kv_cache_shape, dtype=kv_init_dtype).to(device)

    # Keep a copy for TRT-LLM which uses different strides
    if "trtllm-gen" in backends:
        kv_cache_for_trt = kv_cache.detach().clone()

    kv_cache = kv_cache.as_strided(
        kv_cache.shape,
        (
            2 * page_size * num_kv_heads * head_dim_qk,
            page_size * num_kv_heads * head_dim_qk,
            head_dim_qk,
            num_kv_heads * head_dim_qk,
            1,
        ),
    )
    k_cache_view, v_cache_view = kv_cache[:, 0, :, :, :], kv_cache[:, 1, :, :, :]

    if "trtllm-gen" in backends:
        # kv_cache now has different tensor stride and logical values. Copy over values to kv_cache_for_trt.
        # Result is kv_cache and kv_cache_for_trt have the same logical values but different tensor strides.
        kv_cache_for_trt.copy_(kv_cache)

    v_cache = v_cache_view.as_strided(
        v_cache_view.shape,
        (
            2 * page_size * num_kv_heads * head_dim_qk,
            head_dim_qk,
            num_kv_heads * head_dim_qk,
            1,
        ),
    )
    k_cache = k_cache_view.as_strided(
        k_cache_view.shape,
        (
            2 * page_size * num_kv_heads * head_dim_qk,
            head_dim_qk,
            num_kv_heads * head_dim_qk,
            1,
        ),
    )

    # Now initialize the page tables
    block_tables = torch.tensor(
        [
            [k + i * num_pages_per_seq for k in torch.randperm(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype=torch.int,
        device=device,
    )

    kv_indptr = (
        torch.cat(
            [
                torch.tensor([0], device=device),
                torch.cumsum(
                    (actual_seq_lens_kv.flatten() + page_size - 1) // page_size, dim=0
                ),
            ]
        )
        .int()
        .to(device)
    )

    # kv_indices[-1] is the total number of actual pages
    kv_indices = torch.zeros(kv_indptr[-1], device=device, dtype=torch.int32)
    for i in range(len(kv_indptr) - 1):
        start_idx = kv_indptr[i]
        end_idx = kv_indptr[i + 1]
        kv_indices[start_idx:end_idx] = block_tables[i, : end_idx - start_idx]

    kv_last_page_len = (
        torch.where(
            actual_seq_lens_kv.flatten() % page_size == 0,
            torch.full((batch_size,), page_size, device=device),
            actual_seq_lens_kv.flatten() % page_size,
        )
        .int()
        .to(device)
    )

    ragged_q = (
        torch.arange(0, batch_size + 1, device=device)
        * (s_qo * num_qo_heads * head_dim_qk)
    ).long()  # For cuDNN
    speculative_mask = (
        generate_speculative_mask(batch_size, s_qo, device, spec_dec_mask_mode)
        if speculative_decode
        else None
    )

    scale = float(1.0 / (head_dim_qk**0.5))
    workspace_buffer = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device=device)
    gqa_multi_ctas_kv_counter_buffer = None
    if "trtllm-native" in backends:
        counter_bytes = get_trtllm_gen_multi_ctas_kv_counter_bytes(
            batch_size, num_qo_heads, get_device_sm_count(device)
        )
        gqa_multi_ctas_kv_counter_buffer = torch.zeros(
            counter_bytes, dtype=torch.uint8, device=device
        )

    if args.verbose >= 2:
        print(f"[VVERBOSE] {kv_cache.shape = }")
        print(f"[VVERBOSE] {kv_cache.stride() = }")
        print(f"[VVERBOSE] {block_tables.shape = }")
        print(f"[VVERBOSE] {kv_indptr.shape = }")
        print(f"[VVERBOSE] {kv_indices.shape = }")
        print(f"[VVERBOSE] {kv_last_page_len.shape = }")
        print(f"[VVERBOSE] {scale = }")

    # Mode kwargs are only passed when requested so default lines plan exactly
    # as before.
    plan_mode_kwargs = {}
    if args.window_left >= 0:
        plan_mode_kwargs["window_left"] = args.window_left
    if args.logits_soft_cap > 0:
        plan_mode_kwargs["logits_soft_cap"] = args.logits_soft_cap
    sinks = _make_sinks(args, num_qo_heads, device)
    run_mode_kwargs = {}
    if sinks is not None:
        run_mode_kwargs["sinks"] = sinks
    if args.skip_softmax_threshold_scale_factor is not None:
        run_mode_kwargs["skip_softmax_threshold_scale_factor"] = (
            args.skip_softmax_threshold_scale_factor
        )

    if args.use_token_head_sf:
        # decode.py plan(use_token_head_sf=True): fp8 KV, fp16/bf16 Q, fa2 only.
        if kv_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2) or q_dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            print("[ERROR] --use_token_head_sf needs an fp8 KV cache and fp16/bf16 Q.")
            return []
        for backend in list(backends):
            if backend == "fa2":
                # The library forces the tensor-core kernel, i.e. fa2_tc.
                _drop_backend(
                    backends, backend, "runs as fa2_tc with --use_token_head_sf"
                )
            elif backend != "fa2_tc":
                _drop_backend(backends, backend, "does not take --use_token_head_sf")
        if not backends:
            print("[ERROR] No backends to test. Exiting.")
            return []

    # Prepare wrappers
    backend_wrappers = {}
    resolved_backends = {}
    for backend in list(backends):
        if backend in [
            "fa2",
            "fa2_tc",
            "auto",
            "trtllm-gen",
            "cudnn",
            "cute-dsl",
        ]:
            plan_kv_indptr = (
                kv_indptr.clone().detach() if backend == "trtllm-gen" else kv_indptr
            )
            # Map fa2_tc to fa2 for the actual backend parameter
            # fa2_tc is a benchmark-specific name meaning "fa2 with tensor cores"
            actual_backend = "fa2" if backend == "fa2_tc" else backend
            backend_wrappers[backend] = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer,
                "HND",
                use_cuda_graph=is_cuda_graph_compatible,
                use_tensor_cores=(backend not in ("fa2", "cudnn")),
                paged_kv_indptr_buffer=plan_kv_indptr,
                paged_kv_indices_buffer=kv_indices,
                paged_kv_last_page_len_buffer=kv_last_page_len,
                backend=actual_backend,
            )
            plan_call = functools.partial(
                backend_wrappers[backend].plan,
                plan_kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim_qk,
                page_size,
                q_data_type=q_dtype,
                data_type=kv_dtype,
                o_data_type=o_data_type,
                block_tables=block_tables,
                # Multi-token rows are a plan-time property (the deprecated
                # run-time q_len_per_req only validates against it).
                q_len_per_req=s_qo,
                **plan_mode_kwargs,
                **({"use_token_head_sf": True} if args.use_token_head_sf else {}),
            )
            if backend == "cute-dsl":
                # The CuTe DSL backend compiles at plan(); a failed compile skips it.
                if not _probe_backend(backend, plan_call):
                    backends.remove(backend)
                    continue
            else:
                plan_call()
            resolved_backends[backend] = backend_wrappers[backend].resolved_backend
        elif backend == "trtllm-native":
            # backend="auto" in trtllm_batch_decode_with_kv_cache (decode.py)
            resolved_backends[backend] = (
                "trtllm-gen" if get_compute_capability(device)[0] == 10 else "xqa"
            )
        else:
            resolved_backends[backend] = backend

    ## Prepare dtype-specific data
    k_scale, v_scale = None, None
    kv_cache_sf = None
    kv_cache_nvfp4 = None
    if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        q = q.to(q_dtype)
    if is_nvfp4_kv:
        kv_cache_nvfp4, kv_cache_sf, k_scale, v_scale = nvfp4_quantize_paged_kv_cache(
            kv_cache[:, 0], kv_cache[:, 1]
        )
    elif kv_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        k_data, v_data = torch.chunk(kv_cache, 2, dim=1)
        k_scale = k_data.amax().item() / 256
        v_scale = v_data.amax().item() / 256
        k_fp8 = (k_data / k_scale).to(kv_dtype)
        v_fp8 = (v_data / v_scale).to(kv_dtype)
        kv_cache = torch.cat([k_fp8, v_fp8], dim=1)
        if args.use_token_head_sf:
            # One float32 scale per (page, kv head, slot) in the HND layout;
            # the per-tensor scale value keeps the result comparable.
            sf_shape = (kv_cache.shape[0], num_kv_heads, page_size)
            kv_cache_sf = (
                torch.full(sf_shape, k_scale, dtype=torch.float32, device=device),
                torch.full(sf_shape, v_scale, dtype=torch.float32, device=device),
            )
            k_scale, v_scale = None, None
        if "trtllm-gen" in backends:
            k_data, v_data = torch.chunk(kv_cache_for_trt, 2, dim=1)
            k_fp8 = (k_data / k_scale).to(kv_dtype)
            v_fp8 = (v_data / v_scale).to(kv_dtype)
            kv_cache_for_trt = torch.cat([k_fp8, v_fp8], dim=1)

    prims_ts_kv_cache = None
    prims_ts_kv_scale_factors = None
    prims_ts_out = None
    prims_ts_bmm2_scale = None
    if "prims-ts" in backends:
        prims_ts = _get_prims_ts_module()
        prims_ts_mask_type = (
            "causal" if not speculative_decode or effective_causal else "dense"
        )
        prims_ts_q_shape = (
            (batch_size, num_qo_heads, head_dim_qk)
            if s_qo == 1
            else (batch_size, s_qo, num_qo_heads, head_dim_qk)
        )
        # The common fixture intentionally exposes nonstandard outer strides;
        # PrimTS accepts compact HND pages, so preserve the logical values in a
        # backend-specific compact cache. Matching K/V dtypes keep the combined
        # cache so both operands stay in one allocation.
        prims_ts_v_scale = v_scale
        if is_nvfp4_kv:
            prims_ts_kv_cache = tuple(cache.contiguous() for cache in kv_cache_nvfp4)
            prims_ts_kv_scale_factors = tuple(
                scale_factors.contiguous() for scale_factors in kv_cache_sf
            )
        elif v_dtype != kv_dtype:
            prims_ts_k_cache = kv_cache[:, 0].contiguous()
            prims_ts_v_cache, prims_ts_v_scale_t = to_float8(
                kv_cache[:, 1].contiguous(), v_dtype
            )
            prims_ts_v_scale = prims_ts_v_scale_t.item()
            prims_ts_kv_cache = (prims_ts_k_cache, prims_ts_v_cache)
            # Write the FP8-rounded V back into the caches every other backend
            # reads so --refcheck compares the same V across backends.
            v_dequant = prims_ts_v_cache.to(torch.float32) * prims_ts_v_scale
            kv_cache[:, 1].copy_(v_dequant)
            if "trtllm-gen" in backends:
                kv_cache_for_trt[:, 1].copy_(v_dequant)
        else:
            prims_ts_kv_cache = kv_cache.contiguous()
        prims_ts_bmm2_scale = 1.0 if prims_ts_v_scale is None else prims_ts_v_scale
        prims_ts_out = torch.empty(prims_ts_q_shape, device=device, dtype=o_data_type)
        backend_wrappers["prims-ts"] = prims_ts.BatchDecodePagedTSWrapper("HND")
        backend_wrappers["prims-ts"].plan(
            device,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim_qk,
            page_size,
            s_kv,
            max_seq_len_q=s_qo,
            packed_query=False,
            q_data_type=q_dtype,
            k_data_type=kv_dtype,
            v_data_type=v_dtype,
            o_data_type=o_data_type,
            mask_type=prims_ts_mask_type,
            **({"window_left": args.window_left} if args.window_left >= 0 else {}),
            seq_lens=actual_seq_lens_kv.flatten().tolist(),
            workspace_buffer=workspace_buffer,
        )

    backend_outputs = {}
    for backend in backends:
        if backend == "prims-ts":
            backend_outputs[backend] = prims_ts_out
        else:
            backend_outputs[backend] = torch.empty(
                batch_size * s_qo,
                num_qo_heads,
                head_dim_vo,
                device=device,
                dtype=o_data_type,
            )

    def run_backend_wrapper(
        backend,
        q,
        kv_cache,
        k_cache,
        v_cache,
        workspace_buffer,
        block_tables,
        actual_seq_lens_kv,
        ragged_q,
        speculative_mask,
        out,
    ):
        if backend in [
            "fa2",
            "fa2_tc",
            "auto",
            "trtllm-gen",
            "cudnn",
            "cute-dsl",
        ]:
            wrapper_kv = kv_cache_nvfp4 if is_nvfp4_kv else kv_cache
            return backend_wrappers[backend].run(
                q,
                wrapper_kv,
                k_scale=k_scale,
                v_scale=v_scale,
                q_len_per_req=s_qo,
                kv_cache_sf=kv_cache_sf,
                enable_pdl=args.enable_pdl,
                out=out,
                **run_mode_kwargs,
            )
        elif backend == "cudnn-native":
            native_mode_kwargs = {}
            if args.window_left >= 0:
                native_mode_kwargs["window_left"] = args.window_left
            if sinks is not None:
                native_mode_kwargs["sinks"] = sinks
            return flashinfer.decode.cudnn_batch_decode_with_kv_cache(
                q,
                k_cache,
                v_cache,
                scale,
                workspace_buffer,
                max_sequence_kv=s_kv,
                actual_seq_lens_kv=actual_seq_lens_kv,
                block_tables=block_tables,
                is_cuda_graph_compatible=is_cuda_graph_compatible,
                batch_offsets_q=ragged_q,
                batch_offsets_o=ragged_q,
                out=out,
                **native_mode_kwargs,
            )
        elif backend in ("trtllm-native", "xqa"):
            # trtllm-native keeps backend="auto" (trtllm-gen on SM10x, else xqa);
            # xqa forces that implementation.
            native_kv = kv_cache_nvfp4 if is_nvfp4_kv else kv_cache
            native_mode_kwargs = dict(run_mode_kwargs)
            if args.window_left >= 0:
                native_mode_kwargs["window_left"] = args.window_left
            return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                query=q.contiguous(),
                kv_cache=native_kv,
                workspace_buffer=workspace_buffer,
                block_tables=block_tables,
                seq_lens=actual_seq_lens_kv,
                max_seq_len=s_kv,
                bmm1_scale=scale if k_scale is None else k_scale * scale,
                bmm2_scale=1.0 if v_scale is None else v_scale,
                kv_layout="HND",
                backend="auto" if backend == "trtllm-native" else backend,
                q_len_per_req=s_qo,
                mask=speculative_mask,
                kv_cache_sf=kv_cache_sf,
                enable_pdl=args.enable_pdl,
                multi_ctas_kv_counter_buffer=gqa_multi_ctas_kv_counter_buffer,
                out=out,
                out_dtype=o_data_type,
                **native_mode_kwargs,
            )
        elif backend == "prims-ts":
            runtime_q = (
                q.view(batch_size, num_qo_heads, head_dim_qk)
                if s_qo == 1
                else q.view(batch_size, s_qo, num_qo_heads, head_dim_qk)
            )
            result = backend_wrappers[backend].run(
                runtime_q,
                kv_cache,
                None,
                block_tables,
                kv_scale_factors=prims_ts_kv_scale_factors,
                bmm1_scale=scale if k_scale is None else k_scale * scale,
                bmm2_scale=prims_ts_bmm2_scale,
                out=out,
                validate=False,
            )
            return result.view_as(q)
        else:
            print(f"[ERROR] Backend {backend} not supported")
            return None

    has_reference_output = False
    reference_backend = None
    # Iterate over each backend:
    for cur_backend in backends:
        # Clear workspace buffer to prevent unexpected interactions between backends.
        workspace_buffer.zero_()
        runtime_kv_cache = prims_ts_kv_cache if cur_backend == "prims-ts" else kv_cache
        runtime_out = backend_outputs[cur_backend]
        runtime_k_cache = k_cache if cur_backend == "cudnn-native" else None
        runtime_v_cache = v_cache if cur_backend == "cudnn-native" else None
        runtime_workspace = None if cur_backend == "prims-ts" else workspace_buffer
        if cur_backend in ("xqa", "cute-dsl") and not _probe_backend(
            cur_backend,
            lambda: run_backend_wrapper(
                cur_backend,
                q,
                runtime_kv_cache,
                runtime_k_cache,
                runtime_v_cache,
                runtime_workspace,
                block_tables,
                actual_seq_lens_kv,
                ragged_q,
                speculative_mask,
                runtime_out,
            ),
        ):
            continue
        if run_refcheck:
            outputs[cur_backend] = (
                run_backend_wrapper(
                    cur_backend,
                    q,
                    runtime_kv_cache,
                    runtime_k_cache,
                    runtime_v_cache,
                    runtime_workspace,
                    block_tables,
                    actual_seq_lens_kv,
                    ragged_q,
                    speculative_mask,
                    runtime_out,
                )
                .detach()
                .clone()
            )
            if cur_backend == "fa2":
                has_reference_output = True
                reference_output = outputs[cur_backend]
                reference_backend = "fa2"

        # Unified benchmark entry: prefer graph if compatible and not using CUPTI
        def run_timed_backend(q_arg, kv_arg, k_arg, v_arg, out_arg):
            return run_backend_wrapper(
                cur_backend,
                q_arg,
                kv_arg,
                k_arg,
                v_arg,
                workspace_buffer,
                block_tables,
                actual_seq_lens_kv.flatten(),
                ragged_q,
                speculative_mask,
                out_arg,
            )

        backend_times[cur_backend] = bench_gpu_time(
            fn=run_timed_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=False,
            enable_cupti=args.use_cupti,
            use_cuda_graph=(is_cuda_graph_compatible and cur_backend != "fa2"),
            cold_l2_cache=True,
            input_args=(
                q,
                runtime_kv_cache,
                runtime_k_cache,
                runtime_v_cache,
                runtime_out,
            ),
        )

    # Perform reference check
    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if run_refcheck and "prims-ts" in outputs and is_cuda_graph_compatible:
        prims_ts_runtime_q = (
            q.view(batch_size, num_qo_heads, head_dim_qk)
            if s_qo == 1
            else q.view(batch_size, s_qo, num_qo_heads, head_dim_qk)
        )
        graph_output = _replay_cuda_graph_once(
            lambda: backend_wrappers["prims-ts"].run(
                prims_ts_runtime_q,
                prims_ts_kv_cache,
                None,
                block_tables,
                kv_scale_factors=prims_ts_kv_scale_factors,
                bmm1_scale=scale if k_scale is None else k_scale * scale,
                bmm2_scale=prims_ts_bmm2_scale,
                out=prims_ts_out,
                validate=False,
            ),
            prims_ts_out,
        )
        _validate_graph_output(graph_output, outputs["prims-ts"], rtol, atol)
        if args.verbose >= 1:
            print("[INFO] prims-ts CUDA-graph replay matched eager decode output.")
    if run_refcheck and not has_reference_output and len(tested_backends) > 1:
        reference_backend, reference_output = _select_reference_output(
            outputs,
            [
                "trtllm-gen",
                "trtllm-native",
                "cudnn",
                "auto",
                "prims-ts",
                "xqa",
                "cute-dsl",
            ],
        )
        has_reference_output = reference_backend is not None
    if len(tested_backends) > 1:
        if run_refcheck and has_reference_output:
            if reference_output.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                if args.verbose >= 2:
                    print(
                        "[VVERBOSE] Reference output is FP8. Converting to float32 for reference check."
                    )
                reference_output = reference_output.to(torch.float32)
                tested_outputs = [output.to(torch.float32) for output in tested_outputs]
            for i in range(len(tested_outputs)):
                (
                    num_different_elements,
                    num_elements,
                    num_different_elements_percentage,
                ) = is_close_stats(reference_output, tested_outputs[i], rtol, atol)
                if num_different_elements > 0:
                    print(
                        f"[ERROR] Output tensor mismatch between backends {reference_backend} and {tested_backends[i]}: "
                        f"{num_different_elements} / {num_elements} ({num_different_elements_percentage:.2f}%) elements are different"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {tested_backends[i]} output mismatch"
                        )
    # Compute perf metrics
    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            actual_seq_lens_kv_flat = actual_seq_lens_kv.flatten().to("cpu")
            actual_seq_lens_q_flat = torch.full_like(actual_seq_lens_kv_flat, s_qo)
            # With --window_left the metrics count only the in-window keys.
            metric_q, metric_kv, metric_causal, metric_bytes_kv = _metric_seq_lens(
                args, actual_seq_lens_q_flat, actual_seq_lens_kv_flat, effective_causal
            )
            tflops = attention_tflops_per_sec_with_actual_seq_lens(
                metric_q,
                metric_kv,
                head_dim_qk,
                head_dim_vo,
                num_qo_heads,
                metric_causal,
                median_time,
            )
            tb_per_sec = attention_tb_per_sec_with_actual_seq_lens(
                metric_q,
                metric_bytes_kv,
                head_dim_qk,
                head_dim_vo,
                num_qo_heads,
                num_kv_heads,
                median_time,
                q_dtype=q_dtype,
                k_dtype=kv_dtype,
                v_dtype=v_dtype if backend == "prims-ts" else kv_dtype,
                o_dtype=o_data_type,
            )
            resolved_backend = resolved_backends.get(backend, backend)
            wrapper = backend_wrappers.get(backend)
            if (
                wrapper is not None
                and resolved_backend == "fa2"
                and wrapper.use_tensor_cores
            ):
                resolved_backend = "fa2_tc"
            display_backend = (
                f"{backend}({resolved_backend})"
                if backend in ("auto", "trtllm-native")
                else resolved_backend
            )
            print_perf_metrics(
                display_backend, median_time, std_time, tflops, tb_per_sec
            )

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["backend"] = backend
                cur_res["resolved_backend"] = resolved_backend
                cur_res["page_size"] = page_size
                cur_res["batch_size"] = batch_size
                cur_res["s_qo"] = s_qo
                cur_res["s_kv"] = s_kv
                cur_res["num_qo_heads"] = num_qo_heads
                cur_res["num_kv_heads"] = num_kv_heads
                cur_res["head_dim_qk"] = head_dim_qk
                cur_res["head_dim_vo"] = head_dim_vo
                cur_res["causal"] = effective_causal
                cur_res["q_dtype"] = q_dtype
                cur_res["kv_dtype"] = kv_dtype
                cur_res["v_dtype"] = v_dtype if backend == "prims-ts" else kv_dtype
                cur_res["out_dtype"] = o_data_type
                cur_res["avg_actual_seq_len"] = avg_seq_len_kv
                cur_res["random_actual_seq_len"] = args.random_actual_seq_len
                cur_res["case_tag"] = args.case_tag
                _record_mode_columns(cur_res, args)
                res.append(cur_res)
    return res


def testBatchPrefillWithPagedKVCacheWrapper(args):
    """
    Test BatchPrefillWithPagedKVCacheWrapper API and equivalent cuDNN API.
    Supports fa2, fa3, auto, trtllm-gen, trtllm-native, cudnn, and prims-ts
    backends.

    This test:
    1. Creates paged KV cache and query tensors for prefill
    2. Runs prefill attention with different backends
    3. Verifies outputs match between backends (if refcheck enabled)
    4. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: Dictionary containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testBatchPrefillWithPagedKVCacheWrapper")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    # Basic setup
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    q_init_dtype = torch.bfloat16
    kv_init_dtype = torch.bfloat16
    rtol = 2e-1
    atol = 1e-2
    res = []

    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    if q_dtype not in [torch.float16, torch.bfloat16, torch.float8_e4m3fn]:
        print(f"[ERROR] Unsupported q_dtype: {args.q_dtype}")
        return res
    q_init_dtype = torch.float16 if q_dtype == torch.float16 else torch.bfloat16

    is_nvfp4_kv = args.kv_dtype == "nvfp4"
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    if kv_dtype not in [
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.uint8,
    ]:
        print(f"[ERROR] Unsupported kv_dtype: {args.kv_dtype}")
        return res
    kv_init_dtype = torch.float16 if kv_dtype == torch.float16 else torch.bfloat16

    v_dtype = dtype_str_to_torch_dtype(args.v_dtype) if args.v_dtype else kv_dtype

    o_data_type = (
        dtype_str_to_torch_dtype(args.out_dtype) if args.out_dtype else q_dtype
    )
    if o_data_type not in [torch.bfloat16, torch.float16, torch.float8_e4m3fn]:
        print(f"[ERROR] Unsupported out_dtype: {args.out_dtype}")
        return res

    # Increase tolerances for FP8 due to lower precision
    if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] or kv_dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ]:
        rtol = 5e-1  # Relaxed relative tolerance for FP8
        atol = 1e-1  # Relaxed absolute tolerance for FP8

    # Parse and validate backend configurations
    backends = args.backends
    page_size = args.page_size
    batch_size = args.batch_size
    s_qo = args.s_qo
    s_kv = args.s_kv
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim_qk = args.head_dim_qk
    head_dim_vo = args.head_dim_vo if args.head_dim_vo is not None else head_dim_qk
    causal = args.causal
    is_cuda_graph_compatible = not args.no_cuda_graph
    # return_lse = not args.no_lse # TO-DO: Add support for this
    run_refcheck = args.refcheck

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    # Check for backend-specific constraints
    if "fa2" in backends:
        remove_fa2 = False
        if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            print("[INFO] FA2 backend does not support FP8. Skipping.")
            remove_fa2 = True
        if remove_fa2:
            backends.remove("fa2")
    if "cudnn" in backends:
        remove_cudnn = False
        # cuDNN FP8 prefill requires cuDNN >= 9.17.1 (backend version 91701)
        if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] or kv_dtype in [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]:
            if not CUDNN_AVAILABLE or CUDNN_BACKEND_VERSION < 91701:
                print(
                    f"[INFO] cuDNN FP8 prefill requires cuDNN >= 9.17.1. "
                    f"Current version: {CUDNN_BACKEND_VERSION}. Skipping cudnn backend."
                )
                remove_cudnn = True
        if remove_cudnn:
            backends.remove("cudnn")

    if "cudnn-native" in backends:
        remove_cudnn_native = False
        # cuDNN-native does not yet support FP8 prefill
        if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] or kv_dtype in [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]:
            if not CUDNN_AVAILABLE or CUDNN_BACKEND_VERSION < 91701:
                print(
                    f"[INFO] cuDNN FP8 prefill requires cuDNN >= 9.17.1. "
                    f"Current version: {CUDNN_BACKEND_VERSION}. Skipping cudnn-native backend."
                )
                remove_cudnn_native = True
        if remove_cudnn_native:
            backends.remove("cudnn-native")

    # trtllm-gen paged context supports non-causal (dense) attention, but not a
    # sliding window without causal masking (prefill.py, trtllm-gen plan/context).
    for trtllm_backend in ("trtllm-gen", "trtllm-native"):
        if not causal and args.window_left >= 0:
            _drop_backend(
                backends, trtllm_backend, "does not support non-causal sliding window"
            )
    if "cute-dsl" in backends:
        # BatchPrefillWithPagedKVCacheWrapper(backend="cute-dsl") plan/run (prefill.py)
        if q_dtype != kv_dtype or is_nvfp4_kv:
            _drop_backend(backends, "cute-dsl", "requires kv_dtype == q_dtype")
        elif q_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # run() rejects the q/k/v scale factors FP8 inputs need
            _drop_backend(
                backends,
                "cute-dsl",
                "paged path does not take q/k/v scale factors (FP8)",
            )
        elif head_dim_qk != head_dim_vo:
            _drop_backend(backends, "cute-dsl", "requires head_dim_qk == head_dim_vo")
    if "batch-attention" in backends:
        # flashinfer.BatchAttention (attention/_core.py): holistic persistent kernel
        if not (q_dtype == kv_dtype == o_data_type) or q_dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            _drop_backend(
                backends, "batch-attention", "requires one fp16/bf16 q/kv/out dtype"
            )
        elif max(head_dim_qk, head_dim_vo) > 256:
            _drop_backend(backends, "batch-attention", "requires head_dim <= 256")
    if "trtllm-fmha-v2" in backends and is_nvfp4_kv:
        print("[INFO] trtllm-fmha-v2 backend does not support NVFP4. Skipping.")
        backends.remove("trtllm-fmha-v2")

    if "cutlass" in backends:
        print("[INFO] CUTLASS backend does not support prefill. Skipping.")
        remove_cutlass = True
        if remove_cutlass:
            backends.remove("cutlass")

    if "prims-ts" in backends:
        if is_nvfp4_kv:
            _drop_backend(backends, "prims-ts", "does not support NVFP4 K/V")
        elif q_dtype != kv_dtype:
            _drop_backend(backends, "prims-ts", "requires matching Q and K dtypes")
        elif v_dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn):
            _drop_backend(
                backends, "prims-ts", "supports only FP16, BF16, or FP8-E4M3 V"
            )
        elif v_dtype != kv_dtype and not (
            kv_dtype == torch.bfloat16 and v_dtype == torch.float8_e4m3fn
        ):
            _drop_backend(
                backends,
                "prims-ts",
                "requires matching K/V dtypes, except QK-BF16/PV-FP8",
            )
        elif head_dim_qk != head_dim_vo or head_dim_qk not in (128, 192, 256):
            _drop_backend(
                backends,
                "prims-ts",
                "requires equal QK/VO head dimensions in {128, 192, 256}",
            )
        elif page_size not in (16, 32, 64, 128):
            _drop_backend(
                backends,
                "prims-ts",
                "requires page_size in {16, 32, 64, 128}",
            )
        elif num_qo_heads % num_kv_heads != 0:
            _drop_backend(backends, "prims-ts", "requires Hq to be divisible by Hkv")
        elif args.enable_pdl:
            print("[WARNING] prims-ts does not expose PDL; ignoring --enable_pdl.")

    # Sliding windows on fmha-v2 / prims-ts require causal masking.
    window_ok = ["fa2", "fa3", "auto", "trtllm-gen", "trtllm-native", "cute-dsl"]
    if causal:
        window_ok += ["trtllm-fmha-v2", "prims-ts"]
    _drop_backends_without_mode(
        backends,
        args,
        window_ok=window_ok,
        soft_cap_ok=("fa2", "fa3", "auto", "cute-dsl", "batch-attention"),
        # fa2/fa3/auto serve sinks through BatchAttentionWithAttentionSinkWrapper
        sinks_ok=("fa2", "fa3", "auto", "trtllm-gen", "trtllm-native"),
    )
    _drop_backends_without_trtllm_knobs(
        backends, args, device, ("trtllm-gen", "trtllm-native")
    )
    if args.sinks and q_dtype != o_data_type:
        for sink_wrapper_backend in ("fa2", "fa3", "auto"):
            _drop_backend(
                backends,
                sink_wrapper_backend,
                "attention-sink wrapper requires out_dtype == q_dtype",
            )

    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    # Check for layer-specific constraints
    layer_not_supported = False
    if s_qo > s_kv:
        print("[ERROR] s_qo > s_kv is not supported. Exiting.")
        layer_not_supported = True
    if layer_not_supported:
        print("[ERROR] Layer not supported. Exiting.")
        return res

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}

    # Sample sequence lengths.
    # If s_qo == s_kv, then make sampled actual_seq_lens_kv the same as actual_seq_lens_q.
    # IF s_qo < s_kv, then sample actual_seq_lens_kv separately. Then ensure actual_seq_lens_kv is at least as long as actual_seq_lens_q.
    actual_seq_lens_q = sample_actual_seq_lens(
        s_qo, batch_size, None, args.random_actual_seq_len
    )
    if s_qo == s_kv:
        if args.verbose >= 2:
            print(
                "[VVERBOSE] s_qo == s_kv, making actual_seq_lens_kv the same as actual_seq_lens_q"
            )
        actual_seq_lens_kv = actual_seq_lens_q.clone()
    else:  # s_qo < s_kv
        if args.verbose >= 2:
            print("[VVERBOSE] s_qo < s_kv, sampling actual_seq_lens_kv")
        actual_seq_lens_kv = sample_actual_seq_lens(
            s_kv, batch_size, None, args.random_actual_seq_len
        )
        actual_seq_lens_kv = torch.maximum(actual_seq_lens_kv, actual_seq_lens_q)

    avg_seq_len_q = actual_seq_lens_q.sum().item() // batch_size
    avg_seq_len_kv = actual_seq_lens_kv.sum().item() // batch_size
    if args.verbose >= 1:
        print(f"[VERBOSE] Average actual qo seq len: {avg_seq_len_q}")
        print(f"[VERBOSE] Average actual kv seq len: {avg_seq_len_kv}")
    if args.verbose >= 2:
        print(f"[VVERBOSE] {actual_seq_lens_q.flatten() = }")
        print(f"[VVERBOSE] {actual_seq_lens_kv.flatten() = }")

    cumsum_s_qo = torch.sum(actual_seq_lens_q)
    q = torch.randn(
        cumsum_s_qo, num_qo_heads, head_dim_qk, device=device, dtype=q_init_dtype
    )
    if args.verbose >= 2:
        print(f"[VVERBOSE] {q.shape = }")

    # Create KV cache
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    if args.verbose >= 2:
        print(f"[VVERBOSE] {num_pages_per_seq = }")
        print(f"[VVERBOSE] {total_num_pages = }")

    kv_cache_shape = (total_num_pages, 2, num_kv_heads, page_size, head_dim_qk)
    kv_cache = torch.randn(size=kv_cache_shape, dtype=kv_init_dtype).to(device)
    kv_cache = kv_cache.as_strided(
        kv_cache.shape,
        (
            2 * page_size * num_kv_heads * head_dim_qk,
            page_size * num_kv_heads * head_dim_qk,
            head_dim_qk,
            num_kv_heads * head_dim_qk,
            1,
        ),
    )
    k_cache_view, v_cache_view = kv_cache[:, 0, :, :, :], kv_cache[:, 1, :, :, :]

    v_cache = v_cache_view.as_strided(
        v_cache_view.shape,
        (
            2 * page_size * num_kv_heads * head_dim_qk,
            head_dim_qk,
            num_kv_heads * head_dim_qk,
            1,
        ),
    )
    k_cache = k_cache_view.as_strided(
        k_cache_view.shape,
        (
            2 * page_size * num_kv_heads * head_dim_qk,
            head_dim_qk,
            num_kv_heads * head_dim_qk,
            1,
        ),
    )

    # Now initialize the page tables
    block_tables = torch.tensor(
        [
            [k + i * num_pages_per_seq for k in torch.randperm(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype=torch.int,
        device=device,
    )

    actual_seq_lens_q_device = actual_seq_lens_q.to(device)
    actual_seq_lens_kv_device = actual_seq_lens_kv.to(device)
    q_indptr = (
        torch.cat(
            [
                torch.tensor([0], device=device),
                torch.cumsum(actual_seq_lens_q_device.view(-1), dim=0)
                * head_dim_qk
                * num_qo_heads,
            ]
        )
        .long()
        .to(device)
    )  # Element-unit offsets, for the low-level cudnn-native call only
    qo_indptr = (
        torch.cat(
            [
                torch.tensor([0], device=device),
                torch.cumsum(actual_seq_lens_q_device.view(-1), dim=0),
            ]
        )
        .int()
        .to(device)
    )

    # Page-based indptr for FlashInfer paged attention (cumulative page counts)
    kv_indptr = (
        torch.cat(
            [
                torch.tensor([0], device=device),
                torch.cumsum(
                    (actual_seq_lens_kv_device.flatten() + page_size - 1) // page_size,
                    dim=0,
                ),
            ]
        )
        .int()
        .to(device)
    )
    # Token-based indptr for TRT-LLM backends (cumulative token counts)
    kv_token_indptr = (
        torch.cat(
            [
                torch.tensor([0], device=device),
                torch.cumsum(actual_seq_lens_kv_device.flatten(), dim=0),
            ]
        )
        .int()
        .to(device)
    )
    kv_indices = torch.zeros(kv_indptr[-1], device=device, dtype=torch.int32)
    for i in range(len(kv_indptr) - 1):
        start_idx = kv_indptr[i]
        end_idx = kv_indptr[i + 1]
        kv_indices[start_idx:end_idx] = block_tables[i, : end_idx - start_idx]
    kv_last_page_len = (
        torch.where(
            actual_seq_lens_kv_device.flatten() % page_size == 0,
            torch.full((batch_size,), page_size, device=device),
            actual_seq_lens_kv_device.flatten() % page_size,
        )
        .int()
        .to(device)
    )

    scale = float(1.0 / (head_dim_qk**0.5))
    workspace_buffer = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {kv_cache.shape = }")
        print(f"[VVERBOSE] {kv_cache.stride() = }")
        print(f"[VVERBOSE] {block_tables.shape = }")
        print(f"[VVERBOSE] {qo_indptr.shape = }")
        print(f"[VVERBOSE] {qo_indptr.dtype = }")
        print(f"[VVERBOSE] {kv_indptr.shape = }")
        print(f"[VVERBOSE] {kv_indices.shape = }")
        print(f"[VVERBOSE] {kv_last_page_len.shape = }")
        print(f"[VVERBOSE] {scale = }")

    # Compute scales and convert to FP8 if needed (before creating wrappers)
    q_scale, k_scale, v_scale = None, None, None
    q_scale_tensor, k_scale_tensor, v_scale_tensor = None, None, None
    kv_cache_sf = None
    # Separate K/V caches for cuDNN (which requires separate tensors, not combined kv_cache)
    k_cache_cudnn, v_cache_cudnn = k_cache, v_cache

    if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        q, q_scale_t = to_float8(q, q_dtype)
        q_scale = q_scale_t.item()
        q_scale_tensor = q_scale_t.reshape(1, 1, 1, 1)
        # o_data_type stays as q_dtype (FP8 output)

    if is_nvfp4_kv:
        kv_cache_nvfp4, kv_cache_sf, k_scale, v_scale = nvfp4_quantize_paged_kv_cache(
            kv_cache[:, 0], kv_cache[:, 1]
        )
        kv_cache = kv_cache_nvfp4
    elif kv_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        # Convert k_cache and v_cache to quantized dtype for cuDNN
        k_cache_cudnn, k_scale_t = to_float8(k_cache, kv_dtype)
        v_cache_cudnn, v_scale_t = to_float8(v_cache, kv_dtype)
        k_scale = k_scale_t.item()
        v_scale = v_scale_t.item()
        k_scale_tensor = k_scale_t.reshape(1, 1, 1, 1)
        v_scale_tensor = v_scale_t.reshape(1, 1, 1, 1)

        # Also convert the full kv_cache for non-cuDNN backends
        k_data, v_data = torch.chunk(kv_cache, 2, dim=1)
        k_quantized, _ = to_float8(k_data, kv_dtype)
        v_quantized, _ = to_float8(v_data, kv_dtype)
        kv_cache = torch.cat([k_quantized, v_quantized], dim=1)

    prims_ts_v_cache = None
    prims_ts_v_scale = v_scale
    if "prims-ts" in backends and v_dtype != kv_dtype:
        # prims-ts consumes an FP8 V while every other backend keeps kv_dtype.
        # Write the FP8-rounded values back into the shared cache (k_cache and
        # v_cache are views of it) so all backends and the reference see the
        # same V and --refcheck compares like with like.
        prims_ts_v_cache, prims_ts_v_scale_t = to_float8(
            kv_cache[:, 1].contiguous(), v_dtype
        )
        prims_ts_v_scale = prims_ts_v_scale_t.item()
        kv_cache[:, 1].copy_(prims_ts_v_cache.to(torch.float32) * prims_ts_v_scale)

    # Ensure trtllm-fmha-v2 sees contiguous HND-physical paged KV cache.
    # Skip if kv_cache is not a plain Tensor (e.g., NVFP4 packed tuple).
    # backend filter further down also drops trtllm-fmha-v2 in that case.
    if "trtllm-fmha-v2" in backends and isinstance(kv_cache, torch.Tensor):
        _fmha_v2_kv_cache = kv_cache.contiguous()
    else:
        _fmha_v2_kv_cache = kv_cache

    prims_ts_k_cache = None
    prims_ts_out = None
    prims_ts_sm_scale = None
    prims_ts_output_scale = None
    if "prims-ts" in backends:
        prims_ts = _get_prims_ts_module()
        prims_ts_k_cache = kv_cache[:, 0].contiguous()
        if prims_ts_v_cache is None:
            prims_ts_v_cache = kv_cache[:, 1].contiguous()
        prims_ts_out = torch.empty_like(q, dtype=o_data_type)
        backend_wrappers_prims_ts = prims_ts.BatchPrefillPagedTSWrapper("HND")
        _q_scale = q_scale if q_scale is not None else 1.0
        _k_scale = k_scale if k_scale is not None else 1.0
        _v_scale = prims_ts_v_scale if prims_ts_v_scale is not None else 1.0
        prims_ts_sm_scale = _q_scale * _k_scale * scale
        prims_ts_output_scale = _v_scale
        backend_wrappers_prims_ts.plan(
            device=device,
            batch_size=batch_size,
            max_seq_len_q=s_qo,
            max_kv_len=s_kv,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim_qk,
            q_dtype=q_dtype,
            k_dtype=kv_dtype,
            v_dtype=v_dtype,
            out_dtype=o_data_type,
            page_size=page_size,
            mask_type="causal" if causal else "dense",
            sm_scale=prims_ts_sm_scale,
            output_scale=prims_ts_output_scale,
            uniform_packed_lengths=not args.random_actual_seq_len,
            has_q_offset=causal and s_qo != s_kv,
            **({"window_left": args.window_left} if args.window_left >= 0 else {}),
        )

    # Mode kwargs are only passed when requested so default lines plan exactly
    # as before.
    plan_mode_kwargs = {}
    if args.window_left >= 0:
        plan_mode_kwargs["window_left"] = args.window_left
    if args.logits_soft_cap > 0:
        plan_mode_kwargs["logits_soft_cap"] = args.logits_soft_cap
    sinks = _make_sinks(args, num_qo_heads, device)
    trtllm_mode_kwargs = {}
    if sinks is not None:
        trtllm_mode_kwargs["sinks"] = sinks
    if args.skip_softmax_threshold_scale_factor is not None:
        trtllm_mode_kwargs["skip_softmax_threshold_scale_factor"] = (
            args.skip_softmax_threshold_scale_factor
        )
    if args.use_fp16_softmax:
        trtllm_mode_kwargs["use_fp16_softmax"] = True
    if args.uses_spcompress:
        trtllm_mode_kwargs["uses_spcompress"] = True

    # Prepare wrappers (after FP8 conversion so we have correct dtypes)
    backend_wrappers = {}
    resolved_backends = {}
    if "cute-dsl" in backends:
        # The CuTe DSL backend compiles at plan(); a failed compile skips it.
        cute_dsl_wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer,
            "HND",
            use_cuda_graph=is_cuda_graph_compatible,
            qo_indptr_buf=qo_indptr,
            paged_kv_indptr_buf=kv_indptr,
            paged_kv_indices_buf=kv_indices,
            paged_kv_last_page_len_buf=kv_last_page_len,
            backend="cute-dsl",
        )
        if _probe_backend(
            "cute-dsl",
            lambda: cute_dsl_wrapper.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim_qk,
                page_size,
                causal=causal,
                q_data_type=q_dtype,
                kv_data_type=kv_dtype,
                o_data_type=o_data_type,
                **plan_mode_kwargs,
            ),
        ):
            backend_wrappers["cute-dsl"] = cute_dsl_wrapper
            resolved_backends["cute-dsl"] = "cute-dsl"
        else:
            backends.remove("cute-dsl")
    if "batch-attention" in backends:
        # Holistic persistent kernel over the same paged batch. BatchAttention
        # owns a fixed-size workspace, so plan() rejects batches that overflow it.
        batch_attention = flashinfer.BatchAttention(kv_layout="HND", device=str(device))
        if _probe_backend(
            "batch-attention",
            lambda: batch_attention.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                actual_seq_lens_kv_device.flatten().int(),
                num_qo_heads,
                num_kv_heads,
                head_dim_qk,
                head_dim_vo,
                page_size,
                causal=causal,
                sm_scale=scale,
                logits_soft_cap=args.logits_soft_cap,
                q_data_type=q_dtype,
                kv_data_type=kv_dtype,
            ),
        ):
            backend_wrappers["batch-attention"] = batch_attention
            resolved_backends["batch-attention"] = "batch-attention"
        else:
            backends.remove("batch-attention")
    for backend in list(backends):
        if backend == "prims-ts":
            backend_wrappers[backend] = backend_wrappers_prims_ts
            resolved_backends[backend] = backend
            continue
        if backend in ("cute-dsl", "batch-attention"):
            continue
        if backend in ["fa2", "fa3", "auto"] and sinks is not None:
            # Attention sinks on fa2/fa3 are a JIT attention variant with its own
            # wrapper (flashinfer.attention.BatchAttentionWithAttentionSinkWrapper).
            backend_wrappers[backend] = (
                flashinfer.BatchAttentionWithAttentionSinkWrapper(
                    workspace_buffer,
                    "HND",
                    use_cuda_graph=is_cuda_graph_compatible
                    if backend != "fa2"
                    else False,
                    qo_indptr_buf=qo_indptr,
                    paged_kv_indptr_buf=kv_indptr,
                    paged_kv_indices_buf=kv_indices,
                    paged_kv_last_page_len_buf=kv_last_page_len,
                    backend=backend,
                    q_data_type=q_dtype,
                    kv_data_type=kv_dtype,
                    head_dim_qk=head_dim_qk,
                    head_dim_vo=head_dim_vo,
                    window_left=args.window_left,
                )
            )
            backend_wrappers[backend].plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim_qk,
                page_size,
                causal=causal,
                q_data_type=q_dtype,
                kv_data_type=kv_dtype,
                **plan_mode_kwargs,
            )
            resolved_backends[backend] = backend_wrappers[backend]._backend
        elif backend in [
            "fa2",
            "fa3",
            "auto",
            "trtllm-gen",
            "cute-dsl-prims",
        ]:
            backend_wrappers[backend] = (
                flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
                    workspace_buffer,
                    "HND",
                    use_cuda_graph=is_cuda_graph_compatible
                    if backend != "fa2"
                    else False,
                    qo_indptr_buf=qo_indptr,
                    paged_kv_indptr_buf=kv_indptr,
                    paged_kv_indices_buf=kv_indices,
                    paged_kv_last_page_len_buf=kv_last_page_len,
                    backend=backend,
                )
            )
            backend_wrappers[backend].plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim_qk,
                page_size,
                pos_encoding_mode="NONE",
                causal=causal,
                q_data_type=q_dtype,
                kv_data_type=kv_dtype,
                o_data_type=o_data_type,
                block_tables=block_tables,
                **plan_mode_kwargs,
            )
            resolved_backends[backend] = backend_wrappers[backend]._backend
        elif backend == "cudnn":
            # K/V cache tensors use [pages, heads, page_size, dim] (HND).
            backend_wrappers[backend] = (
                flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
                    workspace_buffer,
                    "HND",
                    backend="cudnn",
                )
            )
            plan_call = functools.partial(
                backend_wrappers["cudnn"].plan,
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim_qk,
                page_size,
                pos_encoding_mode="NONE",
                causal=causal,
                q_data_type=q_dtype,
                o_data_type=o_data_type,
                seq_lens=actual_seq_lens_kv_device,
                seq_lens_q=actual_seq_lens_q_device,
                sm_scale=scale,
                max_token_per_sequence=s_qo,
                max_sequence_kv=s_kv,
                block_tables=block_tables,
            )
            # The cuDNN plan rejects some installs (e.g. "Mixed-form sequence
            # lengths ... require cuDNN 9.25.0"); skip it instead of losing the
            # other backends of the line.
            if not _probe_backend(backend, plan_call):
                backends.remove(backend)
                continue
            resolved_backends[backend] = backend_wrappers[backend]._backend
        else:
            resolved_backends[backend] = backend

    backend_outputs = {
        backend: prims_ts_out
        if backend == "prims-ts"
        else torch.empty(
            q.shape[0],
            num_qo_heads,
            head_dim_vo,
            device=device,
            dtype=o_data_type,
        )
        for backend in backends
    }

    def run_backend_wrapper(
        backend,
        q,
        kv_cache,
        k_cache,
        v_cache,
        workspace_buffer,
        block_tables,
        actual_seq_lens_q_device,
        actual_seq_lens_kv_device,
        q_indptr,
        qo_indptr,
        kv_indptr,
        kv_token_indptr,
        out,
    ):
        if backend in ["fa2", "fa3", "auto"] and sinks is not None:
            # BatchAttentionWithAttentionSinkWrapper takes (sink, sm_scale) as
            # its JIT variant's extra run arguments.
            return backend_wrappers[backend].run(
                q,
                kv_cache,
                sinks,
                scale,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                enable_pdl=args.enable_pdl,
                out=out,
            )
        if backend in ["fa2", "fa3", "auto", "trtllm-gen", "cute-dsl"]:
            return backend_wrappers[backend].run(
                q,
                kv_cache,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                kv_cache_sf=kv_cache_sf,
                enable_pdl=args.enable_pdl,
                out=out,
                **(trtllm_mode_kwargs if backend == "trtllm-gen" else {}),
            )
        elif backend == "batch-attention":
            return backend_wrappers[backend].run(
                q, kv_cache, out=out, logits_soft_cap=args.logits_soft_cap
            )[0]
        elif backend == "cute-dsl-prims":
            return backend_wrappers[backend].run(
                q,
                kv_cache,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        elif backend == "cudnn":
            # cuDNN uses wrapper API with tensor scales for FP8
            return backend_wrappers[backend].run(
                q,
                (k_cache, v_cache),
                q_scale=q_scale_tensor,
                k_scale=k_scale_tensor,
                v_scale=v_scale_tensor,
                enable_pdl=args.enable_pdl,
                out=out,
            )
        elif backend == "trtllm-native":
            # Compute combined bmm1_scale: q_scale * k_scale * sm_scale
            # For FP8: all scales are float values
            _q_scale = q_scale if q_scale is not None else 1.0
            _k_scale = k_scale if k_scale is not None else 1.0
            _v_scale = v_scale if v_scale is not None else 1.0
            bmm1_scale = _q_scale * _k_scale * scale
            bmm2_scale = _v_scale
            native_mode_kwargs = dict(trtllm_mode_kwargs)
            if args.window_left >= 0:
                native_mode_kwargs["window_left"] = args.window_left
            return flashinfer.prefill.trtllm_batch_context_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=workspace_buffer,
                block_tables=block_tables,
                seq_lens=actual_seq_lens_kv_device,
                max_q_len=s_qo,
                max_kv_len=s_kv,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                batch_size=batch_size,
                cum_seq_lens_q=qo_indptr,
                cum_seq_lens_kv=kv_token_indptr,
                causal=causal,
                kv_cache_sf=kv_cache_sf,
                enable_pdl=args.enable_pdl,
                out=out,
                out_dtype=o_data_type,
                **native_mode_kwargs,
            )
        elif backend == "cudnn-native":
            # Direct cudnn_batch_prefill_with_kv_cache call (similar to trtllm-native)
            return flashinfer.prefill.cudnn_batch_prefill_with_kv_cache(
                q,
                k_cache,
                v_cache,
                scale,
                workspace_buffer,
                max_token_per_sequence=s_qo,
                max_sequence_kv=s_kv,
                actual_seq_lens_q=actual_seq_lens_q_device,
                actual_seq_lens_kv=actual_seq_lens_kv_device,
                block_tables=block_tables,
                causal=causal,
                return_lse=True,
                is_cuda_graph_compatible=is_cuda_graph_compatible,
                batch_offsets_q=q_indptr,
                batch_offsets_o=q_indptr,
                q_scale=q_scale_tensor,
                k_scale=k_scale_tensor,
                v_scale=v_scale_tensor,
                o_data_type=o_data_type,
                out=out,
            )[0]
        elif backend == "trtllm-fmha-v2":
            _q_scale = q_scale if q_scale is not None else 1.0
            _k_scale = k_scale if k_scale is not None else 1.0
            _fmha_v2_bmm2_scale = v_scale if v_scale is not None else 1.0
            return trtllm_fmha_v2_prefill(
                qkv=(q, kv_cache),
                input_layout="Q_PAGED_KV_HND",
                workspace_buffer=workspace_buffer,
                seq_lens=actual_seq_lens_kv_device.flatten(),
                max_q_len=s_qo,
                max_kv_len=s_kv,
                bmm1_scale=_q_scale * _k_scale * scale,
                bmm2_scale=_fmha_v2_bmm2_scale,
                batch_size=batch_size,
                cum_seq_lens_q=qo_indptr,
                cum_seq_lens_kv=kv_token_indptr,
                block_tables=block_tables,
                mask_mode="causal" if causal else "padding",
                out=out,
                out_dtype=o_data_type,
                **({"window_left": args.window_left} if args.window_left >= 0 else {}),
            )
        elif backend == "prims-ts":
            return backend_wrappers[backend].run(
                q,
                k_cache,
                v_cache,
                qo_indptr,
                block_tables,
                actual_seq_lens_kv_device.flatten(),
                out=out,
                validate=False,
            )
        else:
            print(f"[ERROR] Backend {backend} not supported")
            return None

    has_reference_output = False
    reference_backend = None
    # Iterate over each backend:
    for cur_backend in backends:
        # Clear workspace buffer to prevent unexpected interactions between backends.
        workspace_buffer.zero_()
        if cur_backend == "prims-ts":
            runtime_k_cache = prims_ts_k_cache
            runtime_v_cache = prims_ts_v_cache
        elif cur_backend in ("cudnn", "cudnn-native"):
            runtime_k_cache = k_cache_cudnn
            runtime_v_cache = v_cache_cudnn
        else:
            runtime_k_cache = None
            runtime_v_cache = None
        runtime_out = backend_outputs[cur_backend]
        runtime_kv_cache = (
            None
            if cur_backend == "prims-ts"
            else _fmha_v2_kv_cache
            if cur_backend == "trtllm-fmha-v2"
            else kv_cache
        )
        runtime_workspace = None if cur_backend == "prims-ts" else workspace_buffer
        if cur_backend in ("cute-dsl", "cudnn", "cudnn-native") and not _probe_backend(
            cur_backend,
            lambda: run_backend_wrapper(
                cur_backend,
                q,
                runtime_kv_cache,
                runtime_k_cache,
                runtime_v_cache,
                runtime_workspace,
                block_tables,
                actual_seq_lens_q_device,
                actual_seq_lens_kv_device,
                q_indptr,
                qo_indptr,
                kv_indptr,
                kv_token_indptr,
                runtime_out,
            ),
        ):
            continue
        if run_refcheck:
            outputs[cur_backend] = (
                run_backend_wrapper(
                    cur_backend,
                    q,
                    runtime_kv_cache,
                    runtime_k_cache,
                    runtime_v_cache,
                    runtime_workspace,
                    block_tables,
                    actual_seq_lens_q_device,
                    actual_seq_lens_kv_device,
                    q_indptr,
                    qo_indptr,
                    kv_indptr,
                    kv_token_indptr,
                    runtime_out,
                )
                .detach()
                .clone()
            )
            if cur_backend == "fa2":
                has_reference_output = True
                reference_output = outputs[cur_backend]
                reference_backend = "fa2"

        def run_timed_backend(q_arg, kv_arg, k_arg, v_arg, out_arg):
            return run_backend_wrapper(
                cur_backend,
                q_arg,
                kv_arg,
                k_arg,
                v_arg,
                workspace_buffer,
                block_tables,
                actual_seq_lens_q_device,
                actual_seq_lens_kv_device,
                q_indptr,
                qo_indptr,
                kv_indptr,
                kv_token_indptr,
                out_arg,
            )

        backend_times[cur_backend] = bench_gpu_time(
            fn=run_timed_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=False,
            enable_cupti=args.use_cupti,
            use_cuda_graph=(is_cuda_graph_compatible and cur_backend != "fa2"),
            cold_l2_cache=True,
            input_args=(
                q,
                runtime_kv_cache,
                runtime_k_cache,
                runtime_v_cache,
                runtime_out,
            ),
        )

    # Perform reference check
    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())

    if run_refcheck and "prims-ts" in outputs:
        sample_count, max_abs_error = _validate_prims_ts_context_samples(
            q=q,
            k=prims_ts_k_cache,
            v=prims_ts_v_cache,
            out=outputs["prims-ts"],
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            sm_scale=prims_ts_sm_scale,
            output_scale=prims_ts_output_scale,
            causal=causal,
            paged_kv_indices=kv_indices,
            kv_lens=actual_seq_lens_kv_device,
        )
        if args.verbose >= 1:
            print(
                "[INFO] prims-ts sampled FP32 context reference passed: "
                f"{sample_count} samples, max_abs_error={max_abs_error:.6g}"
            )
        if is_cuda_graph_compatible:
            graph_output = _replay_cuda_graph_once(
                lambda: backend_wrappers["prims-ts"].run(
                    q,
                    prims_ts_k_cache,
                    prims_ts_v_cache,
                    qo_indptr,
                    block_tables,
                    actual_seq_lens_kv_device.flatten(),
                    out=prims_ts_out,
                    validate=False,
                ),
                prims_ts_out,
            )
            graph_sample_count, graph_max_abs_error = (
                _validate_prims_ts_context_samples(
                    q=q,
                    k=prims_ts_k_cache,
                    v=prims_ts_v_cache,
                    out=graph_output,
                    qo_indptr=qo_indptr,
                    kv_indptr=kv_indptr,
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    sm_scale=prims_ts_sm_scale,
                    output_scale=prims_ts_output_scale,
                    causal=causal,
                    paged_kv_indices=kv_indices,
                    kv_lens=actual_seq_lens_kv_device,
                )
            )
            if args.verbose >= 1:
                print(
                    "[INFO] prims-ts CUDA-graph replay reference passed: "
                    f"{graph_sample_count} samples, "
                    f"max_abs_error={graph_max_abs_error:.6g}"
                )

    # When cases where FA2 is not available, try to find an alternative reference
    # Priority: cudnn > cudnn-native > trtllm-gen > trtllm-native > trtllm-fmha-v2
    if run_refcheck and not has_reference_output and len(tested_backends) > 1:
        reference_priority = [
            "cudnn",
            "cudnn-native",
            "trtllm-gen",
            "trtllm-native",
            "trtllm-fmha-v2",
            "auto",
            "prims-ts",
            "cute-dsl",
        ]
        for candidate in reference_priority:
            if candidate in tested_backends:
                has_reference_output = True
                reference_backend = candidate
                reference_output = outputs[candidate]
                if args.verbose >= 1:
                    print(
                        f"[INFO] FA2 not available for reference. Using {candidate} as reference backend for cross-comparison."
                    )
                break

    if len(tested_backends) > 1:
        if run_refcheck and has_reference_output:
            if reference_output.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                if args.verbose >= 2:
                    print(
                        "[VVERBOSE] Reference output is FP8. Converting to float32 for reference check."
                    )
                reference_output = reference_output.to(torch.float32)
                tested_outputs = [output.to(torch.float32) for output in tested_outputs]
            for i in range(len(tested_backends)):
                (
                    num_different_elements,
                    num_elements,
                    num_different_elements_percentage,
                ) = is_close_stats(reference_output, tested_outputs[i], rtol, atol)
                if num_different_elements > 0:
                    print(
                        f"[ERROR] Output tensor mismatch between backends {reference_backend} and {tested_backends[i]}: "
                        f"{num_different_elements} / {num_elements} ({num_different_elements_percentage:.2f}%) elements are different"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {tested_backends[i]} output mismatch"
                        )

    # Compute perf metrics
    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            actual_seq_lens_q_flat = actual_seq_lens_q.flatten().to("cpu")
            actual_seq_lens_kv_flat = actual_seq_lens_kv.flatten().to("cpu")
            # With --window_left the metrics count only the in-window keys.
            metric_q, metric_kv, metric_causal, metric_bytes_kv = _metric_seq_lens(
                args, actual_seq_lens_q_flat, actual_seq_lens_kv_flat, causal
            )
            tflops = attention_tflops_per_sec_with_actual_seq_lens(
                metric_q,
                metric_kv,
                head_dim_qk,
                head_dim_vo,
                num_qo_heads,
                metric_causal,
                median_time,
            )
            tb_per_sec = attention_tb_per_sec_with_actual_seq_lens(
                metric_q,
                metric_bytes_kv,
                head_dim_qk,
                head_dim_vo,
                num_qo_heads,
                num_kv_heads,
                median_time,
                q_dtype=q_dtype,
                k_dtype=kv_dtype,
                v_dtype=v_dtype if backend == "prims-ts" else kv_dtype,
                o_dtype=o_data_type,
            )
            resolved_backend = resolved_backends.get(backend, backend)
            display_backend = (
                f"auto({resolved_backend})" if backend == "auto" else backend
            )
            print_perf_metrics(
                display_backend, median_time, std_time, tflops, tb_per_sec
            )

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["backend"] = backend
                cur_res["resolved_backend"] = resolved_backend
                cur_res["page_size"] = page_size
                cur_res["batch_size"] = batch_size
                cur_res["s_qo"] = s_qo
                cur_res["s_kv"] = s_kv
                cur_res["num_qo_heads"] = num_qo_heads
                cur_res["num_kv_heads"] = num_kv_heads
                cur_res["head_dim_qk"] = head_dim_qk
                cur_res["head_dim_vo"] = head_dim_vo
                cur_res["causal"] = causal
                cur_res["q_dtype"] = q_dtype
                cur_res["kv_dtype"] = kv_dtype
                cur_res["v_dtype"] = v_dtype if backend == "prims-ts" else kv_dtype
                cur_res["out_dtype"] = o_data_type
                cur_res["avg_actual_seq_len"] = avg_seq_len_q
                cur_res["random_actual_seq_len"] = args.random_actual_seq_len
                cur_res["case_tag"] = args.case_tag
                _record_mode_columns(cur_res, args)
                res.append(cur_res)
    return res


def testBatchPrefillWithRaggedKVCacheWrapper(args):
    """
    Test BatchPrefillWithRaggedKVCacheWrapper API and equivalent cuDNN API.
    Supports fa2, fa3, cutlass, cudnn, trtllm-native, trtllm-fmha-v2, and
    prims-ts backends.

    This test:
    1. Creates ragged KV cache and query tensors for prefill
    2. Runs prefill attention with different backends
    3. Verifies outputs match between backends (if refcheck enabled)
    4. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: Dictionary containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testBatchPrefillWithRaggedKVCacheWrapper")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    # Basic setup
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    q_init_dtype = torch.bfloat16
    kv_init_dtype = torch.bfloat16
    rtol = 2e-1
    atol = 1e-2
    res = []

    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    if q_dtype not in [
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ]:
        print(f"[ERROR] Unsupported q_dtype: {args.q_dtype}")
        return res
    q_init_dtype = torch.float16 if q_dtype == torch.float16 else torch.bfloat16
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    if kv_dtype not in [
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ]:
        print(f"[ERROR] Unsupported kv_dtype: {args.kv_dtype}")
        return res
    kv_init_dtype = torch.float16 if kv_dtype == torch.float16 else torch.bfloat16
    out_dtype = dtype_str_to_torch_dtype(args.out_dtype) if args.out_dtype else q_dtype

    # Parse and validate backend configurations
    backends = args.backends
    batch_size = args.batch_size
    s_qo = args.s_qo
    s_kv = args.s_kv
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim_qk = args.head_dim_qk
    head_dim_vo = args.head_dim_vo if args.head_dim_vo is not None else head_dim_qk
    causal = args.causal
    is_cuda_graph_compatible = not args.no_cuda_graph
    # return_lse = not args.no_lse # TO-DO: Add support for this
    run_refcheck = args.refcheck
    row_activity_mode = args.row_activity_mode or "cpu_mirror"

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    # Check for backend-specific constraints
    if "fa2" in backends:
        remove_fa2 = False
        if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            print("[INFO] FA2 backend does not support FP8. Skipping.")
            remove_fa2 = True
        if remove_fa2:
            backends.remove("fa2")
    if "cudnn" in backends:
        remove_cudnn = False
        if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] or kv_dtype in [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]:
            print("[INFO] CUDNN backend does not support FP8. Skipping.")
            remove_cudnn = True
        if remove_cudnn:
            backends.remove("cudnn")

    if "cudnn-native" in backends:
        remove_cudnn_native = False
        if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] or kv_dtype in [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]:
            print("[INFO] CUDNN-native backend does not support FP8. Skipping.")
            remove_cudnn_native = True
        if remove_cudnn_native:
            backends.remove("cudnn-native")

    if "cutlass" in backends:
        remove_cutlass = False
        if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] or kv_dtype in [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]:
            print("[INFO] CUTLASS backend does not support FP8. Skipping.")
            remove_cutlass = True
        if not (
            (head_dim_qk == 128 and head_dim_qk == head_dim_vo) or head_dim_qk == 192
        ):
            print("[INFO] CUTLASS backend requires head dimension to be 128 or 192")
            remove_cutlass = True
        if remove_cutlass:
            backends.remove("cutlass")

    if "trtllm-gen" in backends:
        print("[INFO] trtllm-gen backend does not support ragged prefill. Skipping.")
        remove_trtllm = True
        if remove_trtllm:
            backends.remove("trtllm-gen")
    if "trtllm-native" in backends:
        remove_trtllm_native = False
        if not (head_dim_qk == 192 and head_dim_vo == 128) and not (
            head_dim_qk == 128 and head_dim_vo == 128
        ):
            print(
                "[INFO] trtllm-native backend requires head_dim_qk == 192 and head_dim_vo == 128 or head_dim_qk == 128 and head_dim_vo == 128. Skipping."
            )
            remove_trtllm_native = True
        if remove_trtllm_native:
            backends.remove("trtllm-native")

    fmha_v2_layout = None
    if "trtllm-fmha-v2" in backends:
        same_token_count = s_qo == s_kv
        same_head_dim = head_dim_qk == head_dim_vo
        if same_token_count and same_head_dim:
            if num_qo_heads == num_kv_heads:
                fmha_v2_layout = "PACKED_QKV"
            else:
                fmha_v2_layout = "CONTIGUOUS_Q_KV"
        else:
            fp8_requested = (
                q_dtype == torch.float8_e4m3fn or kv_dtype == torch.float8_e4m3fn
            )
            if is_sm12x_supported(device):
                print(
                    "[INFO] trtllm-fmha-v2 backend has no compatible input layout "
                    f"on SM12x for s_qo={s_qo} != s_kv={s_kv} or "
                    f"head_dim_qk={head_dim_qk} != head_dim_vo={head_dim_vo} "
                    "(SEPARATE_Q_K_V is not compiled for SM12x). Skipping."
                )
                backends.remove("trtllm-fmha-v2")
            elif fp8_requested:
                print(
                    "[INFO] trtllm-fmha-v2 backend does not support FP8 with the "
                    "SEPARATE_Q_K_V layout (required by s_qo != s_kv or "
                    "head_dim_qk != head_dim_vo). Skipping."
                )
                backends.remove("trtllm-fmha-v2")
            else:
                fmha_v2_layout = "SEPARATE_Q_K_V"

    if "prims-ts" in backends:
        if q_dtype != kv_dtype:
            _drop_backend(backends, "prims-ts", "requires matching Q and K/V dtypes")
        elif q_dtype not in (
            torch.bfloat16,
            torch.float16,
            torch.float8_e4m3fn,
        ):
            _drop_backend(
                backends,
                "prims-ts",
                "supports FP16, BF16, and FP8 E4M3 inputs only",
            )
        elif (head_dim_qk, head_dim_vo) not in (
            (128, 128),
            (192, 128),
            (192, 192),
            (256, 256),
        ):
            _drop_backend(
                backends,
                "prims-ts",
                "requires QK/VO head dimensions (128,128), (192,128), (192,192), or (256,256)",
            )
        elif num_qo_heads % num_kv_heads != 0:
            _drop_backend(backends, "prims-ts", "requires Hq to be divisible by Hkv")
        elif args.enable_pdl:
            print("[WARNING] prims-ts does not expose PDL; ignoring --enable_pdl.")

    if "cute-dsl" in backends and (head_dim_qk, head_dim_vo) not in (
        (128, 128),
        (192, 128),
        (256, 256),
    ):
        # trtllm_ragged_attention_deepseek(backend="cute-dsl") FMHA head dims (prefill.py)
        _drop_backend(
            backends,
            "cute-dsl",
            "requires head dims (128,128), (192,128), or (256,256)",
        )
    if "auto" in backends and q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        _drop_backend(
            backends, "auto", "may resolve to FA2, which does not support FP8"
        )
    if "cutile" in backends:
        # cuTile ragged prefill (prefill.py): qo_indptr == kv_indptr only
        if s_qo != s_kv or q_dtype not in (torch.float16, torch.bfloat16):
            _drop_backend(
                backends, "cutile", "requires equal q/kv lengths and fp16/bf16 inputs"
            )
    _drop_backends_without_mode(
        backends,
        args,
        window_ok=("fa2", "fa3", "auto", "trtllm-native", "cute-dsl", "prims-ts"),
        soft_cap_ok=("fa2", "fa3", "auto"),
        sinks_ok=("trtllm-native", "cute-dsl"),
    )
    if args.window_left >= 0 and not causal:
        _drop_backend(backends, "prims-ts", "requires causal masking for a window")
    _drop_backends_without_trtllm_knobs(backends, args, device, ("trtllm-native",))

    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    prims_ts_only = set(backends) == {"prims-ts"}
    if (
        q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
        and args.out_dtype is None
        and not prims_ts_only
    ):
        print(
            "[ERROR] --out_dtype must be set to bfloat16 or float16 for FP8 "
            "ragged prefill unless prims-ts is the only effective backend."
        )
        return res

    supported_out_dtypes = [torch.bfloat16, torch.float16]
    if prims_ts_only:
        supported_out_dtypes.append(torch.float8_e4m3fn)
    if out_dtype not in supported_out_dtypes:
        print(f"[ERROR] Unsupported out_dtype: {args.out_dtype}")
        return res

    # Check for layer-specific constraints
    layer_not_supported = False
    if s_qo > s_kv:
        print("[ERROR] s_qo > s_kv is not supported. Exiting.")
        layer_not_supported = True
    if layer_not_supported:
        print("[ERROR] Layer not supported. Exiting.")
        return res

    backend_times = {backend: [] for backend in backends}
    outputs = {}

    # Sample sequence lengths.
    # If s_qo == s_kv, then make sampled actual_seq_lens_kv the same as actual_seq_lens_q.
    # IF s_qo < s_kv, then sample actual_seq_lens_kv separately. Then ensure actual_seq_lens_kv is at least as long as actual_seq_lens_q.
    actual_seq_lens_q = sample_actual_seq_lens(
        s_qo, batch_size, None, args.random_actual_seq_len
    )
    if s_qo == s_kv:
        if args.verbose >= 2:
            print(
                "[VVERBOSE] s_qo == s_kv, making actual_seq_lens_kv the same as actual_seq_lens_q"
            )
        actual_seq_lens_kv = actual_seq_lens_q.clone()
    else:  # s_qo < s_kv
        if args.verbose >= 2:
            print("[VVERBOSE] s_qo < s_kv, sampling actual_seq_lens_kv")
        actual_seq_lens_kv = sample_actual_seq_lens(
            s_kv, batch_size, None, args.random_actual_seq_len
        )
        actual_seq_lens_kv = torch.maximum(actual_seq_lens_kv, actual_seq_lens_q)

    avg_seq_len_q = actual_seq_lens_q.sum().item() // batch_size
    avg_seq_len_kv = actual_seq_lens_kv.sum().item() // batch_size
    if args.verbose >= 1:
        print(f"[VERBOSE] Average actual qo seq len: {avg_seq_len_q}")
        print(f"[VERBOSE] Average actual kv seq len: {avg_seq_len_kv}")
    if args.verbose >= 2:
        print(f"[VVERBOSE] {actual_seq_lens_q.flatten() = }")
        print(f"[VVERBOSE] {actual_seq_lens_kv.flatten() = }")

    cumsum_s_qo = torch.sum(actual_seq_lens_q)
    cumsum_s_kv = torch.sum(actual_seq_lens_kv)

    q = torch.randn(
        cumsum_s_qo,
        num_qo_heads,
        head_dim_qk,
        device=device,
        dtype=q_init_dtype,
    )
    if args.verbose >= 2:
        print(f"[VVERBOSE] {q.shape = }")

    k = torch.randn(
        cumsum_s_kv,
        num_kv_heads,
        head_dim_qk,
        device=device,
        dtype=kv_init_dtype,
    )
    v = torch.randn(
        cumsum_s_kv,
        num_kv_heads,
        head_dim_vo,
        device=device,
        dtype=kv_init_dtype,
    )

    block_tables = None

    ## The following are for BatchPrefillWithRaggedKVCacheWrapper
    actual_seq_lens_q_device = actual_seq_lens_q.to(device)
    actual_seq_lens_kv_device = actual_seq_lens_kv.to(device)
    # CPU mirrors for trtllm_ragged_attention_deepseek CUDA graph capture:
    # its capture path requires per-row lengths without a device sync.
    actual_seq_lens_q_cpu_flat = (
        actual_seq_lens_q.reshape(-1).to(torch.int32).cpu().contiguous()
    )
    actual_seq_lens_kv_cpu_flat = (
        actual_seq_lens_kv.reshape(-1).to(torch.int32).cpu().contiguous()
    )

    q_indptr = (
        torch.cat(
            [
                torch.tensor([0], device=device),
                torch.cumsum(actual_seq_lens_q_device.view(-1), dim=0)
                * head_dim_qk
                * num_qo_heads,
            ]
        )
        .long()
        .to(device)
    )  # Element-unit offsets, for the low-level cudnn-native call only

    k_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_kv_device.view(-1), dim=0)
            * head_dim_qk
            * num_kv_heads,
        ]
    ).long()

    v_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_kv_device.view(-1), dim=0)
            * head_dim_vo
            * num_kv_heads,
        ]
    ).long()

    o_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q_device.view(-1), dim=0)
            * head_dim_vo
            * num_qo_heads,
        ]
    ).long()

    batch_offsets_stats = torch.cat(
        [
            torch.zeros(
                1,
                device=actual_seq_lens_q_device.device,
                dtype=actual_seq_lens_q_device.dtype,
            ),
            torch.cumsum(actual_seq_lens_q_device.flatten(), dim=0) * num_qo_heads,
        ]
    ).to(device)

    qo_indptr = (
        torch.cat(
            [
                torch.tensor([0], device=device),
                torch.cumsum(actual_seq_lens_q_device.view(-1), dim=0),
            ]
        )
        .int()
        .to(device)
    )
    # Because actual_seq_lens_kv is the same as actual_seq_lens_q, kv_indptr will become the same as qo_indptr
    kv_indptr = (
        torch.cat(
            [
                torch.tensor([0], device=device),
                torch.cumsum(actual_seq_lens_kv_device.view(-1), dim=0),
            ]
        )
        .int()
        .to(device)
    )

    scale = float(1.0 / (head_dim_qk**0.5))
    workspace_buffer = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {k.shape = }")
        print(f"[VVERBOSE] {v.shape = }")
        print(f"[VVERBOSE] {qo_indptr.shape = }")
        print(f"[VVERBOSE] {kv_indptr.shape = }")
        print(f"[VVERBOSE] {scale = }")

    # Mode kwargs are only passed when requested so default lines plan exactly
    # as before.
    plan_mode_kwargs = {}
    if args.window_left >= 0:
        plan_mode_kwargs["window_left"] = args.window_left
    if args.logits_soft_cap > 0:
        plan_mode_kwargs["logits_soft_cap"] = args.logits_soft_cap
    sinks = _make_sinks(args, num_qo_heads, device)
    trtllm_mode_kwargs = {}
    if args.skip_softmax_threshold_scale_factor is not None:
        trtllm_mode_kwargs["skip_softmax_threshold_scale_factor"] = (
            args.skip_softmax_threshold_scale_factor
        )
    if args.use_fp16_softmax:
        trtllm_mode_kwargs["use_fp16_softmax"] = True
    if args.uses_spcompress:
        trtllm_mode_kwargs["uses_spcompress"] = True

    # Prepare wrappers
    backend_wrappers = {}
    resolved_backends = {}
    for backend in backends:
        if backend in [
            "cutlass",
            "fa2",
            "fa3",
            "trtllm-gen",
            "cute-dsl-prims",
            "auto",
            "cutile",
        ]:
            backend_wrappers[backend] = (
                flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
                    workspace_buffer,
                    "NHD",
                    use_cuda_graph=is_cuda_graph_compatible
                    if backend not in ("fa2", "cutlass")
                    else False,
                    qo_indptr_buf=qo_indptr,
                    kv_indptr_buf=kv_indptr,
                    backend=backend,
                )
            )
            backend_wrappers[backend].plan(
                qo_indptr,
                kv_indptr,
                num_qo_heads,
                num_kv_heads,
                head_dim_qk,
                head_dim_vo=head_dim_vo,
                causal=causal,
                q_data_type=q_dtype,
                kv_data_type=kv_dtype,
                o_data_type=out_dtype,
                **plan_mode_kwargs,
            )
            resolved_backends[backend] = backend_wrappers[backend]._backend
        elif backend == "cudnn":
            # cuDNN uses NHD layout and the wrapper API
            backend_wrappers[backend] = (
                flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
                    workspace_buffer,
                    "NHD",
                    backend="cudnn",
                )
            )
            backend_wrappers[backend].plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim_qk,
                head_dim_vo=head_dim_vo,
                causal=causal,
                sm_scale=scale,
                q_data_type=q_dtype,
                kv_data_type=kv_dtype,
                o_data_type=out_dtype,
                seq_lens=actual_seq_lens_kv_device,
                seq_lens_q=actual_seq_lens_q_device,
                max_token_per_sequence=s_qo,
                max_sequence_kv=s_kv,
            )

    q_scale, k_scale, v_scale = None, None, None
    if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        q_scale = q.abs().amax().item() / 256
        q = (q / q_scale).to(q_dtype)
    if kv_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        k_scale = k.abs().amax().item() / 256
        v_scale = v.abs().amax().item() / 256
        k = (k / k_scale).to(kv_dtype)
        v = (v / v_scale).to(kv_dtype)

    prims_ts_out = None
    prims_ts_sm_scale = None
    prims_ts_output_scale = None
    if "prims-ts" in backends:
        prims_ts = _get_prims_ts_module()
        prims_ts_out = torch.empty(
            q.shape[0],
            q.shape[1],
            head_dim_vo,
            device=q.device,
            dtype=out_dtype,
        )
        _q_scale = q_scale if q_scale is not None else 1.0
        _k_scale = k_scale if k_scale is not None else 1.0
        _v_scale = v_scale if v_scale is not None else 1.0
        prims_ts_sm_scale = _q_scale * _k_scale * scale
        prims_ts_output_scale = _v_scale
        backend_wrappers["prims-ts"] = prims_ts.BatchPrefillTSWrapper()
        backend_wrappers["prims-ts"].plan(
            device=q.device,
            batch_size=batch_size,
            max_seq_len_q=s_qo,
            max_kv_len=s_kv,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim_qk,
            head_dim_vo=head_dim_vo,
            q_dtype=q.dtype,
            k_dtype=k.dtype,
            packed=True,
            mask_type="causal" if causal else "dense",
            **({"window_left": args.window_left} if args.window_left >= 0 else {}),
            sm_scale=prims_ts_sm_scale,
            output_scale=prims_ts_output_scale,
            out_dtype=out_dtype,
        )

    # Build the input argument for trtllm-fmha-v2 once, in whichever layout was
    # selected during backend filtering. Done after FP8 quantization so the
    # stacked tensor inherits the final dtype.
    fmha_v2_qkv = None
    if "trtllm-fmha-v2" in backends:
        if fmha_v2_layout == "PACKED_QKV":
            fmha_v2_qkv = torch.stack([q, k, v], dim=1)
        elif fmha_v2_layout == "CONTIGUOUS_Q_KV":
            fmha_v2_qkv = (q, torch.stack([k, v], dim=1))
        else:
            fmha_v2_qkv = (q, k, v)

    backend_outputs = {
        backend: prims_ts_out
        if backend == "prims-ts"
        else torch.empty(
            q.shape[0],
            q.shape[1],
            v.shape[2],
            device=q.device,
            dtype=out_dtype,
        )
        for backend in backends
    }

    def run_backend_wrapper(
        backend,
        q,
        k,
        v,
        workspace_buffer,
        block_tables,
        actual_seq_lens_q_device,
        actual_seq_lens_kv_device,
        q_indptr,
        k_indptr,
        v_indptr,
        o_indptr,
        batch_offsets_stats,
        qo_indptr,
        kv_indptr,
        out,
    ):
        if backend in ["cutlass", "fa2", "fa3", "trtllm-gen", "auto", "cutile"]:
            return backend_wrappers[backend].run_return_lse(
                q, k, v, enable_pdl=args.enable_pdl, out=out
            )[0]
        elif backend == "cute-dsl-prims":
            return backend_wrappers[backend].run(
                q,
                k,
                v,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        elif backend == "cute-dsl":
            _q_scale = q_scale if q_scale is not None else 1.0
            _k_scale = k_scale if k_scale is not None else 1.0
            _v_scale = v_scale if v_scale is not None else 1.0
            return flashinfer.prefill.trtllm_ragged_attention_deepseek(
                query=q,
                key=k,
                value=v,
                workspace_buffer=workspace_buffer,
                seq_lens=actual_seq_lens_kv_device,
                max_q_len=s_qo,
                max_kv_len=s_kv,
                bmm1_scale=_q_scale * _k_scale * scale,
                bmm2_scale=_v_scale,
                o_sf_scale=-1,
                batch_size=batch_size,
                window_left=args.window_left,
                cum_seq_lens_q=qo_indptr,
                cum_seq_lens_kv=kv_indptr,
                enable_pdl=args.enable_pdl,
                is_causal=causal,
                return_lse=True,
                attention_sinks=sinks,
                out=out,
                backend="cute-dsl",
                q_seq_lens_cpu=actual_seq_lens_q_cpu_flat,
                kv_seq_lens_cpu=actual_seq_lens_kv_cpu_flat,
            )[0]
        elif backend == "cudnn":
            # cuDNN uses wrapper API
            return backend_wrappers[backend].run(
                q, k, v, enable_pdl=args.enable_pdl, out=out
            )
        elif backend == "cudnn-native":
            # Direct cudnn_batch_prefill_with_kv_cache call
            return flashinfer.prefill.cudnn_batch_prefill_with_kv_cache(
                q,
                k,
                v,
                scale,
                workspace_buffer,
                max_token_per_sequence=s_qo,
                max_sequence_kv=s_kv,
                actual_seq_lens_q=actual_seq_lens_q_device,
                actual_seq_lens_kv=actual_seq_lens_kv_device,
                block_tables=block_tables,
                causal=causal,
                return_lse=True,
                batch_offsets_q=q_indptr,
                batch_offsets_k=k_indptr,
                batch_offsets_v=v_indptr,
                batch_offsets_o=o_indptr,
                batch_offsets_stats=batch_offsets_stats,
                is_cuda_graph_compatible=True,
                o_data_type=out_dtype,
                out=out,
            )[0]
        elif backend == "trtllm-native":
            _q_scale = q_scale if q_scale is not None else 1.0
            _k_scale = k_scale if k_scale is not None else 1.0
            _v_scale = v_scale if v_scale is not None else 1.0
            return flashinfer.prefill.trtllm_ragged_attention_deepseek(
                query=q,
                key=k,
                value=v,
                workspace_buffer=workspace_buffer,
                seq_lens=actual_seq_lens_kv_device,
                max_q_len=s_qo,
                max_kv_len=s_kv,
                bmm1_scale=_q_scale * _k_scale * scale,
                bmm2_scale=_v_scale,
                o_sf_scale=-1,
                batch_size=batch_size,
                window_left=args.window_left,
                cum_seq_lens_q=qo_indptr,
                cum_seq_lens_kv=kv_indptr,
                enable_pdl=args.enable_pdl,
                is_causal=causal,
                return_lse=True,
                attention_sinks=sinks,
                out=out,
                **_trtllm_ragged_row_activity_kwargs(
                    row_activity_mode,
                    actual_seq_lens_q_cpu_flat,
                    actual_seq_lens_kv_cpu_flat,
                ),
                **trtllm_mode_kwargs,
            )[0]
        elif backend == "trtllm-fmha-v2":
            _q_scale = q_scale if q_scale is not None else 1.0
            _k_scale = k_scale if k_scale is not None else 1.0
            _fmha_v2_bmm2_scale = v_scale if v_scale is not None else 1.0
            return trtllm_fmha_v2_prefill(
                qkv=fmha_v2_qkv,
                input_layout=fmha_v2_layout,
                workspace_buffer=workspace_buffer,
                seq_lens=actual_seq_lens_kv_device.flatten(),
                max_q_len=s_qo,
                max_kv_len=s_kv,
                bmm1_scale=_q_scale * _k_scale * scale,
                bmm2_scale=_fmha_v2_bmm2_scale,
                batch_size=batch_size,
                cum_seq_lens_q=qo_indptr,
                cum_seq_lens_kv=kv_indptr,
                mask_mode="causal" if causal else "padding",
                out=out,
                out_dtype=out_dtype,
            )
        elif backend == "prims-ts":
            return backend_wrappers[backend].run(
                q,
                k,
                v,
                qo_indptr,
                kv_indptr,
                out=out,
                validate=False,
            )
        else:
            print(f"[ERROR] Backend {backend} not supported")
            return None

    has_reference_output = False
    reference_backend = None
    # Iterate over each backend:
    for cur_backend in backends:
        # Clear workspace buffer to prevent unexpected interactions between backends.
        workspace_buffer.zero_()
        runtime_workspace = None if cur_backend == "prims-ts" else workspace_buffer
        runtime_out = backend_outputs[cur_backend]
        if cur_backend == "cutile" and not _probe_backend(
            cur_backend,
            lambda: run_backend_wrapper(
                cur_backend,
                q,
                k,
                v,
                runtime_workspace,
                block_tables,
                actual_seq_lens_q_device,
                actual_seq_lens_kv_device,
                q_indptr,
                k_indptr,
                v_indptr,
                o_indptr,
                batch_offsets_stats,
                qo_indptr,
                kv_indptr,
                runtime_out,
            ),
        ):
            continue
        if run_refcheck:
            outputs[cur_backend] = (
                run_backend_wrapper(
                    cur_backend,
                    q,
                    k,
                    v,
                    runtime_workspace,
                    block_tables,
                    actual_seq_lens_q_device,
                    actual_seq_lens_kv_device,
                    q_indptr,
                    k_indptr,
                    v_indptr,
                    o_indptr,
                    batch_offsets_stats,
                    qo_indptr,
                    kv_indptr,
                    runtime_out,
                )
                .detach()
                .clone()
            )
            if cur_backend == "fa2":
                has_reference_output = True
                reference_output = outputs[cur_backend]
                reference_backend = "fa2"

        def run_timed_backend(q_arg, k_arg, v_arg, out_arg):
            result = None
            for _ in range(args.calls_per_sample):
                result = run_backend_wrapper(
                    cur_backend,
                    q_arg,
                    k_arg,
                    v_arg,
                    workspace_buffer,
                    block_tables,
                    actual_seq_lens_q_device,
                    actual_seq_lens_kv_device,
                    q_indptr,
                    k_indptr,
                    v_indptr,
                    o_indptr,
                    batch_offsets_stats,
                    qo_indptr,
                    kv_indptr,
                    out_arg,
                )
            return result

        sample_times = bench_gpu_time(
            fn=run_timed_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=True,
            enable_cupti=args.use_cupti,
            # cutlass re-allocates plan buffers per plan() and refuses CUDA-graph
            # mode since #5133 (prefill.py), so it is timed eagerly like fa2.
            use_cuda_graph=(
                is_cuda_graph_compatible and cur_backend not in ("fa2", "cutlass")
            ),
            cold_l2_cache=True,
            input_args=(
                q,
                k,
                v,
                runtime_out,
            ),
        )
        backend_times[cur_backend] = [
            sample_time / args.calls_per_sample for sample_time in sample_times
        ]

    # Perform reference check
    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if run_refcheck and "prims-ts" in outputs:
        sample_count, max_abs_error = _validate_prims_ts_context_samples(
            q=q,
            k=k,
            v=v,
            out=outputs["prims-ts"],
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            sm_scale=prims_ts_sm_scale,
            output_scale=prims_ts_output_scale,
            causal=causal,
        )
        if args.verbose >= 1:
            print(
                "[INFO] prims-ts sampled FP32 context reference passed: "
                f"{sample_count} samples, max_abs_error={max_abs_error:.6g}"
            )
        if is_cuda_graph_compatible:
            graph_output = _replay_cuda_graph_once(
                lambda: backend_wrappers["prims-ts"].run(
                    q,
                    k,
                    v,
                    qo_indptr,
                    kv_indptr,
                    out=prims_ts_out,
                    validate=False,
                ),
                prims_ts_out,
            )
            graph_sample_count, graph_max_abs_error = (
                _validate_prims_ts_context_samples(
                    q=q,
                    k=k,
                    v=v,
                    out=graph_output,
                    qo_indptr=qo_indptr,
                    kv_indptr=kv_indptr,
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    sm_scale=prims_ts_sm_scale,
                    output_scale=prims_ts_output_scale,
                    causal=causal,
                )
            )
            if args.verbose >= 1:
                print(
                    "[INFO] prims-ts CUDA-graph replay reference passed: "
                    f"{graph_sample_count} samples, "
                    f"max_abs_error={graph_max_abs_error:.6g}"
                )
    if run_refcheck and not has_reference_output and len(tested_backends) > 1:
        reference_backend, reference_output = _select_reference_output(
            outputs,
            [
                "trtllm-native",
                "trtllm-fmha-v2",
                "cudnn",
                "cudnn-native",
                "cutlass",
                "cute-dsl",
                "prims-ts",
                "auto",
                "cutile",
            ],
        )
        has_reference_output = reference_backend is not None
    if len(tested_backends) > 1:
        if run_refcheck and has_reference_output:
            if reference_output.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                if args.verbose >= 2:
                    print(
                        "[VVERBOSE] Reference output is FP8. Converting to float32 for reference check."
                    )
                reference_output = reference_output.to(torch.float32)
                tested_outputs = [output.to(torch.float32) for output in tested_outputs]
            for i in range(len(tested_backends)):
                (
                    num_different_elements,
                    num_elements,
                    num_different_elements_percentage,
                ) = is_close_stats(reference_output, tested_outputs[i], rtol, atol)
                if num_different_elements > 0:
                    print(
                        f"[ERROR] Output tensor mismatch between backends {reference_backend} and {tested_backends[i]}: "
                        f"{num_different_elements} / {num_elements} ({num_different_elements_percentage:.2f}%) elements are different"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {tested_backends[i]} output mismatch"
                        )

    # Compute perf metrics
    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            actual_seq_lens_q_flat = actual_seq_lens_q.flatten().to("cpu")
            actual_seq_lens_kv_flat = actual_seq_lens_kv.flatten().to("cpu")
            # With --window_left the metrics count only the in-window keys.
            metric_q, metric_kv, metric_causal, metric_bytes_kv = _metric_seq_lens(
                args, actual_seq_lens_q_flat, actual_seq_lens_kv_flat, causal
            )
            tflops = attention_tflops_per_sec_with_actual_seq_lens(
                metric_q,
                metric_kv,
                head_dim_qk,
                head_dim_vo,
                num_qo_heads,
                metric_causal,
                median_time,
            )
            tb_per_sec = attention_tb_per_sec_with_actual_seq_lens(
                metric_q,
                metric_bytes_kv,
                head_dim_qk,
                head_dim_vo,
                num_qo_heads,
                num_kv_heads,
                median_time,
                q_dtype=q_dtype,
                k_dtype=kv_dtype,
                v_dtype=kv_dtype,
                o_dtype=out_dtype,
            )

            display_backend = (
                f"auto({resolved_backends['auto']})" if backend == "auto" else backend
            )
            print_perf_metrics(
                display_backend, median_time, std_time, tflops, tb_per_sec
            )

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["backend"] = backend
                if backend == "auto":
                    cur_res["resolved_backend"] = resolved_backends["auto"]
                cur_res["page_size"] = 0  # No page size for ragged
                cur_res["batch_size"] = batch_size
                cur_res["s_qo"] = s_qo
                cur_res["s_kv"] = s_kv
                cur_res["num_qo_heads"] = num_qo_heads
                cur_res["num_kv_heads"] = num_kv_heads
                cur_res["head_dim_qk"] = head_dim_qk
                cur_res["head_dim_vo"] = head_dim_vo
                cur_res["causal"] = causal
                cur_res["q_dtype"] = q_dtype
                cur_res["kv_dtype"] = kv_dtype
                cur_res["out_dtype"] = out_dtype
                cur_res["avg_actual_seq_len"] = avg_seq_len_q
                cur_res["random_actual_seq_len"] = args.random_actual_seq_len
                cur_res["case_tag"] = args.case_tag
                if args.row_activity_mode is not None:
                    cur_res["timing_metric"] = TRTLLM_RAGGED_TIMING_METRIC
                    cur_res["row_activity_mode"] = args.row_activity_mode
                    cur_res["calls_per_sample"] = args.calls_per_sample
                _record_mode_columns(cur_res, args)
                res.append(cur_res)
    return res


def testAppendPagedKVCache(args):
    """
    Test the paged KV-cache write path (flashinfer.page).

    --kv_cache_kind mha appends K/V rows with append_paged_kv_cache
    (bf16/fp16/fp8 caches; the append rows carry the cache dtype) or, with
    --kv_dtype nvfp4, quantizes them with nvfp4_quantize_append_paged_kv_cache
    (--slot_mapping: nvfp4_quantize_append_paged_kv_cache_with_slot_mapping).
    --kv_cache_kind mla appends ckv/kpe rows with append_paged_mla_kv_cache or,
    with --kv_dtype nvfp4, nvfp4_quantize_append_paged_mla_kv_cache.

    Each request holds s_kv tokens after the append, the last s_qo of which are
    appended (random lengths with --random_actual_seq_len). Bandwidth counts the
    appended rows read plus the cache bytes written; tflops is 0.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testAppendPagedKVCache")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )
    res = []

    backends = filter_backends_by_compute_capability(
        args.backends, args.routine, device
    )
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    is_mla = args.kv_cache_kind == "mla"
    is_nvfp4 = args.kv_dtype == "nvfp4"
    kv_layout = args.kv_layout
    batch_size = args.batch_size
    page_size = args.page_size
    s_qo = args.s_qo
    s_kv = args.s_kv
    if page_size <= 0:
        print("[ERROR] append_paged_kv_cache requires --page_size > 0.")
        return res
    if s_qo > s_kv:
        print("[ERROR] s_qo (append length) must be <= s_kv. Exiting.")
        return res
    if args.slot_mapping and (is_mla or not is_nvfp4):
        print("[ERROR] --slot_mapping requires --kv_cache_kind mha --kv_dtype nvfp4.")
        return res
    # Rows to append are written in the input dtype for the quantizing paths and
    # in the cache dtype otherwise (append_paged_kv_cache copies, it does not cast).
    input_dtype = dtype_str_to_torch_dtype(args.q_dtype if is_nvfp4 else args.kv_dtype)
    if input_dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn):
        print(f"[ERROR] Unsupported append dtype: {input_dtype}")
        return res
    if is_nvfp4 and input_dtype not in (torch.float16, torch.bfloat16):
        print("[ERROR] NVFP4 quantize-append takes fp16/bf16 rows (--q_dtype).")
        return res
    if is_mla:
        head_dim_ckv = args.head_dim_ckv or 512
        head_dim_kpe = args.head_dim_kpe or 64
        num_kv_heads = 1
        row_widths = (head_dim_ckv, head_dim_kpe)
    else:
        num_kv_heads = args.num_kv_heads
        head_dim = args.head_dim_qk
        row_widths = (num_kv_heads * head_dim, num_kv_heads * head_dim)

    # Per-request append lengths and the cache lengths after the append.
    append_lens = sample_actual_seq_lens(
        s_qo, batch_size, None, args.random_actual_seq_len
    ).flatten()
    kv_lens = torch.full_like(append_lens, s_kv)
    nnz = int(append_lens.sum().item())
    pages_per_req = (kv_lens + page_size - 1) // page_size
    total_pages = int(pages_per_req.sum().item())
    append_indptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(append_lens, 0).int()]
    ).to(device)
    kv_indptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(pages_per_req, 0).int()]
    ).to(device)
    # Shuffled page ids so the write pattern is not one contiguous slab.
    kv_indices = torch.randperm(total_pages, device=device).int()
    kv_last_page_len = ((kv_lens - 1) % page_size + 1).int().to(device)
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        append_indptr,
        flashinfer.get_seq_lens(kv_indptr, kv_last_page_len, page_size),
        nnz,
    )

    def page_shape(width):
        # One cache tensor (K, V, ckv or kpe) in the requested layout.
        if is_mla:
            return (total_pages, page_size, width)
        if kv_layout == "NHD":
            return (total_pages, page_size, num_kv_heads, width)
        return (total_pages, num_kv_heads, page_size, width)

    if is_mla:
        append_a = torch.randn(nnz, head_dim_ckv, device=device, dtype=torch.bfloat16)
        append_b = torch.randn(nnz, head_dim_kpe, device=device, dtype=torch.bfloat16)
    else:
        append_a = torch.randn(
            nnz, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        append_b = torch.randn_like(append_a)
    append_a = append_a.to(input_dtype)
    append_b = append_b.to(input_dtype)
    cache_width_a = row_widths[0] if is_mla else head_dim
    cache_width_b = row_widths[1] if is_mla else head_dim

    if is_nvfp4:
        # Packed E2M1 (2 values per byte) + FP8 E4M3 block scales per 16 values.
        cache_a = torch.zeros(
            page_shape(cache_width_a // 2), dtype=torch.uint8, device=device
        )
        sf_a = torch.zeros(
            page_shape(cache_width_a // 16), dtype=torch.float8_e4m3fn, device=device
        )
        if is_mla:
            # The rope part stays FP8 E4M3 (page.py nvfp4_quantize_append_paged_mla_kv_cache).
            cache_b = torch.zeros(
                page_shape(cache_width_b), dtype=torch.float8_e4m3fn, device=device
            )
            sf_b = None
        else:
            cache_b = torch.zeros_like(cache_a)
            sf_b = torch.zeros_like(sf_a)
        scale_a = float(append_a.abs().amax()) / (448.0 * 6.0)
        scale_b = float(append_b.abs().amax()) / (448.0 * (1.0 if is_mla else 6.0))
    else:
        cache_a = torch.zeros(
            page_shape(cache_width_a), dtype=input_dtype, device=device
        )
        cache_b = torch.zeros(
            page_shape(cache_width_b), dtype=input_dtype, device=device
        )
        sf_a = sf_b = None

    slot_mapping = None
    if args.slot_mapping:
        page_ids = kv_indices[kv_indptr[batch_indices.long()] + positions // page_size]
        slot_mapping = (page_ids * page_size + positions % page_size).int()
        k_scale_t = torch.tensor([scale_a], dtype=torch.float32, device=device)
        v_scale_t = torch.tensor([scale_b], dtype=torch.float32, device=device)

    def run_append():
        if is_mla and is_nvfp4:
            flashinfer.nvfp4_quantize_append_paged_mla_kv_cache(
                append_a,
                append_b,
                batch_indices,
                positions,
                cache_a,
                sf_a,
                cache_b,
                kv_indices,
                kv_indptr,
                kv_last_page_len,
                scale_a,
                scale_b,
            )
        elif is_mla:
            flashinfer.append_paged_mla_kv_cache(
                append_a,
                append_b,
                batch_indices,
                positions,
                cache_a,
                cache_b,
                kv_indices,
                kv_indptr,
                kv_last_page_len,
            )
        elif args.slot_mapping:
            flashinfer.nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
                append_a,
                append_b,
                slot_mapping,
                (cache_a, cache_b),
                (sf_a, sf_b),
                k_scale_t,
                v_scale_t,
                kv_layout=kv_layout,
            )
        elif is_nvfp4:
            flashinfer.nvfp4_quantize_append_paged_kv_cache(
                append_a,
                append_b,
                batch_indices,
                positions,
                (cache_a, cache_b),
                (sf_a, sf_b),
                kv_indices,
                kv_indptr,
                kv_last_page_len,
                scale_a,
                scale_b,
                kv_layout=kv_layout,
            )
        else:
            flashinfer.append_paged_kv_cache(
                append_a,
                append_b,
                batch_indices,
                positions,
                (cache_a, cache_b),
                kv_indices,
                kv_indptr,
                kv_last_page_len,
                kv_layout=kv_layout,
            )

    if args.refcheck:
        run_append()
        if is_nvfp4:
            print("[INFO] refcheck skipped for NVFP4 quantize-append (lossy write).")
        else:
            # Every appended row must land at (page, entry) of its logical position.
            page_ids = kv_indices[
                kv_indptr[batch_indices.long()] + positions // page_size
            ].long()
            entries = (positions % page_size).long()
            if is_mla or kv_layout == "NHD":
                got_a, got_b = cache_a[page_ids, entries], cache_b[page_ids, entries]
            else:
                got_a = cache_a[page_ids, :, entries]
                got_b = cache_b[page_ids, :, entries]
            if not (
                torch.equal(got_a.float(), append_a.float())
                and torch.equal(got_b.float(), append_b.float())
            ):
                print("[ERROR] Appended rows do not match the paged cache contents.")
                if not args.allow_output_mismatch:
                    raise AssertionError("[ERROR] append_paged_kv_cache mismatch")

    times = bench_gpu_time(
        fn=run_append,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        sleep_after_run=False,
        enable_cupti=args.use_cupti,
        use_cuda_graph=not args.no_cuda_graph,
        cold_l2_cache=True,
    )
    median_time = np.median(times)
    std_time = np.std(times)
    # Bytes moved: appended rows read + the same rows written in the cache format.
    read_bytes = (append_a.numel() + append_b.numel()) * append_a.element_size()
    if is_nvfp4:
        written_values = append_a.numel() + (0 if is_mla else append_b.numel())
        write_bytes = written_values // 2 + written_values // 16
        if is_mla:
            write_bytes += append_b.numel()  # FP8 rope part
    else:
        write_bytes = read_bytes
    tb_per_sec = (read_bytes + write_bytes) / (median_time * 1e-3) / 1e12
    backend = backends[0]
    print_perf_metrics(backend, median_time, std_time, 0.0, tb_per_sec)

    if args.output_path is not None:
        cur_res = defaultdict(str)
        cur_res["routine"] = args.routine
        cur_res["median_time"] = median_time
        cur_res["std_time"] = std_time
        cur_res["tflops"] = 0.0
        cur_res["tb_per_sec"] = tb_per_sec
        cur_res["backend"] = backend
        cur_res["page_size"] = page_size
        cur_res["batch_size"] = batch_size
        cur_res["s_qo"] = s_qo
        cur_res["s_kv"] = s_kv
        cur_res["num_kv_heads"] = num_kv_heads
        if is_mla:
            cur_res["head_dim_ckv"] = head_dim_ckv
            cur_res["head_dim_kpe"] = head_dim_kpe
        else:
            cur_res["head_dim_qk"] = head_dim
        cur_res["q_dtype"] = input_dtype
        cur_res["kv_dtype"] = args.kv_dtype
        cur_res["kv_layout"] = "NHD" if is_mla else kv_layout
        cur_res["kv_cache_kind"] = args.kv_cache_kind
        cur_res["slot_mapping"] = args.slot_mapping
        cur_res["avg_actual_seq_len"] = nnz // batch_size
        cur_res["random_actual_seq_len"] = args.random_actual_seq_len
        cur_res["case_tag"] = args.case_tag
        res.append(cur_res)
    return res


def testSingleRequestAttention(args):
    """
    Test single_decode_with_kv_cache / single_prefill_with_kv_cache: one request
    with a contiguous (non-paged) KV tensor, the batch-1 APIs behind the batch
    wrappers.

    Decode backends: fa2 (CUDA-core decode kernel) and fa2_tc
    (use_tensor_cores=True). Prefill backends: fa2, fa3, auto. --s_qo is the
    query length (ignored for decode), --s_kv the KV length; --kv_layout picks
    NHD/HND; --window_left and --logits_soft_cap pass through. FP8 KV takes
    per-tensor k/v scales.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print(f"[INFO] Running testSingleRequestAttention ({args.routine})")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )
    res = []
    is_decode = args.routine == "single_decode_with_kv_cache"
    backends = filter_backends_by_compute_capability(
        args.backends, args.routine, device
    )
    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    if q_dtype not in (torch.float16, torch.bfloat16) or kv_dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
    ):
        print("[ERROR] single_* routines take fp16/bf16 q and fp16/bf16/fp8_e4m3 KV.")
        return res
    if "fa3" in backends and kv_dtype != q_dtype:
        _drop_backend(backends, "fa3", "requires kv_dtype == q_dtype here")
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    s_qo = 1 if is_decode else args.s_qo
    s_kv = args.s_kv
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim = args.head_dim_qk
    causal = args.causal and not is_decode
    if s_qo > s_kv:
        print("[ERROR] s_qo > s_kv is not supported. Exiting.")
        return res

    q = torch.randn(s_qo, num_qo_heads, head_dim, device=device, dtype=q_dtype)
    if is_decode:
        q = q[0]
    kv_shape = (
        (s_kv, num_kv_heads, head_dim)
        if args.kv_layout == "NHD"
        else (num_kv_heads, s_kv, head_dim)
    )
    k = torch.randn(kv_shape, device=device, dtype=q_dtype)
    v = torch.randn(kv_shape, device=device, dtype=q_dtype)
    k_scale = v_scale = None
    if kv_dtype == torch.float8_e4m3fn:
        k, k_scale_t = to_float8(k, kv_dtype)
        v, v_scale_t = to_float8(v, kv_dtype)
        k_scale, v_scale = k_scale_t.item(), v_scale_t.item()
    mode_kwargs = {}
    if args.window_left >= 0:
        mode_kwargs["window_left"] = args.window_left
    if args.logits_soft_cap > 0:
        mode_kwargs["logits_soft_cap"] = args.logits_soft_cap

    def run_backend(backend):
        if is_decode:
            return flashinfer.single_decode_with_kv_cache(
                q,
                k,
                v,
                kv_layout=args.kv_layout,
                use_tensor_cores=backend == "fa2_tc",
                k_scale=k_scale,
                v_scale=v_scale,
                **mode_kwargs,
            )
        return flashinfer.single_prefill_with_kv_cache(
            q,
            k,
            v,
            causal=causal,
            kv_layout=args.kv_layout,
            backend=backend,
            k_scale=k_scale,
            v_scale=v_scale,
            **mode_kwargs,
        )

    outputs = {}
    backend_times = {}
    for backend in backends:
        if args.refcheck:
            outputs[backend] = run_backend(backend).detach().clone()
        backend_times[backend] = bench_gpu_time(
            fn=lambda: run_backend(backend),
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=False,
            enable_cupti=args.use_cupti,
            use_cuda_graph=not args.no_cuda_graph,
            cold_l2_cache=True,
        )

    if args.refcheck and len(outputs) > 1:
        reference_backend, reference_output = next(iter(outputs.items()))
        for backend, output in outputs.items():
            num_diff, num_elems, pct = is_close_stats(
                reference_output.float(), output.float(), 2e-1, 1e-2
            )
            if num_diff > 0:
                print(
                    f"[ERROR] Output tensor mismatch between backends {reference_backend} and {backend}: "
                    f"{num_diff} / {num_elems} ({pct:.2f}%) elements are different"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(f"[ERROR] Backend {backend} output mismatch")

    seq_q = torch.tensor([s_qo], dtype=torch.int32)
    seq_kv = torch.tensor([s_kv], dtype=torch.int32)
    for backend in backends:
        median_time = np.median(backend_times[backend])
        std_time = np.std(backend_times[backend])
        # With --window_left the metrics count only the in-window keys.
        metric_q, metric_kv, metric_causal, metric_bytes_kv = _metric_seq_lens(
            args, seq_q, seq_kv, causal
        )
        tflops = attention_tflops_per_sec_with_actual_seq_lens(
            metric_q,
            metric_kv,
            head_dim,
            head_dim,
            num_qo_heads,
            metric_causal,
            median_time,
        )
        tb_per_sec = attention_tb_per_sec_with_actual_seq_lens(
            metric_q,
            metric_bytes_kv,
            head_dim,
            head_dim,
            num_qo_heads,
            num_kv_heads,
            median_time,
            q_dtype=q_dtype,
            k_dtype=kv_dtype,
            v_dtype=kv_dtype,
            o_dtype=q_dtype,
        )
        print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)
        if args.output_path is not None:
            cur_res = defaultdict(str)
            cur_res["routine"] = args.routine
            cur_res["median_time"] = median_time
            cur_res["std_time"] = std_time
            cur_res["tflops"] = tflops
            cur_res["tb_per_sec"] = tb_per_sec
            cur_res["backend"] = backend
            cur_res["batch_size"] = 1
            cur_res["s_qo"] = s_qo
            cur_res["s_kv"] = s_kv
            cur_res["num_qo_heads"] = num_qo_heads
            cur_res["num_kv_heads"] = num_kv_heads
            cur_res["head_dim_qk"] = head_dim
            cur_res["head_dim_vo"] = head_dim
            cur_res["causal"] = causal
            cur_res["q_dtype"] = q_dtype
            cur_res["kv_dtype"] = kv_dtype
            cur_res["out_dtype"] = q_dtype
            cur_res["kv_layout"] = args.kv_layout
            cur_res["case_tag"] = args.case_tag
            _record_mode_columns(cur_res, args)
            res.append(cur_res)
    return res


def testMergeStates(args):
    """
    Test the cascade attention-state merge (flashinfer.cascade): merge_state for
    --num_states 2 (two KV segments, e.g. shared prefix + unique suffix) and
    merge_states for more. Rows: --batch_size tokens x --num_qo_heads heads x
    --head_dim_vo; fp32 LSE. Reports bandwidth only (tflops 0).

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testMergeStates")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )
    res = []
    backends = filter_backends_by_compute_capability(
        args.backends, args.routine, device
    )
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res
    num_states = args.num_states
    if num_states < 2:
        print("[ERROR] --num_states must be >= 2.")
        return res
    v_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    seq_len = args.batch_size
    num_heads = args.num_qo_heads
    head_dim = args.head_dim_vo or args.head_dim_qk
    v = torch.randn(
        seq_len, num_states, num_heads, head_dim, device=device, dtype=v_dtype
    )
    s = torch.randn(seq_len, num_states, num_heads, device=device, dtype=torch.float32)
    v_a, v_b = v[:, 0].contiguous(), v[:, 1].contiguous()
    s_a, s_b = s[:, 0].contiguous(), s[:, 1].contiguous()

    def run_merge():
        if num_states == 2:
            return flashinfer.merge_state(v_a, s_a, v_b, s_b)
        return flashinfer.merge_states(v, s)

    if args.refcheck:
        v_out, s_out = run_merge()
        s_max = s.amax(dim=1, keepdim=True)
        w = torch.exp2(s - s_max)  # flashinfer LSE is base-2
        v_ref = (v.float() * w.unsqueeze(-1)).sum(1) / w.sum(1).unsqueeze(-1)
        num_diff, num_elems, pct = is_close_stats(v_ref, v_out.float(), 2e-2, 2e-2)
        if num_diff > 0:
            print(
                f"[ERROR] merge output mismatch vs fp32 reference: "
                f"{num_diff} / {num_elems} ({pct:.2f}%) elements are different"
            )
            if not args.allow_output_mismatch:
                raise AssertionError("[ERROR] merge_states mismatch")

    times = bench_gpu_time(
        fn=run_merge,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        sleep_after_run=False,
        enable_cupti=args.use_cupti,
        use_cuda_graph=not args.no_cuda_graph,
        cold_l2_cache=True,
    )
    median_time = np.median(times)
    std_time = np.std(times)
    # num_states (v, s) read + one (v, s) written.
    moved = (num_states + 1) * (seq_len * num_heads * (head_dim * v.element_size() + 4))
    tb_per_sec = moved / (median_time * 1e-3) / 1e12
    backend = backends[0]
    print_perf_metrics(backend, median_time, std_time, 0.0, tb_per_sec)
    if args.output_path is not None:
        cur_res = defaultdict(str)
        cur_res["routine"] = args.routine
        cur_res["median_time"] = median_time
        cur_res["std_time"] = std_time
        cur_res["tflops"] = 0.0
        cur_res["tb_per_sec"] = tb_per_sec
        cur_res["backend"] = backend
        cur_res["batch_size"] = seq_len
        cur_res["num_qo_heads"] = num_heads
        cur_res["head_dim_vo"] = head_dim
        cur_res["q_dtype"] = v_dtype
        cur_res["num_states"] = num_states
        cur_res["case_tag"] = args.case_tag
        res.append(cur_res)
    return res


def testBatchMLAPagedAttentionWrapper(args):
    """
    Test BatchMLAPagedAttentionWrapper and equivalent APIs.
    Supports fa2, fa3, cutlass, cutile and trtllm-gen through the wrapper;
    trtllm-native, cute-dsl and auto through
    trtllm_batch_decode_with_kv_cache_mla; and prims-ts.
    --sparse_mla_top_k > 0 benchmarks token-sparse MLA (trtllm-native/auto).

    This test:
    1. Creates paged query and key-value cache tensors
    2. Runs MLA with different backends
    3. Verifies outputs match between backends
    4. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testBatchMLAPagedAttentionWrapper")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    # Basic setup
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    q_init_dtype = torch.bfloat16
    kv_init_dtype = torch.bfloat16
    rtol = 2e-1
    atol = 1e-2
    res = []

    # Handle different query data types.
    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    if q_dtype not in [torch.bfloat16, torch.float8_e4m3fn]:
        print(f"[ERROR] Unsupported q_dtype: {args.q_dtype}")
        return res

    # Handle different KV cache data types.
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    if kv_dtype not in [torch.bfloat16, torch.float8_e4m3fn]:
        print(f"[ERROR] Unsupported kv_dtype: {args.kv_dtype}")
        return res

    if args.out_dtype is not None:
        print(
            "[WARNING] --out_dtype is not yet supported for BatchMLAPagedAttentionWrapper; ignoring."
        )

    backends = args.backends
    page_size = args.page_size
    batch_size = args.batch_size
    s_qo = args.s_qo
    s_kv = args.s_kv
    num_qo_heads = args.num_qo_heads
    # num_kv_heads not used in MLA
    # head_dim_qk = args.head_dim_qk
    assert args.head_dim_ckv is not None, "head_dim_ckv must be provided for MLA"
    assert args.head_dim_kpe is not None, "head_dim_kpe must be provided for MLA"
    head_dim_ckv = args.head_dim_ckv
    head_dim_kpe = args.head_dim_kpe
    # Un-absorbed q/k nope width of the model the latent rank comes from:
    # DeepSeek-V3/R1 (kv_lora_rank 512) uses 128, the smaller MLA config
    # (kv_lora_rank 256) uses 64. It sets the softmax scale.
    qk_nope_head_dim = 64 if head_dim_ckv == 256 else 128
    sparse_mla_top_k = args.sparse_mla_top_k
    if sparse_mla_top_k < 0:
        print("[ERROR] --sparse_mla_top_k must be >= 0. Exiting.")
        return res
    is_cuda_graph_compatible = not args.no_cuda_graph
    # Multi-query MLA decode uses bottom-right causal masking. For SQ=1 this
    # has the same visible K/V domain as dense decode.
    causal = True
    run_refcheck = args.refcheck

    if s_qo > s_kv:
        print("[ERROR] Causal MLA decode requires s_qo <= s_kv. Exiting.")
        return res

    # Resolve the MLA is_var_seq override (selects var-seq vs. persistent
    # scheduler). None => do not pass is_var_seq to the API, keeping its default
    # so existing cases and perf baselines are unchanged.
    mla_is_var_seq_arg = getattr(args, "mla_is_var_seq", None)
    if mla_is_var_seq_arg is None:
        resolved_is_var_seq = None
    elif mla_is_var_seq_arg == "auto":
        resolved_is_var_seq = getattr(args, "random_actual_seq_len", False)
    else:
        resolved_is_var_seq = mla_is_var_seq_arg == "true"
    # Only forwarded to the direct trtllm API when explicitly resolved.
    mla_api_extra_kwargs = (
        {} if resolved_is_var_seq is None else {"is_var_seq": resolved_is_var_seq}
    )
    # Resolve the MLA cute_dsl_impl override (selects modular vs. monolithic
    # CuTe DSL decode kernel). None => do not pass cute_dsl_impl, keeping the
    # API default ('auto') so existing cases and perf baselines are unchanged.
    mla_cute_dsl_impl_arg = getattr(args, "mla_cute_dsl_impl", None)
    mla_softmax_rescale_threshold_log2 = float(
        getattr(args, "mla_softmax_rescale_threshold_log2", 0.0)
    )
    if mla_cute_dsl_impl_arg is not None:
        mla_api_extra_kwargs["cute_dsl_impl"] = mla_cute_dsl_impl_arg

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    # Check for backend-specific constraints
    if "fa2" in backends:
        remove_fa2 = False
        if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] or kv_dtype in [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]:
            print("[INFO] FA2 backend does not support FP8. Skipping.")
            remove_fa2 = True
        if remove_fa2:
            backends.remove("fa2")
    if "fa3" in backends:
        remove_fa3 = False
        if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] or kv_dtype in [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]:
            print("[INFO] FA3 backend does not support FP8. Skipping.")
            remove_fa3 = True
        if remove_fa3:
            backends.remove("fa3")
    if "cutlass" in backends:
        remove_cutlass = False
        # The CUTLASS page table is consumed in 128-token tiles, so the page size
        # must divide 128 and the per-request page count must fill whole tiles.
        num_pages_per_seq = (s_kv + page_size - 1) // page_size
        if page_size > 128 or 128 % page_size != 0:
            print(
                "[INFO] Cutlass MLA backend requires a page size dividing 128. Skipping."
            )
            remove_cutlass = True
        elif num_pages_per_seq % (128 // page_size) != 0:
            print(
                "[INFO] Cutlass MLA backend requires s_kv pages to fill 128-token "
                "tiles. Skipping."
            )
            remove_cutlass = True
        if num_qo_heads != 128 or (head_dim_ckv, head_dim_kpe) != (512, 64):
            print(
                "[INFO] Cutlass MLA backend requires 128 heads, head_dim_ckv=512 "
                "and head_dim_kpe=64. Skipping."
            )
            remove_cutlass = True
        if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] or kv_dtype in [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]:
            print("[INFO] Cutlass MLA backend does not support FP8. Skipping.")
            remove_cutlass = True
        if remove_cutlass:
            backends.remove("cutlass")
    for backend in ("trtllm-native", "trtllm-gen"):
        if backend not in backends:
            continue
        if page_size not in [32, 64]:
            _drop_backend(backends, backend, "only supports page size 32 or 64")
        elif 64 < num_qo_heads < 128:
            _drop_backend(backends, backend, "does not support 64 < num_qo_heads < 128")
    if "trtllm-gen" in backends and q_dtype != kv_dtype:
        _drop_backend(
            backends, "trtllm-gen", "requires matching query and cache dtypes"
        )
    if "cutile" in backends:
        if q_dtype != torch.bfloat16 or kv_dtype != torch.bfloat16:
            _drop_backend(backends, "cutile", "requires bfloat16 query and cache")
        elif head_dim_ckv not in (256, 512) or head_dim_kpe != 64:
            _drop_backend(
                backends, "cutile", "requires head_dim_ckv in (256, 512), kpe=64"
            )
        elif s_qo != 1:
            _drop_backend(backends, "cutile", "supports s_qo=1 only")
    if "cute-dsl-rubin-mtp" in backends and (
        q_dtype != torch.float8_e4m3fn
        or kv_dtype != torch.float8_e4m3fn
        or num_qo_heads != 128
        or (head_dim_ckv, head_dim_kpe) != (512, 64)
        or page_size not in (64, 128)
        or s_qo not in (2, 4)
        or args.enable_pdl
    ):
        # _batch_mla/_backends/cute_dsl_rubin_mtp_backend.py preflight checks
        _drop_backend(
            backends,
            "cute-dsl-rubin-mtp",
            "requires fp8_e4m3 Q/KV, 128 heads, ckv 512 / kpe 64, page 64/128, "
            "s_qo 2 or 4 (MTP) and no PDL",
        )
    if sparse_mla_top_k > 0:
        for backend in list(backends):
            if backend not in ("trtllm-native", "auto"):
                _drop_backend(
                    backends, backend, "does not run token-sparse MLA in this benchmark"
                )
    if args.use_fp16_softmax and "trtllm-native" in backends:
        if get_compute_capability(device) != (10, 7):
            _drop_backend(
                backends, "trtllm-native", "supports use_fp16_softmax on SM107 only"
            )
    if s_qo > 1:
        for backend in ("fa2", "fa3", "cutlass"):
            _drop_backend(
                backends,
                backend,
                "is not validated for multi-query MLA in this benchmark",
            )
    if "prims-ts" in backends:
        if q_dtype != kv_dtype:
            _drop_backend(
                backends, "prims-ts", "requires matching query and cache dtypes"
            )
        elif (head_dim_ckv, head_dim_kpe) != (512, 64):
            _drop_backend(
                backends,
                "prims-ts",
                "requires head_dim_ckv=512 and head_dim_kpe=64",
            )
        elif page_size not in (16, 32, 64, 128):
            _drop_backend(
                backends,
                "prims-ts",
                "requires page_size in {16, 32, 64, 128}",
            )
        elif args.enable_pdl:
            print("[WARNING] prims-ts does not expose PDL; ignoring --enable_pdl.")
    if mla_softmax_rescale_threshold_log2 != 0.0 and "prims-ts" not in backends:
        print(
            "[WARNING] --mla_softmax_rescale_threshold_log2 applies only to "
            "prims-ts; ignoring it."
        )
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}

    actual_seq_lens_kv = sample_actual_seq_lens(
        s_kv, batch_size, device, args.random_actual_seq_len
    ).clamp_min(s_qo)
    sum_seq_kv = torch.sum(actual_seq_lens_kv).item()
    avg_seq_len_kv = sum_seq_kv // batch_size

    if args.verbose >= 1:
        print(f"[VERBOSE] Average actual seq len: {avg_seq_len_kv}")
    if args.verbose >= 2:
        print(f"[VVERBOSE] {actual_seq_lens_kv.flatten() = }")

    q_nope = torch.rand(
        batch_size * s_qo,
        num_qo_heads,
        head_dim_ckv,
        dtype=q_init_dtype,
        device=device,
    )
    q_pe = torch.zeros(
        batch_size * s_qo,
        num_qo_heads,
        head_dim_kpe,
        dtype=q_init_dtype,
        device=device,
    )
    q = torch.cat([q_nope, q_pe], dim=2)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {q_nope.shape = }")
        print(f"[VVERBOSE] {q_pe.shape = }")
        print(f"[VVERBOSE] {q.shape = }")

    # Create KV cache
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    # Now initialize the page tables
    block_tables = torch.tensor(
        [
            [k + i * num_pages_per_seq for k in torch.randperm(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype=torch.int,
        device=device,
    )

    if args.verbose >= 2:
        print(f"[VVERBOSE] {num_pages_per_seq = }")
        print(f"[VVERBOSE] {total_num_pages = }")
        print(f"[VVERBOSE] {block_tables.shape = }")

    # Initialize KV cache with appropriate shape and stride
    ckv_cache_shape = (
        total_num_pages,
        page_size,
        head_dim_ckv,
    )
    ckv_cache = torch.randn(size=ckv_cache_shape, dtype=kv_init_dtype, device=device)

    kpe_cache_shape = (
        total_num_pages,
        page_size,
        head_dim_kpe,
    )
    kpe_cache = torch.randn(size=kpe_cache_shape, dtype=kv_init_dtype, device=device)
    kv_cache = torch.cat([ckv_cache, kpe_cache], dim=2)

    qo_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * s_qo
    kv_indptr = (
        torch.cat(
            [
                torch.tensor([0], device=device),
                torch.cumsum(
                    (actual_seq_lens_kv.flatten() + page_size - 1) // page_size, dim=0
                ),
            ]
        )
        .int()
        .to(device)
    )

    # kv_indices[-1] is the total number of actual pages
    kv_indices = torch.zeros(kv_indptr[-1], device=device, dtype=torch.int32)
    for i in range(len(kv_indptr) - 1):
        start_idx = kv_indptr[i]
        end_idx = kv_indptr[i + 1]
        kv_indices[start_idx:end_idx] = block_tables[i, : end_idx - start_idx]

    # Softmax scale of the un-absorbed head: 1/sqrt(192) for DeepSeek-R1.
    sm_scale = 1.0 / ((qk_nope_head_dim + head_dim_kpe) ** 0.5)
    # trtllm-native/auto read the token-sparse index table instead of the
    # page table when --sparse_mla_top_k is set.
    direct_block_tables = block_tables
    sparse_active_rows = None
    mla_sparse_kwargs = (
        {"sparse_mla_top_k": sparse_mla_top_k} if sparse_mla_top_k > 0 else {}
    )
    mla_fp16_softmax_kwargs = (
        {"use_fp16_softmax": True} if args.use_fp16_softmax else {}
    )
    if sparse_mla_top_k > 0:
        direct_block_tables, sparse_active_rows = _mla_sparse_token_indices(
            block_tables,
            actual_seq_lens_kv.flatten(),
            s_qo,
            sparse_mla_top_k,
            page_size,
        )
    workspace_buffer = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device=device)
    mla_multi_ctas_kv_counter_buffer = None
    if "trtllm-native" in backends:
        counter_bytes = get_trtllm_gen_multi_ctas_kv_counter_bytes(
            batch_size, num_qo_heads, get_device_sm_count(device)
        )
        mla_multi_ctas_kv_counter_buffer = torch.zeros(
            counter_bytes, dtype=torch.uint8, device=device
        )

    if args.verbose >= 2:
        print(f"[VVERBOSE] {ckv_cache.shape = }")
        print(f"[VVERBOSE] {kpe_cache.shape = }")
        print(f"[VVERBOSE] {kv_cache.shape = }")
        print(f"[VVERBOSE] {qo_indptr.shape = }")
        print(f"[VVERBOSE] {kv_indptr.shape = }")
        print(f"[VVERBOSE] {kv_indices.shape = }")
        print(f"[VVERBOSE] {actual_seq_lens_kv.shape = }")
        print(f"[VVERBOSE] {sm_scale = }")
        print(f"[VVERBOSE] {workspace_buffer.shape = }")

    # Create wrapper
    # The shared sampler retains singleton dimensions for other attention
    # routines, but MLA CSR metadata requires one KV length per request.
    mla_kv_len_arr = actual_seq_lens_kv.flatten()
    backend_wrappers = {}
    for backend in list(backends):
        if backend in ["fa2", "fa3", "cutlass"]:
            backend_wrappers[backend] = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                float_workspace_buffer=workspace_buffer,
                use_cuda_graph=is_cuda_graph_compatible,
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                kv_len_arr=mla_kv_len_arr,
                backend=backend,
            )
            if backend != "cutlass":
                backend_wrappers[backend].plan(
                    qo_indptr=qo_indptr,
                    kv_indptr=kv_indptr,
                    kv_indices=kv_indices,
                    kv_len_arr=mla_kv_len_arr,
                    num_heads=num_qo_heads,
                    head_dim_ckv=head_dim_ckv,
                    head_dim_kpe=head_dim_kpe,
                    page_size=page_size,
                    causal=causal,
                    sm_scale=sm_scale,
                    q_data_type=q_dtype,
                    kv_data_type=kv_dtype,
                )
        elif backend in ("cutile", "trtllm-gen", "cute-dsl-rubin-mtp"):
            # Wrapper-native backends planned from the canonical dense metadata
            # and run on packed [total_q, H, ckv + kpe] / [pages, page, ckv + kpe]
            # tensors. cute-dsl-rubin-mtp (SM107) writes FP8 output.
            backend_wrappers[backend] = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                float_workspace_buffer=workspace_buffer,
                use_cuda_graph=is_cuda_graph_compatible,
                backend=backend,
            )
            plan_call = lambda backend=backend: backend_wrappers[backend].plan(
                metadata=flashinfer.mla.MLAPlanMetadata.dense(
                    cum_seq_lens_q=qo_indptr,
                    block_tables=block_tables,
                    seq_lens=mla_kv_len_arr,
                    max_q_len=s_qo,
                ),
                num_heads=num_qo_heads,
                head_dim_ckv=head_dim_ckv,
                head_dim_kpe=head_dim_kpe,
                page_size=page_size,
                causal=causal,
                sm_scale=sm_scale,
                q_data_type=q_dtype,
                kv_data_type=kv_dtype,
                output_dtype=(
                    torch.float8_e4m3fn
                    if backend == "cute-dsl-rubin-mtp"
                    else torch.bfloat16
                ),
                enable_pdl=(args.enable_pdl if backend == "trtllm-gen" else None),
            )
            if backend in ("cute-dsl-rubin-mtp", "cutile"):
                # CuTe DSL compiles at plan(); cutile needs cuda-tile + tileiras
                # in the environment. A failed plan skips the backend.
                if not _probe_backend(backend, plan_call):
                    backends.remove(backend)
            else:
                plan_call()

    if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        q = q.to(q_dtype)
        q_pe = q_pe.to(q_dtype)
        q_nope = q_nope.to(q_dtype)
    if kv_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        ckv_cache = ckv_cache.to(kv_dtype)
        kpe_cache = kpe_cache.to(kv_dtype)
        kv_cache = kv_cache.to(kv_dtype)

    prims_ts_out = None
    if "prims-ts" in backends:
        prims_ts = _get_prims_ts_module()
        prims_ts_out = torch.empty(
            batch_size,
            s_qo,
            num_qo_heads,
            head_dim_ckv,
            device=device,
            dtype=torch.bfloat16,
        )
        backend_wrappers["prims-ts"] = prims_ts.BatchMLADecodePagedTSWrapper()
        backend_wrappers["prims-ts"].plan(
            device,
            batch_size,
            num_qo_heads,
            head_dim_ckv,
            head_dim_kpe,
            page_size,
            s_kv,
            max_seq_len_q=s_qo,
            packed_query=False,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
            o_data_type=torch.bfloat16,
            mask_type="causal",
            workspace_buffer=workspace_buffer,
            tuning=prims_ts.MLAAttentionTuning(
                softmax_rescale_threshold_log2=mla_softmax_rescale_threshold_log2
            ),
        )

    direct_out = None
    if any(
        backend in backends
        for backend in (
            "trtllm-native",
            "auto",
            "cute-dsl",
            "cutile",
            "trtllm-gen",
        )
    ):
        direct_out = torch.empty(
            batch_size,
            s_qo,
            num_qo_heads,
            head_dim_ckv,
            device=device,
            dtype=torch.bfloat16,
        )
    rubin_mtp_out = (
        torch.empty(
            batch_size,
            s_qo,
            num_qo_heads,
            head_dim_ckv,
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        if "cute-dsl-rubin-mtp" in backends
        else None
    )

    def run_backend_wrapper(
        backend,
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        q,
        kv_cache,
        workspace_buffer,
        block_tables,
        actual_seq_lens_kv,
        out,
    ):
        """
        Run a single MLA decode backend and return its output tensor.

        Dispatches to the BatchMLAPagedAttentionWrapper for fa2/fa3/cutlass or
        to the direct trtllm_batch_decode_with_kv_cache_mla API for
        trtllm-native/auto/cute-dsl. The trtllm/auto/cute-dsl branches also
        forward the resolved MLA overrides (is_var_seq / cute_dsl_impl) via
        mla_api_extra_kwargs.
        """
        if backend in ["fa2", "fa3"]:
            # BatchMLAPagedAttentionWrapper.run() does not accept enable_pdl;
            # FA2/FA3 use their planned CSR metadata and do not accept the
            # CUTLASS-only page_table argument. trtllm-native/auto/cute-dsl
            # branches below pass args.enable_pdl to the direct API.
            return backend_wrappers[backend].run(
                q_nope,
                q_pe,
                ckv_cache,
                kpe_cache,
                return_lse=False,
            )
        elif backend == "cutlass":
            # BatchMLAPagedAttentionWrapper.run() does not accept enable_pdl.
            return backend_wrappers[backend].run(
                q_nope,
                q_pe,
                ckv_cache,
                kpe_cache,
                kv_len=actual_seq_lens_kv.flatten(),
                page_table=block_tables,
                return_lse=False,
            )
        elif backend == "trtllm-native":
            return flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
                query=q.view(
                    batch_size,
                    s_qo,
                    num_qo_heads,
                    head_dim_ckv + head_dim_kpe,
                ),
                kv_cache=kv_cache.unsqueeze(1),
                workspace_buffer=workspace_buffer,
                qk_nope_head_dim=qk_nope_head_dim,
                kv_lora_rank=head_dim_ckv,
                qk_rope_head_dim=head_dim_kpe,
                block_tables=direct_block_tables,
                seq_lens=actual_seq_lens_kv.flatten(),
                max_seq_len=s_kv,
                out=out,
                bmm1_scale=sm_scale,
                bmm2_scale=1.0,
                backend="trtllm-gen",
                enable_pdl=args.enable_pdl,
                multi_ctas_kv_counter_buffer=mla_multi_ctas_kv_counter_buffer,
                **mla_sparse_kwargs,
                **mla_fp16_softmax_kwargs,
                **mla_api_extra_kwargs,
            ).reshape(-1, num_qo_heads, head_dim_ckv)
        elif backend in ("cutile", "trtllm-gen", "cute-dsl-rubin-mtp"):
            return backend_wrappers[backend].run(
                query=q.view(-1, num_qo_heads, head_dim_ckv + head_dim_kpe),
                kv_cache=kv_cache,
                out=out.view(-1, num_qo_heads, head_dim_ckv),
            )
        elif backend == "auto":
            # Autotune dispatcher: picks between trtllm-gen and cute-dsl per
            # input shape. Becomes meaningful when combined with --autotune,
            # which pre-tunes the cache before the timed bench loop.
            return flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
                query=q.view(
                    batch_size,
                    s_qo,
                    num_qo_heads,
                    head_dim_ckv + head_dim_kpe,
                ),
                kv_cache=kv_cache.unsqueeze(1),
                workspace_buffer=workspace_buffer,
                qk_nope_head_dim=qk_nope_head_dim,
                kv_lora_rank=head_dim_ckv,
                qk_rope_head_dim=head_dim_kpe,
                block_tables=direct_block_tables,
                seq_lens=actual_seq_lens_kv.flatten(),
                max_seq_len=s_kv,
                out=out,
                bmm1_scale=sm_scale,
                bmm2_scale=1.0,
                backend="auto",
                enable_pdl=args.enable_pdl,
                **mla_sparse_kwargs,
                **mla_api_extra_kwargs,
            ).reshape(-1, num_qo_heads, head_dim_ckv)
        elif backend == "cute-dsl":
            return flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
                query=q.view(
                    batch_size,
                    s_qo,
                    num_qo_heads,
                    head_dim_ckv + head_dim_kpe,
                ),
                kv_cache=kv_cache.unsqueeze(1),
                workspace_buffer=workspace_buffer,
                qk_nope_head_dim=qk_nope_head_dim,
                kv_lora_rank=head_dim_ckv,
                qk_rope_head_dim=head_dim_kpe,
                block_tables=block_tables,
                seq_lens=actual_seq_lens_kv.flatten(),
                max_seq_len=s_kv,
                out=out,
                bmm1_scale=sm_scale,
                bmm2_scale=1.0,
                backend="cute-dsl",
                enable_pdl=args.enable_pdl,
                **mla_api_extra_kwargs,
            ).reshape(-1, num_qo_heads, head_dim_ckv)
        elif backend == "prims-ts":
            return (
                backend_wrappers[backend]
                .run(
                    q.view(
                        batch_size,
                        s_qo,
                        num_qo_heads,
                        head_dim_ckv + head_dim_kpe,
                    ),
                    kv_cache,
                    block_tables,
                    actual_seq_lens_kv.flatten(),
                    bmm1_scale=sm_scale,
                    bmm2_scale=1.0,
                    out=out,
                    validate=False,
                )
                .reshape(-1, num_qo_heads, head_dim_ckv)
            )
        else:
            print(f"[ERROR] Unsupported backend: {backend}")
            return None

    # Autotune warmup: pre-tunes supported backends so the steady-state bench
    # reflects the chosen tactic rather than the fallback. Only the ``auto``
    # backend has runner choice today (it profiles both trtllm-gen and cute-dsl
    # internally).
    autotune_supported_backends = {"auto"}
    cache_path = getattr(args, "autotune_cache", None)
    if getattr(args, "autotune", False):
        warmup_iters = (
            args.dry_run_iters if args.dry_run_iters and args.dry_run_iters > 0 else 10
        )
        for cur_backend in backends:
            if cur_backend in autotune_supported_backends:
                if args.verbose >= 1:
                    print(
                        f"[INFO] Autotune warmup for BatchMLAPagedAttentionWrapper "
                        f"backend={cur_backend}: {warmup_iters} iters"
                    )
                workspace_buffer.zero_()
                with autotune(True, cache=cache_path):
                    for _ in range(warmup_iters):
                        run_backend_wrapper(
                            cur_backend,
                            q_nope,
                            q_pe,
                            ckv_cache,
                            kpe_cache,
                            q,
                            kv_cache,
                            workspace_buffer,
                            block_tables,
                            actual_seq_lens_kv,
                            direct_out,
                        )
    elif cache_path:
        with autotune(False, cache=cache_path):
            pass

    has_reference_output = False
    reference_backend = None
    # Iterate over each backend:
    for cur_backend in backends:
        # Clear workspace buffer to prevent unexpected interactions between backends.
        workspace_buffer.zero_()
        legacy_wrapper = cur_backend in ("fa2", "fa3", "cutlass")
        direct_backend = cur_backend in (
            "trtllm-native",
            "auto",
            "cute-dsl",
            "cutile",
            "trtllm-gen",
            "cute-dsl-rubin-mtp",
        )
        runtime_q_nope = q_nope if legacy_wrapper else None
        runtime_q_pe = q_pe if legacy_wrapper else None
        runtime_ckv_cache = ckv_cache if legacy_wrapper else None
        runtime_kpe_cache = kpe_cache if legacy_wrapper else None
        runtime_q = q if (direct_backend or cur_backend == "prims-ts") else None
        runtime_kv_cache = (
            kv_cache if (direct_backend or cur_backend == "prims-ts") else None
        )
        runtime_workspace = None if cur_backend == "prims-ts" else workspace_buffer
        runtime_out = (
            prims_ts_out
            if cur_backend == "prims-ts"
            else rubin_mtp_out
            if cur_backend == "cute-dsl-rubin-mtp"
            else direct_out
            if direct_backend
            else None
        )
        if run_refcheck:
            if runtime_out is not None:
                # direct_out is shared by the direct-API / wrapper-native
                # backends; poison it so a backend that does not write `out`
                # cannot pass by matching the previous backend's result.
                runtime_out.fill_(float("nan"))
            outputs[cur_backend] = (
                run_backend_wrapper(
                    cur_backend,
                    runtime_q_nope,
                    runtime_q_pe,
                    runtime_ckv_cache,
                    runtime_kpe_cache,
                    runtime_q,
                    runtime_kv_cache,
                    runtime_workspace,
                    block_tables,
                    actual_seq_lens_kv,
                    runtime_out,
                )
                .detach()
                .clone()
            )
            if cur_backend == "fa2":
                has_reference_output = True
                reference_output = outputs[cur_backend]
                reference_backend = "fa2"

        def run_timed_backend(
            q_nope_arg,
            q_pe_arg,
            ckv_cache_arg,
            kpe_cache_arg,
            q_arg,
            kv_cache_arg,
            out_arg,
        ):
            return run_backend_wrapper(
                cur_backend,
                q_nope_arg,
                q_pe_arg,
                ckv_cache_arg,
                kpe_cache_arg,
                q_arg,
                kv_cache_arg,
                workspace_buffer,
                block_tables,
                actual_seq_lens_kv,
                out_arg,
            )

        backend_times[cur_backend] = bench_gpu_time(
            fn=run_timed_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=False,
            enable_cupti=args.use_cupti,
            use_cuda_graph=(is_cuda_graph_compatible and cur_backend != "fa2"),
            cold_l2_cache=True,
            input_args=(
                runtime_q_nope,
                runtime_q_pe,
                runtime_ckv_cache,
                runtime_kpe_cache,
                runtime_q,
                runtime_kv_cache,
                runtime_out,
            ),
        )

    # Perform reference check
    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if run_refcheck and "prims-ts" in outputs and is_cuda_graph_compatible:
        graph_output = _replay_cuda_graph_once(
            lambda: backend_wrappers["prims-ts"].run(
                q.view(
                    batch_size,
                    s_qo,
                    num_qo_heads,
                    head_dim_ckv + head_dim_kpe,
                ),
                kv_cache,
                block_tables,
                actual_seq_lens_kv.flatten(),
                bmm1_scale=sm_scale,
                bmm2_scale=1.0,
                out=prims_ts_out,
                validate=False,
            ),
            prims_ts_out,
        )
        _validate_graph_output(graph_output, outputs["prims-ts"], rtol, atol)
        if args.verbose >= 1:
            print("[INFO] prims-ts CUDA-graph replay matched eager MLA output.")
    if run_refcheck and not has_reference_output and len(tested_backends) > 1:
        reference_backend, reference_output = _select_reference_output(
            outputs,
            ["trtllm-native", "auto", "cute-dsl", "cutlass"],
        )
        has_reference_output = reference_backend is not None
    if len(tested_backends) > 1:
        if run_refcheck and has_reference_output:
            fp8_types = [torch.float8_e4m3fn, torch.float8_e5m2]
            if reference_output.dtype in fp8_types or any(
                output.dtype in fp8_types for output in tested_outputs
            ):
                # FP8-output backends (cute-dsl-rubin-mtp) compare in fp32.
                reference_output = reference_output.to(torch.float32)
                tested_outputs = [output.to(torch.float32) for output in tested_outputs]
            for i in range(len(tested_outputs)):
                (
                    num_different_elements,
                    num_elements,
                    num_different_elements_percentage,
                ) = is_close_stats(reference_output, tested_outputs[i], rtol, atol)
                if num_different_elements > 0:
                    print(
                        f"[ERROR] Output tensor mismatch between backends {reference_backend} and {tested_backends[i]}: "
                        f"{num_different_elements} / {num_elements} ({num_different_elements_percentage:.2f}%) elements are different"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {tested_backends[i]} output mismatch"
                        )
    if run_refcheck and sparse_mla_top_k > 0 and outputs:
        # trtllm-native and auto share one sparse kernel, so the cross-backend
        # compare above cannot catch a wrong index contract; check sampled
        # query rows against an FP32 gather-attention oracle instead.
        try:
            sample_count = _validate_mla_sparse_samples(
                q=q,
                kv_cache=kv_cache,
                sparse_indices=direct_block_tables,
                output=next(iter(outputs.values())),
                num_qo_heads=num_qo_heads,
                head_dim_ckv=head_dim_ckv,
                sm_scale=sm_scale,
                rtol=rtol,
                # FP8 kernels also quantize P to e4m3 before the PV MMA, which
                # the FP32 oracle does not model; short top-k rows (few, large
                # probabilities) expose it.
                atol=(5e-2 if kv_dtype == torch.float8_e4m3fn else atol),
            )
            if args.verbose >= 1:
                print(
                    f"[INFO] Sparse MLA sampled FP32 reference passed: "
                    f"{sample_count} query rows"
                )
        except AssertionError as error:
            print(f"[ERROR] Sparse MLA sampled FP32 reference mismatch: {error}")
            if not args.allow_output_mismatch:
                raise
    # Compute perf metrics
    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            actual_seq_lens_kv_flat = actual_seq_lens_kv.flatten().to("cpu")
            actual_seq_lens_q_flat = torch.full_like(
                actual_seq_lens_kv.flatten().to("cpu"), s_qo
            )

            # Query bytes (q_nope + q_pe): batch_size * num_heads * head_dim
            q_mem_bytes = (
                q_nope.numel() * q_nope.element_size()
                + q_pe.numel() * q_pe.element_size()
            )

            # KV cache bytes: based on actual sequence lengths accessed, not full allocation
            actual_kv_tokens = actual_seq_lens_kv_flat.sum().item()
            kv_elem_size = ckv_cache.element_size()  # Same dtype for ckv and kpe
            kv_mem_bytes = (
                actual_kv_tokens * (head_dim_ckv + head_dim_kpe) * kv_elem_size
            )

            # Output bytes: batch_size * num_heads * head_dim_ckv
            o_elem_size = torch.empty((), dtype=torch.bfloat16).element_size()
            o_mem_bytes = batch_size * s_qo * num_qo_heads * head_dim_ckv * o_elem_size

            total_mem_bytes = q_mem_bytes + kv_mem_bytes + o_mem_bytes
            tb_per_sec = total_mem_bytes / (median_time * 1e9)
            attended_pairs = (
                torch.dot(
                    actual_seq_lens_q_flat.to(torch.float32),
                    2 * actual_seq_lens_kv_flat.to(torch.float32)
                    - actual_seq_lens_q_flat.to(torch.float32)
                    + 1,
                )
                / 2
            )
            if sparse_mla_top_k > 0:
                # Token-sparse rows read and attend only their selected tokens.
                attended_pairs = sparse_active_rows.sum().to("cpu", torch.float32)
                kv_mem_bytes = (
                    int(attended_pairs.item())
                    * (head_dim_ckv + head_dim_kpe)
                    * kv_elem_size
                )
                total_mem_bytes = q_mem_bytes + kv_mem_bytes + o_mem_bytes
                tb_per_sec = total_mem_bytes / (median_time * 1e9)
            tflops_total = (
                2 * attended_pairs * num_qo_heads * (2 * head_dim_ckv + head_dim_kpe)
            )
            tflops = (tflops_total / (median_time * 1e9)).item()

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            # TO-Do:
            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["backend"] = backend
                cur_res["page_size"] = page_size
                cur_res["batch_size"] = batch_size
                cur_res["s_qo"] = s_qo
                cur_res["s_kv"] = s_kv
                cur_res["num_qo_heads"] = num_qo_heads
                cur_res["head_dim_ckv"] = head_dim_ckv
                cur_res["head_dim_kpe"] = head_dim_kpe
                cur_res["causal"] = causal
                cur_res["q_dtype"] = q_dtype
                cur_res["kv_dtype"] = kv_dtype
                cur_res["avg_actual_seq_len"] = avg_seq_len_kv
                cur_res["random_actual_seq_len"] = args.random_actual_seq_len
                # Leave empty (null) when not explicitly overridden so legacy
                # var-seq rows keep matching historical null baselines.
                if resolved_is_var_seq is not None:
                    cur_res["is_var_seq"] = resolved_is_var_seq
                # Same null-preserving rule for cute_dsl_impl.
                if mla_cute_dsl_impl_arg is not None:
                    cur_res["cute_dsl_impl"] = mla_cute_dsl_impl_arg
                cur_res["sparse_mla_top_k"] = sparse_mla_top_k
                # use_fp16_softmax only reaches the trtllm-native call.
                cur_res["use_fp16_softmax"] = bool(
                    args.use_fp16_softmax and backend == "trtllm-native"
                )
                # Same null-preserving rule for the prims-ts skip-correction threshold.
                if backend == "prims-ts" and mla_softmax_rescale_threshold_log2 != 0.0:
                    cur_res["softmax_rescale_threshold_log2"] = (
                        mla_softmax_rescale_threshold_log2
                    )
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def _mla_sparse_token_indices(block_tables, kv_lens, s_qo, top_k, page_size):
    """Select top_k causally visible KV tokens per query row.

    Returns physical token slots ``[batch, s_qo, top_k]`` (page_id * page_size
    + offset, -1 padded), the layout trtllm-gen sparse MLA reads in place of
    the page table, and the active token count of every query row.
    """
    batch_size, pages_per_seq = block_tables.shape
    device = block_tables.device
    max_kv = pages_per_seq * page_size
    # Query j of a request sees tokens [0, kv_len - s_qo + j].
    visible = (
        kv_lens.to(torch.int64)[:, None]
        - s_qo
        + 1
        + torch.arange(s_qo, device=device, dtype=torch.int64)[None, :]
    )
    scores = torch.rand(batch_size, s_qo, max_kv, device=device)
    positions = torch.arange(max_kv, device=device, dtype=torch.int64)
    scores.masked_fill_(positions[None, None, :] >= visible[:, :, None], -1.0)
    width = min(top_k, max_kv)
    values, logical = scores.topk(width, dim=-1)
    pages = torch.gather(
        block_tables.to(torch.int64), 1, (logical // page_size).view(batch_size, -1)
    ).view(batch_size, s_qo, width)
    slots = pages * page_size + logical % page_size
    slots = torch.where(values >= 0, slots, -1)
    if width < top_k:
        slots = torch.nn.functional.pad(slots, (0, top_k - width), value=-1)
    active_rows = visible.clamp(min=0, max=top_k)
    return slots.to(torch.int32).contiguous(), active_rows


@torch.inference_mode()
def _validate_mla_sparse_samples(
    *,
    q,
    kv_cache,
    sparse_indices,
    output,
    num_qo_heads,
    head_dim_ckv,
    sm_scale,
    rtol,
    atol,
    sample_limit=8,
):
    """Check sampled sparse MLA query rows against an FP32 oracle."""
    rows = sparse_indices.view(-1, sparse_indices.shape[-1])
    queries = q.view(-1, num_qo_heads, q.shape[-1])
    outputs = output.view(-1, num_qo_heads, head_dim_ckv)
    kv_rows = kv_cache.reshape(-1, kv_cache.shape[-1])
    total_rows = rows.shape[0]
    sample_rows = sorted(
        {0, total_rows // 4, total_rows // 2, (3 * total_rows) // 4, total_rows - 1}
    )[:sample_limit]
    for row in sample_rows:
        slots = rows[row]
        selected = kv_rows.index_select(0, slots[slots >= 0].to(torch.int64)).float()
        scores = torch.matmul(queries[row].float(), selected.T) * sm_scale
        expected = torch.matmul(
            torch.softmax(scores, dim=-1), selected[:, :head_dim_ckv]
        )
        num_different, num_elements, _ = is_close_stats(
            expected, outputs[row].float(), rtol, atol
        )
        if num_different > 0:
            max_abs_error = (expected - outputs[row].float()).abs().max().item()
            raise AssertionError(
                f"query row {row}: {num_different} / {num_elements} elements differ "
                f"(max_abs_error={max_abs_error:.4g})"
            )
    return len(sample_rows)


_DSV4_HEAD_DIM = 512
_DSV4_SWA_TOPK = 128
_DSV4_HCA_RATIO = 128  # raw tokens per HCA compressed row
# Sampled-oracle bound on ||out - ref|| / ||ref|| per query row (FP8 Q/KV
# quantization stays well below it; zero or wrong-row outputs give >= ~1).
_DSV4_REFCHECK_REL_TOL = 0.1
_DSV4_WORKSPACE_BYTES = 128 * 1024 * 1024


_CONCAT_MLA_HEADS = 128
_CONCAT_MLA_NOPE_DIM = 128
_CONCAT_MLA_ROPE_DIM = 64


def testConcatMlaK(args):
    """Benchmark concat_mla_k, the MLA prefill key assembly k = [k_nope | k_rope].

    num_tokens = batch_size * s_kv. The kernel shape is fixed (DeepSeek-V3/R1:
    128 heads, nope 128, rope 64). As in the model and bench_concat_mla.py,
    k_nope is a strided view of the [nope | v] kv_b_proj output and k_rope a
    strided view of the shared rope row. Backend: cuda (backend="default").
    """
    if args.verbose >= 1:
        print("[INFO] Running testConcatMlaK")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            "[INFO] To reproduce this test case, run the following command: "
            f"{args.repro_command}"
        )

    backends = filter_backends_by_compute_capability(
        list(args.backends), args.routine, device
    )
    if not backends:
        print("[ERROR] No backends to test. Exiting.")
        return []
    head_dim_qk = args.head_dim_qk or _CONCAT_MLA_NOPE_DIM + _CONCAT_MLA_ROPE_DIM
    head_dim_kpe = args.head_dim_kpe or _CONCAT_MLA_ROPE_DIM
    if (
        args.num_kv_heads != _CONCAT_MLA_HEADS
        or head_dim_qk != _CONCAT_MLA_NOPE_DIM + _CONCAT_MLA_ROPE_DIM
        or head_dim_kpe != _CONCAT_MLA_ROPE_DIM
    ):
        raise ValueError(
            "concat_mla_k requires num_kv_heads=128, head_dim_qk=192, head_dim_kpe=64"
        )
    dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    if dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        raise ValueError("concat_mla_k supports bfloat16, float16, fp8_e4m3, fp8_e5m2")

    num_tokens = args.batch_size * args.s_kv
    k_nope = torch.randn(
        num_tokens,
        _CONCAT_MLA_HEADS,
        2 * _CONCAT_MLA_NOPE_DIM,
        dtype=torch.float32,
        device=device,
    ).to(dtype)[:, :, :_CONCAT_MLA_NOPE_DIM]
    k_rope = torch.randn(
        num_tokens,
        1,
        _CONCAT_MLA_NOPE_DIM + _CONCAT_MLA_ROPE_DIM,
        dtype=torch.float32,
        device=device,
    ).to(dtype)[:, :, -_CONCAT_MLA_ROPE_DIM:]
    k = torch.empty(
        num_tokens,
        _CONCAT_MLA_HEADS,
        _CONCAT_MLA_NOPE_DIM + _CONCAT_MLA_ROPE_DIM,
        dtype=dtype,
        device=device,
    )

    from flashinfer.concat_ops import concat_mla_k

    def run_backend(backend, k_arg, k_nope_arg, k_rope_arg):
        concat_mla_k(
            k_arg,
            k_nope_arg,
            k_rope_arg,
            backend="default" if backend == "cuda" else backend,
        )
        return k_arg

    results = []
    for backend in backends:
        if args.refcheck:
            k.zero_()
            run_backend(backend, k, k_nope, k_rope)
            expected = torch.cat(
                (k_nope, k_rope.expand(-1, _CONCAT_MLA_HEADS, -1)), dim=-1
            )
            # A pure copy must be bit-exact.
            if not torch.equal(k.view(torch.uint8), expected.view(torch.uint8)):
                print(f"[ERROR] concat_mla_k {backend} output differs from torch.cat")
                if not args.allow_output_mismatch:
                    raise AssertionError(f"[ERROR] Backend {backend} output mismatch")

        times = bench_gpu_time(
            fn=lambda *inputs, backend=backend: run_backend(backend, *inputs),
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=False,
            enable_cupti=args.use_cupti,
            use_cuda_graph=not args.no_cuda_graph,
            cold_l2_cache=True,
            input_args=(k, k_nope, k_rope),
        )
        median_time = float(np.median(times))
        std_time = float(np.std(times))
        # Reads k_nope + one rope row per token, writes k.
        total_bytes = (k_nope.numel() + k_rope.numel() + k.numel()) * k.element_size()
        tb_per_sec = total_bytes / (median_time * 1e9)
        print_perf_metrics(backend, median_time, std_time, 0.0, tb_per_sec)

        result = defaultdict(str)
        result.update(
            {
                "routine": args.routine,
                "median_time": median_time,
                "std_time": std_time,
                "tflops": 0.0,
                "tb_per_sec": tb_per_sec,
                "backend": backend,
                "batch_size": args.batch_size,
                "s_kv": args.s_kv,
                "num_qo_heads": args.num_qo_heads,
                "num_kv_heads": args.num_kv_heads,
                "head_dim_qk": head_dim_qk,
                "head_dim_kpe": head_dim_kpe,
                "kv_dtype": args.kv_dtype,
                "case_tag": args.case_tag,
            }
        )
        results.append(result)
    return results


def testConcatMlaKvQuantFp8(args):
    """Benchmark concat_mla_kv_quant_fp8, the fused MLA context K/V pack + FP8 cast.

    kv_nope = [k_nope | v] (bf16 [T, H, 256]) and the shared, RoPE'd k_pe
    (bf16 [T, 64]) become fp8 key = [k_nope | k_pe] ([T, H, 192]) and fp8
    value = v ([T, H, 128]), the separate-K/V inputs of an FP8 ragged MLA
    prefill. num_tokens = batch_size * s_kv; H = --num_kv_heads (local heads
    after tensor parallelism). Backends: specialized (the CUDA kernel,
    SM100+) and auto (the library's per-head-count choice; the CSV
    resolved_backend names the kernel that served the call). A backend that
    would silently take the composable torch fallback is skipped.
    """
    if args.verbose >= 1:
        print("[INFO] Running testConcatMlaKvQuantFp8")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            "[INFO] To reproduce this test case, run the following command: "
            f"{args.repro_command}"
        )

    backends = filter_backends_by_compute_capability(
        list(args.backends), args.routine, device
    )
    if not backends:
        print("[ERROR] No backends to test. Exiting.")
        return []
    if args.kv_dtype != "bfloat16":
        raise ValueError("concat_mla_kv_quant_fp8 takes bfloat16 inputs only")
    from flashinfer import mla_kv_pack

    nope_dim, rope_dim, v_dim = (
        _CONCAT_MLA_NOPE_DIM,
        _CONCAT_MLA_ROPE_DIM,
        _CONCAT_MLA_NOPE_DIM,
    )
    num_heads = args.num_kv_heads
    num_tokens = args.batch_size * args.s_kv
    kv_nope = torch.randn(
        num_tokens, num_heads, nope_dim + v_dim, dtype=torch.bfloat16, device=device
    )
    k_pe = torch.randn(num_tokens, rope_dim, dtype=torch.bfloat16, device=device)
    key = torch.empty(
        num_tokens,
        num_heads,
        nope_dim + rope_dim,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    value = torch.empty(
        num_tokens, num_heads, v_dim, dtype=torch.float8_e4m3fn, device=device
    )

    def run_backend(backend, kv_nope_arg, k_pe_arg, key_arg, value_arg):
        return mla_kv_pack.concat_mla_kv_quant_fp8(
            kv_nope_arg, k_pe_arg, key_arg, value_arg, backend=backend
        )

    def served_by(backend):
        # Dispatch counters tell which fused kernel (if any) ran the call.
        before = mla_kv_pack._concat_mla_kv_quant_fp8_stats()["backend_dispatches"]
        run_backend(backend, kv_nope, k_pe, key, value)
        torch.cuda.synchronize()
        after = mla_kv_pack._concat_mla_kv_quant_fp8_stats()["backend_dispatches"]
        for name, count in after.items():
            if count > before.get(name, 0):
                return name
        return None

    results = []
    for backend in backends:
        resolved = served_by(backend)
        if resolved is None:
            print(
                f"[INFO] {backend} backend takes the composable torch fallback for "
                "this case (no fused kernel). Skipping."
            )
            continue
        if args.refcheck:
            key.zero_()
            value.zero_()
            run_backend(backend, kv_nope, k_pe, key, value)
            ref_key = torch.cat(
                (
                    kv_nope[..., :nope_dim],
                    k_pe.unsqueeze(1).expand(-1, num_heads, -1),
                ),
                dim=-1,
            ).to(torch.float8_e4m3fn)
            ref_value = kv_nope[..., nope_dim:].to(torch.float8_e4m3fn)
            # The library documents bit-exactness with Tensor.to(float8_e4m3fn).
            if not (
                torch.equal(key.view(torch.uint8), ref_key.view(torch.uint8))
                and torch.equal(value.view(torch.uint8), ref_value.view(torch.uint8))
            ):
                print(
                    f"[ERROR] concat_mla_kv_quant_fp8 {backend} output differs from "
                    "the torch cast + copy"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(f"[ERROR] Backend {backend} output mismatch")

        times = bench_gpu_time(
            fn=lambda *inputs, backend=backend: run_backend(backend, *inputs),
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=False,
            enable_cupti=args.use_cupti,
            use_cuda_graph=not args.no_cuda_graph,
            cold_l2_cache=True,
            input_args=(kv_nope, k_pe, key, value),
        )
        median_time = float(np.median(times))
        std_time = float(np.std(times))
        # Reads kv_nope (bf16) + one k_pe row per token, writes fp8 key and value.
        total_bytes = (
            kv_nope.numel() * 2 + k_pe.numel() * 2 + key.numel() + value.numel()
        )
        tb_per_sec = total_bytes / (median_time * 1e9)
        print_perf_metrics(
            backend if backend == resolved else f"{backend}({resolved})",
            median_time,
            std_time,
            0.0,
            tb_per_sec,
        )

        result = defaultdict(str)
        result.update(
            {
                "routine": args.routine,
                "median_time": median_time,
                "std_time": std_time,
                "tflops": 0.0,
                "tb_per_sec": tb_per_sec,
                "backend": backend,
                "resolved_backend": resolved,
                "batch_size": args.batch_size,
                "s_kv": args.s_kv,
                "num_kv_heads": num_heads,
                "head_dim_qk": nope_dim + rope_dim,
                "head_dim_vo": v_dim,
                "head_dim_kpe": rope_dim,
                "kv_dtype": args.kv_dtype,
                "out_dtype": "fp8_e4m3",
                "case_tag": args.case_tag,
            }
        )
        results.append(result)
    return results


def _dsv4_cache_to_flat_rows(cache, kv_layout):
    if kv_layout == "HND":
        cache = cache.transpose(1, 2)
    return cache.reshape(-1, cache.shape[-1])


def _dsv4_coprime_strides(lengths):
    """Choose deterministic, non-unit strides for sparse-index permutations."""
    strides = []
    for length in lengths.to("cpu").tolist():
        if length <= 1:
            strides.append(1)
            continue
        candidate = max(1, length // 2 + 1)
        if candidate >= length:
            candidate = 1
        while math.gcd(candidate, length) != 1:
            candidate += 1
            if candidate >= length:
                candidate = 1
        strides.append(candidate)
    return torch.tensor(strides, dtype=torch.int64, device=lengths.device)


def _dsv4_make_sparse_indices(
    *,
    batch_size,
    s_qo,
    s_kv,
    swa_topk,
    swa_page_size,
    compressed_topk,
    compressed_kv_len,
    compressed_page_size,
    device,
):
    """Build causal SWA rows plus causal compressed-cache selections."""
    query_positions = torch.arange(s_kv - s_qo, s_kv, dtype=torch.int64, device=device)

    swa_valid_lens = torch.minimum(
        query_positions + 1,
        torch.tensor(swa_topk, dtype=torch.int64, device=device),
    )
    swa_columns = torch.arange(swa_topk, dtype=torch.int64, device=device)
    swa_starts = query_positions - swa_valid_lens + 1
    swa_local = swa_starts[:, None] + swa_columns[None, :]
    swa_local = torch.where(
        swa_columns[None, :] < swa_valid_lens[:, None], swa_local, -1
    )

    # DSV4's compressed pool stores one row per four raw KV tokens. The active
    # top-k grows causally during a full-prompt prefill and saturates at the
    # configured maximum.
    compressed_available = torch.minimum(
        torch.div(query_positions + 4, 4, rounding_mode="floor"),
        torch.tensor(compressed_kv_len, dtype=torch.int64, device=device),
    ).clamp_min(1)
    compressed_active_lens = torch.minimum(
        compressed_available,
        torch.tensor(compressed_topk, dtype=torch.int64, device=device),
    )
    compressed_columns = torch.arange(compressed_topk, dtype=torch.int64, device=device)
    compressed_strides = _dsv4_coprime_strides(compressed_available)
    compressed_local = (
        compressed_columns[None, :] * compressed_strides[:, None]
        + query_positions[:, None] * 29
    ) % compressed_available[:, None]
    compressed_local = torch.where(
        compressed_columns[None, :] < compressed_active_lens[:, None],
        compressed_local,
        -1,
    )

    swa_rows_per_request = math.ceil(s_kv / swa_page_size) * swa_page_size
    compressed_rows_per_request = (
        math.ceil(compressed_kv_len / compressed_page_size) * compressed_page_size
    )
    swa_bases = (
        torch.arange(batch_size, dtype=torch.int64, device=device)
        * swa_rows_per_request
    )
    compressed_bases = (
        torch.arange(batch_size, dtype=torch.int64, device=device)
        * compressed_rows_per_request
    )
    swa_indices = torch.where(
        swa_local[None, :, :] >= 0,
        swa_local[None, :, :] + swa_bases[:, None, None],
        -1,
    )
    compressed_indices = torch.where(
        compressed_local[None, :, :] >= 0,
        compressed_local[None, :, :] + compressed_bases[:, None, None],
        -1,
    )
    sparse_indices = torch.cat((swa_indices, compressed_indices), dim=-1)
    sparse_indices = sparse_indices.reshape(
        batch_size * s_qo, swa_topk + compressed_topk
    ).to(torch.int32)

    compressed_active_lens = compressed_active_lens.repeat(batch_size)
    sparse_topk_lens = (compressed_active_lens + swa_topk).to(torch.int32)
    active_kv_lens = (swa_valid_lens + compressed_active_lens[:s_qo]).repeat(batch_size)
    return sparse_indices.contiguous(), sparse_topk_lens.contiguous(), active_kv_lens


def _dsv4_make_hca_sparse_indices(
    *,
    batch_size,
    s_qo,
    s_kv,
    swa_page_size,
    compressed_kv_len,
    compressed_page_size,
    device,
):
    """Build HCA rows: the causal SWA-128 window plus every visible compressed row.

    The compressed segment is the canonical page-aligned expansion that
    convert_compressed_page_aligned_sparse_indices_to_hca_metadata accepts, so
    the same table drives both trtllm-gen and the cute-dsl HCA kernel. Each row
    holds 128 + floor(visible / 128) active entries.
    """
    query_positions = torch.arange(s_kv - s_qo, s_kv, dtype=torch.int64, device=device)
    visible = query_positions + 1
    swa_valid_lens = visible.clamp(max=_DSV4_SWA_TOPK)
    swa_columns = torch.arange(_DSV4_SWA_TOPK, dtype=torch.int64, device=device)
    swa_local = (visible - swa_valid_lens)[:, None] + swa_columns[None, :]
    swa_local = torch.where(
        swa_columns[None, :] < swa_valid_lens[:, None], swa_local, -1
    )

    compressed_active_lens = torch.div(
        visible, _DSV4_HCA_RATIO, rounding_mode="floor"
    ).clamp(max=compressed_kv_len)
    # Index capacity must be a multiple of 4 (HCA metadata contract).
    compressed_capacity = max(4, -(-int(compressed_active_lens.max().item()) // 4) * 4)
    compressed_columns = torch.arange(
        compressed_capacity, dtype=torch.int64, device=device
    )
    compressed_local = torch.where(
        compressed_columns[None, :] < compressed_active_lens[:, None],
        compressed_columns[None, :],
        -1,
    )

    swa_rows_per_request = math.ceil(s_kv / swa_page_size) * swa_page_size
    compressed_rows_per_request = (
        math.ceil(compressed_kv_len / compressed_page_size) * compressed_page_size
    )
    request_ids = torch.arange(batch_size, dtype=torch.int64, device=device)
    swa_indices = torch.where(
        swa_local[None] >= 0,
        swa_local[None] + (request_ids * swa_rows_per_request)[:, None, None],
        -1,
    )
    compressed_indices = torch.where(
        compressed_local[None] >= 0,
        compressed_local[None]
        + (request_ids * compressed_rows_per_request)[:, None, None],
        -1,
    )
    sparse_indices = (
        torch.cat((swa_indices, compressed_indices), dim=-1)
        .reshape(batch_size * s_qo, _DSV4_SWA_TOPK + compressed_capacity)
        .to(torch.int32)
    )
    sparse_topk_lens = (
        (compressed_active_lens + _DSV4_SWA_TOPK).repeat(batch_size).to(torch.int32)
    )
    active_kv_lens = (swa_valid_lens + compressed_active_lens).repeat(batch_size)
    return sparse_indices.contiguous(), sparse_topk_lens.contiguous(), active_kv_lens


@torch.inference_mode()
def _validate_dsv4_sparse_mla_samples(
    *,
    query,
    swa_kv_cache,
    compressed_kv_cache,
    sparse_indices,
    sparse_topk_lens,
    output,
    swa_topk,
    bmm1_scale,
    bmm2_scale,
    kv_layout,
    sample_limit=8,
):
    """Check sampled query rows against an independent FP32 attention oracle."""
    if not torch.isfinite(output.float()).all():
        raise AssertionError("DSV4 sparse MLA output contains nonfinite values")

    total_q = query.shape[0]
    sample_rows = sorted(
        {
            0,
            total_q - 1,
            total_q // 4,
            total_q // 2,
            (3 * total_q) // 4,
        }
    )[:sample_limit]
    swa_rows = _dsv4_cache_to_flat_rows(swa_kv_cache, kv_layout)
    compressed_rows = _dsv4_cache_to_flat_rows(compressed_kv_cache, kv_layout)
    scale_qk = float(bmm1_scale.item()) if torch.is_tensor(bmm1_scale) else bmm1_scale
    scale_out = float(bmm2_scale.item()) if torch.is_tensor(bmm2_scale) else bmm2_scale
    max_abs_error = 0.0
    max_rel_error = 0.0

    for query_row in sample_rows:
        active_len = int(sparse_topk_lens[query_row].item())
        row_indices = sparse_indices[query_row]
        swa_indices = row_indices[:swa_topk]
        swa_indices = swa_indices[swa_indices >= 0].to(torch.int64)
        compressed_len = max(0, active_len - swa_topk)
        compressed_indices = row_indices[swa_topk : swa_topk + compressed_len].to(
            torch.int64
        )
        selected_kv = torch.cat(
            (
                swa_rows.index_select(0, swa_indices),
                compressed_rows.index_select(0, compressed_indices),
            ),
            dim=0,
        ).float()
        scores = torch.matmul(query[query_row].float(), selected_kv.T) * scale_qk
        expected = torch.matmul(torch.softmax(scores, dim=-1), selected_kv) * scale_out
        actual = output[query_row].float()
        max_abs_error = max(
            max_abs_error, float((actual - expected).abs().max().item())
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
        # The inputs are scaled by 0.05, so outputs are ~1e-3 and the absolute
        # check above alone would also accept an all-zero or wrong-row output;
        # require the error to be small relative to the output itself.
        rel_error = float(
            (
                torch.linalg.vector_norm(actual - expected)
                / torch.linalg.vector_norm(expected).clamp_min(1e-30)
            ).item()
        )
        max_rel_error = max(max_rel_error, rel_error)
        if not rel_error <= _DSV4_REFCHECK_REL_TOL:
            raise AssertionError(
                f"query row {query_row}: relative L2 error {rel_error:.4g} > "
                f"{_DSV4_REFCHECK_REL_TOL}"
            )
    return len(sample_rows), max_abs_error, max_rel_error


def testTrtllmBatchDecodeSparseMlaDsv4(args):
    """Benchmark the public DSV4 sparse-MLA API on SM100/SM103.

    Backends: trtllm-gen and (HCA layer only) cute-dsl.
    """
    if args.verbose >= 1:
        print("[INFO] Running testTrtllmBatchDecodeSparseMlaDsv4")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            "[INFO] To reproduce this test case, run the following command: "
            f"{args.repro_command}"
        )

    backends = filter_backends_by_compute_capability(
        list(args.backends), args.routine, device
    )
    if not backends:
        print("[ERROR] No backends to test. Exiting.")
        return []
    if args.random_actual_seq_len:
        raise ValueError(
            "trtllm_batch_decode_sparse_mla_dsv4 does not support "
            "--random_actual_seq_len"
        )
    if args.batch_size <= 0 or args.s_qo <= 0 or args.s_kv <= 0:
        raise ValueError("batch_size, s_qo, and s_kv must be positive")
    if args.s_qo > args.s_kv:
        raise ValueError("DSV4 sparse MLA requires s_qo <= s_kv")
    if args.num_qo_heads not in (8, 16, 32, 64, 128):
        raise ValueError("DSV4 sparse MLA num_qo_heads must be one of 8,16,32,64,128")
    if args.num_kv_heads != 1:
        raise ValueError("DSV4 sparse MLA requires num_kv_heads=1")
    if args.head_dim_qk != _DSV4_HEAD_DIM:
        raise ValueError("DSV4 sparse MLA requires head_dim_qk=512")
    head_dim_vo = args.head_dim_qk if args.head_dim_vo is None else args.head_dim_vo
    if head_dim_vo != _DSV4_HEAD_DIM:
        raise ValueError("DSV4 sparse MLA requires head_dim_vo=512")
    if args.swa_topk != _DSV4_SWA_TOPK:
        raise ValueError("DSV4 sparse MLA requires swa_topk=128")
    if args.page_size <= 0 or args.compressed_page_size <= 0:
        raise ValueError("page sizes must be positive")
    hca = args.dsv4_layer == "hca"
    if hca:
        # HCA compresses 128 raw tokens per row and attends every visible row,
        # so the compressed pool size follows from s_kv.
        compressed_kv_len = max(1, math.ceil(args.s_kv / _DSV4_HCA_RATIO))
        if (
            args.compressed_page_size <= 1
            or _DSV4_SWA_TOPK % args.compressed_page_size != 0
        ):
            raise ValueError("DSV4 HCA compressed_page_size must divide 128")
    else:
        compressed_kv_len = (
            math.ceil(args.s_kv / 4)
            if args.compressed_kv_len is None
            else args.compressed_kv_len
        )
        if args.compressed_topk <= 0 or compressed_kv_len < args.compressed_topk:
            raise ValueError("compressed_kv_len must be >= compressed_topk > 0")

    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    if q_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("DSV4 sparse MLA Q supports bfloat16 or fp8_e4m3")
    if kv_dtype != q_dtype:
        raise ValueError("DSV4 sparse MLA requires matching Q and KV dtypes")
    if args.out_dtype not in (None, "bfloat16"):
        raise ValueError("DSV4 sparse MLA output dtype is bfloat16")
    if "cute-dsl" in backends:
        # cute-dsl is the CuTe DSL HCA kernel (_core.py backend="cute-dsl").
        if not hca:
            _drop_backend(
                backends, "cute-dsl", "runs the HCA layer only (--dsv4_layer hca)"
            )
        elif q_dtype != torch.float8_e4m3fn:
            _drop_backend(backends, "cute-dsl", "HCA requires fp8_e4m3 Q/KV")
    if args.enable_pdl:
        _drop_backend(backends, "cute-dsl", "does not support --enable_pdl")
    if not backends:
        print("[ERROR] No backends to test. Exiting.")
        return []

    batch_size = args.batch_size
    total_q = batch_size * args.s_qo
    query = (
        torch.randn(
            total_q,
            args.num_qo_heads,
            _DSV4_HEAD_DIM,
            dtype=torch.float32,
            device=device,
        )
        .mul_(0.05)
        .clamp_(-1.0, 1.0)
        .to(q_dtype)
    )

    def make_cache(logical_len, page_size):
        pages_per_request = math.ceil(logical_len / page_size)
        cache = (
            torch.randn(
                batch_size * pages_per_request,
                page_size,
                1,
                _DSV4_HEAD_DIM,
                dtype=torch.float32,
                device=device,
            )
            .mul_(0.05)
            .clamp_(-1.0, 1.0)
            .to(kv_dtype)
        )
        if args.kv_layout == "HND":
            cache = cache.transpose(1, 2).contiguous()
        return cache

    swa_kv_cache = make_cache(args.s_kv, args.page_size)
    compressed_kv_cache = make_cache(compressed_kv_len, args.compressed_page_size)
    if hca:
        sparse_indices, sparse_topk_lens, active_kv_lens = (
            _dsv4_make_hca_sparse_indices(
                batch_size=batch_size,
                s_qo=args.s_qo,
                s_kv=args.s_kv,
                swa_page_size=args.page_size,
                compressed_kv_len=compressed_kv_len,
                compressed_page_size=args.compressed_page_size,
                device=device,
            )
        )
        compressed_topk = sparse_indices.shape[1] - _DSV4_SWA_TOPK
    else:
        sparse_indices, sparse_topk_lens, active_kv_lens = _dsv4_make_sparse_indices(
            batch_size=batch_size,
            s_qo=args.s_qo,
            s_kv=args.s_kv,
            swa_topk=args.swa_topk,
            swa_page_size=args.page_size,
            compressed_topk=args.compressed_topk,
            compressed_kv_len=compressed_kv_len,
            compressed_page_size=args.compressed_page_size,
            device=device,
        )
        compressed_topk = args.compressed_topk
    seq_lens = torch.full((batch_size,), args.s_kv, dtype=torch.int32, device=device)
    cum_seq_lens_q = torch.arange(
        0,
        (batch_size + 1) * args.s_qo,
        args.s_qo,
        dtype=torch.int32,
        device=device,
    )
    workspace = torch.empty(_DSV4_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    output = torch.empty(
        total_q,
        args.num_qo_heads,
        _DSV4_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    qk_scale_value = _DSV4_HEAD_DIM**-0.55
    bmm1_scale = (
        torch.tensor([qk_scale_value], dtype=torch.float32, device=device)
        if q_dtype == torch.float8_e4m3fn
        else qk_scale_value
    )
    bmm2_scale = (
        torch.tensor([1.0], dtype=torch.float32, device=device)
        if q_dtype == torch.float8_e4m3fn
        else 1.0
    )

    hca_metadata = None
    if "cute-dsl" in backends:
        # The HCA kernel consumes gather metadata built once from the same
        # compressed-page-aligned indices; the conversion synchronizes, so it
        # stays out of the timed (graph-captured) call.
        hca_metadata = flashinfer.mla.convert_compressed_page_aligned_sparse_indices_to_hca_metadata(
            sparse_indices,
            sparse_topk_lens,
            seq_lens,
            swa_kv_cache,
            compressed_kv_cache,
            q_len=args.s_qo,
            kv_layout=args.kv_layout,
        )
    float_bmm1_scale = qk_scale_value
    float_bmm2_scale = 1.0

    def run_backend(
        backend,
        query_arg,
        swa_cache_arg,
        compressed_cache_arg,
        sparse_indices_arg,
        sparse_topk_lens_arg,
        seq_lens_arg,
        cum_seq_lens_q_arg,
        output_arg,
    ):
        if backend == "cute-dsl":
            # Dense [B, Q, H, D] query, explicit HCA metadata, float scales.
            return flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
                query=query_arg.view(batch_size, args.s_qo, args.num_qo_heads, -1),
                swa_kv_cache=swa_cache_arg,
                workspace_buffer=workspace,
                compressed_kv_cache=compressed_cache_arg,
                sparse_topk_lens=sparse_topk_lens_arg,
                out=output_arg.view(batch_size, args.s_qo, args.num_qo_heads, -1),
                bmm1_scale=float_bmm1_scale,
                bmm2_scale=float_bmm2_scale,
                kv_layout=args.kv_layout,
                swa_topk_lens=hca_metadata.swa_topk_lens,
                hca_swa_indices=hca_metadata.hca_swa_indices,
                hca_compressed_block_tables=hca_metadata.hca_compressed_block_tables,
                hca_seq_lens=hca_metadata.hca_seq_lens,
                enable_pdl=False,
                backend="cute-dsl",
            )
        return flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
            query=query_arg,
            swa_kv_cache=swa_cache_arg,
            workspace_buffer=workspace,
            sparse_indices=sparse_indices_arg,
            compressed_kv_cache=compressed_cache_arg,
            sparse_topk_lens=sparse_topk_lens_arg,
            seq_lens=seq_lens_arg,
            out=output_arg,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            kv_layout=args.kv_layout,
            cum_seq_lens_q=cum_seq_lens_q_arg,
            max_q_len=args.s_qo,
            enable_pdl=args.enable_pdl,
            backend=backend,
        )

    runtime_args = (
        query,
        swa_kv_cache,
        compressed_kv_cache,
        sparse_indices,
        sparse_topk_lens,
        seq_lens,
        cum_seq_lens_q,
        output,
    )
    results = []
    for backend in backends:
        if args.refcheck:
            # Poison the shared output so a backend that writes nothing fails
            # instead of being checked against the previous backend's result.
            output.fill_(float("nan"))
            run_backend(backend, *runtime_args)
            try:
                (
                    sample_count,
                    max_abs_error,
                    max_rel_error,
                ) = _validate_dsv4_sparse_mla_samples(
                    query=query,
                    swa_kv_cache=swa_kv_cache,
                    compressed_kv_cache=compressed_kv_cache,
                    sparse_indices=sparse_indices,
                    sparse_topk_lens=sparse_topk_lens,
                    output=output,
                    swa_topk=args.swa_topk,
                    bmm1_scale=bmm1_scale,
                    bmm2_scale=bmm2_scale,
                    kv_layout=args.kv_layout,
                )
                if args.verbose >= 1:
                    print(
                        f"[INFO] DSV4 {backend} sampled FP32 reference passed: "
                        f"{sample_count} query rows, max_abs_error={max_abs_error:.6g}, "
                        f"max_rel_l2_error={max_rel_error:.4g}"
                    )
            except AssertionError as error:
                print(
                    f"[ERROR] DSV4 {backend} sampled FP32 reference mismatch: {error}"
                )
                if not args.allow_output_mismatch:
                    raise

        times = bench_gpu_time(
            fn=lambda *inputs, backend=backend: run_backend(backend, *inputs),
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=False,
            enable_cupti=args.use_cupti,
            use_cuda_graph=not args.no_cuda_graph,
            cold_l2_cache=True,
            input_args=runtime_args,
        )
        median_time = float(np.median(times))
        std_time = float(np.std(times))
        active_pairs = int(active_kv_lens.sum().item())
        flops = 4 * active_pairs * args.num_qo_heads * _DSV4_HEAD_DIM
        logical_bytes = (
            query.numel() * query.element_size()
            + active_pairs
            * _DSV4_HEAD_DIM
            * torch.empty((), dtype=kv_dtype).element_size()
            + output.numel() * output.element_size()
        )
        tflops = flops / (median_time * 1e9)
        tb_per_sec = logical_bytes / (median_time * 1e9)
        print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

        result = defaultdict(str)
        result.update(
            {
                "routine": args.routine,
                "median_time": median_time,
                "std_time": std_time,
                "tflops": tflops,
                "tb_per_sec": tb_per_sec,
                "backend": backend,
                "resolved_backend": backend,
                "page_size": args.page_size,
                "batch_size": batch_size,
                "s_qo": args.s_qo,
                "s_kv": args.s_kv,
                "num_qo_heads": args.num_qo_heads,
                "num_kv_heads": args.num_kv_heads,
                "head_dim_qk": _DSV4_HEAD_DIM,
                "head_dim_vo": _DSV4_HEAD_DIM,
                "causal": True,
                "q_dtype": args.q_dtype,
                "kv_dtype": args.kv_dtype,
                "out_dtype": "bfloat16",
                "avg_actual_seq_len": args.s_kv,
                "random_actual_seq_len": False,
                "swa_topk": args.swa_topk,
                "compressed_topk": compressed_topk,
                "compressed_kv_len": compressed_kv_len,
                "compressed_page_size": args.compressed_page_size,
                "dsv4_layer": args.dsv4_layer,
                "kv_layout": args.kv_layout,
                "case_tag": args.case_tag,
            }
        )
        results.append(result)
    return results
