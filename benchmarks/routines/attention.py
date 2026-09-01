from collections import defaultdict

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
)


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


def _drop_backend(backends, backend, reason):
    """Remove an unsupported backend while keeping CLI behavior non-fatal."""
    if backend in backends:
        print(f"[INFO] {backend} backend {reason}. Skipping.")
        backends.remove(backend)


def _get_prims_ts_module():
    """Import the experimental backend only when a benchmark requests it."""
    from flashinfer.attention import prims_ts

    return prims_ts


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
        default=["fa2"],
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
        ],
        help="Kernel backends to test. Default: fa2. prims-ts selects the experimental task-scheduled Blackwell backend for all attention routines. backend=auto is supported for BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper, and BatchMLAPagedAttentionWrapper (where it pairs with --autotune to select between trtllm-gen and cute-dsl).",
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

    args = parser.parse_args(line)

    # Normalize backend names (handle deprecated names)
    args.backends = normalize_backends(args.backends)
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

    if "cudnn" in backends:
        remove_cudnn = False
        if speculative_decode:
            print("[INFO] cuDNN backend does not support speculative decode. Skipping.")
            remove_cudnn = True
        if o_data_type != torch.bfloat16:
            print("[INFO] cuDNN decode requires BF16 output. Skipping.")
            remove_cudnn = True
        if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] or kv_dtype in [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]:
            print("[INFO] cuDNN backend does not support FP8. Skipping.")
            remove_cudnn = True
        if remove_cudnn:
            backends.remove("cudnn")

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
        if is_nvfp4_kv:
            _drop_backend(backends, "prims-ts", "does not support NVFP4 K/V")
        elif q_dtype != kv_dtype:
            _drop_backend(backends, "prims-ts", "requires matching Q and K/V dtypes")
        elif head_dim_qk != head_dim_vo or head_dim_qk not in (64, 128, 256):
            _drop_backend(
                backends,
                "prims-ts",
                "requires equal QK/VO head dimensions in {64, 128, 256}",
            )
        elif page_size not in (16, 32, 64, 128):
            _drop_backend(
                backends,
                "prims-ts",
                "requires page_size in {16, 32, 64, 128}",
            )
        elif num_qo_heads % num_kv_heads != 0 or not (
            1 <= num_qo_heads // num_kv_heads <= 32
        ):
            _drop_backend(
                backends,
                "prims-ts",
                "requires an integral Q/KV head ratio between 1 and 32",
            )
        elif q_dtype == torch.bfloat16 and o_data_type != torch.bfloat16:
            _drop_backend(
                backends,
                "prims-ts",
                "requires BF16 output for BF16 inputs",
            )
        elif q_dtype == torch.float16 and o_data_type != torch.float16:
            _drop_backend(
                backends,
                "prims-ts",
                "requires FP16 output for FP16 inputs",
            )
        elif q_dtype == torch.float8_e4m3fn and o_data_type not in (
            torch.float8_e4m3fn,
            torch.float16,
        ):
            _drop_backend(
                backends,
                "prims-ts",
                "supports FP16 or FP8 output for FP8 inputs",
            )
        elif args.enable_pdl:
            print("[WARNING] prims-ts does not expose PDL; ignoring --enable_pdl.")

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

    # Prepare wrappers
    backend_wrappers = {}
    resolved_backends = {}
    for backend in backends:
        if backend in ["fa2", "fa2_tc", "auto", "trtllm-gen"]:
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
                use_tensor_cores=(backend != "fa2"),
                paged_kv_indptr_buffer=plan_kv_indptr,
                paged_kv_indices_buffer=kv_indices,
                paged_kv_last_page_len_buffer=kv_last_page_len,
                backend=actual_backend,
            )
            backend_wrappers[backend].plan(
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
            )
            resolved_backends[backend] = backend_wrappers[backend]._backend
        else:
            resolved_backends[backend] = backend

    ## Prepare dtype-specific data
    k_scale, v_scale = None, None
    kv_cache_sf = None
    kv_cache_nvfp4 = None
    if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        q = q.to(q_dtype)
    if is_nvfp4_kv:
        # NVFP4 KV requires FP8 query
        if q_dtype != torch.float8_e4m3fn:
            print("[ERROR] NVFP4 KV cache requires --q_dtype fp8_e4m3.")
            return res
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
        if "trtllm-gen" in backends:
            k_data, v_data = torch.chunk(kv_cache_for_trt, 2, dim=1)
            k_fp8 = (k_data / k_scale).to(kv_dtype)
            v_fp8 = (v_data / v_scale).to(kv_dtype)
            kv_cache_for_trt = torch.cat([k_fp8, v_fp8], dim=1)

    prims_ts_kv_cache = None
    prims_ts_out = None
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
        # backend-specific compact cache.
        prims_ts_kv_cache = kv_cache.contiguous()
        prims_ts_out = torch.empty(prims_ts_q_shape, device=device, dtype=o_data_type)
        backend_wrappers["prims-ts"] = prims_ts.BatchDecodePagedTSWrapper("HND")
        backend_wrappers["prims-ts"].plan(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim_qk,
            page_size,
            seq_len_q=s_qo,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
            o_data_type=o_data_type,
            mask_type=prims_ts_mask_type,
            max_kv_len=s_kv,
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
        if backend in ["fa2", "fa2_tc", "auto", "trtllm-gen"]:
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
            )
        elif backend == "cudnn":
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
            )
        elif backend == "trtllm-native":
            native_kv = kv_cache_nvfp4 if is_nvfp4_kv else kv_cache
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
                backend="auto",
                q_len_per_req=s_qo,
                mask=speculative_mask,
                kv_cache_sf=kv_cache_sf,
                enable_pdl=args.enable_pdl,
                multi_ctas_kv_counter_buffer=gqa_multi_ctas_kv_counter_buffer,
                out=out,
                out_dtype=o_data_type,
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
                bmm1_scale=scale if k_scale is None else k_scale * scale,
                bmm2_scale=1.0 if v_scale is None else v_scale,
                out=out,
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
        runtime_k_cache = k_cache if cur_backend == "cudnn" else None
        runtime_v_cache = v_cache if cur_backend == "cudnn" else None
        runtime_workspace = None if cur_backend == "prims-ts" else workspace_buffer
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
                actual_seq_lens_kv,
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
                bmm1_scale=scale if k_scale is None else k_scale * scale,
                bmm2_scale=1.0 if v_scale is None else v_scale,
                out=prims_ts_out,
            ),
            prims_ts_out,
        )
        _validate_graph_output(graph_output, outputs["prims-ts"], rtol, atol)
        if args.verbose >= 1:
            print("[INFO] prims-ts CUDA-graph replay matched eager decode output.")
    if run_refcheck and not has_reference_output and len(tested_backends) > 1:
        reference_backend, reference_output = _select_reference_output(
            outputs,
            ["trtllm-gen", "trtllm-native", "cudnn", "auto", "prims-ts"],
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
            tflops = attention_tflops_per_sec_with_actual_seq_lens(
                actual_seq_lens_q_flat,
                actual_seq_lens_kv_flat,
                head_dim_qk,
                head_dim_vo,
                num_qo_heads,
                effective_causal,
                median_time,
            )
            tb_per_sec = attention_tb_per_sec_with_actual_seq_lens(
                actual_seq_lens_q_flat,
                actual_seq_lens_kv_flat,
                head_dim_qk,
                head_dim_vo,
                num_qo_heads,
                num_kv_heads,
                median_time,
                q_dtype=q_dtype,
                kv_dtype=kv_dtype,
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
                f"auto({resolved_backend})" if backend == "auto" else resolved_backend
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
                cur_res["out_dtype"] = o_data_type
                cur_res["avg_actual_seq_len"] = avg_seq_len_kv
                cur_res["random_actual_seq_len"] = args.random_actual_seq_len
                cur_res["case_tag"] = args.case_tag
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

    if "trtllm-gen" in backends:
        remove_trtllm = False
        if not causal:
            print("[INFO] trtllm-gen backend currently requires causal = True")
            remove_trtllm = True
        if remove_trtllm:
            backends.remove("trtllm-gen")
    if "trtllm-native" in backends:
        remove_trtllm_native = False
        if not causal:
            print("[INFO] trtllm-native backend currently requires causal = True")
            remove_trtllm_native = True
        if remove_trtllm_native:
            backends.remove("trtllm-native")
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
            _drop_backend(backends, "prims-ts", "requires matching Q and K/V dtypes")
        elif head_dim_qk != head_dim_vo or head_dim_qk not in (128, 256):
            _drop_backend(
                backends,
                "prims-ts",
                "requires equal QK/VO head dimensions in {128, 256}",
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
    )  # For cuDNN
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

    # Helper function to convert to FP8 (matches test_trtllm_gen_attention_decode.py approach)
    def to_float8(x, dtype=torch.float8_e4m3fn):
        finfo = torch.finfo(dtype)
        min_val, max_val = x.aminmax()
        amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
        scale = finfo.max / amax * 0.1
        x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
        return x_scl_sat.to(dtype), scale.float().reciprocal()

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

    # Ensure trtllm-fmha-v2 sees contiguous HND-physical paged KV cache.
    # Skip if kv_cache is not a plain Tensor (e.g., NVFP4 packed tuple).
    # backend filter further down also drops trtllm-fmha-v2 in that case.
    if "trtllm-fmha-v2" in backends and isinstance(kv_cache, torch.Tensor):
        _fmha_v2_kv_cache = kv_cache.contiguous()
    else:
        _fmha_v2_kv_cache = kv_cache

    prims_ts_k_cache = None
    prims_ts_v_cache = None
    prims_ts_out = None
    prims_ts_sm_scale = None
    prims_ts_output_scale = None
    if "prims-ts" in backends:
        prims_ts = _get_prims_ts_module()
        prims_ts_k_cache = kv_cache[:, 0].contiguous()
        prims_ts_v_cache = kv_cache[:, 1].contiguous()
        prims_ts_out = torch.empty_like(q, dtype=o_data_type)
        backend_wrappers_prims_ts = prims_ts.BatchPrefillPagedTSWrapper("HND")
        _q_scale = q_scale if q_scale is not None else 1.0
        _k_scale = k_scale if k_scale is not None else 1.0
        _v_scale = v_scale if v_scale is not None else 1.0
        prims_ts_sm_scale = _q_scale * _k_scale * scale
        prims_ts_output_scale = _v_scale
        backend_wrappers_prims_ts.plan(
            q,
            prims_ts_k_cache,
            prims_ts_v_cache,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            page_size=page_size,
            mask_type="causal" if causal else "dense",
            sm_scale=prims_ts_sm_scale,
            output_scale=prims_ts_output_scale,
            out_dtype=o_data_type,
        )

    # Prepare wrappers (after FP8 conversion so we have correct dtypes)
    backend_wrappers = {}
    resolved_backends = {}
    for backend in backends:
        if backend == "prims-ts":
            backend_wrappers[backend] = backend_wrappers_prims_ts
            resolved_backends[backend] = backend
            continue
        if backend in ["fa2", "fa3", "auto", "trtllm-gen"]:
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
            )
            resolved_backends[backend] = backend_wrappers[backend]._backend
        elif backend == "cudnn":
            # cuDNN uses NHD layout and the wrapper API
            backend_wrappers[backend] = (
                flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
                    workspace_buffer,
                    "NHD",
                    backend="cudnn",
                )
            )
            backend_wrappers["cudnn"].plan(
                q_indptr,
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
        if backend in ["fa2", "fa3", "auto", "trtllm-gen"]:
            return backend_wrappers[backend].run(
                q,
                kv_cache,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                kv_cache_sf=kv_cache_sf,
                enable_pdl=args.enable_pdl,
                out=out,
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
            )
        elif backend == "prims-ts":
            return backend_wrappers[backend].run(
                q,
                k_cache,
                v_cache,
                out=out,
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
                    out=prims_ts_out,
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
            tflops = attention_tflops_per_sec_with_actual_seq_lens(
                actual_seq_lens_q_flat,
                actual_seq_lens_kv_flat,
                head_dim_qk,
                head_dim_vo,
                num_qo_heads,
                causal,
                median_time,
            )
            tb_per_sec = attention_tb_per_sec_with_actual_seq_lens(
                actual_seq_lens_q_flat,
                actual_seq_lens_kv_flat,
                head_dim_qk,
                head_dim_vo,
                num_qo_heads,
                num_kv_heads,
                median_time,
                q_dtype=q_dtype,
                kv_dtype=kv_dtype,
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
                cur_res["out_dtype"] = o_data_type
                cur_res["avg_actual_seq_len"] = avg_seq_len_q
                cur_res["random_actual_seq_len"] = args.random_actual_seq_len
                cur_res["case_tag"] = args.case_tag
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
        elif head_dim_qk != head_dim_vo or head_dim_qk not in (128, 256):
            _drop_backend(
                backends,
                "prims-ts",
                "requires equal QK/VO head dimensions in {128, 256}",
            )
        elif num_qo_heads % num_kv_heads != 0:
            _drop_backend(backends, "prims-ts", "requires Hq to be divisible by Hkv")
        elif args.enable_pdl:
            print("[WARNING] prims-ts does not expose PDL; ignoring --enable_pdl.")

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
    )  # For cuDNN

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

    # Prepare wrappers
    backend_wrappers = {}
    for backend in backends:
        if backend in ["cutlass", "fa2", "fa3", "trtllm-gen"]:
            backend_wrappers[backend] = (
                flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
                    workspace_buffer,
                    "NHD",
                    use_cuda_graph=is_cuda_graph_compatible
                    if backend != "fa2"
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
            )
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
                qo_indptr=q_indptr,
                kv_indptr=k_indptr,
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
                v_indptr=v_indptr,
                o_indptr=o_indptr,
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
            q,
            k,
            v,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            mask_type="causal" if causal else "dense",
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
        if backend in ["cutlass", "fa2", "fa3", "trtllm-gen"]:
            return backend_wrappers[backend].run_return_lse(
                q, k, v, enable_pdl=args.enable_pdl, out=out
            )[0]
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
                window_left=-1,
                cum_seq_lens_q=qo_indptr,
                cum_seq_lens_kv=kv_indptr,
                enable_pdl=args.enable_pdl,
                is_causal=causal,
                return_lse=True,
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
                window_left=-1,
                cum_seq_lens_q=qo_indptr,
                cum_seq_lens_kv=kv_indptr,
                enable_pdl=args.enable_pdl,
                is_causal=causal,
                return_lse=True,
                out=out,
                q_seq_lens_cpu=actual_seq_lens_q_cpu_flat,
                kv_seq_lens_cpu=actual_seq_lens_kv_cpu_flat,
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
            return backend_wrappers[backend].run(q, k, v, out=out)
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
            return run_backend_wrapper(
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

        backend_times[cur_backend] = bench_gpu_time(
            fn=run_timed_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=True,
            enable_cupti=args.use_cupti,
            use_cuda_graph=(is_cuda_graph_compatible and cur_backend != "fa2"),
            cold_l2_cache=True,
            input_args=(
                q,
                k,
                v,
                runtime_out,
            ),
        )

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
                    out=prims_ts_out,
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
            tflops = attention_tflops_per_sec_with_actual_seq_lens(
                actual_seq_lens_q_flat,
                actual_seq_lens_kv_flat,
                head_dim_qk,
                head_dim_vo,
                num_qo_heads,
                causal,
                median_time,
            )
            tb_per_sec = attention_tb_per_sec_with_actual_seq_lens(
                actual_seq_lens_q_flat,
                actual_seq_lens_kv_flat,
                head_dim_qk,
                head_dim_vo,
                num_qo_heads,
                num_kv_heads,
                median_time,
                q_dtype=q_dtype,
                kv_dtype=kv_dtype,
                o_dtype=out_dtype,
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
                res.append(cur_res)
    return res


def testBatchMLAPagedAttentionWrapper(args):
    """
    Test BatchMLAPagedAttentionWrapper and equivalent APIs.
    Supports fa2, fa3, cutlass, trtllm-native, cute-dsl, and prims-ts.

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
        if page_size not in [32, 64]:
            print(
                "[INFO] Cutlass MLA backend only supports page size 32 or 64. Skipping."
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
    if "trtllm-native" in backends:
        remove_trtllm_native = False
        if page_size not in [32, 64]:
            print(
                "[INFO] trtllm-native backend only supports page size 32 or 64. Skipping."
            )
            remove_trtllm_native = True
        if remove_trtllm_native:
            backends.remove("trtllm-native")
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

    sm_scale = 1.0 / ((128 + 64) ** 0.5)  # For DeepSeek-R1
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
    for backend in backends:
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
            block_tables,
            actual_seq_lens_kv.flatten(),
            num_qo_heads,
            head_dim_ckv,
            head_dim_kpe,
            page_size,
            seq_len_q=s_qo,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
            o_data_type=torch.bfloat16,
            mask_type="causal",
            max_kv_len=s_kv,
        )

    direct_out = None
    if any(backend in backends for backend in ("trtllm-native", "auto", "cute-dsl")):
        direct_out = torch.empty(
            batch_size,
            s_qo,
            num_qo_heads,
            head_dim_ckv,
            device=device,
            dtype=torch.bfloat16,
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
                qk_nope_head_dim=128,  # To-do: Why??
                kv_lora_rank=head_dim_ckv,
                qk_rope_head_dim=head_dim_kpe,
                block_tables=block_tables,
                seq_lens=actual_seq_lens_kv.flatten(),
                max_seq_len=s_kv,
                out=out,
                bmm1_scale=sm_scale,
                bmm2_scale=1.0,
                backend="trtllm-gen",
                enable_pdl=args.enable_pdl,
                multi_ctas_kv_counter_buffer=mla_multi_ctas_kv_counter_buffer,
                **mla_api_extra_kwargs,
            ).reshape(-1, num_qo_heads, head_dim_ckv)
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
                qk_nope_head_dim=128,
                kv_lora_rank=head_dim_ckv,
                qk_rope_head_dim=head_dim_kpe,
                block_tables=block_tables,
                seq_lens=actual_seq_lens_kv.flatten(),
                max_seq_len=s_kv,
                out=out,
                bmm1_scale=sm_scale,
                bmm2_scale=1.0,
                backend="auto",
                enable_pdl=args.enable_pdl,
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
                qk_nope_head_dim=128,
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
                    bmm1_scale=sm_scale,
                    bmm2_scale=1.0,
                    out=out,
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
        direct_backend = cur_backend in ("trtllm-native", "auto", "cute-dsl")
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
            else direct_out
            if direct_backend
            else None
        )
        if run_refcheck:
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
                bmm1_scale=sm_scale,
                bmm2_scale=1.0,
                out=prims_ts_out,
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
            if reference_output.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
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
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res
