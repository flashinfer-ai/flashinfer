from collections import defaultdict

import numpy as np
import torch

import flashinfer

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
        else:
            normalized.append(backend)
    return normalized


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
            "trtllm-gen-native",  # Deprecated, will be removed in future
        ],
        help="Kernel backends to test. Default: fa2. backend=auto is only supported for BatchDecodeWithPagedKVCacheWrapper and BatchPrefillWithPagedKVCacheWrapper.",
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
        help="Max sequence length of the query. Should be 1 for decode.",
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
        help="Data type of the query. Currently only bfloat16 is supported.",
    )
    parser.add_argument(
        "--kv_dtype",
        type=str,
        required=False,
        default="bfloat16",
        help="Data type of the key and value. Currently only bfloat16 is supported.",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        default=False,
        help="Causal masking. Note: not padding masking. Only used for prefill tests.",
    )
    parser.add_argument(
        "--random_actual_seq_len",
        action="store_true",
        default=False,
        help="Use random actual sequence lengths for the query and key and value. Random values are generated between 1 and maximum sequence length. If False, use maximum sequence length.",
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


def testBatchDecodeWithPagedKVCacheWrapper(args):
    """
    Test BatchDecodeWithPagedKVCacheWrapper API and equivalent cuDNN API.
    Supports fa2, fa2_tc, auto, cudnn, trtllm-gen, trtllm-native backends.

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
    if q_dtype not in [torch.bfloat16, torch.float8_e4m3fn]:
        print(f"[ERROR] Unsupported q_dtype: {args.q_dtype}")
        return res

    # Handle different KV cache data types.
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    if kv_dtype not in [torch.bfloat16, torch.float8_e4m3fn]:
        print(f"[ERROR] Unsupported kv_dtype: {args.kv_dtype}")
        return res

    # Parse and validate backend configurations
    backends = args.backends
    page_size = args.page_size
    batch_size = args.batch_size
    s_qo = args.s_qo
    s_kv = args.s_kv
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim_qk = args.head_dim_qk
    head_dim_vo = args.head_dim_vo
    is_cuda_graph_compatible = not args.no_cuda_graph
    # return_lse = not args.no_lse # TO-DO: Add support for this
    run_refcheck = args.refcheck

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    # Check for backend-specific constraints
    if "fa2" in backends:
        remove_fa2 = False
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
        if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] or kv_dtype in [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]:
            print("[INFO] FA2_TC backend does not support FP8. Skipping.")
            remove_fa2_tc = True
        if remove_fa2_tc:
            backends.remove("fa2_tc")

    if "cudnn" in backends:
        remove_cudnn = False
        if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] or kv_dtype in [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]:
            print("[INFO] cuDNN backend does not support FP8. Skipping.")
            remove_cudnn = True
        if remove_cudnn:
            backends.remove("cudnn")

    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}

    # Sample sequence lengths and create tensors
    actual_seq_lens_kv = sample_actual_seq_lens(
        s_kv, batch_size, device, args.random_actual_seq_len
    )
    sum_seq_kv = torch.sum(actual_seq_lens_kv).item()
    avg_seq_len_kv = sum_seq_kv // batch_size

    if args.verbose >= 1:
        print(f"[VERBOSE] Average actual seq len: {avg_seq_len_kv}")
    if args.verbose >= 2:
        print(f"[VVERBOSE] {actual_seq_lens_kv.flatten() = }")

    # Create query tensor
    q = torch.rand(
        batch_size, num_qo_heads, head_dim_qk, device=device, dtype=q_init_dtype
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
        torch.arange(0, batch_size + 1, device=device) * (num_qo_heads * head_dim_qk)
    ).long()  # For cuDNN

    scale = float(1.0 / (head_dim_qk**0.5))
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

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
                block_tables=block_tables,
            )
            resolved_backends[backend] = backend_wrappers[backend]._backend
        else:
            resolved_backends[backend] = backend

    ## If FP8, prepare
    k_scale, v_scale = None, None
    if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        q = q.to(q_dtype)
    if kv_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
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
    ):
        if backend in ["fa2", "fa2_tc", "auto", "trtllm-gen"]:
            return backend_wrappers[backend].run(
                q, kv_cache, k_scale=k_scale, v_scale=v_scale
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
            )
        elif backend == "trtllm-native":
            return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                query=q.contiguous(),
                kv_cache=kv_cache,
                workspace_buffer=workspace_buffer,
                block_tables=block_tables,
                seq_lens=actual_seq_lens_kv,
                max_seq_len=s_kv,
                bmm1_scale=scale if k_scale is None else k_scale * scale,
                bmm2_scale=1.0 if v_scale is None else v_scale,
            )
        else:
            print(f"[ERROR] Backend {backend} not supported")
            return None

    has_reference_output = False
    # Iterate over each backend:
    for cur_backend in backends:
        # Clear workspace buffer to prevent unexpected interactions between backends.
        workspace_buffer.zero_()
        if run_refcheck:
            outputs[cur_backend] = (
                run_backend_wrapper(
                    cur_backend,
                    q,
                    kv_cache,
                    k_cache,
                    v_cache,
                    workspace_buffer,
                    block_tables,
                    actual_seq_lens_kv,
                    ragged_q,
                )
                .detach()
                .clone()
            )
            if cur_backend == "fa2":
                has_reference_output = True
                reference_output = outputs[cur_backend]
        # Unified benchmark entry: prefer graph if compatible and not using CUPTI
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend_wrapper,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=False,
            enable_cupti=args.use_cupti,
            use_cuda_graph=(is_cuda_graph_compatible and cur_backend != "fa2"),
            cold_l2_cache=True,
            input_args=(
                cur_backend,
                q,
                kv_cache,
                k_cache,
                v_cache,
                workspace_buffer,
                block_tables,
                actual_seq_lens_kv,
                ragged_q,
            ),
        )

    # Perform reference check
    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
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
                        f"[ERROR] Output tensor mismatch between backends fa2 and {tested_backends[i]}: "
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
            actual_seq_lens_q_flat = torch.ones_like(actual_seq_lens_kv_flat)
            tflops = attention_tflops_per_sec_with_actual_seq_lens(
                actual_seq_lens_q_flat,
                actual_seq_lens_kv_flat,
                head_dim_qk,
                head_dim_vo,
                num_qo_heads,
                False,
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
                o_dtype=q_dtype,
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
                cur_res["causal"] = False
                cur_res["q_dtype"] = q_dtype
                cur_res["kv_dtype"] = kv_dtype
                cur_res["avg_actual_seq_len"] = avg_seq_len_kv
                cur_res["random_actual_seq_len"] = args.random_actual_seq_len
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testBatchPrefillWithPagedKVCacheWrapper(args):
    """
    Test BatchPrefillWithPagedKVCacheWrapper API and equivalent cuDNN API.
    Supports fa2, fa3, auto, trtllm-gen, trtllm-native, and cudnn backends.

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
    if q_dtype not in [torch.bfloat16, torch.float8_e4m3fn]:
        print(f"[ERROR] Unsupported q_dtype: {args.q_dtype}")
        return res

    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    if kv_dtype not in [torch.bfloat16, torch.float8_e4m3fn]:
        print(f"[ERROR] Unsupported kv_dtype: {args.kv_dtype}")
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
    head_dim_vo = args.head_dim_vo
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

    if "cutlass" in backends:
        print("[INFO] CUTLASS backend does not support prefill. Skipping.")
        remove_cutlass = True
        if remove_cutlass:
            backends.remove("cutlass")

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

    # Because actual_seq_lens_kv is the same as actual_seq_lens_q, kv_indptr will become the same as qo_indptr
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
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

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

    # Helper function to convert to FP8 (matches test_trtllm_gen_attention.py approach)
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
    o_data_type = q_dtype  # Default output dtype
    # Separate K/V caches for cuDNN (which requires separate tensors, not combined kv_cache)
    k_cache_cudnn, v_cache_cudnn = k_cache, v_cache

    if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        q, q_scale_t = to_float8(q, q_dtype)
        q_scale = q_scale_t.item()
        q_scale_tensor = q_scale_t.reshape(1, 1, 1, 1)
        # o_data_type stays as q_dtype (FP8 output)

    if kv_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
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

    # Prepare wrappers (after FP8 conversion so we have correct dtypes)
    backend_wrappers = {}
    resolved_backends = {}
    for backend in backends:
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
    ):
        if backend in ["fa2", "fa3", "auto", "trtllm-gen"]:
            return backend_wrappers[backend].run(
                q, kv_cache, q_scale=q_scale, k_scale=k_scale, v_scale=v_scale
            )
        elif backend == "cudnn":
            # cuDNN uses wrapper API with tensor scales for FP8
            return backend_wrappers[backend].run(
                q,
                (k_cache_cudnn, v_cache_cudnn),
                q_scale=q_scale_tensor,
                k_scale=k_scale_tensor,
                v_scale=v_scale_tensor,
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
                cum_seq_lens_kv=kv_indptr,
            )
        elif backend == "cudnn-native":
            # Direct cudnn_batch_prefill_with_kv_cache call (similar to trtllm-native)
            return flashinfer.prefill.cudnn_batch_prefill_with_kv_cache(
                q,
                k_cache_cudnn,
                v_cache_cudnn,
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
            )[0]
        else:
            print(f"[ERROR] Backend {backend} not supported")
            return None

    has_reference_output = False
    reference_backend = None
    # Iterate over each backend:
    for cur_backend in backends:
        # Clear workspace buffer to prevent unexpected interactions between backends.
        workspace_buffer.zero_()
        if run_refcheck:
            outputs[cur_backend] = (
                run_backend_wrapper(
                    cur_backend,
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
                )
                .detach()
                .clone()
            )
            if cur_backend == "fa2":
                has_reference_output = True
                reference_output = outputs[cur_backend]
                reference_backend = "fa2"
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend_wrapper,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=False,
            enable_cupti=args.use_cupti,
            use_cuda_graph=(is_cuda_graph_compatible and cur_backend != "fa2"),
            cold_l2_cache=True,
            input_args=(
                cur_backend,
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
            ),
        )

    # Perform reference check
    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())

    # When cases where FA2 is not available, try to find an alternative reference
    # Priority: cudnn > cudnn-native > trtllm-gen > trtllm-native
    if run_refcheck and not has_reference_output and len(tested_backends) > 1:
        reference_priority = ["cudnn", "cudnn-native", "trtllm-gen", "trtllm-native"]
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
                o_dtype=q_dtype,
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
                cur_res["avg_actual_seq_len"] = avg_seq_len_q
                cur_res["random_actual_seq_len"] = args.random_actual_seq_len
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testBatchPrefillWithRaggedKVCacheWrapper(args):
    """
    Test BatchPrefillWithRaggedKVCacheWrapper API and equivalent cuDNN API.
    Supports fa2, fa3, cutlass, and cudnn backends.

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
    if q_dtype not in [torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]:
        print(f"[ERROR] Unsupported q_dtype: {args.q_dtype}")
        return res
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    if kv_dtype not in [torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]:
        print(f"[ERROR] Unsupported kv_dtype: {args.kv_dtype}")
        return res

    # Parse and validate backend configurations
    backends = args.backends
    batch_size = args.batch_size
    s_qo = args.s_qo
    s_kv = args.s_kv
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim_qk = args.head_dim_qk
    head_dim_vo = args.head_dim_vo
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
        if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] or kv_dtype in [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]:
            print("[INFO] trtllm-native backend does not support FP8. Skipping.")
            remove_trtllm_native = True
        if not (head_dim_qk == 192 and head_dim_vo == 128):
            print(
                "[INFO] trtllm-native backend requires head_dim_qk == 192 and head_dim_vo == 128"
            )
            remove_trtllm_native = True
        if remove_trtllm_native:
            backends.remove("trtllm-native")

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
        cumsum_s_qo, num_qo_heads, head_dim_qk, device=device, dtype=q_init_dtype
    )
    if args.verbose >= 2:
        print(f"[VVERBOSE] {q.shape = }")

    k = torch.randn(
        cumsum_s_kv, num_kv_heads, head_dim_qk, device=device, dtype=kv_init_dtype
    )
    v = torch.randn(
        cumsum_s_kv, num_kv_heads, head_dim_vo, device=device, dtype=kv_init_dtype
    )

    block_tables = None

    ## The following are for BatchPrefillWithRaggedKVCacheWrapper
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
    ).cuda()

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
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

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
                o_data_type=q_dtype,
                seq_lens=actual_seq_lens_kv_device,
                seq_lens_q=actual_seq_lens_q_device,
                max_token_per_sequence=s_qo,
                max_sequence_kv=s_kv,
                v_indptr=v_indptr,
                o_indptr=o_indptr,
            )

    k_scale, v_scale = None, None
    if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        q = q.to(q_dtype)
    if kv_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        k_scale = k.amax().item() / 256
        v_scale = v.amax().item() / 256
        k = (k / k_scale).to(kv_dtype)
        v = (v / v_scale).to(kv_dtype)

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
    ):
        if backend in ["cutlass", "fa2", "fa3", "trtllm-gen"]:
            return backend_wrappers[backend].run_return_lse(q, k, v)[0]
        elif backend == "cudnn":
            # cuDNN uses wrapper API
            return backend_wrappers[backend].run(q, k, v)
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
            )[0]
        elif backend == "trtllm-native":
            return flashinfer.prefill.trtllm_ragged_attention_deepseek(
                query=q,
                key=k,
                value=v,
                workspace_buffer=workspace_buffer,
                seq_lens=actual_seq_lens_kv_device,
                max_q_len=s_qo,
                max_kv_len=s_kv,
                bmm1_scale=scale,
                bmm2_scale=1.0,
                o_sf_scale=-1,
                batch_size=batch_size,
                window_left=-1,
                cum_seq_lens_q=qo_indptr,
                cum_seq_lens_kv=kv_indptr,
                enable_pdl=False,
                is_causal=causal,
                return_lse=True,
            )[0]
        else:
            print(f"[ERROR] Backend {backend} not supported")
            return None

    has_reference_output = False
    # Iterate over each backend:
    for cur_backend in backends:
        # Clear workspace buffer to prevent unexpected interactions between backends.
        workspace_buffer.zero_()
        if run_refcheck:
            outputs[cur_backend] = (
                run_backend_wrapper(
                    cur_backend,
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
                )
                .detach()
                .clone()
            )
            if cur_backend == "fa2":
                has_reference_output = True
                reference_output = outputs[cur_backend]
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend_wrapper,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=True,
            enable_cupti=args.use_cupti,
            use_cuda_graph=(is_cuda_graph_compatible and cur_backend != "fa2"),
            cold_l2_cache=True,
            input_args=(
                cur_backend,
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
            ),
        )

    # Perform reference check
    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
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
                        f"[ERROR] Output tensor mismatch between backends fa2 and {tested_backends[i]}: "
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
                cur_res["avg_actual_seq_len"] = avg_seq_len_q
                cur_res["random_actual_seq_len"] = args.random_actual_seq_len
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testBatchMLAPagedAttentionWrapper(args):
    """
    Test BatchMLAPagedAttentionWrapper and equivalent APIs.
    Supports fa2, fa3, cutlass, and trtllm-native.

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
    causal = False  # False for MLA
    run_refcheck = args.refcheck

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
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}

    actual_seq_lens_kv = sample_actual_seq_lens(
        s_kv, batch_size, device, args.random_actual_seq_len
    )
    sum_seq_kv = torch.sum(actual_seq_lens_kv).item()
    avg_seq_len_kv = sum_seq_kv // batch_size

    if args.verbose >= 1:
        print(f"[VERBOSE] Average actual seq len: {avg_seq_len_kv}")
    if args.verbose >= 2:
        print(f"[VVERBOSE] {actual_seq_lens_kv.flatten() = }")

    q_nope = torch.rand(
        batch_size, num_qo_heads, head_dim_ckv, dtype=q_init_dtype, device="cuda"
    )
    q_pe = torch.zeros(
        batch_size, num_qo_heads, head_dim_kpe, dtype=q_init_dtype, device="cuda"
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

    qo_indptr = torch.arange(0, batch_size + 1, device=device).int()
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
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

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
    backend_wrappers = {}
    for backend in backends:
        if backend in ["fa2", "fa3", "cutlass"]:
            backend_wrappers[backend] = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                float_workspace_buffer=workspace_buffer,
                use_cuda_graph=is_cuda_graph_compatible,
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                kv_len_arr=actual_seq_lens_kv,
                backend=backend,
            )
            if backend != "cutlass":
                backend_wrappers[backend].plan(
                    qo_indptr=qo_indptr,
                    kv_indptr=kv_indptr,
                    kv_indices=kv_indices,
                    kv_len_arr=actual_seq_lens_kv,
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
    ):
        if backend in ["fa2", "fa3"]:
            return backend_wrappers[backend].run(
                q_nope,
                q_pe,
                ckv_cache,
                kpe_cache,
                page_table=block_tables,
                return_lse=False,
            )
        elif backend == "cutlass":
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
                query=q.unsqueeze(1),
                kv_cache=kv_cache.unsqueeze(1),
                workspace_buffer=workspace_buffer,
                qk_nope_head_dim=128,  # To-do: Why??
                kv_lora_rank=head_dim_ckv,
                qk_rope_head_dim=head_dim_kpe,
                block_tables=block_tables,
                seq_lens=actual_seq_lens_kv,
                max_seq_len=s_kv,
                bmm1_scale=sm_scale,
                bmm2_scale=1.0,
            ).squeeze(1)
        else:
            print(f"[ERROR] Unsupported backend: {backend}")
            return None

    has_reference_output = False
    # Iterate over each backend:
    for cur_backend in backends:
        # Clear workspace buffer to prevent unexpected interactions between backends.
        workspace_buffer.zero_()
        if run_refcheck:
            outputs[cur_backend] = (
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
                )
                .detach()
                .clone()
            )
            if cur_backend == "fa2":
                has_reference_output = True
                reference_output = outputs[cur_backend]
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend_wrapper,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=False,
            enable_cupti=args.use_cupti,
            use_cuda_graph=(is_cuda_graph_compatible and cur_backend != "fa2"),
            cold_l2_cache=True,
            input_args=(
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
            ),
        )

    # Perform reference check
    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
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
                        f"[ERROR] Output tensor mismatch between backends fa2 and {tested_backends[i]}: "
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
            actual_seq_lens_q_flat = torch.ones_like(
                actual_seq_lens_kv.flatten().to("cpu")
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
            o_elem_size = q_nope.element_size()  # Output has same dtype as query
            o_mem_bytes = batch_size * num_qo_heads * head_dim_ckv * o_elem_size

            total_mem_bytes = q_mem_bytes + kv_mem_bytes + o_mem_bytes
            tb_per_sec = total_mem_bytes / (median_time * 1e9)
            tflops_total = (
                2
                * torch.dot(
                    actual_seq_lens_q_flat.to(torch.float32),
                    actual_seq_lens_kv_flat.to(torch.float32),
                )
                * num_qo_heads
                * (2 * head_dim_ckv + head_dim_kpe)
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
                cur_res["causal"] = False
                cur_res["q_dtype"] = q_dtype
                cur_res["kv_dtype"] = kv_dtype
                cur_res["avg_actual_seq_len"] = avg_seq_len_kv
                cur_res["random_actual_seq_len"] = args.random_actual_seq_len
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res
