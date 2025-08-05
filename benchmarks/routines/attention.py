from collections import defaultdict

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import (
    attention_tb_per_sec_with_actual_seq_lens,
    attention_tflops_per_sec_with_actual_seq_lens,
    bench_gpu_time,
    bench_gpu_time_with_cudagraph,
)

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    get_device,
    print_perf_metrics,
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
        raise ValueError(f"Unsupported routine: {args.routine}")


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
        choices=["fa2", "fa2_tc", "fa3", "cudnn", "cutlass", "trtllm-gen"],
        help="Kernel backends to test. Default: fa2",
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
    Supports fa2, fa2_tc, cudnn, and trtllm-gen backends.

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

    q_init_dtype = torch.bfloat16
    kv_init_dtype = torch.bfloat16
    rtol = 2e-1
    atol = 1e-2

    # Handle different query data types.
    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    if q_dtype not in [torch.bfloat16, torch.float8_e4m3fn]:
        raise ValueError(f"Unsupported q_dtype: {args.q_dtype}")

    # Handle different KV cache data types.
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    if kv_dtype not in [torch.bfloat16, torch.float8_e4m3fn]:
        raise ValueError(f"Unsupported kv_dtype: {args.kv_dtype}")

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

    # Derived parameters
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
        return

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
            [k + i * num_pages_per_seq for k in range(num_pages_per_seq)]
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
        kv_indices[start_idx:end_idx] = torch.arange(
            i * num_pages_per_seq,
            i * num_pages_per_seq + (end_idx - start_idx),
            device=device,
        )

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
    for backend in backends:
        if backend in ["fa2", "fa2_tc", "trtllm-gen"]:
            plan_kv_indptr = (
                kv_indptr.clone().detach() if backend == "trtllm-gen" else kv_indptr
            )
            backend_wrappers[backend] = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer,
                "HND",
                use_cuda_graph=is_cuda_graph_compatible,
                use_tensor_cores=(backend != "fa2"),
                paged_kv_indptr_buffer=plan_kv_indptr,
                paged_kv_indices_buffer=kv_indices,
                paged_kv_last_page_len_buffer=kv_last_page_len,
                backend=backend,
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
            )

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

    def run_backend_wrapper(backend):
        if backend in ["fa2", "fa2_tc", "trtllm-gen"]:
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
        else:
            raise ValueError(f"Backend {backend} not supported")

    has_reference_output = False
    if run_refcheck and "fa2" in backends:
        reference_output = (
            backend_wrappers["fa2"]
            .run(q, kv_cache, k_scale=k_scale, v_scale=v_scale)
            .detach()
        )
        has_reference_output = True

    # Iterate over each backend:
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend_wrapper(cur_backend).detach()
        if is_cuda_graph_compatible:
            backend_times[cur_backend] = bench_gpu_time_with_cudagraph(
                fn=lambda: run_backend_wrapper(cur_backend),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                num_iters_within_graph=20,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
                sleep_after_run=False,
            )
        else:
            backend_times[cur_backend] = bench_gpu_time(
                fn=lambda: run_backend_wrapper(cur_backend),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
                sleep_after_run=False,
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
                try:
                    torch.testing.assert_close(
                        reference_output, tested_outputs[i], rtol=rtol, atol=atol
                    )
                except AssertionError as e:
                    print(
                        f"[ERROR] Output tensor mismatch between backends {tested_backends[0]} and {tested_backends[i]}"
                    )
                    if not args.allow_output_mismatch:
                        print(e)
                        raise
    # Compute perf metrics
    res = []
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
            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

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
                cur_res["num_kv_heads"] = num_kv_heads
                cur_res["head_dim_qk"] = head_dim_qk
                cur_res["head_dim_vo"] = head_dim_vo
                cur_res["causal"] = False
                cur_res["q_dtype"] = q_dtype
                cur_res["kv_dtype"] = kv_dtype
                cur_res["avg_actual_seq_len"] = avg_seq_len_kv
                cur_res["random_actual_seq_len"] = args.random_actual_seq_len
                res.append(cur_res)
    return res


def testBatchPrefillWithPagedKVCacheWrapper(args):
    """
    Test BatchPrefillWithPagedKVCacheWrapper API and equivalent cuDNN API.
    Supports fa2, fa3, trtllm-gen, and cudnn backends.

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

    q_init_dtype = torch.bfloat16
    kv_init_dtype = torch.bfloat16
    rtol = 2e-1
    atol = 1e-2

    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    if q_dtype not in [torch.bfloat16, torch.float8_e4m3fn]:
        raise ValueError(f"Unsupported q_dtype: {args.q_dtype}")

    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    if kv_dtype not in [torch.bfloat16, torch.float8_e4m3fn]:
        raise ValueError(f"Unsupported kv_dtype: {args.kv_dtype}")

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

    # Check for backend-specific constraints
    if "fa2" in backends:
        remove_fa2 = False
        if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            print("[INFO] FA2 backend does not support FP8. Skipping.")
            remove_fa2 = True
        if remove_fa2:
            backends.remove("fa2")
    if "fa3" in backends:
        remove_fa3 = False
        device_capability = torch.cuda.get_device_capability()
        if device_capability[0] != 9:
            print(
                f"[INFO] FA3 backend does not support capability {device_capability}. Skipping."
            )
            remove_fa3 = True
        if remove_fa3:
            backends.remove("fa3")
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

    if "trtllm-gen" in backends:
        remove_trtllm = False
        if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] or kv_dtype in [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]:
            print("[INFO] trtllm-gen backend does not support FP8. Skipping.")
            remove_trtllm = True
        if remove_trtllm:
            backends.remove("trtllm-gen")

    if "cutlass" in backends:
        print("[INFO] CUTLASS backend does not support prefill. Skipping.")
        remove_cutlass = True
        if remove_cutlass:
            backends.remove("cutlass")

    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return

    # Check for layer-specific constraints
    layer_not_supported = False
    if not ((head_dim_qk == 128 and head_dim_qk == head_dim_vo) or head_dim_qk == 192):
        print("[ERROR] Head dimension must be 128 or 192")
        layer_not_supported = True
    if layer_not_supported:
        print("[ERROR] Layer not supported. Exiting.")
        return

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}

    # Randomly sample actual_seq_lens_q. Assume actual_seq_lens_kv is the same as actual_seq_lens_q.
    actual_seq_lens_q = sample_actual_seq_lens(
        s_qo, batch_size, None, args.random_actual_seq_len
    )
    actual_seq_lens_kv = actual_seq_lens_q.clone()

    avg_seq_len_q = actual_seq_lens_q.sum().item() // batch_size
    if args.verbose >= 1:
        print(f"[VERBOSE] Average actual seq len: {avg_seq_len_q}")
    if args.verbose >= 2:
        print(f"[VVERBOSE] {actual_seq_lens_q.flatten() = }")

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
            [k + i * num_pages_per_seq for k in range(num_pages_per_seq)]
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
    qo_indptr_cudnn = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q_device.view(-1), dim=0)
            * head_dim_qk
            * num_qo_heads,
        ]
    ).int()

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
        kv_indices[start_idx:end_idx] = torch.arange(
            i * num_pages_per_seq,
            i * num_pages_per_seq + (end_idx - start_idx),
            device=device,
        )
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

    # Prepare wrappers
    backend_wrappers = {}
    for backend in backends:
        if backend in ["fa2", "fa3", "trtllm-gen"]:
            backend_wrappers[backend] = (
                flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
                    workspace_buffer,
                    "HND",
                    use_cuda_graph=is_cuda_graph_compatible,
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
            )

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

    def run_backend_wrapper(backend):
        if backend in ["fa2", "fa3", "trtllm-gen"]:
            return backend_wrappers[backend].run(
                q, kv_cache, k_scale=k_scale, v_scale=v_scale
            )
        elif backend == "cudnn":
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
            )[0]
        else:
            raise ValueError(f"Backend {backend} not supported")

    has_reference_output = False
    if run_refcheck and "fa2" in backends:
        reference_output = backend_wrappers["fa2"].run(
            q, kv_cache, k_scale=k_scale, v_scale=v_scale
        )
        has_reference_output = True

    # Iterate over each backend:
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend_wrapper(cur_backend)
        if is_cuda_graph_compatible:
            backend_times[cur_backend] = bench_gpu_time_with_cudagraph(
                fn=lambda: run_backend_wrapper(cur_backend),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                num_iters_within_graph=20,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
                sleep_after_run=False,
            )
        else:
            backend_times[cur_backend] = bench_gpu_time(
                fn=lambda: run_backend_wrapper(cur_backend),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
                sleep_after_run=False,
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
                try:
                    torch.testing.assert_close(
                        reference_output, tested_outputs[i], rtol=rtol, atol=atol
                    )
                except AssertionError as e:
                    print(
                        f"[ERROR] Output tensor mismatch between backends {tested_backends[0]} and {tested_backends[i]}"
                    )
                    if not args.allow_output_mismatch:
                        print(e)
                        raise

    # Compute perf metrics
    res = []
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

    q_init_dtype = torch.bfloat16
    kv_init_dtype = torch.bfloat16
    rtol = 2e-1
    atol = 1e-2

    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    if q_dtype not in [torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]:
        raise ValueError(f"Unsupported q_dtype: {args.q_dtype}")
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    if kv_dtype not in [torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]:
        raise ValueError(f"Unsupported kv_dtype: {args.kv_dtype}")

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

    # Check for backend-specific constraints
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

    if "cutlass" in backends:
        remove_cutlass = False
        if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] or kv_dtype in [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]:
            print("[INFO] CUTLASS backend does not support FP8. Skipping.")
            remove_cutlass = True
        if remove_cutlass:
            backends.remove("cutlass")

    if "trtllm-gen" in backends:
        print("[INFO] trtllm-gen backend does not support ragged prefill. Skipping.")
        remove_trtllm = True
        if remove_trtllm:
            backends.remove("trtllm-gen")

    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return

    # Check for layer-specific constraints
    layer_not_supported = False
    if not ((head_dim_qk == 128 and head_dim_qk == head_dim_vo) or head_dim_qk == 192):
        print("[ERROR] Head dimension must be 128 or 192")
        layer_not_supported = True
    if layer_not_supported:
        print("[ERROR] Layer not supported. Exiting.")
        return

    backend_times = {backend: [] for backend in backends}
    outputs = {}

    # Randomly sample actual_seq_lens_q. Assume actual_seq_lens_kv is the same as actual_seq_lens_q.
    actual_seq_lens_q = sample_actual_seq_lens(
        s_qo, batch_size, None, args.random_actual_seq_len
    )
    actual_seq_lens_kv = actual_seq_lens_q.clone()

    avg_seq_len_q = actual_seq_lens_q.sum().item() // batch_size
    if args.verbose >= 1:
        print(f"[VERBOSE] Average actual seq len: {avg_seq_len_q}")
    if args.verbose >= 2:
        print(f"[VVERBOSE] {actual_seq_lens_q.flatten() = }")

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
                    use_cuda_graph=is_cuda_graph_compatible,
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

    k_scale, v_scale = None, None
    if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        q = q.to(q_dtype)
    if kv_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        k_scale = k.amax().item() / 256
        v_scale = v.amax().item() / 256
        k = (k / k_scale).to(kv_dtype)
        v = (v / v_scale).to(kv_dtype)

    def run_backend_wrapper(backend):
        if backend in ["cutlass", "fa2", "fa3", "trtllm-gen"]:
            return backend_wrappers[backend].run_return_lse(q, k, v)[0]
        elif backend == "cudnn":
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
        else:
            raise ValueError(f"Backend {backend} not supported")

    has_reference_output = False
    if run_refcheck and "fa2" in backends:
        reference_output = backend_wrappers["fa2"].run_return_lse(q, k, v)[0]
        has_reference_output = True

    # Iterate over each backend:
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend_wrapper(cur_backend)
        if is_cuda_graph_compatible:
            backend_times[cur_backend] = bench_gpu_time_with_cudagraph(
                fn=lambda: run_backend_wrapper(cur_backend),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                num_iters_within_graph=20,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
                sleep_after_run=True,
            )
        else:
            backend_times[cur_backend] = bench_gpu_time(
                fn=lambda: run_backend_wrapper(cur_backend),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
                sleep_after_run=True,
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
                try:
                    torch.testing.assert_close(
                        reference_output, tested_outputs[i], rtol=rtol, atol=atol
                    )
                except AssertionError as e:
                    print(
                        f"[ERROR] Output tensor mismatch between backends {tested_backends[0]} and {tested_backends[i]}"
                    )
                    if not args.allow_output_mismatch:
                        print(e)
                        raise

    # Compute perf metrics
    res = []
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
                res.append(cur_res)
    return res


def testBatchMLAPagedAttentionWrapper(args):
    """
    Test BatchMLAPagedAttentionWrapper and equivalent APIs.
    Supports fa2. trtllm-gen is not supported yet.

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

    q_init_dtype = torch.bfloat16
    kv_init_dtype = torch.bfloat16
    rtol = 2e-1
    atol = 1e-2

    # Handle different query data types.
    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    if q_dtype not in [torch.bfloat16, torch.float8_e4m3fn]:
        raise ValueError(f"Unsupported q_dtype: {args.q_dtype}")

    # Handle different KV cache data types.
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    if kv_dtype not in [torch.bfloat16, torch.float8_e4m3fn]:
        raise ValueError(f"Unsupported kv_dtype: {args.kv_dtype}")

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
    if args.verbose >= 2:
        print(f"[VVERBOSE] {q_nope.shape = }")
        print(f"[VVERBOSE] {q_pe.shape = }")

    # Create KV cache
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    if args.verbose >= 2:
        print(f"[VVERBOSE] {num_pages_per_seq = }")
        print(f"[VVERBOSE] {total_num_pages = }")

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
    kpe_cache = torch.randn(size=kpe_cache_shape, dtype=q_init_dtype, device=device)

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
        kv_indices[start_idx:end_idx] = torch.arange(
            i * num_pages_per_seq,
            i * num_pages_per_seq + (end_idx - start_idx),
            device=device,
        )

    sm_scale = 1.0 / ((head_dim_ckv + head_dim_kpe) ** 0.5)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {ckv_cache.shape = }")
        print(f"[VVERBOSE] {kpe_cache.shape = }")
        print(f"[VVERBOSE] {qo_indptr.shape = }")
        print(f"[VVERBOSE] {kv_indptr.shape = }")
        print(f"[VVERBOSE] {kv_indices.shape = }")
        print(f"[VVERBOSE] {actual_seq_lens_kv.shape = }")
        print(f"[VVERBOSE] {sm_scale = }")
        print(f"[VVERBOSE] {workspace_buffer.shape = }")

    # Create wrapper
    if "fa2" in backends:
        fi_fa2_mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            float_workspace_buffer=workspace_buffer,
            use_cuda_graph=is_cuda_graph_compatible,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_len_arr=actual_seq_lens_kv,
            backend="fa2",
        )
        fi_fa2_mla_wrapper.plan(
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
    if "trtllm-gen" in backends:
        ## Input preparation for flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla
        head_dim_ckv = head_dim_ckv
        head_dim_kpe = head_dim_kpe
        page_size = page_size

    if q_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        q_pe = q_pe.to(q_dtype)
        q_nope = q_nope.to(q_dtype)

    def run_backend_wrapper(backend):
        if backend == "fa2":
            return fi_fa2_mla_wrapper.run(
                q_nope, q_pe, ckv_cache, kpe_cache, return_lse=False
            )
        # if backend == "trtllm-gen":
        #     return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        #         query = query,
        #         kv_cache = kv_cache,
        #         workspace_buffer = workspace_buffer,
        #         qk_nope_head_dim = head_dim_ckv,
        #         kv_lora_rank = head_dim_ckv,
        #         qk_rope_head_dim = head_dim_kpe,
        #         block_tables = kv_indices,
        #         seq_lens = actual_seq_lens_kv,
        #         max_seq_len = s_kv,
        #         # out
        #         # bmm1_scale
        #         # bmm2_scale
        #         # bmm1_scale_log2_tensor
        #         # bmm2_scale_tensor
        #     )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    if run_refcheck and "fa2" in backends:
        reference_output = fi_fa2_mla_wrapper.run(
            q_nope, q_pe, ckv_cache, kpe_cache, return_lse=False
        )
        has_reference_output = True
    else:
        has_reference_output = False

    # Iterate over each backend:
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend_wrapper(cur_backend).detach()
        if is_cuda_graph_compatible:
            backend_times[cur_backend] = bench_gpu_time_with_cudagraph(
                fn=lambda: run_backend_wrapper(cur_backend),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                num_iters_within_graph=20,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
                sleep_after_run=False,
            )
        else:
            backend_times[cur_backend] = bench_gpu_time(
                fn=lambda: run_backend_wrapper(cur_backend),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
                sleep_after_run=False,
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
                try:
                    torch.testing.assert_close(
                        reference_output, tested_outputs[i], rtol=rtol, atol=atol
                    )
                except AssertionError as e:
                    print(
                        f"[ERROR] Output tensor mismatch between backends {tested_backends[0]} and {tested_backends[i]}"
                    )
                    if not args.allow_output_mismatch:
                        print(e)
                        raise

    # Compute perf metrics
    res = []
    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            actual_seq_lens_kv_flat = actual_seq_lens_kv.flatten().to("cpu")
            actual_seq_lens_q_flat = torch.ones_like(
                actual_seq_lens_kv.flatten().to("cpu")
            )
            o_mem_bytes = (
                actual_seq_lens_q_flat.numel()
                * num_qo_heads
                * head_dim_ckv
                * q_dtype.itemsize
            )
            qkv_mem_bytes = sum(
                [
                    _.numel() * _.element_size()
                    for _ in [q_nope, q_pe, ckv_cache, kpe_cache]
                ]
            )
            total_mem_bytes = o_mem_bytes + qkv_mem_bytes
            tb_per_sec = (total_mem_bytes / (median_time * 1e9)).item()
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
                res.append(cur_res)
    return res
