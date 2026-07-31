"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
from collections import defaultdict

import numpy as np
import torch

import flashinfer
from flashinfer.gdn_decode import (
    gated_delta_rule_decode,
    gated_delta_rule_decode_pretranspose,
    gated_delta_rule_mtp,
)
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.testing.utils import bench_gpu_time

# Add tests/gdn to sys.path so the torch GDN reference is importable (same
# pattern as routines/mamba.py with tests/mamba), and benchmarks/ for the
# shared Triton GDN kernels.
_repo_root = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
_tests_gdn = os.path.join(_repo_root, "tests", "gdn")
if _tests_gdn not in sys.path:
    sys.path.insert(0, _tests_gdn)
_benchmarks_dir = os.path.join(_repo_root, "benchmarks")
if _benchmarks_dir not in sys.path:
    sys.path.insert(0, _benchmarks_dir)

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    get_device,
    is_close_stats,
    print_perf_metrics,
    filter_backends_by_compute_capability,
    warn_if_pdl_unsupported,
)

from reference_delta_rule import blockwise_delta_rule, decode_delta_rule
from gdn_triton_reference import (
    TRITON_AVAILABLE,
    triton_gdn_decode,
    triton_gdn_decode_pretranspose,
    triton_gdn_mtp,
)

try:
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd as fla_gdn

    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False


# ==============================================================================
# Benchmark infrastructure
# ==============================================================================


def run_gdn_test(args):
    """
    Run a GDN (Gated Delta Net) test.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.routine == "gated_delta_rule_decode":
        return testGatedDeltaRuleDecode(args)
    elif args.routine == "gated_delta_rule_mtp":
        return testGatedDeltaRuleMtp(args)
    elif args.routine == "chunk_gated_delta_rule":
        return testChunkGatedDeltaRule(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")


def parse_gdn_args(line, parser):
    """
    Parse command line arguments for GDN test configuration.

    Args:
        line: Command line arguments
        parser: ArgumentParser object already populated with shared arguments

    Returns:
        Parsed argument namespace
    """
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size. Decode/MTP: number of concurrent requests. "
        "Prefill: number of sequences.",
    )
    parser.add_argument(
        "--num_q_heads",
        type=int,
        default=16,
        help="Number of query heads.",
    )
    parser.add_argument(
        "--num_k_heads",
        type=int,
        default=16,
        help="Number of key heads.",
    )
    parser.add_argument(
        "--num_v_heads",
        type=int,
        default=32,
        help="Number of value heads (GVA when > num_q_heads).",
    )
    parser.add_argument(
        "--head_size",
        type=int,
        default=128,
        help="Head dimension (K = V = head_size for GDN).",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        required=False,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Data type for q/k/v/a/b input tensors.",
    )
    parser.add_argument(
        "--state_dtype",
        type=str,
        required=False,
        default="float32",
        choices=["float32", "bfloat16"],
        help="Data type for the recurrent state. bfloat16 selects the BF16 "
        "state kernels (decode/MTP, requires head_size=128 and pretranspose "
        "layout).",
    )
    parser.add_argument(
        "--state_layout",
        type=str,
        required=False,
        default="pretranspose",
        choices=["pretranspose", "nontranspose"],
        help="Decode state layout: pretranspose [B, HV, V, K] "
        "(gated_delta_rule_decode_pretranspose) or nontranspose [B, HV, K, V] "
        "(gated_delta_rule_decode). Decode routine only.",
    )
    parser.add_argument(
        "--pool_mode",
        type=str,
        required=False,
        default="single",
        choices=["single", "split"],
        help="State pool indexing mode. 'single': state pool of size B with "
        "read == write slots. 'split': pool of size 2B; reads from slots "
        "[0..B), writes to slots [B..2B) (speculative-decoding shape). "
        "'split' requires the pretranspose layout (decode) or bfloat16 state "
        "(MTP).",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        required=False,
        default=None,
        help="Tokens per request for gated_delta_rule_mtp (must be >= 2, "
        "default 2). Not applicable to other GDN routines.",
    )
    parser.add_argument(
        "--s_qo",
        type=int,
        required=False,
        default=2048,
        help="Per-sequence length for chunk_gated_delta_rule (uniform across "
        "the batch).",
    )
    parser.add_argument(
        "--update_state",
        action="store_true",
        default=False,
        help="MTP: write the final state back (disable_state_update=False). "
        "The BF16 state path always updates state in-place.",
    )
    parser.add_argument(
        "--cache_intermediate_states",
        action="store_true",
        default=False,
        help="MTP (float32 state only): cache per-token intermediate states.",
    )
    parser.add_argument(
        "--no_qk_l2norm",
        action="store_true",
        default=False,
        help="Decode/MTP: disable in-kernel Q/K L2 normalization.",
    )
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=["flashinfer"],
        choices=["flashinfer", "triton", "fla"],
        help="Kernel backends to benchmark. 'triton' is available for "
        "decode/MTP; 'fla' (flash-linear-attention) for prefill.",
    )

    args = parser.parse_args(line)

    is_decode = args.routine == "gated_delta_rule_decode"
    is_mtp = args.routine == "gated_delta_rule_mtp"
    is_prefill = args.routine == "chunk_gated_delta_rule"

    # Resolve seq_len per routine
    if is_mtp:
        if args.seq_len is None:
            args.seq_len = 2
        if args.seq_len < 2:
            raise ValueError(
                f"gated_delta_rule_mtp requires --seq_len >= 2, got {args.seq_len}"
            )
    elif args.seq_len is not None and args.seq_len != 1:
        raise ValueError(
            f"--seq_len is only applicable to gated_delta_rule_mtp, got "
            f"--seq_len {args.seq_len} for {args.routine}"
        )

    if is_decode or is_mtp:
        if args.state_dtype == "bfloat16":
            if args.head_size != 128:
                raise ValueError(
                    "bfloat16 state requires head_size=128 (BF16 state kernels "
                    f"support K=V=128 only), got head_size={args.head_size}"
                )
            if args.state_layout != "pretranspose":
                raise ValueError("bfloat16 state requires --state_layout pretranspose")
            if "triton" in args.backends:
                raise ValueError(
                    "The triton backend only supports float32 state; use "
                    "--state_dtype float32 or drop the triton backend"
                )
        if args.state_layout == "nontranspose":
            if args.pool_mode != "single":
                raise ValueError("--pool_mode split requires the pretranspose layout")
            if is_mtp:
                raise ValueError(
                    "gated_delta_rule_mtp uses the K-last [pool, HV, V, K] "
                    "state layout; --state_layout nontranspose is not "
                    "applicable"
                )
        if args.pool_mode == "split":
            if "triton" in args.backends:
                raise ValueError(
                    "The triton backend does not support --pool_mode split"
                )
            if is_mtp and args.state_dtype != "bfloat16":
                raise ValueError(
                    "gated_delta_rule_mtp only supports --pool_mode split "
                    "with --state_dtype bfloat16 (the float32 MTP API has no "
                    "output_state_indices)"
                )
        if is_mtp and args.state_dtype == "bfloat16":
            if args.cache_intermediate_states:
                raise ValueError(
                    "--cache_intermediate_states requires --state_dtype "
                    "float32 (the public BF16 MTP path does not expose "
                    "intermediate state caching)"
                )
        if is_decode and (args.update_state or args.cache_intermediate_states):
            raise ValueError(
                "--update_state / --cache_intermediate_states are only "
                "applicable to gated_delta_rule_mtp"
            )
        if "fla" in args.backends:
            raise ValueError(
                "The fla backend is only available for chunk_gated_delta_rule"
            )

    if is_prefill:
        if "triton" in args.backends:
            raise ValueError(
                "The triton backend is only available for decode/MTP routines; "
                "prefill supports flashinfer and fla"
            )

    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


# ==============================================================================
# FLOPs / bytes models
# ==============================================================================


def gdn_decode_flops(
    batch_size: int,
    num_q_heads: int,
    num_v_heads: int,
    head_size: int,
    seq_len: int = 1,
) -> int:
    """FLOPs for gated delta rule decode/MTP.

    Per token per output head, three K x V matrix-vector products
    (k @ state, outer-product update, q @ state): 6 * head_size^2 FLOPs.
    """
    num_o_heads = max(num_q_heads, num_v_heads)
    return 6 * seq_len * batch_size * num_o_heads * head_size * head_size


def gdn_decode_bytes(
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    input_dtype: torch.dtype,
    seq_len: int = 1,
    state_writeback: bool = True,
    state_dtype_bytes: int = 4,
    cache_intermediate_states: bool = False,
) -> int:
    """Memory bytes for gated delta rule decode/MTP.

    Counts q/k/v/output traffic, the state read (+ optional write-back),
    GDN parameters, and the optional MTP intermediate state writes.
    """
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads
    elem_size = input_dtype.itemsize

    q_bytes = batch_size * seq_len * num_q_heads * head_size * elem_size
    k_bytes = batch_size * seq_len * num_k_heads * head_size * elem_size
    v_bytes = batch_size * seq_len * num_v_heads * head_size * elem_size
    o_bytes = batch_size * seq_len * num_o_heads * head_size * elem_size

    state_elems = batch_size * num_sab_heads * head_size * head_size
    state_bytes = state_elems * state_dtype_bytes * (2 if state_writeback else 1)

    # Parameters: A_log [HV] fp32, dt_bias [HV] fp32, a/b [B, T, HV]
    param_bytes = (
        num_sab_heads * 4
        + num_sab_heads * 4
        + 2 * batch_size * seq_len * num_sab_heads * elem_size
    )

    intermediate_bytes = 0
    if cache_intermediate_states and seq_len > 1:
        intermediate_bytes = (
            batch_size
            * seq_len
            * num_sab_heads
            * head_size
            * head_size
            * state_dtype_bytes
        )

    return (
        q_bytes
        + k_bytes
        + v_bytes
        + o_bytes
        + state_bytes
        + param_bytes
        + intermediate_bytes
    )


def gdn_prefill_flops(total_tokens: int, num_sab_heads: int, head_size: int) -> int:
    """FLOPs for chunked GDN prefill.

    Counts the two dominant GEMMs per token per head (kv outer-product
    accumulation and q @ state), matching bench_gdn_prefill.py. Intra-chunk
    attention terms are excluded for consistency with that convention.
    """
    return 2 * 2 * total_tokens * num_sab_heads * head_size * head_size


def gdn_prefill_bytes(
    total_tokens: int,
    num_seqs: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    input_dtype: torch.dtype,
) -> int:
    """Memory bytes for chunked GDN prefill (q/k/v/g/beta reads, output and
    final-state writes)."""
    num_sab_heads = max(num_q_heads, num_v_heads)
    elem_size = input_dtype.itemsize

    qkv_bytes = (
        total_tokens * (num_q_heads + num_k_heads + num_v_heads) * head_size * elem_size
    )
    gate_bytes = 2 * total_tokens * num_sab_heads * 4  # g, beta fp32
    o_bytes = total_tokens * num_sab_heads * head_size * elem_size
    state_bytes = num_seqs * num_sab_heads * head_size * head_size * 4  # fp32
    return qkv_bytes + gate_bytes + o_bytes + state_bytes


# ==============================================================================
# Decode / MTP
# ==============================================================================


def testGatedDeltaRuleDecode(args):
    """Test gated_delta_rule_decode (T=1, pretranspose or nontranspose layout)."""
    return _testGdnDecodeLike(args, seq_len=1)


def testGatedDeltaRuleMtp(args):
    """Test gated_delta_rule_mtp (T>1 multi-token processing)."""
    return _testGdnDecodeLike(args, seq_len=args.seq_len)


def _testGdnDecodeLike(args, seq_len):
    """
    Shared implementation for GDN decode (T=1) and MTP (T>1).

    This test:
    1. Generates random q/k/v, GDN gating parameters, and a recurrent state
       pool in the requested layout and dtype
    2. Runs the requested backend(s):
       - 'flashinfer': CuTe-DSL kernels via the public gdn_decode APIs
       - 'triton': Triton reference kernels (tests/gdn/gdn_triton_reference.py)
    3. Optionally checks outputs against the torch reference
       (tests/gdn/reference_delta_rule.py::decode_delta_rule)
    4. Measures performance (memory bandwidth is the primary metric)

    Args:
        args: Parsed command line arguments containing test configuration
        seq_len: Tokens per request (1 = decode, >1 = MTP)

    Returns:
        dict: List of dictionaries containing performance results
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print(f"[INFO] Running {args.routine}")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    num_q_heads = args.num_q_heads
    num_k_heads = args.num_k_heads
    num_v_heads = args.num_v_heads
    head_size = args.head_size
    state_layout = args.state_layout if seq_len == 1 else "pretranspose"
    pool_split = args.pool_mode == "split"
    use_qk_l2norm = not args.no_qk_l2norm
    is_mtp = seq_len > 1
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if "triton" in backends and not TRITON_AVAILABLE:
        print("[WARNING] triton is not installed. Skipping triton backend.")
        backends.remove("triton")
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    state_dtype = dtype_str_to_torch_dtype(args.state_dtype)
    use_bf16_state = state_dtype == torch.bfloat16
    ## Done parsing input arguments

    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads
    T = seq_len

    ## Prepare input tensors
    q = torch.randn(
        batch_size, T, num_q_heads, head_size, dtype=input_dtype, device=device
    )
    k = torch.randn(
        batch_size, T, num_k_heads, head_size, dtype=input_dtype, device=device
    )
    v = torch.randn(
        batch_size, T, num_v_heads, head_size, dtype=input_dtype, device=device
    )
    A_log = torch.randn(num_sab_heads, dtype=torch.float32, device=device)
    a = torch.randn(batch_size, T, num_sab_heads, dtype=input_dtype, device=device)
    dt_bias = torch.randn(num_sab_heads, dtype=torch.float32, device=device)
    b = torch.randn(batch_size, T, num_sab_heads, dtype=input_dtype, device=device)
    scale = head_size**-0.5

    # State pool. Layout interpretation:
    #   pretranspose / MTP: [pool, HV, V, K] (K-last)
    #   nontranspose:       [B, HV, K, V] (V-last)
    # K = V = head_size, so the allocation shape is the same either way.
    pool_size = 2 * batch_size if pool_split else batch_size
    state_pool = torch.randn(
        pool_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=state_dtype,
        device=device,
    )
    initial_state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    if pool_split:
        output_state_indices = torch.arange(
            batch_size, 2 * batch_size, dtype=torch.int32, device=device
        )
    else:
        output_state_indices = None

    # Intermediate states buffer (fp32 MTP only)
    intermediate_states_buffer = None
    if is_mtp and args.cache_intermediate_states:
        intermediate_states_buffer = torch.zeros(
            batch_size,
            T,
            num_sab_heads,
            head_size,
            head_size,
            dtype=torch.float32,
            device=device,
        )

    output = torch.empty(
        batch_size, T, num_o_heads, head_size, dtype=input_dtype, device=device
    )

    # The BF16 state path always updates state in-place; the fp32 MTP API
    # makes the write-back optional.
    state_writeback = True if (not is_mtp or use_bf16_state) else args.update_state
    disable_state_update = not state_writeback

    if args.verbose >= 2:
        print(f"[VVERBOSE] Mode: {'MTP' if is_mtp else 'decode'} (T={T})")
        print(f"[VVERBOSE] {q.shape = }, {q.dtype = }")
        print(f"[VVERBOSE] {v.shape = }, {v.dtype = }")
        print(f"[VVERBOSE] {state_pool.shape = }, {state_pool.dtype = }")
        print(f"[VVERBOSE] {state_layout = }, {args.pool_mode = }")
        print(f"[VVERBOSE] {use_qk_l2norm = }, {state_writeback = }")

    def run_backend(backend, state):
        if backend == "flashinfer":
            if is_mtp:
                if use_bf16_state:
                    # BF16 MTP path is reached through the pretranspose API
                    return gated_delta_rule_decode_pretranspose(
                        q,
                        k,
                        v,
                        None,
                        A_log,
                        a,
                        dt_bias,
                        b,
                        scale,
                        output,
                        use_qk_l2norm,
                        initial_state=state,
                        initial_state_indices=initial_state_indices,
                        output_state_indices=output_state_indices,
                    )[0]
                return gated_delta_rule_mtp(
                    q,
                    k,
                    v,
                    state,
                    initial_state_indices,
                    A_log,
                    a,
                    dt_bias,
                    b,
                    scale,
                    output,
                    intermediate_states_buffer=intermediate_states_buffer,
                    disable_state_update=disable_state_update,
                    use_qk_l2norm=use_qk_l2norm,
                )[0]
            if state_layout == "nontranspose":
                return gated_delta_rule_decode(
                    q, k, v, state, A_log, a, dt_bias, b, scale, output, use_qk_l2norm
                )[0]
            # BF16 state always goes through the pool path with pre-allocated
            # indices: the non-pool API path synthesizes an arange index
            # tensor per call, whose tiny kernel would pollute CUPTI timing.
            if pool_split or use_bf16_state:
                return gated_delta_rule_decode_pretranspose(
                    q,
                    k,
                    v,
                    None,
                    A_log,
                    a,
                    dt_bias,
                    b,
                    scale,
                    output,
                    use_qk_l2norm,
                    initial_state=state,
                    initial_state_indices=initial_state_indices,
                    output_state_indices=output_state_indices,
                )[0]
            return gated_delta_rule_decode_pretranspose(
                q, k, v, state, A_log, a, dt_bias, b, scale, output, use_qk_l2norm
            )[0]
        elif backend == "triton":
            if is_mtp:
                return triton_gdn_mtp(
                    q,
                    k,
                    v,
                    state,
                    initial_state_indices,
                    A_log,
                    a,
                    dt_bias,
                    b,
                    scale,
                    output,
                    intermediate_states_buffer=intermediate_states_buffer,
                    disable_state_update=disable_state_update,
                    use_qk_l2norm=use_qk_l2norm,
                )[0]
            if state_layout == "nontranspose":
                return triton_gdn_decode(
                    q, k, v, state, A_log, a, dt_bias, b, scale, output, use_qk_l2norm
                )[0]
            return triton_gdn_decode_pretranspose(
                q, k, v, state, A_log, a, dt_bias, b, scale, output, use_qk_l2norm
            )[0]
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Reference check against the torch reference, which uses the K-major
    # state layout [B, HV, K, V] and processes one token at a time.
    # bench_gpu_time mutates state_pool in-place, so all refcheck runs use
    # clones of the clean snapshot taken before any benchmarking.
    has_reference_output = False
    clean_state_snapshot = state_pool.clone() if run_refcheck else None
    outputs = {}
    if run_refcheck:
        ref_state = clean_state_snapshot[:batch_size]
        if state_layout == "pretranspose":
            # [B, HV, V, K] -> [B, HV, K, V]
            ref_state = ref_state.transpose(-1, -2)
        ref_state = ref_state.contiguous().float()
        ref_outs = []
        for t in range(T):
            ref_o, ref_state = decode_delta_rule(
                q[:, t].float(),
                k[:, t].float(),
                v[:, t].float(),
                ref_state,
                A_log=A_log,
                a=a[:, t].float(),
                dt_bias=dt_bias,
                b=b[:, t].float(),
                scale_factor=scale,
                use_l2_norm=use_qk_l2norm,
                state_dtype=state_dtype,
            )
            ref_outs.append(ref_o)
        reference_output = torch.stack(ref_outs, dim=1)  # [B, T, HV, V]
        has_reference_output = True

        for cur_backend in backends:
            fresh_state = clean_state_snapshot.clone()
            outputs[cur_backend] = (
                run_backend(cur_backend, fresh_state).detach().clone()
            )

    # Storage for timing results
    backend_times = {backend: [] for backend in backends}
    for cur_backend in backends:
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, state_pool),
        )

    # Compare outputs against the torch reference
    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 0 and run_refcheck and has_reference_output:
        # bf16 state rounds the recurrent state every step in the reference
        # but is kept in higher precision inside the kernels, so use looser
        # tolerances there.
        rtol, atol = (2e-2, 2e-2) if use_bf16_state else (5e-3, 5e-3)
        for i in range(len(tested_backends)):
            (
                num_different_elements,
                num_elements,
                num_different_elements_percentage,
            ) = is_close_stats(
                reference_output.float(),
                tested_outputs[i].float(),
                rtol=rtol,
                atol=atol,
            )
            mismatch_threshold_pct = 0.01
            if num_different_elements_percentage > mismatch_threshold_pct:
                print(
                    f"[ERROR] Output tensor mismatch from backend {tested_backends[i]}: "
                    f"{num_different_elements}/{num_elements} ({num_different_elements_percentage:.4f}%) elements differ "
                    f"(threshold: {mismatch_threshold_pct}%)"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(
                        f"[ERROR] Backend {tested_backends[i]} output mismatch with {num_different_elements} elements"
                    )
            elif args.verbose >= 1:
                print(
                    f"[REFCHECK] Backend {tested_backends[i]}: PASSED "
                    f"({num_different_elements}/{num_elements} elements differ "
                    f"({num_different_elements_percentage:.4f}%), within {mismatch_threshold_pct}% threshold)"
                )

    # Compute and report performance metrics
    problem_flops = gdn_decode_flops(batch_size, num_q_heads, num_v_heads, head_size, T)
    problem_bytes = gdn_decode_bytes(
        batch_size,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        input_dtype,
        T,
        state_writeback=state_writeback,
        state_dtype_bytes=state_dtype.itemsize,
        cache_intermediate_states=args.cache_intermediate_states,
    )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            tflops = problem_flops / (10**9 * median_time)
            tb_per_sec = problem_bytes / (10**9 * median_time)

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["backend"] = backend
                cur_res["batch_size"] = batch_size
                cur_res["num_q_heads"] = num_q_heads
                cur_res["num_k_heads"] = num_k_heads
                cur_res["num_v_heads"] = num_v_heads
                cur_res["head_size"] = head_size
                cur_res["seq_len"] = T
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["state_dtype"] = str(state_dtype)
                cur_res["state_layout"] = state_layout
                cur_res["pool_mode"] = args.pool_mode
                cur_res["update_state"] = state_writeback
                cur_res["use_qk_l2norm"] = use_qk_l2norm
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


# ==============================================================================
# Prefill
# ==============================================================================


def testChunkGatedDeltaRule(args):
    """
    Test chunk_gated_delta_rule (varlen GDN prefill).

    This test:
    1. Generates varlen q/k/v (uniform per-sequence length --s_qo), forget
       gate alpha and update gate beta. k is pre-L2-normalized and the kernel
       is called with use_qk_l2norm_in_kernel=False so that the torch
       reference (which performs no normalization) sees identical inputs.
    2. Runs the requested backend(s):
       - 'flashinfer': SM90 C++ / SM100 CuTe-DSL chunked GDN prefill
       - 'fla': flash-linear-attention Triton baseline (perf-only, excluded
         from refcheck; requires pip install flash-linear-attention)
    3. Optionally checks the flashinfer output against
       tests/gdn/reference_delta_rule.py::blockwise_delta_rule
    4. Measures performance metrics

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print("[INFO] Running testChunkGatedDeltaRule")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    num_seqs = args.batch_size
    s_qo = args.s_qo
    num_q_heads = args.num_q_heads
    num_k_heads = args.num_k_heads
    num_v_heads = args.num_v_heads
    head_size = args.head_size
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if "fla" in backends and not FLA_AVAILABLE:
        print(
            "[WARNING] fla is not installed (pip install flash-linear-attention). "
            "Skipping fla backend."
        )
        backends.remove("fla")
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    ## Done parsing input arguments

    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads
    seq_lens = [s_qo] * num_seqs
    total_tokens = num_seqs * s_qo
    cu_seqlens = torch.arange(
        0, total_tokens + 1, s_qo, dtype=torch.int64, device=device
    )

    ## Prepare input tensors
    q = torch.randn(
        total_tokens, num_q_heads, head_size, dtype=input_dtype, device=device
    )
    # Pre-normalize k for numerical stability (matches bench_gdn_prefill.py);
    # the kernel and the reference then see identical inputs.
    k = torch.nn.functional.normalize(
        torch.randn(
            total_tokens, num_k_heads, head_size, dtype=torch.float32, device=device
        ),
        p=2.0,
        dim=-1,
    ).to(input_dtype)
    v = torch.randn(
        total_tokens, num_v_heads, head_size, dtype=input_dtype, device=device
    )
    # Forget gate alpha in (0, 1) and update gate beta in (0, 1), fp32
    g = torch.rand(total_tokens, num_sab_heads, dtype=torch.float32, device=device)
    beta = torch.rand(total_tokens, num_sab_heads, dtype=torch.float32, device=device)
    scale = head_size**-0.5

    output = torch.empty(
        total_tokens, num_o_heads, head_size, dtype=input_dtype, device=device
    )
    output_state = torch.empty(
        num_seqs,
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device=device,
    )

    # FLA baseline tensors: batch-dim layout [1, T, H, D], log-space forget
    # gate, q expanded to HV heads for GVA, int32 cu_seqlens.
    if "fla" in backends:
        q_fla = (
            q.repeat_interleave(num_sab_heads // num_q_heads, dim=1)
            if num_sab_heads > num_q_heads
            else q
        ).unsqueeze(0)
        k_fla = (
            k.repeat_interleave(num_sab_heads // num_k_heads, dim=1)
            if num_sab_heads > num_k_heads
            else k
        ).unsqueeze(0)
        v_fla = v.unsqueeze(0)
        g_fla = torch.log(g.clamp_min(1e-6)).unsqueeze(0)
        beta_fla = beta.unsqueeze(0)
        h0_fla = torch.zeros(
            num_seqs,
            num_sab_heads,
            head_size,
            head_size,
            dtype=torch.float32,
            device=device,
        )
        cu_seqlens_fla = cu_seqlens.to(torch.int32)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {q.shape = }, {q.dtype = }")
        print(f"[VVERBOSE] {v.shape = }, {v.dtype = }")
        print(f"[VVERBOSE] {cu_seqlens = }")
        print(f"[VVERBOSE] {g.shape = }, {beta.shape = }")

    def run_backend(backend):
        if backend == "flashinfer":
            return chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                scale,
                None,  # initial_state (zero state)
                True,  # output_final_state
                cu_seqlens,
                False,  # use_qk_l2norm_in_kernel (k is pre-normalized)
                output=output,
                output_state=output_state,
            )[0]
        elif backend == "fla":
            return fla_gdn(
                q_fla,
                k_fla,
                v_fla,
                g_fla,
                beta_fla,
                None,
                initial_state=h0_fla,
                output_final_state=True,
                cu_seqlens=cu_seqlens_fla,
            )[0]
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Reference check (flashinfer only; fla uses a different gate
    # parameterization and is benchmarked perf-only)
    has_reference_output = False
    outputs = {}
    if run_refcheck:
        reference_output = blockwise_delta_rule(
            q.float(),
            k.float(),
            v.float(),
            seq_lens,
            alpha=g,
            beta=beta,
            scale_factor=scale,
            state_dtype=torch.float32,
        )[0]
        has_reference_output = True
        for cur_backend in backends:
            if cur_backend == "fla":
                continue
            outputs[cur_backend] = run_backend(cur_backend).detach().clone()

    # Storage for timing results
    backend_times = {backend: [] for backend in backends}
    for cur_backend in backends:
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend,),
        )

    # Compare outputs against the torch reference
    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 0 and run_refcheck and has_reference_output:
        rtol, atol = 1e-2, 1e-2
        for i in range(len(tested_backends)):
            (
                num_different_elements,
                num_elements,
                num_different_elements_percentage,
            ) = is_close_stats(
                reference_output.float(),
                tested_outputs[i].float(),
                rtol=rtol,
                atol=atol,
            )
            mismatch_threshold_pct = 0.01
            if num_different_elements_percentage > mismatch_threshold_pct:
                print(
                    f"[ERROR] Output tensor mismatch from backend {tested_backends[i]}: "
                    f"{num_different_elements}/{num_elements} ({num_different_elements_percentage:.4f}%) elements differ "
                    f"(threshold: {mismatch_threshold_pct}%)"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(
                        f"[ERROR] Backend {tested_backends[i]} output mismatch with {num_different_elements} elements"
                    )
            elif args.verbose >= 1:
                print(
                    f"[REFCHECK] Backend {tested_backends[i]}: PASSED "
                    f"({num_different_elements}/{num_elements} elements differ "
                    f"({num_different_elements_percentage:.4f}%), within {mismatch_threshold_pct}% threshold)"
                )

    # Compute and report performance metrics
    problem_flops = gdn_prefill_flops(total_tokens, num_sab_heads, head_size)
    problem_bytes = gdn_prefill_bytes(
        total_tokens,
        num_seqs,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        input_dtype,
    )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            tflops = problem_flops / (10**9 * median_time)
            tb_per_sec = problem_bytes / (10**9 * median_time)

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["backend"] = backend
                cur_res["batch_size"] = num_seqs
                cur_res["s_qo"] = s_qo
                cur_res["num_q_heads"] = num_q_heads
                cur_res["num_k_heads"] = num_k_heads
                cur_res["num_v_heads"] = num_v_heads
                cur_res["head_size"] = head_size
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res
