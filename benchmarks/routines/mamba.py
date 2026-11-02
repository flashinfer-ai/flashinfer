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
from flashinfer.testing.utils import bench_gpu_time

# Add tests/mamba to sys.path so triton_reference is importable as a package
_repo_root = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
_tests_mamba = os.path.join(_repo_root, "tests", "mamba")
if _tests_mamba not in sys.path:
    sys.path.insert(0, _tests_mamba)

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    get_device,
    is_close_stats,
    print_perf_metrics,
    filter_backends_by_compute_capability,
    warn_if_pdl_unsupported,
)

from triton_reference.selective_state_update import (
    selective_state_update_triton as selective_state_update_triton_reference,
)
from triton_reference.selective_state_update_varlen import (
    selective_state_update_varlen_triton,
)


# ==============================================================================
# Routine / backend tables
# ==============================================================================

# Harness --backends names per routine.  What each name means at the library
# level (kept in sync with evidence/.../mamba/backend_map.md):
#   selective_state_update
#     flashinfer  : selective_state_update(backend="auto") -> CUDA-JIT SSU
#                   (sm80 / sm90 / sm100 module picked by compute-capability major)
#     triton      : tests/mamba/triton_reference (vLLM/SGLang production kernel)
#   ssd_combined (Mamba2 chunked prefill)
#     cute-dsl    : SSDCombined(backend="cute")  (tcgen05 CuTe-DSL kernel)
#     triton      : tests/mamba/triton_reference/ssd_combined (5 Triton kernels)
#   checkpointing_ssu (ReplaySSM MTP decode)
#     flashinfer             : algorithm="auto" (scratch trio provided)
#     flashinfer-monolith    : algorithm="monolith"
#     flashinfer-two-kernel  : algorithm="two-kernel" (precompute + main)
#     triton                 : triton_reference replay_selective_state_update
#                              (TMA persistent, mode=persistent_dynamic)
#   replayssm_materialize
#     flashinfer  : replayssm_materialize (CUDA JIT); refcheck is a torch recurrence
#
# The Cake backends (selective_state_update(backend="cake"),
# SSDCombined(backend="cake") / ssd_combined_fwd) are out of scope for this
# harness revision and not offered.
ROUTINE_BACKENDS = {
    "selective_state_update": ["flashinfer", "triton"],
    "ssd_combined": ["cute-dsl", "triton"],
    "checkpointing_ssu": [
        "flashinfer",
        "flashinfer-monolith",
        "flashinfer-two-kernel",
        "triton",
    ],
    "replayssm_materialize": ["flashinfer"],
}
_SSU, _SSD = "selective_state_update", "ssd_combined"
_CKPT, _REPLAY = "checkpointing_ssu", "replayssm_materialize"
#: Routine-specific flags -> the routines that consume them.
_ROUTINE_FLAGS = {
    "cache_steps": (_SSU, _CKPT),
    "weight_dtype": (_SSU, _CKPT),
    "has_z": (_SSU, _CKPT, _SSD),
    "dt_softplus": (_SSU, _CKPT, _SSD),
    "varlen": (_SSU, _SSD),
    "algorithm": (_SSU,),
    "intermediate_states": (_SSU,),
    "disable_state_update": (_SSU,),
    "dst_checkpoint_step": (_SSU,),
    "philox_rounds": (_SSU, _CKPT, _REPLAY),
    "seqlen": (_SSD,),
    "chunk_size": (_SSD,),
    "initial_states": (_SSD,),
    "no_d": (_SSD,),
    "d_has_hdim": (_SSD,),
    "max_window": (_CKPT, _REPLAY),
    "prev_tokens": (_CKPT,),
    "d_split": (_CKPT,),
    "num_layers": (_REPLAY,),
    "replay_prefix_len": (_REPLAY,),
    "active_requests": (_REPLAY,),
}

DEFAULT_BACKENDS = {
    "selective_state_update": ["flashinfer"],
    "ssd_combined": ["cute-dsl"],
    "checkpointing_ssu": ["flashinfer"],
    "replayssm_materialize": ["flashinfer"],
}
ALL_BACKENDS = sorted({b for v in ROUTINE_BACKENDS.values() for b in v})

_STATE_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "int16": torch.int16,
    "int8": torch.int8,
    "fp8_e4m3": torch.float8_e4m3fn,
}
_ROUTINE_STATE_DTYPES = {
    "selective_state_update": ["bfloat16", "float16", "float32", "int16"],
    "ssd_combined": ["bfloat16", "float16"],
    "checkpointing_ssu": ["bfloat16", "float16", "float32", "int8", "fp8_e4m3"],
    "replayssm_materialize": ["bfloat16", "float16", "float32", "int8", "fp8_e4m3"],
}


# ==============================================================================
# Benchmark infrastructure
# ==============================================================================


def run_mamba_test(args):
    """
    Run a mamba test.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.routine == "selective_state_update":
        return testSelectiveStateUpdate(args)
    elif args.routine == "ssd_combined":
        return testSSDCombined(args)
    elif args.routine == "checkpointing_ssu":
        return testCheckpointingSSU(args)
    elif args.routine == "replayssm_materialize":
        return testReplaySSMMaterialize(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")


def parse_mamba_args(line, parser):
    """
    Parse command line arguments for mamba test configuration.

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
        help="Batch size (number of sequences / decode requests).",
    )
    parser.add_argument(
        "--nheads",
        type=int,
        required=True,
        help="Number of SSM heads.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        required=True,
        help="Head dimension (headdim).",
    )
    parser.add_argument(
        "--dstate",
        type=int,
        required=True,
        help="SSM state size.",
    )
    parser.add_argument(
        "--ngroups",
        type=int,
        required=False,
        default=8,
        help="Number of groups for B and C matrices. nheads must be divisible by ngroups.",
    )
    parser.add_argument(
        "--cache_steps",
        type=int,
        required=False,
        default=0,
        help="selective_state_update: tokens per request for multi-token prediction "
        "(0 = single-token). checkpointing_ssu: predicted tokens T per step (>= 1).",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        required=False,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Data type for input tensors (x, B, C, z). float16 is selective_state_update only.",
    )
    parser.add_argument(
        "--state_dtype",
        type=str,
        required=False,
        default="bfloat16",
        choices=list(_STATE_DTYPES.keys()),
        help="Data type for the SSM state cache. int16 (selective_state_update) and "
        "int8/fp8_e4m3 (checkpointing_ssu, replayssm_materialize) are block-scaled "
        "and allocate a float32 state_scale.",
    )
    parser.add_argument(
        "--weight_dtype",
        type=str,
        required=False,
        default="float32",
        choices=["bfloat16", "float32"],
        help="Data type for weight tensors (dt, D, dt_bias).",
    )
    parser.add_argument(
        "--has_z",
        action="store_true",
        default=False,
        help="Include z tensor for gating (z * sigmoid(z) applied to output).",
    )
    parser.add_argument(
        "--dt_softplus",
        action="store_true",
        default=False,
        help="Apply softplus to dt before use.",
    )
    parser.add_argument(
        "--varlen",
        action="store_true",
        default=False,
        help="selective_state_update: 3D flattened inputs with cu_seqlens, 2D state "
        "indices and dst_state_batch_indices (requires --cache_steps >= 1 = max_seqlen). "
        "ssd_combined: pack batch_size sequences of --seqlen tokens into one row "
        "with seq_idx / chunk_indices / chunk_offsets.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=False,
        default="auto",
        choices=["auto", "simple", "vertical", "horizontal", "async_horizontal"],
        help="selective_state_update kernel algorithm. async_horizontal is a "
        "backward-compat alias the CUDA kernel maps to 'simple'.",
    )
    # --- selective_state_update speculative-decode / quantization knobs ---
    parser.add_argument(
        "--intermediate_states",
        action="store_true",
        default=False,
        help="selective_state_update MTP: pass intermediate_states_buffer "
        "(batch, T, nheads, dim, dstate) + intermediate_state_indices (spec-decode "
        "verify). int16 state also passes intermediate_state_scales.",
    )
    parser.add_argument(
        "--disable_state_update",
        action="store_true",
        default=False,
        help="selective_state_update: do not write the state back (spec-decode verify).",
    )
    parser.add_argument(
        "--dst_checkpoint_step",
        type=int,
        required=False,
        default=-1,
        help="selective_state_update MTP: pass a 2D dst_state_batch_indices (batch, T) "
        "that writes the state only after token k (all other columns = pad). -1 = off.",
    )
    parser.add_argument(
        "--philox_rounds",
        type=int,
        required=False,
        default=0,
        help="Stochastic rounding of the state store: pass rand_seed and this many "
        "Philox-4x32 rounds (0 = off, round-to-nearest).",
    )
    # --- ssd_combined knobs ---
    parser.add_argument(
        "--seqlen",
        type=int,
        required=False,
        default=None,
        help="ssd_combined: tokens per sequence (multiple of --chunk_size).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        required=False,
        default=128,
        help="ssd_combined chunk size.",
    )
    parser.add_argument(
        "--initial_states",
        action="store_true",
        default=False,
        help="ssd_combined: pass random initial_states (always on in --varlen).",
    )
    parser.add_argument(
        "--no_d",
        action="store_true",
        default=False,
        help="ssd_combined: no D skip connection.",
    )
    parser.add_argument(
        "--d_has_hdim",
        action="store_true",
        default=False,
        help="ssd_combined: D has shape (nheads, headdim) instead of (nheads,).",
    )
    # --- checkpointing_ssu / replayssm_materialize knobs ---
    parser.add_argument(
        "--max_window",
        type=int,
        required=False,
        default=None,
        help="checkpointing_ssu / replayssm_materialize: replay window (ring rows "
        "= max_window + T), in [1, 16]. Default: 2*T (ckpt) / 16 (replay).",
    )
    parser.add_argument(
        "--prev_tokens",
        type=int,
        required=False,
        default=None,
        help="checkpointing_ssu: accepted tokens to replay per slot (prev_num_accepted_tokens). "
        "prev_tokens + T > max_window selects the checkpoint-write path. Default: max_window.",
    )
    parser.add_argument(
        "--d_split",
        type=int,
        required=False,
        default=None,
        help="checkpointing_ssu: force the per-head DIM split (1 or 2). Default: library choice.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        required=False,
        default=1,
        help="replayssm_materialize: layers in the pointer tables.",
    )
    parser.add_argument(
        "--replay_prefix_len",
        type=int,
        required=False,
        default=None,
        help="replayssm_materialize: ring entries to apply (0 = exact copy). Default: max_window.",
    )
    parser.add_argument(
        "--active_requests",
        type=int,
        required=False,
        default=None,
        help="replayssm_materialize: requests in the active map. Default: batch_size.",
    )
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=None,
        choices=ALL_BACKENDS,
        help="Kernel backends to benchmark. Valid names depend on --routine: "
        + "; ".join(f"{r}: {', '.join(b)}" for r, b in ROUTINE_BACKENDS.items()),
    )

    args = parser.parse_args(line)
    routine = args.routine

    if args.backends is None:
        args.backends = DEFAULT_BACKENDS[routine][:]
    bad = [b for b in args.backends if b not in ROUTINE_BACKENDS[routine]]
    if bad:
        raise ValueError(
            f"Backends {bad} are not valid for {routine}; valid: {ROUTINE_BACKENDS[routine]}"
        )
    if args.state_dtype not in _ROUTINE_STATE_DTYPES[routine]:
        raise ValueError(
            f"--state_dtype {args.state_dtype} is not supported by {routine}; "
            f"valid: {_ROUTINE_STATE_DTYPES[routine]}"
        )
    if routine != "selective_state_update" and args.input_dtype != "bfloat16":
        raise ValueError(f"{routine} only supports --input_dtype bfloat16")
    # A flag the routine does not consume would be silently ignored (and still
    # echoed into the CSV); reject it.
    for dest, routines in _ROUTINE_FLAGS.items():
        if routine not in routines and getattr(args, dest) != parser.get_default(dest):
            raise ValueError(f"--{dest} does not apply to {routine}")

    # Validate nheads divisibility
    if args.nheads % args.ngroups != 0:
        raise ValueError(
            f"nheads ({args.nheads}) must be divisible by ngroups ({args.ngroups})."
        )

    if routine == "selective_state_update":
        _validate_ssu_args(args)
    elif routine == "ssd_combined":
        if args.seqlen is None:
            raise ValueError("ssd_combined requires --seqlen")
        if args.seqlen % args.chunk_size != 0:
            raise ValueError(
                f"--seqlen ({args.seqlen}) must be a multiple of --chunk_size ({args.chunk_size})"
            )
        if args.varlen:
            args.initial_states = True
    elif routine == "checkpointing_ssu":
        if args.cache_steps < 1:
            raise ValueError("checkpointing_ssu requires --cache_steps >= 1 (T)")
        if args.max_window is None:
            args.max_window = min(16, 2 * args.cache_steps)
        if not (args.cache_steps <= args.max_window <= 16):
            raise ValueError(
                f"checkpointing_ssu needs cache_steps <= max_window <= 16, got "
                f"T={args.cache_steps}, max_window={args.max_window}"
            )
        if args.prev_tokens is None:
            args.prev_tokens = args.max_window
        if not (0 <= args.prev_tokens <= args.max_window):
            raise ValueError("--prev_tokens must be in [0, max_window]")
    elif routine == "replayssm_materialize":
        if args.max_window is None:
            args.max_window = 16
        if not (1 <= args.max_window <= 16):
            raise ValueError("--max_window must be in [1, 16]")
        if args.replay_prefix_len is None:
            args.replay_prefix_len = args.max_window
        if not (0 <= args.replay_prefix_len <= args.max_window):
            raise ValueError("--replay_prefix_len must be in [0, max_window]")
        if args.active_requests is None:
            args.active_requests = args.batch_size
        if not (0 <= args.active_requests <= args.batch_size):
            raise ValueError("--active_requests must be in [0, batch_size]")

    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


def _validate_ssu_args(args):
    # Validate dim is supported by the CUDA kernel dispatch
    supported_dims = [64, 128, 256]
    if "flashinfer" in args.backends and args.dim not in supported_dims:
        raise ValueError(
            f"dim ({args.dim}) is not supported by the FlashInfer kernel. "
            f"Supported dim values: {supported_dims}. "
            f"Use --backends triton for unsupported dim values."
        )

    # Validate dstate is supported by the CUDA kernel dispatch
    supported_dstates = [64, 128, 256]
    if "flashinfer" in args.backends and args.dstate not in supported_dstates:
        raise ValueError(
            f"dstate ({args.dstate}) is not supported by the FlashInfer kernel. "
            f"Supported dstate values: {supported_dstates}. "
            f"Use --backends triton for unsupported dstate values."
        )

    # NOTE (2026-09-23): the former nheads/ngroups ratio whitelist [1, 8, 16] was
    # removed.  ToT tests/mamba/test_selective_state_update_stp.py
    # (TestSelectiveStateUpdateVariousNgroups) covers ratios 1..64 on the CUDA
    # kernel, and the whitelist blocked the real Nemotron-Ultra-V3 ratio (32).

    if args.varlen:
        if args.cache_steps < 1:
            raise ValueError(
                "--varlen requires --cache_steps >= 1 (specifies max_seqlen)"
            )
    # MTP vertical/horizontal exist in the SM100 CUDA module
    # (include/flashinfer/mamba/invoke_selective_state_update_mtp.cuh) but not for
    # varlen or block-scaled (int16) state; the SM80/SM90 modules reject them.
    if (
        args.algorithm in ("vertical", "horizontal")
        and (args.varlen or (args.cache_steps >= 1 and args.state_dtype == "int16"))
        and "flashinfer" in args.backends
    ):
        raise ValueError(
            f"varlen / int16-state MTP only supports 'auto' or 'simple' algorithm, "
            f"got '{args.algorithm}'"
        )
    if args.intermediate_states and (args.cache_steps < 1 or args.varlen):
        raise ValueError(
            "--intermediate_states requires --cache_steps >= 1 without --varlen"
        )
    if args.dst_checkpoint_step >= 0:
        if args.cache_steps < 1 or args.varlen:
            raise ValueError(
                "--dst_checkpoint_step requires --cache_steps >= 1 without --varlen"
            )
        if args.dst_checkpoint_step >= args.cache_steps:
            raise ValueError("--dst_checkpoint_step must be < --cache_steps")
        if args.intermediate_states:
            raise ValueError(
                "--dst_checkpoint_step and --intermediate_states are mutually exclusive"
            )
        # Only the simple MTP kernel honours a 2D dst column; vertical /
        # horizontal (and auto, which picks horizontal at batch >= 32) would
        # silently ignore it.
        if args.algorithm != "simple":
            raise ValueError("--dst_checkpoint_step requires --algorithm simple")
    # The SSU kernels implement Philox stochastic rounding for fp16 state only
    # (a static_assert otherwise).
    if args.philox_rounds > 0 and args.state_dtype != "float16":
        raise ValueError("--philox_rounds requires --state_dtype float16")


# ==============================================================================
# Shared helpers
# ==============================================================================


def _refcheck(
    args, backend, reference, tested, rtol, atol, what="output", threshold_pct=0.01
):
    """Compare one backend's tensor against the reference; raise on mismatch."""
    (
        num_different_elements,
        num_elements,
        num_different_elements_percentage,
    ) = is_close_stats(reference.float(), tested.float(), rtol=rtol, atol=atol)
    if num_different_elements_percentage > threshold_pct:
        print(
            f"[ERROR] {what} mismatch from backend {backend}: "
            f"{num_different_elements}/{num_elements} ({num_different_elements_percentage:.4f}%) elements differ "
            f"(threshold: {threshold_pct}%, rtol={rtol}, atol={atol})"
        )
        if not args.allow_output_mismatch:
            raise AssertionError(
                f"[ERROR] Backend {backend} {what} mismatch with {num_different_elements} elements"
            )
    elif args.verbose >= 1:
        print(
            f"[REFCHECK] Backend {backend} {what}: PASSED "
            f"({num_different_elements}/{num_elements} elements differ, "
            f"{num_different_elements_percentage:.4f}% <= {threshold_pct}%)"
        )


def _report(args, res, backend, times, problem_bytes, problem_flops, cols):
    if len(times) == 0:
        return
    median_time = np.median(times)
    std_time = np.std(times)
    tflops = problem_flops / (10**9 * median_time)  # TFLOPs/sec
    tb_per_sec = problem_bytes / (10**9 * median_time)  # TB/sec
    print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)
    if args.output_path is not None:
        cur_res = defaultdict(str)
        cur_res["routine"] = args.routine
        cur_res["median_time"] = median_time
        cur_res["std_time"] = std_time
        cur_res["tflops"] = tflops
        cur_res["tb_per_sec"] = tb_per_sec
        cur_res["backend"] = backend
        cur_res["case_tag"] = args.case_tag
        for k, v in cols.items():
            cur_res[k] = v
        res.append(cur_res)


def _make_state(shape, state_dtype, device, quant_dim=-1):
    """Random state; block-quantized dtypes also return a float32 decode scale
    reduced over the last (dstate) axis."""
    if state_dtype in (torch.int16, torch.int8, torch.float8_e4m3fn):
        f32 = torch.randn(*shape, dtype=torch.float32, device=device)
        qmax = {
            torch.int16: float(torch.iinfo(torch.int16).max),
            torch.int8: 127.0,
            torch.float8_e4m3fn: 448.0,
        }[state_dtype]
        amax = f32.abs().amax(dim=quant_dim).clamp(min=1e-30)
        encode = qmax / amax
        scaled = f32 * encode.unsqueeze(quant_dim)
        if state_dtype == torch.float8_e4m3fn:
            q = scaled.clamp(-qmax, qmax).to(state_dtype)
        else:
            q = scaled.round().clamp(-qmax, qmax).to(state_dtype)
        return q, (1.0 / encode).contiguous()
    return torch.randn(*shape, dtype=state_dtype, device=device), None


def _clone_opt(t):
    return None if t is None else t.clone()


# ==============================================================================
# selective_state_update
# ==============================================================================


def testSelectiveStateUpdate(args):
    """
    Test selective_state_update API for Mamba layers.

    This test:
    1. Generates random input tensors for SSM state update
    2. Runs selective_state_update with the requested backend(s)
       - 'flashinfer': FlashInfer CUDA kernel (architecture-specific: base/SM90/SM100+)
       - 'triton': Triton reference implementation
    3. Optionally runs reference check (compares against Triton reference)
    4. Measures performance metrics (memory bandwidth)

    Supports single-token (STP, cache_steps=0), multi-token (MTP, cache_steps>=1)
    and varlen modes, plus the spec-decode (intermediate states,
    disable_state_update, dst checkpoint), int16 block-scaled state and Philox
    stochastic-rounding knobs.

    Note: selective_state_update is memory-bandwidth bound, so TB/sec is the
    primary performance metric.
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print("[INFO] Running testSelectiveStateUpdate")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]  # Make a copy to avoid modifying the original
    batch_size = args.batch_size
    nheads = args.nheads
    dim = args.dim
    dstate = args.dstate
    ngroups = args.ngroups
    cache_steps = args.cache_steps
    has_z = args.has_z
    dt_softplus = args.dt_softplus
    is_varlen = args.varlen
    algorithm = args.algorithm
    disable_state_update = args.disable_state_update
    use_intermediate = args.intermediate_states
    dst_ckpt_step = args.dst_checkpoint_step
    philox_rounds = args.philox_rounds
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    state_dtype = _STATE_DTYPES[args.state_dtype]
    weight_dtype = dtype_str_to_torch_dtype(args.weight_dtype)
    ## Done parsing input arguments

    ## Determine mode
    is_mtp = cache_steps >= 1 and not is_varlen
    T = cache_steps if is_mtp else None

    if is_varlen:
        n_seqs = batch_size
        max_seqlen = cache_steps
        total_tokens = n_seqs * max_seqlen

    ## Prepare input tensors
    ssm_state_cache_size = max(384, batch_size * 10)

    # State cache: (total_entries, nheads, dim, dstate) - contiguous.
    # int16 state is block-quantized along dstate with a float32 decode scale
    # (state_cache_size, nheads, dim).
    state_cache, state_scale = _make_state(
        (ssm_state_cache_size, nheads, dim, dstate), state_dtype, device
    )

    if is_varlen:
        # Varlen: 3D flattened inputs (total_tokens, ...)
        x = torch.randn(total_tokens, nheads, dim, dtype=input_dtype, device=device)
        dt_base = torch.randn(total_tokens, nheads, dtype=weight_dtype, device=device)
        dt = dt_base.as_strided((total_tokens, nheads, dim), (nheads, 1, 0))
        B = torch.randn(total_tokens, ngroups, dstate, dtype=input_dtype, device=device)
        C = torch.randn(total_tokens, ngroups, dstate, dtype=input_dtype, device=device)
        z = None
        if has_z:
            z = torch.randn(total_tokens, nheads, dim, dtype=input_dtype, device=device)
    elif T is not None:
        # MTP: 4D inputs (batch_size, T, ...)
        x = torch.randn(batch_size, T, nheads, dim, dtype=input_dtype, device=device)
        dt_base = torch.randn(batch_size, T, nheads, dtype=weight_dtype, device=device)
        dt = dt_base.as_strided(
            (batch_size, T, nheads, dim), (T * nheads, nheads, 1, 0)
        )
        B = torch.randn(
            batch_size, T, ngroups, dstate, dtype=input_dtype, device=device
        )
        C = torch.randn(
            batch_size, T, ngroups, dstate, dtype=input_dtype, device=device
        )
        z = None
        if has_z:
            z = torch.randn(
                batch_size, T, nheads, dim, dtype=input_dtype, device=device
            )
    else:
        # STP: 3D inputs (batch_size, ...)
        x = torch.randn(batch_size, nheads, dim, dtype=input_dtype, device=device)
        dt_base = torch.randn(batch_size, nheads, dtype=weight_dtype, device=device)
        dt = dt_base.as_strided((batch_size, nheads, dim), (nheads, 1, 0))
        B = torch.randn(batch_size, ngroups, dstate, dtype=input_dtype, device=device)
        C = torch.randn(batch_size, ngroups, dstate, dtype=input_dtype, device=device)
        z = None
        if has_z:
            z = torch.randn(batch_size, nheads, dim, dtype=input_dtype, device=device)

    # A: (nheads, dim, dstate) - negative values, broadcasting (one value per head)
    A_base = -torch.rand(nheads, dtype=torch.float32, device=device) - 1.0
    A = A_base.as_strided((nheads, dim, dstate), (1, 0, 0))

    # D: (nheads, dim) - broadcasting (one value per head)
    D_base = torch.randn(nheads, dtype=weight_dtype, device=device)
    D = D_base.as_strided((nheads, dim), (1, 0))

    # dt_bias: (nheads, dim) - broadcasting (one value per head)
    dt_bias_base = torch.rand(nheads, dtype=weight_dtype, device=device) - 4.0
    dt_bias = dt_bias_base.as_strided((nheads, dim), (1, 0))

    # Varlen-specific tensors
    dst_ckpt_indices = None
    if is_varlen:
        cu_seqlens = torch.arange(
            0, total_tokens + 1, max_seqlen, device=device, dtype=torch.int32
        )
        perm = torch.randperm(ssm_state_cache_size, device=device).to(torch.int32)
        src_indices = perm[: n_seqs * max_seqlen].reshape(n_seqs, max_seqlen)
        dst_indices = perm[n_seqs * max_seqlen : 2 * n_seqs * max_seqlen].reshape(
            n_seqs, max_seqlen
        )
        num_accepted = torch.ones(n_seqs, device=device, dtype=torch.int32)
        slot_idx = None
    else:
        cu_seqlens = None
        src_indices = None
        dst_indices = None
        num_accepted = None
        perm = torch.randperm(ssm_state_cache_size, dtype=torch.int64, device=device)
        slot_idx = perm[:batch_size]
        if dst_ckpt_step >= 0:
            # Checkpoint-at-token-k layout: one destination slot per request in
            # column k, pad everywhere else (Cake's dynamic_checkpoint{0,1,3,7}).
            dst_ckpt_indices = torch.full(
                (batch_size, T), -1, dtype=torch.int64, device=device
            )
            dst_ckpt_indices[:, dst_ckpt_step] = perm[batch_size : 2 * batch_size]

    # Spec-decode intermediate state buffer
    intermediate_states_buffer = None
    intermediate_state_indices = None
    intermediate_state_scales = None
    if use_intermediate:
        intermediate_states_buffer = torch.zeros(
            batch_size, T, nheads, dim, dstate, dtype=state_dtype, device=device
        )
        intermediate_state_indices = torch.arange(
            batch_size, dtype=slot_idx.dtype, device=device
        )
        if state_scale is not None:
            intermediate_state_scales = torch.ones(
                batch_size, T, nheads, dim, dtype=torch.float32, device=device
            )

    rand_seed = None
    if philox_rounds > 0:
        rand_seed = torch.tensor(42, dtype=torch.int64, device=device)
    # Triton's cvt.rs path needs sm_100a; elsewhere its reference rounds to
    # nearest while the CUDA kernel uses its software SR (tests do the same).
    triton_rand_seed = rand_seed
    if rand_seed is not None:
        from flashinfer.utils import is_cvt_rs_supported

        if not is_cvt_rs_supported():
            triton_rand_seed = None

    if args.verbose >= 2:
        mode_str = "varlen" if is_varlen else ("MTP" if is_mtp else "STP")
        print(f"[VVERBOSE] Mode: {mode_str}")
        print(f"[VVERBOSE] {state_cache.shape = }, {state_cache.dtype = }")
        print(f"[VVERBOSE] {x.shape = }, {x.dtype = }")
        print(f"[VVERBOSE] {dt.shape = }, {dt.dtype = }")
        print(f"[VVERBOSE] {A.shape = }, {A.dtype = }")
        print(f"[VVERBOSE] {B.shape = }, {B.dtype = }")
        print(f"[VVERBOSE] {C.shape = }, {C.dtype = }")
        print(f"[VVERBOSE] {D.shape = }, {D.dtype = }")
        print(f"[VVERBOSE] {dt_bias.shape = }, {dt_bias.dtype = }")
        if slot_idx is not None:
            print(f"[VVERBOSE] {slot_idx.shape = }")
        if cu_seqlens is not None:
            print(f"[VVERBOSE] {cu_seqlens.shape = }")
            print(f"[VVERBOSE] {src_indices.shape = }, {dst_indices.shape = }")
        print(f"[VVERBOSE] {has_z = }, {dt_softplus = }")
        print(
            f"[VVERBOSE] {disable_state_update = }, {use_intermediate = }, "
            f"{dst_ckpt_step = }, {philox_rounds = }, "
            f"state_scale={'yes' if state_scale is not None else 'no'}"
        )
        if z is not None:
            print(f"[VVERBOSE] {z.shape = }, {z.dtype = }")

    # Cache steps for Triton reference (None for STP, integer for MTP)
    triton_cache_steps = cache_steps if cache_steps > 0 else None

    # dt / A / D / dt_bias are stride-0 broadcast views: keep them out of
    # bench_gpu_time's input_args, whose rotating-buffer clone
    # (_clone_structure: empty_strided + copy_) cannot write a stride-0 tensor
    # ("more than one element of the written-to tensor refers to a single
    # memory location") whenever the working set is small enough to rotate.
    def run_backend(backend, state, x, B, C, s_scale, i_buf, i_scales):
        if is_varlen:
            if backend == "flashinfer":
                return flashinfer.mamba.selective_state_update(
                    state,
                    x,
                    dt,
                    A,
                    B,
                    C,
                    D,
                    z=z,
                    dt_bias=dt_bias,
                    dt_softplus=dt_softplus,
                    state_batch_indices=src_indices,
                    state_scale=s_scale,
                    disable_state_update=disable_state_update,
                    rand_seed=rand_seed,
                    philox_rounds=philox_rounds if philox_rounds > 0 else 10,
                    algorithm=algorithm,
                    dst_state_batch_indices=dst_indices,
                    num_accepted_tokens=num_accepted,
                    cu_seqlens=cu_seqlens,
                    cache_steps=cache_steps,
                    backend="auto",
                )
            elif backend == "triton":
                return selective_state_update_varlen_triton(
                    state,
                    x,
                    dt,
                    A,
                    B,
                    C,
                    D=D,
                    z=z,
                    dt_bias=dt_bias,
                    dt_softplus=dt_softplus,
                    state_batch_indices=src_indices,
                    dst_state_batch_indices=dst_indices,
                    num_accepted_tokens=num_accepted,
                    cu_seqlens=cu_seqlens,
                )
            else:
                raise ValueError(f"Unsupported backend: {backend}")
        else:
            if backend == "flashinfer":
                return flashinfer.mamba.selective_state_update(
                    state,
                    x,
                    dt,
                    A,
                    B,
                    C,
                    D,
                    z=z,
                    dt_bias=dt_bias,
                    dt_softplus=dt_softplus,
                    state_batch_indices=slot_idx,
                    state_scale=s_scale,
                    disable_state_update=disable_state_update,
                    intermediate_states_buffer=i_buf,
                    intermediate_state_indices=intermediate_state_indices,
                    intermediate_state_scales=i_scales,
                    rand_seed=rand_seed,
                    philox_rounds=philox_rounds if philox_rounds > 0 else 10,
                    cache_steps=cache_steps,
                    algorithm=algorithm,
                    dst_state_batch_indices=dst_ckpt_indices,
                    backend="auto",
                )
            elif backend == "triton":
                # Triton reference has no dst_state_batch_indices: with
                # --dst_checkpoint_step it writes the state in place instead,
                # which leaves the output unchanged.
                return selective_state_update_triton_reference(
                    state,
                    x,
                    dt,
                    A,
                    B,
                    C,
                    D,
                    z=z,
                    dt_bias=dt_bias,
                    dt_softplus=dt_softplus,
                    state_batch_indices=slot_idx,
                    disable_state_update=disable_state_update,
                    intermediate_states_buffer=i_buf,
                    cache_steps=triton_cache_steps,
                    intermediate_state_indices=intermediate_state_indices,
                    state_scale=s_scale,
                    intermediate_state_scales=i_scales,
                    rand_seed=triton_rand_seed,
                    philox_rounds=philox_rounds,
                )
            else:
                raise ValueError(f"Unsupported backend: {backend}")

    def fresh_inputs(state_src):
        return (
            state_src.clone(),
            x,
            B,
            C,
            _clone_opt(state_scale),
            _clone_opt(intermediate_states_buffer),
            _clone_opt(intermediate_state_scales),
        )

    # Reference check: use Triton as golden reference
    # Save a clean snapshot of state_cache before any benchmarking, because
    # bench_gpu_time mutates state_cache in-place across many iterations.
    # All refcheck clones must come from this clean snapshot.
    has_reference_output = False
    clean_state_snapshot = state_cache.clone()
    if run_refcheck:
        reference_output = (
            run_backend("triton", *fresh_inputs(clean_state_snapshot)).detach().clone()
        )
        has_reference_output = True

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    kernel_label = {}
    bench_inputs = (
        state_cache,
        x,
        B,
        C,
        state_scale,
        intermediate_states_buffer,
        intermediate_state_scales,
    )
    for cur_backend in backends:
        if run_refcheck and cur_backend != "triton":
            # Always clone from the clean snapshot, not from state_cache
            # (which may have been mutated by previous backend's bench_gpu_time)
            outputs[cur_backend] = (
                run_backend(cur_backend, *fresh_inputs(clean_state_snapshot))
                .detach()
                .clone()
            )
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, *bench_inputs),
        )
        if cur_backend == "flashinfer":
            major, _ = torch.cuda.get_device_capability(device)
            variant = "sm100" if major >= 10 else ("sm90" if major >= 9 else "base")
            kernel_label[cur_backend] = f"cuda_{variant}"
        else:
            kernel_label[cur_backend] = "triton_reference"

    # Compare outputs against Triton reference.  int16 state is block-quantized
    # and Philox rounding differs in op order, so they get the looser test tol.
    atol = 1e-1 if state_dtype == torch.int16 else 1e-3
    if run_refcheck and has_reference_output:
        for backend, out in outputs.items():
            _refcheck(args, backend, reference_output, out, rtol=1e-2, atol=atol)

    # Compute and report performance metrics
    T_val = cache_steps if cache_steps > 0 else 1
    sb = state_dtype.itemsize
    ib = input_dtype.itemsize
    wb = weight_dtype.itemsize
    # Memory traffic (physical bytes accessed; dt/A/D/dt_bias broadcast per head)
    read_bytes = (
        batch_size * nheads * dim * dstate * sb  # state
        + batch_size * T_val * nheads * dim * ib  # x
        + batch_size * T_val * nheads * wb  # dt (broadcast)
        + nheads * 4  # A (float32, broadcast)
        + 2 * batch_size * T_val * ngroups * dstate * ib  # B, C
        + 2 * nheads * wb  # D, dt_bias (broadcast)
    )
    if has_z:
        read_bytes += batch_size * T_val * nheads * dim * ib
    write_bytes = batch_size * T_val * nheads * dim * ib  # output
    if not disable_state_update:
        # Varlen writes the state after every step (one dst slot per token,
        # kernel_selective_state_update_mtp_simple.cuh:244-249); STP/MTP once.
        state_writes = T_val if is_varlen else 1
        write_bytes += state_writes * batch_size * nheads * dim * dstate * sb
    if use_intermediate:
        write_bytes += batch_size * T_val * nheads * dim * dstate * sb
    if state_scale is not None:
        read_bytes += batch_size * nheads * dim * 4
        write_bytes += batch_size * nheads * dim * 4
    problem_bytes = read_bytes + write_bytes
    # FLOPs estimate (TIE_HDIM case): ~5 FLOPs per (dim, dstate) element per token
    problem_flops = batch_size * T_val * nheads * dim * dstate * 5

    for backend in backends:
        _report(
            args,
            res,
            backend,
            backend_times[backend],
            problem_bytes,
            problem_flops,
            {
                "batch_size": batch_size,
                "nheads": nheads,
                "dim": dim,
                "dstate": dstate,
                "ngroups": ngroups,
                "cache_steps": cache_steps,
                "input_dtype": str(input_dtype),
                "state_dtype": str(state_dtype),
                "weight_dtype": str(weight_dtype),
                "has_z": has_z,
                "dt_softplus": dt_softplus,
                "varlen": is_varlen,
                "algorithm": algorithm,
                "intermediate_states": use_intermediate,
                "disable_state_update": disable_state_update,
                "dst_checkpoint_step": dst_ckpt_step,
                "philox_rounds": philox_rounds,
                "mamba_kernel": kernel_label.get(backend, ""),
            },
        )
    return res


# ==============================================================================
# ssd_combined (Mamba2 chunked prefill)
# ==============================================================================


def _ssd_varlen_metadata(num_seqs, seqlen, chunk_size, device):
    """Packed equal-length sequences: seq_idx (1, total), chunk_indices/offsets."""
    total = num_seqs * seqlen
    seq_idx = (
        torch.arange(num_seqs, dtype=torch.int32, device=device)
        .repeat_interleave(seqlen)
        .unsqueeze(0)
    )
    cu_seqlens = torch.arange(0, total + 1, seqlen, dtype=torch.int32, device=device)
    # Sequence starts are chunk-aligned (seqlen % chunk_size == 0), so every
    # physical chunk holds exactly one logical segment at offset 0.
    nchunks = total // chunk_size
    chunk_indices = torch.arange(nchunks, dtype=torch.int32, device=device)
    chunk_offsets = torch.zeros(nchunks, dtype=torch.int32, device=device)
    return seq_idx, cu_seqlens, chunk_indices, chunk_offsets


def testSSDCombined(args):
    """
    Benchmark the Mamba2 SSD chunked-prefill forward (ssd_combined).

    Backends:
      - 'cute-dsl': SSDCombined(backend="cute"), tcgen05 CuTe-DSL kernel
        (cc major 10/11: SM100/SM103/SM107/SM110)
      - 'triton'  : tests/mamba/triton_reference _mamba_chunk_scan_combined_fwd

    Refcheck compares output and final states against the Triton reference.
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print("[INFO] Running testSSDCombined")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    backends = args.backends[:]
    res = []
    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    from triton_reference.ssd_combined import _mamba_chunk_scan_combined_fwd

    batch = args.batch_size
    seqlen = args.seqlen
    nheads = args.nheads
    headdim = args.dim
    dstate = args.dstate
    ngroups = args.ngroups
    chunk_size = args.chunk_size
    is_varlen = args.varlen
    has_init = args.initial_states
    has_d = not args.no_d
    has_z = args.has_z
    dt_softplus = args.dt_softplus
    io_dtype = torch.bfloat16
    state_dtype = _STATE_DTYPES[args.state_dtype]
    is_cuda_graph_compatible = not args.no_cuda_graph

    rows = 1 if is_varlen else batch
    L = batch * seqlen if is_varlen else seqlen
    nchunks = L // chunk_size
    num_seqs = batch

    x = torch.randn(rows, L, nheads, headdim, dtype=io_dtype, device=device)
    dt = torch.randn(rows, L, nheads, dtype=torch.float32, device=device)
    A = -torch.rand(nheads, dtype=torch.float32, device=device) - 1.0
    B = torch.randn(rows, L, ngroups, dstate, dtype=io_dtype, device=device)
    C = torch.randn(rows, L, ngroups, dstate, dtype=io_dtype, device=device)
    D = None
    if has_d:
        D = (
            torch.randn(nheads, headdim, dtype=io_dtype, device=device)
            if args.d_has_hdim
            else torch.randn(nheads, dtype=io_dtype, device=device)
        )
    z = torch.randn_like(x) if has_z else None
    dt_bias = torch.rand(nheads, dtype=torch.float32, device=device) - 4.0
    initial_states = None
    if has_init:
        initial_states = torch.randn(
            num_seqs, nheads, headdim, dstate, dtype=state_dtype, device=device
        )
    seq_idx = cu_seqlens = chunk_indices = chunk_offsets = None
    if is_varlen:
        seq_idx, cu_seqlens, chunk_indices, chunk_offsets = _ssd_varlen_metadata(
            num_seqs, seqlen, chunk_size, device
        )

    if args.verbose >= 2:
        print(
            f"[VVERBOSE] mode={'varlen' if is_varlen else 'batched'} x={tuple(x.shape)} "
            f"B={tuple(B.shape)} nchunks={nchunks} D={None if D is None else tuple(D.shape)} "
            f"z={has_z} init_states={has_init} state_dtype={state_dtype}"
        )

    # With z, the Triton reference's _chunk_scan_fwd returns out_x (the output
    # BEFORE the z * sigmoid(z) gate) and writes the gated output into `out`;
    # FlashInfer returns the gated output, so compare against `out`.
    triton_out = torch.empty_like(x)

    def triton_call(x=x, dt=dt, B=B, C=C):
        return _mamba_chunk_scan_combined_fwd(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            D=D,
            z=z,
            dt_bias=dt_bias,
            initial_states=initial_states,
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            cu_seqlens=cu_seqlens,
            dt_softplus=dt_softplus,
            state_dtype=state_dtype,
            out=triton_out,
        )

    def triton_split(ret):
        # (out_x, dt, dA_cumsum, states, final_states[, varlen_states])
        return triton_out, (ret[5] if is_varlen else ret[4])

    triton_ok = not (has_init and not is_varlen)
    if "triton" in backends and not triton_ok:
        print(
            "[ERROR] triton: the Triton reference supports initial_states only in varlen "
            "mode (batch==1 with seq_idx/chunk metadata); dropping triton."
        )
        backends.remove("triton")

    runners = {}
    for b in list(backends):
        if b == "triton":
            continue
        lib_backend = "cute"
        try:
            runners[b] = flashinfer.mamba.SSDCombined(
                chunk_size=chunk_size,
                nheads=nheads,
                headdim=headdim,
                dstate=dstate,
                ngroups=ngroups,
                io_dtype=io_dtype,
                state_dtype=state_dtype,
                has_d=has_d,
                d_has_hdim=args.d_has_hdim,
                has_initial_states=has_init,
                has_varlen=is_varlen,
                has_z=has_z,
                seq_idx_dtype=torch.int32,
                backend=lib_backend,
            )
        except (ValueError, AssertionError) as e:
            print(
                f"[ERROR] {b}: SSDCombined(backend={lib_backend!r}) rejected this case: {e}"
            )
            backends.remove(b)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    outs = {
        b: torch.empty(
            rows, nheads, headdim, nchunks, chunk_size, dtype=io_dtype, device=device
        )
        for b in runners
    }

    # x / dt / B / C come in as arguments so bench_gpu_time can rotate them for
    # a cold L2 under CUDA-graph timing; everything else is small or fixed.
    def run_backend(backend, x=x, dt=dt, B=B, C=C):
        if backend == "triton":
            return triton_split(triton_call(x, dt, B, C))
        return runners[backend].run(
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            initial_states=initial_states,
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            out=outs[backend],
            return_final_states=True,
        )

    reference = None
    if args.refcheck:
        if triton_ok:
            ref_out, ref_states = triton_split(triton_call())
            reference = (ref_out.detach().clone(), ref_states.detach().clone())
        else:
            print(
                "[INFO] refcheck skipped: no Triton reference for batched initial_states."
            )

    backend_times = {}
    for b in backends:
        if reference is not None and b != "triton":
            o, s = run_backend(b)
            torch.cuda.synchronize()
            # Tolerances from tests/mamba/test_chunk_scan_combined.py (7e-2).
            _refcheck(args, b, reference[0].to(o.dtype), o, rtol=7e-2, atol=7e-2)
            _refcheck(
                args,
                b,
                reference[1].to(s.dtype),
                s,
                rtol=7e-2,
                atol=7e-2,
                what="final_states",
            )
        backend_times[b] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(b, x, dt, B, C),
        )

    # Traffic: x, z, out (io), B, C, dt (fp32), initial + final states
    tokens = rows * L
    ib = io_dtype.itemsize
    problem_bytes = (
        tokens * nheads * headdim * ib * (2 + (1 if has_z else 0))
        + 2 * tokens * ngroups * dstate * ib
        + tokens * nheads * 4
        + num_seqs
        * nheads
        * headdim
        * dstate
        * state_dtype.itemsize
        * (2 if has_init else 1)
    )
    # FLOPs (dense SSD): intra-chunk CB (L*chunk*dstate per group) + scan
    # (L*chunk*headdim per head) + chunk states and state->output
    # (2 * L*headdim*dstate per head), 2 FLOPs per MAC.
    problem_flops = (
        2
        * tokens
        * (
            chunk_size * dstate * ngroups
            + chunk_size * headdim * nheads
            + 2 * headdim * dstate * nheads
        )
    )
    labels = {
        "cute-dsl": "cute_dsl:SSDCombined",
        "triton": "triton_reference",
    }
    for b in backends:
        _report(
            args,
            res,
            b,
            backend_times[b],
            problem_bytes,
            problem_flops,
            {
                "batch_size": batch,
                "nheads": nheads,
                "dim": headdim,
                "dstate": dstate,
                "ngroups": ngroups,
                "ssm_seqlen": seqlen,
                "chunk_size": chunk_size,
                "input_dtype": str(io_dtype),
                "state_dtype": str(state_dtype),
                "has_z": has_z,
                "dt_softplus": dt_softplus,
                "varlen": is_varlen,
                "mamba_kernel": labels[b],
            },
        )
    return res


# ==============================================================================
# checkpointing_ssu (ReplaySSM MTP decode)
# ==============================================================================


def testCheckpointingSSU(args):
    """
    Benchmark flashinfer.mamba.checkpointing_ssu (ReplaySSM MTP decode with a
    ring-buffer replay window) against the Triton replay reference.

    Backends: flashinfer (algorithm=auto with the two-kernel scratch trio),
    flashinfer-monolith, flashinfer-two-kernel, triton (replay reference,
    persistent_dynamic mode).  prev_tokens + T > max_window selects the
    checkpoint-write path; otherwise the no-write (replay only) path runs.
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print("[INFO] Running testCheckpointingSSU")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )
    backends = args.backends[:]
    res = []
    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    from flashinfer.mamba.checkpointing_ssu import (
        allocate_checkpointing_ssu_scratch,
        checkpointing_ssu,
    )
    from triton_reference.replay_selective_state_update import (
        REPLAY_WORK_CACHE_BUF_IDX,
        REPLAY_WORK_CACHE_SLOT,
        REPLAY_WORK_ITEM_WIDTH,
        REPLAY_WORK_PNAT,
        REPLAY_WORK_POSITION_IN_DECODE_BATCH,
        replay_selective_state_update,
    )

    batch = args.batch_size
    nheads = args.nheads
    dim = args.dim
    dstate = args.dstate
    ngroups = args.ngroups
    T = args.cache_steps
    max_window = args.max_window
    pnat = args.prev_tokens
    act_dtype = torch.bfloat16
    state_dtype = _STATE_DTYPES[args.state_dtype]
    weight_dtype = dtype_str_to_torch_dtype(args.weight_dtype)
    philox_rounds = args.philox_rounds
    is_cuda_graph_compatible = not args.no_cuda_graph
    write_path = (pnat + T) > max_window

    cache_size = max(2 * batch, 8)
    state_batch_indices = torch.randperm(cache_size, device=device)[:batch].to(
        torch.int32
    )
    state0, state_scale0 = _make_state(
        (cache_size, nheads, dim, dstate), state_dtype, device
    )
    ring_len = max_window + T
    x_cache0 = torch.randn(
        cache_size, nheads, ring_len, dim, dtype=act_dtype, device=device
    )
    B_cache0 = torch.randn(
        cache_size, ngroups, ring_len, dstate, dtype=act_dtype, device=device
    )
    # Positive (softplus-output) dt keeps recomputed decays <= 1.
    dt_cache0 = torch.randn(
        cache_size, nheads, ring_len, dtype=torch.float32, device=device
    ).abs()
    ring_start = torch.randint(
        0, ring_len, (cache_size,), dtype=torch.int32, device=device
    )
    prev_tokens = torch.full((cache_size,), pnat, dtype=torch.int32, device=device)

    A = (-torch.rand(nheads, device=device) - 0.5).as_strided(
        (nheads, dim, dstate), (1, 0, 0)
    )
    dt_bias = torch.randn(nheads, dtype=weight_dtype, device=device).as_strided(
        (nheads, dim), (1, 0)
    )
    D = torch.randn(nheads, dtype=weight_dtype, device=device).as_strided(
        (nheads, dim), (1, 0)
    )
    x = torch.randn(batch, T, nheads, dim, dtype=act_dtype, device=device)
    dt = torch.randn(batch, T, nheads, dtype=weight_dtype, device=device).as_strided(
        (batch, T, nheads, dim), (T * nheads, nheads, 1, 0)
    )
    B = torch.randn(batch, T, ngroups, dstate, dtype=act_dtype, device=device)
    C = torch.randn(batch, T, ngroups, dstate, dtype=act_dtype, device=device)
    z = (
        torch.randn(batch, T, nheads, dim, dtype=act_dtype, device=device)
        if args.has_z
        else None
    )
    # The ReplaySSM kernels take a 1-D single-element seed tensor.
    rand_seed = (
        torch.tensor([42], dtype=torch.int64, device=device)
        if philox_rounds > 0
        else None
    )
    scratch = allocate_checkpointing_ssu_scratch(
        batch, nheads, T, max_window, act_dtype, device
    )

    # Triton replay work items (write-first sorted), mirrors the test helper.
    pos = torch.arange(batch, device=device, dtype=torch.int32)
    slot = state_batch_indices
    p = prev_tokens[slot.long()]
    wmask = (p + T) > max_window
    n_writes = wmask.sum().to(torch.int32).reshape(1)
    order = torch.argsort((~wmask).to(torch.int32), stable=True).long()
    work = torch.empty(batch, REPLAY_WORK_ITEM_WIDTH, device=device, dtype=torch.int32)
    work[:, REPLAY_WORK_POSITION_IN_DECODE_BATCH] = pos[order]
    work[:, REPLAY_WORK_CACHE_SLOT] = slot[order]
    work[:, REPLAY_WORK_PNAT] = p[order]
    work[:, REPLAY_WORK_CACHE_BUF_IDX] = 0
    work = work.contiguous()

    def make_mutables():
        return dict(
            state=state0.clone(),
            state_scale=_clone_opt(state_scale0),
            x_cache=x_cache0.clone(),
            B_cache=B_cache0.clone(),
            dt_cache=dt_cache0.clone(),
            out=torch.zeros(batch, T, nheads, dim, dtype=act_dtype, device=device),
        )

    def run_backend(backend, m):
        if backend == "triton":
            replay_selective_state_update(
                m["state"],
                m["x_cache"],
                m["B_cache"],
                m["dt_cache"],
                ring_start,
                prev_tokens,
                x,
                dt,
                A,
                B,
                C,
                m["out"],
                n_writes=n_writes,
                replay_work_items=work,
                D=D,
                z=z,
                dt_bias=dt_bias,
                dt_softplus=args.dt_softplus,
                state_batch_indices=state_batch_indices,
                rand_seed=rand_seed,
                philox_rounds=philox_rounds,
                state_scales=m["state_scale"],
                mode="persistent_dynamic",
            )
            return m["out"]
        algo = {
            "flashinfer": "auto",
            "flashinfer-monolith": "monolith",
            "flashinfer-two-kernel": "two-kernel",
        }[backend]
        kw = {}
        if algo != "monolith":
            kw.update(cb_scaled=scratch[0], cumAdt_vec=scratch[1], cb_old=scratch[2])
        if args.d_split is not None:
            kw["d_split"] = args.d_split
        return checkpointing_ssu(
            m["state"],
            m["x_cache"],
            m["B_cache"],
            m["dt_cache"],
            ring_start,
            prev_tokens,
            x,
            dt,
            A,
            B,
            C,
            m["out"],
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=args.dt_softplus,
            state_batch_indices=state_batch_indices,
            state_scale=m["state_scale"],
            rand_seed=rand_seed,
            philox_rounds=philox_rounds if philox_rounds > 0 else 10,
            algorithm=algo,
            **kw,
        )

    reference = None
    if args.refcheck:
        reference = run_backend("triton", make_mutables()).detach().clone()

    backend_times = {}
    for b in list(backends):
        try:
            if reference is not None and b != "triton":
                o = run_backend(b, make_mutables())
                torch.cuda.synchronize()
                # Tolerances from tests/mamba/test_checkpointing_ssu.py
                _refcheck(args, b, reference, o, rtol=2e-2, atol=5e-1)
            m = make_mutables()
            backend_times[b] = bench_gpu_time(
                fn=run_backend,
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                enable_cupti=args.use_cupti,
                use_cuda_graph=is_cuda_graph_compatible,
                input_args=(b, m),
            )
        except ValueError as e:
            # Library guard (e.g. two-kernel rejects 8-bit state); quote it.
            print(f"[ERROR] {b}: checkpointing_ssu rejected this case: {e}")
            backends.remove(b)

    sb = state_dtype.itemsize
    ab = act_dtype.itemsize
    problem_bytes = (
        batch * nheads * dim * dstate * sb * (2 if write_path else 1)
        + batch * pnat * (nheads * dim * ab + ngroups * dstate * ab + nheads * 4)
        + batch * T * (2 * nheads * dim * ab + 2 * ngroups * dstate * ab + nheads * 4)
    )
    problem_flops = batch * (pnat + T) * nheads * dim * dstate * 5
    labels = {
        # Not autotuned here, so auto takes the library's fallback heuristic.
        "flashinfer": "cuda:auto-untuned",
        "flashinfer-monolith": "cuda:monolith",
        "flashinfer-two-kernel": "cuda:two-kernel",
        "triton": "triton_reference:persistent_dynamic",
    }
    for b in backends:
        _report(
            args,
            res,
            b,
            backend_times.get(b, []),
            problem_bytes,
            problem_flops,
            {
                "batch_size": batch,
                "nheads": nheads,
                "dim": dim,
                "dstate": dstate,
                "ngroups": ngroups,
                "cache_steps": T,
                "max_window": max_window,
                "prev_tokens": pnat,
                "input_dtype": str(act_dtype),
                "state_dtype": str(state_dtype),
                "weight_dtype": str(weight_dtype),
                "has_z": args.has_z,
                "dt_softplus": args.dt_softplus,
                "philox_rounds": philox_rounds,
                "mamba_kernel": labels[b] + (":write" if write_path else ":nowrite"),
            },
        )
    return res


# ==============================================================================
# replayssm_materialize
# ==============================================================================


def testReplaySSMMaterialize(args):
    """
    Benchmark flashinfer.mamba.replayssm_materialize: apply replay_prefix_len
    ring entries to a source state slot and write a destination slot, for
    num_layers x active_requests x nheads work items.  Refcheck is a torch
    recurrence with a state-dtype rounding after every token (as the test).
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print("[INFO] Running testReplaySSMMaterialize")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )
    backends = args.backends[:]
    res = []
    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    from flashinfer.mamba.replayssm_materialize import replayssm_materialize

    batch = args.batch_size
    layers = args.num_layers
    nheads = args.nheads
    dim = args.dim
    dstate = args.dstate
    ngroups = args.ngroups
    heads_per_group = nheads // ngroups
    max_window = args.max_window
    prefix = args.replay_prefix_len
    active = args.active_requests
    state_dtype = _STATE_DTYPES[args.state_dtype]
    philox_rounds = args.philox_rounds
    ring_len = max_window + 1
    quantized = state_dtype in (torch.int8, torch.float8_e4m3fn)

    # slots: [0, batch) sources, [batch, 2*batch) destinations
    slots = 2 * batch
    states, scales = [], []
    for _ in range(layers):
        s, sc = _make_state((slots, nheads, dim, dstate), state_dtype, device)
        states.append(s)
        scales.append(sc)
    x_cache = [
        torch.randn(slots, nheads, ring_len, dim, dtype=torch.bfloat16, device=device)
        for _ in range(layers)
    ]
    b_cache = [
        torch.randn(
            slots, ngroups, ring_len, dstate, dtype=torch.bfloat16, device=device
        )
        for _ in range(layers)
    ]
    dt_cache = [
        torch.rand(slots, nheads, ring_len, dtype=torch.float32, device=device)
        for _ in range(layers)
    ]
    a_vec = [
        -torch.rand(nheads, dtype=torch.float32, device=device) for _ in range(layers)
    ]

    def ptrs(ts):
        return torch.tensor(
            [t.data_ptr() for t in ts], dtype=torch.int64, device=device
        )

    def strides(ts):
        return torch.tensor([t.stride(0) for t in ts], dtype=torch.int64, device=device)

    zero = torch.zeros(layers, dtype=torch.int64, device=device)
    src_slots = (
        torch.arange(batch, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .repeat(layers, 1)
    )
    dst_slots = src_slots + batch
    ring_start = torch.randint(0, ring_len, (batch,), dtype=torch.int32, device=device)
    active_idx = torch.randperm(batch, device=device)[:active].to(torch.int32)
    replay_len = torch.full((batch,), -1, dtype=torch.int32, device=device)
    replay_len[active_idx.long()] = prefix
    active_request_indices = torch.full((batch,), -1, dtype=torch.int32, device=device)
    active_request_indices[:active] = active_idx
    # The ReplaySSM kernels take a 1-D single-element seed tensor.
    rand_seed = (
        torch.tensor([42], dtype=torch.int64, device=device)
        if philox_rounds > 0
        else None
    )
    tables = (
        ptrs(states),
        strides(states),
        ptrs(x_cache),
        strides(x_cache),
        ptrs(b_cache),
        strides(b_cache),
        ptrs(dt_cache),
        strides(dt_cache),
        ptrs(a_vec),
        ptrs(scales) if quantized else zero,
        strides(scales) if quantized else zero,
    )
    before = [s.clone() for s in states] if args.refcheck else None

    def run_backend(backend):
        replayssm_materialize(
            *tables,
            src_slots,
            dst_slots,
            ring_start,
            replay_len,
            active_request_indices,
            state_dtype=state_dtype,
            input_dtype=torch.bfloat16,
            matrixA_dtype=torch.float32,
            dim=dim,
            dstate=dstate,
            num_heads=nheads,
            heads_per_group=heads_per_group,
            max_window=max_window,
            ring_buffer_len=ring_len,
            rand_seed=rand_seed,
            philox_rounds=philox_rounds,
        )

    backend_times = {}
    for b in backends:
        if args.refcheck:
            run_backend(b)
            torch.cuda.synchronize()
            ai = active_idx.long()
            gidx = torch.arange(nheads, device=device) // heads_per_group
            for layer in range(layers):
                if quantized:
                    ref = before[layer][ai].float() * scales[layer][ai].unsqueeze(-1)
                else:
                    ref = before[layer][ai].float()
                starts = ring_start[ai].long()
                for tok in range(prefix):
                    row = (starts + tok) % ring_len  # (active,)
                    # Non-adjacent advanced indices put the broadcast (active,)
                    # axis first: (active, H) / (active, H, dim) / (active, G, dstate)
                    dtv = dt_cache[layer][ai, :, row]
                    xv = x_cache[layer][ai, :, row]
                    bv = b_cache[layer][ai, :, row][:, gidx]  # (active, H, dstate)
                    decay = torch.exp(a_vec[layer][None, :] * dtv)  # (active, H)
                    ref = (
                        ref * decay[..., None, None]
                        + (xv.float()[..., :, None] * bv.float()[..., None, :])
                        * dtv[..., None, None]
                    )
                    if not quantized:
                        ref = ref.to(state_dtype).float()
                got = states[layer][(ai + batch)].float()
                if quantized:
                    got = got * scales[layer][(ai + batch)].unsqueeze(-1)
                _refcheck(
                    args,
                    b,
                    ref,
                    got,
                    rtol=2e-2,
                    atol=5e-2 if not quantized else 2e-1,
                    what=f"layer{layer} materialized state",
                    threshold_pct=0.1,
                )
        # The layer pointer tables bake data_ptrs, so the CUDA-graph timer
        # cannot rotate buffers for a cold L2; time eagerly, where L2 is
        # flushed before every iteration, and record no_cuda_graph=True.
        backend_times[b] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=False,
            input_args=(b,),
        )

    sb = state_dtype.itemsize
    per_item = 2 * dim * dstate * sb + prefix * (dim * 2 + 4)
    per_group_b = prefix * dstate * 2
    problem_bytes = layers * active * (nheads * per_item + ngroups * per_group_b)
    problem_flops = layers * active * nheads * prefix * dim * dstate * 4
    for b in backends:
        _report(
            args,
            res,
            b,
            backend_times[b],
            problem_bytes,
            problem_flops,
            {
                "batch_size": batch,
                "nheads": nheads,
                "dim": dim,
                "dstate": dstate,
                "ngroups": ngroups,
                "num_layers": layers,
                "max_window": max_window,
                "replay_prefix_len": prefix,
                "active_requests": active,
                "state_dtype": str(state_dtype),
                "philox_rounds": philox_rounds,
                "no_cuda_graph": True,
                "mamba_kernel": "cuda:replayssm_materialize",
            },
        )
    return res
