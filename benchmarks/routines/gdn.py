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

import math
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

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
# shared Triton GDN kernels. The repo root is added so that the cuDNN
# linear-attention oracles in tests/test_helpers are importable too.
_repo_root = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
_tests_gdn = os.path.join(_repo_root, "tests", "gdn")
if _tests_gdn not in sys.path:
    sys.path.insert(0, _tests_gdn)
_benchmarks_dir = os.path.join(_repo_root, "benchmarks")
if _benchmarks_dir not in sys.path:
    sys.path.insert(0, _benchmarks_dir)
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

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
# Backend names
# ==============================================================================
#
# Every harness backend name maps to exactly one library dispatch so that a
# column never silently changes kernel family:
#
#   flashinfer   the public API with its default ``backend="auto"`` (and
#                prefill ``use_cp="auto"``): whatever FlashInfer dispatches
#                to on this GPU (#7486).
#   cute         decode only: ``backend="flashinfer"``, the generic CuTe-DSL
#                kernels (#7486).
#   cute-dsl     MTP / prefill / ucache: library ``backend="flashinfer"``
#                (CuTe-DSL kernels, never the Cake path auto may try), prefill
#                additionally ``use_cp=False``.
#   cake_gdn     decode only: ``backend="cake_gdn"``, the Cake GDN kernels
#                (SM100/SM103/SM107, bfloat16 inputs, indexed pools; #7486).
#   cute-dsl-cp  prefill ``backend="flashinfer", use_cp=True`` (CuTe-DSL
#                context-parallel prefill).
#   rubin        decode only: ``backend="rubin"``, the Rubin-tuned T=1
#                bf16-state kernel (SM107 only, #6604).
#   rubin_ugpu   ``rubin`` with a ``ugpu_state_pool`` (each die updates the
#                half of every state that lives in its own memory).
#   cudnn        cuDNN fused SM100-family linear-attention engine.
#
#   cute-dsl-wy  MTP only: the output-only WY BF16-state kernel
#                (gdn_kernels.gdn_decode_bf16_wy_output_only).
#   specialized  gdn_fused_decode_step only: the SM120 specialized kernels;
#                fails when the library would fall back to the composable
#                torch path.
#   triton / fla external baselines.
#
# Explicit backends never fall back: an unsupported request raises, the
# harness prints [ERROR] for the line and records no row for that backend.

ROUTINE_BACKENDS = {
    "gated_delta_rule_decode": [
        "flashinfer",
        "cute",
        "cake_gdn",
        "rubin",
        "rubin_ugpu",
        "triton",
    ],
    "gated_delta_rule_mtp": [
        "flashinfer",
        "cute-dsl",
        "cute-dsl-wy",
        "triton",
    ],
    "chunk_gated_delta_rule": [
        "flashinfer",
        "cute-dsl",
        "cute-dsl-cp",
        "cudnn",
        "fla",
    ],
    "chunk_gated_delta_rule2": ["cudnn"],
    "chunk_gated_delta_product": ["cudnn"],
    "gated_delta_rule_mtp_ucache": ["cute-dsl"],
    "gated_delta_rule_mtp_ucache_flush": ["cute-dsl"],
    "gdn_fused_decode_step": ["flashinfer", "specialized"],
    # gated_delta_rule_replayssm_commit(backend=): flashinfer = "auto"
    "gated_delta_rule_replayssm_commit": ["flashinfer", "simt", "tcgen05"],
}
_ALL_BACKENDS = sorted({b for v in ROUTINE_BACKENDS.values() for b in v})

# Harness backend -> chunk_gated_delta_rule(backend=, use_cp=)
_PREFILL_DISPATCH = {
    "flashinfer": ("auto", "auto"),
    "cute-dsl": ("flashinfer", False),
    "cute-dsl-cp": ("flashinfer", True),
    "cudnn": ("cudnn", False),
}
# Harness backend -> gated_delta_rule_decode*(backend=)
_DECODE_DISPATCH = {
    "flashinfer": "auto",
    "cute": "flashinfer",
    "cute-dsl": "flashinfer",
    "cake_gdn": "cake_gdn",
    "rubin": "rubin",
}

_STATE_DTYPE_CHOICES = [
    "float32",
    "bfloat16",
    "float16",
    "float8_e4m3fn",
    "float8_e5m2",
]


def _torch_dtype(name):
    return {
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
    }.get(name) or dtype_str_to_torch_dtype(name)


# ==============================================================================
# Benchmark infrastructure
# ==============================================================================


_DECODE_LIKE = ("gated_delta_rule_decode", "gated_delta_rule_mtp")
_PREFILL_FAMILY = (
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule2",
    "chunk_gated_delta_product",
)
_UCACHE = ("gated_delta_rule_mtp_ucache", "gated_delta_rule_mtp_ucache_flush")
_COMMIT = ("gated_delta_rule_replayssm_commit",)
#: Routine-specific flags -> the routines that consume them.
_ROUTINE_FLAGS = {
    "input_dtype": _DECODE_LIKE + _PREFILL_FAMILY + _UCACHE,
    "state_dtype": _DECODE_LIKE + _PREFILL_FAMILY + _UCACHE,
    "cache_replayssm": ("gated_delta_rule_mtp",),
    "num_layers": _COMMIT,
    "state_layout": ("gated_delta_rule_decode",),
    "pool_mode": _DECODE_LIKE,
    "s_qo": _PREFILL_FAMILY,
    "update_state": ("gated_delta_rule_mtp",),
    "output_only": ("gated_delta_rule_mtp",),
    "cache_intermediate_states": ("gated_delta_rule_mtp",),
    "use_ssm_state_indices": ("gated_delta_rule_mtp",),
    "strided_qkv": _DECODE_LIKE,
    "no_qk_l2norm": _DECODE_LIKE + _COMMIT,
    "initial_state": _PREFILL_FAMILY,
    "seqlen_dist": _PREFILL_FAMILY,
    "use_state_indices": ("chunk_gated_delta_rule",),
    "checkpoint_every_n_tokens": ("chunk_gated_delta_rule",),
    "qk_l2norm_in_kernel": _PREFILL_FAMILY,
    "num_householder": ("chunk_gated_delta_product",),
    "hist_len": _UCACHE,
    "flush_rate": ("gated_delta_rule_mtp_ucache_flush",),
    "hidden_size": ("gdn_fused_decode_step",),
    "conv_layout": ("gdn_fused_decode_step",),
}


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
    elif args.routine == "chunk_gated_delta_rule2":
        return testChunkGatedDeltaRule2(args)
    elif args.routine == "chunk_gated_delta_product":
        return testChunkGatedDeltaProduct(args)
    elif args.routine in (
        "gated_delta_rule_mtp_ucache",
        "gated_delta_rule_mtp_ucache_flush",
    ):
        return testGatedDeltaRuleMtpUcache(args)
    elif args.routine == "gdn_fused_decode_step":
        return testGdnFusedDecodeStep(args)
    elif args.routine == "gated_delta_rule_replayssm_commit":
        return testGatedDeltaRuleReplaySSMCommit(args)
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
        choices=_STATE_DTYPE_CHOICES,
        help="Data type for the recurrent state. Decode/MTP: float32 or "
        "bfloat16 (bfloat16 selects the BF16 state kernels, requires "
        "head_size=128 and pretranspose layout). Prefill "
        "(chunk_gated_delta_rule*): dtype of the initial/final state (and "
        "checkpoints); the library accepts float32/bfloat16/float16/fp8 on "
        "some backends and rejects the rest loudly.",
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
        choices=["single", "split", "indexed"],
        help="State pool indexing mode. 'single': state pool of size B with "
        "read == write slots (fp32 pretranspose decode passes the state "
        "directly, no indices). 'indexed': same pool, but always passed as "
        "initial_state + initial_state_indices (the pool path). 'split': pool of size 2B; reads from "
        "slots [0..B), writes to slots [B..2B) (speculative-decoding shape). "
        "'split'/'indexed' require the pretranspose layout.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        required=False,
        default=None,
        help="Tokens per request for gated_delta_rule_mtp (must be >= 2, "
        "default 2) and the ucache routines (4 or 8, default 4). Not "
        "applicable to other GDN routines.",
    )
    parser.add_argument(
        "--s_qo",
        type=int,
        required=False,
        default=2048,
        help="Per-sequence length for the chunked prefill routines (mean "
        "length when --seqlen_dist ragged).",
    )
    parser.add_argument(
        "--update_state",
        action="store_true",
        default=False,
        help="MTP (float32 state): write the final state back "
        "(disable_state_update=False). The BF16 state path always updates "
        "state in-place unless --output_only.",
    )
    parser.add_argument(
        "--output_only",
        action="store_true",
        default=False,
        help="MTP: frozen-state verify (disable_state_update=True) for every "
        "backend, including the BF16 state path. Required by cute-dsl-wy.",
    )
    parser.add_argument(
        "--cache_intermediate_states",
        action="store_true",
        default=False,
        help="MTP: cache per-token intermediate states (buffer dtype = "
        "state dtype; not supported by triton / cute-dsl-wy).",
    )
    parser.add_argument(
        "--use_ssm_state_indices",
        action="store_true",
        default=False,
        help="MTP (float32 state, flashinfer/cute-dsl): pass per-token "
        "ssm_state_indices [B, T] so each h_{t+1} is scattered to its own "
        "pool slot (FLA-style spec decode). Pool grows to B*(T+1). Requires "
        "--update_state.",
    )
    parser.add_argument(
        "--strided_qkv",
        action="store_true",
        default=False,
        help="Decode/MTP: q/k/v are non-contiguous views of one packed "
        "[B, T, (Hq+Hk+Hv)*D] buffer, the way vLLM/SGLang hand them over.",
    )
    parser.add_argument(
        "--no_qk_l2norm",
        action="store_true",
        default=False,
        help="Decode/MTP: disable in-kernel Q/K L2 normalization.",
    )
    parser.add_argument(
        "--cache_replayssm",
        action="store_true",
        default=False,
        help="gated_delta_rule_mtp (float32 state, --output_only, flashinfer / "
        "cute-dsl): also persist the ReplaySSM raw input window "
        "(cache_replayssm=True; SM10x, T>=3) for a later "
        "gated_delta_rule_replayssm_commit.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="gated_delta_rule_replayssm_commit: layers folded by the one "
        "all-layer commit call.",
    )
    # ---- prefill flags ----
    parser.add_argument(
        "--initial_state",
        action="store_true",
        default=False,
        help="Prefill: start from a random initial state (dtype "
        "--state_dtype) instead of the zero state.",
    )
    parser.add_argument(
        "--seqlen_dist",
        type=str,
        default="uniform",
        choices=["uniform", "ragged"],
        help="Prefill: 'uniform' gives every sequence --s_qo tokens; 'ragged' "
        "spreads lengths linearly over [0.5, 1.5] * s_qo (deterministic). "
        "max_seqlen is always passed.",
    )
    parser.add_argument(
        "--use_state_indices",
        action="store_true",
        default=False,
        help="chunk_gated_delta_rule: read/write the state through "
        "state_indices into a 2*batch_size pool (odd slots), in place "
        "(output_state is initial_state). Implies --initial_state.",
    )
    parser.add_argument(
        "--checkpoint_every_n_tokens",
        type=int,
        default=0,
        help="chunk_gated_delta_rule: store an intermediate state every N "
        "tokens (multiple of 64); checkpoints use --state_dtype.",
    )
    parser.add_argument(
        "--qk_l2norm_in_kernel",
        action="store_true",
        default=False,
        help="Prefill: pass raw q/k with use_qk_l2norm_in_kernel=True "
        "(default: k pre-normalized on the host, flag False).",
    )
    parser.add_argument(
        "--num_householder",
        type=int,
        default=2,
        help="chunk_gated_delta_product: Householder updates per token.",
    )
    # ---- ucache flags ----
    parser.add_argument(
        "--hist_len",
        type=int,
        default=8,
        help="ucache routines: filled history-ring slots P per request "
        "(verify: P + T <= 16; flush: rows with P >= 13 fold).",
    )
    parser.add_argument(
        "--flush_rate",
        type=int,
        default=20,
        help="gated_delta_rule_mtp_ucache_flush: percent of requests at "
        "P=13 (fold) rather than --hist_len (verify).",
    )
    # ---- fused decode step flags ----
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=None,
        help="gdn_fused_decode_step: model hidden size (w_ba is "
        "[hidden, 2*num_v_heads]); default 5120.",
    )
    parser.add_argument(
        "--conv_layout",
        type=str,
        default="SD",
        choices=["SD", "DS"],
        help="gdn_fused_decode_step: physical conv-state pool layout.",
    )
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=None,
        choices=_ALL_BACKENDS,
        help="Kernel backends to benchmark. Valid per routine: "
        + "; ".join(f"{r}: {', '.join(b)}" for r, b in ROUTINE_BACKENDS.items())
        + ". Default: the routine's first entry.",
    )

    args = parser.parse_args(line)

    routine = args.routine
    valid = ROUTINE_BACKENDS[routine]
    if args.backends is None:
        args.backends = [valid[0]]
    bad = [b for b in args.backends if b not in valid]
    if bad:
        raise ValueError(
            f"backend(s) {bad} are not valid for {routine}; valid: {valid}"
        )

    is_decode = routine == "gated_delta_rule_decode"
    is_mtp = routine == "gated_delta_rule_mtp"
    is_prefill = routine == "chunk_gated_delta_rule"
    is_prefill_family = routine in (
        "chunk_gated_delta_rule",
        "chunk_gated_delta_rule2",
        "chunk_gated_delta_product",
    )
    is_ucache = routine.startswith("gated_delta_rule_mtp_ucache")

    # A flag a routine does not consume would otherwise be silently ignored
    # (and still echoed into the CSV); reject it instead.
    for dest, routines in _ROUTINE_FLAGS.items():
        if routine not in routines and getattr(args, dest) != parser.get_default(dest):
            raise ValueError(f"--{dest} does not apply to {routine}")
    # hidden_size is a shared ("general") CSV column that GDN rows on main left
    # empty; keep it empty unless the routine uses it.
    if routine == "gdn_fused_decode_step":
        if args.hidden_size is None:
            args.hidden_size = 5120
    else:
        args.hidden_size = ""

    # Resolve seq_len per routine
    if is_mtp:
        if args.seq_len is None:
            args.seq_len = 2
        if args.seq_len < 2:
            raise ValueError(
                f"gated_delta_rule_mtp requires --seq_len >= 2, got {args.seq_len}"
            )
    elif is_ucache:
        if args.seq_len is None:
            args.seq_len = 4
        if args.seq_len not in (4, 8):
            raise ValueError(f"{routine} supports --seq_len 4 or 8")
    elif routine in _COMMIT:
        if args.seq_len is None:
            args.seq_len = 8
        if args.seq_len < 1:
            raise ValueError(f"{routine} needs --seq_len >= 1 (the window T)")
        if "tcgen05" in args.backends and (
            args.no_qk_l2norm or not 4 <= args.seq_len <= 8
        ):
            raise ValueError("tcgen05 commit needs normalized K and 4 <= T <= 8")
    elif args.seq_len is not None and args.seq_len != 1:
        raise ValueError(
            f"--seq_len is only applicable to gated_delta_rule_mtp / ucache, "
            f"got --seq_len {args.seq_len} for {routine}"
        )

    prefill_only = [
        ("--initial_state", args.initial_state),
        ("--seqlen_dist ragged", args.seqlen_dist != "uniform"),
        ("--qk_l2norm_in_kernel", args.qk_l2norm_in_kernel),
    ]
    if not is_prefill_family:
        for flag, set_ in prefill_only:
            if set_:
                raise ValueError(f"{flag} only applies to the prefill routines")
    if not is_prefill:
        if args.use_state_indices or args.checkpoint_every_n_tokens:
            raise ValueError(
                "--use_state_indices / --checkpoint_every_n_tokens only apply "
                "to chunk_gated_delta_rule"
            )
    if not is_mtp and (
        args.output_only or args.use_ssm_state_indices or args.update_state
    ):
        raise ValueError(
            "--output_only / --use_ssm_state_indices / --update_state only "
            "apply to gated_delta_rule_mtp"
        )

    if args.strided_qkv and not (is_decode or is_mtp):
        raise ValueError("--strided_qkv only applies to decode/MTP")
    if is_decode or is_mtp:
        if args.state_dtype not in ("float32", "bfloat16"):
            raise ValueError("decode/MTP support --state_dtype float32 or bfloat16")
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
        if set(args.backends) & {"rubin", "rubin_ugpu"} and (
            args.state_dtype != "bfloat16" or args.head_size != 128
        ):
            raise ValueError(
                "The rubin / rubin_ugpu backends are only available for "
                "gated_delta_rule_decode with --state_dtype bfloat16 and "
                "head_size 128"
            )
        if "cake_gdn" in args.backends and args.input_dtype != "bfloat16":
            raise ValueError("The cake_gdn backend requires --input_dtype bfloat16")
        if args.state_layout == "nontranspose":
            if args.pool_mode != "single":
                raise ValueError(
                    "--pool_mode split/indexed requires the pretranspose layout"
                )
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
        if (
            is_mtp
            and args.state_dtype == "bfloat16"
            and args.cache_intermediate_states
            and set(args.backends) & {"triton", "cute-dsl-wy"}
        ):
            raise ValueError(
                "--cache_intermediate_states with bfloat16 state is not "
                "supported by triton / cute-dsl-wy"
            )
        if is_decode and (args.update_state or args.cache_intermediate_states):
            raise ValueError(
                "--update_state / --cache_intermediate_states are only "
                "applicable to gated_delta_rule_mtp"
            )
        if is_mtp:
            if args.output_only and args.update_state:
                raise ValueError("--output_only and --update_state are exclusive")
            if "cute-dsl-wy" in args.backends:
                if args.state_dtype != "bfloat16" or not args.output_only:
                    raise ValueError(
                        "cute-dsl-wy is the output-only BF16-state MTP kernel: "
                        "requires --state_dtype bfloat16 --output_only"
                    )
                if args.pool_mode == "split" or args.no_qk_l2norm:
                    raise ValueError(
                        "cute-dsl-wy supports neither split pools nor --no_qk_l2norm"
                    )
                if args.seq_len not in (4, 8):
                    # The WY kernel pads any other T to 16 tokens
                    # (gdn_decode_bf16_wy_output_only.py:2149), so the row
                    # would time 16-token work under a smaller T.
                    raise ValueError("cute-dsl-wy requires --seq_len 4 or 8")
            if args.cache_replayssm:
                if (
                    args.state_dtype != "float32"
                    or not args.output_only
                    or args.head_size != 128
                    or args.seq_len < 3
                    or args.cache_intermediate_states
                    or args.use_ssm_state_indices
                    or args.pool_mode != "single"
                    or set(args.backends) - {"flashinfer", "cute-dsl"}
                ):
                    raise ValueError(
                        "--cache_replayssm needs --state_dtype float32 "
                        "--output_only, head_size 128, --seq_len >= 3, a single "
                        "pool, no intermediate/ssm-index caching and the "
                        "flashinfer / cute-dsl backends"
                    )
            if args.use_ssm_state_indices:
                if args.state_dtype != "float32" or not args.update_state:
                    raise ValueError(
                        "--use_ssm_state_indices requires --state_dtype "
                        "float32 and --update_state"
                    )
                if args.cache_intermediate_states or args.pool_mode != "single":
                    raise ValueError(
                        "--use_ssm_state_indices excludes "
                        "--cache_intermediate_states and split pools"
                    )
                if set(args.backends) - {"flashinfer", "cute-dsl"}:
                    raise ValueError(
                        "--use_ssm_state_indices is only exposed by "
                        "gated_delta_rule_mtp (flashinfer / cute-dsl backends)"
                    )

    if is_prefill:
        if args.use_state_indices:
            args.initial_state = True
            if "fla" in args.backends:
                raise ValueError("fla does not take state_indices")
        if args.checkpoint_every_n_tokens % 64 != 0:
            raise ValueError("--checkpoint_every_n_tokens must be a multiple of 64")
        if "fla" in args.backends and args.checkpoint_every_n_tokens:
            raise ValueError("fla does not take checkpoints")
    if is_prefill_family and routine != "chunk_gated_delta_rule":
        if args.state_dtype not in ("float32", "bfloat16"):
            raise ValueError(f"{routine} (cuDNN) takes float32 or bfloat16 state only")

    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


# ==============================================================================
# Shared helpers
# ==============================================================================


def _gdn_flags(args, **extra):
    """Compact string of the non-default knobs that change the measured work."""
    parts = []
    for name in (
        "strided_qkv",
        "initial_state",
        "use_state_indices",
        "qk_l2norm_in_kernel",
        "output_only",
        "use_ssm_state_indices",
        "cache_intermediate_states",
        "cache_replayssm",
    ):
        if getattr(args, name, False):
            parts.append(name)
    if getattr(args, "seqlen_dist", "uniform") != "uniform":
        parts.append(f"seqlen_dist={args.seqlen_dist}")
    if getattr(args, "checkpoint_every_n_tokens", 0):
        parts.append(f"ckpt={args.checkpoint_every_n_tokens}")
    for k, v in extra.items():
        # total_tokens is batch_size * s_qo unless lengths are ragged; keep it
        # out of the flags then, so the flags only name non-default modes.
        if k == "total_tokens" and getattr(args, "seqlen_dist", "uniform") == "uniform":
            continue
        parts.append(f"{k}={v}")
    return ";".join(parts)


def _check_and_report(args, tested, reference, rtol, atol, mismatch_threshold_pct=0.01):
    """Elementwise refcheck used by every routine (same policy as before)."""
    for name, out in tested.items():
        (
            num_different_elements,
            num_elements,
            num_different_elements_percentage,
        ) = is_close_stats(reference.float(), out.float(), rtol=rtol, atol=atol)
        if num_different_elements_percentage > mismatch_threshold_pct:
            print(
                f"[ERROR] Output tensor mismatch from backend {name}: "
                f"{num_different_elements}/{num_elements} ({num_different_elements_percentage:.4f}%) elements differ "
                f"(threshold: {mismatch_threshold_pct}%)"
            )
            if not args.allow_output_mismatch:
                raise AssertionError(
                    f"[ERROR] Backend {name} output mismatch with {num_different_elements} elements"
                )
        elif args.verbose >= 1:
            print(
                f"[REFCHECK] Backend {name}: PASSED "
                f"({num_different_elements}/{num_elements} elements differ "
                f"({num_different_elements_percentage:.4f}%), within {mismatch_threshold_pct}% threshold)"
            )


def _check_rel_err(args, tested, reference, tol):
    """Relative-Frobenius refcheck (the cuDNN linear-attention test policy)."""
    ref = reference.float()
    for name, out in tested.items():
        err = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()
        if not math.isfinite(err) or err > tol:
            print(
                f"[ERROR] Output tensor mismatch from backend {name}: "
                f"relative error {err:.3e} > {tol:.1e}"
            )
            if not args.allow_output_mismatch:
                raise AssertionError(f"[ERROR] Backend {name} relative error {err:.3e}")
        elif args.verbose >= 1:
            print(
                f"[REFCHECK] Backend {name}: PASSED (relative error {err:.3e} "
                f"<= {tol:.1e})"
            )


def _time_backends(args, backends, run_backend, extra_args=(), cuda_graph=None):
    use_graph = (not args.no_cuda_graph) if cuda_graph is None else cuda_graph
    times = {}
    for cur_backend in backends:
        times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=use_graph,
            input_args=(cur_backend, *extra_args),
        )
    return times


def _emit_rows(args, backends, times, flops, bytes_, fields, kernels):
    res = []
    for backend in backends:
        if len(times[backend]) == 0:
            continue
        median_time = np.median(times[backend])
        std_time = np.std(times[backend])
        tflops = flops / (10**9 * median_time)
        tb_per_sec = bytes_ / (10**9 * median_time)
        print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)
        if args.verbose >= 1:
            print(f"[INFO] {backend}: gdn_kernel={kernels.get(backend, backend)}")
        if args.output_path is not None:
            cur_res = defaultdict(str)
            cur_res["routine"] = args.routine
            cur_res["median_time"] = median_time
            cur_res["std_time"] = std_time
            cur_res["tflops"] = tflops
            cur_res["tb_per_sec"] = tb_per_sec
            cur_res["backend"] = backend
            cur_res["gdn_kernel"] = kernels.get(backend, backend)
            for k, v in fields.items():
                cur_res[k] = v
            cur_res["case_tag"] = args.case_tag
            res.append(cur_res)
    return res


def _prepare(args):
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print(f"[INFO] Running {args.routine}")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )
    backends = filter_backends_by_compute_capability(
        args.backends[:], args.routine, device
    )
    return device, backends


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
    per_token_state_writes: bool = False,
) -> int:
    """Memory bytes for gated delta rule decode/MTP.

    Counts q/k/v/output traffic, the state read (+ optional write-back),
    GDN parameters, and the optional MTP intermediate state writes. With
    ``per_token_state_writes`` (MTP ``ssm_state_indices``) the kernel scatters
    every h_{t+1} to its own slot and skips the final write-back
    (gdn_decode.py:1338-1345): one state read plus ``seq_len`` writes.
    """
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads
    elem_size = input_dtype.itemsize

    q_bytes = batch_size * seq_len * num_q_heads * head_size * elem_size
    k_bytes = batch_size * seq_len * num_k_heads * head_size * elem_size
    v_bytes = batch_size * seq_len * num_v_heads * head_size * elem_size
    o_bytes = batch_size * seq_len * num_o_heads * head_size * elem_size

    state_elems = batch_size * num_sab_heads * head_size * head_size
    if per_token_state_writes:
        state_bytes = state_elems * state_dtype_bytes * (1 + seq_len)
    else:
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
    state_dtype_bytes: int = 4,
    read_initial_state: bool = False,
    gate_bytes_per_token_head: int = 8,
    kv_rows: int = None,
    num_checkpoints: int = 0,
) -> int:
    """Memory bytes for chunked GDN prefill (q/k/v/gate reads, output and
    final-state writes, optional initial-state read, and ``num_checkpoints``
    intermediate-state writes).

    ``gate_bytes_per_token_head`` defaults to g + beta at fp32; the GDN-2 /
    GDP routines pass their own gate footprint. ``kv_rows`` (GDP) is the
    expanded sub-token row count for k/v.
    """
    num_sab_heads = max(num_q_heads, num_v_heads)
    elem_size = input_dtype.itemsize
    kv_rows = total_tokens if kv_rows is None else kv_rows

    qkv_bytes = (
        (total_tokens * num_q_heads + kv_rows * (num_k_heads + num_v_heads))
        * head_size
        * elem_size
    )
    gate_bytes = total_tokens * num_sab_heads * gate_bytes_per_token_head
    o_bytes = total_tokens * num_sab_heads * head_size * elem_size
    state_bytes = (
        num_seqs
        * num_sab_heads
        * head_size
        * head_size
        * state_dtype_bytes
        * (2 if read_initial_state else 1)
    )
    ckpt_bytes = (
        num_checkpoints * num_sab_heads * head_size * head_size * state_dtype_bytes
    )
    return qkv_bytes + gate_bytes + o_bytes + state_bytes + ckpt_bytes


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
       - 'flashinfer': public gdn_decode APIs; backend="flashinfer" for T=1
                       pretranspose (origin/main meaning), else "auto"
       - 'cute-dsl':   same APIs with backend="flashinfer" (CuTe-DSL only)
       - 'rubin' / 'rubin_ugpu': backend="rubin" / a ugpu_state_pool (SM107)
       - 'cute-dsl-wy': output-only WY BF16-state MTP kernel
       - 'triton':     Triton reference kernels (benchmarks/gdn_triton_reference.py)
    3. Optionally checks outputs against the torch reference
       (tests/gdn/reference_delta_rule.py::decode_delta_rule)
    4. Measures performance (memory bandwidth is the primary metric)

    Args:
        args: Parsed command line arguments containing test configuration
        seq_len: Tokens per request (1 = decode, >1 = MTP)

    Returns:
        dict: List of dictionaries containing performance results
    """
    device, backends = _prepare(args)

    ## Parse input arguments
    batch_size = args.batch_size
    num_q_heads = args.num_q_heads
    num_k_heads = args.num_k_heads
    num_v_heads = args.num_v_heads
    head_size = args.head_size
    state_layout = args.state_layout if seq_len == 1 else "pretranspose"
    pool_split = args.pool_mode == "split"
    pool_indexed = args.pool_mode == "indexed"
    use_qk_l2norm = not args.no_qk_l2norm
    is_mtp = seq_len > 1
    run_refcheck = args.refcheck
    res = []

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
    if args.strided_qkv:
        packed = torch.randn(
            batch_size,
            T,
            (num_q_heads + num_k_heads + num_v_heads) * head_size,
            dtype=input_dtype,
            device=device,
        )
        q, k, v = (
            t.unflatten(-1, (-1, head_size))
            for t in packed.split(
                [
                    num_q_heads * head_size,
                    num_k_heads * head_size,
                    num_v_heads * head_size,
                ],
                dim=-1,
            )
        )
    else:
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
    use_ssm_idx = is_mtp and args.use_ssm_state_indices
    if use_ssm_idx:
        pool_size = batch_size * (T + 1)
    else:
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
    ssm_state_indices = (
        torch.arange(
            batch_size, batch_size * (T + 1), dtype=torch.int32, device=device
        ).view(batch_size, T)
        if use_ssm_idx
        else None
    )

    # Intermediate states buffer (MTP only; dtype follows the state)
    intermediate_states_buffer = None
    if is_mtp and args.cache_intermediate_states:
        intermediate_states_buffer = torch.zeros(
            batch_size,
            T,
            num_sab_heads,
            head_size,
            head_size,
            dtype=state_dtype,
            device=device,
        )

    output = torch.empty(
        batch_size, T, num_o_heads, head_size, dtype=input_dtype, device=device
    )

    # ReplaySSM raw input windows, indexed by pool slot (gated_delta_rule_mtp
    # cache_replayssm=True); overwritten on every verify.
    replayssm_kwargs = {}
    replayssm_bytes = 0
    if is_mtp and args.cache_replayssm:
        replayssm_kwargs = dict(
            cache_replayssm=True,
            replayssm_rawv=torch.empty(
                pool_size,
                num_sab_heads,
                T,
                head_size,
                dtype=torch.bfloat16,
                device=device,
            ),
            replayssm_rawk=torch.empty(
                pool_size,
                num_k_heads,
                T,
                head_size,
                dtype=torch.bfloat16,
                device=device,
            ),
            replayssm_g=torch.empty(
                pool_size, num_sab_heads, T, dtype=torch.float32, device=device
            ),
            replayssm_beta=torch.empty(
                pool_size, num_sab_heads, T, dtype=torch.float32, device=device
            ),
        )
        replayssm_bytes = (
            batch_size
            * T
            * ((num_sab_heads + num_k_heads) * head_size * 2 + 2 * num_sab_heads * 4)
        )

    # State write-back. Decode (T=1) always writes back. MTP: the fp32 API
    # makes it optional (--update_state), the BF16 pool path always writes
    # back unless --output_only.
    if not is_mtp:
        state_writeback = True
    elif args.output_only:
        state_writeback = False
    elif use_bf16_state:
        state_writeback = True
    else:
        state_writeback = args.update_state
    disable_state_update = not state_writeback

    # The pretranspose "pool" call path (initial_state + indices) is used for
    # bf16 state, split/indexed pools, all MTP calls routed through the
    # pretranspose API, and the rubin backends.
    decode_pool_path = pool_split or pool_indexed or use_bf16_state

    # uGPU localization: a copy of the pool in the [pool, HV, 2, V/2, K] layout
    # whose halves live in the two dies' memory (flashinfer.gdn_decode, #6604)
    ugpu_pool = None
    if "rubin_ugpu" in backends:
        from flashinfer.gdn_decode import ugpu_state_pool

        try:
            ugpu_pool = ugpu_state_pool(pool_size, num_sab_heads, head_size, device)
        except Exception as e:  # noqa: BLE001
            # e.g. a GPU (or driver) exposing a single locality domain; keep the
            # other backends' rows.
            print(f"[ERROR] rubin_ugpu: ugpu_state_pool failed: {e}")
            backends.remove("rubin_ugpu")
        else:

            def to_ugpu(src):
                ugpu_pool.copy_(src.view(pool_size, num_sab_heads, 2, -1, head_size))

            to_ugpu(state_pool)

    if args.verbose >= 2:
        print(f"[VVERBOSE] Mode: {'MTP' if is_mtp else 'decode'} (T={T})")
        print(f"[VVERBOSE] {q.shape = }, {q.dtype = }")
        print(f"[VVERBOSE] {v.shape = }, {v.dtype = }")
        print(f"[VVERBOSE] {state_pool.shape = }, {state_pool.dtype = }")
        print(f"[VVERBOSE] {state_layout = }, {args.pool_mode = }")
        print(f"[VVERBOSE] {use_qk_l2norm = }, {state_writeback = }")

    wy_mtp = None
    if "cute-dsl-wy" in backends:
        from flashinfer.gdn_kernels import gated_delta_rule_mtp_wy_output_only

        if gated_delta_rule_mtp_wy_output_only is None:
            raise RuntimeError(
                "cute-dsl-wy: gdn_decode_bf16_wy_output_only failed to import"
            )
        wy_mtp = gated_delta_rule_mtp_wy_output_only

    def _pretranspose_pool_call(state, lib_backend, out):
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
            out,
            use_qk_l2norm,
            initial_state=state,
            initial_state_indices=initial_state_indices,
            output_state_indices=output_state_indices,
            intermediate_states_buffer=intermediate_states_buffer if is_mtp else None,
            disable_state_update=disable_state_update if is_mtp else False,
            backend=lib_backend,
        )[0]

    def run_backend(backend, state, out=output):
        if backend == "rubin_ugpu":
            # refcheck passes a fresh copy of the pool, timing passes
            # state_pool itself (the state then evolves in place)
            if state is not state_pool:
                to_ugpu(state)
            return _pretranspose_pool_call(ugpu_pool, "auto", out)
        if backend in _DECODE_DISPATCH:
            lib_backend = _DECODE_DISPATCH[backend]
            if backend == "rubin":
                return _pretranspose_pool_call(state, lib_backend, out)
            if is_mtp:
                # fp32 flashinfer/cute-dsl keep the dedicated MTP API (the
                # only path exposing ssm_state_indices); everything else goes
                # through the pretranspose pool path, which is where the
                # backend selector lives.
                if not use_bf16_state and backend in ("flashinfer", "cute-dsl"):
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
                        out,
                        intermediate_states_buffer=intermediate_states_buffer,
                        ssm_state_indices=ssm_state_indices,
                        disable_state_update=disable_state_update,
                        use_qk_l2norm=use_qk_l2norm,
                        output_state_indices=output_state_indices,
                        **replayssm_kwargs,
                    )[0]
                return _pretranspose_pool_call(state, lib_backend, out)
            if state_layout == "nontranspose":
                return gated_delta_rule_decode(
                    q,
                    k,
                    v,
                    state,
                    A_log,
                    a,
                    dt_bias,
                    b,
                    scale,
                    out,
                    use_qk_l2norm,
                    backend=lib_backend,
                )[0]
            # BF16 state always goes through the pool path with pre-allocated
            # indices: the non-pool API path synthesizes an arange index
            # tensor per call, whose tiny kernel would pollute CUPTI timing.
            # Cake GDN serves indexed pools only.
            if decode_pool_path or backend == "cake_gdn":
                return _pretranspose_pool_call(state, lib_backend, out)
            return gated_delta_rule_decode_pretranspose(
                q,
                k,
                v,
                state,
                A_log,
                a,
                dt_bias,
                b,
                scale,
                out,
                use_qk_l2norm,
                backend=lib_backend,
            )[0]
        elif backend == "cute-dsl-wy":
            return wy_mtp(
                A_log=A_log,
                a=a,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                b=b,
                initial_state_source=state,
                initial_state_indices=initial_state_indices,
                disable_state_update=True,
                use_qk_l2norm_in_kernel=use_qk_l2norm,
                scale=scale,
                output=out,
            )
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
                    out,
                    intermediate_states_buffer=intermediate_states_buffer,
                    disable_state_update=disable_state_update,
                    use_qk_l2norm=use_qk_l2norm,
                )[0]
            if state_layout == "nontranspose":
                return triton_gdn_decode(
                    q, k, v, state, A_log, a, dt_bias, b, scale, out, use_qk_l2norm
                )[0]
            return triton_gdn_decode_pretranspose(
                q, k, v, state, A_log, a, dt_bias, b, scale, out, use_qk_l2norm
            )[0]
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Kernel label per backend. 'flashinfer' is library auto (which may pick a
    # Cake or Rubin kernel the harness does not identify).
    clean_state_snapshot = state_pool.clone()
    kernels = {
        b_: {
            "cute-dsl-wy": "cute-dsl-wy",
            "triton": "triton",
            "rubin_ugpu": "rubin-ugpu",
        }.get(b_, b_)
        for b_ in backends
    }
    if "flashinfer" in backends:
        kernels["flashinfer"] = "auto"
    if "cute" in backends:
        kernels["cute"] = "cute-dsl"

    # Reference check against the torch reference, which uses the K-major
    # state layout [B, HV, K, V] and processes one token at a time.
    # bench_gpu_time mutates state_pool in-place, so all refcheck runs use
    # clones of the clean snapshot taken before any benchmarking.
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

        for cur_backend in backends:
            fresh_state = clean_state_snapshot.clone()
            outputs[cur_backend] = (
                run_backend(cur_backend, fresh_state, torch.empty_like(output))
                .detach()
                .clone()
            )

    times = _time_backends(args, backends, run_backend, extra_args=(state_pool,))

    if run_refcheck and outputs:
        # bf16 state rounds the recurrent state every step in the reference
        # but is kept in higher precision inside the kernels, so use looser
        # tolerances there.
        rtol, atol = (2e-2, 2e-2) if use_bf16_state else (5e-3, 5e-3)
        _check_and_report(args, outputs, reference_output, rtol, atol)

    # Compute and report performance metrics
    problem_flops = gdn_decode_flops(batch_size, num_q_heads, num_v_heads, head_size, T)
    problem_bytes = (
        gdn_decode_bytes(
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
            per_token_state_writes=use_ssm_idx,
        )
        + replayssm_bytes
    )
    fields = {
        "batch_size": batch_size,
        "num_q_heads": num_q_heads,
        "num_k_heads": num_k_heads,
        "num_v_heads": num_v_heads,
        "head_size": head_size,
        "seq_len": T,
        "input_dtype": str(input_dtype),
        "state_dtype": str(state_dtype),
        "state_layout": state_layout,
        "pool_mode": args.pool_mode,
        "update_state": state_writeback,
        "use_qk_l2norm": use_qk_l2norm,
        "gdn_flags": _gdn_flags(args),
    }
    res += _emit_rows(
        args, backends, times, problem_flops, problem_bytes, fields, kernels
    )
    return res


# ==============================================================================
# Prefill
# ==============================================================================


def _prefill_seq_lens(num_seqs, s_qo, dist):
    if dist == "uniform" or num_seqs == 1:
        return [s_qo] * num_seqs
    # Deterministic ragged spread over [0.5, 1.5] * s_qo.
    return [
        max(1, int(round(s_qo * (0.5 + i / (num_seqs - 1))))) for i in range(num_seqs)
    ]


def testChunkGatedDeltaRule(args):
    """
    Test chunk_gated_delta_rule (varlen GDN prefill).

    This test:
    1. Generates varlen q/k/v (uniform or ragged per-sequence lengths),
       forget gate alpha and update gate beta. By default k is
       pre-L2-normalized and the kernel is called with
       use_qk_l2norm_in_kernel=False so that the torch reference (which
       performs no normalization) sees identical inputs; with
       --qk_l2norm_in_kernel the kernel normalizes and the reference is fed
       normalized q/k.
    2. Runs the requested backend(s) (see _PREFILL_DISPATCH):
       - 'flashinfer'   backend="auto", use_cp="auto"
       - 'cute-dsl'     backend="flashinfer", use_cp=False
       - 'cute-dsl-cp'  backend="flashinfer", use_cp=True
       - 'cudnn'        backend="cudnn"
       - 'fla'          flash-linear-attention Triton baseline (perf-only)
    3. Optionally checks every non-fla output against
       tests/gdn/reference_delta_rule.py::blockwise_delta_rule
    4. Measures performance metrics
    """
    device, backends = _prepare(args)

    ## Parse input arguments
    num_seqs = args.batch_size
    s_qo = args.s_qo
    num_q_heads = args.num_q_heads
    num_k_heads = args.num_k_heads
    num_v_heads = args.num_v_heads
    head_size = args.head_size
    run_refcheck = args.refcheck
    res = []

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
    state_dtype = _torch_dtype(args.state_dtype)
    ## Done parsing input arguments

    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads
    seq_lens = _prefill_seq_lens(num_seqs, s_qo, args.seqlen_dist)
    total_tokens = sum(seq_lens)
    max_seqlen = max(seq_lens)
    offsets = [0]
    for n in seq_lens:
        offsets.append(offsets[-1] + n)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int64, device=device)

    ## Prepare input tensors
    q = torch.randn(
        total_tokens, num_q_heads, head_size, dtype=input_dtype, device=device
    )
    k_raw = torch.randn(
        total_tokens, num_k_heads, head_size, dtype=torch.float32, device=device
    )
    if args.qk_l2norm_in_kernel:
        k = k_raw.to(input_dtype)
    else:
        # Pre-normalize k for numerical stability (matches
        # bench_gdn_prefill.py); the kernel and the reference then see
        # identical inputs.
        k = F.normalize(k_raw, p=2.0, dim=-1).to(input_dtype)
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
    state_shape = (num_sab_heads, head_size, head_size)
    state_indices = None
    initial_state = None
    if args.use_state_indices:
        # 2x pool, sequence i lives in slot 2*i+1; read and write in place.
        pool = (0.1 * torch.randn(2 * num_seqs, *state_shape, device=device)).to(
            state_dtype
        )
        state_indices = torch.arange(num_seqs, device=device, dtype=torch.int32) * 2 + 1
        initial_state = pool
        output_state = pool
    else:
        if args.initial_state:
            initial_state = (
                0.1 * torch.randn(num_seqs, *state_shape, device=device)
            ).to(state_dtype)
        output_state = torch.empty(
            num_seqs, *state_shape, dtype=state_dtype, device=device
        )
    state_checkpoints = None
    checkpoint_cu_starts = None
    ckpt_n = args.checkpoint_every_n_tokens
    if ckpt_n:
        counts = [n // ckpt_n for n in seq_lens]
        cks = [0]
        for c in counts:
            cks.append(cks[-1] + c)
        checkpoint_cu_starts = torch.tensor(cks, dtype=torch.int32, device=device)
        state_checkpoints = torch.empty(
            max(cks[-1], 1), *state_shape, dtype=state_dtype, device=device
        )[: cks[-1]]

    # Snapshot of the initial states in sequence order (the in-place pool is
    # overwritten by every benchmarked call).
    initial_by_seq = None
    if initial_state is not None:
        initial_by_seq = (
            initial_state[state_indices.long()]
            if state_indices is not None
            else initial_state
        ).clone()
    initial_state_clean = initial_state.clone() if initial_state is not None else None

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
        h0_fla = (
            initial_by_seq.float()
            if initial_by_seq is not None
            else torch.zeros(
                num_seqs,
                num_sab_heads,
                head_size,
                head_size,
                dtype=torch.float32,
                device=device,
            )
        )
        cu_seqlens_fla = cu_seqlens.to(torch.int32)
    if args.verbose >= 2:
        print(f"[VVERBOSE] {q.shape = }, {q.dtype = }")
        print(f"[VVERBOSE] {v.shape = }, {v.dtype = }")
        print(f"[VVERBOSE] {seq_lens = }")
        print(f"[VVERBOSE] {g.shape = }, {beta.shape = }")
        print(f"[VVERBOSE] {state_dtype = }, initial_state={initial_state is not None}")

    def run_backend(backend, out=output, init=initial_state, out_state=output_state):
        if backend in _PREFILL_DISPATCH:
            lib_backend, use_cp = _PREFILL_DISPATCH[backend]
            return chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                scale,
                init,
                True,  # output_final_state
                cu_seqlens,
                args.qk_l2norm_in_kernel,
                output=out,
                output_state=out_state,
                state_checkpoints=state_checkpoints,
                checkpoint_cu_starts=checkpoint_cu_starts,
                checkpoint_every_n_tokens=ckpt_n,
                use_cp=use_cp,
                state_indices=state_indices,
                backend=lib_backend,
                max_seqlen=max_seqlen,
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

    def fresh_state_args():
        if initial_state is None:
            return {"init": None, "out_state": torch.empty_like(output_state)}
        init = initial_state_clean.clone()
        return {
            "init": init,
            "out_state": init
            if state_indices is not None
            else torch.empty_like(output_state),
        }

    # Which kernel does 'flashinfer' (auto) run? Auto takes the CuTe-DSL CP
    # path when the low-parallelism heuristic matches and CP dispatch is
    # available (else it warns and runs non-CP). cuDNN is only auto-picked for
    # raw-gate calls (use_gate_in_kernel), which this routine does not make.
    kernels = {b_: b_ for b_ in backends}
    if "flashinfer" in backends:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_backend("flashinfer", torch.empty_like(output), **fresh_state_args())
            torch.cuda.synchronize()
        from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import (
            should_use_cp_host,
        )
        from flashinfer.utils import (
            get_compute_capability,
            get_device_name,
            get_device_sm_count,
        )

        cc = get_compute_capability(device)
        # SM8x has its own CP rule on the exact longest sequence
        # (gdn_prefill.py should_use_cp_sm80_host); not reproduced here.
        heuristic = cc[0] in (9, 10, 12) and should_use_cp_host(
            num_seqs * num_sab_heads,
            get_device_sm_count(device),
            get_device_name(device),
            device_capability=cc,
        )
        fell_back = any("CP dispatch is unavailable" in str(w.message) for w in caught)
        kernels["flashinfer"] = (
            "cute-dsl-cp" if heuristic and not fell_back else "cute-dsl"
        )
        print(f"[INFO] flashinfer (auto) dispatches to: {kernels['flashinfer']}")

    # Reference check (all but fla; fla uses a different gate
    # parameterization and is benchmarked perf-only)
    outputs = {}
    if run_refcheck:
        q_ref, k_ref = q.float(), k.float()
        if args.qk_l2norm_in_kernel:
            q_ref = F.normalize(q_ref, dim=-1)
            k_ref = F.normalize(k_ref, dim=-1)
        reference_output = blockwise_delta_rule(
            q_ref,
            k_ref,
            v.float(),
            seq_lens,
            alpha=g,
            beta=beta,
            scale_factor=scale,
            state_dtype=torch.float32,
            initial_state=(
                initial_by_seq.float().transpose(-1, -2).contiguous()
                if initial_by_seq is not None
                else None
            ),
        )[0]
        for cur_backend in backends:
            if cur_backend == "fla":
                continue
            outputs[cur_backend] = (
                run_backend(cur_backend, torch.empty_like(output), **fresh_state_args())
                .detach()
                .clone()
            )

    times = _time_backends(args, backends, run_backend)

    if run_refcheck and outputs:
        _check_and_report(args, outputs, reference_output, 1e-2, 1e-2)

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
        state_dtype_bytes=state_dtype.itemsize,
        read_initial_state=initial_state is not None,
        num_checkpoints=0 if state_checkpoints is None else state_checkpoints.shape[0],
    )
    fields = {
        "batch_size": num_seqs,
        "s_qo": s_qo,
        "num_q_heads": num_q_heads,
        "num_k_heads": num_k_heads,
        "num_v_heads": num_v_heads,
        "head_size": head_size,
        "input_dtype": str(input_dtype),
        # Prefill rows have always written the --state_dtype spelling ("float32"),
        # unlike decode/MTP ("torch.float32"); keep it so existing series keep their key.
        "state_dtype": args.state_dtype,
        "gdn_flags": _gdn_flags(args, total_tokens=total_tokens),
    }
    res += _emit_rows(
        args, backends, times, problem_flops, problem_bytes, fields, kernels
    )
    return res


def _linear_attention_oracles():
    """cuDNN linear-attention fp32 serial oracles (tests/test_helpers)."""
    from tests.test_helpers.cudnn_linear_attention import (
        serial_delta_product,
        serial_delta_rule2,
    )

    return serial_delta_rule2, serial_delta_product


def _cudnn_prefill_common(args):
    device, backends = _prepare(args)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    state_dtype = _torch_dtype(args.state_dtype)
    seq_lens = _prefill_seq_lens(args.batch_size, args.s_qo, args.seqlen_dist)
    offsets = [0]
    for n in seq_lens:
        offsets.append(offsets[-1] + n)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=device)
    return device, backends, input_dtype, state_dtype, seq_lens, cu_seqlens


def testChunkGatedDeltaRule2(args):
    """
    Test chunk_gated_delta_rule2 (GDN-2: channel-wise g/beta/w gates).

    cuDNN is the only implementation (backend 'cudnn' ->
    chunk_gated_delta_rule2(backend="cudnn")). Refcheck: the fp32 token-serial
    GDN-2 oracle from tests/test_helpers/cudnn_linear_attention.py, relative
    error <= 5e-2 (the library test's SERIAL_TOLERANCE).
    """
    from flashinfer.gdn2_prefill import chunk_gated_delta_rule2

    device, backends, input_dtype, state_dtype, seq_lens, cu_seqlens = (
        _cudnn_prefill_common(args)
    )
    res = []
    if len(backends) == 0:
        return res
    H_q, H_k, H_v, D = (
        args.num_q_heads,
        args.num_k_heads,
        args.num_v_heads,
        args.head_size,
    )
    H_o = max(H_q, H_v)
    total = sum(seq_lens)

    def norm_or_raw(heads):
        x = torch.randn(total, heads, D, dtype=torch.float32, device=device)
        if not args.qk_l2norm_in_kernel:
            x = F.normalize(x, dim=-1)
        return x.to(input_dtype).contiguous()

    q = norm_or_raw(H_q)
    k = norm_or_raw(H_k)
    v = torch.randn(total, H_v, D, dtype=input_dtype, device=device)
    g = (
        -F.softplus(torch.randn(total, H_o, D, device=device) * 0.5 - 2.0)
    ).contiguous()  # ln(alpha) <= 0, fp32
    beta = torch.rand(total, H_o, D, device=device).to(input_dtype)
    w = torch.rand(total, H_o, D, device=device).to(input_dtype)
    initial_state = (
        (0.01 * torch.randn(len(seq_lens), H_o, D, D, device=device)).to(state_dtype)
        if args.initial_state
        else None
    )
    output = torch.empty(total, H_o, D, dtype=input_dtype, device=device)
    output_state = torch.empty(
        len(seq_lens), H_o, D, D, dtype=state_dtype, device=device
    )
    scale = D**-0.5

    def run_backend(backend, out=output, out_state=output_state):
        if backend != "cudnn":
            raise ValueError(f"Unsupported backend: {backend}")
        return chunk_gated_delta_rule2(
            q,
            k,
            v,
            g,
            beta,
            w,
            scale,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=args.qk_l2norm_in_kernel,
            output=out,
            output_state=out_state,
            backend="cudnn",
        )[0]

    outputs = {}
    if args.refcheck:
        serial_delta_rule2, _ = _linear_attention_oracles()
        ref_out, _ = serial_delta_rule2(
            q,
            k,
            v,
            cu_seqlens,
            alpha=g.float().exp(),
            beta=beta,
            w=w,
            initial_state=initial_state,
            scale=scale,
            l2norm=args.qk_l2norm_in_kernel,
        )
        for b_ in backends:
            outputs[b_] = run_backend(
                b_, torch.empty_like(output), torch.empty_like(output_state)
            ).clone()

    times = _time_backends(args, backends, run_backend)
    if args.refcheck and outputs:
        _check_rel_err(args, outputs, ref_out, 5e-2)

    flops = gdn_prefill_flops(total, H_o, D)
    bytes_ = gdn_prefill_bytes(
        total,
        len(seq_lens),
        H_q,
        H_k,
        H_v,
        D,
        input_dtype,
        state_dtype_bytes=state_dtype.itemsize,
        read_initial_state=initial_state is not None,
        gate_bytes_per_token_head=D * (4 + 2 * input_dtype.itemsize),
    )
    fields = {
        "batch_size": len(seq_lens),
        "s_qo": args.s_qo,
        "num_q_heads": H_q,
        "num_k_heads": H_k,
        "num_v_heads": H_v,
        "head_size": D,
        "input_dtype": str(input_dtype),
        "state_dtype": args.state_dtype,
        "gdn_flags": _gdn_flags(args, total_tokens=total),
    }
    res += _emit_rows(args, backends, times, flops, bytes_, fields, {})
    return res


def testChunkGatedDeltaProduct(args):
    """
    Test chunk_gated_delta_product (GDP: n Householder updates per token).

    cuDNN is the only implementation (backend 'cudnn' ->
    chunk_gated_delta_product(backend="cudnn")). k/v/beta live on the
    expanded sub-token timeline (total * num_householder rows). Refcheck: the
    fp32 token-serial GDP oracle, relative error <= 5e-2.
    """
    from flashinfer.gdp_prefill import chunk_gated_delta_product

    device, backends, input_dtype, state_dtype, seq_lens, cu_seqlens = (
        _cudnn_prefill_common(args)
    )
    res = []
    if len(backends) == 0:
        return res
    n = args.num_householder
    if n < 1:
        raise ValueError("--num_householder must be >= 1")
    H_q, H_k, H_v, D = (
        args.num_q_heads,
        args.num_k_heads,
        args.num_v_heads,
        args.head_size,
    )
    H_o = max(H_q, H_v)
    total = sum(seq_lens)
    expanded = total * n

    def norm_or_raw(rows, heads):
        x = torch.randn(rows, heads, D, dtype=torch.float32, device=device)
        if not args.qk_l2norm_in_kernel:
            x = F.normalize(x, dim=-1)
        return x.to(input_dtype).contiguous()

    q = norm_or_raw(total, H_q)
    k = norm_or_raw(expanded, H_k)
    v = torch.randn(expanded, H_v, D, dtype=input_dtype, device=device)
    g = torch.exp(
        -F.softplus(torch.randn(total, H_o, device=device) * 0.5 - 2.0)
    ).contiguous()  # linear alpha in (0, 1], fp32
    beta = torch.rand(expanded, H_o, device=device).contiguous()
    initial_state = (
        (0.01 * torch.randn(len(seq_lens), H_o, D, D, device=device)).to(state_dtype)
        if args.initial_state
        else None
    )
    output = torch.empty(total, H_o, D, dtype=input_dtype, device=device)
    output_state = torch.empty(
        len(seq_lens), H_o, D, D, dtype=state_dtype, device=device
    )
    scale = D**-0.5

    def run_backend(backend, out=output, out_state=output_state):
        if backend != "cudnn":
            raise ValueError(f"Unsupported backend: {backend}")
        return chunk_gated_delta_product(
            q,
            k,
            v,
            g,
            beta,
            n,
            scale,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=args.qk_l2norm_in_kernel,
            output=out,
            output_state=out_state,
            backend="cudnn",
        )[0]

    outputs = {}
    if args.refcheck:
        _, serial_delta_product = _linear_attention_oracles()
        ref_out, _ = serial_delta_product(
            q,
            k,
            v,
            cu_seqlens,
            num_householder=n,
            alpha=g,
            beta=beta,
            initial_state=initial_state,
            scale=scale,
            l2norm=args.qk_l2norm_in_kernel,
        )
        for b_ in backends:
            outputs[b_] = run_backend(
                b_, torch.empty_like(output), torch.empty_like(output_state)
            ).clone()

    times = _time_backends(args, backends, run_backend)
    if args.refcheck and outputs:
        _check_rel_err(args, outputs, ref_out, 5e-2)

    # n Householder updates per token: 2 GEMM-equivalents per update + readout.
    flops = 2 * 2 * expanded * H_o * D * D
    bytes_ = gdn_prefill_bytes(
        total,
        len(seq_lens),
        H_q,
        H_k,
        H_v,
        D,
        input_dtype,
        state_dtype_bytes=state_dtype.itemsize,
        read_initial_state=initial_state is not None,
        gate_bytes_per_token_head=4 + 4 * n,
        kv_rows=expanded,
    )
    fields = {
        "batch_size": len(seq_lens),
        "s_qo": args.s_qo,
        "num_q_heads": H_q,
        "num_k_heads": H_k,
        "num_v_heads": H_v,
        "head_size": D,
        "input_dtype": str(input_dtype),
        "state_dtype": args.state_dtype,
        "gdn_flags": _gdn_flags(args, total_tokens=total, num_householder=n),
    }
    res += _emit_rows(args, backends, times, flops, bytes_, fields, {})
    return res


# ==============================================================================
# ucache (ReplaySSM history-ring) MTP verify / verify+flush
# ==============================================================================

_UCACHE_FLUSH_MIN = 13  # the serving default used by bench_gdn_ucache_flush.py
_UCACHE_FLAT_RING = 16  # verify-only kernel: legacy 16-deep flat ring
_UCACHE_ROT_RING = 32  # flush kernel: 32-slot rotating ring


def _ucache_ref_fp32(q, k, v, a, b, A_log, dt_bias, S0, kc, uc, gc, P, scale):
    """fp32 oracle for one request (port of tests/gdn/test_decode_ucache.py).

    S0 is [HV, V, K]; ring rows [0, P) hold the live history (base 0).
    """
    f = torch.float32
    HV, H = v.shape[1], q.shape[1]
    grp = HV // H
    S = S0.to(f).clone()
    if P > 0:
        GP = gc[:, P - 1].to(f)
        w = torch.exp(GP[:, None] - gc[:, :P].to(f))
        kc_hv = kc[:, :P].to(f).repeat_interleave(grp, dim=0)
        S = torch.exp(GP)[:, None, None] * S + torch.einsum(
            "hpv,hpk->hvk", w[:, :, None] * uc[:, :P].to(f), kc_hv
        )
    khat = F.normalize(k.to(f), dim=-1)
    qhat = F.normalize(q.to(f), dim=-1) * scale
    T = q.shape[0]
    y = torch.zeros(T, HV, v.shape[-1], dtype=f, device=q.device)
    for t in range(T):
        la = -torch.exp(A_log.to(f)) * F.softplus(a[t].to(f) + dt_bias.to(f))
        beta = torch.sigmoid(b[t].to(f))
        k_hv = khat[t].repeat_interleave(grp, dim=0)
        q_hv = qhat[t].repeat_interleave(grp, dim=0)
        S = S * torch.exp(la)[:, None, None]
        pred = torch.einsum("hvk,hk->hv", S, k_hv)
        u_t = (v[t].to(f) - pred) * beta[:, None]
        S = S + u_t[:, :, None] * k_hv[:, None, :]
        y[t] = torch.einsum("hvk,hk->hv", S, q_hv)
    return y


def testGatedDeltaRuleMtpUcache(args):
    """
    Test the ReplaySSM u/d-cache spec-decode kernels (CuTe-DSL, bf16).

    - gated_delta_rule_mtp_ucache: verify only (checkpoint state read-only,
      T new ring entries appended at [P, P+T) of a 16-deep flat ring).
    - gated_delta_rule_mtp_ucache_flush: fused verify + per-request fold;
      --flush_rate percent of requests sit at P=13 (>= flush_min) and fold
      their ring into the state pool in place. The caller-owned cursor-commit
      path (restart_hist_on_flush=False) is timed, i.e. kernel only.

    Only the default bf16 dtype arm of the flush module is exercised (its IO/
    state/ring dtypes are fixed at import time via GDN_UCACHE_*_DTYPE).
    Refcheck: fp32 ring-replay oracle, max |y - ref| < 8e-3 (the library
    test's Y_TOL).
    """
    device, backends = _prepare(args)
    res = []
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res
    is_flush = args.routine == "gated_delta_rule_mtp_ucache_flush"
    from flashinfer import gdn_kernels as _gk

    fn = (
        _gk.gated_delta_rule_mtp_ucache_flush
        if is_flush
        else _gk.gated_delta_rule_mtp_ucache
    )
    if fn is None:
        raise RuntimeError(f"{args.routine}: CuTe-DSL kernel module failed to import")

    B, T = args.batch_size, args.seq_len
    H, HV, D = args.num_q_heads, args.num_v_heads, args.head_size
    if args.num_k_heads != H:
        raise ValueError("ucache kernels require num_k_heads == num_q_heads")
    if args.input_dtype != "bfloat16" or args.state_dtype not in ("bfloat16",):
        raise ValueError(
            "ucache routines run the bf16 arm only: --input_dtype bfloat16 "
            "--state_dtype bfloat16"
        )
    io = torch.bfloat16
    P = args.hist_len
    ring = _UCACHE_ROT_RING if is_flush else _UCACHE_FLAT_RING
    if not is_flush and P + T > _UCACHE_FLAT_RING:
        raise ValueError(f"verify ring needs hist_len + seq_len <= 16, got {P}+{T}")
    if is_flush and not (0 <= P <= 16):
        raise ValueError("flush hist_len must be in [0, 16]")
    scale = D**-0.5

    def rn(*s, sc=1.0):
        return (torch.randn(*s, device=device) * sc).to(io)

    q, k = rn(B, T, H, D), rn(B, T, H, D)
    v, a, b = rn(B, T, HV, D, sc=0.5), rn(B, T, HV, sc=0.5), rn(B, T, HV)
    A_log = (
        torch.full((HV,), -3.0, device=device) + torch.rand(HV, device=device) * 0.3
    ).to(io)
    dt_bias = rn(HV, sc=0.5)
    pool = (torch.randn(B, HV, D, D, device=device) * 0.5).to(io)
    hist = [P] * B
    if is_flush:
        nf = 0 if args.flush_rate == 0 else max(1, round(B * args.flush_rate / 100))
        for r in range(nf):  # deterministic: first nf rows fold
            hist[(r * max(1, B // max(nf, 1))) % B] = _UCACHE_FLUSH_MIN
    maxP = max(hist)
    kc = torch.zeros(B, H, ring, D, dtype=io, device=device)
    uc = torch.zeros(B, HV, ring, D, dtype=io, device=device)
    gc = torch.zeros(B, HV, ring, dtype=torch.float32, device=device)
    if maxP > 0:
        kc[:, :, :maxP] = F.normalize(
            torch.randn(B, H, maxP, D, device=device), dim=-1
        ).to(io)
        uc[:, :, :maxP] = (torch.randn(B, HV, maxP, D, device=device) * 0.3).to(io)
        la = -(torch.rand(B, HV, maxP, device=device) * 0.3 + 0.003)
        gc[:, :, :maxP] = torch.cumsum(la, dim=-1)
    hl = torch.tensor(hist, dtype=torch.int32, device=device)
    cb = torch.zeros(B, dtype=torch.int32, device=device)
    idx = torch.arange(B, dtype=torch.int32, device=device)
    pool0, kc0, uc0, gc0 = pool.clone(), kc.clone(), uc.clone(), gc.clone()

    def run_backend(backend, pool_=pool, kc_=kc, uc_=uc, gc_=gc):
        if backend != "cute-dsl":
            raise ValueError(f"Unsupported backend: {backend}")
        kw = dict(
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=pool_,
            initial_state_indices=idx,
            k_cache=kc_,
            u_cache=uc_,
            g_cache=gc_,
            hist_len=hl,
            scale=scale,
        )
        if is_flush:
            kw.update(
                cache_base=cb,
                flush_min=_UCACHE_FLUSH_MIN,
                restart_hist_on_flush=False,
            )
        return fn(A_log, a, dt_bias, **kw)

    outputs = {}
    if args.refcheck:
        ref = torch.stack(
            [
                _ucache_ref_fp32(
                    q[r],
                    k[r],
                    v[r],
                    a[r],
                    b[r],
                    A_log,
                    dt_bias,
                    pool0[r],
                    kc0[r],
                    uc0[r],
                    gc0[r],
                    hist[r],
                    scale,
                )
                for r in range(B)
            ]
        )
        for b_ in backends:
            outputs[b_] = run_backend(
                b_, pool0.clone(), kc0.clone(), uc0.clone(), gc0.clone()
            ).clone()

    times = _time_backends(args, backends, run_backend)
    if args.refcheck and outputs:
        for name, out in outputs.items():
            err = (out.float() - ref).abs().max().item()
            if not math.isfinite(err) or err > 8e-3:
                print(
                    f"[ERROR] Output tensor mismatch from backend {name}: "
                    f"max |y - fp32 ref| = {err:.3e} > 8e-3"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(f"[ERROR] Backend {name} max err {err:.3e}")
            elif args.verbose >= 1:
                print(f"[REFCHECK] Backend {name}: PASSED (max abs err {err:.3e})")

    n_fold = sum(1 for p in hist if is_flush and p >= _UCACHE_FLUSH_MIN)
    flops = gdn_decode_flops(B, H, HV, D, T)
    bytes_ = gdn_decode_bytes(
        B, H, H, HV, D, io, T, state_writeback=False, state_dtype_bytes=2
    )
    # ring reads (P rows of k/u) + appends (T rows) + folded state writes
    bytes_ += sum(hist) * (H + HV) * D * 2 + B * T * (H + HV) * D * 2
    bytes_ += n_fold * HV * D * D * 2
    fields = {
        "batch_size": B,
        "num_q_heads": H,
        "num_k_heads": H,
        "num_v_heads": HV,
        "head_size": D,
        "seq_len": T,
        "input_dtype": str(io),
        "state_dtype": str(io),
        "update_state": n_fold > 0,
        "gdn_flags": _gdn_flags(
            args,
            hist_len=P,
            **({"flush_rate": args.flush_rate, "n_fold": n_fold} if is_flush else {}),
        ),
    }
    res += _emit_rows(args, backends, times, flops, bytes_, fields, {})
    return res


# ==============================================================================
# Fused decode step (SM120 specialized kernels / composable torch path)
# ==============================================================================


def testGdnFusedDecodeStep(args):
    """
    Test gdn_fused_decode_step (b/a GEMV + causal conv1d update + q/k/v split
    + gated delta-rule decode, over paged conv/ssm state pools).

    The op has no backend selector: the library runs a registered SM120
    specialized kernel or its composable torch path.
    - 'flashinfer'  whatever the library picks; gdn_kernel records
                    'specialized' or 'composable' (via
                    gdn_fused_decode_step_supported).
    - 'specialized' fails unless the specialized kernel would serve this
                    geometry on this device (never times the torch path under
                    a kernel label).
    Refcheck: against the module's composable implementation on cloned pools.
    """
    from flashinfer import gdn_fused_decode_step, gdn_fused_decode_step_supported
    from flashinfer.gdn_kernels.experimental import gdn_fused_decode as _fd

    device, backends = _prepare(args)
    res = []
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res
    B, hidden = args.batch_size, args.hidden_size
    H, HV, D = args.num_q_heads, args.num_v_heads, args.head_size
    if args.num_k_heads != H:
        raise ValueError("gdn_fused_decode_step requires num_k_heads == num_q_heads")
    n_ba = 2 * HV
    qkv_dim = (2 * H + HV) * D
    width, state_len = 4, 3
    pool_n = max(48, B + 1)

    def probe():
        return gdn_fused_decode_step_supported(
            B,
            hidden_size=hidden,
            n_ba=n_ba,
            qkv_dim=qkv_dim,
            num_qk_heads=H,
            num_v_heads=HV,
            head_dim=D,
            conv_width=width,
            conv_state_len=state_len,
            device=device,
            conv_state_layout=args.conv_layout,
        )

    supported = probe()
    kernel = "specialized" if supported else "composable"
    print(f"[INFO] gdn_fused_decode_step dispatches to: {kernel}")
    if "specialized" in backends and not supported:
        raise RuntimeError(
            "specialized: gdn_fused_decode_step_supported() is False for this "
            "geometry/device (the library would run its composable torch path)"
        )

    def bf(*s, sc=1.0):
        return (torch.randn(*s, device=device) * sc).bfloat16()

    if args.conv_layout == "SD":
        conv_state = bf(pool_n, state_len, qkv_dim, sc=0.5).transpose(-1, -2)
    else:
        conv_state = bf(pool_n, qkv_dim, state_len, sc=0.5)
    inputs = {
        "hidden_states": bf(B, hidden, sc=0.5),
        "w_ba": bf(hidden, n_ba, sc=0.02),
        "mixed_qkv": bf(B, qkv_dim, sc=0.5),
        "conv_weight": bf(qkv_dim, width, sc=0.3),
        "conv_bias": bf(qkv_dim, sc=0.1),
        "conv_state": conv_state,
        "A_log": torch.randn(HV, device=device) * 0.5,
        "dt_bias": bf(HV, sc=0.1),
        "scale": 1.0 / math.sqrt(D),
        "ssm_state": torch.randn(pool_n, HV, D, D, device=device) * 0.05,
        "state_indices": torch.arange(
            pool_n - 1, pool_n - 1 - B, -1, device=device
        ).int(),
        "use_qk_l2norm": True,
    }
    out = torch.empty(B, 1, HV, D, dtype=torch.bfloat16, device=device)

    def run_backend(backend, conv=None, ssm=None, o=out):
        kw = dict(inputs)
        if conv is not None:
            kw["conv_state"], kw["ssm_state"] = conv, ssm
        return gdn_fused_decode_step(**kw, out=o)[0]

    def clone_pools():
        c = inputs["conv_state"]
        c2 = (
            c.transpose(-1, -2).clone().transpose(-1, -2)
            if args.conv_layout == "SD"
            else c.clone()
        )
        return c2, inputs["ssm_state"].clone()

    outputs = {}
    if args.refcheck:
        c_ref, s_ref = clone_pools()
        kw = dict(inputs)
        kw["conv_state"], kw["ssm_state"] = c_ref, s_ref
        ref = _fd._gdn_fused_decode_step_fallback(**kw)[0]
        for b_ in backends:
            c2, s2 = clone_pools()
            outputs[b_] = run_backend(b_, c2, s2, torch.empty_like(out)).clone()

    times = _time_backends(args, backends, run_backend)
    # A specialized kernel that fails at run time is latched off for the rest of
    # the process and the library silently serves the composable path
    # (gdn_fused_decode.py:414-416); re-probe so such a row is not labelled
    # specialized.
    if supported and not probe():
        kernel = "composable"
        if "specialized" in times:
            print(
                "[ERROR] specialized: the library latched the specialized kernel "
                "off during the run (it served the composable path); no row."
            )
            times["specialized"] = []
    if args.refcheck and outputs:
        _check_and_report(args, outputs, ref, 2e-2, 2e-2)

    flops = gdn_decode_flops(B, H, HV, D, 1) + 2 * B * hidden * n_ba
    bytes_ = (
        gdn_decode_bytes(B, H, H, HV, D, torch.bfloat16, 1, True, 4)
        + hidden * n_ba * 2
        + B * hidden * 2
        + B * qkv_dim * 2 * (1 + 2 * state_len)
    )
    fields = {
        "batch_size": B,
        "num_q_heads": H,
        "num_k_heads": H,
        "num_v_heads": HV,
        "head_size": D,
        "seq_len": 1,
        "input_dtype": "torch.bfloat16",
        "state_dtype": "torch.float32",
        "state_layout": "pretranspose",
        "gdn_flags": _gdn_flags(args, hidden=hidden, conv_layout=args.conv_layout),
    }
    kernels = {"flashinfer": kernel, "specialized": "specialized"}
    res += _emit_rows(args, backends, times, flops, bytes_, fields, kernels)
    return res


# ==============================================================================
# ReplaySSM commit (fold the accepted prefix into every layer's checkpoint)
# ==============================================================================


def _replayssm_commit_reference(ckpt, rawv, rawk, g, beta, slots, accept, l2norm):
    """FP64 fold of the accepted prefix (tests/gdn/test_replayssm_spec_fold.py)."""
    out = ckpt.clone()
    hv, h = rawv.shape[2], rawk.shape[2]
    for layer in range(ckpt.shape[0]):
        for i, slot in enumerate(slots.tolist()):
            n = int(accept[i])
            if n == 0:
                continue
            S = ckpt[layer, slot].double()  # [HV, V, K]
            k = rawk[layer, slot].double()  # [H, T, K]
            if l2norm:
                k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
            k = k.repeat_interleave(hv // h, dim=0)
            v = rawv[layer, slot].double()
            for t in range(n):
                S = S * g[layer, slot, :, t].double().exp()[:, None, None]
                pred = (S * k[:, t, None, :]).sum(-1)
                delta = (v[:, t] - pred) * beta[layer, slot, :, t].double()[:, None]
                S = S + delta[..., None] * k[:, t, None, :]
            out[layer, slot] = S.float()
    return out


def testGatedDeltaRuleReplaySSMCommit(args):
    """
    Time gated_delta_rule_replayssm_commit: after a frozen-state MTP verify that
    cached the raw ReplaySSM window (gated_delta_rule_mtp(cache_replayssm=True)),
    fold each request's accepted prefix into every layer's FP32 checkpoint in
    one call.

    Backends: 'flashinfer' (backend="auto"), 'simt' (FP32 SIMT), 'tcgen05'
    (TF32 tcgen05; normalized K, 4 <= T <= 8). SM10x only (major == 10).
    Accepted lengths are seeded uniform in [1, T]. Refcheck: FP64 fold.
    """
    from flashinfer.gdn_decode import gated_delta_rule_replayssm_commit

    device, backends = _prepare(args)
    res = []
    if not backends:
        print("[ERROR] No backends to test. Exiting.")
        return res
    if args.head_size != 128:
        print("[ERROR] gated_delta_rule_replayssm_commit needs head_size 128.")
        return res
    L, B, T = args.num_layers, args.batch_size, args.seq_len
    H, HV, D = args.num_k_heads, args.num_v_heads, args.head_size
    slots_n = B + 1  # one spare slot, left untouched
    gen = torch.Generator(device="cpu").manual_seed(args.random_seed)
    ckpt = torch.randn(L, slots_n, HV, D, D, device=device) * 0.1
    rawv = (torch.randn(L, slots_n, HV, T, D, device=device) * 0.1).bfloat16()
    rawk = (torch.randn(L, slots_n, H, T, D, device=device) * 0.1).bfloat16()
    g = -F.softplus(torch.randn(L, slots_n, HV, T, device=device))
    beta = torch.rand(L, slots_n, HV, T, device=device)
    slots = (torch.randperm(B, generator=gen) + 1).to(torch.int32).to(device)
    accept = torch.randint(1, T + 1, (B,), generator=gen, dtype=torch.int32).to(device)
    l2norm = not args.no_qk_l2norm
    clean = ckpt.clone()

    def run_backend(backend, state=ckpt):
        gated_delta_rule_replayssm_commit(
            state,
            rawv,
            rawk,
            g,
            beta,
            slots,
            accept,
            use_qk_l2norm=l2norm,
            backend="auto" if backend == "flashinfer" else backend,
        )
        return state

    outputs = {}
    if args.refcheck:
        reference = _replayssm_commit_reference(
            clean, rawv, rawk, g, beta, slots, accept.cpu(), l2norm
        )
        for b_ in backends:
            outputs[b_] = run_backend(b_, clean.clone()).detach().clone()

    times = _time_backends(args, backends, run_backend)
    if args.refcheck and outputs:
        _check_and_report(args, outputs, reference, 2e-2, 2e-2)

    accepted = int(accept.sum().item())
    state_bytes = L * B * HV * D * D * 4 * 2
    window_bytes = L * accepted * ((HV + H) * D * 2 + 2 * HV * 4)
    flops = L * accepted * HV * 4 * D * D
    fields = {
        "batch_size": B,
        "num_q_heads": args.num_q_heads,
        "num_k_heads": H,
        "num_v_heads": HV,
        "head_size": D,
        "seq_len": T,
        "state_dtype": "torch.float32",
        "use_qk_l2norm": l2norm,
        "gdn_flags": _gdn_flags(args, num_layers=L),
    }
    kernels = {b_: ("auto" if b_ == "flashinfer" else b_) for b_ in backends}
    res += _emit_rows(
        args, backends, times, flops, state_bytes + window_bytes, fields, kernels
    )
    return res
