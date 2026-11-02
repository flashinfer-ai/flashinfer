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

"""KDA (Kimi Delta Attention) benchmark routines.

Routines and the library entry each backend name reaches:

``recurrent_kda_prefill``
    ``flashinfer``        ``flashinfer.recurrent_kda(backend="auto")``
    ``cute-dsl``          ``recurrent_kda(backend="cute-dsl")`` (SM100 BT16 / SM120)
    ``cute-dsl-persistent``  ``recurrent_kda(backend="cute-dsl-persistent")`` (SM100/SM103
                          persistent kernel; FP32 state, H % 8 == 0)
    ``ptx``               ``recurrent_kda(backend="ptx")`` (static SM103a programs;
                          B=1, H in {64, 96}, T >= 32, lower_bound=-5, FP32 state)
    ``small-bh``          ``recurrent_kda(backend="small-bh")``
    ``cudnn``             ``recurrent_kda(backend="cudnn")``
    ``flashinfer-decomp`` / ``flashinfer-fused``  SM120 variants, pinned
    ``cutekda`` / ``flash-kda``  external baselines, when installed
``recurrent_kda_decode``      T=1 and speculative (1+S) decode:
    ``flashinfer`` (auto), ``cute-dsl``
``recurrent_kda_output_only``  ``recurrent_kda(disable_state_update=True)``:
    ``cute-dsl``
``fused_kda_decode``          ``flashinfer.fused_kda_decode``: ``cute-dsl``
``packed_fused_kda_decode``   ``flashinfer.packed_fused_kda_decode`` (MTP): ``cute-dsl``

The Cake backends (``backend="cake"``, ``packed_kda_decode``, the prepared
BF16/TF32 prefill portfolios and the evolution kernel) are out of scope for
this harness and not offered. ``flashinfer`` (library auto) can still route to
a Cake program internally; ``resolved_backend`` then names it.

An explicitly named backend is passed to the library as that backend, never
``auto``; when the library cannot serve the call it raises, the row reports
``[ERROR]`` and no ``[PERF]`` line is printed for it. The ``resolved_backend``
column records the implementation that actually ran, observed by wrapping the
library's internal launch functions for the one untimed warmup call.
"""

import contextlib
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    filter_backends_by_compute_capability,
    get_device,
    print_perf_metrics,
)

HEAD_DIM = 128

#: recurrent_kda_prefill backends. ``flashinfer`` is the public entry point
#: with its own policy (``backend="auto"``); the others are the library's own
#: strict ``backend=`` values (plus the experimental plan/run wrapper). The
#: library's ``tirx`` backend is not offered: it needs an external TVM/TIRx
#: install (``tirx_kernels``) that is not a FlashInfer dependency.
#: ``flashinfer-decomp`` / ``flashinfer-fused`` pin one SM120 variant each.
_PREFILL_API_BACKENDS = {
    "flashinfer": "auto",
    "cute-dsl": "cute-dsl",
    "cute-dsl-persistent": "cute-dsl-persistent",
    "ptx": "ptx",
    "small-bh": "small-bh",
    "cudnn": "cudnn",
}
_SM120_VARIANT_BACKENDS = {"flashinfer-decomp": "decomp", "flashinfer-fused": "fused"}
#: ``cutekda`` and ``flash-kda`` are external and optional. They are imported
#: only when asked for and only if already installed; a missing one is an
#: explicit skip rather than a silent omission.
_EXTERNAL_BACKENDS = ("cutekda", "flash-kda")
_FLASHINFER_BACKENDS = tuple(_PREFILL_API_BACKENDS) + tuple(_SM120_VARIANT_BACKENDS)

ROUTINE_BACKENDS = {
    "recurrent_kda_prefill": _FLASHINFER_BACKENDS + _EXTERNAL_BACKENDS,
    "recurrent_kda_decode": ("flashinfer", "cute-dsl"),
    "recurrent_kda_output_only": ("cute-dsl",),
    "fused_kda_decode": ("cute-dsl",),
    "packed_fused_kda_decode": ("cute-dsl",),
}
_ALL_BACKENDS = tuple(
    dict.fromkeys(name for names in ROUTINE_BACKENDS.values() for name in names)
)

#: fused_kda_decode and packed_fused_kda_decode T=1 accept H=8 (Kimi-K3 TP12);
#: the packed T>1 multitoken kernel does not.
_FUSED_KDA_HEADS = (8, 12, 24, 32, 48, 96)

_PREFILL = ("recurrent_kda_prefill",)
#: Routine-specific flags -> the routines that consume them.
_ROUTINE_FLAGS = {
    "s_qo": _PREFILL + ("recurrent_kda_output_only",),
    "packed": _PREFILL,
    "ragged": _PREFILL,
    "offsets_dtype": _PREFILL,
    "has_initial_state": _PREFILL,
    "use_state_indices": _PREFILL + ("recurrent_kda_decode",),
    "state_pool_slots": _PREFILL
    + (
        "recurrent_kda_decode",
        "recurrent_kda_output_only",
    ),
    "state_indices_affine": _PREFILL,
    "lower_bound": tuple(ROUTINE_BACKENDS),
    "use_gate_in_kernel": _PREFILL
    + ("recurrent_kda_decode", "recurrent_kda_output_only"),
    "beta_is_logit": _PREFILL + ("recurrent_kda_decode", "recurrent_kda_output_only"),
    "num_spec_tokens": ("recurrent_kda_decode", "packed_fused_kda_decode"),
    "verify_caches": ("recurrent_kda_output_only",),
}
#: Columns written from these flags would otherwise carry the flag default on
#: rows of routines that never read it.
_CLEARED_WHEN_UNUSED = (
    "num_spec_tokens",
    "verify_caches",
    "offsets_dtype",
)
_PACKED_FUSED_MTP_HEADS = (12, 24, 32, 48, 96)
_PACKED_KDA_HEADS = 12


def run_kda_test(args):
    """Route a KDA benchmark case to its routine."""
    routines = {
        "recurrent_kda_prefill": testRecurrentKDAPrefill,
        "recurrent_kda_decode": testRecurrentKDADecode,
        "recurrent_kda_output_only": testRecurrentKDAOutputOnly,
        "fused_kda_decode": testFusedKDADecode,
        "packed_fused_kda_decode": testPackedFusedKDADecode,
    }
    if args.routine not in routines:
        raise ValueError(f"Unsupported routine: {args.routine}")
    return routines[args.routine](args)


def _lower_bound_arg(value):
    if str(value).lower() in ("none", "null", "softplus"):
        return None
    return float(value)


def parse_kda_args(line, parser):
    """Parse the KDA-specific arguments.

    Args:
        line: Command line arguments
        parser: ArgumentParser already populated with the shared arguments

    Returns:
        Parsed argument namespace
    """
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Number of sequences (decode: number of decode rows / sequences).",
    )
    parser.add_argument(
        "--s_qo",
        type=int,
        default=1,
        help="Tokens per sequence. Prefill routines only; decode routines use "
        "--num_spec_tokens and output-only uses --s_qo as the verify window. "
        "With --packed the sequences are packed into one [1, B * T, H, 128] "
        "activation with explicit offsets; without it they are [B, T, H, 128].",
    )
    parser.add_argument(
        "--num_q_heads",
        type=int,
        default=16,
        help="Number of query/key heads H.",
    )
    parser.add_argument(
        "--num_v_heads",
        type=int,
        default=None,
        help="Number of value heads HV (GQA when HV != H; HV must be a multiple "
        "of H). Defaults to --num_q_heads. GQA is accepted only by the "
        "precomputed-gate (--no_use_gate_in_kernel) paths.",
    )
    parser.add_argument(
        "--head_size",
        type=int,
        default=HEAD_DIM,
        help="Head dimension. Fixed at 128.",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16"],
        help="Data type for q/k/v/g/beta.",
    )
    parser.add_argument(
        "--state_dtype",
        type=str,
        default=None,
        choices=["bfloat16", "float32"],
        help="Recurrent state dtype. Default: bfloat16, except "
        "packed_fused_kda_decode (float32).",
    )
    parser.add_argument(
        "--packed",
        action="store_true",
        help="Use packed varlen input with explicit cu_seqlens.",
    )
    parser.add_argument(
        "--ragged",
        action="store_true",
        help="Packed input with unequal, seeded sequence lengths drawn from "
        "[s_qo/2, 3*s_qo/2]. Implies --packed.",
    )
    parser.add_argument(
        "--offsets_dtype",
        type=str,
        default="int32",
        choices=["int32", "int64"],
        help="cu_seqlens dtype for packed input.",
    )
    parser.add_argument(
        "--has_initial_state",
        action="store_true",
        help="Pass an initial state, updated in place.",
    )
    parser.add_argument(
        "--use_state_indices",
        action="store_true",
        help="Recurrent state lives in a pool of --state_pool_slots slots and "
        "each sequence addresses its slot through ssm_state_indices / "
        "state_indices. Implies --has_initial_state.",
    )
    parser.add_argument(
        "--state_pool_slots",
        type=int,
        default=0,
        help="State pool size for --use_state_indices. 0 = auto: 257 for "
        "recurrent_kda_prefill (the exact capacity the generated indexed Cake "
        "kernel accepts) and batch_size + 2 elsewhere.",
    )
    parser.add_argument(
        "--state_indices_affine",
        type=int,
        nargs=2,
        default=None,
        metavar=("A", "B"),
        help="With --use_state_indices, use slot (A * i + B) % state_pool_slots "
        "for sequence i instead of a seeded random permutation. The generated "
        "indexed Cake prefill rows pin exact index contents; e.g. its packed "
        "H12 8x128 row was generated for A=175 B=246.",
    )
    parser.add_argument(
        "--lower_bound",
        type=_lower_bound_arg,
        default=-5.0,
        help="Gate lower bound in [-5.0, 0.0), or 'none' for the unbounded "
        "softplus gate -exp(A_log) * softplus(g + dt_bias) (the Cake "
        "affine / T=1 frozen-decode route).",
    )
    parser.add_argument(
        "--no_use_gate_in_kernel",
        dest="use_gate_in_kernel",
        action="store_false",
        help="Pass a precomputed log-space gate (A_log/dt_bias/lower_bound are "
        "not used). Default: raw gate transformed in kernel.",
    )
    parser.add_argument(
        "--no_beta_is_logit",
        dest="beta_is_logit",
        action="store_false",
        help="Pass beta already sigmoided. Default: beta logits, sigmoid in kernel.",
    )
    parser.add_argument(
        "--num_spec_tokens",
        type=int,
        default=0,
        help="recurrent_kda_decode / packed_fused_kda_decode: speculative "
        "tokens S; each sequence processes 1+S tokens. 0 = plain T=1 decode.",
    )
    parser.add_argument(
        "--verify_caches",
        action="store_true",
        help="recurrent_kda_output_only: also write the slot-indexed "
        "correction/kg verify caches (packed frozen-verify kernel).",
    )
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=["flashinfer"],
        choices=list(_ALL_BACKENDS),
        help="Backends to benchmark; the accepted set depends on --routine "
        "(see benchmarks/routines/kda.py). 'flashinfer' is the library's "
        "auto policy; every other name selects that implementation strictly.",
    )
    args = parser.parse_args(line)
    # A flag the routine does not consume would be silently ignored and still
    # echoed into the CSV; reject it. Flags left at their default for such a
    # routine are cleared so the CSV backfill writes "" rather than a value
    # that did not take part in the run.
    for dest, routines in _ROUTINE_FLAGS.items():
        if args.routine in routines:
            continue
        if getattr(args, dest) != parser.get_default(dest):
            raise ValueError(f"--{dest} does not apply to {args.routine}")
        if dest in _CLEARED_WHEN_UNUSED:
            setattr(args, dest, "")
    if args.num_v_heads is None:
        args.num_v_heads = args.num_q_heads
    if args.ragged:
        args.packed = True
    if args.use_state_indices:
        args.has_initial_state = True
    if args.state_dtype is None:
        args.state_dtype = (
            "float32" if args.routine == "packed_fused_kda_decode" else "bfloat16"
        )
    if (
        set(args.backends) & {"cute-dsl-persistent", "ptx"}
        and not args.has_initial_state
    ):
        print(
            "[WARNING] cute-dsl-persistent / ptx without --has_initial_state: the "
            "library clones the final state inside the timed call "
            "(kda_prefill_persistent.py:275, kda_prefill_ptx.py:315)."
        )
    if args.routine == "recurrent_kda_prefill" and args.s_qo < 2:
        raise ValueError(f"{args.routine} is multi-token prefill: pass --s_qo >= 2")
    return args


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _preamble(args, name):
    if args.verbose >= 1:
        print(f"[INFO] Running {name}")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: "
            f"{args.repro_command}"
        )
    return device


def _select_backends(args, device):
    """Validate the routine's backend names, then apply the cc whitelist."""
    allowed = ROUTINE_BACKENDS[args.routine]
    backends = []
    for backend in args.backends:
        if backend not in allowed:
            print(
                f"[ERROR] backend {backend} is not a {args.routine} backend "
                f"(accepted: {', '.join(allowed)}). Skipping it."
            )
            continue
        backends.append(backend)
    return filter_backends_by_compute_capability(backends, args.routine, device)


def _sequence_lengths(args):
    """Per-sequence token counts: uniform, or seeded ragged lengths."""
    num_seqs, seq_len = args.batch_size, args.s_qo
    if not args.ragged:
        return [seq_len] * num_seqs
    rng = np.random.default_rng(args.random_seed)
    low = max(1, seq_len // 2)
    high = max(low, (3 * seq_len) // 2)
    return [int(x) for x in rng.integers(low, high + 1, size=num_seqs)]


def _offsets(lengths):
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return offsets


def _l2norm(x):
    return F.normalize(x.float(), p=2.0, dim=-1)


def _log_gate(g, A_log, dt_bias, use_gate_in_kernel, lower_bound):
    """Log-space per-channel decay [*, HV, K] for the three gate contracts."""
    if not use_gate_in_kernel:
        return g.float()
    num_heads = A_log.numel()
    bias = dt_bias.float().reshape(num_heads, -1)
    shifted = g.float() + bias
    a = torch.exp(A_log.float()).reshape(num_heads, 1)
    if lower_bound is None:
        return -a * F.softplus(shifted)
    return lower_bound * torch.sigmoid(a * shifted)


def _reference_recurrence(
    q, k, v, log_gate, beta_prob, offsets, initial_states, state_dtype, scale=None
):
    """Token-serial delta-rule recurrence, the public-contract reference.

    Deliberately independent of every kernel's own derivation. Shapes (packed
    token axis): q/k ``[T, H, K]``, v ``[T, HV, V]``, log_gate ``[T, HV, K]``,
    beta_prob ``[T, HV]``; ``initial_states`` ``[N, HV, V, K]``. The state is
    rounded to ``state_dtype`` after every token, as the kernels store it.

    Returns ``(out [T, HV, V] float32, final_states [N, HV, V, K] float32)``.
    """
    total, num_heads, head_dim = q.shape
    num_v_heads = v.shape[1]
    group = num_v_heads // num_heads
    scale = head_dim**-0.5 if scale is None else scale
    qh = _l2norm(q).repeat_interleave(group, dim=1)
    kh = _l2norm(k).repeat_interleave(group, dim=1)
    vf = v.float()
    decay = torch.exp(log_gate.float())
    beta_f = beta_prob.float()
    lengths = [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
    starts = torch.tensor(offsets[:-1], device=q.device, dtype=torch.long)
    lengths_t = torch.tensor(lengths, device=q.device, dtype=torch.long)
    state = initial_states.float().clone()
    out = torch.zeros((total, num_v_heads, head_dim), device=q.device)
    for position in range(max(lengths) if lengths else 0):
        active = (lengths_t > position).nonzero().flatten()
        if active.numel() == 0:
            break
        tokens = starts[active] + position
        s = state[active] * decay[tokens].unsqueeze(2)
        predicted = torch.einsum("nhvk,nhk->nhv", s, kh[tokens])
        residual = beta_f[tokens].unsqueeze(-1) * (vf[tokens] - predicted)
        s = s + residual.unsqueeze(-1) * kh[tokens].unsqueeze(2)
        s = s.to(state_dtype).float()
        state[active] = s
        out[tokens] = scale * torch.einsum("nhvk,nhk->nhv", s, qh[tokens])
    return out, state


def _refcheck(args, backend, reference, actual, tol=2e-2):
    """Relative-RMSE check. Raises unless --allow_output_mismatch."""
    reference = reference.float()
    actual = actual.float().reshape(reference.shape)
    if not torch.isfinite(actual).all():
        rel = float("inf")
    else:
        denom = reference.norm().clamp_min(1e-12)
        rel = ((actual - reference).norm() / denom).item()
    close = torch.isclose(actual, reference, rtol=3e-2, atol=2e-2)
    pct = 100.0 * (1.0 - close.float().mean().item())
    if rel > tol:
        print(
            f"[ERROR] Output tensor mismatch from backend {backend}: rel-RMSE "
            f"{rel:.3e} > {tol:.0e} ({pct:.4f}% elements outside rtol=3e-2/atol=2e-2)"
        )
        if not args.allow_output_mismatch:
            raise AssertionError(
                f"[ERROR] Backend {backend} output mismatch (rel-RMSE {rel:.3e})"
            )
    elif args.verbose >= 1:
        print(
            f"[REFCHECK] Backend {backend}: PASSED (rel-RMSE {rel:.3e}, "
            f"{pct:.4f}% outside rtol=3e-2/atol=2e-2)"
        )


def _time(args, fn):
    """Time ``fn``; returns ``(times, used_cuda_graph)``.

    Graph replay times the kernel rather than Python dispatch; most of these
    calls take a few microseconds, so eager timing would mostly measure the
    host. A call that cannot be captured (a host sync, or a prefill route that
    needs a caller-owned workspace warmed on the capture stream) falls back to
    eager timing, and the row says so through ``no_cuda_graph``.
    """
    common = dict(
        fn=fn,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        enable_cupti=args.use_cupti,
    )
    if not args.no_cuda_graph:
        try:
            return bench_gpu_time(**common, use_cuda_graph=True), True
        except Exception as exc:  # noqa: BLE001 -- reported, then timed eagerly
            torch.cuda.synchronize()
            print(
                f"[WARNING] CUDA-graph capture failed ({type(exc).__name__}: "
                f"{exc}); timing eagerly (row records no_cuda_graph=True)."
            )
    return bench_gpu_time(**common, use_cuda_graph=False), False


def _enum_name(value):
    return str(getattr(value, "value", getattr(value, "name", value)))


class _RouteSpy(contextlib.AbstractContextManager):
    """Record which internal KDA implementation a call reached.

    Wraps the library's module-level launch functions for one untimed call and
    restores them on exit. Reporting only: a function that does not exist in
    this FlashInfer build is simply not wrapped.
    """

    def __init__(self):
        self.routes = []
        self._restore = []

    def _wrap(self, module_name, attr, label):
        import importlib

        try:
            module = importlib.import_module(module_name)
            original = getattr(module, attr)
        except Exception:  # noqa: BLE001 -- reporting only
            return
        spy = self

        def wrapper(*a, **kw):
            spy.routes.append(label(a, kw) if callable(label) else label)
            return original(*a, **kw)

        setattr(module, attr, wrapper)
        self._restore.append((module, attr, original))

    def __enter__(self):
        w = self._wrap
        w(
            "flashinfer.kda_prefill_cute_small_bh",
            "_run_kda_prefill_cute_small_bh",
            "small-bh",
        )
        w("flashinfer.kda_prefill", "_run_sm120_kda_prefill", "sm120")
        w(
            "flashinfer.jit.flash_kda_indexed",
            "_run_flash_kda_indexed_prefill",
            "cake-indexed",
        )
        w("flashinfer.kda_prefill_cute", "_run_cute_dsl_kda_prefill", "cute-dsl-bt16")
        w(
            "flashinfer.kda_prefill_persistent",
            "_run_persistent_kda",
            "cute-dsl-persistent",
        )
        w("flashinfer.kda_prefill_ptx", "_run_ptx_kda", "ptx")
        w("flashinfer.kda_prefill", "_run_flash_kda_prefill", "cake")
        w("flashinfer.cudnn", "cudnn_recurrent_kda", "cudnn")
        w("flashinfer.kda_decode", "_run_frozen_recurrent_kda", "cute-dsl-frozen")
        w(
            "flashinfer.kda_kernels.recurrent_kda",
            "_run_flash_kda_decode",
            lambda a, kw: f"cake:{a[0] if a else kw.get('variant')}",
        )
        w(
            "flashinfer.kda_kernels.recurrent_kda",
            "launch_unpacked_kda_decode_cute",
            "cute-dsl-t1-fast",
        )
        # Cake prefill schedule: the physical module the dispatcher loads.
        w(
            "flashinfer.kda_prefill",
            "_get_flash_kda_prefill_module",
            lambda a, kw: (
                f"cake-module:{_enum_name(a[0])}/{_enum_name(a[1])}"
                if len(a) >= 2
                else "cake-module"
            ),
        )
        w(
            "flashinfer.kda_prefill",
            "_get_cake_kda_prefill_module",
            lambda a, kw: f"cake-module:{_enum_name(a[0])}" if a else "cake-module",
        )
        w("flashinfer.kda_prefill", "_get_cake_kda_affine_module_bundle", "cake-affine")
        # Non-frozen decode dispatch that is not a Cake frozen variant.
        w("flashinfer.kda_decode", "_run_recurrent_kda", "cute-dsl-recurrent")
        w(
            "flashinfer.kda_prefill",
            "_get_flash_kda_generated_module",
            lambda a, kw: (
                "cake-generated:" + str((a[0] if a else {}).get("route", "?"))
            ),
        )
        w(
            "flashinfer.kda_kernels.fused_kda_decode",
            "_run_cake_variant",
            lambda a, kw: f"cake:{getattr(a[0], 'name', a[0]) if a else '?'}",
        )
        return self

    def __exit__(self, *exc):
        for module, attr, original in reversed(self._restore):
            setattr(module, attr, original)
        self._restore.clear()
        return False

    def label(self, default):
        routes = list(dict.fromkeys(self.routes))
        # The generic decode entry wraps the frozen Cake variants and the T=1
        # fast path; report the specific implementation when one was reached.
        if len(routes) > 1 and "cute-dsl-recurrent" in routes:
            routes.remove("cute-dsl-recurrent")
        return "+".join(routes) if routes else default


def _warm_and_time(args, backend, runner, restore, spy_default):
    """Restore inputs, run once under the route spy, capture, restore, time.

    Returns ``(times, route, captured)``; ``times`` is empty on failure.
    """
    try:
        restore()
        with _RouteSpy() as spy:
            captured = runner()
            torch.cuda.synchronize()
        route = spy.label(spy_default)
    except Exception as exc:  # noqa: BLE001 -- reported, never silent
        print(f"[ERROR] backend {backend} failed to run: {type(exc).__name__}: {exc}")
        return [], "", None
    captured = captured.clone() if isinstance(captured, torch.Tensor) else captured
    restore()
    if args.verbose >= 1:
        print(f"[INFO] backend {backend} resolved to: {route}")
    times, used_graph = _time(args, runner)
    # Read back by _emit for this backend's row.
    args.kda_timed_eagerly = not used_graph
    return times, route, captured


def _emit(args, backend, times, flops, nbytes, extra):
    median_time = np.median(times)
    std_time = np.std(times)
    tflops = flops / (10**9 * median_time)
    tb_per_sec = nbytes / (10**9 * median_time)
    print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)
    if args.output_path is None:
        return None
    row = defaultdict(str)
    row["routine"] = args.routine
    row["median_time"] = median_time
    row["std_time"] = std_time
    row["tflops"] = tflops
    row["tb_per_sec"] = tb_per_sec
    row["backend"] = backend
    row["batch_size"] = args.batch_size
    row["s_qo"] = args.s_qo
    row["num_q_heads"] = args.num_q_heads
    row["num_v_heads"] = args.num_v_heads
    row["head_size"] = args.head_size
    row["input_dtype"] = str(dtype_str_to_torch_dtype(args.input_dtype))
    row["state_dtype"] = args.state_dtype
    row["packed"] = args.packed
    row["ragged"] = args.ragged
    row["has_initial_state"] = args.has_initial_state
    row["lower_bound"] = "none" if args.lower_bound is None else args.lower_bound
    row["gate_mode"] = (
        "precomputed"
        if not args.use_gate_in_kernel
        else ("softplus" if args.lower_bound is None else "lower_bound")
    )
    row["beta_is_logit"] = args.beta_is_logit
    row["sm_count"] = torch.cuda.get_device_properties(0).multi_processor_count
    row["no_cuda_graph"] = getattr(args, "kda_timed_eagerly", args.no_cuda_graph)
    # Every routine but prefill always starts from a caller-owned state.
    if args.routine != "recurrent_kda_prefill":
        row["has_initial_state"] = True
    row["case_tag"] = args.case_tag
    row.update(extra)
    return row


def _state_bytes(dtype):
    return torch.finfo(dtype).bits // 8


def _recurrence_flops(total_tokens, num_v_heads, head_size):
    """Three [K, V] rank-updates/projections per token per value head."""
    return 3.0 * 2.0 * total_tokens * num_v_heads * head_size * head_size


def _recurrence_bytes(
    total_tokens, num_heads, num_v_heads, head_size, state_dtype, state_slabs_moved
):
    """Compulsory DRAM traffic: q/k in, v/g/out, beta, state slabs read+written."""
    act = 2  # bf16
    activations = act * total_tokens * head_size * (2 * num_heads + 3 * num_v_heads)
    activations += act * total_tokens * num_v_heads
    state = (
        state_slabs_moved
        * num_v_heads
        * head_size
        * head_size
        * _state_bytes(state_dtype)
    )
    return activations + state


# ---------------------------------------------------------------------------
# recurrent_kda_prefill
# ---------------------------------------------------------------------------


def kda_prefill_flops(total_tokens: int, num_heads: int, head_size: int) -> float:
    """Matmul FLOPs of one KDA prefill (see _recurrence_flops)."""
    return _recurrence_flops(total_tokens, num_heads, head_size)


def _make_prefill_inputs(args, device):
    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    state_dtype = dtype_str_to_torch_dtype(args.state_dtype)
    num_seqs = args.batch_size
    num_heads, num_v_heads, head_size = (
        args.num_q_heads,
        args.num_v_heads,
        args.head_size,
    )
    lengths = _sequence_lengths(args)
    offsets = _offsets(lengths)

    if args.packed:
        batch, tokens = 1, offsets[-1]
        offsets_dtype = torch.int32 if args.offsets_dtype == "int32" else torch.int64
        cu_seqlens = torch.tensor(offsets, dtype=offsets_dtype, device=device)
    else:
        batch, tokens = num_seqs, args.s_qo
        cu_seqlens = None

    def normalized(heads):
        raw = torch.randn((batch, tokens, heads, head_size), device=device)
        return F.normalize(raw, p=2.0, dim=-1).to(input_dtype)

    if args.use_gate_in_kernel:
        g = 0.1 * torch.randn((batch, tokens, num_v_heads, head_size), device=device)
    else:
        g = F.logsigmoid(
            torch.randn((batch, tokens, num_v_heads, head_size), device=device) + 3.0
        )
    beta = torch.randn((batch, tokens, num_v_heads), device=device)
    if not args.beta_is_logit:
        beta = torch.sigmoid(beta)
    inputs = {
        "q": normalized(num_heads),
        "k": normalized(num_heads),
        "v": torch.randn((batch, tokens, num_v_heads, head_size), device=device).to(
            input_dtype
        ),
        "g": g.to(input_dtype),
        "beta": beta.to(input_dtype),
        "A_log": 0.1 * torch.randn(num_v_heads, dtype=torch.float32, device=device),
        "dt_bias": 0.1
        * torch.randn((num_v_heads, head_size), dtype=torch.float32, device=device),
        "cu_seqlens": cu_seqlens,
        "lengths": lengths,
        "offsets": offsets,
        "ssm_state_indices": None,
    }
    state_shape_tail = (num_v_heads, head_size, head_size)
    if args.use_state_indices:
        slots = args.state_pool_slots or max(257, num_seqs + 1)
        if slots < num_seqs + 1:
            raise ValueError(f"--state_pool_slots {slots} < batch_size + 1")
        if args.state_indices_affine is not None:
            a, b = args.state_indices_affine
            chosen = torch.tensor([(a * i + b) % slots for i in range(num_seqs)])
            if len(set(chosen.tolist())) != num_seqs:
                raise ValueError("--state_indices_affine produced repeated slots")
        else:
            gen = torch.Generator(device="cpu").manual_seed(args.random_seed)
            chosen = torch.randperm(slots - 1, generator=gen)[:num_seqs] + 1
        inputs["ssm_state_indices"] = chosen.to(torch.int32).to(device)
        inputs["initial_state"] = (
            0.1 * torch.randn((slots, *state_shape_tail), device=device)
        ).to(state_dtype)
    elif args.has_initial_state:
        inputs["initial_state"] = (
            0.1 * torch.randn((num_seqs, *state_shape_tail), device=device)
        ).to(state_dtype)
    else:
        inputs["initial_state"] = None
    return inputs


def _reference_prefill(inputs, args, device):
    """Token-serial reference for the prefill contract actually requested."""
    q = inputs["q"]
    head_size = q.shape[-1]
    flat = lambda t: t.reshape(-1, *t.shape[2:])  # noqa: E731
    log_gate = _log_gate(
        flat(inputs["g"]),
        inputs["A_log"],
        inputs["dt_bias"],
        args.use_gate_in_kernel,
        args.lower_bound,
    )
    beta = flat(inputs["beta"]).float()
    beta_prob = torch.sigmoid(beta) if args.beta_is_logit else beta
    offsets = (
        inputs["offsets"]
        if args.packed
        else [i * q.shape[1] for i in range(q.shape[0] + 1)]
    )
    num_seqs = len(offsets) - 1
    state_dtype = dtype_str_to_torch_dtype(args.state_dtype)
    shape = (num_seqs, args.num_v_heads, head_size, head_size)
    if inputs["initial_state"] is None:
        init = torch.zeros(shape, device=device)
    elif inputs["ssm_state_indices"] is not None:
        init = inputs["initial_state"][inputs["ssm_state_indices"].long()]
    else:
        init = inputs["initial_state"]
    out, _ = _reference_recurrence(
        flat(q),
        flat(inputs["k"]),
        flat(inputs["v"]),
        log_gate,
        beta_prob,
        offsets,
        init,
        state_dtype,
    )
    return out.reshape(inputs["v"].shape)


def _prefill_runner(backend, inputs, args, output, final_state, device):
    """Zero-argument callable for one recurrent_kda_prefill backend."""
    common = dict(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        A_log=inputs["A_log"] if args.use_gate_in_kernel else None,
        dt_bias=inputs["dt_bias"] if args.use_gate_in_kernel else None,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=args.use_gate_in_kernel,
        beta_is_logit=args.beta_is_logit,
        lower_bound=args.lower_bound if args.use_gate_in_kernel else None,
        initial_state=inputs["initial_state"],
        output=output,
    )
    if backend in _PREFILL_API_BACKENDS:
        api_backend = _PREFILL_API_BACKENDS[backend]

        def call():
            flashinfer.recurrent_kda(
                **common,
                cu_seqlens=inputs["cu_seqlens"],
                ssm_state_indices=inputs["ssm_state_indices"],
                output_final_state=True,
                backend=api_backend,
            )
            return output

        return call

    if backend in _SM120_VARIANT_BACKENDS:
        # Direct SM120 variant entry: lets a report show what the policy chose
        # and what it declined. Not a second public API.
        from flashinfer.kda_kernels import sm120_prefill

        variant = _SM120_VARIANT_BACKENDS[backend]

        def call():
            sm120_prefill.run_kda_prefill_sm120(
                q=inputs["q"],
                k=inputs["k"],
                v=inputs["v"],
                g=inputs["g"],
                beta=inputs["beta"],
                A_log=inputs["A_log"],
                dt_bias=inputs["dt_bias"],
                lower_bound=args.lower_bound,
                cu_seqlens=inputs["cu_seqlens"],
                initial_state=inputs["initial_state"],
                final_state=final_state,
                output=output,
                variant=variant,
            )
            return output

        return call

    return _external_runner(backend, inputs, args, output, final_state)


def _external_runner(backend, inputs, args, output, final_state):
    """A callable for an external baseline, or ``None`` if it is not installed.

    Both baselines write the caller's ``output`` and ``final_state`` in place,
    exactly as the FlashInfer path does.
    """
    scale = args.head_size**-0.5
    try:
        if backend == "cutekda":
            import cute_kda as module
        elif backend == "flash-kda":
            import flash_kda as module
        else:
            return None
    except ImportError:
        return None
    if not inputs:
        return lambda: None

    def call():
        module.fwd(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            scale,
            output,
            inputs["A_log"],
            inputs["dt_bias"],
            args.lower_bound,
            inputs["initial_state"],
            final_state,
            inputs["cu_seqlens"],
        )
        return output

    return call


def testRecurrentKDAPrefill(args):
    """Time recurrent-KDA multi-token prefill, one row per backend.

    Every backend gets the same tensors, stream, pre-allocated ``output`` and
    state buffer, and timing method. Compilation, descriptor construction and
    the first launch happen before the timed region.
    """
    device = _preamble(args, "testRecurrentKDAPrefill")
    res = []
    if args.head_size != HEAD_DIM:
        print(f"[ERROR] KDA prefill is fixed at head_size={HEAD_DIM}. Exiting.")
        return res

    backends = _select_backends(args, device)
    for backend in list(backends):
        if backend in _EXTERNAL_BACKENDS and (
            _external_runner(backend, {}, args, None, None) is None
        ):
            print(
                f"[WARNING] the {backend} baseline is not installed in this "
                f"environment. Skipping it; the report must say so."
            )
            backends.remove(backend)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    state_dtype = dtype_str_to_torch_dtype(args.state_dtype)
    inputs = _make_prefill_inputs(args, device)
    total_tokens = inputs["offsets"][-1]
    output = torch.empty_like(inputs["v"])
    final_state = (
        inputs["initial_state"]
        if inputs["initial_state"] is not None
        else torch.empty(
            (args.batch_size, args.num_v_heads, HEAD_DIM, HEAD_DIM),
            dtype=state_dtype,
            device=device,
        )
    )

    # The SM120 variant policy is meaningful only when the SM120 backend runs.
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    sm120_variant, policy_source = "", ""
    if torch.cuda.get_device_capability(device) == (12, 0):
        try:
            from flashinfer.kda_kernels import sm120_prefill

            sm120_variant = sm120_prefill.choose_variant(
                args.batch_size, args.num_q_heads, args.s_qo, sm_count=sm_count
            )
            profile_key, _ = sm120_prefill.auto_profile(sm_count)
            policy_source = "tuned" if profile_key == sm_count else "fallback"
            if args.verbose >= 1:
                print(f"[INFO] {sm120_prefill.describe_variant_policy(sm_count)}")
        except Exception as exc:  # noqa: BLE001 -- reporting only
            if args.verbose >= 1:
                print(f"[WARNING] could not read the SM120 variant policy: {exc}")

    reference_output = (
        _reference_prefill(inputs, args, device) if args.refcheck else None
    )
    if args.verbose >= 2:
        print(f"[VVERBOSE] {inputs['q'].shape = }, lengths = {inputs['lengths']}")

    # KDA updates the state in place: without a snapshot the first backend's
    # warmup rewrites the state every later backend starts from.
    snapshot = (
        inputs["initial_state"].clone() if inputs["initial_state"] is not None else None
    )

    def restore():
        if snapshot is not None:
            inputs["initial_state"].copy_(snapshot)

    problem_flops = kda_prefill_flops(total_tokens, args.num_v_heads, HEAD_DIM)
    slabs = args.batch_size * ((1 if args.has_initial_state else 0) + 1)
    problem_bytes = _recurrence_bytes(
        total_tokens,
        args.num_q_heads,
        args.num_v_heads,
        HEAD_DIM,
        state_dtype,
        slabs,
    )

    for backend in backends:
        try:
            runner = _prefill_runner(backend, inputs, args, output, final_state, device)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] backend {backend} failed to set up: {exc}")
            continue
        if runner is None:
            continue
        times, route, captured = _warm_and_time(
            args, backend, runner, restore, "unknown"
        )
        if not times:
            continue
        if reference_output is not None:
            _refcheck(args, backend, reference_output, captured)
        if backend in _SM120_VARIANT_BACKENDS or backend in _EXTERNAL_BACKENDS:
            route = backend
        row = _emit(
            args,
            backend,
            times,
            problem_flops,
            problem_bytes,
            {
                "resolved_backend": route,
                "kda_variant": sm120_variant if "sm120" in route else route,
                "kda_variant_policy": policy_source if "sm120" in route else "",
                "state_pool": (
                    inputs["initial_state"].shape[0] if args.use_state_indices else ""
                ),
            },
        )
        if row is not None:
            res.append(row)
    return res


# ---------------------------------------------------------------------------
# recurrent_kda_decode (T=1 and speculative 1+S)
# ---------------------------------------------------------------------------


def testRecurrentKDADecode(args):
    """Time recurrent-KDA decode: T=1, or packed speculative 1+S tokens.

    T=1 uses ``q [N, 1, H, K]`` and either a dense ``[N, HV, V, K]`` state or,
    with ``--use_state_indices``, a pool addressed by ``ssm_state_indices [N]``.
    Speculative decode uses the packed ``[1, N*(1+S), ...]`` layout with
    ``cu_seqlens``, a checkpoint pool and ``ssm_state_indices [N, 1+S]``.
    """
    device = _preamble(args, "testRecurrentKDADecode")
    res = []
    backends = _select_backends(args, device)
    if not backends:
        print("[ERROR] No backends to test. Exiting.")
        return res

    N, H, HV, K = args.batch_size, args.num_q_heads, args.num_v_heads, HEAD_DIM
    S = args.num_spec_tokens
    T = 1 + S
    if args.state_dtype != "bfloat16":
        print("[ERROR] recurrent_kda decode keeps a bfloat16 state. Exiting.")
        return res
    state_dtype = torch.bfloat16
    spec = S > 0
    token_shape = (1, N * T) if spec else (N, 1)

    q = F.normalize(torch.randn((*token_shape, H, K), device=device), dim=-1)
    k = F.normalize(torch.randn((*token_shape, H, K), device=device), dim=-1)
    v = torch.randn((*token_shape, HV, K), device=device)
    if args.use_gate_in_kernel:
        g = 0.5 * torch.randn((*token_shape, HV, K), device=device)
    else:
        g = F.logsigmoid(torch.randn((*token_shape, HV, K), device=device) + 3.0)
    beta = torch.randn((*token_shape, HV), device=device)
    if not args.beta_is_logit:
        beta = torch.sigmoid(beta)
    q, k, v, g, beta = (t.to(torch.bfloat16) for t in (q, k, v, g, beta))
    A_log = torch.log(torch.rand(HV, device=device) + 1.0)
    dt_bias = 0.1 * torch.randn(HV * K, device=device)

    cu_seqlens = ssm_state_indices = num_accepted_tokens = None
    if spec:
        cu_seqlens = torch.arange(0, N * T + 1, T, dtype=torch.int32, device=device)
        ssm_state_indices = torch.arange(
            1, N * T + 1, dtype=torch.int32, device=device
        ).reshape(N, T)
        num_accepted_tokens = torch.ones(N, dtype=torch.int32, device=device)
        slots = N * T + 2
    elif args.use_state_indices:
        slots = args.state_pool_slots or N + 2
        gen = torch.Generator(device="cpu").manual_seed(args.random_seed)
        ssm_state_indices = (
            (torch.randperm(slots - 1, generator=gen)[:N] + 1)
            .to(torch.int32)
            .to(device)
        )
    else:
        slots = N
    state = (0.05 * torch.randn((slots, HV, K, K), device=device)).to(state_dtype)
    snapshot = state.clone()
    output = torch.empty_like(v)

    def restore():
        state.copy_(snapshot)

    if spec:
        init_slots = ssm_state_indices[:, 0].long()
    elif ssm_state_indices is not None:
        init_slots = ssm_state_indices.long()
    else:
        init_slots = torch.arange(N, device=device)
    reference = None
    if args.refcheck:
        flat = lambda t: t.reshape(-1, *t.shape[2:])  # noqa: E731
        log_gate = _log_gate(
            flat(g), A_log, dt_bias, args.use_gate_in_kernel, args.lower_bound
        )
        bp = flat(beta).float()
        ref, _ = _reference_recurrence(
            flat(q),
            flat(k),
            flat(v),
            log_gate,
            torch.sigmoid(bp) if args.beta_is_logit else bp,
            [i * T for i in range(N + 1)],
            snapshot[init_slots],
            state_dtype,
        )
        reference = ref.reshape(v.shape)

    api_backend = {"flashinfer": "auto", "cute-dsl": "cute-dsl"}
    total_tokens = N * T
    flops = _recurrence_flops(total_tokens, HV, K)
    nbytes = _recurrence_bytes(total_tokens, H, HV, K, state_dtype, N * (1 + T))

    for backend in backends:

        def call(api=api_backend[backend]):
            flashinfer.recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=A_log if args.use_gate_in_kernel else None,
                dt_bias=dt_bias if args.use_gate_in_kernel else None,
                initial_state=state,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=args.use_gate_in_kernel,
                lower_bound=args.lower_bound if args.use_gate_in_kernel else None,
                beta_is_logit=args.beta_is_logit,
                cu_seqlens=cu_seqlens,
                ssm_state_indices=ssm_state_indices,
                num_spec_tokens=S if spec else None,
                num_accepted_tokens=num_accepted_tokens,
                output=output,
                backend=api,
            )
            return output

        times, route, captured = _warm_and_time(
            args, backend, call, restore, "cute-dsl"
        )
        if not times:
            continue
        if reference is not None:
            _refcheck(args, backend, reference, captured)
        row = _emit(
            args,
            backend,
            times,
            flops,
            nbytes,
            {"resolved_backend": route, "kda_variant": route, "state_pool": slots},
        )
        if row is not None:
            res.append(row)
    return res


# ---------------------------------------------------------------------------
# recurrent_kda_output_only (disable_state_update=True)
# ---------------------------------------------------------------------------


def testRecurrentKDAOutputOnly(args):
    """Time frozen / speculative-verify KDA: outputs for T<=16 tokens from a
    committed state pool, no state write-back.

    Batched ``[B, T, ...]`` goes to the output-only dispatcher (WY tensor-core
    kernel or grouped register recurrence); ``--verify_caches`` adds the
    slot-indexed correction/kg caches (packed frozen-verify kernel).
    """
    device = _preamble(args, "testRecurrentKDAOutputOnly")
    res = []
    backends = _select_backends(args, device)
    if not backends:
        print("[ERROR] No backends to test. Exiting.")
        return res
    B, T, H, HV, K = (
        args.batch_size,
        args.s_qo,
        args.num_q_heads,
        args.num_v_heads,
        HEAD_DIM,
    )
    if not 1 <= T <= 16:
        print("[ERROR] output-only mode takes 1 <= --s_qo <= 16. Exiting.")
        return res
    q = F.normalize(torch.randn((B, T, H, K), device=device), dim=-1)
    k = F.normalize(torch.randn((B, T, H, K), device=device), dim=-1)
    v = torch.randn((B, T, HV, K), device=device)
    if args.use_gate_in_kernel:
        g = 0.5 * torch.randn((B, T, HV, K), device=device)
    else:
        g = F.logsigmoid(torch.randn((B, T, HV, K), device=device) + 3.0)
    beta = torch.randn((B, T, HV), device=device)
    if not args.beta_is_logit:
        beta = torch.sigmoid(beta)
    q, k, v, g, beta = (t.to(torch.bfloat16) for t in (q, k, v, g, beta))
    A_log = torch.log(torch.rand(HV, device=device) + 1.0)
    dt_bias = 0.1 * torch.randn(HV * K, device=device)
    slots = args.state_pool_slots or B + 2
    gen = torch.Generator(device="cpu").manual_seed(args.random_seed)
    indices = (
        (torch.randperm(slots - 1, generator=gen)[:B] + 1).to(torch.int32).to(device)
    )
    pool = (0.05 * torch.randn((slots, HV, K, K), device=device)).to(torch.bfloat16)
    output = torch.empty_like(v)
    caches = {}
    if args.verify_caches:
        caches["correction_cache"] = torch.zeros(
            (slots, HV, T, K), dtype=torch.float32, device=device
        )
        caches["kg_cache"] = torch.zeros(
            (slots, HV, T, 2 * K), dtype=torch.bfloat16, device=device
        )

    reference = None
    if args.refcheck:
        flat = lambda t: t.reshape(-1, *t.shape[2:])  # noqa: E731
        bp = flat(beta).float()
        ref, _ = _reference_recurrence(
            flat(q),
            flat(k),
            flat(v),
            _log_gate(
                flat(g), A_log, dt_bias, args.use_gate_in_kernel, args.lower_bound
            ),
            torch.sigmoid(bp) if args.beta_is_logit else bp,
            [i * T for i in range(B + 1)],
            pool[indices.long()],
            torch.bfloat16,
        )
        reference = ref.reshape(v.shape)

    flops = _recurrence_flops(B * T, HV, K)
    nbytes = _recurrence_bytes(B * T, H, HV, K, torch.bfloat16, B)

    for backend in backends:

        def call():
            out, _ = flashinfer.recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=A_log if args.use_gate_in_kernel else None,
                dt_bias=dt_bias if args.use_gate_in_kernel else None,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=args.use_gate_in_kernel,
                lower_bound=args.lower_bound if args.use_gate_in_kernel else None,
                beta_is_logit=args.beta_is_logit,
                initial_state_source=pool,
                initial_state_indices=indices,
                output=output,
                disable_state_update=True,
                backend="cute-dsl",
                **caches,
            )
            return out

        times, route, captured = _warm_and_time(
            args, backend, call, lambda: None, "cute-dsl-frozen"
        )
        if not times:
            continue
        if reference is not None:
            _refcheck(args, backend, reference, captured)
        row = _emit(
            args,
            backend,
            times,
            flops,
            nbytes,
            {"resolved_backend": route, "kda_variant": route, "state_pool": slots},
        )
        if row is not None:
            res.append(row)
    return res


# ---------------------------------------------------------------------------
# fused_kda_decode / packed_fused_kda_decode (conv + KDA + gated RMSNorm)
# ---------------------------------------------------------------------------


def _page_strides(num_heads, state_dtype, conv_history):
    conv_slot_bytes = 3 * num_heads * 128 * conv_history * 2
    element = torch.empty((), dtype=state_dtype).element_size()
    page_bytes = conv_slot_bytes + num_heads * 128 * 128 * element
    return page_bytes // 2, page_bytes // element


def _make_fused_inputs(num_heads, num_rows, device, state_dtype, conv_history):
    """Paged conv/state caches sharing one page per slot (serving layout)."""
    num_slots = num_rows + 1
    hidden = num_heads * 128
    conv_stride, state_stride = _page_strides(num_heads, state_dtype, conv_history)
    conv_state = torch.empty_strided(
        (num_slots, 3 * hidden, conv_history),
        (conv_stride, 1, 3 * hidden),
        dtype=torch.bfloat16,
        device=device,
    )
    conv_state.copy_(
        0.1 * torch.randn((num_slots, 3 * hidden, conv_history), device=device)
    )
    state = torch.empty_strided(
        (num_slots, num_heads, 128, 128),
        (state_stride, 128 * 128, 128, 1),
        dtype=state_dtype,
        device=device,
    )
    state.copy_(0.01 * torch.randn((num_slots, num_heads, 128, 128), device=device))
    return {
        "x": torch.randn((num_rows, 3 * hidden), device=device).to(torch.bfloat16),
        "weight": 0.1 * torch.randn((3, 4, hidden), device=device),
        "conv_state": conv_state,
        "raw_gate": torch.randn((1, num_rows, num_heads, 128), device=device).to(
            torch.bfloat16
        ),
        "raw_beta": torch.randn((1, num_rows, num_heads), device=device).to(
            torch.bfloat16
        ),
        "A_log": 0.5 * torch.randn((num_heads,), device=device),
        "dt_bias": 0.1 * torch.randn((hidden,), device=device),
        "state_indices": torch.arange(
            num_rows, 0, -1, dtype=torch.int32, device=device
        ),
        "state": state,
        "output_gate": torch.randn((num_rows, num_heads, 128), device=device).to(
            torch.bfloat16
        ),
        "norm_weight": torch.randn((128,), device=device),
    }


def _clone_strided(tensor):
    clone = torch.empty_strided(
        tensor.shape, tensor.stride(), dtype=tensor.dtype, device=tensor.device
    )
    clone.copy_(tensor)
    return clone


@torch.no_grad()
def _fused_reference(inputs, conv_state, state, lower_bound):
    """conv(4)+SiLU -> one KDA step -> gated RMSNorm, per decode row (T=1)."""
    x = inputs["x"]
    num_rows = x.shape[0]
    num_heads = inputs["A_log"].numel()
    hidden = num_heads * 128
    slots = inputs["state_indices"].long()
    taps = inputs["weight"].permute(0, 2, 1).reshape(3 * hidden, 4)
    history = conv_state.index_select(0, slots).float()
    window = torch.cat((history, x.float().unsqueeze(-1)), dim=-1)
    mixed = F.silu((window * taps).sum(-1)).to(torch.bfloat16).float()
    query, key, value = mixed.view(num_rows, 3, num_heads, 128).unbind(1)
    query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
    query = query * 128**-0.5
    key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    gate = inputs["raw_gate"][0].float() + inputs["dt_bias"].view(num_heads, 128)
    a = inputs["A_log"].exp()[None, :, None]
    if lower_bound is None:
        gate = -a * F.softplus(gate)
    else:
        gate = lower_bound * torch.sigmoid(a * gate)
    s = state.index_select(0, slots).float() * gate.exp().unsqueeze(-2)
    delta = (value - torch.einsum("nhvk,nhk->nhv", s, key)) * inputs["raw_beta"][
        0
    ].float().sigmoid().unsqueeze(-1)
    s = s + delta.unsqueeze(-1) * key.unsqueeze(-2)
    out = torch.einsum("nhvk,nhk->nhv", s, query).to(torch.bfloat16).float()
    inv = torch.rsqrt(out.square().mean(-1, keepdim=True) + 1e-5)
    out = out * inv * inputs["norm_weight"] * inputs["output_gate"].float().sigmoid()
    return out.unsqueeze(0).to(torch.bfloat16)


@torch.no_grad()
def _packed_fused_reference(inputs, conv_state, state, lower_bound):
    """Multi-token packed fused reference (accepted=1 source, per-token
    checkpoints). Mirrors tests/kda/test_fused_kda_decode.py."""
    indices = inputs["state_indices"]
    qsl = inputs["query_start_loc"].cpu().tolist()
    accepted = inputs["num_accepted_tokens"].cpu().tolist()
    num_seqs = indices.shape[0]
    num_heads = inputs["A_log"].numel()
    hidden = num_heads * 128
    taps = inputs["weight"].permute(0, 2, 1).reshape(3 * hidden, 4).float()
    output = torch.zeros((1, inputs["x"].shape[0], num_heads, 128), device=state.device)
    a = inputs["A_log"].exp().view(num_heads, 1)
    for n in range(num_seqs):
        begin, end = qsl[n : n + 2]
        conv_slot = int(indices[n, 0])
        if end <= begin or conv_slot <= 0:
            continue
        acc = accepted[n] - 1
        history = conv_state[conv_slot, :, acc : acc + 3].float()
        s = state[int(indices[n, acc])].float().clone()
        for t in range(end - begin):
            row = begin + t
            window = torch.cat((history, inputs["x"][row].float().unsqueeze(-1)), -1)
            history = window[:, 1:]
            mixed = F.silu((window * taps).sum(-1)).to(torch.bfloat16)
            q_, k_, v_ = mixed.float().view(3, num_heads, 128).unbind(0)
            q_ = q_ * torch.rsqrt(q_.square().sum(-1, keepdim=True) + 1e-6) * 128**-0.5
            k_ = k_ * torch.rsqrt(k_.square().sum(-1, keepdim=True) + 1e-6)
            raw = inputs["raw_gate"][0, row].float() + inputs["dt_bias"].view(
                num_heads, 128
            )
            if lower_bound is None:
                decay = torch.exp(-a * F.softplus(raw))
            else:
                decay = torch.exp(lower_bound * torch.sigmoid(a * raw))
            s = s * decay.unsqueeze(-2)
            delta = (v_ - torch.einsum("hvk,hk->hv", s, k_)) * inputs["raw_beta"][
                0, row
            ].float().sigmoid().unsqueeze(-1)
            s = s + delta.unsqueeze(-1) * k_.unsqueeze(-2)
            output[0, row] = torch.einsum("hvk,hk->hv", s, q_).to(torch.bfloat16)
    inv = torch.rsqrt(output.square().mean(-1, keepdim=True) + 1e-5)
    return (
        output
        * inv
        * inputs["norm_weight"]
        * inputs["output_gate"].float().sigmoid().reshape(1, -1, num_heads, 128)
    ).to(torch.bfloat16)


def _fused_bytes(num_rows, num_heads, state_dtype, conv_history, num_seqs=None):
    """Activations per row; per sequence one conv window read + written and one
    state read; one state written per row. ``num_seqs`` (default ``num_rows``)
    is the sequence count of a packed multi-token call."""
    num_seqs = num_rows if num_seqs is None else num_seqs
    hidden = num_heads * 128
    act = 2 * num_rows * (3 * hidden + hidden + num_heads + hidden + hidden)
    conv = 2 * 2 * num_seqs * 3 * hidden * conv_history
    state_slabs = num_seqs + num_rows
    state = state_slabs * num_heads * 128 * 128 * _state_bytes(state_dtype)
    return act + conv + state


def testFusedKDADecode(args):
    """Time ``flashinfer.fused_kda_decode`` (conv + KDA + gated RMSNorm, T=1)."""
    device = _preamble(args, "testFusedKDADecode")
    res = []
    backends = _select_backends(args, device)
    if not backends:
        print("[ERROR] No backends to test. Exiting.")
        return res
    H, rows = args.num_q_heads, args.batch_size
    if H not in _FUSED_KDA_HEADS or args.num_v_heads != H:
        print(f"[ERROR] fused_kda_decode takes H in {_FUSED_KDA_HEADS}, HV == H.")
        return res
    state_dtype = dtype_str_to_torch_dtype(args.state_dtype)
    inputs = _make_fused_inputs(H, rows, device, state_dtype, 3)
    conv_snap = _clone_strided(inputs["conv_state"])
    state_snap = _clone_strided(inputs["state"])

    def restore():
        inputs["conv_state"].copy_(conv_snap)
        inputs["state"].copy_(state_snap)

    reference = None
    if args.refcheck:
        reference = _fused_reference(
            inputs,
            _clone_strided(conv_snap),
            _clone_strided(state_snap),
            args.lower_bound,
        )
    output = torch.empty((1, rows, H, 128), dtype=torch.bfloat16, device=device)
    flops = _recurrence_flops(rows, H, 128)
    nbytes = _fused_bytes(rows, H, state_dtype, 3)
    for backend in backends:

        def call():
            flashinfer.fused_kda_decode(
                **inputs,
                lower_bound=args.lower_bound,
                output=output,
                backend="cute-dsl",
            )
            return output

        times, route, captured = _warm_and_time(
            args, backend, call, restore, "cute-dsl"
        )
        if not times:
            continue
        if reference is not None:
            _refcheck(args, backend, reference, captured)
        row = _emit(
            args,
            backend,
            times,
            flops,
            nbytes,
            {
                "resolved_backend": route,
                "kda_variant": route,
            },
        )
        if row is not None:
            res.append(row)
    return res


def testPackedFusedKDADecode(args):
    """Time ``flashinfer.packed_fused_kda_decode``: MTP with T = 1+S tokens per
    sequence, per-token recurrent checkpoints and a T+2 rolling conv cache."""
    device = _preamble(args, "testPackedFusedKDADecode")
    res = []
    backends = _select_backends(args, device)
    if not backends:
        print("[ERROR] No backends to test. Exiting.")
        return res
    H, N, T = args.num_q_heads, args.batch_size, 1 + args.num_spec_tokens
    heads = _FUSED_KDA_HEADS if T == 1 else _PACKED_FUSED_MTP_HEADS
    if H not in heads or args.num_v_heads != H:
        print(f"[ERROR] packed_fused_kda_decode at T={T} takes H in {heads}.")
        return res
    state_dtype = dtype_str_to_torch_dtype(args.state_dtype)
    conv_history = T + 2 if T > 1 else 3
    inputs = _make_fused_inputs(H, N * T, device, state_dtype, conv_history)
    inputs["state_indices"] = inputs["state_indices"].reshape(N, T)
    inputs["query_start_loc"] = torch.arange(
        0, N * T + 1, T, dtype=torch.int32, device=device
    )
    inputs["num_accepted_tokens"] = torch.ones(N, dtype=torch.int32, device=device)
    conv_snap = _clone_strided(inputs["conv_state"])
    state_snap = _clone_strided(inputs["state"])

    def restore():
        inputs["conv_state"].copy_(conv_snap)
        inputs["state"].copy_(state_snap)

    reference = None
    if args.refcheck:
        reference = _packed_fused_reference(
            inputs,
            _clone_strided(conv_snap),
            _clone_strided(state_snap),
            args.lower_bound,
        )
    output = torch.empty((1, N * T, H, 128), dtype=torch.bfloat16, device=device)
    flops = _recurrence_flops(N * T, H, 128)
    nbytes = _fused_bytes(N * T, H, state_dtype, conv_history, num_seqs=N)
    for backend in backends:

        def call():
            flashinfer.packed_fused_kda_decode(
                **inputs, lower_bound=args.lower_bound, output=output
            )
            return output

        times, route, captured = _warm_and_time(
            args, backend, call, restore, "cute-dsl"
        )
        if not times:
            continue
        if reference is not None:
            _refcheck(args, backend, reference, captured)
        row = _emit(
            args,
            backend,
            times,
            flops,
            nbytes,
            {
                "resolved_backend": route,
                "kda_variant": "cute-dsl-t1" if T == 1 else "cute-dsl-multitoken",
            },
        )
        if row is not None:
            res.append(row)
    return res
