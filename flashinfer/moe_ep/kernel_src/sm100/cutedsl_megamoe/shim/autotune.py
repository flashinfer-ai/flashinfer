# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Online (warmup-time) knob autotuning for the CuTeDSL MegaMoE frontends.

Times a curated candidate knob set on the live problem and applies the winner
to the session's frontend, replacing the static two-profile heuristic in
:mod:`.tuner` with a measured choice.  The candidate space mirrors the
restricted sweep used with the kernel team's tester
(``tester.tester --sweep --use_knob ...``).  Every dtype includes
``in_kernel_fc2_reduce`` when the session permits it (the tester's overall
NVFP4 winners at 8 and 2048 tokens are in-flight-reduce candidates): each
session's combine destinations are all sym-heap allocated up front, so the
knob can flip per-compile.  Note an ikr winner makes the session's output
nondeterministic in accumulation order; callers that need bit-reproducible
outputs should leave ``enable_in_kernel_fc2_reduce=False``.

The tune is a COLLECTIVE operation: the mega kernel's dispatch/combine spans
all EP ranks, so every rank must call the autotune entry point in the same
iteration with the same candidate list.  Ranks compile and launch each
candidate in lockstep (barriers around compile and timing), and the winner is
agreed on by all-reducing per-candidate times with MAX (the slowest rank is
the real latency of a collective kernel) — the argmin index is then identical
everywhere.

Cost: one ``cute.compile`` per candidate (minutes each), paid once per
session at the first launch.  Narrow ``candidates`` to trade quality for
startup time.
"""

from __future__ import annotations

import math
import statistics
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from flashinfer.moe_ep.backends.mega.kernel.sm100.common.activation_batch_invariant import (
    batch_invariant_variant_kwargs,
)

from .tuner import (
    default_knobs,
    describe_invalid_knobs,
    is_valid,
    is_valid_bf16,
    is_valid_bf16_for_config,
    is_valid_bf16_mxfp8,
    is_valid_bf16_mxfp8_for_config,
    is_valid_nvfp4_for_config,
    is_valid_mxfp8_for_config,
)

# Shared base of the sweep restriction (values that won every profile so far).
_SWEEP_BASE: Dict[str, Any] = {
    "cluster_shape_mnk": (2, 1, 1),
    "group_hint": 512,
    "epi_flag_batch": (2, 4),
    "load_balance_mode": "atomic_counter",
}


def nvfp4_candidates(
    *,
    combine_format: str = "bf16",
    enable_in_kernel_fc2_reduce: bool = False,
    batch_invariant: bool = False,
) -> List[Dict[str, Any]]:
    """Default NVFP4 candidate knob dicts (tile x flag_batch x token-back x ikr).

    12 candidates for the default bf16 combine, doubled to 24 (and with them
    the one-time compile cost) when ``enable_in_kernel_fc2_reduce`` adds the
    ikr axis; quantized ``combine_format`` values prune to the valid subset
    (dispatch-warp token-back only, no ikr).  Leave the permission unset when
    the session cannot run ikr (``apply_routing_weights_before_fc2=False``) or
    must stay deterministic.
    """
    if batch_invariant:
        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.nvfp4_batch_invariant import (
            nvfp4_batch_invariant_default_knobs,
        )

        if enable_in_kernel_fc2_reduce:
            raise ValueError("batch_invariant NVFP4 requires no in-kernel reduction")
        base = nvfp4_batch_invariant_default_knobs("legacy")
        return [
            dict(base, token_back_mode=carrier, load_balance_mode=balance)
            for carrier in (
                ("epi_warps", "standalone_warps", "reuse_dispatch_warps")
                if combine_format == "bf16"
                else ("reuse_dispatch_warps",)
            )
            for balance in ("static", "atomic_counter")
        ]
    out: List[Dict[str, Any]] = []
    for tile in ((256, 128, 256), (256, 256, 256)):
        for flag_batch in (4, 8):
            for token_back in (
                "epi_warps",
                "standalone_warps",
                "reuse_dispatch_warps",
            ):
                for ikr in (False, True) if enable_in_kernel_fc2_reduce else (False,):
                    knobs = dict(
                        _SWEEP_BASE,
                        mma_tiler_mnk=tile,
                        flag_batch=flag_batch,
                        token_back_mode=token_back,
                        in_kernel_fc2_reduce=ikr,
                    )
                    if is_valid(knobs, combine_format=combine_format):
                        out.append(knobs)
    return out


def mxfp8_candidates(
    *,
    enable_in_kernel_fc2_reduce: bool = False,
    batch_invariant: bool = False,
) -> List[Dict[str, Any]]:
    """Default MXFP8 candidate knob dicts (flag_batch x token-back x ikr).

    Four candidates, six with the ikr axis.  The MXFP8 kernel's tile is fixed
    at ``(256, 256)`` so there is no tile axis. ikr is not supported with reuse dispatch warps.
    """
    if batch_invariant:
        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.mxfp8_batch_invariant import (
            mxfp8_batch_invariant_default_knobs,
        )

        base = mxfp8_batch_invariant_default_knobs("legacy")
        return [
            {**base, "load_balance_mode": schedule, "token_back_mode": carrier}
            for schedule in ("static", "atomic_counter")
            for carrier in ("epi_warps", "reuse_dispatch_warps")
        ]
    out: List[Dict[str, Any]] = []
    for flag_batch in (4, 8):
        for token_back in ("epi_warps", "reuse_dispatch_warps"):
            for ikr in (False, True) if enable_in_kernel_fc2_reduce else (False,):
                if ikr and token_back != "epi_warps":
                    continue
                knobs = dict(
                    _SWEEP_BASE,
                    flag_batch=flag_batch,
                    token_back_mode=token_back,
                    in_kernel_fc2_reduce=ikr,
                )
                if is_valid(knobs):
                    out.append(knobs)
    return out


def bf16_candidates(
    *,
    enable_in_kernel_fc2_reduce: bool = False,
    batch_invariant: bool = False,
) -> List[Dict[str, Any]]:
    """Return supported BF16 candidates for a session's output contract.

    Ordinary sessions keep their baseline geometry. BI sessions sweep the
    admitted geometry/schedule/carrier product; the final session filter checks
    canonical K padding and resource feasibility before any launch.
    """
    if batch_invariant:
        import dataclasses

        from flashinfer.moe_ep.batch_invariant.spec import (
            ALLOWED_PLAN_CONFIGS,
            ALLOWED_PLANS,
            PlanConfig,
        )

        baseline = PlanConfig()
        geometries = (baseline,) + tuple(
            geometry for geometry in ALLOWED_PLAN_CONFIGS if geometry != baseline
        )
        base = default_knobs(0, dtype="bf16")
        return [
            {
                **base,
                **dataclasses.asdict(geometry),
                "load_balance_mode": schedule,
                "token_back_mode": carrier,
            }
            for geometry in geometries
            for schedule, carrier in ALLOWED_PLANS
            if is_valid_bf16({**base, **dataclasses.asdict(geometry)})
        ]
    out: List[Dict[str, Any]] = []
    for ikr in (
        (False, True)
        if enable_in_kernel_fc2_reduce and not batch_invariant
        else (False,)
    ):
        knobs = default_knobs(0, dtype="bf16", enable_in_kernel_fc2_reduce=ikr)
        if is_valid_bf16(knobs):
            out.append(knobs)
    return out


def _bf16_mxfp8_epi_flag_batches(token_back: str) -> Tuple[Tuple[int, int], ...]:
    """``epi_flag_batch`` values worth timing for a token-back mode.
    Emperically we found that epi_flag_batch shouldnt matter much for epi_warps mode.
    """
    if token_back == "epi_warps":
        return ((2, 4),)
    return ((1, 1), (2, 4))


def bf16_mxfp8_candidates(
    *,
    enable_in_kernel_fc2_reduce: bool = False,
    batch_invariant: bool = False,
) -> List[Dict[str, Any]]:
    """Default BF16/MXFP8 candidate knob dicts (ikr x impl tuple x flag_batch x token-back).

    Eighteen candidates: three legal implementation tuples (N128/tmem,
    N256/smem, N256/tmem-overlap) x ``flag_batch`` {1, 4} x
    ``token_back_mode`` {``epi_warps``, ``reuse_dispatch_warps``}, with the
    dispatch-warp half also timing ``epi_flag_batch=(1, 1)`` (see
    :func:`_bf16_mxfp8_epi_flag_batches`).  ``standalone_warps`` is
    unsupported.
    Enabling IKR doubles the number of candidates to 36.
    """
    if batch_invariant:
        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.a16_batch_invariant import (
            a16_batch_invariant_default_knobs,
        )

        bi_base = a16_batch_invariant_default_knobs("bf16_mxfp8_e4m3")
        return [
            {**bi_base, "load_balance_mode": schedule, "token_back_mode": carrier}
            for schedule in ("static", "atomic_counter")
            for carrier in ("epi_warps", "reuse_dispatch_warps")
        ]
    out: List[Dict[str, Any]] = []
    impl_specs = (
        {
            "mma_tiler_mnk": (256, 128, 128),
            "transform_buffer": "tmem",
            "accumulator_overlap": False,
            "transform_k_tile": 128,
        },
        {
            "mma_tiler_mnk": (256, 256, 128),
            "transform_buffer": "smem",
            "accumulator_overlap": False,
            "transform_k_tile": 128,
        },
        {
            "mma_tiler_mnk": (256, 256, 128),
            "transform_buffer": "tmem",
            "accumulator_overlap": True,
            "transform_k_tile": 64,
        },
    )
    base: Dict[str, Any] = {
        "cluster_shape_mnk": (2, 1, 1),
        "use_2cta_instrs": True,
        "group_hint": 512,
        "load_balance_mode": "static",
    }
    for ikr in (False, True) if enable_in_kernel_fc2_reduce else (False,):
        for impl in impl_specs:
            for flag_batch in (1, 4):
                for token_back in ("epi_warps", "reuse_dispatch_warps"):
                    for epi_flag_batch in _bf16_mxfp8_epi_flag_batches(token_back):
                        knobs = dict(
                            base,
                            **impl,
                            flag_batch=flag_batch,
                            epi_flag_batch=epi_flag_batch,
                            token_back_mode=token_back,
                            in_kernel_fc2_reduce=ikr,
                        )
                        if is_valid_bf16_mxfp8(knobs):
                            out.append(knobs)
    return out


def _session_candidates(
    candidates: List[Dict[str, Any]],
    config: Any,
    predicate: Callable[[Any, Dict[str, Any]], bool],
    *,
    what: str,
) -> List[Dict[str, Any]]:
    """Drop candidates a session cannot run (e.g. ikr without permission).

    Every drop is reported with the offending knob, so a caller never loses a
    pinned value silently.  The filter is a pure function of the config and the
    candidate list, both identical on every rank, so the surviving order stays
    in collective lockstep (see :func:`autotune_knobs`).
    """
    kept: List[Dict[str, Any]] = []
    dropped: List[str] = []
    for knobs in candidates:
        if predicate(config, knobs):
            kept.append(knobs)
        else:
            dropped.append(
                f"    {knobs}\n      -> {describe_invalid_knobs(config, knobs, predicate)}"
            )
    if not kept:
        detail = ("; all were rejected:\n" + "\n".join(dropped)) if dropped else ""
        raise ValueError(
            f"no valid {what} autotune candidates for this session{detail}"
        )
    if dropped:
        warnings.warn(
            f"[cutedsl-autotune] {what}: dropped {len(dropped)}/{len(candidates)} "
            f"candidate(s) this session cannot run:\n" + "\n".join(dropped),
            RuntimeWarning,
            stacklevel=3,
        )
    return kept


def _apply_config_overrides(frontend, overrides):
    """Apply numerical arguments to an unallocated trial, or a single-rank session."""
    import dataclasses

    values = dict(overrides)
    if "gate_up_clamp" in values and hasattr(frontend, "set_gate_up_clamp"):
        frontend.set_gate_up_clamp(values.pop("gate_up_clamp"))
    if values:
        config = dataclasses.replace(frontend.config, **values)
        if config != frontend.config:
            frontend.release()
            frontend._config = config


def _validate_tuning_output(y, frontend, num_tokens):
    config = frontend.config
    n = config.num_tokens_per_rank if num_tokens is None else num_tokens
    if type(n) is not int or not 0 <= n <= config.num_tokens_per_rank:
        raise ValueError("num_tokens must be within the workspace capacity")
    if y is not None and (
        y.dtype != torch.bfloat16 or tuple(y.shape) != (n, config.hidden)
    ):
        raise ValueError(f"y must be bfloat16 with shape ({n}, {config.hidden})")


def _tuning_specification(value):
    """Compare metadata exactly, including NaN payloads and signed zero."""
    import struct

    if type(value) is float:
        return ("float", struct.pack(">d", value))
    if isinstance(value, dict):
        return tuple(
            (key, _tuning_specification(item)) for key, item in sorted(value.items())
        )
    if isinstance(value, (tuple, list)):
        return (
            type(value).__name__,
            tuple(_tuning_specification(item) for item in value),
        )
    return value


def autotune_knobs(
    frontend: Any,
    launch: Callable[[], None],
    candidates: List[Dict[str, Any]],
    *,
    label: str,
    warmup_iters: int = 3,
    timed_iters: int = 10,
    on_winner: Optional[Callable[[Dict[str, Any], float], None]] = None,
    prepare_launch: Optional[Callable[[], Callable[[], None]]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    candidate_filter: Optional[
        Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]
    ] = None,
    process_group: Any = None,
    measure_launch: Optional[Callable[[Callable[[], None]], float]] = None,
) -> Dict[str, Any]:
    """Time each candidate on the live problem and apply the winner.

    ``frontend`` is a NVFP4/MXFP8 mega frontend (must have ``apply_knobs``);
    ``launch`` is a zero-arg closure that runs one synchronized forward with
    the caller's real staged inputs (e.g. a ``nvfp4_mega_moe(...)`` call).
    ``prepare_launch``, when supplied, builds that callable after each knob
    change, outside the timed interval.
    ``candidate_filter`` runs after agreement on the unfiltered proposal;
    filtering errors and the surviving candidate order are agreed collectively.

    ``on_winner`` (optional) is called once with ``(winner, p50_seconds)``
    after the winner is applied — used to persist the result in the knob
    cache. It runs on every rank; the callback decides who writes.

    COLLECTIVE: every EP rank must call this in the same iteration with the
    same ``candidates`` (order included).  Returns the winning knob dict.
    """
    from .comm import ensure_not_capturing

    # The sweep barriers, compiles per candidate, wall-clock times with
    # internal syncs, and all_reduces -- none of it can run mid-capture.
    ensure_not_capturing("knobs='auto' collective autotune sweep")

    import torch.distributed as dist

    collective = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank(group=process_group) if collective else 0
    world_size = dist.get_world_size(group=process_group) if collective else 1
    multi_rank = world_size > 1

    def phase(name, prepare):
        result, error = None, None
        try:
            result = prepare()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = [error]
        if multi_rank:
            errors = [None] * world_size
            dist.all_gather_object(errors, error, group=process_group)
        if any(errors):
            raise RuntimeError(
                f"[cutedsl-autotune] {label}: {name} failed on EP ranks: {errors}. "
                "The collective sweep is aborted; do not reuse partially prepared "
                "storage. Restart the EP communicator/job after a collective failure."
            )
        return result

    def preflight():
        if not candidates:
            raise ValueError("autotune_knobs needs a non-empty candidate list.")
        if (
            type(warmup_iters) is not int
            or warmup_iters < 0
            or type(timed_iters) is not int
            or timed_iters < 1
        ):
            raise ValueError(
                "warmup_iters must be nonnegative and timed_iters positive integers"
            )
        if multi_rank and (
            not callable(getattr(frontend, "prepare_tuning_local", None))
            or not callable(getattr(frontend, "prepare_tuning_collective", None))
            or prepare_launch is None
        ):
            raise ValueError(
                "collective autotune requires local/storage preparation and a compile-only prepare_launch hook"
            )
        cfg = getattr(frontend, "config", None)
        if multi_rank and getattr(cfg, "world_size", world_size) != world_size:
            raise ValueError(
                "legacy autotune requires the process group to match the EP world"
            )

    if multi_rank:
        phase("preflight", preflight)
        import dataclasses

        cfg = getattr(frontend, "config", None)
        config = (
            dataclasses.asdict(cfg)
            if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type)
            else None
        )
        if config is not None:
            config.pop("rank", None)
        proposal = _tuning_specification(
            (
                label,
                candidates,
                warmup_iters,
                timed_iters,
                config_overrides,
                candidate_filter is not None,
                config,
            )
        )
        proposals = [None] * world_size
        dist.all_gather_object(proposals, proposal, group=process_group)
        if any(p != proposal for p in proposals):
            raise ValueError(
                "all EP ranks must use the same legacy autotune specification"
            )
    else:
        preflight()

    if candidate_filter is not None:
        if multi_rank:
            candidates = phase(
                "candidate filtering", lambda: candidate_filter(candidates)
            )
            admitted = _tuning_specification(candidates)
            admitted_proposals = [None] * world_size
            dist.all_gather_object(admitted_proposals, admitted, group=process_group)
            if any(proposal != admitted for proposal in admitted_proposals):
                raise ValueError(
                    "all EP ranks must admit the same legacy autotune candidates"
                )
        else:
            candidates = candidate_filter(candidates)

    def _barrier() -> None:
        if collective:
            dist.barrier(group=process_group)

    def install(knobs):
        if (
            callable(getattr(frontend, "prepare_tuning_local", None))
            and prepare_launch is not None
        ):
            prepared = phase(
                "local preparation",
                lambda: frontend.prepare_tuning_local(
                    knobs, config_overrides=config_overrides
                )
                if config_overrides
                else frontend.prepare_tuning_local(knobs),
            )
            if multi_rank:
                pending = getattr(prepared, "_compile_prepare", None)
                storage = (
                    (
                        pending["shared_bytes"],
                        getattr(frontend, "_mega", None) is not None,
                    )
                    if pending is not None
                    else None
                )
                storages = [None] * world_size
                dist.all_gather_object(storages, storage, group=process_group)
                if any(s != storage for s in storages):
                    raise RuntimeError(
                        "EP ranks disagree on symmetric storage size or lifetime; "
                        "abort the sweep before collective allocation/free"
                    )
            release = getattr(frontend, "prepare_tuning_release", None)
            allocate = getattr(frontend, "prepare_tuning_allocate_local", None)
            if callable(release) and callable(allocate):
                phase("collective storage release", lambda: release(prepared))
                phase("local storage allocation", lambda: allocate(prepared))
            phase(
                "collective storage preparation",
                lambda: frontend.prepare_tuning_collective(prepared),
            )

            def compile_launch():
                ready = prepare_launch()
                if not callable(ready):
                    raise TypeError("prepare_launch must return a prepared callable")
                return ready

            return phase("compilation/launch preparation", compile_launch)
        frontend.apply_knobs(knobs)
        if config_overrides:
            _apply_config_overrides(frontend, config_overrides)
        return prepare_launch() if prepare_launch is not None else launch

    def measure(candidate_launch):
        if measure_launch is not None:
            return measure_launch(candidate_launch)
        for _ in range(warmup_iters):
            candidate_launch()
        _barrier()
        iters = []
        for _ in range(timed_iters):
            t0 = time.perf_counter()
            candidate_launch()
            iters.append(time.perf_counter() - t0)
        return statistics.median(iters)

    scores: List[float] = []
    for knobs in candidates:
        candidate_launch = None
        if multi_rank:
            # No rank enters a communication kernel before every peer has
            # completed local allocation, symmetric allocation and compilation.
            candidate_launch = install(knobs)
            try:
                scores.append(measure(candidate_launch))
            except Exception as exc:
                raise RuntimeError(
                    f"[cutedsl-autotune] {label}: collective candidate execution failed; "
                    "abort the EP job instead of continuing with possibly dirty storage"
                ) from exc
            finally:
                candidate_launch = None
            continue
        try:
            candidate_launch = install(knobs)
            scores.append(measure(candidate_launch))
        except Exception as exc:  # noqa: BLE001 -- score-and-continue by design
            warnings.warn(
                f"[cutedsl-autotune] {label}: candidate {knobs} failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            scores.append(math.inf)
        finally:
            candidate_launch = None
    # Gloo status/timing exchange remains usable for CPU-only fault injection.
    device = (
        "cuda"
        if (collective and dist.get_backend(group=process_group) == "nccl")
        or (not collective and torch.cuda.is_available())
        else "cpu"
    )
    t = torch.tensor(scores, dtype=torch.float64, device=device)
    if collective:
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=process_group)
    best = int(torch.argmin(t).item())
    if not math.isfinite(float(t[best])):
        raise RuntimeError(
            f"[cutedsl-autotune] {label}: every candidate failed to compile/run."
        )
    winner = candidates[best]
    if multi_rank:
        install(winner)
    else:
        frontend.apply_knobs(winner)
        if config_overrides:
            _apply_config_overrides(frontend, config_overrides)
    if on_winner is not None:
        on_winner(winner, float(t[best]))
    if rank == 0:
        ranked = sorted(zip(t.tolist(), candidates, strict=False), key=lambda kv: kv[0])
        summary = "\n".join(f"    {us * 1e6:10.1f} us  {knobs}" for us, knobs in ranked)
        print(
            f"[cutedsl-autotune] {label}: winner {winner} "
            f"({float(t[best]) * 1e6:.1f} us median, max across ranks) "
            f"out of {len(candidates)} candidates:\n{summary}",
            flush=True,
        )
    return winner


def autotune_nvfp4_mega_moe(
    y: torch.Tensor,
    transformed_l1: Any,
    transformed_l2: Any,
    symm_buffer: Any,
    *,
    num_tokens: Optional[int] = None,
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
    warmup_iters: int = 3,
    timed_iters: int = 10,
) -> Dict[str, Any]:
    """Autotune the NVFP4 mega session on the caller's staged inputs.

    Arguments mirror :func:`.nvfp4.nvfp4_mega_moe`; ``y`` is clobbered by the
    candidate launches. Deferred sessions leave ``y`` untouched and time only
    the megakernel, excluding the caller's finalizer. Apply the winner and
    return its knob dict; rebuild any earlier launch thunks or borrowed views.
    COLLECTIVE -- see :func:`autotune_knobs`.
    """
    from .comm import resolve_gate_up_clamp
    from .nvfp4 import (
        COMBINE_FORMAT_NAMES,
        nvfp4_mega_launch_thunk,
        nvfp4_mega_moe,
    )

    def launch() -> None:
        # sync=True: the tune loop times launches with perf_counter, so the
        # call must block until the kernel (and output copy) complete.
        nvfp4_mega_moe(
            y,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            gate_up_clamp=gate_up_clamp,
            activation_clamp=activation_clamp,
            sync=True,
        )

    frontend = symm_buffer._frontend
    overrides: Dict[str, Any] = {}
    if swiglu_alpha is not None or swiglu_beta is not None:
        overrides.update(swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta)
    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp, activation_clamp=activation_clamp
    )
    if clamp is not None:
        overrides["gate_up_clamp"] = clamp
    cfg = frontend.config

    def prepare_launch() -> Callable[[], None]:
        _validate_tuning_output(y, frontend, num_tokens)
        nvfp4_mega_launch_thunk(transformed_l1, transformed_l2, symm_buffer)
        return launch

    if cfg.use_custom_finalize:
        n = num_tokens if num_tokens is not None else cfg.num_tokens_per_rank
        if n < 0 or n > cfg.num_tokens_per_rank:
            raise ValueError(
                f"num_tokens must be in [0, {cfg.num_tokens_per_rank}], got {n}."
            )

        def prepare_launch() -> Callable[[], None]:
            thunk = nvfp4_mega_launch_thunk(transformed_l1, transformed_l2, symm_buffer)

            def launch_unreduced() -> None:
                thunk()
                torch.cuda.synchronize()

            return launch_unreduced

    if candidates is None:
        # Session-aware default sweep: prune ikr when the session did not
        # permit it or cannot run it, and quantized-combine-invalid combos.
        candidates = nvfp4_candidates(
            batch_invariant=getattr(cfg, "batch_invariant", False),
            combine_format=COMBINE_FORMAT_NAMES[cfg.combine_dtype],
            enable_in_kernel_fc2_reduce=(
                cfg.enable_in_kernel_fc2_reduce
                and cfg.apply_routing_weights_before_fc2
                and not cfg.use_custom_finalize
            ),
        )

    def _record(winner: Dict[str, Any], p50_s: float) -> None:
        # Persist for future pure-lookup engine starts; rank 0 writes (the
        # winner is identical on all ranks after the all_reduce).
        from flashinfer.moe_ep.core.tuning import tuning_live_tokens

        cfg = symm_buffer._frontend.config
        live_tokens = tuning_live_tokens(num_tokens, cfg.num_tokens_per_rank)
        if cfg.rank == 0:
            from .knob_cache import record_knobs

            record_knobs(
                winner,
                dtype="nvfp4",
                world_size=cfg.world_size,
                hidden=cfg.hidden,
                intermediate=cfg.intermediate,
                num_experts=cfg.num_total_experts,
                topk=cfg.num_topk,
                max_tokens=cfg.num_tokens_per_rank,
                live_tokens=live_tokens,
                combine_dtype=cfg.combine_dtype,
                apply_routing_weights_before_fc2=cfg.apply_routing_weights_before_fc2,
                use_custom_finalize=cfg.use_custom_finalize,
                p50_us=p50_s * 1e6,
                source="autotune",
                batch_invariant=getattr(cfg, "batch_invariant", False),
                gate_up_clamp=cfg.gate_up_clamp,
                input_norm_const=getattr(cfg, "input_norm_const", 1.0),
                **(batch_invariant_variant_kwargs(cfg) if cfg.batch_invariant else {}),
            )

    return autotune_knobs(
        symm_buffer._frontend,
        launch,
        candidates,
        label="nvfp4_mega",
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        on_winner=_record,
        prepare_launch=prepare_launch,
        config_overrides=overrides,
        candidate_filter=(
            lambda proposed: _session_candidates(
                proposed, cfg, is_valid_nvfp4_for_config, what="NVFP4 MegaMoE"
            )
        )
        if getattr(cfg, "batch_invariant", False)
        else None,
    )


def autotune_mxfp8_mega_moe(
    y: torch.Tensor,
    transformed_l1: Any,
    transformed_l2: Any,
    symm_buffer: Any,
    *,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
    warmup_iters: int = 3,
    timed_iters: int = 10,
) -> Dict[str, Any]:
    """MXFP8 twin of :func:`autotune_nvfp4_mega_moe` (COLLECTIVE)."""
    from .mxfp8 import mxfp8_mega_moe, mxfp8_mega_launch_thunk
    from .comm import resolve_gate_up_clamp

    def launch() -> None:
        # sync=True: the tune loop times launches with perf_counter, so the
        # call must block until the kernel (and output copy) complete.
        mxfp8_mega_moe(
            y,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
            gate_up_clamp=gate_up_clamp,
            activation_clamp=activation_clamp,
            sync=True,
        )

    frontend = symm_buffer._frontend
    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp, activation_clamp=activation_clamp
    )
    overrides = {"gate_up_clamp": clamp} if clamp is not None else {}
    cfg = frontend.config

    def prepare_launch():
        _validate_tuning_output(y, frontend, num_tokens)
        mxfp8_mega_launch_thunk(transformed_l1, transformed_l2, symm_buffer)
        return launch

    if candidates is None:
        candidates = mxfp8_candidates(
            enable_in_kernel_fc2_reduce=cfg.enable_in_kernel_fc2_reduce,
            batch_invariant=getattr(cfg, "batch_invariant", False),
        )

    def _record(winner: Dict[str, Any], p50_s: float) -> None:
        # Persist for future pure-lookup engine starts; rank 0 writes (the
        # winner is identical on all ranks after the all_reduce).
        from flashinfer.moe_ep.core.tuning import tuning_live_tokens

        cfg = symm_buffer._frontend.config
        live_tokens = tuning_live_tokens(num_tokens, cfg.num_tokens_per_rank)
        if cfg.rank == 0:
            from .knob_cache import record_knobs

            record_knobs(
                winner,
                dtype=cfg.kind,
                # Preserve the ordinary cache's historical key convention.
                apply_routing_weights_before_fc2=getattr(cfg, "batch_invariant", False),
                combine_dtype=getattr(cfg, "combine_dtype", "bf16"),
                use_custom_finalize=getattr(cfg, "use_custom_finalize", False),
                world_size=cfg.world_size,
                hidden=cfg.hidden,
                intermediate=cfg.intermediate,
                num_experts=cfg.num_total_experts,
                topk=cfg.num_topk,
                max_tokens=cfg.num_tokens_per_rank,
                live_tokens=live_tokens,
                p50_us=p50_s * 1e6,
                source="autotune",
                batch_invariant=getattr(cfg, "batch_invariant", False),
                gate_up_clamp=cfg.gate_up_clamp,
                input_norm_const=getattr(cfg, "input_norm_const", 1.0),
                **(batch_invariant_variant_kwargs(cfg) if cfg.batch_invariant else {}),
            )

    return autotune_knobs(
        symm_buffer._frontend,
        launch,
        candidates,
        label="mxfp8_mega",
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        on_winner=_record,
        prepare_launch=prepare_launch,
        config_overrides=overrides,
        candidate_filter=(
            lambda proposed: _session_candidates(
                proposed, cfg, is_valid_mxfp8_for_config, what="MXFP8 MegaMoE"
            )
        )
        if getattr(cfg, "batch_invariant", False)
        else None,
    )


def autotune_bf16_mega_moe(
    y: torch.Tensor,
    transformed_l1: Any,
    transformed_l2: Any,
    symm_buffer: Any,
    *,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
    warmup_iters: int = 3,
    timed_iters: int = 10,
) -> Dict[str, Any]:
    """Autotune admitted BF16 plans, filtering against the session's shape."""
    from .bf16 import bf16_mega_moe, bf16_mega_launch_thunk
    from .comm import resolve_gate_up_clamp

    def launch() -> None:
        bf16_mega_moe(
            y,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
            gate_up_clamp=gate_up_clamp,
            activation_clamp=activation_clamp,
            sync=True,
        )

    frontend = symm_buffer._frontend
    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp, activation_clamp=activation_clamp
    )
    overrides = {"gate_up_clamp": clamp} if clamp is not None else {}

    def prepare_launch():
        _validate_tuning_output(y, frontend, num_tokens)
        bf16_mega_launch_thunk(transformed_l1, transformed_l2, symm_buffer)
        return launch

    cfg = frontend.config
    builtin_candidates = candidates is None
    if candidates is None:
        # Shape and SMEM filtering is expected for the built-in geometry
        # product. Warn only when a caller explicitly supplies an invalid plan.
        candidates = bf16_candidates(
            enable_in_kernel_fc2_reduce=cfg.enable_in_kernel_fc2_reduce,
            batch_invariant=cfg.batch_invariant,
        )

    def filter_candidates(proposed):
        if builtin_candidates:
            proposed = [
                knobs for knobs in proposed if is_valid_bf16_for_config(cfg, knobs)
            ]
        return _session_candidates(
            proposed, cfg, is_valid_bf16_for_config, what="BF16 MegaMoE"
        )

    def _record(winner: Dict[str, Any], p50_s: float) -> None:
        # Persist for future pure-lookup engine starts; rank 0 writes (the
        # winner is identical on all ranks after the all_reduce).
        from flashinfer.moe_ep.core.tuning import tuning_live_tokens

        # A launch-level clamp override may have updated an existing frontend
        # during the sweep. Record the arithmetic that was actually measured.
        cfg = symm_buffer._frontend.config
        live_tokens = tuning_live_tokens(num_tokens, cfg.num_tokens_per_rank)
        if cfg.rank == 0:
            from .knob_cache import record_knobs

            record_knobs(
                winner,
                dtype="bf16",
                world_size=cfg.world_size,
                hidden=cfg.hidden,
                intermediate=cfg.intermediate,
                num_experts=cfg.num_total_experts,
                topk=cfg.num_topk,
                max_tokens=cfg.num_tokens_per_rank,
                live_tokens=live_tokens,
                apply_routing_weights_before_fc2=cfg.apply_routing_weights_before_fc2,
                use_custom_finalize=cfg.use_custom_finalize,
                batch_invariant=cfg.batch_invariant,
                gate_up_clamp=cfg.gate_up_clamp,
                p50_us=p50_s * 1e6,
                source="autotune",
                **(batch_invariant_variant_kwargs(cfg) if cfg.batch_invariant else {}),
            )

    return autotune_knobs(
        symm_buffer._frontend,
        launch,
        candidates,
        label="bf16_mega",
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        on_winner=_record,
        prepare_launch=prepare_launch,
        config_overrides=overrides,
        candidate_filter=filter_candidates,
    )


def autotune_bf16_mxfp8_mega_moe(
    y: torch.Tensor,
    transformed_l1: Any,
    transformed_l2: Any,
    symm_buffer: Any,
    *,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
    warmup_iters: int = 3,
    timed_iters: int = 10,
) -> Dict[str, Any]:
    """Autotune the BF16×MXFP8 MegaMoE session on staged inputs."""
    from .bf16_mxfp8 import bf16_mxfp8_mega_moe, bf16_mxfp8_mega_launch_thunk

    def launch() -> None:
        bf16_mxfp8_mega_moe(
            y,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
            gate_up_clamp=gate_up_clamp,
            sync=True,
        )

    frontend = symm_buffer._frontend
    overrides = {"gate_up_clamp": gate_up_clamp} if gate_up_clamp is not None else {}

    def prepare_launch():
        _validate_tuning_output(y, frontend, num_tokens)
        bf16_mxfp8_mega_launch_thunk(transformed_l1, transformed_l2, symm_buffer)
        return launch

    cfg = frontend.config
    if candidates is None:
        candidates = bf16_mxfp8_candidates(
            enable_in_kernel_fc2_reduce=cfg.enable_in_kernel_fc2_reduce,
            batch_invariant=cfg.batch_invariant,
        )

    def _record(winner: Dict[str, Any], p50_s: float) -> None:
        # Persist for future pure-lookup engine starts; rank 0 writes (the
        # winner is identical on all ranks after the all_reduce).
        from flashinfer.moe_ep.core.tuning import tuning_live_tokens

        cfg = symm_buffer._frontend.config
        live_tokens = tuning_live_tokens(num_tokens, cfg.num_tokens_per_rank)
        if cfg.rank == 0:
            from .knob_cache import record_knobs

            record_knobs(
                winner,
                dtype=cfg.kind,
                world_size=cfg.world_size,
                hidden=cfg.hidden,
                intermediate=cfg.intermediate,
                num_experts=cfg.num_total_experts,
                topk=cfg.num_topk,
                max_tokens=cfg.num_tokens_per_rank,
                live_tokens=live_tokens,
                p50_us=p50_s * 1e6,
                source="autotune",
                batch_invariant=cfg.batch_invariant,
                apply_routing_weights_before_fc2=cfg.batch_invariant,
                gate_up_clamp=cfg.gate_up_clamp,
                use_custom_finalize=cfg.use_custom_finalize,
                **(batch_invariant_variant_kwargs(cfg) if cfg.batch_invariant else {}),
            )

    return autotune_knobs(
        symm_buffer._frontend,
        launch,
        candidates,
        label="bf16_mxfp8_mega",
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        on_winner=_record,
        prepare_launch=prepare_launch,
        config_overrides=overrides,
        candidate_filter=lambda proposed: _session_candidates(
            proposed,
            cfg,
            is_valid_bf16_mxfp8_for_config,
            what="mixed BF16/MXFP8 MegaMoE",
        ),
    )


__all__ = [
    "autotune_knobs",
    "autotune_bf16_mega_moe",
    "autotune_bf16_mxfp8_mega_moe",
    "autotune_mxfp8_mega_moe",
    "autotune_nvfp4_mega_moe",
    "bf16_candidates",
    "bf16_mxfp8_candidates",
    "mxfp8_candidates",
    "nvfp4_candidates",
]
