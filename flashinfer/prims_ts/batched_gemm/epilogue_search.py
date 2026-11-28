# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""Finite, lazy search over supported FIFO epilogue schedules.

The full search grows rapidly even for eight fragments. Callers should bound
the number of compilations, measure complete kernels and retain the legacy
implementation as a candidate. This module performs no GPU work or caching.
"""

from dataclasses import fields, replace
from itertools import product

from .epilogue_schedule import (
    EpilogueStep as Step,
    epilogue_steps_for_prefix,
    fc1_streaming_steps,
)


def iter_epilogue_steps(*, finalize, stages, exhaustive=False, gated=False, warps=8):
    """Yield useful schedules first, optionally followed by exhaustive FIFO plans.

    Exhaustive means every load/stage count from zero through eight, complete
    output-unit grouping, every legal release point, and read-ahead 0/1.
    Instruction widths and ownership are resolved by the supported mapping.
    Scratch-overflow branches are pruned before configuration validation.
    """
    if stages not in (1, 2):
        raise ValueError("epilogue search supports one/two C slots")
    fragments = 32 // warps if gated else 8
    unit = 2 if gated else (4 if finalize else 1)
    if gated:
        # Establish parity with the retained schedule before exploring earlier
        # release points or additional live register/SMEM state.
        yield fc1_streaming_steps(warps=warps)
    # Include the full-tile single-group schedule before smaller prefixes.
    yield (Step(load_fragments=1, stage_fragments=1),) * fragments + (
        Step(release_tmem=True),
        Step(emit_fragments=fragments),
    )
    prefixes = (0, 2, 1, 4, 6, 8) if gated else (2, 0, 1, 4, 6, 8)
    for prefix, ahead in product(prefixes, (0, 1)):
        if prefix <= fragments:
            yield epilogue_steps_for_prefix(
                prefix, finalize=finalize, read_ahead=ahead, gated=gated, warps=warps
            )
    # Streaming output and delayed release are also useful: they bound both
    # register retention and scratch occupancy independently of tile size.
    for chunk, ahead in product((1, 2, 4), (0, 1)):
        steps = []
        for _base in range(0, fragments, unit):
            for _ in range(0, unit, min(chunk, unit)):
                steps.append(
                    Step(
                        load_fragments=min(chunk, unit),
                        stage_fragments=min(chunk, unit),
                        read_ahead=ahead,
                    )
                )
            steps.append(Step(emit_fragments=unit))
        yield tuple(steps) + (Step(release_tmem=True),)
    if not exhaustive:
        return

    def visit(loaded, staged, emitted, released, steps):
        if (loaded, staged, emitted, released) == (
            fragments,
            fragments,
            fragments,
            True,
        ):
            yield steps
            return
        if loaded == fragments and not released:
            yield from visit(
                loaded, staged, emitted, True, steps + (Step(release_tmem=True),)
            )
        for emit in range(unit, staged - emitted + 1, unit):
            yield from visit(
                loaded,
                staged,
                emitted + emit,
                released,
                steps + (Step(emit_fragments=emit),),
            )
        for load in range(0, (0 if released else fragments - loaded) + 1):
            available = min(loaded + load - staged, stages * unit - (staged - emitted))
            for stage in range(available + 1):
                if not (load or stage):
                    continue
                for ahead in (0, 1) if load else (0,):
                    yield from visit(
                        loaded + load,
                        staged + stage,
                        emitted,
                        released,
                        steps
                        + (
                            Step(
                                load_fragments=load,
                                stage_fragments=stage,
                                read_ahead=ahead,
                            ),
                        ),
                    )

    yield from visit(0, 0, 0, False, ())


def iter_epilogue_configs(
    base,
    *,
    warps=(4, 8),
    stages=(1, 2),
    register_budgets=None,
    exhaustive=False,
    max_candidates=64,
    problem_mnk=None,
):
    """Yield distinct, fully validated kernel configs, without compiling them.

    The base supplies operand queues, CGA and output semantics. Register
    budgets default to the established 248/168 allocations. An optional
    ``{warps: (budgets, ...)}`` mapping sweeps other supported allocations.
    Change the base for additional operand/CGA profiles. A finite candidate limit
    also bounds the deduplication set's memory use.
    """
    from .batched_gemm_config import validate_config
    from .batched_gemm_kernel import build_batched_gemm_task_manager

    if type(max_candidates) is not int or max_candidates <= 0:
        raise ValueError("max_candidates must be a positive integer")
    if register_budgets is None:
        register_budgets = {4: (248,), 8: (144,) if base.has_gated_epilogue else (168,)}
    seen = set()
    # Round-robin resource profiles so a bounded search includes both warp
    # counts and slot depths before exhausting any individual schedule space.
    profiles = list(product(warps, stages))
    if base.has_gated_epilogue:
        profiles.sort(
            key=lambda profile: (
                profile[0] != base.dense_epilogue_warps,
                profile[1] != base.num_stages_c_smem,
            )
        )
    streams = [
        (
            w,
            s,
            regs,
            iter_epilogue_steps(
                finalize=bool(base.fuse_moe_finalize),
                stages=s,
                exhaustive=exhaustive,
                gated=base.has_gated_epilogue,
                warps=w,
            ),
        )
        for w, s in profiles
        for regs in register_budgets.get(w, ())
    ]
    while streams:
        active = []
        for w, s, regs, stream in streams:
            try:
                steps = next(stream)
            except StopIteration:
                continue
            active.append((w, s, regs, stream))
            cfg = replace(
                base,
                dense_epilogue_warps=w,
                num_stages_c_smem=s,
                epilogue_regs=regs,
                dense_preload_epilogue=1,
                dense_preload_store_prefix_fragments=0,
                epilogue_tmem_load_width=0,
                epilogue_tmem_load_ahead=-1,
                epilogue_smem_staging=0,
                epilogue_tmem_release=0,
                epilogue_steps=steps,
            )
            try:
                validate_config(cfg, problem_mnk=problem_mnk)
            except ValueError:
                continue
            identity = (w, s, regs, cfg.preload_epilogue_plan.signature())
            if identity in seen:
                continue
            # Payload estimates alone do not cover register redistribution to
            # other tasks. Validate the actual schedule's complete resource
            # budget on the host before requesting an expensive compilation.
            try:
                build_batched_gemm_task_manager(
                    verbose=False,
                    **{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init},
                )
            except ValueError:
                continue
            seen.add(identity)
            yield cfg
            if len(seen) == max_candidates:
                return
        streams = active
