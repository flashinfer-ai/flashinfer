# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Immutable configuration of the MegaMoE megakernel.

Two frozen dataclasses hold every option that is neither part of the problem
shape (world size, hidden size, top-k, ...) nor a scheduler parameter that the
runner already exposes as a CLI argument (``group_hint``, ``epi_flag_batch``):

* :class:`MegaMoEConfig` carries the codegen options that a deployment may
  set: the register split of the GEMM task warpgroup, the TMA L2 cache-hint
  mask, the in-kernel FP32 tail reduce, and the intranode token-back atomic
  batch.
* :class:`MegaMoEDiagnostics` (nested in the config) carries the profiling
  and debugging switches. The first group changes the generated kernel
  (timing probes, IKET range budget, fan-out skips, ...); the second group
  only changes host behaviour of the runner.

The migrated GIN controls use these objects. FlashInfer's existing non-GIN
FP4 reducer still selects its decoder through ``MEGA_F4CVT_USE_MANUAL``;
that legacy environment-controlled path is outside this port. No ``os.environ``
is consulted at import or construction time in ``pre_unification/comm/token_comm.py``,
``pre_unification/moe_nvfp4_swapab/megamoe_kernel.py``, ``pre_unification/moe_nvfp4_swapab/kernel_fc12.py``,
``pre_unification/moe_nvfp4_swapab/topk_reduce.py`` or ``common/pipeline_timing.py``. The
runner builds the config from its CLI arguments and parses the diagnostic
``MEGA_*`` environment variables once, through
:meth:`MegaMoEDiagnostics.from_env`.

GIN configuration contributes to the compiled-kernel name through
:meth:`MegaMoEConfig.name_suffix`: non-default migrated codegen options add
ordered tokens. This does not replace FlashInfer's pre-existing non-GIN
FP4 decoder environment selection or its compilation lifecycle.

This module imports neither torch nor cutlass; it is usable on a login node.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import os
from typing import Mapping, Optional


IKET_TRACE_MODES = ("diagnostic", "production_bounded")


@dataclass(frozen=True)
class MegaMoEExperimental:
    """Opt-in performance candidates still under evaluation (default off).

    A candidate lives here until its gates decide: adopted -> its default branch
    becomes the only implementation and the field is removed; rejected -> code,
    field, kernel-name token, runner argument, launcher field and tests are
    removed together (one history line stays in the design document).

    * ``early_stage1`` (2026-09-21 dispatch campaign, former launcher field 60):
      the relay warps pack Stage 1 right after their own dispatch_prep and skip
      dispatch_barrier, message m of destination slot s is packed by relay warp
      (prefix_s + m) mod W over the whole grid, and the Stage 1 issuer lane
      posts the Step 3 metadata row right before its range put on the same
      queue pair. Requires the GIN path.
    * ``direct_l1_sf_staging`` (former launcher field 61): the fan-out writes
      each hit's scale-factor row with 16-byte vectors into a per-row staging
      area on the expert owner and raises a per-row flag; the owner's dispatch
      warps scatter the row into the swizzled atoms and release the readiness
      credit. Requires the GIN path.
    """

    early_stage1: bool = False
    direct_l1_sf_staging: bool = False

    def name_suffix(self) -> str:
        tokens = []
        if self.early_stage1:
            tokens.append("_es1")
        if self.direct_l1_sf_staging:
            tokens.append("_sfs1")
        return "".join(tokens)


def _env_flag(environ: Mapping[str, str], name: str, default: str = "0") -> bool:
    text = environ.get(name, default)
    if text not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1; got {text!r}")
    return text == "1"


@dataclass(frozen=True)
class MegaMoEDiagnostics:
    """Profiling and debugging switches of the MegaMoE kernel and runner.

    Codegen group (each changes the generated kernel and is part of the
    kernel name):

    * ``pipeline_timing``: ``%globaltimer`` stage envelopes written to the
      ``pipeline_timing`` workspace region (``MEGA_PIPELINE_TIMING``).
    * ``fc_tile_timeline``: per-tile FC1/FC2 timeline rows; requires
      ``pipeline_timing`` (``MEGA_FC_TILE_TIMELINE``).
    * ``iket_enabled``: the CuTe DSL compiler keeps ``iket.*`` ranges
      (``iket`` in ``CUTE_DSL_COMPILER_OPT``); without it the range calls
      compile out.
    * ``iket_trace_mode``: ``diagnostic`` emits every range, including the
      per-message and per-pull-iteration ranges; ``production_bounded`` keeps
      only fixed-count envelopes, barriers, waits and landmarks
      (``MEGA_IKET_TRACE_MODE``).
    * ``diag_fanout_skip``: bit mask that skips parts of the fan-out hit for
      attribution only (results are wrong when non-zero): 1 skips the
      scale-factor scatter, 2 the token DATA copy (``MEGA_DIAG_FANOUT_SKIP``).
    * ``fc2_done_gpu_scope``: publish the FC2 done counters with GPU-scope
      releases although GIN peers read them over NVLink; measures the price
      of the system-scope fence, not a candidate default (Invariant I9)
      (``MEGA_FC2_DONE_GPU_SCOPE``).
    * ``use_ncu``: pad the NVLink barrier cycle and reset the monotonic
      arrival signals so Nsight Compute can replay one launch
      (``MEGA_USE_NCU``).
    * ``diag_stage1_flush``: after SM0 posted the Stage 1 range puts its relay
      warps flush the GIN context, so the ``push_flush`` sub-envelope measures
      post-to-landed time on the sender's clock; a timing perturbation, not a
      performance candidate (``MEGA_DIAG_STAGE1_FLUSH``).

    Host-only group (runner behaviour, not part of the kernel name):
    ``cuda_event_only``, ``lifecycle_trace``, ``hang_dump_sec``,
    ``peek_workspace``, ``no_dist``, ``combine_theory_us``,
    ``fc_tile_timeline_dump``.
    """

    pipeline_timing: bool = False
    fc_tile_timeline: bool = False
    iket_enabled: bool = False
    iket_trace_mode: str = "diagnostic"
    diag_fanout_skip: int = 0
    fc2_done_gpu_scope: bool = False
    use_ncu: bool = False
    diag_stage1_flush: bool = False

    cuda_event_only: bool = False
    lifecycle_trace: bool = False
    hang_dump_sec: float = 0.0
    peek_workspace: bool = False
    no_dist: bool = False
    combine_theory_us: Optional[float] = None
    fc_tile_timeline_dump: bool = True

    def __post_init__(self) -> None:
        if self.fc_tile_timeline and not self.pipeline_timing:
            raise ValueError("fc_tile_timeline requires pipeline_timing")
        if self.iket_trace_mode not in IKET_TRACE_MODES:
            raise ValueError(
                "iket_trace_mode must be diagnostic or production_bounded; "
                f"got {self.iket_trace_mode!r}"
            )
        if not isinstance(self.diag_fanout_skip, int) or not 0 <= self.diag_fanout_skip <= 3:
            raise ValueError(
                f"diag_fanout_skip must be 0, 1, 2 or 3; got {self.diag_fanout_skip!r}"
            )
        if self.hang_dump_sec < 0:
            raise ValueError("hang_dump_sec must be non-negative")

    @property
    def iket_detailed_ranges(self) -> bool:
        """Diagnostic traces emit the message- and iteration-scaled ranges."""
        return self.iket_trace_mode == "diagnostic"

    @property
    def diag_fanout_skip_sf(self) -> bool:
        return bool(self.diag_fanout_skip & 1)

    @property
    def diag_fanout_skip_data(self) -> bool:
        return bool(self.diag_fanout_skip & 2)

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "MegaMoEDiagnostics":
        """Parse the diagnostic ``MEGA_*`` variables (the runner calls this once)."""
        env = os.environ if environ is None else environ
        skip_text = env.get("MEGA_DIAG_FANOUT_SKIP", "0")
        if skip_text not in ("0", "1", "2", "3"):
            raise ValueError(f"MEGA_DIAG_FANOUT_SKIP must be 0, 1, 2 or 3; got {skip_text!r}")
        trace_mode = env.get("MEGA_IKET_TRACE_MODE", "diagnostic")
        if trace_mode not in IKET_TRACE_MODES:
            raise ValueError(
                "MEGA_IKET_TRACE_MODE must be diagnostic or production_bounded; "
                f"got {trace_mode!r}"
            )
        hang_text = env.get("MEGA_HANG_DUMP_SEC", "0")
        try:
            hang_dump_sec = float(hang_text)
        except ValueError as exc:
            raise ValueError("MEGA_HANG_DUMP_SEC must be a number") from exc
        theory_text = env.get("MEGA_COMBINE_THEORY_US", "").strip()
        no_dist_text = env.get("MEGA_NO_DIST", "0")
        try:
            no_dist = bool(int(no_dist_text))
        except ValueError as exc:
            raise ValueError("MEGA_NO_DIST must be an integer") from exc
        return cls(
            pipeline_timing=_env_flag(env, "MEGA_PIPELINE_TIMING"),
            fc_tile_timeline=(
                _env_flag(env, "MEGA_PIPELINE_TIMING")
                and _env_flag(env, "MEGA_FC_TILE_TIMELINE")
            ),
            iket_enabled="iket" in env.get("CUTE_DSL_COMPILER_OPT", "").split(","),
            iket_trace_mode=trace_mode,
            diag_fanout_skip=int(skip_text),
            fc2_done_gpu_scope=_env_flag(env, "MEGA_FC2_DONE_GPU_SCOPE"),
            use_ncu=_env_flag(env, "MEGA_USE_NCU"),
            diag_stage1_flush=_env_flag(env, "MEGA_DIAG_STAGE1_FLUSH"),
            cuda_event_only=_env_flag(env, "MEGA_CUDA_EVENT_ONLY"),
            lifecycle_trace=_env_flag(env, "MEGA_LIFECYCLE_TRACE"),
            hang_dump_sec=hang_dump_sec,
            peek_workspace=_env_flag(env, "MEGA_PEEK_WORKSPACE"),
            no_dist=no_dist,
            combine_theory_us=float(theory_text) if theory_text else None,
            fc_tile_timeline_dump=env.get("MEGA_FC_TILE_TIMELINE_DUMP", "1") != "0",
        )

    def name_suffix(self) -> str:
        """Kernel-name tokens of the codegen group (non-default values only)."""
        tokens = []
        if self.pipeline_timing:
            tokens.append("_ptim")
        if self.fc_tile_timeline:
            tokens.append("_fctl1")
        if self.diag_fanout_skip:
            tokens.append(f"_dfs{self.diag_fanout_skip}")
        if self.fc2_done_gpu_scope:
            tokens.append("_fdg1")
        if self.iket_enabled:
            tokens.append("_iket")
            if not self.iket_detailed_ranges:
                tokens.append("prod")
        if self.use_ncu:
            tokens.append("_ncu")
        if self.diag_stage1_flush:
            tokens.append("_dsf1")
        return "".join(tokens)


@dataclass(frozen=True)
class MegaMoEConfig:
    """Codegen options of the MegaMoE megakernel that a deployment may set.

    * ``task_reg_cnt``: registers per thread of the sched/TMA/MMA task
      warpgroup; multiple of 8 in [24, 112] (budget 4*256 + 4*task + 8*72 <=
      2048). 72 is the historical split.
    * ``tma_cache_hint``: bit mask of TMA L2 cache hints on the mainloop
      operand loads; 1 = weights evict-first, 2 = token / FC1-output B operand
      evict-last.
    * ``tail_topk_reduce``: ordered FP32 top-k reduce inside the GIN kernel
      tail instead of the separate TopkReduce launch. Requires the GIN path.
    * ``token_back_atomic_batch``: token-back publish batch of the intranode
      ``atomic_counter`` scheduling mode (developer knob; the GIN path uses
      static scheduling and ignores it).
    * ``experimental``: see :class:`MegaMoEExperimental` (opt-in candidates).
    * ``diagnostics``: see :class:`MegaMoEDiagnostics`.
    """

    task_reg_cnt: int = 72
    tma_cache_hint: int = 0
    tail_topk_reduce: bool = False
    token_back_atomic_batch: int = 1
    experimental: MegaMoEExperimental = field(default_factory=MegaMoEExperimental)
    diagnostics: MegaMoEDiagnostics = field(default_factory=MegaMoEDiagnostics)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.task_reg_cnt, int)
            or self.task_reg_cnt % 8
            or not 24 <= self.task_reg_cnt <= 112
        ):
            raise ValueError(
                "task_reg_cnt must be a multiple of 8 in [24, 112], got "
                f"{self.task_reg_cnt!r}"
            )
        if not isinstance(self.tma_cache_hint, int) or not 0 <= self.tma_cache_hint <= 3:
            raise ValueError(f"tma_cache_hint must be 0, 1, 2 or 3; got {self.tma_cache_hint!r}")
        if not isinstance(self.token_back_atomic_batch, int) or self.token_back_atomic_batch < 1:
            raise ValueError(
                f"token_back_atomic_batch must be a positive integer; got {self.token_back_atomic_batch!r}"
            )
        if not isinstance(self.experimental, MegaMoEExperimental):
            raise TypeError("experimental must be a MegaMoEExperimental")
        if not isinstance(self.diagnostics, MegaMoEDiagnostics):
            raise TypeError("diagnostics must be a MegaMoEDiagnostics")

    def with_diagnostics(self, diagnostics: MegaMoEDiagnostics) -> "MegaMoEConfig":
        return replace(self, diagnostics=diagnostics)

    def name_suffix(self) -> str:
        """Kernel-name tokens of every codegen option that is not at its default.

        Order: ``_treg{N}``, ``_tmach{M}``, ``_tailreduce``, the experimental
        tokens (``_es1``, ``_sfs1``), then the diagnostics tokens (``_ptim``,
        ``_fctl1``, ``_dfs{n}``, ``_fdg1``, ``_iket`` / ``_iketprod``, ``_ncu``,
        ``_dsf1``).
        ``token_back_atomic_batch`` is a runtime batch of the atomic scheduler
        (not codegen) and has no token.
        """
        tokens = []
        if self.task_reg_cnt != 72:
            tokens.append(f"_treg{self.task_reg_cnt}")
        if self.tma_cache_hint:
            tokens.append(f"_tmach{self.tma_cache_hint}")
        if self.tail_topk_reduce:
            tokens.append("_tailreduce")
        return "".join(tokens) + self.experimental.name_suffix() + self.diagnostics.name_suffix()
