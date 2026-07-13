# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared host-driver scaffolding for the MegaMoE fused fc1+fc2 runners.

This module holds everything the NVFP4 (``moe_nvfp4_swapab/runner_fc12.py``)
and MXFP8 (``moe_mxfp8_glu/runner_fc12.py``) fused-fc12 runners have in common:

  * The three configuration descriptors (:class:`ProblemDesc`,
    :class:`ImplDesc`, :class:`MiscDesc`).  They are quantization-kind
    *parameterized* (each carries a ``kind`` and the descriptors validate
    against it), not kind-specific, so both runners reuse them verbatim.

  * :class:`Fc12TesterBase` -- the host-side input/reference/launch/
    validation driver.  All kind-agnostic logic (offs generation, physical
    offsets, raw-scale assembly, workspace partition, kernel launch
    plumbing, determinism check, scheduler-layout preview, top-level
    ``run`` orchestration) lives here.  The handful of genuinely
    kind-entangled steps are expressed as overridable hooks that the two
    concrete subclasses implement:

      - :meth:`_create_input_data_tensors` -- activation / fc1 / fc2 weights
      - :meth:`_alloc_fc2_output`          -- fc2 output buffer (3D vs 2D)
      - :meth:`_quantize_fc1`              -- post-SwiGLU quantize (nvfp4 vs mxfp8)
      - :meth:`_apply_topk_post_fc2`       -- transformers topk post-multiply
      - :meth:`_instantiate_kernel`        -- pick / construct the kernel class
      - :meth:`_validate_fc1_phase`        -- fc1 workspace readback diagnostics
      - :meth:`_fc2_tolerance`             -- (atol, rtol) for the fc2 compare

  * Shared argparse helpers (:func:`add_common_fc12_arguments`,
    :func:`parse_tuple`, :func:`parse_output_dtype`).
"""

import argparse
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
import cutlass  # noqa: F401
from cutlass.utils import HardwareInfo

from moe_nvfp4_swapab.epilogue import (
    EpilogueTokenTile,
    Fc1GateUpInterleave as Nvfp4Fc1GateUpInterleave,
)
from moe_mxfp8_glu.epilogue_mxfp8 import (
    Fc1GateUpInterleave as Mxfp8Fc1GateUpInterleave,
)
from common.megamoe_constants import (
    Nvfp4BlockSize,
    Mxfp8BlockSize,
    SfPaddingBlock,
    SupportedMmaTileM,
    SupportedMmaTileN,
)
from moe_nvfp4_swapab.simulate_fc1_fc2_sched import FusedFc12Simulator

from moe_nvfp4_swapab.runner_common import (
    Nvfp4DataDtype,
    _create_raw_scale_tensor,
    assemble_raw_scales_grouped_token,
    assemble_raw_scales_stacked_expert,
    ceil_div,
    offs_to_group_sizes,
    round_up,
    slice_tensor_logical_dim,
)
from common.host_utils import (
    kind_data_dtype,
    kind_scale_dtype,
    kind_sf_vec_size,
    compare_and_report_mismatches,
)


# =============================================================================
# Configuration descriptors (shared, kind-parameterized)
# =============================================================================


@dataclass
class ProblemDesc:
    """Problem-side configuration and host-visible layout locks."""

    tokens_after_topk: int = 0
    experts: int = 0
    balance_route: bool = True
    hidden: int = 0
    intermediate: int = 0
    simulate_ep: Optional[int] = None
    fc2_output_dtype: torch.dtype = torch.bfloat16
    # DeepSeek-V4 swiglu_limit: asymmetric clamp on the real gate/up
    # pre-activations before SiLU.  NVFP4-only knob; ``None`` disables it (and
    # other kinds leave it None).
    gate_up_clamp: Optional[float] = None

    scenario: Literal["2Dx3D"] = "2Dx3D"
    kind: Literal["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"] = "nvfp4"
    acc_dtype: torch.dtype = torch.float32
    fc1_activation_layout: Literal["k_major"] = "k_major"
    fc1_weight_layout: Literal["k_major"] = "k_major"
    fc2_weight_layout: Literal["k_major", "n_major"] = "k_major"
    fc2_output_layout: Literal["n_major"] = "n_major"

    def __post_init__(self) -> None:
        if self.tokens_after_topk < 0:
            raise ValueError(
                f"tokens_after_topk must be non-negative, got {self.tokens_after_topk}."
            )
        if self.experts <= 0:
            raise ValueError(f"experts must be positive, got {self.experts}.")
        if self.hidden <= 0 or self.hidden % Nvfp4BlockSize != 0:
            raise ValueError(
                f"hidden ({self.hidden}) must be a positive multiple of {Nvfp4BlockSize}."
            )
        _interleave = (
            Nvfp4Fc1GateUpInterleave
            if self.kind == "nvfp4"
            else Mxfp8Fc1GateUpInterleave
        )
        if self.intermediate <= 0 or self.intermediate % (2 * _interleave) != 0:
            raise ValueError(
                f"intermediate ({self.intermediate}) must be a positive multiple of "
                f"{2 * _interleave} (gate/up interleave granularity for kind={self.kind!r})."
            )
        _sf_vec = Nvfp4BlockSize if self.kind == "nvfp4" else Mxfp8BlockSize
        if (self.intermediate // 2) % _sf_vec != 0:
            raise ValueError(
                f"intermediate/2 ({self.intermediate // 2}) must be a multiple of "
                f"sf_vec_size ({_sf_vec}) for fc2's per-block SF coverage."
            )
        if self.simulate_ep is not None and self.simulate_ep <= 0:
            raise ValueError(f"simulate_ep must be positive, got {self.simulate_ep}.")
        if self.gate_up_clamp is not None and self.gate_up_clamp < 0.0:
            raise ValueError(
                f"gate_up_clamp must be None or non-negative, got {self.gate_up_clamp}."
            )

        # -- Locked-domain fields --
        # Run BEFORE the TMA alignment block: the alignment derivation reads
        # the layout fields to pick which dim is the stride-1 (leading) row,
        # so an unsupported layout would otherwise surface as a confusing
        # KeyError instead of a clear "layout is locked" message.
        if self.scenario != "2Dx3D":
            raise ValueError(
                f"Only scenario='2Dx3D' is supported in v1, got {self.scenario!r}."
            )
        if self.kind not in ("nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"):
            raise ValueError(
                f"kind must be one of 'nvfp4', 'mxfp8_e4m3', 'mxfp8_e5m2'; "
                f"got {self.kind!r}."
            )
        if self.fc2_output_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"fc2_output_dtype must be torch.bfloat16 or torch.float16, "
                f"got {self.fc2_output_dtype}."
            )
        if self.acc_dtype != torch.float32:
            raise ValueError(
                f"acc_dtype is locked to torch.float32 in v1, got {self.acc_dtype}."
            )
        if self.fc1_activation_layout != "k_major":
            raise ValueError(
                f"fc1_activation_layout is locked to 'k_major' in v1, got "
                f"{self.fc1_activation_layout!r}."
            )
        if self.fc1_weight_layout != "k_major":
            raise ValueError(
                f"fc1_weight_layout is locked to 'k_major' in v1, got "
                f"{self.fc1_weight_layout!r}."
            )
        if self.fc2_weight_layout != "k_major":
            raise ValueError(
                f"fc2_weight_layout='n_major' is reserved for MXFP8 and not "
                f"supported in v1; got {self.fc2_weight_layout!r}."
            )
        if self.fc2_output_layout != "n_major":
            raise ValueError(
                f"fc2_output_layout is locked to 'n_major' in v1, got "
                f"{self.fc2_output_layout!r}."
            )

        # -- TMA leading-dim 16-byte alignment --
        # Each host-exposed tensor's stride-1 row is selected by the
        # corresponding layout field above; the leading-dim element count
        # is then picked from ProblemDesc dims via the per-tensor mapping
        # tables below.  Each table covers all currently-typed layout
        # choices (the v1-locked guards above narrow them to the ones
        # actually supported, but the tables stay forward-compat for when
        # the locks are relaxed).  Bytes per leading row are derived from
        # the dtype by ``check_tma_leading_dim_align`` (NVFP4 = 0.5B/elem,
        # bf16/fp16 = 2B, fp32 = 4B, fp8 = 1B).
        #
        # ``fc1_output`` is the kernel-internal staging buffer (post-fc1
        # NVFP4 + per-block SF, consumed as fc2's B operand).  Its layout
        # is fixed by the kernel (k_major over (tokens, intermediate/2))
        # and not exposed via ProblemDesc, but its leading-dim alignment
        # still has to hold or fc2's TMA load reads garbage -- check
        # explicitly here with the kernel-fixed layout inlined.
        from moe_nvfp4_swapab.runner_common import (
            check_tma_leading_dim_align as _check_tma_leading_dim_align,
        )

        _check_tma_leading_dim_align(
            "activation",
            {"k_major": self.hidden}[self.fc1_activation_layout],
            Nvfp4DataDtype,
        )
        _check_tma_leading_dim_align(
            "fc1_weight",
            {"k_major": self.hidden}[self.fc1_weight_layout],
            Nvfp4DataDtype,
        )
        _check_tma_leading_dim_align(
            "fc2_weight",
            {"k_major": self.intermediate // 2, "n_major": self.hidden}[
                self.fc2_weight_layout
            ],
            Nvfp4DataDtype,
        )
        _check_tma_leading_dim_align(
            "fc1_output (kernel-internal, fixed k_major)",
            self.intermediate // 2,
            Nvfp4DataDtype,
        )
        _check_tma_leading_dim_align(
            "fc2_output",
            {"n_major": self.hidden}[self.fc2_output_layout],
            self.fc2_output_dtype,
        )

    def __str__(self) -> str:
        d = lambda t: str(t).split(".")[-1]
        route = "balanced" if self.balance_route else "random"
        ep_str = f" ep={self.simulate_ep}" if self.simulate_ep is not None else ""
        return (
            f"ProblemDesc: {self.scenario} | kind={self.kind} | "
            f"tokens_after_topk={self.tokens_after_topk} experts={self.experts} "
            f"route={route} | "
            f"hidden={self.hidden} intermediate={self.intermediate} | "
            f"fc2_output_dtype={d(self.fc2_output_dtype)} "
            f"layouts=(act={self.fc1_activation_layout},"
            f"fc1w={self.fc1_weight_layout},"
            f"fc2w={self.fc2_weight_layout},"
            f"fc2out={self.fc2_output_layout}) "
            f"acc={d(self.acc_dtype)}{ep_str}"
        )


@dataclass
class ImplDesc:
    """Kernel-instantiation-side configuration.

    Defaults: mma_tiler ``(M=128, N=128, K=256)``, single-cluster, single-CTA
    MMA, static sched, static load balance.

    Fused fc12 fields:
      - ``load_balance_mode``: chooses the persistent scheduler's load-balance
        strategy.  ``"static"`` is the v1 default (stride mode);
        ``"atomic_counter"`` is the dynamic alternative.
      - ``group_hint``: per-group fc1 tile threshold that drives expert
        packing.  ``None`` means the runner fills in
        ``HardwareInfo().get_max_active_clusters(cluster_size)`` at kernel
        launch time.
      - ``enable_static_expert_shape``: when True, ``run_kernel`` constructs
        the kernel with
        ``static_expert_shape = (experts, intermediate_gateup, hidden)``
        Python int triple read from ``ProblemDesc``, so the scheduler sees
        codegen-const dims (lets div/mod over ``intermediate_gateup`` /
        ``hidden`` fold to magic-mul).  When False the kernel reads those
        three dims from the per-launch tensors at runtime (dynamic Int32).
        Either path is correct; ``True`` typically gives a small perf win
        at the cost of one extra ``cute.compile`` per problem shape.
    """

    mma_tiler_mnk: Tuple[int, int, int] = (128, 128, 256)
    cluster_shape_mnk: Tuple[int, int, int] = (1, 1, 1)
    use_2cta_instrs: bool = False
    enable_static_expert_shape: bool = False
    force_static_sched: bool = True
    clc_bundle_size: Optional[int] = None
    num_sched_stages: Optional[int] = None
    load_balance_mode: Literal["static", "atomic_counter"] = "static"
    group_hint: Optional[int] = None
    non_ubulk_fc2_store: bool = True
    in_kernel_fc2_reduce: bool = False
    token_back_mode: Literal[
        "epi_warps", "standalone_warps", "reuse_dispatch_warps"
    ] = "epi_warps"
    flag_batch: int = 4
    epi_flag_batch: Optional[Tuple[int, int]] = (1, 1)
    # Cross-rank combine transfer format for the reuse_dispatch_warps
    # token-back push.  "bf16" is the unchanged default; "mxfp8" / "nvfp4"
    # quantize each 32-element block (tile-wise) on the wire to cut combine
    # NVLink volume, with the receiver dequantizing + topk-reducing via the
    # standalone topk_reduce kernel.  Low-precision modes are a form-A-only
    # path (no fp8/fp4 cp.reduce).
    combine_dtype: Literal["bf16", "mxfp8", "nvfp4"] = "bf16"

    def __post_init__(self) -> None:
        m, n, _k = self.mma_tiler_mnk
        cm, cn, cl = self.cluster_shape_mnk

        if m not in SupportedMmaTileM:
            raise ValueError(
                f"mma_tiler_m must be one of {SupportedMmaTileM}, got {m}."
            )
        if n not in SupportedMmaTileN:
            raise ValueError(
                f"mma_tiler_n must be one of {SupportedMmaTileN}, got {n}."
            )
        if cl != 1:
            raise ValueError(f"cluster_l must be 1, got {cl}.")
        if cn != 1:
            raise ValueError(f"cluster_n must be 1 in v1, got {cn}.")
        if cm < 1 or cm > 16 or cm * cn > 16:
            raise ValueError(
                f"cluster_m must be in [1, 16] and cluster_m*cluster_n <=16, "
                f"got cluster=({cm},{cn})."
            )
        if self.use_2cta_instrs != (m == 256):
            raise ValueError(
                f"use_2cta_instrs ({self.use_2cta_instrs}) must equal "
                f"(mma_tiler_m == 256), got mma_tiler_m={m}."
            )
        if self.load_balance_mode not in ("static", "atomic_counter"):
            raise ValueError(
                f"load_balance_mode must be 'static' or 'atomic_counter' "
                f"(the 'clc' mode is handled by a separate scheduler class "
                f"and is not wired through the fused fc12 kernel); "
                f"got {self.load_balance_mode!r}."
            )
        if self.token_back_mode not in (
            "epi_warps",
            "standalone_warps",
            "reuse_dispatch_warps",
        ):
            raise ValueError(
                f"token_back_mode must be 'epi_warps', 'standalone_warps', "
                f"or 'reuse_dispatch_warps'; got {self.token_back_mode!r}."
            )
        if self.combine_dtype not in ("bf16", "mxfp8", "nvfp4"):
            raise ValueError(
                f"combine_dtype must be 'bf16', 'mxfp8', or 'nvfp4'; "
                f"got {self.combine_dtype!r}."
            )
        if self.combine_dtype != "bf16":
            # Tile-wise low-precision combine has no fp8/fp4 cp.reduce, so the
            # topk reduction must run host-side (form A: one staged cell per
            # (src_token, src_topk)).  In-kernel fc2 reduce (form B) and the
            # epi_warps STG-redirect are therefore incompatible, and only the
            # reuse_dispatch_warps push is wired for the quantized path.
            if self.in_kernel_fc2_reduce:
                raise ValueError(
                    "combine_dtype != 'bf16' requires in_kernel_fc2_reduce="
                    "False (low-precision combine is a form-A path)."
                )
            if self.token_back_mode != "reuse_dispatch_warps":
                raise ValueError(
                    "combine_dtype != 'bf16' is currently only wired for "
                    "token_back_mode='reuse_dispatch_warps'; got "
                    f"{self.token_back_mode!r}."
                )
        if self.group_hint is not None and self.group_hint <= 0:
            raise ValueError(
                f"group_hint must be positive when set, got {self.group_hint}."
            )
        if self.flag_batch < 1:
            raise ValueError(f"flag_batch must be >= 1, got {self.flag_batch}.")
        # epi done-counter batch is a ``(fc1, fc2)`` pair, published
        # warp-cooperatively (one lane per pending tile), so a single warp-wide
        # flush caps each at 32.
        eb = self.epi_flag_batch if self.epi_flag_batch is not None else (1, 1)
        if len(eb) != 2:
            raise ValueError(
                f"epi_flag_batch must be a (fc1, fc2) pair, got {self.epi_flag_batch}."
            )
        for _leg, _val in (("fc1", eb[0]), ("fc2", eb[1])):
            if _val < 1 or _val > 32:
                raise ValueError(
                    f"epi_flag_batch[{_leg}] must be in [1, 32], got {_val}."
                )

    @property
    def fc2_reduces_topk(self) -> bool:
        return self.in_kernel_fc2_reduce

    @property
    def combine_is_quantized(self) -> bool:
        """True when the cross-rank combine push uses a tile-wise
        low-precision wire format (mxfp8 / nvfp4) instead of bf16."""
        return self.combine_dtype != "bf16"

    @property
    def token_back_by_dispatch(self) -> bool:
        """True when fc2 is staged to a local workspace and pushed back by
        dispatch-area warps (standalone or reused), i.e. any non-epi mode."""
        return self.token_back_mode != "epi_warps"

    def __str__(self) -> str:
        tile = ",".join(map(str, self.mma_tiler_mnk))
        cluster = ",".join(map(str, self.cluster_shape_mnk))
        static_shape = "static" if self.enable_static_expert_shape else "dynamic"
        sched = "static" if self.force_static_sched else "dynamic_clc"
        bundle = self.clc_bundle_size if self.clc_bundle_size is not None else "default"
        stages = (
            self.num_sched_stages if self.num_sched_stages is not None else "default"
        )
        group_hint_str = (
            str(self.group_hint)
            if self.group_hint is not None
            else "max_active_clusters"
        )
        return (
            f"ImplDesc: tile={tile} cluster={cluster} 2cta={self.use_2cta_instrs} | "
            f"expert_shape={static_shape} sched={sched} bundle={bundle} stages={stages} | "
            f"load_balance={self.load_balance_mode} group_hint={group_hint_str} | "
            f"non_ubulk_fc2_store={self.non_ubulk_fc2_store} "
            f"in_kernel_fc2_reduce={self.in_kernel_fc2_reduce} "
            f"token_back_mode={self.token_back_mode} "
            f"combine_dtype={self.combine_dtype}"
        )


@dataclass
class MiscDesc:
    """Misc run-time switches for the fused fc12 runner.

    ``ref_compute_graph`` selects which reference compute graph to use:

      - ``"transformers"``: HuggingFace transformers semantics --
        ``out = (down_proj @ swiglu) * topk_score`` with topk_score
        applied AFTER fc2 GEMM in fp32, then cast to bf16/fp16.

      - ``"deepgemm"``:  DeepGEMM mega-MoE semantics (Path A) --
        ``out = down_proj @ (swiglu * topk_score)`` with topk_score
        pre-multiplied into swiglu in fp32 BEFORE NVFP4 quantize, so
        per-block SF absorbs the per-token scaling.  This is what the
        kernel does when its PostSwigluHalf path is enabled (the path
        adopted to avoid an fc2-side warp shuffle for weight broadcast).

    The two paths are NOT bit-exact (NVFP4 round-to-nearest at fp4 ULP
    differs slightly when the inputs differ), but in NVFP4 the additional
    error introduced by Path A is on the order of fp8 SF rounding (~0.4%)
    -- much smaller than NVFP4's inherent ~6% quant error.

    Default ``"deepgemm"`` matches the kernel path; switch to
    ``"transformers"`` to compare against the ground-truth reference
    semantics (expect tolerance to absorb the small extra mismatch).
    """

    perf_run: bool = False
    skip_ref_check: bool = False
    run_target_kernel_only: bool = False
    enable_debug_checks: bool = False
    ref_compute_graph: Literal["transformers", "deepgemm"] = "deepgemm"
    seed: int = 1234
    # Preserve iket.* ops through lowering; runtime backend is selected outside
    # this runner via DKG_IKET_INSTRUMENTATION_METHOD.
    enable_iket: bool = False
    verbose: bool = False

    @property
    def profile_friendly(self) -> bool:
        """Alias for ``run_target_kernel_only``."""
        return self.run_target_kernel_only

    def __post_init__(self) -> None:
        if self.ref_compute_graph not in ("transformers", "deepgemm"):
            raise ValueError(
                f"ref_compute_graph must be 'transformers' or 'deepgemm', "
                f"got {self.ref_compute_graph!r}."
            )

    def __str__(self) -> str:
        return (
            f"MiscDesc: perf={self.perf_run} skip_ref={self.skip_ref_check} "
            f"target_only={self.run_target_kernel_only} "
            f"debug_checks={'on' if self.enable_debug_checks else 'off'} "
            f"ref_graph={self.ref_compute_graph} "
            f"iket={'on' if self.enable_iket else 'off'} seed={self.seed}"
        )


# =============================================================================
# Tester base (kind-agnostic host driver)
# =============================================================================


class Fc12TesterBase:
    """Host-side input/reference/launch/validation driver (kind-agnostic).

    Concrete subclasses (``SwapABSwigluFp4Fc12Tester`` for NVFP4,
    ``SwigluMxfp8Fc12Tester`` for MXFP8) implement the small set of
    kind-entangled hooks documented at the bottom of this class.
    """

    def __init__(
        self,
        problem: ProblemDesc,
        impl: ImplDesc,
        misc: MiscDesc,
    ) -> None:
        if impl.fc2_reduces_topk:
            raise ValueError(
                "in_kernel_fc2_reduce=True requires MegaMoE token_comm "
                "metadata; lean runner has no topk combine semantics."
            )

        self.problem = problem
        self.impl = impl
        self.misc = misc

        torch.manual_seed(misc.seed)
        np.random.seed(misc.seed)
        self._np_rng: np.random.Generator = np.random.default_rng(misc.seed)

        # Dirichlet concentration parameter (only consulted when
        # ``balance_route`` is False).
        self.dirichlet_alpha: float = 0.5

        # -- State populated by generate_inputs / compute_reference --
        self.offs: Optional[torch.Tensor] = None
        self.valid_tokens_per_expert: Optional[List[int]] = None
        self.data_physical_offsets: Optional[List[int]] = None
        self.sf_physical_offsets: Optional[List[int]] = None

        self.activation: Optional[torch.Tensor] = None
        self.fc1_weight: Optional[torch.Tensor] = None
        self.fc2_weight: Optional[torch.Tensor] = None
        self.activation_sf: Optional[torch.Tensor] = None
        self.fc1_weight_sf: Optional[torch.Tensor] = None
        self.fc2_weight_sf: Optional[torch.Tensor] = None
        self.raw_activation_sf_list: Optional[List[torch.Tensor]] = None
        self.raw_fc1_weight_sf_list: Optional[List[torch.Tensor]] = None
        self.raw_fc2_weight_sf_list: Optional[List[torch.Tensor]] = None
        self.activation_global_scale: Optional[torch.Tensor] = None
        self.fc1_weight_global_scale: Optional[torch.Tensor] = None
        self.fc2_weight_global_scale: Optional[torch.Tensor] = None
        self.norm_const: Optional[torch.Tensor] = None
        # Optional per-expert fc1/fc2 alpha (real-value rescale folded into the
        # epilogue).  ``None`` => no alpha (kernel + reference both treat as 1.0);
        # the NVFP4 tester fills them, mxfp8 leaves them None.
        self.fc1_alpha: Optional[torch.Tensor] = None
        self.fc2_alpha: Optional[torch.Tensor] = None
        self.topk_scores: Optional[torch.Tensor] = None
        self.fc2_output: Optional[torch.Tensor] = None
        self.workspace: Optional[torch.Tensor] = None

        self.fc2_output_ref: Optional[torch.Tensor] = None

        # Compiled-kernel cache (run_kernel populates on first invocation).
        self._compiled_kernel = None

        # -- Per-phase debug ablation state (compute_reference + run_kernel
        # populate; _validate_fc1_phase / _check_kernel_determinism consume) --
        #
        # Indexed by expert idx; ``None`` slot for 0-token experts (which
        # ``compute_reference`` skips and the kernel never writes for).  All
        # tensors live on the same CUDA device as the launch outputs so
        # diff arithmetic stays GPU-side.
        self._ref_fc1_q_per_expert: List[Optional[torch.Tensor]] = []
        self._ref_fc1_raw_sf_per_expert: List[Optional[torch.Tensor]] = []

        # Workspace + launch handles re-used by the kernel-determinism
        # re-launch (``_check_kernel_determinism``) and by the per-expert
        # fc1 readback (``_validate_fc1_phase``).  ``run_kernel`` writes
        # them at the end of the first launch path.
        self._ws_fc1_output_torch: Optional[torch.Tensor] = None
        self._ws_fc1_output_sf_torch: Optional[torch.Tensor] = None
        self._launch_runtime_kwargs: Optional[dict] = None

    # ------------------------------------------------------------------
    # Offs generation (per-expert VALID token counts; not padded)
    # ------------------------------------------------------------------

    def _generate_offs(self) -> torch.Tensor:
        """Build a cumulative-end ``offs`` tensor where each value is a *valid*
        token count cumsum.

        Per-expert valid count can be any non-negative integer (Dirichlet
        path produces zero-token experts in skewed routing scenarios).  No
        per-expert 128 / 64 alignment constraint is imposed at the offs
        level -padding policy is enforced separately by ``generate_inputs``
        when laying out physical buffers.

        ``simulate_ep`` divides the effective valid token budget to mimic
        EP fan-out.
        """
        ep = self.problem.simulate_ep
        if ep is not None and ep > 1:
            total = ceil_div(self.problem.tokens_after_topk, ep)
        else:
            total = self.problem.tokens_after_topk

        experts = self.problem.experts
        if total < 0:
            raise ValueError(
                f"Effective valid token total ({total}) must be non-negative."
            )

        if total == 0:
            return torch.zeros((experts,), dtype=torch.int32, device="cuda")

        if self.problem.balance_route:
            base, remainder = divmod(total, experts)
            counts = [base + (1 if i < remainder else 0) for i in range(experts)]
        else:
            proportions = self._np_rng.dirichlet([self.dirichlet_alpha] * experts)
            raw = np.floor(proportions * total).astype(int)
            deficit = total - int(raw.sum())
            # Greedy fill of the deficit (token slot allocator).
            while deficit > 0:
                idx = int(np.argmin(raw / (proportions * total + 1e-12)))
                raw[idx] += 1
                deficit -= 1
            while deficit < 0:
                ratios = np.where(
                    raw > 0,
                    raw / (proportions * total + 1e-12),
                    -np.inf,
                )
                idx = int(np.argmax(ratios))
                raw[idx] -= 1
                deficit += 1
            counts = raw.tolist()

        assert sum(counts) == total, (
            f"Internal: valid-count allocation mismatch ({sum(counts)} != {total})"
        )

        cum = 0
        offsets: List[int] = []
        for c in counts:
            cum += int(c)
            offsets.append(cum)
        return torch.tensor(offsets, dtype=torch.int32, device="cuda")

    # ------------------------------------------------------------------
    # Raw scale generation
    # ------------------------------------------------------------------

    def _generate_raw_scales(
        self, valid_tokens_per_expert: List[int]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Build per-expert raw (un-swizzled) SF tensors for the three SF sides.

        - ``raw_activation_sf``: per expert ``(valid_tokens_e, hidden / blocksize)``
        - ``raw_fc1_weight_sf``: per expert ``(intermediate, hidden / blocksize)``
                                  (identical shape across experts)
        - ``raw_fc2_weight_sf``: per expert ``(hidden, (intermediate/2) / blocksize)``
                                  (identical shape across experts)
        """
        hidden = self.problem.hidden
        intermediate = self.problem.intermediate
        experts = self.problem.experts

        sf_vec_size = kind_sf_vec_size(self.problem.kind)
        scale_dtype = kind_scale_dtype(self.problem.kind)

        raw_activation_sf = [
            _create_raw_scale_tensor(
                non_k_size=v,
                k_size=hidden,
                blocksize=sf_vec_size,
                scale_dtype=scale_dtype,
            )
            for v in valid_tokens_per_expert
        ]
        raw_fc1_weight_sf = [
            _create_raw_scale_tensor(
                non_k_size=intermediate,
                k_size=hidden,
                blocksize=sf_vec_size,
                scale_dtype=scale_dtype,
            )
            for _ in range(experts)
        ]
        raw_fc2_weight_sf = [
            _create_raw_scale_tensor(
                non_k_size=hidden,
                k_size=intermediate // 2,
                blocksize=sf_vec_size,
                scale_dtype=scale_dtype,
            )
            for _ in range(experts)
        ]
        return raw_activation_sf, raw_fc1_weight_sf, raw_fc2_weight_sf

    # ------------------------------------------------------------------
    # Input construction
    # ------------------------------------------------------------------

    @property
    def _epilogue_token_tile(self) -> int:
        """Per-expert token-padding granularity, owned by the impl epilogue.

        Subclasses override to read their own epilogue class constant instead of
        a runner-side hardcode; the base default keeps the shared value so
        kinds that don't override are unchanged.
        """
        return EpilogueTokenTile

    def _compute_physical_offsets(
        self, valid_tokens_per_expert: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Return ``(data_physical_offsets, sf_physical_offsets)``.

        Each is a length-``(experts+1)`` list; entry ``e`` is the physical
        starting row of expert ``e`` in the corresponding tensor.  The last
        entry is the total physical row count.
        """
        data_offsets: List[int] = [0]
        sf_offsets: List[int] = [0]
        for v in valid_tokens_per_expert:
            data_offsets.append(
                data_offsets[-1] + round_up(v, self._epilogue_token_tile)
            )
            sf_offsets.append(sf_offsets[-1] + round_up(v, SfPaddingBlock))
        return data_offsets, sf_offsets

    def _generate_inputs_skeleton(
        self,
        valid_tokens_per_expert: List[int],
        data_total_rows: int,
        sf_total_rows: int,
    ) -> None:
        """Allocate tensors with correct shape / stride / dtype but no data init.

        Only ``offs`` carries real values; every other buffer is
        ``torch.empty`` -zero CUDA kernel launches.  Intended for simulator
        runs where only the target kernel should execute.
        """
        problem = self.problem
        hidden = problem.hidden
        intermediate = problem.intermediate
        experts = problem.experts

        # -- activation: (data_total_rows, hidden) fp4, hidden stride-1 --
        data_dtype = kind_data_dtype(problem.kind)
        scale_dtype = kind_scale_dtype(problem.kind)
        sf_vec_size = kind_sf_vec_size(problem.kind)

        if problem.kind == "nvfp4":
            self.activation = torch.empty(
                (data_total_rows, hidden // 2),
                dtype=torch.uint8,
                device="cuda",
            ).view(data_dtype)
            self.fc1_weight = (
                torch.empty(
                    (experts, intermediate, hidden // 2),
                    dtype=torch.uint8,
                    device="cuda",
                )
                .view(data_dtype)
                .permute(0, 2, 1)
            )
            self.fc2_weight = (
                torch.empty(
                    (experts, hidden, (intermediate // 2) // 2),
                    dtype=torch.uint8,
                    device="cuda",
                )
                .view(data_dtype)
                .permute(0, 2, 1)
            )
        else:
            self.activation = torch.empty(
                (data_total_rows, hidden), dtype=data_dtype, device="cuda"
            )
            # fc1_weight: (experts, intermediate, hidden) → permute → (experts, hidden, intermediate), hidden stride-1
            self.fc1_weight = torch.empty(
                (experts, intermediate, hidden), dtype=data_dtype, device="cuda"
            ).permute(0, 2, 1)
            # fc2_weight: (experts, hidden, inter//2) → permute → (experts, inter//2, hidden), inter//2 stride-1
            self.fc2_weight = torch.empty(
                (experts, hidden, intermediate // 2), dtype=data_dtype, device="cuda"
            ).permute(0, 2, 1)

        # -- SFs (atom-layout 2D buffers, byte-allocated) --
        sfa_cols = round_up(ceil_div(hidden, sf_vec_size), 4)
        self.activation_sf = torch.empty(
            (sf_total_rows, sfa_cols),
            dtype=torch.uint8,
            device="cuda",
        ).view(scale_dtype)

        sfb_per_expert = round_up(intermediate, SfPaddingBlock) * round_up(
            ceil_div(hidden, sf_vec_size), 4
        )
        self.fc1_weight_sf = torch.empty(
            (experts, sfb_per_expert),
            dtype=torch.uint8,
            device="cuda",
        ).view(scale_dtype)

        sfb2_per_expert = round_up(hidden, SfPaddingBlock) * round_up(
            ceil_div(intermediate // 2, sf_vec_size), 4
        )
        self.fc2_weight_sf = torch.empty(
            (experts, sfb2_per_expert),
            dtype=torch.uint8,
            device="cuda",
        ).view(scale_dtype)

        # -- global scales / norm_const --
        self.activation_global_scale = torch.empty(
            (experts,), dtype=torch.float32, device="cuda"
        )
        self.fc1_weight_global_scale = torch.empty(
            (experts,), dtype=torch.float32, device="cuda"
        )
        self.fc2_weight_global_scale = torch.empty(
            (experts,), dtype=torch.float32, device="cuda"
        )
        self.norm_const = torch.empty((1,), dtype=torch.float32, device="cuda")

        # -- topk_scores --
        self.topk_scores = torch.empty(
            (data_total_rows,),
            dtype=torch.float32,
            device="cuda",
        )

        # -- fc2_output --
        self._alloc_fc2_output_skeleton(data_total_rows)

        # -- Workspace --
        # Allocate a 1 MiB placeholder buffer so callers see a non-None
        # ``self.workspace`` even on the short-circuit path.  ``run_kernel``
        # queries the kernel for the precise required size and reallocates
        # if this is too small (then zero-inits).
        self.workspace = torch.zeros((1 << 20,), dtype=torch.uint8, device="cpu").to(
            "cuda"
        )

        self.raw_activation_sf_list = None
        self.raw_fc1_weight_sf_list = None
        self.raw_fc2_weight_sf_list = None

    def _alloc_fc2_output_skeleton(self, data_total_rows: int) -> None:
        """Allocate an un-initialized ``fc2_output`` for the skeleton path.

        Shape differs by kind (NVFP4 carries a lean ``topk=1`` axis), so
        defer to the same ``_alloc_fc2_output`` shape contract subclasses
        use, but with ``torch.empty`` instead of a 0xFF sentinel fill.
        """
        problem = self.problem
        self.fc2_output = torch.empty(
            self._fc2_output_shape(data_total_rows),
            dtype=problem.fc2_output_dtype,
            device="cuda",
        )

    def generate_inputs(self) -> None:
        """Build offs and all input / output tensors.

        Memory layout summary:

          ``activation``    physical rows = ``sum_e round_up(valid_e, 64)``
          ``fc2_output``    physical rows = same as ``activation``
          ``topk_scores``   physical rows = same as ``activation``
          ``activation_sf`` physical rows = ``sum_e round_up(valid_e, 128)``

        Padding rows for each expert sit after that expert's valid rows but
        before the next expert's valid rows.  The kernel must (a) not depend
        on the contents of padding rows, and (b) not write outputs into
        padding rows -both invariants are achieved by epilogue subtile-level
        early exit driven by ``valid_tokens_in_tile``.

        For correctness sentinel detection, ``fc2_output`` and ``activation``
        are filled with 0xFF before the kernel runs; padding rows are left
        at 0xFF after the kernel writes valid rows.

        Kind-specific data tensor creation is delegated to
        :meth:`_create_input_data_tensors` and :meth:`_alloc_fc2_output`.
        """
        # -- 1. offs (valid cumsum) --
        self.offs = self._generate_offs()
        valid_tokens = offs_to_group_sizes(self.offs)
        self.valid_tokens_per_expert = valid_tokens

        # -- 2. Physical offsets (data side 64-padded, SF side 128-padded) --
        data_offsets, sf_offsets = self._compute_physical_offsets(valid_tokens)
        self.data_physical_offsets = data_offsets
        self.sf_physical_offsets = sf_offsets
        data_total_rows = data_offsets[-1]
        sf_total_rows = sf_offsets[-1]

        # Short-circuit on two paths that don't need fully-initialized data:
        #   - ``run_target_kernel_only``: perf simulator runs the kernel
        #     against undefined inputs to measure kernel-only latency.
        #   - ``data_total_rows == 0``: a high-EP rank with no routed tokens
        #     this step; SF assemble's ``reshape(0, -1)`` is ambiguous when
        #     all experts contribute 0 rows, so dodge it via skeleton.
        if self.misc.run_target_kernel_only or data_total_rows == 0:
            self._generate_inputs_skeleton(valid_tokens, data_total_rows, sf_total_rows)
            return

        # -- 3-5. activation / fc1_weight / fc2_weight (kind-specific) --
        self._create_input_data_tensors(data_total_rows)

        # -- 6. Raw scales + atom-layout assemble --
        (
            self.raw_activation_sf_list,
            self.raw_fc1_weight_sf_list,
            self.raw_fc2_weight_sf_list,
        ) = self._generate_raw_scales(valid_tokens)
        self.activation_sf = assemble_raw_scales_grouped_token(
            self.raw_activation_sf_list
        )
        self.fc1_weight_sf = assemble_raw_scales_stacked_expert(
            self.raw_fc1_weight_sf_list
        )
        self.fc2_weight_sf = assemble_raw_scales_stacked_expert(
            self.raw_fc2_weight_sf_list
        )

        # -- 7-8. Global scales + norm_const (v1 pinned to 1.0) --
        self._init_global_scales_and_norm()

        # -- 9. topk_scores --
        self._init_topk_scores(data_total_rows)

        # -- 10. fc2_output (kind-specific shape, 0xFF sentinel fill) --
        self._alloc_fc2_output(data_total_rows)

        # -- 11. Workspace placeholder --
        self._alloc_workspace_placeholder()

        torch.cuda.synchronize()

    def _init_global_scales_and_norm(self) -> None:
        """Per-expert tensor-wise global scales + NVFP4 norm const, pinned 1.0.

        Holding all three global scales at 1.0 makes the ``alpha =
        global_scale_a * global_scale_b`` factor a no-op, so a kernel that
        ignores them validates correctly.  When wired through, randomize
        these.
        """
        experts = self.problem.experts
        self.activation_global_scale = torch.ones(
            (experts,), dtype=torch.float32, device="cuda"
        )
        self.fc1_weight_global_scale = torch.ones(
            (experts,), dtype=torch.float32, device="cuda"
        )
        self.fc2_weight_global_scale = torch.ones(
            (experts,), dtype=torch.float32, device="cuda"
        )
        self.norm_const = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    def _init_topk_scores(self, data_total_rows: int) -> None:
        """``(data_total_rows,)`` fp32 topk scores.

        Valid rows get random values in [0.5, 1.5] (typical routing weight
        range); padding rows are explicitly zeroed so the kernel -if it
        accidentally writes a padding row -produces a deterministic 0
        that's easy to inspect.
        """
        valid_tokens = self.valid_tokens_per_expert
        data_offsets = self.data_physical_offsets
        self.topk_scores = torch.zeros(
            (data_total_rows,),
            dtype=torch.float32,
            device="cuda",
        )
        for e in range(self.problem.experts):
            v_e = valid_tokens[e]
            if v_e == 0:
                continue
            phys = data_offsets[e]
            self.topk_scores[phys : phys + v_e] = (
                torch.rand((v_e,), dtype=torch.float32, device="cuda") + 0.5
            )

    def _alloc_workspace_placeholder(self) -> None:
        """Ensure ``self.workspace`` is non-None before kernel launch.

        ``run_kernel`` queries the precise required size via the kernel's
        ``get_workspace_size_in_bytes`` and reallocates if needed; this
        placeholder just guarantees any code path touching ``self.workspace``
        before the launch sees a real buffer.
        """
        if self.workspace is None:
            self.workspace = torch.zeros(
                (1 << 20,), dtype=torch.uint8, device="cpu"
            ).to("cuda")

    # ------------------------------------------------------------------
    # Reference (per-expert manual path)
    # ------------------------------------------------------------------

    def compute_reference(self) -> None:
        """Per-expert manual PyTorch reference for fused fc1+fc2.

        Reference strategy: ``use_quantized_fc1`` -- fc1's quantized output
        is dequantized and fed to fc2, mirroring what the kernel sees at the
        in-buffer hand-off point.  This keeps the reference bit-accurate to
        the kernel's data path (modulo fp32 vs tcgen05 MMA rounding inside
        the GEMMs themselves).

        Per expert ``e`` with ``v_e`` valid tokens (skipping ``v_e == 0``):

          1. dequant activation slice -> ``act_fp32``  (v_e, hidden)
          2. dequant fc1 weight        -> ``w1_fp32``   (hidden, intermediate)
          3. ``fc1_fp32 = act_fp32 @ w1_fp32``
          4. SwiGLU fold over gate/up interleave        (v_e, intermediate/2)
          4.5. (kind hook) topk pre-multiply into swiglu  [NVFP4 Path A only]
          5. quantize swiglu  -> ``(fc1_q, fc1_sf)``    [kind hook]
          6. dequant fc1's quantized output             (v_e, intermediate/2)
          7. dequant fc2 weight        -> ``w2_fp32``   (intermediate/2, hidden)
          8. ``fc2_fp32 = fc1_dq @ w2_fp32``            (v_e, hidden)
          9. (kind hook) topk post-multiply             [NVFP4 transformers only]
         10. cast to fc2_output_dtype and store into ``fc2_output_ref``

        Padding rows (per-expert ``round_up(v_e, 64) - v_e`` extras) remain
        at the buffer's initial fill (0); ``validate`` skips them.
        """
        if self.activation is None or self.offs is None:
            raise RuntimeError("compute_reference requires generate_inputs first.")
        if self.misc.skip_ref_check:
            return

        from moe_nvfp4_swapab.mega_reference import (
            _BlockScaledGemmReferenceLauncher,
            reference_expert_fc12,
        )

        problem = self.problem
        valid_tokens = self.valid_tokens_per_expert
        data_offsets = self.data_physical_offsets
        data_total_rows = data_offsets[-1]
        norm_const_val = float(self.norm_const[0].item())
        sf_vec_size = kind_sf_vec_size(problem.kind)
        gate_up_interleave = (
            Nvfp4Fc1GateUpInterleave
            if problem.kind == "nvfp4"
            else Mxfp8Fc1GateUpInterleave
        )
        # ``gate_up_clamp`` lives on the NVFP4 ProblemDesc only (None elsewhere).
        gate_up_clamp = getattr(problem, "gate_up_clamp", None)

        # Bit-exact blockscaled GEMM launcher, shared across this expert sweep
        # (compiles per dtype-key on first use, cached internally).
        ref_scaled_mm = _BlockScaledGemmReferenceLauncher(
            sf_vec_size=sf_vec_size,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
        )

        # -- fc2_output_ref allocation --
        # Allocate via uint8 bytes then reinterpret to bf16/fp16, mirroring
        # the same trick used for ``self.fc2_output`` (some pytorch versions
        # lack a direct ``torch.empty(dtype=bfloat16, ...)`` path on some
        # backends; bytes + view is bulletproof).
        ref_bytes = torch.zeros(
            (data_total_rows, problem.hidden * problem.fc2_output_dtype.itemsize),
            dtype=torch.uint8,
            device="cuda",
        )
        self.fc2_output_ref = ref_bytes.view(problem.fc2_output_dtype).reshape(
            data_total_rows, problem.hidden
        )

        # Per-expert fc1 NVFP4 hand-off snapshots for the fc1-phase ablation
        # (reset each call so a unit-test loop doesn't leak stale entries).
        self._ref_fc1_q_per_expert = [None] * problem.experts
        self._ref_fc1_raw_sf_per_expert = [None] * problem.experts

        for expert_idx in range(problem.experts):
            v_e = valid_tokens[expert_idx]
            if v_e == 0:
                continue

            d_start = data_offsets[expert_idx]
            topk_slice = self.topk_scores[d_start : d_start + v_e]

            act_slice = slice_tensor_logical_dim(
                self.activation, dim=0, start=d_start, end=d_start + v_e
            )

            # Shared bit-exact core: packed fc1 GEMM -> SwiGLU(+clamp) ->
            # (deepgemm: topk pre-mult) -> NVFP4 round-trip -> packed fc2 GEMM.
            # ``_quantize_fc1`` is the per-kind quantizer hook; global scales are
            # pinned to 1.0 in v1 so alpha/norm fold them away (no host dequant).
            fc2_fp32, fc1_q, fc1_sf = reference_expert_fc12(
                ref_scaled_mm=ref_scaled_mm,
                quantize_fn=self._quantize_fc1,
                act_packed=act_slice,
                act_sf=self.raw_activation_sf_list[expert_idx],
                fc1_weight_packed=self.fc1_weight[expert_idx],
                fc1_weight_sf=self.raw_fc1_weight_sf_list[expert_idx],
                fc2_weight_packed=self.fc2_weight[expert_idx],
                fc2_weight_sf=self.raw_fc2_weight_sf_list[expert_idx],
                intermediate=problem.intermediate,
                hidden=problem.hidden,
                fc1_alpha=(
                    float(self.fc1_alpha[expert_idx].item())
                    if self.fc1_alpha is not None
                    else 1.0
                ),
                fc2_alpha=(
                    float(self.fc2_alpha[expert_idx].item())
                    if self.fc2_alpha is not None
                    else 1.0
                ),
                fc1_norm_const=norm_const_val,
                gate_up_interleave=gate_up_interleave,
                gate_up_clamp=gate_up_clamp,
                topk_weights=topk_slice,
                ref_compute_graph=self.misc.ref_compute_graph,
            )

            # ``transformers`` applies the topk weight after fc2 (kind hook;
            # ``deepgemm`` already pre-multiplied it into SwiGLU inside the core).
            fc2_fp32 = self._apply_topk_post_fc2(fc2_fp32, topk_slice)
            self.fc2_output_ref[d_start : d_start + v_e] = fc2_fp32.to(
                problem.fc2_output_dtype
            )

            # fc1 NVFP4 hand-off snapshot for the fc1-phase ablation.
            self._ref_fc1_q_per_expert[expert_idx] = fc1_q
            self._ref_fc1_raw_sf_per_expert[expert_idx] = fc1_sf

    # ------------------------------------------------------------------
    # Workspace partition helper
    # ------------------------------------------------------------------

    def _partition_workspace(
        self,
        mma_tiler_n: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Slice the opaque byte ``self.workspace`` into per-section views.

        The layout MUST match the kernel's ``get_workspace_size_in_bytes``
        allocation order (the kernel doesn't see this partition explicitly
        -- each section is forwarded as a separate cute.Tensor argument,
        but the underlying bytes share the same host-allocated buffer):

          0:                                 fc1_output (packed)
          fc1_output_end:                    fc1_output_sf (atom-tiled)
          fc1_output_sf_end:                 fc1_done_counter (Int32 1D)
          fc1_done_counter_end:              load_balance_counter (Int32 scalar,
                                             atomic_counter mode only)

        Returns ``(fc1_output_torch, fc1_output_sf_torch,
        fc1_done_counter_torch, load_balance_counter_torch_or_None)``
        as ``torch.Tensor`` views over ``self.workspace``.  Callers should
        further wrap each into a ``cute.Tensor`` via
        ``cutlass_torch.from_dlpack`` before passing to the kernel.

        ``mma_tiler_n`` is the token-axis CTA tile size under swap-AB
        (= ``mma_tiler_mnk[1]``).  Drives the fc1_done counter sizing
        because the counter is indexed along the token axis: slot =
        ``cumulative_token_block_count + tile_n_idx``, and each
        token-block covers ``mma_tiler_n`` consecutive tokens.
        """
        problem = self.problem
        experts = problem.experts
        intermediate = problem.intermediate  # intermediate_gateup
        intermediate_downproj = intermediate // 2
        data_total_rows = int(self.data_physical_offsets[-1])

        # Kind-specific workspace layout to match kernel's get_workspace_size_in_bytes.
        is_nvfp4 = self.problem.kind == "nvfp4"
        data_dtype = kind_data_dtype(self.problem.kind)
        scale_dtype = kind_scale_dtype(self.problem.kind)
        sf_vec_size = kind_sf_vec_size(self.problem.kind)
        # For FP4: 4-bit = 2 per byte. For FP8: 8-bit = 1 per byte.
        elem_bits = 4 if is_nvfp4 else 8

        sf_total_rows_upper = data_total_rows + experts * SfPaddingBlock
        sf_block_cols = ((intermediate_downproj // sf_vec_size) + 3) // 4 * 4
        counter_slots_upper = (
            data_total_rows + mma_tiler_n - 1
        ) // mma_tiler_n + experts

        fc1_output_byte_count = data_total_rows * intermediate_downproj * elem_bits // 8
        fc1_output_sf_byte_count = sf_total_rows_upper * sf_block_cols
        fc1_done_counter_byte_count = counter_slots_upper * 4

        ws = self.workspace
        offset = 0

        # -- fc1_output --
        if is_nvfp4:
            fc1_output_torch = (
                ws[offset : offset + fc1_output_byte_count]
                .view(torch.uint8)
                .view(data_dtype)
                .reshape(data_total_rows, intermediate_downproj // 2)
            )
        else:
            fc1_output_torch = (
                ws[offset : offset + fc1_output_byte_count]
                .view(torch.uint8)
                .view(data_dtype)
                .reshape(data_total_rows, intermediate_downproj)
            )
        offset += fc1_output_byte_count

        # -- fc1_output_sf --
        fc1_output_sf_torch = (
            ws[offset : offset + fc1_output_sf_byte_count]
            .view(torch.uint8)
            .view(scale_dtype)
            .reshape(sf_total_rows_upper, sf_block_cols)
        )
        offset += fc1_output_sf_byte_count

        # -- fc1_done_counter: Int32 1D, zero-init (host responsibility,
        # done by ``self.workspace.zero_()`` in run_kernel above).
        fc1_done_counter_torch = ws[offset : offset + fc1_done_counter_byte_count].view(
            torch.int32
        )
        offset += fc1_done_counter_byte_count

        # -- load_balance_counter: Int32 scalar (atomic_counter mode only).
        if self.impl.load_balance_mode == "atomic_counter":
            load_balance_counter_torch = ws[offset : offset + 4].view(torch.int32)
            offset += 4
        else:
            load_balance_counter_torch = None

        return (
            fc1_output_torch,
            fc1_output_sf_torch,
            fc1_done_counter_torch,
            load_balance_counter_torch,
        )

    # ------------------------------------------------------------------
    # Kernel call
    # ------------------------------------------------------------------

    def run_kernel(self) -> None:
        """Instantiate the fused kernel, partition workspace, compile, launch.

          1. Instantiate the kernel via :meth:`_instantiate_kernel` (forwards
             ``ProblemDesc`` dims as ``static_expert_shape`` when
             ``impl.enable_static_expert_shape`` is set).
          2. Query ``HardwareInfo().get_max_active_clusters(cluster_size)``
             and use it both for the kernel's ``max_active_clusters``
             constexpr and to default-fill ``impl.group_hint`` when it was
             passed as ``None``.
          3. Query ``kernel.get_workspace_size_in_bytes(...)``; reallocate
             ``self.workspace`` and zero-initialize it.
          4. Partition ``self.workspace`` into per-section torch views.
          5. Convert every torch tensor to a cute tensor.
          6. ``cute.compile`` once and stash the compiled callable.
          7. Launch.

        Lazy-imports ``cuda`` / ``cutlass`` so the harness-only paths
        (``generate_inputs`` + ``compute_reference``) can be exercised on
        machines without a working cute install.
        """
        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        import cutlass.utils as utils

        required = (
            self.activation,
            self.fc1_weight,
            self.fc2_weight,
            self.activation_sf,
            self.fc1_weight_sf,
            self.fc2_weight_sf,
            self.topk_scores,
            self.fc2_output,
            self.offs,
        )
        if any(t is None for t in required):
            raise RuntimeError("run_kernel requires generate_inputs first.")

        # Cluster size + max_active_clusters + group_hint default fill.
        cluster_size = self.impl.cluster_shape_mnk[0] * self.impl.cluster_shape_mnk[1]
        max_active_clusters = utils.HardwareInfo().get_max_active_clusters(cluster_size)
        group_hint = self.impl.group_hint
        if group_hint is None:
            group_hint = max_active_clusters

        # Static expert shape (codegen-time triple) -- when the
        # ImplDesc flag is set, hand the kernel the three dims as Python
        # ints; otherwise None lets the kernel read them as dynamic Int32
        # from the per-launch tensors.
        if self.impl.enable_static_expert_shape:
            static_expert_shape = (
                self.problem.experts,
                self.problem.intermediate,
                self.problem.hidden,
            )
        else:
            static_expert_shape = None

        # -- 1. Instantiate the kernel (kind hook) --
        common_kwargs = dict(
            mma_tiler_mnk=self.impl.mma_tiler_mnk,
            cluster_shape_mnk=self.impl.cluster_shape_mnk,
            use_2cta_instrs=self.impl.use_2cta_instrs,
            group_hint=group_hint,
            token_padding_block=self._epilogue_token_tile,
            sf_padding_block=SfPaddingBlock,
            load_balance_mode=self.impl.load_balance_mode,
            static_expert_shape=static_expert_shape,
            force_static_sched=self.impl.force_static_sched,
            clc_bundle_size=self.impl.clc_bundle_size,
            num_sched_stages=self.impl.num_sched_stages,
            sf_vec_size=kind_sf_vec_size(self.problem.kind),
        )
        kernel = self._instantiate_kernel(common_kwargs)

        # -- 2. Workspace sizing + zero-init --
        # Zero-init is REQUIRED before each launch (fc1_done_counter and,
        # under atomic_counter mode, load_balance_counter both rely on
        # zero-init semantics).
        required_workspace_bytes = kernel.get_workspace_size_in_bytes(
            self.activation, self.fc1_weight
        )
        self.workspace = torch.zeros(
            (required_workspace_bytes,), dtype=torch.uint8, device="cpu"
        ).to("cuda")

        # -- 3. Workspace partition (torch views) --
        #
        # ``mma_tiler_n`` is the token-axis CTA tile size under swap-AB
        # (see ``_partition_workspace`` docstring); it drives the
        # fc1_done counter slot count via the same formula the kernel
        # side uses in ``get_workspace_size_in_bytes``.
        mma_tiler_n = self.impl.mma_tiler_mnk[1]
        (
            fc1_output_torch,
            fc1_output_sf_torch,
            fc1_done_counter_torch,
            load_balance_counter_torch,
        ) = self._partition_workspace(mma_tiler_n)

        # -- 4. Torch -> cute --
        def _to_cute(tensor: torch.Tensor, assumed_align: int = 16):
            cute_tensor = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
            leading_dim = cutlass_torch.get_leading_dim(tensor)
            return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)

        activation_cute = _to_cute(self.activation)
        fc1_weight_cute = _to_cute(self.fc1_weight)
        activation_sf_cute = _to_cute(self.activation_sf)
        fc1_weight_sf_cute = _to_cute(self.fc1_weight_sf)
        fc1_output_cute = _to_cute(fc1_output_torch)
        fc1_output_sf_cute = _to_cute(fc1_output_sf_torch)
        fc2_weight_cute = _to_cute(self.fc2_weight)
        fc2_weight_sf_cute = _to_cute(self.fc2_weight_sf)
        fc2_output_cute = _to_cute(self.fc2_output)
        topk_scores_cute = _to_cute(self.topk_scores)
        fc1_done_counter_cute = _to_cute(fc1_done_counter_torch, assumed_align=4)
        offs_cute = _to_cute(self.offs)

        # ``load_balance_counter`` is required iff
        # ``load_balance_mode == 'atomic_counter'``.  In 'static' mode the
        # workspace partition returns None and we omit the kwarg entirely;
        # the kernel's __call__ defaults it to None and const_expr-skips
        # the counter wiring.
        load_balance_counter_cute = (
            _to_cute(load_balance_counter_torch, assumed_align=4)
            if load_balance_counter_torch is not None
            else None
        )

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        # -- 5. cute.compile --
        # All kernel inputs go through ``runtime_kwargs``; ``compile_kwargs``
        # adds compile-only items (``max_active_clusters`` Constexpr,
        # cute.compile ``options`` flag).  The two dicts share their
        # runtime keys so the runtime invocation reuses the same dict
        # without recomputing.
        runtime_kwargs = dict(
            activation=activation_cute,
            fc1_weight=fc1_weight_cute,
            activation_sf=activation_sf_cute,
            fc1_weight_sf=fc1_weight_sf_cute,
            fc1_output=fc1_output_cute,
            fc1_output_sf=fc1_output_sf_cute,
            fc2_weight=fc2_weight_cute,
            fc2_weight_sf=fc2_weight_sf_cute,
            fc2_output=fc2_output_cute,
            topk_scores=topk_scores_cute,
            fc1_done_counter=fc1_done_counter_cute,
            offs=offs_cute,
            stream=stream,
        )
        if load_balance_counter_cute is not None:
            runtime_kwargs["load_balance_counter"] = load_balance_counter_cute
        # Optional per-expert alpha (NVFP4 tester only).  Omitted entirely when
        # unset so kinds whose kernel has no alpha param (mxfp8) are unaffected.
        if self.fc1_alpha is not None:
            runtime_kwargs["fc1_alpha"] = _to_cute(self.fc1_alpha, assumed_align=4)
        if self.fc2_alpha is not None:
            runtime_kwargs["fc2_alpha"] = _to_cute(self.fc2_alpha, assumed_align=4)

        compile_kwargs = dict(runtime_kwargs)
        compile_kwargs["max_active_clusters"] = max_active_clusters
        if self.misc.enable_iket:
            compile_kwargs["options"] = "iket"

        compiled_kernel = cute.compile(kernel, **compile_kwargs)
        self._compiled_kernel = compiled_kernel

        # Stash launch-side handles + workspace partition torch views so the
        # debug paths (``_check_kernel_determinism`` re-launches the same
        # kernel; ``_validate_fc1_phase`` reads the kernel-written
        # fc1_output / fc1_output_sf bytes) can do their work without
        # re-running the entire setup.  Cheap; no extra alloc / launch.
        self._ws_fc1_output_torch = fc1_output_torch
        self._ws_fc1_output_sf_torch = fc1_output_sf_torch
        self._launch_runtime_kwargs = runtime_kwargs

        # -- 6. Launch --
        if not self.misc.run_target_kernel_only:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
            ) as prof:
                compiled_kernel(**runtime_kwargs)
                torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=-1))
        else:
            compiled_kernel(**runtime_kwargs)
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Optional debug checks
    # ------------------------------------------------------------------

    def _check_kernel_determinism(self) -> None:
        """Relaunch and byte-compare fc2 output plus fc1 workspace."""
        if (
            self._compiled_kernel is None
            or self._launch_runtime_kwargs is None
            or self.fc2_output is None
            or self._ws_fc1_output_torch is None
            or self._ws_fc1_output_sf_torch is None
        ):
            print("[determinism check] skipped (kernel not launched yet)")
            return

        # Snapshot first-launch outputs as raw bytes.
        fc2_first = self.fc2_output.view(torch.uint8).clone()
        fc1_out_first = self._ws_fc1_output_torch.view(torch.uint8).clone()
        fc1_sf_first = self._ws_fc1_output_sf_torch.view(torch.uint8).clone()

        # Counters require zero-init per launch.
        self.workspace.zero_()

        self._compiled_kernel(**self._launch_runtime_kwargs)
        torch.cuda.synchronize()

        fc2_curr = self.fc2_output.view(torch.uint8)
        fc1_out_curr = self._ws_fc1_output_torch.view(torch.uint8)
        fc1_sf_curr = self._ws_fc1_output_sf_torch.view(torch.uint8)

        fc2_byte_diff = int((fc2_first != fc2_curr).sum().item())
        fc1_out_byte_diff = int((fc1_out_first != fc1_out_curr).sum().item())
        fc1_sf_byte_diff = int((fc1_sf_first != fc1_sf_curr).sum().item())

        print("=" * 60)
        print("[determinism check] re-launched kernel with identical inputs")
        print(
            f"  fc2_output     byte-diff: {fc2_byte_diff:>10d} / "
            f"{fc2_first.numel():>10d} "
            f"({fc2_byte_diff / max(fc2_first.numel(), 1) * 100:7.4f}%)"
        )
        print(
            f"  fc1_output     byte-diff: {fc1_out_byte_diff:>10d} / "
            f"{fc1_out_first.numel():>10d} "
            f"({fc1_out_byte_diff / max(fc1_out_first.numel(), 1) * 100:7.4f}%)"
        )
        print(
            f"  fc1_output_sf  byte-diff: {fc1_sf_byte_diff:>10d} / "
            f"{fc1_sf_first.numel():>10d} "
            f"({fc1_sf_byte_diff / max(fc1_sf_first.numel(), 1) * 100:7.4f}%)"
        )
        if fc2_byte_diff == 0 and fc1_out_byte_diff == 0 and fc1_sf_byte_diff == 0:
            print(
                "  -> kernel is BIT-DETERMINISTIC across re-launches; any "
                "mismatch with ref must come from kernel-vs-host "
                "accumulation order, not HW race"
            )
        else:
            print(
                "  -> kernel is NON-DETERMINISTIC (different bytes on "
                "back-to-back identical launches); race condition / stale "
                "barrier suspected"
            )
        print("=" * 60)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _actual_for_compare(
        self, actual: torch.Tensor, d_start: int, v_e: int
    ) -> torch.Tensor:
        """Make the kernel's fc2 output comparable to the reference.

        In ``transformers`` mode the fused fc12 kernel stores the **unweighted**
        fc2 output (the standalone ``topk_reduce`` kernel applies the per-token
        topk weight downstream), while the reference is topk-weighted.  Emulate
        that downstream weight -- with the same store-dtype round-trip the real
        reduce sees -- so the comparison is apples-to-apples.  ``deepgemm``
        already folds the topk weight into fc1, so the output is returned as-is.
        """
        if self.misc.ref_compute_graph == "transformers":
            topk = self.topk_scores[d_start : d_start + v_e].to(torch.float32)
            actual = actual * topk.unsqueeze(-1)
            actual = actual.to(self.problem.fc2_output_dtype).to(torch.float32)
        return actual

    def validate(self) -> None:
        """Compare kernel output against the reference per-expert.

        Tolerance is supplied by :meth:`_fc2_tolerance` (kind-specific).
        Per-expert valid prefix is sliced via ``data_physical_offsets``;
        padding rows are not compared (the kernel must not write them).

        Optional debug checks re-launch the kernel and inspect the fc1
        workspace (:meth:`_validate_fc1_phase`) before the fc2 comparison.
        """
        if self.misc.skip_ref_check:
            return
        if self.fc2_output_ref is None:
            raise RuntimeError("validate requires compute_reference first.")
        if self.fc2_output is None:
            raise RuntimeError("validate requires generate_inputs first.")

        self._check_kernel_determinism()
        self._validate_fc1_phase()

        atol, rtol = self._fc2_tolerance()

        valid_tokens = self.valid_tokens_per_expert
        data_offsets = self.data_physical_offsets

        # NVFP4: fc2_output is 3D (M, 1, H); squeeze axis 1 for comparison.
        # MXFP8: fc2_output is 2D (M, H); squeeze(1) is a no-op.
        actual_fp32 = self.fc2_output.squeeze(1).to(torch.float32)
        ref_fp32 = self.fc2_output_ref.to(torch.float32)

        print("=" * 60)
        print("[fc2 output] kernel vs ref per expert")
        print(
            f"{'expert':>6} {'v_e':>6} {'max_diff':>12} "
            f"{'mean_diff':>12} {'n_bad':>10} {'%bad':>8}"
        )

        failed_experts = []
        for expert_idx in range(self.problem.experts):
            v_e = valid_tokens[expert_idx]
            if v_e == 0:
                continue
            d_start = data_offsets[expert_idx]
            actual = self._actual_for_compare(
                actual_fp32[d_start : d_start + v_e], d_start, v_e
            )
            ref = ref_fp32[d_start : d_start + v_e]
            diff = (actual - ref).abs()
            max_d = diff.max().item()
            mean_d = diff.mean().item()
            n = diff.numel()
            n_bad = int((diff > atol).sum().item())
            pct = 100.0 * n_bad / max(n, 1)
            print(
                f"{expert_idx:>6} {v_e:>6} {max_d:>12.4g} "
                f"{mean_d:>12.4g} {n_bad:>10} {pct:>7.3f}%"
            )
            if n_bad > 0:
                failed_experts.append(expert_idx)

        print("=" * 60)

        if failed_experts:
            # Print detailed mismatch for each failing expert.
            for expert_idx in failed_experts:
                v_e = valid_tokens[expert_idx]
                d_start = data_offsets[expert_idx]
                actual = self._actual_for_compare(
                    actual_fp32[d_start : d_start + v_e], d_start, v_e
                )
                ref = ref_fp32[d_start : d_start + v_e]
                compare_and_report_mismatches(
                    actual,
                    ref,
                    name=f"fc2_output_expert{expert_idx}",
                    atol=atol,
                    rtol=rtol,
                    max_mismatches=8,
                )
        else:
            print("Validation PASSED (fc2_output within tolerance)")

    # ------------------------------------------------------------------
    # Scheduler layout preview (group-cut + per-expert tile counts)
    # ------------------------------------------------------------------

    def _print_scheduler_layout(self) -> None:
        """Print scheduler-side per-expert + group layout preview.

        Thin wrapper that maps the user-view ``ImplDesc`` into the
        post-swap form ``FusedFc12Simulator`` consumes, then defers to
        ``FusedFc12Simulator.print_layout``.  Always invoked from
        ``run()`` right after the physical-offset summary, before
        reference / kernel execution -- intended as a hand-eyeball check
        of how the static scheduler will pack experts into groups and
        how many fc1 / fc2 work tiles each group emits.

        User-view -> post-swap mapping:
          - ``cta_tile_token``       = ``mma_tiler_mnk[1]``
                                       (user N = post-swap M = token axis)
          - ``cluster_tile_n_post``  = ``mma_tiler_mnk[0] * cluster_shape_mnk[0]``
                                       (user M dim x user cluster_m)
          - ``num_fc1_n_blocks``     = ``ceil_div(intermediate, cluster_tile_n_post)``
          - ``num_fc2_n_blocks``     = ``ceil_div(hidden,       cluster_tile_n_post)``
          - ``group_hint``           = ``impl.group_hint`` (fall back to
                                       ``HardwareInfo().get_max_active_clusters``)
        """
        impl = self.impl
        problem = self.problem
        cta_tile_token = impl.mma_tiler_mnk[1]
        cluster_tile_n_post_swap = impl.mma_tiler_mnk[0] * impl.cluster_shape_mnk[0]
        num_fc1_n_blocks = ceil_div(problem.intermediate, cluster_tile_n_post_swap)
        num_fc2_n_blocks = ceil_div(problem.hidden, cluster_tile_n_post_swap)

        group_hint = impl.group_hint
        if group_hint is None:
            cluster_size = impl.cluster_shape_mnk[0] * impl.cluster_shape_mnk[1]
            group_hint = HardwareInfo().get_max_active_clusters(cluster_size)

        sim = FusedFc12Simulator(
            offsets=self.offs.cpu().tolist(),
            cta_tile_token=cta_tile_token,
            num_fc1_n_blocks=num_fc1_n_blocks,
            num_fc2_n_blocks=num_fc2_n_blocks,
            group_hint=group_hint,
        )

        print()
        sim.print_layout()
        print()

    # ------------------------------------------------------------------
    # Verbose information dump
    # ------------------------------------------------------------------

    def _print_layout_info(self) -> None:
        """Print every host tensor's shape/stride/dtype + the scheduler-layout
        preview.  Gated behind ``misc.verbose`` by the caller."""
        valid_tokens = self.valid_tokens_per_expert
        data_offsets = self.data_physical_offsets
        sf_offsets = self.sf_physical_offsets

        print(
            f"activation: shape={tuple(self.activation.shape)}  "
            f"stride={self.activation.stride()}  dtype={self.activation.dtype}"
        )
        print(
            f"fc1_weight: shape={tuple(self.fc1_weight.shape)}  "
            f"stride={self.fc1_weight.stride()}  dtype={self.fc1_weight.dtype}"
        )
        print(
            f"fc2_weight: shape={tuple(self.fc2_weight.shape)}  "
            f"stride={self.fc2_weight.stride()}  dtype={self.fc2_weight.dtype}"
        )
        print(
            f"activation_sf: shape={tuple(self.activation_sf.shape)}  "
            f"stride={self.activation_sf.stride()}  dtype={self.activation_sf.dtype}"
        )
        print(
            f"fc1_weight_sf: shape={tuple(self.fc1_weight_sf.shape)}  "
            f"stride={self.fc1_weight_sf.stride()}  dtype={self.fc1_weight_sf.dtype}"
        )
        print(
            f"fc2_weight_sf: shape={tuple(self.fc2_weight_sf.shape)}  "
            f"stride={self.fc2_weight_sf.stride()}  dtype={self.fc2_weight_sf.dtype}"
        )
        print(
            f"topk_scores: shape={tuple(self.topk_scores.shape)}  "
            f"stride={self.topk_scores.stride()}  dtype={self.topk_scores.dtype}"
        )
        print(
            f"fc2_output: shape={tuple(self.fc2_output.shape)}  "
            f"stride={self.fc2_output.stride()}  dtype={self.fc2_output.dtype}"
        )
        print(
            f"workspace: shape={tuple(self.workspace.shape)}  "
            f"stride={self.workspace.stride()}  dtype={self.workspace.dtype}"
        )
        if not self.misc.run_target_kernel_only:
            print(
                f"activation_global_scale: {self.activation_global_scale.cpu().tolist()}"
            )
            print(
                f"fc1_weight_global_scale: {self.fc1_weight_global_scale.cpu().tolist()}"
            )
            print(
                f"fc2_weight_global_scale: {self.fc2_weight_global_scale.cpu().tolist()}"
            )
        print(f"offs (valid cumsum): {self.offs.cpu().tolist()}")
        print(f"  valid_tokens_per_expert: {valid_tokens}")
        print(f"  data_physical_offsets:   {data_offsets}")
        print(f"  sf_physical_offsets:     {sf_offsets}")

        self._print_scheduler_layout()

    # ------------------------------------------------------------------
    # Top-level entry
    # ------------------------------------------------------------------

    def run(self) -> None:
        # generate test input / output tensors
        self.generate_inputs()

        if self.misc.verbose:
            print(self.problem)
            print(self.impl)
            print(self.misc)
            self._print_layout_info()

        # ``perf_run`` would imply a benchmark + skip_ref path, but the
        # benchmark path is intentionally not built in this runner.  Honour
        # ``perf_run`` here only as a "use random-byte" data switch (already
        # wired through the per-kind tensor creation hooks).
        skip_ref = (
            self.misc.skip_ref_check
            or self.misc.run_target_kernel_only
            or self.misc.perf_run
        )

        if not skip_ref:
            self.compute_reference()
            if self.fc2_output_ref is not None:
                print(
                    f"fc2_output_ref: shape={tuple(self.fc2_output_ref.shape)}  "
                    f"stride={self.fc2_output_ref.stride()}  "
                    f"dtype={self.fc2_output_ref.dtype}"
                )

        # Kernel launch.  ``run_kernel`` instantiates the fused kernel,
        # queries / partitions the workspace, transitions torch tensors to
        # cute, compiles, and launches.
        self.run_kernel()

        if not skip_ref:
            self.validate()
        print("DONE")

    # ------------------------------------------------------------------
    # Kind-entangled hooks (implemented by concrete subclasses)
    # ------------------------------------------------------------------

    def _fc2_output_shape(self, data_total_rows: int) -> Tuple[int, ...]:
        """Storage shape of ``fc2_output`` for this kind."""
        raise NotImplementedError

    def _create_input_data_tensors(self, data_total_rows: int) -> None:
        """Populate ``self.activation`` / ``fc1_weight`` / ``fc2_weight``."""
        raise NotImplementedError

    def _alloc_fc2_output(self, data_total_rows: int) -> None:
        """Allocate ``self.fc2_output`` with the 0xFF sentinel byte fill."""
        raise NotImplementedError

    def _quantize_fc1(
        self, swiglu: torch.Tensor, norm_const_val: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize the post-SwiGLU fc1 output; return ``(q, raw_sf)``."""
        raise NotImplementedError

    def _apply_topk_post_fc2(
        self, fc2_fp32: torch.Tensor, topk_slice: torch.Tensor
    ) -> torch.Tensor:
        """Optionally post-multiply topk weight after the fc2 GEMM.

        Default: no-op.
        """
        return fc2_fp32

    def _instantiate_kernel(self, common_kwargs: dict):
        """Construct and return the kind-specific fused fc12 kernel object."""
        raise NotImplementedError

    def _validate_fc1_phase(self) -> None:
        """Read back kernel-written fc1 workspace and compare to reference."""
        raise NotImplementedError

    def _fc2_tolerance(self) -> Tuple[float, float]:
        """Return ``(atol, rtol)`` for the fc2-output comparison."""
        raise NotImplementedError


# =============================================================================
# Shared CLI helpers
# =============================================================================


def parse_tuple(s: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def parse_output_dtype(s: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
    }
    if s not in mapping:
        raise argparse.ArgumentTypeError(
            f"fc2_output_dtype must be one of {sorted(mapping.keys())}, got {s!r}"
        )
    return mapping[s]


def add_common_fc12_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the argparse flags shared by both fused-fc12 runners.

    Excludes ``--kind`` and any kind-specific flags (e.g. NVFP4's
    ``--use_bulk_fc2_store`` / ``--in_kernel_fc2_reduce``), which each
    runner adds for itself.
    """
    # -- Problem --
    parser.add_argument(
        "--tokens_after_topk",
        type=int,
        default=2048,
        help="Total VALID token slots after top-k routing (NOT padded).",
    )
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument(
        "--balance_route",
        action="store_true",
        default=False,
        help="Distribute tokens evenly across experts; otherwise Dirichlet(alpha=0.5).",
    )
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument(
        "--simulate_ep",
        type=int,
        default=None,
        help="Simulate Expert Parallelism by reducing per-rank tokens.",
    )
    parser.add_argument(
        "--fc2_output_dtype",
        type=parse_output_dtype,
        default=torch.bfloat16,
        help="fc2 output dtype: bf16 (default) or fp16.",
    )

    # -- Impl --
    parser.add_argument(
        "--mma_tiler_mnk",
        type=str,
        default="128,128,256",
        help="Comma-separated (M, N, K); K is refined at compile time.",
    )
    parser.add_argument("--cluster_shape_mnk", type=str, default="1,1,1")
    parser.add_argument("--use_2cta_instrs", action="store_true", default=False)
    parser.add_argument(
        "--enable_static_expert_shape",
        action="store_true",
        default=False,
        help=(
            "Bind ``static_expert_shape = (experts, intermediate, hidden)`` "
            "at codegen time, taken from the ProblemDesc.  Default (off) "
            "binds the three dims as dynamic Int32 values at launch."
        ),
    )
    parser.add_argument(
        "--dynamic_sched",
        action="store_true",
        default=False,
        help="Use CLC-based dynamic scheduler (default: static lean 7-warp).",
    )
    parser.add_argument(
        "--clc_bundle_size",
        type=int,
        default=None,
        help="Static CLC bundle size S (only meaningful when --dynamic_sched).",
    )
    parser.add_argument(
        "--num_sched_stages",
        type=int,
        default=None,
        help="Scheduler pipeline stages (only meaningful when --dynamic_sched).",
    )
    parser.add_argument(
        "--load_balance_mode",
        type=str,
        default="static",
        choices=["static", "atomic_counter"],
        help=(
            "Load-balance strategy for the fused-fc12 scheduler.  "
            "'static' is the v1 default (stride mode); 'atomic_counter' "
            "uses a GMEM counter to claim cluster-linear tile ids."
        ),
    )
    parser.add_argument(
        "--group_hint",
        type=int,
        default=None,
        help="Per-group fc1 tile threshold; None means defer to "
        "HardwareInfo().get_max_active_clusters(cluster_size).",
    )

    # -- Misc --
    parser.add_argument(
        "--perf_run",
        action="store_true",
        default=False,
        help="Use random-byte FP4/FP8 data instead of sparse {0, +/-1} (data switch only).",
    )
    parser.add_argument(
        "--skip_ref_check",
        action="store_true",
        default=False,
        help="Skip both compute_reference and validate.",
    )
    parser.add_argument(
        "--run_target_kernel_only",
        action="store_true",
        default=False,
        help="Only for perf simulators: all tensors except offs are empty / undefined.",
    )
    parser.add_argument(
        "--enable_debug_checks",
        action="store_true",
        default=False,
        help="Run determinism and fc1 workspace diagnostics during validate.",
    )
    parser.add_argument(
        "--ref_compute_graph",
        type=str,
        default="deepgemm",
        choices=["transformers", "deepgemm"],
        help=(
            "Reference compute graph (see MiscDesc docstring).  "
            "'deepgemm' (default): Path A -topk weight pre-multiplied "
            "into swiglu fp32 BEFORE NVFP4 quantize, matches the kernel.  "
            "'transformers': HF semantics -topk weight applied AFTER "
            "fc2 GEMM in fp32, then cast to bf16/fp16.  (NVFP4 only; the "
            "MXFP8 reference applies no topk weighting.)"
        ),
    )
    parser.add_argument(
        "--enable_iket",
        action="store_true",
        default=False,
        help=(
            'Compile with ``options="iket"`` so ``iket.range_push`` / '
            "``iket.range_pop`` / ``iket.mark`` ops survive ``strip-iket-ops`` "
            "and reach LLVM lowering.  Pair with ``run-iket --output-dir ... "
            "profile -- env DKG_IKET_INSTRUMENTATION_METHOD=NativeDump python "
            "<runner>.py --enable_iket ...`` to produce a warp-phase trace."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help=(
            "Print the per-tensor layout dump (shape/stride/dtype) and the "
            "scheduler-layout preview before the kernel runs.  Off by default."
        ),
    )
