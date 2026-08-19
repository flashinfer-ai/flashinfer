# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""SM107 (Rubin) block-scaled swap-AB inference mega-MoE frontend.

Wraps the vendored ``sources.kernel_src.rubin.inference.mega`` kernel
(``BlockScaledSwapAbMegaMoeKernel``) behind the standard two-entry-point mega
contract:

- :func:`get_symm_buffer_for_sm107_block_scaled_mega_moe` — workspace allocator
- :func:`sm107_block_scaled_mega_moe` — fused dispatch + FC1 + SwiGLU + FC2 +
  combine compute entry

The kernel is generic over the drop's ``QuantKind``; this shim wires up the
``nvfp4`` and ``mxfp8_e4m3`` / ``mxfp8_e5m2`` kinds (``mxfp4`` /
``mxfp4_mxfp8`` need a w4 weight-transform path and are not exposed yet).

All ``sources`` / ``cutlass`` imports are function-local so importing this
module stays CPU-safe (the package ``__init__`` re-exports from here).

The staging/launch protocol mirrors the drop's own runner
(``next/repo_internal_only/test_megamoe_rubin.py``): activation, activation
SF, topk scores, the shared workspace, and (for the in-kernel-reduce path) the
output live on the symmetric heap; routing indices are local int32 (16-byte
aligned); workspaces are 128-byte aligned with their leading bytes zeroed
before the first launch.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Literal, Optional, Tuple

import torch

from . import comm
from .kernel_helpers import Mxfp8BlockSize, Nvfp4BlockSize, swizzled_flat_sf_size

Sm107QuantKind = Literal["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"]
Sm107TokenBackMode = Literal["epi_warps", "standalone_warps", "reuse_dispatch_warps"]
Sm107WorkIdMode = Literal["grid_stride", "atomic_counter"]
Sm107ScheduleMode = Literal["grouped", "phase_interleave"]

# (act data dtype, act SF dtype, sf_vec_size, 2x-mode instruction K) per kind.
# The mxf8f6f4 UMMA path has instruction K 32 (64 in 2x mode); the fp4 pair
# kinds have 64 (128 in 2x mode) — see quant_def.QuantKind.instruction_k.
_KIND_TABLE: dict = {
    "nvfp4": (None, torch.float8_e4m3fn, Nvfp4BlockSize, 128),
    "mxfp8_e4m3": (torch.float8_e4m3fn, torch.float8_e8m0fnu, Mxfp8BlockSize, 64),
    "mxfp8_e5m2": (torch.float8_e5m2, torch.float8_e8m0fnu, Mxfp8BlockSize, 64),
}

TransformedBlockScaledWeights = Tuple[torch.Tensor, torch.Tensor]


def _require_sm107() -> None:
    cc = torch.cuda.get_device_capability()
    if cc != (10, 7):
        raise RuntimeError(
            f"the SM107 block-scaled mega kernel requires compute capability "
            f"(10, 7) (Rubin); this device reports {cc}."
        )


def _configure_dsl_arch() -> None:
    configured = os.environ.get("CUTE_DSL_ARCH")
    if configured not in (None, "sm_107", "sm_107a"):
        raise RuntimeError(
            f"CUTE_DSL_ARCH must target SM107 for the Rubin mega kernel, "
            f"got {configured!r}."
        )
    os.environ["CUTE_DSL_ARCH"] = "sm_107a"


def _fp4_storage_dtype() -> torch.dtype:
    return getattr(torch, "float4_e2m1fn_x2", torch.uint8)


def _to_cute(
    tensor: torch.Tensor, assumed_align: int = 16, dynamic_leading: bool = True
):
    """torch -> cute tensor via dlpack; leading dim marked dynamic by default."""
    import cutlass.torch as cutlass_torch

    cute_tensor = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
    if not dynamic_leading:
        return cute_tensor
    leading_dim = cutlass_torch.get_leading_dim(tensor)
    return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)


def _to_cute_ptr(tensor: torch.Tensor, assumed_align: int = 128):
    """Raw cute ``Uint8`` gmem pointer at ``tensor``'s base (opaque workspaces)."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.typing import AddressSpace

    return cute.runtime.make_ptr(
        cutlass.Uint8, tensor.data_ptr(), AddressSpace.gmem, assumed_align=assumed_align
    )


def _cu_stream(stream: torch.cuda.Stream):
    import cuda.bindings.driver as cuda

    return cuda.CUstream(stream.cuda_stream)


_MAX_ACTIVE_CLUSTERS_CACHE: dict = {}


def _max_active_clusters(cluster_size: int) -> int:
    """Occupancy probe (compiles a helper kernel; cached per cluster size)."""
    cached = _MAX_ACTIVE_CLUSTERS_CACHE.get(cluster_size)
    if cached is None:
        import cutlass.utils as cutlass_utils

        cached = int(cutlass_utils.HardwareInfo().get_max_active_clusters(cluster_size))
        _MAX_ACTIVE_CLUSTERS_CACHE[cluster_size] = cached
    return cached


@dataclasses.dataclass(frozen=True)
class Sm107BlockScaledMoeConfig:
    """Validated construction parameters for ``BlockScaledSwapAbMegaMoeKernel``."""

    num_total_experts: int
    max_tokens_per_rank: int
    num_topk: int
    hidden: int
    intermediate: int  # post-SwiGLU width; the FC1 GEMM N is 2*intermediate
    rank: int
    world_size: int
    quant_kind: Sm107QuantKind = "mxfp8_e4m3"
    mma_tiler_mnk: Optional[Tuple[int, int, int]] = None  # None -> (256, 128, 4*ik)
    cluster_shape_mn: Tuple[int, int] = (2, 1)
    # Mixed-CGA launch: alongside the preferred clusters, fill leftover SMs with
    # smaller fallback clusters (upstream commit a5b4d33). None = uniform launch.
    fallback_cluster_shape_mn: Optional[Tuple[int, int]] = None
    # ("grouped" | "phase_interleave", optional hint); typed loosely so backend
    # configs can carry it as a plain (str, int|None) tuple.
    schedule_policy: Tuple[str, Optional[int]] = ("grouped", None)
    work_id_mode: Sm107WorkIdMode = "grid_stride"
    fc2_use_bulk: bool = False
    fc2_tma_stages: Optional[int] = None
    epi_flag_batches: Tuple[int, int] = (4, 2)
    token_in_flag_batch: int = 1
    token_padding_block: int = 64
    sf_padding_block: int = 128
    gate_up_clamp: Optional[float] = None
    reduce_topk_in_kernel: bool = False
    token_back_mode: Sm107TokenBackMode = "epi_warps"
    apply_topk_at_fc1: bool = True
    max_sm_count: Optional[int] = None

    def __post_init__(self) -> None:
        if self.quant_kind not in _KIND_TABLE:
            raise ValueError(f"unsupported quant_kind {self.quant_kind!r}.")
        if self.world_size < 1 or not (0 <= self.rank < self.world_size):
            raise ValueError(f"invalid rank/world_size {self.rank}/{self.world_size}.")
        if self.num_total_experts % self.world_size != 0:
            raise ValueError(
                f"num_total_experts ({self.num_total_experts}) must divide evenly "
                f"across world_size ({self.world_size})."
            )
        vec = self.sf_vec_size
        if self.hidden % (2 * vec) != 0:
            raise ValueError(
                f"hidden ({self.hidden}) must be a multiple of {2 * vec} "
                f"for {self.quant_kind}."
            )
        if self.intermediate % (2 * vec) != 0 or self.intermediate % 32 != 0:
            raise ValueError(
                f"intermediate ({self.intermediate}) must be a multiple of "
                f"{max(2 * vec, 32)} (gate/up interleave + SF blocks) for "
                f"{self.quant_kind}."
            )
        tiler = self.resolved_mma_tiler_mnk
        instruction_k = self.instruction_k
        if len(tiler) != 3:
            raise ValueError("mma_tiler_mnk must be a 3-tuple.")
        if tiler[0] not in (128, 256):
            raise ValueError(f"Rubin instruction M must be 128 or 256, got {tiler[0]}.")
        if tiler[1] not in (64, 128, 256):
            raise ValueError(
                f"Rubin instruction N must be 64, 128, or 256, got {tiler[1]}."
            )
        if tiler[2] not in (2 * instruction_k, 4 * instruction_k):
            raise ValueError(
                f"Rubin {self.quant_kind} tile K must be {2 * instruction_k} or "
                f"{4 * instruction_k}, got {tiler[2]}."
            )
        if (
            self.quant_kind == "nvfp4"
            and tiler[1] == 256
            and tiler[2] == 4 * instruction_k
        ):
            raise ValueError("nvfp4 does not support (N=256, K=4x-instruction) tiles.")
        if len(self.cluster_shape_mn) != 2:
            raise ValueError("cluster_shape_mn must be a 2-tuple.")
        if tiler[0] == 256 and self.cluster_shape_mn[0] % 2 != 0:
            raise ValueError("instruction M 256 requires an even cluster M.")
        if self.fallback_cluster_shape_mn is not None:
            fallback = tuple(self.fallback_cluster_shape_mn)
            if fallback == tuple(self.cluster_shape_mn):
                # Kernel-side NonClcMixedCgaConfig collapses this to a uniform
                # launch; normalize here so downstream logic sees one spelling.
                object.__setattr__(self, "fallback_cluster_shape_mn", None)
            else:
                if len(fallback) != 2:
                    raise ValueError("fallback_cluster_shape_mn must be a 2-tuple.")
                if any(dim <= 0 for dim in fallback):
                    raise ValueError(
                        "fallback_cluster_shape_mn dimensions must be positive."
                    )
                if fallback[1] != 1:
                    raise ValueError(
                        "the swap-AB FC12 fallback path requires cluster N=1."
                    )
                if any(
                    preferred % fb != 0
                    for preferred, fb in zip(
                        self.cluster_shape_mn, fallback, strict=True
                    )
                ):
                    raise ValueError(
                        "every preferred cluster dimension must be divisible by "
                        "its fallback dimension."
                    )
                if tiler[0] == 256 and fallback[0] % 2 != 0:
                    raise ValueError(
                        "instruction M 256 requires an even fallback cluster M."
                    )
                if self.max_sm_count is not None:
                    raise ValueError(
                        "max_sm_count is not supported together with "
                        "fallback_cluster_shape_mn (mixed-CGA occupancy is "
                        "resolved from the hardware probe)."
                    )
        if self.token_padding_block % 64 != 0:
            raise ValueError("token_padding_block must be a multiple of 64.")
        if tiler[1] % self.token_padding_block != 0:
            raise ValueError(
                f"mma tiler N ({tiler[1]}) must be a whole number of token "
                f"padding blocks ({self.token_padding_block})."
            )
        mode, hint = self.schedule_policy
        if mode not in ("grouped", "phase_interleave"):
            raise ValueError(f"unknown schedule mode {mode!r}.")
        if hint is not None and hint <= 0:
            raise ValueError("schedule hint must be positive or None.")
        if mode == "phase_interleave" and self.work_id_mode != "atomic_counter":
            raise ValueError(
                "schedule_policy 'phase_interleave' requires "
                "work_id_mode='atomic_counter'."
            )
        if len(self.epi_flag_batches) != 2:
            raise ValueError("epi_flag_batches requires (FC1, FC2) values.")
        if max(self.epi_flag_batches) > 4:
            raise ValueError("Rubin epi_flag_batches values must be <= 4.")
        if self.fc2_tma_stages is not None and not (
            1 <= self.fc2_tma_stages <= tiler[1] // 64
        ):
            raise ValueError(
                f"fc2_tma_stages must be in [1, {tiler[1] // 64}] for tiler N "
                f"{tiler[1]}."
            )
        if self.reduce_topk_in_kernel and not self.apply_topk_at_fc1:
            raise ValueError(
                "reduce_topk_in_kernel requires apply_topk_at_fc1=True (the "
                "in-kernel reduce red-adds already-weighted terms)."
            )

    @property
    def experts_per_rank(self) -> int:
        return self.num_total_experts // self.world_size

    @property
    def sf_vec_size(self) -> int:
        return _KIND_TABLE[self.quant_kind][2]

    @property
    def instruction_k(self) -> int:
        return _KIND_TABLE[self.quant_kind][3]

    @property
    def resolved_mma_tiler_mnk(self) -> Tuple[int, int, int]:
        if self.mma_tiler_mnk is not None:
            return self.mma_tiler_mnk
        return (256, 128, 2 * self.instruction_k)

    @property
    def torch_act_data_dtype(self) -> torch.dtype:
        dtype = _KIND_TABLE[self.quant_kind][0]
        return _fp4_storage_dtype() if dtype is None else dtype

    @property
    def torch_act_sf_dtype(self) -> torch.dtype:
        return _KIND_TABLE[self.quant_kind][1]


class Sm107BlockScaledSymmBuffer:
    """Session workspace for the SM107 block-scaled inference mega kernel.

    Owns the staging tensors the backend fills (``x``, ``x_sf``, ``topk_idx``,
    ``topk_weights``), the kernel + device workspaces, and the compile cache.
    Create via :func:`get_symm_buffer_for_sm107_block_scaled_mega_moe`.
    """

    def __init__(self, config: Sm107BlockScaledMoeConfig) -> None:
        comm.ensure_not_capturing("SM107 mega workspace allocation")
        _require_sm107()
        _configure_dsl_arch()
        self.config = config
        cfg = config

        kernel = self._build_kernel(cfg)
        self.kernel = kernel

        tokens = cfg.max_tokens_per_rank

        # Staging tensors. Activation / SF / scores are pulled by peer dispatch
        # warps, so they live on the symmetric heap; the routing indices are a
        # local (16B-aligned) int32 tensor read by this rank's router only.
        if cfg.quant_kind == "nvfp4":
            self.x = comm.sym_zeros((tokens, cfg.hidden // 2), cfg.torch_act_data_dtype)
        else:
            self.x = comm.sym_zeros((tokens, cfg.hidden), cfg.torch_act_data_dtype)
        self.sf_cols = int(kernel.token_comm.activation_sf_hidden_padded)
        self.x_sf = comm.sym_zeros((tokens, self.sf_cols), cfg.torch_act_sf_dtype)
        self.topk_weights = comm.sym_zeros((tokens, cfg.num_topk), torch.float32)
        self.topk_idx = torch.full(
            (tokens, cfg.num_topk), -1, dtype=torch.int32, device="cuda"
        )
        if self.topk_idx.data_ptr() % 16 != 0:
            raise RuntimeError("routing index tensor must be 16-byte aligned.")

        # In-kernel reduce: the combine peer-writes (red.add) each topk term
        # straight into the output, so it must live on the SYMMETRIC heap and
        # start zeroed; the separate-reduce path only sees local stores.
        if cfg.reduce_topk_in_kernel:
            self.output_activation = comm.sym_zeros(
                (tokens, cfg.hidden), torch.bfloat16
            )
        else:
            self.output_activation = torch.zeros(
                (tokens, cfg.hidden), dtype=torch.bfloat16, device="cuda"
            )

        workspace = kernel._device_workspace
        local_bytes = int(workspace.total_bytes("local"))
        shared_bytes = int(workspace.total_bytes("shared"))
        # Allocated fully zeroed, which covers require_zero_workspace_leading_bytes.
        self.local_workspace = torch.zeros(
            (max(local_bytes, 1),), dtype=torch.uint8, device="cuda"
        )
        self.shared_workspace = comm.sym_zeros((max(shared_bytes, 1),), torch.uint8)
        if (
            self.local_workspace.data_ptr() % 128 != 0
            or self.shared_workspace.data_ptr() % 128 != 0
        ):
            raise RuntimeError("SM107 mega workspaces must be 128-byte aligned.")

        base, peer_offsets = comm.compute_peer_offsets(
            self.shared_workspace, cfg.world_size
        )
        self._symmetric_base = base
        self._peer_offsets = peer_offsets

        self._compiled: Optional[Any] = None
        self._launch_key: Optional[tuple] = None
        self._launch_kwargs: Optional[dict] = None
        self._staged_tokens: Optional[int] = None
        self._destroyed = False

    @staticmethod
    def _build_kernel(cfg: Sm107BlockScaledMoeConfig):
        import cutlass
        from cutlass.cute.nvgpu import OperandMajorMode

        from sources.api import ImplDesc, ProblemDesc
        from sources.kernel_src.rubin.inference.mega import (
            BlockScaledSwapAbMegaMoeKernel,
        )
        from sources.quant_def import CombineFormat

        tiler = cfg.resolved_mma_tiler_mnk
        cluster_size = cfg.cluster_shape_mn[0] * cfg.cluster_shape_mn[1]
        preferred_count: Optional[int] = None
        fallback_count: Optional[int] = None
        if cfg.fallback_cluster_shape_mn is None:
            launch_clusters = _max_active_clusters(cluster_size)
            if cfg.max_sm_count is not None:
                requested = cfg.max_sm_count // cluster_size
                if requested <= 0:
                    raise ValueError(
                        "max_sm_count must cover at least one full cluster."
                    )
                launch_clusters = min(launch_clusters, requested)
        else:
            # Mirror the drop's launch_cluster_configuration()/max_active_clusters()
            # mixed recipe (tester/host_utils.py): pack preferred clusters at their
            # occupancy limit, then convert the leftover fallback CTA capacity into
            # whole preferred-sized groups of fallback clusters.
            fallback_size = (
                cfg.fallback_cluster_shape_mn[0] * cfg.fallback_cluster_shape_mn[1]
            )
            if cluster_size % fallback_size != 0:
                raise ValueError(
                    "preferred cluster size must be a multiple of the fallback "
                    "cluster size."
                )
            preferred_count = _max_active_clusters(cluster_size)
            fallback_occupancy = _max_active_clusters(fallback_size)
            fallback_capacity = fallback_occupancy * fallback_size
            preferred_capacity = preferred_count * cluster_size
            if fallback_capacity < preferred_capacity:
                raise ValueError(
                    "fallback cluster CTA capacity must not be smaller than "
                    "preferred cluster CTA capacity."
                )
            split_factor = cluster_size // fallback_size
            remaining = (fallback_capacity - preferred_capacity) // fallback_size
            fallback_count = remaining // split_factor * split_factor
            launch_clusters = preferred_count + fallback_count // split_factor

        problem_desc = ProblemDesc(
            {
                "expert_count": cfg.num_total_experts,
                "intermediate_gateup_size": 2 * cfg.intermediate,
                "hidden_size": cfg.hidden,
                "quant_kind": cfg.quant_kind,
                "a_major_mode": OperandMajorMode.K,
                "b_major_mode": OperandMajorMode.K,
                "combine_format": CombineFormat.parse("bf16"),
                "gate_up_clamp": cfg.gate_up_clamp,
                "world_size": cfg.world_size,
                "topk": cfg.num_topk,
                "topk_index_dtype": cutlass.Int32,
                "max_tokens_per_rank": cfg.max_tokens_per_rank,
                "apply_topk_at_fc1": cfg.apply_topk_at_fc1,
            }
        )
        impl_fields = {
            "mma_instruction_mnk": (tiler[0], tiler[1], cfg.instruction_k),
            "mma_tiler_mnk": tiler,
            "mma_k_mode": "2x",
            "cluster_shape_mn": tuple(cfg.cluster_shape_mn),
            "use_2cta_instrs": tiler[0] == 256,
            "schedule_policy": tuple(cfg.schedule_policy),
            "token_padding_block": cfg.token_padding_block,
            "sf_padding_block": cfg.sf_padding_block,
            "work_id_mode": cfg.work_id_mode,
            "fc2_use_bulk": cfg.fc2_use_bulk,
            "epi_flag_batches": tuple(cfg.epi_flag_batches),
            "launch_cluster_count": launch_clusters,
            "token_in_flag_batch": cfg.token_in_flag_batch,
            "token_back_mode": cfg.token_back_mode,
            "reduce_topk_in_kernel": cfg.reduce_topk_in_kernel,
        }
        if cfg.fc2_tma_stages is not None:
            impl_fields["fc2_tma_stages"] = cfg.fc2_tma_stages
        if cfg.fallback_cluster_shape_mn is not None:
            impl_fields["fallback_cluster_shape_mn"] = tuple(
                cfg.fallback_cluster_shape_mn
            )
            impl_fields["preferred_cluster_count"] = preferred_count
            impl_fields["fallback_cluster_count"] = fallback_count
        return BlockScaledSwapAbMegaMoeKernel(problem_desc, ImplDesc(impl_fields))

    def note_staged_tokens(self, num_tokens: int) -> None:
        self._staged_tokens = int(num_tokens)

    def staged_tokens(self) -> Optional[int]:
        return self._staged_tokens

    def _peer_mapper(self):
        from sources.communication.nvlink_domain.symmetric_buffer import (
            SymmetricBufferHost,
        )

        return SymmetricBufferHost(
            base_address=self._symmetric_base,
            offsets=tuple(self._peer_offsets),
            rank=self.config.rank,
            max_ranks=self.config.world_size,
        )

    def _runtime_kwargs(
        self,
        transformed_l1: TransformedBlockScaledWeights,
        transformed_l2: TransformedBlockScaledWeights,
    ) -> dict:
        stream = torch.cuda.current_stream()
        return {
            "activation": _to_cute(self.x),
            "activation_sf": _to_cute(self.x_sf),
            "topk_indices": _to_cute(self.topk_idx),
            "topk_scores": _to_cute(self.topk_weights, assumed_align=4),
            "fc1_weight": _to_cute(transformed_l1[0]),
            "fc1_weight_sf": _to_cute(transformed_l1[1]),
            "fc2_weight": _to_cute(transformed_l2[0]),
            "fc2_weight_sf": _to_cute(transformed_l2[1]),
            "output_activation": _to_cute(self.output_activation),
            "local_workspace": _to_cute_ptr(self.local_workspace),
            "shared_workspace": _to_cute_ptr(self.shared_workspace),
            "peer_rank_ptr_mapper_host": self._peer_mapper(),
            "stream": _cu_stream(stream),
            # nvfp4 per-expert dequant scalars (fc1_alpha / fc2_alpha /
            # fc1_norm_const) are omitted: the weight/staging transforms
            # quantize with norm_const=1.0, so the scalars are identically 1
            # and the epilogue's const_expr None-path is exact.
        }

    def launch(
        self,
        transformed_l1: TransformedBlockScaledWeights,
        transformed_l2: TransformedBlockScaledWeights,
    ) -> None:
        """Compile on first use, then launch the fused mega kernel."""
        import cutlass.cute as cute

        key = (
            transformed_l1[0].data_ptr(),
            transformed_l1[1].data_ptr(),
            transformed_l2[0].data_ptr(),
            transformed_l2[1].data_ptr(),
            torch.cuda.current_stream().cuda_stream,
        )
        if self._compiled is None or self._launch_key != key:
            comm.ensure_not_capturing("SM107 mega kernel compile")
            kwargs = self._runtime_kwargs(transformed_l1, transformed_l2)
            if self._compiled is None:
                self._compiled = cute.compile(self.kernel, **kwargs)
            self._launch_key = key
            self._launch_kwargs = kwargs
        if self.config.reduce_topk_in_kernel:
            # red.add accumulation base: zero the output every launch.
            self.output_activation.zero_()
        self._compiled(**self._launch_kwargs)

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        comm.ensure_not_capturing("SM107 mega workspace free")
        # nvshmem free is collective and does not wait for in-flight work;
        # drain this rank's device before releasing symmetric memory that a
        # still-running launch (ours or a peer's dispatch pull) may touch.
        torch.cuda.synchronize()
        self._compiled = None
        self._launch_kwargs = None
        for name in ("x", "x_sf", "topk_weights", "shared_workspace"):
            comm.free_sym_tensor(getattr(self, name, None))
        if self.config.reduce_topk_in_kernel:
            comm.free_sym_tensor(self.output_activation)


def get_symm_buffer_for_sm107_block_scaled_mega_moe(
    num_total_experts: int,
    num_max_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate: int,
    rank: int,
    world_size: int,
    *,
    quant_kind: Sm107QuantKind = "mxfp8_e4m3",
    mma_tiler_mnk: Optional[Tuple[int, int, int]] = None,
    cluster_shape_mn: Tuple[int, int] = (2, 1),
    fallback_cluster_shape_mn: Optional[Tuple[int, int]] = None,
    schedule_policy: Tuple[str, Optional[int]] = ("grouped", None),
    work_id_mode: Sm107WorkIdMode = "grid_stride",
    fc2_use_bulk: bool = False,
    fc2_tma_stages: Optional[int] = None,
    epi_flag_batches: Tuple[int, int] = (4, 2),
    token_in_flag_batch: int = 1,
    token_padding_block: int = 64,
    sf_padding_block: int = 128,
    gate_up_clamp: Optional[float] = None,
    reduce_topk_in_kernel: bool = False,
    token_back_mode: Sm107TokenBackMode = "epi_warps",
    apply_topk_at_fc1: bool = True,
    max_sm_count: Optional[int] = None,
) -> Sm107BlockScaledSymmBuffer:
    """Allocate the SM107 block-scaled mega session workspace.

    Problem sizes positional, tuning knobs keyword-only (the standard mega
    allocator contract). ``intermediate`` is the post-SwiGLU width. Expert
    weights are NOT owned by the workspace; they are passed per launch.
    """
    config = Sm107BlockScaledMoeConfig(
        num_total_experts=num_total_experts,
        max_tokens_per_rank=num_max_tokens,
        num_topk=num_topk,
        hidden=hidden,
        intermediate=intermediate,
        rank=rank,
        world_size=world_size,
        quant_kind=quant_kind,
        mma_tiler_mnk=mma_tiler_mnk,
        cluster_shape_mn=cluster_shape_mn,
        fallback_cluster_shape_mn=fallback_cluster_shape_mn,
        schedule_policy=schedule_policy,
        work_id_mode=work_id_mode,
        fc2_use_bulk=fc2_use_bulk,
        fc2_tma_stages=fc2_tma_stages,
        epi_flag_batches=epi_flag_batches,
        token_in_flag_batch=token_in_flag_batch,
        token_padding_block=token_padding_block,
        sf_padding_block=sf_padding_block,
        gate_up_clamp=gate_up_clamp,
        reduce_topk_in_kernel=reduce_topk_in_kernel,
        token_back_mode=token_back_mode,
        apply_topk_at_fc1=apply_topk_at_fc1,
        max_sm_count=max_sm_count,
    )
    return Sm107BlockScaledSymmBuffer(config)


def _expected_weight_shapes(
    cfg: Sm107BlockScaledMoeConfig,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], int, int]:
    """(fc1 shape, fc2 shape, fc1 SF numel/expert, fc2 SF numel/expert)."""
    experts = cfg.experts_per_rank
    fc1_out = 2 * cfg.intermediate
    vec = cfg.sf_vec_size
    if cfg.quant_kind == "nvfp4":
        fc1_shape = (experts, cfg.hidden // 2, fc1_out)
        fc2_shape = (experts, cfg.intermediate // 2, cfg.hidden)
    else:
        fc1_shape = (experts, cfg.hidden, fc1_out)
        fc2_shape = (experts, cfg.intermediate, cfg.hidden)
    fc1_sf = swizzled_flat_sf_size(fc1_out, cfg.hidden // vec)
    fc2_sf = swizzled_flat_sf_size(cfg.hidden, cfg.intermediate // vec)
    return fc1_shape, fc2_shape, fc1_sf, fc2_sf


def _validate_weight_leg(
    name: str,
    leg: TransformedBlockScaledWeights,
    expected_weight_shape: Tuple[int, ...],
    expected_sf_numel: int,
) -> None:
    weight, scale = leg
    if tuple(weight.shape) != expected_weight_shape:
        raise ValueError(
            f"{name} weight shape {tuple(weight.shape)} != expected "
            f"{expected_weight_shape}."
        )
    if scale.numel() != expected_sf_numel * expected_weight_shape[0]:
        raise ValueError(
            f"{name} scale numel {scale.numel()} != expected "
            f"{expected_sf_numel * expected_weight_shape[0]}."
        )


def sm107_block_scaled_mega_moe(
    y: Optional[torch.Tensor],
    transformed_l1: TransformedBlockScaledWeights,
    transformed_l2: TransformedBlockScaledWeights,
    symm_buffer: Sm107BlockScaledSymmBuffer,
    *,
    num_tokens: Optional[int] = None,
    fast_math: bool = True,  # accepted for mega API parity; the kernel has no toggle
    sync: bool = False,
) -> Optional[torch.Tensor]:
    """Fused dispatch + FC1 + SwiGLU + FC2 + combine; writes ``y[:num_tokens]``.

    The caller must have staged ``symm_buffer.x`` / ``.x_sf`` and the routing
    slices first. With ``y=None`` returns a workspace view (valid under stream
    ordering until the next launch).
    """
    if not fast_math:
        import warnings

        warnings.warn(
            "fast_math=False is a no-op for the SM107 block-scaled kernel.",
            stacklevel=2,
        )
    cfg = symm_buffer.config
    if num_tokens is None:
        num_tokens = int(y.shape[0]) if y is not None else symm_buffer.staged_tokens()
    if num_tokens is None:
        raise ValueError(
            "num_tokens unset and no tokens were staged; call the backend's "
            "stage_inputs (or note_staged_tokens) first."
        )
    if not (0 <= num_tokens <= cfg.max_tokens_per_rank):
        raise ValueError(
            f"num_tokens ({num_tokens}) out of range [0, {cfg.max_tokens_per_rank}]."
        )

    fc1_shape, fc2_shape, fc1_sf, fc2_sf = _expected_weight_shapes(cfg)
    _validate_weight_leg("fc1", transformed_l1, fc1_shape, fc1_sf)
    _validate_weight_leg("fc2", transformed_l2, fc2_shape, fc2_sf)

    symm_buffer.launch(transformed_l1, transformed_l2)
    if sync:
        torch.cuda.synchronize()

    out = symm_buffer.output_activation[:num_tokens]
    if y is None:
        return out
    y[:num_tokens].copy_(out)
    return None


def sm107_block_scaled_mega_launch_thunk(
    transformed_l1: TransformedBlockScaledWeights,
    transformed_l2: TransformedBlockScaledWeights,
    symm_buffer: Sm107BlockScaledSymmBuffer,
):
    """Zero-arg relauncher over pre-staged inputs, for steady-state timing loops."""

    def _thunk() -> None:
        symm_buffer.launch(transformed_l1, transformed_l2)

    return _thunk


__all__ = [
    "Sm107BlockScaledMoeConfig",
    "Sm107BlockScaledSymmBuffer",
    "Sm107QuantKind",
    "TransformedBlockScaledWeights",
    "get_symm_buffer_for_sm107_block_scaled_mega_moe",
    "sm107_block_scaled_mega_launch_thunk",
    "sm107_block_scaled_mega_moe",
]
