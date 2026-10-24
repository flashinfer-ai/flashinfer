# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Workspace, descriptor construction, compilation and launch for Blackwell GenPhase.

The Blackwell frontend supplies native device checks and compiler targets.
The private source package is imported only inside this shim; DSL imports stay lazy.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Literal, Optional, Tuple

import torch

from flashinfer.moe_ep._compat import deprecated_aliases

from . import comm
from .kernel_helpers import Mxfp8BlockSize, Nvfp4BlockSize, swizzled_flat_sf_size

BlockScaledQuantKind = Literal["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2", "mxfp4_mxfp8"]
BlockScaledTokenBackMode = Literal[
    "epi_warps", "standalone_warps", "reuse_dispatch_warps"
]
BlockScaledWorkIdMode = Literal["grid_stride", "atomic_counter"]
BlockScaledScheduleMode = Literal["grouped", "phase_interleave"]

# (act data dtype, act SF dtype, sf_vec_size, 2x-mode instruction K) per kind.
# The mxf8f6f4 UMMA path has instruction K 32 (64 in 2x mode); the fp4 pair
# kinds have 64 (128 in 2x mode) — see quant_def.QuantKind.instruction_k.
_KIND_TABLE: dict = {
    "nvfp4": (None, torch.float8_e4m3fn, Nvfp4BlockSize, 128),
    "mxfp8_e4m3": (torch.float8_e4m3fn, torch.float8_e8m0fnu, Mxfp8BlockSize, 64),
    "mxfp8_e5m2": (torch.float8_e5m2, torch.float8_e8m0fnu, Mxfp8BlockSize, 64),
    "mxfp4_mxfp8": (torch.float8_e4m3fn, torch.float8_e8m0fnu, Mxfp8BlockSize, 64),
}

TransformedBlockScaledWeights = Tuple[torch.Tensor, torch.Tensor]


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
    """Occupancy probe, cached per device and cluster size."""
    key = (torch.cuda.current_device(), cluster_size)
    cached = _MAX_ACTIVE_CLUSTERS_CACHE.get(key)
    if cached is None:
        import cutlass.utils as cutlass_utils

        cached = int(cutlass_utils.HardwareInfo().get_max_active_clusters(cluster_size))
        _MAX_ACTIVE_CLUSTERS_CACHE[key] = cached
    return cached


@deprecated_aliases(apply_topk_at_fc1="apply_routing_weights_before_fc2")
@dataclasses.dataclass(frozen=True)
class BlockScaledMoeConfig:
    """Validated construction parameters for block-scaled inference and GenPhase."""

    num_total_experts: int
    max_tokens_per_rank: int
    num_topk: int
    hidden: int
    intermediate: int  # width after activation; the FC1 GEMM N is 2*intermediate
    rank: int
    world_size: int
    quant_kind: BlockScaledQuantKind = "mxfp8_e4m3"
    mma_tiler_mnk: Optional[Tuple[int, int, int]] = None  # None -> (256, 128, 2*ik)
    cluster_shape_mn: Tuple[int, int] = (2, 1)
    # Mixed-CGA launch: alongside the preferred clusters, fill leftover SMs with
    # smaller fallback clusters (upstream commit a5b4d33). None = uniform launch.
    fallback_cluster_shape_mn: Optional[Tuple[int, int]] = None
    # ("grouped" | "phase_interleave", optional hint); typed loosely so backend
    # configs can carry it as a plain (str, int|None) tuple.
    schedule_policy: Tuple[str, Optional[int]] = ("grouped", None)
    work_id_mode: BlockScaledWorkIdMode = "grid_stride"
    fc2_use_bulk: bool = False
    fc2_tma_stages: Optional[int] = None
    epi_flag_batches: Tuple[int, int] = (4, 2)
    token_in_flag_batch: int = 1
    token_padding_block: int = 64
    sf_padding_block: int = 128
    gate_up_clamp: Optional[float] = None
    reduce_topk_in_kernel: bool = False
    token_back_mode: BlockScaledTokenBackMode = "epi_warps"
    apply_routing_weights_before_fc2: bool = True
    max_sm_count: Optional[int] = None
    activation: Literal["swiglu", "situ"] = "swiglu"
    situ_beta: Optional[float] = None
    situ_linear_beta: Optional[float] = None

    kernel_variant: Literal["inference", "genphase"] = "inference"
    combine_dtype: Literal["bf16", "nvfp4", "mxfp8"] = "bf16"

    def __post_init__(self) -> None:
        if self.quant_kind not in _KIND_TABLE:
            raise ValueError(f"unsupported quant_kind {self.quant_kind!r}.")
        if self.kernel_variant not in ("inference", "genphase"):
            raise ValueError(f"unsupported kernel_variant {self.kernel_variant!r}.")
        if self.combine_dtype not in ("bf16", "nvfp4", "mxfp8"):
            raise ValueError(f"unsupported combine_dtype {self.combine_dtype!r}.")
        if self.kernel_variant == "genphase" and self.combine_dtype != "bf16":
            raise ValueError("GenPhase requires BF16 combine.")
        if self.combine_dtype != "bf16" and self.reduce_topk_in_kernel:
            raise ValueError("Quantized combine requires separate top-k reduction.")
        if (
            self.combine_dtype == "nvfp4"
            and self.fc2_use_bulk
            and self.token_back_mode == "epi_warps"
        ):
            raise ValueError(
                "NVFP4 combine bulk stores require a token-back data path."
            )
        if self.activation not in ("swiglu", "situ"):
            raise ValueError("activation must be 'swiglu' or 'situ'.")
        if self.activation == "situ":
            for name in ("situ_beta", "situ_linear_beta"):
                value = getattr(self, name)
                if value is None or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"SiTU requires positive, finite {name}.")
            if self.gate_up_clamp is not None:
                raise ValueError("SiTU does not support gate_up_clamp.")
        elif self.situ_beta is not None or self.situ_linear_beta is not None:
            raise ValueError("SiTU beta parameters require activation='situ'.")
        for name in (
            "num_total_experts",
            "max_tokens_per_rank",
            "num_topk",
            "hidden",
            "intermediate",
            "world_size",
            "token_padding_block",
            "sf_padding_block",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive Python integer.")
        if type(self.rank) is not int or not (0 <= self.rank < self.world_size):
            raise ValueError(f"invalid rank/world_size {self.rank}/{self.world_size}.")
        for name in (
            "fc2_use_bulk",
            "reduce_topk_in_kernel",
            "apply_routing_weights_before_fc2",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a Python bool.")
        if self.num_total_experts > 16384:
            raise ValueError(
                "block-scaled routing supports at most 16384 global experts."
            )
        if self.num_topk > self.num_total_experts:
            raise ValueError("num_topk must not exceed num_total_experts.")
        if self.padded_tokens_per_rank * self.num_topk > 2097152:
            raise ValueError(
                "block-scaled routing supports at most 2097152 routes per rank."
            )
        if self.num_total_experts % self.world_size != 0:
            raise ValueError(
                f"num_total_experts ({self.num_total_experts}) must divide evenly "
                f"across world_size ({self.world_size})."
            )
        vec = self.sf_vec_size
        if self.hidden % (4 * vec) != 0:
            raise ValueError(
                f"hidden ({self.hidden}) must be a multiple of {4 * vec} "
                f"for {self.quant_kind}."
            )
        if self.intermediate % (2 * vec) != 0 or self.intermediate % 32 != 0:
            raise ValueError(
                f"intermediate ({self.intermediate}) must be a multiple of "
                f"{max(2 * vec, 32)} (gate/up interleave + SF blocks) for "
                f"{self.quant_kind}."
            )
        if self.quant_kind == "mxfp4_mxfp8" and self.intermediate % 128:
            raise ValueError("mxfp4_mxfp8 intermediate must be a multiple of 128.")
        tiler = self.resolved_mma_tiler_mnk
        instruction_k = self.instruction_k
        if len(tiler) != 3:
            raise ValueError("mma_tiler_mnk must be a 3-tuple.")
        if any(type(dim) is not int or dim <= 0 for dim in tiler):
            raise ValueError(
                "mma_tiler_mnk dimensions must be positive Python integers."
            )
        if tiler[0] not in (128, 256):
            raise ValueError(
                f"block-scaled instruction M must be 128 or 256, got {tiler[0]}."
            )
        if tiler[1] not in (64, 128, 256):
            raise ValueError(
                f"block-scaled instruction N must be 64, 128, or 256, got {tiler[1]}."
            )
        if tiler[2] not in (2 * instruction_k, 4 * instruction_k):
            raise ValueError(
                f"block-scaled {self.quant_kind} tile K must be {2 * instruction_k} or "
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
        cm, cn = self.cluster_shape_mn
        if type(cm) is not int or cm <= 0 or cm > 16 or cm & (cm - 1):
            raise ValueError("cluster M must be a power of two no greater than 16.")
        if type(cn) is not int or cn != 1:
            raise ValueError("the swap-AB FC12 path requires cluster N=1.")
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
                if any(type(dim) is not int or dim <= 0 for dim in fallback):
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
        if self.sf_padding_block % 128 != 0:
            raise ValueError("sf_padding_block must be a multiple of 128.")
        if tiler[1] % self.token_padding_block != 0:
            raise ValueError(
                f"mma tiler N ({tiler[1]}) must be a whole number of token "
                f"padding blocks ({self.token_padding_block})."
            )
        if len(self.schedule_policy) != 2:
            raise ValueError("schedule_policy requires (mode, hint).")
        mode, hint = self.schedule_policy
        if mode not in ("grouped", "phase_interleave"):
            raise ValueError(f"unknown schedule mode {mode!r}.")
        if hint is not None and (type(hint) is not int or hint <= 0):
            raise ValueError("schedule hint must be a positive Python integer or None.")
        if self.work_id_mode not in ("grid_stride", "atomic_counter"):
            raise ValueError(f"unknown work_id_mode {self.work_id_mode!r}.")
        if self.token_back_mode not in (
            "epi_warps",
            "standalone_warps",
            "reuse_dispatch_warps",
        ):
            raise ValueError(f"unknown token_back_mode {self.token_back_mode!r}.")
        if mode == "phase_interleave" and self.work_id_mode != "atomic_counter":
            raise ValueError(
                "schedule_policy 'phase_interleave' requires "
                "work_id_mode='atomic_counter'."
            )
        if len(self.epi_flag_batches) != 2:
            raise ValueError("epi_flag_batches requires (FC1, FC2) values.")
        if any(
            type(batch) is not int or not 1 <= batch <= 4
            for batch in self.epi_flag_batches
        ):
            raise ValueError("block-scaled epi_flag_batches values must be in [1, 4].")
        if (
            type(self.token_in_flag_batch) is not int
            or not 1 <= self.token_in_flag_batch <= 32
        ):
            raise ValueError("token_in_flag_batch must be in [1, 32].")
        if self.max_sm_count is not None and (
            type(self.max_sm_count) is not int or self.max_sm_count < cm
        ):
            raise ValueError("max_sm_count must accommodate at least one cluster.")
        if self.gate_up_clamp is not None and (
            not math.isfinite(self.gate_up_clamp) or self.gate_up_clamp < 0
        ):
            raise ValueError("gate_up_clamp must be finite and nonnegative.")
        if self.fc2_tma_stages is not None and not (
            type(self.fc2_tma_stages) is int
            and 1 <= self.fc2_tma_stages <= tiler[1] // 64
        ):
            raise ValueError(
                f"fc2_tma_stages must be in [1, {tiler[1] // 64}] for tiler N "
                f"{tiler[1]}."
            )
        if self.reduce_topk_in_kernel and not self.apply_routing_weights_before_fc2:
            raise ValueError(
                "reduce_topk_in_kernel requires "
                "apply_routing_weights_before_fc2=True (the in-kernel reduce "
                "red-adds already-weighted terms)."
            )

        if self.kernel_variant == "genphase":
            if self.padded_tokens_per_rank > 1024:
                raise ValueError("GenPhase supports at most 1024 tokens per rank.")
            n64_decode = (
                self.cluster_shape_mn in ((2, 1), (4, 1))
                and tiler == (256, 64, 128)
                and (
                    self.quant_kind == "mxfp8_e4m3"
                    or (
                        self.quant_kind == "mxfp4_mxfp8"
                        and self.cluster_shape_mn == (4, 1)
                    )
                )
            )
            if (
                self.cluster_shape_mn != (4, 1) and not n64_decode
            ) or self.fallback_cluster_shape_mn is not None:
                raise ValueError(
                    "GenPhase requires uniform cluster (4, 1), or an admitted N64 decode geometry at (2, 1) or (4, 1)."
                )
            if not self.fc2_use_bulk:
                raise ValueError("GenPhase requires fc2_use_bulk=True.")
            if tiler[1] not in (128, 256) and not n64_decode:
                raise ValueError(
                    "GenPhase requires tile N 128 or 256, or an admitted N64 decode geometry."
                )
            if n64_decode and self.world_size > 1 and mode == "phase_interleave":
                # EP1 phase-interleave coverage does not establish multirank
                # liveness: an EP2 dense/replay case stalls with this geometry.
                # Reject before allocation or compilation; autotune applies
                # this same config validation before compilation.
                raise ValueError(
                    "GenPhase N64 with world_size > 1 requires grouped scheduling; "
                    "multirank phase_interleave is not qualified."
                )
            gather_tile_k = 2 * _KIND_TABLE[self.quant_kind][3]
            if tiler[2] != gather_tile_k:
                raise ValueError(
                    f"GenPhase {self.quant_kind} requires tile K {gather_tile_k} "
                    "so each gathered row occupies one 128-byte swizzle atom."
                )
            if self.sf_padding_block != 128 or 128 % self.token_padding_block:
                raise ValueError(
                    "GenPhase requires SF padding 128 and token padding 64 or 128."
                )
            if self.token_back_mode != "epi_warps" or self.token_in_flag_batch != 1:
                raise ValueError(
                    "GenPhase does not use token-back or token-in flag knobs."
                )

    @property
    def experts_per_rank(self) -> int:
        return self.num_total_experts // self.world_size

    @property
    def padded_tokens_per_rank(self) -> int:
        """Physical capacity keeps the router's four-int vector loads in bounds."""
        multiple = 4 // math.gcd(self.num_topk, 4)
        return (self.max_tokens_per_rank + multiple - 1) // multiple * multiple

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
    def torch_weight_data_dtype(self) -> torch.dtype:
        if self.quant_kind in ("nvfp4", "mxfp4_mxfp8"):
            return _fp4_storage_dtype()
        return self.torch_act_data_dtype

    @property
    def torch_act_sf_dtype(self) -> torch.dtype:
        return _KIND_TABLE[self.quant_kind][1]


class BlockScaledSymmBuffer:
    """Shared session workspace for the block-scaled inference mega kernel.

    Owns the staging tensors the backend fills (``x``, ``x_sf``, ``topk_idx``,
    ``topk_weights``), the kernel + device workspaces, and the compile cache.
    Construct through an architecture-specific subclass.
    """

    _architecture: str

    @staticmethod
    def _require_device() -> None:
        raise NotImplementedError("An architecture adapter must validate the device.")

    @staticmethod
    def _configure_dsl() -> None:
        raise NotImplementedError(
            "An architecture adapter must select the compiler target."
        )

    def __init__(self, config: BlockScaledMoeConfig) -> None:
        """Allocate staging, output and kernel storage collectively before capture."""
        comm.ensure_not_capturing("block-scaled mega workspace allocation")
        self._require_device()
        self._configure_dsl()
        self.config = config
        self.device = torch.device("cuda", torch.cuda.current_device())
        cfg = config

        kernel = self._build_kernel(cfg)
        self.kernel = kernel

        tokens = cfg.padded_tokens_per_rank

        # Staging tensors. Activation / SF / scores are pulled by peer dispatch
        # warps, so they live on the symmetric heap; the routing indices are a
        # local (16B-aligned) int32 tensor read by this rank's router only.
        if cfg.quant_kind == "nvfp4":
            self.x = comm.sym_zeros((tokens, cfg.hidden // 2), cfg.torch_act_data_dtype)
        else:
            self.x = comm.sym_zeros((tokens, cfg.hidden), cfg.torch_act_data_dtype)
        self.sf_cols = int(
            kernel.token_comm.plain_activation_sf_row_elements
            if cfg.kernel_variant == "genphase"
            else kernel.token_comm.activation_sf_hidden_padded
        )
        self.x_sf = comm.sym_zeros((tokens, self.sf_cols), cfg.torch_act_sf_dtype)
        self.topk_weights = comm.sym_zeros((tokens, cfg.num_topk), torch.float32)
        self.topk_idx = torch.full(
            (tokens, cfg.num_topk), -1, dtype=torch.int32, device="cuda"
        )
        # Stable local-expert pointers allow per-call scaling updates in graphs.
        # Defaults and overrides are copied on every backend staging call.
        for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const"):
            setattr(
                self,
                name,
                torch.ones(
                    cfg.experts_per_rank, dtype=torch.float32, device=self.device
                )
                if cfg.quant_kind == "nvfp4"
                else None,
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

        self._allocate_kernel_workspace(kernel)

        self._compiled: Optional[Any] = None
        self._launch_key: Optional[tuple] = None
        self._launch_kwargs: Optional[dict] = None
        self._launch_weights: Optional[
            Tuple[TransformedBlockScaledWeights, TransformedBlockScaledWeights]
        ] = None
        self._staged_tokens: Optional[int] = None
        self._destroyed = False

    def _allocate_kernel_workspace(self, kernel) -> None:
        """Allocate kernel-owned storage; staging and output pointers stay stable."""
        cfg = self.config
        tokens = cfg.padded_tokens_per_rank
        workspace = kernel._device_workspace
        local_bytes = int(workspace.total_bytes("local"))
        shared_bytes = int(workspace.total_bytes("shared"))
        # Allocated fully zeroed, which covers require_zero_workspace_leading_bytes.
        self.local_workspace = torch.zeros(
            (max(local_bytes, 1),), dtype=torch.uint8, device="cuda"
        )
        self.shared_workspace = comm.sym_zeros((max(shared_bytes, 1),), torch.uint8)
        self._pre_reduced_activation = None
        self._pre_reduced_activation_sf = None
        if not cfg.reduce_topk_in_kernel:
            region = (
                kernel.pre_reduced_activation_region
                if cfg.kernel_variant == "genphase"
                else kernel.token_comm.pre_reduced_activation_region
            )
            start = workspace.offset(region)
            end = start + workspace.nbytes(region)
            # Byte views also cover packed FP4/FP8 payloads, which do not
            # support Torch masked_fill directly.
            self._pre_reduced_activation = self.shared_workspace[start:end].view(
                tokens, cfg.num_topk, -1
            )
            if cfg.combine_dtype != "bf16":
                region = kernel.token_comm.pre_reduced_activation_sf_region
                start = workspace.offset(region)
                end = start + workspace.nbytes(region)
                self._pre_reduced_activation_sf = self.shared_workspace[start:end].view(
                    tokens, cfg.num_topk, -1
                )
        if (
            self.local_workspace.data_ptr() % 128 != 0
            or self.shared_workspace.data_ptr() % 128 != 0
        ):
            raise RuntimeError("block-scaled mega workspaces must be 128-byte aligned.")

        base, peer_offsets = comm.compute_peer_offsets(
            self.shared_workspace, cfg.world_size
        )
        self._symmetric_base = base
        self._peer_offsets = peer_offsets

    @classmethod
    def _build_kernel(cls, cfg: BlockScaledMoeConfig):
        import cutlass
        from blackwell_genphase_sources import BlackwellInferenceGenphaseMegaMoE
        from blackwell_genphase_sources.api import ImplDesc, ProblemDesc
        from blackwell_genphase_sources.quant_def import CombineFormat
        from cutlass.cute.nvgpu import OperandMajorMode

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
                "combine_format": CombineFormat.parse(
                    {
                        "bf16": "bf16",
                        "nvfp4": "16e2m1xbf16",
                        "mxfp8": "32e4m3xe8m0",
                    }[cfg.combine_dtype]
                ),
                "gate_up_clamp": cfg.gate_up_clamp,
                "situ_beta": cfg.situ_beta,
                "situ_linear_beta": cfg.situ_linear_beta,
                "world_size": cfg.world_size,
                "topk": cfg.num_topk,
                "topk_index_dtype": cutlass.Int32,
                "max_tokens_per_rank": cfg.padded_tokens_per_rank,
                "apply_topk_at_fc1": cfg.apply_routing_weights_before_fc2,
            }
        )
        impl_fields = {
            "architecture": cls._architecture,
            "mma_instruction_mnk": (tiler[0], tiler[1], cfg.instruction_k),
            "mma_tiler_mnk": tiler,
            "mma_k_mode": "1x",
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
        return BlackwellInferenceGenphaseMegaMoE(problem_desc, ImplDesc(impl_fields))

    def note_staged_tokens(self, num_tokens: int) -> None:
        self._staged_tokens = int(num_tokens)

    def staged_tokens(self) -> Optional[int]:
        return self._staged_tokens

    def _peer_mapper(self):
        from blackwell_genphase_sources.communication.nvlink_domain.symmetric_buffer import (
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
            **{
                name: _to_cute(getattr(self, name), assumed_align=4)
                for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const")
                if self.config.quant_kind == "nvfp4"
            },
        }

    def launch(
        self,
        transformed_l1: TransformedBlockScaledWeights,
        transformed_l2: TransformedBlockScaledWeights,
    ) -> None:
        """Compile on first use, then launch the fused mega kernel."""
        import cutlass.cute as cute

        if self._destroyed:
            raise RuntimeError("cannot launch a destroyed block-scaled workspace.")
        if (
            self.device.type == "cuda"
            and self.device.index != torch.cuda.current_device()
        ):
            raise RuntimeError(
                "block-scaled workspace must launch on the CUDA device that owns it."
            )
        key = (
            transformed_l1[0].data_ptr(),
            transformed_l1[1].data_ptr(),
            transformed_l2[0].data_ptr(),
            transformed_l2[1].data_ptr(),
            torch.cuda.current_stream().cuda_stream,
        )
        if self._compiled is None or self._launch_key != key:
            if self._compiled is None:
                comm.ensure_not_capturing("block-scaled mega kernel compile")
            # DLPack metadata binding and stream conversion are host-only.
            # A warmed workspace may bind another layer or stream in capture.
            kwargs = self._runtime_kwargs(transformed_l1, transformed_l2)
            if self._compiled is None:
                self._compiled = cute.compile(self.kernel, **kwargs)
            self._launch_key = key
            self._launch_kwargs = kwargs
            # Keep pointers alive until the cached bindings are replaced.
            self._launch_weights = (transformed_l1, transformed_l2)
        if self.config.reduce_topk_in_kernel:
            # red.add accumulation base: zero the output every launch.
            self.output_activation.zero_()
        elif not getattr(self.kernel, "masks_invalid_combine_slots", False):
            # Invalid routes do not write a combine slot. Clear those slots on
            # every launch so a masked route cannot reuse an earlier result.
            invalid = (self.topk_idx < 0) | (
                self.topk_idx >= self.config.num_total_experts
            )
            self._pre_reduced_activation.masked_fill_(invalid.unsqueeze(-1), 0)
            if self.config.combine_dtype != "bf16":
                # Clear BF16 amax for NVFP4; E8M0 byte 127 is unit scale for MXFP8.
                sf_byte = 0 if self.config.combine_dtype == "nvfp4" else 127
                self._pre_reduced_activation_sf.masked_fill_(
                    invalid.unsqueeze(-1), sf_byte
                )
        self._compiled(**self._launch_kwargs)

    def destroy(self) -> None:
        if self._destroyed:
            return
        comm.ensure_not_capturing("block-scaled mega workspace free")
        # nvshmem free is collective and does not wait for in-flight work;
        # drain this rank's device before releasing symmetric memory that a
        # still-running launch (ours or a peer's dispatch pull) may touch.
        torch.cuda.synchronize()
        self._compiled = None
        self._launch_kwargs = None
        self._launch_weights = None
        self._pre_reduced_activation = None
        self._pre_reduced_activation_sf = None
        for name in ("x", "x_sf", "topk_weights", "shared_workspace"):
            comm.free_sym_tensor(getattr(self, name, None))
            setattr(self, name, None)
        if self.config.reduce_topk_in_kernel:
            comm.free_sym_tensor(self.output_activation)
        self.output_activation = None
        self.local_workspace = None
        self.topk_idx = None
        self.fc1_alpha = self.fc2_alpha = self.fc1_norm_const = None
        self._destroyed = True


def _expected_weight_shapes(
    cfg: BlockScaledMoeConfig,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], int, int]:
    """(fc1 shape, fc2 shape, fc1 SF numel/expert, fc2 SF numel/expert)."""
    experts = cfg.experts_per_rank
    fc1_out = 2 * cfg.intermediate
    vec = cfg.sf_vec_size
    if cfg.quant_kind in ("nvfp4", "mxfp4_mxfp8"):
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
    cfg: BlockScaledMoeConfig,
    device: torch.device,
) -> None:
    if (
        not isinstance(leg, tuple)
        or len(leg) != 2
        or not all(isinstance(t, torch.Tensor) for t in leg)
    ):
        raise ValueError(f"{name} must be a (weight, scale) tensor pair.")
    weight, scale = leg
    if tuple(weight.shape) != expected_weight_shape:
        raise ValueError(
            f"{name} weight shape {tuple(weight.shape)} != expected "
            f"{expected_weight_shape}."
        )
    if tuple(scale.shape) != (expected_weight_shape[0], expected_sf_numel):
        raise ValueError(
            f"{name} scale numel {scale.numel()} != expected "
            f"{expected_sf_numel * expected_weight_shape[0]}."
        )
    if (
        weight.dtype != cfg.torch_weight_data_dtype
        or scale.dtype != cfg.torch_act_sf_dtype
    ):
        raise ValueError(
            f"{name} requires {cfg.torch_weight_data_dtype} weights and {cfg.torch_act_sf_dtype} scales."
        )
    if not weight.permute(0, 2, 1).is_contiguous() or not scale.is_contiguous():
        raise ValueError(
            f"{name} requires K-major weights and contiguous scale planes."
        )
    if any(t.device != device or t.data_ptr() % 16 for t in leg):
        raise ValueError(f"{name} must be on {device} with 16-byte aligned storage.")


def block_scaled_mega_moe(
    y: Optional[torch.Tensor],
    transformed_l1: TransformedBlockScaledWeights,
    transformed_l2: TransformedBlockScaledWeights,
    symm_buffer: BlockScaledSymmBuffer,
    *,
    num_tokens: Optional[int] = None,
    fast_math: bool = True,  # accepted for mega API parity; the kernel has no toggle
    sync: bool = False,
) -> Optional[torch.Tensor]:
    """Fused dispatch, FC1, SwiGLU/SiTU, FC2, and combine; writes ``y[:num_tokens]``.

    The caller must have staged ``symm_buffer.x`` / ``.x_sf`` and the routing
    slices first. With ``y=None`` returns a workspace view (valid under stream
    ordering until the next launch).
    """
    if not fast_math:
        import warnings

        warnings.warn(
            "fast_math=False is a no-op for the block-scaled block-scaled kernel.",
            stacklevel=2,
        )
    cfg = symm_buffer.config
    if y is not None and (
        y.ndim != 2
        or y.shape[1] != cfg.hidden
        or y.dtype != torch.bfloat16
        or y.device != symm_buffer.device
    ):
        raise ValueError(
            f"y must be a BF16 matrix with {cfg.hidden} columns on {symm_buffer.device}."
        )
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
    if y is not None and y.shape[0] < num_tokens:
        raise ValueError("y does not have enough rows for num_tokens.")

    fc1_shape, fc2_shape, fc1_sf, fc2_sf = _expected_weight_shapes(cfg)
    _validate_weight_leg(
        "fc1", transformed_l1, fc1_shape, fc1_sf, cfg, symm_buffer.device
    )
    _validate_weight_leg(
        "fc2", transformed_l2, fc2_shape, fc2_sf, cfg, symm_buffer.device
    )

    symm_buffer.launch(transformed_l1, transformed_l2)
    out = symm_buffer.output_activation[:num_tokens]
    if y is None:
        if sync:
            torch.cuda.synchronize()
        return out
    y[:num_tokens].copy_(out)
    if sync:
        torch.cuda.synchronize()
    return None


def block_scaled_mega_launch_thunk(
    transformed_l1: TransformedBlockScaledWeights,
    transformed_l2: TransformedBlockScaledWeights,
    symm_buffer: BlockScaledSymmBuffer,
):
    """Zero-arg relauncher over pre-staged inputs, for steady-state timing loops."""
    cfg = symm_buffer.config
    fc1_shape, fc2_shape, fc1_sf, fc2_sf = _expected_weight_shapes(cfg)
    _validate_weight_leg(
        "fc1", transformed_l1, fc1_shape, fc1_sf, cfg, symm_buffer.device
    )
    _validate_weight_leg(
        "fc2", transformed_l2, fc2_shape, fc2_sf, cfg, symm_buffer.device
    )

    def _thunk() -> None:
        symm_buffer.launch(transformed_l1, transformed_l2)

    return _thunk


__all__ = [
    "BlockScaledMoeConfig",
    "BlockScaledSymmBuffer",
    "BlockScaledQuantKind",
    "TransformedBlockScaledWeights",
    "block_scaled_mega_launch_thunk",
    "block_scaled_mega_moe",
]
