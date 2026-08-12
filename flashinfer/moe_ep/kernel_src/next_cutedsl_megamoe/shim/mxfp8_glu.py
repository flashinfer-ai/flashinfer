# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""SM107 (Rubin) mxfp8 GLU fprop mega-MoE frontend.

Wraps the vendored ``sources.kernel_src.rubin.training.mega.fwd_glu`` kernel
(``Sm107MegaMoEMxfp8GluKernel``) behind the standard two-entry-point mega
contract:

- :func:`get_symm_buffer_for_sm107_mxfp8_glu_mega_moe` — workspace allocator
- :func:`sm107_mxfp8_glu_mega_moe` — fused dispatch + FC1 + SwiGLU + FC2 +
  combine compute entry

All ``sources`` / ``cutlass`` imports are function-local so importing this
module stays CPU-safe (the package ``__init__`` re-exports from here).

The staging/launch protocol mirrors the drop's own runner
(``next/repo_internal_only/test_megamoe_training_fwd_glu_mxfp8_rubin.py``):
activation, activation SF, topk scores, the shared workspace, and (for the
in-kernel-reduce path) the output live on the symmetric heap; routing indices
are local int32 (16-byte aligned); workspaces are 128-byte aligned with their
leading bytes zeroed before the first launch.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Literal, Optional, Tuple

import torch

from . import comm
from .kernel_helpers import Mxfp8BlockSize

Sm107Mxfp8GluKind = Literal["mxfp8_e4m3", "mxfp8_e5m2"]

_KIND_TO_TORCH_DTYPE = {
    "mxfp8_e4m3": torch.float8_e4m3fn,
    "mxfp8_e5m2": torch.float8_e5m2,
}

TransformedMxfp8GluWeights = Tuple[torch.Tensor, torch.Tensor]


def _require_sm107() -> None:
    cc = torch.cuda.get_device_capability()
    if cc != (10, 7):
        raise RuntimeError(
            f"the SM107 mxfp8 GLU mega kernel requires compute capability (10, 7) "
            f"(Rubin); this device reports {cc}."
        )


def _configure_dsl_arch() -> None:
    configured = os.environ.get("CUTE_DSL_ARCH")
    if configured not in (None, "sm_107", "sm_107a"):
        raise RuntimeError(
            f"CUTE_DSL_ARCH must target SM107 for the Rubin mega kernel, "
            f"got {configured!r}."
        )
    os.environ["CUTE_DSL_ARCH"] = "sm_107a"


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


def _to_cute_ptr(tensor: torch.Tensor, assumed_align: int = 16):
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
class Sm107MegaMoEMxfp8GluConfig:
    """Validated construction parameters for ``Sm107MegaMoEMxfp8GluKernel``."""

    num_total_experts: int
    max_tokens_per_rank: int
    num_topk: int
    hidden: int
    intermediate: int  # post-SwiGLU width; the FC1 GEMM N is 2*intermediate
    rank: int
    world_size: int
    kind: Sm107Mxfp8GluKind = "mxfp8_e4m3"
    mma_tiler_mnk: Tuple[int, int, int] = (256, 256, 128)
    cluster_shape_mnk: Tuple[int, int, int] = (2, 1, 1)
    group_hint: Optional[int] = (
        768  # tuned fc1->fc2 scheduler lead; <=0/None -> HW clusters
    )
    token_padding_block: int = 128
    sf_padding_block: int = 128
    gate_up_clamp: Optional[float] = None
    in_kernel_fc2_reduce: bool = False
    token_back_mode: Literal[
        "epi_warps", "standalone_warps", "reuse_dispatch_warps"
    ] = "epi_warps"
    epi_flag_batch: Tuple[int, int] = (4, 2)
    flag_batch: int = 1
    apply_topk_in_fc1: bool = True
    max_sm_count: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind not in _KIND_TO_TORCH_DTYPE:
            raise ValueError(f"unsupported kind {self.kind!r}.")
        if self.world_size < 1 or not (0 <= self.rank < self.world_size):
            raise ValueError(f"invalid rank/world_size {self.rank}/{self.world_size}.")
        if self.num_total_experts % self.world_size != 0:
            raise ValueError(
                f"num_total_experts ({self.num_total_experts}) must divide evenly "
                f"across world_size ({self.world_size})."
            )
        if self.hidden % Mxfp8BlockSize != 0:
            raise ValueError(f"hidden ({self.hidden}) must be a multiple of 32.")
        if self.intermediate % 32 != 0:
            raise ValueError(
                f"intermediate ({self.intermediate}) must be a multiple of the "
                "gate/up interleave (32)."
            )
        if len(self.mma_tiler_mnk) != 3 or len(self.cluster_shape_mnk) != 3:
            raise ValueError("mma_tiler_mnk / cluster_shape_mnk must be 3-tuples.")
        if self.cluster_shape_mnk[2] != 1:
            raise ValueError("cluster_shape_mnk K must be one.")
        if len(self.epi_flag_batch) != 2:
            raise ValueError("epi_flag_batch requires (FC1, FC2) values.")
        if self.in_kernel_fc2_reduce and not self.apply_topk_in_fc1:
            raise ValueError(
                "in_kernel_fc2_reduce requires apply_topk_in_fc1=True (the epilogue "
                "red-adds already-weighted terms)."
            )
        if self.in_kernel_fc2_reduce and self.token_back_mode != "epi_warps":
            raise ValueError(
                "in_kernel_fc2_reduce requires token_back_mode='epi_warps'."
            )

    @property
    def experts_per_rank(self) -> int:
        return self.num_total_experts // self.world_size

    @property
    def torch_data_dtype(self) -> torch.dtype:
        return _KIND_TO_TORCH_DTYPE[self.kind]


class Sm107Mxfp8GluSymmBuffer:
    """Session workspace for the SM107 mxfp8 GLU mega kernel.

    Owns the staging tensors the backend fills (``x``, ``x_sf``, ``topk_idx``,
    ``topk_weights``), the kernel + device workspaces, and the compile cache.
    Create via :func:`get_symm_buffer_for_sm107_mxfp8_glu_mega_moe`.
    """

    def __init__(self, config: Sm107MegaMoEMxfp8GluConfig) -> None:
        comm.ensure_not_capturing("SM107 mega workspace allocation")
        _require_sm107()
        _configure_dsl_arch()
        self.config = config
        cfg = config

        kernel = self._build_kernel(cfg)
        self.kernel = kernel

        tokens = cfg.max_tokens_per_rank
        data_dtype = cfg.torch_data_dtype

        # Staging tensors. Activation / SF / scores are pulled by peer dispatch
        # warps, so they live on the symmetric heap; the routing indices are a
        # local (16B-aligned) int32 tensor read by this rank's router only.
        self.x = comm.sym_zeros((tokens, cfg.hidden), data_dtype)
        self.sf_cols = int(kernel.token_comm.activation_sf_hidden_padded)
        self.x_sf = comm.sym_zeros((tokens, self.sf_cols), torch.float8_e8m0fnu)
        self.topk_weights = comm.sym_zeros((tokens, cfg.num_topk), torch.float32)
        self.topk_idx = torch.full(
            (tokens, cfg.num_topk), -1, dtype=torch.int32, device="cuda"
        )
        if self.topk_idx.data_ptr() % 16 != 0:
            raise RuntimeError("routing index tensor must be 16-byte aligned.")

        # In-kernel reduce: the epilogue peer-writes (red.add) each topk term
        # straight into the output, so it must live on the SYMMETRIC heap and
        # start zeroed; the separate-reduce path only sees local stores.
        if cfg.in_kernel_fc2_reduce:
            self.output_activation = comm.sym_zeros(
                (tokens, cfg.hidden), torch.bfloat16
            )
        else:
            self.output_activation = torch.zeros(
                (tokens, cfg.hidden), dtype=torch.bfloat16, device="cuda"
            )
        self.overflow_flag = torch.zeros((1,), dtype=torch.int32, device="cuda")

        workspace = kernel._mega_device_workspace
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
    def _build_kernel(cfg: Sm107MegaMoEMxfp8GluConfig):
        import cutlass

        from sources.kernel_src.rubin.training.mega import Sm107MegaMoEMxfp8GluKernel
        from sources.quant_def import CombineFormat

        ab_dtype = (
            cutlass.Float8E4M3FN if cfg.kind == "mxfp8_e4m3" else cutlass.Float8E5M2
        )
        cluster_size = cfg.cluster_shape_mnk[0] * cfg.cluster_shape_mnk[1]
        hardware_clusters = _max_active_clusters(cluster_size)
        if cfg.max_sm_count is not None:
            requested = cfg.max_sm_count // cluster_size
            if requested <= 0:
                raise ValueError("max_sm_count must cover at least one full cluster.")
            hardware_clusters = min(hardware_clusters, requested)
        group_hint = (
            hardware_clusters
            if (cfg.group_hint is None or cfg.group_hint <= 0)
            else cfg.group_hint
        )
        return Sm107MegaMoEMxfp8GluKernel.from_kwargs(
            mma_tiler_mnk=cfg.mma_tiler_mnk,
            cluster_shape_mnk=cfg.cluster_shape_mnk,
            use_2cta_instrs=cfg.mma_tiler_mnk[0] == 256,
            group_hint=group_hint,
            token_padding_block=cfg.token_padding_block,
            sf_padding_block=cfg.sf_padding_block,
            load_balance_mode="static",
            static_expert_shape=(
                cfg.experts_per_rank,
                2 * cfg.intermediate,
                cfg.hidden,
            ),
            force_static_sched=True,
            ab_dtype=ab_dtype,
            sf_vec_size=Mxfp8BlockSize,
            world_size=cfg.world_size,
            local_rank=cfg.rank,
            num_topk=cfg.num_topk,
            max_tokens_per_rank=cfg.max_tokens_per_rank,
            hidden=cfg.hidden,
            launch_cluster_count=hardware_clusters,
            fc2_in_kernel_topk_reduce=cfg.in_kernel_fc2_reduce,
            token_back_mode=cfg.token_back_mode,
            epi_flag_batch=cfg.epi_flag_batch,
            flag_batch=cfg.flag_batch,
            gate_up_clamp=cfg.gate_up_clamp,
            apply_topk_in_fc1=cfg.apply_topk_in_fc1,
            generate_c=False,
            combine_format=CombineFormat.parse("bf16"),
            act_func="swiglu",
        )

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
        transformed_l1: TransformedMxfp8GluWeights,
        transformed_l2: TransformedMxfp8GluWeights,
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
            "overflow_flag": _to_cute(self.overflow_flag, assumed_align=4),
            "local_workspace": _to_cute_ptr(self.local_workspace),
            "shared_workspace": _to_cute_ptr(self.shared_workspace),
            "peer_rank_ptr_mapper_host": self._peer_mapper(),
            "stream": _cu_stream(stream),
            "fc1_c": None,
        }

    def launch(
        self,
        transformed_l1: TransformedMxfp8GluWeights,
        transformed_l2: TransformedMxfp8GluWeights,
    ) -> None:
        """Compile on first use, then launch router + mega (+ topk reduce)."""
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
        if self.config.in_kernel_fc2_reduce:
            # red.add accumulation base: zero the output every launch.
            self.output_activation.zero_()
        self._compiled(**self._launch_kwargs)

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        comm.ensure_not_capturing("SM107 mega workspace free")
        self._compiled = None
        self._launch_kwargs = None
        for name in ("x", "x_sf", "topk_weights", "shared_workspace"):
            comm.free_sym_tensor(getattr(self, name, None))
        if self.config.in_kernel_fc2_reduce:
            comm.free_sym_tensor(self.output_activation)


def get_symm_buffer_for_sm107_mxfp8_glu_mega_moe(
    num_total_experts: int,
    num_max_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate: int,
    rank: int,
    world_size: int,
    *,
    kind: Sm107Mxfp8GluKind = "mxfp8_e4m3",
    mma_tiler_mnk: Tuple[int, int, int] = (256, 256, 128),
    cluster_shape_mnk: Tuple[int, int, int] = (2, 1, 1),
    group_hint: Optional[int] = 768,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    in_kernel_fc2_reduce: bool = False,
    token_back_mode: Literal[
        "epi_warps", "standalone_warps", "reuse_dispatch_warps"
    ] = "epi_warps",
    epi_flag_batch: Tuple[int, int] = (4, 2),
    flag_batch: int = 1,
    apply_topk_in_fc1: bool = True,
    max_sm_count: Optional[int] = None,
) -> Sm107Mxfp8GluSymmBuffer:
    """Allocate the SM107 mxfp8 GLU mega session workspace.

    Problem sizes positional, tuning knobs keyword-only (the standard mega
    allocator contract). ``intermediate`` is the post-SwiGLU width. Expert
    weights are NOT owned by the workspace; they are passed per launch.
    """
    if activation_clamp is not None:
        import warnings

        warnings.warn(
            "activation_clamp is deprecated; use gate_up_clamp.",
            DeprecationWarning,
            stacklevel=2,
        )
        if gate_up_clamp is not None and gate_up_clamp != activation_clamp:
            raise ValueError(
                "gate_up_clamp and activation_clamp disagree; pass only one."
            )
        gate_up_clamp = activation_clamp
    config = Sm107MegaMoEMxfp8GluConfig(
        num_total_experts=num_total_experts,
        max_tokens_per_rank=num_max_tokens,
        num_topk=num_topk,
        hidden=hidden,
        intermediate=intermediate,
        rank=rank,
        world_size=world_size,
        kind=kind,
        mma_tiler_mnk=mma_tiler_mnk,
        cluster_shape_mnk=cluster_shape_mnk,
        group_hint=group_hint,
        gate_up_clamp=gate_up_clamp,
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        token_back_mode=token_back_mode,
        epi_flag_batch=epi_flag_batch,
        flag_batch=flag_batch,
        apply_topk_in_fc1=apply_topk_in_fc1,
        max_sm_count=max_sm_count,
    )
    return Sm107Mxfp8GluSymmBuffer(config)


def _validate_weight_leg(
    name: str,
    leg: TransformedMxfp8GluWeights,
    expected_weight_shape: Tuple[int, ...],
    expected_sf_numel: int,
    data_dtype: torch.dtype,
) -> None:
    weight, scale = leg
    if tuple(weight.shape) != expected_weight_shape:
        raise ValueError(
            f"{name} weight shape {tuple(weight.shape)} != expected "
            f"{expected_weight_shape}."
        )
    if weight.dtype != data_dtype:
        raise ValueError(f"{name} weight dtype {weight.dtype} != {data_dtype}.")
    if scale.numel() != expected_sf_numel * expected_weight_shape[0]:
        raise ValueError(
            f"{name} scale numel {scale.numel()} != expected "
            f"{expected_sf_numel * expected_weight_shape[0]}."
        )


def sm107_mxfp8_glu_mega_moe(
    y: Optional[torch.Tensor],
    transformed_l1: TransformedMxfp8GluWeights,
    transformed_l2: TransformedMxfp8GluWeights,
    symm_buffer: Sm107Mxfp8GluSymmBuffer,
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
            "fast_math=False is a no-op for the SM107 mxfp8 GLU kernel.",
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

    experts_per_rank = cfg.experts_per_rank
    fc1_out = 2 * cfg.intermediate
    from .kernel_helpers import swizzled_flat_sf_size

    _validate_weight_leg(
        "fc1",
        transformed_l1,
        (experts_per_rank, cfg.hidden, fc1_out),
        swizzled_flat_sf_size(fc1_out, cfg.hidden // Mxfp8BlockSize),
        cfg.torch_data_dtype,
    )
    _validate_weight_leg(
        "fc2",
        transformed_l2,
        (experts_per_rank, cfg.intermediate, cfg.hidden),
        swizzled_flat_sf_size(cfg.hidden, cfg.intermediate // Mxfp8BlockSize),
        cfg.torch_data_dtype,
    )

    symm_buffer.launch(transformed_l1, transformed_l2)
    if sync:
        torch.cuda.synchronize()

    out = symm_buffer.output_activation[:num_tokens]
    if y is None:
        return out
    y[:num_tokens].copy_(out)
    return None


def sm107_mxfp8_glu_mega_launch_thunk(
    transformed_l1: TransformedMxfp8GluWeights,
    transformed_l2: TransformedMxfp8GluWeights,
    symm_buffer: Sm107Mxfp8GluSymmBuffer,
):
    """Zero-arg relauncher over pre-staged inputs, for steady-state timing loops."""

    def _thunk() -> None:
        symm_buffer.launch(transformed_l1, transformed_l2)

    return _thunk


__all__ = [
    "Sm107MegaMoEMxfp8GluConfig",
    "Sm107Mxfp8GluKind",
    "Sm107Mxfp8GluSymmBuffer",
    "TransformedMxfp8GluWeights",
    "get_symm_buffer_for_sm107_mxfp8_glu_mega_moe",
    "sm107_mxfp8_glu_mega_launch_thunk",
    "sm107_mxfp8_glu_mega_moe",
]
