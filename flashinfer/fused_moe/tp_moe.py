# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""Fused tensor-parallel MoE layer for DeepSeek-V4-Flash and DeepSeek-V4.1-Flash decode.

One persistent kernel per call runs the whole TP MoE layer after the router GEMM:

  gate (sqrtsoftplus + bias, top-k, renormalize x routed_scaling_factor)
  -> x quantization -> routed MXFP4 experts (this rank's TP slice)
  -> TP-sharded FP8 shared expert -> cross-rank all-reduce

The all-reduce runs inside the kernel over CUDA-IPC-linked buffers that the
:class:`TpMoeWorkspace` owns, so every rank returns the same reduced output.
Hash layers bypass the gate through the ``topk_ids`` / ``topk_weights`` entry.

Supported: SM90a (Hopper) and SM100a (B200), TP 4 and 8, at most 64 tokens per
call (decode). Anything else raises; the caller keeps its own path for those cases.
"""

from __future__ import annotations

import dataclasses
from types import ModuleType
from typing import Literal, Optional

import torch

__all__ = [
    "TpMoeConfig",
    "TpMoeWeights",
    "TpMoeWorkspace",
    "tp_moe",
    "tp_moe_prepare_weights",
    "tp_moe_warmup",
]


@dataclasses.dataclass(frozen=True)
class _ModelShape:
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    shared_intermediate_size: int


_MODEL_SHAPES = {
    "dsv4_flash": _ModelShape(4096, 2048, 256, 6, 2048),
    "dsv41_flash": _ModelShape(5120, 2304, 384, 6, 2304),
}


@dataclasses.dataclass(frozen=True)
class TpMoeConfig:
    """Static description of one TP MoE layer family.

    The model fixes every shape; the kernels are compiled for these shapes only.
    """

    model: Literal["dsv4_flash", "dsv41_flash"]
    tp_size: Literal[4, 8]
    max_num_tokens: int = 64
    routed_scaling_factor: float = 1.5

    def __post_init__(self):
        if self.model not in _MODEL_SHAPES:
            raise ValueError(
                f"unsupported model {self.model!r}; expected one of {sorted(_MODEL_SHAPES)}"
            )
        if self.tp_size not in (4, 8):
            raise ValueError(f"tp_size must be 4 or 8, got {self.tp_size}")
        if not 1 <= self.max_num_tokens <= 64:
            raise ValueError(
                f"max_num_tokens must be in [1, 64], got {self.max_num_tokens}"
            )

    @property
    def shape(self) -> _ModelShape:
        return _MODEL_SHAPES[self.model]


def _arch_impl(device: torch.device) -> ModuleType:
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) == (9, 0):
        from . import _tp_moe_sm90

        return _tp_moe_sm90
    if (major, minor) == (10, 0):
        from . import _tp_moe_sm100

        return _tp_moe_sm100
    raise RuntimeError(f"tp_moe supports SM90a and SM100a only, got sm_{major}{minor}")


@dataclasses.dataclass
class TpMoeWeights:
    """This rank's weights in the kernel layout, from :func:`tp_moe_prepare_weights`.

    Opaque to callers: the arch module owns the fields in ``state``.
    """

    config: TpMoeConfig
    rank: int
    state: object


class TpMoeWorkspace:
    """Per-TP-group kernel state, shared by every layer of the model.

    Holds this rank's scratch tensors and the CUDA-IPC-linked buffers the
    in-kernel all-reduce exchanges through. Construction and :meth:`destroy`
    are collective over ``group``.
    """

    def __init__(
        self,
        config: TpMoeConfig,
        group: torch.distributed.ProcessGroup,
        device: Optional[torch.device] = None,
    ):
        import torch.distributed as dist

        if dist.get_world_size(group) != config.tp_size:
            raise ValueError(
                f"process group has {dist.get_world_size(group)} ranks, "
                f"config.tp_size is {config.tp_size}"
            )
        self.config = config
        self.group = group
        self.rank = dist.get_rank(group)
        self.device = torch.device(
            device if device is not None else torch.cuda.current_device()
        )
        self._impl = _arch_impl(self.device)
        self.state = self._impl.create_workspace(config, group, self.device)

    def destroy(self) -> None:
        if self.state is not None:
            self._impl.destroy_workspace(self.state, self.group)
            self.state = None


def tp_moe_prepare_weights(
    config: TpMoeConfig,
    rank: int,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    shared_w13: torch.Tensor,
    shared_w13_scale: torch.Tensor,
    shared_w2: torch.Tensor,
    shared_w2_scale: torch.Tensor,
    device: Optional[torch.device] = None,
) -> TpMoeWeights:
    """Build this rank's kernel-layout weights from the full (unsharded) checkpoint tensors.

    Args:
        w13: routed gate|up, packed FP4 (two e2m1 per uint8), ``[E, 2 * I, H // 2]``.
        w13_scale: UE8M0 block-32 scales, uint8 ``[E, 2 * I, H // 32]``.
        w2: routed down, packed FP4, ``[E, H, I // 2]``.
        w2_scale: UE8M0 block-32 scales, uint8 ``[E, H, I // 32]``.
        shared_w13 / shared_w2: shared expert gate|up ``[2 * I_s, H]`` and down
            ``[H, I_s]`` in float8_e4m3fn, with the checkpoint's block scales.

    The inputs may live on the CPU. The result holds one copy of this rank's
    slice (or expert group) on ``device``.
    """
    device = torch.device(device if device is not None else torch.cuda.current_device())
    if not 0 <= rank < config.tp_size:
        raise ValueError(f"rank {rank} out of range for tp_size {config.tp_size}")
    impl = _arch_impl(device)
    state = impl.prepare_weights(
        config,
        rank,
        device,
        w13=w13,
        w13_scale=w13_scale,
        w2=w2,
        w2_scale=w2_scale,
        shared_w13=shared_w13,
        shared_w13_scale=shared_w13_scale,
        shared_w2=shared_w2,
        shared_w2_scale=shared_w2_scale,
    )
    return TpMoeWeights(config, rank, state)


def tp_moe_warmup(config: TpMoeConfig, num_tokens=None, device=None) -> None:
    """Compile (or load from cache) the kernel builds that serve ``num_tokens``.

    Call before CUDA graph capture. ``None`` warms every build up to
    ``config.max_num_tokens``.
    """
    device = torch.device(device if device is not None else torch.cuda.current_device())
    impl = _arch_impl(device)
    ms = (
        range(1, config.max_num_tokens + 1)
        if num_tokens is None
        else ([num_tokens] if isinstance(num_tokens, int) else num_tokens)
    )
    for variant in {impl.build_for_m(config, m) for m in ms}:
        impl.load_module(variant)


def tp_moe(
    x: torch.Tensor,
    weights: TpMoeWeights,
    workspace: TpMoeWorkspace,
    *,
    router_logits: Optional[torch.Tensor] = None,
    gate_bias: Optional[torch.Tensor] = None,
    gate_bias_vision: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    image_token_id: Optional[int] = None,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the TP MoE layer on this rank's ``x`` and return the all-reduced output.

    Exactly one routing entry:

    * gate entry: ``router_logits`` fp32 ``[M, E]`` and ``gate_bias`` fp32 ``[E]``.
      DeepSeek-V4.1 also takes ``gate_bias_vision``, ``input_ids`` int32 ``[M]``
      and ``image_token_id``: image tokens use the vision bias.
    * ids entry (hash layers): ``topk_ids`` int32 ``[M, top_k]`` (global expert
      ids) and ``topk_weights`` fp32 ``[M, top_k]``: the renormalized weights; the
      op applies ``routed_scaling_factor``.

    Every rank must call with the same ``M``. The result is bitwise identical
    on all ranks and across repeated calls with the same inputs.
    """
    config = workspace.config
    shape = config.shape
    if weights.config != config:
        raise ValueError("weights and workspace were built for different configs")
    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != shape.hidden_size:
        raise ValueError(
            f"x must be bf16 [M, {shape.hidden_size}], got {x.dtype} {tuple(x.shape)}"
        )
    m = x.shape[0]
    if not 1 <= m <= config.max_num_tokens:
        raise ValueError(f"M = {m} outside [1, {config.max_num_tokens}]")

    gate_entry = router_logits is not None
    if gate_entry == (topk_ids is not None):
        raise ValueError("pass exactly one of router_logits or topk_ids")
    if gate_entry:
        if gate_bias is None:
            raise ValueError("the gate entry needs gate_bias")
        vision = (gate_bias_vision, input_ids, image_token_id)
        if config.model == "dsv41_flash":
            if any(v is None for v in vision):
                raise ValueError(
                    "dsv41_flash needs gate_bias_vision, input_ids and image_token_id"
                )
        elif any(v is not None for v in vision):
            raise ValueError("vision gate inputs apply to dsv41_flash only")
    elif topk_weights is None:
        raise ValueError("the ids entry needs topk_weights")

    if out is None:
        out = torch.empty_like(x)
    workspace._impl.run(
        config,
        x,
        weights.state,
        workspace.state,
        out,
        router_logits=router_logits,
        gate_bias=gate_bias,
        gate_bias_vision=gate_bias_vision,
        input_ids=input_ids,
        image_token_id=image_token_id,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    return out
