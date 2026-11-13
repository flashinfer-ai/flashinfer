"""Small-batch option of :func:`~flashinfer.moe_ep.MoEEpLayer`: the SM120 nosync kernel.

``MoEEpLayer(..., backend=<unchanged>, nosync=NosyncOptions())`` builds the layer the
deployment configures, exactly as without the option, and next to it the SM120 MXFP4 x MXFP8
nosync MegaMoE kernel (``sm120_mxfp4_mxfp8_bf16_nosync``). A call whose tokens per rank stay
within ``NosyncOptions.max_tokens_per_rank`` on every rank runs the nosync kernel; every other
call runs the configured layer. The option needs pre-quantized MXFP4 weights, the input of the
cake and cutedsl SM120 MegaMoE backends, whose packed FP4 tensors it shares. Where the nosync
kernel cannot run (not compute capability 12.0, weights that are not MXFP4 such as the BF16
weights of a ``SplitConfig`` layer, an unsupported geometry), the option does nothing and every
call runs the configured layer.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

from ..config import BootstrapConfig, FleetParams
from ..core.bootstrap_utils import bootstrap_comm_group
from ..core.validation.common import MoEEpConfigError
from ..weights import MoEWeightPack
from .config import MegaConfig

if TYPE_CHECKING:
    from ..tensors import MoEEpTensors

logger = logging.getLogger(__name__)

__all__ = ["MoEEpNosyncLayer", "NosyncOptions"]


@dataclass
class NosyncOptions:
    """Run calls of at most ``max_tokens_per_rank`` tokens per rank on the nosync kernel.

    ``max_tokens_per_rank`` (at most 128, the range where the kernel is measured faster than
    the cake and cutedsl backends) must be the same on every rank. ``transport`` and
    ``tokens_per_item`` are the nosync kernel's
    (:class:`~flashinfer.moe_ep.Sm120_Mxfp4_Mxfp8_Bf16_Nosync_MegaMoeConfig`); ``transport``
    ``None`` (default) takes the per-pair table the configured cake backend resolved
    (``transport_table``, including ``"auto"`` / ``"tuned"``), so both kernels move data
    between each pair of ranks the same way; ``"p2p"`` with any other backend. ``top_k`` and
    ``swiglu_limit`` (``inf`` for unclamped SwiGLU) default to the configured backend's when it
    is the cake or cutedsl SM120 MXFP4 backend; set them for any other backend.
    """

    max_tokens_per_rank: int = 128
    transport: str | tuple[int, ...] | None = None
    tokens_per_item: int | None = None
    top_k: int | None = None
    swiglu_limit: float | None = None


def _configured(backend: Any, name: str) -> Any:
    """``name`` of the configured backend's kernel config, when it has one."""
    kernel = getattr(backend, "megakernel", None) or getattr(backend, "kernel", None)
    return getattr(kernel, name, None)


def _configured_swiglu_limit(backend: Any) -> float | None:
    kernel = getattr(backend, "megakernel", None)
    name = getattr(kernel, "kernel_name", None)
    if name == "sm120_mxfp4_mxfp8_bf16_cake":
        return float(kernel.swiglu_limit) if kernel.swiglu_limit else math.inf
    if name == "sm120_mxfp4_mxfp8_bf16_cutedsl":
        limit = kernel.gate_up_clamp
        if limit is None:
            limit = kernel.activation_clamp
        return math.inf if limit is None else float(limit)
    return None


class MoEEpNosyncLayer(nn.Module):
    """The configured layer (``base``) plus the nosync kernel for small calls.

    Every rank must take the same kernel for a call. The layer decides by the call's token
    bound: ``compile_tokens_per_rank`` of :meth:`stage_inputs` when given, the call's own token
    count during CUDA graph capture (a captured bucket holds the same padded count on every
    rank), and otherwise the largest token count of any rank (one all-reduce and a host
    synchronization). Attributes the layer does not define come from ``base``.
    """

    def __init__(
        self,
        base: nn.Module,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
        weights: MoEWeightPack,
        backend: Any,
        options: NosyncOptions,
    ) -> None:
        super().__init__()
        self.base = base
        self.nosync: Optional[nn.Module] = None
        self.last_call_nosync = False
        self._group = bootstrap_comm_group(bootstrap)
        self._world = bootstrap.world_size
        self._staged: Optional[nn.Module] = None
        reason = self._build(bootstrap, fleet_params, weights, backend, options)
        if reason is not None:
            warnings.warn(
                f"NosyncOptions: the nosync kernel is off ({reason}); every call runs the "
                "configured backend",
                UserWarning,
                stacklevel=3,
            )

    def _build(self, bootstrap, fleet_params, weights, backend, options) -> str | None:
        from ..backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_nosync import (
            Sm120_Mxfp4_Mxfp8_Bf16_Nosync_MegaMoeConfig,
        )
        from ..backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_nosync.config import (
            MAX_TOKENS_PER_RANK,
        )
        from ..backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_nosync.weights import (
            is_mxfp4_pack,
            preprocess_mega_weights,
            shared_fp4,
        )
        from .mega_layer import MoEEpMegaLayer

        if not 1 <= options.max_tokens_per_rank <= MAX_TOKENS_PER_RANK:
            raise MoEEpConfigError(
                f"NosyncOptions.max_tokens_per_rank must be in [1, {MAX_TOKENS_PER_RANK}]"
            )
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            return "it needs compute capability 12.0"
        if not is_mxfp4_pack(weights):
            return "the weights are not pre-quantized MXFP4"
        top_k = (
            options.top_k
            if options.top_k is not None
            else _configured(backend, "top_k")
        )
        limit = options.swiglu_limit
        if limit is None:
            limit = _configured_swiglu_limit(backend)
        if top_k is None or limit is None:
            raise MoEEpConfigError(
                "NosyncOptions needs top_k and swiglu_limit for this backend (set "
                "NosyncOptions(top_k=..., swiglu_limit=...), inf for unclamped SwiGLU)"
            )
        intermediate = weights.w13.shape[1] // 2
        hidden = fleet_params.token_hidden_size
        transformed = getattr(self.base, "_transformed", None)
        kernel_name = _configured(backend, "kernel_name")
        fp4 = shared_fp4(transformed, kernel_name) if transformed is not None else None
        nosync_weights = preprocess_mega_weights(
            weights, hidden_size=hidden, intermediate_size=intermediate, fp4=fp4
        )
        if fp4 is None:
            extra = nosync_weights.w13.nbytes + nosync_weights.w2.nbytes
            logger.info(
                "NosyncOptions: the configured backend keeps its own weight layout; the nosync "
                "kernel holds a copy of the packed weights (%.2f GiB on this rank)",
                extra / 2**30,
            )
        transport = options.transport
        if transport is None:
            transport = "p2p"
            if kernel_name == "sm120_mxfp4_mxfp8_bf16_cake":
                # collective, like the nosync workspace below; cake resolves its table here
                table = self.base._ensure_workspace().transport_table
                transport = tuple(int(c) for c in table.cells)
                logger.info(
                    "NosyncOptions: transport follows the cake table %s %s",
                    table.name,
                    transport,
                )
        config = Sm120_Mxfp4_Mxfp8_Bf16_Nosync_MegaMoeConfig(
            intermediate_size=intermediate,
            top_k=top_k,
            swiglu_limit=limit,
            transport=transport,
            tokens_per_item=options.tokens_per_item,
        )
        params = FleetParams(
            num_experts=fleet_params.num_experts,
            max_tokens_per_rank=min(
                fleet_params.max_tokens_per_rank, options.max_tokens_per_rank
            ),
            token_hidden_size=hidden,
        )
        try:
            self.nosync = MoEEpMegaLayer(
                bootstrap,
                params,
                weights,
                MegaConfig(
                    megakernel=config,
                    preprocess_weights=False,
                    transformed_weights=nosync_weights,
                ),
            )
        except MoEEpConfigError as error:  # geometry the nosync kernel does not cover
            return str(error)
        self.nosync._ensure_workspace()  # collective: every rank builds the layer together
        self._limit = params.max_tokens_per_rank
        return None

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["base"], name)

    def _uses_nosync(self, rows: int, bound: int | None) -> bool:
        if self.nosync is None:
            return False
        if bound is None:
            bound = rows
            if self._world > 1 and not torch.cuda.is_current_stream_capturing():
                largest = torch.tensor([rows], dtype=torch.int32, device="cuda")
                torch.distributed.all_reduce(
                    largest, op=torch.distributed.ReduceOp.MAX, group=self._group
                )
                bound = int(largest.item())
        return bound <= self._limit

    def forward(self, t: "MoEEpTensors", **kwargs: Any) -> torch.Tensor:
        """``base.forward`` for large calls; the nosync kernel for small ones (``kwargs`` of
        ``base.forward`` that select its workspaces or graph state do not apply there)."""
        self.last_call_nosync = self._uses_nosync(t.num_tokens, None)
        if self.last_call_nosync:
            view = bool(kwargs.get("return_workspace_view", False))
            return self.nosync.forward(t, return_workspace_view=view)
        return self.base.forward(t, **kwargs)

    def stage_inputs(
        self,
        t: "MoEEpTensors",
        *,
        compile_tokens_per_rank: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.last_call_nosync = self._uses_nosync(t.num_tokens, compile_tokens_per_rank)
        self._staged = self.nosync if self.last_call_nosync else self.base
        if self.last_call_nosync:
            self.nosync.stage_inputs(t, compile_tokens_per_rank=compile_tokens_per_rank)
        else:
            self.base.stage_inputs(
                t, compile_tokens_per_rank=compile_tokens_per_rank, **kwargs
            )

    def compute_staged(self, *, output: torch.Tensor | None) -> torch.Tensor:
        if self._staged is None:
            raise MoEEpConfigError(
                "compute_staged() requires a prior stage_inputs() call"
            )
        return self._staged.compute_staged(output=output)

    def destroy(self) -> None:
        if self.nosync is not None:
            self.nosync.destroy()
        self.base.destroy()
