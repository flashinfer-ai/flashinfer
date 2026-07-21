"""MoEEpMegaLayer — fused mega-MoE kernel path."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

from ..config import BootstrapConfig, FleetParams
from ..core.kernel.registry import create_mega_kernel
from ..core.runtime import (
    bootstrap_moe_ep_runtime,
    ensure_moe_ep_cuda_device,
    finalize_moe_ep_runtime,
)
from ..core.validation.common import (
    MoEEpConfigError,
    ensure_bootstrap_dist_validated,
    validate_bootstrap_world_size,
    validate_fleet_weights,
)
from ..weights import MoEWeightPack
from .config import MegaConfig

if TYPE_CHECKING:
    from ..tensors import MoEEpTensors


class MoEEpMegaLayer(nn.Module):
    """Fused EP mega kernel — no separate dispatch/combine transport.

    Memory invariant: the source ``MoEWeightPack`` is released as soon as the
    kernel's transformed weights exist — the transformed tensors own the
    memory. Retaining the pack would hold a per-layer dequant copy (multiple
    GB at large-model geometry) across every MoE layer and OOM at model load.
    When ``backend.transformed_weights`` is supplied, the source pack is never
    stored at all.

    CUDA graphs: call :meth:`warmup` on ALL EP ranks first, then capture
    ``forward``. Under capture the output tensor returned at capture time is
    the one the graph writes on every replay — consume that same tensor
    across replays (standard graph practice). Lazy compile/alloc/autotune
    paths raise if they would fire mid-capture instead of corrupting it.
    """

    def __init__(
        self,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
        weights: MoEWeightPack,
        backend: MegaConfig,
    ) -> None:
        super().__init__()
        self._bootstrap = bootstrap
        self._fleet_params = fleet_params
        self._mega_config = backend
        self._megakernel_config = backend.megakernel

        ensure_moe_ep_cuda_device(bootstrap)

        self._kernel = create_mega_kernel(self._megakernel_config)
        self._kernel.bind_ep_bootstrap(bootstrap)
        self._runtime = None
        if bootstrap.auto_bootstrap:
            self._runtime = bootstrap_moe_ep_runtime(
                bootstrap,
                self._kernel.runtime_requirements(bootstrap),
            )

        validate_bootstrap_world_size(bootstrap)
        self._kernel.validate_init(bootstrap, fleet_params)

        if backend.transformed_weights is None:
            validate_fleet_weights(weights, fleet_params, bootstrap.world_size)

        self._weights: Optional[MoEWeightPack] = (
            weights if backend.transformed_weights is None else None
        )
        self._transformed: Optional[Any] = None
        self._workspace: Any = None

        if backend.transformed_weights is not None:
            self._transformed = backend.transformed_weights
            self._kernel.validate_transformed_weights(
                self._transformed,
                self._bootstrap,
                self._fleet_params,
            )
        elif backend.preprocess_weights:
            self._preprocess_weights()

    def _preprocess_weights(self) -> None:
        if self._transformed is not None:
            return
        assert self._weights is not None, (
            "source weight pack was released but no transformed weights exist"
        )
        self._transformed = self._kernel.preprocess_weights(
            self._weights, self._fleet_params
        )
        self._weights = None

    def _ensure_workspace(self) -> Any:
        if self._workspace is None:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise MoEEpConfigError(
                    "mega workspace allocation (symmetric heap) cannot run "
                    "during CUDA graph capture; call warmup() on all EP ranks "
                    "before capturing"
                )
            self._workspace = self._kernel.prepare_workspace(
                self._bootstrap, self._fleet_params
            )
        return self._workspace

    def warmup(self, t: Optional["MoEEpTensors"] = None) -> None:
        """Run one full eager forward so ``forward`` becomes graph-capturable.

        Forces every lazy host-side step — workspace allocation (symmetric
        heap), ``cute.compile``, the ``knobs="auto"`` autotune sweep, and one
        real kernel launch (module load) — then synchronizes the device.

        COLLECTIVE: call on ALL EP ranks together before any rank starts
        capturing (the kernel has cross-rank device-side barriers, and the
        lazy steps include collective symmetric-heap allocation).

        ``t`` defaults to a max-shape dummy batch. Pass a real batch when
        ``quantize_input=False`` — pre-quantized activations and scales
        cannot be fabricated here.
        """
        if t is None:
            if not self._mega_config.quantize_input:
                raise MoEEpConfigError(
                    "warmup() cannot build a dummy pre-quantized batch; pass "
                    "MoEEpTensors explicitly when quantize_input=False"
                )
            from ..tensors import MoEEpTensors

            fp = self._fleet_params
            device = torch.device("cuda", torch.cuda.current_device())
            # Every mega kernel config declares top_k: int; the MegaConfig
            # field is duck-typed `object` (kernel-specific config union).
            top_k = int(self._megakernel_config.top_k)  # type: ignore[attr-defined]
            num_tokens = fp.max_tokens_per_rank
            t = MoEEpTensors(
                hidden_states=torch.zeros(
                    num_tokens,
                    fp.token_hidden_size,
                    dtype=torch.bfloat16,
                    device=device,
                ),
                # Distinct in-range experts per row (top_k <= num_experts is
                # validated at init), spread across all experts.
                topk_ids=(
                    torch.arange(num_tokens * top_k, device=device) % fp.num_experts
                ).view(num_tokens, top_k),
                topk_weights=torch.full(
                    (num_tokens, top_k),
                    1.0 / top_k,
                    dtype=torch.float32,
                    device=device,
                ),
            )
        self.forward(t)
        torch.cuda.synchronize()

    def _resolve_quantize_input(self, t: "MoEEpTensors") -> bool:
        if not self._mega_config.quantize_input:
            return False
        if t.hidden_states.dtype != torch.bfloat16:
            raise MoEEpConfigError(
                f"MegaConfig.quantize_input=True expects bf16 hidden_states; "
                f"got {t.hidden_states.dtype}. Set quantize_input=False and provide "
                f"MoEEpTensors.scales for pre-quantized activations."
            )
        return True

    def forward(self, t: "MoEEpTensors") -> torch.Tensor:
        ensure_bootstrap_dist_validated(self._bootstrap)
        quantize_input = self._resolve_quantize_input(t)

        self._kernel.validate_forward(
            t,
            self._fleet_params,
            quantize_input=quantize_input,
        )

        if self._transformed is None:
            if not self._mega_config.preprocess_weights:
                raise MoEEpConfigError(
                    "preprocess_weights=False requires "
                    "MegaConfig.transformed_weights at init"
                )
            self._preprocess_weights()
        assert self._transformed is not None

        workspace = self._ensure_workspace()

        self._kernel.stage_inputs(
            t,
            workspace,
            quantize_input=quantize_input,
        )

        y = torch.empty(
            t.num_tokens,
            self._fleet_params.token_hidden_size,
            dtype=torch.bfloat16,
            device=t.hidden_states.device,
        )
        return self._kernel.compute(
            workspace,
            self._transformed,
            output=y,
        )

    def destroy(self) -> None:
        if self._workspace is not None:
            self._kernel.destroy(self._workspace)
            self._workspace = None
        if self._runtime is not None:
            finalize_moe_ep_runtime(self._runtime)
            self._runtime = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()
