"""MoEEpMegaLayer — fused mega-MoE kernel path."""

from __future__ import annotations

import contextlib
import dataclasses
import weakref
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


class MoEEpMegaWorkspace:
    """Capacity-specific, reusable workspace owned by one MegaMoE layer.

    Create handles with :meth:`MoEEpMegaLayer.create_workspace`; do not
    construct them directly.  The backing symmetric workspace and output
    buffer keep stable addresses until :meth:`destroy` or layer destruction.
    """

    def __init__(
        self,
        layer: "MoEEpMegaLayer",
        fleet_params: FleetParams,
        backend_workspace: Any,
    ) -> None:
        self._layer_ref = weakref.ref(layer)
        self._fleet_params = fleet_params
        self._backend_workspace = backend_workspace
        self._destroyed = False

    @property
    def max_tokens_per_rank(self) -> int:
        """Maximum live-token count accepted by this handle."""

        return self._fleet_params.max_tokens_per_rank

    @property
    def is_destroyed(self) -> bool:
        """Whether this handle has released its backing workspace."""

        return self._destroyed

    def destroy(self) -> None:
        """Release this handle's pool reference; idempotent."""

        if self._destroyed:
            return
        layer = self._layer_ref()
        if layer is not None:
            layer._destroy_workspace_handle(self)
        else:
            self._destroyed = True
            self._backend_workspace = None

    close = destroy

    def __enter__(self) -> "MoEEpMegaWorkspace":
        if self._destroyed:
            raise MoEEpConfigError("MegaMoE workspace has been destroyed")
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.destroy()


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
        # Strong tracking is intentional: workspace create/destroy is an EP
        # collective and therefore must never be triggered by rank-local GC.
        self._workspaces: set[MoEEpMegaWorkspace] = set()
        self._preprocessing_count = 0
        self._destroyed = False

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
        self._preprocessing_count += 1
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

    def create_workspace(self, max_tokens_per_rank: int) -> MoEEpMegaWorkspace:
        """Allocate a reusable capacity-specific workspace for this layer.

        The handle reuses this layer's transformed weights and backend.  Call
        it collectively on all EP ranks, before CUDA graph capture.
        """

        if isinstance(max_tokens_per_rank, bool) or not isinstance(
            max_tokens_per_rank, int
        ):
            raise TypeError("max_tokens_per_rank must be an int")
        if max_tokens_per_rank <= 0:
            raise ValueError("max_tokens_per_rank must be positive")
        if self._destroyed:
            raise MoEEpConfigError("MegaMoE layer has been destroyed")
        if getattr(self._megakernel_config, "knobs", None) == "auto":
            raise MoEEpConfigError(
                "create_workspace() requires fixed/offline-tuned knobs; "
                "knobs='auto' has mutable per-capacity tuning state"
            )
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise MoEEpConfigError(
                "MegaMoE workspace allocation cannot run during CUDA graph capture"
            )
        fleet_params = dataclasses.replace(
            self._fleet_params,
            max_tokens_per_rank=max_tokens_per_rank,
        )
        self._kernel.validate_init(self._bootstrap, fleet_params)
        backend_workspace = self._kernel.prepare_workspace(
            self._bootstrap,
            fleet_params,
        )
        handle = MoEEpMegaWorkspace(self, fleet_params, backend_workspace)
        self._workspaces.add(handle)
        return handle

    def _resolve_workspace(
        self,
        workspace: MoEEpMegaWorkspace | None,
    ) -> tuple[FleetParams, Any]:
        if workspace is None:
            if self._destroyed:
                raise MoEEpConfigError("MegaMoE layer has been destroyed")
            return self._fleet_params, self._ensure_workspace()
        if not isinstance(workspace, MoEEpMegaWorkspace):
            raise TypeError(
                "workspace must be created by MoEEpMegaLayer.create_workspace()"
            )
        if workspace._layer_ref() is not self:
            raise MoEEpConfigError("MegaMoE workspace belongs to a different layer")
        if workspace._destroyed or workspace._backend_workspace is None:
            raise MoEEpConfigError("MegaMoE workspace has been destroyed")
        return workspace._fleet_params, workspace._backend_workspace

    def _destroy_workspace_handle(self, workspace: MoEEpMegaWorkspace) -> None:
        if workspace._destroyed:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise MoEEpConfigError(
                "MegaMoE workspace destruction cannot run during CUDA graph capture"
            )
        backend_workspace = workspace._backend_workspace
        if backend_workspace is not None:
            self._kernel.destroy(backend_workspace)
        # Commit the closed state only after the collective backend release
        # succeeds.  A failed release remains retryable on every rank.
        workspace._destroyed = True
        workspace._backend_workspace = None
        self._workspaces.discard(workspace)

    @property
    def supports_output_view(self) -> bool:
        """Whether ``forward(return_workspace_view=True)`` is supported."""
        return self._kernel.supports_output_view

    @property
    def preprocessing_count(self) -> int:
        """Number of successful weight transformations owned by this layer."""
        return self._preprocessing_count

    @property
    def workspace_pool_refcount(self) -> int:
        """Actual shared-workspace refcount, or zero before/after ownership."""
        if self._workspace is None:
            return 0
        from ..core.kernel.workspace_pool import pooled_workspace_refcount

        return pooled_workspace_refcount(self._workspace)

    @property
    def transformed_weights(self) -> Any:
        """Return the one-time transformed weights for sibling capacity layers."""
        if self._transformed is None:
            if not self._mega_config.preprocess_weights:
                raise MoEEpConfigError(
                    "preprocess_weights=False requires "
                    "MegaConfig.transformed_weights at init"
                )
            self._preprocess_weights()
        assert self._transformed is not None
        return self._transformed

    def warmup(
        self,
        t: Optional["MoEEpTensors"] = None,
        *,
        workspace: MoEEpMegaWorkspace | None = None,
    ) -> None:
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
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise MoEEpConfigError(
                "MegaMoE warmup cannot run during CUDA graph capture"
            )
        fleet_params, _backend_workspace = self._resolve_workspace(workspace)
        if t is None:
            if not self._mega_config.quantize_input:
                raise MoEEpConfigError(
                    "warmup() cannot build a dummy pre-quantized batch; pass "
                    "MoEEpTensors explicitly when quantize_input=False"
                )
            from ..tensors import MoEEpTensors

            fp = fleet_params
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
        self.forward(t, workspace=workspace)
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

    def forward(
        self,
        t: "MoEEpTensors",
        *,
        workspace: MoEEpMegaWorkspace | None = None,
        return_workspace_view: bool = False,
    ) -> torch.Tensor:
        """Run MegaMoE and return either an owned tensor or a workspace view.

        Without an explicit handle, the default allocates an owned output and
        preserves the existing API.  A reusable ``workspace`` returns its
        stable output view when the backend supports views.  The explicit
        ``return_workspace_view=True`` opt-in does the same for the default
        workspace.  A view remains valid under stream ordering until the next
        launch reuses that workspace.
        """
        ensure_bootstrap_dist_validated(self._bootstrap)
        quantize_input = self._resolve_quantize_input(t)

        if return_workspace_view and not self.supports_output_view:
            raise MoEEpConfigError(
                "return_workspace_view=True is not supported by this MegaMoE backend"
            )

        fleet_params, backend_workspace = self._resolve_workspace(workspace)
        self._kernel.validate_forward(
            t,
            fleet_params,
            quantize_input=quantize_input,
        )

        transformed_weights = self.transformed_weights

        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            self._kernel.validate_capture_ready(
                backend_workspace,
                transformed_weights,
            )

        y = None
        use_workspace_view = (
            return_workspace_view or workspace is not None
        ) and self.supports_output_view
        if not use_workspace_view:
            # Owned-output allocation must stay ahead of the staging round
            # (allocator work between stage and compute can sync the device
            # mid-round).
            y = torch.empty(
                t.num_tokens,
                fleet_params.token_hidden_size,
                dtype=torch.bfloat16,
                device=t.hidden_states.device,
            )
        self._kernel.stage_inputs(
            t,
            backend_workspace,
            quantize_input=quantize_input,
        )
        return self._kernel.compute(
            backend_workspace,
            transformed_weights,
            output=y,
        )

    def destroy(self) -> None:
        if self._destroyed:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise MoEEpConfigError(
                "MegaMoE layer destruction cannot run during CUDA graph capture"
            )
        for workspace in list(self._workspaces):
            self._destroy_workspace_handle(workspace)
        if self._workspace is not None:
            self._kernel.destroy(self._workspace)
            self._workspace = None
        if self._runtime is not None:
            finalize_moe_ep_runtime(self._runtime)
            self._runtime = None
        self._destroyed = True

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()
