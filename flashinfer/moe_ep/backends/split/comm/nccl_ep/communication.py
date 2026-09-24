"""NCCL-EP MoE communication, built on the NCCL-EP :class:`Fleet`/``Handle``.

Uses the low-latency RANK_MAJOR layout, whose receive buffer is
``[ep_size, max_tokens_per_rank, hidden]`` and whose combine sums per-rank
partial results. Layouts outside the rank-major contract (EXPERT_MAJOR,
high-throughput FLAT) and split send/receive staging remain available
through the Fleet/Handle API directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from .....algo_knobs import (
    AlgoKnob,
    HandleAlgoKnobTopKWeights,
    HandleAlgoKnobUserStream,
)
from .....config import (
    CombineInputParams,
    DispatchInputParams,
    EpAlgorithm,
    EpLayout,
    FleetParams,
    HandleParams,
)
from .....core.comm.communication import (
    MoEEpCommParams,
    MoEEpCommunication,
    MoEEpDispatchResult,
    register_communication,
)
from .....core.comm.fleet import Fleet, create_fleet
from .config import NcclEpConfig

if TYPE_CHECKING:
    import torch

    from .....config import BootstrapConfig
    from .....core.comm.handle import Handle


@register_communication("nccl_ep")
class NcclEpCommunication(MoEEpCommunication):
    """Rank-major dispatch/combine over an NCCL-EP low-latency fleet.

    The fleet is created on the first :meth:`dispatch` (a collective over the
    EP group); ``fleet_knobs`` configure it (fault tolerance, allocator, ...).
    ``topk_weights`` are required. Received ids of picks owned by other ranks
    are reported as ``invalid_expert_id``. Each dispatch creates its own
    handle, a host-side allocation, so dispatch cannot be captured into a CUDA
    graph; capture through ``Fleet``/``Handle`` persistent handles instead.
    """

    supports_cuda_graph = False

    def __init__(
        self,
        bootstrap: "BootstrapConfig",
        params: MoEEpCommParams,
        config: Optional[NcclEpConfig] = None,
        fleet_knobs: Sequence[AlgoKnob] = (),
    ) -> None:
        super().__init__(bootstrap, params)
        self.config = NcclEpConfig() if config is None else config
        self.fleet_params = FleetParams(
            num_experts=params.num_experts,
            max_tokens_per_rank=params.max_tokens_per_rank,
            token_hidden_size=params.hidden_size,
            dtype_bytes=params.token_dtype.itemsize,
            algorithm=EpAlgorithm.LOW_LATENCY,
            layout=EpLayout.RANK_MAJOR,
        )
        self._fleet_knobs = tuple(fleet_knobs)
        self._fleet: Optional[Fleet] = None
        self._handle: Optional["Handle"] = None

    @classmethod
    def is_platform_supported(cls) -> bool:
        from ..... import have_nccl_ep

        return have_nccl_ep()

    @property
    def fleet(self) -> Fleet:
        """The underlying NCCL-EP fleet, created on first use."""
        if self._fleet is None:
            self._fleet = create_fleet(
                self.bootstrap,
                self.fleet_params,
                self._fleet_knobs,
                backend=self.config,
            )
        return self._fleet

    def dispatch(
        self,
        hidden_states: "torch.Tensor",
        topk_ids: "torch.Tensor",
        topk_weights: "torch.Tensor | None" = None,
        *,
        hidden_states_scale: "torch.Tensor | None" = None,
        max_tokens_per_rank: Optional[int] = None,
        eplb_local_stats: "torch.Tensor | None" = None,
    ) -> MoEEpDispatchResult:
        """See :meth:`MoEEpCommunication.dispatch`.

        The receive buffer is always sized for
        ``MoEEpCommParams.max_tokens_per_rank``; ``max_tokens_per_rank`` is
        accepted for interface compatibility and ignored.
        """
        import torch

        if self._handle is not None:
            raise RuntimeError("dispatch called twice without an intervening combine")
        if topk_weights is None:
            raise ValueError("NcclEpCommunication.dispatch requires topk_weights")
        if hidden_states_scale is not None:
            raise NotImplementedError(
                "NcclEpCommunication does not forward per-token scale factors"
            )
        if eplb_local_stats is not None:
            raise NotImplementedError(
                "NcclEpCommunication does not gather EPLB statistics"
            )

        handle = self.fleet.create_handle(
            HandleParams(topk_ids=topk_ids),
            algo_knobs=[
                HandleAlgoKnobUserStream(
                    stream=torch.cuda.current_stream().cuda_stream
                ),
                HandleAlgoKnobTopKWeights(weights=topk_weights),
            ],
        )
        try:
            out = handle.dispatch(DispatchInputParams(x=[hidden_states]))
        except Exception:
            handle.destroy()
            raise
        self._handle = handle

        world, tokens_per_rank = out.expert_tensors.shape[:2]
        local_ids = out.recv_topk_idx
        # RANK_MAJOR reports ids local to this rank, -1 for picks owned by
        # other ranks, and leaves rows past each source rank's count unwritten.
        row_valid = (
            torch.arange(tokens_per_rank, device=local_ids.device)[None, :]
            < out.expert_counts[:, None]
        ).reshape(world * tokens_per_rank, 1)
        is_local = (local_ids >= 0) & (local_ids < self.num_local_experts) & row_valid
        global_ids = torch.where(
            is_local,
            local_ids + self.ep_rank * self.num_local_experts,
            torch.full_like(local_ids, self.params.invalid_expert_id),
        )
        return MoEEpDispatchResult(
            hidden_states=out.expert_tensors.flatten(0, 1),
            topk_ids=global_ids,
            topk_weights=out.recv_topk_weights,
            tokens_per_rank=tokens_per_rank,
        )

    def combine(
        self,
        expert_output: "torch.Tensor",
        *,
        output: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        handle = self._handle
        if handle is None:
            raise RuntimeError("combine called before dispatch")
        if expert_output.dim() == 2:
            expert_output = expert_output.view(
                self.ep_size, self.params.max_tokens_per_rank, -1
            )
        try:
            combined = handle.combine(
                CombineInputParams(x=[expert_output], out=output)
            ).x
            handle.complete()
        finally:
            handle.destroy()
            self._handle = None
        return combined

    def destroy(self) -> None:
        if self._handle is not None:
            self._handle.destroy()
            self._handle = None
        if self._fleet is not None:
            self._fleet.destroy()
            self._fleet = None
