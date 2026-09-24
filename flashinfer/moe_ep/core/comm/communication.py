"""MoEEpCommunication — MoE-level expert-parallel token exchange.

A communication backend moves routed tokens to the ranks that own their
experts (``dispatch``) and brings the expert outputs back to the ranks the
tokens came from (``combine``). It is the level a MoE layer programs against;
how a backend moves the bytes (symmetric-memory kernels, an NCCL-EP
``Fleet``/``Handle`` pair, ...) is an implementation detail beneath it.

Every backend implements the same rank-major contract:

* ``dispatch`` returns ``ep_size * tokens_per_rank`` receive rows. Each token
  arrives at most once per rank, in one row, together with its full top-k
  routing. Which row a token lands in is backend-defined; ``combine`` expects
  each row's result in the same row.
* Received ``topk_ids`` are GLOBAL expert ids. Rows that carry no token have
  every id set to ``MoEEpCommParams.invalid_expert_id``. Picks owned by
  another rank are either that rank's expert id or ``invalid_expert_id``; the
  expert computation must only evaluate ids in this rank's local expert range.
* The expert computation applies ``topk_weights`` and reduces over this
  rank's local experts, producing one output row per received row.
* ``combine`` sums each token's per-rank partial results into
  ``[local_num_tokens, hidden]`` on its source rank.

Backends register themselves with :func:`register_communication` at import
time and are instantiated through :func:`create_communication`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional

if TYPE_CHECKING:
    import torch

    from ...config import BootstrapConfig


_COMMUNICATION_REGISTRY: "dict[str, type[MoEEpCommunication]]" = {}


@dataclass(frozen=True)
class MoEEpCommParams:
    """EP sizing shared by every communication backend.

    ``max_tokens_per_rank`` bounds the number of tokens any rank dispatches in
    one call and sizes the backend's receive buffers. ``hidden_size`` and
    ``dtype`` describe the unquantized token row; backends size their payload
    buffers for it, which also bounds quantized rows plus their scale factors.
    """

    num_experts: int
    top_k: int
    max_tokens_per_rank: int
    hidden_size: int
    dtype: "torch.dtype | None" = None
    invalid_expert_id: int = -1

    def __post_init__(self) -> None:
        for name in ("num_experts", "top_k", "max_tokens_per_rank", "hidden_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"MoEEpCommParams.{name} must be a positive int")
        if self.top_k > self.num_experts:
            raise ValueError(
                f"MoEEpCommParams.top_k ({self.top_k}) exceeds num_experts "
                f"({self.num_experts})"
            )
        if 0 <= self.invalid_expert_id < self.num_experts:
            raise ValueError(
                "MoEEpCommParams.invalid_expert_id must lie outside "
                f"[0, num_experts={self.num_experts}), got {self.invalid_expert_id}"
            )

    @property
    def token_dtype(self) -> "torch.dtype":
        """``dtype``, defaulting to BF16."""
        import torch

        return torch.bfloat16 if self.dtype is None else self.dtype


@dataclass(frozen=True)
class MoEEpDispatchResult:
    """Rank-major receive buffers produced by :meth:`MoEEpCommunication.dispatch`.

    All tensors have ``ep_size * tokens_per_rank`` rows (see the module
    docstring for the row contract). They may be views into backend-owned
    buffers that stay valid until the matching :meth:`combine`.
    """

    hidden_states: "torch.Tensor"
    topk_ids: "torch.Tensor"
    topk_weights: "torch.Tensor | None"
    tokens_per_rank: int
    hidden_states_scale: "torch.Tensor | None" = None
    eplb_gathered_stats: "torch.Tensor | None" = None


class MoEEpCommunication(ABC):
    """Abstract MoE expert-parallel dispatch/combine.

    One instance serves one EP group and one in-flight dispatch/combine pair
    at a time: every :meth:`dispatch` must be followed by exactly one
    :meth:`combine` before the next :meth:`dispatch`. All ranks of the group
    must issue the same sequence of calls.
    """

    backend_name: ClassVar[str]

    def __init__(
        self,
        bootstrap: "BootstrapConfig",
        params: MoEEpCommParams,
    ) -> None:
        if not self.is_platform_supported():
            raise RuntimeError(
                f"{type(self).__name__} is not supported on this platform."
            )
        self.bootstrap = bootstrap
        self.params = params
        self.ep_rank = bootstrap.rank
        self.ep_size = bootstrap.world_size
        if params.num_experts % self.ep_size != 0:
            raise ValueError(
                f"{type(self).__name__} requires num_experts ({params.num_experts}) "
                f"divisible by the EP size ({self.ep_size})"
            )
        self.num_local_experts = params.num_experts // self.ep_size

    @classmethod
    @abstractmethod
    def is_platform_supported(cls) -> bool:
        """Whether this backend can run on the current machine.

        Callable before construction; performs no collective work.
        """

    def is_workload_feasible(self, max_tokens_per_rank: int) -> bool:
        """Whether a step whose busiest rank dispatches ``max_tokens_per_rank``
        tokens fits this instance's buffers."""
        return max_tokens_per_rank <= self.params.max_tokens_per_rank

    @abstractmethod
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
        """Send each token to the ranks that own its selected experts.

        Parameters
        ----------
        hidden_states : torch.Tensor
            ``[local_num_tokens, *]`` token rows, unquantized or quantized.
        topk_ids : torch.Tensor
            ``[local_num_tokens, top_k]`` global expert ids.
        topk_weights : torch.Tensor, optional
            ``[local_num_tokens, top_k]`` routing weights, forwarded with the
            tokens so the expert computation can apply them.
        hidden_states_scale : torch.Tensor, optional
            ``[local_num_tokens, *]`` per-token scale factors of quantized
            ``hidden_states``, forwarded row-aligned with them.
        max_tokens_per_rank : int, optional
            Largest ``local_num_tokens`` over all ranks for this step; it sets
            ``tokens_per_rank`` of the result. Every rank must pass the same
            value. ``None`` uses ``MoEEpCommParams.max_tokens_per_rank``, which
            keeps shapes static (e.g. for CUDA graphs). Backends with fixed
            receive buffers ignore it.
        eplb_local_stats : torch.Tensor, optional
            This rank's expert-load statistics, all-gathered alongside the
            dispatch by backends that support it.
        """

    @abstractmethod
    def combine(
        self,
        expert_output: "torch.Tensor",
        *,
        output: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Return each token's expert outputs, summed over ranks.

        Parameters
        ----------
        expert_output : torch.Tensor
            ``[ep_size * tokens_per_rank, hidden]`` (or the equivalent
            ``[ep_size, tokens_per_rank, hidden]``) per-row outputs of the
            expert computation, already weighted and reduced over this rank's
            local experts.
        output : torch.Tensor, optional
            Preallocated ``[local_num_tokens, hidden]`` result buffer.
        """

    def get_combine_input_buffer(self, dtype: "torch.dtype") -> "torch.Tensor | None":
        """Backend-owned buffer the expert computation may write its output into.

        Passing the returned tensor to :meth:`combine` saves a copy. Valid only
        between :meth:`dispatch` and :meth:`combine`. Backends without such a
        buffer return ``None``.
        """
        return None

    def destroy(self) -> None:  # noqa: B027 - intentional no-op default
        """Release backend resources. Collective on every rank, idempotent."""


def register_communication(
    name: str,
) -> Callable[[type[MoEEpCommunication]], type[MoEEpCommunication]]:
    """Class decorator registering a :class:`MoEEpCommunication` backend."""

    def decorator(cls: type[MoEEpCommunication]) -> type[MoEEpCommunication]:
        cls.backend_name = name
        _COMMUNICATION_REGISTRY[name] = cls
        return cls

    return decorator


def communication_backend_name(backend: Any) -> str:
    """Resolve a backend string or config object to its registered name."""
    name = getattr(backend, "backend_name", backend)
    if not isinstance(name, str):
        raise TypeError(
            "backend must be a string or have a .backend_name str attr; "
            f"got {backend!r}"
        )
    return name


def is_communication_backend(backend: Any) -> bool:
    """Whether ``backend`` names a registered communication backend."""
    try:
        return communication_backend_name(backend) in _COMMUNICATION_REGISTRY
    except TypeError:
        return False


def available_communication_backends() -> list[str]:
    """Registered communication backends usable on this machine."""
    return sorted(
        name
        for name, cls in _COMMUNICATION_REGISTRY.items()
        if cls.is_platform_supported()
    )


def create_communication(
    bootstrap: "BootstrapConfig",
    params: MoEEpCommParams,
    backend: Any,
    **options: Any,
) -> MoEEpCommunication:
    """Instantiate the registered communication backend named by ``backend``.

    ``backend`` is a backend name or a backend config object (anything with a
    ``backend_name`` attribute). A config object is passed to the backend as
    ``config``; ``options`` are forwarded as keyword arguments.
    """
    name = communication_backend_name(backend)
    cls = _COMMUNICATION_REGISTRY.get(name)
    if cls is None:
        raise KeyError(
            f"unknown communication backend {name!r}; available: "
            f"{sorted(_COMMUNICATION_REGISTRY)}"
        )
    if not isinstance(backend, str):
        options["config"] = backend
    return cls(bootstrap, params, **options)
