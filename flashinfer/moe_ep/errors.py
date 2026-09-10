"""MoE EP v2 exceptions."""

from __future__ import annotations


class MoEEpNotBuiltError(RuntimeError):
    """Raised when an EP backend is invoked but its native libs are missing."""


class MoEEpFaultToleranceUnsupportedError(RuntimeError):
    """Raised when an FT API is called on a Fleet/transport that can't serve it.

    Either the Fleet was built without
    :class:`~flashinfer.moe_ep.algo_knobs.FleetAlgoKnobFaultTolerance`, or the
    installed transport predates the mask API (see
    :func:`flashinfer.moe_ep.supports_fault_tolerance`).
    """


class MoEEpRankEvictedError(RuntimeError):
    """This rank was masked out by its peers during mask reconciliation.

    The survivors agreed we are dead — typically because our dispatch stalled
    long enough for their kernels to time out on us. We cannot apply that
    decision (a rank cannot mask itself, and both transports reject it), and
    we must not keep serving as if we were still in the group: our peers are
    no longer sending us tokens.

    The framework owns the recovery: tear this worker down, or rejoin through
    a fresh Fleet once the survivors call ``clear_faults(readmit=True)``.
    """


class MoEEpTransportError(RuntimeError):
    """A native transport call returned a non-success status."""

    def __init__(
        self, fn: str, code: int, detail: str = "", code_name: str | None = None
    ) -> None:
        self.fn = fn
        self.code = code
        self.code_name = code_name
        shown = f"{code_name} ({code})" if code_name else str(code)
        super().__init__(f"{fn} failed: {shown}{detail}")
