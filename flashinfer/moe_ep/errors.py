"""MoE EP v2 exceptions."""


class MoEEpNotBuiltError(RuntimeError):
    """Raised when an EP backend is invoked but its native libs are missing."""
