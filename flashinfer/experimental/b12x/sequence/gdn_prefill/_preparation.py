"""GDN prefill declaration and shared native pipeline materialization."""
from .._shared.delta_prefill.preparation import invocation_from_tensors
from .._shared.delta_prefill.preparation import make_plan as _make_plan
from . import _impl, _tuning


def make_plan(caps, *, invocation, override):
    return _make_plan(caps, _impl, _tuning, invocation=invocation, override=override)
