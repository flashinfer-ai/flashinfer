"""Restore argument names and READY defaults for loaded MegaMoE functions."""

from inspect import signature


def make_megamoe_aot_callable(function, kernel):
    """Use TVM-FFI's native adapter without attaching scheduler metadata."""
    from tvm_ffi.utils.kwargs_wrapper import make_kwargs_wrapper

    argument_names = list(signature(kernel.__call__).parameters)
    # Trailing operands, including per-token scales and live counts, default to None.
    trailing = len(argument_names) - 1 - argument_names.index("hot_expert_weight_ready_generation")
    # Home-only entries can omit the unused READY pair, as with direct JIT
    # calls. Helper-enabled entries require both READY operands explicitly.
    ready = 2 if kernel.helper_expert_count == 0 else 0
    defaults = (None,) * (ready + trailing)
    return make_kwargs_wrapper(function, argument_names, arg_defaults=defaults)
