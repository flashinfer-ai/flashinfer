"""Exported MegaMoE KernelClass entry points."""

from importlib import import_module as _import_module

_kernel_imports = {
    'BlackwellInferenceGenphaseMegaMoE': ('.kernel_src.blackwell.inference.mega.block_scaled_swap_ab_mega_moe_kernel_gen_specialized', 'BlockScaledSwapAbGenphaseMoeKernel'),
    'BlackwellInferenceMegaMoE': ('.kernel_src.blackwell.inference.mega.block_scaled_swap_ab_mega_moe_kernel', 'BlockScaledSwapAbMegaMoeKernel'),
}

__all__ = list(_kernel_imports)


def __getattr__(name):
    try:
        module_name, class_name = _kernel_imports[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(_import_module(module_name, __name__), class_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
