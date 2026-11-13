"""Exported MegaMoE entry points, loaded on first access."""

from importlib import import_module

_exports = {
    'RubinInferenceGenphaseMegaMoE': ('sources.kernel_src.rubin.inference.mega.block_scaled_swap_ab_mega_moe_kernel_gen_specialized', 'BlockScaledSwapAbGenphaseMoeKernel'),
    'RubinInferenceMegaMoE': ('sources.kernel_src.rubin.inference.mega.block_scaled_swap_ab_mega_moe_kernel', 'BlockScaledSwapAbMegaMoeKernel'),
}

__all__ = list(_exports)


def __getattr__(name):
    if name not in _exports:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol = _exports[name]
    value = getattr(import_module('.' + module_name, __name__), symbol)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
