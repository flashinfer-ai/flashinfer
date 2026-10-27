"""Resolve compiler defaults into the options used for both caching and codegen."""

from __future__ import annotations

import copy
import re
from functools import lru_cache


_OPERATIONAL_OPTIONS = {
    "KeepPTX",
    "KeepCUBIN",
    "KeepSASS",
    "DumpDir",
    "NvdisasmOptions",
    "RemarkFilter",
    "RemarkOutput",
    "WarningsFilter",
    "CollectCompilerDiagnostics",
}


def options_key(options) -> tuple[str, ...]:
    key = []
    for kind, option in sorted(
        options.options.items(), key=lambda item: item[0].__name__
    ):
        if kind.__name__ in _OPERATIONAL_OPTIONS:
            continue
        value = option.value
        if kind.__name__ == "ExtraCompilerOpts":
            value = " ".join(
                token for token in value.split() if not token.startswith("diagnostic=")
            )
        elif kind._value_kind == "int":
            value = int(value)
        key.append(f"{kind.__name__}={value}")
    key.append(f"DebugSelectors={','.join(sorted(options._debug_selectors))}")
    return tuple(key)


@lru_cache(maxsize=1)
def _resolved_options_type():
    from cutlass.base_dsl.compiler import CompileOptions

    class ResolvedCompileOptions(CompileOptions):
        def apply_envar_settings(self, envar, function_name):
            # Codegen settings were resolved before cache lookup. Only artifact
            # destinations need the compiler's final mangled function name.
            operational = copy.copy(envar)
            for name in (
                "enable_pyir",
                "debug",
                "enable_assertions",
                "lineinfo",
                "enable_tvm_ffi",
            ):
                setattr(operational, name, False)
            operational.compiler_opt = ""
            super().apply_envar_settings(operational, function_name)

    return ResolvedCompileOptions


@lru_cache(maxsize=128)
def resolve_compile_options(
    compile_callable, bracket_options=None, string_options=None
):
    from cutlass.base_dsl.compiler import (
        CompileCallable,
        CompileOptions,
        DeviceTarget,
        FrontendNext,
        GPUArch,
        RDC,
        _parse_compile_options_from_str,
    )
    from cutlass.cutlass_dsl import CuTeDSL

    if not isinstance(compile_callable, CompileCallable) and bracket_options is None:
        # Test doubles and alternate embedded runtimes retain their own contract.
        return compile_callable
    base = (
        CompileCallable(bracket_options)
        if bracket_options is not None
        else compile_callable
    )
    if string_options:
        explicit = _parse_compile_options_from_str(string_options)
        if base._compile_options.options[FrontendNext].value:
            explicit.options[FrontendNext].value = True
    else:
        explicit = copy.deepcopy(base._compile_options)
    envar = CuTeDSL._get_dsl().envar
    resolved = _resolved_options_type()()
    CompileOptions.apply_envar_settings(resolved, envar, "kernel")
    defaults = CompileOptions()
    # Explicit non-default settings override process defaults. Boolean false
    # overrides supplied as bracket options are retained as well.
    supplied = (
        bracket_options if isinstance(bracket_options, tuple) else (bracket_options,)
    )
    supplied_types = {
        item if isinstance(item, type) else type(item)
        for item in supplied
        if item is not None
    }
    if string_options:
        tokens = set(re.findall(r"[\w][\w-]*", string_options))
        supplied_types.update(
            kind
            for kind in explicit.options
            if kind._cli_flag in tokens or kind._option_name in tokens
        )
    for kind, option in explicit.options.items():
        if option.value != defaults.options[kind].value or kind in supplied_types:
            resolved.options[kind] = copy.deepcopy(option)
    resolved._debug_selectors.update(explicit._debug_selectors)
    resolved._ptxas_diagnostics_enabled |= explicit._ptxas_diagnostics_enabled
    if not resolved.options[GPUArch].value:
        resolved.options[GPUArch].value = envar.arch
    if resolved.options[DeviceTarget].value:
        resolved.options[RDC].value = True
    result = CompileCallable()
    result._compile_options = resolved
    result._b12x_options_key = options_key(resolved)
    return result


def clear_options_cache():
    resolve_compile_options.cache_clear()
