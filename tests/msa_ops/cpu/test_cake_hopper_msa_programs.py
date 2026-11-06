"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Registry-form consistency of the Cake Hopper (SM90) MSA programs, on the CPU.

``flashinfer.jit.cake_hopper_msa`` registers one program per physical
schedule: ``MODULES`` carries each program's two sources, compile flags, FFI
entry and argument order, ``ROUTES`` maps each logical route to the program
that serves it.  A binding source is shared by the programs of one role and a
kernel source by the programs that differ only in a compile-time constant; the
record's ``-D`` compile flags select the program (its kernel symbol, dynamic
shared memory and block size for the binding).  These checks hold the registry
to its sources under ``csrc/cake_hopper_msa/`` and to the host module that
binds the argument order: each record's kernel and binding unit exist, the
binding has the generated form (its namespace, the runtime header, the two
exported functions) and names the kernel through the record's flags, the
kernel defines that symbol, every argument plan ends in the grid triple and
names only arguments the host binds, every tensor the binding checks for a
dtype is checked for the dtype the host passes, each program serves exactly
one route, and the JIT key is the content digest the loader documents.

No CUDA device and no build: the units are read as text and the host module
as source.  No program count, program name or digest is pinned here; the
registry is the input, so a regenerated tree is checked without an edit.
"""

import ast
import hashlib
import inspect
import json
import re

import pytest

from flashinfer.jit import cake_hopper_msa as jm
from flashinfer.msa_ops import cake_hopper_sm90 as ch

_PROGRAMS = sorted(jm.MODULES)
# The loader's own source lookup (``_csrc_dir``): the directory the JIT spec reads the units from.
_CSRC = jm._csrc_dir() / "cake_hopper_msa"

# Route family (``cake_hopper_sm90._route_family``) -> the two host functions that bind a program's
# arguments: the launch template, whose ``constants`` hold the per-geometry scalars and retained
# buffers, and the public launcher, whose ``launch(...)`` call passes the per-call tensors.
_HOST_SITES = {
    "decode_v5": ("_decode_template", "hopper_msa_sparse_decode_attention"),
    "proxy_decode": ("_proxy_decode_template", "hopper_msa_proxy_score_decode"),
    "proxy_prefill": ("_proxy_prefill_template", "hopper_msa_proxy_score_prefill"),
    "topk_select": ("_topk_template", "hopper_msa_topk_select"),
    "prefill_v1": ("_prefill_template", "hopper_msa_sparse_attention"),
    "prefill_v2": ("_prefill_template", "hopper_msa_sparse_attention"),
}


# The DLPack dtype the host module passes for every tensor of a program's plan, as the binding's
# ``check_dtype(<tensor>, DLDataType{<code>, <bits>, 1}, "<name>")`` compares it (code, bits and lanes
# exactly; ``cake_ffi_checks.h``).  Transcribed by hand from the host (edit with any .view(torch.<dtype>) change at the launch sites; a drift fails at the first launch, not here): ``_admit_sparse_decode`` admits bf16 ``q``, an
# fp8 e4m3 cache, int32 tables and a bf16 ``out``; ``hopper_msa_sparse_decode_attention`` passes
# ``q.view(torch.uint32)`` and the ``uint8`` views of the two cache halves; ``_scratch`` allocates
# uint32 / float32 / uint32.  ``_admit_sparse_prefill`` admits the same dtypes and
# ``hopper_msa_sparse_attention`` passes ``Q=q`` beside ``Q32``, the cache halves as fp8 e4m3 views for
# the v1 program and as uint8 views for v2 (``plan_sparse_prefill`` picks).  ``_admit_proxy_decode``
# admits ``q`` and the index cache in one dtype (fp8 e4m3 or bf16, the route's first coordinate) and a
# float32 ``per_head``; ``hopper_msa_proxy_score_decode`` passes 2-D views in that dtype.
# ``_admit_proxy_prefill`` admits fp8 e4m3 ``q`` / cache; ``hopper_msa_proxy_score_prefill`` passes them
# as is.  ``_admit_topk_select`` admits float32 ``max_score`` (passed as ``S=max_score.view(torch.uint32)``),
# int32 ``num_valid_pages`` (or the int32 filler of ``_int32_dummy``) and an int32 ``output``.  The table
# names every tensor the host passes; a plan may take a subset of them.
_I32, _U32, _U8 = ("kDLInt", 32), ("kDLUInt", 32), ("kDLUInt", 8)
_F32, _BF16, _E4M3 = ("kDLFloat", 32), ("kDLBfloat", 16), ("kDLFloat8_e4m3fn", 8)
_HOST_DTYPES = {
    "decode_v5": {
        "Q32": _U32,
        "K": _U8,
        "V": _U8,
        "O": _BF16,
        "q2k_indices": _I32,
        "page_table": _I32,
        "seqused_k": _I32,
        "part_o": _U32,
        "part_ml": _F32,
        "counters": _U32,
    },
    "prefill_v1": {
        "Q": _BF16,
        "Q32": _U32,
        "K": _E4M3,
        "V": _E4M3,
        "O": _BF16,
        "q2k_indices": _I32,
        "cu_seqlens_q": _I32,
        "page_table": _I32,
        "seqused_k": _I32,
        "q_offset": _I32,
    },
    "prefill_v2": {
        "Q": _BF16,
        "Q32": _U32,
        "K": _U8,
        "V": _U8,
        "O": _BF16,
        "q2k_indices": _I32,
        "cu_seqlens_q": _I32,
        "page_table": _I32,
        "seqused_k": _I32,
        "q_offset": _I32,
    },
    "proxy_decode_fp8": {
        "Q": _E4M3,
        "K": _E4M3,
        "out": _F32,
        "page_table": _I32,
        "seqused_k": _I32,
        "q_offset": _I32,
    },
    "proxy_decode_bf16": {
        "Q": _BF16,
        "K": _BF16,
        "out": _F32,
        "page_table": _I32,
        "seqused_k": _I32,
        "q_offset": _I32,
    },
    "proxy_prefill": {
        "Q": _E4M3,
        "K": _E4M3,
        "out": _F32,
        "cu_seqlens_q": _I32,
        "page_table": _I32,
        "seqused_k": _I32,
        "q_offset": _I32,
    },
    "topk_select": {"S": _U32, "nvp": _I32, "out": _I32},
}
_DTYPE_CHECK = re.compile(
    r'check_dtype\([^,]*, DLDataType\{(\w+), (\d+), (\d+)\}, "([^"]+)"\);'
)
# The three launch constants a shared binding takes from the record's compile flags, and the sites
# that must spend them (the kernel symbol at its declaration and the launch, the dynamic shared
# memory at the launch configuration, the block size at the launch configuration).
_BINDING_FLAGS = ("CAKE_PROGRAM_KERNEL", "CAKE_BIND_SMEM", "CAKE_BIND_THREADS")
_DEFINE = re.compile(r"^-D(CAKE_[A-Z0-9_]+)=(.+)$")


def _dtype_key(route: str) -> str:
    """The ``_HOST_DTYPES`` key of a route: its family, split on the dtype coordinate for proxy-score
    decode (``proxy_decode:<fp8|bf16>:...``)."""

    family = ch._route_family(route)
    if family == "proxy_decode":
        return f"proxy_decode_{route.split(':')[1]}"
    return family


def _grid_triple() -> list[tuple[str, str]]:
    """The grid entries every plan ends in, in the axis order the host indexes them by
    (``_GRID_AXES``: ``grid_x``, ``grid_y``, ``grid_z``)."""

    axes = sorted(ch._GRID_AXES, key=ch._GRID_AXES.__getitem__)
    return [("grid", axis) for axis in axes]


def _plan(name: str) -> list[tuple[str, str]]:
    return [
        (str(kind), str(argument)) for kind, argument in jm.MODULES[name]["arg_plan"]
    ]


def _defines(name: str) -> dict[str, str]:
    """The ``-D<macro>=<value>`` compile flags of a record, by macro (each defined once)."""

    defines: dict[str, str] = {}
    for flag in jm.MODULES[name]["compile_flags"]:
        match = _DEFINE.match(flag)
        if match:
            macro, value = match.groups()
            assert macro not in defines, f"{name}: {macro} defined twice"
            defines[macro] = value
    return defines


def _knobs(name: str) -> tuple[tuple[str, str], ...]:
    """The record's knob flags: its ``-D`` defines other than the three binding constants."""

    defines = _defines(name)
    return tuple(sorted(d for d in defines.items() if d[0] not in _BINDING_FLAGS))


def _is_name(node: ast.AST, identifier: str) -> bool:
    return isinstance(node, ast.Name) and node.id == identifier


def _bound_names(function: ast.FunctionDef) -> set[str]:
    """The argument names one host function binds into a program's plan: the keywords of the
    ``constants`` dict and of ``constants.update(...)``, the string keys assigned into
    ``constants[...]``, and the keywords of its ``launch(...)`` call."""

    names: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if _is_name(target, "constants") and isinstance(node.value, ast.Call):
                    names.update(k.arg for k in node.value.keywords if k.arg)
                if (
                    isinstance(target, ast.Subscript)
                    and _is_name(target.value, "constants")
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    names.add(target.slice.value)
        elif isinstance(node, ast.Call):
            func = node.func
            updates_constants = (
                isinstance(func, ast.Attribute)
                and _is_name(func.value, "constants")
                and func.attr == "update"
            )
            if updates_constants or _is_name(func, "launch"):
                names.update(k.arg for k in node.keywords if k.arg)
    return names


def _host_bound_names(function_names: tuple[str, ...]) -> set[str]:
    """Union of ``_bound_names`` over the named top-level functions of the live host module."""

    tree = ast.parse(inspect.getsource(ch))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    }
    missing = sorted(set(function_names) - set(functions))
    assert not missing, f"host functions not found: {missing}"
    names: set[str] = set()
    for function in functions.values():
        names |= _bound_names(function)
    return names


@pytest.mark.cpu_only
def test_every_program_serves_exactly_one_route():
    """``ROUTES`` names each registered program once; no program is unrouted; every record's plan is one
    of the named plans."""

    assert jm.MODULES
    assert sorted(jm.ROUTES.values()) == sorted(jm.MODULES)
    for route, name in jm.ROUTES.items():
        assert jm.route_program(route) == name
    plans = list(jm._ARG_PLANS.values())
    for name in _PROGRAMS:
        assert any(jm.MODULES[name]["arg_plan"] is plan for plan in plans), name


@pytest.mark.cpu_only
def test_every_route_family_has_a_host_binding_site():
    """The host's op-kind table and this file's site table name the same route families, and every
    delivered route belongs to one of them."""

    kinds = {family for families in ch._KIND_FAMILIES.values() for family in families}
    assert set(_HOST_SITES) == kinds
    assert {ch._route_family(route) for route in jm.ROUTES} <= kinds


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", _PROGRAMS)
def test_units_match_the_record(name):
    """Both units the record names exist under the unit directory; the binding has the generated form (its
    namespace, the runtime header, the two exported functions) and takes the kernel symbol, the dynamic
    shared memory and the block size from the record's flags; the kernel unit defines that symbol, either
    literally or through the same ``CAKE_PROGRAM_KERNEL`` flag."""

    record = jm.MODULES[name]
    kernel_path = _CSRC / record["kernel"]
    binding_path = _CSRC / record["binding"]
    assert kernel_path.is_file(), kernel_path
    assert binding_path.is_file(), binding_path
    assert kernel_path.parent == _CSRC and binding_path.parent == _CSRC
    kernel = kernel_path.read_text()
    binding = binding_path.read_text()
    symbol = f"kernel_{name}"
    defines = _defines(name)
    assert defines["CAKE_PROGRAM_KERNEL"] == symbol
    assert (
        defines["CAKE_BIND_SMEM"].isdigit() and defines["CAKE_BIND_THREADS"].isdigit()
    )
    for macro in _BINDING_FLAGS:
        assert f"#ifndef {macro}\n#error" in binding, (
            f"{record['binding']}: {macro} unguarded"
        )

    assert 'extern "C" __global__ void CAKE_PROGRAM_KERNEL(' in binding
    assert re.search(
        r"__global__[^;{]*?\b(CAKE_PROGRAM_KERNEL|" + re.escape(symbol) + r")\s*\(",
        kernel,
    )
    assert "namespace CAKE_PROGRAM_SHIM {" in binding
    assert "}  // namespace CAKE_PROGRAM_SHIM" in binding
    assert (
        "#define CAKE_PROGRAM_SHIM CAKE_CAT(cake_binding_, CAKE_PROGRAM_KERNEL)"
        in binding
    )
    assert '#include "cake_ffi_checks.h"' in binding
    assert "config.blockDim = dim3(CAKE_BIND_THREADS, 1, 1);" in binding
    assert "config.dynamicSmemBytes = CAKE_BIND_SMEM;" in binding
    assert "reinterpret_cast<const void*>(CAKE_PROGRAM_KERNEL)" in binding
    assert not re.search(r"cake_hopper_msa_[0-9a-f]{20}", binding), (
        "a shared binding names no program"
    )
    entry = record["ffi_entry"]
    assert f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({entry}, CAKE_PROGRAM_SHIM::Run);" in binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(build_cuda_version, " in binding


@pytest.mark.cpu_only
def test_shared_bindings_serve_one_route_family_each():
    """A binding source is shared only within a route family (its tensor-map geometry and argument plan
    are the family's), and the programs sharing it differ in their flag values; a kernel source is shared
    only within a family too, by programs whose knob flags differ, and a shared kernel takes its symbol
    from the ``CAKE_PROGRAM_KERNEL`` flag."""

    by_binding: dict[str, list[str]] = {}
    for name in _PROGRAMS:
        by_binding.setdefault(jm.MODULES[name]["binding"], []).append(name)
    for binding, names in by_binding.items():
        routes = [next(r for r, n in jm.ROUTES.items() if n == name) for name in names]
        assert len({ch._route_family(route) for route in routes}) == 1, binding
        plans = {id(jm.MODULES[name]["arg_plan"]) for name in names}
        assert len(plans) == 1, f"{binding}: shared by programs of different plans"
        flags = [tuple(sorted(_defines(name).items())) for name in names]
        assert len(set(flags)) == len(flags), (
            f"{binding}: two programs with the same flags"
        )
    by_kernel: dict[str, list[str]] = {}
    for name in _PROGRAMS:
        by_kernel.setdefault(jm.MODULES[name]["kernel"], []).append(name)
    for kernel, names in by_kernel.items():
        routes = [next(r for r, n in jm.ROUTES.items() if n == name) for name in names]
        assert len({ch._route_family(route) for route in routes}) == 1, kernel
        knobs = [_knobs(name) for name in names]
        assert len(set(knobs)) == len(knobs), (
            f"{kernel}: two programs with the same knobs"
        )
        if len(names) > 1:
            assert "void CAKE_PROGRAM_KERNEL(" in (_CSRC / kernel).read_text(), kernel


@pytest.mark.cpu_only
@pytest.mark.parametrize("plan_name", sorted(jm._ARG_PLANS))
def test_plan_ends_in_the_grid_triple(plan_name):
    """Each named argument plan ends in the three grid scalars, names each argument once, places no grid
    entry elsewhere and carries no ``trace`` argument."""

    plan = [(str(kind), str(argument)) for kind, argument in jm._ARG_PLANS[plan_name]]
    assert plan[-3:] == _grid_triple()
    assert all(kind != "grid" for kind, _argument in plan[:-3])
    names = [argument for _kind, argument in plan]
    assert len(names) == len(set(names))
    assert "trace" not in names


@pytest.mark.cpu_only
@pytest.mark.parametrize("family", sorted(_HOST_SITES))
def test_plan_names_are_arguments_the_host_binds(family):
    """Every non-grid name of a plan of this route family is an argument its launch template or
    public launcher binds; a name the host never supplies would raise at the first launch."""

    routes = sorted(route for route in jm.ROUTES if ch._route_family(route) == family)
    assert routes, f"no delivered route of family {family!r}"
    bound = _host_bound_names(_HOST_SITES[family])
    for route in routes:
        plan = _plan(jm.ROUTES[route])
        names = {argument for kind, argument in plan if kind != "grid"}
        missing = sorted(names - bound)
        assert not missing, f"{route}: {missing} not bound by {_HOST_SITES[family]}"


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", _PROGRAMS)
def test_binding_checks_the_dtypes_the_host_passes(name):
    """Every tensor of the plan is checked by the binding, once, for the DLPack dtype the host passes
    it in (code, bits, one lane); a disagreement is a ``TypeError`` at the program's first launch."""

    route = next(route for route, program in jm.ROUTES.items() if program == name)
    expected = _HOST_DTYPES[_dtype_key(route)]
    binding = (_CSRC / jm.MODULES[name]["binding"]).read_text()
    checks: dict[str, list[tuple[str, int, int]]] = {}
    for code, bits, lanes, argument in _DTYPE_CHECK.findall(binding):
        checks.setdefault(argument, []).append((code, int(bits), int(lanes)))
    for kind, argument in _plan(name):
        if kind != "buffer":
            continue
        assert argument in expected, f"{route}: no host dtype recorded for {argument!r}"
        assert len(checks.get(argument, [])) == 1, (
            f"{route}: {argument!r} checked {len(checks.get(argument, []))} times"
        )
        code, bits, lanes = checks[argument][0]
        assert (code, bits, lanes) == (*expected[argument], 1), f"{route}: {argument!r}"


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", _PROGRAMS)
def test_jit_key_is_the_content_digest(name):
    """The JIT / AOT key is the program name, the target and the first twenty hex digits of the
    SHA-256 over the record's kernel unit, its binding unit and the compact JSON of its compile flags
    and target, recomputed here from the file bytes."""

    record = jm.MODULES[name]
    digest = hashlib.sha256()
    digest.update((_CSRC / record["kernel"]).read_bytes())
    digest.update((_CSRC / record["binding"]).read_bytes())
    identity = {"options": list(record["compile_flags"]), "target": jm._ARCHES[0]}
    digest.update(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    expected = f"{name}_{jm._TARGET}_{digest.hexdigest()[:20]}"
    uri = jm.get_hopper_msa_uri(name)
    assert uri == expected
    assert re.fullmatch(rf"{re.escape(name)}_sm90a_[0-9a-f]{{20}}", uri)
