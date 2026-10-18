# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host geometry of the parametric Cake SM120 MegaMoE device source.

The device source ``cake_sm120_megamoe_kernel.cuh`` (``kernel_src/sm120/cake_mxfp4_megamoe/src/``)
is one CUDA source with compile-time parameters (EP world size, global expert count, top-k,
max rows per rank, hidden size, intermediate size, shared expert on/off, SwiGLU clamp,
return-codec overflow capacity, LSA copier count).  Every
derived quantity the host needs — the structural switches that select code paths, the
scratch and window extents, the dynamic shared memory, the kernel argument list and the
kernel's own guards — is recorded by the generator in
``cake_sm120_megamoe_geometry.json`` as an integer expression of those parameters.  This
module evaluates them with a small whitelisted expression interpreter (no ``eval``).

:func:`describe_kernel` turns one parameter set into the per-deployment description the
backend consumes (windows, GIN/LSA transport constants, and for each of the four
schedule instances its constants, flags, TMA boxes, SMEM and symbol).
"""

from __future__ import annotations

import ast
import functools
from dataclasses import dataclass
from typing import Any, Callable

MAX_ROWS_QUANTUM = 128
# The kernel's ``active_rows`` scalar packs this call's rows (low 16 bits) and the largest
# per-rank row count of the call (high 16 bits): a host ABI limit on the row capacity.
ACTIVE_ROWS_FIELD_MAX = 0xFFFF
# Supported region: the parameter domain the generator traced and verified (bucket discovery
# and compile grid).  The source's own guards cover the switch tuples and integer ranges;
# these bounds keep deployments inside the verified expert-count domain.
SUPPORTED_RANKS = (2, 32)
SUPPORTED_MAX_LOCAL_EXPERTS = 128
SUPPORTED_MAX_EXPERTS = 1024
INSTANCE_NAMES = ("m128_copier", "m64_copier", "m128_nocopier", "m64_nocopier")
_KERNEL_WINDOWS = (
    ("dispatch_header_out", "HEADER_WINDOW_BYTES"),
    ("dispatch_payload_out", "PAYLOAD_WINDOW_BYTES"),
    ("dispatch_header_inbox", "HEADER_WINDOW_BYTES"),
    ("dispatch_payload_inbox", "PAYLOAD_WINDOW_BYTES"),
    ("result_out", "RESULT_WINDOW_BYTES"),
    ("result_inbox", "RESULT_WINDOW_BYTES"),
    ("ack_out", "ACK_WINDOW_BYTES"),
    ("ack_inbox", "ACK_WINDOW_BYTES"),
)
_ABI_SCALARS = {"int": "c_int", "unsigned int": "c_uint"}
_PARAMETER_NAMES = frozenset({"gin_dev_comm"})


@dataclass(frozen=True)
class KernelParams:
    """Compile-time parameters of one device build (one deployment geometry)."""

    ranks: int
    experts: int
    top_k: int
    max_rows: int
    hidden: int
    intermediate: int
    shared_expert: bool
    swiglu_limit: int = 10
    # Raw-overflow capacity of the lossless BF16 return codec, in 64-value blocks per row.
    codec_ovf_blocks: int = 16
    # Copier CTAs of the copier instances (one per LSA destination of a rank); the
    # copier-free instances ignore it.
    lsa_copiers: int = 1

    @property
    def local_experts(self) -> int:
        return self.experts // self.ranks

    def device_parameters(self) -> tuple[int, ...]:
        """Values in the order of ``flashinfer.moe_ep.kernel_src.sm120.cake_mxfp4_megamoe.DEVICE_PARAMETERS``."""

        return (
            self.ranks,
            self.experts,
            self.top_k,
            self.max_rows,
            self.hidden,
            self.intermediate,
            int(self.shared_expert),
            self.swiglu_limit,
            self.codec_ovf_blocks,
            self.lsa_copiers,
        )

    def env(self) -> dict[str, int]:
        return {
            "MOE_RANKS": self.ranks,
            "MOE_EXPERTS": self.experts,
            "MOE_TOPK": self.top_k,
            "MOE_MAX_ROWS": self.max_rows,
            "MOE_HIDDEN": self.hidden,
            "MOE_INTERMEDIATE": self.intermediate,
            "MOE_SWIGLU_LIMIT": self.swiglu_limit,
            "MOE_CODEC_OVF_BLOCKS": self.codec_ovf_blocks,
            "MOE_LSA_COPIERS": self.lsa_copiers,
        }


def kernel_max_rows(max_tokens_per_rank: int) -> int:
    """Kernel row capacity for a workspace: ``max_tokens_per_rank`` rounded up to 128."""

    if max_tokens_per_rank <= 0:
        raise ValueError("max_tokens_per_rank must be positive")
    q = MAX_ROWS_QUANTUM
    return (max_tokens_per_rank + q - 1) // q * q


# ----------------------------------------------------------------- expressions


def _cdiv(a: int, b: int) -> int:
    """C integer division (truncates toward zero), as in the device source."""

    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def _cmod(a: int, b: int) -> int:
    return a - _cdiv(a, b) * b


_FUNCTIONS: dict[str, Callable[..., int]] = {
    "_cdiv": _cdiv,
    "_cmod": _cmod,
    "max": max,
    "min": min,
    "int": int,
}
_BINOPS: dict[type, Callable[[int, int], int]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
}
_COMPARE: dict[type, Callable[[int, int], bool]] = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}
Expr = Callable[[dict[str, int]], int]


def _compile(node: ast.AST) -> Expr:
    """Compile one whitelisted integer-expression node into a closure."""

    if isinstance(node, ast.Expression):
        return _compile(node.body)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        value = node.value
        return lambda env: value
    if isinstance(node, ast.Name):
        name = node.id
        return lambda env: env[name]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        op, lhs, rhs = _BINOPS[type(node.op)], _compile(node.left), _compile(node.right)
        return lambda env: op(lhs(env), rhs(env))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        inner = _compile(node.operand)
        return lambda env: int(not inner(env))
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        cmp, lhs, rhs = (
            _COMPARE[type(node.ops[0])],
            _compile(node.left),
            _compile(node.comparators[0]),
        )
        return lambda env: int(cmp(lhs(env), rhs(env)))
    if isinstance(node, ast.IfExp):
        test, then, other = (
            _compile(node.test),
            _compile(node.body),
            _compile(node.orelse),
        )
        return lambda env: then(env) if test(env) else other(env)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        fn, args = _FUNCTIONS[node.func.id], [_compile(a) for a in node.args]
        return lambda env: int(fn(*(a(env) for a in args)))
    raise ValueError(f"unsupported geometry expression node {ast.dump(node)[:120]}")


@functools.lru_cache(maxsize=None)
def compile_expression(text: str) -> Expr:
    return _compile(ast.parse(text, mode="eval"))


# --------------------------------------------------------------------- tables


@dataclass(frozen=True)
class _Geometry:
    params: tuple[str, ...]
    switches: tuple[str, ...]
    ladder: dict[str, Any]
    buckets: dict[tuple[int, ...], int]
    formulas: tuple[str, ...]
    abis: tuple[tuple[tuple[str, str], ...], ...]
    tables: tuple[dict[str, Any], ...]


def _from_json(data: dict[str, Any]) -> _Geometry:
    ladder: dict[str, Any] = {}
    for name, entry in data["switch_ladder"].items():
        if isinstance(entry, str):
            ladder[name] = entry
        else:
            ladder[name] = [
                (formula, {tuple(p) for p in prefixes}) for formula, prefixes in entry
            ]
    return _Geometry(
        params=tuple(data["params"]),
        switches=tuple(data["switches"]),
        ladder=ladder,
        buckets={tuple(key): int(index) for key, index in data["buckets"]},
        formulas=tuple(data["formulas"]),
        abis=tuple(tuple((str(t), str(n)) for t, n in abi) for abi in data["abis"]),
        tables=tuple(data["tables"]),
    )


@functools.cache
def _geometry() -> _Geometry:
    from ......kernel_src.sm120.cake_mxfp4_megamoe import read_geometry

    return _from_json(read_geometry())


def _table(g: _Geometry, index: int) -> dict[str, Any]:
    base = g.tables[0]
    if index == 0:
        return base
    delta = g.tables[index]
    consts = {
        k: v for k, v in base["consts"].items() if k not in delta["consts_absent"]
    }
    consts.update(delta["consts"])
    buffers = {
        k: v for k, v in base["buffers"].items() if k not in delta["buffers_absent"]
    }
    buffers.update(delta["buffers"])
    return {
        "consts": consts,
        "buffers": buffers,
        "guards": delta["guards"],
        "abi": delta["abi"],
    }


class GeometryError(ValueError):
    """The parameter set is outside the generated source's supported region."""


def check_supported_region(params: KernelParams) -> None:
    """Raise :class:`GeometryError` outside the generator's verified parameter domain."""

    lo, hi = SUPPORTED_RANKS
    if not lo <= params.ranks <= hi:
        raise GeometryError(f"EP world size {params.ranks} outside {lo}..{hi}")
    if params.experts % params.ranks:
        raise GeometryError(
            f"num_experts {params.experts} is not divisible by the EP world size {params.ranks}"
        )
    if params.local_experts > SUPPORTED_MAX_LOCAL_EXPERTS:
        raise GeometryError(
            f"{params.local_experts} experts per rank exceed the supported "
            f"{SUPPORTED_MAX_LOCAL_EXPERTS}"
        )
    if params.experts > SUPPORTED_MAX_EXPERTS:
        raise GeometryError(
            f"{params.experts} experts exceed the supported {SUPPORTED_MAX_EXPERTS}"
        )


def resolve(
    params: KernelParams, instance: int, *, geometry: Any = None
) -> dict[str, Any]:
    """Switches, constants, buffer extents, SMEM, windows and ABI of one instance.

    Raises :class:`GeometryError` naming the violated kernel guard (the same condition the
    device source checks with ``static_assert`` / ``#error``).
    """

    g = _geometry() if geometry is None else geometry
    check_supported_region(params)
    if params.max_rows > ACTIVE_ROWS_FIELD_MAX:
        raise GeometryError(
            f"max_rows {params.max_rows} exceeds the kernel ABI's 16-bit active_rows field "
            f"(at most {ACTIVE_ROWS_FIELD_MAX // MAX_ROWS_QUANTUM * MAX_ROWS_QUANTUM} rows per rank)"
        )
    env = params.env()
    axes = {
        "MOE_SHARED_EXPERT": int(params.shared_expert),
        "MOE_INSTANCE": int(instance),
    }
    for k, name in enumerate(g.switches):
        entry = g.ladder[name]
        if isinstance(entry, str):
            axes[name] = compile_expression(entry)(env)
            continue
        prefix = (
            axes["MOE_SHARED_EXPERT"],
            axes["MOE_INSTANCE"],
            *(axes[n] for n in g.switches[:k]),
        )
        for formula, prefixes in entry:
            if prefix in prefixes:
                axes[name] = compile_expression(formula)(env)
                break
        else:
            raise GeometryError(f"parameter region not covered (switch {name})")
    key = (
        axes["MOE_SHARED_EXPERT"],
        axes["MOE_INSTANCE"],
        *(axes[n] for n in g.switches),
    )
    if key not in g.buckets:
        raise GeometryError(
            "unsupported combination of structural switches "
            + ", ".join(f"{n}={axes[n]}" for n in g.switches)
        )
    table = _table(g, g.buckets[key])
    for guard in table["guards"]:
        if not compile_expression(g.formulas[guard])(env):
            raise GeometryError(
                f"parameters violate a kernel guard: {g.formulas[guard]}"
            )
    consts = {
        k: compile_expression(g.formulas[v])(env) for k, v in table["consts"].items()
    }
    buffers: dict[str, tuple[str, Any]] = {}
    for name, (dtype, dims) in table["buffers"].items():
        if isinstance(dims, list):
            buffers[name] = (
                dtype,
                tuple(compile_expression(g.formulas[d])(env) for d in dims),
            )
        else:
            buffers[name] = (dtype, dims)  # sized per call (source expression)
    return {
        "switches": {n: axes[n] for n in g.switches},
        "consts": consts,
        "buffers": buffers,
        "smem_bytes": consts["PRODUCTION_SMEM_BYTES"],
        "threads": consts["THREADS"],
        "abi": g.abis[table["abi"]],
    }


def _descriptors(c: dict[str, int]) -> dict[str, dict[str, Any]]:
    """TMA boxes of the kernel's twelve tensor maps (declared on the generator side as
    ``(MMA_TILE_K, TASK_M)`` / ``(MMA_TILE_K, MMA_TILE_N)`` / ``(TASK_M, 1)`` /
    ``(MMA_N_SUBTILES, 8)`` / ``(64, TASK_M)`` boxes)."""

    k, n, m = c["MMA_TILE_K"], c["MMA_TILE_N"], c["TASK_M"]
    swz, none = [4, 3, 3], [0, 0, 0]

    def d(dtype: str, box: list[int], swizzle: list[int], fp4: bool = False) -> dict:
        return {
            "dtype": dtype,
            "native_fp4": fp4,
            "box": box,
            "swizzle": swizzle,
            "l2_promotion": "none",
        }

    out = {}
    for g in ("W1", "W2"):
        out[f"{g}_A"] = d("u8", [k, m], swz)
        out[f"{g}_B"] = d("fp4_e2m1", [k, n], swz, fp4=True)
        out[f"{g}_B8"] = d("u8", [k, n], swz)
        out[f"{g}_SFA"] = d("u32", [m, 1], none)
        out[f"{g}_SFB"] = d("u32", [c["MMA_N_SUBTILES"], 8], none)
        out[f"{g}_D"] = d("bf16", [64, m], swz)
    return out


def abi_plan(abi: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    """Kernel argument plan (index, name, kind, ctypes ABI) from the ordered C signature."""

    args = []
    for index, (ctype, name) in enumerate(abi):
        if ctype in _ABI_SCALARS:
            kind, width = "parameter", _ABI_SCALARS[ctype]
        elif ctype == "ncclWindow_t" or name in _PARAMETER_NAMES:
            kind, width = "parameter", "ptr"
        elif "*" in ctype:
            kind, width = "buffer", "ptr"
        else:
            raise GeometryError(f"unsupported kernel argument type {ctype!r} ({name})")
        args.append(
            {"index": index, "name": name, "var": name, "kind": kind, "abi": width}
        )
    return {"args": args, "binding_names": sorted({a["name"] for a in args})}


@functools.lru_cache(maxsize=64)
def describe_kernel(params: KernelParams) -> dict[str, Any]:
    """Per-deployment description of the four schedule instances of one parameter set."""

    instances: dict[str, dict[str, Any]] = {}
    abi = None
    for index, name in enumerate(INSTANCE_NAMES):
        r = resolve(params, index)
        c, sw = r["consts"], r["switches"]
        copiers = c["PR_COPIER_CTAS_EFFECTIVE"] > 0
        if copiers != name.endswith("_copier") or c["TASK_M"] != (
            64 if name.startswith("m64") else 128
        ):
            raise GeometryError(f"instance {name}: unexpected schedule constants")
        plan = abi_plan(r["abi"])
        if abi is not None and [a["name"] for a in plan["args"]] != [
            a["name"] for a in abi["args"]
        ]:
            raise GeometryError(
                f"instance {name}: kernel ABI differs between instances"
            )
        abi = abi or plan
        instances[name] = {
            "instance": index,
            "symbol": f"cake_megamoe_{name}",
            "constants": c,
            "buffers": r["buffers"],
            "smem_bytes": int(r["smem_bytes"]),
            "task_m": c["TASK_M"],
            "copiers_active": copiers,
            "descriptors": _descriptors(c),
            "switches": sw,
            "flags": {
                "_COPIERS_ACTIVE": copiers,
                "_PCIE_M64": c["TASK_M"] == 64,
                "_RESIDENT_PEER_FOLD": bool(sw["MOE_SW_RESIDENT_PEER_FOLD"]),
                "_TASK_GATE_U64": bool(sw["MOE_SW_TASK_GATE_U64"]),
            },
        }
    assert abi is not None
    c0 = instances["m128_copier"]["constants"]
    for name, inst in instances.items():
        for k in (
            "HEADER_WINDOW_BYTES",
            "PAYLOAD_WINDOW_BYTES",
            "RESULT_WINDOW_BYTES",
            "ACK_WINDOW_BYTES",
            "LSA_CHANNELS",
            "LSA_SIGNAL_COUNT",
            "LSA_SIGNAL_STRIDE",
            "DISPATCH_SIGNAL_BASE",
            "PCIE_M64_MAX_ROWS",
            "THREADS",
        ):
            if inst["constants"][k] != c0[k]:
                raise GeometryError(f"session constant {k} differs in instance {name}")
    variant = "fp8shared" if params.shared_expert else "noshared"
    return {
        "key": f"{variant}_ep{params.ranks}_e{params.experts}_k{params.top_k}"
        f"_r{params.max_rows}_h{params.hidden}_i{params.intermediate}",
        "params": params,
        "variant": variant,
        "world_size": params.ranks,
        "local_experts": params.local_experts,
        "expert_slots": c0["EXPERT_SLOTS"],
        "max_rows": params.max_rows,
        "top_k": params.top_k,
        "hidden": params.hidden,
        "intermediate": params.intermediate,
        "threads": c0["THREADS"],
        "abi": abi,
        "windows": [(n, c0[k]) for n, k in _KERNEL_WINDOWS]
        + [("lsa_signals", c0["LSA_CHANNELS"] * c0["LSA_SIGNAL_STRIDE"] * 4)],
        "gin": {
            "context_count": 2,
            "signal_count": c0["LSA_SIGNAL_COUNT"],
            "world_barrier_count": 1,
            "strong_signals_required": True,
        },
        "transport": {
            "lsa_channels": c0["LSA_CHANNELS"],
            "lsa_signal_stride": c0["LSA_SIGNAL_STRIDE"],
            "dispatch_signal_base": c0["DISPATCH_SIGNAL_BASE"],
            "signal_count": c0["LSA_SIGNAL_COUNT"],
            "codec_row_bytes": c0["CODEC_ROW_BYTES"],
            "record_bytes": c0["RECORD_BYTES"],
            "dispatch_chunk_min_records": c0["DISPATCH_CHUNK_MIN_RECORDS"],
            "m64_max_rows": c0["PCIE_M64_MAX_ROWS"],
        },
        "instances": instances,
    }


def window_bytes(params: KernelParams) -> int:
    """Symmetric window memory per rank (the result windows dominate)."""

    return sum(b for _, b in describe_kernel(params)["windows"])


__all__ = [
    "ACTIVE_ROWS_FIELD_MAX",
    "GeometryError",
    "INSTANCE_NAMES",
    "KernelParams",
    "SUPPORTED_MAX_EXPERTS",
    "SUPPORTED_MAX_LOCAL_EXPERTS",
    "SUPPORTED_RANKS",
    "abi_plan",
    "check_supported_region",
    "compile_expression",
    "describe_kernel",
    "kernel_max_rows",
    "resolve",
    "window_bytes",
]
