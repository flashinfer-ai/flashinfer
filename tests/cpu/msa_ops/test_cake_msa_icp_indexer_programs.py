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

CPU tests of the Cake MiniMax-M3 ICP indexer (``flashinfer.msa_ops.msa_icp_*``
and the vendorable package ``flashinfer.experimental.cake_msa_icp_indexer``).

* Registry and build contract: every (role, local K rows, architecture)
  program the backend routes to is delivered for ``sm_100a``, ``sm_103a`` and
  ``sm_107a`` (a missing one fails here; nothing is skipped), the tensor-core
  roles run their own ``sm_107a`` programs, each record binds only the shared
  kernel-parameter vocabulary that the backend passes, and ``build.json``
  (schema ``cake-build/1``) holds exactly one entry per (program,
  architecture) whose digests match the bytes of its listed sources.
* Host checks: every entry point rejects invalid metadata with ``ValueError``
  before any launch, accepts the documented strided supersets, binds the
  layout scalars and strides the contract defines, and reads no tensor
  content (valid calls run on ``meta`` tensors).
* Sizing helpers against the MSA / vLLM formulas, the comm layout against the
  independent oracle helper, the H = 4 limit, the capability dispatch, and the
  oracle self-test with its golden vectors.

No CUDA device and no build: launches are intercepted at
``cake_jit.run_module``.  No digest literal and no program or file count is
pinned; the registry and the manifest are the inputs.
"""

from __future__ import annotations

import hashlib
import re
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

import flashinfer.msa_ops as msa_ops
from flashinfer.experimental.cake_msa_icp_indexer import cake_backend as cb
from flashinfer.experimental.cake_msa_icp_indexer import cake_indexer, cake_jit
from tests.test_helpers import cake_msa_icp_inputs as ref

FP8 = torch.float8_e4m3fn
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
PACKAGE = "flashinfer.experimental.cake_msa_icp_indexer"
LAYOUTS = tuple((p, w) for p in (128, 256) for w in (2, 4))
LAYOUT_ROWS = tuple(sorted({p // w for p, w in LAYOUTS}))
# Roles whose program depends on the local K rows per page (tensor-core scorers).
LAYOUT_ROLES = ("score", "decode_plane", "decode_topk")
REQUIRED_ROLES = (
    "plan",
    "score",
    "decode_plane",
    "select_prefill",
    "select_publish",
    "merge",
    "merge_legacy",
    "merge_prefill",
)
# The fused decode chain ships as one unit (DESIGN O-4): both roles or neither.
FUSED_ROLES = ("decode_topk", "topk16_publish")
SM_COUNT = 148
H = 4


def _routes() -> dict[tuple[str, int | None, int | None, str], str]:
    """``(role, rows, page, arch) -> record``; ``page`` ``None`` serves every page size."""
    cake_jit._routes.cache_clear()
    return cake_jit._routes()


def _keys_of(role: str) -> tuple[tuple[int | None, int | None], ...]:
    """``(rows, page)`` keys the backend routes ``role`` by: every layout's local
    K rows per page and its page size for the tensor-core roles."""
    if role in LAYOUT_ROLES:
        return tuple(sorted({(p // w, p) for p, w in LAYOUTS}))
    return ((None, None),)


def _served(
    routes: dict, role: str, rows: int | None, page: int | None, arch: str
) -> str | None:
    """The record ``route`` resolves: the page-specific one, else the page-independent one."""
    return routes.get((role, rows, page, arch)) or routes.get((role, rows, None, arch))


def _fp8_bytes_of(bound: torch.Tensor, given: torch.Tensor) -> bool:
    """``bound`` is the caller's FP8 tensor reinterpreted as bytes (the TMA
    descriptors read uint8), without a copy: same storage, offset, shape and strides."""
    return (
        bound.dtype == torch.uint8
        and given.element_size() == 1
        and bound.untyped_storage()._cdata == given.untyped_storage()._cdata
        and bound.storage_offset() == given.storage_offset()
        and bound.shape == given.shape
        and bound.stride() == given.stride()
    )


def _vocabulary() -> set[str]:
    return set(cake_jit.LOGICAL_TENSORS) | set(cake_jit.LOGICAL_SCALARS)


# --------------------------------------------------------------------------- launch interception


@dataclass
class Launch:
    name: str
    role: str
    rows: int | None
    values: dict[str, Any]
    grid_args: dict[str, int]
    arch: str
    enable_pdl: bool
    device_index: int | None = None


class Launches(list):
    def roles(self) -> list[str]:
        return [launch.role for launch in self]


def _intercept(monkeypatch, arch: str, *, real_registry: bool) -> Launches:
    """Run the backend on CPU / meta tensors: device lookups return a fixed
    device of ``arch`` with :data:`SM_COUNT` SMs, and launches are recorded
    instead of run.  With ``real_registry`` the exported records route and
    shape the launches; otherwise every route resolves to a stand-in name."""
    launches = Launches()
    monkeypatch.setattr(cb, "_device_index", lambda t: 0)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(cake_jit, "device_arch", lambda index: arch)
    monkeypatch.setattr(cake_jit, "sm_count", lambda index: SM_COUNT)
    route_of: dict[str, tuple[str, int | None]] = {}
    grids: dict[str, dict[str, int]] = {}
    if real_registry:
        real_route, real_grid = cake_jit.route, cake_jit.launch_grid
        for name, record in cake_jit.MODULES.items():
            route_of[name] = (record["role"], record["rows"])

        def route(role, rows, arch_, page=None):
            return real_route(role, rows, arch_, page=page)

        def launch_grid(name, **kw):
            grids[name] = kw
            return real_grid(name, **kw)
    else:

        def route(role, rows, arch_, page=None):
            name = f"{role}|{rows}|{arch_}"
            route_of[name] = (role, rows)
            return name

        def launch_grid(name, **kw):
            grids[name] = kw
            return (1, 1, 1)

        monkeypatch.setattr(cake_jit, "ctas_per_sm", lambda name: 1)
    monkeypatch.setattr(cake_jit, "route", route)
    monkeypatch.setattr(cake_jit, "launch_grid", launch_grid)

    def run_module(name, values, *, grid, arch, enable_pdl=False, device_index=None):
        role, rows = route_of[name]
        launches.append(
            Launch(
                name,
                role,
                rows,
                dict(values),
                dict(grids.get(name, {})),
                arch,
                enable_pdl,
                device_index,
            )
        )

    monkeypatch.setattr(cake_jit, "run_module", run_module)
    return launches


@pytest.fixture
def fake(monkeypatch) -> Launches:
    return _intercept(monkeypatch, "sm_103a", real_registry=False)


# --------------------------------------------------------------------------- argument builders (meta tensors)

META = torch.device("meta")


def _t(shape, dtype, device=META) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=device)


def _plan(*, total_q: int = 6, batch: int = 2, columns: int = 10, device=META) -> Any:
    i32 = lambda n: _t((n,), torch.int32, device)  # noqa: E731
    return cb.MSAIcpScorePlan(
        qo_segment_lens=i32(batch),
        kv_segment_lens=i32(batch),
        qo_segment_offsets=i32(batch + 1),
        kv_segment_offsets=i32(batch + 1),
        qo_offset=i32(batch),
        packed_work_range=_t((SM_COUNT,), torch.int64, device),
        packed_work_info=_t((1024,), torch.int64, device),
        workspace=_t((0,), torch.uint8, device),
        total_q=total_q,
        num_qo_heads=H,
        batch=batch,
        max_query_len=total_q,
        columns=columns,
    )


def _k(pages: int, rows: int, *, pitch: int | None = None, device=META) -> torch.Tensor:
    pitch = rows * 128 if pitch is None else pitch
    flat = _t((pages * pitch,), FP8, device)
    return flat.as_strided((pages, rows, 128), (pitch, 128, 1))


def score_args(
    page: int = 128, tp: int = 2, *, compact: bool = False, device=META
) -> dict:
    t, columns = 6, 10
    rows = page // tp
    k_cols = (columns + 1) // 2 if compact else columns
    return dict(
        q=_t((t, H, 128), FP8, device),
        k=_k(7, rows, device=device),
        block_table=_t(
            (2, (columns + page // 128 - 1) // (page // 128)), torch.int32, device
        ),
        plan=_plan(total_q=t, columns=columns, device=device),
        rank=tp - 1,
        max_score=_t((t, H, k_cols), torch.float32, device),
        tp=tp,
        compact=compact,
    )


def select_prefill_args(device=META) -> dict:
    t, k = 5, 40
    return dict(
        scores=_t((t, H, k), torch.float32, device),
        num_valid_pages=_t((t,), torch.int32, device),
        carrier=_t((H, t, 33), torch.int32, device),
        force_begin_blocks=1,
        force_end_blocks=2,
        index_scale=1,
        index_offset=0,
    )


def decode_args(
    page: int = 128, world: int = 2, *, dql: int = 2, batch: int = 3, device=META
) -> dict:
    t = batch * dql
    return dict(
        index_q=_t((t, H, 128), FP8, device),
        index_k=_k(9, page // world, device=device),
        block_table=_t((batch, 4), torch.int32, device),
        seq_lens=_t((batch,), torch.int32, device),
        score_out=_t((t, H, 16), torch.float32, device),
        rank=world - 1,
        world_size=world,
        page_size=page,
        decode_query_len=dql,
        max_decode_query_len=4,
    )


def publish_args(
    world: int = 2, *, tagged: bool = False, t: int = 5, cap: int = 8, device=META
) -> dict:
    hpr = H // world
    return dict(
        scores=_t((t, H, 24), torch.float32, device),
        num_valid_pages=_t((t,), torch.int32, device),
        recv_bases=_t((world,), torch.int64, device),
        errors=_t((1,), torch.int32, device),
        slot=2,
        rank=world - 1,
        world_size=world,
        heads_per_rank=hpr,
        capacity=cap,
        force_begin_blocks=0,
        force_end_blocks=1,
        generations=_t((3, world * hpr, cap), torch.int32, device) if tagged else None,
    )


def fused_args(
    page: int = 256,
    world: int = 4,
    *,
    tagged: bool = False,
    batch: int = 2,
    device=META,
) -> dict:
    dql = 4
    t = batch * dql
    cap = max(16, t)
    return dict(
        index_q=_t((t, H, 128), FP8, device),
        index_k=_k(9, page // world, device=device),
        block_table=_t((batch, 4), torch.int32, device),
        seq_lens=_t((batch,), torch.int32, device),
        num_valid_pages=_t((t,), torch.int32, device),
        pool=_t((cb.decode_pool_bytes(max_decode_tokens=t),), torch.uint8, device),
        recv_bases=_t((world,), torch.int64, device),
        errors=_t((1,), torch.int32, device),
        rank=world - 1,
        world_size=world,
        page_size=page,
        decode_query_len=dql,
        slot=1,
        capacity=cap,
        force_begin_blocks=0,
        force_end_blocks=1,
        generations=_t((3, H, cap), torch.int32, device) if tagged else None,
    )


def merge_args(
    world: int = 2,
    *,
    tagged: bool = False,
    prefill: bool = False,
    t: int = 5,
    cap: int = 8,
    device=META,
) -> dict:
    hpr = H // world
    if tagged:
        recv = _t((3, world, hpr, cap, 32), torch.int64, device)[1]
    else:
        recv = _t((3, world, hpr, cap, 33), torch.int32, device)[1]
    return dict(
        recv_slot=recv,
        num_valid_pages=_t((t,), torch.int32, device),
        output=_t((t, hpr, 16), torch.int32, device),
        errors=_t((1,), torch.int32, device),
        init_blocks=0,
        local_blocks=1,
        expected_generations=_t((3, hpr, cap), torch.int32, device)[1]
        if tagged
        else None,
        prefill=prefill,
    )


def plan_device_args(batch: int = 3, total_q: int = 300, device=META) -> dict:
    i32 = lambda n: _t((n,), torch.int32, device)  # noqa: E731
    storage = ref.plan_storage(H, total_q, batch)
    return dict(
        qo_segment_lens=i32(batch),
        kv_segment_lens=i32(batch),
        qo_segment_offsets=i32(batch + 1),
        kv_segment_offsets=i32(batch + 1),
        qo_offset=i32(batch),
        num_qo_heads=H,
        total_q=total_q,
        max_query_len=200,
        max_seq_len=5000,
        packed_work_range=_t((SM_COUNT,), torch.int64, device),
        packed_work_info=_t((storage["work_info_numel_min"],), torch.int64, device),
        workspace=_t((storage["workspace_bytes_min"],), torch.uint8, device),
    )


def _call(entry: Callable, args: dict) -> None:
    args = dict(args)
    positional = {
        msa_ops.msa_icp_index_score_sharded: ("q", "k", "block_table", "plan"),
        msa_ops.msa_icp_topk_select_prefill: ("scores", "num_valid_pages", "carrier"),
        msa_ops.msa_icp_decode_index_score_sharded: (
            "index_q",
            "index_k",
            "block_table",
            "seq_lens",
            "score_out",
        ),
        msa_ops.msa_icp_topk_select_publish: (
            "scores",
            "num_valid_pages",
            "recv_bases",
            "errors",
        ),
        msa_ops.msa_icp_decode_score_publish: (
            "index_q",
            "index_k",
            "block_table",
            "seq_lens",
            "num_valid_pages",
            "pool",
            "recv_bases",
            "errors",
        ),
        msa_ops.msa_icp_merge_candidates: (
            "recv_slot",
            "num_valid_pages",
            "output",
            "errors",
        ),
        msa_ops.msa_icp_index_score_sharded_plan_device: (
            "qo_segment_lens",
            "kv_segment_lens",
            "qo_segment_offsets",
            "kv_segment_offsets",
            "qo_offset",
            "num_qo_heads",
        ),
    }[entry]
    entry(*[args.pop(name) for name in positional], **args)


def _strided(
    shape, dtype, *, last_stride: int = 1, row_pad: int = 0, device=META
) -> torch.Tensor:
    """A ``shape`` view with ``last_stride`` and ``row_pad`` extra elements per
    innermost row (a slice of a wider buffer)."""
    inner = shape[-1] * last_stride + row_pad
    full = (*shape[:-1], inner)
    base = _t(full, dtype, device)
    if last_stride == 1:
        return base[..., : shape[-1]]
    return base[..., : shape[-1] * last_stride : last_stride]


# --------------------------------------------------------------------------- registry


def test_architecture_table_names_exact_targets() -> None:
    assert tuple(cake_jit.ARCHES) == ARCHES
    assert cake_jit.ARCH_BY_CAPABILITY == {
        (10, 0): "sm_100a",
        (10, 3): "sm_103a",
        (10, 7): "sm_107a",
    }
    for capability in ((9, 0), (10, 1), (10, 8), (11, 0), (12, 0), (12, 1)):
        assert cake_jit.arch_for_capability(capability) is None, capability


@pytest.mark.parametrize("arch", ARCHES)
def test_every_routed_program_is_delivered(arch: str) -> None:
    """Every (role, local K rows) the backend can route to resolves on ``arch``;
    a missing program fails (the GPU tests would raise NotImplementedError)."""
    routes = _routes()
    missing = [
        (role, rows, page)
        for role in REQUIRED_ROLES
        for rows, page in _keys_of(role)
        if _served(routes, role, rows, page, arch) is None
    ]
    assert not missing, f"no {arch} program for {missing}"
    for role in REQUIRED_ROLES:
        for rows, page in _keys_of(role):
            name = _served(routes, role, rows, page, arch)
            assert cake_jit.route(role, rows, arch, page=page) == name
            assert cake_jit.has_role(role, arch)


def test_fused_decode_programs_are_all_or_nothing() -> None:
    routes = _routes()
    want = {
        (role, rows, page, arch)
        for role in FUSED_ROLES
        for rows, page in _keys_of(role)
        for arch in ARCHES
    }
    have = {key for key in want if _served(routes, *key) is not None}
    assert have in (set(), want), (
        f"partial fused decode path: missing {sorted(want - have, key=str)}"
    )


def test_route_names_a_missing_program() -> None:
    with pytest.raises(NotImplementedError, match="decode_plane.*48 K rows.*sm_107a"):
        cake_jit.route("decode_plane", 48, "sm_107a")
    with pytest.raises(NotImplementedError, match="merge.*sm_90a"):
        cake_jit.route("merge", None, "sm_90a")


def test_records_are_well_formed() -> None:
    assert cake_jit.MODULES, "the package registers no exported program"
    vocabulary = _vocabulary()
    grid_kind = {
        "score": ("per_sm",),
        "decode_plane": ("request_splits",),
        "decode_topk": ("request_splits",),
        "select_prefill": ("tokens_per_cta_by_head",),
        "select_publish": ("tokens_per_cta_by_head",),
        "topk16_publish": ("tokens_per_cta_by_head",),
        "merge": ("tokens_per_cta_by_head",),
        "merge_legacy": ("tokens_per_cta_by_head",),
        "merge_prefill": ("tokens_per_cta_by_head",),
        "plan": cake_jit.GRID_RULES,
    }
    for name, record in cake_jit.MODULES.items():
        assert re.fullmatch(r"[a-z][a-z0-9_]*", name), name
        assert re.fullmatch(rf"{cake_jit.PREFIX}_[0-9a-f]{{20}}", record["library"]), (
            name
        )
        role = record["role"]
        assert role in cake_jit.ROLES, name
        arches = tuple(record["arches"])
        assert (
            arches and len(set(arches)) == len(arches) and set(arches) <= set(ARCHES)
        ), name
        if role in LAYOUT_ROLES:
            assert record["rows"] in LAYOUT_ROWS, name
            assert record["page"] in (None, *{p for p, _ in LAYOUTS}), name
        else:
            assert record["rows"] is None and record["page"] is None, name
        rule = record["grid_rule"]
        assert rule["kind"] in grid_kind[role], (name, rule)
        if rule["kind"] == "request_splits":
            assert int(rule["ctas_per_sm"]) >= 1, name
        if rule["kind"] == "tokens_per_cta_by_head":
            assert int(rule["rows"]) >= 1, name
        assert isinstance(record["ffi_entry"], str) and record["ffi_entry"], name
        # The exporter binds FlashInfer arguments by the kernels' own parameter
        # names: the logical map is the identity on the shared vocabulary.
        logical = record["logical"]
        assert all(k == v for k, v in logical.items()), name
        assert set(logical.values()) <= vocabulary, (
            name,
            set(logical.values()) - vocabulary,
        )
        plan = [tuple(entry) for entry in record["arg_plan"]]
        kinds = [kind for kind, _ in plan]
        assert [key[-1] for kind, key in plan if kind == "grid"] == ["x", "y", "z"], (
            name
        )
        assert kinds[-3:] == ["grid", "grid", "grid"], name
        runtime_pdl = record["launch"].get("use_pdl") == "runtime"
        assert kinds.count("launch") == int(runtime_pdl), name
        if runtime_pdl:
            assert kinds[-4] == "launch", name
        if role == "plan":
            assert not runtime_pdl, f"{name}: the device plan launches without PDL"
        values = [key for kind, key in plan if kind not in ("grid", "launch")]
        assert len(values) == len(set(values)) and set(values) == set(logical), name


def test_routes_are_unique_and_layout_roles_split_by_rows() -> None:
    routes = _routes()  # raises RuntimeError if two records serve one key
    assert routes, "the package registers no exported program"
    for (role, rows, page, arch), name in routes.items():
        record = cake_jit.MODULES[name]
        assert (record["role"], record["rows"], record["page"]) == (role, rows, page)
        assert arch in record["arches"], name
        if page is not None:
            # A page-specific program serves a layout of that page size.
            assert role in LAYOUT_ROLES and (page, page // rows) in LAYOUTS, name


@pytest.mark.parametrize("role", ("score", "decode_plane"))
def test_sm107a_tensor_core_roles_run_their_own_programs(role: str) -> None:
    """VR200 never runs an sm_100a / sm_103a schedule: each tensor-core program
    serving sm_107a is built for sm_107a alone (fused scorer included when it ships)."""
    routes = _routes()
    roles = [role] + (
        ["decode_topk"]
        if role == "decode_plane"
        and any(key[0] == "decode_topk" and key[3] == "sm_107a" for key in routes)
        else []
    )
    for r in roles:
        for rows, page in _keys_of(r):
            native = _served(routes, r, rows, page, "sm_107a")
            assert native is not None, (
                f"no sm_107a program for {r} at {rows} K rows per page (page {page})"
            )
            assert tuple(cake_jit.MODULES[native]["arches"]) == ("sm_107a",), native
            for other in ("sm_100a", "sm_103a"):
                assert _served(routes, r, rows, page, other) not in (None, native), (
                    r,
                    rows,
                    page,
                    other,
                )


def _manifest() -> tuple[Any, dict]:
    from flashinfer.experimental.cake_msa_icp_indexer import cake_build

    return cake_build, cake_build.manifest()


def test_build_manifest_has_one_entry_per_program_and_architecture() -> None:
    cake_build, manifest = _manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == PACKAGE
    flags = manifest["arch_flags"]
    assert set(flags) == set(ARCHES)
    for arch, arch_flags in flags.items():
        num = arch[len("sm_") :]
        assert arch_flags == [f"-gencode=arch=compute_{num},code={arch}"], arch
    libraries = cake_build.libraries_by_name()
    names = [entry["name"] for entry in manifest["libraries"]]
    assert len(names) == len(set(names)) == len(libraries)
    package_dir = Path(cake_build.__file__).parent
    referenced = set()
    for name, record in cake_jit.MODULES.items():
        assert not {
            "sources",
            "source_sha256",
            "compile_flags",
            "sha256",
            "nvcc_flags",
        } & set(record), name
        for arch in record["arches"]:
            library = cake_jit.library_name(name, arch)
            referenced.add(library)
            entry = libraries[library]
            assert entry["arch"] == arch and entry["gencode"] == flags[arch][0], library
            assert entry["entry"] == record["ffi_entry"], library
            assert len(entry["sources"]) == 2 and set(entry["sha256"]) == set(
                entry["sources"]
            ), library
            for source in entry["sources"]:
                path = package_dir / source
                assert source.startswith("csrc/") and path.is_file(), source
                data = path.read_bytes()
                assert hashlib.sha256(data).hexdigest() == entry["sha256"][source], (
                    source
                )
                assert b"by FlashInfer team" in b"\n".join(data.splitlines()[:3]), (
                    source
                )
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"]), library
            nvcc = entry["nvcc_flags"]
            assert not {"-use_fast_math", "--use_fast_math"} & set(nvcc), library
    assert referenced == set(libraries), sorted(set(libraries) ^ referenced)


def test_kernel_units_define_the_symbols_their_bindings_launch() -> None:
    cake_build, manifest = _manifest()
    libraries = cake_build.libraries_by_name()
    package_dir = Path(cake_build.__file__).parent
    for name, record in cake_jit.MODULES.items():
        for arch in record["arches"]:
            entry = libraries[cake_jit.library_name(name, arch)]
            kernel, binding = (package_dir / s for s in entry["sources"])
            assert kernel.name.endswith("_kernel.cu") and binding.name.endswith(
                "_binding.cu"
            ), entry["sources"]
            ktext, btext = kernel.read_text(), binding.read_text()
            symbol = record.get("kernel_symbol")
            assert symbol, name
            assert re.search(rf"__global__[^;{{]*\b{re.escape(symbol)}\s*\(", ktext), (
                name,
                symbol,
            )
            assert re.search(rf"\b{re.escape(symbol)}\b", btext), (name, symbol)
            assert re.search(
                rf"TVM_FFI_DLL_EXPORT_TYPED_FUNC\(\s*{re.escape(record['ffi_entry'])}\b",
                btext,
            ), name


def test_package_supersedes_the_top_level_sources() -> None:
    import flashinfer

    repo = Path(flashinfer.__file__).resolve().parent.parent
    if (repo / "csrc").is_dir():
        assert not (repo / "csrc" / "cake_msa_icp_indexer").exists(), (
            "csrc/cake_msa_icp_indexer is superseded by the package"
        )


@pytest.mark.parametrize("arch", ARCHES)
def test_backend_binds_every_kernel_parameter(monkeypatch, arch: str) -> None:
    """Drive one valid call per role through the exported registry and check
    that each launched record receives a value for every kernel parameter."""
    launches = _intercept(monkeypatch, arch, real_registry=True)
    vocabulary = _vocabulary()
    for page, tp in LAYOUTS:
        _call(
            msa_ops.msa_icp_index_score_sharded,
            score_args(page, tp, compact=page == 256),
        )
        _call(msa_ops.msa_icp_decode_index_score_sharded, decode_args(page, tp))
        if cake_jit.has_role("decode_topk", arch):
            for tagged in (False, True):
                _call(
                    msa_ops.msa_icp_decode_score_publish,
                    fused_args(page, tp, tagged=tagged),
                )
    _call(msa_ops.msa_icp_index_score_sharded_plan_device, plan_device_args())
    _call(msa_ops.msa_icp_topk_select_prefill, select_prefill_args())
    for tp in (2, 4):
        for tagged in (False, True):
            _call(msa_ops.msa_icp_topk_select_publish, publish_args(tp, tagged=tagged))
            _call(msa_ops.msa_icp_merge_candidates, merge_args(tp, tagged=tagged))
        _call(msa_ops.msa_icp_merge_candidates, merge_args(tp, prefill=True))
        # legacy decode merges of 8+ tokens take the one-token-per-CTA program
        _call(msa_ops.msa_icp_merge_candidates, merge_args(tp, t=9, cap=9))
    launched = {launch.name for launch in launches}
    want = {
        name for name, record in cake_jit.MODULES.items() if arch in record["arches"]
    }
    assert launched == want, f"records never launched: {sorted(want - launched)}"
    for launch in launches:
        record = cake_jit.MODULES[launch.name]
        params = set(record["logical"].values())
        assert params <= set(launch.values), (launch.name, params - set(launch.values))
        assert set(launch.values) <= vocabulary, set(launch.values) - vocabulary
        grid = cake_jit.launch_grid(launch.name, **launch.grid_args)
        assert all(isinstance(g, int) and g >= 1 for g in grid), (launch.name, grid)


# --------------------------------------------------------------------------- host checks


def test_other_backends_are_rejected_before_any_cuda_call(monkeypatch) -> None:
    def boom(*a, **k):
        raise AssertionError("reached the device")

    for attr in ("device_arch", "run_module", "load", "sm_count"):
        monkeypatch.setattr(cake_jit, attr, boom)
    calls = [
        lambda: _call(
            msa_ops.msa_icp_index_score_sharded, {**score_args(), "backend": "cuda"}
        ),
        lambda: _call(
            msa_ops.msa_icp_topk_select_prefill,
            {**select_prefill_args(), "backend": "msa"},
        ),
        lambda: _call(
            msa_ops.msa_icp_decode_index_score_sharded,
            {**decode_args(), "backend": "vllm"},
        ),
        lambda: _call(
            msa_ops.msa_icp_topk_select_publish, {**publish_args(), "backend": "auto"}
        ),
        lambda: _call(
            msa_ops.msa_icp_decode_score_publish, {**fused_args(), "backend": "triton"}
        ),
        lambda: _call(
            msa_ops.msa_icp_merge_candidates, {**merge_args(), "backend": "cuda"}
        ),
        lambda: _call(
            msa_ops.msa_icp_index_score_sharded_plan_device,
            {**plan_device_args(), "backend": "cuda"},
        ),
        lambda: msa_ops.msa_icp_index_score_sharded_plan(
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            H,
            qo_offset=torch.zeros(1, dtype=torch.int32),
            backend="cuda",
        ),
        lambda: msa_ops.prepare_msa_icp_indexer(**_facade_kwargs(), backend="cuda"),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="backend='cake'"):
            call()


def _expect_rejected(
    fake: Launches, entry: Callable, args: dict, match: str | None = None
) -> None:
    before = len(fake)
    with pytest.raises(ValueError, match=match):
        _call(entry, args)
    assert len(fake) == before, "an invalid call launched a program"


def _with(args: dict, **changes) -> dict:
    return {**args, **changes}


def test_score_rejects_invalid_metadata(fake) -> None:
    e = msa_ops.msa_icp_index_score_sharded
    a = score_args(128, 2)
    t = a["q"].shape[0]
    _expect_rejected(fake, e, _with(a, q=_t((t, 8, 128), FP8)[:, ::2]), "contiguous")
    _expect_rejected(fake, e, _with(a, q=_t((t, H, 128), torch.bfloat16)))
    _expect_rejected(fake, e, _with(a, q=_t((t, 8, 128), FP8)), "4 index heads")
    _expect_rejected(fake, e, _with(a, q=_t((t + 1, H, 128), FP8)), "plan")
    _expect_rejected(fake, e, _with(a, plan="not a plan"))
    for scale in (float("nan"), float("inf")):
        _expect_rejected(fake, e, _with(a, sm_scale=scale), "sm_scale")
    _expect_rejected(fake, e, _with(a, tp=3))
    _expect_rejected(fake, e, _with(a, rank=2))
    _expect_rejected(fake, e, _with(a, k=_k(7, 48)), "rows per page")
    _expect_rejected(
        fake, e, _with(a, k=_t((7, 64, 256), FP8)[..., :128]), "dense rows"
    )
    _expect_rejected(
        fake, e, _with(a, k=_k(7, 64, pitch=64 * 128 + 8)), "multiple of 16"
    )
    cpu = score_args(128, 2, device="cpu")
    misaligned = torch.empty(7 * 64 * 128 + 16, dtype=FP8)[1:].as_strided(
        (7, 64, 128), (64 * 128, 128, 1)
    )
    _expect_rejected(fake, e, _with(cpu, k=misaligned), "16-byte aligned")
    _expect_rejected(fake, e, _with(a, compact=True), "P256")
    _expect_rejected(
        fake, e, _with(a, max_score=_t((t, H, 9), torch.float32)), "columns"
    )
    _expect_rejected(
        fake, e, _with(a, max_score=_strided((t, H, 10), torch.float32, last_stride=2))
    )
    _expect_rejected(fake, e, _with(a, max_score=_t((t, H, 10), torch.float16)))
    _expect_rejected(
        fake,
        e,
        _with(
            a,
            max_score=_t((t, H, 10), torch.float32).as_strided((t, H, 10), (40, 5, 1)),
        ),
    )
    _expect_rejected(
        fake, e, _with(a, block_table=_t((2, 10), torch.int64)), "block_table"
    )
    _expect_rejected(
        fake, e, _with(a, block_table=_t((3, 10), torch.int32)), "block_table"
    )
    _expect_rejected(
        fake, e, _with(a, block_table=_t((2, 9), torch.int32)), "block_table"
    )
    _expect_rejected(
        fake,
        e,
        _with(a, block_table=_strided((2, 10), torch.int32, last_stride=2)),
        "block_table",
    )
    c = score_args(256, 4, compact=True)
    _expect_rejected(
        fake, e, _with(c, max_score=_t((t, H, 4), torch.float32)), "columns"
    )
    _expect_rejected(
        fake, e, _with(c, block_table=_t((2, 4), torch.int32)), "block_table"
    )
    assert not fake


@pytest.mark.parametrize("page,tp", LAYOUTS)
@pytest.mark.parametrize("compact", (False, True))
def test_score_binds_the_layout(fake, page: int, tp: int, compact: bool) -> None:
    if compact and page == 128:
        pytest.raises(
            ValueError,
            _call,
            msa_ops.msa_icp_index_score_sharded,
            score_args(page, tp, compact=True),
        )
        return
    a = score_args(page, tp, compact=compact)
    t, k_cols = a["q"].shape[0], a["max_score"].shape[2]
    # strided superset: a token / column slice of a wider score buffer
    a["max_score"] = _t((t + 3, H, k_cols + 16), torch.float32)[:t, :, :k_cols]
    _call(msa_ops.msa_icp_index_score_sharded, a)
    (launch,) = fake
    v = launch.values
    rank = a["rank"]
    assert (launch.role, launch.rows) == ("score", page // tp)
    s, o = (2, rank * 2 // tp) if compact else (1, 0)
    assert (v["index_scale"], v["index_offset"], v["compact"]) == (s, o, int(compact))
    assert (v["rank"], v["world"], v["page_size"]) == (rank, tp, page)
    assert (v["columns"], v["score_cols"], v["num_heads"], v["total_q"]) == (
        10,
        k_cols,
        H,
        t,
    )
    assert (v["score_stride_t"], v["score_stride_h"]) == (
        H * (k_cols + 16),
        k_cols + 16,
    )
    assert v["bt_stride"] == a["block_table"].stride(0)
    assert v["max_score"] is a["max_score"]
    assert _fp8_bytes_of(v["q"], a["q"]) and _fp8_bytes_of(v["k"], a["k"])
    assert launch.grid_args == {"sm_count": SM_COUNT}
    assert launch.enable_pdl is False


def test_score_initialize_fills_the_rows_first(fake) -> None:
    a = score_args(256, 2, compact=True, device="cpu")
    a["max_score"].fill_(7.0)
    _call(msa_ops.msa_icp_index_score_sharded, _with(a, initialize=False))
    assert bool((a["max_score"] == 7.0).all())
    _call(msa_ops.msa_icp_index_score_sharded, a)
    assert bool(torch.isneginf(a["max_score"]).all())
    assert len(fake) == 2


def test_plan_device_rejects_invalid_metadata(fake) -> None:
    e = msa_ops.msa_icp_index_score_sharded_plan_device
    a = plan_device_args()
    # the batch is qo_segment_lens' length: a longer one mismatches every other vector
    _expect_rejected(
        fake, e, _with(a, qo_segment_lens=_t((4,), torch.int32)), "segment"
    )
    for name in ("qo_segment_lens", "kv_segment_lens", "qo_offset"):
        if name != "qo_segment_lens":
            _expect_rejected(fake, e, _with(a, **{name: _t((4,), torch.int32)}), name)
        _expect_rejected(fake, e, _with(a, **{name: _t((3,), torch.int64)}), name)
    for name in ("qo_segment_offsets", "kv_segment_offsets"):
        _expect_rejected(fake, e, _with(a, **{name: _t((3,), torch.int32)}), name)
    _expect_rejected(fake, e, _with(a, num_qo_heads=8), "4 index heads")
    _expect_rejected(fake, e, _with(a, max_query_len=301))
    _expect_rejected(fake, e, _with(a, max_query_len=0))
    _expect_rejected(fake, e, _with(a, max_seq_len=0))
    _expect_rejected(
        fake,
        e,
        _with(a, packed_work_range=_t((SM_COUNT - 1,), torch.int64)),
        "packed_work_range",
    )
    _expect_rejected(
        fake,
        e,
        _with(a, packed_work_range=_t((SM_COUNT,), torch.int32)),
        "packed_work_range",
    )
    need = a["packed_work_info"].numel()
    _expect_rejected(
        fake,
        e,
        _with(a, packed_work_info=_t((need - 1,), torch.int64)),
        "packed_work_info",
    )
    _expect_rejected(
        fake, e, _with(a, workspace=_t(((32 << 20) - 1,), torch.uint8)), "32 MiB"
    )
    _expect_rejected(fake, e, _with(a, workspace=_t((32 << 20,), torch.int8)), "32 MiB")
    assert not fake


def test_plan_device_binds_the_plan_storage(fake) -> None:
    a = plan_device_args(batch=3, total_q=300)
    plan = msa_ops.msa_icp_index_score_sharded_plan_device(
        *[
            a.pop(n)
            for n in (
                "qo_segment_lens",
                "kv_segment_lens",
                "qo_segment_offsets",
                "kv_segment_offsets",
                "qo_offset",
                "num_qo_heads",
            )
        ],
        **a,
    )
    (launch,) = fake
    assert launch.role == "plan" and launch.enable_pdl is False
    v = launch.values
    assert (v["batch"], v["num_heads"], v["num_ctas"], v["total_q"]) == (
        3,
        H,
        SM_COUNT,
        300,
    )
    assert v["columns"] == plan.columns == (5000 + 127) // 128
    assert (
        v["work_range"] is plan.packed_work_range
        and v["work_info"] is plan.packed_work_info
    )
    assert (plan.total_q, plan.batch, plan.num_qo_heads, plan.max_query_len) == (
        300,
        3,
        H,
        200,
    )


def test_host_plan_derives_the_offsets(fake) -> None:
    lens = torch.tensor([0, 65, 0, 3], dtype=torch.int32)
    kv = torch.tensor([127, 259, 0, 130], dtype=torch.int32)
    qo = kv - lens
    plan = msa_ops.msa_icp_index_score_sharded_plan(
        lens, kv, H, qo_offset=qo, device="cpu"
    )
    assert plan.qo_segment_offsets.tolist() == [0, 0, 65, 65, 68]
    assert plan.kv_segment_offsets.tolist() == [0, 127, 386, 386, 516]
    assert (plan.total_q, plan.batch, plan.max_query_len, plan.columns) == (
        68,
        4,
        65,
        3,
    )
    assert plan.packed_work_range.numel() == SM_COUNT
    assert (
        plan.packed_work_info.numel()
        == ref.plan_storage(H, 68, 4)["work_info_numel_min"]
    )
    assert plan.workspace.numel() == ref.PLAN_WORKSPACE_BYTES
    assert fake.roles() == ["plan"]
    with pytest.raises(ValueError):
        msa_ops.msa_icp_index_score_sharded_plan(
            lens.long(), kv, H, qo_offset=qo, device="cpu"
        )
    with pytest.raises(ValueError):
        msa_ops.msa_icp_index_score_sharded_plan(
            lens, kv[:3], H, qo_offset=qo, device="cpu"
        )


def test_host_plan_rejects_negative_or_overflowing_lengths(fake) -> None:
    """The eager plan sizes its storage from the lengths' sum: a negative
    length (a smaller sum than the items the plan writes) or a sum past int32
    (a wrapped cumsum) must be rejected before any allocation or launch."""
    e = msa_ops.msa_icp_index_score_sharded_plan
    kv = torch.tensor([6400, 6400, 6400, 6400], dtype=torch.int32)
    for lens in ([6400, 6400, 6400, -12800], [0, 0, 0, -1]):
        q = torch.tensor(lens, dtype=torch.int32)
        with pytest.raises(ValueError, match="non-negative"):
            e(q, kv, H, qo_offset=torch.zeros(4, dtype=torch.int32), device="cpu")
    with pytest.raises(ValueError, match="non-negative"):
        e(
            torch.ones(4, dtype=torch.int32),
            kv,
            H,
            qo_offset=torch.tensor([0, -1, 0, 0], dtype=torch.int32),
            device="cpu",
        )
    big = torch.full((5,), 1 << 30, dtype=torch.int32)
    with pytest.raises(ValueError, match="2\\*\\*31"):
        e(big, big, H, qo_offset=torch.zeros(5, dtype=torch.int32), device="cpu")
    assert not fake


def test_plan_rejects_windows_past_the_packed_fields(fake) -> None:
    """The plan packs the segment and item indices and the columns per item in
    16 bits and the column total in int32 (work_info / work_range)."""
    e = msa_ops.msa_icp_index_score_sharded_plan_device
    a = plan_device_args()
    batch = a["qo_segment_lens"].numel()
    # Up to (total_q + 63) // 64 + batch - 1 items: 65535 is the most the plan packs.
    ok = (65535 - batch + 1) * cb.PLAN_ITEM_TOKENS
    cb._plan_limits(batch, ok, 128)
    with pytest.raises(ValueError, match="65535 query items"):
        cb._plan_limits(batch, ok + 1, 128)
    with pytest.raises(ValueError, match="65536 segments"):
        cb._plan_limits(65537, 65537, 128)
    cb._plan_limits(1, 64, 65535 * 128)
    with pytest.raises(ValueError, match="65535 per item"):
        cb._plan_limits(1, 64, 65535 * 128 + 1)
    with pytest.raises(ValueError, match="int32 total"):
        cb._plan_limits(1, 40000 * 64, 65535 * 128)  # 40000 items x 65535 columns
    _expect_rejected(fake, e, _with(a, max_seq_len=65535 * 128 + 1), "per item")
    assert not fake


def test_launches_name_the_device_of_their_tensors(fake) -> None:
    """Every launch passes the device the backend derived from its tensors, so
    residency and the launch stream follow the tensors, not the current device."""
    _call(msa_ops.msa_icp_index_score_sharded_plan_device, plan_device_args())
    _call(msa_ops.msa_icp_decode_index_score_sharded, decode_args())
    assert fake and all(launch.device_index == 0 for launch in fake)


def test_load_and_run_follow_the_given_device(monkeypatch) -> None:
    """``run_module(..., device_index=1)`` with device 0 current makes the
    kernel resident and binds the stream on device 1: ``load`` enters device 1
    before its capture check and residency calls."""
    entered: list[int] = []
    current = [0]

    class Device:
        def __init__(self, index):
            self.index, self.saved = index, None

        def __enter__(self):
            entered.append(self.index)
            self.saved, current[0] = current[0], self.index

        def __exit__(self, *exc):
            current[0] = self.saved

    resident_on: list[tuple[str, int, int]] = []
    streams: list[tuple[int, int]] = []
    module = types.SimpleNamespace(entry=lambda *args: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    monkeypatch.setattr(torch.cuda, "device", Device)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch._C, "_cuda_getCurrentRawStream", lambda d: 100 + d, raising=False
    )
    monkeypatch.setattr(
        cake_jit,
        "_make_resident",
        lambda name, arch, device: resident_on.append((name, device, current[0])),
    )
    monkeypatch.setattr(
        cake_jit,
        "_set_ffi_stream",
        lambda kind, device, stream: streams.append((device, stream)) or 0,
    )
    name = next(iter(cake_jit.MODULES))
    record = dict(cake_jit.MODULES[name], ffi_entry="entry", arg_plan=[])
    monkeypatch.setitem(cake_jit.MODULES, name, record)
    monkeypatch.setattr(cake_jit, "_argument_plan", lambda n: ())
    monkeypatch.setattr(cake_jit, "_LOADED", {(name, "sm_103a"): module})
    monkeypatch.setattr(cake_jit, "_RESIDENT", set())
    monkeypatch.setattr(cake_jit, "_LAUNCHERS", {})
    cake_jit.run_module(name, {}, grid=(1, 1, 1), arch="sm_103a", device_index=1)
    assert resident_on == [(name, 1, 1)] and entered == [1]
    assert sorted(cake_jit._RESIDENT) == [(name, 1)]
    assert (name, "sm_103a", 1) in cake_jit._LAUNCHERS
    assert streams[0] == (1, 101) and current[0] == 0
    # A second launch on device 1 is a cache hit: no device switch, no reload.
    cake_jit.run_module(name, {}, grid=(1, 1, 1), arch="sm_103a", device_index=1)
    assert entered == [1] and len(resident_on) == 1


def test_load_refuses_residency_under_capture(monkeypatch) -> None:
    """A program loaded on one device but not yet resident on the capturing
    device is refused under CUDA Graph capture, before any residency call."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    def refuse(*args):
        raise AssertionError("made resident during capture")

    monkeypatch.setattr(cake_jit, "_make_resident", refuse)
    name = next(iter(cake_jit.MODULES))
    module = types.SimpleNamespace()
    monkeypatch.setattr(cake_jit, "_LOADED", {(name, "sm_103a"): module})
    monkeypatch.setattr(cake_jit, "_RESIDENT", {(name, 0)})
    with pytest.raises(RuntimeError, match="not resident on device 1"):
        cake_jit.load(name, "sm_103a")
    # Resident on the capturing device: returned without a residency call.
    cake_jit._RESIDENT.add((name, 1))
    assert cake_jit.load(name, "sm_103a") is module


def test_select_prefill_rejects_invalid_metadata(fake) -> None:
    e = msa_ops.msa_icp_topk_select_prefill
    a = select_prefill_args()
    t = a["scores"].shape[0]
    _expect_rejected(fake, e, _with(a, scores=_t((t, H, 0), torch.float32)), "columns")
    _expect_rejected(
        fake, e, _with(a, scores=_t((t, H, 12288), torch.float32)), "12288"
    )
    _expect_rejected(
        fake, e, _with(a, scores=_t((t, 8, 40), torch.float32)), "4 index heads"
    )
    _expect_rejected(fake, e, _with(a, scores=_t((t, H, 40), torch.float64)))
    _expect_rejected(
        fake, e, _with(a, scores=_strided((t, H, 40), torch.float32, last_stride=2))
    )
    _expect_rejected(fake, e, _with(a, scores=_t((t, H * 40), torch.float32)))
    _expect_rejected(
        fake, e, _with(a, num_valid_pages=_t((t,), torch.int64)), "num_valid_pages"
    )
    _expect_rejected(
        fake, e, _with(a, num_valid_pages=_t((t + 1,), torch.int32)), "num_valid_pages"
    )
    _expect_rejected(fake, e, _with(a, carrier=_t((H, t, 32), torch.int32)), "carrier")
    _expect_rejected(fake, e, _with(a, carrier=_t((t, H, 33), torch.int32)), "carrier")
    _expect_rejected(fake, e, _with(a, carrier=_t((H, t, 33), torch.int64)), "carrier")
    _expect_rejected(
        fake,
        e,
        _with(a, carrier=_strided((H, t, 33), torch.int32, last_stride=2)),
        "carrier",
    )
    _expect_rejected(
        fake,
        e,
        _with(
            a,
            carrier=_t((H * t * 33,), torch.int32).as_strided(
                (H, t, 33), (t * 33, 32, 1)
            ),
        ),
        "carrier",
    )
    for fb, fe in ((-1, 0), (0, -1), (9, 8), (17, 0)):
        _expect_rejected(
            fake, e, _with(a, force_begin_blocks=fb, force_end_blocks=fe), "forced"
        )
    for s, o in ((3, 0), (0, 0), (1, 1), (2, 2), (2, -1)):
        _expect_rejected(
            fake, e, _with(a, index_scale=s, index_offset=o), "index_scale"
        )
    assert not fake


def test_select_prefill_writes_a_carrier_slice(fake) -> None:
    a = select_prefill_args()
    t = a["scores"].shape[0]
    send = _t((H, 64, 33), torch.int32)
    a["carrier"] = send[:, 17 : 17 + t]
    a["scores"] = _t((t + 2, H, 48), torch.float32)[1 : 1 + t, :, :40]
    a.update(index_scale=2, index_offset=1, force_begin_blocks=16, force_end_blocks=0)
    _call(msa_ops.msa_icp_topk_select_prefill, a)
    (launch,) = fake
    v = launch.values
    assert launch.role == "select_prefill" and launch.rows is None
    assert (v["carrier_stride_h"], v["carrier_stride_t"]) == (64 * 33, 33)
    assert (v["score_stride_t"], v["score_stride_h"], v["score_cols"]) == (
        H * 48,
        48,
        40,
    )
    assert (v["force_begin"], v["force_end"], v["index_scale"], v["index_offset"]) == (
        16,
        0,
        2,
        1,
    )
    assert (v["num_tokens"], v["num_heads"]) == (t, H)
    assert launch.grid_args == {"tokens": t, "heads": H}
    a["scores"] = _t((t, H, 12287), torch.float32)
    _call(msa_ops.msa_icp_topk_select_prefill, a)
    assert fake[-1].values["score_cols"] == 12287
    _call(
        msa_ops.msa_icp_topk_select_prefill,
        _with(
            select_prefill_args(),
            scores=_t((0, H, 40), torch.float32),
            num_valid_pages=_t((0,), torch.int32),
            carrier=_t((H, 0, 33), torch.int32),
        ),
    )
    assert len(fake) == 2


def test_decode_scorer_rejects_invalid_metadata(fake) -> None:
    e = msa_ops.msa_icp_decode_index_score_sharded
    a = decode_args(128, 2, dql=2, batch=3)
    t = a["index_q"].shape[0]
    _expect_rejected(fake, e, _with(a, decode_query_len=5))
    _expect_rejected(fake, e, _with(a, decode_query_len=0))
    _expect_rejected(fake, e, _with(a, max_decode_query_len=9), "32")
    _expect_rejected(fake, e, _with(a, index_q=_t((t + 1, H, 128), FP8)), "batch")
    _expect_rejected(fake, e, _with(a, index_q=_t((t, 8, 128), FP8)), "4 index heads")
    _expect_rejected(
        fake, e, _with(a, score_out=_t((t, H, 18), torch.float32)), "multiple of 4"
    )
    _expect_rejected(fake, e, _with(a, score_out=_t((t, H, 8196), torch.float32)))
    _expect_rejected(fake, e, _with(a, score_out=_t((t, H, 0), torch.float32)))
    _expect_rejected(
        fake, e, _with(a, score_out=_strided((t, H, 16), torch.float32, last_stride=2))
    )
    _expect_rejected(fake, e, _with(a, world_size=3))
    _expect_rejected(fake, e, _with(a, rank=2))
    _expect_rejected(fake, e, _with(a, page_size=64))
    _expect_rejected(
        fake, e, _with(a, page_size=256)
    )  # shard holds 64 rows: P256 TP2 needs 128
    _expect_rejected(fake, e, _with(a, seq_lens=_t((4,), torch.int32)), "seq_lens")
    _expect_rejected(fake, e, _with(a, seq_lens=_t((3,), torch.int64)), "seq_lens")
    _expect_rejected(
        fake, e, _with(a, block_table=_t((3, 4), torch.int64)), "block_table"
    )
    _expect_rejected(
        fake,
        e,
        _with(a, block_table=_strided((3, 4), torch.int32, last_stride=2)),
        "block_table",
    )
    assert not fake


@pytest.mark.parametrize("page,world", LAYOUTS)
def test_decode_scorer_binds_the_layout_and_splits(fake, page: int, world: int) -> None:
    for batch, splits in ((1, 128), (2, 74), (14, 10), (148, 1), (500, 1)):
        a = decode_args(page, world, dql=1, batch=batch)
        a["score_out"] = _t((batch + 2, H, 64), torch.float32)[:batch, :, :32]
        _call(msa_ops.msa_icp_decode_index_score_sharded, a)
        launch = fake[-1]
        v = launch.values
        assert (launch.role, launch.rows) == ("decode_plane", page // world)
        assert (v["rank"], v["world"], v["page_size"], v["query_len"]) == (
            world - 1,
            world,
            page,
            1,
        )
        assert (v["batch"], v["num_tokens"], v["score_cols"], v["splits"]) == (
            batch,
            batch,
            32,
            splits,
        )
        assert (v["score_stride_t"], v["score_stride_h"]) == (H * 64, 64)
        # The plane form reads no candidate inputs: zero forced blocks, the
        # column bound of its output and any int32 vector for the page counts.
        assert (v["columns"], v["init_blocks"], v["local_blocks"]) == (32, 0, 0)
        assert v["num_valid_pages"] is a["seq_lens"]
        assert launch.grid_args == {"batch": batch, "splits": splits}
    assert len(fake) == 5


def test_publish_rejects_invalid_metadata(fake) -> None:
    e = msa_ops.msa_icp_topk_select_publish
    for tagged in (False, True):
        a = publish_args(2, tagged=tagged, t=5, cap=8)
        _expect_rejected(
            fake, e, _with(a, heads_per_rank=1), "world_size \\* heads_per_rank"
        )
        _expect_rejected(fake, e, _with(a, slot=3), "slot")
        _expect_rejected(fake, e, _with(a, slot=-1), "slot")
        _expect_rejected(fake, e, _with(a, capacity=4), "capacity")
        _expect_rejected(
            fake, e, _with(a, recv_bases=_t((2,), torch.int32)), "recv_bases"
        )
        _expect_rejected(
            fake, e, _with(a, recv_bases=_t((4,), torch.int64)), "recv_bases"
        )
        _expect_rejected(fake, e, _with(a, errors=_t((1,), torch.int64)), "errors")
        _expect_rejected(fake, e, _with(a, errors=_t((0,), torch.int32)), "errors")
        _expect_rejected(fake, e, _with(a, spin_limit=0), "spin_limit")
        _expect_rejected(fake, e, _with(a, spin_limit=1 << 31), "spin_limit")
        _expect_rejected(
            fake, e, _with(a, force_begin_blocks=10, force_end_blocks=7), "forced"
        )
        _expect_rejected(
            fake, e, _with(a, index_scale=2, index_offset=2), "index_scale"
        )
        _expect_rejected(fake, e, _with(a, world_size=3, heads_per_rank=1))
        _expect_rejected(fake, e, _with(a, rank=2))
        _expect_rejected(
            fake, e, _with(a, scores=_t((5, H, 12288), torch.float32)), "12288"
        )
        _expect_rejected(
            fake, e, _with(a, num_valid_pages=_t((4,), torch.int32)), "num_valid_pages"
        )
    a = publish_args(2, tagged=True, t=5, cap=8)
    _expect_rejected(
        fake, e, _with(a, generations=_t((3, 4, 7), torch.int32)), "generations"
    )
    _expect_rejected(
        fake, e, _with(a, generations=_t((2, 4, 8), torch.int32)), "generations"
    )
    _expect_rejected(
        fake, e, _with(a, generations=_t((3, 4, 8), torch.int64)), "generations"
    )
    _expect_rejected(
        fake,
        e,
        _with(a, generations=_t((3, 8, 4), torch.int32).transpose(1, 2)),
        "generations",
    )
    assert not fake


@pytest.mark.parametrize("world", (2, 4))
@pytest.mark.parametrize("tagged", (False, True))
def test_publish_binds_the_protocol(fake, world: int, tagged: bool) -> None:
    a = publish_args(world, tagged=tagged, t=5, cap=8)
    a.update(
        index_scale=2,
        index_offset=world - 1 if world == 2 else 1,
        spin_limit=12345,
        enable_pdl=True,
    )
    a["scores"] = _t((5, H, 40), torch.float32)[:, :, :24]
    _call(msa_ops.msa_icp_topk_select_publish, a)
    (launch,) = fake
    v = launch.values
    assert launch.role == "select_publish" and launch.enable_pdl is True
    assert v["tagged"] == int(tagged)
    # legacy rows never touch the counters; the binding still receives an int32 pointer
    assert v["generations"] is (a["generations"] if tagged else a["errors"])
    assert (v["slot"], v["rank"], v["world"], v["heads_per_rank"], v["capacity"]) == (
        2,
        world - 1,
        world,
        H // world,
        8,
    )
    assert (v["spin_limit"], v["force_begin"], v["force_end"]) == (12345, 0, 1)
    assert (v["index_scale"], v["index_offset"]) == (
        a["index_scale"],
        a["index_offset"],
    )
    assert (v["score_cols"], v["score_stride_t"], v["score_stride_h"]) == (
        24,
        H * 40,
        40,
    )
    assert v["recv_bases"] is a["recv_bases"] and v["errors"] is a["errors"]
    # every (token, head) row is published, padding rows included
    assert launch.grid_args == {"tokens": 5, "heads": H}


def test_fused_decode_rejects_invalid_metadata(fake) -> None:
    e = msa_ops.msa_icp_decode_score_publish
    a = fused_args(256, 4)
    need = a["pool"].numel()
    _expect_rejected(fake, e, _with(a, pool=_t((need - 1,), torch.uint8)), "pool")
    _expect_rejected(fake, e, _with(a, pool=_t((need,), torch.int8)), "pool")
    _expect_rejected(fake, e, _with(a, pool=_t((need + 8,), torch.uint8)[::2]), "pool")
    _expect_rejected(fake, e, _with(a, pool=_t((need + 1,), torch.uint8)), "pool")
    misaligned = torch.empty(need + 16, dtype=torch.uint8)[8 : 8 + need]
    _expect_rejected(
        fake, e, _with(fused_args(256, 4, device="cpu"), pool=misaligned), "pool"
    )
    _expect_rejected(
        fake,
        e,
        _with(
            a,
            decode_query_len=9,
            index_q=_t((18, H, 128), FP8),
            num_valid_pages=_t((18,), torch.int32),
        ),
        "32",
    )
    _expect_rejected(fake, e, _with(a, capacity=7), "capacity")
    _expect_rejected(fake, e, _with(a, index_k=_k(9, 128)))
    _expect_rejected(
        fake, e, _with(a, generations=_t((3, H, 15), torch.int32)), "generations"
    )
    assert not fake


@pytest.mark.parametrize("page,world", LAYOUTS)
def test_fused_decode_launches_scorer_then_publisher(
    fake, page: int, world: int
) -> None:
    a = fused_args(page, world, tagged=world == 4)
    _call(msa_ops.msa_icp_decode_score_publish, a)
    scorer, publisher = fake
    assert (scorer.role, scorer.rows, publisher.role, publisher.rows) == (
        "decode_topk",
        page // world,
        "topk16_publish",
        None,
    )
    assert scorer.enable_pdl and publisher.enable_pdl
    pool = scorer.values["pool"]
    assert pool.dtype == torch.uint64 and pool.numel() * 8 == a["pool"].numel()
    lists = (
        cake_jit.MAX_DECODE_SPLITS * 16
    )  # uint64 keys per (token, head) row of the pool
    assert scorer.values["pool_row_stride"] == lists
    assert scorer.values == publisher.values
    v = scorer.values
    s, o = (2, (world - 1) * 2 // world) if page == 256 else (1, 0)
    assert (v["index_scale"], v["index_offset"], v["heads_per_rank"]) == (
        s,
        o,
        H // world,
    )
    assert v["tagged"] == int(world == 4) and v["splits"] == scorer.grid_args["splits"]
    # The list form scores one local column per block-table page and forces
    # the caller's begin / end blocks.
    assert (v["columns"], v["init_blocks"], v["local_blocks"]) == (
        a["block_table"].shape[1],
        a["force_begin_blocks"],
        a["force_end_blocks"],
    )


class _MetaTorch:
    """``torch`` as the facade module sees it while constructing a plan: its
    buffers are allocated on the meta device."""

    def __getattr__(self, name: str) -> Any:
        return getattr(torch, name)

    @staticmethod
    def empty(*args, **kwargs) -> torch.Tensor:
        return torch.empty(*args, **{**kwargs, "device": META})

    @staticmethod
    def zeros(*args, **kwargs) -> torch.Tensor:
        return torch.zeros(*args, **{**kwargs, "device": META})


def _decode_facade(page: int, world: int, *, capacity: int) -> Any:
    """A real :class:`MSAIcpIndexerPlan` (its own constructor) whose buffers
    live on the meta device, so it builds on a CPU node; needs the device
    stubs of :func:`_intercept`.  ``device`` is the meta device, so
    ``check_comm`` accepts the meta comm of :func:`_meta_comm`."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cake_indexer, "torch", _MetaTorch())
        plan = cake_indexer.MSAIcpIndexerPlan(
            world_size=world,
            rank=world - 1,
            num_index_heads=H,
            page_size=page,
            init_blocks=0,
            local_blocks=1,
            max_decode_tokens=capacity,
            max_decode_query_len=4,
            max_num_batched_tokens=256,
            max_num_seqs=4,
            max_model_len=4096,
            num_sparse_layers=2,
            device="cuda:0",
        )
    plan.device = META
    return plan


def _meta_comm(world: int, capacity: int) -> Any:
    """A comm that passes ``check_comm``: legacy and tagged receive rows are
    views of one storage, the tagged rows at ``tagged_offset_bytes``."""
    lay = cb.comm_layout(world, H // world, capacity)
    storage = _t((lay["storage_words"],), torch.int32)
    off = lay["tagged_offset_bytes"] // 4
    return types.SimpleNamespace(
        recv=storage[:off].view(lay["legacy_shape"]),
        tagged_recv=storage[off:].view(torch.int64).view(lay["tagged_shape"]),
        recv_bases=_t((world,), torch.int64),
        tagged_recv_bases=_t((world,), torch.int64),
        publish_generations=_t(lay["generations_shape"], torch.int32),
        expected_generations=_t(lay["expected_shape"], torch.int32),
        errors=_t((1,), torch.int32),
    )


def _decode_case(
    page: int, world: int, tagged: bool
) -> tuple[Any, Any, list, dict, dict]:
    """``(plan, comm, inputs, kwargs, args)`` of one decode layer of
    ``fused_args`` (T = 8 tokens, receive capacity 16)."""
    a = fused_args(page, world, tagged=tagged)
    t, cap = a["index_q"].shape[0], 16
    plan = _decode_facade(page, world, capacity=cap)
    comm = _meta_comm(world, cap)
    scores = _t((t + 3, H, 32), torch.float32)
    output = _t((t, H // world, 16), torch.int32)
    inputs = [a[k] for k in ("index_q", "index_k", "block_table", "seq_lens")]
    inputs += [a["num_valid_pages"], scores, output]
    kwargs = dict(comm=comm, slot=1, tagged=tagged, decode_query_len=4)
    return plan, comm, inputs, kwargs, a


def _same_storage(x: Any, y: torch.Tensor) -> bool:
    return (
        isinstance(x, torch.Tensor)
        and x.untyped_storage()._cdata == y.untyped_storage()._cdata
    )


@pytest.mark.parametrize("tagged", (False, True), ids=("legacy", "tagged"))
@pytest.mark.parametrize(("page", "world"), LAYOUTS)
def test_facade_decode_defaults_to_plane_and_select_publish(
    fake, page: int, world: int, tagged: bool
) -> None:
    """Without ``pool`` a decode layer is score plane -> select publish ->
    merge; with ``pool`` it is the fused list scorer -> list publisher ->
    merge.  Both chains publish into the same receive rows."""
    plan, comm, inputs, kwargs, a = _decode_case(page, world, tagged)
    t = a["index_q"].shape[0]
    bases = comm.tagged_recv_bases if tagged else comm.recv_bases
    plan.decode(*inputs, **kwargs)
    # T = 8 tokens: a legacy merge takes merge_legacy (one token per CTA)
    merge_role = "merge" if tagged or t < cb.LEGACY_MERGE_MIN_TOKENS else "merge_legacy"
    assert fake.roles() == ["decode_plane", "select_publish", merge_role]
    plane, publish, merge = fake
    assert all(launch.enable_pdl for launch in fake)
    assert publish.values["recv_bases"] is bases
    assert plane.rows == page // world
    fake.clear()
    plan.decode(*inputs, **kwargs, pool=a["pool"])
    assert fake.roles() == ["decode_topk", "topk16_publish", merge_role]
    assert fake[1].values["recv_bases"] is bases
    a_recv, b_recv = fake[2].values["recv"], merge.values["recv"]
    assert _same_storage(a_recv, b_recv)
    assert (a_recv.storage_offset(), a_recv.shape, a_recv.stride()) == (
        b_recv.storage_offset(),
        b_recv.shape,
        b_recv.stride(),
    )


# The per-call tensors and arguments ``decode`` still checks once its comm is
# validated (L5 5f16464d: check_comm on the first use of a comm, then the
# trusted publish / merge check scores, num_valid_pages, output and slot).
_BAD_PER_CALL = {
    "scores_dtype": ("scores", lambda x: _t(x.shape, torch.float16)),
    "scores_short": ("scores", lambda x: _t((7, *x.shape[1:]), torch.float32)),
    "scores_heads": (
        "scores",
        lambda x: _t((x.shape[0], H + 1, x.shape[2]), torch.float32),
    ),
    "scores_last_stride": (
        "scores",
        lambda x: _t((x.shape[0], x.shape[1], x.shape[2] * 2), torch.float32)[
            :, :, ::2
        ],
    ),
    "valid_dtype": ("num_valid_pages", lambda x: _t(x.shape, torch.int64)),
    "valid_length": ("num_valid_pages", lambda x: _t((x.numel() + 1,), torch.int32)),
    "output_dtype": ("output", lambda x: _t(x.shape, torch.int64)),
    "output_heads": (
        "output",
        lambda x: _t((x.shape[0], x.shape[1] + 1, 16), torch.int32),
    ),
    "output_topk": ("output", lambda x: _t((x.shape[0], x.shape[1], 15), torch.int32)),
    "output_tokens": (
        "output",
        lambda x: _t((x.shape[0] - 1, x.shape[1], 16), torch.int32),
    ),
    "output_over_capacity": ("output", lambda x: _t((17, x.shape[1], 16), torch.int32)),
    "slot_high": ("slot", lambda x: 3),
    "slot_negative": ("slot", lambda x: -1),
}
_INPUT_INDEX = {"num_valid_pages": 4, "scores": 5, "output": 6}


@pytest.mark.parametrize(
    ("chain", "case"),
    [
        (chain, case)
        for chain in ("plane", "pool")
        for case in sorted(_BAD_PER_CALL)
        if not (chain == "pool" and _BAD_PER_CALL[case][0] == "scores")
    ],
)
@pytest.mark.parametrize("tagged", (False, True), ids=("legacy", "tagged"))
def test_facade_decode_checks_per_call_tensors_after_its_comm_is_checked(
    fake, tagged: bool, chain: str, case: str
) -> None:
    """After the first decode validated the comm, a later decode with one bad
    per-call tensor (or slot) still raises, and the bad value reaches no
    launch.  The pool chain does not read ``scores`` (no scores cases)."""
    field, make_bad = _BAD_PER_CALL[case]
    plan, comm, inputs, kwargs, a = _decode_case(128, 2, tagged)
    extra = dict(pool=a["pool"]) if chain == "pool" else {}
    plan.decode(*inputs, **kwargs, **extra)
    assert len(fake) == 3
    fake.clear()
    bad_inputs, bad_kwargs = list(inputs), dict(kwargs)
    if field == "slot":
        bad = make_bad(None)
        bad_kwargs["slot"] = bad
    else:
        bad = make_bad(inputs[_INPUT_INDEX[field]])
        bad_inputs[_INPUT_INDEX[field]] = bad
    with pytest.raises(ValueError):
        plan.decode(*bad_inputs, **bad_kwargs, **extra)
    if isinstance(bad, torch.Tensor):
        assert not any(
            _same_storage(v, bad) for launch in fake for v in launch.values.values()
        )
    else:
        assert all(launch.values.get("slot") != bad for launch in fake)
    # the good tensors still run the whole layer
    fake.clear()
    plan.decode(*inputs, **kwargs, **extra)
    assert len(fake) == 3


def _comm_readers(field: str, tagged: bool) -> tuple[tuple[int, str], ...]:
    """``(launch index, argument)`` pairs of the default chain that read a comm
    field (none when the mode does not use it)."""
    return {
        "recv": () if tagged else ((2, "recv"),),
        "tagged_recv": ((2, "recv"),) if tagged else (),
        "recv_bases": () if tagged else ((1, "recv_bases"),),
        "tagged_recv_bases": ((1, "recv_bases"),) if tagged else (),
        "publish_generations": ((1, "generations"),) if tagged else (),
        "expected_generations": ((2, "expected"),) if tagged else (),
        "errors": ((1, "errors"), (2, "errors")),
    }[field]


@pytest.mark.parametrize("field", cake_indexer._COMM_FIELDS)
@pytest.mark.parametrize("tagged", (False, True), ids=("legacy", "tagged"))
def test_facade_decode_catches_comm_buffers_swapped_without_check_comm(
    fake, tagged: bool, field: str
) -> None:
    """Replacing a comm field after the comm was validated, without calling
    ``check_comm`` again, is caught by the next decode: a bad replacement
    raises before any launch, and a valid one is either rejected or used;
    the old tensor (or a view of it cached for the slot) is never launched."""
    plan, comm, inputs, kwargs, _ = _decode_case(128, 2, tagged)
    plan.decode(*inputs, **kwargs)
    assert len(fake) == 3
    old = getattr(comm, field)
    readers = _comm_readers(field, tagged)
    for i, arg in readers:
        assert _same_storage(fake[i].values[arg], old), (i, arg)
    # a bad replacement: same shape, wrong dtype
    setattr(comm, field, _t(old.shape, torch.float32))
    fake.clear()
    with pytest.raises(ValueError):
        plan.decode(*inputs, **kwargs)
    assert len(fake) == 0, fake.roles()
    # a valid replacement (a fresh tensor of the same layout)
    if field in ("recv", "tagged_recv"):
        fresh_comm = _meta_comm(2, 16)
        fresh = getattr(fresh_comm, field)
        other = "tagged_recv" if field == "recv" else "recv"
        setattr(comm, other, getattr(fresh_comm, other))  # one new receive storage
    else:
        fresh = _t(old.shape, old.dtype)
    setattr(comm, field, fresh)
    fake.clear()
    try:
        plan.decode(*inputs, **kwargs)
    except ValueError:
        assert len(fake) == 0, fake.roles()
        return
    assert len(fake) == 3
    for launch in fake:
        assert not any(_same_storage(v, old) for v in launch.values.values()), (
            launch.role
        )
    for i, arg in readers:
        assert _same_storage(fake[i].values[arg], fresh), (i, arg)


@pytest.mark.parametrize("arch", ARCHES)
def test_fused_decode_splits_are_capped_per_architecture(
    monkeypatch, arch: str
) -> None:
    """The list scorer splits a request over at most TOPK16_SPLIT_CAP[arch]
    CTAs; the score plane keeps the 128 cap (both: one wave of the device)."""
    fake = _intercept(monkeypatch, arch, real_registry=False)  # 148 SMs, 1 CTA/SM
    assert set(cake_jit.TOPK16_SPLIT_CAP) == set(ARCHES)
    cap = cake_jit.TOPK16_SPLIT_CAP[arch]
    assert 1 <= cap <= cake_jit.MAX_DECODE_SPLITS
    for batch, want_fused, want_plane in (
        (1, min(cap, 128), 128),
        (2, min(cap, 74), 74),
        (148, 1, 1),
        (500, 1, 1),
    ):
        fake.clear()
        _call(msa_ops.msa_icp_decode_score_publish, fused_args(batch=batch))
        scorer, publisher = fake
        assert scorer.values["splits"] == publisher.values["splits"] == want_fused
        assert scorer.grid_args == {"batch": batch, "splits": want_fused}
        fake.clear()
        _call(msa_ops.msa_icp_decode_index_score_sharded, decode_args(batch=batch))
        assert fake[0].values["splits"] == want_plane, batch
    monkeypatch.setitem(cake_jit.TOPK16_SPLIT_CAP, arch, 7)
    fake.clear()
    _call(msa_ops.msa_icp_decode_score_publish, fused_args(batch=1))
    assert fake[0].values["splits"] == 7


@pytest.mark.parametrize("t", (1, 7, 8, 9, 64))
@pytest.mark.parametrize("mode", ("legacy", "tagged", "prefill"))
def test_merge_routes_wide_legacy_decode_merges_to_merge_legacy(
    fake, mode: str, t: int
) -> None:
    """Prefill carrier merges run the prefill program ``merge_prefill``; legacy
    decode merges of at least 8 tokens run the one-token-per-CTA
    ``merge_legacy`` program; tagged rows and fewer tokens run ``merge`` (4
    tokens per CTA).  Same launch arguments either way."""
    assert cb.LEGACY_MERGE_MIN_TOKENS == 8
    world = 2
    a = merge_args(
        world,
        tagged=mode == "tagged",
        prefill=mode == "prefill",
        t=t,
        cap=max(8, t),
    )
    if mode != "prefill":
        a["enable_pdl"] = True
    _call(msa_ops.msa_icp_merge_candidates, a)
    (launch,) = fake
    want = (
        "merge_prefill"
        if mode == "prefill"
        else "merge_legacy"
        if mode == "legacy" and t >= 8
        else "merge"
    )
    assert (launch.role, launch.rows) == (want, None), (mode, t)
    role = cb.merge_role(
        prefill=mode == "prefill", tagged=mode == "tagged", num_tokens=t
    )
    assert role == want
    assert launch.grid_args == {"tokens": t, "heads": H // world}
    assert launch.values["num_tokens"] == t
    assert launch.values["tagged"] == int(mode == "tagged")
    assert launch.values["prefill"] == int(mode == "prefill")
    assert launch.enable_pdl == (mode != "prefill")


@pytest.mark.parametrize("arch", ARCHES)
def test_merge_programs_launch_one_or_four_tokens_per_cta(
    monkeypatch, arch: str
) -> None:
    """With the exported records: ``merge_legacy`` launches (T, hpr) CTAs,
    ``merge`` and ``merge_prefill`` (ceil(T / 4), hpr)."""
    fake = _intercept(monkeypatch, arch, real_registry=True)
    rules = {
        name: record["grid_rule"]
        for name, record in cake_jit.MODULES.items()
        if record["role"] in ("merge", "merge_legacy", "merge_prefill")
    }
    assert sorted(r["rows"] for r in rules.values()) == [1, 4, 4], rules
    for mode, t, want_x in (
        ("legacy", 9, 9),
        ("legacy", 7, 2),
        ("tagged", 9, 3),
        ("prefill", 9, 3),
    ):
        fake.clear()
        a = merge_args(
            2,
            tagged=mode == "tagged",
            prefill=mode == "prefill",
            t=t,
            cap=max(8, t),
        )
        _call(msa_ops.msa_icp_merge_candidates, a)
        (launch,) = fake
        grid = cake_jit.launch_grid(launch.name, **launch.grid_args)
        assert grid == (want_x, 2, 1), (mode, t, launch.name)


def test_merge_rejects_invalid_metadata(fake) -> None:
    e = msa_ops.msa_icp_merge_candidates
    a = merge_args(2, t=5, cap=8)
    _expect_rejected(fake, e, _with(a, prefill=True, enable_pdl=True), "prefill")
    _expect_rejected(
        fake, e, _with(merge_args(2, tagged=True), prefill=True), "prefill"
    )
    _expect_rejected(
        fake, e, _with(a, recv_slot=_t((2, 2, 8 * 33), torch.int32)), "recv_slot"
    )
    _expect_rejected(
        fake, e, _with(a, recv_slot=_t((3, 1, 8, 33), torch.int32)), "2 or 4"
    )
    _expect_rejected(
        fake, e, _with(a, recv_slot=_t((2, 2, 8, 32), torch.int32)), "legacy"
    )
    _expect_rejected(
        fake, e, _with(a, recv_slot=_t((2, 2, 8, 33), torch.int64)), "legacy"
    )
    _expect_rejected(
        fake,
        e,
        _with(a, recv_slot=_strided((2, 2, 8, 33), torch.int32, last_stride=2)),
        "legacy",
    )
    _expect_rejected(
        fake,
        e,
        _with(
            a,
            recv_slot=_t((2, 4, 8, 33), torch.int32),
            output=_t((5, 4, 16), torch.int32),
        ),
        "4 index heads",
    )
    _expect_rejected(fake, e, _with(a, output=_t((5, 2, 15), torch.int32)), "output")
    _expect_rejected(fake, e, _with(a, output=_t((5, 1, 16), torch.int32)), "output")
    _expect_rejected(fake, e, _with(a, output=_t((5, 2, 16), torch.int64)), "output")
    _expect_rejected(
        fake,
        e,
        _with(a, output=_strided((5, 2, 16), torch.int32, last_stride=2)),
        "output",
    )
    _expect_rejected(
        fake,
        e,
        _with(
            a, output=_t((9, 2, 16), torch.int32), num_valid_pages=_t((9,), torch.int32)
        ),
        "capacity",
    )
    _expect_rejected(
        fake, e, _with(a, num_valid_pages=_t((4,), torch.int32)), "num_valid_pages"
    )
    _expect_rejected(fake, e, _with(a, errors=_t((1,), torch.float32)), "errors")
    _expect_rejected(fake, e, _with(a, init_blocks=-1), "init_blocks")
    _expect_rejected(fake, e, _with(a, local_blocks=-1), "init_blocks")
    _expect_rejected(fake, e, _with(a, spin_limit=0), "spin_limit")
    b = merge_args(2, tagged=True, t=5, cap=8)
    _expect_rejected(
        fake, e, _with(b, recv_slot=_t((2, 2, 8, 33), torch.int32)), "tagged"
    )
    _expect_rejected(
        fake,
        e,
        _with(b, expected_generations=_t((2, 7), torch.int32)),
        "expected_generations",
    )
    _expect_rejected(
        fake,
        e,
        _with(b, expected_generations=_t((2, 8), torch.int64)),
        "expected_generations",
    )
    _expect_rejected(
        fake,
        e,
        _with(b, expected_generations=_t((8, 2), torch.int32).t()),
        "expected_generations",
    )
    assert not fake


@pytest.mark.parametrize("world", (2, 4))
@pytest.mark.parametrize("mode", ("legacy", "tagged", "prefill"))
def test_merge_binds_the_rows(fake, world: int, mode: str) -> None:
    a = merge_args(
        world, tagged=mode == "tagged", prefill=mode == "prefill", t=5, cap=8
    )
    a.update(init_blocks=2, local_blocks=3, spin_limit=777)
    if mode != "prefill":
        a["enable_pdl"] = True
    _call(msa_ops.msa_icp_merge_candidates, a)
    (launch,) = fake
    v = launch.values
    hpr = H // world
    # T = 5 < 8: every decode form takes the general merge, the prefill carrier
    # merge its own program (it follows the caller's exchange without PDL;
    # decode merges use PDL).
    want = "merge_prefill" if mode == "prefill" else "merge"
    assert launch.role == want and launch.enable_pdl == (mode != "prefill")
    assert (v["tagged"], v["prefill"]) == (
        int(mode == "tagged"),
        int(mode == "prefill"),
    )
    # one program reads both row forms through int32 words: a tagged row is 64 int32 words
    words = 64 if mode == "tagged" else 33
    recv = v["recv"]
    assert recv.dtype == torch.int32 and tuple(recv.shape) == (world, hpr, 8, words)
    assert (v["recv_stride_src"], v["recv_stride_h"], v["recv_stride_t"]) == (
        hpr * 8 * words,
        8 * words,
        words,
    )
    assert (v["world"], v["heads_per_rank"], v["capacity"], v["num_tokens"]) == (
        world,
        hpr,
        8,
        5,
    )
    assert (v["init_blocks"], v["local_blocks"], v["spin_limit"]) == (2, 3, 777)
    assert (v["out_stride_t"], v["out_stride_h"]) == (hpr * 16, 16)
    want_expected = (
        a["expected_generations"] if mode == "tagged" else a["num_valid_pages"]
    )
    assert v["expected"] is want_expected
    assert launch.grid_args == {"tokens": 5, "heads": hpr}


def test_zero_token_calls_launch_nothing(fake) -> None:
    _call(
        msa_ops.msa_icp_topk_select_publish,
        _with(
            publish_args(),
            scores=_t((0, H, 24), torch.float32),
            num_valid_pages=_t((0,), torch.int32),
        ),
    )
    _call(
        msa_ops.msa_icp_merge_candidates,
        _with(
            merge_args(),
            output=_t((0, 2, 16), torch.int32),
            num_valid_pages=_t((0,), torch.int32),
        ),
    )
    _call(
        msa_ops.msa_icp_decode_index_score_sharded,
        _with(
            decode_args(batch=1, dql=1),
            index_q=_t((0, H, 128), FP8),
            block_table=_t((0, 4), torch.int32),
            seq_lens=_t((0,), torch.int32),
            score_out=_t((0, H, 16), torch.float32),
        ),
    )
    assert not fake


# --------------------------------------------------------------------------- sizing and layout helpers


@pytest.mark.parametrize("world", (2, 4))
@pytest.mark.parametrize("capacity", (1, 3, 7, 56, 4096))
def test_comm_layout_matches_the_oracle(world: int, capacity: int) -> None:
    hpr = H // world
    got = msa_ops.msa_icp_comm_layout(world, hpr, capacity)
    want = ref.comm_layout(world, hpr, capacity)
    for key in ("storage_words", "legacy_offset_bytes", "tagged_offset_bytes"):
        assert got[key] == want[key], key
    for key in ("legacy_shape", "tagged_shape", "generations_shape", "expected_shape"):
        assert tuple(got[key]) == tuple(want[key]), key
    assert got["tagged_offset_bytes"] % 8 == 0
    assert got["tagged_offset_bytes"] == 3 * world * hpr * capacity * 33 * 4
    assert (
        got["storage_words"] * 4
        == got["tagged_offset_bytes"] + 3 * world * hpr * capacity * 32 * 8
    )
    # every publish destination lies inside the storage
    last = ref.publish_row_index(
        2, world, world - 1, hpr, H - 1, capacity, capacity - 1
    )
    assert (last + 1) * 33 * 4 <= got["tagged_offset_bytes"]
    assert got["tagged_offset_bytes"] + (last + 1) * 32 * 8 == got["storage_words"] * 4


def test_comm_layout_rejects_invalid_geometry() -> None:
    for args in ((3, 1, 8), (1, 4, 8), (2, 0, 8), (2, 2, 0)):
        with pytest.raises(ValueError):
            msa_ops.msa_icp_comm_layout(*args)


@pytest.mark.parametrize("page", (128, 256))
def test_decode_columns_follow_vllm(page: int) -> None:
    for length in (
        1,
        127,
        128,
        129,
        page * 4,
        page * 4 + 1,
        8191,
        163840,
        172031,
        196608,
        1 << 20,
        (1 << 21) + 5,
    ):
        want = min(8192, -(-length // (4 * page)) * 4)
        assert msa_ops.msa_icp_decode_score_columns(length, page) == want, length
    for bad in ((0, page), (128, 64), (128, 512)):
        with pytest.raises(ValueError):
            msa_ops.msa_icp_decode_score_columns(*bad)


@pytest.mark.parametrize("page", (128, 256))
def test_prefill_columns_and_chunks_follow_vllm(page: int) -> None:
    for length in (1, 127, 128, 129, 255, 256, 257, 8192, 32768, 163840, 172031):
        blocks = -(-length // 128)
        want = -(-blocks // 2) if page == 256 else blocks
        assert msa_ops.msa_icp_index_score_columns(length, page) == want, length
        pitch = cake_indexer.prefill_score_columns(length, page)
        assert pitch == ref.prefill_columns(length, page, compact=True), length
        chunk = cake_indexer.prefill_chunk_tokens(length, page, H)
        assert chunk == min(
            8192, max(128, (256 << 20) // (H * pitch * 5) // 128 * 128)
        ), length
    for bad in ((0, page), (128, 64)):
        with pytest.raises(ValueError):
            msa_ops.msa_icp_index_score_columns(*bad)


def test_plan_storage_matches_msa(monkeypatch) -> None:
    monkeypatch.setattr(cake_jit, "sm_count", lambda index: SM_COUNT)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    for total_q, batch in (
        (1, 1),
        (128, 1),
        (129, 1),
        (8192, 14),
        (8192, 256),
        (300, 3),
    ):
        got = msa_ops.msa_icp_index_score_plan_storage(
            num_qo_heads=H, total_q=total_q, batch=batch, device="cuda:0"
        )
        want = ref.plan_storage(H, total_q, batch)
        assert got == {
            "work_range_numel": SM_COUNT,
            "work_info_numel": want["work_info_numel_min"],
            "workspace_bytes": want["workspace_bytes_min"],
        }
    for bad in (
        dict(num_qo_heads=0, total_q=1, batch=1),
        dict(num_qo_heads=H, total_q=0, batch=1),
        dict(num_qo_heads=H, total_q=1, batch=0),
    ):
        with pytest.raises(ValueError):
            msa_ops.msa_icp_index_score_plan_storage(device="cuda:0", **bad)


def test_decode_pool_bytes() -> None:
    for tokens in (1, 14, 56, 4096):
        assert (
            msa_ops.msa_icp_decode_pool_bytes(max_decode_tokens=tokens)
            == tokens * H * 128 * 16 * 8
        )
    with pytest.raises(ValueError):
        msa_ops.msa_icp_decode_pool_bytes(max_decode_tokens=0)
    with pytest.raises(ValueError, match="4 index heads"):
        msa_ops.msa_icp_decode_pool_bytes(max_decode_tokens=8, num_heads=8)


@pytest.mark.parametrize("page,world", LAYOUTS)
def test_index_layout_matches_the_oracle(page: int, world: int) -> None:
    lay = ref.IcpLayout(page, world)
    for rank in range(world):
        assert cb.index_layout(page, world, rank) == (
            lay.index_scale,
            lay.index_offset(rank),
        )


# --------------------------------------------------------------------------- head limit and facade validation


def _facade_kwargs(**changes) -> dict:
    kw = dict(
        world_size=2,
        rank=0,
        num_index_heads=H,
        page_size=128,
        init_blocks=0,
        local_blocks=1,
        max_decode_tokens=56,
        max_decode_query_len=4,
        max_num_batched_tokens=8192,
        max_num_seqs=14,
        max_model_len=172032,
        num_sparse_layers=57,
        device="cpu",
    )
    kw.update(changes)
    return kw


def test_only_four_index_heads_are_supported(fake) -> None:
    for heads in (1, 2, 8, 16):
        with pytest.raises(ValueError, match="4 index heads"):
            msa_ops.prepare_msa_icp_indexer(**_facade_kwargs(num_index_heads=heads))
        with pytest.raises(ValueError, match="4 index heads"):
            _call(
                msa_ops.msa_icp_topk_select_prefill,
                _with(
                    select_prefill_args(),
                    scores=_t((5, heads, 40), torch.float32),
                    carrier=_t((heads, 5, 33), torch.int32),
                ),
            )
        with pytest.raises(ValueError, match="4 index heads"):
            _call(
                msa_ops.msa_icp_index_score_sharded_plan_device,
                _with(plan_device_args(), num_qo_heads=heads),
            )
    with pytest.raises(ValueError, match="4 index heads"):
        _call(
            msa_ops.msa_icp_topk_select_publish,
            _with(
                publish_args(2), scores=_t((5, 8, 24), torch.float32), heads_per_rank=4
            ),
        )
    assert not fake


def test_facade_validates_limits_before_touching_the_device(monkeypatch) -> None:
    def boom(*a, **k):
        raise AssertionError("reached the device")

    monkeypatch.setattr(cake_jit, "device_arch", boom)
    cases = [
        (dict(world_size=3), "world_size"),
        (dict(rank=2), "world_size"),
        (dict(page_size=64), "page_size"),
        (dict(init_blocks=9, local_blocks=8), "forced"),
        (dict(max_decode_query_len=9), "32"),
        (dict(max_decode_query_len=0), "32"),
        (dict(max_decode_tokens=0), "positive"),
        (dict(max_model_len=0), "positive"),
        (dict(num_sparse_layers=0), "num_sparse_layers"),
        (dict(), "CUDA device"),
    ]
    for changes, match in cases:
        with pytest.raises(ValueError, match=match):
            msa_ops.prepare_msa_icp_indexer(**_facade_kwargs(**changes))


# --------------------------------------------------------------------------- capability dispatch


def test_device_arch_dispatches_exact_capabilities(monkeypatch) -> None:
    caps = {0: (10, 0), 1: (10, 3), 2: (10, 7), 3: (9, 0), 4: (12, 0), 5: (10, 1)}
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda index=None: caps[index]
    )
    cake_jit.device_arch.cache_clear()
    try:
        assert [cake_jit.device_arch(i) for i in range(3)] == list(ARCHES)
        for i in (3, 4, 5):
            with pytest.raises(ValueError, match="10.0, 10.3 and 10.7"):
                cake_jit.device_arch(i)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert [cb.is_supported(i) for i in range(6)] == [
            True,
            True,
            True,
            False,
            False,
            False,
        ]
    finally:
        cake_jit.device_arch.cache_clear()


def test_unloaded_program_is_refused_under_graph_capture(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setitem(
        cake_jit.MODULES,
        "cake_msa_icp_indexer_probe",
        {"library": "cake_msa_icp_indexer_probe"},
    )
    with pytest.raises(RuntimeError, match="warm"):
        cake_jit.load("cake_msa_icp_indexer_probe", "sm_103a")


# --------------------------------------------------------------------------- oracle


def test_oracle_self_test() -> None:
    counts = ref.self_test(verbose=False)
    assert counts and all(n > 0 for n in counts.values()), counts


def test_golden_vectors_match_the_oracle() -> None:
    assert ref.check_golden() == []
