# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JIT loader for the source-only Cake GDN backend.

Every variant in ``csrc/gdn/cake/manifest.json`` names one kernel source, the
compile-time specialization it is built with (passed as ``-D`` defines), and
the template and parameters of its TVM-FFI host shim.  The kernel is compiled
to a cubin with ``nvcc`` and embedded into the rendered host shim.
"""

from __future__ import annotations

import bisect
import functools
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

from filelock import FileLock
from tvm_ffi import cpp

from . import env as jit_env
from .cpp_ext import get_cuda_path, get_nvcc_parallelism_flags

CakeGDNArch = Literal["sm_100a", "sm_103a", "sm_107a"]

_EXPORT_SCHEMA = "flashinfer-cake-gdn-decode-standalone-export-v1"
_DEFINE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RECORD_NAME = re.compile(r"^(decode|prefill)_(.+)_[0-9a-f]{12}$")
_RECORD_TMA_ABI = {"decode": "pointer", "prefill": "grid_constant"}
_ARCH_ACTIVE_CLUSTERS: dict[CakeGDNArch, int] = {
    "sm_100a": 148,
    "sm_103a": 160,
    "sm_107a": 212,
}


class CakeGDNUnsupportedError(NotImplementedError):
    """The explicit GDN non-CP backend cannot serve the requested GDN contract."""


class CakeGDNRoute(NamedTuple):
    route_id: str
    variant_name: str
    # The record reads int32 or int64 ``cu_seqlens`` / ``state_indices`` / ``checkpoint_cu_starts``
    # directly (manifest field ``metadata_i64``: every sm_107a non-CP prefill record); the host
    # passes the caller's buffers without a per-call cast.  False on every other record.
    metadata_i64: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nvcc_identity() -> tuple[Path, str]:
    nvcc = (Path(get_cuda_path()) / "bin" / "nvcc").resolve()
    if not nvcc.is_file():
        raise RuntimeError(f"nvcc was not found at {nvcc}")
    result = subprocess.run(
        [str(nvcc), "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to identify nvcc at {nvcc}:\n{result.stdout}")
    return nvcc, result.stdout


def _compile_cache_digest(
    *,
    arch: CakeGDNArch,
    cuda_sha256: str,
    header_sha256s: tuple[str, ...],
    compile_options: tuple[str, ...],
    nvcc: Path,
    nvcc_version: str,
) -> str:
    return hashlib.sha256(
        "\0".join(
            [
                arch,
                cuda_sha256,
                *header_sha256s,
                *compile_options,
                str(nvcc),
                nvcc_version,
            ]
        ).encode()
    ).hexdigest()


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "gdn" / "cake"
    if installed.exists():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "gdn" / "cake"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        f"frozen Cake GDN sources were not found; checked {installed} and {checkout}"
    )


@functools.cache
def _manifest() -> dict[str, Any]:
    path = _source_dir() / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != _EXPORT_SCHEMA:
        raise RuntimeError(
            f"unexpected Cake GDN manifest schema at {path}: {manifest.get('schema')!r}"
        )
    for record in manifest["variants"]:
        _derive_record_fields(record)
    return manifest


def _derive_record_fields(record: dict[str, Any]) -> None:
    """Fill the record fields the export derives from its name and outputs.

    ``<domain>_<schedule>_<12 hex>`` names every record: ``domain`` is its
    first token, ``source_schedule`` is
    ``flashinfer.cake_gdn.generated.<domain>:<schedule>``, ``tma_abi`` is
    fixed per domain, ``host_binding.entry`` is ``run_<name>`` and the
    record-level ``architectures`` is the sorted union of its outputs'.  A
    field the manifest spells out must agree with the derived value.
    """

    name = record["name"]
    match = _RECORD_NAME.match(name)
    if match is None:
        raise RuntimeError(f"unparseable Cake GDN record name {name!r}")
    domain, schedule = match.groups()
    derived: dict[str, Any] = {
        "domain": domain,
        "source_schedule": f"flashinfer.cake_gdn.generated.{domain}:{schedule}",
        "tma_abi": _RECORD_TMA_ABI[domain],
        "architectures": sorted(
            {arch for output in record["outputs"] for arch in output["architectures"]}
        ),
    }
    targets = [
        (record, "", derived),
        (record["host_binding"], "host_binding.", {"entry": f"run_{name}"}),
    ]
    for target, prefix, fields in targets:
        for field, value in fields.items():
            if field in target and target[field] != value:
                raise RuntimeError(
                    f"Cake GDN record {name!r} spells {prefix}{field}="
                    f"{target[field]!r}; its name and outputs derive {value!r}"
                )
            target[field] = value


def _kernel_record(name: str) -> dict[str, Any]:
    records = [record for record in _manifest()["variants"] if record["name"] == name]
    if len(records) != 1:
        raise ValueError(f"unknown Cake GDN kernel: {name!r}")
    return records[0]


def _cuda_record(record: dict[str, Any], arch: CakeGDNArch) -> dict[str, Any]:
    outputs = [
        output for output in record["outputs"] if arch in output["architectures"]
    ]
    if len(outputs) != 1:
        raise ValueError(f"kernel {record['name']!r} does not support {arch}")
    return outputs[0]


def _cubin_path(source: Path, *, arch: CakeGDNArch, digest: str) -> Path:
    return (
        jit_env.FLASHINFER_JIT_DIR
        / "cake_gdn"
        / arch
        / f"{source.stem}-{digest[:16]}.cubin"
    )


def _compile_cubin(
    source: Path,
    *,
    arch: CakeGDNArch,
    digest: str,
    compile_options: tuple[str, ...],
    include_paths: tuple[Path, ...],
    nvcc: Path,
) -> bytes:
    cubin = _cubin_path(source, arch=arch, digest=digest)
    cache_dir = cubin.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(f"{cubin}.lock", thread_local=False)
    with lock:
        if not cubin.exists():
            with tempfile.NamedTemporaryFile(
                dir=cache_dir,
                prefix=f".{cubin.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            try:
                command = [
                    str(nvcc),
                    "--cubin",
                    "--std=c++17",
                    "-O3",
                    f"--gpu-architecture={arch}",
                    *(f"-I{path}" for path in include_paths),
                    *compile_options,
                    *get_nvcc_parallelism_flags(),
                    str(source),
                    "-o",
                    str(temporary),
                ]
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"failed to compile {source.name} for {arch}:\n{result.stdout}"
                    )
                os.replace(temporary, cubin)
            finally:
                temporary.unlink(missing_ok=True)
    return cubin.read_bytes()


def _define_flags(record: dict[str, Any]) -> tuple[str, ...]:
    """``-D`` flags for the variant's compile-time specialization.

    A record passes its ``specializations`` unless it spells out ``defines``:
    a body with its constants in-file takes none (``{}``), and some bodies
    take only a subset of the specialization keys.
    """

    flags = []
    for key, value in record.get("defines", record["specializations"]).items():
        if not _DEFINE_NAME.match(key):
            raise RuntimeError(f"invalid Cake GDN define name {key!r}")
        flags.append(f"-D{key}={value}")
    return tuple(flags)


def _compile_options(record: dict[str, Any]) -> tuple[str, ...]:
    """``nvcc`` options of the variant: ``--use_fast_math`` and its ``-D`` flags."""

    options = tuple(record.get("compile_options", ("--use_fast_math",)))
    unsupported_options = set(options) - {"--use_fast_math"}
    if unsupported_options:
        raise RuntimeError(
            f"unsupported Cake GDN compile options: {sorted(unsupported_options)!r}"
        )
    return options + _define_flags(record)


@functools.cache
def compile_cake_gdn_cubin(name: str, arch: CakeGDNArch) -> Path:
    """Compile one Cake GDN kernel variant for ``arch`` and return its cubin path."""

    if arch not in _ARCH_ACTIVE_CLUSTERS:
        raise CakeGDNUnsupportedError(f"unsupported Cake GDN architecture: {arch!r}")
    record = _kernel_record(name)
    cuda = _cuda_record(record, arch)
    root = _source_dir()
    cuda_path = root / cuda["path"]
    headers = _manifest().get("cuda_headers", [])
    compile_options = _compile_options(record)
    nvcc, nvcc_version = _nvcc_identity()
    compile_digest = _compile_cache_digest(
        arch=arch,
        cuda_sha256=_sha256(cuda_path),
        header_sha256s=tuple(_sha256(root / header["path"]) for header in headers),
        compile_options=compile_options,
        nvcc=nvcc,
        nvcc_version=nvcc_version,
    )
    _compile_cubin(
        cuda_path,
        arch=arch,
        digest=compile_digest,
        compile_options=compile_options,
        # The shared device header includes its helper definitions from the
        # content-addressed ``csrc/cake_device_helpers/`` files relative to ``csrc``
        # (root.parents[1], as the host-shim build below already passes it).
        include_paths=tuple(
            dict.fromkeys(
                [
                    *((root / header["path"]).parent for header in headers),
                    root.parents[1],
                ]
            )
        ),
        nvcc=nvcc,
    )
    return _cubin_path(cuda_path, arch=arch, digest=compile_digest)


_HOST_SHIM_PLACEHOLDER = re.compile(r"@([A-Z][A-Z0-9_]*)@")
_HOST_SHIM_CONDITIONAL = re.compile(
    r"^//@if (\w+)\n(.*?)^//@else\n(.*?)^//@endif\n", re.S | re.M
)


def render_cake_gdn_host_shim(record: dict[str, Any]) -> str:
    """Render the TVM-FFI host shim of one variant from its class template.

    ``host_binding.template`` names the template under ``host/templates/``.
    A ``//@if metadata_i64`` / ``//@else`` / ``//@endif`` block of lines keeps
    its first arm for a record whose ``metadata_i64`` is true and its second
    arm otherwise; the directive lines are dropped before the placeholders.
    ``@ENTRY@``, ``@MODULE_IDENT@``, ``@CUDA_SYMBOL@`` and ``@SHIM_NAMESPACE@``
    come from ``host_binding``; every other ``@NAME@`` placeholder takes the
    value of ``host_binding.params[NAME]``.  The result is byte-identical to
    the per-variant shim file the template replaced.
    """

    host = record["host_binding"]
    template = (_source_dir() / host["template"]).read_text(encoding="utf-8")
    flags = {"metadata_i64": bool(record.get("metadata_i64", False))}
    template = _HOST_SHIM_CONDITIONAL.sub(
        lambda match: match.group(2) if flags[match.group(1)] else match.group(3),
        template,
    )
    if "//@" in template:
        raise RuntimeError(f"{host['template']} has an unresolved host-shim directive")
    values = {
        "ENTRY": host["entry"],
        "MODULE_IDENT": host["module_ident"],
        "CUDA_SYMBOL": host["cuda_symbol"],
        "SHIM_NAMESPACE": host["shim_namespace"],
    }
    for key, value in host["params"].items():
        if key in values:
            raise RuntimeError(f"reserved Cake GDN host-shim parameter {key!r}")
        values[key] = str(value)
    unused = set(values)

    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise RuntimeError(
                f"{host['template']} needs {key!r}; {record['name']!r} lacks it"
            )
        unused.discard(key)
        return values[key]

    text = _HOST_SHIM_PLACEHOLDER.sub(substitute, template)
    if unused:
        raise RuntimeError(
            f"unused host-shim parameters for {record['name']!r}: {sorted(unused)}"
        )
    return text


@functools.cache
def load_cake_gdn_kernel(name: str, arch: CakeGDNArch):
    """Compile and load one Cake GDN host entrypoint."""

    cubin = compile_cake_gdn_cubin(name, arch)
    record = _kernel_record(name)
    host = record["host_binding"]
    root = _source_dir()
    host_source = render_cake_gdn_host_shim(record)
    host_digest = hashlib.sha256(host_source.encode("utf-8")).hexdigest()
    module_key = f"{cubin.stem}\0{host_digest}".encode()
    module_digest = hashlib.sha256(module_key).hexdigest()
    module = cpp.load_inline(
        f"flashinfer_cake_gdn_{name}_{arch}_{module_digest[:12]}",
        cpp_sources=host_source,
        embed_cubin={host["module_ident"]: cubin.read_bytes()},
        extra_include_paths=[
            str(Path(get_cuda_path()) / "include"),
            str(root / "host"),
            str(root.parent),
            str(root.parents[1]),
            str(root.parents[2] / "include"),
        ],
        extra_ldflags=["-lcuda"],
    )
    return module[host["entry"]]


def _power_of_two_log2(value: int) -> int:
    return value.bit_length() - 1 if value > 0 and value & (value - 1) == 0 else -1


# Host grid rule of the persistent non-CP prefill kernel.  The constants and the decision mirror the Cake
# generator's grid rule (one source of truth; the exporter's mirror test keeps the copies identical).  Every
# role of the kernel enumerates ``range(bid, total_tiles, gridDim.x)`` and the tiles are sequence-major
# (tile ``t`` is sequence ``t // heads``, head ``t % heads``).  Costs are modelled in 64-token-chunk units
# from host integers only; the kernel binary is unchanged by the grid.  The model is evaluated per sequence
# (never per tile), so the eager launch path pays a few microseconds per call.
SHORT_GRID = 128
# Per-tile fixed cost (prologue, seed, final-state readout) in 64-token chunk units.
TILE_OVERHEAD_CHUNKS = 4
# Extra cost of a freshly dispatched CTA over a persistent tile boundary (fitted on the
# measured Hq16/Hv32 rows: the model is within -1.5 .. +3.1 points of the measured gain).
DISPATCH_OVERHEAD_CHUNKS = 1
# One CTA per tile only when the modelled critical path shrinks by at least this.
RAGGED_MIN_GAIN_PERCENT = 4


def _prefill_residency_grid(total_tiles: int, sm_count: int, dvsplit: bool) -> int:
    """The residency rule: one CTA per tile up to the SM count for DV-split launches or at most
    128 tiles; otherwise the cap G in {128, sm_count} with the smaller per-CTA tile quantum
    ceil(total_tiles / G) wins and ties keep 128."""

    if dvsplit or total_tiles <= SHORT_GRID:
        return min(sm_count, total_tiles)
    g_short = min(SHORT_GRID, total_tiles)
    g_sm = min(sm_count, total_tiles)
    if -(-total_tiles // g_sm) < -(-total_tiles // g_short):
        return g_sm
    return g_short


def _prefill_tile_chunks(seq_len: int) -> int:
    """The kernel's chunk count for one sequence: ``2 * ceil(seq_len / 128)`` 64-token chunks
    (0 for an empty one)."""

    return 2 * (-(-max(int(seq_len), 0) // 128))


def _prefill_seq_costs(seq_lens: Sequence[int]) -> list[int]:
    """One entry per sequence: the cost of each of its tiles (its chunks plus the per-tile
    overhead); the ``heads`` tiles of a sequence share it."""

    return [_prefill_tile_chunks(length) + TILE_OVERHEAD_CHUNKS for length in seq_lens]


def _prefill_strided_critical_path(
    seq_costs: Sequence[int], heads: int, grid: int
) -> int:
    """Persistent schedule: CTA ``b`` runs tiles ``b, b + grid, ...`` back to back; the longest CTA.

    The ``heads`` tiles of sequence ``s`` are the index range ``[s * heads, (s + 1) * heads)``:
    every CTA gets ``heads // grid`` of them and the ``heads % grid`` residues from
    ``(s * heads) % grid`` on (wrapping at ``grid``) one more.  The per-CTA sums are therefore a
    sweep over the 2S range endpoints instead of a pass over the tiles; the maximum of the
    piecewise-constant sum sits at an endpoint.
    """

    full, part = divmod(heads, grid)
    base = full * sum(seq_costs)
    if part == 0:
        return base
    events = []
    for index, cost in enumerate(seq_costs):
        start = (index * heads) % grid
        end = start + part
        events.append((start, cost))
        if end <= grid:
            events.append((end, -cost))
        else:
            events.append((0, cost))
            events.append((end - grid, -cost))
    events.sort()
    best = level = 0
    i, n = 0, len(events)
    while i < n:
        pos = events[i][0]
        while i < n and events[i][0] == pos:
            level += events[i][1]
            i += 1
        if level > best:
            best = level
    return base + best


def _prefill_dispatch_critical_path(
    seq_costs: Sequence[int], heads: int, sm_count: int
) -> int:
    """One CTA per tile: the tiles are dispatched in index order, one resident CTA per SM, each
    SM takes the next CTA when free, and every CTA pays ``DISPATCH_OVERHEAD_CHUNKS`` on top of
    its tile cost -- the makespan of the in-order list schedule on ``min(sm_count, tiles)``
    machines.

    Exact and group-wise.  The machines' free times are a sorted run-length multiset; a tile
    starts at the smallest free time.  The ``heads`` equal-cost tiles of one sequence are taken
    in chunks: the smallest free times within ``p`` (the tile's processing time) of the smallest
    one.  The completions of such a chunk, all at least ``smallest + p``, cannot precede any
    start of the chunk, so the chunk reproduces the per-tile schedule exactly (a tie gives the
    same values either way); its completions are merged back as runs.  A batch of S sequences
    costs O(S) chunks unless the free times are spread wider than ``p``.
    """

    total = heads * len(seq_costs)
    machines = max(1, min(sm_count, total))
    vals = [0]
    cnts = [machines]
    finish = 0
    for cost in seq_costs:
        p = cost + DISPATCH_OVERHEAD_CHUNKS
        need = heads
        while need:
            limit = vals[0] + p
            taken_v: list[int] = []
            taken_n: list[int] = []
            i = 0
            while need and i < len(vals) and vals[i] <= limit:
                n = cnts[i]
                taken_v.append(vals[i])
                if n <= need:
                    taken_n.append(n)
                    need -= n
                    i += 1
                else:
                    taken_n.append(need)
                    cnts[i] = n - need
                    need = 0
            if i:
                del vals[:i]
                del cnts[:i]
            done = taken_v[-1] + p
            if done > finish:
                finish = done
            for v, n in zip(taken_v, taken_n, strict=True):
                w = v + p
                j = bisect.bisect_left(vals, w)
                if j < len(vals) and vals[j] == w:
                    cnts[j] += n
                else:
                    vals.insert(j, w)
                    cnts.insert(j, n)
    return finish


@functools.lru_cache(maxsize=4096)
def _cake_gdn_prefill_grid_x(
    total_tiles: int,
    sm_count: int,
    dvsplit: bool,
    bound_tokens: int,
    host_seq_lens: tuple[int, ...] | None = None,
) -> int:
    """Persistent CTA count for ``total_tiles`` work tiles on ``sm_count`` SMs.

    Mirrors the Cake generator's grid rule exactly; the exporter's mirror test keeps the
    copies identical.  Host integers only: nothing is read back from the device and the
    kernel binary is unchanged by the grid.  The launch (``gdn_prefill.py``) and the variant
    resolver below (the gate-stream rule) share this one copy.

    The residency rule (``_prefill_residency_grid``): DV-split launches and at most 128
    tiles take one CTA per tile up to the SM count; otherwise the cap G in {128, sm_count}
    with the smaller per-CTA tile quantum ceil(total_tiles / G) wins and ties keep 128
    (grid A/B: 1024-tile rows 9-10 % faster on 152 CTAs, 512/256-tile rows ~2 % faster
    on 128).

    The ragged-batch route: ``host_seq_lens`` (the per-sequence token counts in
    ``cu_seqlens`` order, known only from ``cu_seqlens_cpu``) lets the rule model both
    schedules in 64-token-chunk units.  The persistent schedule runs tile ``t`` on CTA
    ``t % grid`` (``_prefill_strided_critical_path``), so a ragged full-DV batch can chain
    a long tile and further tiles on one CTA while other CTAs idle; with one CTA per tile
    the hardware dispatches the CTAs in index order onto the free SMs
    (``_prefill_dispatch_critical_path``).  The rule launches ``total_tiles`` CTAs when the
    modelled critical path shrinks by at least ``RAGGED_MIN_GAIN_PERCENT``; the grid may
    then exceed the SM count (one resident CTA per SM).  Equal-length batches never route:
    with one cost c per tile the strided path is rounds(grid) * c and the dispatched one
    ceil(tiles / M) * (c + 1), and rounds(grid) <= ceil(tiles / M) by the choice of the
    residency grid, so they return the residency grid without a model.  Slightly ragged
    and long-last batches keep the residency grid, as do DV-split launches, batches whose
    residency grid already gives one CTA per tile and length lists that do not divide
    ``total_tiles``; without ``host_seq_lens`` the residency rule decides.

    Cost: O(S) for a batch of S sequences in the common case -- the per-sequence costs, the
    strided path as an endpoint sweep, then exact bounds on the dispatched makespan (at
    least the longest tile and the work per machine; at most the work per machine plus the
    longest tile's share of the other machines, the list-scheduling bound) that settle most
    batches; only the rest run the group-wise dispatch simulation.  The result is memoised
    on its arguments (``host_seq_lens`` must be a tuple), so the other layers of a step pay
    a lookup.

    ``bound_tokens`` is an upper bound on the longest sequence (exact from
    ``cu_seqlens_cpu``, else the ``max_seqlen`` hint, else the packed token count); it is
    part of the contract signature and validated by the caller, but the rule does not
    depend on it, so a loose bound never changes the grid.  The result is the residency
    grid or ``total_tiles``, so the host shim's ``0 < grid_x <= total_tiles`` holds for
    every input.

    The launch calls this rule with the device's SM count, the resolver with the
    architecture's nominal active-cluster count.  Near the gain margin the two can differ
    on a device that exposes fewer SMs than the nominal count (a GB300 with 152 of the
    nominal 160): the launch grid is authoritative for the workspace; the resolver only
    picks the body.
    """

    grid = _prefill_residency_grid(total_tiles, sm_count, dvsplit)
    if dvsplit or host_seq_lens is None or grid >= total_tiles:
        return grid
    num_seqs = len(host_seq_lens)
    if num_seqs == 0 or total_tiles % num_seqs:
        return grid
    costs = _prefill_seq_costs(host_seq_lens)
    if min(costs) == max(costs):
        return grid
    heads = total_tiles // num_seqs
    strided = _prefill_strided_critical_path(costs, heads, grid)
    machines = max(1, min(sm_count, total_tiles))
    longest = max(costs) + DISPATCH_OVERHEAD_CHUNKS
    work = heads * (sum(costs) + num_seqs * DISPATCH_OVERHEAD_CHUNKS)
    gain = strided * (100 - RAGGED_MIN_GAIN_PERCENT)
    if gain < max(longest, -(-work // machines)) * 100:
        return grid
    if gain * machines >= (work + longest * (machines - 1)) * 100:
        return total_tiles
    dispatch = _prefill_dispatch_critical_path(costs, heads, sm_count)
    if gain >= dispatch * 100:
        return total_tiles
    return grid


# Mirror of the Cake program rule ``gate_stream_flat_default`` (exporter ``_prefill_plan``), same constants: the
# full-DV stateful bodies take their ``_gatestream_full`` twin (the epilogue warp streams the strided successor
# tile's gate/beta prologue during the current tile) when a CTA owns at least two work tiles under the grid rule
# and the mean sequence length is at most 512 tokens; single-chunk and indexed-state launches and every other key
# stay on their base.  Host scalars only.
_GATE_STREAM_FULL_MIN_TILES_PER_CTA = 2
_GATE_STREAM_FULL_MAX_MEAN_SEQ_LEN = 512
_GATE_STREAM_FULL_BASES = frozenset(
    {
        "flashinfer_blackwell_gdn_prefill_initial",
        "flashinfer_blackwell_gdn_prefill_initial_bf16state",
    }
)


def _cake_gdn_prefill_gate_stream_full(
    *, num_seqs: int, total_seq_len: int, total_tiles: int, grid_x: int
) -> bool:
    tiles_per_cta = -(-total_tiles // grid_x)
    return (
        tiles_per_cta >= _GATE_STREAM_FULL_MIN_TILES_PER_CTA
        and total_seq_len <= _GATE_STREAM_FULL_MAX_MEAN_SEQ_LEN * num_seqs
    )


def _prefill_schedule_attr(
    *,
    dvsplit: bool,
    use_initial_state: bool,
    io_dtype: str,
    state_dtype: str,
    single_chunk: bool,
    gatepipe4: bool,
    fullgrid_vhold: bool,
    sm100a_state_inp_late: bool,
    sm103a_role_rebalanced: bool,
    gate_stream_full: bool,
) -> str:
    if gatepipe4:
        return "flashinfer_blackwell_gdn_prefill_dvsplit_initial_f16io_gatepipe4"
    if fullgrid_vhold:
        return "flashinfer_blackwell_gdn_prefill_dvsplit_initial_f16io_fullgrid_vhold"
    stem = (
        "flashinfer_blackwell_gdn_prefill_dvsplit"
        if dvsplit
        else "flashinfer_blackwell_gdn_prefill"
    )
    if (
        single_chunk
        and io_dtype == "bfloat16"
        and not use_initial_state
        and state_dtype == "float32"
    ):
        return f"{stem}_single_chunk"
    if state_dtype in {"float16", "float8_e4m3fn", "float8_e5m2"}:
        suffix = {
            "float16": "f16state",
            "float8_e4m3fn": "e4m3state",
            "float8_e5m2": "e5m2state",
        }[state_dtype]
        return f"{stem}_initial_{suffix}"
    pieces = [stem]
    if use_initial_state:
        pieces.append("initial")
    if io_dtype == "float16":
        pieces.append("f16io")
    if state_dtype == "bfloat16":
        pieces.append("bf16state")
    name = "_".join(pieces)
    if sm100a_state_inp_late:
        # The four bf16-I/O DV-split bodies keep the pre-L2 (late ``state_inp_ready``) state epilogue on sm_100a.
        name += "_sm100a_state_inp_late"
    if sm103a_role_rebalanced:
        # The exact SM103a Hq16/Hv16 fp16 stateful DV-split key builds the 224/56 register role-split body.
        name += "_sm103a_role_rebalanced"
    if gate_stream_full and name in _GATE_STREAM_FULL_BASES:
        # The full-DV stateful bodies with several short tiles per CTA stream the successor's gate prologue.
        name += "_gatestream_full"
    return name


def _serves_arch(record: dict[str, Any], arch: CakeGDNArch) -> bool:
    """Whether a manifest record lists ``arch`` and carries a kernel body for it."""

    return arch in record["architectures"] and any(
        arch in output["architectures"] for output in record["outputs"]
    )


# Rubin (sm_107a) twins of the sequential (non-CP) prefill kernel, routed per (architecture,
# shipped schedule): the dvsplit fp32-state schedules run the `l7_vec2` body, the dvsplit
# bf16-state schedules the `l7b_vec2lp` body (8-byte state I/O on the initial-state
# load, the final-state store and the checkpoint stores; every arithmetic operation keeps its
# operands, order and rounding, so the output is bitwise identical to the shipped body).  An
# indexed-state plan (`USE_STATE_INDICES`) traces its own `_indexed` schedule, keyed as such.
# Every other shipped schedule runs the `i64meta` body on sm_107a (the per-architecture default,
# `CAKE_GDN_PREFILL_SEQ_DEFAULT_BODIES`): the same kernel reading int32 or int64 launch metadata
# directly, so the host passes the caller's `cu_seqlens` / `state_indices` /
# `checkpoint_cu_starts` without a per-call cast; every sm_107a twin (the routed bodies too)
# carries that contract (record field `metadata_i64`).  No sm_107a record is a base body.
# Mirrors `PREFILL_SEQ_ARCH_BODIES` / `PREFILL_SEQ_DEFAULT_BODIES` in
# the Cake prefill kernel program (the Cake tree cross-checks this copy).
_SM107A_SEQ_TWINS: dict[str, tuple[str, ...]] = {
    # fp32-state dvsplit schedules: 8-byte fp32 state I/O
    "l7_vec2": (
        "dvsplit",
        "dvsplit_f16io",
        "dvsplit_initial",
        "dvsplit_initial_f16io",
        "dvsplit_initial_f16io_gatepipe4",
    ),
    # bf16 / f16-state dvsplit schedules with shipped records (the bf16 one indexed or
    # not): the same width lever on the low-precision state loads and stores,
    # alignment-guarded on the indexed plans
    "l7b_vec2lp": (
        "dvsplit_initial_bf16state",
        "dvsplit_initial_bf16state_indexed",
    ),
    # The f16-state schedule runs the smem-staged final-state store twin
    # (`vec2_state_io_lp` + `st2_stage`) on CC 10.7; a per-plan record whose outputs
    # are bitwise identical to the shipped body.
    "st2s_vec2lp": ("dvsplit_initial_f16state",),
    # The fp8-state (e4m3 / e5m2) schedules run the smem-staged packed final-state store
    # twin (`st8_stage`) on CC 10.7; per-plan records whose
    # outputs are bitwise identical to the `i64meta` body they supersede.
    "st8s_pack": ("dvsplit_initial_e4m3state", "dvsplit_initial_e5m2state"),
}
CAKE_GDN_PREFILL_SEQ_ARCH_BODIES: dict[tuple[str, str], str] = {
    ("sm_107a", f"flashinfer_blackwell_gdn_prefill_{suffix}"): body
    for body, suffixes in _SM107A_SEQ_TWINS.items()
    for suffix in suffixes
}
# The body of every shipped schedule the table above does not route, per architecture
# (absent = the shipped schedule itself).
CAKE_GDN_PREFILL_SEQ_DEFAULT_BODIES: dict[str, str] = {"sm_107a": "i64meta"}


def cake_gdn_prefill_seq_schedule(
    arch: CakeGDNArch, schedule_attr: str, *, use_state_indices: bool
) -> str:
    """The sequential prefill schedule `arch` runs for the shipped `schedule_attr`.

    The shipped schedule itself, or its per-architecture twin
    (`<schedule>_sm107a_<body>`, a record of its own in the manifest): the body
    `CAKE_GDN_PREFILL_SEQ_ARCH_BODIES` routes, else the architecture's default
    body (`CAKE_GDN_PREFILL_SEQ_DEFAULT_BODIES`; on sm_107a every schedule is a twin).
    """
    symbol = f"{schedule_attr}_indexed" if use_state_indices else schedule_attr
    body = CAKE_GDN_PREFILL_SEQ_ARCH_BODIES.get(
        (arch, symbol), CAKE_GDN_PREFILL_SEQ_DEFAULT_BODIES.get(arch, "base")
    )
    return schedule_attr if body == "base" else f"{schedule_attr}_sm107a_{body}"


def _variant_for(
    *,
    arch: CakeGDNArch,
    domain: str,
    schedule_attr: str,
    specializations: dict[str, int | float],
) -> dict[str, Any]:
    """The one frozen record ``arch`` runs for an exact (schedule, specializations) key.

    A record that does not list ``arch`` (or has no body for it) is not a
    match: an architecture the row was never exported for fails closed here
    with ``CakeGDNUnsupportedError`` instead of surfacing as a ``ValueError``
    from the cubin loader.
    """

    source_schedule = {
        "decode": "flashinfer.cake_gdn.generated.decode",
        "prefill": "flashinfer.cake_gdn.generated.prefill",
    }[domain]
    expected_source = f"{source_schedule}:{schedule_attr}"
    expected_tma_abi = "grid_constant" if domain == "prefill" else "pointer"
    matches = [
        record
        for record in _manifest()["variants"]
        if record["domain"] == domain
        and record["source_schedule"] == expected_source
        and record["specializations"] == specializations
        and record["tma_abi"] == expected_tma_abi
        and _serves_arch(record, arch)
    ]
    if len(matches) != 1:
        raise CakeGDNUnsupportedError(
            "no exact frozen Cake GDN variant for "
            f"{expected_source} with {specializations!r} on {arch}"
        )
    return matches[0]


def _has_variant(
    *,
    arch: CakeGDNArch,
    domain: str,
    schedule_attr: str,
    specializations: dict[str, int | float],
) -> bool:
    """Whether the manifest ships exactly one record for the key on ``arch`` (see :func:`_variant_for`)."""

    try:
        _variant_for(
            arch=arch,
            domain=domain,
            schedule_attr=schedule_attr,
            specializations=specializations,
        )
    except CakeGDNUnsupportedError:
        return False
    return True


@functools.lru_cache(maxsize=4096)
def select_cake_gdn_prefill_variant(
    *,
    arch: CakeGDNArch,
    io_dtype: str,
    state_dtype: str,
    num_seqs: int,
    total_seq_len: int,
    max_seq_len: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    use_initial_state: bool,
    store_final_state: bool,
    checkpoint_every_n_tokens: int,
    use_state_indices: bool,
    gates_present: bool,
    host_seq_lens: tuple[int, ...] | None = None,
) -> CakeGDNRoute:
    """Resolve host-visible non-CP metadata to one frozen GDN non-CP variant.

    Every input is host-known: the dtypes, head counts and flags come from the
    call, ``num_seqs`` and ``total_seq_len`` from tensor shapes, and
    ``max_seq_len`` is an *upper bound* on the longest sequence -- exact when
    the caller knows the lengths (``cu_seqlens_cpu``), otherwise the public
    ``max_seqlen`` hint or ``total_seq_len``.  Nothing here is read back from
    the device, so the resolution is CUDA Graph capture safe.

    ``single_chunk`` (the ``SINGLE_CHUNK_NO_STATE`` schedule) is selected only
    when the bound is at most 64 tokens and the row carries no initial state,
    no final state and no checkpoints.  A loose bound resolves the general
    stateless specialization instead, which fails closed when the manifest
    does not ship it (like any other unshipped specialization): stateless
    callers whose packed batch exceeds 64 tokens pass ``max_seqlen`` or
    ``cu_seqlens_cpu`` to reach the single-chunk kernel.  The SGLang TP4 BF16
    indexed checkpoint route is admitted on structural conditions alone
    (BF16 I/O and state, heads (4, 4, 8), interval 64, indexed initial and
    final state, DV-split); ``cu_seqlens`` / ``checkpoint_cu_starts``
    consistency is a caller precondition the kernel guards on the device.

    ``host_seq_lens`` is the optional host copy of the per-sequence lengths
    (a tuple in ``cu_seqlens`` order -- the resolver is memoised, so it must
    be hashable), derived from ``cu_seqlens_cpu`` only.  It enables the
    ragged-batch launch grid of ``_cake_gdn_prefill_grid_x`` so the
    gate-stream twin follows the routed grid.  The resolver evaluates the
    grid rule with the architecture's nominal active-cluster count while
    the launch uses the device's SM count; near the gain margin the two can
    differ on a device that exposes fewer SMs than the nominal count (a
    GB300 with 152 of the nominal 160) -- the launch grid is authoritative
    for the workspace, the resolver only picks the body.
    """

    if arch not in _ARCH_ACTIVE_CLUSTERS:
        raise CakeGDNUnsupportedError(f"unsupported architecture {arch}")
    if io_dtype not in {"float16", "bfloat16"}:
        raise CakeGDNUnsupportedError(f"unsupported I/O dtype {io_dtype}")
    if state_dtype not in {
        "float32",
        "bfloat16",
        "float16",
        "float8_e4m3fn",
        "float8_e5m2",
    }:
        raise CakeGDNUnsupportedError(f"unsupported state dtype {state_dtype}")
    min_heads = min(num_q_heads, num_v_heads)
    num_o_heads = max(num_q_heads, num_v_heads)
    if min_heads <= 0 or num_k_heads != min_heads or num_o_heads % min_heads:
        raise CakeGDNUnsupportedError("unsupported GDN head mapping")
    if (
        num_seqs <= 0
        or total_seq_len <= 0
        or max_seq_len <= 0
        or max_seq_len > total_seq_len
    ):
        raise CakeGDNUnsupportedError(
            "Cake GDN prefill requires positive, consistent sequence metadata"
        )
    if checkpoint_every_n_tokens < 0 or (
        checkpoint_every_n_tokens and checkpoint_every_n_tokens % 64
    ):
        raise CakeGDNUnsupportedError(
            "checkpoint interval must be zero or a positive multiple of 64"
        )
    dvsplit = 2 * num_seqs * num_o_heads <= _ARCH_ACTIVE_CLUSTERS[arch]
    enable_checkpoints = checkpoint_every_n_tokens > 0
    if use_state_indices and (not use_initial_state or not store_final_state):
        raise CakeGDNUnsupportedError("indexed state requires initial and final state")
    low_precision_state = state_dtype in {
        "float16",
        "float8_e4m3fn",
        "float8_e5m2",
    }
    if low_precision_state and (
        io_dtype != "bfloat16"
        or not use_initial_state
        or not store_final_state
        or enable_checkpoints
        or use_state_indices
    ):
        raise CakeGDNUnsupportedError(
            "low-precision state requires BF16 I/O, initial+final state, "
            "no checkpoints, and packed state"
        )
    legacy_checkpoint_route = (
        io_dtype == "float16"
        and state_dtype == "float32"
        and not use_initial_state
        and store_final_state
        and not use_state_indices
    )
    sglang_tp4_checkpoint_route = (
        io_dtype == "bfloat16"
        and state_dtype == "bfloat16"
        and use_initial_state
        and store_final_state
        and use_state_indices
        and checkpoint_every_n_tokens == 64
        and (num_q_heads, num_k_heads, num_v_heads) == (4, 4, 8)
        and dvsplit
    )
    if enable_checkpoints and not (
        legacy_checkpoint_route or sglang_tp4_checkpoint_route
    ):
        raise CakeGDNUnsupportedError(
            "checkpoint route requires the frozen FP16/FP32 packed contract or "
            "the SGLang TP4 BF16 indexed DV-split contract"
        )
    if low_precision_state and not dvsplit:
        raise CakeGDNUnsupportedError(
            "low-precision state requires the DV-split physical schedule"
        )
    # Bound-based: with an exact ``max_seq_len`` this is the stock rule; a
    # loose bound (``total_seq_len``) only fires when the whole batch fits one
    # chunk, which is always correct (every sequence is then <= 64 tokens).
    single_chunk = (
        max_seq_len <= 64
        and not use_initial_state
        and not store_final_state
        and not enable_checkpoints
    )
    total_tiles = num_seqs * num_o_heads * (2 if dvsplit else 1)
    gatepipe4 = (
        dvsplit
        and total_tiles != 128
        and not single_chunk
        and gates_present
        and use_initial_state
        and store_final_state
        and not enable_checkpoints
        and io_dtype == "float16"
        and state_dtype == "float32"
        and not use_state_indices
    )
    fullgrid_vhold = (
        arch == "sm_100a"
        and dvsplit
        and total_tiles == 128
        and not single_chunk
        and gates_present
        and use_initial_state
        and store_final_state
        and not enable_checkpoints
        and io_dtype == "float16"
        and state_dtype == "float32"
        and not use_state_indices
    )
    # Mirror of the Cake program rule ``state_inp_early_dvsplit_default_for_arch`` (exporter ``_prefill_plan``):
    # on sm_100a the bf16-I/O DV-split bodies (fp32 / bf16 state) are the late-``state_inp_ready`` twins; f16 I/O
    # (gatepipe4 / fullgrid_vhold included), low-precision state, single-chunk and indexed-state routes stay base.
    sm100a_state_inp_late = (
        arch == "sm_100a"
        and dvsplit
        and not single_chunk
        and io_dtype == "bfloat16"
        and state_dtype in ("float32", "bfloat16")
        and not use_state_indices
    )
    # Mirror of the Cake program rule ``_should_use_sm103a_initial_state_role_rebalance`` (exporter ``_prefill_plan``):
    # on sm_103a the exact Hq16/Hv16 fp16-I/O fp32-state DV-split key with initial + final state (no checkpoints, no
    # state indices, not single-chunk, not the gatepipe4 schedule) is the 224/56 register role-split twin
    # ``..._dvsplit_initial_f16io_sm103a_role_rebalanced``; every other key and architecture stays on its base.
    sm103a_role_rebalanced = (
        arch == "sm_103a"
        and dvsplit
        and not single_chunk
        and not gatepipe4
        and use_initial_state
        and store_final_state
        and not enable_checkpoints
        and io_dtype == "float16"
        and state_dtype == "float32"
        and not use_state_indices
        and num_q_heads >= num_v_heads
        and num_o_heads == 16
        and num_o_heads == min_heads
    )
    # Mirror of the Cake program rule ``gate_stream_flat_default`` (exporter ``_prefill_plan``): the launch grid
    # from the architecture's active-cluster count (the same host-static count that decides ``dvsplit``) and
    # the host sequence lengths when the caller gave them (the ragged-batch route), then the
    # ``_gatestream_full`` twin for the full-DV stateful bases whose CTAs own several short tiles.  The launch
    # evaluates the same rule with the device's SM count; near the gain margin the two can differ (a GB300
    # exposing 152 of the nominal 160 SMs) and the launch grid is authoritative for the workspace.
    grid_x = _cake_gdn_prefill_grid_x(
        total_tiles, _ARCH_ACTIVE_CLUSTERS[arch], dvsplit, max_seq_len, host_seq_lens
    )
    gate_stream_full = (
        not single_chunk
        and not use_state_indices
        and _cake_gdn_prefill_gate_stream_full(
            num_seqs=num_seqs,
            total_seq_len=total_seq_len,
            total_tiles=total_tiles,
            grid_x=grid_x,
        )
    )
    specializations: dict[str, int | float] = {
        "ENABLE_CHECKPOINTS": int(enable_checkpoints),
        "HEAD_GROUP_LOG2": _power_of_two_log2(num_o_heads // min_heads),
        "IS_GQA": int(num_q_heads >= num_v_heads),
        "NUM_O_HEADS_LOG2": _power_of_two_log2(num_o_heads),
        "SINGLE_CHUNK_NO_STATE": int(single_chunk),
        "STORE_FINAL_STATE": int(store_final_state),
        "UNIT_GATES": 0,
        "USE_INITIAL_STATE": int(use_initial_state),
        "USE_STATE_INDICES": int(use_state_indices),
    }

    def _schedule_attr(*, gate_stream_full: bool) -> str:
        return _prefill_schedule_attr(
            dvsplit=dvsplit,
            use_initial_state=use_initial_state,
            io_dtype=io_dtype,
            state_dtype=state_dtype,
            single_chunk=single_chunk,
            gatepipe4=gatepipe4,
            fullgrid_vhold=fullgrid_vhold,
            sm100a_state_inp_late=sm100a_state_inp_late,
            sm103a_role_rebalanced=sm103a_role_rebalanced,
            gate_stream_full=gate_stream_full,
        )

    # The sm_107a sequential twin of the resolved schedule (`cake_gdn_prefill_seq_schedule`): a
    # record of its own, applied to the schedule the base rule settles on -- including the
    # gate-stream twin, whose `_sm107a_i64meta` record is the one that exists on sm_107a (the
    # gate-stream base record lists the other architectures only), so the gate-stream
    # fallback below probes the twin-resolved name.
    def _seq_schedule_attr(*, gate_stream_full: bool) -> str:
        return cake_gdn_prefill_seq_schedule(
            arch,
            _schedule_attr(gate_stream_full=gate_stream_full),
            use_state_indices=use_state_indices,
        )

    schedule_attr = _seq_schedule_attr(gate_stream_full=gate_stream_full)
    if (
        gate_stream_full
        and "_gatestream_full" in schedule_attr
        and not _has_variant(
            arch=arch,
            domain="prefill",
            schedule_attr=schedule_attr,
            specializations=specializations,
        )
    ):
        # The gate-stream twin is a performance specialization of the base body: the exporter
        # ships it for the head geometries its representative rows reach (the Hq16/Hv32,
        # Hq8/Hv16 and Hq4/Hv8 families).  A key whose twin is not in the manifest runs its
        # base record, exactly as the route did before the twin existed; only a missing base
        # fails closed below.
        gate_stream_full = False
        schedule_attr = _seq_schedule_attr(gate_stream_full=False)
    record = _variant_for(
        arch=arch,
        domain="prefill",
        schedule_attr=schedule_attr,
        specializations=specializations,
    )
    if single_chunk:
        route = "flashinfer.gdn_prefill.noncp.single_chunk"
    elif enable_checkpoints:
        route = "flashinfer.gdn_prefill.noncp.checkpoints"
    else:
        route = "flashinfer.gdn_prefill.noncp"
    return CakeGDNRoute(
        f"{route}.{'dvsplit' if dvsplit else 'full_dv'}",
        record["name"],
        bool(record.get("metadata_i64", False)),
    )


def cake_gdn_prefill_geometries(
    *,
    arch: CakeGDNArch,
    io_dtype: str,
    state_dtype: str,
    use_initial_state: bool,
    store_final_state: bool,
    dvsplit: bool,
) -> frozenset[tuple[int, int, int]]:
    """Frozen ``(HEAD_GROUP_LOG2, IS_GQA, NUM_O_HEADS_LOG2)`` triples of one prefill contract on ``arch``.

    The contract is the plain non-CP prefill route of one physical schedule
    (DV-split or full-DV) for the given I/O and state dtypes with or without
    an initial and a final state: no checkpoints, packed (non-indexed) state,
    gates taken at runtime.  The triples are read from the manifest records
    the resolver can select on ``arch``; nothing is tabulated here.  A head
    pair maps to ``HEAD_GROUP_LOG2 = log2(max(Hq, Hv) / min(Hq, Hv))``,
    ``IS_GQA = int(Hq >= Hv)`` and ``NUM_O_HEADS_LOG2 = log2(max(Hq, Hv))``;
    ``-1`` marks a head count that is not a power of two (a dynamic-head body
    takes it at runtime).
    """

    if arch not in _ARCH_ACTIVE_CLUSTERS:
        raise CakeGDNUnsupportedError(f"unsupported architecture {arch}")
    if io_dtype not in {"float16", "bfloat16"}:
        raise CakeGDNUnsupportedError(f"unsupported I/O dtype {io_dtype}")
    if state_dtype not in {
        "float32",
        "bfloat16",
        "float16",
        "float8_e4m3fn",
        "float8_e5m2",
    }:
        raise CakeGDNUnsupportedError(f"unsupported state dtype {state_dtype}")
    # The schedules the resolver can select for the contract on ``arch``: the
    # base template and the twin it is forced to substitute on this
    # architecture.  On sm_100a the late-``state_inp_ready`` twin replaces the
    # bf16-I/O DV-split bases for every geometry, so only the twin records
    # list sm_100a; on sm_103a the role-rebalanced twin replaces the f16-I/O
    # fp32-state stateful DV-split base for the one geometry it is exported
    # for, so its records are read next to the base.  The contract fixes the
    # remaining terms of both rules (packed state, no single chunk, no
    # checkpoints); the geometry terms of the sm_103a rule are the twin
    # record's own specializations.  The other performance twins (gatepipe4,
    # fullgrid_vhold, gate-stream full) are exported for exactly the geometry
    # triples of their base record.  On an architecture whose sequential rule
    # replaces every base by a per-architecture twin record (sm_107a,
    # `cake_gdn_prefill_seq_schedule`: the base record does not list it), the
    # twin records enumerate the contract, so every expected schedule is read
    # through that rule.
    sm100a_state_inp_late = (
        arch == "sm_100a"
        and dvsplit
        and io_dtype == "bfloat16"
        and state_dtype in ("float32", "bfloat16")
    )
    sm103a_role_rebalanced = (
        arch == "sm_103a"
        and dvsplit
        and use_initial_state
        and store_final_state
        and io_dtype == "float16"
        and state_dtype == "float32"
    )

    def _source(*, role_rebalanced: bool) -> str:
        schedule_attr = cake_gdn_prefill_seq_schedule(
            arch,
            _prefill_schedule_attr(
                dvsplit=dvsplit,
                use_initial_state=use_initial_state,
                io_dtype=io_dtype,
                state_dtype=state_dtype,
                single_chunk=False,
                gatepipe4=False,
                fullgrid_vhold=False,
                sm100a_state_inp_late=sm100a_state_inp_late,
                sm103a_role_rebalanced=role_rebalanced,
                gate_stream_full=False,
            ),
            use_state_indices=False,
        )
        return f"flashinfer.cake_gdn.generated.prefill:{schedule_attr}"

    expected_sources = {_source(role_rebalanced=False)}
    if sm103a_role_rebalanced:
        expected_sources.add(_source(role_rebalanced=True))
    fixed: dict[str, int] = {
        "ENABLE_CHECKPOINTS": 0,
        "SINGLE_CHUNK_NO_STATE": 0,
        "STORE_FINAL_STATE": int(store_final_state),
        "UNIT_GATES": 0,
        "USE_INITIAL_STATE": int(use_initial_state),
        "USE_STATE_INDICES": 0,
    }
    geometry_keys = ("HEAD_GROUP_LOG2", "IS_GQA", "NUM_O_HEADS_LOG2")
    geometries: set[tuple[int, int, int]] = set()
    for record in _manifest()["variants"]:
        specializations = record["specializations"]
        if (
            record["domain"] != "prefill"
            or record["source_schedule"] not in expected_sources
            or record["tma_abi"] != "grid_constant"
            or not _serves_arch(record, arch)
            or set(specializations) != set(fixed) | set(geometry_keys)
            or any(specializations[key] != value for key, value in fixed.items())
        ):
            continue
        geometries.add(
            tuple(int(specializations[key]) for key in geometry_keys)  # type: ignore[arg-type]
        )
    return frozenset(geometries)


# BF16-state T=1 decode: every per-rank (H, HV) geometry is admitted at any
# batch; the body and grid tile follow the state-head count B*HV in bands
# (mirrors the Cake program rule ``select_bf16_t1_route``; calibrated against
# the FlashInfer BF16 pool kernel on B200/B300).  ``vec8`` is the
# latency-first instance of the vec8 body (``__launch_bounds__(128, 1)``, every
# first-block load ahead of the reductions), ``vec8r56`` the same schedule under
# a 56-register cap (``__launch_bounds__(128, 9)``; the sm_103a build spills
# nothing and reads 1.4-4.3 % faster than ``vec8`` at TILE_V=16 while the
# sm_100a build is 4-5 % slower, so it is routed per architecture through
# ``CAKE_GDN_BF16_T1_ROUTE_ARCH_BODIES``), ``vec8occ`` the occupancy-first
# instance (default launch bounds, eight CTAs per SM so the 1024-CTA grids at
# 256 and 512 state heads run in one wave) and ``wide`` the MTP wide-tile body
# at ``T_STEPS=1`` (one 128-row CTA per state head) for the bandwidth-bound
# large pools.  The grid is ``V / TILE_V`` 128-thread CTAs per state head.
CAKE_GDN_BF16_T1_BODIES = ("vec8", "vec8r56", "vec8occ", "wide")
# (largest state-head count of the band or None for the open last band, body, TILE_V)
CAKE_GDN_BF16_T1_ROUTE_BANDS = (
    (8, "vec8", 32),
    # 9-192 heads: TILE_V=16 (``vec8r56`` on sm_103a).  TILE_V=32 ties it at
    # 192 heads in the calibration sweeps on both GPUs but its 768-CTA grid is
    # launch-order sensitive on GB300 (the first of two back-to-back launches
    # runs ~7 % slower; export rows b6 H16/HV32 and b24 H4/HV8 failed the
    # directional gate every round).
    (192, "vec8", 16),
    (256, "vec8occ", 32),
    (384, "vec8", 64),
    (512, "vec8occ", 64),
    (768, "vec8", 64),
    (None, "wide", 128),
)


# Per-architecture body overrides of the band rule ({arch: {band upper bound: body}};
# mirrors Cake's ``BF16_T1_ROUTE_ARCH_BODIES``): GB300 takes the 56-register
# ``vec8r56`` instance in the 9-192 band, B200 keeps ``vec8``.
CAKE_GDN_BF16_T1_ROUTE_ARCH_BODIES: dict[str, dict[int, str]] = {
    "sm_103a": {192: "vec8r56"}
}

# Architectures whose calibration does not follow the shared bands carry a
# complete band table of their own (mirrors Cake's ``BF16_T1_ROUTE_ARCH_BANDS``).
# sm_107a (Rubin R200, 212 SMs): calibrated on R200 against the per-row fastest
# of the ten admitted (body, TILE_V) instances over the 94 Qwen3.5 serving rows
# plus 17 synthetic batches; the shared table costs 1.064x of the
# per-row fastest there (1.268x at 1536 state heads), this table 1.001x.  The
# ``vec8occ`` instance at TILE_V=16 leads below 192 heads (one wave of 128-row
# CTAs; ``vec8``/``vec8r56`` 0.4-3.3 % behind), TILE_V=32 takes over at 193-256
# (``vec8``) and 257-368 / 417-3072 heads (``vec8occ``), 369-416 heads return to
# ``vec8occ`` at TILE_V=16, and the wide 128-row body only wins above 3072 heads
# (the shared table hands over at 768).
CAKE_GDN_BF16_T1_ROUTE_ARCH_BANDS: dict[
    str, tuple[tuple[int | None, str, int], ...]
] = {
    "sm_107a": (
        (192, "vec8occ", 16),
        (256, "vec8", 32),
        (368, "vec8occ", 32),
        (416, "vec8occ", 16),
        (768, "vec8", 32),
        (3072, "vec8occ", 32),
        (None, "wide", 128),
    ),
}


def cake_gdn_bf16_t1_route(
    batch_size: int, num_v_heads: int, arch: CakeGDNArch
) -> tuple[str, int]:
    """``(body, TILE_V)`` of the BF16-state T=1 decode route for ``batch_size * num_v_heads`` state heads on ``arch``."""

    state_heads = int(batch_size) * int(num_v_heads)
    bands = CAKE_GDN_BF16_T1_ROUTE_ARCH_BANDS.get(arch)
    overrides = (
        CAKE_GDN_BF16_T1_ROUTE_ARCH_BODIES.get(arch, {}) if bands is None else {}
    )
    if bands is None:
        bands = CAKE_GDN_BF16_T1_ROUTE_BANDS
    for max_state_heads, body, tile_v in bands:
        if max_state_heads is None or state_heads <= max_state_heads:
            return overrides.get(max_state_heads, body), tile_v
    raise AssertionError("BF16 T=1 route band tables must end with an open band")


# The full-warp tile-v16 BF16 verify body per architecture: Rubin (sm_107a) runs the
# `tile16_vpre` schedule (every draft token's v values staged in shared memory ahead
# of the serial recurrence, fully unrolled token loop; bitwise-identical output),
# B200/B300 keep the shipped body.  Mirrors
# the Cake decode kernel program's `select_bf16_verify_tile16_body`.
CAKE_GDN_BF16_VERIFY_TILE16_ARCH_BODIES: dict[str, str] = {
    "sm_107a": "tile16_vpre_es_pro"
}


def cake_gdn_bf16_verify_tile16_schedule(arch: CakeGDNArch) -> str:
    """Schedule attribute of the full-warp tile-v16 BF16 verify kernel on ``arch``."""
    body = CAKE_GDN_BF16_VERIFY_TILE16_ARCH_BODIES.get(arch, "tile16")
    return f"gdn_decode_pretranspose_t4_bf16state_{body}"


# The wide (TILE_V_WIDE 32/64/128) BF16-state MTP body of the multi-token
# verify / update / checkpoint rows, per architecture: sm_107a routes the
# `wide128_vpre` schedule (the draft window's v values prefetched before the
# barrier, token loop unrolled; bitwise-identical output), every other
# architecture and every T=1 band row the shipped wide body.  Mirrors
# `BF16_WIDE_ARCH_BODIES` / `select_bf16_wide_body` of the Cake exporter.
# sm_107a takes the `wide128_vpre_xchg` twin (exchange-based
# output path of `wide128_vpre`; bitwise-identical output) for the T >= 2 wide rows.
CAKE_GDN_BF16_WIDE_ARCH_BODIES: dict[str, str] = {"sm_107a": "wide128_vpre_xchg"}
# The promoted fp32-state MTP update body (B4 T4 with the intermediate-state cache,
# `..._mtp_t4_splitv8`) is routed per architecture as well: Rubin (sm_107a) runs its
# `_pro` twin (producer phase parallel over the draft tokens, every independent global
# load issued before the first reduction; bitwise-identical output), B200 / B300 keep
# the shipped body.  The verify body `..._mtp_t2_inline_tile8` runs its `_bb` twin on
# Rubin as well (softplus branch as a select, cache-row moves hoisted out of the
# slot-valid branch, merged output-store conditions; bitwise-identical output).
# Mirrors `FP32_MTP_ARCH_BODIES` / `select_fp32_mtp_body` in
# the Cake decode kernel program, keyed by (architecture, base schedule).
CAKE_GDN_FP32_MTP_ARCH_BODIES: dict[tuple[str, str], str] = {
    (
        "sm_107a",
        "gdn_decode_pretranspose_mtp_t4_splitv8",
    ): "gdn_decode_pretranspose_mtp_t4_splitv8_pro",
    # Adopted 2026-10-18, per record: the fp32
    # MTP tile64 record runs its ``_sl2`` state-load twin on sm_107a.
    (
        "sm_107a",
        "gdn_decode_pretranspose_mtp_t4_splitv2_tile64",
    ): "gdn_decode_pretranspose_mtp_t4_splitv2_tile64_sl2",
    (
        "sm_107a",
        "gdn_decode_pretranspose_mtp_t2_inline_tile8",
    ): "gdn_decode_pretranspose_mtp_t2_inline_tile8_bb",
}
# The fp32-state T=1 pretranspose body (`..._splitv8`, route `indexed_fp32_t1_splitv8`)
# is routed per architecture by the state-head count B*HV: Rubin (sm_107a) runs the
# shipped `splitv8` body (TILE_V 16, eight CTAs per state head) below 256 state heads
# and its `splitv8_tilev32` twin (TILE_V 32, four CTAs per state head; bitwise-identical
# output, only the row -> CTA mapping changes) from 256 state heads on: 5-8.5 % faster
# on R200 from 256 through 2048 state heads, 11-17 % slower at 32 and 128.  B200 / B300
# keep `splitv8` at every count.  Mirrors `FP32_T1_SPLITV8_TILEV32_MIN_STATE_HEADS` /
# `FP32_T1_SPLITV8_ARCH_BANDS` / `select_fp32_t1_splitv8_route` in
# the Cake decode kernel program (the Cake tree cross-checks these copies).
CAKE_GDN_FP32_T1_SPLITV8_TILEV32_MIN_STATE_HEADS = 256
# Rubin register-direct body `splitv8_reg` (eight CTAs per state head, state rows loaded
# straight into registers) in the band 64 < state heads < 256; `splitv8` up to 64 heads,
# the `splitv8_tilev32` twin from 256 on.  Bitwise-identical output, only the state load
# path changes.  Mirrors `FP32_T1_SPLITV8_REG_MIN_STATE_HEADS`.
# The band takes the `splitv8_reg_ost` twin of `splitv8_reg`
# (same loads, FMA and reduction order; the warp's four output rows are remapped contiguous so
# lane 0 issues one 64-bit store instead of four scalar bf16 stores).  Bitwise-identical output.
CAKE_GDN_FP32_T1_SPLITV8_REG_MIN_STATE_HEADS = 64
# (largest state-head count of the band or None for the open last band, body)
CAKE_GDN_FP32_T1_SPLITV8_ARCH_BANDS: dict[str, tuple[tuple[int | None, str], ...]] = {
    "sm_107a": (
        (CAKE_GDN_FP32_T1_SPLITV8_REG_MIN_STATE_HEADS, "splitv8"),
        (CAKE_GDN_FP32_T1_SPLITV8_TILEV32_MIN_STATE_HEADS - 1, "splitv8_reg_ost"),
        (None, "splitv8_tilev32"),
    ),
}
# CTAs per state head each fp32 T=1 body launches (the host grid is B * HV times this),
# keyed by the record's schedule: the resolved record carries its own grid instead of a
# fixed eight.
CAKE_GDN_FP32_T1_CTAS_PER_STATE: dict[str, int] = {
    "gdn_decode_pretranspose_splitv8": 8,
    "gdn_decode_pretranspose_splitv8_tilev32": 4,
    "gdn_decode_pretranspose_splitv8_reg_ost": 8,  # replaces the unrouted splitv8_reg entry
}


def cake_gdn_fp32_t1_route(batch_size: int, num_v_heads: int, arch: CakeGDNArch) -> str:
    """Body (``splitv8`` or ``splitv8_tilev32``) of the fp32-state T=1 pretranspose route for ``batch_size * num_v_heads`` state heads on ``arch``."""

    bands = CAKE_GDN_FP32_T1_SPLITV8_ARCH_BANDS.get(arch)
    if bands is None:
        return "splitv8"
    state_heads = int(batch_size) * int(num_v_heads)
    for max_state_heads, body in bands:
        if max_state_heads is None or state_heads <= max_state_heads:
            return body
    raise AssertionError("fp32 T=1 route band tables must end with an open band")


@functools.cache
def cake_gdn_fp32_t1_ctas_per_state(variant_name: str) -> int:
    """CTAs per state head the fp32-state T=1 pretranspose record ``variant_name`` launches (read from its schedule)."""

    record = _kernel_record(variant_name)
    schedule = record["source_schedule"].rsplit(":", 1)[-1]
    ctas_per_state = CAKE_GDN_FP32_T1_CTAS_PER_STATE.get(schedule)
    if ctas_per_state is None:
        raise CakeGDNUnsupportedError(
            f"{variant_name!r} ({schedule}) is not an fp32-state T=1 pretranspose record"
        )
    return ctas_per_state


# The fp32-state T=1 K-major (nontranspose) bodies (`direct_fp32_t1_nontranspose_*`) are
# routed per (architecture, rule band): the shipped `_small` body (eight CTAs per state
# head) below 32 batch rows and the shipped large body (one CTA per state head) from 32
# on, except on Rubin (sm_107a), which runs the `mid4` twin in the small band -- four CTAs
# per state head, each owning one Kx32 value panel over 128-byte state segments -- and the
# `tilev32_x2` twin in the large band's wave-fit range 32 <= B < 35 -- two CTAs per state
# head, each owning half of the V ring tiles (the single-CTA grid of 1024 .. 1088 CTAs on
# 212 SMs becomes 2048 .. 2176 CTAs of half the work; the split pays on this edge of the
# band only).  Both twins keep the per-lane statements and the accumulation order:
# bitwise-identical output and state.  B200 / B300 keep the shipped bodies.  The rule
# band folds onto the band of the shipped body / route id through the `_RULE_BANDS`
# table, so route ids keep naming `small` / `large` (plus the twin suffix).  Mirrors
# `NONTRANSPOSE_SMALL_BATCH_THRESHOLD` / `NONTRANSPOSE_LARGE_WAVEFIT_BATCH_LIMIT` /
# `NONTRANSPOSE_FP32_T1_RULE_BANDS` / `NONTRANSPOSE_FP32_T1_ARCH_BODIES` /
# `NONTRANSPOSE_FP32_T1_CTAS_PER_STATE` / `select_nontranspose_fp32_t1_body` in
# the Cake decode kernel program (the Cake tree cross-checks these copies).
CAKE_GDN_NONTRANSPOSE_FP32_T1_SMALL_BATCH_THRESHOLD = 32
CAKE_GDN_NONTRANSPOSE_FP32_T1_LARGE_WAVEFIT_BATCH_LIMIT = (
    37  # 35 -> 37; see the Cake launcher
)
CAKE_GDN_NONTRANSPOSE_FP32_T1_BASE_SCHEDULES: dict[str, str] = {
    "small": "gdn_decode_nontranspose_fp32_t1_small",
    "large": "gdn_decode_nontranspose_fp32_t1",
}
# (rule band -> band of the shipped body and of the route id)
CAKE_GDN_NONTRANSPOSE_FP32_T1_RULE_BANDS: dict[str, str] = {
    "small": "small",
    "large_wavefit": "large",
    "large": "large",
}
# ((architecture, rule band) -> schedule of the twin the rule routes there)
CAKE_GDN_NONTRANSPOSE_FP32_T1_ARCH_BODIES: dict[tuple[str, str], str] = {
    ("sm_107a", "small"): "gdn_decode_nontranspose_fp32_t1_mid4",
    ("sm_107a", "large_wavefit"): "gdn_decode_nontranspose_fp32_t1_tilev32_x2",
    # Adopted 2026-10-18: the large band
    # (B >= 37) runs the ``hqsplit`` recurrence twin on sm_107a.
    ("sm_107a", "large"): "gdn_decode_nontranspose_fp32_t1_hqsplit",
}
# CTAs per state head each nontranspose fp32 T=1 body launches (the host grid is B * HV
# times this), keyed by the record's schedule.
CAKE_GDN_NONTRANSPOSE_FP32_T1_CTAS_PER_STATE: dict[str, int] = {
    "gdn_decode_nontranspose_fp32_t1_small": 8,
    "gdn_decode_nontranspose_fp32_t1": 1,
    "gdn_decode_nontranspose_fp32_t1_mid4": 4,
    "gdn_decode_nontranspose_fp32_t1_tilev32_x2": 2,
    "gdn_decode_nontranspose_fp32_t1_hqsplit": 1,
}


def cake_gdn_nontranspose_fp32_t1_rule_band(batch_size: int) -> str:
    """Rule band of ``batch_size`` rows: ``small`` (B < 32), ``large_wavefit`` (32 <= B < 37) or ``large``."""

    if int(batch_size) < CAKE_GDN_NONTRANSPOSE_FP32_T1_SMALL_BATCH_THRESHOLD:
        return "small"
    if int(batch_size) < CAKE_GDN_NONTRANSPOSE_FP32_T1_LARGE_WAVEFIT_BATCH_LIMIT:
        return "large_wavefit"
    return "large"


def cake_gdn_nontranspose_fp32_t1_route(
    batch_size: int, arch: CakeGDNArch
) -> tuple[str, str]:
    """``(schedule, route id)`` of the fp32-state T=1 K-major decode for ``batch_size`` rows on ``arch``."""

    rule_band = cake_gdn_nontranspose_fp32_t1_rule_band(batch_size)
    band = CAKE_GDN_NONTRANSPOSE_FP32_T1_RULE_BANDS[rule_band]
    base = CAKE_GDN_NONTRANSPOSE_FP32_T1_BASE_SCHEDULES[band]
    schedule = CAKE_GDN_NONTRANSPOSE_FP32_T1_ARCH_BODIES.get((arch, rule_band), base)
    route = f"flashinfer.gdn_decode.direct_fp32_t1_nontranspose_{band}"
    if schedule != base:
        route += "_" + schedule.removeprefix("gdn_decode_nontranspose_fp32_t1_")
    return schedule, route


@functools.cache
def cake_gdn_nontranspose_fp32_t1_ctas_per_state(variant_name: str) -> int:
    """CTAs per state head the fp32-state T=1 K-major record ``variant_name`` launches (read from its schedule)."""

    record = _kernel_record(variant_name)
    schedule = record["source_schedule"].rsplit(":", 1)[-1]
    ctas_per_state = CAKE_GDN_NONTRANSPOSE_FP32_T1_CTAS_PER_STATE.get(schedule)
    if ctas_per_state is None:
        raise CakeGDNUnsupportedError(
            f"{variant_name!r} ({schedule}) is not an fp32-state T=1 nontranspose record"
        )
    return ctas_per_state


def cake_gdn_bf16_wide_schedule(arch: CakeGDNArch, seq_len: int) -> str:
    if int(seq_len) <= 1:
        return "gdn_decode_pretranspose_mtp_t4_bf16state_wide128"
    body = CAKE_GDN_BF16_WIDE_ARCH_BODIES.get(arch, "wide128")
    return f"gdn_decode_pretranspose_mtp_t4_bf16state_{body}"


def cake_gdn_bf16_route_tile_v(route_id: str) -> int:
    """The grid tile a BF16 decode route id carries (``.vec8_t<N>``, ``.vec8r56_t<N>``, ``.vec8occ_t<N>``, ``.wide<N>`` or ``.tile16_fullwarp``)."""

    if route_id.endswith(".tile16_fullwarp"):
        return 16
    match = re.fullmatch(
        r".*\.(?:vec8_t|vec8r56_t|vec8occ_t|wide)(16|32|64|128)", route_id
    )
    if match is None:
        raise CakeGDNUnsupportedError(
            f"BF16 decode route {route_id!r} carries no grid tile"
        )
    return int(match.group(1))


@functools.lru_cache(maxsize=1024)
def select_cake_gdn_decode_variant(
    *,
    arch: CakeGDNArch,
    batch_size: int,
    io_dtype: str,
    state_dtype: str,
    head_size: int,
    layout: str,
    num_k_heads: int,
    num_q_heads: int,
    num_v_heads: int,
    scale: float,
    seq_len: int,
    use_qk_l2norm: bool,
    strided_inputs: bool = False,
    disable_state_update: bool = False,
    cache_intermediate_states: bool = False,
    cache_steps: int = 0,
) -> CakeGDNRoute:
    """Resolve one frozen FP32 T=1/MTP row, one BF16 T=1 serving geometry or one exact promoted BF16 verify/update row."""

    if arch not in _ARCH_ACTIVE_CLUSTERS:
        raise CakeGDNUnsupportedError(f"unsupported architecture {arch}")
    if io_dtype != "bfloat16" or state_dtype not in {"float32", "bfloat16"}:
        raise CakeGDNUnsupportedError(
            "GDN non-CP decode requires BF16 I/O and FP32 or BF16 state"
        )
    if head_size != 128:
        raise CakeGDNUnsupportedError("GDN non-CP decode requires K=V=128")
    if not use_qk_l2norm:
        raise CakeGDNUnsupportedError(
            "GDN non-CP T=1 child contract requires in-kernel Q/K L2 normalization"
        )
    if (
        num_q_heads <= 0
        or num_k_heads != num_q_heads
        or num_v_heads <= 0
        or num_v_heads % num_q_heads
    ):
        raise CakeGDNUnsupportedError("unsupported decode head mapping")
    if batch_size <= 0:
        raise CakeGDNUnsupportedError("decode batch size must be positive")
    if state_dtype == "bfloat16":
        if layout != "pretranspose":
            raise CakeGDNUnsupportedError(
                "BF16 decode requires the pretranspose state-pool layout"
            )
        if seq_len == 1:
            # Single-token serving decode: every per-rank (H, HV) geometry at
            # any batch (Qwen3.5-35B TP1/TP2/TP4, Qwen3.5-397B TP2/TP4/TP8);
            # every T=1 body takes its strides at runtime.  The body and grid
            # tile follow the state-head count; an unlisted (geometry, body,
            # tile) fails closed in _variant_for.
            if disable_state_update or cache_intermediate_states or cache_steps:
                raise CakeGDNUnsupportedError(
                    "BF16 T=1 decode updates the state and caches nothing"
                )
            body, tile_v = cake_gdn_bf16_t1_route(batch_size, num_v_heads, arch)
            if body == "wide":
                record = _variant_for(
                    arch=arch,
                    domain="decode",
                    schedule_attr="gdn_decode_pretranspose_mtp_t4_bf16state_wide128",
                    specializations={
                        "CACHE_INTERMEDIATE_STATES": 0,
                        "H": num_q_heads,
                        "HV": num_v_heads,
                        "INTERMEDIATE_BATCH_STRIDE": 128 * 128,
                        "INTERMEDIATE_TOKEN_STRIDE": 128 * 128,
                        "SCALE": scale,
                        "STRIDED_INPUTS": 1,
                        "TILE_V_WIDE": tile_v,
                        "T_STEPS": 1,
                        "UPDATE_STATE": 1,
                    },
                )
                return CakeGDNRoute(
                    f"flashinfer.gdn_decode.indexed_bf16_t1.wide{tile_v}",
                    record["name"],
                )
            record = _variant_for(
                arch=arch,
                domain="decode",
                schedule_attr=f"gdn_decode_pretranspose_t1_bf16state_{body}",
                specializations={
                    "H": num_q_heads,
                    "HV": num_v_heads,
                    "SCALE": scale,
                    "STRIDED_INPUTS": 1,
                    "TILE_V": tile_v,
                },
            )
            return CakeGDNRoute(
                f"flashinfer.gdn_decode.indexed_bf16_t1.{body}_t{tile_v}",
                record["name"],
            )
        promoted = {
            (4, 2, 16, 32, False, True, True, 4),
            (8, 3, 16, 64, True, True, True, 3),
            (8, 4, 16, 64, True, True, True, 4),
            (8, 4, 16, 32, True, True, True, 4),
            (1, 4, 4, 8, True, True, True, 4),
            (2, 4, 4, 8, True, True, True, 4),
            (3, 4, 4, 8, True, True, True, 4),
            (4, 4, 4, 8, True, True, True, 4),
            (5, 4, 4, 8, True, True, True, 4),
            (5, 4, 4, 8, False, True, True, 4),
            (6, 4, 4, 8, True, True, True, 4),
            (7, 4, 4, 8, True, True, True, 4),
            (8, 4, 4, 8, True, True, True, 4),
            (8, 2, 16, 64, True, False, False, 0),
            (8, 4, 16, 64, True, False, True, 5),
            (1, 7, 8, 16, True, True, True, 7),
            (2, 7, 8, 16, True, True, True, 7),
            (3, 7, 8, 16, True, True, True, 7),
            (4, 7, 8, 16, True, True, True, 7),
            (5, 7, 8, 16, True, True, True, 7),
            (6, 7, 8, 16, True, True, True, 7),
            (7, 7, 8, 16, True, True, True, 7),
            (8, 7, 8, 16, True, True, True, 7),
            (1, 8, 8, 16, True, True, True, 8),
            (1, 7, 16, 32, True, True, True, 7),
            # SGLang hands the verify step contiguous [B, T, H, K] views; the
            # STRIDED_INPUTS=1 variants above serve them through runtime strides.
            (1, 7, 8, 16, False, True, True, 7),
            (2, 7, 8, 16, False, True, True, 7),
            (3, 7, 8, 16, False, True, True, 7),
            (4, 7, 8, 16, False, True, True, 7),
            (5, 7, 8, 16, False, True, True, 7),
            (6, 7, 8, 16, False, True, True, 7),
            (7, 7, 8, 16, False, True, True, 7),
            (8, 7, 8, 16, False, True, True, 7),
            (1, 8, 8, 16, False, True, True, 8),
            (1, 7, 16, 32, False, True, True, 7),
        }
        key = (
            batch_size,
            seq_len,
            num_q_heads,
            num_v_heads,
            strided_inputs,
            disable_state_update,
            cache_intermediate_states,
            cache_steps,
        )
        if key not in promoted:
            raise CakeGDNUnsupportedError(
                "BF16 multi-token decode is limited to the exact promoted verify/update rows"
            )
        if num_q_heads == 8 and num_v_heads == 16 and batch_size <= 4:
            # Qwen3.5-35B-A3B TP=2 per-rank verify (speculative_num_draft_tokens=7
            # verifies T=7; T=8 is the adjacent window): B<=4 runs the full-warp
            # tile-v16 kernel with T_STEPS specialized (the `tile16_vpre` body on
            # sm_107a), B>=5 falls through to wide32.
            record = _variant_for(
                arch=arch,
                domain="decode",
                schedule_attr=cake_gdn_bf16_verify_tile16_schedule(arch),
                specializations={
                    "H": num_q_heads,
                    "HV": num_v_heads,
                    "INTERMEDIATE_BATCH_STRIDE": cache_steps * num_v_heads * 128 * 128,
                    "INTERMEDIATE_TOKEN_STRIDE": num_v_heads * 128 * 128,
                    "SCALE": scale,
                    "STRIDED_INPUTS": 1,
                    "T_STEPS": seq_len,
                },
            )
            return CakeGDNRoute(
                f"flashinfer.gdn_decode.indexed_bf16_verify_t{seq_len}.tile16_fullwarp",
                record["name"],
            )
        if num_q_heads == 4 and num_v_heads == 8:
            schedule_attr = cake_gdn_bf16_verify_tile16_schedule(arch)
            specializations = {
                "H": num_q_heads,
                "HV": num_v_heads,
                "INTERMEDIATE_BATCH_STRIDE": (cache_steps * num_v_heads * 128 * 128),
                "INTERMEDIATE_TOKEN_STRIDE": num_v_heads * 128 * 128,
                "SCALE": scale,
                "STRIDED_INPUTS": 1,
            }
            route = "flashinfer.gdn_decode.indexed_bf16_verify_t4.tile16_fullwarp"
            record = _variant_for(
                arch=arch,
                domain="decode",
                schedule_attr=schedule_attr,
                specializations=specializations,
            )
            return CakeGDNRoute(route, record["name"])

        state_heads = batch_size * num_v_heads
        tile_v = 128 if state_heads >= 1024 else 64 if state_heads >= 512 else 32
        update_state = not cache_intermediate_states
        record = _variant_for(
            arch=arch,
            domain="decode",
            schedule_attr=cake_gdn_bf16_wide_schedule(arch, seq_len),
            specializations={
                "CACHE_INTERMEDIATE_STATES": int(cache_intermediate_states),
                "H": num_q_heads,
                "HV": num_v_heads,
                "INTERMEDIATE_BATCH_STRIDE": (
                    cache_steps * num_v_heads * 128 * 128
                    if cache_intermediate_states
                    else 128 * 128
                ),
                "INTERMEDIATE_TOKEN_STRIDE": (
                    num_v_heads * 128 * 128 if cache_intermediate_states else 128 * 128
                ),
                "SCALE": scale,
                # The T>=7 wide32 variants exist only as STRIDED_INPUTS=1 and take
                # their strides at runtime, so contiguous callers use them too.
                "STRIDED_INPUTS": int(strided_inputs or seq_len >= 7),
                "TILE_V_WIDE": tile_v,
                "T_STEPS": seq_len,
                "UPDATE_STATE": int(update_state),
            },
        )
        if disable_state_update:
            route = f"flashinfer.gdn_decode.indexed_bf16_verify_t{seq_len}"
        elif cache_intermediate_states:
            route = f"flashinfer.gdn_decode.indexed_bf16_checkpoint_t{seq_len}"
        else:
            route = f"flashinfer.gdn_decode.indexed_bf16_update_t{seq_len}"
        return CakeGDNRoute(f"{route}.wide{tile_v}", record["name"])

    if seq_len != 1:
        promoted_fp32_mtp = {
            (1, 2, True, True, 2): (
                "gdn_decode_pretranspose_mtp_t2_inline_tile8",
                "flashinfer.gdn_decode.indexed_fp32_mtp_t2.inline_tile8_verify_cache",
            ),
            (4, 4, True, False, 4): (
                "gdn_decode_pretranspose_mtp_t4_splitv8",
                "flashinfer.gdn_decode.indexed_fp32_mtp_t4.splitv8_update_cache",
            ),
            (16, 4, True, False, 4): (
                "gdn_decode_pretranspose_mtp_t4_splitv2_tile64",
                "flashinfer.gdn_decode.indexed_fp32_mtp_t4.tile64_update_cache",
            ),
            (64, 4, True, False, 4): (
                "gdn_decode_pretranspose_mtp_t4_splitv2_tile64",
                "flashinfer.gdn_decode.indexed_fp32_mtp_t4.tile64_update_cache",
            ),
        }
        fp32_mtp_key = (
            batch_size,
            seq_len,
            strided_inputs,
            disable_state_update,
            cache_steps,
        )
        selected = promoted_fp32_mtp.get(fp32_mtp_key)
        if (
            layout != "pretranspose"
            or (num_q_heads, num_k_heads, num_v_heads) != (16, 16, 32)
            or not cache_intermediate_states
            or selected is None
        ):
            raise CakeGDNUnsupportedError(
                "FP32 MTP decode is limited to promoted B1/T2 verify and "
                "B4/B16/B64 T4 update rows"
            )
        base_attr, route = selected
        schedule_attr = CAKE_GDN_FP32_MTP_ARCH_BODIES.get((arch, base_attr), base_attr)
        mtp_specializations: dict[str, int | float] = {
            "H": num_q_heads,
            "HV": num_v_heads,
            "INTERMEDIATE_BATCH_STRIDE": cache_steps * num_v_heads * 128 * 128,
            "INTERMEDIATE_TOKEN_STRIDE": num_v_heads * 128 * 128,
            "SCALE": scale,
            "STRIDED_INPUTS": 1,
        }
        if base_attr != "gdn_decode_pretranspose_mtp_t2_inline_tile8":
            mtp_specializations.update(
                {
                    "CACHE_INTERMEDIATE_STATES": 1,
                    "T_STEPS": seq_len,
                    "UPDATE_STATE": 1,
                }
            )
        record = _variant_for(
            arch=arch,
            domain="decode",
            schedule_attr=schedule_attr,
            specializations=mtp_specializations,
        )
        return CakeGDNRoute(route, record["name"])
    if disable_state_update or cache_intermediate_states or cache_steps:
        raise CakeGDNUnsupportedError(
            "FP32-state T=1 child contract requires state update and no cache"
        )
    if layout == "pretranspose":
        # The body follows the state-head count per architecture (sm_107a runs the
        # `splitv8_tilev32` twin from 256 state heads); the route id names a twin and
        # the host reads the record's CTAs per state head for the grid.
        body = cake_gdn_fp32_t1_route(batch_size, num_v_heads, arch)
        schedule_attr = f"gdn_decode_pretranspose_{body}"
        route = "flashinfer.gdn_decode.indexed_fp32_t1_splitv8" + body.removeprefix(
            "splitv8"
        )
    elif layout == "nontranspose":
        # The body follows the batch band per architecture (sm_107a runs the `mid4`
        # twin below 32 rows); the route id names a twin and the host reads the
        # record's CTAs per state head for the grid.
        schedule_attr, route = cake_gdn_nontranspose_fp32_t1_route(batch_size, arch)
    else:
        raise CakeGDNUnsupportedError(f"unsupported decode state layout {layout}")
    record = _variant_for(
        arch=arch,
        domain="decode",
        schedule_attr=schedule_attr,
        specializations={"H": num_q_heads, "HV": num_v_heads, "SCALE": scale},
    )
    return CakeGDNRoute(route, record["name"])


def arch_for_compute_capability(major: int, minor: int) -> CakeGDNArch:
    if (major, minor) == (10, 0):
        return "sm_100a"
    if (major, minor) == (10, 3):
        return "sm_103a"
    if (major, minor) == (10, 7):
        return "sm_107a"
    raise CakeGDNUnsupportedError(
        "Cake GDN supports only SM100a/SM103a/SM107a, "
        f"got compute capability {major}.{minor}"
    )


__all__ = [
    "CakeGDNArch",
    "CakeGDNRoute",
    "CakeGDNUnsupportedError",
    "arch_for_compute_capability",
    "cake_gdn_prefill_geometries",
    "cake_gdn_prefill_seq_schedule",
    "compile_cake_gdn_cubin",
    "load_cake_gdn_kernel",
    "render_cake_gdn_host_shim",
    "select_cake_gdn_decode_variant",
    "select_cake_gdn_prefill_variant",
]
