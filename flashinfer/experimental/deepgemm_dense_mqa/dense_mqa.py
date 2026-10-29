"""Prepared dense FP4/FP8 MQA lightning-indexer logits on SM100a/SM103a.

Production runtime has no source compiler, quantizer or native oracle dependency.
Plans bind user tensors once; run() submits the route's generated programs on
the current PyTorch stream without allocating, graph capture, or a fallback.

Routes are selected from host-known scalars only (precision, head count, query
count, KV length range); the generated programs take the query count and the KV
length at runtime, so one program per physical schedule serves every catalogued
architecture. Every program takes the launch grid of the logits consumers (the
device's SM count, or a plan's CTA-budget override) as the compile-line
definition ``SM_COUNT``: the per-SM cost partition and the metadata offsets are
compile-time literals in every build, and one source text per program serves
every SM count. The paged p32/p64 and n1/n2 programs of one head count and
scheduler share one parameterised text whose catalog record fixes the page size
and the token count on the same compile line (``fixed_definitions``:
``K_BLOCK_KV``, ``K_NEXT_N``).

Head counts: the catalog policy lists the exported head counts (``heads``);
``BLOCK_Q = 128 // num_heads`` queries share one 128-row MMA tile (4 for 32
heads, 2 for 64 heads). The one-shot ``fp8_mqa_logits`` mirrors DeepGEMM's
``fp8_mqa_logits`` signature on top of :class:`DenseMqaPlan`.

32-head routes come in two launch shapes: the metadata program followed by the
logits program (the ``q1`` and ``<tier>`` routes), or ONE logits launch that
computes the schedule in-kernel (the ``:short`` routes: the fused single-query
chain ``q1:short``, the warp-scan classes ``q128:short`` / ``le32:short`` and
the CTA-scan classes ``le768:short`` (FP8) / ``le1024:short`` (FP4)). A class
is named by its query-BLOCK ceiling and served while the KV length is within
the catalog policy's bound for that class (``fused_*_max_kv``); above the
bound the two-launch route of the same query count serves the call. The FP4
single-launch programs carry a first-tile L2 prefetch of the CTA's first KV tile
and the block's Q / scale / weight boxes (identical logits and schedule words) on
the forms the per-architecture policy list ``policy.fp4.first_tile_prefetch_forms``
names: ``q1`` (``fp4:q1:short``; both architectures), ``le32`` (``fp4:le32:short``;
sm_103a only -- one route, the route record binds the prefetch program on sm_103a
and the plain warp-scan program on sm_100a) and ``qle1024``: where it is listed
(both architectures) the FP4 ``le1024`` class ships two programs,
``fp4:qle1024:short`` for ``Q <= policy.fp4.first_tile_prefetch_max_queries_scan1024``
(the CTA-scan text with the prefetch) and ``fp4:le1024:short`` above that query
count.

64-head routes are ONE logits launch each: the grid-stride production text
(the ``q1`` and ``<tier>`` routes, no metadata buffer), its one-split text
(the ``:short`` routes: ``K <= policy.dense_short.max_kv`` and at least two
query blocks), its tile-schedule-words text (the ``:ts`` routes: the per-arch
production text whose load_q lane publishes each Q block's window words through
the Q stage, no metadata buffer; not a ``:short`` shape,
``policy.dense_ts.min_kv_exclusive < K <= dense_ts.max_kv``,
``Q <= dense_ts.max_queries`` and at least ``dense_ts.min_q_blocks`` query
blocks) or -- where the catalog carries ``policy.dense_kvsplit`` -- the
production text under its in-kernel KV-split plan (the ``:kvsplit`` routes:
``K > dense_kvsplit.min_kv`` and fewer query blocks than the architecture's
catalogued SM count ``dense_kvsplit.sm_count[arch]``, or -- where
``dense_kvsplit.fullgrid[arch]`` -- ``K > dense_kvsplit.fullgrid_min_kv`` and
``sm_count[arch] <= ceil(Q / 2) <= dense_kvsplit.max_q_blocks``, the ``qle2048``
plan's whole-grid band; the plan text of the
query count -- ``q1`` / ``qle128`` / ``qle2048`` -- recomputes the tiered
schedule in every CTA and writes it into the plan's own ``metadata`` buffer,
launch grid = the SM count).  The KV-split gate is the only SM-count dependent
rule and is applied at the catalogued count, not the device's; it is consulted
before the ``:short`` / ``:ts`` shape gates, so a call inside both it and the
``:ts`` gate is named by its KV-split plan.

A ``:kvsplit`` route record of the full kind may carry ``stage_buckets``
(``{arch: [{"kv_max": K, "stages": [[stage, program], ...]}, ...]}``): a plan
whose KV length is at most an entry's ``kv_max`` launches the first such entry's
stages on that architecture (:func:`route_stages`) -- its partial-kind twin's
program, with the same host call, logits and schedule words as the record's
program; the route name, its admission and its metadata buffer are the record's.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

_ARCHES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
CATALOG_SCHEMA = "dense_mqa.v13"
NUM_HEADS = 32  # default head count of the public helpers
BLOCK_QH = 128  # query rows x heads per MMA tile
BLOCK_Q = BLOCK_QH // NUM_HEADS
BLOCK_KV = 256
HEAD_DIM = 128
# Query ceiling of the single-launch fused Q <= 128 route (``fp8:qle128:short``): the ``le128`` tier's token ceiling
# (32 scan lanes x BLOCK_Q); its KV bound is the catalog policy ``fused_qle128_max_kv``.
FUSED_QLE128_MAX_Q = 128
# Query ceiling of the single-launch fused Q <= 2048 route (``fp8:qle2048:short``, 128 < Q <= 2048): the ``le2048``
# tier's token ceiling (512 scan entries x BLOCK_Q); its KV bound is the catalog policy ``fused_qle2048_max_kv`` -- a
# per-architecture map (34048 on sm_100a, 65536 on sm_103a; ``_fused_qle2048_bounds``), applied by ``served_route``.
# Both fused tiered routes are consulted in the order ``policy.single_launch_precedence[arch][tier]`` against the
# dense_mqa.v9 scan classes of the same query band (``_single_launch_precedence``).
FUSED_QLE2048_MAX_Q = 2048


@functools.cache
def _catalog():
    catalog = json.loads(Path(__file__).with_name("dense_mqa_catalog.json").read_text())
    if catalog.get("schema") != CATALOG_SCHEMA:
        raise RuntimeError(
            f"dense MQA catalog schema {catalog.get('schema')!r} is not {CATALOG_SCHEMA!r}; regenerate the export"
        )
    return catalog


def _fused_qle2048_bounds():
    """``policy.fused_qle2048_max_kv``: the KV bound of the single-launch fused ``fp8:qle2048:short`` route PER
    ARCHITECTURE (``{"sm_100a": 34048, "sm_103a": 65536}``; the one per-arch policy key -- each bound was measured on
    its device)."""
    return {a: int(v) for a, v in _catalog()["policy"]["fused_qle2048_max_kv"].items()}


@functools.cache
def device_facts(device_index):
    """``(arch, sm_count)`` of one CUDA device through FlashInfer's cached device queries."""
    import torch

    from flashinfer.utils import get_compute_capability, get_device_sm_count

    device = torch.device("cuda", device_index)
    capability = get_compute_capability(device)
    arch = _ARCHES.get(capability)
    catalogued = sorted(_catalog()["arches"])
    if arch is None or arch not in catalogued:
        raise RuntimeError(
            f"Dense MQA has no exported programs for compute capability "
            f"{capability}; catalogued architectures: {catalogued}"
        )
    return arch, int(get_device_sm_count(device))


def device_arch(device):
    """Generated-program architecture for ``device`` (raises when none is catalogued)."""
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("Dense MQA requires a CUDA device")
    index = device.index if device.index is not None else torch.cuda.current_device()
    return device_facts(index)[0]


def _nvcc_flags(arch):
    from flashinfer.jit.core import sm100a_nvcc_flags, sm103a_nvcc_flags

    return {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}[arch]


def stage_program(stage, arch):
    """The program one route stage launches on ``arch``; ``stage`` is one ``[name, program]`` entry of a route
    record's ``stages`` -- a plain program name where the architectures share one text, a ``{arch: program}``
    map where the production text is chosen per architecture."""
    name, by_arch = stage
    if isinstance(by_arch, str):
        return by_arch
    try:
        return by_arch[arch]
    except KeyError:
        raise RuntimeError(
            f"route stage {name} has no program for {arch}; catalogued architectures: {sorted(by_arch)}"
        ) from None


def stage_programs(stage, arches):
    """``{arch: program}`` of one route stage: the record's own map, or -- for a stage that names one
    program -- that program on every architecture in ``arches``."""
    _name, by_arch = stage
    if isinstance(by_arch, str):
        return dict.fromkeys(arches, by_arch)
    return dict(by_arch)


def route_stages(record, arch, keys):
    """The stages a route record launches on ``arch`` for the KV length ``keys`` (``kv.shape[0]``): the first
    ``stage_buckets[arch]`` entry with ``keys <= kv_max`` (``[[stage, program], ...]``, one program name per
    stage), else the record's ``stages``."""
    for bucket in record.get("stage_buckets", {}).get(arch, ()):
        if keys <= int(bucket["kv_max"]):
            return bucket["stages"]
    return record["stages"]


def program_names(route=None, arch=None):
    """Names of the catalogued programs: all of them, those a route may launch on ``arch`` (stage order, then
    its ``stage_buckets`` programs for ``arch``, then the sequence; a plan's ``program_names`` are the ones it
    launches), or -- without ``arch`` -- the sorted union over the architectures."""
    catalog = _catalog()
    if route is None:
        return sorted(catalog["programs"])
    record = catalog["routes"].get(route) or catalog["paged_routes"][route]
    buckets = record.get("stage_buckets", {})
    if arch is None:
        names = sorted(
            {
                program
                for stage in record["stages"]
                for program in stage_programs(stage, catalog["arches"]).values()
            }
            | {
                program
                for entries in buckets.values()
                for entry in entries
                for _stage, program in entry["stages"]
            }
        )
    else:
        names = [stage_program(stage, arch) for stage in record["stages"]]
        for entry in buckets.get(arch, ()):
            for _stage, program in entry["stages"]:
                if program not in names:
                    names.append(program)
    if record.get("sequence"):
        names.append(record["sequence"])
    return names


def program_spec(arch, name, num_sms):
    """FlashInfer JIT build specification of one generated program for ``arch`` and the launch grid ``num_sms``
    (the compile-line definition ``SM_COUNT`` of every program -- the SM count, or a plan's CTA-budget
    override -- and the ``fixed_definitions`` of a record built from a shared parameterised text, the
    paged ``K_BLOCK_KV`` page size and ``K_NEXT_N`` token count; the name carries every supplied
    value)."""
    from flashinfer.jit import env
    from flashinfer.jit.core import gen_jit_spec

    record = _catalog()["programs"][name]
    if arch not in record["arches"]:
        raise RuntimeError(f"program {name} is not exported for {arch}")
    definitions = {
        name: int(value) for name, value in record.get("fixed_definitions", {}).items()
    }
    definitions["SM_COUNT"] = int(num_sms)
    suffix = "".join(
        f"_{key.lower()}{value}" for key, value in sorted(definitions.items())
    )
    return gen_jit_spec(
        name=f"{name}_{arch}{suffix}",
        sources=[
            env.FLASHINFER_CSRC_DIR / p.removeprefix("csrc/") for p in record["sources"]
        ],
        extra_cuda_cflags=[
            *_nvcc_flags(arch),
            *record["compile_flags"],
            *(f"-D{key}={value}" for key, value in sorted(definitions.items())),
            "--device-entity-has-hidden-visibility=false",
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[env.FLASHINFER_CSRC_DIR, env.FLASHINFER_INCLUDE_DIR],
        use_fast_math=False,  # Only explicit stage flags select fast math.
    )


@functools.cache
def load_program(arch, name, num_sms):
    record = _catalog()["programs"][name]
    spec = program_spec(arch, name, num_sms)
    module = spec.build_and_load()
    return module, {**record, "library_path": str(spec.get_library_path())}


def heads():
    """Exported head counts (catalog policy)."""
    return tuple(int(h) for h in _catalog()["policy"]["heads"])


def block_q(num_heads=NUM_HEADS):
    """Queries per 128-row MMA tile for ``num_heads`` (the query block of the schedule)."""
    if num_heads not in heads():
        raise ValueError(f"num_heads must be one of {heads()}, got {num_heads}")
    return BLOCK_QH // int(num_heads)


def _route_prefix(precision, num_heads):
    return precision if num_heads == NUM_HEADS else f"{precision}:h{num_heads}"


def _overlap_tier(queries):
    """The overlap tier of a 32-head query count -- ``le128`` (1 < Q <= 128) or ``le2048`` (128 < Q <= 2048), the two bands
    on which BOTH single-launch families have a program -- or ``None`` (Q = 1 is the fused Q1 chain's, Q > 2048 the le768
    scan class's alone)."""
    if queries <= 1:
        return None
    if queries <= FUSED_QLE128_MAX_Q:
        return "le128"
    if queries <= FUSED_QLE2048_MAX_Q:
        return "le2048"
    return None


def _single_launch_precedence(arch, tier):
    """``policy.single_launch_precedence[arch][tier]``: the consultation order of the two single-launch families of the
    32-head FP8 text -- ``le_class`` (the in-kernel scan classes ``fp8:{full,partial}:le{32,768}:short``) and ``qle`` (the
    fused tiered texts ``fp8:qle128:short`` / ``fp8:qle2048:short``) -- on an overlap tier; outside the tiers the scan
    classes alone.  The first family whose own KV bound admits the call serves it (the published catalog order is
    ``["qle", "le_class"]`` on both tiers and architectures: the fused tiered text while K is within its bound, the
    scan class above it; ``["le_class", "qle"]`` would restore the v9 routing)."""
    if tier is None:
        return ("le_class",)
    table = _catalog()["policy"]["single_launch_precedence"]
    if arch not in table:
        raise ValueError(f"arch must be one of {sorted(table)}, got {arch!r}")
    return tuple(table[arch][tier])


def _route_name_on(precision, queries, keys, num_heads, arch, *, qle2048_max_kv=None):
    """The route of a problem under ONE architecture's rule -- its single-launch precedence and its fused Q <= 2048 bound
    (``qle2048_max_kv`` overrides that bound: the logical name applies the most permissive one).  Host scalars only, no
    K-rule check (``_keys_served``)."""
    policy = _catalog()["policy"]
    if precision not in ("fp4", "fp8"):
        raise ValueError("precision must be 'fp4' or 'fp8'")
    prefix = _route_prefix(precision, num_heads)
    kind = "full" if queries % block_q(num_heads) == 0 else "partial"
    if precision == "fp4":
        if num_heads == NUM_HEADS:
            fp4 = policy["fp4"]
            if queries == 1:
                if keys <= int(fp4["fused_q1_max_kv"]):
                    return f"{prefix}:q1:short"
                return f"{prefix}:q1"
            scan = _fused_scan_class(queries, keys, fp4)
            if scan:
                # the le1024 class is served by its first-tile-prefetch program at and below the
                # policy's query bound and by the plain CTA-scan program above it (one class, one
                # K bound, two programs) on the architectures whose
                # ``policy.fp4.first_tile_prefetch_forms`` list ``qle1024``; the le32 class is one
                # route whose program the route record binds per architecture
                split = "qle1024" in fp4["first_tile_prefetch_forms"][arch]
                if (
                    scan == 1024
                    and split
                    and queries <= int(fp4["first_tile_prefetch_max_queries_scan1024"])
                ):
                    return f"{prefix}:qle1024:short"
                return f"{prefix}:le{scan}:short"
        return (
            f"{prefix}:q1"
            if queries == 1
            else f"{prefix}:{metadata_tier(queries, num_heads)}"
        )
    if num_heads != NUM_HEADS:
        # the in-kernel KV-split plan of a gated 64-head call (K > dense_kvsplit.min_kv: disjoint from the one-split
        # rule; consulted before the ``:ts`` shape gate, whose 256 < K <= 2048 band it overlaps below the SM count)
        plan = h64_kvsplit_plan(queries, keys, num_heads, arch=arch)
        if plan == "q1":
            return f"{prefix}:q1:kvsplit"
        if plan is not None:
            return f"{prefix}:{kind}:{plan}:kvsplit"
    if queries == 1:
        if num_heads == NUM_HEADS and keys <= policy["fused_q1_max_kv"]:
            return f"{prefix}:q1:short"
        return f"{prefix}:q1"
    if num_heads == NUM_HEADS:
        if queries == 128 and keys <= policy["fused_q128_max_kv"]:
            return f"{prefix}:q128:short"
        tier = _overlap_tier(queries)
        for family in _single_launch_precedence(arch, tier):
            if family == "le_class":
                # Q = 128 is owned by the fused-Q128 rule: above its bound it takes the
                # two-launch pair, never the le32 class.
                scan = 0 if queries == 128 else _fused_scan_class(queries, keys, policy)
                if scan:
                    return f"{prefix}:{kind}:le{scan}:short"
            elif family == "qle":
                if tier == "le128" and keys <= int(policy["fused_qle128_max_kv"]):
                    return f"{prefix}:qle128:short"
                if tier == "le2048":
                    bound = (
                        _fused_qle2048_bounds()[arch]
                        if qle2048_max_kv is None
                        else int(qle2048_max_kv)
                    )
                    if keys <= bound:
                        return f"{prefix}:qle2048:short"
            else:
                raise ValueError(
                    f"unknown single-launch family {family!r} in policy.single_launch_precedence"
                )
    if num_heads != NUM_HEADS and h64_short_shape(queries, keys, num_heads):
        return f"{prefix}:{kind}:short"
    if num_heads != NUM_HEADS and h64_ts_shape(queries, keys, num_heads):
        return f"{prefix}:{kind}:ts"
    return f"{prefix}:{kind}:{metadata_tier(queries, num_heads)}"


def route_name(precision, queries, keys, num_heads=NUM_HEADS):
    """Logical route of a problem: host-known scalars only, KV length by range.

    The 32-head names are the shipped ones, the ``dense_mqa.v9`` single-launch
    scan classes and the ``dense_mqa.v10`` fused tiered routes ``fp8:qle128:short``
    (the single-launch fused Q <= 128 program for ``1 < Q <= 128`` and
    ``K <= policy.fused_qle128_max_kv``) and ``fp8:qle2048:short`` (the
    single-launch fused Q <= 2048 program for ``128 < Q <= 2048`` and
    ``K <= policy.fused_qle2048_max_kv[arch]`` on the most permissive
    architecture -- the key is per architecture, 34048 on sm_100a and 65536 on
    sm_103a; a device whose bound the K exceeds is served the next family or the
    two-launch tier route, :func:`served_route`); their K rule is each route
    record's ``kv_alignment`` for the device's architecture (256 for FP4, 256 or 1
    per FP8 route and architecture as the catalog publishes it) with the record's
    ``kv_min``; other head counts carry an ``h<H>`` infix, their programs take any
    ``K >= 1`` (in-kernel KV tail) and their tiers are named by the token ceiling
    of the shared block ceilings (``le8`` / ``le64`` / ``le1024`` / ``any`` for 64
    heads); a 64-head call with ``K <= policy.dense_short.max_kv`` and at least
    ``policy.dense_short.min_q_blocks`` query blocks names the one-split
    ``:short`` route of its kind (:func:`h64_short_shape`), and a 64-head call
    with ``K > policy.dense_kvsplit.min_kv`` and fewer query blocks than the
    architecture's catalogued SM count names the in-kernel KV-split route of its
    plan text -- ``fp8:h64:q1:kvsplit`` or ``fp8:h64:{full,partial}:{qle128,qle2048}:kvsplit``
    (:func:`h64_kvsplit_plan`; the one SM-count dependent rule, so a query count
    the catalogued counts disagree on has no logical route -- ``ValueError``,
    pass the device to :func:`served_route`). Whether the catalog
    ships the route is a separate question (:func:`dense_route_available`), and
    which route the device is served is :func:`served_route`.

    32-head precedence (every bound is a catalog policy key). FP8: the fused
    single-query chain ``fp8:q1:short`` (``K <= fused_q1_max_kv``), the plain
    ``fp8:q1`` pair, the fused ``fp8:q128:short`` text (``Q = 128``,
    ``K <= fused_q128_max_kv``), then -- on the overlap tier of the query count
    (``le128``: ``1 < Q <= 128``, ``le2048``: ``128 < Q <= 2048``) -- the two
    single-launch families in the order ``policy.single_launch_precedence[arch][tier]``
    (``le_class``: the in-kernel scan class of the query count qualified by block
    kind, ``fp8:{full,partial}:le{32,768}:short``, ``Q = 128`` belongs to the Q128
    rule only; ``qle``: the fused tiered text), the first whose own bound admits
    the call, then the metadata + indexer pair. FP4: ``fp4:q1:short``
    (``K <= fp4.fused_q1_max_kv``), the scan class (``fp4:le32:short``; the
    1024-block class as ``fp4:qle1024:short`` for
    ``Q <= fp4.first_tile_prefetch_max_queries_scan1024`` -- the CTA-scan program
    with the first-tile L2 prefetch -- and ``fp4:le1024:short`` above that query
    count), then the pair. A scan class is the smallest ``fused_scan_ceilings``
    entry covering ``ceil(Q / 4)`` and is served only while
    ``K <= fused_scan<ceiling>_max_kv`` of that class (:func:`_fused_scan_class`).
    The logical name needs one rule: every architecture must share the tier's
    precedence (``ValueError`` otherwise -- pass the device to :func:`served_route`).
    """
    arches = sorted(_catalog()["arches"])
    union = max(_fused_qle2048_bounds().values())
    verdicts = {
        a: _route_name_on(precision, queries, keys, num_heads, a, qle2048_max_kv=union)
        for a in arches
    }
    if len(set(verdicts.values())) != 1:
        raise ValueError(
            f"dense MQA route of {(precision, num_heads, queries, keys)} differs by architecture {verdicts} "
            "(policy.single_launch_precedence / policy.dense_kvsplit.sm_count); pass arch=device_arch(device) to served_route"
        )
    return next(iter(verdicts.values()))


def _fused_scan_class(queries, keys, bounds):
    """The in-kernel scan class that serves a 32-head call, or 0: the SMALLEST
    query-block ceiling of ``bounds["fused_scan_ceilings"]`` covering ``ceil(Q / 4)``
    when ``K <= bounds["fused_scan<ceiling>_max_kv"]``. The class is fixed by the
    query count and only its own KV bound is consulted; above it the call takes
    the two-launch route, never a larger class."""
    blocks = (queries + BLOCK_Q - 1) // BLOCK_Q
    for ceiling in bounds["fused_scan_ceilings"]:
        ceiling = int(ceiling)
        if blocks <= ceiling:
            if keys <= int(bounds[f"fused_scan{ceiling}_max_kv"]):
                return ceiling
            return 0
    return 0


def h64_short_shape(queries, keys, num_heads=64):
    """Shape-only one-split gate of the 64-head ``:short`` routes (``policy.dense_short``): ``K <= max_kv``
    (every query block covers one 256-row KV split whatever the windows hold) and at least ``min_q_blocks``
    query blocks (a one-block call does not gain from the one-split text). False when the catalog has no
    ``dense_short`` policy."""
    short = _catalog()["policy"].get("dense_short")
    if short is None:
        return False
    blocks = (queries + block_q(num_heads) - 1) // block_q(num_heads)
    return keys <= int(short["max_kv"]) and blocks >= int(short["min_q_blocks"])


def h64_ts_shape(queries, keys, num_heads=64):
    """Shape-only gate of the 64-head ``:ts`` tile-schedule-words routes (``policy.dense_ts``): not a ``:short``
    shape, ``min_kv_exclusive < K <= max_kv``, ``Q <= max_queries`` and at least ``min_q_blocks`` query blocks.
    False when the catalog has no ``dense_ts`` policy."""
    ts = _catalog()["policy"].get("dense_ts")
    if ts is None or h64_short_shape(queries, keys, num_heads):
        return False
    blocks = (queries + block_q(num_heads) - 1) // block_q(num_heads)
    return (
        int(ts["min_kv_exclusive"]) < keys <= int(ts["max_kv"])
        and queries <= int(ts["max_queries"])
        and blocks >= int(ts["min_q_blocks"])
    )


def _h64_tier_fallback(
    precision, queries, num_heads, route, arch=None, *, shipped=True
):
    kind = route.split(":")[-2]
    return _h64_grid_stride_route(
        precision, queries, num_heads, kind, arch, shipped=shipped
    )


def _h64_grid_stride_route(
    precision, queries, num_heads, kind, arch=None, *, shipped=True
):
    """The grid-stride tier route of ``kind`` (``full`` / ``partial``) a withheld or uncatalogued 64-head ``:short`` /
    ``:ts`` / ``:kvsplit`` call is served by: among the tier of the query count (:func:`metadata_tier`) and the wider
    tiers up to ``:any``, the first route the catalog carries and ``arch`` admits, else the first the catalog carries (the
    caller's admission check then refuses the call) -- a fallback always names a program the catalog ships, never a
    tier it leaves unshipped.  ``shipped=False`` returns the query count's tier route regardless (the export's own
    resolution).  ``arch=None`` requires the architectures to agree on the admission of every tier consulted."""
    prefix = _route_prefix(precision, num_heads)
    tier = metadata_tier(queries, num_heads)
    if not shipped:
        return f"{prefix}:{kind}:{tier}"
    bq = block_q(num_heads)
    tiers = [
        "any" if max_blocks is None else f"le{int(max_blocks) * bq}"
        for max_blocks in _catalog()["policy"]["metadata_tier_blocks"]
    ]
    routes = _catalog()["routes"]
    candidates = [
        f"{prefix}:{kind}:{candidate}"
        for candidate in tiers[tiers.index(tier) :]
        if f"{prefix}:{kind}:{candidate}" in routes
    ]
    if not candidates:
        raise ValueError(
            f"the catalog ships no grid-stride {kind} route for {queries} queries of {num_heads} heads"
        )
    for route in candidates:
        if _admitted(route, arch):
            return route
    return candidates[0]


def h64_kvsplit_plan(queries, keys, num_heads=64, *, arch):
    """The shape-only gate of the 64-head in-kernel KV-split routes (``policy.dense_kvsplit``) at the catalogued SM
    count of ``arch`` (``dense_kvsplit.sm_count[arch]``; no device query -- a device whose SM count differs from its
    architecture's entry is routed as the catalogue says, the program takes the launch grid at runtime): the plan
    text (``q1`` / ``qle128`` / ``qle2048``, the inclusive query bands ``dense_kvsplit.plans``) when
    ``K > dense_kvsplit.min_kv`` and ``ceil(Q / block_q) < sm_count``, or -- the whole-grid band, where
    ``dense_kvsplit.fullgrid[arch]`` -- ``K > dense_kvsplit.fullgrid_min_kv`` and
    ``sm_count <= ceil(Q / block_q) <= dense_kvsplit.max_q_blocks`` (the ``qle2048`` plan: the same program at grid =
    the SM count); else ``None`` (the grid-stride route).  ``None`` as well for 32 heads, when the catalog has no
    ``dense_kvsplit`` policy, or when the catalogued SM count exceeds ``dense_kvsplit.max_sms`` (the plan's CTA bound)."""
    gate = _catalog()["policy"].get("dense_kvsplit")
    if gate is None or num_heads == NUM_HEADS:
        return None
    counts = gate["sm_count"]
    if arch not in counts:
        raise ValueError(f"arch must be one of {sorted(counts)}, got {arch!r}")
    sm_count = int(counts[arch])
    if sm_count > int(gate["max_sms"]):
        return None
    bq = block_q(num_heads)
    blocks = (queries + bq - 1) // bq
    if keys <= int(gate["min_kv"]):
        return None
    if blocks >= sm_count:
        # the whole-grid band of the qle2048 plan (per architecture; its KV bound is the ``:ts`` routes' bound, so the
        # ``:ts`` gate keeps 256 < K <= fullgrid_min_kv at these block counts)
        fullgrid = gate["fullgrid"]
        enabled = fullgrid[arch] if isinstance(fullgrid, dict) else bool(fullgrid)
        if not (
            enabled
            and blocks <= int(gate["max_q_blocks"])
            and keys > int(gate["fullgrid_min_kv"])
        ):
            return None
    for plan, (low, high) in gate["plans"].items():
        if int(low) <= queries <= int(high):
            return plan
    return None


def _h64_kvsplit_fallback(
    precision, queries, num_heads, route, arch=None, *, shipped=True
):
    """The grid-stride route a ``:kvsplit`` route the catalog does not carry or ``arch`` withholds is served by
    (``policy.dense_kvsplit.fallback == "tier"``): the ``q1`` route for the Q = 1 closed form, else the shipped tier
    route of the same query count and block kind (:func:`_h64_grid_stride_route`: the next wider tier where the
    catalog leaves the query count's own tier unshipped or ``arch`` withholds it)."""
    parts = route.split(":")
    prefix = _route_prefix(precision, num_heads)
    if parts[-2] == "q1":
        return f"{prefix}:q1"
    return _h64_grid_stride_route(
        precision, queries, num_heads, parts[-3], arch, shipped=shipped
    )


def route_owns_metadata(route, num_heads=NUM_HEADS):
    """True when a plan on ``route`` owns a ``metadata`` buffer: every route of a head count whose schedule runs a
    metadata stage (:func:`schedules_metadata`: the 32-head routes, their ``:short`` routes write the buffer
    in-kernel) and the 64-head ``:kvsplit`` routes (the plan text writes the tiered schedule into it in-kernel;
    ``metadata_words(Q, num_sms, 64)`` int32 words).  The other 64-head routes have no buffer."""
    return bool(schedules_metadata(num_heads)) or str(route).endswith(":kvsplit")


def served_route(precision, queries, keys, num_heads=NUM_HEADS, *, arch=None):
    """The route the runtime serves for a problem on ``arch``: :func:`_route_name_on` under the architecture's own rule --
    (i) its single-launch precedence (``policy.single_launch_precedence[arch]``) and its own
    ``policy.fused_qle2048_max_kv[arch]`` (a 32-head call whose K exceeds the device's bound is served by the next family
    in the tier's order or by the two-launch tier route of its query count) -- and (ii) a 64-head ``:short`` / ``:ts``
    route the catalog does not carry or ``arch`` withholds (``policy.dense_admission``) is served by the tier route of
    the same query count -- the production program the tier table admits -- never by the stock kernel
    (``policy.dense_short.fallback == policy.dense_ts.fallback == "tier"``), and (iii) a 64-head ``:kvsplit`` route
    the catalog does not carry or ``arch`` withholds is served by the grid-stride ``q1`` / tier route of the same query
    count (``policy.dense_kvsplit.fallback == "tier"``). ``arch=None`` is accepted only where the architectures agree (on the
    fused Q <= 2048 bound for ``keys``, on the tier's precedence, on the KV-split gate at their catalogued SM counts and
    on the short / ts / KV-split route's admission; ``ValueError`` otherwise, as :func:`_admitted`)."""
    arches = sorted(_catalog()["arches"])
    if arch is None:
        verdicts = {
            a: _route_name_on(precision, queries, keys, num_heads, a) for a in arches
        }
        if len(set(verdicts.values())) != 1:
            raise ValueError(
                f"dense MQA route of {(precision, num_heads, queries, keys)} is served per architecture as {verdicts} "
                f"(policy.fused_qle2048_max_kv {_fused_qle2048_bounds()} / policy.single_launch_precedence / "
                "policy.dense_kvsplit.sm_count); pass arch=device_arch(device)"
            )
        route = next(iter(verdicts.values()))
    else:
        if arch not in arches:
            raise ValueError(f"arch must be one of {arches}, got {arch!r}")
        route = _route_name_on(precision, queries, keys, num_heads, arch)
    if num_heads != NUM_HEADS and route.endswith((":short", ":ts")):
        if route not in _catalog()["routes"] or not _admitted(route, arch):
            return _h64_tier_fallback(precision, queries, num_heads, route, arch)
    if num_heads != NUM_HEADS and route.endswith(":kvsplit"):
        if route not in _catalog()["routes"] or not _admitted(route, arch):
            return _h64_kvsplit_fallback(precision, queries, num_heads, route, arch)
    return route


def _export_route(precision, queries, keys, num_heads=NUM_HEADS):
    """The route the export resolves for a point (admission is routing policy, not a precondition of the
    program): :func:`route_name`, or the grid-stride route when a 64-head ``:short`` / ``:ts`` / ``:kvsplit`` route has
    no catalog record."""
    route = route_name(precision, queries, keys, num_heads)
    if num_heads != NUM_HEADS and route not in _catalog()["routes"]:
        if route.endswith((":short", ":ts")):
            return _h64_tier_fallback(
                precision, queries, num_heads, route, shipped=False
            )
        if route.endswith(":kvsplit"):
            return _h64_kvsplit_fallback(
                precision, queries, num_heads, route, shipped=False
            )
    return route


def kv_alignment(num_heads=NUM_HEADS, precision="fp8"):
    """Required divisor of the KV length on the ``(precision, num_heads)`` routes: 256 where the programs
    take whole 256-row KV blocks (the FP4 32-head programs), 1 where they handle the KV tail in-kernel
    (TMA zero-fill past K + the per-row window mask: the 64-head programs and, when the catalog publishes
    it, the FP8 32-head programs); ``policy.kv_alignment[precision][num_heads]``."""
    return int(
        _catalog()["policy"]["kv_alignment"][str(precision)][str(int(num_heads))]
    )


def route_kv_alignment(route, arch=None):
    """Required divisor of the KV length on one catalogued route on ``arch``: the record's ``kv_alignment[arch]``
    (the producer publishes it per route AND per architecture; a 32-head FP8 route class may take any K on one
    architecture while its siblings -- or the other architecture -- keep 256). ``arch=None`` is accepted only
    where the architectures agree (``ValueError`` otherwise: no silent admit, pass :func:`device_arch`)."""
    rules = _catalog()["routes"][route]["kv_alignment"]
    if arch is not None:
        if arch not in rules:
            raise ValueError(f"arch must be one of {sorted(rules)}, got {arch!r}")
        return int(rules[arch])
    values = {int(v) for v in rules.values()}
    if len(values) != 1:
        raise ValueError(
            f"dense MQA route {route} takes any K on {sorted(a for a, v in rules.items() if int(v) == 1)} only; "
            "pass arch=device_arch(device)"
        )
    return values.pop()


def _export_kv_alignment(route):
    """The divisor the export applies to a route's validation rows: the most permissive architecture's rule
    (a K admitted on one architecture is validated on every architecture; admission is routing policy)."""
    return min(int(v) for v in _catalog()["routes"][route]["kv_alignment"].values())


def route_kv_min(route):
    """Smallest KV length the route's programs accept (the record's ``kv_min``): 256 where the KV boxes are fitted
    to one 256-row block (the shipped 32-head programs), 1 where they may extend past the tensor (the fused
    Q <= 128 and Q <= 2048 programs, the 64-head programs)."""
    return int(_catalog()["routes"][route]["kv_min"])


def _keys_served(route, keys, arch):
    """True when ``keys`` satisfies the route record's K rule (``kv_min`` and the architecture's ``kv_alignment``)."""
    if keys < route_kv_min(route):
        return False
    rules = _catalog()["routes"][route]["kv_alignment"]
    if arch is None:
        verdicts = {a: keys % int(v) == 0 for a, v in rules.items()}
        if len(set(verdicts.values())) == 1:
            return next(iter(verdicts.values()))
        raise ValueError(
            f"dense MQA route {route} serves K = {keys} on {sorted(a for a, v in verdicts.items() if v)} only "
            f"(not on {sorted(a for a, v in verdicts.items() if not v)}); pass arch=device_arch(device)"
        )
    return keys % route_kv_alignment(route, arch) == 0


def align4(count):
    return (int(count) + 3) // 4 * 4


def _kv_scales_view(kv_scales, keys):
    """The ``[1, K]`` TMA operand of the FP32 KV scales with a 16-byte aligned row stride (``align4(K)``);
    the caller's storage must hold ``align4(K)`` elements past the tensor's offset."""
    needed = align4(keys)
    available = (
        kv_scales.untyped_storage().nbytes() - kv_scales.storage_offset() * 4
    ) // 4
    if available < needed:
        raise ValueError(
            f"kv_scales storage must hold align4(K) = {needed} FP32 elements past its offset (has {available}); "
            "allocate the scale buffer with a 4-element aligned length"
        )
    return kv_scales.as_strided((1, keys), (needed, 1))


def dense_route_available(num_heads, queries, keys, precision="fp8", *, arch=None):
    """True when the shipped catalog carries a route for ``(precision, num_heads, queries, keys)`` and the
    route is admitted on ``arch`` (``policy.dense_admission``, decided per architecture for the 64-head
    family: a tier withheld on an architecture has its record but is not served there). False, not an
    error, for a precision or a head count the catalog does not export (``policy.heads``).

    Host-only (no device query): the engine admission consults the table the export delivered instead of
    restating its rules -- the route record's ``kv_min`` and its ``kv_alignment`` for ``arch`` decide the KV
    length. ``arch=None`` is accepted only where the architectures agree on the route (the per-arch fused
    Q <= 2048 bound included) and on its K rule for ``keys``; a route (or a K) admitted on some architectures
    only raises ``ValueError`` (no silent admit) -- pass the device's :func:`device_arch`.
    """
    if precision not in ("fp4", "fp8") or num_heads not in heads():
        return False
    bound = max_queries(num_heads)
    if queries < 1 or keys < 1 or (bound is not None and queries > bound):
        return False
    # the route the device is served; with ``arch=None`` the architectures must agree on it (the per-arch fused Q <= 2048
    # bound, the per-arch single-launch precedence: ``served_route`` raises instead of a silent verdict)
    route = served_route(precision, queries, keys, num_heads, arch=arch)
    if route not in _catalog()["routes"]:
        return False
    if not _keys_served(route, keys, arch):
        return False
    return _admitted(route, arch)


def h64_admission(arch):
    """``policy.dense_admission`` of one architecture: ``{"admitted_routes": [...], "withheld_routes": [...],
    "reason": str | None}`` -- the 64-head routes served on ``arch`` (every query count of those tiers), the
    routes the producer measured and did not admit there (a withheld TIER route keeps the engine's stock
    kernel; a withheld ``:short`` / ``:ts`` route is served by its tier route and a withheld ``:kvsplit`` route by the
    grid-stride ``q1`` / tier route, see :func:`served_route`; ``reason`` says
    why), two disjoint lists. Empty / ``None`` when the catalog has no 64-head family. Host-only."""
    record = _catalog()["policy"].get("dense_admission")
    arches = (
        sorted(record["admitted_routes"])
        if record is not None
        else sorted(_catalog()["arches"])
    )
    if arch not in arches:
        raise ValueError(f"arch must be one of {arches}, got {arch!r}")
    if record is None:
        return {"admitted_routes": [], "withheld_routes": [], "reason": None}
    return {
        "admitted_routes": sorted(record["admitted_routes"][arch]),
        "withheld_routes": sorted(record["withheld_routes"][arch]),
        "reason": record["reason"][arch],
    }


def _admitted(route, arch):
    """Per-architecture verdict of a catalogued route: routes outside the published family are admitted
    everywhere; a published route is admitted on ``arch`` iff listed there; with ``arch=None`` the
    architectures must agree."""
    record = _catalog()["policy"].get("dense_admission")
    if record is None:
        return True
    verdicts = {}
    for name, admitted in record["admitted_routes"].items():
        if route in admitted or route in record["withheld_routes"][name]:
            verdicts[name] = route in admitted
    if not verdicts:
        return True
    if arch is None:
        if len(set(verdicts.values())) == 1:
            return next(iter(verdicts.values()))
        raise ValueError(
            f"dense MQA route {route} is admitted on {sorted(a for a, v in verdicts.items() if v)} only "
            f"(withheld on {sorted(a for a, v in verdicts.items() if not v)}); pass arch=device_arch(device)"
        )
    if arch not in record["admitted_routes"]:
        raise ValueError(
            f"arch must be one of {sorted(record['admitted_routes'])}, got {arch!r}"
        )
    return verdicts.get(arch, True)


def route_record(precision, queries, keys, num_heads=NUM_HEADS, *, arch=None):
    """The catalog record of a problem's route (``stages``, ``sequence``, ``num_heads``, ``block_q``,
    ``clean_logits``, ``kv_alignment``); raises ``ValueError`` when the catalog does not ship it or the
    route is withheld on ``arch``."""
    if not dense_route_available(num_heads, queries, keys, precision, arch=arch):
        raise ValueError(
            f"no exported dense MQA route for {(precision, num_heads, queries, keys)} on {arch or 'every arch'}"
        )
    served = served_route(precision, queries, keys, num_heads, arch=arch)
    return _catalog()["routes"][served]


def metadata_tier(queries, num_heads=NUM_HEADS):
    """Metadata program tier of a query count: the smallest block ceiling that covers it, named by its
    token ceiling for ``num_heads`` (``le16`` = 4 blocks of 4 queries, ``le8`` = 4 blocks of 2, ...; ``any``)."""
    bq = block_q(num_heads)
    blocks = (queries + bq - 1) // bq
    for max_blocks in _catalog()["policy"]["metadata_tier_blocks"]:
        if max_blocks is None:
            return "any"
        if blocks <= max_blocks:
            return f"le{int(max_blocks) * bq}"
    raise ValueError(f"queries must be in 1..{max_queries(num_heads)}")


def schedules_metadata(num_heads=NUM_HEADS):
    """True when the routes of ``num_heads`` run a metadata stage (the shipped 32-head schedule). The
    64-head programs are single-stage: no metadata program; the grid-stride and one-split texts bind their
    ``ScheduleMeta`` operand to ``ks`` and never read it, the ``:kvsplit`` plan texts write their schedule into the
    plan's own buffer (:func:`route_owns_metadata`)."""
    records = _catalog()["routes"].values()
    return any(
        int(record["num_heads"]) == int(num_heads)
        and any(stage == "metadata" for stage, _p in record["stages"])
        for record in records
    )


def max_queries(num_heads=NUM_HEADS):
    """Largest query count the generated metadata schedule accepts for ``num_heads``; ``None`` (no bound)
    for head counts whose routes have no metadata stage."""
    if num_heads in heads() and not schedules_metadata(num_heads):
        return None
    return int(_catalog()["policy"]["max_q_blocks"]) * block_q(num_heads)


def logits_stride(num_kv_tokens):
    """Physical FP32 logits row stride: align(K + 256, 8) elements."""
    return (num_kv_tokens + BLOCK_KV + 7) // 8 * 8


def metadata_words(num_q_tokens, num_sms, num_heads=NUM_HEADS):
    """int32 words of the schedule metadata: per-SM headers, then two words per query block."""
    bq = block_q(num_heads)
    return (3 * num_sms + 1) // 2 * 2 + 2 * ((num_q_tokens + bq - 1) // bq)


def stage_bindings(
    precision,
    q,
    kv,
    weights,
    starts,
    ends,
    output,
    metadata,
    *,
    q_scales=None,
    kv_scales=None,
    num_sms,
    num_heads=NUM_HEADS,
):
    """Encode the public tensor contract into the two generated argument plans."""
    import torch

    queries, keys = starts.numel(), kv.shape[0]
    bq = block_q(num_heads)
    blocks = (queries + bq - 1) // bq
    # Single-stage routes without a metadata buffer (the 64-head grid-stride and one-split programs) bind their
    # ScheduleMeta operand to ks and never read it; the producer plan is unused.  A 64-head ``:kvsplit`` plan passes
    # the buffer its program writes in-kernel (``route_owns_metadata``).
    schedule_meta = (metadata if metadata is not None else starts).view(torch.uint32)
    producer = dict(
        Starts=starts.view(torch.uint32),
        Ends=ends.view(torch.uint32),
        Metadata=schedule_meta,
        num_q_tokens=queries,
        num_kv_tokens=keys,
        grid_x=1,
        grid_y=1,
        grid_z=1,
    )
    consumer = dict(
        Weights=weights,
        Logits=output,
        ScheduleMeta=schedule_meta,
        cu_seq_len_k_start=starts,
        cu_seq_len_k_end=ends,
        seq_len=queries,
        seq_len_kv=keys,
        stride_logits=output.stride(0),
        num_q_blocks=blocks,
        grid_x=num_sms,
        grid_y=1,
        grid_z=1,
    )
    if precision == "fp4":
        consumer.update(
            Q=q.view(torch.uint8).reshape(-1, 64),
            KV=kv.view(torch.uint8).reshape(keys, 64),
            SF_Q=q_scales.view(torch.uint8).reshape(-1, 16),
            SF_KV=kv_scales.view(torch.uint8).reshape(-1, 16),
        )
    elif precision == "fp8":
        consumer.update(
            Q=q.view(torch.uint8).reshape(-1, HEAD_DIM),
            # DeepGEMM retains the otherwise-unused FP8 Q-scale descriptor in the
            # physical ABI; the live KV-scale map is aliased exactly as the source route.
            Q_scales_alias=_kv_scales_view(kv_scales, keys),
            KV=kv.view(torch.uint8).reshape(keys, HEAD_DIM),
            KV_scales=_kv_scales_view(kv_scales, keys),
            CandidateValues=output,
            CandidateIndices=starts,
            CandidateCounts=ends,
            ScoreThresholds=output,
            num_kv_splits=1,
            candidate_capacity=1,
        )
    else:
        raise ValueError("precision must be 'fp4' or 'fp8'")
    return {"metadata": producer, "logits": consumer}


def _submission(arch, program, bindings, num_sms, *, stage=None):
    module, record = load_program(arch, program, num_sms)
    arguments = []
    for kind, key in record["arg_plan"]:
        if kind == "workspace":
            raise NotImplementedError(
                "Dense route unexpectedly requires external descriptor storage"
            )
        if stage is None:
            selected, name = key.split(".", 1)
        else:
            selected, name = stage, key
        arguments.append(bindings[selected][name])
    return (module[record["ffi_entry"]], tuple(arguments)), (module, record)


def _resolve_device(q, sm_count):
    import torch

    if q.device.type != "cuda":
        raise RuntimeError("Dense MQA requires CUDA tensors")
    device_index = (
        q.device.index if q.device.index is not None else torch.cuda.current_device()
    )
    arch, num_sms = device_facts(device_index)
    if sm_count is not None:
        # CTA budget override (tests, restricted serving partitions, an engine's
        # pipeline-parallel SM reservation): the schedule partitions over this many
        # CTAs, the FP8 indexer is built with this count defined and the metadata is
        # sized for it. Any positive count is a legal grid; above the device's SM
        # count the extra CTAs run as a second wave.
        if int(sm_count) < 1:
            raise ValueError("sm_count must be a positive CTA budget")
        num_sms = int(sm_count)
    return arch, num_sms


class DenseMqaPlan:
    """Bind packed operands for repeated metadata-to-logits submissions.

    FP4 q/kv contain packed E2M1 bytes [Q,H,64]/[K,64]; q_scales and kv_scales
    are contiguous UE8M0 bytes [Q,H,4]/[K,4]. FP8 q/kv use E4M3
    [Q_storage,H,128]/[K,128], where Q_storage >= max(BLOCK_Q,Q); kv_scales are
    FP32[K]. weights are FP32[Q,H] (FP4) or FP32[Q_storage,H] (FP8). H is one
    of the catalog's head counts and BLOCK_Q = 128 // H. starts/ends are
    int32[Q] windows with 0 <= start <= end <= K; (precision, H, Q, K) must
    have a shipped route (``dense_route_available``). logits[q, k] =
    sum_h max(0, Q[q,h] . KV[k]) * weights[q,h] for start[q] <= k < end[q]
    and -inf elsewhere.

    Inputs, output and metadata stay bound to the plan. Updating their contents
    is supported, including CUDA Graph replay. One plan is not concurrently
    reusable across streams because output and metadata are mutable. Output is
    FP32 with physical row stride logits_stride(K) = align8(K + 256); consume
    output[:, :K] (``logical_output``). On the 32-head routes every cell of the
    output is written by each submission (``clean_logits == "fused"``); the
    64-head routes are single-stage and store raw tiles only (cells outside the
    windows are unspecified): the grid-stride and one-split routes have no
    metadata buffer (``metadata`` must be None, ``plan.metadata`` is None, the
    program is gridDim-strided over the tiles from the ``sm_count`` launch grid),
    the ``:kvsplit`` routes own the ``metadata_words(Q, num_sms, 64)`` int32
    buffer their plan text writes in-kernel (``metadata`` may be passed or is
    allocated; the plan is one launch of ``sm_count`` CTAs and a CTA budget above
    ``policy.dense_kvsplit.max_sms`` is refused); the 32-head ``:short`` routes
    (fused single-query chain, in-kernel warp / CTA scan classes, the fused tiered
    ``fp8:qle128:short`` / ``fp8:qle2048:short`` texts) have no metadata stage but
    still own the buffer their program writes in-kernel (:func:`route_owns_metadata`).
    FP4 output storage must include the final query tile.
    """

    def __init__(
        self,
        precision,
        q,
        kv,
        weights,
        starts,
        ends,
        *,
        q_scales=None,
        kv_scales=None,
        output=None,
        metadata=None,
        sm_count=None,
        enforce_admission=True,
    ):
        import torch

        arch, num_sms = _resolve_device(q, sm_count)
        if precision not in ("fp4", "fp8"):
            raise ValueError("precision must be 'fp4' or 'fp8'")
        if q.ndim != 3 or kv.ndim != 2:
            raise ValueError("q must be [Q_storage, H, D] and kv [K, D]")
        num_heads = int(q.shape[1])
        if num_heads not in heads():
            raise ValueError(
                f"q has {num_heads} heads; exported head counts: {heads()}"
            )
        bq = block_q(num_heads)
        queries, keys = starts.numel(), kv.shape[0]
        bound = max_queries(num_heads)
        if queries < 1 or (bound is not None and queries > bound):
            raise ValueError(
                f"Q must be in 1..{bound}" if bound is not None else "Q must be >= 1"
            )
        if keys < 1:
            raise ValueError("positive K is required")
        available = dense_route_available(
            num_heads, queries, keys, precision, arch=arch
        )
        if not available and not enforce_admission:
            # Export validation runs every catalogued row on every architecture; the per-arch admission
            # (policy.dense_admission, the route record's per-arch K rule) is routing policy for the engines,
            # not a precondition of the program.
            try:
                export_route = _export_route(precision, queries, keys, num_heads)
                available = (
                    export_route in _catalog()["routes"]
                    and keys >= route_kv_min(export_route)
                    and keys % _export_kv_alignment(export_route) == 0
                )
            except ValueError:
                available = False
        if not available:
            try:
                route = route_name(precision, queries, keys, num_heads)
            except ValueError:
                route = None
            if route in _catalog()["routes"] and keys >= route_kv_min(route):
                alignment = route_kv_alignment(route, arch)
                if keys % alignment:
                    raise ValueError(
                        f"dense MQA route {route} requires K % {alignment} == 0 on {arch} (K = {keys}); "
                        "keep the stock path for this call"
                    )
                raise ValueError(
                    f"dense MQA route {route} is withheld on {arch} ({h64_admission(arch)['reason']}); "
                    "keep the stock path for this call"
                )
            raise ValueError(
                f"no exported dense MQA route for precision {precision!r}, {num_heads} heads, "
                f"Q = {queries}, K = {keys} (K must be at least the route record's kv_min and a multiple of its "
                "kv_alignment for this architecture)"
            )
        if (
            starts.dtype != torch.int32
            or ends.dtype != torch.int32
            or starts.shape != ends.shape
            or starts.ndim != 1
        ):
            raise ValueError("starts and ends must be equal-shaped int32 vectors")
        if weights.dtype != torch.float32:
            raise ValueError("weights must be FP32")
        packed_dtype = q.dtype in (torch.uint8, torch.int8)
        if precision == "fp4":
            if (
                not packed_dtype
                or kv.dtype != q.dtype
                or tuple(q.shape) != (queries, num_heads, 64)
                or tuple(kv.shape) != (keys, 64)
            ):
                raise ValueError("FP4 q/kv must be packed int8/uint8[Q,H,64]/[K,64]")
            if (
                q_scales is None
                or kv_scales is None
                or q_scales.dtype != torch.uint8
                or kv_scales.dtype != torch.uint8
            ):
                raise ValueError("FP4 q_scales/kv_scales must be UE8M0 uint8 tensors")
            if tuple(q_scales.shape) != (queries, num_heads, 4) or tuple(
                kv_scales.shape
            ) != (keys, 4):
                raise ValueError("FP4 scales must have shape [Q,H,4]/[K,4]")
            q_rows = queries
        else:
            if (
                q.dtype != torch.float8_e4m3fn
                or kv.dtype != torch.float8_e4m3fn
                or tuple(q.shape[1:]) != (num_heads, HEAD_DIM)
                or tuple(kv.shape) != (keys, HEAD_DIM)
            ):
                raise ValueError("FP8 q/kv must be E4M3[Q_storage,H,128]/[K,128]")
            q_rows = q.shape[0]
            # The shipped 32-head programs address a whole 4-query block of Q rows; the 64-head programs
            # take Q unpadded (the TMA box's out-of-bounds rows are zero-filled).
            min_rows = max(bq, queries) if num_heads == NUM_HEADS else queries
            if (
                q_rows < min_rows
                or kv_scales is None
                or kv_scales.dtype != torch.float32
                or tuple(kv_scales.shape) != (keys,)
            ):
                raise ValueError(
                    f"FP8 requires Q_storage >= {min_rows} rows and FP32 KV scales[K]"
                )
            _kv_scales_view(kv_scales, keys)
        if tuple(weights.shape) != (q_rows, num_heads):
            raise ValueError(
                "weights must match the physical Q rows and the head count"
            )
        tensors = [q, kv, weights, starts, ends, kv_scales]
        if q_scales is not None:
            tensors.append(q_scales)
        if any(t.device != q.device or not t.is_contiguous() for t in tensors):
            raise ValueError("all inputs must be contiguous on one CUDA device")
        self.arch = arch
        self.num_sms = num_sms
        self.num_heads = num_heads
        # The served route: a withheld 64-head :short route runs its tier program on this arch; export validation
        # (enforce_admission=False) runs every catalogued route on every arch.
        self.route_name = (
            served_route(precision, queries, keys, num_heads, arch=arch)
            if enforce_admission
            else _export_route(precision, queries, keys, num_heads)
        )
        self.route = _catalog()["routes"][self.route_name]
        self.config = dict(
            precision=precision,
            num_heads=num_heads,
            queries=queries,
            keys=keys,
            num_sms=num_sms,
            route=self.route_name,
        )
        stride = logits_stride(keys)
        padded_rows = (queries + bq - 1) // bq * bq
        if output is None:
            rows = padded_rows if precision == "fp4" else queries
            output = torch.empty((rows, stride), dtype=torch.float32, device=q.device)[
                :queries
            ]
        if (
            output.dtype != torch.float32
            or tuple(output.shape) != (queries, stride)
            or output.device != q.device
            or not output.is_contiguous()
        ):
            raise ValueError("output must be contiguous FP32[Q,logits_stride(K)]")
        # The FP4 route's final CTA addresses the padded query rows of its last tile.
        if precision == "fp4" and (
            output.untyped_storage().nbytes() - output.storage_offset() * 4
            < padded_rows * stride * 4
        ):
            raise ValueError(
                f"FP4 output backing storage must include the final {bq}-row tile"
            )
        # The metadata buffer belongs to the route's program, not to the route's stage list: the shipped 32-head
        # ``:short`` routes (fp8:q1:short, fp8:q128:short, the le32 / le768 / le1024 scan classes, fp4:q1:short)
        # have no metadata stage yet WRITE their schedule into ScheduleMeta in-kernel, so every 32-head plan owns
        # the buffer; so does a 64-head ``:kvsplit`` plan (the plan text recomputes the tiered schedule in every CTA
        # and writes it; ``route_owns_metadata``); the other 64-head programs never touch it (ScheduleMeta bound to ks).
        if self.route_name.endswith(":kvsplit"):
            max_sms = int(_catalog()["policy"]["dense_kvsplit"]["max_sms"])
            if num_sms > max_sms:
                raise ValueError(
                    f"route {self.route_name}: the in-kernel KV-split plan runs on at most {max_sms} CTAs "
                    f"(policy.dense_kvsplit.max_sms); sm_count = {num_sms}"
                )
        if route_owns_metadata(self.route_name, num_heads):
            words = metadata_words(queries, num_sms, num_heads)
            if metadata is None:
                metadata = torch.empty(words, dtype=torch.int32, device=q.device)
            if (
                metadata.dtype != torch.int32
                or tuple(metadata.shape) != (words,)
                or metadata.device != q.device
                or not metadata.is_contiguous()
            ):
                raise ValueError(
                    "metadata has the wrong device, dtype or physical extent"
                )
        elif metadata is not None:
            raise ValueError(
                f"route {self.route_name}: the {num_heads}-head programs have no metadata buffer; metadata must be None"
            )
        bindings = stage_bindings(
            precision,
            q,
            kv,
            weights,
            starts,
            ends,
            output,
            metadata,
            q_scales=q_scales,
            kv_scales=kv_scales,
            num_sms=num_sms,
            num_heads=num_heads,
        )
        self._submissions, self._programs = [], []
        sequence = self.route.get("sequence")
        if sequence:
            selections = [(None, sequence)]
        else:
            stages = route_stages(self.route, arch, keys)
            selections = [(stage[0], stage_program(stage, arch)) for stage in stages]
        self.program_names = [program for _stage, program in selections]
        for stage_name, program in selections:
            submit, loaded = _submission(
                arch, program, bindings, num_sms, stage=stage_name
            )
            self._submissions.append(submit)
            self._programs.append(loaded)
        self.output, self.metadata = output, metadata
        self.logical_output = output[:, :keys]
        self._retained = (
            (*tensors, output, metadata) if metadata is not None else (*tensors, output)
        )

    @property
    def launch_count(self):
        """Number of FFI submissions ``run()`` issues (1 for sequence and fused routes)."""
        return len(self._submissions)

    def run(self):
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            for entry, args in self._submissions:
                entry(*args)
        return self.logical_output


def fp8_mqa_logits(
    q,
    kv,
    weights,
    ks,
    ke,
    clean_logits=False,
    max_seqlen_k=0,
    *,
    sm_count=None,
):
    """One-shot FP8 dense MQA logits with DeepGEMM's ``fp8_mqa_logits`` signature.

    ``q`` E4M3 ``[Q, H, 128]`` (``H`` in :func:`heads`), ``kv = (E4M3 [K, 128],
    FP32 scales [K])``, FP32 ``weights [Q, H]``, int32 ``ks`` / ``ke [Q]``.
    Returns the FP32 ``[Q, K]`` view of a freshly allocated
    ``[Q, logits_stride(K)]`` buffer (``logits_stride(K) = align8(K + 256)``, no
    padding rows). On a ``"fused"`` route every cell is written (``-inf``
    outside each row's window) and the ``clean_logits`` flag is accepted for
    signature parity without changing the result.
    ``max_seqlen_k`` must be 0 (no narrowing of the returned view).
    ``sm_count`` is the CTA budget (default: the device's SM count). Rows below
    the query block (``Q < BLOCK_Q``) are padded here where the route's program
    needs a whole block, and a KV-scale buffer whose storage is shorter than
    ``align4(K)`` elements is copied into an aligned one, so callers pass exact
    ``[Q]`` / ``[K]`` slices. Allocates; for allocation-free replay prepare a
    :class:`DenseMqaPlan` on caller-owned storage.

    The route record's ``clean_logits`` field says whether the program writes
    ``-inf`` outside the windows (``"fused"``, the shipped 32-head programs) or
    stores raw tiles only (``"raw"``, DeepGEMM's own ``clean_logits=False``
    semantics); ``clean_logits=True`` is rejected on a ``"raw"`` route.
    """
    import torch

    if max_seqlen_k != 0:
        raise ValueError("max_seqlen_k must be 0: the returned view is [Q, K]")
    if not isinstance(kv, (tuple, list)) or len(kv) != 2:
        raise ValueError("kv must be the pair (kv E4M3 [K, 128], kv_scales FP32 [K])")
    kv_values, kv_scales = kv
    if q.ndim != 3:
        raise ValueError("q must be [Q, H, 128]")
    queries, num_heads = int(q.shape[0]), int(q.shape[1])
    if num_heads not in heads():
        raise ValueError(f"q has {num_heads} heads; exported head counts: {heads()}")
    bq = block_q(num_heads)
    if tuple(weights.shape) != (queries, num_heads):
        raise ValueError("weights must be [Q, H]")
    keys = int(kv_values.shape[0]) if kv_values.ndim == 2 else 0
    if (
        clean_logits
        and route_record(
            "fp8", queries, keys, num_heads, arch=device_arch(q.device)
        ).get("clean_logits")
        != "fused"
    ):
        raise ValueError(
            "clean_logits=True is not available on this route (the program stores raw tiles, DeepGEMM "
            "clean_logits=False semantics); pass clean_logits=False as the engine does"
        )
    if kv_scales.ndim == 1 and (
        kv_scales.untyped_storage().nbytes() - kv_scales.storage_offset() * 4
    ) // 4 < align4(keys):
        aligned = torch.empty(
            align4(keys), dtype=kv_scales.dtype, device=kv_scales.device
        )
        aligned[:keys] = kv_scales
        kv_scales = aligned[:keys]
    q_rows = max(bq, queries) if num_heads == NUM_HEADS else queries
    if q_rows != queries:
        q_storage = torch.zeros(
            (q_rows, num_heads, HEAD_DIM), dtype=q.dtype, device=q.device
        )
        q_storage[:queries] = q
        w_storage = torch.zeros(
            (q_rows, num_heads), dtype=weights.dtype, device=q.device
        )
        w_storage[:queries] = weights
        q, weights = q_storage, w_storage
    plan = DenseMqaPlan(
        "fp8", q, kv_values, weights, ks, ke, kv_scales=kv_scales, sm_count=sm_count
    )
    return plan.run()
