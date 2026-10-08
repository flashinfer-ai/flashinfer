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
every SM count.

Head counts: the catalog policy lists the exported head counts (``heads``);
``BLOCK_Q = 128 // num_heads`` queries share one 128-row MMA tile (4 for 32
heads, 2 for 64 heads). The one-shot ``fp8_mqa_logits`` mirrors DeepGEMM's
``fp8_mqa_logits`` signature on top of :class:`DenseMqaPlan`.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

_ARCHES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
CATALOG_SCHEMA = "dense_mqa.v6"
NUM_HEADS = 32  # default head count of the public helpers
BLOCK_QH = 128  # query rows x heads per MMA tile
BLOCK_Q = BLOCK_QH // NUM_HEADS
BLOCK_KV = 256
HEAD_DIM = 128


@functools.cache
def _catalog():
    catalog = json.loads(Path(__file__).with_name("dense_mqa_catalog.json").read_text())
    if catalog.get("schema") != CATALOG_SCHEMA:
        raise RuntimeError(
            f"dense MQA catalog schema {catalog.get('schema')!r} is not {CATALOG_SCHEMA!r}; regenerate the export"
        )
    return catalog


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


def program_names(route=None):
    """Names of the catalogued programs (optionally only those a route can launch)."""
    catalog = _catalog()
    if route is None:
        return sorted(catalog["programs"])
    record = catalog["routes"].get(route) or catalog["paged_routes"][route]
    names = [program for _stage, program in record["stages"]]
    if record.get("sequence"):
        names.append(record["sequence"])
    return names


def program_definitions(record, num_sms):
    """Compile-line definitions of a program record: ``SM_COUNT`` is the launch grid of the logits
    consumers (the SM count, or a plan's CTA-budget override)."""
    values = {"SM_COUNT": int(num_sms)}
    unknown = sorted(set(record["definitions"]) - set(values))
    if unknown:
        raise RuntimeError(
            f"catalog program requires definitions this runtime cannot supply: {unknown}"
        )
    return {name: values[name] for name in record["definitions"]}


def program_spec(arch, name, num_sms):
    """FlashInfer JIT build specification of one generated program for ``arch`` and the launch grid ``num_sms``
    (a compile-line definition of every program; the name carries every supplied value)."""
    from flashinfer.jit import env
    from flashinfer.jit.core import gen_jit_spec

    record = _catalog()["programs"][name]
    if arch not in record["arches"]:
        raise RuntimeError(f"program {name} is not exported for {arch}")
    definitions = program_definitions(record, num_sms)
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


def route_name(precision, queries, keys, num_heads=NUM_HEADS):
    """Logical route of a problem: host-known scalars only, KV length by range.

    The 32-head names are the shipped ones (``K % 256 == 0``); other head counts
    carry an ``h<H>`` infix, their programs take any ``K >= 1`` (in-kernel KV
    tail) and their tiers are named by the token ceiling of the shared block
    ceilings (``le8`` / ``le64`` / ``le1024`` / ``any`` for 64 heads). Whether the
    catalog ships the route is a separate question (:func:`dense_route_available`).
    """
    policy = _catalog()["policy"]
    if precision not in ("fp4", "fp8"):
        raise ValueError("precision must be 'fp4' or 'fp8'")
    prefix = _route_prefix(precision, num_heads)
    if precision == "fp4":
        return (
            f"{prefix}:q1"
            if queries == 1
            else f"{prefix}:{metadata_tier(queries, num_heads)}"
        )
    if queries == 1:
        if num_heads == NUM_HEADS and keys <= policy["fused_q1_max_kv"]:
            return f"{prefix}:q1:short"
        return f"{prefix}:q1"
    if (
        num_heads == NUM_HEADS
        and queries == 128
        and keys <= policy["fused_q128_max_kv"]
    ):
        return f"{prefix}:q128:short"
    kind = "full" if queries % block_q(num_heads) == 0 else "partial"
    return f"{prefix}:{kind}:{metadata_tier(queries, num_heads)}"


def kv_alignment(num_heads=NUM_HEADS):
    """Required divisor of the KV length for ``num_heads`` (256 for the shipped 32-head programs, 1 where
    the programs handle the KV tail in-kernel); a catalog policy value."""
    return int(_catalog()["policy"]["kv_alignment"][str(int(num_heads))])


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
    family: a tier withheld on an architecture has its record but is not served there).

    Host-only (no device query): the engine admission consults the table the export delivered instead of
    restating its rules. ``arch=None`` is accepted only where the architectures agree on the route; a route
    admitted on some architectures only raises ``ValueError`` (no silent admit) -- pass the device's
    :func:`device_arch`.
    """
    try:
        bound = max_queries(num_heads)
        if queries < 1 or keys < 1 or (bound is not None and queries > bound):
            return False
        if keys % kv_alignment(num_heads):
            return False
        route = route_name(precision, queries, keys, num_heads)
    except (ValueError, KeyError):
        return False
    if route not in _catalog()["routes"]:
        return False
    return _admitted(route, arch)


def h64_admission(arch):
    """``policy.dense_admission`` of one architecture: ``{"admitted_routes": [...], "withheld_routes": [...],
    "reason": str | None}`` -- the 64-head tiers served on ``arch`` (every query count of those tiers), the
    tiers the producer measured and did not admit there (the engine keeps its stock kernel; ``reason`` says
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
    return _catalog()["routes"][route_name(precision, queries, keys, num_heads)]


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
    64-head programs are single-stage: gridDim-strided over the tiles, no metadata program, their
    ``ScheduleMeta`` operand bound to ``ks`` and unused."""
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
    # Single-stage routes (the 64-head programs) have no metadata buffer: their ScheduleMeta operand is
    # bound to ks and never read; the producer plan is unused.
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
    64-head routes are single-stage without a metadata buffer (``metadata`` must
    be None, ``plan.metadata`` is None, the program is gridDim-strided over the
    tiles from the ``sm_count`` launch grid) and store raw tiles only (cells
    outside the windows are unspecified); the 32-head fused short-KV routes have
    no metadata stage but still own the buffer their program writes in-kernel.
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
            # (policy.dense_admission) is routing policy for the engines, not a precondition of the program.
            try:
                available = route_name(
                    precision, queries, keys, num_heads
                ) in _catalog()["routes"] and (keys % kv_alignment(num_heads) == 0)
            except ValueError:
                available = False
        if not available:
            try:
                route = route_name(precision, queries, keys, num_heads)
            except ValueError:
                route = None
            if route in _catalog()["routes"] and keys % kv_alignment(num_heads) == 0:
                raise ValueError(
                    f"dense MQA route {route} is withheld on {arch} ({h64_admission(arch)['reason']}); "
                    "keep the stock path for this call"
                )
            raise ValueError(
                f"no exported dense MQA route for precision {precision!r}, {num_heads} heads, "
                f"Q = {queries}, K = {keys} (K must be a multiple of kv_alignment({num_heads}))"
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
        self.route_name = route_name(precision, queries, keys, num_heads)
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
        # The metadata buffer belongs to the head count's schedule, not to the route's stage list: the
        # shipped 32-head fused routes (fp8:q1:short / fp8:q128:short) have no metadata stage yet WRITE
        # their schedule into ScheduleMeta in-kernel, so every 32-head plan owns the buffer; the 64-head
        # programs never touch it (ScheduleMeta bound to ks).
        if schedules_metadata(num_heads):
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
        selections = (
            [(None, sequence)]
            if sequence
            else [(stage, program) for stage, program in self.route["stages"]]
        )
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
