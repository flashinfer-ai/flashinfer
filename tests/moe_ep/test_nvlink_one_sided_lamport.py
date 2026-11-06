"""Lamport transport of NVLinkOneSidedAlltoAll against its fence transport.

Every multi-rank test runs the same steps through two instances on the same EP
group, one with the Lamport transport (``lamport=None``, the default) and one
without (``lamport=False``), and requires on every rank:

* the same received rows from every source, byte for byte (as a set: the
  fence transport compacts rows in arrival order), with the Lamport rows at
  ``source * tokens_per_rank + token``;
* a bitwise identical combine output, with an expert stand-in whose partial
  results cancel across ranks, so that the reduction order shows.

Launched one process per GPU, across nodes too:

    torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_nvlink_one_sided_lamport.py
    mpirun -np 8 pytest tests/moe_ep/test_nvlink_one_sided_lamport.py

Checks are gathered across ranks, so all ranks pass or fail each test together.
``test_rounds_under_rank_skew`` runs its rounds in a child process of every
rank, so that a deadlock (the kernels trap) fails that test only.
``FLASHINFER_TEST_MOE_A2A_LAMPORT_DEFECT=<n>`` runs the suite on a module built
from a copy of the sources with planted defect ``n`` (``_DEFECTS``), which it
must detect. The host-only tests at the end need no launcher.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import os
import traceback
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch

from flashinfer.moe_ep.backends.split.comm.nvlink_one_sided import (
    communication as one_sided,
)

from ._mpi_launch import adopt_mpi_launch

_DEFECT_ENV = "FLASHINFER_TEST_MOE_A2A_LAMPORT_DEFECT"
_IDS_SENTINEL = -2004318072  # 0x88888888, the expert-id rows' sentinel word
# Multipliers of the expert stand-in per receiving rank (exact powers of two):
# ranks 0 and 1 cancel and leave small residues, so the BF16 combine output
# depends on the order in which the partial results are added.
_ASSOC = (1.0, -1.0, 2.0**-20, 2.0**-21, -(2.0**-9), 2.0**-30, 2.0**-15, -(2.0**-25))
_TOP_K = (1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 22)
_FORMATS = ("bf16", "mxfp8", "fp8_block128", "nvfp4")
_DISPATCH_FORMAT = {
    "bf16": None,
    "mxfp8": "MXFP8",
    "fp8_block128": "DeepSeekFp8",
    "nvfp4": "NVFP4",
}
# Contributing ranks per token of the "groups" routing: both sides of every
# boundary of the combine's groups of 8 sources (n <= 8: one group; n <= 16:
# g0 + g1; else (g0 + g1) + (g2 + g3)).
_GROUP_SIZES = (1, 2, 3, 8, 9, 12, 16, 17, 20, 22)

adopt_mpi_launch()
_MULTI_RANK = int(os.environ.get("WORLD_SIZE", "1")) > 1


# ----------------------------------------------------------------------------
# Process group and collective test bodies
# ----------------------------------------------------------------------------


@functools.cache
def _group() -> SimpleNamespace:
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            device_id=torch.device("cuda", local_rank),
            timeout=timedelta(minutes=30),
        )
    return SimpleNamespace(rank=dist.get_rank(), world=dist.get_world_size())


def _all_gather(obj):
    import torch.distributed as dist

    gathered = [None] * _group().world
    dist.all_gather_object(gathered, obj)
    return gathered


def _fail_on_every_rank(failures: list) -> None:
    gathered = _all_gather(failures[:4])
    bad = [(rank, f) for rank, fs in enumerate(gathered) for f in fs]
    if bad:
        pytest.fail("; ".join(f"rank {rank}: {f}" for rank, f in bad[:8]))


def collective(test):
    """Run a test body on every rank; an exception on one rank fails the test
    on all of them instead of leaving the others in the next collective."""

    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        try:
            failures = test(*args, **kwargs) or []
        except Exception as exc:  # noqa: BLE001 - pytest outcomes are BaseException
            trace = traceback.format_exc(limit=-3).replace("\n", " | ")
            failures = [f"{type(exc).__name__}: {exc} [{trace[-600:]}]"]
        _fail_on_every_rank(failures)

    return wrapper


# Planted defects: (source file, its text, the defective text), patched into a
# copy of the module's sources. The production sources carry no test hooks.
_HEADER = "include/flashinfer/comm/moe_alltoall_lamport.cuh"
_DEFECTS = {
    1: (  # wrong target: tokens t % 7 == 3 go to the next rank
        _HEADER,
        "expert_id / p.experts_per_rank : -1;\n",
        "expert_id / p.experts_per_rank : -1;\n"
        "    if (rank >= 0 && token % 7 == 3) rank = (rank + 1) % p.ep_size;\n",
    ),
    2: (  # dropped token: token 0 never reaches its lowest target rank
        _HEADER,
        "static_cast<uint64_t>(high_ranks) << 32;\n",
        "static_cast<uint64_t>(high_ranks) << 32;\n"
        "  if (token == 0 && target_mask) {\n"
        "    if (target_rank == __ffsll(static_cast<long long>(target_mask)) - 1)"
        " target_rank = -1;\n"
        "    target_mask &= target_mask - 1;\n"
        "  }\n",
    ),
    3: (  # combine order: sources 0 and 2 of the first group swapped
        _HEADER,
        "    ranks[j] = 8 * group + j < num_sources ? s_sources[8 * group + j] : -1;\n",
        "    ranks[j] = 8 * group + j < num_sources ? s_sources[8 * group + j] : -1;\n"
        "  if (group == 0 && num_sources > 2) {\n"
        "    int const swapped = ranks[0];\n"
        "    ranks[0] = ranks[2];\n"
        "    ranks[2] = swapped;\n"
        "  }\n",
    ),
    4: (  # wrong invalid id: -1 whatever invalid_expert_id is
        "csrc/nvlink_one_sided_alltoall.cu",
        "params.invalid_expert_id = static_cast<int32_t>(invalidExpertId);",
        "params.invalid_expert_id = -1;",
    ),
    5: (  # group join bound: a token with 9 sources loses its 9th
        _HEADER,
        "if (num_sources > 8) sum",
        "if (num_sources > 9) sum",
    ),
    6: (  # group join order: (g0 + g2) + (g1 + g3) for more than 16 sources
        _HEADER,
        "sum = __fadd_rn(sum, __fadd_rn(group_sums[2][e], last));",
        "sum = __fadd_rn(__fadd_rn(group_sums[0][e], group_sums[2][e]),"
        " __fadd_rn(group_sums[1][e], last));",
    ),
    7: (  # no epoch advance: every round's ids rows land in the same set
        _HEADER,
        "      atomicAdd(p.round_state + kEpoch, 1);\n",
        "",
    ),
    8: (  # no D5 re-arm: received payload and scale vectors stay in their rows
        _HEADER,
        "      store_volatile(address(j), splat(kDataSentinel));\n",
        "",
    ),
    9: (  # no C3 re-arm: received partials stay in their combine rows
        _HEADER,
        "if ((polled >> j) & 1u) store_volatile(address(j), splat(kDataSentinel));",
        "(void)polled;",
    ),
}


@functools.cache
def _defect_module(defect: int):
    """The kernel module built from a copy of its sources with planted defect
    ``defect`` (in the JIT's generated-source directory)."""
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.comm import gen_nvlink_one_sided_alltoall_module

    spec = gen_nvlink_one_sided_alltoall_module()
    name = f"{spec.name}_lamport_test_defect{defect}"
    root = jit_env.FLASHINFER_GEN_SRC_DIR / name
    path, text, defective = _DEFECTS[defect]
    ops = "csrc/nvlink_one_sided_alltoall.cu"
    original = {
        _HEADER: jit_env.FLASHINFER_INCLUDE_DIR
        / "flashinfer/comm/moe_alltoall_lamport.cuh",
        ops: jit_env.FLASHINFER_CSRC_DIR / "nvlink_one_sided_alltoall.cu",
    }
    copies = {
        _HEADER: root / _HEADER,
        path: root / path,
    }  # the header shadows the original
    for key, copy in copies.items():
        source = original[key].read_text()
        if key == path:
            assert source.count(text) == 1, f"defect {defect}: its anchor moved"
            source = source.replace(text, defective)
        copy.parent.mkdir(parents=True, exist_ok=True)
        tmp = copy.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(source)
        os.replace(tmp, copy)  # every rank writes the same bytes
    sources = [
        copies[ops] if ops in copies and source == original[ops] else source
        for source in spec.sources
    ]
    spec = dataclasses.replace(
        spec,
        name=name,
        sources=sources,
        extra_include_dirs=[root / "include", *(spec.extra_include_dirs or [])],
    )
    return spec.build_and_load()


def _use_planted_defect(setattr_) -> None:
    """Swap in the module with the planted defect (if any) and its constants."""
    defect = int(os.environ.get(_DEFECT_ENV, "0") or 0)
    if defect:
        module = _defect_module(defect)
        setattr_(one_sided, "get_nvlink_one_sided_alltoall_module", lambda: module)
        names, values = module.moe_a2a_get_metainfo_index_pairs()
        constants = SimpleNamespace(
            **{
                str(n).removeprefix("MOE_A2A_"): int(v)
                for n, v in zip(names, values, strict=True)
            }
        )
        setattr_(one_sided, "layout_constants", lambda: constants)


@pytest.fixture(autouse=True)
def _planted_defect(monkeypatch):
    _use_planted_defect(monkeypatch.setattr)


multi_rank = pytest.mark.gpu_2  # tests/conftest.py skips it without a launcher


def _supported() -> str:
    """Why this group cannot run the Lamport transport ('' if it can)."""
    from flashinfer.moe_ep import NVLinkOneSidedAlltoAll

    local = ""
    module = one_sided.get_nvlink_one_sided_alltoall_module()
    if not NVLinkOneSidedAlltoAll.is_platform_supported():
        local = "NVLink one-sided is not supported here"
    elif not module.moe_a2a_lamport_available():
        local = "the kernel module has no Lamport kernels for this GPU"
    elif _group().world > one_sided.layout_constants().LAMPORT_MAX_RANKS:
        local = "the Lamport transport supports at most 64 ranks"
    reasons = [r for r in _all_gather(local) if r]
    return reasons[0] if reasons else ""


@pytest.fixture(scope="module")
def group():
    if not _MULTI_RANK:
        pytest.skip("needs a multi-rank launch (torchrun or mpirun)")
    g = _group()
    reason = _supported()
    if reason:
        pytest.skip(reason)
    return g


# ----------------------------------------------------------------------------
# Instances
# ----------------------------------------------------------------------------

_INSTANCES: dict = {}


@pytest.fixture(autouse=True)
def _destroy_instances():
    """Each test's instances are destroyed after it (collective on every rank)."""
    yield
    for comm in _INSTANCES.values():
        comm.destroy()
    _INSTANCES.clear()


def _instance(
    fmt: str,
    top_k: int,
    num_experts: int,
    hidden: int,
    max_tokens: int,
    *,
    lamport,
    tag: int = 0,
    invalid_expert_id: int = -1,
    **config,
):
    """One instance per configuration and test (construction is collective)."""
    from flashinfer.fused_moe import QuantFormat
    from flashinfer.moe_ep import (
        BootstrapConfig,
        MoEEpCommParams,
        NVLinkOneSidedConfig,
        create_communication,
    )

    key = (
        fmt,
        top_k,
        num_experts,
        hidden,
        max_tokens,
        lamport,
        tag,
        invalid_expert_id,
        tuple(sorted(config.items())),
    )
    if key not in _INSTANCES:
        g = _group()
        name = _DISPATCH_FORMAT[fmt]
        params = MoEEpCommParams(
            num_experts=num_experts,
            top_k=top_k,
            max_tokens_per_rank=max_tokens,
            hidden_size=hidden,
            dtype=torch.bfloat16,
            dispatch_format=None if name is None else getattr(QuantFormat, name),
            invalid_expert_id=invalid_expert_id,
        )
        config.setdefault("timeout_sec", 60)
        config.setdefault("cft", False)  # Lamport against the fence transport
        _INSTANCES[key] = create_communication(
            BootstrapConfig(world_size=g.world, rank=g.rank),
            params,
            NVLinkOneSidedConfig(lamport=lamport, **config),
        )
    return _INSTANCES[key]


def _pair(fmt, top_k, num_experts, hidden, max_tokens, **config):
    """(Lamport instance, fence instance) of one configuration."""
    lamport = _instance(
        fmt, top_k, num_experts, hidden, max_tokens, lamport=None, **config
    )
    fence = _instance(
        fmt, top_k, num_experts, hidden, max_tokens, lamport=False, **config
    )
    assert lamport.lamport_enabled and not fence.lamport_enabled
    return lamport, fence


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------


def _generator(seed: int, source: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(
        seed * 1_000_003 + source * 7919 + 17
    )


def _token_counts(seed: int, world: int, tokens_per_rank: int) -> list:
    g = torch.Generator().manual_seed(seed)
    counts = torch.randint(0, tokens_per_rank + 1, (world,), generator=g).tolist()
    counts[seed % world] = tokens_per_rank  # the step's maximum is reached
    return counts


def _expert_ids(
    g, count, top_k, num_experts, world, source, routing, invalid_expert_id
):
    if count == 0:
        return torch.empty(0, top_k, dtype=torch.int32, device="cuda")
    per_rank = num_experts // world
    if routing == "groups":  # token t reaches exactly n of _GROUP_SIZES ranks
        sizes = [n for n in _GROUP_SIZES if n <= min(top_k, world)]
        n = torch.tensor(
            [sizes[(t + source) % len(sizes)] for t in range(count)], device="cuda"
        )[:, None]
        ranks = torch.rand(count, world, generator=g, device="cuda").argsort(dim=1)
        slot = torch.arange(top_k, device="cuda")[None, :]
        owner = ranks.gather(1, (slot % n).expand(count, top_k))
        return (owner * per_rank + slot // n).to(torch.int32).contiguous()
    scores = torch.rand(count, num_experts, generator=g, device="cuda")
    if routing == "hot":  # most routes to rank 1
        scores[:, per_rank : 2 * per_rank] += 1.0
    elif routing == "self":  # the source's own experts first
        scores[:, source * per_rank : (source + 1) * per_rank] += 1.0
    ids = scores.argsort(dim=1, descending=True)[:, :top_k].to(torch.int32)
    if routing == "invalid":
        draw = torch.rand(count, top_k, generator=g, device="cuda")
        ids = torch.where(draw < 0.15, invalid_expert_id, ids)
        ids = torch.where((draw >= 0.15) & (draw < 0.25), num_experts + 3, ids)
        ids = torch.where((draw >= 0.25) & (draw < 0.3), _IDS_SENTINEL, ids)
        ids[torch.arange(count, device="cuda") % 5 == 3] = invalid_expert_id
    return ids.to(torch.int32).contiguous()


def _random_bytes(g, *shape) -> torch.Tensor:
    return torch.randint(
        0, 256, shape, generator=g, device="cuda", dtype=torch.int32
    ).to(torch.uint8)


def _inputs(
    seed,
    source,
    count,
    fmt,
    top_k,
    num_experts,
    hidden,
    world,
    routing="uniform",
    special=False,
    invalid_expert_id=-1,
    weights=True,
    subnormal=False,
):
    """Deterministic inputs of rank ``source``: (hidden_states, scale, ids, weights).
    ``subnormal`` scales the weights so that the expert partials are BF16
    subnormals, which both combines flush alike."""
    g = _generator(seed, source)
    ids = _expert_ids(
        g, count, top_k, num_experts, world, source, routing, invalid_expert_id
    )
    if fmt == "bf16":
        x = (torch.randn(count, hidden, generator=g, device="cuda") * 0.5).to(
            torch.bfloat16
        )
        scale = None
        if special and count:
            bits = x.view(torch.int16)
            table = (0x7FC0, 0x7F80, -0x80, -0x8000, 0x0001, -0x7FC1, 0x7F7F, -0x81)
            pick = torch.randint(0, len(table), bits.shape, generator=g, device="cuda")
            hit = torch.rand(bits.shape, generator=g, device="cuda") < 0.05
            for i, value in enumerate(table):
                bits[hit & (pick == i)] = value
            words = x.view(torch.int32)
            words[torch.rand(words.shape, generator=g, device="cuda") < 0.01] = -1
            words[::3, 0] = -1  # a whole sentinel word
    else:
        columns, scale_shape = {
            "mxfp8": (hidden, (hidden // 32, torch.uint8)),
            "fp8_block128": (hidden, (hidden // 128, torch.float32)),
            "nvfp4": (hidden // 2, (hidden // 16, torch.uint8)),
        }[fmt]
        x = _random_bytes(g, count, columns)
        scale_columns, scale_dtype = scale_shape
        scale = _random_bytes(g, count, scale_columns * scale_dtype.itemsize).view(
            scale_dtype
        )
        if special and count:
            x.view(torch.int32)[::3, 0] = -1  # the data sentinel word
            if fmt == "nvfp4":
                x.view(torch.int32)[1::3, 1] = _IDS_SENTINEL  # eight -0 nibbles
            scale.view(torch.uint8).view(-1, 4)[::5, :] = 0xFF
        if fmt != "nvfp4":
            x = x.view(torch.float8_e4m3fn)
    w = None
    if weights:
        w = torch.rand(count, top_k, generator=g, device="cuda")
        if subnormal:
            w = w * 2.0**-128
        if special and count:
            w.view(torch.int32)[
                torch.rand(w.shape, generator=g, device="cuda") < 0.05
            ] = -1
            w[::4, 0] = float("nan")
    return x.contiguous(), None if scale is None else scale.contiguous(), ids, w


# ----------------------------------------------------------------------------
# Expert stand-in, references and comparisons
# ----------------------------------------------------------------------------


@functools.cache
def _expert_table(num_experts: int, hidden: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(4242)
    return (torch.randn(num_experts, hidden, generator=g, device="cuda") * 0.25).to(
        torch.bfloat16
    )


def _expert_output(comm, received) -> torch.Tensor:
    """BF16 partial result of every received row: sum over this rank's slots of
    weight * table[expert], times the rank's _ASSOC multiplier. Element-wise
    operations only, so a row's result does not depend on its position."""
    ids = received.topk_ids.long()
    weights = received.topk_weights
    first = comm.ep_rank * comm.num_local_experts
    table = _expert_table(comm.params.num_experts, comm.params.hidden_size).float()
    out = torch.zeros(ids.shape[0], comm.params.hidden_size, device="cuda")
    for k in range(ids.shape[1]):
        local = (ids[:, k] >= first) & (ids[:, k] < first + comm.num_local_experts)
        row = table[ids[:, k].clamp(0, comm.params.num_experts - 1)]
        weight = (
            weights[:, k]
            if weights is not None
            else torch.ones_like(local, dtype=torch.float)
        )
        out = out + torch.where(
            local[:, None], weight[:, None] * row, torch.zeros_like(row)
        )
    return (out * _ASSOC[comm.ep_rank % len(_ASSOC)]).to(torch.bfloat16)


def _words(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8).view(t.shape[0], -1)


def _canonical_rows(fmt, x, scale, ids, weights, invalid_expert_id):
    """The bytes a Lamport step delivers for these rows: words equal to a
    landing sentinel rewritten as the senders rewrite them."""

    def word_rule(t, replacement):
        w = t.contiguous().view(torch.uint8).view(t.shape[0], -1).view(torch.int32)
        return torch.where(w == -1, torch.full_like(w, replacement), w)

    if fmt == "bf16":
        x = word_rule(x, 0x7FFF7FFF)
    elif fmt == "nvfp4":
        w = x.contiguous().view(torch.uint8).view(x.shape[0], -1).view(torch.int32)
        magnitude = w & 0x77777777
        nonzero = (magnitude | (magnitude >> 1) | (magnitude >> 2)) & 0x11111111
        x = w & (0x77777777 | (nonzero << 3))
    else:
        x = word_rule(x, 0x7FFFFFFF)
    if scale is not None and fmt != "mxfp8":
        scale = word_rule(scale, 0x7FFFFFFF)
    ids = torch.where(ids == _IDS_SENTINEL, invalid_expert_id, ids)
    if weights is not None:
        weights = word_rule(weights, 0x7FFFFFFF)
    parts = [_words(x)] + ([] if scale is None else [_words(scale)])
    parts += [_words(ids)] + ([] if weights is None else [_words(weights)])
    return torch.cat(parts, dim=1)


def _received_rows(fmt, received, world, invalid_expert_id, canonical):
    """Per source: [rows, bytes] of its valid received rows (some id differs
    from invalid_expert_id) and their positions in the source's slice."""
    tokens_per_rank = received.tokens_per_rank
    ids = received.topk_ids.view(world, tokens_per_rank, -1)
    parts = [
        received.hidden_states,
        received.hidden_states_scale,
        received.topk_ids,
        received.topk_weights,
    ]
    if canonical:
        rows = _canonical_rows(fmt, *parts, invalid_expert_id)
    else:
        rows = torch.cat([_words(p) for p in parts if p is not None], dim=1)
    rows = rows.view(world, tokens_per_rank, -1)
    result = []
    for source in range(world):
        valid = (ids[source] != invalid_expert_id).any(dim=1)
        result.append((rows[source][valid], torch.nonzero(valid).flatten()))
    return result


def _sorted_rows(rows: torch.Tensor) -> torch.Tensor:
    """Rows in the order of an exact integer hash of their bytes (equal rows
    are interchangeable), so two sets of rows compare with torch.equal."""
    if rows.shape[0] <= 1:
        return rows
    g = torch.Generator(device=rows.device).manual_seed(rows.shape[1])
    coefficients = torch.randint(
        1, 1 << 20, (rows.shape[1],), generator=g, device=rows.device, dtype=torch.int64
    )
    return rows[(rows.long() * coefficients).sum(dim=1).argsort()]


def _compare_dispatch(
    fmt, lamport_recv, fence_recv, world, invalid_expert_id, expected, label
) -> list:
    """Lamport vs fence received rows (as sets per source) and the Lamport rows'
    positions against the routed tokens of each source (``expected``)."""
    failures = []
    got = _received_rows(fmt, lamport_recv, world, invalid_expert_id, canonical=False)
    ref = _received_rows(fmt, fence_recv, world, invalid_expert_id, canonical=True)
    for source in range(world):
        rows, positions = got[source]
        ref_rows, _ = ref[source]
        if rows.shape != ref_rows.shape or not torch.equal(
            _sorted_rows(rows), _sorted_rows(ref_rows)
        ):
            failures.append(f"{label}: rows from source {source} differ")
        elif not torch.equal(positions.cpu(), expected[source].cpu()):
            failures.append(f"{label}: Lamport rows from source {source} misplaced")
    return failures


def _routed_tokens(ids, rank, num_experts, world, mask=None) -> torch.Tensor:
    """Tokens of a source that reach ``rank`` (by position)."""
    per_rank = num_experts // world
    valid = (ids >= 0) & (ids < num_experts)
    targets = torch.where(valid, ids.long() // per_rank, -1)
    routed = (targets == rank).any(dim=1)
    if mask is not None and not (mask >> rank) & 1:
        routed[:] = False
    return torch.nonzero(routed).flatten()


def _step(comm, inputs, tokens_per_rank, active_rank_mask=None, sleeps=(0, 0)):
    """Dispatch, expert stand-in, combine; returns (received copies, output).
    ``sleeps``: GPU cycles this rank spins before dispatch and before combine."""
    x, scale, ids, weights = inputs
    kwargs = {} if active_rank_mask is None else {"active_rank_mask": active_rank_mask}
    if sleeps[0]:
        torch.cuda._sleep(sleeps[0])
    received = comm.dispatch(
        x,
        ids,
        weights,
        hidden_states_scale=scale,
        runtime_max_tokens_per_rank=tokens_per_rank,
        **kwargs,
    )
    snapshot = SimpleNamespace(
        hidden_states=received.hidden_states.clone(),
        hidden_states_scale=None
        if received.hidden_states_scale is None
        else received.hidden_states_scale.clone(),
        topk_ids=received.topk_ids.clone(),
        topk_weights=None
        if received.topk_weights is None
        else received.topk_weights.clone(),
        tokens_per_rank=received.tokens_per_rank,
    )
    expert_output = _expert_output(comm, received)
    if sleeps[1]:
        torch.cuda._sleep(sleeps[1])
    output = comm.combine(expert_output, **kwargs)
    return snapshot, output


def _round_inputs(params, fmt, seed, tokens_per_rank, offset=0, **kwargs):
    """Inputs of one step as a function of the source rank: token counts from
    ``seed``, values from ``seed + offset`` (refilled inputs keep their shapes)."""
    g = _group()
    counts = _token_counts(seed, g.world, tokens_per_rank)
    make = functools.partial(
        _inputs,
        seed + offset,
        fmt=fmt,
        top_k=params.top_k,
        num_experts=params.num_experts,
        hidden=params.hidden_size,
        world=g.world,
        **kwargs,
    )
    return lambda source: make(source, counts[source])


def _compare_round(
    fmt,
    inputs,
    got,
    ref,
    took_lamport,
    num_experts,
    *,
    invalid_expert_id=-1,
    label="",
    expect_lamport=True,
) -> list:
    """A step's (received rows, output) on the Lamport instance, ``got``,
    against the same step's on the fence instance, ``ref``."""
    g = _group()
    failures = []
    if took_lamport != expect_lamport:
        failures.append(
            f"{label}: Lamport transport {took_lamport}, expected {expect_lamport}"
        )
    if not torch.equal(got[1].view(torch.int16), ref[1].view(torch.int16)):
        bad = (got[1].view(torch.int16) != ref[1].view(torch.int16)).sum().item()
        failures.append(f"{label}: combine differs in {bad} elements")
    if took_lamport:
        expected = [
            _routed_tokens(inputs(source)[2], g.rank, num_experts, g.world)
            for source in range(g.world)
        ]
        failures += _compare_dispatch(
            fmt, got[0], ref[0], g.world, invalid_expert_id, expected, label
        )
    return failures


def _check_round(
    lamport,
    fence,
    fmt,
    seed,
    tokens_per_rank,
    *,
    invalid_expert_id=-1,
    label="",
    expect_lamport=True,
    **kwargs,
) -> list:
    """One step on both instances with the same inputs; Lamport vs fence."""
    inputs = _round_inputs(
        lamport.params,
        fmt,
        seed,
        tokens_per_rank,
        invalid_expert_id=invalid_expert_id,
        **kwargs,
    )
    mine = inputs(_group().rank)
    got = _step(lamport, mine, tokens_per_rank)
    took_lamport = lamport._last_step_lamport
    ref = _step(fence, mine, tokens_per_rank)
    torch.cuda.synchronize()
    return _compare_round(
        fmt,
        inputs,
        got,
        ref,
        took_lamport,
        lamport.params.num_experts,
        invalid_expert_id=invalid_expert_id,
        label=label,
        expect_lamport=expect_lamport,
    )


def _record_transports(setattr_) -> None:
    """Record whether each instance's last dispatch took the Lamport transport."""
    dispatch = one_sided.NVLinkOneSidedAlltoAll.dispatch

    def recording(self, *args, **kwargs):
        result = dispatch(self, *args, **kwargs)
        self._last_step_lamport = self._round["lamport"]
        return result

    setattr_(one_sided.NVLinkOneSidedAlltoAll, "dispatch", recording)


@pytest.fixture(autouse=True)
def _record_transport(monkeypatch):
    _record_transports(monkeypatch.setattr)


# ----------------------------------------------------------------------------
# Multi-rank tests
# ----------------------------------------------------------------------------


@multi_rank
@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("top_k", _TOP_K)
@collective
def test_bitwise_vs_fence(group, fmt, top_k):
    """Every format and top-k: rows and combine against the fence transport,
    over steps of 1 to 33 tokens with special values and invalid ids."""
    num_experts = 32 * group.world
    lamport, fence = _pair(fmt, top_k, num_experts, 2048, 64)
    failures = []
    for i, (tokens, routing, special) in enumerate(
        (
            (1, "uniform", False),
            (7, "hot", True),
            (33, "invalid", True),
            (16, "self", False),
        )
    ):
        failures += _check_round(
            lamport,
            fence,
            fmt,
            100 + i,
            tokens,
            routing=routing,
            special=special,
            label=f"{fmt} k{top_k} T{tokens} {routing}",
        )
    return failures


@multi_rank
@pytest.mark.parametrize("fmt", _FORMATS)
@collective
def test_wide_rows_and_invalid_id(group, fmt):
    """Rows spanning several blocks (hidden 7168), invalid_expert_id =
    num_experts, no routing weights, and a whole-capacity step."""
    num_experts = 32 * group.world
    failures = []
    for top_k, invalid, weights in (
        (8, num_experts, True),
        (6, -1, False),
        (22, num_experts, False),
    ):
        lamport, fence = _pair(
            fmt, top_k, num_experts, 7168, 40, invalid_expert_id=invalid
        )
        for seed, tokens in ((200, 40), (201, 5)):
            failures += _check_round(
                lamport,
                fence,
                fmt,
                seed,
                tokens,
                routing="invalid",
                special=True,
                invalid_expert_id=invalid,
                weights=weights,
                label=f"{fmt} k{top_k} inv{invalid} w{weights} T{tokens}",
            )
    return failures


# test_combine_order: (routing, top_k). "many_ranks": random routing over many
# ranks (up to 16 at W >= 16), special values and subnormal partials (flushed
# alike by both combines); "group_join": tokens with exactly n contributing
# ranks on both sides of the group boundaries n 8 / 9 and 16 / 17 (_GROUP_SIZES).
_COMBINE_ORDER_CASES = (
    ("many_ranks", 8),
    ("many_ranks", 16),
    ("many_ranks", 22),
    ("group_join", 16),
    ("group_join", 18),
    ("group_join", 22),
)


@multi_rank
@pytest.mark.parametrize(("routing", "top_k"), _COMBINE_ORDER_CASES)
@collective
def test_combine_order(group, routing, top_k):
    """The combine's tree per 8 sources and the join of the groups, bitwise
    against the fence transport. The group joins need W >= 16 (17 for the
    second)."""
    if routing == "group_join":
        if group.world < 16:
            pytest.skip("joining groups of 8 sources needs at least 16 ranks")
        num_experts, seeds = 32 * group.world, range(1100, 1104)
    else:
        num_experts, seeds = 2 * group.world * 8, range(300, 308)
    lamport, fence = _pair("bf16", top_k, num_experts, 1024, 32)
    failures = []
    for seed in seeds:
        failures += _check_round(
            lamport,
            fence,
            "bf16",
            seed,
            32,
            routing="groups" if routing == "group_join" else "uniform",
            special=seed % 2 == 1,
            subnormal=routing == "many_ranks" and seed >= 306,
            label=f"{routing} k{top_k} seed{seed}",
        )
    return failures


@multi_rank
@collective
def test_transport_switching(group):
    """Steps at and above lamport_max_tokens on one instance, with an explicit
    limit (every format) and with the default limit from the SM count (at a
    DeepSeek-V3 shape): the Lamport and fence transports alternate on one
    workspace, rows and combine as above. On the default instance a step with
    EPLB statistics and a payload outside the instance's format also take the
    fence transport."""
    num_experts = 32 * group.world
    failures = []
    for fmt in _FORMATS:
        lamport, fence = _pair(fmt, 8, num_experts, 2048, 24, lamport_max_tokens=12)
        assert lamport.lamport_max_tokens == 12
        for i, tokens in enumerate((12, 13, 1, 24, 12, 13, 12)):
            failures += _check_round(
                lamport,
                fence,
                fmt,
                400 + i,
                tokens,
                special=True,
                label=f"{fmt} T{tokens}",
                expect_lamport=tokens <= 12,
            )

    num_experts = 256 if 256 % group.world == 0 else 32 * group.world
    lamport, fence = _pair(
        "bf16", 8, num_experts, 7168, 512, eplb_stats_num_experts=num_experts
    )
    limit = lamport.lamport_max_tokens
    device = torch.cuda.current_device()
    sms = min(
        _all_gather(torch.cuda.get_device_properties(device).multi_processor_count)
    )
    expected = one_sided._default_lamport_max_tokens(
        group.world, sms, lamport._lamport_row, one_sided.layout_constants()
    )
    if limit != expected:
        failures.append(f"default limit {limit} != {expected}")
    failures += _check_round(lamport, fence, "bf16", 500, limit, label=f"T{limit}")
    failures += _check_round(
        lamport,
        fence,
        "bf16",
        501,
        limit + 1,
        label=f"T{limit + 1}",
        expect_lamport=False,
    )
    counts = _token_counts(1000, group.world, 8)
    x, _, ids, w = _inputs(
        1000, group.rank, counts[group.rank], "bf16", 8, num_experts, 7168, group.world
    )
    combine_input = torch.zeros(
        group.world * 8, 7168, dtype=torch.bfloat16, device="cuda"
    )
    stats = torch.arange(num_experts, dtype=torch.int32, device="cuda")
    lamport.dispatch(x, ids, w, runtime_max_tokens_per_rank=8, eplb_local_stats=stats)
    if lamport._last_step_lamport:
        failures.append("a step with EPLB statistics took the Lamport transport")
    lamport.combine(combine_input)
    lamport.dispatch(x.view(torch.float16), ids, w, runtime_max_tokens_per_rank=8)
    if lamport._last_step_lamport:
        failures.append("an FP16 payload took the Lamport transport")
    lamport.combine(combine_input)
    torch.cuda.synchronize()
    return failures


@multi_rank
@pytest.mark.parametrize("fmt", ("bf16", "nvfp4"))
@collective
def test_rank_mask(group, fmt):
    """A masked rank takes no part; the others match the fence transport."""
    if group.world < 3:
        pytest.skip("needs at least 3 ranks")
    num_experts = 32 * group.world
    lamport, fence = _pair(fmt, 8, num_experts, 2048, 16, enable_rank_mask=True)
    dead = group.world - 1
    mask = lamport.active_rank_mask([r for r in range(group.world) if r != dead])
    failures = []
    for seed in (600, 601, 602):
        if group.rank == dead:
            continue
        counts = _token_counts(seed, group.world, 16)
        make = functools.partial(
            _inputs,
            seed,
            fmt=fmt,
            top_k=8,
            num_experts=num_experts,
            hidden=2048,
            world=group.world,
            special=True,
        )
        mine = make(group.rank, counts[group.rank])
        got_recv, got = _step(lamport, mine, 16, mask)
        ref_recv, ref = _step(fence, mine, 16, mask)
        torch.cuda.synchronize()
        if not lamport._last_step_lamport:
            failures.append("masked step did not take the Lamport transport")
        if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
            failures.append(f"seed {seed}: combine differs")
        expected = [
            _routed_tokens(make(s, counts[s])[2], group.rank, num_experts, group.world)
            if s != dead
            else torch.empty(0, dtype=torch.long, device="cuda")
            for s in range(group.world)
        ]
        failures += _compare_dispatch(
            fmt, got_recv, ref_recv, group.world, -1, expected, f"seed {seed}"
        )
    return failures


def _refill(static, fresh) -> None:
    """Copy fresh inputs into the static input tensors of a captured graph."""
    for dst, src in zip(static, fresh, strict=True):
        if dst is not None:
            dst.copy_(src)


def _graph_rounds(comm, fmt, seeds, tokens):
    """Static inputs for a captured sequence and a function refilling them."""
    rank = _group().rank

    def inputs(offset):
        return [
            _round_inputs(comm.params, fmt, seed, t, offset, special=True)(rank)
            for seed, t in zip(seeds, tokens, strict=True)
        ]

    static = [[t, list(fresh)] for t, fresh in zip(tokens, inputs(0), strict=True)]

    def refill(offset):
        for (_, tensors), fresh in zip(static, inputs(offset), strict=True):
            _refill(tensors, fresh)

    return static, refill


@multi_rank
@pytest.mark.parametrize("fmt", ("bf16", "mxfp8"))
@collective
def test_cuda_graph_replay(group, fmt):
    """A captured sequence mixing Lamport and fence steps, replayed with new
    inputs, against the same steps run eagerly on the fence instance."""
    num_experts = 32 * group.world
    lamport, fence = _pair(fmt, 8, num_experts, 2048, 24, lamport_max_tokens=12, tag=1)
    seeds, tokens = (700, 701, 702, 703), (12, 20, 3, 12)
    static, refill = _graph_rounds(lamport, fmt, seeds, tokens)

    def run(comm):
        return [_step(comm, inputs, t)[1] for t, inputs in static]

    run(lamport)  # warm up, as captures require
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run(lamport)
    failures = []
    for replay in range(3):
        refill(10 * replay)
        graph.replay()
        references = run(fence)
        torch.cuda.synchronize()
        for i, (got, ref) in enumerate(zip(outputs, references, strict=True)):
            if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                failures.append(
                    f"replay {replay} step {i} (T {tokens[i]}): combine differs"
                )
    return failures


# test_rounds_under_rank_skew: _SKEW_ROUNDS Lamport steps of 1 to _SKEW_LIMIT
# tokens per rank, then steps on both sides of the limit. Each round one rank
# spins before its dispatch and one before its combine, rotating; the spins
# outlast the host's work per round, so the GPUs run the rounds back to back.
_SKEW_LIMIT = 16
_SKEW_ROUNDS = 56
_SKEW_TOKENS = (1, 16, 3, 9, 16, 2, 12, 5, 16, 1, 7, 14, 4)
_SKEW_MIXED_TOKENS = (17, 16, 24, 1, 20, 12, 24, 3)
_SKEW_ROUTINGS = ("uniform", "self", "hot", "invalid")
_SKEW_SLEEP_CYCLES = 2_000_000
_SKEW_TIMEOUT_SEC = 10


def _skew_schedule(world: int) -> list:
    """(seed, tokens per rank, routing, dispatch sleeper, combine sleeper) of
    every round."""
    tokens = [_SKEW_TOKENS[i % len(_SKEW_TOKENS)] for i in range(_SKEW_ROUNDS)]
    return [
        (
            1300 + i,
            t,
            _SKEW_ROUTINGS[i % len(_SKEW_ROUTINGS)],
            i % world,
            (i + 1 + i // world) % world,
        )
        for i, t in enumerate(tokens + list(_SKEW_MIXED_TOKENS))
    ]


def _landing_rows(comm) -> list:
    """This rank's Lamport landing rows: dispatch (payload, scale, weights, ids)
    and combine, as bytes."""
    index, meta = comm._index, comm.metainfo
    bounds = [
        int(meta[getattr(index, f"LAMPORT_{name}_OFFSET_INDEX")])
        for name in ("LANDING", "COMBINE_LANDING", "RECV")
    ]
    return [comm._rank_bytes[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True)]


def _rounds_under_skew(fmt) -> list:
    """test_rounds_under_rank_skew, in the child process of every rank."""
    g = _group()
    num_experts = 32 * g.world
    lamport, fence = _pair(
        fmt,
        8,
        num_experts,
        7168,
        24,
        lamport_max_tokens=_SKEW_LIMIT,
        timeout_sec=_SKEW_TIMEOUT_SEC,
    )
    armed = [rows.clone() for rows in _landing_rows(lamport)]
    schedule = _skew_schedule(g.world)

    def inputs(offset):
        return [
            _round_inputs(
                lamport.params,
                fmt,
                seed,
                t,
                offset,
                routing=routing,
                special=seed % 3 == 1,
            )
            for seed, t, routing, _, _ in schedule
        ]

    def run(static):
        steps = []
        for (_, t, _, *sleepers), mine in zip(schedule, static, strict=True):
            sleeps = [_SKEW_SLEEP_CYCLES if g.rank == r else 0 for r in sleepers]
            got = _step(lamport, mine, t, sleeps=sleeps)
            steps.append((got, lamport._last_step_lamport))
        return steps

    def check(sources, static, steps, label):
        torch.cuda.synchronize()
        failures = [
            f"{label}: {n} bytes of the {name} landing rows not re-armed"
            for name, rows, ref in zip(
                ("dispatch", "combine"), _landing_rows(lamport), armed, strict=True
            )
            if (n := (rows != ref).sum().item())
        ]
        for (seed, t, routing, *_), source, mine, (got, took) in zip(
            schedule, sources, static, steps, strict=True
        ):
            failures += _compare_round(
                fmt,
                source,
                got,
                _step(fence, mine, t),
                took,
                num_experts,
                label=f"{label} seed {seed} T{t} {routing}",
                expect_lamport=t <= _SKEW_LIMIT,
            )
        return failures

    sources = inputs(0)
    static = [source(g.rank) for source in sources]
    failures = check(sources, static, run(static), "eager")  # also the warm-up
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        steps = run(static)
    for replay in (1, 2):
        sources = inputs(1000 * replay)
        for tensors, source in zip(static, sources, strict=True):
            _refill(tensors, source(g.rank))
        graph.replay()
        failures += check(sources, static, steps, f"graph replay {replay}")
    return failures


def _child_main(body, args, rendezvous, rank, world, device, pipe) -> None:
    """Child process of _in_child_process: body(*args) in a gloo group of the
    children, on the parent's module (planted defect included)."""
    import torch.distributed as dist

    try:
        torch.cuda.set_device(device)
        # A child of a torchrun worker inherits TORCHELASTIC_USE_AGENT_STORE,
        # with which a tcp:// rendezvous only connects to a store that the
        # elastic agent hosts. The children's store is hosted by rank 0's child.
        os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)
        dist.init_process_group(
            "gloo",
            init_method=rendezvous,
            rank=rank,
            world_size=world,
            timeout=timedelta(minutes=5),
        )
        _use_planted_defect(setattr)
        _record_transports(setattr)
        failures = body(*args)
        dist.barrier()  # every rank has ended its last step
        for comm in _INSTANCES.values():
            comm.destroy()
    except Exception as exc:  # noqa: BLE001 - reported to the parent
        trace = traceback.format_exc(limit=-3).replace("\n", " | ")
        failures = [f"{type(exc).__name__}: {exc} [{trace[-600:]}]"]
    pipe.send(failures[:8])
    pipe.close()
    os._exit(0)  # no teardown of a CUDA context that a trap may have lost


def _in_child_process(body, *args, timeout=600) -> list:
    """body(*args) -> failures in a child process of every rank, with its own
    process group: a kernel that traps loses the child's CUDA context only."""
    import socket
    import sys

    import torch.multiprocessing as mp

    # The child re-imports this module by its dotted name (tests.moe_ep...) to
    # unpickle its target, on the parent's sys.path, which spawn restores before
    # unpickling. pytest.ini selects --import-mode=importlib, which does not put
    # the repo root there, and plain ``pytest`` (unlike ``python -m pytest``)
    # does not add the working directory either.
    test_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(test_dir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    g = _group()
    rendezvous = None
    if g.rank == 0:
        with socket.socket() as sock:
            sock.bind(("", 0))
            rendezvous = f"tcp://{socket.gethostname()}:{sock.getsockname()[1]}"
    rendezvous = _all_gather(rendezvous)[0]
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(
        target=_child_main,
        args=(body, args, rendezvous, g.rank, g.world, torch.cuda.current_device()),
        kwargs={"pipe": sender},
    )
    child.start()
    sender.close()
    failures = None
    if receiver.poll(timeout):
        with contextlib.suppress(EOFError):  # the child ended without a result
            failures = receiver.recv()
    child.join(30)
    if child.is_alive():
        child.kill()
        child.join()
    if failures is None:
        return [f"child process ended without a result, exit code {child.exitcode}"]
    return failures


@multi_rank
@collective
def test_rounds_under_rank_skew(group):
    """Rounds back to back, eagerly and in a captured graph, with no host
    synchronization or fence step in between, while one rank spins before its
    dispatch and one before its combine: fast ranks start round n + 1 while
    peers are still in round n (what the two sets of ids rows, the single
    landing rows of the payload and the order of the re-arms are for). T and
    routing change every round (self-routed tokens and ranks without tokens
    included), and the last rounds cross lamport_max_tokens. Afterwards every
    round against the fence instance, and every landing row re-armed. In a
    child process with a short timeout_sec, since a deadlock traps."""
    return _in_child_process(_rounds_under_skew, "bf16")


@multi_rank
@collective
def test_workspace_sharing(group):
    """Two equal Lamport instances share one workspace (and its epoch) and
    alternate steps."""
    num_experts = 32 * group.world
    first = _instance("bf16", 8, num_experts, 2048, 16, lamport=None, tag=2)
    second = _instance("bf16", 8, num_experts, 2048, 16, lamport=None, tag=3)
    fence = _instance("bf16", 8, num_experts, 2048, 16, lamport=False)
    failures = [] if first._state is second._state else ["instances do not share"]
    for i in range(6):
        failures += _check_round(
            first if i % 2 else second,
            fence,
            "bf16",
            800 + i,
            16,
            special=True,
            label=f"step {i}",
        )
    return failures


@multi_rank
@collective
def test_checkpoint_prepare_restore(group):
    """Steps and a captured graph after checkpoint_prepare / checkpoint_restore."""
    from flashinfer.comm.comm_backend import TorchDistBackend

    num_experts = 32 * group.world
    lamport, fence = _pair("bf16", 8, num_experts, 2048, 16, tag=4)
    static, refill = _graph_rounds(lamport, "bf16", (900, 901), (16, 5))

    def run(comm):
        return [_step(comm, inputs, t)[1] for t, inputs in static]

    run(lamport)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run(lamport)
    failures = []
    for cycle in range(2):
        lamport.checkpoint_prepare()
        lamport.checkpoint_restore(TorchDistBackend())
        failures += _check_round(
            lamport,
            fence,
            "bf16",
            910 + cycle,
            16,
            special=True,
            label=f"eager after restore {cycle}",
        )
        refill(cycle + 1)
        graph.replay()
        references = run(fence)
        torch.cuda.synchronize()
        for i, (got, ref) in enumerate(zip(outputs, references, strict=True)):
            if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                failures.append(
                    f"graph after restore {cycle}, step {i}: combine differs"
                )
    return failures


@multi_rank
@collective
def test_combine_takes_its_dispatch_transport(group):
    """The combine op refuses another transport than its dispatch's (a Lamport
    combine reads routing only a Lamport dispatch writes)."""
    num_experts = 32 * group.world
    comm = _instance(
        "bf16", 8, num_experts, 2048, 16, lamport=None, lamport_max_tokens=8, tag=6
    )
    counts = _token_counts(1100, group.world, 8)
    x, _, ids, w = _inputs(
        1100, group.rank, counts[group.rank], "bf16", 8, num_experts, 2048, group.world
    )
    failures = []
    for tokens, lamport_step in ((8, True), (16, False)):
        comm.dispatch(x, ids, w, runtime_max_tokens_per_rank=tokens)
        if comm._last_step_lamport != lamport_step:
            failures.append(f"T{tokens}: Lamport transport {comm._last_step_lamport}")
        expert_output = torch.zeros(
            group.world * tokens, 2048, dtype=torch.bfloat16, device="cuda"
        )
        round_state = dict(comm._round)
        comm._round["lamport"] = not lamport_step
        try:
            comm.combine(expert_output)
            failures.append(f"T{tokens}: a combine on the other transport ran")
        except Exception as exc:  # noqa: BLE001 - the op's check
            if "Lamport" not in str(exc):
                failures.append(f"T{tokens}: unexpected {type(exc).__name__}: {exc}")
        comm._round = round_state
        comm._state["round_owner"] = comm
        comm.combine(expert_output)
    torch.cuda.synchronize()
    return failures


@multi_rank
@collective
def test_mixed_configs(group):
    """Ranks constructed with different configs agree: rank 0 alone turning
    Lamport (or CFT) off turns it off on every rank, whose steps then match the
    fence instance, and different explicit lamport_max_tokens fail everywhere."""
    num_experts = 32 * group.world
    first = group.rank == 0
    fence = _instance("bf16", 8, num_experts, 2048, 16, lamport=False)
    no_lamport = _instance(
        "bf16", 8, num_experts, 2048, 16, lamport=False if first else None, tag=7
    )
    no_cft = _instance(
        "bf16", 8, num_experts, 2048, 16, lamport=None, cft=False if first else None
    )
    failures = []
    if no_lamport.lamport_enabled or no_cft._cft_enabled or not no_cft.lamport_enabled:
        failures.append("an instance did not take rank 0's transports")
    for i, (comm, lamport_step) in enumerate(((no_lamport, False), (no_cft, True))):
        failures += _check_round(
            comm,
            fence,
            "bf16",
            1300 + i,
            16,
            special=True,
            label=f"instance {i}",
            expect_lamport=lamport_step,
        )
    try:
        limit = 8 if first else 12
        _instance(
            "bf16", 8, num_experts, 2048, 16, lamport=None, lamport_max_tokens=limit
        )
        failures.append("different lamport_max_tokens were accepted")
    except ValueError as exc:
        if "lamport_max_tokens" not in str(exc):
            failures.append(f"unexpected ValueError: {exc}")
    return failures


# ----------------------------------------------------------------------------
# Host-only tests
# ----------------------------------------------------------------------------


def _params(**overrides):
    from flashinfer.moe_ep import MoEEpCommParams

    fields = dict(num_experts=256, top_k=8, max_tokens_per_rank=512, hidden_size=7168)
    fields.update(overrides)
    return MoEEpCommParams(**fields)


_LAYOUT = SimpleNamespace(
    LAMPORT_MAX_RANKS=64,
    LAMPORT_MAX_TOP_K=22,
    LAMPORT_MAX_TOKENS_PER_RANK=512,
    LAMPORT_DISPATCH_BLOCKS_PER_SM=6,
    LAMPORT_IDS_SENTINEL=_IDS_SENTINEL,
    LAMPORT_FORMAT_BF16=0,
    LAMPORT_FORMAT_MXFP8=1,
    LAMPORT_FORMAT_FP8_BLOCK128=2,
    LAMPORT_FORMAT_NVFP4=3,
)
Row = one_sided._LamportRow


def test_default_token_limit_rule():
    rule = one_sided._default_lamport_max_tokens
    # DeepSeek-V3 BF16 (hidden 7168): 7 dispatch blocks per token, so one wave
    # of 6 blocks per SM holds 137 tokens, below the 159-token floor.
    dsv3 = Row(14336, 0, 7, 43168)
    assert rule(4, 160, dsv3, _LAYOUT) == 159
    assert rule(64, 152, dsv3, _LAYOUT) == 159
    # Hidden 4096 BF16: 4 blocks, so the one-wave bound decides up to 8 ranks;
    # at 16 ranks the 64 MiB bound on all Lamport regions does.
    h4096 = Row(8192, 0, 4, 24736)
    assert rule(4, 160, h4096, _LAYOUT) == 239
    assert rule(8, 152, h4096, _LAYOUT) == 227
    assert rule(16, 152, h4096, _LAYOUT) == (64 << 20) // (16 * 24736) == 169
    # NVFP4 rows take 2 blocks; the 64 MiB bound lowers the limit as ep_size
    # grows, down to the floor.
    nvfp4 = Row(3584, 448, 2, 22560)
    assert rule(4, 160, nvfp4, _LAYOUT) == 479
    assert rule(8, 160, nvfp4, _LAYOUT) == (64 << 20) // (8 * 22560) == 371
    assert rule(32, 160, nvfp4, _LAYOUT) == 159
    # The kernels' 512-token bound.
    assert rule(2, 152, Row(512, 64, 1, 3360), _LAYOUT) == 512


@pytest.mark.skipif(not torch.cuda.is_available(), reason="loads the kernel module")
def test_lamport_row_geometry():
    """The row geometry the kernel module reports for the Python-side rules."""
    module = one_sided.get_nvlink_one_sided_alltoall_module()
    index = one_sided.layout_constants()

    def row(fmt, hidden, top_k):
        return Row(*map(int, module.moe_a2a_lamport_row_bytes(fmt, hidden, top_k)))

    assert row(index.LAMPORT_FORMAT_BF16, 7168, 8) == Row(14336, 0, 7, 43168)
    assert row(index.LAMPORT_FORMAT_NVFP4, 7168, 8) == Row(3584, 448, 2, 22560)
    assert row(index.LAMPORT_FORMAT_MXFP8, 3584, 16) == Row(3584, 112, 2, 14992)
    assert row(index.LAMPORT_FORMAT_FP8_BLOCK128, 7168, 8) == Row(7168, 224, 4, 29280)
    with pytest.raises(Exception, match="% 256"):
        row(index.LAMPORT_FORMAT_MXFP8, 2880, 8)


class _FakeModule:
    def __init__(self, available=True):
        self.available = available

    def moe_a2a_lamport_available(self):
        return self.available

    def moe_a2a_lamport_row_bytes(self, fmt, hidden, top_k):
        if hidden % 256:
            raise ValueError(f"Lamport rows need hidden_size % 256 == 0, got {hidden}")
        return [hidden, hidden // 32, 2, 4 * hidden]


def test_lamport_probe_reasons():
    from flashinfer.fused_moe import QuantFormat
    from flashinfer.moe_ep import NVLinkOneSidedConfig

    def reason(params=None, config=None, ep_size=8, module=None):
        return one_sided._lamport_unsupported_reason(
            module or _FakeModule(),
            params or _params(),
            config or NVLinkOneSidedConfig(),
            ep_size,
            _LAYOUT,
        )

    assert reason() is None
    assert reason(_params(dispatch_format=QuantFormat.BF16)) is None
    assert "no Lamport kernels" in reason(module=_FakeModule(available=False))
    assert "64 EP ranks" in reason(ep_size=72)
    assert "BF16 expert outputs" in reason(_params(dtype=torch.float16))
    assert "FP8 combine" in reason(
        config=NVLinkOneSidedConfig(use_low_precision_combine=True)
    )
    assert "dispatch format" in reason(
        _params(dispatch_format=QuantFormat.FP8PerTensor)
    )
    assert "% 256" in reason(
        _params(hidden_size=2880, dispatch_format=QuantFormat.MXFP8)
    )
    # The sentinel as a signed or an unsigned int32.
    assert "sentinel" in reason(_params(invalid_expert_id=_IDS_SENTINEL))
    assert "sentinel" in reason(_params(invalid_expert_id=0x88888888))
    # A module without the Lamport ops is a reason, not an exception.
    assert "probe failed" in one_sided._lamport_unsupported_reason(
        SimpleNamespace(), _params(), NVLinkOneSidedConfig(), 8, _LAYOUT
    )


def _host_instance(fmt, params, max_tokens=16, row=None, **config):
    """An instance with only the attributes the per-step choice reads."""
    from flashinfer.moe_ep import NVLinkOneSidedConfig

    comm = object.__new__(one_sided.NVLinkOneSidedAlltoAll)
    comm.config = NVLinkOneSidedConfig(**config)
    comm.params = params
    comm.lamport_max_tokens = max_tokens
    comm._lamport_format = fmt
    comm._lamport_row = row
    comm._state = {"cft_ready": config.get("cft") is not False}
    return comm


def test_lamport_carries():
    """The per-step decision uses the step's token count, EPLB statistics and
    the payload layout only."""
    from flashinfer.fused_moe import QuantFormat

    def carries(fmt, row, params, x, scale=None, w=None, tokens=8, eplb=None):
        comm = _host_instance(fmt, params, row=row)
        return comm._lamport_carries(tokens, x, scale, w, eplb)

    p = _params(hidden_size=512)
    bf16 = Row(1024, 0, 1, 0)
    x = torch.zeros(4, 512, dtype=torch.bfloat16)
    w = torch.zeros(4, 8)
    assert carries("BF16", bf16, p, x, w=w) and carries("BF16", bf16, p, x)
    assert carries("BF16", bf16, p, x[:0], w=w[:0])  # a rank without tokens
    assert not carries("BF16", bf16, p, x, w=w, tokens=17)
    assert not carries("BF16", bf16, p, x, w=w, eplb=torch.zeros(4))
    assert not carries("BF16", bf16, p, x.view(torch.float16), w=w)
    assert not carries("BF16", bf16, p, x, w=w.half())
    assert not carries("BF16", bf16, p, x, scale=torch.zeros(4, 16, dtype=torch.uint8))
    fp4 = _params(hidden_size=512, dispatch_format=QuantFormat.NVFP4)
    nvfp4 = Row(256, 32, 1, 0)
    q = torch.zeros(4, 256, dtype=torch.uint8)
    s = torch.zeros(4, 32, dtype=torch.uint8)
    assert carries("NVFP4", nvfp4, fp4, q, s, w)
    assert not carries("NVFP4", nvfp4, fp4, q, None, w)
    assert not carries("NVFP4", nvfp4, fp4, q, s[:, :16], w)
    assert not carries("NVFP4", nvfp4, fp4, q.view(torch.bfloat16), s, w)
    b128 = _params(hidden_size=512, dispatch_format=QuantFormat.DeepSeekFp8)
    block128 = Row(512, 16, 1, 0)
    e4m3 = torch.zeros(4, 512, dtype=torch.uint8)
    assert carries("FP8_BLOCK128", block128, b128, e4m3, torch.zeros(4, 4), w)
    assert not carries(
        "FP8_BLOCK128", block128, b128, e4m3, torch.zeros(4, 32, dtype=torch.uint8), w
    )


def test_split_layer_keeps_packed_rows_off_lamport():
    """A split-layer kernel that packs its dispatch rows into other bytes
    turns the Lamport transport off, also when the comm backend is a name."""
    from flashinfer.moe_ep import NVLinkOneSidedConfig
    from flashinfer.moe_ep.modes.split_layer import _communication_backend

    config = NVLinkOneSidedConfig()
    packing = SimpleNamespace(packs_dispatch_payload=True)
    assert _communication_backend(config, packing).lamport is False
    assert config.lamport is None
    named = _communication_backend("nvlink_one_sided", packing)
    assert named == NVLinkOneSidedConfig(lamport=False)
    plain = SimpleNamespace(packs_dispatch_payload=False)
    assert _communication_backend(config, plain) is config
    assert _communication_backend("nvlink_one_sided", plain) == "nvlink_one_sided"
    other = SimpleNamespace(backend_name="cake")
    assert _communication_backend(other, packing) is other
    assert _communication_backend("cake", packing) == "cake"
