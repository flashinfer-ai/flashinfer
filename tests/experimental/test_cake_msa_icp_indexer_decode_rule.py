"""The decode score plane's record and split count per layout, batch and architecture.

Layouts (page size ``P``, world size ``W``, local K rows per page ``R = P // W``): P128 W4 (R32), P256 W4 and P128 W2
(R64), P256 W2 (R128).  The rule:

* sm_103a: R32 runs the four-CTA-per-SM plane record (``decode_plane_r32_s2k4``) at one wave
  (``min(128, SMs * 4 // B)``) below batch 14 and at ``min(one wave, 32)`` from batch 14 on; R64 / R128 run the base
  plane records (history K read before the grid dependency wait) at one wave.
* sm_100a: every layout runs the base plane record at one wave.
* sm_107a: every layout runs its own plane record at one wave.

The list scorer of the fused path keeps its records and split cap.  The facade's decode without a caller pool runs
the fused chain (list scorer and list publisher, in a pool the facade allocates) on sm_107a at P128 W4 and W2
from 8 requests on, and the score plane everywhere else.  The CPU tests pin the records, the split counts and the chain rule
exactly; the GPU test checks that the plane the rule launches is bit-identical to the plane at one split and at one
wave on the device's architecture.
"""

from __future__ import annotations

import importlib
import os

import pytest
import torch

PACKAGE = os.environ.get(
    "CAKE_MSA_ICP_PACKAGE", "flashinfer.experimental.cake_msa_icp_indexer"
)
cake_jit = importlib.import_module(f"{PACKAGE}.cake_jit")
cake_backend = importlib.import_module(f"{PACKAGE}.cake_backend")

BATCHES = (1, 2, 4, 8, 13, 14, 16)
LAYOUTS = ((128, 4), (256, 4), (128, 2), (256, 2))  # (page size, world size)
SMS = {"sm_100a": 148, "sm_103a": 152, "sm_107a": 216}

# (arch, local rows per page) -> (plane record, CTAs per SM, splits at BATCHES)
PLANE = {
    ("sm_103a", 32): ("decode_plane_r32_s2k4", 4, (128, 128, 128, 76, 46, 32, 32)),
    ("sm_103a", 64): ("decode_plane_r64", 2, (128, 128, 76, 38, 23, 21, 19)),
    ("sm_103a", 128): ("decode_plane_r128", 1, (128, 76, 38, 19, 11, 10, 9)),
    ("sm_100a", 32): ("decode_plane_r32", 2, (128, 128, 74, 37, 22, 21, 18)),
    ("sm_100a", 64): ("decode_plane_r64", 2, (128, 128, 74, 37, 22, 21, 18)),
    ("sm_100a", 128): ("decode_plane_r128", 1, (128, 74, 37, 18, 11, 10, 9)),
    ("sm_107a", 32): ("decode_plane_r32_sm107", 1, (128, 108, 54, 27, 16, 15, 13)),
    ("sm_107a", 64): ("decode_plane_r64_sm107", 1, (128, 108, 54, 27, 16, 15, 13)),
    ("sm_107a", 128): ("decode_plane_r128_sm107", 1, (128, 108, 54, 27, 16, 15, 13)),
}
# (arch, local rows per page) -> (list record, splits at BATCHES): unchanged by the plane rule.
TOPK = {
    ("sm_103a", 32): ("decode_topk16_r32", (48, 48, 48, 38, 23, 21, 19)),
    ("sm_103a", 64): ("decode_topk16_r64", (48, 48, 48, 38, 23, 21, 19)),
    ("sm_103a", 128): ("decode_topk16_r128", (48, 48, 38, 19, 11, 10, 9)),
    ("sm_100a", 32): ("decode_topk16_r32", (48, 48, 48, 37, 22, 21, 18)),
    ("sm_100a", 64): ("decode_topk16_r64", (48, 48, 48, 37, 22, 21, 18)),
    ("sm_100a", 128): ("decode_topk16_r128", (48, 48, 37, 18, 11, 10, 9)),
    ("sm_107a", 32): ("decode_topk16_r32_sm107", (48, 48, 48, 27, 16, 15, 13)),
    ("sm_107a", 64): ("decode_topk16_r64_sm107", (48, 48, 48, 27, 16, 15, 13)),
    ("sm_107a", 128): ("decode_topk16_r128_sm107", (48, 48, 48, 27, 16, 15, 13)),
}


@pytest.fixture
def sms(monkeypatch: pytest.MonkeyPatch):
    """Device index -> SM count: device ``i`` is the ``i``-th architecture of ``SMS``."""
    counts = list(SMS.values())
    monkeypatch.setattr(cake_jit, "sm_count", lambda device_index: counts[device_index])
    return {arch: index for index, arch in enumerate(SMS)}


@pytest.mark.parametrize("arch", list(SMS))
@pytest.mark.parametrize(("page", "world"), LAYOUTS)
def test_the_plane_record_and_splits(sms, arch: str, page: int, world: int) -> None:
    if not any(arch in record["arches"] for record in cake_jit.MODULES.values()):
        pytest.skip(f"no {arch} records in this build")
    rows = page // world
    name, ctas, want = PLANE[(arch, rows)]
    assert cake_jit.route("decode_plane", rows, arch) == name
    assert cake_jit.route("decode_plane", rows, arch, page) == name
    assert cake_jit.ctas_per_sm(name) == ctas
    got = tuple(
        cake_backend.plane_splits(name, rows, arch, batch, sms[arch])
        for batch in BATCHES
    )
    assert got == want


@pytest.mark.parametrize("arch", list(SMS))
@pytest.mark.parametrize(("page", "world"), LAYOUTS)
def test_the_list_scorer_is_unchanged(sms, arch: str, page: int, world: int) -> None:
    if not any(arch in record["arches"] for record in cake_jit.MODULES.values()):
        pytest.skip(f"no {arch} records in this build")
    rows = page // world
    name, want = TOPK[(arch, rows)]
    assert cake_jit.route("decode_topk", rows, arch) == name
    cap = cake_jit.TOPK16_SPLIT_CAP[arch]
    got = tuple(
        cake_backend.decode_splits(name, batch, sms[arch], cap=cap) for batch in BATCHES
    )
    assert got == want


def test_the_rows_32_cap_starts_at_batch_14_on_sm_103a(sms) -> None:
    name = "decode_plane_r32_s2k4"
    got = {
        batch: cake_backend.plane_splits(name, 32, "sm_103a", batch, sms["sm_103a"])
        for batch in range(1, 41)
    }
    one_wave = {batch: min(128, 152 * 4 // batch) for batch in got}
    assert (
        cake_backend.PLANE_R32_LARGE_BATCH == 14
        and cake_backend.PLANE_R32_LARGE_BATCH_SPLITS == 32
    )
    assert all(got[batch] == one_wave[batch] for batch in range(1, 14))
    assert all(got[batch] == min(one_wave[batch], 32) for batch in range(14, 41))
    assert [got[batch] for batch in (13, 14, 15, 16, 17, 19, 20, 40)] == [
        46,
        32,
        32,
        32,
        32,
        32,
        30,
        15,
    ]
    # The cap belongs to the rows 32 plane on sm_103a only.
    assert (
        cake_backend.plane_splits("decode_plane_r64", 64, "sm_103a", 8, sms["sm_103a"])
        == 38
    )
    assert (
        cake_backend.plane_splits("decode_plane_r32", 32, "sm_100a", 14, sms["sm_100a"])
        == 21
    )


# (arch, page size, world size) -> first decode batch of the facade's fused chain; every other layout: None.
FUSED_RULE = {("sm_107a", 128, 4): 8, ("sm_107a", 128, 2): 8}


@pytest.mark.parametrize("arch", list(SMS))
@pytest.mark.parametrize(("page", "world"), LAYOUTS)
def test_the_facade_chain_rule(arch: str, page: int, world: int) -> None:
    if not any(arch in record["arches"] for record in cake_jit.MODULES.values()):
        pytest.skip(f"no {arch} records in this build")
    want = FUSED_RULE.get((arch, page, world))
    assert cake_backend.fused_decode_min_batch(arch, page, world) == want
    if want is not None:
        # the rule's chain is delivered on that architecture
        assert cake_jit.has_role("decode_topk", arch)
        assert cake_jit.has_role("topk16_publish", arch)
        assert cake_jit.route("decode_topk", page // world, arch)
    assert cake_backend.FUSED_DECODE_MIN_BATCH == FUSED_RULE


def test_the_rule_reads_no_environment(sms, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [key for key in os.environ if key.startswith("FI_CAKE_MSA_ICP")]:
        monkeypatch.delenv(key)
    before = cake_backend.plane_splits(
        "decode_plane_r32_s2k4", 32, "sm_103a", 14, sms["sm_103a"]
    )
    monkeypatch.setenv("FI_CAKE_MSA_ICP_EXP_PLANE_SPLITS", "7")
    monkeypatch.setenv("FI_CAKE_MSA_ICP_EXP_R32_PLANE", "base")
    monkeypatch.setenv("FI_CAKE_MSA_ICP_EXP_T16_MIN_BATCH", "1")
    fused_107 = cake_jit.has_role("decode_topk", "sm_107a") and cake_jit.has_role(
        "topk16_publish", "sm_107a"
    )
    assert cake_backend.fused_decode_min_batch("sm_107a", 128, 4) == (
        8 if fused_107 else None
    )
    assert cake_backend.fused_decode_min_batch("sm_107a", 128, 2) == (
        8 if fused_107 else None
    )
    assert cake_backend.fused_decode_min_batch("sm_107a", 256, 4) is None
    assert cake_backend.fused_decode_min_batch("sm_103a", 128, 4) is None
    assert (
        cake_backend.plane_splits(
            "decode_plane_r32_s2k4", 32, "sm_103a", 14, sms["sm_103a"]
        )
        == before
        == 32
    )
    assert not hasattr(cake_backend, "EXP_PLANE_SPLITS_ENV") and not hasattr(
        cake_backend, "EXP_R32_PLANE_ENV"
    )


# --------------------------------------------------------------------------- GPU


def _device_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    return cake_jit.arch_for_capability(torch.cuda.get_device_capability(0))


@pytest.mark.skipif(
    _device_arch() is None, reason="needs an sm_100a / sm_103a / sm_107a GPU"
)
@pytest.mark.parametrize(("page", "world"), LAYOUTS)
def test_the_plane_at_the_rule_matches_one_split_and_one_wave(
    monkeypatch: pytest.MonkeyPatch, page: int, world: int
) -> None:
    arch = _device_arch()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    rows = page // world
    name = cake_jit.route("decode_plane", rows, arch)
    launched: list[int] = []
    run_module = cake_jit.run_module
    rule_splits = cake_backend.plane_splits
    forced: list[int] = []

    def spy(record, values, **kwargs):
        launched.append(values["splits"])
        return run_module(record, values, **kwargs)

    monkeypatch.setattr(cake_jit, "run_module", spy)
    monkeypatch.setattr(
        cake_backend,
        "plane_splits",
        lambda *args: forced[0] if forced else rule_splits(*args),
    )
    gen = torch.Generator(device=dev).manual_seed(page * 10 + world)
    heads = cake_backend.NUM_INDEX_HEADS
    ctx = 24 * page
    pages_per_req = ctx // page
    for batch in BATCHES:
        for dql in (1, 4):
            total = batch * pages_per_req + 3
            k = (torch.randn(total, rows, 128, device=dev, generator=gen) * 2).to(
                torch.float8_e4m3fn
            )
            q = (
                torch.randn(batch * dql, heads, 128, device=dev, generator=gen) * 2
            ).to(torch.float8_e4m3fn)
            bt = torch.randperm(total, device=dev, generator=gen)[
                : batch * pages_per_req
            ].to(torch.int32)
            bt = bt.view(batch, pages_per_req).contiguous()
            seq = torch.randint(
                ctx // 3,
                ctx + 1,
                (batch,),
                device=dev,
                generator=gen,
                dtype=torch.int32,
            )
            seq[0] = ctx
            cols = (pages_per_req + 3) // 4 * 4
            rule = rule_splits(name, rows, arch, batch, 0)
            wave = cake_backend.decode_splits(name, batch, 0)
            planes = {}
            for label, splits in (("rule", None), ("one", 1), ("wave", wave)):
                forced[:] = [] if splits is None else [splits]
                launched.clear()
                out = torch.full((batch * dql, heads, cols), float("nan"), device=dev)
                cake_backend.decode_index_score_sharded(
                    q,
                    k,
                    bt,
                    seq,
                    out,
                    rank=world - 1,
                    world_size=world,
                    page_size=page,
                    decode_query_len=dql,
                    max_decode_query_len=dql,
                    enable_pdl=label == "rule",
                )
                torch.cuda.synchronize()
                assert launched == [rule if splits is None else splits], (
                    label,
                    launched,
                )
                planes[label] = out.view(torch.int32)
            assert torch.isfinite(planes["rule"].view(torch.float32)).any()
            for label in ("one", "wave"):
                assert torch.equal(planes[label], planes["rule"]), (batch, dql, label)
