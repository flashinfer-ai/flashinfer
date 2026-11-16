"""Source tests (no GPU) of the sm_107a full-grid admission of the ``_tail``
pair program and the per-capability chunk-parallel workspace cap
(CAKE-2549 round 9, Round A; measured on R200, Hecate lease 799171)."""

import pytest

import importlib
import math


def _module():
    return importlib.import_module("flashinfer.mamba.cake_ssd_combined")


@pytest.mark.cpu_only
def test_source_tail_full_grid_rule_is_rubin_only():
    """One admission rule, keyed on (10, 7); Blackwell capabilities have none."""

    module = _module()
    assert set(module._TAIL_FULL_GRID_RULES) == {(10, 7)}
    rule = module.tail_full_grid_rule((10, 7))
    assert rule == module._TailFullGridRule(
        min_chunks_batched=16,
        min_chunks_varlen=8,
        render_penalty_batched=1 / 12,
        render_penalty_varlen=1 / 5,
        long_chunks_varlen=16,
        render_penalty_varlen_long=1 / 8,
    )
    assert rule.min_chunks(False) == 16 and rule.min_chunks(True) == 8
    assert rule.render_penalty(False) == 1 / 12 and rule.render_penalty(True) == 1 / 5
    # varlen items of 16 or more chunks take the long-item penalty unless the
    # call has z; the batched penalty depends on neither
    assert rule.render_penalty(True, 15) == 1 / 5
    assert rule.render_penalty(True, 16) == 1 / 8
    assert rule.render_penalty(True, 64) == 1 / 8
    assert rule.render_penalty(True, 64, has_z=True) == 1 / 5
    assert rule.render_penalty(False, 64) == 1 / 12
    assert rule.render_penalty(False, 64, has_z=True) == 1 / 12
    assert module.tail_full_grid_rule((10, 0)) is None
    assert module.tail_full_grid_rule((10, 3)) is None
    assert module._TAIL_MIN_CHUNKS == 16


@pytest.mark.cpu_only
def test_source_tail_route_default_arguments_keep_the_blackwell_rule():
    """With the default arguments ``tail_selected`` is the round-7 body on
    every device, and a full persistent grid never takes the pair program."""

    module = _module()

    def expected(items, chunks, sm_count):
        pairs = sm_count // 2
        grid = module._persistent_grid_size(total_work=items, sm_count=sm_count)
        return (
            items > sm_count
            and grid < sm_count
            and chunks >= 16
            and math.ceil(items / pairs) / 2 < math.ceil(items / grid)
        )

    for sm_count in (148, 152, 212):
        for items in (
            64,
            128,
            148,
            152,
            212,
            216,
            256,
            296,
            304,
            384,
            512,
            640,
            768,
            1024,
            2048,
            4096,
        ):
            for chunks in (1, 4, 8, 15, 16, 17, 31, 32, 64):
                for mode_varlen in (False, True):
                    assert module.tail_selected(
                        total_work=items,
                        chunks_per_item=chunks,
                        sm_count=sm_count,
                        mode_varlen=mode_varlen,
                    ) == expected(items, chunks, sm_count), (sm_count, items, chunks)
    # the round-7 Blackwell rows keep their route
    for sm_count in (148, 152):
        assert module.tail_selected(
            total_work=512, chunks_per_item=16, sm_count=sm_count
        )
        assert module.tail_selected(
            total_work=512, chunks_per_item=32, sm_count=sm_count
        )
        assert not module.tail_selected(
            total_work=256, chunks_per_item=32, sm_count=sm_count
        )
    # on 212 SMs the default rule routes none of the gate geometry
    for items, chunks in (
        (256, 32),
        (512, 16),
        (512, 32),
        (256, 8),
        (512, 8),
        (640, 16),
    ):
        for mode_varlen in (False, True):
            assert not module.tail_selected(
                total_work=items,
                chunks_per_item=chunks,
                sm_count=212,
                mode_varlen=mode_varlen,
            )


# The round-9 R200 screens (6-round focused A/B vs the l1b tree, ratio =
# start / candidate): label, items, chunks per item, varlen, admitted.
R200_SCREEN_ROWS = (
    ("perf_varlen_32x32", 256, 32, True, True),  # 1.096
    ("perf_batched_64x16", 512, 16, False, True),  # 1.047
    # 0.936 in round 9; the current varlen pair render wins 1.159 (long-item
    # varlen penalty)
    ("perf_varlen_64x32", 512, 32, True, True),
    ("perf_varlen_32x8", 256, 8, True, True),  # 1.026 (chunk floor 8)
    ("perf_varlen_64x8", 512, 8, True, False),  # 0.864 (main kernel 0.815)
    (
        "nem_v1024x2",
        256,
        8,
        True,
        True,
    ),  # 1.027 bf16 / 1.072 f32 (2 x 1024 tokens x 128 heads)
    ("perf_varlen_128x32", 1024, 32, True, False),  # control 1.0006
    ("perf_batched_128x16", 1024, 16, False, False),  # control 0.9996
    (
        "nem_v128p896",
        256,
        4,
        True,
        False,
    ),  # control 0.9995 (mean 512 tokens = 4 chunks)
    ("perf_batched_128x4", 1024, 4, False, False),  # control 0.998
)


@pytest.mark.cpu_only
def test_source_tail_full_grid_rule_reproduces_the_r200_screens():
    module = _module()
    rule = module.tail_full_grid_rule((10, 7))
    for label, items, chunks, mode_varlen, admitted in R200_SCREEN_ROWS:
        assert (
            module.tail_selected(
                total_work=items,
                chunks_per_item=chunks,
                sm_count=212,
                mode_varlen=mode_varlen,
                full_grid_rule=rule,
            )
            is admitted
        ), label


@pytest.mark.cpu_only
def test_source_tail_full_grid_rule_geometry():
    """The (items x chunks x form) grid measured on R200 (persistent /
    pair-program time): wins are admitted, losses and equal makespans are
    refused."""

    module = _module()
    rule = module.tail_full_grid_rule((10, 7))

    def route(items, chunks, mode_varlen):
        return module.tail_selected(
            total_work=items,
            chunks_per_item=chunks,
            sm_count=212,
            mode_varlen=mode_varlen,
            full_grid_rule=rule,
        )

    # 256 items: 2 waves vs 1.5 rounds -- every form wins (batched 1.12-1.26,
    # varlen 1.04-1.10, 128-head varlen 1.08-1.13)
    for chunks in (8, 16, 32):
        assert route(256, chunks, True)
    for chunks in (16, 32):
        assert route(256, chunks, False)
    assert not route(256, 8, False)  # batched floor 16 (a fused-window call)
    # 512 items: 3 waves vs 2.5 rounds -- batched 16 / 32 chunks win (1.06 /
    # 1.13); batched 8 chunks (0.98) does not; varlen items of 16 or more
    # chunks take the long-item penalty (2.5 x 9 / 8 + one hand-off chunk
    # step < 3) and win (32 chunks: 1.111), 8-chunk varlen items do not
    assert route(512, 16, False) and route(512, 32, False)
    assert not route(512, 8, False)
    assert not route(512, 8, True)
    for chunks in (16, 32, 64):
        assert route(512, chunks, True)
    # 640 items: 4 waves vs 3.5 rounds -- batched wins (1.07-1.13); varlen
    # items need more than 16 chunks (3.5 x 9 / 8 + 1 / 16 = 4 is not below 4)
    assert route(640, 16, False) and route(640, 32, False)
    for chunks in (8, 16):
        assert not route(640, chunks, True)
    assert route(640, 32, True)
    # equal makespans (384: 2 vs 2, 768: 4 vs 4, 1024: 5 vs 5, 2048: 10 vs
    # 10) lose by the pair overhead on every form
    for items in (384, 768, 1024, 2048):
        for chunks in (8, 16, 32, 64):
            for mode_varlen in (False, True):
                assert not route(items, chunks, mode_varlen), (
                    items,
                    chunks,
                    mode_varlen,
                )
    # the item ladder at 16 chunks: 213..255 items win 1.09-1.27 on both
    # forms; at most one item per SM never reaches the pair program
    for items in (216, 224, 240, 248):
        assert route(items, 16, False) and route(items, 16, True)
    for items in (128, 200, 208, 212):
        for mode_varlen in (False, True):
            assert not route(items, 16, mode_varlen)
    # 4096 items: 20 waves vs 19.5 rounds, a saving below the render penalty
    assert not route(4096, 32, False) and not route(4096, 32, True)


@pytest.mark.cpu_only
def test_source_tail_full_grid_rule_stays_disjoint_from_the_fused_window():
    """No shape is both ``fused_selected`` and admitted to the pair program on
    212 SMs: the admitted calls exceed one item per SM, so batched calls of
    >= 16 chunks exceed 2048 chunk-items and varlen calls of >= 8 chunks
    exceed 512 chunk-heads."""

    module = _module()
    rule = module.tail_full_grid_rule((10, 7))
    for nheads in (8, 128):
        for num_sequences in (1, 2, 4, 8, 16, 19, 27, 32, 37, 64, 80, 128, 256):
            items = num_sequences * nheads
            for chunks in (1, 2, 4, 8, 15, 16, 17, 32, 64):
                for mode_varlen in (False, True):
                    tail = module.tail_selected(
                        total_work=items,
                        chunks_per_item=chunks,
                        sm_count=212,
                        mode_varlen=mode_varlen,
                        full_grid_rule=rule,
                    )
                    fused = module.fused_selected(
                        nheads=nheads,
                        num_sequences=num_sequences,
                        nchunks=chunks,
                        mode_varlen=mode_varlen,
                        total_tokens=num_sequences * chunks * 128,
                    )
                    assert not (tail and fused), (
                        nheads,
                        num_sequences,
                        chunks,
                        mode_varlen,
                    )


@pytest.mark.cpu_only
def test_source_chunk_parallel_workspace_cap_rubin():
    """The chunk-parallel workspace cap is 512 MiB on (10, 7) and 256 MiB
    elsewhere (the 1 x 1024-chunk and 4 x 256-chunk 8-head calls, 384 MiB,
    take the chunk-parallel family on R200: 7.1x / 1.98x over the exact scan)."""

    module = _module()
    assert module._CHUNK_PARALLEL_WORKSPACE_CAP_BYTES == 256 << 20
    assert module._CHUNK_PARALLEL_WORKSPACE_CAP_BYTES_BY_CAPABILITY == {
        (10, 7): 512 << 20
    }
    assert module.chunk_parallel_workspace_cap_bytes((10, 7)) == 512 << 20
    for capability in ((10, 0), (10, 3), (9, 0)):
        assert module.chunk_parallel_workspace_cap_bytes(capability) == 256 << 20
    assert 8192 * 48 * 1024 == 384 << 20
    assert module._CHUNK_PARALLEL_WORKSPACE_CAP_BYTES < 384 << 20
    assert module.chunk_parallel_workspace_cap_bytes((10, 7)) >= 384 << 20


@pytest.mark.cpu_only
def test_source_tail_full_grid_long_varlen_items_without_z():
    """The long-item varlen penalty (1/8 from 16 chunks per item) admits
    full-grid varlen calls without z of 425-530 items with 16 or more chunks
    and of 637-742 items with more than 16 chunks; calls with z, 8-chunk
    items and the batched form keep their admission."""

    module = _module()
    rule = module.tail_full_grid_rule((10, 7))

    def route(items, chunks, mode_varlen=True, has_z=False):
        return module.tail_selected(
            total_work=items,
            chunks_per_item=chunks,
            sm_count=212,
            mode_varlen=mode_varlen,
            full_grid_rule=rule,
            has_z=has_z,
        )

    def full_grid(items):
        return module._persistent_grid_size(total_work=items, sm_count=212) == 212

    for items in range(213, 1100):
        if not full_grid(items):
            continue
        for chunks in (8, 15, 16, 17, 24, 32, 64):
            pair_rounds = math.ceil(items / 106)
            waves = math.ceil(items / 212)
            hand_off = (pair_rounds % 2) / chunks

            def admitted(penalty):
                return pair_rounds / 2 * (1 + penalty) + hand_off < waves

            short = admitted(1 / 5)
            long = admitted(1 / 8) if chunks >= 16 else short
            assert route(items, chunks) is long, (items, chunks)
            assert route(items, chunks, has_z=True) is short, (items, chunks)
            if long and not short:
                assert 425 <= items <= 530 or (637 <= items <= 742 and chunks > 16)
            batched = module.tail_selected(
                total_work=items,
                chunks_per_item=chunks,
                sm_count=212,
                mode_varlen=False,
                full_grid_rule=rule,
                has_z=False,
            )
            assert batched is module.tail_selected(
                total_work=items,
                chunks_per_item=chunks,
                sm_count=212,
                mode_varlen=False,
                full_grid_rule=rule,
                has_z=True,
            )
    # perf_varlen_64x32 (512 items x 32 chunks, no z): 1.16x on R200
    assert route(512, 32) and not route(512, 32, has_z=True)
    assert route(512, 16) and not route(512, 16, has_z=True)
    assert not route(512, 8) and not route(640, 16) and route(640, 32)
    # the Blackwell default (no rule) never admits a full grid
    assert not module.tail_selected(
        total_work=512, chunks_per_item=32, sm_count=212, mode_varlen=True
    )
