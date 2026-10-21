"""Page-aligned raw selector history for speculative rollback."""

import pytest
import torch

from b12x.attention import qsa


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def reject_cuda():
        pytest.fail("raw-ring layout tests must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", reject_cuda)


@pytest.mark.parametrize(
    "page,drafts,ring",
    [
        (64, 0, 4),
        (64, 1, 8),
        (64, 3, 8),
        (64, 4, 8),
        (64, 7, 16),
        (64, 15, 32),
        (64, 31, 64),
        (64, 60, 64),
        (48, 7, 12),
        (96, 7, 12),
        (1520, 15, 20),
    ],
)
def test_raw_ring_covers_horizon_without_changing_valid_layouts(page, drafts, ring):
    layout = qsa.cache_requirements(
        main_page_size=page,
        kv_heads=2,
        head_dim=256,
        index_head_dim=128,
        max_speculative_tokens=drafts,
        position_axes=3,
    )
    assert layout.raw_ring_capacity == ring
    assert page % ring == 0
    assert ring >= drafts + 4
    assert layout.raw_k_ring_shape == (ring, 128)
    assert layout.raw_logical_positions_shape == (ring,)
    assert layout.raw_rope_positions_shape == (ring, 3)
    assert layout.raw_page_nbytes == ring * (128 * 2 + 8 + 3 * 8) + 8
    assert not torch.cuda.is_initialized()


def test_seven_draft_caps_use_sixteen_raw_rows_at_every_verification_width():
    for width in (1, 2, 4, 8):
        caps = qsa.Caps(
            device="cuda",
            max_batch=4,
            max_raw_state_slots=4,
            max_q_rows=4 * width,
            max_seq_len=262144,
            num_main_cache_pages=4096,
            num_compressed_cache_pages=4096,
            main_page_size=64,
            compressed_page_size=16,
            q_heads=24,
            kv_heads=2,
            head_dim=256,
            index_heads=1,
            index_kv_heads=1,
            index_head_dim=128,
            index_rotary_dim=64,
            compress_ratio=4,
            budget=2048,
            max_speculative_tokens=7,
            position_axes=3,
            mrope_sections=(11, 11, 10),
        )
        assert caps.raw_ring_capacity == 16
        assert caps.cache_requirements.raw_page_nbytes == 4616
        assert not caps.cache_requirements.shared_compressed_raw_storage_legal


@pytest.mark.parametrize("page", [16, 32, 64, 96, 1520])
def test_horizon_larger_than_page_rejects_instead_of_truncating(page):
    with pytest.raises(ValueError, match="cover speculative history"):
        qsa.cache_requirements(
            main_page_size=page,
            kv_heads=2,
            head_dim=256,
            index_head_dim=128,
            max_speculative_tokens=page - 3,
        )


def test_ring_preserves_every_rollback_prefix_across_page_boundaries():
    ring = qsa.cache_requirements(
        main_page_size=64,
        kv_heads=2,
        head_dim=256,
        index_head_dim=128,
        max_speculative_tokens=7,
    ).raw_ring_capacity
    for start in range(61, 69):
        tags = {position % ring: position for position in range(start - 4, start + 8)}
        for accepted in range(1, 9):
            continuation = start + accepted
            group_start = (continuation // 4) * 4
            for position in range(group_start, continuation):
                assert tags[position % ring] == position
        assert start % ring == (start + 64) % ring
