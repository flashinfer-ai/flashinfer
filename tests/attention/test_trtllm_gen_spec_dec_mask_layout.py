import torch
import pytest

import flashinfer.decode
from flashinfer.decode import (
    _pack_trtllm_gen_spec_dec_mask,
    _select_trtllm_gen_spec_dec_tree_kernel,
    _should_force_trtllm_gen_spec_dec_tree_keeps,
)


def test_untrimmed_swa_spec_dec_tree_capability_flag():
    # Cross-repo contract: sglang's trtllm_mha backend reads this attribute
    # via getattr to gate its untrimmed SWA tree-verify path.
    assert (
        getattr(
            flashinfer.decode,
            "trtllm_gen_supports_untrimmed_swa_spec_dec_tree",
            False,
        )
        is True
    )


def test_spec_dec_tree_kernel_selection_matches_trtllm_gen_thresholds():
    cases = [
        (4, 2, "swaps", 8),
        (4, 4, "swaps", 16),
        (8, 8, "swaps", 32),
        (16, 4, "swaps", 32),
        (16, 5, "keeps", 128),
    ]

    for num_heads_q_per_kv, q_len, kernel_layout, tile_size_q in cases:
        layout = _select_trtllm_gen_spec_dec_tree_kernel(
            torch.bfloat16,
            torch.bfloat16,
            num_heads_q_per_kv,
            q_len,
        )
        assert layout.kernel_layout == kernel_layout
        assert layout.tile_size_q == tile_size_q


def test_spec_dec_tree_mask_words_per_sequence_for_keeps_and_swaps():
    swaps = _select_trtllm_gen_spec_dec_tree_kernel(
        torch.bfloat16,
        torch.bfloat16,
        num_heads_q_per_kv=8,
        q_len=8,
    )
    keeps = _select_trtllm_gen_spec_dec_tree_kernel(
        torch.bfloat16,
        torch.bfloat16,
        num_heads_q_per_kv=16,
        q_len=5,
    )

    assert (
        swaps.words_per_sequence(q_len=8, max_seq_len=257, window_left=128)
        == 8 * 2 * 2 * 128
    )
    assert (
        keeps.words_per_sequence(q_len=5, max_seq_len=257, window_left=128)
        == 1 * 2 * 2 * 512
    )


def test_spec_dec_tree_native_window_uses_tail_only_mask_stride():
    layout = _select_trtllm_gen_spec_dec_tree_kernel(
        torch.bfloat16,
        torch.bfloat16,
        num_heads_q_per_kv=8,
        q_len=8,
    )

    fallback_words = layout.words_per_sequence(
        q_len=8, max_seq_len=4096, window_left=128
    )
    native_words = layout.words_per_sequence(q_len=8, max_seq_len=4096, window_left=-1)

    assert fallback_words == 8 * 16 * 2 * 128
    assert native_words == 8 * 2 * 2 * 128


def test_spec_dec_tree_swaps_default_forces_keeps_until_perf_boundary():
    layout = _select_trtllm_gen_spec_dec_tree_kernel(
        torch.bfloat16,
        torch.bfloat16,
        num_heads_q_per_kv=5,
        q_len=4,
    )

    assert layout.kernel_layout == "swaps"
    assert _should_force_trtllm_gen_spec_dec_tree_keeps(
        layout,
        torch.tensor([1280], dtype=torch.int32),
        q_len=4,
        window_left=-1,
    )


def test_spec_dec_tree_swaps_opt_in_straddling_tail_forces_keeps_fallback(monkeypatch):
    monkeypatch.setenv("FLASHINFER_TRTLLM_GEN_SPEC_DEC_TREE_ENABLE_SWAPS", "1")
    layout = _select_trtllm_gen_spec_dec_tree_kernel(
        torch.bfloat16,
        torch.bfloat16,
        num_heads_q_per_kv=5,
        q_len=4,
    )

    assert layout.kernel_layout == "swaps"
    assert _should_force_trtllm_gen_spec_dec_tree_keeps(
        layout,
        torch.tensor([1281], dtype=torch.int32),
        q_len=4,
        window_left=-1,
    )
    assert not _should_force_trtllm_gen_spec_dec_tree_keeps(
        layout,
        torch.tensor([1280], dtype=torch.int32),
        q_len=4,
        window_left=-1,
    )
    assert _should_force_trtllm_gen_spec_dec_tree_keeps(
        layout,
        torch.tensor([1280], dtype=torch.int32),
        q_len=4,
        window_left=333,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_dec_tree_capture_forces_keeps_without_host_sync(monkeypatch):
    layout = _select_trtllm_gen_spec_dec_tree_kernel(
        torch.bfloat16,
        torch.bfloat16,
        num_heads_q_per_kv=5,
        q_len=4,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    def fail_on_host_sync(*args, **kwargs):
        raise AssertionError("capture path must not inspect device sequence lengths")

    monkeypatch.setattr(torch, "any", fail_on_host_sync)

    assert _should_force_trtllm_gen_spec_dec_tree_keeps(
        layout,
        torch.tensor([1280], dtype=torch.int32, device="cuda"),
        q_len=4,
        window_left=-1,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pack_spec_dec_tree_mask_swaps_layout_bits():
    tree_mask = torch.tensor([[[True, False], [True, True]]], device="cuda")
    seq_lens = torch.tensor([6], dtype=torch.int32, device="cuda")

    packed, offsets, first_sparse_offsets = _pack_trtllm_gen_spec_dec_mask(
        tree_mask,
        seq_lens,
        q_len_per_req=2,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        num_heads_q_per_kv=4,
        batch_size=1,
        window_left=-1,
    )
    torch.cuda.synchronize()
    words = packed.cpu()[0]

    assert offsets.cpu().tolist() == [0]
    assert first_sparse_offsets.cpu().tolist() == [0]

    def swaps_word_bit(
        token_idx_q: int, token_idx_kv: int, head_idx_q: int
    ) -> tuple[int, int]:
        tile_size_q = 32
        tile_size_kv = 128
        num_insts_kv = 2
        tile_size_kv_per_cta = tile_size_kv * num_insts_kv
        tile_idx_kv = token_idx_kv // tile_size_kv_per_cta
        inst_idx_kv = (token_idx_kv % tile_size_kv_per_cta) // tile_size_kv
        token_idx_in_tile_kv = token_idx_kv % tile_size_kv
        tile_idx_q = token_idx_q
        token_idx_in_tile_q = head_idx_q
        tile_offset = tile_idx_q * 1 + tile_idx_kv
        inst_offset = tile_offset * num_insts_kv + inst_idx_kv
        mask_offset = inst_offset * tile_size_q * tile_size_kv
        thread_idx_q = (token_idx_in_tile_q % 8) // 2
        thread_idx_kv = (token_idx_in_tile_kv % 8) + (token_idx_in_tile_kv // 32) * 8
        token_idx_in_warp_tile_kv = token_idx_in_tile_kv % 32
        elt_idx_in_thread = (
            (token_idx_in_tile_q % 2)
            + ((token_idx_in_warp_tile_kv // 8) % 2) * 2
            + (token_idx_in_tile_q // 8) * 4
            + (token_idx_in_warp_tile_kv // 16) * 16
        )
        mask_offset += (thread_idx_kv * 4 + thread_idx_q) * 32 + elt_idx_in_thread
        return mask_offset // 32, mask_offset % 32

    visible_pairs = [(0, kv, head) for kv in [0, 4] for head in range(4)]
    visible_pairs += [(1, kv, head) for kv in [0, 4, 5] for head in range(4)]
    for token_idx_q, token_idx_kv, head_idx_q in visible_pairs:
        word_idx, bit_idx = swaps_word_bit(token_idx_q, token_idx_kv, head_idx_q)
        assert words[word_idx].item() & (1 << bit_idx)

    for head_idx_q in range(4):
        word_idx, bit_idx = swaps_word_bit(0, 5, head_idx_q)
        assert not (words[word_idx].item() & (1 << bit_idx))
