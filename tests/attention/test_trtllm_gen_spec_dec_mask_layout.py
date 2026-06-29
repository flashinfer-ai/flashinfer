import flashinfer.decode
import torch


def test_untrimmed_swa_spec_dec_tree_capability_flag():
    assert (
        getattr(
            flashinfer.decode,
            "trtllm_gen_supports_untrimmed_swa_spec_dec_tree",
            False,
        )
        is True
    )


def test_spec_dec_tree_kernel_selection_matches_trtllm_gen_thresholds():
    select_layout = getattr(
        flashinfer.decode, "_select_trtllm_gen_spec_dec_tree_kernel", None
    )
    assert callable(select_layout)

    cases = [
        (4, 2, "swaps", 8),
        (4, 4, "swaps", 16),
        (8, 8, "swaps", 32),
        (16, 4, "swaps", 32),
        (16, 5, "keeps", 128),
    ]
    for num_heads_q_per_kv, q_len, kernel_layout, tile_size_q in cases:
        layout = select_layout(
            torch.bfloat16,
            torch.bfloat16,
            num_heads_q_per_kv,
            q_len,
        )
        assert layout.kernel_layout == kernel_layout
        assert layout.tile_size_q == tile_size_q


def test_native_window_uses_tail_only_mask_stride():
    select_layout = getattr(
        flashinfer.decode, "_select_trtllm_gen_spec_dec_tree_kernel", None
    )
    assert callable(select_layout)

    layout = select_layout(
        torch.bfloat16,
        torch.bfloat16,
        num_heads_q_per_kv=8,
        q_len=8,
    )
    fallback_words = layout.words_per_sequence(
        q_len=8, max_seq_len=4096, window_left=128
    )
    native_words = layout.words_per_sequence(
        q_len=8, max_seq_len=4096, window_left=-1
    )

    assert fallback_words == 8 * 16 * 2 * 128
    assert native_words == 8 * 2 * 2 * 128
    assert native_words < fallback_words


def test_spec_dec_tree_swaps_straddling_tail_forces_keeps():
    select_layout = getattr(
        flashinfer.decode, "_select_trtllm_gen_spec_dec_tree_kernel", None
    )
    should_force_keeps = getattr(
        flashinfer.decode, "_should_force_trtllm_gen_spec_dec_tree_keeps", None
    )
    assert callable(select_layout)
    assert callable(should_force_keeps)

    layout = select_layout(
        torch.bfloat16,
        torch.bfloat16,
        num_heads_q_per_kv=5,
        q_len=4,
    )
    assert layout.kernel_layout == "swaps"
    assert should_force_keeps(
        layout,
        torch.tensor([1281], dtype=torch.int32),
        q_len=4,
        window_left=-1,
    )
    assert not should_force_keeps(
        layout,
        torch.tensor([1280], dtype=torch.int32),
        q_len=4,
        window_left=-1,
    )
