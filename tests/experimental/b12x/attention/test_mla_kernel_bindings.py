from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


def test_b12x_mla_custom_ops_have_fake_dispatch() -> None:
    # Import modules for registration side effects.
    __import__("b12x.attention._shared.mla.kernel")
    __import__("b12x.attention._shared.mla.prefill")
    # prefill_mg registers the single-cache MG op + the new dual-cache MG op
    # (sparse_mla_sm120_prefill_mg_dual). prefill.py imports it lazily, so import it
    # explicitly here for the binding/fake-dispatch coverage.
    __import__("b12x.attention._shared.mla.prefill_mg")

    with FakeTensorMode():
        q_all = torch.empty((2, 2, 512), dtype=torch.bfloat16)
        cache = torch.empty((4, 1024), dtype=torch.uint8)
        indices = torch.empty((2, 4), dtype=torch.int32)
        lengths = torch.empty((2,), dtype=torch.int32)
        scalar_i32 = torch.empty((1,), dtype=torch.int32)
        tmp_output = torch.empty((2, 2, 4, 512), dtype=torch.bfloat16)
        tmp_lse = torch.empty((2, 2, 4), dtype=torch.float32)
        attn_sink = torch.empty((2,), dtype=torch.float32)
        output = torch.empty((2, 2, 512), dtype=torch.bfloat16)

        torch.ops.b12x.sparse_mla_sm120_split_decode_merge(
            tmp_output,
            tmp_lse,
            scalar_i32,
            output,
            attn_sink,
            tmp_output,
            tmp_lse,
            output,
            4,
            True,
        )

        mid_output = torch.empty((2, 2, 2, 512), dtype=torch.bfloat16)
        mid_lse = torch.empty((2, 2, 2), dtype=torch.float32)
        torch.ops.b12x.sparse_mla_sm120_decode_grid(
            q_all,
            cache,
            indices,
            mid_output,
            mid_lse,
            lengths,
            cache,
            indices,
            lengths,
            0.1,
            0,
            0,
            0,
            0,
            False,
            64,
            4,
            4,
            1,
            2,
            1,
            1024,
            64,
            1024,
            1,
            2,
            0,
            True,
            False,
        )

        prefill_lse = torch.empty((2, 2), dtype=torch.float32)
        # The single-cache decode-reuse prefill op was REMOVED (no fallback kernel
        # in prefill.py); the only prefill op is the MG dual-cache op below.
        # DUAL-CACHE MG prefill op (the new op DSV4 has_extra routes through).
        torch.ops.b12x.sparse_mla_sm120_prefill_mg_dual(
            q_all,        # q
            cache,        # kv_flat (MAIN)
            indices,      # topk_indices
            lengths,      # topk_length
            attn_sink,    # attn_sink_t
            output,       # output
            prefill_lse,  # lse_out
            cache,        # extra_kv_flat
            indices,      # extra_indices_t
            lengths,      # extra_len_t
            0.1,          # sm_scale
            1.0,          # latent_scale
            64,           # page_block_size
            4,            # topk
            2,            # num_tiles
            1024,         # stride_kv_block
            True,         # has_sink
            1,            # compute_mode (BF16)
            2,            # mg_n_hg
            0,            # model_type (DSV4)
            0,            # scale_format
            4,            # extra_topk
            1,            # num_main_tiles
            2,            # pbs_extra
            1024,         # stride_extra_kv_block
            True,         # row_xor
        )


def test_sm120_prefill_dual_odd_multiple_heads_splits_to_mg(monkeypatch) -> None:
    # DSV4 dual-cache heads=80 is a paired 64-head MG prefix plus one 16-head
    # single-group tail. This is Python dispatch coverage only; the focused CUDA
    # numerics live in the SM120 test suite.
    __import__("b12x.attention._shared.mla.prefill")
    import b12x.attention._shared.mla.prefill_mg as prefill_mg
    from b12x.attention._shared.mla.prefill import run_unified_prefill

    calls = []

    def fake_run_unified_prefill_mg(**kwargs):
        calls.append(kwargs)
        return kwargs["output"], kwargs["lse_out"]

    monkeypatch.setattr(prefill_mg, "run_unified_prefill_mg", fake_run_unified_prefill_mg)

    topk = 128
    q = torch.empty((2, 80, 512), dtype=torch.bfloat16)
    kv_cache = torch.empty((4, 1024), dtype=torch.uint8)
    topk_indices = torch.zeros((2, topk), dtype=torch.int32)
    extra_kv_cache = torch.empty((4, 1024), dtype=torch.uint8)
    extra_indices = torch.zeros((2, 128), dtype=torch.int32)

    output, lse = run_unified_prefill(
        q=q,
        kv_cache=kv_cache,
        topk_indices=topk_indices,
        sm_scale=0.1,
        page_block_size=64,
        stride_kv_block=37440,
        extra_kv_cache=extra_kv_cache,
        extra_indices=extra_indices,
        extra_page_block_size=2,
    )

    assert len(calls) == 2
    assert output.shape == (2, 80, 512)
    assert lse.shape == (2, 80)
    assert calls[0]["mg_n_hg"] == 2
    assert calls[0]["active_heads"] == 64
    assert calls[0]["head_offset"] == 0
    assert calls[1]["mg_n_hg"] == 1
    assert calls[1]["active_heads"] == 16
    assert calls[1]["head_offset"] == 64
    assert calls[0]["output"] is output
    assert calls[1]["output"] is output
    assert calls[0]["lse_out"] is lse
    assert calls[1]["lse_out"] is lse


def test_glm_prefill_partitions_120_heads_as_32_16_8(monkeypatch) -> None:
    __import__("b12x.attention._shared.mla.prefill")
    import b12x.attention._shared.mla.prefill_mg as prefill_mg
    from b12x.attention._shared.mla.prefill import run_unified_prefill
    from b12x.attention._shared.mla.traits import ComputeMode, ModelType, ScaleFormat

    calls = []

    def fake_run_unified_prefill_mg(**kwargs):
        calls.append(kwargs)
        return kwargs["output"], kwargs["lse_out"]

    monkeypatch.setattr(prefill_mg, "run_unified_prefill_mg", fake_run_unified_prefill_mg)

    topk = 2048
    q = torch.empty((2, 120, 576), dtype=torch.bfloat16)
    kv_cache = torch.empty((4, 656), dtype=torch.uint8)
    topk_indices = torch.zeros((2, topk), dtype=torch.int32)

    output, lse = run_unified_prefill(
        q=q,
        kv_cache=kv_cache,
        topk_indices=topk_indices,
        sm_scale=0.1,
        page_block_size=64,
        stride_kv_block=41984,
    )

    assert output.shape == (2, 120, 512)
    assert lse.shape == (2, 120)
    assert [call["mg_n_hg"] for call in calls] == [2, 1, 1]
    assert [call["active_heads"] for call in calls] == [96, 16, 8]
    assert [call["head_offset"] for call in calls] == [0, 96, 112]
    assert {call["compute_mode"] for call in calls} == {ComputeMode.FP8}
    assert {call["model_type"] for call in calls} == {ModelType.GLM_NSA}
    assert {call["scale_format"] for call in calls} == {ScaleFormat.ARBITRARY_FP32}


def test_dsv4_bf16_prefill_partitions_24_heads(monkeypatch) -> None:
    __import__("b12x.attention._shared.mla.prefill")
    import b12x.attention._shared.mla.prefill_mg as prefill_mg
    from b12x.attention._shared.mla.prefill import run_unified_prefill
    from b12x.attention._shared.mla.traits import ComputeMode, ModelType, ScaleFormat

    calls = []

    def fake_run_unified_prefill_mg(**kwargs):
        calls.append(kwargs)
        return kwargs["output"], kwargs["lse_out"]

    monkeypatch.setattr(prefill_mg, "run_unified_prefill_mg", fake_run_unified_prefill_mg)

    topk = 128
    q = torch.empty((2, 24, 512), dtype=torch.bfloat16)
    kv_cache = torch.empty((4, 1024), dtype=torch.uint8)
    topk_indices = torch.zeros((2, topk), dtype=torch.int32)

    run_unified_prefill(
        q=q,
        kv_cache=kv_cache,
        topk_indices=topk_indices,
        sm_scale=0.1,
        page_block_size=64,
        stride_kv_block=37440,
    )

    assert [call["mg_n_hg"] for call in calls] == [1, 1]
    assert [call["active_heads"] for call in calls] == [16, 8]
    assert [call["head_offset"] for call in calls] == [0, 16]
    assert {call["compute_mode"] for call in calls} == {ComputeMode.BF16}
    assert {call["model_type"] for call in calls} == {ModelType.DSV4}
    assert {call["scale_format"] for call in calls} == {ScaleFormat.UE8M0_BYTE}

@pytest.mark.parametrize(
    ("heads", "expected_active", "expected_offsets"),
    [(24, [16, 8], [0, 16]), (32, [32], [0])],
)
def test_dsv41_prefill_partitions_tp_shards(
    monkeypatch, heads, expected_active, expected_offsets
) -> None:
    import b12x.attention._shared.mla.prefill_mg as prefill_mg
    from b12x.attention._shared.mla.prefill import run_unified_prefill
    from b12x.attention._shared.mla.traits import ModelType, ScaleFormat

    calls = []

    def fake_run_unified_prefill_mg(**kwargs):
        calls.append(kwargs)
        return kwargs["output"], kwargs["lse_out"]

    monkeypatch.setattr(prefill_mg, "run_unified_prefill_mg", fake_run_unified_prefill_mg)

    q = torch.empty((2, heads, 512), dtype=torch.bfloat16)
    kv_cache = torch.empty((4, 432), dtype=torch.uint8)
    topk_indices = torch.zeros((2, 512), dtype=torch.int32)

    run_unified_prefill(
        q=q,
        kv_cache=kv_cache,
        topk_indices=topk_indices,
        sm_scale=0.1,
        page_block_size=64,
        stride_kv_block=64 * 432,
        model_type=ModelType.DSV41,
        scale_format=ScaleFormat.NVFP4_E4M3,
    )

    assert [call["mg_n_hg"] for call in calls] == [1] * len(calls)
    assert [call.get("active_heads", heads) for call in calls] == expected_active
    assert [call.get("head_offset", 0) for call in calls] == expected_offsets


def test_sm120_prefill_dual_partitions_40_heads_with_8_tail(monkeypatch) -> None:
    __import__("b12x.attention._shared.mla.prefill")
    import b12x.attention._shared.mla.prefill_mg as prefill_mg
    from b12x.attention._shared.mla.prefill import run_unified_prefill

    calls = []

    def fake_run_unified_prefill_mg(**kwargs):
        calls.append(kwargs)
        return kwargs["output"], kwargs["lse_out"]

    monkeypatch.setattr(prefill_mg, "run_unified_prefill_mg", fake_run_unified_prefill_mg)

    topk = 128
    q = torch.empty((2, 40, 512), dtype=torch.bfloat16)
    kv_cache = torch.empty((4, 1024), dtype=torch.uint8)
    topk_indices = torch.zeros((2, topk), dtype=torch.int32)
    extra_kv_cache = torch.empty((4, 1024), dtype=torch.uint8)
    extra_indices = torch.zeros((2, 128), dtype=torch.int32)

    run_unified_prefill(
        q=q,
        kv_cache=kv_cache,
        topk_indices=topk_indices,
        sm_scale=0.1,
        page_block_size=64,
        stride_kv_block=37440,
        extra_kv_cache=extra_kv_cache,
        extra_indices=extra_indices,
        extra_page_block_size=2,
    )

    assert [call["mg_n_hg"] for call in calls] == [2, 1]
    assert [call["active_heads"] for call in calls] == [32, 8]
    assert [call["head_offset"] for call in calls] == [0, 32]
    assert calls[0]["extra_kv_cache"] is extra_kv_cache
    assert calls[1]["extra_kv_cache"] is extra_kv_cache


def test_glm_tp8_prefill_routes_to_single_group_mg(monkeypatch) -> None:
    __import__("b12x.attention._shared.mla.prefill")
    import b12x.attention._shared.mla.prefill_mg as prefill_mg
    from b12x.attention._shared.mla.prefill import run_unified_prefill
    from b12x.attention._shared.mla.traits import ComputeMode, ModelType, ScaleFormat

    calls = []

    def fake_run_unified_prefill_mg(**kwargs):
        calls.append(kwargs)
        return kwargs["output"], kwargs["lse_out"]

    monkeypatch.setattr(prefill_mg, "run_unified_prefill_mg", fake_run_unified_prefill_mg)

    topk = 2048
    q = torch.empty((2, 8, 576), dtype=torch.bfloat16)
    kv_cache = torch.empty((4, 656), dtype=torch.uint8)
    topk_indices = torch.zeros((2, topk), dtype=torch.int32)

    output, lse = run_unified_prefill(
        q=q,
        kv_cache=kv_cache,
        topk_indices=topk_indices,
        sm_scale=0.1,
        page_block_size=64,
        stride_kv_block=41984,
    )

    assert len(calls) == 1
    assert output.shape == (2, 8, 512)
    assert lse.shape == (2, 8)
    assert calls[0]["mg_n_hg"] == 1
    assert calls[0]["compute_mode"] == ComputeMode.FP8
    assert calls[0]["model_type"] == ModelType.GLM_NSA
    assert calls[0]["scale_format"] == ScaleFormat.ARBITRARY_FP32


def test_prefill_mg_heads8_uses_flat_valid_hpb_launcher(monkeypatch) -> None:
    __import__("b12x.attention._shared.mla.prefill_mg")
    import b12x.attention._shared.mla.prefill_mg as prefill_mg
    from b12x.attention._shared.mla.prefill_mg import run_unified_prefill_mg
    from b12x.attention._shared.mla.traits import ComputeMode, ModelType, ScaleFormat

    calls = []

    def fake_flat_launch(*args, **kwargs):
        del args
        calls.append(kwargs)

    monkeypatch.setattr(prefill_mg, "_sparse_mla_prefill_mg_flat_launch", fake_flat_launch)

    topk = 2048
    q = torch.empty((2, 8, 576), dtype=torch.bfloat16)
    kv_cache = torch.empty((4, 656), dtype=torch.uint8)
    topk_indices = torch.zeros((2, topk), dtype=torch.int32)

    output, lse = run_unified_prefill_mg(
        q=q,
        kv_cache=kv_cache,
        topk_indices=topk_indices,
        sm_scale=0.1,
        page_block_size=64,
        stride_kv_block=41984,
        compute_mode=ComputeMode.FP8,
        mg_n_hg=1,
        model_type=ModelType.GLM_NSA,
        scale_format=ScaleFormat.ARBITRARY_FP32,
    )

    assert len(calls) == 1
    assert output.shape == (2, 8, 512)
    assert lse.shape == (2, 8)
    assert calls[0]["active_heads"] == 8
    assert calls[0]["head_offset"] == 0


def test_sm120_prefill_dual_non_eligible_raises() -> None:
    # DSV4 dual-cache prefill is MG-only (topk==128, heads divisible by 8);
    # everything else RAISEs (the decode-reuse has_extra fallback was removed).
    # topk != 128 (here topk == 64) is non-eligible -> ValueError, raised in the
    # Python dispatch BEFORE any kernel launch (so this runs on CPU tensors).
    __import__("b12x.attention._shared.mla.prefill")
    from b12x.attention._shared.mla.prefill import run_unified_prefill

    topk = 64  # != 128 -> non-eligible dual
    q = torch.empty((2, 32, 512), dtype=torch.bfloat16)
    kv_cache = torch.empty((4, 1024), dtype=torch.uint8)
    topk_indices = torch.zeros((2, topk), dtype=torch.int32)
    extra_kv_cache = torch.empty((4, 1024), dtype=torch.uint8)
    extra_indices = torch.zeros((2, 64), dtype=torch.int32)

    with pytest.raises(ValueError, match="requires MG dispatch"):
        run_unified_prefill(
            q=q,
            kv_cache=kv_cache,
            topk_indices=topk_indices,
            sm_scale=0.1,
            page_block_size=64,
            stride_kv_block=37440,
            extra_kv_cache=extra_kv_cache,
            extra_indices=extra_indices,
            extra_page_block_size=2,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("shape,stride", [
    ((3, 8), (8, 1)), ((1, 8), (1, 1)), ((3, 8), (24, 3)), ((0,), (1,)),
])
def test_prefill_compile_descriptors_never_export_fake_storage(
    monkeypatch, device, dynamic, shape, stride,
):
    """Compilation preserves native layouts without exporting FakeTensor storage."""
    import cutlass
    from b12x._lib.compile_plan import compile_only_launches
    from b12x.attention._shared.mla import prefill_mg

    real = torch.empty_strided(shape, stride, dtype=torch.float32)
    expected = prefill_mg._to_cute(real, cutlass.Float32, align=4, dynamic_layout=dynamic)

    from_dlpack = prefill_mg.from_dlpack

    def reject_fake_dlpack(tensor, *args, **kwargs):
        assert not hasattr(tensor, "fake_mode"), "FakeTensor must not export storage"
        assert tensor.numel() == 0
        return from_dlpack(tensor, *args, **kwargs)

    monkeypatch.setattr(prefill_mg, "from_dlpack", reject_fake_dlpack)
    with FakeTensorMode(), compile_only_launches():
        fake = torch.empty_strided(shape, stride, dtype=torch.float32, device=device)
        actual = prefill_mg._to_cute(fake, cutlass.Float32, align=4, dynamic_layout=dynamic)
    from cutlass._mlir import ir

    assert actual.element_type == expected.element_type
    assert actual.dynamic_shapes_mask == expected.dynamic_shapes_mask
    assert actual.dynamic_strides_mask == expected.dynamic_strides_mask
    with ir.Context(), ir.Location.unknown():
        # CPU DLPack uses generic memory; compile descriptors describe GPU memory.
        expected_type = str(expected.mlir_type).replace(", generic,", ", gmem,")
        assert str(actual.mlir_type).replace(", generic,", ", gmem,") == expected_type
