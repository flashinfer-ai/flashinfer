# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side contracts for DeepSeek V4 HCA decode; no GPU execution."""

import pytest
import torch

import flashinfer.mla._core as mla_core
from flashinfer.mla import (
    batch_decode_sparse_mla_dsv4,
    trtllm_batch_decode_sparse_mla_dsv4,
)
from tests.test_helpers.hca_dsv4 import (
    _cpu_hca_inputs,
    _cpu_compressed_page_aligned_sparse_hca_inputs,
    _cpu_compressed_page_aligned_conversion_inputs,
)


@pytest.mark.cpu_only
def test_dsv4_backend_resolution_is_explicit(monkeypatch):
    monkeypatch.setattr(mla_core, "get_compute_capability", lambda _device: (10, 0))
    assert (
        mla_core._resolve_dsv4_sparse_mla_backend(torch.device("cpu"), "auto")
        == "trtllm-gen"
    )
    sm100_default = mla_core._resolve_dsv4_sparse_mla_backend(
        torch.device("cpu"), "auto"
    )
    assert (
        mla_core._resolve_dsv4_sparse_mla_backend(torch.device("cpu"), "cute-dsl")
        == "cute-dsl"
    )

    monkeypatch.setattr(mla_core, "get_compute_capability", lambda _device: (12, 0))
    assert (
        mla_core._resolve_dsv4_sparse_mla_backend(torch.device("cpu"), "auto")
        == "sparse"
    )
    with pytest.raises(ValueError, match="requires SM100/SM103/SM107, got SM120"):
        mla_core._resolve_dsv4_sparse_mla_backend(torch.device("cpu"), "cute-dsl")

    monkeypatch.setattr(mla_core, "get_compute_capability", lambda _device: (10, 1))
    with pytest.raises(ValueError, match="supports SM100/SM103"):
        mla_core._resolve_dsv4_sparse_mla_backend(torch.device("cpu"), "auto")
    with pytest.raises(ValueError, match="requires SM100/SM103/SM107, got SM101"):
        mla_core._resolve_dsv4_sparse_mla_backend(torch.device("cpu"), "cute-dsl")

    monkeypatch.setattr(mla_core, "get_compute_capability", lambda _device: (10, 7))
    sm107_default = mla_core._resolve_dsv4_sparse_mla_backend(
        torch.device("cpu"), "auto"
    )
    assert sm107_default == sm100_default
    assert (
        mla_core._resolve_dsv4_sparse_mla_backend(torch.device("cpu"), sm100_default)
        == sm100_default
    )
    assert (
        mla_core._resolve_dsv4_sparse_mla_backend(torch.device("cpu"), "cute-dsl")
        == "cute-dsl"
    )
    with pytest.raises(
        ValueError, match="requires SM100/SM103 or SM120/SM121, got SM107"
    ):
        mla_core._resolve_dsv4_sparse_mla_backend(torch.device("cpu"), "cake")

    for cc in ((10, 0), (10, 3), (12, 0), (12, 1)):
        monkeypatch.setattr(mla_core, "get_compute_capability", lambda _device: cc)
        assert (
            mla_core._resolve_dsv4_sparse_mla_backend(torch.device("cpu"), "cake")
            == "cake"
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("backend", "compute_capability"),
    (("cute-dsl", (10, 0)), ("sparse", (12, 0))),
)
@pytest.mark.parametrize(
    "remap_arg",
    ("remapped_sparse_indices_buffer", "sparse_indices_are_storage_offsets"),
)
def test_non_trtllm_dsv4_backends_reject_index_remapping_args(
    monkeypatch, backend, compute_capability, remap_arg
):
    args = _cpu_hca_inputs()
    args["backend"] = backend
    args[remap_arg] = (
        torch.empty((1, 128), dtype=torch.int32)
        if remap_arg == "remapped_sparse_indices_buffer"
        else False
    )
    monkeypatch.setattr(
        mla_core, "get_compute_capability", lambda _device: compute_capability
    )

    with pytest.raises(ValueError, match=f"backend='{backend}' does not accept"):
        trtllm_batch_decode_sparse_mla_dsv4(**args)


@pytest.mark.cpu_only
def test_dsv4_backend_neutral_alias_is_backward_compatible():
    assert batch_decode_sparse_mla_dsv4 is trtllm_batch_decode_sparse_mla_dsv4
    assert (
        trtllm_batch_decode_sparse_mla_dsv4.__name__
        == "trtllm_batch_decode_sparse_mla_dsv4"
    )


@pytest.mark.cpu_only
def test_hca_gather4_cutedsl_ir_support_is_available():
    pytest.importorskip("cutlass")
    from flashinfer.cute_dsl.attention.dsa.gather4_utils import (
        _ensure_gather4_ir_support,
    )

    _ensure_gather4_ir_support()


@pytest.mark.cpu_only
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
def test_dsv4_hca_public_api_forwards_gather_indices(monkeypatch, kv_layout):
    pytest.importorskip("cutlass")
    from flashinfer.cute_dsl.attention.wrappers import batch_hca

    args = _cpu_hca_inputs()
    args["kv_layout"] = kv_layout
    args["hca_use_persistent"] = True
    if kv_layout == "NHD":
        args["swa_kv_cache"] = args["swa_kv_cache"].unsqueeze(2)
        args["compressed_kv_cache"] = args["compressed_kv_cache"].unsqueeze(2)
    captured = {}

    def fake_hca_decode(**kwargs):
        captured.update(kwargs)
        return kwargs["out"]

    monkeypatch.setattr(mla_core, "get_compute_capability", lambda _device: (10, 0))
    monkeypatch.setattr(batch_hca, "cute_dsl_hca_decode", fake_hca_decode)

    result = trtllm_batch_decode_sparse_mla_dsv4(**args)
    assert result is args["out"]
    assert captured["window_indices"] is args["hca_swa_indices"]
    assert captured["compressed_block_tables"] is args["hca_compressed_block_tables"]
    assert captured["hca_seq_lens"] is args["hca_seq_lens"]
    assert captured["window_valid_lens"] is args["swa_topk_lens"]
    assert captured["is_persistent"] is True
    assert captured["window_kv_cache"].shape == (16 * 32, 512)
    assert captured["compressed_kv_cache"].shape == (5, 128, 512)
    assert captured["window_kv_cache"].data_ptr() == args["swa_kv_cache"].data_ptr()
    assert (
        captured["compressed_kv_cache"].data_ptr()
        == args["compressed_kv_cache"].data_ptr()
    )


@pytest.mark.cpu_only
def test_dsv4_hca_generates_metadata_from_compressed_page_aligned_sparse_indices(
    monkeypatch,
):
    pytest.importorskip("cutlass")
    from flashinfer.cute_dsl.attention.wrappers import batch_hca

    args, expected = _cpu_compressed_page_aligned_sparse_hca_inputs()
    captured = {}

    def fake_hca_decode(**kwargs):
        captured.update(kwargs)
        return kwargs["out"]

    monkeypatch.setattr(mla_core, "get_compute_capability", lambda _device: (10, 0))
    monkeypatch.setattr(batch_hca, "cute_dsl_hca_decode", fake_hca_decode)

    result = trtllm_batch_decode_sparse_mla_dsv4(**args)
    assert result is args["out"]
    for name, value in expected.items():
        torch.testing.assert_close(captured[name], value)
    assert captured["is_causal"] is True


@pytest.mark.cpu_only
def test_dsv4_hca_rejects_out_of_range_swa_index(monkeypatch):
    args, _ = _cpu_compressed_page_aligned_sparse_hca_inputs()
    args["sparse_indices"][1, 0] = (
        args["swa_kv_cache"].shape[0] * args["swa_kv_cache"].shape[1]
    )
    monkeypatch.setattr(mla_core, "get_compute_capability", lambda _device: (10, 0))
    with pytest.raises(ValueError, match="active SWA sparse_indices"):
        trtllm_batch_decode_sparse_mla_dsv4(**args)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("window_page_size", "compressed_page_size"),
    [(32, 128), (16, 32), (32, 16)],
)
def test_dsv4_hca_compressed_page_aligned_conversion_ignores_inactive_padding(
    window_page_size, compressed_page_size
):
    minimal = _cpu_compressed_page_aligned_conversion_inputs(
        window_page_size, compressed_page_size
    )
    padded = _cpu_compressed_page_aligned_conversion_inputs(
        window_page_size, compressed_page_size, extra_padding=512
    )

    def convert(inputs):
        return mla_core.convert_compressed_page_aligned_sparse_indices_to_hca_metadata(
            inputs["sparse_indices"],
            inputs["sparse_topk_lens"],
            inputs["seq_lens"],
            inputs["swa_kv_cache"],
            inputs["compressed_kv_cache"],
            q_len=inputs["q_len"],
        )

    minimal_metadata = convert(minimal)
    padded_metadata = convert(padded)
    for field in (
        "hca_swa_indices",
        "hca_compressed_block_tables",
        "hca_seq_lens",
        "swa_topk_lens",
    ):
        minimal_value = getattr(minimal_metadata, field)
        padded_value = getattr(padded_metadata, field)
        torch.testing.assert_close(minimal_value, padded_value)

    torch.testing.assert_close(
        minimal_metadata.hca_swa_indices, minimal["expected_window"]
    )
    torch.testing.assert_close(
        minimal_metadata.hca_compressed_block_tables,
        minimal["expected_compressed"],
    )
    torch.testing.assert_close(
        minimal_metadata.swa_topk_lens, minimal["expected_window_lens"]
    )
    assert minimal_metadata.hca_compressed_block_tables.shape[1] == (
        256 // compressed_page_size
    )


@pytest.mark.cpu_only
def test_dsv4_hca_compressed_page_aligned_conversion_rejects_in_page_reordering():
    inputs = _cpu_compressed_page_aligned_conversion_inputs(32, 16)
    inputs["sparse_indices"][3, 129] += 16
    with pytest.raises(ValueError, match="canonical compressed-page expansion"):
        mla_core.convert_compressed_page_aligned_sparse_indices_to_hca_metadata(
            inputs["sparse_indices"],
            inputs["sparse_topk_lens"],
            inputs["seq_lens"],
            inputs["swa_kv_cache"],
            inputs["compressed_kv_cache"],
            q_len=inputs["q_len"],
        )


@pytest.mark.cpu_only
def test_dsv4_hca_compressed_page_aligned_conversion_checks_across_chunks(monkeypatch):
    monkeypatch.setattr(mla_core, "_DSV4_HCA_CANONICAL_CHECK_MAX_ELEMENTS", 8)
    inputs = _cpu_compressed_page_aligned_conversion_inputs(32, 16)

    mla_core.convert_compressed_page_aligned_sparse_indices_to_hca_metadata(
        inputs["sparse_indices"],
        inputs["sparse_topk_lens"],
        inputs["seq_lens"],
        inputs["swa_kv_cache"],
        inputs["compressed_kv_cache"],
        q_len=inputs["q_len"],
    )
    inputs["sparse_indices"][3, 128 + 17] += 16
    with pytest.raises(ValueError, match="canonical compressed-page expansion"):
        mla_core.convert_compressed_page_aligned_sparse_indices_to_hca_metadata(
            inputs["sparse_indices"],
            inputs["sparse_topk_lens"],
            inputs["seq_lens"],
            inputs["swa_kv_cache"],
            inputs["compressed_kv_cache"],
            q_len=inputs["q_len"],
        )


@pytest.mark.cpu_only
def test_dsv4_hca_compressed_page_aligned_conversion_fills_fully_inactive_pages():
    window_page_size = 32
    offsets = torch.arange(window_page_size, dtype=torch.int32)
    sparse_indices = (
        torch.arange(4, dtype=torch.int32).unsqueeze(-1) * window_page_size + offsets
    ).reshape(1, 128)
    sparse_indices[:, 1:] = torch.iinfo(torch.int32).max
    metadata = mla_core.convert_compressed_page_aligned_sparse_indices_to_hca_metadata(
        sparse_indices,
        torch.tensor([128], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.empty((4, window_page_size, 512), dtype=torch.float8_e4m3fn),
        torch.empty((1, 128, 512), dtype=torch.float8_e4m3fn),
        q_len=1,
    )
    torch.testing.assert_close(
        metadata.hca_swa_indices,
        torch.zeros((1, 128), dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.hca_compressed_block_tables,
        torch.tensor([[0]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.hca_seq_lens, torch.tensor([128], dtype=torch.int32)
    )
    torch.testing.assert_close(
        metadata.swa_topk_lens, torch.tensor([1], dtype=torch.int32)
    )


@pytest.mark.cpu_only
def test_dsv4_hca_compressed_page_aligned_conversion_rejects_out_of_range_page():
    inputs = _cpu_compressed_page_aligned_conversion_inputs(32, 16)
    inputs["sparse_indices"][3, 128] = 32 * 16
    with pytest.raises(ValueError, match="valid physical page IDs"):
        mla_core.convert_compressed_page_aligned_sparse_indices_to_hca_metadata(
            inputs["sparse_indices"],
            inputs["sparse_topk_lens"],
            inputs["seq_lens"],
            inputs["swa_kv_cache"],
            inputs["compressed_kv_cache"],
            q_len=inputs["q_len"],
        )


@pytest.mark.cpu_only
def test_dsv4_hca_compressed_page_aligned_tag_rejects_query_batch_mismatch(monkeypatch):
    args, _ = _cpu_compressed_page_aligned_sparse_hca_inputs()
    args["seq_lens"] = torch.tensor([16512], dtype=torch.int32)
    monkeypatch.setattr(mla_core, "get_compute_capability", lambda _device: (10, 0))
    with pytest.raises(ValueError, match="seq_lens with 2 entries"):
        trtllm_batch_decode_sparse_mla_dsv4(**args)


@pytest.mark.cpu_only
def test_dsv4_hca_rejects_legacy_token_indices(monkeypatch):
    args = _cpu_hca_inputs()
    args["sparse_indices"] = torch.zeros((4, 256), dtype=torch.int32)
    monkeypatch.setattr(mla_core, "get_compute_capability", lambda _device: (10, 0))
    with pytest.raises(ValueError, match="sparse_indices must be None"):
        trtllm_batch_decode_sparse_mla_dsv4(**args)


@pytest.mark.cpu_only
def test_dsv4_hca_rejects_ambiguous_page_aligned_format(monkeypatch):
    args = _cpu_hca_inputs()
    args["hca_sparse_indices_format"] = "page-aligned"
    monkeypatch.setattr(mla_core, "get_compute_capability", lambda _device: (10, 0))
    with pytest.raises(ValueError, match="compressed-page-aligned"):
        trtllm_batch_decode_sparse_mla_dsv4(**args)


@pytest.mark.cpu_only
def test_dsv4_hca_rejects_backend_specific_metadata(monkeypatch):
    args = _cpu_hca_inputs()
    args["hca_is_causal"] = False
    monkeypatch.setattr(mla_core, "get_compute_capability", lambda _device: (10, 0))
    with pytest.raises(ValueError, match="supports causal HCA only"):
        trtllm_batch_decode_sparse_mla_dsv4(**args)

    args = _cpu_hca_inputs()
    args["seq_lens"] = torch.tensor([129, 257], dtype=torch.int32)
    with pytest.raises(ValueError, match="uses hca_seq_lens"):
        trtllm_batch_decode_sparse_mla_dsv4(**args)

    args = _cpu_hca_inputs()
    args["backend"] = "trtllm-gen"
    args["sparse_indices"] = torch.zeros((4, 256), dtype=torch.int32)
    with pytest.raises(ValueError, match="does not accept hca_swa_indices"):
        trtllm_batch_decode_sparse_mla_dsv4(**args)

    args = _cpu_hca_inputs()
    args["backend"] = "trtllm-gen"
    args["sparse_indices"] = torch.zeros((4, 256), dtype=torch.int32)
    args["hca_swa_indices"] = None
    args["hca_compressed_block_tables"] = None
    args["hca_seq_lens"] = None
    args["hca_is_causal"] = False
    with pytest.raises(ValueError, match="does not accept hca_is_causal"):
        trtllm_batch_decode_sparse_mla_dsv4(**args)


@pytest.mark.cpu_only
def test_hca_compile_rejects_arch_the_installed_dsl_cannot_target(monkeypatch):
    pytest.importorskip("cutlass")
    from flashinfer.cute_dsl.attention.wrappers import batch_hca

    monkeypatch.setattr(batch_hca, "is_cute_dsl_arch_supported", lambda *_: False)
    with pytest.raises(NotImplementedError, match="does not support sm_107"):
        batch_hca._compile_hca_kernel(128, 1, True, False, True, (10, 7))


@pytest.mark.cpu_only
def test_hca_rejects_nonpersistent_grid_overflow():
    pytest.importorskip("cutlass")
    from flashinfer.cute_dsl.attention.wrappers.batch_hca import (
        _check_nonpersistent_grid,
    )

    _check_nonpersistent_grid(batch_size=1, q_len=8192)
    with pytest.raises(ValueError, match=r"batch_size \* q_len <= 65535"):
        _check_nonpersistent_grid(batch_size=8, q_len=8192)


@pytest.mark.cpu_only
def test_hca_opt_in_metadata_value_validation(monkeypatch):
    pytest.importorskip("cutlass")
    from flashinfer.cute_dsl.attention.wrappers.batch_hca import (
        _validate_hca_values,
    )

    args = _cpu_hca_inputs()
    validation_args = {
        "window_kv_cache": args["swa_kv_cache"].reshape(-1, 512),
        "compressed_kv_cache": args["compressed_kv_cache"],
        "window_indices": args["hca_swa_indices"],
        "compressed_block_tables": args["hca_compressed_block_tables"],
        "hca_seq_lens": args["hca_seq_lens"],
        "sparse_topk_lens": args["sparse_topk_lens"],
        "window_valid_lens": args["swa_topk_lens"],
        "q_len": 2,
        "is_causal": True,
    }
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")

    # Every gather coordinate must be legal, while only the tile-rounded
    # compressed footprint is validated; later page columns may use -1.
    validation_args["compressed_block_tables"] = torch.tensor(
        [[0, -1], [0, -1], [0, 1], [0, 1]], dtype=torch.int32
    )
    _validate_hca_values(**validation_args)

    validation_args["hca_seq_lens"] = torch.tensor([129, 385], dtype=torch.int32)
    with pytest.raises(ValueError, match="block-table capacity 384"):
        _validate_hca_values(**validation_args)

    validation_args["hca_seq_lens"] = args["hca_seq_lens"]
    validation_args["compressed_block_tables"][2, 1] = -1
    with pytest.raises(ValueError, match="compressed_block_tables active values"):
        _validate_hca_values(**validation_args)

    validation_args["compressed_block_tables"][2, 1] = 1
    validation_args["window_valid_lens"][0] = 0
    validation_args["window_indices"][0, 3] = -1
    with pytest.raises(ValueError, match="window_indices values"):
        _validate_hca_values(**validation_args)


@pytest.mark.cpu_only
def test_hca_static_contract_accepts_fp8_to_bf16():
    pytest.importorskip("cutlass")
    import cutlass

    from flashinfer.cute_dsl.attention.dsa.hca_fp8 import (
        MAX_SPLITS,
        BlackwellHeavilyCompressedAttentionForwardFP8,
    )

    can_implement = BlackwellHeavilyCompressedAttentionForwardFP8.can_implement
    config = {
        "B": 2,
        "S": 2,
        "K": 384,
        "H": 128,
        "L": 512,
        "in_dtype": cutlass.Float8E4M3FN,
        "out_dtype": cutlass.BFloat16,
        "acc_dtype": cutlass.Float32,
        "lse_dtype": cutlass.Float32,
        "mma_qk_tiler_mn": (128, 128),
        "mma_pv_tiler_mn": (128, 256),
        "split_kv": 1,
        "is_persistent": False,
        "is_var_seq": True,
        "is_var_split_kv": False,
        "page_size_cmp": 128,
    }

    def supports(**overrides):
        return can_implement(**{**config, **overrides})

    assert supports()
    assert not supports(page_size_cmp=0)
    assert not supports(page_size_cmp=48)
    assert not supports(split_kv=0)
    assert not supports(split_kv=MAX_SPLITS + 1)
    assert not supports(B=8, S=8192)
    assert supports(B=8, S=8192, is_persistent=True)


@pytest.mark.cpu_only
def test_hca_static_contract_distributes_swa_gather():
    pytest.importorskip("cutlass")
    import cutlass

    from flashinfer.cute_dsl.attention.dsa.hca_fp8 import (
        BlackwellHeavilyCompressedAttentionForwardFP8,
    )

    kernel = BlackwellHeavilyCompressedAttentionForwardFP8(
        cutlass.Float32,
        cutlass.Float32,
        (128, 128),
        (128, 256),
        1,
        128,
        0.0,
        True,
        False,
        False,
    )
    kernel._setup_attributes()

    assert kernel.swa_load_warp_ids == (12, 13, 14, 15)
    assert kernel.swa_k_iters_per_warp == 1
    assert kernel.swa_v_iters_per_warp == 1
    assert kernel.load_k_stage == 3
    assert kernel.load_v_stage == 2
