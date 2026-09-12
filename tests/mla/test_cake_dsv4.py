import pytest
import torch

from flashinfer.jit.cake_dsv4 import get_cake_dsv4_spec
from flashinfer.mla.cake_dsv4 import (
    _descriptor_workspace,
    _padded_sparse_indices,
    _route,
)


# Both cache layouts and sink settings retain the same physical selection.
@pytest.mark.parametrize(
    "dtype,num_heads,batch_size,max_q_len,ragged,sparse_topk,page_size,expected",
    [
        pytest.param(
            torch.bfloat16,
            64,
            3,
            5,
            True,
            128,
            1,
            "bf16_swa128_single_cta",
            id="case-00",
        ),
        pytest.param(
            torch.bfloat16,
            64,
            3,
            5,
            True,
            640,
            64,
            "bf16_h64_compressed_q8_v38",
            id="case-01",
        ),
        pytest.param(
            torch.bfloat16,
            64,
            3,
            5,
            True,
            260,
            2,
            "bf16_h64_compressed_q8_v38",
            id="case-02",
        ),
        pytest.param(
            torch.bfloat16,
            64,
            3,
            5,
            True,
            128,
            1,
            "bf16_swa128_single_cta",
            id="case-03",
        ),
        pytest.param(
            torch.bfloat16,
            64,
            3,
            5,
            True,
            640,
            64,
            "bf16_h64_compressed_q8_v38",
            id="case-04",
        ),
        pytest.param(
            torch.bfloat16,
            64,
            3,
            5,
            True,
            260,
            2,
            "bf16_h64_compressed_q8_v38",
            id="case-05",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            64,
            3,
            5,
            True,
            128,
            1,
            "fp8_lowhead_prefill",
            id="case-06",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            64,
            3,
            5,
            True,
            640,
            64,
            "fp8_lowhead_h64",
            id="case-07",
        ),
        pytest.param(
            torch.float8_e4m3fn, 64, 3, 5, True, 260, 2, "fp8_lowhead_h64", id="case-08"
        ),
        pytest.param(
            torch.float8_e4m3fn,
            64,
            3,
            5,
            True,
            128,
            1,
            "fp8_lowhead_prefill",
            id="case-09",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            64,
            3,
            5,
            True,
            640,
            64,
            "fp8_lowhead_h64",
            id="case-10",
        ),
        pytest.param(
            torch.float8_e4m3fn, 64, 3, 5, True, 260, 2, "fp8_lowhead_h64", id="case-11"
        ),
        pytest.param(
            torch.bfloat16,
            64,
            3,
            5,
            True,
            128,
            1,
            "bf16_swa128_single_cta",
            id="case-12",
        ),
        pytest.param(
            torch.bfloat16,
            64,
            3,
            5,
            True,
            640,
            64,
            "bf16_h64_compressed_q8_v38",
            id="case-13",
        ),
        pytest.param(
            torch.bfloat16,
            64,
            3,
            5,
            True,
            388,
            2,
            "bf16_h64_compressed_q8_v38",
            id="case-14",
        ),
        pytest.param(
            torch.bfloat16,
            64,
            3,
            5,
            True,
            128,
            1,
            "bf16_swa128_single_cta",
            id="case-15",
        ),
        pytest.param(
            torch.bfloat16,
            64,
            3,
            5,
            True,
            640,
            64,
            "bf16_h64_compressed_q8_v38",
            id="case-16",
        ),
        pytest.param(
            torch.bfloat16,
            64,
            3,
            5,
            True,
            388,
            2,
            "bf16_h64_compressed_q8_v38",
            id="case-17",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            64,
            3,
            5,
            True,
            128,
            1,
            "fp8_lowhead_prefill",
            id="case-18",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            64,
            3,
            5,
            True,
            640,
            64,
            "fp8_lowhead_h64",
            id="case-19",
        ),
        pytest.param(
            torch.float8_e4m3fn, 64, 3, 5, True, 388, 2, "fp8_lowhead_h64", id="case-20"
        ),
        pytest.param(
            torch.float8_e4m3fn,
            64,
            3,
            5,
            True,
            128,
            1,
            "fp8_lowhead_prefill",
            id="case-21",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            64,
            3,
            5,
            True,
            640,
            64,
            "fp8_lowhead_h64",
            id="case-22",
        ),
        pytest.param(
            torch.float8_e4m3fn, 64, 3, 5, True, 388, 2, "fp8_lowhead_h64", id="case-23"
        ),
        pytest.param(
            torch.bfloat16, 128, 3, 5, True, 128, 1, "bf16_h128_swa128", id="case-24"
        ),
        pytest.param(
            torch.bfloat16,
            128,
            3,
            5,
            True,
            1152,
            64,
            "bf16_h128_topk4x_v52",
            id="case-25",
        ),
        pytest.param(
            torch.bfloat16, 128, 3, 5, True, 260, 2, "bf16_h128_topk128x", id="case-26"
        ),
        pytest.param(
            torch.bfloat16, 128, 3, 5, True, 128, 1, "bf16_h128_swa128", id="case-27"
        ),
        pytest.param(
            torch.bfloat16,
            128,
            3,
            5,
            True,
            1152,
            64,
            "bf16_h128_topk4x_v52",
            id="case-28",
        ),
        pytest.param(
            torch.bfloat16, 128, 3, 5, True, 260, 2, "bf16_h128_topk128x", id="case-29"
        ),
        pytest.param(
            torch.float8_e4m3fn, 128, 3, 5, True, 128, 1, "fp8_h128", id="case-30"
        ),
        pytest.param(
            torch.float8_e4m3fn, 128, 3, 5, True, 1152, 64, "fp8_h128", id="case-31"
        ),
        pytest.param(
            torch.float8_e4m3fn, 128, 3, 5, True, 260, 2, "fp8_h128", id="case-32"
        ),
        pytest.param(
            torch.float8_e4m3fn, 128, 3, 5, True, 128, 1, "fp8_h128", id="case-33"
        ),
        pytest.param(
            torch.float8_e4m3fn, 128, 3, 5, True, 1152, 64, "fp8_h128", id="case-34"
        ),
        pytest.param(
            torch.float8_e4m3fn, 128, 3, 5, True, 260, 2, "fp8_h128", id="case-35"
        ),
        pytest.param(
            torch.bfloat16, 128, 3, 5, True, 128, 1, "bf16_h128_swa128", id="case-36"
        ),
        pytest.param(
            torch.bfloat16,
            128,
            3,
            5,
            True,
            1152,
            64,
            "bf16_h128_topk4x_v52",
            id="case-37",
        ),
        pytest.param(
            torch.bfloat16, 128, 3, 5, True, 388, 2, "bf16_h128_topk128x", id="case-38"
        ),
        pytest.param(
            torch.bfloat16, 128, 3, 5, True, 128, 1, "bf16_h128_swa128", id="case-39"
        ),
        pytest.param(
            torch.bfloat16,
            128,
            3,
            5,
            True,
            1152,
            64,
            "bf16_h128_topk4x_v52",
            id="case-40",
        ),
        pytest.param(
            torch.bfloat16, 128, 3, 5, True, 388, 2, "bf16_h128_topk128x", id="case-41"
        ),
        pytest.param(
            torch.float8_e4m3fn, 128, 3, 5, True, 128, 1, "fp8_h128", id="case-42"
        ),
        pytest.param(
            torch.float8_e4m3fn, 128, 3, 5, True, 1152, 64, "fp8_h128", id="case-43"
        ),
        pytest.param(
            torch.float8_e4m3fn, 128, 3, 5, True, 388, 2, "fp8_h128", id="case-44"
        ),
        pytest.param(
            torch.float8_e4m3fn, 128, 3, 5, True, 128, 1, "fp8_h128", id="case-45"
        ),
        pytest.param(
            torch.float8_e4m3fn, 128, 3, 5, True, 1152, 64, "fp8_h128", id="case-46"
        ),
        pytest.param(
            torch.float8_e4m3fn, 128, 3, 5, True, 388, 2, "fp8_h128", id="case-47"
        ),
        pytest.param(
            torch.bfloat16, 8, 3, 5, True, 128, 1, "bf16_h8_swa128_v43", id="case-48"
        ),
        pytest.param(
            torch.bfloat16,
            8,
            3,
            5,
            True,
            192,
            64,
            "bf16_h8_h16_source_exact",
            id="case-49",
        ),
        pytest.param(
            torch.bfloat16,
            8,
            3,
            5,
            True,
            260,
            2,
            "bf16_h8_h16_source_exact",
            id="case-50",
        ),
        pytest.param(
            torch.bfloat16, 8, 3, 5, True, 128, 1, "bf16_h8_swa128_v43", id="case-51"
        ),
        pytest.param(
            torch.bfloat16,
            8,
            3,
            5,
            True,
            192,
            64,
            "bf16_h8_h16_source_exact",
            id="case-52",
        ),
        pytest.param(
            torch.bfloat16,
            8,
            3,
            5,
            True,
            260,
            2,
            "bf16_h8_h16_source_exact",
            id="case-53",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            8,
            3,
            5,
            True,
            128,
            1,
            "fp8_lowhead_prefill",
            id="case-54",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            8,
            3,
            5,
            True,
            192,
            64,
            "fp8_lowhead_one_partition",
            id="case-55",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            8,
            3,
            5,
            True,
            260,
            2,
            "fp8_lowhead_split",
            id="case-56",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            8,
            3,
            5,
            True,
            128,
            1,
            "fp8_lowhead_prefill",
            id="case-57",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            8,
            3,
            5,
            True,
            192,
            64,
            "fp8_lowhead_one_partition",
            id="case-58",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            8,
            3,
            5,
            True,
            260,
            2,
            "fp8_lowhead_split",
            id="case-59",
        ),
        pytest.param(
            torch.bfloat16,
            16,
            3,
            5,
            True,
            128,
            1,
            "bf16_h16_h32_swa128_v44",
            id="case-60",
        ),
        pytest.param(
            torch.bfloat16,
            16,
            3,
            5,
            True,
            256,
            64,
            "bf16_h8_h16_source_exact",
            id="case-61",
        ),
        pytest.param(
            torch.bfloat16,
            16,
            3,
            5,
            True,
            260,
            2,
            "bf16_h8_h16_source_exact",
            id="case-62",
        ),
        pytest.param(
            torch.bfloat16,
            16,
            3,
            5,
            True,
            128,
            1,
            "bf16_h16_h32_swa128_v44",
            id="case-63",
        ),
        pytest.param(
            torch.bfloat16,
            16,
            3,
            5,
            True,
            256,
            64,
            "bf16_h8_h16_source_exact",
            id="case-64",
        ),
        pytest.param(
            torch.bfloat16,
            16,
            3,
            5,
            True,
            260,
            2,
            "bf16_h8_h16_source_exact",
            id="case-65",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            16,
            3,
            5,
            True,
            128,
            1,
            "fp8_lowhead_prefill",
            id="case-66",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            16,
            3,
            5,
            True,
            256,
            64,
            "fp8_lowhead_one_partition",
            id="case-67",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            16,
            3,
            5,
            True,
            260,
            2,
            "fp8_lowhead_split",
            id="case-68",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            16,
            3,
            5,
            True,
            128,
            1,
            "fp8_lowhead_prefill",
            id="case-69",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            16,
            3,
            5,
            True,
            256,
            64,
            "fp8_lowhead_one_partition",
            id="case-70",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            16,
            3,
            5,
            True,
            260,
            2,
            "fp8_lowhead_split",
            id="case-71",
        ),
        pytest.param(
            torch.bfloat16,
            32,
            3,
            5,
            True,
            128,
            1,
            "bf16_h16_h32_swa128_v44",
            id="case-72",
        ),
        pytest.param(
            torch.bfloat16, 32, 3, 5, True, 384, 64, "bf16_h32_topk4x_v38", id="case-73"
        ),
        pytest.param(
            torch.bfloat16,
            32,
            3,
            5,
            True,
            260,
            2,
            "bf16_h32_topk128x_early_v47",
            id="case-74",
        ),
        pytest.param(
            torch.bfloat16,
            32,
            3,
            5,
            True,
            128,
            1,
            "bf16_h16_h32_swa128_v44",
            id="case-75",
        ),
        pytest.param(
            torch.bfloat16, 32, 3, 5, True, 384, 64, "bf16_h32_topk4x_v38", id="case-76"
        ),
        pytest.param(
            torch.bfloat16,
            32,
            3,
            5,
            True,
            260,
            2,
            "bf16_h32_topk128x_early_v47",
            id="case-77",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            32,
            3,
            5,
            True,
            128,
            1,
            "fp8_lowhead_prefill",
            id="case-78",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            32,
            3,
            5,
            True,
            384,
            64,
            "fp8_lowhead_split",
            id="case-79",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            32,
            3,
            5,
            True,
            260,
            2,
            "fp8_lowhead_split",
            id="case-80",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            32,
            3,
            5,
            True,
            128,
            1,
            "fp8_lowhead_prefill",
            id="case-81",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            32,
            3,
            5,
            True,
            384,
            64,
            "fp8_lowhead_split",
            id="case-82",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            32,
            3,
            5,
            True,
            260,
            2,
            "fp8_lowhead_split",
            id="case-83",
        ),
        pytest.param(
            torch.bfloat16,
            64,
            1,
            128,
            True,
            128,
            1,
            "bf16_h64_guard_q_tma_batch_r25",
            id="case-84",
        ),
        pytest.param(
            torch.bfloat16, 64, 2, 5, False, 640, 64, "bf16_h64_fixed_q", id="case-85"
        ),
        pytest.param(
            torch.bfloat16, 64, 2, 257, True, 640, 64, "bf16_h64_prefill", id="case-86"
        ),
        pytest.param(
            torch.float8_e4m3fn,
            64,
            2,
            257,
            True,
            640,
            64,
            "fp8_h64_source_exact",
            id="case-87",
        ),
        pytest.param(
            torch.bfloat16,
            128,
            2,
            257,
            True,
            1152,
            64,
            "bf16_h128_prefill_v42",
            id="case-88",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            128,
            2,
            257,
            True,
            1152,
            64,
            "fp8_h128_prefill_source_persistent",
            id="case-89",
        ),
        pytest.param(
            torch.bfloat16, 64, 2, 257, True, 640, 64, "bf16_h64_prefill", id="case-90"
        ),
        pytest.param(
            torch.float8_e4m3fn,
            64,
            2,
            257,
            True,
            640,
            64,
            "fp8_h64_source_exact",
            id="case-91",
        ),
        pytest.param(
            torch.bfloat16,
            128,
            2,
            257,
            True,
            1152,
            64,
            "bf16_h128_prefill_v42",
            id="case-92",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            128,
            2,
            257,
            True,
            1152,
            64,
            "fp8_h128_prefill_source_persistent",
            id="case-93",
        ),
    ],
)
def test_cake_dsv4_semantic_routes(
    dtype, num_heads, batch_size, max_q_len, ragged, sparse_topk, page_size, expected
):
    assert (
        _route(
            dtype=dtype,
            num_heads=num_heads,
            batch_size=batch_size,
            max_q_len=max_q_len,
            ragged=ragged,
            sparse_topk=sparse_topk,
            compressed_page_size=page_size,
        )
        == expected
    )


@pytest.mark.parametrize(
    "num_heads,batch_size,page_size,expected",
    [
        (64, 1, 64, "fp8_lowhead_prefill"),
        (64, 2, 2, "fp8_lowhead_prefill"),
        (128, 1, 64, "fp8_h128"),
        (128, 2, 2, "fp8_h128"),
    ],
)
def test_fp8_prefill_keeps_batch_and_cache_layout_predicates(
    num_heads, batch_size, page_size, expected
):
    assert (
        _route(
            dtype=torch.float8_e4m3fn,
            num_heads=num_heads,
            batch_size=batch_size,
            max_q_len=257,
            ragged=True,
            sparse_topk=640 if num_heads == 64 else 1152,
            compressed_page_size=page_size,
        )
        == expected
    )


def test_unexported_variant_fails():
    with pytest.raises(ValueError, match="no generated source contract"):
        get_cake_dsv4_spec("unexported_variant")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_descriptor_workspace_preserves_each_tensor_layout():
    workspace = torch.empty(1, dtype=torch.uint8, device="cuda")
    tensor = torch.empty((4, 8), dtype=torch.bfloat16, device="cuda")
    descriptors = _descriptor_workspace(workspace, "test_variant", [tensor], 256)
    assert descriptors.numel() == 256
    assert descriptors.data_ptr() % 128 == 0
    repeated = _descriptor_workspace(workspace, "test_variant", [tensor], 256)
    assert repeated.data_ptr() == descriptors.data_ptr()

    reshaped = tensor.reshape(8, 4)
    other_layout = _descriptor_workspace(workspace, "test_variant", [reshaped], 256)
    assert other_layout.data_ptr() != descriptors.data_ptr()
    another = _descriptor_workspace(workspace, "test_variant", [tensor.clone()], 256)
    assert another.data_ptr() not in (descriptors.data_ptr(), other_layout.data_ptr())
    assert (
        _descriptor_workspace(workspace, "test_variant", [tensor], 256).data_ptr()
        == descriptors.data_ptr()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sparse_index_tail_tracks_in_place_updates():
    workspace = torch.empty(1, dtype=torch.uint8, device="cuda")
    indices = torch.arange(2 * 388, dtype=torch.int32, device="cuda").reshape(2, 388)
    padded = _padded_sparse_indices(workspace, indices)
    torch.testing.assert_close(padded[: indices.numel()], indices.reshape(-1))
    assert padded.numel() == 388 + 1152
    indices.add_(10)
    refreshed = _padded_sparse_indices(workspace, indices)
    torch.testing.assert_close(refreshed[: indices.numel()], indices.reshape(-1))
    assert torch.all(refreshed[indices.numel() :] == -1).item()
