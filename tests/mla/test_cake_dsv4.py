import re

import pytest
import torch

from flashinfer.jit.cake_dsv4 import _ARCH_REGISTRATIONS, get_cake_dsv4_spec
from flashinfer.mla import cake_dsv4 as cake
from flashinfer.mla.cake_dsv4 import (
    KERNEL_METADATA_PARAMS,
    _route,
    cake_dsv4_workspace_layout,
    cake_dsv4_workspace_reset,
    get_cake_dsv4_workspace_bytes,
    resolve_cake_dsv4_sparse_metadata,
)


# Both cache layouts and sink settings retain the same physical selection.
def _canonical_query_tokens(batch_size: int, max_q_len: int, ragged: bool) -> int:
    """Query-token count of a canonical contract row.

    Ragged rows follow the contract's ``linspace_half_to_max`` rule (request
    lengths ``linspace(ceil(q/2), q, batch)`` rounded), so the 3 x q_len 5
    decode rows carry 12 tokens; dense rows carry ``batch_size * max_q_len``.
    """
    if not ragged:
        return batch_size * max_q_len
    minimum = max(1, (max_q_len + 1) // 2)
    return int(
        torch.linspace(minimum, max_q_len, batch_size).round().to(torch.int32).sum()
    )


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
            "fp8_h128_prefill_source_persistent_uniform",
            id="case-07",
        ),
        pytest.param(
            torch.float8_e4m3fn, 64, 3, 5, True, 260, 2, "fp8_lowhead_h64", id="case-08"
        ),
        # FP8/H64 rows admitted to the persistent body with >= 128
        # tokens (dense 2 x 64) run the single-CTA M64 program; 127 tokens keep
        # the FP8/H128 persistent program.
        pytest.param(
            torch.float8_e4m3fn,
            64,
            2,
            64,
            False,
            260,
            2,
            "fp8_h64_prefill_source_persistent_m64_multi_tile",
            id="w14-h64-w260-128tok",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            64,
            1,
            127,
            False,
            260,
            2,
            "fp8_h128_prefill_source_persistent_uniform",
            id="w14-h64-w260-127tok",
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
            "fp8_h128_prefill_source_persistent_uniform",
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
            "fp8_h128_prefill_source_persistent_uniform",
            id="case-19",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            64,
            3,
            5,
            True,
            388,
            2,
            "fp8_h128_prefill_source_persistent_uniform",
            id="case-20",
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
            "fp8_h128_prefill_source_persistent_uniform",
            id="case-22",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            64,
            3,
            5,
            True,
            388,
            2,
            "fp8_h128_prefill_source_persistent_uniform",
            id="case-23",
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
            torch.float8_e4m3fn,
            128,
            3,
            5,
            True,
            128,
            1,
            "fp8_h128_prefill_source_persistent_uniform",
            id="case-30",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            128,
            3,
            5,
            True,
            1152,
            64,
            "fp8_h128_prefill_source_persistent_uniform",
            id="case-31",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            128,
            3,
            5,
            True,
            260,
            2,
            "fp8_h128_prefill_source_persistent_uniform",
            id="case-32",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            128,
            3,
            5,
            True,
            128,
            1,
            "fp8_h128_prefill_source_persistent_uniform",
            id="case-33",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            128,
            3,
            5,
            True,
            1152,
            64,
            "fp8_h128_prefill_source_persistent_uniform",
            id="case-34",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            128,
            3,
            5,
            True,
            260,
            2,
            "fp8_h128_prefill_source_persistent_uniform",
            id="case-35",
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
            torch.float8_e4m3fn,
            128,
            3,
            5,
            True,
            128,
            1,
            "fp8_h128_prefill_source_persistent_uniform",
            id="case-42",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            128,
            3,
            5,
            True,
            1152,
            64,
            "fp8_h128_prefill_source_persistent_uniform",
            id="case-43",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            128,
            3,
            5,
            True,
            388,
            2,
            "fp8_h128_prefill_source_persistent_uniform",
            id="case-44",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            128,
            3,
            5,
            True,
            128,
            1,
            "fp8_h128_prefill_source_persistent_uniform",
            id="case-45",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            128,
            3,
            5,
            True,
            1152,
            64,
            "fp8_h128_prefill_source_persistent_uniform",
            id="case-46",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            128,
            3,
            5,
            True,
            388,
            2,
            "fp8_h128_prefill_source_persistent_uniform",
            id="case-47",
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
            "fp8_h8_h16_source_exact",
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
            "fp8_h8_h16_source_exact",
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
            "fp8_h8_h16_source_exact",
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
            "fp8_h8_h16_source_exact",
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
            8,
            3,
            5,
            True,
            256,
            64,
            "fp8_lowhead_one_partition",
            id="fp8-h8-w256-off-profile-one-partition",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            16,
            2,
            5,
            True,
            256,
            64,
            "fp8_lowhead_one_partition",
            id="fp8-h16-b2-off-profile-one-partition",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            16,
            3,
            5,
            True,
            256,
            64,
            "fp8_h8_h16_source_exact",
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
            "fp8_h8_h16_source_exact",
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
            "fp8_h8_h16_source_exact",
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
            "fp8_h8_h16_source_exact",
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
            torch.bfloat16,
            32,
            3,
            5,
            True,
            384,
            64,
            "bf16_h32_topk128x_early_v47",
            id="case-73",
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
            torch.bfloat16,
            32,
            3,
            5,
            True,
            384,
            64,
            "bf16_h32_topk128x_early_v47",
            id="case-76",
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
            "fp8_lowhead_one_partition",
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
            "fp8_lowhead_one_partition",
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
            "fp8_lowhead_one_partition",
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
            "fp8_lowhead_one_partition",
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
            torch.bfloat16,
            64,
            2,
            5,
            False,
            640,
            64,
            "bf16_h64_compressed_q8_v38",
            id="case-85",
        ),
        pytest.param(
            torch.bfloat16,
            64,
            2,
            5,
            False,
            128,
            1,
            "bf16_swa128_single_cta",
            id="case-85-swa",
        ),
        pytest.param(
            torch.bfloat16,
            64,
            6,
            4,
            False,
            640,
            64,
            "bf16_h64_prefill",
            id="case-85-dense24",
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
        # FP8/H64 many-token rows (dense 3 x 64 = 192 query tokens) with
        # >= 2 complete sparse tiles take the persistent FP8 body on both
        # targets (the cluster producers sat at 0.19-0.79x
        # vs trtllm-gen from 64 tokens on); from 128 tokens the H64-specific
        # single-CTA M64 program.
        pytest.param(
            torch.float8_e4m3fn,
            64,
            3,
            64,
            False,
            640,
            64,
            "fp8_h64_prefill_source_persistent_m64_multi_tile",
            id="h64-w640-192tok",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            64,
            3,
            64,
            False,
            388,
            2,
            "fp8_h64_prefill_source_persistent_m64_multi_tile",
            id="h64-w388-192tok",
        ),
        # Off-contract low-head width beyond the three sparse tiles one
        # producer partition owns: no kernel is exported, rejected up front.
        pytest.param(
            torch.float8_e4m3fn,
            32,
            3,
            5,
            True,
            640,
            64,
            ValueError,
            id="lowhead-w640-rejected",
        ),
        # BF16 H8/H16 rows with a compressed cache outside the source-exact
        # shape lock have no kernel either.
        pytest.param(
            torch.bfloat16,
            16,
            8,
            8,
            True,
            260,
            2,
            ValueError,
            id="bf16-h16-compressed-rejected",
        ),
    ],
)
def test_cake_dsv4_semantic_routes(
    dtype, num_heads, batch_size, max_q_len, ragged, sparse_topk, page_size, expected
):
    kwargs = dict(
        dtype=dtype,
        num_heads=num_heads,
        batch_size=batch_size,
        max_q_len=max_q_len,
        ragged=ragged,
        sparse_topk=sparse_topk,
        compressed_page_size=page_size,
        num_query_tokens=_canonical_query_tokens(batch_size, max_q_len, ragged),
    )
    if expected is ValueError:
        # Shapes without an exported kernel are rejected before any launch.
        with pytest.raises(ValueError, match="has no .* kernel"):
            _route(**kwargs)
    else:
        assert _route(**kwargs) == expected


@pytest.mark.parametrize(
    "num_query_tokens,sparse_topk,page_size,expected",
    [
        # SWA-only rows: dedicated producer below the bound, persistent prefill body from 64 tokens.
        (12, 128, None, "bf16_h128_swa128"),
        (63, 128, None, "bf16_h128_swa128"),
        (64, 128, None, "bf16_h128_prefill_v42"),  # hardening-000023-like
        (128, 128, None, "bf16_h128_prefill_v42"),
        (512, 128, None, "bf16_h128_prefill_v42"),  # hardening-000035
        # topk4x rows: split5 / single owner at 12 tokens, persistent prefill body from 64 tokens.
        (12, 1152, 64, "bf16_h128_topk4x_v52"),
        # Width 640 below the token bound has no H128 kernel: rejected up front.
        (12, 640, 64, ValueError),
        (64, 640, 64, "bf16_h128_prefill_v42"),  # hardening-000021
        (512, 640, 64, "bf16_h128_prefill_v42"),  # hardening-000037
        # topk128x rows keep their own rule (W12), whatever the token count.
        (64, 260, 2, "bf16_h128_topk128x"),
    ],
)
def test_bf16_h128_swa_and_topk4x_rows_use_the_persistent_prefill_body_from_64_tokens(
    num_query_tokens, sparse_topk, page_size, expected
):
    kwargs = dict(
        dtype=torch.bfloat16,
        num_heads=128,
        batch_size=64,
        max_q_len=8,
        ragged=True,
        sparse_topk=sparse_topk,
        compressed_page_size=page_size,
        num_query_tokens=num_query_tokens,
    )
    if expected is ValueError:
        with pytest.raises(ValueError, match="has no BF16 H128 kernel"):
            _route(**kwargs)
    else:
        assert _route(**kwargs) == expected


@pytest.mark.parametrize(
    "num_query_tokens,page_size,sparse_topk,expected",
    [
        (
            12,
            64,
            1152,
            "fp8_h128_prefill_source_persistent_uniform",
        ),  # 12-token decode rows too
        (15, 2, 260, "fp8_h128_prefill_source_persistent_uniform"),
        (16, 64, 640, "fp8_h128_prefill_source_persistent_uniform"),
        (32, 64, 640, "fp8_h128_prefill_source_persistent_uniform"),
        (128, 2, 260, "fp8_h128_prefill_source_persistent"),
        (128, None, 128, "fp8_h128_prefill_source_persistent"),
    ],
)
def test_fp8_h128_rows_all_use_the_persistent_body(
    num_query_tokens, page_size, sparse_topk, expected
):
    assert (
        _route(
            dtype=torch.float8_e4m3fn,
            num_heads=128,
            batch_size=8,
            max_q_len=8,
            ragged=True,
            sparse_topk=sparse_topk,
            compressed_page_size=page_size,
            num_query_tokens=num_query_tokens,
        )
        == expected
    )


@pytest.mark.parametrize(
    "num_query_tokens,page_size,sparse_topk,expected",
    [
        (
            12,
            64,
            640,
            "bf16_h64_compressed_q8_v38",
        ),  # canonical 12-token rows keep the portfolio producer
        (16, 2, 260, "bf16_h64_compressed_q8_v38"),
        (24, 2, 260, "bf16_h64_prefill"),
        (64, 64, 640, "bf16_h64_prefill"),
        (512, 64, 640, "bf16_h64_prefill"),
    ],
)
def test_bf16_h64_compressed_rows_use_the_prefill_body_from_24_tokens(
    num_query_tokens, page_size, sparse_topk, expected
):
    assert (
        _route(
            dtype=torch.bfloat16,
            num_heads=64,
            batch_size=8,
            max_q_len=8,
            ragged=True,
            sparse_topk=sparse_topk,
            compressed_page_size=page_size,
            num_query_tokens=num_query_tokens,
        )
        == expected
    )


@pytest.mark.parametrize(
    "num_heads,batch_size,page_size,expected",
    [
        (64, 1, 64, "fp8_lowhead_prefill"),
        (64, 2, 2, "fp8_lowhead_prefill"),
        # FP8/H128 rows with 16+ tokens use the persistent body for every
        # batch size and cache layout (measured 1.12-1.75x vs trtllm-gen on
        # the 16-256 token MTP rows, topk4x and topk128x alike).
        (128, 1, 64, "fp8_h128_prefill_source_persistent"),
        (128, 2, 2, "fp8_h128_prefill_source_persistent"),
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
            num_query_tokens=batch_size * 257,
        )
        == expected
    )


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_unexported_variant_fails(arch):
    with pytest.raises(ValueError, match="no generated source contract"):
        get_cake_dsv4_spec("unexported_variant", arch=arch)


# --------------------------------------------------------------------------- #
# Hardening (flashinfer#4671): name-based binding, workspace, metadata         #
# --------------------------------------------------------------------------- #

_ARCHES = ("sm_100a", "sm_103a")
_DEVICE = torch.device("cpu")


def _aligned_u8(num_bytes: int) -> torch.Tensor:
    """CPU byte buffer whose data pointer is 128-byte aligned (CPU allocs are 64B)."""
    backing = torch.empty(num_bytes + 128, dtype=torch.uint8)
    offset = (-backing.data_ptr()) % 128
    return backing[offset : offset + num_bytes]


def _combined_metadata(rows: int, compressed: int, *, value_base: int = 0):
    topk = 128 + compressed
    table = (
        torch.arange(rows * topk, dtype=torch.int32).reshape(rows, topk) + value_base
    )
    lens = torch.full((rows,), 128 + compressed // 2, dtype=torch.int32)
    return table, lens


# Tensors run_cake_dsv4 places in the host value table that a generated TMA
# descriptor may alias (see cake._TMA_SOURCE_ALIASES).
_HOST_TMA_SOURCE_TENSORS = frozenset(
    {
        "Q",
        "SWA_cache",
        "compressed_KV_cache",
        "O",
        # NVFP4 route: the partial-O tile view and the gather4 cache views are
        # tensor values of run_cake_dsv4_nvfp4 (see _TENSOR_VALUE_NAMES).
        "partial_O_tiles",
        "main_cache_g4d",
        "main_cache_g4f",
        "extra_cache_g4d",
        "extra_cache_g4f",
    }
)


@pytest.mark.parametrize("arch", _ARCHES)
def test_registered_arg_plans_use_known_names(arch):
    """Every generated argument is either bindable by name or a documented retired name."""
    unknown = []
    unaliased_tma = []
    for variant, spec in _ARCH_REGISTRATIONS[arch]["variants"].items():
        for kind, name in spec["arg_plan"]:
            canonical = cake.canonical_arg_name(kind, name)
            if kind == "tma_buffer" and canonical not in _HOST_TMA_SOURCE_TENSORS:
                # A descriptor name the host binds only by vocabulary would still
                # fail at launch: ``_bind_argument`` needs a tensor value for it.
                unaliased_tma.append((variant, name, canonical))
            if (
                cake.is_bindable_arg(kind, name)
                or canonical in cake._RETIRED_ARG_REASONS
            ):
                continue
            unknown.append((variant, kind, name))
    assert unknown == []
    assert unaliased_tma == []


_PUBLIC_COMPILE_FLAGS = {
    "-std=c++17",
    "--use_fast_math",
    "-Xptxas=--register-usage-level=10",
}
# The NVFP4 decode swap / tile / pv schedules ship as one kernel-template unit each.  A
# variant's module compiles that unit with two preprocessor defines (a public nvcc option):
# the member's select macro and the variant's own instance macro, so it instantiates only
# its kernel.  Every instantiation is proven SASS-identical to the per-knob program it
# replaced before the unit ships.
_TEMPLATE_SELECT_DEFINE = re.compile(
    r"-DCAKE_DSV4_NVFP4_(?P<member>[A-Z0-9_]+)_SELECT=1"
)
_TEMPLATE_MEMBERS = {"DECODE_SWAP", "DECODE_TILE", "DECODE_PV"}


@pytest.mark.parametrize("arch", _ARCHES)
def test_registered_min_cuda_version_marks_the_nvfp4_programs(arch):
    """The NVFP4 programs spell the Blackwell QMUL4 as the PTX ISA 9.4 packed
    multiply, so their registrations require CUDA 13.4 (the loader refuses older
    toolkits by name); every other generated program carries no toolkit floor."""
    for variant, spec in _ARCH_REGISTRATIONS[arch]["variants"].items():
        if variant.startswith("nvfp4_"):
            assert spec.get("min_cuda_version") == "13.4", variant
        else:
            assert "min_cuda_version" not in spec, variant


@pytest.mark.parametrize("arch", _ARCHES)
def test_registered_compile_flags_are_public(arch):
    """Exported programs must build with public nvcc/ptxas options only.

    The FP8/H128 persistent prefill variant additionally pins the ptxas
    register-usage level: without it ptxas re-orders the softmax exp2/convert
    chains across the P-publication fence and the exported build runs 3-5 %
    slower than the source build on the 16-tile prefill shapes.  The uniform
    (sub-128-token decode) variant of the same body ships without the pin:
    its softmax chain is not what paces the item and the pinned build reads
    1-2 % slower on the uniform decode rows, so the flag must stay off there.

    The kernel-template members (NVFP4 decode swap, tile, pv) additionally
    carry their select and instance defines: the select macro names one of
    the three members and the variant, the instance macro is the variant's
    own; nothing else may be added.
    """
    pin = "-Xptxas=--register-usage-level=10"
    variants = _ARCH_REGISTRATIONS[arch]["variants"]
    assert "fp8_h128_prefill_source_persistent" in variants
    assert "fp8_h128_prefill_source_persistent_uniform" in variants
    for variant, spec in variants.items():
        flags = set(spec["compile_flags"])
        selects = {flag for flag in flags if _TEMPLATE_SELECT_DEFINE.fullmatch(flag)}
        if selects:
            assert len(selects) == 1, (variant, sorted(selects))
            select = next(iter(selects))
            member = _TEMPLATE_SELECT_DEFINE.fullmatch(select).group("member")
            assert member in _TEMPLATE_MEMBERS, (variant, select)
            assert variant.upper().startswith(f"NVFP4_{member}_"), (variant, select)
            instance = f"-DCAKE_DSV4_{variant.upper()}=1"
            assert instance in flags, (variant, sorted(flags))
            flags -= {select, instance}
        assert flags <= _PUBLIC_COMPILE_FLAGS, (variant, sorted(flags))
        if variant == "fp8_h128_prefill_source_persistent":
            assert pin in flags, variant
        elif variant == "fp8_h128_prefill_source_persistent_uniform":
            assert pin not in flags, variant


def test_metadata_param_vocabulary_is_bindable():
    for name in KERNEL_METADATA_PARAMS:
        kind = "buffer" if name.endswith(("indices", "lens")) else "parameter"
        assert cake.is_bindable_arg(kind, name), name
    for tma_name in (
        "tmap_q",
        "tmap_swa_k",
        "tmap_swa_kv",
        "tmap_compressed_v",
        "tmap_o",
    ):
        assert cake.is_bindable_arg("tma_buffer", tma_name)
    assert cake.canonical_arg_name("parameter", "num_q_heads") == "num_heads"
    assert cake.canonical_arg_name("parameter", "num_split") == "num_splits"
    assert not cake.is_bindable_arg("buffer", "completion_base")
    assert not cake.is_bindable_arg("parameter", "completion_base")


_FAKE_PLAN = [
    ("parameter", "sparse_topk_lens_offset"),
    ("tma_buffer", "tmap_swa_kv"),
    ("buffer", "compressed_indices"),
    ("grid", "grid_z"),
    ("tma_buffer", "tmap_q"),
    ("buffer", "swa_indices"),
    ("parameter", "compressed_index_stride"),
    ("parameter", "swa_index_stride"),
    ("buffer", "sparse_topk_lens"),
    ("parameter", "num_query_tokens"),
    ("parameter", "sparse_topk"),
    ("workspace", "tma_descriptor_workspace"),
    ("parameter", "num_q_heads"),
    ("grid", "grid_x"),
    ("grid", "grid_y"),
    ("tma_buffer", "tmap_compressed_kv"),
    ("buffer", "O"),
]


class _Recorder:
    def __init__(self):
        self.calls = []

    def run(self, *args):
        self.calls.append(args)


def _install_fake_variants(monkeypatch, plans, *, tma_bytes=384):
    """Route get_cake_dsv4_spec / module loading to fake plans and a recorder."""
    import flashinfer.jit.cake_dsv4 as jit

    recorder = _Recorder()

    def fake_spec(variant, *, arch):
        if variant not in plans:
            raise ValueError(f"no generated source contract: {variant}")
        return {
            "arch": arch,
            "entry": "run",
            "arg_plan": plans[variant],
            "tma_workspace_bytes": tma_bytes
            if any(kind == "workspace" for kind, _ in plans[variant])
            else 0,
        }

    class _Sequence:
        """Stand-in for the run_sequence host helper: calls each launcher in order."""

        @staticmethod
        def run_sequence(*flat):
            i = 0
            while i < len(flat):
                fn, count = flat[i], flat[i + 1]
                fn(*flat[i + 2 : i + 2 + count])
                i += 2 + count

    monkeypatch.setattr(jit, "get_cake_dsv4_spec", fake_spec)
    monkeypatch.setattr(cake, "_variant_module", lambda variant, *, arch: recorder)
    monkeypatch.setattr(cake, "_sequence_module", lambda: _Sequence)
    return recorder


def test_launch_variant_binds_by_name_with_fake_arg_plan(monkeypatch):
    recorder = _install_fake_variants(monkeypatch, {"fake": _FAKE_PLAN})
    rows, compressed = 3, 132
    table, lens = _combined_metadata(rows, compressed)
    meta = resolve_cake_dsv4_sparse_metadata(table, lens, query_rows=rows)
    raw = _aligned_u8(cake._PARTIAL_OFFSET)
    q = torch.empty((rows, 64, 512), dtype=torch.bfloat16)
    swa = torch.empty((16, 512), dtype=torch.bfloat16)
    comp = torch.empty((32, 512), dtype=torch.bfloat16)
    out = torch.empty((rows, 64, 512), dtype=torch.bfloat16)
    values = {
        "Q": q,
        "SWA_cache": swa,
        "compressed_KV_cache": comp,
        "O": out,
        "num_heads": 64,
        **meta.kernel_kwargs(),
    }
    cake._launch_variant("fake", arch="sm_103a", grid=(7, 2, 1), values=values)
    (args,) = recorder.calls
    assert len(args) == len(_FAKE_PLAN)
    bound = dict(zip((name for _, name in _FAKE_PLAN), args, strict=True))
    assert bound["tmap_q"] is q
    assert bound["tmap_swa_kv"] is swa
    assert bound["tmap_compressed_kv"] is comp
    assert bound["O"] is out
    assert bound["swa_indices"] is table
    assert bound["compressed_indices"].data_ptr() == table.data_ptr() + 128 * 4
    assert bound["sparse_topk_lens"] is lens
    assert bound["swa_index_stride"] == 128 + compressed
    assert bound["compressed_index_stride"] == 128 + compressed
    assert bound["sparse_topk_lens_offset"] == 0
    assert bound["sparse_topk"] == 128 + compressed
    assert bound["num_query_tokens"] == rows
    assert bound["num_q_heads"] == 64
    assert (bound["grid_x"], bound["grid_y"], bound["grid_z"]) == (7, 2, 1)
    slab = bound["tma_descriptor_workspace"]
    assert slab.numel() == cake._DESCRIPTOR_SLAB_BYTES
    assert slab.data_ptr() % 128 == 0
    assert all(isinstance(bound[n], int) for n in ("sparse_topk", "grid_x"))
    # Descriptor storage follows the TMA source geometry: the same tensors reuse
    # it, another KV cache of the same shape gets its own, and the storage is
    # private (not carved from the caller's workspace).
    assert not (raw.data_ptr() <= slab.data_ptr() < raw.data_ptr() + raw.numel())
    cake._launch_variant("fake", arch="sm_103a", grid=(7, 2, 1), values=values)
    assert recorder.calls[-1][11] is slab
    other = torch.empty((32, 512), dtype=torch.bfloat16)
    cake._launch_variant(
        "fake",
        arch="sm_103a",
        grid=(7, 2, 1),
        values={**values, "compressed_KV_cache": other},
    )
    assert recorder.calls[-1][11] is not slab


def test_descriptor_storage_pool_is_bounded_and_capture_safe(monkeypatch):
    """Eager descriptor sets share a bounded pool; captured sets are retained."""
    recorder = _install_fake_variants(monkeypatch, {"fake": _FAKE_PLAN})
    rows, compressed = 3, 132
    table, lens = _combined_metadata(rows, compressed)
    meta = resolve_cake_dsv4_sparse_metadata(table, lens, query_rows=rows)
    q = torch.empty((rows, 64, 512), dtype=torch.bfloat16)
    swa = torch.empty((16, 512), dtype=torch.bfloat16)
    base_values = {
        "Q": q,
        "SWA_cache": swa,
        "O": torch.empty((rows, 64, 512), dtype=torch.bfloat16),
        "num_heads": 64,
        **meta.kernel_kwargs(),
    }
    caches = [torch.empty((32, 512), dtype=torch.bfloat16) for _ in range(5)]
    pool_key = ("fake", "sm_103a", q.device)
    cake._descriptor_pools.pop(pool_key, None)
    monkeypatch.setattr(cake, "_DESCRIPTOR_POOL_CAPACITY", 2)
    monkeypatch.setattr(cake, "_is_capturing", lambda device: False)

    def launch(cache):
        cake._launch_variant(
            "fake",
            arch="sm_103a",
            grid=(7, 2, 1),
            values={**base_values, "compressed_KV_cache": cache},
        )
        return recorder.calls[-1][11]

    s0, s1 = launch(caches[0]), launch(caches[1])
    assert s0.data_ptr() != s1.data_ptr()
    assert launch(caches[0]) is s0  # hit: no reassignment
    s2 = launch(
        caches[2]
    )  # pool full: the least recently used storage (s1) is reassigned
    assert s2 is s1
    assert launch(caches[0]) is s0
    assert launch(caches[1]) is s2  # cache 2 was least recently used
    pool = cake._descriptor_pools[pool_key]
    assert len(pool.live) == 2 and not pool.captured
    distinct = {s.tensor.data_ptr() for s in pool.live.values()}
    assert distinct == {s0.data_ptr(), s1.data_ptr()}

    # Capture: a resident set is retained for the process lifetime, a new one is
    # refused before any allocation or binding call.
    monkeypatch.setattr(cake, "_is_capturing", lambda device: True)
    calls = len(recorder.calls)
    with pytest.raises(RuntimeError, match="before capture"):
        launch(caches[3])
    assert len(recorder.calls) == calls and len(pool.live) == 2
    assert launch(caches[0]) is s0
    assert set(pool.captured) and len(pool.live) == 1
    monkeypatch.setattr(cake, "_is_capturing", lambda device: False)
    churned = {launch(c).data_ptr() for c in caches[1:] for _ in range(2)}
    assert s0.data_ptr() not in churned  # the captured storage is never reassigned
    assert len(pool.live) <= 2 and len(pool.captured) == 1
    assert launch(caches[0]) is s0
    # The pool allocated exactly capacity + captured storages.
    all_storages = {s.tensor.data_ptr() for s in pool.live.values()}
    all_storages |= {s.tensor.data_ptr() for s in pool.captured.values()}
    all_storages |= {s.tensor.data_ptr() for s in pool.spare}
    assert len(all_storages) == 3
    cake._descriptor_pools.pop(pool_key, None)
    assert recorder.calls[-1][11].numel() == cake._DESCRIPTOR_SLAB_BYTES


def test_launch_variant_reports_unknown_retired_and_unavailable_names(monkeypatch):
    plans = {
        "retired": [
            ("parameter", "completion_base"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "unknown": [
            ("buffer", "mystery"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "legacy": [
            ("buffer", "sparse_indices"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "ragged_only": [
            ("buffer", "cum_seq_lens_q"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "too_many_descriptors": [
            ("workspace", "tma_descriptor_workspace"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
    }
    recorder = _install_fake_variants(monkeypatch, plans, tma_bytes=4096)
    table, lens = _combined_metadata(2, 4)
    combined = resolve_cake_dsv4_sparse_metadata(table, lens, query_rows=2)
    separate = resolve_cake_dsv4_sparse_metadata(
        table[:, :128],
        extra_sparse_indices=table[:, 128:],
        extra_sparse_topk_lens=lens - 128,
        query_rows=2,
    )

    def launch(variant, **values):
        cake._launch_variant(variant, arch="sm_103a", grid=(1, 1, 1), values=values)

    with pytest.raises(ValueError, match="retired argument 'completion_base'"):
        launch("retired", completion_base=0)
    with pytest.raises(ValueError, match="'mystery' \\(buffer\\) has no host value"):
        launch("unknown")
    with pytest.raises(ValueError, match="predates the split-table metadata ABI"):
        launch("legacy", sparse_indices=separate.legacy_combined_table)
    launch("legacy", sparse_indices=combined.legacy_combined_table)
    assert recorder.calls[-1][0] is table
    # No host value is None for a real call (dense calls bind seq_lens in place
    # of cum_seq_lens_q); a None value is reported as unavailable.
    with pytest.raises(ValueError, match="not available for this call"):
        launch("ragged_only", cum_seq_lens_q=None)
    with pytest.raises(ValueError, match="TMA descriptor bytes"):
        launch("too_many_descriptors")
    with pytest.raises(TypeError, match="must be an int"):
        launch("retired_int_check") if False else cake._bind_argument(
            {"num_heads": 3.5},
            "parameter",
            "num_heads",
            variant="x",
            grid={},
            descriptor_slab=None,
        )
    with pytest.raises(ValueError, match="three positive ints"):
        cake._launch_variant(
            "legacy", arch="sm_103a", grid=(0, 1, 1), values={"sparse_indices": table}
        )


def _plan(*names):
    return [*names, ("grid", "grid_x"), ("grid", "grid_y"), ("grid", "grid_z")]


_MAIN_PLAN = _plan(
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_kv"),
    ("tma_buffer", "tmap_compressed_kv"),
    ("buffer", "partial_O"),
    ("buffer", "partial_lse"),
    ("buffer", "swa_indices"),
    ("buffer", "compressed_indices"),
    ("buffer", "sparse_topk_lens"),
    ("buffer", "sinks"),
    ("buffer", "bmm1_scale"),
    ("buffer", "bmm2_scale"),
    ("parameter", "num_heads"),
    ("parameter", "swa_index_stride"),
    ("parameter", "compressed_index_stride"),
    ("parameter", "sparse_topk_lens_offset"),
    ("parameter", "sparse_topk"),
    ("parameter", "num_query_tokens"),
    ("parameter", "num_splits"),
    ("parameter", "has_sinks"),
    ("workspace", "tma_descriptor_workspace"),
)
_REDUCE_PLAN = _plan(
    ("buffer", "partial_O"),
    ("buffer", "partial_lse"),
    ("buffer", "O"),
    ("parameter", "num_heads"),
    ("parameter", "num_splits"),
)


def _run_fake_dense_h64(monkeypatch, *, query_rows, metadata, workspace, out=None):
    """Drive run_cake_dsv4 on CPU tensors through the bf16 H64 dense split route."""
    recorder = _install_fake_variants(
        monkeypatch,
        {
            "bf16_h64_compressed_q8_v38": _MAIN_PLAN,
            "bf16_h64_compressed_reduce": _REDUCE_PLAN,
        },
    )
    monkeypatch.setattr(cake, "_target_arch", lambda device: "sm_103a")
    num_heads = 64
    query = torch.zeros((query_rows, num_heads, 512), dtype=torch.bfloat16)
    if out is None:
        out = torch.zeros((query_rows, num_heads, 512), dtype=torch.bfloat16)
    swa_cache = torch.zeros((4, 1, 256, 512), dtype=torch.bfloat16)
    compressed_cache = torch.zeros((8, 1, 64, 512), dtype=torch.bfloat16)
    sinks = torch.zeros((num_heads,), dtype=torch.float32)
    result = cake.run_cake_dsv4(
        query=query,
        swa_kv_cache=swa_cache,
        compressed_kv_cache=compressed_cache,
        workspace_buffer=workspace,
        out=out,
        bmm1_scale=0.5,
        bmm2_scale=1.0,
        sinks=sinks,
        max_q_len=2,
        cum_seq_lens_q=None,
        seq_lens=torch.full((3,), 1000, dtype=torch.int32),
        backend="cake",
        **metadata,
    )
    assert result is out
    return recorder, query, out, swa_cache, compressed_cache, sinks


def test_run_cake_dsv4_binds_uniform_metadata_and_prefix_views(monkeypatch):
    rows, compressed, query_rows = 4, 512, 6
    table, lens = _combined_metadata(rows, compressed)
    num_splits = -(-(128 + compressed) // 128)
    layout = cake_dsv4_workspace_layout(rows, 64, num_splits)
    workspace = _aligned_u8(layout.total_bytes)
    recorder, query, out, swa_cache, compressed_cache, sinks = _run_fake_dense_h64(
        monkeypatch,
        query_rows=query_rows,
        metadata={"sparse_indices": table, "sparse_topk_lens": lens},
        workspace=workspace,
    )
    main, reduce = recorder.calls
    m = dict(zip((name for _, name in _MAIN_PLAN), main, strict=True))
    r = dict(zip((name for _, name in _REDUCE_PLAN), reduce, strict=True))
    # Prefix views: only metadata rows are exposed, no copies.
    assert m["tmap_q"].shape[0] == rows and m["tmap_q"].data_ptr() == query.data_ptr()
    assert r["O"].shape[0] == rows and r["O"].data_ptr() == out.data_ptr()
    assert m["tmap_swa_kv"].data_ptr() == swa_cache.data_ptr()
    assert m["tmap_compressed_kv"].data_ptr() == compressed_cache.data_ptr()
    assert m["tmap_swa_kv"].shape == (4 * 256, 512)
    # Uniform metadata vocabulary.
    assert m["swa_indices"] is table
    assert m["compressed_indices"].data_ptr() == table.data_ptr() + 128 * 4
    assert m["sparse_topk_lens"] is lens
    assert m["swa_index_stride"] == m["compressed_index_stride"] == 128 + compressed
    assert m["sparse_topk_lens_offset"] == 0
    assert m["sparse_topk"] == 128 + compressed
    assert m["num_query_tokens"] == rows
    assert m["num_splits"] == r["num_splits"] == num_splits
    assert m["has_sinks"] == 1 and m["sinks"] is sinks
    assert float(m["bmm1_scale"]) == 0.5 and float(m["bmm2_scale"]) == 1.0
    # Grids derive from metadata rows, not query rows.
    assert (m["grid_x"], m["grid_y"], m["grid_z"]) == (rows * num_splits * 2, 1, 1)
    assert (r["grid_x"], r["grid_y"], r["grid_z"]) == (rows, 64, 1)
    # Private descriptor storage; partial buffers carved at the documented offsets.
    slab = m["tma_descriptor_workspace"]
    assert slab.numel() == cake._DESCRIPTOR_SLAB_BYTES and slab.data_ptr() % 128 == 0
    assert not (
        workspace.data_ptr()
        <= slab.data_ptr()
        < workspace.data_ptr() + workspace.numel()
    )
    assert m["partial_O"].data_ptr() == workspace.data_ptr() + layout.partial_o[0]
    assert m["partial_O"].dtype == torch.bfloat16
    assert m["partial_O"].numel() == rows * 64 * num_splits * 512
    assert m["partial_lse"].data_ptr() == workspace.data_ptr() + layout.partial_lse[0]
    assert m["partial_lse"].dtype == torch.float32
    assert m["partial_lse"].numel() == rows * 64 * num_splits
    assert r["partial_O"] is m["partial_O"] and r["partial_lse"] is m["partial_lse"]


def test_run_cake_dsv4_separate_tables_and_offset(monkeypatch):
    rows, compressed = 3, 512
    table, lens = _combined_metadata(rows, compressed)
    swa = table[:, :128]
    extra = table[:, 128:].clone()
    workspace = _aligned_u8(
        get_cake_dsv4_workspace_bytes(rows, 64, 128 + compressed, torch.bfloat16)
    )
    recorder, *_ = _run_fake_dense_h64(
        monkeypatch,
        query_rows=rows,
        metadata={
            "sparse_indices": swa,
            "sparse_topk_lens": None,
            "extra_sparse_indices": extra,
            "extra_sparse_topk_lens": lens - 128,
            "sparse_topk_lens_offset": -7,
        },
        workspace=workspace,
    )
    m = dict(zip((name for _, name in _MAIN_PLAN), recorder.calls[0], strict=True))
    # The SWA table is a column view of the combined table: the kernel gets the
    # contiguous span with the same base pointer and the combined row stride.
    assert m["swa_indices"].data_ptr() == swa.data_ptr()
    assert m["swa_indices"].is_contiguous() and m["swa_indices"].ndim == 1
    assert m["swa_index_stride"] == 128 + compressed
    assert (
        m["compressed_indices"] is extra and m["compressed_index_stride"] == compressed
    )
    assert m["sparse_topk_lens_offset"] == 128 - 7
    assert m["sparse_topk"] == 128 + compressed
    assert torch.equal(m["sparse_topk_lens"], lens - 128)


def test_run_cake_dsv4_rejects_undersized_workspace_and_copies(monkeypatch):
    rows, compressed = 2, 512
    table, lens = _combined_metadata(rows, compressed)
    small = _aligned_u8(cake._PARTIAL_OFFSET)
    with pytest.raises(ValueError, match="get_cake_dsv4_workspace_bytes"):
        _run_fake_dense_h64(
            monkeypatch,
            query_rows=rows,
            metadata={"sparse_indices": table, "sparse_topk_lens": lens},
            workspace=small,
        )
    workspace = _aligned_u8(
        get_cake_dsv4_workspace_bytes(rows, 64, 128 + compressed, torch.bfloat16)
    )
    with pytest.raises(ValueError, match="unit column stride"):
        _run_fake_dense_h64(
            monkeypatch,
            query_rows=rows,
            metadata={
                "sparse_indices": table[:, ::2][:, : 128 + compressed // 2],
                "sparse_topk_lens": lens,
            },
            workspace=workspace,
        )
    with pytest.raises(ValueError, match="out must be contiguous"):
        _run_fake_dense_h64(
            monkeypatch,
            query_rows=rows,
            metadata={"sparse_indices": table, "sparse_topk_lens": lens},
            workspace=workspace,
            out=torch.zeros((rows, 512, 64), dtype=torch.bfloat16).transpose(1, 2),
        )


def test_workspace_formula_and_layout():
    tokens, heads, topk = 5, 128, 1152
    splits = max(-(-topk // 128), 5)
    expected = (
        1024
        + 256 * 1024
        + -(-(tokens * heads * splits * 512 * 2) // 128) * 128
        + -(-(tokens * heads * splits * 4) // 128) * 128
    )
    assert (
        get_cake_dsv4_workspace_bytes(tokens, heads, topk, torch.bfloat16) == expected
    )
    assert (
        get_cake_dsv4_workspace_bytes(tokens, heads, topk, torch.float8_e4m3fn)
        == expected
    )
    layout = cake_dsv4_workspace_layout(tokens, heads, splits)
    assert layout.descriptor_slab == (0, 1024)
    assert layout.counters == (1024, 256 * 1024)
    assert layout.partial_o[0] == 1024 + 256 * 1024
    assert layout.partial_lse[0] == layout.partial_o[0] + layout.partial_o[1]
    assert layout.total_bytes == expected
    for offset, size in (
        layout.descriptor_slab,
        layout.counters,
        layout.partial_o,
        layout.partial_lse,
    ):
        assert offset % 128 == 0 and size % 128 == 0
    # Default split bound: max(ceil(topk / 128), 5); explicit splits override it.
    assert (
        get_cake_dsv4_workspace_bytes(tokens, heads, 260, torch.bfloat16)
        == cake_dsv4_workspace_layout(tokens, heads, 5).total_bytes
    )
    assert (
        get_cake_dsv4_workspace_bytes(tokens, heads, 260, torch.bfloat16, num_splits=1)
        == cake_dsv4_workspace_layout(tokens, heads, 1).total_bytes
    )
    assert get_cake_dsv4_workspace_bytes(
        1, 8, 128, torch.bfloat16
    ) < get_cake_dsv4_workspace_bytes(2, 8, 128, torch.bfloat16)
    # One split: no partial_O region, the LSE region follows the counters; the
    # optional offsets region of a row-tiled chunk trails the LSE region.
    one = cake_dsv4_workspace_layout(tokens, heads, 1)
    assert one.partial_o == (1024 + 256 * 1024, 0)
    assert one.partial_lse == (1024 + 256 * 1024, -(-(tokens * heads * 4) // 128) * 128)
    assert one.query_offsets == (one.partial_lse[0] + one.partial_lse[1], 0)
    assert one.total_bytes == one.partial_lse[0] + one.partial_lse[1]
    tiled = cake_dsv4_workspace_layout(tokens, heads, 1, num_query_offsets=5)
    assert tiled.query_offsets == (one.total_bytes, 128)
    assert tiled.total_bytes == one.total_bytes + 128
    with pytest.raises(ValueError, match="num_query_offsets"):
        cake_dsv4_workspace_layout(tokens, heads, 1, num_query_offsets=-1)
    with pytest.raises(ValueError, match="dtype"):
        get_cake_dsv4_workspace_bytes(tokens, heads, topk, torch.float16)
    with pytest.raises(ValueError, match="multiple of 4"):
        get_cake_dsv4_workspace_bytes(tokens, heads, 130, torch.bfloat16)
    with pytest.raises(ValueError, match="positive int"):
        cake_dsv4_workspace_layout(0, heads, 1)


def test_workspace_reset_zeroes_only_the_counter_region():
    total = cake._PARTIAL_OFFSET + 4096
    workspace = _aligned_u8(total)
    workspace.fill_(0xFF)
    assert not cake._counters_primed(workspace, cake._workspace_bytes(workspace))
    cake_dsv4_workspace_reset(workspace)
    assert torch.all(workspace[:1024] == 0xFF)
    assert torch.all(workspace[1024 : cake._PARTIAL_OFFSET] == 0)
    assert torch.all(workspace[cake._PARTIAL_OFFSET :] == 0xFF)
    assert cake._counters_primed(workspace, cake._workspace_bytes(workspace))
    # Views of the registered buffer share the registration (same owner).
    view = workspace[:]
    assert cake._counters_primed(view, cake._workspace_bytes(view))
    counters = cake._counters(workspace, 4)
    assert counters.dtype == torch.uint32 and counters.numel() == 4
    assert counters.data_ptr() == workspace.data_ptr() + 1024
    with pytest.raises(ValueError, match="counter region holds"):
        cake._counters(workspace, cake._MAX_MERGE_GROUPS + 1)
    with pytest.raises(ValueError, match="at least"):
        cake_dsv4_workspace_reset(_aligned_u8(1024))
    with pytest.raises(ValueError, match="contiguous"):
        cake_dsv4_workspace_reset(_aligned_u8(2 * total)[::2])
    misaligned = torch.empty(total + 128, dtype=torch.uint8)
    misaligned = misaligned[((-misaligned.data_ptr()) % 128) + 64 :][:total]
    with pytest.raises(ValueError, match="128-byte aligned"):
        cake_dsv4_workspace_reset(misaligned)


def test_counter_zeroing_is_part_of_the_launch_and_never_raises_in_capture(monkeypatch):
    """CAKE-939: counter zeroing is a launch-contract step, not a host priming step.

    Unregistered workspace under capture: a zero fill of exactly the counters
    the launch uses is recorded (executed here), nothing else is touched and
    the workspace stays unregistered (the fill lives in the graph). Eager
    first use zeroes the whole region once and registers the workspace; a
    registered workspace is never filled again, eagerly or under capture.
    """
    workspace = _aligned_u8(cake._PARTIAL_OFFSET)
    workspace.fill_(0xAB)
    launcher = cake._Launcher(
        arch="sm_103a", workspace=workspace, raw=workspace, values={}
    )
    monkeypatch.setattr(cake, "_is_capturing", lambda device: True)
    counters = launcher.counters(8)
    assert counters.numel() == 8 and torch.all(counters == 0)
    assert torch.all(workspace[:1024] == 0xAB)
    assert torch.all(workspace[1024 + 32 : cake._PARTIAL_OFFSET] == 0xAB)
    assert not cake._counters_primed(workspace, workspace)
    monkeypatch.setattr(cake, "_is_capturing", lambda device: False)
    assert torch.all(launcher.counters(3) == 0)
    assert torch.all(workspace[1024 : cake._PARTIAL_OFFSET] == 0)
    assert cake._counters_primed(workspace, workspace)
    workspace[1024:1036].fill_(7)
    assert torch.all(launcher.counters(3) == 0x07070707)  # registered: no second fill
    monkeypatch.setattr(cake, "_is_capturing", lambda device: True)
    assert torch.all(launcher.counters(3) == 0x07070707)  # nor under capture
    # The registration follows the storage owner: an explicit reset registers,
    # a freed owner invalidates the entry and the next registration prunes it.
    other = _aligned_u8(cake._PARTIAL_OFFSET)
    other.fill_(0xAB)
    key = cake._primed_key(other)
    cake_dsv4_workspace_reset(other)
    assert cake._primed_workspaces[key]() is other._base
    assert cake._counters_primed(other, other)
    del other
    import gc

    gc.collect()
    assert cake._primed_workspaces[key]() is None
    cake._register_primed(workspace, workspace)
    assert key not in cake._primed_workspaces


def test_metadata_resolution_combined_table():
    rows, compressed = 5, 132
    table, lens = _combined_metadata(rows, compressed)
    meta = resolve_cake_dsv4_sparse_metadata(table, lens, query_rows=rows)
    assert meta.num_query_tokens == rows
    assert meta.sparse_topk == 128 + compressed and meta.compressed_width == compressed
    assert meta.swa_indices is table
    assert meta.compressed_indices.shape == (rows, compressed)
    assert meta.compressed_indices.data_ptr() == table.data_ptr() + 128 * 4
    assert meta.swa_index_stride == meta.compressed_index_stride == 128 + compressed
    assert meta.sparse_topk_lens_offset == 0 and not meta.separate_tables
    assert meta.legacy_combined_table is table
    assert set(meta.kernel_kwargs()) == set(KERNEL_METADATA_PARAMS)
    # Explicit offset passes through and disables the legacy combined binding.
    shifted = resolve_cake_dsv4_sparse_metadata(
        table, lens - 40, sparse_topk_lens_offset=40, query_rows=rows
    )
    assert (
        shifted.sparse_topk_lens_offset == 40 and shifted.legacy_combined_table is None
    )
    # SWA-only tables alias the compressed view onto the SWA table.
    swa_only, swa_lens = _combined_metadata(rows, 0)
    only = resolve_cake_dsv4_sparse_metadata(swa_only, swa_lens, query_rows=rows)
    assert only.compressed_indices is swa_only and only.sparse_topk == 128
    # Dense [B, Q, cols] / [B, Q] inputs fold without copies.
    folded = resolve_cake_dsv4_sparse_metadata(
        table.reshape(1, rows, -1), lens.reshape(1, rows), query_rows=rows
    )
    assert folded.swa_indices.data_ptr() == table.data_ptr()
    assert folded.sparse_topk_lens.data_ptr() == lens.data_ptr()


def test_metadata_resolution_separate_tables():
    rows, compressed = 4, 256
    table, lens = _combined_metadata(rows, compressed)
    swa = table[:, :128]
    extra = table[:, 128:]
    # Compressed-only lengths imply the 128 offset.
    meta = resolve_cake_dsv4_sparse_metadata(
        swa,
        extra_sparse_indices=extra,
        extra_sparse_topk_lens=lens - 128,
        query_rows=rows,
    )
    assert meta.separate_tables and meta.legacy_combined_table is None
    assert meta.swa_indices is swa and meta.compressed_indices is extra
    assert (
        meta.swa_index_stride == 128 + compressed
        and meta.compressed_index_stride == 128 + compressed
    )
    assert meta.sparse_topk_lens_offset == 128 and meta.sparse_topk == 128 + compressed
    assert meta.sparse_topk_lens is not None and torch.equal(
        meta.sparse_topk_lens, lens - 128
    )
    # Combined-convention lengths with separate tables keep offset 0 (+ explicit).
    plain = resolve_cake_dsv4_sparse_metadata(
        swa,
        lens,
        extra_sparse_indices=extra.contiguous(),
        sparse_topk_lens_offset=3,
        query_rows=rows,
    )
    assert (
        plain.sparse_topk_lens_offset == 3
        and plain.compressed_index_stride == compressed
    )
    # Zero-width compressed table aliases the SWA table.
    zero = resolve_cake_dsv4_sparse_metadata(
        swa, lens, extra_sparse_indices=table[:, 128:128], query_rows=rows
    )
    assert zero.compressed_indices is swa and zero.sparse_topk == 128


def test_metadata_resolution_padded_rows():
    rows, compressed = 3, 4
    table, lens = _combined_metadata(rows, compressed)
    assert (
        resolve_cake_dsv4_sparse_metadata(table, lens, query_rows=10).num_query_tokens
        == rows
    )
    with pytest.raises(
        ValueError, match="metadata has 3 rows but the query only has 2"
    ):
        resolve_cake_dsv4_sparse_metadata(table, lens, query_rows=2)
    with pytest.raises(ValueError, match="at least one query token"):
        resolve_cake_dsv4_sparse_metadata(table[:0], lens[:0], query_rows=2)


def test_metadata_resolution_errors():
    rows, compressed = 3, 8
    table, lens = _combined_metadata(rows, compressed)
    with pytest.raises(ValueError, match="unit column stride"):
        resolve_cake_dsv4_sparse_metadata(
            table[:, ::1][:, :].t().t()[:, ::2].contiguous()[:, ::1]
            if False
            else table.repeat(1, 2)[:, ::2],
            lens,
            query_rows=rows,
        )
    with pytest.raises(ValueError, match="must be int32"):
        resolve_cake_dsv4_sparse_metadata(table.to(torch.int64), lens, query_rows=rows)
    with pytest.raises(ValueError, match="sparse_topk_lens must be int32"):
        resolve_cake_dsv4_sparse_metadata(table, lens.to(torch.int64), query_rows=rows)
    with pytest.raises(ValueError, match="must have 3 entries"):
        resolve_cake_dsv4_sparse_metadata(table, lens[:2], query_rows=rows)
    with pytest.raises(ValueError, match="unit stride"):
        resolve_cake_dsv4_sparse_metadata(table, lens.repeat(2)[::2], query_rows=rows)
    with pytest.raises(ValueError, match="at least 128 columns"):
        resolve_cake_dsv4_sparse_metadata(table[:, :64], lens, query_rows=rows)
    with pytest.raises(ValueError, match="multiple of 4"):
        resolve_cake_dsv4_sparse_metadata(table[:, : 128 + 6], lens, query_rows=rows)
    with pytest.raises(ValueError, match="must have 128 columns"):
        resolve_cake_dsv4_sparse_metadata(
            table, lens, extra_sparse_indices=table[:, 128:], query_rows=rows
        )
    with pytest.raises(ValueError, match="must have 3 rows"):
        resolve_cake_dsv4_sparse_metadata(
            table[:, :128], lens, extra_sparse_indices=table[:2, 128:], query_rows=rows
        )
    with pytest.raises(ValueError, match="not both"):
        resolve_cake_dsv4_sparse_metadata(
            table[:, :128],
            lens,
            extra_sparse_indices=table[:, 128:],
            extra_sparse_topk_lens=lens,
            query_rows=rows,
        )
    with pytest.raises(ValueError, match="requires extra_sparse_indices"):
        resolve_cake_dsv4_sparse_metadata(
            table, extra_sparse_topk_lens=lens, query_rows=rows
        )
    with pytest.raises(ValueError, match="sparse_topk_lens is required"):
        resolve_cake_dsv4_sparse_metadata(table, None, query_rows=rows)
    with pytest.raises(TypeError, match="sparse_topk_lens_offset"):
        resolve_cake_dsv4_sparse_metadata(
            table, lens, sparse_topk_lens_offset=1.5, query_rows=rows
        )
    with pytest.raises(ValueError, match="densely packed"):
        resolve_cake_dsv4_sparse_metadata(
            table.reshape(1, rows, -1).transpose(0, 1).expand(rows, 2, -1),
            lens,
            query_rows=rows,
        )


def test_kernel_kwargs_hand_over_contiguous_index_spans():
    rows, compressed = 5, 132
    table, lens = _combined_metadata(rows, compressed)
    kw = resolve_cake_dsv4_sparse_metadata(table, lens, query_rows=rows).kernel_kwargs()
    # Combined table: the SWA table is the contiguous table itself; the compressed
    # column view is not contiguous (the generated bindings reject it) and becomes
    # the span from its first to its last element with the same base pointer.
    assert kw["swa_indices"] is table
    span = kw["compressed_indices"]
    assert span.is_contiguous() and span.ndim == 1
    assert span.data_ptr() == table.data_ptr() + 128 * 4
    assert span.numel() == (rows - 1) * (128 + compressed) + compressed
    assert kw["compressed_index_stride"] == 128 + compressed
    assert torch.equal(span[:compressed], table[0, 128:])
    assert torch.equal(span[(rows - 1) * (128 + compressed) :], table[rows - 1, 128:])
    # Separate contiguous tables pass through by identity.
    swa, extra = table[:, :128].clone(), table[:, 128:].clone()
    sep = resolve_cake_dsv4_sparse_metadata(
        swa,
        extra_sparse_indices=extra,
        extra_sparse_topk_lens=lens - 128,
        query_rows=rows,
    ).kernel_kwargs()
    assert sep["swa_indices"] is swa and sep["compressed_indices"] is extra


_PERSISTENT_PLAN = _plan(
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_kv"),
    ("tma_buffer", "tmap_compressed_kv"),
    ("buffer", "O"),
    ("buffer", "partial_lse"),
    ("buffer", "swa_indices"),
    ("buffer", "compressed_indices"),
    ("buffer", "sparse_topk_lens"),
    ("buffer", "seq_lens"),
    ("buffer", "cum_seq_lens_q"),
    ("buffer", "sinks"),
    ("buffer", "bmm1_scale"),
    ("buffer", "bmm2_scale"),
    ("parameter", "num_heads"),
    ("parameter", "swa_index_stride"),
    ("parameter", "compressed_index_stride"),
    ("parameter", "sparse_topk_lens_offset"),
    ("parameter", "num_query_tokens"),
    ("parameter", "sparse_topk"),
    ("parameter", "has_sinks"),
    ("parameter", "total_work_items"),
    ("parameter", "max_q_len"),
    ("parameter", "batch_size"),
)


def _run_fake_persistent_fp8_h128(monkeypatch, *, cum_seq_lens_q, max_q_len):
    """Drive run_cake_dsv4 on CPU tensors through the FP8 H128 persistent route."""
    recorder = _install_fake_variants(
        monkeypatch, {"fp8_h128_prefill_source_persistent_uniform": _PERSISTENT_PLAN}
    )
    monkeypatch.setattr(cake, "_target_arch", lambda device: "sm_103a")
    rows, compressed = 6, 512
    table, lens = _combined_metadata(rows, compressed)
    workspace = _aligned_u8(
        get_cake_dsv4_workspace_bytes(rows, 128, 128 + compressed, torch.float8_e4m3fn)
    )
    query = torch.zeros((rows, 128, 512), dtype=torch.float8_e4m3fn)
    out = torch.zeros((rows, 128, 512), dtype=torch.bfloat16)
    cake.run_cake_dsv4(
        query=query,
        swa_kv_cache=torch.zeros((4, 1, 256, 512), dtype=torch.float8_e4m3fn),
        compressed_kv_cache=torch.zeros((8, 1, 64, 512), dtype=torch.float8_e4m3fn),
        workspace_buffer=workspace,
        out=out,
        bmm1_scale=0.5,
        bmm2_scale=1.0,
        sinks=None,
        max_q_len=max_q_len,
        cum_seq_lens_q=cum_seq_lens_q,
        seq_lens=torch.full((3,), 1000, dtype=torch.int32),
        backend="cake",
        sparse_indices=table,
        sparse_topk_lens=lens,
    )
    (call,) = recorder.calls
    return dict(zip((name for _, name in _PERSISTENT_PLAN), call, strict=True))


def test_persistent_route_synthesizes_dense_query_offsets(monkeypatch):
    # Dense [batch=3, q_len=2] query: every request is max_q_len long, so the
    # ragged-only producer receives cum_seq_lens_q = [0, 2, 4, 6] without a
    # per-call allocation (the offsets are a cached process-lifetime constant).
    first = _run_fake_persistent_fp8_h128(monkeypatch, cum_seq_lens_q=None, max_q_len=2)
    offsets = first["cum_seq_lens_q"]
    assert offsets.dtype == torch.int32 and offsets.is_contiguous()
    assert offsets.tolist() == [0, 2, 4, 6]
    assert first["batch_size"] == 3 and first["max_q_len"] == 2
    assert first["total_work_items"] == first["num_query_tokens"] == 6
    assert (first["grid_x"], first["grid_y"], first["grid_z"]) == (12, 1, 1)
    again = _run_fake_persistent_fp8_h128(monkeypatch, cum_seq_lens_q=None, max_q_len=2)
    assert again["cum_seq_lens_q"] is offsets
    # Ragged callers keep their own offsets.
    indptr = torch.tensor([0, 1, 3, 6], dtype=torch.int32)
    ragged = _run_fake_persistent_fp8_h128(
        monkeypatch, cum_seq_lens_q=indptr, max_q_len=3
    )
    assert ragged["cum_seq_lens_q"].data_ptr() == indptr.data_ptr()
    assert torch.equal(ragged["cum_seq_lens_q"], indptr) and ragged["max_q_len"] == 3


class _RecordingLauncher:
    """Minimal stand-in for ``_Launcher`` that records the launched variants."""

    def __init__(self, arch: str, **values):
        self.arch = arch
        self.values = values
        self.calls: list[tuple[str, dict]] = []

    def variant(self, name, *, grid, **overrides):
        return (name, {"grid": grid, **overrides})

    def run(self, *launches):
        # Launches are recorded in the order the dispatcher issues them.
        self.calls.extend(launches)

    def partials(self, num_splits):
        return {
            "partial_O": f"partial_O[{num_splits}]",
            "partial_lse": f"partial_lse[{num_splits}]",
            "num_splits": num_splits,
        }

    def counters(self, merge_groups):
        return f"counters[{merge_groups}]"

    def reduce(self, reducer, **overrides):
        tokens = self.values["num_query_tokens"]
        heads = self.values["num_heads"]
        return self.variant(reducer, grid=(tokens, heads, 1), **overrides)


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize(
    "num_query_tokens,sparse_topk,expected_producer,expected_splits",
    [
        # Three live KV tiles run the four-owner producer (fourth tile fully
        # masked): 14.8 -> 12.7 us on GB300, 15.7 -> 13.6 us on B200.
        (12, 260, "bf16_h128_topk128x_split4_sm100", 4),
        (16, 260, "bf16_h128_topk128x_split4_sm100", 4),
        (12, 388, "bf16_h128_topk128x_split4_sm100", 4),
        (16, 388, "bf16_h128_topk128x_split4_sm100", 4),
        # Above the token bound one row-first owner per token; up to 37
        # tokens the owner runs in the V-half split form (two 2-CTA clusters
        # per token = grid 4 * tokens, identical bits).
        (17, 260, "bf16_h128_topk128x_row_first_vsplit", 1),
        (32, 260, "bf16_h128_topk128x_row_first_vsplit", 1),  # hardening-000025
        (37, 260, "bf16_h128_topk128x_row_first_vsplit", 1),
        (38, 260, "bf16_h128_topk128x_row_first", 1),
        (64, 260, "bf16_h128_topk128x_row_first", 1),  # hardening-000031
        (32, 388, "bf16_h128_topk128x_row_first_vsplit", 1),
        (64, 388, "bf16_h128_topk128x_row_first", 1),
    ],
)
def test_bf16_h128_topk128x_launches_split4_or_row_first_owners(
    arch, num_query_tokens, sparse_topk, expected_producer, expected_splits
):
    L = _RecordingLauncher(
        arch,
        num_query_tokens=num_query_tokens,
        num_heads=128,
        sparse_topk=sparse_topk,
    )
    cake._dispatch_route("bf16_h128_topk128x", L)
    parts = L.partials(expected_splits)
    if expected_splits == 1:
        # The V-half split program runs two 2-CTA clusters per token; the
        # plain row-first owner one.  total_work_items stays the token count.
        ctas_per_token = 4 if expected_producer.endswith("_vsplit") else 2
        assert L.calls == [
            (
                expected_producer,
                {
                    "grid": (ctas_per_token * num_query_tokens, 1, 1),
                    "total_work_items": num_query_tokens,
                    **parts,
                },
            )
        ]
        return
    work_items = num_query_tokens * expected_splits
    assert L.calls == [
        (
            expected_producer,
            {
                "grid": (2 * work_items, 1, 1),
                "total_work_items": work_items,
                **parts,
                "O": parts["partial_O"],
            },
        ),
        ("split_reduce", {"grid": (num_query_tokens, 128, 1), **parts}),
    ]


@pytest.mark.parametrize("arch", _ARCHES)
def test_bf16_h128_topk4x_launches_five_owners_and_the_split5_reducer(arch):
    tokens = 12
    L = _RecordingLauncher(
        arch, num_query_tokens=tokens, num_heads=128, sparse_topk=1152
    )
    cake._dispatch_route("bf16_h128_topk4x_v52", L)
    parts = L.partials(5)
    assert L.calls == [
        (
            "bf16_h128_topk4x_v52",
            {
                "grid": (2 * tokens * 5, 1, 1),
                "total_work_items": tokens * 5,
                **parts,
                "O": parts["partial_O"],
            },
        ),
        # The sm_103a split-5 reducer launches one CTA per four heads, the
        # sm_100a one per head (their former family libraries' grids).
        (
            "bf16_h128_split5_reduce",
            {"grid": (tokens, 32 if arch == "sm_103a" else 128, 1), **parts},
        ),
    ]


_SWEEP_WIDTHS = (
    (128, 1),
    (192, 64),
    (256, 64),
    (260, 2),
    (384, 64),
    (388, 2),
    (512, 64),
    (640, 64),
    (1152, 64),
)
# (batch_size, max_q_len, ragged): decode, prefill, dense and ragged MTP rows.
_SWEEP_QUERIES = (
    (3, 5, True),
    (2, 257, True),
    (8, 8, True),
    (1, 64, False),
    (4, 128, False),
    (1, 512, False),
)


@pytest.mark.parametrize("arch", _ARCHES)
def test_every_route_result_has_a_registered_kernel(monkeypatch, arch):
    """``_route`` rejects a shape up front or dispatches only registered kernels."""
    monkeypatch.setattr(cake, "_bf16_h128_prefill_num_clusters", lambda device: 74)
    registered = set(_ARCH_REGISTRATIONS[arch]["variants"])
    routed = rejected = 0
    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
        for num_heads in (8, 16, 32, 64, 128):
            for sparse_topk, page_size in _SWEEP_WIDTHS:
                for batch_size, max_q_len, ragged in _SWEEP_QUERIES:
                    tokens = _canonical_query_tokens(batch_size, max_q_len, ragged)
                    try:
                        route = _route(
                            dtype=dtype,
                            num_heads=num_heads,
                            max_q_len=max_q_len,
                            ragged=ragged,
                            sparse_topk=sparse_topk,
                            batch_size=batch_size,
                            compressed_page_size=page_size,
                            num_query_tokens=tokens,
                        )
                    except ValueError:
                        rejected += 1
                        continue
                    L = _RecordingLauncher(
                        arch,
                        Q=torch.empty(0),
                        num_query_tokens=tokens,
                        num_heads=num_heads,
                        sparse_topk=sparse_topk,
                        max_q_len=max_q_len,
                        batch_size=batch_size,
                    )
                    cake._dispatch_route(route, L)
                    launched = [name for name, _ in L.calls]
                    assert launched, (arch, route)
                    missing = [name for name in launched if name not in registered]
                    assert missing == [], (arch, route, missing)
                    routed += 1
    assert routed > 0 and rejected > 0


@pytest.mark.parametrize("clusters", [74, 76])
@pytest.mark.parametrize(
    "num_query_tokens,sparse_topk,expected",
    [
        (512, 640, True),  # hardening-000037
        (128, 640, True),  # hardening-000027
        (256, 640, False),  # hardening-000033
        (386, 1152, False),  # prefill-style-000088/92
        (64, 640, False),  # T <= clusters: one item per cluster
        (512, 128, False),  # one KV tile per item (SWA-only)
        (128, 128, False),
        (12, 1152, False),
    ],
)
def test_bf16_h128_prefill_snake_feed_predicate(
    clusters, num_query_tokens, sparse_topk, expected
):
    # Mirrors the Cake seed's bf16_h128_prefill_uses_snake_feed.
    assert (
        cake._bf16_h128_prefill_uses_snake_feed(num_query_tokens, sparse_topk, clusters)
        is expected
    )


@pytest.mark.parametrize("clusters", [74, 76])
def test_bf16_h128_prefill_snake_feed_predicate_full_rounds(clusters):
    assert cake._bf16_h128_prefill_uses_snake_feed(8 * clusters, 640, clusters) is False


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize(
    "num_query_tokens,sparse_topk,expected_program",
    [
        (512, 640, "bf16_h128_prefill_v42_snake"),  # hardening-000037
        (128, 640, "bf16_h128_prefill_v42_snake"),  # hardening-000027
        (256, 640, "bf16_h128_prefill_v42"),  # hardening-000033
        (512, 128, "bf16_h128_prefill_v42"),  # hardening-000035 (SWA-only)
        (386, 1152, "bf16_h128_prefill_v42"),  # prefill-style-000088/92
    ],
)
def test_bf16_h128_prefill_launches_the_snake_body_for_tail_majority_rows(
    monkeypatch, arch, num_query_tokens, sparse_topk, expected_program
):
    # The route id stays bf16_h128_prefill_v42; only the launched body changes.
    # 74 clusters = B200 (148 SMs); the persistent grid is one two-CTA cluster
    # per item, at most one cluster per SM pair.
    monkeypatch.setattr(cake, "_bf16_h128_prefill_num_clusters", lambda device: 74)
    L = _RecordingLauncher(
        arch,
        Q=torch.empty(0),
        num_query_tokens=num_query_tokens,
        num_heads=128,
        sparse_topk=sparse_topk,
    )
    cake._dispatch_route("bf16_h128_prefill_v42", L)
    assert L.calls == [
        (
            expected_program,
            {
                "grid": (min(2 * num_query_tokens, 148), 1, 1),
                "total_work_items": num_query_tokens,
                **L.partials(1),
            },
        )
    ]


@pytest.mark.parametrize(
    "num_query_tokens,page_size,sparse_topk,expected",
    [
        (
            12,
            64,
            640,
            "fp8_h128_prefill_source_persistent_uniform",
        ),  # W5: 12 tokens, 5 tiles
        (12, 2, 260, "fp8_lowhead_h64"),  # W5: 12 tokens, 2 tiles keep the cluster body
        (16, 64, 640, "fp8_h128_prefill_source_persistent_uniform"),  # hardening-000018
        (64, 64, 640, "fp8_h128_prefill_source_persistent_uniform"),  # hardening-000030
        (
            128,
            2,
            260,
            "fp8_h64_prefill_source_persistent_m64_multi_tile",
        ),  # hardening-000022 (W14; multi-tile program)
        (
            512,
            2,
            260,
            "fp8_h64_prefill_source_persistent_m64_multi_tile",
        ),  # hardening-000034 (W14; multi-tile program)
        (
            64,
            None,
            128,
            "fp8_lowhead_prefill",
        ),  # hardening-000020: SWA producer < 128 tokens
        (127, None, 128, "fp8_lowhead_prefill"),  # SWA producer below 128 tokens
        (
            128,
            None,
            128,
            "fp8_h64_prefill_source_persistent_m64",
        ),  # hardening-000026 (W14)
        (
            256,
            None,
            128,
            "fp8_h64_prefill_source_persistent_m64",
        ),  # hardening-000032 (W14)
    ],
)
def test_fp8_h64_rows_follow_the_persistent_body_rule(
    num_query_tokens, page_size, sparse_topk, expected
):
    assert (
        _route(
            dtype=torch.float8_e4m3fn,
            num_heads=64,
            batch_size=8,
            max_q_len=8,
            ragged=True,
            sparse_topk=sparse_topk,
            compressed_page_size=page_size,
            num_query_tokens=num_query_tokens,
        )
        == expected
    )


@pytest.mark.parametrize("num_query_tokens", [128, 256, 512])
@pytest.mark.parametrize(
    "sparse_topk,page_size,expected",
    [
        # The M64 body is two exported programs selected by the item width:
        # the box-K-gather program for single-tile items (the SWA tile is the
        # whole item: hardening-000026 / 000032), the program without the box
        # block for every wider item (hardening-000022 / 000028 / 000034 /
        # 000038).  Same ABI, same bits.
        (128, None, "fp8_h64_prefill_source_persistent_m64"),
        (260, 2, "fp8_h64_prefill_source_persistent_m64_multi_tile"),
        (388, 2, "fp8_h64_prefill_source_persistent_m64_multi_tile"),
        (640, 64, "fp8_h64_prefill_source_persistent_m64_multi_tile"),
        (1152, 64, "fp8_h64_prefill_source_persistent_m64_multi_tile"),
    ],
)
def test_fp8_h64_m64_program_is_selected_by_the_item_width(
    num_query_tokens, sparse_topk, page_size, expected
):
    assert (
        _route(
            dtype=torch.float8_e4m3fn,
            num_heads=64,
            batch_size=8,
            max_q_len=8,
            ragged=True,
            sparse_topk=sparse_topk,
            compressed_page_size=page_size,
            num_query_tokens=num_query_tokens,
        )
        == expected
    )
    # Both M64 programs are ragged-only persistent routes with one CTA per token.
    assert expected in cake._RAGGED_ONLY_ROUTES


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize(
    "route",
    [
        "fp8_h64_prefill_source_persistent_m64",
        "fp8_h64_prefill_source_persistent_m64_multi_tile",
    ],
)
def test_fp8_h64_m64_programs_launch_one_cta_per_token(arch, route):
    tokens = 192
    L = _RecordingLauncher(arch, num_query_tokens=tokens, num_heads=64, sparse_topk=388)
    cake._dispatch_route(route, L)
    assert L.calls == [
        (route, {"grid": (tokens, 1, 1), "total_work_items": tokens, **L.partials(1)})
    ]


# --------------------------------------------------------------------------- CAKE-957 / CAKE-939: query layout, route plan, row tiling
_QUERY_LAYOUT_PLAN = (
    ("buffer", "seq_lens"),
    ("buffer", "cum_seq_lens_q"),
    ("parameter", "ragged_query"),
    ("parameter", "max_q_len"),
    ("parameter", "batch_size"),
)
# The bf16 H32 merge producer (one KV tile per split, in-kernel last-arriver
# merge) with the five query-layout parameters of the regenerated bindings.
_H32_MERGE_PLAN = _plan(
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_kv"),
    ("tma_buffer", "tmap_compressed_kv"),
    ("buffer", "partial_O"),
    ("buffer", "partial_lse"),
    ("buffer", "O"),
    ("buffer", "partition_arrivals"),
    ("buffer", "swa_indices"),
    ("buffer", "compressed_indices"),
    ("buffer", "sparse_topk_lens"),
    *_QUERY_LAYOUT_PLAN,
    ("buffer", "sinks"),
    ("buffer", "bmm1_scale"),
    ("buffer", "bmm2_scale"),
    ("parameter", "num_heads"),
    ("parameter", "swa_index_stride"),
    ("parameter", "compressed_index_stride"),
    ("parameter", "sparse_topk_lens_offset"),
    ("parameter", "sparse_topk"),
    ("parameter", "num_splits"),
    ("parameter", "num_head_tiles"),
    ("parameter", "num_query_tokens"),
    ("parameter", "has_sinks"),
)


def _run_fake_bf16_h32(
    monkeypatch,
    *,
    rows,
    compressed,
    workspace,
    cum_seq_lens_q,
    max_q_len,
    seq_lens,
    query_rows=None,
):
    """Drive run_cake_dsv4 on CPU tensors through bf16_h32_topk128x_early_v47
    (page-2 compressed cache: the counter route) and return the bound calls."""
    recorder = _install_fake_variants(
        monkeypatch, {"bf16_h32_topk128x_early_v47": _H32_MERGE_PLAN}
    )
    monkeypatch.setattr(cake, "_target_arch", lambda device: "sm_103a")
    table, lens = _combined_metadata(rows, compressed)
    query = torch.zeros((query_rows or rows, 32, 512), dtype=torch.bfloat16)
    out = torch.zeros((query_rows or rows, 32, 512), dtype=torch.bfloat16)
    cake.run_cake_dsv4(
        query=query,
        swa_kv_cache=torch.zeros((4, 1, 256, 512), dtype=torch.bfloat16),
        compressed_kv_cache=torch.zeros((8, 1, 2, 512), dtype=torch.bfloat16),
        workspace_buffer=workspace,
        out=out,
        bmm1_scale=0.5,
        bmm2_scale=1.0,
        sinks=None,
        max_q_len=max_q_len,
        cum_seq_lens_q=cum_seq_lens_q,
        seq_lens=seq_lens,
        backend="cake",
        sparse_indices=table,
        sparse_topk_lens=lens,
    )
    calls = [
        dict(zip((name for _, name in _H32_MERGE_PLAN), call, strict=True))
        for call in recorder.calls
    ]
    return calls, table, lens, query, out


def test_query_layout_params_are_bindable_and_bound_for_dense_and_ragged_calls(
    monkeypatch,
):
    """Every variant can bind the five query-layout parameters; dense calls bind
    seq_lens for the never-read cum_seq_lens_q, ragged_query 0, the dense
    per-request length and batch_size = seq_lens.numel()."""
    for kind, name in _QUERY_LAYOUT_PLAN:
        assert cake.is_bindable_arg(kind, name), name
    assert tuple(name for _, name in _QUERY_LAYOUT_PLAN) == cake.QUERY_LAYOUT_PARAMS
    seq_lens = torch.tensor([300, 40, 7], dtype=torch.int32)
    workspace = _aligned_u8(
        get_cake_dsv4_workspace_bytes(6, 32, 128 + 132, torch.bfloat16)
    )
    (dense,), *_ = _run_fake_bf16_h32(
        monkeypatch,
        rows=6,
        compressed=132,
        workspace=workspace,
        cum_seq_lens_q=None,
        max_q_len=2,
        seq_lens=seq_lens,
    )
    assert dense["seq_lens"] is seq_lens
    assert dense["cum_seq_lens_q"] is seq_lens
    assert dense["ragged_query"] == 0
    assert dense["max_q_len"] == 2 and dense["batch_size"] == 3
    indptr = torch.tensor([0, 1, 3, 6], dtype=torch.int32)
    (ragged,), *_ = _run_fake_bf16_h32(
        monkeypatch,
        rows=6,
        compressed=132,
        workspace=workspace,
        cum_seq_lens_q=indptr,
        max_q_len=3,
        seq_lens=seq_lens,
    )
    assert ragged["cum_seq_lens_q"].data_ptr() == indptr.data_ptr()
    assert ragged["ragged_query"] == 1
    assert ragged["max_q_len"] == 3 and ragged["batch_size"] == 3
    # A dense layout that cannot hold the metadata rows is rejected up front.
    with pytest.raises(ValueError, match="dense query layout"):
        _run_fake_bf16_h32(
            monkeypatch,
            rows=7,
            compressed=132,
            workspace=_aligned_u8(
                get_cake_dsv4_workspace_bytes(7, 32, 128 + 132, torch.bfloat16)
            ),
            cum_seq_lens_q=None,
            max_q_len=2,
            seq_lens=seq_lens,
        )


def test_run_cake_dsv4_tiles_rows_to_the_workspace(monkeypatch):
    """CAKE-939: a workspace below the single-launch carve tiles the metadata
    rows; every chunk sees row views, its own partials and counters, and the
    chunks after the first run as ragged rows of the shifted request offsets
    written into the workspace."""
    rows, compressed = 10, 132  # sparse_topk 260: three KV tiles, four counters per row
    seq_lens = torch.tensor([300, 40, 7], dtype=torch.int32)
    indptr = torch.tensor([0, 3, 7, 10], dtype=torch.int32)
    plan = cake._route_plan(
        "bf16_h32_topk128x_early_v47",
        num_query_tokens=rows,
        num_heads=32,
        sparse_topk=260,
    )
    assert plan == cake._RoutePlan(3, True, 4)
    chunk_bytes = cake._launch_workspace_bytes(plan, 4, 32, len(indptr))
    assert (
        cake._rows_per_launch(
            plan,
            num_query_tokens=rows,
            num_heads=32,
            batch_size=3,
            workspace_bytes=chunk_bytes,
        )
        == 4
    )
    workspace = _aligned_u8(chunk_bytes)
    calls, table, lens, query, out = _run_fake_bf16_h32(
        monkeypatch,
        rows=rows,
        compressed=compressed,
        workspace=workspace,
        cum_seq_lens_q=indptr,
        max_q_len=4,
        seq_lens=seq_lens,
    )
    assert [c["num_query_tokens"] for c in calls] == [4, 4, 2]
    row_bytes = 32 * 512 * 2
    for first, c in zip((0, 4, 8), calls, strict=True):
        n = c["num_query_tokens"]
        assert c["tmap_q"].data_ptr() == query.data_ptr() + first * row_bytes
        assert c["tmap_q"].shape[0] == n and c["O"].shape[0] == n
        assert c["O"].data_ptr() == out.data_ptr() + first * row_bytes
        assert c["swa_indices"].data_ptr() == table.data_ptr() + first * 260 * 4
        assert (
            c["compressed_indices"].data_ptr()
            == table.data_ptr() + (first * 260 + 128) * 4
        )
        assert c["swa_index_stride"] == c["compressed_index_stride"] == 260
        assert c["sparse_topk_lens"].data_ptr() == lens.data_ptr() + first * 4
        assert c["sparse_topk_lens"].numel() == n
        assert c["num_splits"] == 3 and c["num_head_tiles"] == 4
        assert c["partition_arrivals"].numel() == n * 4
        assert torch.all(c["partition_arrivals"] == 0)
        assert c["partial_O"].numel() == n * 32 * 3 * 512
        assert c["partial_lse"].numel() == n * 32 * 3
        assert (c["grid_x"], c["grid_y"], c["grid_z"]) == (n * 3 * 4, 1, 1)
        assert c["seq_lens"] is seq_lens
        assert c["batch_size"] == 3 and c["max_q_len"] == 4 and c["ragged_query"] == 1
        if first == 0:
            assert c["cum_seq_lens_q"].data_ptr() == indptr.data_ptr()
        else:
            layout = cake_dsv4_workspace_layout(n, 32, 3, num_query_offsets=4)
            assert layout.total_bytes <= workspace.numel()
            assert (
                c["cum_seq_lens_q"].data_ptr()
                == workspace.data_ptr() + layout.query_offsets[0]
            )
            assert c["cum_seq_lens_q"].dtype == torch.int32
            assert c["cum_seq_lens_q"].tolist() == (indptr - first).tolist()
    # Dense caller: the first chunk keeps the dense layout, the later chunks
    # run as ragged rows of the (cached) dense request offsets minus their start.
    seq_lens5 = torch.full((5,), 100, dtype=torch.int32)
    calls, *_ = _run_fake_bf16_h32(
        monkeypatch,
        rows=rows,
        compressed=compressed,
        workspace=_aligned_u8(cake._launch_workspace_bytes(plan, 4, 32, 6)),
        cum_seq_lens_q=None,
        max_q_len=2,
        seq_lens=seq_lens5,
    )
    assert [c["num_query_tokens"] for c in calls] == [4, 4, 2]
    assert calls[0]["cum_seq_lens_q"] is seq_lens5 and calls[0]["ragged_query"] == 0
    assert calls[1]["ragged_query"] == 1
    assert calls[1]["cum_seq_lens_q"].tolist() == [-4, -2, 0, 2, 4, 6]
    assert calls[2]["ragged_query"] == 1
    assert calls[2]["cum_seq_lens_q"].tolist() == [-8, -6, -4, -2, 0, 2]
    # A workspace that holds one launch is not tiled.
    calls, *_ = _run_fake_bf16_h32(
        monkeypatch,
        rows=rows,
        compressed=compressed,
        workspace=_aligned_u8(
            get_cake_dsv4_workspace_bytes(rows, 32, 260, torch.bfloat16)
        ),
        cum_seq_lens_q=indptr,
        max_q_len=4,
        seq_lens=seq_lens,
    )
    assert len(calls) == 1 and calls[0]["num_query_tokens"] == rows
    # Below the one-row carve the call is rejected and names the minimum.
    with pytest.raises(ValueError, match="needs at least"):
        _run_fake_bf16_h32(
            monkeypatch,
            rows=rows,
            compressed=compressed,
            workspace=_aligned_u8(cake._launch_workspace_bytes(plan, 1, 32, 4) - 128),
            cum_seq_lens_q=indptr,
            max_q_len=4,
            seq_lens=seq_lens,
        )


def test_workspace_requirement_is_exact_per_route():
    mib128 = 128 * 1024 * 1024
    # Routes without partial buffers need no workspace bytes.
    swa = cake.cake_dsv4_workspace_requirement(
        dtype=torch.bfloat16,
        num_heads=64,
        num_query_tokens=12,
        sparse_topk=128,
        compressed_page_size=256,
        max_q_len=5,
        batch_size=3,
        ragged=True,
    )
    assert swa.route == "bf16_swa128_single_cta" and not swa.uses_workspace
    assert swa.single_launch_bytes == swa.minimum_bytes == 0
    assert swa.rows_per_launch(0) == 12
    # One-split persistent route: the LSE region only, so 1024 dense FP8/H128
    # rows run inside sglang's fixed 128 MiB buffer in one launch (the former
    # carve asked for a 128 MiB partial_O region it never used).
    fp8 = cake.cake_dsv4_workspace_requirement(
        dtype=torch.float8_e4m3fn,
        num_heads=128,
        num_query_tokens=1024,
        sparse_topk=128,
        compressed_page_size=256,
        max_q_len=1,
        batch_size=1024,
        ragged=False,
    )
    assert fp8.route == "fp8_h128_prefill_source_persistent" and fp8.num_splits == 1
    assert fp8.single_launch_bytes == cake._PARTIAL_OFFSET + 1024 * 128 * 4
    assert fp8.rows_per_launch(mib128) == 1024
    # One row per launch: 128 heads x 4 B of LSE plus the 1025 shifted offsets.
    assert fp8.minimum_bytes == cake._PARTIAL_OFFSET + 512 + -(-(1025 * 4) // 128) * 128
    # The route-agnostic single-launch bound stays conservative (five splits).
    assert get_cake_dsv4_workspace_bytes(1024, 128, 128, torch.float8_e4m3fn) > mib128
    # The split route without a token bound tiles inside 128 MiB; the counter
    # region caps one launch at 16384 rows of four counters.
    h32 = cake.cake_dsv4_workspace_requirement(
        dtype=torch.bfloat16,
        num_heads=32,
        num_query_tokens=4096,
        sparse_topk=260,
        compressed_page_size=2,
        max_q_len=1,
        batch_size=4096,
        ragged=False,
    )
    assert h32.route == "bf16_h32_topk128x_early_v47" and h32.num_splits == 3
    assert h32.single_launch_bytes > mib128
    rows = h32.rows_per_launch(mib128)
    assert 0 < rows < 4096
    assert cake._launch_workspace_bytes(h32.plan, rows, 32, 4097) <= mib128
    assert cake._launch_workspace_bytes(h32.plan, rows + 1, 32, 4097) > mib128
    assert h32.minimum_bytes == cake._launch_workspace_bytes(h32.plan, 1, 32, 4097)
    assert h32.rows_per_launch(h32.minimum_bytes) == 1
    assert h32.rows_per_launch(h32.minimum_bytes - 1) == 0
    wide = cake.cake_dsv4_workspace_requirement(
        dtype=torch.bfloat16,
        num_heads=32,
        num_query_tokens=20000,
        sparse_topk=260,
        compressed_page_size=2,
        max_q_len=1,
        batch_size=20000,
        ragged=False,
    )
    assert wide.rows_per_launch(1 << 40) == cake._MAX_MERGE_GROUPS // 4
    # Shapes without a kernel are rejected like the launch would reject them.
    with pytest.raises(ValueError, match="no BF16 kernel"):
        cake.cake_dsv4_workspace_requirement(
            dtype=torch.bfloat16,
            num_heads=16,
            num_query_tokens=4,
            sparse_topk=260,
            compressed_page_size=2,
            max_q_len=1,
            batch_size=4,
            ragged=False,
        )


@pytest.mark.parametrize("arch", _ARCHES)
def test_route_plan_matches_the_dispatcher(monkeypatch, arch):
    """The sizing table and the launches agree on splits, partial buffers and counters."""
    monkeypatch.setattr(cake, "_bf16_h128_prefill_num_clusters", lambda device: 74)
    checked = 0
    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
        for num_heads in (8, 16, 32, 64, 128):
            for sparse_topk, page_size in _SWEEP_WIDTHS:
                for batch_size, max_q_len, ragged in _SWEEP_QUERIES:
                    tokens = _canonical_query_tokens(batch_size, max_q_len, ragged)
                    try:
                        route = _route(
                            dtype=dtype,
                            num_heads=num_heads,
                            max_q_len=max_q_len,
                            ragged=ragged,
                            sparse_topk=sparse_topk,
                            batch_size=batch_size,
                            compressed_page_size=page_size,
                            num_query_tokens=tokens,
                        )
                    except ValueError:
                        continue
                    plan = cake._route_plan(
                        route,
                        num_query_tokens=tokens,
                        num_heads=num_heads,
                        sparse_topk=sparse_topk,
                    )
                    L = _RecordingLauncher(
                        arch,
                        Q=torch.empty(0),
                        num_query_tokens=tokens,
                        num_heads=num_heads,
                        sparse_topk=sparse_topk,
                        max_q_len=max_q_len,
                        batch_size=batch_size,
                    )
                    cake._dispatch_route(route, L)
                    overrides = [kw for _, kw in L.calls]
                    if plan.uses_partials:
                        assert overrides[0]["num_splits"] == plan.num_splits, (
                            route,
                            tokens,
                        )
                        assert (
                            overrides[0]["partial_lse"]
                            == f"partial_lse[{plan.num_splits}]"
                        )
                    else:
                        assert all(
                            "partial_O" not in kw and "partial_lse" not in kw
                            for kw in overrides
                        ), route
                    if plan.merge_groups_per_row:
                        assert overrides[0]["partition_arrivals"] == (
                            f"counters[{tokens * plan.merge_groups_per_row}]"
                        ), route
                    else:
                        assert all(
                            "partition_arrivals" not in kw for kw in overrides
                        ), route
                    checked += 1
    assert checked > 0
