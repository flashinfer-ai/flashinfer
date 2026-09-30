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
        # CAKE-624 W14: FP8/H64 rows admitted to the persistent body with >= 128
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
            "fp8_h64_prefill_source_persistent_m64",
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
        # targets (CAKE-624 W10: the cluster producers sat at 0.19-0.79x
        # vs trtllm-gen from 64 tokens on); from 128 tokens the H64-specific
        # single-CTA M64 program (CAKE-624 W14).
        pytest.param(
            torch.float8_e4m3fn,
            64,
            3,
            64,
            False,
            640,
            64,
            "fp8_h64_prefill_source_persistent_m64",
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
            "fp8_h64_prefill_source_persistent_m64",
            id="h64-w388-192tok",
        ),
        # Off-contract low-head width beyond three sparse tiles keeps the
        # two-partition producer + reducer path.
        pytest.param(
            torch.float8_e4m3fn,
            32,
            3,
            5,
            True,
            640,
            64,
            "fp8_lowhead_split",
            id="lowhead-w640-split",
        ),
    ],
)
def test_cake_dsv4_semantic_routes(
    dtype, num_heads, batch_size, max_q_len, ragged, sparse_topk, page_size, expected
):
    for arch in ("sm_100a", "sm_103a"):
        assert _route(
            arch=arch,
            dtype=dtype,
            num_heads=num_heads,
            batch_size=batch_size,
            max_q_len=max_q_len,
            ragged=ragged,
            sparse_topk=sparse_topk,
            compressed_page_size=page_size,
            num_query_tokens=_canonical_query_tokens(batch_size, max_q_len, ragged),
        ) == (expected[arch] if isinstance(expected, dict) else expected)


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
        (12, 640, 64, "bf16_h128_topk128x"),
        (64, 640, 64, "bf16_h128_prefill_v42"),  # hardening-000021
        (512, 640, 64, "bf16_h128_prefill_v42"),  # hardening-000037
        # topk128x rows keep their own rule (W12), whatever the token count.
        (64, 260, 2, "bf16_h128_topk128x"),
    ],
)
def test_bf16_h128_swa_and_topk4x_rows_use_the_persistent_prefill_body_from_64_tokens(
    num_query_tokens, sparse_topk, page_size, expected
):
    for arch in ("sm_100a", "sm_103a"):
        assert (
            _route(
                arch=arch,
                dtype=torch.bfloat16,
                num_heads=128,
                batch_size=64,
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
    for arch in ("sm_100a", "sm_103a"):
        assert (
            _route(
                arch=arch,
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
    for arch in ("sm_100a", "sm_103a"):
        assert (
            _route(
                arch=arch,
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
    for arch in ("sm_100a", "sm_103a"):
        assert (
            _route(
                arch=arch,
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
_HOST_TMA_SOURCE_TENSORS = frozenset({"Q", "SWA_cache", "compressed_KV_cache", "O"})


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
    """
    pin = "-Xptxas=--register-usage-level=10"
    variants = _ARCH_REGISTRATIONS[arch]["variants"]
    assert "fp8_h128_prefill_source_persistent" in variants
    assert "fp8_h128_prefill_source_persistent_uniform" in variants
    for variant, spec in variants.items():
        flags = set(spec["compile_flags"])
        assert flags <= _PUBLIC_COMPILE_FLAGS, (variant, sorted(flags))
        if variant == "fp8_h128_prefill_source_persistent":
            assert pin in flags, variant
        elif variant == "fp8_h128_prefill_source_persistent_uniform":
            assert pin not in flags, variant


@pytest.mark.parametrize("arch", _ARCHES)
def test_program_signatures_use_known_names(arch):
    unknown = []
    for program_id, spec in _ARCH_REGISTRATIONS[arch]["programs"].items():
        signature = spec["signature"]
        for kind, names in (
            ("buffer", signature["tensor_keys"]),
            ("workspace", signature["workspace_keys"]),
            ("parameter", signature["scalar_names"]),
        ):
            for name in names:
                if not cake.is_bindable_arg(kind, name):
                    unknown.append((program_id, kind, name))
    assert unknown == []


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

    monkeypatch.setattr(jit, "get_cake_dsv4_spec", fake_spec)
    monkeypatch.setattr(cake, "_variant_module", lambda variant, *, arch: recorder)
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
    cake._launch_variant(
        "fake", arch="sm_103a", grid=(7, 2, 1), workspace_raw=raw, values=values
    )
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
    assert slab.data_ptr() == raw.data_ptr()
    assert slab.numel() == cake._DESCRIPTOR_SLAB_BYTES
    assert slab.data_ptr() % 128 == 0
    assert all(isinstance(bound[n], int) for n in ("sparse_topk", "grid_x"))


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
    raw = _aligned_u8(cake._PARTIAL_OFFSET)
    table, lens = _combined_metadata(2, 4)
    combined = resolve_cake_dsv4_sparse_metadata(table, lens, query_rows=2)
    separate = resolve_cake_dsv4_sparse_metadata(
        table[:, :128],
        extra_sparse_indices=table[:, 128:],
        extra_sparse_topk_lens=lens - 128,
        query_rows=2,
    )

    def launch(variant, **values):
        cake._launch_variant(
            variant, arch="sm_103a", grid=(1, 1, 1), workspace_raw=raw, values=values
        )

    with pytest.raises(ValueError, match="retired argument 'completion_base'"):
        launch("retired", completion_base=0)
    with pytest.raises(ValueError, match="'mystery' \\(buffer\\) has no host value"):
        launch("unknown")
    with pytest.raises(ValueError, match="predates the split-table metadata ABI"):
        launch("legacy", sparse_indices=separate.legacy_combined_table)
    launch("legacy", sparse_indices=combined.legacy_combined_table)
    assert recorder.calls[-1][0] is table
    with pytest.raises(ValueError, match="needs ragged queries"):
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
            "legacy",
            arch="sm_103a",
            grid=(0, 1, 1),
            workspace_raw=raw,
            values={"sparse_indices": table},
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
    monkeypatch.setattr(cake, "_stream_ptr", lambda device: 0)
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
    # Workspace carve at the documented offsets.
    assert m["tma_descriptor_workspace"].data_ptr() == workspace.data_ptr()
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
    assert not getattr(workspace, cake._PRIMED_ATTR, False)
    cake_dsv4_workspace_reset(workspace)
    assert torch.all(workspace[:1024] == 0xFF)
    assert torch.all(workspace[1024 : cake._PARTIAL_OFFSET] == 0)
    assert torch.all(workspace[cake._PARTIAL_OFFSET :] == 0xFF)
    assert getattr(workspace, cake._PRIMED_ATTR)
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


def test_counter_initialisation_refused_during_capture(monkeypatch):
    workspace = _aligned_u8(cake._PARTIAL_OFFSET)
    workspace.fill_(0xAB)
    launcher = cake._Launcher(
        arch="sm_103a", workspace=workspace, raw=workspace, stream=0, values={}
    )
    monkeypatch.setattr(cake, "_is_capturing", lambda device: True)
    with pytest.raises(RuntimeError, match="cake_dsv4_workspace_reset"):
        launcher.counters(8)
    assert torch.all(workspace[1024:1032] == 0xAB)  # nothing touched during capture
    cake_dsv4_workspace_reset(workspace)
    counters = launcher.counters(8)
    assert torch.all(counters == 0)
    # Eager first use primes the tensor exactly once.
    eager = cake._Launcher(
        arch="sm_103a",
        workspace=_aligned_u8(cake._PARTIAL_OFFSET),
        raw=None,
        stream=0,
        values={},
    )
    eager.raw = eager.workspace
    eager.workspace.fill_(0xAB)
    monkeypatch.setattr(cake, "_is_capturing", lambda device: False)
    assert torch.all(eager.counters(3) == 0)
    assert getattr(eager.workspace, cake._PRIMED_ATTR)
    eager.workspace[1024:1036].fill_(7)
    assert torch.all(eager.counters(3) == 0x07070707)  # primed: no second reset


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
    monkeypatch.setattr(cake, "_stream_ptr", lambda device: 0)
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


_FP8_SPLIT_PRODUCER_PLAN = _plan(
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_kv"),
    ("tma_buffer", "tmap_compressed_kv"),
    ("buffer", "O"),
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
    ("parameter", "num_query_tokens"),
    ("parameter", "sparse_topk"),
    ("parameter", "has_sinks"),
    ("parameter", "total_work_items"),
)
_FP8_SPLIT_REDUCE_PLAN = _plan(
    ("buffer", "partial_O"),
    ("buffer", "partial_lse"),
    ("buffer", "O"),
    ("parameter", "num_q_heads"),
    ("parameter", "num_split"),
)


def test_fp8_split_producer_writes_partials_through_o(monkeypatch):
    """fp8_lowhead_split's ``O`` is the [tokens, heads, 2, 512] partial buffer."""
    recorder = _install_fake_variants(
        monkeypatch,
        {
            "fp8_lowhead_split": _FP8_SPLIT_PRODUCER_PLAN,
            "split_reduce": _FP8_SPLIT_REDUCE_PLAN,
        },
    )
    monkeypatch.setattr(cake, "_target_arch", lambda device: "sm_103a")
    monkeypatch.setattr(cake, "_stream_ptr", lambda device: 0)
    # Width 128 + 512 = 640 = 5 sparse tiles: beyond the three tiles one
    # partition owns, so the two-partition producer + reducer path is taken.
    rows, compressed, heads = 4, 512, 32
    table, lens = _combined_metadata(rows, compressed)
    workspace = _aligned_u8(
        get_cake_dsv4_workspace_bytes(
            rows, heads, 128 + compressed, torch.float8_e4m3fn
        )
    )
    out = torch.zeros((rows, heads, 512), dtype=torch.bfloat16)
    cake.run_cake_dsv4(
        query=torch.zeros((rows, heads, 512), dtype=torch.float8_e4m3fn),
        swa_kv_cache=torch.zeros((4, 1, 256, 512), dtype=torch.float8_e4m3fn),
        compressed_kv_cache=torch.zeros((8, 1, 64, 512), dtype=torch.float8_e4m3fn),
        workspace_buffer=workspace,
        out=out,
        bmm1_scale=0.5,
        bmm2_scale=1.0,
        sinks=None,
        max_q_len=2,
        cum_seq_lens_q=None,
        seq_lens=torch.full((2,), 1000, dtype=torch.int32),
        backend="cake",
        sparse_indices=table,
        sparse_topk_lens=lens,
    )
    producer, reduce = recorder.calls
    m = dict(zip((name for _, name in _FP8_SPLIT_PRODUCER_PLAN), producer, strict=True))
    r = dict(zip((name for _, name in _FP8_SPLIT_REDUCE_PLAN), reduce, strict=True))
    layout = cake_dsv4_workspace_layout(rows, heads, 2)
    assert m["O"].data_ptr() == workspace.data_ptr() + layout.partial_o[0]
    assert m["O"].numel() == rows * heads * 2 * 512 and m["O"].dtype == torch.bfloat16
    assert m["partial_lse"].data_ptr() == workspace.data_ptr() + layout.partial_lse[0]
    assert m["total_work_items"] == rows * 2
    assert (m["grid_x"], m["grid_y"], m["grid_z"]) == (rows * 2 * 2, 1, 1)
    assert r["partial_O"].data_ptr() == m["O"].data_ptr()
    assert r["partial_lse"].data_ptr() == m["partial_lse"].data_ptr()
    assert r["O"].data_ptr() == out.data_ptr() and r["O"].numel() == out.numel()
    assert r["num_split"] == 2 and r["num_q_heads"] == heads
    assert (r["grid_x"], r["grid_y"], r["grid_z"]) == (rows, heads, 1)


class _RecordingLauncher:
    """Minimal stand-in for ``_Launcher`` that records program/variant selections."""

    def __init__(self, arch: str, **values):
        self.arch = arch
        self.values = values
        self.calls: list[tuple[str, str, dict]] = []

    def variant(self, name, *, grid, **overrides):
        self.calls.append(("variant", name, {"grid": grid, **overrides}))

    def program(self, name, **overrides):
        self.calls.append(("program", name, overrides))

    def partials(self, num_splits):
        return {"num_splits": num_splits}


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
@pytest.mark.parametrize(
    "num_query_tokens,sparse_topk,expected_program,expected_splits",
    [
        # Three live KV tiles run the four-owner program (fourth tile fully
        # masked): 14.8 -> 12.7 us on GB300, 15.7 -> 13.6 us on B200.
        (12, 260, "bf16_h128_topk128x_split4_sm100", 4),
        (16, 260, "bf16_h128_topk128x_split4_sm100", 4),
        (12, 388, "bf16_h128_topk128x_split4_sm100", 4),
        (16, 388, "bf16_h128_topk128x_split4_sm100", 4),
        # CAKE-624 W12: above the token bound one row-first owner per token.
        (17, 260, "bf16_h128_topk128x_row_first", 1),
        (32, 260, "bf16_h128_topk128x_row_first", 1),  # hardening-000025
        (64, 260, "bf16_h128_topk128x_row_first", 1),  # hardening-000031
        (32, 388, "bf16_h128_topk128x_row_first", 1),
        (64, 388, "bf16_h128_topk128x_row_first", 1),
        (12, 640, "bf16_h128_topk128x", 1),
        (64, 512, "bf16_h128_topk128x", 1),
    ],
)
def test_bf16_h128_topk128x_split_programs_dispatch_on_both_targets(
    arch, num_query_tokens, sparse_topk, expected_program, expected_splits
):
    L = _RecordingLauncher(
        arch,
        num_query_tokens=num_query_tokens,
        num_heads=128,
        sparse_topk=sparse_topk,
    )
    cake._dispatch_route("bf16_h128_topk128x", L)
    assert L.calls == [
        (
            "program",
            expected_program,
            {
                "total_work_items": num_query_tokens * expected_splits,
                "num_splits": expected_splits,
            },
        )
    ]


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
    # CAKE-624 W17: mirrors the Cake seed's bf16_h128_prefill_uses_snake_feed.
    assert (
        cake._bf16_h128_prefill_uses_snake_feed(num_query_tokens, sparse_topk, clusters)
        is expected
    )


@pytest.mark.parametrize("clusters", [74, 76])
def test_bf16_h128_prefill_snake_feed_predicate_full_rounds(clusters):
    assert cake._bf16_h128_prefill_uses_snake_feed(8 * clusters, 640, clusters) is False


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
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
def test_bf16_h128_prefill_launches_the_snake_program_for_tail_majority_rows(
    monkeypatch, arch, num_query_tokens, sparse_topk, expected_program
):
    # CAKE-624 W17: the route id stays bf16_h128_prefill_v42; only the launched
    # program alias changes.  74 clusters = B200 (148 SMs).
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
            "program",
            expected_program,
            {"total_work_items": num_query_tokens, "num_splits": 1},
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
            "fp8_h64_prefill_source_persistent_m64",
        ),  # hardening-000022 (W14)
        (
            512,
            2,
            260,
            "fp8_h64_prefill_source_persistent_m64",
        ),  # hardening-000034 (W14)
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
    for arch in ("sm_100a", "sm_103a"):
        assert (
            _route(
                arch=arch,
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
