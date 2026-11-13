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

# The SM count the routing tests assume (B200 / B300).
_SM_COUNT = 148


@pytest.fixture(autouse=True)
def _cpu_sm_count(monkeypatch):
    """The host reads the device's SM count; these CPU tests route as on a 148-SM part."""
    monkeypatch.setattr(cake, "_device_sm_count", lambda device: _SM_COUNT)


# The K=64 and four-stage sibling gates as the host defines them; the tests of
# the gates call them.
_K64_SIBLING_LAUNCHABLE = cake._k64_sibling_launchable
_KV4_SIBLING_LAUNCHABLE = cake._kv4_sibling_launchable


@pytest.fixture(autouse=True)
def _launchable_siblings(monkeypatch):
    """The sibling gates probe the process's CUDA runtime and the device; these
    CPU tests route as in a process that can launch every K=64 and four-stage
    sibling."""
    monkeypatch.setattr(cake, "_k64_sibling_launchable", lambda arch, device: True)
    monkeypatch.setattr(cake, "_kv4_sibling_launchable", lambda arch, device: True)


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


@pytest.mark.cpu_only
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
            torch.float8_e4m3fn,
            64,
            3,
            5,
            True,
            260,
            2,
            "fp8_h64_prefill_source_persistent_m64_multi_tile",
            id="case-08",
        ),
        # FP8/H64 decode rows whose two-CTA-per-token grid exceeds the SM
        # count (2 x 64 and 127 tokens on 148 SMs) run the single-CTA M64
        # program; 74 tokens keep the FP8/H128 persistent program.
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
            "fp8_h64_prefill_source_persistent_m64_multi_tile",
            id="w14-h64-w260-127tok",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            64,
            1,
            74,
            False,
            260,
            2,
            "fp8_h128_prefill_source_persistent_uniform",
            id="w14-h64-w260-74tok",
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
            torch.float8_e4m3fn,
            64,
            3,
            5,
            True,
            260,
            2,
            "fp8_h64_prefill_source_persistent_m64_multi_tile",
            id="case-11",
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
        # Low-head width beyond the three sparse tiles one producer
        # partition owns: the M64 body (no architecture given).
        pytest.param(
            torch.float8_e4m3fn,
            32,
            3,
            5,
            True,
            640,
            64,
            "fp8_h64_prefill_source_persistent_m64_multi_tile",
            id="lowhead-w640-m64",
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
        sm_count=_SM_COUNT,
    )
    if expected is ValueError:
        # Shapes without an exported kernel are rejected before any launch.
        with pytest.raises(ValueError, match="has no .* kernel"):
            _route(**kwargs)
    else:
        assert _route(**kwargs) == expected


@pytest.mark.cpu_only
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
        sm_count=_SM_COUNT,
    )
    if expected is ValueError:
        with pytest.raises(ValueError, match="has no BF16 H128 kernel"):
            _route(**kwargs)
    else:
        assert _route(**kwargs) == expected


@pytest.mark.cpu_only
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
            sm_count=_SM_COUNT,
        )
        == expected
    )


@pytest.mark.cpu_only
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
            sm_count=_SM_COUNT,
        )
        == expected
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "num_heads,batch_size,page_size,expected",
    [
        # FP8 prefill rows of 8, 16, 32 and 64 heads run the single-CTA M64
        # body (heads past num_heads predicated) for every batch size and
        # cache layout: faster than the low-head prefill producer on the
        # contract's 512-8192-token rows (GB300).
        (64, 1, 64, "fp8_h64_prefill_source_persistent_m64_multi_tile"),
        (64, 2, 2, "fp8_h64_prefill_source_persistent_m64_multi_tile"),
        (32, 1, 64, "fp8_h64_prefill_source_persistent_m64_multi_tile"),
        (32, 2, 2, "fp8_h64_prefill_source_persistent_m64_multi_tile"),
        (16, 1, 64, "fp8_h64_prefill_source_persistent_m64_multi_tile"),
        (16, 2, 2, "fp8_h64_prefill_source_persistent_m64_multi_tile"),
        (8, 1, 64, "fp8_h64_prefill_source_persistent_m64_multi_tile"),
        (8, 2, 2, "fp8_h64_prefill_source_persistent_m64_multi_tile"),
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
            sm_count=_SM_COUNT,
        )
        == expected
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_unexported_variant_fails(arch):
    with pytest.raises(ValueError, match="no generated source contract"):
        get_cake_dsv4_spec("unexported_variant", arch=arch)


@pytest.mark.cpu_only
def test_sm107a_registers_the_fp8_decode_route_only():
    """SM107 builds the FP8 programs the decode route picks, each from its own
    sm_107a/ sources, and has no BF16 or NVFP4 variant."""
    from flashinfer.jit.cake_dsv4 import _ARCH_REGISTRATIONS

    assert set(_ARCH_REGISTRATIONS["sm_107a"]["variants"]) == {
        "fp8_h128_prefill_source_persistent",
        "fp8_h128_prefill_source_persistent_uniform",
        "fp8_h128_prefill_source_persistent_uniform_k64",
        "fp8_h128_prefill_source_persistent_uniform_k64_split",
        "fp8_h128_prefill_source_persistent_uniform_k64_split_reduce",
        "fp8_h128_prefill_source_persistent_uniform_k64_split_reduce_c5",
        "fp8_h128_prefill_source_persistent_uniform_k64_split_reduce_c9",
        "fp8_h128_prefill_source_persistent_uniform_k64_split_reduce_s3",
        "fp8_h64_prefill_source_persistent_m64",
        "fp8_h64_prefill_source_persistent_m64_multi_tile",
        "fp8_h64_prefill_source_persistent_m64_multi_tile_kv4",
        "fp8_h64_prefill_source_persistent_m64_multi_tile_kv4_bcl",
        "fp8_h64_prefill_source_persistent_m64_multi_tile_kv4_l2pf",
        "fp8_lowhead_prefill",
        "fp8_lowhead_prefill_overlap",
        "fp8_lowhead_prefill_vblock",
    }
    for spec in _ARCH_REGISTRATIONS["sm_107a"]["variants"].values():
        assert all(source.startswith("sm_107a/") for source in spec["sources"])
        assert ("launch", "enable_pdl") in spec["arg_plan"]
    with pytest.raises(ValueError, match="SM107 runs the FP8 decode route"):
        get_cake_dsv4_spec("bf16_h128_prefill_v42", arch="sm_107a")


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_registered_sources_exist_and_shared_sources_live_in_common(arch):
    """Every registered source is on disk; a source both architectures compile
    lives under ``common/`` and every ``common/`` source is compiled by both.
    A variant one architecture alone registers (the NVFP4 H128 ``v_unit_pairs``
    build, sm_103a) has its sources in that architecture's directory.
    """
    from flashinfer.jit.cake_dsv4 import _get_csrc_dir

    csrc_dir = _get_csrc_dir()
    other = "sm_103a" if arch == "sm_100a" else "sm_100a"
    for variant, spec in _ARCH_REGISTRATIONS[arch]["variants"].items():
        twin_spec = _ARCH_REGISTRATIONS[other]["variants"].get(variant)
        if twin_spec is None:
            assert variant == "nvfp4_h128_v_unit_pairs_prefill", (arch, variant)
            assert all(s.startswith(f"{arch}/") for s in spec["sources"]), spec
        twin = twin_spec["sources"] if twin_spec is not None else []
        for source in spec["sources"]:
            assert (csrc_dir / source).is_file(), (arch, variant, source)
            assert (source in twin) == source.startswith("common/"), (
                arch,
                variant,
                source,
            )


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
        # NVFP4 prefill bodies: the pools' scale-row views (run_cake_dsv4_nvfp4
        # builds them beside the data-row views the SWA_cache / compressed_KV_cache
        # aliases carry).
        "SWA_sf",
        "compressed_KV_sf",
        # SM107 split-KV producer: the [pieces, heads, 512] partial-O view.
        "partial_O_pieces",
    }
)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", [*_ARCHES, "sm_107a"])
def test_registered_arg_plans_use_known_names(arch):
    """Every generated argument is bindable by name."""
    unknown = []
    unaliased_tma = []
    for variant, spec in _ARCH_REGISTRATIONS[arch]["variants"].items():
        for kind, name in spec["arg_plan"]:
            canonical = cake.canonical_arg_name(kind, name)
            if kind == "tma_buffer" and canonical not in _HOST_TMA_SOURCE_TENSORS:
                # A descriptor name the host binds only by vocabulary would still
                # fail at launch: ``_bind_argument`` needs a tensor value for it.
                unaliased_tma.append((variant, name, canonical))
            if cake.is_bindable_arg(kind, name):
                continue
            unknown.append((variant, kind, name))
    assert unknown == []
    assert unaliased_tma == []


_PUBLIC_COMPILE_FLAGS = {
    "-std=c++17",
    "--use_fast_math",
    "-Xptxas=--register-usage-level=10",
}


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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

    The NVFP4 kernel-template units (decode swap, tile, tile_h64, pv) compile
    with every explicit instantiation; no record selects instantiations through
    preprocessor defines.
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


@pytest.mark.cpu_only
@pytest.mark.parametrize("table", ["ARG_PLANS", "MODULES", "VARIANTS"])
def test_bf16_fp8_registration_tables_are_plain_literals(table):
    """The BF16 / FP8 registrations are read from tables that a regenerated
    delivery rewrites as data, so each one stays a single literal (no name, call
    or operator inside it)."""
    import ast
    from pathlib import Path

    import flashinfer.jit.cake_dsv4 as jit

    tree = ast.parse(Path(jit.__file__).read_text(encoding="utf-8"))
    (node,) = [
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == table
    ]
    assert ast.literal_eval(node.value) == getattr(jit, table)


@pytest.mark.cpu_only
def test_bf16_fp8_programs_are_registered_on_exactly_their_architectures():
    """Every BF16 / FP8 program is named by a variant on each architecture its
    record lists and on no other, with one identity per such architecture."""
    import flashinfer.jit.cake_dsv4 as jit

    routed: dict[str, set[str]] = {}
    for arch, variants in jit.VARIANTS.items():
        for program in variants.values():
            routed.setdefault(program, set()).add(arch)
    assert set(routed) == set(jit.MODULES)
    for program, record in jit.MODULES.items():
        assert set(record["arches"]) == routed[program], program
        assert set(record["identity"]) == routed[program], program
        assert record["arg_plan"] in jit.ARG_PLANS, program
        assert all(pair[0] == "grid" for pair in jit.ARG_PLANS[record["arg_plan"]][-3:])


@pytest.mark.cpu_only
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
        "tmap_partial_o",
    ):
        assert cake.is_bindable_arg("tma_buffer", tma_name)
    # The SM107 split-KV producer's additions to the uniform plan.
    assert cake.canonical_arg_name("tma_buffer", "tmap_partial_o") == "partial_O_pieces"
    for name in ("num_whole_items", "num_splits"):
        assert cake.is_bindable_arg("parameter", name), name
    assert cake.canonical_arg_name("parameter", "num_q_heads") == "num_heads"
    assert cake.canonical_arg_name("parameter", "num_split") == "num_splits"
    assert not cake.is_bindable_arg("buffer", "completion_base")
    assert not cake.is_bindable_arg("parameter", "completion_base")
    assert not cake.is_bindable_arg("workspace", "tma_descriptor_workspace")


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


def _install_fake_variants(monkeypatch, plans):
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


@pytest.mark.cpu_only
def test_launch_variant_binds_by_name_with_fake_arg_plan(monkeypatch):
    recorder = _install_fake_variants(monkeypatch, {"fake": _FAKE_PLAN})
    rows, compressed = 3, 132
    table, lens = _combined_metadata(rows, compressed)
    meta = resolve_cake_dsv4_sparse_metadata(table, lens, query_rows=rows)
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
    assert all(isinstance(bound[n], int) for n in ("sparse_topk", "grid_x"))


@pytest.mark.cpu_only
def test_launch_variant_reports_unknown_and_unavailable_names(monkeypatch):
    plans = {
        "unknown": [
            ("buffer", "mystery"),
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
        "slab": [
            ("workspace", "tma_descriptor_workspace"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
    }
    _install_fake_variants(monkeypatch, plans)

    def launch(variant, **values):
        cake._launch_variant(variant, arch="sm_103a", grid=(1, 1, 1), values=values)

    with pytest.raises(ValueError, match="'mystery' \\(buffer\\) has no host value"):
        launch("unknown")
    # The descriptor-slab kind left the vocabulary with the by-value TMA ABI.
    with pytest.raises(ValueError, match="unknown argument kind 'workspace'"):
        launch("slab")
    # No host value is None for a real call (dense calls bind seq_lens in place
    # of cum_seq_lens_q); a None value is reported as unavailable.
    with pytest.raises(ValueError, match="not available for this call"):
        launch("ragged_only", cum_seq_lens_q=None)
    with pytest.raises(TypeError, match="must be an int"):
        cake._bind_argument(
            {"num_heads": 3.5}, "parameter", "num_heads", variant="x", grid={}
        )
    with pytest.raises(ValueError, match="three positive ints"):
        cake._launch_variant("unknown", arch="sm_103a", grid=(0, 1, 1), values={})


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


@pytest.mark.cpu_only
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
    # Partial buffers carved at the documented offsets.
    assert m["partial_O"].data_ptr() == workspace.data_ptr() + layout.partial_o[0]
    assert m["partial_O"].dtype == torch.bfloat16
    assert m["partial_O"].numel() == rows * 64 * num_splits * 512
    assert m["partial_lse"].data_ptr() == workspace.data_ptr() + layout.partial_lse[0]
    assert m["partial_lse"].dtype == torch.float32
    assert m["partial_lse"].numel() == rows * 64 * num_splits
    assert r["partial_O"] is m["partial_O"] and r["partial_lse"] is m["partial_lse"]


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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
    assert set(meta.kernel_kwargs()) == set(KERNEL_METADATA_PARAMS)
    # Explicit offset passes through.
    shifted = resolve_cake_dsv4_sparse_metadata(
        table, lens - 40, sparse_topk_lens_offset=40, query_rows=rows
    )
    assert shifted.sparse_topk_lens_offset == 40
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


@pytest.mark.cpu_only
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
    assert meta.separate_tables
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


def _aligned_parent(rows: int, cols: int) -> torch.Tensor:
    """A contiguous ``[rows, cols]`` int32 table at a 64-byte aligned address,
    holding distinct values."""
    raw = torch.zeros(rows * cols + 16, dtype=torch.int32)
    start = ((-raw.data_ptr()) % 64) // raw.element_size()
    parent = raw[start : start + rows * cols].view(rows, cols)
    parent.copy_(torch.arange(rows * cols, dtype=torch.int32).view(rows, cols))
    assert parent.data_ptr() % 64 == 0
    return parent


_OFF_BASE = r"must start at a 16-byte aligned address, got data_ptr\(\) % 16 == 4"


@pytest.mark.cpu_only
@pytest.mark.parametrize("compressed", [0, 132])
def test_metadata_resolution_rejects_tables_off_the_index_row_loads(compressed):
    """The BF16 / FP8 kernels load index rows 16 bytes at a time: a table whose
    base is not 16-byte aligned, or whose row stride is not a multiple of 4,
    is rejected before any launch, in the combined form and in the separate
    form (either table, including a compressed column view)."""
    rows, width = 5, 128 + compressed
    lens = torch.full((rows,), 128 + compressed // 2, dtype=torch.int32)
    parent = _aligned_parent(rows, width + 1)
    odd = f"row stride that is a multiple of 4 elements, got {width + 1}"
    # parent[:, 1:]: base +4 bytes, odd row stride.
    with pytest.raises(ValueError, match=f"^sparse_indices {_OFF_BASE}"):
        resolve_cake_dsv4_sparse_metadata(parent[:, 1:], lens, query_rows=rows)
    # parent[:, :W]: aligned base, odd row stride.
    with pytest.raises(ValueError, match=f"^sparse_indices must have a {odd}"):
        resolve_cake_dsv4_sparse_metadata(parent[:, :width], lens, query_rows=rows)
    # A one-row table has no row stride to keep.
    one = resolve_cake_dsv4_sparse_metadata(parent[:1, :width], lens[:1], query_rows=1)
    assert one.swa_indices.data_ptr() == parent.data_ptr()
    if not compressed:
        return
    extra_lens = lens - 128
    swa = _aligned_parent(rows, 128)
    # Separate form, the SWA table off: parent[:, 1:129].
    with pytest.raises(ValueError, match=f"^sparse_indices {_OFF_BASE}"):
        resolve_cake_dsv4_sparse_metadata(
            parent[:, 1:129],
            extra_sparse_indices=parent[:, 129:],
            extra_sparse_topk_lens=extra_lens,
            query_rows=rows,
        )
    # Separate form, the compressed column view of parent[:, 1:] off.
    with pytest.raises(ValueError, match=f"^extra_sparse_indices {_OFF_BASE}"):
        resolve_cake_dsv4_sparse_metadata(
            swa,
            extra_sparse_indices=parent[:, 129:],
            extra_sparse_topk_lens=extra_lens,
            query_rows=rows,
        )
    # Separate form, the compressed table on an aligned base with an odd row stride.
    with pytest.raises(ValueError, match=f"^extra_sparse_indices must have a {odd}"):
        resolve_cake_dsv4_sparse_metadata(
            swa,
            extra_sparse_indices=parent[:, 128:width],
            extra_sparse_topk_lens=extra_lens,
            query_rows=rows,
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("compressed", [0, 132])
def test_metadata_resolution_keeps_16_byte_offset_views_in_place(compressed):
    """A view 16 bytes into a larger table with a row stride that is a multiple
    of 4 is accepted as is: the bound tables keep its data_ptr and row stride
    (no copy), in the combined and the separate form."""
    rows, width = 5, 128 + compressed
    lens = torch.full((rows,), 128 + compressed // 2, dtype=torch.int32)
    # parent[:, 4:] of a [T, W + 4] parent: base +16 bytes, row stride W + 4.
    parent = _aligned_parent(rows, width + 4)
    view = parent[:, 4:]
    assert view.data_ptr() == parent.data_ptr() + 16 and view.stride(0) == width + 4
    meta = resolve_cake_dsv4_sparse_metadata(view, lens, query_rows=rows)
    assert meta.swa_indices is view
    assert meta.compressed_indices.data_ptr() == view.data_ptr() + (
        128 * 4 if compressed else 0
    )
    assert meta.swa_index_stride == meta.compressed_index_stride == width + 4
    kw = meta.kernel_kwargs()
    assert kw["swa_indices"].data_ptr() == view.data_ptr()
    assert kw["compressed_indices"].data_ptr() == meta.compressed_indices.data_ptr()
    assert kw["swa_indices"].numel() == (rows - 1) * (width + 4) + width
    if not compressed:
        return
    swa, extra = view[:, :128], view[:, 128:]
    sep = resolve_cake_dsv4_sparse_metadata(
        swa,
        extra_sparse_indices=extra,
        extra_sparse_topk_lens=lens - 128,
        query_rows=rows,
    )
    assert sep.swa_indices is swa and sep.compressed_indices is extra
    assert sep.swa_index_stride == sep.compressed_index_stride == width + 4
    kw = sep.kernel_kwargs()
    assert kw["swa_indices"].data_ptr() == view.data_ptr()
    assert kw["compressed_indices"].data_ptr() == view.data_ptr() + 128 * 4


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
# The L2 prefetch sibling's stand-in plan: the persistent plan with the
# compressed-KV pool and its row count before the grid.
_L2PF_PLAN = [
    *_PERSISTENT_PLAN[:-3],
    ("buffer", "compressed_KV_cache"),
    ("parameter", "compressed_KV_rows"),
    *_PERSISTENT_PLAN[-3:],
]


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


@pytest.mark.cpu_only
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

    def __init__(self, arch: str, *, split=None, **values):
        self.arch = arch
        # The call's PDL switch and longest request (run_cake_dsv4 sets both):
        # PDL off and one-token requests unless given.
        self.values = {"enable_pdl": False, "max_q_len": 1, **values}
        # The whole launch's split-KV plan (run_cake_dsv4 sets it; None unless given).
        self.split = split
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

    def pieces(self, rows, num_splits):
        return {
            "partial_O": f"partial_O[{rows}x{num_splits}]",
            "partial_O_pieces": f"partial_O_pieces[{rows}x{num_splits}]",
            "partial_lse": f"partial_lse[{rows}x{num_splits}]",
            "num_splits": num_splits,
        }

    def counters(self, merge_groups):
        return f"counters[{merge_groups}]"

    def reduce(self, reducer, **overrides):
        tokens = self.values["num_query_tokens"]
        heads = self.values["num_heads"]
        return self.variant(reducer, grid=(tokens, heads, 1), **overrides)


# The source-exact routes on a padded ragged call: (route, heads, dtype,
# request lengths, sparse width, compressed page size).
_SOURCE_EXACT_PADDED = [
    ("bf16_h8_h16_source_exact", 16, torch.bfloat16, [5, 3, 4], 260, 2),
    ("fp8_h8_h16_source_exact", 8, torch.float8_e4m3fn, [5, 3, 4], 192, 64),
    ("fp8_h64_source_exact", 64, torch.float8_e4m3fn, [257, 100], 640, 64),
]


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize(
    "route,num_heads,dtype,q_lens,sparse_topk,page", _SOURCE_EXACT_PADDED
)
def test_source_exact_programs_bind_the_metadata_rows(
    monkeypatch, arch, route, num_heads, dtype, q_lens, sparse_topk, page
):
    """The source-exact programs launch one CTA per request position: the grid
    is ``max_q_len`` x ``batch_size`` whatever the metadata rows ``T``.  A
    padded ragged call (``cum_seq_lens_q`` describing more rows than ``T``)
    through run_cake_dsv4, bound by the route's registered plan, passes
    ``num_query_tokens = T``, the first ``T`` rows of the query and of
    ``out`` and the caller's ``cum_seq_lens_q``: the program runs every
    position at or past ``T`` as padding."""
    plan = get_cake_dsv4_spec(route, arch=arch)["arg_plan"]
    names = [name for _, name in plan]
    recorder = _install_fake_variants(monkeypatch, {route: plan})
    monkeypatch.setattr(cake, "_target_arch", lambda device: arch)
    query_rows, batch_size, max_q_len = sum(q_lens), len(q_lens), max(q_lens)
    cum = torch.tensor([0, *q_lens], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    grid_y = 1 if route == "fp8_h64_source_exact" else num_heads // 8 * 4
    for rows in (1, query_rows - 1):
        table, lens = _combined_metadata(rows, sparse_topk - 128)
        out = (
            _aligned_u8(query_rows * num_heads * 512 * 2)
            .view(torch.bfloat16)
            .view(query_rows, num_heads, 512)
        )
        recorder.calls.clear()
        cake.run_cake_dsv4(
            query=torch.zeros((query_rows, num_heads, 512), dtype=dtype),
            swa_kv_cache=torch.zeros((4, 1, 256, 512), dtype=dtype),
            compressed_kv_cache=torch.zeros((8, 1, page, 512), dtype=dtype),
            workspace_buffer=_aligned_u8(
                get_cake_dsv4_workspace_bytes(
                    rows, num_heads, sparse_topk, dtype, num_splits=1
                )
            ),
            out=out,
            bmm1_scale=0.5,
            bmm2_scale=1.0,
            sinks=None,
            max_q_len=max_q_len,
            cum_seq_lens_q=cum,
            seq_lens=torch.full((batch_size,), 1000, dtype=torch.int32),
            backend="cake",
            sparse_indices=table,
            sparse_topk_lens=lens,
        )
        (call,) = recorder.calls
        bound = dict(zip(names, call, strict=True))
        assert bound["num_query_tokens"] == rows
        assert bound["tmap_q"].shape[0] == rows
        assert bound["O"].shape[0] == rows
        assert bound["O"].data_ptr() == out.data_ptr()
        assert bound["cum_seq_lens_q"].data_ptr() == cum.data_ptr()
        assert (bound["max_q_len"], bound["batch_size"]) == (max_q_len, batch_size)
        grid = (bound["grid_x"], bound["grid_y"], bound["grid_z"])
        assert grid == (max_q_len, grid_y, batch_size)
        if route == "fp8_h64_source_exact":
            assert bound["total_work_items"] == max_q_len * batch_size


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize("route", [route for route, *_ in _SOURCE_EXACT_PADDED])
def test_source_exact_bindings_take_their_arg_plans(arch, route):
    """The binding ``Run`` of each source-exact program takes the route's
    registered plan, name for name and in order (``a_<name>`` for a kernel
    argument, then ``grid_x``, ``grid_y``, ``grid_z``).  The host binds the
    plan by name and calls ``Run`` positionally, so a plan naming
    ``num_query_tokens`` that its program's binding does not take, or the
    reverse, shifts every later argument."""
    import re

    from flashinfer.jit.cake_dsv4 import _get_csrc_dir

    spec = get_cake_dsv4_spec(route, arch=arch)
    (binding,) = [name for name in spec["sources"] if name.endswith("_binding.cu")]
    text = (_get_csrc_dir() / binding).read_text(encoding="utf-8")
    (params,) = re.findall(r"^void Run\(([^)]*)\)", text, re.MULTILINE)
    taken = [param.split()[-1].removeprefix("a_") for param in params.split(",")]
    assert taken == [name for _, name in spec["arg_plan"]], binding


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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
                            sm_count=_SM_COUNT,
                            arch=arch,
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
@pytest.mark.parametrize("clusters", [74, 76])
def test_bf16_h128_prefill_snake_feed_predicate_full_rounds(clusters):
    assert cake._bf16_h128_prefill_uses_snake_feed(8 * clusters, 640, clusters) is False


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "num_query_tokens,page_size,sparse_topk,expected",
    [
        (
            12,
            64,
            640,
            "fp8_h128_prefill_source_persistent_uniform",
        ),  # W5: 12 tokens, 5 tiles
        (
            12,
            2,
            260,
            "fp8_h64_prefill_source_persistent_m64_multi_tile",
        ),  # 12 tokens, 2 tiles, no arch: the M64 body (SM100 / SM103: below)
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
        (
            74,
            None,
            128,
            "fp8_lowhead_prefill",
        ),  # SWA producer while 2 x tokens fit the SMs
        (75, None, 128, "fp8_h64_prefill_source_persistent_m64"),  # second wave: M64
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
            sm_count=_SM_COUNT,
        )
        == expected
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a", None])
@pytest.mark.parametrize("page_size", [2, 64])
@pytest.mark.parametrize("sparse_topk", [252, 256, 260, 380, 384])
def test_fp8_h64_two_tile_decode_rows_take_the_h64_cluster_body_on_sm100(
    arch, page_size, sparse_topk
):
    """FP8/H64 decode rows of at most 12 tokens with two complete tiles and a
    partial third run the H64 cluster body on SM100 and the persistent body
    on SM103; width 256 and one complete tile keep the M64 body, three tiles
    or 13+ tokens the persistent body, and SM107 or an unknown architecture
    the M64 body (SM107 runs width 252 past 12 tokens on the persistent
    body)."""
    for tokens in range(1, 14):
        route = _route(
            dtype=torch.float8_e4m3fn,
            num_heads=64,
            batch_size=3,
            max_q_len=5,
            ragged=True,
            sparse_topk=sparse_topk,
            compressed_page_size=page_size,
            num_query_tokens=tokens,
            sm_count=_SM_COUNT,
            arch=arch,
        )
        full_tiles = sparse_topk // 128
        partial_third = 256 < sparse_topk < 384
        if (
            full_tiles >= (3 if tokens <= 12 else 2)
            or (tokens > 12 and arch == "sm_107a")
            or (partial_third and arch == "sm_103a")
        ):
            expected = "fp8_h128_prefill_source_persistent_uniform"
        elif partial_third and arch == "sm_100a":
            expected = "fp8_lowhead_h64"
        else:
            expected = "fp8_h64_prefill_source_persistent_m64_multi_tile"
        assert route == expected, (tokens, route)
        if arch is not None:
            assert route in _ARCH_REGISTRATIONS[arch]["variants"], (arch, route)


@pytest.mark.cpu_only
def test_workspace_requirement_routes_by_the_device_arch(monkeypatch):
    """The requirement's route follows the architecture the launch routes by;
    the workspace numbers do not depend on it."""
    shape = dict(
        dtype=torch.float8_e4m3fn,
        num_heads=64,
        num_query_tokens=12,
        sm_count=_SM_COUNT,
        sparse_topk=260,
        compressed_page_size=2,
        max_q_len=5,
        batch_size=3,
        ragged=True,
    )
    h64 = cake.cake_dsv4_workspace_requirement(**shape, arch="sm_100a")
    m64 = cake.cake_dsv4_workspace_requirement(**shape, arch="sm_107a")
    assert h64.route == "fp8_lowhead_h64"
    assert m64.route == "fp8_h64_prefill_source_persistent_m64_multi_tile"
    assert (h64.single_launch_bytes, h64.minimum_bytes) == (
        m64.single_launch_bytes,
        m64.minimum_bytes,
    )
    # arch=None: the current CUDA device's architecture, whatever sm_count says.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(cake, "_target_arch", lambda device: "sm_100a")
    assert cake.cake_dsv4_workspace_requirement(**shape).route == "fp8_lowhead_h64"
    # Without CUDA: the route every architecture builds.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cake.cake_dsv4_workspace_requirement(**shape).route == m64.route


@pytest.mark.cpu_only
@pytest.mark.parametrize("sm_count", [148, 152, 160])
@pytest.mark.parametrize(
    "sparse_topk,page_size,below",
    [
        (128, None, "fp8_lowhead_prefill"),
        (640, 64, "fp8_h128_prefill_source_persistent_uniform"),
        (204, 2, "fp8_h64_prefill_source_persistent_m64_multi_tile"),
    ],
)
def test_fp8_h64_decode_rows_take_the_m64_body_once_two_ctas_per_token_exceed_the_sms(
    sm_count, sparse_topk, page_size, below
):
    """The rule follows the device: ``2 * tokens > sm_count`` selects the M64 body."""
    m64 = cake._fp8_h64_m64_program(sparse_topk)

    def route(tokens):
        return _route(
            dtype=torch.float8_e4m3fn,
            num_heads=64,
            batch_size=tokens // 4 + 1,
            max_q_len=4,
            ragged=True,
            sparse_topk=sparse_topk,
            compressed_page_size=page_size,
            num_query_tokens=tokens,
            sm_count=sm_count,
        )

    assert route(sm_count // 2) == below
    assert route(sm_count // 2 + 1) == m64
    for tokens in (108, 120):
        assert route(tokens) == m64


# The SM count of the R200 board the SM107 low-head rules were measured on.
_SM107_SM_COUNT = 216


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a", None])
@pytest.mark.parametrize("sm_count", [_SM_COUNT, _SM107_SM_COUNT])
@pytest.mark.parametrize(
    "sparse_topk,page_size,m64",
    [
        (128, 256, "fp8_h64_prefill_source_persistent_m64"),
        (640, 64, "fp8_h64_prefill_source_persistent_m64_multi_tile"),
        (1152, 64, "fp8_h64_prefill_source_persistent_m64_multi_tile"),
        (260, 2, "fp8_h64_prefill_source_persistent_m64_multi_tile"),
        (1152, 2, "fp8_h64_prefill_source_persistent_m64_multi_tile"),
    ],
)
def test_fp8_lowhead_prefill_rows_take_the_m64_body(
    arch, sm_count, sparse_topk, page_size, m64
):
    """FP8 8-, 16- and 32-head prefill rows (``max_q_len >= 257``) run the M64
    body on every architecture at every token count: the single-tile program
    for the SWA-only width, the multi-tile program for wider items of either
    compressed cache layout."""
    for batch_size, max_q_len, tokens, ragged in (
        (1, 257, 257, True),
        (2, 257, 386, True),
        (2, 512, 1024, False),
        (2, 1024, 2048, True),
        (1, 2048, 2048, True),
        (2, 5462, 8192, True),
    ):
        for num_heads in (8, 16, 32):
            route = _route(
                dtype=torch.float8_e4m3fn,
                num_heads=num_heads,
                batch_size=batch_size,
                max_q_len=max_q_len,
                ragged=ragged,
                sparse_topk=sparse_topk,
                compressed_page_size=page_size,
                num_query_tokens=tokens,
                sm_count=sm_count,
                arch=arch,
            )
            assert route == m64, (num_heads, batch_size, max_q_len, tokens, route)
            if arch is not None:
                assert route in _ARCH_REGISTRATIONS[arch]["variants"], (arch, route)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a", None])
@pytest.mark.parametrize("num_heads", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("page_size", [2, 64])
def test_fp8_wider_than_1152_is_rejected_on_every_path(arch, num_heads, page_size):
    """No FP8 body takes an item wider than 1152 sparse slots (the persistent and
    M64 bindings check it; the low-head producers would write past their
    shared-memory index buffer).  Every FP8 path -- decode, MTP and prefill
    rows of every head count -- raises one ValueError from the route before
    any launch, naming the width, the limit, the C128 layers past 128K context
    that reach it and the routes that run them (the NVFP4 cache only up to its
    route's 1536 slots, and not on SM107, which registers no NVFP4 program);
    1152 itself still routes."""
    assert cake._FP8_MAX_WIDTH == 1152
    sm_count = _SM107_SM_COUNT if arch == "sm_107a" else _SM_COUNT
    shapes = (
        (1, 1, 1, False),  # one decode token
        (64, 1, 64, False),
        (256, 1, 256, False),  # past one wave
        (16, 4, 64, False),  # MTP
        (3, 5, 9, True),  # ragged MTP
        (1, 257, 257, True),  # prefill
        (2, 2048, 4096, True),
    )
    for batch_size, max_q_len, tokens, ragged in shapes:
        kwargs = dict(
            dtype=torch.float8_e4m3fn,
            num_heads=num_heads,
            batch_size=batch_size,
            max_q_len=max_q_len,
            ragged=ragged,
            compressed_page_size=page_size,
            num_query_tokens=tokens,
            sm_count=sm_count,
            arch=arch,
        )
        if arch != "sm_107a" or num_heads in (64, 128) or max_q_len >= 257:
            assert _route(sparse_topk=1152, **kwargs).startswith("fp8_")
        for sparse_topk in (1153, 1156, 1280, 1536, 1537, 2176, 4224):
            with pytest.raises(ValueError) as excinfo:
                _route(sparse_topk=sparse_topk, **kwargs)
            message = str(excinfo.value)
            assert f"sparse_topk {sparse_topk}" in message
            assert "at most 1152 sparse slots" in message
            assert "past 128K context" in message
            assert "backend='trtllm-gen'" in message
            # The NVFP4 cache is named only for rows its route takes, and only
            # where NVFP4 programs are registered (SM107 has none).
            assert ("kv_cache_format='nvfp4'" in message) == (
                sparse_topk <= 1536 and arch != "sm_107a"
            )


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a", None])
@pytest.mark.parametrize("sm_count", [_SM_COUNT, _SM107_SM_COUNT])
def test_sm107a_fp8_h64_source_exact_row_takes_the_m64_body(arch, sm_count):
    """The FP8/H64 source-exact row (two ragged requests of up to 257 tokens,
    topk4x items 640 wide) runs the multi-tile M64 program on SM107, which
    registers no source-exact program, as its dense twin does; SM100, SM103
    and an unknown architecture keep the source-exact program.  The route
    reads the requests through ``max_q_len`` only: 2 x 257, 257 + 100 and the
    contract's 129 + 257 tokens route alike, with and without the four-stage
    edges."""
    for name, registration in _ARCH_REGISTRATIONS.items():
        on_m64 = name in cake._FP8_H64_SOURCE_EXACT_M64_ARCHES
        assert ("fp8_h64_source_exact" in registration["variants"]) != on_m64, name
    m64 = "fp8_h64_prefill_source_persistent_m64_multi_tile"
    expected = m64 if arch == "sm_107a" else "fp8_h64_source_exact"
    for tokens in (514, 357, _canonical_query_tokens(2, 257, True)):
        for kv4 in (False, True):
            route = _route(
                dtype=torch.float8_e4m3fn,
                num_heads=64,
                batch_size=2,
                max_q_len=257,
                ragged=True,
                sparse_topk=640,
                compressed_page_size=64,
                num_query_tokens=tokens,
                sm_count=sm_count,
                arch=arch,
                kv4=kv4,
            )
            assert route == expected, (tokens, kv4, route)
            if arch is not None:
                assert get_cake_dsv4_spec(route, arch=arch)["arg_plan"]
    # The dense twin runs the M64 program on every architecture.
    twin = _route(
        dtype=torch.float8_e4m3fn,
        num_heads=64,
        batch_size=2,
        max_q_len=257,
        ragged=False,
        sparse_topk=640,
        compressed_page_size=64,
        num_query_tokens=514,
        sm_count=sm_count,
        arch=arch,
    )
    assert twin == m64


@pytest.mark.cpu_only
@pytest.mark.parametrize("tokens", [514, 357])
def test_sm107a_fp8_h64_source_exact_row_launches_the_m64_program(monkeypatch, tokens):
    """On SM107 the ragged source-exact row launches the multi-tile M64 program
    (its four-stage sibling where this process can launch it), one CTA per
    token with the LSE region of one split, and sizes its workspace as every
    one-split M64 launch does; SM100 keeps the source-exact program, which
    takes no workspace."""
    shape = dict(
        dtype=torch.float8_e4m3fn,
        num_heads=64,
        num_query_tokens=tokens,
        sm_count=_SM107_SM_COUNT,
        sparse_topk=640,
        compressed_page_size=64,
        max_q_len=257,
        batch_size=2,
        ragged=True,
    )
    route = "fp8_h64_prefill_source_persistent_m64_multi_tile"
    m64 = cake.cake_dsv4_workspace_requirement(**shape, arch="sm_107a")
    assert (m64.route, m64.num_splits, m64.uses_workspace) == (route, 1, True)
    layout = cake_dsv4_workspace_layout(tokens, 64, 1)
    assert m64.single_launch_bytes == layout.total_bytes
    bound = get_cake_dsv4_workspace_bytes(tokens, 64, 640, torch.float8_e4m3fn)
    assert m64.single_launch_bytes <= bound
    exact = cake.cake_dsv4_workspace_requirement(**shape, arch="sm_100a")
    assert (exact.route, exact.uses_workspace) == ("fp8_h64_source_exact", False)
    assert cake.cake_dsv4_out_phase(route, num_heads=64, sparse_topk=640) is None
    registered = set(_ARCH_REGISTRATIONS["sm_107a"]["variants"])
    for launchable, program in ((True, f"{route}_kv4"), (False, route)):
        monkeypatch.setattr(
            cake, "_kv4_sibling_launchable", lambda arch, device: launchable
        )
        L = _RecordingLauncher(
            "sm_107a",
            Q=torch.empty(0),
            num_query_tokens=tokens,
            num_heads=64,
            sparse_topk=640,
            max_q_len=257,
            batch_size=2,
            enable_pdl=True,
        )
        cake._dispatch_route(route, L)
        parts = L.partials(1)
        launch = {"grid": (tokens, 1, 1), "total_work_items": tokens, **parts}
        assert L.calls == [(program, launch)]
        assert program in registered


# FP8 routes SM107 refuses by design: it builds no source-exact program for
# the 8/16-head shape lock (get_cake_dsv4_spec names the programs it runs), so
# run_cake_dsv4 raises for it.  Per route, the head counts that reach it.  The
# 8/16/32-head decode rows over compressed items up to 384 wide run the M64
# body on SM107 (_fp8_lowhead_narrow_uses_one_partition), not the
# one-partition producer it does not build.
_SM107_REFUSED_FP8_ROUTES = {
    # The 8/16-head shape lock of three ragged requests of up to 5 tokens.
    "fp8_h8_h16_source_exact": (8, 16),
}


@pytest.mark.cpu_only
@pytest.mark.parametrize("kv4", [False, True])
@pytest.mark.parametrize("sm_count", [208, 212, _SM107_SM_COUNT])
def test_every_sm107a_fp8_route_result_has_a_registered_kernel_or_is_refused(
    monkeypatch, sm_count, kv4
):
    """Over the route sweep of the SM100 / SM103 guard, at the SM counts of the
    SM107 parts and with and without the four-stage edges, every FP8 route
    ``_route`` returns on SM107 launches registered SM107 programs only, or is
    one of ``_SM107_REFUSED_FP8_ROUTES`` at one of its head counts; every
    entry of that list is reached and none of them is registered.  The
    dispatch sees the same SM count and four-stage launch check as the route."""
    monkeypatch.setattr(cake, "_device_sm_count", lambda device: sm_count)
    monkeypatch.setattr(cake, "_kv4_sibling_launchable", lambda arch, device: kv4)
    registered = set(_ARCH_REGISTRATIONS["sm_107a"]["variants"])
    refused: dict[str, set] = {}
    routed = rejected = 0
    for num_heads in (8, 16, 32, 64, 128):
        # Every FP8 width up to 1152 routes on SM107; a C128 item past
        # 128K context (2176 slots) is the width the route rejects.
        for sparse_topk, page_size in (*_SWEEP_WIDTHS, (2176, 2)):
            for batch_size, max_q_len, ragged in _SWEEP_QUERIES:
                tokens = _canonical_query_tokens(batch_size, max_q_len, ragged)
                try:
                    route = _route(
                        dtype=torch.float8_e4m3fn,
                        num_heads=num_heads,
                        max_q_len=max_q_len,
                        ragged=ragged,
                        sparse_topk=sparse_topk,
                        batch_size=batch_size,
                        compressed_page_size=page_size,
                        num_query_tokens=tokens,
                        sm_count=sm_count,
                        arch="sm_107a",
                        kv4=kv4,
                    )
                except ValueError:
                    rejected += 1
                    continue
                heads = _SM107_REFUSED_FP8_ROUTES.get(route)
                if heads is not None:
                    assert num_heads in heads, (route, num_heads)
                    refused.setdefault(route, set()).add(num_heads)
                    continue
                L = _RecordingLauncher(
                    "sm_107a",
                    Q=torch.empty(0),
                    num_query_tokens=tokens,
                    num_heads=num_heads,
                    sparse_topk=sparse_topk,
                    max_q_len=max_q_len,
                    batch_size=batch_size,
                )
                cake._dispatch_route(route, L)
                launched = [name for name, _ in L.calls]
                assert launched, route
                missing = [name for name in launched if name not in registered]
                assert missing == [], (num_heads, sparse_topk, max_q_len, route)
                routed += 1
    assert routed > 0 and rejected > 0
    want = {route: set(heads) for route, heads in _SM107_REFUSED_FP8_ROUTES.items()}
    assert refused == want
    for route in _SM107_REFUSED_FP8_ROUTES:
        assert route not in registered, route
        with pytest.raises(ValueError, match="SM107 runs the FP8 decode route"):
            get_cake_dsv4_spec(route, arch="sm_107a")


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a", None])
@pytest.mark.parametrize("num_heads", [8, 16, 32])
@pytest.mark.parametrize("sm_count", [_SM_COUNT, 152, _SM107_SM_COUNT])
@pytest.mark.parametrize("page_size", [2, 64])
def test_fp8_lowhead_wide_decode_rows_take_the_persistent_or_m64_body(
    arch, num_heads, sm_count, page_size
):
    """FP8 decode rows (``max_q_len < 257``) of 8, 16 or 32 heads wider than one
    producer partition (385-1152 sparse slots: C4 at 640 or 1152, C128 past
    32K context).  32 heads on SM100 and SM103 run the FP8/H128 persistent
    body while its two CTAs per token fit one wave and the M64 body past it;
    8 and 16 heads (whose O rows the persistent epilogue's 32-row boxes would
    overrun), SM107 and an unknown architecture run the M64 body at every
    token count."""
    m64 = "fp8_h64_prefill_source_persistent_m64_multi_tile"
    for sparse_topk in (385, 512, 640, 1024, 1152):
        for tokens in (1, 12, 13, 64, sm_count // 2, sm_count // 2 + 1, 256, 1024):
            for max_q_len, ragged in ((1, False), (4, False), (6, True)):
                if max_q_len > tokens:
                    continue
                route = _route(
                    dtype=torch.float8_e4m3fn,
                    num_heads=num_heads,
                    batch_size=-(-tokens // max_q_len),
                    max_q_len=max_q_len,
                    ragged=ragged,
                    sparse_topk=sparse_topk,
                    compressed_page_size=page_size,
                    num_query_tokens=tokens,
                    sm_count=sm_count,
                    arch=arch,
                )
                one_wave = 2 * tokens <= sm_count
                if num_heads == 32 and arch in ("sm_100a", "sm_103a") and one_wave:
                    expected = cake._fp8_persistent_program(tokens)
                else:
                    expected = m64
                assert route == expected, (sparse_topk, tokens, max_q_len, route)
                if arch is not None:
                    assert route in _ARCH_REGISTRATIONS[arch]["variants"]


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a", None])
@pytest.mark.parametrize("num_heads", [8, 16, 32])
@pytest.mark.parametrize("sm_count", [_SM_COUNT, 152, _SM107_SM_COUNT])
@pytest.mark.parametrize("page_size", [2, 64])
def test_fp8_lowhead_narrow_decode_rows_leave_one_partition_past_one_wave(
    arch, num_heads, sm_count, page_size
):
    """FP8 decode rows (``max_q_len < 257``) of 8, 16 or 32 heads with a
    compressed cache and at most 384 sparse slots run the one-partition
    producer while its two CTAs per token fit one wave and the M64 body past
    it; SM107, which builds no one-partition program, runs the M64 body at
    every token count."""
    m64 = "fp8_h64_prefill_source_persistent_m64_multi_tile"
    for sparse_topk in (132, 192, 256, 260, 320, 384):
        for tokens in (1, 12, 16, 64, sm_count // 2, sm_count // 2 + 1, 256):
            for max_q_len, ragged in ((1, False), (4, False), (6, True)):
                if max_q_len > tokens:
                    continue
                route = _route(
                    dtype=torch.float8_e4m3fn,
                    num_heads=num_heads,
                    batch_size=-(-tokens // max_q_len),
                    max_q_len=max_q_len,
                    ragged=ragged,
                    sparse_topk=sparse_topk,
                    compressed_page_size=page_size,
                    num_query_tokens=tokens,
                    sm_count=sm_count,
                    arch=arch,
                )
                one_wave = 2 * tokens <= sm_count
                if one_wave and arch != "sm_107a":
                    expected = "fp8_lowhead_one_partition"
                else:
                    expected = m64
                assert route == expected, (sparse_topk, tokens, max_q_len, route)
                if arch is not None:
                    assert route in _ARCH_REGISTRATIONS[arch]["variants"]


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a", None])
@pytest.mark.parametrize("num_heads", [8, 16, 32])
@pytest.mark.parametrize("sm_count", [_SM_COUNT, 152, _SM107_SM_COUNT])
def test_fp8_lowhead_swa_decode_rows_take_the_m64_body_past_one_wave(
    arch, num_heads, sm_count
):
    """FP8 SWA-only decode rows (``max_q_len < 257``) of 8, 16 and 32 heads run
    the single-tile M64 program on SM100, SM103 and SM107 once the low-head
    prefill producer's two CTAs per token exceed the SM count (216 SMs: 108
    tokens keep the producer, 109 take the M64 program; 152 SMs: 76 and 77);
    an unknown architecture keeps the producer at every token count."""
    m64 = "fp8_h64_prefill_source_persistent_m64"
    assert cake._fp8_h64_m64_program(128) == m64
    for tokens in (1, 12, 96, sm_count // 2, sm_count // 2 + 1, 112, 144, 256, 1024):
        two_waves = 2 * tokens > sm_count
        expected = m64 if arch is not None and two_waves else "fp8_lowhead_prefill"
        for max_q_len, ragged in ((1, False), (4, True), (256, True)):
            if max_q_len > tokens:
                continue
            route = _route(
                dtype=torch.float8_e4m3fn,
                num_heads=num_heads,
                batch_size=-(-tokens // max_q_len),
                max_q_len=max_q_len,
                ragged=ragged,
                sparse_topk=128,
                compressed_page_size=256,
                num_query_tokens=tokens,
                sm_count=sm_count,
                arch=arch,
            )
            assert route == expected, (tokens, max_q_len, route)
            if arch is not None:
                assert route in _ARCH_REGISTRATIONS[arch]["variants"], (arch, route)


@pytest.mark.cpu_only
def test_sm107a_lowhead_m64_rule_does_not_follow_the_h64_rule(monkeypatch):
    """The SM107 low-head decode rule is its own: a different 64-head wave rule,
    stricter or looser, moves the 64-head rows only."""

    def route(num_heads, tokens):
        return _route(
            dtype=torch.float8_e4m3fn,
            num_heads=num_heads,
            batch_size=tokens,
            max_q_len=1,
            ragged=False,
            sparse_topk=128,
            compressed_page_size=256,
            num_query_tokens=tokens,
            sm_count=_SM107_SM_COUNT,
            arch="sm_107a",
        )

    m64 = "fp8_h64_prefill_source_persistent_m64"
    for num_heads in (8, 16, 32, 64):
        assert route(num_heads, 108) == "fp8_lowhead_prefill"
        assert route(num_heads, 109) == m64
    monkeypatch.setattr(cake, "_fp8_h64_uses_m64_body", lambda *a, **kw: False)
    assert route(64, 109) == "fp8_lowhead_prefill"
    for num_heads in (8, 16, 32):
        assert route(num_heads, 108) == "fp8_lowhead_prefill"
        assert route(num_heads, 109) == m64
    monkeypatch.setattr(cake, "_fp8_h64_uses_m64_body", lambda *a, **kw: True)
    assert route(64, 108) == m64
    for num_heads in (8, 16, 32):
        assert route(num_heads, 108) == "fp8_lowhead_prefill"
        assert route(num_heads, 109) == m64


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "num_heads,num_query_tokens,max_q_len,batch_size,sparse_topk,page_size",
    [
        (8, 257, 257, 1, 128, 256),
        (8, 386, 257, 2, 640, 64),
        (8, 2048, 1024, 2, 1152, 64),
        (8, 112, 4, 28, 128, 256),
        (16, 144, 4, 36, 128, 256),
        (32, 256, 4, 64, 128, 256),
    ],
)
def test_sm107a_lowhead_m64_routes_need_the_low_head_workspace(
    num_heads, num_query_tokens, max_q_len, batch_size, sparse_topk, page_size
):
    """The sizing API routes by the architecture as the launch does; the M64
    routes that replace the low-head prefill producer (the SWA-only decode
    rows past one wave on a known architecture, the prefill rows everywhere)
    carve the same one-split workspace (LSE region only)."""
    shape = dict(
        dtype=torch.float8_e4m3fn,
        num_heads=num_heads,
        num_query_tokens=num_query_tokens,
        sm_count=_SM107_SM_COUNT,
        sparse_topk=sparse_topk,
        compressed_page_size=page_size,
        max_q_len=max_q_len,
        batch_size=batch_size,
        ragged=True,
    )
    m64 = cake.cake_dsv4_workspace_requirement(**shape, arch="sm_107a")
    assert m64.route == cake._fp8_h64_m64_program(sparse_topk)
    assert m64.num_splits == 1 and m64.uses_workspace
    plan = dict(
        num_query_tokens=num_query_tokens, num_heads=num_heads, sparse_topk=sparse_topk
    )
    assert cake._route_plan(m64.route, **plan) == cake._route_plan(
        "fp8_lowhead_prefill", **plan
    )
    assert m64.single_launch_bytes == cake._PARTIAL_OFFSET + cake._align_up(
        num_query_tokens * num_heads * 4
    )


def _run_fake_lowhead_swa_fp8(monkeypatch, *, arch, num_heads, batch_size, q_len):
    """Drive run_cake_dsv4 on CPU tensors for a dense FP8 SWA-only call on
    ``arch`` (216 SMs); return the launched program and its bound arguments."""
    plans = {
        route: _PERSISTENT_PLAN
        for route in (
            "fp8_lowhead_prefill",
            "fp8_lowhead_prefill_overlap",
            "fp8_lowhead_prefill_vblock",
            "fp8_h64_prefill_source_persistent_m64",
            "fp8_h64_prefill_source_persistent_m64_multi_tile",
        )
    }
    recorder = _install_fake_variants(monkeypatch, plans)
    seen = []

    def module(variant, **kwargs):
        seen.append(variant)
        return recorder

    monkeypatch.setattr(cake, "_variant_module", module)
    monkeypatch.setattr(cake, "_target_arch", lambda device: arch)
    monkeypatch.setattr(cake, "_device_sm_count", lambda device: _SM107_SM_COUNT)
    rows = batch_size * q_len
    table, lens = _combined_metadata(rows, 0)
    seq_lens = torch.full((batch_size,), 1000, dtype=torch.int32)
    out = cake.run_cake_dsv4(
        query=torch.zeros((rows, num_heads, 512), dtype=torch.float8_e4m3fn),
        swa_kv_cache=torch.zeros((4, 1, 256, 512), dtype=torch.float8_e4m3fn),
        compressed_kv_cache=torch.zeros((4, 1, 256, 512), dtype=torch.float8_e4m3fn),
        workspace_buffer=_aligned_u8(
            get_cake_dsv4_workspace_bytes(rows, num_heads, 128, torch.float8_e4m3fn)
        ),
        out=None,
        bmm1_scale=0.5,
        bmm2_scale=1.0,
        sinks=None,
        max_q_len=q_len,
        cum_seq_lens_q=None,
        seq_lens=seq_lens,
        backend="cake",
        sparse_indices=table,
        sparse_topk_lens=lens,
    )
    (program,) = seen
    (call,) = recorder.calls
    bound = dict(zip((name for _, name in _PERSISTENT_PLAN), call, strict=True))
    return program, bound, seq_lens, out


@pytest.mark.cpu_only
@pytest.mark.parametrize("num_heads", [8, 16, 32])
@pytest.mark.parametrize("arch", ["sm_103a", "sm_107a"])
def test_lowhead_swa_rows_launch_the_m64_program_with_dense_offsets(
    monkeypatch, num_heads, arch
):
    """A dense SWA-only call of two waves (60 x 2 = 120 tokens on 216 SMs) runs
    the single-tile M64 program: one CTA per token and the cached dense
    request offsets (the M64 programs read cum_seq_lens_q unconditionally);
    a call of one wave (50 x 2 = 100 tokens) runs the low-head prefill
    producer (two CTAs per token, seq_lens in place of the never-read
    offsets) on SM103.  Neither pins the output phase for these head
    counts."""
    route, bound, _, out = _run_fake_lowhead_swa_fp8(
        monkeypatch, arch=arch, num_heads=num_heads, batch_size=60, q_len=2
    )
    assert route == "fp8_h64_prefill_source_persistent_m64"
    assert route in cake._RAGGED_ONLY_ROUTES
    assert (bound["grid_x"], bound["grid_y"], bound["grid_z"]) == (120, 1, 1)
    assert bound["total_work_items"] == bound["num_query_tokens"] == 120
    assert bound["cum_seq_lens_q"].tolist() == list(range(0, 122, 2))
    assert bound["O"].data_ptr() == out.data_ptr()
    assert tuple(out.shape) == (120, num_heads, 512)
    route, bound, seq_lens, _ = _run_fake_lowhead_swa_fp8(
        monkeypatch, arch="sm_103a", num_heads=num_heads, batch_size=50, q_len=2
    )
    assert route == "fp8_lowhead_prefill"
    assert (bound["grid_x"], bound["grid_y"], bound["grid_z"]) == (200, 1, 1)
    assert bound["total_work_items"] == 200
    assert bound["cum_seq_lens_q"].data_ptr() == seq_lens.data_ptr()
    for name in ("fp8_lowhead_prefill", "fp8_h64_prefill_source_persistent_m64"):
        phase = cake.cake_dsv4_out_phase(name, num_heads=num_heads, sparse_topk=128)
        assert phase is None and cake._OUT_ALIGN_BY_ROUTE[name] == 32


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a", None])
@pytest.mark.parametrize(
    "sparse_topk,page_size,sm107_first",
    [
        # (width, page size, first M64 token count on SM107 at 208 / 216 SMs)
        (160, 2, (13, 13)),  # one complete tile, 32 keys more: M64 throughout
        (175, 2, (13, 13)),
        (176, 2, (17, 17)),  # 48 keys more: the persistent body up to 16 tokens
        (191, 2, (17, 17)),
        (192, 2, (32, 33)),  # 64+ keys more: 20 * tokens > 3 * sm_count
        (252, 2, (32, 33)),
        (256, 2, (27, 28)),  # two complete tiles: 8 * tokens > sm_count
        (260, 2, (54, 56)),  # three tiles, two complete: 200 * t > 51 * sm
        (320, 64, (54, 56)),
        (384, 2, (53, 55)),  # three complete tiles: 4 * tokens > sm_count
        (400, 2, (63, 65)),  # four tiles, three complete: 10 * t > 3 * sm
        (500, 2, (63, 65)),
        (512, 2, (55, 57)),  # four complete tiles: 50 * t > 13 * sm
        (516, 2, (105, 109)),  # five tiles and up: the half wave
        (640, 64, (105, 109)),
    ],
)
def test_fp8_h64_decode_rows_take_m64_before_the_half_wave_on_sm107(
    arch, sparse_topk, page_size, sm107_first
):
    """On SM107 the FP8/H64 decode rows of two to four sparse tiles take the
    M64 body before ``2 * tokens > sm_count``, by their tile and complete-tile
    counts, and the rows of one complete tile and 64 or more keys of a second
    run the persistent body until then (48-63 keys: up to 16 tokens); other
    widths and architectures keep the half-wave rule."""
    m64 = cake._fp8_h64_m64_program(sparse_topk)
    for sm_count, sm107 in zip((208, 216), sm107_first, strict=True):

        def route(tokens):
            return _route(
                dtype=torch.float8_e4m3fn,
                num_heads=64,
                batch_size=tokens // 4 + 1,
                max_q_len=4,
                ragged=True,
                sparse_topk=sparse_topk,
                compressed_page_size=page_size,
                num_query_tokens=tokens,
                sm_count=sm_count,
                arch=arch,
            )

        if arch == "sm_107a":
            first = sm107
        elif sparse_topk < 256:
            first = 13
        else:
            first = sm_count // 2 + 1
        below = "fp8_h128_prefill_source_persistent_uniform"
        for tokens in range(13, first):
            assert route(tokens) == below, (sm_count, tokens)
        for tokens in range(first, sm_count // 2 + 2):
            assert route(tokens) == m64, (sm_count, tokens)
        if arch is not None:
            assert m64 in _ARCH_REGISTRATIONS[arch]["variants"]


@pytest.mark.cpu_only
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
            sm_count=_SM_COUNT,
        )
        == expected
    )
    # Both M64 programs are ragged-only persistent routes with one CTA per token.
    assert expected in cake._RAGGED_ONLY_ROUTES


@pytest.mark.cpu_only
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
    L = _RecordingLauncher(
        arch, Q=torch.empty(0), num_query_tokens=tokens, num_heads=64, sparse_topk=388
    )
    cake._dispatch_route(route, L)
    assert L.calls == [
        (route, {"grid": (tokens, 1, 1), "total_work_items": tokens, **L.partials(1)})
    ]


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_calls_without_pdl_launch_the_pdl_free_twins(arch):
    """SM100 and SM103 route the FP8 H128 persistent route's PDL-free twin to
    programs of their own.  Without PDL, SM103 runs it on launches of at least
    384 rows and SM100 on calls whose longest request has at least 257 tokens;
    other launches run the route.  SM107 does not route the twin and launches
    the route."""
    from flashinfer.jit.cake_dsv4 import VARIANTS

    assert {"fp8_h128_prefill_source_persistent"} == cake._NO_PDL_TWIN_ROUTES
    assert cake._NO_PDL_TWIN_MIN_TOKENS == {"sm_103a": 384}
    assert cake._NO_PDL_TWIN_MIN_QUERY_LEN == {"sm_100a": 257}
    for route in sorted(cake._NO_PDL_TWIN_ROUTES):
        twin = f"{route}_no_pdl"
        assert (twin in VARIANTS[arch]) == (arch != "sm_107a")
        own = arch in ("sm_100a", "sm_103a")
        program = VARIANTS[arch][route]
        assert (VARIANTS[arch].get(twin, program) != program) == own
        for tokens in (128, 256, 383, 384, 386, 512, 4096):
            for max_q_len in (1, 8, 128, 256, 257, 512):
                if max_q_len > tokens:
                    continue
                takes = (arch == "sm_103a" and tokens >= 384) or (
                    arch == "sm_100a" and max_q_len >= 257
                )
                for enable_pdl in (False, True):
                    expected = twin if takes and not enable_pdl else route
                    assert (
                        cake._pdl_variant(
                            route,
                            arch=arch,
                            enable_pdl=enable_pdl,
                            num_query_tokens=tokens,
                            max_q_len=max_q_len,
                        )
                        == expected
                    ), (tokens, max_q_len, enable_pdl)
        if own:
            spec = get_cake_dsv4_spec(twin, arch=arch)
            assert ("launch", "enable_pdl") not in spec["arg_plan"]
            pdl = get_cake_dsv4_spec(route, arch=arch)["arg_plan"]
            assert spec["arg_plan"] == tuple(
                pair for pair in pdl if pair != ("launch", "enable_pdl")
            )
    for other in ("fp8_lowhead_prefill", "fp8_h64_prefill_source_persistent_m64"):
        assert (
            cake._pdl_variant(
                other,
                arch=arch,
                enable_pdl=False,
                num_query_tokens=512,
                max_q_len=512,
            )
            == other
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
@pytest.mark.parametrize("tokens", [256, 386, 512])
@pytest.mark.parametrize("max_q_len", [8, 257])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_fp8_h128_persistent_launch_takes_the_twin_at_its_floors(
    arch, tokens, max_q_len, enable_pdl
):
    route = "fp8_h128_prefill_source_persistent"
    L = _RecordingLauncher(
        arch,
        Q=torch.empty(0),
        num_query_tokens=tokens,
        num_heads=128,
        sparse_topk=260,
        max_q_len=max_q_len,
        enable_pdl=enable_pdl,
    )
    cake._dispatch_route(route, L)
    takes = (arch == "sm_103a" and tokens >= 384) or (
        arch == "sm_100a" and max_q_len >= 257
    )
    program = f"{route}_no_pdl" if takes and not enable_pdl else route
    assert L.calls == [
        (
            program,
            {"grid": (tokens * 2, 1, 1), "total_work_items": tokens, **L.partials(1)},
        )
    ]


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_sm107a_launches_take_the_k64_sibling_from_its_floor(arch):
    """SM107 routes the FP8 H128 uniform-gather route's K=64 sibling to a
    program of its own, with the route's arguments, and runs it on launches of
    at least 48 rows; SM100 and SM103 route the sibling to the route's program
    and launch the route."""
    from flashinfer.jit.cake_dsv4 import VARIANTS

    assert {"fp8_h128_prefill_source_persistent_uniform"} == cake._K64_SIBLING_ROUTES
    assert cake._K64_SIBLING_MIN_TOKENS == {"sm_107a": 48}
    for route in sorted(cake._K64_SIBLING_ROUTES):
        sibling = f"{route}_k64"
        own = arch == "sm_107a"
        assert (VARIANTS[arch][sibling] != VARIANTS[arch][route]) == own
        for tokens in (1, 8, 32, 47, 48, 64, 96, 127):
            expected = sibling if own and tokens >= 48 else route
            assert (
                cake._k64_variant(
                    route, arch=arch, num_query_tokens=tokens, device=_DEVICE
                )
                == expected
            ), tokens
        assert (
            get_cake_dsv4_spec(sibling, arch=arch)["arg_plan"]
            == get_cake_dsv4_spec(route, arch=arch)["arg_plan"]
        )
    for other in (
        "fp8_h128_prefill_source_persistent",
        "fp8_h64_prefill_source_persistent_m64",
        "fp8_lowhead_prefill",
    ):
        assert (
            cake._k64_variant(other, arch=arch, num_query_tokens=64, device=_DEVICE)
            == other
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
@pytest.mark.parametrize("tokens", [8, 47, 48, 127, 128, 160, 1024])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("launchable", [True, False])
def test_fp8_h128_uniform_launch_takes_the_k64_sibling_at_its_floor(
    monkeypatch, arch, tokens, enable_pdl, launchable
):
    """SM107 runs the uniform route's launches of at least 48 rows on its K=64
    sibling.  Where the process cannot launch the sibling on the query's device
    (the gate is asked only there), launches below the lane-gather floor run
    the route's own program and the others the lane-issued program, over the
    same grid and partials."""
    asked = []

    def gate(gate_arch, device):
        asked.append((gate_arch, device))
        return launchable

    monkeypatch.setattr(cake, "_k64_sibling_launchable", gate)
    route = "fp8_h128_prefill_source_persistent_uniform"
    L = _RecordingLauncher(
        arch,
        Q=torch.empty(0),
        num_query_tokens=tokens,
        num_heads=128,
        sparse_topk=1152,
        max_q_len=4,
        enable_pdl=enable_pdl,
    )
    cake._dispatch_route(route, L)
    sibling = arch == "sm_107a" and tokens >= 48
    if sibling and launchable:
        program = f"{route}_k64"
    elif sibling and tokens >= 128:
        program = "fp8_h128_prefill_source_persistent"
    else:
        program = route
    assert L.calls == [
        (
            program,
            {"grid": (tokens * 2, 1, 1), "total_work_items": tokens, **L.partials(1)},
        )
    ]
    assert asked == ([("sm_107a", _DEVICE)] if sibling else [])


# (sparse_topk, compressed_page_size) of a topk4x, a topk128x and a SWA-only item.
_SHORT_QUERY_ITEMS = {"topk4x": (1152, 64), "topk128x": (1152, 2), "swa": (128, 64)}


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", [None, "sm_100a", "sm_103a", "sm_107a"])
def test_sm107a_short_query_decode_keeps_the_uniform_route(arch):
    """On SM107, FP8 H128 launches of at least 160 rows keep the uniform-gather
    route (its K=64 sibling runs them) when their requests hold one query row
    (at most 2 from 256 rows), or on topk4x items at most 4 (at most 8 from 256
    rows); fewer rows, longer requests and the other architectures take the
    lane-gather route from 128 rows, but SM103 keeps the uniform-gather route
    for SWA-only decode launches of at most 2 query rows per request."""
    assert cake._FP8_PERSISTENT_SWA_DECODE_UNIFORM == {"sm_103a": 2}
    assert cake._FP8_PERSISTENT_SHORT_QUERY_UNIFORM == {
        "sm_107a": ((160, 1, False), (256, 2, False), (160, 4, True), (256, 8, True))
    }
    for item, tokens, max_q_len, sm107_uniform in (
        ("topk4x", 127, 8, True),
        ("topk4x", 128, 1, False),
        ("topk4x", 159, 4, False),
        ("topk4x", 160, 1, True),
        ("topk4x", 160, 4, True),
        ("topk4x", 160, 5, False),
        ("topk4x", 255, 8, False),
        ("topk4x", 256, 8, True),
        ("topk4x", 4096, 8, True),
        ("topk4x", 4096, 9, False),
        ("topk4x", 514, 257, False),
        ("topk4x", 255, 2, True),
        ("topk128x", 127, 4, True),
        ("topk128x", 159, 1, False),
        ("topk128x", 160, 1, True),
        ("topk128x", 1024, 1, True),
        ("topk128x", 255, 2, False),
        ("topk128x", 256, 2, True),
        ("topk128x", 1024, 3, False),
        ("topk128x", 1024, 4, False),
        ("swa", 160, 1, True),
        ("swa", 160, 2, False),
        ("swa", 4096, 2, True),
        ("swa", 384, 4, False),
        ("swa", 1024, 4, False),
        ("swa", 4096, 8, False),
        ("swa", 4096, 3, False),
        ("swa", 514, 257, False),
    ):
        sparse_topk, page_size = _SHORT_QUERY_ITEMS[item]
        uniform = (
            tokens < 128
            or (arch == "sm_107a" and sm107_uniform)
            or (arch == "sm_103a" and item == "swa" and max_q_len <= 2)
        )
        expected = (
            "fp8_h128_prefill_source_persistent_uniform"
            if uniform
            else "fp8_h128_prefill_source_persistent"
        )
        assert (
            _route(
                dtype=torch.float8_e4m3fn,
                num_heads=128,
                batch_size=max(1, tokens // max_q_len),
                max_q_len=max_q_len,
                ragged=True,
                sparse_topk=sparse_topk,
                compressed_page_size=page_size,
                num_query_tokens=tokens,
                sm_count=_SM_COUNT,
                arch=arch,
            )
            == expected
        ), (item, tokens, max_q_len)


@pytest.mark.cpu_only
@pytest.mark.parametrize("tokens", [160, 1024])
def test_sm107a_short_query_decode_runs_the_k64_sibling(tokens):
    """The short-query launches the uniform route keeps on SM107 run its K=64
    sibling over the same grid and partials, and need the workspace the
    lane-gather route needs on the other architectures."""
    route = "fp8_h128_prefill_source_persistent_uniform"
    L = _RecordingLauncher(
        "sm_107a",
        Q=torch.empty(0),
        num_query_tokens=tokens,
        num_heads=128,
        sparse_topk=1152,
        max_q_len=4,
        enable_pdl=True,
    )
    cake._dispatch_route(route, L)
    assert L.calls == [
        (
            f"{route}_k64",
            {"grid": (tokens * 2, 1, 1), "total_work_items": tokens, **L.partials(1)},
        )
    ]
    shape = dict(
        dtype=torch.float8_e4m3fn,
        num_heads=128,
        num_query_tokens=tokens,
        sm_count=_SM_COUNT,
        sparse_topk=1152,
        compressed_page_size=64,
        max_q_len=4,
        batch_size=tokens // 4,
        ragged=False,
    )
    short = cake.cake_dsv4_workspace_requirement(**shape, arch="sm_107a")
    lane = cake.cake_dsv4_workspace_requirement(**shape, arch="sm_103a")
    assert short.route == route
    assert lane.route == "fp8_h128_prefill_source_persistent"
    assert (short.single_launch_bytes, short.minimum_bytes, short.num_splits) == (
        lane.single_launch_bytes,
        lane.minimum_bytes,
        lane.num_splits,
    )


# --------------------------------------------------------------------------- #
# K=64 sibling gate: CUDA 13.4's oversized shared-memory mode                 #
# --------------------------------------------------------------------------- #

_UNIFORM_ROUTE = "fp8_h128_prefill_source_persistent_uniform"


@pytest.fixture
def oversized_check():
    """The host's oversized launch check (both sibling gates ask it), its cache
    per shared-memory size and device empty before and after."""
    cake._oversized_smem_launchable.cache_clear()
    yield cake._oversized_smem_launchable
    cake._oversized_smem_launchable.cache_clear()


@pytest.fixture
def k64_gate(oversized_check):
    """The host's K=64 gate over an empty launch-check cache."""
    return _K64_SIBLING_LAUNCHABLE


@pytest.fixture
def kv4_gate(oversized_check):
    """The host's four-stage gate over an empty launch-check cache."""
    return _KV4_SIBLING_LAUNCHABLE


def _stub_gate_probes(monkeypatch, *, toolkit, runtime, ceiling):
    """Stub the gates' probes: the CUDA toolkit (nvcc) version, the version of
    the process's runtime and the device's oversized ceiling (one value, or a
    dict by device index).  Returns the probe calls."""
    from packaging.version import Version

    from flashinfer.jit import cpp_ext

    asked = {"runtime": [], "ceiling": []}

    def runtime_version(major):
        asked["runtime"].append(major)
        return runtime

    def oversized_ceiling(device):
        asked["ceiling"].append(device)
        return ceiling[device.index] if isinstance(ceiling, dict) else ceiling

    monkeypatch.setattr(cpp_ext, "get_cuda_version", lambda: Version(toolkit))
    monkeypatch.setattr(cake, "_cuda_runtime_version", runtime_version)
    monkeypatch.setattr(cake, "_device_oversized_smem_per_block", oversized_ceiling)
    return asked


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "toolkit,runtime,ceiling,launchable",
    [
        ("13.4", 13040, 272512, True),
        ("13.5", 13050, 300000, True),
        # The driver does not report the ceiling: it has no oversized mode.
        ("13.4", 13040, None, False),
        # A 13.0 runtime wheel next to a 13.4 toolkit: the binding's
        # cudaFuncSetAttribute fails with "invalid argument".
        ("13.4", 13000, 272512, False),
        ("13.4", 13000, None, False),
        # The runtime the JIT modules link does not load.
        ("13.4", None, 272512, False),
        # The oversized ceiling does not hold the program's shared memory.
        ("13.4", 13040, 272511, False),
        ("13.4", 13040, 0, False),
        # The binding needs a 13.4 toolkit to build.
        ("13.3", 13040, 272512, False),
    ],
)
def test_k64_gate_needs_cuda_13_4_and_the_oversized_ceiling(
    monkeypatch, k64_gate, toolkit, runtime, ceiling, launchable
):
    """SM107's K=64 sibling launches only with a CUDA toolkit and a process
    runtime of at least 13.4 and a driver that reports an oversized ceiling of
    at least its 272512 B; the probes stop at the first refusal and the
    runtime is asked by the toolkit's major (the JIT modules' soname)."""
    from packaging.version import Version

    asked = _stub_gate_probes(
        monkeypatch, toolkit=toolkit, runtime=runtime, ceiling=ceiling
    )
    device = torch.device("cuda", 0)
    assert k64_gate("sm_107a", device) is launchable
    built = Version(toolkit) >= Version("13.4")
    loaded = built and runtime is not None and runtime >= 13040
    assert asked == {
        "runtime": [13] if built else [],
        "ceiling": [device] if loaded else [],
    }


@pytest.mark.cpu_only
def test_k64_gate_probes_each_device_once(monkeypatch, k64_gate):
    """The gate's launch check is cached per device: the probes run once per
    device, and an architecture without an oversized sibling needs none."""
    asked = _stub_gate_probes(
        monkeypatch, toolkit="13.4", runtime=13040, ceiling={0: 272512, 1: 232448}
    )
    first, second = torch.device("cuda", 0), torch.device("cuda", 1)
    for _ in range(3):
        assert k64_gate("sm_107a", first)
        assert not k64_gate("sm_107a", second)
    assert k64_gate("sm_100a", first) and k64_gate("sm_103a", second)
    assert asked == {"runtime": [13, 13], "ceiling": [first, second]}


class _FakeCudart:
    """``libcudart`` stand-in whose cudaRuntimeGetVersion reports ``version``
    and returns ``status``."""

    def __init__(self, version, status=0):
        self.version = version
        self.status = status

    def cudaRuntimeGetVersion(self, out):
        out[0] = self.version
        return self.status


class _FakeLibcuda:
    """``libcuda`` stand-in: device ``i`` has the handle ``100 + i`` and every
    attribute asked (recorded with the handle) reads ``ceiling`` with
    ``status``."""

    def __init__(self, ceiling, status=0):
        self.ceiling = ceiling
        self.status = status
        self.asked = []

    def cuInit(self, flags):
        return 0

    def cuDeviceGet(self, out, ordinal):
        out[0] = 100 + ordinal
        return 0

    def cuDeviceGetAttribute(self, out, attribute, handle):
        self.asked.append((attribute, handle.value))
        out[0] = self.ceiling
        return self.status


@pytest.mark.cpu_only
def test_k64_gate_probes_read_the_process_runtime_and_the_driver(monkeypatch):
    """The runtime probe reads cudaRuntimeGetVersion of ``libcudart.so.<major>``;
    the ceiling probe reads the driver's attribute 150 for the device's handle.
    A library that does not load or a call that fails reports nothing."""
    libraries = {}
    opened = []

    def shared_library(soname):
        opened.append(soname)
        return libraries.get(soname)

    monkeypatch.setattr(cake, "_shared_library", shared_library)
    assert cake._cuda_runtime_version(13) is None
    libraries["libcudart.so.13"] = _FakeCudart(13000)
    assert cake._cuda_runtime_version(13) == 13000
    libraries["libcudart.so.13"] = _FakeCudart(13040, status=35)
    assert cake._cuda_runtime_version(13) is None
    assert opened == ["libcudart.so.13"] * 3

    device = torch.device("cuda", 2)
    assert cake._device_oversized_smem_per_block(device) is None
    libcuda = libraries["libcuda.so.1"] = _FakeLibcuda(272512)
    assert cake._device_oversized_smem_per_block(device) == 272512
    assert libcuda.asked == [(150, 102)]
    # CUDA_ERROR_INVALID_VALUE: a driver that does not know the attribute.
    libraries["libcuda.so.1"] = _FakeLibcuda(272512, status=1)
    assert cake._device_oversized_smem_per_block(device) is None


@pytest.mark.cpu_only
def test_shared_library_prefers_the_copy_already_in_the_process(monkeypatch):
    """A soname already mapped resolves without loading (the copy the JIT
    modules bind to); otherwise the dynamic linker loads it; a library that
    loads neither way is ``None``."""
    import ctypes
    import os

    mapped = {"libcudart.so.13"}
    opened = []

    def cdll(soname, mode=ctypes.DEFAULT_MODE):
        opened.append((soname, mode))
        if mode & os.RTLD_NOLOAD and soname not in mapped:
            raise OSError(f"{soname}: not mapped")
        if soname == "libmissing.so.1":
            raise OSError(f"{soname}: not found")
        return f"handle:{soname}"

    monkeypatch.setattr(cake.ctypes, "CDLL", cdll)
    noload = os.RTLD_NOLOAD | os.RTLD_NOW
    assert cake._shared_library("libcudart.so.13") == "handle:libcudart.so.13"
    assert opened == [("libcudart.so.13", noload)]
    opened.clear()
    assert cake._shared_library("libcuda.so.1") == "handle:libcuda.so.1"
    assert opened == [
        ("libcuda.so.1", noload),
        ("libcuda.so.1", ctypes.DEFAULT_MODE),
    ]
    opened.clear()
    assert cake._shared_library("libmissing.so.1") is None
    assert len(opened) == 2


@pytest.mark.cpu_only
@pytest.mark.parametrize("runtime,program", [(13000, ""), (13040, "_k64")])
def test_sm107a_uniform_launch_follows_the_process_runtime(
    monkeypatch, k64_gate, runtime, program
):
    """With a 13.4 toolkit, a 13.0 runtime in the process runs SM107's 48-row
    uniform launches on the route's program instead of failing in the K=64
    sibling's binding; a 13.4 runtime runs the sibling.  The probes run once."""
    from types import SimpleNamespace

    from packaging.version import Version

    from flashinfer.jit import cpp_ext

    monkeypatch.setattr(cake, "_k64_sibling_launchable", k64_gate)
    monkeypatch.setattr(cpp_ext, "get_cuda_version", lambda: Version("13.4"))
    cudart, libcuda = _FakeCudart(runtime), _FakeLibcuda(272512)
    libraries = {"libcudart.so.13": cudart, "libcuda.so.1": libcuda}
    monkeypatch.setattr(cake, "_shared_library", libraries.get)
    for _ in range(2):
        L = _RecordingLauncher(
            "sm_107a",
            Q=SimpleNamespace(device=torch.device("cuda", 0)),
            num_query_tokens=48,
            num_heads=128,
            sparse_topk=1152,
            max_q_len=4,
        )
        cake._dispatch_route(_UNIFORM_ROUTE, L)
        assert L.calls == [
            (
                f"{_UNIFORM_ROUTE}{program}",
                {"grid": (96, 1, 1), "total_work_items": 48, **L.partials(1)},
            )
        ]
    assert libcuda.asked == ([(150, 100)] if runtime >= 13040 else [])


@pytest.mark.cpu_only
def test_sibling_gates_know_every_oversized_binding():
    """The programs whose generated bindings launch in CUDA 13.4's oversized
    shared-memory mode are exactly the K=64 and four-stage siblings the gates
    hold, the SM107 split producer the K=64 gate holds and the four-stage
    sibling's TMEM-batch and L2 prefetch siblings where they have programs of
    their own (they launch only where the four-stage gate passed), each with
    the dynamic
    shared memory its gate compares with the device's oversized ceiling; every
    other program, including those the gates fall back to, launches in the
    standard mode."""
    import re

    from flashinfer.jit.cake_dsv4 import VARIANTS, _get_csrc_dir

    siblings = (
        ("_k64", cake._K64_SIBLING_ROUTES, cake._K64_SIBLING_OVERSIZED_SMEM_BYTES),
        ("_kv4", cake._KV4_SIBLING_ROUTES, cake._KV4_SIBLING_OVERSIZED_SMEM_BYTES),
    )
    gated = {
        VARIANTS[arch][f"{route}{suffix}"]: smem
        for suffix, routes, smem_by_arch in siblings
        for arch, smem in smem_by_arch.items()
        for route in routes
    }
    # The split-KV producer launches only where the K=64 gate passes
    # (cake._k64_split), so it shares the K=64 sibling's shared memory.
    for arch, smem in cake._K64_SIBLING_OVERSIZED_SMEM_BYTES.items():
        split = VARIANTS[arch].get(cake._K64_SPLIT_VARIANT)
        if split is not None and split != VARIANTS[arch][cake._K64_SPLIT_ROUTE]:
            gated[split] = smem
    # The TMEM-batch sibling launches only where the four-stage gate passes
    # (cake._kv4_bcl_variant), so it shares the four-stage sibling's.
    for arch, smem in cake._KV4_SIBLING_OVERSIZED_SMEM_BYTES.items():
        for route in cake._KV4_SIBLING_ROUTES:
            four_stage = VARIANTS[arch][f"{route}_kv4"]
            bcl = VARIANTS[arch].get(f"{route}_kv4_bcl", four_stage)
            if bcl != four_stage:
                gated[bcl] = smem
    # So does the L2 prefetch sibling (cake._kv4_l2pf_variant).
    for arch, smem in cake._KV4_SIBLING_OVERSIZED_SMEM_BYTES.items():
        for route in cake._KV4_SIBLING_ROUTES:
            four_stage = VARIANTS[arch][f"{route}_kv4"]
            l2pf = VARIANTS[arch].get(f"{route}_kv4_l2pf", four_stage)
            if l2pf != four_stage:
                gated[l2pf] = smem
    oversized = {}
    for path in sorted(_get_csrc_dir().glob("*/*_binding.cu")):
        text = path.read_text(encoding="utf-8")
        if "cudaSharedMemoryModeAllowOversizedSharedMemory" in text:
            (smem,) = re.findall(r"config\.dynamicSmemBytes = (\d+);", text)
            oversized[path.name.removesuffix("_binding.cu")] = int(smem)
    assert oversized == gated


# --------------------------------------------------------------------------- #
# Four-stage sibling: SM107's multi-tile M64 program in the oversized mode    #
# --------------------------------------------------------------------------- #

_M64_MULTI_TILE_ROUTE = "fp8_h64_prefill_source_persistent_m64_multi_tile"


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_sm107a_m64_multi_tile_launches_take_the_kv4_sibling(arch):
    """SM107 routes the multi-tile M64 route's four-stage sibling to a program
    of its own, with the route's arguments, and runs it on every launch; SM100
    and SM103 route the sibling to the route's program and launch the route."""
    from flashinfer.jit.cake_dsv4 import VARIANTS

    assert {_M64_MULTI_TILE_ROUTE} == cake._KV4_SIBLING_ROUTES
    assert cake._KV4_SIBLING_OVERSIZED_SMEM_BYTES == {"sm_107a": 322688}
    for route in sorted(cake._KV4_SIBLING_ROUTES):
        sibling = f"{route}_kv4"
        own = arch == "sm_107a"
        assert (VARIANTS[arch][sibling] != VARIANTS[arch][route]) == own
        expected = sibling if own else route
        assert cake._kv4_variant(route, arch=arch, device=_DEVICE) == expected
        assert (
            get_cake_dsv4_spec(sibling, arch=arch)["arg_plan"]
            == get_cake_dsv4_spec(route, arch=arch)["arg_plan"]
        )
    for other in (
        "fp8_h64_prefill_source_persistent_m64",
        "fp8_h128_prefill_source_persistent_uniform",
        "fp8_lowhead_prefill",
    ):
        assert cake._kv4_variant(other, arch=arch, device=_DEVICE) == other


@pytest.mark.cpu_only
def test_kv4_variant_needs_a_sibling_program_of_its_own(monkeypatch):
    """A launch takes the four-stage sibling only where its architecture routes
    the sibling to a program other than the route's: a table without the
    sibling, one that routes it to the route's program and one of another
    architecture run the route without asking the gate.  The FP8/H64 decode
    rows take the four-stage edges exactly where the launch takes the sibling,
    through the same check."""
    import flashinfer.jit.cake_dsv4 as jit

    asked = []

    def gate(arch, device):
        asked.append((arch, device))
        return True

    monkeypatch.setattr(cake, "_kv4_sibling_launchable", gate)
    route, sibling = _M64_MULTI_TILE_ROUTE, f"{_M64_MULTI_TILE_ROUTE}_kv4"
    two, four = "cake_dsv4_two_stage", "cake_dsv4_four_stage"
    for table, expected in (
        ({"sm_107a": {route: two, sibling: four}}, sibling),
        ({"sm_107a": {route: two}}, route),
        ({"sm_107a": {route: two, sibling: two}}, route),
        ({"sm_103a": {route: two, sibling: four}}, route),
    ):
        monkeypatch.setattr(jit, "VARIANTS", table)
        assert cake._kv4_variant(route, arch="sm_107a", device=_DEVICE) == expected
        edges = cake._fp8_h64_kv4_edges(
            dtype=torch.float8_e4m3fn,
            num_heads=64,
            max_q_len=1,
            sparse_topk=384,
            arch="sm_107a",
            device=_DEVICE,
        )
        assert edges == (expected == sibling)
    assert asked == [("sm_107a", _DEVICE)] * 2


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
@pytest.mark.parametrize(
    "route", ["fp8_h64_prefill_source_persistent_m64", _M64_MULTI_TILE_ROUTE]
)
@pytest.mark.parametrize("tokens", [1, 64, 208, 4096])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("launchable", [True, False])
def test_fp8_h64_m64_launch_takes_the_kv4_sibling(
    monkeypatch, arch, route, tokens, enable_pdl, launchable
):
    """SM107 runs every multi-tile M64 launch of one-token requests (outside
    the TMEM-batch sibling's window) on its four-stage sibling over the same
    grid and partials.  Where the process cannot launch the sibling on the
    query's device (the gate is asked only there), the route's program runs;
    the single-tile program and SM100 and SM103 launch the route."""
    asked = []

    def gate(gate_arch, device):
        asked.append((gate_arch, device))
        return launchable

    monkeypatch.setattr(cake, "_kv4_sibling_launchable", gate)
    L = _RecordingLauncher(
        arch,
        Q=torch.empty(0),
        num_query_tokens=tokens,
        num_heads=64,
        sparse_topk=388,
        enable_pdl=enable_pdl,
        # One-token requests: outside the TMEM-batch sibling's window.
        max_q_len=1,
    )
    cake._dispatch_route(route, L)
    sibling = arch == "sm_107a" and route == _M64_MULTI_TILE_ROUTE
    program = f"{route}_kv4" if sibling and launchable else route
    assert L.calls == [
        (program, {"grid": (tokens, 1, 1), "total_work_items": tokens, **L.partials(1)})
    ]
    assert asked == ([("sm_107a", _DEVICE)] if sibling else [])


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "toolkit,runtime,ceiling,launchable",
    [
        # SM107's oversized ceiling, and one that holds exactly 322688 B.
        ("13.4", 13040, 334848, True),
        ("13.5", 13050, 334848, True),
        ("13.4", 13040, 322688, True),
        # The ceiling holds the K=64 sibling's 272512 B but not 322688 B.
        ("13.4", 13040, 322687, False),
        ("13.4", 13040, 272512, False),
        # The driver does not report the ceiling: it has no oversized mode.
        ("13.4", 13040, None, False),
        # A 13.0 runtime wheel next to a 13.4 toolkit.
        ("13.4", 13000, 334848, False),
        # The runtime the JIT modules link does not load.
        ("13.4", None, 334848, False),
        # The binding needs a 13.4 toolkit to build.
        ("13.3", 13040, 334848, False),
    ],
)
def test_kv4_gate_needs_cuda_13_4_and_the_oversized_ceiling(
    monkeypatch, kv4_gate, toolkit, runtime, ceiling, launchable
):
    """SM107's four-stage sibling launches under the K=64 sibling's conditions
    with an oversized ceiling of at least its 322688 B; the probes run as for
    the K=64 sibling."""
    from packaging.version import Version

    asked = _stub_gate_probes(
        monkeypatch, toolkit=toolkit, runtime=runtime, ceiling=ceiling
    )
    device = torch.device("cuda", 0)
    assert kv4_gate("sm_107a", device) is launchable
    built = Version(toolkit) >= Version("13.4")
    loaded = built and runtime is not None and runtime >= 13040
    assert asked == {
        "runtime": [13] if built else [],
        "ceiling": [device] if loaded else [],
    }


@pytest.mark.cpu_only
def test_sibling_gates_share_the_launch_check(monkeypatch, k64_gate, kv4_gate):
    """Both gates ask the one launch check, cached per shared-memory size and
    device: a ceiling that holds the K=64 sibling's 272512 B but not the
    four-stage sibling's 322688 B launches only the former, each size probes
    each device once, and SM100 and SM103 need no probe."""
    asked = _stub_gate_probes(
        monkeypatch, toolkit="13.4", runtime=13040, ceiling={0: 334848, 1: 300000}
    )
    first, second = torch.device("cuda", 0), torch.device("cuda", 1)
    for _ in range(3):
        assert k64_gate("sm_107a", first) and kv4_gate("sm_107a", first)
        assert k64_gate("sm_107a", second) and not kv4_gate("sm_107a", second)
    assert kv4_gate("sm_100a", first) and kv4_gate("sm_103a", second)
    assert asked == {"runtime": [13] * 4, "ceiling": [first, first, second, second]}


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "runtime,ceiling,program",
    [
        (13040, 334848, "_kv4"),
        # A 13.0 runtime wheel next to a 13.4 toolkit.
        (13000, 334848, ""),
        # A ceiling that holds the K=64 sibling but not the four-stage one.
        (13040, 300000, ""),
    ],
)
def test_sm107a_m64_multi_tile_launch_follows_the_process_runtime(
    monkeypatch, kv4_gate, runtime, ceiling, program
):
    """With a 13.4 toolkit, SM107's multi-tile M64 launches run the four-stage
    sibling where the process runtime is at least 13.4 and the device's
    oversized ceiling holds its 322688 B, and the route's two-stage program
    otherwise instead of failing in the sibling's binding.  The probes run
    once."""
    from types import SimpleNamespace

    from packaging.version import Version

    from flashinfer.jit import cpp_ext

    monkeypatch.setattr(cake, "_kv4_sibling_launchable", kv4_gate)
    monkeypatch.setattr(cpp_ext, "get_cuda_version", lambda: Version("13.4"))
    cudart, libcuda = _FakeCudart(runtime), _FakeLibcuda(ceiling)
    libraries = {"libcudart.so.13": cudart, "libcuda.so.1": libcuda}
    monkeypatch.setattr(cake, "_shared_library", libraries.get)
    for _ in range(2):
        L = _RecordingLauncher(
            "sm_107a",
            Q=SimpleNamespace(device=torch.device("cuda", 0)),
            num_query_tokens=208,
            num_heads=64,
            sparse_topk=388,
            # One-token requests: outside the TMEM-batch sibling's window.
            max_q_len=1,
        )
        cake._dispatch_route(_M64_MULTI_TILE_ROUTE, L)
        assert L.calls == [
            (
                f"{_M64_MULTI_TILE_ROUTE}{program}",
                {"grid": (208, 1, 1), "total_work_items": 208, **L.partials(1)},
            )
        ]
    assert libcuda.asked == ([(150, 100)] if runtime >= 13040 else [])


# --------------------------------------------------------------------------- #
# SM107 FP8/H64 decode routes where the four-stage sibling launches           #
# --------------------------------------------------------------------------- #

# The token counts up to the half wave that take the multi-tile M64 body on
# SM107 (FP8, 64 heads, max_q_len < 257), per width, per SM count and without
# or with the four-stage edges ("-": none, the half wave).  The other token
# counts up to the half wave run the persistent body, every token count past
# it the M64 body.
_SM107_KV4_M64_SPANS = """
widths          | 208         | 208 kv4           | 216         | 216 kv4           | 224 kv4
176             | 1-12 17-104 | 1-104             | 1-12 17-108 | 1-108             | 1-112
177 191         | 1-12 17-104 | 1-12 17-104       | 1-12 17-108 | 1-12 17-108       | 1-12 17-112
192 255         | 1-12 32-104 | 1-12 32-104       | 1-12 33-108 | 1-12 33-108       | 1-12 34-112
256             | 1-12 27-104 | 1-104             | 1-12 28-108 | 1-108             | 1-112
257 288 289 352 | 1-12 54-104 | 1-16 24-47 52-104 | 1-12 56-108 | 1-16 24-47 54-108 | 1-16 24-47 56-112
353 383         | 1-12 54-104 | 1-12 44-47 52-104 | 1-12 56-108 | 1-12 44-47 54-108 | 1-12 44-47 56-112
384             | 53-104      | 1-104             | 55-108      | 1-108             | 1-112
385 511         | 63-104      | 60-104            | 65-108      | 63-108            | 65-112
512             | 55-104      | 40-104            | 57-108      | 41-108            | 43-112
513 639         | -           | 64-104            | -           | 67-108            | 69-112
640             | -           | 32-47 56-104      | -           | 32-47 58-108      | 32-47 60-112
641 767         | -           | 59-104            | -           | 62-108            | 64-112
768             | -           | 55-104            | -           | 58-108            | 60-112
769 895         | -           | 77-104            | -           | 80-108            | 83-112
896             | -           | 63-104            | -           | 66-108            | 68-112
897 1023        | -           | 78-104            | -           | 81-108            | 83-112
1024            | -           | 81-104            | -           | 84-108            | 87-112
1025 1151       | -           | 91-104            | -           | 94-108            | 98-112
1152            | -           | 77-104            | -           | 79-108            | 82-112
"""


def _span_tokens(cell):
    """The token counts of ``"a-b c-d"`` spans (``"-"``: none)."""
    tokens = set()
    for span in cell.split():
        if span != "-":
            first, last = span.split("-")
            tokens.update(range(int(first), int(last) + 1))
    return tokens


def _sm107_kv4_m64_spans():
    """``_SM107_KV4_M64_SPANS`` as ``(widths, {(sm_count, kv4): M64 tokens})`` rows."""
    header, *rows = _SM107_KV4_M64_SPANS.strip().splitlines()
    columns = [
        (int(column.split()[0]), column.split()[-1] == "kv4")
        for column in header.split("|")[1:]
    ]
    table = []
    for row in rows:
        widths, *cells = row.split("|")
        spans = {
            column: _span_tokens(cell)
            for column, cell in zip(columns, cells, strict=True)
        }
        table.append((tuple(int(width) for width in widths.split()), spans))
    return table


def _sm107_kv4_widths():
    """The widths of ``_SM107_KV4_M64_SPANS``: every edge of the SM107 rules."""
    return sorted({width for widths, _ in _sm107_kv4_m64_spans() for width in widths})


def _h64_decode_route(width, tokens, sm_count, arch, **kwargs):
    """``_route`` of a dense FP8/H64 decode call of ``tokens`` one-row requests."""
    return _route(
        dtype=torch.float8_e4m3fn,
        num_heads=64,
        max_q_len=1,
        ragged=False,
        sparse_topk=width,
        batch_size=tokens,
        compressed_page_size=64,
        num_query_tokens=tokens,
        sm_count=sm_count,
        arch=arch,
        **kwargs,
    )


@pytest.mark.cpu_only
def test_fp8_h64_decode_rows_route_by_the_kv4_edges_on_sm107():
    """On SM107 every FP8/H64 decode row up to two tokens past the half wave
    (208, 216 and 224 SMs) routes as ``_SM107_KV4_M64_SPANS`` lists, without
    and with the four-stage edges.  They take widths 176, 256 and 384 at every
    token count; widths 257-352 at 1-16 and 24-47 tokens and widths 353-383 at
    44-47 tokens, below the uniform route's K=64 floor (then the persistent
    body again until 52 tokens at 208 SMs); widths 385-512 earlier; and
    widths 513-1152 before the half wave, width 640 also at 32-47 tokens."""
    table = _sm107_kv4_m64_spans()
    assert len(_sm107_kv4_widths()) == sum(len(widths) for widths, _ in table)
    for widths, spans in table:
        for (sm_count, kv4), m64_tokens in spans.items():
            assert max(m64_tokens, default=0) <= sm_count // 2, (widths, sm_count)
            for width in widths:
                for tokens in range(1, sm_count // 2 + 3):
                    m64 = tokens in m64_tokens or 2 * tokens > sm_count
                    expected = _M64_MULTI_TILE_ROUTE if m64 else _UNIFORM_ROUTE
                    route = _h64_decode_route(
                        width, tokens, sm_count, "sm_107a", kv4=kv4
                    )
                    assert route == expected, (width, sm_count, kv4, tokens)


@pytest.mark.cpu_only
def test_kv4_edges_only_promote_sm107a_h64_decode_rows():
    """The four-stage edges only move SM107 FP8/H64 decode rows of more than
    one tile from the persistent body to the multi-tile M64 body: the three
    tables key SM107 alone, no M64 edge comes later than today's, no
    short-window width keeps a row persistent that today runs on the M64 body,
    the windows below the K=64 floor start past 16 tokens and below that
    floor, on disjoint widths of a key with an edge and two or more complete
    tiles (rows the persistent body runs past 12 tokens), and the other
    architectures, head counts, BF16 and prefill rows route as without
    them."""
    import itertools

    short = cake._FP8_H64_PERSISTENT_SHORT_MIN_WIDTH_KV4
    early = cake._FP8_H64_M64_EARLY_WAVE_KV4
    band = cake._FP8_H64_M64_BELOW_K64_KV4
    assert set(short) == set(early) == set(band) == {"sm_107a"}
    for arch, edges in early.items():
        base = cake._FP8_H64_M64_EARLY_WAVE[arch]
        for tiles, (num, den) in edges.items():
            assert tiles[0] >= 2, tiles
            if tiles in base:
                base_num, base_den = base[tiles]
                assert num * base_den <= base_num * den, tiles
    for arch, keys in band.items():
        floor = cake._K64_SIBLING_MIN_TOKENS[arch]
        for tiles, windows in keys.items():
            assert tiles in early[arch] and tiles[1] >= 2, tiles
            last = 0
            for min_width, max_width, first in sorted(windows):
                assert last < min_width <= max_width, (tiles, min_width)
                for width in (min_width, max_width):
                    assert (-(-width // 128), width // 128) == tiles, width
                assert 17 <= first < floor, (tiles, first)
                last = max_width
    for arch, entries in short.items():
        # Today the one-tile rows run the persistent body up to 16 tokens from
        # the one-tile short-window width on; wider rows from two tiles on.
        assert entries[1] >= cake._FP8_H64_PERSISTENT_ONE_TILE_MIN_WIDTH[arch][1]
    moved = 0
    sweeps = [("sm_107a", range(129, 1153))]
    sweeps += [(arch, _sm107_kv4_widths()) for arch in ("sm_100a", "sm_103a", None)]
    for arch, widths in sweeps:
        for sm_count in (208, 216):
            for width in widths:
                for tokens in range(1, sm_count // 2 + 3):
                    old = _h64_decode_route(width, tokens, sm_count, arch, kv4=False)
                    new = _h64_decode_route(width, tokens, sm_count, arch, kv4=True)
                    if new != old:
                        assert arch == "sm_107a", (arch, sm_count, width, tokens)
                        assert (old, new) == (_UNIFORM_ROUTE, _M64_MULTI_TILE_ROUTE)
                        moved += 1
    assert moved

    def outcome(**shape):
        try:
            return _route(**shape)
        except ValueError as error:
            return str(error)

    shapes = itertools.product(
        ("sm_100a", "sm_103a", "sm_107a", None),
        (torch.float8_e4m3fn, torch.bfloat16),
        (8, 16, 32, 64, 128),
        (1, 4, 257),
        (128, 129, 256, 384, 640, 1152),
        (2, 64),
        (1, 12, 16, 52, 104, 109, 257, 514),
    )
    for arch, dtype, num_heads, max_q_len, width, page_size, tokens in shapes:
        h64_decode = (
            dtype == torch.float8_e4m3fn and num_heads == 64 and max_q_len < 257
        )
        if h64_decode or max_q_len > tokens:
            continue
        shape = dict(
            dtype=dtype,
            num_heads=num_heads,
            max_q_len=max_q_len,
            ragged=max_q_len > 1,
            sparse_topk=width,
            batch_size=-(-tokens // max_q_len),
            compressed_page_size=page_size,
            num_query_tokens=tokens,
            sm_count=216,
            arch=arch,
        )
        assert outcome(**shape, kv4=True) == outcome(**shape, kv4=False), shape


# Today's SM107 early-wave edges of the FP8/H64 decode rows, frozen.
_TODAYS_SM107_H64_EARLY_WAVE = {
    (2, 1): (3, 20),
    (2, 2): (1, 8),
    (3, 2): (51, 200),
    (3, 3): (1, 4),
    (4, 3): (3, 10),
    (4, 4): (13, 50),
}


def _todays_h64_decode_route(width, tokens, sm_count, arch):
    """The FP8/H64 decode route (``max_q_len < 257``) before the four-stage
    edges, restated: the M64 body past the half wave and from today's SM107
    edges on; else the persistent body for three complete tiles up to 12
    tokens and for two past them (SM103: also widths 257-383 up to 12 tokens;
    SM107: also one complete tile from width 192, from 176 up to 16 tokens);
    else the SWA producer (width 128), the H64 cluster body on SM100 (widths
    257-383 up to 12 tokens) or the M64 body."""
    m64 = _M64_MULTI_TILE_ROUTE
    if width == 128:
        m64 = "fp8_h64_prefill_source_persistent_m64"
    full_tiles = width // 128
    edge = None
    if arch == "sm_107a":
        edge = _TODAYS_SM107_H64_EARLY_WAVE.get((-(-width // 128), full_tiles))
    if 2 * tokens > sm_count or (edge and edge[1] * tokens > edge[0] * sm_count):
        return m64
    if tokens <= 12:
        persistent = full_tiles >= 3 or (
            arch == "sm_103a" and full_tiles == 2 and width > 256
        )
    elif full_tiles == 1 and arch == "sm_107a":
        persistent = width >= (176 if tokens <= 16 else 192)
    else:
        persistent = full_tiles >= 2
    if persistent and tokens < 128:
        return _UNIFORM_ROUTE
    if persistent:
        return "fp8_h128_prefill_source_persistent"
    if width == 128:
        return "fp8_lowhead_prefill"
    if arch == "sm_100a" and tokens <= 12 and 256 < width < 384:
        return "fp8_lowhead_h64"
    return m64


@pytest.mark.cpu_only
def test_kv4_false_routes_are_todays():
    """Without the four-stage edges (``kv4=False``, the default: the sibling
    does not launch, or another architecture) the FP8/H64 decode rows route as
    before them: today's tables are unchanged and every row of the sweep
    routes as the restated rule."""
    assert cake._FP8_H64_M64_EARLY_WAVE == {"sm_107a": _TODAYS_SM107_H64_EARLY_WAVE}
    assert cake._FP8_H64_PERSISTENT_ONE_TILE_MIN_WIDTH == {"sm_107a": (192, 176)}
    edges = sorted({128, 129, 175, *_sm107_kv4_widths()})
    for arch in ("sm_100a", "sm_103a", "sm_107a", None):
        widths = range(128, 1153) if arch == "sm_107a" else edges
        for sm_count in (208, 216):
            for width in widths:
                for tokens in range(1, sm_count // 2 + 3):
                    expected = _todays_h64_decode_route(width, tokens, sm_count, arch)
                    default = _h64_decode_route(width, tokens, sm_count, arch)
                    off = _h64_decode_route(width, tokens, sm_count, arch, kv4=False)
                    assert default == off == expected, (arch, sm_count, width, tokens)


# SM107's previous four-stage edges of the FP8/H64 decode rows (no windows
# below the K=64 floor), frozen.
_PREVIOUS_SM107_H64_EARLY_WAVE_KV4 = {
    (2, 1): (3, 20),
    (2, 2): (0, 1),
    (3, 2): (14, 57),
    (3, 3): (0, 1),
    (4, 3): (3, 10),
    (4, 4): (19, 79),
    (7, 7): (77, 208),
    (8, 8): (23, 52),
    (9, 9): (77, 208),
}


@pytest.mark.cpu_only
def test_kv4_tables_only_promote_relative_to_the_previous_kv4_tables(monkeypatch):
    """Relative to the previous four-stage edges (frozen above, without
    windows below the K=64 floor), the current tables move SM107 FP8/H64
    decode rows of widths 129-1152 only from the uniform route to the
    multi-tile M64 route: 22775, 23411 and 23285 rows up to two tokens past
    the half wave at 208, 212 and 216 SMs."""
    sm_counts = (208, 212, 216)
    current = {
        (sm_count, width, tokens): _h64_decode_route(
            width, tokens, sm_count, "sm_107a", kv4=True
        )
        for sm_count in sm_counts
        for width in range(129, 1153)
        for tokens in range(1, sm_count // 2 + 3)
    }
    monkeypatch.setattr(
        cake,
        "_FP8_H64_M64_EARLY_WAVE_KV4",
        {"sm_107a": _PREVIOUS_SM107_H64_EARLY_WAVE_KV4},
    )
    monkeypatch.setattr(cake, "_FP8_H64_M64_BELOW_K64_KV4", {})
    moved = dict.fromkeys(sm_counts, 0)
    for row, new in current.items():
        sm_count, width, tokens = row
        old = _h64_decode_route(width, tokens, sm_count, "sm_107a", kv4=True)
        if new != old:
            assert (old, new) == (_UNIFORM_ROUTE, _M64_MULTI_TILE_ROUTE), row
            moved[sm_count] += 1
    assert moved == {208: 22775, 212: 23411, 216: 23285}


def _bf16_out_at(rows, num_heads, offset):
    """A bf16 ``[rows, num_heads, 512]`` output whose data pointer is
    ``offset`` bytes past a 32-byte boundary."""
    numel = rows * num_heads * 512
    backing = torch.empty(numel + 16, dtype=torch.bfloat16)
    start = ((offset - backing.data_ptr()) % 32) // 2
    return backing[start : start + numel].view(rows, num_heads, 512)


def _run_fake_h64_decode_fp8(
    monkeypatch,
    *,
    arch,
    width,
    tokens,
    workspace=None,
    out=None,
    max_q_len=1,
    sm_count=208,
    page=64,
    batch_size=None,
    cum_seq_lens_q=None,
):
    """Drive run_cake_dsv4 on CPU tensors for a dense FP8/H64 decode call of
    ``tokens`` rows in requests of ``max_q_len`` rows (default one) over
    ``width`` keys on ``arch`` (``sm_count`` SMs, default 208), the
    compressed cache on pages of ``page`` rows (default 64); return the
    launched programs, their bound arguments and the output.
    ``batch_size`` (default ``tokens // max_q_len``) is the request count
    and ``cum_seq_lens_q`` the request offsets of a ragged call."""
    plans = {
        route: _PERSISTENT_PLAN
        for route in (
            _UNIFORM_ROUTE,
            f"{_UNIFORM_ROUTE}_k64",
            _M64_MULTI_TILE_ROUTE,
            f"{_M64_MULTI_TILE_ROUTE}_kv4",
            f"{_M64_MULTI_TILE_ROUTE}_kv4_bcl",
        )
    }
    plans[f"{_M64_MULTI_TILE_ROUTE}_kv4_l2pf"] = _L2PF_PLAN
    if batch_size is None:
        batch_size = tokens // max_q_len
    recorder = _install_fake_variants(monkeypatch, plans)
    programs = []

    def module(variant, **kwargs):
        programs.append(variant)
        return recorder

    monkeypatch.setattr(cake, "_variant_module", module)
    monkeypatch.setattr(cake, "_target_arch", lambda device: arch)
    monkeypatch.setattr(cake, "_device_sm_count", lambda device: sm_count)
    table, lens = _combined_metadata(tokens, width - 128)
    if workspace is None:
        workspace = _aligned_u8(
            get_cake_dsv4_workspace_bytes(tokens, 64, width, torch.float8_e4m3fn)
        )
    result = cake.run_cake_dsv4(
        query=torch.zeros((tokens, 64, 512), dtype=torch.float8_e4m3fn),
        swa_kv_cache=torch.zeros((4, 1, 256, 512), dtype=torch.float8_e4m3fn),
        compressed_kv_cache=torch.zeros((4, 1, page, 512), dtype=torch.float8_e4m3fn),
        workspace_buffer=workspace,
        out=out,
        bmm1_scale=0.5,
        bmm2_scale=1.0,
        sinks=None,
        max_q_len=max_q_len,
        cum_seq_lens_q=cum_seq_lens_q,
        seq_lens=torch.full((batch_size,), 1000, dtype=torch.int32),
        backend="cake",
        sparse_indices=table,
        sparse_topk_lens=lens,
    )
    calls = [
        dict(zip((name for _, name in plans[program]), call, strict=True))
        for program, call in zip(programs, recorder.calls, strict=True)
    ]
    return programs, calls, result


@pytest.mark.cpu_only
@pytest.mark.parametrize("width,tokens", [(384, 4), (256, 20)])
def test_run_cake_dsv4_routes_h64_decode_rows_by_the_kv4_launch_check(
    monkeypatch, width, tokens
):
    """run_cake_dsv4 routes SM107's FP8/H64 decode rows by the four-stage
    edges where the process can launch the sibling on the query's device:
    rows they promote (208 SMs) run the sibling, one CTA per token, after one
    gate ask at route time and one at launch.  Where it cannot, they keep the
    uniform route, two CTAs per token, after one ask; SM103 never asks.  A
    caller ``out`` 16 bytes past a 32-byte boundary is staged for the M64
    route only (its stores need 32 bytes, the uniform route's 16)."""
    answer = {}
    asked = []

    def gate(arch, device):
        asked.append((arch, device))
        return answer["launchable"]

    monkeypatch.setattr(cake, "_kv4_sibling_launchable", gate)
    for arch, launchable, program, grid, asks, staged in (
        ("sm_107a", True, f"{_M64_MULTI_TILE_ROUTE}_kv4", tokens, 2, True),
        ("sm_107a", False, _UNIFORM_ROUTE, 2 * tokens, 1, False),
        ("sm_103a", True, _UNIFORM_ROUTE, 2 * tokens, 0, False),
    ):
        answer["launchable"] = launchable
        asked.clear()
        out = _bf16_out_at(tokens, 64, 16)
        assert out.data_ptr() % 32 == 16
        programs, calls, result = _run_fake_h64_decode_fp8(
            monkeypatch, arch=arch, width=width, tokens=tokens, out=out
        )
        (bound,) = calls
        assert programs == [program], (arch, launchable)
        assert (bound["grid_x"], bound["grid_y"], bound["grid_z"]) == (grid, 1, 1)
        assert bound["total_work_items"] == bound["num_query_tokens"] == tokens
        assert asked == [("sm_107a", _DEVICE)] * asks
        assert result is out
        assert (bound["O"].data_ptr() != out.data_ptr()) == staged


@pytest.mark.cpu_only
def test_row_tiled_h64_decode_chunks_run_the_kv4_sibling(monkeypatch):
    """A workspace below the single-launch carve tiles a promoted FP8/H64
    decode call (width 384, 20 tokens, 208 SMs): every chunk runs the
    four-stage sibling, one CTA per row, and the gate is asked once at route
    time and once per chunk."""
    asked = []

    def gate(arch, device):
        asked.append((arch, device))
        return True

    monkeypatch.setattr(cake, "_kv4_sibling_launchable", gate)
    tokens = 20
    plan = cake._route_plan(
        _M64_MULTI_TILE_ROUTE, num_query_tokens=tokens, num_heads=64, sparse_topk=384
    )
    chunk_bytes = cake._launch_workspace_bytes(plan, 8, 64, tokens + 1)
    rows = cake._rows_per_launch(
        plan,
        num_query_tokens=tokens,
        num_heads=64,
        batch_size=tokens,
        workspace_bytes=chunk_bytes,
    )
    assert rows == 8
    programs, calls, _ = _run_fake_h64_decode_fp8(
        monkeypatch,
        arch="sm_107a",
        width=384,
        tokens=tokens,
        workspace=_aligned_u8(chunk_bytes),
    )
    assert programs == [f"{_M64_MULTI_TILE_ROUTE}_kv4"] * 3
    grids = [(c["grid_x"], c["grid_y"], c["grid_z"]) for c in calls]
    assert grids == [(8, 1, 1), (8, 1, 1), (4, 1, 1)]
    assert [c["total_work_items"] for c in calls] == [8, 8, 4]
    assert asked == [("sm_107a", _DEVICE)] * 4


@pytest.mark.cpu_only
@pytest.mark.parametrize("width,tokens", [(384, 48), (512, 52), (640, 56)])
def test_k64_only_ceiling_keeps_todays_h64_decode_routes(
    monkeypatch, k64_gate, kv4_gate, width, tokens
):
    """A device whose oversized ceiling holds the K=64 sibling's 272512 B but
    not the four-stage sibling's 322688 B routes without the four-stage edges:
    rows they would promote (208 SMs) keep the uniform route and run its K=64
    sibling from 48 rows, two CTAs per token."""
    monkeypatch.setattr(cake, "_k64_sibling_launchable", k64_gate)
    monkeypatch.setattr(cake, "_kv4_sibling_launchable", kv4_gate)
    _stub_gate_probes(monkeypatch, toolkit="13.4", runtime=13040, ceiling=300000)
    promoted = _h64_decode_route(width, tokens, 208, "sm_107a", kv4=True)
    assert promoted == _M64_MULTI_TILE_ROUTE
    programs, calls, _ = _run_fake_h64_decode_fp8(
        monkeypatch, arch="sm_107a", width=width, tokens=tokens
    )
    assert programs == [f"{_UNIFORM_ROUTE}_k64"]
    (bound,) = calls
    assert (bound["grid_x"], bound["grid_y"], bound["grid_z"]) == (2 * tokens, 1, 1)
    assert bound["total_work_items"] == tokens


@pytest.mark.cpu_only
def test_sm107a_kv4_band_ends_at_the_k64_floor(monkeypatch):
    """With the four-stage edges, SM107's FP8/H64 decode rows of widths
    257-352 take the M64 body from 24 tokens, those of widths 353-383 from
    44 tokens and those of width 640 from 32 tokens up to the uniform route's
    K=64 floor (48 rows, read when the call routes), then the persistent body
    until the early edge; without them, and on the other architectures, the
    rows route as before.  At 208 SMs those rows launch the four-stage
    sibling, one CTA per token; the rows from the floor run the uniform
    route's K=64 sibling and the rows of widths 353-383 below 44 tokens its
    K=32 program, two CTAs per token."""
    assert cake._FP8_H64_M64_BELOW_K64_KV4 == {
        "sm_107a": {
            (3, 2): ((257, 352, 24), (353, 383, 44)),
            (5, 5): ((640, 640, 32),),
        },
    }
    assert cake._K64_SIBLING_MIN_TOKENS == {"sm_107a": 48}
    # The first M64 token count of the early edge with and without them.
    kv4_edge = {208: 52, 212: 53, 216: 54}
    base_edge = {208: 54, 212: 55, 216: 56}

    def m64_body(width, tokens, sm_count, arch, kv4):
        return cake._fp8_h64_uses_m64_body(
            tokens, sm_count, sparse_topk=width, arch=arch, kv4=kv4
        )

    for sm_count in (208, 212, 216):
        for width in range(260, 384, 4):
            first = 24 if width <= 352 else 44
            for tokens in range(13, 61):
                key = (sm_count, width, tokens)
                band = first <= tokens <= 47
                with_kv4 = band or tokens >= kv4_edge[sm_count]
                assert m64_body(width, tokens, sm_count, "sm_107a", True) == with_kv4
                today = tokens >= base_edge[sm_count]
                assert m64_body(width, tokens, sm_count, "sm_107a", False) == today
                for arch in ("sm_100a", "sm_103a", None):
                    assert not m64_body(width, tokens, sm_count, arch, False), key
                    assert not m64_body(width, tokens, sm_count, arch, True), key
    # Width 640 (five complete tiles) from 32 tokens up to the floor, then the
    # persistent body until its early edge.
    for sm_count, edge in ((208, 56), (212, 57), (216, 58)):
        for tokens in range(13, 61):
            with_kv4 = 32 <= tokens <= 47 or tokens >= edge
            assert m64_body(640, tokens, sm_count, "sm_107a", True) == with_kv4
            assert not m64_body(640, tokens, sm_count, "sm_107a", False)
    sibling = f"{_M64_MULTI_TILE_ROUTE}_kv4"
    for width, tokens, program, grid in (
        (296, 30, sibling, 30),
        (296, 48, f"{_UNIFORM_ROUTE}_k64", 96),
        (296, 50, f"{_UNIFORM_ROUTE}_k64", 100),
        (296, 52, sibling, 52),
        (380, 30, _UNIFORM_ROUTE, 60),
        (380, 45, sibling, 45),
    ):
        programs, calls, _ = _run_fake_h64_decode_fp8(
            monkeypatch, arch="sm_107a", width=width, tokens=tokens
        )
        (bound,) = calls
        assert programs == [program], (width, tokens)
        assert (bound["grid_x"], bound["grid_y"], bound["grid_z"]) == (grid, 1, 1)
        assert bound["total_work_items"] == tokens
    # The windows end at the floor the call reads: 40 rows end them at 39.
    monkeypatch.setattr(cake, "_K64_SIBLING_MIN_TOKENS", {"sm_107a": 40})
    for tokens in range(13, 61):
        lower = m64_body(296, tokens, 208, "sm_107a", True)
        upper = m64_body(380, tokens, 208, "sm_107a", True)
        assert lower == (24 <= tokens <= 39 or tokens >= 52), tokens
        assert upper == (tokens >= 52), tokens
        wide = m64_body(640, tokens, 208, "sm_107a", True)
        assert wide == (32 <= tokens <= 39 or tokens >= 56), tokens


@pytest.mark.cpu_only
def test_kv4_edges_follow_each_devices_launch_check(monkeypatch, kv4_gate):
    """In one process, devices whose oversized ceilings differ route the same
    FP8/H64 decode row apart (width 384, 4 tokens, 208 SMs): the device that
    can launch the four-stage sibling takes the M64 route and runs the
    sibling, the other keeps the uniform route.  The launch check probes each
    device once."""
    monkeypatch.setattr(cake, "_kv4_sibling_launchable", kv4_gate)
    asked = _stub_gate_probes(
        monkeypatch, toolkit="13.4", runtime=13040, ceiling={0: 334848, 1: 300000}
    )
    first, second = torch.device("cuda", 0), torch.device("cuda", 1)
    for _ in range(2):
        for device, route in ((first, _M64_MULTI_TILE_ROUTE), (second, _UNIFORM_ROUTE)):
            kv4 = cake._fp8_h64_kv4_edges(
                dtype=torch.float8_e4m3fn,
                num_heads=64,
                max_q_len=1,
                sparse_topk=384,
                arch="sm_107a",
                device=device,
            )
            assert kv4 == (device == first)
            assert _h64_decode_route(384, 4, 208, "sm_107a", kv4=kv4) == route
    program = cake._kv4_variant(_M64_MULTI_TILE_ROUTE, arch="sm_107a", device=first)
    assert program == f"{_M64_MULTI_TILE_ROUTE}_kv4"
    assert asked == {"runtime": [13, 13], "ceiling": [first, second]}


@pytest.mark.cpu_only
def test_workspace_requirement_takes_the_kv4_edges_from_the_device(monkeypatch):
    """The sizing API takes the four-stage edges from the current device's
    launch check when it reads the architecture from the device too, so its
    route is the one run_cake_dsv4 launches there.  With an explicit ``arch``
    it takes them from ``kv4`` alone (default: without), whether or not CUDA
    is available.  The workspace numbers do not depend on them."""
    shape = dict(
        dtype=torch.float8_e4m3fn,
        num_heads=64,
        num_query_tokens=4,
        sm_count=208,
        sparse_topk=384,
        compressed_page_size=64,
        max_q_len=1,
        batch_size=4,
        ragged=False,
    )
    asked = []

    def gate(arch, device):
        asked.append((arch, device))
        return True

    monkeypatch.setattr(cake, "_kv4_sibling_launchable", gate)
    requirement = cake.cake_dsv4_workspace_requirement
    with_kv4 = requirement(**shape, arch="sm_107a", kv4=True)
    without = requirement(**shape, arch="sm_107a", kv4=False)
    assert (with_kv4.route, without.route) == (_M64_MULTI_TILE_ROUTE, _UNIFORM_ROUTE)
    for name in ("single_launch_bytes", "minimum_bytes", "num_splits"):
        assert getattr(with_kv4, name) == getattr(without, name), name
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    for available in (False, True):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
        assert requirement(**shape, arch="sm_107a").route == _UNIFORM_ROUTE
    assert asked == []
    # arch=None: the current CUDA device's architecture and launch check.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cake, "_target_arch", lambda device: "sm_107a")
    current = requirement(**shape)
    assert current.route == _M64_MULTI_TILE_ROUTE
    assert asked == [("sm_107a", torch.device("cuda", 0))]
    programs, _, _ = _run_fake_h64_decode_fp8(
        monkeypatch, arch="sm_107a", width=384, tokens=4
    )
    assert programs == [f"{current.route}_kv4"]
    # Without CUDA: no launch check, the architecture-independent route.
    asked.clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert requirement(**shape).route == _UNIFORM_ROUTE
    assert asked == []


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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_workspace_requirement_is_exact_per_route():
    mib128 = 128 * 1024 * 1024
    # Routes without partial buffers need no workspace bytes.
    swa = cake.cake_dsv4_workspace_requirement(
        dtype=torch.bfloat16,
        num_heads=64,
        num_query_tokens=12,
        sm_count=_SM_COUNT,
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
        sm_count=_SM_COUNT,
        sparse_topk=128,
        compressed_page_size=256,
        max_q_len=1,
        batch_size=1024,
        ragged=False,
        arch="sm_103a",
    )
    assert fp8.route == "fp8_h128_prefill_source_persistent_uniform"
    assert fp8.num_splits == 1
    assert fp8.single_launch_bytes == cake._PARTIAL_OFFSET + 1024 * 128 * 4
    assert fp8.rows_per_launch(mib128) == 1024
    # One row per launch: 128 heads x 4 B of LSE plus the 1025 shifted offsets.
    assert fp8.minimum_bytes == cake._PARTIAL_OFFSET + 512 + -(-(1025 * 4) // 128) * 128
    # The route-agnostic single-launch bound stays conservative (five splits).
    assert get_cake_dsv4_workspace_bytes(1024, 128, 128, torch.float8_e4m3fn) > mib128
    # The uniform route at a split-KV shape reports the split carve (40
    # pieces of 128 x 512 BF16 and 128 FP32 each at 8 rows of five pieces)
    # on every architecture; the one-row minimum stays the route's.
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        split = cake.cake_dsv4_workspace_requirement(
            dtype=torch.float8_e4m3fn,
            num_heads=128,
            num_query_tokens=8,
            sm_count=_SM_COUNT,
            sparse_topk=1152,
            compressed_page_size=64,
            max_q_len=1,
            batch_size=8,
            ragged=False,
            arch=arch,
        )
        assert split.route == "fp8_h128_prefill_source_persistent_uniform"
        assert split.num_splits == 1 and split.uses_workspace
        assert split.single_launch_bytes == 5_526_528 == 263168 + 40 * (131072 + 512)
        # One row per launch: 128 heads x 4 B of LSE plus the 9 shifted offsets.
        assert split.minimum_bytes == cake._PARTIAL_OFFSET + 512 + 128
        assert split.rows_per_launch(split.minimum_bytes) == 1
        # The route's own launch over all rows fits its LSE region.
        assert split.rows_per_launch(cake._PARTIAL_OFFSET + 8 * 128 * 4) == 8
    # The split route without a token bound tiles inside 128 MiB; the counter
    # region caps one launch at 16384 rows of four counters.
    h32 = cake.cake_dsv4_workspace_requirement(
        dtype=torch.bfloat16,
        num_heads=32,
        num_query_tokens=4096,
        sm_count=_SM_COUNT,
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
        sm_count=_SM_COUNT,
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
            sm_count=_SM_COUNT,
            sparse_topk=260,
            compressed_page_size=2,
            max_q_len=1,
            batch_size=4,
            ragged=False,
        )


@pytest.mark.cpu_only
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
                            sm_count=_SM_COUNT,
                            arch=arch,
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


# --------------------------------------------------------------------------- #
# SM107 split-KV launches of the FP8/H128 uniform-gather route                #
# --------------------------------------------------------------------------- #

# Pieces per row of the split launches SM107 runs with PDL (128 heads, items
# 1025-1152 wide): the first of 9, 5 and 3 that keeps rows x pieces <= 40
# (<= 60 for three).
_SPLIT_PIECES = {
    **dict.fromkeys(range(1, 5), 9),
    **dict.fromkeys(range(5, 9), 5),
    **dict.fromkeys(range(9, 21), 3),
}
# The merges built for one piece count: the merge for nine and five pieces
# alone (``_c{s}``, one CTA per row and head) and the three-piece merge
# (``_s3``, one warp per row and head, two heads per CTA).
_SPLIT_MERGE_C9 = f"{_UNIFORM_ROUTE}_k64_split_reduce_c9"
_SPLIT_MERGE_C5 = f"{_UNIFORM_ROUTE}_k64_split_reduce_c5"
_SPLIT_MERGE_S3 = f"{_UNIFORM_ROUTE}_k64_split_reduce_s3"
_SPLIT_COUNT_MERGES = (_SPLIT_MERGE_C9, _SPLIT_MERGE_C5, _SPLIT_MERGE_S3)
# The merge and its heads per CTA of each row count's 128-head split launch
# where the registry routes the per-count merges to programs of their own:
# the merge for nine pieces at 1-4 rows and for five at 5-8, the merge of any
# count at 9-15 and the three-piece merge at 16-20.
_SPLIT_MERGES = {
    **dict.fromkeys(range(1, 5), (_SPLIT_MERGE_C9, 1)),
    **dict.fromkeys(range(5, 9), (_SPLIT_MERGE_C5, 1)),
    **dict.fromkeys(range(9, 16), (f"{_UNIFORM_ROUTE}_k64_split_reduce", 1)),
    **dict.fromkeys(range(16, 21), (_SPLIT_MERGE_S3, 2)),
}


def _split_plan(**shape):
    """``cake._k64_split_plan`` of an SM107 uniform launch of 8 PDL rows over
    1152-wide items with 128 heads, with ``shape`` changed."""
    args = dict(
        route=_UNIFORM_ROUTE,
        arch="sm_107a",
        num_heads=128,
        num_query_tokens=8,
        sparse_topk=1152,
        enable_pdl=True,
    )
    args.update(shape)
    route = args.pop("route")
    return cake._k64_split_plan(route, **args)


@pytest.mark.cpu_only
def test_k64_split_rule_is_sm107a_only():
    """Only SM107 splits, and only with PDL, the uniform route's 128-head
    launches of at most 20 rows over nine-tile items, at most 40 pieces (60
    for three per row); the launches of nine and five pieces merge with the
    merge for their count alone, those of three at 16-20 rows with the
    three-piece merge."""
    assert _UNIFORM_ROUTE == cake._K64_SPLIT_ROUTE
    assert f"{_UNIFORM_ROUTE}_k64_split" == cake._K64_SPLIT_VARIANT
    assert f"{_UNIFORM_ROUTE}_k64_split_reduce" == cake._K64_SPLIT_MERGE
    assert {
        _SPLIT_MERGE_C9: 1,
        _SPLIT_MERGE_C5: 1,
        _SPLIT_MERGE_S3: 2,
    } == cake._K64_SPLIT_COUNT_MERGES
    rule = cake._K64SplitRule(
        num_tiles=9,
        max_tokens_pdl=20,
        max_tokens_no_pdl=0,
        splits=((9, 40), (5, 40), (3, 60)),
        merge_by_splits=(
            (9, 1, _SPLIT_MERGE_C9),
            (5, 5, _SPLIT_MERGE_C5),
            (3, 16, _SPLIT_MERGE_S3),
        ),
    )
    assert {"sm_107a": rule} == cake._K64_SPLIT_RULES
    # The split keys are not K=64 siblings: the K=64 floor never applies to them.
    assert cake._K64_SPLIT_VARIANT not in cake._K64_SIBLING_ROUTES
    assert cake._K64_SPLIT_MERGE not in cake._K64_SIBLING_ROUTES
    for key in _SPLIT_COUNT_MERGES:
        assert key not in cake._K64_SIBLING_ROUTES, key


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "tokens,suffix",
    [
        (1, "_c9"),
        (4, "_c9"),
        (5, "_c5"),
        (8, "_c5"),
        (9, ""),
        (14, ""),
        (15, ""),
        (16, "_s3"),
        (17, "_s3"),
        (18, "_s3"),
        (20, "_s3"),
    ],
)
def test_k64_split_merge_at_the_rule_boundaries(monkeypatch, tokens, suffix):
    """At both ends of every merge band of the SM107 128-head rule, with the
    registry routing every per-count merge to a program of its own, a split
    launch merges with the merge for nine pieces at 1 and 4 rows and for five
    at 5 and 8 (one head per CTA), with the merge of any count at 9, 14 and
    15 and with the three-piece merge at 16, 17, 18 and 20 (two heads per
    CTA); the same launch with 64 heads, or on SM100 / SM103, merges with the
    merge of any count."""
    import flashinfer.jit.cake_dsv4 as jit

    monkeypatch.setattr(jit, "VARIANTS", _fake_split_variants(routed=True))
    splits = _SPLIT_PIECES[tokens]
    expected = (f"{cake._K64_SPLIT_MERGE}{suffix}", 2 if suffix == "_s3" else 1)
    assert _SPLIT_MERGES[tokens] == expected

    def merge(arch="sm_107a", num_heads=128):
        return cake._k64_split_merge(
            arch=arch, num_heads=num_heads, rows=tokens, num_splits=splits
        )

    assert merge() == expected
    assert merge(num_heads=64) == (cake._K64_SPLIT_MERGE, 1)
    for arch in ("sm_100a", "sm_103a"):
        assert merge(arch=arch) == (cake._K64_SPLIT_MERGE, 1), arch


@pytest.mark.cpu_only
@pytest.mark.parametrize("tokens", [*range(1, 22), 24, 48, 127])
@pytest.mark.parametrize("sparse_topk", [1028, 1088, 1152])
def test_k64_split_plan_pins(tokens, sparse_topk):
    """Rows 1-4 split into nine pieces, 5-8 into five and 9-20 into three; no
    whole rows; 21 rows and more keep the route."""
    expected = (0, tokens, _SPLIT_PIECES[tokens]) if tokens in _SPLIT_PIECES else None
    assert _split_plan(num_query_tokens=tokens, sparse_topk=sparse_topk) == expected
    if expected is not None:
        assert tokens * expected[2] <= (60 if expected[2] == 3 else 40)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "change",
    [
        {"arch": "sm_100a"},
        {"arch": "sm_103a"},
        {"arch": "sm_120a"},
        {"route": "fp8_h128_prefill_source_persistent"},
        {"route": "fp8_h64_prefill_source_persistent_m64_multi_tile"},
        {"route": "bf16_h128_topk4x_v52"},
        {"num_heads": 32},
        {"num_heads": 16},
        {"num_heads": 8},
        {"sparse_topk": 1024},
        {"sparse_topk": 896},
        {"sparse_topk": 640},
        {"sparse_topk": 128},
        {"sparse_topk": 1156},
        {"enable_pdl": False},
        {"num_query_tokens": 0},
        {"num_query_tokens": 21},
    ],
    ids=lambda change: ",".join(f"{k}={v}" for k, v in change.items()),
)
def test_k64_split_plan_keeps_every_other_launch(change):
    assert _split_plan() == (0, 8, 5)
    assert _split_plan(**change) is None


@pytest.mark.cpu_only
def test_k64_split_plan_without_pdl_follows_its_own_ceiling(monkeypatch):
    """Launches without PDL split up to the rule's PDL-off row ceiling (0 on
    main: none) with the same pieces as with PDL."""
    import dataclasses

    rule = cake._K64_SPLIT_RULES["sm_107a"]
    monkeypatch.setattr(
        cake,
        "_K64_SPLIT_RULES",
        {"sm_107a": dataclasses.replace(rule, max_tokens_no_pdl=13)},
    )
    for tokens in range(1, 22):
        split = (0, tokens, _SPLIT_PIECES[tokens]) if tokens in _SPLIT_PIECES else None
        assert _split_plan(num_query_tokens=tokens, enable_pdl=True) == split
        assert _split_plan(num_query_tokens=tokens, enable_pdl=False) == (
            split if tokens <= 13 else None
        )


@pytest.mark.cpu_only
def test_k64_split_checks_the_workspace_registry_and_gate(monkeypatch):
    """A planned split runs only when the workspace holds its carve, the
    architecture routes both split keys to programs of their own and the K=64
    gate passes; the gate is asked last.  The per-count merges are no
    condition: an 8-row launch splits into five pieces and a 16-row launch
    into three without them, and both merge with the merge of any count, one
    CTA per (row, head)."""
    import flashinfer.jit.cake_dsv4 as jit

    carve = cake_dsv4_workspace_layout(8, 128, 5).total_bytes
    asked = []
    launchable = [True]

    def gate(arch, device):
        asked.append((arch, device))
        return launchable[0]

    monkeypatch.setattr(cake, "_k64_sibling_launchable", gate)

    def split(**changes):
        args = dict(
            arch="sm_107a",
            num_heads=128,
            num_query_tokens=8,
            sparse_topk=1152,
            enable_pdl=True,
            workspace_bytes=carve,
            device=_DEVICE,
        )
        args.update(changes)
        return cake._k64_split(_UNIFORM_ROUTE, **args)

    keys = (cake._K64_SPLIT_VARIANT, cake._K64_SPLIT_MERGE, *_SPLIT_COUNT_MERGES)
    table = {
        arch: {key: program for key, program in variants.items() if key not in keys}
        for arch, variants in jit.VARIANTS.items()
    }
    monkeypatch.setattr(jit, "VARIANTS", table)
    assert split() is None
    sm107 = table["sm_107a"]
    sm107[cake._K64_SPLIT_VARIANT] = "fake_split_producer"
    assert split() is None
    sm107[cake._K64_SPLIT_MERGE] = sm107[_UNIFORM_ROUTE]
    assert split() is None
    sm107[cake._K64_SPLIT_MERGE] = "fake_split_merge"
    assert asked == []
    assert split() == (0, 8, 5)
    assert asked == [("sm_107a", _DEVICE)]
    assert split(workspace_bytes=carve - 1) is None
    assert split(enable_pdl=False) is None
    assert split(num_query_tokens=21) is None
    assert len(asked) == 1
    assert split(workspace_bytes=carve + 4096) == (0, 8, 5)
    assert _SPLIT_MERGE_C5 not in sm107
    assert cake._k64_split_merge(
        arch="sm_107a", num_heads=128, rows=8, num_splits=5
    ) == (cake._K64_SPLIT_MERGE, 1)
    # Sixteen rows, three pieces each, with the three-piece merge unrouted.
    carve16 = cake_dsv4_workspace_layout(16, 128, 3).total_bytes
    assert _SPLIT_MERGE_S3 not in sm107
    assert split(num_query_tokens=16, workspace_bytes=carve16 - 1) is None
    assert len(asked) == 2
    assert split(num_query_tokens=16, workspace_bytes=carve16) == (0, 16, 3)
    assert len(asked) == 3
    assert cake._k64_split_merge(
        arch="sm_107a", num_heads=128, rows=16, num_splits=3
    ) == (cake._K64_SPLIT_MERGE, 1)
    launchable[0] = False
    assert split() is None
    assert split(num_query_tokens=16, workspace_bytes=carve16) is None
    assert len(asked) == 5


def _split_merge(tokens, num_heads=128):
    """The merge key and grid of an SM107 split launch of ``tokens`` rows of
    ``num_heads`` heads where the registry routes the per-count merges to
    programs of their own: for 128 heads the merge of ``_SPLIT_MERGES`` over
    (rows, heads / its heads per CTA), else the merge of any count over
    (rows, heads)."""
    if num_heads == 128:
        key, heads_per_cta = _SPLIT_MERGES[tokens]
        return key, (tokens, num_heads // heads_per_cta, 1)
    return cake._K64_SPLIT_MERGE, (tokens, num_heads, 1)


@pytest.mark.cpu_only
@pytest.mark.parametrize("tokens", [1, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20])
def test_split_launch_runs_the_producer_then_the_merge(monkeypatch, tokens):
    """A split launch issues the split producer over 2 CTAs per piece and the
    merge, in one run, both on the piece views: the three-piece merge over one
    CTA per (row, two heads) at three pieces per row and 16-20 rows, the
    merge for its count alone (nine and five pieces) and the merge of any
    count (three pieces at 9-15 rows) over one CTA per (row, head)."""
    import flashinfer.jit.cake_dsv4 as jit

    monkeypatch.setattr(jit, "VARIANTS", _fake_split_variants(routed=True))
    splits = _SPLIT_PIECES[tokens]
    out = torch.empty((tokens, 128, 512), dtype=torch.bfloat16)
    L = _RecordingLauncher(
        "sm_107a",
        split=(0, tokens, splits),
        Q=torch.empty(0),
        O=out,
        num_query_tokens=tokens,
        num_heads=128,
        sparse_topk=1152,
        max_q_len=1,
        enable_pdl=True,
    )
    cake._dispatch_route(_UNIFORM_ROUTE, L)
    pieces = L.pieces(tokens, splits)
    (producer, producer_args), (merge, merge_args) = L.calls
    assert producer == cake._K64_SPLIT_VARIANT
    assert producer_args == {
        "grid": (2 * tokens * splits, 1, 1),
        "total_work_items": tokens * splits,
        "num_whole_items": 0,
        **pieces,
    }
    key, grid = _split_merge(tokens)
    assert merge == key
    merge_out = merge_args.pop("O")
    assert merge_args == {"grid": grid, **pieces}
    assert merge_out.data_ptr() == out.data_ptr() and merge_out.shape == out.shape


def _uniform_plan():
    import flashinfer.jit.cake_dsv4 as jit

    return [tuple(pair) for pair in jit.ARG_PLANS[_UNIFORM_ROUTE]]


def _split_producer_plan():
    """The uniform plan with the split producer's additions, in the order of
    its kernel signature: ``tmap_partial_o`` after ``tmap_o``, then
    ``num_whole_items`` and ``num_splits`` after ``batch_size``."""
    plan = _uniform_plan()
    at = plan.index(("tma_buffer", "tmap_o")) + 1
    plan[at:at] = [("tma_buffer", "tmap_partial_o")]
    at = plan.index(("parameter", "batch_size")) + 1
    plan[at:at] = [("parameter", "num_whole_items"), ("parameter", "num_splits")]
    return plan


# The merge's argument plan (its kernel signature: the pieces, O, H and s;
# PDL is a launch attribute of each call).
_SPLIT_MERGE_PLAN = _plan(
    ("buffer", "partial_O"),
    ("buffer", "partial_lse"),
    ("buffer", "O"),
    ("parameter", "num_heads"),
    ("parameter", "num_splits"),
    ("launch", "enable_pdl"),
)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_split_programs_take_the_plans_the_host_binds(arch):
    """SM107 routes both split keys and the three per-count merges to
    programs of their own, no two the same: the producer's plan is the
    uniform plan with the split additions and every merge's plan is the merge
    kernel signature.  SM100 / SM103 route the five keys to the uniform
    program.  On SM107 the 128-head launch of every row count merges with
    the merge of ``_SPLIT_MERGES``."""
    import flashinfer.jit.cake_dsv4 as jit

    variants = jit.VARIANTS[arch]
    uniform = variants[_UNIFORM_ROUTE]
    keys = (cake._K64_SPLIT_VARIANT, cake._K64_SPLIT_MERGE, *_SPLIT_COUNT_MERGES)
    own = [variants.get(key, uniform) != uniform for key in keys]
    assert own == [arch == "sm_107a"] * len(keys)
    if arch == "sm_107a":
        assert len({variants[key] for key in keys}) == len(keys)
        merge_plans = [_SPLIT_MERGE_PLAN] * (len(keys) - 1)
        plans = (_split_producer_plan(), *merge_plans)
        for key, plan in zip(keys, plans, strict=True):
            assert get_cake_dsv4_spec(key, arch=arch)["arg_plan"] == tuple(plan), key
        merges = {
            rows: cake._k64_split_merge(
                arch=arch, num_heads=128, rows=rows, num_splits=_SPLIT_PIECES[rows]
            )
            for rows in _SPLIT_MERGES
        }
        assert merges == _SPLIT_MERGES
    else:
        for key in _SPLIT_COUNT_MERGES:
            assert variants[key] == uniform, key


class _SplitRun:
    """The launches of one fake run: ``calls`` = (variant, bound arguments by
    name) in launch order; ``sequences`` = the run_sequence calls (launches
    per call)."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.sequences: list[int] = []


def _fake_split_variants(*, routed: bool, count_merge: str = "own"):
    """The registry without the split keys; with ``routed``, SM107 routes the
    two split keys to programs of their own and SM100 / SM103 to the uniform
    program, and ``count_merge`` places the per-count merges' keys on SM107:
    ``own`` each a program of its own (folded into the uniform program on
    SM100 / SM103), ``unrouted`` nowhere, ``route`` / ``merge`` the uniform
    route's / the merge's program (retired keys)."""
    import flashinfer.jit.cake_dsv4 as jit

    keys = (cake._K64_SPLIT_VARIANT, cake._K64_SPLIT_MERGE, *_SPLIT_COUNT_MERGES)
    table = {
        arch: {key: program for key, program in variants.items() if key not in keys}
        for arch, variants in jit.VARIANTS.items()
    }
    if routed:
        for arch, variants in table.items():
            own = arch == "sm_107a"
            uniform = variants[_UNIFORM_ROUTE]
            variants[cake._K64_SPLIT_VARIANT] = (
                "fake_split_producer" if own else uniform
            )
            variants[cake._K64_SPLIT_MERGE] = "fake_split_merge" if own else uniform
            for key in _SPLIT_COUNT_MERGES:
                if not own:
                    if count_merge != "unrouted":
                        variants[key] = uniform
                elif count_merge == "own":
                    variants[key] = f"fake_split_merge_{key[-2:]}"
                elif count_merge == "route":
                    variants[key] = uniform
                elif count_merge == "merge":
                    variants[key] = "fake_split_merge"
                else:
                    assert count_merge == "unrouted", count_merge
    return table


def _install_fake_split_registry(
    monkeypatch, *, routed: bool, count_merge: str = "own"
) -> _SplitRun:
    """Fake modules for the uniform route, its K=64 sibling, the two split
    programs and the per-count merges; with ``routed``, a registry in which
    SM107 routes both split keys to programs of their own and SM100 / SM103
    to the uniform program, with the per-count merges' keys placed by
    ``count_merge`` (:func:`_fake_split_variants`)."""
    import flashinfer.jit.cake_dsv4 as jit

    plans = {
        _UNIFORM_ROUTE: _uniform_plan(),
        f"{_UNIFORM_ROUTE}_k64": _uniform_plan(),
        cake._K64_SPLIT_VARIANT: _split_producer_plan(),
        cake._K64_SPLIT_MERGE: _SPLIT_MERGE_PLAN,
        **dict.fromkeys(_SPLIT_COUNT_MERGES, _SPLIT_MERGE_PLAN),
    }
    _install_fake_variants(monkeypatch, plans)
    record = _SplitRun()

    class _Module:
        def __init__(self, variant):
            self.variant = variant

        def run(self, *args):
            names = (name for _, name in plans[self.variant])
            record.calls.append((self.variant, dict(zip(names, args, strict=True))))

    class _Sequence:
        @staticmethod
        def run_sequence(*flat):
            launches = 0
            i = 0
            while i < len(flat):
                fn, count = flat[i], flat[i + 1]
                fn(*flat[i + 2 : i + 2 + count])
                i += 2 + count
                launches += 1
            record.sequences.append(launches)

    monkeypatch.setattr(
        cake, "_variant_module", lambda variant, *, arch: _Module(variant)
    )
    monkeypatch.setattr(cake, "_sequence_module", lambda: _Sequence)
    monkeypatch.setattr(
        jit, "VARIANTS", _fake_split_variants(routed=routed, count_merge=count_merge)
    )
    return record


def _split_inputs(tokens, *, workspace_bytes=None, num_heads=128, sparse_topk=1152):
    """run_cake_dsv4 keyword arguments of a dense one-row-per-request FP8
    decode call over ``sparse_topk``-wide items (page size 64)."""
    table = torch.arange(tokens * sparse_topk, dtype=torch.int32).reshape(
        tokens, sparse_topk
    )
    if workspace_bytes is None:
        workspace_bytes = get_cake_dsv4_workspace_bytes(
            tokens, num_heads, sparse_topk, torch.float8_e4m3fn
        )
    return dict(
        query=torch.zeros((tokens, num_heads, 512), dtype=torch.float8_e4m3fn),
        swa_kv_cache=torch.zeros((4, 1, 256, 512), dtype=torch.float8_e4m3fn),
        compressed_kv_cache=torch.zeros((8, 1, 64, 512), dtype=torch.float8_e4m3fn),
        workspace_buffer=_aligned_u8(workspace_bytes),
        out=torch.zeros((tokens, num_heads, 512), dtype=torch.bfloat16),
        bmm1_scale=0.5,
        bmm2_scale=1.0,
        sinks=None,
        max_q_len=1,
        cum_seq_lens_q=None,
        seq_lens=torch.full((tokens,), 4096, dtype=torch.int32),
        backend="cake",
        sparse_indices=table,
        sparse_topk_lens=torch.full((tokens,), sparse_topk, dtype=torch.int32),
    )


def _run_split(
    monkeypatch,
    inputs,
    *,
    arch="sm_107a",
    enable_pdl=True,
    routed=True,
    count_merge="own",
):
    record = _install_fake_split_registry(
        monkeypatch, routed=routed, count_merge=count_merge
    )
    monkeypatch.setattr(cake, "_target_arch", lambda device: arch)
    assert cake.run_cake_dsv4(**inputs, enable_pdl=enable_pdl) is inputs["out"]
    return record


def _launch_record(calls):
    """``calls`` with every tensor as (address, shape, strides, dtype)."""

    def key(value):
        if isinstance(value, torch.Tensor):
            return (value.data_ptr(), tuple(value.shape), value.stride(), value.dtype)
        return value

    return [
        (variant, {name: key(value) for name, value in args.items()})
        for variant, args in calls
    ]


@pytest.mark.cpu_only
@pytest.mark.parametrize("tokens", [1, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20])
def test_run_cake_dsv4_split_launch_binds_the_piece_views(monkeypatch, tokens):
    """The split launch carves cake_dsv4_workspace_layout(T, H, s): the
    producer stores its pieces through the [T * s, H, 512] partial-O view at
    byte 263168 and the LSE right after it, final O of single-piece rows
    through O; the merge (the three-piece merge over (T, H / 2) at three
    pieces per row and 16-20 rows, else the merge for its count alone, nine
    and five pieces, or the merge of any count, three pieces at 9-15 rows,
    over (T, H)) reads the same views and writes O; both bind the call's PDL
    and run in one run_sequence call."""
    splits = _SPLIT_PIECES[tokens]
    pieces = tokens * splits
    inputs = _split_inputs(tokens)
    record = _run_split(monkeypatch, inputs)
    assert record.sequences == [2]
    (producer, p), (merge, m) = record.calls
    key, grid = _split_merge(tokens)
    assert (producer, merge) == (cake._K64_SPLIT_VARIANT, key)
    base = inputs["workspace_buffer"].data_ptr()
    out = inputs["out"]
    layout = cake_dsv4_workspace_layout(tokens, 128, splits)
    assert layout.partial_o == (263168, pieces * 128 * 512 * 2)
    lse_offset = 263168 + pieces * 131072
    assert layout.partial_lse == (lse_offset, pieces * 128 * 4)
    # Producer.
    view = p["tmap_partial_o"]
    assert view.data_ptr() == base + 263168 and view.is_contiguous()
    assert view.shape == (pieces, 128, 512) and view.dtype == torch.bfloat16
    assert p["partial_lse"].data_ptr() == base + lse_offset
    assert p["partial_lse"].numel() == pieces * 128
    assert p["partial_lse"].dtype == torch.float32
    assert p["tmap_o"] is p["O"] and p["O"].data_ptr() == out.data_ptr()
    assert p["O"].shape == (tokens, 128, 512)
    assert (p["grid_x"], p["grid_y"], p["grid_z"]) == (2 * pieces, 1, 1)
    assert p["total_work_items"] == pieces and p["num_whole_items"] == 0
    assert p["num_splits"] == splits and p["num_query_tokens"] == tokens
    assert p["num_heads"] == 128 and p["sparse_topk"] == 1152
    assert p["enable_pdl"] == 1
    # Merge.
    assert m["partial_O"].data_ptr() == base + 263168
    assert m["partial_O"].numel() == pieces * 128 * 512
    assert m["partial_lse"].data_ptr() == base + lse_offset
    assert m["partial_lse"].numel() == pieces * 128
    assert m["O"].data_ptr() == out.data_ptr() and m["O"].shape == (tokens, 128, 512)
    assert m["num_splits"] == splits and m["num_heads"] == 128
    assert (m["grid_x"], m["grid_y"], m["grid_z"]) == grid
    assert m["enable_pdl"] == 1


@pytest.mark.cpu_only
@pytest.mark.parametrize("tokens", [1, 8, 17, 20])
@pytest.mark.parametrize(
    "fallback", ["workspace", "no_pdl", "sm_103a", "gate", "unrouted"]
)
def test_split_fallbacks_launch_the_route_as_main(monkeypatch, tokens, fallback):
    """A workspace one byte short of the split carve, a call without PDL, an
    architecture other than SM107, a process that cannot launch the K=64
    programs and a registry without the split programs each run the route's
    own single launch, argument for argument as without the split."""
    carve = cake_dsv4_workspace_layout(tokens, 128, _SPLIT_PIECES[tokens]).total_bytes
    inputs = _split_inputs(
        tokens, workspace_bytes=carve - 1 if fallback == "workspace" else carve
    )
    arch = "sm_103a" if fallback == "sm_103a" else "sm_107a"
    enable_pdl = fallback != "no_pdl"
    asked = []
    if fallback == "gate":
        monkeypatch.setattr(
            cake,
            "_k64_sibling_launchable",
            lambda gate_arch, device: asked.append((gate_arch, device)) or False,
        )
    got = _run_split(
        monkeypatch,
        inputs,
        arch=arch,
        enable_pdl=enable_pdl,
        routed=fallback != "unrouted",
    )
    main = _run_split(
        monkeypatch, inputs, arch=arch, enable_pdl=enable_pdl, routed=False
    )
    assert _launch_record(got.calls) == _launch_record(main.calls)
    assert got.sequences == []
    ((variant, args),) = got.calls
    assert variant == _UNIFORM_ROUTE
    assert (args["grid_x"], args["grid_y"], args["grid_z"]) == (2 * tokens, 1, 1)
    assert args["total_work_items"] == tokens
    base = inputs["workspace_buffer"].data_ptr()
    assert args["partial_lse"].data_ptr() == base + 263168
    assert args["partial_lse"].numel() == tokens * 128
    assert args["O"].data_ptr() == inputs["out"].data_ptr()
    assert args["enable_pdl"] == int(enable_pdl)
    assert asked == ([("sm_107a", torch.device("cpu"))] if fallback == "gate" else [])


_GRID = ("grid_x", "grid_y", "grid_z")


@pytest.mark.cpu_only
@pytest.mark.parametrize("tokens", [1, 4, 5, 8, 9, 14, 15, 16, 17, 18, 20])
@pytest.mark.parametrize("count_merge", ["unrouted", "route", "merge"])
def test_split_without_the_count_merge_merges_with_the_merge_of_any_count(
    monkeypatch, tokens, count_merge
):
    """A registry in which the per-count merges have no program of their own
    (their keys absent, or retired onto the uniform route's or onto the
    merge's program) still splits a launch and merges it with the merge of
    any count over (T, 128): the same producer launch and the same merge
    arguments as with the per-count merges routed, under which a launch of
    16-20 rows differs only in its merge key and its grid (T, 64), one of
    1-8 rows only in its merge key and one of 9-15 rows not at all."""
    inputs = _split_inputs(tokens)
    got = _run_split(monkeypatch, inputs, count_merge=count_merge)
    own = _run_split(monkeypatch, inputs)
    assert got.sequences == own.sequences == [2]
    (producer, p), (merge, m) = _launch_record(got.calls)
    (own_producer, own_p), (own_merge, own_m) = _launch_record(own.calls)
    assert (producer, p) == (own_producer, own_p)
    assert producer == cake._K64_SPLIT_VARIANT
    key, grid = _split_merge(tokens)
    assert (merge, own_merge) == (cake._K64_SPLIT_MERGE, key)
    assert tuple(m[name] for name in _GRID) == (tokens, 128, 1)
    assert tuple(own_m[name] for name in _GRID) == grid
    rest = {name: value for name, value in m.items() if name not in _GRID}
    assert rest == {name: value for name, value in own_m.items() if name not in _GRID}


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "tokens,num_heads,sparse_topk",
    [
        (21, 128, 1152),
        (24, 128, 1152),
        (8, 128, 1024),
        (8, 128, 640),
        (21, 64, 1152),
        (10, 64, 640),
        (5, 64, 512),
    ],
)
def test_split_keeps_the_route_outside_its_rule(
    monkeypatch, tokens, num_heads, sparse_topk
):
    """Uniform-route launches outside the rules (128 heads: more than 20 rows
    or items of fewer than nine tiles; 64 heads: rows past those of the rule
    of their item width) keep the route's single launch on SM107 with the
    split programs registered."""
    inputs = _split_inputs(tokens, num_heads=num_heads, sparse_topk=sparse_topk)
    got = _run_split(monkeypatch, inputs)
    main = _run_split(monkeypatch, inputs, routed=False)
    assert _launch_record(got.calls) == _launch_record(main.calls)
    ((variant, args),) = got.calls
    assert variant == _UNIFORM_ROUTE and args["grid_x"] == 2 * tokens


@pytest.mark.cpu_only
def test_row_tiled_launches_never_split(monkeypatch):
    """A workspace below the route's single launch tiles the rows; no chunk
    splits, although each chunk alone is a split shape."""
    tokens = 8
    plan = cake._route_plan(
        _UNIFORM_ROUTE, num_query_tokens=tokens, num_heads=128, sparse_topk=1152
    )
    workspace_bytes = cake._launch_workspace_bytes(plan, 4, 128, tokens + 1)
    assert (
        cake._rows_per_launch(
            plan,
            num_query_tokens=tokens,
            num_heads=128,
            batch_size=tokens,
            workspace_bytes=workspace_bytes,
        )
        == 4
    )
    splits = []
    dispatch = cake._dispatch_route

    def recording_dispatch(route, L):
        splits.append(L.split)
        dispatch(route, L)

    monkeypatch.setattr(cake, "_dispatch_route", recording_dispatch)
    record = _run_split(
        monkeypatch, _split_inputs(tokens, workspace_bytes=workspace_bytes)
    )
    assert splits == [None, None]
    assert record.sequences == []
    assert [variant for variant, _ in record.calls] == [_UNIFORM_ROUTE] * 2
    assert [args["num_query_tokens"] for _, args in record.calls] == [4, 4]
    assert all(args["grid_x"] == 8 for _, args in record.calls)
    assert _split_plan(num_query_tokens=4) == (0, 4, 9)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
@pytest.mark.parametrize("sm_count", [148, 216])
def test_workspace_requirement_covers_the_split_launch(arch, sm_count):
    """At a split shape the requirement reports the split carve on every
    architecture at these SM counts (the PDL setting is no input; below 69
    SMs some 64-head split shapes take an M64 route, which never splits);
    elsewhere and in the one-row minimum it reports the route's own numbers,
    and get_cake_dsv4_workspace_bytes bounds every split carve."""
    for tokens in (*range(1, 22), 24, 48):
        for sparse_topk in (1024, 1028, 1152):
            for num_heads in (64, 128):
                req = cake.cake_dsv4_workspace_requirement(
                    dtype=torch.float8_e4m3fn,
                    num_heads=num_heads,
                    num_query_tokens=tokens,
                    sm_count=sm_count,
                    sparse_topk=sparse_topk,
                    compressed_page_size=64,
                    max_q_len=1,
                    batch_size=tokens,
                    ragged=False,
                    arch=arch,
                )
                own = cake._launch_workspace_bytes(req.plan, tokens, num_heads, 0)
                pieces = _planned_pieces(num_heads, sparse_topk, tokens)
                if req.route == _UNIFORM_ROUTE and pieces is not None:
                    carve = cake_dsv4_workspace_layout(
                        tokens, num_heads, pieces
                    ).total_bytes
                    assert req.single_launch_bytes == carve > own
                    assert carve <= get_cake_dsv4_workspace_bytes(
                        tokens, num_heads, sparse_topk, torch.float8_e4m3fn
                    )
                else:
                    assert req.single_launch_bytes == own, (tokens, sparse_topk)
                assert req.num_splits == req.plan.num_splits
                assert req.minimum_bytes == min(
                    own,
                    cake._launch_workspace_bytes(req.plan, 1, num_heads, tokens + 1),
                )
    # The largest carve: 20 rows of three pieces (60 pieces).
    assert (
        max(
            cake._k64_split_workspace_bytes(
                _UNIFORM_ROUTE, num_heads=128, num_query_tokens=tokens, sparse_topk=1152
            )
            for tokens in range(1, 22)
        )
        == 8_158_208
        == 263168 + 60 * (131072 + 512)
    )


# --------------------------------------------------------------------------- #
# SM107 split-KV launches of the uniform route with 64 heads                  #
# --------------------------------------------------------------------------- #

# Pieces per row of the 64-head split launches SM107 runs with PDL, per item
# width in tiles (``_K64_SPLIT_RULES_H64``): rows -> pieces.
_H64_SPLIT_PIECES = {
    4: dict.fromkeys(range(1, 5), 4),
    5: dict.fromkeys(range(1, 10), 5),
    6: {**dict.fromkeys(range(1, 10), 6), **dict.fromkeys(range(10, 19), 3)},
    7: {**dict.fromkeys(range(1, 10), 7), **dict.fromkeys(range(10, 18), 4)},
    8: {
        **dict.fromkeys(range(1, 8), 8),
        **dict.fromkeys(range(8, 16), 4),
        **dict.fromkeys(range(16, 21), 3),
    },
    9: {
        **dict.fromkeys(range(1, 7), 9),
        **dict.fromkeys(range(7, 13), 5),
        **dict.fromkeys(range(13, 21), 3),
    },
}
# The narrowest and the widest item of every tile count with a 64-head rule.
_H64_SPLIT_WIDTHS = [
    width for tiles in _H64_SPLIT_PIECES for width in (128 * tiles - 124, 128 * tiles)
]


def _planned_pieces(num_heads, sparse_topk, tokens):
    """Pieces per row SM107 plans for a PDL launch of the uniform route of this
    shape (``None``: no split)."""
    tiles = -(-sparse_topk // 128)
    if num_heads == 128:
        return _SPLIT_PIECES.get(tokens) if tiles == 9 else None
    if num_heads == 64:
        return _H64_SPLIT_PIECES.get(tiles, {}).get(tokens)
    return None


@pytest.mark.cpu_only
def test_k64_split_h64_rules_are_sm107a_only():
    """Only SM107 splits 64-head launches, and only with PDL: one rule per item
    width of four to nine tiles, at most nine pieces per row (the split
    programs' limit) and never more pieces than tiles.  The 128-head rule
    stays a table of its own."""
    assert cake._K64_SPLIT_HEADS_H64 == 64 != cake._K64_SPLIT_HEADS
    assert set(cake._K64_SPLIT_RULES_H64) == {"sm_107a"}
    rules = cake._K64_SPLIT_RULES_H64["sm_107a"]
    assert [rule.num_tiles for rule in rules] == sorted(_H64_SPLIT_PIECES)
    for rule in rules:
        assert rule.max_tokens_no_pdl == 0
        assert rule.max_tokens_pdl == max(_H64_SPLIT_PIECES[rule.num_tiles])
        assert max(s for s, _ in rule.splits) <= min(rule.num_tiles, 9)


@pytest.mark.cpu_only
@pytest.mark.parametrize("sparse_topk", [*_H64_SPLIT_WIDTHS, 128, 256, 260, 384, 1156])
def test_k64_split_plan_h64_pins(sparse_topk):
    """64-head SM107 PDL launches split by the rule of their item width in
    tiles, with no whole rows; items of at most three tiles or of more than
    nine never split, no 64-head launch splits without PDL and no other
    architecture splits one."""
    for tokens in (*range(0, 22), 24, 48, 127):
        pieces = _planned_pieces(64, sparse_topk, tokens)
        expected = None if pieces is None else (0, tokens, pieces)
        shape = dict(num_heads=64, num_query_tokens=tokens, sparse_topk=sparse_topk)
        assert _split_plan(**shape) == expected, tokens
        assert _split_plan(**shape, enable_pdl=False) is None, tokens
        for arch in ("sm_100a", "sm_103a", "sm_120a"):
            for enable_pdl in (False, True):
                plan = _split_plan(**shape, arch=arch, enable_pdl=enable_pdl)
                assert plan is None, (arch, tokens)


def _h64_plans(enable_pdl=True):
    """SM107 plans of every 64-head width class at 0-21 rows."""
    return [
        _split_plan(
            num_heads=64,
            num_query_tokens=tokens,
            sparse_topk=width,
            enable_pdl=enable_pdl,
        )
        for width in _H64_SPLIT_WIDTHS
        for tokens in range(0, 22)
    ]


def _h128_plans(enable_pdl=True):
    """SM107 plans of nine-tile 128-head launches of 0-21 rows."""
    return [
        _split_plan(num_query_tokens=tokens, sparse_topk=width, enable_pdl=enable_pdl)
        for width in (1028, 1152)
        for tokens in range(0, 22)
    ]


@pytest.mark.cpu_only
def test_k64_split_rule_tables_are_independent(monkeypatch):
    """The 64-head plans read only ``_K64_SPLIT_RULES_H64`` and the 128-head
    plans only ``_K64_SPLIT_RULES``: changing or emptying one table leaves
    the other's plans, with and without PDL, as they are."""
    import dataclasses

    h64 = [_h64_plans(pdl) for pdl in (False, True)]
    h128 = [_h128_plans(pdl) for pdl in (False, True)]
    rule = cake._K64_SPLIT_RULES["sm_107a"]
    splits = tuple((s, 45) for s, _ in rule.splits)
    changed = dataclasses.replace(rule, splits=splits, max_tokens_no_pdl=13)
    for table in ({"sm_107a": changed}, {}):
        monkeypatch.setattr(cake, "_K64_SPLIT_RULES", table)
        assert [_h64_plans(pdl) for pdl in (False, True)] == h64
    monkeypatch.undo()
    monkeypatch.setattr(cake, "_K64_SPLIT_RULES_H64", {})
    assert [_h128_plans(pdl) for pdl in (False, True)] == h128
    assert set(_h64_plans()) == {None}


@pytest.mark.cpu_only
def test_k64_split_plan_h64_without_pdl_follows_its_own_ceiling(monkeypatch):
    """64-head launches without PDL split up to their rule's PDL-off row
    ceiling (0 on main: none; 7 or the PDL ceiling here) with the same pieces
    as with PDL."""
    import dataclasses

    rules = cake._K64_SPLIT_RULES_H64["sm_107a"]
    ceiling = tuple(
        dataclasses.replace(rule, max_tokens_no_pdl=min(7, rule.max_tokens_pdl))
        for rule in rules
    )
    monkeypatch.setattr(cake, "_K64_SPLIT_RULES_H64", {"sm_107a": ceiling})
    for width in _H64_SPLIT_WIDTHS:
        for tokens in range(0, 22):
            shape = dict(num_heads=64, num_query_tokens=tokens, sparse_topk=width)
            expected = _split_plan(**shape) if tokens <= 7 else None
            assert _split_plan(**shape, enable_pdl=False) == expected, (width, tokens)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "change",
    [
        {"arch": "sm_100a"},
        {"arch": "sm_103a"},
        {"arch": "sm_120a"},
        {"route": "fp8_h128_prefill_source_persistent"},
        {"route": "fp8_h64_prefill_source_persistent_m64"},
        {"route": "fp8_h64_prefill_source_persistent_m64_multi_tile"},
        {"route": "fp8_h64_source_exact"},
        {"route": "fp8_lowhead_h64"},
        {"num_heads": 128},
        {"num_heads": 32},
        {"num_heads": 16},
        {"num_heads": 8},
        {"sparse_topk": 384},
        {"sparse_topk": 256},
        {"sparse_topk": 128},
        {"enable_pdl": False},
        {"num_query_tokens": 0},
        {"num_query_tokens": 10},
    ],
    ids=lambda change: ",".join(f"{k}={v}" for k, v in change.items()),
)
def test_k64_split_plan_h64_keeps_every_other_launch(change):
    """A 64-head SM107 PDL launch of 8 rows over 640-wide items splits into
    five pieces; another architecture or route, another head count, an item of
    at most three tiles, a launch without PDL and rows past the five-tile rule
    keep the route."""
    shape = dict(num_heads=64, sparse_topk=640)
    assert _split_plan(**shape) == (0, 8, 5)
    assert _split_plan(**{**shape, **change}) is None


@pytest.mark.cpu_only
@pytest.mark.parametrize("sm_count", [148, 208, 212, 216])
@pytest.mark.parametrize("kv4", [False, True])
def test_h64_split_rows_already_take_the_uniform_route(sm_count, kv4):
    """Every 64-head SM107 decode launch the 64-head rules split already runs
    the uniform route, so the split needs no route change: every row count of
    every rule, both widths of each tile count, q_len 1 and 4, dense and
    ragged, both cache layouts, with and without the four-stage M64 sibling."""
    for sparse_topk in _H64_SPLIT_WIDTHS:
        for tokens in _H64_SPLIT_PIECES[-(-sparse_topk // 128)]:
            for max_q_len, ragged in ((1, False), (4, False), (4, True)):
                if not ragged and tokens % max_q_len:
                    continue
                for page_size in (64, 2):
                    route = _route(
                        dtype=torch.float8_e4m3fn,
                        num_heads=64,
                        max_q_len=max_q_len,
                        ragged=ragged,
                        sparse_topk=sparse_topk,
                        batch_size=-(-tokens // max_q_len),
                        compressed_page_size=page_size,
                        num_query_tokens=tokens,
                        sm_count=sm_count,
                        arch="sm_107a",
                        kv4=kv4,
                    )
                    assert route == cake._K64_SPLIT_ROUTE, (sparse_topk, tokens)


@pytest.mark.cpu_only
@pytest.mark.parametrize("kv4", [False, True])
def test_h64_split_leaves_three_tile_and_prefill_rows(kv4):
    """Three-tile decode items run the multi-tile M64 route with the
    four-stage sibling and the uniform route without it, with no rule either
    way; prefill rows (at least 257 query tokens per request, past every
    rule) run the multi-tile M64 route.  None of them splits."""
    shapes = [(tokens, 384, 1, False) for tokens in (1, 4, 8, 16)]
    shapes += [(257, 640, 257, True), (512, 1152, 512, False)]
    for tokens, sparse_topk, max_q_len, ragged in shapes:
        route = _route(
            dtype=torch.float8_e4m3fn,
            num_heads=64,
            max_q_len=max_q_len,
            ragged=ragged,
            sparse_topk=sparse_topk,
            batch_size=-(-tokens // max_q_len),
            compressed_page_size=64,
            num_query_tokens=tokens,
            sm_count=212,
            arch="sm_107a",
            kv4=kv4,
        )
        uniform = not kv4 and max_q_len < 257
        expected = _UNIFORM_ROUTE if uniform else _M64_MULTI_TILE_ROUTE
        assert route == expected, (route, tokens)
        plan = _split_plan(
            route=route, num_heads=64, num_query_tokens=tokens, sparse_topk=sparse_topk
        )
        assert plan is None, (route, tokens)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_h64_workspace_requirement_covers_the_split_launch(arch):
    """At a 64-head split shape the requirement reports the split carve on
    every architecture, as for 128 heads (the number reads neither the
    architecture nor PDL; only SM107 launches the split); elsewhere it reports
    the route's own numbers, and get_cake_dsv4_workspace_bytes bounds every
    carve."""
    for sparse_topk in (*_H64_SPLIT_WIDTHS, 384):
        for tokens in (*range(1, 22), 24):
            req = cake.cake_dsv4_workspace_requirement(
                dtype=torch.float8_e4m3fn,
                num_heads=64,
                num_query_tokens=tokens,
                sm_count=212,
                sparse_topk=sparse_topk,
                compressed_page_size=64,
                max_q_len=1,
                batch_size=tokens,
                ragged=False,
                arch=arch,
            )
            assert req.route == _UNIFORM_ROUTE
            own = cake._launch_workspace_bytes(req.plan, tokens, 64, 0)
            pieces = _planned_pieces(64, sparse_topk, tokens)
            if pieces is None:
                assert req.single_launch_bytes == own, (tokens, sparse_topk)
                continue
            carve = cake_dsv4_workspace_layout(tokens, 64, pieces).total_bytes
            assert req.single_launch_bytes == carve > own
            assert carve <= get_cake_dsv4_workspace_bytes(
                tokens, 64, sparse_topk, torch.float8_e4m3fn
            )
    # The largest carve: 17 rows of four pieces over seven tiles (68 pieces).
    assert (
        max(
            cake._k64_split_workspace_bytes(
                _UNIFORM_ROUTE, num_heads=64, num_query_tokens=tokens, sparse_topk=width
            )
            for width in _H64_SPLIT_WIDTHS
            for tokens in range(1, 22)
        )
        == 4_737_024
    )


@pytest.mark.cpu_only
def test_h64_split_requirement_below_69_sms_follows_the_route():
    """Below 69 SMs a 64-head split shape can take an M64 route, which never
    splits: 18 rows over six tiles with the four-stage sibling take it at 68
    SMs and report that route's own bytes; at 69 SMs, without the sibling or
    off SM107 they take the uniform route and report the three-piece carve."""
    shape = dict(
        dtype=torch.float8_e4m3fn,
        num_heads=64,
        num_query_tokens=18,
        sparse_topk=768,
        compressed_page_size=64,
        max_q_len=1,
        batch_size=18,
        ragged=False,
    )
    carve = cake_dsv4_workspace_layout(18, 64, 3).total_bytes
    m64 = cake.cake_dsv4_workspace_requirement(
        **shape, sm_count=68, arch="sm_107a", kv4=True
    )
    assert m64.route == _M64_MULTI_TILE_ROUTE
    own = cake._launch_workspace_bytes(m64.plan, 18, 64, 0)
    assert m64.single_launch_bytes == own < carve
    for change in (
        dict(sm_count=69, arch="sm_107a", kv4=True),
        dict(sm_count=68, arch="sm_107a", kv4=False),
        dict(sm_count=68, arch="sm_100a", kv4=False),
    ):
        req = cake.cake_dsv4_workspace_requirement(**shape, **change)
        assert req.route == _UNIFORM_ROUTE, change
        assert req.single_launch_bytes == carve, change


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "tokens,sparse_topk",
    [
        (1, 512),
        (4, 388),
        (9, 640),
        (9, 768),
        (18, 644),
        (10, 896),
        (17, 772),
        (7, 1024),
        (20, 900),
        (6, 1152),
        (12, 1028),
        (20, 1152),
    ],
)
def test_run_cake_dsv4_h64_split_launch_binds_the_piece_views(
    monkeypatch, tokens, sparse_topk
):
    """A 64-head split launch binds like a 128-head one with H = 64: the
    producer, two CTAs per piece, stores through the [T * s, 64, 512] partial-O
    view at byte 263168 and the LSE right after it; the merge, one CTA per
    (row, head), reads the same views and writes O; both bind the call's PDL
    and run in one run_sequence call."""
    splits = _planned_pieces(64, sparse_topk, tokens)
    pieces = tokens * splits
    inputs = _split_inputs(tokens, num_heads=64, sparse_topk=sparse_topk)
    record = _run_split(monkeypatch, inputs)
    assert record.sequences == [2]
    (producer, p), (merge, m) = record.calls
    assert (producer, merge) == (cake._K64_SPLIT_VARIANT, cake._K64_SPLIT_MERGE)
    base = inputs["workspace_buffer"].data_ptr()
    out = inputs["out"]
    lse_offset = 263168 + pieces * 65536
    layout = cake_dsv4_workspace_layout(tokens, 64, splits)
    assert layout.partial_o == (263168, pieces * 64 * 512 * 2)
    assert layout.partial_lse == (lse_offset, pieces * 64 * 4)
    # Producer.
    view = p["tmap_partial_o"]
    assert view.data_ptr() == base + 263168 and view.is_contiguous()
    assert view.shape == (pieces, 64, 512) and view.dtype == torch.bfloat16
    assert p["partial_lse"].data_ptr() == base + lse_offset
    assert p["partial_lse"].numel() == pieces * 64
    assert p["partial_lse"].dtype == torch.float32
    assert p["tmap_o"] is p["O"] and p["O"].data_ptr() == out.data_ptr()
    assert p["O"].shape == (tokens, 64, 512)
    # The 64-head query rows: the producer's second 64-head Q box lies past
    # them (zero-filled by the tensor map).
    assert p["tmap_q"].data_ptr() == inputs["query"].data_ptr()
    assert p["tmap_q"].shape == (tokens, 64, 512)
    assert (p["grid_x"], p["grid_y"], p["grid_z"]) == (2 * pieces, 1, 1)
    assert p["total_work_items"] == pieces and p["num_whole_items"] == 0
    assert p["num_splits"] == splits and p["num_query_tokens"] == tokens
    assert p["num_heads"] == 64 and p["sparse_topk"] == sparse_topk
    assert p["enable_pdl"] == 1
    # Merge.
    assert m["partial_O"].data_ptr() == base + 263168
    assert m["partial_O"].numel() == pieces * 64 * 512
    assert m["partial_lse"].data_ptr() == base + lse_offset
    assert m["partial_lse"].numel() == pieces * 64
    assert m["O"].data_ptr() == out.data_ptr() and m["O"].shape == (tokens, 64, 512)
    assert m["num_splits"] == splits and m["num_heads"] == 64
    assert (m["grid_x"], m["grid_y"], m["grid_z"]) == (tokens, 64, 1)
    assert m["enable_pdl"] == 1


@pytest.mark.cpu_only
@pytest.mark.parametrize("tokens,sparse_topk", [(1, 1152), (9, 640), (20, 1024)])
@pytest.mark.parametrize(
    "fallback", ["workspace", "no_pdl", "sm_100a", "sm_103a", "gate", "unrouted"]
)
def test_h64_split_fallbacks_launch_the_route_as_main(
    monkeypatch, tokens, sparse_topk, fallback
):
    """As for 128 heads: a workspace one byte short of the split carve, a call
    without PDL, SM100 / SM103, a process that cannot launch the K=64 programs
    and a registry without the split programs each run the route's own single
    launch of a 64-head split shape, argument for argument as without the
    split."""
    splits = _planned_pieces(64, sparse_topk, tokens)
    carve = cake_dsv4_workspace_layout(tokens, 64, splits).total_bytes
    inputs = _split_inputs(
        tokens,
        num_heads=64,
        sparse_topk=sparse_topk,
        workspace_bytes=carve - 1 if fallback == "workspace" else carve,
    )
    arch = fallback if fallback.startswith("sm_") else "sm_107a"
    enable_pdl = fallback != "no_pdl"
    asked = []
    if fallback == "gate":
        monkeypatch.setattr(
            cake,
            "_k64_sibling_launchable",
            lambda gate_arch, device: asked.append((gate_arch, device)) or False,
        )
    got = _run_split(
        monkeypatch,
        inputs,
        arch=arch,
        enable_pdl=enable_pdl,
        routed=fallback != "unrouted",
    )
    main = _run_split(
        monkeypatch, inputs, arch=arch, enable_pdl=enable_pdl, routed=False
    )
    assert _launch_record(got.calls) == _launch_record(main.calls)
    assert got.sequences == []
    ((variant, args),) = got.calls
    assert variant == _UNIFORM_ROUTE
    assert (args["grid_x"], args["grid_y"], args["grid_z"]) == (2 * tokens, 1, 1)
    assert args["num_heads"] == 64 and args["total_work_items"] == tokens
    base = inputs["workspace_buffer"].data_ptr()
    assert args["partial_lse"].data_ptr() == base + 263168
    assert args["partial_lse"].numel() == tokens * 64
    assert args["O"].data_ptr() == inputs["out"].data_ptr()
    assert args["enable_pdl"] == int(enable_pdl)
    assert asked == ([("sm_107a", torch.device("cpu"))] if fallback == "gate" else [])


# --------------------------------------------------------------------------- #
# Early-PV sibling: SM107's low-head prefill program up to a row ceiling      #
# --------------------------------------------------------------------------- #

_LOWHEAD_PREFILL_ROUTE = "fp8_lowhead_prefill"
_OVERLAP_SIBLING = f"{_LOWHEAD_PREFILL_ROUTE}_overlap"
_VBLOCK_SIBLING = f"{_LOWHEAD_PREFILL_ROUTE}_vblock"
_OVERLAP_TOKENS = (1, 12, 48, 61, 62, 63, 64, 65, 72, 80, 104, 108)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
@pytest.mark.parametrize(
    "sm_count,ceiling", [(216, 64), (212, 62), (208, 61), (264, 64)]
)
def test_sm107a_lowhead_launches_take_the_overlap_sibling_up_to_its_ceiling(
    monkeypatch, arch, sm_count, ceiling
):
    """SM107 routes the low-head prefill route's early-PV sibling to a program
    of its own, with the route's arguments, and runs it on launches of at most
    64 rows on 216 SMs, a ceiling scaled down with a smaller SM count and kept
    on a larger one; SM100 and SM103 route the sibling to the route's program
    and launch the route."""
    from flashinfer.jit.cake_dsv4 import VARIANTS

    monkeypatch.setattr(cake, "_device_sm_count", lambda device: sm_count)
    assert {_LOWHEAD_PREFILL_ROUTE} == cake._OVERLAP_SIBLING_ROUTES
    assert cake._OVERLAP_SIBLING_MAX_TOKENS == {"sm_107a": 64}
    assert cake._OVERLAP_SIBLING_MEASURED_SMS == {"sm_107a": 216}
    route, sibling = _LOWHEAD_PREFILL_ROUTE, _OVERLAP_SIBLING
    own = arch == "sm_107a"
    assert (VARIANTS[arch][sibling] != VARIANTS[arch][route]) == own
    assert (
        get_cake_dsv4_spec(sibling, arch=arch)["arg_plan"]
        == get_cake_dsv4_spec(route, arch=arch)["arg_plan"]
    )
    for tokens in _OVERLAP_TOKENS:
        expected = sibling if own and tokens <= ceiling else route
        variant = cake._overlap_variant(
            route, arch=arch, num_query_tokens=tokens, device=_DEVICE
        )
        assert variant == expected, tokens
    for other in (
        "fp8_lowhead_one_partition",
        "fp8_lowhead_h64",
        "fp8_h128_prefill_source_persistent_uniform",
        "fp8_h64_prefill_source_persistent_m64",
        "fp8_h64_prefill_source_persistent_m64_multi_tile",
    ):
        variant = cake._overlap_variant(
            other, arch=arch, num_query_tokens=12, device=_DEVICE
        )
        assert variant == other


@pytest.mark.cpu_only
def test_overlap_variant_needs_a_sibling_program_of_its_own(monkeypatch):
    """A launch takes the early-PV sibling only where its architecture routes
    the sibling to a program other than the route's: a table without the
    sibling, one that routes it to the route's program and one of another
    architecture run the route; an architecture without a ceiling never reads
    the device's SM count."""
    import flashinfer.jit.cake_dsv4 as jit

    monkeypatch.setattr(cake, "_device_sm_count", lambda device: 216)
    route, sibling = _LOWHEAD_PREFILL_ROUTE, _OVERLAP_SIBLING
    low_head, early_pv = "cake_dsv4_low_head", "cake_dsv4_early_pv"
    for table, expected in (
        ({"sm_107a": {route: low_head, sibling: early_pv}}, sibling),
        ({"sm_107a": {route: low_head}}, route),
        ({"sm_107a": {route: low_head, sibling: low_head}}, route),
        ({"sm_103a": {route: low_head, sibling: early_pv}}, route),
    ):
        monkeypatch.setattr(jit, "VARIANTS", table)
        variant = cake._overlap_variant(
            route, arch="sm_107a", num_query_tokens=12, device=_DEVICE
        )
        assert variant == expected

    def no_sm_count(device):
        raise AssertionError("read the SM count")

    monkeypatch.setattr(cake, "_device_sm_count", no_sm_count)
    for arch in ("sm_100a", "sm_103a"):
        variant = cake._overlap_variant(
            route, arch=arch, num_query_tokens=12, device=_DEVICE
        )
        assert variant == route


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
@pytest.mark.parametrize("tokens", [12, 27, 28, 64, 65, 108])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_fp8_lowhead_prefill_launch_takes_the_overlap_sibling(
    monkeypatch, arch, tokens, enable_pdl
):
    """A low-head prefill launch on 216 SMs runs one CTA per work item over
    two items per row with ``partials(1)``; SM107 runs it on the V-block
    sibling over four items per row up to 27 rows, on the early-PV sibling
    up to 64 rows and on the route's program above, with either PDL switch."""
    monkeypatch.setattr(cake, "_device_sm_count", lambda device: _SM107_SM_COUNT)
    L = _RecordingLauncher(
        arch,
        Q=torch.empty(0),
        num_query_tokens=tokens,
        num_heads=64,
        sparse_topk=128,
        max_q_len=4,
        enable_pdl=enable_pdl,
    )
    cake._dispatch_route(_LOWHEAD_PREFILL_ROUTE, L)
    program = (
        _OVERLAP_SIBLING
        if arch == "sm_107a" and tokens <= 64
        else _LOWHEAD_PREFILL_ROUTE
    )
    items = 2 * tokens
    if arch == "sm_107a" and tokens <= 27:
        program, items = _VBLOCK_SIBLING, 4 * tokens
    work = {"grid": (items, 1, 1), "total_work_items": items}
    assert L.calls == [(program, {**work, **L.partials(1)})]


@pytest.mark.cpu_only
def test_sm107a_units_define_and_launch_their_own_program():
    """Every sm_107a program's kernel unit defines ``kernel_<program>`` and its
    binding names and launches that kernel; the early-PV sibling is a program
    of its own whose binding launches it with the dynamic shared memory of the
    route's program, in the standard mode."""
    import re

    from flashinfer.jit.cake_dsv4 import MODULES, VARIANTS, _get_csrc_dir

    csrc_dir = _get_csrc_dir()
    smem = {}
    for program in sorted(set(VARIANTS["sm_107a"].values())):
        kernel_unit = f"sm_107a/{program}_kernel.cu"
        binding_unit = f"sm_107a/{program}_binding.cu"
        assert {kernel_unit, binding_unit} <= set(MODULES[program]["sources"])
        kernel = (csrc_dir / kernel_unit).read_text(encoding="utf-8")
        binding = (csrc_dir / binding_unit).read_text(encoding="utf-8")
        assert f"kernel_{program}(" in kernel, program
        assert f'KERNEL_NAME = "kernel_{program}";' in binding, program
        launched = re.findall(
            r"reinterpret_cast<const void\*>\((kernel_\w+)\)", binding
        )
        assert launched and set(launched) == {f"kernel_{program}"}, program
        smem[program] = re.findall(r"config\.dynamicSmemBytes = (\d+);", binding)
    route = VARIANTS["sm_107a"][_LOWHEAD_PREFILL_ROUTE]
    sibling = VARIANTS["sm_107a"][_OVERLAP_SIBLING]
    assert sibling != route
    assert smem[sibling] == smem[route] == ["205952"]
    binding = (csrc_dir / f"sm_107a/{sibling}_binding.cu").read_text(encoding="utf-8")
    assert "cudaSharedMemoryModeAllowOversizedSharedMemory" not in binding


@pytest.mark.cpu_only
def test_sm107a_one_wave_lowhead_swa_rows_launch_the_overlap_sibling(monkeypatch):
    """A dense 64-head SWA-only call of one wave (30 x 2 = 60 rows on 216 SMs)
    runs the early-PV sibling on SM107 with the route's launch: two CTAs per
    row, ``seq_lens`` in place of the never-read offsets and the route's
    output phase."""
    program, bound, seq_lens, out = _run_fake_lowhead_swa_fp8(
        monkeypatch, arch="sm_107a", num_heads=64, batch_size=30, q_len=2
    )
    assert program == _OVERLAP_SIBLING
    assert (bound["grid_x"], bound["grid_y"], bound["grid_z"]) == (120, 1, 1)
    assert bound["total_work_items"] == 120 and bound["num_query_tokens"] == 60
    assert bound["cum_seq_lens_q"].data_ptr() == seq_lens.data_ptr()
    assert bound["O"].data_ptr() == out.data_ptr()
    assert out.data_ptr() % 4096 == 0x800


@pytest.mark.cpu_only
def test_sm107a_lowhead_row_tiles_pick_their_program_per_launch(monkeypatch):
    """A 100-row FP8 64-head SWA-only call on SM107 (216 SMs) whose workspace
    holds 70 rows runs two launches: the first 70 rows on the route's program
    (above the 64-row ceiling) and the last 30 on the early-PV sibling, each
    with two CTAs and two work items per row."""
    route, sibling = _LOWHEAD_PREFILL_ROUTE, _OVERLAP_SIBLING
    recorder = _install_fake_variants(
        monkeypatch, {route: _PERSISTENT_PLAN, sibling: _PERSISTENT_PLAN}
    )
    seen = []

    def module(variant, **kwargs):
        seen.append(variant)
        return recorder

    monkeypatch.setattr(cake, "_variant_module", module)
    monkeypatch.setattr(cake, "_target_arch", lambda device: "sm_107a")
    monkeypatch.setattr(cake, "_device_sm_count", lambda device: _SM107_SM_COUNT)
    batch_size, q_len, num_heads = 25, 4, 64
    rows = batch_size * q_len
    plan = cake._route_plan(
        route, num_query_tokens=rows, num_heads=num_heads, sparse_topk=128
    )
    workspace_bytes = cake._launch_workspace_bytes(plan, 70, num_heads, batch_size + 1)
    assert (
        cake._rows_per_launch(
            plan,
            num_query_tokens=rows,
            num_heads=num_heads,
            batch_size=batch_size,
            workspace_bytes=workspace_bytes,
        )
        == 70
    )
    table, lens = _combined_metadata(rows, 0)
    cake.run_cake_dsv4(
        query=torch.zeros((rows, num_heads, 512), dtype=torch.float8_e4m3fn),
        swa_kv_cache=torch.zeros((4, 1, 256, 512), dtype=torch.float8_e4m3fn),
        compressed_kv_cache=torch.zeros((4, 1, 256, 512), dtype=torch.float8_e4m3fn),
        workspace_buffer=_aligned_u8(workspace_bytes),
        out=None,
        bmm1_scale=0.5,
        bmm2_scale=1.0,
        sinks=None,
        max_q_len=q_len,
        cum_seq_lens_q=None,
        seq_lens=torch.full((batch_size,), 1000, dtype=torch.int32),
        backend="cake",
        sparse_indices=table,
        sparse_topk_lens=lens,
    )
    names = [name for _, name in _PERSISTENT_PLAN]
    launches = [
        (variant, bound["grid_x"], bound["total_work_items"], bound["num_query_tokens"])
        for variant, bound in zip(
            seen,
            (dict(zip(names, call, strict=True)) for call in recorder.calls),
            strict=True,
        )
    ]
    assert launches == [(route, 140, 140, 70), (sibling, 60, 60, 30)]


# --------------------------------------------------------------------------- #
# SM107 per-count merges of the 128-head split launches                       #
# --------------------------------------------------------------------------- #


@pytest.mark.cpu_only
def test_only_the_128_head_rule_names_a_count_merge():
    """The 128-head SM107 rule names the merge for nine pieces from 1 row, for
    five from 5 and the three-piece merge from 16 (three pieces at 9-15 rows
    keep the merge of any count); no 64-head rule names a merge, no rule
    names two merges for one (pieces, fewest rows), every per-count merge a
    rule names is built for the pieces of its entry, and the per-count merges
    are exactly those the rules name."""
    assert cake._K64_SPLIT_RULES["sm_107a"].merge_by_splits == (
        (9, 1, _SPLIT_MERGE_C9),
        (5, 5, _SPLIT_MERGE_C5),
        (3, 16, _SPLIT_MERGE_S3),
    )
    for rules in cake._K64_SPLIT_RULES_H64.values():
        assert all(rule.merge_by_splits == () for rule in rules)
    named = set()
    for rule in cake._K64_SPLIT_RULES.values():
        bands = [(s, fewest) for s, fewest, _ in rule.merge_by_splits]
        assert len(set(bands)) == len(bands), bands
        for s, _, key in rule.merge_by_splits:
            if key != cake._K64_SPLIT_MERGE:
                built_for = (
                    f"{cake._K64_SPLIT_MERGE}_c{s}",
                    f"{cake._K64_SPLIT_MERGE}_s{s}",
                )
                assert key in built_for, (s, key)
                named.add(key)
    assert named == set(cake._K64_SPLIT_COUNT_MERGES)


@pytest.mark.cpu_only
@pytest.mark.parametrize("count_merge", ["own", "unrouted", "route", "merge"])
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_k64_split_merge_pairs_each_key_with_its_grid(monkeypatch, arch, count_merge):
    """For every split SM107 plans (128 heads over nine tiles, 64 heads over
    every width class), the per-count merges are chosen exactly for the
    128-head launches on SM107 that ``_SPLIT_MERGES`` gives them (nine pieces
    at 1-4 rows, five at 5-8 and, the three-piece merge with two heads per
    CTA, 16-20) where the registry routes them to programs of their own;
    every other launch (64 heads, three pieces at 9-15 rows, another
    architecture, the keys unrouted or retired onto the uniform route's or
    the merge's program) merges with the merge of any count, one head per
    CTA.  No launch pairs the three-piece merge with a grid of H heads or
    another merge with one of H / 2."""
    import flashinfer.jit.cake_dsv4 as jit

    monkeypatch.setattr(
        jit, "VARIANTS", _fake_split_variants(routed=True, count_merge=count_merge)
    )
    plans = [
        (128, _split_plan(num_query_tokens=tokens, sparse_topk=width))
        for width in (1028, 1152)
        for tokens in range(1, 22)
    ]
    plans += [
        (64, _split_plan(num_heads=64, num_query_tokens=tokens, sparse_topk=width))
        for width in _H64_SPLIT_WIDTHS
        for tokens in range(1, 22)
    ]
    plans = [(heads, plan) for heads, plan in plans if plan is not None]
    assert {(heads, plan[2]) for heads, plan in plans} >= {
        (128, 9),
        (128, 5),
        (128, 3),
        (64, 9),
        (64, 5),
        (64, 3),
    }
    for heads, (_, rows, splits) in plans:
        got = cake._k64_split_merge(
            arch=arch, num_heads=heads, rows=rows, num_splits=splits
        )
        count = arch == "sm_107a" and heads == 128 and count_merge == "own"
        expected = _SPLIT_MERGES[rows] if count else (cake._K64_SPLIT_MERGE, 1)
        assert got == expected, (heads, rows, splits)


@pytest.mark.cpu_only
def test_k64_split_merge_follows_the_fewest_rows(monkeypatch):
    """``merge_by_splits`` bounds the rows from below, and the entry for the
    launch's pieces with the most fewest rows at most its rows wins in any
    order: with ((3, 1, three-piece),) every three-piece launch takes the
    three-piece merge, with ((3, 17, three-piece),) the launches of 17-20
    rows, with the three-piece merge from 1 row and the merge of any count
    from 12 the launches of 9-11 rows the first and of 12-20 the second, and
    with () none.  Removing one per-count entry of the SM107 rule returns
    exactly the launches it serves to the merge of any count; the split plans
    do not change."""
    import dataclasses

    import flashinfer.jit.cake_dsv4 as jit

    monkeypatch.setattr(jit, "VARIANTS", _fake_split_variants(routed=True))
    plans = _h128_plans()
    rule = cake._K64_SPLIT_RULES["sm_107a"]

    def merges(merge_by_splits):
        changed = dataclasses.replace(rule, merge_by_splits=merge_by_splits)
        monkeypatch.setattr(cake, "_K64_SPLIT_RULES", {"sm_107a": changed})
        assert _h128_plans() == plans
        return {
            tokens: cake._k64_split_merge(
                arch="sm_107a", num_heads=128, rows=tokens, num_splits=splits
            )
            for tokens, splits in _SPLIT_PIECES.items()
        }

    any_count = (cake._K64_SPLIT_MERGE, 1)
    s3 = (_SPLIT_MERGE_S3, 2)

    def three_pieces(merge_of_rows):
        return {
            tokens: merge_of_rows(tokens) if splits == 3 else any_count
            for tokens, splits in _SPLIT_PIECES.items()
        }

    assert merges(((3, 1, _SPLIT_MERGE_S3),)) == three_pieces(lambda t: s3)
    assert merges(((3, 17, _SPLIT_MERGE_S3),)) == three_pieces(
        lambda t: s3 if t >= 17 else any_count
    )
    two = ((3, 1, _SPLIT_MERGE_S3), (3, 12, cake._K64_SPLIT_MERGE))
    for entries in (two, two[::-1]):
        assert merges(entries) == three_pieces(lambda t: any_count if t >= 12 else s3)
    assert merges(()) == dict.fromkeys(_SPLIT_PIECES, any_count)
    full = merges(rule.merge_by_splits)
    assert full == _SPLIT_MERGES
    for entry in rule.merge_by_splits:
        if entry[2] == cake._K64_SPLIT_MERGE:
            continue
        served = {tokens for tokens, merge in full.items() if merge[0] == entry[2]}
        assert served, entry
        kept = tuple(other for other in rule.merge_by_splits if other != entry)
        dropped = merges(kept)
        assert {t for t in full if dropped[t] != full[t]} == served, entry
        assert all(dropped[t] == any_count for t in served), entry


# --------------------------------------------------------------------------- #
# TMEM-batch sibling: SM107's four-stage M64 program for 64-head MTP decode   #
# --------------------------------------------------------------------------- #

_KV4_SIBLING = f"{_M64_MULTI_TILE_ROUTE}_kv4"
_BCL_SIBLING = f"{_KV4_SIBLING}_bcl"


def _bcl_variants():
    """The registry with SM107's TMEM-batch sibling routed to a program of its
    own (a stand-in name); every other entry is the registry's."""
    import flashinfer.jit.cake_dsv4 as jit

    table = {arch: dict(variants) for arch, variants in jit.VARIANTS.items()}
    table["sm_107a"][_BCL_SIBLING] = "cake_dsv4_tmem_batch"
    return table


@pytest.mark.cpu_only
def test_bcl_window_bounds():
    """The TMEM-batch window: 64 heads, a longest request of 2-4 query
    tokens, 108 rows up to two per SM and items up to 768 wide."""
    assert cake._KV4_BCL_HEADS == 64
    assert (cake._KV4_BCL_MIN_Q, cake._KV4_BCL_MAX_Q) == (2, 4)
    assert cake._KV4_BCL_MIN_TOKENS == 108
    assert cake._KV4_BCL_MAX_TOKENS_PER_SM == 2
    assert cake._KV4_BCL_MAX_TOPK == 768


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
@pytest.mark.parametrize("sm_count,ceiling", [(208, 416), (212, 424), (216, 432)])
def test_four_stage_launches_take_the_bcl_sibling_in_its_window(
    monkeypatch, arch, sm_count, ceiling
):
    """Where SM107 routes the TMEM-batch sibling to a program of its own, a
    four-stage launch takes it when the call has 64 heads, a longest request
    of 2-4 query tokens and items at most 768 wide and the launch has 108 rows
    up to two per SM of the device; every other four-stage launch keeps the
    four-stage key.  SM100 and SM103 route the sibling to the four-stage key's
    program and keep the key; no other variant ever changes."""
    import flashinfer.jit.cake_dsv4 as jit

    monkeypatch.setattr(jit, "VARIANTS", _bcl_variants())
    monkeypatch.setattr(cake, "_device_sm_count", lambda device: sm_count)
    own = arch == "sm_107a"
    for num_heads in (32, 64):
        for max_q_len in (1, 2, 3, 4, 5):
            for sparse_topk in (256, 384, 640, 768, 769, 1152):
                for tokens in (1, 96, 107, 108, 120, ceiling, ceiling + 1, 4096):
                    shape = dict(
                        num_heads=num_heads,
                        max_q_len=max_q_len,
                        num_query_tokens=tokens,
                        sparse_topk=sparse_topk,
                    )
                    takes = (
                        own
                        and num_heads == 64
                        and 2 <= max_q_len <= 4
                        and sparse_topk <= 768
                        and 108 <= tokens <= ceiling
                    )
                    variant = cake._kv4_bcl_variant(
                        _KV4_SIBLING, arch=arch, device=_DEVICE, **shape
                    )
                    assert variant == (_BCL_SIBLING if takes else _KV4_SIBLING), shape
    window = dict(num_heads=64, max_q_len=4, num_query_tokens=120, sparse_topk=640)
    for other in (
        _M64_MULTI_TILE_ROUTE,
        _BCL_SIBLING,
        "fp8_h64_prefill_source_persistent_m64",
        "fp8_lowhead_prefill",
        "fp8_lowhead_prefill_kv4",
        f"{_UNIFORM_ROUTE}_k64",
    ):
        variant = cake._kv4_bcl_variant(other, arch=arch, device=_DEVICE, **window)
        assert variant == other


@pytest.mark.cpu_only
def test_bcl_variant_needs_a_program_of_its_own(monkeypatch):
    """A four-stage launch in the window takes the TMEM-batch sibling only
    where its architecture routes the sibling to a program other than the
    four-stage one: a table without the sibling, one that routes it to the
    four-stage program and one of another architecture keep the four-stage
    key without reading the device's SM count."""
    import flashinfer.jit.cake_dsv4 as jit

    read = []

    def sm_count(device):
        read.append(device)
        return _SM107_SM_COUNT

    monkeypatch.setattr(cake, "_device_sm_count", sm_count)
    window = dict(num_heads=64, max_q_len=4, num_query_tokens=120, sparse_topk=640)
    four, tmem = "cake_dsv4_four_stage", "cake_dsv4_tmem_batch"
    for table, expected in (
        ({"sm_107a": {_KV4_SIBLING: four, _BCL_SIBLING: tmem}}, _BCL_SIBLING),
        ({"sm_107a": {_KV4_SIBLING: four}}, _KV4_SIBLING),
        ({"sm_107a": {_KV4_SIBLING: four, _BCL_SIBLING: four}}, _KV4_SIBLING),
        ({"sm_103a": {_KV4_SIBLING: four, _BCL_SIBLING: tmem}}, _KV4_SIBLING),
    ):
        monkeypatch.setattr(jit, "VARIANTS", table)
        variant = cake._kv4_bcl_variant(
            _KV4_SIBLING, arch="sm_107a", device=_DEVICE, **window
        )
        assert variant == expected
    assert read == [_DEVICE]


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_bcl_sibling_shares_the_four_stage_contract(arch):
    """SM100 and SM103 route the TMEM-batch sibling to their four-stage key's
    program, the route's own, so none of their launches takes it.  Where SM107
    names it, it is a program of its own built for SM107 only.  Wherever it
    is named it binds the four-stage sibling's argument plan, and a launch in
    the window takes it exactly where its program is not the four-stage
    one."""
    from flashinfer.jit.cake_dsv4 import MODULES, VARIANTS

    variants = VARIANTS[arch]
    four_stage = variants[_KV4_SIBLING]
    bcl = variants.get(_BCL_SIBLING, four_stage)
    if arch == "sm_107a":
        if bcl != four_stage:
            assert MODULES[bcl]["arches"] == ["sm_107a"]
    else:
        assert bcl == four_stage == variants[_M64_MULTI_TILE_ROUTE]
        assert _BCL_SIBLING in variants
    if _BCL_SIBLING in variants:
        assert MODULES[bcl]["arg_plan"] == MODULES[four_stage]["arg_plan"]
        assert (
            get_cake_dsv4_spec(_BCL_SIBLING, arch=arch)["arg_plan"]
            == get_cake_dsv4_spec(_KV4_SIBLING, arch=arch)["arg_plan"]
        )
    window = dict(num_heads=64, max_q_len=4, num_query_tokens=120, sparse_topk=640)
    variant = cake._kv4_bcl_variant(_KV4_SIBLING, arch=arch, device=_DEVICE, **window)
    assert variant == (_BCL_SIBLING if bcl != four_stage else _KV4_SIBLING)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
@pytest.mark.parametrize(
    "route", ["fp8_h64_prefill_source_persistent_m64", _M64_MULTI_TILE_ROUTE]
)
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("launchable", [True, False])
def test_fp8_h64_m64_launch_takes_the_bcl_sibling(
    monkeypatch, arch, route, enable_pdl, launchable
):
    """On 216 SMs with the TMEM-batch sibling routed to a program of its own,
    SM107 runs the multi-tile M64 launches of 108-432 rows of a 64-head call
    whose longest request has 2-4 tokens on it, over the four-stage sibling's
    grid and partials, with either PDL switch.  Its other multi-tile launches
    run the four-stage sibling; where the process cannot launch that sibling,
    and for the single-tile program and on SM100 and SM103, the route's
    program runs."""
    import flashinfer.jit.cake_dsv4 as jit

    monkeypatch.setattr(jit, "VARIANTS", _bcl_variants())
    monkeypatch.setattr(cake, "_device_sm_count", lambda device: _SM107_SM_COUNT)
    monkeypatch.setattr(
        cake, "_kv4_sibling_launchable", lambda gate_arch, device: launchable
    )
    four_stage = arch == "sm_107a" and route == _M64_MULTI_TILE_ROUTE and launchable
    for tokens in (96, 107, 108, 432, 433):
        for max_q_len in (1, 2, 4, 5):
            L = _RecordingLauncher(
                arch,
                Q=torch.empty(0),
                num_query_tokens=tokens,
                num_heads=64,
                sparse_topk=640,
                max_q_len=max_q_len,
                enable_pdl=enable_pdl,
            )
            cake._dispatch_route(route, L)
            window = 108 <= tokens <= 432 and 2 <= max_q_len <= 4
            if four_stage:
                program = _BCL_SIBLING if window else _KV4_SIBLING
            else:
                program = route
            launch = {"grid": (tokens, 1, 1), "total_work_items": tokens}
            assert L.calls == [(program, {**launch, **L.partials(1)})], (
                tokens,
                max_q_len,
            )


@pytest.mark.cpu_only
def test_run_cake_dsv4_launches_the_bcl_sibling_in_its_window(monkeypatch):
    """run_cake_dsv4 on SM107 (208 SMs: up to 416 rows) with the TMEM-batch
    sibling routed to a program of its own: dense FP8/H64 decode calls on the
    multi-tile M64 route launch it inside the window and the four-stage
    sibling outside it (96, 104 or 420 rows, one-token requests, items 772
    wide), one CTA per row either way, with the call's request length bound."""
    import flashinfer.jit.cake_dsv4 as jit

    monkeypatch.setattr(jit, "VARIANTS", _bcl_variants())
    for width, tokens, max_q_len, program in (
        (640, 108, 4, _BCL_SIBLING),
        (384, 120, 3, _BCL_SIBLING),
        (768, 416, 2, _BCL_SIBLING),
        (640, 96, 4, _KV4_SIBLING),
        (640, 104, 4, _KV4_SIBLING),
        (640, 420, 4, _KV4_SIBLING),
        (640, 120, 1, _KV4_SIBLING),
        (772, 120, 4, _KV4_SIBLING),
    ):
        shape = (width, tokens, max_q_len)
        programs, calls, _ = _run_fake_h64_decode_fp8(
            monkeypatch,
            arch="sm_107a",
            width=width,
            tokens=tokens,
            max_q_len=max_q_len,
        )
        (bound,) = calls
        assert programs == [program], shape
        assert (bound["grid_x"], bound["grid_y"], bound["grid_z"]) == (tokens, 1, 1)
        assert bound["total_work_items"] == bound["num_query_tokens"] == tokens
        assert bound["max_q_len"] == max_q_len, shape


@pytest.mark.cpu_only
def test_row_tiled_mtp_decode_chunks_pick_their_program_per_launch(monkeypatch):
    """A workspace of 120 rows tiles a 200-row FP8/H64 call of 4-token
    requests over items 640 wide (208 SMs) into 120 + 80 rows: the first chunk
    runs the TMEM-batch sibling and the second, below 108 rows, the four-stage
    sibling, one CTA per row each."""
    import flashinfer.jit.cake_dsv4 as jit

    monkeypatch.setattr(jit, "VARIANTS", _bcl_variants())
    tokens, max_q_len = 200, 4
    batch_size = tokens // max_q_len
    plan = cake._route_plan(
        _M64_MULTI_TILE_ROUTE, num_query_tokens=tokens, num_heads=64, sparse_topk=640
    )
    chunk_bytes = cake._launch_workspace_bytes(plan, 120, 64, batch_size + 1)
    rows = cake._rows_per_launch(
        plan,
        num_query_tokens=tokens,
        num_heads=64,
        batch_size=batch_size,
        workspace_bytes=chunk_bytes,
    )
    assert rows == 120
    programs, calls, _ = _run_fake_h64_decode_fp8(
        monkeypatch,
        arch="sm_107a",
        width=640,
        tokens=tokens,
        max_q_len=max_q_len,
        workspace=_aligned_u8(chunk_bytes),
    )
    assert programs == [_BCL_SIBLING, _KV4_SIBLING]
    grids = [(c["grid_x"], c["grid_y"], c["grid_z"]) for c in calls]
    assert grids == [(120, 1, 1), (80, 1, 1)]
    assert [c["total_work_items"] for c in calls] == [120, 80]


# --------------------------------------------------------------------------- #
# TMEM-batch sibling: SM107's one-token decode window (c4, items 1152 wide)    #
# --------------------------------------------------------------------------- #


@pytest.mark.cpu_only
def test_bcl_decode_window_bounds():
    """One decode window: 64 heads, one-token requests over items 1152 wide
    with compressed pages of 64 rows (the c4 layer), 176-208 rows, at most one
    row per SM, on devices of 212 or 216 SMs."""
    assert cake._KV4_BCL_DECODE_WINDOWS == {(1, 1152, 64): (176, 208)}
    assert frozenset({212, 216}) == cake._KV4_BCL_DECODE_SM_COUNTS


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
@pytest.mark.parametrize("sm_count", [148, 208, 212, 216, 232])
def test_four_stage_launches_take_the_bcl_sibling_in_its_decode_window(
    monkeypatch, arch, sm_count
):
    """Where SM107 routes the TMEM-batch sibling to a program of its own, a
    four-stage launch of a 64-head call of one-token requests over items 1152
    wide with compressed pages of 64 rows takes it at 176-208 rows on 212 or
    216 SMs.  The same launch on other SM counts, at 175 or 209 rows, with
    compressed pages of 2 rows or of an unknown size, over other widths, of
    two-token requests or of other head counts keeps the four-stage key, and
    SM100 and SM103 keep it."""
    import flashinfer.jit.cake_dsv4 as jit

    monkeypatch.setattr(jit, "VARIANTS", _bcl_variants())
    monkeypatch.setattr(cake, "_device_sm_count", lambda device: sm_count)
    own = arch == "sm_107a"
    for num_heads in (32, 64, 128):
        for max_q_len in (1, 2):
            for sparse_topk in (1024, 1151, 1152, 1153):
                for page in (None, 2, 64):
                    for tokens in (144, 160, 175, 176, 177, 192, 207, 208, 209, 256):
                        shape = dict(
                            num_heads=num_heads,
                            max_q_len=max_q_len,
                            num_query_tokens=tokens,
                            sparse_topk=sparse_topk,
                        )
                        takes = (
                            own
                            and num_heads == 64
                            and max_q_len == 1
                            and sparse_topk == 1152
                            and page == 64
                            and 176 <= tokens <= 208
                            and tokens <= sm_count
                            and sm_count in (212, 216)
                        )
                        variant = cake._kv4_bcl_variant(
                            _KV4_SIBLING,
                            arch=arch,
                            device=_DEVICE,
                            compressed_page_size=page,
                            **shape,
                        )
                        expected = _BCL_SIBLING if takes else _KV4_SIBLING
                        assert variant == expected, (shape, page)


@pytest.mark.cpu_only
@pytest.mark.parametrize("sm_count,ceiling", [(208, 416), (212, 424), (216, 432)])
@pytest.mark.parametrize("page", [None, 2, 64])
def test_bcl_mtp_window_does_not_read_the_page_size(
    monkeypatch, sm_count, ceiling, page
):
    """The MTP window (2-4 query tokens) takes the same launches whatever the
    compressed page size, as before the decode window."""
    import flashinfer.jit.cake_dsv4 as jit

    monkeypatch.setattr(jit, "VARIANTS", _bcl_variants())
    monkeypatch.setattr(cake, "_device_sm_count", lambda device: sm_count)
    for max_q_len in (2, 4):
        for sparse_topk, tokens, takes in (
            (640, 107, False),
            (640, 108, True),
            (768, ceiling, True),
            (768, ceiling + 1, False),
            (769, 120, False),
            (1152, 192, False),
        ):
            variant = cake._kv4_bcl_variant(
                _KV4_SIBLING,
                arch="sm_107a",
                device=_DEVICE,
                num_heads=64,
                max_q_len=max_q_len,
                num_query_tokens=tokens,
                sparse_topk=sparse_topk,
                compressed_page_size=page,
            )
            expected = _BCL_SIBLING if takes else _KV4_SIBLING
            assert variant == expected, (max_q_len, sparse_topk, tokens)


@pytest.mark.cpu_only
def test_bcl_decode_window_reads_the_sm_count_last(monkeypatch):
    """A decode-window shape reads the device's SM count only once its rows
    are in the window and SM107 routes the sibling to a program of its own;
    shapes outside the window never read it."""
    import flashinfer.jit.cake_dsv4 as jit

    read = []

    def sm_count(device):
        read.append(device)
        return _SM107_SM_COUNT

    monkeypatch.setattr(jit, "VARIANTS", _bcl_variants())
    monkeypatch.setattr(cake, "_device_sm_count", sm_count)
    decode = dict(num_heads=64, max_q_len=1, sparse_topk=1152)
    for tokens, page in ((175, 64), (209, 64), (192, 2), (192, None)):
        variant = cake._kv4_bcl_variant(
            _KV4_SIBLING,
            arch="sm_107a",
            device=_DEVICE,
            num_query_tokens=tokens,
            compressed_page_size=page,
            **decode,
        )
        assert variant == _KV4_SIBLING, (tokens, page)
    assert read == []
    variant = cake._kv4_bcl_variant(
        _KV4_SIBLING,
        arch="sm_107a",
        device=_DEVICE,
        num_query_tokens=192,
        compressed_page_size=64,
        **decode,
    )
    assert variant == _BCL_SIBLING
    assert read == [_DEVICE]


@pytest.mark.cpu_only
@pytest.mark.parametrize("page", [64, 2, None])
def test_fp8_h64_m64_launch_takes_the_bcl_sibling_in_its_decode_window(
    monkeypatch, page
):
    """On 216 SMs the multi-tile M64 launches of a 64-head call of one-token
    requests over items 1152 wide run the TMEM-batch sibling at 176-208 rows
    when the launch values carry compressed pages of 64 rows, over the
    four-stage sibling's grid and partials; launch values with pages of 2 rows
    or without a page size, and 175 or 209 rows, run the four-stage sibling."""
    import flashinfer.jit.cake_dsv4 as jit

    monkeypatch.setattr(jit, "VARIANTS", _bcl_variants())
    monkeypatch.setattr(cake, "_device_sm_count", lambda device: _SM107_SM_COUNT)
    page_values = {} if page is None else {"compressed_page_size": page}
    for tokens in (175, 176, 192, 208, 209):
        L = _RecordingLauncher(
            "sm_107a",
            Q=torch.empty(0),
            num_query_tokens=tokens,
            num_heads=64,
            sparse_topk=1152,
            max_q_len=1,
            **page_values,
        )
        cake._dispatch_route(_M64_MULTI_TILE_ROUTE, L)
        window = page == 64 and 176 <= tokens <= 208
        program = _BCL_SIBLING if window else _KV4_SIBLING
        launch = {"grid": (tokens, 1, 1), "total_work_items": tokens}
        assert L.calls == [(program, {**launch, **L.partials(1)})], tokens


@pytest.mark.cpu_only
def test_run_cake_dsv4_launches_the_bcl_sibling_in_its_decode_window(monkeypatch):
    """run_cake_dsv4 on SM107 with the TMEM-batch sibling routed to a program
    of its own: dense FP8/H64 one-token decode calls over items 1152 wide on
    the multi-tile M64 route launch it at 176-208 rows with compressed pages
    of 64 rows on 212 and 216 SMs, and the four-stage sibling at 175 and 209
    rows, with compressed pages of 2 rows and on 208 SMs, one CTA per row
    either way."""
    import flashinfer.jit.cake_dsv4 as jit

    monkeypatch.setattr(jit, "VARIANTS", _bcl_variants())
    for sm_count, page, tokens, program in (
        (216, 64, 176, _BCL_SIBLING),
        (216, 64, 192, _BCL_SIBLING),
        (216, 64, 208, _BCL_SIBLING),
        (212, 64, 176, _BCL_SIBLING),
        (212, 64, 208, _BCL_SIBLING),
        (216, 64, 175, _KV4_SIBLING),
        (216, 64, 209, _KV4_SIBLING),
        (212, 64, 209, _KV4_SIBLING),
        (216, 2, 192, _KV4_SIBLING),
        (212, 2, 192, _KV4_SIBLING),
        (208, 64, 192, _KV4_SIBLING),
        (208, 64, 208, _KV4_SIBLING),
    ):
        shape = (sm_count, page, tokens)
        programs, calls, _ = _run_fake_h64_decode_fp8(
            monkeypatch,
            arch="sm_107a",
            width=1152,
            tokens=tokens,
            sm_count=sm_count,
            page=page,
        )
        (bound,) = calls
        assert programs == [program], shape
        assert (bound["grid_x"], bound["grid_y"], bound["grid_z"]) == (tokens, 1, 1)
        assert bound["total_work_items"] == bound["num_query_tokens"] == tokens
        assert bound["max_q_len"] == 1, shape


# --------------------------------------------------------------------------- #
# L2 prefetch sibling: SM107's four-stage M64 program for few-row c4 decode   #
# --------------------------------------------------------------------------- #

_L2PF_SIBLING = f"{_KV4_SIBLING}_l2pf"
# The delivered rule: (sparse_topk, max_q_len) -> most rows of the launch.
_L2PF_CEILINGS = {(640, 1): 47, (640, 4): 64, (512, 1): 41, (512, 4): 56}


def _l2pf_variants():
    """The registry with SM107's L2 prefetch and TMEM-batch siblings routed to
    programs of their own (stand-in names); every other entry is the
    registry's."""
    import flashinfer.jit.cake_dsv4 as jit

    table = {arch: dict(variants) for arch, variants in jit.VARIANTS.items()}
    table["sm_107a"][_BCL_SIBLING] = "cake_dsv4_tmem_batch"
    table["sm_107a"][_L2PF_SIBLING] = "cake_dsv4_l2_prefetch"
    return table


def _l2pf_takes(
    arch, sm_count, *, num_heads, max_q_len, num_query_tokens, sparse_topk, batch, page
):
    """The delivered rule restated: SM107 with the sibling routed to a program
    of its own, 64 heads, compressed pages of 64 rows, a (width, request
    length) of the rule at most its rows, a uniform launch past one-token
    requests, 212 or 216 SMs and at most one row per SM."""
    ceiling = _L2PF_CEILINGS.get((sparse_topk, max_q_len))
    uniform = batch is not None and num_query_tokens == batch * max_q_len
    return (
        arch == "sm_107a"
        and num_heads == 64
        and page == 64
        and ceiling is not None
        and num_query_tokens <= ceiling
        and (max_q_len == 1 or uniform)
        and sm_count in (212, 216)
        and num_query_tokens <= sm_count
    )


@pytest.mark.cpu_only
def test_l2pf_window_bounds():
    """The L2 prefetch window: 64 heads, compressed pages of 64 rows (the c4
    layer), four (item width, request length) entries with their row
    ceilings, devices of 212 or 216 SMs."""
    assert cake._KV4_L2PF_HEADS == 64
    assert cake._KV4_L2PF_PAGE == 64
    assert cake._KV4_L2PF_MAX_TOKENS == _L2PF_CEILINGS
    assert frozenset({212, 216}) == cake._KV4_L2PF_SM_COUNTS


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
@pytest.mark.parametrize("sm_count", [148, 208, 212, 216, 232])
def test_four_stage_launches_take_the_l2pf_sibling_in_its_window(
    monkeypatch, arch, sm_count
):
    """Where SM107 routes the L2 prefetch sibling to a program of its own, a
    four-stage launch takes it exactly inside the delivered rule: at the row
    ceiling of each entry and not one row above, never for other widths, head
    counts, request lengths (2, 3 or 5 tokens) or page sizes (2 rows or
    unknown), never for a non-uniform or unknown request count past one-token
    requests, and never on other SM counts.  SM100 and SM103 keep the
    four-stage key."""
    import flashinfer.jit.cake_dsv4 as jit

    monkeypatch.setattr(jit, "VARIANTS", _l2pf_variants())
    monkeypatch.setattr(cake, "_device_sm_count", lambda device: sm_count)
    tokens_grid = (1, 31, 32, 40, 41, 42, 47, 48, 55, 56, 57, 58, 63, 64, 65, 66, 120)
    for num_heads in (32, 64, 128):
        for max_q_len in (1, 2, 3, 4, 5):
            for sparse_topk in (384, 511, 512, 513, 639, 640, 641, 1152):
                for page in (None, 2, 64):
                    for tokens in tokens_grid:
                        batches = {tokens // max_q_len + 1, None}
                        if tokens % max_q_len == 0:
                            batches.add(tokens // max_q_len)
                        for batch in batches:
                            shape = dict(
                                num_heads=num_heads,
                                max_q_len=max_q_len,
                                num_query_tokens=tokens,
                                sparse_topk=sparse_topk,
                            )
                            takes = _l2pf_takes(
                                arch, sm_count, batch=batch, page=page, **shape
                            )
                            variant = cake._kv4_l2pf_variant(
                                _KV4_SIBLING,
                                arch=arch,
                                device=_DEVICE,
                                batch_size=batch,
                                compressed_page_size=page,
                                **shape,
                            )
                            expected = _L2PF_SIBLING if takes else _KV4_SIBLING
                            assert variant == expected, (shape, batch, page)
    window = dict(num_heads=64, max_q_len=1, num_query_tokens=40, sparse_topk=640)
    for other in (
        _M64_MULTI_TILE_ROUTE,
        _BCL_SIBLING,
        _L2PF_SIBLING,
        "fp8_h64_prefill_source_persistent_m64",
        "fp8_lowhead_prefill",
        "fp8_lowhead_prefill_kv4",
        f"{_UNIFORM_ROUTE}_k64",
    ):
        variant = cake._kv4_l2pf_variant(
            other,
            arch=arch,
            device=_DEVICE,
            batch_size=40,
            compressed_page_size=64,
            **window,
        )
        assert variant == other


@pytest.mark.cpu_only
@pytest.mark.parametrize("sm_count", [212, 216])
@pytest.mark.parametrize(
    "sparse_topk,max_q_len,ceiling", [(w, q, t) for (w, q), t in _L2PF_CEILINGS.items()]
)
def test_l2pf_rule_edges(monkeypatch, sm_count, sparse_topk, max_q_len, ceiling):
    """Each entry takes a uniform launch at its row ceiling and keeps the
    four-stage key one request above it, for the same launch over compressed
    pages of 2 rows, on 208 SMs and, past one-token requests, as a ragged
    launch of one request fewer."""
    import flashinfer.jit.cake_dsv4 as jit

    monkeypatch.setattr(jit, "VARIANTS", _l2pf_variants())
    count = {"sm": sm_count}
    monkeypatch.setattr(cake, "_device_sm_count", lambda device: count["sm"])

    def variant(tokens, *, batch=None, page=64):
        if batch is None:
            batch = tokens // max_q_len
        return cake._kv4_l2pf_variant(
            _KV4_SIBLING,
            arch="sm_107a",
            device=_DEVICE,
            num_heads=64,
            max_q_len=max_q_len,
            num_query_tokens=tokens,
            sparse_topk=sparse_topk,
            batch_size=batch,
            compressed_page_size=page,
        )

    edge = ceiling - ceiling % max_q_len
    assert variant(edge) == _L2PF_SIBLING
    assert variant(edge + max_q_len) == _KV4_SIBLING
    assert variant(edge, page=2) == _KV4_SIBLING
    if max_q_len > 1:
        assert variant(edge - 1, batch=edge // max_q_len) == _KV4_SIBLING
    count["sm"] = 208
    assert variant(edge) == _KV4_SIBLING


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_l2pf_sibling_extends_the_four_stage_contract(arch):
    """SM100 and SM103 route the L2 prefetch sibling to their four-stage key's
    program, the route's own, so none of their launches takes it.  SM107
    routes it to a program of its own, built for SM107 only, whose argument
    plan is the four-stage sibling's with the compressed-KV pool and its row
    count before the PDL switch; the host binds both names."""
    from flashinfer.jit.cake_dsv4 import MODULES, VARIANTS

    variants = VARIANTS[arch]
    four_stage = variants[_KV4_SIBLING]
    assert _L2PF_SIBLING in variants
    l2pf = variants[_L2PF_SIBLING]
    plan = get_cake_dsv4_spec(_L2PF_SIBLING, arch=arch)["arg_plan"]
    four_stage_plan = get_cake_dsv4_spec(_KV4_SIBLING, arch=arch)["arg_plan"]
    if arch != "sm_107a":
        assert l2pf == four_stage == variants[_M64_MULTI_TILE_ROUTE]
        assert plan == four_stage_plan
        return
    assert l2pf != four_stage
    assert l2pf not in (variants[_M64_MULTI_TILE_ROUTE], variants[_BCL_SIBLING])
    assert MODULES[l2pf]["arches"] == ["sm_107a"]
    added = (("buffer", "compressed_KV_cache"), ("parameter", "compressed_KV_rows"))
    at = four_stage_plan.index(("launch", "enable_pdl"))
    assert plan == (*four_stage_plan[:at], *added, *four_stage_plan[at:])
    assert all(cake.is_bindable_arg(kind, name) for kind, name in added)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("launchable", [True, False])
def test_fp8_h64_m64_launch_takes_the_l2pf_sibling(
    monkeypatch, arch, enable_pdl, launchable
):
    """On 216 SMs with the L2 prefetch sibling routed to a program of its own,
    SM107 runs the multi-tile M64 launches of the rule (here items 640 wide)
    on it over the four-stage sibling's grid and partials, with either PDL
    switch; the launches one request above the ceiling, of 3-token requests
    or over compressed pages of 2 rows run the four-stage sibling; where the
    process cannot launch that sibling, and on SM100 and SM103, the route's
    program runs."""
    import flashinfer.jit.cake_dsv4 as jit

    monkeypatch.setattr(jit, "VARIANTS", _l2pf_variants())
    monkeypatch.setattr(cake, "_device_sm_count", lambda device: _SM107_SM_COUNT)
    monkeypatch.setattr(
        cake, "_kv4_sibling_launchable", lambda gate_arch, device: launchable
    )
    four_stage = arch == "sm_107a" and launchable
    for tokens, max_q_len, batch, page, window in (
        (32, 1, 32, 64, True),
        (47, 1, 47, 64, True),
        (48, 1, 48, 64, False),
        (64, 4, 16, 64, True),
        (68, 4, 17, 64, False),
        (60, 3, 20, 64, False),
        (40, 1, 40, 2, False),
        (40, 1, 40, None, False),
    ):
        page_values = {} if page is None else {"compressed_page_size": page}
        L = _RecordingLauncher(
            arch,
            Q=torch.empty(0),
            num_query_tokens=tokens,
            num_heads=64,
            sparse_topk=640,
            max_q_len=max_q_len,
            batch_size=batch,
            enable_pdl=enable_pdl,
            **page_values,
        )
        cake._dispatch_route(_M64_MULTI_TILE_ROUTE, L)
        if four_stage:
            program = _L2PF_SIBLING if window else _KV4_SIBLING
        else:
            program = _M64_MULTI_TILE_ROUTE
        launch = {"grid": (tokens, 1, 1), "total_work_items": tokens}
        assert L.calls == [(program, {**launch, **L.partials(1)})], (tokens, max_q_len)


@pytest.mark.cpu_only
@pytest.mark.parametrize("sm_count", [208, 212, 216])
def test_l2pf_and_bcl_windows_do_not_overlap(monkeypatch, sm_count):
    """No launch is in both siblings' windows: the TMEM-batch sibling never
    takes a launch of the L2 prefetch rule, and the L2 prefetch step keeps
    every TMEM-batch launch."""
    import flashinfer.jit.cake_dsv4 as jit

    monkeypatch.setattr(jit, "VARIANTS", _l2pf_variants())
    monkeypatch.setattr(cake, "_device_sm_count", lambda device: sm_count)
    for (sparse_topk, max_q_len), ceiling in _L2PF_CEILINGS.items():
        for tokens in range(max_q_len, ceiling + 1, max_q_len):
            shape = dict(
                num_heads=64,
                max_q_len=max_q_len,
                num_query_tokens=tokens,
                sparse_topk=sparse_topk,
            )
            bcl = cake._kv4_bcl_variant(
                _KV4_SIBLING,
                arch="sm_107a",
                device=_DEVICE,
                compressed_page_size=64,
                **shape,
            )
            assert bcl == _KV4_SIBLING, shape
    for shape in (
        dict(num_heads=64, max_q_len=4, num_query_tokens=120, sparse_topk=640),
        dict(num_heads=64, max_q_len=2, num_query_tokens=108, sparse_topk=512),
        dict(num_heads=64, max_q_len=1, num_query_tokens=192, sparse_topk=1152),
    ):
        variant = cake._kv4_l2pf_variant(
            _BCL_SIBLING,
            arch="sm_107a",
            device=_DEVICE,
            batch_size=shape["num_query_tokens"] // shape["max_q_len"],
            compressed_page_size=64,
            **shape,
        )
        assert variant == _BCL_SIBLING, shape


@pytest.mark.cpu_only
def test_run_cake_dsv4_launches_the_l2pf_sibling_in_its_window(monkeypatch):
    """run_cake_dsv4 on SM107 with the L2 prefetch sibling routed to a program
    of its own: dense FP8/H64 decode calls over compressed pages of 64 rows
    that the host sends to the four-stage sibling launch the L2 prefetch
    sibling inside the rule on 212 and 216 SMs, one CTA per row, bound to the
    compressed-KV pool and its row count; the first rows past each ceiling,
    the c128 layer (pages of 2 rows) and a 208-SM device launch the
    four-stage sibling."""
    import flashinfer.jit.cake_dsv4 as jit

    monkeypatch.setattr(jit, "VARIANTS", _l2pf_variants())
    for sm_count, width, max_q_len, tokens, page, program in (
        (216, 640, 1, 32, 64, _L2PF_SIBLING),
        (216, 640, 1, 47, 64, _L2PF_SIBLING),
        (212, 640, 1, 40, 64, _L2PF_SIBLING),
        (216, 640, 4, 60, 64, _L2PF_SIBLING),
        (212, 640, 4, 64, 64, _L2PF_SIBLING),
        (216, 512, 1, 41, 64, _L2PF_SIBLING),
        (212, 512, 1, 40, 64, _L2PF_SIBLING),
        (216, 512, 4, 56, 64, _L2PF_SIBLING),
        (212, 512, 4, 44, 64, _L2PF_SIBLING),
        (216, 640, 1, 58, 64, _KV4_SIBLING),
        (212, 640, 1, 57, 64, _KV4_SIBLING),
        (216, 640, 4, 68, 64, _KV4_SIBLING),
        (216, 512, 1, 42, 64, _KV4_SIBLING),
        (216, 512, 4, 60, 64, _KV4_SIBLING),
        (216, 640, 1, 40, 2, _KV4_SIBLING),
        (208, 640, 1, 40, 64, _KV4_SIBLING),
        (208, 512, 4, 56, 64, _KV4_SIBLING),
    ):
        shape = (sm_count, width, max_q_len, tokens, page)
        programs, calls, _ = _run_fake_h64_decode_fp8(
            monkeypatch,
            arch="sm_107a",
            width=width,
            tokens=tokens,
            max_q_len=max_q_len,
            sm_count=sm_count,
            page=page,
        )
        (bound,) = calls
        assert programs == [program], shape
        assert (bound["grid_x"], bound["grid_y"], bound["grid_z"]) == (tokens, 1, 1)
        assert bound["total_work_items"] == bound["num_query_tokens"] == tokens
        assert bound["max_q_len"] == max_q_len, shape
        if program == _L2PF_SIBLING:
            pool = bound["compressed_KV_cache"]
            assert bound["compressed_KV_rows"] == pool.shape[0] == 4 * page, shape
            assert pool.dtype == torch.uint8 and pool.shape[1] == 512, shape


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "sm_count,width,tokens,chunk,program",
    [
        (216, 640, 96, 32, _L2PF_SIBLING),
        (212, 512, 120, 40, _L2PF_SIBLING),
        (216, 1152, 384, 192, _BCL_SIBLING),
        (212, 1152, 384, 192, _BCL_SIBLING),
        (208, 640, 96, 32, _KV4_SIBLING),
        (208, 1152, 384, 192, _KV4_SIBLING),
    ],
)
def test_row_tiled_one_token_chunks_take_the_rule_siblings(
    monkeypatch, sm_count, width, tokens, chunk, program
):
    """A workspace of ``chunk`` rows tiles a dense FP8/H64 one-token decode
    call outside both one-token rules (items 640 and 512 wide past the L2
    prefetch ceilings, items 1152 wide past the decode window) into chunks
    inside one: on 212 and 216 SMs every chunk launches the rule's sibling, on
    208 SMs the four-stage sibling, one CTA per row, each bound to one-token
    requests and the call's batch size (chunks after the first are ragged
    launches over all the call's requests)."""
    import flashinfer.jit.cake_dsv4 as jit

    monkeypatch.setattr(jit, "VARIANTS", _l2pf_variants())
    plan = cake._route_plan(
        _M64_MULTI_TILE_ROUTE, num_query_tokens=tokens, num_heads=64, sparse_topk=width
    )
    chunk_bytes = cake._launch_workspace_bytes(plan, chunk, 64, tokens + 1)
    rows = cake._rows_per_launch(
        plan,
        num_query_tokens=tokens,
        num_heads=64,
        batch_size=tokens,
        workspace_bytes=chunk_bytes,
    )
    assert rows == chunk
    programs, calls, _ = _run_fake_h64_decode_fp8(
        monkeypatch,
        arch="sm_107a",
        width=width,
        tokens=tokens,
        sm_count=sm_count,
        workspace=_aligned_u8(chunk_bytes),
    )
    shape = (sm_count, width, tokens, chunk)
    assert programs == [program] * (tokens // chunk), shape
    for bound in calls:
        assert (bound["grid_x"], bound["grid_y"], bound["grid_z"]) == (chunk, 1, 1)
        assert bound["total_work_items"] == bound["num_query_tokens"] == chunk
        assert bound["max_q_len"] == 1, shape
        assert bound["batch_size"] == tokens, shape


# --------------------------------------------------------------------------- #
# V-block sibling: SM107's low-head prefill over four items per row           #
# --------------------------------------------------------------------------- #

_VBLOCK_TOKENS = (1, 8, 12, 24, 25, 26, 27, 28, 48, 64, 65)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
@pytest.mark.parametrize(
    "sm_count,ceiling", [(216, 27), (212, 26), (208, 26), (264, 27)]
)
def test_sm107a_lowhead_launches_take_the_vblock_sibling_up_to_its_ceiling(
    monkeypatch, arch, sm_count, ceiling
):
    """SM107 routes the low-head prefill route's V-block sibling to a program
    of its own, with the route's arguments, and runs it on launches of at most
    27 rows on 216 SMs, a ceiling scaled down with a smaller SM count and kept
    on a larger one; SM100 and SM103 route the sibling to the route's program
    and launch the route."""
    from flashinfer.jit.cake_dsv4 import VARIANTS

    monkeypatch.setattr(cake, "_device_sm_count", lambda device: sm_count)
    assert {_LOWHEAD_PREFILL_ROUTE} == cake._VBLOCK_SIBLING_ROUTES
    assert cake._VBLOCK_SIBLING_MAX_TOKENS == {"sm_107a": 27}
    assert cake._VBLOCK_SIBLING_MEASURED_SMS == {"sm_107a": 216}
    assert cake._VBLOCK_ITEMS_PER_TOKEN == 4
    route, sibling = _LOWHEAD_PREFILL_ROUTE, _VBLOCK_SIBLING
    own = arch == "sm_107a"
    programs = VARIANTS[arch]
    assert (programs[sibling] != programs[route]) == own
    assert (programs[sibling] != programs[_OVERLAP_SIBLING]) == own
    assert (
        get_cake_dsv4_spec(sibling, arch=arch)["arg_plan"]
        == get_cake_dsv4_spec(route, arch=arch)["arg_plan"]
    )
    for tokens in _VBLOCK_TOKENS:
        expected = sibling if own and tokens <= ceiling else route
        variant = cake._vblock_variant(
            route, arch=arch, num_query_tokens=tokens, device=_DEVICE
        )
        assert variant == expected, tokens
    for other in (
        _OVERLAP_SIBLING,
        "fp8_lowhead_one_partition",
        "fp8_lowhead_h64",
        "fp8_h64_prefill_source_persistent_m64",
    ):
        variant = cake._vblock_variant(
            other, arch=arch, num_query_tokens=12, device=_DEVICE
        )
        assert variant == other


@pytest.mark.cpu_only
def test_vblock_variant_needs_a_sibling_program_of_its_own(monkeypatch):
    """A launch takes the V-block sibling only where its architecture routes
    the sibling to a program other than the route's and the early-PV
    sibling's: a table without the sibling, one that routes it to either of
    those programs and one of another architecture run the route; an
    architecture without a ceiling never reads the device's SM count."""
    import flashinfer.jit.cake_dsv4 as jit

    monkeypatch.setattr(cake, "_device_sm_count", lambda device: 216)
    route, overlap, sibling = (
        _LOWHEAD_PREFILL_ROUTE,
        _OVERLAP_SIBLING,
        _VBLOCK_SIBLING,
    )
    low, early, block = "cake_dsv4_low_head", "cake_dsv4_early_pv", "cake_dsv4_v_block"
    for table, expected in (
        ({"sm_107a": {route: low, overlap: early, sibling: block}}, sibling),
        ({"sm_107a": {route: low, overlap: early}}, route),
        ({"sm_107a": {route: low, overlap: early, sibling: low}}, route),
        ({"sm_107a": {route: low, overlap: early, sibling: early}}, route),
        ({"sm_103a": {route: low, overlap: early, sibling: block}}, route),
    ):
        monkeypatch.setattr(jit, "VARIANTS", table)
        variant = cake._vblock_variant(
            route, arch="sm_107a", num_query_tokens=12, device=_DEVICE
        )
        assert variant == expected

    def no_sm_count(device):
        raise AssertionError("read the SM count")

    monkeypatch.setattr(cake, "_device_sm_count", no_sm_count)
    for arch in ("sm_100a", "sm_103a"):
        variant = cake._vblock_variant(
            route, arch=arch, num_query_tokens=12, device=_DEVICE
        )
        assert variant == route


@pytest.mark.cpu_only
def test_sm107a_vblock_sibling_is_a_program_of_its_own():
    """The V-block sibling's sm_107a program differs from the route's and the
    early-PV sibling's and launches with the same dynamic shared memory."""
    import re

    from flashinfer.jit.cake_dsv4 import VARIANTS, _get_csrc_dir

    programs = VARIANTS["sm_107a"]
    route, overlap, sibling = (
        programs[_LOWHEAD_PREFILL_ROUTE],
        programs[_OVERLAP_SIBLING],
        programs[_VBLOCK_SIBLING],
    )
    assert len({route, overlap, sibling}) == 3
    smem = {
        program: re.findall(
            r"config\.dynamicSmemBytes = (\d+);",
            (_get_csrc_dir() / f"sm_107a/{program}_binding.cu").read_text(
                encoding="utf-8"
            ),
        )
        for program in (route, overlap, sibling)
    }
    assert smem[sibling] == smem[overlap] == smem[route] == ["205952"]
