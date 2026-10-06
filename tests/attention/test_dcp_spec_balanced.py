"""Route mirror, JIT keys and GPU parity of the on-device load-balanced DCP routes."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import flashinfer.cake_dcp as cake_dcp
from flashinfer.cake_dcp import (
    DCP_BALANCED_BF16_LONG_TILE_BLOCKS,
    DCP_BALANCED_BF16_MAX_Q_LEN,
    DCP_BALANCED_BF16_MIN_ITEMS,
    DCP_BALANCED_BF16_MIN_Q_LEN,
    DCP_BALANCED_CHUNK_TOKENS,
    DCP_BALANCED_D256_LONG_TILE_BLOCKS,
    DCP_BALANCED_D256_MAX_Q_LEN,
    DCP_BALANCED_D256_MIN_ITEMS,
    DCP_BALANCED_D256_MIN_Q_LEN,
    DCP_BALANCED_D256_ONE_WAVE_LONG_TILE_BLOCKS,
    DCP_BALANCED_D256_ONE_WAVE_LONG_TILE_BLOCKS_BY_ARCH,
    DCP_BALANCED_D256_ONE_WAVE_MIN_Q_LEN,
    DCP_BALANCED_FP8_LONG_TILE_BLOCKS,
    DCP_BALANCED_FP8_LONG_TILE_BLOCKS_BY_ARCH,
    DCP_BALANCED_FP8_MAX_Q_LEN,
    DCP_BALANCED_FP8_MIN_ITEMS,
    DCP_BALANCED_FP8_MIN_Q_LEN,
    DCP_BALANCED_FP8_TWO_WAVE_MIN_ITEMS,
    DCP_BALANCED_KINDS,
    DCP_BALANCED_MAX_REQUESTS,
    dcp_balanced_band,
    dcp_balanced_n_rows,
    dcp_balanced_program,
    dcp_balanced_route,
    dcp_static_shape,
    get_dcp_spec_balanced_counter_bytes,
    get_dcp_spec_balanced_workspace_bytes,
    run_dcp_spec_decode,
)
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from flashinfer.jit.cake_dcp import (
    DCP_BALANCED_FAMILIES,
    DCP_BALANCED_N_ROWS,
    dcp_balanced_program_variants,
    get_dcp_spec_balanced_uri,
    get_dcp_spec_registry,
)
from flashinfer.jit.cake_fmha import CAKE_FMHA_JIT_TAG
from flashinfer.utils import get_compute_capability, is_sm100a_supported

_LOG2_E = math.log2(math.e)
# Multiprocessor counts of the two measured architectures.
_SM_COUNT = {"sm_100a": 148, "sm_103a": 152}
_KIND_HEADS = {"bf16_p16": (64, 8), "fp8_p64": (64, 8), "fp8_p64_d256": (16, 1)}
_KIND_HEAD_DIM = {"bf16_p16": 128, "fp8_p64": 128, "fp8_p64_d256": 256}
_KIND_PAGE_SIZE = {"bf16_p16": 16, "fp8_p64": 64, "fp8_p64_d256": 64}
_KIND_KV_DTYPE = {
    "bf16_p16": torch.bfloat16,
    "fp8_p64": torch.float8_e4m3fn,
    "fp8_p64_d256": torch.float8_e4m3fn,
}
_KIND_TOLERANCE = {
    "bf16_p16": (1e-2, 1e-2),
    "fp8_p64": (0.1, 0.1),
    "fp8_p64_d256": (0.1, 0.1),
}
_KIND_FAMILY = {
    "bf16_p16": "dcp_spec_bf16_balanced",
    "fp8_p64": "dcp_spec_bf16_fp8_balanced",
    "fp8_p64_d256": "dcp_spec_bf16_fp8_d256_balanced",
}
# The FlashInfer bench's ragged prefix pattern (flashinfer-ai/flashinfer#4832).
AGENTX = [
    8193, 57345, 73729, 81921, 98305, 106497, 114689, 131073,
    139265, 147457, 163841, 180225, 196609, 212993, 229377, 237569,
]  # fmt: skip


def _local_len(prefix: int, q_len: int, cp_world: int, cp_rank: int) -> int:
    """Rank-local keys visible to the last speculative row (the planner length)."""

    last = prefix + q_len - 1 - cp_rank
    return 0 if last < 0 else last // cp_world + 1


def _max_local(prefixes, q_len, cp_world, cp_rank) -> int:
    return max(_local_len(int(p), q_len, cp_world, cp_rank) for p in prefixes)


def _band(kind, *, batch, q_len, prefix, cp_world, cp_rank, arch, heads=None):
    num_q_heads, num_kv_heads = heads or _KIND_HEADS[kind]
    prefixes = prefix if isinstance(prefix, (list, tuple)) else [prefix]
    return dcp_balanced_band(
        kind,
        batch_size=batch,
        q_len=q_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=_KIND_HEAD_DIM[kind],
        max_local_seq_len=_max_local(prefixes, q_len, cp_world, cp_rank),
        cp_world=cp_world,
        sm_count=_SM_COUNT.get(arch, 148),
        arch=arch,
    )


def test_balanced_band_constants_match_the_cake_dispatcher() -> None:
    assert DCP_BALANCED_KINDS == ("bf16_p16", "fp8_p64", "fp8_p64_d256")
    assert DCP_BALANCED_MAX_REQUESTS == 1024
    assert DCP_BALANCED_CHUNK_TOKENS == 256
    assert (DCP_BALANCED_BF16_MIN_Q_LEN, DCP_BALANCED_BF16_MAX_Q_LEN) == (3, 8)
    assert DCP_BALANCED_BF16_MIN_ITEMS == 128
    assert DCP_BALANCED_BF16_LONG_TILE_BLOCKS == 16
    assert (DCP_BALANCED_FP8_MIN_Q_LEN, DCP_BALANCED_FP8_MAX_Q_LEN) == (3, 8)
    assert DCP_BALANCED_FP8_MIN_ITEMS == 160
    # round 5: 24 -> 17 on sm_103a; the 17-block class is a tie band on sm_100a (floor 18)
    assert DCP_BALANCED_FP8_LONG_TILE_BLOCKS == 18
    assert DCP_BALANCED_FP8_LONG_TILE_BLOCKS_BY_ARCH == {"sm_100a": 18, "sm_103a": 17}
    assert DCP_BALANCED_FP8_TWO_WAVE_MIN_ITEMS == {"sm_100a": 160, "sm_103a": 160}
    assert (DCP_BALANCED_D256_MIN_Q_LEN, DCP_BALANCED_D256_MAX_Q_LEN) == (1, 8)
    assert DCP_BALANCED_D256_MIN_ITEMS == 160
    assert DCP_BALANCED_D256_LONG_TILE_BLOCKS == 96
    assert DCP_BALANCED_D256_ONE_WAVE_MIN_Q_LEN == 3
    assert DCP_BALANCED_D256_ONE_WAVE_LONG_TILE_BLOCKS == 22
    assert DCP_BALANCED_D256_ONE_WAVE_LONG_TILE_BLOCKS_BY_ARCH == {
        "sm_100a": 22,
        "sm_103a": 22,
    }


# (label, kind, batch, q_len, prefix(es), cp_world, cp_rank, expected route)
# The expectation is the measured winner of the Cake round-3 band probes and
# standings (design document, units 2-4); the band routes every one of them
# to its winner on both architectures unless a per-arch dict says otherwise.
_ROUTE_ROWS = [
    # bf16 / page 16: the 19-row band probe, the three crossover rows and the perf rows
    ("band_b1_s4096_q8_w4_r0", "bf16_p16", 1, 8, 4096, 4, 0, "static"),
    ("band_b2_s4096_q4_w4_r0", "bf16_p16", 2, 4, 4096, 4, 0, "static"),
    ("band_b4_s4096_q4_w4_r0", "bf16_p16", 4, 4, 4096, 4, 0, "static"),
    ("band_b1_s8192_q4_w4_r0", "bf16_p16", 1, 4, 8192, 4, 0, "static"),
    ("band_b1_s8192_q8_w4_r0", "bf16_p16", 1, 8, 8192, 4, 0, "static"),
    ("band_b2_s16384_q8_w4_r0", "bf16_p16", 2, 8, 16384, 4, 0, "balanced"),
    ("band_b1_s32768_q4_w4_r0", "bf16_p16", 1, 4, 32768, 4, 0, "balanced"),
    ("band_b1_s65536_q4_w4_r0", "bf16_p16", 1, 4, 65536, 4, 0, "balanced"),
    ("band_b8_s512_q4_w4_r0", "bf16_p16", 8, 4, 512, 4, 0, "static"),
    ("band_b8_s1024_q4_w4_r0", "bf16_p16", 8, 4, 1024, 4, 0, "balanced"),
    ("band_b64_s1024_q4_w4_r0", "bf16_p16", 64, 4, 1024, 4, 0, "balanced"),
    ("band_b8_s4096_q3_w4_r0", "bf16_p16", 8, 3, 4096, 4, 0, "balanced"),
    ("band_b8_s4096_q6_w4_r0", "bf16_p16", 8, 6, 4096, 4, 0, "balanced"),
    ("band_b8_s4096_q4_w8_r5", "bf16_p16", 8, 4, 4096, 8, 5, "balanced"),
    ("band_b8_s4096_q4_w2_r1", "bf16_p16", 8, 4, 4096, 2, 1, "balanced"),
    ("band_b8_s4096_q2_w4_r0", "bf16_p16", 8, 2, 4096, 4, 0, "static"),
    ("band_b8_s4096_q1_w4_r0", "bf16_p16", 8, 1, 4096, 4, 0, "static"),
    ("band_b256_s4096_q4_w4_r0", "bf16_p16", 256, 4, 4096, 4, 0, "balanced"),
    ("band_b128_s16384_q8_w4_r0", "bf16_p16", 128, 8, 16384, 4, 0, "balanced"),
    ("band_b1_s6144_q4_w4_r0", "bf16_p16", 1, 4, 6144, 4, 0, "static"),
    ("band_b1_s7168_q4_w4_r0", "bf16_p16", 1, 4, 7168, 4, 0, "static"),
    ("band_b2_s6144_q8_w4_r0", "bf16_p16", 2, 8, 6144, 4, 0, "static"),
    ("perf_b1_s4096_q4_w4_r0", "bf16_p16", 1, 4, 4096, 4, 0, "static"),
    ("perf_b8_s4096_q4_w4_r0", "bf16_p16", 8, 4, 4096, 4, 0, "balanced"),
    ("perf_b64_s4096_q4_w4_r0", "bf16_p16", 64, 4, 4096, 4, 0, "balanced"),
    ("perf_b1_s16384_q8_w4_r0", "bf16_p16", 1, 8, 16384, 4, 0, "balanced"),
    ("perf_b8_s16384_q8_w4_r0", "bf16_p16", 8, 8, 16384, 4, 0, "balanced"),
    ("perf_b64_s16384_q8_w4_r0", "bf16_p16", 64, 8, 16384, 4, 0, "balanced"),
    ("perf_b8_s16383_q8_w4_r3_tail", "bf16_p16", 8, 8, 16383, 4, 3, "balanced"),
    ("dcp_bf16_agentx_b16_q4_cp4_r0", "bf16_p16", 16, 4, AGENTX, 4, 0, "balanced"),
    ("dcp_bf16_agentx_b16_q8_cp4_r3", "bf16_p16", 16, 8, AGENTX, 4, 3, "balanced"),
    # fp8 e4m3 / page 64 / D128: the 27-row band probe and production rows
    ("bandfp8_b1_s4096_q8_w4_r0", "fp8_p64", 1, 8, 4096, 4, 0, "static"),
    ("bandfp8_b2_s4096_q4_w4_r0", "fp8_p64", 2, 4, 4096, 4, 0, "static"),
    ("bandfp8_b4_s4096_q4_w4_r0", "fp8_p64", 4, 4, 4096, 4, 0, "static"),
    ("bandfp8_b1_s8192_q4_w4_r0", "fp8_p64", 1, 4, 8192, 4, 0, "static"),
    ("bandfp8_b1_s8192_q8_w4_r0", "fp8_p64", 1, 8, 8192, 4, 0, "static"),
    ("bandfp8_b2_s16384_q8_w4_r0", "fp8_p64", 2, 8, 16384, 4, 0, "balanced"),
    (
        "bandfp8_b1_s32768_q4_w4_r0",
        "fp8_p64",
        1,
        4,
        32768,
        4,
        0,
        "balanced",
    ),  # round 5: 22 blocks per CTA, 1.15 / 1.11
    ("bandfp8_b1_s65536_q4_w4_r0", "fp8_p64", 1, 4, 65536, 4, 0, "balanced"),
    ("bandfp8_b8_s512_q4_w4_r0", "fp8_p64", 8, 4, 512, 4, 0, "static"),
    ("bandfp8_b8_s1024_q4_w4_r0", "fp8_p64", 8, 4, 1024, 4, 0, "static"),
    ("bandfp8_b64_s1024_q4_w4_r0", "fp8_p64", 64, 4, 1024, 4, 0, "balanced"),
    (
        "bandfp8_b8_s4096_q3_w4_r0",
        "fp8_p64",
        8,
        3,
        4096,
        4,
        0,
        "balanced",
    ),  # round 5: both parts (320 items, two waves)
    ("bandfp8_b8_s4096_q6_w4_r0", "fp8_p64", 8, 6, 4096, 4, 0, "balanced"),
    ("bandfp8_b8_s4096_q4_w8_r5", "fp8_p64", 8, 4, 4096, 8, 5, "static"),
    ("bandfp8_b8_s4096_q4_w2_r1", "fp8_p64", 8, 4, 4096, 2, 1, "balanced"),
    ("bandfp8_b8_s4096_q2_w4_r0", "fp8_p64", 8, 2, 4096, 4, 0, "static"),
    ("bandfp8_b8_s4096_q1_w4_r0", "fp8_p64", 8, 1, 4096, 4, 0, "static"),
    ("bandfp8_b256_s4096_q4_w4_r0", "fp8_p64", 256, 4, 4096, 4, 0, "balanced"),
    ("bandfp8_b1_s6144_q4_w4_r0", "fp8_p64", 1, 4, 6144, 4, 0, "static"),
    ("bandfp8_b1_s7168_q4_w4_r0", "fp8_p64", 1, 4, 7168, 4, 0, "static"),
    ("bandfp8_b1_s16384_q4_w4_r0", "fp8_p64", 1, 4, 16384, 4, 0, "static"),
    (
        "bandfp8_b1_s24576_q4_w4_r0",
        "fp8_p64",
        1,
        4,
        24576,
        4,
        0,
        {"sm_100a": "static", "sm_103a": "balanced"},
    ),  # round 5: 17 blocks per CTA (1.055 GB300, tie 0.981 B200)
    ("bandfp8_b2_s6144_q8_w4_r0", "fp8_p64", 2, 8, 6144, 4, 0, "static"),
    ("bandfp8_b128_s16384_q8_w4_r0", "fp8_p64", 128, 8, 16384, 4, 0, "balanced"),
    (
        "bandfp8_b1_s8192_q4_w1_r0",
        "fp8_p64",
        1,
        4,
        8192,
        1,
        0,
        {"sm_100a": "static", "sm_103a": "balanced"},
    ),  # round 5: cp1, 17 blocks per CTA (1.053 GB300, tie 0.994 B200)
    ("bandfp8_b1_s4096_q4_w1_r0", "fp8_p64", 1, 4, 4096, 1, 0, "static"),
    ("bandfp8_b8_s8192_q4_w1_r0", "fp8_p64", 8, 4, 8192, 1, 0, "balanced"),
    (
        "prod_b8_s4096_q4_cp4",
        "fp8_p64",
        8,
        4,
        4096,
        4,
        0,
        "balanced",
    ),  # round 5: 1.29 GB300 / 1.17 B200
    ("prod_b1_s8192_q4_cp4_graph", "fp8_p64", 1, 4, 8192, 4, 0, "static"),
    ("prod_b8_s8192_q4_cp4_graph", "fp8_p64", 8, 4, 8192, 4, 0, "balanced"),
    ("prod_b32_s8192_q4_cp4_graph", "fp8_p64", 32, 4, 8192, 4, 0, "balanced"),
    ("prod_b256_s8192_q4_cp4_graph", "fp8_p64", 256, 4, 8192, 4, 0, "balanced"),
    ("prod_b8_s4096_q8_cp4", "fp8_p64", 8, 8, 4096, 4, 0, "balanced"),
    ("prod_b64_s16384_q8_cp4", "fp8_p64", 64, 8, 16384, 4, 0, "balanced"),
    ("prod_b64_s8191_q4_cp4_residue", "fp8_p64", 64, 4, 8191, 4, 0, "balanced"),
    ("prod_b64_s8192_q4_cp2", "fp8_p64", 64, 4, 8192, 2, 0, "balanced"),
    ("prod_b64_s8192_q4_cp8", "fp8_p64", 64, 4, 8192, 8, 0, "balanced"),
    (
        "cp1_peer_b1_s8192_q4",
        "fp8_p64",
        1,
        4,
        8192,
        1,
        0,
        {"sm_100a": "static", "sm_103a": "balanced"},
    ),  # round 5: 17 blocks per CTA (1.053 GB300; B200 tie 0.994 probe / 1.03 bench)
    ("cp1_peer_b8_s8192_q4", "fp8_p64", 8, 4, 8192, 1, 0, "balanced"),
    ("stretch_b384_s8192_q4_cp4", "fp8_p64", 384, 4, 8192, 4, 0, "balanced"),
    ("dcp_fp8_agentx_b16_q4_cp4_r0", "fp8_p64", 16, 4, AGENTX, 4, 0, "balanced"),
    # fp8 e4m3 / page 64 / D256 GQA-16: the 15 production rows and the ragged rows
    ("prod_d256_b1_ctx32768_q4_cp4_graph", "fp8_p64_d256", 1, 4, 32764, 4, 0, "static"),
    ("prod_d256_b8_ctx32768_q4_cp4_graph", "fp8_p64_d256", 8, 4, 32764, 4, 0, "static"),
    (
        "prod_d256_b16_ctx32768_q4_cp4_graph",
        "fp8_p64_d256",
        16,
        4,
        32764,
        4,
        0,
        "balanced",
    ),  # round 5: the one-wave row-tile regime
    (
        "prod_d256_b32_ctx32768_q4_cp4_graph",
        "fp8_p64_d256",
        32,
        4,
        32764,
        4,
        0,
        "balanced",
    ),  # round 5: the one-wave row-tile regime
    (
        "prod_d256_b64_ctx32768_q4_cp4_graph",
        "fp8_p64_d256",
        64,
        4,
        32764,
        4,
        0,
        "balanced",
    ),
    (
        "prod_d256_b128_ctx32768_q4_cp4_graph",
        "fp8_p64_d256",
        128,
        4,
        32764,
        4,
        0,
        "balanced",
    ),
    (
        "prod_d256_b192_ctx32768_q4_cp4_graph",
        "fp8_p64_d256",
        192,
        4,
        32764,
        4,
        0,
        "balanced",
    ),
    (
        "prod_d256_b256_ctx32768_q4_cp4_graph",
        "fp8_p64_d256",
        256,
        4,
        32764,
        4,
        0,
        "balanced",
    ),
    (
        "prod_d256_b128_ctx32768_q1_cp4_graph",
        "fp8_p64_d256",
        128,
        1,
        32767,
        4,
        0,
        "static",
    ),
    (
        "prod_d256_b128_ctx32768_q2_cp4_graph",
        "fp8_p64_d256",
        128,
        2,
        32766,
        4,
        0,
        "balanced",
    ),
    (
        "prod_d256_b128_ctx32768_q3_cp4_graph",
        "fp8_p64_d256",
        128,
        3,
        32765,
        4,
        0,
        "balanced",
    ),
    (
        "prod_d256_b128_ctx32768_q5_cp4_graph",
        "fp8_p64_d256",
        128,
        5,
        32763,
        4,
        0,
        "balanced",
    ),
    (
        "prod_d256_b128_ctx32768_q6_cp4_graph",
        "fp8_p64_d256",
        128,
        6,
        32762,
        4,
        0,
        "balanced",
    ),
    (
        "prod_d256_b128_ctx32768_q7_cp4_graph",
        "fp8_p64_d256",
        128,
        7,
        32761,
        4,
        0,
        "balanced",
    ),
    (
        "prod_d256_b128_ctx32768_q8_cp4_graph",
        "fp8_p64_d256",
        128,
        8,
        32760,
        4,
        0,
        "balanced",
    ),
    ("dcp_d256_agentx_b16_q4_cp4_r0", "fp8_p64_d256", 16, 4, AGENTX, 4, 0, "balanced"),
    ("dcp_d256_agentx_b64_q4_cp4_r0", "fp8_p64_d256", 64, 4, AGENTX, 4, 0, "balanced"),
]


@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a"))
@pytest.mark.parametrize("row", _ROUTE_ROWS, ids=[row[0] for row in _ROUTE_ROWS])
def test_balanced_band_routes_every_measured_row_to_its_winner(row, arch) -> None:
    _label, kind, batch, q_len, prefix, cp_world, cp_rank, expected = row
    if isinstance(expected, dict):
        expected = expected[arch]
    band = _band(
        kind,
        batch=batch,
        q_len=q_len,
        prefix=prefix,
        cp_world=cp_world,
        cp_rank=cp_rank,
        arch=arch,
    )
    assert band.route == expected, band
    assert (
        dcp_balanced_route(
            kind,
            batch_size=batch,
            q_len=q_len,
            num_q_heads=_KIND_HEADS[kind][0],
            num_kv_heads=_KIND_HEADS[kind][1],
            head_dim=_KIND_HEAD_DIM[kind],
            max_local_seq_len=_max_local(
                prefix if isinstance(prefix, list) else [prefix],
                q_len,
                cp_world,
                cp_rank,
            ),
            cp_world=cp_world,
            sm_count=_SM_COUNT[arch],
            arch=arch,
        )
        == expected
    )


@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a"))
def test_d256_one_wave_row_tile_regime(arch) -> None:
    # One static wave of the prod_d256 ctx32768 geometry (8192 local tokens, 64 blocks per
    # tile): the static route streams each request's KV once per speculative row, the
    # balanced row tile of up to four rows once per tile -- rows at q_len >= 3 whose static
    # tile streams >= 22 blocks per CTA route balanced (b12 / b16 / b32 at q_len 4, b16 at
    # q_len 3, b16 at q_len 5 and 8, b8 at q_len 8); the b8 tie band (16 blocks) and b1 stay
    # static, q_len 1 (no sharing) and q_len 2 (two rows per tile, <= 6 %) stay static.
    def band(batch, q_len):
        return _band(
            "fp8_p64_d256",
            batch=batch,
            q_len=q_len,
            prefix=32764,
            cp_world=4,
            cp_rank=0,
            arch=arch,
        )

    b12, b16, b32 = band(12, 4), band(16, 4), band(32, 4)
    assert (b12.waves, b12.blocks_per_cta, b12.route, b12.reason) == (
        1,
        22,
        "balanced",
        "one_wave_row_tiles",
    )  # 48 tiles, split 3: 1.108 GB300 / 1.142 B200
    assert (b16.waves, b16.blocks_per_cta, b16.route, b16.reason) == (
        1,
        32,
        "balanced",
        "one_wave_row_tiles",
    )
    assert (b32.waves, b32.blocks_per_cta, b32.route, b32.reason) == (
        1,
        64,
        "balanced",
        "one_wave_row_tiles",
    )
    assert (band(8, 4).blocks_per_cta, band(8, 4).route, band(8, 4).reason) == (
        16,
        "static",
        "one_wave",
    )  # tie band 1.005 / 1.041
    assert (band(1, 4).route, band(1, 4).reason) == ("static", "one_wave")
    assert (band(16, 3).blocks_per_cta, band(16, 3).route, band(16, 3).reason) == (
        22,
        "balanced",
        "one_wave_row_tiles",
    )  # 1.071 / 1.064
    # q_len 5 / 8: the last speculative row puts 8193 keys on rank 0 (65 blocks; split 1 / split 2)
    assert (band(16, 5).blocks_per_cta, band(16, 5).route) == (65, "balanced")
    assert (band(8, 8).blocks_per_cta, band(8, 8).route) == (33, "balanced")
    assert (band(32, 1).waves, band(32, 1).route) == (
        1,
        "static",
    )  # q_len 1: no row-tile sharing (0.888 / 0.897)
    assert (band(64, 2).blocks_per_cta, band(64, 2).route) == (
        64,
        "static",
    )  # q_len 2: two rows per tile, 1.051 / 1.055 -- recorded, kept static
    assert (band(16, 2).blocks_per_cta, band(16, 2).route) == (
        16,
        "static",
    )  # 0.937 / 0.911
    assert (band(16, 8).waves, band(16, 8).route, band(16, 8).reason) == (
        1,
        "balanced",
        "one_wave_row_tiles",
    )


def test_two_wave_floor_is_keyed_on_the_architecture() -> None:
    # prod_b8_s4096_q4_cp4: 320 chunk-pair items at exactly two static waves.  Round 3 kept
    # sm_100a static here (floor 384); the round-5 programs win the row on both parts, so both
    # per-architecture floors sit at the family's items floor and the row routes balanced.
    b200 = _band(
        "fp8_p64", batch=8, q_len=4, prefix=4096, cp_world=4, cp_rank=0, arch="sm_100a"
    )
    gb300 = _band(
        "fp8_p64", batch=8, q_len=4, prefix=4096, cp_world=4, cp_rank=0, arch="sm_103a"
    )
    assert (b200.num_split, b200.waves, b200.blocks_per_cta, b200.items) == (
        1,
        2,
        9,
        320,
    )
    assert (gb300.num_split, gb300.waves, gb300.blocks_per_cta, gb300.items) == (
        1,
        2,
        9,
        320,
    )
    assert (b200.route, b200.reason) == ("balanced", "two_waves")
    assert (gb300.route, gb300.reason) == ("balanced", "two_waves")
    # the per-architecture floor still gates a two-wave row below it (the dictionary is the contract)
    assert DCP_BALANCED_FP8_TWO_WAVE_MIN_ITEMS["sm_100a"] == DCP_BALANCED_FP8_MIN_ITEMS
    # 576 items at two waves clear the sm_100a floor; three waves need no floor.
    assert (
        _band(
            "fp8_p64",
            batch=8,
            q_len=4,
            prefix=8192,
            cp_world=4,
            cp_rank=0,
            arch="sm_100a",
        ).route
        == "balanced"
    )
    assert (
        _band(
            "fp8_p64",
            batch=8,
            q_len=6,
            prefix=4096,
            cp_world=4,
            cp_rank=0,
            arch="sm_100a",
        ).reason
        == "waves"
    )
    # An unmeasured architecture key takes the family's items floor.
    assert (
        _band(
            "fp8_p64",
            batch=8,
            q_len=4,
            prefix=4096,
            cp_world=4,
            cp_rank=0,
            arch="sm100f",
        ).route
        == "balanced"
    )
    # The floor is FP8-D128 only: the same geometry on bf16 needs 128 items.
    assert (
        _band(
            "bf16_p16",
            batch=8,
            q_len=4,
            prefix=4096,
            cp_world=4,
            cp_rank=0,
            arch="sm_100a",
        ).route
        == "balanced"
    )


def test_balanced_band_contract_gates_come_before_the_regimes() -> None:
    long_row = dict(
        batch=64, q_len=8, prefix=16384, cp_world=4, cp_rank=0, arch="sm_100a"
    )
    assert _band("bf16_p16", **long_row).route == "balanced"
    assert _band("bf16_p16", heads=(8, 8), **long_row).reason == "group"  # MHA
    assert _band("bf16_p16", heads=(32, 8), **long_row).reason == "group"  # GQA-4
    assert (
        _band("fp8_p64_d256", heads=(8, 1), **{**long_row, "batch": 128}).reason
        == "group"
    )
    assert _band("bf16_p16", **{**long_row, "batch": 1025}).reason == "batch"
    assert _band("bf16_p16", **{**long_row, "q_len": 2}).reason == "q_len"
    assert _band("fp8_p64", **{**long_row, "q_len": 1}).reason == "q_len"
    assert (
        _band(
            "fp8_p64_d256",
            batch=128,
            q_len=1,
            prefix=32767,
            cp_world=4,
            cp_rank=0,
            arch="sm_100a",
        ).reason
        == "one_wave"
    )
    assert (
        _band(
            "bf16_p16",
            batch=8,
            q_len=4,
            prefix=512,
            cp_world=4,
            cp_rank=0,
            arch="sm_100a",
        ).reason
        == "items"
    )
    with pytest.raises(ValueError, match="kind"):
        dcp_balanced_band(
            "fp16_p16", batch_size=1, q_len=4, num_q_heads=64, num_kv_heads=8, head_dim=128,
            max_local_seq_len=1, cp_world=4, sm_count=148, arch="sm_100a",
        )  # fmt: skip


@pytest.mark.parametrize(
    ("kind", "batch", "q_len", "prefix", "cp_world", "sm_count", "expected", "items"),
    [
        # (num_split, waves, blocks_per_cta) of the static route and the planner's items bound,
        # from the round-3 fp8 band-fit table (GB300 152 SMs; B200 148 SMs).
        ("fp8_p64", 1, 8, 4096, 4, 152, (2, 1, 5), 40),
        ("fp8_p64", 1, 4, 8192, 4, 152, (3, 1, 6), 72),
        ("fp8_p64", 2, 8, 16384, 4, 152, (1, 1, 33), 272),
        ("fp8_p64", 1, 4, 32768, 4, 152, (3, 1, 22), 264),
        ("fp8_p64", 8, 4, 512, 4, 152, (1, 2, 2), 64),
        ("fp8_p64", 64, 4, 1024, 4, 148, (1, 14, 3), 1024),
        ("fp8_p64", 256, 4, 4096, 4, 152, (1, 54, 9), 10240),
        ("fp8_p64", 256, 4, 4096, 4, 148, (1, 56, 9), 10240),
        ("fp8_p64", 1, 4, 8192, 1, 148, (4, 1, 17), 264),
        ("fp8_p64", 8, 4, 8192, 1, 148, (1, 2, 65), 2112),
        # bf16 v4 geometry (unit 2): b1 x 16k q8 splits twice into 17-block shards.
        ("bf16_p16", 1, 8, 16384, 4, 148, (2, 1, 17), 136),
        ("bf16_p16", 8, 8, 16383, 4, 148, (1, 4, 33), 1088),
        # D256 (unit 4): the discrete cp4 split family and the row-tile multiplier.
        ("fp8_p64_d256", 1, 4, 32764, 4, 148, (8, 1, 8), 32),
        ("fp8_p64_d256", 16, 4, 32764, 4, 148, (2, 1, 32), 512),
        ("fp8_p64_d256", 32, 4, 32764, 4, 148, (1, 1, 64), 1024),
        ("fp8_p64_d256", 128, 5, 32763, 4, 148, (1, 5, 64), 8192),
    ],
)
def test_static_shape_mirror_matches_the_measured_geometry(
    kind, batch, q_len, prefix, cp_world, sm_count, expected, items
) -> None:
    num_q_heads, num_kv_heads = _KIND_HEADS[kind]
    max_local = _local_len(prefix, q_len, cp_world, 0)
    assert (
        dcp_static_shape(
            kind,
            batch_size=batch,
            q_len=q_len,
            num_kv_heads=num_kv_heads,
            max_local_seq_len=max_local,
            cp_world=cp_world,
            sm_count=sm_count,
        )
        == expected
    )
    band = dcp_balanced_band(
        kind,
        batch_size=batch,
        q_len=q_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=_KIND_HEAD_DIM[kind],
        max_local_seq_len=max_local,
        cp_world=cp_world,
        sm_count=sm_count,
        arch="sm_100a" if sm_count == 148 else "sm_103a",
    )
    assert (band.num_split, band.waves, band.blocks_per_cta, band.items) == (
        *expected,
        items,
    )


def test_packed_row_instance_follows_the_live_rows() -> None:
    assert [dcp_balanced_n_rows("bf16_p16", q) for q in range(1, 9)] == [32] * 4 + [
        64
    ] * 4
    assert [dcp_balanced_n_rows("fp8_p64", q) for q in range(1, 9)] == [32] * 4 + [
        64
    ] * 4
    assert [dcp_balanced_n_rows("fp8_p64_d256", q) for q in range(1, 9)] == [32] * 2 + [
        64
    ] * 6
    for q_len in (0, 9):
        with pytest.raises(ValueError, match="q_len"):
            dcp_balanced_n_rows("bf16_p16", q_len)


def test_balanced_scratch_sizes_are_shape_independent() -> None:
    # 148 SMs: 2368 split items x FP32 [64, head_dim] partials (+ 2369 stats
    # slots of 128 words after a 256-byte alignment); 1184 split tiles x 4
    # counter words plus the four queue counters.
    assert get_dcp_spec_balanced_workspace_bytes(148) == 78_807_552
    assert get_dcp_spec_balanced_workspace_bytes(148, 256) == 156_402_176
    assert get_dcp_spec_balanced_counter_bytes(148) == 18_960
    assert get_dcp_spec_balanced_workspace_bytes(
        152
    ) > get_dcp_spec_balanced_workspace_bytes(148)
    with pytest.raises(ValueError, match="head_dim"):
        get_dcp_spec_balanced_workspace_bytes(148, 64)
    with pytest.raises(ValueError, match="sm_count"):
        get_dcp_spec_balanced_counter_bytes(0)
    assert (
        cake_dcp.get_dcp_spec_balanced_workspace_bytes
        is get_dcp_spec_balanced_workspace_bytes
    )


def test_balanced_uri_names_the_family_instance_and_pins() -> None:
    # the BF16 head_dim-128 family ships two programs (1 planner, 2 whole-tile static): its URI names the program
    assert get_dcp_spec_balanced_uri("dcp_spec_bf16_balanced", "sm100a", 32, 2) == (
        f"cake_fmha_dcp_spec_bf16_balanced_n32_program2_sm100a_{CAKE_FMHA_JIT_TAG}"
    )
    assert get_dcp_spec_balanced_uri(
        "dcp_spec_bf16_fp8_d256_balanced", "sm103a", 64
    ).startswith("cake_fmha_dcp_spec_bf16_fp8_d256_balanced_n64_sm103a_")
    assert (
        tuple(_KIND_FAMILY[kind] for kind in DCP_BALANCED_KINDS)
        == DCP_BALANCED_FAMILIES
    )
    assert DCP_BALANCED_N_ROWS == (32, 64)
    for family in DCP_BALANCED_FAMILIES:
        variants = dcp_balanced_program_variants(family)
        if variants is None:
            with pytest.raises(ValueError, match="one program"):
                get_dcp_spec_balanced_uri(family, "sm100a", 32, 0)
            continue
        key = variants["key"]
        for program in variants["values"]:
            assert get_dcp_spec_balanced_uri(family, "sm100a", 32, program).startswith(
                f"cake_fmha_{family}_n32_{key}{program}_sm100a_"
            )
        with pytest.raises(ValueError, match=key):
            get_dcp_spec_balanced_uri(family, "sm100a", 32)
        with pytest.raises(ValueError, match=key):
            get_dcp_spec_balanced_uri(family, "sm100a", 32, 7)
    with pytest.raises(ValueError, match="n_rows"):
        get_dcp_spec_balanced_uri("dcp_spec_bf16_balanced", "sm100a", 48)
    with pytest.raises(ValueError, match="family"):
        get_dcp_spec_balanced_uri("dcp_spec_bf16_v4", "sm100a", 32)
    with pytest.raises(ValueError, match="target"):
        get_dcp_spec_balanced_uri("dcp_spec_bf16_balanced", "sm90a", 32)


def test_balanced_program_rule_mirrors_the_manifest() -> None:
    """A family without program variants launches ``None``; one with them follows the manifest rule: the items lower
    bound against the grid, then the static one-wave regime from the page-table width."""

    for kind in DCP_BALANCED_KINDS:
        variants = dcp_balanced_program_variants(_KIND_FAMILY[kind])
        if variants is None:
            for batch in (1, 256):
                assert (
                    dcp_balanced_program(
                        kind,
                        batch_size=batch,
                        num_kv_heads=8,
                        max_pages_per_seq=36,
                        sm_count=148,
                    )
                    is None
                )
            continue
        assert variants["items_lower_bound"] == "batch_size * num_kv_heads"
        below, above, static = (
            int(variants["below_grid"]),
            int(variants["at_or_above_grid"]),
            int(variants["static_one_wave"]),
        )
        regime = variants["static_one_wave_regime"]
        # the rule's programs cover the family's values (the BF16 family runs one program below and at or above the grid
        # and the swapped-QK whole-tile program on the regime's architectures / packed-row instances)
        rule_values = {below, above, static}
        if "swapped" in regime:
            rule_values.add(int(regime["swapped"]["program"]))
        assert sorted(rule_values) == sorted(int(v) for v in variants["values"])

        def program(batch, hkv, pages, sm, **selection):
            return dcp_balanced_program(
                kind,
                batch_size=batch,
                num_kv_heads=hkv,
                max_pages_per_seq=pages,
                sm_count=sm,
                **selection,
            )

        if regime.get("form") == "whole_tiles":
            # the BF16 head_dim-128 family: planner / whole-tile static, from the export-time regime table
            assert kind == "bf16_p16" and (below, above, static) == (1, 1, 2)
            assert regime["page_size"] == 16 and regime["one_request"] == 1
            table = regime["whole_pages_max"]
            assert table["148"]["8"] == 112 and table["152"]["8"] == 112
            for sm in (148, 152):
                # the b1 s4096 cp4 row (65 pages, 8 KV-head tiles) and the table's bound; 113 pages is the planner
                assert program(1, 8, 65, sm) == static
                assert (
                    program(1, 8, 112, sm) == static and program(1, 8, 113, sm) == below
                )
                # the 16k row (257 pages), two and eight requests: the planner program
                assert program(1, 8, 257, sm) == below
                assert program(2, 8, 65, sm) == below and program(8, 8, 65, sm) == below
                # at or above the grid the family keeps its planner body
                assert program(-(-sm // 8), 8, 65, sm) == above == below
            # an SM count outside the table keeps the planner program
            assert program(1, 8, 65, 160) == below
            # unit 81 lever A'': the swapped-QK whole-tile program on sm_100a for the 32-row instance only; the 64-row
            # instance, sm_103a, the SM107 family target and unknown selection keep the whole-tile program
            swapped = regime["swapped"]
            assert swapped == {"program": 3, "arches": ["sm_100a"], "n_rows": [32]}
            for sm in (148, 152):
                assert (
                    program(1, 8, 65, sm, arch="sm_100a", n_rows=32)
                    == swapped["program"]
                )
                assert (
                    program(1, 8, 112, sm, arch="sm_100a", n_rows=32)
                    == swapped["program"]
                )
                assert program(1, 8, 113, sm, arch="sm_100a", n_rows=32) == below
                assert program(2, 8, 65, sm, arch="sm_100a", n_rows=32) == below
                assert program(1, 8, 65, sm, arch="sm_100a", n_rows=64) == static
                assert program(1, 8, 65, sm, arch="sm_103a", n_rows=32) == static
                assert program(1, 8, 65, sm, arch="sm107a", n_rows=32) == static
                assert program(1, 8, 65, sm, n_rows=32) == static
                assert program(1, 8, 65, sm, arch="sm_100a") == static
            continue
        assert regime == {
            "page_size": 64,
            "chunk_tokens": 256,
            "min_chunks": 3,
            "reduce_tickets_per_tile": 8,
        }

        for sm in (148, 152):
            # the b1 s8192 cp4 row (36 pages -> 9 chunks per pair; 8 x (9 + 8) = 136 tickets) is the static regime
            assert program(1, 8, 36, sm) == static
            # the regime bound: 8 pairs x (n_max + 8) <= grid
            n_edge = sm // 8 - 8
            assert program(1, 8, 4 * n_edge, sm) == static
            assert program(1, 8, 4 * (n_edge + 1), sm) == below
            # fewer than three chunks per pair: the planner's whole / two-chunk plans
            assert program(1, 8, 8, sm) == below and program(1, 8, 9, sm) == static
            # cp1_peer_b1 (132 pages), b2 at 36 pages and b8 at any width: planner program with the fold
            assert program(1, 8, 132, sm) == below
            assert program(2, 8, 36, sm) == below
            assert program(8, 8, 36, sm) == below
            assert program(8, 8, 20, sm) == below
            assert program((sm - 1) // 8, 8, 36, sm) == below
            # at least one chunk ticket per pair fills the grid: the fold can never be taken
            assert program(-(-sm // 8), 8, 36, sm) == above
            assert program(64, 8, 36, sm) == above
            assert program(256, 8, 36, sm) == above
        assert program(37, 4, 36, 148) == above and program(36, 4, 36, 148) == below
        # single KV head: nine requests of seven chunks fit (9 x 15 = 135), ten do not
        assert program(9, 1, 25, 148) == static and program(10, 1, 25, 148) == below
    with pytest.raises(ValueError, match="sm_count"):
        dcp_balanced_program(
            "fp8_p64", batch_size=1, num_kv_heads=8, max_pages_per_seq=36, sm_count=0
        )
    with pytest.raises(ValueError, match="max_pages_per_seq"):
        dcp_balanced_program(
            "fp8_p64", batch_size=1, num_kv_heads=8, max_pages_per_seq=0, sm_count=148
        )
    with pytest.raises(ValueError, match="kind"):
        dcp_balanced_program(
            "bf16_p64", batch_size=1, num_kv_heads=8, max_pages_per_seq=36, sm_count=148
        )


def test_fp8_program_row_selection(monkeypatch) -> None:
    """The b1 one-wave row runs the static one-wave program when the caller forces the balanced route (the
    ``auto`` band keeps that one-wave row on the static specialization), b8 the planner with the eight-slice
    fold and the multi-wave uniform rows the planner without that fold body."""

    variants = dcp_balanced_program_variants("dcp_spec_bf16_fp8_balanced")
    if variants is None:
        pytest.skip("the shipped E4M3 head_dim-128 family has one program")
    calls, _launches = _patch_loaders(monkeypatch)
    for batch, route, expected in (
        (1, "balanced", variants["static_one_wave"]),
        (8, "auto", variants["below_grid"]),
        (64, "auto", variants["at_or_above_grid"]),
    ):
        calls["balanced"].clear()
        calls["static"].clear()
        inputs = _rank_inputs(
            "fp8_p64",
            batch=batch,
            q_len=4,
            prefixes=[8192] * batch,
            cp_world=4,
            cp_rank=0,
        )
        run_dcp_spec_decode(**inputs, route=route)
        assert calls["balanced"] == [
            ("dcp_spec_bf16_fp8_balanced", "sm100a", 32, int(expected))
        ]
        assert not calls["static"]
    # ``auto`` routes the one-wave b1 row to the static specialization (band reason ``one_wave``).
    calls["balanced"].clear()
    calls["static"].clear()
    inputs = _rank_inputs(
        "fp8_p64", batch=1, q_len=4, prefixes=[8192], cp_world=4, cp_rank=0
    )
    run_dcp_spec_decode(**inputs)
    assert not calls["balanced"] and len(calls["static"]) == 1


def test_bf16_program_row_selection(monkeypatch) -> None:
    """The bf16 b1 s4096 cp4 row (one request, 65 pages, eight KV-head tiles) runs the whole-tile static program when
    the caller forces the balanced route (the ``auto`` band keeps that one-wave row on the static specialization);
    b8 and the 16k b1 row run the planner program."""

    variants = dcp_balanced_program_variants("dcp_spec_bf16_balanced")
    if variants is None:
        pytest.skip("the shipped BF16 head_dim-128 family has one program")
    swapped = variants["static_one_wave_regime"]["swapped"]
    for target, batch, prefix, route, q_len, n_rows, expected in (
        # unit 81 lever A'': on sm_100a the 32-row whole-tile launch runs the swapped-QK program (3); the 64-row instance
        # (q_len 8) and sm_103a run the whole-tile program (2)
        ("sm100a", 1, 4096, "balanced", 4, 32, swapped["program"]),
        ("sm100a", 1, 4096, "balanced", 8, 64, variants["static_one_wave"]),
        ("sm103a", 1, 4096, "balanced", 4, 32, variants["static_one_wave"]),
        ("sm100a", 8, 4096, "auto", 4, 32, variants["below_grid"]),
        ("sm100a", 1, 16384, "balanced", 4, 32, variants["below_grid"]),
    ):
        calls, _launches = _patch_loaders(monkeypatch, target=target)
        inputs = _rank_inputs(
            "bf16_p16",
            batch=batch,
            q_len=q_len,
            prefixes=[prefix] * batch,
            cp_world=4,
            cp_rank=0,
        )
        run_dcp_spec_decode(**inputs, route=route)
        assert calls["balanced"] == [
            ("dcp_spec_bf16_balanced", target, n_rows, int(expected))
        ], (target, batch, prefix, route, q_len)
        assert not calls["static"]
    calls, _launches = _patch_loaders(monkeypatch)
    # ``auto`` routes the one-wave b1 row to the static specialization (band reason ``one_wave``).
    calls["balanced"].clear()
    calls["static"].clear()
    inputs = _rank_inputs(
        "bf16_p16", batch=1, q_len=4, prefixes=[4096], cp_world=4, cp_rank=0
    )
    run_dcp_spec_decode(**inputs)
    assert not calls["balanced"] and len(calls["static"]) == 1


# Launch resources and argument order the export pins per family
# (EXPORT_R3_SPEC section 3: the bodies' THREADS / SMEM_TOTAL defines; the
# launch binding owns both). The adapters pass the arguments in this order.
_FAMILY_SMEM = {
    "dcp_spec_bf16_balanced": 226304,
    "dcp_spec_bf16_fp8_balanced": 160768,
    "dcp_spec_bf16_fp8_d256_balanced": 226304,
}
_LAUNCH_PARAMETERS = [
    "Q", "K", "V", "O_ptr", "LSE_ptr", "page_table", "causal_seqlens_kv_global",
    "partial_o", "partial_stats", "tile_counters", "queue_counters",
    "max_pages_per_seq", "softmax_scale_log2", "output_scale",
    "num_q_heads", "num_kv_heads", "batch_size", "q_len", "cp_rank", "cp_world_log2", "max_items",
]  # fmt: skip
_KIND_MANIFEST_BAND = {
    "bf16_p16": "bf16_p16",
    "fp8_p64": "fp8_p64",
    "fp8_p64_d256": "fp8_d256",
}


def test_balanced_families_ship_one_program_per_selector_with_both_packed_instances() -> (
    None
):
    families = get_dcp_spec_registry()["families"]
    csrc_dir = Path(__file__).resolve().parents[2] / "csrc" / "cake_fmha"
    header = (csrc_dir / "include" / "cake_fmha.h").read_text()
    for family in DCP_BALANCED_FAMILIES:
        entry = families[family]
        variants = dcp_balanced_program_variants(family)
        if variants is None:
            # one shape-independent program: the two packed tiles are its -DN_ROWS instances
            programs = {None: f"cuda/dcp_spec/{family}/kernel.cu"}
            expected_selectors = [{"n_rows": n_rows} for n_rows in (32, 64)]
        else:
            # one such program per value of the program selector; the host picks it from launch metadata
            key = variants["key"]
            assert sorted(variants["values"]) == sorted(
                int(v) for v in variants["bodies"]
            )
            assert variants["default"] in variants["values"]
            programs = {
                int(value): f"cuda/dcp_spec/{family}/kernel_{key}{int(value)}.cu"
                for value in variants["values"]
            }
            expected_selectors = [
                {key: int(value), "n_rows": n_rows}
                for value in sorted(variants["values"])
                for n_rows in (32, 64)
            ]
        assert [
            dict(sorted(member["selector"].items()))
            for member in entry["source_family"]
        ] == expected_selectors, family
        assert sorted(
            path.name for path in (csrc_dir / "cuda" / "dcp_spec" / family).iterdir()
        ) == sorted(Path(program).name for program in programs.values())
        texts = {}
        for value, program in programs.items():
            assert (csrc_dir / program).is_file(), program
            program_text = (csrc_dir / program).read_text()
            assert (
                program_text.count("#ifndef N_ROWS\n#define N_ROWS 64\n#endif\n") == 1
            )
            # one text for both architectures: codegen's device-pass guard selects the sm_100a lowering
            assert "#if __CUDA_ARCH__ == 1000" in program_text
            assert "const __grid_constant__ CUtensorMap" in program_text
            assert "TensorMap const*" not in program_text
            texts[value] = program_text
        assert len(set(texts.values())) == len(texts)  # distinct programs
        for member in entry["source_family"]:
            program = programs[
                None if variants is None else int(member["selector"][variants["key"]])
            ]
            assert member["sources"] == {"sm_100a": program, "sm_103a": program}
            assert member["defines"] == {"N_ROWS": member["selector"]["n_rows"]}
        assert entry["binding_source"] == f"bindings/cake_fmha_{family}_binding.cu"
        assert (csrc_dir / entry["binding_source"]).is_file()
        assert entry["launch_binding"] == f"cake_fmha_launch_{family}"
        assert f"cudaError_t cake_fmha_launch_{family}(" in header
        assert entry["public_symbol"].endswith(f"cake_fmha_{family}")
        assert entry["threads"] == 384
        assert entry["dynamic_shared_memory_bytes"] == _FAMILY_SMEM[family]
        assert entry["parametric_macros"] == ["N_ROWS"]
        expected = list(_LAUNCH_PARAMETERS)
        if family == "dcp_spec_bf16_balanced":
            expected.remove("output_scale")
        assert [parameter["name"] for parameter in entry["parameters"]] == expected


def test_balanced_route_constants_match_the_shipped_manifest() -> None:
    routes = get_dcp_spec_registry()["balanced_routes"]
    pins = {
        "bf16_p16": (DCP_BALANCED_BF16_MIN_Q_LEN, DCP_BALANCED_BF16_MAX_Q_LEN, DCP_BALANCED_BF16_MIN_ITEMS, DCP_BALANCED_BF16_LONG_TILE_BLOCKS, None),
        "fp8_p64": (DCP_BALANCED_FP8_MIN_Q_LEN, DCP_BALANCED_FP8_MAX_Q_LEN, DCP_BALANCED_FP8_MIN_ITEMS, DCP_BALANCED_FP8_LONG_TILE_BLOCKS, DCP_BALANCED_FP8_TWO_WAVE_MIN_ITEMS),
        "fp8_p64_d256": (DCP_BALANCED_D256_MIN_Q_LEN, DCP_BALANCED_D256_MAX_Q_LEN, DCP_BALANCED_D256_MIN_ITEMS, DCP_BALANCED_D256_LONG_TILE_BLOCKS, None),
    }  # fmt: skip
    for kind, (
        min_q_len,
        max_q_len,
        min_items,
        long_tile_blocks,
        two_wave,
    ) in pins.items():
        route = routes[_KIND_FAMILY[kind]]
        band = route["dispatch_band"]
        if "band" not in band:  # the producer block keyed by profile
            band = band[_KIND_MANIFEST_BAND[kind]]
        assert band["family"] == _KIND_FAMILY[kind]
        assert band["band"]["chunk_tokens"] == DCP_BALANCED_CHUNK_TOKENS
        assert band["band"]["min_items"] == min_items
        assert band["band"]["long_tile_blocks"] == long_tile_blocks
        if two_wave is not None:
            assert band["band"]["two_wave_min_items"] == two_wave
            assert band["band"]["two_wave_min_items_default"] == min_items
        else:
            assert "two_wave_min_items" not in band["band"]
        if kind == "fp8_p64":
            assert (
                band["band"]["long_tile_blocks_by_arch"]
                == DCP_BALANCED_FP8_LONG_TILE_BLOCKS_BY_ARCH
            )
        else:
            assert "long_tile_blocks_by_arch" not in band["band"]
        if kind == "fp8_p64_d256":
            assert (
                band["band"]["one_wave_min_q_len"]
                == DCP_BALANCED_D256_ONE_WAVE_MIN_Q_LEN
            )
            assert (
                band["band"]["one_wave_long_tile_blocks"]
                == DCP_BALANCED_D256_ONE_WAVE_LONG_TILE_BLOCKS
            )
            assert (
                band["band"]["one_wave_long_tile_blocks_by_arch"]
                == DCP_BALANCED_D256_ONE_WAVE_LONG_TILE_BLOCKS_BY_ARCH
            )
        else:
            assert "one_wave_long_tile_blocks" not in band["band"]
        contract = band["contract"]
        assert (contract["min_q_len"], contract["max_q_len"]) == (min_q_len, max_q_len)
        assert contract["max_requests"] == DCP_BALANCED_MAX_REQUESTS
        assert contract["head_dim"] == _KIND_HEAD_DIM[kind]
        num_q_heads, num_kv_heads = _KIND_HEADS[kind]
        assert contract["q_heads_per_kv_head"] == num_q_heads // num_kv_heads
        selector = route["selector"]
        assert selector["key"] == "n_rows" and selector["supported"] == [32, 64]
        assert selector["by_q_len"] == {
            str(q_len): dcp_balanced_n_rows(kind, q_len)
            for q_len in range(min_q_len, max_q_len + 1)
        }
        assert route["launch"]["threads"] == 384
        assert route["launch"]["max_requests"] == DCP_BALANCED_MAX_REQUESTS
        assert route["profile"]["head_dim"] == _KIND_HEAD_DIM[kind]
        assert route["profile"]["page_size"] == _KIND_PAGE_SIZE[kind]
        assert route["profile"]["cp_world"] == [1, 2, 4, 8]


@pytest.mark.parametrize("target", ["sm100a", "sm103a", "sm100f"])
def test_balanced_jit_selects_the_packed_instance_and_launch_binding(
    monkeypatch, target
) -> None:
    jit_dcp = importlib.import_module("flashinfer.jit.cake_dcp")
    source_dir = Path(__file__).resolve().parents[2] / "csrc" / "cake_fmha"
    monkeypatch.setattr(jit_dcp, "get_cake_fmha_csrc_dir", lambda: source_dir)
    monkeypatch.setattr(
        jit_dcp, "gen_jit_spec", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    jit_dcp.gen_dcp_spec_balanced_module.cache_clear()
    try:
        arch = target.replace("sm", "", 1)
        for family in DCP_BALANCED_FAMILIES:
            variants = jit_dcp.dcp_balanced_program_variants(family)
            programs = [None] if variants is None else list(variants["values"])
            for n_rows in DCP_BALANCED_N_ROWS:
                for program in programs:
                    spec = jit_dcp.gen_dcp_spec_balanced_module(
                        family, target, n_rows, program
                    )
                    assert (
                        f"-gencode=arch=compute_{arch},code=sm_{arch}"
                        in spec.extra_cuda_cflags
                    )
                    assert spec.name == get_dcp_spec_balanced_uri(
                        family, target, n_rows, program
                    )
                    body, launch_binding, api_binding = (
                        Path(source) for source in spec.sources
                    )
                    expected_body = (
                        "kernel.cu"
                        if variants is None
                        else f"kernel_{variants['key']}{program}.cu"
                    )
                    assert body.name == expected_body and body.parent.name == family
                    assert body.parent.parent.name == "dcp_spec"
                    assert f"-DN_ROWS={n_rows}" in spec.extra_cuda_cflags
                    assert launch_binding.name == f"cake_fmha_{family}_binding.cu"
                    assert launch_binding.parent.name == "bindings"
                    assert api_binding.name == f"cake_fmha_{family}_jit_binding.cu"
                    assert api_binding.parent.name == "jit"
                    assert (
                        f"-DCAKE_FMHA_DCP_BALANCED_LAUNCH=cake_fmha_launch_{family}"
                        in spec.extra_cuda_cflags
                    )
                    assert "-lcuda" in spec.extra_ldflags
                    assert not any(
                        flag.startswith(("-DQ_LEN", "-DBATCH_SIZE", "-DCP_WORLD"))
                        for flag in spec.extra_cuda_cflags
                    )
    finally:
        jit_dcp.gen_dcp_spec_balanced_module.cache_clear()


# ---------------------------------------------------------------------------
# Wiring of run_dcp_spec_decode (CPU, the module loaders replaced)
# ---------------------------------------------------------------------------


def _rank_inputs(
    kind, *, batch, q_len, prefixes, cp_world, cp_rank, device="cpu", sm_count=148
):
    num_q_heads, num_kv_heads = _KIND_HEADS[kind]
    head_dim, page_size = _KIND_HEAD_DIM[kind], _KIND_PAGE_SIZE[kind]
    max_local = _max_local(prefixes, q_len, cp_world, cp_rank)
    pages = -(-max(max_local, 1) // page_size)
    workspace = torch.empty(
        get_dcp_spec_balanced_workspace_bytes(sm_count, head_dim),
        dtype=torch.uint8,
        device=device,
    )
    counter = torch.zeros(
        get_dcp_spec_balanced_counter_bytes(sm_count), dtype=torch.uint8, device=device
    )
    return {
        "query": torch.zeros(
            (batch * q_len, num_q_heads, head_dim), dtype=torch.bfloat16, device=device
        ),
        "k_cache": torch.zeros(
            (batch * pages + 1, num_kv_heads, page_size, head_dim),
            dtype=_KIND_KV_DTYPE[kind],
            device=device,
        ),
        "v_cache": torch.zeros(
            (batch * pages + 1, num_kv_heads, page_size, head_dim),
            dtype=_KIND_KV_DTYPE[kind],
            device=device,
        ),
        "workspace_buffer": workspace,
        "block_tables": torch.zeros((batch, pages), dtype=torch.int32, device=device),
        "seq_lens": torch.tensor(
            [_local_len(int(p), q_len, cp_world, cp_rank) for p in prefixes],
            dtype=torch.int32,
            device=device,
        ),
        "causal_seqlens_kv_global": torch.tensor(
            list(prefixes), dtype=torch.int32, device=device
        ),
        "max_local_seq_len": max_local,
        "bmm1_scale": head_dim**-0.5,
        "bmm2_scale": 1.0,
        "cp_world": cp_world,
        "cp_rank": cp_rank,
        "q_len_per_req": q_len,
        "out": torch.empty(
            (batch * q_len, num_q_heads, head_dim), dtype=torch.bfloat16, device=device
        ),
        "lse": torch.empty(
            (batch * q_len, num_q_heads), dtype=torch.float32, device=device
        ),
        "completion_buffer": counter,
    }


def _patch_loaders(monkeypatch, *, sm_count=148, target="sm100a"):
    dcp = importlib.import_module("flashinfer.cake_dcp")
    jit_dcp = importlib.import_module("flashinfer.jit.cake_dcp")
    calls = {"balanced": [], "static": [], "fp8": [], "d256": []}
    launches = {"balanced": [], "static": [], "fp8": [], "d256": []}

    def loader(name):
        def load(*args):
            calls[name].append(args)
            return SimpleNamespace(
                run=lambda *run_args: launches[name].append(run_args)
            )

        return load

    monkeypatch.setattr(dcp, "get_device_sm_count", lambda _device: sm_count)
    monkeypatch.setattr(dcp, "_select_target", lambda _device: target)
    monkeypatch.setattr(jit_dcp, "load_dcp_spec_balanced_module", loader("balanced"))
    monkeypatch.setattr(jit_dcp, "load_dcp_spec_static_module", loader("static"))
    return calls, launches


def test_bf16_band_row_launches_the_balanced_program(monkeypatch) -> None:
    calls, launches = _patch_loaders(monkeypatch)
    inputs = _rank_inputs(
        "bf16_p16", batch=8, q_len=4, prefixes=[4096] * 8, cp_world=4, cp_rank=0
    )
    run_dcp_spec_decode(**inputs)
    # eight requests: the planner program (1); the whole-tile static program (2) is the one-request regime
    assert calls["balanced"] == [("dcp_spec_bf16_balanced", "sm100a", 32, 1)]
    assert (
        dcp_balanced_program(
            "bf16_p16",
            batch_size=8,
            num_kv_heads=8,
            max_pages_per_seq=int(inputs["block_tables"].shape[1]),
            sm_count=148,
        )
        == 1
    )
    assert not calls["static"]
    (args,) = launches["balanced"]
    assert (
        args[0] is inputs["query"]
        and args[1] is inputs["k_cache"]
        and args[2] is inputs["v_cache"]
    )
    assert args[3] is inputs["out"] and args[4] is inputs["lse"]
    assert (
        args[5] is inputs["block_tables"]
        and args[6] is inputs["causal_seqlens_kv_global"]
    )
    assert (
        args[7] is inputs["workspace_buffer"] and args[8] is inputs["completion_buffer"]
    )
    assert args[9] == pytest.approx(128**-0.5 * _LOG2_E)
    assert args[10:] == (0, 4, 64, 8, 8, 4, 148)


def test_bf16_q8_row_uses_the_64_row_instance(monkeypatch) -> None:
    calls, _launches = _patch_loaders(monkeypatch)
    inputs = _rank_inputs(
        "bf16_p16", batch=1, q_len=8, prefixes=[16384], cp_world=4, cp_rank=0
    )
    run_dcp_spec_decode(**inputs)
    # 4097 rank-local keys = 257 pages: outside the whole-tile regime, the planner program
    assert calls["balanced"] == [("dcp_spec_bf16_balanced", "sm100a", 64, 1)]


def test_band_row_without_balanced_scratch_keeps_the_static_route(monkeypatch) -> None:
    calls, launches = _patch_loaders(monkeypatch)
    inputs = _rank_inputs(
        "bf16_p16", batch=8, q_len=4, prefixes=[4096] * 8, cp_world=4, cp_rank=0
    )
    inputs["workspace_buffer"] = torch.empty(1, dtype=torch.uint8)
    inputs["completion_buffer"] = None
    run_dcp_spec_decode(**inputs)
    assert not calls["balanced"]
    # 9 local blocks: the unsplit specialization
    assert calls["static"][0][:2] == ("bf16_v1", "retain1")
    assert len(launches["static"]) == 1


def test_forced_balanced_route_reports_the_missing_scratch(monkeypatch) -> None:
    _patch_loaders(monkeypatch)
    inputs = _rank_inputs(
        "bf16_p16", batch=8, q_len=4, prefixes=[4096] * 8, cp_world=4, cp_rank=0
    )
    inputs["completion_buffer"] = None
    with pytest.raises(ValueError, match="multi_ctas_kv_counter_buffer is required"):
        run_dcp_spec_decode(**inputs, route="balanced")
    inputs["completion_buffer"] = torch.zeros(16, dtype=torch.uint8)
    with pytest.raises(ValueError, match="too small for the balanced DCP route"):
        run_dcp_spec_decode(**inputs, route="balanced")
    with pytest.raises(ValueError, match="route must be one of"):
        run_dcp_spec_decode(**inputs, route="fastest")


def test_forced_static_route_bypasses_the_band(monkeypatch) -> None:
    calls, launches = _patch_loaders(monkeypatch)
    inputs = _rank_inputs(
        "bf16_p16", batch=8, q_len=4, prefixes=[4096] * 8, cp_world=4, cp_rank=0
    )
    run_dcp_spec_decode(**inputs, route="static")
    assert (
        not calls["balanced"]
        and calls["static"][0][:2] == ("bf16_v1", "retain1")
        and len(launches["static"]) == 1
    )


def test_forced_balanced_route_serves_a_row_outside_the_band(monkeypatch) -> None:
    calls, launches = _patch_loaders(monkeypatch)
    inputs = _rank_inputs(
        "bf16_p16", batch=1, q_len=4, prefixes=[4096], cp_world=4, cp_rank=0
    )
    assert (
        _band(
            "bf16_p16",
            batch=1,
            q_len=4,
            prefix=4096,
            cp_world=4,
            cp_rank=0,
            arch="sm_100a",
        ).route
        == "static"
    )
    run_dcp_spec_decode(**inputs, route="balanced")
    # one request of 65 pages on 148 SMs is inside the whole-tile regime (whole_pages_max 112): the swapped
    # whole-tile program (3) on sm100a (sm103a keeps program 2)
    assert (
        calls["balanced"] == [("dcp_spec_bf16_balanced", "sm100a", 32, 3)]
        and len(launches["balanced"]) == 1
    )


def test_bf16_balanced_route_rejects_nonunit_bmm2_scale(monkeypatch) -> None:
    _patch_loaders(monkeypatch)
    inputs = _rank_inputs(
        "bf16_p16", batch=8, q_len=4, prefixes=[4096] * 8, cp_world=4, cp_rank=0
    )
    inputs["bmm2_scale"] = 0.5
    with pytest.raises(ValueError, match="BF16/page16"):
        run_dcp_spec_decode(**inputs)


def test_fp8_band_row_launches_the_e4m3_program_with_both_scales(monkeypatch) -> None:
    calls, launches = _patch_loaders(monkeypatch)
    inputs = _rank_inputs(
        "fp8_p64", batch=8, q_len=4, prefixes=[8192] * 8, cp_world=4, cp_rank=0
    )
    inputs["bmm1_scale"] = 0.125
    inputs["bmm2_scale"] = 0.25
    run_dcp_spec_decode(**inputs)
    assert calls["balanced"] == [
        (
            "dcp_spec_bf16_fp8_balanced",
            "sm100a",
            32,
            dcp_balanced_program(
                "fp8_p64",
                batch_size=8,
                num_kv_heads=8,
                max_pages_per_seq=int(inputs["block_tables"].shape[1]),
                sm_count=148,
            ),
        )
    ]
    assert not calls["static"]
    (args,) = launches["balanced"]
    assert args[1].dtype == torch.uint8 and args[2].dtype == torch.uint8
    assert args[1].data_ptr() == inputs["k_cache"].data_ptr()
    assert args[9] == pytest.approx(0.125 * _LOG2_E)
    assert args[10] == pytest.approx(0.25)
    assert args[11:] == (0, 4, 64, 8, 8, 4, 148)


def test_fp8_two_wave_row_follows_the_architecture_floor(monkeypatch) -> None:
    # prod_b8_s4096_q4_cp4 (320 items at exactly two static waves): round 3 kept sm_100a
    # (148 SMs) static behind a 384-item floor; the round-5 programs win the row on both
    # parts (1.29 GB300 / 1.17 B200), so the per-architecture floor admits it on both.
    calls, _launches = _patch_loaders(monkeypatch, sm_count=148, target="sm100a")
    inputs = _rank_inputs(
        "fp8_p64", batch=8, q_len=4, prefixes=[4096] * 8, cp_world=4, cp_rank=0
    )
    run_dcp_spec_decode(**inputs)
    assert (
        calls["balanced"]
        == [
            (
                "dcp_spec_bf16_fp8_balanced",
                "sm100a",
                32,
                dcp_balanced_program(
                    "fp8_p64",
                    batch_size=8,
                    num_kv_heads=8,
                    max_pages_per_seq=int(inputs["block_tables"].shape[1]),
                    sm_count=148,
                ),
            )
        ]
        and not calls["static"]
    )
    calls, _launches = _patch_loaders(monkeypatch, sm_count=152, target="sm103a")
    inputs = _rank_inputs(
        "fp8_p64",
        batch=8,
        q_len=4,
        prefixes=[4096] * 8,
        cp_world=4,
        cp_rank=0,
        sm_count=152,
    )
    run_dcp_spec_decode(**inputs)
    assert (
        calls["balanced"]
        == [
            (
                "dcp_spec_bf16_fp8_balanced",
                "sm103a",
                32,
                dcp_balanced_program(
                    "fp8_p64",
                    batch_size=8,
                    num_kv_heads=8,
                    max_pages_per_seq=int(inputs["block_tables"].shape[1]),
                    sm_count=152,
                ),
            )
        ]
        and not calls["static"]
    )


def test_d256_band_row_launches_the_gqa16_program(monkeypatch) -> None:
    calls, launches = _patch_loaders(monkeypatch)
    inputs = _rank_inputs(
        "fp8_p64_d256", batch=64, q_len=5, prefixes=[32763] * 64, cp_world=4, cp_rank=0
    )
    inputs["bmm2_scale"] = 0.5
    run_dcp_spec_decode(**inputs)
    assert calls["balanced"] == [
        (
            "dcp_spec_bf16_fp8_d256_balanced",
            "sm100a",
            64,
            dcp_balanced_program(
                "fp8_p64_d256",
                batch_size=64,
                num_kv_heads=1,
                max_pages_per_seq=int(inputs["block_tables"].shape[1]),
                sm_count=148,
            ),
        )
    ]
    assert not calls["static"]
    (args,) = launches["balanced"]
    assert args[10] == pytest.approx(0.5) and args[11:] == (0, 4, 16, 1, 64, 5, 148)
    # b8 q4 at ctx 32768 stays on the static D256 family (one wave of split-4 tiles).
    inputs = _rank_inputs(
        "fp8_p64_d256", batch=8, q_len=4, prefixes=[32764] * 8, cp_world=4, cp_rank=0
    )
    run_dcp_spec_decode(**inputs)
    assert len(calls["balanced"]) == 1
    assert calls["static"][0][:2] == ("fp8_d256", "splitn")
    # compile-line split count of the one-wave split-4 launch
    assert calls["static"][0][3]["NUM_SPLIT"] == 4


# ---------------------------------------------------------------------------
# GPU parity and graph replay
# ---------------------------------------------------------------------------


def _require_blackwell_dcp() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if get_compute_capability(torch.device("cuda")) == (10, 7):
        pytest.skip("Cake FMHA DCP supports SM100 and SM103, not SM107")
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Cake FMHA DCP requires SM100 or SM103")


def _device_arch() -> str:
    return (
        "sm_103a"
        if get_compute_capability(torch.device("cuda")) == (10, 3)
        else "sm_100a"
    )


def _rank_positions(rank: int, global_len: int, cp_world: int) -> torch.Tensor:
    return torch.arange(rank, global_len, cp_world, dtype=torch.long, device="cuda")


def _local_visible_count(prefix: int, row: int, rank: int, cp_world: int) -> int:
    return max(0, (prefix + row - rank) // cp_world + 1)


def _quantize_fp8(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    scale = max(float(x.abs().amax().item()) / fp8_max, 1.0e-8)
    storage = (x.float() / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return storage, scale


def _pack_rank_cache(global_k, global_v, global_lens, *, rank, cp_world, page_size):
    """Rank-local HND paged caches of ``global_k`` / ``global_v`` (shuffled pages, dummy padding)."""

    batch_size, _, num_kv_heads, head_dim = global_k.shape
    local_lens = [
        int(_rank_positions(rank, length, cp_world).numel()) for length in global_lens
    ]
    data_pages = [max(1, math.ceil(length / page_size)) for length in local_lens]
    total_pages = sum(data_pages)
    dummy_page = total_pages
    k_cache = torch.zeros(
        (total_pages + 1, num_kv_heads, page_size, head_dim),
        dtype=global_k.dtype,
        device="cuda",
    )
    v_cache = torch.zeros_like(k_cache)
    max_local = max(local_lens)
    blocks = max(1, math.ceil(max_local / 128))
    blocks += blocks % 2
    max_pages_per_seq = blocks * 128 // page_size
    block_tables = torch.full(
        (batch_size, max_pages_per_seq), dummy_page, dtype=torch.int32, device="cuda"
    )
    order = torch.randperm(total_pages, device="cuda").to(torch.int32)
    next_page = 0
    for batch_idx, (length, page_count) in enumerate(
        zip(local_lens, data_pages, strict=True)
    ):
        pages = order[next_page : next_page + page_count]
        block_tables[batch_idx, :page_count] = pages
        if length:
            positions = _rank_positions(rank, global_lens[batch_idx], cp_world)
            padded_k = torch.zeros(
                (page_count * page_size, num_kv_heads, head_dim),
                dtype=global_k.dtype,
                device="cuda",
            )
            padded_v = torch.zeros_like(padded_k)
            padded_k[:length].copy_(global_k[batch_idx, positions])
            padded_v[:length].copy_(global_v[batch_idx, positions])
            k_cache[pages.long()] = padded_k.view(
                page_count, page_size, num_kv_heads, head_dim
            ).permute(0, 2, 1, 3)
            v_cache[pages.long()] = padded_v.view(
                page_count, page_size, num_kv_heads, head_dim
            ).permute(0, 2, 1, 3)
        next_page += page_count
    seq_lens = torch.tensor(local_lens, dtype=torch.int32, device="cuda")
    return k_cache, v_cache, block_tables, seq_lens, max_local


def _rank_reference(
    query, represented_k, represented_v, prefix_lens, *, rank, cp_world, sm_scale
):
    batch_size, q_len, num_q_heads, _head_dim = query.shape
    group = num_q_heads // represented_k.shape[2]
    output = torch.zeros_like(query)
    lse = torch.full(
        (batch_size, q_len, num_q_heads),
        -float("inf"),
        dtype=torch.float32,
        device="cuda",
    )
    for batch_idx, prefix in enumerate(prefix_lens):
        positions = _rank_positions(rank, prefix + q_len, cp_world)
        keys = represented_k[batch_idx, positions].repeat_interleave(group, dim=1)
        values = represented_v[batch_idx, positions].repeat_interleave(group, dim=1)
        for row in range(q_len):
            visible = _local_visible_count(prefix, row, rank, cp_world)
            if visible == 0:
                continue
            scores = (
                torch.einsum(
                    "hd,khd->hk", query[batch_idx, row].float(), keys[:visible].float()
                )
                * sm_scale
            )
            output[batch_idx, row] = torch.einsum(
                "hk,khd->hd", torch.softmax(scores, dim=-1), values[:visible].float()
            ).to(torch.bfloat16)
            lse[batch_idx, row] = torch.logsumexp(scores, dim=-1) * _LOG2_E
    return output, lse


class _GpuCase:
    """One rank-local DCP problem: caches, scratch, oracle and both routes."""

    def __init__(
        self, kind, *, prefixes, q_len, cp_world, cp_rank, seed, capacity=None
    ):
        torch.manual_seed(seed)
        self.kind, self.q_len, self.cp_world, self.cp_rank = (
            kind,
            q_len,
            cp_world,
            cp_rank,
        )
        self.prefixes = [int(p) for p in prefixes]
        # ``capacity`` is the per-request prefix the rank cache is packed for: every
        # length vector the case later runs (eagerly or by graph replay) must stay
        # within it, request by request, or the kernel reads pages the rank never
        # received.
        self.capacity = [
            int(c) for c in (self.prefixes if capacity is None else capacity)
        ]
        assert len(self.capacity) == len(self.prefixes)
        assert all(
            0 <= p <= c for p, c in zip(self.prefixes, self.capacity, strict=True)
        )
        self.num_q_heads, self.num_kv_heads = _KIND_HEADS[kind]
        self.head_dim, self.page_size = _KIND_HEAD_DIM[kind], _KIND_PAGE_SIZE[kind]
        batch = len(self.prefixes)
        max_global = max(self.capacity) + q_len
        self.sm_scale = self.head_dim**-0.5
        self.query = (
            torch.randn(
                batch,
                q_len,
                self.num_q_heads,
                self.head_dim,
                dtype=torch.float32,
                device="cuda",
            )
            * 0.2
        ).to(torch.bfloat16)
        k_source = (
            torch.randn(
                batch,
                max_global,
                self.num_kv_heads,
                self.head_dim,
                dtype=torch.float32,
                device="cuda",
            )
            * 0.2
        ).to(torch.bfloat16)
        v_source = (torch.randn_like(k_source, dtype=torch.float32) * 0.2).to(
            torch.bfloat16
        )
        if _KIND_KV_DTYPE[kind] == torch.float8_e4m3fn:
            k_storage, k_scale = _quantize_fp8(k_source)
            v_storage, v_scale = _quantize_fp8(v_source)
            self.represented_k = k_storage.float() * k_scale
            self.represented_v = v_storage.float() * v_scale
            self.bmm1_scale, self.bmm2_scale = self.sm_scale * k_scale, v_scale
        else:
            k_storage, v_storage = k_source, v_source
            self.represented_k, self.represented_v = k_source.float(), v_source.float()
            self.bmm1_scale, self.bmm2_scale = self.sm_scale, 1.0
        self.k_cache, self.v_cache, self.block_tables, self.seq_lens, self.max_local = (
            _pack_rank_cache(
                k_storage,
                v_storage,
                [c + q_len for c in self.capacity],
                rank=cp_rank,
                cp_world=cp_world,
                page_size=self.page_size,
            )
        )
        # The planner lengths of the first vector; ``max_local`` stays the capacity bound.
        self.seq_lens.copy_(
            torch.tensor(
                [_local_len(p, q_len, cp_world, cp_rank) for p in self.prefixes],
                dtype=torch.int32,
                device="cuda",
            )
        )
        self.prefix_tensor = torch.tensor(
            self.prefixes, dtype=torch.int32, device="cuda"
        )
        sm_count = torch.cuda.get_device_properties(
            torch.device("cuda")
        ).multi_processor_count
        static_split = 16 if kind == "bf16_p16" else 16 if kind == "fp8_p64_d256" else 4
        self.workspace = torch.empty(
            max(
                get_dcp_spec_balanced_workspace_bytes(sm_count, self.head_dim),
                cake_dcp.get_dcp_spec_workspace_size_bytes(
                    batch, q_len, self.num_q_heads, static_split, head_dim=self.head_dim
                ),
            ),
            dtype=torch.uint8,
            device="cuda",
        )
        self.counter = torch.zeros(
            max(
                get_dcp_spec_balanced_counter_bytes(sm_count),
                cake_dcp.get_dcp_spec_counter_bytes(batch, q_len, self.num_kv_heads),
            ),
            dtype=torch.uint8,
            device="cuda",
        )
        self.out = torch.empty_like(self.query)
        self.lse = torch.empty(
            (batch, q_len, self.num_q_heads), dtype=torch.float32, device="cuda"
        )

    def band(self):
        return dcp_balanced_band(
            self.kind,
            batch_size=len(self.prefixes),
            q_len=self.q_len,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_local_seq_len=self.max_local,
            cp_world=self.cp_world,
            sm_count=torch.cuda.get_device_properties(
                torch.device("cuda")
            ).multi_processor_count,
            arch=_device_arch(),
        )

    def run(self, route: str) -> None:
        run_dcp_spec_decode(
            query=self.query.flatten(0, 1),
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            workspace_buffer=self.workspace,
            block_tables=self.block_tables,
            seq_lens=self.seq_lens,
            causal_seqlens_kv_global=self.prefix_tensor,
            max_local_seq_len=self.max_local,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=self.bmm2_scale,
            cp_world=self.cp_world,
            cp_rank=self.cp_rank,
            q_len_per_req=self.q_len,
            out=self.out.flatten(0, 1),
            lse=self.lse.flatten(0, 1),
            completion_buffer=self.counter,
            route=route,
        )

    def run_public(self):
        return trtllm_batch_decode_with_kv_cache(
            self.query.flatten(0, 1),
            (self.k_cache, self.v_cache),
            self.workspace,
            self.block_tables,
            self.seq_lens,
            self.max_local,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=self.bmm2_scale,
            out=self.out.flatten(0, 1),
            kv_layout="HND",
            backend="cake",
            q_len_per_req=self.q_len,
            lse=self.lse.flatten(0, 1),
            return_lse=True,
            multi_ctas_kv_counter_buffer=self.counter,
            cp_world=self.cp_world,
            cp_rank=self.cp_rank,
            causal_seqlens_kv_global=self.prefix_tensor,
        )

    def reference(self, prefixes=None):
        return _rank_reference(
            self.query,
            self.represented_k,
            self.represented_v,
            self.prefixes if prefixes is None else prefixes,
            rank=self.cp_rank,
            cp_world=self.cp_world,
            sm_scale=self.sm_scale,
        )

    def check(self, expected_o, expected_lse, prefixes=None) -> None:
        atol, rtol = _KIND_TOLERANCE[self.kind]
        prefixes = self.prefixes if prefixes is None else prefixes
        empty = torch.tensor(
            [
                [
                    _local_visible_count(p, row, self.cp_rank, self.cp_world) == 0
                    for row in range(self.q_len)
                ]
                for p in prefixes
            ],
            dtype=torch.bool,
            device="cuda",
        )
        if empty.any():
            assert torch.count_nonzero(self.out[empty]) == 0
            assert torch.isneginf(self.lse[empty]).all()
        torch.testing.assert_close(self.out, expected_o, atol=atol, rtol=rtol)
        torch.testing.assert_close(self.lse, expected_lse, atol=atol, rtol=rtol)
        assert torch.count_nonzero(self.counter) == 0


# (kind, prefixes, q_len, cp_world, cp_rank): rows inside the balanced band on both arches,
# covering both packed instances, empty rows and the D256 two-row-tile requests.
_GPU_CASES = {
    "bf16_b8_s4096_q4_cp4_r0_n32": ("bf16_p16", [4096] * 8, 4, 4, 0),
    "bf16_b1_s16384_q8_cp4_r0_n64_longtile": ("bf16_p16", [16384], 8, 4, 0),
    "bf16_b4_ragged_q4_cp4_r1_empty_rows": (
        "bf16_p16",
        [5000, 20000, 0, 12345],
        4,
        4,
        1,
    ),
    "fp8_b8_s8192_q4_cp4_r0_n32": ("fp8_p64", [8192] * 8, 4, 4, 0),
    "fp8_b8_s4096_q8_cp4_r0_n64": ("fp8_p64", [4096] * 8, 8, 4, 0),
    "fp8_b4_ragged_q3_cp4_r3_empty_rows": ("fp8_p64", [30000, 8192, 0, 63], 3, 4, 3),
    "d256_b64_ctx32768_q4_cp4_r0_n64": ("fp8_p64_d256", [32764] * 64, 4, 4, 0),
    "d256_b128_ctx16384_q2_cp4_r0_n32": ("fp8_p64_d256", [16382] * 128, 2, 4, 0),
    "d256_b16_ragged_q6_cp4_r3_two_row_tiles": (
        "fp8_p64_d256",
        [60000, 0, 5000, 100, 32764, 8192, 1, 40000] * 2,
        6,
        4,
        3,
    ),  # fmt: skip
}


@pytest.mark.gpu
@pytest.mark.parametrize("case", list(_GPU_CASES), ids=list(_GPU_CASES))
def test_gpu_balanced_route_matches_static_route_and_reference(
    case, monkeypatch
) -> None:
    _require_blackwell_dcp()
    kind, prefixes, q_len, cp_world, cp_rank = _GPU_CASES[case]
    problem = _GpuCase(
        kind,
        prefixes=prefixes,
        q_len=q_len,
        cp_world=cp_world,
        cp_rank=cp_rank,
        seed=685_300 + len(case),
    )
    assert problem.band().route == "balanced", problem.band()
    expected_o, expected_lse = problem.reference()

    problem.run("static")
    torch.cuda.synchronize()
    problem.check(expected_o, expected_lse)
    static_o, static_lse = problem.out.clone(), problem.lse.clone()

    problem.out.fill_(float("nan"))
    problem.lse.fill_(float("nan"))
    problem.run("balanced")
    torch.cuda.synchronize()
    problem.check(expected_o, expected_lse)
    balanced_o, balanced_lse = problem.out.clone(), problem.lse.clone()
    atol, rtol = _KIND_TOLERANCE[kind]
    torch.testing.assert_close(balanced_o, static_o, atol=2 * atol, rtol=2 * rtol)
    torch.testing.assert_close(balanced_lse, static_lse, atol=2 * atol, rtol=2 * rtol)

    # The public entry point routes the row to the balanced program.  The programs
    # fold split tiles in arrival order, so two runs agree to rounding, not bitwise:
    # the route is observed on the launcher and the numbers on the tolerance.
    launched = []
    real_launch = cake_dcp._run_dcp_spec_balanced

    def observed_launch(**kwargs):
        launched.append(kwargs["kind"])
        return real_launch(**kwargs)

    monkeypatch.setattr(cake_dcp, "_run_dcp_spec_balanced", observed_launch)
    problem.out.fill_(float("nan"))
    problem.lse.fill_(float("nan"))
    problem.run_public()
    torch.cuda.synchronize()
    assert launched == [kind]
    problem.check(expected_o, expected_lse)
    torch.testing.assert_close(problem.out, balanced_o, atol=atol, rtol=rtol)
    torch.testing.assert_close(problem.lse, balanced_lse, atol=atol, rtol=rtol)


_GRAPH_CASES = {
    # (kind, captured prefixes, q_len, cp_world, cp_rank, two more prefix vectors; the cache is packed for the per-request maximum)
    "bf16_b4_q4_cp4_r1": (
        "bf16_p16",
        [5000, 20000, 0, 12345],
        4,
        4,
        1,
        [[3, 19999, 63, 4096], [20000, 0, 1, 7]],
    ),  # fmt: skip
    "fp8_b4_q3_cp4_r3": (
        "fp8_p64",
        [30000, 8192, 0, 63],
        3,
        4,
        3,
        [[29999, 0, 8192, 1], [1, 2, 3, 30000]],
    ),  # fmt: skip
    "d256_b16_q6_cp4_r3": (
        "fp8_p64_d256",
        [60000, 0, 5000, 100, 32764, 8192, 1, 40000] * 2,
        6,
        4,
        3,
        [
            [59999, 1, 0, 5000, 100, 32764, 8192, 40000] * 2,
            [0, 0, 60000, 3, 4, 5, 6, 7] * 2,
        ],
    ),  # fmt: skip
}


@pytest.mark.gpu
@pytest.mark.parametrize("case", list(_GRAPH_CASES), ids=list(_GRAPH_CASES))
def test_gpu_balanced_graph_replays_three_length_vectors(case) -> None:
    _require_blackwell_dcp()
    kind, prefixes, q_len, cp_world, cp_rank, more = _GRAPH_CASES[case]
    vectors = (prefixes, *more)
    assert all(len(vector) == len(prefixes) for vector in vectors)
    # One rank cache holds every vector: packed for the per-request maximum, so a
    # replay with a longer prefix in some request still reads pages the rank owns.
    capacity = [max(vector[i] for vector in vectors) for i in range(len(prefixes))]
    problem = _GpuCase(
        kind,
        prefixes=prefixes,
        q_len=q_len,
        cp_world=cp_world,
        cp_rank=cp_rank,
        seed=685_400 + len(case),
        capacity=capacity,
    )
    assert problem.band().route == "balanced", problem.band()
    # Prewarm the exact tensor/layout binding so its TMA descriptor slots exist before capture.
    problem.run("balanced")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        problem.run("balanced")
    for vector in vectors:
        assert all(0 <= p <= c for p, c in zip(vector, capacity, strict=True))
        problem.prefix_tensor.copy_(
            torch.tensor(vector, dtype=torch.int32, device="cuda")
        )
        problem.seq_lens.copy_(
            torch.tensor(
                [_local_len(p, q_len, cp_world, cp_rank) for p in vector],
                dtype=torch.int32,
                device="cuda",
            )
        )
        problem.out.fill_(float("nan"))
        problem.lse.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        expected_o, expected_lse = problem.reference(vector)
        problem.check(expected_o, expected_lse, vector)
