"""Aligned head-dim-64 single-query grouped paged decode program of the Cake FMHA
module (``backend="cake"``).

Host-only: the module's Python mirror of the source selector against the source
selector test's expectations (hard-coded here, per target), the pinned program
record against the generated loader, and the measured shape rows through
``select_cake_fmha_decode_route`` per target.
GPU: the measured shape rows through
``trtllm_batch_decode_with_kv_cache(backend="cake")`` against an FP32 reference at
atol = rtol = 1e-2 (the reference's largest ``|O|`` above 1e-2 on every row, so no
row passes vacuously), the route taken, CUDA-graph replay bitwise equal to the
eager result on the aligned program (also with shorter lengths under the captured
KV bound), the kernel test's group-8 cases on the aligned program and its group-4
case on the present route.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import flashinfer.cake_fmha as cake_api
from flashinfer.decode import trtllm_batch_decode_with_kv_cache

HEAD_DIM = 64
PAGE_SIZE = 16
# Query heads per KV head of the aligned program (one 8-row query tile per KV head).
GROUP = 8
ATOL = 1e-2
RTOL = 1e-2
# The reference's largest |O| must exceed this on every row: a peaked softmax.  A
# smaller reference would pass the tolerance check without exercising the kernel.
MIN_REF_ABSMAX = 1e-2
COMPONENT = "decode_aligned_bf16_hd64"
PRESENT = "decode_balanced_bf16_hd64"
PROGRAM_ID = "ALIGNED_HD64_BF16"
ARCHS = ("sm_100a", "sm_103a")
# The module's device target -> the program target the selector reads.
TARGET_ARCH = {"sm100a": "sm_100a", "sm103a": "sm_103a"}
# The binding's host parameters in the kernel's order (then the grid arguments).
HOST_PARAMS = (
    "Q",
    "K",
    "V",
    "O",
    "page_table",
    "seq_lens_kv",
    "max_pages_per_seq",
    "softmax_scale_log2",
    "num_q_heads",
    "num_kv_heads",
    "heads_per_kv",
    "batch_size",
    "skip_corr_threshold",
)

# The sixteen long-context lengths; a batch of requests cycles the list.
# fmt: off
LONG_CONTEXT_LENGTHS = [
    8193, 57345, 73729, 81921, 98305, 106497, 114689, 131073,
    139265, 147457, 163841, 180225, 196609, 212993, 229377, 237569,
]
# fmt: on


def _cycled(batch: int) -> list[int]:
    return [LONG_CONTEXT_LENGTHS[i % len(LONG_CONTEXT_LENGTHS)] for i in range(batch)]


def _random_lengths(batch: int, low: int, high: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(low, high + 1, (batch,), generator=generator).tolist()


# The measured shape rows: name -> (num_kv_heads, seq_lens); eight query heads per
# KV head, BF16, page 16, HND, one page table shared by K and V.
SHAPE_ROWS = {
    "agentx_b16_hkv1": (1, _cycled(16)),
    "agentx_b32_hkv1": (1, _cycled(32)),
    "agentx_b64_hkv1": (1, _cycled(64)),
    "agentx_b128_hkv1": (1, _cycled(128)),
    "agentx_b192_hkv1": (1, _cycled(192)),
    "agentx_b256_hkv1": (1, _cycled(256)),
    "random_128_128k_b256_hkv1": (1, _random_lengths(256, 128, 131072, 1)),
    "random_128_65k_b64_hkv8": (8, _random_lengths(64, 128, 65536, 2)),
    "uniform_b4_hkv8_s65536": (8, [65536] * 4),
    "uniform_b16_hkv8_s65536": (8, [65536] * 16),
    "uniform_b64_hkv8_s32768": (8, [32768] * 64),
    "uniform_b128_hkv8_s32768": (8, [32768] * 128),
    "uniform_b128_hkv8_s4096": (8, [4096] * 128),
}
# (batch, shortest, longest) of the random rows as measured: the generator above
# must reproduce them.
RANDOM_ROW_IDENTITY = {
    "random_128_128k_b256_hkv1": (256, 469, 130588),
    "random_128_65k_b64_hkv8": (64, 1485, 65369),
}

# The kernel test's cases: name -> (num_q_heads, num_kv_heads, seq_lens).  The
# first four are group-8 cases (a partial page, ragged lengths over the K / V and
# page-id stages, a single-token request); the last is the group-4 case, which
# the aligned program's selector never takes.
CASE_ROWS = {
    "case_b4_hq16_hkv2_s300": (16, 2, [300] * 4),
    "case_b3_hq32_hkv4_ragged": (32, 4, [129, 700, 1100]),
    "case_b2_hq8_hkv1_s2048": (8, 1, [2048] * 2),
    "case_b5_hq40_hkv5_ragged": (40, 5, [17, 4096, 513, 1, 8191]),
}
GROUP4_CASE = (8, 2, [1500] * 2)

# The probed grid of the selector's derivation (tiles = batch x KV heads: the
# KV-heads-8 batches 8..256 and the KV-heads-1 batches 128 / 256, over the KV
# lengths 2048..131072), with one value past each end of both axes.
TILES = [1, 8, 63, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 2049, 4096]
MAX_KV = [16, 2047, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 131073, 262144]

# --- BEGIN SEL_HD64_EXPECTATIONS (the source selector's derivation; pasted verbatim)
# (arch, tiles, max_kv, selected): every probed (tiles, max_kv) of the selector's
# derivation per target, every band corner and one value past each bound, every
# pinned point and its four neighbours.  True = the aligned program, False = the
# present route.
EDGES: list[tuple[str, int, int, bool]] = [
    ("sm_100a", 15, 237569, False),
    ("sm_100a", 16, 237568, False),
    ("sm_100a", 16, 237569, False),
    ("sm_100a", 16, 237570, False),
    ("sm_100a", 17, 237569, False),
    ("sm_100a", 31, 65536, False),
    ("sm_100a", 31, 237569, False),
    ("sm_100a", 32, 65535, False),
    ("sm_100a", 32, 65536, False),
    ("sm_100a", 32, 65537, False),
    ("sm_100a", 32, 237568, False),
    ("sm_100a", 32, 237569, False),
    ("sm_100a", 32, 237570, False),
    ("sm_100a", 33, 65536, False),
    ("sm_100a", 33, 237569, False),
    ("sm_100a", 63, 2048, False),
    ("sm_100a", 63, 237569, False),
    ("sm_100a", 64, 2047, False),
    ("sm_100a", 64, 2048, True),
    ("sm_100a", 64, 4096, True),
    ("sm_100a", 64, 4097, False),
    ("sm_100a", 64, 8192, False),
    ("sm_100a", 64, 16384, False),
    ("sm_100a", 64, 32768, False),
    ("sm_100a", 64, 65536, False),
    ("sm_100a", 64, 237568, False),
    ("sm_100a", 64, 237569, False),
    ("sm_100a", 64, 237570, False),
    ("sm_100a", 65, 2048, False),
    ("sm_100a", 65, 4096, False),
    ("sm_100a", 65, 237569, False),
    ("sm_100a", 127, 8192, False),
    ("sm_100a", 127, 65536, False),
    ("sm_100a", 127, 237569, False),
    ("sm_100a", 128, 2048, False),
    ("sm_100a", 128, 4096, False),
    ("sm_100a", 128, 8191, False),
    ("sm_100a", 128, 8192, True),
    ("sm_100a", 128, 16384, True),
    ("sm_100a", 128, 32768, True),
    ("sm_100a", 128, 65535, True),
    ("sm_100a", 128, 65536, True),
    ("sm_100a", 128, 65537, True),
    ("sm_100a", 128, 131072, True),
    ("sm_100a", 128, 131073, False),
    ("sm_100a", 128, 237568, False),
    ("sm_100a", 128, 237569, False),
    ("sm_100a", 128, 237570, False),
    ("sm_100a", 129, 8192, False),
    ("sm_100a", 129, 65536, False),
    ("sm_100a", 129, 131072, False),
    ("sm_100a", 129, 237569, False),
    ("sm_100a", 191, 237569, False),
    ("sm_100a", 192, 237568, False),
    ("sm_100a", 192, 237569, False),
    ("sm_100a", 192, 237570, False),
    ("sm_100a", 193, 237569, False),
    ("sm_100a", 255, 130588, False),
    ("sm_100a", 255, 237569, False),
    ("sm_100a", 256, 2048, False),
    ("sm_100a", 256, 4096, False),
    ("sm_100a", 256, 8192, False),
    ("sm_100a", 256, 16384, False),
    ("sm_100a", 256, 32768, False),
    ("sm_100a", 256, 65536, False),
    ("sm_100a", 256, 130587, False),
    ("sm_100a", 256, 130588, False),
    ("sm_100a", 256, 130589, False),
    ("sm_100a", 256, 131072, False),
    ("sm_100a", 256, 237568, False),
    ("sm_100a", 256, 237569, False),
    ("sm_100a", 256, 237570, False),
    ("sm_100a", 257, 130588, False),
    ("sm_100a", 257, 237569, False),
    ("sm_100a", 383, 8192, False),
    ("sm_100a", 384, 2048, False),
    ("sm_100a", 384, 4096, False),
    ("sm_100a", 384, 8191, False),
    ("sm_100a", 384, 8192, True),
    ("sm_100a", 384, 8193, True),
    ("sm_100a", 384, 16384, True),
    ("sm_100a", 384, 16385, True),
    ("sm_100a", 384, 32768, True),
    ("sm_100a", 384, 32769, True),
    ("sm_100a", 384, 65536, True),
    ("sm_100a", 384, 65537, False),
    ("sm_100a", 385, 8192, True),
    ("sm_100a", 385, 65536, False),
    ("sm_100a", 511, 32768, True),
    ("sm_100a", 511, 65369, False),
    ("sm_100a", 512, 2048, False),
    ("sm_100a", 512, 4096, False),
    ("sm_100a", 512, 8192, True),
    ("sm_100a", 512, 16384, True),
    ("sm_100a", 512, 32767, True),
    ("sm_100a", 512, 32768, True),
    ("sm_100a", 512, 32769, False),
    ("sm_100a", 512, 65368, False),
    ("sm_100a", 512, 65369, False),
    ("sm_100a", 512, 65370, False),
    ("sm_100a", 512, 65536, False),
    ("sm_100a", 513, 8192, True),
    ("sm_100a", 513, 32768, False),
    ("sm_100a", 513, 65369, False),
    ("sm_100a", 767, 4096, False),
    ("sm_100a", 768, 2048, False),
    ("sm_100a", 768, 4095, False),
    ("sm_100a", 768, 4096, True),
    ("sm_100a", 768, 8192, True),
    ("sm_100a", 768, 8193, True),
    ("sm_100a", 768, 16384, True),
    ("sm_100a", 768, 16385, False),
    ("sm_100a", 768, 32768, False),
    ("sm_100a", 768, 65536, False),
    ("sm_100a", 769, 4096, True),
    ("sm_100a", 769, 8192, True),
    ("sm_100a", 769, 16384, False),
    ("sm_100a", 1023, 4096, True),
    ("sm_100a", 1023, 32768, False),
    ("sm_100a", 1024, 2048, False),
    ("sm_100a", 1024, 4095, False),
    ("sm_100a", 1024, 4096, True),
    ("sm_100a", 1024, 4097, True),
    ("sm_100a", 1024, 8192, True),
    ("sm_100a", 1024, 16384, False),
    ("sm_100a", 1024, 32767, False),
    ("sm_100a", 1024, 32768, False),
    ("sm_100a", 1024, 32769, False),
    ("sm_100a", 1024, 65536, False),
    ("sm_100a", 1025, 4096, True),
    ("sm_100a", 1025, 32768, False),
    ("sm_100a", 1535, 2048, False),
    ("sm_100a", 1536, 2047, False),
    ("sm_100a", 1536, 2048, True),
    ("sm_100a", 1536, 4096, True),
    ("sm_100a", 1536, 8192, True),
    ("sm_100a", 1536, 16384, True),
    ("sm_100a", 1536, 16385, True),
    ("sm_100a", 1536, 32768, True),
    ("sm_100a", 1536, 65536, True),
    ("sm_100a", 1536, 65537, False),
    ("sm_100a", 1537, 2048, True),
    ("sm_100a", 1537, 65536, False),
    ("sm_100a", 2048, 2048, True),
    ("sm_100a", 2048, 4096, True),
    ("sm_100a", 2048, 8192, True),
    ("sm_100a", 2048, 8193, True),
    ("sm_100a", 2048, 16384, True),
    ("sm_100a", 2048, 16385, False),
    ("sm_100a", 2048, 32768, False),
    ("sm_100a", 2048, 65536, False),
    ("sm_100a", 2049, 2048, False),
    ("sm_100a", 2049, 4096, False),
    ("sm_100a", 2049, 8192, False),
    ("sm_100a", 2049, 16384, False),
    ("sm_103a", 15, 237569, False),
    ("sm_103a", 16, 237568, False),
    ("sm_103a", 16, 237569, False),
    ("sm_103a", 16, 237570, False),
    ("sm_103a", 17, 237569, False),
    ("sm_103a", 31, 65536, False),
    ("sm_103a", 31, 237569, False),
    ("sm_103a", 32, 65535, False),
    ("sm_103a", 32, 65536, False),
    ("sm_103a", 32, 65537, False),
    ("sm_103a", 32, 237568, False),
    ("sm_103a", 32, 237569, False),
    ("sm_103a", 32, 237570, False),
    ("sm_103a", 33, 65536, False),
    ("sm_103a", 33, 237569, False),
    ("sm_103a", 63, 2048, False),
    ("sm_103a", 63, 237569, False),
    ("sm_103a", 64, 2047, False),
    ("sm_103a", 64, 2048, True),
    ("sm_103a", 64, 2049, False),
    ("sm_103a", 64, 4096, False),
    ("sm_103a", 64, 8192, False),
    ("sm_103a", 64, 16384, False),
    ("sm_103a", 64, 32768, False),
    ("sm_103a", 64, 65536, False),
    ("sm_103a", 64, 237568, False),
    ("sm_103a", 64, 237569, False),
    ("sm_103a", 64, 237570, False),
    ("sm_103a", 65, 2048, False),
    ("sm_103a", 65, 237569, False),
    ("sm_103a", 127, 8192, False),
    ("sm_103a", 127, 65536, False),
    ("sm_103a", 127, 237569, False),
    ("sm_103a", 128, 2048, False),
    ("sm_103a", 128, 4096, False),
    ("sm_103a", 128, 8191, False),
    ("sm_103a", 128, 8192, True),
    ("sm_103a", 128, 8193, True),
    ("sm_103a", 128, 16384, True),
    ("sm_103a", 128, 32768, True),
    ("sm_103a", 128, 65535, True),
    ("sm_103a", 128, 65536, True),
    ("sm_103a", 128, 65537, True),
    ("sm_103a", 128, 131072, True),
    ("sm_103a", 128, 131073, False),
    ("sm_103a", 128, 237568, False),
    ("sm_103a", 128, 237569, False),
    ("sm_103a", 128, 237570, False),
    ("sm_103a", 129, 8192, True),
    ("sm_103a", 129, 65536, False),
    ("sm_103a", 129, 131072, False),
    ("sm_103a", 129, 237569, False),
    ("sm_103a", 191, 237569, False),
    ("sm_103a", 192, 237568, False),
    ("sm_103a", 192, 237569, False),
    ("sm_103a", 192, 237570, False),
    ("sm_103a", 193, 237569, False),
    ("sm_103a", 255, 130588, False),
    ("sm_103a", 255, 237569, False),
    ("sm_103a", 256, 2048, False),
    ("sm_103a", 256, 4096, False),
    ("sm_103a", 256, 8192, True),
    ("sm_103a", 256, 16384, False),
    ("sm_103a", 256, 32768, False),
    ("sm_103a", 256, 65536, False),
    ("sm_103a", 256, 130587, False),
    ("sm_103a", 256, 130588, False),
    ("sm_103a", 256, 130589, False),
    ("sm_103a", 256, 131072, False),
    ("sm_103a", 256, 237568, False),
    ("sm_103a", 256, 237569, False),
    ("sm_103a", 256, 237570, False),
    ("sm_103a", 257, 130588, False),
    ("sm_103a", 257, 237569, False),
    ("sm_103a", 383, 8192, True),
    ("sm_103a", 384, 2048, False),
    ("sm_103a", 384, 4096, False),
    ("sm_103a", 384, 8191, False),
    ("sm_103a", 384, 8192, True),
    ("sm_103a", 384, 16384, True),
    ("sm_103a", 384, 16385, True),
    ("sm_103a", 384, 32768, True),
    ("sm_103a", 384, 65536, True),
    ("sm_103a", 384, 65537, False),
    ("sm_103a", 385, 8192, True),
    ("sm_103a", 385, 65536, False),
    ("sm_103a", 511, 32768, False),
    ("sm_103a", 511, 65369, False),
    ("sm_103a", 512, 2048, False),
    ("sm_103a", 512, 4096, False),
    ("sm_103a", 512, 8192, True),
    ("sm_103a", 512, 16384, True),
    ("sm_103a", 512, 16385, False),
    ("sm_103a", 512, 32767, False),
    ("sm_103a", 512, 32768, False),
    ("sm_103a", 512, 32769, False),
    ("sm_103a", 512, 65368, False),
    ("sm_103a", 512, 65369, False),
    ("sm_103a", 512, 65370, False),
    ("sm_103a", 512, 65536, False),
    ("sm_103a", 513, 8192, True),
    ("sm_103a", 513, 16384, False),
    ("sm_103a", 513, 32768, False),
    ("sm_103a", 513, 65369, False),
    ("sm_103a", 767, 4096, False),
    ("sm_103a", 768, 2048, False),
    ("sm_103a", 768, 4095, False),
    ("sm_103a", 768, 4096, True),
    ("sm_103a", 768, 8192, True),
    ("sm_103a", 768, 8193, False),
    ("sm_103a", 768, 16384, False),
    ("sm_103a", 768, 32768, False),
    ("sm_103a", 768, 65536, False),
    ("sm_103a", 1023, 4096, True),
    ("sm_103a", 1023, 32768, False),
    ("sm_103a", 1024, 2048, False),
    ("sm_103a", 1024, 4095, False),
    ("sm_103a", 1024, 4096, True),
    ("sm_103a", 1024, 4097, True),
    ("sm_103a", 1024, 8192, True),
    ("sm_103a", 1024, 16384, True),
    ("sm_103a", 1024, 32767, True),
    ("sm_103a", 1024, 32768, True),
    ("sm_103a", 1024, 32769, True),
    ("sm_103a", 1024, 65536, True),
    ("sm_103a", 1024, 65537, False),
    ("sm_103a", 1025, 4096, True),
    ("sm_103a", 1025, 32768, True),
    ("sm_103a", 1535, 2048, False),
    ("sm_103a", 1536, 2047, False),
    ("sm_103a", 1536, 2048, True),
    ("sm_103a", 1536, 4096, True),
    ("sm_103a", 1536, 8192, True),
    ("sm_103a", 1536, 16384, True),
    ("sm_103a", 1536, 32768, True),
    ("sm_103a", 1536, 65536, True),
    ("sm_103a", 1536, 65537, False),
    ("sm_103a", 2048, 2048, True),
    ("sm_103a", 2048, 4096, True),
    ("sm_103a", 2048, 8192, True),
    ("sm_103a", 2048, 8193, True),
    ("sm_103a", 2048, 16384, True),
    ("sm_103a", 2048, 32768, True),
    ("sm_103a", 2048, 65536, True),
    ("sm_103a", 2048, 65537, False),
    ("sm_103a", 2049, 2048, False),
    ("sm_103a", 2049, 4096, False),
    ("sm_103a", 2049, 8192, False),
    ("sm_103a", 2049, 65536, False),
]
# row -> {arch: selected}: the served program of every shape row per target.
ROW_ROUTES: dict[str, dict[str, bool]] = {
    "agentx_b16_hkv1": {"sm_100a": False, "sm_103a": False},
    "agentx_b32_hkv1": {"sm_100a": False, "sm_103a": False},
    "agentx_b64_hkv1": {"sm_100a": False, "sm_103a": False},
    "agentx_b128_hkv1": {"sm_100a": False, "sm_103a": False},
    "agentx_b192_hkv1": {"sm_100a": False, "sm_103a": False},
    "agentx_b256_hkv1": {"sm_100a": False, "sm_103a": False},
    "random_128_128k_b256_hkv1": {"sm_100a": False, "sm_103a": False},
    "random_128_65k_b64_hkv8": {"sm_100a": False, "sm_103a": False},
    "uniform_b4_hkv8_s65536": {"sm_100a": False, "sm_103a": False},
    "uniform_b16_hkv8_s65536": {"sm_100a": True, "sm_103a": True},
    "uniform_b64_hkv8_s32768": {"sm_100a": True, "sm_103a": False},
    "uniform_b128_hkv8_s32768": {"sm_100a": False, "sm_103a": True},
    "uniform_b128_hkv8_s4096": {"sm_100a": True, "sm_103a": True},
}
# --- END SEL_HD64_EXPECTATIONS ---


def _require_table() -> None:
    """The expectation table must be filled for every shape row and target: an
    empty table fails every dependent item instead of passing vacuously."""

    filled = (
        bool(EDGES)
        and set(ROW_ROUTES) == set(SHAPE_ROWS)
        and all(set(routes) == set(ARCHS) for routes in ROW_ROUTES.values())
        and all(
            isinstance(selected, bool)
            for routes in ROW_ROUTES.values()
            for selected in routes.values()
        )
        and all(
            arch in ARCHS and isinstance(selected, bool)
            for arch, _tiles, _max_kv, selected in EDGES
        )
    )
    if not filled:
        pytest.fail(
            "the selector expectation table (EDGES, ROW_ROUTES) is not filled for "
            "every shape row and target"
        )


def test_aligned_hd64_expectation_table_is_filled() -> None:
    _require_table()


@pytest.mark.parametrize("row", sorted(RANDOM_ROW_IDENTITY))
def test_random_shape_rows_reproduce_the_measured_lengths(row) -> None:
    lengths = SHAPE_ROWS[row][1]
    assert (len(lengths), min(lengths), max(lengths)) == RANDOM_ROW_IDENTITY[row]


def test_case_rows_spell_the_kernel_test_groups() -> None:
    for num_q_heads, num_kv_heads, seq_lens in CASE_ROWS.values():
        assert num_q_heads == GROUP * num_kv_heads
        assert len(seq_lens) >= 2
    num_q_heads, num_kv_heads, _ = GROUP4_CASE
    assert num_q_heads == 4 * num_kv_heads


# --- The selector mirror against the source selector test ------------------


def _select(tiles, max_kv, arch, **overrides):
    kwargs = dict(q_len=1, num_q_heads=GROUP, arch=arch)
    kwargs.update(overrides)
    return cake_api.aligned_hd64_selected(tiles, 1, max_kv, **kwargs)


@pytest.mark.parametrize("arch", ARCHS)
def test_aligned_hd64_selector_gives_one_verdict_on_the_probed_grid(arch) -> None:
    # One bool for every probed (tiles, max_kv) and one value past each end.
    for tiles in TILES:
        for max_kv in MAX_KV:
            assert isinstance(_select(tiles, max_kv, arch), bool), (tiles, max_kv)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(q_len=2),
        dict(num_q_heads=GROUP + 1),
        dict(num_q_heads=GROUP // 2),
        dict(num_q_heads=2 * GROUP),
        dict(arch="sm_90a"),
    ],
)
def test_aligned_hd64_selector_other_shapes_keep_the_present_route(kwargs) -> None:
    overrides = {"arch": "sm_100a", **kwargs}
    for tiles in TILES:
        for max_kv in MAX_KV:
            assert _select(tiles, max_kv, **overrides) is False, (tiles, max_kv)


@pytest.mark.parametrize("case", EDGES, ids=lambda c: f"{c[0]}-t{c[1]}-kv{c[2]}")
def test_aligned_hd64_selector_edges(case) -> None:
    arch, tiles, max_kv, selected = case
    assert _select(tiles, max_kv, arch) is selected


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("row", sorted(SHAPE_ROWS))
def test_aligned_hd64_selector_shape_rows_route_as_derived(row, arch) -> None:
    _require_table()
    num_kv_heads, seq_lens = SHAPE_ROWS[row]
    got = cake_api.aligned_hd64_selected(
        len(seq_lens),
        num_kv_heads,
        max(seq_lens),
        q_len=1,
        num_q_heads=GROUP * num_kv_heads,
        arch=arch,
    )
    assert got is ROW_ROUTES[row][arch]


# --- The pinned program record ----------------------------------------------


def test_cake_fmha_aligned_hd64_program_table_and_launch_grid() -> None:
    """The identifier table pins one record; the launch grid (the parity lane's
    host grid) is one CTA per (request, KV head): ``(1, KV heads, requests)``."""

    assert cake_api._ALIGNED_HD64_COMPONENT == COMPONENT
    assert cake_api._ALIGNED_HD64_PROGRAM_ID == PROGRAM_ID
    table = cake_api._ALIGNED_HD64_PROGRAMS
    assert list(table) == [PROGRAM_ID]
    name = table[PROGRAM_ID]
    assert name.startswith("cake_fmha_aligned_hd64_")
    digest = name.removeprefix("cake_fmha_aligned_hd64_")
    assert len(digest) == 20 and set(digest) <= set("0123456789abcdef")
    assert tuple(cake_api._ALIGNED_HD64_HOST_PARAMS) == HOST_PARAMS
    record = SimpleNamespace(cluster=(1, 1, 1))
    assert cake_api.cake_fmha_aligned_hd64_launch_grid(record, 3, 8) == (1, 8, 3)
    assert cake_api.cake_fmha_aligned_hd64_launch_grid(record, 256, 1) == (1, 1, 256)


def test_cake_fmha_aligned_hd64_program_table_matches_the_generated_loader() -> None:
    """The pinned record names the generated loader's program of the same
    identifier (the load-time checks, host-only): name, cluster, binding
    argument order and both targets; the loader publishes exactly that program."""

    from flashinfer.jit import cake_fmha_aligned_hd64 as loader
    from flashinfer.jit.cake_program import GRID_ARGS, CakeProgram

    name = cake_api._ALIGNED_HD64_PROGRAMS[PROGRAM_ID]
    program = getattr(loader, PROGRAM_ID)
    assert isinstance(program, CakeProgram)
    assert program.name == name
    assert tuple(program.cluster) == (1, 1, 1)
    assert tuple(program.block) == (512, 1, 1)
    assert tuple(program.args) == (*cake_api._ALIGNED_HD64_HOST_PARAMS, *GRID_ARGS)
    assert tuple(program.tensors) == ("Q", "K", "V", "O", "page_table", "seq_lens_kv")
    assert {"sm_100a", "sm_103a"} <= set(program.targets)
    assert program.use_pdl is False and program.cooperative is False
    assert program.dynamic_smem_bytes == 156544
    assert [p.name for p in loader.PROGRAMS] == [name]


# --- The shape rows through the module's route selection (host-only) --------


def _decode_kwargs(
    *, batch_size, num_kv_heads, seq_lens, num_q_heads=None, stacked_kv=False
):
    """Host-side ``select_cake_fmha_decode_route`` inputs of one shape row.

    A four-page pool is enough: the selector reads shapes, strides and dtypes.
    ``stacked_kv`` hands the K / V views of one ``[pages, 2, ...]`` tensor, the
    layout the aligned program cannot address.
    """

    sm_count = 148
    num_q_heads = GROUP * num_kv_heads if num_q_heads is None else num_q_heads
    dtype = torch.bfloat16
    query = torch.empty((batch_size, num_q_heads, HEAD_DIM), dtype=dtype)
    if stacked_kv:
        key, value = torch.empty(
            (4, 2, num_kv_heads, PAGE_SIZE, HEAD_DIM), dtype=dtype
        ).unbind(1)
    else:
        key = torch.empty((4, num_kv_heads, PAGE_SIZE, HEAD_DIM), dtype=dtype)
        value = torch.empty_like(key)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    max_seq_len = int(seq_lens_t.max())
    return dict(
        query=query,
        key_cache=key,
        value_cache=value,
        out=torch.empty_like(query),
        workspace_buffer=torch.empty(
            cake_api.cake_fmha_balanced_workspace_bytes(sm_count, 1, head_dim=HEAD_DIM),
            dtype=torch.uint8,
        ),
        block_tables=torch.zeros(
            (batch_size, (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE), dtype=torch.int32
        ),
        seq_lens=seq_lens_t,
        batch_size=batch_size,
        q_len=1,
        max_seq_len=max_seq_len,
        window_left=-1,
        bmm1_scale=HEAD_DIM**-0.5,
        bmm2_scale=1.0,
        o_scale=1.0,
        sinks=None,
        kv_layout="HND",
        uses_shared_paged_kv_idx=True,
        cum_seq_lens_q=None,
        key_block_scales=None,
        value_block_scales=None,
        skip_softmax_threshold_scale_factor=None,
        enable_block_sparse_attention=False,
        multi_ctas_kv_counter_buffer=torch.zeros(
            cake_api.cake_fmha_balanced_counter_bytes(sm_count, 1, head_dim=HEAD_DIM),
            dtype=torch.uint8,
        ),
    )


def _select_route(kwargs):
    return cake_api.select_cake_fmha_decode_route(kwargs["query"].device, **kwargs)


def _patch_host_only(monkeypatch, target):
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: target)
    monkeypatch.setattr(cake_api, "_smallm_sm_count", lambda device: 148)
    monkeypatch.setattr(
        cake_api,
        "_is_cake_fmha_decode_native_bf16_available",
        lambda *args, **kwargs: True,
    )


@pytest.mark.parametrize("target", sorted(TARGET_ARCH))
@pytest.mark.parametrize("row", sorted(SHAPE_ROWS))
def test_cake_fmha_aligned_hd64_route_serves_shape_rows_as_derived(
    monkeypatch, target, row
) -> None:
    _require_table()
    _patch_host_only(monkeypatch, target)
    num_kv_heads, seq_lens = SHAPE_ROWS[row]
    batch_size = len(seq_lens)
    kwargs = _decode_kwargs(
        batch_size=batch_size, num_kv_heads=num_kv_heads, seq_lens=seq_lens
    )
    route = _select_route(kwargs)
    assert route is not None
    if not ROW_ROUTES[row][TARGET_ARCH[target]]:
        # The module's present head-dim-64 route, unchanged.
        assert route.component == PRESENT
        assert route.aligned_program is None
        return
    assert route == cake_api.CakeFmhaDecodeRoute(
        target=target,
        batch_size=batch_size,
        q_len=1,
        num_q_heads=GROUP * num_kv_heads,
        num_kv_heads=num_kv_heads,
        has_sink=False,
        has_window=False,
        use_scale_ptr=False,
        retain_kv_l2=(max(seq_lens) + 127) // 128 <= 9,
        component=COMPONENT,
        page_size=PAGE_SIZE,
        aligned_program=PROGRAM_ID,
    )
    assert cake_api.cake_fmha_route_is_optimized(route)
    assert cake_api._route_components(route) == (PROGRAM_ID,)


@pytest.mark.parametrize("target", sorted(TARGET_ARCH))
def test_cake_fmha_aligned_hd64_group4_case_keeps_the_present_route(
    monkeypatch, target
) -> None:
    _patch_host_only(monkeypatch, target)
    num_q_heads, num_kv_heads, seq_lens = GROUP4_CASE
    kwargs = _decode_kwargs(
        batch_size=len(seq_lens),
        num_kv_heads=num_kv_heads,
        seq_lens=seq_lens,
        num_q_heads=num_q_heads,
    )
    route = _select_route(kwargs)
    assert route is not None
    assert route.component == PRESENT
    assert route.aligned_program is None


def test_cake_fmha_aligned_hd64_route_fails_closed_outside_its_contract(
    monkeypatch,
) -> None:
    _require_table()
    _patch_host_only(monkeypatch, "sm100a")
    rows = [row for row in sorted(SHAPE_ROWS) if ROW_ROUTES[row]["sm_100a"]]
    if not rows:
        pytest.fail("no shape row routes to the aligned program on sm_100a")
    num_kv_heads, seq_lens = SHAPE_ROWS[rows[0]]
    batch_size = len(seq_lens)
    num_q_heads = GROUP * num_kv_heads
    dtype = torch.bfloat16
    kwargs = _decode_kwargs(
        batch_size=batch_size, num_kv_heads=num_kv_heads, seq_lens=seq_lens
    )
    route = _select_route(kwargs)
    assert route is not None and route.component == COMPONENT

    stacked = _decode_kwargs(
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        seq_lens=seq_lens,
        stacked_kv=True,
    )
    assert not stacked["key_cache"].is_contiguous()
    pages = kwargs["block_tables"].shape[1]
    for mutation in (
        {"sinks": torch.zeros(num_q_heads, dtype=torch.float32)},
        {"window_left": 128},
        {"bmm1_scale": torch.ones(1, dtype=torch.float32)},
        {"bmm2_scale": 0.5},
        {"bmm2_scale": torch.ones(1, dtype=torch.float32)},
        {"o_scale": 0.5},
        {"kv_layout": "NHD"},
        {"lse": torch.empty((batch_size, num_q_heads), dtype=torch.float32)},
        {"block_tables": kwargs["block_tables"].to(torch.uint32)},
        {"seq_lens": kwargs["seq_lens"].to(torch.uint32)},
        {"key_cache": stacked["key_cache"], "value_cache": stacked["value_cache"]},
        {"value_cache": stacked["value_cache"]},
        {
            "query": torch.empty((batch_size, num_q_heads, 2 * HEAD_DIM), dtype=dtype)[
                ..., :HEAD_DIM
            ]
        },
        {
            "key_cache": torch.empty((4, num_kv_heads, 32, HEAD_DIM), dtype=dtype),
            "value_cache": torch.empty((4, num_kv_heads, 32, HEAD_DIM), dtype=dtype),
        },
        {
            "uses_shared_paged_kv_idx": False,
            "block_tables": torch.zeros((batch_size, 2, pages), dtype=torch.int32),
        },
        {
            "key_block_scales": torch.ones(
                (4, num_kv_heads, PAGE_SIZE), dtype=torch.float32
            )
        },
        {
            "q_len": 2,
            "query": torch.empty((2 * batch_size, num_q_heads, HEAD_DIM), dtype=dtype),
            "out": torch.empty((2 * batch_size, num_q_heads, HEAD_DIM), dtype=dtype),
        },
        {
            "query": torch.empty((batch_size, num_q_heads // 2, HEAD_DIM), dtype=dtype),
            "out": torch.empty((batch_size, num_q_heads // 2, HEAD_DIM), dtype=dtype),
        },
        {
            "query": torch.empty((batch_size, 2 * num_q_heads, HEAD_DIM), dtype=dtype),
            "out": torch.empty((batch_size, 2 * num_q_heads, HEAD_DIM), dtype=dtype),
        },
    ):
        mutated = {**kwargs, **mutation}
        selected = _select_route(mutated)
        assert selected is None or selected.component != COMPONENT, mutation
        if selected is not None:
            assert selected.aligned_program is None, mutation


# --- GPU: the shape rows through backend="cake" -----------------------------

_requires_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Cake FMHA requires SM100 or SM103",
)


def _hd64_case(device, *, num_q_heads, num_kv_heads, seq_lens, seed):
    """Query ``[batch, num_q_heads, 64]`` (``randn x 3``: a peaked softmax, so
    ``|O|`` sits well above the tolerance), separate contiguous K / V pools
    ``[pages, num_kv_heads, 16, 64]`` (``randn``; the layout the aligned program
    reads in place), a shuffled page table and the device lengths."""

    generator = torch.Generator(device=device).manual_seed(seed)
    batch = len(seq_lens)
    pages_per_req = [(s + PAGE_SIZE - 1) // PAGE_SIZE for s in seq_lens]
    total_pages = sum(pages_per_req)
    query = (
        torch.randn((batch, num_q_heads, HEAD_DIM), device=device, generator=generator)
        * 3
    ).to(torch.bfloat16)
    pool_shape = (total_pages, num_kv_heads, PAGE_SIZE, HEAD_DIM)
    key = torch.randn(pool_shape, device=device, generator=generator).to(torch.bfloat16)
    value = torch.randn(pool_shape, device=device, generator=generator).to(
        torch.bfloat16
    )
    order = torch.randperm(total_pages, device=device, generator=generator)
    block_tables = torch.zeros(
        (batch, max(pages_per_req)), dtype=torch.int32, device=device
    )
    cursor = 0
    for b, n in enumerate(pages_per_req):
        block_tables[b, :n] = order[cursor : cursor + n]
        cursor += n
    return (
        query,
        key,
        value,
        block_tables,
        torch.tensor(seq_lens, dtype=torch.int32, device=device),
    )


def _reference_decode(query, key, value, block_tables, seq_lens, *, scale):
    """FP32 GQA decode over the paged HND pools: request ``b`` attends its first
    ``seq_lens[b]`` tokens, query head ``h`` reads KV head ``h // group``."""

    batch = int(seq_lens.numel())
    num_q_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    group = num_q_heads // num_kv_heads
    out = torch.empty(query.shape, dtype=torch.float32, device=query.device)
    for b in range(batch):
        length = int(seq_lens[b])
        pages = block_tables[b, : (length + PAGE_SIZE - 1) // PAGE_SIZE]
        k = key[pages].permute(1, 0, 2, 3).reshape(num_kv_heads, -1, HEAD_DIM)
        v = value[pages].permute(1, 0, 2, 3).reshape(num_kv_heads, -1, HEAD_DIM)
        k = k[:, :length].float()
        v = v[:, :length].float()
        q = query[b].float().view(num_kv_heads, group, HEAD_DIM)
        logits = torch.einsum("hgd,hnd->hgn", q, k) * scale
        probs = torch.softmax(logits, dim=-1)
        out[b] = torch.einsum("hgn,hnd->hgd", probs, v).reshape(num_q_heads, HEAD_DIM)
    return out


def _shorter_lengths(seq_lens):
    """A second length vector under the same captured KV bound: the longest
    request keeps its length, every other request loses a third."""

    shorter = [max(1, s - s // 3) for s in seq_lens]
    longest = max(range(len(seq_lens)), key=seq_lens.__getitem__)
    shorter[longest] = seq_lens[longest]
    return shorter


def _check_row(
    monkeypatch,
    *,
    query,
    key,
    value,
    block_tables,
    seq_lens_t,
    seq_lens,
    expect_component,
):
    """One shape row through ``backend="cake"``: the route taken, the FP32
    reference at the tolerance with a non-vacuous reference, and CUDA-graph
    replay -- bitwise equal to the eager result on the aligned program (the
    present route is held to the tolerance), then under shorter lengths."""

    device = query.device
    scale = HEAD_DIM**-0.5
    max_seq_len = max(seq_lens)
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    bitwise = expect_component == COMPONENT

    # Record the route the dispatch resolves (decode.py imports the resolver
    # from the module at call time).
    resolved = []
    resolve = cake_api._resolve_cake_fmha_decode_module

    def spy(resolve_device, route):
        resolved.append(route)
        return resolve(resolve_device, route)

    monkeypatch.setattr(cake_api, "_resolve_cake_fmha_decode_module", spy)

    def run():
        return trtllm_batch_decode_with_kv_cache(
            query,
            (key, value),
            workspace,
            block_tables,
            seq_lens_t,
            max_seq_len,
            bmm1_scale=scale,
            bmm2_scale=1.0,
            window_left=-1,
            kv_layout="HND",
            q_len_per_req=1,
            backend="cake",
        )

    actual = run()
    torch.cuda.synchronize()
    route = resolved[-1]
    assert route is not None and route.component == expect_component
    if bitwise:
        assert route.aligned_program == PROGRAM_ID
    else:
        assert route.aligned_program is None
    expected = _reference_decode(
        query, key, value, block_tables, seq_lens_t, scale=scale
    )
    assert float(expected.abs().amax()) > MIN_REF_ABSMAX
    torch.testing.assert_close(actual.float(), expected, atol=ATOL, rtol=RTOL)

    def check_replay(replayed, reference):
        if bitwise:
            assert torch.equal(replayed, actual)
        torch.testing.assert_close(replayed.float(), reference, atol=ATOL, rtol=RTOL)

    # Graph replay: the captured launch must reproduce the eager result and,
    # with shorter lengths under the same KV bound, follow the device lengths
    # without a re-plan.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            replayed = run()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    check_replay(replayed, expected)
    shorter = _shorter_lengths(seq_lens)
    seq_lens_t.copy_(torch.tensor(shorter, dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()
    expected_short = _reference_decode(
        query, key, value, block_tables, seq_lens_t, scale=scale
    )
    assert float(expected_short.abs().amax()) > MIN_REF_ABSMAX
    torch.testing.assert_close(replayed.float(), expected_short, atol=ATOL, rtol=RTOL)
    seq_lens_t.copy_(torch.tensor(seq_lens, dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()
    check_replay(replayed, expected)


@_requires_blackwell
@pytest.mark.parametrize("row", sorted(SHAPE_ROWS))
def test_cake_decode_aligned_hd64_shape_rows_match_reference_and_replay(
    monkeypatch, row
) -> None:
    _require_table()
    device = torch.device("cuda")
    arch = TARGET_ARCH[cake_api._cake_fmha_target(device)]
    num_kv_heads, seq_lens = SHAPE_ROWS[row]
    query, key, value, block_tables, seq_lens_t = _hd64_case(
        device,
        num_q_heads=GROUP * num_kv_heads,
        num_kv_heads=num_kv_heads,
        seq_lens=seq_lens,
        seed=20261024 + len(seq_lens) + num_kv_heads,
    )
    _check_row(
        monkeypatch,
        query=query,
        key=key,
        value=value,
        block_tables=block_tables,
        seq_lens_t=seq_lens_t,
        seq_lens=seq_lens,
        expect_component=COMPONENT if ROW_ROUTES[row][arch] else PRESENT,
    )


@_requires_blackwell
@pytest.mark.parametrize("case", sorted(CASE_ROWS))
def test_cake_decode_aligned_hd64_kernel_cases_match_reference_on_the_program(
    monkeypatch, case
) -> None:
    """The kernel test's group-8 cases on the aligned program itself: the
    selector is held to the aligned program so the small shapes (a partial
    page, ragged lengths, a single-token request) exercise it."""

    device = torch.device("cuda")
    num_q_heads, num_kv_heads, seq_lens = CASE_ROWS[case]
    monkeypatch.setattr(cake_api, "aligned_hd64_selected", lambda *a, **k: True)
    query, key, value, block_tables, seq_lens_t = _hd64_case(
        device,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        seq_lens=seq_lens,
        seed=20261024 + len(seq_lens) + num_kv_heads,
    )
    _check_row(
        monkeypatch,
        query=query,
        key=key,
        value=value,
        block_tables=block_tables,
        seq_lens_t=seq_lens_t,
        seq_lens=seq_lens,
        expect_component=COMPONENT,
    )


@_requires_blackwell
def test_cake_decode_aligned_hd64_group4_case_runs_on_the_present_route(
    monkeypatch,
) -> None:
    device = torch.device("cuda")
    num_q_heads, num_kv_heads, seq_lens = GROUP4_CASE
    query, key, value, block_tables, seq_lens_t = _hd64_case(
        device,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        seq_lens=seq_lens,
        seed=20261024 + len(seq_lens) + num_kv_heads,
    )
    _check_row(
        monkeypatch,
        query=query,
        key=key,
        value=value,
        block_tables=block_tables,
        seq_lens_t=seq_lens_t,
        seq_lens=seq_lens,
        expect_component=PRESENT,
    )
