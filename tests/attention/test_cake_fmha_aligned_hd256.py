"""Aligned head-dim-256 single-query grouped paged decode programs of the Cake
FMHA module (``backend="cake"``): the persistent program and the cluster
programs of 2, 4 and 8 CTAs.

Host-only: the module's Python mirror of the source selector against the source
selector test's expectations (hard-coded here, per target, at each target's SM
count of record, plus the trtllm-gen split at other SM counts), the pinned
program records against the generated loader, the load-time checks against a
stand-in loader (each record that differs from the pinned launch fails closed),
and the measured shape rows through ``select_cake_fmha_decode_route`` per target
and SM count.
GPU: the measured shape rows through
``trtllm_batch_decode_with_kv_cache(backend="cake")`` against an FP32 reference at
atol = rtol = 1e-2 (the reference's largest ``|O|`` above 1e-2 on every row, so no
row passes vacuously), the route taken, CUDA-graph replay bitwise equal to the
eager result on an aligned program (also with shorter lengths under the captured
KV bound), the kernel tests' group-8 cases on each aligned program, the
group-4 case on the present route, and the row pinned to the cluster program of
8 CTAs bitwise equal to ``backend="trtllm-gen"`` and within the tolerance of an
FP64 reference.
"""

from __future__ import annotations

import dataclasses
import sys
import types
from types import SimpleNamespace

import pytest
import torch

import flashinfer.cake_fmha as cake_api
import flashinfer.jit
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from flashinfer.jit.cake_program import GRID_ARGS, CakeProgram

HEAD_DIM = 256
PAGE_SIZE = 16
# Query heads per KV head of the aligned programs (one 8-row query tile per KV head).
GROUP = 8
ATOL = 1e-2
RTOL = 1e-2
# The reference's largest |O| must exceed this on every row: a peaked softmax.  A
# smaller reference would pass the tolerance check without exercising the kernel.
MIN_REF_ABSMAX = 1e-2
COMPONENT = "decode_aligned_bf16_hd256"
PRESENT = "decode_balanced_bf16_hd256"
# The selector's verdicts -> the program identifiers of the generated loader.
PROGRAMS = {
    ("persistent", 1): "ALIGNED_HD256_BF16",
    ("cga", 2): "ALIGNED_HD256_CGA2_BF16",
    ("cga", 4): "ALIGNED_HD256_CGA4_BF16",
    ("cga", 8): "ALIGNED_HD256_CGA8_BF16",
}
# Dynamic shared memory of each program's launch (the kernels' computed sizes).
PROGRAM_SMEM = {
    "ALIGNED_HD256_BF16": 147328,
    "ALIGNED_HD256_CGA2_BF16": 178560,
    "ALIGNED_HD256_CGA4_BF16": 178560,
    "ALIGNED_HD256_CGA8_BF16": 178560,
}
ARCHS = ("sm_100a", "sm_103a")
# The module's device target -> the program target the selector reads.
TARGET_ARCH = {"sm100a": "sm_100a", "sm103a": "sm_103a"}
# The bindings' host parameters in the kernels' order (then the grid arguments).
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
    # The product row the selector pins to the cluster program of 8 CTAs.
    "uniform_b1_hkv8_s4096": (8, [4096]),
}
# (batch, shortest, longest) of the random rows as measured: the generator above
# must reproduce them.
RANDOM_ROW_IDENTITY = {
    "random_128_128k_b256_hkv1": (256, 469, 130588),
    "random_128_65k_b64_hkv8": (64, 1485, 65369),
}

# The kernel tests' group-8 cases: name -> (num_q_heads, num_kv_heads, seq_lens): a
# partial page, ragged lengths over the K / V and page-id stages and over the
# cluster ranks, a single-token request, requests shorter than a cluster's rank
# count, a rank past the end, one to seven active ranks and empty last ranks of
# a cluster of 8, and the long uniform row.  Each case runs on every aligned
# program.  The group-4 case is the one the selector never takes.
CASE_ROWS = {
    "case_b4_hq16_hkv2_s300": (16, 2, [300] * 4),
    "case_b3_hq32_hkv4_ragged": (32, 4, [129, 700, 1100]),
    "case_b2_hq8_hkv1_s2048": (8, 1, [2048] * 2),
    "case_b5_hq40_hkv5_ragged": (40, 5, [17, 4096, 513, 1, 8191]),
    "case_b4_hq8_hkv1_short": (8, 1, [1, 200, 256, 257]),
    "case_b2_hq8_hkv1_s1025_s1537": (8, 1, [1025, 1537]),
    "case_b7_hq64_hkv8_ragged": (64, 8, [1, 300, 600, 1000, 1025, 1400, 1700]),
    "case_b4_hq8_hkv1_s2049_s4097": (8, 1, [2049, 3584, 3585, 4097]),
    "case_b4_hq64_hkv8_s65536": (64, 8, [65536] * 4),
}
GROUP4_CASE = (8, 2, [1500] * 2)

# --- BEGIN SEL_HD256_EXPECTATIONS (the source selector test's tables; pasted verbatim)
# Tiles (batch x KV heads) and max-KV values the selector is swept over: every
# probed value, every pinned value and every band bound, each with its
# neighbours, plus 1 and twice the largest.
# fmt: off
TILES: list[int] = [
    1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255,
    256, 257, 383, 384, 385, 511, 512, 513, 767, 768, 769, 1023, 1024, 1025, 1535, 1536,
    1537, 2047, 2048, 2049, 4096,
]
# fmt: on
# fmt: off
MAX_KV: list[int] = [
    1, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 8193, 16383, 16384, 16385, 32767,
    32768, 32769, 65368, 65369, 65370, 65535, 65536, 65537, 130587, 130588, 130589,
    131071, 131072, 131073, 237568, 237569, 237570, 475138,
]
# fmt: on
# The SM count of each target's records of derivation (the trtllm-gen split
# reads it) and the other SM counts the split tests sweep.
SM_COUNT: dict[str, int] = {"sm_100a": 148, "sm_103a": 152}
OTHER_SM_COUNTS: list[int] = [1, 2, 64, 100, 132, 148, 152, 160, 296]
# (arch, tiles, max_kv, program) at the target's SM count of record: every
# probed (tiles, max_kv) of the selector's derivation per target.
PROBED: list[tuple[str, int, int, tuple[str, int] | None]] = [
    ("sm_100a", 64, 2048, ("cga", 2)),
    ("sm_100a", 64, 4096, ("cga", 2)),
    ("sm_100a", 64, 8192, ("cga", 2)),
    ("sm_100a", 64, 16384, ("cga", 2)),
    ("sm_100a", 64, 32768, ("cga", 2)),
    ("sm_100a", 64, 65536, ("cga", 2)),
    ("sm_100a", 128, 2048, ("persistent", 1)),
    ("sm_100a", 128, 4096, ("persistent", 1)),
    ("sm_100a", 128, 8192, ("persistent", 1)),
    ("sm_100a", 128, 16384, ("persistent", 1)),
    ("sm_100a", 128, 32768, ("persistent", 1)),
    ("sm_100a", 128, 65536, ("persistent", 1)),
    ("sm_100a", 128, 131072, ("persistent", 1)),
    ("sm_100a", 256, 2048, ("persistent", 1)),
    ("sm_100a", 256, 4096, ("persistent", 1)),
    ("sm_100a", 256, 8192, ("persistent", 1)),
    ("sm_100a", 256, 16384, ("persistent", 1)),
    ("sm_100a", 256, 32768, ("persistent", 1)),
    ("sm_100a", 256, 65536, ("persistent", 1)),
    ("sm_100a", 256, 131072, ("persistent", 1)),
    ("sm_100a", 384, 2048, ("persistent", 1)),
    ("sm_100a", 384, 4096, ("persistent", 1)),
    ("sm_100a", 384, 8192, ("persistent", 1)),
    ("sm_100a", 384, 16384, ("persistent", 1)),
    ("sm_100a", 384, 32768, ("persistent", 1)),
    ("sm_100a", 384, 65536, ("persistent", 1)),
    ("sm_100a", 512, 2048, ("persistent", 1)),
    ("sm_100a", 512, 4096, ("persistent", 1)),
    ("sm_100a", 512, 8192, ("persistent", 1)),
    ("sm_100a", 512, 16384, None),
    ("sm_100a", 512, 32768, ("persistent", 1)),
    ("sm_100a", 512, 65536, None),
    ("sm_100a", 768, 2048, ("persistent", 1)),
    ("sm_100a", 768, 4096, ("persistent", 1)),
    ("sm_100a", 768, 8192, ("persistent", 1)),
    ("sm_100a", 768, 16384, None),
    ("sm_100a", 768, 32768, None),
    ("sm_100a", 768, 65536, None),
    ("sm_100a", 1024, 2048, ("persistent", 1)),
    ("sm_100a", 1024, 4096, ("persistent", 1)),
    ("sm_100a", 1024, 8192, ("persistent", 1)),
    ("sm_100a", 1024, 16384, ("persistent", 1)),
    ("sm_100a", 1024, 32768, ("persistent", 1)),
    ("sm_100a", 1024, 65536, ("persistent", 1)),
    ("sm_100a", 1536, 2048, ("persistent", 1)),
    ("sm_100a", 1536, 4096, ("persistent", 1)),
    ("sm_100a", 1536, 8192, ("persistent", 1)),
    ("sm_100a", 1536, 16384, ("persistent", 1)),
    ("sm_100a", 1536, 32768, ("persistent", 1)),
    ("sm_100a", 1536, 65536, ("persistent", 1)),
    ("sm_100a", 2048, 2048, ("persistent", 1)),
    ("sm_100a", 2048, 4096, ("persistent", 1)),
    ("sm_100a", 2048, 8192, ("persistent", 1)),
    ("sm_100a", 2048, 16384, ("persistent", 1)),
    ("sm_100a", 2048, 32768, ("persistent", 1)),
    ("sm_100a", 2048, 65536, ("persistent", 1)),
    ("sm_103a", 64, 2048, ("cga", 2)),
    ("sm_103a", 64, 4096, ("cga", 2)),
    ("sm_103a", 64, 8192, ("cga", 2)),
    ("sm_103a", 64, 16384, ("cga", 2)),
    ("sm_103a", 64, 32768, ("cga", 2)),
    ("sm_103a", 64, 65536, ("cga", 2)),
    ("sm_103a", 128, 2048, ("persistent", 1)),
    ("sm_103a", 128, 4096, ("persistent", 1)),
    ("sm_103a", 128, 8192, ("persistent", 1)),
    ("sm_103a", 128, 16384, ("persistent", 1)),
    ("sm_103a", 128, 32768, ("persistent", 1)),
    ("sm_103a", 128, 65536, ("persistent", 1)),
    ("sm_103a", 128, 131072, ("persistent", 1)),
    ("sm_103a", 256, 2048, ("persistent", 1)),
    ("sm_103a", 256, 4096, ("persistent", 1)),
    ("sm_103a", 256, 8192, ("persistent", 1)),
    ("sm_103a", 256, 16384, ("persistent", 1)),
    ("sm_103a", 256, 32768, ("persistent", 1)),
    ("sm_103a", 256, 65536, ("persistent", 1)),
    ("sm_103a", 256, 131072, ("persistent", 1)),
    ("sm_103a", 384, 2048, ("persistent", 1)),
    ("sm_103a", 384, 4096, ("persistent", 1)),
    ("sm_103a", 384, 8192, ("persistent", 1)),
    ("sm_103a", 384, 16384, ("persistent", 1)),
    ("sm_103a", 384, 32768, ("persistent", 1)),
    ("sm_103a", 384, 65536, ("persistent", 1)),
    ("sm_103a", 512, 2048, ("persistent", 1)),
    ("sm_103a", 512, 4096, ("persistent", 1)),
    ("sm_103a", 512, 8192, ("persistent", 1)),
    ("sm_103a", 512, 16384, None),
    ("sm_103a", 512, 32768, None),
    ("sm_103a", 512, 65536, None),
    ("sm_103a", 768, 2048, ("persistent", 1)),
    ("sm_103a", 768, 4096, None),
    ("sm_103a", 768, 8192, None),
    ("sm_103a", 768, 16384, None),
    ("sm_103a", 768, 32768, None),
    ("sm_103a", 768, 65536, None),
    ("sm_103a", 1024, 2048, ("persistent", 1)),
    ("sm_103a", 1024, 4096, ("persistent", 1)),
    ("sm_103a", 1024, 8192, ("persistent", 1)),
    ("sm_103a", 1024, 16384, ("persistent", 1)),
    ("sm_103a", 1024, 32768, ("persistent", 1)),
    ("sm_103a", 1024, 65536, ("persistent", 1)),
    ("sm_103a", 1536, 2048, ("persistent", 1)),
    ("sm_103a", 1536, 4096, ("persistent", 1)),
    ("sm_103a", 1536, 8192, ("persistent", 1)),
    ("sm_103a", 1536, 16384, ("persistent", 1)),
    ("sm_103a", 1536, 32768, ("persistent", 1)),
    ("sm_103a", 1536, 65536, ("persistent", 1)),
    ("sm_103a", 2048, 2048, ("persistent", 1)),
    ("sm_103a", 2048, 4096, ("persistent", 1)),
    ("sm_103a", 2048, 8192, ("persistent", 1)),
    ("sm_103a", 2048, 16384, ("persistent", 1)),
    ("sm_103a", 2048, 32768, ("persistent", 1)),
    ("sm_103a", 2048, 65536, ("persistent", 1)),
]
# (arch, tiles, max_kv, program): every band corner and one value past each
# bound, every pinned point and its four neighbours.
EDGES: list[tuple[str, int, int, tuple[str, int] | None]] = [
    ("sm_100a", 7, 4096, None),
    ("sm_100a", 8, 4095, None),
    ("sm_100a", 8, 4096, ("cga", 8)),
    ("sm_100a", 8, 4097, None),
    ("sm_100a", 9, 4096, None),
    ("sm_100a", 15, 237569, None),
    ("sm_100a", 16, 237568, None),
    ("sm_100a", 16, 237569, None),
    ("sm_100a", 16, 237570, None),
    ("sm_100a", 17, 237569, None),
    ("sm_100a", 31, 65536, None),
    ("sm_100a", 31, 237569, None),
    ("sm_100a", 32, 65535, None),
    ("sm_100a", 32, 65536, ("cga", 4)),
    ("sm_100a", 32, 65537, None),
    ("sm_100a", 32, 237568, None),
    ("sm_100a", 32, 237569, None),
    ("sm_100a", 32, 237570, None),
    ("sm_100a", 33, 65536, None),
    ("sm_100a", 33, 237569, None),
    ("sm_100a", 63, 2048, None),
    ("sm_100a", 63, 237569, None),
    ("sm_100a", 64, 2047, None),
    ("sm_100a", 64, 2048, ("cga", 2)),
    ("sm_100a", 64, 65536, ("cga", 2)),
    ("sm_100a", 64, 65537, None),
    ("sm_100a", 64, 237568, None),
    ("sm_100a", 64, 237569, None),
    ("sm_100a", 64, 237570, None),
    ("sm_100a", 65, 2048, None),
    ("sm_100a", 65, 65536, None),
    ("sm_100a", 65, 237569, None),
    ("sm_100a", 127, 2048, None),
    ("sm_100a", 127, 65536, None),
    ("sm_100a", 127, 131072, None),
    ("sm_100a", 127, 237569, None),
    ("sm_100a", 128, 2047, None),
    ("sm_100a", 128, 2048, ("persistent", 1)),
    ("sm_100a", 128, 8192, ("persistent", 1)),
    ("sm_100a", 128, 8193, ("persistent", 1)),
    ("sm_100a", 128, 65535, ("persistent", 1)),
    ("sm_100a", 128, 65536, ("persistent", 1)),
    ("sm_100a", 128, 65537, None),
    ("sm_100a", 128, 131071, None),
    ("sm_100a", 128, 131072, ("persistent", 1)),
    ("sm_100a", 128, 131073, None),
    ("sm_100a", 128, 237568, None),
    ("sm_100a", 128, 237569, None),
    ("sm_100a", 128, 237570, None),
    ("sm_100a", 129, 65536, ("persistent", 1)),
    ("sm_100a", 129, 237569, None),
    ("sm_100a", 191, 237569, None),
    ("sm_100a", 192, 237568, None),
    ("sm_100a", 192, 237569, None),
    ("sm_100a", 192, 237570, None),
    ("sm_100a", 193, 237569, None),
    ("sm_100a", 255, 130588, None),
    ("sm_100a", 255, 237569, None),
    ("sm_100a", 256, 130587, None),
    ("sm_100a", 256, 130588, None),
    ("sm_100a", 256, 130589, None),
    ("sm_100a", 256, 131072, ("persistent", 1)),
    ("sm_100a", 256, 131073, None),
    ("sm_100a", 256, 237568, None),
    ("sm_100a", 256, 237569, None),
    ("sm_100a", 256, 237570, None),
    ("sm_100a", 257, 130588, None),
    ("sm_100a", 257, 131072, None),
    ("sm_100a", 257, 237569, None),
    ("sm_100a", 384, 2048, ("persistent", 1)),
    ("sm_100a", 384, 65536, ("persistent", 1)),
    ("sm_100a", 384, 65537, None),
    ("sm_100a", 385, 2048, ("persistent", 1)),
    ("sm_100a", 385, 65536, None),
    ("sm_100a", 511, 32768, None),
    ("sm_100a", 511, 65369, None),
    ("sm_100a", 512, 32767, None),
    ("sm_100a", 512, 32768, ("persistent", 1)),
    ("sm_100a", 512, 32769, None),
    ("sm_100a", 512, 65368, None),
    ("sm_100a", 512, 65369, None),
    ("sm_100a", 512, 65370, None),
    ("sm_100a", 513, 32768, None),
    ("sm_100a", 513, 65369, None),
    ("sm_100a", 1023, 2048, ("persistent", 1)),
    ("sm_100a", 1023, 4096, ("persistent", 1)),
    ("sm_100a", 1023, 32768, None),
    ("sm_100a", 1024, 2047, None),
    ("sm_100a", 1024, 2048, ("persistent", 1)),
    ("sm_100a", 1024, 4095, ("persistent", 1)),
    ("sm_100a", 1024, 4096, ("persistent", 1)),
    ("sm_100a", 1024, 4097, ("persistent", 1)),
    ("sm_100a", 1024, 32767, ("persistent", 1)),
    ("sm_100a", 1024, 32768, ("persistent", 1)),
    ("sm_100a", 1024, 32769, ("persistent", 1)),
    ("sm_100a", 1024, 65536, ("persistent", 1)),
    ("sm_100a", 1024, 65537, None),
    ("sm_100a", 1025, 4096, ("persistent", 1)),
    ("sm_100a", 1025, 32768, ("persistent", 1)),
    ("sm_100a", 2048, 2048, ("persistent", 1)),
    ("sm_100a", 2048, 8192, ("persistent", 1)),
    ("sm_100a", 2048, 8193, ("persistent", 1)),
    ("sm_100a", 2048, 65536, ("persistent", 1)),
    ("sm_100a", 2048, 65537, None),
    ("sm_100a", 2049, 2048, None),
    ("sm_100a", 2049, 8192, None),
    ("sm_100a", 2049, 65536, None),
    ("sm_103a", 7, 4096, None),
    ("sm_103a", 8, 4095, None),
    ("sm_103a", 8, 4096, ("cga", 8)),
    ("sm_103a", 8, 4097, None),
    ("sm_103a", 9, 4096, None),
    ("sm_103a", 15, 237569, None),
    ("sm_103a", 16, 237568, None),
    ("sm_103a", 16, 237569, None),
    ("sm_103a", 16, 237570, None),
    ("sm_103a", 17, 237569, None),
    ("sm_103a", 31, 65536, None),
    ("sm_103a", 31, 237569, None),
    ("sm_103a", 32, 65535, None),
    ("sm_103a", 32, 65536, ("cga", 4)),
    ("sm_103a", 32, 65537, None),
    ("sm_103a", 32, 237568, None),
    ("sm_103a", 32, 237569, None),
    ("sm_103a", 32, 237570, None),
    ("sm_103a", 33, 65536, None),
    ("sm_103a", 33, 237569, None),
    ("sm_103a", 63, 2048, None),
    ("sm_103a", 63, 237569, None),
    ("sm_103a", 64, 2047, None),
    ("sm_103a", 64, 2048, ("cga", 2)),
    ("sm_103a", 64, 65536, ("cga", 2)),
    ("sm_103a", 64, 65537, None),
    ("sm_103a", 64, 237568, None),
    ("sm_103a", 64, 237569, None),
    ("sm_103a", 64, 237570, None),
    ("sm_103a", 65, 2048, None),
    ("sm_103a", 65, 65536, None),
    ("sm_103a", 65, 237569, None),
    ("sm_103a", 127, 2048, None),
    ("sm_103a", 127, 65536, None),
    ("sm_103a", 127, 131072, None),
    ("sm_103a", 127, 237569, None),
    ("sm_103a", 128, 2047, None),
    ("sm_103a", 128, 2048, ("persistent", 1)),
    ("sm_103a", 128, 2049, ("persistent", 1)),
    ("sm_103a", 128, 8192, ("persistent", 1)),
    ("sm_103a", 128, 8193, ("persistent", 1)),
    ("sm_103a", 128, 65535, ("persistent", 1)),
    ("sm_103a", 128, 65536, ("persistent", 1)),
    ("sm_103a", 128, 65537, None),
    ("sm_103a", 128, 131071, None),
    ("sm_103a", 128, 131072, ("persistent", 1)),
    ("sm_103a", 128, 131073, None),
    ("sm_103a", 128, 237568, None),
    ("sm_103a", 128, 237569, None),
    ("sm_103a", 128, 237570, None),
    ("sm_103a", 129, 65536, ("persistent", 1)),
    ("sm_103a", 129, 237569, None),
    ("sm_103a", 191, 237569, None),
    ("sm_103a", 192, 237568, None),
    ("sm_103a", 192, 237569, None),
    ("sm_103a", 192, 237570, None),
    ("sm_103a", 193, 237569, None),
    ("sm_103a", 255, 130588, None),
    ("sm_103a", 255, 237569, None),
    ("sm_103a", 256, 130587, None),
    ("sm_103a", 256, 130588, None),
    ("sm_103a", 256, 130589, None),
    ("sm_103a", 256, 131072, ("persistent", 1)),
    ("sm_103a", 256, 131073, None),
    ("sm_103a", 256, 237568, None),
    ("sm_103a", 256, 237569, None),
    ("sm_103a", 256, 237570, None),
    ("sm_103a", 257, 130588, None),
    ("sm_103a", 257, 131072, None),
    ("sm_103a", 257, 237569, None),
    ("sm_103a", 384, 2048, ("persistent", 1)),
    ("sm_103a", 384, 65536, ("persistent", 1)),
    ("sm_103a", 384, 65537, None),
    ("sm_103a", 385, 2048, ("persistent", 1)),
    ("sm_103a", 385, 65536, None),
    ("sm_103a", 511, 32768, None),
    ("sm_103a", 511, 65369, None),
    ("sm_103a", 512, 2048, ("persistent", 1)),
    ("sm_103a", 512, 8192, ("persistent", 1)),
    ("sm_103a", 512, 8193, None),
    ("sm_103a", 512, 32767, None),
    ("sm_103a", 512, 32768, None),
    ("sm_103a", 512, 32769, None),
    ("sm_103a", 512, 65368, None),
    ("sm_103a", 512, 65369, None),
    ("sm_103a", 512, 65370, None),
    ("sm_103a", 513, 2048, ("persistent", 1)),
    ("sm_103a", 513, 8192, None),
    ("sm_103a", 513, 32768, None),
    ("sm_103a", 513, 65369, None),
    ("sm_103a", 1023, 2048, ("persistent", 1)),
    ("sm_103a", 1023, 4096, None),
    ("sm_103a", 1023, 32768, None),
    ("sm_103a", 1024, 2047, None),
    ("sm_103a", 1024, 2048, ("persistent", 1)),
    ("sm_103a", 1024, 4095, ("persistent", 1)),
    ("sm_103a", 1024, 4096, ("persistent", 1)),
    ("sm_103a", 1024, 4097, ("persistent", 1)),
    ("sm_103a", 1024, 32767, ("persistent", 1)),
    ("sm_103a", 1024, 32768, ("persistent", 1)),
    ("sm_103a", 1024, 32769, ("persistent", 1)),
    ("sm_103a", 1024, 65536, ("persistent", 1)),
    ("sm_103a", 1024, 65537, None),
    ("sm_103a", 1025, 4096, ("persistent", 1)),
    ("sm_103a", 1025, 32768, ("persistent", 1)),
    ("sm_103a", 2048, 2048, ("persistent", 1)),
    ("sm_103a", 2048, 2049, ("persistent", 1)),
    ("sm_103a", 2048, 65536, ("persistent", 1)),
    ("sm_103a", 2048, 65537, None),
    ("sm_103a", 2049, 2048, None),
    ("sm_103a", 2049, 65536, None),
]
# row -> (batch, KV heads, max KV length) of the derivation's standings rows.
ROW_SHAPES: dict[str, tuple[int, int, int]] = {
    "agentx_b16_hkv1": (16, 1, 237569),
    "agentx_b32_hkv1": (32, 1, 237569),
    "agentx_b64_hkv1": (64, 1, 237569),
    "agentx_b128_hkv1": (128, 1, 237569),
    "agentx_b192_hkv1": (192, 1, 237569),
    "agentx_b256_hkv1": (256, 1, 237569),
    "random_128_128k_b256_hkv1": (256, 1, 130588),
    "random_128_65k_b64_hkv8": (64, 8, 65369),
    "uniform_b4_hkv8_s65536": (4, 8, 65536),
    "uniform_b16_hkv8_s65536": (16, 8, 65536),
    "uniform_b64_hkv8_s32768": (64, 8, 32768),
    "uniform_b128_hkv8_s32768": (128, 8, 32768),
    "uniform_b128_hkv8_s4096": (128, 8, 4096),
    "uniform_b1_hkv8_s4096": (1, 8, 4096),
}
# row -> {arch: program}: the served program of every shape row per target at
# the target's SM count of record (None: the present route).
ROW_ROUTES: dict[str, dict[str, tuple[str, int] | None]] = {
    "agentx_b16_hkv1": {"sm_100a": None, "sm_103a": None},
    "agentx_b32_hkv1": {"sm_100a": None, "sm_103a": None},
    "agentx_b64_hkv1": {"sm_100a": None, "sm_103a": None},
    "agentx_b128_hkv1": {"sm_100a": None, "sm_103a": None},
    "agentx_b192_hkv1": {"sm_100a": None, "sm_103a": None},
    "agentx_b256_hkv1": {"sm_100a": None, "sm_103a": None},
    "random_128_128k_b256_hkv1": {"sm_100a": None, "sm_103a": None},
    "random_128_65k_b64_hkv8": {"sm_100a": None, "sm_103a": None},
    "uniform_b4_hkv8_s65536": {"sm_100a": ("cga", 4), "sm_103a": ("cga", 4)},
    "uniform_b16_hkv8_s65536": {
        "sm_100a": ("persistent", 1),
        "sm_103a": ("persistent", 1),
    },
    "uniform_b64_hkv8_s32768": {"sm_100a": ("persistent", 1), "sm_103a": None},
    "uniform_b128_hkv8_s32768": {
        "sm_100a": ("persistent", 1),
        "sm_103a": ("persistent", 1),
    },
    "uniform_b128_hkv8_s4096": {
        "sm_100a": ("persistent", 1),
        "sm_103a": ("persistent", 1),
    },
    "uniform_b1_hkv8_s4096": {"sm_100a": ("cga", 8), "sm_103a": ("cga", 8)},
}
# --- END SEL_HD256_EXPECTATIONS ---


def _require_table() -> None:
    """The expectation table must be filled for every shape row and target: an
    empty table fails every dependent item instead of passing vacuously."""

    verdicts = (None, *PROGRAMS)
    filled = (
        bool(TILES)
        and bool(MAX_KV)
        and bool(OTHER_SM_COUNTS)
        and set(SM_COUNT) == set(ARCHS)
        and bool(PROBED)
        and bool(EDGES)
        and set(ROW_SHAPES) == set(SHAPE_ROWS)
        and set(ROW_ROUTES) == set(SHAPE_ROWS)
        and all(set(routes) == set(ARCHS) for routes in ROW_ROUTES.values())
        and all(
            program in verdicts
            for routes in ROW_ROUTES.values()
            for program in routes.values()
        )
        and all(
            arch in ARCHS and program in verdicts
            for arch, _tiles, _max_kv, program in PROBED + EDGES
        )
    )
    if not filled:
        pytest.fail(
            "the selector expectation table (PROBED, EDGES, ROW_ROUTES, ...) is not "
            "filled for every shape row and target"
        )


def _ctas_kv(tiles: int, max_kv: int, sm_count: int) -> int:
    # The trtllm-gen split of the hd256 group-8 q_len-1 decode: the CTAs one
    # (request, KV head) tile's KV sequence is split over.
    return min(-(-max_kv // 512), max(1, sm_count // tiles))


def test_aligned_hd256_expectation_table_is_filled() -> None:
    _require_table()


@pytest.mark.parametrize("row", sorted(RANDOM_ROW_IDENTITY))
def test_random_shape_rows_reproduce_the_measured_lengths(row) -> None:
    lengths = SHAPE_ROWS[row][1]
    assert (len(lengths), min(lengths), max(lengths)) == RANDOM_ROW_IDENTITY[row]


@pytest.mark.parametrize("row", sorted(SHAPE_ROWS))
def test_shape_rows_are_the_derivation_rows(row) -> None:
    _require_table()
    num_kv_heads, seq_lens = SHAPE_ROWS[row]
    assert (len(seq_lens), num_kv_heads, max(seq_lens)) == ROW_SHAPES[row]


def test_case_rows_spell_the_kernel_test_groups() -> None:
    for num_q_heads, num_kv_heads, seq_lens in CASE_ROWS.values():
        assert num_q_heads == GROUP * num_kv_heads
        assert len(seq_lens) >= 2
    num_q_heads, num_kv_heads, _ = GROUP4_CASE
    assert num_q_heads == 4 * num_kv_heads


# --- The selector mirror against the source selector test ------------------


def _select(tiles, max_kv, arch, sm_count=None, **overrides):
    kwargs = dict(
        q_len=1,
        num_q_heads=GROUP,
        arch=arch,
        sm_count=SM_COUNT[arch] if sm_count is None else sm_count,
    )
    kwargs.update(overrides)
    return cake_api.aligned_hd256_program(tiles, 1, max_kv, **kwargs)


def test_aligned_hd256_selector_constants_of_record() -> None:
    assert cake_api.SEL_HD256_GROUP == GROUP
    assert cake_api.SEL_HD256_ARCHS == ARCHS
    assert cake_api.SEL_HD256_KV_PER_CTA_SPLIT == 512


@pytest.mark.parametrize("arch", ARCHS)
def test_aligned_hd256_selector_gives_one_verdict_on_every_shape(arch) -> None:
    _require_table()
    verdicts = (None, *PROGRAMS)
    for tiles in TILES:
        for max_kv in MAX_KV:
            for sm_count in [SM_COUNT[arch], *OTHER_SM_COUNTS]:
                got = _select(tiles, max_kv, arch, sm_count)
                assert got in verdicts, (tiles, max_kv, sm_count, got)
                # A program only where the split at this SM count picks its class.
                assert got is None or _ctas_kv(tiles, max_kv, sm_count) == got[1], (
                    tiles,
                    max_kv,
                    sm_count,
                    got,
                )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(q_len=2),
        dict(num_q_heads=GROUP + 1),
        dict(num_q_heads=GROUP // 2),
        dict(num_q_heads=2 * GROUP),
        dict(arch="sm_90a"),
        dict(sm_count=0),
    ],
)
def test_aligned_hd256_selector_other_shapes_keep_the_present_route(kwargs) -> None:
    _require_table()
    overrides = {"arch": "sm_100a", "sm_count": SM_COUNT["sm_100a"], **kwargs}
    for tiles in TILES:
        for max_kv in MAX_KV:
            assert _select(tiles, max_kv, **overrides) is None, (tiles, max_kv)


@pytest.mark.parametrize("arch", ARCHS)
def test_aligned_hd256_selector_empty_launch_keeps_the_present_route(arch) -> None:
    _require_table()
    assert _select(0, 4096, arch) is None
    assert _select(1, 0, arch) is None


@pytest.mark.parametrize("case", PROBED, ids=lambda c: f"{c[0]}-t{c[1]}-kv{c[2]}")
def test_aligned_hd256_selector_probed_grid(case) -> None:
    arch, tiles, max_kv, program = case
    assert _select(tiles, max_kv, arch) == program


@pytest.mark.parametrize("case", EDGES, ids=lambda c: f"{c[0]}-t{c[1]}-kv{c[2]}")
def test_aligned_hd256_selector_edges(case) -> None:
    arch, tiles, max_kv, program = case
    assert _select(tiles, max_kv, arch) == program


@pytest.mark.parametrize(
    "case",
    [c for c in PROBED + EDGES if c[3] is not None],
    ids=lambda c: f"{c[0]}-t{c[1]}-kv{c[2]}",
)
def test_aligned_hd256_selector_programs_follow_the_split(case) -> None:
    arch, tiles, max_kv, program = case
    for sm_count in OTHER_SM_COUNTS:
        expect = program if _ctas_kv(tiles, max_kv, sm_count) == program[1] else None
        assert _select(tiles, max_kv, arch, sm_count) == expect, sm_count


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("row", sorted(SHAPE_ROWS))
def test_aligned_hd256_selector_shape_rows_route_as_derived(row, arch) -> None:
    _require_table()
    num_kv_heads, seq_lens = SHAPE_ROWS[row]
    got = cake_api.aligned_hd256_program(
        len(seq_lens),
        num_kv_heads,
        max(seq_lens),
        q_len=1,
        num_q_heads=GROUP * num_kv_heads,
        arch=arch,
        sm_count=SM_COUNT[arch],
    )
    assert got == ROW_ROUTES[row][arch]


# --- The pinned program records ---------------------------------------------


def test_cake_fmha_aligned_hd256_program_table_and_launch_grid() -> None:
    """The identifier table pins one record per program; the launch grid (the
    parity lane's host grid) is ``(CTAs, KV heads, requests)``: one CTA or one
    cluster of the program's CTAs per (request, KV head)."""

    assert cake_api._ALIGNED_HD256_COMPONENT == COMPONENT
    assert cake_api._ALIGNED_HD256_PROGRAM_IDS == PROGRAMS
    table = cake_api._ALIGNED_HD256_PROGRAMS
    assert sorted(table) == sorted(PROGRAMS.values())
    names = set()
    for (_kind, ctas), ident in PROGRAMS.items():
        table_ctas, name = table[ident]
        assert table_ctas == ctas
        assert name.startswith("cake_fmha_aligned_hd256_")
        digest = name.removeprefix("cake_fmha_aligned_hd256_")
        assert len(digest) == 20 and set(digest) <= set("0123456789abcdef")
        names.add(name)
        record = SimpleNamespace(cluster=(ctas, 1, 1))
        grid = cake_api.cake_fmha_aligned_hd256_launch_grid
        assert grid(record, 3, 8) == (ctas, 8, 3)
        assert grid(record, 256, 1) == (ctas, 1, 256)
    assert len(names) == len(PROGRAMS)
    assert tuple(cake_api._ALIGNED_HD256_HOST_PARAMS) == HOST_PARAMS


def test_cake_fmha_aligned_hd256_program_table_matches_the_generated_loader() -> None:
    """The pinned records name the generated loader's programs of the same
    identifiers (the load-time checks, host-only): name, cluster, binding
    argument order and both targets; the loader publishes exactly those
    programs."""

    from flashinfer.jit import cake_fmha_aligned_hd256 as loader
    from flashinfer.jit.cake_program import GRID_ARGS, CakeProgram

    names = []
    for ident, (ctas, name) in cake_api._ALIGNED_HD256_PROGRAMS.items():
        program = getattr(loader, ident)
        assert isinstance(program, CakeProgram)
        assert program.name == name
        assert tuple(program.cluster) == (ctas, 1, 1)
        assert tuple(program.block) == (512, 1, 1)
        assert tuple(program.args) == (*cake_api._ALIGNED_HD256_HOST_PARAMS, *GRID_ARGS)
        assert tuple(program.tensors) == (
            "Q",
            "K",
            "V",
            "O",
            "page_table",
            "seq_lens_kv",
        )
        assert {"sm_100a", "sm_103a"} <= set(program.targets)
        assert program.use_pdl is False and program.cooperative is False
        assert program.dynamic_smem_bytes == PROGRAM_SMEM[ident]
        names.append(name)
    assert sorted(p.name for p in loader.PROGRAMS) == sorted(names)


# --- The load-time checks against a stand-in loader (host-only) -------------


def _stand_in_records() -> dict[str, CakeProgram]:
    """One record per identifier with exactly the launch the module pins."""

    records = {}
    for ident, (ctas, name) in cake_api._ALIGNED_HD256_PROGRAMS.items():
        records[ident] = CakeProgram(
            name=name,
            operator_dir="cake_fmha_aligned_hd256",
            kernel="kernel",
            sources=("kernel.cu", "binding.cu"),
            targets=ARCHS,
            compile_options=tuple((arch, ()) for arch in ARCHS),
            args=(*HOST_PARAMS, *GRID_ARGS),
            tensors=HOST_PARAMS[:6],
            block=(512, 1, 1),
            cluster=(ctas, 1, 1),
            dynamic_smem_bytes=PROGRAM_SMEM[ident],
            use_pdl=False,
            cooperative=False,
            needs_device_linking=False,
        )
    return records


def _install_stand_in_loader(monkeypatch, records) -> None:
    """Publish ``records`` as ``flashinfer.jit.cake_fmha_aligned_hd256`` for the
    test; the loader is imported lazily, at load."""

    loader = types.ModuleType("flashinfer.jit.cake_fmha_aligned_hd256")
    for ident, record in records.items():
        setattr(loader, ident, record)
    loader.PROGRAMS = tuple(records.values())
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    monkeypatch.setattr(
        flashinfer.jit, "cake_fmha_aligned_hd256", loader, raising=False
    )


def _aligned_hd256_route(aligned_program, target="sm100a"):
    return cake_api.CakeFmhaDecodeRoute(
        target=target,
        batch_size=16,
        q_len=1,
        num_q_heads=GROUP * 8,
        num_kv_heads=8,
        has_sink=False,
        has_window=False,
        use_scale_ptr=False,
        retain_kv_l2=False,
        component=COMPONENT,
        page_size=PAGE_SIZE,
        aligned_program=aligned_program,
    )


def _record_defects(ident, records):
    """Records that differ from the pinned launch of ``ident`` (None: the loader
    publishes no such identifier), with the load-time refusal each must meet."""

    record = records[ident]
    ctas = record.cluster[0]
    other = next(i for i in sorted(records) if i != ident)
    swapped_args = (record.args[1], record.args[0], *record.args[2:])
    return {
        "missing": (None, "publishes no CakeProgram"),
        "not_a_cake_program": (
            SimpleNamespace(**dataclasses.asdict(record)),
            "publishes no CakeProgram",
        ),
        "another_name": (
            dataclasses.replace(record, name=records[other].name),
            "this module pins",
        ),
        "another_programs_record": (records[other], "this module pins"),
        "no_sm_100a_build": (
            dataclasses.replace(record, targets=("sm_103a",)),
            "ships no 'sm_100a' build",
        ),
        "argument_order": (
            dataclasses.replace(record, args=swapped_args),
            "binding takes",
        ),
        "no_grid_arguments": (
            dataclasses.replace(record, args=HOST_PARAMS),
            "binding takes",
        ),
        "cluster_size": (
            dataclasses.replace(record, cluster=(1 if ctas != 1 else 4, 1, 1)),
            "launches cluster",
        ),
        "cluster_shape": (
            dataclasses.replace(record, cluster=(ctas, 1, 2)),
            "launches cluster",
        ),
    }


@pytest.mark.parametrize("target", sorted(TARGET_ARCH))
@pytest.mark.parametrize("ident", sorted(PROGRAMS.values()))
def test_cake_fmha_aligned_hd256_load_accepts_the_pinned_record(
    monkeypatch, ident, target
) -> None:
    records = _stand_in_records()
    _install_stand_in_loader(monkeypatch, records)
    monkeypatch.setattr(CakeProgram, "entry", lambda self, arch: (self.name, arch))
    module = cake_api._load_cake_fmha_aligned_hd256_decode_module(
        _aligned_hd256_route(ident, target)
    )
    assert isinstance(module, cake_api.CakeFmhaAlignedHd256DecodeModule)
    assert module.program is records[ident]
    assert module.target == TARGET_ARCH[target]
    assert module.entry == (records[ident].name, TARGET_ARCH[target])


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "not_a_cake_program",
        "another_name",
        "another_programs_record",
        "no_sm_100a_build",
        "argument_order",
        "no_grid_arguments",
        "cluster_size",
        "cluster_shape",
    ],
)
@pytest.mark.parametrize("ident", sorted(PROGRAMS.values()))
def test_cake_fmha_aligned_hd256_load_fails_closed_on_a_record_that_differs(
    monkeypatch, ident, defect
) -> None:
    records = _stand_in_records()
    bad, message = _record_defects(ident, records)[defect]
    if bad is None:
        del records[ident]
    else:
        records[ident] = bad
    _install_stand_in_loader(monkeypatch, records)

    def no_entry(self, arch):
        raise AssertionError("a refused record reached the build")

    monkeypatch.setattr(CakeProgram, "entry", no_entry)
    with pytest.raises(RuntimeError, match=message):
        cake_api._load_cake_fmha_aligned_hd256_decode_module(
            _aligned_hd256_route(ident)
        )


def test_cake_fmha_aligned_hd256_load_fails_closed_on_an_unknown_program(
    monkeypatch,
) -> None:
    _install_stand_in_loader(monkeypatch, _stand_in_records())
    with pytest.raises(RuntimeError, match="names no program"):
        cake_api._load_cake_fmha_aligned_hd256_decode_module(
            _aligned_hd256_route("ALIGNED_HD256_C8_BF16")
        )


# --- The shape rows through the module's route selection (host-only) --------


def _decode_kwargs(
    *,
    batch_size,
    num_kv_heads,
    seq_lens,
    num_q_heads=None,
    stacked_kv=False,
    sm_count=148,
):
    """Host-side ``select_cake_fmha_decode_route`` inputs of one shape row.

    A four-page pool is enough: the selector reads shapes, strides and dtypes.
    ``stacked_kv`` hands the K / V views of one ``[pages, 2, ...]`` tensor, the
    layout the aligned programs cannot address.
    """

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


def _patch_host_only(monkeypatch, target, sm_count):
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: target)
    monkeypatch.setattr(cake_api, "_smallm_sm_count", lambda device: sm_count)
    monkeypatch.setattr(
        cake_api,
        "_is_cake_fmha_decode_native_bf16_available",
        lambda *args, **kwargs: True,
    )


def _expect_row_route(route, *, target, program, num_kv_heads, seq_lens) -> None:
    assert route is not None
    if program is None:
        # The module's present head-dim-256 route, unchanged.
        assert route.component == PRESENT
        assert route.aligned_program is None
        return
    assert route == cake_api.CakeFmhaDecodeRoute(
        target=target,
        batch_size=len(seq_lens),
        q_len=1,
        num_q_heads=GROUP * num_kv_heads,
        num_kv_heads=num_kv_heads,
        has_sink=False,
        has_window=False,
        use_scale_ptr=False,
        retain_kv_l2=(max(seq_lens) + 127) // 128 <= 9,
        component=COMPONENT,
        page_size=PAGE_SIZE,
        aligned_program=PROGRAMS[program],
    )
    assert cake_api.cake_fmha_route_is_optimized(route)
    assert cake_api._route_components(route) == (PROGRAMS[program],)


@pytest.mark.parametrize("target", sorted(TARGET_ARCH))
@pytest.mark.parametrize("row", sorted(SHAPE_ROWS))
def test_cake_fmha_aligned_hd256_route_serves_shape_rows_as_derived(
    monkeypatch, target, row
) -> None:
    _require_table()
    arch = TARGET_ARCH[target]
    _patch_host_only(monkeypatch, target, SM_COUNT[arch])
    num_kv_heads, seq_lens = SHAPE_ROWS[row]
    kwargs = _decode_kwargs(
        batch_size=len(seq_lens),
        num_kv_heads=num_kv_heads,
        seq_lens=seq_lens,
        sm_count=SM_COUNT[arch],
    )
    _expect_row_route(
        _select_route(kwargs),
        target=target,
        program=ROW_ROUTES[row][arch],
        num_kv_heads=num_kv_heads,
        seq_lens=seq_lens,
    )


@pytest.mark.parametrize("target", sorted(TARGET_ARCH))
@pytest.mark.parametrize("row", sorted(SHAPE_ROWS))
def test_cake_fmha_aligned_hd256_route_reads_the_device_sm_count(
    monkeypatch, target, row
) -> None:
    """The route takes the program the selector names at the device's SM count:
    on another SM count the trtllm-gen split may pick another kernel class, and
    the present route then serves the launch."""

    _require_table()
    arch = TARGET_ARCH[target]
    num_kv_heads, seq_lens = SHAPE_ROWS[row]
    for sm_count in OTHER_SM_COUNTS:
        _patch_host_only(monkeypatch, target, sm_count)
        kwargs = _decode_kwargs(
            batch_size=len(seq_lens),
            num_kv_heads=num_kv_heads,
            seq_lens=seq_lens,
            sm_count=sm_count,
        )
        _expect_row_route(
            _select_route(kwargs),
            target=target,
            program=cake_api.aligned_hd256_program(
                len(seq_lens),
                num_kv_heads,
                max(seq_lens),
                q_len=1,
                num_q_heads=GROUP * num_kv_heads,
                arch=arch,
                sm_count=sm_count,
            ),
            num_kv_heads=num_kv_heads,
            seq_lens=seq_lens,
        )


@pytest.mark.parametrize("target", sorted(TARGET_ARCH))
def test_cake_fmha_aligned_hd256_group4_case_keeps_the_present_route(
    monkeypatch, target
) -> None:
    _require_table()
    sm_count = SM_COUNT[TARGET_ARCH[target]]
    _patch_host_only(monkeypatch, target, sm_count)
    num_q_heads, num_kv_heads, seq_lens = GROUP4_CASE
    kwargs = _decode_kwargs(
        batch_size=len(seq_lens),
        num_kv_heads=num_kv_heads,
        seq_lens=seq_lens,
        num_q_heads=num_q_heads,
        sm_count=sm_count,
    )
    route = _select_route(kwargs)
    assert route is not None
    assert route.component == PRESENT
    assert route.aligned_program is None


@pytest.mark.parametrize("program", sorted(PROGRAMS))
def test_cake_fmha_aligned_hd256_route_fails_closed_outside_its_contract(
    monkeypatch, program
) -> None:
    _require_table()
    _patch_host_only(monkeypatch, "sm100a", SM_COUNT["sm_100a"])
    # Hold the selector to one program so the contract checks around it decide.
    monkeypatch.setattr(cake_api, "aligned_hd256_program", lambda *a, **k: program)
    num_kv_heads, seq_lens = 8, [4096] * 16
    batch_size = len(seq_lens)
    num_q_heads = GROUP * num_kv_heads
    dtype = torch.bfloat16
    kwargs = _decode_kwargs(
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        seq_lens=seq_lens,
        sm_count=SM_COUNT["sm_100a"],
    )
    route = _select_route(kwargs)
    assert route is not None and route.component == COMPONENT
    assert route.aligned_program == PROGRAMS[program]

    stacked = _decode_kwargs(
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        seq_lens=seq_lens,
        stacked_kv=True,
        sm_count=SM_COUNT["sm_100a"],
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


def _hd256_case(device, *, num_q_heads, num_kv_heads, seq_lens, seed):
    """Query ``[batch, num_q_heads, 256]`` (``randn x 3``: a peaked softmax, so
    ``|O|`` sits well above the tolerance), separate contiguous K / V pools
    ``[pages, num_kv_heads, 16, 256]`` (``randn``; the layout the aligned
    programs read in place), a shuffled page table and the device lengths."""

    generator = torch.Generator(device=device).manual_seed(seed)
    batch = len(seq_lens)
    pages_per_req = [(s + PAGE_SIZE - 1) // PAGE_SIZE for s in seq_lens]
    total_pages = sum(pages_per_req)
    query = (
        torch.randn((batch, num_q_heads, HEAD_DIM), device=device, generator=generator)
        * 3
    ).to(torch.bfloat16)
    pool_shape = (total_pages, num_kv_heads, PAGE_SIZE, HEAD_DIM)
    key = torch.randn(
        pool_shape, device=device, generator=generator, dtype=torch.bfloat16
    )
    value = torch.randn(
        pool_shape, device=device, generator=generator, dtype=torch.bfloat16
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


def _reference_decode(
    query, key, value, block_tables, seq_lens, *, scale, dtype=torch.float32
):
    """GQA decode in ``dtype`` (FP32 unless given) over the paged HND pools:
    request ``b`` attends its first ``seq_lens[b]`` tokens, query head ``h`` reads
    KV head ``h // group``."""

    batch = int(seq_lens.numel())
    num_q_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    group = num_q_heads // num_kv_heads
    out = torch.empty(query.shape, dtype=dtype, device=query.device)
    for b in range(batch):
        length = int(seq_lens[b])
        pages = block_tables[b, : (length + PAGE_SIZE - 1) // PAGE_SIZE]
        k = key[pages].permute(1, 0, 2, 3).reshape(num_kv_heads, -1, HEAD_DIM)
        v = value[pages].permute(1, 0, 2, 3).reshape(num_kv_heads, -1, HEAD_DIM)
        k = k[:, :length].to(dtype)
        v = v[:, :length].to(dtype)
        q = query[b].to(dtype).view(num_kv_heads, group, HEAD_DIM)
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


def _device_sm_count(device) -> int:
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def _check_row(
    monkeypatch,
    *,
    query,
    key,
    value,
    block_tables,
    seq_lens_t,
    seq_lens,
    expect_program,
):
    """One shape row through ``backend="cake"``: the route taken, the FP32
    reference at the tolerance with a non-vacuous reference, and CUDA-graph
    replay -- bitwise equal to the eager result on an aligned program (the
    present route is held to the tolerance), then under shorter lengths."""

    device = query.device
    scale = HEAD_DIM**-0.5
    max_seq_len = max(seq_lens)
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    bitwise = expect_program is not None

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
    assert route is not None
    if bitwise:
        assert route.component == COMPONENT
        assert route.aligned_program == PROGRAMS[expect_program]
    else:
        assert route.component == PRESENT
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
    return actual


@_requires_blackwell
@pytest.mark.parametrize("row", sorted(SHAPE_ROWS))
def test_cake_decode_aligned_hd256_shape_rows_match_reference_and_replay(
    monkeypatch, row
) -> None:
    _require_table()
    device = torch.device("cuda")
    arch = TARGET_ARCH[cake_api._cake_fmha_target(device)]
    sm_count = _device_sm_count(device)
    num_kv_heads, seq_lens = SHAPE_ROWS[row]
    # The program the selector names at this device's SM count: the derivation's
    # verdict on a device of the SM count of record.
    program = cake_api.aligned_hd256_program(
        len(seq_lens),
        num_kv_heads,
        max(seq_lens),
        q_len=1,
        num_q_heads=GROUP * num_kv_heads,
        arch=arch,
        sm_count=sm_count,
    )
    if sm_count == SM_COUNT[arch]:
        assert program == ROW_ROUTES[row][arch]
    query, key, value, block_tables, seq_lens_t = _hd256_case(
        device,
        num_q_heads=GROUP * num_kv_heads,
        num_kv_heads=num_kv_heads,
        seq_lens=seq_lens,
        seed=20261102 + len(seq_lens) + num_kv_heads,
    )
    _check_row(
        monkeypatch,
        query=query,
        key=key,
        value=value,
        block_tables=block_tables,
        seq_lens_t=seq_lens_t,
        seq_lens=seq_lens,
        expect_program=program,
    )


@_requires_blackwell
@pytest.mark.parametrize("program", sorted(PROGRAMS))
@pytest.mark.parametrize("case", sorted(CASE_ROWS))
def test_cake_decode_aligned_hd256_kernel_cases_match_reference_on_each_program(
    monkeypatch, case, program
) -> None:
    """The kernel tests' group-8 cases on each aligned program itself: the
    selector is held to the program so the small shapes (a partial page,
    ragged lengths, a single-token request, requests shorter than the cluster's
    rank count) exercise it."""

    device = torch.device("cuda")
    num_q_heads, num_kv_heads, seq_lens = CASE_ROWS[case]
    monkeypatch.setattr(cake_api, "aligned_hd256_program", lambda *a, **k: program)
    query, key, value, block_tables, seq_lens_t = _hd256_case(
        device,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        seq_lens=seq_lens,
        seed=20261102 + len(seq_lens) + num_kv_heads,
    )
    _check_row(
        monkeypatch,
        query=query,
        key=key,
        value=value,
        block_tables=block_tables,
        seq_lens_t=seq_lens_t,
        seq_lens=seq_lens,
        expect_program=program,
    )


@_requires_blackwell
def test_cake_decode_aligned_hd256_group4_case_runs_on_the_present_route(
    monkeypatch,
) -> None:
    device = torch.device("cuda")
    num_q_heads, num_kv_heads, seq_lens = GROUP4_CASE
    query, key, value, block_tables, seq_lens_t = _hd256_case(
        device,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        seq_lens=seq_lens,
        seed=20261102 + len(seq_lens) + num_kv_heads,
    )
    _check_row(
        monkeypatch,
        query=query,
        key=key,
        value=value,
        block_tables=block_tables,
        seq_lens_t=seq_lens_t,
        seq_lens=seq_lens,
        expect_program=None,
    )


# The product row the source selector pins to the cluster program of 8 CTAs on
# both targets, where trtllm-gen's split at the SM count of record also takes a
# cluster of 8: the program reproduces trtllm-gen's output bit for bit.
PINNED_CGA8_ROW = "uniform_b1_hkv8_s4096"


@_requires_blackwell
def test_cake_decode_aligned_hd256_pinned_cga8_row_matches_trtllm_gen_bitwise(
    monkeypatch,
) -> None:
    """The pinned row (one request, 8 KV heads, 4096 tokens) through
    ``backend="cake"`` on the cluster program of 8 CTAs: the route taken, the
    FP32 reference and CUDA-graph replay bitwise equal to the eager result
    (``_check_row``), the FP64 reference at the tolerance, and the output bitwise
    equal to ``backend="trtllm-gen"`` on the same inputs."""

    _require_table()
    device = torch.device("cuda")
    arch = TARGET_ARCH[cake_api._cake_fmha_target(device)]
    sm_count = _device_sm_count(device)
    num_kv_heads, seq_lens = SHAPE_ROWS[PINNED_CGA8_ROW]
    assert ROW_ROUTES[PINNED_CGA8_ROW][arch] == ("cga", 8)
    program = cake_api.aligned_hd256_program(
        len(seq_lens),
        num_kv_heads,
        max(seq_lens),
        q_len=1,
        num_q_heads=GROUP * num_kv_heads,
        arch=arch,
        sm_count=sm_count,
    )
    if sm_count == SM_COUNT[arch]:
        assert program == ("cga", 8)
    if program != ("cga", 8):
        pytest.skip(f"the trtllm-gen split at {sm_count} SMs takes no cluster of 8")
    query, key, value, block_tables, seq_lens_t = _hd256_case(
        device,
        num_q_heads=GROUP * num_kv_heads,
        num_kv_heads=num_kv_heads,
        seq_lens=seq_lens,
        seed=20261102 + len(seq_lens) + num_kv_heads,
    )
    actual = _check_row(
        monkeypatch,
        query=query,
        key=key,
        value=value,
        block_tables=block_tables,
        seq_lens_t=seq_lens_t,
        seq_lens=seq_lens,
        expect_program=program,
    )
    scale = HEAD_DIM**-0.5
    expected = _reference_decode(
        query, key, value, block_tables, seq_lens_t, scale=scale, dtype=torch.float64
    )
    assert float(expected.abs().amax()) > MIN_REF_ABSMAX
    torch.testing.assert_close(actual.double(), expected, atol=ATOL, rtol=RTOL)
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    reference = trtllm_batch_decode_with_kv_cache(
        query,
        (key, value),
        workspace,
        block_tables,
        seq_lens_t,
        max(seq_lens),
        bmm1_scale=scale,
        bmm2_scale=1.0,
        window_left=-1,
        kv_layout="HND",
        q_len_per_req=1,
        backend="trtllm-gen",
    )
    torch.cuda.synchronize()
    assert reference.dtype == actual.dtype and reference.shape == actual.shape
    assert torch.equal(actual, reference)
