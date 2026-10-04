# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cake SM120 DeepSeek-V4.1 mixed-cache sparse-MLA decode (``backend="cake"`` on SM120/SM121).

The DeepSeek-V4.1 decode reads two differently encoded caches:

* the **main (SWA) cache**, 528 bytes per token: 512 E4M3 values (RoPE lanes
  included) followed, per page, by a 16-byte footer of UE8M0 scales over
  32-wide groups (``kv_cache_format="fp8_dsv41"`` / ``kv_scale_format=
  "ue8m0_g32"``); physical page = ``page_size * 512`` data bytes then
  ``page_size * 16`` scale bytes;
* the optional **extra (compressed) cache**, 288 bytes per token: 512 E2M1
  values packed in 256 bytes followed, per page, by a 32-byte footer of E4M3
  scales over 16-wide groups (the V41_FP4 ABI written by
  :func:`flashinfer.mla.dsv41_fp4_quantize_pack_sparse_mla_cache`); physical
  page = ``page_size * 256`` data bytes then ``page_size * 32`` scale bytes.

Together they are the public ``kv_cache_format="fp8_dsv41_fp4_ca"`` form. The
device code is generated from the Cake kernel schedules into
``csrc/cake_dsv4/sm_120a`` (family ``cake_sparse_mla_dsv41_mixed``); this
module owns the host side: format validation (row bytes, footer page geometry,
16-byte page strides, independent runtime page sizes for both caches, HND / NHD
/ 3-D / 2-D views), the split and head-tile planners, caller-owned split
scratch, the public-API workspace carve, the compute-precision selection and
the ``SparseMLASm120Wrapper`` route.

Contract (shared with the SM120 ``"sparse"`` backend for this format):

* ``q`` ``[T, H, 512]`` BF16 (never quantized by the ``"bf16"`` route);
  ``output`` ``[T, H, 512]`` BF16; ``out_lse`` ``[T, H]`` fp32 base-2 (times
  ``lse_scale``).
* ``indices`` / ``extra_indices`` ``[T, topk]`` (or ``[T, 1, topk]``) int32,
  ``-1`` masks a slot; ``topk_length`` / ``extra_topk_length`` ``[T]`` int32
  clamp the valid prefix to ``[0, topk]``; ``attn_sink`` ``[H]`` fp32
  (sigmoid gate of the output, logaddexp into LSE).
* An empty row (no valid slot in either cache) writes zeros and ``-inf`` LSE,
  or ``sink * log2(e) * lse_scale`` when ``attn_sink`` is given.
* ``compute_precision``: ``"bf16"`` (default; both caches dequantized exactly
  to BF16 on chip, BF16 x BF16 QK with fp32 accumulation, fp32 softmax, BF16 P,
  fp32 PV accumulation and split merge, one BF16 output rounding) or
  ``"fp8"`` (an optional, separately validated route with FP8 QK; only
  available when the generated family exports it).
* Split-K: ``num_splits`` CTAs per (token, head block) write BF16 / fp32
  partials to caller-owned ``mid_out`` ``[T, H, S, 512]`` / ``mid_lse``
  ``[T, H, S]`` (``S >= num_splits``); a merge launch combines them.
  ``num_splits == 1`` writes the output directly. Allocation-free and
  CUDA-graph safe when the scratch is caller-owned.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional, Sequence, Tuple, Union

import torch

from ...jit.cake_sparse_mla_sm120_dsv41_mixed import (
    cake_sparse_mla_sm120_dsv41_mixed_available,
    cake_sparse_mla_sm120_dsv41_mixed_manifest,
    gen_cake_sparse_mla_sm120_dsv41_mixed_module,
)
from ...utils import (
    register_custom_op,
    register_fake_op,
    supported_compute_capability,
)

_D_QK = 512
_D_V = 512

# Main (SWA) cache: DSV4_1 FP8 footer layout.
MAIN_BYTES_PER_TOKEN = 528
MAIN_DATA_BYTES = 512
MAIN_SCALE_BYTES = 16
MAIN_SCALE_GROUP = 32
# Extra (compressed) cache: V41_FP4 footer layout.
EXTRA_BYTES_PER_TOKEN = 288
EXTRA_DATA_BYTES = 256
EXTRA_SCALE_BYTES = 32
EXTRA_SCALE_GROUP = 16

# Numerics routes. "default" (the wrapper's construction default) resolves to the BF16 route:
# the DeepSeek-V4.1 formats are QAT-trained with the query kept at high precision, and the
# BF16 route is the one whose quantization points are not lower than FlashInfer's
# ``compute_precision="bf16"`` baseline. "nvfp4" is never valid for this format.
COMPUTE_PRECISIONS: Tuple[str, ...] = ("bf16", "fp8")
DEFAULT_COMPUTE_PRECISION = "bf16"
_PRECISION_CODES: Dict[str, int] = {"bf16": 0, "fp8": 1}


@dataclass(frozen=True)
class CakeDsv41MixedGeometry:
    """Kernel-family geometry the planners and the scratch layout depend on.

    Read from the generated manifest when the family is present in the tree;
    ``provisional`` marks the pre-export placeholder (the SM120 DSv4 NVFP4 decode
    geometry) used so the host side and its CPU tests stay runnable before the
    kernels land. Nothing is guessed at launch time: loading the kernel module
    requires the manifest.
    """

    heads_per_block: int
    candidates_per_chunk: int
    extra_candidates_per_chunk: int
    max_chunks_per_block: int
    head_counts: Tuple[int, ...]
    two_tile_head_counts: Tuple[int, ...]
    precisions: Tuple[str, ...]
    provisional: bool
    kernel_commit: Optional[str]


# TODO(mixed-cache kernel): provisional until the exporter writes the manifest; mirrors the
# mixed-cache kernel module's v1 constants (16-head CTA tiles, 64-candidate chunks for both
# caches, 16-chunk index table, head counts 8 and every multiple of 16 up to 128, BF16 route
# only, no two-tile instances).
PROVISIONAL_GEOMETRY = CakeDsv41MixedGeometry(
    heads_per_block=16,
    candidates_per_chunk=64,
    extra_candidates_per_chunk=64,
    max_chunks_per_block=16,
    head_counts=(8, 16, 32, 48, 64, 80, 96, 112, 128),
    two_tile_head_counts=(),
    precisions=("bf16",),
    provisional=True,
    kernel_commit=None,
)


@functools.cache
def kernel_geometry() -> CakeDsv41MixedGeometry:
    """Geometry of the exported family, or :data:`PROVISIONAL_GEOMETRY` before the export."""

    if not cake_sparse_mla_sm120_dsv41_mixed_available():
        return PROVISIONAL_GEOMETRY
    manifest = cake_sparse_mla_sm120_dsv41_mixed_manifest()
    if (
        int(manifest["main_bytes_per_token"]) != MAIN_BYTES_PER_TOKEN
        or int(manifest["extra_bytes_per_token"]) != EXTRA_BYTES_PER_TOKEN
    ):
        raise ValueError(
            "Cake SM120 DSv4.1 mixed-cache manifest describes "
            f"{manifest['main_bytes_per_token']} / {manifest['extra_bytes_per_token']} "
            f"bytes per token; this route serves {MAIN_BYTES_PER_TOKEN} / {EXTRA_BYTES_PER_TOKEN}"
        )
    precisions = tuple(str(p) for p in manifest["precisions"])
    unknown = sorted(set(precisions) - set(COMPUTE_PRECISIONS))
    if unknown or DEFAULT_COMPUTE_PRECISION not in precisions:
        raise ValueError(
            f"Cake SM120 DSv4.1 mixed-cache manifest exports precisions {precisions}; "
            f"expected a subset of {COMPUTE_PRECISIONS} containing {DEFAULT_COMPUTE_PRECISION!r}"
        )
    return CakeDsv41MixedGeometry(
        heads_per_block=int(manifest["heads_per_block"]),
        candidates_per_chunk=int(manifest["candidates_per_chunk"]),
        extra_candidates_per_chunk=int(
            manifest.get("extra_candidates_per_chunk", manifest["candidates_per_chunk"])
        ),
        max_chunks_per_block=int(manifest["max_chunks_per_block"]),
        head_counts=tuple(int(h) for h in manifest["head_counts"]),
        two_tile_head_counts=tuple(
            int(h) for h in manifest.get("two_tile_head_counts", ())
        ),
        precisions=precisions,
        provisional=False,
        kernel_commit=str(manifest.get("kernel_commit", "")) or None,
    )


@dataclass(frozen=True)
class CakeDsv41MixedVariant:
    """One exact variant of the exported family: ``code`` is the binding scalar, ``decode`` / ``merge`` name its bodies."""

    name: str
    code: int
    decode: str
    merge: str


@dataclass(frozen=True)
class CakeDsv41MixedDispatch:
    """The exported family's exact-variant dispatch (manifest ``dispatch`` / ``variants`` / ``decode_kernels``).

    Every variant is an exact render of the same kernel (bit-identical up to fp32
    re-association); the choice is a pure performance decision the kernel module
    fitted per card (SM count: 170 = RTX 5090, 188 = RTX PRO 6000, 48 with unified
    memory = GB10), cache layout, head count and token bucket, with separate cells
    for ragged rows (per-token lengths) and a fallback for toolchains without the
    direct BF16x2 converts (CUDA < 13.2).  ``pow2_page_only_parts`` are the variant
    parts that resolve pages with a shift and a mask: the host drops them from the
    dispatched label unless every page size in use is a power of two, and the export
    carries the stripped body next to every such cell.  ``provisional`` carries only
    the plain render (before the export).
    """

    token_buckets: Tuple[Tuple[Optional[int], str], ...]
    unified_memory_sm_count: int
    rules: Dict[Tuple[int, bool, int, bool], Dict[str, str]]
    fallback_dual_two_tile_variant: str
    pow2_page_only_parts: Tuple[str, ...]
    variants: Dict[str, CakeDsv41MixedVariant]
    decode_tiles: Dict[Tuple[int, bool, str], int]
    merge_heads_per_cta: Dict[Tuple[int, str], int]
    provisional: bool


DEFAULT_VARIANT = "default"


def _provisional_dispatch(geometry: CakeDsv41MixedGeometry) -> CakeDsv41MixedDispatch:
    tiles = {
        (h, dual, DEFAULT_VARIANT): (2 if h in geometry.two_tile_head_counts else 1)
        for h in geometry.head_counts
        for dual in (False, True)
    }
    return CakeDsv41MixedDispatch(
        token_buckets=((None, "t128"),),
        unified_memory_sm_count=48,
        rules={},
        fallback_dual_two_tile_variant=DEFAULT_VARIANT,
        pow2_page_only_parts=(),
        variants={
            DEFAULT_VARIANT: CakeDsv41MixedVariant(
                DEFAULT_VARIANT, 0, DEFAULT_VARIANT, DEFAULT_VARIANT
            )
        },
        decode_tiles=tiles,
        merge_heads_per_cta={(h, DEFAULT_VARIANT): 1 for h in geometry.head_counts},
        provisional=True,
    )


def dispatch_from_manifest(manifest: dict) -> CakeDsv41MixedDispatch:
    """Build the dispatch tables from a manifest (pure; the CPU tests feed synthetic manifests)."""

    missing = [
        key
        for key in ("dispatch", "variants", "decode_kernels", "merge_kernels")
        if key not in manifest
    ]
    if missing:
        raise ValueError(
            f"Cake SM120 DSv4.1 mixed-cache manifest lacks {missing}: it predates the exact-variant export; "
            "regenerate the family with the kernel exporter"
        )
    d = manifest["dispatch"]
    if "pow2_page_only_parts" not in d:
        raise ValueError(
            "Cake SM120 DSv4.1 mixed-cache manifest lacks dispatch.pow2_page_only_parts: it predates the "
            "page-size-aware export; regenerate the family with the kernel exporter"
        )
    pow2_parts = tuple(str(p) for p in d["pow2_page_only_parts"])
    rules: Dict[Tuple[int, bool, int, bool], Dict[str, str]] = {}
    for rule in d["rules"]:
        key = (
            int(rule["sm_count"]),
            bool(rule["dual"]),
            int(rule["heads"]),
            bool(rule["ragged"]),
        )
        if key in rules:
            raise ValueError(f"duplicate dispatch rule {key}")
        rules[key] = {str(b): str(v) for b, v in rule["buckets"].items()}
    variants = {
        str(v["name"]): CakeDsv41MixedVariant(
            str(v["name"]), int(v["code"]), str(v["decode"]), str(v["merge"])
        )
        for v in manifest["variants"]
    }
    codes = sorted(v.code for v in variants.values())
    if codes != list(range(len(variants))) or DEFAULT_VARIANT not in variants:
        raise ValueError(
            "variant codes must be dense from 0 and include the plain render"
        )
    decode_tiles = {
        (int(k["heads"]), bool(k["dual"]), str(k["variant"])): int(k["tiles"])
        for k in manifest["decode_kernels"]
    }
    merge_hpc = {
        (int(k["heads"]), str(k["variant"])): int(k["heads_per_cta"])
        for k in manifest["merge_kernels"]
    }

    def has_bodies(heads: int, dual: bool, label: str) -> bool:
        v = variants.get(label)
        return (
            v is not None
            and (heads, dual, v.decode) in decode_tiles
            and (heads, v.merge) in merge_hpc
        )

    for key, buckets in rules.items():
        for bucket, label in buckets.items():
            if not has_bodies(key[2], key[1], label):
                raise ValueError(
                    f"dispatch cell {key}/{bucket} -> {label!r} has no exported body"
                )
            fallback = _strip_pow2_only_parts(label, pow2_parts)
            if fallback != label and not has_bodies(key[2], key[1], fallback):
                raise ValueError(
                    f"dispatch cell {key}/{bucket} -> {label!r} has no exported body for its "
                    f"non-power-of-two page fallback {fallback!r}"
                )
    return CakeDsv41MixedDispatch(
        token_buckets=tuple(
            (None if hi is None else int(hi), str(name))
            for hi, name in d["token_buckets"]
        ),
        unified_memory_sm_count=int(d["unified_memory_sm_count"]),
        rules=rules,
        fallback_dual_two_tile_variant=str(d["fallback_dual_two_tile_variant"]),
        pow2_page_only_parts=pow2_parts,
        variants=variants,
        decode_tiles=decode_tiles,
        merge_heads_per_cta=merge_hpc,
        provisional=False,
    )


@functools.cache
def kernel_dispatch() -> CakeDsv41MixedDispatch:
    """Dispatch tables of the exported family, or the plain-render placeholder before the export."""

    if not cake_sparse_mla_sm120_dsv41_mixed_available():
        return _provisional_dispatch(kernel_geometry())
    return dispatch_from_manifest(cake_sparse_mla_sm120_dsv41_mixed_manifest())


def cake_sparse_mla_sm120_dsv41_mixed_format_info() -> dict:
    """Static facts of the Cake SM120 DSv4.1 mixed-cache route."""

    geometry = kernel_geometry()
    return {
        "query_dim": _D_QK,
        "value_dim": _D_V,
        "main_bytes_per_token": MAIN_BYTES_PER_TOKEN,
        "main_data_bytes": MAIN_DATA_BYTES,
        "main_scale_bytes": MAIN_SCALE_BYTES,
        "main_scale_group": MAIN_SCALE_GROUP,
        "extra_bytes_per_token": EXTRA_BYTES_PER_TOKEN,
        "extra_data_bytes": EXTRA_DATA_BYTES,
        "extra_scale_bytes": EXTRA_SCALE_BYTES,
        "extra_scale_group": EXTRA_SCALE_GROUP,
        "chunk_width": geometry.candidates_per_chunk,
        "extra_chunk_width": geometry.extra_candidates_per_chunk,
        "heads_per_block": geometry.heads_per_block,
        "max_chunks_per_block": geometry.max_chunks_per_block,
        "heads": geometry.head_counts,
        "two_tile_heads": geometry.two_tile_head_counts,
        "compute_precisions": geometry.precisions,
        "default_compute_precision": DEFAULT_COMPUTE_PRECISION,
        "variants": tuple(
            sorted(
                kernel_dispatch().variants,
                key=lambda n: kernel_dispatch().variants[n].code,
            )
        ),
        "runtime_page": True,
        "runtime_extra_page": True,
        "kernels_available": not geometry.provisional,
        "kernel_commit": geometry.kernel_commit,
    }


def cake_sparse_mla_sm120_dsv41_mixed_supported_heads() -> Tuple[int, ...]:
    return kernel_geometry().head_counts


def normalize_compute_precision(compute_precision: str) -> str:
    """Map the wrapper / API precision word to a route: ``"default"`` is the BF16 route."""

    if compute_precision == "default":
        return DEFAULT_COMPUTE_PRECISION
    if compute_precision not in COMPUTE_PRECISIONS:
        raise ValueError(
            "backend='cake' with kv_cache_format='fp8_dsv41_fp4_ca' supports "
            f"compute_precision 'default', 'bf16' or 'fp8', got {compute_precision!r}"
        )
    return compute_precision


# ---------------------------------------------------------------------------
# Planners: pure functions of (num_tokens, num_heads, topk, extra_topk, num_sms, geometry).
# The split rules are the SM120 DSv4 NVFP4 decode rules re-measured for the mixed-cache family
# (RTX PRO 6000 split sweep: the inherited rule is within 0.5 % geomean of the best measured split);
# the head-tile count is fixed per head count by the exported kernels (see the manifest).
# ---------------------------------------------------------------------------


def cake_sparse_mla_sm120_dsv41_mixed_num_chunks(
    topk: int,
    extra_topk: int = 0,
    *,
    geometry: Optional[CakeDsv41MixedGeometry] = None,
) -> int:
    """Candidate chunks of one token (main chunks + extra chunks): the upper bound on ``num_splits``."""

    g = geometry or kernel_geometry()
    main = -(-int(topk) // g.candidates_per_chunk) if topk else 0
    extra = -(-int(extra_topk) // g.extra_candidates_per_chunk) if extra_topk else 0
    return main + extra


def cake_sparse_mla_sm120_dsv41_mixed_plan_head_tiles(
    *,
    num_tokens: int,
    num_heads: int,
    topk: int,
    extra_topk: int = 0,
    num_sms: int,
    geometry: Optional[CakeDsv41MixedGeometry] = None,
) -> int:
    """Return the number of 16-head tiles per decode CTA (1 or 2).

    The mixed-cache family compiles one CTA shape per head count: two tiles
    (``2 * heads_per_block`` heads sharing one candidate gather, 32-candidate
    stages) for the head counts listed in ``geometry.two_tile_head_counts``
    (every head count from 32 up), one tile otherwise.  The choice is fixed by
    the exported kernels, so the token count / chunk count do not enter; the
    arguments stay for signature compatibility with the NVFP4 route.
    """

    g = geometry or kernel_geometry()
    num_heads = int(num_heads)
    if int(num_sms) < 1:
        raise ValueError(f"num_sms must be positive, got {num_sms}")
    return 2 if num_heads in g.two_tile_head_counts else 1


def cake_sparse_mla_sm120_dsv41_mixed_token_bucket(
    num_tokens: int, *, dispatch: Optional[CakeDsv41MixedDispatch] = None
) -> str:
    """The token bucket of a decode call (the nearest measured representative row: T = 1 | 8 | 32 | 64 | 128)."""

    d = dispatch or kernel_dispatch()
    for hi, name in d.token_buckets:
        if hi is None or int(num_tokens) <= hi:
            return name
    raise ValueError("token buckets must end in an open bucket")


def _is_pow2(value: int) -> bool:
    return int(value) > 0 and (int(value) & (int(value) - 1)) == 0


def _strip_pow2_only_parts(label: str, parts: Sequence[str]) -> str:
    """Drop the shift-and-mask page-resolve parts from a variant label (``default`` when nothing is left)."""

    if label == DEFAULT_VARIANT or not parts:
        return label
    kept = [p for p in label.split("+") if p not in parts]
    return "+".join(kept) if kept else DEFAULT_VARIANT


def cake_sparse_mla_sm120_dsv41_mixed_plan_variant(
    *,
    num_tokens: int,
    num_heads: int,
    dual: bool,
    num_sms: int,
    unified_memory: bool,
    ragged: bool,
    direct_cvt: bool = True,
    pow2_pages: bool = True,
    dispatch: Optional[CakeDsv41MixedDispatch] = None,
    geometry: Optional[CakeDsv41MixedGeometry] = None,
) -> str:
    """Return the exact variant the host launches for one decode call (``"default"`` = the plain render).

    Mirrors the kernel module's ``default_variant`` + ``strip_pow2_only_parts``: the
    dispatch is keyed by the SM count (170 = RTX 5090, 188 = RTX PRO 6000, 48 with
    unified memory = GB10), the cache layout (``dual`` = main + compressed cache), the
    head count and the token bucket; ``ragged`` (per-token ``topk_length`` /
    ``extra_topk_length`` given) consults the ragged cells first; ``direct_cvt`` is
    the toolchain probe -- without the direct BF16x2 converts (CUDA < 13.2)
    discrete-memory cards run the fallback variant on dual two-tile head counts and
    the plain render elsewhere (the GB10 was measured on CUDA >= 13.2 only).  Any
    other card, or a unified-memory part that is not the GB10, keeps the plain render.
    ``pow2_pages`` says whether every page size in use (main, and the compressed cache
    when ``dual``) is a power of two; otherwise the shift-and-mask page-resolve parts
    (``pw``) are dropped from the label and the same variant's generic-division body runs.
    """

    d = dispatch or kernel_dispatch()
    g = geometry or kernel_geometry()
    label = _table_variant(
        num_tokens=num_tokens,
        num_heads=num_heads,
        dual=dual,
        num_sms=num_sms,
        unified_memory=unified_memory,
        ragged=ragged,
        direct_cvt=direct_cvt,
        dispatch=d,
        geometry=g,
    )
    return (
        label if pow2_pages else _strip_pow2_only_parts(label, d.pow2_page_only_parts)
    )


def _table_variant(
    *,
    num_tokens: int,
    num_heads: int,
    dual: bool,
    num_sms: int,
    unified_memory: bool,
    ragged: bool,
    direct_cvt: bool,
    dispatch: CakeDsv41MixedDispatch,
    geometry: CakeDsv41MixedGeometry,
) -> str:
    """The dispatch tables' label before the page-size rule (``default_variant`` in the kernel module)."""

    d = dispatch
    g = geometry
    sms = int(num_sms)
    if sms < 1:
        raise ValueError(f"num_sms must be positive, got {num_sms}")
    if bool(unified_memory) != (sms == d.unified_memory_sm_count):
        return DEFAULT_VARIANT
    if not direct_cvt:
        if unified_memory:
            return DEFAULT_VARIANT
        two_tile = int(num_heads) in g.two_tile_head_counts
        return (
            d.fallback_dual_two_tile_variant if (dual and two_tile) else DEFAULT_VARIANT
        )
    bucket = cake_sparse_mla_sm120_dsv41_mixed_token_bucket(num_tokens, dispatch=d)
    key = (sms, bool(dual), int(num_heads))
    if ragged:
        cells = d.rules.get(key + (True,))
        if cells and bucket in cells:
            return cells[bucket]
    cells = d.rules.get(key + (False,))
    if not cells:
        return DEFAULT_VARIANT
    return cells.get(bucket, DEFAULT_VARIANT)


@functools.cache
def _direct_cvt_toolchain() -> bool:
    """True when the nvcc that builds the family accepts PTX ISA 9.2 (CUDA >= 13.2): the direct BF16x2 convert path."""

    from packaging.version import Version

    from ...jit.cpp_ext import get_cuda_version

    return get_cuda_version() >= Version("13.2")


def cake_sparse_mla_sm120_dsv41_mixed_plan_splits(
    *,
    num_tokens: int,
    num_heads: int,
    topk: int,
    extra_topk: int = 0,
    num_sms: int,
    max_splits: int = 16,
    head_tiles: int = 1,
    unified_memory: bool = False,
    ragged: bool = False,
    geometry: Optional[CakeDsv41MixedGeometry] = None,
) -> Tuple[int, int]:
    """Return ``(num_splits, chunks_per_block)`` for one decode call.

    Mirrors the kernel module's ``plan_splits`` (fitted to the Cake split
    sweeps on RTX PRO 6000 / RTX 5090 / GB10), with the grid counted in CTAs of
    ``heads_per_block * head_tiles`` heads:

    * one chunk never splits; two chunks split in two only on a discrete-memory
      card whose unsplit grid is below a tenth of the SMs (the halved serial
      chain wins 4-15 % there and loses 8-38 % on larger grids);
    * ``ragged`` (per-token ``topk_length`` / ``extra_topk_length`` given) is the
      hint the quarter-to-half-SM 10-chunk grids use to take four splits on the
      RTX PRO 6000 (the dense rows of the same grid keep three);
    * ``unified_memory`` (GB10, CC 12.1): split in two while the doubled grid
      stays within two thirds of the SMs, keep doubling while the grid stays
      within half the SMs, always with at least two chunks per CTA; a >= 8-chunk
      grid between one and 4/3 waves unsplit also runs in two splits;
    * otherwise: the fewest chunks per CTA whose split grid stays within ~1.15
      waves of the SMs; once the unsplit grid exceeds that, >= 8-chunk work
      runs in the smallest of two or three splits whose grid ends in a wave at
      least half full; everything else runs unsplit;
    * independently ``chunks_per_block <= max_chunks_per_block`` always holds
      (the CTA index table), so long candidate lists force
      ``num_splits >= ceil(chunks / max_chunks_per_block)``; a ``max_splits``
      below that raises.
    """

    g = geometry or kernel_geometry()
    hpb = g.heads_per_block
    max_cpb = g.max_chunks_per_block
    num_sms = int(num_sms)
    if num_sms < 1:
        raise ValueError(f"num_sms must be positive, got {num_sms}")
    if int(head_tiles) not in (1, 2):
        raise ValueError(f"head_tiles must be 1 or 2, got {head_tiles}")
    chunks = cake_sparse_mla_sm120_dsv41_mixed_num_chunks(topk, extra_topk, geometry=g)
    if chunks < 1:
        raise ValueError("topk (or extra_topk) must select at least one candidate")
    min_splits = -(-chunks // max_cpb)
    if min_splits > max_splits:
        raise ValueError(
            f"{chunks} chunks need at least {min_splits} splits (at most {max_cpb} "
            f"chunks per block), max_splits={max_splits}"
        )
    cta_heads = hpb * int(head_tiles)
    head_blocks = (int(num_heads) + cta_heads - 1) // cta_heads
    base_ctas = int(num_tokens) * head_blocks
    splits = min_splits
    if (
        chunks == 2
        and max_splits > 1
        and not unified_memory
        and base_ctas * 10 <= num_sms
    ):
        # A two-chunk row on a near-empty discrete-memory card halves its serial chain by splitting.
        splits = 2
    if chunks > 2 and max_splits > 1:
        if unified_memory:
            want = min_splits
            while want * 2 <= max_splits and -(-chunks // (want * 2)) >= 2:
                grid_cap = num_sms * 2 // 3 if want == 1 else num_sms // 2
                if base_ctas * want * 2 > grid_cap:
                    break
                want *= 2
            if (
                want == 1
                and chunks >= 8
                and num_sms < base_ctas
                and base_ctas * 2 <= num_sms * 8 // 3
            ):
                # An unsplit grid between one and 4/3 waves splits in two (the 2-2.67-wave grid wins 4 %).
                want = 2
            splits = want
        else:
            # Fewest chunks per CTA whose split grid still fits ~1.15 waves (the SM-level
            # chain is latency-bound: the longest CTAs that keep every SM busy win; a
            # 1.4-1.5-wave grid loses 8-23 %).
            grid_cap = num_sms * 23 // 20
            chosen = None
            for cpb in range(1, max_cpb + 1):
                s = -(-chunks // cpb)
                if s < min_splits or s > max_splits:
                    continue
                if base_ctas * s <= grid_cap:
                    chosen = s
                    break
            if chosen is not None:
                splits = chosen
            elif base_ctas >= num_sms and chunks >= 8:
                # Full grid: the smallest of 1 / 2 / 3 splits whose grid fills its waves to at
                # least 90 % on average (waves / ceil(waves)); one CTA per SM, so a grid of
                # 3.01 waves pays for four.
                for s in (1, 2, 3):
                    if max(splits, s) > max_splits:
                        break
                    full_waves = -(-base_ctas * s // num_sms)
                    if base_ctas * s * 10 >= 9 * num_sms * full_waves:
                        splits = max(splits, s)
                        break
            # The fewest-cpb-within-1.15-waves choice puts the 64-CTA T = 32 grids of 6 and 18
            # chunks and the 96-CTA 10-chunk grid onto 192 CTAs (1.02 / 1.13 waves), where they
            # lose 8-44 % to the neighbouring split count: two splits (128 CTAs) for 6 and >= 16
            # chunks when the unsplit grid is between a quarter and a half of the SMs; four
            # splits (3 chunks, ~2 waves) for 10 chunks when it is between a half and two thirds.
            if (
                num_sms // 4 <= base_ctas <= num_sms // 2
                and (chunks == 6 or chunks >= 16)
                and min_splits <= 2 <= max_splits
            ):
                splits = 2
            elif (
                num_sms // 2 < base_ctas <= num_sms * 2 // 3
                and chunks == 10
                and max_splits >= 4
            ):
                splits = 4
            # Ragged rows (per-token lengths) of the 10-chunk quarter-to-half-SM grid take four
            # splits where the quadrupled grid stays within 1.4 waves (188 SMs: 256 CTAs = 1.36
            # waves; 170 SMs: 1.51 waves keeps three): the short, uneven CTAs fill the second wave.
            if (
                ragged
                and chunks == 10
                and num_sms // 4 <= base_ctas <= num_sms // 2
                and max_splits >= 4
                and base_ctas * 4 * 5 <= num_sms * 7
            ):
                splits = 4
    cpb = -(-chunks // splits)
    splits = -(-chunks // cpb)
    return splits, cpb


def _is_unified_memory_device(device: torch.device) -> bool:
    """GB10 (CC 12.1) is the only unified-LPDDR5x SM12x part; the planner keys off it."""

    return tuple(torch.cuda.get_device_capability(device)) == (12, 1)


def _resolve_plan(
    *,
    num_tokens: int,
    num_heads: int,
    topk: int,
    extra_topk: int,
    dual: bool,
    ragged: bool,
    device: torch.device,
    num_splits: Optional[int],
    max_splits: int,
    head_tiles: Optional[int],
    variant: Optional[str],
    page_size: int,
    extra_page_size: int,
) -> Tuple[str, int, int, int]:
    """Resolve ``(variant, head_tiles, num_splits, chunks_per_block)`` like the kernel module's launcher.

    The exact variant comes from the dispatch tables (or the caller's ``variant`` pin);
    the page sizes decide whether the shift-and-mask page-resolve parts may run (a
    dispatched label drops them, a pinned one is rejected); its decode body fixes the
    tiles per CTA of the launch (``t1`` bodies use one tile where the plain render uses
    two); the split planner counts the plain render's grid like the kernel module's
    launcher.
    """

    g = kernel_geometry()
    d = kernel_dispatch()
    num_sms = _num_sms(device)
    unified = _is_unified_memory_device(device)
    pow2_pages = _is_pow2(page_size) and (not dual or _is_pow2(extra_page_size))
    label = (
        str(variant)
        if variant is not None
        else cake_sparse_mla_sm120_dsv41_mixed_plan_variant(
            num_tokens=num_tokens,
            num_heads=num_heads,
            dual=dual,
            num_sms=num_sms,
            unified_memory=unified,
            ragged=ragged,
            direct_cvt=_direct_cvt_toolchain(),
            pow2_pages=pow2_pages,
            dispatch=d,
            geometry=g,
        )
    )
    v = d.variants.get(label)
    if v is None:
        raise ValueError(
            f"unknown exact variant {label!r}; this export carries {sorted(d.variants, key=lambda n: d.variants[n].code)}"
        )
    if (
        not pow2_pages
        and _strip_pow2_only_parts(label, d.pow2_page_only_parts) != label
    ):
        pages = f"page_size={page_size}" + (
            f", extra_page_size={extra_page_size}" if dual else ""
        )
        raise ValueError(
            f"exact variant {label!r} resolves pages with a shift and a mask "
            f"({' / '.join(p for p in label.split('+') if p in d.pow2_page_only_parts)}) and needs "
            f"power-of-two page sizes; got {pages}"
        )
    tiles = d.decode_tiles.get((int(num_heads), bool(dual), v.decode))
    if tiles is None or (int(num_heads), v.merge) not in d.merge_heads_per_cta:
        raise ValueError(
            f"exact variant {label!r} is not exported for {num_heads} heads with "
            f"{'dual' if dual else 'main-only'} caches"
        )
    if head_tiles is not None and int(head_tiles) != tiles:
        raise ValueError(
            f"head_tiles={head_tiles} is not valid for {num_heads} heads with variant {label!r}: its exported "
            f"decode body uses {tiles} tile(s) per CTA"
        )
    chunks = cake_sparse_mla_sm120_dsv41_mixed_num_chunks(topk, extra_topk, geometry=g)
    if num_splits is None:
        # The split planner was fitted on the plain render's grid and the kernel module keeps counting that grid
        # for every variant (a `t1` body launches twice the head blocks of the plan it was measured with).
        plain_tiles = cake_sparse_mla_sm120_dsv41_mixed_plan_head_tiles(
            num_tokens=num_tokens,
            num_heads=num_heads,
            topk=topk,
            extra_topk=extra_topk,
            num_sms=num_sms,
            geometry=g,
        )
        splits, cpb = cake_sparse_mla_sm120_dsv41_mixed_plan_splits(
            num_tokens=num_tokens,
            num_heads=num_heads,
            topk=topk,
            extra_topk=extra_topk,
            num_sms=num_sms,
            max_splits=max_splits,
            head_tiles=plain_tiles,
            unified_memory=unified,
            ragged=ragged,
            geometry=g,
        )
        return label, tiles, splits, cpb
    num_splits = int(num_splits)
    if num_splits < 1:
        raise ValueError(f"num_splits must be positive, got {num_splits}")
    cpb = -(-chunks // num_splits)
    if cpb > g.max_chunks_per_block:
        raise ValueError(
            f"num_splits={num_splits} leaves {cpb} chunks per CTA; the index table holds at most "
            f"{g.max_chunks_per_block}"
        )
    return label, tiles, -(-chunks // cpb), cpb


def cake_sparse_mla_sm120_dsv41_mixed_scratch_bytes(
    num_tokens: int, num_heads: int, topk: int, extra_topk: int = 0
) -> int:
    """Workspace bytes that cover every split plan of this shape (partials + LSE + alignment slack)."""

    chunks = cake_sparse_mla_sm120_dsv41_mixed_num_chunks(topk, extra_topk)
    rows = int(num_tokens) * int(num_heads)
    return rows * chunks * (_D_V * 2 + 4) + rows * 4 + 3 * 16


@functools.cache
def _num_sms(device: torch.device) -> int:
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


# ---------------------------------------------------------------------------
# Cache geometry.
# ---------------------------------------------------------------------------


def cache_page_geometry(
    shape: Sequence[int],
    strides: Sequence[int],
    *,
    bytes_per_token: int,
    name: str,
) -> Tuple[int, int, int]:
    """Validate a paged footer-layout cache view and return ``(num_pages, page_size, page_stride_bytes)``.

    Accepted views (uint8 element strides): 2-D ``[num_pages, page_bytes]`` with
    ``page_bytes`` a positive multiple of ``bytes_per_token``, 3-D ``[num_pages,
    page_size, bytes_per_token]``, HND ``[num_pages, 1, page_size,
    bytes_per_token]`` and NHD ``[num_pages, page_size, 1, bytes_per_token]``.
    Rows inside a page must stay packed (the footer layout interleaves the
    page's data rows and scale rows, so the last dimension is not a per-token
    record); the page stride may exceed the payload (padded / vLLM-packed
    pools) and must be a multiple of 16 bytes. Pure function of the view
    metadata so the rules are testable without a GPU.
    """

    shape = tuple(int(s) for s in shape)
    strides = tuple(int(s) for s in strides)
    if len(shape) != len(strides):
        raise ValueError(
            f"{name}: shape {shape} and strides {strides} disagree in rank"
        )
    if len(shape) == 2:
        num_pages, page_bytes = shape
        if page_bytes < bytes_per_token or page_bytes % bytes_per_token:
            raise ValueError(
                f"{name} 2-D view must be [num_pages, page_bytes] with page_bytes a positive "
                f"multiple of {bytes_per_token}, got shape={shape}"
            )
        if strides[1] != 1:
            raise ValueError(
                f"{name} byte dimension must be contiguous, got strides {strides}"
            )
        page_size = page_bytes // bytes_per_token
    elif len(shape) in (3, 4) and shape[-1] == bytes_per_token:
        if len(shape) == 3:
            page_dim = 1
        elif shape[1] == 1:
            page_dim = 2
        elif shape[2] == 1:
            page_dim = 1
        else:
            raise ValueError(
                f"{name} must have a singleton latent-head dimension at axis 1 or 2 "
                f"(HND or NHD), got shape={shape}"
            )
        num_pages, page_size = shape[0], shape[page_dim]
        if strides[-1] != 1 or strides[page_dim] != bytes_per_token:
            raise ValueError(
                f"{name} rows must stay packed inside each page (footer layout) with strides "
                f"(..., {bytes_per_token}, 1), got {strides}"
            )
    else:
        raise ValueError(
            f"{name} must be [num_pages, page_bytes], [num_pages, page_size, {bytes_per_token}], "
            f"HND [num_pages, 1, page_size, {bytes_per_token}] or NHD "
            f"[num_pages, page_size, 1, {bytes_per_token}], got shape={shape}"
        )
    if num_pages < 1 or page_size < 1:
        raise ValueError(f"{name} must hold at least one page with at least one row")
    page_stride = strides[0]
    if page_stride < page_size * bytes_per_token:
        raise ValueError(
            f"{name} page stride {page_stride} is smaller than the logical "
            f"{page_size * bytes_per_token}-byte page"
        )
    if page_stride % 16:
        raise ValueError(
            f"{name} page stride must be a multiple of 16 bytes, got {page_stride}"
        )
    return num_pages, page_size, page_stride


def _cache_geometry(
    cache: torch.Tensor, name: str, *, bytes_per_token: int
) -> Tuple[torch.Tensor, int, int]:
    """Flatten a paged cache view to ``(flat uint8 storage span, page_size, page_stride_bytes)``."""

    if not cache.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor, got {cache.device}")
    if cache.dtype != torch.uint8:
        raise ValueError(f"{name} must have dtype torch.uint8, got {cache.dtype}")
    num_pages, page_size, page_stride = cache_page_geometry(
        cache.shape, cache.stride(), bytes_per_token=bytes_per_token, name=name
    )
    if cache.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned")
    span = (num_pages - 1) * page_stride + page_size * bytes_per_token
    flat = cache.as_strided((span,), (1,), cache.storage_offset())
    return flat, page_size, page_stride


def _normalize_indices(
    indices: torch.Tensor, name: str, num_tokens: int
) -> torch.Tensor:
    if indices.ndim == 3 and indices.shape[1] == 1:
        indices = indices.squeeze(1)
    if indices.ndim != 2 or indices.dtype != torch.int32:
        raise ValueError(
            f"{name} must be a [num_tokens, topk] or [num_tokens, 1, topk] int32 tensor"
        )
    if indices.shape[0] != num_tokens:
        raise ValueError(f"{name} must have {num_tokens} rows, got {indices.shape[0]}")
    if indices.shape[1] < 1:
        raise ValueError(f"{name} must select at least one slot per token")
    return indices.contiguous()


def _normalize_length(
    length: Optional[torch.Tensor], name: str, num_tokens: int
) -> Optional[torch.Tensor]:
    if length is None:
        return None
    if length.ndim != 1 or length.dtype != torch.int32 or length.shape[0] < num_tokens:
        raise ValueError(
            f"{name} must be a 1-D int32 tensor with at least {num_tokens} entries"
        )
    return length.contiguous()


# ---------------------------------------------------------------------------
# Kernel module.
# ---------------------------------------------------------------------------

# Binding ABI of the generated TVM-FFI entry: the SM120 DSv4 NVFP4 family's parameter order without
# its ``head_tiles`` scalar (the tile count is a property of the exact variant's decode body), plus
# ``variant`` (the exact-variant code from the manifest's ``variants`` table) and ``precision``
# (0 = bf16 route). The exporter fills ``manifest["decode_params"]`` / ``manifest["merge_params"]``
# from the rendered kernel signatures and ``manifest["binding_params"]`` from its binding template;
# this list is the host-side expectation and is checked against the manifest when the module loads.
BINDING_PARAMS: Tuple[str, ...] = (
    "q",
    "kv_cache",
    "indices",
    "extra_kv_cache",
    "extra_indices",
    "topk_length",
    "extra_topk_length",
    "attn_sink",
    "output",
    "out_lse",
    "mid_out",
    "mid_lse",
    "page_size",
    "page_stride_bytes",
    "extra_page_size",
    "extra_page_stride_bytes",
    "num_splits",
    "chunks_per_block",
    "variant",
    "precision",
    "sm_scale",
    "lse_scale",
    "enable_pdl",
)


def _resolve_enable_pdl(enable_pdl: Optional[bool], device: torch.device) -> bool:
    """``None`` follows FlashInfer's device default (PDL on every CC >= 9.0 device); the split merge is
    launched with the Programmatic Dependent Launch attribute when the result is ``True``."""

    if enable_pdl is None:
        from ...utils import device_support_pdl

        return bool(device_support_pdl(device))
    return bool(enable_pdl)


@functools.cache
def get_cake_sparse_mla_sm120_dsv41_mixed_module():
    """Build and load the generated family; raises ``FileNotFoundError`` before the export."""

    manifest = cake_sparse_mla_sm120_dsv41_mixed_manifest()
    binding_params = tuple(manifest.get("binding_params", BINDING_PARAMS))
    if binding_params != BINDING_PARAMS:
        raise ValueError(
            "Cake SM120 DSv4.1 mixed-cache binding parameter list changed: manifest "
            f"{list(binding_params)} vs host {list(BINDING_PARAMS)}; update BINDING_PARAMS and "
            "the launch below together"
        )
    module = gen_cake_sparse_mla_sm120_dsv41_mixed_module().build_and_load()
    entry = getattr(module, manifest["entry"])

    def _decode(
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        extra_kv_cache: Optional[torch.Tensor],
        extra_indices: Optional[torch.Tensor],
        topk_length: Optional[torch.Tensor],
        extra_topk_length: Optional[torch.Tensor],
        attn_sink: Optional[torch.Tensor],
        output: torch.Tensor,
        out_lse: torch.Tensor,
        mid_out: Optional[torch.Tensor],
        mid_lse: Optional[torch.Tensor],
        page_size: int,
        page_stride_bytes: int,
        extra_page_size: int,
        extra_page_stride_bytes: int,
        num_splits: int,
        chunks_per_block: int,
        variant: int,
        precision: int,
        sm_scale: float,
        lse_scale: float,
        enable_pdl: bool,
    ) -> None:
        entry(
            q,
            kv_cache,
            indices,
            extra_kv_cache,
            extra_indices,
            topk_length,
            extra_topk_length,
            attn_sink,
            output,
            out_lse,
            mid_out,
            mid_lse,
            page_size,
            page_stride_bytes,
            extra_page_size,
            extra_page_stride_bytes,
            num_splits,
            chunks_per_block,
            variant,
            precision,
            sm_scale,
            lse_scale,
            enable_pdl,
        )

    wrapper_params = tuple(inspect.signature(_decode).parameters)
    if wrapper_params != BINDING_PARAMS:
        raise ValueError(
            "Cake SM120 DSv4.1 mixed-cache decode wrapper drifted from BINDING_PARAMS: "
            f"{list(wrapper_params)} vs {list(BINDING_PARAMS)}"
        )
    _decode = register_custom_op(
        "flashinfer::cake_sparse_mla_sm120_dsv41_mixed_decode",
        mutates_args=("output", "out_lse", "mid_out", "mid_lse"),
    )(_decode)

    @register_fake_op("flashinfer::cake_sparse_mla_sm120_dsv41_mixed_decode")
    def _fake_decode(*_args, **_kwargs) -> None:
        return None

    return SimpleNamespace(decode=_decode, raw_decode=entry)


@supported_compute_capability([120, 121])
def cake_sparse_mla_sm120_dsv41_mixed_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    out_lse: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    mid_out: Optional[torch.Tensor] = None,
    mid_lse: Optional[torch.Tensor] = None,
    lse_scale: float = 1.0,
    num_splits: Optional[int] = None,
    max_splits: int = 16,
    head_tiles: Optional[int] = None,
    variant: Optional[str] = None,
    compute_precision: str = DEFAULT_COMPUTE_PRECISION,
    enable_pdl: Optional[bool] = None,
) -> Dict[str, Union[int, str]]:
    """Run the allocation-free Cake SM120 DSv4.1 mixed-cache sparse-MLA decode.

    ``kv_cache`` is the 528-byte FP8 main (SWA) pool and ``extra_kv_cache`` the
    optional 288-byte V41_FP4 compressed pool, each with its own positive
    runtime page size and 16-byte-multiple page stride. Writes ``output`` and
    ``out_lse`` in place and returns the resolved plan ``{"variant", "head_tiles",
    "num_splits", "chunks_per_block", "precision"}``. ``mid_out`` / ``mid_lse``
    are required when the plan splits (``num_splits > 1``); size them with
    :func:`cake_sparse_mla_sm120_dsv41_mixed_num_chunks` splits to cover every
    plan. ``variant`` pins an exact variant of the export (every variant computes
    the same result; see :func:`cake_sparse_mla_sm120_dsv41_mixed_plan_variant`),
    ``head_tiles`` must match that variant's body and ``num_splits`` overrides
    the split planner.
    """

    precision = normalize_compute_precision(compute_precision)
    geometry = kernel_geometry()
    if precision not in geometry.precisions:
        raise ValueError(
            f"compute_precision={precision!r} is not exported by the Cake SM120 DSv4.1 "
            f"mixed-cache family (available: {geometry.precisions})"
        )
    if q.ndim != 3 or q.shape[-1] != _D_QK:
        raise ValueError(f"q must be [T, H, {_D_QK}], got {tuple(q.shape)}")
    if q.dtype != torch.bfloat16 or not q.is_cuda or not q.is_contiguous():
        raise ValueError("q must be a contiguous CUDA bfloat16 tensor")
    num_tokens, num_heads = int(q.shape[0]), int(q.shape[1])
    heads = cake_sparse_mla_sm120_dsv41_mixed_supported_heads()
    if num_heads not in heads:
        raise ValueError(
            f"Cake SM120 DSv4.1 mixed-cache sparse MLA supports {heads} query heads, got {num_heads}"
        )
    if (
        output.shape != q.shape
        or output.dtype != torch.bfloat16
        or not output.is_contiguous()
    ):
        raise ValueError(
            f"output must be a contiguous bfloat16 tensor of shape {tuple(q.shape)}"
        )
    if (
        out_lse.shape != (num_tokens, num_heads)
        or out_lse.dtype != torch.float32
        or not out_lse.is_contiguous()
    ):
        raise ValueError(
            f"out_lse must be a contiguous float32 tensor of shape {(num_tokens, num_heads)}"
        )
    if (extra_kv_cache is None) != (extra_indices is None):
        raise ValueError("extra_kv_cache and extra_indices must be provided together")
    if extra_topk_length is not None and extra_indices is None:
        raise ValueError("extra_topk_length requires extra_indices")
    indices = _normalize_indices(indices, "indices", num_tokens)
    topk = int(indices.shape[1])
    kv_flat, page_size, page_stride = _cache_geometry(
        kv_cache, "kv_cache", bytes_per_token=MAIN_BYTES_PER_TOKEN
    )
    extra_flat = None
    extra_page_size = 0
    extra_page_stride = 0
    extra_topk = 0
    if extra_kv_cache is not None:
        extra_indices = _normalize_indices(extra_indices, "extra_indices", num_tokens)
        extra_topk = int(extra_indices.shape[1])
        extra_flat, extra_page_size, extra_page_stride = _cache_geometry(
            extra_kv_cache, "extra_kv_cache", bytes_per_token=EXTRA_BYTES_PER_TOKEN
        )
    topk_length = _normalize_length(topk_length, "topk_length", num_tokens)
    extra_topk_length = _normalize_length(
        extra_topk_length, "extra_topk_length", num_tokens
    )
    if attn_sink is not None:
        if (
            attn_sink.ndim != 1
            or attn_sink.dtype != torch.float32
            or attn_sink.shape[0] < num_heads
        ):
            raise ValueError(
                f"attn_sink must be a 1-D float32 tensor with at least {num_heads} entries"
            )
        attn_sink = attn_sink.contiguous()
    label, ht, splits, cpb = _resolve_plan(
        num_tokens=num_tokens,
        num_heads=num_heads,
        topk=topk,
        extra_topk=extra_topk,
        dual=extra_flat is not None,
        ragged=topk_length is not None or extra_topk_length is not None,
        device=q.device,
        num_splits=num_splits,
        max_splits=max_splits,
        head_tiles=head_tiles,
        variant=variant,
        page_size=page_size,
        extra_page_size=extra_page_size,
    )
    if splits > 1:
        if mid_out is None or mid_lse is None:
            raise ValueError(
                f"this shape splits into {splits} CTAs per head block; pass caller-owned mid_out / mid_lse scratch"
            )
        if (
            mid_out.ndim != 4
            or mid_out.shape[0] != num_tokens
            or mid_out.shape[1] != num_heads
            or mid_out.shape[2] < splits
            or mid_out.shape[3] != _D_V
            or mid_out.dtype != torch.bfloat16
            or not mid_out.is_contiguous()
        ):
            raise ValueError(
                f"mid_out must be a contiguous bfloat16 [{num_tokens}, {num_heads}, >= {splits}, {_D_V}] tensor, "
                f"got {tuple(mid_out.shape)} {mid_out.dtype}"
            )
        if (
            mid_lse.ndim != 3
            or tuple(mid_lse.shape) != tuple(mid_out.shape[:3])
            or mid_lse.dtype != torch.float32
            or not mid_lse.is_contiguous()
        ):
            raise ValueError(
                f"mid_lse must be a contiguous float32 {tuple(mid_out.shape[:3])} tensor, got {tuple(mid_lse.shape)}"
            )
    else:
        mid_out = None
        mid_lse = None
    get_cake_sparse_mla_sm120_dsv41_mixed_module().decode(
        q,
        kv_flat,
        indices,
        extra_flat,
        extra_indices if extra_flat is not None else None,
        topk_length,
        extra_topk_length,
        attn_sink,
        output,
        out_lse,
        mid_out,
        mid_lse,
        page_size,
        page_stride,
        extra_page_size,
        extra_page_stride,
        splits,
        cpb,
        kernel_dispatch().variants[label].code,
        _PRECISION_CODES[precision],
        float(sm_scale),
        float(lse_scale),
        _resolve_enable_pdl(enable_pdl, q.device),
    )
    return {
        "variant": label,
        "head_tiles": ht,
        "num_splits": splits,
        "chunks_per_block": cpb,
        "precision": _PRECISION_CODES[precision],
    }


@supported_compute_capability([120, 121])
def _cake_dsv41_mixed_sparse_mla_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    lse_scale: float = 1.0,
    num_splits: Optional[int] = None,
    head_tiles: Optional[int] = None,
    variant: Optional[str] = None,
    compute_precision: str = DEFAULT_COMPUTE_PRECISION,
    enable_pdl: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocating convenience entry (tests / benchmarks): returns ``(output, out_lse)``."""

    if q.ndim != 3:
        raise ValueError(f"q must be [T, H, {_D_QK}], got {tuple(q.shape)}")
    num_tokens, num_heads = int(q.shape[0]), int(q.shape[1])
    topk = int(indices.shape[-1])
    extra_topk = int(extra_indices.shape[-1]) if extra_indices is not None else 0
    chunks = cake_sparse_mla_sm120_dsv41_mixed_num_chunks(topk, extra_topk)
    mid_out = torch.empty(
        (num_tokens, num_heads, chunks, _D_V), dtype=torch.bfloat16, device=q.device
    )
    mid_lse = torch.empty(
        (num_tokens, num_heads, chunks), dtype=torch.float32, device=q.device
    )
    output = torch.empty_like(q)
    out_lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=q.device)
    cake_sparse_mla_sm120_dsv41_mixed_decode(
        q,
        kv_cache,
        indices,
        output,
        out_lse,
        sm_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_kv_cache=extra_kv_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_topk_length,
        mid_out=mid_out,
        mid_lse=mid_lse,
        lse_scale=lse_scale,
        num_splits=num_splits,
        head_tiles=head_tiles,
        variant=variant,
        compute_precision=compute_precision,
        enable_pdl=enable_pdl,
    )
    return output, out_lse


def functional_run(
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    scale: float,
    *,
    lengths: Optional[torch.Tensor] = None,
    sink: Optional[torch.Tensor] = None,
    extra: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_lengths: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    lse_scale: float = 1.0,
    compute_precision: str = DEFAULT_COMPUTE_PRECISION,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Public-API route: carve the split scratch (and the LSE when the caller passes none) from ``workspace``."""

    from ._prepared import _workspace_tensor_view

    num_tokens, num_heads = int(q.shape[0]), int(q.shape[1])
    indices = _normalize_indices(indices, "indices", num_tokens)
    if extra_indices is not None:
        extra_indices = _normalize_indices(extra_indices, "extra_indices", num_tokens)
    topk = int(indices.shape[1])
    extra_topk = int(extra_indices.shape[1]) if extra_indices is not None else 0
    label, ht, splits, _ = _resolve_plan(
        num_tokens=num_tokens,
        num_heads=num_heads,
        topk=topk,
        extra_topk=extra_topk,
        dual=extra is not None,
        ragged=lengths is not None or extra_lengths is not None,
        device=q.device,
        num_splits=None,
        max_splits=16,
        head_tiles=None,
        variant=None,
        page_size=_cache_geometry(cache, "cache", bytes_per_token=MAIN_BYTES_PER_TOKEN)[
            1
        ],
        extra_page_size=(
            _cache_geometry(extra, "extra", bytes_per_token=EXTRA_BYTES_PER_TOKEN)[1]
            if extra is not None
            else 0
        ),
    )
    requirements: list[tuple[tuple[int, ...], torch.dtype]] = []
    if splits > 1:
        requirements.append(((num_tokens, num_heads, splits, _D_V), torch.bfloat16))
        requirements.append(((num_tokens, num_heads, splits), torch.float32))
    if lse is None:
        requirements.append(((num_tokens, num_heads), torch.float32))
    views: list[torch.Tensor] = []
    offset = 0
    for shape, dtype in requirements:
        view, offset = _workspace_tensor_view(
            workspace, byte_offset=offset, shape=shape, dtype=dtype, alignment=16
        )
        if view is None:
            raise ValueError(
                "attention workspace insufficient for the resolved Cake plan: need at least "
                f"{cake_sparse_mla_sm120_dsv41_mixed_scratch_bytes(num_tokens, num_heads, topk, extra_topk)} "
                f"bytes for {num_tokens} tokens x {num_heads} heads x topk {topk}"
                + (f" + extra topk {extra_topk}" if extra_topk else "")
            )
        views.append(view)
    mid_out = views[0] if splits > 1 else None
    mid_lse = views[1] if splits > 1 else None
    result = lse if lse is not None else views[-1]
    cake_sparse_mla_sm120_dsv41_mixed_decode(
        q,
        cache,
        indices,
        output,
        result,
        scale,
        topk_length=lengths,
        attn_sink=sink,
        extra_kv_cache=extra,
        extra_indices=extra_indices,
        extra_topk_length=extra_lengths,
        mid_out=mid_out,
        mid_lse=mid_lse,
        lse_scale=lse_scale,
        num_splits=splits,
        head_tiles=ht,
        variant=label,
        compute_precision=compute_precision,
        enable_pdl=enable_pdl,
    )
    return result


def _arena(
    wrapper, name: str, numel: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    arenas: Dict[str, torch.Tensor] = wrapper.__dict__.setdefault(
        "_cake_dsv41_arenas", {}
    )
    current = arenas.get(name)
    if current is None or current.numel() < numel or current.device != device:
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("warm up this attention shape before CUDA graph capture")
        current = torch.empty(max(numel, 1), dtype=dtype, device=device)
        arenas[name] = current
    return current


def wrapper_run(
    wrapper,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    out_lse: Optional[torch.Tensor] = None,
    mid_out: Optional[torch.Tensor] = None,
    mid_lse: Optional[torch.Tensor] = None,
    prefill_impl: Optional[str] = None,
    return_lse: bool = False,
    lse_scale: float = 1.0,
    compute_precision: str = "default",
    enable_pdl: Optional[bool] = None,
) -> Optional[torch.Tensor]:
    """``SparseMLASm120Wrapper.run`` for ``backend="cake"`` on the DSv4.1 mixed cache: wrapper-owned grow-only scratch."""

    if q.ndim == 4 and q.shape[1] == 1:
        q = q.squeeze(1)
    if output.ndim == 4 and output.shape[1] == 1:
        output = output.squeeze(1)
    if q.ndim != 3:
        raise ValueError("q must be [T,H,D] or [T,1,H,D]")
    if q.device != wrapper._device:
        raise ValueError("tensors must be on the Wrapper device")
    num_tokens, num_heads = int(q.shape[0]), int(q.shape[1])
    if (
        wrapper._max_num_tokens is not None and num_tokens > wrapper._max_num_tokens
    ) or (wrapper._max_num_heads is not None and num_heads > wrapper._max_num_heads):
        raise ValueError("query exceeds max_num_tokens/max_num_heads")
    if prefill_impl not in (None, "auto"):
        raise ValueError(
            "backend='cake' on the DSv4.1 mixed cache is a decode-only route; prefill_impl must be None or 'auto'"
        )
    precision = normalize_compute_precision(compute_precision)
    indices = _normalize_indices(indices, "indices", num_tokens)
    if extra_indices is not None:
        extra_indices = _normalize_indices(extra_indices, "extra_indices", num_tokens)
    topk = int(indices.shape[1])
    extra_topk = int(extra_indices.shape[1]) if extra_indices is not None else 0
    label, ht, splits, _ = _resolve_plan(
        num_tokens=num_tokens,
        num_heads=num_heads,
        topk=topk,
        extra_topk=extra_topk,
        dual=extra_kv_cache is not None,
        ragged=topk_length is not None or extra_topk_length is not None,
        device=q.device,
        num_splits=None,
        max_splits=16,
        head_tiles=None,
        variant=None,
        page_size=_cache_geometry(
            kv_cache, "kv_cache", bytes_per_token=MAIN_BYTES_PER_TOKEN
        )[1],
        extra_page_size=(
            _cache_geometry(
                extra_kv_cache, "extra_kv_cache", bytes_per_token=EXTRA_BYTES_PER_TOKEN
            )[1]
            if extra_kv_cache is not None
            else 0
        ),
    )
    if (mid_out is None) != (mid_lse is None):
        raise ValueError("mid_out and mid_lse must be provided together")
    if splits > 1 and mid_out is None:
        rows = num_tokens * num_heads * splits
        mid_out = _arena(wrapper, "mid_out", rows * _D_V, torch.bfloat16, q.device)[
            : rows * _D_V
        ].view(num_tokens, num_heads, splits, _D_V)
        mid_lse = _arena(wrapper, "mid_lse", rows, torch.float32, q.device)[:rows].view(
            num_tokens, num_heads, splits
        )
    if out_lse is None:
        lse = _arena(wrapper, "lse", num_tokens * num_heads, torch.float32, q.device)[
            : num_tokens * num_heads
        ].view(num_tokens, num_heads)
    else:
        lse = out_lse[:num_tokens, :num_heads]
        if not lse.is_contiguous():
            raise ValueError(
                "out_lse must be a contiguous [num_tokens, num_heads] float32 buffer"
            )
    cake_sparse_mla_sm120_dsv41_mixed_decode(
        q,
        kv_cache,
        indices,
        output,
        lse,
        sm_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_kv_cache=extra_kv_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_topk_length,
        mid_out=mid_out,
        mid_lse=mid_lse,
        lse_scale=lse_scale,
        num_splits=splits,
        head_tiles=ht,
        variant=label,
        compute_precision=precision,
        enable_pdl=enable_pdl,
    )
    return lse if return_lse else None


__all__ = [
    "BINDING_PARAMS",
    "COMPUTE_PRECISIONS",
    "CakeDsv41MixedDispatch",
    "CakeDsv41MixedGeometry",
    "CakeDsv41MixedVariant",
    "DEFAULT_COMPUTE_PRECISION",
    "DEFAULT_VARIANT",
    "EXTRA_BYTES_PER_TOKEN",
    "MAIN_BYTES_PER_TOKEN",
    "PROVISIONAL_GEOMETRY",
    "cache_page_geometry",
    "cake_sparse_mla_sm120_dsv41_mixed_decode",
    "cake_sparse_mla_sm120_dsv41_mixed_format_info",
    "cake_sparse_mla_sm120_dsv41_mixed_num_chunks",
    "cake_sparse_mla_sm120_dsv41_mixed_plan_head_tiles",
    "cake_sparse_mla_sm120_dsv41_mixed_plan_splits",
    "cake_sparse_mla_sm120_dsv41_mixed_plan_variant",
    "cake_sparse_mla_sm120_dsv41_mixed_scratch_bytes",
    "cake_sparse_mla_sm120_dsv41_mixed_supported_heads",
    "get_cake_sparse_mla_sm120_dsv41_mixed_module",
    "kernel_geometry",
    "normalize_compute_precision",
]
