"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Lossless compressed expert weights (WCX) for the SM90 BF16 decode GEMM units.

Code (version 1), per 32-element chunk of bf16 values:

* header: 1 byte, the largest biased exponent ``E`` of the 32 elements, or
  ``0xFF`` for an escaped (raw) chunk.  A chunk escapes when ``E == 255``
  (an inf or NaN is present) or when a nonzero element has ``E - e > 14``.
* element: 12 bits = sign (1) + 7-bit mantissa + 4-bit exponent offset
  ``o = E - e`` (0..14); offset code 15 is the zero code (+0 / -0, sign kept).
* data: 48 bytes per chunk = 32 bytes of ``sign << 7 | mantissa`` followed by
  16 bytes of offset nibbles (element ``2j`` in the low nibble of byte ``j``).
* escape: the raw chunk is 64 bytes (little-endian bf16); its first 48 bytes
  sit in the chunk's data position and its last 16 bytes in the next free
  16-byte escape slot of the stage (slots in ascending chunk order).

Stage layout: one stage is the B operand of one k-step of one GEMM tile, a
``[256 (n), 64 (k)]`` bf16 block (32 KiB), as 512 chunks in row-major
``(n, k / 32)`` order: ``[512 x 48 B data][512 header bytes][S x 16 B escape
slots]``.  ``S`` is fixed per weight tensor (the largest escape count of any
stage of that tensor, over all experts), so every stage of a tensor has the
same size and the GEMM computes stage addresses directly.  A tensor
``[E, N, K]`` is stored as ``[E, N / 256, K / 64, stage_bytes(S)]`` bytes.

Decoding is exact by construction: the decoded tile equals the bf16 tile bit
for bit, so the GEMM computes the same products in the same order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "WCX_BLOCK_K",
    "WCX_BLOCK_N",
    "WCX_CODEC_VERSION",
    "WCX_RING_SLOTS",
    "WCX_SLOT_STRIDE",
    "WCX_S_CAP",
    "WcxTensorStats",
    "wcx_decode_reference",
    "wcx_encode",
    "wcx_encode_reference",
    "wcx_stage_bytes",
    "wcx_stage_escape_counts",
]

WCX_CODEC_VERSION = 1
CHUNK = 32
ESC = 0xFF
ZERO_CODE = 15
MAX_OFF = 14
DATA_BYTES = 48
RAW_BYTES = 64
SLOT_BYTES = 16
WCX_BLOCK_N = 256
WCX_BLOCK_K = 64
CHUNKS_PER_STAGE = WCX_BLOCK_N * WCX_BLOCK_K // CHUNK  # 512
HEADER_OFFSET = CHUNKS_PER_STAGE * DATA_BYTES  # 24576
SLOT_OFFSET = HEADER_OFFSET + CHUNKS_PER_STAGE  # 25088
# The GEMM units keep WCX_RING_SLOTS compressed stages in shared memory, each in a
# slot of WCX_SLOT_STRIDE bytes; the escape-slot count of a tensor is therefore
# bounded by WCX_S_CAP.  A layer whose tensors need more escape slots keeps the
# bf16 path.
WCX_RING_SLOTS = 3
WCX_SLOT_STRIDE = 28160
WCX_S_CAP = (WCX_SLOT_STRIDE - SLOT_OFFSET) // SLOT_BYTES  # 192


def wcx_stage_bytes(slots: int) -> int:
    """Bytes of one compressed stage with ``slots`` escape slots."""
    if slots < 0 or slots > CHUNKS_PER_STAGE:
        raise ValueError(
            f"escape slots must be in [0, {CHUNKS_PER_STAGE}], got {slots}"
        )
    return SLOT_OFFSET + SLOT_BYTES * int(slots)


@dataclass(frozen=True)
class WcxTensorStats:
    """Encoding statistics of one ``[E, N, K]`` weight tensor."""

    slots: int  # escape slots per stage (S)
    stage_bytes: int
    escaped_chunks: int
    chunks: int
    compressed_bytes: int
    bf16_bytes: int

    @property
    def escape_rate(self) -> float:
        return self.escaped_chunks / self.chunks if self.chunks else 0.0

    @property
    def ratio(self) -> float:
        return self.compressed_bytes / self.bf16_bytes if self.bf16_bytes else 0.0


# ---------------------------------------------------------------- CPU reference
def _analyze(c: np.ndarray):
    e = ((c >> 7) & 0xFF).astype(np.int16)
    zero = (c & 0x7FFF) == 0
    exp_max = e.max(axis=1)
    off = exp_max[:, None] - e
    fits = zero | (off <= MAX_OFF)
    esc = (exp_max == 255) | ~fits.all(axis=1)
    return exp_max, off, zero, esc


def _encode_chunks(c: np.ndarray):
    """``c``: ``[n, 32]`` uint16 bf16 patterns -> data ``[n, 48]``, headers ``[n]``, tails ``[n_esc, 16]``, escape mask."""
    n = c.shape[0]
    exp_max, off, zero, esc = _analyze(c)
    smb = (((c >> 8) & 0x80) | (c & 0x7F)).astype(np.uint8)
    nib = np.where(zero, ZERO_CODE, off)
    nib = np.where(esc[:, None], 0, nib).astype(np.uint8)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
    data = np.concatenate([smb, packed], axis=1)
    raw = np.ascontiguousarray(c.astype("<u2")).view(np.uint8).reshape(n, RAW_BYTES)
    if esc.any():
        data[esc] = raw[esc, :DATA_BYTES]
    tails = raw[esc, DATA_BYTES:].copy()
    hdr = np.where(esc, ESC, exp_max).astype(np.uint8)
    return data, hdr, tails, esc


def _decode_chunks(data: np.ndarray, hdr: np.ndarray, tails: np.ndarray) -> np.ndarray:
    n = data.shape[0]
    esc = hdr == ESC
    smb = data[:, :32].astype(np.uint16)
    packed = data[:, 32:DATA_BYTES]
    nib = np.empty((n, CHUNK), np.int16)
    nib[:, 0::2] = packed & 0xF
    nib[:, 1::2] = packed >> 4
    zero = nib == ZERO_CODE
    e = hdr.astype(np.int16)[:, None] - nib
    e = np.where(zero | esc[:, None], 0, e).astype(np.uint16)
    out = (
        ((smb & 0x80) << 8) | (e << 7) | np.where(zero, 0, smb & 0x7F).astype(np.uint16)
    )
    out = out.astype(np.uint16)
    if esc.any():
        raw = np.concatenate([data[esc], tails], axis=1)
        out[esc] = np.ascontiguousarray(raw).view("<u2").reshape(-1, CHUNK)
    return out


def _stage_chunks_np(w: np.ndarray) -> np.ndarray:
    """``w``: ``[N, K]`` uint16 -> chunks ``[N/256 * K/64 * 512, 32]`` in stage order."""
    n, k = w.shape
    ns, ks = n // WCX_BLOCK_N, k // WCX_BLOCK_K
    st = w.reshape(ns, WCX_BLOCK_N, ks, WCX_BLOCK_K).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(st).reshape(ns * ks * CHUNKS_PER_STAGE, CHUNK)


def _check_shape(n: int, k: int) -> None:
    if n % WCX_BLOCK_N != 0 or k % WCX_BLOCK_K != 0:
        raise ValueError(
            f"WCX tensors need N % {WCX_BLOCK_N} == 0 and K % {WCX_BLOCK_K} == 0, got N={n}, K={k}"
        )


def wcx_encode_reference(w: np.ndarray, slots: int | None = None):
    """CPU reference encoder of one ``[E, N, K]`` (or ``[N, K]``) uint16 tensor.

    Returns ``(buf, stats)`` with ``buf`` uint8 ``[E, N/256, K/64, stage_bytes]``.
    ``slots`` (S) defaults to the largest stage escape count; a stage with more
    escapes than ``slots`` raises ``ValueError``.
    """
    w = np.asarray(w)
    if w.dtype != np.uint16:
        raise ValueError("wcx_encode_reference expects uint16 bf16 bit patterns")
    if w.ndim == 2:
        w = w[None]
    e_cnt, n, k = w.shape
    _check_shape(n, k)
    ns, ks = n // WCX_BLOCK_N, k // WCX_BLOCK_K
    nst = ns * ks
    per_expert = []
    s_need = 0
    for e in range(e_cnt):
        data, hdr, tails, esc = _encode_chunks(_stage_chunks_np(w[e]))
        esc2 = esc.reshape(nst, CHUNKS_PER_STAGE)
        cnt = esc2.sum(axis=1)
        s_need = max(s_need, int(cnt.max()) if nst else 0)
        per_expert.append((data, hdr, tails, esc2))
    if slots is None:
        slots = s_need
    if s_need > slots:
        raise ValueError(f"stage escape overflow: {s_need} > S={slots}")
    sb = wcx_stage_bytes(slots)
    buf = np.zeros((e_cnt, nst, sb), np.uint8)
    n_esc = 0
    for e, (data, hdr, tails, esc2) in enumerate(per_expert):
        b = buf[e]
        b[:, :HEADER_OFFSET] = data.reshape(nst, HEADER_OFFSET)
        b[:, HEADER_OFFSET:SLOT_OFFSET] = hdr.reshape(nst, CHUNKS_PER_STAGE)
        if esc2.any():
            st, ch = np.nonzero(esc2)
            sidx = (np.cumsum(esc2, axis=1) - 1)[st, ch]
            cols = (
                SLOT_OFFSET
                + sidx[:, None] * SLOT_BYTES
                + np.arange(SLOT_BYTES)[None, :]
            )
            b[st[:, None], cols] = tails
        n_esc += int(esc2.sum())
    stats = WcxTensorStats(
        slots=int(slots),
        stage_bytes=sb,
        escaped_chunks=n_esc,
        chunks=e_cnt * nst * CHUNKS_PER_STAGE,
        compressed_bytes=int(buf.size),
        bf16_bytes=2 * int(w.size),
    )
    return buf.reshape(e_cnt, ns, ks, sb), stats


def wcx_decode_reference(buf: np.ndarray, n: int, k: int) -> np.ndarray:
    """Inverse of :func:`wcx_encode_reference`: ``[E, NS, KS, sb]`` uint8 -> ``[E, N, K]`` uint16."""
    _check_shape(n, k)
    e_cnt = buf.shape[0]
    ns, ks = n // WCX_BLOCK_N, k // WCX_BLOCK_K
    nst = ns * ks
    out = np.empty((e_cnt, n, k), np.uint16)
    for e in range(e_cnt):
        b = np.asarray(buf[e]).reshape(nst, -1)
        data = b[:, :HEADER_OFFSET].reshape(nst * CHUNKS_PER_STAGE, DATA_BYTES)
        hdr = b[:, HEADER_OFFSET:SLOT_OFFSET].reshape(nst * CHUNKS_PER_STAGE)
        esc2 = (hdr == ESC).reshape(nst, CHUNKS_PER_STAGE)
        tails = np.zeros((0, SLOT_BYTES), np.uint8)
        if esc2.any():
            st, ch = np.nonzero(esc2)
            sidx = (np.cumsum(esc2, axis=1) - 1)[st, ch]
            cols = (
                SLOT_OFFSET
                + sidx[:, None] * SLOT_BYTES
                + np.arange(SLOT_BYTES)[None, :]
            )
            tails = b[st[:, None], cols]
        dec = _decode_chunks(data, hdr, tails)
        out[e] = (
            dec.reshape(ns, ks, WCX_BLOCK_N, WCX_BLOCK_K)
            .transpose(0, 2, 1, 3)
            .reshape(n, k)
        )
    return out


# ---------------------------------------------------------------- torch encoder
def _stage_chunks_t(w: torch.Tensor) -> torch.Tensor:
    """``[N, K]`` int16 -> int32 chunks ``[N/256 * K/64 * 512, 32]`` in stage order (bit patterns in 0..65535)."""
    n, k = w.shape
    ns, ks = n // WCX_BLOCK_N, k // WCX_BLOCK_K
    st = w.view(ns, WCX_BLOCK_N, ks, WCX_BLOCK_K).permute(0, 2, 1, 3).reshape(-1, CHUNK)
    return st.to(torch.int32) & 0xFFFF


def _escape_mask_t(c: torch.Tensor):
    e = (c >> 7) & 0xFF
    zero = (c & 0x7FFF) == 0
    exp_max = e.max(dim=1).values
    off = exp_max[:, None] - e
    fits = zero | (off <= MAX_OFF)
    esc = (exp_max == 255) | ~fits.all(dim=1)
    return zero, exp_max, off, esc


def wcx_stage_escape_counts(w: torch.Tensor) -> torch.Tensor:
    """Escape count per stage of one ``[N, K]`` bf16 / int16 tensor (int64, ``[N/256 * K/64]``)."""
    n, k = w.shape
    _check_shape(n, k)
    c = _stage_chunks_t(w.view(torch.int16))
    esc = _escape_mask_t(c)[3]
    return esc.view(-1, CHUNKS_PER_STAGE).sum(dim=1)


def _pack_expert_t(w: torch.Tensor, slots: int, out: torch.Tensor) -> int:
    """Pack one ``[N, K]`` int16 tensor into ``out`` (uint8 ``[NS * KS, stage_bytes]``); returns its escape count."""
    n, k = w.shape
    c = _stage_chunks_t(w)
    nst = c.shape[0] // CHUNKS_PER_STAGE
    zero, exp_max, off, esc = _escape_mask_t(c)
    smb = ((c >> 8) & 0x80) | (c & 0x7F)
    nib = torch.where(zero, torch.full_like(off, ZERO_CODE), off)
    nib = torch.where(esc[:, None], torch.zeros_like(nib), nib)
    packed = nib[:, 0::2] | (nib[:, 1::2] << 4)
    data = torch.cat([smb, packed], dim=1).to(torch.uint8)
    del smb, nib, packed, off
    raw = (
        w.view(n // WCX_BLOCK_N, WCX_BLOCK_N, k // WCX_BLOCK_K, WCX_BLOCK_K)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(torch.uint8)
        .view(-1, RAW_BYTES)
    )
    data[esc] = raw[esc, :DATA_BYTES]
    hdr = torch.where(esc, torch.full_like(exp_max, ESC), exp_max).to(torch.uint8)
    esc2 = esc.view(nst, CHUNKS_PER_STAGE)
    cnt = esc2.sum(dim=1)
    if nst and int(cnt.max()) > slots:
        raise ValueError(f"stage escape overflow: {int(cnt.max())} > S={slots}")
    out.zero_()
    out[:, :HEADER_OFFSET] = data.view(nst, HEADER_OFFSET)
    out[:, HEADER_OFFSET:SLOT_OFFSET] = hdr.view(nst, CHUNKS_PER_STAGE)
    n_esc = int(esc.sum())
    if n_esc:
        st, ch = torch.nonzero(esc2, as_tuple=True)
        sidx = (torch.cumsum(esc2.to(torch.int32), dim=1) - 1)[st, ch]
        cols = (
            SLOT_OFFSET
            + sidx[:, None].long() * SLOT_BYTES
            + torch.arange(SLOT_BYTES, device=w.device)[None, :]
        )
        out[st[:, None], cols] = raw[st * CHUNKS_PER_STAGE + ch][:, DATA_BYTES:]
    return n_esc


def wcx_encode(w: torch.Tensor, slots: int | None = None):
    """Encode one ``[E, N, K]`` bf16 tensor (any device) one expert at a time.

    Pass 1 finds ``S`` (the largest stage escape count over all experts) unless
    ``slots`` is given; pass 2 packs.  Returns ``(buf, stats)`` with ``buf``
    uint8 ``[E, N/256, K/64, stage_bytes(S)]`` on ``w``'s device, byte-identical
    to :func:`wcx_encode_reference`.
    """
    if w.ndim != 3 or w.dtype not in (torch.bfloat16, torch.int16):
        raise ValueError("wcx_encode expects a [E, N, K] bf16 tensor")
    e_cnt, n, k = w.shape
    _check_shape(n, k)
    w16 = w.contiguous().view(torch.int16)
    if slots is None:
        slots = 0
        for e in range(e_cnt):
            cnt = wcx_stage_escape_counts(w16[e])
            if cnt.numel():
                slots = max(slots, int(cnt.max()))
    sb = wcx_stage_bytes(slots)
    ns, ks = n // WCX_BLOCK_N, k // WCX_BLOCK_K
    buf = torch.empty((e_cnt, ns, ks, sb), dtype=torch.uint8, device=w.device)
    n_esc = 0
    for e in range(e_cnt):
        n_esc += _pack_expert_t(w16[e], slots, buf[e].view(ns * ks, sb))
    stats = WcxTensorStats(
        slots=int(slots),
        stage_bytes=sb,
        escaped_chunks=n_esc,
        chunks=e_cnt * ns * ks * CHUNKS_PER_STAGE,
        compressed_bytes=int(buf.numel()),
        bf16_bytes=2 * e_cnt * n * k,
    )
    return buf, stats


# ================================================================ codec version 2
# Fragment-major stages for the swap-AB register decode (W = the RS wgmma A operand).
#
# Same chunks as version 1 (32 elements = one W row x one 32-wide k half of a
# stage), same 12-bit element code and header, one more escape rule: a chunk
# that holds a zero (+0 / -0) escapes.  Version 2 has no zero code, so the
# register decode of a non-escaped element is (sign | mantissa) | (E - offset)
# << 7 with no zero mask, and E - offset never borrows across 16-bit halves.
#
# Math thread ``ct`` (0..255) of the decoding CTA = WG ``h``, warp ``w``,
# lane ``4 g + t``; group ``G = ct >> 2 = 32 h + 8 w + g`` (64 groups).  Its
# chunks ``q = 4 b + 2 rg + hk`` (8 per stage) are W row ``R(h, w, g, rg, b)``
# of the gate/up-interleaved n-tile (``b`` 0 = gate, 1 = up block; I column
# ``c = 64 h + 16 w + g + 8 rg``; row ``64 (c >> 5) + (c & 31) + 32 b``) and
# k half ``hk``.  The thread owns elements ``16 kk' + (2t, 2t+1, 2t+8, 2t+9)``
# (``kk' = 0, 1``) of each chunk = its RS A fragments of k16 steps
# ``kk = 2 hk + kk'``.
#
# Stage = [6 planes x 4096 B][512 B headers][64 B escape prefix][S x 16 B]:
# * plane ``kk`` (0..3), thread ``ct``: 4 words ``wd = 2 b + rg``, bytes ``e``
#   = sign/mantissa byte of element ``e`` (order 2t, 2t+1, 2t+8, 2t+9) of k16
#   step ``kk`` of chunk ``(b, rg, hk = kk >> 1)``;
# * plane ``4 + hk``, thread ``ct``: 4 words ``2 kk' + b``, nibble ``4 rg + e``
#   = offset code of that element;
# * header byte ``8 G + q``; prefix byte ``G`` = escaped chunks of groups < G;
# * escaped chunk: its raw bf16 pairs (e0, e1) of kk' = 0 / 1 sit in its two sm
#   words, its raw e2 (kk' = 0) in the rg half of nibble word (0, b) and raw e3
#   (kk' = 0) in the rg half of nibble word (1, b); word ``t`` of its escape
#   slot (slots in ascending ``8 G + q`` order) holds raw (e2, e3) of kk' = 1.
WCX2_CODEC_VERSION = 2
WCX2_PLANE_BYTES = 4096
WCX2_HDR_OFF = 6 * WCX2_PLANE_BYTES  # 24576
WCX2_PRE_OFF = WCX2_HDR_OFF + CHUNKS_PER_STAGE  # 25088
WCX2_SLOT_OFF = WCX2_PRE_OFF + 64  # 25152
WCX2_S_CAP = (WCX_SLOT_STRIDE - WCX2_SLOT_OFF) // SLOT_BYTES  # 188

__all__ += [
    "WCX2_CODEC_VERSION",
    "WCX2_S_CAP",
    "wcx2_decode_reference",
    "wcx2_encode",
    "wcx2_encode_reference",
    "wcx2_fragments_emulated",
    "wcx2_stage_bytes",
]


def wcx2_stage_bytes(slots: int) -> int:
    """Bytes of one version-2 stage with ``slots`` escape slots.

    Version 2 bounds ``slots`` by ``WCX2_S_CAP``: a stage must fit one ring slot of ``WCX_SLOT_STRIDE`` bytes, and the
    per-quad escape prefix entries are single bytes.  A tensor with more escaped chunks in a stage keeps the bf16 path.
    """
    if slots < 0 or slots > WCX2_S_CAP:
        raise ValueError(
            f"version-2 escape slots must be in [0, {WCX2_S_CAP}], got {slots}"
        )
    return WCX2_SLOT_OFF + SLOT_BYTES * int(slots)


def _v2_chunk_map():
    """chunk c2 = 8 G + q -> (stage row, k half) as int64 arrays of 512."""
    G = np.arange(64)[:, None]
    q = np.arange(8)[None, :]
    h, w, g = G >> 5, (G >> 3) & 3, G & 7
    b, rg, hk = q >> 2, (q >> 1) & 1, q & 1
    c = 64 * h + 16 * w + g + 8 * rg
    row = 64 * (c >> 5) + (c & 31) + 32 * b
    return row.reshape(-1).astype(np.int64), np.broadcast_to(hk, (64, 8)).reshape(
        -1
    ).astype(np.int64)


_V2_ROW, _V2_HK = _v2_chunk_map()
# thread ct -> (G, t); element e -> position in the 16-wide k16 step
_V2_CT = np.arange(256)
_V2_G, _V2_T = _V2_CT >> 2, _V2_CT & 3
_V2_POS = np.stack(
    [2 * _V2_T, 2 * _V2_T + 1, 2 * _V2_T + 8, 2 * _V2_T + 9], axis=1
)  # [256, 4]


def _analyze_v2(c: np.ndarray):
    exp_max, off, zero, esc = _analyze(c)
    esc = esc | zero.any(axis=1)
    return exp_max, off, zero, esc


def _v2_stage_tiles_np(w: np.ndarray) -> np.ndarray:
    """``[N, K]`` uint16 -> ``[NS * KS, 512 (c2), 32]`` v2-ordered chunks."""
    n, k = w.shape
    ns, ks = n // WCX_BLOCK_N, k // WCX_BLOCK_K
    st = (
        w.reshape(ns, WCX_BLOCK_N, ks, WCX_BLOCK_K)
        .transpose(0, 2, 1, 3)
        .reshape(ns * ks, WCX_BLOCK_N, WCX_BLOCK_K)
    )
    cols = 32 * _V2_HK[:, None] + np.arange(32)[None, :]
    return st[:, _V2_ROW[:, None], cols]


def _pack_v2_np(ch: np.ndarray, slots: int):
    """``ch``: ``[nst, 512, 32]`` uint16 v2-ordered chunks -> (``[nst, stage_bytes]`` uint8, escape mask ``[nst, 512]``)."""
    nst = ch.shape[0]
    flat = ch.reshape(-1, CHUNK)
    exp_max, off, zero, esc = _analyze_v2(flat)
    smb = (((flat >> 8) & 0x80) | (flat & 0x7F)).astype(np.uint32).reshape(nst, 512, 32)
    nib = np.where(zero, ZERO_CODE, off)
    nib = np.where(esc[:, None], 0, nib).astype(np.uint32).reshape(nst, 512, 32)
    raw = flat.astype(np.uint32).reshape(nst, 512, 32)
    esc = esc.reshape(nst, 512)
    hdr = np.where(esc.reshape(-1), ESC, exp_max).astype(np.uint8).reshape(nst, 512)
    cnt = esc.sum(axis=1)
    if nst and int(cnt.max()) > slots:
        raise ValueError(f"stage escape overflow: {int(cnt.max())} > S={slots}")
    sb = wcx2_stage_bytes(slots)
    out = np.zeros((nst, sb), np.uint8)
    G, T = _V2_G, _V2_T
    planes = np.zeros((nst, 6, 256, 4), np.uint32)
    for kk in range(4):
        hk, k1 = kk >> 1, kk & 1
        for b in range(2):
            for rg in range(2):
                c2 = 8 * G + 4 * b + 2 * rg + hk  # [256]
                pos = 16 * k1 + _V2_POS  # [256, 4]
                s = smb[:, c2[:, None], pos]  # [nst, 256, 4]
                word = (
                    s[..., 0] | (s[..., 1] << 8) | (s[..., 2] << 16) | (s[..., 3] << 24)
                )
                r = raw[:, c2[:, None], pos]
                rword = r[..., 0] | (r[..., 1] << 16)
                e = esc[:, c2]
                planes[:, kk, :, 2 * b + rg] = np.where(e, rword, word)
                nb = nib[:, c2[:, None], pos]
                half = (
                    nb[..., 0]
                    | (nb[..., 1] << 4)
                    | (nb[..., 2] << 8)
                    | (nb[..., 3] << 12)
                )
                half = np.where(e, 0, half)
                planes[:, 4 + hk, :, 2 * k1 + b] |= half << (16 * rg)
    for hk in range(2):
        for b in range(2):
            for rg in range(2):
                c2 = 8 * G + 4 * b + 2 * rg + hk
                e = esc[:, c2]
                r2 = raw[:, c2, 2 * T + 8]
                r3 = raw[:, c2, 2 * T + 9]
                planes[:, 4 + hk, :, b] |= np.where(e, r2, 0) << (16 * rg)
                planes[:, 4 + hk, :, 2 + b] |= np.where(e, r3, 0) << (16 * rg)
    out[:, :WCX2_HDR_OFF] = (
        planes.astype("<u4").view(np.uint8).reshape(nst, WCX2_HDR_OFF)
    )
    out[:, WCX2_HDR_OFF:WCX2_PRE_OFF] = hdr
    grp = esc.reshape(nst, 64, 8).sum(axis=2)
    out[:, WCX2_PRE_OFF:WCX2_SLOT_OFF] = (np.cumsum(grp, axis=1) - grp).astype(np.uint8)
    if esc.any():
        st, c2 = np.nonzero(esc)
        sidx = (np.cumsum(esc, axis=1) - 1)[st, c2]
        for t in range(4):
            tail = raw[st, c2, 16 + 2 * t + 8] | (raw[st, c2, 16 + 2 * t + 9] << 16)
            cols = (
                WCX2_SLOT_OFF
                + sidx[:, None] * SLOT_BYTES
                + 4 * t
                + np.arange(4)[None, :]
            )
            out[st[:, None], cols] = tail.astype("<u4").view(np.uint8).reshape(-1, 4)
    return out, esc


def wcx2_encode_reference(w: np.ndarray, slots: int | None = None):
    """CPU reference encoder (version 2) of one ``[E, N, K]`` (or ``[N, K]``) uint16 tensor.

    Returns ``(buf, stats)`` with ``buf`` uint8 ``[E, N/256, K/64, stage_bytes]``.
    """
    w = np.asarray(w)
    if w.dtype != np.uint16:
        raise ValueError("wcx2_encode_reference expects uint16 bf16 bit patterns")
    if w.ndim == 2:
        w = w[None]
    e_cnt, n, k = w.shape
    _check_shape(n, k)
    ns, ks = n // WCX_BLOCK_N, k // WCX_BLOCK_K
    nst = ns * ks
    tiles = [_v2_stage_tiles_np(w[e]) for e in range(e_cnt)]
    if slots is None:
        slots = 0
        for t in tiles:
            if nst:
                slots = max(
                    slots,
                    int(
                        _analyze_v2(t.reshape(-1, CHUNK))[3]
                        .reshape(nst, 512)
                        .sum(axis=1)
                        .max()
                    ),
                )
    sb = wcx2_stage_bytes(slots)
    buf = np.zeros((e_cnt, nst, sb), np.uint8)
    n_esc = 0
    for e in range(e_cnt):
        buf[e], esc = _pack_v2_np(tiles[e], slots)
        n_esc += int(esc.sum())
    stats = WcxTensorStats(
        slots=int(slots),
        stage_bytes=sb,
        escaped_chunks=n_esc,
        chunks=e_cnt * nst * CHUNKS_PER_STAGE,
        compressed_bytes=int(buf.size),
        bf16_bytes=2 * int(w.size),
    )
    return buf.reshape(e_cnt, ns, ks, sb), stats


def wcx2_fragments_emulated(stage: np.ndarray) -> np.ndarray:
    """Emulate the kernel's branch-free register decode of one v2 stage (uint8 ``[stage_bytes]``) with uint32 lane math.

    Returns ``[256 (ct), 4 (kk), 2 (b), 4 (reg)]`` uint32 RS A fragments: reg 0 = (row g, e0 e1), 1 = (row g + 8, e0 e1),
    2 = (row g, e2 e3), 3 = (row g + 8, e2 e3) -- the wgmma m64nNk16 A register order.
    """
    st = np.asarray(stage, np.uint8)
    planes = st[:WCX2_HDR_OFF].view("<u4").reshape(6, 256, 4).astype(np.uint64)
    hdr = st[WCX2_HDR_OFF:WCX2_PRE_OFF].astype(np.uint64)
    pre = st[WCX2_PRE_OFF:WCX2_SLOT_OFF].astype(np.uint64)
    M = np.uint64(0xFFFFFFFF)
    T = _V2_T.astype(np.uint64)
    hb = hdr.reshape(64, 8)[_V2_G]  # [256, 8] header bytes of the thread's chunks
    escq = hb == 0xFF  # [256, 8]
    out = np.zeros((256, 4, 2, 4), np.uint64)

    def prmt(a, b, sel):
        x = (a & M) | ((b & M) << np.uint64(32))
        r = np.zeros_like(a)
        for i in range(4):
            s = (sel >> (4 * i)) & 0xF
            byte = (x >> np.uint64(8 * (s & 7))) & np.uint64(0xFF)
            if s & 8:
                byte = np.where(byte & np.uint64(0x80), np.uint64(0xFF), np.uint64(0))
            r |= byte << np.uint64(8 * i)
        return r

    for kk in range(4):
        hk, k1 = kk >> 1, kk & 1
        for b in range(2):
            nw = planes[4 + hk, :, 2 * k1 + b]
            lo = nw & np.uint64(0x0F0F0F0F)
            hi = (nw >> np.uint64(4)) & np.uint64(0x0F0F0F0F)
            for rg in range(2):
                q = 4 * b + 2 * rg + hk
                E = hb[:, q]
                ex7 = (E * np.uint64(0x00800080)) & M
                smw = planes[kk, :, 2 * b + rg]
                esc = escq[:, q]
                for p in range(2):
                    smp = prmt(smw, np.zeros_like(smw), 0x1100 if p == 0 else 0x3322)
                    j = 2 * rg + p
                    x = prmt(lo, hi, 0x8080 | (j << 0) | ((4 + j) << 8))
                    e = (ex7 - x * np.uint64(128)) & M
                    dec = (smp & np.uint64(0x807F807F)) | e
                    if p == 0:
                        rawv = smw
                    elif k1 == 0:
                        rawv = prmt(
                            planes[4 + hk, :, b],
                            planes[4 + hk, :, 2 + b],
                            0x7632 if rg else 0x5410,
                        )
                    else:
                        m8 = (
                            escq[:, :q].sum(axis=1).astype(np.uint64)
                            if q
                            else np.zeros(256, np.uint64)
                        )
                        slot = pre[_V2_G] + m8
                        off = WCX2_SLOT_OFF + slot * np.uint64(16) + np.uint64(4) * T
                        off = np.where(esc, off, np.uint64(WCX2_SLOT_OFF))
                        safe = np.minimum(off, np.uint64(st.size - 4))
                        rawv = st.view(np.uint8)[
                            safe.astype(np.int64)[:, None] + np.arange(4)[None, :]
                        ]
                        rawv = rawv.astype(np.uint64)
                        rawv = (
                            rawv[:, 0]
                            | (rawv[:, 1] << np.uint64(8))
                            | (rawv[:, 2] << np.uint64(16))
                            | (rawv[:, 3] << np.uint64(24))
                        )
                    out[:, kk, b, 2 * p + rg] = np.where(esc, rawv, dec)
    return out.astype(np.uint32)


def wcx2_expected_fragments(tile: np.ndarray) -> np.ndarray:
    """``[256 (row), 64 (k)]`` uint16 stage tile -> the fragments the decoder must produce (same shape as the emulator)."""
    out = np.zeros((256, 4, 2, 4), np.uint32)
    for kk in range(4):
        for b in range(2):
            for rg in range(2):
                c2 = 8 * _V2_G + 4 * b + 2 * rg + (kk >> 1)
                row = _V2_ROW[c2]
                for p in range(2):
                    k0 = 16 * kk + _V2_POS[:, 2 * p]
                    out[:, kk, b, 2 * p + rg] = tile[row, k0].astype(np.uint32) | (
                        tile[row, k0 + 1].astype(np.uint32) << 16
                    )
    return out


def wcx2_decode_reference(buf: np.ndarray, n: int, k: int) -> np.ndarray:
    """Inverse of :func:`wcx2_encode_reference` through the register-decode emulator: ``[E, NS, KS, sb]`` -> ``[E, N, K]``."""
    _check_shape(n, k)
    e_cnt = buf.shape[0]
    ns, ks = n // WCX_BLOCK_N, k // WCX_BLOCK_K
    out = np.zeros((e_cnt, n, k), np.uint16)
    for e in range(e_cnt):
        for a in range(ns):
            for c in range(ks):
                fr = wcx2_fragments_emulated(np.asarray(buf[e, a, c]))
                tile = np.zeros((256, 64), np.uint16)
                for kk in range(4):
                    for b in range(2):
                        for rg in range(2):
                            c2 = 8 * _V2_G + 4 * b + 2 * rg + (kk >> 1)
                            row = _V2_ROW[c2]
                            for p in range(2):
                                k0 = 16 * kk + _V2_POS[:, 2 * p]
                                v = fr[:, kk, b, 2 * p + rg]
                                tile[row, k0] = (v & 0xFFFF).astype(np.uint16)
                                tile[row, k0 + 1] = (v >> 16).astype(np.uint16)
                out[e, a * 256 : (a + 1) * 256, c * 64 : (c + 1) * 64] = tile
    return out


# ---------------------------------------------------------------- torch encoder (version 2)
def _v2_index_t(device):
    row = torch.as_tensor(_V2_ROW, device=device)
    hk = torch.as_tensor(_V2_HK, device=device)
    G = torch.as_tensor(_V2_G, device=device)
    T = torch.as_tensor(_V2_T, device=device)
    pos = torch.as_tensor(_V2_POS, device=device)
    return row, hk, G, T, pos


def _pack_expert_v2_t(w: torch.Tensor, slots: int, out: torch.Tensor) -> int:
    """Pack one ``[N, K]`` int16 tensor into ``out`` (uint8 ``[NS * KS, wcx2_stage_bytes(slots)]``); returns its escapes."""
    n, k = w.shape
    ns, ks = n // WCX_BLOCK_N, k // WCX_BLOCK_K
    nst = ns * ks
    dev = w.device
    row, hk, G, T, pos = _v2_index_t(dev)
    st = (
        w.view(ns, WCX_BLOCK_N, ks, WCX_BLOCK_K)
        .permute(0, 2, 1, 3)
        .reshape(nst, WCX_BLOCK_N, WCX_BLOCK_K)
    )
    cols = 32 * hk[:, None] + torch.arange(32, device=dev)[None, :]
    ch = st[:, row[:, None], cols].to(torch.int64) & 0xFFFF  # [nst, 512, 32]
    flat = ch.reshape(-1, CHUNK)
    zero, exp_max, off, esc = _escape_mask_t(flat)
    esc = esc | zero.any(dim=1)
    smb = (((flat >> 8) & 0x80) | (flat & 0x7F)).reshape(nst, 512, 32)
    nib = torch.where(zero, torch.full_like(off, ZERO_CODE), off)
    nib = torch.where(esc[:, None], torch.zeros_like(nib), nib).reshape(nst, 512, 32)
    del zero, off
    esc2 = esc.reshape(nst, 512)
    cnt = esc2.sum(dim=1)
    if nst and int(cnt.max()) > slots:
        raise ValueError(f"stage escape overflow: {int(cnt.max())} > S={slots}")
    hdr = torch.where(esc, torch.full_like(exp_max, ESC), exp_max).reshape(nst, 512)
    planes = torch.zeros((nst, 6, 256, 4), dtype=torch.int64, device=dev)
    for kk in range(4):
        h_, k1 = kk >> 1, kk & 1
        p = 16 * k1 + pos  # [256, 4]
        for b in range(2):
            for rg in range(2):
                c2 = 8 * G + 4 * b + 2 * rg + h_
                s = smb[:, c2[:, None], p]
                word = (
                    s[..., 0] | (s[..., 1] << 8) | (s[..., 2] << 16) | (s[..., 3] << 24)
                )
                r = ch[:, c2[:, None], p]
                rword = r[..., 0] | (r[..., 1] << 16)
                e = esc2[:, c2]
                planes[:, kk, :, 2 * b + rg] = torch.where(e, rword, word)
                nb = nib[:, c2[:, None], p]
                half = (
                    nb[..., 0]
                    | (nb[..., 1] << 4)
                    | (nb[..., 2] << 8)
                    | (nb[..., 3] << 12)
                )
                half = torch.where(e, torch.zeros_like(half), half)
                planes[:, 4 + h_, :, 2 * k1 + b] |= half << (16 * rg)
    for h_ in range(2):
        for b in range(2):
            for rg in range(2):
                c2 = 8 * G + 4 * b + 2 * rg + h_
                e = esc2[:, c2]
                r2 = ch[:, c2, 2 * T + 8]
                r3 = ch[:, c2, 2 * T + 9]
                planes[:, 4 + h_, :, b] |= torch.where(e, r2, torch.zeros_like(r2)) << (
                    16 * rg
                )
                planes[:, 4 + h_, :, 2 + b] |= torch.where(
                    e, r3, torch.zeros_like(r3)
                ) << (16 * rg)
    pw = planes.reshape(nst, -1)
    by = torch.stack([(pw >> (8 * i)) & 0xFF for i in range(4)], dim=-1).reshape(
        nst, WCX2_HDR_OFF
    )
    out.zero_()
    out[:, :WCX2_HDR_OFF] = by.to(torch.uint8)
    out[:, WCX2_HDR_OFF:WCX2_PRE_OFF] = hdr.to(torch.uint8)
    grp = esc2.reshape(nst, 64, 8).sum(dim=2)
    out[:, WCX2_PRE_OFF:WCX2_SLOT_OFF] = (torch.cumsum(grp, dim=1) - grp).to(
        torch.uint8
    )
    n_esc = int(esc.sum())
    if n_esc:
        si, c2 = torch.nonzero(esc2, as_tuple=True)
        sidx = (torch.cumsum(esc2.to(torch.int64), dim=1) - 1)[si, c2]
        for t in range(4):
            tail = ch[si, c2, 16 + 2 * t + 8] | (ch[si, c2, 16 + 2 * t + 9] << 16)
            tb = torch.stack([(tail >> (8 * i)) & 0xFF for i in range(4)], dim=-1).to(
                torch.uint8
            )
            cols_ = (
                WCX2_SLOT_OFF
                + sidx[:, None] * SLOT_BYTES
                + 4 * t
                + torch.arange(4, device=dev)[None, :]
            )
            out[si[:, None], cols_] = tb
    return n_esc


def wcx2_stage_escape_counts(w: torch.Tensor) -> torch.Tensor:
    """Version-2 escape count per stage of one ``[N, K]`` bf16 / int16 tensor (int64, ``[N/256 * K/64]``)."""
    n, k = w.shape
    _check_shape(n, k)
    c = _stage_chunks_t(w.view(torch.int16))
    zero, exp_max, _, esc = _escape_mask_t(c)
    esc = esc | zero.any(dim=1)
    return esc.view(-1, CHUNKS_PER_STAGE).sum(dim=1)


def wcx2_encode(w: torch.Tensor, slots: int | None = None):
    """Version-2 encoder of one ``[E, N, K]`` bf16 tensor (any device), byte-identical to :func:`wcx2_encode_reference`."""
    if w.ndim != 3 or w.dtype not in (torch.bfloat16, torch.int16):
        raise ValueError("wcx2_encode expects a [E, N, K] bf16 tensor")
    e_cnt, n, k = w.shape
    _check_shape(n, k)
    w16 = w.contiguous().view(torch.int16)
    if slots is None:
        slots = 0
        for e in range(e_cnt):
            cnt = wcx2_stage_escape_counts(w16[e])
            if cnt.numel():
                slots = max(slots, int(cnt.max()))
    sb = wcx2_stage_bytes(slots)
    ns, ks = n // WCX_BLOCK_N, k // WCX_BLOCK_K
    buf = torch.empty((e_cnt, ns, ks, sb), dtype=torch.uint8, device=w.device)
    n_esc = 0
    for e in range(e_cnt):
        n_esc += _pack_expert_v2_t(w16[e], slots, buf[e].view(ns * ks, sb))
    stats = WcxTensorStats(
        slots=int(slots),
        stage_bytes=sb,
        escaped_chunks=n_esc,
        chunks=e_cnt * ns * ks * CHUNKS_PER_STAGE,
        compressed_bytes=int(buf.numel()),
        bf16_bytes=2 * e_cnt * n * k,
    )
    return buf, stats
