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

SM90 (Hopper: H100 / H200 / H20) fused MoE with NVFP4 expert weights and BF16
activations (W4A16), for LLM inference: decode, speculative verify and prefill.

Hopper has no FP4 tensor cores. The kernels decode the E2M1 codes and E4M3
block scales in registers into exact BF16 products and run BF16 tensor-core MMAs
with FP32 accumulation, so activations are never quantized. One call computes,
for every token ``t``,

    out[t] = sum_j w[t, j] * post(down_e(silu(gate_e(x[t])) * up_e(x[t]))),  e = ids[t, j]

with ``post`` the identity or, optionally, a per-route RMSNorm applied to each
expert output before the weighted sum (``post_norm_weight``).

Two kernels:

* small / medium batches (up to 1024 tokens per launch): one persistent launch
  runs gate/up, SwiGLU, down and the top-k reduction;
* large batches: a persistent expert-grouped kernel on CTA pairs with TMA
  multicast, plus a finalize kernel. By default it takes over from the token
  count given by :func:`sm90_nvfp4_w4a16_moe_prefill_min_tokens`.

Weights are repacked once by :func:`prepare_sm90_nvfp4_w4a16_weights` into the
layout the in-register decoder reads. The repack is lossless; it requires the
nonzero block scales of each (expert, gate / up / down) matrix to span a limited
range: always accepted up to a factor of 112, up to 224 depending on the largest
scale (see that function).

The kernels were adapted from gauravjain14/muse-okc-submissions (muse_moe_w4a16,
Apache-2.0), which builds on vjeux/okc-2026 and alygg77/okc-muse-spark-kernels
(Apache-2.0); see NOTICE.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from ..api_logging import flashinfer_api
from ..jit.sm90_nvfp4_w4a16_moe import (
    check_sm90_nvfp4_w4a16_moe_shape,
    gen_sm90_nvfp4_w4a16_moe_module,
)
from ..trace.templates.moe import sm90_nvfp4_w4a16_moe_trace
from ..utils import backend_requirement, supported_compute_capability

#: Bytes of one packed block (256 output rows x 256 reduction columns).
WEIGHT_BLOCK_BYTES = 35840
#: Experts one call can address (one bit per warp lane in the kernels).
MAX_LOCAL_EXPERTS = 32
#: Top-k slots of the kernels; smaller top-k is padded.
TOPK_SLOTS = 8
#: Token count one small-batch launch covers.
DECODE_MAX_TOKENS = 1024
#: Upper bound of the automatic switch to the large-batch kernel.
DEFAULT_PREFILL_MIN_TOKENS = 2049
DEFAULT_PREFILL_CHUNK_TOKENS = 8192

# Automatic kernel choice (measured on H200 over 16 shapes, uniform and skewed
# routing): the large-batch kernel is faster once each expert receives about 64
# routes and the call has enough work to fill every SM, about 2**33
# routes x hidden x intermediate multiply-adds per projection.
_AUTO_ROUTES_PER_EXPERT = 64
_AUTO_MIN_WORK = 1 << 33

# The 6-bit scale encoding holds E4M3 block scales 0 or in [2.0, 448].
_SCALE_MIN, _SCALE_MAX = 2.0, 448.0
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# ---------------------------------------------------------------------------
# Weight packing
# ---------------------------------------------------------------------------


def _decoded(pair: int) -> int:
    """The 16 FP4 codes (4 bits each) the kernels' register decoder reads from
    a stored 64-bit word pair (``dq4`` / ``decode`` in the CUDA sources)."""
    w0, w1 = pair & 0xFFFFFFFF, pair >> 32

    def rotl(x, n):
        return (x << n | x >> (32 - n)) & 0xFFFFFFFF

    # __byte_perm(w0, w1, 0x6341)
    f = (w0 >> 8 & 255) | (w1 & 255) << 8 | (w0 >> 24) << 16 | (w1 >> 16 & 255) << 24
    regs = (
        w0,
        w0 << 2 ^ f,
        w0 << 6,
        rotl(w0, 12),
        w1,
        w1 << 4,
        rotl(w1, 10),
        rotl(w1, 14) ^ f,
    )
    codes = 0
    for n in range(16):
        o = regs[n // 2] >> 16 * (n % 2)
        codes |= ((o >> 6 & 7) | (o >> 15 & 1) << 3) << 4 * n
    return codes


def _signed64(v: int) -> int:
    return v - (v >> 63 << 64)


@functools.cache
def _code_tables() -> Tuple[torch.Tensor, torch.Tensor]:
    """``enc[i, b]``: the stored word pair (int64) that decodes to byte ``b`` at
    code bits ``8i .. 8i + 7`` and 0 elsewhere; ``dec[i, b]``: the codes a pair
    holding byte ``b`` at bits ``8i .. 8i + 7`` (0 elsewhere) decodes to."""
    # the decoder is linear over GF(2): Gauss-Jordan on the 64 one-bit pairs
    rows = [[_decoded(1 << k), 1 << k] for k in range(64)]
    for i in range(64):
        p = next(r for r in rows[i:] if r[0] >> i & 1)
        rows.remove(p)
        rows.insert(i, p)
        for r in rows:
            if r is not p and r[0] >> i & 1:
                r[0] ^= p[0]
                r[1] ^= p[1]
    byte = torch.arange(256, dtype=torch.int64)
    enc = torch.zeros(8, 256, dtype=torch.int64)
    for k, (_, pair) in enumerate(rows):
        enc[k // 8] ^= (byte >> k % 8 & 1) * _signed64(pair)
    dec = torch.tensor(
        [[_signed64(_decoded(b << 8 * i)) for b in range(256)] for i in range(8)],
        dtype=torch.int64,
    )
    return enc, dec


def _gate_up_rows(rows: int, device) -> torch.Tensor:
    """Row order of the packed gate/up matrix: 8 gate rows, then 8 up rows."""
    j = torch.arange(rows // 16, device=device).view(-1, 1)
    r = torch.arange(16, device=device).view(1, 16)
    return torch.where(r < 8, 8 * j + r, rows // 2 + 8 * j + (r - 8)).flatten()


def _pack_matrix(packed: torch.Tensor, sf: torch.Tensor, gate_up: bool) -> torch.Tensor:
    """One expert matrix ``[R, K / 2]`` codes + ``[R, K / 16]`` scale bytes ->
    ``[K / 256, R / 256, 35840]`` (the layout of ``weight_block`` in common.cuh)."""
    R, K = packed.shape[0], packed.shape[1] * 2
    KC = K // 256
    dev = packed.device
    enc = _code_tables()[0].to(dev)
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(R, K)
    sb = sf
    # a zero scale zeroes its block (the 6-bit field of 0 decodes as 2.0)
    codes = codes.masked_fill((sb == 0).repeat_interleave(16, dim=1), 0)
    if gate_up:
        rows = _gate_up_rows(R, dev)
        codes, sb = codes[rows], sb[rows]
    c = (
        codes.view(R // 16, 2, 8, KC, 4, 4, 4, 2, 2)
        .permute(3, 0, 4, 2, 5, 6, 7, 1, 8)
        .reshape(-1, 16)
        .long()
    )
    pairs = c.new_zeros(c.shape[0])
    for i in range(8):
        pairs ^= enc[i][c[:, 2 * i] | c[:, 2 * i + 1] << 4]
    del c
    s = (
        sb.reshape(R // 16, 2, 8, KC, 4, 4)
        .permute(3, 0, 2, 5, 4, 1)
        .reshape(KC, R // 32, 2, 32, 4, 2)
        .long()
    )
    f = torch.where(s == 0, torch.zeros_like(s), s - 0x40)

    def P(rg, kk):
        return f[:, :, rg, :, kk, 0], f[:, :, rg, :, kk, 1]

    (al, ah), (bl, bh), (cl, ch), (dl, dh) = P(0, 0), P(1, 0), P(0, 1), P(1, 1)
    (el, eh), (fl, fh), (gl, gh), (hl, hh) = P(0, 2), P(1, 2), P(0, 3), P(1, 3)
    w0 = (
        (al << 4) | (ah << 20) | (bl << 10) | (bh << 26) | (gl & 15) | ((gh & 15) << 16)
    )
    w1 = (
        (cl << 4) | (ch << 20) | (dl << 10) | (dh << 26) | (hl & 15) | ((hh & 15) << 16)
    )
    w2 = (
        (el << 4)
        | (eh << 20)
        | (fl << 10)
        | (fh << 26)
        | (gl >> 4)
        | ((hl >> 4) << 2)
        | ((gh >> 4) << 16)
        | ((hh >> 4) << 18)
    )
    s6 = torch.stack((w0, w1, w2), -1).view(torch.int32).view(-1, 2)[:, 0]
    s6 = s6.contiguous().view(torch.uint8).view(KC, R // 256, 3072)
    return torch.cat([pairs.view(torch.uint8).view(KC, R // 256, 32768), s6], -1)


def _unpack_matrix(
    blocks: torch.Tensor, gate_up: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`_pack_matrix`: ``[K / 256, R / 256, 35840]`` -> codes
    ``[R, K]`` (int64) and scale bytes ``[R, K / 16]`` (int64)."""
    KC, RB = blocks.shape[0], blocks.shape[1]
    R, K = RB * 256, KC * 256
    dev = blocks.device
    dec = _code_tables()[1].to(dev)
    pairs = blocks[..., :32768].contiguous().view(torch.int64).reshape(-1)
    words = torch.zeros_like(pairs)
    for i in range(8):
        words ^= dec[i][(pairs >> (8 * i)) & 255]
    c = torch.stack([(words >> (4 * n)) & 15 for n in range(16)], -1)
    codes = (
        c.view(KC, R // 16, 4, 8, 4, 4, 2, 2, 2)
        .permute(1, 7, 3, 0, 2, 4, 5, 6, 8)
        .reshape(R, K)
    )
    w = blocks[..., 32768:].contiguous().view(torch.int32).long() & 0xFFFFFFFF
    w = w.view(KC, R // 32, 32, 3)
    w0, w1, w2 = w[..., 0], w[..., 1], w[..., 2]
    fields = {
        (0, 0): ((w0 >> 4) & 63, (w0 >> 20) & 63),
        (1, 0): ((w0 >> 10) & 63, (w0 >> 26) & 63),
        (0, 1): ((w1 >> 4) & 63, (w1 >> 20) & 63),
        (1, 1): ((w1 >> 10) & 63, (w1 >> 26) & 63),
        (0, 2): ((w2 >> 4) & 63, (w2 >> 20) & 63),
        (1, 2): ((w2 >> 10) & 63, (w2 >> 26) & 63),
        (0, 3): (
            (w0 & 15) | ((w2 & 3) << 4),
            ((w0 >> 16) & 15) | (((w2 >> 16) & 3) << 4),
        ),
        (1, 3): (
            (w1 & 15) | (((w2 >> 2) & 3) << 4),
            ((w1 >> 16) & 15) | (((w2 >> 18) & 3) << 4),
        ),
    }
    f = torch.empty(KC, R // 32, 2, 32, 4, 2, dtype=torch.int64, device=dev)
    for (rg, kk), (lo, hi) in fields.items():
        f[:, :, rg, :, kk, 0] = lo
        f[:, :, rg, :, kk, 1] = hi
    sb = (
        (f + 0x40)
        .view(KC, R // 16, 8, 4, 4, 2)
        .permute(1, 5, 2, 0, 4, 3)
        .reshape(R, K // 16)
    )
    if gate_up:
        inv = torch.empty(R, dtype=torch.long, device=dev)
        inv[_gate_up_rows(R, dev)] = torch.arange(R, device=dev)
        codes, sb = codes[inv], sb[inv]
    return codes, sb


def _e2m1_values(codes: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(_E2M1, dtype=torch.float32, device=codes.device)
    v = table[codes & 7]
    return torch.where((codes & 8).bool(), -v, v)


def _normalize_scales(
    sf: torch.Tensor, halves: int
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Scale the E4M3 block scales of one matrix (split into ``halves`` row ranges
    that each have their own global scale) by a power of two each, so that the
    largest becomes as large as E4M3 allows. Returns the scaled bytes, the factor
    of every range (to divide the global scale by) and the number of nonzero
    scales still below 2.0 (not representable in the packed format)."""
    v = sf.view(torch.float8_e4m3fn).float().view(halves, -1)
    if not bool(torch.isfinite(v).all()):
        raise ValueError("NVFP4 block scales must be finite")
    mx = v.amax(dim=1)
    k = torch.where(
        mx > 0, torch.floor(torch.log2(_SCALE_MAX / mx.clamp_min(1e-30))), 0.0
    )
    # guard the float log against off-by-one at exact powers of two
    k = torch.where(mx * torch.exp2(k) > _SCALE_MAX, k - 1, k)
    k = torch.where((mx > 0) & (mx * torch.exp2(k + 1) <= _SCALE_MAX), k + 1, k)
    factor = torch.exp2(k.clamp_min(0))
    scaled = v * factor[:, None]
    out = scaled.to(torch.float8_e4m3fn)
    if not torch.equal(out.float(), scaled):
        raise AssertionError("power-of-two scale shift is not exact")
    bad = int(((scaled != 0) & ((scaled < _SCALE_MIN) | (scaled > _SCALE_MAX))).sum())
    return out.view(torch.uint8).view(sf.shape), factor, bad


@dataclass(frozen=True)
class Sm90Nvfp4W4A16Weights:
    """Packed expert weights for :func:`sm90_nvfp4_w4a16_moe`.

    Attributes
    ----------
    w13 : torch.Tensor
        ``[E, hidden / 256, 2 * intermediate / 256, 35840]`` uint8, gate/up.
    w13_global_scale : torch.Tensor
        ``[2, E]`` float32: dequantization multipliers of the gate (row 0) and
        up (row 1) halves.
    w2 : torch.Tensor
        ``[E, intermediate / 256, hidden / 256, 35840]`` uint8, down.
    w2_global_scale : torch.Tensor
        ``[E]`` float32.
    """

    w13: torch.Tensor
    w13_global_scale: torch.Tensor
    w2: torch.Tensor
    w2_global_scale: torch.Tensor

    @property
    def num_experts(self) -> int:
        return self.w13.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.w13.shape[1] * 256

    @property
    def intermediate_size(self) -> int:
        return self.w13.shape[2] * 128


def _as_scale_bytes(t: torch.Tensor, name: str) -> torch.Tensor:
    if t.dtype == torch.float8_e4m3fn:
        return t.view(torch.uint8)
    if t.dtype == torch.uint8:
        return t
    raise ValueError(f"{name} must be float8_e4m3fn (or its uint8 view), got {t.dtype}")


@torch.no_grad()
def prepare_sm90_nvfp4_w4a16_weights(
    w13_fp4: torch.Tensor,
    w13_block_scale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_block_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
) -> Sm90Nvfp4W4A16Weights:
    r"""Repack NVFP4 expert weights for :func:`sm90_nvfp4_w4a16_moe`.

    The dequantized weight is ``e2m1(code) * e4m3(block_scale) * global_scale``
    with one E4M3 scale per 16 consecutive reduction columns (the ModelOpt NVFP4
    layout, scales not swizzled).

    Parameters
    ----------
    w13_fp4 : torch.Tensor
        ``[E, 2 * intermediate, hidden / 2]`` uint8, two E2M1 codes per byte (low
        nibble first). Rows ``[0, intermediate)`` are the gate projection,
        ``[intermediate, 2 * intermediate)`` the up projection.
    w13_block_scale : torch.Tensor
        ``[E, 2 * intermediate, hidden / 16]`` float8_e4m3fn (or uint8 view).
    w13_global_scale : torch.Tensor
        ``[E]`` float32 (gate and up share it) or ``[E, 2]`` (gate, up).
    w2_fp4 : torch.Tensor
        ``[E, hidden, intermediate / 2]`` uint8.
    w2_block_scale : torch.Tensor
        ``[E, hidden, intermediate / 16]`` float8_e4m3fn (or uint8 view).
    w2_global_scale : torch.Tensor
        ``[E]`` float32.

    Returns
    -------
    Sm90Nvfp4W4A16Weights
        The packed weights (on the input device).

    Notes
    -----
    The packed format stores block scales in 6 bits: E4M3 values 0 or in
    ``[2, 448]``. The block scales of each (expert, gate / up / down) matrix are
    first multiplied by the largest power of two that keeps their maximum at
    most 448, which brings it into ``(224, 448]`` (the global scale is divided
    by the same power, so the weights are unchanged). The repack is lossless
    when, after this shift, every nonzero block scale of the matrix is at least
    2; otherwise this function raises ``ValueError``. This always holds when the
    nonzero block scales of a matrix span at most a factor of 112, and for spans
    up to 224 it depends on the largest scale: block scales 448 and 2 are
    accepted, 256 and 1.25 (a span of 204.8) are not. Block scales must be
    non-negative.
    """
    if w13_fp4.dtype != torch.uint8 or w2_fp4.dtype != torch.uint8:
        raise ValueError("w13_fp4 and w2_fp4 must be uint8 (packed E2M1 pairs)")
    if w13_fp4.dim() != 3 or w2_fp4.dim() != 3:
        raise ValueError("w13_fp4 and w2_fp4 must be [E, rows, cols / 2]")
    E, two_i, h_half = w13_fp4.shape
    H, I = h_half * 2, two_i // 2
    if two_i % 2 != 0:
        raise ValueError("w13_fp4 must have an even number of rows (gate, up)")
    check_sm90_nvfp4_w4a16_moe_shape(H, I)
    if not 1 <= E <= MAX_LOCAL_EXPERTS:
        raise ValueError(f"at most {MAX_LOCAL_EXPERTS} experts per call, got {E}")
    if tuple(w2_fp4.shape) != (E, H, I // 2):
        raise ValueError(
            f"w2_fp4 must be [{E}, {H}, {I // 2}], got {tuple(w2_fp4.shape)}"
        )
    s13 = _as_scale_bytes(w13_block_scale, "w13_block_scale")
    s2 = _as_scale_bytes(w2_block_scale, "w2_block_scale")
    if tuple(s13.shape) != (E, 2 * I, H // 16) or tuple(s2.shape) != (E, H, I // 16):
        raise ValueError(
            "block scales must be [E, rows, cols / 16] (one per 16 columns)"
        )
    gs13 = w13_global_scale.float().reshape(E, -1)
    if gs13.shape[1] == 1:
        gs13 = gs13.expand(E, 2)
    if gs13.shape[1] != 2 or w2_global_scale.numel() != E:
        raise ValueError("w13_global_scale must be [E] or [E, 2], w2_global_scale [E]")
    gs2 = w2_global_scale.float().reshape(E)
    dev = w13_fp4.device
    w13 = torch.empty(
        (E, H // 256, 2 * I // 256, WEIGHT_BLOCK_BYTES), dtype=torch.uint8, device=dev
    )
    w2 = torch.empty(
        (E, I // 256, H // 256, WEIGHT_BLOCK_BYTES), dtype=torch.uint8, device=dev
    )
    gs13_out = torch.empty((2, E), dtype=torch.float32, device=dev)
    gs2_out = torch.empty((E,), dtype=torch.float32, device=dev)
    bad: List[str] = []
    for e in range(E):
        sf13, f13, n13 = _normalize_scales(s13[e].contiguous(), 2)
        sf2, f2, n2 = _normalize_scales(s2[e].contiguous(), 1)
        if n13 or n2:
            bad.append(f"expert {e}: {n13} w13 / {n2} w2 block scales")
            continue
        w13[e] = _pack_matrix(w13_fp4[e].contiguous(), sf13, gate_up=True)
        w2[e] = _pack_matrix(w2_fp4[e].contiguous(), sf2, gate_up=False)
        gs13_out[:, e] = gs13[e].to(dev) / f13.to(dev)
        gs2_out[e] = gs2[e].to(dev) / f2.to(dev)[0]
    if bad:
        raise ValueError(
            "NVFP4 block scales span too wide a range within a matrix for the 6-bit "
            "packed scales: after the power-of-two shift that brings the largest block "
            "scale of a matrix into (224, 448], every nonzero block scale must be at "
            "least 2. Block scales below that: " + "; ".join(bad[:8])
        )
    return Sm90Nvfp4W4A16Weights(w13, gs13_out.contiguous(), w2, gs2_out)


@torch.no_grad()
def dequantize_sm90_nvfp4_w4a16_weights(
    weights: Sm90Nvfp4W4A16Weights, expert: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FP32 weights of one expert recovered from the packed format:
    ``(w13 [2 * intermediate, hidden], w2 [hidden, intermediate])``. Intended for
    tests and debugging."""
    I = weights.intermediate_size
    c13, s13 = _unpack_matrix(weights.w13[expert], gate_up=True)
    c2, s2 = _unpack_matrix(weights.w2[expert], gate_up=False)

    def deq(codes, sbytes):
        sv = sbytes.to(torch.uint8).view(torch.float8_e4m3fn).float()
        return _e2m1_values(codes) * sv.repeat_interleave(16, dim=1)

    w13 = deq(c13, s13)
    w13[:I] *= weights.w13_global_scale[0, expert]
    w13[I:] *= weights.w13_global_scale[1, expert]
    w2 = deq(c2, s2) * weights.w2_global_scale[expert]
    return w13, w2


# ---------------------------------------------------------------------------
# Modules and workspace
# ---------------------------------------------------------------------------


@functools.cache
def _get_module(hidden_size: int, intermediate_size: int, post_norm: bool):
    return gen_sm90_nvfp4_w4a16_moe_module(
        hidden_size, intermediate_size, post_norm
    ).build_and_load()


class Sm90Nvfp4W4A16MoEWorkspace:
    """Persistent device state of :func:`sm90_nvfp4_w4a16_moe`.

    Holds the kernels' self-resetting counters (zeroed here, once) and, after
    :meth:`reserve_prefill`, the expert-grouped activation and intermediate
    buffers of the large-batch kernel and their tensor maps. A workspace must
    not be used by two calls that can run concurrently (different streams);
    create one per stream. Create it, and reserve prefill buffers, before CUDA
    graph capture.

    CUDA graphs that captured the large-batch kernel keep using the prefill
    buffers they were captured with. If a later call grows the reservation
    (a larger ``prefill_chunk_tokens``), the superseded buffers are therefore
    kept alive for the lifetime of the workspace instead of being freed;
    reserve the largest chunk size before capture to avoid holding both.
    """

    def __init__(
        self,
        device: torch.device,
        hidden_size: int,
        intermediate_size: int,
        post_norm: bool = False,
    ):
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # the buffer layout does not depend on post_norm; it only picks the module
        # (of the calls this workspace serves) that answers the size queries
        self._post_norm = post_norm
        module = _get_module(hidden_size, intermediate_size, post_norm)
        self.decode_counters = torch.zeros(
            module.decode_counter_words(), dtype=torch.int32, device=self.device
        )
        self.decode_scratch = torch.empty(
            module.decode_scratch_bytes(self.device.index),
            dtype=torch.uint8,
            device=self.device,
        )
        self.prefill_chunk_tokens = 0
        self.xt: Optional[torch.Tensor] = None
        self.h: Optional[torch.Tensor] = None
        self.maps: Optional[torch.Tensor] = None
        self.prefill_counters: Optional[torch.Tensor] = None
        # set when a CUDA graph captured the large-batch kernel with the current buffers
        self._prefill_captured = False
        # superseded buffers that captured graphs may still read and write
        self._retired_prefill: List[Tuple[torch.Tensor, ...]] = []

    def reserve_prefill(self, chunk_tokens: int = DEFAULT_PREFILL_CHUNK_TOKENS) -> None:
        """Allocate the large-batch buffers for chunks of ``chunk_tokens`` tokens
        (``(8 * chunk_tokens + 1152) * (hidden + intermediate) * 2`` bytes).

        Only grows: a smaller ``chunk_tokens`` keeps the current buffers. When a
        CUDA graph has captured the large-batch kernel with the current buffers,
        growing keeps them alive (the graph still uses them) next to the new
        ones."""
        if chunk_tokens <= 0 or chunk_tokens % 128 != 0:
            raise ValueError("chunk_tokens must be a positive multiple of 128")
        if self.prefill_chunk_tokens >= chunk_tokens:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "reserve the prefill workspace before CUDA graph capture"
            )
        module = _get_module(self.hidden_size, self.intermediate_size, self._post_norm)
        slots = module.prefill_slots(chunk_tokens)
        kw = dict(dtype=torch.bfloat16, device=self.device)
        # build the new reservation completely before replacing the current one, so a
        # failed allocation leaves the workspace as it was
        xt = torch.empty((slots, self.hidden_size), **kw)
        h = torch.empty((slots, self.intermediate_size), **kw)
        maps = torch.empty(
            module.prefill_map_bytes(), dtype=torch.uint8, device=self.device
        )
        counters = torch.zeros(
            module.prefill_counter_words(), dtype=torch.int32, device=self.device
        )
        module.prefill_init(xt, h, maps)
        if self._prefill_captured:
            self._retired_prefill.append(
                (self.xt, self.h, self.maps, self.prefill_counters)
            )
            self._prefill_captured = False
        self.xt, self.h, self.maps, self.prefill_counters = xt, h, maps, counters
        self.prefill_chunk_tokens = chunk_tokens


_DEFAULT_WORKSPACES: Dict[Tuple[int, int, int], Sm90Nvfp4W4A16MoEWorkspace] = {}


def get_sm90_nvfp4_w4a16_moe_workspace(
    device: torch.device,
    hidden_size: int,
    intermediate_size: int,
    post_norm: bool = False,
) -> Sm90Nvfp4W4A16MoEWorkspace:
    """The default workspace of a device and shape (shared by every call that
    passes no ``workspace``; see :class:`Sm90Nvfp4W4A16MoEWorkspace`)."""
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, hidden_size, intermediate_size)
    ws = _DEFAULT_WORKSPACES.get(key)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "create the sm90_nvfp4_w4a16_moe workspace before CUDA graph capture "
                "(one eager call, or get_sm90_nvfp4_w4a16_moe_workspace)"
            )
        ws = Sm90Nvfp4W4A16MoEWorkspace(
            torch.device("cuda", index), hidden_size, intermediate_size, post_norm
        )
        _DEFAULT_WORKSPACES[key] = ws
    return ws


@functools.lru_cache(maxsize=None)
def _decode_buffer_bytes(
    hidden_size: int, intermediate_size: int, post_norm: bool, num_tokens: int
) -> int:
    module = _get_module(hidden_size, intermediate_size, post_norm)
    return module.decode_buffer_bytes(num_tokens)


# ---------------------------------------------------------------------------
# Operation
# ---------------------------------------------------------------------------


def sm90_nvfp4_w4a16_moe_prefill_min_tokens(
    hidden_size: int,
    intermediate_size: int,
    num_local_experts: int,
    top_k: int,
    local_route_fraction: float = 1.0,
) -> int:
    """Token count from which :func:`sm90_nvfp4_w4a16_moe` uses the large-batch
    kernel when ``prefill_min_tokens`` is not given.

    The large-batch kernel wins once every expert receives about 64 routes and
    the call carries enough work to fill the GPU; below that the small-batch
    kernel is faster. The result is at most ``DEFAULT_PREFILL_MIN_TOKENS``.

    ``local_route_fraction`` is the expected share of routes that hit the local
    experts. The default 1.0 assumes every route is local; under expert
    parallelism, where routes to other ranks' experts are passed as skipped ids,
    pass ``num_local_experts / num_global_experts`` (for balanced routing) and
    give the result as ``prefill_min_tokens``.
    """
    if not 0.0 < local_route_fraction <= 1.0:
        raise ValueError("local_route_fraction must be in (0, 1]")
    routes_per_token = top_k * local_route_fraction
    by_experts = _AUTO_ROUTES_PER_EXPERT * num_local_experts / routes_per_token
    by_work = _AUTO_MIN_WORK / (routes_per_token * hidden_size * intermediate_size)
    return min(DEFAULT_PREFILL_MIN_TOKENS, math.ceil(max(by_experts, by_work)))


def _routing_inputs(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[M, k]`` routing -> int32 / float32 ``[M, 8]`` (no copy for int32 / float32 k = 8)."""
    M, k = topk_ids.shape
    ids, w = topk_ids, topk_weights
    if ids.dtype != torch.int32:
        # narrowing wraps (2**32 -> 0): clamp first so ids outside [0, E) stay outside
        ids = ids.clamp(-1, MAX_LOCAL_EXPERTS)
    if k < TOPK_SLOTS:
        ids = torch.nn.functional.pad(ids, (0, TOPK_SLOTS - k), value=-1)
        w = torch.nn.functional.pad(w, (0, TOPK_SLOTS - k), value=0.0)
    ids = ids.to(torch.int32).contiguous()
    w = w.to(torch.float32).contiguous()
    return ids, w


@supported_compute_capability([90])
def _check_sm90_nvfp4_w4a16_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_global_scale: torch.Tensor,
    post_norm_weight: Optional[torch.Tensor] = None,
    post_norm_eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[Sm90Nvfp4W4A16MoEWorkspace] = None,
    enable_pdl: bool = True,
    prefill_min_tokens: Optional[int] = None,
    prefill_chunk_tokens: int = DEFAULT_PREFILL_CHUNK_TOKENS,
) -> bool:
    if not hidden_states.is_cuda:
        raise ValueError("hidden_states must be a CUDA tensor")
    if hidden_states.dtype != torch.bfloat16 or hidden_states.dim() != 2:
        raise ValueError(
            "hidden_states must be a 2D bfloat16 tensor [num_tokens, hidden]"
        )
    if (
        w13.dtype != torch.uint8
        or w13.dim() != 4
        or w13.shape[-1] != WEIGHT_BLOCK_BYTES
    ):
        raise ValueError("w13 must be packed by prepare_sm90_nvfp4_w4a16_weights")
    if w2.dtype != torch.uint8 or w2.dim() != 4 or w2.shape[-1] != WEIGHT_BLOCK_BYTES:
        raise ValueError("w2 must be packed by prepare_sm90_nvfp4_w4a16_weights")
    E, H, I = w13.shape[0], w13.shape[1] * 256, w13.shape[2] * 128
    check_sm90_nvfp4_w4a16_moe_shape(H, I)
    if tuple(w2.shape[:3]) != (E, I // 256, H // 256):
        raise ValueError("w2 does not match w13 (experts, hidden, intermediate)")
    if not 1 <= E <= MAX_LOCAL_EXPERTS:
        raise ValueError(f"at most {MAX_LOCAL_EXPERTS} experts per call, got {E}")
    if hidden_states.shape[1] != H:
        raise ValueError(
            f"hidden_states has hidden size {hidden_states.shape[1]}, weights {H}"
        )
    M = hidden_states.shape[0]
    if (
        topk_ids.dim() != 2
        or topk_ids.shape[0] != M
        or not 1 <= topk_ids.shape[1] <= TOPK_SLOTS
    ):
        raise ValueError(
            f"topk_ids must be [num_tokens, top_k] with top_k <= {TOPK_SLOTS}"
        )
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must be int32 or int64")
    if (
        tuple(topk_weights.shape) != tuple(topk_ids.shape)
        or not topk_weights.is_floating_point()
    ):
        raise ValueError("topk_weights must be a floating tensor shaped like topk_ids")
    if (
        tuple(w13_global_scale.shape) != (2, E)
        or w13_global_scale.dtype != torch.float32
    ):
        raise ValueError("w13_global_scale must be float32 [2, E] (gate, up)")
    if tuple(w2_global_scale.shape) != (E,) or w2_global_scale.dtype != torch.float32:
        raise ValueError("w2_global_scale must be float32 [E]")
    if post_norm_weight is not None and tuple(post_norm_weight.shape) != (H,):
        raise ValueError(f"post_norm_weight must be [{H}]")
    if out is not None and (
        tuple(out.shape) != (M, H)
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must be a contiguous bfloat16 [num_tokens, hidden] tensor"
        )
    tensors = [
        hidden_states,
        topk_ids,
        topk_weights,
        w13,
        w13_global_scale,
        w2,
        w2_global_scale,
    ]
    tensors += [t for t in (post_norm_weight, out) if t is not None]
    if any(t.device != hidden_states.device for t in tensors):
        raise ValueError("all tensors must be on the same device")
    if prefill_chunk_tokens <= 0 or prefill_chunk_tokens % 128 != 0:
        raise ValueError("prefill_chunk_tokens must be a positive multiple of 128")
    if workspace is not None and (
        workspace.hidden_size,
        workspace.intermediate_size,
    ) != (H, I):
        raise ValueError("workspace was created for another hidden / intermediate size")
    return True


@flashinfer_api(trace=sm90_nvfp4_w4a16_moe_trace)
@backend_requirement(backend_checks={}, common_check=_check_sm90_nvfp4_w4a16_moe)
def sm90_nvfp4_w4a16_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_global_scale: torch.Tensor,
    post_norm_weight: Optional[torch.Tensor] = None,
    post_norm_eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[Sm90Nvfp4W4A16MoEWorkspace] = None,
    enable_pdl: bool = True,
    prefill_min_tokens: Optional[int] = None,
    prefill_chunk_tokens: int = DEFAULT_PREFILL_CHUNK_TOKENS,
) -> torch.Tensor:
    r"""Fused MoE with NVFP4 expert weights and BF16 activations on SM90.

    Computes ``out[t] = sum_j w[t, j] * post(y[t, j])`` with
    ``y[t, j] = W2_e (silu(W1_e x[t]) * (W3_e x[t]))`` for ``e = topk_ids[t, j]``
    (gate ``W1``, up ``W3``, down ``W2``). FP32 accumulation; the SwiGLU output
    is rounded to BF16 once before the down projection.

    Parameters
    ----------
    hidden_states : torch.Tensor
        ``[num_tokens, hidden]`` bfloat16, contiguous.
    topk_ids : torch.Tensor
        ``[num_tokens, top_k]`` int32 or int64, ``top_k <= 8``: expert ids local
        to ``w13`` / ``w2``. Ids outside ``[0, E)`` (for example the experts of
        other ranks under expert parallelism, or -1) contribute nothing.
    topk_weights : torch.Tensor
        ``[num_tokens, top_k]`` routing weights (any float dtype; used as FP32).
    w13, w13_global_scale, w2, w2_global_scale : torch.Tensor
        Packed weights from :func:`prepare_sm90_nvfp4_w4a16_weights` (the fields
        of :class:`Sm90Nvfp4W4A16Weights`), at most 32 experts.
    post_norm_weight : Optional[torch.Tensor]
        ``[hidden]`` float32 (other dtypes are cast on every call; convert
        once at load time to avoid that). When given, each expert output is
        normalized before the weighted sum:
        ``v = y * post_norm_weight; post(y) = v / sqrt(mean(v^2) + post_norm_eps)``
        (the weight is applied before the RMS, as in the post-expert norm of
        Muse-Spark-style models).
    post_norm_eps : float
        Epsilon of the per-route RMSNorm.
    out : Optional[torch.Tensor]
        ``[num_tokens, hidden]`` bfloat16 output buffer.
    workspace : Optional[Sm90Nvfp4W4A16MoEWorkspace]
        Persistent state; defaults to one shared workspace per device and shape.
        Calls that can run concurrently need separate workspaces.
    enable_pdl : bool
        Programmatic dependent launch for the short launches.
    prefill_min_tokens : Optional[int]
        Token count from which the large-batch kernel is used. ``None`` (default)
        picks it from the problem shape with
        :func:`sm90_nvfp4_w4a16_moe_prefill_min_tokens`, assuming every route
        is local (pass it explicitly under expert parallelism). The large-batch
        kernel needs the prefill buffers of the workspace, allocated on first
        use; during CUDA graph capture the automatic choice falls back to the
        small-batch kernel when they are not reserved yet (call
        :meth:`Sm90Nvfp4W4A16MoEWorkspace.reserve_prefill` before capture to
        capture the large-batch kernel).
    prefill_chunk_tokens : int
        Tokens per large-batch launch (sizes the prefill workspace).

    Returns
    -------
    torch.Tensor
        ``[num_tokens, hidden]`` bfloat16.

    Notes
    -----
    Supported: SM90 (sm_90a) only; hidden size a multiple of 512 in
    ``[1024, 8192]``; intermediate size a multiple of 256 in ``[512, 8192]``;
    up to 32 local experts per call; top-k up to 8; SwiGLU activation (gate and
    up may have different global scales). Not supported: other activations,
    biases, weight-only scales other than NVFP4 (16-element E4M3 blocks),
    activation quantization. Every call with up to 1024 tokens, and every
    large-batch call, is a fixed sequence of launches whose buffers are taken
    from the PyTorch allocator, so it can be captured in a CUDA graph once the
    workspace exists.

    Memory: besides the workspace, every large-batch call allocates the
    per-route down rows ``[8, num_tokens, hidden]`` BF16 (``16 * num_tokens *
    hidden`` bytes, about 1 GiB at 16384 tokens and hidden 4096) and small
    routing buffers. They are sized by the whole call, not by
    ``prefill_chunk_tokens``; split very large batches to bound them.

    Reproducibility: each kernel is deterministic (the same inputs give the
    same bits), but outputs are not batch-invariant. The two kernels round the
    down projection differently (the large-batch kernel stores every route's
    down row in BF16 before the top-k sum, the small-batch kernel keeps the
    rows of K-split tiles in FP32), so the switch at ``prefill_min_tokens``
    changes the result bits, and which tiles the small-batch kernel splits
    also depends on the batch. A token's output can therefore differ in the
    last bits between batches of different sizes or composition; pass a fixed
    ``prefill_min_tokens`` to at least keep one kernel.
    """
    H, I, E = hidden_states.shape[1], w13.shape[2] * 128, w13.shape[0]
    M = hidden_states.shape[0]
    device = hidden_states.device
    post_norm = post_norm_weight is not None
    module = _get_module(H, I, post_norm)
    if workspace is None:
        workspace = get_sm90_nvfp4_w4a16_moe_workspace(device, H, I, post_norm)
    if out is None:
        out = torch.empty((M, H), dtype=torch.bfloat16, device=device)
    if M == 0:
        return out
    x = hidden_states.contiguous()
    ids, w = _routing_inputs(topk_ids, topk_weights)
    nw = (
        post_norm_weight.to(torch.float32).contiguous()
        if post_norm
        else torch.empty(4, dtype=torch.float32, device=device)
    )
    gs_gate, gs_up = w13_global_scale[0], w13_global_scale[1]
    eps = float(post_norm_eps)

    min_tokens = prefill_min_tokens
    if min_tokens is None:
        min_tokens = sm90_nvfp4_w4a16_moe_prefill_min_tokens(H, I, E, topk_ids.shape[1])
    use_prefill = min_tokens <= M
    if (
        use_prefill
        and prefill_min_tokens is None
        and workspace.prefill_chunk_tokens < prefill_chunk_tokens
        and torch.cuda.is_current_stream_capturing()
    ):
        use_prefill = False  # its buffers cannot be allocated during capture
    if use_prefill:
        workspace.reserve_prefill(prefill_chunk_tokens)
        recv = torch.empty((TOPK_SLOTS, M, H), dtype=torch.bfloat16, device=device)
        rev = torch.empty(
            ((MAX_LOCAL_EXPERTS + TOPK_SLOTS) * M,), dtype=torch.int32, device=device
        )
        ucnt = torch.empty((((M + 127) // 128) * 64,), dtype=torch.int32, device=device)
        if torch.cuda.is_current_stream_capturing():
            workspace._prefill_captured = True
        module.prefill_run(
            x, ids, w, w13, gs_gate, gs_up, w2, w2_global_scale, nw, out, recv,
            workspace.xt, workspace.maps, rev, ucnt, workspace.prefill_counters,
            E, workspace.prefill_chunk_tokens, eps, enable_pdl,
        )  # fmt: skip
        return out

    for s in range(0, M, DECODE_MAX_TOKENS):
        e_ = min(M, s + DECODE_MAX_TOKENS)
        n = e_ - s
        buf = torch.empty(
            _decode_buffer_bytes(H, I, post_norm, n), dtype=torch.uint8, device=device
        )
        module.decode_run(
            x[s:e_], ids[s:e_], w[s:e_], w13, gs_gate, gs_up, w2, w2_global_scale, nw,
            out[s:e_], buf, workspace.decode_counters, workspace.decode_scratch,
            E, eps, enable_pdl,
        )  # fmt: skip
    return out
