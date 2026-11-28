# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""SM8x dense ``mm_bf16_fp4`` on the W4A16 MoE records GEMM (``csrc/w4a16_moe_sm80.cu``).

Every ``mm_bf16_fp4`` backend name keeps its SM100+ requirements and runs here on SM8x as the
records GEMM2 of one expert: activation none, top-k 1, no finalize, the identity row map of
the MoE prep (one expert keeps the entries in token order). A float16 (float32) output runs
the BF16 module whose GEMM2 writes FP16 (``RECORDS_OUT_F16``), rounded once from FP32 (FP32,
``RECORDS_OUT_F32``).

``prepare_bf16_fp4_weights`` on SM8x returns the records of every backend: ``b`` uint8
``[N, ceil(size / N)]`` whose first ``size`` bytes are the records, ``b_descale`` an empty
float8_e4m3fn tensor, ``alpha`` unchanged. ``mm_bf16_fp4`` also takes each backend's SM100+
format (``cute-dsl-native``'s canonical weight among them) and builds its records per call.

``mm_fp4`` (NVFP4 or MXFP4 operands) runs on SM8x for every backend as the prep's exact BF16
restore of ``a`` and the dense GEMM ``gemm_fp4``, which reads the ``[N, K/2]`` weight and its
128x4-swizzled scales as given, with a scalar or per-token ``alpha``. For MXFP4 the restore
leaves out ``a``'s E8M0 scales, which ``gemm_fp4`` applies with ``b``'s per 32-value block.
The ``trtllm`` backend's ``b`` and ``b_descale`` (``nvfp4_quantize(do_shuffle=True)``) are read
in their row shuffle, and an 8x4-swizzled ``a_descale`` is restored as such.
``mm_nvfp4_svdquant`` runs the same GEMM into FP32 and adds the low-rank term there, rounded
once to ``out``.
"""

import functools
from typing import List, Optional, Tuple

import torch

from ..api_logging import _is_current_stream_capturing
from ..autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ..fused_moe.core import _RECORDS_NO_EXT, _w4a16_moe_sm80_module
from ..fused_moe.prepare import (
    W4A16_SM80_TACTICS,
    _w4a16_records,
    _w4a16_records_geometry,
)
from ..jit.fused_moe import gen_w4a16_moe_sm80_module
from ..utils import _get_stream_cache_buf, get_compute_capability, get_device_sm_count

# (row tile, GEMM2 variant) of the MoE tactics, GEMM1's variant dropped
_TACTICS = tuple(dict.fromkeys((r, g2) for r, _, g2 in W4A16_SM80_TACTICS))
_DEFAULT_TACTIC = (64, 0)

_ones_cache: dict = {}


def _ones(device: torch.device) -> torch.Tensor:
    """The fp32 scalar 1 for calls without an alpha (a constant, not derived from a weight)."""
    key = (device.type, device.index)
    value = _ones_cache.get(key)
    if value is None:
        if _is_current_stream_capturing():
            raise RuntimeError(
                "SM8x dense FP4 needs a constant on the device; run one call before "
                "capturing it in a CUDA graph."
            )
        value = _ones_cache[key] = torch.ones(1, dtype=torch.float32, device=device)
    return value


def _pad(x: int, y: int) -> int:
    return -(-x // y) * y


def _flat_bytes(t: torch.Tensor) -> torch.Tensor:
    """``t``'s bytes in storage order as a flat uint8 alias (no copy). A strided view must
    cover one dense block (the 6-D scale view of the SM10x cute-dsl prep), read from its
    first element."""
    if t.is_contiguous():
        return t.view(torch.uint8).reshape(-1)
    span = 1
    for stride, size in sorted(
        (st, sz) for st, sz in zip(t.stride(), t.shape, strict=True) if sz != 1
    ):
        if stride != span:
            raise ValueError(
                f"b_descale of shape {tuple(t.shape)} and strides {t.stride()} is not a "
                "view of one dense block"
            )
        span *= size
    offset, nbytes = t.storage_offset() * t.element_size(), span * t.element_size()
    if t.untyped_storage().nbytes() - offset < nbytes:
        raise ValueError("b_descale's view runs past its storage")
    return torch.empty(0, dtype=torch.uint8, device=t.device).set_(
        t.untyped_storage(), offset, (nbytes,), (1,)
    )


def _records_size(n: int, k: int) -> int:
    return _w4a16_records_geometry(n, k, False)[3]


def _is_records(a: torch.Tensor, b: torch.Tensor, b_descale: torch.Tensor) -> bool:
    """Whether ``(b, b_descale)`` is the SM8x prep's records of an ``[N, a.shape[1]]`` weight."""
    if b.dim() != 2 or b.dtype != torch.uint8 or a.dim() != 2 or b_descale.numel():
        return False
    n, k = int(b.shape[0]), int(a.shape[1])
    return n > 0 and k > 0 and int(b.shape[1]) == -(-_records_size(n, k) // n)


def sm8x_records_nk(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    block_size: int,
) -> Optional[Tuple[int, int]]:
    """``(N, K)`` of SM8x records ``b`` (checked as a backend checks its prepared weight), or
    None when ``a`` is not on an SM8x device or ``b`` is not records."""
    if not a.is_cuda or get_compute_capability(a.device)[0] != 8:
        return None
    if not _is_records(a, b, b_descale):
        return None
    n, k = int(b.shape[0]), int(a.shape[1])
    if block_size != 16 or k % 16:
        raise ValueError(
            f"SM8x records require block_size=16 and K%16=0; got {block_size}, K={k}"
        )
    if n * k >= 2**31:
        raise ValueError("SM8x records require N*K below 2**31")
    if not b.is_contiguous() or b.data_ptr() % 16:
        raise ValueError("SM8x records must be contiguous and 16 B aligned")
    tensors = [b, b_descale]
    if alpha is not None:
        if (
            alpha.dtype != torch.float32
            or tuple(alpha.shape) != (1,)
            or not alpha.is_contiguous()
        ):
            raise ValueError("alpha must be a contiguous float32 tensor of shape (1,)")
        tensors.append(alpha)
    if any(t.device != a.device for t in tensors):
        raise ValueError("a, b, b_descale and alpha must be on the same device")
    return n, k


def check_sm8x_blackwell_records(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor],
    block_size: int,
    backend: str,
) -> bool:
    """The source-built Blackwell backends' checks on SM8x records (False: not records)."""
    nk = sm8x_records_nk(a, b, b_descale, alpha, block_size)
    if nk is None:
        return False
    n, _ = nk
    if a.dtype != torch.bfloat16:
        raise ValueError(
            "source-built BF16 x FP4 GEMM expects a as bfloat16 [M, K]; "
            f"got dtype={a.dtype}, shape={tuple(a.shape)}"
        )
    if backend == "blackwell-native":
        if out_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"blackwell-native supports bfloat16 or float16 output; got {out_dtype}"
            )
    elif backend == "blackwell-tiled":
        if n % 64 != 0:
            raise ValueError(
                f"blackwell-tiled requires N to be a multiple of 64; got N={n}"
            )
        if out_dtype != torch.bfloat16:
            raise ValueError(
                f"blackwell-tiled requires bfloat16 output; got {out_dtype}"
            )
    else:
        raise ValueError(f"unknown source-built BF16 x FP4 backend {backend!r}")
    m = int(a.shape[0])
    if out is not None and (
        tuple(out.shape) != (m, n)
        or out.dtype != out_dtype
        or out.device != a.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must be contiguous, on the same device, and have shape/dtype "
            f"{(m, n)}/{out_dtype}"
        )
    if not a.is_contiguous():
        raise ValueError("a, b, and b_descale must be contiguous")
    if m <= 0:
        raise ValueError(f"M and N must be positive; got M={m}, N={n}")
    return True


# ---------------------------------------------------------------------------------------------
# Records of each backend's format
# ---------------------------------------------------------------------------------------------


def _unswizzle(b_descale: torch.Tensor, n: int, k_sf: int) -> torch.Tensor:
    """Linear ``(N, K_sf)`` uint8 scales of a 128x4-swizzled buffer (1-D, 2-D or the 6-D view)."""
    from .gemm_bf16_fp4 import _unswizzle_sf_128x4

    nbytes = _pad(n, 128) * _pad(k_sf, 4)
    flat = _flat_bytes(b_descale)
    if flat.numel() < nbytes:
        raise ValueError(
            f"b_descale has {flat.numel()} bytes but the 128x4-swizzled layout for "
            f"N={n}, K_sf={k_sf} requires at least {nbytes}"
        )
    return _unswizzle_sf_128x4(flat, n, k_sf)


def _unpack_tiled(b: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """``(N, K/2)`` uint8 of the int32 ``(K/16, 2N)`` cute-dsl repack (its byte gather inverted)."""
    from .gemm_bf16_fp4_cute_dsl import _cute_dsl_pack_fp4_weight

    idx = torch.arange(512, device=b.device)
    lo, hi = (
        _cute_dsl_pack_fp4_weight(v.to(torch.uint8).view(8, 64))  # one 16 K x 64 N tile
        .view(torch.uint8)
        .reshape(-1)
        .long()
        for v in (idx & 255, idx >> 8)
    )
    src = hi * 256 + lo  # packed byte j holds tile byte src[j]
    k_tiles, n_tiles = k // 16, n // 64
    tile = torch.empty((k_tiles, n_tiles, 512), dtype=torch.uint8, device=b.device)
    tile[:, :, src] = b.contiguous().view(torch.uint8).view(k_tiles, n_tiles, 512)
    return (
        tile.view(k_tiles, n_tiles, 8, 64)
        .permute(0, 2, 1, 3)
        .reshape(k // 2, n)
        .t()
        .contiguous()
    )


def _records(w: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Records ``[1, size]`` of ``w`` uint8 ``(N, K/2)`` and linear E4M3 ``sf`` ``(N, K/16)``."""
    return _w4a16_records(w.contiguous()[None], sf.contiguous()[None], gated=False)


def _format_of(backend: str, b: torch.Tensor) -> str:
    """The SM100+ format of ``backend``'s prepared ``b`` (as on SM90)."""
    if backend in ("cudnn", "blackwell-native"):
        return "linear"
    if backend == "blackwell-tiled":
        return "tiled"
    if backend == "cute-dsl-native":
        return "swz128x4"
    if backend == "cute-dsl":
        return "swz128x4" if b.dtype == torch.uint8 else "tiled"
    raise ValueError(f"unknown mm_bf16_fp4 backend {backend!r}")


def _call_records(
    a: torch.Tensor, b: torch.Tensor, b_descale: torch.Tensor, backend: str
) -> Tuple[torch.Tensor, int]:
    """Records ``[1, size]`` and N of one call's weight: SM8x records as given, an SM100+
    format converted."""
    k = int(a.shape[1])
    if _is_records(a, b, b_descale):
        n = int(b.shape[0])
        if not b.is_contiguous() or b.data_ptr() % 16 or b.device != a.device:
            raise ValueError(
                "SM8x records must be contiguous, 16 B aligned and on a's device"
            )
        size = _records_size(n, k)
        return b.view(-1)[:size].view(1, size), n
    fmt = _format_of(backend, b)
    if fmt == "tiled":
        if b.dim() != 2 or b.dtype != torch.int32 or int(b.shape[1]) % 128:
            raise ValueError(
                f"{backend} expects the int32 [K/16, N*2] weight; got {b.dtype} "
                f"{tuple(b.shape)}"
            )
        n, kb = int(b.shape[1]) // 2, int(b.shape[0]) * 16
    else:
        if b.dim() != 2 or b.dtype != torch.uint8:
            raise ValueError(
                f"{backend} expects the uint8 [N, K/2] weight; got {b.dtype} "
                f"{tuple(b.shape)}"
            )
        n, kb = int(b.shape[0]), int(b.shape[1]) * 2
    if kb != k:
        raise ValueError(f"a.shape[1]={k} but prepared b encodes K={kb}")
    if b.device != a.device or b_descale.device != a.device:
        raise ValueError("a, b, and b_descale must be on the same device")
    k_sf = k // 16
    if fmt == "linear":
        if tuple(b_descale.shape) != (n, k_sf) or b_descale.element_size() != 1:
            raise ValueError(
                f"{backend} expects linear E4M3 scales of shape {(n, k_sf)}; got "
                f"{b_descale.dtype} {tuple(b_descale.shape)}"
            )
        return _records(b, b_descale.view(torch.uint8)), n
    if fmt == "swz128x4":
        return _records(b, _unswizzle(b_descale, n, k_sf)), n
    if b_descale.dtype != torch.uint8 or tuple(b_descale.shape) != (k_sf, n):
        raise ValueError(
            f"{backend} expects S0E5M3 uint8 scales of shape {(k_sf, n)}; got "
            f"{b_descale.dtype} {tuple(b_descale.shape)}"
        )
    # S0E5M3 (FP16 bits 14..7 of the positive E4M3 value) back to E4M3
    sf = (
        (b_descale.t().to(torch.int16) << 7)
        .view(torch.float16)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    return _records(_unpack_tiled(b, n, k), sf), n


# ---------------------------------------------------------------------------------------------
# Prep and GEMM
# ---------------------------------------------------------------------------------------------


def sm8x_prepare_bf16_fp4(
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    block_size: int,
    backend: str,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """``prepare_bf16_fp4_weights`` on SM8x (after its common checks): the records of the
    canonical weight, under ``backend``'s own preparation checks."""
    n, k = int(b.shape[0]), int(b.shape[1]) * 2
    if backend == "cute-dsl-native":
        from .kernels.native_bf16_fp4.support import check_weights

        check_weights(b, b_descale, alpha, block_size)
    elif backend in ("blackwell-native", "blackwell-tiled"):
        if block_size != 16:
            raise ValueError(
                "source-built BF16 x FP4 preparation requires block_size=16; "
                f"got {block_size}"
            )
        if b.device != b_descale.device:
            raise ValueError(
                "b and b_descale must be on the same device; got "
                f"{b.device} and {b_descale.device}"
            )
        if alpha is not None and alpha.device != b.device:
            raise ValueError(
                f"alpha must be on the same device as b ({b.device}); got {alpha.device}"
            )
        if backend == "blackwell-tiled" and n % 64 != 0:
            raise ValueError(
                f"blackwell-tiled requires N to be a multiple of 64; got N={n}"
            )
    rec = _records(b, _unswizzle(b_descale, n, k // block_size))
    size = int(rec.shape[1])
    row = -(-size // n)
    buf = torch.zeros(n * row, dtype=torch.uint8, device=b.device)
    buf[:size] = rec.view(-1)
    return (
        buf.view(n, row),
        torch.empty(0, dtype=torch.float8_e4m3fn, device=b.device),
        alpha,
    )


@functools.cache
def _out_module(dtype: torch.dtype):
    """The BF16 module whose GEMM2 writes ``dtype``, float16 or float32."""
    out = "f16" if dtype == torch.float16 else "f32"
    return gen_w4a16_moe_sm80_module(out=out).build_and_load()


def _run(
    a: torch.Tensor,
    rec: torch.Tensor,
    n: int,
    alpha: torch.Tensor,
    out: torch.Tensor,
    tactic: Tuple[int, int],
) -> None:
    """``out[:, :n] = a @ W^T * alpha`` (``out`` BF16 or FP16) on the records: one expert's
    GEMM2, rows in token order."""
    r, g2 = tactic
    m, k = int(a.shape[0]), int(a.shape[1])
    max_blocks = -(-m // r)
    ints = (1 + 2 * max_blocks, max_blocks * r, -(-m // 2048) + 1)
    meta, rowmap, workspace = torch.empty(
        sum(ints), dtype=torch.int32, device=a.device
    ).split(ints)
    topk_ids = torch.zeros((m, 1), dtype=torch.int32, device=a.device)
    module = _w4a16_moe_sm80_module(None, False, None, False)
    # the BF16 module's prep; an FP16 or FP32 output takes that output module's GEMM2
    gemm = module.gemm if out.dtype == torch.bfloat16 else _out_module(out.dtype).gemm
    module.prep(
        topk_ids, meta, rowmap, workspace, 1, r, max_blocks, None, None, None, False, None,
        None, 0, None, None, None, None, None,
    )  # fmt: skip
    cols = _w4a16_records_geometry(n, k, False)[0]
    gemm(
        rec, alpha, a, meta, rowmap, out, r, 0, 1, cols, k, int(out.shape[1]), max_blocks,
        g2, None, None, None, None, None, None, None, 1, *_RECORDS_NO_EXT,
    )  # fmt: skip


class Sm8xFp4DenseRunner(TunableRunner):
    """``_run`` behind ``mm_bf16_fp4`` as an autotuner candidate; inputs ``[a, records,
    b_descale, alpha, out_dtype, out]`` (``out`` the GEMM target, of ``out_dtype``); tactic
    (row tile, GEMM2 variant)."""

    def __init__(self, key: tuple, n: int):
        self.key, self.n = key, n

    def __hash__(self) -> int:
        return hash((self.key, self.n))

    def get_cache_key_extras(self, inputs) -> tuple:
        return (self.key, inputs[0].dtype, inputs[4])

    def get_valid_tactics(self, inputs, profile: OptimizationProfile) -> List:
        return list(_TACTICS)

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        a, rec, _, alpha, _, out = inputs
        if tactic is None or tactic == -1:
            tactic = _DEFAULT_TACTIC
        _run(a, rec, self.n, alpha, out, tuple(tactic))
        return out


def sm8x_mm_bf16_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    *,
    backend: str,
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    """``mm_bf16_fp4`` on SM8x (after the backend's requirement check), with that backend's
    output contract. An unaligned, strided or N % 8 != 0 output is written through a buffer
    of its dtype."""
    if backend == "cute-dsl" and out_dtype != torch.bfloat16:
        raise NotImplementedError(
            f"cute-dsl currently requires a bfloat16 output; got {out_dtype}."
        )
    if not a.is_contiguous():
        raise ValueError("a, b, and b_descale must be contiguous")
    rec, n = _call_records(a, b, b_descale, backend)
    m = int(a.shape[0])
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    elif tuple(out.shape) != (m, n):
        raise ValueError(f"out shape {tuple(out.shape)} != expected {(m, n)}")
    elif out.dtype != out_dtype:
        raise TypeError(f"out dtype {out.dtype} != requested out_dtype {out_dtype}")
    elif out.device != a.device:
        raise ValueError(f"out is on {out.device} but a is on {a.device}")
    if m == 0:
        return out
    if a.data_ptr() % 16:
        a = a.clone()
    if alpha is None:
        alpha = _ones(a.device)
    direct = n % 8 == 0 and out.is_contiguous() and out.data_ptr() % 16 == 0
    target = (
        out
        if direct
        else torch.empty((m, _pad(n, 8)), dtype=out.dtype, device=a.device)
    )
    from .gemm_bf16_fp4_cudnn import _BF16_FP4_TUNING_CONFIG

    runner = Sm8xFp4DenseRunner(("mm_bf16_fp4", backend), n)
    inputs = [a, rec, b_descale, alpha, out_dtype, target]
    chosen, tactic = AutoTuner.get().choose_one(
        "bf16_fp4_gemm", [runner], _BF16_FP4_TUNING_CONFIG, inputs
    )
    chosen(inputs=inputs, tactic=tactic)
    if not direct:
        out.copy_(target[:, :n])
    return out


# ---------------------------------------------------------------------------------------------
# mm_fp4 (NVFP4 x NVFP4, MXFP4 x MXFP4)
# ---------------------------------------------------------------------------------------------


def _run_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out: torch.Tensor,
    r: int,
    mx: bool = False,
    shuffled: bool = False,
    sf_8x4: bool = False,
    split: int = 1,
) -> None:
    """``out = (a @ b) * alpha`` for ``mm_fp4``'s NVFP4 (MXFP4: ``mx``) operands: the prep
    restores ``a`` to BF16 exactly (MXFP4: without its scales; ``sf_8x4``: its scales 8x4
    swizzled), then the dense GEMM (``gemm_fp4``) reads ``b`` and its 128x4-swizzled scales as
    given (MXFP4: and ``a``'s; ``shuffled``: both in the trtllm row shuffle). ``split`` > 1
    (NVFP4, unshuffled, K % 128 == 0): each output tile's K is split over ``split`` CTAs, whose
    FP32 partial sums the tile's last CTA adds in split order (``gemm_fp4_sk``)."""
    m, k, n = int(a.shape[0]), 2 * int(a.shape[1]), int(b.shape[1])
    if m == 0:
        return
    a = a.view(torch.uint8)  # mm_fp4 also takes the native packed FP4 dtype
    if not a.is_contiguous() or a.data_ptr() % 16:
        a = a.clone(memory_format=torch.contiguous_format)
    w = b.view(torch.uint8).t()
    if not w.is_contiguous() or w.data_ptr() % 16:
        w = w.clone(memory_format=torch.contiguous_format)
    sf = _flat_bytes(b_descale)
    if sf.data_ptr() % 16:
        sf = sf.clone()
    if alpha is None:
        alpha = _ones(a.device)
    max_blocks = -(-m // r)
    ints = (1 + 2 * max_blocks, max_blocks * r, -(-m // 2048) + 1)
    meta, rowmap, workspace = torch.empty(
        sum(ints), dtype=torch.int32, device=a.device
    ).split(ints)
    topk_ids = torch.zeros((m, 1), dtype=torch.int32, device=a.device)
    act = torch.empty((m, k), dtype=torch.bfloat16, device=a.device)
    asf = _flat_bytes(a_descale)
    if asf.data_ptr() % 4:
        asf = asf.clone()
    module = _w4a16_moe_sm80_module(None, False, None, False)
    module.prep(
        topk_ids, meta, rowmap, workspace, 1, r, max_blocks, a, asf, act,
        True, None, None, 5 if mx else 6 if sf_8x4 else 0, None, None, None, None, None,
    )  # fmt: skip
    direct = n % 8 == 0 and out.is_contiguous() and out.data_ptr() % 16 == 0
    target = (
        out
        if direct
        else torch.empty((m, _pad(n, 8)), dtype=out.dtype, device=a.device)
    )
    gemm = module if out.dtype == torch.bfloat16 else _out_module(out.dtype)
    args = (
        w, sf, alpha, act, meta, rowmap, target, r, n, k, max_blocks, alpha.numel() > 1,
        asf if mx else None, shuffled, None, 0,
    )  # fmt: skip
    if split > 1:
        tiles = max_blocks * -(-n // 128)
        ws_bytes, cnt_bytes = tiles * split * r * 512, tiles * 4
        if _is_current_stream_capturing():
            # the replay stream is unknown and two graphs may replay concurrently: each
            # captured call owns its scratch in the graph's pool, and the captured memset
            # re-zeroes its arrival counts on every replay
            ws = torch.empty(ws_bytes, dtype=torch.uint8, device=a.device)
            cnt = torch.zeros(cnt_bytes, dtype=torch.uint8, device=a.device)
        else:
            # launches on one stream are ordered and the kernel leaves every tile's arrival
            # count at 0 on exit, so the stream's scratch, zeroed once, stays valid
            ws = _get_stream_cache_buf("sm8x_mm_fp4_splitk_ws", ws_bytes, a.device)
            cnt = _get_stream_cache_buf(
                "sm8x_mm_fp4_splitk_cnt", cnt_bytes, a.device, zero_init=True
            )
        gemm.gemm_fp4_sk(*args, split, ws.view(torch.float32), cnt.view(torch.int32))
    else:
        gemm.gemm_fp4(*args)
    if not direct:
        out.copy_(target[:, :n])


_FP4_SPLITS = (2, 3, 4, 6, 8, 12, 16)


class Sm8xMmFp4Runner(TunableRunner):
    """``_run_fp4`` behind ``mm_fp4`` as an autotuner candidate on ``mm_fp4``'s inputs; tactic:
    the row tile, or (row tile, split) for a split over K (NVFP4, unshuffled, K % 128 == 0, at
    most 8 CTAs per SM, which bounds the partial sums' scratch)."""

    def __init__(self, shuffled: bool, sf_8x4: bool):
        self.shuffled = shuffled
        self.sf_8x4 = sf_8x4

    def get_cache_key_extras(self, inputs) -> tuple:
        alpha_rows = inputs[4] is not None and inputs[4].numel() > 1
        return (inputs[5], alpha_rows, inputs[8], self.shuffled, self.sf_8x4)

    def get_valid_tactics(self, inputs, profile: OptimizationProfile) -> List:
        tactics: List = [8, 16, 32, 64]
        a, b = inputs[0], inputs[1]
        m, k, n = int(a.shape[0]), 2 * int(a.shape[1]), int(b.shape[1])
        if not inputs[8] or self.shuffled or k % 128 or m == 0:
            return tactics
        ctas = 8 * get_device_sm_count(a.device)
        for r in (8, 16, 32, 64):
            tiles = -(-m // r) * -(-n // 128)
            tactics += [
                (r, s) for s in _FP4_SPLITS if s <= k // 128 and tiles * s <= ctas
            ]
        return tactics

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        a, b, a_descale, b_descale, alpha, _, out = inputs[:7]
        if tactic is None or tactic == -1:
            tactic = 64
        r, split = (tactic, 1) if isinstance(tactic, int) else tuple(tactic)
        _run_fp4(
            a,
            b,
            a_descale,
            b_descale,
            alpha,
            out,
            r,
            not inputs[8],
            self.shuffled,
            self.sf_8x4,
            split,
        )
        return out


def sm8x_mm_fp4_runner(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out: torch.Tensor,
    use_8x4_sf_layout: bool,
    shuffled: bool = False,
) -> Sm8xMmFp4Runner:
    """The ``mm_fp4`` runner of every backend on SM8x (after the backend's requirement
    check); ``shuffled``: ``b`` and its scales in the trtllm backend's row shuffle."""
    m, k, n = int(a.shape[0]), 2 * int(a.shape[1]), int(b.shape[1])
    if k % 32:
        raise ValueError(f"mm_fp4 on SM8x requires K % 32 == 0, got K={k}")
    if tuple(out.shape) != (m, n):
        raise ValueError(f"out shape {tuple(out.shape)} != expected {(m, n)}")
    if out.dtype != out_dtype:
        raise TypeError(f"out dtype {out.dtype} != requested out_dtype {out_dtype}")
    if any(
        t is not None and t.device != a.device
        for t in (b, a_descale, b_descale, alpha, out)
    ):
        raise ValueError(
            "a, b, a_descale, b_descale, alpha and out must be on one device"
        )
    if alpha is None:
        _ones(a.device)  # outside a tuner's graph capture
    return Sm8xMmFp4Runner(shuffled, use_8x4_sf_layout)


@functools.cache
def _nvfp4_svdquant_tuning() -> TuningConfig:
    from ..fused_moe.utils import (
        get_hybrid_num_tokens_buckets,
        map_to_hybrid_bucket_uncapped,
    )

    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                (0,), (0,), get_hybrid_num_tokens_buckets, map_to_hybrid_bucket_uncapped
            ),
        ),
        constraint_specs=(
            ConstraintSpec(2, 0, lambda s: _pad(s[0][0], 128) * _pad(s[0][1] // 8, 4)),
            ConstraintSpec(6, 0, lambda s: s[0][0]),
        ),
    )


def sm8x_mm_nvfp4_svdquant(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    d: torch.Tensor,
    l1: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
) -> None:
    """``mm_nvfp4_svdquant`` on SM8x: ``out = alpha * (a @ b.T + d @ l1.T) + bias`` from the
    FP32 product of the NVFP4 ``a`` ``(M, K/2)`` and ``b`` ``(N, K/2)`` with their
    128x4-swizzled scales, rounded once; one ``alpha`` on the summed terms as the SM100
    epilogue."""
    m, n = int(out.shape[0]), int(out.shape[1])
    if m == 0:
        return
    acc = torch.empty((m, n), dtype=torch.float32, device=a.device)
    inputs = [
        a, b.t(), _flat_bytes(a_sf), _flat_bytes(b_sf), _ones(a.device), torch.float32,
        acc, 16, True,
    ]  # fmt: skip
    chosen, tactic = AutoTuner.get().choose_one(
        "mm_nvfp4_svdquant_sm8x",
        [Sm8xMmFp4Runner(False, False)],
        _nvfp4_svdquant_tuning(),
        inputs,
    )
    chosen(inputs=inputs, tactic=tactic)
    acc += d.float() @ l1.float().t()
    acc *= alpha.reshape(1)
    if bias is not None:
        acc += bias.float()
    out.copy_(acc)
