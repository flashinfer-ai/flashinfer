# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""SM100a (B200) implementation of :mod:`flashinfer.fused_moe.tp_moe`.

Protocol (called by tp_moe.py only):

* ``build_for_m(config, m)`` -> hashable variant (the builds serving ``m`` tokens).
* ``load_module(variant)`` -> the built and loaded JIT module(s).
* ``create_workspace(config, group, device)`` / ``destroy_workspace(state, group)``: collective.
* ``prepare_weights(config, rank, device, **checkpoint_tensors)`` -> weight state.
* ``run(config, x, weights, workspace, out, **routing)``: launches on the current stream.

Two kernel families; each call is one persistent kernel (gate, x quantization, routed MXFP4 experts, TP-sharded FP8 shared
expert, all-reduce over CUDA-IPC buffers), compiled per token range ``[mmin, mmax]``, M <= 64:

* ``flash`` (``csrc/tp_moe/sm100/flash.cu``): DeepSeek-V4-Flash, TP4 / TP8.  The fused-gate entry and the ids entry are
  separate builds (the gate is compiled in or out).
* ``generic`` (``csrc/tp_moe/sm100/generic.cu``): DeepSeek-V4.1-Flash, the shape from compile-time flags (plain
  TP: every rank holds all experts' ``I / tp`` slice).  One build serves both entries.

Weights.  Both models keep one copy of this rank's routed slice in the trtllm-gen MXFP4 layout (what flashinfer's
trtllm-gen MoE reads) and read it in place.  DeepSeek-V4.1-Flash's slice (576 / 288 columns at TP4 / TP8) is
zero-padded to whole 128-column tiles (640 / 384), as trtllm-gen needs; the kernel never reads the padding, and runs
the last partial chunk of each FC2 row from a small copy of those K tails (see ``prepare_weights``).  The shared expert
is this rank's TP slice of the FP8 checkpoint weights plus UE8M0 scale images.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Dict, List, Tuple

import torch

# every build is compiled for exactly this grid (one CTA per SM of a B200)
NUM_SMS = 148
TOP_K = 6
MAX_M = 64  # tokens per call (decode)

# --------------------------------------------------------------------------------------------- build tables
# fmt: off
# DeepSeek-V4-Flash ("flash" family): (mmin, mmax, tuning flags).  Every other flag keeps the source default.
_FLASH_BUCKETS: Tuple[Tuple[int, int, Tuple[Tuple[str, int], ...]], ...] = (
    (1, 1, (("GUNR", 1), ("TABW", 1), ("FC2PAIR_OFFM", 0), ("ACTU", 1), ("FC2CAP", 1), ("PUBRED", 1), ("ARV", 40))),
    (2, 2, (("GUNR", 1), ("SFBI", 1), ("ACTU", 1), ("FC2CAP", 1), ("ARV", 40))),
    (3, 4, (("GUNR", 1), ("SFBI", 1), ("ACTU", 1), ("ARV", 40))),
    (5, 5, (("GUNR", 1), ("SFBI", 1), ("SHGU_MAXM", 8), ("ACTU", 1), ("ARV", 40))),
    (6, 6, (("GUNR", 1), ("SFBI", 1), ("SHGU_MAXM", 8), ("ACTU", 1), ("ARV", 40))),
    (7, 7, (("GUNR", 1), ("SFBI", 1), ("SHGU_MAXM", 8), ("ACTU", 1), ("ARV", 40))),
    (8, 8, (("GUNR", 1), ("SFBI", 1), ("SHGU_MAXM", 8), ("PUBRED", 1), ("ACTU", 1), ("ARV", 40))),
    (9, 15, (("SFBI", 1), ("FC2S_HI", 16), ("PUBRED", 1), ("ARV", 40))),
    (16, 16, (("SFBI", 1), ("FC2S_HI", 16), ("PUBRED", 1), ("ARV", 40))),
    (17, 19, (("SFBI", 1), ("SHGU_LO2", 17), ("SHGU_HI2", 31), ("ARV", 40))),
    (20, 20, (("SFBI", 1), ("SHGU_LO2", 17), ("SHGU_HI2", 31), ("ARV", 40))),
    (21, 23, (("SFBI", 1), ("SHGU_LO2", 17), ("SHGU_HI2", 31), ("ARV", 40))),
    (24, 24, (("SFBI", 1), ("SHGU_LO2", 17), ("SHGU_HI2", 31), ("ARV", 40))),
    (25, 31, (("SFBI", 1), ("SHGU_LO2", 17), ("SHGU_HI2", 31), ("ARV", 40))),
    (32, 32, (("ACTX", 1), ("SFBI", 1), ("ARV", 40))),
    (33, 48, (("ACTX", 1), ("SFBI", 1), ("ARV", 34), ("ONESHOT_MAXM", 32))),
    (49, 63, (("ACTX", 1), ("SFBI", 1), ("ARV", 34))),
    (64, 64, (("ACTX", 1), ("SFBI", 1), ("ARV", 34))),
)
# DeepSeek-V4.1-Flash ("generic" family, plain TP)
_V41_BUCKETS = ((1, 7), (8, 8), (9, 15), (16, 16), (17, 31), (32, 32), (33, 63), (64, 64))
# fmt: on


@dataclasses.dataclass(frozen=True)
class Build:
    """One compiled kernel: family source + -D flags (name, value) pairs."""

    family: str
    flags: Tuple[Tuple[str, int], ...]

    @property
    def flag_dict(self) -> Dict[str, int]:
        return dict(self.flags)

    @property
    def uri(self) -> str:
        import hashlib

        f = self.flag_dict
        h = hashlib.sha1(repr(self.flags).encode()).hexdigest()[:10]
        return f"tp_moe_sm100_{self.family}_tp{f['NDEV']}_m{f['MMIN']}_{f['MMAX']}_{h}"


@dataclasses.dataclass(frozen=True)
class Variant:
    """The builds that serve one M range: ``gate`` runs the fused-gate entry, ``ids`` the ids entry."""

    mmin: int
    mmax: int
    gate: Build
    ids: Build


def _flash_build(tp: int, mmin: int, mmax: int, tune, gate: bool) -> Build:
    flags = (("MMIN", mmin), ("MMAX", mmax), ("NDEV", tp), ("GATE", int(gate))) + tuple(
        tune
    )
    return Build("flash", flags)


def _generic_build(tp: int, mmin: int, mmax: int) -> Build:
    return Build(
        "generic",
        _shape_flags(tp)
        + (("MMIN", mmin), ("MMAX", mmax), ("NDEV", tp), ("NMAX", 24), ("ACC", 12)),
    )


def _ipad(tp: int) -> int:
    """DeepSeek-V4.1-Flash: this rank's routed intermediate I / tp zero-padded to whole 128 columns (the trtllm-gen
    layout); the shared slice is padded the same way"""
    return (2304 // tp + 127) // 128 * 128


def _shape_flags(tp: int) -> Tuple[Tuple[str, int], ...]:
    return (
        ("DIM", 5120),
        ("IFULL", 2304),
        ("NEXP", 384),
        ("ISPLIT", tp),
        ("IPAD", _ipad(tp)),
        ("SHI", _ipad(tp)),
    )


def build_for_m(config, m: int) -> Variant:
    if not 1 <= m <= min(config.max_num_tokens, MAX_M):
        raise ValueError(f"tp_moe: M = {m} outside [1, {config.max_num_tokens}]")
    tp = config.tp_size
    if config.model == "dsv4_flash":
        lo, hi, tune = next(b for b in _FLASH_BUCKETS if b[0] <= m <= b[1])
        return Variant(
            lo,
            hi,
            _flash_build(tp, lo, hi, tune, True),
            _flash_build(tp, lo, hi, tune, False),
        )
    lo, hi = next(b for b in _V41_BUCKETS if b[0] <= m <= b[1])
    b = _generic_build(tp, lo, hi)
    return Variant(lo, hi, b, b)


def all_variants(config) -> List[Variant]:
    out, m = [], 1
    while m <= config.max_num_tokens:
        v = build_for_m(config, m)
        out.append(v)
        m = v.mmax + 1
    return out


@functools.cache
def _load_build(build: Build):
    from ..jit.tp_moe import gen_tp_moe_sm100_module

    return gen_tp_moe_sm100_module(build).build_and_load()


def load_module(variant: Variant):
    gate = _load_build(variant.gate)
    ids = gate if variant.ids == variant.gate else _load_build(variant.ids)
    return gate, ids


# --------------------------------------------------------------------------------------------- workspace
@dataclasses.dataclass
class _FamilyWorkspace:
    layout: Tuple[
        int, ...
    ]  # (workspace bytes, rs bytes, ag bytes, 0xFF-filled bytes at the start of rs)
    scratch: (
        torch.Tensor
    )  # zeroed device bytes: counters (epoch-parity sets) + per-call scratch
    rs: List[int]  # peer pointers of the reduce-scatter (all-reduce message) buffers
    ag: List[
        int
    ]  # peer pointers of the all-gather buffers (empty: the family uses rs only)


@dataclasses.dataclass
class Workspace:
    config: object
    rank: int
    device: torch.device
    families: Dict[str, _FamilyWorkspace]


def _family(config) -> str:
    return "flash" if config.model == "dsv4_flash" else "generic"


@functools.cache
def _layout_of(build: Build) -> Tuple[int, ...]:
    return tuple(int(v) for v in _load_build(build).layout())


def _representative(config) -> Build:
    return build_for_m(config, min(config.max_num_tokens, MAX_M)).gate


def _memset(ptr: int, value: int, nbytes: int) -> None:
    import ctypes

    from ..comm.cuda_ipc import cudart

    cudart.cudaMemset(ctypes.c_void_p(ptr), value, nbytes)


def create_workspace(config, group, device) -> Workspace:
    from ..comm.cuda_ipc import create_shared_buffer
    import torch.distributed as dist

    _check_device(device)
    torch.cuda.set_device(device)
    # every build's code on this device now, not at its first launch
    for variant in all_variants(config):
        for mod in load_module(variant):
            mod.load()
    layout = _layout_of(_representative(config))
    ws_bytes, rs_bytes, ag_bytes, ll_bytes = layout
    scratch = torch.zeros(ws_bytes, dtype=torch.uint8, device=device)
    rs = create_shared_buffer(rs_bytes, group)
    _memset(rs[dist.get_rank(group)], 0, rs_bytes)
    if ll_bytes:  # all-reduce message slots start empty (the 0xFF sentinel); their readers re-arm them
        _memset(rs[dist.get_rank(group)], 0xFF, ll_bytes)
    ag = []
    if ag_bytes:
        ag = create_shared_buffer(ag_bytes, group)
        _memset(ag[dist.get_rank(group)], 0, ag_bytes)
    fams = {_family(config): _FamilyWorkspace(layout, scratch, rs, ag)}
    torch.cuda.synchronize(device)
    dist.barrier(group)
    return Workspace(config, dist.get_rank(group), torch.device(device), fams)


def destroy_workspace(state: Workspace, group) -> None:
    from ..comm.cuda_ipc import free_shared_buffer

    torch.cuda.synchronize(state.device)
    for fw in state.families.values():
        free_shared_buffer(fw.rs, group)
        if fw.ag:
            free_shared_buffer(fw.ag, group)
    state.families.clear()


def _check_device(device) -> None:
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (10, 0):
        raise RuntimeError(
            f"tp_moe sm100: needs sm_100a, got sm_{props.major}{props.minor}"
        )
    if props.multi_processor_count != NUM_SMS:
        raise RuntimeError(
            f"tp_moe sm100: the kernels are compiled for {NUM_SMS} SMs, the device has {props.multi_processor_count}"
        )


# --------------------------------------------------------------------------------------------- weights
# trtllm-gen w2 layout: the output row (inside a 128-row tile) held by TMEM lane l of the FC2 accumulator
_TRTLLM_W2_ROW_OF_LANE = tuple(
    32 * (l // 32) + 4 * (l % 8) + (l % 32) // 8 for l in range(128)
)


@dataclasses.dataclass
class Weights:
    w13: torch.Tensor  # routed gate|up, packed FP4 (kernel layout)
    w13_sf: torch.Tensor  # its UE8M0 scale images
    w2: torch.Tensor
    w2_sf: torch.Tensor
    lanes: torch.Tensor  # uint8 [128]: output row of FC2 accumulator lane l
    w2t: torch.Tensor  # (ragged I) w2 K tails [E, 128, H / 128 * tail / 2]
    w2t_sf: torch.Tensor  # their UE8M0 scale images [E, tail chunks, 512]
    shared_w13: torch.Tensor  # e4m3 [2 SHI, H]: this rank's shared gate rows, then up rows (zero-padded to SHI)
    shared_w2: torch.Tensor  # e4m3 [H, SHI]
    shared_sfa1: torch.Tensor  # UE8M0 SFA images of the shared gate|up block scales
    shared_sfa2: torch.Tensor  # ... of the down projection


def _e8m0_bytes(scale: torch.Tensor) -> torch.Tensor:
    """checkpoint block scales (float8_e8m0fnu / uint8 bytes / fp32 powers of two) -> uint8 biased exponents"""
    if scale.dtype == torch.uint8:
        return scale
    if scale.dtype == getattr(torch, "float8_e8m0fnu", None):
        return scale.view(torch.uint8)
    s = scale.float()
    e = torch.log2(s).round()
    if not torch.equal(torch.exp2(e), s):
        raise ValueError("shared expert scales must be powers of two (UE8M0)")
    return (e + 127).to(torch.uint8)


def _sfa_images(rows_kb: torch.Tensor, n_tiles: int, n_chunks: int) -> torch.Tensor:
    """UE8M0 bytes per (row, K32 block) [rows, 4 n_chunks] -> SFA images [n_tiles][n_chunks][512] (block_scale_interleave
    order within a 128-row x 128-column tile: byte 16 i + 4 g + kb holds row 32 g + i, K block kb); pad bytes 2^0"""
    R, KB = rows_kb.shape
    pad = torch.full(
        (128 * n_tiles, 4 * n_chunks), 127, dtype=torch.uint8, device=rows_kb.device
    )
    pad[:R, :KB] = rows_kb
    return (
        pad.view(n_tiles, 4, 32, n_chunks, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(n_tiles, n_chunks, 512)
    )


def _shared_slice(config, rank, w13, s13, w2, s2, device):
    """this rank's TP slice of the shared expert: gate rows | up rows (zero-padded to SHI), down columns, SFA images"""
    H, I = config.shape.hidden_size, config.shape.shared_intermediate_size
    tp = config.tp_size
    i = I // tp
    shi = dict(_shape_flags(tp))["SHI"] if config.model == "dsv41_flash" else i
    s13 = _e8m0_bytes(s13).to(device)
    s2 = _e8m0_bytes(s2).to(device)
    w13 = w13.to(device)
    w2 = w2.to(device)
    blk = (
        H // s13.shape[1]
    )  # 128 (DeepSeek-V4-Flash) or 32 (DeepSeek-V4.1-Flash) square blocks
    if not (
        s13.shape == (2 * I // blk, H // blk)
        and s2.shape == (H // blk, I // blk)
        and i % blk == 0
    ):
        raise ValueError(
            f"shared expert scales: {tuple(s13.shape)} / {tuple(s2.shape)} for {blk} x {blk} blocks"
        )
    c0, c1 = rank * i, (rank + 1) * i
    gu = torch.zeros(2 * shi, H, dtype=torch.float8_e4m3fn, device=device)
    gu[:i] = w13[c0:c1]
    gu[shi : shi + i] = w13[I + c0 : I + c1]
    dn = torch.zeros(H, shi, dtype=torch.float8_e4m3fn, device=device)
    dn[:, :i] = w2[:, c0:c1]
    # per (row, K32) scale bytes
    r13 = torch.full((2 * shi, H // 32), 127, dtype=torch.uint8, device=device)
    r13[:i] = (
        s13[c0 // blk : c1 // blk]
        .repeat_interleave(blk, 0)
        .repeat_interleave(blk // 32, 1)
    )
    r13[shi : shi + i] = (
        s13[(I + c0) // blk : (I + c1) // blk]
        .repeat_interleave(blk, 0)
        .repeat_interleave(blk // 32, 1)
    )
    r2 = torch.full((H, shi // 32), 127, dtype=torch.uint8, device=device)
    r2[:, : i // 32] = (
        s2[:, c0 // blk : c1 // blk]
        .repeat_interleave(blk, 0)
        .repeat_interleave(blk // 32, 1)
    )
    return (
        gu,
        dn,
        _sfa_images(r13, 2 * shi // 128, H // 128),
        _sfa_images(r2, H // 128, shi // 128),
    )


def _trtllm_routed(config, rank, w13, s13, w2, s2, device, ipad):
    """this rank's TP slice in the trtllm-gen MXFP4 layout (flashinfer's trtllm_fp4_block_scale_moe weight preparation),
    the intermediate zero-padded to ipad columns (zero weights, scales 2^0)"""
    from ..quantization.fp4_quantization import block_scale_interleave
    from .core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    E = config.shape.num_experts
    I = config.shape.intermediate_size
    i = I // config.tp_size
    c0, c1 = rank * i, (rank + 1) * i
    u8 = lambda t: t.view(torch.uint8).to(device)  # noqa: E731
    w13, s13, w2, s2 = u8(w13), u8(s13), u8(w2), u8(s2)

    def pad(t, dim, n, fill):  # n more rows (dim 1) / bytes (dim 2) of fill
        if n == 0:
            return t
        shape = list(t.shape)
        shape[dim] = n
        return torch.cat(
            [t, torch.full(shape, fill, dtype=t.dtype, device=device)], dim
        )

    # w3 (up) rows first, then w1 (gate): the trtllm-gen gated-activation order; each half padded to ipad rows
    w31 = torch.cat(
        [
            pad(w13[:, I + c0 : I + c1], 1, ipad - i, 0),
            pad(w13[:, c0:c1], 1, ipad - i, 0),
        ],
        1,
    ).contiguous()
    s31 = torch.cat(
        [
            pad(s13[:, I + c0 : I + c1], 1, ipad - i, 127),
            pad(s13[:, c0:c1], 1, ipad - i, 127),
        ],
        1,
    ).contiguous()
    w2s = pad(w2[:, :, c0 // 2 : c1 // 2], 2, (ipad - i) // 2, 0).contiguous()
    s2s = pad(s2[:, :, c0 // 32 : c1 // 32], 2, (ipad - i) // 32, 127).contiguous()
    cache: dict = {}
    o13, os13, o2, os2 = [], [], [], []
    for e in range(E):
        p = _maybe_get_cached_w3_w1_permute_indices(cache, w31[e], 128)
        o13.append(w31[e][p.to(device)].contiguous())
        p = _maybe_get_cached_w3_w1_permute_indices(
            cache, s31[e], 128, num_elts_per_sf=16
        )
        os13.append(block_scale_interleave(s31[e][p.to(device)].contiguous()))
        p = get_w2_permute_indices_with_cache(cache, w2s[e], 128)
        o2.append(w2s[e][p.to(device)].contiguous())
        p = get_w2_permute_indices_with_cache(cache, s2s[e], 128, num_elts_per_sf=16)
        os2.append(block_scale_interleave(s2s[e][p.to(device)].contiguous()))
    w13k = torch.stack(o13)
    w2k = torch.stack(o2)
    s13k = torch.stack(os13).view(torch.uint8).reshape(E, -1).contiguous()
    s2k = torch.stack(os2).view(torch.uint8).reshape(E, -1).contiguous()
    lanes = torch.tensor(_TRTLLM_W2_ROW_OF_LANE, dtype=torch.uint8, device=device)
    return w13k, s13k, w2k, s2k, lanes


def _w2_tails(w2k, s2k, i, ipad):
    """(ragged intermediate: i % 128 = 64 or 32) the K tail of every stored w2 row -- columns 128 (i // 128) .. i --
    re-laid out for the kernel's tail-first FC2: [E][row in tile][row tile][tail / 2 bytes], so one 128-column chunk
    holds the tails of 128 / tail consecutive row tiles; and its scale images [E][chunk][512] (block_scale_interleave
    order: byte 16 r + 4 g + b = row 32 g + r of the tile, K block b)"""
    tail = i % 128
    E, H = w2k.shape[0], w2k.shape[1]
    if tail == 0:
        return w2k.new_empty(0), s2k.new_empty(0)
    nrt, tb, kc = H // 128, tail // 32, ipad // 128
    rpc = 4 // tb
    w2t = (
        w2k[:, :, (i - tail) // 2 : i // 2]
        .reshape(E, nrt, 128, tail // 2)
        .permute(0, 2, 1, 3)
        .reshape(E, 128, nrt * tail // 2)
        .contiguous()
    )
    # the tail blocks sit in K atom i // 128 of each row tile's images, blocks 0 .. tb - 1
    src = s2k.view(E, nrt, kc, 32, 4, 4)[:, :, i // 128]  # [E][row tile][r][g][b]
    src = src.reshape(E, nrt // rpc, rpc, 32, 4, 4)
    out = torch.empty(E, nrt // rpc, 32, 4, 4, dtype=torch.uint8, device=w2k.device)
    for kq in range(
        4
    ):  # block kq of a tail chunk: row tile kq // tb of the chunk, its tail block kq % tb
        out[..., kq] = src[:, :, kq // tb, :, :, kq % tb]
    return w2t, out.reshape(E, nrt // rpc, 512).contiguous()


def prepare_weights(
    config,
    rank,
    device,
    *,
    w13,
    w13_scale,
    w2,
    w2_scale,
    shared_w13,
    shared_w13_scale,
    shared_w2,
    shared_w2_scale,
) -> Weights:
    """This rank's kernel weights, built once (nothing is re-laid out per call).

    The routed slice is the trtllm-gen MXFP4 layout of the rank's ``I / tp`` intermediate columns, read in place.
    For DeepSeek-V4.1-Flash, ``I / tp`` is not a multiple of 128 (576 at TP4, 288 at TP8). The kernel then runs each
    FC2 row tile's last partial 128-column chunk from a separate copy of those K tails (``w2t``, ``w2t_sf``). Per MoE
    layer and rank this costs 66.8 MB at TP4 (62.9 MB tails + 3.9 MB scale images) and 33.4 MB at TP8 (31.5 + 2.0 MB): 3.3 % / 2.8 %
    of the padded routed slice (2.0 / 1.2 GB), 3.7 % of the unpadded one.
    """
    shape = config.shape
    E, H, I = shape.num_experts, shape.hidden_size, shape.intermediate_size
    if tuple(w13.shape) != (E, 2 * I, H // 2) or tuple(w2.shape) != (E, H, I // 2):
        raise ValueError(
            f"routed weights: w13 {tuple(w13.shape)}, w2 {tuple(w2.shape)} for E {E} H {H} I {I}"
        )
    if tuple(w13_scale.shape) != (E, 2 * I, H // 32) or tuple(w2_scale.shape) != (
        E,
        H,
        I // 32,
    ):
        raise ValueError(
            f"routed scales: {tuple(w13_scale.shape)} / {tuple(w2_scale.shape)} (UE8M0 per 32)"
        )
    ipad = (
        I // config.tp_size if config.model == "dsv4_flash" else _ipad(config.tp_size)
    )
    w13k, s13k, w2k, s2k, lanes = _trtllm_routed(
        config, rank, w13, w13_scale, w2, w2_scale, device, ipad
    )
    w2t, w2t_sf = _w2_tails(w2k, s2k, I // config.tp_size, ipad)
    gu, dn, sfa1, sfa2 = _shared_slice(
        config, rank, shared_w13, shared_w13_scale, shared_w2, shared_w2_scale, device
    )
    return Weights(w13k, s13k, w2k, s2k, lanes, w2t, w2t_sf, gu, dn, sfa1, sfa2)


# --------------------------------------------------------------------------------------------- run
def run(
    config,
    x,
    weights: Weights,
    workspace: Workspace,
    out,
    *,
    router_logits=None,
    gate_bias=None,
    gate_bias_vision=None,
    input_ids=None,
    image_token_id=None,
    topk_ids=None,
    topk_weights=None,
    enable_pdl: bool = True,
) -> None:
    m = x.shape[0]
    variant = build_for_m(config, m)
    gate_mod, ids_mod = load_module(variant)
    family = variant.gate.family
    fw = workspace.families[family]
    gate_entry = router_logits is not None
    mod = gate_mod if gate_entry else ids_mod
    build = variant.gate if gate_entry else variant.ids
    if _layout_of(build) != fw.layout:
        raise RuntimeError(
            f"tp_moe: build {build.uri} does not match the workspace layout"
        )
    # the routed scaling factor at both entries (the ids entry's topk_weights are the renormalized
    # weights): flash applies it in T1 (factor), generic to the routing weights (gate_scale: the gate
    # to its own, the ids entry to the given ones)
    factor = config.routed_scaling_factor if family == "flash" else 1.0
    if not gate_entry and (
        topk_ids.dtype != torch.int32
        or topk_weights.dtype != torch.float32
        or tuple(topk_ids.shape) != (m, TOP_K)
    ):
        raise ValueError("topk_ids int32 / topk_weights fp32 [M, 6]")
    w = weights
    if family == "flash":
        mod.run(
            x,
            router_logits,
            gate_bias,
            topk_ids,
            topk_weights,
            float(factor),
            w.w13,
            w.w13_sf,
            w.w2,
            w.w2_sf,
            w.lanes,
            w.shared_w13,
            w.shared_w2,
            w.shared_sfa1,
            w.shared_sfa2,
            fw.scratch,
            fw.rs,
            fw.ag,
            out,
            workspace.rank,
            enable_pdl,
        )
    else:
        mod.run(
            x,
            router_logits,
            gate_bias,
            gate_bias_vision,
            input_ids,
            -1 if image_token_id is None else int(image_token_id),
            float(config.routed_scaling_factor),
            topk_ids,
            topk_weights,
            float(factor),
            w.w13,
            w.w13_sf,
            w.w2,
            w.w2_sf,
            w.w2t,
            w.w2t_sf,
            w.lanes,
            w.shared_w13,
            w.shared_w2,
            w.shared_sfa1,
            w.shared_sfa2,
            fw.scratch,
            fw.rs,
            out,
            workspace.rank,
            enable_pdl,
        )
