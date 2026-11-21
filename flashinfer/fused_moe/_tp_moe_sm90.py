# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""SM90a (H20 / H100 / H200) implementation of :mod:`flashinfer.fused_moe.tp_moe`.

Protocol (called by tp_moe.py only):

* ``build_for_m(config, m)`` -> hashable variant (source, flags, M range) serving ``m`` tokens.
* ``load_module(variant)`` -> the built and loaded JIT module.
* ``create_workspace(config, group, device)`` / ``destroy_workspace(state, group)``: collective.
* ``prepare_weights(config, rank, device, **checkpoint_tensors)`` -> weight state.
* ``run(config, x, weights, workspace, out, **routing)``: launches on the current stream.

One persistent kernel per call (csrc/tp_moe/sm90, include/flashinfer/tp_moe/sm90): fused gate, FP8 quantization of x, the routed
MXFP4 experts of this rank's TP slice, the TP-sliced FP8 shared expert and the cross-rank all-reduce over CUDA-IPC buffers.

Routed weights use Humming's W4A8 fused-E8M0 layout (the layout SGLang's Humming MoE backend keeps on Hopper); DeepSeek-V4.1 keeps
its intermediate tail (2304 / TP is not a multiple of 128) in separate 64-row blocks.  DeepSeek-V4.1's image tokens are gated with
their own bias inside the kernel's fused gate (input_ids / image_token_id select it per token).  ``prepare_weights`` builds both layouts from
checkpoint tensors with the routines below, bit for bit what Humming (humming-kernels 0.1.10) produces.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Dict, List, Tuple

import numpy as np
import torch

__all__: List[str] = []

MAX_TOKENS = 64  # tokens per call (the kernels' MAX_TOKENS)


# ----------------------------------------------------------------------------------------------------------------- builds
@dataclasses.dataclass(frozen=True)
class Variant:
    """One compiled kernel: model, TP size and the compile-time flags of an M range."""

    model: str
    tp_size: int
    tag: str
    flags: Tuple[Tuple[str, int], ...] = ()

    @property
    def name(self) -> str:
        return f"tp_moe_sm90_{self.model}_tp{self.tp_size}_{self.tag}"


# One build per (model, TP size) serves every M; DeepSeek-V4-Flash TP8 keeps its FC2 outputs out of the L2 evict_last set.
_DISPATCH: Dict[Tuple[str, int], List[Tuple[int, int, Variant]]] = {
    ("dsv4_flash", 8): [
        (1, 64, Variant("dsv4_flash", 8, "l2off", (("TP_MOE_OUTP_L2", 0),)))
    ],
    ("dsv4_flash", 4): [(1, 64, Variant("dsv4_flash", 4, "base"))],
    ("dsv41_flash", 4): [(1, 64, Variant("dsv41_flash", 4, "base"))],
    ("dsv41_flash", 8): [(1, 64, Variant("dsv41_flash", 8, "base"))],
}


def build_for_m(config, m: int) -> Variant:
    if not 1 <= m <= min(config.max_num_tokens, MAX_TOKENS):
        raise ValueError(f"M = {m} outside [1, {config.max_num_tokens}]")
    for lo, hi, variant in _DISPATCH[(config.model, config.tp_size)]:
        if lo <= m <= hi:
            return variant
    raise AssertionError("unreachable")


def _variants(config) -> List[Variant]:
    out: List[Variant] = []
    for _, _, v in _DISPATCH[(config.model, config.tp_size)]:
        if v not in out:
            out.append(v)
    return out


@functools.cache
def load_module(variant: Variant):
    from ..jit.tp_moe import gen_tp_moe_sm90_module

    return gen_tp_moe_sm90_module(variant).build_and_load()


@dataclasses.dataclass(frozen=True)
class _Constants:
    maxm: int
    ndev: int
    dim: int
    nexp: int
    inter: int
    topk: int
    rs_bytes: int
    ag_bytes: int


def _constants(variant: Variant) -> _Constants:
    return _Constants(*(int(v) for v in load_module(variant).constants()))


# ----------------------------------------------------------------------------------------------------------------- workspace
@dataclasses.dataclass
class _Workspace:
    """Per-rank kernel state, shared by every layer: one zeroed scratch buffer plus the CUDA-IPC exchange buffers."""

    variants: List[Variant]
    rank: int
    device: torch.device
    buf: torch.Tensor
    rs_ptrs: List[int]
    ag_ptrs: List[int]


def create_workspace(config, group, device):
    import torch.distributed as dist

    from ..comm.cuda_ipc import create_shared_buffer

    torch.cuda.set_device(device)
    variants = _variants(config)
    consts = [_constants(v) for v in variants]
    c0 = consts[0]
    for v, c in zip(variants, consts, strict=False):
        if (c.rs_bytes, c.ag_bytes, c.maxm, c.ndev) != (
            c0.rs_bytes,
            c0.ag_bytes,
            c0.maxm,
            c0.ndev,
        ):
            raise RuntimeError(
                f"build {v.name} disagrees on the workspace layout: {c} vs {c0}"
            )
    if c0.ndev != config.tp_size:
        raise RuntimeError(f"build is TP{c0.ndev}, config TP{config.tp_size}")
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    mod = load_module(variants[0])
    nbytes = int(mod.workspace_bytes(num_sms))
    for v in variants[1:]:
        assert int(load_module(v).workspace_bytes(num_sms)) == nbytes
    buf = torch.zeros(nbytes, dtype=torch.uint8, device=device)
    rs = create_shared_buffer(c0.rs_bytes, group)
    ag = create_shared_buffer(c0.ag_bytes, group)
    rank = dist.get_rank(group)
    mod.zero_device_memory(rs[rank], c0.rs_bytes)
    mod.zero_device_memory(ag[rank], c0.ag_bytes)
    torch.cuda.synchronize(device)
    dist.barrier(group)
    return _Workspace(variants, rank, torch.device(device), buf, rs, ag)


def destroy_workspace(state: _Workspace, group) -> None:
    from ..comm.cuda_ipc import free_shared_buffer

    torch.cuda.synchronize(state.device)
    free_shared_buffer(state.rs_ptrs, group)
    free_shared_buffer(state.ag_ptrs, group)
    state.buf = None


# ----------------------------------------------------------------------------------------------------------------- weights
@functools.cache
def _rescale_lut() -> np.ndarray:
    """[16 codes][256 deltas] -> e2m1 code of code * 2^-delta, rounded as Humming's process_mxfp4_w4a8 kernel rounds it."""
    codes = np.arange(16, dtype=np.uint32)[:, None]
    deltas = np.arange(256, dtype=np.uint32)[None, :]
    v = np.where(codes == 8, 0, codes).astype(np.uint32)  # -0 -> +0
    f = (((v & 8) << 28) | ((v & 7) << 22)).astype(np.uint32).view(np.float32)
    s = (
        ((np.uint32(0x3F800000) - (deltas << np.uint32(23))) & np.uint32(0xFFFFFFFF))
        .astype(np.uint32)
        .view(np.float32)
    )
    with np.errstate(all="ignore"):
        val = (f * s).astype(np.float32)
        u = val.view(np.uint32)
        rz = (u & np.uint32(0x81C00000)).view(np.float32)
        ru = ((u + np.uint32(0x00200000)) & np.uint32(0x81C00000)).view(np.float32)
        d_rz = np.abs(val - rz).astype(np.float32)
        d_ru = np.abs(val - ru).astype(np.float32)
        rn = np.where(d_rz >= d_ru, ru, rz).view(np.uint32)
    q = ((rn & np.uint32(0x80000000)) >> 28) | ((rn & np.uint32(0x01C00000)) >> 22)
    out = np.where(deltas == 0, np.broadcast_to(v, q.shape), q)
    return out.astype(np.uint8)


@functools.cache
def _repack_index() -> Tuple[np.ndarray, ...]:
    """Humming's weight_repack_nk (4-bit weights, 8-bit activations, interleave mode 2) as a gather: for one 64 x 64 (N x K) tile,
    output words [2 rows][256 columns][8 nibbles] take their low 3 bits from code (lo_r, lo_k) and their sign bit from (sg_r, sg_k)."""
    shp = (2, 256, 8)
    lo_r, lo_k, sg_r, sg_k = (np.zeros(shp, np.int64) for _ in range(4))
    for tid in range(32):
        for oi in range(2):
            for oj in range(2):
                for ok in range(4):
                    pi, pj, c = oi * 2 + oj, ok, oj * 128 + tid * 4 + ok
                    for pk in range(8):
                        for dst_r, dst_k, kk in (
                            (lo_r, lo_k, pk),
                            (sg_r, sg_k, (pk % 2) * 4 + pk // 2),
                        ):
                            f = pi * 32 + pj * 8 + kk  # tmp[2][2][2][2][2][4]
                            i6, f = f % 4, f // 4
                            i5, f = f % 2, f // 2
                            i4, f = f % 2, f // 2
                            i3, f = f % 2, f // 2
                            i2, i1 = f % 2, f // 2
                            row = (i2 * 4 + i3 * 2 + i4) * 8 + tid // 4
                            kidx = (i1 * 2 + i5) * 16 + (tid % 4) * 4 + i6
                            dst_r[oi, c, pk], dst_k[oi, c, pk] = row, kidx
    return lo_r, lo_k, sg_r, sg_k


def _unpack_fp4(w: torch.Tensor) -> torch.Tensor:
    """packed [.., K/2] (element 2i in the low nibble) -> codes [.., K] uint8"""
    return torch.stack((w & 0xF, w >> 4), dim=-1).flatten(-2)


def _humming_w4a8(w: torch.Tensor, s: torch.Tensor, pad_n: int, pad_k: int):
    """One Humming sublayer: MXFP4 weights uint8 [E, N, K/2] + UE8M0 scales uint8 [E, N, K/32] -> (int32 [E, K'/32, 4 N'],
    uint8 scales [E, K'/32, N'], fp32 global scale [E]) with N, K padded to multiples of pad_n, pad_k (zero codes, scale 2^0).
    Fused-E8M0 form for FP8 activations: each expert's exponent window is clamped to 12 values, codes below the window rescaled."""
    E, N, K = w.shape[0], w.shape[1], w.shape[2] * 2
    Np, Kp = -(-N // pad_n) * pad_n, -(-K // pad_k) * pad_k
    if (Np, Kp) != (N, K):
        w = torch.nn.functional.pad(w, (0, (Kp - K) // 2, 0, Np - N), value=0)
        s = torch.nn.functional.pad(s, (0, (Kp - K) // 32, 0, Np - N), value=127)
    s32 = s.to(torch.int32)
    smax = s32.view(E, -1).amax(1)
    smin = s32.view(E, -1).amin(1)
    lo = smax - (smax - smin).clamp(max=11)
    s_clamped = torch.maximum(s32, lo.view(E, 1, 1))
    delta = (s_clamped - s32).to(torch.int64)
    s_new = (s_clamped - lo.view(E, 1, 1) + 1).to(torch.uint8)
    glob = torch.from_numpy(
        np.ldexp(np.float32(1.0), (lo.cpu().numpy() - 128).astype(np.int32)).astype(
            np.float32
        )
    ).to(w.device)
    lut = torch.from_numpy(_rescale_lut()).to(w.device)
    idx = [torch.from_numpy(a).to(w.device) for a in _repack_index()]
    out = torch.empty((E, Kp // 32, 4 * Np), dtype=torch.int32, device=w.device)
    for e in range(E):
        codes = _unpack_fp4(w[e])  # [Np, Kp]
        codes = lut[codes.long(), delta[e].repeat_interleave(32, dim=-1)]
        tiles = codes.view(Np // 64, 64, Kp // 64, 64).permute(
            0, 2, 1, 3
        )  # [bx, by, row, k]
        lo3 = tiles[:, :, idx[0], idx[1]] & 7
        sgn = tiles[:, :, idx[2], idx[3]] & 8
        nib = lo3 | sgn  # [bx, by, 2, 256, 8]
        byte = (nib[..., 0::2] | (nib[..., 1::2] << 4)).contiguous()
        words = byte.view(torch.int32).squeeze(-1)  # [bx, by, 2, 256]
        out[e] = words.permute(1, 2, 0, 3).reshape(Kp // 32, 4 * Np)
    s_out = (
        s_new.transpose(1, 2)
        .reshape(E, Kp // 32, Np // 64, 8, 8)
        .transpose(-1, -2)
        .reshape(E, Kp // 32, Np)
        .contiguous()
    )
    return out, s_out, glob


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.element_size() == 1 else t


def _scale_f32(s: torch.Tensor) -> torch.Tensor:
    """checkpoint block scales (UE8M0 bytes / float8_e8m0fnu / fp32) -> fp32"""
    if s.dtype == torch.uint8:
        s = s.view(torch.float8_e8m0fnu)
    return s.float()


@dataclasses.dataclass
class _Weights:
    tensors: Dict[str, torch.Tensor]
    maps: torch.Tensor  # device uint8: the routed weights' TMA descriptors
    sh_maps: torch.Tensor  # CPU uint8: the shared expert's TMA descriptors (copied into the kernel parameters)
    n_exp: int


def prepare_weights(config, rank, device, **t):
    shp = config.shape
    tp = config.tp_size
    H, I, E, Is = (
        shp.hidden_size,
        shp.intermediate_size,
        shp.num_experts,
        shp.shared_intermediate_size,
    )
    i = I // tp
    dev = torch.device(device)
    w13, s13 = _u8(t["w13"]), _u8(t["w13_scale"])
    w2, s2 = _u8(t["w2"]), _u8(t["w2_scale"])
    if tuple(w13.shape) != (E, 2 * I, H // 2) or tuple(s13.shape) != (
        E,
        2 * I,
        H // 32,
    ):
        raise ValueError(
            f"w13 / w13_scale must be [{E}, {2 * I}, {H // 2}] / [{E}, {2 * I}, {H // 32}] uint8"
        )
    if tuple(w2.shape) != (E, H, I // 2) or tuple(s2.shape) != (E, H, I // 32):
        raise ValueError(
            f"w2 / w2_scale must be [{E}, {H}, {I // 2}] / [{E}, {H}, {I // 32}] uint8"
        )
    r0 = i * rank
    g, gs = w13[:, r0 : r0 + i].to(dev), s13[:, r0 : r0 + i].to(dev)
    u, us = w13[:, I + r0 : I + r0 + i].to(dev), s13[:, I + r0 : I + r0 + i].to(dev)
    d, ds = (
        w2[:, :, r0 // 2 : (r0 + i) // 2].to(dev),
        s2[:, :, r0 // 32 : (r0 + i) // 32].to(dev),
    )
    out: Dict[str, torch.Tensor] = {}
    if config.model == "dsv4_flash":
        out["w1"], out["s1"], out["f1"] = _humming_w4a8(
            torch.cat([g, u], 1), torch.cat([gs, us], 1), 256, 128
        )
        out["w2"], out["s2"], out["f2"] = _humming_w4a8(
            d.contiguous(), ds.contiguous(), 256, 128
        )
        out.update(_flash_shared(t, H, Is, tp, rank, dev))
    else:
        out.update(_v41_routed(g, gs, u, us, d.contiguous(), ds.contiguous(), H, i, tp))
        out.update(_v41_shared(t, H, Is, tp, rank, dev))
    del g, gs, u, us, d, ds
    out = {k: v.contiguous() for k, v in out.items()}
    mod = load_module(_variants(config)[0])
    maps = torch.zeros(8 * 128, dtype=torch.uint8)
    sh_maps = torch.zeros(2 * 128, dtype=torch.uint8)
    empty_u8 = torch.empty(0, dtype=torch.uint8, device=dev)
    args = [out["w1"], out["s1"], out["w2"], out["s2"], out["sh_w1"], out["sh_w2"]]
    if config.model == "dsv41_flash":
        args += [
            out.get("w1t", empty_u8),
            out.get("s1t", empty_u8),
            out.get("w1t4", empty_u8),
            out.get("s1t4", empty_u8),
        ]
    else:
        maps = maps[: 4 * 128].contiguous()
    n_exp = int(mod.encode_weight_maps(*args, maps, sh_maps))
    return _Weights(out, maps.to(dev), sh_maps, n_exp)


def _flash_shared(t, H, Is, tp, rank, dev):
    """this rank's TP slice of the 128 x 128 block-FP8 shared expert (the SGLang shared expert's own sharding, used in place)"""
    w13, s13 = t["shared_w13"], _scale_f32(t["shared_w13_scale"])
    w2, s2 = t["shared_w2"], _scale_f32(t["shared_w2_scale"])
    if tuple(w13.shape) != (2 * Is, H) or tuple(s13.shape) != (2 * Is // 128, H // 128):
        raise ValueError(
            f"shared_w13 / scale must be [{2 * Is}, {H}] / [{2 * Is // 128}, {H // 128}] (128 x 128 blocks)"
        )
    if tuple(w2.shape) != (H, Is) or tuple(s2.shape) != (H // 128, Is // 128):
        raise ValueError(
            f"shared_w2 / scale must be [{H}, {Is}] / [{H // 128}, {Is // 128}] (128 x 128 blocks)"
        )
    inter, nb = Is // tp, Is // tp // 128
    w13u, w2u = _u8(w13), _u8(w2)
    gu_w = torch.cat(
        [
            w13u[inter * rank : inter * (rank + 1)],
            w13u[Is + inter * rank : Is + inter * (rank + 1)],
        ]
    )
    gu_s = torch.cat(
        [
            s13[nb * rank : nb * (rank + 1)],
            s13[Is // 128 + nb * rank : Is // 128 + nb * (rank + 1)],
        ]
    )
    dn_w = w2u[:, inter * rank : inter * (rank + 1)]
    dn_s = s2[:, nb * rank : nb * (rank + 1)]
    return dict(
        sh_w1=gu_w.to(dev), sh_s1=gu_s.to(dev), sh_w2=dn_w.to(dev), sh_s2=dn_s.to(dev)
    )


def _v41_routed(g, gs, u, us, d, ds, H, i, tp):
    """DeepSeek-V4.1 routed layout: main w13 stacks [gate | up] of the first im = 128 floor(i / 128) rows; the tail rows in one
    64-row block per stack (zero codes below; their scales copy the last real row, so the exponent window is the real one's); at TP8
    the 32-row tail additionally split 4 ways over K; w2 with K = i (Humming pads K to 64, the zero pad slice is dropped)."""
    E = g.shape[0]
    out: Dict[str, torch.Tensor] = {}
    im = (i // 128) * 128
    out["w1"], out["s1"], out["f1"] = _humming_w4a8(
        torch.cat([g[:, :im], u[:, :im]], 1),
        torch.cat([gs[:, :im], us[:, :im]], 1),
        256,
        128,
    )
    nt = i - im
    if nt:
        assert nt <= 64
        z = torch.zeros(E, 64 - nt, H // 2, dtype=torch.uint8, device=g.device)
        zg = gs[:, im : im + 1].expand(E, 64 - nt, H // 32)
        zu = us[:, im : im + 1].expand(E, 64 - nt, H // 32)
        out["w1t"], out["s1t"], out["f1t"] = _humming_w4a8(
            torch.cat([g[:, im:], z, u[:, im:], z], 1),
            torch.cat([gs[:, im:], zg, us[:, im:], zu], 1),
            64,
            128,
        )
        if nt == 32:
            b64 = torch.cat([g[:, im:], u[:, im:]], 1)
            s64 = torch.cat([gs[:, im:], us[:, im:]], 1)
            nj = H // 512
            w4 = (
                b64.view(E, 64, nj, 4, 64)
                .permute(0, 3, 1, 2, 4)
                .reshape(E, 256, H // 8)
                .contiguous()
            )
            s4 = (
                s64.view(E, 64, nj, 4, 4)
                .permute(0, 3, 1, 2, 4)
                .reshape(E, 256, H // 128)
                .contiguous()
            )
            out["w1t4"], out["s1t4"], out["f1t4"] = _humming_w4a8(w4, s4, 128, 128)
    w2, s2, f2 = _humming_w4a8(d, ds, 256, 64)
    nsl = (
        i // 32
    )  # k32 slices of w2 actually holding weights (the K pad slice of TP8 goes)
    if w2.shape[1] > nsl:
        assert not w2[:, nsl:].any(), "w2 pad slices are not zero"
        w2, s2 = w2[:, :nsl], s2[:, :nsl]
    out["w2"], out["s2"], out["f2"] = w2, s2, f2
    return out


def _v41_shared(t, H, Is, tp, rank, dev):
    """this rank's slice of the DSV4.1 shared expert (FP8, UE8M0 32 x 32 block scales), zero-padded to the kernel's INTER columns
    (384 at TP8, 640 at TP4), scales expanded to fp32 per (32 rows, 32 k) with 1.0 in the padding"""
    w13, s13 = t["shared_w13"], _scale_f32(t["shared_w13_scale"])
    w2, s2 = t["shared_w2"], _scale_f32(t["shared_w2_scale"])
    if tuple(w13.shape) != (2 * Is, H) or tuple(s13.shape) != (2 * Is // 32, H // 32):
        raise ValueError(
            f"shared_w13 / scale must be [{2 * Is}, {H}] / [{2 * Is // 32}, {H // 32}] (32 x 32 blocks)"
        )
    if tuple(w2.shape) != (H, Is) or tuple(s2.shape) != (H // 32, Is // 32):
        raise ValueError(
            f"shared_w2 / scale must be [{H}, {Is}] / [{H // 32}, {Is // 32}] (32 x 32 blocks)"
        )
    i = Is // tp
    inter_pad = -(-i // 128) * 128
    r0 = i * rank
    w13u, w2u = _u8(w13).to(dev), _u8(w2).to(dev)
    s13, s2 = s13.to(dev), s2.to(dev)
    gu = torch.zeros(2 * inter_pad, H, dtype=torch.uint8, device=dev)
    gu[:i] = w13u[r0 : r0 + i]
    gu[inter_pad : inter_pad + i] = w13u[Is + r0 : Is + r0 + i]
    gus = torch.ones(2 * inter_pad // 32, H // 32, dtype=torch.float32, device=dev)
    gus[: i // 32] = s13[r0 // 32 : (r0 + i) // 32]
    gus[inter_pad // 32 : (inter_pad + i) // 32] = s13[
        (Is + r0) // 32 : (Is + r0 + i) // 32
    ]
    dn = torch.zeros(H, inter_pad, dtype=torch.uint8, device=dev)
    dn[:, :i] = w2u[:, r0 : r0 + i]
    dns = torch.ones(H // 32, inter_pad // 32, dtype=torch.float32, device=dev)
    dns[:, : i // 32] = s2[:, r0 // 32 : (r0 + i) // 32]
    return dict(sh_w1=gu, sh_s1=gus, sh_w2=dn, sh_s2=dns)


# ----------------------------------------------------------------------------------------------------------------- run
def run(
    config,
    x,
    weights: _Weights,
    workspace: _Workspace,
    out,
    *,
    router_logits=None,
    gate_bias=None,
    gate_bias_vision=None,
    input_ids=None,
    image_token_id=None,
    topk_ids=None,
    topk_weights=None,
) -> None:
    """Launch on the current stream (graph-capture safe, PDL on).

    The kernel tags its flags with a launch epoch kept in the workspace, so captured launches replay without any reset."""
    m = x.shape[0]
    variant = build_for_m(config, m)
    mod = load_module(variant)
    ws = workspace
    w = weights.tensors
    if router_logits is not None:
        router_logits = router_logits.float().contiguous()
        gate_bias = gate_bias.float().contiguous()
    else:
        topk_ids = topk_ids.to(torch.int32).contiguous()
        topk_weights = topk_weights.float().contiguous()
    args = [
        x.contiguous(),
        weights.maps,
        w["f1"],
        w["f2"],
        weights.n_exp,
        weights.sh_maps,
        w["sh_s1"],
        w["sh_s2"],
    ]
    if config.model == "dsv41_flash":
        # image tokens (input_ids == image_token_id) are gated with gate_bias_vision inside the kernel's fused gate
        if router_logits is not None and input_ids is not None:
            gate_bias_vision = gate_bias_vision.float().contiguous()
            input_ids = input_ids.to(torch.int32).contiguous()
        args += [
            w["f1t"],
            w.get("f1t4", w["f1t"]),
            router_logits,
            gate_bias,
            gate_bias_vision,
            input_ids,
            int(image_token_id) if image_token_id is not None else -1,
        ]
    else:
        args += [router_logits, gate_bias]
    mod.run(
        *args,
        topk_ids,
        topk_weights,
        ws.buf,
        ws.rs_ptrs,
        ws.ag_ptrs,
        out,
        ws.rank,
        float(config.routed_scaling_factor),
        True,
    )
