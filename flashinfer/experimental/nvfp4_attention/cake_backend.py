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
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import tvm_ffi

from .cake_jit import MODULES, load_cake_nvfp4_attention_module

SF_VEC = 16


def _quantize_nvfp4(x):
    """Quantize the last dimension to packed E2M1 plus positive E4M3 scales."""
    import torch

    x_f32 = x.float()
    groups = x_f32.shape[-1] // SF_VEC
    blocks = x_f32.reshape(*x_f32.shape[:-1], groups, SF_VEC)
    raw_scale = (blocks.abs().amax(dim=-1) / 6.0).clamp_min(2.0**-9)
    scale_fp8 = raw_scale.to(torch.float8_e4m3fn)
    scale = scale_fp8.float()
    normalized = blocks / scale.unsqueeze(-1)
    ax = normalized.abs().clamp_max(6.0)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    code = torch.bucketize(ax, boundaries).to(torch.uint8)
    code |= ((normalized < 0) & (code != 0)).to(torch.uint8) << 3
    code = code.reshape(*x_f32.shape)
    pairs = code.reshape(*code.shape[:-1], code.shape[-1] // 2, 2)
    packed = (pairs[..., 0] & 0x0F) | ((pairs[..., 1] & 0x0F) << 4)
    return packed.contiguous(), scale_fp8.view(torch.uint8).contiguous()


def _prepack_sf_tiles(scale_u8):
    """Pack ``[tiles,128,8]`` natural UE4M3 bytes into UTCCP source tiles."""
    tiles = scale_u8.shape[0]
    sf = scale_u8.reshape(tiles, 4, 32, 1, 8)
    sf = sf.reshape(tiles, 4, 4, 8, 1, 2, 4)
    sf = sf.permute(0, 4, 2, 5, 3, 1, 6).contiguous()
    return sf.reshape(tiles * 32, 32)


def _prepack_sf_tiles_split_ksets(scale_u8):
    """Pack two 4-scale K-sets as separate complete 512B UTCCP tensors."""
    tiles = scale_u8.shape[0]

    def pack_half(values):
        sf = values.contiguous().reshape(tiles, 4, 32, 1, 4)
        sf = sf.reshape(tiles, 4, 4, 8, 1, 1, 4)
        sf = sf.permute(0, 4, 2, 5, 3, 1, 6).contiguous()
        return sf.reshape(tiles, 512)

    lo = pack_half(scale_u8[..., :4]).reshape(tiles * 16, 32)
    hi = pack_half(scale_u8[..., 4:]).reshape(tiles * 16, 32)
    return lo, hi


@dataclass(frozen=True)
class PreparedNVFP4Attention:
    """Prepared packed tensors, host launch metadata and caller-owned output."""

    module_name: str
    main_kwargs: dict
    out: torch.Tensor
    entry: object
    arguments: tuple

    def launch(self):
        # The binding encodes host-known TMA descriptors by value. No device
        # descriptor update, scratch allocation or graph capture occurs here.
        with tvm_ffi.use_torch_stream():
            self.entry(*self.arguments)
        return self.out

    __call__ = launch


def bind_attention_payload(module_name, main_kwargs, out):
    """Bind prepacked buffers to the generated physical argument order."""
    record = MODULES[module_name]
    if torch.cuda.get_device_capability(out.device) != (10, 3):
        raise ValueError("NVFP4 attention requires compute capability 10.3")
    grid = main_kwargs["grid"]
    grid_values = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    arguments = tuple(
        grid_values[key] if kind == "grid" else main_kwargs[key]
        for kind, key in record["arg_plan"]
    )
    module = load_cake_nvfp4_attention_module(module_name)
    return PreparedNVFP4Attention(
        module_name, main_kwargs, out, getattr(module, record["ffi_entry"]), arguments
    )


def prepare_nvfp4_attention(q, k, v, out, *, causal=False, backend="cake"):
    if backend != "cake":
        raise ValueError("NVFP4 attention supports backend='cake'")
    if (
        q.ndim != 4
        or q.shape[-1] != 128
        or k.shape != q.shape
        or v.shape != q.shape
        or out.shape != q.shape
    ):
        raise ValueError("Expected matching [B,H,S,128] inputs and output")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v, out)):
        raise TypeError("NVFP4 attention inputs and output must be BF16")
    if not all(
        x.is_cuda and x.is_contiguous() and x.device == q.device for x in (q, k, v, out)
    ):
        raise ValueError("Expected contiguous tensors on one CUDA device")
    batch, heads, seqlen, dim = q.shape
    if min(batch, heads, seqlen) <= 0 or causal or seqlen % 512:
        raise ValueError(
            "This route requires positive extents, noncausal attention and S divisible by 512"
        )
    if torch.cuda.get_device_capability(q.device) != (10, 3):
        raise ValueError("NVFP4 attention requires compute capability 10.3")
    bh = batch * heads
    sources = [x.reshape(bh, seqlen, dim) for x in (q, k, v)]
    packed_q, sq = _quantize_nvfp4(sources[0])
    packed_k, sk = _quantize_nvfp4(sources[1])
    blocks = seqlen // 128
    sq = _prepack_sf_tiles(sq.reshape(bh, blocks, 128, 8).reshape(-1, 128, 8))
    sk = _prepack_sf_tiles(sk.reshape(bh, blocks, 128, 8).reshape(-1, 128, 8))
    packed_v, sv = _quantize_nvfp4(sources[2].transpose(-1, -2).contiguous())
    scales = sv.reshape(bh, 128, blocks, 8).permute(0, 2, 1, 3).contiguous()
    lo, hi = _prepack_sf_tiles_split_ksets(scales.reshape(-1, 128, 8))
    tiles = bh * (seqlen // 512)
    grid_x = 2 * min(
        torch.cuda.get_device_properties(q.device).multi_processor_count // 2, tiles
    )
    kwargs = dict(
        grid=(grid_x, 1, 1),
        Q=packed_q.reshape(bh * seqlen, 64),
        K=packed_k.reshape(bh * seqlen, 64),
        SFQ=sq,
        SFK=sk,
        Vt=packed_v.reshape(bh * 128, seqlen // 2),
        SFVtLo=lo,
        SFVtHi=hi,
        O=out.reshape(bh, seqlen, dim),
        seqlen_q=seqlen,
        seqlen_kv=seqlen,
        q_stride=seqlen,
        kv_stride=seqlen,
        total_bh=bh,
        softmax_scale_log2=1.0 / math.sqrt(dim) / math.log(2.0),
    )
    return bind_attention_payload(
        "cake_nvfp4_attention_8194d2d0334d525c11e4", kwargs, out
    )
