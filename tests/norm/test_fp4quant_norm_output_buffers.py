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
"""
Output buffers of the fused RMSNorm + FP4 quantization APIs
(``rmsnorm_fp4quant`` and ``add_rmsnorm_fp4quant``): the padding of a swizzled
scale-factor buffer is zero-filled like ``fp4_quantize`` does, and strided
caller buffers are rejected instead of being written through a temporary copy.
Numerics are covered by test_rmsnorm_fp4_quant_cute_dsl.py and
test_add_rmsnorm_fp4_quant_cute_dsl.py.
"""

import pytest
import torch

from flashinfer.cute_dsl.utils import is_cute_dsl_arch_supported, is_cute_dsl_available


def _fp4_norms_supported():
    if not torch.cuda.is_available() or not is_cute_dsl_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major >= 10 and is_cute_dsl_arch_supported(major, minor)


pytestmark = pytest.mark.skipif(
    not _fp4_norms_supported(),
    reason="FP4 norms need SM100+ and a CuTe DSL that targets this GPU",
)

# fmt -> (block_size, scale dtype)
_FORMATS = {
    "nvfp4": (16, torch.float8_e4m3fn),
    "mxfp4": (32, torch.uint8),
}


def _inputs(m, h, dtype=torch.bfloat16):
    gen = torch.Generator(device="cuda").manual_seed(1234 + 7 * m + h)
    x = torch.randn(m, h, generator=gen, device="cuda").to(dtype)
    r = torch.randn(m, h, generator=gen, device="cuda").to(dtype)
    w = (torch.rand(h, generator=gen, device="cuda") + 0.5).to(dtype)
    gs = torch.tensor([2.0], dtype=torch.float32, device="cuda")
    return x, r, w, gs


def _call(api, fmt, swizzled, x, r, w, gs, **outputs):
    from flashinfer.cute_dsl import add_rmsnorm_fp4quant, rmsnorm_fp4quant

    kwargs = dict(
        global_scale=gs,
        block_size=_FORMATS[fmt][0],
        is_sf_swizzled_layout=swizzled,
        **outputs,
    )
    if api == "rms":
        return rmsnorm_fp4quant(x, w, **kwargs)
    return add_rmsnorm_fp4quant(x, r, w, **kwargs)


def _valid_swizzled_scale_mask(m, nsf):
    """Positions of a 128x4-tile swizzled scale buffer that hold a real scale."""
    k_tiles = (nsf + 3) // 4
    mask = torch.zeros(
        ((m + 127) // 128) * k_tiles * 512, dtype=torch.bool, device="cuda"
    )
    rows = torch.arange(m, device="cuda").view(-1, 1)
    cols = torch.arange(nsf, device="cuda").view(1, -1)
    offset = (
        (rows // 128) * (k_tiles * 512)
        + (cols // 4) * 512
        + (rows % 32) * 16
        + ((rows % 128) // 32) * 4
        + cols % 4
    )
    mask[offset.flatten()] = True
    return mask


# (M, H): M = 1 and 129 leave 127 padding rows in the last 128-row tile;
# H = 64 with 32-wide blocks leaves two padding k-blocks in its 4-wide k tile.
@pytest.mark.parametrize("fmt", list(_FORMATS))
@pytest.mark.parametrize("shape", [(1, 2048), (129, 2048), (129, 64), (256, 4096)])
@pytest.mark.parametrize("api", ["rms", "add"])
def test_swizzled_scale_padding_is_zero(api, shape, fmt):
    """Rows past M and k-blocks past H // block_size of the swizzled scale
    buffer are zero, as fp4_quantize writes them, whatever the buffer held."""
    m, h = shape
    block, sdtype = _FORMATS[fmt]
    x, r, w, gs = _inputs(m, h)
    valid = _valid_swizzled_scale_mask(m, h // block)

    from flashinfer import fp4_quantize

    _, sf_ref = fp4_quantize(
        x,
        gs,
        sf_vec_size=block,
        sf_use_ue8m0=fmt == "mxfp4",
        is_sf_swizzled_layout=True,
    )
    sf_ref = sf_ref.view(torch.uint8).flatten()
    assert sf_ref.numel() == valid.numel()
    assert (sf_ref[~valid] == 0).all()

    _, s_auto = _call(api, fmt, True, x.clone(), r.clone(), w, gs)
    for poison in (0xFF, 0x00):
        s = torch.full((valid.numel(),), poison, dtype=torch.uint8, device="cuda")
        _call(api, fmt, True, x.clone(), r.clone(), w, gs, block_scale=s.view(sdtype))
        torch.cuda.synchronize()
        pad = s[~valid]
        assert (pad == 0).all(), (
            f"{int((pad != 0).sum())} of {pad.numel()} padding bytes are not zero"
        )
        assert torch.equal(s, s_auto.view(torch.uint8).flatten())


@pytest.mark.parametrize("fmt", list(_FORMATS))
@pytest.mark.parametrize(
    "api,which",
    [
        ("rms", "y_fp4"),
        ("rms", "block_scale"),
        ("add", "y_fp4"),
        ("add", "block_scale"),
        ("add", "residual"),
    ],
)
def test_strided_buffers_are_rejected(api, which, fmt):
    """A row-padded caller buffer is rejected: the kernel addresses it as
    row-major, and writing a temporary copy would leave the caller's buffer
    unchanged."""
    m, h = 129, 2048
    block, sdtype = _FORMATS[fmt]
    x, r, w, gs = _inputs(m, h)
    outputs = {}
    if which == "y_fp4":
        storage = torch.zeros(m, h // 2 + 32, dtype=torch.uint8, device="cuda")
        outputs["y_fp4"] = storage[:, 16 : 16 + h // 2].view(torch.float4_e2m1fn_x2)
    elif which == "block_scale":
        storage = torch.zeros(m, h // block + 8, dtype=torch.uint8, device="cuda")
        outputs["block_scale"] = storage[:, 4 : 4 + h // block].view(sdtype)
    else:
        storage = torch.zeros(m, h + 16, dtype=r.dtype, device="cuda")
        r = storage[:, 8 : 8 + h]
    with pytest.raises((AssertionError, ValueError, RuntimeError), match="contiguous"):
        _call(api, fmt, False, x, r, w, gs, **outputs)
