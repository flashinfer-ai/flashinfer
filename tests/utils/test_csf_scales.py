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

import numpy as np
import pytest
import torch

from flashinfer.quantization import (
    SfLayout,
    block_scale_interleave,
    encode_csf_scales,
    expand_csf_scales,
)
from flashinfer.utils import get_compute_capability, round_up


def encode_ref(scales: np.ndarray):
    """The CSF fixed and exception streams of one ``[rows, cols]`` uint8 matrix."""
    rows, cols = scales.shape
    flat = scales.astype(np.int64) + 256 * np.arange(rows)[:, None]
    hist = np.bincount(flat.ravel(), minlength=256 * rows).reshape(rows, 256)
    cum = np.pad(hist.cumsum(1), ((0, 0), (1, 0)))
    base = (cum[:, 16:] - cum[:, :-16]).argmax(1)
    diff = scales.astype(np.int64) - base[:, None]
    outside = (diff < 0) | (diff > 15)
    codes = np.where(outside, 0, diff).astype(np.uint8)
    fixed = np.concatenate(
        (
            base.reshape(-1, 16).astype(np.uint8),
            (codes[:, 0::2] | codes[:, 1::2] << 4).reshape(rows // 16, -1),
        ),
        axis=1,
    )
    pos = np.flatnonzero(outside)
    return fixed, (pos | scales.ravel()[pos].astype(np.int64) << 24).astype(np.uint32)


def decode_ref(fixed: np.ndarray, exceptions: np.ndarray, rows: int, cols: int):
    packed = fixed[:, 16:].reshape(rows, cols // 2)
    out = np.stack((packed & 15, packed >> 4), -1).reshape(rows, cols)
    out = (out + fixed[:, :16].reshape(rows, 1)).ravel()
    out[exceptions & 0xFFFFFF] = exceptions >> 24
    return out.reshape(rows, cols)


def check_encode(scales: np.ndarray):
    num_matrices, rows, cols = scales.shape
    csf = encode_csf_scales(torch.from_numpy(scales).cuda().view(torch.float8_e4m3fn))
    fixed = csf.fixed.cpu().numpy()
    exceptions = csf.exceptions.cpu().numpy()
    offsets = csf.exc_offsets.cpu().numpy()
    slabs = rows // 16
    assert (csf.rows, csf.cols) == (rows, cols)
    assert (
        offsets.shape == (num_matrices * slabs + 1,) and offsets[-1] == exceptions.size
    )
    for i in range(num_matrices):
        ref_fixed, ref_exceptions = encode_ref(scales[i])
        own = exceptions[offsets[i * slabs] : offsets[(i + 1) * slabs]]
        np.testing.assert_array_equal(fixed[i], ref_fixed)
        np.testing.assert_array_equal(own, ref_exceptions)
        np.testing.assert_array_equal(
            np.diff(offsets[i * slabs : (i + 1) * slabs + 1]),
            np.bincount((ref_exceptions & 0xFFFFFF) // (16 * cols), minlength=slabs),
        )
        np.testing.assert_array_equal(decode_ref(fixed[i], own, rows, cols), scales[i])


@pytest.mark.parametrize(
    "num_matrices,rows,cols", [(1, 16, 2), (3, 16, 40), (4, 128, 160), (2, 64, 256)]
)
@pytest.mark.parametrize("spread", [3.0, 12.0, None])
def test_encode_csf_scales_random(num_matrices, rows, cols, spread):
    rng = np.random.default_rng(0)
    shape = (num_matrices, rows, cols)
    if spread is None:  # mostly exceptions
        scales = rng.integers(0, 256, shape)
    else:
        center = rng.integers(0, 256, (num_matrices, rows, 1))
        scales = np.clip(np.rint(rng.normal(center, spread, shape)), 0, 255)
    check_encode(scales.astype(np.uint8))


def test_encode_csf_scales_edges():
    rows = np.zeros((16, 32), np.uint8)
    rows[0] = np.repeat(np.arange(0, 256, 16), 2)  # every window covers two bytes
    rows[1] = 240 + np.arange(32) % 16  # the highest base
    rows[2] = 255
    rows[3, :16], rows[3, 16:] = 10, 100  # tied windows
    rows[4] = np.arange(32) * 8
    check_encode(rows[None])


def test_encode_csf_scales_position_limit():
    rng = np.random.default_rng(1)
    scales = rng.integers(0, 256, (1, 1 << 16, 256), dtype=np.uint8)
    check_encode(scales)
    with pytest.raises(Exception, match="2\\^24"):
        encode_csf_scales(torch.zeros(1 << 16, 258, dtype=torch.uint8, device="cuda"))


def random_scales(rng, shape, spread=6.0):
    center = rng.integers(0, 256, (*shape[:2], 1))
    return torch.from_numpy(
        np.clip(np.rint(rng.normal(center, spread, shape)), 0, 255).astype(np.uint8)
    ).cuda()


def swap_halves(scales: torch.Tensor):
    rows = scales.shape[1]
    return torch.cat((scales[:, rows // 2 :], scales[:, : rows // 2]), 1)


def swizzle_128x4_ref(scales: torch.Tensor):
    num_matrices, rows, cols = scales.shape
    padded = scales.new_zeros(num_matrices, round_up(rows, 128), round_up(cols, 4))
    padded[:, :rows, :cols] = scales
    m, k = padded.shape[1:]
    padded = padded.reshape(num_matrices, m // 128, 4, 32, k // 4, 4)
    return padded.permute(0, 1, 4, 3, 2, 5).reshape(num_matrices, m, k)


@pytest.mark.parametrize(
    "num_matrices,rows,cols",
    [(3, 32, 2), (2, 256, 40), (4, 384, 10), (2, 160, 20), (2, 256, 448)],
)
@pytest.mark.parametrize("swap_row_halves", [False, True])
def test_expand_csf_scales_layouts(num_matrices, rows, cols, swap_row_halves):
    scales = random_scales(
        np.random.default_rng(rows + cols), (num_matrices, rows, cols)
    )
    csf = encode_csf_scales(scales)
    if swap_row_halves:
        scales = swap_halves(scales)
    for ref, layout in (
        (scales, SfLayout.layout_linear),
        (swizzle_128x4_ref(scales), SfLayout.layout_128x4),
    ):
        out = torch.full_like(ref, 0x5A)
        expand_csf_scales(csf, out, layout=layout, swap_row_halves=swap_row_halves)
        assert torch.equal(out, ref), layout
    if get_compute_capability(out.device)[0] >= 9:  # block_scale_interleave needs SM90+
        assert torch.equal(out, block_scale_interleave(scales).view_as(out))


def test_expand_csf_scales_no_rows():
    csf = encode_csf_scales(torch.zeros(1, 0, 40, dtype=torch.uint8, device="cuda"))
    out = torch.empty(1, 0, 40, dtype=torch.uint8, device="cuda")
    for layout in (SfLayout.layout_linear, SfLayout.layout_128x4):
        expand_csf_scales(csf, out, layout=layout)


def test_expand_csf_scales_too_wide():
    csf = encode_csf_scales(torch.zeros(16, 2048, dtype=torch.uint8, device="cuda"))
    out = torch.empty(1, 16, 2048, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="supports up to") as e:
        expand_csf_scales(csf, out)
    check_encode(np.zeros((1, 32, 32), np.uint8))  # no CUDA error left behind
    # A full 128-row tile at the widest width the device accepts expands.
    cols = int(str(e.value).split("up to ")[1].split()[0])
    scales = random_scales(np.random.default_rng(cols), (1, 128, cols))
    out = torch.empty_like(scales)
    expand_csf_scales(encode_csf_scales(scales), out)
    assert torch.equal(out, scales)


def check_routed(out, ref, poison, routed):
    for e in range(ref.shape[0]):
        assert torch.equal(
            out[e], ref[e] if e in routed else torch.full_like(out[e], poison)
        )


@pytest.mark.parametrize("swap_row_halves", [False, True])
@pytest.mark.parametrize("local_expert_offset", [0, 16])
def test_expand_csf_scales_routed(swap_row_halves, local_expert_offset):
    scales = random_scales(np.random.default_rng(2), (16, 256, 40))
    csf = encode_csf_scales(scales)
    ref = swizzle_128x4_ref(swap_halves(scales) if swap_row_halves else scales)
    # Repeated experts and ids outside [0, 16), shifted to global ids.
    ids = torch.tensor([[3, 7], [7, 15], [3, 0], [-1, 16]], dtype=torch.int32).cuda()
    out = torch.full_like(ref, 0xAB)
    expand_csf_scales(
        csf,
        out,
        ids + local_expert_offset,
        layout=SfLayout.layout_128x4,
        swap_row_halves=swap_row_halves,
        local_expert_offset=local_expert_offset,
    )
    check_routed(out, ref, 0xAB, {0, 3, 7, 15})


def test_expand_csf_scales_cuda_graph():
    num_experts, top_k = 32, 4
    scales = random_scales(np.random.default_rng(3), (num_experts, 384, 40))
    csf = encode_csf_scales(scales)
    ref = swizzle_128x4_ref(swap_halves(scales))
    ids = torch.zeros(8, top_k, dtype=torch.int32, device="cuda")
    out = torch.empty_like(ref)

    def expand():
        expand_csf_scales(
            csf, out, ids, layout=SfLayout.layout_128x4, swap_row_halves=True
        )

    expand()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        expand()
    for seed in range(3):
        routing = torch.rand(
            8, num_experts, generator=torch.Generator().manual_seed(seed)
        )
        ids.copy_(routing.topk(top_k).indices)
        out.fill_(0xFF)
        graph.replay()
        check_routed(out, ref, 0xFF, set(ids.flatten().tolist()))
