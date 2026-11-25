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

"""SM90 NVFP4-weight x BF16-activation fused MoE (``sm90_nvfp4_w4a16_moe``).

Every GPU test compares against an FP32 PyTorch reference built from the
*original* checkpoint tensors (E2M1 codes x E4M3 block scales x global scale,
dequantized to FP32, FP32 matmuls, no intermediate rounding), with the
acceptance band of FlashInfer's W4A16 oracle tests: ``atol = 2e-3 * max|ref| +
2 ulp_bf16(max|ref|)``, ``rtol = 0.05``. Weights are sampled directly as E2M1
codes with E4M3 scale bytes in ``0x28..0x3F`` and random per-expert global
scales (gate and up differ). Shapes cover the tuned one (hidden =
intermediate = 4096, 32 local experts, top-8) and several other MoE layouts.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
from typing import Optional

import pytest
import torch

from flashinfer.fused_moe import (
    Sm90Nvfp4W4A16MoEWorkspace,
    prepare_sm90_nvfp4_w4a16_weights,
    sm90_nvfp4_w4a16_moe,
    sm90_nvfp4_w4a16_moe_prefill_min_tokens,
)
from flashinfer.utils import is_sm90a_supported

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm90a_supported(torch.device("cuda")),
    reason="sm90_nvfp4_w4a16_moe needs an SM90 (sm_90a) GPU",
)

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
RTOL = 0.05
ATOL_FRAC = 2e-3

# (hidden, intermediate, local experts, top_k)
SHAPES = [
    (4096, 4096, 32, 8),  # tuned shape (Muse Spark EP8 slice)
    (2048, 768, 32, 8),  # Qwen3-30B-A3B-like EP4 slice; odd K chunk count
    (7168, 2048, 8, 8),  # DeepSeek-V3-like hidden size (8 local experts)
    (4096, 1536, 16, 8),  # Qwen3-235B-like EP8 slice
    (3072, 1024, 8, 2),  # small top-2
    (1024, 512, 4, 1),  # smallest supported
]
DECODE_TOKENS = [1, 5, 16, 32, 64, 100, 256, 700, 1024]


def bf16_ulp(x: float) -> float:
    return 0.0 if x <= 0.0 else 2.0 ** (math.floor(math.log2(x)) - 7)


def assert_in_band(y: torch.Tensor, ref: torch.Tensor) -> None:
    amax = float(ref.abs().max()) if ref.numel() else 0.0
    atol = ATOL_FRAC * amax + 2.0 * bf16_ulp(amax)
    torch.testing.assert_close(y.float(), ref, atol=atol, rtol=RTOL)


class Checkpoint:
    """Random NVFP4 expert weights in the ModelOpt layout plus their FP32 values."""

    def __init__(self, hidden, inter, experts, seed=0, device="cuda"):
        g = torch.Generator(device="cpu").manual_seed(seed)
        H, I, E = hidden, inter, experts
        self.H, self.I, self.E = H, I, E

        def codes(rows, cols):
            return torch.randint(
                0, 256, (E, rows, cols // 2), generator=g, dtype=torch.uint8
            )

        def scales(rows, cols):
            return torch.randint(
                0x28, 0x40, (E, rows, cols // 16), generator=g, dtype=torch.uint8
            )

        self.w13_fp4, self.w2_fp4 = codes(2 * I, H), codes(H, I)
        self.w13_sf, self.w2_sf = scales(2 * I, H), scales(H, I)
        # keep |gate|, |up|, |out| around 1 for unit-variance activations
        self.w13_gs = (torch.rand((E, 2), generator=g) + 0.5) / (2.0 * math.sqrt(H))
        self.w2_gs = (torch.rand((E,), generator=g) + 0.5) / (2.0 * math.sqrt(I))
        to = dict(device=device)
        self.w13_fp4, self.w2_fp4 = self.w13_fp4.to(**to), self.w2_fp4.to(**to)
        self.w13_sf = self.w13_sf.to(**to).view(torch.float8_e4m3fn)
        self.w2_sf = self.w2_sf.to(**to).view(torch.float8_e4m3fn)
        self.w13_gs, self.w2_gs = self.w13_gs.to(**to), self.w2_gs.to(**to)

    def packed(self):
        return prepare_sm90_nvfp4_w4a16_weights(
            self.w13_fp4, self.w13_sf, self.w13_gs, self.w2_fp4, self.w2_sf, self.w2_gs
        )

    @staticmethod
    def _deq(fp4, sf):
        c = torch.stack((fp4 & 15, fp4 >> 4), -1).reshape(fp4.shape[0], -1).long()
        v = _E2M1.to(fp4.device)[c & 7]
        v = torch.where((c & 8).bool(), -v, v)
        return v * sf.float().repeat_interleave(16, dim=1)

    def expert(self, e):
        """FP32 (w13 [2I, H], w2 [H, I]) of expert e."""
        w13 = self._deq(self.w13_fp4[e], self.w13_sf[e])
        w13[: self.I] *= self.w13_gs[e, 0]
        w13[self.I :] *= self.w13_gs[e, 1]
        w2 = self._deq(self.w2_fp4[e], self.w2_sf[e]) * self.w2_gs[e]
        return w13, w2


def reference_moe(
    ckpt: Checkpoint,
    x: torch.Tensor,
    ids: torch.Tensor,
    w: torch.Tensor,
    post_norm_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    xf = x.float()
    out = torch.zeros(x.shape[0], ckpt.H, dtype=torch.float32, device=x.device)
    for e in range(ckpt.E):
        tok, slot = torch.nonzero(ids == e, as_tuple=True)
        if tok.numel() == 0:
            continue
        w13, w2 = ckpt.expert(e)
        gu = xf[tok] @ w13.T
        h = torch.nn.functional.silu(gu[:, : ckpt.I]) * gu[:, ckpt.I :]
        y = h @ w2.T
        if post_norm_weight is not None:
            v = y * post_norm_weight.float()
            y = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
        out.index_add_(0, tok, y * w[tok, slot].float()[:, None])
    return out


def routing(M, E, top_k, device, seed=0, invalid=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = min(top_k, E)
    ids = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(M)]).to(
        torch.int32
    )
    if k < top_k:
        ids = torch.cat([ids, torch.full((M, top_k - k), -1, dtype=torch.int32)], 1)
    if invalid:
        # expert parallelism: some routes go to experts this call does not hold
        drop = torch.rand(ids.shape, generator=g) < 0.25
        far = torch.where(torch.rand(ids.shape, generator=g) < 0.5, -1, E + 3)
        ids = torch.where(drop, far.to(torch.int32), ids)
    w = torch.rand((M, top_k), generator=g) + 0.25
    w = w / w.sum(-1, keepdim=True)
    return ids.to(device), w.to(device)


_CKPTS = {}


def checkpoint(shape):
    if shape not in _CKPTS:
        H, I, E, _ = shape
        ckpt = Checkpoint(H, I, E, seed=hash(shape) & 0xFFFF)
        _CKPTS[shape] = (ckpt, ckpt.packed())
    return _CKPTS[shape]


def run(packed, x, ids, w, **kw):
    return sm90_nvfp4_w4a16_moe(
        x,
        ids,
        w,
        packed.w13,
        packed.w13_global_scale,
        packed.w2,
        packed.w2_global_scale,
        **kw,
    )


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "h{}_i{}_e{}_k{}".format(*s))
@pytest.mark.parametrize("num_tokens", DECODE_TOKENS)
def test_decode_matches_reference(shape, num_tokens):
    """The small-batch kernel (forced; the automatic choice may pick the other one)."""
    ckpt, packed = checkpoint(shape)
    H, _, E, top_k = shape
    torch.manual_seed(num_tokens)
    x = torch.randn(num_tokens, H, device="cuda").bfloat16()
    ids, w = routing(num_tokens, E, top_k, "cuda", seed=num_tokens)
    y = run(packed, x, ids, w, prefill_min_tokens=1 << 30)
    assert y.shape == (num_tokens, H) and y.dtype == torch.bfloat16
    assert_in_band(y, reference_moe(ckpt, x, ids, w))


@pytest.mark.parametrize(
    "shape", [SHAPES[0], SHAPES[3]], ids=lambda s: "h{}_i{}".format(*s)
)
@pytest.mark.parametrize("num_tokens", [1, 33, 300, 1024, 2500])
def test_post_norm_matches_reference(shape, num_tokens):
    ckpt, packed = checkpoint(shape)
    H, _, E, top_k = shape
    x = torch.randn(num_tokens, H, device="cuda").bfloat16()
    ids, w = routing(num_tokens, E, top_k, "cuda", seed=7)
    nw = torch.rand(H, device="cuda") + 0.5
    y = run(packed, x, ids, w, post_norm_weight=nw, post_norm_eps=1e-6)
    assert_in_band(y, reference_moe(ckpt, x, ids, w, nw, 1e-6))


@pytest.mark.parametrize(
    "shape", [SHAPES[0], SHAPES[1], SHAPES[4]], ids=lambda s: "h{}_i{}".format(*s)
)
@pytest.mark.parametrize("num_tokens", [1, 64, 513])
def test_skipped_routes_and_int64_ids(shape, num_tokens):
    """Routes to experts outside [0, E) contribute nothing; int64 ids and BF16 weights
    work. Skipped int64 ids include values that wrap into [0, E) when narrowed to
    int32 (+-2**32 + e)."""
    ckpt, packed = checkpoint(shape)
    H, _, E, top_k = shape
    x = torch.randn(num_tokens, H, device="cuda").bfloat16()
    ids, w = routing(num_tokens, E, top_k, "cuda", seed=3, invalid=True)
    ids = ids.long()
    skipped = (ids < 0) | (ids >= E)
    pos = torch.arange(ids.numel(), device=ids.device).view(ids.shape)
    wrap = torch.where(ids < 0, -(2**32), 2**32) + pos % E
    ids = torch.where(skipped & (pos % 2 == 0), wrap, ids)
    y = run(packed, x, ids, w.bfloat16())
    assert_in_band(y, reference_moe(ckpt, x, ids, w.bfloat16()))


@pytest.mark.parametrize(
    "shape", [SHAPES[0], SHAPES[2], SHAPES[4]], ids=lambda s: "h{}_i{}".format(*s)
)
@pytest.mark.parametrize("num_tokens,chunk", [(300, 128), (2049, 8192), (5000, 2048)])
def test_prefill_kernel_matches_reference(shape, num_tokens, chunk):
    """The large-batch kernel, also over several chunks (cursor carried between launches)."""
    ckpt, packed = checkpoint(shape)
    H, I, E, top_k = shape
    # a fresh workspace: its prefill buffers are sized by this chunk length
    ws = Sm90Nvfp4W4A16MoEWorkspace(torch.device("cuda"), H, I)
    x = torch.randn(num_tokens, H, device="cuda").bfloat16()
    ids, w = routing(num_tokens, E, top_k, "cuda", seed=11, invalid=True)
    y = run(
        packed,
        x,
        ids,
        w,
        workspace=ws,
        prefill_min_tokens=1,
        prefill_chunk_tokens=chunk,
    )
    assert ws.prefill_chunk_tokens == chunk
    assert_in_band(y, reference_moe(ckpt, x, ids, w))


@pytest.mark.parametrize("num_tokens", [255, 256, 300])
def test_auto_kernel_choice(num_tokens):
    """Without ``prefill_min_tokens`` the call runs the kernel the shape rule picks
    (256 tokens for the tuned shape): bitwise equal to forcing that kernel."""
    ckpt, packed = checkpoint(SHAPES[0])
    H, I, E, top_k = SHAPES[0]
    assert sm90_nvfp4_w4a16_moe_prefill_min_tokens(H, I, E, top_k) == 256
    x = torch.randn(num_tokens, H, device="cuda").bfloat16()
    ids, w = routing(num_tokens, E, top_k, "cuda", seed=17)
    y = run(packed, x, ids, w)
    forced = 1 if num_tokens >= 256 else 1 << 30
    assert torch.equal(y, run(packed, x, ids, w, prefill_min_tokens=forced))
    assert_in_band(y, reference_moe(ckpt, x, ids, w))


def test_cuda_graph_auto_choice_without_prefill_buffers():
    """Captured with a workspace whose prefill buffers are not reserved, a batch
    above the automatic threshold runs the small-batch kernel."""
    ckpt, packed = checkpoint(SHAPES[0])
    H, I, E, top_k = SHAPES[0]
    num_tokens = 300
    ws = Sm90Nvfp4W4A16MoEWorkspace(torch.device("cuda"), H, I)
    x = torch.randn(num_tokens, H, device="cuda").bfloat16()
    ids, w = routing(num_tokens, E, top_k, "cuda", seed=19)
    out = torch.empty(num_tokens, H, device="cuda", dtype=torch.bfloat16)
    run(packed, x, ids, w, out=out, workspace=ws, prefill_min_tokens=1 << 30)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(packed, x, ids, w, out=out, workspace=ws)
    assert ws.prefill_chunk_tokens == 0
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    ref = run(packed, x, ids, w, workspace=ws, prefill_min_tokens=1 << 30)
    assert torch.equal(out, ref)


def test_long_decode_is_chunked():
    """More than 1024 tokens below the prefill threshold run as several small-batch launches."""
    ckpt, packed = checkpoint(SHAPES[1])
    H, _, E, top_k = SHAPES[1]
    x = torch.randn(1500, H, device="cuda").bfloat16()
    ids, w = routing(1500, E, top_k, "cuda", seed=5)
    y = run(packed, x, ids, w, prefill_min_tokens=1 << 30)
    assert_in_band(y, reference_moe(ckpt, x, ids, w))


@pytest.mark.parametrize("num_tokens", [8, 200, 3000])
def test_deterministic(num_tokens):
    """Fixed reduction orders: repeated calls give bit-identical outputs."""
    _, packed = checkpoint(SHAPES[0])
    H, _, E, top_k = SHAPES[0]
    x = torch.randn(num_tokens, H, device="cuda").bfloat16()
    ids, w = routing(num_tokens, E, top_k, "cuda", seed=13)
    y0 = run(packed, x, ids, w)
    for _ in range(3):
        assert torch.equal(run(packed, x, ids, w), y0)


@pytest.mark.parametrize("num_tokens", [4, 96])
def test_cuda_graph_replay(num_tokens):
    """Capture once, replay with new activations and routing in the captured buffers."""
    ckpt, packed = checkpoint(SHAPES[0])
    H, _, E, top_k = SHAPES[0]
    ws = Sm90Nvfp4W4A16MoEWorkspace(torch.device("cuda"), H, SHAPES[0][1])
    x = torch.randn(num_tokens, H, device="cuda").bfloat16()
    ids, w = routing(num_tokens, E, top_k, "cuda", seed=1)
    out = torch.empty(num_tokens, H, device="cuda", dtype=torch.bfloat16)
    run(packed, x, ids, w, out=out, workspace=ws)  # warm-up: build the module
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(packed, x, ids, w, out=out, workspace=ws)
    for seed in (2, 3):
        x.copy_(torch.randn(num_tokens, H, device="cuda").bfloat16())
        new_ids, new_w = routing(num_tokens, E, top_k, "cuda", seed=seed)
        ids.copy_(new_ids)
        w.copy_(new_w)
        graph.replay()
        torch.cuda.synchronize()
        assert_in_band(out, reference_moe(ckpt, x, ids, w))


def test_prefill_graph_survives_reservation_growth():
    """A graph that captured the large-batch kernel keeps working after a later
    call grows the workspace's prefill reservation (the captured buffers stay
    alive; their memory is not handed to other tensors)."""
    _, packed = checkpoint(SHAPES[4])
    H, I, E, top_k = SHAPES[4]
    num_tokens = 600
    ws = Sm90Nvfp4W4A16MoEWorkspace(torch.device("cuda"), H, I)
    ws.reserve_prefill(128)
    x = torch.randn(num_tokens, H, device="cuda").bfloat16()
    ids, w = routing(num_tokens, E, top_k, "cuda", seed=23)
    out = torch.empty(num_tokens, H, device="cuda", dtype=torch.bfloat16)
    kw = dict(workspace=ws, prefill_min_tokens=1, prefill_chunk_tokens=128)
    run(packed, x, ids, w, out=out, **kw)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(packed, x, ids, w, out=out, **kw)
    captured = [t.data_ptr() for t in (ws.xt, ws.h, ws.maps, ws.prefill_counters)]
    ref_ws = Sm90Nvfp4W4A16MoEWorkspace(torch.device("cuda"), H, I)
    ref = run(packed, x, ids, w, **dict(kw, workspace=ref_ws))
    # an eager call with a larger chunk grows the reservation
    run(packed, x, ids, w, **dict(kw, prefill_chunk_tokens=512))
    assert ws.prefill_chunk_tokens == 512
    assert ws.xt.data_ptr() not in captured
    # take whatever the allocator has free and fill it with garbage
    junk = [
        torch.full((n,), -1, dtype=torch.int32, device="cuda")
        for n in [1 << 20] * 64 + [1 << 12] * 256
    ]
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    del junk


@pytest.mark.parametrize(
    "shape", [SHAPES[0], SHAPES[4]], ids=lambda s: "h{}_i{}".format(*s)
)
def test_cuda_graph_replay_prefill_kernel(shape):
    """The large-batch kernel captured with a reserved workspace, over several
    chunks (cursor carried between the launches, PDL between them), replayed
    with new activations and routing."""
    ckpt, packed = checkpoint(shape)
    H, I, E, top_k = shape
    num_tokens, chunk = 700, 256  # three launches per call
    ws = Sm90Nvfp4W4A16MoEWorkspace(torch.device("cuda"), H, I)
    ws.reserve_prefill(chunk)
    kw = dict(workspace=ws, prefill_min_tokens=1, prefill_chunk_tokens=chunk)
    x = torch.randn(num_tokens, H, device="cuda").bfloat16()
    ids, w = routing(num_tokens, E, top_k, "cuda", seed=29, invalid=True)
    out = torch.empty(num_tokens, H, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(packed, x, ids, w, out=out, **kw)
    assert ws.prefill_chunk_tokens == chunk
    for seed in (31, 37, 37):
        x.copy_(torch.randn(num_tokens, H, device="cuda").bfloat16())
        new_ids, new_w = routing(num_tokens, E, top_k, "cuda", seed=seed, invalid=True)
        ids.copy_(new_ids)
        w.copy_(new_w)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, run(packed, x, ids, w, **kw))
        assert_in_band(out, reference_moe(ckpt, x, ids, w))


_PARTITION_SCRIPT = textwrap.dedent(
    """
    import sys
    import torch
    from flashinfer.green_ctx import split_device_green_ctx
    sys.path.insert(0, sys.argv[1])
    import test_sm90_nvfp4_w4a16_moe as t

    dev = torch.device("cuda", torch.cuda.current_device())
    streams, resources = split_device_green_ctx(dev, 1, 16)
    H, I, E, top_k = t.SHAPES[4]
    ckpt, packed = t.checkpoint(t.SHAPES[4])
    ws = t.Sm90Nvfp4W4A16MoEWorkspace(dev, H, I)
    ws.reserve_prefill(256)
    x = torch.randn(700, H, device=dev).bfloat16()
    ids, w = t.routing(700, E, top_k, "cuda", seed=41)
    torch.cuda.synchronize()
    with torch.cuda.stream(streams[0]):
        y = t.run(packed, x, ids, w, workspace=ws, prefill_min_tokens=1,
                  prefill_chunk_tokens=256)
        yd = t.run(packed, x, ids, w, workspace=ws, prefill_min_tokens=1 << 30)
    streams[0].synchronize()
    ref = t.reference_moe(ckpt, x, ids, w)
    t.assert_in_band(y, ref)
    t.assert_in_band(yd, ref)
    print("partition SMs", resources[0].sm.smCount)
    """
)


def test_kernels_complete_on_sm_partition():
    """Both kernels launch one CTA per SM of the device. On a green-context SM
    partition only part of that grid is resident at a time; the kernels must
    still complete (no CTA may wait on work owned by a CTA that has not
    started). Runs in a subprocess so a hang fails the test instead of the run."""
    pytest.importorskip("cuda.bindings")
    if torch.cuda.get_device_properties(0).multi_processor_count < 64:
        pytest.skip("needs a device much larger than the 16-SM partition")
    H, I, _, _ = SHAPES[4]
    checkpoint(SHAPES[4])
    Sm90Nvfp4W4A16MoEWorkspace(torch.device("cuda"), H, I)  # build the module here
    torch.cuda.synchronize()
    try:
        res = subprocess.run(
            [sys.executable, "-c", _PARTITION_SCRIPT, os.path.dirname(__file__)],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("sm90_nvfp4_w4a16_moe did not complete on a 16-SM partition")
    if res.returncode != 0 and "Failed to split device" in res.stderr:
        pytest.skip("green contexts are not available")
    assert res.returncode == 0, res.stderr[-4000:]


def test_zero_tokens():
    _, packed = checkpoint(SHAPES[5])
    H = SHAPES[5][0]
    x = torch.empty(0, H, device="cuda", dtype=torch.bfloat16)
    ids = torch.empty(0, 1, device="cuda", dtype=torch.int32)
    w = torch.empty(0, 1, device="cuda")
    assert run(packed, x, ids, w).shape == (0, H)


def test_rejects_bad_inputs():
    _, packed = checkpoint(SHAPES[5])
    H, _, E, _ = SHAPES[5]
    x = torch.randn(4, H, device="cuda").bfloat16()
    ids, w = routing(4, E, 1, "cuda")
    with pytest.raises(ValueError):
        run(packed, x.float(), ids, w)
    with pytest.raises(ValueError):
        run(packed, x[:, : H // 2].contiguous(), ids, w)
    with pytest.raises(ValueError):
        run(
            packed,
            x,
            torch.zeros(4, 9, dtype=torch.int32, device="cuda"),
            torch.zeros(4, 9, device="cuda"),
        )
    with pytest.raises(ValueError):
        run(packed, x, ids, w, post_norm_weight=torch.ones(H + 1, device="cuda"))


def test_prefill_chunk_tokens_limit():
    """A large-batch launch has up to chunk_tokens / 32 + 32 tiles of 256 rows,
    and its per-tile counters hold 1024: past MAX_PREFILL_CHUNK_TOKENS the done
    word of one tile would be the ready word of another. The workspace, the
    wrapper and the launcher reject such chunks."""
    from flashinfer.fused_moe.sm90_nvfp4_w4a16 import (
        MAX_PREFILL_CHUNK_TOKENS,
        _get_module,
        _routing_inputs,
    )

    _, packed = checkpoint(SHAPES[5])
    H, I, E, _ = SHAPES[5]
    big = MAX_PREFILL_CHUNK_TOKENS + 128
    ws = Sm90Nvfp4W4A16MoEWorkspace(torch.device("cuda"), H, I)
    with pytest.raises(ValueError, match="chunk_tokens"):
        ws.reserve_prefill(big)
    assert ws.prefill_chunk_tokens == 0
    x = torch.randn(4, H, device="cuda").bfloat16()
    ids, w = routing(4, E, 1, "cuda")
    with pytest.raises(ValueError, match="prefill_chunk_tokens"):
        run(
            packed,
            x,
            ids,
            w,
            workspace=ws,
            prefill_min_tokens=1,
            prefill_chunk_tokens=big,
        )
    # the launcher has the same bound: the largest chunk passes the chunk check and
    # then fails on the (deliberately small) `xt`
    module = _get_module(H, I, False)
    ids8, w8 = _routing_inputs(ids, w)
    i32 = dict(dtype=torch.int32, device="cuda")

    def launch(chunk):
        module.prefill_run(
            x, ids8, w8, packed.w13, packed.w13_global_scale[0],
            packed.w13_global_scale[1], packed.w2, packed.w2_global_scale,
            torch.empty(4, device="cuda"), torch.empty_like(x),
            torch.empty(8 * 4 * H, dtype=torch.bfloat16, device="cuda"),
            torch.empty(128, H, dtype=torch.bfloat16, device="cuda"),
            torch.empty(module.prefill_map_bytes(), dtype=torch.uint8, device="cuda"),
            torch.empty(1 << 12, **i32), torch.empty(64, **i32),
            torch.zeros(module.prefill_counter_words(), **i32), E, chunk, 1e-6, False,
        )  # fmt: skip

    with pytest.raises(Exception, match="chunk_tokens must be"):
        launch(big)
    with pytest.raises(Exception, match="xt must be"):
        launch(MAX_PREFILL_CHUNK_TOKENS)


def test_launcher_checks_expert_params():
    """Direct launcher calls reject global scales / norm weights of the wrong dtype
    or size and packed weights with fewer experts than ``num_experts``."""
    from flashinfer.fused_moe.sm90_nvfp4_w4a16 import _decode_buffer_bytes, _get_module

    _, packed = checkpoint(SHAPES[5])
    H, I, E, _ = SHAPES[5]
    module = _get_module(H, I, False)
    ws = Sm90Nvfp4W4A16MoEWorkspace(torch.device("cuda"), H, I)
    x = torch.randn(4, H, device="cuda").bfloat16()
    ids = torch.zeros(4, 8, dtype=torch.int32, device="cuda")
    w = torch.full((4, 8), 0.125, device="cuda")
    out = torch.empty_like(x)
    buf = torch.empty(
        _decode_buffer_bytes(H, I, False, 4), dtype=torch.uint8, device="cuda"
    )
    gs13, gs2 = packed.w13_global_scale, packed.w2_global_scale

    def call(gate=gs13[0], up=gs13[1], down=gs2, nw=None, n=E):
        nw = torch.empty(4, device="cuda") if nw is None else nw
        module.decode_run(
            x, ids, w, packed.w13, gate, up, packed.w2, down, nw, out, buf,
            ws.decode_counters, ws.decode_scratch, n, 1e-6, False,
        )  # fmt: skip

    call()
    torch.cuda.synchronize()
    with pytest.raises(Exception, match="w2_scale"):
        call(down=gs2.half())
    with pytest.raises(Exception, match="norm_weight"):
        call(nw=torch.empty(4, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(Exception, match="global scales"):
        call(down=gs2[: E - 1].contiguous())
    one = torch.ones(E + 1, device="cuda")
    with pytest.raises(Exception, match="w13 must hold"):
        call(gate=one, up=one, down=one, n=E + 1)
