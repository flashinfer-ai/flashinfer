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

import pytest
import torch
import torch.nn.functional as F

from flashinfer.gemm import w8a16_fp8_gemm, w8a16_fp8_gemm_workspace_bytes
from flashinfer.gemm.w8a16_fp8 import get_w8a16_fp8_gemm_module
from flashinfer.utils import get_compute_capability


@pytest.fixture(autouse=True)
def _needs_hopper_or_blackwell():
    """The op has Hopper (sm_90a) and Blackwell (sm_100a) builds only."""
    if get_compute_capability(torch.device("cuda"))[0] not in (9, 10):
        pytest.skip("w8a16_fp8_gemm needs Hopper or Blackwell")


def _major() -> int:
    return get_compute_capability(torch.device("cuda"))[0]


def _skip_unless_arch(algo: str) -> None:
    """Skip a case whose forced variant belongs to the other architecture."""
    table = _ALGOS_SM100 if _major() == 10 else _ALGOS_SM90
    if algo not in table:
        pytest.skip(f"{algo} is not a variant of this architecture")


def _check_against_reference(out, act, w_fp8, scale, bias, tag):
    """The kernel must be finite and no worse than the cast-copy path (16-bit
    pre-scaled weight) against the fp32 reference, with a small floor."""
    ref32 = act.float() @ (w_fp8.float() * scale).T
    if bias is not None:
        ref32 = ref32 + bias.float()
    cast_err = (
        (_cast_copy_reference(act, w_fp8, scale, bias).float() - ref32)
        .abs()
        .max()
        .item()
    )
    err = (out.float() - ref32).abs().max().item()
    assert torch.isfinite(out.float()).all(), tag
    assert err <= max(2.0 * cast_err, 2e-3 * ref32.abs().max().item()), (
        tag,
        err,
        cast_err,
    )


def _quantize(w32: torch.Tensor):
    scale = (w32.abs().amax() / 448.0).to(torch.float32).reshape(1)
    return (w32 / scale).to(torch.float8_e4m3fn), scale


def _cast_copy_reference(act, w_fp8, scale, bias):
    """TRT-LLM W8A16FP8LinearMethod.apply, verbatim semantics."""
    weight = w_fp8.to(act.dtype).mul_(scale.to(act.dtype))
    return F.linear(act, weight, bias)


@pytest.mark.parametrize("m", [1, 7, 128, 1024])
@pytest.mark.parametrize(
    "n,k", [(1024, 5120), (5120, 1024), (640, 25600), (1280, 1536)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bias", [False, True])
def test_w8a16_fp8_gemm(m, n, k, dtype, bias):
    torch.manual_seed(0)
    dev = torch.device("cuda")
    act = torch.randn(m, k, device=dev, dtype=torch.float32).to(dtype)
    w32 = torch.randn(n, k, device=dev, dtype=torch.float32) * 0.02
    w_fp8, scale = _quantize(w32)
    b = torch.randn(n, device=dev, dtype=torch.float32).to(dtype) if bias else None

    out = w8a16_fp8_gemm(act, w_fp8, scale, bias=b)
    assert out.shape == (m, n)
    assert out.dtype == dtype

    ref32 = act.float() @ (w_fp8.float() * scale).T
    if b is not None:
        ref32 = ref32 + b.float()
    cast_out = _cast_copy_reference(act, w_fp8, scale, b)

    # Both the kernel and the cast-copy path differ from the fp32 reference
    # only by 16-bit rounding / accumulation order; the kernel must not be
    # worse than the cast-copy path by more than a small margin.
    kernel_err = (out.float() - ref32).abs().max().item()
    cast_err = (cast_out.float() - ref32).abs().max().item()
    floor = 2e-3 * ref32.abs().max().item()
    assert torch.isfinite(out.float()).all()
    assert kernel_err <= max(2.0 * cast_err, floor), (kernel_err, cast_err, floor)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "algo",
    [
        None,
        # Blackwell: the smem-A / chunk-partitioned transform, two TMEM-A tiles, the
        # 1SM tile, stream-K and the split-K reduce (K = 256 is two 128-deep tiles).
        "2sm256x256",
        "2sm256x128",
        "1sm128x128",
        "2sm256x128_streamk",
        "2sm256x128_splitk",
        # Hopper: a cooperative tile, a ping-pong tile and stream-K (K = 256 is too
        # short for a Hopper split).
        "128x256",
        "64x256",
        "128x256_streamk",
    ],
)
def test_w8a16_fp8_gemm_every_fp8_value(dtype, algo):
    """Every finite e4m3 byte (denormals included) must dequantize exactly, on every
    structurally different conversion path.

    act = identity over K, so out[m, n] = scale * weight[n, m] with no accumulation
    error beyond the final 16-bit rounding; the weight table enumerates all 254
    finite encodings (the two NaN bytes are excluded)."""
    if algo is not None:
        _skip_unless_arch(algo)
    dev = torch.device("cuda")
    K = 256
    codes = torch.arange(256, dtype=torch.int32, device=dev)
    codes = codes[
        (codes & 0x7F) != 0x7F
    ]  # drop the two e4m3fn NaN encodings (254 left)
    n = 8
    table = torch.zeros(n, K, dtype=torch.uint8, device=dev)
    for row in range(n):
        table[row, : codes.numel()] = codes.roll(row * 31).to(torch.uint8)
    w_fp8 = table.view(torch.float8_e4m3fn)
    scale = torch.tensor([3.0], dtype=torch.float32, device=dev)
    act = torch.eye(K, device=dev, dtype=torch.float32).to(dtype)
    out = w8a16_fp8_gemm(act, w_fp8, scale, algo=algo)
    ref = (w_fp8.float() * scale).T.to(dtype)
    assert torch.equal(out, ref), (algo, (out.float() - ref.float()).abs().max())


def test_w8a16_fp8_gemm_3d_and_out():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    act = torch.randn(2, 33, 256, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(512, 256, device=dev) * 0.02)
    out = torch.empty(2, 33, 512, device=dev, dtype=torch.bfloat16)
    ret = w8a16_fp8_gemm(act, w_fp8, scale, out=out)
    assert ret is out
    ref = act.float().reshape(-1, 256) @ (w_fp8.float() * scale).T
    torch.testing.assert_close(out.float().reshape(-1, 512), ref, atol=5e-2, rtol=2e-2)
    # A non-contiguous out would be silently replaced by a copy: rejected instead.
    with pytest.raises(ValueError, match="contiguous"):
        w8a16_fp8_gemm(
            act,
            w_fp8,
            scale,
            out=torch.empty(2, 512, 33, device=dev, dtype=torch.bfloat16).transpose(
                1, 2
            ),
        )
    with pytest.raises(ValueError, match="out must be"):
        w8a16_fp8_gemm(
            act,
            w_fp8,
            scale,
            out=torch.empty(66, 512, device=dev, dtype=torch.bfloat16),
        )


def test_w8a16_fp8_gemm_caller_workspace():
    """A caller-owned workspace (any dtype, oversized) gives the same result as the
    allocated one; an undersized one is rejected before launch.  The split-K variant
    is forced so the call needs a workspace on both architectures."""
    torch.manual_seed(0)
    dev = torch.device("cuda")
    major = get_compute_capability(dev)[0]
    algo = {9: "128x128_splitk", 10: "2sm256x128_splitk"}.get(major)
    if algo is None:
        pytest.skip("w8a16_fp8_gemm needs Hopper or Blackwell")
    m, n, k = 256, 1024, 5120
    act = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
    need = w8a16_fp8_gemm_workspace_bytes(m, n, k, dev, algo)
    assert need > 0
    assert (
        w8a16_fp8_gemm_workspace_bytes(m, n, k, dev, algo) == need
    )  # pure in its inputs
    # Same answer from the fp16 module: the size does not depend on the activation dtype.
    assert (
        w8a16_fp8_gemm_workspace_bytes(m, n, k, dev, algo, act_dtype=torch.float16)
        == need
    )
    ref = w8a16_fp8_gemm(act, w_fp8, scale, algo=algo)
    ws = torch.full(
        (need + 1000,), 0x7F, dtype=torch.uint8, device=dev
    )  # garbage, oversized
    out = w8a16_fp8_gemm(act, w_fp8, scale, algo=algo, workspace=ws)
    assert torch.equal(out, ref)
    ws16 = torch.empty((need // 2 + 8,), dtype=torch.bfloat16, device=dev)
    assert torch.equal(
        w8a16_fp8_gemm(act, w_fp8, scale, algo=algo, workspace=ws16), ref
    )
    with pytest.raises(ValueError, match="workspace too small"):
        w8a16_fp8_gemm(act, w_fp8, scale, algo=algo, workspace=ws[: need - 4])
    # A single-pass shape needs no scratch (0 bytes), so any valid buffer is accepted.
    plain = {9: "128x256", 10: "2sm256x128"}[major]
    assert w8a16_fp8_gemm_workspace_bytes(4096, 4096, 5120, dev, plain) == 0
    tiny = torch.empty(1, dtype=torch.uint8, device=dev)
    big_act = torch.randn(4096, k, device=dev, dtype=torch.bfloat16)
    w2, s2 = _quantize(torch.randn(4096, k, device=dev) * 0.02)
    assert torch.equal(
        w8a16_fp8_gemm(big_act, w2, s2, algo=plain, workspace=tiny),
        w8a16_fp8_gemm(big_act, w2, s2, algo=plain),
    )


_MAX_DIM = 2**31 - 512  # the admitted bound on each of M, N, K (a multiple of 8 and 16)
_MAX_TILES64 = 2**28 - 1  # the admitted bound on ceil(M / 64) x ceil(N / 64)


def test_w8a16_fp8_gemm_workspace_bytes_extreme_dims():
    """The workspace query runs the dispatch cost models and, for stream-K, CUTLASS's
    own scheduler sizing without allocating anything, so it reaches the largest
    dimensions the launcher admits (a single dimension at the bound is a 32 GiB weight
    or output, so it is reachable; the tile-count bound is not).  Every variant's query
    must answer there (the tile-count arithmetic must not overflow), and one past the
    bound is rejected: a dimension over 2**31 - 512 would overflow CUTLASS's 32-bit
    tile-count ceiling division, which adds the tile (up to 256) before subtracting one,
    an output beyond 2**28 - 1 tiles of 64x64 its 32-bit tile product."""
    dev = torch.device("cuda")
    big = _MAX_DIM
    algos = [None] + (_ALGOS_SM100 if _major() == 10 else _ALGOS_SM90)
    # ceil(big / 64) is 2^25 - 8, so 8 tile columns (N = 512) give 2^28 - 64 output
    # tiles of 64x64, still under the cap; K at the bound exercises the k-tile count of
    # the stream-K sizing and the split rules.
    for m, n, k in [
        (big, 448, 1024),
        (big, 512, 1024),
        (448, big, 1024),
        (8, 8, big),
        (big, 8, 25600),
    ]:
        for algo in algos:
            assert w8a16_fp8_gemm_workspace_bytes(m, n, k, dev, algo) >= 0, (
                m,
                n,
                k,
                algo,
            )
    for m, n, k in [(big + 8, 8, 16), (8, big + 8, 16), (8, 8, big + 16)]:
        with pytest.raises(ValueError, match=f"<= {_MAX_DIM}"):
            w8a16_fp8_gemm_workspace_bytes(m, n, k, dev)
    # 9 tile columns (N = 576) give 9 x (2^25 - 8) tiles, over the cap.
    with pytest.raises(ValueError, match="too large"):
        w8a16_fp8_gemm_workspace_bytes(big, 576, 1024, dev)
    with pytest.raises(ValueError, match="too large"):
        w8a16_fp8_gemm_workspace_bytes(2**20, 2**20, 1024, dev)


def test_w8a16_fp8_gemm_workspace_bytes_splitk_overflow():
    """A forced split-K at large dimensions asks for `splits x M x N` fp32 partials.
    Within the admitted tile count that stays far inside the signed 64-bit range the
    query returns in, and it is reported exactly (plus the CUTLASS scheduler
    workspace, if any) even though no one would allocate it; the split policies
    themselves must not overflow on the way there (the Blackwell long-K rule compares
    wave counts scaled by the split count).  Dimensions whose partials would leave the
    range are rejected by the tile-count bound before the query is asked."""
    dev = torch.device("cuda")
    algo = {9: "128x128_splitk", 10: "2sm256x128_splitk"}[_major()]
    big = _MAX_DIM
    # 2 slices x 2^20 x 2^19 floats = 2^42 bytes (4 TiB).  The forced split on this
    # short K is the smallest, 2.
    assert w8a16_fp8_gemm_workspace_bytes(2**20, 2**19, 1024, dev, algo) >= 2**42
    # 2 slices x (2^31 - 256) x 448 floats, about 7.7 TB.
    assert (
        w8a16_fp8_gemm_workspace_bytes(big, 448, 1024, dev, algo) >= 2 * big * 448 * 4
    )
    # Long K with a large grid: the wave-fill split rules run at their extremes.
    assert w8a16_fp8_gemm_workspace_bytes(2**20, 2**19, 2**30, dev, algo) >= 2**42
    # The launcher's own choice at the extreme dimensions never picks a split it cannot
    # size (the Hopper cost model's partials cap is evaluated without overflowing).
    for m, n, k in [(big, 448, 1024), (2**20, 2**19, 2**30), (8, 8, big)]:
        assert w8a16_fp8_gemm_workspace_bytes(m, n, k, dev) >= 0
    for m, n, k in [(big, big, 1024), (2**30, 2**30, 2**30)]:
        with pytest.raises(ValueError, match="too large"):
            w8a16_fp8_gemm_workspace_bytes(m, n, k, dev, algo)


def test_w8a16_fp8_gemm_rejects_aliasing():
    """The two buffers a call writes (out, workspace) must not share bytes with anything
    it reads, or with each other; inputs may alias one another.  Checked before launch,
    including on an empty call."""
    torch.manual_seed(0)
    dev = torch.device("cuda")
    major = get_compute_capability(dev)[0]
    algo = {9: "128x128_splitk", 10: "2sm256x128_splitk"}[major]
    m, n, k = 256, 1024, 5120
    act = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
    ref = w8a16_fp8_gemm(act, w_fp8, scale, algo=algo)
    need = w8a16_fp8_gemm_workspace_bytes(m, n, k, dev, algo)
    assert need > 0

    # out aliasing an input: a square problem lets out = act pass every shape check,
    # as does a window of the same storage that only partly covers act.
    sq = torch.randn(1536, 1024, device=dev, dtype=torch.bfloat16)
    w_sq, s_sq = _quantize(torch.randn(1024, 1024, device=dev) * 0.02)
    with pytest.raises(ValueError, match="out must not overlap act"):
        w8a16_fp8_gemm(sq[:1024], w_sq, s_sq, out=sq[:1024])
    with pytest.raises(ValueError, match="out must not overlap act"):
        w8a16_fp8_gemm(sq[:1024], w_sq, s_sq, out=sq[512:])
    out_buf = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="out must not overlap bias"):
        w8a16_fp8_gemm(act, w_fp8, scale, bias=out_buf[0], out=out_buf)

    # workspace aliasing the scale: the split-K reduction reads the scale after the
    # partials have been written over it.
    ws_f32 = torch.empty(need // 4 + 4, device=dev, dtype=torch.float32)
    ws_f32[0] = scale[0]
    with pytest.raises(ValueError, match="workspace must not overlap weight_scale"):
        w8a16_fp8_gemm(act, w_fp8, ws_f32[:1], algo=algo, workspace=ws_f32)
    # workspace containing out / weight / act / bias.
    buf = torch.empty(
        need + max(m * n * 2, m * k * 2, n * k) + 4096, device=dev, dtype=torch.uint8
    )
    out_in_buf = buf[: m * n * 2].view(torch.bfloat16).view(m, n)
    with pytest.raises(ValueError, match="workspace must not overlap out"):
        w8a16_fp8_gemm(act, w_fp8, scale, algo=algo, out=out_in_buf, workspace=buf)
    with pytest.raises(ValueError, match="workspace must not overlap weight"):
        w8a16_fp8_gemm(
            act,
            buf[: n * k].view(torch.float8_e4m3fn).view(n, k),
            scale,
            algo=algo,
            workspace=buf,
        )
    with pytest.raises(ValueError, match="workspace must not overlap act"):
        w8a16_fp8_gemm(
            buf[: m * k * 2].view(torch.bfloat16).view(m, k),
            w_fp8,
            scale,
            algo=algo,
            workspace=buf,
        )
    with pytest.raises(ValueError, match="workspace must not overlap bias"):
        w8a16_fp8_gemm(
            act,
            w_fp8,
            scale,
            bias=buf[: n * 2].view(torch.bfloat16),
            algo=algo,
            workspace=buf,
        )
    # The empty call checks the same rule for the buffers that do have bytes.
    with pytest.raises(ValueError, match="workspace must not overlap weight_scale"):
        w8a16_fp8_gemm(act[:0], w_fp8, ws_f32[:1], algo=algo, workspace=ws_f32)

    # Inputs may alias one another (bias as a row of act), and a workspace that merely
    # sits next to out in one allocation is fine.
    bias = act[0, :n]  # contiguous row slice, 16-byte aligned
    assert torch.equal(
        w8a16_fp8_gemm(act, w_fp8, scale, bias=bias, algo=algo),
        w8a16_fp8_gemm(act, w_fp8, scale, bias=bias.clone(), algo=algo),
    )
    ws_after = buf[m * n * 2 :]
    assert torch.equal(
        w8a16_fp8_gemm(
            act, w_fp8, scale, algo=algo, out=out_in_buf, workspace=ws_after
        ),
        ref,
    )


def test_w8a16_fp8_gemm_rejects_bad_inputs():
    dev = torch.device("cuda")
    act = torch.randn(4, 64, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(32, 64, device=dev))
    with pytest.raises(ValueError):
        w8a16_fp8_gemm(act.float(), w_fp8, scale)
    with pytest.raises(ValueError):
        w8a16_fp8_gemm(act, w_fp8.to(torch.bfloat16), scale)
    with pytest.raises(ValueError):
        w8a16_fp8_gemm(act[:, :32], w_fp8, scale)
    # No implicit cast of the scale: the caller passes float32 or casts it.
    for bad in (scale.half(), scale.to(torch.bfloat16), scale.double()):
        with pytest.raises(ValueError, match="weight_scale must be float32"):
            w8a16_fp8_gemm(act, w_fp8, bad)
    assert torch.equal(
        w8a16_fp8_gemm(act, w_fp8, scale.half().float()),
        w8a16_fp8_gemm(act, w_fp8, scale.half().float().reshape(())),
    )
    # bias: shape, dtype, device and contiguity are all checked before launch.
    with pytest.raises(ValueError, match="bias must be"):
        w8a16_fp8_gemm(
            act, w_fp8, scale, bias=torch.zeros(31, device=dev, dtype=act.dtype)
        )
    with pytest.raises(ValueError, match="bias must be"):
        w8a16_fp8_gemm(act, w_fp8, scale, bias=torch.zeros(32, device=dev))
    with pytest.raises(ValueError, match="device"):
        w8a16_fp8_gemm(act, w_fp8, scale, bias=torch.zeros(32, dtype=act.dtype))
    with pytest.raises(ValueError, match="contiguous"):
        w8a16_fp8_gemm(
            act, w_fp8, scale, bias=torch.zeros(64, device=dev, dtype=act.dtype)[::2]
        )


def test_w8a16_fp8_gemm_forced_splitk_unsplittable_k():
    """A split-K variant forced on a K it cannot slice is rejected with the rule, by
    the workspace query as well as by the call (both resolve the same plan), instead
    of the query reporting a size and the launch failing."""
    dev = torch.device("cuda")
    major = get_compute_capability(dev)[0]
    algo = {9: "128x128_splitk", 10: "2sm256x128_splitk"}[major]
    m, n, k = (
        256,
        1024,
        960,
    )  # K % 16 == 0, but neither 64 x (2 x 8) nor 128 x 2 divides it
    act = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
    with pytest.raises(RuntimeError, match="cannot slice K=960"):
        w8a16_fp8_gemm_workspace_bytes(m, n, k, dev, algo)
    with pytest.raises(RuntimeError, match="cannot slice K=960"):
        w8a16_fp8_gemm(act, w_fp8, scale, algo=algo)
    # The launcher's own choice never asks for an impossible split.
    assert w8a16_fp8_gemm_workspace_bytes(m, n, k, dev) >= 0
    out = w8a16_fp8_gemm(act, w_fp8, scale)
    ref = act.float() @ (w_fp8.float() * scale).T
    torch.testing.assert_close(out.float(), ref, atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize("n,k", [(1024, 5128), (1028, 5120)])
def test_w8a16_fp8_gemm_rejects_misaligned(n, k):
    """K % 16 and N % 8 are rejected upfront with a clear message (no silent fallback)."""
    dev = torch.device("cuda")
    act = torch.randn(64, k, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev))
    with pytest.raises(ValueError, match="multiple of"):
        w8a16_fp8_gemm(act, w_fp8, scale)


def test_w8a16_fp8_gemm_rejects_misaligned_storage():
    """A contiguous view at an odd storage offset is rejected (a misaligned pointer
    would otherwise fault the CUDA context from the TMA / vector-store paths)."""
    dev = torch.device("cuda")
    major = get_compute_capability(dev)[0]
    algo = {9: "128x128_splitk", 10: "2sm256x128_splitk"}[major]
    m, n, k = 7, 8, 1024
    act = torch.ones(m, k, device=dev, dtype=torch.bfloat16)
    w_fp8 = torch.ones(n, k, device=dev).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.5], device=dev)
    out = torch.full((m * n + 1,), -1.0, device=dev, dtype=torch.bfloat16)[1:].view(
        m, n
    )
    assert out.is_contiguous()
    with pytest.raises(ValueError, match="aligned"):
        w8a16_fp8_gemm(act, w_fp8, scale, out=out, algo=algo)
    need = w8a16_fp8_gemm_workspace_bytes(m, n, k, dev, algo)
    ws = torch.empty(need // 4 + 1, device=dev, dtype=torch.float32)[1:]
    with pytest.raises(ValueError, match="aligned"):
        w8a16_fp8_gemm(act, w_fp8, scale, algo=algo, workspace=ws)
    act_off = torch.ones(m * k + 1, device=dev, dtype=torch.bfloat16)[1:].view(m, k)
    with pytest.raises(ValueError, match="aligned"):
        w8a16_fp8_gemm(act_off, w_fp8, scale, algo=algo)
    bias_off = torch.ones(n + 1, device=dev, dtype=torch.bfloat16)[1:]
    with pytest.raises(ValueError, match="aligned"):
        w8a16_fp8_gemm(act, w_fp8, scale, bias=bias_off, algo=algo)
    # A view at a 16-byte-aligned offset is fine (and the context is still alive).
    out_ok = torch.full((m * n + 8,), -1.0, device=dev, dtype=torch.bfloat16)[8:].view(
        m, n
    )
    w8a16_fp8_gemm(act, w_fp8, scale, out=out_ok, algo=algo)
    torch.cuda.synchronize()
    assert torch.equal(
        out_ok, torch.full((m, n), 512.0, device=dev, dtype=torch.bfloat16)
    )


def test_w8a16_fp8_gemm_empty_m():
    """M = 0 returns the empty output without launching (the tile schedulers divide
    by the tile count); N = 0 and K = 0 are rejected upfront."""
    dev = torch.device("cuda")
    w_fp8, scale = _quantize(torch.randn(64, 256, device=dev))
    out = w8a16_fp8_gemm(
        torch.empty(0, 256, device=dev, dtype=torch.bfloat16), w_fp8, scale
    )
    assert out.shape == (0, 64) and out.dtype == torch.bfloat16
    out3 = w8a16_fp8_gemm(
        torch.empty(2, 0, 256, device=dev, dtype=torch.float16), w_fp8, scale
    )
    assert out3.shape == (2, 0, 64) and out3.dtype == torch.float16
    given = torch.empty(0, 64, device=dev, dtype=torch.bfloat16)
    assert (
        w8a16_fp8_gemm(
            torch.empty(0, 256, device=dev, dtype=torch.bfloat16),
            w_fp8,
            scale,
            out=given,
            algo="128x128" if get_compute_capability(dev)[0] == 9 else "1sm128x128",
        )
        is given
    )
    assert w8a16_fp8_gemm_workspace_bytes(0, 64, 256, dev) == 0
    # The empty call validates like a non-empty one: a CPU out, a CPU or
    # mis-shaped bias and a CPU weight are rejected, not returned untouched.
    empty = torch.empty(0, 256, device=dev, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="device"):
        w8a16_fp8_gemm(
            empty, w_fp8, scale, out=torch.empty(0, 64, dtype=torch.bfloat16)
        )
    with pytest.raises(ValueError, match="device"):
        w8a16_fp8_gemm(empty, w_fp8, scale, bias=torch.zeros(64, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="bias must be"):
        w8a16_fp8_gemm(
            empty, w_fp8, scale, bias=torch.zeros(65, device=dev, dtype=torch.bfloat16)
        )
    with pytest.raises(ValueError, match="device"):
        w8a16_fp8_gemm(empty, w_fp8.cpu(), scale)
    # weight contiguity and weight / bias alignment are checked on the empty call too.
    wide = torch.randn(64, 512, device=dev).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="weight must be contiguous"):
        w8a16_fp8_gemm(empty, wide[:, ::2], scale)
    w_off = (
        torch.randn(64 * 256 + 1, device=dev).to(torch.float8_e4m3fn)[1:].view(64, 256)
    )
    with pytest.raises(ValueError, match="aligned"):
        w8a16_fp8_gemm(empty, w_off, scale)
    b_off = torch.zeros(65, device=dev, dtype=torch.bfloat16)[1:]
    with pytest.raises(ValueError, match="aligned"):
        w8a16_fp8_gemm(empty, w_fp8, scale, bias=b_off)
    act = torch.empty(4, 0, device=dev, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="K > 0"):
        w8a16_fp8_gemm(
            act, torch.empty(64, 0, device=dev).to(torch.float8_e4m3fn), scale
        )
    with pytest.raises(ValueError, match="N > 0"):
        w8a16_fp8_gemm(
            torch.empty(4, 256, device=dev, dtype=torch.bfloat16),
            torch.empty(0, 256, device=dev).to(torch.float8_e4m3fn),
            scale,
        )
    with pytest.raises(ValueError, match="N > 0"):
        w8a16_fp8_gemm_workspace_bytes(4, 0, 256, dev)
    with pytest.raises(ValueError, match="K > 0"):
        w8a16_fp8_gemm_workspace_bytes(4, 64, 0, dev)


def test_w8a16_fp8_gemm_strided_act():
    """No implicit copies: a non-contiguous activation (row slice, transposed
    last dims, permuted 3D) is rejected in Python; the caller's own
    .contiguous() of it works."""
    torch.manual_seed(0)
    dev = torch.device("cuda")
    k, n = 1024, 64
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
    base = torch.randn(14, k, device=dev, dtype=torch.bfloat16)
    strided = [
        base[::2],
        torch.randn(k, 7, device=dev, dtype=torch.bfloat16).T,
        torch.randn(3, 5, k, device=dev, dtype=torch.bfloat16).permute(1, 0, 2),
    ]
    for t in strided:
        assert not t.is_contiguous()
        with pytest.raises(ValueError, match="act must be contiguous"):
            w8a16_fp8_gemm(t, w_fp8, scale)
        out = w8a16_fp8_gemm(t.contiguous(), w_fp8, scale)
        assert out.shape == t.shape[:-1] + (n,)
        ref = t.float().reshape(-1, k) @ (w_fp8.float() * scale).T
        torch.testing.assert_close(
            out.float().reshape(-1, n), ref, atol=5e-2, rtol=2e-2
        )


def test_w8a16_fp8_gemm_device_resolution():
    """``torch.device("cuda")`` without an index means the current device, same as
    omitting it; a CPU device is rejected."""
    dev = torch.device("cuda")
    cur = torch.device("cuda", torch.cuda.current_device())
    assert w8a16_fp8_gemm_workspace_bytes(
        256, 1024, 5120, dev
    ) == w8a16_fp8_gemm_workspace_bytes(256, 1024, 5120, cur)
    assert w8a16_fp8_gemm_workspace_bytes(
        256, 1024, 5120
    ) == w8a16_fp8_gemm_workspace_bytes(256, 1024, 5120, cur)
    with pytest.raises(ValueError, match="CUDA"):
        w8a16_fp8_gemm_workspace_bytes(256, 1024, 5120, torch.device("cpu"))


def test_w8a16_fp8_gemm_rejects_unknown_algo_name():
    """An algo name outside the architecture's table is rejected ("hand" is not one)."""
    dev = torch.device("cuda")
    act = torch.randn(16, 256, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(128, 256, device=dev))
    with pytest.raises(ValueError):
        w8a16_fp8_gemm(act, w_fp8, scale, algo="hand")


_ALGOS_SM100 = [
    "2sm256x128",
    "2sm256x192",
    "2sm256x256",
    "1sm128x128",
    "2sm256x128_splitk",
    "2sm256x128_streamk",
    "2sm256x160",
]
_ALGOS_SM90 = [
    "128x256",
    "128x192",
    "128x160",
    "128x128",
    "64x256",
    "128x256_streamk",
    "128x192_streamk",
    "64x128",
    "128x128_splitk",
    "128x256_splitk",
]


@pytest.mark.parametrize("algo", _ALGOS_SM100 + _ALGOS_SM90)
@pytest.mark.parametrize(
    "m,n,k", [(1024, 1024, 5120), (200, 1024, 5120), (4096, 128, 5120)]
)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_w8a16_fp8_gemm_algo(algo, m, n, k, bias, dtype):
    """Every kernel variant reachable through ``algo`` must agree with the cast-copy
    reference; a variant of another architecture is rejected."""
    dev = torch.device("cuda")
    major = get_compute_capability(dev)[0]
    mine = algo in (_ALGOS_SM100 if major == 10 else _ALGOS_SM90 if major == 9 else [])
    torch.manual_seed(0)
    act = torch.randn(m, k, device=dev, dtype=dtype)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
    b = torch.randn(n, device=dev, dtype=dtype) if bias else None
    if not mine:
        with pytest.raises(ValueError):
            w8a16_fp8_gemm(act, w_fp8, scale, bias=b, algo=algo)
        return
    out = w8a16_fp8_gemm(act, w_fp8, scale, bias=b, algo=algo)
    ref32 = act.float() @ (w_fp8.float() * scale).T
    if b is not None:
        ref32 = ref32 + b.float()
    cast_err = (
        (_cast_copy_reference(act, w_fp8, scale, b).float() - ref32).abs().max().item()
    )
    err = (out.float() - ref32).abs().max().item()
    assert torch.isfinite(out.float()).all()
    assert err <= max(2.0 * cast_err, 2e-3 * ref32.abs().max().item()), (
        algo,
        err,
        cast_err,
    )
    with pytest.raises(ValueError):
        w8a16_fp8_gemm(act, w_fp8, scale, algo="nope")


# ---------------------------------------------------------------------------
# Grouped / swizzled tile traversal
# ---------------------------------------------------------------------------

_GROUPED_CASES = [
    # Hopper: weight >= 40 MB and M >= 16384 turn the custom grouped scheduler on; 65
    # activation tiles of 256 leave a last group of 1 (group 2) or 1 (group 4).
    ("128x256", 16385, 4096, 10240),  # 32 weight tiles: group 2
    ("128x256", 16385, 8192, 5120),  # 64 weight tiles: group 4
    ("128x256_splitk", 16385, 4096, 10240),  # group 2 under the split-K batch index
    (
        "64x256",
        16385,
        4096,
        10240,
    ),  # ping-pong tile, 64 weight tiles (group 4 if grouped)
    # Blackwell: weight >= 64 MiB and M >= 4096 turn the swizzle on; 4097 activation
    # rows leave a residual tile, 16392 weight rows a residual 2SM tile.
    ("2sm256x128", 4097, 16392, 4096),
    ("2sm256x256", 4097, 16392, 4096),  # the smem-A tile under the swizzle
    ("2sm256x128_splitk", 4097, 16392, 4096),
]


@pytest.mark.parametrize("algo,m,n,k", _GROUPED_CASES)
def test_w8a16_fp8_gemm_grouped_traversal(algo, m, n, k):
    """Shapes that actually enable the L2 tile grouping (Hopper grouped scheduler,
    Blackwell swizzle), with a final incomplete group and, for split-K, the batch
    index on top.  The output is pre-filled with NaN so a tile the traversal skips
    fails the finiteness check, and a tile visited twice still has to agree with the
    reference."""
    _skip_unless_arch(algo)
    torch.manual_seed(0)
    dev = torch.device("cuda")
    act = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
    out = torch.full((m, n), float("nan"), device=dev, dtype=torch.bfloat16)
    ret = w8a16_fp8_gemm(act, w_fp8, scale, algo=algo, out=out)
    assert ret is out
    _check_against_reference(out, act, w_fp8, scale, None, (algo, m, n, k))
    del act, w_fp8, out


# ---------------------------------------------------------------------------
# K tails, short prologues, N tails on every single-pass variant
# ---------------------------------------------------------------------------

_SINGLE_PASS_ALGOS = [a for a in _ALGOS_SM100 + _ALGOS_SM90 if "splitk" not in a]


@pytest.mark.parametrize("algo", _SINGLE_PASS_ALGOS)
@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 136, 16),  # one k-tile, partial; N tail of 8 on a 128-wide tile
        (3, 264, 80),  # K not a multiple of either tile depth; N tail of 8 on 256
        (
            77,
            136,
            144,
        ),  # two Hopper k-tiles plus a tail; one Blackwell tile plus a tail
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_w8a16_fp8_gemm_small_tails(algo, m, n, k, dtype):
    """Every single-pass variant on problems shorter than its pipeline: TMA zero-fill
    of the K tail, a prologue of one or two k-tiles, N tails under the epilogue
    predication, odd M, both activation dtypes, with bias."""
    _skip_unless_arch(algo)
    torch.manual_seed(0)
    dev = torch.device("cuda")
    act = torch.randn(m, k, device=dev, dtype=dtype)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
    bias = torch.randn(n, device=dev, dtype=dtype)
    out = w8a16_fp8_gemm(act, w_fp8, scale, bias=bias, algo=algo)
    _check_against_reference(out, act, w_fp8, scale, bias, (algo, m, n, k, dtype))


@pytest.mark.parametrize(
    "algo", [a for a in _ALGOS_SM100 + _ALGOS_SM90 if "splitk" in a]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_w8a16_fp8_gemm_minimal_splitk(algo, dtype):
    """Each split-K variant at the shortest K it can slice (Hopper: 1024 = 2 x 8
    k-tiles of 64; Blackwell: 256 = 2 k-tiles of 128), on an N tail and odd M; the
    next shorter legal K is rejected by both the query and the call."""
    _skip_unless_arch(algo)
    torch.manual_seed(0)
    dev = torch.device("cuda")
    k, k_short = (1024, 512) if _major() == 9 else (256, 128)
    m, n = 77, 136
    act = torch.randn(m, k, device=dev, dtype=dtype)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
    bias = torch.randn(n, device=dev, dtype=dtype)
    assert w8a16_fp8_gemm_workspace_bytes(m, n, k, dev, algo) >= 2 * m * n * 4
    out = w8a16_fp8_gemm(act, w_fp8, scale, bias=bias, algo=algo)
    _check_against_reference(out, act, w_fp8, scale, bias, (algo, dtype))
    with pytest.raises(RuntimeError, match=f"cannot slice K={k_short}"):
        w8a16_fp8_gemm_workspace_bytes(m, n, k_short, dev, algo)
    with pytest.raises(RuntimeError, match=f"cannot slice K={k_short}"):
        w8a16_fp8_gemm(
            act[:, :k_short].contiguous(),
            w_fp8[:, :k_short].contiguous(),
            scale,
            algo=algo,
        )


# ---------------------------------------------------------------------------
# Caller workspace: stream-K and split-K, reuse on one non-default stream
# ---------------------------------------------------------------------------


def _shape_needing_workspace(algo, dev):
    """A problem on which the forced variant reports a nonzero workspace (a stream-K
    variant may decompose a problem data-parallel and need none)."""
    for m, n, k in [
        (1024, 512, 5120),
        (300, 1024, 25600),
        (2048, 1024, 5120),
        (1024, 1024, 8192),
        (256, 1024, 5120),
    ]:
        if w8a16_fp8_gemm_workspace_bytes(m, n, k, dev, algo) > 0:
            return m, n, k
    raise AssertionError(f"no candidate shape needs a workspace under {algo}")


@pytest.mark.parametrize(
    "algo",
    ["2sm256x128_splitk", "2sm256x128_streamk", "128x128_splitk", "128x256_streamk"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_w8a16_fp8_gemm_caller_workspace_reuse(algo, dtype):
    """One poisoned caller buffer reused by several calls queued on a non-default
    stream, with changed activations, scales and biases, distinct outputs and one
    smaller problem; checked after a single synchronization.  Canary bytes around the
    supplied workspace and output views must survive (scratch and vector-store
    bounds).  Nothing in the buffer may be assumed between calls: per-launch barrier
    initialization and partial overwrite are what this checks."""
    _skip_unless_arch(algo)
    torch.manual_seed(0)
    dev = torch.device("cuda")
    m, n, k = _shape_needing_workspace(algo, dev)
    problems = [(m, 1.0, True), (m, 0.5, False), (m // 2, 2.0, True), (m, 1.0, False)]
    # One buffer sized for the largest need among the problems it serves (a smaller
    # problem may resolve to a different split count).
    need = max(
        w8a16_fp8_gemm_workspace_bytes(mi, n, k, dev, algo, act_dtype=dtype)
        for mi, _, _ in problems
    )
    assert need > 0
    pad = 256  # bytes; keeps the views 16-byte aligned
    buf = torch.full((pad + need + pad,), 0x7F, dtype=torch.uint8, device=dev)
    ws = buf[pad : pad + need]
    pad_el = 64  # elements of 2 bytes
    outs = []
    cases = []
    base_act = torch.randn(m, k, device=dev, dtype=dtype)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i, (mi, s_mult, with_bias) in enumerate(problems):
            act = (base_act[:mi] * (i + 1)).contiguous()
            sc = scale * s_mult
            bias = torch.randn(n, device=dev, dtype=dtype) if with_bias else None
            obuf = torch.full(
                (pad_el + mi * n + pad_el,), float("nan"), device=dev, dtype=dtype
            )
            out = obuf[pad_el : pad_el + mi * n].view(mi, n)
            ret = w8a16_fp8_gemm(
                act, w_fp8, sc, bias=bias, algo=algo, out=out, workspace=ws
            )
            assert ret is out
            outs.append(obuf)
            cases.append((mi, act, sc, bias))
    torch.cuda.synchronize()
    for obuf, (mi, act, sc, bias) in zip(outs, cases, strict=True):
        out = obuf[pad_el : pad_el + mi * n].view(mi, n)
        assert torch.isnan(obuf[:pad_el]).all() and torch.isnan(obuf[-pad_el:]).all()
        _check_against_reference(out, act, w_fp8, sc, bias, (algo, dtype, mi))
    assert (buf[:pad] == 0x7F).all() and (buf[-pad:] == 0x7F).all()


# ---------------------------------------------------------------------------
# The native boundary, called directly
# ---------------------------------------------------------------------------


def test_w8a16_fp8_gemm_native_checks():
    """The module's own FFI entry points reject what the contract says they reject,
    independently of the Python guards: the other activation dtype, misaligned out
    and workspace, out / workspace overlapping an input, a missing or short workspace
    for a variant that needs one (also when auto picked it: on Blackwell the launcher's
    own choice at this small shape needs scratch, which the Python wrapper allocates
    silently), dimensions past the bounds, and algo codes outside the table, including
    values whose low 32 bits are a valid code (an int64 narrowed to int would accept
    them).  The context is alive afterwards: a valid direct call still computes."""
    dev = torch.device("cuda")
    major = _major()
    mod = get_w8a16_fp8_gemm_module(dev, torch.bfloat16)
    m, n, k = 16, 128, 1024
    act = torch.ones(m, k, device=dev, dtype=torch.bfloat16)
    w_fp8 = torch.ones(n, k, device=dev).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.5], device=dev)
    out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
    launch = mod.w8a16_fp8_gemm
    query = mod.w8a16_fp8_gemm_workspace_bytes
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    # What the launcher's own choice needs here (0 on Hopper, scratch on Blackwell).
    ws_auto_bytes = query(m, n, k, idx, 0)
    ws_auto = (
        torch.empty(ws_auto_bytes, device=dev, dtype=torch.uint8)
        if ws_auto_bytes
        else None
    )
    # A forced variant that needs scratch at a larger problem, for the capacity check.
    first_valid = 2 if major == 9 else 1
    m2, n2, k2 = 1024, 1024, 8192
    code_ws, need = next(
        (code, query(m2, n2, k2, idx, code))
        for code in range(first_valid, 8)
        if query(m2, n2, k2, idx, code) > 0
    )
    act2 = torch.ones(m2, k2, device=dev, dtype=torch.bfloat16)
    w2 = torch.ones(n2, k2, device=dev).to(torch.float8_e4m3fn)
    out2 = torch.empty(m2, n2, device=dev, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="workspace too small: 0 bytes"):
        launch(out2, act2, w2, scale, None, None, code_ws)
    with pytest.raises(RuntimeError, match=f"workspace too small: {need - 16} bytes"):
        launch(
            out2,
            act2,
            w2,
            scale,
            None,
            torch.empty(need - 16, device=dev, dtype=torch.uint8),
            code_ws,
        )
    if ws_auto is None:
        # Hopper's choice here needs no scratch, so an unneeded buffer is simply unused;
        # on Blackwell the same call without one must say so.
        launch(
            out,
            act,
            w_fp8,
            scale,
            None,
            torch.empty(64, device=dev, dtype=torch.uint8),
            0,
        )
    else:
        with pytest.raises(RuntimeError, match="workspace too small: 0 bytes"):
            launch(out, act, w_fp8, scale, None, None, 0)

    # The bf16 module refuses fp16 activations (one module per dtype).
    with pytest.raises(RuntimeError):
        launch(out.half(), act.half(), w_fp8, scale, None, None, 0)
    # Misaligned out / workspace (contiguous views at a one-element offset).
    out_off = torch.empty(m * n + 1, device=dev, dtype=torch.bfloat16)[1:].view(m, n)
    with pytest.raises(RuntimeError, match="aligned"):
        launch(out_off, act, w_fp8, scale, None, None, 0)
    ws_off = torch.empty(1 << 20, device=dev, dtype=torch.uint8)[1:]
    with pytest.raises(RuntimeError, match="aligned"):
        launch(out, act, w_fp8, scale, None, ws_off, 0)
    # Overlap: out on act's storage (a square problem), workspace on the weight.
    sq = torch.ones(k, k, device=dev, dtype=torch.bfloat16)
    w_sq = torch.ones(k, k, device=dev).to(torch.float8_e4m3fn)
    with pytest.raises(RuntimeError, match="out must not overlap act"):
        launch(sq, sq, w_sq, scale, None, None, 0)
    ws_on_w = w_fp8.view(torch.uint8).view(-1)
    with pytest.raises(RuntimeError, match="workspace must not overlap weight"):
        launch(out, act, w_fp8, scale, None, ws_on_w, 0)
    # Dimension bounds at the native boundary.
    with pytest.raises(RuntimeError, match="at most"):
        query(_MAX_DIM + 8, 8, 16, idx, 0)
    with pytest.raises(RuntimeError, match="too large"):
        query(_MAX_DIM, _MAX_DIM, 1024, idx, 0)
    # Algo codes: reserved, past the table, negative, and 2**32 + valid (low 32 bits
    # are 0 = auto or a real variant).
    reserved = 1 if major == 9 else 8
    for code in (reserved, 99, -1, 2**32, 2**32 + first_valid, 2**40 + first_valid):
        with pytest.raises(RuntimeError, match="algo"):
            query(m, n, k, idx, code)
        with pytest.raises(RuntimeError, match="algo"):
            launch(out, act, w_fp8, scale, None, ws_auto, code)
    # Still alive: a direct valid call computes 0.5 * K.
    launch(out, act, w_fp8, scale, None, ws_auto, 0)
    torch.cuda.synchronize()
    assert torch.equal(out, torch.full((m, n), 512.0, device=dev, dtype=torch.bfloat16))
