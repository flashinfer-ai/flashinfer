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

from flashinfer.gemm import (
    mm_bf16_fp8,
    mm_bf16_fp8_workspace_bytes,
    mm_fp16_fp8,
    mm_fp16_fp8_workspace_bytes,
)
from flashinfer.utils import get_compute_capability

_MM = {torch.bfloat16: mm_bf16_fp8, torch.float16: mm_fp16_fp8}
_WS = {
    torch.bfloat16: mm_bf16_fp8_workspace_bytes,
    torch.float16: mm_fp16_fp8_workspace_bytes,
}


def _mm(dtype):
    """The public function for an activation dtype (same kernels behind both)."""
    return _MM[dtype]


def _ws(dtype):
    return _WS[dtype]


@pytest.fixture(autouse=True)
def _needs_hopper_or_blackwell():
    """The op has Hopper (sm_90a) and Blackwell (sm_100a / sm_103a) kernels only.

    Ask the op itself rather than testing the major version: Rubin (compute
    capability 10.7) shares Blackwell's major and is not supported."""
    major, minor = get_compute_capability(torch.device("cuda"))
    if not mm_bf16_fp8.is_compute_capability_supported(major * 10 + minor):
        pytest.skip("mm_bf16_fp8 needs Hopper or Blackwell (sm_90a, sm_100a, sm_103a)")


def _major() -> int:
    return get_compute_capability(torch.device("cuda"))[0]


def _skip_unless_arch(tactic: str) -> None:
    """Skip a case whose forced variant belongs to the other architecture."""
    table = _TACTICS_SM100 if _major() == 10 else _TACTICS_SM90
    if tactic not in table:
        pytest.skip(f"{tactic} is not a variant of this architecture")


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
def test_mm_bf16_fp8(m, n, k, dtype, bias):
    torch.manual_seed(0)
    dev = torch.device("cuda")
    act = torch.randn(m, k, device=dev, dtype=torch.float32).to(dtype)
    w32 = torch.randn(n, k, device=dev, dtype=torch.float32) * 0.02
    w_fp8, scale = _quantize(w32)
    b = torch.randn(n, device=dev, dtype=torch.float32).to(dtype) if bias else None

    out = _mm(dtype)(act, w_fp8, scale, bias=b)
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
    "tactic",
    [
        None,
        # Blackwell: a 2SM tile, the 1SM tile, stream-K and the split-K reduce
        # (K = 256 is two 128-deep tiles).
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
def test_mm_bf16_fp8_every_fp8_value(dtype, tactic):
    """Every finite e4m3 byte (denormals included) must dequantize exactly, on every
    structurally different conversion path.

    act = identity over K, so out[m, n] = scale * weight[n, m] with no accumulation
    error beyond the final 16-bit rounding; the weight table enumerates all 254
    finite encodings (the two NaN bytes are excluded)."""
    if tactic is not None:
        _skip_unless_arch(tactic)
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
    out = _mm(dtype)(act, w_fp8, scale, tactic=tactic)
    ref = (w_fp8.float() * scale).T.to(dtype)
    assert torch.equal(out, ref), (tactic, (out.float() - ref.float()).abs().max())


def test_mm_bf16_fp8_3d_and_out():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    act = torch.randn(2, 33, 256, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(512, 256, device=dev) * 0.02)
    out = torch.empty(2, 33, 512, device=dev, dtype=torch.bfloat16)
    ret = mm_bf16_fp8(act, w_fp8, scale, out=out)
    assert ret is out
    ref = act.float().reshape(-1, 256) @ (w_fp8.float() * scale).T
    torch.testing.assert_close(out.float().reshape(-1, 512), ref, atol=5e-2, rtol=2e-2)
    # A non-contiguous out would be silently replaced by a copy: rejected instead.
    with pytest.raises(ValueError, match="contiguous"):
        mm_bf16_fp8(
            act,
            w_fp8,
            scale,
            out=torch.empty(2, 512, 33, device=dev, dtype=torch.bfloat16).transpose(
                1, 2
            ),
        )
    with pytest.raises(ValueError, match="out must be"):
        mm_bf16_fp8(
            act,
            w_fp8,
            scale,
            out=torch.empty(66, 512, device=dev, dtype=torch.bfloat16),
        )


def test_mm_bf16_fp8_caller_workspace():
    """A caller-owned workspace (any dtype, oversized) gives the same result as the
    allocated one; an undersized one is rejected before launch.  The split-K variant
    is forced so the call needs a workspace on both architectures."""
    torch.manual_seed(0)
    dev = torch.device("cuda")
    major = get_compute_capability(dev)[0]
    tactic = {9: "128x128_splitk", 10: "2sm256x128_splitk"}.get(major)
    if tactic is None:
        pytest.skip("mm_bf16_fp8 needs Hopper or Blackwell")
    m, n, k = 256, 1024, 5120
    act = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
    need = mm_bf16_fp8_workspace_bytes(m, n, k, dev, tactic)
    assert need > 0
    assert (
        mm_bf16_fp8_workspace_bytes(m, n, k, dev, tactic) == need
    )  # pure in its inputs
    # Same answer for fp16: the size does not depend on the activation dtype.
    assert mm_fp16_fp8_workspace_bytes(m, n, k, dev, tactic) == need
    ref = mm_bf16_fp8(act, w_fp8, scale, tactic=tactic)
    ws = torch.full(
        (need + 1000,), 0x7F, dtype=torch.uint8, device=dev
    )  # garbage, oversized
    out = mm_bf16_fp8(act, w_fp8, scale, tactic=tactic, workspace=ws)
    assert torch.equal(out, ref)
    ws16 = torch.empty((need // 2 + 8,), dtype=torch.bfloat16, device=dev)
    assert torch.equal(
        mm_bf16_fp8(act, w_fp8, scale, tactic=tactic, workspace=ws16), ref
    )
    with pytest.raises(ValueError, match="workspace too small"):
        mm_bf16_fp8(act, w_fp8, scale, tactic=tactic, workspace=ws[: need - 4])
    # The launcher below the public op never allocates: handed no workspace for a
    # plan that needs one, it refuses rather than silently owning the buffer.
    from flashinfer.gemm import w8a16_fp8_dispatch as dispatch
    from flashinfer.gemm.gemm_bf16_fp8 import (
        MM_BF16_FP8_TACTICS_SM90,
        MM_BF16_FP8_TACTICS_SM100,
    )

    code = {**MM_BF16_FP8_TACTICS_SM90, **MM_BF16_FP8_TACTICS_SM100}[tactic]
    out_buf = torch.empty(m, n, dtype=act.dtype, device=dev)
    with pytest.raises(ValueError, match="needs a workspace"):
        dispatch.launch_w8a16_fp8_gemm(act, w_fp8, scale, None, out_buf, code, None)
    with pytest.raises(ValueError, match="needs a workspace"):
        dispatch.launch_w8a16_fp8_gemm(
            act, w_fp8, scale, None, out_buf, code, ws[: need - 4]
        )
    dispatch.launch_w8a16_fp8_gemm(act, w_fp8, scale, None, out_buf, code, ws)
    assert torch.equal(out_buf, ref)
    # A single-pass shape needs no scratch (0 bytes), so any valid buffer is accepted.
    plain = {9: "128x256", 10: "2sm256x128"}[major]
    assert mm_bf16_fp8_workspace_bytes(4096, 4096, 5120, dev, plain) == 0
    tiny = torch.empty(1, dtype=torch.uint8, device=dev)
    big_act = torch.randn(4096, k, device=dev, dtype=torch.bfloat16)
    w2, s2 = _quantize(torch.randn(4096, k, device=dev) * 0.02)
    assert torch.equal(
        mm_bf16_fp8(big_act, w2, s2, tactic=plain, workspace=tiny),
        mm_bf16_fp8(big_act, w2, s2, tactic=plain),
    )


_MAX_DIM = 2**31 - 512  # the admitted bound on each of M, N, K (a multiple of 8 and 16)
_MAX_TILES64 = 2**28 - 1  # the admitted bound on ceil(M / 64) x ceil(N / 64)


def test_mm_bf16_fp8_workspace_bytes_extreme_dims():
    """The workspace query runs the dispatcher's cost models and tail / split planning
    without allocating anything, so it reaches the largest dimensions the launcher
    admits (a single dimension at the bound is a 32 GiB weight or output, so it is
    reachable; the tile-count bound is not).  Every variant's query must answer there
    (the tile-count arithmetic must not overflow), and one past the bound is rejected:
    a dimension over 2**31 - 512 would overflow the kernels' 32-bit tile-count ceiling
    division, which adds the tile (up to 256) before subtracting one, an output beyond
    2**28 - 1 tiles of 64x64 their 32-bit tile product."""
    dev = torch.device("cuda")
    big = _MAX_DIM
    tactics = [None] + (_TACTICS_SM100 if _major() == 10 else _TACTICS_SM90)
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
        for tactic in tactics:
            assert mm_bf16_fp8_workspace_bytes(m, n, k, dev, tactic) >= 0, (
                m,
                n,
                k,
                tactic,
            )
    for m, n, k in [(big + 8, 8, 16), (8, big + 8, 16), (8, 8, big + 16)]:
        with pytest.raises(ValueError, match=f"<= {_MAX_DIM}"):
            mm_bf16_fp8_workspace_bytes(m, n, k, dev)
    # 9 tile columns (N = 576) give 9 x (2^25 - 8) tiles, over the cap.
    with pytest.raises(ValueError, match="too large"):
        mm_bf16_fp8_workspace_bytes(big, 576, 1024, dev)
    with pytest.raises(ValueError, match="too large"):
        mm_bf16_fp8_workspace_bytes(2**20, 2**20, 1024, dev)


def test_mm_bf16_fp8_workspace_bytes_splitk_overflow():
    """The split policies and the workspace query must not overflow at large
    dimensions.  On Blackwell a forced split-K asks for `splits x M x N` fp32
    partials, which within the admitted tile count stays far inside the signed
    64-bit range the query returns in, and is reported exactly even though no one
    would allocate it (the long-K rule compares wave counts scaled by the split
    count).  On Hopper the split-K variants run a one-wave fused tail: a tail
    whose slices would not fit one wave of SMs runs the plain tile and needs no
    workspace, so the query reports 0 there.  Dimensions whose partials would
    leave the range are rejected by the tile-count bound before the query is
    asked."""
    dev = torch.device("cuda")
    tactic = {9: "128x128_splitk", 10: "2sm256x128_splitk"}[_major()]
    big = _MAX_DIM
    # 2 slices x 2^20 x 2^19 floats = 2^42 bytes (4 TiB).  The forced split on this
    # short K is the smallest, 2.
    expect_partials = _major() == 10
    for m, n, k, floor in [
        (2**20, 2**19, 1024, 2**42),
        # Long K with a large grid: the wave-fill split rules run at their extremes.
        (2**20, 2**19, 2**30, 2**42),
    ]:
        need = mm_bf16_fp8_workspace_bytes(m, n, k, dev, tactic)
        assert need >= floor if expect_partials else need == 0, (m, n, k, need)
    # The launcher's own choice at the extreme dimensions never picks a split it cannot
    # size (the Hopper cost model's partials cap is evaluated without overflowing).
    for m, n, k in [(big, 448, 1024), (2**20, 2**19, 2**30), (8, 8, big)]:
        assert mm_bf16_fp8_workspace_bytes(m, n, k, dev) >= 0
    for m, n, k in [(big, big, 1024), (2**30, 2**30, 2**30)]:
        with pytest.raises(ValueError, match="too large"):
            mm_bf16_fp8_workspace_bytes(m, n, k, dev, tactic)
    # Blackwell stacks the slices' partials as one output of splits x M rows and
    # derives row coordinates in 32 bits: a forced split on an M whose stacked extent
    # would pass 2^31 - 1 runs plain (no workspace), the query says so, and the
    # largest M that still stacks (2 slices) reports its partials.
    if expect_partials:
        assert mm_bf16_fp8_workspace_bytes(2**30 + 128, 8, 256, dev, tactic) == 0
        assert mm_bf16_fp8_workspace_bytes(2**30 + 128, 8, 256, dev) == 0
        # M at the dimension bound: 2 slices would stack to 2^32 - 1024 rows.
        assert mm_bf16_fp8_workspace_bytes(big, 448, 1024, dev, tactic) == 0
        assert (
            mm_bf16_fp8_workspace_bytes(2**30 - 128, 8, 256, dev, tactic)
            == 2 * (2**30 - 128) * 8 * 4
        )


def test_fast_divmod_magic_bounds():
    """The Blackwell tile scheduler divides by a magic multiplier, which is exact
    only while x * d < 2^40 and x * magic fits in 64 bits (magic is about 2^40 / d,
    so at d == 1 the product overflows from x = 2^23, long before the first bound
    fails).  The helper must fall back to the exact division (magic 0) at either
    bound and divide exactly just below it.  Runs the host-side DSL code itself,
    with no kernel launch."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.gemm.kernels.cute_dsl.w8a16_fp8_gemm_sm100 import (
        _FAST_DIVMOD_LIMIT,
        _fast_divmod,
        _fast_divmod_magic,
    )

    @cute.jit
    def probe(
        d: cutlass.Int32, x_max: cutlass.Int32, x: cutlass.Int32, out: cute.Tensor
    ):
        magic = _fast_divmod_magic(d, x_max)
        q, r = _fast_divmod(x, d, magic)
        out[0] = magic
        out[1] = q.to(cutlass.Int64)
        out[2] = r.to(cutlass.Int64)

    out = torch.zeros(3, dtype=torch.int64)

    def run(d, x_max, x):
        out.zero_()
        probe(d, x_max, x, from_dlpack(out))
        return out[0].item(), (out[1].item(), out[2].item())

    int32_max = 2**31 - 1
    int64_max = 2**63 - 1
    for d in (1, 2, 3, 132, 148, 1000, 2**20, int32_max):
        magic = -(-_FAST_DIVMOD_LIMIT // d)
        # The largest x_max under each bound, and one past it.
        exact_bound = (_FAST_DIVMOD_LIMIT - 1) // d
        product_bound = int64_max // magic
        edges = {exact_bound, exact_bound + 1, product_bound, product_bound + 1, 1000}
        for x_max in sorted(x for x in edges if 1 <= x <= int32_max):
            fast = x_max * d < _FAST_DIVMOD_LIMIT and x_max <= product_bound
            x = x_max - 1
            got_magic, got = run(d, x_max, x)
            assert (got_magic != 0) == fast, (d, x_max, got_magic)
            assert got == divmod(x, d), (d, x_max, x, got)
    # The shape that crashed: M=1, N=(2^23 + 1) * 128 on 128x128 tiles is 2^23 + 1
    # tiles along one axis and a single one along the other (d == 1); 2^23 - 1 tiles
    # take the fast path exactly.
    assert run(1, 2**23 + 1, 2**23)[0] == 0
    magic, got = run(1, 2**23 - 1, 2**23 - 2)
    assert magic == 1 << 40 and got == (2**23 - 2, 0)


def test_mm_bf16_fp8_rejects_aliasing():
    """The two buffers a call writes (out, workspace) must not share bytes with anything
    it reads, or with each other; inputs may alias one another.  Checked before launch,
    including on an empty call."""
    torch.manual_seed(0)
    dev = torch.device("cuda")
    major = get_compute_capability(dev)[0]
    tactic = {9: "128x128_splitk", 10: "2sm256x128_splitk"}[major]
    m, n, k = 256, 1024, 5120
    act = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
    ref = mm_bf16_fp8(act, w_fp8, scale, tactic=tactic)
    need = mm_bf16_fp8_workspace_bytes(m, n, k, dev, tactic)
    assert need > 0

    # out aliasing an input: a square problem lets out = act pass every shape check,
    # as does a window of the same storage that only partly covers act.
    sq = torch.randn(1536, 1024, device=dev, dtype=torch.bfloat16)
    w_sq, s_sq = _quantize(torch.randn(1024, 1024, device=dev) * 0.02)
    with pytest.raises(ValueError, match="out must not overlap a in memory"):
        mm_bf16_fp8(sq[:1024], w_sq, s_sq, out=sq[:1024])
    with pytest.raises(ValueError, match="out must not overlap a in memory"):
        mm_bf16_fp8(sq[:1024], w_sq, s_sq, out=sq[512:])
    out_buf = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="out must not overlap bias"):
        mm_bf16_fp8(act, w_fp8, scale, bias=out_buf[0], out=out_buf)

    # workspace aliasing the scale: the split-K reduction reads the scale after the
    # partials have been written over it.
    ws_f32 = torch.empty(need // 4 + 4, device=dev, dtype=torch.float32)
    ws_f32[0] = scale[0]
    with pytest.raises(ValueError, match="workspace must not overlap alpha in memory"):
        mm_bf16_fp8(act, w_fp8, ws_f32[:1], tactic=tactic, workspace=ws_f32)
    # workspace containing out / weight / act / bias.
    buf = torch.empty(
        need + max(m * n * 2, m * k * 2, n * k) + 4096, device=dev, dtype=torch.uint8
    )
    out_in_buf = buf[: m * n * 2].view(torch.bfloat16).view(m, n)
    with pytest.raises(ValueError, match="workspace must not overlap out"):
        mm_bf16_fp8(act, w_fp8, scale, tactic=tactic, out=out_in_buf, workspace=buf)
    with pytest.raises(ValueError, match="workspace must not overlap b in memory"):
        mm_bf16_fp8(
            act,
            buf[: n * k].view(torch.float8_e4m3fn).view(n, k),
            scale,
            tactic=tactic,
            workspace=buf,
        )
    with pytest.raises(ValueError, match="workspace must not overlap a in memory"):
        mm_bf16_fp8(
            buf[: m * k * 2].view(torch.bfloat16).view(m, k),
            w_fp8,
            scale,
            tactic=tactic,
            workspace=buf,
        )
    with pytest.raises(ValueError, match="workspace must not overlap bias"):
        mm_bf16_fp8(
            act,
            w_fp8,
            scale,
            bias=buf[: n * 2].view(torch.bfloat16),
            tactic=tactic,
            workspace=buf,
        )
    # The empty call checks the same rule for the buffers that do have bytes.
    with pytest.raises(ValueError, match="workspace must not overlap alpha in memory"):
        mm_bf16_fp8(act[:0], w_fp8, ws_f32[:1], tactic=tactic, workspace=ws_f32)

    # Inputs may alias one another (bias as a row of act), and a workspace that merely
    # sits next to out in one allocation is fine.
    bias = act[0, :n]  # contiguous row slice, 16-byte aligned
    assert torch.equal(
        mm_bf16_fp8(act, w_fp8, scale, bias=bias, tactic=tactic),
        mm_bf16_fp8(act, w_fp8, scale, bias=bias.clone(), tactic=tactic),
    )
    ws_after = buf[m * n * 2 :]
    assert torch.equal(
        mm_bf16_fp8(
            act, w_fp8, scale, tactic=tactic, out=out_in_buf, workspace=ws_after
        ),
        ref,
    )


def test_mm_bf16_fp8_rejects_bad_inputs():
    dev = torch.device("cuda")
    act = torch.randn(4, 64, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(32, 64, device=dev))
    with pytest.raises(ValueError):
        mm_bf16_fp8(act.float(), w_fp8, scale)
    with pytest.raises(ValueError):
        mm_bf16_fp8(act, w_fp8.to(torch.bfloat16), scale)
    with pytest.raises(ValueError):
        mm_bf16_fp8(act[:, :32], w_fp8, scale)
    # No implicit cast of the scale: the caller passes float32 or casts it.
    for bad in (scale.half(), scale.to(torch.bfloat16), scale.double()):
        with pytest.raises(ValueError, match="alpha must be float32"):
            mm_bf16_fp8(act, w_fp8, bad)
    # The scale is shape () or (1,), as documented and traced; one element in any
    # other shape is rejected, as is a 0-d activation.
    for bad in (scale.reshape(1, 1), scale.reshape(1, 1, 1)):
        with pytest.raises(ValueError, match="alpha is one per-tensor value"):
            mm_bf16_fp8(act, w_fp8, bad)
    with pytest.raises(ValueError, match="at least one dimension"):
        mm_bf16_fp8(act[0, 0], w_fp8, scale)
    # Each public name takes one activation dtype; the message points at the other.
    with pytest.raises(
        ValueError, match="mm_bf16_fp8 takes a torch.bfloat16.*mm_fp16_fp8"
    ):
        mm_bf16_fp8(act.half(), w_fp8, scale)
    with pytest.raises(
        ValueError, match="mm_fp16_fp8 takes a torch.float16.*mm_bf16_fp8"
    ):
        mm_fp16_fp8(act, w_fp8, scale)
    # out_dtype exists for uniformity with the mm_* family; only a's dtype is served.
    with pytest.raises(ValueError, match="out_dtype must be a's dtype"):
        mm_bf16_fp8(act, w_fp8, scale, out_dtype=torch.float16)
    assert torch.equal(
        mm_bf16_fp8(act, w_fp8, scale, out_dtype=torch.bfloat16),
        mm_bf16_fp8(act, w_fp8, scale),
    )
    # backend: "auto" and the one implementation; anything else is refused.
    assert torch.equal(
        mm_bf16_fp8(act, w_fp8, scale, backend="cute-dsl"),
        mm_bf16_fp8(act, w_fp8, scale),
    )
    with pytest.raises(Exception, match="cublas"):
        mm_bf16_fp8(act, w_fp8, scale, backend="cublas")
    assert torch.equal(
        mm_bf16_fp8(act, w_fp8, scale.half().float()),
        mm_bf16_fp8(act, w_fp8, scale.half().float().reshape(())),
    )
    # bias: shape, dtype, device and contiguity are all checked before launch.
    with pytest.raises(ValueError, match="bias must be"):
        mm_bf16_fp8(
            act, w_fp8, scale, bias=torch.zeros(31, device=dev, dtype=act.dtype)
        )
    with pytest.raises(ValueError, match="bias must be"):
        mm_bf16_fp8(act, w_fp8, scale, bias=torch.zeros(32, device=dev))
    with pytest.raises(ValueError, match="device"):
        mm_bf16_fp8(act, w_fp8, scale, bias=torch.zeros(32, dtype=act.dtype))
    with pytest.raises(ValueError, match="contiguous"):
        mm_bf16_fp8(
            act, w_fp8, scale, bias=torch.zeros(64, device=dev, dtype=act.dtype)[::2]
        )


def test_mm_bf16_fp8_forced_splitk_unsplittable_k():
    """A split-K variant forced on a K it cannot slice is rejected with the rule, by
    the workspace query as well as by the call (both resolve the same plan), instead
    of the query reporting a size and the launch failing."""
    dev = torch.device("cuda")
    major = get_compute_capability(dev)[0]
    tactic = {9: "128x128_splitk", 10: "2sm256x128_splitk"}[major]
    m, n, k = (
        256,
        1024,
        960,
    )  # K % 16 == 0, but neither 64 x (2 x 8) nor 128 x 2 divides it
    act = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
    with pytest.raises(ValueError, match="cannot slice K=960"):
        mm_bf16_fp8_workspace_bytes(m, n, k, dev, tactic)
    with pytest.raises(ValueError, match="cannot slice K=960"):
        mm_bf16_fp8(act, w_fp8, scale, tactic=tactic)
    # The launcher's own choice never asks for an impossible split.
    assert mm_bf16_fp8_workspace_bytes(m, n, k, dev) >= 0
    out = mm_bf16_fp8(act, w_fp8, scale)
    ref = act.float() @ (w_fp8.float() * scale).T
    torch.testing.assert_close(out.float(), ref, atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize("n,k", [(1024, 5128), (1028, 5120)])
def test_mm_bf16_fp8_rejects_misaligned(n, k):
    """K % 16 and N % 8 are rejected upfront with a clear message (no silent fallback)."""
    dev = torch.device("cuda")
    act = torch.randn(64, k, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev))
    with pytest.raises(ValueError, match="multiple of"):
        mm_bf16_fp8(act, w_fp8, scale)


def test_mm_bf16_fp8_rejects_misaligned_storage():
    """A contiguous view at an odd storage offset is rejected (a misaligned pointer
    would otherwise fault the CUDA context from the TMA / vector-store paths)."""
    dev = torch.device("cuda")
    major = get_compute_capability(dev)[0]
    tactic = {9: "128x128_splitk", 10: "2sm256x128_splitk"}[major]
    m, n, k = 7, 8, 1024
    act = torch.ones(m, k, device=dev, dtype=torch.bfloat16)
    w_fp8 = torch.ones(n, k, device=dev).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.5], device=dev)
    out = torch.full((m * n + 1,), -1.0, device=dev, dtype=torch.bfloat16)[1:].view(
        m, n
    )
    assert out.is_contiguous()
    with pytest.raises(ValueError, match="aligned"):
        mm_bf16_fp8(act, w_fp8, scale, out=out, tactic=tactic)
    need = mm_bf16_fp8_workspace_bytes(m, n, k, dev, tactic)
    ws = torch.empty(need // 4 + 1, device=dev, dtype=torch.float32)[1:]
    with pytest.raises(ValueError, match="aligned"):
        mm_bf16_fp8(act, w_fp8, scale, tactic=tactic, workspace=ws)
    act_off = torch.ones(m * k + 1, device=dev, dtype=torch.bfloat16)[1:].view(m, k)
    with pytest.raises(ValueError, match="aligned"):
        mm_bf16_fp8(act_off, w_fp8, scale, tactic=tactic)
    bias_off = torch.ones(n + 1, device=dev, dtype=torch.bfloat16)[1:]
    with pytest.raises(ValueError, match="aligned"):
        mm_bf16_fp8(act, w_fp8, scale, bias=bias_off, tactic=tactic)
    # A view at a 16-byte-aligned offset is fine (and the context is still alive).
    out_ok = torch.full((m * n + 8,), -1.0, device=dev, dtype=torch.bfloat16)[8:].view(
        m, n
    )
    mm_bf16_fp8(act, w_fp8, scale, out=out_ok, tactic=tactic)
    torch.cuda.synchronize()
    assert torch.equal(
        out_ok, torch.full((m, n), 512.0, device=dev, dtype=torch.bfloat16)
    )


def test_mm_bf16_fp8_empty_m():
    """M = 0 returns the empty output without launching (the tile schedulers divide
    by the tile count); N = 0 and K = 0 are rejected upfront."""
    dev = torch.device("cuda")
    w_fp8, scale = _quantize(torch.randn(64, 256, device=dev))
    out = mm_bf16_fp8(
        torch.empty(0, 256, device=dev, dtype=torch.bfloat16), w_fp8, scale
    )
    assert out.shape == (0, 64) and out.dtype == torch.bfloat16
    out3 = mm_fp16_fp8(
        torch.empty(2, 0, 256, device=dev, dtype=torch.float16), w_fp8, scale
    )
    assert out3.shape == (2, 0, 64) and out3.dtype == torch.float16
    given = torch.empty(0, 64, device=dev, dtype=torch.bfloat16)
    assert (
        mm_bf16_fp8(
            torch.empty(0, 256, device=dev, dtype=torch.bfloat16),
            w_fp8,
            scale,
            out=given,
            tactic="128x128" if get_compute_capability(dev)[0] == 9 else "1sm128x128",
        )
        is given
    )
    assert mm_bf16_fp8_workspace_bytes(0, 64, 256, dev) == 0
    # The empty call validates like a non-empty one: a CPU out, a CPU or
    # mis-shaped bias and a CPU weight are rejected, not returned untouched.
    empty = torch.empty(0, 256, device=dev, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="device"):
        mm_bf16_fp8(empty, w_fp8, scale, out=torch.empty(0, 64, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="device"):
        mm_bf16_fp8(empty, w_fp8, scale, bias=torch.zeros(64, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="bias must be"):
        mm_bf16_fp8(
            empty, w_fp8, scale, bias=torch.zeros(65, device=dev, dtype=torch.bfloat16)
        )
    with pytest.raises(ValueError, match="device"):
        mm_bf16_fp8(empty, w_fp8.cpu(), scale)
    # weight contiguity and weight / bias alignment are checked on the empty call too.
    wide = torch.randn(64, 512, device=dev).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="b must be contiguous"):
        mm_bf16_fp8(empty, wide[:, ::2], scale)
    w_off = (
        torch.randn(64 * 256 + 1, device=dev).to(torch.float8_e4m3fn)[1:].view(64, 256)
    )
    with pytest.raises(ValueError, match="aligned"):
        mm_bf16_fp8(empty, w_off, scale)
    b_off = torch.zeros(65, device=dev, dtype=torch.bfloat16)[1:]
    with pytest.raises(ValueError, match="aligned"):
        mm_bf16_fp8(empty, w_fp8, scale, bias=b_off)
    act = torch.empty(4, 0, device=dev, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="K > 0"):
        mm_bf16_fp8(act, torch.empty(64, 0, device=dev).to(torch.float8_e4m3fn), scale)
    with pytest.raises(ValueError, match="N > 0"):
        mm_bf16_fp8(
            torch.empty(4, 256, device=dev, dtype=torch.bfloat16),
            torch.empty(0, 256, device=dev).to(torch.float8_e4m3fn),
            scale,
        )
    with pytest.raises(ValueError, match="N > 0"):
        mm_bf16_fp8_workspace_bytes(4, 0, 256, dev)
    with pytest.raises(ValueError, match="K > 0"):
        mm_bf16_fp8_workspace_bytes(4, 64, 0, dev)


def test_mm_bf16_fp8_strided_act():
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
        with pytest.raises(ValueError, match="a must be contiguous"):
            mm_bf16_fp8(t, w_fp8, scale)
        out = mm_bf16_fp8(t.contiguous(), w_fp8, scale)
        assert out.shape == t.shape[:-1] + (n,)
        ref = t.float().reshape(-1, k) @ (w_fp8.float() * scale).T
        torch.testing.assert_close(
            out.float().reshape(-1, n), ref, atol=5e-2, rtol=2e-2
        )


def test_mm_bf16_fp8_device_resolution():
    """``torch.device("cuda")`` without an index means the current device, same as
    omitting it; a CPU device is rejected."""
    dev = torch.device("cuda")
    cur = torch.device("cuda", torch.cuda.current_device())
    assert mm_bf16_fp8_workspace_bytes(
        256, 1024, 5120, dev
    ) == mm_bf16_fp8_workspace_bytes(256, 1024, 5120, cur)
    assert mm_bf16_fp8_workspace_bytes(256, 1024, 5120) == mm_bf16_fp8_workspace_bytes(
        256, 1024, 5120, cur
    )
    with pytest.raises(ValueError, match="CUDA"):
        mm_bf16_fp8_workspace_bytes(256, 1024, 5120, torch.device("cpu"))


def test_mm_bf16_fp8_rejects_unknown_tactic_name():
    """A tactic name outside the architecture's table is rejected ("hand" is not one)."""
    dev = torch.device("cuda")
    act = torch.randn(16, 256, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(128, 256, device=dev))
    with pytest.raises(ValueError):
        mm_bf16_fp8(act, w_fp8, scale, tactic="hand")


_TACTICS_SM100 = [
    "2sm256x128",
    "2sm256x192",
    "1sm128x128",
    "2sm256x128_splitk",
    "2sm256x128_streamk",
    "2sm256x160",
]
_TACTICS_SM90 = [
    "128x256",
    "128x192",
    "128x128",
    "64x256",
    "128x256_streamk",
    "128x192_streamk",
    "64x128",
    "128x128_splitk",
    "128x256_splitk",
    "64x128_splitk",
]


@pytest.mark.parametrize("tactic", _TACTICS_SM100 + _TACTICS_SM90)
@pytest.mark.parametrize(
    "m,n,k", [(1024, 1024, 5120), (200, 1024, 5120), (4096, 128, 5120)]
)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mm_bf16_fp8_tactic(tactic, m, n, k, bias, dtype):
    """Every kernel variant reachable through ``tactic`` must agree with the cast-copy
    reference; a variant of another architecture is rejected."""
    dev = torch.device("cuda")
    major = get_compute_capability(dev)[0]
    mine = tactic in (
        _TACTICS_SM100 if major == 10 else _TACTICS_SM90 if major == 9 else []
    )
    torch.manual_seed(0)
    act = torch.randn(m, k, device=dev, dtype=dtype)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
    b = torch.randn(n, device=dev, dtype=dtype) if bias else None
    if not mine:
        with pytest.raises(ValueError):
            _mm(dtype)(act, w_fp8, scale, bias=b, tactic=tactic)
        return
    out = _mm(dtype)(act, w_fp8, scale, bias=b, tactic=tactic)
    ref32 = act.float() @ (w_fp8.float() * scale).T
    if b is not None:
        ref32 = ref32 + b.float()
    cast_err = (
        (_cast_copy_reference(act, w_fp8, scale, b).float() - ref32).abs().max().item()
    )
    err = (out.float() - ref32).abs().max().item()
    assert torch.isfinite(out.float()).all()
    assert err <= max(2.0 * cast_err, 2e-3 * ref32.abs().max().item()), (
        tactic,
        err,
        cast_err,
    )
    with pytest.raises(ValueError):
        _mm(dtype)(act, w_fp8, scale, tactic="nope")


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
    ("2sm256x128_splitk", 4097, 16392, 4096),
]


@pytest.mark.parametrize("tactic,m,n,k", _GROUPED_CASES)
def test_mm_bf16_fp8_grouped_traversal(tactic, m, n, k):
    """Shapes that actually enable the L2 tile grouping (Hopper grouped scheduler,
    Blackwell swizzle), with a final incomplete group and, for split-K, the batch
    index on top.  The output is pre-filled with NaN so a tile the traversal skips
    fails the finiteness check, and a tile visited twice still has to agree with the
    reference."""
    _skip_unless_arch(tactic)
    torch.manual_seed(0)
    dev = torch.device("cuda")
    act = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
    out = torch.full((m, n), float("nan"), device=dev, dtype=torch.bfloat16)
    ret = mm_bf16_fp8(act, w_fp8, scale, tactic=tactic, out=out)
    assert ret is out
    _check_against_reference(out, act, w_fp8, scale, None, (tactic, m, n, k))
    del act, w_fp8, out


# ---------------------------------------------------------------------------
# K tails, short prologues, N tails on every single-pass variant
# ---------------------------------------------------------------------------

_SINGLE_PASS_ALGOS = [a for a in _TACTICS_SM100 + _TACTICS_SM90 if "splitk" not in a]


@pytest.mark.parametrize("tactic", _SINGLE_PASS_ALGOS)
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
def test_mm_bf16_fp8_small_tails(tactic, m, n, k, dtype):
    """Every single-pass variant on problems shorter than its pipeline: TMA zero-fill
    of the K tail, a prologue of one or two k-tiles, N tails under the epilogue
    predication, odd M, both activation dtypes, with bias."""
    _skip_unless_arch(tactic)
    torch.manual_seed(0)
    dev = torch.device("cuda")
    act = torch.randn(m, k, device=dev, dtype=dtype)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
    bias = torch.randn(n, device=dev, dtype=dtype)
    out = _mm(dtype)(act, w_fp8, scale, bias=bias, tactic=tactic)
    _check_against_reference(out, act, w_fp8, scale, bias, (tactic, m, n, k, dtype))


@pytest.mark.parametrize(
    "tactic", [a for a in _TACTICS_SM100 + _TACTICS_SM90 if "splitk" in a]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mm_bf16_fp8_minimal_splitk(tactic, dtype):
    """Each split-K variant at the shortest K it can slice (Hopper: 1024 = 2 x 8
    k-tiles of 64; Blackwell: 256 = 2 k-tiles of 128), on an N tail and odd M; the
    next shorter legal K is rejected by both the query and the call."""
    _skip_unless_arch(tactic)
    torch.manual_seed(0)
    dev = torch.device("cuda")
    k, k_short = (1024, 512) if _major() == 9 else (256, 128)
    m, n = 77, 136
    act = torch.randn(m, k, device=dev, dtype=dtype)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
    bias = torch.randn(n, device=dev, dtype=dtype)
    assert mm_bf16_fp8_workspace_bytes(m, n, k, dev, tactic) >= 2 * m * n * 4
    out = _mm(dtype)(act, w_fp8, scale, bias=bias, tactic=tactic)
    _check_against_reference(out, act, w_fp8, scale, bias, (tactic, dtype))
    with pytest.raises(ValueError, match=f"cannot slice K={k_short}"):
        mm_bf16_fp8_workspace_bytes(m, n, k_short, dev, tactic)
    with pytest.raises(ValueError, match=f"cannot slice K={k_short}"):
        _mm(dtype)(
            act[:, :k_short].contiguous(),
            w_fp8[:, :k_short].contiguous(),
            scale,
            tactic=tactic,
        )


# ---------------------------------------------------------------------------
# Caller workspace: stream-K and split-K, reuse on one non-default stream
# ---------------------------------------------------------------------------


def _shape_needing_workspace(tactic, dev):
    """A problem on which the forced variant reports a nonzero workspace (a stream-K
    variant may decompose a problem data-parallel and need none)."""
    for m, n, k in [
        (1024, 512, 5120),
        (300, 1024, 25600),
        (2048, 1024, 5120),
        (1024, 1024, 8192),
        (256, 1024, 5120),
    ]:
        if mm_bf16_fp8_workspace_bytes(m, n, k, dev, tactic) > 0:
            return m, n, k
    raise AssertionError(f"no candidate shape needs a workspace under {tactic}")


@pytest.mark.parametrize(
    "tactic",
    ["2sm256x128_splitk", "2sm256x128_streamk", "128x128_splitk", "128x256_streamk"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mm_bf16_fp8_caller_workspace_reuse(tactic, dtype):
    """One poisoned caller buffer reused by several calls queued on a non-default
    stream, with changed activations, scales and biases, distinct outputs and one
    smaller problem; checked after a single synchronization.  Canary bytes around the
    supplied workspace and output views must survive (scratch and vector-store
    bounds).  Nothing in the buffer may be assumed between calls: per-launch barrier
    initialization and partial overwrite are what this checks."""
    _skip_unless_arch(tactic)
    torch.manual_seed(0)
    dev = torch.device("cuda")
    m, n, k = _shape_needing_workspace(tactic, dev)
    problems = [(m, 1.0, True), (m, 0.5, False), (m // 2, 2.0, True), (m, 1.0, False)]
    # One buffer sized for the largest need among the problems it serves (a smaller
    # problem may resolve to a different split count).
    need = max(_ws(dtype)(mi, n, k, dev, tactic) for mi, _, _ in problems)
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
            ret = _mm(dtype)(
                act, w_fp8, sc, bias=bias, tactic=tactic, out=out, workspace=ws
            )
            assert ret is out
            outs.append(obuf)
            cases.append((mi, act, sc, bias))
    torch.cuda.synchronize()
    for obuf, (mi, act, sc, bias) in zip(outs, cases, strict=True):
        out = obuf[pad_el : pad_el + mi * n].view(mi, n)
        assert torch.isnan(obuf[:pad_el]).all() and torch.isnan(obuf[-pad_el:]).all()
        _check_against_reference(out, act, w_fp8, sc, bias, (tactic, dtype, mi))
    assert (buf[:pad] == 0x7F).all() and (buf[-pad:] == 0x7F).all()


# ---------------------------------------------------------------------------
# K tails under a forced stream-K, concurrent streams with their own workspaces,
# graph replay, and a minimally aligned scalar scale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tactic", [a for a in _TACTICS_SM100 + _TACTICS_SM90 if "streamk" in a]
)
@pytest.mark.parametrize("m,n", [(1, 128), (300, 1024), (77, 136)])
def test_mm_bf16_fp8_streamk_medium_k_tail(tactic, m, n):
    """A forced stream-K variant on a K that is not a multiple of the tile depth
    (2064 = 16 x 128 + 16 = 32 x 64 + 16): the launcher's split substitute must not
    slice such a K, and no decomposition may drop the 16-element tail.  The weight is
    zero except on the tail columns, so every output equals the tail's dot product."""
    _skip_unless_arch(tactic)
    dev = torch.device("cuda")
    k = 2064
    act = torch.ones(m, k, device=dev, dtype=torch.bfloat16)
    w = torch.zeros(n, k, device=dev)
    w[:, 2048:] = 1.0
    w_fp8 = w.to(torch.float8_e4m3fn)
    scale = torch.ones(1, device=dev)
    out = mm_bf16_fp8(act, w_fp8, scale, tactic=tactic)
    assert torch.equal(out, torch.full((m, n), 16.0, device=dev, dtype=out.dtype))
    # And a random problem of the same shape against the reference.
    torch.manual_seed(0)
    act = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
    out = mm_bf16_fp8(act, w_fp8, scale, tactic=tactic)
    _check_against_reference(out, act, w_fp8, scale, None, (tactic, m, n, k))


def _tail_tactic_and_shape(dev):
    """A forced variant and a shape on which this architecture splits K and needs a
    workspace (Hopper: a stream-K tail; Blackwell: split-K)."""
    tactic = {9: "128x256_streamk", 10: "2sm256x128_splitk"}[_major()]
    return tactic, _shape_needing_workspace(tactic, dev)


def test_mm_bf16_fp8_two_streams_own_workspaces():
    """Two streams, each running the same split problem on its own workspace and
    output, interleaved and unsynchronized until the end: the slices of a tile may
    only ever count arrivals of their own call (the counters live in the call's
    workspace), so neither stream can release the other's tiles early."""
    dev = torch.device("cuda")
    tactic, (m, n, k) = _tail_tactic_and_shape(dev)
    torch.manual_seed(0)
    act = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
    ref = mm_bf16_fp8(act, w_fp8, scale, tactic=tactic)
    need = mm_bf16_fp8_workspace_bytes(m, n, k, dev, tactic)
    assert need > 0
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    wss = [torch.full((need,), 0x7F, dtype=torch.uint8, device=dev) for _ in streams]
    outs = [[] for _ in streams]
    rounds = 24
    for s in streams:
        s.wait_stream(torch.cuda.current_stream())
    for _ in range(rounds):
        for i, s in enumerate(streams):
            with torch.cuda.stream(s):
                o = torch.empty(m, n, device=dev, dtype=act.dtype)
                mm_bf16_fp8(act, w_fp8, scale, tactic=tactic, out=o, workspace=wss[i])
                outs[i].append(o)
    torch.cuda.synchronize()
    for i in range(len(streams)):
        for r, o in enumerate(outs[i]):
            assert torch.equal(o, ref), (tactic, i, r)


def test_mm_bf16_fp8_graph_replay_two_graphs():
    """One split call per stream captured into a CUDA graph, each with its own
    inputs, output and workspace, the two graphs replayed repeatedly and interleaved
    on their streams.  Before every replay, on the replay stream, the output is set
    to NaN and the workspace to garbage: a replay must rebuild everything it reads
    back (partials, arrival counters) inside the graph, and stale partials of the
    previous replay cannot stand in for a slice released early."""
    dev = torch.device("cuda")
    tactic, (m, n, k) = _tail_tactic_and_shape(dev)
    torch.manual_seed(0)
    need = mm_bf16_fp8_workspace_bytes(m, n, k, dev, tactic)
    graphs, outs, streams, wss, refs, inputs = [], [], [], [], [], []
    for i in range(2):
        act = torch.randn(m, k, device=dev, dtype=torch.bfloat16) * (i + 1)
        w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
        inputs.append((act, w_fp8, scale))  # the graphs hold raw pointers
        refs.append(mm_bf16_fp8(act, w_fp8, scale, tactic=tactic))
        s = torch.cuda.Stream()
        ws = torch.full((need,), 0x7F, dtype=torch.uint8, device=dev)
        out = torch.empty(m, n, device=dev, dtype=act.dtype)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            # Warm-up on the side stream (compiles, allocates), then capture.
            mm_bf16_fp8(act, w_fp8, scale, tactic=tactic, out=out, workspace=ws)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                mm_bf16_fp8(act, w_fp8, scale, tactic=tactic, out=out, workspace=ws)
        graphs.append(g)
        outs.append(out)
        streams.append(s)
        wss.append(ws)  # ...so every buffer a graph reads or writes stays alive
    torch.cuda.synchronize()
    for rep in range(12):
        for g, s, out, ws in zip(graphs, streams, outs, wss, strict=True):
            with torch.cuda.stream(s):
                out.fill_(float("nan"))
                ws.fill_(0x7F if rep % 2 else 0x00)
                g.replay()
        torch.cuda.synchronize()
        for i, (out, ref) in enumerate(zip(outs, refs, strict=True)):
            assert torch.equal(out, ref), (tactic, rep, i)


@pytest.mark.parametrize("split", [False, True])
def test_mm_bf16_fp8_scalar_scale_minimally_aligned(split, monkeypatch):
    """A legal scale is any one-element fp32 view, so it may be only 4-byte aligned
    (an odd element of a larger buffer); nothing in the launch may declare more for
    it, with or without a workspace in use.

    The declared alignments are checked only when a kernel compiles, so the
    misaligned calls run with both cache layers off (in-process and on-disk): a
    warm cache from the reference call would hide an over-declaration."""
    from flashinfer.gemm import w8a16_fp8_dispatch as dispatch

    dev = torch.device("cuda")
    if split:
        tactic, (m, n, k) = _tail_tactic_and_shape(dev)
    else:
        tactic, (m, n, k) = (
            {9: "128x256", 10: "2sm256x128"}[_major()],
            (256, 1024, 2048),
        )
    torch.manual_seed(0)
    act = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
    ref = mm_bf16_fp8(act, w_fp8, scale, tactic=tactic)
    buf = torch.zeros(3, device=dev, dtype=torch.float32)
    buf[1] = scale[0]
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "1")
    for view in (buf[1:2], buf[1]):
        assert view.data_ptr() % 16 == 4
        dispatch._KERNEL_CACHE.clear()
        out = mm_bf16_fp8(act, w_fp8, view, tactic=tactic)
        torch.cuda.synchronize()
        assert torch.equal(out, ref)


@pytest.mark.parametrize("wait_ns", [0, 2_000])
def test_mm_bf16_fp8_tail_withdrawal(wait_ns, monkeypatch):
    """The slices of a Hopper tail tile wait for each other only for a bounded time;
    a slice whose wait runs out withdraws, and the slices present when the last one
    arrives share the tile.  Compiled with a wait of 0 ns (every slice that is not
    last withdraws at its first look) or 2 us (some do, depending on arrival skew),
    the fix-up lands on changing subsets of the slices, down to the last slice alone.
    The summation order is fixed, so the output must be bitwise the default kernel's
    whatever the subset."""
    if _major() != 9:
        pytest.skip("the bounded tail wait is the Hopper kernel's")
    from flashinfer.gemm import w8a16_fp8_dispatch as dispatch

    dev = torch.device("cuda")
    torch.manual_seed(0)
    cases = [
        (tactic, _shape_needing_workspace(tactic, dev))
        for tactic in (
            "128x256_streamk",
            "128x192_streamk",
            "128x128_splitk",
            "64x128_splitk",
        )
    ]
    # A shape whose automatic plan cuts a multi-slice tail on H100 (43 tiles x 3).
    cases.append((None, (1283, 3080, 25600)))
    problems = []
    for tactic, (m, n, k) in cases:
        act = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
        bias = torch.randn(n, device=dev, dtype=torch.bfloat16)
        ref = mm_bf16_fp8(act, w_fp8, scale, bias=bias, tactic=tactic)
        problems.append((tactic, act, w_fp8, scale, bias, ref))
    torch.cuda.synchronize()
    monkeypatch.setattr(dispatch, "_SM90_TAIL_WAIT_NS", wait_ns)
    dispatch._KERNEL_CACHE.clear()
    for rep in range(6):
        for tactic, act, w_fp8, scale, bias, ref in problems:
            out = mm_bf16_fp8(act, w_fp8, scale, bias=bias, tactic=tactic)
            torch.cuda.synchronize()
            assert torch.equal(out, ref), (wait_ns, tactic, rep)
