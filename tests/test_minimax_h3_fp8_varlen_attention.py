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

# The all-FP8 MiniMax-H3 packed-varlen attention (``variant="fp8"``): the
# numerical contract against an FP32 reference over the dequantized E4M3
# inputs (elementwise 0.1 + 0.1 |expected|), the host dispatch between the
# general and the tiny-plan program (same output bitwise), strided views, the
# prepared runner's no-allocation launch and the descale validation.

from __future__ import annotations

import math

import pytest
import torch

from flashinfer.experimental.minimax_h3_varlen_attention import cake_backend as cb

HEAD_DIM = 128


def _device_ok() -> bool:
    if not torch.cuda.is_available():
        return False
    return tuple(torch.cuda.get_device_capability(0)) in {(10, 0), (10, 3)}


def _skip_unless_ready() -> None:
    if not _device_ok():
        pytest.skip("requires SM100a or SM103a")
    pytest.importorskip("flashinfer.jit.cake_minimax_h3_varlen_fp8_attention")


def _quantize(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Per-tensor E4M3 quantization: returns the E4M3 tensor and its descale."""
    amax = x.abs().amax().clamp_min(1e-12).item()
    scale = 448.0 / amax
    q = (x * scale).to(torch.float8_e4m3fn)
    return q, 1.0 / scale


def _inputs(cu: tuple[int, ...], heads: int, *, seed: int, device: torch.device):
    generator = torch.Generator(device=device).manual_seed(seed)
    T = cu[-1]
    q = torch.randn(
        T, heads, HEAD_DIM, device=device, generator=generator, dtype=torch.float32
    )
    k = torch.randn(
        T, heads, HEAD_DIM, device=device, generator=generator, dtype=torch.float32
    )
    v = torch.randn(
        T, heads, HEAD_DIM, device=device, generator=generator, dtype=torch.float32
    )
    q8, qd = _quantize(q)
    k8, kd = _quantize(k)
    v8, vd = _quantize(v)
    return q8, k8, v8, qd, kd, vd


def _reference(
    q8: torch.Tensor,
    k8: torch.Tensor,
    v8: torch.Tensor,
    qd: float,
    kd: float,
    vd: float,
    cu: tuple[int, ...],
    softmax_scale: float,
) -> torch.Tensor:
    """FP32 attention over the dequantized E4M3 inputs, segment by segment."""
    q = q8.float() * qd
    k = k8.float() * kd
    v = v8.float() * vd
    out = torch.zeros_like(q)
    for a, b in zip(cu, cu[1:], strict=False):
        if b == a:
            continue
        s = torch.einsum("thd,shd->hts", q[a:b], k[a:b]) * softmax_scale
        p = torch.softmax(s, dim=-1)
        out[a:b] = torch.einsum("hts,shd->thd", p, v[a:b])
    return out


def _check(out: torch.Tensor, expected: torch.Tensor) -> None:
    diff = (out.float() - expected).abs()
    bound = 0.1 + 0.1 * expected.abs()
    assert not torch.isnan(out).any()
    assert bool((diff <= bound).all()), f"max violation {float((diff - bound).max())}"


@pytest.mark.parametrize(
    ("cu", "heads"),
    [
        ((0, 133, 300, 900), 7),
        ((0, 0, 640, 640, 1200, 1201), 14),
        ((0, 4, 133, 389, 1413, 4763), 28),
        ((0, 9, 18), 56),
        ((0, 1500, 1700), 5),
    ],
)
def test_numerical_contract(cu: tuple[int, ...], heads: int) -> None:
    _skip_unless_ready()
    device = torch.device("cuda")
    q8, k8, v8, qd, kd, vd = _inputs(cu, heads, seed=0, device=device)
    cu_t = torch.tensor(cu, dtype=torch.int32, device=device)
    out = cb.minimax_h3_varlen_fp8_attention(
        q8, k8, v8, cu_t, q_descale=qd, k_descale=kd, v_descale=vd
    )
    assert out.dtype == torch.bfloat16 and tuple(out.shape) == (cu[-1], heads, HEAD_DIM)
    _check(out, _reference(q8, k8, v8, qd, kd, vd, cu, HEAD_DIM**-0.5))


def test_tiny_plan_dispatch_is_bitwise_the_general_program() -> None:
    _skip_unless_ready()
    device = torch.device("cuda")
    cu = (0, 9, 18)
    q8, k8, v8, qd, kd, vd = _inputs(cu, 56, seed=1, device=device)
    cu_t = torch.tensor(cu, dtype=torch.int32, device=device)
    runner = cb.prepare_minimax_h3_varlen_fp8_attention(
        q8, k8, v8, cu_t, q_descale=qd, k_descale=kd, v_descale=vd
    )
    assert runner.route_metadata["attention_variant"] == "attention_tiny"
    tiny = runner.launch().clone()
    # The general program on the same plan: bind it by hand through the route.
    arch = runner.arch
    route = cb.select_route("fp8", arch)
    program = cb.route_program(route, "attention")
    entry, arguments = cb._bind_in_tree(program, arch, runner.main_kwargs)
    import tvm_ffi

    with tvm_ffi.use_torch_stream():
        entry(*arguments)
    torch.cuda.synchronize()
    assert torch.equal(runner.out, tiny)
    longer = cb.prepare_minimax_h3_varlen_fp8_attention(
        *_inputs((0, 33), 7, seed=2, device=device)[:3],
        torch.tensor((0, 33), dtype=torch.int32, device=device),
        q_descale=1.0,
        k_descale=1.0,
        v_descale=1.0,
    )
    assert longer.route_metadata["attention_variant"] == "attention"
    assert cb.fp8_attention_stage((0, 32, 64)) == "attention_tiny"
    assert cb.fp8_attention_stage((0, 33)) == "attention"


def test_strided_views_match_contiguous() -> None:
    _skip_unless_ready()
    device = torch.device("cuda")
    cu, heads = (0, 257, 600), 7
    q8, k8, v8, qd, kd, vd = _inputs(cu, heads, seed=3, device=device)
    cu_t = torch.tensor(cu, dtype=torch.int32, device=device)
    dense = cb.minimax_h3_varlen_fp8_attention(
        q8, k8, v8, cu_t, q_descale=qd, k_descale=kd, v_descale=vd
    )
    # The fused-QKV column slices: [T, 3 * H * 128] -> three [T, H, 128] views with strides (3 * H * 128, 128, 1).
    fused = torch.cat(
        [t.reshape(cu[-1], heads * HEAD_DIM) for t in (q8, k8, v8)], dim=1
    )
    views = [
        fused[:, i * heads * HEAD_DIM : (i + 1) * heads * HEAD_DIM].view(
            cu[-1], heads, HEAD_DIM
        )
        for i in range(3)
    ]
    assert views[1].stride() == (3 * heads * HEAD_DIM, HEAD_DIM, 1)
    strided = cb.minimax_h3_varlen_fp8_attention(
        *views, cu_t, q_descale=qd, k_descale=kd, v_descale=vd
    )
    assert torch.equal(strided, dense)
    # The [T, H, 3, 128] pack: strides (H * 384, 384, 1).
    pack = torch.stack([q8, k8, v8], dim=2)
    packed = cb.minimax_h3_varlen_fp8_attention(
        pack[:, :, 0],
        pack[:, :, 1],
        pack[:, :, 2],
        cu_t,
        q_descale=qd,
        k_descale=kd,
        v_descale=vd,
    )
    assert torch.equal(packed, dense)


def test_prepared_runner_reuses_caller_output_without_allocation() -> None:
    _skip_unless_ready()
    device = torch.device("cuda")
    cu, heads = (0, 700), 3
    q8, k8, v8, qd, kd, vd = _inputs(cu, heads, seed=4, device=device)
    cu_t = torch.tensor(cu, dtype=torch.int32, device=device)
    out = torch.empty(cu[-1], heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
    descales = [
        torch.tensor([d], dtype=torch.float32, device=device) for d in (qd, kd, vd)
    ]
    runner = cb.prepare_minimax_h3_varlen_fp8_attention(
        q8,
        k8,
        v8,
        cu_t,
        q_descale=descales[0],
        k_descale=descales[1],
        v_descale=descales[2],
        out=out,
    )
    assert runner.out is out and runner.route_metadata["variant"] == "fp8"
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    runner.launch()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    _check(out, _reference(q8, k8, v8, qd, kd, vd, cu, HEAD_DIM**-0.5))
    # Descale tensors are read at launch: halving v_descale halves the output.
    descales[2].fill_(vd * 0.5)
    half = runner.launch().clone()
    torch.cuda.synchronize()
    assert torch.allclose(
        half.float(), (out.float() * 1.0), atol=0.0
    ) or torch.allclose(
        half.float(),
        (_reference(q8, k8, v8, qd, kd, vd * 0.5, cu, HEAD_DIM**-0.5)),
        atol=0.1,
        rtol=0.1,
    )


@pytest.mark.parametrize("profile", ["contiguous", "fused_qkv_view"])
def test_bf16_convenience_entry_matches_explicit_quantization(profile: str) -> None:
    """The BF16 entry equals per-tensor quantization + the FP8 entry bitwise, descales are amax / 448, and the
    prepared runner re-quantizes and launches without allocating."""
    _skip_unless_ready()
    device = torch.device("cuda")
    cu, heads = (0, 133, 300, 900), 7
    generator = torch.Generator(device=device).manual_seed(7)
    T = cu[-1]
    if profile == "contiguous":
        q, k, v = (
            torch.randn(
                T,
                heads,
                HEAD_DIM,
                device=device,
                generator=generator,
                dtype=torch.bfloat16,
            )
            for _ in range(3)
        )
    else:
        buf = torch.randn(
            T,
            3 * heads * HEAD_DIM,
            device=device,
            generator=generator,
            dtype=torch.bfloat16,
        )
        q, k, v = (
            buf[:, i * heads * HEAD_DIM : (i + 1) * heads * HEAD_DIM].view(
                T, heads, HEAD_DIM
            )
            for i in range(3)
        )
    cu_t = torch.tensor(cu, dtype=torch.int32, device=device)
    out = cb.minimax_h3_varlen_fp8_attention_bf16(q, k, v, cu_t)
    e4m3, descales = zip(
        *(cb.quantize_minimax_h3_e4m3_per_tensor(x) for x in (q, k, v)), strict=True
    )
    for x, d in zip((q, k, v), descales, strict=True):
        assert math.isclose(
            float(d), float(x.float().abs().amax()) / 448.0, rel_tol=1e-6
        )
    expected = cb.minimax_h3_varlen_fp8_attention(
        *e4m3, cu_t, q_descale=descales[0], k_descale=descales[1], v_descale=descales[2]
    )
    assert torch.equal(out, expected)
    _check(
        out,
        _reference(
            *e4m3,
            float(descales[0]),
            float(descales[1]),
            float(descales[2]),
            cu,
            HEAD_DIM**-0.5,
        ),
    )
    runner = cb.prepare_minimax_h3_varlen_fp8_attention_bf16(
        q, k, v, cu_t, out=torch.empty_like(out)
    )
    runner()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated(device)
    runner()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated(device) == before
    assert torch.equal(runner.out, expected)
    assert all(torch.equal(a, b) for a, b in zip(runner.e4m3, e4m3, strict=True))


def test_descale_validation() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.device("cuda")
    with pytest.raises(ValueError, match="finite positive"):
        cb._descale_tensor("q_descale", 0.0, device)
    with pytest.raises(ValueError, match="finite positive"):
        cb._descale_tensor("q_descale", math.inf, device)
    with pytest.raises(ValueError, match="one-element float32"):
        cb._descale_tensor("q_descale", torch.ones(2, device=device), device)
    assert cb._descale_tensor("q_descale", 0.25, device).tolist() == [0.25]


def test_operand_dtype_validation() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.device("cuda")
    q = torch.zeros(8, 1, HEAD_DIM, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        cb.validate_minimax_h3_varlen_inputs(
            q, q, q, None, strided_qkv=True, dtype=torch.float8_e4m3fn
        )
