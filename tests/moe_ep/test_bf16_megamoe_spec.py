"""Device conformance of the pinned BF16 MegaMoE arithmetic specification."""

from __future__ import annotations

import math
from itertools import product

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Float32, T, dsl_user_op
import pytest

from flashinfer.batch_invariant.fp_interface import exact
from flashinfer.moe_ep.batch_invariant.device import PackedF32, ScalarF32
from flashinfer.moe_ep.batch_invariant.spec import (
    fc1_activation,
    fc1_activation_accumulator,
)

from .test_bf16_megamoe_batch_invariant import _assert_bits, _require_cuda

pytestmark = pytest.mark.gpu


@dsl_user_op
def _reference_atom(which: str, x: Float32, *, loc=None, ip=None):
    # Deliberately independent of the production backend: the exact CPU
    # oracle samples only the two architecture-defined approximate functions.
    opcode = "ex2.approx.ftz.f32" if which == "ex2" else "rcp.approx.ftz.f32"
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [x.ir_value(loc=loc, ip=ip)],
            opcode + " $0, $1;",
            "=f,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


class _AtomSampler:
    def __init__(self, which):
        self.which = which

    @cute.jit
    def __call__(self, x: cute.Tensor, y: cute.Tensor, stream):
        self.kernel(x, y).launch(
            grid=(cute.ceil_div(x.shape[0], 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, x: cute.Tensor, y: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 128 + tidx
        if i < x.shape[0]:
            y[i] = _reference_atom(self.which, x[i])


class _Activation:
    def __init__(self, packed, weighted, clamp):
        self.packed, self.weighted, self.clamp = packed, weighted, clamp

    @cute.jit
    def __call__(
        self,
        g: cute.Tensor,
        u: cute.Tensor,
        p: cute.Tensor,
        a: cute.Tensor,
        h: cute.Tensor,
        stream,
    ):
        self.kernel(g, u, p, a, h).launch(
            grid=(cute.ceil_div(g.shape[0], 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        g: cute.Tensor,
        u: cute.Tensor,
        p: cute.Tensor,
        a: cute.Tensor,
        h: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 128 + tidx
        if cutlass.const_expr(self.packed):
            i = i * 2
            if i + 1 < g.shape[0]:
                probability = None
                if cutlass.const_expr(self.weighted):
                    probability = (p[i], p[i + 1])
                accum = fc1_activation_accumulator(
                    PackedF32,
                    (g[i], g[i + 1]),
                    (u[i], u[i + 1]),
                    probability,
                    self.clamp,
                )
                handoff = fc1_activation(
                    PackedF32,
                    (g[i], g[i + 1]),
                    (u[i], u[i + 1]),
                    probability,
                    self.clamp,
                )
                a[i], a[i + 1] = accum
                h[i], h[i + 1] = handoff
        else:
            if i < g.shape[0]:
                probability = None
                if cutlass.const_expr(self.weighted):
                    probability = p[i]
                a[i] = fc1_activation_accumulator(
                    ScalarF32, g[i], u[i], probability, self.clamp
                )
                h[i] = fc1_activation(ScalarF32, g[i], u[i], probability, self.clamp)


def _launch(kernel, *tensors):
    import torch
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack

    args = [from_dlpack(t) for t in tensors]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(kernel, *args, stream)
    compiled(*args, stream)
    torch.cuda.synchronize()


def _sample(which, values):
    import torch

    x = torch.tensor(values, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)
    _launch(_AtomSampler(which), x, y)
    return y.cpu().tolist()


def _activation_oracle(gates, ups, weights, clamp):
    # The association and rounding here are intentionally written independently
    # of spec.fc1_activation. Only the opaque approximate atoms run on device.
    inputs = []
    if clamp is not None:
        clamp = abs(clamp)

    def minimum(a, b):
        if math.isnan(a) or math.isnan(b):
            return float("nan")
        if a == b == 0:
            return -0.0 if math.copysign(1, a) < 0 or math.copysign(1, b) < 0 else 0.0
        return min(a, b)

    def maximum(a, b):
        if math.isnan(a) or math.isnan(b):
            return float("nan")
        if a == b == 0:
            return -0.0 if math.copysign(1, a) < 0 and math.copysign(1, b) < 0 else 0.0
        return max(a, b)

    for gate, up in zip(gates, ups, strict=True):
        if clamp is not None:
            gate = minimum(gate, clamp)
            up = maximum(minimum(up, clamp), -clamp)
        inputs.append((gate, up))
    exponents = [exact.mul(g, -1.4426950216293335) for g, _ in inputs]
    exponentials = _sample("ex2", exponents)
    denominators = [exact.add(e, 1.0) for e in exponentials]
    reciprocals = _sample("rcp", denominators)
    result = []
    for i, ((gate, up), sigmoid) in enumerate(zip(inputs, reciprocals, strict=True)):
        value = exact.mul(exact.mul(up, gate), sigmoid)
        if weights is not None:
            value = exact.mul(value, weights[i])
        result.append(value)
    return result


def _assert_numeric_bits(actual, values, *, bf16=False):
    import torch

    expected = torch.tensor(values, dtype=torch.float32)
    if bf16:
        expected = torch.tensor(
            [exact.to_format(v, exact.BF16) for v in values], dtype=torch.bfloat16
        )
    expected = expected.to(actual.device)
    # exact.py models NaN values, not architecture-specific NaN payloads.
    # Packed/scalar payload agreement is separately checked without masking.
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    valid = ~torch.isnan(expected)
    integer = torch.int16 if bf16 else torch.int32
    assert torch.equal(actual[valid].view(integer), expected[valid].view(integer))


@pytest.mark.parametrize("clamp", (None, 1.25, 0.0, -0.0, float("inf"), float("nan")))
@pytest.mark.parametrize("weighted", (False, True))
def test_fc1_chain_matches_independent_exact_model(clamp, weighted):
    _require_cuda()
    import torch

    tiny = 2.0**-149
    gates = [
        0.0,
        -0.0,
        tiny,
        -tiny,
        2.0**-126,
        -(2.0**-126),
        0.125,
        -0.125,
        1.25,
        -1.25,
        80.0,
        -80.0,
        90.0,
        -90.0,
        float("inf"),
        float("-inf"),
        float("nan"),
        3.4028234663852886e38,
    ]
    generator = torch.Generator().manual_seed(704)
    gates += torch.randn(78, generator=generator).mul_(4).tolist()
    g = torch.tensor(gates, dtype=torch.float32, device="cuda")
    u = torch.randn(g.numel(), generator=generator, dtype=torch.float32).to("cuda")
    p = torch.randn(g.numel(), generator=generator, dtype=torch.float32).to("cuda")
    u[:6] = torch.tensor([0.0, -0.0, 1.0, -1.0, tiny, -tiny], device="cuda")
    p[:4] = torch.tensor([0.0, -0.0, 1.0, -1.0], device="cuda")
    expected = _activation_oracle(
        g.cpu().tolist(),
        u.cpu().tolist(),
        p.cpu().tolist() if weighted else None,
        clamp,
    )
    results = []
    for packed in (False, True):
        a = torch.empty_like(g)
        handoff = torch.empty_like(g, dtype=torch.bfloat16)
        _launch(_Activation(packed, weighted, clamp), g, u, p, a, handoff)
        _assert_numeric_bits(a, expected)
        _assert_numeric_bits(handoff, expected, bf16=True)
        results.append((a, handoff))
    assert torch.equal(results[0][0].view(torch.int32), results[1][0].view(torch.int32))
    _assert_bits(results[0][1], results[1][1], "scalar/packed activation")


@pytest.mark.parametrize("weight_before_fc2", (False, True))
def test_finalize_matches_ordered_exact_fold(weight_before_fc2):
    _require_cuda()
    import torch
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack

    import flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe  # noqa: F401
    from moe_nvfp4_swapab.topk_reduce import TopkReduce
    from src.token_comm import CombineFormat

    generator = torch.Generator(device="cuda").manual_seed(172)
    terms = torch.randn(
        4, 3, 256, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    # Three differently scaled contributions expose reassociation and the
    # distinction between FMA and a rounded multiply followed by addition.
    terms[:, 0] *= 256
    terms[:, 2] *= 0.00390625
    probabilities = torch.randn(4, 3, device="cuda", generator=generator)
    # BF16 subnormals are exact FP32 inputs; the finalizer must not silently
    # flush them or change the sign rules of a zero-seeded fold.
    tiny = 2.0**-133
    terms[0, :, :4] = torch.tensor(
        [
            [tiny, tiny, -tiny, -0.0],
            [-tiny, tiny, tiny, 0.0],
            [tiny, -tiny, tiny, -0.0],
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    probabilities[0] = torch.tensor([1.0, 128.0 / 129.0, 1.0], device="cuda")
    # The middle product rounds to 1 if rounded separately, but its FMA with
    # -1 retains a nonzero residual. The adjacent columns distinguish FP32
    # from BF16 accumulation and fixed slot order from a reassociated tree.
    terms[0, :, -3:] = torch.tensor(
        [[2.0**24, 256.0, -1.0], [1.0, 1.0, 1.0078125], [-(2.0**24), -256.0, 0.0]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    ids = torch.tensor(
        [[0, 1, 2], [-1, 1, 2], [0, -1, 2], [-1, -1, -1]],
        device="cuda",
        dtype=torch.int64,
    )
    terms[ids < 0] = float("nan")
    probabilities[ids < 0] = float("nan")
    output = torch.empty(4, 256, dtype=torch.bfloat16, device="cuda")
    score = None if weight_before_fc2 else probabilities
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    major, minor = torch.cuda.get_device_capability()
    reducer = TopkReduce(
        256,
        3,
        CombineFormat.parse("bf16"),
        sm_arch=f"sm_{major}{minor}a",
        num_experts=4,
    )
    args = (
        from_dlpack(terms),
        None,
        from_dlpack(output),
        None if score is None else from_dlpack(score),
        stream,
    )
    compiled = cute.compile(reducer, *args, topk_indices=from_dlpack(ids))
    compiled(*args, topk_indices=from_dlpack(ids))
    values, weights, routes = (
        terms.cpu().float().tolist(),
        probabilities.cpu().tolist(),
        ids.cpu().tolist(),
    )
    expected = []
    for row in range(4):
        for column in range(256):
            accumulator = 0.0
            for slot in range(3):
                valid = routes[row][slot] >= 0
                term = values[row][slot][column] if valid else 0.0
                weight = weights[row][slot] if valid else 0.0
                accumulator = (
                    exact.add(accumulator, term)
                    if weight_before_fc2
                    else exact.fma(term, weight, accumulator)
                )
            expected.append(exact.to_format(accumulator, exact.BF16))
    expected = torch.tensor(expected, dtype=torch.bfloat16, device="cuda").view_as(
        output
    )
    _assert_bits(output, expected, "ordered finalize exact reference")


@pytest.mark.parametrize("weight_before_fc2", (False, True))
@pytest.mark.parametrize("clamp", (1.25, -1.25, -0.0))
@pytest.mark.parametrize("tile", tuple(product((128, 256), (128, 256), (64, 128, 256))))
def test_whole_kernel_matches_independent_dense_reference(
    weight_before_fc2, clamp, tile, monkeypatch
):
    _require_cuda()
    import torch

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.batch_invariant import fingerprint
    from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_bf16_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.bf16_staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        bf16_mega_moe,
        compute_megamoe_reference_bf16,
        get_symm_buffer_for_bf16_mega_moe,
    )

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    # K256 needs the same 256-wide canonical padding in both GEMMs. H224
    # also admits the single-stage M128/N256/K256 configuration in SMEM.
    hidden, intermediate = (224, 256) if tile[2] == 256 else (96, 128)
    experts, topk, tokens = 4, 3, 17
    generator = torch.Generator(device="cuda").manual_seed(184)
    x = torch.randn(
        tokens, hidden, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    w13 = (
        torch.randn(
            experts,
            2 * intermediate,
            hidden,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.05
    )
    w2 = (
        torch.randn(
            experts,
            hidden,
            intermediate,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.05
    )
    weights = torch.rand(tokens, topk, device="cuda", generator=generator)
    ids = (
        torch.tensor([0, 1, 2], device="cuda", dtype=torch.int64)
        .expand(tokens, topk)
        .contiguous()
    )
    transformed = preprocess_mega_weights(
        MoEWeightPack(w13=w13, w2=w2),
        intermediate_size=intermediate,
        hidden_size=hidden,
    )
    # The dense reference has independent gather/scatter, a 1-CTA GEMM and
    # scalar SwiGLU. Clamp bounds its exponent away from subnormal outputs,
    # where that older reference's non-FTZ exponential would be a different atom.
    partials = (
        compute_megamoe_reference_bf16(
            input_activation=x.unsqueeze(0),
            input_topk_idx=ids.unsqueeze(0),
            input_topk_weights=weights.unsqueeze(0),
            fc1_weight=transformed[0][0].unsqueeze(0),
            fc2_weight=transformed[1][0].unsqueeze(0),
            ref_compute_graph="deepgemm" if weight_before_fc2 else "transformers",
            gate_up_clamp=abs(clamp),
            apply_routing_weights_before_fc2=weight_before_fc2,
        )[0]
        .cpu()
        .float()
        .tolist()
    )
    weight_values = weights.cpu().tolist()
    expected = []
    for token in range(tokens):
        for column in range(hidden):
            value = 0.0
            for slot in range(topk):
                term = partials[token][slot][column]
                value = (
                    exact.add(value, term)
                    if weight_before_fc2
                    else exact.fma(term, weight_values[token][slot], value)
                )
            expected.append(exact.to_format(value, exact.BF16))
    expected = torch.tensor(expected, dtype=torch.bfloat16, device="cuda").view(
        tokens, hidden
    )
    buffer = get_symm_buffer_for_bf16_mega_moe(
        experts,
        129,
        topk,
        hidden,
        intermediate,
        0,
        1,
        batch_invariant=True,
        gate_up_clamp=clamp,
        apply_routing_weights_before_fc2=weight_before_fc2,
        knobs={"mma_tiler_mnk": tile},
    )
    try:
        receipt = fingerprint(
            "megamoe_bf16",
            hidden=hidden,
            intermediate=intermediate,
            top_k=topk,
            num_experts=experts,
            apply_routing_weights_before_fc2=weight_before_fc2,
            gate_up_clamp=clamp,
        )
        assert buffer._frontend.batch_invariant_fingerprint == receipt
        assert receipt == fingerprint(
            "megamoe_bf16",
            hidden=hidden,
            intermediate=intermediate,
            top_k=topk,
            num_experts=experts,
            apply_routing_weights_before_fc2=weight_before_fc2,
            gate_up_clamp=abs(clamp),
        )
        stage_mega_moe_inputs(
            x, weights, ids, buffer.x, buffer.topk_idx, buffer.topk_weights
        )
        output = torch.empty_like(expected)
        bf16_mega_moe(output, *transformed, buffer, num_tokens=tokens)
        _assert_bits(output, expected, "independent dense-reference whole kernel")
    finally:
        torch.cuda.synchronize()
        buffer.destroy()
