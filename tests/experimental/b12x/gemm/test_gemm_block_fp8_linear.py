from __future__ import annotations

import pytest
import torch

from b12x.gemm import block_fp8_linear as bfl
from contextlib import contextmanager

from b12x.preparation import Plan, PreparationSession, PreparedCall
from b12x.testing.reference.helpers import require_b12x


def test_block_fp8_plan_is_declarative() -> None:
    declaration = bfl.plan(
        bfl.Caps(
            device="cpu",
            max_tokens=8,
            in_features=256,
            out_features=384,
            output_dtype=torch.bfloat16,
        )
    )
    assert isinstance(declaration, Plan)
    assert declaration.query.max_tokens == 8
    assert declaration.query.in_features == 256
    assert declaration.query.out_features == 384


def test_block_fp8_runtime_rejects_invalid_plan() -> None:
    source = torch.empty((1, 128), dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="requires a Plan"):
        bfl.run(source, object(), plan=object())


def _make_block_fp8_weight(
    out_features: int,
    in_features: int,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = (
        torch.randn((out_features, in_features), device="cuda", dtype=torch.bfloat16)
        / 8
    ).to(torch.float8_e4m3fn)
    if block_size == 32:
        n_blocks = (out_features + 31) // 32
        k_blocks = in_features // 32
        scale_u8 = (
            (
                torch.arange(n_blocks, device="cuda")[:, None]
                + 2 * torch.arange(k_blocks, device="cuda")[None, :]
            )
            % 7
            + 123
        ).to(torch.uint8)
        return weight, scale_u8.view(torch.float8_e8m0fnu)
    scale_u8 = (
        torch.arange(
            (out_features // 128) * (in_features // 128),
            device="cuda",
            dtype=torch.int32,
        )
        % 3
        + 126
    ).to(torch.uint8)
    scale = scale_u8.view(torch.float8_e8m0fnu).reshape(
        out_features // 128,
        in_features // 128,
    )
    return weight, scale


def _v41_dequantized_operands(x, weight, scale):
    from tests.experimental.b12x.gemm.test_fp8_quant_deepgemm_parity import (
        _per_token_cast_to_fp8,
    )

    values, scales = _per_token_cast_to_fp8(x, gran_k=32)
    x_deq = values.float() * scales.repeat_interleave(32, dim=1)
    w_deq = (
        weight.float()
        * (
            scale.float()
            .repeat_interleave(32, dim=0)
            .repeat_interleave(32, dim=1)[: weight.shape[0], : weight.shape[1]]
        )
    )
    return x_deq, w_deq


def _assert_v41_accumulation_matches_reference(
    source, weight, scale, actual, *, atomic_slices=1
):
    x_deq, w_deq = _v41_dequantized_operands(source, weight, scale)
    a, b = x_deq.double(), w_deq.double()
    exact = a @ b.T
    absolute_products = a.abs() @ b.abs().T
    # Native MXF8 MMA and the source K32 GEMM both accumulate in FP32, not
    # FP64. Their summation orders need not match, particularly when tiny
    # K32 groups precede nearly cancelling larger groups. The standard dot
    # product bound is gamma_K * sum(abs(a_i*b_i)), u = 2**-24. UE8M0
    # scaling is an exact power-of-two operation for these finite operands.
    k_u = source.shape[-1] * 2.0**-24
    accumulation_error = (k_u / (1.0 - k_u)) * absolute_products
    if atomic_slices > 1:
        # Atomic split-K rounds each partial and every accumulated BF16 sum.
        # Bound those extra roundings using the sum of absolute exact partials.
        partial_k = source.shape[-1] // atomic_slices
        absolute_partials = sum(
            (a[:, start : start + partial_k] @ b[:, start : start + partial_k].T).abs()
            for start in range(0, source.shape[-1], partial_k)
        )
        rounding_u = 2.0**-8
        operations = atomic_slices + 1
        gamma = operations * rounding_u / (1.0 - operations * rounding_u)
        accumulation_error = (
            1.0 + gamma
        ) * accumulation_error + gamma * absolute_partials
    # Final BF16/FP16 round-to-nearest contributes at most half the local ULP.
    actual64 = actual.double()
    below = torch.nextafter(actual, torch.full_like(actual, -float("inf"))).double()
    above = torch.nextafter(actual, torch.full_like(actual, float("inf"))).double()
    rounding_error = 0.5 * torch.maximum(actual64 - below, above - actual64)
    assert torch.isfinite(actual).all()
    assert torch.all((actual64 - exact).abs() <= accumulation_error + rounding_error)


def _reference_from_quantized_operands(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    if block_size == 32:
        x_deq, w_deq = _v41_dequantized_operands(x, weight, scale)
    else:
        from tests.experimental.b12x.gemm.test_fp8_quant_deepgemm_parity import (
            _per_token_cast_to_fp8,
        )

        values, scales = _per_token_cast_to_fp8(x, gran_k=32)
        x_deq = values.float() * scales.repeat_interleave(32, dim=1)
        w_deq = weight.float() * scale.float().repeat_interleave(
            128, dim=0
        ).repeat_interleave(128, dim=1)
    reference = x_deq.double() @ w_deq.double().T
    rounded = reference.to(x.dtype)
    # Torch's half constructors convert through FP32; correct double rounding.
    for direction in (-float("inf"), float("inf")):
        adjacent = torch.nextafter(rounded, torch.full_like(rounded, direction))
        closer = (reference - adjacent.double()).abs() < (
            reference - rounded.double()
        ).abs()
        rounded = torch.where(closer, adjacent, rounded)
    return rounded


@contextmanager
def _prepared(source, packed, *, capacity=None, functional=False, config=None):
    capacity = source.shape[0] if capacity is None else capacity
    caps = bfl.Caps(
        device=source.device,
        max_tokens=capacity,
        in_features=source.shape[1],
        out_features=packed.out_features,
        source_dtype=source.dtype,
        output_dtype=source.dtype,
        block_size=packed.block_size,
        output_mode="functional" if functional else "provided",
    )
    plan = bfl.plan(caps, override=config)

    def prepare(state):
        if functional:
            return PreparedCall(run=lambda: state.run(source, packed))
        (spec,) = state.scratch.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=source.device)
        output = torch.empty(
            (source.shape[0], packed.out_features, 1),
            dtype=source.dtype,
            device=source.device,
        )
        binding = state.bind(
            scratch=scratch, source=source, packed_weight=packed, output=output
        )
        return PreparedCall(
            run=lambda: state.run_binding(binding),
            output=output,
            owners=(scratch, binding),
        )

    with PreparationSession(
        device=source.device, autotune=False, compile_workers=2
    ) as session:
        session.prepare((plan.request(name="block-fp8", prepare_call=prepare),))
        session.freeze()
        yield plan


def _storage(plan, source, packed):
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=source.device)
    output = torch.empty(
        (source.shape[0], packed.out_features, 1),
        dtype=source.dtype,
        device=source.device,
    )
    binding = bfl.bind(
        plan, scratch=scratch, source=source, packed_weight=packed, output=output
    )
    return binding, scratch, output


@pytest.mark.parametrize("tokens", [8, 9, 127, 128, 129])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_block_fp8_padding_poison_and_graph(tokens, dtype):
    require_b12x()
    torch.manual_seed(20260901)
    source = torch.randn(tokens, 256, device="cuda", dtype=dtype).mul_(0.25)
    weight, scale = _make_block_fp8_weight(384, 256)
    packed = bfl.pack_weight(weight, scale)
    expected = _reference_from_quantized_operands(source, weight, scale)
    with _prepared(source, packed) as plan:
        binding, scratch, output = _storage(plan, source, packed)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            bfl.run(binding=binding)
        for poison in (0, 255):
            scratch.fill_(poison)
            output.fill_(float("nan"))
            graph.replay()
            torch.testing.assert_close(output[:, :, 0], expected, rtol=0, atol=0)
        graph.reset()


@pytest.mark.parametrize("tokens", [8, 9, 127, 128, 129])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_block_fp8_functional_padding_poison(tokens, dtype, monkeypatch):
    from b12x.gemm._shared import block_fp8 as impl

    require_b12x()
    torch.manual_seed(20260902)
    source = torch.randn(tokens, 256, device="cuda", dtype=dtype).mul_(0.25)
    weight, scale = _make_block_fp8_weight(384, 256)
    packed = bfl.pack_weight(weight, scale)
    expected = _reference_from_quantized_operands(source, weight, scale)
    allocate = impl.empty_mxfp8_rows_bases
    poison = 0

    def poisoned(*args, **kwargs):
        bases = allocate(*args, **kwargs)
        if not kwargs.get("initialize_scales", True):
            bases[1].fill_(poison)
            bases[2].fill_(poison)
        return bases

    monkeypatch.setattr(impl, "empty_mxfp8_rows_bases", poisoned)
    with _prepared(source, packed, functional=True) as plan:
        first = bfl.run(source, packed, plan=plan)
        poison = 255
        second = bfl.run(source, packed, plan=plan)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(first, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_block_fp8_public_quantizer_preserves_scale_padding(dtype):
    require_b12x()
    source = torch.randn(129, 256, device="cuda", dtype=dtype).mul_(0.25)
    weight, scale = _make_block_fp8_weight(384, 256)
    with _prepared(source, bfl.pack_weight(weight, scale)) as plan:
        rows = bfl.quantize_input(source, plan=plan)
        physical = (
            rows.scale_mma.view(torch.uint8)
            .permute(5, 2, 1, 0, 4, 3)
            .reshape(1, 256, 8)
        )
        torch.testing.assert_close(physical[:, :129], rows.scale_rows.view(torch.uint8))
        assert torch.all(physical[:, 129:] == 127)


@pytest.mark.parametrize(
    "capacity,k,n,counts",
    [
        (1, 128, 256, (1,)),
        (8, 256, 384, (1, 2, 4, 8)),
        (4096, 128, 1536, (1824,)),
        (512, 1024, 8192, (16, 32, 128)),
        (4096, 1024, 16384, (16,)),
    ],
)
def test_block_fp8_planned_capacity_replays_live_rows(capacity, k, n, counts):
    require_b12x()
    torch.manual_seed(20260528)
    source = torch.randn(capacity, k, device="cuda", dtype=torch.bfloat16).mul_(0.25)
    weight, scale = _make_block_fp8_weight(n, k)
    packed = bfl.pack_weight(weight, scale)
    with _prepared(source, packed) as plan:
        _, scratch, output = _storage(plan, source, packed)
        pointers = source.data_ptr(), scratch.data_ptr(), output.data_ptr()
        for rows in counts:
            binding = bfl.bind(
                plan,
                scratch=scratch,
                source=source[:rows],
                packed_weight=packed,
                output=output[:rows],
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                bfl.run(binding=binding)
            source.neg_()
            output.fill_(float("nan"))
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            assert pointers == (
                source.data_ptr(),
                scratch.data_ptr(),
                output.data_ptr(),
            )
            expected = _reference_from_quantized_operands(source[:rows], weight, scale)
            torch.testing.assert_close(
                output[:rows, :, 0], expected, rtol=0.01, atol=1e-4
            )
            assert torch.isnan(output[rows:]).all()
            graph.reset()


@pytest.mark.parametrize(
    "tokens,in_features,out_features",
    [
        (1, 5120, 1280),
        (8, 5120, 512),
        (9, 256, 96),
        (129, 256, 160),
        (33, 1280, 8192),
        (7, 1024, 5120),
        (9, 6144, 25600),
    ],
)
def test_block_fp8_linear_v41_independent_k32_n32_scales(
    tokens: int,
    in_features: int,
    out_features: int,
) -> None:
    """Neither K128 replication nor N128 sharing may corrupt V4.1 scales."""
    require_b12x()
    torch.manual_seed(20260910)
    source = torch.randn(
        (tokens, in_features),
        device="cuda",
        dtype=torch.bfloat16,
    ).mul_(0.25)
    # Include tiny groups: unfloored legacy activation quantization differs here.
    source[:, :32].mul_(1e-5)
    weight, scale = _make_block_fp8_weight(out_features, in_features, 32)
    packed = bfl.pack_weight(weight, scale, block_size=(32, 32))
    torch.testing.assert_close(
        packed.weight.values.view(torch.uint8),
        weight.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    from tests.experimental.b12x.gemm.test_fp8_quant_deepgemm_parity import (
        _per_token_cast_to_fp8,
        _sf_fp32_to_e8m0_u8,
    )

    x_values, x_scales = _per_token_cast_to_fp8(source, 32)
    with _prepared(source, packed, functional=True) as plan:
        x_q = bfl.quantize_input(source, plan=plan)
        actual = bfl.run(source, packed, plan=plan)
    torch.testing.assert_close(
        x_q.values.view(torch.uint8),
        x_values.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        x_q.scale_rows.view(torch.uint8)[0],
        _sf_fp32_to_e8m0_u8(x_scales),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        packed.weight.scale_rows.view(torch.uint8)[0],
        scale.view(torch.uint8).repeat_interleave(32, dim=0)[:out_features],
        rtol=0,
        atol=0,
    )
    _assert_v41_accumulation_matches_reference(source, weight, scale, actual)


def test_block_fp8_linear_v41_rejects_lossy_weight_scale_repacking() -> None:
    require_b12x()
    weight = torch.ones((64, 128), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.full((2, 4), 0.3, device="cuda")
    with pytest.raises(ValueError, match="exact UE8M0"):
        bfl.pack_weight(weight, scales, block_size=(32, 32))


@torch.no_grad()
def test_shared_declarations_bind_distinct_weights_and_replay_after_alias_release():
    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    source = torch.randn(8, 256, dtype=torch.bfloat16, device=device) / 4
    weights = [_make_block_fp8_weight(384, 256, block_size=32) for _ in range(2)]
    packed = [
        bfl.pack_weight(weight, scale, block_size=(32, 32)) for weight, scale in weights
    ]
    caps = bfl.Caps(
        device=device,
        max_tokens=8,
        in_features=256,
        out_features=384,
        output_dtype=torch.bfloat16,
        block_size=(32, 32),
    )
    plans = [bfl.plan(caps) for _ in range(2)]
    primed = []

    def prepare(state):
        primed.append(state)
        (spec,) = state.scratch.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        output = torch.empty((8, 384, 1), dtype=torch.bfloat16, device=device)
        binding = state.bind(
            scratch=scratch, source=source, packed_weight=packed[0], output=output
        )
        return PreparedCall(run=lambda: state.run_binding(binding), output=output)

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        progress = []
        session.prepare(
            tuple(
                plan.request(name=f"layer.{index}", prepare_call=prepare)
                for index, plan in enumerate(plans)
            ),
            progress=progress.append,
        )
        assert len(primed) == 1
        assert progress[-1].total_requests == progress[-1].completed_requests == 1
        assert plans[0].prepared is plans[1].prepared
        storage = [
            _storage(plan, source, weight) for plan, weight in zip(plans, packed)
        ]
        for (binding, scratch, output), (weight, scale) in zip(storage, weights):
            bfl.run(binding=binding)
            _assert_v41_accumulation_matches_reference(
                source, weight, scale, output[:, :, 0]
            )
            assert torch.count_nonzero(output) > 0
        assert not torch.equal(storage[0][2], storage[1][2])
        session.release(plans[0])
        session.freeze()
        binding, scratch, output = storage[1]
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture(), torch.cuda.graph(graph):
                bfl.run(binding=binding)
            source.neg_()
            scratch.fill_(255)
            output.fill_(float("nan"))
            pointer = output.data_ptr()
            allocated = torch.cuda.memory_allocated(device)
            graph.replay()
            torch.cuda.synchronize(device)
            assert (
                output.data_ptr() == pointer
                and torch.cuda.memory_allocated(device) == allocated
            )
            _assert_v41_accumulation_matches_reference(
                source, *weights[1], output[:, :, 0]
            )
            assert torch.count_nonzero(output) > 0
        finally:
            graph.reset()
