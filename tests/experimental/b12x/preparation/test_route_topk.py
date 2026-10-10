"""Prepared top-k routing retains the compiled Triton launch ABI."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from b12x.moe._shared import routing


@dataclass(frozen=True)
class _TensorMetadata:
    shape: tuple[int, ...]
    dtype: torch.dtype
    strides: tuple[int, ...]
    device: torch.device = torch.device("cuda", 0)
    is_cuda: bool = True

    @property
    def ndim(self):
        return len(self.shape)

    def stride(self, axis):
        return self.strides[axis]


class _CompiledLauncher:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def run(*args, stream=None):
            self.calls.append((grid, args, stream))

        return run


@pytest.mark.parametrize("score_func", ["softmax", "sqrtsoftplus"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_prepared_route_uses_complete_positional_abi(score_func, with_bias):
    logits = _TensorMetadata((7, 384), torch.bfloat16, (400, 1))
    selected = _TensorMetadata((7, 6), torch.float32, (9, 1))
    ids = _TensorMetadata((7, 6), torch.int32, (8, 1))
    weights = _TensorMetadata((7, 6), torch.float32, (10, 1))
    bias = _TensorMetadata((384,), torch.float32, (1,)) if with_bias else None
    image_bias = _TensorMetadata((384,), torch.float32, (1,)) if with_bias else None
    mask = _TensorMetadata((7,), torch.bool, (1,)) if with_bias else None
    launcher = _CompiledLauncher()
    routing.route_topk(
        logits,
        selected,
        ids,
        weights,
        renormalize=True,
        score_func=score_func,
        correction_bias=bias,
        image_correction_bias=image_bias,
        image_mask=mask,
        routed_scaling_factor=1.5,
        _launcher=launcher,
    )
    assert launcher.calls == [
        (
            (7, 1, 1),
            (
                logits,
                selected,
                ids,
                weights,
                bias,
                image_bias,
                mask,
                400,
                9,
                8,
                10,
                int(with_bias),
                int(with_bias),
                int(with_bias),
                1.5,
                384,
                512,
                6,
                8,
                True,
                score_func,
                with_bias,
                with_bias,
            ),
            None,
        )
    ]


def test_unprepared_route_preserves_jit_arguments(monkeypatch):
    calls = []

    class JitLauncher:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append((grid, args, kwargs))

    monkeypatch.setattr(routing, "_route_topk_kernel", JitLauncher())
    logits = _TensorMetadata((1, 128), torch.float32, (128, 1))
    selected = _TensorMetadata((1, 2), torch.float32, (2, 1))
    ids = _TensorMetadata((1, 2), torch.int32, (2, 1))
    weights = _TensorMetadata((1, 2), torch.float32, (2, 1))
    routing.route_topk(logits, selected, ids, weights, renormalize=False)
    grid, args, options = calls[0]
    assert grid == (1,)
    assert len(args) == 16 and args[:4] == (logits, selected, ids, weights)
    assert options == {
        "BLOCK_E": 128,
        "TOP_K": 2,
        "TOPK_BLOCK": 2,
        "RENORMALIZE": False,
        "SCORE_FUNC": "softmax",
        "HAS_CORRECTION_BIAS": False,
        "HAS_IMAGE_BIAS": False,
        "num_warps": 4,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compile_workers", [0, 2])
@pytest.mark.parametrize("score_func", ["softmax", "sqrtsoftplus"])
@pytest.mark.parametrize("compile_forward", [False, True])
def test_public_prepared_route_graph_replays_inputs_and_modality(
    compile_workers,
    score_func,
    compile_forward,
):
    from b12x.moe import fused_moe
    from b12x.preparation import PreparationSession, PreparedCall

    device = torch.device("cuda", 0)
    logits = torch.zeros((7, 400), dtype=torch.bfloat16, device=device)[:, :384]
    selected = torch.empty((7, 9), device=device)[:, :6]
    ids = torch.empty((7, 8), dtype=torch.int32, device=device)[:, :6]
    weights = torch.empty((7, 10), device=device)[:, :6]
    text_bias = torch.zeros(384, device=device)
    text_bias[:6] = 10
    image_bias = text_bias.flip(0)
    mask = torch.zeros(7, dtype=torch.bool, device=device)
    options = dict(
        correction_bias=text_bias, image_correction_bias=image_bias, image_mask=mask
    )
    invocation = fused_moe.RouteTopKInvocation(
        num_tokens=7,
        num_experts=384,
        top_k=6,
        logits_dtype="bfloat16",
        score_func=score_func,
        renormalize=True,
        routed_scaling_factor=1.5,
        has_correction_bias=True,
        has_image_correction_bias=True,
        has_image_mask=True,
        logits_row_stride=400,
        topk_row_stride=9,
        ids_row_stride=8,
        weights_row_stride=10,
    )
    plan = fused_moe.plan_route_topk(invocation)
    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    with PreparationSession(
        device=device, autotune=False, compile_workers=compile_workers
    ) as session:
        session.prepare(
            (
                plan.request(
                    name="strided_route",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(
                            logits, selected, ids, weights, **options
                        ),
                        owners=(
                            logits,
                            selected,
                            ids,
                            weights,
                            text_bias,
                            image_bias,
                            mask,
                        ),
                    ),
                ),
            )
        )
        stream.wait_stream(torch.cuda.current_stream(device))

        def forward():
            fused_moe.route_topk(plan, logits, selected, ids, weights, **options)

        run = (
            torch.compile(forward, fullgraph=True, dynamic=False)
            if compile_forward
            else forward
        )
        try:
            with torch.cuda.stream(stream):
                run()
            stream.synchronize()
            with session.capture(), torch.cuda.graph(graph, stream=stream):
                run()
            session.freeze()
            for iteration in range(2):
                values = (
                    torch.linspace(-6 + iteration, 6 + iteration, 384)
                    .bfloat16()
                    .repeat(7, 1)
                )
                values[0].zero_()
                image_rows = torch.arange(7).remainder(2) == iteration
                logits.copy_(values)
                mask.copy_(image_rows)
                selected.fill_(float("nan"))
                ids.fill_(-1)
                weights.fill_(float("nan"))
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                with torch.cuda.stream(stream):
                    graph.replay()
                stream.synchronize()
                after = torch.cuda.memory_stats(device)
                assert (
                    before["allocation.all.allocated"]
                    == after["allocation.all.allocated"]
                )
                scores = (
                    values.float().softmax(-1)
                    if score_func == "softmax"
                    else torch.nn.functional.softplus(values.float()).sqrt()
                )
                selection = scores + torch.where(
                    image_rows[:, None], image_bias.cpu(), text_bias.cpu()
                )
                expected_ids = (
                    383
                    - selection.flip(1).argsort(dim=1, descending=True, stable=True)[
                        :, :6
                    ]
                )
                expected_logits = values.float().gather(1, expected_ids)
                expected_weights = (
                    expected_logits.softmax(-1)
                    if score_func == "softmax"
                    else scores.gather(1, expected_ids)
                )
                if score_func == "sqrtsoftplus":
                    expected_weights /= expected_weights.sum(1, keepdim=True) + 1e-20
                expected_weights *= 1.5
                assert torch.equal(ids.cpu(), expected_ids.int())
                assert torch.equal(selected.cpu(), expected_logits)
                torch.testing.assert_close(
                    weights.cpu(), expected_weights, rtol=2e-6, atol=1e-7
                )
        finally:
            stream.synchronize()
            graph.reset()
