"""Single-rank training value-forward, snapshot refresh and graph replay contract.

The trainer reuses a layer across optimizer steps. Rollout independently loads
the current parameter snapshot into a fresh layer with identical kernel settings.
Backward is supplied by a differentiable training forward through recomputation;
it is not a derivative check for the FlashInfer quantized operator.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest
import torch

pytestmark = pytest.mark.arch_blackwell


@pytest.fixture(autouse=True)
def single_rank(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100/SM103")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank contract test")
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        pytest.skip("requires a single-rank process group")
    pytest.importorskip("flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe")
    monkeypatch.setenv("MEGA_NO_DIST", "1")


def _bits(tensor):
    return tensor.detach().contiguous().view(torch.uint8)


def _assert_bits(left, right):
    assert torch.isfinite(left).all()
    assert torch.isfinite(right).all()
    assert torch.equal(_bits(left), _bits(right))


def _config(backend):
    from flashinfer.moe_ep import (
        FinalizeMode,
        Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig,
        Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
    )

    common = dict(
        intermediate_size=1024,
        top_k=2,
        fast_math=False,
        enable_in_kernel_fc2_reduce=False,
        finalize_mode=FinalizeMode.DEFAULT,
        apply_routing_weights_before_fc2=True,
    )
    if backend == "bf16":
        return Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
            **common, knobs={"in_kernel_fc2_reduce": False}
        )
    if backend == "mxfp8_genphase":
        return Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
            **common,
            kernel_variant="genphase",
            mma_tiler_mnk=(256, 128, 128),
            cluster_shape_mn=(4, 1),
            schedule_policy=("grouped", None),
        )
    return Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        **common,
        kernel_name=(
            "sm100_mxfp8_mxfp8_bf16_cutedsl_next"
            if backend == "mxfp8_next"
            else "sm100_mxfp8_mxfp8_bf16_cutedsl"
        ),
        knobs={"in_kernel_fc2_reduce": False},
    )


def _training_recompute(x, w13, w2, ids, probabilities):
    """Small differentiable trainer used only for the backward contract."""
    output = torch.zeros_like(x)
    for expert in range(w13.shape[0]):
        gate, up = torch.nn.functional.linear(x, w13[expert]).chunk(2, dim=-1)
        rows = torch.nn.functional.linear(
            torch.nn.functional.silu(gate) * up, w2[expert]
        )
        weight = ((ids == expert) * probabilities).sum(-1).to(x.dtype)
        output = output + rows * weight.unsqueeze(-1)
    return output


@pytest.mark.parametrize("backend", ["bf16", "mxfp8", "mxfp8_next", "mxfp8_genphase"])
def test_training_forward_optimizer_refit_and_captured_weights(backend):
    from torch.utils.checkpoint import checkpoint

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEEpTensors,
        MoEWeightPack,
        preprocess_bf16_cutedsl_mega_weights,
        preprocess_mxfp8_cutedsl_mega_weights,
    )

    gen = torch.Generator(device="cuda").manual_seed(7936)
    w13 = torch.nn.Parameter(
        (
            torch.randn(4, 2048, 2048, device="cuda", generator=gen) / 2048**0.5
        ).bfloat16()
    )
    w2 = torch.nn.Parameter(
        (
            torch.randn(4, 2048, 1024, device="cuda", generator=gen) / 1024**0.5
        ).bfloat16()
    )
    source = MoEWeightPack(w13, w2)
    x = torch.randn(32, 2048, device="cuda", generator=gen).bfloat16().requires_grad_()
    ids = (torch.arange(64, device="cuda").reshape(32, 2) % 4).long()
    probabilities = (
        torch.tensor([0.375, 0.625], device="cuda").expand(32, 2).contiguous()
    ).requires_grad_()
    batch = MoEEpTensors(x, ids, probabilities)
    layers = []
    graph = None

    def make_layer(weights, *, prepared=None):
        layer = MoEEpMegaLayer(
            BootstrapConfig(world_size=1, rank=0, device=torch.cuda.current_device()),
            FleetParams(num_experts=4, max_tokens_per_rank=64, token_hidden_size=2048),
            weights,
            MegaConfig(
                megakernel=_config(backend),
                preprocess_weights=prepared is None,
                transformed_weights=prepared,
            ),
        )
        layers.append(layer)
        return layer

    def loaded_rollout():
        # An independent checkpoint/load path, not the trainer's prepared tensors.
        checkpoint_weights = MoEWeightPack(w13.detach().clone(), w2.detach().clone())
        prepared = None
        if backend == "bf16":
            prepared = preprocess_bf16_cutedsl_mega_weights(
                checkpoint_weights, hidden_size=2048, intermediate_size=1024
            )
        elif backend != "mxfp8_next":
            prepared = preprocess_mxfp8_cutedsl_mega_weights(
                checkpoint_weights,
                hidden_size=2048,
                intermediate_size=1024,
                kernel_variant=(
                    "genphase" if backend == "mxfp8_genphase" else "inference"
                ),
            )
        rollout = make_layer(checkpoint_weights, prepared=prepared)
        with torch.no_grad():
            rollout.warmup(batch)
            return rollout(batch).clone()

    try:
        trainer = make_layer(source)
        with torch.no_grad():
            trainer.warmup(batch)
        expected = loaded_rollout()

        def train_forward(hidden):
            if torch.is_grad_enabled():
                return _training_recompute(hidden, w13, w2, ids, probabilities)
            return trainer(MoEEpTensors(hidden, ids, probabilities))

        training_output = checkpoint(train_forward, x, use_reentrant=True)
        _assert_bits(training_output, expected)
        training_output.float().square().mean().backward()
        assert batch.hidden_states is x and batch.topk_weights is probabilities
        for value in (x, w13, w2, probabilities):
            assert value.grad is not None
            assert torch.isfinite(value.grad).all()
            assert torch.count_nonzero(value.grad) > 0

        tensors = [t for pair in trainer._transformed for t in pair if t is not None]
        addresses = [t.data_ptr() for t in tensors]
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            graph_output = trainer(batch)
        graph_address = graph_output.data_ptr()

        optimizer = torch.optim.SGD([w13, w2], lr=0.125)
        # A representable update avoids a vacuous equality after BF16 rounding.
        w13.grad = torch.ones_like(w13)
        w2.grad = torch.ones_like(w2)
        optimizer.step()
        with torch.no_grad():
            _assert_bits(trainer(batch), expected)
        graph.replay()
        _assert_bits(graph_output, expected)

        for refit in (False, True):
            if refit:
                with torch.no_grad():
                    w13.mul_(0.5)
                    w2.mul_(-0.5)
            expected_new = loaded_rollout()
            assert not torch.equal(_bits(expected_new), _bits(expected))
            trainer.refresh_weights(source)
            assert [t.data_ptr() for t in tensors] == addresses
            with torch.no_grad():
                _assert_bits(trainer(batch), expected_new)
            graph.replay()
            _assert_bits(graph_output, expected_new)
            assert graph_output.data_ptr() == graph_address
            expected = expected_new
    finally:
        torch.cuda.synchronize()
        if graph is not None:
            graph.reset()
        for layer in reversed(layers):
            layer.destroy()


@pytest.mark.parametrize("kernel_layout", [False, True])
@pytest.mark.parametrize("kind", ["mxfp8_e4m3", "mxfp8_e5m2"])
def test_prequantized_snapshot_refresh_preserves_payload(kind, kernel_layout):
    from flashinfer.moe_ep import MoEWeightPack, preprocess_mxfp8_cutedsl_mega_weights

    gen = torch.Generator(device="cuda").manual_seed(7937)
    dtype = torch.float8_e4m3fn if kind == "mxfp8_e4m3" else torch.float8_e5m2
    w13 = torch.randn(2, 256, 256, device="cuda", generator=gen).to(dtype)
    w2 = torch.randn(2, 256, 128, device="cuda", generator=gen).to(dtype)
    if kernel_layout:
        w13 = w13.transpose(1, 2).contiguous()
        w2 = w2.transpose(1, 2).contiguous()
    # Packing preserves all payload bits, including signed zero and NaN encodings.
    w13.view(torch.uint8).flatten()[:5].copy_(
        torch.tensor([0, 128, 1, 127, 255], device="cuda", dtype=torch.uint8)
    )
    sf13 = torch.full((2, 256, 8), 127, device="cuda", dtype=torch.uint8)
    sf2 = torch.full((2, 256, 4), 127, device="cuda", dtype=torch.uint8)
    source = MoEWeightPack(w13, w2, sf13, sf2)

    def prepare(*, out=None):
        return preprocess_mxfp8_cutedsl_mega_weights(
            source, hidden_size=256, intermediate_size=128, kind=kind, out=out
        )

    with mock.patch(
        "flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp8_bf16_cutedsl."
        "weights._quantize_mxfp8_weight_k_major",
        side_effect=AssertionError("prequantized weights must not be requantized"),
    ):
        prepared = prepare()
        tensors = [t for pair in prepared for t in pair]
        saved = [(_bits(t).clone(), t.data_ptr()) for t in tensors]
        w13.view(torch.uint8).bitwise_xor_(1)
        w2.view(torch.uint8).bitwise_xor_(1)
        sf13.add_(1)
        sf2.add_(1)
        for t, (old, _) in zip(tensors, saved, strict=True):
            assert torch.equal(_bits(t), old)
        assert prepare(out=prepared) is prepared
        fresh = [t for pair in prepare() for t in pair]
        for t, ref, (_, pointer) in zip(tensors, fresh, saved, strict=True):
            assert torch.equal(_bits(t), _bits(ref))
            assert t.data_ptr() == pointer

    if kernel_layout:
        assert torch.equal(_bits(prepared[0][0]), _bits(w13))
        assert torch.equal(_bits(prepared[1][0]), _bits(w2))
    else:
        interleaved = prepared[0][0].transpose(1, 2).reshape(2, 4, 2, 32, 256)
        gate = interleaved[:, :, 0].reshape(2, 128, 256)
        up = interleaved[:, :, 1].reshape(2, 128, 256)
        assert torch.equal(torch.cat((_bits(gate), _bits(up)), dim=1), _bits(w13))
        assert torch.equal(_bits(prepared[1][0].transpose(1, 2)), _bits(w2))
