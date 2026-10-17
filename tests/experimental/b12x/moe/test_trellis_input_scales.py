"""Input gain broadcasting against explicit gate/up scalar products."""

import math

import pytest
import torch

from b12x.moe.fused_moe.config import ScaleGranularity, TrellisScaleFactorsConfig
from b12x.moe.fused_moe.trellis import _effective_input_scales
from b12x.moe.fused_moe.weights import ScaleFactors


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("per_expert", [False, True])
@pytest.mark.parametrize("projection_vectors", [False, True])
@pytest.mark.parametrize(
    "gain_granularity,gain_values",
    [
        (ScaleGranularity.NONE, None),
        (ScaleGranularity.UNIFORM, 2.0),
        (ScaleGranularity.UNIFORM, [2.0]),
        (ScaleGranularity.UNIFORM, [2.0, 3.0]),
        (ScaleGranularity.PER_LAYER, [2.0, 3.0]),
        (ScaleGranularity.PER_EXPERT, [1.0, 2.0, 4.0]),
        (ScaleGranularity.PER_EXPERT, [[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]]),
    ],
)
def test_input_gain_broadcasting(
    device, per_expert, projection_vectors, gain_granularity, gain_values
):
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        if torch.cuda.get_device_capability()[0] != 12:
            pytest.skip("SM12x required")
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if device == "cuda"
        else torch.device("cpu")
    )
    experts, hidden = 3, 8
    shape = ((experts,) if per_expert else ()) + (
        (2, hidden) if projection_vectors else (hidden,)
    )
    vectors = torch.arange(
        1, 1 + math.prod(shape), device=device, dtype=torch.float16
    ).reshape(shape)
    gains = (
        None
        if gain_values is None
        else torch.tensor(gain_values, device=device, dtype=torch.float16)
    )
    declaration = TrellisScaleFactorsConfig(
        vectors=(
            ScaleGranularity.PER_EXPERT if per_expert else ScaleGranularity.PER_LAYER
        ),
        gains=gain_granularity,
    )
    gate, up = _effective_input_scales(
        ScaleFactors(vectors=vectors, gains=gains),
        declaration,
        num_experts=experts,
        hidden_size=hidden,
        device=device,
    )
    gain_per_expert = gain_granularity is ScaleGranularity.PER_EXPERT
    rows = experts if per_expert or gain_per_expert else 1
    expected = torch.empty((rows, 2, hidden), dtype=torch.float16, device=device)
    for expert in range(rows):
        for projection in range(2):
            vector = vectors[expert] if per_expert else vectors
            if projection_vectors:
                vector = vector[projection]
            gain = gains[expert] if gain_per_expert else gains
            if gain is not None and gain.ndim:
                gain = gain[projection if gain.numel() == 2 else 0]
            expected[expert, projection] = vector if gain is None else vector * gain
    assert torch.equal(gate, expected[:, 0])
    assert torch.equal(up, expected[:, 1])
    assert gate.is_contiguous() and up.is_contiguous()
