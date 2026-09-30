from __future__ import annotations

import pytest
import torch

from b12x.gemm import bf16_vocab_projection as projection
from b12x.gemm.bf16_vocab_projection._tuning import TUNING
from b12x.preparation import DeviceIdentity, PreparationSession, PreparedCall

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def test_unknown_device_default_uses_selected_torch_backend() -> None:
    query = projection.Bf16VocabProjectionQuery(
        dtype="bfloat16", max_tokens=1, in_features=2_560, out_features=248_320,
    )
    device = DeviceIdentity(
        vendor="nvidia", compute_capability=(9, 0), sm_count=120,
        product_name="Synthetic GPU",
    )

    config = TUNING.configure(query, device=device).default

    assert config.backend == "torch"


@cuda_required
def test_prepared_projection_matches_reference_and_replays_graph() -> None:
    torch.manual_seed(4)
    device = torch.device("cuda")
    source = torch.randn(1, 256, device=device, dtype=torch.bfloat16)
    weight = torch.randn(4_096, 256, device=device, dtype=torch.bfloat16)
    declaration = projection.plan(
        projection.Caps(
            device=device, max_tokens=1, in_features=256, out_features=4_096,
        ),
        override=projection.Bf16VocabProjectionConfig(
            backend="triton", algorithm="row", block_k=256, num_warps=8,
        ),
    )

    def prepare_call(state):
        return PreparedCall(run=lambda: state.run(source, weight))

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((declaration.request(
            name="vocab", prepare_call=prepare_call,
        ),))
        binding = projection.bind(
            declaration, source=source, weight=weight,
        )
        expected = torch.nn.functional.linear(source, weight)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            actual = projection.run(binding)
        actual.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=1e-2)
