"""Native FP4 prefill specializations through the FlashInfer execution binding."""

import pytest
import torch

from tests.attention.sparse_mla_prefill_utils import (
    native_prefill_calls as prefill_calls,
)
from tests.attention.sparse_mla_prefill_utils import prefill_fixture, numerical_metrics


def quantized_query_reference_lse(f, rows):
    # The native numeric route quantizes Q's 448 non-RoPE features to NVFP4.
    # Account for that specified rounding before testing attention arithmetic.
    from tests.attention.sparse_mla_prefill_utils import rope_reference
    from tests.attention.sparse_mla_test_utils import _dequantize_nvfp4_query

    q = rope_reference(f["q"][rows], f["positions"][rows], f["cs"]).bfloat16()
    q = _dequantize_nvfp4_query(q).double()
    selected, valid = [], []
    for decoded, ids, lengths in zip(
        f["decoded"], f["indices"], f["lengths"], strict=True
    ):
        ix = ids[rows]
        selected.append(decoded[ix.clamp_min(0).long()].double())
        valid.append(
            (ix >= 0)
            & (torch.arange(ix.shape[1], device=ix.device)[None] < lengths[rows, None])
        )
    scores = (q @ torch.cat(selected, 1).transpose(1, 2)) * 512**-0.5
    scores.masked_fill_(~torch.cat(valid, 1)[:, None], -torch.inf)
    lse = torch.logsumexp(scores, -1)
    if f["sink"] is not None:
        lse = torch.logaddexp(lse, f["sink"].double()[None])
    return lse


@pytest.mark.parametrize("heads", [16, 32, 64])
@pytest.mark.parametrize("rank", [0, 3])
def test_nvfp4_prefill_mega(heads, rank):
    f = prefill_fixture(
        2,
        "nvfp4",
        heads=heads,
        prefill_tokens=65,
        prefill_cp_size=4,
        prefill_cp_rank=rank,
        history_tokens=128,
        with_sink=heads != 16,
    )
    mega, baseline, lse = prefill_calls(f)
    mega()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mega()
    rows = torch.tensor([0, 1, 2, 32, 64, 65, 66, 97, 129], device="cuda")
    for length in (0, 1, 31, 32, 33, 63, 65):
        for lens, ids in zip(f["lengths"], f["indices"], strict=True):
            lens[0] = 0
            lens[1] = length
            ids[2, ::7] = -1
            ids[66, :] = -1
        f["positions"].add_(1)
        graph.replay()
        metrics = numerical_metrics(f, lse, rows)
        assert metrics["finite_output"]
        # Native NVFP4 Q/K arithmetic uses the existing NVFP4 accuracy gates.
        assert metrics["relative_l2"] < 0.20, metrics
        # LSE uses quantized Q operands; check against the corresponding FP64 math.
        expected_lse = quantized_query_reference_lse(f, rows)
        torch.testing.assert_close(lse[rows].double(), expected_lse, atol=0.02, rtol=0)
        fused = f["out"].float().clone(), f["scales"].clone(), lse.clone()
        baseline()
        for actual, expected in zip(
            (f["out"].float(), f["scales"], lse), fused, strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nvfp4_prefill_infinite_sink():
    f = prefill_fixture(
        1,
        "nvfp4",
        heads=32,
        prefill_tokens=65,
        prefill_cp_size=4,
        history_tokens=128,
    )
    f["sink"][::2] = torch.inf
    f["sink"][1::2] = -torch.inf
    for lens in f["lengths"]:
        lens[0] = 0
    mega, baseline, lse = prefill_calls(f)
    for run in (mega, baseline):
        run()
        assert (f["out"].float()[:, ::2] == 0).all()
        assert torch.isposinf(lse[:, ::2]).all()
        assert torch.isneginf(lse[0, 1::2]).all()
        assert (f["out"].float()[0] == 0).all()
