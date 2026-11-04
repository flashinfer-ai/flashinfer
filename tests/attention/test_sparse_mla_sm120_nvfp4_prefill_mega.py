"""Native FP4 prefill specializations through the FlashInfer execution binding."""

import math

import pytest
import torch

from flashinfer.jit.mla import gen_sparse_mla_sm120_module
from tests.attention.dspark_mla_utils import dspark_fixture, numerical_metrics


def prefill_calls(f, module=None):
    module = module or gen_sparse_mla_sm120_module().build_and_load()
    q, caches, indices, lengths = f["q"], f["caches"], f["indices"], f["lengths"]
    device = torch.cuda.get_device_properties(q.device)
    plan = module.resolve_dsv4_nvfp4(
        q.shape[0],
        q.shape[1],
        indices[0].shape[1],
        indices[1].shape[1],
        caches[0].shape[1],
        caches[1].shape[1],
        caches[0].stride(0),
        caches[1].stride(0),
        0,
        device.multi_processor_count,
        device.shared_memory_per_block_optin,
        True,
        False,
        lengths[0] is not None,
        lengths[1] is not None,
        f["sink"] is not None,
    )
    lse = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
    mid, mlse = f["context"], f["context"].view(torch.float32)

    def arguments(query):
        return (
            plan,
            query,
            caches[0],
            indices[0],
            mid,
            mlse,
            f["context"],
            lse,
            512**-0.5,
            lengths[0],
            f["sink"],
            caches[1],
            indices[1],
            lengths[1],
            math.log(2),
        )

    def mega():
        module.dsv4_nvfp4_execute_mega_attention(
            *arguments(q),
            f["positions"],
            f["cs"],
            f["out"].view(torch.uint8),
            f["scales"],
        )
        return lse

    def baseline():
        f["prologue"]()
        module.dsv4_nvfp4_execute_attention(*arguments(f["rotated"]))
        f["post"]()
        return lse

    return mega, baseline, lse


def quantized_query_reference_lse(f, rows):
    # The native numeric route quantizes Q's 448 non-RoPE features to NVFP4.
    # Account for that specified rounding before testing attention arithmetic.
    from tests.attention.mega_mla_utils import rope_reference
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
    f = dspark_fixture(
        2,
        "prefill",
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
        # A one-candidate row has raw-Q LSE error .2156 in both the original
        # and this kernel. Do not hide that quantization error by raising .20;
        # check the native quantized-Q arithmetic against its own FP64 reference.
        expected_lse = quantized_query_reference_lse(f, rows)
        torch.testing.assert_close(lse[rows].double(), expected_lse, atol=0.02, rtol=0)
        fused = f["out"].float().clone(), f["scales"].clone(), lse.clone()
        baseline()
        for actual, expected in zip(
            (f["out"].float(), f["scales"], lse), fused, strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nvfp4_prefill_infinite_sink():
    f = dspark_fixture(
        1,
        "prefill",
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
