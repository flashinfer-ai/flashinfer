"""Explicitly loaded CuTe C++ prefill prototype; never changes public dispatch."""

import os

import pytest
import torch

from benchmarks.sparse_mla_sm120_cute_prefill.evaluate import candidate, ROOT
from tests.attention.dspark_mla_utils import dspark_fixture, numerical_metrics

pytestmark = pytest.mark.skipif(not (ROOT / 'kernel.so').exists(), reason='Build the CuTe C++ prototype first')

@pytest.mark.parametrize('heads', [8, 16, 32, 64])
@pytest.mark.parametrize('rank', [0, 3])
def test_cute_cpp_prefill(heads, rank):
    f = dspark_fixture(2, 'prefill', 'v41_fp8', heads=heads, prefill_tokens=65,
                       prefill_cp_size=4, prefill_cp_rank=rank, history_tokens=128,
                       with_sink=heads != 8)
    run = candidate(f)
    reference_path = os.getenv('CUTE_PREFILL_REFERENCE_MODULE')
    reference = candidate(f, reference_path) if reference_path else None
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    rows = torch.tensor([0,1,2,32,64,65,66,97,129], device='cuda')
    q_before = f['q'].clone()
    for step in range(6):
        for lengths, ids in zip(f['lengths'], f['indices'], strict=True):
            lengths[0] = 0
            lengths[1] = [1,31,32,33,63,65][step]
            lengths[65] = 0
            ids[2, ::7] = -1
            ids[66, :] = -1  # nonempty length with entirely invalid indices
        f['positions'].add_(1)
        f['q'].copy_(q_before * (1.0 if step % 2 else .5))
        graph.replay()
        metrics = numerical_metrics(f, run.lse, rows)
        assert metrics['finite_output'], metrics
        assert metrics['relative_l2'] < .07, metrics
        assert metrics['lse_max_abs'] < .06, metrics
        saved = f['out'].float().clone(), f['scales'].clone(), run.lse.clone()
        if reference is not None:
            reference()
            for actual, expected in zip((f['out'].float(), f['scales'], reference.lse), saved, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        graph.replay()
        for actual, expected in zip((f['out'].float(), f['scales'], run.lse), saved, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_infinite_sink():
    f = dspark_fixture(1, 'prefill', 'v41_fp8', heads=32, prefill_tokens=65,
                       prefill_cp_size=4, history_tokens=128)
    f['sink'][::2] = torch.inf
    f['sink'][1::2] = -torch.inf
    for lengths in f['lengths']:
        lengths[0] = 0
    run = candidate(f)
    run()
    assert (f['out'].float()[:, ::2] == 0).all()
    assert torch.isposinf(run.lse[:, ::2]).all()
    assert torch.isneginf(run.lse[0, 1::2]).all()
    assert (f['out'].float()[0] == 0).all()
    assert torch.isfinite(f['out'].float()).all()
