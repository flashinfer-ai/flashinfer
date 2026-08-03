"""Per-token ``num_valid_pages`` for ``msa_topk_select``.

Callers whose query tokens have differing causal KV extents (decode batches,
chunked prefill) previously had to recover per-token semantics on the host with
a scatter + mask + sort. These tests pin the in-kernel behaviour that replaces
that: per-token clamping, a per-token trailing local window, and the output
contract (ascending, ``-1`` tail-padded).
"""

import pytest
import torch

from flashinfer.msa_ops import msa_topk_select
from flashinfer.utils import is_sm12x_supported

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")),
    reason="msa_ops requires SM120/SM121",
)

TOPK = 16


def _distinct_scores(H, P, S, seed):
    """A permutation, so no two blocks tie: count-rank and radix agree only on
    distinct inputs, and these tests compare across that dispatch boundary."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    flat = torch.randperm(H * P * S, generator=g, device="cuda").to(torch.float32)
    return flat.reshape(H, P, S).contiguous()


@pytest.mark.parametrize("P", [24, 160])
@pytest.mark.parametrize("fb,fe", [(0, 0), (0, 1), (1, 1), (2, 3)])
def test_per_token_matches_scalar_per_token(P, fb, fe):
    """A per-token tensor must equal invoking the scalar path once per token.

    At P=160 the tensor call dispatches to radix (bounded by max_k_tiles) while
    most single-token reference calls dispatch to count-rank, so this also
    cross-checks the two kernels against each other.
    """
    H, S = 2, 12
    max_score = _distinct_scores(H, P, S, seed=7 + P + fb * 10 + fe)
    g = torch.Generator(device="cuda").manual_seed(3)
    nvp = torch.randint(
        max(fb + fe, 1), P + 1, (S,), generator=g, device="cuda", dtype=torch.int32
    )

    got = msa_topk_select(
        max_score, TOPK, num_valid_pages=nvp, force_begin_blocks=fb, force_end_blocks=fe
    )
    assert got.shape == (S, H, TOPK)

    for s in range(S):
        ref = msa_topk_select(
            max_score[:, :, s : s + 1].contiguous(),
            TOPK,
            num_valid_pages=int(nvp[s]),
            force_begin_blocks=fb,
            force_end_blocks=fe,
        )
        torch.testing.assert_close(got[s : s + 1], ref, msg=f"token {s} differs")


@pytest.mark.parametrize("P", [24, 160])
def test_per_token_output_contract(P):
    """Every token, including ones shorter than the forced region, must yield
    ascending in-range indices with ``-1`` only as a tail pad."""
    H, S = 2, 64
    fb, fe = 1, 2
    max_score = _distinct_scores(H, P, S, seed=11 + P)
    # Deliberately include extents below fb+fe: the scalar API rejects those,
    # so only the in-kernel clamp can handle them.
    g = torch.Generator(device="cuda").manual_seed(5)
    nvp = torch.randint(1, P + 1, (S,), generator=g, device="cuda", dtype=torch.int32)

    out = msa_topk_select(
        max_score, TOPK, num_valid_pages=nvp, force_begin_blocks=fb, force_end_blocks=fe
    ).cpu()
    nvp_c = nvp.cpu()

    for s in range(S):
        n = int(nvp_c[s])
        for h in range(H):
            row = out[s, h]
            valid = row[row >= 0]
            assert len(valid) == min(TOPK, n), f"token {s} head {h}: {row.tolist()}"
            assert (valid < n).all(), f"token {s} head {h} exceeds extent {n}"
            assert (valid[1:] > valid[:-1]).all(), f"token {s} head {h} not ascending"
            # -1 must be a pure tail pad, never a hole.
            assert (row[len(valid) :] == -1).all(), f"token {s} head {h} mid-array -1"


def test_per_token_forces_each_tokens_own_local_window():
    """``force_end_blocks`` must denote each token's own trailing blocks, which
    is the property a batch-wide scalar cannot express."""
    H, S, P = 1, 8, 40
    fe = 2
    max_score = _distinct_scores(H, P, S, seed=23)
    nvp = torch.arange(4, 4 + S, device="cuda", dtype=torch.int32)

    out = msa_topk_select(
        max_score, TOPK, num_valid_pages=nvp, force_begin_blocks=0, force_end_blocks=fe
    ).cpu()

    for s in range(S):
        n = int(nvp[s])
        sel = set(out[s, 0].tolist())
        for j in range(n - fe, n):
            assert j in sel, f"token {s}: local block {j} not forced (extent {n})"


def test_per_token_input_guards():
    H, P, S = 2, 32, 4
    max_score = torch.randn(H, P, S, device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="1D int32"):
        msa_topk_select(
            max_score,
            TOPK,
            num_valid_pages=torch.ones(S, device="cuda", dtype=torch.int64),
        )
    with pytest.raises(ValueError, match="total_qo_len"):
        msa_topk_select(
            max_score,
            TOPK,
            num_valid_pages=torch.ones(S + 1, device="cuda", dtype=torch.int32),
        )


def test_per_token_is_cuda_graph_safe():
    """The whole point of moving these bounds in-kernel is to stop the caller
    doing host-side repair; that only pays off if the call captures."""
    H, P, S = 2, 32, 8
    max_score = _distinct_scores(H, P, S, seed=31)
    nvp = torch.randint(1, P + 1, (S,), device="cuda", dtype=torch.int32)
    out = torch.empty((S, H, TOPK), dtype=torch.int32, device="cuda")

    def run():
        msa_topk_select(
            max_score,
            TOPK,
            num_valid_pages=nvp,
            force_begin_blocks=0,
            force_end_blocks=1,
            output=out,
        )

    run()  # warm the compile cache outside capture
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            run()
    torch.cuda.current_stream().wait_stream(s)

    eager = msa_topk_select(
        max_score, TOPK, num_valid_pages=nvp, force_begin_blocks=0, force_end_blocks=1
    )
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, eager)
