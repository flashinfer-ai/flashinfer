"""Parent-indexed (tree) verify on the KDA WY output-only kernel.

KDA's decay is per-channel and it solves all T corrections through a T x T
triangular inverse rather than a loop, so the tree case is not obviously the
same problem as GDN's. It is: the cumulative gate becomes a scan over each
token's root-to-node path, and the two triangular masks become ancestor
masks. Nothing else changes -- the per-channel operand scaling already carries
exp(cum_t - cum_s) into khat @ ktil^T, so no weight matrix is needed.

Checked against an FP32 oracle that replays each node's own ancestor path from
the frozen checkpoint, plus the tightest available cross-check: the chain
parent array must reproduce the untouched prefix path.
"""

import pytest
import torch
import torch.nn.functional as F

from flashinfer.utils import is_sm100a_supported
from flashinfer.kda_decode import _KDA_OUTPUT_ONLY_AVAILABLE

if _KDA_OUTPUT_ONLY_AVAILABLE:
    from flashinfer.kda_kernels.kda_decode_wy_output_only import kda_wy_output_only
else:
    kda_wy_output_only = None

CHAIN = [-1, 0, 1, 2, 3, 4, 5, 6]
TREES = {
    "chain": CHAIN,
    "fork_at_root": [-1, 0, 0, 1, 2, 3, 4, 5],
    "topk4_depth2": [-1, 0, 0, 0, 0, 1, 1, 2],
    "lopsided": [-1, 0, 1, 1, 3, 3, 5, 2],
    "star": [-1, 0, 0, 0, 0, 0, 0, 0],
    "topk2_depth4": [-1, 0, 0, 1, 1, 2, 2, 3],
}


@pytest.fixture(autouse=True)
def _require_kda():
    if not torch.cuda.is_available():
        pytest.skip("KDA output-only decode requires CUDA")
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("KDA output-only decode requires SM100a (Blackwell)")
    if not _KDA_OUTPUT_ONLY_AVAILABLE:
        pytest.skip("kda_wy_output_only unavailable (missing cutlass DSL deps)")


def _inputs(B=2, T=8, H=2, HV=4, K=128, V=128):
    torch.manual_seed(11)
    dev = "cuda"
    l2n = lambda x: F.normalize(x.float(), dim=-1).to(torch.bfloat16)
    pool = B + 2
    return dict(
        q=l2n(torch.randn(B, T, H, K, device=dev)),
        k=l2n(torch.randn(B, T, H, K, device=dev)),
        v=(torch.randn(B, T, HV, V, device=dev) * 0.1).to(torch.bfloat16),
        # log-space per-channel gate, strictly negative as the real gate is
        g=(-5.0 * torch.rand(B, T, HV, K, device=dev)).to(torch.bfloat16),
        beta=(torch.rand(B, T, HV, device=dev) * 0.8 + 0.1).to(torch.bfloat16),
        h0=(torch.randn(pool, HV, V, K, device=dev) * 0.1).to(torch.bfloat16),
        idx=torch.arange(B, dtype=torch.int32, device=dev),
        scale=0.37,
    )


def _ancestors(parents, i):
    out, cur = [], parents[i]
    while cur >= 0:
        out.append(cur)
        cur = parents[cur]
    return out[::-1]


def _oracle(a, parents):
    """Replay each node's own root-to-node path from the checkpoint."""
    q, k, v = a["q"].float() * a["scale"], a["k"].float(), a["v"].float()
    g, beta = a["g"].float(), a["beta"].float()
    B, T, H, _ = q.shape
    HV = v.shape[2]
    rep = HV // H
    out = torch.zeros(B, T, HV, v.shape[3], device=q.device)
    for b in range(B):
        ckpt = a["h0"][a["idx"][b]].float()
        for i in range(T):
            S = ckpt.clone()
            for t in _ancestors(parents, i) + [i]:
                S = S * g[b, t].exp()[:, None, :]
                k_hv = k[b, t].repeat_interleave(rep, dim=0)
                u = torch.einsum("hvk,hk->hv", S, k_hv)
                w = beta[b, t][:, None] * (v[b, t] - u)
                S = S + torch.einsum("hv,hk->hvk", w, k_hv)
            q_hv = q[b, i].repeat_interleave(rep, dim=0)
            out[b, i] = torch.einsum("hvk,hk->hv", S, q_hv)
    return out


def _run(a, parents=None):
    kw = {}
    if parents is not None:
        kw["verify_parents"] = torch.tensor(
            [parents] * a["q"].shape[0], dtype=torch.int32, device="cuda"
        )
    return kda_wy_output_only(
        a["q"],
        a["k"],
        a["v"],
        a["g"],
        a["beta"],
        a["h0"],
        initial_state_indices=a["idx"],
        scale=a["scale"],
        backend="wy",
        **kw,
    )


def _rel(got, ref):
    return ((got.float() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("name", list(TREES))
def test_tree_parents_match_the_ancestor_replay_oracle(name):
    a = _inputs()
    parents = TREES[name]
    err = _rel(_run(a, parents), _oracle(a, parents))
    assert err < 2e-2, f"{name}: relative error {err:.3e}"


def test_chain_parents_reproduce_the_prefix_path():
    """parents=[-1,0,1,...] is a tree whose answer is the prefix answer."""
    a = _inputs()
    prefix = _run(a).float()
    masked = _run(a, CHAIN).float()
    assert _rel(masked, prefix) < 2e-2


def test_tree_differs_from_chain_where_the_tree_forks():
    """Guards against the kernel silently ignoring verify_parents."""
    a = _inputs()
    prefix = _run(a).float()
    forked = _run(a, TREES["star"]).float()
    # Node 0 has no ancestors either way, so it must agree.
    assert _rel(forked[:, 0], prefix[:, 0]) < 2e-2
    # Later nodes have different ancestor sets, so they must not.
    assert _rel(forked[:, 1:], prefix[:, 1:]) > 1e-1


def test_rejects_malformed_parents():
    a = _inputs()
    with pytest.raises(ValueError, match="int32"):
        _run_bad(a, torch.zeros(2, 8, dtype=torch.int64, device="cuda"))
    with pytest.raises(ValueError, match="shape"):
        _run_bad(a, torch.zeros(2, 4, dtype=torch.int32, device="cuda"))


def _run_bad(a, parents):
    return kda_wy_output_only(
        a["q"],
        a["k"],
        a["v"],
        a["g"],
        a["beta"],
        a["h0"],
        initial_state_indices=a["idx"],
        scale=a["scale"],
        backend="wy",
        verify_parents=parents,
    )
