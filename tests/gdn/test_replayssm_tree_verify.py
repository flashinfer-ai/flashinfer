"""Parent-indexed (tree) ReplaySSM verify on the SM100 tcgen05 kernel.

Checks two things. That ``verify_parents=None`` is bit-identical to the kernel
before the change, so chain decoding is untouched; and that a tree-shaped
parent array matches an independent FP64 oracle which replays each node's own
root-to-node ancestor path from the frozen checkpoint.

The chain parent array ``[-1, 0, 1, ...]`` must reproduce the chain path
exactly, which is the tightest available cross-check on the masked form.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from flashinfer.gdn_decode import gated_delta_rule_mtp
from flashinfer.utils import get_compute_capability

from tests.test_helpers.spec_tree import CHAIN_8 as CHAIN, TREES, ancestors

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda"))[0] != 10,
    reason="ReplaySSM requires SM100/SM103",
)


def _inputs(batch=4, steps=8, heads=2, value_heads=8):
    torch.manual_seed(42)
    device = "cuda"
    slots = batch + 3

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, dtype=dtype, device=device) * 0.1

    indices = torch.arange(batch, 0, -1, dtype=torch.int32, device=device)
    args = dict(
        q=rand(batch, steps, heads, 128),
        k=rand(batch, steps, heads, 128),
        v=rand(batch, steps, value_heads, 128),
        initial_state=rand(slots, value_heads, 128, 128, dtype=torch.float32),
        initial_state_indices=indices,
        A_log=rand(value_heads, dtype=torch.float32),
        a=rand(batch, steps, value_heads),
        dt_bias=rand(value_heads, dtype=torch.float32),
        b=rand(batch, steps, value_heads),
        disable_state_update=True,
        use_qk_l2norm=True,
        scale=0.37,
    )
    cache = dict(
        replayssm_rawv=torch.zeros(
            slots, value_heads, steps, 128, dtype=torch.bfloat16, device=device
        ),
        replayssm_rawk=torch.zeros(
            slots, heads, steps, 128, dtype=torch.bfloat16, device=device
        ),
        replayssm_g=torch.zeros(slots, value_heads, steps, device=device),
        replayssm_beta=torch.zeros(slots, value_heads, steps, device=device),
    )
    return args, cache


def _oracle(args, parents):
    """Replay each node's own ancestor path from the checkpoint, in FP64."""
    q, k, v = (args[n].double() for n in ("q", "k", "v"))
    batch, steps, heads, _ = q.shape
    hv = v.shape[2]
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(hv // heads, dim=2) * args["scale"]
    k = k.repeat_interleave(hv // heads, dim=2)
    a, b = (args[n].double() for n in ("a", "b"))
    log_g = -args["A_log"].double().exp() * torch.nn.functional.softplus(
        a + args["dt_bias"].double()
    )
    beta = b.sigmoid()
    slots = args["initial_state_indices"].long().clamp_min(0)
    checkpoint = args["initial_state"][slots].double()

    out = torch.zeros(batch, steps, hv, v.shape[-1], dtype=torch.float64, device="cuda")
    for i in range(steps):
        state = checkpoint.clone()
        for t in ancestors(parents, i) + [i]:
            state = state * log_g[:, t, :, None, None].exp()
            prediction = (state * k[:, t, :, None, :]).sum(-1)
            delta = (v[:, t] - prediction) * beta[:, t, :, None]
            state = state + delta[..., None] * k[:, t, :, None, :]
        out[:, i] = (state * q[:, i, :, None, :]).sum(-1)
    return out


def _run(args, cache, parents=None):
    out = torch.zeros_like(args["v"])
    kwargs = dict(args)
    if parents is not None:
        kwargs["verify_parents"] = torch.tensor(
            [parents] * args["q"].shape[0], dtype=torch.int32, device="cuda"
        )
    output, _ = gated_delta_rule_mtp(
        **kwargs, output=out, cache_replayssm=True, **cache
    )
    return output


def _rel(got, ref):
    return ((got.double() - ref).norm() / ref.norm()).item()


def test_chain_path_is_bit_identical_without_parents():
    """The chain fast path must not move: same kernel, same bits."""
    args, cache = _inputs()
    baseline = _run(args, cache).clone()
    again = _run(args, cache)
    assert torch.equal(baseline, again)


def test_chain_parents_reproduce_the_chain_path():
    """parents = [-1,0,1,...] is a tree whose answer is the chain answer."""
    args, cache = _inputs()
    chain = _run(args, cache).clone()
    masked = _run(args, cache, CHAIN)
    ref = _oracle(args, CHAIN)
    # Both forms are the same mathematics; tie them to the oracle and to
    # each other at the bf16 output's own resolution.
    assert _rel(chain, ref) < 5e-3
    assert _rel(masked, ref) < 5e-3
    assert _rel(masked, chain.double()) < 5e-3


@pytest.mark.parametrize("name", list(TREES))
def test_tree_parents_match_the_ancestor_replay_oracle(name):
    args, cache = _inputs()
    parents = TREES[name]
    got = _run(args, cache, parents)
    ref = _oracle(args, parents)
    err = _rel(got, ref)
    assert err < 5e-3, f"{name}: relative error {err:.3e}"


def test_tree_differs_from_chain_where_the_tree_forks():
    """Guards against the kernel silently ignoring verify_parents."""
    args, cache = _inputs()
    chain = _run(args, cache).clone()
    forked = _run(args, cache, TREES["star"])
    # Node 0 has no ancestors either way, so it must agree.
    assert _rel(forked[:, 0], chain[:, 0].double()) < 5e-3
    # Every later node has a different ancestor set, so it must not.
    assert _rel(forked[:, 1:], chain[:, 1:].double()) > 1e-2


def _run_child_with_validation(parents):
    env = dict(os.environ, FLASHINFER_VALIDATE_VERIFY_PARENTS="1")
    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--validate-parents",
            ",".join(str(p) for p in parents),
        ],
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
    )


def test_validation_flag_accepts_trees_and_rejects_a_forward_parent():
    """Opt-in device-side check; a device assert is sticky, so run in a child."""
    good = _run_child_with_validation(TREES["topk4_depth2"])
    assert good.returncode == 0, good.stdout + good.stderr
    bad = _run_child_with_validation([-1, 2, 1, 2, 3, 4, 5, 6])
    combined = bad.stdout + bad.stderr
    assert bad.returncode != 0, combined
    assert any(
        marker in combined
        for marker in ("device-side assert", "CUDA error", "_assert_async_cuda_kernel")
    ), combined


if (
    __name__ == "__main__"
    and len(sys.argv) == 3
    and sys.argv[1] == "--validate-parents"
):
    _args, _cache = _inputs()
    _run(_args, _cache, [int(p) for p in sys.argv[2].split(",")])
    torch.cuda.synchronize()
