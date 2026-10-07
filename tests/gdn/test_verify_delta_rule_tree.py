# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parent-indexed (tree) verify on the GDN reference.

``verify_delta_rule`` replays a speculative draft as a prefix: token ``t``
sees tokens ``0..t-1``. A tree draft gives each token its own root-to-node
path instead. Without ``parents`` the reference cannot say so and silently
returns the chain answer for tree-shaped input; these tests pin the
parent-indexed form against an independent FP64 per-node replay and check
that the chain topology leaves the prefix path bit-identical.

Pure torch, no kernel, so it runs on every CI lane including CPU-only ones.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.test_helpers.spec_tree import (
    CHAIN_8,
    TREES,
    ancestors,
    batch_parents,
    chain_parents,
)

from .reference_delta_rule import verify_delta_rule

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _inputs(device, batch=2, steps=8, heads=2, value_heads=4, dim=32):
    torch.manual_seed(7)

    def rand(*shape):
        return torch.randn(*shape, dtype=torch.float32, device=device) * 0.5

    return dict(
        q=rand(batch, steps, heads, dim),
        k=rand(batch, steps, heads, dim),
        v=rand(batch, steps, value_heads, dim),
        state=rand(batch, value_heads, dim, dim),
        A_log=rand(value_heads),
        a=rand(batch, steps, value_heads),
        dt_bias=rand(value_heads),
        b=rand(batch, steps, value_heads),
        scale_factor=0.37,
    )


def _oracle(args, parents):
    """Replay each token's own ancestor path from the initial state, in FP64."""
    q, k, v = (args[n].double() for n in ("q", "k", "v"))
    state0 = args["state"].double()
    batch, steps, _, _ = q.shape
    value_heads = v.shape[2]
    rep = value_heads // q.shape[2]
    q = F.normalize(q, dim=-1).repeat_interleave(rep, dim=2) * args["scale_factor"]
    k = F.normalize(k, dim=-1).repeat_interleave(rep, dim=2)
    g = torch.exp(
        -args["A_log"].double().exp()
        * F.softplus(args["a"].double() + args["dt_bias"].double(), threshold=20.0)
    )
    beta = args["b"].double().sigmoid()
    parents = torch.as_tensor(parents).reshape(-1, steps)
    if parents.shape[0] == 1:
        parents = parents.expand(batch, steps)

    out = torch.zeros(batch, steps, value_heads, v.shape[-1], dtype=torch.float64)
    states = torch.zeros(batch, steps, *state0.shape[1:], dtype=torch.float64)
    for row in range(batch):
        for i in range(steps):
            s = state0[row].clone()
            for t in ancestors(parents[row].tolist(), i) + [i]:
                s = s * g[row, t, :, None, None]
                err = v[row, t] - torch.einsum("hk,hkv->hv", k[row, t], s)
                s = (
                    s
                    + k[row, t, :, :, None] * (err * beta[row, t, :, None])[:, None, :]
                )
            out[row, i] = torch.einsum("hk,hkv->hv", q[row, i], s)
            states[row, i] = s
    return out.to(q.device), states.to(q.device)


def _rel(got, ref):
    return ((got.double() - ref.double()).norm() / ref.double().norm()).item()


def _run(args, parents=None, cache=False):
    return verify_delta_rule(**args, cache_intermediate_states=cache, parents=parents)


@pytest.mark.parametrize("device", DEVICES)
def test_chain_parents_are_bit_identical_to_the_prefix_path(device):
    args = _inputs(device)
    out, state, inter = _run(args, cache=True)
    out_p, state_p, inter_p = _run(args, chain_parents(8), cache=True)
    assert torch.equal(out, out_p)
    assert torch.equal(state, state_p)
    assert torch.equal(inter, inter_p)
    out_t, _, _ = _run(args, batch_parents(CHAIN_8, args["q"].shape[0], device))
    assert torch.equal(out, out_t)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", list(TREES))
def test_tree_parents_match_the_per_node_ancestor_replay(name, device):
    args = _inputs(device)
    parents = TREES[name]
    out, state, inter = _run(args, parents, cache=True)
    ref_out, ref_states = _oracle(args, parents)
    assert _rel(out, ref_out) < 1e-5, f"{name}: output {_rel(out, ref_out):.3e}"
    assert _rel(inter, ref_states) < 1e-5, f"{name}: states"
    assert torch.equal(state, inter[:, -1])


@pytest.mark.parametrize("device", DEVICES)
def test_each_request_may_carry_its_own_topology(device):
    args = _inputs(device)
    names = ["star", "topk2_depth4"]
    parents = torch.tensor([TREES[n] for n in names], dtype=torch.int32, device=device)
    out, _, _ = _run(args, parents)
    for row, name in enumerate(names):
        ref_out, _ = _oracle(args, TREES[name])
        assert _rel(out[row], ref_out[row]) < 1e-5, name


@pytest.mark.parametrize("device", DEVICES)
def test_tree_differs_from_chain_where_the_tree_forks(device):
    """The silent-wrongness guard: a fork must not reproduce the chain answer."""
    args = _inputs(device)
    chain, _, _ = _run(args)
    forked, _, _ = _run(args, TREES["star"])
    assert torch.equal(forked[:, 0], chain[:, 0])
    assert _rel(forked[:, 1:], chain[:, 1:]) > 1e-2


def test_rejects_malformed_parents():
    args = _inputs("cpu")
    for bad in (
        [-1, 2, 1, 2, 3, 4, 5, 6],
        [-1, 1, 1, 2, 3, 4, 5, 6],
        [-2, 0, 1, 2, 3, 4, 5, 6],
        [0, 0, 1, 2, 3, 4, 5, 6],
        [-1, 0, 1, 2, 3, 4, 5],
    ):
        with pytest.raises(ValueError):
            _run(args, bad)
    with pytest.raises(ValueError):
        _run(args, batch_parents(CHAIN_8, 3))
