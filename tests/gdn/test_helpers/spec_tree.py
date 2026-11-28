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

"""Parent-indexed draft-tree topologies shared by the linear-attention verify tests.

A speculative draft of ``T`` tokens is described by ``parents[t]``: the index of
token ``t``'s parent within the same draft, or ``-1`` when its parent is the
committed state. The root entry ``parents[0]`` is ignored by the kernels, so
engines that write 0 there (SGLang's conv kernel does) need no copy. A chain is
``[-1, 0, 1, ..., T-2]``; EAGLE-style drafts with
``topk > 1`` fork, so a token's history is its root-to-node path rather than
the prefix ``0..t-1``.
"""

import torch

CHAIN_8 = [-1, 0, 1, 2, 3, 4, 5, 6]

TREES = {
    "chain": CHAIN_8,
    "fork_at_root": [-1, 0, 0, 1, 2, 3, 4, 5],
    "topk4_depth2": [-1, 0, 0, 0, 0, 1, 1, 2],
    "lopsided": [-1, 0, 1, 1, 3, 3, 5, 2],
    "star": [-1, 0, 0, 0, 0, 0, 0, 0],
    "topk2_depth4": [-1, 0, 0, 1, 1, 2, 2, 3],
}


def chain_parents(steps):
    return list(range(-1, steps - 1))


def make_tree(kind, steps, seed=0):
    """A ``steps``-node draft tree of the given shape (root parent is -1).

    ``binary`` is a complete binary tree in level order, ``star`` hangs every
    node off the root, and ``random`` draws each parent uniformly from the
    earlier nodes with a fixed seed.
    """
    if kind == "chain":
        return chain_parents(steps)
    if kind == "binary":
        return [-1] + [(node - 1) // 2 for node in range(1, steps)]
    if kind == "star":
        return [-1] + [0] * (steps - 1)
    if kind == "random":
        generator = torch.Generator().manual_seed(seed * 1000 + steps)
        return [-1] + [
            int(torch.randint(0, node, (1,), generator=generator))
            for node in range(1, steps)
        ]
    raise ValueError(f"unknown tree kind {kind!r}")


def ancestors(parents, node):
    """Proper ancestors of ``node``, root first."""
    path, cur = [], int(parents[node])
    while cur >= 0:
        path.append(cur)
        cur = int(parents[cur])
    return path[::-1]


def batch_parents(parents, batch, device=None):
    """Tile one topology into the ``int32 [B, T]`` layout the kernels take."""
    return torch.tensor([list(parents)] * batch, dtype=torch.int32, device=device)
