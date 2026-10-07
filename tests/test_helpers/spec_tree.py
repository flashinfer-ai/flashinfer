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
committed state. A chain is ``[-1, 0, 1, ..., T-2]``; EAGLE-style drafts with
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
