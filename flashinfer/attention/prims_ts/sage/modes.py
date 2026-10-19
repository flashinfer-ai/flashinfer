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

"""Names for how a Sage softmax obtains and applies the ``sfK`` scales.

Each attention kernel picks one transport and one scaling per plan and keeps
its own choice, resources and code; the names are shared so that the same
mechanism reads the same in both kernels. The decode kernel stages every K
tile's words in its ring and scales blocks of ``k_block_size`` columns, or
dequantizes one-token blocks. The context kernel loads the words of
four-token and coarser blocks from global memory and stages one-token blocks
in its ring, and scales 16-column blocks or dequantizes finer ones.
"""

import enum


class SageKTransport(enum.Enum):
    """How the softmax obtains a K tile's ``sfK`` words."""

    # The softmax lanes load the words from the prepared buffer.
    GLOBAL = "global"
    # The load warp stages the words in the kernel's shared-memory ring.
    RING = "ring"


class SageScoreScaling(enum.Enum):
    """How the softmax applies ``sfK`` to a K tile's scores."""

    # The scores stay raw; each block's maximum and exp2 multiplier carry
    # ``sfQ * sfK``.
    BLOCK = "block"
    # The max pass multiplies the scores by ``sfK`` in place.
    DEQUANTIZED = "dequantized"
