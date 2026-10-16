# Copyright (c) 2025 by FlashInfer team.
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
"""What more than one moe_gemm arm shares: shape-based tile selection, ported from flashinfer."""

from .....utils import ceil_div

PLAIN_TILE_OVERHEAD = 48


def select_plain_bm_64_or_128(
    m_per_expert: int, n: int, num_experts: int, num_sms: int
) -> int:
    def cost(bm: int) -> int:
        num_tiles = num_experts * ceil_div(m_per_expert, bm) * ceil_div(n, 128)
        return ceil_div(num_tiles, num_sms) * (bm + PLAIN_TILE_OVERHEAD)

    return 64 if cost(64) < cost(128) else 128


FC1_ACT_HEURISTIC_BM = (64, 32)


def select_fc1_act_tile(
    *, total_rows: int, n: int, num_experts: int, num_sms: int, tiles, gran_k: int
):
    by_bm = dict(tiles)
    m_per_expert = total_rows // num_experts if num_experts > 0 else 0
    smallest = min(FC1_ACT_HEURISTIC_BM)
    if m_per_expert <= 32:
        bm = smallest
    else:
        bm = select_plain_bm_64_or_128(m_per_expert, n, num_experts, num_sms)
    bm = min(max(bm, smallest), max(FC1_ACT_HEURISTIC_BM))
    return (bm, by_bm[bm], gran_k)
