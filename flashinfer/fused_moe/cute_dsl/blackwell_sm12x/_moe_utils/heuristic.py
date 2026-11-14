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


def tile_count(*, total_rows, n, num_experts, bm, bn):
    low, extra = divmod(total_rows, num_experts)
    m_tiles = (num_experts - extra) * ceil_div(low, bm) + extra * ceil_div(low + 1, bm)
    return m_tiles * ceil_div(n, bn)


def select_fp4_fc1_act_q1_tile(
    *, total_rows: int, n: int, k: int, num_experts: int, num_sms: int
):
    if total_rows < 0 or num_experts < 0 or min(n, k, num_sms) <= 0:
        raise ValueError(
            "positive shapes and SM count, nonnegative rows and experts required"
        )
    bn = 128 if n % 128 == 0 else 64
    if total_rows == 0 or num_experts == 0:
        return ((8 if bn == 128 else 32), bn, 128)
    m_per_expert = ceil_div(total_rows, num_experts)

    tiles8 = tile_count(
        total_rows=total_rows, n=n, num_experts=num_experts, bm=8, bn=bn
    )
    tiles32 = tile_count(
        total_rows=total_rows, n=n, num_experts=num_experts, bm=32, bn=bn
    )
    tiles64 = tile_count(
        total_rows=total_rows, n=n, num_experts=num_experts, bm=64, bn=bn
    )

    if bn == 128 and (
        m_per_expert <= 32 or (tiles32 < num_sms // 2 and tiles8 <= num_sms)
    ):
        bm = 8
    elif m_per_expert <= 32:
        bm = 32
    elif m_per_expert >= (128 if k <= 2048 else 256):
        bm = 128
    elif k > 2048 and bn == 128 and m_per_expert <= 128:
        bm = 32
    else:
        waves32, waves64 = ceil_div(tiles32, num_sms), ceil_div(tiles64, num_sms)
        tail32 = tiles32 - (waves32 - 1) * num_sms
        tail64 = tiles64 - (waves64 - 1) * num_sms
        bm = (
            32
            if (waves64 >= waves32 or (tail64 * 2 < num_sms and tail64 < tail32))
            else 64
        )
    return (bm, bn, 128)


def select_fp4_fc2_finalize_tile(
    *, total_rows: int, n: int, k: int, num_experts: int, num_sms: int
):
    if total_rows < 0 or num_experts < 0 or min(n, k, num_sms) <= 0:
        raise ValueError(
            "positive shapes and SM count, nonnegative rows and experts required"
        )
    if total_rows == 0 or num_experts == 0:
        return (32, 128, 128)
    m_per_expert = ceil_div(total_rows, num_experts)
    short_k = k <= 1024
    full_k_tiles = k % 128 == 0

    tiles8 = tile_count(
        total_rows=total_rows, n=n, num_experts=num_experts, bm=8, bn=128
    )
    tiles32 = tile_count(
        total_rows=total_rows, n=n, num_experts=num_experts, bm=32, bn=128
    )
    tiles64 = tile_count(
        total_rows=total_rows, n=n, num_experts=num_experts, bm=64, bn=128
    )

    if full_k_tiles and (
        m_per_expert <= (16 if short_k else 8)
        or (tiles32 < num_sms // 2 and tiles8 <= num_sms)
    ):
        bm = 8
    elif m_per_expert <= (24 if short_k else 32):
        bm = 32
    elif m_per_expert >= 64:
        bm = 128
    else:
        waves32 = ceil_div(tiles32, num_sms)
        waves64 = ceil_div(tiles64, num_sms)
        if short_k and m_per_expert < 48 and waves64 < 16:
            bm = 32
        elif m_per_expert >= 48 and waves64 >= 32:
            bm = 128
        else:
            bm = 64 if short_k or waves64 < waves32 else 32
    return (bm, 128, 128)
