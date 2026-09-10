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

from dataclasses import dataclass

import cutlass.cute as cute


@dataclass
class BatchedStaticWorkDesc:
    qo_tile_idx: int
    qo_head_idx: int
    kv_head_idx: int
    batch_idx: int


class BatchedStaticSchedulerMixin:
    def get_grid_config(self, seqlen_q, num_qo_heads, batch_size):
        tile_size_m = self.tile_shape_qk[0]
        num_q_tiles = cute.ceil_div(seqlen_q, tile_size_m)
        return (num_q_tiles, num_qo_heads, batch_size)

    def get_work_desc(self):
        qo_tile_idx, qo_head_idx, batch_idx = cute.arch.block_idx()
        kv_head_idx = qo_head_idx // self.gqa_ratio
        return BatchedStaticWorkDesc(qo_tile_idx, qo_head_idx, kv_head_idx, batch_idx)
