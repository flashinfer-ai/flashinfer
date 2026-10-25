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
"""Fixed FP8 MQA geometry shared by API validation and kernel preparation."""

# The kernel derives mma_inst_tile_k by integer division, so head_dim must
# be an exact multiple of this instruction width.
_FP8_MMA_INST_K = 32
# Register and shared-memory weight paths consume this many heads at a time.
# UMMA's next_n * num_heads alignment alone does not imply this constraint.
_EPI_UNROLL = 4
_COMPUTE_BLOCK_KV = 128
# Maximum TMA sub-copies per compute tile.
_MAX_BLOCKS_PER_MMA = 4
