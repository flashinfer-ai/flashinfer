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

"""Task definitions for the throughput 2CTA MLA decode TS kernel.

The selected graph depends on dtype and scheduler policy. BF16 uses a 12-warp
combined TMA/MMA graph, with an additional scheduler task under CLC. FP8 uses a
16-warp split K/V and QK/PV graph with two softmax groups.

Domain semantics: domain = k_tile_count (total k-tiles to process).
Tasks with domain_start=1 handle the first k-tile in HEAD, then LOOP
covers k-tiles 1..N-1, and TAIL handles cleanup.

Stagger pattern (K loads 1 ahead of V, QK MMA 1 ahead of PV MMA):
  k-tile 0: load K[0], QK MMA -> S[0]
  k-tile 1: load K[1]+V[0], PV MMA(P[0],V[0])->O[0], QK MMA->S[1]
  ...
  k-tile N-1: load K[N-1]+V[N-2], PV MMA(P[N-2],V[N-2])->O[N-2], QK MMA->S[N-1]
  tail: load V[N-1], PV MMA(P[N-1],V[N-1])->O[N-1]

The BF16 register roles are:
  LoadTmaTask     (warp 9,    1 warp,   96 regs; owns page-ID window)
  MmaTask         (warp 8,    1 warp,   96 regs)
  SoftmaxTask     (warps 0-3, 4 warps, 192 regs)
  CorrectionTask  (warps 4-7, 4 warps, 208 regs)
  PaddingTask     (warps 10-11,          96 regs; non-CLC alignment)
  SchedulerTask   (warp 11,              96 regs; CLC only)
"""
