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

"""Resource definitions for the throughput 2CTA MLA decode TS kernel.

Resource classes matching the throughput 2CTA pipeline structure:

GMEM/register resources (not pipelined)
----------------------------------------
- PageOffsetWindowResource: one warp-wide, 32-page-ID register window

SMEM resources (pipelined)
--------------------------
- SmemQResource     : TmaUmmaAsync(1 stage),   LoadTma -> Mma
- SmemKVResource    : TmaUmmaAsync(7 stages), LoadTma -> Mma

SMEM/TMEM resources (pipelined)
-------------------------------
- SmemPResource     : UmmaConsumerAsync(2 stages), SoftmaxTask -> MmaTask
- TmemSResource     : UmmaProducerAsync(2 stages), MmaTask -> SoftmaxTask
- TmemCorrResource  : Async(2 stages),              SoftmaxTask -> CorrectionTask
- TmemOResource     : UmmaProducerAsync(1 stage),   MmaTask -> CorrectionTask

Output
------
- GmemOResource     : Direct stores, or Async(2 stages) Correction -> TMA -> GMEM
"""

from .common import (
    HighThroughputMlaResource,
    WorkThrottleBarrierResource,
    with_cluster_acquire_waits,
)
from .gmem_o import GmemOResource
from .smem_p import SmemPResource
from .smem_resources import (
    PageOffsetWindowResource,
    SmemKResource,
    SmemKVResource,
    SmemQResource,
    SmemVResource,
)
from .tmem_corr import TmemCorrResource
from .tmem_o import TmemOResource
from .tmem_s import TmemSResource

__all__ = [
    "GmemOResource",
    "HighThroughputMlaResource",
    "PageOffsetWindowResource",
    "SmemKResource",
    "SmemKVResource",
    "SmemPResource",
    "SmemQResource",
    "SmemVResource",
    "TmemCorrResource",
    "TmemOResource",
    "TmemSResource",
    "WorkThrottleBarrierResource",
    "with_cluster_acquire_waits",
]
