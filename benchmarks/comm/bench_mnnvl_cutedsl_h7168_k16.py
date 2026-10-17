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

"""Kimi K3 H7168/K16 MNNVL CuTe DSL reference benchmark.

This configures the shared H5120 benchmark driver for Kimi K3's exact bf16
semantics: hidden_size=7168, top_k=16, rms_eps=1e-5, zero gamma bias, routed
scale 1.0, shared-expert add, and a nonzero residual. It compares LL/BT/HT and
their shipped M-range routes with an unfused torch MoE finalize + NCCL
all-reduce + FlashInfer fused_add_rmsnorm baseline.

TP4 runs on one GB200 node. TP8 requires two nodes joined by torchrun over
multi-node NVLink. Timing captures 20 back-to-back calls in a CUDA graph and
reports the per-call median of the max across ranks.

Examples::

    torchrun --standalone --nproc-per-node=4 \
        benchmarks/comm/bench_mnnvl_cutedsl_h7168_k16.py --csv tp4.csv

    torchrun --nnodes=2 --nproc-per-node=4 --node-rank=$NODE_RANK \
        --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT \
        benchmarks/comm/bench_mnnvl_cutedsl_h7168_k16.py --csv tp8.csv
"""

import bench_mnnvl_cutedsl_h5120 as benchmark
from flashinfer.comm.mnnvl_cutedsl.kernel_bt import (
    BT_ALL_REDUCE_GB300_TP4_H7168_PRESET_0,
    BT_ALL_REDUCE_GB300_TP4_H7168_PRESET_1,
    BT_ALL_REDUCE_GB300_TP8_H7168_PRESET_0,
    BT_ALL_REDUCE_GB300_TP8_H7168_PRESET_1,
    BT_FINALIZE_GB300_TP4_H7168_K16_PRESET_0,
    BT_FINALIZE_GB300_TP4_H7168_K16_PRESET_1,
    BT_FINALIZE_GB300_TP8_H7168_K16_PRESET_0,
    BT_FINALIZE_GB300_TP8_H7168_K16_PRESET_1,
)


class Inputs(benchmark.Inputs):
    """Use Kimi's nonzero residual for the finalize path."""

    def __init__(self, max_m, max_top_k, device):
        super().__init__(max_m, max_top_k, device)
        self.zero_residual = self.residual


def main() -> int:
    benchmark.HIDDEN_SIZE = 7168
    benchmark.TOP_K_STAGES = (16,)
    benchmark.RMS_EPS = 1e-5
    benchmark.WEIGHT_BIAS = 0.0
    benchmark.DEFAULT_M_LIST = (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
    )
    benchmark._BT_FINALIZE_PRESETS = {
        (4, 16): (
            BT_FINALIZE_GB300_TP4_H7168_K16_PRESET_0,
            BT_FINALIZE_GB300_TP4_H7168_K16_PRESET_1,
        ),
        (8, 16): (
            BT_FINALIZE_GB300_TP8_H7168_K16_PRESET_0,
            BT_FINALIZE_GB300_TP8_H7168_K16_PRESET_1,
        ),
    }
    benchmark._BT_ALL_REDUCE_PRESETS = {
        4: (
            BT_ALL_REDUCE_GB300_TP4_H7168_PRESET_0,
            BT_ALL_REDUCE_GB300_TP4_H7168_PRESET_1,
        ),
        8: (
            BT_ALL_REDUCE_GB300_TP8_H7168_PRESET_0,
            BT_ALL_REDUCE_GB300_TP8_H7168_PRESET_1,
        ),
    }
    benchmark.Inputs = Inputs
    return benchmark.main()


if __name__ == "__main__":
    raise SystemExit(main())
