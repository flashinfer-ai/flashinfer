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

"""High-throughput MNNVL protocol."""

from .protocol import (
    HTAllReduceTuning,
    HTFinalizeTuning,
    HT_ALL_REDUCE_GB300_TP16_H8192,
    HT_ALL_REDUCE_GB300_TP4_H5120,
    HT_ALL_REDUCE_GB300_TP8_H8192,
    HT_FINALIZE_GB300_TP16_H8192_K10,
    HT_FINALIZE_GB300_TP4_H5120_K3,
    HT_FINALIZE_GB300_TP4_H5120_K6,
    HT_FINALIZE_GB300_TP8_H8192_K10,
    HT_FINALIZE_GB300_TP8_H8192_K10_M_GE_2049,
    HT_FINALIZE_GB300_TP8_H8192_K10_M_LE_2048,
)

__all__ = [
    "HT_ALL_REDUCE_GB300_TP8_H8192",
    "HT_ALL_REDUCE_GB300_TP16_H8192",
    "HT_FINALIZE_GB300_TP8_H8192_K10",
    "HT_FINALIZE_GB300_TP8_H8192_K10_M_GE_2049",
    "HT_FINALIZE_GB300_TP8_H8192_K10_M_LE_2048",
    "HT_FINALIZE_GB300_TP16_H8192_K10",
    "HTAllReduceTuning",
    "HTFinalizeTuning",
    "HT_ALL_REDUCE_GB300_TP4_H5120",
    "HT_FINALIZE_GB300_TP4_H5120_K3",
    "HT_FINALIZE_GB300_TP4_H5120_K6",
]
