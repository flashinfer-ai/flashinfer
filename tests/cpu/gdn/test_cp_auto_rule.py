# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``use_cp="auto"`` boundaries of ``should_use_cp_host`` and
``should_use_cp_sm110_host``, and the SM110 F/P ratio.

No CI lane runs SM110, so this pins its thresholds without hardware, and pins
the SM100/SM107/SM120 thresholds of ``should_use_cp_host``.
"""

import pytest

from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import (
    cp_short_workload_ratio_host,
    should_use_cp_host,
    should_use_cp_sm110_host,
)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "device_name, cc, num_sms, work, expected",
    [
        ("NVIDIA B200", (10, 0), 148, 36, True),
        ("NVIDIA B200", (10, 0), 148, 37, False),
        ("NVIDIA R200", (10, 7), 212, 52, True),
        ("NVIDIA R200", (10, 7), 212, 53, False),
        ("NVIDIA GeForce RTX 5090", (12, 0), 170, 56, True),
        ("NVIDIA GeForce RTX 5090", (12, 0), 170, 57, False),
    ],
)
def test_should_use_cp_host(device_name, cc, num_sms, work, expected) -> None:
    assert should_use_cp_host(work, num_sms, device_name, cc) is expected


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "cc, work, max_seq_len, expected",
    [
        ((11, 0), 1, 2048, True),
        ((11, 0), 1, 2047, False),
        ((11, 0), 2, 3072, True),
        ((11, 0), 2, 3071, False),
        ((11, 0), 3, 3072, True),
        ((11, 0), 3, 3071, False),
        ((11, 0), 4, 4096, True),
        ((11, 0), 4, 4095, False),
        ((11, 0), 5, 65536, False),
        ((11, 0), 1, None, False),
        ((10, 0), 1, 65536, False),
        (None, 1, 65536, False),
    ],
)
def test_should_use_cp_sm110_host(cc, work, max_seq_len, expected) -> None:
    assert should_use_cp_sm110_host(work, cc, max_seq_len) is expected


@pytest.mark.cpu_only
def test_cp_short_workload_ratio_sm110() -> None:
    assert cp_short_workload_ratio_host((11, 0)) == (4, 5)
