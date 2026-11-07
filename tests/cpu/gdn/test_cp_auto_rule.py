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

"""``use_cp="auto"`` boundaries of ``should_use_cp_host``.

No CI lane runs SM110, so this pins its thresholds without hardware, and pins
that SM100/SM107/SM120 keep their answers whatever ``max_seq_len`` is.
"""

import pytest

from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import should_use_cp_host


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "device_name, cc, num_sms, work, max_seq_len, expected",
    [
        ("NVIDIA Thor", (11, 0), 20, 2, 4096, True),
        ("NVIDIA Thor", (11, 0), 20, 3, 65536, False),
        ("NVIDIA Thor", (11, 0), 20, 2, 4095, False),
        ("NVIDIA B200", (10, 0), 148, 36, None, True),
        ("NVIDIA B200", (10, 0), 148, 37, 65536, False),
        ("NVIDIA R200", (10, 7), 212, 52, None, True),
        ("NVIDIA R200", (10, 7), 212, 53, 65536, False),
        ("NVIDIA GeForce RTX 5090", (12, 0), 170, 56, None, True),
        ("NVIDIA GeForce RTX 5090", (12, 0), 170, 57, 65536, False),
    ],
)
def test_should_use_cp_host(
    device_name, cc, num_sms, work, max_seq_len, expected
) -> None:
    assert should_use_cp_host(work, num_sms, device_name, cc, max_seq_len) is expected
