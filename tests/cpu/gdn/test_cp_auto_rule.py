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

"""CPU-only tests of the ``use_cp="auto"`` rule in ``should_use_cp_host``.

No CI lane runs SM110, so these pin its thresholds without hardware, and pin
that the SM100/SM103/SM107 and SM120 answers do not depend on the SM110 rule.
"""

import pytest

from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import should_use_cp_host

THOR = dict(num_sms=20, device_name="NVIDIA Thor", device_capability=(11, 0))


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "num_parallel_work, max_seq_len, expected",
    [
        (1, 4096, True),
        (2, 4096, True),
        (2, 65536, True),
        (3, 4096, False),
        (3, 65536, False),
        (1, 4095, False),
        (2, 4095, False),
        (1, None, False),
    ],
)
def test_sm110_rule(num_parallel_work, max_seq_len, expected) -> None:
    assert (
        should_use_cp_host(num_parallel_work, max_seq_len=max_seq_len, **THOR)
        is expected
    )


@pytest.mark.cpu_only
def test_sm110_rule_ignores_the_sm_count_and_device_name() -> None:
    for num_sms, device_name in ((148, "NVIDIA B200"), (170, "NVIDIA GeForce")):
        assert should_use_cp_host(
            2, num_sms, device_name, device_capability=(11, 0), max_seq_len=4096
        )
        assert not should_use_cp_host(
            3, num_sms, device_name, device_capability=(11, 0), max_seq_len=65536
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("device_capability", [(10, 0), (10, 3), (10, 7)])
@pytest.mark.parametrize("max_seq_len", [None, 1024, 4096, 65536])
def test_sm10x_rule_is_four_times_work_below_the_sm_count(
    device_capability, max_seq_len
) -> None:
    # CP iff 4 * num_parallel_work < num_sms; the longest sequence is ignored.
    for num_parallel_work, expected in ((1, True), (36, True), (37, False)):
        assert (
            should_use_cp_host(
                num_parallel_work,
                148,
                "NVIDIA B200",
                device_capability=device_capability,
                max_seq_len=max_seq_len,
            )
            is expected
        )
    # Thor's SM count under the SM10x rule: CP up to 4 states, unlike SM110.
    assert should_use_cp_host(4, 20, "x", device_capability=device_capability)
    assert not should_use_cp_host(5, 20, "x", device_capability=device_capability)


@pytest.mark.cpu_only
@pytest.mark.parametrize("max_seq_len", [None, 65536])
def test_sm120_rule_uses_the_memory_class_threshold(max_seq_len) -> None:
    cc = (12, 0)
    # GDDR names: CP iff 3 * work < num_sms.
    gddr = "NVIDIA GeForce RTX 5090"
    assert should_use_cp_host(56, 170, gddr, cc, max_seq_len)
    assert not should_use_cp_host(57, 170, gddr, cc, max_seq_len)
    # Other names: CP iff 2 * work < num_sms.
    hbm = "NVIDIA B40"
    assert should_use_cp_host(84, 170, hbm, cc, max_seq_len)
    assert not should_use_cp_host(85, 170, hbm, cc, max_seq_len)
