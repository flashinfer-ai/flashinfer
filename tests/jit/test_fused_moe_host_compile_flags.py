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

from unittest.mock import Mock

import pytest

from flashinfer.jit import fused_moe


@pytest.mark.parametrize(
    ("machine", "is_gcc", "expected"),
    [
        ("aarch64", True, ["-Xcompiler", "-fno-schedule-insns"]),
        ("aarch64", False, []),
        ("x86_64", True, []),
        ("x86_64", False, []),
    ],
)
def test_manifest_host_compile_flags(monkeypatch, machine, is_gcc, expected):
    monkeypatch.setattr(fused_moe.platform, "machine", lambda: machine)
    compiler_check = Mock(return_value=is_gcc)
    monkeypatch.setattr(fused_moe, "host_compiler_is_gcc", compiler_check)

    assert fused_moe._manifest_host_compile_flags() == expected
    if machine == "aarch64":
        compiler_check.assert_called_once_with()
    else:
        compiler_check.assert_not_called()
