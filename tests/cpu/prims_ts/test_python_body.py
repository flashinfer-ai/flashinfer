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

"""Source-body audits must survive CuTe replacing its wrapped function."""

import sys

import cutlass.cute as cute
import pytest

from tests.prims_ts.python_body import python_body


_increment = 3


@cute.jit
def _body(value):
    return value + _increment


class _Resource:
    @cute.jit
    def body(self, value):
        return value + self.increment + _increment


@pytest.mark.parametrize("function", [_body, _Resource.body])
def test_python_body_uses_original_source_and_live_globals(monkeypatch, function):
    def transformed(*args):
        raise AssertionError("transformed DSL must not execute in a host audit")

    monkeypatch.setattr(function, "__wrapped__", transformed)
    body = python_body(function)
    monkeypatch.setattr(sys.modules[__name__], "_increment", 7)
    if function is _body:
        assert body(5) == 12
    else:
        resource = _Resource()
        resource.increment = 11
        assert body(resource, 5) == 23
