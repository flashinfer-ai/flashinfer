# Copyright (c) 2025 by FlashInfer team.
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

import pytest


@pytest.fixture(autouse=True)
def native_auto_policy(monkeypatch):
    """The native kernel suites test FlashInfer's own KDA prefill, so
    ``backend="auto"`` stays on the native route here even where the cuDNN
    policy region would take it (an SM100 box with cudnn-frontend 1.31+).
    Same mechanism as the benchmark's ``dispatch_policy``; a module that
    tests the auto resolution itself overrides this fixture by name."""
    from flashinfer import kda

    monkeypatch.setattr(kda, "_prefer_cudnn_kda_prefill", lambda *args, **kwargs: False)
