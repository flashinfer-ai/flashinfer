"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch

import flashinfer.utils as flashinfer_utils
from flashinfer.utils import resolve_enable_pdl


@pytest.mark.parametrize("supported", (False, True))
def test_resolve_enable_pdl(monkeypatch, supported):
    """``None`` follows device support; only a bool overrides it."""
    assert resolve_enable_pdl(None, torch.device("cpu")) is False
    device = torch.device("cuda", 0)
    queried = []

    def device_support_pdl(queried_device):
        queried.append(queried_device)
        return supported

    monkeypatch.setattr(flashinfer_utils, "device_support_pdl", device_support_pdl)
    assert resolve_enable_pdl(None, device) is supported
    for enable_pdl in (False, True):
        assert resolve_enable_pdl(enable_pdl, device) is enable_pdl
    assert queried == [device]
    for invalid in (0, 1, "yes"):
        with pytest.raises(TypeError, match="enable_pdl must be a bool or None"):
            resolve_enable_pdl(invalid, device)
