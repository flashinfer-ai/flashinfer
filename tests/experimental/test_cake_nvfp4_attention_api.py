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

Public-API tests of ``flashinfer.prefill.prepare_nvfp4_attention``, the
``backend="cake"`` entry of the experimental NVFP4 attention package; the
package's own suite is ``test_cake_nvfp4_attention.py``.
"""

import pytest
import torch

from flashinfer.prefill import prepare_nvfp4_attention

SM103 = (10, 3)


def _skip_unless_sm103():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != SM103:
        pytest.skip("SM103 required")


def test_nvfp4_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        prepare_nvfp4_attention(None, None, None, None, backend="unknown")


def test_nvfp4_attention_public_entry_runs_the_cake_route():
    _skip_unless_sm103()
    torch.manual_seed(5)
    q = torch.randn((1, 2, 1024, 128), dtype=torch.bfloat16, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    out = torch.empty_like(q)
    attention = prepare_nvfp4_attention(q, k, v, out, backend="cake")
    assert attention() is out
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(out, expected, atol=1.0, rtol=0.1)
