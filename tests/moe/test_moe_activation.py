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

"""moe_utils' ``moe_activation`` against a torch reference.

Gated activations read ``[linear | gate]`` halves and compute
``act(gate) * linear``.
"""

import pytest
import torch
import torch.nn.functional as F

from flashinfer.fused_moe.cute_dsl.moe_utils import MoeActivationType, moe_activation
from flashinfer.utils import get_compute_capability

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and get_compute_capability(torch.device("cuda"))[0] >= 8
    ),
    reason="moe_utils builds for SM80 and newer",
)

_REFERENCE = {
    MoeActivationType.Identity: lambda x: x,
    MoeActivationType.Relu: F.relu,
    MoeActivationType.Silu: F.silu,
    MoeActivationType.Gelu: F.gelu,
}
_GATED_REFERENCE = {
    MoeActivationType.Swiglu: F.silu,
    MoeActivationType.Geglu: F.gelu,
}


def _reference(x: torch.Tensor, activation: MoeActivationType, interm_size: int):
    x = x.float()
    if activation in _GATED_REFERENCE:
        linear, gate = x[:, :interm_size], x[:, interm_size:]
        return _GATED_REFERENCE[activation](gate) * linear
    return _REFERENCE[activation](x)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("activation", list(MoeActivationType))
@pytest.mark.parametrize("interm_size", [64, 512, 768, 1536, 4096])
@pytest.mark.parametrize(
    ("num_rows", "tile_size"), [(16, 16), (384, 16), (1024, 128), (32768, 16)]
)
def test_moe_activation(num_rows, tile_size, interm_size, activation, dtype):
    torch.manual_seed(0)
    device = torch.device("cuda")
    gated = activation in _GATED_REFERENCE
    x = torch.randn(
        num_rows, interm_size * (2 if gated else 1), dtype=dtype, device=device
    )
    out = torch.empty(num_rows, interm_size, dtype=dtype, device=device)
    # Every tile full: all rows are activated.
    num_tiles = num_rows // tile_size
    mn_limit = ((torch.arange(num_tiles, device=device) + 1) * tile_size).to(
        torch.int32
    )
    num_live_tiles = torch.tensor([num_tiles], dtype=torch.int32, device=device)

    moe_activation(x, out, mn_limit, num_live_tiles, activation, num_rows, tile_size)

    torch.testing.assert_close(
        out.float(), _reference(x, activation, interm_size), rtol=1e-2, atol=1e-2
    )
