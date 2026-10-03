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

import pytest
import torch

from flashinfer.kda_kernels.fused_kda_decode_multitoken import _pick_tile


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (10, 7)])
@pytest.mark.parametrize("sm_count", [148, 160])
@pytest.mark.parametrize("tokens", [5, 8])
@pytest.mark.parametrize(
    ("waves", "offset", "expected"),
    [
        (1, -1, (2, 16, 1, 1, 1)),
        (1, 0, (2, 16, 1, 1, 1)),
        (1, 1, (2, 16, 2, 2, 1)),
        (2, -1, (2, 16, 2, 2, 1)),
        (2, 0, (2, 16, 3, 2, 1)),
        (2, 1, (2, 16, 3, 2, 1)),
    ],
)
def test_multitoken_dispatch_wave_boundary(
    capability, sm_count, tokens, waves, offset, expected
):
    sequence_heads = waves * sm_count + offset
    assert _pick_tile(tokens, sequence_heads, sm_count, capability) == expected


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (10, 7)])
@pytest.mark.parametrize("sm_count", [148, 160])
@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 6, 7, 9])
@pytest.mark.parametrize("waves", [1, 2])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_multitoken_dispatch_other_token_counts(
    capability, sm_count, tokens, waves, offset
):
    sequence_heads = waves * sm_count + offset
    if tokens == 1:
        expected = (2, 16, 4, 4, 1)
    elif waves == 1 or offset == -1:
        expected = (2, 16, 1, 1, 1)
    elif tokens <= 3:
        expected = (1, 16, 4, 4, 1)
    else:
        expected = (2, 16, 3, 2, 1)
    assert _pick_tile(tokens, sequence_heads, sm_count, capability) == expected


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (10, 7)])
@pytest.mark.parametrize("sm_count", [148, 160])
@pytest.mark.parametrize("tokens", [1, 5, 8])
@pytest.mark.parametrize(
    ("divisor", "offset", "expected"),
    [
        (8, -1, (1, 32, 1, 1, 8)),
        (8, 0, (1, 32, 1, 1, 8)),
        (8, 1, (1, 16, 1, 1, 4)),
        (4, -1, (1, 16, 1, 1, 4)),
        (4, 0, (1, 16, 1, 1, 4)),
        (4, 1, (2, 16, 1, 1, 1)),
    ],
)
def test_multitoken_dispatch_split_boundary(
    capability, sm_count, tokens, divisor, offset, expected
):
    sequence_heads = sm_count // divisor + offset
    if tokens == 1:
        expected = (2, 16, 4, 4, 1)
    assert _pick_tile(tokens, sequence_heads, sm_count, capability) == expected


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (10, 7)])
@pytest.mark.parametrize("sm_count", [144, 192])
@pytest.mark.parametrize("tokens", range(1, 10))
@pytest.mark.parametrize("heads", [12, 24, 32, 48, 96])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_multitoken_norm_order_dispatch(
    monkeypatch, capability, sm_count, tokens, heads, state_dtype, offset
):
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module(
        "flashinfer.kda_kernels.fused_kda_decode_multitoken"
    )
    sequences = 2 * sm_count // heads + offset
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            multi_processor_count=sm_count, shared_memory_per_block_optin=232448
        ),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    chosen = []

    def compile_kernel(*args):
        chosen.append(args[10])
        return lambda *args: None

    monkeypatch.setattr(module, "_get_compiled_kernel", compile_kernel)
    x = torch.empty(0, 0, dtype=torch.bfloat16)
    module._run_fused_kda_decode_multitoken(
        x,
        None,
        None,
        None,
        torch.empty(1, 0, 0),
        torch.empty(heads),
        None,
        torch.empty(sequences, tokens, dtype=torch.int32),
        None,
        None,
        torch.empty(0, dtype=state_dtype),
        x,
        None,
        -5.0,
        1e-5,
        None,
    )
    expected = 1
    sequence_heads = sequences * heads
    if tokens > 3 and (
        sequence_heads >= 2 * sm_count
        or (tokens in (5, 8) and sm_count < sequence_heads)
    ):
        expected = 3
    assert chosen == [expected]


def test_multitoken_norm_order_cache_key(monkeypatch):
    import importlib

    module = importlib.import_module(
        "flashinfer.kda_kernels.fused_kda_decode_multitoken"
    )
    monkeypatch.setattr(
        module,
        "build_and_load_cute_dsl_kernel",
        lambda name, key, compiler, **kwargs: key,
    )
    builder = module._get_compiled_kernel.__wrapped__
    args = (5, 96, -5.0, 1e-5, 2, 16, 3, 2, 1, True)
    original = builder(*args)
    ordered = builder(*args, 3)
    assert original != ordered
    assert ordered.endswith("_sbf16_normorder3")
