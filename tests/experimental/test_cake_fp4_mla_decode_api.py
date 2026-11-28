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

Public-API tests of ``flashinfer.mla.prepare_fp4_mla_decode``, the
``backend="cake"`` entry of the experimental MLA decode over TensorRT-LLM's
``fp4_mla`` KV cache; the package's own suite is
``test_cake_fp4_mla_decode.py``.
"""

import math

import pytest
import torch

from flashinfer.mla import prepare_fp4_mla_decode
from tests.experimental.test_cake_fp4_mla_decode import (
    _device_or_skip,
    _run,
    _write_case,
    pkg,
)


def test_fp4_mla_decode_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        prepare_fp4_mla_decode(
            batch=1, q_len=1, num_heads=128, page_table_width=1, backend="unknown"
        )


def test_fp4_mla_decode_public_entry_runs_the_cake_route():
    device = _device_or_skip()
    heads, q_len, kv_lens, width = 128, 4, (3000, 1100), 300
    sm_scale = 1.0 / math.sqrt(192.0)
    case = _write_case(5, heads, q_len, kv_lens, width, device)
    kwargs = dict(
        batch=2, q_len=q_len, num_heads=heads, page_table_width=width, device=device
    )
    public = prepare_fp4_mla_decode(**kwargs, max_kv=4096, backend="cake")
    package = pkg.plan_fp4_mla_decode(**kwargs, max_kv=4096)
    assert public.plan == package.plan and not public.plan.long_rows
    assert public.decode == package.decode and public.merge == package.merge
    rows = 2 * q_len * heads
    a = torch.full((rows, 512), float("nan"), dtype=torch.bfloat16, device=device)
    b = torch.full_like(a, float("nan"))
    _run(public, case, a, sm_scale)
    _run(package, case, b, sm_scale)
    torch.cuda.synchronize()
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    assert prepare_fp4_mla_decode(**kwargs).plan.long_rows
