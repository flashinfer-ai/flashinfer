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

flashinfer public-API tests of the Cake QSA NVFP4 sparse paged attention
(``flashinfer.experimental.cake_qsa_nvfp4``).

The backend suite ``test_cake_qsa_nvfp4.py`` exercises the package.  This file covers
what needs flashinfer itself: the ``backend`` keyword of
``flashinfer.sparse.qsa_sparse_paged_attention`` -- ``backend="cake"`` reaches the
package with every operand, the default keeps the existing implementation, an unknown
value is refused -- and one device run of the route against the package, bit for bit.
"""

import contextlib
import inspect
import warnings
from pathlib import Path

import pytest
import torch

import flashinfer.experimental.qsa_nvfp4.backend as existing_backend
import flashinfer.sparse as public_api
from flashinfer.api_logging import ExperimentalWarning
from flashinfer.experimental.cake_qsa_nvfp4 import cake_backend, cake_jit

PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
HEAD_DIM = 256


@contextlib.contextmanager
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        yield


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    if cake_jit.arch_for_capability(capability) is None:
        pytest.skip(
            f"compute capability {capability} is not an exported target {cake_jit.ARCHES}"
        )
    if not (PACKAGE_DIR / "build.json").exists():
        pytest.skip("the package has no build.json yet (not exported)")
    return torch.device("cuda", index)


def _recorder(calls, tag):
    def run(*args, **kwargs):
        calls.append((tag, args, kwargs))
        return tag

    return run


def _patch_both_backends(monkeypatch):
    """Replace both implementations' entries by recorders; returns the call log."""
    calls = []
    monkeypatch.setattr(
        cake_backend, "qsa_sparse_paged_attention", _recorder(calls, "cake")
    )
    monkeypatch.setattr(
        existing_backend, "qsa_sparse_paged_attention", _recorder(calls, "existing")
    )
    return calls


def test_entry_point_is_marked_experimental():
    assert public_api.qsa_sparse_paged_attention.is_experimental


def test_backend_is_a_keyword_with_a_default():
    parameter = inspect.signature(public_api.qsa_sparse_paged_attention).parameters[
        "backend"
    ]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is not inspect.Parameter.empty


def test_unknown_backend_is_rejected(monkeypatch):
    calls = _patch_both_backends(monkeypatch)
    t = torch.empty(0)
    with pytest.raises(ValueError, match="backend"), _quiet():
        public_api.qsa_sparse_paged_attention(
            t, t, t, t, t, t, False, output_gate=t, k_sf=t, v_sf=t, backend="other"
        )
    assert calls == []


def test_cake_backend_reaches_the_package_with_every_operand(monkeypatch):
    signature = inspect.signature(cake_backend.qsa_sparse_paged_attention)
    calls = _patch_both_backends(monkeypatch)
    q, k_cache, v_cache, indices, table, token_to_req, out, gate, k_sf, v_sf = (
        torch.empty(i) for i in range(10)
    )
    with _quiet():
        result = public_api.qsa_sparse_paged_attention(
            q,
            k_cache,
            v_cache,
            indices,
            table,
            token_to_req,
            True,
            out,
            0.5,
            0.25,
            output_gate=gate,
            k_sf=k_sf,
            v_sf=v_sf,
            backend="cake",
        )
    assert result == "cake"
    assert [tag for tag, _, _ in calls] == ["cake"]
    _, args, kwargs = calls[0]
    received = signature.bind(*args, **kwargs).arguments
    tensors = dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        logical_indices=indices,
        block_table=table,
        token_to_req=token_to_req,
        out=out,
        output_gate=gate,
        k_sf=k_sf,
        v_sf=v_sf,
    )
    for name, value in tensors.items():
        assert received[name] is value, name
    assert received["use_prefill_config"] is True
    assert received["k_scale"] == 0.5 and received["v_scale"] == 0.25


def test_default_backend_keeps_the_existing_implementation(monkeypatch):
    calls = _patch_both_backends(monkeypatch)
    t = torch.empty(0)
    with _quiet():
        result = public_api.qsa_sparse_paged_attention(
            t, t, t, t, t, t, False, output_gate=t, k_sf=t, v_sf=t
        )
    assert result == "existing"
    assert [tag for tag, _, _ in calls] == ["existing"]


def _operands(device):
    """A small random fixture: 4 rows of 24 x 2 heads over 40 blocks of 16 tokens, two
    requests of 16 pages, 200 distinct selections per row (4 tiles); random E2M1 codes
    and finite E4M3 block scales from {0.5, 0.75, 1, 1.5, 2}."""
    rows, num_qo_heads, num_kv_heads = 4, 24, 2
    blocks, page, table_width, width = 40, 16, 16, 200
    gen = torch.Generator(device=device).manual_seed(3)
    q, gate = (
        torch.randn(
            rows, num_qo_heads, HEAD_DIM, device=device, generator=gen
        ).bfloat16()
        for _ in range(2)
    )
    data = tuple(
        torch.randint(
            0,
            256,
            (blocks, page, num_kv_heads, HEAD_DIM // 2),
            dtype=torch.int32,
            device=device,
            generator=gen,
        ).to(torch.uint8)
        for _ in range(2)
    )
    sf_bytes = torch.tensor(
        [0x30, 0x34, 0x38, 0x3C, 0x40], dtype=torch.uint8, device=device
    )
    sf = tuple(
        sf_bytes[
            torch.randint(
                0,
                5,
                (blocks, page, num_kv_heads, HEAD_DIM // 16),
                device=device,
                generator=gen,
            )
        ].view(torch.float8_e4m3fn)
        for _ in range(2)
    )
    table = torch.randperm(blocks, device=device, generator=gen)[: 2 * table_width]
    table = table.view(2, table_width).int()
    token_to_req = torch.tensor([0, 1, 1, 0], dtype=torch.int32, device=device)
    indices = torch.full((rows, width + 1), -1, dtype=torch.int32, device=device)
    for row in range(rows):
        tokens = torch.randperm(table_width * page, device=device, generator=gen)
        indices[row, :width] = tokens[:width].int()
    indices[:, width] = width
    return dict(
        q=q,
        k_cache=data[0],
        v_cache=data[1],
        logical_indices=indices,
        block_table=table,
        token_to_req=token_to_req,
        use_prefill_config=False,
        k_scale=0.05,
        v_scale=0.05,
        output_gate=gate,
        k_sf=sf[0],
        v_sf=sf[1],
    )


def test_cake_route_runs_the_package():
    device = _device_or_skip()
    operands = _operands(device)
    want = cake_backend.qsa_sparse_paged_attention(**operands)
    with _quiet():
        got = public_api.qsa_sparse_paged_attention(**operands, backend="cake")
    torch.cuda.synchronize()
    assert torch.isfinite(got.float()).all() and got.any()
    assert torch.equal(got, want)
    out = torch.empty_like(want)
    with _quiet():
        result = public_api.qsa_sparse_paged_attention(
            **operands, out=out, backend="cake"
        )
    assert result is out
    assert torch.equal(out, want)
