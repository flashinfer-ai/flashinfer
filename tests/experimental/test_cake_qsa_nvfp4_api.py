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
value is refused -- one device run of the route against the package, bit for bit, and
the NVFP4 paged writer against the torch port the backend suite fills its caches with.
It also holds the host's program rule on the CPU: which records a call launches for its
form, architecture and grid class, and when a grid is wide (above one resident wave with
at least 16 tiles per CTA).
"""

import contextlib
import inspect
import warnings
from pathlib import Path

import pytest
import torch

import flashinfer
import flashinfer.experimental.qsa_nvfp4.backend as existing_backend
import flashinfer.sparse as public_api
from flashinfer.api_logging import ExperimentalWarning
from flashinfer.experimental.cake_qsa_nvfp4 import cake_backend, cake_jit
from tests.test_helpers.cake_qsa_nvfp4_reference import write_slots

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


def test_program_rule_selects_the_records_of_a_call():
    """On sm_80 a wide (multi-wave, long-stream) split-form call launches the whole-tile
    decode, every other call the K-first decode; the merge follows above one split; the
    fused form is the cooperative decode alone on either architecture; an unknown form,
    architecture or split count is refused."""
    select = cake_jit.select_records
    assert select(form="split", splits=4, arch="sm_80", wide=True) == (
        "decode_whole_tile",
        "merge",
    )
    assert select(form="split", splits=4, arch="sm_80", wide=False) == (
        "decode",
        "merge",
    )
    assert select(form="split", splits=1, arch="sm_80", wide=True) == (
        "decode_whole_tile",
    )
    assert select(form="split", splits=1, arch="sm_80", wide=False) == ("decode",)
    for wide in (False, True):
        assert select(form="split", splits=4, arch="sm_90a", wide=wide) == (
            "decode",
            "merge",
        )
        assert select(form="split", splits=1, arch="sm_90a", wide=wide) == ("decode",)
        for arch in cake_jit.ARCHES:
            assert select(form="fused", splits=4, arch=arch, wide=wide) == (
                "decode_coop",
            )
    for name in ("decode", "decode_whole_tile", "decode_coop", "merge"):
        assert cake_jit.MODULES[name]["stage"] in cake_jit.STAGES
    with pytest.raises(ValueError, match="form"):
        select(form="other", splits=1, arch="sm_80", wide=False)
    with pytest.raises(ValueError, match="arch"):
        select(form="split", splits=1, arch="sm_100a", wide=False)
    with pytest.raises(ValueError, match="splits"):
        select(form="split", splits=0, arch="sm_80", wide=False)
    with pytest.raises(ValueError, match="merge"):
        cake_jit.select_record("merge", form="fused", arch="sm_90a", wide=False)


def test_wide_grid_is_multi_wave_with_long_streams():
    """``select_wide`` at 108 SMs (216 resident CTAs) over the owner's rows at two KV heads
    with the host's split counts and the 33 tiles of a 2051-token selection: rows 128
    (2 splits, 512 CTAs, 17 tiles per CTA) are wide; rows 64 (4 splits, 512 CTAs, 9 tiles
    per CTA) and rows 1-32 (at most one wave) are not; the 2048-row single split (33 tiles
    per CTA) is; the fused form never is.  The grid bound is strict and the stream bound
    inclusive: 16 tiles per CTA (1024 selected tokens) above one wave is wide, 15 is not."""
    sm_count = 108
    tiles = cake_backend.num_tiles(2051)
    assert tiles == 33
    assert cake_jit.DECODE_LONG_STREAM_TILES == 16
    assert cake_backend.LONG_STREAM_TILES == cake_jit.DECODE_LONG_STREAM_TILES
    rows_to_splits = {
        rows: cake_backend.num_splits(rows, 2, 2051, sm_count)
        for rows in (1, 2, 4, 8, 16, 32, 64, 128)
    }
    assert rows_to_splits[64] == 4 and rows_to_splits[128] == 2
    wide = cake_backend.select_wide
    assert wide(128, 2, rows_to_splits[128], tiles, "split", sm_count)
    assert not wide(128, 2, rows_to_splits[128], tiles, "fused", sm_count)
    # Rows 64: 512 CTAs exceed one wave, but ceil(33 / 4) = 9 tiles per CTA is a short stream.
    assert not wide(64, 2, rows_to_splits[64], tiles, "split", sm_count)
    for rows in (1, 2, 4, 8, 16, 32):
        splits = rows_to_splits[rows]
        assert rows * 2 * splits <= cake_backend.CTAS_PER_SM * sm_count, rows
        assert not wide(rows, 2, splits, tiles, "split", sm_count), rows
    assert wide(2048, 2, 1, tiles, "split", sm_count)
    assert not wide(2048, 2, 1, tiles, "fused", sm_count)
    # The grid bound is strict: one full wave of long streams is not wide.
    capacity = cake_backend.CTAS_PER_SM * sm_count
    assert not wide(capacity, 1, 1, 16, "split", sm_count)
    assert wide(capacity + 1, 1, 1, 16, "split", sm_count)
    # The stream bound is inclusive and counts ceil(tiles / splits).
    assert wide(512, 1, 1, 16, "split", sm_count)
    assert not wide(512, 1, 1, 15, "split", sm_count)
    assert wide(128, 2, 2, 31, "split", sm_count)  # ceil(31 / 2) = 16
    assert not wide(128, 2, 2, 30, "split", sm_count)  # 15
    # On 132 SMs (264 CTAs) the owner's rows 1-128 never exceed one wave in the split form.
    for rows in (1, 2, 4, 8, 16, 32, 64, 128):
        splits = cake_backend.num_splits(rows, 2, 2051, 132)
        assert not wide(rows, 2, splits, tiles, "split", 132), rows
    for bad in (
        (0, 2, 1, tiles, "split", sm_count),
        (1, 0, 1, tiles, "split", sm_count),
        (1, 2, 0, tiles, "split", sm_count),
        (1, 2, 1, 0, "split", sm_count),
        (1, 2, 1, tiles, "split", 0),
        (1, 2, 1, tiles, "other", sm_count),
    ):
        with pytest.raises(ValueError):
            wide(*bad)


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


def test_writer_format_is_the_port_the_backend_suite_uses():
    """The production NVFP4 paged writer and the backend suite's torch port of it write the
    same format: equal block-scale bytes, equal signs, and codes that agree or differ by
    one ladder step.  The writer's reciprocal of the scale is nvcc's fast-math one (a unit
    low at times), so a product exactly at a ladder midpoint rounds down in the writer
    where the port rounds to even; a layout, nibble-order or scale mismatch would move
    most codes by more than one step.  The writer's cache then runs through the Cake
    route."""
    device = _device_or_skip()
    blocks, page, num_kv_heads = 40, 16, 2
    slots = blocks * page
    gen = torch.Generator(device=device).manual_seed(7)
    k, v = (
        torch.randn(slots, num_kv_heads, HEAD_DIM, device=device, generator=gen)
        .mul(scale)
        .bfloat16()
        for scale in (1.0, 3.0)
    )
    scales = tuple(x.abs().max().item() / (6 * 448) for x in (k, v))
    caches = []
    for layout in ("NHD", "HND"):
        dims = (num_kv_heads, page) if layout == "HND" else (page, num_kv_heads)
        for writer in ("flashinfer", "port"):
            data = tuple(
                torch.zeros(
                    blocks, *dims, HEAD_DIM // 2, dtype=torch.uint8, device=device
                )
                for _ in range(2)
            )
            sf = tuple(
                torch.zeros(
                    blocks,
                    *dims,
                    HEAD_DIM // 16,
                    dtype=torch.float8_e4m3fn,
                    device=device,
                )
                for _ in range(2)
            )
            if writer == "flashinfer":
                flashinfer.nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
                    k,
                    v,
                    torch.arange(slots, device=device),
                    data,
                    sf,
                    *scales,
                    kv_layout=layout,
                )
            else:
                write_slots(k, v, data, sf, *scales, kv_layout=layout)
            caches.append((layout, writer, data, sf))
        (_, _, data_fi, sf_fi), (_, _, data_port, sf_port) = caches[-2:]
        torch.cuda.synchronize()
        for side in range(2):
            sf_bytes = tuple(x[side].view(torch.uint8) for x in (sf_fi, sf_port))
            assert torch.equal(*sf_bytes), (layout, side, "block scales")
            for shift in (0, 4):
                code_fi = (data_fi[side] >> shift) & 15
                code_port = (data_port[side] >> shift) & 15
                assert torch.equal(code_fi >> 3, code_port >> 3), (layout, side, "sign")
                step = ((code_fi & 7).int() - (code_port & 7).int()).abs()
                assert step.max().item() <= 1, (layout, side, shift, step.max().item())
                differing = (step != 0).sum().item()
                assert differing <= step.numel() * 1e-2, (layout, side, differing)
    # The writer's bytes through the Cake route, in the NHD layout.
    _, _, data, sf = caches[0]
    rows, num_qo_heads, width, table_width = 4, 24, 200, 16
    q, gate = (
        torch.randn(
            rows, num_qo_heads, HEAD_DIM, device=device, generator=gen
        ).bfloat16()
        for _ in range(2)
    )
    table = torch.randperm(blocks, device=device, generator=gen)[: 2 * table_width]
    table = table.view(2, table_width).int()
    indices = torch.full((rows, width + 1), -1, dtype=torch.int32, device=device)
    for row in range(rows):
        tokens = torch.randperm(table_width * page, device=device, generator=gen)
        indices[row, :width] = tokens[:width].int()
    indices[:, width] = width
    with _quiet():
        out = public_api.qsa_sparse_paged_attention(
            q,
            data[0],
            data[1],
            indices,
            table,
            torch.tensor([0, 1, 1, 0], dtype=torch.int32, device=device),
            False,
            None,
            *scales,
            output_gate=gate,
            k_sf=sf[0],
            v_sf=sf[1],
            backend="cake",
        )
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all() and out.any()
