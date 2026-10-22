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

flashinfer public-API tests of the Cake dense NVFP4 MLA decode on Rubin
(``flashinfer.experimental.cake_rubin_mla_nvfp4_decode``).

The backend suite ``test_cake_rubin_mla_nvfp4_decode.py`` imports only the package.  This file covers what needs
flashinfer itself: the public entry points ``flashinfer.mla.cake_mla_nvfp4_paged_decode`` /
``prepare_cake_mla_nvfp4_paged_decode`` / ``quantize_mla_nvfp4_query`` (their experimental marking, the
capability dispatch between the SM100 / SM103 and the SM107 packages, and the keywords they forward).
"""

import inspect
import math
import warnings
from pathlib import Path

import pytest
import torch

import flashinfer.mla as public_api
import flashinfer.mla._core as core
from flashinfer.api_logging import ExperimentalWarning
from flashinfer.experimental.cake_rubin_mla_nvfp4_decode import cake_backend, cake_jit

ENTRY_POINTS = (
    public_api.cake_mla_nvfp4_paged_decode,
    public_api.prepare_cake_mla_nvfp4_paged_decode,
    public_api.quantize_mla_nvfp4_query,
)
RUBIN_KEYWORDS = (
    "seq_lens_host",
    "kv_len_global_host",
    "cum_seq_lens_q_host",
    "cp_interleave",
    "units",
    "pairs",
)


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    if (
        cake_jit.arch_for_capability(tuple(torch.cuda.get_device_capability(index)))
        is None
    ):
        pytest.skip("not an exported compute capability")
    if not (Path(cake_jit.__file__).resolve().parent / "build.json").exists():
        pytest.skip("the package has no build.json yet (not exported)")
    return torch.device("cuda", index)


def test_entry_points_are_marked_experimental() -> None:
    for fn in ENTRY_POINTS:
        assert getattr(fn, "is_experimental", False) is True, fn.__name__
        assert "SM107" in fn.experimental_feature and "SM100" in fn.experimental_feature
        assert ".. warning::" in fn.__doc__


def test_signatures_take_the_rubin_keywords() -> None:
    for fn in ENTRY_POINTS[:2]:
        params = inspect.signature(fn).parameters
        for name in RUBIN_KEYWORDS:
            assert (
                name in params and params[name].kind is inspect.Parameter.KEYWORD_ONLY
            ), (fn.__name__, name)
        assert params["cp_interleave"].default == 1
        assert all(
            params[n].default is None for n in RUBIN_KEYWORDS if n != "cp_interleave"
        )
    # The backend's own entry points take the same keywords (the routing forwards them one for one).
    for fn in (
        cake_backend.cake_mla_nvfp4_paged_decode,
        cake_backend.prepare_cake_mla_nvfp4_paged_decode,
    ):
        params = inspect.signature(fn).parameters
        assert all(n in params for n in RUBIN_KEYWORDS), fn.__name__


def test_dispatch_forwards_every_keyword_to_the_rubin_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def recorder(name):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        return fake

    monkeypatch.setattr(core, "_cake_nvfp4_mla_is_rubin", lambda device: True)
    monkeypatch.setattr(cake_backend, "cake_mla_nvfp4_paged_decode", recorder("decode"))
    monkeypatch.setattr(
        cake_backend, "prepare_cake_mla_nvfp4_paged_decode", recorder("prepare")
    )
    monkeypatch.setattr(cake_backend, "quantize_mla_nvfp4_query", recorder("quantize"))
    t = torch.empty(0)
    operands = (t,) * 10  # q_nope .. seq_lens, workspace_buffer
    extra = dict(seq_lens_host=[1, 2], kv_len_global_host=[3, 4], cum_seq_lens_q_host=[0, 1, 2], cp_interleave=4,
                 units=46, pairs=14)  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        assert (
            public_api.cake_mla_nvfp4_paged_decode(
                *operands, sm_scale=0.5, ckv_scale=0.25, out=t, **extra
            )
            == "decode"
        )
        assert (
            public_api.prepare_cake_mla_nvfp4_paged_decode(
                *operands[:9], t, t, sm_scale=0.5, ckv_scale=0.25, **extra
            )
            == "prepare"
        )
        assert (
            public_api.quantize_mla_nvfp4_query(t, 0.25, 0.5, out=(t, t, t, t))
            == "quantize"
        )
    assert [c[0] for c in calls] == ["decode", "prepare", "quantize"]
    decode_kwargs = calls[0][2]
    for name, value in extra.items():
        assert decode_kwargs[name] == value and calls[1][2][name] == value, name
    assert (
        decode_kwargs["backend"] == "cake"
        and decode_kwargs["out"] is t
        and decode_kwargs["return_lse"] is False
    )
    assert calls[0][1] == operands and calls[1][1] == operands[:9] + (t, t)
    assert calls[2][1] == (t, 0.25, 0.5) and calls[2][2] == {"out": (t, t, t, t)}


def test_dispatch_keeps_the_sm100_path_free_of_rubin_only_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flashinfer.experimental.cake_mla_nvfp4_paged_decode import (
        cake_backend as blackwell,
    )

    calls = []
    monkeypatch.setattr(core, "_cake_nvfp4_mla_is_rubin", lambda device: False)
    monkeypatch.setattr(
        blackwell,
        "cake_mla_nvfp4_paged_decode",
        lambda *a, **k: calls.append(("decode", k)) or "sm100",
    )
    monkeypatch.setattr(
        blackwell,
        "CakeMlaNvfp4PagedDecode",
        lambda **k: calls.append(("prepare", k)) or "sm100",
    )
    t = torch.empty(0)
    operands = (t,) * 10
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        # The host lists and cluster counts are hints the SM100 / SM103 package does not need: accepted, not passed.
        assert (
            public_api.cake_mla_nvfp4_paged_decode(
                *operands,
                sm_scale=0.5,
                ckv_scale=0.25,
                out=t,
                seq_lens_host=[1],
                units=46,
                pairs=14,
            )
            == "sm100"
        )
        assert (
            public_api.prepare_cake_mla_nvfp4_paged_decode(
                *operands[:9], t, t, sm_scale=0.5, ckv_scale=0.25
            )
            == "sm100"
        )
        # A key interleave other than 1 changes the sharding: served by the Rubin package only.
        with pytest.raises(ValueError, match="cp_interleave"):
            public_api.cake_mla_nvfp4_paged_decode(
                *operands, sm_scale=0.5, ckv_scale=0.25, out=t, cp_interleave=2
            )
        with pytest.raises(ValueError, match="cp_interleave"):
            public_api.prepare_cake_mla_nvfp4_paged_decode(
                *operands[:9], t, t, sm_scale=0.5, ckv_scale=0.25, cp_interleave=2
            )
    assert [c[0] for c in calls] == ["decode", "prepare"]
    assert not any(k in calls[0][1] for k in RUBIN_KEYWORDS) and not any(
        k in calls[1][1] for k in RUBIN_KEYWORDS
    )


def test_routing_predicate_reads_the_capability() -> None:
    assert (
        core._CAKE_NVFP4_MLA_RUBIN_CAPABILITY
        == (10, 7)
        == next(iter(cake_jit.ARCH_BY_CAPABILITY))
    )
    assert core._cake_nvfp4_mla_is_rubin(torch.device("cpu")) is False
    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
        expected = tuple(torch.cuda.get_device_capability(device)) == (10, 7)
        assert core._cake_nvfp4_mla_is_rubin(device) is expected
        assert cake_backend.is_supported(device) is expected


def test_public_entries_run_the_rubin_backend() -> None:
    device = _device_or_skip()
    batch, heads, kv = 2, 16, [1000, 300]
    gen = torch.Generator(device=device).manual_seed(11)
    pages = [-(-k // 64) for k in kv]
    pool = sum(pages) + 1
    ckv_cache = torch.randint(
        0, 256, (pool, 64, 256), dtype=torch.int32, device=device, generator=gen
    ).to(torch.uint8)
    sf_bytes = torch.tensor(
        [0x30, 0x34, 0x38, 0x3C, 0x40], dtype=torch.uint8, device=device
    )
    ckv_sf_cache = sf_bytes[
        torch.randint(0, 5, (pool, 64, 32), device=device, generator=gen)
    ].view(torch.float8_e4m3fn)
    kpe_cache = (
        torch.randint(
            0, 120, (pool, 64, 64), dtype=torch.int32, device=device, generator=gen
        )
        .to(torch.uint8)
        .view(torch.float8_e4m3fn)
    )
    block_tables = torch.zeros((batch, max(pages)), dtype=torch.int32, device=device)
    block_tables[0, : pages[0]] = torch.arange(
        pages[0], dtype=torch.int32, device=device
    )
    block_tables[1, : pages[1]] = torch.arange(
        pages[0], pages[0] + pages[1], dtype=torch.int32, device=device
    )
    seq_lens = torch.tensor(kv, dtype=torch.int32, device=device)
    q = torch.randn((batch, 1, heads, 576), generator=gen, device=device).to(
        torch.bfloat16
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        qws = torch.zeros(
            cake_backend.query_workspace_bytes(batch * heads),
            dtype=torch.uint8,
            device=device,
        )
        got = public_api.quantize_mla_nvfp4_query(
            q,
            1.0,
            1.0,
            out=cake_backend.mla_nvfp4_query_buffers(qws, (batch, 1, heads)),
        )
        qws2 = torch.zeros_like(qws)
        want = cake_backend.quantize_mla_nvfp4_query(
            q,
            1.0,
            1.0,
            out=cake_backend.mla_nvfp4_query_buffers(qws2, (batch, 1, heads)),
        )
        torch.cuda.synchronize()
        for g, w in zip(got, want, strict=True):
            assert torch.equal(g.view(torch.uint8), w.view(torch.uint8))
        q_nope, q_sf, q_rope, q_scale = got
        out = torch.full(
            (batch, 1, heads, 512), math.nan, dtype=torch.bfloat16, device=device
        )
        lse = torch.full(
            (batch, 1, heads), math.nan, dtype=torch.float32, device=device
        )
        workspace = torch.zeros(
            cake_backend.max_workspace_bytes(batch * heads, batch),
            dtype=torch.uint8,
            device=device,
        )
        runner = public_api.prepare_cake_mla_nvfp4_paged_decode(
            q_nope, q_sf, q_rope, q_scale, ckv_cache, ckv_sf_cache, kpe_cache, block_tables, seq_lens, out,
            workspace, sm_scale=1.0 / math.sqrt(192), ckv_scale=1.0, lse=lse, seq_lens_host=kv,
        )  # fmt: skip
        assert isinstance(runner, cake_backend.CakeMlaNvfp4PagedDecode)
        assert runner.route_metadata["arch"] == "sm_107a"
        runner.launch()
        torch.cuda.synchronize()
        first, first_lse = out.clone(), lse.clone()
        out2 = torch.full_like(out, math.nan)
        lse2 = torch.full_like(lse, math.nan)
        result = public_api.cake_mla_nvfp4_paged_decode(
            q_nope, q_sf, q_rope, q_scale, ckv_cache, ckv_sf_cache, kpe_cache, block_tables, seq_lens,
            torch.zeros_like(workspace), sm_scale=1.0 / math.sqrt(192), ckv_scale=1.0, out=out2, lse=lse2,
            return_lse=True,
        )  # fmt: skip
        torch.cuda.synchronize()
    assert result == (out2, lse2)
    assert torch.isfinite(first.float()).all() and torch.isfinite(first_lse).all()
    assert torch.equal(first, out2) and torch.equal(first_lse, lse2)
