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

"""CPU-only tests of the Cake GDN non-CP prefill host route.

The route reads nothing back from the device: these tests pin the host-side
contract -- the ``cu_seqlens_cpu`` validation, the single host-known grid rule,
the int32 metadata path and the public forwarding of ``max_seqlen`` /
``cu_seqlens_cpu`` -- without a GPU.  The GPU behaviour (zero-sync proof, CUDA
Graph capture and replay, chunked state carry) lives in
``test_cake_gdn_prefill_gpu.py``.
"""

import pytest
import torch

from flashinfer import gdn_prefill
from flashinfer.jit import cake_gdn

_CPU_OFFSETS_CASES = (
    pytest.param([0, 128, 192], torch.int32, id="int32-tensor"),
    pytest.param([0, 128, 192], torch.int64, id="int64-tensor"),
    pytest.param([0, 128, 192], None, id="list"),
    pytest.param((0, 128, 192), None, id="tuple"),
    pytest.param([0, 0, 192], None, id="leading-empty-sequence"),
    pytest.param([0, 192, 192], None, id="trailing-empty-sequence"),
)


@pytest.mark.parametrize("values,dtype", _CPU_OFFSETS_CASES)
def test_host_offsets_accept_cpu_tensor_and_int_sequences(values, dtype) -> None:
    host_copy = values if dtype is None else torch.tensor(values, dtype=dtype)
    offsets = gdn_prefill._cake_gdn_prefill_host_offsets(
        host_copy, num_seqs=2, total_tokens=192
    )
    assert offsets == tuple(values)
    assert all(type(value) is int for value in offsets)


@pytest.mark.parametrize(
    "host_copy,match",
    (
        pytest.param([0, 128], "num_seqs \\+ 1 = 3 entries", id="too-short"),
        pytest.param([0, 64, 128, 192], "num_seqs \\+ 1 = 3 entries", id="too-long"),
        pytest.param(
            [1, 128, 192], "monotonic cu_seqlens spanning", id="first-nonzero"
        ),
        pytest.param([0, 128, 191], "monotonic cu_seqlens spanning", id="last-short"),
        pytest.param([0, 128, 193], "monotonic cu_seqlens spanning", id="last-long"),
        pytest.param([0, 150, 100], "monotonic cu_seqlens spanning", id="decreasing"),
        pytest.param([0, True, 192], "not bools", id="bool-entry"),
        pytest.param([0, 128.0, 192], "sequence of ints", id="float-entry"),
        pytest.param(
            torch.tensor([[0, 128, 192]], dtype=torch.int32),
            "one-dimensional int32/int64 CPU tensor",
            id="rank-2-tensor",
        ),
        pytest.param(
            torch.tensor([0.0, 128.0, 192.0]),
            "one-dimensional int32/int64 CPU tensor",
            id="float-tensor",
        ),
        pytest.param(
            torch.empty(3, dtype=torch.int32, device="meta"),
            "one-dimensional int32/int64 CPU tensor",
            id="non-cpu-tensor",
        ),
    ),
)
def test_host_offsets_fail_closed_on_the_host(host_copy, match) -> None:
    # Every rejection happens before any launch; a non-CPU tensor is refused
    # rather than copied (the copy would be the device-to-host sync).
    with pytest.raises(cake_gdn.CakeGDNUnsupportedError, match=match):
        gdn_prefill._cake_gdn_prefill_host_offsets(
            host_copy, num_seqs=2, total_tokens=192
        )


def test_grid_rule_mirrors_the_launch_contract() -> None:
    """The single host-known persistent-grid rule (LANE_CONTRACT v1).

    grid_x(total_tiles, sm, dvsplit, bound): dvsplit or tiles <= 128 ->
    min(sm, tiles); else G = min(sm, tiles) when ceil(tiles / G) <
    ceil(tiles / 128), else 128 (ties keep 128).  The bound is validated by
    the caller but does not change the grid.
    """

    grid = gdn_prefill._cake_gdn_prefill_grid_x
    # DV-split: one CTA per tile up to the SM count, whatever the bound.
    assert grid(64, 148, True, 16384) == 64
    assert grid(256, 148, True, 16384) == 148
    assert grid(512, 212, True, 65536) == 212
    # Full-DV, few tiles: one CTA per tile.
    assert grid(128, 148, False, 65536) == 128
    assert grid(96, 160, False, 65536) == 96
    # Full-DV, more than 128 tiles, equal quantum: 128 CTAs whatever the bound.
    assert grid(256, 148, False, 512) == 128
    assert grid(200, 160, False, 1) == 128
    assert grid(130, 212, False, 448) == 130
    assert grid(256, 148, False, 513) == 128
    assert grid(256, 160, False, 16384) == 128
    assert grid(512, 148, False, 65536) == 128
    assert grid(200, 148, False, 1024) == 128
    assert grid(256, 128, False, 4096) == 128
    # Full-DV, the SM count gives a smaller per-CTA tile quantum.
    assert grid(384, 212, False, 8192) == 212  # ceil(384/212) = 2 < ceil(384/128) = 3
    assert grid(300, 160, False, 4096) == 160  # ceil(300/160) = 2 < ceil(300/128) = 3
    assert grid(1024, 152, False, 256) == 152
    assert grid(1024, 148, False, 256) == 148
    assert grid(2048, 152, False, 8192) == 152
    assert grid(129, 212, False, 65536) == 129
    assert grid(130, 148, False, 65536) == 130
    # The bound never changes the grid; the shim's 0 < grid_x <= total_tiles
    # holds for every reachable input.
    for sm in (128, 132, 148, 160, 212):
        for tiles in (1, 2, 64, 128, 129, 200, 256, 384, 512, 1000, 2048):
            for bound in (1, 64, 512, 513, 4096, 65536):
                for dvsplit in (False, True):
                    value = grid(tiles, sm, dvsplit, bound)
                    assert 0 < value <= tiles
                    assert value <= max(sm, 128)


def test_grid_rule_is_the_only_grid_text_in_the_route() -> None:
    # The route must call the mirrored rule and nothing else decides grid_x
    # (the exporter's AST mirror test compares this function with Cake's).
    import inspect

    source = inspect.getsource(gdn_prefill._run_cake_gdn_prefill)
    assert "_cake_gdn_prefill_grid_x(" in source
    assert "max_chunks" not in source
    assert ".tolist()" not in source
    assert ".cpu()" not in source
    assert ".item()" not in source


def test_int32_metadata_is_passed_through_and_int64_is_cast_per_call() -> None:
    i32 = torch.tensor([0, 64, 128], dtype=torch.int32)
    assert gdn_prefill._cake_gdn_prefill_i32(i32) is i32
    i64 = torch.tensor([0, 64, 128], dtype=torch.int64)
    first = gdn_prefill._cake_gdn_prefill_i32(i64)
    second = gdn_prefill._cake_gdn_prefill_i32(i64)
    assert first.dtype == second.dtype == torch.int32
    assert torch.equal(first, i64.to(torch.int32))
    assert first is not second, "no identity-keyed cache"
    # The identity-keyed caches of the previous adapter are gone.
    for name in (
        "_cake_gdn_host_ints",
        "_CAKE_GDN_HOST_INTS",
        "_cake_gdn_i32",
        "_CAKE_GDN_I32_COPIES",
        "_cake_gdn_prefill_seq_lens",
    ):
        assert not hasattr(gdn_prefill, name), name


def _public_dispatch_inputs() -> tuple[torch.Tensor, ...]:
    q = torch.zeros((2048, 2, 128), dtype=torch.float16)
    k = torch.zeros_like(q)
    v = torch.zeros((2048, 8, 128), dtype=torch.float16)
    cu_seqlens = torch.tensor([0, 1024, 2048], dtype=torch.int32)
    return q, k, v, cu_seqlens


def _route_public_cake_non_cp(monkeypatch: pytest.MonkeyPatch) -> dict:
    captured: dict = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return kwargs["output"]

    monkeypatch.setattr(gdn_prefill, "_CAKE_GDN_AVAILABLE", True)
    monkeypatch.setattr(gdn_prefill, "_run_cake_gdn_prefill", fake_run)
    monkeypatch.setattr(gdn_prefill, "get_device_sm_count", lambda _device: 148)
    monkeypatch.setattr(gdn_prefill, "get_compute_capability", lambda _device: (10, 0))
    monkeypatch.setattr(gdn_prefill, "get_device_name", lambda _device: "test GPU")
    monkeypatch.setattr(gdn_prefill.torch.version, "cuda", "13.0")
    monkeypatch.setattr(
        gdn_prefill, "should_use_cp_host", lambda *_args, **_kwargs: False
    )
    return captured


def test_public_api_forwards_cu_seqlens_cpu_and_the_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _route_public_cake_non_cp(monkeypatch)
    q, k, v, cu_seqlens = _public_dispatch_inputs()
    host_copy = [0, 1024, 2048]
    output = gdn_prefill.chunk_gated_delta_rule(
        q,
        k,
        v,
        cu_seqlens=cu_seqlens,
        use_cp=False,
        backend="cake_gdn",
        max_seqlen=1024,
        cu_seqlens_cpu=host_copy,
    )
    assert output.shape == (2048, 8, 128)
    assert captured["cu_seqlens_cpu"] is host_copy
    assert captured["max_seqlen"] == 1024
    assert captured["cu_seqlens"] is cu_seqlens


def test_public_api_defaults_the_bound_to_the_packed_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _route_public_cake_non_cp(monkeypatch)
    q, k, v, cu_seqlens = _public_dispatch_inputs()
    gdn_prefill.chunk_gated_delta_rule(
        q, k, v, cu_seqlens=cu_seqlens, use_cp=False, backend="cake_gdn"
    )
    assert captured["cu_seqlens_cpu"] is None
    assert captured["max_seqlen"] == 2048


def test_cu_seqlens_cpu_is_keyword_only() -> None:
    import inspect

    signature = inspect.signature(gdn_prefill.chunk_gated_delta_rule)
    parameter = signature.parameters["cu_seqlens_cpu"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
    assert "cu_seqlens_cpu :" in gdn_prefill.chunk_gated_delta_rule.__doc__


def test_resolver_signature_has_no_device_derived_lengths() -> None:
    import inspect

    parameters = inspect.signature(cake_gdn.select_cake_gdn_prefill_variant).parameters
    assert "seq_lens" not in parameters
    assert "max_seq_len" in parameters
