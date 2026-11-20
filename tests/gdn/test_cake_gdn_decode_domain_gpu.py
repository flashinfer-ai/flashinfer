# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU tests of the locality-domain twin of the Cake nontranspose FP32 T=1 decode
on a dual-die SM107a GPU (a CUDA 13.5 toolkit and driver): on a state homed by
``localize_gdn_decode_state`` (or ``split_empty``) the twin's output and state are
the whole-GPU ``hqsplit_tma3`` body's bit for bit, for any cut, rows0 and grid;
its claim counters are left zero and reused; every state the twin does not serve
runs the whole-GPU body bit for bit; the public call is capturable.  The race gate
runs when ``FLASHINFER_CAKE_GDN_DOMAIN_RACE_LAUNCHES`` gives the launches per
case."""

import os

import pytest
import torch

import flashinfer.cake_gdn_decode_locality as locality
import flashinfer.gdn_decode as gdn_decode
from flashinfer import cake_locality
from flashinfer.gdn_decode import (
    gated_delta_rule_decode,
    localize_gdn_decode_state,
)
from flashinfer.jit import cake_gdn

_HEAD = 128
_SCALE = _HEAD**-0.5
_H, _HV = 16, 32
_ROW_BYTES = _HV * _HEAD * _HEAD * 4
_WHOLE = "gdn_decode_nontranspose_fp32_t1_hqsplit_tma3"
_RACE_ENV = "FLASHINFER_CAKE_GDN_DOMAIN_RACE_LAUNCHES"


def _sm_count(device: int) -> int:
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def _require_locality_device() -> int:
    """A dual-die SM107a device with a CUDA 13.5 toolkit and driver."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.cuda.current_device()
    if torch.cuda.get_device_capability(device) != (10, 7):
        pytest.skip("the locality-domain twin runs on SM107a")
    from flashinfer.jit.cpp_ext import get_cuda_version
    from flashinfer.locality_domain import locality_domain_unsupported_reason

    version = get_cuda_version()
    if (version.major, version.minor) < cake_locality.LOCALITY_CUDA:
        pytest.skip(f"the twin needs a CUDA 13.5 toolkit, found {version}")
    reason = locality_domain_unsupported_reason(device)
    if reason is not None:
        pytest.skip(f"CUDA locality domains are unavailable: {reason}")
    driver = int(cake_locality._module().driver_version())
    if (driver // 1000, driver % 1000 // 10) < cake_locality.LOCALITY_CUDA:
        pytest.skip(f"the twin needs a CUDA 13.5 driver, found {driver}")
    return device


def _require_domain():
    """The device and its SM table, on an SM count with listed cells; past the
    skips, a build or probe failure is a test failure."""
    device = _require_locality_device()
    sm_count = _sm_count(device)
    if (
        "sm_107a",
        sm_count,
    ) not in cake_gdn.CAKE_GDN_NONTRANSPOSE_FP32_T1_DOMAIN_ROUTES:
        pytest.skip(f"no locality-domain cell is listed for {sm_count} SMs")
    table = cake_locality.sm_table(device)
    assert table is not None, "the SM table probe failed on a dual-die SM107a GPU"
    assert sum(table.counts) == sm_count
    return device, table


def _inputs(batch_size: int, *, seed: int = 2026, heads=(_H, _HV)):
    torch.manual_seed(seed + batch_size)
    num_q_heads, num_v_heads = heads
    device = torch.device("cuda")
    bf16, fp32 = torch.bfloat16, torch.float32
    tensors = {
        "q": torch.randn(batch_size, 1, num_q_heads, _HEAD, device=device, dtype=bf16),
        "k": torch.randn(batch_size, 1, num_q_heads, _HEAD, device=device, dtype=bf16),
        "v": torch.randn(batch_size, 1, num_v_heads, _HEAD, device=device, dtype=bf16),
        "A_log": torch.randn(num_v_heads, device=device, dtype=fp32) * 0.1,
        "a": (torch.randn(batch_size, 1, num_v_heads, device=device) * 0.1).to(bf16),
        "dt_bias": torch.randn(num_v_heads, device=device, dtype=fp32) * 0.1,
        "b": torch.randn(batch_size, 1, num_v_heads, device=device, dtype=bf16),
    }
    state = (
        torch.randn(batch_size, num_v_heads, _HEAD, _HEAD, device=device, dtype=fp32)
        * 0.01
    )
    return tensors, state.contiguous()


def _reference(tensors, initial_state, scale):
    ratio = tensors["v"].shape[2] // tensors["q"].shape[2]

    def l2norm(value):
        value = value.float()
        return value / torch.sqrt(torch.sum(value * value, dim=-1, keepdim=True) + 1e-6)

    q = l2norm(tensors["q"].squeeze(1).repeat_interleave(ratio, dim=1)) * scale
    k = l2norm(tensors["k"].squeeze(1).repeat_interleave(ratio, dim=1))
    v = tensors["v"].squeeze(1).float()
    alpha = torch.exp(
        -torch.exp(tensors["A_log"])
        * torch.nn.functional.softplus(
            tensors["a"].squeeze(1).float() + tensors["dt_bias"]
        )
    )
    beta = torch.sigmoid(tensors["b"].squeeze(1).float())
    decayed = initial_state.float() * alpha[:, :, None, None]
    v_delta = (v - torch.einsum("bhk,bhkv->bhv", k, decayed)) * beta[:, :, None]
    state = decayed + k.unsqueeze(-1) * v_delta.unsqueeze(-2)
    out = torch.einsum("bhk,bhkv->bhv", q, state)
    return out.unsqueeze(1).to(torch.bfloat16), state


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(_bits(a), _bits(b))


def _empty_out(batch_size: int, num_v_heads: int = _HV) -> torch.Tensor:
    return torch.full(
        (batch_size, 1, num_v_heads, _HEAD),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )


def _whole_name(batch_size: int, scale: float, heads=(_H, _HV)) -> str:
    num_q_heads, num_v_heads = heads
    route = cake_gdn.select_cake_gdn_decode_variant(
        arch="sm_107a",
        batch_size=batch_size,
        io_dtype="bfloat16",
        state_dtype="float32",
        head_size=_HEAD,
        layout="nontranspose",
        num_k_heads=num_q_heads,
        num_q_heads=num_q_heads,
        num_v_heads=num_v_heads,
        scale=scale,
        seq_len=1,
        use_qk_l2norm=True,
    )
    return route.variant_name


def _whole(tensors, state, scale):
    """The whole-GPU body's ``(out, state)`` of one call on a fresh ordinary copy of
    ``state`` (a direct launch of the routed record)."""
    batch_size, num_v_heads = int(state.shape[0]), int(state.shape[1])
    name = _whole_name(batch_size, scale, (int(tensors["q"].shape[2]), num_v_heads))
    entry = cake_gdn.load_cake_gdn_kernel(name, "sm_107a")
    grid_x = (
        batch_size
        * num_v_heads
        * cake_gdn.cake_gdn_nontranspose_fp32_t1_ctas_per_state(name)
    )
    ordinary = state.clone()
    out = _empty_out(batch_size, num_v_heads)
    entry(*_call_args(tensors, ordinary, out), grid_x, 1, 1)
    torch.cuda.synchronize()
    return out, ordinary


def _call_args(tensors, state, out):
    return (
        tensors["q"],
        tensors["k"],
        tensors["v"],
        state,
        tensors["A_log"],
        tensors["a"],
        tensors["dt_bias"],
        tensors["b"],
        out,
    )


def _public(tensors, state, scale, *, backend="cake_gdn", out=None):
    out = _empty_out(state.shape[0], state.shape[1]) if out is None else out
    gated_delta_rule_decode(
        q=tensors["q"],
        k=tensors["k"],
        v=tensors["v"],
        state=state,
        A_log=tensors["A_log"],
        a=tensors["a"],
        dt_bias=tensors["dt_bias"],
        b=tensors["b"],
        scale=scale,
        output=out,
        backend=backend,
    )
    return out


def _twin_name(batch_size: int, scale: float, table, device: int) -> str:
    name = locality.nontranspose_domain_variant(
        arch="sm_107a",
        sm_count=_sm_count(device),
        die_class=cake_locality.die_class(table.counts),
        batch_size=batch_size,
        num_q_heads=_H,
        num_k_heads=_H,
        num_v_heads=_HV,
        scale=scale,
    )
    assert name is not None, (batch_size, scale, table.counts)
    return name


def _home(state: torch.Tensor, rows0: int) -> torch.Tensor:
    """``state`` copied into a split allocation with rows ``[0, rows0)`` on die 0
    (skips a cut off the allocation granularity)."""
    device = state.device.index
    nbytes = state.numel() * state.element_size()
    cut = rows0 * (nbytes // state.shape[0])
    if 0 < cut < nbytes and cut % cake_locality.granularity(device):
        pytest.skip(f"a cut after row {rows0} is off the allocation granularity")
    homed = cake_locality.split_empty(tuple(state.shape), state.dtype, cut, device)
    homed.copy_(state)
    return homed


def _domain_launch(device: int, batch_size: int, scale: float, table, homed):
    """The launch the public path resolved for ``homed`` (``None`` if none)."""
    record = cake_locality.tensor_homing(homed)
    assert record is not None
    key = (device, batch_size, _H, _H, _HV, scale, True, table.counts, record.cut)
    return gdn_decode._CAKE_GDN_NONTRANSPOSE_DOMAIN_LAUNCHES.get(key)


# --------------------------------------------------------------------------- bits

# The shape list: the edges of the listed batch classes (48 / 63 / 64 / 127 / 128 /
# 255 / 256), a row inside the first and the last class (56 / 512) and B = 1024, at
# both scales.  Below 48 rows no cell is listed: see
# test_states_the_twin_does_not_serve_run_the_whole_body.
_TWIN_BATCHES = [48, 56, 63, 64, 127, 128, 255, 256, 512, 1024]


@pytest.mark.parametrize("scale", [_SCALE, 1.0])
@pytest.mark.parametrize("batch_size", _TWIN_BATCHES)
def test_twin_matches_the_whole_body_bitwise(
    monkeypatch, batch_size: int, scale: float
) -> None:
    # localize_gdn_decode_state homes the state at the byte cut; the public call on it
    # runs the twin (its launch is resolved, not "no twin") and gives the whole body's
    # output and state bit for bit, within tolerance of the reference; with
    # backend="auto" too.
    device, table = _require_domain()
    tensors, initial = _inputs(batch_size)
    whole_out, whole_state = _whole(tensors, initial, scale)
    homed = localize_gdn_decode_state(initial, _H, scale)
    assert homed is not initial
    assert _same_bits(homed, initial)
    out = _public(tensors, homed, scale)
    torch.cuda.synchronize()
    launch = _domain_launch(device, batch_size, scale, table, homed)
    assert launch is not None and launch[0] is not None
    assert launch[1:3] == (batch_size * _HV, (batch_size + 1) // 2)
    assert _same_bits(out, whole_out)
    assert _same_bits(homed, whole_state)
    if batch_size <= 128:
        expected_out, expected_state = _reference(tensors, initial, scale)
        torch.testing.assert_close(
            out.float(), expected_out.float(), atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(homed, expected_state, atol=1e-3, rtol=1e-3)
    homed.copy_(initial)
    monkeypatch.setattr(gdn_decode, "_CAKE_GDN_NONTRANSPOSE_DOMAIN_LAUNCHES", {})
    monkeypatch.setattr(gdn_decode, "_CAKE_GDN_NONTRANSPOSE_HOMED_LAUNCHES", {})
    out = _public(tensors, homed, scale, backend="auto")
    torch.cuda.synchronize()
    launch = _domain_launch(device, batch_size, scale, table, homed)
    assert launch is not None and launch[0] is not None
    assert _same_bits(out, whole_out)
    assert _same_bits(homed, whole_state)


@pytest.mark.parametrize("scale", [_SCALE, 1.0])
@pytest.mark.parametrize("batch_size", [48, 64, 128, 256, 512])
def test_twin_bits_do_not_depend_on_the_cut(batch_size: int, scale: float) -> None:
    # Every row on die 0, every row on die 1, the byte cut, the SM-count cut and two
    # uneven cuts: the same bits as the whole body.
    device, table = _require_domain()
    tensors, initial = _inputs(batch_size)
    whole_out, whole_state = _whole(tensors, initial, scale)
    cuts = {
        0,
        batch_size,
        (batch_size + 1) // 2,
        cake_locality.split_units(batch_size, table.counts)[0],
        13,
        batch_size - 13,
    }
    for rows0 in sorted(cuts):
        homed = _home(initial, rows0)
        out = _public(tensors, homed, scale)
        torch.cuda.synchronize()
        launch = _domain_launch(device, batch_size, scale, table, homed)
        assert launch is not None and launch[0] is not None and launch[2] == rows0
        assert _same_bits(out, whole_out), rows0
        assert _same_bits(homed, whole_state), rows0
        del homed
    homed = localize_gdn_decode_state(initial, _H, scale, rows0=13)
    assert cake_locality.tensor_homing(homed).cut == 13 * _ROW_BYTES
    out = _public(tensors, homed, scale)
    torch.cuda.synchronize()
    assert _same_bits(out, whole_out) and _same_bits(homed, whole_state)


def test_twin_rows0_and_grid_do_not_change_the_bits() -> None:
    # A direct launch of the twin entry: rows0 off the physical cut and any grid of
    # 1 .. B * HV CTAs (capped by the resident CTAs) give the same bits and leave the
    # counters zero; the shim refuses rows0, grid and table arguments outside its
    # contract before launch, the state untouched.
    device, table = _require_domain()
    batch_size = 64
    tensors, initial = _inputs(batch_size)
    whole_out, whole_state = _whole(tensors, initial, _SCALE)
    entry = cake_gdn.load_cake_gdn_kernel(
        _twin_name(batch_size, _SCALE, table, device), "sm_107a"
    )
    homed = _home(initial, batch_size // 2)
    counters = cake_locality.counters(locality.DOMAIN_COUNTER_WORDS)
    sm_count = _sm_count(device)
    grids = sorted({1, 2, sm_count - 1, sm_count, 4 * sm_count, batch_size * _HV})
    for rows0 in (0, 5, batch_size // 2, batch_size // 2 + 3, batch_size):
        for grid_x in grids:
            homed.copy_(initial)
            out = _empty_out(batch_size)
            entry(
                *_call_args(tensors, homed, out),
                rows0,
                *counters,
                table.entries,
                grid_x,
                1,
                1,
            )
            torch.cuda.synchronize()
            assert _same_bits(out, whole_out), (rows0, grid_x)
            assert _same_bits(homed, whole_state), (rows0, grid_x)
            assert all(int(c.abs().sum()) == 0 for c in counters), (rows0, grid_x)
    homed.copy_(initial)
    torch.cuda.synchronize()

    def refused(error, match, *, rows0=0, grid_x=1, pair=counters, sm=table.entries):
        out = _empty_out(batch_size)
        with pytest.raises(error, match=match):
            entry(*_call_args(tensors, homed, out), rows0, *pair, sm, grid_x, 1, 1)
        torch.cuda.synchronize()
        assert _same_bits(homed, initial)

    refused(ValueError, "rows0 must be in", rows0=-1)
    refused(ValueError, "rows0 must be in", rows0=batch_size + 1)
    refused(ValueError, "grid must be", grid_x=0)
    refused(ValueError, "grid must be", grid_x=batch_size * _HV + 1)
    refused(ValueError, "sm_table must be a CPU tensor", sm=table.entries.cuda())
    refused(ValueError, "sm_table must hold exactly", sm=table.entries[:128])
    refused(TypeError, "counters0", pair=(counters[0].long(), counters[1]))
    refused(
        ValueError,
        "counters1 must be a CUDA tensor",
        pair=(counters[0], counters[1].cpu()),
    )


def test_twin_counters_left_zero_and_reused() -> None:
    # Public calls on one stream reuse that stream's zeroed counter pair (left zero by
    # every launch) and allocate nothing; another stream gets its own pair; the bits
    # are the same on every call.
    device, table = _require_domain()
    batch_size = 128
    tensors, initial = _inputs(batch_size)
    whole_out, whole_state = _whole(tensors, initial, _SCALE)
    homed = localize_gdn_decode_state(initial, _H)
    out = _empty_out(batch_size)
    stream = torch.cuda.Stream()
    # The homing copy and the output fill ran on the current stream.
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _public(tensors, homed, _SCALE, out=out)
        pair = cake_locality.launch_counters(locality.DOMAIN_COUNTER_WORDS, device)
    stream.synchronize()
    assert _same_bits(out, whole_out) and _same_bits(homed, whole_state)
    before = torch.cuda.memory_stats()
    with torch.cuda.stream(stream):
        for _ in range(100):
            homed.copy_(initial)
            _public(tensors, homed, _SCALE, out=out)
        again = cake_locality.launch_counters(locality.DOMAIN_COUNTER_WORDS, device)
    stream.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
    assert again[0] is pair[0] and again[1] is pair[1]
    assert all(int(c.abs().sum()) == 0 for c in pair)
    assert _same_bits(out, whole_out) and _same_bits(homed, whole_state)
    other = torch.cuda.Stream()
    with torch.cuda.stream(other):
        homed.copy_(initial)
        _public(tensors, homed, _SCALE, out=out)
        theirs = cake_locality.launch_counters(locality.DOMAIN_COUNTER_WORDS, device)
    other.synchronize()
    assert theirs[0] is not pair[0]
    assert all(int(c.abs().sum()) == 0 for c in theirs)
    assert _same_bits(out, whole_out) and _same_bits(homed, whole_state)


def test_homed_state_rows_live_on_their_dies() -> None:
    # The rows before the cut are memory of die 0, the rest of die 1.
    device, table = _require_domain()
    batch_size = 64
    _, initial = _inputs(batch_size)
    for rows0 in sorted({0, 13, 32, batch_size}):
        homed = _home(initial, rows0)
        base = homed.data_ptr()
        if rows0 > 0:
            assert cake_locality.pointer_domain(base) == 0
            assert cake_locality.pointer_domain(base + rows0 * _ROW_BYTES - 1) == 0
        if rows0 < batch_size:
            assert cake_locality.pointer_domain(base + rows0 * _ROW_BYTES) == 1
            last = base + batch_size * _ROW_BYTES - 1
            assert cake_locality.pointer_domain(last) == 1
        del homed


# --------------------------------------------------------------------------- fallbacks


def test_states_the_twin_does_not_serve_run_the_whole_body(monkeypatch) -> None:
    # An ordinary state, a copy of a homed state, a prefix view of a homed allocation,
    # a homed state below 48 rows (B 9 / 10 too, where the whole route changes body,
    # and B 32 / 37 / 47 of the large band, which lists no twin below 48 rows; both
    # scales), a homed state without an SM table, a homed state of the H4 / HV8 rows
    # and a homed state of a row without a Cake route each run what an ordinary state
    # runs, bit for bit.
    device, table = _require_domain()
    batch_size = 64
    tensors, initial = _inputs(batch_size)
    whole_out, whole_state = _whole(tensors, initial, _SCALE)
    monkeypatch.setattr(gdn_decode, "_CAKE_GDN_NONTRANSPOSE_DOMAIN_LAUNCHES", {})

    def no_twin():
        assert not any(
            launch[0] is not None
            for launch in gdn_decode._CAKE_GDN_NONTRANSPOSE_DOMAIN_LAUNCHES.values()
        )

    def check_whole(state, out_ref, state_ref, scale=_SCALE, backend="cake_gdn"):
        out = _public(tensors, state, scale, backend=backend)
        torch.cuda.synchronize()
        assert _same_bits(out, out_ref) and _same_bits(state, state_ref)
        no_twin()

    check_whole(initial.clone(), whole_out, whole_state)
    homed = localize_gdn_decode_state(initial, _H)
    assert homed is not initial
    check_whole(homed.clone(), whole_out, whole_state)
    bigger = torch.cat([initial, initial[:8]])
    prefix = localize_gdn_decode_state(bigger, _H)[:batch_size]
    assert cake_locality.tensor_homing(prefix) is not None
    check_whole(prefix, whole_out, whole_state)
    with monkeypatch.context() as patch:
        patch.setattr(cake_locality, "sm_table", lambda device=None: None)
        check_whole(homed, whole_out, whole_state)
    for small in (9, 10, 16, 31, 32, 37, 47):
        for scale in (_SCALE, 1.0):
            small_tensors, small_initial = _inputs(small)
            small_out, small_state = _whole(small_tensors, small_initial, scale)
            homed_small = localize_gdn_decode_state(small_initial, _H, scale)
            assert homed_small is small_initial
            ordinary = small_initial.clone()
            small_homed = _home(small_initial, (small + 1) // 2)
            for state in (ordinary, small_homed):
                out = _public(small_tensors, state, scale)
                torch.cuda.synchronize()
                assert _same_bits(out, small_out), (small, scale)
                assert _same_bits(state, small_state), (small, scale)
            no_twin()
    # The H4 / HV8 rows (K^-0.5; their whole route below 32 rows), homed by hand on
    # die 1 only, on die 0 only and at half the rows.
    for small, rows0 in ((4, 0), (4, 4), (16, 8)):
        small_tensors, small_initial = _inputs(small, heads=(4, 8))
        small_out, small_state = _whole(small_tensors, small_initial, _SCALE)
        assert localize_gdn_decode_state(small_initial, 4) is small_initial
        ordinary = small_initial.clone()
        small_homed = _home(small_initial, rows0)
        assert cake_locality.tensor_homing(small_homed) is not None
        for state in (ordinary, small_homed):
            out = _public(small_tensors, state, _SCALE)
            torch.cuda.synchronize()
            assert _same_bits(out, small_out), (small, rows0)
            assert _same_bits(state, small_state), (small, rows0)
        no_twin()
    # No Cake route at H8 / HV16: the homed state is refused as the ordinary one is,
    # before launch, both left untouched.
    other_tensors, other_initial = _inputs(batch_size, heads=(8, 16))
    assert localize_gdn_decode_state(other_initial, 8) is other_initial
    other_homed = _home(other_initial, batch_size // 2)
    for state in (other_initial.clone(), other_homed):
        with pytest.raises(cake_gdn.CakeGDNUnsupportedError):
            _public(other_tensors, state, _SCALE)
        torch.cuda.synchronize()
        assert _same_bits(state, other_initial)


def test_unlisted_boards_keep_the_whole_body() -> None:
    # An SM107a device without listed cells (a 216-SM board) homes nothing and runs
    # the whole body on a state homed by hand.
    device = _require_locality_device()
    sm_count = _sm_count(device)
    if ("sm_107a", sm_count) in cake_gdn.CAKE_GDN_NONTRANSPOSE_FP32_T1_DOMAIN_ROUTES:
        pytest.skip(f"locality-domain cells are listed for {sm_count} SMs")
    batch_size = 64
    tensors, initial = _inputs(batch_size)
    whole_out, whole_state = _whole(tensors, initial, _SCALE)
    assert localize_gdn_decode_state(initial, _H) is initial
    homed = _home(initial, batch_size // 2)
    out = _public(tensors, homed, _SCALE)
    torch.cuda.synchronize()
    assert _same_bits(out, whole_out) and _same_bits(homed, whole_state)


# --------------------------------------------------------------------------- graphs


def test_twin_cuda_graph_capture_and_replay() -> None:
    # After one eager call, the public call on a homed state is captured with a
    # pre-zeroed counter pair of the reserve the helper filled; replays give the eager
    # bits, follow new inputs and leave the pair zero.
    device, table = _require_domain()
    batch_size = 64
    tensors, initial = _inputs(batch_size)
    whole_out, whole_state = _whole(tensors, initial, _SCALE)
    homed = localize_gdn_decode_state(initial, _H)
    out = _empty_out(batch_size)
    stream = torch.cuda.Stream()
    # The homing copy and the output fill ran on the current stream.
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _public(tensors, homed, _SCALE, out=out)
    stream.synchronize()
    assert _same_bits(out, whole_out) and _same_bits(homed, whole_state)
    reserve = cake_locality._counter_state(device).reserves[
        locality.DOMAIN_COUNTER_WORDS
    ]
    issued = reserve.issued
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        _public(tensors, homed, _SCALE, out=out)
    assert reserve.issued == issued + 1
    for _ in range(2):
        homed.copy_(initial)
        out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        assert _same_bits(out, whole_out) and _same_bits(homed, whole_state)
    fresh, _ = _inputs(batch_size, seed=7)
    for name in ("q", "k", "v", "a", "b"):
        tensors[name].copy_(fresh[name])
    expected_out, expected_state = _whole(tensors, initial, _SCALE)
    homed.copy_(initial)
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    assert _same_bits(out, expected_out) and _same_bits(homed, expected_state)


# --------------------------------------------------------------------------- helper


def test_localize_gdn_decode_state_homes_or_returns_the_state() -> None:
    device, table = _require_domain()
    for batch_size in (48, 63, 64):
        _, initial = _inputs(batch_size)
        homed = localize_gdn_decode_state(initial, _H)
        record = cake_locality.tensor_homing(homed)
        assert record is not None and record.device_index == device
        assert record.nbytes == initial.numel() * 4
        assert record.cut == (batch_size + 1) // 2 * _ROW_BYTES
        assert homed.shape == initial.shape and homed.is_contiguous()
        assert _same_bits(homed, initial)
    _, initial = _inputs(64)
    assert localize_gdn_decode_state(initial, _H, rows0=0) is not initial
    assert localize_gdn_decode_state(initial, _H, rows0=64) is not initial
    assert localize_gdn_decode_state(initial, _H, 1.0) is not initial
    for args in ((4,), (_H, 0.5)):
        assert localize_gdn_decode_state(initial, *args) is initial
    half = initial.half()
    assert localize_gdn_decode_state(half, _H) is half
    for small_batch in (16, 32, 47):
        _, small = _inputs(small_batch)
        assert localize_gdn_decode_state(small, _H) is small
    for rows0 in (-1, 65):
        with pytest.raises(ValueError, match="rows0"):
            localize_gdn_decode_state(initial, _H, rows0=rows0)
    with pytest.raises(ValueError, match="contiguous"):
        localize_gdn_decode_state(initial.transpose(2, 3), _H)
    with pytest.raises(ValueError, match="CUDA"):
        localize_gdn_decode_state(initial.cpu(), _H)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        captured = localize_gdn_decode_state(initial, _H)
        initial.add_(0)
    assert captured is initial


# --------------------------------------------------------------------------- races


_RACE_CASES = [
    (64, None),
    (128, None),
    (512, None),
    (64, 64),
    (64, 0),
    (64, 13),
]


@pytest.mark.parametrize("case", _RACE_CASES, ids=lambda c: f"B{c[0]}-rows0{c[1]}")
def test_twin_race_gate(case) -> None:
    # A stress gate: every launch of the twin gives the first launch's bits.
    launches = int(os.environ.get(_RACE_ENV, "0") or 0)
    if launches <= 0:
        pytest.skip(f"a stress gate: set {_RACE_ENV} to the launches per case")
    device, table = _require_domain()
    batch_size, rows0 = case
    if rows0 is None:
        rows0 = (batch_size + 1) // 2
    tensors, initial = _inputs(batch_size)
    whole_out, whole_state = _whole(tensors, initial, _SCALE)
    homed = _home(initial, rows0)
    out = _empty_out(batch_size)

    def assert_the_twin_runs():
        # A twin that fails to build, load or launch leaves the call to the whole
        # body, whose bits pass the comparison too; a launch failure drops the twin
        # for the rest of the process.
        twin = _domain_launch(device, batch_size, _SCALE, table, homed)
        assert twin is not None and twin[0] is not None and twin[2] == rows0

    failures = []
    for launch in range(launches):
        homed.copy_(initial)
        out.fill_(float("nan"))
        _public(tensors, homed, _SCALE, out=out)
        if launch == 0:
            assert_the_twin_runs()
        if not (_same_bits(out, whole_out) and _same_bits(homed, whole_state)):
            failures.append(launch)
            if len(failures) > 5:
                break
    assert not failures, f"launches {failures} differ from the whole body's bits"
    assert_the_twin_runs()


# --------------------------------------------------------------------------- public


def test_public_path_resolves_once_and_follows_the_state(monkeypatch) -> None:
    # The twin launch of a row is resolved and loaded once; the whole launch table
    # keeps its (entry, grid) value; each homed allocation is bound to the twin
    # once; calls alternating between two homed states and an ordinary one each
    # update their own state (the tensor map follows the state).
    device, table = _require_domain()
    batch_size = 64
    tensors, first = _inputs(batch_size)
    _, second = _inputs(batch_size, seed=11)
    expected = {
        "first": _whole(tensors, first, _SCALE),
        "second": _whole(tensors, second, _SCALE),
        "ordinary": _whole(tensors, first, _SCALE),
    }
    states = {
        "first": localize_gdn_decode_state(first, _H),
        "second": localize_gdn_decode_state(second, _H),
        "ordinary": first.clone(),
    }
    initial = {name: state.clone() for name, state in states.items()}
    loads: list[str] = []
    load = cake_gdn.load_cake_gdn_kernel

    def spy(name, arch, **kwargs):
        loads.append(name)
        return load(name, arch, **kwargs)

    monkeypatch.setattr(cake_gdn, "load_cake_gdn_kernel", spy)
    monkeypatch.setattr(gdn_decode, "_CAKE_GDN_NONTRANSPOSE_LAUNCHES", {})
    monkeypatch.setattr(gdn_decode, "_CAKE_GDN_NONTRANSPOSE_DOMAIN_LAUNCHES", {})
    monkeypatch.setattr(gdn_decode, "_CAKE_GDN_NONTRANSPOSE_HOMED_LAUNCHES", {})
    for order in (("first", "second", "ordinary"), ("ordinary", "second", "first")):
        for name in order:
            states[name].copy_(initial[name])
            out = _public(tensors, states[name], _SCALE)
            torch.cuda.synchronize()
            assert _same_bits(out, expected[name][0]), name
            assert _same_bits(states[name], expected[name][1]), name
    twin = _twin_name(batch_size, _SCALE, table, device)
    assert loads.count(twin) == 1
    assert loads.count(_whole_name(batch_size, _SCALE)) == 1
    key = (device, batch_size, _H, _H, _HV, _SCALE, True)
    (whole_launch,) = gdn_decode._CAKE_GDN_NONTRANSPOSE_LAUNCHES.values()
    assert len(whole_launch) == 2 and whole_launch[1] == batch_size * _HV
    assert list(gdn_decode._CAKE_GDN_NONTRANSPOSE_LAUNCHES) == [key]
    (domain_key,) = gdn_decode._CAKE_GDN_NONTRANSPOSE_DOMAIN_LAUNCHES
    assert domain_key[:7] == key and domain_key[7] == table.counts
    records = [
        cake_locality.tensor_homing(states[name]) for name in ("first", "second")
    ]
    bound = gdn_decode._CAKE_GDN_NONTRANSPOSE_HOMED_LAUNCHES
    assert sorted(bound) == sorted(id(record) for record in records)
    for record in records:
        assert bound[id(record)][0] is record and bound[id(record)][2] is not None
