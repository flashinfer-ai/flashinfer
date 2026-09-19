"""One-token stochastic SSU contracts, independent of the MUSA reference."""

import os
import struct

import pytest

torch = pytest.importorskip("torch")
if os.environ.get("FLASHINFER_MAMBA_TEST_DEVICE") != "musa":
    pytest.skip("requires MUSA", allow_module_level=True)

from flashinfer.mamba.selective_state_update import selective_state_update
import flashinfer.mamba.musa_reference as reference
from .test_philox_cpu_oracle import philox4x32_words, cvt_rs_f16_bits


def _inputs(n, table=False, state_dtype=None):
    h, d = 2, 64
    state = torch.zeros(3, h, d, n, device="musa", dtype=state_dtype or torch.float16)
    x = torch.ones(1, h, d, device="musa", dtype=torch.bfloat16)
    dt = torch.tensor(
        [[1.00048828125, -1.00048828125]], device="musa", dtype=torch.float32
    )[:, :, None].expand(-1, -1, d)
    a = torch.zeros(h, device="musa")[:, None, None].expand(h, d, n)
    b = torch.ones(1, 1, n, device="musa", dtype=x.dtype)
    c = torch.zeros_like(b)
    skip = torch.zeros(h, device="musa")[:, None].expand(h, d)
    src = torch.tensor([[1]] if table else [1], device="musa", dtype=torch.int32)
    dst = torch.tensor([[2]] if table else [2], device="musa", dtype=torch.int32)
    return state, x, dt, a, b, c, skip, src, dst


def _expected(n, rounds, seed, src_slot=1, slot_stride=None):
    values = [1.00048828125, -1.00048828125]
    group = min(4, n // 32)
    expected = []
    for h in range(2):
        bits = struct.unpack("<I", struct.pack("<f", values[h]))[0]
        for d in range(64):
            base = src_slot * (slot_stride or (2 * 64 * n)) + h * 64 * n + d * n
            for i in range(0, n, group):
                words = philox4x32_words(seed, base + i, rounds)
                expected.extend(cvt_rs_f16_bits(bits, words[j]) for j in range(group))
    return torch.tensor(expected, dtype=torch.int32)


@pytest.mark.parametrize("n", [64, 128, 256])
@pytest.mark.parametrize("rounds", [5, 10])
@pytest.mark.parametrize("table", [False, True])
def test_seeded_public_dispatch_and_cache_bits(monkeypatch, n, rounds, table):
    def forbidden_reference(*args, **kwargs):
        raise AssertionError("seeded one-token call fell back to reference")

    monkeypatch.setattr(
        reference, "selective_state_update_musa_reference", forbidden_reference
    )
    state, x, dt, a, b, c, skip, src, dst = _inputs(n, table)
    seed_value = 42 + 2**40
    seed = torch.tensor([seed_value], dtype=torch.int64, device="musa")
    out = torch.empty_like(x)
    actual_out = selective_state_update(
        state,
        x,
        dt,
        a,
        b,
        c,
        skip,
        state_batch_indices=src,
        dst_state_batch_indices=dst,
        rand_seed=seed,
        philox_rounds=rounds,
        algorithm="simple",
        out=out,
    )
    assert actual_out.data_ptr() == out.data_ptr()
    actual = state[2].view(torch.int16).cpu().to(torch.int32).flatten() & 0xFFFF
    assert torch.equal(actual, _expected(n, rounds, seed_value))
    assert torch.equal(out.cpu(), torch.zeros_like(out.cpu()))
    assert bool((state[:2] == 0).all().cpu())


@pytest.mark.parametrize("pad", [-1, 0])
def test_padding_preserves_cache(pad):
    state, x, dt, a, b, c, skip, src, dst = _inputs(128)
    src.fill_(pad)
    dst.fill_(pad)
    seed = torch.tensor([42], dtype=torch.int64, device="musa")
    selective_state_update(
        state,
        x,
        dt,
        a,
        b,
        c,
        skip,
        state_batch_indices=src,
        dst_state_batch_indices=dst,
        pad_slot_id=pad,
        rand_seed=seed,
        philox_rounds=5,
        algorithm="simple",
    )
    assert bool((state == 0).all().cpu())


def test_softplus_does_not_overflow():
    state, x, dt, a, b, c, skip, src, dst = _inputs(128, state_dtype=torch.float32)
    dt = torch.full_like(dt, 100.0)
    a = -torch.ones_like(a)
    c = torch.ones_like(c)
    out = selective_state_update(
        state,
        x,
        dt,
        a,
        b,
        c,
        skip,
        state_batch_indices=src,
        dst_state_batch_indices=dst,
        dt_softplus=True,
    )
    torch.testing.assert_close(state[2], torch.full_like(state[2], 100.0))
    assert bool(torch.isfinite(out).all().cpu())


def test_rounding_only_changes_stored_state():
    state, x, dt, a, b, c, skip, src, dst = _inputs(128)
    x, b = x.float(), b.float()
    c = torch.ones_like(b)
    other = state.clone()
    kwargs = dict(
        state_batch_indices=src, dst_state_batch_indices=dst, algorithm="simple"
    )
    plain = selective_state_update(state, x, dt, a, b, c, skip, **kwargs)
    rounded = selective_state_update(
        other,
        x,
        dt,
        a,
        b,
        c,
        skip,
        rand_seed=torch.tensor([42], dtype=torch.int64, device="musa"),
        philox_rounds=5,
        **kwargs,
    )
    assert torch.equal(plain.cpu(), rounded.cpu())
    assert not torch.equal(state[2].cpu(), other[2].cpu())


def test_captured_seed_is_loaded_dynamically():
    state, x, dt, a, b, c, skip, src, dst = _inputs(128, table=True)
    graph_type = getattr(torch.musa, "MUSAGraph", None)
    if graph_type is None:
        pytest.skip("torch.musa.MUSAGraph is unavailable")
    key = torch.tensor([42 + 2**40], dtype=torch.int64, device="musa")
    out = torch.empty_like(x)
    kwargs = dict(
        state_batch_indices=src,
        dst_state_batch_indices=dst,
        rand_seed=key,
        philox_rounds=5,
        algorithm="simple",
        out=out,
    )
    selective_state_update(state, x, dt, a, b, c, skip, **kwargs)
    torch.musa.synchronize()
    graph = graph_type()
    with torch.musa.graph(graph):
        selective_state_update(state, x, dt, a, b, c, skip, **kwargs)
    for seed in [42 + 2**40, 43 + 2**40]:
        key.fill_(seed)
        graph.replay()
        torch.musa.synchronize()
        actual = state[2].view(torch.int16).cpu().to(torch.int32).flatten() & 0xFFFF
        assert torch.equal(actual, _expected(128, 5, seed))


@pytest.mark.parametrize("rounds", [5, 10])
def test_reference_uses_requested_philox_rounds(rounds):
    from flashinfer.mamba.musa_reference import _philox_uniform

    seed = 42 + 2**40
    offset = 2**32 + 1
    value = torch.empty(4, device="musa")
    keys = torch.tensor([seed], dtype=torch.int64, device="musa")
    actual = _philox_uniform(value, keys, offset, rounds)
    words = [philox4x32_words(seed, offset + i, rounds)[0] for i in range(4)]
    expected = (torch.tensor(words, dtype=torch.float32) + 0.5) / 4294967296.0
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_batched_layer_view_and_padding():
    n = 128
    _, x, dt, a, b, c, skip, _, _ = _inputs(n)
    storage = torch.zeros(5, 3, 2, 64, n, device="musa", dtype=torch.float16)
    state = storage[:, 1]
    x = x.repeat(3, 1, 1)
    dt = dt.expand(3, -1, -1)
    b, c = b.expand(3, -1, -1), c.expand(3, -1, -1)
    src = torch.tensor([[1], [2], [0]], device="musa", dtype=torch.int32)
    dst = torch.tensor([[3], [4], [0]], device="musa", dtype=torch.int32)
    seed = 42 + 2**40
    key = torch.tensor([seed], device="musa", dtype=torch.int64)
    selective_state_update(
        state,
        x,
        dt,
        a,
        b,
        c,
        skip,
        state_batch_indices=src,
        dst_state_batch_indices=dst,
        pad_slot_id=0,
        rand_seed=key,
        philox_rounds=5,
        algorithm="simple",
    )
    for source, destination in [(1, 3), (2, 4)]:
        actual = (
            state[destination].view(torch.int16).cpu().to(torch.int32).flatten()
            & 0xFFFF
        )
        assert torch.equal(actual, _expected(n, 5, seed, source, state.stride(0)))
    assert bool((storage[:, 0] == 0).all().cpu())
    assert bool((storage[:, 2] == 0).all().cpu())
    assert bool((state[:3] == 0).all().cpu())
