# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Generated public-API forwarding tests for the frozen Cake GDN contract."""

import pytest
import torch

import flashinfer.gdn_decode as gdn
from flashinfer.jit import cake_gdn as cake_gdn


_PROMOTED_BF16_ROWS = [
    (4, 1, 16, 32, True, False, False, 0),
    (4, 2, 16, 32, False, True, True, 4),
    (8, 3, 16, 64, True, True, True, 3),
    (8, 4, 16, 64, True, True, True, 4),
    (8, 2, 16, 64, True, False, False, 0),
    (8, 4, 16, 64, True, False, True, 5),
    (8, 4, 16, 32, True, True, True, 4),
    (4, 1, 4, 8, True, False, False, 0),
    (1, 4, 4, 8, True, True, True, 4),
    (2, 4, 4, 8, True, True, True, 4),
    (3, 4, 4, 8, True, True, True, 4),
    (4, 4, 4, 8, True, True, True, 4),
    (5, 4, 4, 8, True, True, True, 4),
    (6, 4, 4, 8, True, True, True, 4),
    (7, 4, 4, 8, True, True, True, 4),
    (8, 4, 4, 8, True, True, True, 4),
    (5, 4, 4, 8, False, True, True, 4),
]


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a", "sm_107a"))
@pytest.mark.parametrize(
    "batch_size,seq_len,num_q_heads,num_v_heads,strided,disable,cache,cache_steps",
    _PROMOTED_BF16_ROWS,
)
def test_all_promoted_bf16_rows_resolve_on_every_architecture(
    arch,
    batch_size,
    seq_len,
    num_q_heads,
    num_v_heads,
    strided,
    disable,
    cache,
    cache_steps,
):
    route = cake_gdn.select_cake_gdn_decode_variant(
        arch=arch,
        batch_size=batch_size,
        io_dtype="bfloat16",
        state_dtype="bfloat16",
        head_size=128,
        layout="pretranspose",
        num_k_heads=num_q_heads,
        num_q_heads=num_q_heads,
        num_v_heads=num_v_heads,
        scale=128**-0.5,
        seq_len=seq_len,
        use_qk_l2norm=True,
        strided_inputs=strided,
        disable_state_update=disable,
        cache_intermediate_states=cache,
        cache_steps=cache_steps,
    )
    assert route.route_id.startswith("flashinfer.gdn_decode.")
    assert route.variant_name


def _inputs(*, state_dtype, seq_len=2):
    batch_size, num_q_heads, num_v_heads, dim = 1, 1, 1, 128
    q = torch.zeros(batch_size, seq_len, num_q_heads, dim, dtype=torch.bfloat16)
    return dict(
        q=q,
        k=q.clone(),
        v=torch.zeros(batch_size, seq_len, num_v_heads, dim, dtype=torch.bfloat16),
        state=None,
        A_log=torch.zeros(num_v_heads, dtype=torch.float32),
        a=torch.zeros(batch_size, seq_len, num_v_heads, dtype=torch.bfloat16),
        dt_bias=torch.zeros(num_v_heads, dtype=torch.float32),
        b=torch.zeros(batch_size, seq_len, num_v_heads, dtype=torch.bfloat16),
        output=torch.empty(batch_size, seq_len, num_v_heads, dim, dtype=torch.bfloat16),
        initial_state=torch.zeros(3, num_v_heads, dim, dim, dtype=state_dtype),
        initial_state_indices=torch.tensor([1], dtype=torch.int32),
        output_state_indices=torch.tensor([2], dtype=torch.int32),
    )


@pytest.mark.cpu_only
def test_public_bf16_verify_forwards_caller_cache_and_no_update(monkeypatch):
    tensors = _inputs(state_dtype=torch.bfloat16)
    cache = torch.empty(1, 4, 1, 128, 128, dtype=torch.bfloat16)
    captured = {}

    def fake_mtp(**kwargs):
        captured.update(kwargs)
        return kwargs["output"]

    monkeypatch.setattr(gdn, "_GDN_DECODE_BF16_STATE_AVAILABLE", True)
    monkeypatch.setattr(gdn, "_gated_delta_rule_bf16_state_mtp", fake_mtp)
    output, returned_state = gdn.gated_delta_rule_decode_pretranspose(
        **tensors,
        intermediate_states_buffer=cache,
        disable_state_update=True,
    )
    assert captured["intermediate_states_buffer"] is cache
    assert captured["disable_state_update"] is True
    assert output is tensors["output"]
    assert returned_state is tensors["initial_state"]


@pytest.mark.cpu_only
def test_public_fp32_checkpoint_forwards_caller_cache_and_update(monkeypatch):
    tensors = _inputs(state_dtype=torch.float32)
    cache = torch.empty(1, 3, 1, 128, 128, dtype=torch.float32)
    captured = {}

    def fake_mtp(**kwargs):
        captured.update(kwargs)
        return kwargs["output"], kwargs["initial_state"]

    monkeypatch.setattr(gdn, "gated_delta_rule_mtp", fake_mtp)
    output, returned_state = gdn.gated_delta_rule_decode_pretranspose(
        **tensors,
        intermediate_states_buffer=cache,
        disable_state_update=False,
    )
    assert captured["intermediate_states_buffer"] is cache
    assert captured["disable_state_update"] is False
    assert output is tensors["output"]
    assert returned_state is tensors["initial_state"]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"intermediate_states_buffer": torch.empty(1)}, "only for T > 1"),
        ({"disable_state_update": True}, "only for T > 1"),
    ],
)
def test_public_t1_controls_fail_closed_before_launch(kwargs, match):
    tensors = _inputs(state_dtype=torch.bfloat16, seq_len=1)
    with pytest.raises(ValueError, match=match):
        gdn.gated_delta_rule_decode_pretranspose(**tensors, **kwargs)


@pytest.mark.cpu_only
@pytest.mark.parametrize("backend", ["cake_gdn", "auto"])
def test_public_nontranspose_decode_refuses_cpu_tensors_before_resolving(
    backend, monkeypatch
):
    # A CPU row never reaches the resolver or the launch table: explicit cake_gdn
    # requests fail closed with the device error, auto falls through to FlashInfer.
    batch_size, num_heads, num_v_heads, dim = 2, 16, 32, 128
    launches = {}
    monkeypatch.setattr(gdn, "_CAKE_GDN_NONTRANSPOSE_LAUNCHES", launches)

    def no_select(**kwargs):
        raise AssertionError("resolver called for a CPU row")

    def fallback(*args, **kwargs):
        raise RuntimeError("fell through to FlashInfer")

    monkeypatch.setattr(cake_gdn, "select_cake_gdn_decode_variant", no_select)
    monkeypatch.setattr(gdn, "run_nontranspose_decode", fallback)
    q = torch.zeros(batch_size, 1, num_heads, dim, dtype=torch.bfloat16)
    args = dict(
        q=q,
        k=q.clone(),
        v=torch.zeros(batch_size, 1, num_v_heads, dim, dtype=torch.bfloat16),
        state=torch.zeros(batch_size, num_v_heads, dim, dim, dtype=torch.float32),
        A_log=torch.zeros(num_v_heads, dtype=torch.float32),
        a=torch.zeros(batch_size, 1, num_v_heads, dtype=torch.bfloat16),
        dt_bias=torch.zeros(num_v_heads, dtype=torch.float32),
        b=torch.zeros(batch_size, 1, num_v_heads, dtype=torch.bfloat16),
        output=torch.zeros(batch_size, 1, num_v_heads, dim, dtype=torch.bfloat16),
    )
    if backend == "cake_gdn":
        with pytest.raises(
            cake_gdn.CakeGDNUnsupportedError,
            match="^GDN non-CP nontranspose decode requires one CUDA device$",
        ):
            gdn.gated_delta_rule_decode(**args, backend=backend)
    else:
        with pytest.raises(RuntimeError, match="fell through to FlashInfer"):
            gdn.gated_delta_rule_decode(**args, backend=backend)
    assert launches == {}


# ------------------------------------------------ pretranspose alignment (fake CUDA)

_SCALE = 128**-0.5
# The resolver itself, for the recording spies below.
_SELECT = cake_gdn.select_cake_gdn_decode_variant


class _FakeCuda(torch.Tensor):
    """A meta tensor that the Cake pretranspose host checks take for a tensor of
    cuda:0, at base address ``address``."""

    address = 0

    @property
    def is_cuda(self):
        return True

    @property
    def device(self):
        return torch.device("cuda", 0)

    def get_device(self):
        return 0

    def data_ptr(self):
        return self.address


def _fake(shape, dtype, address=0, stride=None) -> torch.Tensor:
    meta = (
        torch.empty(shape, dtype=dtype, device="meta")
        if stride is None
        else torch.empty_strided(shape, stride, dtype=dtype, device="meta")
    )
    tensor = torch.Tensor._make_subclass(_FakeCuda, meta)
    tensor.address = address
    return tensor


def _moved(tensor, *, address=None, stride=None) -> torch.Tensor:
    """A fake like ``tensor`` at another base address or with other strides."""
    return _fake(
        tuple(tensor.shape),
        tensor.dtype,
        tensor.address if address is None else address,
        tensor.stride() if stride is None else stride,
    )


# (state dtype, B, T, H, HV, checkpoint steps, fused QKV and packed gates): rows the
# Cake kernels serve on sm_100a, sm_103a and sm_107a.  From B * HV = 512 (bf16_t1_wide)
# a BF16 T=1 call they do not serve runs the Rubin kernel on sm_107a.
_ALIGNMENT_ROWS = {
    "bf16_t1": (torch.bfloat16, 4, 1, 16, 32, 0, True),
    "bf16_t1_wide": (torch.bfloat16, 16, 1, 16, 32, 0, True),
    "bf16_checkpoint_t4": (torch.bfloat16, 8, 4, 16, 64, 5, True),
    "fp32_t1": (torch.float32, 32, 1, 16, 32, 0, False),
    "fp32_mtp_t4": (torch.float32, 4, 4, 16, 32, 4, True),
}
# The FlashInfer kernel backend="auto" falls back to on each row.
_ROW_FALLBACKS = {
    "bf16_t1": "_gated_delta_rule_bf16_state",
    "bf16_checkpoint_t4": "_gated_delta_rule_bf16_state_mtp",
    "fp32_t1": "run_pretranspose_decode",
    "fp32_mtp_t4": "gated_delta_rule_mtp",
}


def _pretranspose_fakes(row: str) -> dict:
    """Fake inputs of one row at the smallest base addresses the kernels accept: q and k
    16-byte aligned with a BF16 state (8-byte with an FP32 state), v and the output
    8-byte, the state and the checkpoint 16-byte, every other tensor at its element
    alignment only.  Fused rows take q, k and v as views of one packed row and a and b
    as views of packed gates with an odd row length; the BF16 pool is padded by one
    head per slot and the FP32 MTP pool takes every other head (production layouts)."""
    state_dtype, batch, steps, heads, v_heads, cache_steps, fused = _ALIGNMENT_ROWS[row]
    bf16, fp32 = torch.bfloat16, torch.float32
    qk_align = 16 if state_dtype is bf16 else 8
    head = 128 * 128
    qkv_stride = gate_stride = state_stride = None
    if fused:
        width = (2 * heads + v_heads) * 128
        qkv_stride = (steps * width, width, 128, 1)
        gates = 2 * v_heads + 7
        gate_stride = (steps * gates, gates, 1)
    if state_dtype is bf16:
        state_stride = ((v_heads + 1) * head, head, 128, 1)
    elif steps > 1:
        state_stride = (2 * v_heads * head, 2 * head, 128, 1)
    return {
        "q": _fake((batch, steps, heads, 128), bf16, qk_align, qkv_stride),
        "k": _fake((batch, steps, heads, 128), bf16, 1024 + qk_align, qkv_stride),
        "v": _fake((batch, steps, v_heads, 128), bf16, 8, qkv_stride),
        "state": _fake(
            (2 * batch + 2, v_heads, 128, 128), state_dtype, 16, state_stride
        ),
        "A_log": _fake((v_heads,), fp32, 4),
        "a": _fake((batch, steps, v_heads), bf16, 2, gate_stride),
        "dt_bias": _fake((v_heads,), fp32, 12),
        "b": _fake((batch, steps, v_heads), bf16, 6, gate_stride),
        "output": _fake((batch, steps, v_heads, 128), bf16, 8),
        "initial_state_indices": _fake((batch,), torch.int32, 4),
        "output_state_indices": _fake((batch,), torch.int32, 4),
        "cache": (
            _fake((batch, cache_steps, v_heads, 128, 128), state_dtype, 16)
            if cache_steps
            else None
        ),
    }


def _entry_tensors(row: str, tensors: dict) -> list:
    """The tensor arguments the helper passes to the row's kernel entry, in order."""
    state_dtype, _, steps, _, _, _, _ = _ALIGNMENT_ROWS[row]
    names = ["q", "k", "v", "state", "A_log", "a", "dt_bias", "b", "output"]
    if state_dtype is torch.bfloat16:
        names.append("output" if tensors["cache"] is None else "cache")
    elif steps > 1:
        names.append("cache")
    names += ["initial_state_indices", "output_state_indices"]
    return [tensors[name] for name in names]


class _Backends:
    """Fakes around the Cake pretranspose helper: the device capability, the resolver
    (the real one, recorded), the loader and its kernel entries, the slot checks and
    the FlashInfer fallbacks (all recorded; a fallback raises unless
    ``fallback_ok``)."""

    def __init__(self, monkeypatch, capability, *, fallback_ok=False):
        self.selected: list[str] = []
        self.loads: list[str] = []
        self.calls: list[tuple[str, tuple]] = []
        self.slot_checks: list[str] = []
        self.fallbacks: list[tuple[str, dict]] = []

        def spy(**kwargs):
            route = _SELECT(**kwargs)
            self.selected.append(route.variant_name)
            return route

        def load(name, arch, **kwargs):
            assert not kwargs
            self.loads.append(name)

            def entry(*args):
                self.calls.append((name, args))

            return entry

        def slot_check(indices, pool_size, *, name, allow_minus_one):
            self.slot_checks.append(name)

        def fallback(name, result):
            def run(*args, **kwargs):
                self.fallbacks.append((name, kwargs))
                if not fallback_ok:
                    raise AssertionError(f"fell back to {name}")
                return result(kwargs)

            return run

        monkeypatch.setattr(gdn, "get_compute_capability", lambda _device: capability)
        monkeypatch.setattr(cake_gdn, "select_cake_gdn_decode_variant", spy)
        monkeypatch.setattr(cake_gdn, "load_cake_gdn_kernel", load)
        monkeypatch.setattr(gdn, "_cake_gdn_assert_state_slots", slot_check)
        monkeypatch.setattr(gdn, "_GDN_DECODE_BF16_STATE_AVAILABLE", True)
        for name, result in (
            ("_gated_delta_rule_bf16_state", lambda kw: kw["output"]),
            ("_gated_delta_rule_bf16_state_mtp", lambda kw: kw["output"]),
            ("_gated_delta_rule_t1_rubin", lambda kw: kw["output"]),
            ("gated_delta_rule_mtp", lambda kw: (kw["output"], kw["initial_state"])),
            ("run_pretranspose_decode", lambda kw: None),
        ):
            monkeypatch.setattr(gdn, name, fallback(name, result))

    def counts(self) -> tuple[int, ...]:
        return tuple(
            len(record)
            for record in (
                self.selected,
                self.loads,
                self.calls,
                self.slot_checks,
                self.fallbacks,
            )
        )


def _run_pretranspose(tensors: dict, backend):
    kwargs = dict(
        q=tensors["q"],
        k=tensors["k"],
        v=tensors["v"],
        A_log=tensors["A_log"],
        a=tensors["a"],
        dt_bias=tensors["dt_bias"],
        b=tensors["b"],
        scale=_SCALE,
        use_qk_l2norm=True,
        output=tensors["output"],
        initial_state_indices=tensors["initial_state_indices"],
        output_state_indices=tensors["output_state_indices"],
        intermediate_states_buffer=tensors["cache"],
        disable_state_update=False,
    )
    if backend is None:
        return gdn._run_cake_gdn_decode_pretranspose(
            state_pool=tensors["state"], **kwargs
        )
    output, _ = gdn.gated_delta_rule_decode_pretranspose(
        state=None, initial_state=tensors["state"], backend=backend, **kwargs
    )
    return output


@pytest.mark.cpu_only
@pytest.mark.parametrize("row", list(_ALIGNMENT_ROWS))
@pytest.mark.parametrize("backend", [None, "auto", "cake_gdn"])
def test_pretranspose_unaligned_inputs_raise_before_any_launch(
    monkeypatch, backend, row
):
    # Contiguous views at an unaligned storage offset: the Cake pretranspose kernels
    # read q and k in up to 16-byte vectors (8-byte with an FP32 state), v and the
    # output in 8-byte vectors and move the state and the checkpoint in up to 16-byte
    # copies, so the call raises ValueError before the slot checks, the resolver, the
    # loader and the launch.  The check is row- and GPU-independent: through the
    # public call (``backend`` not None) backend="auto" raises too, also on a device
    # the Cake kernels do not serve; it does not fall back to the FlashInfer kernels.
    bf16 = _ALIGNMENT_ROWS[row][0] is torch.bfloat16
    qk_misaligned = 8 if bf16 else 4
    misaligned = [
        ("q", qk_misaligned),
        ("k", qk_misaligned),
        ("k", 2),
        ("v", 2),
        ("v", 4),
        ("output", 2),
        ("output", 4),
        ("state", 8),
    ]
    if _ALIGNMENT_ROWS[row][5]:
        misaligned.append(("cache", 8))

    def run_misaligned(backends):
        before = backends.counts()
        for name, address in misaligned:
            tensors = _pretranspose_fakes(row)
            tensors[name] = _moved(tensors[name], address=address)
            with pytest.raises(ValueError, match="aligned"):
                _run_pretranspose(tensors, backend)
        assert backends.counts() == before

    for capability in ((10, 0), (10, 3), (10, 7)):
        backends = _Backends(monkeypatch, capability)
        # The smallest aligned bases launch as before, with the tensors passed
        # through: one resolve, one load, one launch, no fallback.  This runs first,
        # so the misaligned calls that follow come after a launched one.
        tensors = _pretranspose_fakes(row)
        assert _run_pretranspose(tensors, backend) is tensors["output"]
        ((name, args),) = backends.calls
        assert backends.selected == backends.loads == [name]
        expected = _entry_tensors(row, tensors)
        assert len(args) == len(expected) + 3
        assert all(arg is tensor for arg, tensor in zip(args, expected, strict=False))
        assert len(backends.slot_checks) == 2 and backends.fallbacks == []
        run_misaligned(backends)
    # On a device the Cake kernels do not serve (SM90), the misaligned calls raise all
    # the same, before the architecture is looked up.  An aligned call there raises
    # CakeGDNUnsupportedError, which backend="auto" turns into a fallback.
    backends = _Backends(monkeypatch, (9, 0))
    run_misaligned(backends)
    assert backends.counts() == (0, 0, 0, 0, 0)
    if backend == "auto":
        unsupported = pytest.raises(AssertionError, match="fell back")
    else:
        unsupported = pytest.raises(
            cake_gdn.CakeGDNUnsupportedError, match=r"compute capability 9\.0"
        )
    with unsupported:
        _run_pretranspose(_pretranspose_fakes(row), backend)
    assert backends.selected == backends.loads == backends.calls == []


# (row, tensor, strides) of views whose strides break the kernels' vector alignment.
_ALIGNMENT_BREAKING_STRIDES = [
    # q and k rows 132 elements (264 bytes) apart: whole 8-byte, not 16-byte, vectors.
    ("bf16_t1", "q", (16 * 132, 16 * 132, 132, 1)),
    ("bf16_t1", "k", (16 * 132, 16 * 132, 132, 1)),
    # v rows 130 elements (260 bytes) apart.
    ("bf16_checkpoint_t4", "v", (4 * 64 * 130, 64 * 130, 130, 1)),
    # Slots 4 elements (8 bytes) past a whole number of 16-byte vectors.
    ("bf16_t1", "state", (32 * 128 * 128 + 4, 128 * 128, 128, 1)),
    # q, k and v rows 130 elements (260 bytes) apart: not whole 8-byte vectors.
    ("fp32_mtp_t4", "q", (4 * 16 * 130, 16 * 130, 130, 1)),
    ("fp32_mtp_t4", "k", (4 * 16 * 130, 16 * 130, 130, 1)),
    ("fp32_mtp_t4", "v", (4 * 32 * 130, 32 * 130, 130, 1)),
    # Heads or slots 2 elements (8 bytes) past a whole number of 16-byte vectors.
    ("fp32_mtp_t4", "state", (32 * 16386, 16386, 128, 1)),
    ("fp32_mtp_t4", "state", (32 * 128 * 128 + 2, 128 * 128, 128, 1)),
]
# Views whose strides keep it (the vectors of v are 8 bytes, those of q and k 8 bytes
# with an FP32 state, and a dimension of extent 1 has no stride to keep).
_ALIGNMENT_KEEPING_STRIDES = [
    ("bf16_t1", "q", (16 * 128, 7, 128, 1)),
    ("bf16_t1", "v", (32 * 132, 32 * 132, 132, 1)),
    ("bf16_t1", "state", (32 * 128 * 128 + 8, 128 * 128, 128, 1)),
    ("fp32_mtp_t4", "q", (4 * 16 * 132, 16 * 132, 132, 1)),
]


@pytest.mark.cpu_only
@pytest.mark.parametrize("backend", [None, "auto", "cake_gdn"])
def test_pretranspose_unaligned_strides_fail_closed_or_fall_back(monkeypatch, backend):
    # The kernels add the q, k, v and state strides to their vector addresses: a view
    # whose stride breaks the alignment is a layout they do not serve.  The call raises
    # CakeGDNUnsupportedError before the slot checks, the resolver and the loader, and
    # backend="auto" runs the FlashInfer kernel once on the same tensors instead.
    # Strides that keep the alignment launch the row with the view passed through.
    for capability in ((10, 0), (10, 3), (10, 7)):
        for row, name, stride in _ALIGNMENT_BREAKING_STRIDES:
            backends = _Backends(monkeypatch, capability, fallback_ok=True)
            tensors = _pretranspose_fakes(row)
            tensors[name] = _moved(tensors[name], stride=stride)
            if backend == "auto":
                assert _run_pretranspose(tensors, backend) is tensors["output"]
                ((fallback, kwargs),) = backends.fallbacks
                assert fallback == _ROW_FALLBACKS[row]
                pool = kwargs.get("initial_state_source", kwargs.get("initial_state"))
                assert pool is tensors["state"]
                assert all(kwargs[arg] is tensors[arg] for arg in ("q", "k", "v"))
            else:
                with pytest.raises(
                    cake_gdn.CakeGDNUnsupportedError,
                    match="keep their vector alignment",
                ):
                    _run_pretranspose(tensors, backend)
                assert backends.fallbacks == []
            assert backends.selected == backends.loads == backends.calls == []
            assert backends.slot_checks == []
        for row, name, stride in _ALIGNMENT_KEEPING_STRIDES:
            backends = _Backends(monkeypatch, capability)
            tensors = _pretranspose_fakes(row)
            tensors[name] = _moved(tensors[name], stride=stride)
            assert _run_pretranspose(tensors, backend) is tensors["output"]
            ((_, args),) = backends.calls
            assert any(arg is tensors[name] for arg in args)
            assert backends.fallbacks == []


@pytest.mark.cpu_only
def test_pretranspose_auto_runs_rubin_only_on_whole_vector_pool_strides(monkeypatch):
    # On sm_107a backend="auto" runs a T=1 BF16-state call with B * HV >= 512 that the
    # Cake kernels do not serve on the Rubin kernel.  That kernel compiles a strided
    # pool with its strides baked into 16-byte state copies and cannot be built unless
    # the pool's slot, head and row strides are multiples of 8 elements, so any other
    # pool runs the generic kernel (the same bits) instead.  Each call falls back once,
    # with the same pool.
    monkeypatch.setattr(gdn, "_GDN_DECODE_BF16_STATE_RUBIN_AVAILABLE", True)
    head = 128 * 128
    # q rows 264 bytes apart: a view the Cake kernels do not serve.
    q_rows_264 = {"q": (16 * 132, 16 * 132, 132, 1)}
    for strides, fallback in (
        # The pool keeps its padding of one head per slot.
        (q_rows_264, "_gated_delta_rule_t1_rubin"),
        # Slots 8 elements (16 bytes) past a whole number of heads: still Rubin.
        (
            {**q_rows_264, "state": (32 * head + 8, head, 128, 1)},
            "_gated_delta_rule_t1_rubin",
        ),
        # Only the slot, only the head or only the row stride 4 elements (8 bytes)
        # past a multiple of 8 elements.
        ({"state": (32 * head + 4, head, 128, 1)}, "_gated_delta_rule_bf16_state"),
        (
            {"state": (32 * (head + 4), head + 4, 128, 1)},
            "_gated_delta_rule_bf16_state",
        ),
        (
            {"state": (32 * 128 * 132, 128 * 132, 132, 1)},
            "_gated_delta_rule_bf16_state",
        ),
    ):
        backends = _Backends(monkeypatch, (10, 7), fallback_ok=True)
        tensors = _pretranspose_fakes("bf16_t1_wide")
        for name, stride in strides.items():
            tensors[name] = _moved(tensors[name], stride=stride)
        assert _run_pretranspose(tensors, "auto") is tensors["output"]
        ((called, kwargs),) = backends.fallbacks
        assert called == fallback
        assert kwargs["initial_state_source"] is tensors["state"]
        assert backends.selected == backends.loads == backends.calls == []
