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

import hashlib
import importlib
import os
import importlib.util
import inspect
import math
import re
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flashinfer.mamba import SSDCombined


def _load_cake_benchmark_module():
    path = Path(__file__).parents[2] / "benchmarks" / "bench_cake_mamba_ssd_combined.py"
    spec = importlib.util.spec_from_file_location("bench_cake_mamba_ssd_combined", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ATOL = _RTOL = 1e-2
# Upper bound on the fraction of Cake outputs / final-state entries outside
# ``atol = rtol = 1e-2`` of the fp64 recurrence.  Measured on GB300 with
# ``benchmarks/cake_ssd_accuracy_probe.py`` (FP16 delta): <= 0.15 % of outputs on the
# ``_case`` distribution, 0.53-0.57 % on the CAKE-950 realistic-decay
# distribution (CuTe, bf16 delta: <= 0.20 % and 1.34-1.40 %); 0 final-state
# entries in every case.
_MAX_OUTSIDE_FRACTION = 0.01


def _declared_num_seqs(arguments):
    """The packed-sequence count a varlen call declares, in the runner's
    precedence: ``initial_states`` rows, ``seq_chunk_cumsum`` entries - 1,
    ``num_seqs``."""

    initial_states = arguments.get("initial_states")
    if initial_states is not None:
        return int(initial_states.shape[0])
    seq_chunk_cumsum = arguments.get("seq_chunk_cumsum")
    if seq_chunk_cumsum is not None:
        return int(seq_chunk_cumsum.numel()) - 1
    return int(arguments["num_seqs"])


def _sequence_lengths(constructor, tensors, arguments):
    """Per-sequence token counts: from ``seq_idx`` in packed-varlen mode (an
    id without tokens counts zero), ``seqlen`` per batch element otherwise."""

    x = tensors[0]
    if not constructor["has_varlen"]:
        return [x.shape[1]] * x.shape[0]
    seq_idx = arguments["seq_idx"].reshape(-1).to(torch.int64)
    return torch.bincount(seq_idx, minlength=_declared_num_seqs(arguments)).tolist()


def _fp64_reference(constructor, tensors, arguments, *, delta_dtype=None):
    """fp64 token-by-token SSM recurrence, the oracle both backends are
    measured against.

    ``dt' = clamp(softplus(dt + dt_bias), dt_limit)`` (softplus only when
    requested); ``state = exp(dt' * A) * state + delta * (x (x) B)``;
    ``y = C . state + D * x``; ``y *= z * sigmoid(z)`` when ``z`` is given.
    ``delta`` is ``dt'`` rounded to ``delta_dtype`` (``None`` = exact); the
    accuracy probe uses fp16 / bf16 here to emulate the kernels' ``delta``
    storage, the decay always uses the exact ``dt'`` (both kernels scan the
    fp32 ``dt * A``).  ``D`` follows the constructor like both backends: a 2D
    ``D`` on a per-head constructor (``d_has_hdim=False``) consumes its first
    column, a 1D ``D`` broadcasts over ``headdim``.
    Returns the token-major fp64 output with ``x``'s shape and the
    ``[num_seqs, nheads, 64, 128]`` final states (zero initial state when
    ``initial_states`` is ``None``).
    """

    x, dt, A, B, C = tensors
    lengths = _sequence_lengths(constructor, tensors, arguments)
    batch, seqlen, nheads, headdim = x.shape
    total = batch * seqlen
    assert sum(lengths) == total, (lengths, total)
    ngroups, dstate = B.shape[2], B.shape[3]
    rep = nheads // ngroups
    f64 = torch.float64
    xf = x.reshape(total, nheads, headdim).to(f64)
    dtf = dt.reshape(total, nheads).to(f64)
    dt_bias = arguments.get("dt_bias")
    if dt_bias is not None:
        dtf = dtf + dt_bias.to(f64)
    if arguments.get("dt_softplus", False):
        dtf = torch.nn.functional.softplus(dtf)
    dt_min, dt_max = arguments.get("dt_limit", (0.0, float("inf")))
    dtf = dtf.clamp(min=float(dt_min), max=float(dt_max))
    delta = dtf if delta_dtype is None else dtf.to(delta_dtype).to(f64)
    decay = torch.exp(A.to(f64)[None, :] * dtf)
    Bf = B.reshape(total, ngroups, dstate).to(f64).repeat_interleave(rep, dim=1)
    Cf = C.reshape(total, ngroups, dstate).to(f64).repeat_interleave(rep, dim=1)
    D = arguments.get("D")
    if D is None:
        Df = torch.zeros((nheads, 1), dtype=f64, device=x.device)
    else:
        Df = D.to(f64)
        Df = Df[:, None] if Df.ndim == 1 else Df
        if not constructor["d_has_hdim"]:
            Df = Df[:, :1]
    initial = arguments.get("initial_states")
    y = torch.empty((total, nheads, headdim), dtype=f64, device=x.device)
    states = torch.empty(
        (len(lengths), nheads, headdim, dstate), dtype=f64, device=x.device
    )
    start = 0
    for sequence, length in enumerate(lengths):
        if initial is None:
            state = torch.zeros((nheads, headdim, dstate), dtype=f64, device=x.device)
        else:
            state = initial[sequence].to(f64).clone()
        for token in range(start, start + length):
            state = state * decay[token][:, None, None] + (
                (delta[token][:, None] * xf[token])[:, :, None] * Bf[token][:, None, :]
            )
            y[token] = torch.einsum("hdn,hn->hd", state, Cf[token]) + Df * xf[token]
        states[sequence] = state
        start += length
    z = arguments.get("z")
    if z is not None:
        zf = z.reshape(total, nheads, headdim).to(f64)
        y = y * (zf * torch.sigmoid(zf))
    return y.reshape(x.shape), states


def _outside_tolerance(actual, reference):
    """``|actual - reference| > atol + rtol * |reference|`` elementwise."""

    difference = (actual.to(torch.float64) - reference).abs()
    return difference > _ATOL + _RTOL * reference.abs()


def _bf16_ulp(value):
    """Spacing of bf16 values at magnitude ``value`` (smallest normal below)."""

    magnitude = max(abs(value), torch.finfo(torch.bfloat16).tiny)
    return 2.0 ** math.floor(math.log2(magnitude)) * torch.finfo(torch.bfloat16).eps


def _count_slack(count):
    """Two-sided Poisson noise of an outlier count: two kernels of equal
    internal accuracy differ by about this many outliers on the same inputs."""

    return 2.0 * math.sqrt(count)


def _assert_cake_accuracy(
    cake,
    cute,
    constructor,
    tensors,
    arguments,
    *,
    max_outside_fraction=_MAX_OUTSIDE_FRACTION,
):
    """Cake must be at least as accurate as CuTe against the fp64 recurrence.

    Both backends evaluate the same chunked algorithm with bf16 operands, so
    each is outside ``atol = rtol = 1e-2`` of the exact recurrence on a sparse
    set of outputs where the cancellation in ``C . state`` amplifies one
    operand rounding.  The Cake kernels store the per-token ``delta`` in fp16
    (CAKE-942); CuTe stores it in bf16, so the two kernels round *differently*
    and their outlier sets no longer coincide: elementwise Cake-vs-CuTe parity
    at 1e-2 (the previous oracle) fails on 0.0-0.2 % of the outputs although
    Cake is the more accurate kernel.  Measured on GB300
    (``benchmarks/cake_ssd_accuracy_probe.py``; outputs outside 1e-2 of the fp64
    recurrence, Cake vs CuTe): H8/G8 batched 23 vs 79 of 131072, H128/G8
    batched S=1024 5352 vs 16813 of 16.8 M, realistic decay varlen 8x128
    0.573 % vs 1.403 %.  CuTe measured against a reference with a bf16-rounded
    ``delta`` drops to Cake's count (5163 of 16.8 M): the whole difference
    between the kernels is the ``delta`` rounding.  Final states: 0 entries
    outside tolerance in every case for Cake.

    Asserted for ``out`` and for ``final_states``:

    1. every Cake value is finite;
    2. Cake is within ``atol = rtol = 1e-2`` of the fp64 recurrence on all but
       at most ``max_outside_fraction`` of the entries (default 1 %; the
       elementwise tolerance is unchanged; the default is >= 6x above the
       measured fractions on the ``_case`` distribution and ~2x on the
       realistic one; a caller whose distribution puts *both* kernels above
       it passes the measured class error explicitly, see
       ``test_cake_ssd_combined_exact_scan_softplus_parity``);
    3. Cake has no more entries outside that tolerance than CuTe on the same
       inputs, up to the Poisson noise ``2 * sqrt(CuTe's count)`` of the
       count (measured 0.27-0.75x of CuTe's count over 15 probe
       configurations; the one near-tie is the softplus-off f16-state row,
       1438 vs 1432 of 131072, where the ``delta`` rounding plays no role);
    4. Cake's largest absolute error does not exceed CuTe's by more than one
       bf16 ulp at the magnitude of Cake's worst entry (kernels of equal
       internal accuracy differ by up to one output rounding there; the
       measured excess is <= 0.8 % of CuTe's maximum, far below one ulp).

    This is not a loosened CuTe parity: CuTe itself is outside 1e-2 of the
    recurrence on 0.04-1.4 % of its outputs, so even an exact kernel would
    fail the old oracle, while this one requires Cake to beat CuTe against
    the truth.
    """

    reference = _fp64_reference(constructor, tensors, arguments)
    for name, actual, baseline, expected in (
        ("out", cake[0], cute[0], reference[0]),
        ("final_states", cake[1], cute[1], reference[1]),
    ):
        assert tuple(actual.shape) == tuple(expected.shape), (
            name,
            tuple(actual.shape),
            tuple(expected.shape),
        )
        actual64 = actual.to(torch.float64)
        assert torch.isfinite(actual64).all(), (
            f"{name}: Cake produced non-finite values"
        )
        cake_outside = int(_outside_tolerance(actual, expected).sum())
        cute_outside = int(_outside_tolerance(baseline, expected).sum())
        cake_error = (actual64 - expected).abs()
        cake_max = float(cake_error.max())
        cute_max = float((baseline.to(torch.float64) - expected).abs().max())
        budget = max_outside_fraction * expected.numel()
        assert cake_outside <= budget, (
            f"{name}: {cake_outside} of {expected.numel()} Cake entries outside "
            f"atol=rtol={_ATOL} of the fp64 recurrence (budget {budget:.0f}; "
            f"CuTe {cute_outside})"
        )
        assert cake_outside <= cute_outside + _count_slack(cute_outside), (
            f"{name}: Cake has {cake_outside} entries outside atol=rtol={_ATOL} "
            f"of the fp64 recurrence, CuTe {cute_outside} on the same inputs "
            f"(slack {_count_slack(cute_outside):.0f})"
        )
        worst = int(cake_error.argmax())
        slack = _bf16_ulp(float(expected.reshape(-1)[worst]))
        assert cake_max <= cute_max + slack, (
            f"{name}: Cake max abs error {cake_max:.4g} exceeds CuTe's "
            f"{cute_max:.4g} by more than one bf16 ulp ({slack:.4g}) at the "
            f"worst entry (flat index {worst}, reference "
            f"{float(expected.reshape(-1)[worst]):.4g})"
        )


def _varlen_metadata(lengths, dtype):
    """Packed-varlen ``seq_idx`` / logical-chunk metadata; ``sum(lengths)`` may
    end inside a physical chunk (the trailing chunk is then partial)."""

    total = sum(lengths)
    seq_idx = torch.empty((1, total), dtype=dtype, device="cuda")
    start = 0
    for sequence, length in enumerate(lengths):
        seq_idx[0, start : start + length] = sequence
        start += length
    chunk_indices = []
    chunk_offsets = []
    for chunk in range(-(-total // 128)):
        values = seq_idx[0, chunk * 128 : (chunk + 1) * 128]
        previous = torch.cat((values[:1] - 1, values[:-1]))
        for offset in (values != previous).nonzero(as_tuple=True)[0].tolist():
            chunk_indices.append(chunk)
            chunk_offsets.append(offset)
    return (
        seq_idx,
        torch.tensor(chunk_indices, dtype=torch.int32, device="cuda"),
        torch.tensor(chunk_offsets, dtype=torch.int32, device="cuda"),
    )


def _seq_chunk_cumsum(lengths):
    """Exclusive prefix sum of per-sequence logical chunk counts (a sequence
    without tokens owns no logical chunk)."""

    cumsum = [0]
    start = 0
    for length in lengths:
        end = start + length
        chunks = (-(-end // 128) - start // 128) if length else 0
        cumsum.append(cumsum[-1] + chunks)
        start = end
    return torch.tensor(cumsum, dtype=torch.int32, device="cuda")


def _without_empty_sequences(constructor, arguments, lengths):
    """The same packed problem restricted to the ids that own tokens (the CuTe
    backend requires every id to own one): ids renumbered densely, metadata
    rebuilt, ``initial_states`` rows of the empty ids dropped.  Returns the
    dense arguments, the dense lengths and the kept (non-empty) ids."""

    kept = [sequence for sequence, length in enumerate(lengths) if length]
    dense_lengths = [lengths[sequence] for sequence in kept]
    seq_idx, chunk_indices, chunk_offsets = _varlen_metadata(
        dense_lengths, constructor["seq_idx_dtype"]
    )
    dense = {
        **arguments,
        "seq_idx": seq_idx,
        "chunk_indices": chunk_indices,
        "chunk_offsets": chunk_offsets,
        "seq_chunk_cumsum": _seq_chunk_cumsum(dense_lengths),
    }
    if arguments.get("initial_states") is not None:
        dense["initial_states"] = arguments["initial_states"][kept].contiguous()
    if arguments.get("num_seqs") is not None:
        dense["num_seqs"] = len(kept)
    return dense, dense_lengths, kept


def _case(
    *,
    nheads=8,
    ngroups=8,
    state_dtype=torch.bfloat16,
    varlen=False,
    seq_idx_dtype=torch.int32,
    preprocess_dtype=torch.float32,
    d_has_hdim=True,
    seqlen=None,
    lengths=(96, 160),
    initial_states=True,
    seed=7,
    batch=2,
):
    torch.manual_seed(seed)
    if varlen:
        batch, seqlen = 1, sum(lengths)
    else:
        seqlen = 128 if seqlen is None else seqlen
    x = torch.randn(batch, seqlen, nheads, 64, device="cuda").to(torch.bfloat16)
    dt = torch.randn(batch, seqlen, nheads, device="cuda").to(preprocess_dtype)
    A = -torch.rand(nheads, device="cuda", dtype=torch.float32) - 1.0
    B = torch.randn(batch, seqlen, ngroups, 128, device="cuda").to(torch.bfloat16)
    C = torch.randn_like(B)
    d_shape = (nheads, 64) if d_has_hdim else (nheads,)
    D = torch.randn(*d_shape, device="cuda").to(torch.bfloat16)
    z = torch.randn_like(x)
    dt_bias = (torch.rand(nheads, device="cuda", dtype=torch.float32) - 4.0).to(
        preprocess_dtype
    )
    state_batch = len(lengths) if varlen else batch
    initial_states_tensor = (
        torch.randn(state_batch, nheads, 64, 128, device="cuda").to(state_dtype)
        if initial_states
        else None
    )
    if varlen:
        seq_idx, chunk_indices, chunk_offsets = _varlen_metadata(lengths, seq_idx_dtype)
        seq_chunk_cumsum = _seq_chunk_cumsum(lengths)
    else:
        seq_idx = chunk_indices = chunk_offsets = seq_chunk_cumsum = None

    constructor = dict(
        chunk_size=128,
        nheads=nheads,
        headdim=64,
        dstate=128,
        ngroups=ngroups,
        io_dtype=torch.bfloat16,
        state_dtype=state_dtype,
        has_d=True,
        d_has_hdim=d_has_hdim,
        has_initial_states=initial_states,
        has_varlen=varlen,
        has_z=True,
        seq_idx_dtype=seq_idx_dtype,
    )
    arguments = dict(
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=True,
        dt_limit=(0.001, 0.1),
        initial_states=initial_states_tensor,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        seq_chunk_cumsum=seq_chunk_cumsum,
        return_final_states=True,
    )
    return constructor, (x, dt, A, B, C), arguments


def _cute_padded_reference(constructor, tensors, arguments, lengths):
    """CuTe reference for a Cake problem the CuTe backend cannot run directly.

    The Cake inputs (batched ``[B, S]`` or packed ``[1, T]``) are flattened
    into one packed stream of ``lengths`` sequences, zero-padded with one
    extra sequence to a multiple of 128 tokens, and run through CuTe varlen
    with explicit (zero when absent) initial states.  Returns the token-major
    output restricted to the real tokens and the per-sequence final states.
    """

    x, dt, A, B, C = tensors
    nheads = constructor["nheads"]
    total = sum(lengths)
    assert x.shape[0] * x.shape[1] == total
    padded = -(-total // 128) * 128
    pad = padded - total

    def stream(value):
        flat = value.reshape(1, total, *value.shape[2:])
        if pad == 0:
            return flat.contiguous()
        padding = torch.zeros(
            (1, pad, *value.shape[2:]), dtype=value.dtype, device=value.device
        )
        return torch.cat((flat, padding), dim=1).contiguous()

    initial_states = arguments["initial_states"]
    if initial_states is None:
        initial_states = torch.zeros(
            (len(lengths), nheads, 64, 128),
            dtype=constructor["state_dtype"],
            device="cuda",
        )
    if pad:
        initial_states = torch.cat(
            (initial_states, torch.zeros_like(initial_states[:1])), dim=0
        )
    padded_lengths = [*lengths, pad] if pad else list(lengths)
    seq_idx, chunk_indices, chunk_offsets = _varlen_metadata(
        padded_lengths, constructor["seq_idx_dtype"]
    )
    cute_constructor = {
        **constructor,
        "has_initial_states": True,
        "has_varlen": True,
    }
    cute_arguments = {
        **arguments,
        "z": stream(arguments["z"]),
        "initial_states": initial_states,
        "seq_idx": seq_idx,
        "chunk_indices": chunk_indices,
        "chunk_offsets": chunk_offsets,
        "seq_chunk_cumsum": _seq_chunk_cumsum(padded_lengths),
    }
    # The CuTe backend takes the sequence count from ``initial_states``.
    cute_arguments.pop("num_seqs", None)
    out, final = SSDCombined(**cute_constructor, backend="cute").run(
        stream(x), stream(dt), A, stream(B), stream(C), **cute_arguments
    )
    return out[:, :total].reshape(x.shape), final[: len(lengths)]


def test_cake_benchmark_validation_policy():
    """The benchmark gates a row on accuracy against the fp64 recurrence, not CuTe parity."""

    module = _load_cake_benchmark_module()

    def accuracy(
        *,
        cake_outside=20,
        cute_outside=60,
        cake_max=0.12,
        cute_max=0.12,
        finite=True,
        numel=131072,
    ):
        return {
            "numel": numel,
            "cake_outside": cake_outside,
            "cute_outside": cute_outside,
            "cake_outside_fraction": cake_outside / numel,
            "cute_outside_fraction": cute_outside / numel,
            "cake_max_abs": cake_max,
            "cute_max_abs": cute_max,
            "cake_finite": finite,
            "cute_finite": True,
            "bf16_ulp_at_cake_worst": 0.03125,
        }

    def report(*, out=None, final_states=None, speedup=1.01):
        return {
            "accuracy": {
                "out": out if out is not None else accuracy(),
                "final_states": final_states
                if final_states is not None
                else accuracy(cake_outside=0, cute_outside=0),
            },
            "speedup": speedup,
        }

    module._validate_report(report(), require_qualified_row=False)
    module._validate_report(report(speedup=0.99), require_qualified_row=False)
    # Poisson slack: CuTe 60 outliers admit up to 60 + 2*sqrt(60) = 75 Cake outliers.
    module._validate_report(
        report(out=accuracy(cake_outside=75)), require_qualified_row=False
    )
    with pytest.raises(AssertionError, match="more entries outside"):
        module._validate_report(
            report(out=accuracy(cake_outside=76)), require_qualified_row=False
        )
    with pytest.raises(AssertionError, match="limit 1 %"):
        module._validate_report(
            report(out=accuracy(cake_outside=1400, cute_outside=5000)),
            require_qualified_row=False,
        )
    with pytest.raises(AssertionError, match="is not finite"):
        module._validate_report(
            report(out=accuracy(finite=False)), require_qualified_row=False
        )
    # one bf16 ulp of headroom on the maximum error, no more
    module._validate_report(
        report(out=accuracy(cake_max=0.15, cute_max=0.12)), require_qualified_row=False
    )
    with pytest.raises(AssertionError, match="by more than one bf16 ulp"):
        module._validate_report(
            report(out=accuracy(cake_max=0.16, cute_max=0.12)),
            require_qualified_row=False,
        )
    with pytest.raises(AssertionError, match="final_states has more entries outside"):
        module._validate_report(
            report(final_states=accuracy(cake_outside=1, cute_outside=0)),
            require_qualified_row=False,
        )
    with pytest.raises(AssertionError, match="must be faster than CuTe"):
        module._validate_report(report(speedup=0.99), require_qualified_row=True)


def _strided_last_dim(value):
    storage = torch.empty(
        (*value.shape[:-1], value.shape[-1] + 1),
        dtype=value.dtype,
        device=value.device,
    )
    view = storage[..., : value.shape[-1]]
    view.copy_(value)
    assert not view.is_contiguous()
    return view


def _sglang_projection_view(value):
    active_width = value.numel() // (value.shape[0] * value.shape[1])
    storage = torch.empty(
        (value.shape[0], value.shape[1], active_width + 8),
        dtype=value.dtype,
        device=value.device,
    )
    view = storage[..., :active_width].view(value.shape)
    view.copy_(value)
    assert not view.is_contiguous() and view.stride(-1) == 1
    return view


@pytest.mark.parametrize(
    "state_dtype,varlen,seq_idx_dtype,nheads,ngroups,preprocess_dtype,d_has_hdim",
    [
        (torch.bfloat16, False, torch.int32, 8, 8, torch.float32, True),
        (torch.float16, False, torch.int32, 8, 8, torch.float32, False),
        (torch.bfloat16, True, torch.int32, 8, 8, torch.float32, True),
        (torch.float16, True, torch.int64, 8, 8, torch.float32, False),
        (torch.bfloat16, False, torch.int32, 8, 8, torch.bfloat16, False),
        # Dynamic public-API boundaries: minimum head/group and one group/head.
        (torch.bfloat16, False, torch.int32, 1, 1, torch.float32, False),
        (torch.bfloat16, False, torch.int32, 12, 3, torch.float32, False),
        (torch.bfloat16, False, torch.int32, 16, 4, torch.float32, False),
        (torch.bfloat16, False, torch.int32, 128, 1, torch.float32, False),
        (torch.bfloat16, False, torch.int32, 128, 128, torch.float32, False),
        # NVIDIA Nemotron-H-8B-Base-8K single-GPU local Mamba shape.
        (torch.bfloat16, False, torch.int32, 128, 8, torch.float32, False),
        (torch.bfloat16, True, torch.int32, 128, 8, torch.float32, False),
    ],
)
def test_cake_ssd_combined_route_matrix(
    state_dtype,
    varlen,
    seq_idx_dtype,
    nheads,
    ngroups,
    preprocess_dtype,
    d_has_hdim,
):
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake SSDCombined requires SM100, SM103 or SM107")

    constructor, tensors, arguments = _case(
        nheads=nheads,
        ngroups=ngroups,
        state_dtype=state_dtype,
        varlen=varlen,
        seq_idx_dtype=seq_idx_dtype,
        preprocess_dtype=preprocess_dtype,
        d_has_hdim=d_has_hdim,
    )
    if nheads == 128 and ngroups == 8:
        # Nemotron-H prefill starts from zero state and uses the unbounded
        # positive-dt interval in both batched and variable-length modes.
        # Finite-clamp and nonzero-initial-state feature rows remain covered
        # independently above; do not invent their Cartesian product with the
        # model-derived head shape.
        arguments["initial_states"].zero_()
        arguments["dt_limit"] = (0.0, float("inf"))
    expected = SSDCombined(**constructor, backend="cute").run(*tensors, **arguments)
    actual = SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)
    _assert_cake_accuracy(actual, expected, constructor, tensors, arguments)


def test_cake_ssd_combined_accepts_framework_strided_input_views():
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake SSDCombined requires SM100, SM103 or SM107")

    constructor, tensors, arguments = _case(varlen=True)
    expected = SSDCombined(**constructor, backend="cute").run(*tensors, **arguments)
    runner = SSDCombined(**constructor, backend="cake")
    contiguous = runner.run(*tensors, **arguments)
    x, dt, A, B, C = tensors
    tensors = (
        _sglang_projection_view(x),
        _sglang_projection_view(dt),
        A,
        _sglang_projection_view(B),
        _sglang_projection_view(C),
    )
    arguments = {
        **arguments,
        "z": _strided_last_dim(arguments["z"]),
        "initial_states": _strided_last_dim(arguments["initial_states"]),
    }

    actual = runner.run(*tensors, **arguments)
    # The views carry the same values, so the kernels must reproduce the
    # contiguous run bit for bit.
    torch.testing.assert_close(actual[0], contiguous[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], contiguous[1], rtol=0, atol=0)
    _assert_cake_accuracy(actual, expected, constructor, tensors, arguments)


@pytest.mark.parametrize(
    "d_has_hdim,runtime_d_has_hdim", [(True, False), (False, True)]
)
def test_cake_ssd_combined_matches_cute_d_shape_coercion(
    d_has_hdim, runtime_d_has_hdim
):
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake SSDCombined requires SM100, SM103 or SM107")

    constructor, tensors, arguments = _case(d_has_hdim=d_has_hdim)
    d_shape = (8, 64) if runtime_d_has_hdim else (8,)
    arguments["D"] = torch.randn(*d_shape, device="cuda").to(torch.bfloat16)
    expected = SSDCombined(**constructor, backend="cute").run(*tensors, **arguments)
    actual = SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)

    _assert_cake_accuracy(actual, expected, constructor, tensors, arguments)


def test_cake_ssd_combined_updates_caller_buffers():
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake SSDCombined requires SM100, SM103 or SM107")

    constructor, tensors, arguments = _case(varlen=True)
    expected = SSDCombined(**constructor, backend="cute").run(*tensors, **arguments)
    expected_cumsum = arguments["seq_chunk_cumsum"]
    actual_cumsum = torch.full_like(expected_cumsum, -1)
    # Token-major caller storage [batch, seqlen, nheads, headdim], written by
    # the kernels and returned as-is.
    out = torch.empty((1, 256, 8, 64), dtype=torch.bfloat16, device="cuda")
    runner = SSDCombined(**constructor, backend="cake")
    cake = runner._cake_runner
    # ``_case``'s triple form carries the caller's trusted table
    # (``update_seq_chunk_cumsum=False``): no kernel publishes metadata, so
    # the call runs the fused single-launch program (trusted-table fused route);
    # 2 sequences x 8 heads over 256 tokens sit inside the fused window.
    allocated = runner.run(*tensors, **arguments)
    program_trusted = cake.last_program_name
    assert program_trusted == "exactfused_bf16_varlen"
    _assert_cake_accuracy(allocated, expected, constructor, tensors, arguments)
    # Without a caller table the table is derived by the program of the
    # capability: on SM100 the light metadata kernel publishes the runner's
    # vector, then the same fused main kernel runs (``exactmeta``, inside the
    # fused window; two launches); on SM103 and SM107 the fused main kernel
    # derives its private table in a CTA prologue (``exactone``, one launch).
    derived = runner.run(
        *tensors, **{**arguments, "seq_chunk_cumsum": None, "num_seqs": 2}
    )
    program_derived = cake.last_program_name
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    derived_family = _derived_family(module, (128, 128))
    assert module._PROGRAMS[program_derived].family == derived_family
    assert program_derived.startswith(f"{derived_family}_bf16_varlen")
    actual = runner.run(
        *tensors,
        **{
            **arguments,
            "seq_chunk_cumsum": actual_cumsum,
            "update_seq_chunk_cumsum": True,
            "out": out,
        },
    )

    # Publishing into the caller's buffer needs a publishing stage, and the
    # kernel-written table is the host's: on SM100 the derived form's
    # exactmeta program again (its metadata kernel writes the caller's
    # buffer); on SM103 and SM107 the two-kernel program (the one-launch
    # exactone program keeps its table private to each CTA, so that form
    # keeps the publishing route), its segment preprocess writes the buffer.
    program_publishing = cake.last_program_name
    if capability == (10, 0):
        assert program_publishing == program_derived
    else:
        _two_kernel_program(module, program_publishing, num_sequences=2, seqlen=256)
    torch.testing.assert_close(actual_cumsum, expected_cumsum, rtol=0, atol=0)
    assert actual[0] is out
    _assert_cake_accuracy(actual, expected, constructor, tensors, arguments)
    # Caller-owned output storage and a kernel-written cumsum must not change
    # the arithmetic (a same-program statement): on SM100 bit-identical to the
    # derived run of the same program; on SM103 and SM107 bit-identical to a
    # second publishing run of the same two-kernel program into a fresh buffer
    # and allocated output storage.
    if capability == (10, 0):
        same_program = derived
    else:
        again_cumsum = torch.full_like(expected_cumsum, -1)
        same_program = runner.run(
            *tensors,
            **{
                **arguments,
                "seq_chunk_cumsum": again_cumsum,
                "update_seq_chunk_cumsum": True,
            },
        )
        assert cake.last_program_name == program_publishing
        torch.testing.assert_close(again_cumsum, expected_cumsum, rtol=0, atol=0)
    torch.testing.assert_close(actual[0], same_program[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], same_program[1], rtol=0, atol=0)

    preserved_cumsum = actual_cumsum.clone()
    trusted_again = runner.run(
        *tensors,
        **{
            **arguments,
            "seq_chunk_cumsum": actual_cumsum,
            "update_seq_chunk_cumsum": False,
        },
    )
    # A trusted caller table is never written, selects the single-launch
    # program again and reproduces the allocating run bit for bit.
    assert cake.last_program_name == program_trusted
    torch.testing.assert_close(actual_cumsum, preserved_cumsum, rtol=0, atol=0)
    torch.testing.assert_close(trusted_again[0], allocated[0], rtol=0, atol=0)
    torch.testing.assert_close(trusted_again[1], allocated[1], rtol=0, atol=0)


@pytest.mark.parametrize("state_dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("varlen", (False, True), ids=("batched", "varlen"))
def test_cake_ssd_combined_writes_selected_checkpoint_state(varlen, state_dtype):
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake SSDCombined requires SM100, SM103 or SM107")

    constructor, tensors, arguments = _case(
        varlen=varlen,
        state_dtype=state_dtype,
    )
    sequence_index = 1 if varlen else 0
    sequence_start = 96 if varlen else 0
    checkpoint_length = 128
    checkpoint_token = (
        sequence_start + checkpoint_length if varlen else checkpoint_length
    )
    checkpoint_states = torch.full(
        (3, *arguments["initial_states"].shape[1:]),
        torch.nan,
        dtype=arguments["initial_states"].dtype,
        device="cuda",
    )
    checkpoint_state = checkpoint_states[2:3]
    if varlen:
        full_arguments = {
            **arguments,
            # Expose sequence 1's checkpoint inside physical chunk 1 as a logical
            # segment boundary. This is the packed shape used by SGLang.
            "chunk_indices": torch.tensor(
                [0, 0, 1, 1], dtype=torch.int32, device="cuda"
            ),
            "chunk_offsets": torch.tensor(
                [0, 96, 0, 96], dtype=torch.int32, device="cuda"
            ),
            "seq_chunk_cumsum": torch.tensor(
                [0, 1, 4], dtype=torch.int32, device="cuda"
            ),
            "checkpoint_token_indices": torch.tensor(
                [-1, checkpoint_token], dtype=torch.int32, device="cuda"
            ),
            "checkpoint_state_slots": torch.tensor(
                [-1, 2], dtype=torch.int32, device="cuda"
            ),
            "checkpoint_states": checkpoint_states,
        }
    else:
        full_arguments = {
            **arguments,
            "checkpoint_token_indices": torch.tensor(
                [checkpoint_token, -1], dtype=torch.int32, device="cuda"
            ),
            "checkpoint_state_slots": torch.tensor(
                [2, -1], dtype=torch.int32, device="cuda"
            ),
            "checkpoint_states": checkpoint_states,
        }
    SSDCombined(**constructor, backend="cake").run(*tensors, **full_arguments)

    x, dt, A, B, C = tensors
    packed_batch_index = 0 if varlen else sequence_index
    prefix_tensors = (
        x[
            packed_batch_index : packed_batch_index + 1,
            sequence_start:checkpoint_token,
        ].contiguous(),
        dt[
            packed_batch_index : packed_batch_index + 1,
            sequence_start:checkpoint_token,
        ].contiguous(),
        A,
        B[
            packed_batch_index : packed_batch_index + 1,
            sequence_start:checkpoint_token,
        ].contiguous(),
        C[
            packed_batch_index : packed_batch_index + 1,
            sequence_start:checkpoint_token,
        ].contiguous(),
    )
    prefix_arguments = {
        **arguments,
        "z": arguments["z"][
            packed_batch_index : packed_batch_index + 1,
            sequence_start:checkpoint_token,
        ].contiguous(),
        "initial_states": arguments["initial_states"][
            sequence_index : sequence_index + 1
        ].contiguous(),
    }
    prefix_constructor = {**constructor, "has_varlen": varlen}
    if varlen:
        prefix_arguments.update(
            seq_idx=torch.zeros(
                (1, checkpoint_length), dtype=torch.int32, device="cuda"
            ),
            chunk_indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
            chunk_offsets=torch.zeros(1, dtype=torch.int32, device="cuda"),
            seq_chunk_cumsum=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        )
    _, expected_state = SSDCombined(**prefix_constructor, backend="cute").run(
        *prefix_tensors, **prefix_arguments
    )
    torch.testing.assert_close(
        checkpoint_state,
        expected_state,
        atol=1e-2,
        rtol=1e-2,
    )
    assert torch.isnan(checkpoint_states[:2]).all()


def test_cake_ssd_combined_allocation_output_lifetime():
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake SSDCombined requires SM100, SM103 or SM107")

    constructor, tensors, arguments = _case()
    runner = SSDCombined(**constructor, backend="cake")
    first, first_final = runner.run(*tensors, **arguments)
    retained = first.clone()
    retained_final = first_final.clone()
    second, second_final = runner.run(*tensors, **arguments)

    assert first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()
    assert (
        first_final.untyped_storage().data_ptr()
        != second_final.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(first, retained, rtol=0, atol=0)
    torch.testing.assert_close(first_final, retained_final, rtol=0, atol=0)

    without_final = runner.run(*tensors, **{**arguments, "return_final_states": False})
    assert isinstance(without_final, tuple)
    assert without_final[1] is None


@pytest.mark.parametrize("varlen", (False, True), ids=("batched", "varlen"))
def test_cake_ssd_combined_f32_state_matches_cute_on_bf16_representable_states(
    varlen,
):
    """FP32 state programs: CuTe has no FP32 state, so feed both backends the
    same bf16-representable initial states (exactly the same values in either
    dtype); the fp32 final states are measured unrounded against the fp64
    recurrence and must be at least as accurate as CuTe's bf16 states."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake SSDCombined requires SM100, SM103 or SM107")

    constructor, tensors, arguments = _case(varlen=varlen, state_dtype=torch.float32)
    bf16_states = arguments["initial_states"].to(torch.bfloat16)
    cute_constructor = {**constructor, "state_dtype": torch.bfloat16}
    cute_arguments = {**arguments, "initial_states": bf16_states}
    arguments["initial_states"] = bf16_states.to(torch.float32)

    expected = SSDCombined(**cute_constructor, backend="cute").run(
        *tensors, **cute_arguments
    )
    actual = SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)

    assert actual[1].dtype == torch.float32
    _assert_cake_accuracy(actual, expected, constructor, tensors, arguments)


@pytest.mark.parametrize(
    "varlen,lengths",
    [
        (False, (1000, 1000)),
        (True, (1000,)),
        (True, (128, 900)),
        (True, (300, 300)),
    ],
    ids=("batched_2x1000", "varlen_1000", "varlen_128_900", "varlen_300_300"),
)
def test_cake_ssd_combined_accepts_unaligned_seqlen(varlen, lengths):
    """``seqlen % 128 != 0``: the partial trailing physical chunk is handled
    in-kernel; the CuTe baseline runs on a zero-padded packed stream."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake SSDCombined requires SM100, SM103 or SM107")

    if varlen:
        constructor, tensors, arguments = _case(varlen=True, lengths=lengths)
    else:
        constructor, tensors, arguments = _case(seqlen=lengths[0])
    assert tensors[0].shape[1] % 128 != 0 or sum(lengths) % 128 != 0
    expected = _cute_padded_reference(constructor, tensors, arguments, lengths)

    actual = SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)

    assert tuple(actual[0].shape) == tuple(tensors[0].shape)
    _assert_cake_accuracy(actual, expected, constructor, tensors, arguments)


_SHORT_TOTAL_TOKEN_CASES = (
    (True, (1,)),
    (True, (8,)),
    (True, (127,)),
    (True, (64, 60)),
    (False, (50, 50)),
    (False, (1,)),
)
_SHORT_TOTAL_TOKEN_IDS = (
    "varlen_1",
    "varlen_8",
    "varlen_127",
    "varlen_64_60",
    "batched_2x50",
    "batched_1x1",
)


def _short_total_tokens_case(varlen, lengths, state_dtype, seed=7):
    """A ``_case`` with fewer than 128 tokens per call (packed varlen
    ``lengths`` or batched ``len(lengths) x lengths[0]``) and its CuTe padded
    oracle.  CuTe has no fp32 state, so an fp32 row starts both backends from
    the same bf16-representable initial states (as the fp32-state parity test
    does)."""

    if varlen:
        constructor, tensors, arguments = _case(
            varlen=True, lengths=lengths, state_dtype=state_dtype, seed=seed
        )
    else:
        assert len(set(lengths)) == 1
        constructor, tensors, arguments = _case(
            batch=len(lengths), seqlen=lengths[0], state_dtype=state_dtype, seed=seed
        )
    assert tensors[0].shape[1] < 128
    cute_constructor = constructor
    cute_arguments = arguments
    if state_dtype == torch.float32:
        bf16_states = arguments["initial_states"].to(torch.bfloat16)
        cute_constructor = {**constructor, "state_dtype": torch.bfloat16}
        cute_arguments = {**arguments, "initial_states": bf16_states}
        arguments["initial_states"] = bf16_states.to(torch.float32)
    expected = _cute_padded_reference(
        cute_constructor, tensors, cute_arguments, lengths
    )
    return constructor, tensors, arguments, expected


@pytest.mark.parametrize(
    "state_dtype",
    (torch.bfloat16, torch.float16, torch.float32),
    ids=("bf16", "f16", "f32"),
)
@pytest.mark.parametrize(
    "varlen,lengths", _SHORT_TOTAL_TOKEN_CASES, ids=_SHORT_TOTAL_TOKEN_IDS
)
def test_cake_ssd_combined_accepts_short_total_tokens(varlen, lengths, state_dtype):
    """CAKE-1063: fewer than 128 tokens per call (batched ``seqlen < 128``,
    packed varlen ``total < 128``, down to a single token).  The token axis
    of the x/B/C/out tensor maps admits a global extent below the 128-row
    box: TMA zero-fills the rows past ``seqlen`` on loads and clips them on
    stores, so the caller's tensors are bound directly; the result must
    match the fp64 recurrence at least as well as CuTe on the zero-padded
    packed stream.

    Single-token f16-state rows: the state is the initial state, so the
    rounding of the f16 initial state to the bf16 ``C . state`` MMA operand
    (done by both kernels; see the softplus-off row of
    ``test_cake_ssd_combined_exact_scan_softplus_parity``) dominates and
    both kernels put 6 of the 512 outputs outside 1e-2 of the recurrence on
    GB300 (1.17 %, identical counts), hence the same 2 % cap for those two
    rows; every other row stays under the 1 % default."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake SSDCombined requires SM100, SM103 or SM107")

    constructor, tensors, arguments, expected = _short_total_tokens_case(
        varlen, lengths, state_dtype
    )

    actual = SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)

    assert tuple(actual[0].shape) == tuple(tensors[0].shape)
    assert tuple(actual[1].shape) == (len(lengths), constructor["nheads"], 64, 128)
    assert actual[1].dtype == state_dtype
    single_token_f16 = state_dtype == torch.float16 and max(lengths) == 1
    _assert_cake_accuracy(
        actual,
        expected,
        constructor,
        tensors,
        arguments,
        max_outside_fraction=0.02 if single_token_f16 else _MAX_OUTSIDE_FRACTION,
    )


def test_cake_ssd_combined_short_call_after_nan_injection_is_clean():
    """CAKE-1063 stale-SMEM check in one process: a long call whose x/B/C rows
    carry NaN leaves NaN in every SMEM input/output stage of the persistent
    kernel; the short call that follows on the same runner must be NaN-free
    and match the reference, i.e. the pad rows the kernel consumes come from
    the TMA out-of-bounds zero fill, never from stale SMEM."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake SSDCombined requires SM100, SM103 or SM107")

    constructor, tensors, arguments = _case(varlen=True, lengths=(300,), seed=11)
    x, dt, A, B, C = tensors
    x, B, C = x.clone(), B.clone(), C.clone()
    x[:, 100:] = float("nan")
    B[:, 100:] = float("nan")
    C[:, 100:] = float("nan")
    runner = SSDCombined(**constructor, backend="cake")
    poisoned = runner.run(x, dt, A, B, C, **arguments)
    # The poison reached the pipeline (output and final state carry NaN).
    assert not torch.isfinite(poisoned[0].float()).all()
    assert not torch.isfinite(poisoned[1].float()).all()

    short_constructor, short_tensors, short_arguments, expected = (
        _short_total_tokens_case(True, (8,), torch.bfloat16, seed=12)
    )
    assert short_constructor == constructor

    actual = runner.run(*short_tensors, **short_arguments)

    assert torch.isfinite(actual[0].float()).all()
    assert torch.isfinite(actual[1].float()).all()
    _assert_cake_accuracy(
        actual, expected, short_constructor, short_tensors, short_arguments
    )


@pytest.mark.parametrize("count_source", ("num_seqs", "seq_chunk_cumsum"))
@pytest.mark.parametrize(
    "lengths", [(128,), (128, 128), (1000,)], ids=("1x128", "2x128", "1x1000")
)
def test_cake_ssd_combined_varlen_without_initial_states(lengths, count_source):
    """``initial_states=None`` in varlen mode starts from zero state; the
    sequence count comes from ``num_seqs`` or ``seq_chunk_cumsum``."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake SSDCombined requires SM100, SM103 or SM107")

    constructor, tensors, arguments = _case(
        varlen=True, lengths=lengths, initial_states=False
    )
    assert constructor["has_initial_states"] is False
    assert arguments["initial_states"] is None
    expected = _cute_padded_reference(constructor, tensors, arguments, lengths)
    if count_source == "num_seqs":
        arguments["seq_chunk_cumsum"] = None
        arguments["num_seqs"] = len(lengths)

    actual = SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)

    assert tuple(actual[1].shape) == (len(lengths), 8, 64, 128)
    _assert_cake_accuracy(actual, expected, constructor, tensors, arguments)


def test_cake_ssd_combined_varlen_without_initial_states_needs_a_count():
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake SSDCombined requires SM100, SM103 or SM107")

    constructor, tensors, arguments = _case(
        varlen=True, lengths=(128, 128), initial_states=False
    )
    arguments["seq_chunk_cumsum"] = None

    with pytest.raises(ValueError, match="requires seq_chunk_cumsum or num_seqs"):
        SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)


@pytest.mark.parametrize("table_source", ("preprocess", "caller"))
@pytest.mark.parametrize("initial_states", (True, False), ids=("initial", "zero"))
@pytest.mark.parametrize(
    "lengths,seq_idx_dtype",
    [
        ((128, 0, 200), torch.int32),
        ((0, 300, 0), torch.int32),
        ((100, 0, 0, 156), torch.int64),
    ],
    ids=("empty-middle", "empty-first-and-last", "two-empty-i64"),
)
def test_cake_ssd_combined_varlen_empty_sequences(
    lengths, seq_idx_dtype, initial_states, table_source
):
    """CAKE-990: a packed-sequence id without tokens is part of the contract.
    The preprocess (or the caller's table) gives it an empty chunk range; its
    final state is ``initial_states[id]`` (zero without initial states); the
    sequences that own tokens match the CuTe reference of the dense problem
    and the fp64 recurrence; the status word stays clear."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake SSDCombined requires SM100, SM103 or SM107")

    constructor, tensors, arguments = _case(
        varlen=True,
        lengths=lengths,
        seq_idx_dtype=seq_idx_dtype,
        initial_states=initial_states,
    )
    if table_source == "preprocess":
        arguments["seq_chunk_cumsum"] = None
        if not initial_states:
            arguments["num_seqs"] = len(lengths)
    dense_arguments, dense_lengths, kept = _without_empty_sequences(
        constructor, arguments, lengths
    )
    assert len(kept) < len(lengths)
    expected = _cute_padded_reference(
        constructor, tensors, dense_arguments, dense_lengths
    )
    runner = SSDCombined(**constructor, backend="cake")
    runner._cake_runner.seq_idx_status(reset=True)

    out, final = runner.run(*tensors, **arguments)

    assert runner._cake_runner.seq_idx_status(reset=True) == 0
    assert tuple(final.shape) == (len(lengths), 8, 64, 128)
    _assert_cake_accuracy(
        (out, final[kept]), expected, constructor, tensors, dense_arguments
    )
    for sequence, length in enumerate(lengths):
        if length:
            continue
        if initial_states:
            assert torch.equal(final[sequence], arguments["initial_states"][sequence])
        else:
            assert not final[sequence].to(torch.float32).any()


@pytest.mark.parametrize("family", ("exact", "exactpre"))
@pytest.mark.parametrize(
    "lengths,tail,seq_idx_dtype",
    [
        ((100, 150), 70, torch.int32),
        ((128, 128), 64, torch.int32),
        ((100, 150), 70, torch.int64),
    ],
    ids=("tail-shares-chunk", "tail-opens-chunk", "tail-shares-chunk-i64"),
)
def test_cake_ssd_combined_flags_out_of_range_seq_idx(
    monkeypatch, lengths, tail, seq_idx_dtype, family
):
    """CAKE-990: ``seq_idx`` ids ``0 .. num_seqs`` with ``num_seqs`` declared
    sequences (the trailing id is out of range).  No out-of-bounds write
    (the caller's ``[num_seqs + 1]`` table receives exactly the in-range
    boundaries), the status word is set, the in-range sequences are bitwise
    those of the clean problem and correct against CuTe / fp64, and a clean
    call afterwards leaves the reset word at 0.

    The bitwise clause is a same-program statement.  The two-kernel route
    picks the two-kernel program on the host from the mean chunks per item
    (``two_kernel_family``), and that quantity counts the out-of-range tail
    tokens too (whether an id is out of range is only known on the device),
    so the clean and the flagged call of ``tail-shares-chunk`` would run the
    retained start ``exact`` program and the round-6 ``exactpre`` program --
    the same scan with different rounding.  Both calls are therefore pinned
    to one family here, and the test runs once per family so both two-kernel
    programs are judged."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake SSDCombined requires SM100, SM103 or SM107")
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    monkeypatch.setattr(module, "two_kernel_family", lambda **_: family)

    num_seqs = len(lengths)
    total = sum(lengths)
    constructor, tensors, arguments = _case(
        varlen=True, lengths=(*lengths, tail), seq_idx_dtype=seq_idx_dtype
    )
    x, dt, A, B, C = tensors
    table = torch.full((num_seqs + 1,), -7, dtype=torch.int32, device="cuda")

    def expected_prefix(tokens):
        # Inside the fused window SM100 routes the triple form deriving its
        # table (the clean call, and the flagged call publishing into the
        # caller's buffer) to the ``exactmeta`` program: one program for both
        # calls, so the bitwise clause holds there without the family pin.
        # SM103 and SM107 keep the pinned two-kernel program.
        fused = capability == (10, 0) and module.fused_selected(
            nheads=constructor.get("nheads", 8),
            num_sequences=num_seqs,
            nchunks=-(-max(lengths) // 128),
            mode_varlen=True,
            total_tokens=tokens,
        )
        return "exactmeta_" if fused else f"{family}_"

    flagged_arguments = {
        **arguments,
        "initial_states": arguments["initial_states"][:num_seqs].contiguous(),
        "seq_chunk_cumsum": table,
        "update_seq_chunk_cumsum": True,
    }
    clean_tensors = tuple(
        value[:, :total].contiguous()
        if value.ndim >= 2 and value.shape[1] == total + tail
        else value
        for value in tensors
    )
    seq_idx, chunk_indices, chunk_offsets = _varlen_metadata(lengths, seq_idx_dtype)
    clean_arguments = {
        **flagged_arguments,
        "z": arguments["z"][:, :total].contiguous(),
        "seq_idx": seq_idx,
        "chunk_indices": chunk_indices,
        "chunk_offsets": chunk_offsets,
        "seq_chunk_cumsum": None,
        "update_seq_chunk_cumsum": False,
    }
    expected = _cute_padded_reference(
        constructor, clean_tensors, clean_arguments, lengths
    )
    # The clean call publishes into its own caller buffer too: on SM103 and
    # SM107 a runner-owned table would run the one-launch ``exactone`` program (the
    # fused rounding, not the pinned two-kernel family's) and the bitwise
    # clause below is a same-program statement; the one-launch program has
    # its own test (``test_cake_ssd_combined_exactone_flags_out_of_range_seq_idx``).
    clean_table = torch.full((num_seqs + 1,), -7, dtype=torch.int32, device="cuda")
    clean_arguments = {
        **clean_arguments,
        "seq_chunk_cumsum": clean_table,
        "update_seq_chunk_cumsum": True,
    }
    runner = SSDCombined(**constructor, backend="cake")
    cake = runner._cake_runner
    cake.seq_idx_status(reset=True)
    clean_out, clean_final = runner.run(*clean_tensors, **clean_arguments)
    assert cake.seq_idx_status() == 0
    assert cake.last_program_name.startswith(expected_prefix(total))
    assert clean_table.tolist() == _seq_chunk_cumsum(lengths).tolist()

    out, final = runner.run(*tensors, **flagged_arguments)

    assert cake.last_program_name.startswith(expected_prefix(total + tail))
    assert cake.seq_idx_status(reset=True) == 1
    # The out-of-range id closes the in-range ranges and opens none of its
    # own: table[num_seqs] is the first segment of the out-of-range tail.
    expected_table = _seq_chunk_cumsum(lengths).tolist()
    expected_table[-1] = int(_seq_chunk_cumsum((*lengths, tail))[num_seqs])
    assert table.tolist() == expected_table
    assert tuple(final.shape) == (num_seqs, 8, 64, 128)
    assert torch.equal(out[:, :total], clean_out)
    assert torch.equal(final, clean_final)
    _assert_cake_accuracy(
        (out[:, :total], final), expected, constructor, clean_tensors, clean_arguments
    )
    runner.run(*clean_tensors, **clean_arguments)
    assert cake.seq_idx_status(reset=True) == 0


@pytest.mark.parametrize(
    "lengths,tail,seq_idx_dtype",
    [
        ((100, 150), 70, torch.int32),
        ((128, 128), 64, torch.int32),
        ((100, 150), 70, torch.int64),
    ],
    ids=("tail-shares-chunk", "tail-opens-chunk", "tail-shares-chunk-i64"),
)
def test_cake_ssd_combined_exactone_flags_out_of_range_seq_idx(
    lengths, tail, seq_idx_dtype
):
    """SM103 and SM107, the one-launch ``exactone`` program (the seq_idx triple
    with a runner-owned table inside the fused window): its CTA prologue applies the
    publishing stages' sequence-boundary rule, so the out-of-range trailing
    id sets the status word, the in-range sequences are bitwise those of the
    clean one-launch call and correct against CuTe / fp64, and a clean call
    afterwards leaves the reset word at 0.  No caller table is involved: the
    derived rows stay private to each CTA."""

    if torch.cuda.get_device_capability() not in ((10, 3), (10, 7)):
        pytest.skip("the one-launch exactone program routes on SM103 and SM107 only")
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    num_seqs = len(lengths)
    total = sum(lengths)
    constructor, tensors, arguments = _case(
        varlen=True, lengths=(*lengths, tail), seq_idx_dtype=seq_idx_dtype
    )
    flagged_arguments = {
        **arguments,
        "initial_states": arguments["initial_states"][:num_seqs].contiguous(),
        "seq_chunk_cumsum": None,
        "update_seq_chunk_cumsum": False,
        "num_seqs": num_seqs,
    }
    clean_tensors = tuple(
        value[:, :total].contiguous()
        if value.ndim >= 2 and value.shape[1] == total + tail
        else value
        for value in tensors
    )
    seq_idx, chunk_indices, chunk_offsets = _varlen_metadata(lengths, seq_idx_dtype)
    clean_arguments = {
        **flagged_arguments,
        "z": arguments["z"][:, :total].contiguous(),
        "seq_idx": seq_idx,
        "chunk_indices": chunk_indices,
        "chunk_offsets": chunk_offsets,
    }
    assert module.fused_selected(
        nheads=constructor.get("nheads", 8),
        num_sequences=num_seqs,
        nchunks=-(-max(lengths) // 128),
        mode_varlen=True,
        total_tokens=total + tail,
    )
    expected = _cute_padded_reference(
        constructor, clean_tensors, clean_arguments, lengths
    )
    runner = SSDCombined(**constructor, backend="cake")
    cake = runner._cake_runner
    cake.seq_idx_status(reset=True)
    clean_out, clean_final = runner.run(*clean_tensors, **clean_arguments)
    assert cake.seq_idx_status() == 0
    assert cake.last_program_name.startswith("exactone_"), cake.last_program_name

    out, final = runner.run(*tensors, **flagged_arguments)

    assert cake.last_program_name.startswith("exactone_"), cake.last_program_name
    assert cake.seq_idx_status(reset=True) == 1
    assert tuple(final.shape) == (num_seqs, 8, 64, 128)
    assert torch.equal(out[:, :total], clean_out)
    assert torch.equal(final, clean_final)
    _assert_cake_accuracy(
        (out[:, :total], final), expected, constructor, clean_tensors, clean_arguments
    )
    runner.run(*clean_tensors, **clean_arguments)
    assert cake.seq_idx_status(reset=True) == 0


# ---------------------------------------------------------------------------
# cu_seqlens varlen form (CAKE-934 item 2, option B): the preprocess derives
# the 128-granularity segment tables on the device from cu_seqlens.
# ---------------------------------------------------------------------------

_CU_SEQLENS_LENGTHS = (
    (1,),
    (127,),
    (128,),
    (129,),
    (1000,),
    (64, 60),
    (128, 896),
    (96, 160),
    (1024, 100),
    (2048, 2048),
    (32768,),
    (128, 0, 200),
)
_HOST_CHUNK = 128
_HOST_PREPROCESS_THREADS = 128


def _cu_seqlens(lengths):
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + int(length))
    return torch.tensor(cu, dtype=torch.int32, device="cuda")


def _cu_seqlens_arguments(arguments, lengths, *, keep_seq_idx=False):
    """The same call on the cu_seqlens form: no chunk-128 triple (``seq_idx``
    may ride along, it is never read), no caller cumsum, no ``num_seqs``."""

    cu_arguments = {
        **arguments,
        "seq_idx": arguments["seq_idx"] if keep_seq_idx else None,
        "chunk_indices": None,
        "chunk_offsets": None,
        "seq_chunk_cumsum": None,
        "update_seq_chunk_cumsum": False,
        "cu_seqlens": _cu_seqlens(lengths),
    }
    cu_arguments.pop("num_seqs", None)
    return cu_arguments


def _host_sequence_count(lo, hi, checkpoint):
    """``nseg`` of one clamped sequence ``[lo, hi)`` (device closed form)."""

    if hi <= lo:
        return 0
    count = -(-hi // _HOST_CHUNK) - lo // _HOST_CHUNK
    if checkpoint is not None and lo < checkpoint < hi and checkpoint % _HOST_CHUNK:
        count += 1
    return count


def _host_boundary(lo, checkpoint, ordinal):
    """Start of the ``ordinal``-th segment of a sequence starting at ``lo``:
    the 128 grid points after the start, with an unaligned checkpoint inserted
    at rank ``cp // 128 - lo // 128 + 1`` shifting the later grid points."""

    first_chunk = lo // _HOST_CHUNK
    start = (first_chunk + ordinal) * _HOST_CHUNK
    if ordinal == 0:
        start = lo
    if checkpoint is not None:
        rank = checkpoint // _HOST_CHUNK - first_chunk + 1
        if ordinal == rank:
            start = checkpoint
        if ordinal > rank:
            start = (first_chunk + ordinal - 1) * _HOST_CHUNK
    return start


def _host_cu_seqlens_metadata(cu, total, checkpoints=None):
    """Host mirror of the preprocess derivation (Cake
    ``CU_SEQLENS_DERIVATION_RULE``; the Cake unit test
    ``loom/tests/infra/test_mamba_ssd_cu_seqlens_metadata.py`` proves it
    against the logical-chunk reference): ``[bound + 1]`` tables, the
    published cumsum, the sentinel and the invalid-input flag."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    num_sequences = len(cu) - 1
    bound = module._segment_bound(total, num_sequences)
    checkpoints = checkpoints or [None] * num_sequences
    flagged = False
    clamped, counts = [], []
    for sequence in range(num_sequences):
        raw_lo, raw_hi = cu[sequence], cu[sequence + 1]
        flagged |= sequence == 0 and raw_lo != 0
        flagged |= sequence == num_sequences - 1 and raw_hi != total
        flagged |= raw_hi < raw_lo or raw_lo < 0 or raw_hi > total
        lo = min(max(raw_lo, 0), total)
        hi = min(max(raw_hi, 0), total)
        checkpoint = checkpoints[sequence]
        if checkpoint is not None and not (
            lo < checkpoint < hi and checkpoint % _HOST_CHUNK
        ):
            checkpoint = None
        clamped.append((lo, hi, checkpoint))
        counts.append(_host_sequence_count(lo, hi, checkpoint))
    exclusive, carry = [], 0
    for block_base in range(0, num_sequences, _HOST_PREPROCESS_THREADS):
        for count in counts[block_base : block_base + _HOST_PREPROCESS_THREADS]:
            exclusive.append(carry)
            carry += count
    flagged |= carry > bound
    num_segments = min(carry, bound)
    chunk_indices = [None] * (bound + 1)
    chunk_offsets = [None] * (bound + 1)
    for segment in range(num_segments):
        owner = max(s for s in range(num_sequences) if exclusive[s] <= segment)
        lo, hi, checkpoint = clamped[owner]
        start = _host_boundary(lo, checkpoint, segment - exclusive[owner])
        chunk_indices[segment] = start // _HOST_CHUNK
        chunk_offsets[segment] = start % _HOST_CHUNK
    chunk_indices[num_segments] = -1
    return dict(
        bound=bound,
        flagged=flagged,
        num_segments=num_segments,
        seq_chunk_cumsum=[min(value, bound) for value in exclusive] + [num_segments],
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )


def _logical_chunk_metadata(lengths, extra_boundaries=()):
    """The sglang logical-chunk set (``_query_start_loc_to_chunk_indices_offsets``
    at chunk 128 plus the given boundaries) as int32 device vectors."""

    starts, start = set(int(b) for b in extra_boundaries), 0
    for length in lengths:
        starts.add(start)
        start += int(length)
    chunk_indices, chunk_offsets = [], []
    for chunk in range(-(-start // _HOST_CHUNK)):
        lo, hi = chunk * _HOST_CHUNK, (chunk + 1) * _HOST_CHUNK
        for offset in sorted({0} | {b - lo for b in starts if lo < b < hi}):
            chunk_indices.append(chunk)
            chunk_offsets.append(offset)
    return (
        torch.tensor(chunk_indices, dtype=torch.int32, device="cuda"),
        torch.tensor(chunk_offsets, dtype=torch.int32, device="cuda"),
    )


def _skip_unless_cake_arch():
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake SSDCombined requires SM100, SM103 or SM107")


def _assert_two_forms_p6(
    monkeypatch, *, lengths, state_dtype, initial_states, programs
):
    """P6 reading of the two packed-varlen metadata forms where they run
    different arithmetic (the host two-kernel route): on the K fixed seeds run the
    seq_idx triple form and the cu_seqlens form of the same inputs on the
    forced exact scan plus the forced three-phase program (bitwise the
    round-5 scan, the baseline reading), and judge each form's output and
    final states against the fp64 recurrence under clause GOAL6-A-20261012
    (``_assert_p6_clause``).  ``programs`` = (cu_seqlens program, triple
    program) the forms must keep running.  The triple form here derives the
    table on the device (no caller ``seq_chunk_cumsum``): with a trusted
    caller table it runs the fused program of the cu_seqlens form (F1b) and
    the two forms are bitwise, not a P6 reading."""

    candidates = {"cu_seqlens": {}, "triple": {}}
    baseline = {}
    # The forced modes below are scoped to this helper: the caller's later
    # calls must run the runner's own program choice again (a leaked
    # ``three_phase`` once made the caller's seq_idx ride-along
    # call run the three-phase program and fail its bitwise clause).
    prior_mode = os.environ.get(_CHUNK_PARALLEL_ENV)
    for seed in _P6_SEEDS:
        constructor, tensors, arguments = _case(
            varlen=True,
            lengths=lengths,
            state_dtype=state_dtype,
            initial_states=initial_states,
            seed=seed,
        )
        runner = SSDCombined(**constructor, backend="cake")
        derived_arguments = {
            **arguments,
            "seq_chunk_cumsum": None,
            "num_seqs": len(lengths),
        }
        monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "never")
        out_triple, final_triple = (
            value.clone() for value in runner.run(*tensors, **derived_arguments)
        )
        assert runner._cake_runner.last_program_name == programs[1]
        out_cu, final_cu = (
            value.clone()
            for value in runner.run(
                *tensors, **_cu_seqlens_arguments(arguments, lengths)
            )
        )
        assert runner._cake_runner.last_program_name == programs[0]
        monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "three_phase")
        out_base, final_base = (
            value.clone() for value in runner.run(*tensors, **derived_arguments)
        )
        assert runner._cake_runner.last_program_name.startswith("chunkpar3_")
        y, states = _fp64_reference(
            constructor, tensors, {**arguments, "num_seqs": len(lengths)}
        )
        for label, cu, triple, base, oracle in (
            ("out", out_cu, out_triple, out_base, y),
            ("final", final_cu, final_triple, final_base, states),
        ):
            candidates["cu_seqlens"].setdefault(label, []).append(
                _p6_metrics(cu, oracle)
            )
            candidates["triple"].setdefault(label, []).append(
                _p6_metrics(triple, oracle)
            )
            baseline.setdefault(label, []).append(_p6_metrics(base, oracle))
    if prior_mode is None:
        monkeypatch.delenv(_CHUNK_PARALLEL_ENV, raising=False)
    else:
        monkeypatch.setenv(_CHUNK_PARALLEL_ENV, prior_mode)
    for form, readings in candidates.items():
        program = programs[0 if form == "cu_seqlens" else 1]
        for label, metrics in readings.items():
            _assert_p6_clause(
                metrics, baseline[label], f"{form} form ({program}) {label}"
            )


@pytest.mark.parametrize("initial_states", (True, False), ids=("initial", "zero"))
@pytest.mark.parametrize(
    "state_dtype",
    (torch.bfloat16, torch.float16, torch.float32),
    ids=("bf16", "f16", "f32"),
)
@pytest.mark.parametrize(
    "lengths",
    _CU_SEQLENS_LENGTHS,
    ids=[
        "x".join(str(length) for length in lengths) for lengths in _CU_SEQLENS_LENGTHS
    ],
)
def test_cake_ssd_combined_cu_seqlens_form_is_bitwise_the_triple(
    monkeypatch, lengths, state_dtype, initial_states
):
    """The cu_seqlens form (the device derives seq_chunk_cumsum and the
    chunk-128 tables) matches the chunk-128 triple call of the same inputs on
    every sequence geometry, with ``seq_idx`` optional and never read; the
    status word stays clear.

    Where both forms run the same arithmetic they are bitwise.  The host
    route sends the two-kernel form of one-chunk items to the retained
    round-5 start program (``two_kernel_family``), while a cu_seqlens call
    inside the fused window runs the round-6 fused program (single-rounding
    FMA state update): those two forms differ by that rounding, so each form
    is judged on its own against the fp64 recurrence under the P6 clause
    GOAL6-A-20261012 (``_assert_two_forms_p6``: the three-phase program's
    reading as the round-5 baseline, ``_P6_SEEDS`` seeds) and the two forms
    are compared at the state dtype tolerance as a sanity check only (the
    tolerance never replaces the P6 reading).  The host contract -- ``run`` /
    ``prepare`` signatures and both metadata forms -- is unchanged.

    The triple form judged here derives ``seq_chunk_cumsum`` on the device
    (no caller table).  With the caller's trusted table instead
    (``_case``'s default, ``update_seq_chunk_cumsum=False``) nothing is
    published, so the triple runs the fused program of the cu_seqlens form
    wherever the fused window admits it (trusted-table fused route) and the two
    forms are then bitwise again; outside the window it runs the two-kernel
    program of the derived triple (last block)."""

    _skip_unless_cake_arch()
    constructor, tensors, arguments = _case(
        varlen=True,
        lengths=lengths,
        state_dtype=state_dtype,
        initial_states=initial_states,
    )
    derived_arguments = {
        **arguments,
        "seq_chunk_cumsum": None,
        "num_seqs": len(lengths),
    }
    runner = SSDCombined(**constructor, backend="cake")
    cake = runner._cake_runner
    cake.seq_idx_status(reset=True)
    out_triple, final_triple = (
        value.clone() for value in runner.run(*tensors, **derived_arguments)
    )
    assert cake.seq_idx_status() == 0
    program_triple = cake.last_program_name

    out, final = runner.run(*tensors, **_cu_seqlens_arguments(arguments, lengths))

    assert cake.seq_idx_status(reset=True) == 0
    program_cu = cake.last_program_name
    # The chunk-parallel rule reads the same host quantities in both forms
    # (round 6 made the families' rounding differ, so a form-dependent family
    # would break this contract before any element compares).  Within the
    # exact-scan family both forms run the fused main kernel inside the fused
    # window -- ``exactfused`` in the cu_seqlens form (one launch), the
    # capability's program for the triple deriving its table (``exactmeta``
    # on SM100, ``exactone`` on SM103 and SM107: ``_derived_family``) --
    # and the two-kernel program of the same variant outside it (``exactpre``
    # from two chunks per item, the retained start ``exact`` program for
    # one-chunk items, the host two-kernel route); the arithmetic is the same
    # in both forms, so they compare bitwise.
    same_arithmetic = True
    fused = False
    if program_triple.startswith("chunkpar_"):
        assert program_cu == program_triple, (
            f"cu_seqlens form ran {program_cu}, the triple {program_triple}"
        )
    else:
        module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
        two_kernel = module.two_kernel_family(
            mode_varlen=True, num_sequences=len(lengths), seqlen=sum(lengths)
        )
        fused = module.fused_selected(
            nheads=8,
            num_sequences=len(lengths),
            nchunks=-(-max(lengths) // 128),
            mode_varlen=True,
            total_tokens=sum(lengths),
        )
        assert program_triple.split("_", 1)[0] == (
            _derived_family(module, lengths) if fused else two_kernel
        ), (
            program_triple,
            fused,
        )
        ring_triple = _without_table_form(
            module, program_triple, state_dtype=state_dtype, num_sequences=len(lengths)
        )
        assert program_cu.split("_", 1)[1] == ring_triple.split("_", 1)[1], (
            program_cu,
            program_triple,
        )
        assert program_cu.split("_", 1)[0] == ("exactfused" if fused else two_kernel), (
            program_cu,
            fused,
        )
    assert final.dtype == state_dtype
    assert tuple(final.shape) == (len(lengths), 8, 64, 128)
    assert torch.isfinite(out.to(torch.float32)).all()
    if same_arithmetic:
        assert torch.equal(out, out_triple), (
            f"cu_seqlens form ({program_cu}) differs from the triple ({program_triple})"
        )
        assert torch.equal(final, final_triple)
    else:
        # Sanity only: the forms differ by the single-rounding state update.
        for name, cu, triple in (
            ("out", out, out_triple),
            ("final_states", final, final_triple),
        ):
            tolerance = {torch.float32: 1e-5}.get(cu.dtype, 1e-2)
            assert torch.allclose(
                cu.to(torch.float32),
                triple.to(torch.float32),
                atol=tolerance,
                rtol=tolerance,
            ), (
                f"{name}: cu_seqlens form ({program_cu}) vs triple ({program_triple}) beyond {tolerance}"
            )
        # The reading that decides: each form passes the P6 clause on its own.
        _assert_two_forms_p6(
            monkeypatch,
            lengths=lengths,
            state_dtype=state_dtype,
            initial_states=initial_states,
            programs=(program_cu, program_triple),
        )
    # seq_idx riding along (callers that still build it) changes nothing.
    out_ride, final_ride = runner.run(
        *tensors, **_cu_seqlens_arguments(arguments, lengths, keep_seq_idx=True)
    )
    assert cake.last_program_name == program_cu, (
        f"seq_idx riding along ran {cake.last_program_name}, the cu_seqlens form {program_cu}"
    )
    assert torch.equal(out_ride, out) and torch.equal(final_ride, final)
    # The triple with the caller's trusted table: the fused program of the
    # cu_seqlens form inside the fused window (one launch, the caller's
    # tables read in-kernel) and bitwise that form; the derived triple's
    # program outside it, bitwise the derived call.
    out_trusted, final_trusted = runner.run(*tensors, **arguments)
    assert cake.seq_idx_status(reset=True) == 0
    program_trusted = cake.last_program_name
    if fused and not program_triple.startswith("chunkpar_"):
        assert program_trusted == program_cu, (
            f"trusted-table triple ran {program_trusted}, cu_seqlens {program_cu}"
        )
        assert torch.equal(out_trusted, out) and torch.equal(final_trusted, final)
    else:
        assert program_trusted == program_triple, (
            f"trusted-table triple ran {program_trusted}, derived {program_triple}"
        )
        assert torch.equal(out_trusted, out_triple)
        assert torch.equal(final_trusted, final_triple)
    # The public functional entry takes the same form (it infers the state
    # dtype from initial_states, BF16 without them).
    if initial_states or state_dtype == torch.bfloat16:
        functional = importlib.import_module("flashinfer.mamba.ssd_combined")
        out_fn, final_fn = functional.ssd_combined_fwd(
            *tensors, **_cu_seqlens_arguments(arguments, lengths)
        )
        assert torch.equal(out_fn, out) and torch.equal(final_fn, final)


@pytest.mark.parametrize(
    "lengths,checkpoints",
    (
        ((96, 160), None),
        ((1,), None),
        ((128, 0, 200), None),
        ((1024, 100), None),
        ((300, 700), [None, 556]),
        ((64, 60), [None, 100]),
        ((96, 160), [96, 224]),
        ((1000,), [128]),
    ),
    ids=(
        "96x160",
        "1",
        "empty-middle",
        "1024x100",
        "checkpoint-556",
        "checkpoint-100",
        "checkpoints-at-end-and-224",
        "checkpoint-on-grid",
    ),
)
def test_cake_ssd_combined_cu_seqlens_publishes_the_derived_metadata(
    lengths, checkpoints
):
    """Read-back of the preprocess-derived tables: ``seq_chunk_cumsum`` (into
    the caller's buffer with ``update_seq_chunk_cumsum=True``), the
    runner-owned ``[bound + 1]`` ``chunk_indices`` / ``chunk_offsets`` and
    the ``-1`` sentinel at ``num_segments`` match the host mirror of the
    derivation and, without checkpoints, the sglang logical-chunk set; a
    chunk-unaligned checkpoint becomes a segment boundary, one on the grid
    or at a sequence end adds nothing."""

    _skip_unless_cake_arch()
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    constructor, tensors, arguments = _case(varlen=True, lengths=lengths)
    num_seqs, total = len(lengths), sum(lengths)
    table = torch.full((num_seqs + 1,), -7, dtype=torch.int32, device="cuda")
    cu_arguments = _cu_seqlens_arguments(arguments, lengths)
    cu_arguments.update(seq_chunk_cumsum=table, update_seq_chunk_cumsum=True)
    if checkpoints is not None:
        slots = [
            index if value is not None else -1
            for index, value in enumerate(checkpoints)
        ]
        cu_arguments.update(
            checkpoint_token_indices=torch.tensor(
                [-1 if value is None else value for value in checkpoints],
                dtype=torch.int32,
                device="cuda",
            ),
            checkpoint_state_slots=torch.tensor(
                slots, dtype=torch.int32, device="cuda"
            ),
            checkpoint_states=torch.full(
                (num_seqs, 8, 64, 128), torch.nan, dtype=torch.bfloat16, device="cuda"
            ),
        )
    runner = SSDCombined(**constructor, backend="cake")
    cake = runner._cake_runner
    cake.seq_idx_status(reset=True)

    out, final = runner.run(*tensors, **cu_arguments)

    assert cake.seq_idx_status(reset=True) == 0
    expected = _host_cu_seqlens_metadata(
        _cu_seqlens(lengths).tolist(), total, checkpoints
    )
    assert not expected["flagged"]
    bound, num_segments = expected["bound"], expected["num_segments"]
    assert (
        bound
        == module._segment_bound(total, num_seqs)
        == -(-total // 128) + 2 * num_seqs
    )
    assert num_segments <= bound - 1  # room for the sentinel on valid input
    chunk_indices = cake._workspace["chunk_indices"]
    chunk_offsets = cake._workspace["chunk_offsets"]
    assert chunk_indices.dtype == chunk_offsets.dtype == torch.int32
    assert chunk_indices.numel() == chunk_offsets.numel() == bound + 1
    assert (
        chunk_indices[:num_segments].tolist()
        == expected["chunk_indices"][:num_segments]
    )
    assert (
        chunk_offsets[:num_segments].tolist()
        == expected["chunk_offsets"][:num_segments]
    )
    assert int(chunk_indices[num_segments]) == -1
    assert table.tolist() == expected["seq_chunk_cumsum"]
    if checkpoints is None:
        assert table.tolist() == _seq_chunk_cumsum(lengths).tolist()
    boundaries = [value for value in (checkpoints or []) if value is not None]
    reference_indices, reference_offsets = _logical_chunk_metadata(lengths, boundaries)
    assert chunk_indices[:num_segments].tolist() == reference_indices.tolist()
    assert chunk_offsets[:num_segments].tolist() == reference_offsets.tolist()
    if checkpoints is not None:
        for sequence, value in enumerate(checkpoints):
            written = cu_arguments["checkpoint_states"][sequence]
            if value is None:
                assert torch.isnan(written).all()
            else:
                assert torch.isfinite(written.to(torch.float32)).all()
    # Exact when the preprocess publishes into the runner-owned buffer too.
    out_again, final_again = runner.run(
        *tensors, **_cu_seqlens_arguments(arguments, lengths)
    )
    if checkpoints is None:
        assert torch.equal(out_again, out) and torch.equal(final_again, final)


@pytest.mark.parametrize("chunk_size", (64, 128, 256, 512))
def test_cake_ssd_combined_chunk_size_is_a_caller_convention(chunk_size):
    """``chunk_size`` selects nothing in the Cake programs (they tile 128
    tokens): the cu_seqlens form gives bitwise the chunk-128 result at every
    positive chunk size, batched calls are unaffected, and the chunk-128
    triple is refused on a runner with another chunk size."""

    _skip_unless_cake_arch()
    lengths = (96, 160, 1000)
    constructor, tensors, arguments = _case(varlen=True, lengths=lengths)
    reference = SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)
    runner = SSDCombined(**{**constructor, "chunk_size": chunk_size}, backend="cake")
    assert runner._cake_runner.chunk_size == chunk_size

    out, final = runner.run(*tensors, **_cu_seqlens_arguments(arguments, lengths))

    assert torch.equal(out, reference[0]) and torch.equal(final, reference[1])
    if chunk_size != 128:
        with pytest.raises(
            ValueError,
            match=f"chunk_size={chunk_size}: the seq_idx / chunk_indices / chunk_offsets",
        ):
            runner.run(*tensors, **arguments)
    batched_constructor, batched_tensors, batched_arguments = _case(varlen=False)
    expected = SSDCombined(**batched_constructor, backend="cake").run(
        *batched_tensors, **batched_arguments
    )
    actual = SSDCombined(
        **{**batched_constructor, "chunk_size": chunk_size}, backend="cake"
    ).run(*batched_tensors, **batched_arguments)
    assert torch.equal(actual[0], expected[0]) and torch.equal(actual[1], expected[1])


@pytest.mark.parametrize(
    "state_dtype", (torch.bfloat16, torch.float32), ids=("bf16", "f32")
)
def test_cake_ssd_combined_cu_seqlens_checkpoint_at_unaligned_sequence_start(
    state_dtype,
):
    """sglang radix-cache tracking at the engine's 256 grid of a sequence that
    starts off the 128 grid: packed [0, 300) + [300, 1000), checkpoint 256
    tokens into sequence 1 (absolute 556, inside physical chunk 4).  With
    cu_seqlens the preprocess inserts the boundary itself; the slot holds
    the fp64 state after 556 tokens within the fixed tolerance, the result
    is bitwise the triple call that exposes the boundary through its
    metadata, and the unused slots stay untouched."""

    _skip_unless_cake_arch()
    lengths = (300, 700)
    boundary = 300 + 256
    constructor, tensors, arguments = _case(
        varlen=True, lengths=lengths, state_dtype=state_dtype
    )
    checkpoint_states = torch.full(
        (3, 8, 64, 128), torch.nan, dtype=state_dtype, device="cuda"
    )
    checkpoint = dict(
        checkpoint_token_indices=torch.tensor(
            [-1, boundary], dtype=torch.int32, device="cuda"
        ),
        checkpoint_state_slots=torch.tensor([-1, 2], dtype=torch.int32, device="cuda"),
    )
    runner = SSDCombined(**constructor, backend="cake")
    runner._cake_runner.seq_idx_status(reset=True)

    out, final = runner.run(
        *tensors,
        **_cu_seqlens_arguments(arguments, lengths),
        **checkpoint,
        checkpoint_states=checkpoint_states,
    )

    assert runner._cake_runner.seq_idx_status(reset=True) == 0
    # Bitwise the triple form with the boundary exposed through the metadata.
    chunk_indices, chunk_offsets = _logical_chunk_metadata(lengths, (boundary,))
    assert chunk_indices.tolist() == [0, 1, 2, 2, 3, 4, 4, 5, 6, 7]
    assert chunk_offsets.tolist() == [0, 0, 0, 44, 0, 0, 44, 0, 0, 0]
    triple_states = torch.full_like(checkpoint_states, torch.nan)
    out_triple, final_triple = runner.run(
        *tensors,
        **{
            **arguments,
            "chunk_indices": chunk_indices,
            "chunk_offsets": chunk_offsets,
            "seq_chunk_cumsum": None,
        },
        **checkpoint,
        checkpoint_states=triple_states,
    )
    assert torch.equal(out, out_triple) and torch.equal(final, final_triple)
    assert torch.equal(checkpoint_states[2], triple_states[2])
    assert torch.isnan(checkpoint_states[:2]).all()
    # The whole result against CuTe / fp64 (the exposed boundary changes
    # nothing); CuTe has no f32 state, so that row goes to the recurrence alone.
    if state_dtype == torch.float32:
        reference_out, reference_final = _fp64_reference(
            constructor, tensors, arguments
        )
        for label, value, expected_value in (
            ("out", out, reference_out),
            ("final_states", final, reference_final),
        ):
            outside = int(_outside_tolerance(value, expected_value).sum())
            assert outside <= _MAX_OUTSIDE_FRACTION * expected_value.numel(), label
    else:
        expected = _cute_padded_reference(constructor, tensors, arguments, lengths)
        _assert_cake_accuracy((out, final), expected, constructor, tensors, arguments)
    # The checkpoint against the fp64 recurrence over sequence 1's first 256
    # tokens from its initial state (a batched single-sequence problem).
    x, dt, A, B, C = tensors
    window = slice(300, boundary)
    prefix_tensors = (
        x[:, window].contiguous(),
        dt[:, window].contiguous(),
        A,
        B[:, window].contiguous(),
        C[:, window].contiguous(),
    )
    prefix_arguments = {
        **arguments,
        "z": arguments["z"][:, window].contiguous(),
        "initial_states": arguments["initial_states"][1:2].contiguous(),
        "seq_idx": None,
        "chunk_indices": None,
        "chunk_offsets": None,
        "seq_chunk_cumsum": None,
    }
    _, oracle = _fp64_reference(
        {**constructor, "has_varlen": False}, prefix_tensors, prefix_arguments
    )
    written = checkpoint_states[2]
    assert torch.isfinite(written.to(torch.float32)).all()
    outside = int(_outside_tolerance(written, oracle[0]).sum())
    assert outside <= _MAX_OUTSIDE_FRACTION * oracle[0].numel(), (
        f"checkpoint: {outside} of {oracle[0].numel()} entries outside "
        f"atol=rtol={_ATOL} of the fp64 state after 556 tokens"
    )


@pytest.mark.parametrize(
    "case",
    ("cu0", "decreasing", "total", "too_many"),
)
def test_cake_ssd_combined_flags_invalid_cu_seqlens(case):
    """CAKE-990 rules on the cu_seqlens form: ``cu[0] != 0``, a decreasing
    entry, ``cu[-1] != total`` and more segments than the bound set the
    status word; every write stays inside the ``[bound + 1]`` tables (the
    read-back matches the host mirror, the sentinel sits at the clamped
    count); the sequences whose derived ranges are not touched by the
    damage keep their correct results.  The main kernel ends a segment at
    the next segment of the same physical chunk, else at the chunk end
    (segments partition the stream): a sequence whose range another
    sequence starts inside is cut there, and the last segment runs over
    trailing unclaimed tokens without a preprocess delta (rows and last
    state undefined) -- both documented in D12, both flagged.  A clean call
    afterwards leaves the reset word at 0."""

    _skip_unless_cake_arch()
    f32 = torch.float32
    if case == "cu0":
        # Tokens [0, 10) belong to no sequence; [10, 100) and [100, 250) do.
        cu, total = [10, 100, 250], 250
        constructor, tensors, arguments = _case(varlen=True, lengths=(90, 160))
    elif case == "decreasing":
        # [0, 100), an empty id (100 > 60 clamps to nothing), [60, 100) (starts
        # inside sequence 0, which the main kernel therefore cuts at 60) and
        # a clean [100, 250).
        cu, total = [0, 100, 60, 100, 250], 250
        constructor, tensors, arguments = _case(varlen=True, lengths=(60, 40, 50, 100))
    elif case == "total":
        # Tokens [200, 250) belong to no sequence: the last sequence absorbs them.
        cu, total = [0, 100, 200], 250
        constructor, tensors, arguments = _case(varlen=True, lengths=(100, 150))
    else:
        # 8 + 0 + 8 segments against a bound of 8 + 2 * 3 = 14: the last two
        # segments of sequence 2 are dropped (its state is not compared); both
        # full sequences start from zero, so their shared rows agree.
        cu, total = [0, 1024, 0, 1024], 1024
        constructor, tensors, arguments = _case(
            varlen=True, lengths=(512, 256, 256), initial_states=False
        )
    num_seqs = len(cu) - 1
    x, dt, A, B, C = tensors
    assert x.shape[1] == total
    table = torch.full((num_seqs + 1,), -7, dtype=torch.int32, device="cuda")
    flagged_arguments = {
        **_cu_seqlens_arguments(arguments, [0] * num_seqs),
        "cu_seqlens": torch.tensor(cu, dtype=torch.int32, device="cuda"),
        "seq_chunk_cumsum": table,
        "update_seq_chunk_cumsum": True,
    }
    runner = SSDCombined(**constructor, backend="cake")
    cake = runner._cake_runner
    cake.seq_idx_status(reset=True)

    out, final = runner.run(*tensors, **flagged_arguments)

    assert cake.seq_idx_status(reset=True) == 1
    expected = _host_cu_seqlens_metadata(cu, total)
    assert expected["flagged"]
    num_segments, bound = expected["num_segments"], expected["bound"]
    chunk_indices = cake._workspace["chunk_indices"]
    chunk_offsets = cake._workspace["chunk_offsets"]
    assert chunk_indices.numel() == chunk_offsets.numel() == bound + 1
    assert (
        chunk_indices[:num_segments].tolist()
        == expected["chunk_indices"][:num_segments]
    )
    assert (
        chunk_offsets[:num_segments].tolist()
        == expected["chunk_offsets"][:num_segments]
    )
    assert int(chunk_indices[num_segments]) == -1
    assert table.tolist() == expected["seq_chunk_cumsum"]
    assert tuple(final.shape) == (num_seqs, 8, 64, 128)

    def clean_run(window, lengths, states):
        """The cu_seqlens call on the token window alone; returns its result
        and the fp64 recurrence over the window (``seq_idx`` names the
        sequences for the oracle only)."""
        clean_tensors = tuple(
            value[:, window].contiguous()
            if value.ndim >= 2 and value.shape[1] == total
            else value
            for value in tensors
        )
        clean_arguments = {
            **_cu_seqlens_arguments(arguments, lengths),
            "z": arguments["z"][:, window].contiguous(),
            "initial_states": states,
        }
        clean = SSDCombined(**constructor, backend="cake")
        result = clean.run(*clean_tensors, **clean_arguments)
        assert clean._cake_runner.seq_idx_status(reset=True) == 0
        oracle_arguments = {
            **clean_arguments,
            "seq_idx": _varlen_metadata(lengths, torch.int32)[0],
            "num_seqs": len(lengths),
        }
        return result, _fp64_reference(constructor, clean_tensors, oracle_arguments)

    def within_oracle(name, actual, oracle):
        for label, value, expected_value in (
            (f"{name}.out", actual[0], oracle[0]),
            (f"{name}.final", actual[1], oracle[1]),
        ):
            assert tuple(value.shape) == tuple(expected_value.shape), label
            assert torch.isfinite(value.to(f32)).all(), label
            outside = int(_outside_tolerance(value, expected_value).sum())
            assert outside <= _MAX_OUTSIDE_FRACTION * expected_value.numel(), (
                f"{label}: {outside} of {expected_value.numel()} entries outside "
                f"atol=rtol={_ATOL} of the fp64 recurrence"
            )

    if case == "cu0":
        # The chunk grid of the flagged call is the absolute one (a clean
        # call on the window tiles differently), so the comparison is the
        # fp64 recurrence over the claimed tokens.
        _, oracle = clean_run(slice(10, 250), (90, 150), arguments["initial_states"])
        within_oracle("cu0", (out[:, 10:], final), oracle)
        valid_lengths = (90, 160)
    elif case == "decreasing":
        # Sequence 3 ([100, 250)) is untouched: bitwise the clean problem with
        # the same absolute chunk alignment; sequence 0's rows before the
        # intruding start are its own; the empty id passes its state through.
        states = arguments["initial_states"]
        (clean_out, clean_final), _ = clean_run(
            slice(0, 250), (100, 150), states[[0, 3]].contiguous()
        )
        assert torch.equal(out[:, :60], clean_out[:, :60])
        assert torch.equal(out[:, 100:], clean_out[:, 100:])
        assert torch.equal(final[3], clean_final[1])
        assert torch.equal(final[1], states[1])
        assert torch.isfinite(out.to(f32)).all() and torch.isfinite(final.to(f32)).all()
        valid_lengths = (60, 40, 50, 100)
    elif case == "total":
        # Sequences 0 and 1 are bitwise the clean problem over the claimed
        # tokens on their rows; the main kernel also runs the last segment
        # over the unclaimed tail [200, 250) (segments partition the stream)
        # for which the preprocess computed no delta, so the tail rows and
        # the last state are undefined (flagged, memory-safe) and not compared.
        (clean_out, clean_final), _ = clean_run(
            slice(0, 200), (100, 100), arguments["initial_states"]
        )
        assert torch.equal(out[:, :200], clean_out)
        assert torch.equal(final[0], clean_final[0])
        valid_lengths = (100, 150)
    else:
        (clean_out, clean_final), _ = clean_run(slice(0, 1024), (1024,), None)
        assert torch.equal(out, clean_out)
        assert torch.equal(final[0], clean_final[0])
        assert not final[1].to(f32).any()
        valid_lengths = (512, 256, 256)
    # A clean call afterwards leaves the reset word at 0.
    runner.run(*tensors, **_cu_seqlens_arguments(arguments, valid_lengths))
    assert cake.seq_idx_status(reset=True) == 0


def _realistic_decay_inputs(lengths, seed, *, nheads=128, ngroups=8, varlen=True):
    """CAKE-950's repro distribution: bf16 ``dt ~ N(-2, 0.5)`` before
    softplus, ``dt_bias = 0.5``, ``A = -exp(N(0, 0.5))``, D = 1, no z."""

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    total = sum(lengths)
    batch, seqlen = (1, total) if varlen else (len(lengths), lengths[0])

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=generator)

    x = randn(batch, seqlen, nheads, 64).to(torch.bfloat16)
    dt = (randn(batch, seqlen, nheads) * 0.5 - 2.0).to(torch.bfloat16)
    A = -torch.exp(randn(nheads) * 0.5)
    # Pin the last heads to the Mamba2 ``A_log`` init extremes (|A| up to 16)
    # so every seed reaches the per-chunk overflow band of CAKE-950.
    A[-4:] = torch.tensor([-4.0, -8.0, -12.0, -16.0], device="cuda")
    B = randn(batch, seqlen, ngroups, 128).to(torch.bfloat16)
    C = randn(batch, seqlen, ngroups, 128).to(torch.bfloat16)
    D = torch.ones(nheads, device="cuda", dtype=torch.bfloat16)
    dt_bias = torch.full((nheads,), 0.5, device="cuda", dtype=torch.bfloat16)
    initial_states = torch.zeros(
        len(lengths), nheads, 64, 128, device="cuda", dtype=torch.bfloat16
    )
    constructor = dict(
        chunk_size=128,
        nheads=nheads,
        headdim=64,
        dstate=128,
        ngroups=ngroups,
        io_dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        has_d=True,
        d_has_hdim=False,
        has_initial_states=True,
        has_varlen=varlen,
        has_z=False,
        seq_idx_dtype=torch.int32,
    )
    arguments = dict(
        D=D,
        z=None,
        dt_bias=dt_bias,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        initial_states=initial_states,
        return_final_states=True,
    )
    if varlen:
        seq_idx, chunk_indices, chunk_offsets = _varlen_metadata(lengths, torch.int32)
        arguments.update(
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            seq_chunk_cumsum=_seq_chunk_cumsum(lengths),
        )
    return constructor, (x, dt, A, B, C), arguments


@pytest.mark.parametrize("seed", (0, 1, 2))
@pytest.mark.parametrize(
    "varlen,lengths",
    [(True, (128,)), (True, (128, 128)), (True, (128,) * 8), (False, (128, 128))],
    ids=("varlen_1x128", "varlen_2x128", "varlen_8x128", "batched_2x128"),
)
def test_cake_ssd_combined_single_chunk_realistic_decay_has_no_nan(
    varlen, lengths, seed
):
    """CAKE-950 regression: every sequence is one 128-token chunk on the
    Nemotron-H head geometry with realistic decay (heads whose per-chunk
    ``A * dt`` cumsum passes ``-126 / log2(e)``).  The retired prefix route
    produced ``0 * inf`` NaNs here; the output and states must be finite and
    at least as accurate as CuTe's."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake SSDCombined requires SM100, SM103 or SM107")

    constructor, tensors, arguments = _realistic_decay_inputs(
        lengths, seed, varlen=varlen
    )
    dt_processed = torch.nn.functional.softplus(
        tensors[1].float() + arguments["dt_bias"].float()
    )
    batch, seqlen, nheads = dt_processed.shape
    chunk_log2_decay = (dt_processed * tensors[2]).reshape(
        batch, seqlen // 128, 128, nheads
    ).sum(2) * 1.4426950408889634
    assert (chunk_log2_decay < -126.0).any(), "inputs must reach the overflow band"

    expected = SSDCombined(**constructor, backend="cute").run(*tensors, **arguments)
    actual = SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)

    assert torch.isfinite(actual[0].float()).all()
    assert torch.isfinite(actual[1].float()).all()
    _assert_cake_accuracy(actual, expected, constructor, tensors, arguments)


def test_cake_ssd_combined_nemotron_accuracy_vs_recurrent_reference():
    """CAKE-942: with FP16 ``delta`` the fraction of outputs outside
    atol = rtol = 1e-2 of the fp64 recurrent reference on the Nemotron-H
    geometry (T = 1024, realistic decay, zero initial state) is about 0.66 %
    (bf16 delta: 1.56 %, stock Triton: 0.62 %)."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake SSDCombined requires SM100, SM103 or SM107")

    constructor, tensors, arguments = _realistic_decay_inputs((1024,), 11)
    reference_out, reference_states = _fp64_reference(constructor, tensors, arguments)

    out, final_states = SSDCombined(**constructor, backend="cake").run(
        *tensors, **arguments
    )

    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(final_states.float()).all()
    assert torch.isfinite(reference_states).all()
    outside = (out.float() - reference_out).abs() > 1e-2 + 1e-2 * reference_out.abs()
    fraction_outside = outside.float().mean().item()
    assert fraction_outside <= 0.010, f"{fraction_outside:.4%} of outputs outside 1e-2"


@pytest.mark.parametrize("state_dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("dt_softplus", (False, True))
def test_cake_ssd_combined_exact_scan_softplus_parity(state_dtype, dt_softplus):
    """``dt_softplus`` on and off with bf16 and f16 states.  With softplus off
    the ``_case`` step sizes clamp to ``dt_min = 1e-3`` almost everywhere, so
    the state barely evolves and the rounding of the f16 initial state to the
    bf16 ``C . state`` MMA operand (done by both kernels) dominates: 1.10 % of
    the outputs are outside 1e-2 of the fp64 recurrence for Cake and 1.09 %
    for CuTe (GB300), hence the 2 % cap for that row; the bf16-state rows and
    the softplus-on rows stay under the 1 % default."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake SSDCombined requires SM100, SM103 or SM107")

    constructor, tensors, arguments = _case(state_dtype=state_dtype)
    arguments["dt_softplus"] = dt_softplus

    expected = SSDCombined(**constructor, backend="cute").run(*tensors, **arguments)
    actual = SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)
    both_above_default = state_dtype == torch.float16 and not dt_softplus
    _assert_cake_accuracy(
        actual,
        expected,
        constructor,
        tensors,
        arguments,
        max_outside_fraction=0.02 if both_above_default else _MAX_OUTSIDE_FRACTION,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("varlen", (False, True), ids=("batched", "varlen_metadata"))
def test_cake_ssd_combined_program_cache_is_multi_device_safe(varlen):
    if any(
        torch.cuda.get_device_capability(index) not in ((10, 0), (10, 3), (10, 7))
        for index in (0, 1)
    ):
        pytest.skip("Cake SSDCombined requires SM100, SM103 or SM107")

    runners = []
    cases = []
    expected = []
    for device_index in (0, 1):
        with torch.cuda.device(device_index):
            constructor, tensors, arguments = _case(
                nheads=1,
                ngroups=1,
                varlen=varlen,
            )
            expected.append(
                SSDCombined(**constructor, backend="cute").run(*tensors, **arguments)
            )
            runners.append(SSDCombined(**constructor, backend="cake"))
            cases.append((constructor, tensors, arguments))

    torch.cuda.set_device(0)
    for device_index in (0, 1, 0, 1):
        assert torch.cuda.current_device() == 0
        constructor, tensors, arguments = cases[device_index]
        actual = runners[device_index].run(*tensors, **arguments)
        assert actual[0].device.index == device_index
        _assert_cake_accuracy(
            actual, expected[device_index], constructor, tensors, arguments
        )
        assert torch.cuda.current_device() == 0


def test_cake_ssd_combined_public_seq_chunk_cumsum_helpers():
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake SSDCombined requires SM100, SM103 or SM107")

    constructor, _, arguments = _case(varlen=True)
    runner = SSDCombined(**constructor, backend="cake")
    seq_idx = arguments["seq_idx"]
    chunk_indices = arguments["chunk_indices"]
    chunk_offsets = arguments["chunk_offsets"]
    expected = arguments["seq_chunk_cumsum"]
    actual = torch.full_like(expected, -1)
    tile_state_bytes = runner.tile_state_size(2)
    tile_state = (
        torch.empty(tile_state_bytes, dtype=torch.uint8, device="cuda")
        if tile_state_bytes
        else None
    )

    returned = runner.compute_seq_chunk_cumsum(
        seq_idx,
        chunk_indices,
        chunk_offsets,
        128,
        2,
        seq_chunk_cumsum=actual,
        tile_state=tile_state,
    )

    assert returned is actual
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    allocated = runner.compute_seq_chunk_cumsum(
        seq_idx,
        chunk_indices,
        chunk_offsets,
        128,
        2,
        seq_chunk_cumsum=None,
        tile_state=None,
    )
    torch.testing.assert_close(allocated, expected, rtol=0, atol=0)

    num_seqs = 2048
    multiblock_seq_idx = (
        torch.arange(num_seqs, dtype=torch.int32, device="cuda")
        .repeat_interleave(128)
        .unsqueeze(0)
    )
    multiblock_chunk_indices = torch.arange(num_seqs, dtype=torch.int32, device="cuda")
    multiblock_chunk_offsets = torch.zeros(num_seqs, dtype=torch.int32, device="cuda")
    multiblock_expected = torch.arange(num_seqs + 1, dtype=torch.int32, device="cuda")
    multiblock_actual = torch.full_like(multiblock_expected, -1)
    multiblock_tile_state_bytes = runner.tile_state_size(num_seqs)
    assert multiblock_tile_state_bytes > 0
    multiblock_tile_state = torch.empty(
        multiblock_tile_state_bytes, dtype=torch.uint8, device="cuda"
    )

    multiblock_returned = runner.compute_seq_chunk_cumsum(
        multiblock_seq_idx,
        multiblock_chunk_indices,
        multiblock_chunk_offsets,
        128,
        num_seqs,
        seq_chunk_cumsum=multiblock_actual,
        tile_state=multiblock_tile_state,
    )

    assert multiblock_returned is multiblock_actual
    torch.testing.assert_close(multiblock_actual, multiblock_expected, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["a_dtype", "out_shape"])
def test_cake_ssd_combined_rejects_invalid_public_inputs_like_cute(invalid):
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake SSDCombined requires SM100, SM103 or SM107")

    constructor, tensors, arguments = _case()
    if invalid == "a_dtype":
        tensors = (*tensors[:2], tensors[2].to(torch.bfloat16), *tensors[3:])
    else:
        arguments = {
            **arguments,
            "out": torch.empty((1,), dtype=torch.bfloat16, device="cuda"),
        }

    errors = {}
    for backend in ("cute", "cake"):
        runner = SSDCombined(**constructor, backend=backend)
        with pytest.raises(AssertionError) as exc_info:
            runner.run(*tensors, **arguments)
        errors[backend] = (type(exc_info.value), str(exc_info.value))

    if invalid == "a_dtype":
        assert errors["cake"] == errors["cute"]
    else:
        # Same exception and message form; each backend names its own kernel
        # output layout (CuTe chunked, Cake token-major).
        assert "out shape torch.Size([1]) doesn't match expected" in errors["cute"][1]
        assert "out shape torch.Size([1]) doesn't match expected" in errors["cake"][1]
        assert errors["cute"][1].endswith("(2, 8, 64, 1, 128)")
        assert errors["cake"][1].endswith("(2, 128, 8, 64)")


def test_cake_ssd_combined_checks_alignment_on_the_bound_tensors():
    """The 16-byte ``z`` / ``D`` alignment the scan epilogue needs (CAKE-991)
    is checked on the tensors the kernel reads: a contiguous view at an 8-byte
    offset is rejected, while a strided view with the same odd base is repacked
    into an aligned workspace copy and runs (bitwise the contiguous result)."""
    _skip_unless_cake_arch()
    constructor, tensors, arguments = _case()
    runner = SSDCombined(**constructor, backend="cake")
    reference = runner.run(*tensors, **arguments)
    z = arguments["z"]
    buffer = torch.empty(z.numel() + 4, dtype=z.dtype, device=z.device)
    misaligned = buffer[4:].view(z.shape).copy_(z)
    assert misaligned.is_contiguous() and misaligned.data_ptr() % 16 == 8
    with pytest.raises(ValueError, match="z must be 16-byte aligned"):
        runner.run(*tensors, **{**arguments, "z": misaligned})
    wide = torch.empty((*z.shape[:-1], z.shape[-1] + 8), dtype=z.dtype, device=z.device)
    strided = wide[..., 4 : 4 + z.shape[-1]].copy_(z)
    assert not strided.is_contiguous() and strided.data_ptr() % 16 == 8
    repacked = runner.run(*tensors, **{**arguments, "z": strided})
    for actual, expected in zip(repacked, reference, strict=False):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_ssd_combined_fwd_caches_by_device_stream_and_config(monkeypatch, request):
    module = importlib.import_module("flashinfer.mamba.ssd_combined")
    runners = []
    active_stream = {"handle": 0x1000}

    class Runner:
        def __init__(self, *args, **kwargs):
            self.constructor_args = args
            self.constructor_kwargs = kwargs
            self.run_calls = []
            runners.append(self)

        def run(self, *args, **kwargs):
            self.run_calls.append((args, kwargs))
            return (self, None)

    module._get_ssd_combined_runner.cache_clear()
    request.addfinalizer(module._get_ssd_combined_runner.cache_clear)
    monkeypatch.setattr(module, "SSDCombined", Runner)
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda *_: SimpleNamespace(cuda_stream=active_stream["handle"]),
    )
    x = SimpleNamespace(
        shape=(1, 128, 8, 64),
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
    )
    B = SimpleNamespace(shape=(1, 128, 8, 128))
    checkpoint_states = SimpleNamespace(dtype=torch.float16)
    D = SimpleNamespace(ndim=2)
    initial_states = SimpleNamespace(dtype=torch.float16)
    seq_idx = SimpleNamespace(dtype=torch.int32)
    optional = {
        "D": D,
        "z": object(),
        "dt_bias": object(),
        "dt_softplus": True,
        "dt_limit": (-0.5, 0.75),
        "initial_states": initial_states,
        "seq_idx": seq_idx,
        "chunk_indices": object(),
        "chunk_offsets": object(),
        "seq_chunk_cumsum": object(),
        "update_seq_chunk_cumsum": True,
        "checkpoint_token_indices": object(),
        "checkpoint_state_slots": object(),
        "checkpoint_states": checkpoint_states,
        "out": object(),
        "return_final_states": False,
        "num_seqs": 3,
    }
    positional = (x, object(), object(), B, object())

    first = module.ssd_combined_fwd(
        *positional,
        **optional,
    )
    repeated = module.ssd_combined_fwd(*positional, **optional)

    active_stream["handle"] = 0x2000
    different_stream = module.ssd_combined_fwd(*positional, **optional)

    active_stream["handle"] = 0x1000
    x_device_one = SimpleNamespace(
        shape=x.shape,
        dtype=x.dtype,
        device=torch.device("cuda:1"),
    )
    different_device = module.ssd_combined_fwd(
        x_device_one,
        *positional[1:],
        **optional,
    )
    second = module.ssd_combined_fwd(
        x,
        object(),
        object(),
        B,
        object(),
        checkpoint_states=checkpoint_states,
    )

    assert isinstance(first, tuple) and isinstance(second, tuple)
    assert first[0] is repeated[0]
    assert first[0] is not different_stream[0]
    assert first[0] is not different_device[0]
    assert first[0] is not second[0]
    assert len(runners) == 4
    assert all(runner.constructor_kwargs["backend"] == "cake" for runner in runners)
    assert all(
        runner.constructor_kwargs["state_dtype"] == torch.float16 for runner in runners
    )
    assert runners[0].constructor_args == (128, 8, 64, 128, 8)
    assert runners[0].constructor_kwargs == {
        "io_dtype": torch.bfloat16,
        "state_dtype": torch.float16,
        "has_d": True,
        "d_has_hdim": True,
        "has_initial_states": True,
        "has_varlen": True,
        "has_z": True,
        "seq_idx_dtype": torch.int32,
        "backend": "cake",
    }
    forwarded = {**optional, "cu_seqlens": None}
    assert runners[0].run_calls == [(positional, forwarded), (positional, forwarded)]
    assert runners[-1].run_calls[0][1]["dt_softplus"] is False


def _signature_contract(callable_, *, drop_self=False):
    parameters = tuple(inspect.signature(callable_).parameters.values())
    if drop_self:
        assert parameters[0].name == "self"
        parameters = parameters[1:]
    return tuple(
        (parameter.name, parameter.kind, parameter.default) for parameter in parameters
    )


def test_source_public_api_signatures_are_stable():
    module = importlib.import_module("flashinfer.mamba.ssd_combined")
    cake_module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
    empty = inspect.Parameter.empty

    constructor_names = (
        "chunk_size",
        "nheads",
        "headdim",
        "dstate",
        "ngroups",
        "io_dtype",
        "state_dtype",
        "has_d",
        "d_has_hdim",
        "has_initial_states",
        "has_varlen",
        "has_z",
        "seq_idx_dtype",
        "backend",
    )
    constructor_defaults = (
        empty,
        empty,
        empty,
        empty,
        empty,
        torch.bfloat16,
        torch.bfloat16,
        True,
        False,
        False,
        False,
        False,
        torch.int64,
        "cute",
    )
    assert _signature_contract(module.SSDCombined) == tuple(
        zip(
            constructor_names,
            (positional,) * len(constructor_names),
            constructor_defaults,
            strict=True,
        )
    )

    run_names = (
        "x",
        "dt",
        "A",
        "B",
        "C",
        "D",
        "z",
        "dt_bias",
        "dt_softplus",
        "dt_limit",
        "initial_states",
        "seq_idx",
        "chunk_indices",
        "chunk_offsets",
        "seq_chunk_cumsum",
        "update_seq_chunk_cumsum",
        "checkpoint_token_indices",
        "checkpoint_state_slots",
        "checkpoint_states",
        "out",
        "return_final_states",
        "num_seqs",
        "cu_seqlens",
    )
    run_defaults = (
        empty,
        empty,
        empty,
        empty,
        empty,
        None,
        None,
        None,
        False,
        (0.0, float("inf")),
        None,
        None,
        None,
        None,
        None,
        False,
        None,
        None,
        None,
        None,
        True,
        None,
        None,
    )
    expected_run = tuple(
        zip(
            run_names,
            (positional,) * len(run_names),
            run_defaults,
            strict=True,
        )
    )
    assert _signature_contract(module.SSDCombined.run, drop_self=True) == expected_run
    assert (
        _signature_contract(cake_module.CakeSSDCombined.run, drop_self=True)
        == expected_run
    )
    assert _signature_contract(module.ssd_combined_fwd) == expected_run

    # The prepared (CUDA-graph-capturable) submission entry: keyword-only,
    # the shape-determining subset of ``run``'s arguments, identical on the
    # public runner and the Cake runner it delegates to.
    prepare_names = (
        "x",
        "dt",
        "A",
        "B",
        "C",
        "D",
        "z",
        "dt_bias",
        "dt_softplus",
        "dt_limit",
        "initial_states",
        "seq_idx",
        "chunk_indices",
        "chunk_offsets",
        "seq_chunk_cumsum",
        "return_final_states",
        "num_seqs",
        "cu_seqlens",
        "out",
    )
    prepare_defaults = (
        empty,
        empty,
        empty,
        empty,
        empty,
        None,
        None,
        None,
        False,
        (0.0, float("inf")),
        None,
        None,
        None,
        None,
        None,
        True,
        None,
        None,
        None,
    )
    expected_prepare = tuple(
        zip(
            prepare_names,
            (inspect.Parameter.KEYWORD_ONLY,) * len(prepare_names),
            prepare_defaults,
            strict=True,
        )
    )
    assert (
        _signature_contract(module.SSDCombined.prepare, drop_self=True)
        == expected_prepare
    )
    assert (
        _signature_contract(cake_module.CakeSSDCombined.prepare, drop_self=True)
        == expected_prepare
    )

    helper_names = (
        "seq_idx",
        "chunk_indices",
        "chunk_offsets",
        "chunk_size",
        "num_seqs",
        "seq_chunk_cumsum",
        "tile_state",
    )
    helper_defaults = (empty, empty, empty, empty, empty, None, None)
    assert _signature_contract(
        module.SSDCombined.compute_seq_chunk_cumsum, drop_self=True
    ) == tuple(
        zip(
            helper_names,
            (positional,) * len(helper_names),
            helper_defaults,
            strict=True,
        )
    )
    assert tuple(inspect.signature(module.SSDCombined.tile_state_size).parameters) == (
        "num_seqs",
    )


def test_source_public_constructor_forwards_complete_cake_contract(monkeypatch):
    module = importlib.import_module("flashinfer.mamba.ssd_combined")
    cake_module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    utils = importlib.import_module("flashinfer.utils")
    calls = []

    class CakeRunner:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(utils, "get_compute_capability", lambda *_: (10, 3))
    monkeypatch.setattr(cake_module, "CakeSSDCombined", CakeRunner)
    runner = module.SSDCombined(
        128,
        128,
        64,
        128,
        8,
        io_dtype=torch.bfloat16,
        state_dtype=torch.float16,
        has_d=False,
        d_has_hdim=True,
        has_initial_states=True,
        has_varlen=True,
        has_z=True,
        seq_idx_dtype=torch.int32,
        backend="cake",
    )

    assert calls == [
        (
            (128, 128, 64, 128, 8),
            {
                "io_dtype": torch.bfloat16,
                "state_dtype": torch.float16,
                "has_d": False,
                "d_has_hdim": True,
                "has_initial_states": True,
                "has_varlen": True,
                "has_z": True,
                "seq_idx_dtype": torch.int32,
            },
        )
    ]
    assert runner._backend == "cake"
    assert runner._cake_runner.__class__ is CakeRunner

    with pytest.raises(ValueError, match="backend must be 'cute' or 'cake'"):
        module.SSDCombined(128, 8, 64, 128, 8, backend="unknown")


def test_source_public_constructor_hardware_error_parity_without_gpu(monkeypatch):
    module = importlib.import_module("flashinfer.mamba.ssd_combined")
    utils = importlib.import_module("flashinfer.utils")
    monkeypatch.setattr(utils, "get_compute_capability", lambda *_: (12, 0))
    errors = {}

    for backend in ("cute", "cake"):
        with pytest.raises(ValueError) as exc_info:
            module.SSDCombined(128, 2, 64, 128, 1, backend=backend)
        errors[backend] = (type(exc_info.value), str(exc_info.value))

    assert errors["cake"] == errors["cute"]


def test_source_public_cake_constructor_rejects_non_exported_arch_without_gpu(
    monkeypatch,
):
    module = importlib.import_module("flashinfer.mamba.ssd_combined")
    utils = importlib.import_module("flashinfer.utils")
    monkeypatch.setattr(utils, "get_compute_capability", lambda *_: (11, 0))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (11, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    cake_module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    cake_module._target_arch.cache_clear()
    try:
        with pytest.raises(
            ValueError, match="requires SM100, SM103 or SM107, got SM110"
        ):
            module.SSDCombined(128, 2, 64, 128, 1, backend="cake")
    finally:
        cake_module._target_arch.cache_clear()


@pytest.mark.parametrize(
    "backend,invalid,exception,match",
    (
        ("cute", "io_dtype", AssertionError, "io_dtype must be bfloat16"),
        ("cute", "state_dtype", AssertionError, "state_dtype must be one of"),
        ("cake", "chunk_size", ValueError, "chunk_size must be a positive int"),
        ("cake", "headdim", ValueError, "requires headdim=64 and dstate=128"),
        ("cake", "dstate", ValueError, "requires headdim=64 and dstate=128"),
        ("cake", "nheads", ValueError, "requires positive nheads"),
        ("cake", "ngroups", ValueError, "requires positive nheads"),
        ("cake", "head_group_ratio", ValueError, "requires positive nheads"),
        ("cake", "io_dtype", ValueError, "requires bfloat16 IO"),
        ("cake", "state_dtype", ValueError, "state dtype must be"),
        ("cake", "seq_idx_dtype", ValueError, "seq_idx dtype must be"),
    ),
)
def test_source_public_backend_constructor_validation_without_gpu(
    monkeypatch, backend, invalid, exception, match
):
    module = importlib.import_module("flashinfer.mamba.ssd_combined")
    utils = importlib.import_module("flashinfer.utils")
    monkeypatch.setattr(utils, "get_compute_capability", lambda *_: (10, 3))
    constructor = {
        "chunk_size": 128,
        "nheads": 2,
        "headdim": 64,
        "dstate": 128,
        "ngroups": 1,
        "io_dtype": torch.bfloat16,
        "state_dtype": torch.bfloat16,
        "seq_idx_dtype": torch.int32,
        "backend": backend,
    }
    replacements = {
        # Any positive chunk size is a valid caller convention (CAKE-934 item 2).
        "chunk_size": 0,
        "headdim": 32,
        "dstate": 64,
        "nheads": 0,
        "ngroups": 0,
        "io_dtype": torch.float16,
        # float32 states are a Cake-only feature; float64 is invalid for both.
        "state_dtype": torch.float64,
        "seq_idx_dtype": torch.float32,
    }
    if invalid == "head_group_ratio":
        constructor.update(nheads=3, ngroups=2)
    else:
        constructor[invalid] = replacements[invalid]

    with pytest.raises(exception, match=match):
        module.SSDCombined(**constructor)


def _public_runner_without_constructor(backend, cake_result=None):
    runner = object.__new__(SSDCombined)
    runner.chunk_size = 128
    runner._backend = backend
    runner._io_torch_dtype = torch.bfloat16
    runner._state_torch_dtype = torch.bfloat16
    runner._has_d = False
    runner._has_init_states = False

    class CakeRunner:
        def __init__(self):
            self.calls = []

        def run(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return cake_result

    runner._cake_runner = CakeRunner()
    return runner


def _cpu_public_run_inputs(batch=1):
    x = torch.empty((batch, 128, 2, 64), dtype=torch.bfloat16)
    dt = torch.empty((batch, 128, 2), dtype=torch.float32)
    A = torch.empty((2,), dtype=torch.float32)
    B = torch.empty((batch, 128, 1, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)
    return x, dt, A, B, C


def test_source_public_cake_dispatch_preserves_full_run_contract():
    result = (object(), None)
    runner = _public_runner_without_constructor("cake", result)
    tensors = _cpu_public_run_inputs()
    sentinels = {
        "D": torch.empty(2, dtype=torch.bfloat16),
        "z": torch.empty((1, 128, 2, 64), dtype=torch.bfloat16),
        "dt_bias": object(),
        "initial_states": torch.empty((1, 2, 64, 128), dtype=torch.bfloat16),
        "seq_idx": torch.zeros((1, 128), dtype=torch.int32),
        "chunk_indices": torch.zeros(1, dtype=torch.int32),
        "chunk_offsets": torch.zeros(1, dtype=torch.int32),
        "seq_chunk_cumsum": object(),
        "checkpoint_token_indices": object(),
        "checkpoint_state_slots": object(),
        "checkpoint_states": object(),
    }
    out = torch.empty((1, 128, 2, 64), dtype=torch.bfloat16)
    kwargs = {
        **sentinels,
        "dt_softplus": True,
        "dt_limit": (-0.25, 0.75),
        "update_seq_chunk_cumsum": True,
        "out": out,
        "return_final_states": False,
        "num_seqs": 1,
        "cu_seqlens": torch.tensor([0, 128], dtype=torch.int32),
    }

    actual = runner.run(*tensors, **kwargs)

    assert actual is result
    assert runner._cake_runner.calls == [(tensors, kwargs)]


@pytest.mark.parametrize(
    "relaxed", ("unaligned_seqlen", "varlen_without_initial_states")
)
def test_source_public_cake_dispatch_relaxed_domain_without_gpu(relaxed):
    """The Cake-only relaxations pass the public pre-dispatch validation."""

    result = (object(), None)
    runner = _public_runner_without_constructor("cake", result)
    x, dt, A, B, C = _cpu_public_run_inputs()
    kwargs = {}
    if relaxed == "unaligned_seqlen":
        tensors = (x[:, :100], dt[:, :100], A, B[:, :100], C[:, :100])
    else:
        tensors = (x, dt, A, B, C)
        kwargs = {
            "seq_idx": torch.zeros((1, 128), dtype=torch.int32),
            "chunk_indices": torch.zeros(1, dtype=torch.int32),
            "chunk_offsets": torch.zeros(1, dtype=torch.int32),
            "num_seqs": 1,
        }

    actual = runner.run(*tensors, **kwargs)

    assert actual is result
    assert len(runner._cake_runner.calls) == 1
    assert runner._cake_runner.calls[0][0] == tensors
    for name, value in kwargs.items():
        assert runner._cake_runner.calls[0][1][name] is value


@pytest.mark.parametrize("backend", ("cute", "cake"))
@pytest.mark.parametrize("invalid", ("shape", "dtype", "contiguous"))
def test_source_public_out_contract_per_backend_without_gpu(backend, invalid):
    """``out`` is validated against the backend's kernel layout: CuTe chunked
    ``[B, EH, D, C, L]``, Cake token-major ``[B, S, EH, D]``."""

    runner = _public_runner_without_constructor(backend)
    tensors = _cpu_public_run_inputs()
    expected = (1, 128, 2, 64) if backend == "cake" else (1, 2, 64, 1, 128)
    layout = "(B, S, EH, D)" if backend == "cake" else "(B, EH, D, C, L)"
    if invalid == "shape":
        out = torch.empty((1,), dtype=torch.bfloat16)
        match = re.escape(
            f"out shape torch.Size([1]) doesn't match expected {expected}"
        )
    elif invalid == "dtype":
        out = torch.empty(expected, dtype=torch.float16)
        match = "out dtype torch.float16 doesn't match x dtype torch.bfloat16"
    else:
        storage = torch.empty((*expected[:-1], expected[-1] + 1), dtype=torch.bfloat16)
        out = storage[..., : expected[-1]]
        assert not out.is_contiguous()
        match = rf"out must be contiguous in {re.escape(layout)} layout"

    with pytest.raises(AssertionError, match=match):
        runner.run(*tensors, out=out)
    assert runner._cake_runner.calls == []


@pytest.mark.parametrize(
    "invalid,exception",
    (
        ("x_rank", ValueError),
        ("a_dtype", AssertionError),
        ("x_dtype", AssertionError),
        ("b_dtype", AssertionError),
        ("c_dtype", AssertionError),
        ("d_dtype", AssertionError),
        ("z_dtype", AssertionError),
        ("initial_dtype", AssertionError),
        ("seq_idx_shape", AssertionError),
        ("seq_idx_dtype", AssertionError),
        ("chunk_indices_ndim", AssertionError),
        ("chunk_indices_dtype", AssertionError),
        ("chunk_offsets_ndim", AssertionError),
        ("chunk_offsets_dtype", AssertionError),
        ("chunk_vector_shape", AssertionError),
    ),
)
def test_source_public_shared_validation_error_parity_without_gpu(invalid, exception):
    tensors = _cpu_public_run_inputs()
    kwargs = {}
    if invalid == "x_rank":
        tensors = (torch.empty((128, 2, 64), dtype=torch.bfloat16), *tensors[1:])
    elif invalid == "a_dtype":
        tensors = (*tensors[:2], tensors[2].to(torch.bfloat16), *tensors[3:])
    elif invalid in {"x_dtype", "b_dtype", "c_dtype"}:
        tensor_index = {"x_dtype": 0, "b_dtype": 3, "c_dtype": 4}[invalid]
        tensors = (
            *tensors[:tensor_index],
            tensors[tensor_index].to(torch.float16),
            *tensors[tensor_index + 1 :],
        )
    elif invalid == "d_dtype":
        kwargs["D"] = torch.empty(2, dtype=torch.float16)
    elif invalid == "z_dtype":
        kwargs["z"] = torch.empty_like(tensors[0], dtype=torch.float16)
    elif invalid == "initial_dtype":
        kwargs["initial_states"] = torch.empty((1, 2, 64, 128), dtype=torch.float16)
    else:
        seq_idx = torch.empty((1, 128), dtype=torch.int32)
        chunk_indices = torch.zeros(1, dtype=torch.int32)
        chunk_offsets = torch.zeros(1, dtype=torch.int32)
        if invalid == "seq_idx_shape":
            seq_idx = torch.empty((2, 128), dtype=torch.int32)
        elif invalid == "seq_idx_dtype":
            seq_idx = torch.empty((1, 128), dtype=torch.float32)
        elif invalid == "chunk_indices_ndim":
            chunk_indices = torch.zeros((1, 1), dtype=torch.int32)
        elif invalid == "chunk_indices_dtype":
            chunk_indices = torch.zeros(1, dtype=torch.int64)
        elif invalid == "chunk_offsets_ndim":
            chunk_offsets = torch.zeros((1, 1), dtype=torch.int32)
        elif invalid == "chunk_offsets_dtype":
            chunk_offsets = torch.zeros(1, dtype=torch.int64)
        elif invalid == "chunk_vector_shape":
            chunk_offsets = torch.zeros(2, dtype=torch.int32)
        kwargs.update(
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            initial_states=torch.empty((1, 2, 64, 128), dtype=torch.bfloat16),
        )

    errors = {}
    for backend in ("cute", "cake"):
        runner = _public_runner_without_constructor(backend)
        runner._has_d = invalid == "d_dtype"
        runner._has_init_states = invalid == "initial_dtype"
        with pytest.raises(exception) as exc_info:
            runner.run(*tensors, **kwargs)
        errors[backend] = (type(exc_info.value), str(exc_info.value))
        assert runner._cake_runner.calls == []

    assert errors["cake"] == errors["cute"]


def test_source_public_seq_cumsum_helper_contract_without_gpu(monkeypatch):
    module = importlib.import_module("flashinfer.mamba.ssd_combined")
    calls = []

    class SeqCumsumModule:
        @staticmethod
        def seq_chunk_cumsum_tile_state_size(num_seqs):
            calls.append(("tile_state_size", num_seqs))
            return 19

        @staticmethod
        def seq_chunk_cumsum(*args):
            calls.append(("seq_chunk_cumsum", args))

    seq_module = SeqCumsumModule()
    monkeypatch.setattr(module, "_get_seq_chunk_cumsum_module", lambda: seq_module)
    runner = object.__new__(module.SSDCombined)
    runner._seq_cumsum_key = None
    runner._seq_cumsum_buf = None
    seq_idx = torch.tensor([[0, 0, 1, 1]], dtype=torch.int32)
    chunk_indices = torch.tensor([0, 0], dtype=torch.int32)
    chunk_offsets = torch.tensor([0, 2], dtype=torch.int32)
    output = torch.full((3,), -1, dtype=torch.int32)
    tile_state = torch.empty(19, dtype=torch.uint8)

    returned = runner.compute_seq_chunk_cumsum(
        seq_idx,
        chunk_indices,
        chunk_offsets,
        128,
        2,
        seq_chunk_cumsum=output,
        tile_state=tile_state,
    )

    assert returned is output
    assert calls == [
        (
            "seq_chunk_cumsum",
            (
                seq_idx,
                chunk_indices,
                chunk_offsets,
                output,
                tile_state,
                128,
                2,
                2,
            ),
        )
    ]
    assert runner.tile_state_size(7) == 19
    assert calls[-1] == ("tile_state_size", 7)


def _source_cake_runner_without_constructor():
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    runner = object.__new__(module.CakeSSDCombined)
    runner.nheads = 2
    runner.ngroups = 1
    runner.state_dtype = torch.bfloat16
    runner.has_d = False
    runner.d_has_hdim = False
    runner.has_initial_states = False
    runner.has_varlen = False
    runner.has_z = False
    runner.seq_idx_dtype = torch.int32
    runner.chunk_size = 128
    return runner


def _source_cake_varlen_arguments(runner, tensors):
    batch, seqlen = tensors[0].shape[:2]
    runner.has_initial_states = True
    runner.has_varlen = True
    return {
        "initial_states": torch.empty(
            (batch, runner.nheads, 64, 128), dtype=runner.state_dtype
        ),
        "seq_idx": torch.empty((batch, seqlen), dtype=runner.seq_idx_dtype),
        "chunk_indices": torch.arange(batch, dtype=torch.int32),
        "chunk_offsets": torch.zeros(batch, dtype=torch.int32),
        "seq_chunk_cumsum": torch.arange(batch + 1, dtype=torch.int32),
    }


# Validation below dispatch is backend-specific: lock each backend's complete
# rejection surface separately while the pre-dispatch test above enforces exact
# exception-type/message parity for the shared public contract.
@pytest.mark.parametrize(
    "invalid,match",
    (
        ("x_shape", "x must have shape"),
        ("b_shape", "B must have shape"),
        ("c_shape", "C must have the same shape as B"),
        ("x_dtype", "x, B, and C must be bfloat16"),
        ("b_dtype", "x, B, and C must be bfloat16"),
        ("c_dtype", "x, B, and C must be bfloat16"),
        ("dt_shape", "dt must have shape"),
        ("a_shape", "A must have shape"),
        ("dt_dtype", "dt must be bfloat16 or float32"),
        ("d_presence", "runtime D/z presence must match"),
        ("z_presence", "runtime D/z presence must match"),
        ("initial_presence", "runtime initial_states presence must match"),
        ("varlen_metadata", "varlen mode requires seq_idx"),
        ("batched_metadata", "batched mode does not accept varlen metadata"),
        ("batched_cumsum", "batched mode does not accept varlen metadata"),
        ("batched_num_seqs", "batched mode does not accept varlen metadata"),
        (
            "varlen_sequence_count",
            "varlen mode without initial_states requires seq_chunk_cumsum or num_seqs",
        ),
        ("num_seqs_conflict", r"num_seqs \(3\) does not match the sequence count"),
        ("initial_dtype", "initial_states dtype must match state_dtype"),
        ("out_shape", "out must have shape"),
        ("out_dtype", "out must have shape"),
        ("out_contiguous", "out must be contiguous"),
        ("d_shape", "D must have shape"),
        ("d_dtype", "D must have shape"),
        ("z_shape", "z must have the same shape and dtype as x"),
        ("z_dtype", "z must have the same shape and dtype as x"),
        ("initial_shape", "initial_states must have shape"),
        ("seq_idx_shape", "seq_idx shape or dtype"),
        ("seq_idx_dtype", "seq_idx shape or dtype"),
        ("chunk_indices_dtype", "matching int32 vectors"),
        ("chunk_offsets_dtype", "matching int32 vectors"),
        ("chunk_indices_ndim", "matching int32 vectors"),
        ("chunk_vector_shape", "matching int32 vectors"),
        ("seq_cumsum_shape", "seq_chunk_cumsum shape or dtype"),
        ("seq_cumsum_dtype", "seq_chunk_cumsum shape or dtype"),
        # cu_seqlens varlen form (CAKE-934 item 2 option B)
        ("batched_cu_seqlens", "batched mode does not accept varlen metadata"),
        (
            "cu_seqlens_with_triple",
            "cu_seqlens excludes chunk_indices / chunk_offsets",
        ),
        ("cu_seqlens_dtype", r"cu_seqlens must be an int32 vector of num_seqs \+ 1"),
        ("cu_seqlens_ndim", r"cu_seqlens must be an int32 vector of num_seqs \+ 1"),
        ("cu_seqlens_short", r"cu_seqlens must be an int32 vector of num_seqs \+ 1"),
        ("cu_seqlens_batch", r"cu_seqlens describes a packed \[1, total\] stream"),
        (
            "cu_seqlens_precomputed_cumsum",
            "with cu_seqlens the preprocess derives seq_chunk_cumsum",
        ),
        (
            "cu_seqlens_count_conflict",
            r"num_seqs \(3\) does not match the sequence count \(2\) implied by cu_seqlens",
        ),
        (
            "cu_seqlens_initial_conflict",
            r"initial_states \(3\) does not match the sequence count \(2\) implied by cu_seqlens",
        ),
        (
            "triple_at_chunk_256",
            "chunk_size=256: the seq_idx / chunk_indices / chunk_offsets triple is "
            "the chunk-128 logical segmentation",
        ),
    ),
)
def test_source_public_cake_domain_validation_without_gpu(invalid, match):
    cake_runner = _source_cake_runner_without_constructor()
    tensors = list(_cpu_public_run_inputs(batch=2))
    kwargs = {}
    cu_seqlens = torch.tensor([0, 100, 128], dtype=torch.int32)

    if invalid.startswith("cu_seqlens_") or invalid == "triple_at_chunk_256":
        cake_runner.has_varlen = True
        if invalid != "cu_seqlens_batch":
            tensors = list(_cpu_public_run_inputs(batch=1))
    if invalid == "x_shape":
        tensors[0] = torch.empty((2, 128, 3, 64), dtype=torch.bfloat16)
    elif invalid == "b_shape":
        tensors[3] = torch.empty((2, 128, 2, 128), dtype=torch.bfloat16)
    elif invalid == "c_shape":
        tensors[4] = torch.empty((2, 128, 1, 127), dtype=torch.bfloat16)
    elif invalid in {"x_dtype", "b_dtype", "c_dtype"}:
        tensor_index = {"x_dtype": 0, "b_dtype": 3, "c_dtype": 4}[invalid]
        tensors[tensor_index] = tensors[tensor_index].to(torch.float16)
    elif invalid == "dt_shape":
        tensors[1] = torch.empty((2, 128, 3), dtype=torch.float32)
    elif invalid == "a_shape":
        tensors[2] = torch.empty((3,), dtype=torch.float32)
    elif invalid == "dt_dtype":
        tensors[1] = tensors[1].to(torch.float16)
    elif invalid == "d_presence":
        cake_runner.has_d = True
    elif invalid == "z_presence":
        cake_runner.has_z = True
    elif invalid == "initial_presence":
        cake_runner.has_initial_states = True
    elif invalid == "varlen_metadata":
        cake_runner.has_initial_states = True
        cake_runner.has_varlen = True
        kwargs["initial_states"] = torch.empty((2, 2, 64, 128), dtype=torch.bfloat16)
    elif invalid == "batched_metadata":
        kwargs["seq_idx"] = torch.empty((2, 128), dtype=torch.int32)
    elif invalid == "batched_cumsum":
        kwargs["seq_chunk_cumsum"] = torch.empty(3, dtype=torch.int32)
    elif invalid == "batched_num_seqs":
        kwargs["num_seqs"] = 2
    elif invalid == "varlen_sequence_count":
        cake_runner.has_varlen = True
        kwargs.update(
            seq_idx=torch.empty((2, 128), dtype=torch.int32),
            chunk_indices=torch.arange(2, dtype=torch.int32),
            chunk_offsets=torch.zeros(2, dtype=torch.int32),
        )
    elif invalid == "num_seqs_conflict":
        kwargs.update(_source_cake_varlen_arguments(cake_runner, tensors))
        kwargs["num_seqs"] = 3
    elif invalid == "out_shape":
        kwargs["out"] = torch.empty((2, 2, 64, 1, 128), dtype=torch.bfloat16)
    elif invalid == "out_dtype":
        kwargs["out"] = torch.empty((2, 128, 2, 64), dtype=torch.float16)
    elif invalid == "out_contiguous":
        kwargs["out"] = torch.empty((2, 128, 2, 65), dtype=torch.bfloat16)[..., :64]
    elif invalid == "initial_dtype":
        cake_runner.has_initial_states = True
        kwargs["initial_states"] = torch.empty((2, 2, 64, 128), dtype=torch.float16)
    elif invalid in {"d_shape", "d_dtype"}:
        cake_runner.has_d = True
        kwargs["D"] = torch.empty(
            (2, 63),
            dtype=torch.bfloat16 if invalid == "d_shape" else torch.float16,
        )
        if invalid == "d_dtype":
            kwargs["D"] = torch.empty(2, dtype=torch.float16)
    elif invalid in {"z_shape", "z_dtype"}:
        cake_runner.has_z = True
        kwargs["z"] = torch.empty(
            (2, 127, 2, 64) if invalid == "z_shape" else tensors[0].shape,
            dtype=torch.bfloat16 if invalid == "z_shape" else torch.float16,
        )
    elif invalid == "batched_cu_seqlens":
        kwargs["cu_seqlens"] = cu_seqlens
    elif invalid == "cu_seqlens_with_triple":
        kwargs.update(
            cu_seqlens=cu_seqlens,
            chunk_indices=torch.zeros(2, dtype=torch.int32),
            chunk_offsets=torch.zeros(2, dtype=torch.int32),
        )
    elif invalid == "cu_seqlens_dtype":
        kwargs["cu_seqlens"] = cu_seqlens.to(torch.int64)
    elif invalid == "cu_seqlens_ndim":
        kwargs["cu_seqlens"] = cu_seqlens.reshape(1, 3)
    elif invalid == "cu_seqlens_short":
        kwargs["cu_seqlens"] = torch.zeros(1, dtype=torch.int32)
    elif invalid == "cu_seqlens_batch":
        kwargs["cu_seqlens"] = torch.tensor([0, 256], dtype=torch.int32)
    elif invalid == "cu_seqlens_precomputed_cumsum":
        kwargs.update(
            cu_seqlens=cu_seqlens,
            seq_chunk_cumsum=torch.zeros(3, dtype=torch.int32),
        )
    elif invalid == "cu_seqlens_count_conflict":
        kwargs.update(cu_seqlens=cu_seqlens, num_seqs=3)
    elif invalid == "cu_seqlens_initial_conflict":
        cake_runner.has_initial_states = True
        kwargs.update(
            cu_seqlens=cu_seqlens,
            initial_states=torch.empty((3, 2, 64, 128), dtype=torch.bfloat16),
        )
    elif invalid == "triple_at_chunk_256":
        cake_runner.chunk_size = 256
        kwargs.update(
            seq_idx=torch.zeros((1, 128), dtype=torch.int32),
            chunk_indices=torch.zeros(1, dtype=torch.int32),
            chunk_offsets=torch.zeros(1, dtype=torch.int32),
            num_seqs=1,
        )
    else:
        kwargs.update(_source_cake_varlen_arguments(cake_runner, tensors))
        if invalid == "initial_shape":
            kwargs["initial_states"] = torch.empty(
                (2, 2, 64, 127), dtype=torch.bfloat16
            )
        elif invalid == "seq_idx_shape":
            kwargs["seq_idx"] = torch.empty((1, 128), dtype=torch.int32)
        elif invalid == "seq_idx_dtype":
            kwargs["seq_idx"] = torch.empty((2, 128), dtype=torch.int64)
        elif invalid == "chunk_indices_dtype":
            kwargs["chunk_indices"] = torch.arange(2, dtype=torch.int64)
        elif invalid == "chunk_offsets_dtype":
            kwargs["chunk_offsets"] = torch.zeros(2, dtype=torch.int64)
        elif invalid == "chunk_indices_ndim":
            kwargs["chunk_indices"] = torch.zeros((1, 2), dtype=torch.int32)
        elif invalid == "chunk_vector_shape":
            kwargs["chunk_offsets"] = torch.zeros(3, dtype=torch.int32)
        elif invalid == "seq_cumsum_shape":
            kwargs["seq_chunk_cumsum"] = torch.empty(2, dtype=torch.int32)
        elif invalid == "seq_cumsum_dtype":
            kwargs["seq_chunk_cumsum"] = torch.empty(3, dtype=torch.int64)

    with pytest.raises(ValueError, match=match):
        cake_runner.run(*tensors, **kwargs)


def _source_cute_runner_without_constructor():
    runner = _public_runner_without_constructor("cute")
    runner._io_torch_dtype = torch.bfloat16
    runner._cumsum_dtype = object()
    runner._state_torch_dtype = torch.bfloat16
    runner._has_d = False
    runner._d_has_hdim = False
    runner._has_init_states = False
    runner._has_varlen = False
    runner._has_z = False
    runner._get_or_alloc_fstate = lambda batch: torch.empty(
        (batch, 2, 64, 128), dtype=runner._state_torch_dtype
    )
    return runner


@pytest.mark.parametrize(
    "invalid,exception,match",
    (
        ("checkpoint", ValueError, "require SSDCombined backend='cake'"),
        ("seq_idx_shape", AssertionError, "seq_idx shape"),
        ("seq_idx_dtype", AssertionError, "seq_idx must be int32 or int64"),
        ("chunk_indices_ndim", AssertionError, "chunk_indices must be 1D"),
        ("chunk_indices_dtype", AssertionError, "chunk_indices must be int32"),
        ("chunk_offsets_ndim", AssertionError, "chunk_offsets must be 1D"),
        ("chunk_offsets_dtype", AssertionError, "chunk_offsets must be int32"),
        ("chunk_vector_shape", AssertionError, "must have the same shape"),
        ("x_dtype", AssertionError, "x dtype"),
        ("b_dtype", AssertionError, "B dtype"),
        ("c_dtype", AssertionError, "C dtype"),
        ("d_dtype", AssertionError, "D dtype"),
        ("z_dtype", AssertionError, "z dtype"),
        ("initial_dtype", AssertionError, "init_states dtype"),
        ("varlen_initial", ValueError, "initial_states must be provided"),
        ("seqlen", AssertionError, "must be divisible by chunk_size"),
        ("num_seqs", ValueError, "num_seqs requires SSDCombined backend='cake'"),
        ("cu_seqlens", ValueError, "cu_seqlens requires SSDCombined backend='cake'"),
    ),
)
def test_source_public_cute_backend_validation_without_gpu(
    monkeypatch, invalid, exception, match
):
    module = importlib.import_module("flashinfer.mamba.ssd_combined")

    def chunk_cumsum(dt, _a, chunk_size, **_kwargs):
        batch, seqlen, nheads = dt.shape
        shape = (batch, nheads, seqlen // chunk_size, chunk_size)
        return torch.empty(shape, dtype=torch.float32), torch.empty(
            shape, dtype=torch.bfloat16
        )

    monkeypatch.setattr(module, "chunk_cumsum_fwd", chunk_cumsum)
    monkeypatch.setattr(module.cutlass_torch, "dtype", lambda _: torch.float32)
    runner = _source_cute_runner_without_constructor()
    tensors = list(_cpu_public_run_inputs(batch=2))
    kwargs = {}
    seq_idx = torch.empty((2, 128), dtype=torch.int32)
    chunk_indices = torch.arange(2, dtype=torch.int32)
    chunk_offsets = torch.zeros(2, dtype=torch.int32)

    if invalid == "checkpoint":
        kwargs.update(
            checkpoint_token_indices=torch.zeros(2, dtype=torch.int32),
            checkpoint_state_slots=torch.zeros(2, dtype=torch.int32),
            checkpoint_states=torch.empty((1, 2, 64, 128), dtype=torch.bfloat16),
        )
    elif invalid == "seq_idx_shape":
        kwargs["seq_idx"] = torch.empty((1, 128), dtype=torch.int32)
    elif invalid == "seq_idx_dtype":
        kwargs["seq_idx"] = torch.empty((2, 128), dtype=torch.float32)
    elif invalid == "chunk_indices_ndim":
        kwargs["chunk_indices"] = torch.empty((1, 2), dtype=torch.int32)
    elif invalid == "chunk_indices_dtype":
        kwargs["chunk_indices"] = torch.empty(2, dtype=torch.int64)
    elif invalid == "chunk_offsets_ndim":
        kwargs["chunk_offsets"] = torch.empty((1, 2), dtype=torch.int32)
    elif invalid == "chunk_offsets_dtype":
        kwargs["chunk_offsets"] = torch.empty(2, dtype=torch.int64)
    elif invalid == "chunk_vector_shape":
        kwargs.update(
            chunk_indices=chunk_indices,
            chunk_offsets=torch.empty(3, dtype=torch.int32),
        )
    elif invalid in {"x_dtype", "b_dtype", "c_dtype"}:
        tensor_index = {"x_dtype": 0, "b_dtype": 3, "c_dtype": 4}[invalid]
        tensors[tensor_index] = tensors[tensor_index].to(torch.float16)
    elif invalid == "d_dtype":
        runner._has_d = True
        kwargs["D"] = torch.empty(2, dtype=torch.float16)
    elif invalid == "z_dtype":
        kwargs["z"] = torch.empty_like(tensors[0], dtype=torch.float16)
    elif invalid == "initial_dtype":
        runner._has_init_states = True
        kwargs["initial_states"] = torch.empty((2, 2, 64, 128), dtype=torch.float16)
    elif invalid == "varlen_initial":
        kwargs.update(
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
        )
    elif invalid == "seqlen":
        x, dt, A, B, C = tensors
        tensors = [x[:, :-1], dt[:, :-1], A, B[:, :-1], C[:, :-1]]
    elif invalid == "num_seqs":
        kwargs["num_seqs"] = 2
    elif invalid == "cu_seqlens":
        kwargs["cu_seqlens"] = torch.tensor([0, 128, 256], dtype=torch.int32)

    with pytest.raises(exception, match=match):
        runner.run(*tensors, **kwargs)


@pytest.mark.parametrize("invalid", ("dt_bias_shape", "dt_bias_dtype"))
def test_source_public_cake_dt_bias_validation_without_gpu(monkeypatch, invalid):
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    monkeypatch.setattr(module, "_target_arch", lambda *_: "sm_103a")
    monkeypatch.setattr(module, "_cuda_device_index", lambda _: 0)
    cake_runner = _source_cake_runner_without_constructor()
    cake_runner._get_workspace = lambda **_: {
        "final": torch.empty((2, 2, 64, 128), dtype=torch.bfloat16)
    }
    cake_runner._dummy = lambda device, dtype: torch.empty(
        1, dtype=dtype, device=device
    )
    runner = _public_runner_without_constructor("cake")
    runner._cake_runner = cake_runner
    tensors = _cpu_public_run_inputs(batch=2)
    dt_bias = torch.empty(
        3 if invalid == "dt_bias_shape" else 2,
        dtype=torch.float32 if invalid == "dt_bias_shape" else torch.float16,
    )

    with pytest.raises(ValueError, match="dt_bias must have shape"):
        runner.run(*tensors, dt_bias=dt_bias)


@pytest.mark.parametrize("empty", ("tokens", "batch"))
def test_source_cake_rejects_zero_token_calls_without_gpu(empty):
    """A zero-token or zero-batch call has no chunk to run (``nchunks == 0``
    would reach the workspace's integer division); it is rejected before any
    workspace or metadata is built."""
    runner = _source_cake_runner_without_constructor()
    x, dt, A, B, C = _cpu_public_run_inputs(batch=0 if empty == "batch" else 1)
    if empty == "tokens":
        x, dt, B, C = (value[:, :0] for value in (x, dt, B, C))

    with pytest.raises(ValueError, match="positive batch and sequence length"):
        runner.run(x, dt, A, B, C)


def test_source_public_cake_rejects_non_cuda_inputs_without_gpu(monkeypatch):
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    monkeypatch.setattr(module, "_target_arch", lambda *_: "sm_103a")
    cake_runner = _source_cake_runner_without_constructor()
    cake_runner._get_workspace = lambda **_: {
        "final": torch.empty((1, 2, 64, 128), dtype=torch.bfloat16)
    }
    runner = _public_runner_without_constructor("cake")
    runner._cake_runner = cake_runner

    with pytest.raises(ValueError, match="inputs must be on a CUDA device"):
        runner.run(*_cpu_public_run_inputs())


@pytest.mark.parametrize(
    "invalid,match",
    (
        ("partial", "must be provided together"),
        ("token_shape", "checkpoint_token_indices must be"),
        ("token_dtype", "checkpoint_token_indices must be"),
        ("token_contiguous", "checkpoint_token_indices must be"),
        ("slot_shape", "checkpoint_state_slots must be"),
        ("slot_dtype", "checkpoint_state_slots must be"),
        ("slot_contiguous", "checkpoint_state_slots must be"),
        ("state_shape", "checkpoint_states must be contiguous"),
        ("state_dtype", "checkpoint_states must be contiguous"),
        ("state_contiguous", "checkpoint_states must be contiguous"),
    ),
)
def test_source_cake_checkpoint_validation_without_gpu(invalid, match):
    runner = _source_cake_runner_without_constructor()
    tensors = _cpu_public_run_inputs(batch=2)
    token_storage = torch.tensor([16, -1, 32, -1], dtype=torch.int32)
    slot_storage = torch.tensor([0, -1, 1, -1], dtype=torch.int32)
    kwargs = {
        "checkpoint_token_indices": token_storage[:2].clone(),
        "checkpoint_state_slots": slot_storage[:2].clone(),
        "checkpoint_states": torch.empty((2, 2, 64, 128), dtype=torch.bfloat16),
    }
    if invalid == "partial":
        kwargs["checkpoint_state_slots"] = None
    elif invalid == "token_shape":
        kwargs["checkpoint_token_indices"] = torch.empty(1, dtype=torch.int32)
    elif invalid == "token_dtype":
        kwargs["checkpoint_token_indices"] = torch.empty(2, dtype=torch.int64)
    elif invalid == "token_contiguous":
        kwargs["checkpoint_token_indices"] = token_storage[::2]
    elif invalid == "slot_shape":
        kwargs["checkpoint_state_slots"] = torch.empty(1, dtype=torch.int32)
    elif invalid == "slot_dtype":
        kwargs["checkpoint_state_slots"] = torch.empty(2, dtype=torch.int64)
    elif invalid == "slot_contiguous":
        kwargs["checkpoint_state_slots"] = slot_storage[::2]
    elif invalid == "state_shape":
        kwargs["checkpoint_states"] = torch.empty((2, 2, 64, 127), dtype=torch.bfloat16)
    elif invalid == "state_dtype":
        kwargs["checkpoint_states"] = torch.empty((2, 2, 64, 128), dtype=torch.float16)
    else:
        state_storage = torch.empty((2, 2, 64, 129), dtype=torch.bfloat16)
        kwargs["checkpoint_states"] = state_storage[..., :128]
        assert not kwargs["checkpoint_states"].is_contiguous()

    with pytest.raises(ValueError, match=match):
        runner.run(*tensors, **kwargs)


def test_source_runner_forwards_softplus_and_checkpoint_count(monkeypatch):
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = {}

    monkeypatch.setattr(module, "_target_arch", lambda *_: "sm_103a")
    monkeypatch.setattr(module, "_cuda_device_index", lambda _: 0)
    monkeypatch.setattr(module, "_sm_count", lambda _: 1)
    monkeypatch.setattr(
        module,
        "_launch_program",
        lambda name, _arch, **kwargs: calls.__setitem__(name, kwargs),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda *_: SimpleNamespace(cuda_stream=0x1234),
    )

    batch, seqlen, nheads, ngroups = 2, 128, 1, 1
    x = torch.empty((batch, seqlen, nheads, 64), dtype=torch.bfloat16)
    dt = torch.empty((batch, seqlen, nheads), dtype=torch.float32)
    A = torch.empty((nheads,), dtype=torch.float32)
    B = torch.empty((batch, seqlen, ngroups, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)
    checkpoint_token_indices = torch.tensor([32, 64], dtype=torch.int32)
    checkpoint_state_slots = torch.tensor([0, 2], dtype=torch.int32)
    checkpoint_states = torch.empty((3, nheads, 64, 128), dtype=torch.bfloat16)
    runner = module.CakeSSDCombined(
        128,
        nheads,
        64,
        128,
        ngroups,
        io_dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        has_d=False,
        d_has_hdim=False,
        has_initial_states=False,
        has_varlen=False,
        has_z=False,
        seq_idx_dtype=torch.int32,
    )

    first = runner.run(
        x,
        dt,
        A,
        B,
        C,
        dt_softplus=False,
        checkpoint_token_indices=checkpoint_token_indices,
        checkpoint_state_slots=checkpoint_state_slots,
        checkpoint_states=checkpoint_states,
    )
    second = runner.run(
        x,
        dt,
        A,
        B,
        C,
        dt_softplus=False,
        checkpoint_token_indices=checkpoint_token_indices,
        checkpoint_state_slots=checkpoint_state_slots,
        checkpoint_states=checkpoint_states,
    )

    exact = calls["exactfused_bf16_batched"]
    assert exact["preprocess"]["dt_softplus"] == 0
    assert exact["preprocess"]["write_seq_chunk_cumsum"] == 0
    # The fused program computes delta / cumsum in-CTA: its preprocess stage
    # is skipped (grid 0) on a batched call.
    assert exact["preprocess_grid"] == (0, 1, 1)
    main = exact["main"]
    assert main["metadata_from_cu_seqlens"] == 0
    assert main["preprocess_status"] is exact["preprocess"]["preprocess_status"]
    assert main["cu_seqlens"] is exact["preprocess"]["cu_seqlens"]
    assert main["dt_softplus"] == 0
    assert main["checkpoint_state_count"] == checkpoint_states.shape[0]
    assert isinstance(first, tuple) and isinstance(second, tuple)
    assert first[0].data_ptr() != second[0].data_ptr()
    assert first[1].data_ptr() != second[1].data_ptr()


@pytest.mark.parametrize(
    "family,state_dtype,mode_varlen,expected",
    [
        ("exact", torch.bfloat16, False, "exact_bf16_batched"),
        ("exact", torch.bfloat16, True, "exact_bf16_varlen"),
        ("exact", torch.float16, False, "exact_f16_batched"),
        ("exact", torch.float16, True, "exact_f16_varlen"),
        ("exact", torch.float32, False, "exact_f32_batched"),
        ("exact", torch.float32, True, "exact_f32_varlen"),
        ("exactfused", torch.bfloat16, False, "exactfused_bf16_batched"),
        ("exactfused", torch.bfloat16, True, "exactfused_bf16_varlen"),
        ("exactfused", torch.float16, False, "exactfused_f16_batched"),
        ("exactfused", torch.float16, True, "exactfused_f16_varlen"),
        ("exactfused", torch.float32, False, "exactfused_f32_batched"),
        ("exactfused", torch.float32, True, "exactfused_f32_varlen"),
        ("chunkpar", torch.bfloat16, False, "chunkpar_bf16_batched"),
        ("chunkpar", torch.bfloat16, True, "chunkpar_bf16_varlen"),
        ("chunkpar", torch.float16, False, "chunkpar_f16_batched"),
        ("chunkpar", torch.float16, True, "chunkpar_f16_varlen"),
        ("chunkpar", torch.float32, False, "chunkpar_f32_batched"),
        ("chunkpar", torch.float32, True, "chunkpar_f32_varlen"),
        ("exactpre", torch.bfloat16, False, "exactpre_bf16_batched"),
        ("exactpre", torch.bfloat16, True, "exactpre_bf16_varlen"),
        ("exactpre", torch.float16, False, "exactpre_f16_batched"),
        ("exactpre", torch.float16, True, "exactpre_f16_varlen"),
        ("exactpre", torch.float32, False, "exactpre_f32_batched"),
        ("exactpre", torch.float32, True, "exactpre_f32_varlen"),
    ],
)
def test_source_program_name_covers_every_state_dtype(
    family, state_dtype, mode_varlen, expected
):
    """Every kernel family serves every admitted input: the program follows
    from the family, the state dtype and the batched/packed mode alone."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")

    actual = module._program_name(family, state_dtype, mode_varlen)

    assert actual == expected
    assert actual in module._PROGRAMS
    assert module._PROGRAMS[actual].family == family
    assert module._STATE_DTYPE_CODES == {
        "bf16": (4, 16),
        "f16": (2, 16),
        "f32": (2, 32),
    }
    assert not hasattr(module, "_select_scan_route")


def _cpu_forwarding_runner(module, monkeypatch, calls, **constructor):
    """A real runner on CPU tensors whose launcher call is captured; the device
    answers as ``arch`` (default ``sm_103a``), which decides the route of the
    seq_idx triple form deriving its table."""

    arch = constructor.pop("arch", "sm_103a")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(module, "_target_arch", lambda *_: arch)
    monkeypatch.setattr(module, "_cuda_device_index", lambda _: 0)
    monkeypatch.setattr(module, "_sm_count", lambda _: 1)
    monkeypatch.setattr(
        module,
        "_launch_program",
        lambda name, _arch, **kwargs: calls.append((name, kwargs)),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda *_: SimpleNamespace(cuda_stream=0x1234),
    )
    return module.CakeSSDCombined(
        constructor.pop("chunk_size", 128),
        constructor.pop("nheads", 1),
        64,
        128,
        constructor.pop("ngroups", 1),
        io_dtype=torch.bfloat16,
        state_dtype=constructor.pop("state_dtype", torch.bfloat16),
        has_d=False,
        d_has_hdim=False,
        has_initial_states=constructor.pop("has_initial_states", False),
        has_varlen=constructor.pop("has_varlen", False),
        has_z=constructor.pop("has_z", False),
        seq_idx_dtype=constructor.pop("seq_idx_dtype", torch.int32),
    )


def _derived_family(module, lengths, *, nheads=8):
    """The family of the seq_idx triple form deriving a runner-owned table
    inside the fused window on this device: ``exactmeta`` on SM100 (the light
    metadata kernel then the fused main kernel, two launches), ``exactone`` on
    SM103 and SM107 (the fused main kernel derives its table in a CTA
    prologue, one launch: the host interval between two launches hides a
    lighter stage there), the two-kernel program elsewhere."""

    capability = torch.cuda.get_device_capability()
    if capability == (10, 0):
        return "exactmeta"
    if capability in ((10, 3), (10, 7)):
        return "exactone"
    return module.two_kernel_family(
        mode_varlen=True, num_sequences=len(lengths), seqlen=sum(lengths)
    )


def _without_table_form(module, name, *, state_dtype, num_sequences, nheads=8):
    """The ring one-launch program a derived-table call stands for on this
    device: ``name`` without the prologue form that ``exactone_table_variant``
    picks for the call's launch grid (SM107: ``tablefirst`` below its grid
    threshold, ``tableroles`` from it on).  The forms move only where the CTA
    prologue derives the table; the arithmetic is the ring program's.  Every
    other program is returned as is."""

    program = module._PROGRAMS[name]
    if program.family != "exactone" or program.variant == "noring":
        return name
    form = module.exactone_table_variant(
        capability=torch.cuda.get_device_capability(),
        variant="",
        state_dtype=state_dtype,
        grid=module._persistent_grid_size(
            total_work=num_sequences * nheads,
            sm_count=module._sm_count(torch.cuda.current_device()),
        ),
    )
    assert program.variant == form, (name, form)
    return name[: -len(form) - 1] if form else name


def _two_kernel_program(module, name, *, num_sequences, seqlen):
    """Assert ``name`` is the two-kernel program of a packed-varlen call
    (``two_kernel_family``) and return its family."""

    family = module.two_kernel_family(
        mode_varlen=True, num_sequences=num_sequences, seqlen=seqlen
    )
    assert module._PROGRAMS[name].family == family, name
    assert name.startswith(f"{family}_bf16_varlen"), name
    return family


_ROUTE_ARCHES = ("sm_100a", "sm_103a", "sm_107a")
# The architectures routing the seq_idx triple form deriving its table
# through the light metadata stage (``exactmeta``): SM100 alone (SM103 and
# SM107 run the one-launch ``exactone`` render).
_LIGHT_STAGE_ARCHES = ("sm_100a",)


@pytest.mark.parametrize("arch", _ROUTE_ARCHES)
@pytest.mark.parametrize(
    "case",
    (
        "precomputed",
        "update_caller_buffer",
        "runner_buffer_from_initial_states",
        "runner_buffer_from_num_seqs",
    ),
)
def test_source_varlen_cumsum_binding_without_gpu(monkeypatch, case, arch):
    """The preprocess owns ``seq_chunk_cumsum``: one launcher call binds the
    same vector to both stages and ``write_seq_chunk_cumsum`` says whether
    the preprocess fills it; ``num_seqs`` supplies the count when there are
    no initial states.  The seq_idx triple form runs, where the table is
    published (runner-owned, or the caller's buffer with
    ``update_seq_chunk_cumsum=True``), the ``exactmeta`` program on SM100
    (the light metadata kernel publishes it ahead of the fused main kernel);
    on SM103 and SM107 the one-launch ``exactone`` program for a runner-owned
    table (the main kernel derives it, nothing is published) and the two-kernel
    program for a caller buffer; and the fused single-launch program where
    the caller's table is trusted (round 6 F1b: its preprocess grid is 0)."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    has_initial_states = case != "runner_buffer_from_num_seqs"
    runner = _cpu_forwarding_runner(
        module,
        monkeypatch,
        calls,
        has_varlen=True,
        has_initial_states=has_initial_states,
        arch=arch,
    )
    seqlen, num_seqs = 300, 2
    x = torch.empty((1, seqlen, 1, 64), dtype=torch.bfloat16)
    dt = torch.empty((1, seqlen, 1), dtype=torch.float32)
    A = torch.empty((1,), dtype=torch.float32)
    B = torch.empty((1, seqlen, 1, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)
    kwargs = {
        "seq_idx": torch.zeros((1, seqlen), dtype=torch.int32),
        "chunk_indices": torch.tensor([0, 1, 2], dtype=torch.int32),
        "chunk_offsets": torch.tensor([0, 0, 0], dtype=torch.int32),
    }
    caller_vector = torch.tensor([0, 1, 3], dtype=torch.int32)
    if case == "precomputed":
        kwargs["seq_chunk_cumsum"] = caller_vector
    elif case == "update_caller_buffer":
        kwargs["seq_chunk_cumsum"] = caller_vector
        kwargs["update_seq_chunk_cumsum"] = True
    else:
        kwargs["num_seqs"] = num_seqs
    if has_initial_states:
        kwargs["initial_states"] = torch.empty(
            (num_seqs, 1, 64, 128), dtype=torch.bfloat16
        )

    out, final = runner.run(x, dt, A, B, C, **kwargs)

    assert tuple(out.shape) == (1, seqlen, 1, 64)
    assert tuple(final.shape) == (num_seqs, 1, 64, 128)
    ((name, launch),) = calls
    # seq_idx triple form: the fused single launch only where its preprocess
    # grid is 0 (a trusted caller table); the ``exactmeta`` program -- the
    # 12-argument metadata stage publishing the table into the vector the
    # main kernel reads, one thread per segment -- where it is published.
    preprocess, main = launch["preprocess"], launch["main"]
    if case == "precomputed":
        assert name == "exactfused_bf16_varlen"
        assert launch["preprocess_grid"] == (0, 1, 1)
        assert set(preprocess) == set(module._PREPROCESS_ARGS)
        assert preprocess["seq_chunk_cumsum"] is main["seq_chunk_cumsum"]
    elif arch in _LIGHT_STAGE_ARCHES:
        assert name == "exactmeta_bf16_varlen"
        assert launch["preprocess_grid"] == (1, 1, 1)  # 3 segments, 256 threads
        assert set(preprocess) == set(module._METADATA_ARGS)
        assert preprocess["sequence_offsets"] is main["seq_chunk_cumsum"]
        assert preprocess["num_segments"] == 3 and preprocess["seqlen"] == seqlen
        assert preprocess["chunk_indices"] is main["chunk_indices"]
        assert preprocess["chunk_offsets"] is main["chunk_offsets"]
    elif case == "update_caller_buffer":
        # SM103 and SM107 keep the two-kernel program where the caller's buffer
        # must receive the table: the full preprocess publishes the vector it reads.
        _two_kernel_program(module, name, num_sequences=num_seqs, seqlen=seqlen)
        assert launch["preprocess_grid"][0] >= 1
        assert set(preprocess) == set(module._PREPROCESS_ARGS)
        assert preprocess["write_seq_chunk_cumsum"] == 1
        assert preprocess["seq_chunk_cumsum"] is main["seq_chunk_cumsum"]
    else:
        # SM103 and SM107 derive a runner-owned table in the fused main kernel's
        # CTA prologue: the one-launch exactone program (the stage's grid is 0,
        # nothing is published), its scratch one row per CTA (one CTA here:
        # 2 items on this 1-SM device; on SM107 a grid that small takes the
        # tablefirst form).
        assert name == (
            "exactone_bf16_varlen_tablefirst"
            if arch == "sm_107a"
            else "exactone_bf16_varlen"
        )
        assert launch["preprocess_grid"] == (0, 1, 1)
        assert launch["main_grid"] == (1, 1, 1)
        assert set(preprocess) == set(module._PREPROCESS_ARGS)
        assert preprocess["write_seq_chunk_cumsum"] == 0
        assert preprocess["seq_chunk_cumsum"] is main["seq_chunk_cumsum"]
    # The published table: the metadata stage's ``sequence_offsets``, the
    # segment preprocess's ``seq_chunk_cumsum`` (the one-launch program's
    # private scratch rides in the same slot, unpublished).
    table_arg = (
        "sequence_offsets" if name.startswith("exactmeta_") else "seq_chunk_cumsum"
    )
    assert main["metadata_from_cu_seqlens"] == (
        module._TABLE_IN_KERNEL_I32 if name.startswith("exactone_") else 0
    )
    assert preprocess["num_sequences"] == main["sequence_count"] == num_seqs
    assert preprocess["seq_idx_i32"] is main["seq_idx_i32"] is kwargs["seq_idx"]
    assert preprocess["seq_idx_int64"] == 0
    assert main["has_initial"] == int(has_initial_states)
    assert main["nchunks"] == 3 and main["seqlen"] == seqlen
    if case == "precomputed":
        assert preprocess["seq_chunk_cumsum"] is caller_vector
        assert preprocess["write_seq_chunk_cumsum"] == 0
    elif case == "update_caller_buffer":
        # The publishing stage writes into the caller's buffer.
        assert preprocess[table_arg] is caller_vector
    else:
        bound = preprocess[table_arg]
        assert bound.dtype == torch.int32 and bound.numel() == num_seqs + 1
        # The runner-owned vector is reused across calls of the same count.
        runner.run(x, dt, A, B, C, **kwargs)
        assert calls[-1][1]["preprocess"][table_arg] is bound


@pytest.mark.parametrize(
    "case",
    (
        "runner_buffer",
        "update_caller_buffer",
        "zero_state",
        "seq_idx_rides_along",
        "chunk_256",
    ),
)
def test_source_cu_seqlens_binding_without_gpu(monkeypatch, case):
    """The cu_seqlens form binds ``metadata_from_cu_seqlens=1``, the caller's
    ``cu_seqlens``, the runner-owned ``[bound + 1]`` chunk tables shared by
    both stages, ``num_segments`` / ``num_logical_chunks`` = the bound,
    ``write_seq_chunk_cumsum=0`` (the derivation publishes the cumsum into
    the runner's or the caller's buffer), the sequence count from the host
    shape, and leaves ``seq_idx`` optional; ``chunk_size`` is a label."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    has_initial_states = case != "zero_state"
    runner = _cpu_forwarding_runner(
        module,
        monkeypatch,
        calls,
        has_varlen=True,
        has_initial_states=has_initial_states,
        chunk_size=256 if case == "chunk_256" else 128,
    )
    seqlen, num_seqs = 300, 2
    x = torch.empty((1, seqlen, 1, 64), dtype=torch.bfloat16)
    dt = torch.empty((1, seqlen, 1), dtype=torch.float32)
    A = torch.empty((1,), dtype=torch.float32)
    B = torch.empty((1, seqlen, 1, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)
    cu_seqlens = torch.tensor([0, 100, 300], dtype=torch.int32)
    kwargs = {"cu_seqlens": cu_seqlens}
    caller_vector = torch.full((num_seqs + 1,), -7, dtype=torch.int32)
    if case == "update_caller_buffer":
        kwargs.update(seq_chunk_cumsum=caller_vector, update_seq_chunk_cumsum=True)
    if case == "seq_idx_rides_along":
        kwargs["seq_idx"] = torch.zeros((1, seqlen), dtype=torch.int32)
    if has_initial_states:
        kwargs["initial_states"] = torch.empty(
            (num_seqs, 1, 64, 128), dtype=torch.bfloat16
        )

    out, final = runner.run(x, dt, A, B, C, **kwargs)

    assert tuple(out.shape) == (1, seqlen, 1, 64)
    assert tuple(final.shape) == (num_seqs, 1, 64, 128)
    ((name, launch),) = calls
    assert name == "exactfused_bf16_varlen"
    preprocess, main = launch["preprocess"], launch["main"]
    bound = module._segment_bound(seqlen, num_seqs)
    assert bound == 3 + 2 * num_seqs
    assert preprocess["metadata_from_cu_seqlens"] == 1
    assert preprocess["cu_seqlens"] is cu_seqlens
    assert preprocess["direct_varlen_metadata"] == 1
    assert preprocess["num_segments"] == bound
    assert main["num_logical_chunks"] == bound
    assert main["mode_varlen"] == 1
    assert preprocess["chunk_indices"] is main["chunk_indices"]
    assert preprocess["chunk_offsets"] is main["chunk_offsets"]
    for table in (main["chunk_indices"], main["chunk_offsets"]):
        assert table.dtype == torch.int32 and table.numel() == bound + 1
    assert preprocess["num_sequences"] == main["sequence_count"] == num_seqs
    assert preprocess["write_seq_chunk_cumsum"] == 0
    assert preprocess["seq_chunk_cumsum"] is main["seq_chunk_cumsum"]
    assert preprocess["checkpoint_state_count"] == main["checkpoint_state_count"] == 0
    assert preprocess["checkpoint_token_indices"] is main["checkpoint_token_indices"]
    assert main["has_initial"] == int(has_initial_states)
    assert main["nchunks"] == 3 and main["seqlen"] == seqlen
    if case == "update_caller_buffer":
        assert preprocess["seq_chunk_cumsum"] is caller_vector
    else:
        vector = preprocess["seq_chunk_cumsum"]
        assert vector.dtype == torch.int32 and vector.numel() == num_seqs + 1
    if case == "seq_idx_rides_along":
        assert preprocess["seq_idx_i32"] is kwargs["seq_idx"]
    else:
        assert preprocess["seq_idx_i32"].numel() == 1  # the dummy: seq_idx is optional
    assert preprocess["seq_idx_int64"] == 0
    # The tables are reused across calls of the same geometry.
    runner.run(x, dt, A, B, C, **kwargs)
    assert calls[-1][1]["main"]["chunk_indices"] is main["chunk_indices"]
    # Checkpoints ride into the derivation (the unaligned one becomes a boundary).
    checkpoints = dict(
        checkpoint_token_indices=torch.tensor([-1, 156], dtype=torch.int32),
        checkpoint_state_slots=torch.tensor([-1, 0], dtype=torch.int32),
        checkpoint_states=torch.empty((1, 1, 64, 128), dtype=torch.bfloat16),
    )
    runner.run(x, dt, A, B, C, **kwargs, **checkpoints)
    preprocess = calls[-1][1]["preprocess"]
    assert preprocess["checkpoint_state_count"] == 1
    assert (
        preprocess["checkpoint_token_indices"]
        is checkpoints["checkpoint_token_indices"]
    )
    assert (
        preprocess["num_segments"] == bound
    )  # the bound already holds one per sequence


def test_source_chunk_size_is_a_caller_convention_without_gpu(monkeypatch):
    """Any positive ``chunk_size`` constructs a runner that launches the same
    chunk-128 programs; a non-positive or non-int one is refused."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    for chunk_size in (64, 256, 512):
        runner = _cpu_forwarding_runner(
            module, monkeypatch, calls, chunk_size=chunk_size
        )
        assert runner.chunk_size == chunk_size
        x = torch.empty((2, 256, 1, 64), dtype=torch.bfloat16)
        dt = torch.empty((2, 256, 1), dtype=torch.float32)
        A = torch.empty((1,), dtype=torch.float32)
        B = torch.empty((2, 256, 1, 128), dtype=torch.bfloat16)
        runner.run(x, dt, A, B, torch.empty_like(B))
        name, launch = calls[-1]
        assert name == "exactfused_bf16_batched"
        assert launch["main"]["nchunks"] == 2  # 128-token tiling regardless
    for chunk_size in (0, -128, 128.0, True):
        with pytest.raises((ValueError, TypeError)):
            module.CakeSSDCombined(
                chunk_size,
                1,
                64,
                128,
                1,
                io_dtype=torch.bfloat16,
                state_dtype=torch.bfloat16,
                has_d=False,
                d_has_hdim=False,
                has_initial_states=False,
                has_varlen=False,
                has_z=False,
                seq_idx_dtype=torch.int32,
            )


def test_source_batched_unaligned_seqlen_binding_without_gpu(monkeypatch):
    """Batched mode: ceil'd chunk count, partial trailing segment length, the
    cumsum dummy with ``write_seq_chunk_cumsum=0``, FP16 delta workspace."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    runner = _cpu_forwarding_runner(module, monkeypatch, calls, nheads=2, ngroups=1)
    batch, seqlen = 2, 1000
    x = torch.empty((batch, seqlen, 2, 64), dtype=torch.bfloat16)
    dt = torch.empty((batch, seqlen, 2), dtype=torch.bfloat16)
    A = torch.empty((2,), dtype=torch.float32)
    B = torch.empty((batch, seqlen, 1, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)

    out, final = runner.run(x, dt, A, B, C)

    assert tuple(out.shape) == (batch, seqlen, 2, 64)
    assert tuple(final.shape) == (batch, 2, 64, 128)
    ((name, launch),) = calls
    assert name == "exactfused_bf16_batched"
    preprocess, main = launch["preprocess"], launch["main"]
    assert main["nchunks"] == 8 and main["num_logical_chunks"] == 8
    assert preprocess["num_segments"] == 16
    assert preprocess["segment_lengths"].tolist() == [128] * 7 + [104] + [128] * 7 + [
        104
    ]
    assert preprocess["segment_starts"].tolist() == [
        *(chunk * 128 for chunk in range(8)),
        *(1000 + chunk * 128 for chunk in range(8)),
    ]
    assert preprocess["delta"].dtype == torch.float16
    assert preprocess["delta"].shape == (32, 128)
    assert preprocess["write_seq_chunk_cumsum"] == 0
    assert preprocess["num_sequences"] == batch
    assert main["mode_varlen"] == 0
    # bf16 dt is widened into the FP32 workspace both stages read.
    assert preprocess["dt"] is main["dt"] and preprocess["dt"].dtype == torch.float32
    # No dt_bias: the zero vector allocated with the workspace is bound.
    assert preprocess["dt_bias"] is main["dt_bias"]
    assert preprocess["dt_bias"].dtype == torch.float32
    assert not preprocess["dt_bias"].any()


@pytest.mark.parametrize("varlen", (False, True), ids=("batched_2x50", "varlen_8"))
def test_source_short_total_tokens_binding_without_gpu(monkeypatch, varlen):
    """CAKE-1063 / round 4: with fewer than 128 tokens the caller's x/B/C/out
    are bound directly -- the exported tensor maps admit a token extent below
    the 128-row box and TMA zero-fills the rows past ``seqlen`` -- and every
    logical argument -- ``seqlen``, ``nchunks``, the preprocess tables -- keeps
    the caller's extent, exactly like a 128-token call.  No runner-owned pad
    buffers, no copy-back."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    runner = _cpu_forwarding_runner(
        module, monkeypatch, calls, nheads=2, ngroups=1, has_varlen=varlen
    )

    def launch(name, _arch, **launch_kwargs):
        calls.append((name, launch_kwargs))
        # Stand in for the kernel: write a row pattern into the bound output.
        bound = launch_kwargs["main"]["out_map"]
        rows = torch.arange(bound.shape[1], dtype=torch.float32) + 1.0
        bound.copy_(rows.reshape(1, -1, 1, 1).expand(bound.shape).to(bound.dtype))

    monkeypatch.setattr(module, "_launch_program", launch)
    batch, seqlen = (1, 8) if varlen else (2, 50)

    def inputs(seqlen):
        x = (torch.arange(batch * seqlen * 2 * 64) % 251).reshape(batch, seqlen, 2, 64)
        B = (torch.arange(batch * seqlen * 128) % 241).reshape(batch, seqlen, 1, 128)
        return (
            x.to(torch.bfloat16),
            torch.zeros((batch, seqlen, 2), dtype=torch.float32),
            torch.zeros((2,), dtype=torch.float32),
            B.to(torch.bfloat16),
            -B.to(torch.bfloat16),
        )

    def kwargs(seqlen):
        if not varlen:
            return {}
        return {
            "seq_idx": torch.zeros((1, seqlen), dtype=torch.int32),
            "chunk_indices": torch.tensor([0], dtype=torch.int32),
            "chunk_offsets": torch.tensor([0], dtype=torch.int32),
            "num_seqs": 1,
        }

    x, dt, A, B, C = inputs(seqlen)
    out, final = runner.run(x, dt, A, B, C, **kwargs(seqlen))

    assert tuple(out.shape) == (batch, seqlen, 2, 64)
    assert tuple(final.shape) == (batch if not varlen else 1, 2, 64, 128)
    ((_, launch_args),) = calls
    preprocess, main = launch_args["preprocess"], launch_args["main"]
    # The caller's tensors are bound directly: no runner-owned pad buffers,
    # no copy-back.
    assert main["x_map"] is x and main["b_map"] is B and main["c_map"] is C
    assert main["out_map"] is out and main["out_native"] is out
    assert float(out[0, -1, 0, 0]) == float(seqlen)
    # Logical extents are the caller's.
    assert main["seqlen"] == seqlen and main["nchunks"] == 1
    assert main["batch"] == batch and main["num_logical_chunks"] == 1
    assert preprocess["seqlen"] == seqlen and preprocess["num_segments"] == batch
    if not varlen:
        assert preprocess["segment_lengths"].tolist() == [seqlen] * batch
    assert main["dt"].shape == (batch, seqlen, 2)

    # A second call of the same extent binds its own tensors.
    x2, dt2, A2, B2, C2 = inputs(seqlen)
    out2, _ = runner.run(x2, dt2, A2, B2, C2, **kwargs(seqlen))
    main2 = calls[-1][1]["main"]
    assert main2["x_map"] is x2 and main2["out_map"] is out2 and out2 is not out

    # One full chunk binds the same way.
    x3, dt3, A3, B3, C3 = inputs(128)
    out3, _ = runner.run(x3, dt3, A3, B3, C3, **kwargs(128))
    main3 = calls[-1][1]["main"]
    assert main3["x_map"] is x3 and main3["b_map"] is B3 and main3["c_map"] is C3
    assert main3["out_map"] is out3 and main3["out_native"] is out3


def test_source_direct_preprocess_and_sequence_argument_order():
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    sentinels = {
        name: object()
        for name in (
            "dt",
            "A",
            "dt_bias",
            "starts",
            "lengths",
            "chunk_indices",
            "chunk_offsets",
            "delta",
            "cumsum",
            "seq_idx_i32",
            "seq_idx_i64",
            "seq_chunk_cumsum",
            "cu_seqlens",
            "checkpoint_token_indices",
            "preprocess_status",
        )
    }
    preprocess, preprocess_grid = module._direct_preprocess_inputs(
        dt=sentinels["dt"],
        A=sentinels["A"],
        dt_bias=sentinels["dt_bias"],
        segment_starts=sentinels["starts"],
        segment_lengths=sentinels["lengths"],
        chunk_indices=sentinels["chunk_indices"],
        chunk_offsets=sentinels["chunk_offsets"],
        delta=sentinels["delta"],
        cumsum=sentinels["cumsum"],
        num_segments=3,
        nheads=128,
        seqlen=256,
        mode_varlen=True,
        dt_softplus=False,
        dt_limit=(0.0, float("inf")),
        tiles_per_block=4,
        seq_idx_i32=sentinels["seq_idx_i32"],
        seq_idx_i64=sentinels["seq_idx_i64"],
        seq_idx_int64=True,
        seq_chunk_cumsum=sentinels["seq_chunk_cumsum"],
        num_sequences=2,
        write_seq_chunk_cumsum=True,
        cu_seqlens=sentinels["cu_seqlens"],
        checkpoint_token_indices=sentinels["checkpoint_token_indices"],
        metadata_from_cu_seqlens=False,
        checkpoint_state_count=0,
        preprocess_status=sentinels["preprocess_status"],
    )
    main = {name: object() for name in module._MAIN_ARGS}

    bound = module._sequence_arguments(
        module._MAIN_ARGS,
        preprocess,
        preprocess_grid,
        main,
        (148, 1, 1),
        cuda_stream=0x1234,
    )

    assert preprocess["chunk_indices"] is sentinels["chunk_indices"]
    assert preprocess["chunk_offsets"] is sentinels["chunk_offsets"]
    assert preprocess["direct_varlen_metadata"] == 1
    assert preprocess["dt_softplus"] == 0
    assert preprocess["seq_idx_i32"] is sentinels["seq_idx_i32"]
    assert preprocess["seq_idx_i64"] is sentinels["seq_idx_i64"]
    assert preprocess["seq_idx_int64"] == 1
    assert preprocess["seq_chunk_cumsum"] is sentinels["seq_chunk_cumsum"]
    assert preprocess["num_sequences"] == 2
    assert preprocess["write_seq_chunk_cumsum"] == 1
    assert preprocess["cu_seqlens"] is sentinels["cu_seqlens"]
    assert (
        preprocess["checkpoint_token_indices"] is sentinels["checkpoint_token_indices"]
    )
    assert preprocess["metadata_from_cu_seqlens"] == 0
    assert preprocess["checkpoint_state_count"] == 0
    assert preprocess["preprocess_status"] is sentinels["preprocess_status"]
    # 3 segments x 128 heads = 384 (segment, head) tiles, four per CTA
    assert preprocess_grid == (96, 1, 1)
    assert set(preprocess) == set(module._PREPROCESS_ARGS)
    # CAKE-990 appended ``preprocess_status`` last; CAKE-934 item 2 inserts the
    # cu_seqlens derivation inputs before it (the status word stays last).
    assert module._PREPROCESS_ARGS[-11:] == (
        "seq_idx_i32",
        "seq_idx_i64",
        "seq_idx_int64",
        "seq_chunk_cumsum",
        "num_sequences",
        "write_seq_chunk_cumsum",
        "cu_seqlens",
        "checkpoint_token_indices",
        "metadata_from_cu_seqlens",
        "checkpoint_state_count",
        "preprocess_status",
    )
    assert bound == (
        *(preprocess[name] for name in module._PREPROCESS_ARGS),
        *preprocess_grid,
        *(main[name] for name in module._MAIN_ARGS),
        148,
        1,
        1,
        0x1234,
    )
    # 44 main values: the 41 round-5 arguments plus the three fused-only inputs
    # (round-6 exact signature; both exact-scan families bind them).
    assert len(bound) == 27 + 3 + 44 + 3 + 1

    assert module._persistent_grid_size(total_work=256, sm_count=148) == 128
    assert module._persistent_grid_size(total_work=384, sm_count=148) == 128
    assert module._persistent_grid_size(total_work=129, sm_count=148) == 129


def _host_prepare_arguments(template, stage):
    """``arg_<name>`` parameters of ``stage_<stage>::Prepare`` in the generated
    host template, in declaration order."""

    namespace = template.index(f"namespace stage_{stage} {{")
    start = template.index("inline void Prepare(PreparedLaunch& prepared, ", namespace)
    return tuple(
        re.findall(r"\barg_(\w+)", template[start : template.index(")", start)])
    )


def _host_run_arguments(template, stage):
    """``<stage>_arg_<name>`` parameters of the host ``Run`` entry, in order."""

    start = template.index("\nvoid Run(")
    signature = template[start : template.index(")", start)]
    return tuple(re.findall(rf"\b{stage}_arg_(\w+)", signature))


def test_source_host_template_binds_stage_arguments_in_loader_order():
    """The generated host launcher's positional ABI (``Prepare`` of both
    stages and the ``Run`` entry) must list exactly the loader's
    ``_PREPROCESS_ARGS`` / ``_MAIN_ARGS`` in order; ``preprocess_status`` is
    the last preprocess tensor (CAKE-990)."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    template = (module._source_dir() / module._HOST_TEMPLATE).read_text(
        encoding="utf-8"
    )

    # The retained start two-kernel template binds the 41 round-5 arguments
    # and always runs its preprocess stage.
    assert _host_prepare_arguments(template, "preprocess") == module._PREPROCESS_ARGS
    assert _host_prepare_arguments(template, "main") == module._MAIN_ARGS_START
    assert _host_run_arguments(template, "preprocess") == module._PREPROCESS_ARGS
    assert _host_run_arguments(template, "main") == module._MAIN_ARGS_START
    assert "if (preprocess_grid_x != 0)" not in template
    # The fused template serves the exactfused, exactone and exactpre
    # families: all bind the 41 start arguments followed by the three inputs
    # the fused kernel takes over from the preprocess.  Its preprocess stage
    # runs when the grid is nonzero; the loader zeroes that grid only for the
    # exactfused / exactone programs, so the exactpre programs always run it.
    fused_template = (
        module._source_dir() / module._EXACTFUSED_HOST_TEMPLATE
    ).read_text(encoding="utf-8")
    assert (
        _host_prepare_arguments(fused_template, "preprocess") == module._PREPROCESS_ARGS
    )
    assert _host_prepare_arguments(fused_template, "main") == module._MAIN_ARGS
    assert _host_run_arguments(fused_template, "preprocess") == module._PREPROCESS_ARGS
    assert _host_run_arguments(fused_template, "main") == module._MAIN_ARGS
    assert len(module._MAIN_ARGS) == 44 and len(module._MAIN_ARGS_START) == 41
    assert module._MAIN_ARGS[:41] == module._MAIN_ARGS_START
    assert module._MAIN_ARGS[41:] == (
        "cu_seqlens",
        "metadata_from_cu_seqlens",
        "preprocess_status",
    )
    assert "if (preprocess_grid_x != 0)" in fused_template
    assert module._FAMILY_HOST["exactpre"][0] == module._EXACTFUSED_HOST_TEMPLATE
    assert module._PREPROCESS_ARGS[-1] == "preprocess_status"
    assert len(module._PREPROCESS_ARGS) == 27
    # The exactmeta template is the fused template with the 12-argument
    # packed-varlen metadata stage (one thread per segment, no dynamic shared
    # memory), always launched, ahead of the same 44-argument main kernel.
    metadata_template = (
        module._source_dir() / module._EXACTMETA_HOST_TEMPLATE
    ).read_text(encoding="utf-8")
    assert (
        _host_prepare_arguments(metadata_template, "preprocess")
        == module._METADATA_ARGS
    )
    assert _host_prepare_arguments(metadata_template, "main") == module._MAIN_ARGS
    assert _host_run_arguments(metadata_template, "preprocess") == module._METADATA_ARGS
    assert _host_run_arguments(metadata_template, "main") == module._MAIN_ARGS
    assert "if (preprocess_grid_x != 0)" not in metadata_template
    assert len(module._METADATA_ARGS) == 12
    assert module._METADATA_ARGS[-1] == "preprocess_status"
    metadata_stage = metadata_template[
        metadata_template.index(
            "namespace stage_preprocess {"
        ) : metadata_template.index("namespace stage_main {")
    ]
    assert "void* kargs[12] = {};" in metadata_stage
    assert "prepared.kargs[11] = &prepared.p_preprocess_status;" in metadata_stage
    assert "prepared.kargs[12]" not in metadata_stage
    assert "prepared.block, stream, 0u)" in metadata_stage
    assert (
        'check_dtype(arg_seq_idx_i64, DLDataType{kDLInt, 64, 1}, "seq_idx_i64");'
        in metadata_stage
    )
    assert (
        'check_dtype(arg_sequence_offsets, DLDataType{kDLInt, 32, 1}, "sequence_offsets");'
        in metadata_stage
    )
    # Beyond its preprocess stage and Run signature the template is the fused
    # template byte for byte.
    assert (
        metadata_template[metadata_template.index("namespace stage_main {") :].split(
            "\nvoid Run("
        )[0]
        == fused_template[fused_template.index("namespace stage_main {") :].split(
            "\nvoid Run("
        )[0]
    )
    assert (
        'check_dtype(arg_cu_seqlens, DLDataType{kDLInt, 32, 1}, "cu_seqlens");'
        in template
    )
    assert (
        "check_dtype(arg_checkpoint_token_indices, DLDataType{kDLInt, 32, 1}, "
        '"checkpoint_token_indices");'
    ) in template
    assert (
        'check_dtype(arg_preprocess_status, DLDataType{kDLInt, 32, 1}, "preprocess_status");'
        in template
    )
    preprocess_stage = template[
        template.index("namespace stage_preprocess {") : template.index(
            "namespace stage_main {"
        )
    ]
    assert "void* kargs[27] = {};" in preprocess_stage
    assert "prepared.kargs[26] = &prepared.p_preprocess_status;" in preprocess_stage
    assert "prepared.kargs[27]" not in preprocess_stage


def test_source_runner_binds_preprocess_status_without_gpu(monkeypatch):
    """The runner owns zeroed int32 status words per device -- two, then one
    per pair of SMs and one per 32 pairs (none on this one-SM device) -- binds
    them as ``preprocess_status`` on every call and exposes word 0 through the
    synchronizing debug accessor ``seq_idx_status``; word 1 is the item
    counter of the sm_107a claim programs, which every launch leaves 0."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    runner = _cpu_forwarding_runner(module, monkeypatch, calls, has_varlen=True)
    seqlen = 300
    x = torch.empty((1, seqlen, 1, 64), dtype=torch.bfloat16)
    dt = torch.empty((1, seqlen, 1), dtype=torch.float32)
    A = torch.empty((1,), dtype=torch.float32)
    B = torch.empty((1, seqlen, 1, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)
    kwargs = {
        "seq_idx": torch.zeros((1, seqlen), dtype=torch.int32),
        "chunk_indices": torch.tensor([0, 1, 2], dtype=torch.int32),
        "chunk_offsets": torch.tensor([0, 0, 0], dtype=torch.int32),
        "num_seqs": 1,
    }
    cpu = torch.device("cpu")

    runner.run(x, dt, A, B, C, **kwargs)
    runner.run(x, dt, A, B, C, **kwargs)

    (first, second) = (launch["preprocess"]["preprocess_status"] for _, launch in calls)
    assert first is second
    assert first.dtype == torch.int32 and tuple(first.shape) == (2,)
    assert first.device == cpu and first.tolist() == [0, 0]
    assert runner.seq_idx_status(device=cpu) == 0
    first[1] = 7
    assert runner.seq_idx_status(device=cpu) == 0
    first[0] = 1
    assert runner.seq_idx_status(device=cpu) == 1
    assert runner.seq_idx_status(reset=True, device=cpu) == 1
    assert runner.seq_idx_status(device=cpu) == 0
    assert first.tolist() == [0, 0]


def test_source_sequence_arguments_fail_closed_on_missing_values():
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    preprocess = {name: object() for name in module._PREPROCESS_ARGS}
    main = {name: object() for name in module._MAIN_ARGS}
    del main["checkpoint_state_count"]

    with pytest.raises(KeyError, match="checkpoint_state_count"):
        module._sequence_arguments(
            module._MAIN_ARGS, preprocess, (1, 1, 1), main, (1, 1, 1), 0
        )


def test_source_program_launch_orders_stage_arguments(monkeypatch):
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []

    class Generated:
        def run(self, *args):
            calls.append(args)

    monkeypatch.setattr(module, "_load_generated_program", lambda *_: Generated())
    preprocess = {name: f"pre:{name}" for name in module._PREPROCESS_ARGS}
    main = {name: f"main:{name}" for name in module._MAIN_ARGS}

    module._launch_program(
        "exactfused_bf16_varlen",
        "sm_103a",
        preprocess=preprocess,
        preprocess_grid=(32, 1, 1),
        main=main,
        main_grid=(148, 1, 1),
        cuda_stream=0x1234,
    )

    assert calls == [
        (
            *(f"pre:{name}" for name in module._PREPROCESS_ARGS),
            32,
            1,
            1,
            *(f"main:{name}" for name in module._MAIN_ARGS),
            148,
            1,
            1,
            0x1234,
        )
    ]
    # An exactmeta program orders its 12-argument metadata stage first.
    calls.clear()
    metadata = {name: f"meta:{name}" for name in module._METADATA_ARGS}
    module._launch_program(
        "exactmeta_bf16_varlen",
        "sm_103a",
        preprocess=metadata,
        preprocess_grid=(1, 1, 1),
        main=main,
        main_grid=(16, 1, 1),
        cuda_stream=0x1234,
    )
    assert calls == [
        (
            *(f"meta:{name}" for name in module._METADATA_ARGS),
            1,
            1,
            1,
            *(f"main:{name}" for name in module._MAIN_ARGS),
            16,
            1,
            1,
            0x1234,
        )
    ]


def test_source_sequence_arguments_take_the_metadata_stage_order():
    """``preprocess_args`` selects the preprocess stage's tuple: the metadata
    stage values are ordered by ``_METADATA_ARGS`` and a segment-preprocess
    dict fails closed on the first metadata name it lacks."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    metadata = {name: f"meta:{name}" for name in module._METADATA_ARGS}
    main = {name: f"main:{name}" for name in module._MAIN_ARGS}
    ordered = module._sequence_arguments(
        module._MAIN_ARGS,
        metadata,
        (1, 1, 1),
        main,
        (2, 1, 1),
        7,
        preprocess_args=module._METADATA_ARGS,
    )
    assert ordered == (
        *(f"meta:{name}" for name in module._METADATA_ARGS),
        1,
        1,
        1,
        *(f"main:{name}" for name in module._MAIN_ARGS),
        2,
        1,
        1,
        7,
    )
    assert set(module._METADATA_ARGS) - set(module._PREPROCESS_ARGS) == {
        "sequence_offsets"
    }
    with pytest.raises(KeyError, match="sequence_offsets"):
        module._sequence_arguments(
            module._MAIN_ARGS,
            {name: object() for name in module._PREPROCESS_ARGS},
            (1, 1, 1),
            main,
            (1, 1, 1),
            0,
            preprocess_args=module._METADATA_ARGS,
        )


_HOST_PLACEHOLDERS = (
    "CAKE_SSD_PROGRAM",
    "CAKE_SSD_PREPROCESS_MODULE",
    "CAKE_SSD_PREPROCESS_KERNEL",
    "CAKE_SSD_PREPROCESS_THREADS",
    "CAKE_SSD_MAIN_MODULE",
    "CAKE_SSD_MAIN_KERNEL",
    "CAKE_SSD_STATE_DTYPE_CODE",
    "CAKE_SSD_STATE_DTYPE_BITS",
    "CAKE_SSD_MAIN_SMEM_BYTES",
)


def test_source_program_table_names_shipped_sources():
    """Every program of the seven families binds shipped generated sources:
    its preprocess stage (the segment preprocess, one warp per tile of its
    block, or the packed-varlen metadata kernel of the ``exactmeta``
    programs), its family's host template and main dynamic SMEM, and a main
    kernel whose symbol is the module identity without its hash."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    source_dir = module._source_dir()

    # The ``_noring`` programs ship for the shapes the variant rule can
    # select: never for the two-kernel family (its items span at least two
    # chunks) and never for an f32 state on the varlen path.
    assert set(module._PROGRAMS) == {
        f"{family}_{state}_{mode}{variant}"
        for family, variants in (
            ("exact", ("", "_noring")),
            ("exactfused", ("", "_noring")),
            ("exactpre", ("", "_tail")),
            ("chunkpar", ("",)),
            ("chunkpar3", ("",)),
        )
        for variant in variants
        for state in ("bf16", "f16", "f32")
        for mode in ("batched", "varlen")
        if (variant, state, mode) != ("_noring", "f32", "varlen")
    } | {
        f"{family}_{state}_varlen{variant}"
        for family in ("exactmeta", "exactone")
        for variant in ("", "_noring")
        for state in ("bf16", "f16", "f32")
        if (variant, state) != ("_noring", "f32")
    } | {"exactpre_bf16_varlen_claim", "exactpre_f32_varlen_claim"} | {
        f"{family}_{state}_varlen_ibulk"
        for family in ("exactpre", "exactfused")
        for state in ("bf16", "f16")
    } | {
        "exactone_bf16_varlen_tablefirst",
        "exactone_bf16_varlen_tableroles",
        "exactone_f16_varlen_tablefirst",
        "exactone_f16_varlen_tableroles",
        "exactone_f32_varlen_tablefirst",
        "exactone_f32_varlen_tableroles",
    } | {f"exactpre_{state}_varlen_tailone" for state in ("bf16", "f16", "f32")}
    # The sm_107a-only programs (the claim programs, the two-kernel and fused
    # initial-state bulk programs, the two-kernel one-CTA pair programs and
    # the one-launch programs in the tablefirst / tableroles prologue forms):
    # their ``_SM107_MODULES`` entry is their sole source.
    assert {
        "exactpre_bf16_varlen_claim",
        "exactpre_f32_varlen_claim",
        "exactpre_bf16_varlen_ibulk",
        "exactpre_f16_varlen_ibulk",
        "exactpre_bf16_varlen_tailone",
        "exactpre_f16_varlen_tailone",
        "exactpre_f32_varlen_tailone",
        "exactfused_bf16_varlen_ibulk",
        "exactfused_f16_varlen_ibulk",
        "exactone_bf16_varlen_tablefirst",
        "exactone_bf16_varlen_tableroles",
        "exactone_f16_varlen_tablefirst",
        "exactone_f16_varlen_tableroles",
        "exactone_f32_varlen_tablefirst",
        "exactone_f32_varlen_tableroles",
    } == module._SM107_ONLY_PROGRAMS
    assert set(module._SCAN_MODULES) == {
        name for name in module._PROGRAMS if name.startswith("exact_")
    }
    # The exactmeta programs are the exactfused varlen main kernels (the same
    # device modules) behind the metadata stage.
    assert set(module._EXACTMETA_MODULES) == {
        name for name in module._PROGRAMS if name.startswith("exactmeta_")
    }
    assert {
        name.replace("exactfused_", "exactmeta_", 1): shared
        for name, shared in module._EXACTFUSED_MODULES.items()
        if "_varlen" in name
    } == module._EXACTMETA_MODULES
    # The exactone programs are the SM103 render of the same six fused varlen
    # entries: their own device modules, never the exactfused files (on
    # sm_107a too: the one-launch render beside the exactfused one).
    assert (
        set(module._EXACTONE_MODULES)
        == {name for name in module._PROGRAMS if name.startswith("exactone_")}
        - module._SM107_ONLY_PROGRAMS
    )
    assert {
        name.replace("exactone_", "exactfused_", 1) for name in module._EXACTONE_MODULES
    } == {name for name in module._EXACTFUSED_MODULES if "_varlen" in name}
    assert not set(module._EXACTONE_MODULES.values()) & set(
        module._EXACTFUSED_MODULES.values()
    )
    assert set(module._EXACTFUSED_MODULES) == {
        name
        for name in module._PROGRAMS
        if name.startswith("exactfused_") and name not in module._SM107_ONLY_PROGRAMS
    }
    assert (
        set(module._EXACTPRE_MODULES)
        == {name for name in module._PROGRAMS if name.startswith("exactpre_")}
        - module._SM107_ONLY_PROGRAMS
    )
    assert set(module._CHUNKPAR_MODULES) == {
        name for name in module._PROGRAMS if name.startswith("chunkpar_")
    }
    assert set(module._CHUNKPAR3_MODULES) == {
        name for name in module._PROGRAMS if name.startswith("chunkpar3_")
    }
    # The sm_107a (Rubin) module table: every program whose Rubin render was
    # regenerated with its own geometry -- the fused, one-launch and
    # two-kernel exact scans (with their ``_noring`` and ``_tail`` variants),
    # the look-back and the three-phase chunk-parallel programs -- each entry
    # the fold module's stem under its own identity (the export's content
    # hash).  Only the retained start ``exact_*`` programs and the SM100-only
    # ``exactmeta_*`` programs build their fold module on sm_107a.  The
    # ``exactone_*`` entry of a suffix is a source of its own (the one-launch
    # render beside the SM100-form render), as on SM103.
    assert (
        set(module._SM107_MODULES)
        == {
            f"{family}_{state}_{mode}{variant}"
            for family, mode, variants in (
                ("exactfused", "batched", ("", "_noring")),
                ("exactfused", "varlen", ("", "_noring")),
                ("exactone", "varlen", ("", "_noring")),
                ("exactpre", "batched", ("", "_tail")),
                ("exactpre", "varlen", ("", "_tail")),
                ("chunkpar", "batched", ("",)),
                ("chunkpar", "varlen", ("",)),
                ("chunkpar3", "batched", ("",)),
                ("chunkpar3", "varlen", ("",)),
            )
            for variant in variants
            for state in ("bf16", "f16", "f32")
            if (variant, state, mode) != ("_noring", "f32", "varlen")
        }
        | module._SM107_ONLY_PROGRAMS
    )
    assert len(module._SM107_MODULES) == 55
    assert set(module._SM107_MODULES) <= set(module._PROGRAMS)
    assert set(module._PROGRAMS) - set(module._SM107_MODULES) == {
        name for name in module._PROGRAMS if name.startswith(("exact_", "exactmeta_"))
    }
    for name, own in module._SM107_MODULES.items():
        if name.startswith("exactone_") and name not in module._SM107_ONLY_PROGRAMS:
            twin = module._SM107_MODULES[name.replace("exactone_", "exactfused_", 1)]
            assert twin != own, (name, twin)
    assert len(set(module._SM107_MODULES.values())) == len(module._SM107_MODULES)
    for name, identity in module._SM107_MODULES.items():
        stem, suffix = identity.rsplit("_", 1)
        assert stem == module._PROGRAMS[name].main.module.rsplit("_", 1)[0], name
        assert re.fullmatch(r"[0-9a-f]+", suffix), name
    assert (
        module._SEGMENT_PREPROCESS.threads
        == 32 * module._SEGMENT_PREPROCESS_TILES_PER_BLOCK
    )
    assert module._PACKED_METADATA.threads == 256
    assert module._PACKED_METADATA.kernel == (
        "kernel_factorized_persistent_packed_varlen_metadata"
    )
    family_host = {
        "exact": (
            module._HOST_TEMPLATE,
            module._MAIN_ARGS_START,
            module._EXACT_SMEM_BYTES,
        ),
        "exactfused": (
            module._EXACTFUSED_HOST_TEMPLATE,
            module._MAIN_ARGS,
            module._EXACTFUSED_SMEM_BYTES,
        ),
        "exactmeta": (
            module._EXACTMETA_HOST_TEMPLATE,
            module._MAIN_ARGS,
            module._EXACTFUSED_SMEM_BYTES,
        ),
        "exactone": (
            module._EXACTFUSED_HOST_TEMPLATE,
            module._MAIN_ARGS,
            module._EXACTONE_SMEM_BYTES,
        ),
        "chunkpar": (
            module._CHUNKPAR_HOST_TEMPLATE,
            module._MAIN_ARGS_CHUNKPAR,
            module._CHUNKPAR_SMEM_BYTES,
        ),
        "exactpre": (
            module._EXACTFUSED_HOST_TEMPLATE,
            module._MAIN_ARGS,
            module._EXACTPRE_SMEM_BYTES,
        ),
        "chunkpar3": (
            module._CHUNKPAR3_HOST_TEMPLATE,
            module._MAIN_ARGS_CHUNKPAR3,
            module._CHUNKPAR3_SMEM_BYTES,
        ),
    }
    assert family_host == module._FAMILY_HOST
    templates = {
        family: (source_dir / template).read_text(encoding="utf-8")
        for family, (template, _, _) in family_host.items()
    }
    device_sources = set()
    for name, program in module._PROGRAMS.items():
        family, state_key, mode, *variant = name.split("_")
        template_path, main_args, smem_bytes = family_host[family]
        if variant:
            (variant_name,) = variant
            assert variant_name in (
                "noring",
                "tail",
                "tailone",
                "claim",
                "ibulk",
                "tablefirst",
                "tableroles",
            )
            assert family in (
                "exact",
                "exactfused",
                "exactmeta",
                "exactone",
                "exactpre",
            )
            if variant_name in ("tail", "tailone", "claim"):
                assert family == "exactpre"
            if variant_name == "tailone":
                assert mode == "varlen"
            if variant_name == "ibulk":
                assert family in ("exactpre", "exactfused")
            if variant_name == "claim":
                assert (state_key, mode) in (("bf16", "varlen"), ("f32", "varlen"))
            if variant_name == "ibulk":
                assert (state_key, mode) in (("bf16", "varlen"), ("f16", "varlen"))
            if variant_name in ("tablefirst", "tableroles"):
                assert family == "exactone" and mode == "varlen"
            sm107_only = ("claim", "ibulk", "tailone", "tablefirst", "tableroles")
            assert (variant_name in sm107_only) == (name in module._SM107_ONLY_PROGRAMS)
            smem_bytes = module._VARIANT_SMEM_BYTES[(family, variant_name)]
            assert (
                smem_bytes
                == {
                    ("exact", "noring"): module._EXACT_NORING_SMEM_BYTES,
                    ("exactfused", "noring"): module._EXACTFUSED_NORING_SMEM_BYTES,
                    ("exactmeta", "noring"): module._EXACTFUSED_NORING_SMEM_BYTES,
                    ("exactone", "tablefirst"): module._EXACTONE_SMEM_BYTES,
                    ("exactone", "tableroles"): module._EXACTONE_SMEM_BYTES,
                    ("exactone", "noring"): module._EXACTONE_NORING_SMEM_BYTES,
                    ("exactpre", "tail"): module._EXACTPRE_TAIL_SMEM_BYTES,
                    ("exactpre", "claim"): module._EXACTPRE_CLAIM_SMEM_BYTES,
                    ("exactpre", "ibulk"): module._EXACTPRE_IBULK_SMEM_BYTES,
                    ("exactpre", "tailone"): module._EXACTPRE_TAILONE_SMEM_BYTES,
                    ("exactfused", "ibulk"): module._EXACTFUSED_IBULK_SMEM_BYTES,
                }[(family, variant_name)]
            )
        assert program.variant == (variant[0] if variant else "")
        assert program.cluster_dims == module._VARIANT_CLUSTER_DIMS.get(
            (family, program.variant), (1, 1, 1)
        )
        assert program.family == family
        assert program.host_template == template_path
        assert program.main_args == main_args
        assert program.main_smem_bytes == smem_bytes
        assert (program.state_dtype_code, program.state_dtype_bits) == (
            module._STATE_DTYPE_CODES[state_key]
        )
        if family == "exactmeta":
            assert program.preprocess is module._PACKED_METADATA
            assert program.preprocess_args == module._METADATA_ARGS
        else:
            assert program.preprocess is module._SEGMENT_PREPROCESS
            assert program.preprocess_args == module._PREPROCESS_ARGS
        assert program.main.module.startswith(
            program.main.kernel.removeprefix("kernel_") + "_"
        )
        # The symbol's trailing identity is
        # ``<state>_<mode>[_precomputed][_<variant>]`` (``_precomputed``: the
        # two-kernel exactpre programs).
        assert program.main.kernel.endswith(
            f"_{state_key}_{mode}"
            + ("_precomputed" if family == "exactpre" else "")
            + "".join(f"_{part}" for part in variant)
        )
        assert program.preprocess.threads > 0 and program.main.threads == 512
        # The integer-only metadata kernel is built with the loader's
        # ``--use_fast_math`` (its launcher's build option); the segment
        # preprocess without it.
        assert program.preprocess.fast_math == (family == "exactmeta")
        assert program.main.fast_math
        for kernel in program.kernels:
            source = source_dir / module._DEVICE_DIR / kernel.source
            assert source.is_file(), source
            assert re.search(
                rf"\b{kernel.kernel}\(", source.read_text(encoding="utf-8")
            )
            device_sources.add(source)
        # The sm_107a render of the program (``_SM107_MODULES``): the same
        # kernel symbol, in its own device source, behind the sm_107a render
        # of the segment preprocess (the ``exactmeta`` programs keep the
        # shared metadata kernel); a program without an entry builds its fold
        # main source on sm_107a too.
        rubin = module._arch_program(name, "sm_107a")
        assert rubin.main.kernel == program.main.kernel
        if program.preprocess is module._SEGMENT_PREPROCESS:
            assert rubin.preprocess is _sm107_segment_preprocess(module, name)
            assert rubin.preprocess.module != program.preprocess.module
            source = source_dir / module._DEVICE_DIR / rubin.preprocess.source
            assert source.is_file(), source
            assert re.search(
                rf"\b{rubin.preprocess.kernel}\(",
                source.read_text(encoding="utf-8"),
            )
            device_sources.add(source)
        else:
            assert rubin.preprocess is program.preprocess
        if name not in module._SM107_MODULES:
            assert rubin.main is program.main
        elif name in module._SM107_ONLY_PROGRAMS:
            # no fold source: the program binds its sm_107a render
            assert rubin.main is program.main
            assert program.main.module == module._SM107_MODULES[name]
        else:
            assert rubin.main.module == module._SM107_MODULES[name]
            assert rubin.main.module != program.main.module
            source = source_dir / module._DEVICE_DIR / rubin.main.source
            assert source.is_file(), source
            assert re.search(
                rf"\b{rubin.main.kernel}\(", source.read_text(encoding="utf-8")
            )
            device_sources.add(source)
        template = templates[family]
        rendered = module._render_host_source(template, name, program)
        assert all(placeholder in template for placeholder in _HOST_PLACEHOLDERS)
        assert not any(placeholder in rendered for placeholder in _HOST_PLACEHOLDERS)
        assert f"TVM_FFI_EMBED_CUBIN({program.preprocess.module});" in rendered
        assert f"TVM_FFI_EMBED_CUBIN({program.main.module});" in rendered
        assert f'"{program.main.kernel}"' in rendered
        assert f"namespace cake_mamba_ssd_combined_host_{name} {{" in rendered
        # The main launch is a plain back-to-back launch after the preprocess
        # (no programmatic dependent launch); its dynamic SMEM is the family
        # literal.
        assert f"stream, {program.main_smem_bytes}u)" in rendered
        assert "LaunchEx(" not in rendered
        assert "PROGRAMMATIC_STREAM_SERIALIZATION" not in rendered
        # The regenerated host shim brace-initialises the DLDataType of the
        # three state tensors from the two state placeholders; delta checks
        # are FP16 for every program (D1) and not factored.
        for state_tensor in ("initial_states", "final_states", "checkpoint_states"):
            state_check = (
                f"check_dtype(arg_{state_tensor}, DLDataType{{"
                f"{program.state_dtype_code}, {program.state_dtype_bits}, 1}}, "
                f'"{state_tensor}");'
            )
            assert state_check in rendered, state_check
        assert "DLDataType{kDLFloat, 16, 1}" in rendered
    # One shared source per physical kernel: eleven retained start ``exact``
    # main sources (six ring + five ``_noring``), eleven fused ``exactfused``
    # main sources (the five varlen ones also serving ``exactmeta``), five
    # one-launch ``exactone`` main sources (the SM103 render of the fused
    # varlen entries), twelve two-kernel ``exactpre`` main sources (six ring,
    # six ``_tail``), six look-back (``chunkpar``) and six three-phase
    # (``chunkpar3``) chunk-parallel main sources plus the segment preprocess
    # and the packed-varlen metadata kernel; plus one sm_107a (Rubin) render
    # per ``_SM107_MODULES`` identity (a program without an entry builds its
    # fold source on sm_107a, whose descriptor mask is an ``__CUDA_ARCH__``
    # branch) and the two sm_107a renders of the segment preprocess (with and
    # without the early dependent-launch trigger), no other architecture copies
    # and no orphans.
    rubin_sources = set(module._SM107_MODULES.values()) | {
        module._SM107_SEGMENT_PREPROCESS_MODULE,
        module._SM107_SEGMENT_PREPROCESS_TRIGGER_MODULE,
    }
    assert len(device_sources) == 53 + len(rubin_sources)
    assert sorted(
        path.name for path in (source_dir / module._DEVICE_DIR).glob("*.cu")
    ) == sorted(path.name for path in device_sources)


def test_active_source_package_declares_cuda_half_types_explicitly():
    """Every program carries ``delta`` in FP16 (the state may also be FP16),
    so each shipped device source must declare the CUDA half types itself."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    source_root = module._source_dir() / module._DEVICE_DIR
    sources = {
        kernel.source
        for program in module._PROGRAMS.values()
        for kernel in program.kernels
    } | {f"{identity}.cu" for identity in module._SM107_MODULES.values()}
    sources.add(module._SM107_SEGMENT_PREPROCESS.source)
    sources.add(module._SM107_SEGMENT_PREPROCESS_TRIGGER.source)
    assert sources
    for name in sorted(sources):
        source = (source_root / name).read_text(encoding="utf-8")
        assert source.count("#include <cuda_fp16.h>") == 1, name
        assert "#include <cuda_bf16.h>\n#include <cuda_fp16.h>\n" in source, name


def test_source_program_loader_builds_one_module_per_arch(monkeypatch, tmp_path):
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    program = module._PROGRAMS["exact_bf16_varlen"]
    device_dir = tmp_path / module._DEVICE_DIR
    device_dir.mkdir(parents=True)
    for kernel in program.kernels:
        (device_dir / kernel.source).write_text(
            f"{kernel.kernel} source\n", encoding="utf-8"
        )
    host = tmp_path / module._HOST_TEMPLATE
    host.parent.mkdir(parents=True)
    host.write_text(
        "namespace host_CAKE_SSD_PROGRAM {}\n"
        "TVM_FFI_EMBED_CUBIN(CAKE_SSD_PREPROCESS_MODULE);\n"
        "TVM_FFI_EMBED_CUBIN(CAKE_SSD_MAIN_MODULE);\n"
        "CAKE_SSD_PREPROCESS_KERNEL CAKE_SSD_PREPROCESS_THREADS CAKE_SSD_MAIN_KERNEL "
        "CAKE_SSD_STATE_DTYPE_CODE CAKE_SSD_STATE_DTYPE_BITS CAKE_SSD_MAIN_SMEM_BYTES\n",
        encoding="utf-8",
    )
    nvcc = tmp_path / "cuda" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        output = module.Path(command[-1])
        source_path = module.Path(command[-3])
        output.write_bytes(source_path.read_bytes())
        return SimpleNamespace(returncode=0, stderr="")

    loaded = object()
    load_calls = []

    def load_inline(*args, **kwargs):
        load_calls.append((args, kwargs))
        return loaded

    monkeypatch.setattr(module, "_source_dir", lambda: tmp_path)
    monkeypatch.setattr(module, "_nvcc", lambda: nvcc)
    monkeypatch.setattr(module.jit_env, "FLASHINFER_JIT_DIR", tmp_path / "jit")
    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "cpp", SimpleNamespace(load_inline=load_inline))
    module._load_generated_program.cache_clear()

    actual = module._load_generated_program("exact_bf16_varlen", "sm_103a")
    again = module._load_generated_program("exact_bf16_varlen", "sm_103a")
    module._load_generated_program.cache_clear()

    assert actual is loaded and again is loaded
    assert len(calls) == 2
    assert all("-arch=sm_103a" in command for command, _ in calls)
    assert "--use_fast_math" not in calls[0][0]
    assert "--use_fast_math" in calls[1][0]
    assert len(load_calls) == 1
    rendered = load_calls[0][1]["cpp_sources"]
    assert "CAKE_SSD_" not in rendered
    assert "namespace host_exact_bf16_varlen {}" in rendered
    assert f"TVM_FFI_EMBED_CUBIN({program.preprocess.module});" in rendered
    assert (
        f"{program.preprocess.kernel} {program.preprocess.threads} "
        f"{program.main.kernel} 4 16 {program.main_smem_bytes}" in rendered
    )
    assert set(load_calls[0][1]["embed_cubin"]) == {
        program.preprocess.module,
        program.main.module,
    }
    assert load_calls[0][0][0].startswith("cake_mamba_ssd_exact_bf16_varlen_sm_103a_")


# ---------------------------------------------------------------------------
# Chunk-parallel program families: the look-back program (``chunkpar_*``, the
# exact scan's arithmetic over item-major tile ranges with a decoupled
# look-back) and the three-phase program (``chunkpar3_*``, the same arithmetic
# in grid-barrier-separated phases, bitwise the round-5 exact scan), selected
# per call by the calibrated cost rule and the measured-region family rule
# mirrored from the Cake look-back seed module.

_CHUNK_PARALLEL_ENV = "FLASHINFER_CAKE_SSD_CHUNK_PARALLEL"
_SELECTION_SM_COUNT = 148
_SELECTION_CAPABILITIES = ((10, 0), (10, 3))  # the 148-SM rule cases below
_COST_MODEL_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
_RUBIN_SM_COUNT = 212
_CHUNK_PARALLEL_WORKSPACE_ARGS = (
    "h_map",
    "s_work",
    "agg_work",
    "h_words",
    "grid_barrier",
)
_CHUNKPAR3_WORKSPACE_ARGS = ("h_map", "s_work", "h_words", "grid_barrier")


@pytest.fixture(autouse=True)
def _without_chunk_parallel_override(monkeypatch):
    """Every test starts with ``FLASHINFER_CAKE_SSD_CHUNK_PARALLEL`` unset
    (``auto``): the runner reads it on every call, so a value inherited from
    the invoking shell would route every call through the other family or
    make every call raise.  Tests of the override set the variable through
    their own ``monkeypatch`` afterwards."""

    monkeypatch.delenv(_CHUNK_PARALLEL_ENV, raising=False)


def _batched_selection(
    batch,
    nheads,
    nchunks,
    *,
    sm_count=_SELECTION_SM_COUNT,
    capability=(10, 0),
    mode="auto",
):
    """``chunk_parallel_selected`` of a batched call: ``batch`` sequences of
    ``nchunks`` 128-token chunks (one segment per chunk)."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    return module.chunk_parallel_selected(
        nheads=nheads,
        num_sequences=batch,
        num_segments=batch * nchunks,
        nchunks=nchunks,
        mode_varlen=False,
        sm_count=sm_count,
        capability=capability,
        mode=mode,
    )


def _family(
    batch,
    nheads,
    nchunks,
    *,
    varlen=False,
    sm_count=_SELECTION_SM_COUNT,
    capability=(10, 0),
    mode="auto",
):
    """``chunk_parallel_family`` of ``batch`` sequences of ``nchunks`` chunks
    each (one segment per chunk; the varlen form passes the same counts)."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    return module.chunk_parallel_family(
        nheads=nheads,
        num_sequences=batch,
        num_segments=batch * nchunks,
        nchunks=nchunks,
        mode_varlen=varlen,
        sm_count=sm_count,
        capability=capability,
        mode=mode,
    )


@pytest.mark.parametrize("capability", _SELECTION_CAPABILITIES, ids=("sm100", "sm103"))
@pytest.mark.parametrize(
    "batch,nheads,nchunks,expected",
    (
        (1, 8, 16, True),
        (1, 8, 8, True),
        (1, 128, 32, False),
        (1, 8, 256, True),
        (4, 8, 64, True),
        (1, 32, 16, True),
    ),
    ids=("1x8x16", "1x8x8", "1x128x32", "1x8x256", "4x8x64", "1x32x16"),
)
def test_source_chunk_parallel_selection_rule(
    capability, batch, nheads, nchunks, expected
):
    """The calibrated family rule at 148 SMs on both measured capabilities.
    Chunk-parallel needs fewer (sequence, head) items than SMs, a workspace
    within the cap and a predicted main-kernel time that beats the exact
    scan's by the selection margin: 1 x 2048 tokens x 8 heads (16 chunks,
    128 tiles), the 32768-token few-head prefill, 4 x 8192 x 8 and -- since
    the round-4 re-fit of the constants -- 1 x 2048 x 32 (512 tiles, measured
    chunk-parallel / exact 0.91) qualify; since the round-6 look-back
    constants (cp_fixed 23.3 / 21.6 us) 1 x 1024 x 8 (8 chunks, 64 tiles;
    measured look-back / exact 0.64 on B200, 0.63 on GB300) qualifies too,
    while 128 heads pay for 4096 tiles."""

    selected = _batched_selection(batch, nheads, nchunks, capability=capability)
    assert selected is expected


@pytest.mark.parametrize(
    "batch,nheads,nchunks,expected",
    (
        (1, 8, 16, True),  # 128 tiles inside the first wave: 20.0 vs 33.5 us
        (1, 8, 8, False),  # 8 chunks: the exact scan (18.8 us) beats cp_fixed (20.0)
        (1, 128, 32, False),  # 128 items x 32 chunks: 4096 tiles, exact 66.9 us
        (1, 8, 256, True),  # the 32768-token calibration row: 75.4 vs 475.4 us
        (4, 8, 64, True),  # 32 items x 64 chunks: 75.4 vs 121.9 us
        (1, 32, 16, True),  # 512 tiles: three-phase 29.0 x 1.05 vs 33.5 us
    ),
    ids=("1x8x16", "1x8x8", "1x128x32", "1x8x256", "4x8x64", "1x32x16"),
)
def test_source_chunk_parallel_selection_rule_rubin(batch, nheads, nchunks, expected):
    """The same rule with the Rubin (10, 7) constants at 212 SMs: the
    cheaper exact scan (serial_fixed 4.07 us, 1.84 us per chunk) keeps the
    8-chunk 1 x 1024 x 8 call that the Blackwell constants hand to
    chunk-parallel; the other cases follow the Blackwell verdicts."""

    selected = _batched_selection(
        batch, nheads, nchunks, capability=(10, 7), sm_count=_RUBIN_SM_COUNT
    )
    assert selected is expected


@pytest.mark.parametrize("capability", _SELECTION_CAPABILITIES, ids=("sm100", "sm103"))
@pytest.mark.parametrize(
    "batch,nheads,nchunks,varlen,expected",
    (
        (1, 8, 16, False, "chunkpar3"),  # perf_batched_1x16: 8 items, 128 tiles
        (1, 8, 32, False, "chunkpar3"),  # 8 items, 256 tiles
        (1, 8, 64, False, "chunkpar"),  # perf_batched_1x64: 512 tiles (tie, look-back)
        (1, 8, 256, False, "chunkpar"),  # perf_batched_1x256: 2048 tiles
        (4, 8, 16, False, "chunkpar"),  # perf_batched_4x16: 32 items x 16 chunks
        (4, 8, 64, False, "chunkpar3"),  # perf_batched_4x64: 32 items x 64 chunks
        (1, 32, 16, False, "chunkpar"),  # cp_b1h32c16: 32 items x 16 chunks, 512 tiles
        (4, 8, 16, True, "chunkpar"),  # varlen never routes three-phase
        (1, 16, 8, False, "chunkpar3"),  # b1x1024 @ 16 heads: 128 tiles
        (1, 16, 8, True, "chunkpar"),  # v1x1024 @ 16 heads: look-back measured faster
        (1, 16, 16, False, "chunkpar3"),  # b1x2048 @ 16 heads: 256 tiles
        (1, 16, 32, False, "chunkpar"),  # b1x4096 @ 16 heads: 512 tiles
        (1, 16, 64, False, "chunkpar"),  # b1x8192 @ 16 heads: 1024 tiles
        (1, 128, 32, False, "exact"),  # 128 heads: the exact scan
        (4, 16, 8, False, "exact"),  # 64 items x 8 chunks: exact (prediction)
    ),
    ids=(
        "1x8x16",
        "1x8x32",
        "1x8x64",
        "1x8x256",
        "4x8x16",
        "4x8x64",
        "1x32x16",
        "v4x8x16",
        "1x16x8",
        "v1x16x8",
        "1x16x16",
        "1x16x32",
        "1x16x64",
        "1x128x32",
        "4x16x8",
    ),
)
def test_source_chunk_parallel_family_rule(
    capability, batch, nheads, nchunks, varlen, expected
):
    """The three-way family rule at 148 SMs on both capabilities: the cost
    rule decides exact scan vs chunk-parallel
    (the faster of the two chunk-parallel predictions against the exact
    scan's), then the measured-region rule picks the chunk-parallel family --
    the three-phase program for batched calls with at most 256 tiles or with
    at least 32 items and 64 chunks per item, the look-back otherwise; varlen
    calls never route to the three-phase program."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    assert module._THREE_PHASE_MAX_TILES == 256
    assert module._THREE_PHASE_MIN_ITEMS == 32 and module._THREE_PHASE_MIN_CHUNKS == 64
    family = _family(batch, nheads, nchunks, varlen=varlen, capability=capability)
    assert family == expected
    assert (
        _batched_selection(batch, nheads, nchunks, capability=capability)
        is (expected != "exact")
        or varlen
    )
    assert (
        module.three_phase_preferred(
            work_items=batch * nheads,
            chunks=nchunks,
            tiles=batch * nheads * nchunks,
            mode_varlen=varlen,
            capability=capability,
        )
        is (expected == "chunkpar3")
        or expected == "exact"
    )


@pytest.mark.parametrize(
    "batch,nheads,nchunks,varlen,expected",
    (
        (1, 8, 16, False, "chunkpar3"),  # perf_batched_1x16: 128 tiles
        (1, 8, 64, False, "chunkpar3"),  # perf_batched_1x64: 512 tiles, below the band
        (1, 8, 128, False, "chunkpar"),  # 1024 tiles: the look-back band (1.14)
        (
            1,
            8,
            256,
            False,
            "chunkpar",
        ),  # perf_batched_1x256: 8 items x 256 chunks (1.01)
        (4, 8, 16, False, "chunkpar"),  # perf_batched_4x16: 32 items x 16 chunks (1.02)
        (
            4,
            8,
            64,
            False,
            "chunkpar",
        ),  # perf_batched_4x64: 2048 tiles, the look-back band (1.10-1.14)
        (1, 32, 16, False, "chunkpar"),  # cp_b1h32c16: 512 tiles, 16 chunks (1.01)
        (1, 32, 32, False, "chunkpar"),  # 1024 tiles: the look-back band (1.07)
        (1, 16, 32, False, "chunkpar3"),  # 512 tiles, 32 chunks (0.97)
        (1, 16, 128, False, "chunkpar"),  # 2048 tiles: the look-back band (1.11)
        (4, 8, 128, False, "chunkpar3"),  # 4096 tiles, above the band (0.97)
        (4, 8, 16, True, "chunkpar"),  # varlen never routes three-phase
        (1, 8, 8, False, "exact"),  # 8 chunks: the exact scan (Rubin constants)
        (1, 128, 32, False, "exact"),  # 128 heads: the exact scan
    ),
    ids=(
        "1x8x16",
        "1x8x64",
        "1x8x128",
        "1x8x256",
        "4x8x16",
        "4x8x64",
        "1x32x16",
        "1x32x32",
        "1x16x32",
        "1x16x128",
        "4x8x128",
        "v4x8x16",
        "1x8x8",
        "1x128x32",
    ),
)
def test_source_chunk_parallel_family_rule_rubin(
    batch, nheads, nchunks, varlen, expected
):
    """The family rule with the Rubin (10, 7) constants at 212 SMs: inside
    the chunk-parallel regime the three-phase program serves every batched
    call with at most 256 tiles or at least 32 chunks per item, except calls
    of more than 512 and at most 2048 tiles (look-back / three-phase time
    0.88-0.96 on Rubin with the PDL-triggered preprocess)
    and at most 8 items with at least 256 chunks (1.01); 32 items x 16
    chunks and varlen calls stay on the look-back."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    assert module._RUBIN_THREE_PHASE_CAPABILITIES == ((10, 7),)
    assert module._RUBIN_THREE_PHASE_MIN_CHUNKS == 32
    assert (
        module._RUBIN_LOOKBACK_MAX_ITEMS == 8
        and module._RUBIN_LOOKBACK_MIN_CHUNKS == 256
    )
    assert module._RUBIN_LOOKBACK_TILES == (512, 2048)
    family = _family(
        batch,
        nheads,
        nchunks,
        varlen=varlen,
        capability=(10, 7),
        sm_count=_RUBIN_SM_COUNT,
    )
    assert family == expected
    # The Blackwell rule at the same geometry: the same verdict except where
    # the Rubin measurements reversed the two chunk-parallel families.
    blackwell = module.three_phase_preferred(
        work_items=batch * nheads,
        chunks=nchunks,
        tiles=batch * nheads * nchunks,
        mode_varlen=varlen,
        capability=(10, 0),
    )
    rubin = module.three_phase_preferred(
        work_items=batch * nheads,
        chunks=nchunks,
        tiles=batch * nheads * nchunks,
        mode_varlen=varlen,
        capability=(10, 7),
    )
    if expected == "exact":
        return  # the cost rule never reaches the family rule
    assert rubin is (expected == "chunkpar3")
    reversed_on_rubin = (batch, nheads, nchunks, varlen) in {
        (1, 8, 64, False),
        (4, 8, 64, False),
        (1, 16, 32, False),
    }
    assert (rubin != blackwell) is reversed_on_rubin


def test_source_chunk_parallel_selection_quantities_follow_the_cost_model():
    """``selection_quantities`` reports the rule's inputs and both programs'
    predicted main-kernel times from the calibrated constants: the exact
    scan's per-chunk cost interpolates linearly between 32 and 128 work
    items, the chunk-parallel program pays its fixed cost plus one per-tile
    cost for every tile beyond the first wave of SMs; capabilities
    without a calibrated row use the B200 constants."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    model = module._CHUNK_PARALLEL_COST_MODEL_US
    assert set(model) == set(_COST_MODEL_CAPABILITIES)
    assert module._CHUNK_PARALLEL_WORKSPACE_BYTES_PER_TILE == 64 * 128 * (4 + 2)

    def quantities(nheads, nchunks, capability, sm_count=_SELECTION_SM_COUNT):
        return module.selection_quantities(
            nheads=nheads,
            num_sequences=1,
            num_segments=nchunks,
            nchunks=nchunks,
            mode_varlen=False,
            sm_count=sm_count,
            capability=capability,
        )

    small = quantities(8, 16, (10, 0))
    assert small["work_items"] == 8 and small["tiles"] == 128
    assert small["chunks"] == 16 and small["sm_count"] == 148
    assert small["workspace_bytes"] == 128 * 48 * 1024
    b200 = model[(10, 0)]
    predicted = small["predicted_us"]
    assert predicted["exact_scan"] == pytest.approx(
        b200["serial_fixed"] + b200["serial_per_chunk_small"] * 16
    )
    # 128 tiles fit the first wave of 148 SMs: the fixed cost alone.
    assert predicted["chunk_parallel"] == pytest.approx(b200["cp_fixed"])
    three_phase = module._THREE_PHASE_COST_MODEL_US
    assert set(three_phase) == set(_COST_MODEL_CAPABILITIES)
    assert predicted["three_phase"] == pytest.approx(three_phase[(10, 0)]["cp_fixed"])
    assert small["mode_varlen"] is False

    # The calibration row: 1 x 32768 tokens x 8 heads (exact-scan side: the
    # round-5 calibration; chunk-parallel side re-fit on the round-6 look-back
    # program against the exact scan: measured 648 -> 223 us on B200 and
    # 587 -> 201 us on B300 at 16 heads).
    long_prefill = quantities(8, 256, (10, 0))["predicted_us"]
    assert long_prefill["exact_scan"] == pytest.approx(668.754)
    assert long_prefill["chunk_parallel"] == pytest.approx(120.77)
    long_prefill = quantities(8, 256, (10, 3))["predicted_us"]
    assert long_prefill["exact_scan"] == pytest.approx(617.45)
    assert long_prefill["chunk_parallel"] == pytest.approx(108.81)

    large = quantities(128, 32, (10, 3))
    b300 = model[(10, 3)]
    assert large["predicted_us"]["exact_scan"] == pytest.approx(
        b300["serial_fixed"] + b300["serial_per_chunk_large"] * 32
    )
    assert large["predicted_us"]["chunk_parallel"] == pytest.approx(
        b300["cp_fixed"] + b300["cp_per_tile"] * (4096 - 148)
    )
    assert large["predicted_us"]["three_phase"] == pytest.approx(
        three_phase[(10, 3)]["cp_fixed"]
        + three_phase[(10, 3)]["cp_per_tile"] * (4096 - 148)
    )

    # 80 work items: halfway between the small and the large per-chunk cost.
    middle = quantities(80, 16, (10, 0))
    per_chunk = (b200["serial_per_chunk_small"] + b200["serial_per_chunk_large"]) / 2
    assert middle["predicted_us"]["exact_scan"] == pytest.approx(
        b200["serial_fixed"] + per_chunk * 16
    )

    other = quantities(8, 16, (12, 0))
    assert other["capability"] == (12, 0)
    assert other["predicted_us"] == small["predicted_us"]
    # Rubin (10, 7): its own calibrated rows (the same 24-cell protocol on
    # 212 SMs), so 1 x 2048 x 8 heads fits the first wave of 212 SMs and the
    # 32768-token calibration row pays 2048 - 212 per-tile costs.
    assert (10, 7) in module._TARGET_ARCHS
    rubin_model = model[(10, 7)]
    rubin = quantities(8, 16, (10, 7), sm_count=_RUBIN_SM_COUNT)
    assert rubin["capability"] == (10, 7) and rubin["sm_count"] == 212
    assert rubin["predicted_us"]["exact_scan"] == pytest.approx(
        rubin_model["serial_fixed"] + rubin_model["serial_per_chunk_small"] * 16
    )
    assert rubin["predicted_us"]["exact_scan"] == pytest.approx(33.526)
    assert rubin["predicted_us"]["chunk_parallel"] == pytest.approx(20.0)
    assert rubin["predicted_us"]["three_phase"] == pytest.approx(21.0)
    long_prefill = quantities(8, 256, (10, 7), sm_count=_RUBIN_SM_COUNT)["predicted_us"]
    assert long_prefill["exact_scan"] == pytest.approx(475.366)
    assert long_prefill["chunk_parallel"] == pytest.approx(75.4472)
    assert long_prefill["three_phase"] == pytest.approx(69.8376)


@pytest.mark.parametrize("form", ("triple", "cu_seqlens"))
def test_source_chunk_parallel_selection_varlen_uses_the_mean_chunk_count(form):
    """Packed varlen: the per-sequence chunk counts live on the device, so
    the rule's ``chunks`` is ``ceil(num_segments / num_sequences)`` in both
    metadata forms (the cu_seqlens form passes its host segment bound as
    ``num_segments``, so its tiles and chunks are upper bounds)."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    num_sequences, total = 2, 32768
    if form == "triple":
        num_segments = 256  # two 16384-token sequences: 128 logical chunks each
    else:
        num_segments = module._segment_bound(total, num_sequences)
        assert num_segments == 256 + 2 * num_sequences
    kwargs = dict(
        nheads=8,
        num_sequences=num_sequences,
        num_segments=num_segments,
        nchunks=-(-total // 128),
        mode_varlen=True,
        sm_count=_SELECTION_SM_COUNT,
        capability=(10, 0),
    )
    quantities = module.selection_quantities(**kwargs)
    assert quantities["work_items"] == 16
    assert quantities["tiles"] == num_segments * 8
    assert quantities["chunks"] == -(-num_segments // num_sequences)
    assert module.chunk_parallel_selected(**kwargs, mode="auto") is True


def test_source_chunk_parallel_needs_idle_sms():
    """With at least as many (sequence, head) items as SMs the exact scan
    already fills the machine: the chunk-parallel program is not selected
    even where the cost model alone would prefer it."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    kwargs = dict(
        nheads=8,
        num_sequences=1,
        num_segments=256,
        nchunks=256,
        mode_varlen=False,
        capability=(10, 0),
    )
    assert module.chunk_parallel_selected(**kwargs, sm_count=148, mode="auto")
    predicted = module.selection_quantities(**kwargs, sm_count=8)["predicted_us"]
    assert predicted["chunk_parallel"] * 1.05 <= predicted["exact_scan"]
    assert not module.chunk_parallel_selected(**kwargs, sm_count=8, mode="auto")


def test_source_chunk_parallel_workspace_cap():
    """The S/H workspace (32 KB f32 + 16 KB bf16 per (chunk, head) tile) is
    capped at 256 MiB: above it the exact scan serves the call although the
    prediction favours chunk-parallel."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    per_tile = module._CHUNK_PARALLEL_WORKSPACE_BYTES_PER_TILE
    assert per_tile == 48 * 1024
    assert module._CHUNK_PARALLEL_WORKSPACE_CAP_BYTES == 256 << 20

    def selected(nchunks):
        kwargs = dict(
            nheads=8,
            num_sequences=1,
            num_segments=nchunks,
            nchunks=nchunks,
            mode_varlen=False,
            sm_count=_SELECTION_SM_COUNT,
            capability=(10, 0),
        )
        quantities = module.selection_quantities(**kwargs)
        predicted = quantities["predicted_us"]
        assert predicted["chunk_parallel"] * 1.05 <= predicted["exact_scan"]
        assert quantities["workspace_bytes"] == nchunks * 8 * per_tile
        return module.chunk_parallel_selected(**kwargs, mode="auto")

    # 1 x 65536 tokens x 8 heads: 4096 tiles = 192 MiB.
    assert selected(512) is True
    # 1 x 131072 tokens x 8 heads: 8192 tiles = 384 MiB.
    assert selected(1024) is False


def test_source_chunk_parallel_mode_override(monkeypatch):
    """``always`` (look-back) / ``three_phase`` / ``never`` (exact scan) force
    the family regardless of the rule; the
    ``FLASHINFER_CAKE_SSD_CHUNK_PARALLEL`` variable is read on every call,
    trimmed and case-folded, defaults to ``auto`` and rejects any other
    value."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    assert module._CHUNK_PARALLEL_ENV == _CHUNK_PARALLEL_ENV
    assert module._CHUNK_PARALLEL_MODES == ("auto", "always", "three_phase", "never")
    assert _batched_selection(1, 8, 8, mode="always") is True
    assert _batched_selection(1, 8, 8, mode="three_phase") is True
    assert _batched_selection(1, 8, 256, mode="never") is False
    assert _family(1, 8, 256, mode="always") == "chunkpar"
    assert _family(1, 8, 16, mode="always") == "chunkpar"
    assert _family(1, 8, 256, mode="three_phase") == "chunkpar3"
    assert _family(1, 128, 32, mode="three_phase") == "chunkpar3"
    assert _family(1, 8, 256, mode="never") == "exact"
    with pytest.raises(ValueError, match="auto, always, three_phase, never"):
        _batched_selection(1, 8, 16, mode="sometimes")
    with pytest.raises(ValueError, match="auto, always, three_phase, never"):
        _family(1, 8, 16, mode="sometimes")

    assert module._chunk_parallel_mode() == "auto"
    for value, expected in (
        ("always", "always"),
        (" Never ", "never"),
        ("AUTO", "auto"),
        ("three_phase", "three_phase"),
        (" Three_Phase ", "three_phase"),
    ):
        monkeypatch.setenv(_CHUNK_PARALLEL_ENV, value)
        assert module._chunk_parallel_mode() == expected
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "1")
    with pytest.raises(ValueError, match=_CHUNK_PARALLEL_ENV):
        module._chunk_parallel_mode()


def test_source_chunk_parallel_main_arguments_extend_the_exact_order():
    """The chunk-parallel main kernel's parameter order (the generated host's
    positional ABI) is the exact scan's with the state-operand tensor map
    ``h_map`` after ``out_map``, the workspace pointers ``s_work``,
    ``agg_work``, ``h_words``, ``grid_barrier`` after ``out_native`` and the
    virtual CTA count ``vcount`` last; each program binds its family's order
    and host template."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    # The 41 round-5 arguments are the order all families share (the retained
    # start family binds exactly them, ``_MAIN_ARGS_START``): the fused and
    # exactpre families append the three fused-only inputs; the look-back inserts its
    # four workspace values and appends ``vcount``, the three-phase program
    # inserts its three workspace values, and neither binds a fused input.
    fused_only = ("cu_seqlens", "metadata_from_cu_seqlens", "preprocess_status")
    assert module._MAIN_ARGS[-3:] == fused_only
    common = list(module._MAIN_ARGS[:-3])
    expected = list(common)
    expected.insert(expected.index("out_map") + 1, "h_map")
    position = expected.index("out_native") + 1
    expected[position:position] = ["s_work", "agg_work", "h_words", "grid_barrier"]
    expected.append("vcount")
    assert tuple(expected) == module._MAIN_ARGS_CHUNKPAR
    assert len(common) == 41 and len(module._MAIN_ARGS) == 44
    assert tuple(common) == module._MAIN_ARGS_START
    assert len(module._MAIN_ARGS_CHUNKPAR) == 47
    assert set(module._MAIN_ARGS_CHUNKPAR) - set(common) == set(
        _CHUNK_PARALLEL_WORKSPACE_ARGS
    ) | {"vcount"}
    assert not set(fused_only) & set(module._MAIN_ARGS_CHUNKPAR)
    # The three-phase program: the exact order with ``h_map`` after ``out_map``
    # and ``s_work``, ``h_words``, ``grid_barrier`` after ``out_native``; no
    # fused input either.
    expected3 = list(common)
    expected3.insert(expected3.index("out_map") + 1, "h_map")
    position = expected3.index("out_native") + 1
    expected3[position:position] = ["s_work", "h_words", "grid_barrier"]
    assert tuple(expected3) == module._MAIN_ARGS_CHUNKPAR3
    assert len(module._MAIN_ARGS_CHUNKPAR3) == 45
    assert set(module._MAIN_ARGS_CHUNKPAR3) - set(common) == set(
        _CHUNKPAR3_WORKSPACE_ARGS
    )
    assert not set(fused_only) & set(module._MAIN_ARGS_CHUNKPAR3)
    assert set(module._PROGRAMS) >= {
        f"chunkpar3_{state}_{mode}"
        for state in ("bf16", "f16", "f32")
        for mode in ("batched", "varlen")
    }
    for name, program in module._PROGRAMS.items():
        family = name.split("_")[0]
        assert program.family == family
        if family == "chunkpar":
            assert program.main_args == module._MAIN_ARGS_CHUNKPAR
            assert program.host_template == module._CHUNKPAR_HOST_TEMPLATE
            assert program.main_smem_bytes == module._CHUNKPAR_SMEM_BYTES
        elif family == "chunkpar3":
            assert program.main_args == module._MAIN_ARGS_CHUNKPAR3
            assert program.host_template == module._CHUNKPAR3_HOST_TEMPLATE
            assert program.main_smem_bytes == module._CHUNKPAR3_SMEM_BYTES
        else:
            assert program.main_args == (
                module._MAIN_ARGS_START if family == "exact" else module._MAIN_ARGS
            )
            assert (
                program.host_template
                == {
                    "exact": module._HOST_TEMPLATE,
                    "exactfused": module._EXACTFUSED_HOST_TEMPLATE,
                    "exactmeta": module._EXACTMETA_HOST_TEMPLATE,
                    "exactone": module._EXACTFUSED_HOST_TEMPLATE,
                    "exactpre": module._EXACTFUSED_HOST_TEMPLATE,
                }[family]
            )


def test_source_chunk_parallel_launch_orders_its_main_arguments(monkeypatch):
    """``_launch_program`` orders a chunk-parallel program's main values by
    ``_MAIN_ARGS_CHUNKPAR`` (47 values between the preprocess grid and the
    main grid) and fails closed when the workspace values are missing."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []

    class Generated:
        def run(self, *args):
            calls.append(args)

    monkeypatch.setattr(module, "_load_generated_program", lambda *_: Generated())
    preprocess = {name: f"pre:{name}" for name in module._PREPROCESS_ARGS}
    main = {name: f"main:{name}" for name in module._MAIN_ARGS_CHUNKPAR}

    module._launch_program(
        "chunkpar_f32_varlen",
        "sm_100a",
        preprocess=preprocess,
        preprocess_grid=(8, 1, 1),
        main=main,
        main_grid=(148, 1, 1),
        cuda_stream=0x1234,
    )

    assert calls == [
        (
            *(f"pre:{name}" for name in module._PREPROCESS_ARGS),
            8,
            1,
            1,
            *(f"main:{name}" for name in module._MAIN_ARGS_CHUNKPAR),
            148,
            1,
            1,
            0x1234,
        )
    ]
    assert len(calls[0]) == 27 + 3 + 47 + 3 + 1
    exact_values = {name: object() for name in module._MAIN_ARGS}
    with pytest.raises(KeyError, match="h_map"):
        module._launch_program(
            "chunkpar_f32_varlen",
            "sm_100a",
            preprocess=preprocess,
            preprocess_grid=(8, 1, 1),
            main=exact_values,
            main_grid=(1, 1, 1),
            cuda_stream=0,
        )


def _sm107_segment_preprocess(module, name):
    """The sm_107a segment preprocess the loader binds to program ``name``: the
    early-trigger twin when its main kernel launches with programmatic
    dependent launch and its family is in ``_PDL_TRIGGER_FAMILIES``, else the
    plain sm_107a render."""

    if (
        module._main_pdl(name, "sm_107a")
        and module._PROGRAMS[name].family in module._PDL_TRIGGER_FAMILIES
    ):
        return module._SM107_SEGMENT_PREPROCESS_TRIGGER
    return module._SM107_SEGMENT_PREPROCESS


def test_source_sm107_preprocess_trigger_twin():
    """The two sm_107a segment preprocess renders differ only by the early
    dependent-launch trigger: the twin (``_SM107_SEGMENT_PREPROCESS_TRIGGER``)
    issues ``griddepcontrol.launch_dependents`` before any work and the plain
    render never does, so a main grid launched with programmatic dependent
    launch behind the plain render starts when the preprocess CTAs exit.  Only
    the ``_PDL_TRIGGER_FAMILIES`` programs that launch their main kernel with
    programmatic dependent launch run the twin; ``exactpre`` is not one of
    those families."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    device_dir = module._source_dir() / module._DEVICE_DIR
    plain_kernel = module._SM107_SEGMENT_PREPROCESS
    twin_kernel = module._SM107_SEGMENT_PREPROCESS_TRIGGER
    plain = (device_dir / plain_kernel.source).read_text(encoding="utf-8")
    twin = (device_dir / twin_kernel.source).read_text(encoding="utf-8")
    trigger = '    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n'
    assert "griddepcontrol" not in plain
    assert twin.count("griddepcontrol") == 1 and twin.count(trigger) == 1
    assert twin.replace(trigger, "", 1) == plain
    assert twin_kernel.kernel == plain_kernel.kernel
    assert (twin_kernel.threads, twin_kernel.fast_math) == (
        plain_kernel.threads,
        plain_kernel.fast_math,
    )
    assert module._PDL_TRIGGER_FAMILIES <= module._PDL_FAMILIES
    assert "exactpre" not in module._PDL_TRIGGER_FAMILIES
    triggered = set()
    for name, program in module._PROGRAMS.items():
        for arch in ("sm_100a", "sm_103a"):
            assert module._arch_program(name, arch).preprocess is program.preprocess
        rubin = module._arch_program(name, "sm_107a")
        if rubin.preprocess is twin_kernel:
            assert module._main_pdl(name, "sm_107a")
            assert program.family in module._PDL_TRIGGER_FAMILIES
            triggered.add(program.family)
        elif program.preprocess is module._SEGMENT_PREPROCESS:
            assert rubin.preprocess is plain_kernel
    assert triggered == set(module._PDL_TRIGGER_FAMILIES)


def test_source_sm107_lookup_shadows_the_fold_main_module():
    """``_main_module`` / ``_arch_program``: on sm_107a (Rubin) a program in
    ``_SM107_MODULES`` builds the table module -- the same kernel symbol,
    thread count and build flags behind the same host template and SMEM
    literal, or the render's own literal for a kind
    ``_SM107_VARIANT_SMEM_BYTES`` lists -- and every other program (the retained start
    ``exact_*`` programs, the SM100-only ``exactmeta_*`` programs and the
    programs whose fold source selects the Rubin descriptor mask by
    ``__CUDA_ARCH__``) its fold module; sm_100a and sm_103a always build the
    fold module.  On sm_107a every program but ``exactmeta_*`` runs the
    Rubin render of the segment preprocess (``_SM107_SEGMENT_PREPROCESS``:
    the same symbol, thread count and build flags in its own module);
    sm_100a and sm_103a build the shared one.  ``_TARGET_ARCHS`` names the
    three capabilities, the
    seq_idx triple form deriving its table routes SM107 like SM103 (the
    one-launch ``exactone`` render), the look-back program
    balances against Rubin's 212 SMs capped at the device's SM count and the
    cost model has no Rubin row yet (the B200 constants until a calibration on
    Rubin fills it)."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    assert module._TARGET_ARCHS == {
        (10, 0): "sm_100a",
        (10, 3): "sm_103a",
        (10, 7): "sm_107a",
    }
    assert module._ARCH_CAPABILITIES["sm_107a"] == (10, 7)
    assert module._SM107_ARCH == "sm_107a"
    assert module._LIGHT_STAGE_CAPABILITIES == ((10, 0),)
    assert module._ONE_LAUNCH_CAPABILITIES == ((10, 3), (10, 7))
    assert module._CHUNK_PARALLEL_VCOUNT_BY_ARCH == {
        (10, 0): 148,
        (10, 3): 152,
        (10, 7): 212,
    }
    assert module._chunk_parallel_vcount((10, 7)) == 212
    # The cap at the device's SM count: the 208-SM R200 board variant gets 208,
    # a 216-SM board keeps 212, no SM count keeps the table value, and the cap
    # only lowers.
    assert module._chunk_parallel_vcount((10, 7), 208) == 208
    assert module._chunk_parallel_vcount((10, 7), 216) == 212
    assert module._chunk_parallel_vcount((10, 7), None) == 212
    assert module._chunk_parallel_vcount((10, 3), 148) == 148
    assert module._chunk_parallel_vcount((10, 0), 1024) == 148
    assert (10, 7) in module._CHUNK_PARALLEL_COST_MODEL_US
    assert (10, 7) in module._THREE_PHASE_COST_MODEL_US
    for name, program in module._PROGRAMS.items():
        for arch in ("sm_100a", "sm_103a"):
            assert module._main_module(name, arch) == program.main.module
            assert module._arch_program(name, arch) is program
        rubin = module._arch_program(name, "sm_107a")
        if program.family == "exactmeta":
            assert program.preprocess is module._PACKED_METADATA
            assert rubin.preprocess is program.preprocess
            expected_preprocess = program.preprocess
        else:
            assert program.preprocess is module._SEGMENT_PREPROCESS
            expected_kernel = _sm107_segment_preprocess(module, name)
            assert rubin.preprocess is expected_kernel
            expected_preprocess = module.replace(
                program.preprocess, module=expected_kernel.module
            )
        assert rubin.preprocess == expected_preprocess
        if name.startswith(("exact_", "exactmeta_")):
            assert name not in module._SM107_MODULES
        if name not in module._SM107_MODULES:
            assert module._main_module(name, "sm_107a") == program.main.module
            if program.family == "exactmeta":
                assert rubin is program
            else:
                assert rubin.main is program.main
                assert module.replace(rubin, preprocess=program.preprocess) == program
            continue
        expected = module._SM107_MODULES[name]
        assert module._main_module(name, "sm_107a") == expected
        assert rubin.main == module._scan(expected)
        assert rubin.main.kernel == program.main.kernel
        assert rubin.main.threads == 512 and rubin.main.fast_math
        kind = (program.family, name.split("_")[2], program.variant)
        assert rubin.main_smem_bytes == module._SM107_VARIANT_SMEM_BYTES.get(
            kind, program.main_smem_bytes
        )
        assert (
            module.replace(
                rubin,
                main=program.main,
                preprocess=program.preprocess,
                main_smem_bytes=program.main_smem_bytes,
            )
            == program
        )


def test_source_sm107_program_builds_its_table_module(monkeypatch, tmp_path):
    """On sm_107a the loader builds the program's ``_SM107_MODULES`` entry,
    never its fold source: the identity is compiled with ``-arch=sm_107a``
    from its own device file and embedded under its own module name next to
    the sm_107a segment preprocess cubin (never the shared one).  The table
    entry is overridden here, so the test holds for a program with or without
    a shipped sm_107a render."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    name = "exactfused_bf16_batched"
    fold = module._PROGRAMS[name]
    stem = fold.main.module.rsplit("_", 1)[0]
    identity = f"{stem}_0123456789"
    monkeypatch.setitem(module._SM107_MODULES, name, identity)
    monkeypatch.setattr(module, "_source_dir", lambda: tmp_path)
    module._load_generated_program.cache_clear()
    rubin = module._arch_program(name, "sm_107a")
    assert rubin.main.module == identity and rubin.main.kernel == fold.main.kernel
    assert rubin.preprocess is module._SM107_SEGMENT_PREPROCESS
    assert rubin.preprocess.module != fold.preprocess.module

    # The fake source tree holds the sm_107a preprocess and main sources
    # only: reading a fold source would fail.
    device_dir = tmp_path / module._DEVICE_DIR
    device_dir.mkdir(parents=True)
    for kernel in rubin.kernels:
        (device_dir / kernel.source).write_text(
            f"{kernel.kernel} source\n", encoding="utf-8"
        )
    host = tmp_path / fold.host_template
    host.parent.mkdir(parents=True)
    host.write_text(
        "namespace host_CAKE_SSD_PROGRAM {}\n"
        "TVM_FFI_EMBED_CUBIN(CAKE_SSD_PREPROCESS_MODULE);\n"
        "TVM_FFI_EMBED_CUBIN(CAKE_SSD_MAIN_MODULE);\n"
        "CAKE_SSD_PREPROCESS_KERNEL CAKE_SSD_PREPROCESS_THREADS CAKE_SSD_MAIN_KERNEL "
        "CAKE_SSD_STATE_DTYPE_CODE CAKE_SSD_STATE_DTYPE_BITS CAKE_SSD_MAIN_SMEM_BYTES\n",
        encoding="utf-8",
    )
    nvcc = tmp_path / "cuda" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    calls, load_calls = [], []

    def run(command, **kwargs):
        calls.append(command)
        module.Path(command[-1]).write_bytes(module.Path(command[-3]).read_bytes())
        return SimpleNamespace(returncode=0, stderr="")

    def load_inline(*args, **kwargs):
        load_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(module, "_nvcc", lambda: nvcc)
    monkeypatch.setattr(module.jit_env, "FLASHINFER_JIT_DIR", tmp_path / "jit")
    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "cpp", SimpleNamespace(load_inline=load_inline))

    module._load_generated_program(name, "sm_107a")
    module._load_generated_program.cache_clear()

    assert [command[-3] for command in calls] == [
        str(device_dir / kernel.source) for kernel in rubin.kernels
    ]
    assert all("-arch=sm_107a" in command for command in calls)
    ((args, kwargs),) = load_calls
    assert args[0].startswith(f"cake_mamba_ssd_{name}_sm_107a_")
    rendered = kwargs["cpp_sources"]
    assert f"TVM_FFI_EMBED_CUBIN({identity});" in rendered
    assert f"TVM_FFI_EMBED_CUBIN({fold.main.module});" not in rendered
    assert f"TVM_FFI_EMBED_CUBIN({rubin.preprocess.module});" in rendered
    assert f"TVM_FFI_EMBED_CUBIN({fold.preprocess.module});" not in rendered
    assert f"{fold.main.kernel} 4 16 {fold.main_smem_bytes}" in rendered
    assert set(kwargs["embed_cubin"]) == {rubin.preprocess.module, identity}
    assert (tmp_path / "jit" / args[0] / f"{identity}.cubin").is_file()


def test_source_sm107_forward_binds_the_rubin_vcount_without_gpu(monkeypatch):
    """Forced chunk-parallel on an sm_107a device, the look-back program
    balances its item-major tile range against min(212, the device's SM
    count) (V = 212 virtual CTA ranges on a 212-SM device, one aggregate per
    range) and caps the main grid at the device's SM count."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    runner = _cpu_forwarding_runner(
        module, monkeypatch, calls, nheads=8, ngroups=8, arch="sm_107a"
    )
    monkeypatch.setattr(module, "_sm_count", lambda _: 212)
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "always")
    seqlen = 32768  # 256 chunks x 8 heads = 2048 tiles
    x = torch.empty((1, seqlen, 8, 64), dtype=torch.bfloat16)
    dt = torch.empty((1, seqlen, 8), dtype=torch.float32)
    A = torch.empty((8,), dtype=torch.float32)
    B = torch.empty((1, seqlen, 8, 128), dtype=torch.bfloat16)

    runner.run(x, dt, A, B, torch.empty_like(B))

    name, launch = calls[-1]
    assert name == runner.last_program_name == "chunkpar_bf16_batched"
    assert launch["main_grid"] == (212, 1, 1)
    main = launch["main"]
    assert main["vcount"] == 212 == module._CHUNK_PARALLEL_VCOUNT_BY_ARCH[(10, 7)]
    assert tuple(main["agg_work"].shape) == (212 * (64 * 128 + 1),)


@pytest.mark.parametrize(
    "arch, capability, sm_count, expected",
    (
        ("sm_107a", (10, 7), 216, 212),
        ("sm_107a", (10, 7), 212, 212),
        ("sm_107a", (10, 7), 208, 208),
        ("sm_100a", (10, 0), 1024, 148),
    ),
)
def test_source_chunk_parallel_vcount_is_capped_at_the_device_sm_count_without_gpu(
    monkeypatch, arch, capability, sm_count, expected
):
    """Forced chunk-parallel, the bound ``vcount`` is the capability's table
    value capped at the device's SM count: a 208-SM R200 board variant gets
    208 ranges (one per CTA, no second round), 212- and 216-SM boards keep
    212, and a board with more SMs than the table value keeps the table value.
    The workspace's ``agg_work`` is sized for the same V on the first call and
    a smaller later call reuses it: V depends on the device, not the shape."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    runner = _cpu_forwarding_runner(
        module, monkeypatch, calls, nheads=8, ngroups=8, arch=arch
    )
    monkeypatch.setattr(module, "_sm_count", lambda _: sm_count)
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "always")

    def inputs(seqlen):
        x = torch.empty((1, seqlen, 8, 64), dtype=torch.bfloat16)
        dt = torch.empty((1, seqlen, 8), dtype=torch.float32)
        A = torch.empty((8,), dtype=torch.float32)
        B = torch.empty((1, seqlen, 8, 128), dtype=torch.bfloat16)
        return x, dt, A, B, torch.empty_like(B)

    runner.run(*inputs(32768))  # 256 chunks x 8 heads = 2048 tiles
    name, launch = calls[-1]
    assert name == runner.last_program_name == "chunkpar_bf16_batched"
    assert launch["main_grid"] == (min(2048, sm_count), 1, 1)
    main = launch["main"]
    assert (
        main["vcount"]
        == expected
        == module._chunk_parallel_vcount(capability, sm_count)
    )
    assert main["vcount"] <= launch["main_grid"][0]
    agg_work = main["agg_work"]
    assert tuple(agg_work.shape) == (expected * (64 * 128 + 1),)

    runner.run(*inputs(2048))  # 16 chunks x 8 heads = 128 tiles
    name, launch = calls[-1]
    assert name == "chunkpar_bf16_batched"
    assert launch["main_grid"] == (128, 1, 1)
    assert launch["main"]["vcount"] == expected
    assert launch["main"]["agg_work"] is agg_work


def test_source_chunk_parallel_program_builds_from_its_own_template(
    monkeypatch, tmp_path
):
    """An exported chunk-parallel program renders the chunk-parallel host
    template (not the exact scan's) with the same nine placeholders, embeds
    its own main cubin and gets its own module name; the exact program of
    the same source tree still renders the exact template."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    exact = module._PROGRAMS["exact_bf16_batched"]
    program = module._Program(
        "chunkpar",
        module._SEGMENT_PREPROCESS,
        module._scan("mamba_ssd_chunk_parallel_bf16_batched_0123456789"),
        4,
        16,
        200704,
        module._CHUNKPAR_HOST_TEMPLATE,
        module._MAIN_ARGS_CHUNKPAR,
        module._PREPROCESS_ARGS,
    )
    name = "chunkpar_bf16_batched_probe"
    monkeypatch.setitem(module._PROGRAMS, name, program)
    device_dir = tmp_path / module._DEVICE_DIR
    device_dir.mkdir(parents=True)
    for kernel in (*exact.kernels, program.main):
        (device_dir / kernel.source).write_text(
            f"{kernel.kernel} source\n", encoding="utf-8"
        )
    placeholders = " ".join(_HOST_PLACEHOLDERS)
    for template, label in (
        (module._HOST_TEMPLATE, "exact"),
        (module._CHUNKPAR_HOST_TEMPLATE, "chunkpar"),
    ):
        host = tmp_path / template
        host.parent.mkdir(parents=True, exist_ok=True)
        host.write_text(
            f"// {label} launcher\n"
            "TVM_FFI_EMBED_CUBIN(CAKE_SSD_PREPROCESS_MODULE);\n"
            "TVM_FFI_EMBED_CUBIN(CAKE_SSD_MAIN_MODULE);\n"
            f"{placeholders}\n",
            encoding="utf-8",
        )
    nvcc = tmp_path / "cuda" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    load_calls = []

    def run(command, **kwargs):
        module.Path(command[-1]).write_bytes(module.Path(command[-3]).read_bytes())
        return SimpleNamespace(returncode=0, stderr="")

    def load_inline(*args, **kwargs):
        load_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(module, "_source_dir", lambda: tmp_path)
    monkeypatch.setattr(module, "_nvcc", lambda: nvcc)
    monkeypatch.setattr(module.jit_env, "FLASHINFER_JIT_DIR", tmp_path / "jit")
    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "cpp", SimpleNamespace(load_inline=load_inline))
    module._load_generated_program.cache_clear()

    module._load_generated_program(name, "sm_100a")
    module._load_generated_program("exact_bf16_batched", "sm_100a")
    module._load_generated_program.cache_clear()

    (chunkpar_args, chunkpar_kwargs), (exact_args, exact_kwargs) = load_calls
    assert chunkpar_args[0].startswith(f"cake_mamba_ssd_{name}_sm_100a_")
    assert exact_args[0].startswith("cake_mamba_ssd_exact_bf16_batched_sm_100a_")
    rendered = chunkpar_kwargs["cpp_sources"]
    assert rendered.startswith("// chunkpar launcher\n")
    assert exact_kwargs["cpp_sources"].startswith("// exact launcher\n")
    assert not any(placeholder in rendered for placeholder in _HOST_PLACEHOLDERS)
    assert f"TVM_FFI_EMBED_CUBIN({program.main.module});" in rendered
    preprocess = module._SEGMENT_PREPROCESS
    assert (
        f"{name} {preprocess.module} {preprocess.kernel} {preprocess.threads} "
        f"{program.main.module} {program.main.kernel} 4 16 200704"
    ) in rendered
    assert set(chunkpar_kwargs["embed_cubin"]) == {
        preprocess.module,
        program.main.module,
    }
    build_dir = tmp_path / "jit" / chunkpar_args[0]
    assert (build_dir / f"{program.main.module}.cubin").is_file()


def test_source_chunk_parallel_host_template_binds_main_arguments_in_loader_order():
    """The chunk-parallel host launcher is its own exported template with
    the same nine placeholders and namespace scheme as the exact scan's;
    its ``Prepare`` / ``Run`` list ``_PREPROCESS_ARGS`` and
    ``_MAIN_ARGS_CHUNKPAR`` in order.  Fails by name until the Cake export
    delivers the template."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    assert (
        Path("generated/host/mamba_ssd_combined_sequence.cpp") == module._HOST_TEMPLATE
    )
    assert (
        Path("generated/host/mamba_ssd_combined_fused_sequence.cpp")
        == module._EXACTFUSED_HOST_TEMPLATE
    )
    assert (
        Path("generated/host/mamba_ssd_combined_chunk_parallel_sequence.cpp")
        == module._CHUNKPAR_HOST_TEMPLATE
    )
    template_path = module._source_dir() / module._CHUNKPAR_HOST_TEMPLATE
    assert template_path.is_file(), f"{template_path} is not exported yet"
    template = template_path.read_text(encoding="utf-8")

    assert _host_prepare_arguments(template, "preprocess") == module._PREPROCESS_ARGS
    assert _host_prepare_arguments(template, "main") == module._MAIN_ARGS_CHUNKPAR
    assert _host_run_arguments(template, "preprocess") == module._PREPROCESS_ARGS
    assert _host_run_arguments(template, "main") == module._MAIN_ARGS_CHUNKPAR
    for placeholder in _HOST_PLACEHOLDERS:
        assert placeholder in template, placeholder
    assert "namespace cake_mamba_ssd_combined_host_CAKE_SSD_PROGRAM {" in template
    assert "stream, CAKE_SSD_MAIN_SMEM_BYTESu)" in template
    assert "LaunchEx(" not in template
    assert "PROGRAMMATIC_STREAM_SERIALIZATION" not in template
    program = module._PROGRAMS["chunkpar_bf16_varlen"]
    rendered = module._render_host_source(template, "chunkpar_bf16_varlen", program)
    assert not any(placeholder in rendered for placeholder in _HOST_PLACEHOLDERS)
    assert "namespace cake_mamba_ssd_combined_host_chunkpar_bf16_varlen {" in rendered
    assert f"TVM_FFI_EMBED_CUBIN({program.main.module});" in rendered


def test_source_three_phase_host_template_binds_main_arguments_in_loader_order():
    """The three-phase chunk-parallel host launcher is the third exported
    template with the same nine placeholders and namespace scheme; its
    ``Prepare`` / ``Run`` list ``_PREPROCESS_ARGS`` and ``_MAIN_ARGS_CHUNKPAR3``
    in order, and every program of the family ships its main source."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    assert (
        Path("generated/host/mamba_ssd_combined_chunk_parallel3_sequence.cpp")
        == module._CHUNKPAR3_HOST_TEMPLATE
    )
    template_path = module._source_dir() / module._CHUNKPAR3_HOST_TEMPLATE
    assert template_path.is_file(), f"{template_path} is not exported yet"
    template = template_path.read_text(encoding="utf-8")

    assert _host_prepare_arguments(template, "preprocess") == module._PREPROCESS_ARGS
    assert _host_prepare_arguments(template, "main") == module._MAIN_ARGS_CHUNKPAR3
    assert _host_run_arguments(template, "preprocess") == module._PREPROCESS_ARGS
    assert _host_run_arguments(template, "main") == module._MAIN_ARGS_CHUNKPAR3
    for placeholder in _HOST_PLACEHOLDERS:
        assert placeholder in template, placeholder
    assert "namespace cake_mamba_ssd_combined_host_CAKE_SSD_PROGRAM {" in template
    assert "stream, CAKE_SSD_MAIN_SMEM_BYTESu)" in template
    assert "LaunchEx(" not in template
    assert "PROGRAMMATIC_STREAM_SERIALIZATION" not in template
    for name, program in module._PROGRAMS.items():
        if program.family != "chunkpar3":
            continue
        assert (
            module._source_dir() / module._DEVICE_DIR / program.main.source
        ).is_file(), name
    program = module._PROGRAMS["chunkpar3_bf16_varlen"]
    rendered = module._render_host_source(template, "chunkpar3_bf16_varlen", program)
    assert not any(placeholder in rendered for placeholder in _HOST_PLACEHOLDERS)
    assert "namespace cake_mamba_ssd_combined_host_chunkpar3_bf16_varlen {" in rendered
    assert f"TVM_FFI_EMBED_CUBIN({program.main.module});" in rendered


def test_source_chunk_parallel_forward_binds_the_workspace_without_gpu(monkeypatch):
    """Forced chunk-parallel, the runner names the ``chunkpar_*`` program and
    binds the five workspace values: ``h_map``, the contiguous bf16
    ``[tiles, 64, 128]`` state operand the host wraps in a TMA descriptor;
    ``s_work``, the f32 ``[tiles, 64, 128]`` state increments; ``agg_work``,
    the f32 range aggregates of the ``vcount`` virtual CTA ranges; ``h_words``,
    the u32 view of ``h_map``'s storage; ``grid_barrier``, 1028 u32 words
    allocated zeroed once per device and never reset by the host.  The main
    grid is one CTA per (chunk, head) tile capped at the SM count.  The
    buffers grow only; ``never`` and the rule's own choice fall back to the
    exact scan on the same runner without the four values."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    runner = _cpu_forwarding_runner(module, monkeypatch, calls, nheads=8, ngroups=8)
    monkeypatch.setattr(module, "_sm_count", lambda _: 148)
    assert runner.last_program_name is None

    def inputs(seqlen):
        x = torch.empty((1, seqlen, 8, 64), dtype=torch.bfloat16)
        dt = torch.empty((1, seqlen, 8), dtype=torch.float32)
        A = torch.empty((8,), dtype=torch.float32)
        B = torch.empty((1, seqlen, 8, 128), dtype=torch.bfloat16)
        return x, dt, A, B, torch.empty_like(B)

    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "always")
    runner.run(*inputs(2048))  # 16 chunks x 8 heads = 128 tiles
    name, launch = calls[-1]
    assert name == runner.last_program_name == "chunkpar_bf16_batched"
    assert launch["main_grid"] == (128, 1, 1)
    main = launch["main"]
    assert set(main) == set(module._MAIN_ARGS_CHUNKPAR)
    h_map, s_work, agg_work, h_words, barrier = (
        main[key] for key in _CHUNK_PARALLEL_WORKSPACE_ARGS
    )
    assert tuple(h_map.shape) == (128, 64, 128) and h_map.dtype == torch.bfloat16
    assert h_map.is_contiguous()
    assert tuple(s_work.shape) == (128, 64, 128) and s_work.dtype == torch.float32
    assert tuple(h_words.shape) == (128, 64, 64) and h_words.dtype == torch.uint32
    assert h_words.data_ptr() == h_map.data_ptr()
    # sm_103a on a 148-SM fake device: V = min(152, 148) = 148 ranges.
    assert (
        main["vcount"]
        == 148
        == min(module._CHUNK_PARALLEL_VCOUNT_BY_ARCH[(10, 3)], 148)
    )
    assert tuple(agg_work.shape) == (148 * (64 * 128 + 1),)
    assert agg_work.dtype == torch.float32
    assert tuple(barrier.shape) == (1028,) and barrier.dtype == torch.uint32
    assert not barrier.view(torch.int32).any()
    # One preprocess (segment, head) row per tile.
    assert launch["preprocess"]["delta"].shape[0] == 128

    # Grow only: a longer call replaces the buffers and caps the grid at the
    # SM count; the barrier words persist untouched (they are the kernel's).
    barrier.view(torch.int32)[1] = 7
    runner.run(*inputs(32768))  # 256 chunks x 8 heads = 2048 tiles
    name, launch = calls[-1]
    assert name == "chunkpar_bf16_batched"
    assert launch["main_grid"] == (148, 1, 1)
    grown = launch["main"]
    assert tuple(grown["h_map"].shape) == (2048, 64, 128)
    assert grown["h_words"].data_ptr() == grown["h_map"].data_ptr()
    assert grown["grid_barrier"] is barrier
    assert grown["agg_work"] is agg_work
    assert barrier.view(torch.int32)[:2].tolist() == [0, 7]
    assert not barrier.view(torch.int32)[2:].any()
    # A shorter call reuses the larger buffers (the kernel addresses only
    # the tiles below its own bound).
    runner.run(*inputs(2048))
    name, launch = calls[-1]
    assert launch["main_grid"] == (128, 1, 1)
    assert launch["main"]["h_map"] is grown["h_map"]
    assert launch["main"]["s_work"] is grown["s_work"]

    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "never")
    runner.run(*inputs(32768))
    name, launch = calls[-1]
    # 256 chunks per sequence exceed the fused window (16): the forced exact
    # scan is the two-kernel ``exactpre`` program (the round-5 schedule).
    assert name == runner.last_program_name == "exactpre_bf16_batched"
    assert set(launch["main"]) == set(module._MAIN_ARGS)
    assert launch["main_grid"] == (8, 1, 1)

    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "auto")
    runner.run(*inputs(32768))  # 8 work items on 148 SMs, 256 chunks
    assert calls[-1][0] == "chunkpar_bf16_batched"
    runner.run(*inputs(512))  # 4 chunks x 8 heads = 32 tiles: the exact scan
    assert calls[-1][0] == "exactfused_bf16_batched"

    # The three-phase family: the same S/H buffers, its own two-word barrier
    # (zeroed once per device, distinct from the look-back's 1028 words), no
    # aggregate workspace or vcount, the same tile-capped grid.
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "three_phase")
    runner.run(*inputs(2048))
    name, launch = calls[-1]
    assert name == runner.last_program_name == "chunkpar3_bf16_batched"
    assert launch["main_grid"] == (128, 1, 1)
    main3 = launch["main"]
    assert set(main3) == set(module._MAIN_ARGS_CHUNKPAR3)
    assert main3["h_map"] is grown["h_map"] and main3["s_work"] is grown["s_work"]
    assert main3["h_words"].data_ptr() == main3["h_map"].data_ptr()
    barrier3 = main3["grid_barrier"]
    assert tuple(barrier3.shape) == (2,) and barrier3.dtype == torch.uint32
    assert not barrier3.view(torch.int32).any()
    assert barrier3 is not barrier
    assert barrier.view(torch.int32)[:2].tolist() == [0, 7]
    runner.run(*inputs(32768))
    name, launch = calls[-1]
    assert name == "chunkpar3_bf16_batched"
    assert launch["main_grid"] == (148, 1, 1)
    assert launch["main"]["grid_barrier"] is barrier3
    # The rule's own choice: 8 items x 16 chunks = 128 tiles is the
    # three-phase region, 8 items x 256 chunks the look-back's.
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "auto")
    runner.run(*inputs(2048))
    assert calls[-1][0] == runner.last_program_name == "chunkpar3_bf16_batched"
    runner.run(*inputs(32768))
    assert calls[-1][0] == "chunkpar_bf16_batched"
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "later")
    with pytest.raises(ValueError, match=_CHUNK_PARALLEL_ENV):
        runner.run(*inputs(512))


def _state_key(state_dtype):
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    return module._STATE_DTYPE_KEYS[state_dtype]


def _forced_runs(monkeypatch, runner, tensors, arguments):
    """The same call on the forced exact scan (``never``), the forced
    look-back program (``always``) and the forced three-phase program
    (``three_phase``), per mode: the cloned output, the cloned final states,
    the checkpoint-state table the call wrote (a fresh copy of the caller's
    table per mode; ``None`` without checkpoints) and the program the runner
    reports."""

    results = {}
    # The forced modes are scoped to this helper; the caller's later calls
    # run the runner's own program choice again.
    prior_mode = os.environ.get(_CHUNK_PARALLEL_ENV)
    for mode in ("never", "always", "three_phase"):
        monkeypatch.setenv(_CHUNK_PARALLEL_ENV, mode)
        call_arguments = dict(arguments)
        checkpoint_states = arguments.get("checkpoint_states")
        if checkpoint_states is not None:
            checkpoint_states = checkpoint_states.clone()
            call_arguments["checkpoint_states"] = checkpoint_states
        out, final = runner.run(*tensors, **call_arguments)
        results[mode] = SimpleNamespace(
            out=out.clone(),
            final=final.clone(),
            checkpoint_states=checkpoint_states,
            program=runner._cake_runner.last_program_name,
        )
    if prior_mode is None:
        monkeypatch.delenv(_CHUNK_PARALLEL_ENV, raising=False)
    else:
        monkeypatch.setenv(_CHUNK_PARALLEL_ENV, prior_mode)
    return results


# GOAL6 numerical contract 1 (P6), clause GOAL6-A-20261012, as the FlashInfer
# reading of the former ``chunk-parallel == exact scan`` bitwise assertions:
# the exact-scan family changed its rounding in round 6 (prescaled decay
# factor table, one-rounding FMA state update) and the look-back program
# (``chunkpar``) re-associates the carried state of later tile ranges, while
# the three-phase program (``chunkpar3``) is bitwise the round-5 exact scan,
# so its metrics against the fp64 recurrence are the round-5 reading the exact
# scan and the look-back are judged against.  K = 8 fixed seeds per shape;
# the element tolerance of m1 is the P6 one and is not loosened.
_P6_SEEDS = tuple(range(8))
_P6_METRICS = ("m1", "m2", "m3", "m4", "m5")


def _p6_metrics(actual, oracle):
    """The P6 metrics of ``actual`` against the fp64 ``oracle``: m1 outliers
    beyond ``1e-2 + 1e-2 |o|``, m2 worst distance to ``RN(o)`` in ulps of the
    output dtype, m3 p99.9 and m4 RMS of ``|c - o| / (1 + |o|)``, m5 max
    ``|c - o|``; ``actual`` must be finite."""

    f64 = torch.float64
    o = oracle.to(f64).reshape(-1)
    c = actual.to(f64).reshape(-1)
    assert torch.isfinite(c).all(), "non-finite output"
    err = (c - o).abs()
    rn = oracle.to(actual.dtype).to(f64).reshape(-1)
    info = torch.finfo(actual.dtype)
    ulp = torch.exp2(torch.floor(torch.log2(rn.abs().clamp(min=info.tiny)))) * info.eps
    e = err / (1.0 + o.abs())
    n = o.numel()
    k = n - math.ceil(0.999 * n) + 1
    return {
        "m1": float((err > _ATOL + _RTOL * o.abs()).sum()),
        "m2": float(((c - rn).abs() / ulp).max()),
        "m3": float(torch.kthvalue(e, n - k + 1).values),
        "m4": float(torch.sqrt((e * e).sum() / n)),
        "m5": float(err.max()),
    }


def _assert_p6_clause(candidate, baseline, label):
    """Clause GOAL6-A-20261012: per metric, the K-seed mean of the candidate
    is at most the baseline's mean plus one baseline seed-to-seed
    (population) standard deviation; per-seed exceedances are disclosed in
    the failure text."""

    assert len(candidate) == len(baseline) == len(_P6_SEEDS)
    for metric in _P6_METRICS:
        cand = torch.tensor([m[metric] for m in candidate], dtype=torch.float64)
        base = torch.tensor([m[metric] for m in baseline], dtype=torch.float64)
        sd = base.std(unbiased=False)
        exceed = int((cand > base).sum())
        assert cand.mean() <= base.mean() + sd, (
            f"{label}: {metric} mean {cand.mean().item():.6g} over {len(_P6_SEEDS)} "
            f"seeds exceeds the three-phase (round-5) reading "
            f"{base.mean().item():.6g} + sd {sd.item():.6g}; per-seed exceedances "
            f"{exceed}; candidate {cand.tolist()}; baseline {base.tolist()}"
        )


def _checkpoint_oracle(constructor, tensors, arguments, *, sequence, start, stop):
    """fp64 state of ``sequence`` after the tokens ``[start, stop)`` of its
    row (batched: the row is the sequence; packed varlen: the token axis),
    scanned from its initial state -- the oracle of a checkpoint there."""

    x, dt, A, B, C = tensors
    row = slice(0, 1) if constructor["has_varlen"] else slice(sequence, sequence + 1)
    window = slice(start, stop)
    prefix_tensors = (
        x[row, window].contiguous(),
        dt[row, window].contiguous(),
        A,
        B[row, window].contiguous(),
        C[row, window].contiguous(),
    )
    prefix_arguments = {
        **arguments,
        "seq_idx": None,
        "chunk_indices": None,
        "chunk_offsets": None,
        "seq_chunk_cumsum": None,
        "cu_seqlens": None,
        "checkpoint_token_indices": None,
        "checkpoint_state_slots": None,
        "checkpoint_states": None,
    }
    prefix_arguments.pop("num_seqs", None)
    if arguments.get("z") is not None:
        prefix_arguments["z"] = arguments["z"][row, window].contiguous()
    if arguments.get("initial_states") is not None:
        prefix_arguments["initial_states"] = arguments["initial_states"][
            sequence : sequence + 1
        ].contiguous()
    _, states = _fp64_reference(
        {**constructor, "has_varlen": False}, prefix_tensors, prefix_arguments
    )
    return states[0]


def _assert_chunk_parallel_p6(monkeypatch, make_case, *, checkpoints=()):
    """The P6 reading of a chunk-parallel == exact-scan test: on the K fixed
    seeds run the three forced programs on the same inputs (``make_case(seed)``
    -> constructor, tensors, arguments, oracle arguments) and judge the exact
    scan's and the look-back program's metrics against the fp64 recurrence
    with the three-phase program's as the round-5 reading -- for the output,
    the final states and every ``(slot, sequence, start, stop)`` checkpoint."""

    candidates = {"never": {}, "always": {}}
    baseline = {}
    for seed in _P6_SEEDS:
        constructor, tensors, arguments, oracle_arguments = make_case(seed)
        runner = SSDCombined(**constructor, backend="cake")
        results = _forced_runs(monkeypatch, runner, tensors, arguments)
        y, states = _fp64_reference(constructor, tensors, oracle_arguments)
        readings = [
            ("out", lambda result: result.out, y),
            ("final", lambda result: result.final, states),
        ]
        for slot, sequence, start, stop in checkpoints:
            oracle = _checkpoint_oracle(
                constructor,
                tensors,
                arguments,
                sequence=sequence,
                start=start,
                stop=stop,
            )
            readings.append(
                (
                    f"checkpoint[{slot}]",
                    lambda result, slot=slot: result.checkpoint_states[slot],
                    oracle,
                )
            )
        for label, pick, oracle in readings:
            baseline.setdefault(label, []).append(
                _p6_metrics(pick(results["three_phase"]), oracle)
            )
            for mode, metrics in candidates.items():
                metrics.setdefault(label, []).append(
                    _p6_metrics(pick(results[mode]), oracle)
                )
    for mode, program in (("never", "exact scan"), ("always", "look-back")):
        for label in baseline:
            _assert_p6_clause(
                candidates[mode][label], baseline[label], f"{program} {label}"
            )


def _checkpoint_arguments(arguments, tokens, slots):
    """``arguments`` plus one checkpoint request per sequence: the absolute
    token index a sequence checkpoints at (``-1``: none) and the slot it
    writes in a fresh NaN-filled ``[3, nheads, 64, 128]`` state table, so
    the slots a call must leave untouched stay NaN."""

    initial_states = arguments["initial_states"]
    return {
        **arguments,
        "checkpoint_token_indices": torch.tensor(
            tokens, dtype=torch.int32, device="cuda"
        ),
        "checkpoint_state_slots": torch.tensor(slots, dtype=torch.int32, device="cuda"),
        "checkpoint_states": torch.full(
            (3, *initial_states.shape[1:]),
            torch.nan,
            dtype=initial_states.dtype,
            device="cuda",
        ),
    }


@pytest.mark.parametrize(
    "state_dtype,varlen,form,lengths",
    (
        (torch.bfloat16, False, "batched", None),
        (torch.float16, False, "batched", None),
        (torch.float32, False, "batched", None),
        (torch.bfloat16, True, "triple", (96, 160)),
        (torch.bfloat16, True, "triple", (128, 896)),
        (torch.float16, True, "cu_seqlens", (1024, 100)),
        (torch.float32, True, "cu_seqlens", (96, 160)),
    ),
    ids=(
        "bf16_batched",
        "f16_batched",
        "f32_batched",
        "bf16_triple_96x160",
        "bf16_triple_128x896",
        "f16_cu_seqlens_1024x100",
        "f32_cu_seqlens_96x160",
    ),
)
def test_cake_ssd_combined_chunk_parallel_is_bitwise_the_exact_scan(
    monkeypatch, state_dtype, varlen, form, lengths
):
    """The chunk-parallel programs compute the exact scan's arithmetic in
    grid-barrier-separated schedules: forced on, the three-phase program is
    the round-5 reading and the forced exact scan's and the look-back's
    output and final states meet the P6 clause against it (round 6: the exact
    scan rounds differently and the look-back re-associates the carried
    state, bitwise equality is no longer the contract) -- batched calls with
    z, D, dt_bias, softplus, a finite clamp and initial states, and
    packed-varlen calls in both metadata forms, for every state dtype."""

    _skip_unless_cake_arch()
    if varlen:
        constructor, tensors, arguments = _case(
            state_dtype=state_dtype, varlen=True, lengths=lengths
        )
        if form == "cu_seqlens":
            arguments = _cu_seqlens_arguments(arguments, lengths)
    else:
        constructor, tensors, arguments = _case(
            state_dtype=state_dtype, batch=2, seqlen=1024
        )
    runner = SSDCombined(**constructor, backend="cake")

    results = _forced_runs(monkeypatch, runner, tensors, arguments)

    exact, chunk_parallel = results["never"], results["always"]
    three_phase = results["three_phase"]
    mode = "varlen" if varlen else "batched"
    # forced exact scan: the fused program where its preprocess grid is 0
    # (batched, cu_seqlens form, and the triple form whose caller cumsum is
    # trusted -- ``_case`` precomputes it -- inside the fused window); the
    # triple form deriving its table runs ``exactmeta`` inside the window
    # and the two-kernel program outside it (exactpre, or the retained start
    # program for one-chunk items).
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    if form == "triple":
        fused = module.fused_selected(
            nheads=8,
            num_sequences=len(lengths),
            nchunks=-(-max(lengths) // 128),
            mode_varlen=True,
            total_tokens=sum(lengths),
        )
        family = (
            "exactfused"
            if fused
            else module.two_kernel_family(
                mode_varlen=True, num_sequences=len(lengths), seqlen=sum(lengths)
            )
        )
    else:
        family = "exactfused"
    assert exact.program == f"{family}_{_state_key(state_dtype)}_{mode}"
    assert chunk_parallel.program == f"chunkpar_{_state_key(state_dtype)}_{mode}"
    assert three_phase.program == f"chunkpar3_{_state_key(state_dtype)}_{mode}"
    assert chunk_parallel.final.dtype == state_dtype
    assert torch.isfinite(chunk_parallel.out.to(torch.float32)).all()
    assert three_phase.final.dtype == state_dtype
    assert torch.isfinite(three_phase.out.to(torch.float32)).all()

    def make_case(seed):
        if varlen:
            case_constructor, case_tensors, case_arguments = _case(
                state_dtype=state_dtype, varlen=True, lengths=lengths, seed=seed
            )
            oracle_arguments = {
                **case_arguments,
                "seq_idx": _varlen_metadata(lengths, torch.int32)[0],
                "num_seqs": len(lengths),
            }
            if form == "cu_seqlens":
                case_arguments = _cu_seqlens_arguments(case_arguments, lengths)
        else:
            case_constructor, case_tensors, case_arguments = _case(
                state_dtype=state_dtype, batch=2, seqlen=1024, seed=seed
            )
            oracle_arguments = case_arguments
        return case_constructor, case_tensors, case_arguments, oracle_arguments

    _assert_chunk_parallel_p6(monkeypatch, make_case)


@pytest.mark.parametrize(
    "state_dtype,form,lengths,tokens,slots",
    (
        (torch.bfloat16, "batched", None, (384, 1024), (0, 2)),
        (torch.float16, "batched", None, (768, -1), (1, -1)),
        (torch.float32, "triple", (300, 700), (-1, 556), (-1, 2)),
        (torch.bfloat16, "cu_seqlens", (300, 700), (-1, 556), (-1, 2)),
        (torch.float16, "cu_seqlens", (1000,), (768,), (1,)),
    ),
    ids=(
        "bf16_batched_384_and_sequence_end",
        "f16_batched_768",
        "f32_triple_300x700_at_556",
        "bf16_cu_seqlens_300x700_at_556",
        "f16_cu_seqlens_1000_on_grid_768",
    ),
)
def test_cake_ssd_combined_chunk_parallel_checkpoints_are_bitwise_the_exact_scan(
    monkeypatch, state_dtype, form, lengths, tokens, slots
):
    """The chunk-parallel programs' checkpoint publish is the exact scan's:
    forced on, the state the three-phase program writes into each requested
    slot is the round-5 reading, the forced exact scan's and the look-back's
    meet the P6 clause against it (output and final states included; round
    6: the exact scan rounds differently and the look-back re-associates the
    carried state, bitwise equality is no longer the contract) and the other
    slots stay untouched -- batched
    checkpoints on the chunk grid and at a sequence end, a chunk-unaligned
    checkpoint exposed through the caller's triple and one the cu_seqlens
    preprocess inserts itself, and an on-grid cu_seqlens checkpoint.
    ``auto`` routes checkpoint calls like any other call, so the
    chunk-parallel path must serve them exactly."""

    _skip_unless_cake_arch()

    def make_case(seed):
        if form == "batched":
            case_constructor, case_tensors, case_arguments = _case(
                state_dtype=state_dtype, batch=2, seqlen=1024, seed=seed
            )
            oracle_arguments = case_arguments
        else:
            case_constructor, case_tensors, case_arguments = _case(
                state_dtype=state_dtype, varlen=True, lengths=lengths, seed=seed
            )
            oracle_arguments = {
                **case_arguments,
                "seq_idx": _varlen_metadata(lengths, torch.int32)[0],
                "num_seqs": len(lengths),
            }
            if form == "cu_seqlens":
                case_arguments = _cu_seqlens_arguments(case_arguments, lengths)
            else:
                # The triple must expose every checkpoint as a logical chunk
                # end; the runner derives the sequence prefix sum on the device.
                chunk_indices, chunk_offsets = _logical_chunk_metadata(
                    lengths, tuple(token for token in tokens if token >= 0)
                )
                case_arguments = {
                    **case_arguments,
                    "chunk_indices": chunk_indices,
                    "chunk_offsets": chunk_offsets,
                    "seq_chunk_cumsum": None,
                }
        case_arguments = _checkpoint_arguments(case_arguments, tokens, slots)
        return case_constructor, case_tensors, case_arguments, oracle_arguments

    constructor, tensors, arguments, _ = make_case(7)
    runner = SSDCombined(**constructor, backend="cake")

    results = _forced_runs(monkeypatch, runner, tensors, arguments)

    exact, chunk_parallel = results["never"], results["always"]
    three_phase = results["three_phase"]
    mode = "batched" if form == "batched" else "varlen"
    # The triple deriving its table (``make_case`` passes no caller table):
    # the program of the device's capability inside the fused window
    # (``_derived_family``), the two-kernel program outside it; every other
    # form runs the fused single launch.
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    if form == "triple":
        family = (
            _derived_family(module, lengths)
            if module.fused_selected(
                nheads=8,
                num_sequences=len(lengths),
                nchunks=-(-max(lengths) // 128),
                mode_varlen=True,
                total_tokens=sum(lengths),
            )
            else module.two_kernel_family(
                mode_varlen=True, num_sequences=len(lengths), seqlen=sum(lengths)
            )
        )
    else:
        family = "exactfused"
    exact_program = exact.program
    if form == "triple":
        exact_program = _without_table_form(
            module, exact_program, state_dtype=state_dtype, num_sequences=len(lengths)
        )
    assert exact_program == f"{family}_{_state_key(state_dtype)}_{mode}"
    assert chunk_parallel.program == f"chunkpar_{_state_key(state_dtype)}_{mode}"
    assert three_phase.program == f"chunkpar3_{_state_key(state_dtype)}_{mode}"
    written = {slot for slot in slots if slot >= 0}
    for slot in range(3):
        exact_slot = exact.checkpoint_states[slot]
        chunk_parallel_slot = chunk_parallel.checkpoint_states[slot]
        three_phase_slot = three_phase.checkpoint_states[slot]
        if slot in written:
            assert torch.isfinite(exact_slot.to(torch.float32)).all(), slot
            assert torch.isfinite(chunk_parallel_slot.to(torch.float32)).all(), slot
            assert torch.isfinite(three_phase_slot.to(torch.float32)).all(), slot
        else:
            assert torch.isnan(exact_slot).all(), slot
            assert torch.isnan(chunk_parallel_slot).all(), slot
            assert torch.isnan(three_phase_slot).all(), slot
    # Each written slot's oracle: the fp64 state after the sequence's prefix
    # up to its checkpoint token (batched: a per-row token index; packed
    # varlen: an absolute token index on the token axis).
    checkpoints = []
    for sequence, (token, slot) in enumerate(zip(tokens, slots, strict=True)):
        if slot < 0:
            continue
        start = 0 if form == "batched" else sum(lengths[:sequence])
        checkpoints.append((slot, sequence, start, token))
    _assert_chunk_parallel_p6(monkeypatch, make_case, checkpoints=tuple(checkpoints))


def test_cake_ssd_combined_auto_selects_chunk_parallel_for_long_prefill(monkeypatch):
    """``auto``: one 32768-token sequence with 8 heads (8 work items, 2048
    tiles) selects the chunk-parallel program on the device's own SM count
    and capability, reports it as ``last_program_name`` and meets the P6
    clause against the forced exact scan (the two-kernel ``exactpre``
    program: 256 chunks per sequence exceed the fused window)."""

    _skip_unless_cake_arch()
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    constructor, tensors, arguments = _case(batch=1, seqlen=32768, nheads=8, ngroups=8)
    device = tensors[0].device
    assert module.chunk_parallel_selected(
        nheads=8,
        num_sequences=1,
        num_segments=256,
        nchunks=256,
        mode_varlen=False,
        sm_count=torch.cuda.get_device_properties(device).multi_processor_count,
        capability=tuple(torch.cuda.get_device_capability(device)),
        mode="auto",
    )
    runner = SSDCombined(**constructor, backend="cake")
    cake = runner._cake_runner
    assert cake.last_program_name is None

    out, final = (value.clone() for value in runner.run(*tensors, **arguments))

    assert cake.last_program_name == "chunkpar_bf16_batched"
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "never")
    out_exact, final_exact = runner.run(*tensors, **arguments)
    assert cake.last_program_name == "exactpre_bf16_batched"
    assert out_exact.shape == out.shape and final_exact.shape == final.shape

    def make_case(seed):
        case = _case(batch=1, seqlen=32768, nheads=8, ngroups=8, seed=seed)
        return (*case, case[2])

    _assert_chunk_parallel_p6(monkeypatch, make_case)


def test_cake_ssd_combined_auto_selects_three_phase_for_a_short_few_head_prefill(
    monkeypatch,
):
    """``auto``: one 2048-token sequence with 8 heads (8 work items, 16
    chunks, 128 tiles) is the three-phase region of the family rule: the
    runner selects the ``chunkpar3`` program on the device's own SM count and
    capability, reports it as ``last_program_name``, and the forced exact
    scan and look-back of the same call meet the P6 clause against it (the
    round-5 reading)."""

    _skip_unless_cake_arch()
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    constructor, tensors, arguments = _case(batch=1, seqlen=2048, nheads=8, ngroups=8)
    device = tensors[0].device
    assert (
        module.chunk_parallel_family(
            nheads=8,
            num_sequences=1,
            num_segments=16,
            nchunks=16,
            mode_varlen=False,
            sm_count=torch.cuda.get_device_properties(device).multi_processor_count,
            capability=tuple(torch.cuda.get_device_capability(device)),
            mode="auto",
        )
        == "chunkpar3"
    )
    runner = SSDCombined(**constructor, backend="cake")
    cake = runner._cake_runner
    assert cake.last_program_name is None

    out, final = (value.clone() for value in runner.run(*tensors, **arguments))

    assert cake.last_program_name == "chunkpar3_bf16_batched"
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "never")
    out_exact, final_exact = runner.run(*tensors, **arguments)
    assert cake.last_program_name == "exactfused_bf16_batched"
    assert out_exact.shape == out.shape and final_exact.shape == final.shape
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "always")
    runner.run(*tensors, **arguments)
    assert cake.last_program_name == "chunkpar_bf16_batched"

    def make_case(seed):
        case = _case(batch=1, seqlen=2048, nheads=8, ngroups=8, seed=seed)
        return (*case, case[2])

    _assert_chunk_parallel_p6(monkeypatch, make_case)


def test_cake_ssd_combined_auto_keeps_the_exact_scan_for_many_heads(monkeypatch):
    """``auto``: 128 heads on a 1024-token sequence (128 work items, 1024
    tiles) stay on the exact scan -- at that head count the chunk-parallel
    per-tile cost exceeds the scan's per-chunk cost."""

    _skip_unless_cake_arch()
    constructor, tensors, arguments = _case(batch=1, seqlen=1024, nheads=128, ngroups=8)
    runner = SSDCombined(**constructor, backend="cake")

    out, final = runner.run(*tensors, **arguments)

    assert runner._cake_runner.last_program_name == "exactfused_bf16_batched"
    assert tuple(out.shape) == (1, 1024, 128, 64)
    assert tuple(final.shape) == (1, 128, 64, 128)


# ---- exact-scan schedule variant (CAKE-956 round 5 host selection) -------------------------------------------

_VARIANT_CASES = (
    # (mode_varlen, has_z, nheads, num_sequences, total_tokens, batch, seqlen, state dtype) -> variant;
    # kernel-level calibration cells (ring / start, GB300 / B200) unless marked FI (the FI-level window
    # gate and the 93-row gate rows, which decide the bf16 2 x 64 and 3 x 64 cells)
    # nem_v8x64 (1.019 / 1.013): the gate's losing row
    ((True, True, 128, 8, 8 * 64, 1, 8 * 64, "bfloat16"), "noring"),
    # v64x64z 1.008 / 1.002
    ((True, True, 128, 64, 64 * 64, 1, 64 * 64, "bfloat16"), "noring"),
    # g6x64z 1.009 / 1.006
    ((True, True, 128, 6, 6 * 64, 1, 6 * 64, "bfloat16"), "noring"),
    # g4x64z tie (1.000-1.003 / 0.999-1.002): start
    ((True, True, 128, 4, 4 * 64, 1, 4 * 64, "bfloat16"), "noring"),
    # FI win_v3x64_bfloat16 0.996 / 0.989 (tie): start
    ((True, True, 128, 3, 3 * 64, 1, 3 * 64, "bfloat16"), "noring"),
    # FI nem_v64_64_bfloat16 1.011-1.018 / 0.997: start
    ((True, True, 128, 2, 2 * 64, 1, 2 * 64, "bfloat16"), "noring"),
    # g1x64z 0.937 / 0.943; FI 0.937 / 0.940
    ((True, True, 128, 1, 64, 1, 64, "bfloat16"), ""),
    # FI win_v3x64_float32 0.978 / 0.974
    ((True, True, 128, 3, 3 * 64, 1, 3 * 64, "float32"), ""),
    # FI nem_v64_64_float32 0.978-1.003 / 0.977
    ((True, True, 128, 2, 2 * 64, 1, 2 * 64, "float32"), ""),
    # FI win_v1x64_float32 0.939 / 0.941
    ((True, True, 128, 1, 64, 1, 64, "float32"), ""),
    # round 6 (prescale-table programs): f32 g4x64z 0.993 / g6x64z 0.983 / g8x64z 0.997 ring wins -> an f32
    # state never takes the varlen start schedule (GB300 / B200 re-calibration)
    ((True, True, 128, 4, 4 * 64, 1, 4 * 64, "float32"), ""),
    (
        (True, True, 128, 8, 8 * 64, 1, 8 * 64, "float32"),
        "",
    ),  # nem_v8x64_float32 0.997 (tie -> ring: rule closed)
    ((True, True, 128, 64, 64 * 64, 1, 64 * 64, "float32"), ""),  # g64x64z f32 0.981
    # f16 state follows bf16 (same state width)
    ((True, True, 128, 2, 2 * 64, 1, 2 * 64, "float16"), "noring"),
    # g8x63z 0.974 / 0.976 (window edge)
    ((True, True, 128, 8, 8 * 63, 1, 8 * 63, "bfloat16"), ""),
    # g8x65z 0.979 / 0.963 (window edge)
    ((True, True, 128, 8, 8 * 65, 1, 8 * 65, "bfloat16"), ""),
    ((True, True, 128, 8, 8 * 60, 1, 8 * 60, "bfloat16"), ""),  # g8x60z 0.961 / 0.960
    # [63, 65] x 4: 1.008 / 1.007
    ((True, True, 128, 8, 4 * (63 + 65), 1, 512, "bfloat16"), "noring"),
    # [32, 96] x 4: 1.006 / 1.005
    ((True, True, 128, 8, 4 * (32 + 96), 1, 512, "bfloat16"), "noring"),
    # 64.5 tokens on average: inside the window, start
    ((True, True, 128, 8, 516, 1, 516, "bfloat16"), "noring"),
    # g5x64z tie (1.001 / 0.994): start
    ((True, True, 128, 5, 5 * 64, 1, 5 * 64, "bfloat16"), "noring"),
    ((True, True, 128, 8, 8 * 32, 1, 8 * 32, "bfloat16"), ""),  # g8x32z 0.988 / 0.985
    # g8x128z 0.915 / 0.917
    ((True, True, 128, 8, 8 * 128, 1, 8 * 128, "bfloat16"), ""),
    # nem_v96p160 0.964 / 0.960
    ((True, True, 128, 2, 96 + 160, 1, 256, "bfloat16"), ""),
    # nem_v64x128 0.901 / 0.906
    ((True, True, 128, 64, 64 * 128, 1, 64 * 128, "bfloat16"), ""),
    # g8x64 (no z) 0.992 / 0.990
    ((True, False, 128, 8, 8 * 64, 1, 8 * 64, "bfloat16"), ""),
    # short_varlen_64_60 (no z) 1.000 / 1.003
    ((True, False, 128, 2, 124, 1, 124, "bfloat16"), ""),
    # 64 work items (below both thresholds): ring
    ((True, True, 8, 8, 8 * 64, 1, 8 * 64, "bfloat16"), ""),
    # 256 work items at 64 tokens, 16-bit state
    ((True, True, 8, 32, 32 * 64, 1, 32 * 64, "bfloat16"), "noring"),
    # 256 work items at 64 tokens, f32 state: ring
    ((True, True, 8, 32, 32 * 64, 1, 32 * 64, "float32"), ""),
    # 512 work items at 64 tokens, f32 state: ring (round-6 f32 window closed)
    ((True, True, 8, 64, 64 * 64, 1, 64 * 64, "float32"), ""),
    # short_batched_2x50 1.008 / 1.005 (round 5); round 6 (prescale-table programs): GB300 wins 0.987-0.995
    # but B200 ties 0.998-1.002 -> not a two-arch win: the batched window stays
    ((False, False, 128, 2, 100, 2, 50, "bfloat16"), "noring"),
    (
        (False, False, 128, 2, 150, 2, 75, "bfloat16"),
        "noring",
    ),  # gb2x75 1.005 / 0.993; round 6 0.987 / 0.995
    # gb2x60 1.000 / 1.000 (tie): start
    ((False, False, 128, 2, 120, 2, 60, "bfloat16"), "noring"),
    ((False, False, 128, 2, 50, 2, 25, "bfloat16"), ""),  # gb2x25 0.990 / 0.991
    ((False, False, 128, 2, 180, 2, 90, "bfloat16"), ""),  # gb2x90 0.987 / 0.995
    ((False, False, 128, 2, 200, 2, 100, "bfloat16"), ""),  # gb2x100 0.995 / 0.991
    ((False, False, 128, 1, 50, 1, 50, "bfloat16"), ""),  # gb1x50 0.970 / 0.955
    ((False, False, 128, 3, 150, 3, 50, "bfloat16"), ""),  # gb3x50 0.992 / 0.986
    # gb2x50z 1.002 / 0.987 (z: ring)
    ((False, True, 128, 2, 100, 2, 50, "bfloat16"), ""),
    ((False, False, 128, 4, 200, 4, 50, "bfloat16"), ""),  # gb4x50 0.973 / 0.960
    # nem_b1x1000 0.925 / 0.911
    ((False, False, 128, 1, 1000, 1, 1000, "bfloat16"), ""),
    # gb64x128z 0.898 / 0.908
    ((False, True, 128, 64, 64 * 128, 64, 128, "bfloat16"), ""),
    # batched rule is dtype-independent (round 6: f32 gb2x50 0.982 GB300 / 0.995 B200 -> stays)
    ((False, False, 128, 2, 100, 2, 50, "float32"), "noring"),
)


@pytest.mark.parametrize("args,expected", _VARIANT_CASES)
def test_source_exact_scan_variant_rule_cells(args, expected):
    """The host rule sends every calibrated losing cell to the start schedule and
    every winning cell to the ring program (CAKE-956 round 5 calibration table in
    the loader); it reads host-known shape quantities only."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    (
        mode_varlen,
        has_z,
        nheads,
        num_sequences,
        total_tokens,
        batch,
        seqlen,
        state_dtype,
    ) = args
    assert (
        module._exact_scan_variant(
            mode_varlen=mode_varlen,
            has_z=has_z,
            nheads=nheads,
            num_sequences=num_sequences,
            total_tokens=total_tokens,
            batch=batch,
            seqlen=seqlen,
            state_dtype=getattr(torch, state_dtype),
        )
        == expected
    )


def test_source_exact_scan_variant_programs():
    """Both exact-scan variants of every (state, mode) the variant rule can
    select (never the two-kernel family, whose items span at least two chunks;
    never an f32 state on the varlen path) are programs of the same family
    (same host template and argument order); the ``_noring`` programs
    carry their own main dynamic SMEM literal."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    smem = {
        "exact": (module._EXACT_SMEM_BYTES, module._EXACT_NORING_SMEM_BYTES),
        "exactfused": (
            module._EXACTFUSED_SMEM_BYTES,
            module._EXACTFUSED_NORING_SMEM_BYTES,
        ),
    }
    for family, (ring_smem, noring_smem) in smem.items():
        for state_dtype in (torch.bfloat16, torch.float16, torch.float32):
            for mode_varlen in (False, True):
                ring = module._program_name(family, state_dtype, mode_varlen)
                noring = module._program_name(
                    family, state_dtype, mode_varlen, "noring"
                )
                assert noring == ring + "_noring"
                if mode_varlen and state_dtype is torch.float32:
                    assert noring not in module._PROGRAMS
                    continue
                a, b = module._PROGRAMS[ring], module._PROGRAMS[noring]
                assert a.family == b.family == family
                assert a.host_template == b.host_template and a.main_args == b.main_args
                assert (a.state_dtype_code, a.state_dtype_bits) == (
                    b.state_dtype_code,
                    b.state_dtype_bits,
                )
                assert a.main_smem_bytes == ring_smem
                assert b.main_smem_bytes == noring_smem
                assert a.main.module != b.main.module
    # the retained and the fused family: six ring programs and the five
    # ``_noring`` programs the variant rule can select; the fused family also
    # the two sm_107a-only ``_ibulk`` programs
    assert len([name for name in module._PROGRAMS if name.startswith("exact_")]) == 11
    assert (
        len([name for name in module._PROGRAMS if name.startswith("exactfused_")]) == 13
    )
    # the two-kernel family: six ring and six ``_tail`` programs, no ``_noring``,
    # and the sm_107a-only programs: two ``_claim``, two ``_ibulk`` and three
    # ``_tailone``
    assert (
        len([name for name in module._PROGRAMS if name.startswith("exactpre_")]) == 19
    )
    assert not any(name.endswith("_noring") for name in module._EXACTPRE_MODULES)
    assert len([name for name in module._PROGRAMS if name.startswith("chunkpar_")]) == 6
    assert (
        len([name for name in module._PROGRAMS if name.startswith("chunkpar3_")]) == 6
    )
    # The exactmeta programs: the five exactfused varlen main kernels (ring and
    # ``_noring``) behind the packed-varlen metadata stage.
    assert (
        len([name for name in module._PROGRAMS if name.startswith("exactmeta_")]) == 5
    )
    # The exactone programs: the SM103 render of the same five fused varlen
    # entries (one launch), their own device modules, and the sm_107a-only
    # programs in the tablefirst / tableroles prologue forms.
    assert (
        len([name for name in module._PROGRAMS if name.startswith("exactone_")]) == 11
    )
    assert len(module._PROGRAMS) == 71


@pytest.mark.parametrize(
    "case",
    (
        "varlen_8x64_z",
        "varlen_8x128_z",
        "varlen_8x64_noz",
        "batched_2x50",
        "batched_2x128",
        "batched_2x50_z",
    ),
)
def test_source_exact_scan_variant_selected_without_gpu(monkeypatch, case):
    """The plan binds the variant the host rule selects from the call shape
    (``cu_seqlens.numel() - 1`` sequences, ``x.shape`` tokens, the runner's
    ``has_z``) -- the same program for :meth:`run` and :meth:`prepare`."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    varlen = case.startswith("varlen")
    has_z = case.endswith("_z")
    runner = _cpu_forwarding_runner(
        module,
        monkeypatch,
        calls,
        nheads=128,
        ngroups=8,
        has_varlen=varlen,
        has_initial_states=True,
        has_z=has_z,
    )
    if varlen:
        tokens = {"varlen_8x64_z": 64, "varlen_8x128_z": 128, "varlen_8x64_noz": 64}[
            case
        ]
        num_seqs, batch, seqlen = 8, 1, 8 * tokens
        cu_seqlens = torch.arange(0, (num_seqs + 1) * tokens, tokens, dtype=torch.int32)
        suffix = "_noring" if tokens == 64 and has_z else ""
    else:
        num_seqs = batch = 2
        seqlen = {"batched_2x50": 50, "batched_2x128": 128, "batched_2x50_z": 50}[case]
        cu_seqlens = None
        suffix = "_noring" if seqlen == 50 and not has_z else ""
    # 8 packed sequences x 128 heads = 1024 items exceed the fused varlen window
    # (256 items): those calls run the two-kernel form -- one-chunk items, so the
    # retained start ``exact`` program; the batched calls (256 items, one chunk)
    # run the fused ``exactfused`` program.
    fused = module.fused_selected(
        nheads=128,
        num_sequences=num_seqs,
        nchunks=-(-seqlen // 128),
        mode_varlen=varlen,
        total_tokens=batch * seqlen,
    )
    assert fused == (not varlen)
    two_kernel = module.two_kernel_family(
        mode_varlen=varlen,
        num_sequences=num_seqs,
        seqlen=batch * seqlen if varlen else seqlen,  # varlen: the packed token count
    )
    assert two_kernel == "exact"  # 64- and 128-token sequences: one chunk per item
    expected = (
        f"{'exactfused' if fused else two_kernel}_bf16_{'varlen' if varlen else 'batched'}"
        + suffix
    )
    x = torch.empty((batch, seqlen, 128, 64), dtype=torch.bfloat16)
    dt = torch.empty((batch, seqlen, 128), dtype=torch.float32)
    A = torch.empty((128,), dtype=torch.float32)
    B = torch.empty((batch, seqlen, 8, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)
    kwargs = {
        "initial_states": torch.empty((num_seqs, 128, 64, 128), dtype=torch.bfloat16)
    }
    if has_z:
        kwargs["z"] = torch.empty_like(x)
    if varlen:
        kwargs["cu_seqlens"] = cu_seqlens
    runner.run(x, dt, A, B, C, **kwargs)
    ((name, _launch),) = calls
    assert name == expected
    monkeypatch.setattr(module, "_load_generated_program", lambda *_: None)
    prepared = runner.prepare(x=x, dt=dt, A=A, B=B, C=C, **kwargs)
    assert prepared.program_name == expected


def test_source_sm107_fp32_one_chunk_items_above_the_bound_take_exactpre():
    """On SM107 one-chunk ring calls with an fp32 state take the two-kernel
    ``exactpre`` program above the 2048-item bound too; 16-bit states, the
    start schedule and other architectures keep the retained ``exact``
    program there."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")

    def family(capability, state_dtype, num_sequences=160, variant=""):
        # 128-token sequences x 128 heads: one chunk per item
        return module.two_kernel_family(
            mode_varlen=True,
            num_sequences=num_sequences,
            seqlen=num_sequences * 128,
            nheads=128,
            sm_count=148,
            capability=capability,
            variant=variant,
            state_dtype=state_dtype,
        )

    # 160 x 128 = 20480 items, above the bound
    assert family((10, 7), torch.float32) == "exactpre"
    assert family((10, 7), torch.bfloat16) == "exact"
    assert family((10, 7), torch.float16) == "exact"
    assert family((10, 7), torch.float32, variant="noring") == "exact"
    assert family((10, 3), torch.float32) == "exact"
    assert family((10, 0), torch.float32) == "exact"
    # 8 x 128 = 1024 items, inside the bound: every state dtype
    assert family((10, 7), torch.bfloat16, num_sequences=8) == "exactpre"
    # 1 x 128 = 128 items, not above the SM count
    assert family((10, 7), torch.float32, num_sequences=1) == "exact"


def test_source_sm107_claim_rule():
    """``claim_selected``: on SM107 a packed-varlen ring call of one chunk per
    item with more than 2048 (sequence, head) items and at least 8 of them per
    CTA of its persistent grid, with a bf16 or fp32 state, runs the claim
    program; f16 states, batched calls, multi-chunk items, the start and
    pair schedules and other architectures do not."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    assert module._CLAIM_MIN_ITEMS == {(10, 7): 2048}
    assert module._CLAIM_MIN_ITEMS_PER_CTA == 8

    def claim(
        num_sequences=64,
        tokens=128,
        capability=(10, 7),
        state_dtype=torch.bfloat16,
        variant="",
        mode_varlen=True,
        sm_count=212,
    ):
        return module.claim_selected(
            mode_varlen=mode_varlen,
            num_sequences=num_sequences,
            seqlen=num_sequences * tokens,
            nheads=128,
            sm_count=sm_count,
            capability=capability,
            variant=variant,
            state_dtype=state_dtype,
        )

    # 64 / 160 x 128 = 8192 / 20480 one-chunk items
    assert claim()
    assert claim(num_sequences=160)
    assert claim(state_dtype=torch.float32)
    assert not claim(state_dtype=torch.float16)
    # 16 x 128 = 2048 items: not above the bound; 17 x 128 = 2176 is
    assert not claim(num_sequences=16)
    assert claim(num_sequences=17)
    # fewer than 8 items per CTA of the persistent grid (1024 SMs: grid 1024)
    assert not claim(num_sequences=17, sm_count=1024)
    # a mean of 129 tokens per sequence: two chunks per item
    assert not claim(tokens=129)
    assert claim(tokens=1)
    assert not claim(variant="noring")
    assert not claim(variant="tail")
    assert not claim(mode_varlen=False)
    assert not claim(capability=(10, 3))
    assert not claim(capability=(10, 0))
    assert not claim(capability=None)
    # the programs it selects are exactly the sm_107a-only claim entries
    assert {
        module._program_name("exactpre", state_dtype, True, "claim")
        for state_dtype in module._CLAIM_STATE_DTYPES[(10, 7)]
    } == {name for name in module._SM107_ONLY_PROGRAMS if name.endswith("_claim")}


def test_source_sm107_exactone_table_rule():
    """``exactone_table_variant``: on SM107 a one-launch ring call takes the
    tablefirst program at a launch grid below 128 CTAs and the
    tableroles program from it on; ``noring`` calls and other
    architectures keep their program."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    assert module._EXACTONE_TABLE_ROLES_MIN_GRID == {(10, 7): 128}

    def variant(grid, capability=(10, 7), current="", state_dtype=torch.bfloat16):
        return module.exactone_table_variant(
            capability=capability, variant=current, state_dtype=state_dtype, grid=grid
        )

    assert variant(grid=1) == "tablefirst"
    assert variant(grid=127) == "tablefirst"
    assert variant(grid=128) == "tableroles"
    assert variant(grid=212) == "tableroles"
    assert variant(grid=8, current="noring") == "noring"
    assert variant(grid=212, current="noring") == "noring"
    assert variant(grid=8, capability=(10, 3)) == ""
    assert variant(grid=8, capability=(10, 0)) == ""
    assert variant(grid=8, capability=None) == ""
    for state_dtype in (torch.bfloat16, torch.float16, torch.float32):
        assert variant(grid=8, state_dtype=state_dtype) == "tablefirst"
        assert variant(grid=212, state_dtype=state_dtype) == "tableroles"
    # the programs it selects are exactly the sm_107a-only one-launch entries
    assert {
        module._program_name("exactone", state_dtype, True, form)
        for state_dtype in module._EXACTONE_TABLE_STATE_DTYPES
        for form in ("tablefirst", "tableroles")
    } == {name for name in module._SM107_ONLY_PROGRAMS if name.startswith("exactone_")}


@pytest.mark.parametrize("arch", ("sm_103a", "sm_107a"))
@pytest.mark.parametrize("state_dtype", (torch.bfloat16, torch.float16, torch.float32))
def test_source_sm107_many_one_chunk_items_take_the_claim_program(
    monkeypatch, arch, state_dtype
):
    """On SM107 a packed-varlen call of 64 128-token sequences x 128 heads
    (8192 one-chunk items, beyond the fused window and the 2048-item one-chunk
    bound) runs the claim program of a bf16 or fp32 state, which launches its
    main kernel with programmatic dependent launch; an f16 state and SM103 keep
    the retained ``exact`` program (fp32 on SM103 too)."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    runner = _cpu_forwarding_runner(
        module,
        monkeypatch,
        calls,
        nheads=128,
        ngroups=8,
        has_varlen=True,
        has_initial_states=True,
        state_dtype=state_dtype,
        arch=arch,
    )
    num_seqs, tokens = 64, 128
    seqlen = num_seqs * tokens
    x = torch.empty((1, seqlen, 128, 64), dtype=torch.bfloat16)
    dt = torch.empty((1, seqlen, 128), dtype=torch.float32)
    A = torch.empty((128,), dtype=torch.float32)
    B = torch.empty((1, seqlen, 8, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)
    kwargs = {
        "initial_states": torch.empty((num_seqs, 128, 64, 128), dtype=state_dtype),
        "cu_seqlens": torch.arange(
            0, (num_seqs + 1) * tokens, tokens, dtype=torch.int32
        ),
    }
    runner.run(x, dt, A, B, C, **kwargs)
    ((name, launch),) = calls
    state = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}[
        state_dtype
    ]
    if arch == "sm_107a" and state_dtype != torch.float16:
        assert name == f"exactpre_{state}_varlen_claim"
        assert module._main_pdl(name, arch)
        program = module._arch_program(name, arch)
        assert program.variant == "claim"
        assert program.main_smem_bytes == module._EXACTPRE_CLAIM_SMEM_BYTES
        assert program.main.kernel == (
            f"kernel_mamba_ssd_q_tmem_alias_{state}_varlen_precomputed_claim"
        )
        status = launch["main"]["preprocess_status"]
        assert tuple(status.shape) == (2,)
        assert launch["preprocess"]["preprocess_status"] is status
    else:
        assert name == f"exact_{state}_varlen"


@pytest.mark.parametrize("arch", ("sm_103a", "sm_107a"))
@pytest.mark.parametrize(
    ("num_seqs", "sm107_program"),
    (
        # 3 / 6 x 128 = 384 / 768 items: at most four items per SM (212 SMs)
        (3, "exactfused_bf16_varlen_noring"),
        (6, "exactfused_bf16_varlen_noring"),
        # 7 x 128 = 896 and 8 x 128 = 1024 items: the ring exactpre program
        (7, "exactpre_bf16_varlen"),
        (8, "exactpre_bf16_varlen"),
        # 24 x 128 = 3072 items: the ring two-kernel family is the retained
        # exact program (above the 2048-item one-chunk bound)
        (24, "exactfused_bf16_varlen_noring"),
    ),
)
def test_source_sm107_retained_noring_call_takes_the_fused_program(
    monkeypatch, arch, num_seqs, sm107_program
):
    """On SM107 a call the retained ``exact_*_noring`` program would serve
    (packed 64-token sequences x 128 heads with z: one-chunk items outside
    the fused varlen window, on the start schedule) runs the ring
    ``exactpre`` program where it has more than four items per SM and its
    ring two-kernel family is ``exactpre``, else the fused program of the
    same variant; SM103 keeps the retained program."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    runner = _cpu_forwarding_runner(
        module,
        monkeypatch,
        calls,
        nheads=128,
        ngroups=8,
        has_varlen=True,
        has_initial_states=True,
        has_z=True,
        arch=arch,
    )
    monkeypatch.setattr(module, "_sm_count", lambda _: 212)
    tokens = 64
    seqlen = num_seqs * tokens
    x = torch.empty((1, seqlen, 128, 64), dtype=torch.bfloat16)
    dt = torch.empty((1, seqlen, 128), dtype=torch.float32)
    A = torch.empty((128,), dtype=torch.float32)
    B = torch.empty((1, seqlen, 8, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)
    kwargs = {
        "initial_states": torch.empty((num_seqs, 128, 64, 128), dtype=torch.bfloat16),
        "z": torch.empty_like(x),
        "cu_seqlens": torch.arange(
            0, (num_seqs + 1) * tokens, tokens, dtype=torch.int32
        ),
    }
    runner.run(x, dt, A, B, C, **kwargs)
    ((name, _launch),) = calls
    expected = sm107_program if arch == "sm_107a" else "exact_bf16_varlen_noring"
    assert name == expected


def test_source_sm107_fused_window_extension_without_z():
    """On SM107 calls without z whose fused program needs no stage take the
    fused program beyond the seed window: batched one-chunk calls up to 2048
    items, packed-varlen calls up to 512 items within 512 packed
    chunk-items; other capabilities and larger calls keep the seed window."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    assert module._FUSED_NO_Z_MAX_ITEMS == {(10, 7): (2048, 512)}

    def ext(capability, num_sequences, chunks, mode_varlen, nheads=8):
        return module.fused_no_z_extended(
            capability=capability,
            nheads=nheads,
            num_sequences=num_sequences,
            nchunks=chunks if not mode_varlen else num_sequences * chunks,
            mode_varlen=mode_varlen,
            total_tokens=num_sequences * chunks * 128,
        )

    # batched: 256 x 8 heads = 2048 one-chunk items (perf_batched_256x1)
    assert ext((10, 7), 256, 1, False)
    assert ext((10, 7), 136, 1, False)
    assert not ext((10, 7), 257, 1, False)
    assert not ext((10, 7), 192, 2, False)
    # varlen: 64 x 8 heads = 512 one-chunk items (perf_varlen_64x1)
    assert ext((10, 7), 64, 1, True)
    assert ext((10, 7), 4, 1, True, nheads=128)
    assert not ext((10, 7), 65, 1, True)
    # 16 x 8 heads x 8 chunks: 1024 packed chunk-items
    assert not ext((10, 7), 16, 8, True)
    for capability in ((10, 0), (10, 3)):
        assert not ext(capability, 256, 1, False)
        assert not ext(capability, 64, 1, True)


@pytest.mark.parametrize("arch", ("sm_103a", "sm_107a"))
@pytest.mark.parametrize(
    ("form", "has_z", "sm107_family"),
    (
        ("batched", False, "exactfused"),
        ("batched", True, "exactpre"),
        ("caller", False, "exactfused"),
        ("cu_seqlens", False, "exactfused"),
        ("caller", True, "exactpre"),
        # the triple form deriving its table keeps the two-kernel program
        ("derived", False, "exactpre"),
    ),
)
def test_source_sm107_no_z_calls_take_the_fused_program(
    monkeypatch, arch, form, has_z, sm107_family
):
    """On SM107 a call without z outside the seed fused window but inside
    the extension (256 x 8 heads batched one-chunk items; 64 x 8 heads
    packed 128-token sequences) runs the fused program where it needs no
    stage ahead of it; calls with z, the derived-table triple form and
    SM103 keep the two-kernel program."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    varlen = form != "batched"
    runner = _cpu_forwarding_runner(
        module,
        monkeypatch,
        calls,
        nheads=8,
        ngroups=8,
        has_varlen=varlen,
        has_initial_states=varlen,
        has_z=has_z,
        arch=arch,
    )
    monkeypatch.setattr(module, "_sm_count", lambda _: 212)
    num_seqs = 64 if varlen else 256
    batch, seqlen = (1, num_seqs * 128) if varlen else (num_seqs, 128)
    x = torch.empty((batch, seqlen, 8, 64), dtype=torch.bfloat16)
    dt = torch.empty((batch, seqlen, 8), dtype=torch.float32)
    A = torch.empty((8,), dtype=torch.float32)
    B = torch.empty((batch, seqlen, 8, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)
    kwargs = {}
    if has_z:
        kwargs["z"] = torch.empty_like(x)
    if varlen:
        kwargs["initial_states"] = torch.empty(
            (num_seqs, 8, 64, 128), dtype=torch.bfloat16
        )
        if form == "cu_seqlens":
            kwargs["cu_seqlens"] = torch.arange(
                0, (num_seqs + 1) * 128, 128, dtype=torch.int32
            )
        else:
            kwargs["seq_idx"] = torch.arange(
                num_seqs, dtype=torch.int32
            ).repeat_interleave(128)[None]
            kwargs["chunk_indices"] = torch.arange(num_seqs, dtype=torch.int32)
            kwargs["chunk_offsets"] = torch.zeros(num_seqs, dtype=torch.int32)
            if form == "caller":
                kwargs["seq_chunk_cumsum"] = torch.arange(
                    num_seqs + 1, dtype=torch.int32
                )
    runner.run(x, dt, A, B, C, **kwargs)
    ((name, _launch),) = calls
    mode = "varlen" if varlen else "batched"
    if arch == "sm_107a":
        # a packed-varlen call without z of these 512 one-chunk items with
        # initial states runs the initial-state bulk form of its family's ring
        # program (two-kernel: ibulk_selected, a call with z needs 8 items per
        # SM; fused: fused_ibulk_selected)
        ibulk = varlen and not has_z
        assert name == f"{sm107_family}_bf16_{mode}" + ("_ibulk" if ibulk else "")
    else:
        # SM103: the seed window; one-chunk two-kernel calls keep the
        # retained start program there (no SM103 one-chunk exactpre rule)
        assert name == f"exact_bf16_{mode}"


@pytest.mark.parametrize("arch", ("sm_103a", "sm_107a"))
@pytest.mark.parametrize("has_z", (False, True))
def test_source_sm107_long_varlen_items_take_the_tail_pair_program(
    monkeypatch, arch, has_z
):
    """On SM107 a packed-varlen call of 64 sequences x 4096 tokens x 8 heads
    without z (512 items of 32 chunks on 212 SMs: 3 item waves vs 2.5 pair
    rounds) runs the tail-pair program under the long-item varlen penalty;
    with z, and on SM103, it keeps the persistent program."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    runner = _cpu_forwarding_runner(
        module,
        monkeypatch,
        calls,
        nheads=8,
        ngroups=8,
        has_varlen=True,
        has_initial_states=True,
        has_z=has_z,
        arch=arch,
    )
    monkeypatch.setattr(module, "_sm_count", lambda _: 212)
    num_seqs, tokens = 64, 4096
    seqlen = num_seqs * tokens
    x = torch.empty((1, seqlen, 8, 64), dtype=torch.bfloat16)
    dt = torch.empty((1, seqlen, 8), dtype=torch.float32)
    A = torch.empty((8,), dtype=torch.float32)
    B = torch.empty((1, seqlen, 8, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)
    kwargs = {
        "cu_seqlens": torch.arange(0, seqlen + 1, tokens, dtype=torch.int32),
        "initial_states": torch.empty((num_seqs, 8, 64, 128), dtype=torch.bfloat16),
    }
    if has_z:
        kwargs["z"] = torch.empty_like(x)
    runner.run(x, dt, A, B, C, **kwargs)
    ((name, _launch),) = calls
    if arch == "sm_107a" and not has_z:
        assert name == "exactpre_bf16_varlen_tail"
    else:
        assert name == "exactpre_bf16_varlen"


def test_source_sm107_tail_one_cta_rule():
    """``tail_one_cta_selected``: on SM107 a packed-varlen pair-program call
    whose items span fewer host-known chunks than the full-grid admission's
    ``long_chunks_varlen`` runs the one-CTA pair program; batched calls,
    longer items and other capabilities keep the 2-CTA pair program."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    assert frozenset({(10, 7)}) == module._TAIL_ONE_CTA_CAPABILITIES
    long_chunks = module.tail_full_grid_rule((10, 7)).long_chunks_varlen
    assert long_chunks == 16

    def one(**over):
        kwargs = {"capability": (10, 7), "mode_varlen": True, "chunks_per_item": 8}
        kwargs.update(over)
        return module.tail_one_cta_selected(**kwargs)

    assert one()
    assert one(chunks_per_item=long_chunks - 1)
    assert not one(chunks_per_item=long_chunks)
    assert not one(chunks_per_item=32)
    assert not one(mode_varlen=False)
    assert not one(capability=(10, 3))
    assert not one(capability=(10, 0))
    assert not one(capability=None)
    # the programs it selects are exactly the sm_107a-only one-CTA entries:
    # the two-kernel family's PDL launch, no cluster, their own SMEM literal
    assert {
        module._program_name("exactpre", state_dtype, True, "tailone")
        for state_dtype in (torch.bfloat16, torch.float16, torch.float32)
    } == {name for name in module._SM107_ONLY_PROGRAMS if name.endswith("_tailone")}
    for state in ("bf16", "f16", "f32"):
        name = f"exactpre_{state}_varlen_tailone"
        program = module._arch_program(name, "sm_107a")
        assert program.family == "exactpre" and program.variant == "tailone"
        assert program.cluster_dims == (1, 1, 1)
        assert program.main_smem_bytes == module._EXACTPRE_TAILONE_SMEM_BYTES
        assert module._main_pdl(name, "sm_107a")
        assert program.main.kernel == (
            f"kernel_mamba_ssd_q_tmem_alias_{state}_varlen_precomputed_tailone"
        )


@pytest.mark.parametrize("has_z", (False, True))
def test_source_sm107_short_varlen_pair_calls_take_the_one_cta_pair_program(
    monkeypatch, has_z
):
    """On SM107 a packed-varlen call of 32 sequences x 1024 tokens x 8 heads
    (256 items of 8 chunks on 212 SMs, the full-grid pair admission), with or
    without z, runs the one-CTA pair program on the pair grid; the runner's
    status words hold one per pair of SMs and one per 32 pairs past the first
    two."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    runner = _cpu_forwarding_runner(
        module,
        monkeypatch,
        calls,
        nheads=8,
        ngroups=8,
        has_varlen=True,
        has_initial_states=True,
        has_z=has_z,
        arch="sm_107a",
    )
    monkeypatch.setattr(module, "_sm_count", lambda _: 212)
    num_seqs, tokens = 32, 1024
    seqlen = num_seqs * tokens
    x = torch.empty((1, seqlen, 8, 64), dtype=torch.bfloat16)
    dt = torch.empty((1, seqlen, 8), dtype=torch.float32)
    A = torch.empty((8,), dtype=torch.float32)
    B = torch.empty((1, seqlen, 8, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)
    kwargs = {
        "cu_seqlens": torch.arange(0, seqlen + 1, tokens, dtype=torch.int32),
        "initial_states": torch.empty((num_seqs, 8, 64, 128), dtype=torch.bfloat16),
    }
    if has_z:
        kwargs["z"] = torch.empty_like(x)
    runner.run(x, dt, A, B, C, **kwargs)
    ((name, launch),) = calls
    assert name == "exactpre_bf16_varlen_tailone"
    assert launch["main_grid"] == (module._tail_grid_size(sm_count=212), 1, 1)
    status = launch["main"]["preprocess_status"]
    assert tuple(status.shape) == (2 + 106 + 4,) and status.dtype == torch.int32
    assert launch["preprocess"]["preprocess_status"] is status


def test_source_sm107_ibulk_rule():
    """``ibulk_selected``: on SM107 a packed-varlen two-kernel ring call
    with a bf16 / f16 state takes the ``_ibulk`` program with at least two
    (sequence, head) items per SM of at most eight host-known chunks, a call
    with z only with one chunk per item and at least eight items per SM;
    batched calls, an f32 state and other capabilities never do."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    assert frozenset({(10, 7)}) == module._IBULK_CAPABILITIES
    assert frozenset({torch.bfloat16, torch.float16}) == module._IBULK_STATE_DTYPES
    assert (
        module._IBULK_MIN_ITEMS_PER_SM,
        module._IBULK_MAX_CHUNKS_PER_ITEM,
        module._IBULK_Z_MIN_ITEMS_PER_SM,
    ) == (2, 8, 8)

    def sel(
        items,
        chunks=1,
        has_z=False,
        state=torch.bfloat16,
        capability=(10, 7),
        mode_varlen=True,
    ):
        return module.ibulk_selected(
            capability=capability,
            state_dtype=state,
            mode_varlen=mode_varlen,
            total_work=items,
            chunks_per_item=chunks,
            sm_count=212,
            has_z=has_z,
        )

    # 64 / 128 / 256 one-chunk sequences x 8 heads (perf_varlen_*x1)
    assert sel(512) and sel(1024) and sel(2048)
    assert sel(2048, state=torch.float16)
    assert sel(1024, chunks=4)
    # perf_varlen_64x8: 64 sequences x 8 heads of 8 chunks
    assert sel(512, chunks=8)
    # bounds: 2 x 212 items, 8 chunks per item
    assert sel(424) and not sel(423)
    assert not sel(512, chunks=9)
    # z: one chunk per item and at least 8 x 212 items (nem_v16x128: 16
    # sequences x 128 heads); fewer items (nem_v4x128: 4 x 128) or more
    # chunks per item keep the ring program
    assert sel(2048, has_z=True) and sel(2048, has_z=True, state=torch.float16)
    assert sel(1696, has_z=True) and not sel(1695, has_z=True)
    assert not sel(512, has_z=True)
    assert not sel(2048, chunks=2, has_z=True)
    # batched calls, an f32 state and the other capabilities keep the ring
    # program
    assert not sel(2048, mode_varlen=False)
    assert not sel(1024, chunks=4, mode_varlen=False)
    assert not sel(2048, state=torch.float32)
    for capability in ((10, 0), (10, 3)):
        assert not sel(2048, capability=capability)


@pytest.mark.parametrize("arch", ("sm_103a", "sm_107a"))
@pytest.mark.parametrize(
    ("mode", "num_seqs", "nheads", "nchunks", "has_z", "state", "ibulk"),
    (
        # perf_varlen_128x1 / 256x1: 1024 / 2048 one-chunk items
        ("varlen", 128, 8, 1, False, torch.bfloat16, True),
        ("varlen", 256, 8, 1, False, torch.bfloat16, True),
        ("varlen", 256, 8, 1, False, torch.float16, True),
        ("varlen", 256, 8, 1, False, torch.float32, False),
        # nem_v4x128: 512 one-chunk items with z
        ("varlen", 4, 128, 1, True, torch.bfloat16, False),
        # nem_v16x128: 2048 one-chunk items with z; two chunks per item keep
        # the ring program
        ("varlen", 16, 128, 1, True, torch.bfloat16, True),
        ("varlen", 16, 128, 2, True, torch.bfloat16, False),
        # perf_varlen_64x8: 8 chunks per item; 9 chunks keep the ring program
        ("varlen", 64, 8, 8, False, torch.bfloat16, True),
        ("varlen", 64, 8, 9, False, torch.bfloat16, False),
        # perf_batched_256x1 / 128x4 / 512x2
        ("batched", 256, 8, 1, False, torch.bfloat16, False),
        ("batched", 128, 8, 4, False, torch.bfloat16, False),
        ("batched", 512, 8, 2, False, torch.float16, False),
    ),
)
def test_source_sm107_short_item_calls_take_the_ibulk_program(
    monkeypatch, arch, mode, num_seqs, nheads, nchunks, has_z, state, ibulk
):
    """On SM107 a packed-varlen two-kernel ring call of many short items runs
    the ``_ibulk`` program of its state (with z: one chunk per item and many
    items); SM103, long items, calls with z of few or longer items, batched
    calls and an f32 state keep their program."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    varlen = mode == "varlen"
    runner = _cpu_forwarding_runner(
        module,
        monkeypatch,
        calls,
        nheads=nheads,
        ngroups=8,
        has_varlen=varlen,
        has_initial_states=varlen,
        has_z=has_z,
        state_dtype=state,
        arch=arch,
    )
    monkeypatch.setattr(module, "_sm_count", lambda _: 212)
    tokens = nchunks * 128
    batch, seqlen = (1, num_seqs * tokens) if varlen else (num_seqs, tokens)
    x = torch.empty((batch, seqlen, nheads, 64), dtype=torch.bfloat16)
    dt = torch.empty((batch, seqlen, nheads), dtype=torch.float32)
    A = torch.empty((nheads,), dtype=torch.float32)
    B = torch.empty((batch, seqlen, 8, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)
    kwargs = {}
    if has_z:
        kwargs["z"] = torch.empty_like(x)
    if varlen:
        kwargs["initial_states"] = torch.empty((num_seqs, nheads, 64, 128), dtype=state)
        kwargs["cu_seqlens"] = torch.arange(
            0, (num_seqs + 1) * tokens, tokens, dtype=torch.int32
        )
    runner.run(x, dt, A, B, C, **kwargs)
    ((name, _launch),) = calls
    if arch == "sm_107a" and ibulk:
        assert name == f"exactpre_{module._STATE_DTYPE_KEYS[state]}_{mode}_ibulk"
    else:
        assert not name.endswith("_ibulk"), name


@pytest.mark.parametrize(
    ("num_seqs", "expected"),
    (
        # 256 one-chunk sequences x 8 heads: the two-kernel ``_ibulk`` program
        (256, "exactpre_bf16_varlen_ibulk"),
        # 64 one-chunk sequences x 8 heads: the fused ``_ibulk`` program
        (64, "exactfused_bf16_varlen_ibulk"),
    ),
)
def test_source_sm107_ibulk_program_binds_aligned_initial_states(
    monkeypatch, num_seqs, expected
):
    """The ``_ibulk`` programs read an item's initial state with one bulk
    copy: a contiguous ``initial_states`` at an address that is not 16-byte
    aligned is bound through the workspace's packed copy (the same values),
    an aligned one directly."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    runner = _cpu_forwarding_runner(
        module,
        monkeypatch,
        calls,
        nheads=8,
        ngroups=8,
        has_varlen=True,
        has_initial_states=True,
        arch="sm_107a",
    )
    monkeypatch.setattr(module, "_sm_count", lambda _: 212)
    tokens = 128
    seqlen = num_seqs * tokens
    x = torch.empty((1, seqlen, 8, 64), dtype=torch.bfloat16)
    dt = torch.empty((1, seqlen, 8), dtype=torch.float32)
    A = torch.empty((8,), dtype=torch.float32)
    B = torch.empty((1, seqlen, 8, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)
    cu_seqlens = torch.arange(0, (num_seqs + 1) * tokens, tokens, dtype=torch.int32)
    count = num_seqs * 8 * 64 * 128
    storage = torch.randn(count + 8).to(torch.bfloat16)
    for offset in (0, 1):
        initial = storage[offset : offset + count].view(num_seqs, 8, 64, 128)
        assert initial.is_contiguous()
        runner.run(x, dt, A, B, C, initial_states=initial, cu_seqlens=cu_seqlens)
        name, launch = calls.pop()
        assert name == expected
        bound = launch["main"]["initial_states"]
        assert bound.data_ptr() % 16 == 0
        assert (bound is initial) == (initial.data_ptr() % 16 == 0)
        assert torch.equal(bound, initial)


def _varlen_metadata_cpu(lengths):
    """CPU ``seq_idx`` / ``chunk_indices`` / ``chunk_offsets`` /
    ``seq_chunk_cumsum`` of a packed batch of non-empty sequences (the values
    :func:`_varlen_metadata` and :func:`_seq_chunk_cumsum` build on the
    device)."""

    seq_idx = torch.arange(len(lengths), dtype=torch.int32).repeat_interleave(
        torch.tensor(lengths)
    )[None]
    total = sum(lengths)
    chunk_indices, chunk_offsets = [], []
    for chunk in range(-(-total // 128)):
        values = seq_idx[0, chunk * 128 : (chunk + 1) * 128]
        previous = torch.cat((values[:1] - 1, values[:-1]))
        for offset in (values != previous).nonzero(as_tuple=True)[0].tolist():
            chunk_indices.append(chunk)
            chunk_offsets.append(offset)
    cumsum, start = [0], 0
    for length in lengths:
        end = start + length
        cumsum.append(cumsum[-1] + (-(-end // 128) - start // 128))
        start = end
    return (
        seq_idx,
        torch.tensor(chunk_indices, dtype=torch.int32),
        torch.tensor(chunk_offsets, dtype=torch.int32),
        torch.tensor(cumsum, dtype=torch.int32),
    )


def test_source_sm107_fused_ibulk_rule():
    """``fused_ibulk_selected``: on SM107 a packed-varlen fused ring call
    with a bf16 / f16 state and initial states takes the fused ``_ibulk``
    program with more (sequence, head) items than SMs of one host-known
    chunk; calls without initial states, an f32 state, items of several
    chunks, at most one item per SM and other capabilities never do."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    assert frozenset({(10, 7)}) == module._FUSED_IBULK_CAPABILITIES
    assert module._FUSED_IBULK_MAX_CHUNKS_PER_ITEM == 1

    def sel(
        items,
        chunks=1,
        init=True,
        state=torch.bfloat16,
        capability=(10, 7),
    ):
        return module.fused_ibulk_selected(
            capability=capability,
            state_dtype=state,
            has_initial_states=init,
            total_work=items,
            chunks_per_item=chunks,
            sm_count=212,
        )

    # 32 / 64 one-chunk sequences x 8 heads (perf_varlen_32x1 / 64x1)
    assert sel(256) and sel(512)
    assert sel(512, state=torch.float16)
    # bound: more items than SMs
    assert sel(213) and not sel(212)
    # 1 / 4 / 8 sequences x 8 heads (perf_varlen_1x1 / 4x1 / 8x1)
    assert not sel(8) and not sel(32) and not sel(64)
    # items of two or eight chunks (perf_varlen_4x8 / 8x8: 8 chunks)
    assert not sel(256, chunks=2)
    assert not sel(64, chunks=8)
    # no initial states, an f32 state and the other capabilities
    assert not sel(512, init=False)
    assert not sel(512, state=torch.float32)
    for capability in ((10, 0), (10, 3)):
        assert not sel(512, capability=capability)


@pytest.mark.parametrize("arch", ("sm_103a", "sm_107a"))
@pytest.mark.parametrize(
    ("form", "num_seqs", "nheads", "tokens", "has_z", "init", "state", "ibulk"),
    (
        # perf_varlen_64x1 / 32x1: 512 / 256 one-chunk items
        ("caller", 64, 8, 128, False, True, torch.bfloat16, True),
        ("caller", 32, 8, 128, False, True, torch.bfloat16, True),
        ("cu_seqlens", 64, 8, 128, False, True, torch.bfloat16, True),
        ("cu_seqlens", 64, 8, 128, False, True, torch.float16, True),
        # 64-token sequences (partial segments): one host-known chunk per item
        ("caller", 64, 8, 64, False, True, torch.bfloat16, True),
        # 2 sequences x 128 heads with z: 256 one-chunk items (seed window)
        ("caller", 2, 128, 128, True, True, torch.bfloat16, True),
        # perf_varlen_8x1: 64 items, at most one per SM
        ("caller", 8, 8, 128, False, True, torch.bfloat16, False),
        # perf_varlen_4x8: 32 items of 8 chunks
        ("caller", 4, 8, 1024, False, True, torch.bfloat16, False),
        # 32 sequences x 2 chunks: 256 items of 2 chunks
        ("caller", 32, 8, 256, False, True, torch.bfloat16, False),
        # no initial states, an f32 state
        ("caller", 64, 8, 128, False, False, torch.bfloat16, False),
        ("caller", 64, 8, 128, False, True, torch.float32, False),
        # the triple form deriving its table takes the one-launch program
        ("derived", 32, 8, 128, False, True, torch.bfloat16, False),
    ),
)
def test_source_sm107_one_chunk_fused_calls_take_the_ibulk_program(
    monkeypatch, arch, form, num_seqs, nheads, tokens, has_z, init, state, ibulk
):
    """On SM107 a packed-varlen call the fused ring program serves with one
    host-known chunk per item, initial states, a bf16 / f16 state and more
    items than SMs runs the fused ``_ibulk`` program of its state; SM103,
    fewer items, longer items, calls without initial states, an f32 state and
    the one-launch triple form keep their program."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    runner = _cpu_forwarding_runner(
        module,
        monkeypatch,
        calls,
        nheads=nheads,
        ngroups=8,
        has_varlen=True,
        has_initial_states=init,
        has_z=has_z,
        state_dtype=state,
        arch=arch,
    )
    monkeypatch.setattr(module, "_sm_count", lambda _: 212)
    seqlen = num_seqs * tokens
    x = torch.empty((1, seqlen, nheads, 64), dtype=torch.bfloat16)
    dt = torch.empty((1, seqlen, nheads), dtype=torch.float32)
    A = torch.empty((nheads,), dtype=torch.float32)
    B = torch.empty((1, seqlen, 8, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)
    kwargs = {}
    if has_z:
        kwargs["z"] = torch.empty_like(x)
    if init:
        kwargs["initial_states"] = torch.empty((num_seqs, nheads, 64, 128), dtype=state)
    if form == "cu_seqlens":
        kwargs["cu_seqlens"] = torch.arange(
            0, (num_seqs + 1) * tokens, tokens, dtype=torch.int32
        )
    else:
        lengths = [tokens] * num_seqs
        seq_idx, chunk_indices, chunk_offsets, cumsum = _varlen_metadata_cpu(lengths)
        kwargs.update(
            seq_idx=seq_idx, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets
        )
        if form == "caller":
            kwargs["seq_chunk_cumsum"] = cumsum
    runner.run(x, dt, A, B, C, **kwargs)
    ((name, _launch),) = calls
    key = module._STATE_DTYPE_KEYS[state]
    if arch == "sm_107a" and ibulk:
        assert name == f"exactfused_{key}_varlen_ibulk"
    else:
        assert not name.endswith("_ibulk"), name
        if form == "derived" and arch == "sm_107a":
            # the one-launch program in the prologue form of its launch grid
            table_form = module.exactone_table_variant(
                capability=(10, 7),
                variant="",
                state_dtype=state,
                grid=module._persistent_grid_size(
                    total_work=num_seqs * nheads, sm_count=212
                ),
            )
            assert name == f"exactone_{key}_varlen_{table_form}"


def test_source_fused_route_mirrors_the_seed_rule():
    """``fused_selected`` reads the seed module's ``fused_route`` constants on
    host-known quantities: batched calls with at most 1024 (sequence, head)
    items, 16 chunks per sequence and 2048 chunk-items; packed-varlen calls
    with at most 256 items and 512 packed chunk-items.  The runner applies
    the window only where the fused program's preprocess grid is 0 (batched
    and cu_seqlens-form calls)."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    fused = module.fused_selected

    assert (
        module._FUSED_ROUTE_MAX_ITEMS_BATCHED,
        module._FUSED_ROUTE_MAX_CHUNKS_BATCHED,
        module._FUSED_ROUTE_MAX_CHUNK_ITEMS_BATCHED,
        module._FUSED_ROUTE_MAX_ITEMS_VARLEN,
        module._FUSED_ROUTE_MAX_CHUNK_ITEMS_VARLEN,
    ) == (1024, 16, 2048, 256, 512)
    # batched: 1 x 128 heads, 16 chunks (one-wave long row) is fused; 17 chunks is not
    assert fused(
        nheads=128, num_sequences=1, nchunks=16, mode_varlen=False, total_tokens=2048
    )
    assert not fused(
        nheads=128, num_sequences=1, nchunks=17, mode_varlen=False, total_tokens=2176
    )
    # 8 x 128 = 1024 items x 2 chunks = 2048 chunk-items is fused; 3 chunks is not
    assert fused(
        nheads=128, num_sequences=8, nchunks=2, mode_varlen=False, total_tokens=256
    )
    assert not fused(
        nheads=128, num_sequences=8, nchunks=3, mode_varlen=False, total_tokens=384
    )
    # more than 1024 items (saturated rows) keep the two-kernel program
    assert not fused(
        nheads=128, num_sequences=9, nchunks=1, mode_varlen=False, total_tokens=128
    )
    assert not fused(
        nheads=128, num_sequences=16, nchunks=1, mode_varlen=False, total_tokens=128
    )
    # varlen: 2 x 128 = 256 items with 4 packed chunks (512 chunk-items) is fused
    assert fused(
        nheads=128, num_sequences=2, nchunks=4, mode_varlen=True, total_tokens=512
    )
    assert not fused(
        nheads=128, num_sequences=2, nchunks=5, mode_varlen=True, total_tokens=513
    )
    assert not fused(
        nheads=128, num_sequences=3, nchunks=1, mode_varlen=True, total_tokens=128
    )
    assert fused(
        nheads=8, num_sequences=1, nchunks=8, mode_varlen=True, total_tokens=1024
    )


def test_source_two_kernel_route_mirrors_the_seed_rule():
    """``two_kernel_family`` mirrors the seed module's
    ``two_kernel_route``: a two-kernel call takes the round-6 ``exactpre``
    program where the host-known chunks per work item reach two -- batched
    ``ceil(seqlen / 128)``, varlen ``ceil(ceil(packed_tokens / num_sequences)
    / 128)`` -- and the retained start ``exact`` program for one-chunk
    items, whose lever-1b fixed cost is not repaid."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    route = module.two_kernel_family

    assert module._TWO_KERNEL_ROUTE_MIN_CHUNKS == 2
    # batched: the per-sequence length decides
    assert route(mode_varlen=False, num_sequences=16, seqlen=128) == "exact"
    assert route(mode_varlen=False, num_sequences=16, seqlen=129) == "exactpre"
    assert route(mode_varlen=False, num_sequences=1, seqlen=1) == "exact"
    assert route(mode_varlen=False, num_sequences=1, seqlen=32768) == "exactpre"
    # varlen: the mean sequence length of the packed stream decides
    assert route(mode_varlen=True, num_sequences=2, seqlen=96 + 160) == "exact"
    assert route(mode_varlen=True, num_sequences=2, seqlen=128 + 896) == "exactpre"
    assert route(mode_varlen=True, num_sequences=64, seqlen=64 * 128) == "exact"
    assert route(mode_varlen=True, num_sequences=1, seqlen=129) == "exactpre"
    assert route(mode_varlen=True, num_sequences=4, seqlen=4) == "exact"
    # a sequence count of 0 counts as one sequence
    assert route(mode_varlen=True, num_sequences=0, seqlen=300) == "exactpre"
    assert route(mode_varlen=True, num_sequences=0, seqlen=100) == "exact"


def test_source_persistent_grid_mirrors_the_seed_rule():
    """``_persistent_grid_size`` mirrors the generator's host-side grid rule:
    one CTA per (sequence, head) item up to one
    wave; beyond it the smallest count of two to four items per CTA that
    divides the items and keeps at least four fifths of the SMs busy; else
    one CTA per SM.  Pinned on the shipped SM counts (B200 148, GB300 152)
    at the gate rows' item counts, and checked as a property over four
    waves."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    grid = module._persistent_grid_size
    for sm_count in (148, 152):
        # one wave: one CTA per item
        for items in (1, 8, 16, 32, 64, 128, sm_count):
            assert grid(total_work=items, sm_count=sm_count) == items
        # 256 / 384 / 512 items: two / three / four per CTA on 128 CTAs
        # (128 x 5 >= 4 x 152)
        assert grid(total_work=256, sm_count=sm_count) == 128
        assert grid(total_work=384, sm_count=sm_count) == 128
        assert grid(total_work=512, sm_count=sm_count) == 128
        # 296 items: two per CTA on 148 CTAs on either part
        assert grid(total_work=296, sm_count=sm_count) == 148
        # 1024 .. 20480 items (the saturated gate rows): one CTA per SM
        for items in (1024, 2048, 4096, 8192, 20480):
            assert grid(total_work=items, sm_count=sm_count) == sm_count
        # odd counts just past one wave: one CTA per SM
        for items in (sm_count + 1, 255, 257):
            assert grid(total_work=items, sm_count=sm_count) == sm_count
        for total_work in range(1, 4 * sm_count + 2):
            value = grid(total_work=total_work, sm_count=sm_count)
            assert 1 <= value <= sm_count
            if total_work <= sm_count:
                assert value == total_work
            elif value != sm_count:
                assert total_work % value == 0 and 2 <= total_work // value <= 4
                assert value * 5 >= sm_count * 4
            else:
                # one CTA per SM only when no smaller balanced grid exists
                # (296 items on 148 SMs: two per CTA on 148 CTAs is both)
                assert not any(
                    total_work % k == 0
                    and total_work // k < sm_count
                    and (total_work // k) * 5 >= sm_count * 4
                    for k in range(2, 5)
                )


def test_source_retained_start_programs_are_the_round5_files():
    """The ``exact_*`` programs are the round-5 start files retained byte for
    byte: the 11 device sources the start loader bound that still ship (the
    f32 varlen ``_noring`` one is retired: an f32 state never takes the varlen
    start schedule, ``_EXACT_NORING_VARLEN_MIN_ITEMS``),
    its host launcher (41 main arguments, ``_MAIN_ARGS_START``) and its main
    dynamic SMEM literals.  The Cake export pins the same digests
    (``_MAMBA_SSD_COMBINED_RETAINED_START_FILES``)."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    source_dir = module._source_dir()
    retained = {
        "device/mamba_ssd_q_tmem_alias_bf16_batched_3876f7208d.cu": "46c3bc0cc2d71728d70ea140c77da9f094a71612e85f1d11d9cf9ad37cfed4c3",
        "device/mamba_ssd_q_tmem_alias_bf16_batched_noring_77156893a6.cu": "3cfae323c2eb5290b05f4590d37029c4af7d7a12200ab3cb5a7fb8058cb15637",
        "device/mamba_ssd_q_tmem_alias_bf16_varlen_636342f6cc.cu": "93992e1526a925b69ccdf4645fdd6873c9314e70859f3f12f398cdbf9bec96cd",
        "device/mamba_ssd_q_tmem_alias_bf16_varlen_noring_0f475595b9.cu": "0112ebe565d1895ce8a1568765c84fc868a02f8bba340b4ab0bcf06856e3e105",
        "device/mamba_ssd_q_tmem_alias_f16_batched_f074fc128b.cu": "99924fb4ba28c8fcc4d1de304ec4533bb2f105628b2084b99575db23edc7cee2",
        "device/mamba_ssd_q_tmem_alias_f16_batched_noring_d19039c163.cu": "beaf9f268cd701c35009581594fc320e3d853872ba6292a6ccedc571369db685",
        "device/mamba_ssd_q_tmem_alias_f16_varlen_9759edcc11.cu": "5bba167a188354bb4393727798114c5f03dfa63d9c3cdc9edbfeb629e6b3a98a",
        "device/mamba_ssd_q_tmem_alias_f16_varlen_noring_d11774ac8e.cu": "af0557e21b54901059c9f656bc88c4242c6a0a426014a1b0b0a94440fa5d4cbd",
        "device/mamba_ssd_q_tmem_alias_f32_batched_2dc39c9955.cu": "11549cdbdcc3329ed7bd42a46f83daf436db88aed498cd8028d37513162f40a3",
        "device/mamba_ssd_q_tmem_alias_f32_batched_noring_bbd73b5202.cu": "69c7298f74a264a8e3db1c0c2a26ed27a4de6703e01178051232fb833a54affd",
        "device/mamba_ssd_q_tmem_alias_f32_varlen_6037fe869d.cu": "4f1059751fb8173b0a1e0c9dad79c4899ed563baf9329a8feac9e177c047072b",
        "host/mamba_ssd_combined_sequence.cpp": "a992780866f3b70c21f009dfdc74753acecc8573925d219dad5e822821771344",
    }
    assert len(retained) == 12
    for relpath, digest in retained.items():
        path = source_dir / "generated" / relpath
        assert path.is_file(), relpath
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, relpath
    assert (
        Path("generated/host/mamba_ssd_combined_sequence.cpp") == module._HOST_TEMPLATE
    )
    assert {f"device/{ident}.cu" for ident in module._SCAN_MODULES.values()} == {
        relpath for relpath in retained if relpath.startswith("device/")
    }
    assert module._FAMILY_HOST["exact"] == (
        module._HOST_TEMPLATE,
        module._MAIN_ARGS_START,
        module._EXACT_SMEM_BYTES,
    )
    assert (module._EXACT_SMEM_BYTES, module._EXACT_NORING_SMEM_BYTES) == (
        231808,
        232448,
    )
    assert len(module._MAIN_ARGS_START) == 41
    for name in module._SCAN_MODULES:
        program = module._PROGRAMS[name]
        assert (
            program.family == "exact" and program.main_args == module._MAIN_ARGS_START
        )
        assert program.preprocess is module._SEGMENT_PREPROCESS


def test_source_saturated_batched_call_keeps_the_two_kernel_program(monkeypatch):
    """A batched call past the fused route (16 sequences x 128 heads) runs
    the two-kernel form: at one chunk per item the retained start ``exact``
    program (41 main arguments), at two the round-6 ``exactpre`` program
    (``two_kernel_family``).  Either preprocess writes the HBM delta / cumsum
    (grid > 0) and the main launch leaves the fused-only inputs unused."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    runner = _cpu_forwarding_runner(module, monkeypatch, calls, nheads=128, ngroups=8)
    batch, seqlen, nheads, ngroups = 16, 128, 128, 8
    x = torch.empty((batch, seqlen, nheads, 64), dtype=torch.bfloat16)
    dt = torch.empty((batch, seqlen, nheads), dtype=torch.float32)
    A = torch.empty((nheads,), dtype=torch.float32)
    B = torch.empty((batch, seqlen, ngroups, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)

    runner.run(x, dt, A, B, C)

    ((name, launch),) = calls
    assert name == runner.last_program_name == "exact_bf16_batched"
    assert module._PROGRAMS[name].family == "exact"
    assert module._PROGRAMS[name].main_args == module._MAIN_ARGS_START
    tiles = batch * nheads  # one segment per sequence at 128 tokens
    assert launch["preprocess_grid"] == (
        -(-tiles // module._SEGMENT_PREPROCESS_TILES_PER_BLOCK),
        1,
        1,
    )
    assert launch["main"]["metadata_from_cu_seqlens"] == 0
    assert launch["main"]["delta_precomputed"].shape == (tiles, 128)

    seqlen = 256  # two chunks per item: the round-6 exactpre program
    x = torch.empty((batch, seqlen, nheads, 64), dtype=torch.bfloat16)
    dt = torch.empty((batch, seqlen, nheads), dtype=torch.float32)
    B = torch.empty((batch, seqlen, ngroups, 128), dtype=torch.bfloat16)
    runner.run(x, dt, A, B, torch.empty_like(B))
    name, launch = calls[-1]
    assert name == runner.last_program_name == "exactpre_bf16_batched"
    assert module._PROGRAMS[name].family == "exactpre"
    assert module._PROGRAMS[name].main_args == module._MAIN_ARGS
    assert launch["preprocess_grid"][0] >= 1
    assert launch["main"]["metadata_from_cu_seqlens"] == 0


@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a", "sm_107a"))
def test_source_batched_preprocess_seqlen_bound_without_gpu(monkeypatch, arch):
    """The segment preprocess of a batched call gets the caller's ``seqlen``
    on sm_100a and sm_103a, and the token count of all rows on sm_107a,
    whose preprocess bounds its aligned-table dt loads by it (the batched
    tables use it for nothing else)."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    runner = _cpu_forwarding_runner(
        module, monkeypatch, calls, nheads=128, ngroups=8, arch=arch
    )
    batch, seqlen, nheads, ngroups = 16, 256, 128, 8
    x = torch.empty((batch, seqlen, nheads, 64), dtype=torch.bfloat16)
    dt = torch.empty((batch, seqlen, nheads), dtype=torch.float32)
    A = torch.empty((nheads,), dtype=torch.float32)
    B = torch.empty((batch, seqlen, ngroups, 128), dtype=torch.bfloat16)
    runner.run(x, dt, A, B, torch.empty_like(B))

    name, launch = calls[-1]
    assert module._PROGRAMS[name].preprocess is module._SEGMENT_PREPROCESS
    preprocess = launch["preprocess"]
    assert preprocess["direct_varlen_metadata"] == 0
    assert preprocess["dt"].numel() == batch * seqlen * nheads
    expected = batch * seqlen if arch == "sm_107a" else seqlen
    assert preprocess["seqlen"] == expected
    assert launch["main"]["seqlen"] == seqlen


@pytest.mark.parametrize("arch", _ROUTE_ARCHES)
def test_source_varlen_triple_form_routes_on_the_caller_cumsum(monkeypatch, arch):
    """A short packed-varlen call in the seq_idx triple form inside the fused
    shape window runs, while ``seq_chunk_cumsum`` is published (runner-owned,
    or the caller's buffer with ``update_seq_chunk_cumsum=True``), the
    program of the device's capability: on SM100 ``exactmeta`` (the light
    metadata kernel publishes the table, one thread per segment, no dt work,
    then the fused main kernel runs); on SM103 the one-launch ``exactone``
    program for a runner-owned table (the fused main kernel derives it in a
    CTA prologue, nothing is published) and the two-kernel program where the
    caller's buffer must receive it (the full preprocess publishes it).  With
    a caller table the preprocess does not rewrite
    (``update_seq_chunk_cumsum=False``) nothing is published, so the fused
    program is one launch reading the caller's tables
    (``_WorkspaceKey.caller_cumsum``).  Outside the window the trusted table
    still runs the two-kernel program.  The same call in the cu_seqlens form
    takes the fused program (next test)."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    runner = _cpu_forwarding_runner(
        module, monkeypatch, calls, has_varlen=True, arch=arch
    )
    seqlen, num_seqs = 300, 2
    x = torch.empty((1, seqlen, 1, 64), dtype=torch.bfloat16)
    dt = torch.empty((1, seqlen, 1), dtype=torch.float32)
    A = torch.empty((1,), dtype=torch.float32)
    B = torch.empty((1, seqlen, 1, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)
    triple = {
        "seq_idx": torch.zeros((1, seqlen), dtype=torch.int32),
        "chunk_indices": torch.tensor([0, 1, 2], dtype=torch.int32),
        "chunk_offsets": torch.tensor([0, 0, 0], dtype=torch.int32),
        "num_seqs": num_seqs,
    }

    def key(caller_cumsum, publish_table=False):
        # (device, batch, seqlen, nchunks, num_segments, num_sequences, state
        # dtype, from_cu_seqlens, caller_cumsum, publish_table, family override)
        return module._WorkspaceKey(
            None,
            1,
            seqlen,
            3,
            3,
            num_seqs,
            torch.bfloat16,
            False,
            caller_cumsum,
            publish_table,
            "auto",
        )

    # inside the fused shape window (2 items, 3 packed chunks) ...
    assert module.fused_selected(
        nheads=1,
        num_sequences=num_seqs,
        nchunks=3,
        mode_varlen=True,
        total_tokens=seqlen,
    )
    runner.run(x, dt, A, B, C, **triple)
    name, launch = calls[-1]
    assert name == runner.last_program_name
    if arch in _LIGHT_STAGE_ARCHES:
        # ... the triple form deriving the table runs the exactmeta program:
        # the metadata stage (3 segments -> one 256-thread CTA) publishes the
        # runner's vector, which the fused main kernel reads
        assert name == "exactmeta_bf16_varlen"
        assert module._PROGRAMS[name].family == "exactmeta"
        assert set(launch["preprocess"]) == set(module._METADATA_ARGS)
        assert launch["preprocess_grid"] == (1, 1, 1)
        assert launch["preprocess"]["num_segments"] == 3
        table_arg = "sequence_offsets"
    else:
        # ... on SM103 and SM107 the one-launch exactone program: no stage
        # (grid 0); the fused main kernel's CTA prologue derives the table into its row
        # of the runner's scratch (one CTA: 2 items on this 1-SM device; on
        # SM107 a grid that small takes the tablefirst form)
        assert name == (
            "exactone_bf16_varlen_tablefirst"
            if arch == "sm_107a"
            else "exactone_bf16_varlen"
        )
        assert module._PROGRAMS[name].family == "exactone"
        assert set(launch["preprocess"]) == set(module._PREPROCESS_ARGS)
        assert launch["preprocess_grid"] == (0, 1, 1)
        assert launch["main_grid"] == (1, 1, 1)
        assert launch["preprocess"]["write_seq_chunk_cumsum"] == 0
        assert launch["main"]["seq_chunk_cumsum"].numel() == num_seqs + 1
        table_arg = "seq_chunk_cumsum"
    assert launch["preprocess"][table_arg] is launch["main"]["seq_chunk_cumsum"]
    assert launch["preprocess"]["seq_idx_i32"] is triple["seq_idx"]
    assert launch["preprocess"]["chunk_indices"] is triple["chunk_indices"]
    assert launch["main"]["metadata_from_cu_seqlens"] == (
        0 if arch in _LIGHT_STAGE_ARCHES else module._TABLE_IN_KERNEL_I32
    )
    assert key(False) in runner._workspaces
    derived_vector = launch["main"]["seq_chunk_cumsum"]

    # ... a trusted caller table makes the fused program one launch: the
    # preprocess grid is 0 and the main kernel reads the caller's tables
    vector = torch.tensor([0, 1, 3], dtype=torch.int32)
    runner.run(x, dt, A, B, C, **triple, seq_chunk_cumsum=vector)
    name, launch = calls[-1]
    assert name == runner.last_program_name == "exactfused_bf16_varlen"
    assert module._PROGRAMS[name].family == "exactfused"
    assert launch["preprocess_grid"] == (0, 1, 1)
    assert launch["preprocess"]["write_seq_chunk_cumsum"] == 0
    assert launch["main"]["metadata_from_cu_seqlens"] == 0
    assert launch["main"]["seq_chunk_cumsum"] is vector
    assert launch["main"]["chunk_indices"] is triple["chunk_indices"]
    assert launch["main"]["chunk_offsets"] is triple["chunk_offsets"]
    assert key(True) in runner._workspaces
    assert runner._workspaces.get(key(True)).workspace["program"] == name

    # ... while a caller buffer the publishing stage must fill is its own
    # call shape (``publish_table``): on SM100 the same exactmeta program,
    # publishing into the caller's buffer; on SM103 the two-kernel program
    # (the one-launch program keeps its rows private, so it is not an option)
    buffer = torch.empty_like(vector)
    runner.run(
        x, dt, A, B, C, **triple, seq_chunk_cumsum=buffer, update_seq_chunk_cumsum=True
    )
    name, launch = calls[-1]
    assert name == runner.last_program_name
    assert name == runner._workspaces.get(key(False, True)).workspace["program"]
    assert module._PROGRAMS[name].family == (
        "exactmeta"
        if arch in _LIGHT_STAGE_ARCHES
        else module.two_kernel_family(
            mode_varlen=True, num_sequences=num_seqs, seqlen=seqlen
        )
    )
    assert launch["preprocess"][table_arg] is buffer
    assert launch["main"]["seq_chunk_cumsum"] is buffer
    assert launch["main"]["metadata_from_cu_seqlens"] == 0
    assert buffer is not derived_vector
    assert launch["preprocess_grid"][0] >= 1
    assert len(runner._workspaces) == 3

    # Outside the fused window (513 one-token sequences: more than 256 items,
    # and more than the 512 items of the sm_107a extension for calls without
    # z) the trusted table keeps the two-kernel program: one chunk per item,
    # so the retained start ``exact`` program (``two_kernel_family``) -- on
    # sm_107a, with more items than SMs (one on this device) and at most
    # 2048, the ``exactpre`` program, there its ``_ibulk`` form (at least two
    # one-chunk items per SM, :func:`ibulk_selected`).
    count = 513
    ids = torch.arange(count, dtype=torch.int32)
    wide = {
        "seq_idx": ids.reshape(1, count),
        "chunk_indices": ids // 128,
        "chunk_offsets": ids % 128,
    }
    assert not module.fused_selected(
        nheads=1, num_sequences=count, nchunks=3, mode_varlen=True, total_tokens=count
    )
    assert (
        module.two_kernel_family(mode_varlen=True, num_sequences=count, seqlen=count)
        == "exact"
    )
    family = module.two_kernel_family(
        mode_varlen=True,
        num_sequences=count,
        seqlen=count,
        nheads=1,
        sm_count=1,
        capability=module._ARCH_CAPABILITIES[arch],
    )
    assert family == ("exactpre" if arch == "sm_107a" else "exact")
    runner.run(
        torch.empty((1, count, 1, 64), dtype=torch.bfloat16),
        torch.empty((1, count, 1), dtype=torch.float32),
        A,
        torch.empty((1, count, 1, 128), dtype=torch.bfloat16),
        torch.empty((1, count, 1, 128), dtype=torch.bfloat16),
        **wide,
        seq_chunk_cumsum=torch.arange(count + 1, dtype=torch.int32),
    )
    name, launch = calls[-1]
    suffix = "_ibulk" if arch == "sm_107a" else ""
    assert name == runner.last_program_name == f"{family}_bf16_varlen{suffix}"
    assert module._PROGRAMS[name].family == family
    assert launch["preprocess"]["write_seq_chunk_cumsum"] == 0
    assert launch["preprocess_grid"][0] >= 1


@pytest.mark.parametrize("arch", _ROUTE_ARCHES)
def test_source_varlen_triple_form_routes_by_capability(monkeypatch, arch):
    """Route matrix of the four packed-varlen call forms inside the fused
    window, per compute capability.  The ``cu_seqlens`` form and the triple
    form with a trusted caller table run the fused program as one launch on
    both architectures (preprocess grid 0; the cu_seqlens form derives its
    tables in-CTA, ``metadata_from_cu_seqlens``).  The triple form deriving
    its table (int32 or int64 ``seq_idx``) runs the program of the
    capability: ``exactmeta`` on SM100 (the 12-argument light metadata stage
    publishes it ahead of the main kernel), ``exactone`` on SM103 (one
    launch, the main kernel's prologue derives it; ``metadata_from_cu_seqlens``
    names the seq_idx dtype).  The triple form publishing the table into the
    caller's buffer needs a publishing stage on both: ``exactmeta`` on SM100,
    the two-kernel program on SM103."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    seqlen, num_seqs = 300, 2
    chunk_indices = torch.tensor([0, 1, 2], dtype=torch.int32)
    chunk_offsets = torch.tensor([0, 0, 0], dtype=torch.int32)
    table = torch.tensor([0, 1, 3], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 150, seqlen], dtype=torch.int32)
    two_kernel = module.two_kernel_family(
        mode_varlen=True, num_sequences=num_seqs, seqlen=seqlen
    )
    derived = "exactmeta" if arch in _LIGHT_STAGE_ARCHES else "exactone"
    published = "exactmeta" if arch in _LIGHT_STAGE_ARCHES else two_kernel
    forms = (
        ("cu_seqlens", torch.int32),
        ("triple_i32", torch.int32),
        ("triple_i64", torch.int64),
        ("trusted", torch.int32),
        ("publish", torch.int32),
    )
    for form, seq_idx_dtype in forms:
        calls = []
        runner = _cpu_forwarding_runner(
            module,
            monkeypatch,
            calls,
            has_varlen=True,
            seq_idx_dtype=seq_idx_dtype,
            arch=arch,
        )
        x = torch.empty((1, seqlen, 1, 64), dtype=torch.bfloat16)
        dt = torch.empty((1, seqlen, 1), dtype=torch.float32)
        A = torch.empty((1,), dtype=torch.float32)
        B = torch.empty((1, seqlen, 1, 128), dtype=torch.bfloat16)
        C = torch.empty_like(B)
        if form == "cu_seqlens":
            kwargs = {"cu_seqlens": cu_seqlens}
        else:
            kwargs = {
                "seq_idx": torch.zeros((1, seqlen), dtype=seq_idx_dtype),
                "chunk_indices": chunk_indices,
                "chunk_offsets": chunk_offsets,
                "num_seqs": num_seqs,
            }
            if form == "trusted":
                kwargs["seq_chunk_cumsum"] = table
            elif form == "publish":
                kwargs["seq_chunk_cumsum"] = torch.empty_like(table)
                kwargs["update_seq_chunk_cumsum"] = True
        runner.run(x, dt, A, B, C, **kwargs)
        name, launch = calls[-1]
        assert name == runner.last_program_name
        family = module._PROGRAMS[name].family
        if form in ("cu_seqlens", "trusted"):
            assert family == "exactfused", (arch, form, name)
            assert launch["preprocess_grid"] == (0, 1, 1)
            assert launch["main"]["metadata_from_cu_seqlens"] == int(
                form == "cu_seqlens"
            )
        elif form == "publish":
            assert family == published, (arch, form, name)
            assert launch["preprocess_grid"][0] >= 1
            assert launch["main"]["metadata_from_cu_seqlens"] == 0
            assert launch["main"]["seq_chunk_cumsum"] is kwargs["seq_chunk_cumsum"]
        elif family == "exactone":
            assert arch not in _LIGHT_STAGE_ARCHES and family == derived, (
                arch,
                form,
                name,
            )
            assert launch["preprocess_grid"] == (0, 1, 1)
            assert launch["main"]["metadata_from_cu_seqlens"] == (
                module._TABLE_IN_KERNEL_I64
                if form == "triple_i64"
                else module._TABLE_IN_KERNEL_I32
            )
            assert set(launch["preprocess"]) == set(module._PREPROCESS_ARGS)
            assert launch["preprocess"]["write_seq_chunk_cumsum"] == 0
            assert launch["preprocess"]["seq_idx_int64"] == int(form == "triple_i64")
        else:
            assert family == derived, (arch, form, name)
            assert launch["preprocess_grid"][0] >= 1
            assert launch["main"]["metadata_from_cu_seqlens"] == 0
            stage_args = (
                module._METADATA_ARGS
                if family == "exactmeta"
                else module._PREPROCESS_ARGS
            )
            assert set(launch["preprocess"]) == set(stage_args)
            assert launch["preprocess"]["seq_idx_int64"] == int(form == "triple_i64")


def test_source_exactone_scratch_has_one_table_row_per_cta(monkeypatch):
    """The one-launch program's ``seq_chunk_cumsum`` scratch is sized with
    the shape's workspace as one ``num_sequences + 1`` int32 row per CTA of
    the main kernel's persistent grid (``_persistent_grid_size``): the CTA
    prologue derives its private table into row ``blockIdx.x``.  It grows
    with the sequence count and the grid, the stage is skipped and the same
    scratch is bound to both stages."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    for num_seqs, sm_count in ((2, 148), (4, 148), (32, 148), (2, 1)):
        calls = []
        runner = _cpu_forwarding_runner(
            module,
            monkeypatch,
            calls,
            has_varlen=True,
            nheads=8,
            ngroups=8,
            arch="sm_103a",
        )
        monkeypatch.setattr(module, "_sm_count", lambda _: sm_count)
        monkeypatch.setenv(module._CHUNK_PARALLEL_ENV, "never")
        seqlen = 64 * num_seqs  # 64-token sequences, two per 128-token chunk
        ids = torch.arange(num_seqs, dtype=torch.int32)
        triple = {
            "seq_idx": torch.repeat_interleave(ids, 64).reshape(1, seqlen),
            "chunk_indices": ids // 2,
            "chunk_offsets": (ids % 2) * 64,
            "num_seqs": num_seqs,
        }
        assert module.fused_selected(
            nheads=8,
            num_sequences=num_seqs,
            nchunks=1,
            mode_varlen=True,
            total_tokens=seqlen,
        )
        runner.run(
            torch.empty((1, seqlen, 8, 64), dtype=torch.bfloat16),
            torch.empty((1, seqlen, 8), dtype=torch.float32),
            torch.empty((8,), dtype=torch.float32),
            torch.empty((1, seqlen, 8, 128), dtype=torch.bfloat16),
            torch.empty((1, seqlen, 8, 128), dtype=torch.bfloat16),
            **triple,
        )
        name, launch = calls[-1]
        assert module._PROGRAMS[name].family == "exactone", (num_seqs, sm_count, name)
        grid = module._persistent_grid_size(total_work=8 * num_seqs, sm_count=sm_count)
        assert launch["main_grid"] == (grid, 1, 1)
        assert launch["preprocess_grid"] == (0, 1, 1)
        scratch = launch["main"]["seq_chunk_cumsum"]
        assert scratch.dtype == torch.int32
        assert scratch.numel() == grid * (num_seqs + 1), (num_seqs, sm_count, grid)
        assert launch["preprocess"]["seq_chunk_cumsum"] is scratch
        assert launch["preprocess"]["write_seq_chunk_cumsum"] == 0


def test_source_other_capabilities_are_refused_before_routing(monkeypatch):
    """A device that is neither SM100 nor SM103 is refused by ``_target_arch``
    (the same ``ValueError`` from ``run`` and ``prepare``): no route is
    guessed for an unmeasured capability."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    target_arch = module._target_arch
    calls = []
    runner = _cpu_forwarding_runner(module, monkeypatch, calls, has_varlen=True)
    monkeypatch.setattr(module, "_target_arch", target_arch.__wrapped__)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (9, 0))
    seqlen = 300
    x = torch.empty((1, seqlen, 1, 64), dtype=torch.bfloat16)
    dt = torch.empty((1, seqlen, 1), dtype=torch.float32)
    A = torch.empty((1,), dtype=torch.float32)
    B = torch.empty((1, seqlen, 1, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)
    triple = {
        "seq_idx": torch.zeros((1, seqlen), dtype=torch.int32),
        "chunk_indices": torch.tensor([0, 1, 2], dtype=torch.int32),
        "chunk_offsets": torch.tensor([0, 0, 0], dtype=torch.int32),
        "num_seqs": 2,
    }
    with pytest.raises(ValueError, match="requires SM100, SM103 or SM107"):
        runner.run(x, dt, A, B, C, **triple)
    with pytest.raises(ValueError, match="requires SM100, SM103 or SM107"):
        runner.prepare(x=x, dt=dt, A=A, B=B, C=C, **triple)
    assert calls == []


def test_source_fused_cu_seqlens_form_derives_the_metadata_in_kernel(monkeypatch):
    """In the ``cu_seqlens`` form the fused program derives the segment
    tables in-CTA (``metadata_from_cu_seqlens``, grid 0); a caller buffer for
    the derived prefix sum brings the preprocess back to publish it."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    runner = _cpu_forwarding_runner(module, monkeypatch, calls, has_varlen=True)
    seqlen = 300
    x = torch.empty((1, seqlen, 1, 64), dtype=torch.bfloat16)
    dt = torch.empty((1, seqlen, 1), dtype=torch.float32)
    A = torch.empty((1,), dtype=torch.float32)
    B = torch.empty((1, seqlen, 1, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)
    cu_seqlens = torch.tensor([0, 150, 300], dtype=torch.int32)

    runner.run(x, dt, A, B, C, cu_seqlens=cu_seqlens)
    name, launch = calls[-1]
    assert name == "exactfused_bf16_varlen"
    assert launch["preprocess_grid"] == (0, 1, 1)
    assert launch["preprocess"]["metadata_from_cu_seqlens"] == 1
    assert launch["main"]["metadata_from_cu_seqlens"] == 1
    assert launch["main"]["cu_seqlens"] is launch["preprocess"]["cu_seqlens"]
    assert (
        launch["main"]["preprocess_status"] is launch["preprocess"]["preprocess_status"]
    )

    buffer = torch.empty(3, dtype=torch.int32)
    runner.run(
        x,
        dt,
        A,
        B,
        C,
        cu_seqlens=cu_seqlens,
        seq_chunk_cumsum=buffer,
        update_seq_chunk_cumsum=True,
    )
    name, launch = calls[-1]
    assert name == "exactfused_bf16_varlen"
    assert launch["preprocess_grid"][0] >= 1
    assert launch["preprocess"]["seq_chunk_cumsum"] is buffer
    assert launch["main"]["seq_chunk_cumsum"] is buffer


def test_source_tail_route_mirrors_the_kernel_tests_rule():
    """``tail_selected`` mirrors the Cake kernel-tests ``tail_route`` for the
    two-kernel family: a two-kernel exact-scan call runs the ``_tail`` pair
    program where its balanced persistent grid leaves a tail wave, the
    host-known chunks per item reach 16 and the pair makespan in items
    ``ceil(items / pairs) / 2`` is below the persistent program's ``ceil(items
    / grid)``.  The expectation is recomputed here from the rule's definition
    over a grid of item and chunk counts on both devices (148 / 152 SMs)."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    route = module.tail_selected

    assert module._TAIL_MIN_CHUNKS == 16
    assert module._tail_grid_size(sm_count=148) == 148
    assert module._tail_grid_size(sm_count=152) == 152
    assert module._tail_grid_size(sm_count=147) == 146

    def expected(items, chunks, sm_count):
        pairs = sm_count // 2
        grid = module._persistent_grid_size(total_work=items, sm_count=sm_count)
        tail_wave = items > sm_count and grid < sm_count
        return (
            tail_wave
            and chunks >= 16
            and math.ceil(items / pairs) / 2 < math.ceil(items / grid)
        )

    for sm_count in (148, 152):
        for items in (64, 128, 148, 152, 256, 296, 304, 384, 512, 640, 768, 1024, 2048):
            for chunks in (1, 4, 8, 15, 16, 17, 31, 32, 64):
                assert route(
                    total_work=items, chunks_per_item=chunks, sm_count=sm_count
                ) == expected(items, chunks, sm_count), (sm_count, items, chunks)
        # the measured rows: 512 items (4 sequences x 128 heads)
        assert route(total_work=512, chunks_per_item=16, sm_count=sm_count)
        assert route(total_work=512, chunks_per_item=32, sm_count=sm_count)
        assert not route(total_work=512, chunks_per_item=8, sm_count=sm_count)
        # 256 items: pair makespan 2.0 equals the persistent program's 2
        assert not route(total_work=256, chunks_per_item=32, sm_count=sm_count)
        # 1024 items: a full wave (grid 148 / 152), no tail wave
        assert not route(total_work=1024, chunks_per_item=32, sm_count=sm_count)
    # the chunks per item are the two-kernel route's quantity
    chunks = module.host_chunks_per_item
    assert chunks(mode_varlen=False, num_sequences=4, seqlen=2048) == 16
    assert chunks(mode_varlen=True, num_sequences=64, seqlen=64 * 4096) == 32
    assert chunks(mode_varlen=True, num_sequences=3, seqlen=3 * 2048 - 1) == 16
    assert chunks(mode_varlen=True, num_sequences=0, seqlen=128) == 1
    assert module.two_kernel_family(
        mode_varlen=False, num_sequences=4, seqlen=2048
    ) == ("exactpre")
    assert module.two_kernel_family(
        mode_varlen=False, num_sequences=16, seqlen=128
    ) == ("exact")
    # no call inside the fused window reaches the pair program's floor with a
    # tail wave: batched fused calls span at most 16 chunks only up to 2048
    # chunk-items, varlen fused calls at most 256 items
    assert not (
        module.fused_selected(
            nheads=128,
            num_sequences=4,
            nchunks=16,
            mode_varlen=False,
            total_tokens=8192,
        )
        and route(total_work=512, chunks_per_item=16, sm_count=148)
    )
    assert (
        module.fused_selected(
            nheads=128,
            num_sequences=4,
            nchunks=16,
            mode_varlen=False,
            total_tokens=8192,
        )
        is False
    )


def test_source_tail_programs_are_bound():
    """The six ``_tail`` programs (two-kernel family, three state dtypes,
    batched and varlen) bind the family's host template and main arguments,
    the tail SMEM literal, the ``tail`` variant and the 2-CTA cluster shape;
    the ring programs keep their variant, the family ships no ``_noring``
    program (its items span at least two chunks), and the fused and
    retained ``exact`` families have no pair program."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    names = [
        f"exactpre_{state}_{mode}_tail"
        for state in ("bf16", "f16", "f32")
        for mode in ("batched", "varlen")
    ]
    assert len(names) == 6
    for name in names:
        program = module._PROGRAMS[name]
        ring = module._PROGRAMS[name.rsplit("_", 1)[0]]
        assert program.variant == "tail"
        assert program.family == ring.family == "exactpre"
        assert (
            program.main_smem_bytes == module._VARIANT_SMEM_BYTES[("exactpre", "tail")]
        )
        assert program.host_template == ring.host_template
        assert program.main_args == ring.main_args
        assert program.cluster_dims == (2, 1, 1) and ring.cluster_dims == (1, 1, 1)
        assert ring.variant == ""
        assert f"{name.rsplit('_', 1)[0]}_noring" not in module._PROGRAMS
    assert not any(name.endswith("_tail") for name in module._SCAN_MODULES)
    assert not any(name.endswith("_tail") for name in module._EXACTFUSED_MODULES)
    assert set(module._VARIANT_CLUSTER_DIMS) == {("exactpre", "tail")}
    assert (
        module._program_name("exactpre", torch.bfloat16, False, "tail")
        == "exactpre_bf16_batched_tail"
    )


def test_source_sm107_varlen_tail_programs_launch_with_their_smem():
    """On sm_107a the varlen tail-pair renders (``_SM107_MODULES``) launch with
    ``_EXACTPRE_TAIL_VARLEN_SM107_SMEM_BYTES`` (their segment-bounds words sit
    past the pool of the other tail programs); the batched tail-pair programs,
    every program on sm_100a / sm_103a and every other sm_107a program keep
    their table literal."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    own_smem = module._EXACTPRE_TAIL_VARLEN_SM107_SMEM_BYTES
    assert {
        ("exactpre", "varlen", "tail"): own_smem
    } == module._SM107_VARIANT_SMEM_BYTES
    assert (own_smem, module._EXACTPRE_TAIL_SMEM_BYTES) == (232448, 232320)
    owners = set()
    for name, program in module._PROGRAMS.items():
        for arch in ("sm_100a", "sm_103a", "sm_107a"):
            built = module._arch_program(name, arch)
            own = (
                arch == "sm_107a"
                and name in module._SM107_MODULES
                and name.startswith("exactpre_")
                and name.endswith("_varlen_tail")
            )
            if own:
                owners.add(name)
            expected = own_smem if own else program.main_smem_bytes
            assert built.main_smem_bytes == expected, (name, arch)
            assert built.cluster_dims == program.cluster_dims
    assert owners == {
        f"exactpre_{state}_varlen_tail" for state in ("bf16", "f16", "f32")
    }


# ---- every name the route composes ships; every shipped program is named ------

# Representative host-shape grid of the exact-scan route: one to three items of
# one to three chunks (50 and 64 tokens lie inside the batched and the varlen
# ``_noring`` window, 128 closes the one-chunk range, 200 and 384 span two and
# three chunks) and the pair program's 512-item x 16-chunk shape at 128 heads;
# 576 heads is the retained family's batched corner above the fused window.
_ROUTE_CELLS = (
    *((items, tokens) for items in (1, 2, 3) for tokens in (50, 64, 128, 200, 384)),
    (4, 2048),
)
_ROUTE_HEADS = (128, 576)
_ROUTE_STATE_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_ROUTE_SM_COUNTS = {"sm_100a": 148, "sm_103a": 152}
# call forms per mode: batched, or packed varlen as ``cu_seqlens``, the seq_idx
# triple deriving its table and the triple with the caller's trusted table
_ROUTE_FORMS = {False: ("batched",), True: ("cu_seqlens", "derived", "trusted")}


def _route_shapes():
    """``(items, tokens, nheads, mode_varlen, has_z, state_dtype)`` over the grid."""

    for nheads in _ROUTE_HEADS:
        for mode_varlen in (False, True):
            for has_z in (False, True):
                for state_dtype in _ROUTE_STATE_DTYPES:
                    for items, tokens in _ROUTE_CELLS:
                        yield items, tokens, nheads, mode_varlen, has_z, state_dtype


def test_source_route_functions_name_shipped_programs():
    """Every name the route functions compose over the grid is a shipped
    program -- ``two_kernel_family`` x ``_exact_scan_variant`` (the two-kernel
    route), the fused families with the same variant, the pair program where
    ``tail_selected`` fires and the chunk-parallel family of every override --
    and the table holds exactly the names so composed, besides its sm_107a-only
    programs: no entry is dead and no route names a missing program.  A ``_noring`` program never pairs with the
    two-kernel family (its items span at least two chunks, the ``_noring``
    windows one) and never with an f32 state on the varlen path."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    produced = set()
    for items, tokens, nheads, mode_varlen, has_z, state_dtype in _route_shapes():
        total = items * tokens
        # the host ``seqlen`` of the call: packed varlen x is [1, total, ...]
        seqlen, batch = (total, 1) if mode_varlen else (tokens, items)
        family = module.two_kernel_family(
            mode_varlen=mode_varlen, num_sequences=items, seqlen=seqlen
        )
        variant = module._exact_scan_variant(
            mode_varlen=mode_varlen,
            has_z=has_z,
            nheads=nheads,
            num_sequences=items,
            total_tokens=total,
            batch=batch,
            seqlen=seqlen,
            state_dtype=state_dtype,
        )
        chunks = module.host_chunks_per_item(
            mode_varlen=mode_varlen, num_sequences=items, seqlen=seqlen
        )
        if variant == "noring":
            assert chunks == 1 and family == "exact", (items, tokens, mode_varlen)
            assert not (mode_varlen and state_dtype is torch.float32)
        fused_families = (
            ("exactfused", "exactmeta", "exactone") if mode_varlen else ("exactfused",)
        )
        names = [
            module._program_name(name_family, state_dtype, mode_varlen, variant)
            for name_family in (family, *fused_families)
        ]
        for sm_count in _ROUTE_SM_COUNTS.values():
            if module.tail_selected(
                total_work=items * nheads, chunks_per_item=chunks, sm_count=sm_count
            ):
                assert family == "exactpre"
                names.append(
                    module._program_name(family, state_dtype, mode_varlen, "tail")
                )
        for override in ("always", "three_phase", "never"):
            forced = module.chunk_parallel_family(
                nheads=nheads,
                num_sequences=items,
                num_segments=items * chunks,
                nchunks=chunks,
                mode_varlen=mode_varlen,
                sm_count=148,
                capability=(10, 0),
                mode=override,
            )
            if forced != "exact":
                names.append(module._program_name(forced, state_dtype, mode_varlen))
        for name in names:
            assert name in module._PROGRAMS, (name, items, tokens, nheads, has_z)
        produced.update(names)
    # The sm_107a-only claim and initial-state bulk programs: ``claim_selected`` and
    # ``ibulk_selected`` name them on capability (10, 7) alone
    # (test_source_sm107_claim_rule, test_source_sm107_ibulk_rule).
    assert produced == set(module._PROGRAMS) - module._SM107_ONLY_PROGRAMS


def _cpu_route_call(items, tokens, nheads, mode_varlen, has_z, form):
    """CPU tensors and keyword arguments of one call of the grid on a runner of
    ``nheads`` heads and 8 groups: batched (``items`` sequences of ``tokens``),
    or packed varlen in the ``cu_seqlens`` form, the seq_idx triple deriving
    its table (``derived``) or the triple with the caller's trusted table
    (``trusted``); the triple is the logical-chunk set of ``items`` equal
    sequences (every chunk start and every sequence start inside a chunk)."""

    batch, seqlen = (1, items * tokens) if mode_varlen else (items, tokens)
    x = torch.empty((batch, seqlen, nheads, 64), dtype=torch.bfloat16)
    dt = torch.empty((batch, seqlen, nheads), dtype=torch.float32)
    A = torch.empty((nheads,), dtype=torch.float32)
    B = torch.empty((batch, seqlen, 8, 128), dtype=torch.bfloat16)
    tensors = (x, dt, A, B, torch.empty_like(B))
    kwargs = {"z": torch.empty_like(x)} if has_z else {}
    if not mode_varlen:
        return tensors, kwargs
    if form == "cu_seqlens":
        kwargs["cu_seqlens"] = torch.arange(0, seqlen + 1, tokens, dtype=torch.int32)
        return tensors, kwargs
    starts = {sequence * tokens for sequence in range(items)}
    chunk_indices, chunk_offsets = [], []
    for chunk in range(-(-seqlen // 128)):
        low = chunk * 128
        inside = {start - low for start in starts if low < start < low + 128}
        for offset in sorted({0} | inside):
            chunk_indices.append(chunk)
            chunk_offsets.append(offset)
    sequence_ids = torch.arange(items, dtype=torch.int32)
    kwargs.update(
        seq_idx=torch.repeat_interleave(sequence_ids, tokens).reshape(1, seqlen),
        chunk_indices=torch.tensor(chunk_indices, dtype=torch.int32),
        chunk_offsets=torch.tensor(chunk_offsets, dtype=torch.int32),
        num_seqs=items,
    )
    if form == "trusted":
        cumsum, start = [0], 0
        for _ in range(items):
            end = start + tokens
            cumsum.append(cumsum[-1] + (-(-end // 128) - start // 128))
            start = end
        kwargs["seq_chunk_cumsum"] = torch.tensor(cumsum, dtype=torch.int32)
    return tensors, kwargs


def test_source_every_selected_program_is_shipped_without_gpu(monkeypatch):
    """``prepare`` of every call of the grid -- both capabilities on their SM
    counts, every call form, every override of the chunk-parallel family --
    resolves a shipped program whose family, state dtype, mode and variant are
    the call's, and the programs so resolved are exactly the table but its
    sm_107a-only programs: the one selection path (``_materialize_workspace``)
    names no missing program and leaves no table entry dead."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    monkeypatch.setattr(module, "_load_generated_program", lambda *_: None)
    seen = set()
    for arch, sm_count in _ROUTE_SM_COUNTS.items():
        capability = module._ARCH_CAPABILITIES[arch]
        for override in (None, "always", "three_phase", "never"):
            if override is None:
                monkeypatch.delenv(_CHUNK_PARALLEL_ENV, raising=False)
            else:
                monkeypatch.setenv(_CHUNK_PARALLEL_ENV, override)
            runner, runner_key = None, None
            for shape in _route_shapes():
                items, tokens, nheads, mode_varlen, has_z, state_dtype = shape
                if nheads != 128 and tokens > 384:
                    continue  # the pair-program shape at the production head count
                if override is not None and (
                    has_z or nheads != 128 or (items, tokens) not in ((2, 64), (2, 384))
                ):
                    continue  # the override only replaces the family
                key = (nheads, mode_varlen, has_z, state_dtype)
                if key != runner_key:
                    runner = _cpu_forwarding_runner(
                        module,
                        monkeypatch,
                        [],
                        arch=arch,
                        nheads=nheads,
                        ngroups=8,
                        state_dtype=state_dtype,
                        has_varlen=mode_varlen,
                        has_z=has_z,
                    )
                    monkeypatch.setattr(module, "_sm_count", lambda _, n=sm_count: n)
                    runner_key = key
                for form in _ROUTE_FORMS[mode_varlen]:
                    (x, dt, A, B, C), kwargs = _cpu_route_call(
                        items, tokens, nheads, mode_varlen, has_z, form
                    )
                    prepared = runner.prepare(x=x, dt=dt, A=A, B=B, C=C, **kwargs)
                    name = prepared.program_name
                    cell = (arch, override, form, items, tokens, *key)
                    assert name in module._PROGRAMS, (name, cell)
                    program = module._PROGRAMS[name]
                    family, state_key, mode, *variant = name.split("_")
                    assert program.family == family, (name, cell)
                    assert state_key == module._STATE_DTYPE_KEYS[state_dtype], cell
                    assert mode == ("varlen" if mode_varlen else "batched"), cell
                    assert program.variant == (variant[0] if variant else ""), cell
                    if override is not None:
                        forced = module.chunk_parallel_family(
                            nheads=nheads,
                            num_sequences=items,
                            num_segments=items * -(-tokens // 128),
                            nchunks=-(-tokens // 128),
                            mode_varlen=mode_varlen,
                            sm_count=sm_count,
                            capability=capability,
                            mode=override,
                        )
                        if forced == "exact":
                            assert not family.startswith("chunkpar"), (name, cell)
                        else:
                            assert family == forced, (name, cell)
                    seen.add(name)
    # The sm_107a-only claim and initial-state bulk programs are routed on capability
    # (10, 7) alone (test_source_sm107_many_one_chunk_items_take_the_claim_program,
    # test_source_sm107_short_item_calls_take_the_ibulk_program).
    assert seen == set(module._PROGRAMS) - module._SM107_ONLY_PROGRAMS
