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

import importlib
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
# ``cake_ssd_accuracy_probe.py`` (FP16 delta): <= 0.15 % of the outputs on the
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
    (``cake_ssd_accuracy_probe.py``; outputs outside 1e-2 of the fp64
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
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

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
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

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
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

    constructor, tensors, arguments = _case(d_has_hdim=d_has_hdim)
    d_shape = (8, 64) if runtime_d_has_hdim else (8,)
    arguments["D"] = torch.randn(*d_shape, device="cuda").to(torch.bfloat16)
    expected = SSDCombined(**constructor, backend="cute").run(*tensors, **arguments)
    actual = SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)

    _assert_cake_accuracy(actual, expected, constructor, tensors, arguments)


def test_cake_ssd_combined_updates_caller_buffers():
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

    constructor, tensors, arguments = _case(varlen=True)
    expected = SSDCombined(**constructor, backend="cute").run(*tensors, **arguments)
    expected_cumsum = arguments["seq_chunk_cumsum"]
    actual_cumsum = torch.full_like(expected_cumsum, -1)
    # Token-major caller storage [batch, seqlen, nheads, headdim], written by
    # the kernels and returned as-is.
    out = torch.empty((1, 256, 8, 64), dtype=torch.bfloat16, device="cuda")
    runner = SSDCombined(**constructor, backend="cake")
    allocated = runner.run(*tensors, **arguments)
    actual = runner.run(
        *tensors,
        **{
            **arguments,
            "seq_chunk_cumsum": actual_cumsum,
            "update_seq_chunk_cumsum": True,
            "out": out,
        },
    )

    torch.testing.assert_close(actual_cumsum, expected_cumsum, rtol=0, atol=0)
    assert actual[0] is out
    # Caller-owned output storage and a kernel-written cumsum must not change
    # the arithmetic: bit-identical to the allocating run.
    torch.testing.assert_close(actual[0], allocated[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], allocated[1], rtol=0, atol=0)
    _assert_cake_accuracy(actual, expected, constructor, tensors, arguments)

    preserved_cumsum = actual_cumsum.clone()
    runner.run(
        *tensors,
        **{
            **arguments,
            "seq_chunk_cumsum": actual_cumsum,
            "update_seq_chunk_cumsum": False,
        },
    )
    torch.testing.assert_close(actual_cumsum, preserved_cumsum, rtol=0, atol=0)


@pytest.mark.parametrize("state_dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("varlen", (False, True), ids=("batched", "varlen"))
def test_cake_ssd_combined_writes_selected_checkpoint_state(varlen, state_dtype):
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

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
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

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
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

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
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

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
    packed varlen ``total < 128``, down to a single token).  The generated
    host pins a 128-row TMA box on the token axis, so the runner binds the
    x/B/C/out maps to zero-padded one-chunk buffers and copies the valid
    output rows back; the result must match the fp64 recurrence at least as
    well as CuTe on the zero-padded packed stream.

    Single-token f16-state rows: the state is the initial state, so the
    rounding of the f16 initial state to the bf16 ``C . state`` MMA operand
    (done by both kernels; see the softplus-off row of
    ``test_cake_ssd_combined_exact_scan_softplus_parity``) dominates and
    both kernels put 6 of the 512 outputs outside 1e-2 of the recurrence on
    GB300 (1.17 %, identical counts), hence the same 2 % cap for those two
    rows; every other row stays under the 1 % default."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

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
    the zero-padded buffers, never from stale SMEM."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

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
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

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
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

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
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

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


@pytest.mark.parametrize(
    "lengths,tail,seq_idx_dtype",
    [
        ((100, 150), 70, torch.int32),
        ((128, 128), 64, torch.int32),
        ((100, 150), 70, torch.int64),
    ],
    ids=("tail-shares-chunk", "tail-opens-chunk", "tail-shares-chunk-i64"),
)
def test_cake_ssd_combined_flags_out_of_range_seq_idx(lengths, tail, seq_idx_dtype):
    """CAKE-990: ``seq_idx`` ids ``0 .. num_seqs`` with ``num_seqs`` declared
    sequences (the trailing id is out of range).  No out-of-bounds write
    (the caller's ``[num_seqs + 1]`` table receives exactly the in-range
    boundaries), the status word is set, the in-range sequences are bitwise
    those of the clean problem and correct against CuTe / fp64, and a clean
    call afterwards leaves the reset word at 0."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

    num_seqs = len(lengths)
    total = sum(lengths)
    constructor, tensors, arguments = _case(
        varlen=True, lengths=(*lengths, tail), seq_idx_dtype=seq_idx_dtype
    )
    x, dt, A, B, C = tensors
    table = torch.full((num_seqs + 1,), -7, dtype=torch.int32, device="cuda")
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
    runner = SSDCombined(**constructor, backend="cake")
    cake = runner._cake_runner
    cake.seq_idx_status(reset=True)
    clean_out, clean_final = runner.run(*clean_tensors, **clean_arguments)
    assert cake.seq_idx_status() == 0

    out, final = runner.run(*tensors, **flagged_arguments)

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
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")


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
    lengths, state_dtype, initial_states
):
    """The cu_seqlens form (the device derives seq_chunk_cumsum and the
    chunk-128 tables) is bitwise the chunk-128 triple call of the same
    inputs on every sequence geometry, with ``seq_idx`` optional and never
    read; the status word stays clear."""

    _skip_unless_cake_arch()
    constructor, tensors, arguments = _case(
        varlen=True,
        lengths=lengths,
        state_dtype=state_dtype,
        initial_states=initial_states,
    )
    runner = SSDCombined(**constructor, backend="cake")
    cake = runner._cake_runner
    cake.seq_idx_status(reset=True)
    out_triple, final_triple = (
        value.clone() for value in runner.run(*tensors, **arguments)
    )
    assert cake.seq_idx_status() == 0

    out, final = runner.run(*tensors, **_cu_seqlens_arguments(arguments, lengths))

    assert cake.seq_idx_status(reset=True) == 0
    assert final.dtype == state_dtype
    assert tuple(final.shape) == (len(lengths), 8, 64, 128)
    assert torch.isfinite(out.to(torch.float32)).all()
    assert torch.equal(out, out_triple), "cu_seqlens form differs from the triple"
    assert torch.equal(final, final_triple)
    # seq_idx riding along (callers that still build it) changes nothing.
    out_ride, final_ride = runner.run(
        *tensors, **_cu_seqlens_arguments(arguments, lengths, keep_seq_idx=True)
    )
    assert torch.equal(out_ride, out_triple) and torch.equal(final_ride, final_triple)
    # The public functional entry takes the same form (it infers the state
    # dtype from initial_states, BF16 without them).
    if initial_states or state_dtype == torch.bfloat16:
        functional = importlib.import_module("flashinfer.mamba.ssd_combined")
        out_fn, final_fn = functional.ssd_combined_fwd(
            *tensors, **_cu_seqlens_arguments(arguments, lengths)
        )
        assert torch.equal(out_fn, out_triple) and torch.equal(final_fn, final_triple)


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
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

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
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

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
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

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
        torch.cuda.get_device_capability(index) not in ((10, 0), (10, 3))
        for index in (0, 1)
    ):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

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
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

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
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

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
        with pytest.raises(ValueError, match="requires SM100 or SM103, got SM110"):
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

    exact = calls["exact_bf16_batched"]
    assert exact["preprocess"]["dt_softplus"] == 0
    assert exact["preprocess"]["write_seq_chunk_cumsum"] == 0
    assert exact["preprocess_grid"] == (1, 1, 1)
    main = exact["main"]
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
        ("chunkpar", torch.bfloat16, False, "chunkpar_bf16_batched"),
        ("chunkpar", torch.bfloat16, True, "chunkpar_bf16_varlen"),
        ("chunkpar", torch.float16, False, "chunkpar_f16_batched"),
        ("chunkpar", torch.float16, True, "chunkpar_f16_varlen"),
        ("chunkpar", torch.float32, False, "chunkpar_f32_batched"),
        ("chunkpar", torch.float32, True, "chunkpar_f32_varlen"),
    ],
)
def test_source_program_name_covers_every_state_dtype(
    family, state_dtype, mode_varlen, expected
):
    """Both kernel families serve every admitted input: the program follows
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
    """A real runner on CPU tensors whose launcher call is captured."""

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(module, "_target_arch", lambda *_: "sm_103a")
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
        has_z=False,
        seq_idx_dtype=constructor.pop("seq_idx_dtype", torch.int32),
    )


@pytest.mark.parametrize(
    "case",
    (
        "precomputed",
        "update_caller_buffer",
        "runner_buffer_from_initial_states",
        "runner_buffer_from_num_seqs",
    ),
)
def test_source_varlen_cumsum_binding_without_gpu(monkeypatch, case):
    """The preprocess owns ``seq_chunk_cumsum``: one launcher call binds the
    same vector to both stages and ``write_seq_chunk_cumsum`` says whether
    the preprocess fills it; ``num_seqs`` supplies the count when there are
    no initial states."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    has_initial_states = case != "runner_buffer_from_num_seqs"
    runner = _cpu_forwarding_runner(
        module,
        monkeypatch,
        calls,
        has_varlen=True,
        has_initial_states=has_initial_states,
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
    assert name == "exact_bf16_varlen"
    preprocess, main = launch["preprocess"], launch["main"]
    assert preprocess["seq_chunk_cumsum"] is main["seq_chunk_cumsum"]
    assert preprocess["num_sequences"] == main["sequence_count"] == num_seqs
    assert preprocess["seq_idx_i32"] is main["seq_idx_i32"] is kwargs["seq_idx"]
    assert preprocess["seq_idx_int64"] == 0
    assert main["has_initial"] == int(has_initial_states)
    assert main["nchunks"] == 3 and main["seqlen"] == seqlen
    if case == "precomputed":
        assert preprocess["seq_chunk_cumsum"] is caller_vector
        assert preprocess["write_seq_chunk_cumsum"] == 0
    elif case == "update_caller_buffer":
        assert preprocess["seq_chunk_cumsum"] is caller_vector
        assert preprocess["write_seq_chunk_cumsum"] == 1
    else:
        assert preprocess["write_seq_chunk_cumsum"] == 1
        bound = preprocess["seq_chunk_cumsum"]
        assert bound.dtype == torch.int32 and bound.numel() == num_seqs + 1
        # The runner-owned vector is reused across calls of the same count.
        runner.run(x, dt, A, B, C, **kwargs)
        assert calls[-1][1]["preprocess"]["seq_chunk_cumsum"] is bound


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
    assert name == "exact_bf16_varlen"
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
        assert name == "exact_bf16_batched"
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
    assert name == "exact_bf16_batched"
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
    """CAKE-1063: with fewer than 128 tokens the x/B/C/out tensor maps are
    bound to runner-owned one-chunk buffers (valid rows copied in, pad rows
    zero, staged output copied back to the returned ``out``) while every
    logical argument -- ``seqlen``, ``nchunks``, the preprocess tables -- keeps
    the caller's extent; a 128-token call binds the caller's tensors."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    runner = _cpu_forwarding_runner(
        module, monkeypatch, calls, nheads=2, ngroups=1, has_varlen=varlen
    )

    def launch(name, _arch, **launch_kwargs):
        calls.append((name, launch_kwargs))
        # Stand in for the kernel: write a row pattern into the bound output.
        staged = launch_kwargs["main"]["out_map"]
        rows = torch.arange(staged.shape[1], dtype=torch.float32) + 1.0
        staged.copy_(rows.reshape(1, -1, 1, 1).expand(staged.shape).to(staged.dtype))

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
    for key, source in (("x_map", x), ("b_map", B), ("c_map", C)):
        bound = main[key]
        assert tuple(bound.shape) == (batch, 128, *source.shape[2:])
        assert bound.dtype == torch.bfloat16
        torch.testing.assert_close(bound[:, :seqlen], source, rtol=0, atol=0)
        assert not bound[:, seqlen:].any(), key
    staged = main["out_map"]
    assert staged is main["out_native"]
    assert tuple(staged.shape) == (batch, 128, 2, 64)
    torch.testing.assert_close(out, staged[:, :seqlen], rtol=0, atol=0)
    assert float(out[0, -1, 0, 0]) == float(seqlen)
    # Logical extents are untouched by the padding.
    assert main["seqlen"] == seqlen and main["nchunks"] == 1
    assert main["batch"] == batch and main["num_logical_chunks"] == 1
    assert preprocess["seqlen"] == seqlen and preprocess["num_segments"] == batch
    if not varlen:
        assert preprocess["segment_lengths"].tolist() == [seqlen] * batch
    assert main["dt"].shape == (batch, seqlen, 2)

    # A second call of the same extent reuses the buffers and leaves the pad
    # rows zero (nothing but the first ``seqlen`` rows is ever written).
    x2, dt2, A2, B2, C2 = inputs(seqlen)
    runner.run(x2, dt2, A2, B2, C2, **kwargs(seqlen))
    main2 = calls[-1][1]["main"]
    assert main2["x_map"] is main["x_map"] and main2["out_map"] is staged
    assert not main2["x_map"][:, seqlen:].any()

    # One full chunk: the caller's tensors are bound directly.
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
        12,
        1,
        1,
        *(main[name] for name in module._MAIN_ARGS),
        148,
        1,
        1,
        0x1234,
    )
    assert len(bound) == 27 + 3 + 41 + 3 + 1

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

    assert _host_prepare_arguments(template, "preprocess") == module._PREPROCESS_ARGS
    assert _host_prepare_arguments(template, "main") == module._MAIN_ARGS
    assert _host_run_arguments(template, "preprocess") == module._PREPROCESS_ARGS
    assert _host_run_arguments(template, "main") == module._MAIN_ARGS
    assert module._PREPROCESS_ARGS[-1] == "preprocess_status"
    assert len(module._PREPROCESS_ARGS) == 27
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
    """The runner owns one zeroed int32 status word per device, binds it as
    ``preprocess_status`` on every call and exposes it through the
    synchronizing debug accessor ``seq_idx_status``."""

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
    assert first.dtype == torch.int32 and tuple(first.shape) == (1,)
    assert first.device == cpu and int(first.item()) == 0
    assert runner.seq_idx_status(device=cpu) == 0
    first[0] = 1
    assert runner.seq_idx_status(device=cpu) == 1
    assert runner.seq_idx_status(reset=True, device=cpu) == 1
    assert runner.seq_idx_status(device=cpu) == 0
    assert int(first.item()) == 0


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
        "exact_bf16_varlen",
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
    """Every program of both families binds shipped generated sources: the
    one preprocess (one warp per tile of its block), its family's host
    template and main dynamic SMEM, and a main kernel whose symbol is the
    module identity without its hash.  Table entries the Cake export has
    not filled fail here by name."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    source_dir = module._source_dir()

    assert set(module._PROGRAMS) == {
        f"{family}_{state}_{mode}"
        for family in ("exact", "chunkpar")
        for state in ("bf16", "f16", "f32")
        for mode in ("batched", "varlen")
    }
    assert set(module._SCAN_MODULES) == {
        name for name in module._PROGRAMS if name.startswith("exact_")
    }
    assert set(module._CHUNKPAR_MODULES) == {
        name for name in module._PROGRAMS if name.startswith("chunkpar_")
    }
    pending = sorted(
        name
        for name, program in module._PROGRAMS.items()
        if module._PENDING_EXPORT in program.main.module or program.main_smem_bytes <= 0
    )
    assert not pending, f"programs awaiting the Cake export: {pending}"
    assert module._PENDING_EXPORT not in module._SEGMENT_PREPROCESS_MODULE
    assert (
        module._SEGMENT_PREPROCESS.threads
        == 32 * module._SEGMENT_PREPROCESS_TILES_PER_BLOCK
    )
    family_host = {
        "exact": (module._HOST_TEMPLATE, module._MAIN_ARGS, module._EXACT_SMEM_BYTES),
        "chunkpar": (
            module._CHUNKPAR_HOST_TEMPLATE,
            module._MAIN_ARGS_CHUNKPAR,
            module._CHUNKPAR_SMEM_BYTES,
        ),
    }
    templates = {
        family: (source_dir / template).read_text(encoding="utf-8")
        for family, (template, _, _) in family_host.items()
    }
    device_sources = set()
    for name, program in module._PROGRAMS.items():
        family, state_key, mode = name.split("_")
        template_path, main_args, smem_bytes = family_host[family]
        assert program.family == family
        assert program.host_template == template_path
        assert program.main_args == main_args
        assert program.main_smem_bytes == smem_bytes
        assert (program.state_dtype_code, program.state_dtype_bits) == (
            module._STATE_DTYPE_CODES[state_key]
        )
        assert program.preprocess is module._SEGMENT_PREPROCESS
        assert program.main.module.startswith(
            program.main.kernel.removeprefix("kernel_") + "_"
        )
        assert program.main.kernel.endswith(f"_{state_key}_{mode}")
        assert program.preprocess.threads > 0 and program.main.threads == 512
        assert not program.preprocess.fast_math and program.main.fast_math
        for kernel in program.kernels:
            source = source_dir / module._DEVICE_DIR / kernel.source
            assert source.is_file(), source
            assert re.search(
                rf"\b{kernel.kernel}\(", source.read_text(encoding="utf-8")
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
        assert f"stream, {program.main_smem_bytes}u)" in rendered
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
    # One shared source per physical kernel: six exact-scan and six
    # chunk-parallel main sources plus the one preprocess kernel, no
    # architecture copies and no orphans.
    assert len(device_sources) == 13
    assert sorted(
        path.name for path in (source_dir / module._DEVICE_DIR).glob("*.cu")
    ) == sorted(path.name for path in device_sources)


def test_active_source_package_declares_cuda_half_types_explicitly():
    """Every program carries ``delta`` in FP16 (the state may also be FP16),
    so each shipped device source must declare the CUDA half types itself."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    source_root = module._source_dir() / module._DEVICE_DIR
    # Programs the Cake export has not filled yet ship no source; they fail
    # by name in test_source_program_table_names_shipped_sources.
    sources = {
        kernel.source
        for program in module._PROGRAMS.values()
        for kernel in program.kernels
        if module._PENDING_EXPORT not in kernel.module
    }
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
# Chunk-parallel program family (``chunkpar_*``): the exact scan's arithmetic
# in a grid-barrier-separated schedule, selected per call by the calibrated
# cost rule mirrored from the Cake seed module.

_CHUNK_PARALLEL_ENV = "FLASHINFER_CAKE_SSD_CHUNK_PARALLEL"
_SELECTION_SM_COUNT = 148
_SELECTION_CAPABILITIES = ((10, 0), (10, 3))
_CHUNK_PARALLEL_WORKSPACE_ARGS = ("h_map", "s_work", "h_words", "grid_barrier")


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


@pytest.mark.parametrize("capability", _SELECTION_CAPABILITIES, ids=("sm100", "sm103"))
@pytest.mark.parametrize(
    "batch,nheads,nchunks,expected",
    (
        (1, 8, 16, True),
        (1, 8, 8, False),
        (1, 128, 32, False),
        (1, 8, 256, True),
        (4, 8, 64, True),
        (1, 32, 16, False),
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
    128 tiles), the 32768-token few-head prefill and 4 x 8192 x 8 qualify;
    8 chunks do not amortise the fixed cost, 128 heads pay for 4096 tiles,
    and 1 x 2048 x 32 (512 tiles) lands inside the margin."""

    selected = _batched_selection(batch, nheads, nchunks, capability=capability)
    assert selected is expected


def test_source_chunk_parallel_selection_quantities_follow_the_cost_model():
    """``selection_quantities`` reports the rule's inputs and both programs'
    predicted main-kernel times from the calibrated constants: the exact
    scan's per-chunk cost interpolates linearly between 32 and 128 work
    items, the chunk-parallel program pays its fixed cost plus one per-tile
    cost for every tile beyond the first wave of SMs; unmeasured
    capabilities use the B200 constants."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    model = module._CHUNK_PARALLEL_COST_MODEL_US
    assert set(model) == set(_SELECTION_CAPABILITIES)
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

    # The calibration row: 1 x 32768 tokens x 8 heads (the seed measured
    # 688 -> 160 us on B200 and 667 -> 153 us on B300).
    long_prefill = quantities(8, 256, (10, 0))["predicted_us"]
    assert long_prefill["exact_scan"] == pytest.approx(691.48)
    assert long_prefill["chunk_parallel"] == pytest.approx(149.03)
    long_prefill = quantities(8, 256, (10, 3))["predicted_us"]
    assert long_prefill["exact_scan"] == pytest.approx(668.24)
    assert long_prefill["chunk_parallel"] == pytest.approx(141.85)

    large = quantities(128, 32, (10, 3))
    b300 = model[(10, 3)]
    assert large["predicted_us"]["exact_scan"] == pytest.approx(
        b300["serial_fixed"] + b300["serial_per_chunk_large"] * 32
    )
    assert large["predicted_us"]["chunk_parallel"] == pytest.approx(
        b300["cp_fixed"] + b300["cp_per_tile"] * (4096 - 148)
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
    """``always`` / ``never`` force the family regardless of the rule; the
    ``FLASHINFER_CAKE_SSD_CHUNK_PARALLEL`` variable is read on every call,
    trimmed and case-folded, defaults to ``auto`` and rejects any other
    value."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    assert module._CHUNK_PARALLEL_ENV == _CHUNK_PARALLEL_ENV
    assert _batched_selection(1, 8, 8, mode="always") is True
    assert _batched_selection(1, 8, 256, mode="never") is False
    with pytest.raises(ValueError, match="auto, always or never"):
        _batched_selection(1, 8, 16, mode="sometimes")

    assert module._chunk_parallel_mode() == "auto"
    for value, expected in (
        ("always", "always"),
        (" Never ", "never"),
        ("AUTO", "auto"),
    ):
        monkeypatch.setenv(_CHUNK_PARALLEL_ENV, value)
        assert module._chunk_parallel_mode() == expected
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "1")
    with pytest.raises(ValueError, match=_CHUNK_PARALLEL_ENV):
        module._chunk_parallel_mode()


def test_source_chunk_parallel_main_arguments_extend_the_exact_order():
    """The chunk-parallel main kernel's parameter order (the generated host's
    positional ABI) is the exact scan's with the state-operand tensor map
    ``h_map`` after ``out_map`` and the workspace pointers ``s_work``,
    ``h_words``, ``grid_barrier`` after ``out_native``; each program binds
    its family's order and host template."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    expected = list(module._MAIN_ARGS)
    expected.insert(expected.index("out_map") + 1, "h_map")
    position = expected.index("out_native") + 1
    expected[position:position] = ["s_work", "h_words", "grid_barrier"]
    assert tuple(expected) == module._MAIN_ARGS_CHUNKPAR
    assert len(module._MAIN_ARGS) == 41 and len(module._MAIN_ARGS_CHUNKPAR) == 45
    assert set(module._MAIN_ARGS_CHUNKPAR) - set(module._MAIN_ARGS) == set(
        _CHUNK_PARALLEL_WORKSPACE_ARGS
    )
    for name, program in module._PROGRAMS.items():
        family = name.split("_")[0]
        assert program.family == family
        if family == "chunkpar":
            assert program.main_args == module._MAIN_ARGS_CHUNKPAR
            assert program.host_template == module._CHUNKPAR_HOST_TEMPLATE
        else:
            assert program.main_args == module._MAIN_ARGS
            assert program.host_template == module._HOST_TEMPLATE


def test_source_chunk_parallel_launch_orders_its_main_arguments(monkeypatch):
    """``_launch_program`` orders a chunk-parallel program's main values by
    ``_MAIN_ARGS_CHUNKPAR`` (45 values between the preprocess grid and the
    main grid) and fails closed when the four workspace values are missing."""

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
    assert len(calls[0]) == 27 + 3 + 45 + 3 + 1
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


@pytest.mark.parametrize("gap", ("module", "smem"))
def test_source_unexported_program_refuses_to_build(monkeypatch, tmp_path, gap):
    """A program whose table entries the Cake export has not filled -- a
    module identity still carrying the placeholder token or a zero main
    dynamic SMEM literal -- is refused by name, naming the export step,
    before any source is read or nvcc is resolved."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    exported = module._PROGRAMS["exact_bf16_batched"]
    module._require_exported("exact_bf16_batched", exported)
    if gap == "module":
        main = module._scan("mamba_ssd_chunk_parallel_bf16_batched_PENDINGEXPORT")
        smem_bytes = exported.main_smem_bytes
    else:
        main = module._scan("mamba_ssd_chunk_parallel_bf16_batched_0123456789")
        smem_bytes = 0
    assert main.kernel == "kernel_mamba_ssd_chunk_parallel_bf16_batched"
    program = module._Program(
        "chunkpar",
        module._SEGMENT_PREPROCESS,
        main,
        4,
        16,
        smem_bytes,
        module._CHUNKPAR_HOST_TEMPLATE,
        module._MAIN_ARGS_CHUNKPAR,
    )
    name = "chunkpar_bf16_batched_probe"
    monkeypatch.setitem(module._PROGRAMS, name, program)
    monkeypatch.setattr(module, "_source_dir", lambda: tmp_path)
    monkeypatch.setattr(
        module, "_nvcc", lambda: pytest.fail("nvcc resolved for an unexported program")
    )
    module._load_generated_program.cache_clear()

    with pytest.raises(RuntimeError, match="not exported yet") as excinfo:
        module._load_generated_program(name, "sm_100a")
    module._load_generated_program.cache_clear()

    message = str(excinfo.value)
    assert name in message and "export_cake_mamba_ssd_combined" in message
    assert not any(tmp_path.iterdir())


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
    program = module._PROGRAMS["chunkpar_bf16_varlen"]
    rendered = module._render_host_source(template, "chunkpar_bf16_varlen", program)
    assert not any(placeholder in rendered for placeholder in _HOST_PLACEHOLDERS)
    assert "namespace cake_mamba_ssd_combined_host_chunkpar_bf16_varlen {" in rendered
    assert f"TVM_FFI_EMBED_CUBIN({program.main.module});" in rendered


def test_source_chunk_parallel_forward_binds_the_workspace_without_gpu(monkeypatch):
    """Forced chunk-parallel, the runner names the ``chunkpar_*`` program and
    binds the four workspace values: ``h_map``, the contiguous bf16
    ``[tiles, 64, 128]`` state operand the host wraps in a TMA descriptor;
    ``s_work``, the f32 ``[tiles, 64, 128]`` state increments; ``h_words``,
    the u32 view of ``h_map``'s storage; ``grid_barrier``, two u32 words
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
    h_map, s_work, h_words, barrier = (
        main[key] for key in _CHUNK_PARALLEL_WORKSPACE_ARGS
    )
    assert tuple(h_map.shape) == (128, 64, 128) and h_map.dtype == torch.bfloat16
    assert h_map.is_contiguous()
    assert tuple(s_work.shape) == (128, 64, 128) and s_work.dtype == torch.float32
    assert tuple(h_words.shape) == (128, 64, 64) and h_words.dtype == torch.uint32
    assert h_words.data_ptr() == h_map.data_ptr()
    assert tuple(barrier.shape) == (2,) and barrier.dtype == torch.uint32
    assert barrier.view(torch.int32).tolist() == [0, 0]
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
    assert barrier.view(torch.int32).tolist() == [0, 7]
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
    assert name == runner.last_program_name == "exact_bf16_batched"
    assert set(launch["main"]) == set(module._MAIN_ARGS)
    assert launch["main_grid"] == (8, 1, 1)

    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "auto")
    runner.run(*inputs(32768))  # 8 work items on 148 SMs, 256 chunks
    assert calls[-1][0] == "chunkpar_bf16_batched"
    runner.run(*inputs(1024))  # 8 chunks: the exact scan
    assert calls[-1][0] == "exact_bf16_batched"
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "later")
    with pytest.raises(ValueError, match=_CHUNK_PARALLEL_ENV):
        runner.run(*inputs(1024))


def _state_key(state_dtype):
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    return module._STATE_DTYPE_KEYS[state_dtype]


def _forced_runs(monkeypatch, runner, tensors, arguments):
    """The same call on the forced exact scan (``never``) and the forced
    chunk-parallel program (``always``), per mode: the cloned output, the
    cloned final states, the checkpoint-state table the call wrote (a fresh
    copy of the caller's table per mode; ``None`` without checkpoints) and
    the program the runner reports."""

    results = {}
    for mode in ("never", "always"):
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
    return results


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
    """The chunk-parallel program computes the exact scan's arithmetic in a
    grid-barrier-separated schedule: forced on, its output and final states
    are bitwise the forced exact scan's -- batched calls with z, D, dt_bias,
    softplus, a finite clamp and initial states, and packed-varlen calls in
    both metadata forms, for every state dtype."""

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
    mode = "varlen" if varlen else "batched"
    assert exact.program == f"exact_{_state_key(state_dtype)}_{mode}"
    assert chunk_parallel.program == f"chunkpar_{_state_key(state_dtype)}_{mode}"
    assert chunk_parallel.final.dtype == state_dtype
    assert torch.isfinite(chunk_parallel.out.to(torch.float32)).all()
    assert torch.equal(chunk_parallel.out, exact.out), (
        "chunk-parallel output differs from the exact scan"
    )
    assert torch.equal(chunk_parallel.final, exact.final)


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
    """The chunk-parallel program's checkpoint publish is the exact scan's:
    forced on, the state it writes into each requested slot is bitwise the
    forced exact scan's (output and final states included) and the other
    slots stay untouched -- batched checkpoints on the chunk grid and at a
    sequence end, a chunk-unaligned checkpoint exposed through the caller's
    triple and one the cu_seqlens preprocess inserts itself, and an on-grid
    cu_seqlens checkpoint.  ``auto`` routes checkpoint calls like any other
    call, so the chunk-parallel path must serve them exactly."""

    _skip_unless_cake_arch()
    if form == "batched":
        constructor, tensors, arguments = _case(
            state_dtype=state_dtype, batch=2, seqlen=1024
        )
    else:
        constructor, tensors, arguments = _case(
            state_dtype=state_dtype, varlen=True, lengths=lengths
        )
        if form == "cu_seqlens":
            arguments = _cu_seqlens_arguments(arguments, lengths)
        else:
            # The triple must expose every checkpoint as a logical chunk end;
            # the runner derives the sequence prefix sum on the device.
            chunk_indices, chunk_offsets = _logical_chunk_metadata(
                lengths, tuple(token for token in tokens if token >= 0)
            )
            arguments = {
                **arguments,
                "chunk_indices": chunk_indices,
                "chunk_offsets": chunk_offsets,
                "seq_chunk_cumsum": None,
            }
    arguments = _checkpoint_arguments(arguments, tokens, slots)
    runner = SSDCombined(**constructor, backend="cake")

    results = _forced_runs(monkeypatch, runner, tensors, arguments)

    exact, chunk_parallel = results["never"], results["always"]
    mode = "batched" if form == "batched" else "varlen"
    assert exact.program == f"exact_{_state_key(state_dtype)}_{mode}"
    assert chunk_parallel.program == f"chunkpar_{_state_key(state_dtype)}_{mode}"
    assert torch.equal(chunk_parallel.out, exact.out)
    assert torch.equal(chunk_parallel.final, exact.final)
    written = {slot for slot in slots if slot >= 0}
    for slot in range(3):
        exact_slot = exact.checkpoint_states[slot]
        chunk_parallel_slot = chunk_parallel.checkpoint_states[slot]
        if slot in written:
            assert torch.isfinite(exact_slot.to(torch.float32)).all(), slot
            assert torch.equal(chunk_parallel_slot, exact_slot), (
                f"chunk-parallel checkpoint slot {slot} differs from the exact scan"
            )
        else:
            assert torch.isnan(exact_slot).all(), slot
            assert torch.isnan(chunk_parallel_slot).all(), slot


def test_cake_ssd_combined_auto_selects_chunk_parallel_for_long_prefill(monkeypatch):
    """``auto``: one 32768-token sequence with 8 heads (8 work items, 2048
    tiles) selects the chunk-parallel program on the device's own SM count
    and capability, reports it as ``last_program_name`` and is bitwise the
    forced exact scan."""

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
    assert cake.last_program_name == "exact_bf16_batched"
    assert torch.equal(out, out_exact) and torch.equal(final, final_exact)


def test_cake_ssd_combined_auto_keeps_the_exact_scan_for_many_heads(monkeypatch):
    """``auto``: 128 heads on a 1024-token sequence (128 work items, 1024
    tiles) stay on the exact scan -- at that head count the chunk-parallel
    per-tile cost exceeds the scan's per-chunk cost."""

    _skip_unless_cake_arch()
    constructor, tensors, arguments = _case(batch=1, seqlen=1024, nheads=128, ngroups=8)
    runner = SSDCombined(**constructor, backend="cake")

    out, final = runner.run(*tensors, **arguments)

    assert runner._cake_runner.last_program_name == "exact_bf16_batched"
    assert tuple(out.shape) == (1, 1024, 128, 64)
    assert tuple(final.shape) == (1, 128, 64, 128)
