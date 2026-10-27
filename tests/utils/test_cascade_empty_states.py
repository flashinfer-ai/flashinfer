"""Empty partitions must be neutral in every cascade merge implementation."""

import math

import pytest
import torch

import flashinfer
from flashinfer.trace.templates import cascade as references


@pytest.fixture(params=["cuda", "triton", "reference"])
def backend(request):
    if request.param != "reference" and not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    if request.param == "triton":
        pytest.importorskip("triton")
    return request.param


def _states(num_states, dtype, device, head_dim):
    torch.manual_seed(0)
    v = torch.randn(4, num_states, 2, head_dim, dtype=dtype, device=device)
    s = torch.randn(4, num_states, 2, device=device)
    s[0] = -torch.inf
    if num_states > 1:
        s[1, ::2, 0] = -torch.inf
        s[1, 1::2, 1] = -torch.inf
    # A finite initializer such as -5e4 must not contribute attention mass.
    s[3] = -60000
    if num_states > 1:
        s[3, 0] = -torch.inf
    v.masked_fill_(torch.isneginf(s).unsqueeze(-1), 0)
    return v, s


def _expected(v, s):
    """Use a float64 softmax over only nonempty partitions as the oracle."""
    values, scores = v.cpu().double(), s.cpu().double()
    seq_len, _, num_heads, head_dim = values.shape
    output = torch.zeros(seq_len, num_heads, head_dim, dtype=torch.float64)
    lse = torch.full((seq_len, num_heads), -torch.inf, dtype=torch.float64)
    for row in range(seq_len):
        for head in range(num_heads):
            valid = torch.isfinite(scores[row, :, head])
            if valid.any():
                logits = scores[row, valid, head] * math.log(2)
                output[row, head] = (
                    logits.softmax(0)[:, None] * values[row, valid, head]
                ).sum(0)
                lse[row, head] = logits.logsumexp(0) / math.log(2)
    return output.to(v), lse.to(s)


def _check(actual, expected, dtype):
    v, s = actual
    v_ref, s_ref = expected
    torch.testing.assert_close(
        v.to(dtype), v_ref, atol=1e-3 if dtype == torch.float16 else 1e-2, rtol=1e-3
    )
    torch.testing.assert_close(s, s_ref, atol=1e-5, rtol=1e-6)
    empty = torch.isneginf(s_ref)
    assert torch.equal(v[empty], torch.zeros_like(v[empty]))
    assert torch.isneginf(s[empty]).all()


def _run(backend, name, v, s, mask=None):
    if name == "merge_states":
        args = (v, s)
    else:
        args = tuple(x.contiguous() for x in (v[:, 0], s[:, 0], v[:, 1], s[:, 1]))
        if name == "merge_state_in_place":
            args += (mask,)
    if backend == "reference":
        return getattr(references, f"{name}_trace").reference(*args)
    if backend == "cuda":
        result = getattr(flashinfer, name)(*args)
        return args[:2] if result is None else result

    # The Python Triton wrappers restrict devices to Hopper. Exercise the
    # portable kernels directly so this regression also runs on SM80/SM89.
    from flashinfer.triton.kernels import cascade as kernels

    seq_len, num_states, num_heads, head_dim = v.shape
    output = torch.empty(seq_len, num_heads, head_dim, dtype=v.dtype, device=v.device)
    lse = torch.empty(seq_len, num_heads, dtype=s.dtype, device=s.device)
    launch = dict(bdx=head_dim, bdy=num_heads)
    if name == "merge_states":
        kernels.merge_states_kernel[(seq_len,)](
            v, s, output, lse, num_states, num_heads, head_dim, **launch
        )
    elif name == "merge_state":
        kernels.merge_state_kernel[(seq_len,)](
            *args, output, lse, num_heads, head_dim, **launch
        )
    else:
        kernels.merge_state_in_place_kernel[(seq_len,)](
            *args[:4], num_heads, head_dim, mask, **launch
        )
        return args[:2]
    return output, lse


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_merge_state_empty_partitions(backend, dtype, head_dim):
    device = "cpu" if backend == "reference" else "cuda"
    v, s = _states(2, dtype, device, head_dim)
    _check(_run(backend, "merge_state", v, s), _expected(v, s), dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("masked", [False, True])
def test_merge_state_in_place_empty_partitions(backend, dtype, masked):
    device = "cpu" if backend == "reference" else "cuda"
    v, s = _states(2, dtype, device, 64)
    expected_v, expected_s = _expected(v, s)
    mask = None
    if masked:
        mask = torch.tensor([True, False, True, False], device=device)
        expected_v[~mask] = v[~mask, 0]
        expected_s[~mask] = s[~mask, 0]
    actual = _run(backend, "merge_state_in_place", v, s, mask)
    _check(actual, (expected_v, expected_s), dtype)
    if masked:
        assert torch.equal(actual[0][~mask], expected_v[~mask])
        assert torch.equal(actual[1][~mask], expected_s[~mask])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_states", [0, 1, 2, 4, 17])
def test_merge_states_empty_partitions(backend, dtype, head_dim, num_states):
    # CUDA selects its large-index-set kernel when num_states >= seq_len.
    # 4 hits that boundary; 17 also leaves empty lanes in the last reduction tile.
    device = "cpu" if backend == "reference" else "cuda"
    v, s = _states(num_states, dtype, device, head_dim)
    _check(_run(backend, "merge_states", v, s), _expected(v, s), dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_triton_variable_length_empty_partitions(dtype, head_dim):
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    pytest.importorskip("triton")
    from flashinfer.triton.kernels.cascade import variable_length_merge_states_kernel

    values, scores, expected_v, expected_s = [], [], [], []
    for length, row in [(0, 0), (2, 0), (3, 1), (17, 3)]:
        v, s = _states(length, dtype, "cuda", head_dim)
        v, s = v[row : row + 1], s[row : row + 1]
        ref_v, ref_s = _expected(v, s)
        values.append(v.squeeze(0))
        scores.append(s.squeeze(0))
        expected_v.append(ref_v)
        expected_s.append(ref_s)
    v, s = torch.cat(values), torch.cat(scores)
    indptr = torch.tensor([0, 0, 2, 5, 22], dtype=torch.int32, device="cuda")
    output = torch.empty(4, 2, head_dim, dtype=dtype, device="cuda")
    lse = torch.empty(4, 2, device="cuda")
    variable_length_merge_states_kernel[(4,)](
        v, s, indptr, output, lse, 2, head_dim, bdx=head_dim, bdy=2
    )
    _check((output, lse), (torch.cat(expected_v), torch.cat(expected_s)), dtype)
