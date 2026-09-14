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

import pytest
import torch

from flashinfer.mamba import SSDCombined

# Match the repository's CuTe parity contract. CAKE's separate tolerance
# against the sequential mathematical reference is not the candidate
# acceptance contract.
ATOL = RTOL = 1e-2


def _varlen_metadata(lengths, dtype):
    total = sum(lengths)
    seq_idx = torch.empty((1, total), dtype=dtype, device="cuda")
    start = 0
    for sequence, length in enumerate(lengths):
        seq_idx[0, start : start + length] = sequence
        start += length
    chunk_indices = []
    chunk_offsets = []
    for chunk in range(total // 128):
        values = seq_idx[0, chunk * 128 : (chunk + 1) * 128]
        previous = torch.cat((values[:1] - 1, values[:-1]))
        for offset in (values != previous).nonzero(as_tuple=True)[0].tolist():
            chunk_indices.append(chunk)
            chunk_offsets.append(offset)
    cumsum = [0]
    start = 0
    for length in lengths:
        end = start + length
        cumsum.append(cumsum[-1] + (end + 127) // 128 - start // 128)
        start = end
    return (
        seq_idx,
        torch.tensor(chunk_indices, dtype=torch.int32, device="cuda"),
        torch.tensor(chunk_offsets, dtype=torch.int32, device="cuda"),
        torch.tensor(cumsum, dtype=torch.int32, device="cuda"),
    )


def _case(
    *,
    nheads=8,
    ngroups=8,
    state_dtype=torch.bfloat16,
    varlen=False,
    seq_idx_dtype=torch.int32,
    preprocess_dtype=torch.float32,
    d_has_hdim=True,
    has_z=True,
    has_d=True,
    batch=2,
    seqlen=128,
    lengths=(96, 160),
    unbounded=False,
    seed=7,
):
    torch.manual_seed(seed)
    if varlen:
        batch = 1
        seqlen = sum(lengths)
    x = torch.randn(batch, seqlen, nheads, 64, device="cuda").to(torch.bfloat16)
    dt = torch.randn(batch, seqlen, nheads, device="cuda", dtype=preprocess_dtype)
    A = -torch.rand(nheads, device="cuda", dtype=torch.float32) - 1.0
    B = torch.randn(batch, seqlen, ngroups, 128, device="cuda").to(torch.bfloat16)
    C = torch.randn_like(B)
    d_shape = (nheads, 64) if d_has_hdim else (nheads,)
    D = torch.randn(*d_shape, device="cuda").to(torch.bfloat16) if has_d else None
    z = torch.randn_like(x) if has_z else None
    dt_bias = (torch.rand(nheads, device="cuda", dtype=torch.float32) - 4.0).to(
        preprocess_dtype
    )
    state_batch = len(lengths) if varlen else batch
    initial_states = torch.randn(state_batch, nheads, 64, 128, device="cuda").to(
        state_dtype
    )
    if varlen:
        seq_idx, chunk_indices, chunk_offsets, seq_chunk_cumsum = _varlen_metadata(
            lengths, seq_idx_dtype
        )
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
        has_d=has_d,
        d_has_hdim=d_has_hdim,
        has_initial_states=True,
        has_varlen=varlen,
        has_z=z is not None,
        seq_idx_dtype=seq_idx_dtype,
    )
    arguments = dict(
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")) if unbounded else (0.001, 0.1),
        initial_states=initial_states,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        seq_chunk_cumsum=seq_chunk_cumsum,
        return_final_states=True,
    )
    return constructor, (x, dt, A, B, C), arguments


def _assert_parity(actual, expected):
    # Numeric comparison only: backends return (output, final_states) in the
    # configured io/state dtypes, so normalize both sides to fp32.
    torch.testing.assert_close(actual[0].float(), expected[0].float(), atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(actual[1].float(), expected[1].float(), atol=ATOL, rtol=RTOL)


def _cute_reference(constructor, tensors, arguments):
    return SSDCombined(**constructor, backend="cute").run(*tensors, **arguments)


@pytest.mark.parametrize(
    "state_dtype,varlen,seq_idx_dtype,nheads,ngroups,preprocess_dtype,d_has_hdim",
    [
        (torch.bfloat16, False, torch.int32, 8, 8, torch.float32, True),
        (torch.float16, False, torch.int32, 8, 8, torch.float32, False),
        (torch.bfloat16, True, torch.int32, 8, 8, torch.float32, True),
        (torch.float16, True, torch.int64, 8, 8, torch.float32, False),
        (torch.bfloat16, False, torch.int32, 8, 8, torch.bfloat16, False),
        (torch.bfloat16, False, torch.int32, 1, 1, torch.float32, False),
        (torch.bfloat16, False, torch.int32, 12, 3, torch.float32, False),
        (torch.bfloat16, False, torch.int32, 16, 4, torch.float32, False),
        (torch.bfloat16, False, torch.int32, 128, 1, torch.float32, False),
        (torch.bfloat16, False, torch.int32, 128, 128, torch.float32, False),
        (torch.bfloat16, False, torch.int32, 128, 8, torch.float32, False),
        (torch.bfloat16, True, torch.int32, 128, 8, torch.float32, False),
    ],
)
def test_vibecuda_ssd_combined_route_matrix(
    state_dtype,
    varlen,
    seq_idx_dtype,
    nheads,
    ngroups,
    preprocess_dtype,
    d_has_hdim,
):
    zero_unbounded = varlen and nheads == 128 and ngroups == 8
    constructor, tensors, arguments = _case(
        nheads=nheads,
        ngroups=ngroups,
        state_dtype=state_dtype,
        varlen=varlen,
        seq_idx_dtype=seq_idx_dtype,
        preprocess_dtype=preprocess_dtype,
        d_has_hdim=d_has_hdim,
        has_z=True,
        unbounded=zero_unbounded,
    )
    if zero_unbounded:
        arguments["initial_states"].zero_()
    expected = _cute_reference(constructor, tensors, arguments)
    actual = SSDCombined(**constructor, backend="vibecuda").run(*tensors, **arguments)
    _assert_parity(actual, expected)


def test_vibecuda_ssd_combined_varlen_multi_chunk():
    constructor, tensors, arguments = _case(varlen=True, lengths=(128, 256, 384, 128))
    expected = _cute_reference(constructor, tensors, arguments)
    actual = SSDCombined(**constructor, backend="vibecuda").run(*tensors, **arguments)
    _assert_parity(actual, expected)


def test_vibecuda_ssd_combined_batched_multi_chunk():
    constructor, tensors, arguments = _case(batch=1, seqlen=512, varlen=False)
    expected = _cute_reference(constructor, tensors, arguments)
    actual = SSDCombined(**constructor, backend="vibecuda").run(*tensors, **arguments)
    _assert_parity(actual, expected)


def test_vibecuda_ssd_combined_full_write_into_caller_out():
    """Sentinel prefill: a pre-allocated caller out must be fully overwritten."""
    constructor, tensors, arguments = _case(varlen=True)
    reference_run = SSDCombined(**constructor, backend="vibecuda").run(
        *tensors, **arguments
    )
    out = torch.full((1, 8, 64, 2, 128), torch.nan, dtype=torch.bfloat16, device="cuda")
    actual = SSDCombined(**constructor, backend="vibecuda").run(
        *tensors, **{**arguments, "out": out}
    )
    assert actual[0].untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
    assert torch.isfinite(out.float()).all(), "out not fully written (NaN remains)"
    torch.testing.assert_close(
        actual[0].float(), reference_run[0].float(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual[1].float(), reference_run[1].float(), rtol=0, atol=0
    )


def test_vibecuda_ssd_combined_accepts_strided_input_views():
    def strided_last_dim(value):
        storage = torch.empty(
            (*value.shape[:-1], value.shape[-1] + 1),
            dtype=value.dtype,
            device=value.device,
        )
        view = storage[..., : value.shape[-1]]
        view.copy_(value)
        return view

    def sglang_projection_view(value):
        active_width = value.numel() // (value.shape[0] * value.shape[1])
        storage = torch.empty(
            (value.shape[0], value.shape[1], active_width + 8),
            dtype=value.dtype,
            device=value.device,
        )
        view = storage[..., :active_width].view(value.shape)
        view.copy_(value)
        return view

    constructor, tensors, arguments = _case(varlen=True)
    x, dt, A, B, C = tensors
    strided_tensors = (
        sglang_projection_view(x),
        sglang_projection_view(dt),
        A,
        sglang_projection_view(B),
        sglang_projection_view(C),
    )
    strided_arguments = {
        **arguments,
        "z": strided_last_dim(arguments["z"]),
        "initial_states": strided_last_dim(arguments["initial_states"]),
    }
    # Only the vibecuda backend accepts strided views; feed the oracle the
    # contiguous originals (identical values) so the candidate is still checked
    # against the same reference on the same numbers.
    reference = _cute_reference(constructor, tensors, arguments)
    actual = SSDCombined(**constructor, backend="vibecuda").run(
        *strided_tensors, **strided_arguments
    )
    _assert_parity(actual, reference)


def test_vibecuda_ssd_combined_rejects_unsupported_geometry():
    with pytest.raises(ValueError, match="chunk_size=128"):
        SSDCombined(
            chunk_size=64,
            nheads=8,
            headdim=64,
            dstate=128,
            ngroups=8,
            backend="vibecuda",
        )
