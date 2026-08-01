# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import math

import pytest
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.kda import recurrent_kda
from flashinfer.kda_prefill import RecurrentKDAPrefillWorkspace
from flashinfer.utils import get_compute_capability

kda_decode_api = importlib.import_module("flashinfer.kda_decode")
kda_api = importlib.import_module("flashinfer.kda")
kda_prefill_api = importlib.import_module("flashinfer.kda_prefill")


def test_public_api_uses_phase_neutral_facade_and_prefill_workspace():
    assert flashinfer.recurrent_kda is kda_api.recurrent_kda
    assert (
        flashinfer.RecurrentKDAPrefillWorkspace
        is kda_prefill_api.RecurrentKDAPrefillWorkspace
    )


def _strict_prefill_kwargs(inputs):
    return {
        **inputs,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "lower_bound": -5.0,
        "beta_is_logit": True,
    }


def _make_inputs(
    *,
    seq_lens,
    num_heads: int,
    packed: bool,
    initial_state: bool = False,
    seed: int = 0,
):
    torch.manual_seed(seed)
    if packed:
        batch_size = 1
        seq_len = sum(seq_lens)
    else:
        if len(set(seq_lens)) != 1:
            raise ValueError("fixed test inputs require equal sequence lengths")
        batch_size = len(seq_lens)
        seq_len = seq_lens[0]
    shape = (batch_size, seq_len, num_heads, 128)
    q = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    g = (0.1 * torch.randn(shape, dtype=torch.float32, device="cuda")).to(
        torch.bfloat16
    )
    beta = torch.randn(
        (batch_size, seq_len, num_heads),
        dtype=torch.bfloat16,
        device="cuda",
    )
    A_log = 0.1 * torch.randn(num_heads, dtype=torch.float32, device="cuda")
    dt_bias = 0.1 * torch.randn((num_heads, 128), dtype=torch.float32, device="cuda")
    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)
    state = None
    if initial_state:
        state = (
            0.1
            * torch.randn(
                (len(seq_lens), num_heads, 128, 128),
                dtype=torch.float32,
                device="cuda",
            )
        ).to(torch.bfloat16)
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "initial_state": state,
        "cu_seqlens": (
            torch.tensor(offsets, dtype=torch.int64, device="cuda") if packed else None
        ),
    }


def _reference(inputs, *, lower_bound=-5.0, scale=None):
    q = inputs["q"]
    batch_size, seq_len, num_heads, head_dim = q.shape
    scale = head_dim**-0.5 if scale is None else scale
    q_flat = F.normalize(q.float(), dim=-1).reshape(-1, num_heads, head_dim)
    k_flat = F.normalize(inputs["k"].float(), dim=-1).reshape(-1, num_heads, head_dim)
    v_flat = inputs["v"].float().reshape(-1, num_heads, head_dim)
    g_flat = inputs["g"].float().reshape(-1, num_heads, head_dim)
    beta_flat = torch.sigmoid(inputs["beta"].float().reshape(-1, num_heads))
    gate = lower_bound * torch.sigmoid(
        torch.exp(inputs["A_log"]).reshape(1, num_heads, 1)
        * (g_flat + inputs["dt_bias"].reshape(1, num_heads, head_dim))
    )
    decay = torch.exp(gate)
    if inputs["cu_seqlens"] is None:
        offsets = [index * seq_len for index in range(batch_size + 1)]
    else:
        offsets = [int(value) for value in inputs["cu_seqlens"].tolist()]
    if inputs["initial_state"] is None:
        state = torch.zeros(
            (len(offsets) - 1, num_heads, head_dim, head_dim),
            dtype=torch.bfloat16,
            device=q.device,
        )
    else:
        state = inputs["initial_state"].clone()
    out = torch.empty_like(q_flat)
    for sequence in range(len(offsets) - 1):
        for token in range(offsets[sequence], offsets[sequence + 1]):
            state_f32 = state[sequence].float()
            decayed = state_f32 * decay[token].unsqueeze(1)
            predicted = torch.einsum("hk,hvk->hv", k_flat[token], decayed)
            residual = beta_flat[token].unsqueeze(-1) * (v_flat[token] - predicted)
            updated = decayed + residual.unsqueeze(-1) * k_flat[token].unsqueeze(1)
            state[sequence] = updated.to(torch.bfloat16)
            projected = torch.einsum(
                "hk,hvk->hv", q_flat[token], state[sequence].float()
            )
            out[token] = (scale * projected).to(torch.bfloat16)
    return out.reshape_as(q), state


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda")


@pytest.fixture
def b200(cuda_device):
    if get_compute_capability(cuda_device) != (10, 0):
        pytest.skip("frozen recurrent KDA prefill requires B200 (cc 10.0)")
    return cuda_device


class _RecorderModule:
    def __init__(self, *, final_value=None):
        self.calls = []
        self.final_value = final_value

    def run(self, *args):
        self.calls.append(args)
        if self.final_value is not None and bool(args[17]):
            args[12].fill_(self.final_value)


def test_decode_and_spec_stay_on_existing_backend(monkeypatch):
    sentinel = (object(), object())
    calls = []

    def old_backend(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(kda_decode_api, "_run_recurrent_kda", old_backend)
    monkeypatch.setattr(
        kda_decode_api,
        "recurrent_kda",
        lambda *args, **kwargs: pytest.fail("facade nested the decorated decode API"),
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant: pytest.fail(f"unexpected frozen route {variant}"),
    )
    q = torch.empty((2, 1, 4, 128), dtype=torch.bfloat16)
    result = recurrent_kda(q, q, q, q, torch.empty((2, 1, 4)))
    assert result is sentinel
    result = recurrent_kda(
        q.expand(2, 2, 4, 128),
        q.expand(2, 2, 4, 128),
        q.expand(2, 2, 4, 128),
        q.expand(2, 2, 4, 128),
        torch.empty((2, 2, 4)),
        num_spec_tokens=1,
    )
    assert result is sentinel
    assert len(calls) == 2


def test_multi_token_gqa_stays_on_existing_backend(cuda_device, monkeypatch):
    sentinel = (object(), object())
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(kda_decode_api, "_run_recurrent_kda", lambda **kwargs: sentinel)
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant: pytest.fail(f"unexpected frozen route {variant}"),
    )
    q = torch.randn((1, 2, 2, 128), dtype=torch.bfloat16, device=cuda_device)
    v = torch.randn((1, 2, 4, 128), dtype=torch.bfloat16, device=cuda_device)
    result = recurrent_kda(
        q,
        q.clone(),
        v,
        v.clone(),
        torch.randn((1, 2, 4), dtype=torch.bfloat16, device=cuda_device),
        A_log=torch.randn(2, device=cuda_device),
        dt_bias=torch.randn((2, 128), device=cuda_device),
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        beta_is_logit=True,
    )
    assert result is sentinel


@pytest.mark.parametrize(
    ("packed", "num_heads", "expected_variant"),
    [(False, 64, "m64"), (True, 64, "m128"), (True, 2, "m128")],
)
def test_frozen_route_and_ffi_abi(
    cuda_device,
    monkeypatch,
    packed,
    num_heads,
    expected_variant,
):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(kda_prefill_api, "_flash_kda_stream_workspaces", {})
    modules = {}

    def get_module(variant):
        modules.setdefault(variant, _RecorderModule())
        return modules[variant]

    monkeypatch.setattr(kda_prefill_api, "_get_flash_kda_prefill_module", get_module)
    inputs = _make_inputs(
        seq_lens=[1, 2] if packed else [2],
        num_heads=num_heads,
        packed=packed,
    )
    if packed and num_heads == 2:
        inputs["cu_seqlens"] = inputs["cu_seqlens"].to(torch.int32)
    output = torch.zeros_like(inputs["q"])
    seq_order = (
        torch.tensor([1, 0], dtype=torch.int32, device="cuda") if packed else None
    )
    actual, state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        seq_order=seq_order,
    )
    assert actual.data_ptr() == output.data_ptr()
    assert state is None
    assert set(modules) == {expected_variant}
    (args,) = modules[expected_variant].calls
    assert len(args) == 21
    assert args[0].data_ptr() == inputs["q"].data_ptr()
    assert args[4].data_ptr() == inputs["beta"].data_ptr()
    assert args[5].shape == (
        max(inputs["q"].numel() // (num_heads * 128), 32),
        max(num_heads, 8),
    )
    assert args[8].dtype == torch.int64
    assert args[9].dtype == torch.int32
    assert args[10].data_ptr() == args[12].data_ptr()
    assert args[13].dtype == torch.uint8
    assert args[13].shape == (768,)
    assert args[14] == 1
    assert args[15] == num_heads
    assert args[16] == 0
    assert args[17] == 0
    assert math.isclose(args[18], 128**-0.5)
    assert args[19] == -5.0
    assert args[20] == int(torch.cuda.current_stream(cuda_device).cuda_stream)
    if args[5].data_ptr() != inputs["beta"].data_ptr():
        total_tokens = inputs["q"].numel() // (num_heads * 128)
        torch.testing.assert_close(
            args[5][:total_tokens, :num_heads],
            inputs["beta"].reshape(-1, num_heads),
        )


def test_frozen_route_passes_nondefault_stream(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    module = _RecorderModule()
    monkeypatch.setattr(
        kda_prefill_api, "_get_flash_kda_prefill_module", lambda variant: module
    )
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False)
    stream = torch.cuda.Stream(device=cuda_device)
    stream.wait_stream(torch.cuda.current_stream(cuda_device))
    with torch.cuda.stream(stream):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=torch.empty_like(inputs["q"]),
        )
    (args,) = module.calls
    assert args[20] == int(stream.cuda_stream)


def test_frozen_route_rejects_output_overlap(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    module = _RecorderModule()
    monkeypatch.setattr(
        kda_prefill_api, "_get_flash_kda_prefill_module", lambda variant: module
    )
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False)
    with pytest.raises(ValueError, match="output must not overlap q"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=inputs["q"].view_as(inputs["q"]),
        )
    assert module.calls == []


def test_initial_state_is_updated_in_place(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    module = _RecorderModule(final_value=0.25)
    monkeypatch.setattr(
        kda_prefill_api, "_get_flash_kda_prefill_module", lambda variant: module
    )
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False, initial_state=True)
    original_state = inputs["initial_state"]
    actual, returned_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
    )
    assert actual.shape == inputs["q"].shape
    assert returned_state is original_state
    (args,) = module.calls
    assert args[10].data_ptr() == original_state.data_ptr()
    assert args[12].data_ptr() == original_state.data_ptr()
    assert args[16] == 1
    assert args[17] == 1
    torch.testing.assert_close(
        original_state,
        torch.full_like(original_state, 0.25),
    )


def test_stream_workspace_does_not_allocate_state_scratch_for_inplace_update(
    cuda_device, monkeypatch
):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(kda_prefill_api, "_flash_kda_stream_workspaces", {})
    module = _RecorderModule(final_value=0.0)
    monkeypatch.setattr(
        kda_prefill_api, "_get_flash_kda_prefill_module", lambda variant: module
    )
    cases = [
        _make_inputs(
            seq_lens=[2],
            num_heads=2,
            packed=False,
            initial_state=True,
        ),
        _make_inputs(
            seq_lens=[1, 1, 2],
            num_heads=2,
            packed=True,
            initial_state=True,
        ),
        _make_inputs(
            seq_lens=[2, 2],
            num_heads=2,
            packed=False,
            initial_state=True,
        ),
    ]
    for inputs in cases:
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=torch.empty_like(inputs["q"]),
        )

    assert len(kda_prefill_api._flash_kda_stream_workspaces) == 1
    (workspace,) = kda_prefill_api._flash_kda_stream_workspaces.values()
    assert workspace._state_scratch is None
    assert workspace._beta_padding.numel() == 32 * 8


@pytest.mark.parametrize(
    ("dtype", "size_delta"),
    [(torch.int64, 0), (torch.int32, 1)],
)
def test_packed_seq_order_validation(cuda_device, monkeypatch, dtype, size_delta):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant: _RecorderModule(),
    )
    inputs = _make_inputs(seq_lens=[1, 2], num_heads=2, packed=True)
    seq_order = torch.arange(2 + size_delta, dtype=dtype, device="cuda")
    with pytest.raises(ValueError, match="seq_order"):
        recurrent_kda(**_strict_prefill_kwargs(inputs), seq_order=seq_order)


def test_fixed_prefill_rejects_seq_order(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False)
    with pytest.raises(ValueError, match="only supported for packed"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            seq_order=torch.zeros(1, dtype=torch.int32, device=cuda_device),
        )


def test_graph_capture_requires_packed_int64_offsets(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    inputs = _make_inputs(seq_lens=[1, 2], num_heads=2, packed=True)
    inputs["cu_seqlens"] = inputs["cu_seqlens"].to(torch.int32)
    workspace = RecurrentKDAPrefillWorkspace(cuda_device)
    with pytest.raises(RuntimeError, match="requires int64 cu_seqlens"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            prefill_workspace=workspace,
        )


def test_graph_capture_requires_explicit_workspace(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False)
    with pytest.raises(
        RuntimeError, match="requires an explicit RecurrentKDAPrefillWorkspace"
    ):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=torch.empty_like(inputs["q"]),
        )


def test_explicit_workspace_descriptor_prepare_and_reuse(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    module = _RecorderModule()
    monkeypatch.setattr(
        kda_prefill_api, "_get_flash_kda_prefill_module", lambda variant: module
    )
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False)
    output = torch.empty_like(inputs["q"])
    workspace = RecurrentKDAPrefillWorkspace(cuda_device)

    for _ in range(2):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=output,
            prefill_workspace=workspace,
        )
    assert [args[14] for args in module.calls] == [1, 0]
    assert (
        module.calls[0][13].data_ptr()
        == module.calls[1][13].data_ptr()
        == workspace._descriptor_storages["m128"].data_ptr()
    )

    changed_output = torch.empty_like(output)
    recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=changed_output,
        prefill_workspace=workspace,
    )
    assert module.calls[-1][14] == 1


def test_captured_workspace_rejects_eager_reuse_and_capture_mismatch(
    cuda_device, monkeypatch
):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    module = _RecorderModule()
    monkeypatch.setattr(
        kda_prefill_api, "_get_flash_kda_prefill_module", lambda variant: module
    )
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False)
    output = torch.empty_like(inputs["q"])
    workspace = RecurrentKDAPrefillWorkspace(cuda_device)
    recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        prefill_workspace=workspace,
    )

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="not warmed for the exact"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=torch.empty_like(output),
            prefill_workspace=workspace,
        )

    recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        prefill_workspace=workspace,
    )
    assert module.calls[-1][14] == 0
    assert workspace._captured

    with pytest.raises(RuntimeError, match="captured by another CUDA graph"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=output,
            prefill_workspace=workspace,
        )

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with pytest.raises(RuntimeError, match="cannot be reused eagerly"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=output,
            prefill_workspace=workspace,
        )


def test_workspace_rejects_a_different_stream(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    module = _RecorderModule()
    monkeypatch.setattr(
        kda_prefill_api, "_get_flash_kda_prefill_module", lambda variant: module
    )
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False)
    output = torch.empty_like(inputs["q"])
    workspace = RecurrentKDAPrefillWorkspace(cuda_device)
    recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        prefill_workspace=workspace,
    )

    other_stream = torch.cuda.Stream(device=cuda_device)
    with (
        torch.cuda.stream(other_stream),
        pytest.raises(RuntimeError, match="different CUDA stream"),
    ):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=output,
            prefill_workspace=workspace,
        )


def test_flash_kda_jit_getter_is_importable():
    import flashinfer
    from flashinfer.jit.flash_kda import get_flash_kda_prefill_module

    assert callable(get_flash_kda_prefill_module)
    assert flashinfer.RecurrentKDAPrefillWorkspace is RecurrentKDAPrefillWorkspace


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("non_default_stream", [False, True])
def test_frozen_prefill_matches_reference(b200, packed, non_default_stream):
    inputs = _make_inputs(
        seq_lens=[3, 5] if packed else [4, 4],
        num_heads=2,
        packed=packed,
        initial_state=True,
        seed=2026,
    )
    reference_inputs = {
        **inputs,
        "initial_state": inputs["initial_state"].clone(),
    }
    expected_output, expected_state = _reference(reference_inputs)
    output = torch.empty_like(inputs["q"])
    state_identity = inputs["initial_state"]
    seq_order = torch.tensor([1, 0], dtype=torch.int32, device=b200) if packed else None

    if non_default_stream:
        stream = torch.cuda.Stream(device=b200)
        stream.wait_stream(torch.cuda.current_stream(b200))
        with torch.cuda.stream(stream):
            actual_output, actual_state = recurrent_kda(
                **_strict_prefill_kwargs(inputs),
                output=output,
                output_final_state=True,
                seq_order=seq_order,
            )
        stream.synchronize()
    else:
        actual_output, actual_state = recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=output,
            output_final_state=True,
            seq_order=seq_order,
        )

    assert actual_output.data_ptr() == output.data_ptr()
    assert actual_state is state_identity
    torch.testing.assert_close(
        actual_output.float(),
        expected_output.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual_state.float(),
        expected_state.float(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_frozen_prefill_without_initial_or_final_state(b200):
    inputs = _make_inputs(seq_lens=[3], num_heads=2, packed=False, initial_state=False)
    expected_output, _ = _reference(inputs)
    output = torch.empty_like(inputs["q"])
    actual_output, actual_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        output_final_state=False,
    )
    assert actual_output.data_ptr() == output.data_ptr()
    assert actual_state is None
    torch.testing.assert_close(
        actual_output.float(),
        expected_output.float(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_frozen_prefill_m64_matches_reference(b200):
    inputs = _make_inputs(
        seq_lens=[2],
        num_heads=64,
        packed=False,
        initial_state=True,
        seed=2027,
    )
    reference_inputs = {
        **inputs,
        "initial_state": inputs["initial_state"].clone(),
    }
    expected_output, expected_state = _reference(reference_inputs)
    output = torch.empty_like(inputs["q"])
    state_identity = inputs["initial_state"]

    actual_output, actual_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        output_final_state=True,
    )

    assert actual_output.data_ptr() == output.data_ptr()
    assert actual_state is state_identity
    torch.testing.assert_close(
        actual_output.float(),
        expected_output.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual_state.float(),
        expected_state.float(),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize(
    ("packed", "num_heads", "has_initial_state"),
    [(False, 64, True), (True, 2, False)],
)
def test_frozen_prefill_cuda_graph_capture_and_replay(
    b200,
    packed,
    num_heads,
    has_initial_state,
):
    inputs = _make_inputs(
        seq_lens=[1, 2] if packed else [2],
        num_heads=num_heads,
        packed=packed,
        initial_state=has_initial_state,
        seed=2028,
    )
    initial_state_seed = (
        inputs["initial_state"].clone() if inputs["initial_state"] is not None else None
    )
    expected_output, expected_state = _reference(
        {
            **inputs,
            "initial_state": (
                initial_state_seed.clone() if initial_state_seed is not None else None
            ),
        }
    )
    output = torch.empty_like(inputs["q"])
    seq_order = torch.tensor([1, 0], dtype=torch.int32, device=b200) if packed else None
    workspace = RecurrentKDAPrefillWorkspace(b200)
    capture_stream = torch.cuda.Stream(device=b200)
    capture_stream.wait_stream(torch.cuda.current_stream(b200))

    call_kwargs = {
        **_strict_prefill_kwargs(inputs),
        "output": output,
        "output_final_state": True,
        "seq_order": seq_order,
        "prefill_workspace": workspace,
    }
    with torch.cuda.stream(capture_stream):
        recurrent_kda(**call_kwargs)
        if initial_state_seed is not None:
            inputs["initial_state"].copy_(initial_state_seed)
        output.zero_()
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output, captured_state = recurrent_kda(**call_kwargs)

    with torch.cuda.stream(capture_stream):
        if initial_state_seed is not None:
            inputs["initial_state"].copy_(initial_state_seed)
        output.fill_(float("nan"))
    capture_stream.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    assert captured_output.data_ptr() == output.data_ptr()
    if inputs["initial_state"] is None:
        assert captured_state.data_ptr() == workspace._state_scratch.data_ptr()
    else:
        assert captured_state is inputs["initial_state"]
    assert workspace._captured
    torch.testing.assert_close(
        captured_output.float(),
        expected_output.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        captured_state.float(),
        expected_state.float(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_frozen_prefill_cuda_graph_workspaces_are_isolated(b200):
    capture_stream = torch.cuda.Stream(device=b200)
    launch_stream = torch.cuda.Stream(device=b200)
    bundles = []

    for seed in (2030, 2031):
        inputs = _make_inputs(
            seq_lens=[2],
            num_heads=2,
            packed=False,
            initial_state=True,
            seed=seed,
        )
        state_seed = inputs["initial_state"].clone()
        expected_output, expected_state = _reference(
            {
                **inputs,
                "initial_state": state_seed.clone(),
            }
        )
        output = torch.empty_like(inputs["q"])
        workspace = RecurrentKDAPrefillWorkspace(b200)
        call_kwargs = {
            **_strict_prefill_kwargs(inputs),
            "output": output,
            "output_final_state": True,
            "prefill_workspace": workspace,
        }
        capture_stream.wait_stream(torch.cuda.current_stream(b200))
        with torch.cuda.stream(capture_stream):
            recurrent_kda(**call_kwargs)
            inputs["initial_state"].copy_(state_seed)
            output.zero_()
        capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            recurrent_kda(**call_kwargs)
        bundles.append(
            (
                graph,
                workspace,
                inputs,
                state_seed,
                output,
                expected_output,
                expected_state,
            )
        )

    assert bundles[0][1]._state_scratch is None
    assert bundles[1][1]._state_scratch is None
    assert (
        bundles[0][1]._descriptor_storages["m128"].data_ptr()
        != bundles[1][1]._descriptor_storages["m128"].data_ptr()
    )

    for bundle_index in (0, 1, 0, 1):
        (
            graph,
            _workspace,
            inputs,
            state_seed,
            output,
            expected_output,
            expected_state,
        ) = bundles[bundle_index]
        with torch.cuda.stream(launch_stream):
            inputs["initial_state"].copy_(state_seed)
            output.fill_(float("nan"))
        launch_stream.synchronize()
        with torch.cuda.stream(launch_stream):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output.float(),
            expected_output.float(),
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            inputs["initial_state"].float(),
            expected_state.float(),
            atol=1e-2,
            rtol=1e-2,
        )
