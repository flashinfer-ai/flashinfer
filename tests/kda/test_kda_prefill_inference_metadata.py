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

import threading
from types import SimpleNamespace

import pytest
import torch

from flashinfer.kda_prefill import (
    _cached_packed_task_metadata,
    _generated_affine_state_indices_i64,
)


@pytest.fixture
def workspace():
    return SimpleNamespace(
        _packed_metadata_lock=threading.Lock(),
        _packed_metadata_tensor=None,
        _packed_metadata_host_key=None,
        _packed_metadata_signature=None,
        _packed_metadata=None,
        _generated_affine_state_indices_tensor=None,
        _generated_affine_state_indices_version=None,
        _generated_affine_state_indices_destination_ptr=None,
    )


def packed_metadata(workspace, offsets):
    return _cached_packed_task_metadata(
        workspace,
        offsets,
        total_tokens=6,
        num_heads=96,
        sm_count=148,
        build_persistent_plan=True,
    )


@pytest.mark.parametrize("inference", [False, True])
def test_packed_metadata_mutation(workspace, monkeypatch, inference):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with torch.inference_mode(inference):
        offsets = torch.tensor([0, 2, 6], dtype=torch.int64)
        first = packed_metadata(workspace, offsets)
        second = packed_metadata(workspace, offsets)
        assert (first is second) is not inference
        assert first[0] == (1, 0)
        assert first[3:] == ((0, 2, 6), (2, 4))

        offsets[1] = 5
        updated = packed_metadata(workspace, offsets)
        assert updated[0] == (0, 1)
        assert updated[3:] == ((0, 5, 6), (5, 1))

        offsets[1] = 0
        with pytest.raises(ValueError, match="strictly increasing"):
            packed_metadata(workspace, offsets)


@pytest.mark.parametrize("inference", [False, True])
def test_packed_metadata_requires_capture_warmup(workspace, monkeypatch, inference):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with torch.inference_mode(inference):
        offsets = torch.tensor([0, 2, 6], dtype=torch.int64)
        with pytest.raises(RuntimeError, match="not warmed"):
            packed_metadata(workspace, offsets)


@pytest.mark.parametrize("inference", [False, True])
def test_affine_state_indices_mutation(workspace, inference):
    with torch.inference_mode(inference):
        source = torch.tensor([2], dtype=torch.int32)
        plan = SimpleNamespace(state_indices_i64=torch.empty(1, dtype=torch.int64))

        def carrier():
            return _generated_affine_state_indices_i64(
                workspace=workspace, plan=plan, state_indices=source, capturing=False
            )

        assert carrier().tolist() == [2]
        plan.state_indices_i64.fill_(7)
        assert carrier().tolist() == ([2] if inference else [7])
        source.fill_(3)
        assert carrier().tolist() == [3]


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph test")
def test_inference_metadata_cuda_graph(workspace, inference):
    with torch.inference_mode(inference):
        offsets = torch.tensor([0, 2, 6], dtype=torch.int64, device="cuda")
        source = torch.tensor([2], dtype=torch.int32, device="cuda")
        plan = SimpleNamespace(
            state_indices_i64=torch.empty(1, dtype=torch.int64, device="cuda")
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            warmed = packed_metadata(workspace, offsets)
            _generated_affine_state_indices_i64(
                workspace=workspace, plan=plan, state_indices=source, capturing=False
            )
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            assert packed_metadata(workspace, offsets) is warmed
            _generated_affine_state_indices_i64(
                workspace=workspace, plan=plan, state_indices=source, capturing=True
            )
        source.fill_(4)
        graph.replay()
        torch.cuda.synchronize()
        assert plan.state_indices_i64.tolist() == [4]


def test_host_offsets_replace_device_read_and_are_cached(workspace, monkeypatch):
    from flashinfer.kda_host_offsets import host_cu_seqlens_scope

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with torch.inference_mode():
        placeholder = torch.zeros(3, dtype=torch.int64)
        with host_cu_seqlens_scope(torch.tensor([0, 2, 6], dtype=torch.int32)):
            first = packed_metadata(workspace, placeholder)
            assert first[3:] == ((0, 2, 6), (2, 4))
            assert packed_metadata(workspace, placeholder) is first
        with host_cu_seqlens_scope([0, 5, 6]):
            updated = packed_metadata(workspace, placeholder)
            assert updated[3:] == ((0, 5, 6), (5, 1))
        with (
            host_cu_seqlens_scope((0, 2, 4, 6)),
            pytest.raises(ValueError, match="cu_seqlens_host has 4 entries"),
        ):
            packed_metadata(workspace, placeholder)


def test_cu_seqlens_to_list_uses_host_hint_only_when_active():
    from flashinfer.kda_host_offsets import cu_seqlens_to_list, host_cu_seqlens_scope

    device_side = torch.tensor([0, 3, 9], dtype=torch.int64)
    assert cu_seqlens_to_list(device_side) == [0, 3, 9]
    with host_cu_seqlens_scope([0, 4, 9]):
        assert cu_seqlens_to_list(device_side) == [0, 4, 9]
    with (
        pytest.raises(ValueError, match="1-D CPU tensor"),
        host_cu_seqlens_scope(torch.zeros(3).reshape(1, 3)),
    ):
        cu_seqlens_to_list(device_side)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "backend",
    ["auto", "small-bh", "cute-dsl", "cake", "cute-dsl-persistent", "ptx", "tirx"],
)
@pytest.mark.parametrize("lengths", [(512,), (200, 1000, 64)])
def test_recurrent_kda_cu_seqlens_host_avoids_sync(backend, lengths):
    import math

    from flashinfer.kda import recurrent_kda
    from flashinfer.utils import get_compute_capability

    capability = get_compute_capability(torch.device("cuda"))
    if capability[0] != 10:
        pytest.skip("KDA chunk prefill routes need SM100/SM103")
    if backend == "ptx" and capability != (10, 3):
        pytest.skip("backend='ptx' requires SM103")
    if backend != "auto" and capability not in ((10, 0), (10, 3)):
        pytest.skip(f"backend={backend!r} supports SM100/SM103 only")
    heads, state_dtype = {
        "ptx": (64, torch.float32),
        "cute-dsl-persistent": (8, torch.float32),
        "tirx": (8, torch.float32),
    }.get(backend, (4, torch.bfloat16))
    dim, total = 128, sum(lengths)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    gen = torch.Generator(device="cuda").manual_seed(0)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(
            *shape, device="cuda", dtype=torch.float32, generator=gen
        ).to(dtype)

    q, k, v = (rand(1, total, heads, dim) for _ in range(3))
    g, beta = rand(1, total, heads, dim), rand(1, total, heads)
    a_log = rand(heads, dtype=torch.float32).abs().log()
    dt_bias = rand(heads * dim, dtype=torch.float32)
    state = (rand(len(lengths), heads, dim, dim, dtype=state_dtype) * 0.1).contiguous()
    with torch.inference_mode():
        cu = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    host = torch.tensor(offsets, dtype=torch.int32)

    def run(**extra):
        return recurrent_kda(
            q,
            k,
            v,
            g,
            beta,
            A_log=a_log,
            dt_bias=dt_bias,
            scale=1 / math.sqrt(dim),
            initial_state=state.clone(),
            output_final_state=True,
            lower_bound=-5.0,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            beta_is_logit=True,
            cu_seqlens=cu,
            backend=backend,
            **extra,
        )

    try:
        with torch.inference_mode():
            ref_out, ref_state = run()
            run(cu_seqlens_host=host)
            torch.cuda.synchronize()
            # ptx still syncs once per call to range-check initial_state.
            torch.cuda.set_sync_debug_mode("default" if backend == "ptx" else "error")
            try:
                out, final_state = run(cu_seqlens_host=host)
            finally:
                torch.cuda.set_sync_debug_mode("default")
            torch.cuda.synchronize()
    except (ImportError, RuntimeError) as exc:
        if backend in ("ptx", "tirx", "cute-dsl-persistent") and (
            isinstance(exc, ImportError)
            or "requires" in str(exc)
            or "ptxas" in str(exc)
        ):
            pytest.skip(f"{backend} unavailable: {exc}")
        raise
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(final_state, ref_state, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    ("dtype", "use_hint"),
    [(torch.int32, False), (torch.int32, True), (torch.int64, True)],
)
def test_cute_dsl_inference_cu_seqlens_refilled_in_place(dtype, use_hint):
    import math

    from flashinfer.kda import recurrent_kda
    from flashinfer.utils import get_compute_capability

    if get_compute_capability(torch.device("cuda"))[0] != 10:
        pytest.skip("KDA chunk prefill routes need SM100/SM103")
    heads, dim, total = 4, 128, 1024
    gen = torch.Generator(device="cuda").manual_seed(0)

    def rand(*shape, dt=torch.bfloat16):
        return torch.randn(
            *shape, device="cuda", dtype=torch.float32, generator=gen
        ).to(dt)

    q, k, v, g = (rand(1, total, heads, dim) for _ in range(4))
    beta = rand(1, total, heads)
    a_log = rand(heads, dt=torch.float32).abs().log()
    dt_bias = rand(heads * dim, dt=torch.float32)
    state = (rand(2, heads, dim, dim) * 0.1).contiguous()

    def run(cu, offsets):
        host = {"cu_seqlens_host": offsets} if use_hint else {}
        return recurrent_kda(
            q, k, v, g, beta, A_log=a_log, dt_bias=dt_bias,
            scale=1 / math.sqrt(dim), initial_state=state.clone(),
            output_final_state=True, lower_bound=-5.0, use_gate_in_kernel=True,
            beta_is_logit=True, cu_seqlens=cu, backend="cute-dsl", **host,
        )  # fmt: skip

    first, second = [0, 512, 1024], [0, 200, 1024]
    with torch.inference_mode():
        buf = torch.tensor(first, dtype=dtype, device="cuda")
        run(buf, first)
        buf.copy_(torch.tensor(second, dtype=dtype))
        out, final_state = run(buf, second)
        ref_out, ref_state = run(
            torch.tensor(second, dtype=dtype, device="cuda"), second
        )
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(final_state, ref_state, rtol=0, atol=0)
