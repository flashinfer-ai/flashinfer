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

import importlib.util
from pathlib import Path

import pytest
import torch

from flashinfer.jit import cake_gdn_noncp_decode as cake_gdn


_HEAD_SIZE = 128
_SCALE = _HEAD_SIZE**-0.5


def _arch() -> cake_gdn.CakeGDNArch:
    if not torch.cuda.is_available():
        pytest.skip("Cake GDN requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    try:
        return cake_gdn.arch_for_compute_capability(major, minor)
    except cake_gdn.CakeGDNUnsupportedError as error:
        pytest.skip(str(error))


def _make_inputs(batch_size: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(2026)
    device = torch.device("cuda")
    num_q_heads, num_v_heads = 16, 32
    tensors = {
        "q": torch.randn(
            batch_size,
            1,
            num_q_heads,
            _HEAD_SIZE,
            device=device,
            dtype=torch.bfloat16,
        ),
        "k": torch.randn(
            batch_size,
            1,
            num_q_heads,
            _HEAD_SIZE,
            device=device,
            dtype=torch.bfloat16,
        ),
        "v": torch.randn(
            batch_size,
            1,
            num_v_heads,
            _HEAD_SIZE,
            device=device,
            dtype=torch.bfloat16,
        ),
        "state": (
            torch.randn(
                batch_size,
                num_v_heads,
                _HEAD_SIZE,
                _HEAD_SIZE,
                device=device,
                dtype=torch.float32,
            )
            * 0.01
        ).contiguous(),
        "A_log": (
            torch.randn(num_v_heads, device=device, dtype=torch.float32) * 0.1
        ).contiguous(),
        "a": (
            torch.randn(batch_size, 1, num_v_heads, device=device, dtype=torch.float32)
            * 0.1
        )
        .to(torch.bfloat16)
        .contiguous(),
        "dt_bias": (
            torch.randn(num_v_heads, device=device, dtype=torch.float32) * 0.1
        ).contiguous(),
        "b": torch.randn(
            batch_size, 1, num_v_heads, device=device, dtype=torch.bfloat16
        ).contiguous(),
        "out": torch.zeros(
            batch_size,
            1,
            num_v_heads,
            _HEAD_SIZE,
            device=device,
            dtype=torch.bfloat16,
        ),
    }
    return tensors


def _reference(
    tensors: dict[str, torch.Tensor], initial_state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q = tensors["q"].squeeze(1).float().repeat_interleave(2, dim=1)
    k = tensors["k"].squeeze(1).float().repeat_interleave(2, dim=1)
    v = tensors["v"].squeeze(1).float()
    q = torch.nn.functional.normalize(q, p=2.0, dim=-1) * _SCALE
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
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
    return out.unsqueeze(1).to(torch.bfloat16).contiguous(), state.contiguous()


def _load(batch_size: int):
    route = cake_gdn.select_cake_gdn_decode_variant(
        arch=_arch(),
        batch_size=batch_size,
        io_dtype="bfloat16",
        state_dtype="float32",
        head_size=_HEAD_SIZE,
        layout="nontranspose",
        num_k_heads=16,
        num_q_heads=16,
        num_v_heads=32,
        scale=_SCALE,
        seq_len=1,
        use_qk_l2norm=True,
    )
    return route, cake_gdn.load_cake_gdn_kernel(route.variant_name, _arch())


def _make_prefill_inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(2027)
    device = torch.device("cuda")
    num_seqs, seq_len = 4, 64
    num_q_heads, num_v_heads = 4, 8
    total_tokens = num_seqs * seq_len
    tensors = {
        "q": torch.randn(
            total_tokens,
            num_q_heads,
            _HEAD_SIZE,
            device=device,
            dtype=torch.bfloat16,
        ),
        "k": torch.nn.functional.normalize(
            torch.randn(
                total_tokens,
                num_q_heads,
                _HEAD_SIZE,
                device=device,
                dtype=torch.float32,
            ),
            p=2.0,
            dim=-1,
        ).to(torch.bfloat16),
        "v": torch.randn(
            total_tokens,
            num_v_heads,
            _HEAD_SIZE,
            device=device,
            dtype=torch.bfloat16,
        ),
        "gate": torch.rand(
            total_tokens, num_v_heads, device=device, dtype=torch.float32
        ),
        "beta": torch.rand(
            total_tokens, num_v_heads, device=device, dtype=torch.float32
        ),
        "cu_seqlens": torch.arange(
            0,
            total_tokens + 1,
            seq_len,
            device=device,
            dtype=torch.int32,
        ),
        "out": torch.zeros(
            total_tokens,
            num_v_heads,
            _HEAD_SIZE,
            device=device,
            dtype=torch.bfloat16,
        ),
    }
    tensors["empty_i32"] = torch.empty(1, dtype=torch.int32, device=device)
    tensors["empty_state"] = torch.empty(1, dtype=torch.float32, device=device)
    tensors["tensormap_workspace"] = torch.empty(
        4 * 8 * 2 * 4 * 128, dtype=torch.uint8, device=device
    )
    return tensors


def _prefill_reference(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    reference_path = Path(__file__).with_name("reference_delta_rule.py")
    spec = importlib.util.spec_from_file_location(
        "_flashinfer_gdn_reference_delta_rule", reference_path
    )
    assert spec is not None and spec.loader is not None
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    output, _ = reference.blockwise_delta_rule(
        tensors["q"].float(),
        tensors["k"].float(),
        tensors["v"].float(),
        [64, 64, 64, 64],
        block_size=64,
        scale_factor=_SCALE,
        alpha=tensors["gate"],
        beta=tensors["beta"],
        state_dtype=torch.float32,
    )
    return output.to(torch.bfloat16)


def _load_prefill():
    route = cake_gdn.select_cake_gdn_prefill_variant(
        arch=_arch(),
        io_dtype="bfloat16",
        state_dtype="float32",
        num_seqs=4,
        total_seq_len=256,
        max_seq_len=64,
        num_q_heads=4,
        num_k_heads=4,
        num_v_heads=8,
        use_initial_state=False,
        store_final_state=False,
        checkpoint_every_n_tokens=0,
        use_state_indices=False,
    )
    return route, cake_gdn.load_cake_gdn_kernel(route.variant_name, _arch())


def _launch_prefill(entry, tensors: dict[str, torch.Tensor]) -> None:
    total_tiles = 4 * 8 * 2
    grid_x = total_tiles
    entry(
        tensors["q"],
        tensors["k"],
        tensors["v"],
        tensors["out"],
        tensors["gate"],
        tensors["beta"],
        tensors["cu_seqlens"],
        tensors["empty_i32"],
        tensors["empty_state"],
        tensors["empty_state"],
        tensors["empty_state"],
        tensors["empty_i32"],
        tensors["tensormap_workspace"],
        8 * _HEAD_SIZE * _HEAD_SIZE,
        8 * _HEAD_SIZE * _HEAD_SIZE,
        0,
        _SCALE,
        4,
        4,
        8,
        total_tiles,
        grid_x,
        1,
        1,
    )


def _launch(entry, tensors: dict[str, torch.Tensor], batch_size: int) -> None:
    blocks_per_state = 8 if batch_size < 32 else 1
    entry(
        tensors["q"],
        tensors["k"],
        tensors["v"],
        tensors["state"],
        tensors["A_log"],
        tensors["a"],
        tensors["dt_bias"],
        tensors["b"],
        tensors["out"],
        batch_size * 32 * blocks_per_state,
        1,
        1,
    )


def _make_bf16_serving_inputs(
    *,
    batch_size: int,
    seq_len: int,
    num_v_heads: int,
    strided_inputs: bool,
    cache_steps: int,
    pack_gates: bool = True,
) -> dict[str, object]:
    torch.manual_seed(2030 + seq_len + num_v_heads)
    device = torch.device("cuda")
    num_q_heads = 16
    if strided_inputs:
        packed_qkv = torch.empty(
            batch_size,
            seq_len,
            2 * num_q_heads * _HEAD_SIZE + num_v_heads * _HEAD_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )
        q_flat, k_flat, v_flat = packed_qkv.split(
            (
                num_q_heads * _HEAD_SIZE,
                num_q_heads * _HEAD_SIZE,
                num_v_heads * _HEAD_SIZE,
            ),
            dim=-1,
        )
        q = q_flat.view(batch_size, seq_len, num_q_heads, _HEAD_SIZE)
        k = k_flat.view(batch_size, seq_len, num_q_heads, _HEAD_SIZE)
        v = v_flat.view(batch_size, seq_len, num_v_heads, _HEAD_SIZE)
        q.normal_()
        k.normal_()
        v.normal_()
        if pack_gates:
            packed_gates = torch.empty(
                batch_size,
                seq_len,
                2 * num_v_heads + 7,
                device=device,
                dtype=torch.bfloat16,
            )
            a = packed_gates[..., :num_v_heads]
            b = packed_gates[..., num_v_heads : 2 * num_v_heads]
            a.normal_().mul_(0.1)
            b.normal_()
        else:
            a = torch.randn(
                batch_size,
                seq_len,
                num_v_heads,
                device=device,
                dtype=torch.bfloat16,
            ).mul_(0.1)
            b = torch.randn(
                batch_size,
                seq_len,
                num_v_heads,
                device=device,
                dtype=torch.bfloat16,
            )
    else:
        q = torch.randn(
            batch_size,
            seq_len,
            num_q_heads,
            _HEAD_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )
        k = torch.randn_like(q)
        v = torch.randn(
            batch_size,
            seq_len,
            num_v_heads,
            _HEAD_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )
        a = (
            torch.randn(
                batch_size,
                seq_len,
                num_v_heads,
                device=device,
                dtype=torch.float32,
            )
            * 0.1
        ).to(torch.bfloat16)
        b = torch.randn(
            batch_size,
            seq_len,
            num_v_heads,
            device=device,
            dtype=torch.bfloat16,
        )

    pool_size = 2 * batch_size + 2
    state_backing = (
        torch.randn(
            pool_size,
            num_v_heads + 1,
            _HEAD_SIZE,
            _HEAD_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.01
    )
    state = state_backing[:, :num_v_heads]
    state[0].zero_()
    initial_state_indices = torch.arange(
        1,
        2 * batch_size,
        2,
        device=device,
        dtype=torch.int32,
    )
    output_state_indices = initial_state_indices + 1
    cache = (
        torch.full(
            (
                batch_size,
                cache_steps,
                num_v_heads,
                _HEAD_SIZE,
                _HEAD_SIZE,
            ),
            3.25,
            device=device,
            dtype=torch.bfloat16,
        )
        if cache_steps
        else None
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "state": state,
        "state_backing": state_backing,
        "A_log": (
            torch.randn(num_v_heads, device=device, dtype=torch.float32) * 0.1
        ).contiguous(),
        "a": a,
        "dt_bias": (
            torch.randn(num_v_heads, device=device, dtype=torch.float32) * 0.1
        ).contiguous(),
        "b": b,
        "out": torch.zeros(
            batch_size,
            seq_len,
            num_v_heads,
            _HEAD_SIZE,
            device=device,
            dtype=torch.bfloat16,
        ),
        "intermediate_state": cache,
        "initial_state_indices": initial_state_indices,
        "output_state_indices": output_state_indices,
    }


def _bf16_serving_reference(
    tensors: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_all = tensors["q"]
    k_all = tensors["k"]
    v_all = tensors["v"]
    state_pool = tensors["state"]
    indices = tensors["initial_state_indices"]
    assert isinstance(q_all, torch.Tensor)
    assert isinstance(k_all, torch.Tensor)
    assert isinstance(v_all, torch.Tensor)
    assert isinstance(state_pool, torch.Tensor)
    assert isinstance(indices, torch.Tensor)
    state = state_pool.index_select(0, indices.long()).float()
    num_v_heads = v_all.shape[2]
    repeats = num_v_heads // q_all.shape[2]
    outputs = []
    checkpoints = []
    for token in range(q_all.shape[1]):
        q = torch.nn.functional.normalize(q_all[:, token].float(), p=2.0, dim=-1)
        k = torch.nn.functional.normalize(k_all[:, token].float(), p=2.0, dim=-1)
        q = q.repeat_interleave(repeats, dim=1) * _SCALE
        k = k.repeat_interleave(repeats, dim=1)
        a = tensors["a"]
        b = tensors["b"]
        A_log = tensors["A_log"]
        dt_bias = tensors["dt_bias"]
        assert isinstance(a, torch.Tensor)
        assert isinstance(b, torch.Tensor)
        assert isinstance(A_log, torch.Tensor)
        assert isinstance(dt_bias, torch.Tensor)
        alpha = torch.exp(
            -torch.exp(A_log)
            * torch.nn.functional.softplus(a[:, token].float() + dt_bias)
        )
        beta = torch.sigmoid(b[:, token].float())
        state = state * alpha[:, :, None, None]
        v_delta = (
            v_all[:, token].float() - torch.einsum("bhk,bhvk->bhv", k, state)
        ) * beta[:, :, None]
        state = state + v_delta.unsqueeze(-1) * k.unsqueeze(-2)
        outputs.append(torch.einsum("bhk,bhvk->bhv", q, state))
        checkpoints.append(state.to(torch.bfloat16))
    return (
        torch.stack(outputs, dim=1).to(torch.bfloat16),
        torch.stack(checkpoints, dim=1),
        state.to(torch.bfloat16),
    )


def _load_bf16_serving(
    *,
    batch_size: int,
    seq_len: int,
    num_v_heads: int,
    strided_inputs: bool,
    disable_state_update: bool,
    cache_steps: int,
):
    route = cake_gdn.select_cake_gdn_decode_variant(
        arch=_arch(),
        batch_size=batch_size,
        io_dtype="bfloat16",
        state_dtype="bfloat16",
        head_size=_HEAD_SIZE,
        layout="pretranspose",
        num_k_heads=16,
        num_q_heads=16,
        num_v_heads=num_v_heads,
        scale=_SCALE,
        seq_len=seq_len,
        use_qk_l2norm=True,
        strided_inputs=strided_inputs,
        disable_state_update=disable_state_update,
        cache_intermediate_states=cache_steps > 0,
        cache_steps=cache_steps,
    )
    return route, cake_gdn.load_cake_gdn_kernel(route.variant_name, _arch())


def _launch_bf16_serving(
    entry,
    tensors: dict[str, object],
    *,
    batch_size: int,
    num_v_heads: int,
) -> None:
    state_heads = batch_size * num_v_heads
    tile_v = 128 if state_heads >= 1024 else 64 if state_heads >= 512 else 32
    cache = tensors["intermediate_state"]
    entry(
        tensors["q"],
        tensors["k"],
        tensors["v"],
        tensors["state"],
        tensors["A_log"],
        tensors["a"],
        tensors["dt_bias"],
        tensors["b"],
        tensors["out"],
        cache if cache is not None else tensors["out"],
        tensors["initial_state_indices"],
        tensors["output_state_indices"],
        batch_size * num_v_heads * (_HEAD_SIZE // tile_v),
        1,
        1,
    )


@pytest.mark.parametrize("batch_size", [1, 32])
def test_exported_decode_matches_torch_on_caller_stream(batch_size: int) -> None:
    tensors = _make_inputs(batch_size)
    initial_state = tensors["state"].clone()
    expected_out, expected_state = _reference(tensors, initial_state)
    route, entry = _load(batch_size)
    assert route.route_id.endswith(
        "nontranspose_small" if batch_size < 32 else "nontranspose_large"
    )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _launch(entry, tensors, batch_size)
    stream.synchronize()

    torch.testing.assert_close(
        tensors["out"].float(), expected_out.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(tensors["state"], expected_state, atol=1e-3, rtol=1e-3)


def test_exported_decode_is_cuda_graph_safe() -> None:
    batch_size = 1
    tensors = _make_inputs(batch_size)
    initial_state = tensors["state"].clone()
    expected_out, expected_state = _reference(tensors, initial_state)
    _, entry = _load(batch_size)
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        _launch(entry, tensors, batch_size)
    stream.synchronize()
    tensors["state"].copy_(initial_state)
    tensors["out"].zero_()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _launch(entry, tensors, batch_size)
    graph.replay()
    stream.synchronize()

    torch.testing.assert_close(
        tensors["out"].float(), expected_out.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(tensors["state"], expected_state, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    (
        "batch_size",
        "seq_len",
        "num_v_heads",
        "strided_inputs",
        "disable_state_update",
        "cache_steps",
        "pack_gates",
    ),
    [
        (4, 1, 32, True, False, 0, True),
        (4, 2, 32, False, True, 4, True),
        (8, 3, 64, True, True, 3, True),
        (8, 4, 64, True, True, 4, True),
        (8, 4, 32, True, True, 4, False),
        (8, 2, 64, True, False, 0, True),
        (8, 4, 64, True, False, 5, True),
    ],
)
def test_exported_bf16_serving_rows_match_torch_on_caller_stream(
    batch_size: int,
    seq_len: int,
    num_v_heads: int,
    strided_inputs: bool,
    disable_state_update: bool,
    cache_steps: int,
    pack_gates: bool,
) -> None:
    tensors = _make_bf16_serving_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        num_v_heads=num_v_heads,
        strided_inputs=strided_inputs,
        cache_steps=cache_steps,
        pack_gates=pack_gates,
    )
    expected_out, expected_cache, expected_final = _bf16_serving_reference(tensors)
    backing = tensors["state_backing"]
    output_indices = tensors["output_state_indices"]
    cache = tensors["intermediate_state"]
    assert isinstance(backing, torch.Tensor)
    assert isinstance(output_indices, torch.Tensor)
    backing_before = backing.clone()
    cache_before = cache.clone() if isinstance(cache, torch.Tensor) else None
    route, entry = _load_bf16_serving(
        batch_size=batch_size,
        seq_len=seq_len,
        num_v_heads=num_v_heads,
        strided_inputs=strided_inputs,
        disable_state_update=disable_state_update,
        cache_steps=cache_steps,
    )
    assert f"t{seq_len}" in route.route_id

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _launch_bf16_serving(
            entry,
            tensors,
            batch_size=batch_size,
            num_v_heads=num_v_heads,
        )
    stream.synchronize()

    out = tensors["out"]
    assert isinstance(out, torch.Tensor)
    torch.testing.assert_close(out.float(), expected_out.float(), atol=1e-2, rtol=1e-2)
    if isinstance(cache, torch.Tensor):
        torch.testing.assert_close(
            cache[:, :seq_len].float(),
            expected_cache.float(),
            atol=1e-2,
            rtol=1e-2,
        )
        assert cache_before is not None
        assert torch.equal(cache[:, seq_len:], cache_before[:, seq_len:])
        assert torch.equal(backing, backing_before)
    else:
        state = tensors["state"]
        assert isinstance(state, torch.Tensor)
        torch.testing.assert_close(
            state.index_select(0, output_indices.long()).float(),
            expected_final.float(),
            atol=1e-2,
            rtol=1e-2,
        )
        assert torch.equal(backing[:, num_v_heads:], backing_before[:, num_v_heads:])


@pytest.mark.parametrize(
    ("num_v_heads", "pack_gates"),
    [(64, True), (32, False)],
)
def test_exported_bf16_verify_is_cuda_graph_safe(
    num_v_heads: int, pack_gates: bool
) -> None:
    batch_size, seq_len, cache_steps = 8, 4, 4
    tensors = _make_bf16_serving_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        num_v_heads=num_v_heads,
        strided_inputs=True,
        cache_steps=cache_steps,
        pack_gates=pack_gates,
    )
    expected_out, expected_cache, _ = _bf16_serving_reference(tensors)
    backing = tensors["state_backing"]
    out = tensors["out"]
    cache = tensors["intermediate_state"]
    assert isinstance(backing, torch.Tensor)
    assert isinstance(out, torch.Tensor)
    assert isinstance(cache, torch.Tensor)
    backing_before = backing.clone()
    _, entry = _load_bf16_serving(
        batch_size=batch_size,
        seq_len=seq_len,
        num_v_heads=num_v_heads,
        strided_inputs=True,
        disable_state_update=True,
        cache_steps=cache_steps,
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _launch_bf16_serving(
            entry,
            tensors,
            batch_size=batch_size,
            num_v_heads=num_v_heads,
        )
    stream.synchronize()
    eager_out = out.clone()
    eager_cache = cache.clone()
    out.zero_()
    cache.fill_(3.25)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _launch_bf16_serving(
            entry,
            tensors,
            batch_size=batch_size,
            num_v_heads=num_v_heads,
        )
    graph.replay()
    stream.synchronize()

    torch.testing.assert_close(out.float(), expected_out.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        cache.float(), expected_cache.float(), atol=1e-2, rtol=1e-2
    )
    assert torch.equal(out, eager_out)
    assert torch.equal(cache, eager_cache)
    assert torch.equal(backing, backing_before)


def test_exported_prefill_matches_torch_and_is_cuda_graph_safe() -> None:
    tensors = _make_prefill_inputs()
    expected = _prefill_reference(tensors)
    route, entry = _load_prefill()
    assert route.route_id == "cake.gdn_prefill.noncp.single_chunk.dvsplit"
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        _launch_prefill(entry, tensors)
    stream.synchronize()
    torch.testing.assert_close(
        tensors["out"].float(), expected.float(), atol=1e-2, rtol=1e-2
    )
    eager = tensors["out"].clone()
    tensors["out"].zero_()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _launch_prefill(entry, tensors)
    graph.replay()
    stream.synchronize()

    torch.testing.assert_close(
        tensors["out"].float(), expected.float(), atol=1e-2, rtol=1e-2
    )
    assert torch.equal(tensors["out"], eager)
