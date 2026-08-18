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
            torch.randn(
                batch_size, 1, num_v_heads, device=device, dtype=torch.float32
            )
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

    torch.testing.assert_close(tensors["out"].float(), expected_out.float(), atol=1e-2, rtol=1e-2)
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

    torch.testing.assert_close(tensors["out"].float(), expected_out.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(tensors["state"], expected_state, atol=1e-3, rtol=1e-3)


def test_exported_prefill_matches_torch_and_is_cuda_graph_safe() -> None:
    tensors = _make_prefill_inputs()
    expected = _prefill_reference(tensors)
    route, entry = _load_prefill()
    assert route.route_id == "cake.gdn_prefill.noncp.single_chunk.dvsplit"
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        _launch_prefill(entry, tensors)
    stream.synchronize()
    torch.testing.assert_close(tensors["out"].float(), expected.float(), atol=1e-2, rtol=1e-2)
    eager = tensors["out"].clone()
    tensors["out"].zero_()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _launch_prefill(entry, tensors)
    graph.replay()
    stream.synchronize()

    torch.testing.assert_close(tensors["out"].float(), expected.float(), atol=1e-2, rtol=1e-2)
    assert torch.equal(tensors["out"], eager)
