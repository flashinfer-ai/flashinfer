from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace

import pytest
import torch

from b12x.preparation import PreparedCall, PreparationSession
from b12x.preparation.types import require_prepared
from b12x.sequence import ple_embedding
from b12x.sequence.ple_embedding import reference
from b12x.sequence.ple_hash.reference import nth_prime_after, ple_hash_packed_reference

from ..conftest import require_b12x


@pytest.fixture
def resources():
    with ExitStack() as stack:
        yield stack


def _small_caps(device, *, tp_rank=1, tp_size=2, max_tokens=5,
                quant_mode="fp8_e4m3_per_tensor", table_memory="device"):
    return ple_embedding.Caps(
        device=device, max_tokens=max_tokens, max_seqs=2, vocab_size=100,
        eos_token_id=99, max_order=3, heads_per_order=2, dense_layer_ordinal=0,
        base_table_size=5, embedding_dim=64, tp_size=tp_size, tp_rank=tp_rank,
        table_alignment=8, quant_mode=quant_mode, table_memory=table_memory,
    )


def _small_geometry(caps):
    return ple_embedding.compute_geometry(
        caps,
        prime_sizes=torch.tensor([5, 7, 11, 13], dtype=torch.int64),
        table_offsets=torch.tensor([0, 5, 12, 23], dtype=torch.int64),
        multipliers=torch.tensor([11, 13, 17], dtype=torch.int64),
    )


def _fp8_weight(shape, device):
    return torch.arange(shape[0] * shape[1], dtype=torch.float32, device=device).remainder(15).sub(7).view(shape).to(torch.float8_e4m3fn).contiguous()


def _nvfp4_weight(shape, device):
    codes = torch.arange(shape[0] * shape[1] * 2, dtype=torch.uint8, device=device).remainder_(16).view(shape[0], shape[1], 2)
    return (codes[..., 0] | (codes[..., 1] << 4)).contiguous()


def _storage(layout):
    if layout.caps.quant_mode == "bf16":
        weight = torch.arange(layout.weight_shape[0] * layout.weight_shape[1], dtype=torch.float32, device=layout.caps.device).remainder(17).sub(8).view(layout.weight_shape).to(torch.bfloat16)
        return weight.contiguous(), None, None
    if layout.caps.quant_mode == "fp8_e4m3_per_tensor":
        return _fp8_weight(layout.weight_shape, layout.caps.device), torch.tensor([0.25], dtype=torch.bfloat16, device=layout.caps.device), None
    scales = torch.arange(layout.weight_scale_shape[0] * layout.weight_scale_shape[1], dtype=torch.float32, device=layout.caps.device).remainder(4).add(1).mul(0.5)
    return (
        _nvfp4_weight(layout.weight_shape, layout.caps.device),
        scales.view(layout.weight_scale_shape).to(torch.float8_e4m3fn).contiguous(),
        torch.tensor([0.25], dtype=torch.float32, device=layout.caps.device),
    )


def _tensors(layout, geometry_tensors, *, num_tokens=4, num_seqs=2):
    weight, weight_scale, weight_scale_2 = _storage(layout)
    max_tokens = layout.caps.max_tokens
    return {
        "weight": weight,
        "weight_scale": weight_scale,
        "weight_scale_2": weight_scale_2,
        "token_ids": torch.arange(max_tokens, dtype=torch.int64, device=layout.caps.device).add_(3),
        "query_start_loc": torch.tensor([0, 2, 4], dtype=torch.int32, device=layout.caps.device),
        "committed_history": torch.tensor([[99, 99], [7, 8]], dtype=torch.int64, device=layout.caps.device),
        "num_seqs": torch.tensor([num_seqs], dtype=torch.int32, device=layout.caps.device),
        "num_tokens": torch.tensor([num_tokens], dtype=torch.int32, device=layout.caps.device),
        "out": torch.full(layout.output_shape, 37, dtype=layout.output_dtype, device=layout.caps.device),
        "_geometry": geometry_tensors,
    }


def _reference(binding):
    layout = binding._state
    geometry = binding._hash_binding.geometry
    return reference.fused(
        binding.weight, binding.weight_scale, binding.token_ids, binding.query_start_loc,
        binding.committed_history, quant_mode=layout.caps.quant_mode,
        weight_scale_2=binding.weight_scale_2, num_seqs=int(binding.num_seqs.item()),
        num_tokens=int(binding.num_tokens.item()), eos_token_id=layout.caps.eos_token_id,
        multipliers=geometry.multipliers, prime_sizes=geometry.prime_sizes,
        table_offsets=geometry.table_offsets, heads_per_order=layout.caps.heads_per_order,
        shard_start=layout.shard_start, embedding_dim=layout.caps.embedding_dim,
        output_dtype=layout.output_dtype,
    )


def _prepared_binding(resources, caps, *, name="embedding", tensors=None):
    geometry = (
        _small_geometry(caps)
        if tensors is None
        else tensors["_geometry"].geometry
    )
    layout = ple_embedding.storage_layout(caps, geometry=geometry)
    geometry_tensors = (
        ple_embedding.allocate_geometry(geometry, device=caps.device)
        if tensors is None
        else tensors["_geometry"]
    )
    tensors = _tensors(layout, geometry_tensors) if tensors is None else tensors
    declaration = ple_embedding.plan(
        caps, geometry=geometry, prime_sizes=geometry_tensors.prime_sizes,
        table_offsets=geometry_tensors.table_offsets, multipliers=geometry_tensors.multipliers,
        invocation=ple_embedding.invocation_from_tensors(**{key: value for key, value in tensors.items() if key != "_geometry"}),
    )

    def prepare_call(state):
        (spec,) = state.layout.scratch_specs()
        tensors["scratch"] = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        trial = state.bind(**{key: value for key, value in tensors.items() if key != "_geometry"})
        return PreparedCall(run=lambda: state.run(trial))

    session = resources.enter_context(PreparationSession(device=caps.device, autotune=False, compile_workers=2))
    result = resources.enter_context(session.prepare((declaration.request(
        name=name, prepare_call=prepare_call),)))
    plan = declaration
    binding = ple_embedding.bind(plan, **{key: value for key, value in tensors.items() if key not in {"_geometry", "scratch"}}, scratch=tensors["scratch"])
    return binding, tensors, layout, session, result


@pytest.mark.parametrize("quant_mode", ["bf16", "fp8_e4m3_per_tensor", "nvfp4_group16"])
def test_host_geometry_and_storage_layout_partition_table(quant_mode):
    caps = _small_caps("cpu", quant_mode=quant_mode)
    geometry = _small_geometry(caps)
    layout = ple_embedding.storage_layout(caps, geometry=geometry)
    storage = ple_embedding.allocate_storage(layout)

    assert geometry.table_vocab_size == 36
    assert geometry.padded_vocab_size == 40
    assert (layout.shard_start, layout.shard_end) == (20, 40)
    assert layout.output_shape == (5, 64)
    assert storage.weight.shape == layout.weight_shape
    if quant_mode == "bf16":
        assert storage.weight_scale is None and storage.weight_scale_2 is None
    elif quant_mode == "fp8_e4m3_per_tensor":
        assert storage.weight_scale is not None and storage.weight_scale_2 is None
    else:
        assert storage.weight_scale is not None and storage.weight_scale_2 is not None
    storage.close()


@pytest.mark.parametrize("tp_rank", range(4))
def test_host_layout_matches_16_head_320m_table_partition(tp_rank):
    caps = ple_embedding.Caps(device="cpu", max_tokens=1, max_seqs=1, vocab_size=248_320,
        eos_token_id=248_044, max_order=3, heads_per_order=8, dense_layer_ordinal=0,
        base_table_size=20_000_000, embedding_dim=2_560, tp_size=4, tp_rank=tp_rank,
        table_alignment=128, quant_mode="bf16")
    layout = ple_embedding.storage_layout(caps, geometry=ple_embedding.compute_geometry(caps))
    assert layout.table_vocab_size == 320_001_446
    assert layout.padded_vocab_size == 320_001_536
    assert (layout.shard_start, layout.shard_end) == (tp_rank * 80_000_384, (tp_rank + 1) * 80_000_384)
    assert layout.weight_shape == (80_000_384, 160)


def test_caps_reject_unsupported_storage_contracts():
    common = dict(device="cpu", max_tokens=2, max_seqs=1, vocab_size=100, eos_token_id=99,
        max_order=3, heads_per_order=2, dense_layer_ordinal=0, base_table_size=5,
        embedding_dim=64, tp_size=2, tp_rank=0, table_alignment=8)
    with pytest.raises(ValueError, match="embedding_dim=.*head_count"):
        ple_embedding.Caps(**{**common, "embedding_dim": 65})
    with pytest.raises(ValueError, match="tp_rank"):
        ple_embedding.Caps(**{**common, "tp_rank": 2})
    with pytest.raises(ValueError, match="quant_mode"):
        ple_embedding.Caps(**{**common, "quant_mode": "int8"})
    with pytest.raises(ValueError, match="table_memory"):
        ple_embedding.Caps(**{**common, "table_memory": "pread"})
    with pytest.raises(TypeError, match="BF16.*scale_dtype must be None"):
        ple_embedding.Caps(
            **{**common, "quant_mode": "bf16", "scale_dtype": torch.bfloat16}
        )
    with pytest.raises(TypeError, match="scale.*torch.bfloat16"):
        ple_embedding.Caps(
            **{
                **common,
                "quant_mode": "fp8_e4m3_per_tensor",
                "scale_dtype": torch.float32,
            }
        )


@torch.inference_mode()
def test_public_bind_rejects_output_aliasing_read_only_table(resources):
    binding, _, _, _, _ = _prepared_binding(
        resources, _small_caps(require_b12x(), quant_mode="bf16")
    )
    kwargs = {
        name: getattr(binding, name)
        for name in (
            "scratch", "weight", "weight_scale", "weight_scale_2", "token_ids",
            "query_start_loc", "committed_history", "num_seqs", "num_tokens",
        )
    }
    with pytest.raises(ValueError, match="mutable out must not overlap"):
        ple_embedding.bind(
            binding.plan,
            **kwargs,
            out=binding.weight.view(binding.out.shape),
        )

@pytest.mark.parametrize("quant_mode", ["bf16", "fp8_e4m3_per_tensor", "nvfp4_group16"])
def test_reference_composes_hash_tp_lookup_and_dequantization(quant_mode):
    caps0, caps1 = _small_caps("cpu", tp_rank=0, quant_mode=quant_mode), _small_caps("cpu", tp_rank=1, quant_mode=quant_mode)
    full_caps = _small_caps("cpu", tp_rank=0, tp_size=1, quant_mode=quant_mode)
    geometry = _small_geometry(caps0)
    layout0 = ple_embedding.storage_layout(caps0, geometry=geometry)
    layout1 = ple_embedding.storage_layout(caps1, geometry=geometry)
    full_layout = ple_embedding.storage_layout(full_caps, geometry=geometry)
    weight, scale, scale_2 = _storage(full_layout)
    token_ids = torch.tensor([3, 4, 5, 6, 0], dtype=torch.int64)
    starts = torch.tensor([0, 2, 4], dtype=torch.int32)
    history = torch.tensor([[99, 99], [7, 8]], dtype=torch.int64)
    outputs = []
    for layout in (layout0, layout1):
        local_scale = scale if quant_mode != "nvfp4_group16" else scale[layout.shard_start:layout.shard_end]
        outputs.append(reference.fused(weight[layout.shard_start:layout.shard_end], local_scale, token_ids, starts, history,
            quant_mode=quant_mode, weight_scale_2=scale_2, num_seqs=2, num_tokens=4,
            eos_token_id=99, multipliers=torch.tensor(geometry.multipliers), prime_sizes=torch.tensor(geometry.prime_sizes),
            table_offsets=torch.tensor(geometry.table_offsets), heads_per_order=2, shard_start=layout.shard_start,
            embedding_dim=64))
    ids = ple_hash_packed_reference(token_ids[:4], starts, history, eos_token_id=99,
        multipliers=torch.tensor(geometry.multipliers), prime_sizes=torch.tensor(geometry.prime_sizes),
        table_offsets=torch.tensor(geometry.table_offsets), heads_per_order=2)
    selected = weight.index_select(0, ids.flatten())
    if quant_mode == "bf16":
        gathered = selected.reshape(4, 4, 16).float()
    elif quant_mode == "fp8_e4m3_per_tensor":
        gathered = selected.reshape(4, 4, 16).float() * scale.float()
    else:
        codes = torch.stack((selected & 0xF, (selected >> 4) & 0xF), dim=-1).reshape(4, 4, 16)
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
        gathered = lut[codes.long()] * scale.index_select(0, ids.flatten()).float().reshape(4, 4, 1) * scale_2.float()
    expected = torch.zeros((5, 64), dtype=torch.bfloat16)
    expected[:4].copy_(gathered.to(torch.bfloat16).flatten(-2))
    torch.testing.assert_close(outputs[0] + outputs[1], expected, rtol=0, atol=0)


def test_nvfp4_reference_decodes_low_nibble_first_and_applies_both_scales():
    codes = torch.stack((torch.arange(16, dtype=torch.uint8), torch.arange(15, -1, -1, dtype=torch.uint8)))
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    actual = reference.lookup(packed, torch.tensor([[2.0], [4.0]], dtype=torch.float8_e4m3fn),
        torch.tensor([[10, 11, 9, 12]], dtype=torch.int64), quant_mode="nvfp4_group16",
        weight_scale_2=torch.tensor([.25]), shard_start=10, embedding_dim=64)
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
    expected = torch.zeros((1, 4, 16), dtype=torch.bfloat16)
    expected[0, 0].copy_((lut * .5).to(torch.bfloat16))
    expected[0, 1].copy_((lut.flip(0)).to(torch.bfloat16))
    torch.testing.assert_close(actual, expected.flatten(-2), rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize("quant_mode", ["bf16", "fp8_e4m3_per_tensor", "nvfp4_group16"])
def test_cuda_prepared_execution_matches_reference_and_preserves_read_only_tensors(resources, quant_mode):
    binding, _, _, _, _ = _prepared_binding(resources, _small_caps(require_b12x(), quant_mode=quant_mode))
    expected = _reference(binding)
    read_names = ["weight", "token_ids", "query_start_loc", "committed_history", "num_seqs", "num_tokens"]
    read_names.extend(name for name in ("weight_scale", "weight_scale_2") if getattr(binding, name) is not None)
    before = {name: getattr(binding, name).clone() for name in read_names}
    actual = ple_embedding.run(binding)
    torch.cuda.synchronize(binding.out.device)
    assert actual.data_ptr() == binding.out.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.count_nonzero(actual[4]).item() == 0
    for name, value in before.items():
        torch.testing.assert_close(getattr(binding, name), value, rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize("quant_mode", ["bf16", "fp8_e4m3_per_tensor", "nvfp4_group16"])
def test_cuda_prepared_execution_compiles_fullgraph(resources, quant_mode):
    binding, _, _, _, _ = _prepared_binding(resources, _small_caps(require_b12x(), quant_mode=quant_mode))
    expected = _reference(binding)
    actual = torch.compile(lambda: ple_embedding.run(binding), fullgraph=True)()
    torch.cuda.synchronize(binding.out.device)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize("quant_mode", ["bf16", "fp8_e4m3_per_tensor", "nvfp4_group16"])
@pytest.mark.parametrize("token_count", [None, 3])
def test_cuda_graph_replay_uses_prepared_execution_without_allocating(resources, quant_mode, token_count):
    binding, _, layout, session, result = _prepared_binding(resources, _small_caps(require_b12x(), quant_mode=quant_mode))
    binding.num_seqs.fill_(2)
    live_tokens = 4 if token_count is None else token_count
    binding.num_tokens.fill_(live_tokens)
    binding.query_start_loc[-1] = live_tokens
    binding.out.fill_(37)
    graph = torch.cuda.CUDAGraph()
    resources.callback(graph.reset)
    with session.capture(), torch.cuda.graph(graph):
        captured = ple_embedding.run(binding, token_count=token_count)
    output_address, scratch_address = captured.data_ptr(), binding.scratch.data_ptr()
    if token_count is not None:
        assert bool((binding.out[token_count:] == 37).all().item())
    if quant_mode == "bf16":
        binding.weight.neg_()
    elif quant_mode == "fp8_e4m3_per_tensor":
        binding.weight.copy_(
            _fp8_weight(layout.weight_shape, binding.out.device)
            .float()
            .neg_()
            .to(torch.float8_e4m3fn)
        )
        binding.weight_scale.fill_(.5)
    else:
        binding.weight.bitwise_xor_(0x88)
        binding.weight_scale.fill_(.5)
        binding.weight_scale_2.fill_(.75)
    binding.token_ids[:2].copy_(torch.tensor([8, 9], dtype=torch.int64, device=binding.out.device))
    binding.query_start_loc.copy_(torch.tensor([0, 2, 0], dtype=torch.int32, device=binding.out.device))
    binding.num_seqs.fill_(1)
    binding.num_tokens.fill_(2)
    expected = _reference(binding)
    allocated_before = torch.cuda.memory_allocated(binding.out.device)
    graph.replay()
    torch.cuda.synchronize(binding.out.device)
    assert torch.cuda.memory_allocated(binding.out.device) == allocated_before
    assert captured.data_ptr() == output_address == binding.out.data_ptr()
    assert binding.scratch.data_ptr() == scratch_address
    torch.testing.assert_close(captured, expected[:token_count], rtol=0, atol=0)


@torch.inference_mode()
def test_cuda_large_local_row_uses_int64_scaled_addressing(resources):
    device = require_b12x()
    prime = nth_prime_after(1 << 24, 1)
    target_id = prime - 1
    caps = ple_embedding.Caps(
        device=device, max_tokens=1, max_seqs=1, vocab_size=prime + 2,
        eos_token_id=prime + 1, max_order=2, heads_per_order=1,
        dense_layer_ordinal=0, base_table_size=prime, embedding_dim=160,
        tp_size=1, tp_rank=0, table_alignment=128,
    )
    geometry = ple_embedding.compute_geometry(
        caps, prime_sizes=torch.tensor([prime], dtype=torch.int64),
        table_offsets=torch.tensor([0], dtype=torch.int64),
        multipliers=torch.tensor([1, 1], dtype=torch.int64),
    )
    layout = ple_embedding.storage_layout(caps, geometry=geometry)
    assert target_id * layout.head_dim > 2**31
    geometry_tensors = ple_embedding.allocate_geometry(geometry, device=device)
    tensors = {
        "weight": torch.empty(
            layout.weight_shape, dtype=layout.weight_dtype, device=device
        ),
        "weight_scale": torch.tensor([.25], dtype=torch.bfloat16, device=device),
        "weight_scale_2": None,
        "token_ids": torch.tensor([target_id], dtype=torch.int64, device=device),
        "query_start_loc": torch.tensor([0, 1], dtype=torch.int32, device=device),
        "committed_history": torch.tensor([[0]], dtype=torch.int64, device=device),
        "num_seqs": torch.tensor([1], dtype=torch.int32, device=device),
        "num_tokens": torch.tensor([1], dtype=torch.int32, device=device),
        "out": torch.empty(layout.output_shape, dtype=layout.output_dtype, device=device),
        "_geometry": geometry_tensors,
    }
    tensors["weight"][target_id].fill_(2.0)
    binding, _, _, _, _ = _prepared_binding(
        resources, caps, name="large-row", tensors=tensors
    )
    torch.testing.assert_close(
        ple_embedding.run(binding), torch.full_like(binding.out, .5), rtol=0, atol=0
    )


def _disk_binding(resources, oracle_binding, tmp_path, *, name="disk"):
    caps = replace(oracle_binding._state.caps, device=oracle_binding.out.device, table_memory="io_uring")
    geometry = oracle_binding._hash_binding.geometry.geometry
    layout = ple_embedding.storage_layout(caps, geometry=geometry)
    table = ple_embedding.DiskTable(layout, shard_rows=7, queue_depth=4)
    payloads = [(False, oracle_binding.weight)]
    if caps.quant_mode == "nvfp4_group16":
        payloads.append((True, oracle_binding.weight_scale))
    for scale, local in payloads:
        full = torch.zeros((layout.padded_vocab_size, local.shape[1]), dtype=local.dtype)
        full[layout.shard_start:layout.shard_end].copy_(local.cpu())
        for index, start in enumerate(range(0, layout.padded_vocab_size, 7)):
            path = tmp_path / f"{scale}-{index}.bin"
            path.write_bytes(bytes(4093) + full[start:start + 7].view(torch.uint8).numpy().tobytes())
            table.add_shard(index, str(path), 4093, scale=scale)
    geometry_tensors = ple_embedding.allocate_geometry(geometry, device=caps.device)
    tensors = _tensors(layout, geometry_tensors)
    tensors.update(weight=None, disk_table=table, weight_scale=(oracle_binding.weight_scale.to(caps.device) if caps.quant_mode == "fp8_e4m3_per_tensor" else None),
        weight_scale_2=None if oracle_binding.weight_scale_2 is None else oracle_binding.weight_scale_2.to(caps.device),
        token_ids=oracle_binding.token_ids.to(caps.device), query_start_loc=oracle_binding.query_start_loc.to(caps.device),
        committed_history=oracle_binding.committed_history.to(caps.device), num_seqs=oracle_binding.num_seqs.to(caps.device), num_tokens=oracle_binding.num_tokens.to(caps.device))
    return _prepared_binding(resources, caps, name=name, tensors=tensors), table


@torch.inference_mode()
@pytest.mark.parametrize("quant_mode", ["bf16", "fp8_e4m3_per_tensor", "nvfp4_group16"])
def test_disk_preparation_matches_resident_and_graph_consumes_only_output(
    resources, quant_mode, tmp_path, monkeypatch
):
    resident, _, _, _, _ = _prepared_binding(
        resources,
        _small_caps(require_b12x(), quant_mode=quant_mode),
        name="resident",
    )
    (binding, _, _, session, result), table = _disk_binding(resources, resident, tmp_path)
    assert table.layout == binding._state
    with pytest.raises(RuntimeError, match="after binding"):
        table.add_shard(0, str(tmp_path / "absent"), 0)
    expected = ple_embedding.run(resident).clone()
    torch.testing.assert_close(ple_embedding.run(binding), expected, rtol=0, atol=0)
    consumed = torch.empty_like(binding.out)
    graph = torch.cuda.CUDAGraph()
    resources.callback(graph.reset)
    with session.capture(), torch.cuda.graph(graph):
        torch.mul(binding.out, 2, out=consumed)
    binding.token_ids[:2].copy_(
        torch.tensor([8, 9], dtype=torch.int64, device=binding.out.device)
    )
    resident.token_ids.copy_(binding.token_ids)
    expected = ple_embedding.run(resident).clone()
    ple_embedding.run(binding)
    graph.replay()
    torch.testing.assert_close(consumed, expected * 2, rtol=0, atol=0)
    with monkeypatch.context() as patch:
        patch.setattr(table._cache, "read_rows", lambda *args: pytest.fail("consumer graph must not issue disk I/O"))
        graph.replay()
        torch.testing.assert_close(consumed, expected * 2, rtol=0, atol=0)
    with monkeypatch.context() as patch:
        patch.setattr(torch.compiler, "is_compiling", lambda: True)
        with pytest.raises(RuntimeError, match="torch.compile"):
            ple_embedding.run(binding)


@torch.inference_mode()
@pytest.mark.parametrize("quant_mode", ["bf16", "fp8_e4m3_per_tensor", "nvfp4_group16"])
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_disk_compact_rows_preserve_duplicates_and_tp_shard_boundaries(resources, quant_mode, tp_rank, tmp_path):
    device = require_b12x()
    resident, _, _, _, _ = _prepared_binding(resources, _small_caps(device, tp_rank=tp_rank, quant_mode=quant_mode), name="resident-rows")
    (binding, _, layout, session, result), table = _disk_binding(resources, resident, tmp_path, name="disk-rows")
    edge = ((layout.shard_start + 7) // 7) * 7
    ids = torch.tensor([[layout.shard_start, layout.shard_start, edge - 1, edge],
        [layout.shard_start - 1, layout.shard_end - 1, layout.shard_end, -1],
        [layout.table_vocab_size, layout.padded_vocab_size, edge, edge],
        [edge + 1, edge - 1, layout.shard_start, -2], [-1, -1, -1, -1]], dtype=torch.int64)
    binding._ids.copy_(ids.to(device))
    torch.cuda.synchronize(binding.out.device)
    with table._cache.transaction():
        table._cache.read_rows(binding._ids, ids.numel())
        torch.cuda.synchronize(binding.out.device)
    binding.num_tokens.fill_(4)
    state = require_prepared(binding.plan, "sequence.ple_embedding", binding.out.device)
    scale = table.weight_scale if quant_mode == "nvfp4_group16" else binding.weight_scale
    with session.capture():
        state.run_lookup(
            table.weight, scale, binding.weight_scale_2,
            binding._ids, binding.num_tokens, binding.out, token_count=layout.caps.max_tokens,
        )
    expected = reference.lookup(resident.weight, resident.weight_scale, ids.masked_fill(ids >= layout.table_vocab_size, -1).to(binding.out.device),
        quant_mode=quant_mode, weight_scale_2=resident.weight_scale_2, num_tokens=4,
        shard_start=layout.shard_start, embedding_dim=layout.caps.embedding_dim)
    torch.testing.assert_close(binding.out, expected, rtol=0, atol=0)
    stats = table.stats()
    assert stats["lookups"] == ids.numel()
    assert stats["unique_blocks"] < stats["lookups"]
