"""Independent writer geometry, cache identity, and mutable captured slot maps."""

from contextlib import contextmanager

import pytest
import torch

from b12x.attention import paged_kv_write
from b12x.attention.paged_kv_write._tuning import PagedKVWriteQuery
from b12x.preparation import PreparationSession, PreparedCall

from ..conftest import require_b12x


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_256_head_contract_does_not_require_attention_support(dtype):
    declaration = paged_kv_write.plan(
        paged_kv_write.Caps(
            device="cuda:0",
            num_kv_heads=2,
            head_dim_k=256,
            head_dim_v=256,
            page_size=64,
            kv_dtype=dtype,
        )
    )
    assert declaration.query.head_dim_k == 256
    assert declaration.component_id == "attention.paged_kv_write"


@pytest.mark.parametrize(
    "override",
    [
        {"num_kv_heads": 0},
        {"num_kv_heads": True},
        {"head_dim_k": 255},
        {"head_dim_v": 2**31},
        {"page_size": 0},
        {"kv_dtype": "float16"},
    ],
)
def test_invalid_geometry_is_rejected_without_device_access(override):
    values = dict(
        num_kv_heads=2,
        head_dim_k=256,
        head_dim_v=256,
        page_size=64,
        kv_dtype="bfloat16",
    )
    with pytest.raises(ValueError):
        PagedKVWriteQuery(**(values | override))


@contextmanager
def _prepared(args, dtype):
    declaration = paged_kv_write.plan(
        paged_kv_write.Caps(
            device=args["key"].device,
            num_kv_heads=2,
            head_dim_k=256,
            head_dim_v=256,
            page_size=64,
            kv_dtype=dtype,
        )
    )
    with PreparationSession(
        device=args["key"].device, autotune=False, compile_workers=1
    ) as session:
        session.prepare(
            (
                declaration.request(
                    name="paged_kv_write",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(**args)
                    ),
                ),
            )
        )
        session.freeze()
        yield declaration


@pytest.mark.parametrize("fp8", [False, True])
def test_256_head_strided_cache_capture_reuses_changed_slot_mapping(fp8):
    device = require_b12x()
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    packed = torch.zeros((3, 2, 2, 64, 512), device=device, dtype=dtype)
    layer = packed[:, 1].transpose(1, 2)
    k_cache, v_cache = layer[..., :256], layer[..., 256:]
    qkv = (torch.randn((8, 3072), device=device) * 60).to(torch.bfloat16)
    key, value = qkv[:, 512:1024].view(8, 2, 256), qkv[:, 2048:2560].view(8, 2, 256)
    slots = torch.full((8,), -1, device=device, dtype=torch.int64)
    k_scale = torch.tensor([3.3], device=device)
    v_scale = torch.tensor([0.7], device=device)
    args = dict(
        key=key,
        value=value,
        k_cache=k_cache,
        v_cache=v_cache,
        slot_mapping=slots,
        k_scale=k_scale if fp8 else None,
        v_scale=v_scale if fp8 else None,
    )
    expected = packed.clone()
    expected_layer = expected[:, 1].transpose(1, 2)
    mappings = (
        [0, 63, 64, 127, 128, 191, -1, 192],
        [191, -1, 1, 126, 65, 129, -9, 2**40],
        [-1] * 8,
    )
    updates = [torch.tensor(mapping, device=device) for mapping in mappings]
    with _prepared(args, dtype) as plan:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            paged_kv_write.run(plan, **args)
        for mapping, update in zip(mappings, updates):
            packed.zero_()
            expected.zero_()
            slots.copy_(update)
            for token, slot in enumerate(mapping):
                if 0 <= slot < 192:
                    page, offset = divmod(slot, 64)
                    k, v = key[token].float(), value[token].float()
                    if fp8:
                        k = (k / k_scale).clamp(-448, 448)
                        v = (v / v_scale).clamp(-448, 448)
                    expected_layer[page, offset, :, :256] = k.to(dtype)
                    expected_layer[page, offset, :, 256:] = v.to(dtype)
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            assert torch.equal(packed.view(torch.uint8), expected.view(torch.uint8))
        graph.reset()


def test_writer_rejects_invalid_metadata_and_overlapping_cache_rows():
    device = require_b12x()
    rows = torch.zeros((2, 2, 256), dtype=torch.bfloat16, device=device)
    cache = torch.zeros((2, 64, 2, 256), dtype=torch.bfloat16, device=device)
    slots = torch.full((2,), -1, dtype=torch.int64, device=device)
    args = dict(
        key=rows, value=rows, k_cache=cache, v_cache=cache.clone(), slot_mapping=slots
    )
    with _prepared(args, torch.bfloat16) as plan:
        with pytest.raises(ValueError, match="int64"):
            paged_kv_write.run(plan, **(args | {"slot_mapping": slots.to(torch.int32)}))
        with pytest.raises(ValueError, match="nonoverlapping"):
            paged_kv_write.run(plan, **(args | {"k_cache": cache[:1].expand_as(cache)}))
        with pytest.raises(ValueError, match="fewer rows"):
            paged_kv_write.run(plan, **(args | {"key": rows[:1]}))
        paged_kv_write.run(plan, **(args | {"slot_mapping": slots[:0]}))
