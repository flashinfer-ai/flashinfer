"""Compressed scale programs participate in frozen MoE startup preparation."""

from types import SimpleNamespace

import pytest
import torch

from b12x._lib.compile_plan import (
    compiled_program_available,
    load_programs,
    program_keys,
)
from b12x._lib.compile_pool import CompileJob, compile_in_process, describe_compilation
from b12x.moe.fused_moe._preparation import (
    _x4t_scale_program_payload,
    compile_x4t_scale_programs,
)


def experts_with(value):
    return SimpleNamespace(
        _impl=SimpleNamespace(
            representation=None if value is None else SimpleNamespace(value=value),
        )
    )


@pytest.mark.parametrize(
    "value", (None, SimpleNamespace(), SimpleNamespace(x4t_w13_scale=None))
)
def test_native_weights_do_not_declare_scale_decoders(value):
    assert _x4t_scale_program_payload(experts_with(value)) == ()


def test_compressed_scale_payload_contains_only_geometry():
    first = SimpleNamespace(
        rows=384, columns=112, exception_task_rows=64, exception_row_rotation=0
    )
    second = SimpleNamespace(
        rows=3584, columns=6, exception_task_rows=64, exception_row_rotation=0
    )
    weights = SimpleNamespace(
        x4t_w13_scale=first, x4t_w2_scale=second, x4t_packed_pair_programs=(object(),)
    )
    payload = _x4t_scale_program_payload(experts_with(weights))
    assert payload == ("packed_pair", ((384, 112, 64, 0), (3584, 6, 64, 0)))
    CompileJob.create(
        "b12x.moe.fused_moe._preparation:compile_x4t_scale_programs", payload, 0
    )


def test_tp12_scale_payload_preserves_projection_rotation():
    weights = SimpleNamespace(
        x4t_w13_scale=object(), x4t_packed_pair_programs=None, x4t_w13_row_rotation=256
    )
    assert _x4t_scale_program_payload(experts_with(weights)) == ("tp12", 256)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA compilation")
@pytest.mark.parametrize(
    "payload",
    (
        ("packed_pair", ((384, 112, 64, 0), (3584, 6, 64, 0))),
        ("packed_pair", ((1152, 160, 64, 0), (5120, 18, 64, 0))),
        ("tp12", 256),
    ),
)
def test_scale_factory_declares_and_retains_all_route_abis(payload):
    ordinal = torch.cuda.current_device()
    job = CompileJob.create(
        "b12x.moe.fused_moe._preparation:compile_x4t_scale_programs",
        payload,
        ordinal,
    )
    description = describe_compilation(job)
    assert len(description.programs) == 4
    compile_in_process((description,))
    assert all(compiled_program_available(key) for key in description.programs)
    programs = compile_x4t_scale_programs(payload, ordinal)
    assert set(program_keys(programs)) == set(description.programs)
    load_programs(programs)
