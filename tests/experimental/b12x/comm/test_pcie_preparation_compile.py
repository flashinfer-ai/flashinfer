"""PCIe owner plans retain compiler identities without connecting peer GPUs."""

import pytest
import torch
from unittest.mock import Mock

from b12x._lib.compile_plan import compiled_program_available, load_programs, program_keys
from b12x._lib.compile_pool import CompileJob, compile_in_process, describe_compilation
from b12x.comm.pcie._owner_preparation import compile_owner_surface
from b12x.comm.pcie._dcp_preparation import compile_dcp_surface
from b12x.comm.pcie._tuning import PcieQuery, TUNING
from b12x.preparation import FrozenMapping


@pytest.mark.parametrize("world", [9, 10, 12])
@pytest.mark.parametrize("owner", ["DcpAllToAll", "DcpAllToAllPool"])
@pytest.mark.parametrize("operation", ["all_gather_heads", "lse_reduce_scatter"])
def test_dcp_attention_accepts_non_power_of_two_groups(world, owner, operation):
    from b12x.comm.pcie.pcie_dcp_a2a import _staging_layout

    query = PcieQuery(
        surface=f"{owner}.{operation}", world_size=world, rank=world - 1,
        topology="pcie_ipc", call=FrozenMapping(), setup=FrozenMapping(),
    )
    TUNING.validate_query(query, None)
    layout = _staging_layout(
        signal_bytes=4096, world_size=world, max_batch_size=8,
        total_heads=world * 10, head_dim=512, query_head_dim=576,
    )
    assert layout.staging1_offset == layout.staging0_offset + layout.slot_bytes


@pytest.mark.parametrize("world", [9, 10, 12])
@pytest.mark.parametrize("operation", [
    "all_gather_pair", "all_gather_pair_kimi_topk", "kimi_topk16",
])
def test_attention_group_support_does_not_enable_kimi_projection_shards(world, operation):
    query = PcieQuery(
        surface=f"DcpAllToAll.{operation}", world_size=world, rank=0,
        topology="pcie_ipc", call=FrozenMapping(), setup=FrozenMapping(),
    )
    with pytest.raises(ValueError, match="supports world sizes"):
        TUNING.validate_query(query, None)


@pytest.mark.parametrize("options,expected_threads", [({}, 512), ({"threads": 128}, 128)])
def test_prepared_pair_call_executes_both_output_bindings(options, expected_threads):
    from b12x.comm.pcie._dcp_preparation import _DcpExecutionState, prepare_call
    runtime = Mock()
    first, second = torch.ones(2, 32), torch.ones(2, 16)
    out_first, out_second = torch.empty(2, 64), torch.empty(2, 32)
    query = PcieQuery(surface="DcpAllToAll.all_gather_pair", world_size=2,
                     rank=0, topology="pcie_ipc", call=FrozenMapping(),
                     setup=FrozenMapping())
    state = _DcpExecutionState(query, runtime, {})
    call = prepare_call(state, local_first=first, local_second=second,
                        out_first=out_first, out_second=out_second, **options)
    call.run()
    runtime._all_gather_pair_on_device.assert_called_once_with(
        first, second, out_first, out_second, state=state, threads=expected_threads,
    )
    assert call.output == (out_first, out_second)


@pytest.mark.parametrize("module,factory", [
    ("_hierarchical_cute", "get_hierarchical_launcher"),
    ("_island_rs_cute", "get_island_rs_launcher"),
    ("_vocab_argmax_cute", "get_vocab_argmax_launcher"),
])
def test_owner_launcher_exposes_compiler_dependencies(monkeypatch, module, factory):
    from importlib import import_module
    from b12x._lib.compile_plan import ProgramKey

    kernels = import_module(f"b12x.comm.pcie.{module}")
    program = ProgramKey("cute", "test-owner", factory)
    raw = Mock(__b12x_programs__=(program,))
    monkeypatch.setattr(kernels, "b12x_compile", lambda *args, **kwargs: raw)
    monkeypatch.setattr(kernels, "make_ptr", lambda *args, **kwargs: 0)
    monkeypatch.setattr(kernels, "current_cuda_stream", lambda: 0)
    monkeypatch.setattr(
        kernels, "raise_if_kernel_resolution_frozen", lambda *args, **kwargs: None
    )

    launcher = getattr(kernels, factory).__wrapped__(16, 0, 0, wait_nanosleep_cycles=24)
    assert program_keys(launcher) == (program,)
    assert launcher.__b12x_dependencies__ == (raw,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA compilation")
@pytest.mark.parametrize("surface,vectorized", [
    ("PCIeHierarchicalAllReduce.all_reduce", False),
    ("PCIeHierarchicalAllReduce.all_reduce", True),
    ("PCIeIslandRSAllReduce.all_reduce", False),
    ("VocabParallelArgmax.fused_add_argmax", False),
])
def test_owner_compile_factory_retains_native_program(surface, vectorized):
    query = PcieQuery(
        surface=surface, world_size=16, rank=0, topology="pcie_ipc",
        call=FrozenMapping({
            "threads": 128 if "IslandRS" in surface else 256,
            "wait_nanosleep_cycles": 24, "double_buffered": False,
            "deferred_consumption": True, "vectorized": vectorized,
        }),
        setup=FrozenMapping(),
    )
    payload = TUNING.encode_query(query)
    ordinal = torch.cuda.current_device()
    job = CompileJob.create(
        "b12x.comm.pcie._owner_preparation:compile_owner_surface", payload, ordinal
    )
    description = describe_compilation(job)
    assert len(description.programs) == 1
    compile_in_process((description,))
    assert all(compiled_program_available(key) for key in description.programs)
    launchers = compile_owner_surface(payload, ordinal)
    assert set(program_keys(launchers)) == set(description.programs)
    load_programs(launchers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA compilation")
@pytest.mark.parametrize("operation", [
    "lse_reduce_scatter", "all_gather_heads", "all_gather_pair",
    "all_gather_pair_kimi_topk", "kimi_topk16",
])
def test_dcp_compile_factory_retains_eager_and_graph_programs(operation):
    query = PcieQuery(
        surface=f"DcpAllToAll.{operation}", world_size=16, rank=0,
        topology="pcie_ipc",
        call=FrozenMapping({"threads": 256, "dtype": "bf16"}),
        setup=FrozenMapping(),
    )
    payload = TUNING.encode_query(query)
    ordinal = torch.cuda.current_device()
    description = describe_compilation(CompileJob.create(
        "b12x.comm.pcie._dcp_preparation:compile_dcp_surface", payload, ordinal
    ))
    assert len(description.programs) == (1 if operation == "kimi_topk16" else 2)
    compile_in_process((description,))
    launchers = compile_dcp_surface(payload, ordinal)
    assert set(program_keys(launchers)) == set(description.programs)
    load_programs(launchers)
