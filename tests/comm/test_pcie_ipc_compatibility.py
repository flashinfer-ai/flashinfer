"""Exercise the original PCIe IPC paths on real non-SM120 devices.

Minimum architecture: SM80. These are correctness tests, not PCIe performance
measurements: an NVLink-connected machine is also a valid CUDA IPC test host.
Topology is probed normally; explicit configurations exercise the island
schedule even when the tuner would omit it for the observed fabric.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from flashinfer import comm
from flashinfer.comm.pcie_ipc_ar import get_pcie_ipc_comm_module
from flashinfer.comm.pcie_ipc_policy import IpcLaunchConfig, IpcVariant
from flashinfer.comm.pcie_ipc_tuning import candidate_tactics
from tests.comm.test_pcie_ipc_all_reduce import (
    _init_process_group,
    multi_process_parallel,
)

# At TP8 the large payload gives every shard two 512-KiB pieces. The smaller
# shape also exercises the production clamp from a requested P2 to actual P1.
_NUMELS = (128 * 1024, 4 * 1024 * 1024)
_GUARD_ELEMS = 8  # Keep the output's starting address 16-byte aligned.


def _require_non_sm120_gpus(world_size: int) -> None:
    available = torch.cuda.device_count()
    if available < world_size:
        pytest.skip(f"need {world_size} CUDA GPUs, found {available}")
    capabilities = [torch.cuda.get_device_capability(i) for i in range(world_size)]
    if any(major < 8 for major, _ in capabilities):
        pytest.skip(f"BF16 compatibility coverage requires SM80+: {capabilities}")
    if any(cc == (12, 0) for cc in capabilities):
        pytest.skip(f"this test requires non-SM120 devices: {capabilities}")
    inaccessible = [
        (rank, peer)
        for rank in range(world_size)
        for peer in range(world_size)
        if peer != rank and not torch.cuda.can_device_access_peer(rank, peer)
    ]
    if inaccessible:
        pytest.skip(f"CUDA IPC test requires peer access; unavailable: {inaccessible}")


def _configs(world_size: int):
    configs = [
        ("seed", None),
        ("unstaged", IpcLaunchConfig(4, 256, IpcVariant.UNSTAGED)),
        ("staged", IpcLaunchConfig(4, 256, IpcVariant.STAGED)),
        ("staged_ring", IpcLaunchConfig(4, 256, IpcVariant.STAGED_RING)),
    ]
    if world_size == 8:
        configs.append(("flat_staged", IpcLaunchConfig(4, 256, IpcVariant.FLAT_STAGED)))
    for pieces in (1, 2):
        configs.append(
            (
                f"ce_ring_p{pieces}",
                IpcLaunchConfig(pieces, 256, IpcVariant.COPY_ENGINE_RING),
            )
        )
        configs.append(
            (
                f"memop_request_fallback_p{pieces}",
                IpcLaunchConfig(pieces, 256, IpcVariant.COPY_ENGINE_RING_MEMOP),
            )
        )
        if world_size == 8:
            configs.append(
                (
                    f"ce_island_p{pieces}",
                    IpcLaunchConfig(pieces, 256, IpcVariant.COPY_ENGINE_ISLAND),
                )
            )
    return configs


def _rank_device_record(rank: int) -> dict:
    props = torch.cuda.get_device_properties(rank)
    return {
        "rank": rank,
        "device": props.name,
        "uuid": str(getattr(props, "uuid", "unavailable")),
        "capability": list(torch.cuda.get_device_capability(rank)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def test_pcie_ipc_module_loads_on_non_sm120() -> None:
    """A runtime load, including driver symbol resolution, on a real device."""
    _require_non_sm120_gpus(1)
    with torch.cuda.device(0):
        module = get_pcie_ipc_comm_module()
        assert module.workspace_size(4, _NUMELS[-1], 2, 128) > 0
        assert not module.memop_supported()
        record = _rank_device_record(0)
        record.update(module_load="passed", memop_supported=False)
        print(
            "PCIE_IPC_COMPAT_MODULE=" + json.dumps(record, sort_keys=True), flush=True
        )


def _compatibility_worker(world_size, rank, port, dtype, result_dir):
    device = torch.device("cuda", rank)
    ws = None
    record = {"rank": rank, "status": "running", "phase": "initialization", "rows": []}

    def save_record():
        path = Path(result_dir) / f"rank-{rank}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)

    try:
        _init_process_group(world_size, rank, port)
        module = get_pcie_ipc_comm_module()
        assert not module.memop_supported()
        # An empty, test-specific cache makes the implicit calls exercise the
        # real seed instead of inheriting a previously tuned configuration.
        ws = comm.PcieIpcAllReduceWorkspace(
            dist.group.WORLD,
            _NUMELS[-1],
            dtype=dtype,
            tune_cache=str(Path(result_dir) / "unused-tune-cache.json"),
        )
        assert not ws.memop_supported, "a non-SM120 group must keep memop disabled"
        record.update(_rank_device_record(rank))
        record.update(
            dtype=str(dtype),
            profile=ws.profile,
            profile_reason=ws.profile_reason,
            module_load="passed",
            memop_supported=ws.memop_supported,
            world_size=world_size,
            rows=[],
        )
        devices = [None] * world_size
        dist.all_gather_object(
            devices, {k: v for k, v in record.items() if k != "rows"}
        )
        if rank == 0:
            print(
                "PCIE_IPC_COMPAT_DEVICES=" + json.dumps(devices, sort_keys=True),
                flush=True,
            )

        configs = _configs(world_size)
        buffers = {}
        for numel in _NUMELS:
            inp = torch.empty(numel, dtype=dtype, device=device)
            storage = torch.empty(numel + 2 * _GUARD_ELEMS, dtype=dtype, device=device)
            out = storage[_GUARD_ELEMS:-_GUARD_ELEMS]
            # A period coprime to the shard lengths makes different chunks
            # distinct; a 16-element pattern would hide wrong-shard copies.
            positions = torch.arange(numel, dtype=torch.int32, device=device) % 31
            buffers[numel] = (inp, storage, out, positions)
            assert ws.supports(inp)
            tactics = candidate_tactics(
                world_size,
                numel=numel,
                elem_size=inp.element_size(),
                profile=ws.profile,
                memop_supported=ws.memop_supported,
            )
            assert all(t[0] != int(IpcVariant.COPY_ENGINE_RING_MEMOP) for t in tactics)

        def prepare(numel, stamp):
            inp, storage, out, positions = buffers[numel]
            # Signed integers and quarter-integers are exactly representable
            # throughout every reduction order, including BF16 partial sums.
            scale = 1.0 if stamp % 2 == 0 else 0.25
            inp.copy_(((positions + rank * 3 + stamp * 5) % 16 - 8).to(dtype) * scale)
            before = inp.clone()
            reference = before.clone()
            dist.all_reduce(reference)
            storage.fill_(42)
            return inp, out, before, reference

        def verify(numel, before, reference):
            inp, storage, out, _ = buffers[numel]
            torch.cuda.synchronize(device)
            torch.testing.assert_close(out, reference, rtol=0, atol=0)
            torch.testing.assert_close(inp, before, rtol=0, atol=0)
            assert torch.all(storage[:_GUARD_ELEMS] == 42).item()
            assert torch.all(storage[-_GUARD_ELEMS:] == 42).item()

        def eager(config, numel, stamp):
            inp, out, before, reference = prepare(numel, stamp)
            ws.rebind_stream()
            if rank == stamp % world_size:
                torch.cuda._sleep(2_000_000)
            ws.all_reduce(inp, out=out, config=config)
            verify(numel, before, reference)

        stamp = 0
        for index, (name, config) in enumerate(configs):
            record["phase"] = f"{name}: eager"
            save_record()
            # Both shapes, both exact numeric families, with the same workspace
            # carrying state across every original and fallback variant.
            for numel in _NUMELS:
                for _ in range(2):
                    eager(config, numel, stamp)
                    stamp += 1
            row = {"config": name, "eager_checks": 4, "graphs": []}
            for depth, numel in zip((3, 4), reversed(_NUMELS), strict=True):
                record["phase"] = f"{name}: capture depth {depth} numel {numel}"
                inp, out, before, reference = prepare(numel, stamp)
                # Eager warmup loads any specialization before capture. A
                # synchronized rebind admits PyTorch's capture-side stream.
                ws.rebind_stream()
                ws.all_reduce(inp, out=out, config=config)
                verify(numel, before, reference)
                dist.barrier()
                ws.rebind_stream()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(depth):
                        ws.all_reduce(inp, out=out, config=config)
                for replay in range(4):
                    record["phase"] = f"{name}: depth {depth} replay {replay}"
                    stamp += 1
                    _, _, before, reference = prepare(numel, stamp)
                    if rank == replay % world_size:
                        torch.cuda._sleep(2_000_000)
                    graph.replay()
                    verify(numel, before, reference)
                    # Return to this captured protocol after another shape and
                    # configuration advanced the same workspace in eager mode.
                    other_config = configs[(index + 1) % len(configs)][1]
                    other_numel = _NUMELS[0] if numel == _NUMELS[1] else _NUMELS[1]
                    stamp += 1
                    record["phase"] = f"{name}: depth {depth} eager interleave {replay}"
                    eager(other_config, other_numel, stamp)
                dist.barrier()
                del graph
                row["graphs"].append(
                    {
                        "depth": depth,
                        "numel": numel,
                        "replays": 4,
                        "eager_interleaves": 4,
                    }
                )
            record["rows"].append(row)
            save_record()
        dist.barrier()
        record["phase"] = "collective cleanup"
        ws.destroy()
        ws = None
        dist.destroy_process_group()
        record["status"] = "passed"
        record["phase"] = "complete"
        save_record()
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        save_record()
        # destroy() contains device synchronization and collective barriers.
        # After a rank-local failure its peers may already be spinning in the
        # next call, so trying that cleanup would hide the original exception.
        # The bounded multiprocessing parent reaps all remaining workers.
        raise


@pytest.mark.parametrize("world_size", [4, 8], ids=["tp4", "tp8"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_pcie_ipc_original_paths_and_graphs_on_non_sm120(world_size, dtype, tmp_path):
    _require_non_sm120_gpus(world_size)
    multi_process_parallel(
        world_size,
        _compatibility_worker,
        args=(dtype, str(tmp_path)),
        timeout_s=600,
    )
    records = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text())
        for rank in range(world_size)
    ]
    expected_configs = [name for name, _ in _configs(world_size)]
    for record in records:
        assert record["status"] == "passed"
        assert record["memop_supported"] is False
        assert [row["config"] for row in record["rows"]] == expected_configs
    print(
        "PCIE_IPC_COMPAT_RESULT="
        + json.dumps(
            {
                "world_size": world_size,
                "dtype": str(dtype),
                "status": "passed",
                "configs": expected_configs,
                "rank_receipts": str(tmp_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
