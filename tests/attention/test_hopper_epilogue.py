"""Exercise the real Hopper epilogue with shared-memory writers deliberately late."""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import torch

from flashinfer.jit import env as jit_env
from flashinfer.jit.core import current_compilation_context, gen_jit_spec
from flashinfer.utils import is_sm90a_supported


def _instrument_epilogue(source):
    """Only add common observation/delay hooks; never remove a production barrier."""
    begin = source.index("template <typename Ktraits>\nstruct CollectiveEpilogue")
    end = source.index("\n};", begin) + 3
    body = source[begin:end].replace(
        "struct CollectiveEpilogue {", "struct TestEpilogue {", 1
    )
    replacements = {
        "TiledMma tiled_mma, int thread_idx, BlockCoord const& block_coord) {": "TiledMma tiled_mma, int thread_idx, BlockCoord const& block_coord, "
        "ProbeContext const& probe) {",
        "    cute::copy(smem_tiled_copy_O, tOrO_retile, tOsO);": "    probe.before_stsm();\n"
        "    cute::copy(smem_tiled_copy_O, tOrO_retile, tOsO);",
        "    cutlass::arch::fence_view_async_shared();  // ensure smem writes are visible to TMA\n": "    cutlass::arch::fence_view_async_shared();\n"
        "    probe.after_stsm();\n",
        "    TiledCopyO gmem_tiled_copy_O;": "    probe.before_global(&sO(cute::Int<5>{}, cute::Int<240>{}));\n"
        "    TiledCopyO gmem_tiled_copy_O;",
    }
    for before, after in replacements.items():
        assert body.count(before) == 1, (
            f"Epilogue instrumentation anchor changed: {before}"
        )
        body = body.replace(before, after, 1)
    return (
        source[: source.index("#ifndef")] + "namespace flashinfer {\n" + body + "\n}\n"
    )


def _spec():
    header = jit_env.FLASHINFER_INCLUDE_DIR / "flashinfer/attention/hopper/epilogue.cuh"
    fixture = Path(__file__).parents[1] / "cuda/hopper_epilogue.cu"
    instrumented = _instrument_epilogue(header.read_text())
    digest = hashlib.sha256(instrumented.encode() + fixture.read_bytes()).hexdigest()[
        :16
    ]
    name = f"test_hopper_epilogue_{digest}"
    generated = jit_env.FLASHINFER_GEN_SRC_DIR / name
    generated.mkdir(parents=True, exist_ok=True)
    destination = generated / "epilogue_test_impl.cuh"
    if not destination.exists() or destination.read_text() != instrumented:
        destination.write_text(instrumented)
    return gen_jit_spec(
        name,
        [fixture],
        extra_include_paths=[generated],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[9]
        ),
    )


def _check_output(output):
    bits = output.view(torch.uint16)
    assert bool((bits[:, 6] == 0x4000).all()), (
        "production epilogue read stale shared data"
    )
    assert bool((bits[:, :6] == 0x4040).all())
    assert bool((bits[:, 7:] == 0x4040).all())


def _worker(destination):
    spec = _spec()
    assert spec.is_compiled, (
        "the parent must build the fixture before starting the worker"
    )
    module = spec.load()
    owner, reader, offset = list(module.mapping())
    assert (owner, reader) == (20, 190), "the target must cross the delayed writer WG"
    (destination / "mapping.json").write_text(
        json.dumps(dict(owner=owner, reader=reader, shared_offset=offset))
    )
    (destination / "active.json").write_text(json.dumps(dict(case="production")))
    output = torch.empty((128, 8, 256), device="cuda", dtype=torch.bfloat16)
    events = torch.empty(21, device="cuda", dtype=torch.int64)
    module.run_probe(output, events)
    torch.cuda.synchronize()
    # Keep full values before the parent makes any correctness assertion.
    torch.save(
        dict(output=output.cpu(), events=events.cpu()), destination / "production.pt"
    )


def _wait_for_worker(process, timeout=90, cleanup_timeout=5):
    try:
        return process.wait(timeout=timeout)
    except BaseException as primary:
        # A separate session does not receive the parent's terminal interrupt.
        # Until wait reaps this child, its PID cannot be reused by another group.
        cleanup_errors = []
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if process.returncode is not None:
                break
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass  # It exited between the failed wait and the signal.
            except BaseException as error:
                cleanup_errors.append(error)
            try:
                process.wait(timeout=cleanup_timeout)
                break
            except subprocess.TimeoutExpired as error:
                if sig == signal.SIGKILL:
                    cleanup_errors.append(error)
            except BaseException as error:
                cleanup_errors.append(error)
        for error in cleanup_errors:
            message = f"epilogue worker cleanup failed: {type(error).__name__}: {error}"
            try:
                add_note = getattr(primary, "add_note", None)
                if add_note is not None:
                    add_note(message)
                else:
                    print(message, file=sys.stderr)
            except BaseException:
                pass  # Reporting must not replace the original interruption.
        raise


def test_epilogue_waits_for_shared_writers(tmp_path):
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("Requires SM90a")

    # Compile before starting the GPU watchdog; the child directly loads this artifact.
    _spec().build()
    environment = os.environ.copy()
    with (tmp_path / "worker.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(tmp_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        returncode = _wait_for_worker(process)
    assert returncode == 0, (tmp_path / "worker.log").read_text()

    payload = torch.load(tmp_path / "production.pt", weights_only=True)
    output, events = payload["output"], payload["events"].tolist()
    assert events[20] == 1
    assert all(events[8 + warp] - events[warp] >= 1 << 22 for warp in range(4))
    _check_output(output)
    assert events[18] == 0x4000 and events[19] == 15
    assert max(events[8:12]) <= events[16] <= events[17]


if __name__ == "__main__":
    assert len(sys.argv) == 3 and sys.argv[1] == "--worker"
    _worker(Path(sys.argv[2]))
