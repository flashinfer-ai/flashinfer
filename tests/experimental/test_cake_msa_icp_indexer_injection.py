"""The Cake MSA ICP indexer under a tool that injects into the CUDA process (Nsight Systems, compute-sanitizer).

A profiling or checking tool loads its injection library into the process and intercepts the CUDA runtime.  The
package's warmup must still load every kernel of the device's architecture (it raises when it cannot), and the decode
exchange and the prefill select + merge must still run.  Each test starts one child Python process under one tool,
with ``CUDA_MODULE_LOADING=LAZY``: the first decode exchange on emulated ranks (``warmup()``, then the process's first
publish + merge launches through the helpers of ``test_cake_msa_icp_indexer``; legacy then tagged rows), every kernel
of the architecture checked resident, then a prefill select and a prefill merge.  Nsight Systems traces CUDA, NVTX
and the OS runtime (``-t cuda,nvtx,osrt``, CUDA Graph nodes, forked children), as a serving profile does; the merges
spin before the publishers' first launch (the lazy-loading first-exchange child); compute-sanitizer serializes kernels,
so there the publishes are enqueued first.  The tool is found from ``CAKE_MSA_ICP_NSYS`` /
``CAKE_MSA_ICP_COMPUTE_SANITIZER``, on ``PATH``, or under ``CUDA_HOME``; a test whose tool is absent skips.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest
import torch

PACKAGE = os.environ.get(
    "CAKE_MSA_ICP_PACKAGE", "flashinfer.experimental.cake_msa_icp_indexer"
)
REPO = pathlib.Path(__file__).resolve().parents[2]
CHILD_TIMEOUT_SECONDS = 30 * 60

CHILD = f"""
import importlib
import torch

import json
import os
import sys

suite = importlib.import_module("tests.experimental.test_cake_msa_icp_indexer")
cake_backend = importlib.import_module("{PACKAGE}.cake_backend")


def ordered_exchange(mode):
    # The first exchange with every publish enqueued before the merges: a tool
    # that serializes kernels (compute-sanitizer) never runs a merge that waits
    # on a later kernel.
    torch.cuda.set_device(0)
    host, errors = suite._host_mapped_errors()
    report = {{"status": "returned", "lazy": os.environ.get("CUDA_MODULE_LOADING"), "mismatch": []}}
    try:
        cake_backend.warmup(suite.DEV)
        world, slot = 2, suite.FIRST_EXCHANGE_SLOT
        ex = suite.FirstExchange.build(world, mode == "tagged")
        comm = suite.EmulatedComm(world, ex.capacity)
        outs = [ex.new_output() for _ in range(world)]
        torch.cuda.synchronize()
        for r in range(world):
            ex.publish(comm.bases(ex.tagged), comm.generations[r], errors, r)
        for o in range(world):
            ex.merge(comm.recv_slot(o, slot, ex.tagged), comm.expected[o][slot], outs[o], errors)
        torch.cuda.synchronize()
        report["mismatch"] = [
            o for o in range(world) if not torch.equal(outs[o].cpu(), ex.expected(o).to(torch.int32))
        ]
    except Exception as e:
        report["status"] = "raised"
        report["message"] = f"{{type(e).__name__}}: {{e}}"[:400]
    report["errors"] = int(host[0])
    print("CHILD_REPORT " + json.dumps(report), flush=True)


for mode in ("legacy", "tagged"):
    if sys.argv[1] == "spin":  # merges spinning before the publishers' first launch
        suite._first_exchange_child(mode)
    else:
        ordered_exchange(mode)

cake_jit = importlib.import_module("{PACKAGE}.cake_jit")
device = torch.device("cuda", 0)
arch = cake_jit.device_arch(0)
names = [name for name, record in cake_jit.MODULES.items() if arch in record["arches"]]
missing = [name for name in names if (name, 0) not in cake_jit._RESIDENT]
assert names and not missing, (arch, missing)
print("CHILD_RESIDENT_OK", arch, len(names), flush=True)

heads, tokens, columns = cake_backend.NUM_INDEX_HEADS, 64, 2048
generator = torch.Generator(device="cuda").manual_seed(0)
scores = torch.randn(tokens, heads, columns, device=device, generator=generator)
num_valid_pages = torch.full((tokens,), columns, dtype=torch.int32, device=device)
carrier = torch.zeros(heads, tokens, cake_backend.ROW_WORDS, dtype=torch.int32, device=device)
cake_backend.topk_select_prefill(scores, num_valid_pages, carrier, force_begin_blocks=0, force_end_blocks=1)
output = torch.full((tokens, heads // 2, cake_backend.TOPK), -7, dtype=torch.int32, device=device)
errors = torch.zeros(1, dtype=torch.int32, device=device)
cake_backend.merge_candidates(
    carrier.view(2, heads // 2, tokens, cake_backend.ROW_WORDS), num_valid_pages, output, errors,
    init_blocks=0, local_blocks=1, prefill=True,
)
torch.cuda.synchronize(device)
assert int(errors.item()) == 0, int(errors.item())
assert bool((output >= 0).all()) and bool((output < columns).all()), "prefill merge left an id out of range"
print("CHILD_PREFILL_OK", flush=True)
"""


def _tool(executable: str, variable: str) -> str | None:
    explicit = os.environ.get(variable)
    if explicit:
        return explicit
    found = shutil.which(executable)
    if found:
        return found
    for root in (os.environ.get("CUDA_HOME"), "/usr/local/cuda"):
        candidate = pathlib.Path(root or "/nonexistent") / "bin" / executable
        if candidate.is_file():
            return str(candidate)
    return None


def _ready() -> str | None:
    if not torch.cuda.is_available():
        return "needs a CUDA device"
    cake_jit = __import__(f"{PACKAGE}.cake_jit", fromlist=["cake_jit"])
    arch = cake_jit.arch_for_capability(torch.cuda.get_device_capability(0))
    if arch is None:
        return "needs an sm_100a / sm_103a / sm_107a GPU"
    if not any(arch in record["arches"] for record in cake_jit.MODULES.values()):
        return f"this checkout registers no ICP indexer program for {arch}"
    return None


def _run_child(
    command: list[str], exchange: str, tmp_path: pathlib.Path
) -> subprocess.CompletedProcess:
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    env = dict(os.environ, CUDA_MODULE_LOADING="LAZY")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO), env.get("PYTHONPATH")])
    )
    return subprocess.run(
        [*command, sys.executable, str(script), exchange],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_SECONDS,
    )


def _check(result: subprocess.CompletedProcess) -> str:
    output = result.stdout + result.stderr
    tail = output[-6000:]
    assert result.returncode == 0, f"child exited {result.returncode}:\n{tail}"
    reports = [
        json.loads(line.split("CHILD_REPORT ", 1)[1])
        for line in result.stdout.splitlines()
        if "CHILD_REPORT " in line
    ]
    assert len(reports) == 2, (
        f"expected the legacy and tagged first-exchange reports:\n{tail}"
    )
    for report in reports:
        assert report["lazy"] == "LAZY", report
        assert report["status"] == "returned", f"first exchange failed: {report}"
        assert report["errors"] == 0 and report["mismatch"] == [], report
    for marker in ("CHILD_RESIDENT_OK", "CHILD_PREFILL_OK"):
        assert marker in output, f"{marker} missing:\n{tail}"
    return output


def test_warmup_decode_and_prefill_under_nsight_systems(tmp_path: pathlib.Path) -> None:
    reason = _ready()
    if reason:
        pytest.skip(reason)
    nsys = _tool("nsys", "CAKE_MSA_ICP_NSYS")
    if nsys is None:
        pytest.skip("Nsight Systems (nsys) is not installed")
    report = tmp_path / "report"
    # OS-runtime tracing (osrt) is the option that once made residency look in a second libcudart copy.
    trace = [
        "-t",
        "cuda,nvtx,osrt",
        "--cuda-graph-trace=node",
        "--trace-fork-before-exec=true",
    ]
    result = _run_child(
        [nsys, "profile", *trace, "--force-overwrite", "true", "-o", str(report)],
        "spin",
        tmp_path,
    )
    _check(result)


def test_warmup_decode_and_prefill_under_compute_sanitizer(
    tmp_path: pathlib.Path,
) -> None:
    reason = _ready()
    if reason:
        pytest.skip(reason)
    sanitizer = _tool("compute-sanitizer", "CAKE_MSA_ICP_COMPUTE_SANITIZER")
    if sanitizer is None:
        pytest.skip("compute-sanitizer is not installed")
    # Memory errors of the kernels fail the test.  API return codes are not reported: importing flashinfer probes
    # driver entry points with cuGetProcAddress, which memcheck would count as 14 errors before any ICP call.
    memcheck = [
        "--tool",
        "memcheck",
        "--report-api-errors",
        "no",
        "--error-exitcode",
        "97",
        "--print-limit",
        "10",
    ]
    result = _run_child([sanitizer, *memcheck], "ordered", tmp_path)
    output = _check(result)
    assert "ERROR SUMMARY: 0 errors" in output, output[-6000:]
