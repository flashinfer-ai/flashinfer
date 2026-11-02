"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The Cake BF16 RMSNorm training forward and backward beside resident NCCL
collectives (issues #7038, #8066): two processes, one per GPU, joined by a
``torch.distributed`` NCCL process group, run this package's forward and
backward on the compute stream while an all-gather or a reduce-scatter is in
flight on a side stream -- with ``deterministic=True`` (the deterministic
package) and ``deterministic=False`` (this package's kernels), BF16 and FP32
weights, the plain and the residual route.

Two kinds of resident collective:

* ``waiting``: one rank issues the collective at once while its peer joins
  ``PEER_DELAY_MS`` later, so the first rank's NCCL kernel holds its SMs and
  waits on its peer while the call runs (the collective of a training step whose
  other ranks lag).  Every timed call ends before that collective completes
  (checked with events; otherwise the run proves nothing and the case fails).
  The call keeps its speed as the resident-kernel stall suite requires: the
  median busy time is at most ``MAX_RATIO`` times, or at most ``MAX_EXTRA_MS``
  above, the median alone.
* ``traffic``: both ranks keep ``TRAFFIC_ITERS`` back-to-back collectives of
  ``TRAFFIC_MIB`` MiB in flight (NVLink traffic and its HBM reads and writes)
  while both run the call; the call ends while the traffic is still in flight.
  The ratio is reported (memory-bandwidth sharing slows the call, not a stall).

In both, a call that waited for the collective would take its lifetime (about
``PEER_DELAY_MS``, or the whole traffic sequence): the median busy time may be at
most ``MAX_ALL_EXTRA_MS`` above the median alone.  Every call's outputs match:
``y``, ``rstd`` and ``dx`` bitwise across the alone and busy calls, ``dw``
bitwise for a deterministic program and within ``DW_REL_L2`` for the
non-deterministic single kernel; the first call alone is checked against FP64 at
the deterministic budget.  Fewer than two supported GPUs: every case is skipped
with that reason (not run).
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import statistics
import subprocess
import sys
import traceback
from pathlib import Path

import pytest
import torch

WORLD = 2
COLLECTIVES = ("all_gather", "reduce_scatter")
MODES = ("waiting", "traffic")
# (hidden, residual, weight dtype): deterministic=True runs BF16 weights only
# (the deterministic package's routes).
ROUTES = (
    (2048, False, "bf16"),
    (6144, False, "f32"),
    (6144, True, "bf16"),
    (6144, True, "f32"),
)
ROWS = (4096, 16384)
SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
WARMUP = 3
RUNS = 5
PRE_SLEEP_CYCLES = 1_000_000
PEER_DELAY_MS = 50.0
WAIT_BUFFER_MIB = 4
TRAFFIC_ITERS = 64
TRAFFIC_MIB = 64
MAX_RATIO = 1.5
MAX_EXTRA_MS = 0.020
MAX_ALL_EXTRA_MS = 1.0
DW_REL_L2 = 1e-6
WORKER_TIMEOUT_S = 2400
SEED = 7038
REPO_ROOT = Path(__file__).resolve().parents[2]


def _case_id(case) -> str:
    collective, mode, deterministic, hidden, residual, w, rows = case
    det = "det" if deterministic else "nondet"
    res = "res-" if residual else ""
    return f"{collective}-{mode}-{det}-{res}h{hidden}-{w}-T{rows}"


def _cases():
    for collective in COLLECTIVES:
        for mode in MODES:
            for deterministic in (True, False):
                for hidden, residual, w in ROUTES:
                    if deterministic and w != "bf16":
                        continue
                    for rows in ROWS:
                        yield (
                            collective,
                            mode,
                            deterministic,
                            hidden,
                            residual,
                            w,
                            rows,
                        )


CASES = list(_cases())


# --------------------------------------------------------------------------- worker (one process per GPU)


def _worker(rank: int, port: int, out_dir: Path) -> int:
    import torch.distributed as dist

    from tests.test_helpers import cake_rmsnorm_train_reference as ref
    from tests.test_helpers.cake_package import cake_package

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD,
        device_id=device,
    )
    cpu = dist.new_group(backend="gloo")
    pkg = cake_package("cake_rmsnorm_train")
    loader = cake_package("cake_rmsnorm_train", "cake_jit")
    arch = loader.arch_for_capability(torch.cuda.get_device_capability(device))
    side = torch.cuda.Stream(device)
    results_path = out_dir / f"rank{rank}.jsonl"
    results_path.write_text("")

    def buffers(mib: int, collective: str):
        n = mib * (1 << 20) // 2
        if collective == "all_gather":
            return (
                torch.ones(n, dtype=torch.bfloat16, device=device),
                torch.empty(WORLD * n, dtype=torch.bfloat16, device=device),
            )
        return (
            torch.ones(WORLD * n, dtype=torch.bfloat16, device=device),
            torch.empty(n, dtype=torch.bfloat16, device=device),
        )

    def issue(collective: str, inp, outp) -> None:
        if collective == "all_gather":
            dist.all_gather_into_tensor(outp, inp)
        else:
            dist.reduce_scatter_tensor(outp, inp)

    def timed(call, comm=None):
        """``(call ms, ms from the call's start to the side stream's completion or None, outputs)``."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        done = None
        if comm is not None:
            with torch.cuda.stream(side):
                comm()
                done = torch.cuda.Event(enable_timing=True)
                done.record(side)
        torch.cuda._sleep(PRE_SLEEP_CYCLES)
        start.record()
        result = call()
        end.record()
        torch.cuda.synchronize()
        return (
            start.elapsed_time(end),
            None if done is None else start.elapsed_time(done),
            result,
        )

    # cycles of the peer's delay: measured against an event pair once
    probe_a, probe_b = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    probe_a.record()
    torch.cuda._sleep(10_000_000)
    probe_b.record()
    torch.cuda.synchronize()
    delay_cycles = int(10_000_000 * PEER_DELAY_MS / probe_a.elapsed_time(probe_b))

    for collective in COLLECTIVES:  # NCCL connections before any timing
        for mib in (WAIT_BUFFER_MIB, TRAFFIC_MIB):
            inp, outp = buffers(mib, collective)
            issue(collective, inp, outp)
    torch.cuda.synchronize()
    dist.barrier(group=cpu)

    for index, case in enumerate(CASES):
        collective, mode, deterministic, hidden, residual, w, rows = case
        record = {"case": _case_id(case), "rank": rank, "arch": arch, "errors": []}
        try:
            generator = torch.Generator(device=device).manual_seed(
                (SEED * 1_000_003 + index * 101 + rank) % (2**31 - 1)
            )
            kw = dict(dtype=torch.float32, device=device, generator=generator)
            x = torch.randn(rows, hidden, **kw).to(torch.bfloat16)
            u = torch.randn(rows, hidden, **kw).to(torch.bfloat16) if residual else None
            g = torch.randn(rows, hidden, **kw).to(torch.bfloat16)
            g_h = (
                torch.randn(rows, hidden, **kw).to(torch.bfloat16) if residual else None
            )
            weight = (1 + 0.1 * torch.randn(hidden, **kw)).to(ref.W_DTYPES[w])
            eps = ref.EPS[hidden]
            ws = pkg.cake_rmsnorm_train_backward_workspace(
                rows, hidden, device, residual=residual, deterministic=deterministic
            )

            def call():
                outputs = pkg.cake_rmsnorm_train_forward(
                    x, weight, eps, residual=u, deterministic=deterministic
                )
                normalized = outputs[2] if residual else x
                dx, dw = pkg.cake_rmsnorm_train_backward(
                    g,
                    normalized,
                    weight,
                    outputs[1],
                    g_residual=g_h,
                    workspace=ws,
                    deterministic=deterministic,
                )
                return tuple(t.clone() for t in (*outputs, dx, dw))

            form = (
                "det"
                if deterministic
                else ref.form(
                    loader,
                    ref.window_records(loader, arch, hidden, "bwd", residual, w, rows),
                )
            )
            record["form"] = form
            for _ in range(WARMUP):
                timed(call)
            # FP64 check of one call alone
            first = call()
            torch.cuda.synchronize()
            y, rstd = first[0], first[1]
            normalized = first[2] if residual else x
            dx, dw = first[-2], first[-1]
            try:
                y_ref, rstd_ref, _n = ref.reference_forward(x, weight, eps, residual=u)
                ref.assert_bf16_close(y, y_ref, "y")
                assert ref.rel_l2(rstd, rstd_ref) <= ref.RSTD_REL_L2, "rstd"
                dx_ref, dw_ref = ref.reference_backward(
                    g, normalized, weight, rstd, g_h=g_h
                )
                ref.assert_bf16_close(dx, dx_ref, "dx")
                ref.assert_dw(
                    dw,
                    dw_ref,
                    g,
                    normalized,
                    rstd,
                    "dw",
                    "pair" if form == "det" else form,
                )
            except AssertionError as exc:
                record["errors"].append(f"fp64: {str(exc)[:400]}")

            alone, busy, comm_ms, outs = [], [], [], [first]
            small = buffers(WAIT_BUFFER_MIB, collective)
            big = buffers(TRAFFIC_MIB, collective)
            for _run in range(RUNS):
                ms, _c, result = timed(call)
                alone.append(ms)
                outs.append(result)
                if mode == "waiting":
                    for measuring in range(WORLD):
                        dist.barrier(group=cpu)
                        if rank == measuring:
                            ms, comm, result = timed(
                                call, comm=lambda: issue(collective, *small)
                            )
                            busy.append(ms)
                            comm_ms.append(comm)
                            outs.append(result)
                        else:
                            with torch.cuda.stream(side):
                                torch.cuda._sleep(delay_cycles)
                                issue(collective, *small)
                            torch.cuda.synchronize()
                else:
                    dist.barrier(group=cpu)

                    def traffic():
                        for _ in range(TRAFFIC_ITERS):
                            issue(collective, *big)

                    ms, comm, result = timed(call, comm=traffic)
                    busy.append(ms)
                    comm_ms.append(comm)
                    outs.append(result)
            dist.barrier(group=cpu)
            for result in outs[1:]:
                for i, (a, b) in enumerate(zip(result, first, strict=True)):
                    is_dw = i == len(first) - 1
                    if is_dw and form == "single":
                        if ref.rel_l2(a, b) > DW_REL_L2:
                            record["errors"].append(
                                f"dw moved: rel-L2 {ref.rel_l2(a, b):.3e}"
                            )
                    elif not torch.equal(a, b):
                        record["errors"].append(f"output {i} differs across calls")
            record.update(
                alone_ms=alone,
                busy_ms=busy,
                comm_ms=comm_ms,
                nccl=str(torch.cuda.nccl.version()).replace(" ", ""),
                device=torch.cuda.get_device_name(device),
            )
        except Exception:  # noqa: BLE001 - recorded, the parent fails the case
            record["errors"].append(traceback.format_exc()[-1500:])
            torch.cuda.synchronize()
            dist.barrier(group=cpu)
        with open(results_path, "a") as out:
            out.write(json.dumps(record) + "\n")
    dist.barrier(group=cpu)
    dist.destroy_process_group()
    return 0


# --------------------------------------------------------------------------- parent (pytest)


def _supported_gpus() -> int:
    if not torch.cuda.is_available():
        return 0
    return sum(
        torch.cuda.get_device_capability(i) in SUPPORTED_CAPABILITIES
        for i in range(torch.cuda.device_count())
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def nccl_results(tmp_path_factory):
    gpus = _supported_gpus()
    if gpus < WORLD:
        pytest.skip(
            f"needs {WORLD} SM100 / SM103 / SM107 GPUs for one process per GPU, found {gpus}"
        )
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_nccl_available():
        pytest.skip("torch.distributed with NCCL is not available")
    out_dir = tmp_path_factory.mktemp("cake_rmsnorm_nccl")
    port = _free_port()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if p
    )
    env.setdefault("NCCL_DEBUG", "WARN")
    procs = []
    with contextlib.ExitStack() as stack:
        logs = [
            stack.enter_context(open(out_dir / f"rank{rank}.log", "w"))
            for rank in range(WORLD)
        ]
        for rank in range(WORLD):
            procs.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        "--worker",
                        str(rank),
                        str(port),
                        str(out_dir),
                    ],
                    env=env,
                    stdout=logs[rank],
                    stderr=subprocess.STDOUT,
                    cwd=str(REPO_ROOT),
                )
            )
        codes = []
        for proc in procs:
            try:
                codes.append(proc.wait(timeout=WORKER_TIMEOUT_S))
            except subprocess.TimeoutExpired:
                for p in procs:
                    p.kill()
                codes.append("timeout")
    tails = {
        rank: (out_dir / f"rank{rank}.log").read_text()[-3000:] for rank in range(WORLD)
    }
    if any(code != 0 for code in codes):
        pytest.fail(f"NCCL workers exited {codes}; log tails: {tails}")
    results = {}
    for rank in range(WORLD):
        for line in (out_dir / f"rank{rank}.jsonl").read_text().splitlines():
            record = json.loads(line)
            results[(record["case"], record["rank"])] = record
    print(f"\nNCCL_WORKERS out={out_dir} records={len(results)} codes={codes}")
    return results


@pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
def test_forward_backward_beside_a_resident_nccl_collective(case, nccl_results) -> None:
    _collective, mode, _deterministic, _hidden, _residual, _w, _rows = case
    case_id = _case_id(case)
    problems = []
    for rank in range(WORLD):
        record = nccl_results.get((case_id, rank))
        assert record is not None, f"{case_id}: no result from rank {rank}"
        if record["errors"]:
            problems.append(f"rank {rank}: {record['errors']}")
            continue
        alone = statistics.median(record["alone_ms"])
        busy = statistics.median(record["busy_ms"])
        ratio, extra = busy / alone, busy - alone
        resident = sum(
            c is not None and c > b
            for b, c in zip(record["busy_ms"], record["comm_ms"], strict=True)
        )
        print(
            f"NCCL {record['arch']} {case_id} rank={rank} form={record['form']} "
            f"alone_ms={alone:.4f} busy_ms={busy:.4f} ratio={ratio:.3f} extra_ms={extra:.4f} "
            f"resident_runs={resident}/{len(record['busy_ms'])} "
            f"comm_ms_min={min(record['comm_ms']):.3f} nccl={record['nccl']}"
        )
        if resident != len(record["busy_ms"]):
            problems.append(
                f"rank {rank}: the collective completed before {len(record['busy_ms']) - resident} "
                f"timed calls ended (busy {record['busy_ms']}, collective {record['comm_ms']} ms)"
            )
        if extra > MAX_ALL_EXTRA_MS:
            problems.append(
                f"rank {rank}: alone {alone:.4f} ms, beside the {mode} collective {busy:.4f} ms "
                f"(at most {MAX_ALL_EXTRA_MS} ms more): the call waited for it"
            )
        if mode == "waiting" and not (ratio <= MAX_RATIO or extra <= MAX_EXTRA_MS):
            problems.append(
                f"rank {rank}: alone {alone:.4f} ms, beside a resident collective {busy:.4f} ms "
                f"(ratio {ratio:.3f} > {MAX_RATIO} and {extra:.4f} ms > {MAX_EXTRA_MS} ms)"
            )
    assert not problems, f"{case_id}: " + "; ".join(problems)


if __name__ == "__main__" and len(sys.argv) == 5 and sys.argv[1] == "--worker":
    sys.exit(_worker(int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])))
