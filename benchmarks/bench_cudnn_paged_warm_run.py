# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare warm public paged calls, including host submission and final drain.

Export a trusted parent with ``git show BASE:flashinfer/prefill.py > /tmp/parent.py``.
Run with ``--parent-source /tmp/parent.py`` on the candidate checkout. Only the
parent public run method is loaded; plan(), inputs, dependencies and runtime
are shared. No provider or engine is forced within cuDNN.
"""

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time

import torch

import cudnn
import flashinfer
import flashinfer.prefill as fp


def parent_method(source):
    raw = source.read_bytes()
    cls = next(
        n
        for n in ast.parse(raw).body
        if isinstance(n, ast.ClassDef)
        and n.name == "BatchPrefillWithPagedKVCacheWrapper"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "run" and len(n.body) > 20
    )
    namespace = dict(vars(fp))
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    result = namespace["run"]
    assert inspect.signature(result) == inspect.signature(
        fp.BatchPrefillWithPagedKVCacheWrapper.run
    )
    return result


def measure(call, validate, repeats, calls):
    for _ in range(10):
        call()
    torch.cuda.synchronize()
    validate()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for _ in range(calls):
            call()
        submitted = time.perf_counter_ns()
        torch.cuda.synchronize()
        drained = time.perf_counter_ns()
        validate()
        samples.append(
            {
                "submit_us": (submitted - start) / calls / 1000,
                "drained_us": (drained - start) / calls / 1000,
            }
        )
    return {key: statistics.median(s[key] for s in samples) for key in samples[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--parent-source",
        type=Path,
        required=True,
        help="Trusted source file whose public method will be executed",
    )
    ap.add_argument("--cases", type=Path, default=Path(__file__).with_suffix(".json"))
    ap.add_argument("--case", help="Optional single case ID")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--expected-backend-version", type=int, default=92400)
    args = ap.parse_args()
    assert args.repeats > 0 and args.calls > 0
    assert cudnn.backend_version() == args.expected_backend_version
    original = parent_method(args.parent_source)
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cudnn": cudnn.backend_version(),
                "frontend": cudnn.__version__,
                "flashinfer_source": fp.__file__,
                "frontend_source": cudnn.__file__,
                "parent_sha256": hashlib.sha256(
                    args.parent_source.read_bytes()
                ).hexdigest(),
                "candidate_sha256": hashlib.sha256(
                    Path(fp.__file__).read_bytes()
                ).hexdigest(),
                "timing": "warm host loop / completed eager, us per call; no JIT or planning",
            }
        ),
        flush=True,
    )
    cases = json.loads(args.cases.read_text())
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
    assert cases
    for c in cases:
        torch.manual_seed(c["tensor_seed"])
        qs, ks = c["q_lengths"], c["kv_lengths"]
        page, dim, hq, hk = c["page_size"], c["head_dim"], c["hq"], c["hk"]
        counts = [(n + page - 1) // page for n in ks]
        qptr = torch.tensor([0] + qs, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        kvptr = torch.tensor([0] + counts, dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        ids = torch.arange(sum(counts), device="cuda", dtype=torch.int32)
        last = torch.tensor([(n - 1) % page + 1 for n in ks], dtype=torch.int32)
        q = torch.randn(sum(qs), hq, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(sum(counts), hk, page, dim, device="cuda", dtype=q.dtype)
        v = torch.randn_like(k)
        wrappers, outputs = {}, {}
        for backend in ("fa2", "cudnn"):
            workspace = torch.empty(256 << 20, device="cuda", dtype=torch.uint8)
            w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace, "HND", backend=backend
            )
            w.plan(
                qptr,
                kvptr,
                ids,
                last,
                hq,
                hk,
                dim,
                page,
                causal=True,
                window_left=c["window_left"],
                q_data_type=q.dtype,
                max_token_per_sequence=max(qs),
                max_sequence_kv=max(ks),
            )
            wrappers[backend], outputs[backend] = w, torch.empty_like(q)
        w, out = wrappers["cudnn"], outputs["cudnn"]
        assert w._cudnn_plan is not None, "This benchmark requires the planned path"
        baseline = lambda: wrappers["fa2"].run(q, (k, v), out=outputs["fa2"])
        parent = lambda: original(w, q, (k, v), out=out)
        candidate = lambda: w.run(q, (k, v), out=out)
        baseline()
        reference = outputs["fa2"].clone()
        entries = [
            ("fa2", baseline, outputs["fa2"]),
            ("parent", parent, out),
            ("candidate", candidate, out),
        ]
        for order, ordered in (("forward", entries), ("reverse", entries[::-1])):
            for name, call, output in ordered:
                output.fill_(float("nan"))

                def validate():
                    torch.testing.assert_close(output, reference, atol=0.02, rtol=0.02)

                metrics = measure(call, validate, args.repeats, args.calls)
                print(
                    json.dumps(
                        {"case": c["id"], "order": order, "entry": name, **metrics}
                    ),
                    flush=True,
                )
        # Compilation caches remain warm across the cohort; no captured graph or
        # fixed pointer cache is retained by this benchmark.
        wrappers.clear()
        outputs.clear()


if __name__ == "__main__":
    main()
