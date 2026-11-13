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
"""

"""Cross-process determinism of the attention backends.

Two fresh interpreters compute the same problem on the same backend. Process
A first plans and runs several other shapes, so any per-process state a
backend keeps (cuDNN's plan cache and heuristics, autotuning) differs from
process B, which runs the target shape cold. Equal output hashes mean the
backend's result depends only on its inputs, which an RL trainer and its
rollout engine rely on when they run in different processes. cuDNN gives no
such guarantee, so it is ``xfail`` non-strict: a match is recorded, not
required.
"""

import hashlib
import json
import os
import subprocess
import sys

import pytest
import torch

import flashinfer
import flashinfer.cudnn.decode as cudnn_decode
from flashinfer.cudnn.prefill import _cudnn_supports_direct_seqlens
from flashinfer.utils import (
    is_sm90a_supported,
    is_sm100a_supported,
    is_sm110a_supported,
)

DTYPE = torch.bfloat16
NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM = 32, 8, 128
PAGE_SIZE = 16
WORKSPACE_BYTES = 256 << 20

PREFILL_TARGET = (4, 1024)
PREFILL_WARMUP = [(1, 128), (8, 512), (2, 2048), (3, 777)]
DECODE_TARGET = (8, 4096)
DECODE_WARMUP = [(1, 64), (16, 512), (4, 8192), (5, 1333)]

CUDNN_XFAIL = pytest.mark.xfail(
    strict=False, reason="cuDNN guarantees bitwise results within one process only"
)


def _digest(out: torch.Tensor) -> str:
    return hashlib.sha256(
        out.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def _ragged_prefill(backend: str, batch_size: int, seq_len: int) -> torch.Tensor:
    torch.manual_seed(batch_size * 100003 + seq_len)
    device = torch.device("cuda")
    indptr = torch.arange(
        0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=device
    )
    q = torch.randn(
        batch_size * seq_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device
    )
    k = torch.randn(
        batch_size * seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device
    )
    v = torch.randn_like(k)
    workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend=backend
    )
    wrapper.plan(
        indptr,
        indptr,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        causal=True,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    assert wrapper._backend == backend
    return wrapper.run(q, k, v)


def _paged_decode(backend: str, batch_size: int, kv_len: int) -> torch.Tensor:
    torch.manual_seed(batch_size * 100003 + kv_len)
    device = torch.device("cuda")
    kv_lens = torch.randint(max(1, kv_len // 2), kv_len + 1, (batch_size,))
    kv_lens[0] = kv_len
    pages_per_seq = (kv_lens + PAGE_SIZE - 1) // PAGE_SIZE
    indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(pages_per_seq, 0)
    total_pages = int(indptr[-1])
    indices = torch.randperm(total_pages, dtype=torch.int32)
    last_page_len = (kv_lens - (pages_per_seq - 1) * PAGE_SIZE).to(torch.int32)
    kv_cache = torch.randn(
        total_pages, 2, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=device
    )
    q = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "HND", use_tensor_cores=backend != "cudnn", backend=backend
    )
    wrapper.plan(
        indptr.to(device),
        indices.to(device),
        last_page_len.to(device),
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    assert wrapper.resolved_backend == backend
    return wrapper.run(q, kv_cache)


WORKLOADS = {
    "ragged_prefill": (_ragged_prefill, PREFILL_TARGET, PREFILL_WARMUP),
    "paged_decode": (_paged_decode, DECODE_TARGET, DECODE_WARMUP),
}


def _child(workload: str, backend: str, warm: bool) -> str:
    fn, target, warmup = WORKLOADS[workload]
    if warm:
        for shape in warmup:
            fn(backend, *shape)
    out = fn(backend, *target)
    torch.cuda.synchronize()
    return _digest(out)


def _run_child(workload: str, backend: str, warm: bool) -> str:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    proc = subprocess.run(
        [sys.executable, __file__, "child", workload, backend, json.dumps(warm)],
        capture_output=True,
        text=True,
        env=env,
        timeout=1800,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return proc.stdout.strip().splitlines()[-1]


def _skip_unless_supported(workload: str, backend: str) -> None:
    device = torch.device("cuda")
    if backend == "fa3" and not is_sm90a_supported(device):
        pytest.skip("fa3 requires SM90a")
    if backend == "cutlass" and not (
        is_sm100a_supported(device) or is_sm110a_supported(device)
    ):
        pytest.skip("cutlass ragged prefill requires SM100a or SM110a")
    if backend == "cudnn":
        if workload == "ragged_prefill" and not _cudnn_supports_direct_seqlens(DTYPE):
            pytest.skip(
                "cuDNN ragged prefill needs cuDNN 9.24+ and cudnn-frontend 1.25+"
            )
        if workload == "paged_decode" and not cudnn_decode.CUDNN_AVAILABLE:
            pytest.skip("requires the cudnn-frontend python package")


@pytest.mark.parametrize(
    "workload,backend",
    [
        ("ragged_prefill", "fa2"),
        ("ragged_prefill", "fa3"),
        ("ragged_prefill", "cutlass"),
        pytest.param("ragged_prefill", "cudnn", marks=CUDNN_XFAIL),
        ("paged_decode", "fa2"),
        pytest.param("paged_decode", "cudnn", marks=CUDNN_XFAIL),
    ],
)
def test_output_matches_across_processes(workload, backend):
    _skip_unless_supported(workload, backend)
    warm = _run_child(workload, backend, warm=True)
    cold = _run_child(workload, backend, warm=False)
    assert warm == cold, (
        f"{backend} {workload} output differs across processes: "
        f"warm={warm[:16]} cold={cold[:16]}"
    )


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "child":
    print(_child(sys.argv[2], sys.argv[3], json.loads(sys.argv[4])))
