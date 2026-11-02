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

Public-API suite of ``flashinfer.gqa_train``, the experimental entry points of the Cake packed-varlen GQA training
attention (head dimension 256).

Without a device: every entry point is an experimental API, importing the module imports no backend, the module reads
no environment variable, and every keyword of an entry point is the backend's keyword of the same name.  On SM100 /
SM103: the entry points return the backend's results, the workspace helpers, and autograd with each ``dkv_acc`` form.
"""

from __future__ import annotations

import ast
import inspect
import math
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch

import flashinfer.gqa_train as api
from flashinfer.api_logging import ExperimentalWarning

ENTRY_POINTS = (
    "varlen_gqa_attention",
    "varlen_gqa_attention_fwd",
    "varlen_gqa_attention_bwd",
    "gqa_train_workspace",
    "gqa_train_workspace_bytes",
)
# Facade name -> the backend function it forwards to.
BACKEND_NAMES = {
    "varlen_gqa_attention": "varlen_gqa_attention",
    "varlen_gqa_attention_fwd": "gqa_attention_fwd",
    "varlen_gqa_attention_bwd": "gqa_attention_bwd",
    "gqa_train_workspace_bytes": "workspace_bytes",
}
D = 256
SCALE = 0.0625


def test_entry_points_are_experimental() -> None:
    assert sorted(api.__all__) == sorted(ENTRY_POINTS)
    for name in ENTRY_POINTS:
        assert getattr(api, name).is_experimental is True, name


def test_importing_the_module_imports_no_backend() -> None:
    code = (
        "import sys\n"
        "import flashinfer.gqa_train\n"
        "print(any(m.startswith('flashinfer.experimental.cake_gqa_train') for m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"


def test_the_module_reads_no_environment_variable() -> None:
    tree = ast.parse(Path(api.__file__).read_text())
    hits = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv")
    ]
    assert not hits, hits


def test_keywords_match_the_backend() -> None:
    from flashinfer.experimental.cake_gqa_train import cake_backend

    for facade, backend in BACKEND_NAMES.items():
        ours = inspect.signature(inspect.unwrap(getattr(api, facade))).parameters
        theirs = inspect.signature(getattr(cake_backend, backend)).parameters
        assert list(ours) == list(theirs), facade
        for name, param in ours.items():
            assert param.kind == theirs[name].kind, (facade, name)
            if param.default is not inspect.Parameter.empty:
                assert param.default == theirs[name].default, (facade, name)
    workspace = inspect.signature(inspect.unwrap(api.gqa_train_workspace)).parameters
    assert list(workspace) == list(inspect.signature(cake_backend.Workspace).parameters)


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) not in (
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        pytest.skip("the Cake GQA training attention ships for SM100 / SM103 / SM107")
    from flashinfer.experimental.cake_gqa_train import cake_jit

    if not (Path(cake_jit.__file__).parent / "build.json").is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    return torch.device("cuda:0")


def _inputs(q_lens, k_lens, device, h_q=24, h_kv=4):
    g = torch.Generator(device=device).manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=device).to(torch.bfloat16)

    cu = lambda lens: torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=device
    )  # noqa: E731
    t_q, t_k = sum(q_lens), sum(k_lens)
    return (
        randn(t_q, h_q, D),
        randn(t_k, h_kv, D),
        randn(t_k, h_kv, D),
        randn(t_q, h_q, D),
        cu(q_lens),
        cu(k_lens),
    )


def test_entry_points_return_the_backend_results() -> None:
    device = _device_or_skip()
    from flashinfer.experimental.cake_gqa_train import cake_backend

    q, k, v, dout, cu_q, cu_k = _inputs((300, 77), (1000, 0), device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        out, lse = api.varlen_gqa_attention_fwd(
            q, k, v, cu_q, cu_k, softmax_scale=SCALE
        )
        ref_out, ref_lse = cake_backend.gqa_attention_fwd(
            q, k, v, cu_q, cu_k, softmax_scale=SCALE
        )
        assert torch.equal(out, ref_out) and torch.equal(lse, ref_lse)
        assert bool((lse[:, 300:] == -math.inf).all())
        dq, dk, dv = api.varlen_gqa_attention_bwd(
            q,
            k,
            v,
            out,
            lse,
            dout,
            cu_q,
            cu_k,
            softmax_scale=SCALE,
            dkv_dtype=torch.float32,
        )
        _rdq, rdk, rdv = cake_backend.gqa_attention_bwd(
            q,
            k,
            v,
            out,
            lse,
            dout,
            cu_q,
            cu_k,
            softmax_scale=SCALE,
            dkv_dtype=torch.float32,
        )
        assert (
            dk.dtype == torch.float32 and torch.equal(dk, rdk) and torch.equal(dv, rdv)
        )
        assert dq.dtype == torch.bfloat16 and dq.shape == q.shape
        ws = api.gqa_train_workspace(device)
        assert ws.nbytes == 0
        api.varlen_gqa_attention(q, k, v, cu_q, cu_k, workspace=ws)
        assert api.gqa_train_workspace_bytes(
            377, 24, 2
        ) == cake_backend.workspace_bytes(377, 24, 2)


@pytest.mark.parametrize("form", ("none", "window", "rowmap"))
def test_autograd_dkv_forms(form) -> None:
    device = _device_or_skip()
    q, k, v, dout, cu_q, cu_k = _inputs((257, 129), (513, 129), device)
    row_map = torch.cat([torch.arange(513), torch.arange(129)]).to(
        device=device, dtype=torch.int32
    )
    holder = {}

    def dkv_acc():
        if form == "window":
            holder["k"] = torch.empty(k.shape, dtype=torch.float32, device=device)
            holder["v"] = torch.empty(v.shape, dtype=torch.float32, device=device)
            return holder["k"], holder["v"]
        holder["k"] = torch.ones(513, 4, D, device=device)
        holder["v"] = torch.ones(513, 4, D, device=device)
        return holder["k"], holder["v"], row_map

    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        out = api.varlen_gqa_attention(
            *leaves,
            cu_q,
            cu_k,
            softmax_scale=SCALE,
            dkv_acc=None if form == "none" else dkv_acc,
        )
        out.backward(dout)
        _, rdk, rdv = api.varlen_gqa_attention_bwd(
            q,
            k,
            v,
            *api.varlen_gqa_attention_fwd(q, k, v, cu_q, cu_k, softmax_scale=SCALE),
            dout,
            cu_q,
            cu_k,
            softmax_scale=SCALE,
            dkv_dtype=torch.float32,
        )
    torch.cuda.synchronize()
    assert leaves[0].grad is not None and leaves[0].grad.dtype == torch.bfloat16
    if form == "none":
        assert leaves[1].grad.dtype == torch.bfloat16
        torch.testing.assert_close(leaves[1].grad.float(), rdk, rtol=2**-8, atol=0)
        return
    assert leaves[1].grad is None and leaves[2].grad is None
    if form == "window":
        assert torch.equal(holder["k"], rdk) and torch.equal(holder["v"], rdv)
    else:
        expected = torch.ones(513, 4, D, device=device).index_add(
            0, row_map.long(), rdk
        )
        torch.testing.assert_close(holder["k"], expected, rtol=1e-6, atol=1e-6)
