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

Public-API suite of ``flashinfer.cake_gqa_nondet_train``, the stable entry points of the Cake non-deterministic
packed-varlen GQA training attention (head dimension 256).

Without a device: no entry point is an experimental API (no ``ExperimentalWarning``), importing the module imports no
backend, the re-exported classes and types are the backend's objects (resolved on first access), the module reads no
environment variable, and every keyword of an entry point is the backend's keyword of the same name.  On SM100 /
SM103 / SM107: the entry points return the backend's results, the workspace helpers, and autograd with each
``dkv_acc`` form, without an ``ExperimentalWarning``.
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

import flashinfer.cake_gqa_nondet_train as api
from flashinfer.api_logging import ExperimentalWarning

BACKEND = "flashinfer.experimental.cake.cake_gqa_nondet_train"
ENTRY_POINTS = (
    "cake_gqa_nondet",
    "cake_gqa_nondet_train_forward",
    "cake_gqa_nondet_train_backward",
    "cake_gqa_nondet_train_workspace",
    "cake_gqa_nondet_train_workspace_bytes",
)
# Backend classes and types the module re-exports, resolved on first access.
BACKEND_OBJECTS = (
    "CakeGQANondetDkvAccSource",
    "CakeGQANondetFunction",
    "CakeGQANondetWorkspace",
)
# Facade name -> the backend function it forwards to (the same name).
BACKEND_NAMES = {
    "cake_gqa_nondet": "cake_gqa_nondet",
    "cake_gqa_nondet_train_forward": "cake_gqa_nondet_train_forward",
    "cake_gqa_nondet_train_backward": "cake_gqa_nondet_train_backward",
    "cake_gqa_nondet_train_workspace_bytes": "cake_gqa_nondet_train_workspace_bytes",
}
D = 256
SCALE = 0.0625


def test_entry_points_are_stable_apis() -> None:
    assert sorted(api.__all__) == sorted(ENTRY_POINTS + BACKEND_OBJECTS)
    for name in ENTRY_POINTS:
        assert getattr(getattr(api, name), "is_experimental", False) is False, name


def test_reexported_objects_are_the_backend_objects() -> None:
    from flashinfer.experimental.cake.cake_gqa_nondet_train import cake_backend

    for name in BACKEND_OBJECTS:
        assert getattr(api, name) is getattr(cake_backend, name), name
    assert set(BACKEND_OBJECTS) <= set(dir(api))
    with pytest.raises(AttributeError):
        api.varlen_gqa_attention  # noqa: B018


def test_importing_the_module_imports_no_backend() -> None:
    code = (
        "import sys\n"
        "import flashinfer.cake_gqa_nondet_train as api\n"
        "names = dir(api)\n"
        f"print(any(m.startswith({BACKEND!r}) for m in sys.modules))\n"
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
    from flashinfer.experimental.cake.cake_gqa_nondet_train import cake_backend

    for facade, backend in BACKEND_NAMES.items():
        ours = inspect.signature(inspect.unwrap(getattr(api, facade))).parameters
        theirs = inspect.signature(getattr(cake_backend, backend)).parameters
        assert list(ours) == list(theirs), facade
        for name, param in ours.items():
            assert param.kind == theirs[name].kind, (facade, name)
            if param.default is not inspect.Parameter.empty:
                assert param.default == theirs[name].default, (facade, name)
    workspace = inspect.signature(
        inspect.unwrap(api.cake_gqa_nondet_train_workspace)
    ).parameters
    assert list(workspace) == list(
        inspect.signature(cake_backend.CakeGQANondetWorkspace).parameters
    )


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) not in (
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        pytest.skip("the Cake GQA training attention ships for SM100 / SM103 / SM107")
    from flashinfer.experimental.cake.cake_gqa_nondet_train import cake_jit

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
    from flashinfer.experimental.cake.cake_gqa_nondet_train import cake_backend

    q, k, v, dout, cu_q, cu_k = _inputs((300, 77), (1000, 0), device)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        out, lse = api.cake_gqa_nondet_train_forward(
            q, k, v, cu_q, cu_k, softmax_scale=SCALE
        )
        ref_out, ref_lse = cake_backend.cake_gqa_nondet_train_forward(
            q, k, v, cu_q, cu_k, softmax_scale=SCALE
        )
        assert torch.equal(out, ref_out) and torch.equal(lse, ref_lse)
        assert bool((lse[:, 300:] == -math.inf).all())
        dq, dk, dv = api.cake_gqa_nondet_train_backward(
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
        rdq, rdk, rdv = cake_backend.cake_gqa_nondet_train_backward(
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
        assert torch.equal(dq, rdq)
        ws = api.cake_gqa_nondet_train_workspace(device)
        assert isinstance(ws, api.CakeGQANondetWorkspace) and ws.nbytes == 0
        api.cake_gqa_nondet(q, k, v, cu_q, cu_k, workspace=ws)
        assert api.cake_gqa_nondet_train_workspace_bytes(
            377, 24, 2
        ) == cake_backend.cake_gqa_nondet_train_workspace_bytes(377, 24, 2)
        small = api.cake_gqa_nondet_train_workspace(device, budget_bytes=1 << 26)
        assert small.budget_bytes == 1 << 26 and small.nbytes == 0
        sizes = dict(
            total_k=1077,
            num_heads_kv=4,
            max_seqlen_k=1000,
            row_map=True,
            budget_bytes=1 << 26,
        )
        assert api.cake_gqa_nondet_train_workspace_bytes(
            377, 24, 2, **sizes
        ) == cake_backend.cake_gqa_nondet_train_workspace_bytes(377, 24, 2, **sizes)


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
        warnings.simplefilter("error", ExperimentalWarning)
        out = api.cake_gqa_nondet(
            *leaves,
            cu_q,
            cu_k,
            softmax_scale=SCALE,
            dkv_acc=None if form == "none" else dkv_acc,
        )
        out.backward(dout)
        _, rdk, rdv = api.cake_gqa_nondet_train_backward(
            q,
            k,
            v,
            *api.cake_gqa_nondet_train_forward(
                q, k, v, cu_q, cu_k, softmax_scale=SCALE
            ),
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
