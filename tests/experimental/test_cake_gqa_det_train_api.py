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

Public-API suite of ``flashinfer.cake_gqa_det_train``, the stable entry points of the Cake deterministic
packed-varlen GQA training attention (head dimension 256).

Without a device: no entry point is an experimental API (no ``ExperimentalWarning``), importing the module imports no
backend, the re-exported classes and types are the backend's objects (resolved on first access), the module reads no
environment variable, every keyword of an entry point is the backend's keyword of the same name (the workspace
budget's ``None`` standing for the backend's 16 GiB default), every entry point takes the parameters of the
non-deterministic package's entry of the same role, the extras of ``cake_gqa_det_train_workspace_bytes`` are
keyword-only, and every entry point forwards every argument, ``budget_bytes`` included, to the backend unchanged.
On SM100 / SM103 / SM107: the entry points return the backend's results bit for bit, the workspace helpers, and
autograd with each ``dkv_acc`` form, without an ``ExperimentalWarning``.
"""

from __future__ import annotations

import ast
import inspect
import math
import subprocess
import sys
import types
import warnings
from pathlib import Path

import pytest
import torch

import flashinfer.cake_gqa_det_train as api
from flashinfer.api_logging import ExperimentalWarning

BACKEND = "flashinfer.experimental.cake.cake_gqa_det_train"
ENTRY_POINTS = (
    "cake_gqa_det",
    "cake_gqa_det_train_forward",
    "cake_gqa_det_train_backward",
    "cake_gqa_det_train_workspace",
    "cake_gqa_det_train_workspace_bytes",
)
# Backend classes and types the module re-exports, resolved on first access.
BACKEND_OBJECTS = (
    "CakeGQADetDkvAccSource",
    "CakeGQADetFunction",
    "CakeGQADetWorkspace",
)
# Facade name -> the backend function it forwards to (the same name).
BACKEND_NAMES = {
    "cake_gqa_det": "cake_gqa_det",
    "cake_gqa_det_train_forward": "cake_gqa_det_train_forward",
    "cake_gqa_det_train_backward": "cake_gqa_det_train_backward",
    "cake_gqa_det_train_workspace_bytes": "cake_gqa_det_train_workspace_bytes",
}
# The keyword-only sizes and switches of ``cake_gqa_det_train_workspace_bytes``.
WORKSPACE_BYTES_EXTRAS = (
    "total_k",
    "num_heads_kv",
    "max_seqlen_k",
    "row_map",
    "budget_bytes",
    "cu_seqlens_q_host",
    "cu_seqlens_k_host",
)
D = 256
SCALE = 0.0625


def _backend():
    from flashinfer.experimental.cake.cake_gqa_det_train import cake_backend

    return cake_backend


def test_entry_points_are_stable_apis() -> None:
    assert sorted(api.__all__) == sorted(ENTRY_POINTS + BACKEND_OBJECTS)
    for name in ENTRY_POINTS:
        assert getattr(getattr(api, name), "is_experimental", False) is False, name


def test_reexported_objects_are_the_backend_objects() -> None:
    cake_backend = _backend()
    for name in BACKEND_OBJECTS:
        assert getattr(api, name) is getattr(cake_backend, name), name
    assert set(BACKEND_OBJECTS) <= set(dir(api))
    with pytest.raises(AttributeError):
        api.varlen_gqa_attention  # noqa: B018
    with pytest.raises(AttributeError):
        api.cake_gqa_nondet  # noqa: B018


def test_importing_the_module_imports_no_backend() -> None:
    code = (
        "import sys\n"
        "import flashinfer.cake_gqa_det_train as api\n"
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
    """Every facade parameter is the backend's of the same name, kind and default; the one exception is the
    workspace budget, whose facade default ``None`` stands for the backend default
    (``cake_jit.DEFAULT_WORKSPACE_BUDGET``, 16 GiB)."""
    cake_backend = _backend()
    from flashinfer.experimental.cake.cake_gqa_det_train import cake_jit

    assert cake_jit.DEFAULT_WORKSPACE_BUDGET == 16 << 30
    pairs = [
        (
            facade,
            inspect.signature(inspect.unwrap(getattr(api, facade))).parameters,
            inspect.signature(getattr(cake_backend, backend)).parameters,
        )
        for facade, backend in BACKEND_NAMES.items()
    ]
    pairs.append(
        (
            "cake_gqa_det_train_workspace",
            inspect.signature(
                inspect.unwrap(api.cake_gqa_det_train_workspace)
            ).parameters,
            inspect.signature(cake_backend.CakeGQADetWorkspace).parameters,
        )
    )
    for facade, ours, theirs in pairs:
        assert list(ours) == list(theirs), facade
        for name, param in ours.items():
            assert param.kind == theirs[name].kind, (facade, name)
            if name == "budget_bytes":
                assert param.default is None, facade
                assert theirs[name].default == cake_jit.DEFAULT_WORKSPACE_BUDGET, facade
            elif param.default is not inspect.Parameter.empty:
                assert param.default == theirs[name].default, (facade, name)


# The one default that differs from the non-deterministic package: ``total_k`` / ``num_heads_kv`` of the workspace-bytes
# entry are ``None`` here (not given: the scratch alone; one without the other raises ``ValueError``) and ``0`` there
# (the scratch alone).  (ours, theirs) per (entry, parameter).
NONDET_DEFAULT_DIFFERENCES = {
    ("cake_gqa_det_train_workspace_bytes", "total_k"): (None, 0),
    ("cake_gqa_det_train_workspace_bytes", "num_heads_kv"): (None, 0),
}


def test_the_interface_matches_the_nondet_package() -> None:
    """Each entry point takes the parameters of the non-deterministic package's entry of the same role, in order,
    with the same kinds and defaults (``cu_seqlens_q_host`` / ``cu_seqlens_k_host`` included; the defaults of
    ``NONDET_DEFAULT_DIFFERENCES`` aside), so a caller switches the mode by the module name alone."""
    import flashinfer.cake_gqa_nondet_train as nondet

    for name in ENTRY_POINTS:
        other = name.replace("cake_gqa_det", "cake_gqa_nondet")
        ours, theirs = (
            inspect.signature(inspect.unwrap(fn)).parameters
            for fn in (getattr(api, name), getattr(nondet, other))
        )
        assert [(p.name, p.kind) for p in ours.values()] == [
            (p.name, p.kind) for p in theirs.values()
        ], name
        for key, param in ours.items():
            pair = (param.default, theirs[key].default)
            want = NONDET_DEFAULT_DIFFERENCES.get((name, key))
            if want is None:
                assert pair[0] == pair[1], (name, key, pair)
            else:
                assert pair == want, (name, key, pair)


def test_workspace_bytes_extras_are_keyword_only() -> None:
    cake_backend = _backend()
    for fn in (
        inspect.unwrap(api.cake_gqa_det_train_workspace_bytes),
        cake_backend.cake_gqa_det_train_workspace_bytes,
    ):
        params = inspect.signature(fn).parameters
        assert list(params)[:3] == ["total_q", "num_heads_q", "num_segments"]
        assert tuple(list(params)[3:]) == WORKSPACE_BYTES_EXTRAS
        for name in WORKSPACE_BYTES_EXTRAS:
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    with pytest.raises(TypeError):
        api.cake_gqa_det_train_workspace_bytes(377, 24, 2, 1000)
    with pytest.raises(TypeError):
        api.cake_gqa_det_train_workspace(None, 1 << 20)


def test_the_facade_forwards_every_argument(monkeypatch) -> None:
    """Each entry point hands every positional argument and keyword to the backend object of the same name as the
    same objects (sentinels here; the backend is replaced by a recorder); the workspace helpers forward
    ``budget_bytes`` when given and leave the backend default otherwise."""
    calls: dict = {}

    def recorder(name):
        def record(*args, **kwargs):
            calls[name] = (args, kwargs)
            return name

        return record

    stub = types.SimpleNamespace(
        **{
            name: recorder(name)
            for name in (*BACKEND_NAMES.values(), "CakeGQADetWorkspace")
        }
    )
    monkeypatch.setattr(api, "_pkg", lambda: stub)
    for facade, backend in BACKEND_NAMES.items():
        params = inspect.signature(inspect.unwrap(getattr(api, facade))).parameters
        positional = [
            object() for p in params.values() if p.kind is p.POSITIONAL_OR_KEYWORD
        ]
        keywords = {
            name: object() for name, p in params.items() if p.kind is p.KEYWORD_ONLY
        }
        assert getattr(api, facade)(*positional, **keywords) == backend
        args, kwargs = calls.pop(backend)
        assert len(args) + len(kwargs) == len(params), facade
        assert all(a is b for a, b in zip(args, positional, strict=True)), facade
        assert set(kwargs) == set(keywords), facade
        assert all(kwargs[name] is keywords[name] for name in keywords), facade
    device, budget = object(), object()
    assert (
        api.cake_gqa_det_train_workspace(device, budget_bytes=budget)
        == "CakeGQADetWorkspace"
    )
    args, kwargs = calls.pop("CakeGQADetWorkspace")
    assert len(args) == 1 and args[0] is device
    assert set(kwargs) == {"budget_bytes"} and kwargs["budget_bytes"] is budget
    api.cake_gqa_det_train_workspace(device)
    assert calls.pop("CakeGQADetWorkspace") == ((device,), {})
    api.cake_gqa_det_train_workspace_bytes(377, 24, 2)
    args, kwargs = calls.pop("cake_gqa_det_train_workspace_bytes")
    assert args == (377, 24, 2) and "budget_bytes" not in kwargs


def test_the_workspace_budget_reaches_the_backend() -> None:
    """Through the real backend, host arithmetic only: the facade's workspace carries the budget it is given (the
    16 GiB default otherwise) and ``cake_gqa_det_train_workspace_bytes`` equals the backend's for every keyword."""
    cake_backend = _backend()
    from flashinfer.experimental.cake.cake_gqa_det_train import cake_jit

    ws = api.cake_gqa_det_train_workspace("cuda:0", budget_bytes=3 << 20)
    assert isinstance(ws, api.CakeGQADetWorkspace) and ws.budget_bytes == 3 << 20
    assert ws.nbytes == 0
    default = api.cake_gqa_det_train_workspace("cuda:0")
    assert default.budget_bytes == cake_jit.DEFAULT_WORKSPACE_BUDGET
    with pytest.raises(ValueError, match="budget_bytes"):
        api.cake_gqa_det_train_workspace("cuda:0", budget_bytes=0)
    for kwargs in (
        {},
        dict(total_k=52310, num_heads_kv=4),
        dict(total_k=52310, num_heads_kv=4, max_seqlen_k=9000, row_map=True),
        dict(total_k=52310, num_heads_kv=4, max_seqlen_k=9000, budget_bytes=1 << 26),
        dict(total_k=52310, num_heads_kv=4, budget_bytes=1),
    ):
        assert api.cake_gqa_det_train_workspace_bytes(
            30193, 24, 12, **kwargs
        ) == cake_backend.cake_gqa_det_train_workspace_bytes(30193, 24, 12, **kwargs), (
            kwargs
        )
    assert api.cake_gqa_det_train_workspace_bytes(
        30193, 24, 12, total_k=52310, num_heads_kv=4, budget_bytes=1
    ) < api.cake_gqa_det_train_workspace_bytes(
        30193, 24, 12, total_k=52310, num_heads_kv=4
    )


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) not in (
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        pytest.skip("the Cake GQA training attention ships for SM100 / SM103 / SM107")
    from flashinfer.experimental.cake.cake_gqa_det_train import cake_jit

    if not (Path(cake_jit.__file__).parent / "build.json").is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    return torch.device("cuda:0")


def _inputs(q_lens, k_lens, device, h_q=24, h_kv=4):
    g = torch.Generator(device=device).manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=device).to(torch.bfloat16)

    cu = lambda lens: torch.tensor(  # noqa: E731
        [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    t_q, t_k = sum(q_lens), sum(k_lens)
    return (
        randn(t_q, h_q, D),
        randn(t_k, h_kv, D),
        randn(t_k, h_kv, D),
        randn(t_q, h_q, D),
        cu(q_lens),
        cu(k_lens),
    )


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype])


def test_entry_points_return_the_backend_results() -> None:
    device = _device_or_skip()
    cake_backend = _backend()
    q, k, v, dout, cu_q, cu_k = _inputs((300, 77), (1000, 0), device)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        out, lse = api.cake_gqa_det_train_forward(
            q, k, v, cu_q, cu_k, softmax_scale=SCALE
        )
        ref_out, ref_lse = cake_backend.cake_gqa_det_train_forward(
            q, k, v, cu_q, cu_k, softmax_scale=SCALE
        )
        assert torch.equal(_bits(out), _bits(ref_out))
        assert torch.equal(_bits(lse), _bits(ref_lse))
        assert bool((lse[:, 300:] == -math.inf).all())
        kwargs = dict(softmax_scale=SCALE, dkv_dtype=torch.float32, max_seqlen_k=1000)
        dq, dk, dv = api.cake_gqa_det_train_backward(
            q, k, v, out, lse, dout, cu_q, cu_k, **kwargs
        )
        rdq, rdk, rdv = cake_backend.cake_gqa_det_train_backward(
            q, k, v, out, lse, dout, cu_q, cu_k, **kwargs
        )
        assert dk.dtype == torch.float32 and dq.dtype == torch.bfloat16
        assert dq.shape == q.shape
        for got, want in ((dq, rdq), (dk, rdk), (dv, rdv)):
            assert torch.equal(_bits(got), _bits(want))
        ws = api.cake_gqa_det_train_workspace(device, budget_bytes=1 << 22)
        assert isinstance(ws, api.CakeGQADetWorkspace) and ws.nbytes == 0
        assert ws.budget_bytes == 1 << 22
        # A small budget chunks the backward: dK / dV keep their bits.
        _cdq, cdk, cdv = api.cake_gqa_det_train_backward(
            q, k, v, out, lse, dout, cu_q, cu_k, workspace=ws, **kwargs
        )
        assert torch.equal(_bits(cdk), _bits(rdk)) and torch.equal(
            _bits(cdv), _bits(rdv)
        )
        api.cake_gqa_det(q, k, v, cu_q, cu_k, workspace=ws)
        assert api.cake_gqa_det_train_workspace_bytes(
            377, 24, 2
        ) == cake_backend.cake_gqa_det_train_workspace_bytes(377, 24, 2)


@pytest.mark.parametrize("form", ("none", "window", "rowmap"))
def test_autograd_dkv_forms(form) -> None:
    device = _device_or_skip()
    q, k, v, dout, cu_q, cu_k = _inputs((257, 129), (513, 129), device)
    # Window rows 0..512 to destination rows 0..512, then 513..641 again to rows 0..128 (repeated rows).
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
        out = api.cake_gqa_det(
            *leaves,
            cu_q,
            cu_k,
            softmax_scale=SCALE,
            dkv_acc=None if form == "none" else dkv_acc,
        )
        out.backward(dout)
        fwd = api.cake_gqa_det_train_forward(q, k, v, cu_q, cu_k, softmax_scale=SCALE)
        _, rdk, rdv = api.cake_gqa_det_train_backward(
            q,
            k,
            v,
            *fwd,
            dout,
            cu_q,
            cu_k,
            softmax_scale=SCALE,
            dkv_dtype=torch.float32,
        )
        _, bdk, bdv = api.cake_gqa_det_train_backward(
            q, k, v, *fwd, dout, cu_q, cu_k, softmax_scale=SCALE
        )
    torch.cuda.synchronize()
    assert leaves[0].grad is not None and leaves[0].grad.dtype == torch.bfloat16
    if form == "none":
        assert leaves[1].grad.dtype == torch.bfloat16
        assert torch.equal(_bits(leaves[1].grad), _bits(bdk))
        assert torch.equal(_bits(leaves[2].grad), _bits(bdv))
        torch.testing.assert_close(leaves[1].grad.float(), rdk, rtol=2**-8, atol=0)
        return
    assert leaves[1].grad is None and leaves[2].grad is None
    if form == "window":
        assert torch.equal(_bits(holder["k"]), _bits(rdk))
        assert torch.equal(_bits(holder["v"]), _bits(rdv))
        return
    # Ascending window order: rows 0..512 add their first window row, then rows 0..128 their second.
    for got, window in ((holder["k"], rdk), (holder["v"], rdv)):
        expected = torch.ones(513, 4, D, device=device) + window[:513]
        expected[:129] = expected[:129] + window[513:]
        assert torch.equal(_bits(got), _bits(expected))
