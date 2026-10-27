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

flashinfer public-API tests of the Cake packed KDA chunked training package
(``flashinfer.experimental.cake_kda_train``).

The backend suite ``test_cake_kda_train.py`` imports only the package.  This file covers what needs flashinfer
itself: the experimental public entry points ``flashinfer.kda_train.chunk_kda_fwd`` / ``chunk_kda_bwd`` /
``chunk_kda`` (their experimental marking, the keywords they forward and the default scale) and the workspace /
byte helpers next to them.
"""

import inspect
import warnings
from pathlib import Path

import pytest
import torch

import flashinfer.kda_train as public_api
from flashinfer.api_logging import ExperimentalWarning
from flashinfer.experimental.cake_kda_train import cake_backend, cake_jit
from tests.test_helpers.cake_kda_train_reference import make_inputs

ENTRY_POINTS = (
    public_api.chunk_kda_fwd,
    public_api.chunk_kda_bwd,
    public_api.chunk_kda,
)


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    if (
        cake_jit.arch_for_capability(tuple(torch.cuda.get_device_capability(index)))
        is None
    ):
        pytest.skip("not an exported compute capability")
    if not (Path(cake_jit.__file__).resolve().parent / "build.json").exists():
        pytest.skip("the package has no build.json yet (not exported)")
    return torch.device("cuda", index)


def test_entry_points_are_marked_experimental() -> None:
    for fn in ENTRY_POINTS:
        assert getattr(fn, "is_experimental", False) is True, fn.__name__
        assert fn.experimental_feature == public_api._FEATURE
        assert ".. warning::" in fn.__doc__


def test_default_scale_is_the_contracts() -> None:
    assert public_api.DEFAULT_SCALE == 128**-0.5 == cake_backend.SCALE


def test_entry_points_forward_every_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def recorder(name):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        return fake

    monkeypatch.setattr(cake_backend, "chunk_kda_fwd", recorder("fwd"))
    monkeypatch.setattr(cake_backend, "chunk_kda_bwd", recorder("bwd"))
    monkeypatch.setattr(cake_backend, "chunk_kda", recorder("autograd"))
    operands = tuple(object() for _ in range(6))
    dout, saved, workspace = object(), object(), object()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        assert (
            public_api.chunk_kda_fwd(
                *operands,
                save_for_backward=False,
                workspace=workspace,
                save_mode="tape",
            )
            == "fwd"
        )
        assert (
            public_api.chunk_kda_bwd(*operands, dout, saved=saved, scale=0.5) == "bwd"
        )
        assert public_api.chunk_kda(*operands, workspace=workspace) == "autograd"
    assert calls == [
        (
            "fwd",
            operands,
            {
                "scale": public_api.DEFAULT_SCALE,
                "save_for_backward": False,
                "workspace": workspace,
                "save_mode": "tape",
                "arena": None,
                "doc_lengths": None,
            },
        ),
        (
            "bwd",
            (*operands, dout),
            {
                "saved": saved,
                "scale": 0.5,
                "workspace": None,
                "arena": None,
                "doc_lengths": None,
            },
        ),
        (
            "autograd",
            operands,
            {
                "scale": public_api.DEFAULT_SCALE,
                "workspace": workspace,
                "save_mode": None,
                "arena": None,
                "doc_lengths": None,
            },
        ),
    ]


def test_arena_option_is_forwarded_and_the_factory_is_the_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arena = public_api.kda_train_arena(lean=True, tail_passes=4)
    assert arena == cake_backend.KdaArena(lean=True, tail_passes=4)
    assert public_api.kda_train_arena() == cake_backend.KdaArena()
    assert public_api.kda_train_arena(
        tail_passes=2
    ).lean  # several passes imply the lean layouts
    with pytest.raises(ValueError, match="tail_passes"):
        public_api.kda_train_arena(tail_passes=0)
    batched = public_api.kda_train_arena(batch_tokens=1000, max_docs_per_batch=2)
    assert batched == cake_backend.KdaArena(batch_tokens=1000, max_docs_per_batch=2)
    assert batched.batched and not batched.lean
    with pytest.raises(ValueError, match="batch_tokens"):
        public_api.kda_train_arena(max_docs_per_batch=1)
    calls = []

    def recorder(name):
        def fake(*args, **kwargs):
            calls.append((name, kwargs.get("arena"), kwargs.get("doc_lengths")))
            return name

        return fake

    monkeypatch.setattr(cake_backend, "chunk_kda_fwd", recorder("fwd"))
    monkeypatch.setattr(cake_backend, "chunk_kda_bwd", recorder("bwd"))
    monkeypatch.setattr(cake_backend, "chunk_kda", recorder("autograd"))
    monkeypatch.setattr(cake_backend, "workspace_bytes", recorder("bytes"))
    operands = tuple(object() for _ in range(6))
    lengths = (300, 700)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        public_api.chunk_kda_fwd(*operands, arena=arena)
        public_api.chunk_kda_bwd(*operands, object(), arena=arena)
        public_api.chunk_kda(*operands, arena=arena)
        public_api.kda_train_workspace_bytes(1, 1, 1, sm_count=148, arena=arena)
        public_api.chunk_kda_fwd(*operands, arena=batched, doc_lengths=lengths)
        public_api.chunk_kda_bwd(
            *operands, object(), arena=batched, doc_lengths=lengths
        )
        public_api.chunk_kda(*operands, arena=batched, doc_lengths=lengths)
        public_api.kda_train_workspace_bytes(
            1000, 2, 1, sm_count=148, arena=batched, doc_lengths=lengths
        )
    assert calls == [
        ("fwd", arena, None),
        ("bwd", arena, None),
        ("autograd", arena, None),
        ("bytes", arena, None),
        ("fwd", batched, lengths),
        ("bwd", batched, lengths),
        ("autograd", batched, lengths),
        ("bytes", batched, lengths),
    ]


def test_entry_points_warn_once_as_experimental(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cake_backend, "chunk_kda_fwd", lambda *a, **k: None)
    # A fresh wrapper so the once-per-function warning is observable regardless of test order.
    fresh = public_api.flashinfer_experimental_api(feature=public_api._FEATURE)(
        lambda *a, **k: cake_backend.chunk_kda_fwd(*a, **k)
    )
    with pytest.warns(ExperimentalWarning):
        fresh(None, None, None, None, None, None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        fresh(None, None, None, None, None, None)


def test_byte_queries_are_the_backends() -> None:
    # An explicit SM count keeps the queries device-free (the segmented route's segment count reads it).
    for S, N, H in ((1, 1, 1), (129, 2, 4), (261393, 4, 64)):
        assert public_api.kda_train_workspace_bytes(
            S, N, H, sm_count=148
        ) == cake_backend.workspace_bytes(S, N, H, sm_count=148)
        assert public_api.kda_train_workspace_bytes(
            S, N, H, backward=False, sm_count=148
        ) == cake_backend.workspace_bytes(S, N, H, backward=False, sm_count=148)
        assert public_api.kda_train_workspace_bytes(
            S, N, H, save_mode="tape", sm_count=148
        ) == cake_backend.workspace_bytes(S, N, H, save_mode="tape", sm_count=148)
        assert (
            public_api.kda_train_saved_bytes(S, H)
            == cake_backend.saved_bytes(S, H)
            == 256 * S * H
        )
        assert public_api.kda_train_saved_bytes(
            S, H, N, save_mode="tape", sm_count=148
        ) == cake_backend.saved_bytes(S, H, N, save_mode="tape", sm_count=148)
    # the batched option: the bound of every group without doc_lengths, the exact groups with them; refused on the
    # rows with at most four (document, head) chains
    lengths = (23496, 88510, 114974, 34413)
    S, N, H = sum(lengths), len(lengths), 4
    batched = public_api.kda_train_arena(lean=True, tail_passes=4, batch_tokens=120000)
    bound = public_api.kda_train_workspace_bytes(S, N, H, sm_count=148, arena=batched)
    exact = public_api.kda_train_workspace_bytes(
        S, N, H, sm_count=148, arena=batched, doc_lengths=lengths
    )
    assert exact == cake_backend.workspace_bytes(
        S, N, H, sm_count=148, arena=batched, doc_lengths=lengths
    )
    assert (
        exact
        <= bound
        == cake_backend.workspace_bytes(S, N, H, sm_count=148, arena=batched)
        < public_api.kda_train_workspace_bytes(S, N, H, sm_count=148)
    )
    with pytest.raises(ValueError, match="proportional"):
        public_api.kda_train_workspace_bytes(
            128, 2, 1, sm_count=148, arena=public_api.kda_train_arena(batch_tokens=64)
        )


def test_support_query_follows_the_capability_table() -> None:
    if not torch.cuda.is_available():
        assert public_api.is_kda_train_supported() is False
        return
    index = torch.cuda.current_device()
    expected = (
        cake_jit.arch_for_capability(tuple(torch.cuda.get_device_capability(index)))
        is not None
    )
    assert public_api.is_kda_train_supported() is expected
    assert public_api.is_kda_train_supported(index) is expected
    assert public_api.is_kda_train_supported(torch.device("cuda", index)) is expected


def test_signatures_keep_the_contracts_keywords() -> None:
    fwd = inspect.signature(public_api.chunk_kda_fwd).parameters
    assert list(fwd) == [
        "q",
        "k",
        "v",
        "g",
        "beta",
        "cu_seqlens",
        "scale",
        "save_for_backward",
        "workspace",
        "save_mode",
        "arena",
        "doc_lengths",
    ]
    bwd = inspect.signature(public_api.chunk_kda_bwd).parameters
    assert list(bwd) == [
        "q",
        "k",
        "v",
        "g",
        "beta",
        "cu_seqlens",
        "dout",
        "saved",
        "scale",
        "workspace",
        "arena",
        "doc_lengths",
    ]
    for name in (
        "scale",
        "save_for_backward",
        "workspace",
        "save_mode",
        "arena",
        "doc_lengths",
    ):
        assert fwd[name].kind is inspect.Parameter.KEYWORD_ONLY
    for name in ("saved", "scale", "workspace", "arena", "doc_lengths"):
        assert bwd[name].kind is inspect.Parameter.KEYWORD_ONLY
    autograd = inspect.signature(public_api.chunk_kda).parameters
    assert list(autograd)[-5:] == [
        "scale",
        "workspace",
        "save_mode",
        "arena",
        "doc_lengths",
    ]
    factory = inspect.signature(public_api.kda_train_arena).parameters
    assert list(factory) == [
        "lean",
        "tail_passes",
        "batch_tokens",
        "max_docs_per_batch",
    ] and all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in factory.values())
    ws = inspect.signature(public_api.kda_train_workspace).parameters
    assert list(ws)[-2:] == ["arena", "doc_lengths"]
    nbytes = inspect.signature(public_api.kda_train_workspace_bytes).parameters
    assert list(nbytes)[-2:] == ["arena", "doc_lengths"]


def test_public_entries_run_the_backend() -> None:
    device = _device_or_skip()
    lengths, heads = (63, 1, 65), 2
    inputs = make_inputs(lengths, heads, device, batched=True)
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        ws = public_api.kda_train_workspace(sum(lengths), len(lengths), heads, device)
        assert isinstance(
            ws, cake_backend.KdaWorkspace
        ) and ws.bytes == public_api.kda_train_workspace_bytes(
            sum(lengths), len(lengths), heads
        )
        out, saved = public_api.chunk_kda_fwd(q, k, v, g, beta, cu, workspace=ws)
        assert out.shape == v.shape and isinstance(saved, cake_backend.KdaChunkSaved)
        grads = public_api.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=ws
        )
        grads = [t.clone() for t in grads]
        ref_out, ref_saved = cake_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu, scale=cake_backend.SCALE
        )
        ref_grads = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=ref_saved, scale=cake_backend.SCALE
        )
        assert torch.equal(out, ref_out)
        for mine, theirs in zip(grads, ref_grads, strict=True):
            assert torch.equal(mine, theirs)
        leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)]
        o = public_api.chunk_kda(*leaves, cu)
        o.backward(dout)
        for leaf, expected in zip(
            leaves, (grads[0], grads[1], grads[2], grads[4], grads[3]), strict=True
        ):
            assert torch.equal(leaf.grad, expected)


def test_public_arena_option_runs_the_backend() -> None:
    device = _device_or_skip()
    lengths, heads = (63, 1, 65), 2
    S, N = sum(lengths), len(lengths)
    inputs = make_inputs(lengths, heads, device, batched=True)
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )
    ref_out, ref_saved = cake_backend.chunk_kda_fwd(
        q, k, v, g, beta, cu, scale=cake_backend.SCALE
    )
    ref_grads = cake_backend.chunk_kda_bwd(
        q, k, v, g, beta, cu, dout, saved=ref_saved, scale=cake_backend.SCALE
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        arena = public_api.kda_train_arena(tail_passes=2)
        ws = public_api.kda_train_workspace(S, N, heads, device, arena=arena)
        assert ws.arena_option == arena
        assert ws.bytes == public_api.kda_train_workspace_bytes(
            S, N, heads, arena=arena
        )
        assert ws.bytes < public_api.kda_train_workspace_bytes(S, N, heads)
        out, saved = public_api.chunk_kda_fwd(q, k, v, g, beta, cu, workspace=ws)
        assert torch.equal(out, ref_out)
        grads = public_api.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=ws
        )
        for mine, theirs in zip(grads, ref_grads, strict=True):
            assert torch.equal(mine, theirs)
        grads = public_api.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved, arena=arena
        )
        for mine, theirs in zip(grads, ref_grads, strict=True):
            assert torch.equal(mine, theirs)
        with pytest.raises(ValueError, match="lean=False"):
            public_api.chunk_kda_bwd(q, k, v, g, beta, cu, dout, workspace=ws)
        leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)]
        o = public_api.chunk_kda(*leaves, cu, arena=arena)
        o.backward(dout)
        for leaf, expected in zip(
            leaves,
            (ref_grads[0], ref_grads[1], ref_grads[2], ref_grads[4], ref_grads[3]),
            strict=True,
        ):
            assert torch.equal(leaf.grad, expected)
        # the batched option through the public entries: two document groups, the results bit for bit
        batched = public_api.kda_train_arena(lean=True, tail_passes=2, batch_tokens=65)
        ws_b = public_api.kda_train_workspace(
            S, N, heads, device, arena=batched, doc_lengths=lengths
        )
        assert ws_b.bytes == public_api.kda_train_workspace_bytes(
            S, N, heads, arena=batched, doc_lengths=lengths
        )
        out_b, saved_b = public_api.chunk_kda_fwd(
            q, k, v, g, beta, cu, workspace=ws_b, doc_lengths=lengths
        )
        assert torch.equal(out_b.view(torch.int16), ref_out.view(torch.int16))
        grads_b = public_api.chunk_kda_bwd(
            q,
            k,
            v,
            g,
            beta,
            cu,
            dout,
            saved=saved_b,
            workspace=ws_b,
            doc_lengths=lengths,
        )
        for mine, theirs in zip(grads_b, ref_grads, strict=True):
            assert torch.equal(
                mine.view(torch.int16 if mine.element_size() == 2 else torch.int32),
                theirs.view(torch.int16 if theirs.element_size() == 2 else torch.int32),
            )
        with pytest.raises(ValueError, match="doc_lengths"):
            public_api.chunk_kda_fwd(q, k, v, g, beta, cu, workspace=ws_b)
        leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)]
        o = public_api.chunk_kda(*leaves, cu, arena=batched, doc_lengths=lengths)
        o.backward(dout)
        for leaf, expected in zip(
            leaves,
            (ref_grads[0], ref_grads[1], ref_grads[2], ref_grads[4], ref_grads[3]),
            strict=True,
        ):
            assert torch.equal(leaf.grad, expected)
