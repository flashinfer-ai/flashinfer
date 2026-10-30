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
(``flashinfer.experimental.cake_kda_det_train``).

The backend suite ``test_cake_kda_det_train.py`` imports only the package.  This file covers what needs flashinfer
itself: the experimental public entry points ``flashinfer.cake_kda_det_train.chunk_kda_det_fwd`` /
``chunk_kda_det_bwd`` / ``chunk_kda_det`` (their experimental marking, the keywords they forward and the default
scale), the workspace / byte helpers next to them, and the KCP entry points of ``flashinfer.kda_train``.
"""

import inspect
import warnings
from pathlib import Path

import pytest
import torch

import flashinfer.cake_kda_det_train as det_api
import flashinfer.kda_train as public_api
from flashinfer.api_logging import ExperimentalWarning
from flashinfer.experimental.cake_kda_det_train import cake_backend, cake_jit
from tests.test_helpers.cake_kda_det_train_reference import make_inputs

ENTRY_POINTS = (
    det_api.chunk_kda_det_fwd,
    det_api.chunk_kda_det_bwd,
    det_api.chunk_kda_det,
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
        assert fn.experimental_feature == det_api._FEATURE
        assert ".. warning::" in fn.__doc__


def test_default_scale_is_the_contracts() -> None:
    assert det_api.DEFAULT_SCALE == 128**-0.5 == cake_backend.SCALE


def test_entry_points_forward_every_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def recorder(name):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        return fake

    monkeypatch.setattr(cake_backend, "chunk_kda_det_fwd", recorder("fwd"))
    monkeypatch.setattr(cake_backend, "chunk_kda_det_bwd", recorder("bwd"))
    monkeypatch.setattr(cake_backend, "chunk_kda_det", recorder("autograd"))
    operands = tuple(object() for _ in range(6))
    dout, saved, workspace = object(), object(), object()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        assert (
            det_api.chunk_kda_det_fwd(
                *operands,
                save_for_backward=False,
                workspace=workspace,
                save_mode="tape",
            )
            == "fwd"
        )
        assert (
            det_api.chunk_kda_det_bwd(*operands, dout, saved=saved, scale=0.5) == "bwd"
        )
        assert det_api.chunk_kda_det(*operands, workspace=workspace) == "autograd"
    assert calls == [
        (
            "fwd",
            operands,
            {
                "scale": det_api.DEFAULT_SCALE,
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
                "scale": det_api.DEFAULT_SCALE,
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
    arena = det_api.kda_det_train_arena(lean=True, tail_passes=4)
    assert arena == cake_backend.KdaDetArena(lean=True, tail_passes=4)
    assert det_api.kda_det_train_arena() == cake_backend.KdaDetArena()
    assert det_api.kda_det_train_arena(
        tail_passes=2
    ).lean  # several passes imply the lean layouts
    with pytest.raises(ValueError, match="tail_passes"):
        det_api.kda_det_train_arena(tail_passes=0)
    batched = det_api.kda_det_train_arena(batch_tokens=1000, max_docs_per_batch=2)
    assert batched == cake_backend.KdaDetArena(batch_tokens=1000, max_docs_per_batch=2)
    assert batched.batched and not batched.lean
    with pytest.raises(ValueError, match="batch_tokens"):
        det_api.kda_det_train_arena(max_docs_per_batch=1)
    calls = []

    def recorder(name):
        def fake(*args, **kwargs):
            calls.append((name, kwargs.get("arena"), kwargs.get("doc_lengths")))
            return name

        return fake

    monkeypatch.setattr(cake_backend, "chunk_kda_det_fwd", recorder("fwd"))
    monkeypatch.setattr(cake_backend, "chunk_kda_det_bwd", recorder("bwd"))
    monkeypatch.setattr(cake_backend, "chunk_kda_det", recorder("autograd"))
    monkeypatch.setattr(cake_backend, "workspace_bytes", recorder("bytes"))
    operands = tuple(object() for _ in range(6))
    lengths = (300, 700)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        det_api.chunk_kda_det_fwd(*operands, arena=arena)
        det_api.chunk_kda_det_bwd(*operands, object(), arena=arena)
        det_api.chunk_kda_det(*operands, arena=arena)
        det_api.kda_det_train_workspace_bytes(1, 1, 1, sm_count=148, arena=arena)
        det_api.chunk_kda_det_fwd(*operands, arena=batched, doc_lengths=lengths)
        det_api.chunk_kda_det_bwd(
            *operands, object(), arena=batched, doc_lengths=lengths
        )
        det_api.chunk_kda_det(*operands, arena=batched, doc_lengths=lengths)
        det_api.kda_det_train_workspace_bytes(
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
    monkeypatch.setattr(cake_backend, "chunk_kda_det_fwd", lambda *a, **k: None)
    # A fresh wrapper so the once-per-function warning is observable regardless of test order.
    fresh = det_api.flashinfer_experimental_api(feature=det_api._FEATURE)(
        lambda *a, **k: cake_backend.chunk_kda_det_fwd(*a, **k)
    )
    with pytest.warns(ExperimentalWarning):
        fresh(None, None, None, None, None, None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        fresh(None, None, None, None, None, None)


def test_byte_queries_are_the_backends() -> None:
    # An explicit SM count keeps the queries device-free (the segmented route's segment count reads it).
    for S, N, H in ((1, 1, 1), (129, 2, 4), (261393, 4, 64)):
        assert det_api.kda_det_train_workspace_bytes(
            S, N, H, sm_count=148
        ) == cake_backend.workspace_bytes(S, N, H, sm_count=148)
        assert det_api.kda_det_train_workspace_bytes(
            S, N, H, backward=False, sm_count=148
        ) == cake_backend.workspace_bytes(S, N, H, backward=False, sm_count=148)
        assert det_api.kda_det_train_workspace_bytes(
            S, N, H, save_mode="tape", sm_count=148
        ) == cake_backend.workspace_bytes(S, N, H, save_mode="tape", sm_count=148)
        assert (
            det_api.kda_det_train_saved_bytes(S, H)
            == cake_backend.saved_bytes(S, H)
            == 256 * S * H
        )
        assert det_api.kda_det_train_saved_bytes(
            S, H, N, save_mode="tape", sm_count=148
        ) == cake_backend.saved_bytes(S, H, N, save_mode="tape", sm_count=148)
    # the batched option: the bound of every group without doc_lengths, the exact groups with them; refused on the
    # rows with at most four (document, head) chains
    lengths = (23496, 88510, 114974, 34413)
    S, N, H = sum(lengths), len(lengths), 4
    batched = det_api.kda_det_train_arena(lean=True, tail_passes=4, batch_tokens=120000)
    bound = det_api.kda_det_train_workspace_bytes(S, N, H, sm_count=148, arena=batched)
    exact = det_api.kda_det_train_workspace_bytes(
        S, N, H, sm_count=148, arena=batched, doc_lengths=lengths
    )
    assert exact == cake_backend.workspace_bytes(
        S, N, H, sm_count=148, arena=batched, doc_lengths=lengths
    )
    assert (
        exact
        <= bound
        == cake_backend.workspace_bytes(S, N, H, sm_count=148, arena=batched)
        < det_api.kda_det_train_workspace_bytes(S, N, H, sm_count=148)
    )
    with pytest.raises(ValueError, match="proportional"):
        det_api.kda_det_train_workspace_bytes(
            128, 2, 1, sm_count=148, arena=det_api.kda_det_train_arena(batch_tokens=64)
        )


def test_kcp_workspace_query_resolves_save_transfer_like_the_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An unset save_transfer resolves to the plan default, as the a2a forward does; an explicit value passes through.
    from flashinfer.experimental.cake_kda_train import cake_kcp_plan

    calls = []
    monkeypatch.setattr(
        cake_kcp_plan, "kcp_workspace_bytes", lambda ctx, **kw: calls.append(kw) or 0
    )
    ctx = object()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        public_api.kcp_workspace_bytes(ctx, backward=False)
        public_api.kcp_workspace_bytes(ctx, backward=False, save_transfer=False)
        public_api.kcp_workspace_bytes(ctx)
    assert [kw["save_transfer"] for kw in calls] == [
        cake_kcp_plan.SAVE_TRANSFER_DEFAULT,
        False,
        cake_kcp_plan.SAVE_TRANSFER_DEFAULT,
    ]
    assert cake_kcp_plan.SAVE_TRANSFER_DEFAULT is True
    assert (
        inspect.signature(public_api.kcp_workspace_bytes)
        .parameters["save_transfer"]
        .default
        is None
    )


def test_support_query_follows_the_capability_table() -> None:
    if not torch.cuda.is_available():
        assert det_api.is_kda_det_train_supported() is False
        return
    index = torch.cuda.current_device()
    expected = (
        cake_jit.arch_for_capability(tuple(torch.cuda.get_device_capability(index)))
        is not None
    )
    assert det_api.is_kda_det_train_supported() is expected
    assert det_api.is_kda_det_train_supported(index) is expected
    assert det_api.is_kda_det_train_supported(torch.device("cuda", index)) is expected


def test_signatures_keep_the_contracts_keywords() -> None:
    fwd = inspect.signature(det_api.chunk_kda_det_fwd).parameters
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
    bwd = inspect.signature(det_api.chunk_kda_det_bwd).parameters
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
    autograd = inspect.signature(det_api.chunk_kda_det).parameters
    assert list(autograd)[-5:] == [
        "scale",
        "workspace",
        "save_mode",
        "arena",
        "doc_lengths",
    ]
    factory = inspect.signature(det_api.kda_det_train_arena).parameters
    assert list(factory) == [
        "lean",
        "tail_passes",
        "batch_tokens",
        "max_docs_per_batch",
    ] and all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in factory.values())
    ws = inspect.signature(det_api.kda_det_train_workspace).parameters
    assert list(ws)[-2:] == ["arena", "doc_lengths"]
    nbytes = inspect.signature(det_api.kda_det_train_workspace_bytes).parameters
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
        ws = det_api.kda_det_train_workspace(sum(lengths), len(lengths), heads, device)
        assert isinstance(
            ws, cake_backend.KdaDetWorkspace
        ) and ws.bytes == det_api.kda_det_train_workspace_bytes(
            sum(lengths), len(lengths), heads
        )
        out, saved = det_api.chunk_kda_det_fwd(q, k, v, g, beta, cu, workspace=ws)
        assert out.shape == v.shape and isinstance(saved, cake_backend.KdaDetChunkSaved)
        grads = det_api.chunk_kda_det_bwd(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=ws
        )
        grads = [t.clone() for t in grads]
        ref_out, ref_saved = cake_backend.chunk_kda_det_fwd(
            q, k, v, g, beta, cu, scale=cake_backend.SCALE
        )
        ref_grads = cake_backend.chunk_kda_det_bwd(
            q, k, v, g, beta, cu, dout, saved=ref_saved, scale=cake_backend.SCALE
        )
        assert torch.equal(out, ref_out)
        for mine, theirs in zip(grads, ref_grads, strict=True):
            assert torch.equal(mine, theirs)
        leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)]
        o = det_api.chunk_kda_det(*leaves, cu)
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
    ref_out, ref_saved = cake_backend.chunk_kda_det_fwd(
        q, k, v, g, beta, cu, scale=cake_backend.SCALE
    )
    ref_grads = cake_backend.chunk_kda_det_bwd(
        q, k, v, g, beta, cu, dout, saved=ref_saved, scale=cake_backend.SCALE
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        arena = det_api.kda_det_train_arena(tail_passes=2)
        ws = det_api.kda_det_train_workspace(S, N, heads, device, arena=arena)
        assert ws.arena_option == arena
        assert ws.bytes == det_api.kda_det_train_workspace_bytes(
            S, N, heads, arena=arena
        )
        assert ws.bytes < det_api.kda_det_train_workspace_bytes(S, N, heads)
        out, saved = det_api.chunk_kda_det_fwd(q, k, v, g, beta, cu, workspace=ws)
        assert torch.equal(out, ref_out)
        grads = det_api.chunk_kda_det_bwd(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=ws
        )
        for mine, theirs in zip(grads, ref_grads, strict=True):
            assert torch.equal(mine, theirs)
        grads = det_api.chunk_kda_det_bwd(
            q, k, v, g, beta, cu, dout, saved=saved, arena=arena
        )
        for mine, theirs in zip(grads, ref_grads, strict=True):
            assert torch.equal(mine, theirs)
        with pytest.raises(ValueError, match="lean=False"):
            det_api.chunk_kda_det_bwd(q, k, v, g, beta, cu, dout, workspace=ws)
        leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)]
        o = det_api.chunk_kda_det(*leaves, cu, arena=arena)
        o.backward(dout)
        for leaf, expected in zip(
            leaves,
            (ref_grads[0], ref_grads[1], ref_grads[2], ref_grads[4], ref_grads[3]),
            strict=True,
        ):
            assert torch.equal(leaf.grad, expected)
        # the batched option through the public entries: two document groups, the results bit for bit
        batched = det_api.kda_det_train_arena(lean=True, tail_passes=2, batch_tokens=65)
        ws_b = det_api.kda_det_train_workspace(
            S, N, heads, device, arena=batched, doc_lengths=lengths
        )
        assert ws_b.bytes == det_api.kda_det_train_workspace_bytes(
            S, N, heads, arena=batched, doc_lengths=lengths
        )
        out_b, saved_b = det_api.chunk_kda_det_fwd(
            q, k, v, g, beta, cu, workspace=ws_b, doc_lengths=lengths
        )
        assert torch.equal(out_b.view(torch.int16), ref_out.view(torch.int16))
        grads_b = det_api.chunk_kda_det_bwd(
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
            det_api.chunk_kda_det_fwd(q, k, v, g, beta, cu, workspace=ws_b)
        leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)]
        o = det_api.chunk_kda_det(*leaves, cu, arena=batched, doc_lengths=lengths)
        o.backward(dout)
        for leaf, expected in zip(
            leaves,
            (ref_grads[0], ref_grads[1], ref_grads[2], ref_grads[4], ref_grads[3]),
            strict=True,
        ):
            assert torch.equal(leaf.grad, expected)


def test_kcp_recompute_handle_keywords_are_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The KCP checkpoint recompute handle on the public module (issue #7815): ``chunk_kda_kcp`` takes
    ``recompute_handle`` / ``checkpoint_keep`` keyword-only and forwards both (``None`` by default);
    ``kcp_saved_bytes`` / ``kcp_comm_bytes`` forward ``checkpoint_keep`` (and ``save_transfer=None`` under a tier,
    ``False`` without one); ``KcpRecomputeHandle`` resolves lazily to
    the KCP module's class.  Device-free (recorders)."""
    from flashinfer.experimental.cake_kda_train import cake_kcp, cake_kcp_plan

    calls = []

    def recorder(name):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        return fake

    monkeypatch.setattr(cake_kcp, "chunk_kda_kcp", recorder("autograd"))
    monkeypatch.setattr(cake_kcp_plan, "kcp_saved_bytes", recorder("saved"))
    monkeypatch.setattr(cake_kcp_plan, "kcp_comm_bytes", recorder("comm"))
    operands = tuple(object() for _ in range(6))
    handle, ctx = object(), object()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        assert (
            public_api.chunk_kda_kcp(
                *operands, recompute_handle=handle, checkpoint_keep="transfer"
            )
            == "autograd"
        )
        assert public_api.chunk_kda_kcp(*operands, exchange="ring") == "autograd"
        assert public_api.kcp_saved_bytes(ctx, checkpoint_keep="states") == "saved"
        assert public_api.kcp_comm_bytes(ctx, H=4, checkpoint_keep="states") == "comm"
        assert public_api.kcp_comm_bytes(ctx, H=4) == "comm"
    base = {
        "scale": None,
        "save_mode": "light",
        "boundary_mode": "saved",
        "workspace": None,
    }
    assert (
        calls
        == [
            (
                "autograd",
                operands,
                {**base, "recompute_handle": handle, "checkpoint_keep": "transfer"},
            ),
            (
                "autograd",
                operands,
                {
                    **base,
                    "recompute_handle": None,
                    "checkpoint_keep": None,
                    "exchange": "ring",
                },
            ),
            (
                "saved",
                (ctx,),
                {
                    "H": 64,
                    "save_mode": "light",
                    "save_transfer": None,
                    "with_conv": True,
                    "C": 8192,
                    "checkpoint_keep": "states",
                },
            ),
            (
                "comm",
                (ctx,),
                {
                    "H": 4,
                    "C": 8192,
                    "save_transfer": None,  # the tier fixes it (an explicit contradicting value raises in the query)
                    "boundary_mode": "saved",
                    "checkpoint_keep": "states",
                },
            ),
            (
                "comm",
                (ctx,),
                {
                    "H": 4,
                    "C": 8192,
                    "save_transfer": False,  # no tier: the public default, unchanged
                    "boundary_mode": "saved",
                    "checkpoint_keep": None,
                },
            ),
        ]
    )
    params = inspect.signature(public_api.chunk_kda_kcp).parameters
    for name in ("recompute_handle", "checkpoint_keep"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[name].default is None
    for fn in (public_api.kcp_saved_bytes, public_api.kcp_comm_bytes):
        p = inspect.signature(fn).parameters["checkpoint_keep"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None
    assert public_api.KcpRecomputeHandle is cake_kcp.KcpRecomputeHandle
    assert "KcpRecomputeHandle" in public_api.__all__
    with pytest.raises(AttributeError):
        _ = public_api.NoSuchName  # type: ignore[attr-defined]
