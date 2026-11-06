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

The deterministic single-rank KDA training entries (``flashinfer.cake_kda_det_train``) refuse a workspace, a saved
set or an arena option of the non-deterministic package (``flashinfer.cake_kda_train`` with
``deterministic=False``) with ``ValueError`` naming the mode.  The non-deterministic types carry the attributes
the deterministic checks read, so only their type tells them apart.  The CPU tests use stand-ins and the
non-deterministic types built on the host; the GPU tests use a workspace and a saved set the non-deterministic
package allocates and writes.
"""

import importlib
import re
import warnings
from pathlib import Path

import pytest
import torch

import flashinfer.cake_kda_det_train as det_api
from flashinfer.api_logging import ExperimentalWarning
from flashinfer.experimental.cake_kda_det_train import cake_backend, cake_jit
from tests.test_helpers.cake_kda_det_train_reference import make_inputs

ND_BACKEND = "flashinfer.experimental.cake.cake_kda_train.cake_backend"

LENGTHS, HEADS = (63, 1, 65), 2


def _refused(name: str, value: object, det_type: str) -> str:
    """The message pattern of a refused ``name`` argument ``value``."""
    return (
        re.escape(
            f"{name}: a {type(value).__name__} from {type(value).__module__} is not a deterministic-package "
        )
        + ".*"
        + re.escape(f"; the deterministic entries take a {det_type} (")
        + ".*"
        + re.escape(
            "; the non-deterministic package is flashinfer.cake_kda_train / flashinfer.cake_kda_train_kcp "
            "(deterministic=False)"
        )
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


def _operands(device: torch.device | str) -> tuple[torch.Tensor, ...]:
    inputs = make_inputs(LENGTHS, HEADS, device, batched=True)
    return tuple(inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout"))


class _Foreign:
    """A stand-in of another package's workspace / saved set / arena option."""


def _nd_saved_on_host() -> object:
    nd = importlib.import_module(ND_BACKEND)
    S = sum(LENGTHS)
    a = torch.zeros(S, HEADS, 64, dtype=torch.bfloat16)
    return nd.KdaChunkSaved(
        version=cake_backend.SAVED_VERSION,
        Aqk=a,
        Akk=a.clone(),
        tables=None,
        S=S,
        H=HEADS,
        num_docs=len(LENGTHS),
        scale=cake_backend.SCALE,
    )


@pytest.mark.parametrize(
    "entry,name",
    [
        ("fwd", "workspace"),
        ("fwd", "arena"),
        ("bwd", "workspace"),
        ("bwd", "arena"),
        ("bwd", "saved"),
        ("autograd", "workspace"),
        ("autograd", "arena"),
    ],
)
def test_entries_refuse_a_foreign_object_on_cpu(entry: str, name: str) -> None:
    q, k, v, g, beta, cu, dout = _operands("cpu")
    foreign = _Foreign()
    det_type = cake_backend._DET_KINDS[name][0].__name__
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        with pytest.raises(ValueError, match=_refused(name, foreign, det_type)):
            if entry == "fwd":
                det_api.chunk_kda_det_fwd(q, k, v, g, beta, cu, **{name: foreign})
            elif entry == "bwd":
                det_api.chunk_kda_det_bwd(q, k, v, g, beta, cu, dout, **{name: foreign})
            else:
                det_api.chunk_kda_det(q, k, v, g, beta, cu, **{name: foreign})


@pytest.mark.parametrize("name", ["workspace", "arena"])
def test_workspace_helper_refuses_a_foreign_object(name: str) -> None:
    foreign = _Foreign()
    det_type = cake_backend._DET_KINDS[name][0].__name__
    with pytest.raises(ValueError, match=_refused(name, foreign, det_type)):
        kwargs = {"arena": foreign} if name == "arena" else {}
        with cake_backend._workspace(
            foreign if name == "workspace" else None,
            1,
            1,
            1,
            torch.device("cpu"),
            backward=True,
            **kwargs,
        ):
            pass


def test_entries_refuse_nd_host_objects() -> None:
    nd = importlib.import_module(ND_BACKEND)
    q, k, v, g, beta, cu, dout = _operands("cpu")
    nd_arena, nd_saved = nd.KdaArena(), _nd_saved_on_host()
    assert type(nd_saved).__module__ == ND_BACKEND
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        with pytest.raises(
            ValueError, match=_refused("saved", nd_saved, "KdaDetChunkSaved")
        ):
            det_api.chunk_kda_det_bwd(q, k, v, g, beta, cu, dout, saved=nd_saved)
        for call in (
            lambda: det_api.chunk_kda_det_fwd(q, k, v, g, beta, cu, arena=nd_arena),
            lambda: det_api.chunk_kda_det_bwd(
                q, k, v, g, beta, cu, dout, arena=nd_arena
            ),
            lambda: det_api.chunk_kda_det(q, k, v, g, beta, cu, arena=nd_arena),
        ):
            with pytest.raises(
                ValueError, match=_refused("arena", nd_arena, "KdaDetArena")
            ):
                call()


def test_det_objects_pass_the_package_check() -> None:
    S = sum(LENGTHS)
    a = torch.zeros(S, HEADS, 64, dtype=torch.bfloat16)
    saved = cake_backend.KdaDetChunkSaved(
        version=cake_backend.SAVED_VERSION,
        Aqk=a,
        Akk=a.clone(),
        tables=None,
        S=S,
        H=HEADS,
        num_docs=len(LENGTHS),
        scale=cake_backend.SCALE,
    )
    cake_backend._check_det_package(
        workspace=None, saved=saved, arena=det_api.kda_det_train_arena(lean=True)
    )
    cake_backend._check_det_package(workspace=None, saved=None, arena=None)


def test_det_entries_refuse_an_nd_workspace_and_saved_set() -> None:
    device = _device_or_skip()
    nd_api = importlib.import_module("flashinfer.cake_kda_train")
    q, k, v, g, beta, cu, dout = _operands(device)
    S, N = sum(LENGTHS), len(LENGTHS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        nd_ws = nd_api.kda_train_workspace(S, N, HEADS, device, deterministic=False)
        assert type(nd_ws).__module__ == ND_BACKEND
        _nd_out, nd_saved = nd_api.chunk_kda_fwd(
            q, k, v, g, beta, cu, workspace=nd_ws, deterministic=False
        )
        assert type(nd_saved).__module__ == ND_BACKEND
        _out, saved = det_api.chunk_kda_det_fwd(q, k, v, g, beta, cu)
        ws_refused = _refused("workspace", nd_ws, "KdaDetWorkspace")
        with pytest.raises(ValueError, match=ws_refused):
            det_api.chunk_kda_det_fwd(q, k, v, g, beta, cu, workspace=nd_ws)
        with pytest.raises(ValueError, match=ws_refused):
            det_api.chunk_kda_det_bwd(
                q, k, v, g, beta, cu, dout, saved=saved, workspace=nd_ws
            )
        with pytest.raises(ValueError, match=ws_refused):
            det_api.chunk_kda_det(q, k, v, g, beta, cu, workspace=nd_ws)
        with pytest.raises(
            ValueError, match=_refused("saved", nd_saved, "KdaDetChunkSaved")
        ):
            det_api.chunk_kda_det_bwd(q, k, v, g, beta, cu, dout, saved=nd_saved)


def test_det_workspace_and_arena_still_run() -> None:
    device = _device_or_skip()
    q, k, v, g, beta, cu, dout = _operands(device)
    S, N = sum(LENGTHS), len(LENGTHS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        ref_out, ref_saved = det_api.chunk_kda_det_fwd(q, k, v, g, beta, cu)
        ref_grads = [
            t.clone()
            for t in det_api.chunk_kda_det_bwd(
                q, k, v, g, beta, cu, dout, saved=ref_saved
            )
        ]
        arena = det_api.kda_det_train_arena()
        ws = det_api.kda_det_train_workspace(S, N, HEADS, device, arena=arena)
        assert isinstance(ws, cake_backend.KdaDetWorkspace)
        out, saved = det_api.chunk_kda_det_fwd(
            q, k, v, g, beta, cu, workspace=ws, arena=arena
        )
        assert isinstance(saved, cake_backend.KdaDetChunkSaved)
        assert torch.equal(out, ref_out)
        grads = det_api.chunk_kda_det_bwd(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=ws, arena=arena
        )
        for mine, theirs in zip(grads, ref_grads, strict=True):
            assert torch.equal(mine, theirs)
        leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)]
        o = det_api.chunk_kda_det(*leaves, cu, workspace=ws)
        o.backward(dout)
        for leaf, expected in zip(
            leaves,
            (ref_grads[0], ref_grads[1], ref_grads[2], ref_grads[4], ref_grads[3]),
            strict=True,
        ):
            assert torch.equal(leaf.grad, expected)
