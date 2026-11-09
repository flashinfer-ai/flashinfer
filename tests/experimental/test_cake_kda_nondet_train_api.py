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

flashinfer public-API tests of the Cake non-deterministic packed KDA chunked training package
(``flashinfer.experimental.cake.cake_kda_nondet_train``).

The backend suite ``test_cake_kda_nondet_train.py`` imports only the package.  This file covers what needs flashinfer
itself and the package's boundary with the deterministic package.  On the CPU: the public entry points
``flashinfer.cake_kda_nondet_train.cake_kda_nondet_train_forward`` / ``cake_kda_nondet_train_backward`` /
``cake_kda_nondet`` forward every keyword to the package entry of the same name (the package entries replaced by
recorders), with the default scale and, from ``cake_kda_nondet``, the checkpoint hand-off; the arena, workspace,
byte and support helpers next to them; the signatures (the deterministic facade's, ``cake_kda_nondet`` adding
``checkpoint_handoff``); the deferred imports; torch's deterministic mode, which no entry reads (the package imports
nothing of the deterministic package, and every entry goes on to its host checks under it, warning nothing); the
refusal of a workspace, a saved set or an arena option of the deterministic package (or of any other type) with
``ValueError`` naming the mode, before any operand check; the raw gate mode (#7816) and the in-kernel Q/K L2 norm
(#7817) with the deterministic package's checks and byte counts; and a dispatch sweep over architecture x path x
heads 1..128 (one record per stage, the fused gradient and the GLM intra program agreeing on S1 as the independent
table says, the plan's records the registry's, every record of the pair stages launched on every architecture it
ships for; the in-kernel Q/K L2 norm modes select the deterministic pair on every cell) with the plan's pairing
guard.  On a GPU: the public entries against the package and the deterministic package, the refusal of the other
package's workspace and saved set, torch's deterministic mode (strict and ``warn_only``: this package runs, nothing
warns or raises), the autograd node and the raw gate mode.  The names are covered by
``test_cake_kda_nondet_train_names.py``.
"""

import ast
import contextlib
import dataclasses
import inspect
import json
import os
import re
import subprocess
import sys
import types
import warnings
from pathlib import Path

import pytest
import torch

import flashinfer.cake_kda_det_train as det_api
import flashinfer.cake_kda_nondet_train as nd_api
from flashinfer.experimental.cake import cake_kda_nondet_train as pkg
from flashinfer.experimental.cake.cake_kda_det_train import cake_backend as det_backend
from flashinfer.experimental.cake.cake_kda_det_train import cake_jit as det_jit
from flashinfer.experimental.cake.cake_kda_nondet_train import cake_backend, cake_jit
from tests.experimental.test_helpers.cake_kda_det_train_reference import (
    LOWER_BOUND,
    make_inputs,
    make_raw_inputs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FLASHINFER_DIR = Path(nd_api.__file__).resolve().parent
GRAD_ORDER = ("dq", "dk", "dv", "dbeta", "dg")  # the order the backward returns
# facade entry -> its twin in the deterministic facade (the same parameters; cake_kda_nondet adds checkpoint_handoff)
DET_TWIN = {
    "cake_kda_nondet_train_forward": "cake_kda_det_train_forward",
    "cake_kda_nondet_train_backward": "cake_kda_det_train_backward",
    "cake_kda_nondet": "cake_kda_det",
    "cake_kda_nondet_train_arena": "cake_kda_det_train_arena",
    "cake_kda_nondet_train_workspace": "cake_kda_det_train_workspace",
    "cake_kda_nondet_train_workspace_bytes": "cake_kda_det_train_workspace_bytes",
    "cake_kda_nondet_train_saved_bytes": "cake_kda_det_train_saved_bytes",
    "is_cake_kda_nondet_train_supported": "is_cake_kda_det_train_supported",
}
ENTRY_POINTS = (
    "cake_kda_nondet_train_forward",
    "cake_kda_nondet_train_backward",
    "cake_kda_nondet",
)
# what this package's entries take, by argument (the type each refusal names)
ND_TYPES = {
    "workspace": "CakeKDANondetWorkspace",
    "saved": "CakeKDANondetChunkSaved",
    "arena": "CakeKDANondetArena",
}
# the refusals of the other package's objects: this package's, then the deterministic package's
ND_REFUSES = "is not a non-deterministic-package"
DET_REFUSES = "is not a deterministic-package"
_ALERT = "does not have a deterministic implementation"
# torch's deterministic-mode queries: no entry reads them
_TORCH_MODE_QUERIES = (
    "are_deterministic_algorithms_enabled",
    "is_deterministic_algorithms_warn_only_enabled",
)


# The S1 cells (the backend suite's table, spelled again: the suites import nothing from each other).  The saved-set
# paths run the S1 pair from S1_SAVED_MIN_H heads, the replay up to S1_REPLAY_MAX_H heads (None: every count), the
# serial rows of 9..15 heads the deterministic pair everywhere; the sm_107a cells are provisional.
SAVED_PATHS = ("light", "light_states", "tape")
S1_SAVED_MIN_H = {"sm_100a": 16, "sm_103a": 1, "sm_107a": 1}
S1_REPLAY_MAX_H = {"sm_100a": None, "sm_103a": 8, "sm_107a": None}
SEG_PLAN_HEADS_BY_ARCH = {"sm_103a": (16,), "sm_107a": (16,)}
S1_GLM = {
    "sm_100a": "glm_intra_v21",
    "sm_103a": "glm_intra_v22",
    "sm_107a": "glm_intra_v23",
}
S1_RECORDS = frozenset(
    (
        "dqkg_davq_v24",
        "dqkg_int_davq_v24",
        "dqkg_davq_nohint_v24",
        "dqkg_int_davq_nohint_v24",
        *S1_GLM.values(),
    )
)


def _s1(arch: str, path: str, h: int) -> bool:
    if path not in (*SAVED_PATHS, "replay"):
        return False
    seg = h <= 8 or h in SEG_PLAN_HEADS_BY_ARCH.get(arch, ())
    if not seg and h < 16:
        return False
    if path == "replay":
        bound = S1_REPLAY_MAX_H[arch]
        return bound is None or h <= bound
    return h >= S1_SAVED_MIN_H[arch]


@contextlib.contextmanager
def _torch_deterministic(warn_only: bool = False):
    """``torch.use_deterministic_algorithms(True, warn_only=...)`` inside the block, the previous setting restored
    after it."""
    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])


@pytest.fixture
def restore_torch_deterministic():
    """The test may switch torch's deterministic algorithms on; the previous setting is restored after it, however
    it ends."""
    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    yield
    torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])


def _run_cpu_subprocess(code: str) -> str:
    """Run ``code`` in a fresh interpreter with no visible GPU; its last stdout line."""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return proc.stdout.strip().splitlines()[-1]


def _flashinfer_warnings(records) -> list[str]:
    """The warnings of ``records`` raised from flashinfer code or naming a deterministic implementation (torch's
    non-determinism alert in ``warn_only`` mode)."""
    found = []
    for w in records:
        try:
            inside = Path(w.filename).resolve().is_relative_to(FLASHINFER_DIR)
        except (OSError, ValueError):
            inside = False
        if inside or "deterministic" in str(w.message):
            found.append(
                f"{w.category.__name__}: {w.message} ({w.filename}:{w.lineno})"
            )
    return found


def _refused(name: str, value: object) -> str:
    """The message pattern of this package's refusal of the ``name`` argument ``value``."""
    return (
        re.escape(
            f"{name}: a {type(value).__name__} from {type(value).__module__} is not a non-deterministic-package "
        )
        + ".*"
        + re.escape(f"; the non-deterministic entries take a {ND_TYPES[name]} (")
        + ".*"
        + re.escape(
            "; the deterministic package is flashinfer.cake_kda_det_train / flashinfer.cake_kda_det_train_kcp"
        )
    )


# --------------------------------------------------------------------------- the facade (CPU)


def test_default_scale_is_the_contracts() -> None:
    assert nd_api.DEFAULT_SCALE == det_api.DEFAULT_SCALE == 128**-0.5
    assert nd_api.DEFAULT_SCALE == cake_backend.SCALE == det_backend.SCALE


def test_entry_points_forward_every_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each entry hands its operands and every keyword to the package entry of the same name (``scale=None``: the
    default scale), ``cake_kda_nondet`` its ``checkpoint_handoff`` too, and returns what the package returns."""
    calls = []

    def recorder(name):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        return fake

    for name in ENTRY_POINTS:
        monkeypatch.setattr(cake_backend, name, recorder(name))
    operands = tuple(object() for _ in range(6))
    dout, saved, workspace, arena = object(), object(), object(), object()
    assert (
        nd_api.cake_kda_nondet_train_forward(
            *operands,
            save_for_backward=False,
            workspace=workspace,
            save_mode="tape",
        )
        == "cake_kda_nondet_train_forward"
    )
    assert (
        nd_api.cake_kda_nondet_train_backward(
            *operands, dout, saved=saved, scale=0.5, arena=arena
        )
        == "cake_kda_nondet_train_backward"
    )
    assert nd_api.cake_kda_nondet(*operands, workspace=workspace) == "cake_kda_nondet"
    assert (
        nd_api.cake_kda_nondet(*operands, checkpoint_handoff="off", doc_lengths=(1, 2))
        == "cake_kda_nondet"
    )
    gates = {
        "use_gate_in_kernel": False,
        "A_log": None,
        "dt_bias": None,
        "lower_bound": None,
        "use_qk_l2norm_in_kernel": False,
    }
    assert calls == [
        (
            "cake_kda_nondet_train_forward",
            operands,
            {
                "scale": nd_api.DEFAULT_SCALE,
                "save_for_backward": False,
                "workspace": workspace,
                "save_mode": "tape",
                "arena": None,
                "doc_lengths": None,
                **gates,
            },
        ),
        (
            "cake_kda_nondet_train_backward",
            (*operands, dout),
            {
                "saved": saved,
                "scale": 0.5,
                "workspace": None,
                "arena": arena,
                "doc_lengths": None,
                **gates,
            },
        ),
        (
            "cake_kda_nondet",
            operands,
            {
                "scale": nd_api.DEFAULT_SCALE,
                "workspace": workspace,
                "save_mode": None,
                "arena": None,
                "doc_lengths": None,
                **gates,
                "checkpoint_handoff": "auto",
            },
        ),
        (
            "cake_kda_nondet",
            operands,
            {
                "scale": nd_api.DEFAULT_SCALE,
                "workspace": None,
                "save_mode": None,
                "arena": None,
                "doc_lengths": (1, 2),
                **gates,
                "checkpoint_handoff": "off",
            },
        ),
    ]


def test_arena_option_is_forwarded_and_the_factory_is_the_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arena = nd_api.cake_kda_nondet_train_arena(lean=True, tail_passes=4)
    assert arena == cake_backend.CakeKDANondetArena(lean=True, tail_passes=4)
    assert type(arena) is cake_backend.CakeKDANondetArena
    assert nd_api.cake_kda_nondet_train_arena() == cake_backend.CakeKDANondetArena()
    assert nd_api.cake_kda_nondet_train_arena(
        tail_passes=2
    ).lean  # several passes imply the lean layouts
    with pytest.raises(ValueError, match="tail_passes"):
        nd_api.cake_kda_nondet_train_arena(tail_passes=0)
    batched = nd_api.cake_kda_nondet_train_arena(
        batch_tokens=1000, max_docs_per_batch=2
    )
    assert batched == cake_backend.CakeKDANondetArena(
        batch_tokens=1000, max_docs_per_batch=2
    )
    assert batched.batched and not batched.lean
    with pytest.raises(ValueError, match="batch_tokens"):
        nd_api.cake_kda_nondet_train_arena(max_docs_per_batch=1)
    calls = []

    def recorder(name):
        def fake(*args, **kwargs):
            calls.append((name, kwargs.get("arena"), kwargs.get("doc_lengths")))
            return name

        return fake

    for name in ENTRY_POINTS:
        monkeypatch.setattr(cake_backend, name, recorder(name))
    monkeypatch.setattr(cake_backend, "workspace_bytes", recorder("bytes"))
    operands = tuple(object() for _ in range(6))
    lengths = (300, 700)
    nd_api.cake_kda_nondet_train_forward(*operands, arena=arena)
    nd_api.cake_kda_nondet_train_backward(*operands, object(), arena=arena)
    nd_api.cake_kda_nondet(*operands, arena=arena)
    nd_api.cake_kda_nondet_train_workspace_bytes(1, 1, 1, sm_count=148, arena=arena)
    nd_api.cake_kda_nondet_train_forward(*operands, arena=batched, doc_lengths=lengths)
    nd_api.cake_kda_nondet_train_backward(
        *operands, object(), arena=batched, doc_lengths=lengths
    )
    nd_api.cake_kda_nondet(*operands, arena=batched, doc_lengths=lengths)
    nd_api.cake_kda_nondet_train_workspace_bytes(
        1000, 2, 1, sm_count=148, arena=batched, doc_lengths=lengths
    )
    assert calls == [
        ("cake_kda_nondet_train_forward", arena, None),
        ("cake_kda_nondet_train_backward", arena, None),
        ("cake_kda_nondet", arena, None),
        ("bytes", arena, None),
        ("cake_kda_nondet_train_forward", batched, lengths),
        ("cake_kda_nondet_train_backward", batched, lengths),
        ("cake_kda_nondet", batched, lengths),
        ("bytes", batched, lengths),
    ]


def test_gate_and_qk_l2norm_modes_are_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The entries hand the raw gate keywords and ``use_qk_l2norm_in_kernel`` (the same objects) to the package; the
    workspace and the byte queries take ``use_gate_in_kernel`` as ``gate`` (``"raw"`` / ``"g"``) and
    ``use_qk_l2norm_in_kernel`` as ``l2n`` (``"fused"`` / ``"off"``), and refuse a value that is not a bool with
    ``ValueError`` before the package is called."""
    log = []

    def recorder(name):
        def fake(*args, **kwargs):
            log.append((name, kwargs))
            return name

        return fake

    for name in ENTRY_POINTS:
        monkeypatch.setattr(cake_backend, name, recorder(name))
    monkeypatch.setattr(cake_backend, "workspace_bytes", recorder("bytes"))
    monkeypatch.setattr(cake_backend, "saved_bytes", recorder("saved"))
    monkeypatch.setattr(
        cake_backend.CakeKDANondetWorkspace, "for_problem", recorder("workspace")
    )
    operands = tuple(object() for _ in range(6))
    gate = {
        "use_gate_in_kernel": True,
        "A_log": object(),
        "dt_bias": object(),
        "lower_bound": -5.0,
        "use_qk_l2norm_in_kernel": True,
    }
    for name, call in (
        ("cake_kda_nondet_train_forward", nd_api.cake_kda_nondet_train_forward),
        (
            "cake_kda_nondet_train_backward",
            lambda *a, **kw: nd_api.cake_kda_nondet_train_backward(*a, object(), **kw),
        ),
        ("cake_kda_nondet", nd_api.cake_kda_nondet),
    ):
        assert call(*operands, **gate) == name
        for key, value in gate.items():
            assert log[-1][1][key] is value, (name, key)
    log.clear()
    queries = (
        (
            "workspace",
            lambda **kw: nd_api.cake_kda_nondet_train_workspace(1, 1, 1, "cpu", **kw),
        ),
        (
            "bytes",
            lambda **kw: nd_api.cake_kda_nondet_train_workspace_bytes(
                1, 1, 1, sm_count=148, **kw
            ),
        ),
    )
    for name, query in queries:
        for flag in (False, True):
            assert query(use_gate_in_kernel=flag) == name
            assert log[-1][1]["gate"] == ("raw" if flag else "g"), (name, flag)
            assert log[-1][1]["l2n"] == "off", (name, flag)
            assert query(use_qk_l2norm_in_kernel=flag) == name
            assert log[-1][1]["l2n"] == ("fused" if flag else "off"), (name, flag)
            assert log[-1][1]["gate"] == "g", (name, flag)
    for flag in (False, True):
        nd_api.cake_kda_nondet_train_saved_bytes(1, 1, use_qk_l2norm_in_kernel=flag)
        assert log[-1][0] == "saved"
        assert log[-1][1]["l2n"] == ("fused" if flag else "off"), flag
    log.clear()
    for bad in (1, None, "True"):
        for _, query in queries:
            with pytest.raises(ValueError, match="use_gate_in_kernel must be a bool"):
                query(use_gate_in_kernel=bad)
            with pytest.raises(
                ValueError, match="use_qk_l2norm_in_kernel must be a bool"
            ):
                query(use_qk_l2norm_in_kernel=bad)
        with pytest.raises(ValueError, match="use_qk_l2norm_in_kernel must be a bool"):
            nd_api.cake_kda_nondet_train_saved_bytes(1, 1, use_qk_l2norm_in_kernel=bad)
    assert log == []


def test_byte_queries_are_the_backends_and_the_deterministic_packages() -> None:
    # An explicit SM count keeps the queries device-free.  The two packages' workspaces and saved sets have the same
    # size (the layouts are the same; the S1 programs bind the deterministic forms' buffers).
    for S, N, H in ((1, 1, 1), (129, 2, 4), (4096, 3, 16), (261393, 4, 64)):
        for kwargs in ({}, {"backward": False}, {"save_mode": "tape"}):
            nd = nd_api.cake_kda_nondet_train_workspace_bytes(
                S, N, H, sm_count=148, **kwargs
            )
            assert nd == cake_backend.workspace_bytes(S, N, H, sm_count=148, **kwargs)
            assert nd == det_backend.workspace_bytes(S, N, H, sm_count=148, **kwargs)
        for kwargs in ({}, {"save_mode": "tape"}):
            nd = nd_api.cake_kda_nondet_train_saved_bytes(
                S, H, N, sm_count=148, **kwargs
            )
            assert nd == cake_backend.saved_bytes(S, H, N, sm_count=148, **kwargs)
            assert nd == det_backend.saved_bytes(S, H, N, sm_count=148, **kwargs)
        assert nd_api.cake_kda_nondet_train_saved_bytes(S, H) == 256 * S * H
    # the batched option: the bound of every group without doc_lengths, the exact groups with them; refused on the
    # rows with at most four (document, head) chains
    lengths = (23496, 88510, 114974, 34413)
    S, N, H = sum(lengths), len(lengths), 4
    batched = nd_api.cake_kda_nondet_train_arena(
        lean=True, tail_passes=4, batch_tokens=120000
    )
    bound = nd_api.cake_kda_nondet_train_workspace_bytes(
        S, N, H, sm_count=148, arena=batched
    )
    exact = nd_api.cake_kda_nondet_train_workspace_bytes(
        S, N, H, sm_count=148, arena=batched, doc_lengths=lengths
    )
    assert exact == cake_backend.workspace_bytes(
        S, N, H, sm_count=148, arena=batched, doc_lengths=lengths
    )
    assert (
        exact
        <= bound
        == cake_backend.workspace_bytes(S, N, H, sm_count=148, arena=batched)
        < nd_api.cake_kda_nondet_train_workspace_bytes(S, N, H, sm_count=148)
    )
    with pytest.raises(ValueError, match="proportional"):
        nd_api.cake_kda_nondet_train_workspace_bytes(
            128,
            2,
            1,
            sm_count=148,
            arena=nd_api.cake_kda_nondet_train_arena(batch_tokens=64),
        )


def test_support_query_follows_the_capability_table() -> None:
    if not torch.cuda.is_available():
        assert nd_api.is_cake_kda_nondet_train_supported() is False
        return
    index = torch.cuda.current_device()
    expected = (
        cake_jit.arch_for_capability(tuple(torch.cuda.get_device_capability(index)))
        is not None
    )
    assert nd_api.is_cake_kda_nondet_train_supported() is expected
    assert nd_api.is_cake_kda_nondet_train_supported(index) is expected
    assert (
        nd_api.is_cake_kda_nondet_train_supported(torch.device("cuda", index))
        is expected
    )


def test_signatures_are_the_deterministic_facades() -> None:
    """Each entry takes the parameters of its deterministic twin, in order, kinds and defaults the same, and nothing
    else; ``cake_kda_nondet`` also takes the keyword-only ``checkpoint_handoff="auto"`` (the checkpoint hand-off)
    after them.  The package entries take every keyword their facade entry forwards."""
    for name, det_name in DET_TWIN.items():
        params = list(inspect.signature(getattr(nd_api, name)).parameters.values())
        if name == "cake_kda_nondet":
            handoff = params.pop()
            assert handoff.name == "checkpoint_handoff", name
            assert handoff.kind is inspect.Parameter.KEYWORD_ONLY, name
            assert handoff.default == "auto", name
        det_params = inspect.signature(getattr(det_api, det_name)).parameters.values()
        assert [(p.name, p.kind, p.default) for p in params] == [
            (p.name, p.kind, p.default) for p in det_params
        ], name
    for name in ENTRY_POINTS:
        facade = inspect.signature(getattr(nd_api, name)).parameters
        backend = inspect.signature(getattr(cake_backend, name)).parameters
        assert set(facade) <= set(backend), (name, set(facade) - set(backend))
        for key, param in facade.items():
            if param.kind is inspect.Parameter.KEYWORD_ONLY:
                assert backend[key].kind is inspect.Parameter.KEYWORD_ONLY, (name, key)


def test_importing_the_facade_imports_no_package() -> None:
    """The facade reaches its package through deferred imports only, and never the deterministic package."""
    code = (
        "import json, sys\n"
        "import flashinfer.cake_kda_nondet_train\n"
        "print(json.dumps(sorted(m for m in sys.modules"
        " if m.startswith('flashinfer.experimental.cake.cake_kda_det_train')"
        " or m.startswith('flashinfer.experimental.cake.cake_kda_nondet_train'))))\n"
    )
    assert json.loads(_run_cpu_subprocess(code)) == []


def test_importing_the_package_imports_no_deterministic_module() -> None:
    """The package and every module of it import nothing of the deterministic package or of its facade (an import
    of either would let a launch reach a deterministic program), nor this package's facade."""
    code = (
        "import json, sys\n"
        "import flashinfer\n"
        "before = set(sys.modules)\n"
        "import flashinfer.experimental.cake.cake_kda_nondet_train\n"
        "from flashinfer.experimental.cake.cake_kda_nondet_train import cake_backend, cake_build, cake_jit\n"
        "new = set(sys.modules) - before\n"
        "assert 'flashinfer.experimental.cake.cake_kda_nondet_train.cake_backend' in new\n"
        "bad = sorted(m for m in sys.modules if 'cake_kda_det_train' in m"
        " or m == 'flashinfer.cake_kda_nondet_train')\n"
        "print(json.dumps(bad))\n"
    )
    assert json.loads(_run_cpu_subprocess(code)) == []


# --------------------------------------------------------------------------- torch's deterministic mode (CPU)


def _callee(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def test_no_entry_reads_torch_mode_or_imports_the_deterministic_package() -> None:
    """Statically, in the facade and every module of the package: no read of torch's deterministic-mode queries
    (``torch.are_deterministic_algorithms_enabled`` / ``is_deterministic_algorithms_warn_only_enabled``, as an
    attribute, a name or a ``getattr`` string) and no import of a ``cake_kda_det_train`` module (``import``,
    ``from ... import``, ``importlib.import_module`` or ``__import__``): no entry can choose the deterministic
    package, whatever torch's mode."""
    files = [Path(nd_api.__file__).resolve()] + sorted(
        Path(cake_backend.__file__).resolve().parent.glob("*.py")
    )
    assert {p.name for p in files} >= {
        "cake_kda_nondet_train.py",
        "__init__.py",
        "cake_backend.py",
        "cake_build.py",
        "cake_jit.py",
    }
    failures = []
    for path in files:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            where = f"{path.name}:{getattr(node, 'lineno', '?')}"
            if (
                (isinstance(node, ast.Attribute) and node.attr in _TORCH_MODE_QUERIES)
                or (isinstance(node, ast.Name) and node.id in _TORCH_MODE_QUERIES)
                or (
                    isinstance(node, ast.Constant) and node.value in _TORCH_MODE_QUERIES
                )
            ):
                failures.append(f"{where}: reads torch's deterministic mode")
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""] + [alias.name for alias in node.names]
            elif isinstance(node, ast.Call) and _callee(node) in (
                "import_module",
                "__import__",
            ):
                imported = [
                    arg.value
                    for arg in node.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                ]
            else:
                imported = []
            if any("cake_kda_det_train" in name for name in imported):
                failures.append(f"{where}: imports {imported}")
    assert not failures, failures


_TORCH_DETERMINISTIC = """
import sys
import warnings

import torch

import flashinfer.cake_kda_nondet_train as nd_api
from flashinfer.experimental.cake.cake_kda_nondet_train import cake_backend

assert not torch.cuda.is_available()
q = torch.zeros(4, 1, 128, dtype=torch.bfloat16)
v = torch.zeros(4, 1, dtype=torch.bfloat16)  # the wrong rank: the entries stop at their host checks
g = torch.zeros(4, 1, 128)
beta = torch.zeros(4, 1, dtype=torch.bfloat16)
ops = (q, q, v, g, beta, torch.tensor([0, 4], dtype=torch.int32))
for warn_only in (False, True):
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert nd_api.cake_kda_nondet_train_saved_bytes(64, 2) == cake_backend.saved_bytes(64, 2)
        assert nd_api.cake_kda_nondet_train_workspace_bytes(
            129, 1, 4, sm_count=148
        ) == cake_backend.workspace_bytes(129, 1, 4, sm_count=148)
        assert type(nd_api.cake_kda_nondet_train_arena()) is cake_backend.CakeKDANondetArena
        assert nd_api.is_cake_kda_nondet_train_supported() is False
        for call in (
            lambda: nd_api.cake_kda_nondet_train_forward(*ops),
            lambda: nd_api.cake_kda_nondet_train_backward(*ops, q),
            lambda: nd_api.cake_kda_nondet(*ops),
        ):
            try:
                call()
            except ValueError as error:
                assert "v: expected 3-D" in str(error), error
            else:
                raise AssertionError("the host check did not run")
    assert not rec, [str(w.message) for w in rec]
bad = sorted(m for m in sys.modules if "cake_kda_det_train" in m)
assert bad == [], bad
print("OK")
"""


def test_torch_deterministic_mode_keeps_this_package_on_cpu() -> None:
    """In a fresh process with torch's deterministic mode on (strict, then ``warn_only``), the facade's helpers answer
    with this package's figures and types and its entries go on to their host checks; nothing warns and no module of
    the deterministic package or of its facade is imported."""
    assert _run_cpu_subprocess(_TORCH_DETERMINISTIC) == "OK"


def _bad_operands():
    """CPU operands whose ``v`` has the wrong rank: a call reaches the host check and stops there."""
    q = torch.zeros(4, 1, 128, dtype=torch.bfloat16)
    v = torch.zeros(4, 1, dtype=torch.bfloat16)
    g = torch.zeros(4, 1, 128)
    beta = torch.zeros(4, 1, dtype=torch.bfloat16)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    return q, q, v, g, beta, cu


def _package_call(name: str):
    """A call of the package entry ``name`` on :func:`_bad_operands` (the backward's with a ``dout``) with the
    keywords given."""
    entry = getattr(cake_backend, name)
    dout = (
        (torch.zeros(4, 1, 128, dtype=torch.bfloat16),)
        if name == "cake_kda_nondet_train_backward"
        else ()
    )
    return lambda **kw: entry(*_bad_operands(), *dout, **kw)


def _node_call():
    """The autograd node on :func:`_bad_operands`: scale, workspace, save_mode, arena, doc_lengths and the
    checkpoint hand-off (``None``: ``"auto"``)."""
    return cake_backend.CakeKDANondetFunction.apply(
        *_bad_operands(), cake_backend.SCALE, None, None, None, None, None
    )


PACKAGE_CALLS = {
    **{name: _package_call(name) for name in ENTRY_POINTS},
    "CakeKDANondetFunction": _node_call,
}


@pytest.mark.parametrize("name", list(PACKAGE_CALLS))
def test_package_entries_do_not_alert_under_torch_deterministic_mode(name) -> None:
    """The S1 schedule has a fixed order and no unordered reduction: under torch's deterministic mode (an alert as
    an error, a warning as an error) each entry proceeds to its host checks, never to an alert and never to the
    deterministic package."""
    for warn_only in (False, True):
        with _torch_deterministic(warn_only), warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="v: expected 3-D") as info:
                PACKAGE_CALLS[name]()
            assert _ALERT not in str(info.value)
            assert "cake_kda_det_train" not in str(info.value)


# --------------------------------------------------------------------------- the deterministic package's objects (CPU)

LENGTHS, HEADS = (63, 1, 65), 2


class _Foreign:
    """A stand-in of another package's workspace / saved set / arena option."""


def _cpu_operands() -> tuple[torch.Tensor, ...]:
    inputs = make_inputs(LENGTHS, HEADS, "cpu", batched=True)
    return tuple(inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout"))


def _saved_on_host(saved_type) -> object:
    """A saved set of ``saved_type`` built on the host (the fields every check reads)."""
    S = sum(LENGTHS)
    a = torch.zeros(S, HEADS, 64, dtype=torch.bfloat16)
    return saved_type(
        version=cake_backend.SAVED_VERSION,
        Aqk=a,
        Akk=a.clone(),
        tables=None,
        S=S,
        H=HEADS,
        num_docs=len(LENGTHS),
        scale=cake_backend.SCALE,
    )


def _other(source: str, name: str) -> object:
    """The ``name`` argument of ``source``: the deterministic package's type (a workspace without device memory:
    the refusal reads its type alone), or a stand-in of another package."""
    if source == "foreign":
        return _Foreign()
    if name == "workspace":
        return object.__new__(det_backend.CakeKDADetWorkspace)
    if name == "saved":
        return _saved_on_host(det_backend.CakeKDADetChunkSaved)
    return det_backend.CakeKDADetArena(lean=True, tail_passes=2)


@pytest.mark.parametrize("source", ["deterministic", "foreign"])
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
def test_entries_refuse_another_packages_object_on_cpu(
    entry: str, name: str, source: str
) -> None:
    """The public entries refuse a workspace, a saved set or an arena option that is not this package's type (the
    deterministic package's, or any other) with ``ValueError`` naming the mode and the type each argument takes,
    before any operand check (the CPU operands would fail the device checks after it)."""
    q, k, v, g, beta, cu, dout = _cpu_operands()
    other = _other(source, name)
    with pytest.raises(ValueError, match=_refused(name, other)):
        if entry == "fwd":
            nd_api.cake_kda_nondet_train_forward(q, k, v, g, beta, cu, **{name: other})
        elif entry == "bwd":
            nd_api.cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, **{name: other}
            )
        else:
            nd_api.cake_kda_nondet(q, k, v, g, beta, cu, **{name: other})


@pytest.mark.parametrize("source", ["deterministic", "foreign"])
@pytest.mark.parametrize("name", ["workspace", "arena"])
def test_workspace_paths_refuse_another_packages_object(name: str, source: str) -> None:
    """The call's workspace helper refuses another package's workspace or arena option, and the workspace
    constructor another package's arena option, before the device is read."""
    other = _other(source, name)
    with pytest.raises(ValueError, match=_refused(name, other)):
        kwargs = {"arena": other} if name == "arena" else {}
        with cake_backend._workspace(
            other if name == "workspace" else None,
            1,
            1,
            1,
            torch.device("cpu"),
            backward=True,
            **kwargs,
        ):
            pass
    if name == "arena":
        plan = cake_backend.plan(129, 1, 4, 148, "sm_100a")
        with pytest.raises(ValueError, match=_refused(name, other)):
            cake_backend.CakeKDANondetWorkspace(plan, "cpu", arena=other)


def test_nondet_objects_pass_the_package_check() -> None:
    saved = _saved_on_host(cake_backend.CakeKDANondetChunkSaved)
    cake_backend._check_nondet_package(
        workspace=object.__new__(cake_backend.CakeKDANondetWorkspace),
        saved=saved,
        arena=nd_api.cake_kda_nondet_train_arena(lean=True),
    )
    cake_backend._check_nondet_package(workspace=None, saved=None, arena=None)
    # the deterministic package's types carry the attributes the other checks read: the type alone tells them apart
    det_saved = _saved_on_host(det_backend.CakeKDADetChunkSaved)
    assert [f.name for f in dataclasses.fields(det_saved)] == [
        f.name for f in dataclasses.fields(saved)
    ]
    with pytest.raises(ValueError, match=_refused("saved", det_saved)):
        cake_backend._check_nondet_package(saved=det_saved)


# ------------------------------------------------------------------------- the raw gate mode and the Q/K L2 norm (CPU)


class _Reached(Exception):
    """Raised by :class:`_StandInWorkspace` when a call reads the workspace's state."""


class _StandInWorkspace(cake_backend.CakeKDANondetWorkspace):
    """This package's workspace type without a plan or device memory: the package check passes it, and the first
    read of its state (the workspace's own checks, after every operand and mode check) raises :class:`_Reached`;
    nothing reads a device."""

    def __init__(self) -> None:
        pass

    def __getattr__(self, name: str):
        raise _Reached(name)


# the raw gate mode (#7816): the keywords at their off values
GATE_OFF = {
    "use_gate_in_kernel": False,
    "A_log": None,
    "dt_bias": None,
    "lower_bound": None,
}


def _gate_on() -> list:
    return [
        {"use_gate_in_kernel": True},
        {"A_log": torch.zeros(4)},
        {"dt_bias": torch.zeros(4, 128)},
        {"lower_bound": -5.0},
        {
            "use_gate_in_kernel": True,
            "A_log": torch.zeros(1),
            "dt_bias": torch.zeros(128),
            "lower_bound": -5.0,
        },
    ]


def _raw_cpu_call():
    """CPU operands of the raw gate mode that pass every operand check (the BF16 raw gate, one head) and its
    keywords: a call with :class:`_StandInWorkspace` passes every host check before the workspace's own and stops
    at the first read of the workspace's state, reading no device."""
    q = torch.zeros(4, 1, 128, dtype=torch.bfloat16)
    beta = torch.zeros(4, 1, dtype=torch.bfloat16)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    raw = {
        "use_gate_in_kernel": True,
        "A_log": torch.zeros(1),
        "dt_bias": torch.zeros(128),
        "lower_bound": -5.0,
    }
    return (q, q, q, q.clone(), beta, cu), raw


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_package_accepts_the_raw_gate(name) -> None:
    """The raw gate mode runs on this package too: each entry takes its keywords and goes on to the host checks (the
    stand-in operands' check, as with the keywords off or omitted); the mode's own checks are the deterministic
    package's (a non-bool ``use_gate_in_kernel``, an FP32 raw gate, a missing lower bound or one outside
    ``[-5, 0)``, the parameters in g mode: ``ValueError``), and a well-formed raw call on CPU tensors passes them all
    and reaches the workspace's checks."""
    call = PACKAGE_CALLS[name]
    for kw in _gate_on():
        with pytest.raises(ValueError, match="v: expected 3-D"):
            call(**kw)
    with pytest.raises(ValueError, match="v: expected 3-D"):
        call(**GATE_OFF)
    with pytest.raises(ValueError, match="use_gate_in_kernel must be a bool"):
        call(use_gate_in_kernel=1)
    entry = getattr(cake_backend, name)
    ops, raw = _raw_cpu_call()
    dout = (
        (torch.zeros(4, 1, 128, dtype=torch.bfloat16),)
        if name == "cake_kda_nondet_train_backward"
        else ()
    )
    g32 = (*ops[:3], torch.zeros(4, 1, 128), *ops[4:])
    # the arena option given: the backward reads no option from the stand-in workspace
    stand_in = {
        "workspace": _StandInWorkspace(),
        "arena": cake_backend.CakeKDANondetArena(),
    }
    with pytest.raises(_Reached):
        entry(*ops, *dout, **stand_in, **raw)
    with pytest.raises(ValueError, match="is torch.bfloat16"):
        entry(*g32, *dout, **raw)
    with pytest.raises(ValueError, match="needs the gate's lower bound"):
        entry(*ops, *dout, **dict(raw, lower_bound=None))
    with pytest.raises(ValueError, match="lower_bound must satisfy"):
        entry(*ops, *dout, **dict(raw, lower_bound=-6.0))
    with pytest.raises(ValueError, match="taken with use_gate_in_kernel=True only"):
        entry(*g32, *dout, A_log=raw["A_log"])


def test_autograd_node_takes_the_raw_gate_after_the_handoff_slot() -> None:
    """``CakeKDANondetFunction.apply`` takes the raw gate mode's arguments after ``checkpoint_handoff``
    (``use_gate_in_kernel``, ``A_log``, ``dt_bias``, ``lower_bound``): a raw call on CPU tensors with
    :class:`_StandInWorkspace` passes the raw checks and reaches the workspace's checks, a workspace of another
    package is refused with ``ValueError`` naming the mode, and an ``apply`` written for the deterministic
    Function's signature (the raw arguments right after ``doc_lengths``) passes a ``bool`` in the
    ``checkpoint_handoff`` slot and raises ``TypeError`` naming it."""
    ops, raw = _raw_cpu_call()
    scale = cake_backend.SCALE
    arena = cake_backend.CakeKDANondetArena()
    with pytest.raises(_Reached):
        cake_backend.CakeKDANondetFunction.apply(
            *ops, scale, _StandInWorkspace(), None, arena, None, None, *raw.values()
        )
    foreign = _Foreign()
    with pytest.raises(ValueError, match=_refused("workspace", foreign)):
        cake_backend.CakeKDANondetFunction.apply(
            *ops, scale, foreign, None, None, None, None, *raw.values()
        )
    with pytest.raises(TypeError, match="checkpoint_handoff"):
        cake_backend.CakeKDANondetFunction.apply(
            *ops, scale, None, None, None, None, *raw.values()
        )
    with pytest.raises(ValueError, match="use_gate_in_kernel must be a bool"):
        cake_backend.CakeKDANondetFunction.apply(
            *ops, scale, None, None, None, None, None, 1, *list(raw.values())[1:]
        )


def test_both_packages_check_the_saved_gate_mode_alike() -> None:
    """Both packages' saved sets carry ``gate`` / ``lower_bound`` and both packages' check (``_check_saved_gate``)
    applies the same rule to a set of either type: a set of the call's gate mode passes, and in raw mode the FP32 bits
    of ``lower_bound`` decide; a set of the other gate mode or of another lower bound is refused with ``ValueError``.
    This package's backward reads its own ``CakeKDANondetChunkSaved`` only: a set of the in-kernel Q/K L2 norm
    (``qk_l2norm``) of either gate mode reaches the host checks of a call with ``use_qk_l2norm_in_kernel=True``, the
    deterministic package's ``CakeKDADetChunkSaved`` is refused first with ``ValueError`` naming the mode."""
    t = torch.empty(0)
    scale = cake_backend.SCALE
    H = 2
    params = (torch.zeros(H), torch.zeros(H * 128), (H * 128,), -5.0)
    checks = (
        (cake_backend._check_saved_gate, cake_backend._GateParams(*params)),
        (det_backend._check_saved_gate, det_backend._GateParams(*params)),
    )
    for saved_type in (
        cake_backend.CakeKDANondetChunkSaved,
        det_backend.CakeKDADetChunkSaved,
    ):
        raw = saved_type(
            4, t, t, None, 1, H, 1, scale, qg=t, gate="raw", lower_bound=-5.0
        )
        explicit = saved_type(4, t, t, None, 1, H, 1, scale, qg=t)
        assert (explicit.gate, explicit.lower_bound) == ("g", None)
        for check, gp in checks:
            check(raw, gp)
            check(explicit, None)
            # a lower bound with the same FP32 value
            check(dataclasses.replace(raw, lower_bound=-5.0 + 1e-12), gp)
            with pytest.raises(ValueError, match="gate mode 'raw'"):
                check(raw, None)
            with pytest.raises(ValueError, match="gate mode 'g'"):
                check(explicit, gp)
            for bound in (-4.0, None, -5.0 + 1e-6):
                with pytest.raises(ValueError, match="lower_bound"):
                    check(dataclasses.replace(raw, lower_bound=bound), gp)
    for gate in ({}, {"gate": "raw", "lower_bound": -5.0}):
        fields = dict(qg=t, qk_l2norm=True, **gate)
        nd_set = cake_backend.CakeKDANondetChunkSaved(
            4, t, t, None, 1, H, 1, scale, **fields
        )
        with pytest.raises(ValueError, match="v: expected 3-D"):
            PACKAGE_CALLS["cake_kda_nondet_train_backward"](
                saved=nd_set, use_qk_l2norm_in_kernel=True
            )
        det_set = det_backend.CakeKDADetChunkSaved(
            4, t, t, None, 1, H, 1, scale, **fields
        )
        with pytest.raises(ValueError, match=_refused("saved", det_set)):
            PACKAGE_CALLS["cake_kda_nondet_train_backward"](
                saved=det_set, use_qk_l2norm_in_kernel=True
            )
    with pytest.raises(ValueError, match=ND_REFUSES):
        PACKAGE_CALLS["cake_kda_nondet_train_backward"](
            saved=types.SimpleNamespace(gate="g", qk_l2norm=True),
            use_qk_l2norm_in_kernel=True,
        )


def test_package_workspace_takes_the_gate_mode() -> None:
    """``workspace_bytes`` and ``CakeKDANondetWorkspace`` take the gate mode of the calls they serve (``"g"``, the
    default, or ``"raw"``): the raw mode's bytes are the deterministic package's raw bytes on every architecture,
    forward or backward, any other value is a ``ValueError`` raised before the device is read, a known one reaches the
    device check, and a constructor whose ``gate`` is not its plan's is refused."""
    S, N, H = 129, 1, 4
    assert cake_backend.workspace_bytes(
        S, N, H, sm_count=148, gate="g"
    ) == cake_backend.workspace_bytes(S, N, H, sm_count=148)
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        for h in (H, 64):
            for backward in (True, False):
                assert cake_backend.workspace_bytes(
                    S, N, h, sm_count=148, arch=arch, gate="raw", backward=backward
                ) == det_backend.workspace_bytes(
                    S, N, h, sm_count=148, arch=arch, gate="raw", backward=backward
                ), (arch, h, backward)
    ws_type = cake_backend.CakeKDANondetWorkspace
    for gate in (None, "G", "activated"):
        with pytest.raises(ValueError, match="gate must be one of"):
            cake_backend.workspace_bytes(S, N, H, sm_count=148, gate=gate)
        with pytest.raises(ValueError, match="gate must be one of"):
            ws_type.for_problem(S, N, H, "cpu", gate=gate)
        with pytest.raises(ValueError, match="gate must be one of"):
            ws_type.for_budget(
                S,
                N,
                H,
                "cpu",
                arena_budget_bytes=1 << 40,
                gate=gate,
            )
    for gate in ("g", "raw"):
        with pytest.raises(ValueError, match="CUDA devices"):
            ws_type.for_problem(S, N, H, "cpu", gate=gate)
        p = cake_backend.plan(S, N, H, 148, "sm_100a", gate=gate)
        other = "g" if gate == "raw" else "raw"
        with pytest.raises(ValueError, match=f"gate={other!r}"):
            ws_type(p, "cpu", gate=other)
        with pytest.raises(ValueError, match="CUDA devices"):
            ws_type(p, "cpu", gate=gate)


def test_facade_runs_the_raw_gate_on_the_package() -> None:
    """Without recorders: the raw gate through the public entries reaches this package, whose entries go on to their
    host checks, and whose byte query gives the raw mode's bytes (the deterministic package's)."""
    for name, args in (
        ("cake_kda_nondet_train_forward", _bad_operands()),
        (
            "cake_kda_nondet_train_backward",
            (*_bad_operands(), torch.zeros(4, 1, 128)),
        ),
        ("cake_kda_nondet", _bad_operands()),
    ):
        with pytest.raises(ValueError, match="v: expected 3-D") as info:
            getattr(nd_api, name)(*args, use_gate_in_kernel=True)
        assert "cake_kda_det_train" not in str(info.value), name
    for backward in (True, False):
        got = nd_api.cake_kda_nondet_train_workspace_bytes(
            129, 1, 4, sm_count=148, backward=backward, use_gate_in_kernel=True
        )
        assert got == cake_backend.workspace_bytes(
            129, 1, 4, sm_count=148, backward=backward, gate="raw"
        )
        assert got == det_api.cake_kda_det_train_workspace_bytes(
            129, 1, 4, sm_count=148, backward=backward, use_gate_in_kernel=True
        )


# the in-kernel Q/K L2 norm (#7817): the deterministic package's keyword
_NOT_BOOL = "use_qk_l2norm_in_kernel must be a bool"


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_package_takes_the_in_kernel_qk_l2norm(name) -> None:
    """The in-kernel Q/K L2 norm runs on this package too: each entry takes ``use_qk_l2norm_in_kernel=True`` (and
    ``False``, as the omitted keyword) on to its host checks, and raises ``ValueError`` for a value that is not a
    bool before them."""
    call = PACKAGE_CALLS[name]
    for value in (True, False):
        with pytest.raises(ValueError, match="v: expected 3-D") as info:
            call(use_qk_l2norm_in_kernel=value, **GATE_OFF)
        assert "L2 norm" not in str(info.value), (value, str(info.value))
    for value in (1, 0, "True", None):
        with pytest.raises(ValueError, match=_NOT_BOOL):
            call(use_qk_l2norm_in_kernel=value)


def _l2n_saved(saved_type, S: int, H: int, **fields):
    t = torch.empty(0)
    return saved_type(4, t, t, None, S, H, 1, cake_backend.SCALE, qg=t, **fields)


def test_both_packages_check_the_l2n_saved_set_alike() -> None:
    """Both packages' saved sets carry ``qk_rstd`` / ``qk_l2norm`` under the same names and both packages' check
    (``_check_saved_l2n``) applies the same rule to a set of either type: a set of the call's in-kernel Q/K L2 norm
    mode passes; a set of the other mode, a missing or stray ``qk_rstd`` and a ``qk_rstd`` that is not FP32
    ``[2, S, H]`` contiguous are refused with ``ValueError``.  This package's backward takes its own l2n set on to the
    host checks and refuses the deterministic package's first (``ValueError`` naming the mode)."""
    S, H = 3, 2
    cpu = torch.device("cpu")
    rstd = torch.zeros(2, S, H, dtype=torch.float32)
    checks = (cake_backend._check_saved_l2n, det_backend._check_saved_l2n)
    passed = 0
    for saved_type in (
        cake_backend.CakeKDANondetChunkSaved,
        det_backend.CakeKDADetChunkSaved,
    ):
        on = _l2n_saved(saved_type, S, H, qk_rstd=rstd, qk_l2norm=True)
        off = _l2n_saved(saved_type, S, H)
        assert (off.qk_rstd, off.qk_l2norm) == (None, False)
        for check in checks:
            for mode in ("fused", "fallback"):
                check(on, mode, S, H, cpu)
                passed += 1
            check(off, "off", S, H, cpu)
            passed += 1
            with pytest.raises(
                ValueError, match="use_qk_l2norm_in_kernel=True; this backward"
            ):
                check(on, "off", S, H, cpu)
            with pytest.raises(
                ValueError, match="use_qk_l2norm_in_kernel=False; this backward"
            ):
                check(off, "fused", S, H, cpu)
            with pytest.raises(ValueError, match="carries none|has none"):
                check(_l2n_saved(saved_type, S, H, qk_l2norm=True), "fused", S, H, cpu)
            with pytest.raises(ValueError, match="without the in-kernel"):
                check(_l2n_saved(saved_type, S, H, qk_rstd=rstd), "off", S, H, cpu)
            for bad in (
                torch.zeros(2, S + 1, H),
                torch.zeros(2, S, H, dtype=torch.float64),
                torch.zeros(2, H, S).transpose(1, 2),
            ):
                with pytest.raises(ValueError):
                    check(
                        _l2n_saved(saved_type, S, H, qk_rstd=bad, qk_l2norm=True),
                        "fused",
                        S,
                        H,
                        cpu,
                    )
    assert passed == 2 * 2 * 3
    # the entry: this package's l2n set reaches the host checks of a call with the flag; the deterministic
    # package's is refused before them
    rstd4 = torch.zeros(2, 4, 1, dtype=torch.float32)
    with pytest.raises(ValueError, match="v: expected 3-D"):
        PACKAGE_CALLS["cake_kda_nondet_train_backward"](
            saved=_l2n_saved(
                cake_backend.CakeKDANondetChunkSaved,
                4,
                1,
                qk_rstd=rstd4,
                qk_l2norm=True,
            ),
            use_qk_l2norm_in_kernel=True,
        )
    det_set = _l2n_saved(
        det_backend.CakeKDADetChunkSaved, 4, 1, qk_rstd=rstd4, qk_l2norm=True
    )
    with pytest.raises(ValueError, match=_refused("saved", det_set)):
        PACKAGE_CALLS["cake_kda_nondet_train_backward"](
            saved=det_set, use_qk_l2norm_in_kernel=True
        )


def test_package_queries_take_the_in_kernel_qk_l2norm_layout() -> None:
    """``workspace_bytes``, ``saved_bytes``, ``output_bytes`` and ``CakeKDANondetWorkspace.for_problem`` take the
    deterministic package's ``l2n``: ``"off"`` gives the bytes of the omitted keyword, an l2n mode (``"fused"`` /
    ``"fallback"``, which runs the deterministic package's records in either gate mode) the deterministic package's
    bytes of that mode, and the saved set adds the reciprocal norms (``8 S H`` bytes); any other value is a
    ``ValueError`` raised before the device is read, a known one reaches the device check."""
    S, N, H = 129, 1, 4
    assert (
        tuple(cake_backend.L2N_MODES)
        == tuple(det_backend.L2N_MODES)
        == (
            "off",
            "fused",
            "fallback",
        )
    )
    assert cake_backend.workspace_bytes(
        S, N, H, sm_count=148, l2n="off"
    ) == cake_backend.workspace_bytes(S, N, H, sm_count=148)
    assert cake_backend.saved_bytes(
        S, H, N, sm_count=148, l2n="off"
    ) == cake_backend.saved_bytes(S, H, N, sm_count=148)
    compared = 0
    for l2n in cake_backend.L2N_MODES:
        for gate in ("g", "raw"):
            if (gate, l2n) == ("g", "off"):
                continue  # the S1 pairs' cells: test_cake_kda_nondet_train.py's byte tests
            for backward in (True, False):
                for save_mode in ("light", "tape"):
                    kw = dict(
                        backward=backward,
                        save_mode=save_mode,
                        sm_count=148,
                        arch="sm_100a",
                        gate=gate,
                        l2n=l2n,
                    )
                    assert cake_backend.workspace_bytes(
                        S, N, H, **kw
                    ) == det_backend.workspace_bytes(S, N, H, **kw), kw
                    compared += 1
            assert cake_backend.output_bytes(
                S, H, backward=True, gate=gate, l2n=l2n
            ) == det_backend.output_bytes(S, H, backward=True, gate=gate, l2n=l2n)
        for save_mode in ("light", "tape"):
            q = dict(save_mode=save_mode, sm_count=148, arch="sm_100a")
            nd = cake_backend.saved_bytes(S, H, N, **q, l2n=l2n)
            off = cake_backend.saved_bytes(S, H, N, **q)
            assert nd == off + (0 if l2n == "off" else 8 * S * H), (l2n, save_mode)
            if l2n != "off":
                assert nd == det_backend.saved_bytes(S, H, N, **q, l2n=l2n)
        with pytest.raises(ValueError, match="CUDA devices"):
            cake_backend.CakeKDANondetWorkspace.for_problem(S, N, H, "cpu", l2n=l2n)
    assert compared == (3 * 2 - 1) * 2 * 2
    for l2n in (None, "OFF", "on", True):
        with pytest.raises(ValueError, match="l2n must be one of"):
            cake_backend.workspace_bytes(S, N, H, sm_count=148, l2n=l2n)
        with pytest.raises(ValueError, match="l2n must be one of"):
            cake_backend.saved_bytes(S, H, N, sm_count=148, l2n=l2n)
        with pytest.raises(ValueError, match="l2n must be one of"):
            cake_backend.CakeKDANondetWorkspace.for_problem(S, N, H, "cpu", l2n=l2n)


def test_facade_runs_the_in_kernel_qk_l2norm_on_the_package() -> None:
    """Without recorders: ``use_qk_l2norm_in_kernel=True`` through the public entries reaches this package, whose
    entries go on to their host checks, whose byte queries give the l2n mode's bytes (the deterministic package's)
    and whose workspace reaches its device check."""
    on = {"use_qk_l2norm_in_kernel": True}
    for name, args in (
        ("cake_kda_nondet_train_forward", _bad_operands()),
        (
            "cake_kda_nondet_train_backward",
            (*_bad_operands(), torch.zeros(4, 1, 128)),
        ),
        ("cake_kda_nondet", _bad_operands()),
    ):
        with pytest.raises(ValueError, match="v: expected 3-D") as info:
            getattr(nd_api, name)(*args, **on)
        assert "cake_kda_det_train" not in str(info.value), name
    for backward in (True, False):
        got = nd_api.cake_kda_nondet_train_workspace_bytes(
            129, 1, 4, sm_count=148, backward=backward, **on
        )
        assert got == cake_backend.workspace_bytes(
            129, 1, 4, sm_count=148, backward=backward, l2n="fused"
        )
        assert got == det_api.cake_kda_det_train_workspace_bytes(
            129, 1, 4, sm_count=148, backward=backward, **on
        )
    got = nd_api.cake_kda_nondet_train_saved_bytes(129, 4, 1, sm_count=148, **on)
    assert got == cake_backend.saved_bytes(129, 4, 1, sm_count=148, l2n="fused")
    assert got == det_api.cake_kda_det_train_saved_bytes(129, 4, 1, sm_count=148, **on)
    with pytest.raises(ValueError, match="CUDA devices"):
        nd_api.cake_kda_nondet_train_workspace(129, 1, 4, "cpu", **on)


def test_package_exports_the_s1_records() -> None:
    """The package exports its S1 records (the registry's), the backend keeps the in-kernel Q/K L2 norm modes, as
    in the deterministic package."""
    assert "S1_RECORDS" in pkg.__all__ and pkg.S1_RECORDS is cake_jit.S1_RECORDS
    assert set(cake_jit.S1_RECORDS) == S1_RECORDS
    assert "L2N_MODES" in cake_backend.__all__ and "L2N_MODES" not in pkg.__all__


# --------------------------------------------------------------------------- the dispatch sweep (CPU)

SWEEP_ROWS = ((4096, 1, 148), (65536, 300, 160))  # (S, num_docs, sm_count)
PAIR_STAGES = ("dqkg", "glm_intra", "gate_epilogue")


def _serves(name: str, l2n: str) -> bool:
    """Whether record ``name`` serves the in-kernel Q/K L2 norm mode ``l2n``: a record of an l2n stage serves the
    modes its ``l2n`` lists (``["off"]`` when absent), every other record every mode."""
    record = cake_jit.MODULES[name]
    return record["stage"] not in cake_jit.L2N_STAGES or l2n in record.get(
        "l2n", ["off"]
    )


def _admitted(stage: str, gate: str = "g", l2n: str = "off", **scalars) -> list[str]:
    """Every record of ``stage`` of the gate class ``gate`` serving the mode ``l2n`` whose rule admits the scalars,
    the ``otherwise`` fallback included."""
    num_segments = cake_jit.segment_plan(
        scalars["num_seqs"], scalars["num_heads"], scalars["sm_count"], scalars["arch"]
    )[2]
    return [
        name
        for name in cake_jit.records_of_stage(stage, gate)
        if _serves(name, l2n)
        and cake_jit._admits(
            cake_jit.MODULES[name]["select_rule"],
            ctas=scalars["num_chunks_pad"] * scalars["num_heads"],
            num_seqs=scalars["num_seqs"],
            num_heads=scalars["num_heads"],
            sm_count=scalars["sm_count"],
            num_segments=num_segments,
            arch=scalars["arch"],
            path=scalars["path"],
            l2n=l2n,
        )
    ]


def test_the_pairs_agree_on_every_head_count() -> None:
    """Over architecture x path x heads 1..128 (two packings each): the plan's record of every stage is the
    registry's, the only one whose rule admits the problem (or the stage's fallback); on every backward path the
    fused gradient and the GLM intra program are both S1 records or neither, exactly on the cells of the independent
    table; every g-class record of the pair stages runs on architectures it ships for, and every program of the pair
    stages (the S1 records, the deterministic copies and the raw class, which shares four fused gradient programs
    with the g class) runs on exactly the architectures it ships for, in g or raw mode."""
    used: dict[str, set] = {}
    raw_used: dict[str, set] = {}
    for arch in cake_jit.ARCHES:
        for H in range(1, 129):
            for S, N, sm in SWEEP_ROWS:
                p = cake_backend.plan(
                    S, N, H, sm, arch
                )  # the plan's pairing guard runs here
                for path, by_stage in p.records.items():
                    for stage, name in by_stage.items():
                        scalars = dict(
                            num_chunks_pad=p.NC_pad,
                            num_seqs=N,
                            num_heads=H,
                            sm_count=sm,
                            arch=arch,
                            path=path,
                        )
                        assert cake_jit.select_record(stage, **scalars) == name
                        admitted = _admitted(stage, **scalars)
                        rules = [
                            cake_jit.MODULES[n]["select_rule"]["kind"] for n in admitted
                        ]
                        positive = [
                            n
                            for n, k in zip(admitted, rules, strict=True)
                            if k != "otherwise"
                        ]
                        if positive:
                            assert positive == [name], (
                                stage,
                                arch,
                                path,
                                H,
                                N,
                                sm,
                                admitted,
                            )
                        else:
                            assert admitted == ([name] if name else []), (
                                stage,
                                arch,
                                path,
                                H,
                            )
                        if name is not None and stage in PAIR_STAGES:
                            assert arch in cake_jit.MODULES[name]["arches"], (
                                name,
                                arch,
                            )
                            used.setdefault(name, set()).add(arch)
                        raw = (
                            cake_jit.select_record(stage, **scalars, gate="raw")
                            if stage in PAIR_STAGES
                            else None
                        )
                        if raw is not None:
                            assert arch in cake_jit.MODULES[raw]["arches"], (raw, arch)
                            raw_used.setdefault(raw, set()).add(arch)
                    if "dqkg" not in by_stage:
                        continue
                    s1 = _s1(arch, path, H)
                    tag = (arch, path, H, N, sm)
                    assert (by_stage["dqkg"] in S1_RECORDS) is s1, tag
                    assert (by_stage["glm_intra"] in S1_RECORDS) is s1, tag
                    if s1:
                        assert by_stage["glm_intra"] == S1_GLM[arch], tag
                    # the gate epilogue follows either GLM intra program of the unfused architecture
                    assert (by_stage["gate_epilogue"] is not None) is (
                        arch == "sm_100a"
                    ), tag
    # the flag-off classes (the in-kernel Q/K L2 norm's are swept by test_the_l2n_modes_select_the_deterministic_pair)
    for stage in PAIR_STAGES:
        assert {n for n in cake_jit.records_of_stage(stage) if _serves(n, "off")} == {
            name for name in used if cake_jit.MODULES[name]["stage"] == stage
        }, stage
        assert {
            n for n in cake_jit.records_of_stage(stage, "raw") if _serves(n, "off")
        } == {name for name in raw_used if cake_jit.MODULES[name]["stage"] == stage}, (
            stage
        )
    # every program of the pair stages (the S1 records, the deterministic copies and the raw class) runs on every
    # architecture it ships for, in one mode or the other: no shipped build of these stages is unreachable
    reached = {
        (cake_jit.MODULES[name]["library"], arch)
        for runs in (used, raw_used)
        for name, arches in runs.items()
        for arch in arches
    }
    shipped = {
        (cake_jit.MODULES[name]["library"], arch)
        for stage in PAIR_STAGES
        for gate in ("g", "raw")
        for name in cake_jit.records_of_stage(stage, gate)
        if _serves(name, "off")
        for arch in cake_jit.MODULES[name]["arches"]
    }
    assert reached == shipped


# the raw gate's parameter finalize of the l2n modes: the deterministic param_finalize record (its program shared)
L2N_PARAM = {"param_finalize_l2n": "param_finalize"}


def _det_name(name: str | None) -> str | None:
    """The deterministic record a record of this package copies: the fused gradient's raw and l2n classes are the
    deterministic ``dqkg`` records with a class suffix, ``param_finalize_l2n`` is ``param_finalize``; every other copy
    keeps its name."""
    if name in L2N_PARAM:
        return L2N_PARAM[name]
    if name is not None and name.startswith("dqkg"):
        return name.removesuffix("_raw").removesuffix("_l2n")
    return name


def test_the_l2n_modes_select_the_deterministic_pair() -> None:
    """Over architecture x path x heads 1..128 (two packings each), both gate modes and both in-kernel Q/K L2 norm
    modes (``"fused"``, ``"fallback"``): the plan's records are the deterministic package's plan of the same mode,
    record for record (the fused gradient's raw and l2n classes and ``param_finalize_l2n`` by their deterministic
    names), never an S1 record,
    never the gate epilogue; on the l2n stages each is the only record of its mode class whose rule admits the
    problem; and every program of the pair stages serving an l2n mode runs on exactly the architectures it ships
    for, but the deterministic glm_intra_l2n's sm_103a build (kept beside its sm_103a form)."""
    reached: set = set()
    cells = 0
    for arch in cake_jit.ARCHES:
        for H in range(1, 129):
            for S, N, sm in SWEEP_ROWS:
                for gate in ("g", "raw"):
                    for l2n in ("fused", "fallback"):
                        p = cake_backend.plan(S, N, H, sm, arch, gate=gate, l2n=l2n)
                        dp = det_backend.plan(S, N, H, sm, arch, gate=gate, l2n=l2n)
                        assert (p.gate, p.l2n) == (dp.gate, dp.l2n) == (gate, l2n)
                        assert set(p.records) == set(dp.records)
                        for path, by_stage in p.records.items():
                            tag = (arch, H, N, sm, gate, l2n, path)
                            assert {
                                stage: _det_name(name)
                                for stage, name in by_stage.items()
                            } == dp.records[path], tag
                            assert not set(by_stage.values()) & S1_RECORDS, tag
                            assert by_stage.get("gate_epilogue") is None, tag
                            for stage, name in by_stage.items():
                                if stage not in cake_jit.L2N_STAGES:
                                    continue
                                scalars = dict(
                                    num_chunks_pad=p.NC_pad,
                                    num_seqs=N,
                                    num_heads=H,
                                    sm_count=sm,
                                    arch=arch,
                                    path=path,
                                )
                                assert (
                                    cake_jit.select_record(
                                        stage, **scalars, gate=gate, l2n=l2n
                                    )
                                    == name
                                ), (*tag, stage)
                                admitted = _admitted(stage, gate, l2n, **scalars)
                                positive = [
                                    n
                                    for n in admitted
                                    if cake_jit.MODULES[n]["select_rule"]["kind"]
                                    != "otherwise"
                                ]
                                assert (positive or admitted) == (
                                    [name] if name else []
                                ), (*tag, stage, admitted)
                                if name is not None and stage in PAIR_STAGES:
                                    assert arch in cake_jit.MODULES[name]["arches"]
                                    reached.add(
                                        (cake_jit.MODULES[name]["library"], arch)
                                    )
                            cells += 1
    assert cells > 0
    # the gate epilogue never runs in an l2n mode (its GLM intra programs write the gate gradients themselves)
    shipped = {
        (cake_jit.MODULES[name]["library"], arch)
        for stage in ("dqkg", "glm_intra")
        for gate in ("g", "raw")
        for name in cake_jit.records_of_stage(stage, gate)
        if _serves(name, "fused") or _serves(name, "fallback")
        for arch in cake_jit.MODULES[name]["arches"]
    }
    # every reached build ships; every shipped one is reached but the deterministic glm_intra_l2n's sm_103a build,
    # which the deterministic package keeps while its sm_103a form glm_intra_v30_l2n serves that architecture
    kept = {(cake_jit.MODULES["glm_intra_l2n"]["library"], "sm_103a")}
    assert reached <= shipped
    assert shipped - reached == kept


def test_the_plan_refuses_a_mixed_pair(monkeypatch) -> None:
    """A registry whose fused gradient and GLM intra program disagree on S1 on some path is refused by the plan:
    swapping the rules of the sm_103a GLM intra programs pairs the S1 fused gradient with the deterministic GLM
    intra program on the saved-set paths of a 4-head row."""
    v14, v22 = cake_jit.MODULES["glm_intra_v14"], cake_jit.MODULES["glm_intra_v22"]
    cake_backend._plan.cache_clear()
    try:
        monkeypatch.setitem(v14, "select_rule", v22["select_rule"])
        monkeypatch.setitem(
            v22, "select_rule", cake_jit.MODULES["glm_intra_v18"]["select_rule"]
        )
        with pytest.raises(ValueError, match="S1_RECORDS"):
            cake_backend.plan(4096, 1, 4, 148, "sm_103a")
    finally:
        monkeypatch.undo()
        cake_backend._plan.cache_clear()
    assert (
        cake_backend.plan(4096, 1, 4, 148, "sm_103a").records["light"]["glm_intra"]
        == (S1_GLM["sm_103a"])
    )


# --------------------------------------------------------------------------- the public entries (GPU)


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    if (
        cake_jit.arch_for_capability(tuple(torch.cuda.get_device_capability(index)))
        is None
    ):
        pytest.skip("not an exported compute capability")
    for module in (cake_jit, det_jit):
        if not (Path(module.__file__).resolve().parent / "build.json").exists():
            pytest.skip("a package has no build.json yet (not exported)")
    return torch.device("cuda", index)


def _operands(inputs):
    return tuple(inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout"))


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    view = torch.int16 if a.element_size() == 2 else torch.int32
    return torch.equal(a.contiguous().view(view), b.contiguous().view(view))


def _same(got, want) -> bool:
    return all(_bits_equal(a, b) for a, b in zip(got, want, strict=True))


# a row that runs the S1 pair from its saved set on every architecture (16 heads) and one that runs it on the replay
# of every architecture (2 heads, segmented)
GPU_ROWS = (((129, 63), 16), ((63, 1, 65), 2))


def _leaf_grads(grads):
    """The backward's ``(dq, dk, dv, dbeta, dg)`` in the order of the leaves ``(q, k, v, g, beta)``."""
    return (grads[0], grads[1], grads[2], grads[4], grads[3])


@pytest.mark.parametrize("shape", GPU_ROWS, ids=["h16", "h2"])
def test_public_entries_run_the_package(shape) -> None:
    """The public entries give this package's bits (its entries' at the default scale) with and without a caller
    workspace (the facade's, this package's type, of the byte query's size) and through the autograd node (this
    package's); ``out``, ``dq``, ``dk``, ``dv`` and ``dbeta`` are the deterministic package's bits.  A workspace
    serves its own package: this package's entries refuse the deterministic package's workspace and the
    deterministic entries refuse this package's (``ValueError`` naming the mode)."""
    device = _device_or_skip()
    lengths, heads = shape
    S, N = sum(lengths), len(lengths)
    q, k, v, g, beta, cu, dout = _operands(
        make_inputs(lengths, heads, device, batched=True)
    )
    scale = nd_api.DEFAULT_SCALE
    det_out, det_saved = det_backend.cake_kda_det_train_forward(
        q, k, v, g, beta, cu, scale=scale
    )
    det_grads = det_backend.cake_kda_det_train_backward(
        q, k, v, g, beta, cu, dout, saved=det_saved, scale=scale
    )
    out_want, saved_want = cake_backend.cake_kda_nondet_train_forward(
        q, k, v, g, beta, cu, scale=scale
    )
    grads_want = [
        t.clone()
        for t in cake_backend.cake_kda_nondet_train_backward(
            q, k, v, g, beta, cu, dout, saved=saved_want, scale=scale
        )
    ]
    assert _bits_equal(out_want, det_out) and _same(grads_want[:4], det_grads[:4])
    ws = nd_api.cake_kda_nondet_train_workspace(S, N, heads, device)
    assert type(ws) is cake_backend.CakeKDANondetWorkspace
    assert ws.bytes == nd_api.cake_kda_nondet_train_workspace_bytes(S, N, heads)
    for workspace in (None, ws):
        out, saved = nd_api.cake_kda_nondet_train_forward(
            q, k, v, g, beta, cu, workspace=workspace
        )
        assert type(saved) is cake_backend.CakeKDANondetChunkSaved
        assert _bits_equal(out, out_want), workspace is None
        grads = nd_api.cake_kda_nondet_train_backward(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=workspace
        )
        assert _same(grads, grads_want), workspace is None
    leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)]
    o = nd_api.cake_kda_nondet(*leaves, cu)
    assert o.grad_fn._forward_cls is cake_backend.CakeKDANondetFunction
    assert _bits_equal(o, out_want)
    o.backward(dout)
    for leaf, want in zip(leaves, _leaf_grads(grads_want), strict=True):
        assert _bits_equal(leaf.grad, want)
    det_ws = det_api.cake_kda_det_train_workspace(S, N, heads, device)
    for call in (
        lambda: nd_api.cake_kda_nondet_train_forward(
            q, k, v, g, beta, cu, workspace=det_ws
        ),
        lambda: nd_api.cake_kda_nondet_train_backward(
            q, k, v, g, beta, cu, dout, saved=saved_want, workspace=det_ws
        ),
        lambda: nd_api.cake_kda_nondet(q, k, v, g, beta, cu, workspace=det_ws),
    ):
        with pytest.raises(ValueError, match=_refused("workspace", det_ws)):
            call()
    with pytest.raises(ValueError, match=DET_REFUSES):
        det_api.cake_kda_det_train_forward(q, k, v, g, beta, cu, workspace=ws)


@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn_only"])
def test_torch_deterministic_mode_runs_this_package(
    warn_only: bool, restore_torch_deterministic, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under ``torch.use_deterministic_algorithms(True)`` (strict, and with ``warn_only=True``) the public entries run
    this package: the forward's saved set is a ``CakeKDANondetChunkSaved``, the autograd node a
    ``CakeKDANondetFunction``, every launch of a caller workspace a record of this package's registry (a program of
    its own library prefix) and the deterministic package launches nothing; the results are the bits of the calls
    with torch's mode off, ``dg`` included, with and without a caller workspace; no call raises and none warns (no
    warning from flashinfer code, none naming a missing deterministic implementation)."""
    device = _device_or_skip()
    lengths, heads = (129, 63), 16
    S, N = sum(lengths), len(lengths)
    q, k, v, g, beta, cu, dout = _operands(make_inputs(lengths, heads, device))
    out_want, saved = nd_api.cake_kda_nondet_train_forward(q, k, v, g, beta, cu)
    grads_want = [
        t.clone()
        for t in nd_api.cake_kda_nondet_train_backward(
            q, k, v, g, beta, cu, dout, saved=saved
        )
    ]
    launched: list = []

    def recording(name, values, *, grid, device_index):
        launched.append(name)
        cake_jit.run_module(name, values, grid=grid, device_index=device_index)

    ws = cake_backend.CakeKDANondetWorkspace.for_problem(
        S, N, heads, device, launcher=recording
    )

    def refuse(*_args, **_kwargs):
        raise AssertionError("the deterministic package launched a kernel")

    monkeypatch.setattr(det_jit, "run_module", refuse)
    monkeypatch.setattr(det_jit, "load", refuse)
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for workspace in (None, ws):
            launched.clear()
            out, saved = nd_api.cake_kda_nondet_train_forward(
                q, k, v, g, beta, cu, workspace=workspace
            )
            assert type(saved) is cake_backend.CakeKDANondetChunkSaved
            grads = nd_api.cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=saved, workspace=workspace
            )
            torch.cuda.synchronize(device)
            assert _bits_equal(out, out_want), workspace is None
            assert _same(grads, grads_want), workspace is None
            if workspace is not None:
                # the caller workspace ran the forward and the backward (the checks above are not vacuous)
                assert launched
                for name in launched:
                    library = cake_jit.MODULES[name]["library"]
                    assert library.startswith("cake_kda_train_"), (name, library)
        leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)]
        o = nd_api.cake_kda_nondet(*leaves, cu)
        assert o.grad_fn._forward_cls is cake_backend.CakeKDANondetFunction
        o.backward(dout)
        torch.cuda.synchronize(device)
    assert torch.are_deterministic_algorithms_enabled()
    assert _bits_equal(o, out_want)
    for leaf, want in zip(leaves, _leaf_grads(grads_want), strict=True):
        assert _bits_equal(leaf.grad, want), warn_only
    assert _flashinfer_warnings(rec) == []


def test_autograd_node_keeps_its_package_under_torch_deterministic_mode() -> None:
    """The autograd node is the forward's package: this facade's ``cake_kda_nondet`` gives this package's node, whose
    backward runs this package even when torch's deterministic mode is switched on after the forward (no warning,
    no alert); the deterministic facade's ``cake_kda_det`` gives the deterministic package's node, whose backward
    runs that package."""
    device = _device_or_skip()
    lengths, heads = (129, 63), 16
    q, k, v, g, beta, cu, dout = _operands(make_inputs(lengths, heads, device))
    scale = nd_api.DEFAULT_SCALE
    want = {}
    for det, fwd, bwd in (
        (
            False,
            cake_backend.cake_kda_nondet_train_forward,
            cake_backend.cake_kda_nondet_train_backward,
        ),
        (
            True,
            det_backend.cake_kda_det_train_forward,
            det_backend.cake_kda_det_train_backward,
        ),
    ):
        _, saved = fwd(q, k, v, g, beta, cu, scale=scale)
        want[det] = [
            t.clone() for t in bwd(q, k, v, g, beta, cu, dout, saved=saved, scale=scale)
        ]
    for det, entry, node in (
        (False, nd_api.cake_kda_nondet, cake_backend.CakeKDANondetFunction),
        (True, det_api.cake_kda_det, det_backend.CakeKDADetFunction),
    ):
        leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)]
        o = entry(*leaves, cu)
        assert o.grad_fn._forward_cls is node, det
        with (
            _torch_deterministic(warn_only=True),
            warnings.catch_warnings(record=True) as rec,
        ):
            warnings.simplefilter("always")
            o.backward(dout)
            torch.cuda.synchronize(device)
        assert _flashinfer_warnings(rec) == [], det
        for leaf, ref in zip(leaves, _leaf_grads(want[det]), strict=True):
            assert _bits_equal(leaf.grad, ref), det


def test_each_backward_refuses_the_other_packages_saved_set() -> None:
    """This package's forward writes its saved set in the deterministic package's format (the same version, ``Aqk``
    / ``Akk`` bits and extras) under its own type: each package's backward reads its own sets and refuses the other
    package's (``ValueError`` naming the mode), the light and the tape set alike; ``dq``, ``dk``, ``dv`` and
    ``dbeta`` are the deterministic package's bits."""
    device = _device_or_skip()
    lengths, heads = (63, 1, 65), 2
    q, k, v, g, beta, cu, dout = _operands(make_inputs(lengths, heads, device))
    scale = nd_api.DEFAULT_SCALE
    for save_mode in (None, "tape"):
        nd_out, nd_saved = nd_api.cake_kda_nondet_train_forward(
            q, k, v, g, beta, cu, scale=scale, save_mode=save_mode
        )
        det_out, det_saved = det_api.cake_kda_det_train_forward(
            q, k, v, g, beta, cu, scale=scale, save_mode=save_mode
        )
        assert type(nd_saved) is cake_backend.CakeKDANondetChunkSaved
        assert _bits_equal(nd_out, det_out)
        assert nd_saved.version == det_saved.version
        assert _bits_equal(nd_saved.Aqk, det_saved.Aqk)
        assert _bits_equal(nd_saved.Akk, det_saved.Akk)
        for name in ("qg", "w", "u", "h", "v_new", "seg_m", "hseg", "seg_state"):
            a, b = getattr(nd_saved, name), getattr(det_saved, name)
            assert (a is None) == (b is None), (save_mode, name)
        own = [
            t.clone()
            for t in nd_api.cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=nd_saved, scale=scale
            )
        ]
        with pytest.raises(ValueError, match=_refused("saved", det_saved)):
            nd_api.cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=det_saved, scale=scale
            )
        det_own = det_api.cake_kda_det_train_backward(
            q, k, v, g, beta, cu, dout, saved=det_saved, scale=scale
        )
        with pytest.raises(ValueError, match=DET_REFUSES):
            det_api.cake_kda_det_train_backward(
                q, k, v, g, beta, cu, dout, saved=nd_saved, scale=scale
            )
        # dq, dk, dv and dbeta are the deterministic package's bits (dg differs by FP32 rounding on the S1 rows)
        assert _same(own[:4], det_own[:4]), save_mode


def test_public_entries_run_the_raw_gate_like_the_deterministic_package() -> None:
    """The raw gate mode through the public entries: this facade and the deterministic facade give the same ``out``
    and seven results bit for bit (explicit calls with and without a workspace of the raw mode, the autograd node:
    this package runs the deterministic package's raw-mode programs); each facade's raw-mode workspace serves its
    package, with the byte query's size; a workspace of the g mode is refused by a raw call (``ValueError``)."""
    device = _device_or_skip()
    lengths, heads = (63, 1, 65), 2
    S, N = sum(lengths), len(lengths)
    inputs = make_raw_inputs(lengths, heads, device)
    q, k, v, raw, beta, cu, dout = _operands(inputs)
    kw = {
        "use_gate_in_kernel": True,
        "A_log": inputs["A_log"],
        "dt_bias": inputs["dt_bias"],
        "lower_bound": LOWER_BOUND,
    }
    results = {}
    for label, api, prefix, own in (
        ("det", det_api, "cake_kda_det", det_backend.CakeKDADetWorkspace),
        ("nd", nd_api, "cake_kda_nondet", cake_backend.CakeKDANondetWorkspace),
    ):
        workspace_of = getattr(api, f"{prefix}_train_workspace")
        ws = workspace_of(S, N, heads, device, use_gate_in_kernel=True)
        assert type(ws) is own and ws.gate == "raw", label
        assert ws.bytes == getattr(api, f"{prefix}_train_workspace_bytes")(
            S, N, heads, use_gate_in_kernel=True
        )
        runs = []
        for workspace in (None, ws):
            out, saved = getattr(api, f"{prefix}_train_forward")(
                q, k, v, raw, beta, cu, workspace=workspace, **kw
            )
            assert saved.gate == "raw", label
            grads = getattr(api, f"{prefix}_train_backward")(
                q, k, v, raw, beta, cu, dout, saved=saved, workspace=workspace, **kw
            )
            assert len(grads) == 7, label
            runs.append((out.clone(), [t.clone() for t in grads]))
        leaves = [
            t.detach().clone().requires_grad_(True)
            for t in (q, k, v, raw, beta, inputs["A_log"], inputs["dt_bias"])
        ]
        o = getattr(api, prefix)(
            *leaves[:5],
            cu,
            use_gate_in_kernel=True,
            A_log=leaves[5],
            dt_bias=leaves[6],
            lower_bound=LOWER_BOUND,
        )
        o.backward(dout)
        # the leaves' order (q, k, v, raw, beta, A_log, dt_bias) -> the backward's
        gr = [leaf.grad for leaf in leaves]
        runs.append((o.detach(), (gr[0], gr[1], gr[2], gr[4], gr[3], gr[5], gr[6])))
        g_ws = workspace_of(S, N, heads, device)
        with pytest.raises(ValueError, match="gate mode"):
            getattr(api, f"{prefix}_train_forward")(
                q, k, v, raw, beta, cu, workspace=g_ws, **kw
            )
        results[label] = runs
    want_out, want_grads = results["det"][0]
    for label, runs in results.items():
        for i, (out, grads) in enumerate(runs):
            assert _bits_equal(out, want_out), (label, i)
            assert _same(grads, want_grads), (label, i)
