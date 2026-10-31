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

The single-rank registry (``cake_jit``) of the non-deterministic causal convolution training package against the
deterministic package's: the forward records are the deterministic forward programs (only their names differ),
every backward route keeps the deterministic route's programs and selection, each backward program is ONE
ordinary launch (no reduce program) that owns an ``accumulators`` workspace (FP32, or FP64 on the ``_fp32g``
routes; ``[C, 4]`` taps then ``[C]`` bias words; uint32 ticket counters per channel tile), and the workspace byte
layout and size depend on ``C`` and the route only.  The ``deterministic`` keyword of every public entry point,
and the package's wiring (test-helper key, package data, the docstring-only ``flashinfer.experimental.cake``).
No device is needed; nothing is built or launched.  The ``build.json`` checks skip until the exporter renders the
package.
"""

from __future__ import annotations

import ast
import filecmp
import hashlib
import importlib.util
import inspect
import re
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

KEY = "cake.cake_causal_conv1d_train"
pkg = cake_package(KEY)
cake_build = cake_package(KEY, "cake_build")
cake_jit = cake_package(KEY, "cake_jit")
kcp_jit = cake_package(KEY, "kcp_jit")
det_build = cake_package("cake_causal_conv1d_det_train", "cake_build")
det_jit = cake_package("cake_causal_conv1d_det_train", "cake_jit")

PACKAGE_DIR = Path(cake_jit.__file__).parent
REPO = Path(__file__).resolve().parents[2]
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
NAME_FIELDS = ("kernel", "kernel_symbol", "library")
# A placeholder name until the exporter renders the package, then the identity rule's prefix + 20 hex digits.
PROGRAM_NAME = re.compile(
    r"cake_causal_conv1d_train_(?:TODOEXPORT_[a-z0-9_]+|[0-9a-f]{20})"
)
CHANNELS = (
    8,
    128,
    136,
    256,
    512,
    768,
    1024,
    1160,
    1280,
    1344,
    1536,
    1600,
    1920,
    2048,
    2560,
    4096,
    8192,
)
# 262143 / 262144: both sides of the sm_107a token-floored backward program's floor.
TOKENS = (1, 1000, 65536, 98304, 147456, 250340, 262143, 262144)


@contextmanager
def warnings_recorded():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def _strip(record: dict, *fields: str) -> dict:
    return {key: value for key, value in record.items() if key not in fields}


def _args(record: dict) -> set[str]:
    return {name for kind, name in record["arg_plan"] if kind != "grid"}


def _pairs(role: str):
    """``(arch, kind, new record name, det record name)`` of every route of ``role``, in route order."""

    for arch in ARCHES:
        for kind in cake_jit.KINDS:
            if cake_jit.route_role(kind) != role:
                continue
            new, det = cake_jit.ROUTES[arch][kind], det_jit.ROUTES[arch][kind]
            assert len(new) == len(det), (arch, kind)
            for name, det_name in zip(new, det, strict=True):
                yield arch, kind, name, det_name


@pytest.mark.cpu_only
def test_loader_is_the_det_loader() -> None:
    """``cake_build.py`` is byte for byte the deterministic package's (one manifest-driven loader)."""

    assert filecmp.cmp(
        Path(cake_build.__file__), Path(det_build.__file__), shallow=False
    )


@pytest.mark.cpu_only
def test_routes_mirror_the_deterministic_package() -> None:
    assert cake_jit.ARCHES == det_jit.ARCHES == ARCHES
    assert cake_jit.KINDS == det_jit.KINDS
    assert cake_jit.REQUIRED_KINDS == det_jit.REQUIRED_KINDS
    assert cake_jit.ROLES == ("fwd", "bwd")
    assert cake_jit.GRID_KINDS == ("co_resident", "occupancy")
    for arch in ARCHES:
        assert set(cake_jit.ROUTES[arch]) == set(det_jit.ROUTES[arch]), arch
    routed = {
        name
        for by_kind in cake_jit.ROUTES.values()
        for names in by_kind.values()
        for name in names
    }
    # No reduce program: every record is a routed forward or backward program.
    assert set(cake_jit.MODULES) == routed
    assert not set(cake_jit.MODULES) & set(det_jit.MODULES)
    for name, record in cake_jit.MODULES.items():
        assert PROGRAM_NAME.fullmatch(name), name
        assert record["library"] == record["kernel"] == name
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert record["role"] in cake_jit.ROLES and "reduce" not in record, name
        assert record["grid_rule"]["kind"] in cake_jit.GRID_KINDS, name
        assert record["grid_rule"]["cooperative"] is False, name
        assert record["launch"]["cooperative"] is False, name


@pytest.mark.cpu_only
def test_forward_records_are_the_deterministic_forward_records() -> None:
    """The forward programs are the deterministic ones (same module, build keywords, plan, launch): ``y`` keeps
    its bits.  Only the three name fields differ."""

    for arch, kind, name, det_name in _pairs("fwd"):
        record, det = cake_jit.MODULES[name], det_jit.MODULES[det_name]
        assert _strip(record, *NAME_FIELDS) == _strip(det, *NAME_FIELDS), (
            arch,
            kind,
            name,
        )


@pytest.mark.cpu_only
def test_backward_records_are_one_launch_with_accumulators() -> None:
    """Every backward program replaces the deterministic program + reduce pair of the same route position: the
    deterministic arguments without ``partial``, plus ``dw_acc`` / ``work`` and the gradient outs it now stores
    (``dw``, and ``db`` on the bias routes); the same build geometry, block, cluster, architectures and selection
    rule; an ordinary occupancy-sized launch without PDL; an ``accumulators`` workspace of the route's dtype and
    tap count."""

    vocabulary = set(cake_jit.LOGICAL_TENSORS) | set(cake_jit.LOGICAL_SCALARS)
    for arch, kind, name, det_name in _pairs("bwd"):
        record, det = cake_jit.MODULES[name], det_jit.MODULES[det_name]
        bias = "bias" in kind
        label = (arch, kind, name)
        assert record["role"] == "bwd" and record["arches"] == det["arches"], label
        assert record["ffi_entry"] == det["ffi_entry"], label
        expected = (
            (_args(det) - {"partial"})
            | {"dw_acc", "work", "dw"}
            | ({"db"} if bias else set())
        )
        assert _args(record) == expected, label
        assert [k for k, _ in record["arg_plan"][-3:]] == ["grid", "grid", "grid"], (
            label
        )
        assert set(record["logical"]) == _args(record), label
        assert set(record["logical"].values()) <= vocabulary, label
        assert (
            record["logical"].get("x_map", "x") == "x"
            and record["logical"].get("dy_map", "dy") == "dy"
        )
        # The global accumulators: kernel parameter ``dw_acc`` (the det fragments' ``acc`` is a register), logical ``acc``.
        assert (
            record["logical"]["dw_acc"] == "acc" and record["logical"]["work"] == "work"
        ), label
        assert ("flags" in _args(record)) is (kind != "bwd"), label
        grid, det_grid = record["grid_rule"], det["grid_rule"]
        assert grid["kind"] == "occupancy" and grid["cooperative"] is False, label
        assert grid["geometry"] == det_grid["geometry"], label
        assert set(grid["grid_mult"]) == set(record["arches"]), label
        # ``ctas_per_sm`` is the rendered program's occupancy (the exporter's leaf).
        assert isinstance(grid["ctas_per_sm"], int) and grid["ctas_per_sm"] >= 1, label
        launch = record["launch"]
        assert launch["cooperative"] is False and launch["use_pdl"] is False, label
        assert launch["block"] == det["launch"]["block"], label
        assert launch["cluster"] == det["launch"]["cluster"], label
        assert (
            isinstance(launch["dynamic_smem_bytes"], int)
            and launch["dynamic_smem_bytes"] >= 0
        )
        assert cake_jit.select_rule(record) == det_jit.select_rule(det), label
        rule = dict(record["workspace_rule"])
        assert "slots" in rule, label  # required: the package never defaults it
        slots = rule.pop("slots")
        assert rule == {
            "acc": "float64" if kind.endswith("fp32g") else "float32",
            "kind": "accumulators",
            "taps": 5 if bias else 4,
            "work": "uint32",
        }, label
        # Accumulator copies, as the render names them: the fp32g programs may keep several, the FP32 routes one.
        assert isinstance(slots, int) and slots >= 1, label
        if not kind.endswith("fp32g"):
            assert slots == 1, label


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_selection_matches_the_deterministic_package(arch) -> None:
    """A call selects the program at the same route position as the deterministic package for every ``(S, C)``,
    and plans the same scalars (``y`` / ``dx`` keep their per-element arithmetic)."""

    for kind in cake_jit.KINDS:
        new, det = cake_jit.ROUTES[arch][kind], det_jit.ROUTES[arch][kind]
        for C in CHANNELS:
            for S in TOKENS:
                (name,) = cake_jit.selected_modules(arch, kind, C=C, S=S)
                (det_name,) = det_jit.selected_modules(arch, kind, C=C, S=S)
                assert new.index(name) == det.index(det_name), (arch, kind, C, S)
                record, det_record = cake_jit.MODULES[name], det_jit.MODULES[det_name]
                geometry = record["grid_rule"]["geometry"]
                assert cake_jit.plan(
                    S, C, 3, geometry=geometry, role=record["role"]
                ) == det_jit.plan(
                    S,
                    C,
                    3,
                    geometry=det_record["grid_rule"]["geometry"],
                    role=det_record["role"],
                ), (arch, kind, C, S)


def _align16(n: int) -> int:
    return -(-n // 16) * 16


@pytest.mark.cpu_only
def test_accumulator_layout() -> None:
    assert cake_jit.WORKSPACE_ALIGN == 16
    assert cake_jit.ACCUMULATOR_DTYPES == {"float32": 4, "float64": 8}
    assert cake_jit.COUNTER_DTYPES == {"uint32": 4}
    for arch, kind, name, _ in _pairs("bwd"):
        record = cake_jit.MODULES[name]
        rule = record["workspace_rule"]
        size = cake_jit.ACCUMULATOR_DTYPES[rule["acc"]]
        for C in CHANNELS:
            _, _, n_ch_tiles = cake_jit.channel_split(
                C, geometry=record["grid_rule"]["geometry"], role="bwd"
            )
            layout = cake_jit.accumulator_layout(record, C, n_ch_tiles)
            words = rule["slots"] * rule["taps"]
            assert cake_jit.accumulator_words_per_channel(record) == words
            acc_bytes = words * C * size
            assert layout == {
                "acc_dtype": rule["acc"],
                "acc_elems": words * C,
                "acc_bytes": acc_bytes,
                "work_dtype": "uint32",
                "work_offset": _align16(acc_bytes),
                "work_words": n_ch_tiles,
                "total_bytes": _align16(_align16(acc_bytes) + 4 * n_ch_tiles),
            }, (arch, kind, C)
            # The views the host takes stay inside the buffer and do not overlap.
            assert layout["acc_bytes"] <= layout["work_offset"]
            assert layout["work_offset"] + 4 * n_ch_tiles <= layout["total_bytes"]
            assert layout["work_offset"] % size == 0
        with pytest.raises(ValueError):
            cake_jit.accumulator_layout(record, 0, 1)
        with pytest.raises(ValueError):
            cake_jit.accumulator_layout(record, 256, 0)
    forward = cake_jit.MODULES[cake_jit.ROUTES["sm_100a"]["fwd"][0]]
    with pytest.raises(ValueError, match="no backward accumulator workspace"):
        cake_jit.accumulator_layout(forward, 256, 1)


@pytest.mark.cpu_only
@pytest.mark.parametrize("kind", ("bwd_fp32g", "bwd_bias_fp32g"))
@pytest.mark.parametrize("C", (512, 8192))
def test_accumulators_are_sized_from_the_rule(kind, C) -> None:
    """The accumulator size comes from the rendered rule, never a fixed ``C x taps``: an fp32g program whose rule
    names ``slots`` FP64 copies gets ``slots x taps x C`` words; a rule without ``slots``, or with a malformed one,
    is refused."""

    import copy

    for arch in ARCHES:
        if kind not in cake_jit.ROUTES[arch]:
            continue
        name = cake_jit.route_modules(arch, kind)[0]
        record = cake_jit.MODULES[name]
        rule = record["workspace_rule"]
        assert rule["acc"] == "float64"
        _, _, n_ch_tiles = cake_jit.channel_split(
            C, geometry=record["grid_rule"]["geometry"], role="bwd"
        )
        rendered = cake_jit.accumulator_layout(record, C, n_ch_tiles)
        assert rendered["acc_elems"] == rule["slots"] * rule["taps"] * C
        for slots in (1, 4):
            variant = copy.deepcopy(record)
            variant["workspace_rule"]["slots"] = slots
            layout = cake_jit.accumulator_layout(variant, C, n_ch_tiles)
            assert layout["acc_elems"] == slots * rule["taps"] * C, (arch, slots)
            assert layout["acc_bytes"] == 8 * slots * rule["taps"] * C
            assert layout["work_offset"] == _align16(layout["acc_bytes"])
            assert layout["total_bytes"] >= layout["work_offset"] + 4 * n_ch_tiles
        for slots in (0, -1, 1.5, True, "4", None):
            variant = copy.deepcopy(record)
            variant["workspace_rule"]["slots"] = slots
            with pytest.raises(ValueError, match="slots"):
                cake_jit.accumulator_layout(variant, C, n_ch_tiles)
        # A rule without the field is refused (no default could size dw_acc for every program).
        variant = copy.deepcopy(record)
        del variant["workspace_rule"]["slots"]
        with pytest.raises(ValueError, match="slots"):
            cake_jit.accumulator_layout(variant, C, n_ch_tiles)
        with pytest.raises(ValueError, match="slots"):
            cake_jit.accumulator_words_per_channel(variant)
        with pytest.raises(ValueError, match="slots"):
            pkg._accumulator_layout(variant, C, n_ch_tiles)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_backward_workspace_serves_every_backward_program(arch) -> None:
    """The public backward workspace is the largest layout over every backward program of the architecture, so one
    zeroed buffer serves whichever program and route a call selects; the capacity owner is an FP64 bias program
    (8 bytes x slots x 5 taps per channel, 40 with one copy)."""

    owner = cake_jit.workspace_record(arch)
    assert (
        owner["workspace_rule"]["acc"] == "float64"
        and owner["workspace_rule"]["taps"] == 5
    )
    owners = cake_jit.workspace_owners(arch)
    assert len(owners) == sum(
        len(cake_jit.route_modules(arch, kind))
        for kind in cake_jit.backward_kinds(arch)
    )
    for C in CHANNELS:
        nbytes = cake_jit.backward_workspace_bytes(arch, C)
        for record in owners:
            _, _, n_ch_tiles = cake_jit.channel_split(
                C, geometry=record["grid_rule"]["geometry"], role="bwd"
            )
            assert (
                cake_jit.accumulator_layout(record, C, n_ch_tiles)["total_bytes"]
                <= nbytes
            ), (arch, C)
        per_channel = 8 * cake_jit.accumulator_words_per_channel(owner)
        assert nbytes == _align16(_align16(per_channel * C) + 4 * cake_jit.channel_split(
            C, geometry=owner["grid_rule"]["geometry"], role="bwd")[2])  # fmt: skip


# ----------------------------------------------------------------------------------------------------------------
# The numerical-mode keyword.

PUBLIC = (
    "cake_causal_conv1d_train_forward",
    "cake_causal_conv1d_train_backward",
    "cake_causal_conv1d",
    "cake_causal_conv1d_train_forward_workspace",
    "cake_causal_conv1d_train_forward_workspace_bytes",
    "cake_causal_conv1d_train_backward_workspace",
    "cake_causal_conv1d_train_backward_workspace_bytes",
)


@pytest.mark.cpu_only
def test_entry_points_take_the_deterministic_keyword() -> None:
    for name in PUBLIC:
        p = inspect.signature(getattr(pkg, name)).parameters["deterministic"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is True, name
        assert name in pkg.__all__
    p = inspect.signature(pkg.CakeCausalConv1dFunction.forward).parameters[
        "deterministic"
    ]
    assert p.default is True
    assert "CakeCausalConv1dFunction" in pkg.__all__


@pytest.mark.cpu_only
def test_entry_points_refuse_a_deterministic_mode_first() -> None:
    """``deterministic`` other than the bool ``False`` is refused (``ValueError`` naming the deterministic package)
    before any argument, device or rendering check, so the refusal needs neither a device nor ``build.json``."""

    x = torch.empty(8, 16, dtype=torch.bfloat16)
    w = torch.empty(16, 4, dtype=torch.float32)
    cu = torch.tensor([0, 8], dtype=torch.int32)
    cpu = torch.device("cpu")
    calls = {
        "forward": lambda d: pkg.cake_causal_conv1d_train_forward(x, w, cu, **d),
        "backward": lambda d: pkg.cake_causal_conv1d_train_backward(x, w, cu, x, **d),
        "autograd": lambda d: pkg.cake_causal_conv1d(x, w, cu, **d),
        "function": lambda d: pkg.CakeCausalConv1dFunction.apply(x, w, cu, None, "bf16", d.get("deterministic", True)),
        "fwd_ws": lambda d: pkg.cake_causal_conv1d_train_forward_workspace(8, 16, cpu, **d),
        "fwd_ws_bytes": lambda d: pkg.cake_causal_conv1d_train_forward_workspace_bytes(8, 16, cpu, **d),
        "bwd_ws": lambda d: pkg.cake_causal_conv1d_train_backward_workspace(8, 16, cpu, **d),
        "bwd_ws_bytes": lambda d: pkg.cake_causal_conv1d_train_backward_workspace_bytes(8, 16, cpu, **d),
    }  # fmt: skip
    for name, call in calls.items():
        for kwargs in (
            {},
            {"deterministic": True},
            {"deterministic": 0},
            {"deterministic": None},
            {"deterministic": "False"},
        ):
            with pytest.raises(ValueError, match="deterministic=False") as info:
                call(kwargs)
            assert "flashinfer.cake_causal_conv1d_det_train" in str(info.value), name
        # With deterministic=False the CPU tensors / device fail the next check instead.
        with pytest.raises((ValueError, RuntimeError)) as info:
            call({"deterministic": False})
        assert "deterministic=False" not in str(info.value), name


@pytest.mark.cpu_only
def test_alert_helper_in_both_torch_modes(monkeypatch) -> None:
    """The backend's own non-determinism alert (no private PyTorch hook: ``torch._C._alert_not_deterministic`` is
    absent from some PyTorch builds, so it is removed here): silent with torch's deterministic algorithms off, a
    ``RuntimeError`` naming the entry and the deterministic package when on, a ``UserWarning`` with the same naming
    under ``warn_only=True``.  The KCP module uses the same helper."""

    from flashinfer.experimental.cake.cake_causal_conv1d_train_backend import (
        kcp as conv_kcp,
    )

    assert conv_kcp._alert_nondeterministic is pkg._alert_nondeterministic
    monkeypatch.delattr(torch._C, "_alert_not_deterministic", raising=False)
    prev, prev_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(False)
        with warnings_recorded() as caught:
            pkg._alert_nondeterministic("probe_entry")
        assert caught == []
        torch.use_deterministic_algorithms(True)
        with pytest.raises(
            RuntimeError,
            match="probe_entry does not have a deterministic implementation",
        ) as info:
            pkg._alert_nondeterministic("probe_entry")
        assert "flashinfer.cake_causal_conv1d_det_train" in str(info.value)
        assert "warn_only=True" in str(info.value)
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.warns(
            UserWarning,
            match="probe_entry does not have a deterministic implementation",
        ) as record:
            pkg._alert_nondeterministic("probe_entry")
        assert any(
            "flashinfer.cake_causal_conv1d_det_train" in str(w.message) for w in record
        )
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


@pytest.mark.cpu_only
def test_backward_entries_alert_before_any_device_work(monkeypatch) -> None:
    """Under torch's deterministic mode the single-rank backward alerts after argument validation and before any
    allocation or launch, and the forward does not alert.  Checked without a device: zero-token CPU tensors with the
    device / placement checks stubbed (the zero-token path returns before any device query)."""

    monkeypatch.setattr(pkg, "_device_arch", lambda device: "sm_100a")
    monkeypatch.setattr(pkg, "_check_placement", lambda *a, **k: None)
    launched = []
    monkeypatch.setattr(pkg, "run_module", lambda *a, **k: launched.append(a))
    x = torch.empty(0, 16, dtype=torch.bfloat16)
    w = torch.zeros(16, 4, dtype=torch.float32)
    cu = torch.tensor([0, 0], dtype=torch.int32)
    prev, prev_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(
            RuntimeError, match="cake_causal_conv1d_train_backward does not have"
        ):
            pkg.cake_causal_conv1d_train_backward(x, w, cu, x, deterministic=False)
        # The forward runs the deterministic forward programs: no alert.
        y = pkg.cake_causal_conv1d_train_forward(x, w, cu, deterministic=False)
        assert tuple(y.shape) == (0, 16)
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.warns(
            UserWarning, match="cake_causal_conv1d_train_backward does not have"
        ):
            dx, dw = pkg.cake_causal_conv1d_train_backward(
                x, w, cu, x, deterministic=False
            )
        assert tuple(dx.shape) == (0, 16) and torch.equal(dw, torch.zeros_like(dw))
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)
    assert launched == []


@pytest.mark.cpu_only
def test_unrendered_package_reports_unsupported() -> None:
    assert cake_jit.is_rendered() is (PACKAGE_DIR / "build.json").is_file()
    if not cake_jit.is_rendered():
        assert pkg.is_cake_causal_conv1d_train_supported(torch.device("cpu")) is False
        with pytest.raises(RuntimeError):
            cake_build.manifest()


# ----------------------------------------------------------------------------------------------------------------
# Wiring.


@pytest.mark.cpu_only
def test_package_wiring() -> None:
    assert (
        pkg.__name__ == "flashinfer.experimental.cake.cake_causal_conv1d_train_backend"
    )
    tree = ast.parse((REPO / "flashinfer/experimental/cake/__init__.py").read_text())
    assert ast.get_docstring(tree)
    assert not [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert len(tree.body) == 1, "flashinfer.experimental.cake holds a docstring only"
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        tomllib = pytest.importorskip("tomli")
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    package_data = data["tool"]["setuptools"]["package-data"]
    assert package_data[pkg.__name__] == ["csrc/**", "build.json"]


@pytest.mark.cpu_only
def test_build_manifest_lists_every_program() -> None:
    """Once rendered: ``build.json`` names this package, carries exactly one entry per (program, architecture) of
    the single-rank and KCP registries, each with its kernel and binding sources under ``csrc/`` and their digests,
    IEEE division / square root and no fast math; no record is a placeholder any more."""

    if not cake_jit.is_rendered():
        pytest.skip("build.json is not rendered yet (TODO-EXPORT)")
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    libraries = cake_build.libraries_by_name()
    referenced = set()
    for registry in (cake_jit, kcp_jit):
        for name, record in registry.MODULES.items():
            assert "TODOEXPORT" not in name, name
            assert set(record["arches"]) <= set(manifest["arch_flags"]), name
            for arch in ARCHES:
                library = f"{name}_{arch}"
                if arch not in record["arches"]:
                    assert library not in libraries, library
                    continue
                entry = libraries[registry.library_name(name, arch)]
                assert entry["arch"] == arch and entry["entry"] == "run", library
                assert entry["sources"] == [
                    f"csrc/{name}_kernel.cu",
                    f"csrc/{name}_binding.cu",
                ]
                assert "--use_fast_math" not in entry["nvcc_flags"], library
                assert {"--prec-div=true", "--prec-sqrt=true", "--ftz=false"} <= set(
                    entry["nvcc_flags"]
                )
                for source in entry["sources"]:
                    path = PACKAGE_DIR / source
                    assert path.is_file(), source
                    assert (
                        hashlib.sha256(path.read_bytes()).hexdigest()
                        == entry["sha256"][source]
                    ), source
                referenced.add(library)
    assert referenced == set(libraries)


def _manifest_checker():
    """``scripts/check_cake_build_manifests.py`` of this checkout, loaded as a module (it never imports flashinfer)."""

    path = REPO / "scripts" / "check_cake_build_manifests.py"
    if not path.is_file():
        pytest.skip(
            "scripts/check_cake_build_manifests.py is not beside this test (installed package)"
        )
    spec = importlib.util.spec_from_file_location("check_cake_build_manifests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault("check_cake_build_manifests", module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.cpu_only
def test_manifest_checker_lists_the_package() -> None:
    """The build-manifest checker lists this package (its ``cake_jit.py`` and the KCP registry ``kcp_jit.py``) and
    its vendored gate finds the package's suite key; once rendered, the in-tree checks report nothing."""

    checker = _manifest_checker()
    try:
        path = str(PACKAGE_DIR.relative_to(REPO))
    except ValueError:
        pytest.skip(
            f"the package is imported from {PACKAGE_DIR}, outside this checkout"
        )
    entries = [p for p in checker.PACKAGES if p.name == pkg.__name__]
    assert len(entries) == 1, entries
    (entry,) = entries
    assert (entry.path, entry.loader) == (path, "cake_jit.py")
    assert entry.refs is checker.refs_with_kcp
    assert checker.suite_key(REPO, entry) == KEY
    if not cake_jit.is_rendered():
        pytest.skip("build.json is not rendered yet (TODO-EXPORT)")
    assert checker.run_in_tree(REPO, [entry]) == []
