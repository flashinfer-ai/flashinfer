# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for setup-bearing Solutions — flashinfer.trace_apply.setup (no GPU).

The preserved setup-bearing package is the reference: it is verified, bound,
served through the live rmsnorm template and released in one test. The rest
pins the contract around it: the four fields and hash neutrality, the
declaration refusals at both layers, identity keying, the capture rules, the
setup context, and the curated loader. CUDA-graph capture is simulated by
monkeypatching the module's ``_is_capturing`` helper."""

from __future__ import annotations

import gc
import inspect
import json
import os
import sys
import weakref

import pytest
import torch

import flashinfer
import flashinfer.trace_apply as ta
from flashinfer.trace.solution import Solution
from flashinfer.trace_apply import packages as pk
from flashinfer.trace_apply import setup as su
from flashinfer.trace_apply.apply import preloaded_entry
from flashinfer.trace_apply.loaders import load as load_solution
from flashinfer.trace_apply.setup import SetupBoundCallable, SetupStateError
from tests.cpu.trace_apply.seal_helpers import (
    DEFINITION,
    GOLDEN_DIGEST,
    GOLDEN_ROOT,
    GOLDEN_SETUP_DIGEST,
    OPERATION,
    PREPARE_KERNEL_SOURCE,
    PREPARE_SOURCE,
    QUALIFICATION,
    SETUP_SOLUTION,
    SETUP_SOURCE,
    SETUP_SPEC,
    SOLUTION,
    json_bytes,
    seal_flashinfer_fixture,
    seal_package,
)

#: Solution.from_dict(golden bundle/solution.json).hash() before the setup
#: fields existed; a Solution without a setup must keep it.
GOLDEN_SOLUTION_HASH = (
    "7d32249988c08eca0bf0cb24879925a8e20fbba5fcdded8d93bc9965ec823aac"
)
#: The same for the golden setup bundle: the hash tail the four fields add.
GOLDEN_SETUP_SOLUTION_HASH = (
    "380cb2da5f3847ce91bec50c6744723a88c8b768cf567ea358614967f9b02b84"
)
GOLDEN_SETUP_KEY = ("rmsnorm_h16", "rmsnorm_h16_prepared_weight")
FI_API = "flashinfer.norm.rmsnorm"


@pytest.fixture(autouse=True)
def clean_install(monkeypatch):
    # apply.py asks CUDA about capture on the uncached path; keep it CPU-only.
    """Keep capture checks CPU-only, then release packages and disable Apply."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    yield
    pk.release_packages()
    if ta.is_enabled():
        ta.disable_apply()


@pytest.fixture(autouse=True)
def private_imports(monkeypatch):
    """Restore ``sys.meta_path`` and drop sealed modules imported during a test."""
    before = set(sys.modules)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    yield
    for name in set(sys.modules) - before:
        if name.startswith("_flashinfer_sealed_"):
            del sys.modules[name]


@pytest.fixture
def sm_gate_off(monkeypatch):
    monkeypatch.setattr(
        pk, "check_target_hardware", lambda solution, *, device=None: None
    )


def setup_package(root, files=None, *, spec=None, definition=None):
    """Seal SETUP_SOLUTION with ``spec`` overrides (None deletes a key)."""
    files = {"kernel.py": SETUP_SOURCE} if files is None else files
    merged = {k: v for k, v in {**SETUP_SPEC, **(spec or {})}.items() if v is not None}
    solution = {
        **SETUP_SOLUTION,
        "spec": merged,
        "sources": [
            {"path": name, "content": data.decode()} for name, data in files.items()
        ],
    }
    bundle = {"solution.json": json_bytes(solution), **files}
    if definition is not None:
        bundle["definition.json"] = json_bytes(definition)
    return pk.verify_package(root, seal_package(root, bundle, sidecar=QUALIFICATION))


class _Lifecycle:
    """In-process setup/run/teardown that records what it was handed."""

    def __init__(self, result=None):
        self.result = result  # setup(context) -> result; default: a serial state
        self.setups = 0
        self.contexts: list = []
        self.runs: list = []
        self.teardowns: list = []

    def setup(self, context):
        """Record the context and return a serial state or ``result(context)``."""
        self.setups += 1
        self.contexts.append(context)
        return {"serial": self.setups} if self.result is None else self.result(context)

    def run(self, x, state, out):
        self.runs.append(state)
        out.copy_(x)

    def teardown(self, state):
        self.teardowns.append(state)


def bind(life, *, run=None, setup_inputs=("x",), state_param="state", definition=None):
    """Wrap ``life`` in a SetupBoundCallable with the test defaults."""
    return SetupBoundCallable(
        run or life.run,
        life.setup,
        life.teardown,
        name=OPERATION,
        setup_inputs=setup_inputs,
        state_param=state_param,
        definition=definition,
    )


def rmsnorm_reference(x, w):
    x = x.to(torch.float32)
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * w


# ---------------------------------------------------------------------------
# The fields, hash neutrality, and the declaration refusals at both layers
# ---------------------------------------------------------------------------


def test_setup_fields_round_trip_and_hash_neutrality():
    """Setup fields round-trip and leave a setup-less Solution's hash unchanged."""
    solution = Solution.from_dict(SETUP_SOLUTION)
    assert solution.spec.setup_inputs == ("x",)
    assert (solution.setup_file(), solution.setup_symbol()) == ("kernel.py", "setup")
    assert solution.teardown_file() == "kernel.py"
    assert Solution.from_dict(solution.to_dict()) == solution
    plain = Solution.from_dict(SOLUTION)
    assert plain.setup_file() is None
    assert not {"setup_entry_point", "setup_inputs", "setup_state_param"} & set(
        plain.to_dict()["spec"]
    )

    # A Solution without a setup keeps the hash it had before the fields
    # existed, with the defaults implicit or spelled out; a declaration is
    # part of the identity.
    golden = json.loads(
        (GOLDEN_ROOT / GOLDEN_DIGEST / "bundle" / "solution.json").read_text()
    )
    assert Solution.from_dict(golden).hash() == GOLDEN_SOLUTION_HASH
    explicit = {**golden, "spec": {**golden["spec"], "setup_inputs": []}}
    assert Solution.from_dict(explicit).hash() == GOLDEN_SOLUTION_HASH
    assert (
        Solution.from_dict(explicit).to_dict() == Solution.from_dict(golden).to_dict()
    )
    changed = {**SETUP_SOLUTION, "spec": {**SETUP_SPEC, "setup_state_param": "scratch"}}
    assert Solution.from_dict(changed).hash() != solution.hash()
    golden_setup = json.loads(
        (GOLDEN_ROOT / GOLDEN_SETUP_DIGEST / "bundle" / "solution.json").read_text()
    )
    assert Solution.from_dict(golden_setup).hash() == GOLDEN_SETUP_SOLUTION_HASH


@pytest.mark.parametrize(
    "spec, definition, spec_message, verify_message",
    [
        (
            {"setup_entry_point": None},
            None,
            "teardown_entry_point requires setup_entry_point",
            "teardown_entry_point requires setup_entry_point",
        ),
        (
            {"setup_entry_point": None, "teardown_entry_point": None},
            None,
            "setup_inputs requires setup_entry_point",
            "setup_inputs requires setup_entry_point",
        ),
        (
            {"destination_passing_style": None},
            None,
            "must be declared explicitly",
            "declared explicitly",
        ),
        (
            {"setup_inputs": ["x", "x"]},
            None,
            "distinct identifiers",
            "distinct identifiers",
        ),
        (
            {"setup_state_param": "x"},
            None,
            "'x' is also a setup input",
            "'x' is also a setup input",
        ),
        (
            # BuildSpec takes no module:symbol here; verify refuses what load would.
            {"setup_entry_point": "kernel:setup"},
            None,
            "Invalid setup_entry_point",
            "must be '<file>.py::<symbol>'",
        ),
        (
            {"setup_entry_point": "prepare.py::setup"},
            None,
            "Setup source file 'prepare.py'",
            "setup_entry_point file 'prepare.py' is not sealed",
        ),
        ({"language": "cuda"}, None, "requires a Python-family language", None),
        (
            {"setup_inputs": ["scale"]},
            DEFINITION,
            None,
            "setup input 'scale' is not a tensor input of the sealed definition.json",
        ),
        (
            # The call already passes every Definition input and DPS output by
            # keyword, so the state keyword cannot be one of them.
            {"setup_state_param": "out"},
            DEFINITION,
            None,
            "setup_state_param 'out' is also an input or output",
        ),
        (
            # The context is filled from the sealed Definition: one for another
            # shape would prepare state for the wrong Definition.
            {},
            {**DEFINITION, "name": "other_op"},
            None,
            "describes 'other_op', not the solution's definition",
        ),
    ],
)
def test_bad_setup_declarations_are_refused(
    tmp_path, spec, definition, spec_message, verify_message
):
    """Bad setup declarations refuse in BuildSpec and at verification alike."""
    merged = {k: v for k, v in {**SETUP_SPEC, **spec}.items() if v is not None}
    if spec_message is not None:
        with pytest.raises(ValueError, match=spec_message):
            Solution.from_dict({**SETUP_SOLUTION, "spec": merged})
    if verify_message is not None:
        with pytest.raises(pk.SealedPackageError, match=verify_message):
            setup_package(tmp_path, spec=spec, definition=definition)


def test_setup_files_must_be_listed_in_sources(tmp_path):
    # Solution.__post_init__ refuses a setup file absent from sources; verify
    # refuses it first, with the standard library, instead of load_packages.
    """A setup file absent from ``sources`` refuses at verification."""
    solution = {
        **SETUP_SOLUTION,
        "spec": {**SETUP_SPEC, "setup_entry_point": "prepare.py::setup"},
        "sources": [{"path": "kernel.py", "content": PREPARE_KERNEL_SOURCE.decode()}],
    }
    files = {"kernel.py": PREPARE_KERNEL_SOURCE, "prepare.py": PREPARE_SOURCE}
    digest = seal_package(
        tmp_path,
        {"solution.json": json_bytes(solution), **files},
        sidecar=QUALIFICATION,
    )
    with pytest.raises(pk.SealedPackageError, match="is not listed in sources"):
        pk.verify_package(tmp_path, digest)


def test_destination_passing_style_must_be_a_boolean_with_a_setup(tmp_path):
    # A null would read as False and leave a destination-passing kernel without
    # its output buffer; both layers refuse it.
    """A null destination_passing_style with a setup refuses at both layers."""
    spec = {**SETUP_SPEC, "destination_passing_style": None}
    with pytest.raises(ValueError, match="as a boolean"):
        Solution.from_dict({**SETUP_SOLUTION, "spec": spec})
    digest = seal_package(
        tmp_path,
        {
            "solution.json": json_bytes({**SETUP_SOLUTION, "spec": spec}),
            "kernel.py": SETUP_SOURCE,
        },
        sidecar=QUALIFICATION,
    )
    with pytest.raises(pk.SealedPackageError, match="as a boolean"):
        pk.verify_package(tmp_path, digest)


def test_setup_fields_are_framed_in_the_hash():
    """The hash frames the setup fields so adjacent values cannot run together."""

    def hashed(state_param, inputs):
        spec = {**SETUP_SPEC, "setup_state_param": state_param, "setup_inputs": inputs}
        return Solution.from_dict({**SETUP_SOLUTION, "spec": spec}).hash()

    assert hashed("a", ["bc"]) != hashed("ab", ["c"])


# ---------------------------------------------------------------------------
# The preserved setup-bearing package: verify, bind, serve, release
# ---------------------------------------------------------------------------


def test_golden_setup_package_verifies_binds_and_serves(sm_gate_off):
    """The golden setup package verifies, binds, serves rmsnorm and releases cleanly."""
    directory = GOLDEN_ROOT / GOLDEN_SETUP_DIGEST
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert set(manifest["files"]) == {
        "definition.json",
        "kernel.py",
        "prepare.py",
        "solution.json",
    }
    assert set(manifest["sealed_files"]) == {"qualification.json"}  # canonical spelling

    sealed = pk.verify_package(GOLDEN_ROOT, GOLDEN_SETUP_DIGEST)
    assert sealed.operation == "rmsnorm_h16"
    assert sealed.setup_entry_point == ("prepare", "setup")
    assert sealed.teardown_entry_point == ("prepare", "teardown")
    assert sealed.setup_inputs == ("weight",) and sealed.setup_state_param == "state"
    assert sealed.definition["name"] == "rmsnorm_h16"
    assert not [m for m in sys.modules if m.startswith("_flashinfer_sealed_")]

    result = pk.load_packages(GOLDEN_ROOT, [GOLDEN_SETUP_DIGEST])
    assert result.applied >= 1 and not result.errors
    bound = result.callables[GOLDEN_SETUP_KEY]
    assert isinstance(bound, SetupBoundCallable)
    assert preloaded_entry(*GOLDEN_SETUP_KEY) is bound
    assert pk.import_entry(sealed) is bound  # one binding per digest and process
    assert list(inspect.signature(bound).parameters) == [
        "hidden_states",
        "weight",
        "output",
    ]
    namespace = bound.__module__.rsplit(".", 1)[0]
    assert namespace.startswith("_flashinfer_sealed_")
    assert f"{namespace}.prepare" in sys.modules and "prepare" not in sys.modules

    hidden, weight = torch.randn(4, 16), torch.randn(16)
    assert torch.allclose(
        flashinfer.norm.rmsnorm(hidden, weight), rmsnorm_reference(hidden, weight)
    )
    flashinfer.norm.rmsnorm(torch.randn(2, 16), weight)  # the same weight set
    assert pk.setup_stats()["rmsnorm_h16"]["runs"] == 1  # prepare.py ran once
    other = torch.randn(16)
    assert torch.allclose(
        flashinfer.norm.rmsnorm(hidden, other), rmsnorm_reference(hidden, other)
    )
    stats = pk.setup_stats()["rmsnorm_h16"]
    assert (stats["runs"], stats["weight_sets"], stats["refused_in_capture"]) == (
        2,
        2,
        0,
    )
    assert ta.stats()[FI_API]["hit"] == 3

    pk.release_packages()  # teardown(state) per weight set, bindings forgotten
    assert pk.setup_stats() == {} and preloaded_entry(*GOLDEN_SETUP_KEY) is None
    assert not ta.is_enabled()  # the released Definition left the mapping too
    assert pk.import_entry(sealed) is not bound  # a later import binds afresh


def test_release_packages_leaves_curated_and_captured_solutions_enabled(
    tmp_path, sm_gate_off, monkeypatch
):
    """Release removes sealed identities only; curated and captured bindings stay."""
    import flashinfer.trace_apply.apply as apply_module

    def curated(hidden_states, weight):
        return hidden_states

    assert ta.enable_apply({"rmsnorm_h16": curated}) >= 1
    sealed = setup_package(tmp_path)
    result = pk.load_packages(tmp_path, [sealed.digest])
    [(definition, name)] = result.callables
    assert set(apply_module._solutions) == {"rmsnorm_h16", OPERATION}
    pk.release_packages()
    assert apply_module._solutions == {"rmsnorm_h16": curated} and ta.is_enabled()
    assert preloaded_entry(definition, name) is None

    # One Definition has one Solution: a sealed package for an enabled one is
    # refused (strict) or recorded and left out (best effort), never substituted,
    # and refused before its module runs: nothing is imported or bound for it.
    modules = {m for m in sys.modules if m.startswith("_flashinfer_sealed_")}
    with pytest.raises(pk.SealedPackageError, match="already has an enabled Solution"):
        pk.load_packages(GOLDEN_ROOT, [GOLDEN_SETUP_DIGEST])
    result = pk.load_packages(GOLDEN_ROOT, [GOLDEN_SETUP_DIGEST], strict=False)
    assert result.applied == 0 and GOLDEN_SETUP_DIGEST in result.errors
    assert apply_module._solutions == {"rmsnorm_h16": curated}
    assert pk.setup_stats() == {} and preloaded_entry(*GOLDEN_SETUP_KEY) is None
    assert {m for m in sys.modules if m.startswith("_flashinfer_sealed_")} == modules

    # Release goes by identity: a curated Solution enabled for the sealed
    # Definition after Apply was disabled is not the sealed one and stays.
    ta.disable_apply()
    pk.load_packages(GOLDEN_ROOT, [GOLDEN_SETUP_DIGEST])
    ta.disable_apply()
    assert ta.enable_apply({"rmsnorm_h16": curated}) >= 1
    pk.release_packages()
    assert apply_module._solutions == {"rmsnorm_h16": curated} and ta.is_enabled()
    assert preloaded_entry(*GOLDEN_SETUP_KEY) is None
    ta.disable_apply()

    # A binding a capture observed keeps its states, its route and its
    # preloaded entry: captured graphs replay against them.
    monkeypatch.setattr(
        pk, "_bound", {}
    )  # the frozen binding must not outlive the test
    result = pk.load_packages(GOLDEN_ROOT, [GOLDEN_SETUP_DIGEST])
    bound = result.callables[GOLDEN_SETUP_KEY]
    hidden, weight, out = torch.randn(2, 16), torch.randn(16), torch.empty(2, 16)
    bound(hidden_states=hidden, weight=weight, output=out)
    monkeypatch.setattr(su, "_is_capturing", lambda device: True)
    bound(hidden_states=hidden, weight=weight, output=out)
    monkeypatch.setattr(su, "_is_capturing", lambda device: False)
    assert bound.frozen
    pk.release_packages()
    assert set(apply_module._solutions) == {"rmsnorm_h16"} and ta.is_enabled()
    assert preloaded_entry(*GOLDEN_SETUP_KEY) is bound
    assert pk.setup_stats() == {"rmsnorm_h16": bound.stats()}


# ---------------------------------------------------------------------------
# SetupBoundCallable: keying, bounds, capture rules, context, malformed calls
# ---------------------------------------------------------------------------


def test_state_is_keyed_on_setup_input_identity():
    """States key on tensor storage identity, and release tears them all down."""
    life = _Lifecycle()
    wrapper = bind(life)
    a, b, out = torch.ones(2, 4), torch.ones(2, 4), torch.empty(2, 4)
    wrapper(x=a, out=out)
    wrapper(x=a.view(2, 4), out=out)  # the same storage and geometry: one state
    wrapper(x=a[:], out=out)
    assert life.setups == 1
    wrapper(x=b, out=out)  # equal contents, another storage: another weight set
    assert life.setups == 2
    assert [state["serial"] for state in life.runs] == [1, 1, 1, 2]
    assert wrapper.stats()["weight_sets"] == 2 and wrapper.stats()["runs"] == 2

    # The retained state keeps its weights alive, so their addresses cannot be
    # recycled and re-keyed; release() tears every state down and lets go.
    life.contexts.clear()
    pinned = weakref.ref(a)
    del a
    gc.collect()
    assert pinned() is not None
    wrapper.release()
    assert life.teardowns == [{"serial": 1}, {"serial": 2}]
    gc.collect()
    assert pinned() is None and wrapper.stats()["weight_sets"] == 0

    # The state arrives under the declared keyword, which the call may not use.
    seen = []

    def run(x, out, *, scratch):
        seen.append(scratch)

    scratch = bind(_Lifecycle(), run=run, state_param="scratch")
    assert list(inspect.signature(scratch).parameters) == ["x", "out"]
    scratch(x=b, out=out)
    assert seen == [{"serial": 1}]
    with pytest.raises(SetupStateError, match="already passes 'scratch'"):
        scratch(x=b, out=out, scratch=None)
    with pytest.raises(SetupStateError, match="no keyword parameter 'state'"):
        bind(life, run=lambda x, out: None)


def test_state_bounds_and_failed_setups(monkeypatch):
    """A failed setup leaves nothing; MAX_WEIGHT_SETS bounds the retained states."""
    life = _Lifecycle()
    wrapper = bind(life, setup_inputs=())  # no setup inputs: one state per process
    for _ in range(3):
        wrapper(x=torch.ones(2), out=torch.empty(2))
    assert life.setups == 1 and life.contexts[0]["setup_inputs"] == {}

    monkeypatch.setattr(su, "MAX_WEIGHT_SETS", 2)
    life = _Lifecycle()
    wrapper = bind(life)
    weights, out = [torch.ones(2) for _ in range(3)], torch.empty(2)
    wrapper(x=weights[0], out=out)
    wrapper(x=weights[1], out=out)
    with pytest.raises(SetupStateError, match="MAX_WEIGHT_SETS = 2"):
        wrapper(x=weights[2], out=out)
    wrapper(x=weights[0], out=out)  # the retained sets still serve

    def refuse(context):
        raise ValueError("no weights")

    life = _Lifecycle(result=refuse)
    wrapper = bind(life)
    with pytest.raises(
        SetupStateError, match="setup raised ValueError: no weights"
    ) as info:
        wrapper(x=weights[0], out=out)
    assert isinstance(info.value.__cause__, ValueError)
    assert life.runs == [] and wrapper.stats()["weight_sets"] == 0
    life.result = None
    wrapper(x=weights[0], out=out)  # the failed attempt left nothing behind
    assert life.setups == 2 and life.runs == [{"serial": 2}]


def test_capture_refuses_first_use_freezes_and_never_releases(tmp_path, monkeypatch):
    # A weight set first seen under capture is refused and counted.
    """Capture refuses a first use, freezes the key set and pins the states for good."""
    life = _Lifecycle()
    wrapper = bind(life)
    a, out = torch.ones(2), torch.empty(2)
    monkeypatch.setattr(su, "_is_capturing", lambda device: True)
    with pytest.raises(SetupStateError, match="inside CUDA-graph capture"):
        wrapper(x=a, out=out)
    assert life.setups == 0 and wrapper.stats()["refused_in_capture"] == 1

    # A captured hit is served and freezes the key set; a later weight set is
    # refused eagerly too; the states a capture observed are never released.
    monkeypatch.setattr(
        pk, "_bound", {}
    )  # the frozen binding must not outlive the test
    monkeypatch.setattr(su, "_is_capturing", lambda device: False)
    bound = pk.import_entry(setup_package(tmp_path))
    bound(x=a, out=out)  # eager: prepared
    monkeypatch.setattr(su, "_is_capturing", lambda device: True)
    bound(x=a, out=out)
    monkeypatch.setattr(su, "_is_capturing", lambda device: False)
    assert bound.frozen
    with pytest.raises(SetupStateError, match="frozen"):
        bound(x=torch.ones(2), out=out)
    bound(x=a, out=out)  # the prepared weight set keeps serving
    assert sys.modules[bound.__module__].CALLS == {"setup": 1, "run": 3, "teardown": 0}
    with pytest.raises(SetupStateError, match="kept for the life of the process"):
        bound.release()
    pk.release_packages()
    assert sys.modules[bound.__module__].CALLS["teardown"] == 0
    assert pk.setup_stats() == {OPERATION: bound.stats()}
    assert bound.stats()["weight_sets"] == 1


def test_a_wrapper_kept_across_release_routes_nowhere(sm_gate_off, monkeypatch):
    """A wrapped API kept across release routes to stock, never to the loader."""
    import flashinfer.trace_apply.apply as apply_module

    def stock(
        hidden_states, weight, *args, **kwargs
    ):  # the kernel's stand-in on a CPU host
        return rmsnorm_reference(hidden_states, weight)

    monkeypatch.setattr(flashinfer.norm, "rmsnorm", stock)
    pk.load_packages(GOLDEN_ROOT, [GOLDEN_SETUP_DIGEST])
    kept = (
        flashinfer.norm.rmsnorm
    )  # the wrapped API, kept before its first matching call
    assert (
        getattr(kept, "_trace_apply", False)
        and "rmsnorm_h16" in apply_module._solutions
    )
    pk.release_packages()
    assert (
        flashinfer.norm.rmsnorm is stock
    )  # the module attribute is the original again
    monkeypatch.setattr(
        apply_module,
        "load_solution",
        lambda solution: pytest.fail(
            "the on-disk loader ran for a released sealed Solution"
        ),
    )
    hidden, weight = torch.randn(4, 16), torch.randn(16)
    # The kept wrapper resolves nothing: stock, never the loader, and no binding.
    assert torch.allclose(kept(hidden, weight), rmsnorm_reference(hidden, weight))
    assert pk.setup_stats() == {}


def test_capture_is_seen_on_the_calls_device_and_on_the_current_one(monkeypatch):
    """_is_capturing checks the call's device and the current one, restoring it."""
    capturing = {0: False, 1: False}
    current = [0]

    class Device:  # torch.cuda.device: the current device, for the block
        def __init__(self, device):
            """Resolve the device index, defaulting to the current one."""
            self.index = current[0] if device.index is None else device.index

        def __enter__(self):
            self.previous, current[0] = current[0], self.index

        def __exit__(self, *exc):
            current[0] = self.previous

    monkeypatch.setattr(torch.cuda, "device", Device)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: capturing[current[0]]
    )
    call = torch.device("cuda", 1)
    assert not su._is_capturing(call)
    capturing[1] = True
    assert su._is_capturing(call)  # the call's device captures, the current does not
    capturing[1], capturing[0] = False, True
    assert su._is_capturing(call)  # the current device captures, the call's does not
    assert current[0] == 0  # the current device is left as it was
    assert not su._is_capturing(torch.device("cpu"))


def test_a_released_binding_refuses_calls_even_from_its_own_teardown():
    """A released binding refuses every call, including one from its teardown."""
    life = _Lifecycle()
    reentered = []

    def teardown(state):
        """Record the state and re-enter the released wrapper."""
        life.teardowns.append(state)
        try:
            wrapper(x=a, out=out)  # a teardown re-entering its own Definition
        except SetupStateError as error:
            reentered.append(str(error))

    wrapper = SetupBoundCallable(
        life.run,
        life.setup,
        teardown,
        name=OPERATION,
        setup_inputs=("x",),
        state_param="state",
    )
    a, out = torch.ones(2), torch.empty(2)
    wrapper(x=a, out=out)
    wrapper.release()
    assert life.teardowns == [{"serial": 1}]
    assert len(reentered) == 1 and "was released" in reentered[0]
    assert (
        life.setups == 1 and wrapper.stats()["weight_sets"] == 0
    )  # nothing prepared behind the release
    with pytest.raises(SetupStateError, match="was released"):
        wrapper(x=a, out=out)  # a wrapper a caller kept across the release


def test_a_state_does_not_survive_fork(monkeypatch):
    """A binding called from another process id refuses."""
    wrapper = bind(_Lifecycle())
    pid = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: pid + 1)
    with pytest.raises(SetupStateError, match="does not survive fork"):
        wrapper(x=torch.ones(2), out=torch.empty(2))


def test_context_has_the_documented_keys():
    """The setup context holds exactly the documented keys, Definition or not."""

    def run(x, out, state, scale):
        out.copy_(x * scale)

    life = _Lifecycle()
    x, out = torch.zeros(3, 4), torch.empty(3, 4)
    bind(life, run=run, definition=DEFINITION)(x=x, out=out, scale=2.0)
    context = life.contexts[0]
    metadata = {"shape": [3, 4], "stride": [4, 1], "dtype": "float32", "device": "cpu"}
    # A Definition scalar: no geometry, the declared dtype and device.
    scalar = {"shape": None, "stride": None, "dtype": "float32", "device": "cuda"}
    assert context["axes"] == {"rows": 3, "width": 4}
    assert context["device"] == "cpu"
    assert (context["rank"], context["world_size"]) == (0, 1)
    assert context["inputs"] == {"x": metadata, "scale": scalar}
    assert list(context["inputs"]) == ["x", "scale"]  # Definition order
    assert context["input_values"] == {"scale": 2.0}
    assert list(context["setup_inputs"]) == ["x"] and context["setup_inputs"]["x"] is x
    assert context["outputs"] == {"out": metadata}
    assert set(context) == {
        "axes",
        "device",
        "rank",
        "world_size",
        "inputs",
        "input_values",
        "setup_inputs",
        "outputs",
    }

    # Without a Definition document: no axes, no outputs, and the output buffer
    # is one more tensor in the call.
    life = _Lifecycle()
    bind(life, run=run)(x=x, out=out, scale=2.0)
    context = life.contexts[0]
    assert context["axes"] == {} and context["outputs"] == {}
    assert context["inputs"] == {"x": metadata, "out": metadata}

    # A value-returning Solution passes no output buffer: a declared output is
    # described from the Definition and the resolved axes instead.
    life = _Lifecycle()
    bind(life, run=lambda x, state, scale: x * scale, definition=DEFINITION)(
        x=x, scale=2.0
    )
    assert life.contexts[0]["outputs"] == {
        "out": {"shape": [3, 4], "stride": None, "dtype": "float32", "device": "cuda"}
    }


def test_result_envelope_and_malformed_calls():
    """The setup result envelope is unpacked and malformed calls refuse by keyword."""
    envelope = _Lifecycle(
        result=lambda context: {"state": "prepared", "input_buffers": {}}
    )
    bind(envelope)(x=torch.ones(2), out=torch.empty(2))
    assert envelope.runs == ["prepared"]
    opaque = _Lifecycle(
        result=lambda context: {"state": 1}
    )  # no input_buffers: a state
    bind(opaque)(x=torch.ones(2), out=torch.empty(2))
    assert opaque.runs == [{"state": 1}]
    for result, message in (
        ({"state": 1, "input_buffers": {"x": torch.ones(2)}}, r"input_buffers \['x'\]"),
        ({"state": 1, "input_buffers": None}, "must be a mapping"),
        (
            {"state": 1, "input_buffers": {}, "extra": 2},
            "exactly state and input_buffers",
        ),
    ):
        life = _Lifecycle(result=lambda context, result=result: result)
        with pytest.raises(SetupStateError, match=message):
            bind(life)(x=torch.ones(2), out=torch.empty(2))
        assert life.runs == []

    wrapper = bind(_Lifecycle())
    out = torch.empty(2)
    with pytest.raises(SetupStateError, match="keyword only"):
        wrapper(torch.ones(2), out=out)
    with pytest.raises(SetupStateError, match="setup input 'x' is absent"):
        wrapper(out=out)
    with pytest.raises(SetupStateError, match="setup input 'x' is not a tensor"):
        wrapper(x=1.0, out=out)


# ---------------------------------------------------------------------------
# Sealed import refusals, the curated loader, import infallibility, exports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "files, spec, message",
    [
        (
            {
                "kernel.py": b"setup = 7\ndef run(x, state, out): pass\ndef teardown(s): pass\n"
            },
            {},
            "declared setup_entry_point is not callable",
        ),
        (
            {
                "kernel.py": b"def setup(c): pass\ndef run(x, out): pass\ndef teardown(s): pass\n"
            },
            {},
            "no keyword parameter 'state'",
        ),
        (
            {"kernel.py": PREPARE_KERNEL_SOURCE, "prepare.py": b"def setup(context:\n"},
            {"setup_entry_point": "prepare.py::setup", "teardown_entry_point": None},
            "cannot load sealed setup_entry_point: ",
        ),
    ],
)
def test_import_entry_refuses_an_unbindable_setup(tmp_path, files, spec, message):
    """A setup that is not callable, unbindable or unloadable refuses at import."""
    with pytest.raises(pk.SealedPackageError, match=message):
        pk.import_entry(setup_package(tmp_path, files, spec=spec))
    assert pk.setup_stats() == {}


def test_curated_loader_binds_the_same_class(tmp_path, monkeypatch):
    """The curated loader binds a setup-bearing Solution as a SetupBoundCallable."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # hermetic loader cache
    split = Solution.from_dict(
        {
            **SETUP_SOLUTION,
            "spec": {
                **SETUP_SPEC,
                "setup_entry_point": "prepare.py::setup",
                "teardown_entry_point": "prepare.py::teardown",
            },
            "sources": [
                {"path": "kernel.py", "content": PREPARE_KERNEL_SOURCE.decode()},
                {"path": "prepare.py", "content": PREPARE_SOURCE.decode()},
            ],
        }
    )
    bound = load_solution(split)
    assert isinstance(bound, SetupBoundCallable)
    out = torch.empty(2)
    bound(x=torch.zeros(2), out=out)
    assert out.tolist() == [1.0, 1.0]  # prepare.py's state reached run
    assert not isinstance(
        load_solution(Solution.from_dict(SOLUTION)), SetupBoundCallable
    )


def test_load_packages_refuses_a_schema_error_as_a_sealed_error(tmp_path, sm_gate_off):
    # Verification reads the Solution with the standard library; a key the
    # schema requires and the verifier does not is a refusal, not an escape
    # out of ``_load_packages_from_env``.
    """A Solution missing a schema key is a sealed-package refusal, not an escape."""
    authorless = {key: value for key, value in SOLUTION.items() if key != "author"}
    digest = seal_flashinfer_fixture(tmp_path, solution=authorless)
    result = pk.load_packages(tmp_path, [digest], strict=False, enable=False)
    assert result.packages == []
    assert "is not a FlashInfer Solution: 'author'" in result.errors[digest]
    with pytest.raises(pk.SealedPackageError, match="is not a FlashInfer Solution"):
        pk.load_packages(tmp_path, [digest], enable=False)


def test_exports():
    """The package re-exports the setup API and API_VERSION is 2."""
    assert ta.SetupBoundCallable is SetupBoundCallable
    assert ta.SetupStateError is SetupStateError
    assert ta.release_packages is pk.release_packages
    assert ta.setup_stats is pk.setup_stats
    assert pk.API_VERSION == 2
    assert issubclass(SetupStateError, RuntimeError)
