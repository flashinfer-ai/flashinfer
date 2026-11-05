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

The setup fields on ``BuildSpec``, the bound callable's identity keying and
refusals, the sealed-package declaration checks and same-namespace binding,
the curated loader, and ``enable_apply`` on the live rmsnorm template. Every
test runs on the CPU; CUDA-graph capture is simulated by monkeypatching the
module's ``_is_capturing`` helper."""

from __future__ import annotations

import gc
import inspect
import json
import os
import sys
import threading
import weakref

import pytest
import torch

import flashinfer
import flashinfer.trace_apply as ta
from flashinfer.trace.solution import BuildSpec, Solution
from flashinfer.trace_apply import packages as pk
from flashinfer.trace_apply import setup as su
from flashinfer.trace_apply.apply import _get_loaded_solution, preloaded_entry
from flashinfer.trace_apply.loaders import load as load_solution
from flashinfer.trace_apply.setup import SetupBoundCallable, SetupStateError
from tests.cpu.trace_apply.seal_helpers import (
    DEFINITION,
    GOLDEN_DIGEST,
    GOLDEN_ROOT,
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
FI_API = "flashinfer.norm.rmsnorm"


@pytest.fixture(autouse=True)
def clean_install(monkeypatch):
    # apply.py asks CUDA about capture on the uncached path; keep it CPU-only.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    yield
    pk.release_packages()
    if ta.is_enabled():
        ta.disable_apply()


@pytest.fixture(autouse=True)
def private_imports(monkeypatch):
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
        self.setups += 1
        self.contexts.append(context)
        return {"serial": self.setups} if self.result is None else self.result(context)

    def run(self, x, state, out):
        self.runs.append(state)
        out.copy_(x)

    def teardown(self, state):
        self.teardowns.append(state)


def bind(life, *, run=None, setup_inputs=("x",), state_param="state", definition=None):
    return SetupBoundCallable(
        run or life.run,
        life.setup,
        life.teardown,
        name=OPERATION,
        setup_inputs=setup_inputs,
        state_param=state_param,
        definition=definition,
    )


# ---------------------------------------------------------------------------
# BuildSpec / Solution: the four fields, hash neutrality, refusals
# ---------------------------------------------------------------------------


def test_setup_fields_round_trip():
    spec = BuildSpec.from_dict(SETUP_SPEC)
    assert spec.setup_entry_point == "kernel.py::setup"
    assert spec.teardown_entry_point == "kernel.py::teardown"
    assert spec.setup_inputs == ("x",) and spec.setup_state_param == "state"
    assert spec.to_dict() == {**SETUP_SPEC, "dependencies": []}
    assert BuildSpec.from_dict(spec.to_dict()) == spec
    without_teardown = BuildSpec.from_dict(
        {k: v for k, v in SETUP_SPEC.items() if k != "teardown_entry_point"}
    )
    assert "teardown_entry_point" not in without_teardown.to_dict()

    solution = Solution.from_dict(SETUP_SOLUTION)
    assert Solution.from_dict(solution.to_dict()) == solution
    assert (solution.setup_file(), solution.setup_symbol()) == ("kernel.py", "setup")
    assert (solution.teardown_file(), solution.teardown_symbol()) == (
        "kernel.py",
        "teardown",
    )
    plain = Solution.from_dict(SOLUTION)
    assert plain.setup_file() is None and plain.teardown_symbol() is None
    assert not {"setup_entry_point", "setup_inputs", "setup_state_param"} & set(
        plain.to_dict()["spec"]
    )


def test_solution_hash_unchanged_without_a_setup():
    golden = json.loads(
        (GOLDEN_ROOT / GOLDEN_DIGEST / "bundle" / "solution.json").read_text()
    )
    assert Solution.from_dict(golden).hash() == GOLDEN_SOLUTION_HASH
    explicit_defaults = {
        **golden,
        "spec": {**golden["spec"], "setup_inputs": [], "setup_state_param": "state"},
    }
    assert Solution.from_dict(explicit_defaults).hash() == GOLDEN_SOLUTION_HASH
    assert (
        Solution.from_dict(explicit_defaults).to_dict()
        == Solution.from_dict(golden).to_dict()
    )
    # A declared setup is part of the identity: a different declaration, a
    # different Solution.
    with_setup = Solution.from_dict(SETUP_SOLUTION).hash()
    assert (
        with_setup
        != Solution.from_dict(
            {**SETUP_SOLUTION, "spec": {**SETUP_SPEC, "setup_inputs": []}}
        ).hash()
    )
    assert (
        with_setup
        != Solution.from_dict(
            {**SETUP_SOLUTION, "spec": {**SETUP_SPEC, "setup_state_param": "scratch"}}
        ).hash()
    )


@pytest.mark.parametrize(
    "spec, message",
    [
        (
            {"setup_entry_point": None},
            "teardown_entry_point requires setup_entry_point",
        ),
        (
            {"setup_entry_point": None, "teardown_entry_point": None},
            "setup_inputs requires setup_entry_point",
        ),
        ({"destination_passing_style": None}, "must be declared explicitly"),
        ({"setup_inputs": ["x", "x"]}, "distinct identifiers"),
        ({"setup_inputs": ["not an identifier"]}, "distinct identifiers"),
        ({"setup_inputs": "x"}, "distinct identifiers"),  # a str is not a list of them
        ({"setup_state_param": "x"}, "'x' is also a setup input"),
        ({"setup_state_param": "1state"}, "Invalid setup_state_param"),
        ({"setup_entry_point": "kernel.py"}, "Invalid setup_entry_point"),
        ({"teardown_entry_point": "kernel.py::"}, "Invalid teardown_entry_point"),
        ({"setup_entry_point": "prepare.py::setup"}, "Setup source file 'prepare.py'"),
        (
            {"teardown_entry_point": "prepare.py::teardown"},
            "Teardown source file 'prepare.py'",
        ),
        ({"language": "cuda"}, "requires a Python-family language"),
    ],
)
def test_spec_refuses_bad_setup_declarations(spec, message):
    merged = {k: v for k, v in {**SETUP_SPEC, **spec}.items() if v is not None}
    with pytest.raises(ValueError, match=message):
        Solution.from_dict({**SETUP_SOLUTION, "spec": merged})


# ---------------------------------------------------------------------------
# SetupBoundCallable: signature, keying, refusals, context, release
# ---------------------------------------------------------------------------


def test_published_signature_lacks_the_state_parameter():
    life = _Lifecycle()
    wrapper = bind(life)
    assert list(inspect.signature(wrapper).parameters) == ["x", "out"]
    assert wrapper.__wrapped__ == life.run and wrapper.__name__ == "run"
    assert wrapper.name == OPERATION
    assert wrapper.setup_inputs == ("x",) and wrapper.state_param == "state"

    def scratch_run(x, out, *, scratch):
        pass

    scratch = bind(life, run=scratch_run, state_param="scratch")
    assert list(inspect.signature(scratch).parameters) == ["x", "out"]
    for run in (
        lambda x, out: None,
        lambda x, out, **kwargs: None,
        lambda x, *state, out: None,
    ):
        with pytest.raises(SetupStateError, match="no keyword parameter 'state'"):
            bind(life, run=run)


def test_state_is_keyed_on_setup_input_identity():
    life = _Lifecycle()
    wrapper = bind(life)
    a, b, out = torch.ones(2, 4), torch.ones(2, 4), torch.empty(2, 4)
    wrapper(x=a, out=out)
    wrapper(x=a, out=out)
    wrapper(x=a.view(2, 4), out=out)  # the same storage and geometry: one state
    wrapper(x=a[:], out=out)
    assert life.setups == 1
    wrapper(x=b, out=out)  # equal contents, another storage: another weight set
    assert life.setups == 2
    assert [state["serial"] for state in life.runs] == [1, 1, 1, 1, 2]
    assert torch.equal(out, b)
    stats = wrapper.stats()
    assert stats["seconds"] >= 0.0
    del stats["seconds"]
    assert stats == {
        "runs": 2,
        "weight_sets": 2,
        "retained_bytes": 0,
        "refused_in_capture": 0,
    }


def test_state_arrives_under_the_declared_keyword():
    seen = []

    def run(x, out, *, scratch):
        seen.append(scratch)
        out.copy_(x)

    life = _Lifecycle()
    wrapper = bind(life, run=run, state_param="scratch")
    wrapper(x=torch.ones(2), out=torch.empty(2))
    assert seen == [{"serial": 1}]
    with pytest.raises(SetupStateError, match="already passes 'scratch'"):
        wrapper(x=torch.ones(2), out=torch.empty(2), scratch=None)


def test_empty_setup_inputs_means_one_state_per_process():
    life = _Lifecycle()
    wrapper = bind(life, setup_inputs=())
    for _ in range(3):
        wrapper(x=torch.ones(2), out=torch.empty(2))
    assert life.setups == 1 and wrapper.stats()["weight_sets"] == 1
    assert life.contexts[0]["setup_inputs"] == {}
    assert life.contexts[0]["device"] == "cpu"


def test_setup_raising_keeps_no_state():
    def refuse(context):
        raise ValueError("no weights")

    life = _Lifecycle(result=refuse)
    wrapper = bind(life)
    x, out = torch.ones(2), torch.empty(2)
    with pytest.raises(
        RuntimeError, match="setup raised ValueError: no weights"
    ) as info:
        wrapper(x=x, out=out)
    assert isinstance(info.value, SetupStateError)
    assert isinstance(info.value.__cause__, ValueError)
    assert life.runs == [] and wrapper.stats()["runs"] == 0
    assert wrapper.stats()["weight_sets"] == 0
    life.result = None
    wrapper(x=x, out=out)  # the failed attempt left nothing: setup runs again
    assert life.setups == 2 and life.runs == [{"serial": 2}]


def test_setup_result_envelope():
    life = _Lifecycle(result=lambda context: {"state": "prepared", "input_buffers": {}})
    bind(life)(x=torch.ones(2), out=torch.empty(2))
    assert life.runs == ["prepared"]
    opaque = _Lifecycle(result=lambda context: {"state": 1})  # no input_buffers key
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


def test_is_capturing_asks_cuda_for_cuda_devices_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert su._is_capturing(torch.device("cpu")) is False
    assert su._is_capturing(torch.device("cuda", 0)) is True


def test_capture_refuses_a_first_use_and_freezes_the_key_set(monkeypatch):
    life = _Lifecycle()
    wrapper = bind(life)
    a, b, out = torch.ones(2), torch.ones(2), torch.empty(2)
    wrapper(x=a, out=out)  # eager: prepared
    monkeypatch.setattr(su, "_is_capturing", lambda device: True)
    wrapper(x=a, out=out)  # a captured hit is served and freezes the key set
    with pytest.raises(SetupStateError, match="inside CUDA-graph capture"):
        wrapper(x=b, out=out)
    assert wrapper.stats()["refused_in_capture"] == 1
    monkeypatch.setattr(su, "_is_capturing", lambda device: False)
    with pytest.raises(SetupStateError, match="frozen"):
        wrapper(x=b, out=out)
    wrapper(x=a, out=out)  # the prepared weight set keeps serving
    assert life.setups == 1 and wrapper.stats()["runs"] == 1
    assert len(life.runs) == 3

    # A miss under capture freezes on its own, with no prior hit.
    fresh = _Lifecycle()
    wrapper = bind(fresh)
    monkeypatch.setattr(su, "_is_capturing", lambda device: True)
    with pytest.raises(SetupStateError, match="inside CUDA-graph capture"):
        wrapper(x=a, out=out)
    monkeypatch.setattr(su, "_is_capturing", lambda device: False)
    with pytest.raises(SetupStateError, match="frozen"):
        wrapper(x=a, out=out)
    assert fresh.setups == 0 and wrapper.stats()["refused_in_capture"] == 1


def test_max_weight_sets_bounds_retained_state(monkeypatch):
    monkeypatch.setattr(su, "MAX_WEIGHT_SETS", 2)
    life = _Lifecycle()
    wrapper = bind(life)
    weights, out = [torch.ones(2) for _ in range(3)], torch.empty(2)
    wrapper(x=weights[0], out=out)
    wrapper(x=weights[1], out=out)
    with pytest.raises(SetupStateError, match="MAX_WEIGHT_SETS = 2"):
        wrapper(x=weights[2], out=out)
    wrapper(x=weights[0], out=out)  # the retained sets still serve
    assert life.setups == 2 and wrapper.stats()["weight_sets"] == 2


def test_positional_and_malformed_calls_refused():
    wrapper = bind(_Lifecycle())
    out = torch.empty(2)
    with pytest.raises(SetupStateError, match="keyword only"):
        wrapper(torch.ones(2), out=out)
    with pytest.raises(SetupStateError, match="setup input 'x' is absent"):
        wrapper(out=out)
    with pytest.raises(SetupStateError, match="setup input 'x' is not a tensor"):
        wrapper(x=1.0, out=out)


def test_release_tears_down_each_state_once_and_unpins_the_weights():
    life = _Lifecycle()
    wrapper = bind(life)
    a, b, out = torch.ones(2), torch.ones(2), torch.empty(2)
    wrapper(x=a, out=out)
    wrapper(x=b, out=out)
    life.contexts.clear()  # the contexts hold the tensors too; only the wrapper may
    pinned = weakref.ref(a)
    del a
    gc.collect()
    assert pinned() is not None  # the storage cannot be recycled under a live state
    wrapper.release()
    assert life.teardowns == [{"serial": 1}, {"serial": 2}]
    assert wrapper.stats()["weight_sets"] == 0 and wrapper.stats()["runs"] == 2
    gc.collect()
    assert pinned() is None
    wrapper.release()
    assert len(life.teardowns) == 2

    def broken(state):
        raise ValueError("busy")

    broken_life = _Lifecycle()
    broken_life.teardown = broken
    wrapper = bind(broken_life)
    wrapper(x=b, out=out)
    with pytest.raises(SetupStateError, match="teardown raised ValueError: busy"):
        wrapper.release()
    assert wrapper.stats()["weight_sets"] == 0


def test_release_attempts_every_teardown_before_raising():
    torn = []

    def flaky(state):
        if state["serial"] == 1:
            raise ValueError("busy")
        torn.append(state)

    life = _Lifecycle()
    life.teardown = flaky
    wrapper = bind(life)
    out = torch.empty(2)
    wrapper(x=torch.ones(2), out=out)
    wrapper(x=torch.ones(2), out=out)
    with pytest.raises(
        SetupStateError,
        match=r"teardown raised ValueError: busy \(1 of 2 teardowns raised\)",
    ):
        wrapper.release()
    assert torn == [{"serial": 2}] and wrapper.stats()["weight_sets"] == 0


def test_a_setup_that_re_enters_its_definition_is_refused_not_hung():
    out = torch.empty(2)
    prepared, fresh = torch.ones(2), torch.ones(2)

    def reenter(context):
        wrapper(x=prepared, out=out)  # a hit: served from the retained state
        wrapper(x=fresh, out=out)  # a miss: a nested setup, refused
        return {"serial": 0}

    life = _Lifecycle()
    wrapper = bind(life)
    wrapper(x=prepared, out=out)
    life.result = reenter
    with pytest.raises(
        SetupStateError, match="setup re-entered its own bound callable"
    ):
        wrapper(x=torch.ones(2), out=out)
    assert wrapper.stats()["weight_sets"] == 1 and len(life.runs) == 2


def test_call_device_prefers_an_accelerator_tensor_over_a_host_one():
    host, accelerated = torch.ones(2), torch.empty(2, device="meta")
    assert su._call_device((), {"lengths": host, "x": accelerated}).type == "meta"
    assert su._call_device((host,), {"x": accelerated}).type == "meta"
    assert su._call_device((), {"lengths": host}).type == "cpu"
    assert su._call_device((), {"scale": 2.0}).type == "cpu"
    # One state per process and a host table first in the call: the capture
    # check and the context's device follow the accelerator the kernel runs on.
    life = _Lifecycle()
    bind(life, run=lambda lengths, x, state, out: None, setup_inputs=())(
        lengths=host, x=accelerated, out=torch.empty(2, device="meta")
    )
    assert life.contexts[0]["device"] == "meta"


def test_a_state_does_not_survive_fork(monkeypatch):
    wrapper = bind(_Lifecycle())
    pid = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: pid + 1)
    with pytest.raises(SetupStateError, match="does not survive fork"):
        wrapper(x=torch.ones(2), out=torch.empty(2))


def test_context_has_the_documented_keys():
    def run(x, out, state, scale):
        out.copy_(x * scale)

    life = _Lifecycle()
    x, out = torch.zeros(3, 4), torch.empty(3, 4)
    bind(life, run=run, definition=DEFINITION)(x=x, out=out, scale=2.0)
    context = life.contexts[0]
    metadata = {"shape": [3, 4], "stride": [4, 1], "dtype": "float32", "device": "cpu"}
    # A Definition scalar: no geometry, the declared dtype
    # and device (the Definition's default, not the host the value lives on).
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


# ---------------------------------------------------------------------------
# Sealed packages: declaration checks, same-namespace binding, registry
# ---------------------------------------------------------------------------


def test_setup_declaration_is_verified_without_import(tmp_path):
    sealed = setup_package(
        tmp_path, {"kernel.py": b"raise AssertionError('imported')\n"}
    )
    assert sealed.setup_entry_point == ("kernel", "setup")
    assert sealed.teardown_entry_point == ("kernel", "teardown")
    assert sealed.setup_inputs == ("x",) and sealed.setup_state_param == "state"
    plain = pk.verify_package(tmp_path, seal_flashinfer_fixture(tmp_path))
    assert plain.setup_entry_point is None and plain.teardown_entry_point is None
    assert plain.setup_inputs == () and plain.setup_state_param == "state"


@pytest.mark.parametrize(
    "spec, files, definition, message",
    [
        (
            {"setup_entry_point": None},
            None,
            None,
            "teardown_entry_point requires setup_entry_point",
        ),
        (
            {"setup_entry_point": None, "teardown_entry_point": None},
            None,
            None,
            "setup_inputs requires setup_entry_point",
        ),
        ({"destination_passing_style": None}, None, None, "declared explicitly"),
        ({"setup_inputs": ["x", "x"]}, None, None, "distinct identifiers"),
        ({"setup_inputs": "x"}, None, None, "distinct identifiers"),
        ({"setup_state_param": "x"}, None, None, "'x' is also a setup input"),
        (
            {"setup_state_param": ""},
            None,
            None,
            "setup_state_param must be an identifier",
        ),
        (
            {"setup_entry_point": "kernel.py"},
            None,
            None,
            "setup_entry_point must be '<file>.py::<symbol>', got 'kernel.py'",
        ),
        (
            # BuildSpec takes no module:symbol here; verify refuses what load would.
            {"setup_entry_point": "kernel:setup"},
            None,
            None,
            "setup_entry_point must be '<file>.py::<symbol>', got 'kernel:setup'",
        ),
        (
            {"teardown_entry_point": "kernel:teardown"},
            None,
            None,
            "teardown_entry_point must be '<file>.py::<symbol>'",
        ),
        (
            {"setup_entry_point": "../kernel.py::setup"},
            None,
            None,
            "invalid Python setup_entry_point",
        ),
        (
            {"setup_entry_point": "prepare.py::setup"},
            None,
            None,
            "setup_entry_point file 'prepare.py' is not sealed",
        ),
        (
            {"teardown_entry_point": "prepare.py::teardown"},
            None,
            None,
            "teardown_entry_point file 'prepare.py' is not sealed",
        ),
        (
            {"setup_entry_point": "prepare.py::setup"},
            {
                "kernel.py": SETUP_SOURCE,
                "prepare.py": PREPARE_SOURCE,
                "prepare/__init__.py": b"",
            },
            None,
            "setup_entry_point file 'prepare.py' is shadowed",
        ),
        (
            {"setup_inputs": ["scale"]},
            None,
            DEFINITION,
            "setup input 'scale' is not a tensor input of the sealed definition.json",
        ),
        (
            {"setup_inputs": ["missing"]},
            None,
            DEFINITION,
            "setup input 'missing' is not a tensor input",
        ),
        (
            # The call already passes every Definition input and DPS output by
            # keyword, so the state keyword cannot be one of them.
            {"setup_state_param": "out"},
            None,
            DEFINITION,
            "setup_state_param 'out' is also an input or output of the sealed definition.json",
        ),
        (
            {"setup_state_param": "scale"},
            None,
            DEFINITION,
            "setup_state_param 'scale' is also an input or output",
        ),
    ],
)
def test_verify_refuses_bad_setup_declarations(
    tmp_path, spec, files, definition, message
):
    with pytest.raises(pk.SealedPackageError, match=message):
        setup_package(tmp_path, files, spec=spec, definition=definition)


def test_setup_symbol_files_must_be_listed_in_sources(tmp_path):
    # Solution.__post_init__ refuses a setup file absent from sources; verify
    # refuses it first, with the standard library, instead of load_packages.
    spec = {k: v for k, v in SETUP_SPEC.items() if k != "teardown_entry_point"}
    solution = {
        **SETUP_SOLUTION,
        "spec": {**spec, "setup_entry_point": "prepare.py::setup"},
        "sources": [{"path": "kernel.py", "content": PREPARE_KERNEL_SOURCE.decode()}],
    }
    digest = seal_package(
        tmp_path,
        {
            "solution.json": json_bytes(solution),
            "kernel.py": PREPARE_KERNEL_SOURCE,
            "prepare.py": PREPARE_SOURCE,
        },
        sidecar=QUALIFICATION,
    )
    with pytest.raises(
        pk.SealedPackageError,
        match="setup_entry_point file 'prepare.py' is not listed in sources",
    ):
        pk.verify_package(tmp_path, digest)


def test_import_entry_binds_setup_from_the_same_sealed_namespace(tmp_path):
    sealed = setup_package(
        tmp_path,
        {"kernel.py": PREPARE_KERNEL_SOURCE, "prepare.py": PREPARE_SOURCE},
        spec={
            "setup_entry_point": "prepare.py::setup",
            "teardown_entry_point": "prepare.py::teardown",
        },
        definition=DEFINITION,
    )
    assert sealed.setup_entry_point == ("prepare", "setup")
    bound = pk.import_entry(sealed)
    assert isinstance(bound, SetupBoundCallable)
    assert bound.__module__.startswith("_flashinfer_sealed_")
    namespace, module = bound.__module__.rsplit(".", 1)
    assert module == "kernel" and f"{namespace}.prepare" in sys.modules
    assert "prepare" not in sys.modules  # nothing outside the sealed namespace
    assert bound is pk.import_entry(sealed)  # one binding per digest and process
    assert list(inspect.signature(bound).parameters) == ["x", "out"]
    out = torch.empty(2)
    bound(x=torch.zeros(2), out=out)
    assert out.tolist() == [1.0, 1.0]  # prepare.py's setup state reached run
    assert pk.setup_stats() == {OPERATION: bound.stats()}
    assert bound.stats()["runs"] == 1
    assert pk.verify_package(tmp_path, sealed.digest) == sealed
    assert not list(sealed.directory.rglob("__pycache__"))


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
            {"kernel.py": b"def setup(c): pass\ndef run(x, state, out): pass\n"},
            {"teardown_entry_point": "kernel.py::missing"},
            "declared teardown_entry_point is not callable",
        ),
        (
            {
                "kernel.py": b"def setup(c): pass\ndef run(x, out): pass\ndef teardown(s): pass\n"
            },
            {},
            "no keyword parameter 'state'",
        ),
    ],
)
def test_import_entry_refuses_an_unbindable_setup(tmp_path, files, spec, message):
    with pytest.raises(pk.SealedPackageError, match=message):
        pk.import_entry(setup_package(tmp_path, files, spec=spec))
    assert pk.setup_stats() == {}


def test_a_setup_module_that_fails_to_import_is_named(tmp_path):
    sealed = setup_package(
        tmp_path,
        {"kernel.py": PREPARE_KERNEL_SOURCE, "prepare.py": b"def setup(context:\n"},
        spec={"setup_entry_point": "prepare.py::setup", "teardown_entry_point": None},
    )
    with pytest.raises(
        pk.SealedPackageError, match="cannot load sealed setup_entry_point: "
    ) as info:
        pk.import_entry(sealed)
    assert isinstance(info.value.__cause__, SyntaxError)
    assert pk.setup_stats() == {}


PREPARE_ASKING_SOURCE = (
    b"from flashinfer.trace_apply import setup_stats\n\n"
    b"BOUND = setup_stats()\n\n\n"
    b"def setup(context):\n    return 1.0\n"
)


def test_binding_imports_sealed_setup_code_outside_the_registry_lock(tmp_path):
    sealed = setup_package(
        tmp_path,
        {"kernel.py": PREPARE_KERNEL_SOURCE, "prepare.py": PREPARE_ASKING_SOURCE},
        spec={"setup_entry_point": "prepare.py::setup", "teardown_entry_point": None},
    )
    result: list = []
    worker = threading.Thread(
        target=lambda: result.append(pk.import_entry(sealed)), daemon=True
    )
    worker.start()
    worker.join(timeout=30)
    assert not worker.is_alive(), "a sealed module body waited on _bound_lock"
    (bound,) = result
    namespace = bound.__module__.rsplit(".", 1)[0]
    assert sys.modules[f"{namespace}.prepare"].BOUND == {}
    assert pk.setup_stats() == {OPERATION: bound.stats()}


def test_a_second_digest_for_a_bound_definition_is_refused(tmp_path):
    first = setup_package(tmp_path / "a")
    second = setup_package(tmp_path / "b", spec={"target_hardware": ["sm100", "sm120"]})
    assert first.digest != second.digest
    bound = pk.import_entry(first)
    with pytest.raises(
        pk.SealedPackageError,
        match=f"same definition {OPERATION!r} in this process: {first.digest}",
    ):
        pk.import_entry(second)
    assert pk.import_entry(first) is bound
    assert pk.setup_stats() == {OPERATION: bound.stats()}
    pk.release_packages()
    assert isinstance(pk.import_entry(second), SetupBoundCallable)


def test_release_packages_releases_every_binding_before_raising(tmp_path):
    broken = setup_package(
        tmp_path / "a",
        {
            "kernel.py": SETUP_SOURCE.replace(
                b'    CALLS["teardown"] += 1\n', b"    raise ValueError('busy')\n"
            )
        },
    )
    other_root = tmp_path / "b"
    other = pk.verify_package(
        other_root,
        seal_package(
            other_root,
            {
                "solution.json": json_bytes(
                    {**SETUP_SOLUTION, "definition": "other_op"}
                ),
                "kernel.py": SETUP_SOURCE,
            },
            sidecar=QUALIFICATION,
        ),
    )
    first, second = pk.import_entry(broken), pk.import_entry(other)
    out = torch.empty(2)
    first(x=torch.ones(2), out=out)
    second(x=torch.ones(2), out=out)
    with pytest.raises(
        SetupStateError,
        match="1 of 2 setup-bearing bindings failed to release: "
        f"{OPERATION}: teardown raised ValueError: busy",
    ):
        pk.release_packages()
    assert sys.modules[second.__module__].CALLS["teardown"] == 1
    assert pk.setup_stats() == {}


def test_bound_registry_is_re_owned_by_a_forked_child(tmp_path, monkeypatch):
    sealed = setup_package(tmp_path)
    bound = pk.import_entry(sealed)
    assert pk.setup_stats() == {OPERATION: bound.stats()}
    pk._bound_lock.acquire()  # a thread inside the registry at the moment of the fork
    pid = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: pid + 1)
    # The child: a fresh lock and none of the parent's bindings, instead of
    # waiting on the copied, held lock for good.
    assert pk.setup_stats() == {}
    assert not pk._bound_lock.locked()
    child = pk.import_entry(sealed)
    assert child is not bound and pk.setup_stats() == {OPERATION: child.stats()}


def test_load_packages_refuses_a_package_whose_solution_the_schema_refuses(
    tmp_path, sm_gate_off
):
    # Verification reads the Solution with the standard library; a key the
    # schema requires and the verifier does not is a refusal, not an escape
    # out of ``_load_packages_from_env``.
    authorless = {key: value for key, value in SOLUTION.items() if key != "author"}
    digest = seal_flashinfer_fixture(tmp_path, solution=authorless)
    result = pk.load_packages(tmp_path, [digest], strict=False, enable=False)
    assert result.packages == []
    assert "is not a FlashInfer Solution: 'author'" in result.errors[digest]
    with pytest.raises(pk.SealedPackageError, match="is not a FlashInfer Solution"):
        pk.load_packages(tmp_path, [digest], enable=False)


def test_load_packages_registers_the_bound_callable_and_release_tears_down(
    tmp_path, sm_gate_off
):
    sealed = setup_package(tmp_path)
    result = pk.load_packages(tmp_path, [sealed.digest])
    bound = result.callables[(OPERATION, "test_solution")]
    assert isinstance(bound, SetupBoundCallable)
    solution = result.solutions[OPERATION]
    assert solution.spec.setup_entry_point == "kernel.py::setup"
    assert preloaded_entry(OPERATION, "test_solution") is bound
    assert _get_loaded_solution(solution) is bound

    calls = sys.modules[bound.__module__].CALLS
    out = torch.empty(2)
    bound(x=torch.ones(2), out=out)
    bound(x=torch.ones(2), out=out)
    assert calls == {"setup": 2, "run": 2, "teardown": 0}
    assert pk.setup_stats()[OPERATION]["weight_sets"] == 2

    pk.release_packages()
    assert calls["teardown"] == 2
    assert preloaded_entry(OPERATION, "test_solution") is None
    assert pk.setup_stats() == {}
    assert pk.import_entry(sealed) is not bound  # a later import binds afresh


def test_a_binding_a_capture_observed_is_kept_for_the_life_of_the_process(
    tmp_path, monkeypatch
):
    # Captured graphs replay against the state addresses, so neither release()
    # nor release_packages() may free them. The registry is swapped for this
    # test so its frozen binding does not outlive it.
    monkeypatch.setattr(pk, "_bound", {})
    sealed = setup_package(tmp_path)
    bound = pk.import_entry(sealed)
    weight, out = torch.ones(2), torch.empty(2)
    bound(x=weight, out=out)  # eager: prepared
    monkeypatch.setattr(su, "_is_capturing", lambda device: True)
    bound(x=weight, out=out)  # a captured hit freezes the key set
    monkeypatch.setattr(su, "_is_capturing", lambda device: False)
    assert bound.frozen
    with pytest.raises(SetupStateError, match="kept for the life of the process"):
        bound.release()
    pk.release_packages()
    assert sys.modules[bound.__module__].CALLS["teardown"] == 0
    assert bound.stats()["weight_sets"] == 1
    assert pk.setup_stats() == {OPERATION: bound.stats()}
    assert pk.import_entry(sealed) is bound  # still the process's binding

    plain = bind(_Lifecycle())
    plain(x=weight, out=out)
    assert not plain.frozen
    plain.release()  # an unfrozen binding releases as before
    assert plain.stats()["weight_sets"] == 0


def test_exports():
    assert ta.SetupBoundCallable is SetupBoundCallable
    assert ta.SetupStateError is SetupStateError
    assert ta.release_packages is pk.release_packages
    assert ta.setup_stats is pk.setup_stats
    assert pk.API_VERSION == 2
    assert issubclass(SetupStateError, RuntimeError)
    assert issubclass(pk.SealedPackageError, RuntimeError)


# ---------------------------------------------------------------------------
# Curated path: the loader binds the same class; enable_apply end to end
# ---------------------------------------------------------------------------


def test_curated_loader_binds_the_same_class(tmp_path, monkeypatch):
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
    assert out.tolist() == [1.0, 1.0]
    assert not isinstance(
        load_solution(Solution.from_dict(SOLUTION)), SetupBoundCallable
    )
    missing = Solution.from_dict(
        {
            **SETUP_SOLUTION,
            "spec": {**SETUP_SPEC, "teardown_entry_point": "kernel.py::absent"},
        }
    )
    with pytest.raises(AttributeError, match="teardown missing function 'absent'"):
        load_solution(missing)


RMSNORM_SETUP_SOURCE = (
    "SERIAL = 0\n"
    "\n"
    "\n"
    "def setup(context):\n"
    "    global SERIAL\n"
    "    SERIAL += 1\n"
    "    return float(SERIAL)\n"
    "\n"
    "\n"
    "def run(hidden_states, weight, state, output):\n"
    "    output.fill_(state)\n"
)


def test_enable_apply_runs_setup_once_per_weight_set(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # hermetic loader cache
    solution = Solution.from_dict(
        {
            "name": "setup_rms",
            "definition": "rmsnorm_h16",
            "author": "tester",
            "spec": {
                "language": "python",
                "target_hardware": ["sm100"],
                "entry_point": "main.py::run",
                "destination_passing_style": True,
                "setup_entry_point": "main.py::setup",
                "setup_inputs": ["weight"],
            },
            "sources": [{"path": "main.py", "content": RMSNORM_SETUP_SOURCE}],
        }
    )
    assert ta.enable_apply({"rmsnorm_h16": solution}) >= 1
    first_weight, second_weight = torch.ones(16), torch.ones(16)
    first = flashinfer.norm.rmsnorm(torch.ones(8, 16), first_weight)
    again = flashinfer.norm.rmsnorm(torch.ones(3, 16), first_weight)  # same weights
    other = flashinfer.norm.rmsnorm(torch.ones(8, 16), second_weight)
    assert first.unique().tolist() == [1.0] and again.unique().tolist() == [1.0]
    assert other.unique().tolist() == [2.0]
    assert ta.stats()[FI_API]["hit"] == 3
    bound = _get_loaded_solution(solution)
    assert isinstance(bound, SetupBoundCallable)
    assert bound.stats()["runs"] == 2 and bound.stats()["weight_sets"] == 2
