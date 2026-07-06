# flashinfer: regression tests for the Ulysses module layout after merging
# the former flashinfer/comm/ulysses_a2a.py submodule into
# flashinfer/comm/ulysses.py. The function `ulysses_a2a` exported from
# flashinfer.comm used to be shadowed by the submodule of the same name;
# these tests pin the new layout and the properties the merge must preserve.

import importlib
import inspect
import subprocess
import sys

import pytest
import torch

import flashinfer.comm as comm


def test_root_exports_identity_and_callability():
    # importing the merged module must not shadow or replace any root export
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    assert callable(comm.ulysses_a2a), "root export must be the function"
    assert comm.ulysses_a2a is ulysses_mod.ulysses_a2a
    assert comm.init_ulysses_a2a is ulysses_mod.init_ulysses_a2a
    assert comm.dispose_ulysses_a2a is ulysses_mod.dispose_ulysses_a2a
    assert comm.get_ulysses_a2a_module is ulysses_mod.get_ulysses_a2a_module
    assert comm.gen_ulysses_a2a_module is ulysses_mod.gen_ulysses_a2a_module
    assert comm.UlyssesCommunicator is ulysses_mod.UlyssesCommunicator
    # topology stays its own module; ALL its root exports keep identity
    topo_mod = importlib.import_module("flashinfer.comm.ulysses_topology")
    assert comm.resolve_ulysses_backend is topo_mod.resolve_ulysses_backend
    assert comm.decide_ulysses_backend is topo_mod.decide_ulysses_backend
    assert comm.probe_ulysses_rank_topology is topo_mod.probe_ulysses_rank_topology
    assert comm.UlyssesBackendDecision is topo_mod.UlyssesBackendDecision
    assert comm.UlyssesRankTopology is topo_mod.UlyssesRankTopology
    assert comm.UlyssesBackendError is topo_mod.UlyssesBackendError
    assert comm.ULYSSES_BACKENDS is topo_mod.ULYSSES_BACKENDS
    assert ulysses_mod.SUPPORTED_WORLD_SIZES is topo_mod.SUPPORTED_WORLD_SIZES


def test_local_gen_binding_is_the_effective_patch_point():
    # the merged module early-binds gen_ulysses_a2a_module from
    # flashinfer.jit.comm; patching the ORIGIN module must therefore not be
    # relied on — the local binding is the documented patch point, and
    # get_ulysses_a2a_module must resolve it at call time
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    get = ulysses_mod.get_ulysses_a2a_module
    get.cache_clear()
    orig = ulysses_mod.gen_ulysses_a2a_module
    calls = {"n": 0}

    def fake_gen():
        calls["n"] += 1
        raise AssertionError("intercepted-gen")

    ulysses_mod.gen_ulysses_a2a_module = fake_gen
    try:
        with pytest.raises(AssertionError, match="intercepted-gen"):
            get()
        assert calls["n"] == 1
    finally:
        ulysses_mod.gen_ulysses_a2a_module = orig
        get.cache_clear()  # never leave a poisoned/half-built entry behind


def test_raw_signatures_unchanged():
    assert list(inspect.signature(comm.init_ulysses_a2a).parameters) == [
        "out_ipc_ptrs",
        "signal_ipc_ptrs",
        "rank",
        "world_size",
        "full_nvlink",
    ]
    assert list(inspect.signature(comm.ulysses_a2a).parameters) == [
        "fa",
        "inp",
        "out",
        "B",
        "S_local",
        "H",
        "D",
        "mode",
    ]
    assert list(inspect.signature(comm.dispose_ulysses_a2a).parameters) == ["fa"]


def test_old_submodule_path_removed():
    # intentional cleanup: the undocumented flashinfer.comm.ulysses_a2a module
    # path is gone (it was unreachable via attribute access anyway, shadowed
    # by the function of the same name)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("flashinfer.comm.ulysses_a2a")


def test_import_triggers_no_jit_build():
    # importing the package and the merged module must not compile anything:
    # the JIT module cache has to still be empty afterwards
    code = (
        "import flashinfer.comm, flashinfer.comm.ulysses; "
        "from flashinfer.comm.ulysses import get_ulysses_a2a_module; "
        "assert get_ulysses_a2a_module.cache_info().currsize == 0, "
        "get_ulysses_a2a_module.cache_info()"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_monkeypatch_targets_intercept(monkeypatch):
    # the documented patch points on the merged module must actually intercept
    # the raw wrapper (resolution happens at call time via module globals)
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")

    def _boom(*args, **kwargs):
        raise AssertionError("intercepted")

    monkeypatch.setattr(ulysses_mod, "get_ulysses_a2a_module", _boom)
    with pytest.raises(AssertionError, match="intercepted"):
        comm.init_ulysses_a2a([0, 0], [0, 0], 0, 2, True)

    # and the communicator's kernel-call site resolves ulysses_a2a from the
    # same module globals. Hardware-independent: operand validation only
    # compares against the communicator's recorded device, so a CPU-bound
    # fake communicator exercises the exact call site without any GPU.
    monkeypatch.setattr(ulysses_mod, "ulysses_a2a", _boom)
    good = torch.zeros(1, 2, 2, 4, dtype=torch.float16)
    comm_obj = object.__new__(ulysses_mod.UlyssesCommunicator)
    comm_obj._state = "open"
    comm_obj.backend = "nvlink"
    comm_obj.world_size = 2
    comm_obj.max_elems = 1 << 20
    comm_obj.dtype = torch.float16
    comm_obj.device = good.device
    comm_obj._fa = 123
    with pytest.raises(AssertionError, match="intercepted"):
        comm_obj.scatter_heads(good)
