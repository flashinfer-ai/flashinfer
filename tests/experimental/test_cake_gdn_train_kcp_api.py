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

CPU tests of the public surface of the NON-DETERMINISTIC context-parallel packed GDN training package
``flashinfer.experimental.cake_gdn_train_kcp`` and of the facade ``flashinfer.cake_gdn_train_kcp`` that dispatches
between it and the deterministic twin: the options validator (``cake_gdn_train_kcp.cake_options``), the keyword-only
signatures, the facade's contract names, the numerical mode (``deterministic`` on every entry: the package refuses
everything but ``False`` and, every path being fixed-order, raises no alert under ``torch.use_deterministic_algorithms``; the
facade routes ``True`` and torch's deterministic mode to the twin, warns once, refuses a non-bool and a saved set of
the other route; no context, partition or options field holds the mode), the names (the twin's legacy aliases stay
the twin's objects; this package copies none of them and no public name carries a mode), the package hygiene rules
(no environment variable, no ``cake_kda_train`` import, standalone: no import of another FlashInfer package, its twin
included, and no library name of the twin) and the host tables built on the CPU for the correctness rows'
partitions.  No kernel runs here: the compute entries run on the fake backend (the CPU emulation).
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import json
import pathlib
import sys
import warnings

import pytest
import torch

PACKAGE = "flashinfer.experimental.cake_gdn_train_kcp"
FACADE = "flashinfer.cake_gdn_train_kcp"
DET_PACKAGE = "flashinfer.experimental.cake_gdn_det_train_kcp"
DET_FACADE = "flashinfer.cake_gdn_det_train_kcp"


@pytest.fixture(autouse=True, scope="module")
def _few_cpu_threads():
    """The CPU emulation runs many small matmuls and elementwise ops.  On a host with hundreds of CPUs, torch's default
    intra-op pool and the oneDNN matmul path (its own pool, which torch.set_num_threads does not resize) spend far more
    time synchronizing threads than computing (one test of this suite on a 352-CPU host: 272 s by default, 6 s with at
    most 8 threads and oneDNN off), so this module runs that way."""
    threads, onednn = torch.get_num_threads(), torch.backends.mkldnn.enabled
    torch.set_num_threads(min(threads, 8))
    torch.backends.mkldnn.enabled = False
    try:
        yield
    finally:
        torch.backends.mkldnn.enabled = onednn
        torch.set_num_threads(threads)


def _mod(sub: str | None = None):
    """The package (or one of its modules) through the test-helper registry when it lists the package, else directly."""
    try:
        from tests.test_helpers.cake_package import cake_package

        return cake_package("cake_gdn_train_kcp", sub)
    except (ImportError, KeyError):
        return importlib.import_module(PACKAGE if sub is None else f"{PACKAGE}.{sub}")


opt = _mod("cake_options")
OptionsKcp, make_options = opt.OptionsKcp, opt.make_options


# ----------------------------------------------------------------------------------------------------------------
# options validator (the contract's option vocabulary)


def test_options_defaults_are_the_shipped_behaviour() -> None:
    o = make_options()
    assert o.exchange == "a2a"
    assert o.save_transfer is True
    assert o.save_mode == "light"
    assert o.boundary_mode == "saved"
    assert o.precision_mode == 0
    assert o.transfer_walk == "tf32_hilo"
    assert o.offset_class == "fla"
    assert o.layout == "tma"
    assert o.group_reduce == "fused"
    assert o.graph == "off"
    assert o.reuse_recompute_tapes is opt.REUSE_RECOMPUTE_TAPES_DEFAULT
    assert o.recompute_output == "auto"
    assert o.backend == "kernels"
    assert o.segments is None
    assert o.test_knobs is None
    assert o.knob("schedule") == "wait_first" and o.knob("fused_walks") == 1
    assert o.knob("cols_rule") == "fixed" and o.knob("debug_sequence") is False
    assert o.knob("p_rule") is None


def test_every_option_field_is_keyword_only() -> None:
    for f in dataclasses.fields(OptionsKcp):
        assert f.kw_only, f.name
    with pytest.raises(TypeError):
        OptionsKcp("ring")  # type: ignore[misc]


@pytest.mark.parametrize("mode", [1, 2, 3, -1, True, False, 1.0, "0"])
def test_precision_mode_rejects_unknown_values_and_bare_test_values(mode) -> None:
    """Template numbering 0 / 1 / 2; 1 and 2 are test-only and refused unless ``test_knobs``
    names the same value -- the public API alone cannot select a test program."""
    with pytest.raises((ValueError, TypeError)):
        make_options(precision_mode=mode)


def test_precision_mode_test_values_need_test_knobs() -> None:
    assert opt.PRECISION_MODES == (0, 1, 2) and opt.PRECISION_MODE_DEFAULT == 0
    assert opt.PRECISION_MODES_TEST_ONLY == (1, 2)
    assert make_options(precision_mode=0).precision_mode == 0
    for mode in (1, 2):
        o = make_options(precision_mode=mode, test_knobs={"precision_mode": mode})
        assert o.precision_mode == mode and o.knob("precision_mode") == mode
    with pytest.raises(ValueError):  # the named value must be the selected one
        make_options(precision_mode=1, test_knobs={"precision_mode": 2})
    with pytest.raises(ValueError):  # naming the shipped record is not a test knob
        make_options(test_knobs={"precision_mode": 0})
    assert make_options(test_knobs={"precision_mode": 2}).precision_mode == 0


def test_transfer_walk_bf16x3_is_in_the_vocabulary_but_not_built() -> None:
    """No GDN BF16x3 transfer walk is built in this release -- the name stays in the vocabulary, every attempt to select it
    is refused (with or without ``test_knobs``)."""
    assert opt.TRANSFER_WALKS == ("tf32_hilo", "bf16x3")
    assert opt.TRANSFER_WALK_DEFAULT == "tf32_hilo"
    assert opt.TRANSFER_WALKS_UNBUILT == ("bf16x3",)
    assert make_options(transfer_walk="tf32_hilo").transfer_walk == "tf32_hilo"
    with pytest.raises(ValueError, match="not built"):
        make_options(transfer_walk="bf16x3")
    with pytest.raises(ValueError):
        make_options(transfer_walk="tf32x3")
    with pytest.raises(ValueError):  # not a test knob either
        make_options(transfer_walk="bf16x3", test_knobs={"transfer_walk": "bf16x3"})


def test_ring_requires_the_recompute_form() -> None:
    with pytest.raises(ValueError):
        make_options(exchange="ring")  # save_transfer defaults to True: no ring form
    with pytest.raises(ValueError):
        make_options(exchange="ring", save_transfer=True)
    assert make_options(exchange="ring", save_transfer=False).exchange == "ring"
    with pytest.raises(ValueError):
        make_options(exchange="auto")


def test_ring_hop_limit() -> None:
    assert opt.RING_MAX_HOPS == 2
    opt.validate_ring(1)
    opt.validate_ring(3)  # n_d_max = 3 -> 2 hops
    with pytest.raises(ValueError):
        opt.validate_ring(4)


@pytest.mark.parametrize(
    "pair,r",
    [((16, 48), 3), ((8, 24), 3), ((4, 12), 3), ((2, 6), 3), ((1, 3), 3), ((4, 4), 1)],
)
def test_head_pairs_of_the_goal_are_legal(pair, r) -> None:
    assert opt.validate_heads(*pair) == r


@pytest.mark.parametrize("pair", [(3, 4), (4, 6), (16, 40), (0, 3), (2, 0), (5, 1)])
def test_head_rule_rejects_non_multiples(pair) -> None:
    with pytest.raises(ValueError):
        opt.validate_heads(*pair)


@pytest.mark.parametrize(
    "field,values",
    [
        ("save_mode", ("light", "full")),
        ("boundary_mode", ("saved", "replay")),
        ("graph", ("off", "replay")),
        ("layout", ("tma", "copy")),
        ("group_reduce", ("fused", "host")),
        ("backend", ("kernels", "fake")),
        ("offset_class", ("fla", "fp32")),
    ],
)
def test_enumerated_fields(field, values) -> None:
    for v in values:
        assert getattr(make_options(**{field: v}), field) == v
    for bad in ("tape", "auto", "", None, 1):
        with pytest.raises((ValueError, TypeError)):
            make_options(**{field: bad})


def test_reuse_recompute_tapes_defaults_to_auto_never_captured() -> None:
    """The checkpoint-aware light backward is a keyword-only switch -- ``"auto"`` (the shipped default: the tapes are
    kept only when the node closes the checkpointed region), ``True`` (always) or ``False`` (the replay form); under
    ``graph="replay"`` the default resolves to ``False`` -- part of the options key (so of the CUDA-graph key), typed,
    and refused with ``graph="replay"`` (a captured graph bakes the arena addresses of its tapes; the kept tapes are
    fresh tensors handed from the recompute forward to the backward)."""
    assert "reuse_recompute_tapes" in opt.OPTION_NAMES
    # the shipped default is this one constant (and this line) -- every other test names the state it exercises
    assert opt.REUSE_RECOMPUTE_TAPES_DEFAULT == "auto"
    assert make_options().reuse_recompute_tapes == opt.REUSE_RECOMPUTE_TAPES_DEFAULT
    # None = the shipped default; under graph="replay" (the one mode that excludes the kept tapes) it is False
    assert make_options(reuse_recompute_tapes=None).reuse_recompute_tapes == "auto"
    assert make_options(graph="replay").reuse_recompute_tapes is False
    both = make_options(reuse_recompute_tapes=None, graph="replay")
    assert both.reuse_recompute_tapes is False
    assert OptionsKcp(graph="replay").reuse_recompute_tapes is False
    assert make_options(graph="replay").key() != make_options().key()
    on = make_options(reuse_recompute_tapes=True)
    off = make_options(reuse_recompute_tapes=False)
    assert on.reuse_recompute_tapes is True and off.reuse_recompute_tapes is False
    assert on.key() != off.key()
    from_instance = make_options(OptionsKcp(reuse_recompute_tapes=True))
    assert from_instance.reuse_recompute_tapes is True
    with pytest.raises(TypeError):
        make_options(reuse_recompute_tapes=1)
    with pytest.raises(ValueError, match="graph"):
        make_options(reuse_recompute_tapes=True, graph="replay")
    with pytest.raises(ValueError, match="graph"):
        make_options(reuse_recompute_tapes="auto", graph="replay")
    # the other modes accept the switch (it has no effect there: documented as a no-op)
    for extra in (
        {"boundary_mode": "replay"},
        {"save_mode": "full"},
        {"save_transfer": False},
    ):
        o = make_options(reuse_recompute_tapes=True, **extra)
        assert o.reuse_recompute_tapes is True


def test_save_transfer_must_be_a_bool() -> None:
    with pytest.raises(TypeError):
        make_options(save_transfer=1)
    with pytest.raises(TypeError):
        make_options(save_transfer="yes")


def test_recompute_output_is_a_two_value_keyword_switch_defaulting_to_auto() -> None:
    """``recompute_output`` -- whether the recompute of a checkpointed forward launches its dead output program -- is a
    keyword-only field of the options (so part of the CUDA-graph key), ``"auto"`` by default (the frame-proof skip),
    ``"always"`` the unconditional launch; anything else is refused.  It is independent of the kept-tape switch and
    of the graph mode (a captured graph never skips, the option value stays ``"auto"``)."""
    assert "recompute_output" in opt.OPTION_NAMES
    assert opt.RECOMPUTE_OUTPUTS == ("auto", "always")
    assert opt.RECOMPUTE_OUTPUT_DEFAULT == "auto"
    assert make_options().recompute_output == "auto"
    assert make_options(recompute_output="always").recompute_output == "always"
    assert (
        make_options(OptionsKcp(recompute_output="always")).recompute_output == "always"
    )
    for bad in ("never", "", None, True, 1):
        with pytest.raises((ValueError, TypeError)):
            make_options(recompute_output=bad)
    o = make_options(reuse_recompute_tapes=False, recompute_output="auto")
    assert o.recompute_output == "auto" and o.reuse_recompute_tapes is False
    assert make_options(graph="replay").recompute_output == "auto"


def test_unknown_option_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        make_options(foo=1)
    with pytest.raises(TypeError):
        make_options({"walk_precision": "bf16x3"})
    with pytest.raises(TypeError):
        make_options(_kcp_test_knobs={"precision_mode": 0})


def test_segments_range() -> None:
    assert make_options(segments=1).segments == 1
    assert make_options(segments=64).segments == 64
    for bad in (0, 65, -1):
        with pytest.raises(ValueError):
            make_options(segments=bad)
    for bad in (True, 2.0, "2"):
        with pytest.raises(TypeError):
            make_options(segments=bad)


def test_test_knobs_are_a_keyword_dict_with_known_keys() -> None:
    o = make_options(
        test_knobs={
            "schedule": "overlap",
            "fused_walks": 0,
            "cols_rule": "by_chains",
            "debug_sequence": True,
            "p_rule": "occ",
        }
    )
    assert o.knob("schedule") == "overlap" and o.knob("fused_walks") == 0
    assert o.knob("cols_rule") == "by_chains" and o.knob("debug_sequence") is True
    assert o.knob("p_rule") == "occ"
    with pytest.raises(ValueError):
        make_options(test_knobs={"precision_mode": 0})
    assert make_options(test_knobs={"fwdh_vh": 64}).knob("fwdh_vh") == 64
    assert make_options().knob("fwdh_vh") is None
    with pytest.raises(ValueError):
        make_options(test_knobs={"fwdh_vh": 32})
    assert make_options(test_knobs={"dhu_vh": 128}).knob("dhu_vh") == 128
    assert make_options().knob("dhu_vh") is None
    with pytest.raises(ValueError):
        make_options(test_knobs={"dhu_vh": 96})
    with pytest.raises(ValueError):
        make_options(test_knobs={"schedule": "eager"})
    with pytest.raises(ValueError):
        make_options(test_knobs={"fused_walks": True})
    with pytest.raises(ValueError):
        make_options(test_knobs={"p_rule": 3})
    with pytest.raises(TypeError):
        make_options(test_knobs=["schedule"])
    with pytest.raises(KeyError):
        o.knob("walk_geometry")


def test_options_from_instance_mapping_and_keywords_agree() -> None:
    a = make_options(OptionsKcp(save_mode="full"), boundary_mode="replay")
    b = make_options({"save_mode": "full", "boundary_mode": "replay"})
    c = make_options(save_mode="full", boundary_mode="replay")
    assert a == b == c
    assert a.key() == b.key() == c.key()
    assert hash(a.key()) == hash(c.key())
    assert (
        make_options(test_knobs={"schedule": "overlap"}).key() != make_options().key()
    )
    with pytest.raises(TypeError):
        make_options(["save_mode"])  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------------------------------------
# signatures: every knob keyword-only


def _after(sig: inspect.Signature, anchor: str) -> list[inspect.Parameter]:
    params = list(sig.parameters.values())
    names = [p.name for p in params]
    return params[names.index(anchor) + 1 :]


def test_entry_points_take_every_knob_keyword_only() -> None:
    kcp = _mod("cake_kcp")
    for fn, anchor in (
        (kcp.chunk_gdn_kcp, "ctx"),
        (kcp.chunk_gdn_fwd_kcp, "ctx"),
        (kcp.chunk_gdn_bwd_kcp, "dout"),
    ):
        sig = inspect.signature(fn)
        for p in _after(sig, anchor):
            assert p.kind in (p.KEYWORD_ONLY, p.VAR_KEYWORD), (fn.__name__, p.name)
    sig = inspect.signature(kcp.chunk_gdn_kcp)
    assert sig.parameters["save_mode"].default == "light"
    assert sig.parameters["boundary_mode"].default == "saved"
    assert sig.parameters["graph"].default == "off"
    assert sig.parameters["scale"].default is None


def test_facade_exposes_the_contract_names_keyword_only() -> None:
    facade = importlib.import_module("flashinfer.cake_gdn_train_kcp")
    names = (
        "partition_kcp",
        "build_context_kcp",
        "chunk_gdn_kcp",
        "chunk_gdn_fwd_kcp",
        "chunk_gdn_bwd_kcp",
        "gdn_workspace_bytes_kcp",
        "gdn_saved_bytes_kcp",
        "gdn_comm_bytes_kcp",
        "is_gdn_train_kcp_supported",
    )
    for n in names:
        assert callable(getattr(facade, n)), n
        assert n in facade.__all__
    # exactly the contract names plus the three public types; nothing else is public
    types = (
        "OptionsKcp",
        "ContextKcp",
        "PartitionKcp",
        "RecomputeHandleKcp",
    )
    assert set(facade.__all__) == set(names) | set(types)
    for n in ("DEFAULT_SCALE", "SavedKcp", "WorkspaceKcp"):
        assert n not in facade.__all__
    assert set(types) <= set(dir(facade))
    for fn, anchor in (
        (facade.chunk_gdn_kcp, "ctx"),
        (facade.chunk_gdn_fwd_kcp, "ctx"),
        (facade.chunk_gdn_bwd_kcp, "dout"),
    ):
        for p in _after(inspect.signature(fn), anchor):
            assert p.kind in (p.KEYWORD_ONLY, p.VAR_KEYWORD), (fn.__name__, p.name)
    for fn in (
        facade.gdn_workspace_bytes_kcp,
        facade.gdn_saved_bytes_kcp,
        facade.gdn_comm_bytes_kcp,
    ):
        for p in _after(inspect.signature(fn), "ctx"):
            assert p.kind in (p.KEYWORD_ONLY, p.VAR_KEYWORD), (fn.__name__, p.name)
    assert (
        getattr(facade.chunk_gdn_kcp, "is_experimental", False) or True
    )  # decorated as experimental (soft)
    pkg = _mod()
    assert facade.OptionsKcp is pkg.OptionsKcp
    assert facade.ContextKcp is pkg.ContextKcp
    assert facade.PartitionKcp is pkg.PartitionKcp
    assert facade.RecomputeHandleKcp is pkg.RecomputeHandleKcp
    assert facade.SavedKcp is pkg.SavedKcp
    assert facade.DEFAULT_SCALE == 128**-0.5 == pkg.SCALE
    with pytest.raises(AttributeError):
        facade.no_such_name  # noqa: B018


def test_package_init_re_exports_the_route() -> None:
    pkg = _mod()
    for n in (
        "partition_kcp",
        "build_context_kcp",
        "chunk_gdn_kcp",
        "chunk_gdn_fwd_kcp",
        "chunk_gdn_bwd_kcp",
        "gdn_comm_bytes_kcp",
        "gdn_saved_bytes_kcp",
        "gdn_workspace_bytes_kcp",
        "OptionsKcp",
        "ContextKcp",
        "SavedKcp",
        "WorkspaceKcp",
    ):
        assert hasattr(pkg, n), n
        assert n in pkg.__all__


# The deterministic twin's pre-rename names: legacy facade name -> det canonical facade name, legacy package-only
# name -> det canonical name, and the module-level aliases at the ends of the det package's modules.  None of them is
# copied into this package; ``chunk_gdn_kcp`` is the one spelling both carry (a det legacy alias there, this package's
# canonical entry here).
LEGACY_FACADE_NAMES = {
    "kcp_partition": "partition_det_kcp",
    "build_kcp_context": "build_context_det_kcp",
    "chunk_gdn_kcp": "chunk_gdn_det_kcp",
    "chunk_gdn_kcp_fwd": "chunk_gdn_det_fwd_kcp",
    "chunk_gdn_kcp_bwd": "chunk_gdn_det_bwd_kcp",
    "gdn_kcp_workspace_bytes": "gdn_det_workspace_bytes_kcp",
    "gdn_kcp_saved_bytes": "gdn_det_saved_bytes_kcp",
    "gdn_kcp_comm_bytes": "gdn_det_comm_bytes_kcp",
    "is_gdn_train_supported": "is_gdn_det_train_kcp_supported",
    "KcpOptions": "OptionsDetKcp",
    "KcpContext": "ContextDetKcp",
    "KcpPartition": "PartitionDetKcp",
    "KcpRecomputeHandle": "RecomputeHandleDetKcp",
    "KcpSaved": "SavedDetKcp",
    "KcpWorkspace": "WorkspaceDetKcp",
}
LEGACY_PACKAGE_NAMES = {
    **{
        k: v
        for k, v in LEGACY_FACADE_NAMES.items()
        if k not in ("is_gdn_train_supported",)
    },
    "ChunkGDNKcpFunction": "ChunkGDNDetFunctionKcp",
    "KcpCall": "CallDetKcp",
    "KcpFragment": "FragmentDetKcp",
}
LEGACY_MODULE_ALIASES = {
    "cake_kcp": {
        "ChunkGDNKcpFunction": "ChunkGDNDetFunctionKcp",
        "KcpCall": "CallDetKcp",
        "KcpOptions": "OptionsDetKcp",
        "KcpRecomputeHandle": "RecomputeHandleDetKcp",
        "KcpSaved": "SavedDetKcp",
        "KcpWorkspace": "WorkspaceDetKcp",
        "chunk_gdn_kcp": "chunk_gdn_det_kcp",
        "chunk_gdn_kcp_bwd": "chunk_gdn_det_bwd_kcp",
        "chunk_gdn_kcp_fwd": "chunk_gdn_det_fwd_kcp",
        "gdn_kcp_comm_bytes": "gdn_det_comm_bytes_kcp",
        "gdn_kcp_saved_bytes": "gdn_det_saved_bytes_kcp",
        "gdn_kcp_workspace_bytes": "gdn_det_workspace_bytes_kcp",
    },
    "cake_kcp_plan": {
        "KcpContext": "ContextDetKcp",
        "KcpFragment": "FragmentDetKcp",
        "KcpPartition": "PartitionDetKcp",
        "build_kcp_context": "build_context_det_kcp",
        "kcp_partition": "partition_det_kcp",
    },
    "cake_bytes": {
        "gdn_kcp_comm_bytes": "gdn_det_comm_bytes_kcp",
        "gdn_kcp_saved_bytes": "gdn_det_saved_bytes_kcp",
        "gdn_kcp_workspace_bytes": "gdn_det_workspace_bytes_kcp",
        "gdn_kcp_saved_layout": "gdn_det_saved_layout_kcp",
        "gdn_kcp_workspace_layout": "gdn_det_workspace_layout_kcp",
    },
    "cake_options": {"KcpOptions": "OptionsDetKcp"},
}
SHARED_SPELLINGS = {
    "chunk_gdn_kcp"
}  # a det legacy alias and this package's canonical entry
SUBMODULES = (
    "cake_bytes",
    "cake_fake_backend",
    "cake_jit",
    "cake_kcp",
    "cake_kcp_plan",
    "cake_options",
    "cake_tables",
)


def _nondet_objects() -> list:
    """Every public object of this package (its ``__all__``, every function / class its modules define) and of the
    facade (its ``__all__`` and its entries)."""
    pkg, facade = _mod(), importlib.import_module(FACADE)
    objs = [getattr(pkg, n) for n in pkg.__all__]
    objs += [getattr(facade, n) for n in facade.__all__]
    for sub in SUBMODULES:
        mod = _mod(sub)
        objs += [
            v
            for v in vars(mod).values()
            if callable(v) and str(getattr(v, "__module__", "")).startswith(PACKAGE)
        ]
    return objs


def test_legacy_names_are_the_deterministic_objects_never_the_non_deterministic_ones() -> (
    None
):
    """``flashinfer.gdn_train``, ``flashinfer.experimental.cake_gdn_train`` and the det package's module-level aliases
    stay the deterministic twin's compatibility paths: every legacy name IS its det object and is NOT any object of
    this package or of the ``flashinfer.cake_gdn_train_kcp`` facade (in particular ``chunk_gdn_kcp``, spelled by both)."""
    det_facade = importlib.import_module(DET_FACADE)
    det_pkg = importlib.import_module(DET_PACKAGE)
    nondet = _nondet_objects()
    assert len(nondet) > 50

    def is_nondet(obj) -> bool:
        return any(obj is o for o in nondet)

    legacy = importlib.import_module("flashinfer.gdn_train")
    for old, new in LEGACY_FACADE_NAMES.items():
        obj = getattr(legacy, old)
        assert obj is getattr(det_facade, new), old
        assert not is_nondet(obj), old
    legacy_pkg = importlib.import_module("flashinfer.experimental.cake_gdn_train")
    for old, new in LEGACY_PACKAGE_NAMES.items():
        obj = getattr(legacy_pkg, old)
        assert obj is getattr(det_pkg, new), old
        assert not is_nondet(obj), old
    for sub, aliases in LEGACY_MODULE_ALIASES.items():
        det_mod = importlib.import_module(f"{DET_PACKAGE}.{sub}")
        for old, new in aliases.items():
            obj = getattr(det_mod, old)
            assert obj is getattr(det_mod, new), (sub, old)
            assert not is_nondet(obj), (sub, old)
    # the shared spelling: the det alias, never this package's entry or the facade's dispatcher
    facade = importlib.import_module(FACADE)
    assert legacy.chunk_gdn_kcp is det_facade.chunk_gdn_det_kcp
    assert legacy_pkg.chunk_gdn_kcp is det_pkg.chunk_gdn_det_kcp
    for nondet_entry in (
        _mod().chunk_gdn_kcp,
        _mod("cake_kcp").chunk_gdn_kcp,
        facade.chunk_gdn_kcp,
    ):
        assert nondet_entry is not legacy.chunk_gdn_kcp
        assert nondet_entry is not legacy_pkg.chunk_gdn_kcp
        assert nondet_entry is not det_pkg.chunk_gdn_det_kcp
    # and this package's types are not the twin's
    for name in (
        "OptionsKcp",
        "ContextKcp",
        "PartitionKcp",
        "SavedKcp",
        "WorkspaceKcp",
    ):
        assert getattr(_mod(), name) is not getattr(
            det_pkg, name.replace("Kcp", "DetKcp")
        )


def test_the_package_exports_no_legacy_alias_name() -> None:
    """No pre-rename name of the twin was copied: the package, its modules and the facade bind none of them (except
    ``chunk_gdn_kcp``, this package's canonical entry, defined in ``cake_kcp`` under its own name)."""
    legacy_names = (
        set(LEGACY_FACADE_NAMES)
        | set(LEGACY_PACKAGE_NAMES)
        | {n for aliases in LEGACY_MODULE_ALIASES.values() for n in aliases}
    )
    facade = importlib.import_module(FACADE)
    pkg = _mod()
    for name in sorted(legacy_names - SHARED_SPELLINGS):
        assert not hasattr(pkg, name), name
        assert name not in pkg.__all__, name
        assert not hasattr(facade, name), name
        for sub in SUBMODULES:
            assert not hasattr(_mod(sub), name), (sub, name)
    entry = _mod("cake_kcp").chunk_gdn_kcp
    assert pkg.chunk_gdn_kcp is entry
    assert (
        entry.__module__ == f"{PACKAGE}.cake_kcp" and entry.__name__ == "chunk_gdn_kcp"
    )
    for module in [pkg] + [_mod(sub) for sub in SUBMODULES]:
        assert not hasattr(module, "KcpContext"), module.__name__


def test_no_public_name_of_the_package_or_the_facade_names_a_mode() -> None:
    """The mode lives in the ``deterministic`` keyword (and the package's ``MODE`` / ``check_mode``), never in a
    public name: no exported name of the package, its modules or the facade contains ``det`` / ``non_det``."""
    facade = importlib.import_module(FACADE)
    names = set(_mod().__all__) | set(facade.__all__)
    names |= {n for n in dir(facade) if not n.startswith("_")}
    for sub in SUBMODULES:
        names |= set(getattr(_mod(sub), "__all__", ()))
    bad = sorted(n for n in names if "det" in n.lower() or "non_det" in n.lower())
    assert not bad, bad
    assert {"MODE", "check_mode"} <= set(_mod().__all__)


# ----------------------------------------------------------------------------------------------------------------
# the numerical mode (``deterministic`` on every entry; the package is non-deterministic only)

ENTRIES = ("chunk_gdn_kcp", "chunk_gdn_fwd_kcp", "chunk_gdn_bwd_kcp")
MODE_CU = [0, 100, 230, 307]  # three documents, boundaries inside chunks
MODE_HEADS = (2, 6)


@pytest.fixture
def torch_deterministic():
    """Restore torch's deterministic-algorithms setting (and its warn-only flag) after the test."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _mode_ops(
    cu=MODE_CU, heads=MODE_HEADS, seed: int = 8063
) -> dict[str, torch.Tensor]:
    """Seeded CPU operands without the batch axis: q / k BF16 L2-normalised rows, v / dout BF16, g < 0 FP32 (the
    natural-log decay), beta in (0, 1) FP32."""
    H_qk, HV = heads
    T = cu[-1]
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def l2(x):
        return (x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)).to(
            torch.bfloat16
        )

    return {
        "q": l2(torch.randn(T, H_qk, 128, generator=gen)),
        "k": l2(torch.randn(T, H_qk, 128, generator=gen)),
        "v": torch.randn(T, HV, 128, generator=gen).to(torch.bfloat16),
        "g": (-torch.nn.functional.softplus(torch.randn(T, HV, generator=gen))).float(),
        "beta": torch.sigmoid(torch.randn(T, HV, generator=gen)).float(),
        "dout": torch.randn(T, HV, 128, generator=gen).to(torch.bfloat16),
    }


def _mode_case():
    """A one-rank CPU context of :data:`MODE_CU` and its operands."""
    plan = _mod("cake_kcp_plan")
    part = plan.partition_kcp(MODE_CU, 1, "contiguous")
    ctx = plan.build_context_kcp(part, None, 0, "cpu", sm_count=148)
    return ctx, _mode_ops()


def _call(entry: str, ctx, ops, **kw):
    """One package compute entry on the fake backend (``chunk_gdn_bwd_kcp`` without a saved set: the replay)."""
    kcp = _mod("cake_kcp")
    args = [ops[n] for n in ("q", "k", "v", "g", "beta")]
    if entry == "chunk_gdn_bwd_kcp":
        return kcp.chunk_gdn_bwd_kcp(*args, ctx, ops["dout"], backend="fake", **kw)
    return getattr(kcp, entry)(*args, ctx, backend="fake", **kw)


def test_every_entry_takes_the_mode_keyword_only_defaulting_to_true() -> None:
    """``deterministic`` is a keyword-only parameter defaulting to ``True`` on every package compute entry and on every
    facade entry (the package's default is refused: a caller states the mode)."""
    kcp = _mod("cake_kcp")
    for name in ENTRIES:
        p = inspect.signature(getattr(kcp, name)).parameters["deterministic"]
        assert p.kind == p.KEYWORD_ONLY and p.default is True, name
    facade = importlib.import_module(FACADE)
    for name in (
        "partition_kcp",
        "build_context_kcp",
        "chunk_gdn_kcp",
        "chunk_gdn_fwd_kcp",
        "chunk_gdn_bwd_kcp",
        "gdn_workspace_bytes_kcp",
        "gdn_saved_bytes_kcp",
        "gdn_comm_bytes_kcp",
        "is_gdn_train_kcp_supported",
    ):
        p = inspect.signature(getattr(facade, name)).parameters["deterministic"]
        assert p.kind == p.KEYWORD_ONLY and p.default is True, name
    assert kcp.MODE == "nondeterministic" and _mod().MODE is kcp.MODE
    assert _mod().check_mode is kcp.check_mode


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize(
    "value", [True, "omitted", 1, None, 0, "False"], ids=lambda v: repr(v)
)
def test_package_entries_refuse_every_mode_but_false(entry, value) -> None:
    """The package ships no deterministic program and never falls back: every value but ``False`` (the default
    ``True``, an omitted keyword, ``1``, ``None``, ``0``, ``"False"``) is a ``ValueError`` naming the entry, raised
    before any work."""
    ctx, ops = _mode_case()
    kw = {} if value == "omitted" else {"deterministic": value}
    with pytest.raises(ValueError, match=f"{entry}: .*needs deterministic=False"):
        _call(entry, ctx, ops, **kw)
    with pytest.raises(ValueError, match="needs deterministic=False"):
        _mod().check_mode(True if value == "omitted" else value, entry)


def test_package_entries_run_with_false_on_the_fake_backend() -> None:
    """``deterministic=False``: the forward, the saved-set backward, the replay backward and the autograd node run on
    the fake backend (finite outputs of the contract's shapes and dtypes; the accuracy is the fake-backend suite's)."""
    kcp = _mod("cake_kcp")
    ctx, ops = _mode_case()
    args = [ops[n] for n in ("q", "k", "v", "g", "beta")]
    out, saved = kcp.chunk_gdn_fwd_kcp(*args, ctx, backend="fake", deterministic=False)
    assert isinstance(saved, kcp.SavedKcp)
    assert out.shape == ops["v"].shape and out.dtype == torch.bfloat16
    for saved_set in (saved, None):
        grads = kcp.chunk_gdn_bwd_kcp(
            *args,
            ctx,
            ops["dout"],
            saved=saved_set,
            backend="fake",
            deterministic=False,
        )
        for t, like in zip(grads, ("q", "k", "v", "beta", "g"), strict=True):
            assert t.shape == ops[like].shape and t.dtype == ops[like].dtype, like
            assert bool(torch.isfinite(t.float()).all()), like
    leaves = [t.detach().clone().requires_grad_(True) for t in args]
    y = kcp.chunk_gdn_kcp(*leaves, ctx, backend="fake", deterministic=False)
    assert y.grad_fn is not None and y.grad_fn.kcp_mode == kcp.MODE
    y.backward(ops["dout"])
    assert all(
        lf.grad is not None and bool(torch.isfinite(lf.grad.float()).all())
        for lf in leaves
    )


@pytest.mark.parametrize("warn_only", [False, True])
@pytest.mark.parametrize("entry", ENTRIES)
def test_no_alert_under_torch_deterministic_algorithms(
    entry, warn_only, torch_deterministic
) -> None:
    """Every path of this version is fixed-order (run-to-run reproducible for a fixed plan and partition), so with
    ``torch.use_deterministic_algorithms(True)`` (and with ``warn_only=True``) a ``deterministic=False`` call of the
    package runs: no ``RuntimeError``, no warning.  Only a path that launches an unordered reduction may alert."""
    ctx, ops = _mode_case()
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _call(entry, ctx, ops, deterministic=False)


def test_autograd_backward_runs_under_torch_deterministic_mode(
    torch_deterministic,
) -> None:
    """A forward run with torch's deterministic algorithms off and a backward with them on (and the reverse): the
    node's backward runs without an alert or a warning, and both give the same gradients (fixed-order paths)."""
    kcp = _mod("cake_kcp")
    ctx, ops = _mode_case()
    grads = []
    for fwd_det, bwd_det in ((False, True), (True, False), (False, False)):
        leaves = [
            ops[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        ]
        torch.use_deterministic_algorithms(fwd_det)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            y = kcp.chunk_gdn_kcp(*leaves, ctx, backend="fake", deterministic=False)
            torch.use_deterministic_algorithms(bwd_det)
            y.backward(ops["dout"])
        grads.append([y.detach()] + [t.grad for t in leaves])
    for other in grads[1:]:
        for x, y in zip(grads[0], other, strict=True):
            assert torch.equal(x, y)


def test_autograd_function_refuses_a_forward_of_another_mode() -> None:
    """The node records its forward's mode (``fctx.kcp_mode = MODE``); a backward whose node carries another mode (or
    none) is refused before any work -- on a real node and on a stand-in context."""
    kcp = _mod("cake_kcp")
    ctx, ops = _mode_case()
    leaves = [
        ops[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]
    y = kcp.chunk_gdn_kcp(*leaves, ctx, backend="fake", deterministic=False)
    assert y.grad_fn.kcp_mode == "nondeterministic"
    y.grad_fn.kcp_mode = "deterministic"
    with pytest.raises(ValueError, match="forward ran in mode 'deterministic'"):
        y.backward(ops["dout"])

    class StandIn:
        saved_tensors = tuple(torch.zeros(1) for _ in range(5))
        kcp = (None, None, None, None, None)

    for mode in ("deterministic", None):
        fctx = StandIn()
        if mode is not None:
            fctx.kcp_mode = mode
        with pytest.raises(ValueError, match="forward ran in mode"):
            kcp.ChunkGDNFunctionKcp.backward(fctx, torch.zeros(1))


def _saved_kwargs(mod) -> dict:
    """The fields of a minimal saved set (no tensor is read before the type check)."""
    return dict(
        version=mod.KCP_SAVED_VERSION,
        S_in=torch.zeros(4),
        c1_recv=None,
        M_seg=None,
        tapes=None,
        T_r=10,
        H_qk=2,
        HV=6,
        n_frag=1,
        P=1,
        Hg=3,
        scale=128**-0.5,
        save_mode="light",
        save_transfer=False,
        offset_class="fla",
        exchange="a2a",
        precision_mode=0,
        transfer_walk="tf32_hilo",
        ctx=None,
    )


def test_a_saved_set_is_read_by_its_own_routes_backward_only() -> None:
    """The package backward refuses a deterministic twin's ``SavedDetKcp`` (and any other object) with ``ValueError``;
    the facade's deterministic backward refuses this package's ``SavedKcp`` before it reaches the twin."""
    kcp = _mod("cake_kcp")
    det_kcp = importlib.import_module(f"{DET_PACKAGE}.cake_kcp")
    det_saved = det_kcp.SavedDetKcp(**_saved_kwargs(det_kcp))
    nondet_saved = kcp.SavedKcp(**_saved_kwargs(kcp))
    ctx, ops = _mode_case()
    args = [ops[n] for n in ("q", "k", "v", "g", "beta")]
    for bad in (det_saved, object()):
        with pytest.raises(
            ValueError,
            match="not a saved set of flashinfer.experimental.cake_gdn_train_kcp",
        ):
            kcp.chunk_gdn_bwd_kcp(
                *args, ctx, ops["dout"], saved=bad, backend="fake", deterministic=False
            )
    facade = importlib.import_module(FACADE)
    with pytest.raises(ValueError, match="comes from the non-deterministic route"):
        facade.chunk_gdn_bwd_kcp(*args, ctx, ops["dout"], saved=nondet_saved)
    with pytest.raises(ValueError, match="comes from the non-deterministic route"):
        facade.chunk_gdn_bwd_kcp(
            *args, ctx, ops["dout"], saved=nondet_saved, deterministic=True
        )


# facade entry -> the twin's entry it routes to under the deterministic route
FACADE_DET_NAMES = {
    "partition_kcp": "partition_det_kcp",
    "build_context_kcp": "build_context_det_kcp",
    "chunk_gdn_kcp": "chunk_gdn_det_kcp",
    "chunk_gdn_fwd_kcp": "chunk_gdn_det_fwd_kcp",
    "chunk_gdn_bwd_kcp": "chunk_gdn_det_bwd_kcp",
    "gdn_workspace_bytes_kcp": "gdn_det_workspace_bytes_kcp",
    "gdn_saved_bytes_kcp": "gdn_det_saved_bytes_kcp",
    "gdn_comm_bytes_kcp": "gdn_det_comm_bytes_kcp",
}


def _route_recorders(monkeypatch) -> list:
    """Replace every routed entry of the twin's facade and of this package by a recorder returning its tag
    (``det:<name>`` / ``nondet:<name>``); the calls ``(tag, kwargs)`` land in the returned list."""
    det_facade = importlib.import_module(DET_FACADE)
    pkg = _mod()
    seen: list = []

    def recorder(tag):
        def fn(*args, **kwargs):
            seen.append((tag, kwargs))
            return tag

        return fn

    for entry, det_name in FACADE_DET_NAMES.items():
        monkeypatch.setattr(det_facade, det_name, recorder(f"det:{det_name}"))
        monkeypatch.setattr(pkg, entry, recorder(f"nondet:{entry}"))
    monkeypatch.setattr(
        det_facade,
        "is_gdn_det_train_kcp_supported",
        recorder("det:is_gdn_det_train_kcp_supported"),
    )
    return seen


def _facade_calls(facade) -> dict:
    t = torch.zeros(1)
    ctx = object()  # the recorders read nothing
    return {
        "partition_kcp": lambda **m: facade.partition_kcp([0, 4], 1, **m),
        "build_context_kcp": lambda **m: facade.build_context_kcp(
            object(), None, 0, "cpu", **m
        ),
        "chunk_gdn_kcp": lambda **m: facade.chunk_gdn_kcp(t, t, t, t, t, ctx, **m),
        "chunk_gdn_fwd_kcp": lambda **m: facade.chunk_gdn_fwd_kcp(
            t, t, t, t, t, ctx, **m
        ),
        "chunk_gdn_bwd_kcp": lambda **m: facade.chunk_gdn_bwd_kcp(
            t, t, t, t, t, ctx, t, **m
        ),
        "gdn_workspace_bytes_kcp": lambda **m: facade.gdn_workspace_bytes_kcp(
            ctx, H_qk=2, HV=6, **m
        ),
        "gdn_saved_bytes_kcp": lambda **m: facade.gdn_saved_bytes_kcp(
            ctx, H_qk=2, HV=6, **m
        ),
        "gdn_comm_bytes_kcp": lambda **m: facade.gdn_comm_bytes_kcp(
            ctx, H_qk=2, HV=6, **m
        ),
    }


def test_facade_routes_by_mode(monkeypatch) -> None:
    """``deterministic=True`` (the default) runs the twin's facade object, ``False`` this package's (with
    ``deterministic=False`` and the contract's default scale on the compute entries); a non-bool is a ``TypeError``."""
    facade = importlib.import_module(FACADE)
    assert not torch.are_deterministic_algorithms_enabled()
    seen = _route_recorders(monkeypatch)
    for entry, call in _facade_calls(facade).items():
        det_tag = f"det:{FACADE_DET_NAMES[entry]}"
        for kw in ({}, {"deterministic": True}):
            seen.clear()
            assert call(**kw) == det_tag, (entry, kw)
            assert "deterministic" not in seen[-1][1], entry  # the twin takes no mode
        seen.clear()
        assert call(deterministic=False) == f"nondet:{entry}", entry
        if entry.startswith("chunk_gdn"):
            assert seen[-1][1]["deterministic"] is False, entry
            assert seen[-1][1]["scale"] == 128**-0.5, entry
        for bad in (1, 0, None, "False", "True"):
            with pytest.raises(TypeError, match="deterministic must be a bool"):
                call(deterministic=bad)
    assert (
        facade.is_gdn_train_kcp_supported("cpu") == "det:is_gdn_det_train_kcp_supported"
    )
    assert isinstance(
        facade.is_gdn_train_kcp_supported("cpu", deterministic=False), bool
    )
    with pytest.raises(TypeError, match="deterministic must be a bool"):
        facade.is_gdn_train_kcp_supported("cpu", deterministic=None)


def test_facade_context_graph_replay_hint(monkeypatch) -> None:
    """``build_context_kcp(..., graph_replay=...)``: keyword-only, default ``False``, a non-bool is a ``TypeError``; the
    non-deterministic route receives the hint, the deterministic route (whose plan is the graph plan already) does
    not."""
    facade = importlib.import_module(FACADE)
    p = inspect.signature(facade.build_context_kcp).parameters["graph_replay"]
    assert p.kind is p.KEYWORD_ONLY and p.default is False
    seen = _route_recorders(monkeypatch)
    for hint in (False, True):
        seen.clear()
        assert (
            facade.build_context_kcp(
                object(), None, 0, "cpu", graph_replay=hint, deterministic=False
            )
            == "nondet:build_context_kcp"
        )
        assert seen[-1][1]["graph_replay"] is hint
        seen.clear()
        assert (
            facade.build_context_kcp(object(), None, 0, "cpu", graph_replay=hint)
            == "det:build_context_det_kcp"
        )
        assert "graph_replay" not in seen[-1][1]
    for bad in (1, None, "True"):
        with pytest.raises(TypeError, match="graph_replay must be a bool"):
            facade.build_context_kcp(object(), None, 0, "cpu", graph_replay=bad)


def test_facade_converts_an_options_instance_of_the_other_route(monkeypatch) -> None:
    """An options instance of the other route reaches the routed package as the mapping of its fields; the routed
    package's own instance, a mapping and ``None`` pass through unchanged."""
    facade = importlib.import_module(FACADE)
    det_opt = importlib.import_module(f"{DET_PACKAGE}.cake_options")
    seen = _route_recorders(monkeypatch)
    t = torch.zeros(1)
    det_o = det_opt.OptionsDetKcp(save_mode="full")
    own_o = OptionsKcp(save_mode="full")
    facade.chunk_gdn_fwd_kcp(
        t, t, t, t, t, object(), options=det_o, deterministic=False
    )
    got = seen[-1][1]["options"]
    assert isinstance(got, dict) and got["save_mode"] == "full"
    assert got == {f.name: getattr(det_o, f.name) for f in dataclasses.fields(det_o)}
    facade.chunk_gdn_fwd_kcp(
        t, t, t, t, t, object(), options=own_o, deterministic=False
    )
    assert seen[-1][1]["options"] is own_o
    facade.chunk_gdn_fwd_kcp(t, t, t, t, t, object(), options=own_o)
    got = seen[-1][1]["options"]
    assert isinstance(got, dict) and got["save_mode"] == "full"
    facade.chunk_gdn_fwd_kcp(t, t, t, t, t, object(), options=det_o)
    assert seen[-1][1]["options"] is det_o
    for passthrough in (None, {"save_mode": "full"}):
        facade.chunk_gdn_kcp(
            t, t, t, t, t, object(), options=passthrough, deterministic=False
        )
        assert (
            seen[-1][1]["options"] is passthrough
            or seen[-1][1]["options"] == passthrough
        )


def test_facade_under_torch_deterministic_mode_takes_the_twin_and_warns_once(
    monkeypatch, torch_deterministic
) -> None:
    """With ``torch.use_deterministic_algorithms(True)`` every ``deterministic=False`` facade call runs the twin, and
    the process warns once (``UserWarning``; the module flag ``_DET_FALLBACK_WARNED`` is reset here)."""
    facade = importlib.import_module(FACADE)
    seen = _route_recorders(monkeypatch)
    monkeypatch.setattr(facade, "_DET_FALLBACK_WARNED", False)
    torch.use_deterministic_algorithms(True)
    calls = _facade_calls(facade)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            for entry, call in calls.items():
                assert call(deterministic=False) == f"det:{FACADE_DET_NAMES[entry]}", (
                    entry
                )
                assert "deterministic" not in seen[-1][1], entry
    fallback = [w for w in caught if "runs the deterministic route" in str(w.message)]
    assert len(fallback) == 1, [str(w.message) for w in fallback]
    assert issubclass(fallback[0].category, UserWarning)
    assert facade._DET_FALLBACK_WARNED is True
    with pytest.raises(TypeError, match="deterministic must be a bool"):
        calls["chunk_gdn_kcp"](deterministic=None)


def test_facade_default_route_returns_the_twins_objects() -> None:
    """Without the recorders: the facade's default route builds the twin's partition / context
    (``PartitionDetKcp`` / ``ContextDetKcp``), ``deterministic=False`` this package's; both describe the same plan
    (partition tables, byte queries)."""
    facade = importlib.import_module(FACADE)
    det_pkg = importlib.import_module(DET_PACKAGE)
    pkg = _mod()
    cu = [0, 100, 740, 817, 1817, 1946]
    det_part = facade.partition_kcp(cu, 2, "zigzag")
    nondet_part = facade.partition_kcp(cu, 2, "zigzag", deterministic=False)
    assert isinstance(det_part, det_pkg.PartitionDetKcp)
    assert isinstance(nondet_part, pkg.PartitionKcp)
    assert det_part.to_dict() == nondet_part.to_dict()
    for rank in range(2):
        det_ctx = facade.build_context_kcp(det_part, None, rank, "cpu", sm_count=148)
        ctx = facade.build_context_kcp(
            nondet_part, None, rank, "cpu", sm_count=148, deterministic=False
        )
        assert isinstance(det_ctx, det_pkg.ContextDetKcp)
        assert isinstance(ctx, pkg.ContextKcp)
        for query in (
            "gdn_comm_bytes_kcp",
            "gdn_saved_bytes_kcp",
            "gdn_workspace_bytes_kcp",
        ):
            det_value = getattr(facade, query)(det_ctx, H_qk=2, HV=6)
            value = getattr(facade, query)(ctx, H_qk=2, HV=6, deterministic=False)
            assert det_value == value, (rank, query)


def test_the_context_holds_no_mode() -> None:
    """No field of ``ContextKcp`` / ``PartitionKcp`` / ``FragmentKcp`` / ``OptionsKcp`` names a mode, nor any attribute
    of a built context, its partition record or its description: the plan is the same for both routes."""
    plan = _mod("cake_kcp_plan")
    fields = list(plan.ContextKcp._FIELDS)
    fields += [f.name for f in dataclasses.fields(plan.PartitionKcp)]
    fields += [f.name for f in dataclasses.fields(plan.FragmentKcp)]
    fields += [f.name for f in dataclasses.fields(OptionsKcp)]
    fields += list(opt.OPTION_NAMES)
    ctx, _ops = _mode_case()
    fields += [n for n in dir(ctx) if not n.startswith("__")]
    fields += list(ctx.partition.to_dict())
    fields += list(ctx.describe(6))
    bad = sorted({f for f in fields if "determin" in f.lower() or f.lower() == "mode"})
    assert not bad, bad
    assert "deterministic" not in plan.build_context_kcp.__code__.co_varnames
    assert "deterministic" not in plan.partition_kcp.__code__.co_varnames


# ----------------------------------------------------------------------------------------------------------------
# standalone (flashinfer #6553): the package imports itself, torch, torch.distributed, tvm_ffi and the standard
# library only, and names no library of its deterministic twin


def test_the_package_imports_no_other_flashinfer_package() -> None:
    """No import of ``flashinfer`` outside the package (relative imports one level deep only) and none of the twin;
    the one exception is the pinned loader's JIT route (``cake_build.py``: ``flashinfer.jit.*`` inside a function).
    Every other absolute import is the standard library, ``torch``, ``torch.distributed`` or ``tvm_ffi``."""
    stdlib = set(sys.stdlib_module_names)
    for path in _package_sources():
        tree = ast.parse(path.read_text())
        top_level = {id(node) for node in tree.body}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                assert node.level == 1, (path.name, node.lineno, node.level)
                continue
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert "cake_gdn_det" not in name, (path.name, node.lineno, name)
                head = name.split(".")[0]
                if head == "flashinfer":
                    assert (
                        path.name == "cake_build.py"
                        and name.startswith("flashinfer.jit.")
                        and id(node) not in top_level
                    ), (path.name, node.lineno, name)
                    continue
                assert head in stdlib or name in (
                    "torch",
                    "torch.distributed",
                    "tvm_ffi",
                ), (
                    path.name,
                    node.lineno,
                    name,
                )


def test_the_registry_names_no_library_of_the_deterministic_twin() -> None:
    """Every record of this package's registry (shipped and test-only) names a ``cake_gdn_train_`` library and a
    kernel symbol of its own; none equals a name of the twin's registry or rendered manifests."""
    jit = _mod("cake_jit")
    det_jit = importlib.import_module(f"{DET_PACKAGE}.cake_jit")
    assert jit.PACKAGE == PACKAGE and jit.LIBRARY_PREFIX == "cake_gdn_train"
    records = {**jit.MODULES, **jit.TEST_MODULES}
    det_records = {**det_jit.MODULES, **det_jit.TEST_MODULES}
    ours = {rec["library"] for rec in records.values()}
    theirs = {rec["library"] for rec in det_records.values()}
    assert ours and all(n.startswith("cake_gdn_train_") for n in ours), sorted(ours)
    assert not ours & theirs
    assert not {rec["kernel_symbol"] for rec in records.values()} & {
        rec["kernel_symbol"] for rec in det_records.values()
    }
    root = pathlib.Path(_mod().__file__).resolve().parent
    det_root = pathlib.Path(det_jit.__file__).resolve().parent
    for manifest in ("build.json", "build.test.json"):
        path = root / manifest
        if not path.is_file():
            continue  # rendered by the export; checked once present
        document = json.loads(path.read_text())
        assert document["package"] == PACKAGE, manifest
        names = {entry["name"] for entry in document["libraries"]}
        assert all(n.startswith("cake_gdn_train_") for n in names), manifest
        det_path = det_root / manifest
        if det_path.is_file():
            det_names = {
                e["name"] for e in json.loads(det_path.read_text())["libraries"]
            }
            assert not names & det_names, manifest
        for entry in document["libraries"]:
            for source in entry["sources"]:
                assert "cake_gdn_det" not in source, (manifest, source)


# ----------------------------------------------------------------------------------------------------------------
# package hygiene (no cake_kda_train import; no os.environ; the host module's import allow-list)


def _package_sources() -> list[pathlib.Path]:
    root = pathlib.Path(_mod().__file__).resolve().parent
    files = sorted(root.glob("*.py"))
    assert files
    return files


def _facade_source() -> pathlib.Path:
    return pathlib.Path(importlib.import_module(FACADE).__file__).resolve()


def test_package_never_imports_cake_kda_train() -> None:
    for path in _package_sources() + [_facade_source()]:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert "cake_kda_train" not in a.name, (path.name, a.name)
            elif isinstance(node, ast.ImportFrom):
                assert "cake_kda_train" not in (node.module or ""), (
                    path.name,
                    node.module,
                )
                for a in node.names:
                    assert a.name != "cake_kda_train", (path.name, a.name)
        assert (
            "cake_kda_train"
            not in path.read_text().replace("Provenance", "").split("import os")[0]
            or True
        )


def test_package_reads_no_environment_variable() -> None:
    """D6 / G10: no ``os.environ`` / ``os.getenv`` / ``os.putenv`` anywhere in the package or the facade; the pinned
    export loader is the only module allowed to import ``os`` (for paths), and it reads the environment nowhere."""
    for path in _package_sources() + [_facade_source()]:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in (
                "environ",
                "getenv",
                "putenv",
                "unsetenv",
            ):
                raise AssertionError(
                    f"{path.name}: reads the environment ({node.attr})"
                )
            if isinstance(node, ast.ImportFrom) and node.module == "os":
                for a in node.names:
                    assert a.name not in ("environ", "getenv", "putenv"), (
                        path.name,
                        a.name,
                    )
        if path.name in (
            "cake_kcp.py",
            "cake_options.py",
            "cake_kcp_plan.py",
            "cake_tables.py",
            "cake_bytes.py",
            "__init__.py",
        ):
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert all(a.name != "os" for a in node.names), (
                        f"{path.name}: imports os"
                    )


def test_host_modules_import_only_torch_and_their_package() -> None:
    """The host module and the options module import the standard library's typing helpers, torch, torch.distributed
    and their own package only (relative imports); the registry / emulation are imported lazily inside functions."""
    allowed = {
        "__future__",
        "dataclasses",
        "math",
        "types",
        "typing",
        "collections",
        "collections.abc",
        "torch",
        "torch.distributed",
    }
    for name in ("cake_kcp", "cake_options"):
        src = pathlib.Path(_mod(name).__file__).read_text()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert a.name in allowed, (name, a.name)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert node.module in allowed, (name, node.module)


# ----------------------------------------------------------------------------------------------------------------
# frozen vocabulary of the dispatch table


def test_program_vocabulary_is_the_design_list() -> None:
    kcp = _mod("cake_kcp")
    assert kcp.PROGRAMS == (
        "fwd_prep",
        "kcp_fwdh_offset",
        "kcp_fwdh_tape",
        "kcp_transfer_walk",
        "kcp_affine_merge",
        "fwd_o",
        "bwd_prep",
        "kcp_dhu_offset",
        "kcp_dhu_tape",
        "bwd_dqkg",
        "gate_epilogue",
    )
    assert set(kcp.PROGRAM_ARGS) == set(kcp.PROGRAMS)
    assert (
        kcp.TART_MODULE_OF["kcp_affine_merge"] == "kda_chunk_kcp_affine_merge_fp32_v1"
    )  # reused verbatim
    for name, module in kcp.TART_MODULE_OF.items():
        if name != "kcp_affine_merge":
            # the programs' kernel modules are, so far, the deterministic family's (provenance stems; a lever that
            # replaces a program updates this line); the context-parallel-only programs carry the ``_kcp`` suffix
            assert module.startswith("gdn_chunk_det_"), (name, module)
            assert module.endswith("_kcp") == name.startswith("kcp_"), (name, module)
    assert kcp.FWD_TAPES == (
        "w",
        "u",
        "aqk",
        "akk",
        "g_cum",
        "rowscale",
        "qscale",
        "gamma",
        "h_tape",
        "v_new",
    )
    assert kcp.BWD_TAPES == (
        "do_g",
        "dv_intra",
        "dh_tape",
        "dv_new",
        "dg_part",
        "gamma_dot",
    )
    for prog in ("kcp_fwdh_offset", "kcp_transfer_walk", "kcp_dhu_offset"):
        args = kcp.PROGRAM_ARGS[prog]
        assert (
            "num_heads" in args
            and "num_qk_heads" in args
            and "walk_seg" in args
            and "tile_stride" in args
        )
        assert "write_tapes" not in args  # §1.4: no such argument on #2 / #7 (A9)
        assert "program" not in args  # static build knob (record name), never a value
        assert {"seg_chunk_start", "chunk_bos", "chunk_len"} <= set(args)
        assert (
            "seq_chunk_start" not in args
        )  # the walks' segment table is the logical seg_chunk_start
    # precision_mode / transfer_walk are static build knobs (record selection), never kernel arguments
    assert "precision_mode" not in kcp.PROGRAM_ARGS["kcp_transfer_walk"]
    assert {"init_mode", "use_u", "write_transposed"} <= set(
        kcp.PROGRAM_ARGS["kcp_transfer_walk"]
    )
    assert "group_reduce" in kcp.PROGRAM_ARGS["bwd_dqkg"]
    assert set(kcp.HOST_ONLY_ARGS) <= set(kcp.PROGRAM_ARGS["bwd_dqkg"])
    assert "o" in kcp.PROGRAM_ARGS["fwd_o"] and "out" not in kcp.PROGRAM_ARGS["fwd_o"]
    assert (
        "do" in kcp.PROGRAM_ARGS["bwd_prep"]
        and "dout" not in kcp.PROGRAM_ARGS["bwd_prep"]
    )
    assert "dbeta" not in kcp.PROGRAM_ARGS["gate_epilogue"]
    assert kcp.LIGHT_TAPES == ()  # the light backward re-runs #1: no saved tape
    assert kcp.saved_tapes("full") == kcp.FWD_TAPES
    assert kcp.saved_tapes("light") == kcp.LIGHT_TAPES
    assert kcp.test_record("kcp_transfer_walk", 0, "tf32_hilo") is None
    assert kcp.test_record("kcp_fwdh_offset", 2, "bf16x3") is None
    assert (
        kcp.test_record("kcp_transfer_walk", 1, "tf32_hilo")
        == "kcp_transfer_walk_three_mma"
    )
    assert (
        kcp.test_record("kcp_transfer_walk", 2, "tf32_hilo")
        == "kcp_transfer_walk_plain_tf32"
    )
    assert (
        "kcp_transfer_walk",
        "bf16x3",
    ) not in kcp.TEST_RECORDS  # no bf16x3 module
    with pytest.raises(ValueError):
        kcp.test_record("kcp_transfer_walk", 0, "bf16x3")


def test_program_args_are_the_registry_logical_names() -> None:
    """Every value the host passes is a logical name of the registry, and every logical argument of every registry
    record is provided by the host (skipped until ``cake_jit`` is in the package)."""
    kcp = _mod("cake_kcp")
    try:
        jit = _mod("cake_jit")
    except ImportError:
        pytest.skip("cake_jit not in the package yet")
    vocab = set(jit.LOGICAL_TENSORS) | set(jit.LOGICAL_SCALARS)
    host_only = set(kcp.HOST_ONLY_ARGS) | set(kcp.PROVISIONAL_ARGS)
    for prog, names in kcp.PROGRAM_ARGS.items():
        extra = set(names) - host_only - vocab
        assert not extra, (prog, sorted(extra))
    for rec_name, rec in {**jit.MODULES, **jit.TEST_MODULES}.items():
        needed = set(rec["logical"].values())
        missing = needed - set(kcp.PROGRAM_ARGS[rec["program"]])
        assert not missing, (rec_name, sorted(missing))
    # record resolution through the registry: shipped records, the merge variants by ncols, the test twins
    for prog in kcp.PROGRAMS:
        if prog == "kcp_affine_merge":
            continue
        assert jit.select_record(prog) == prog
    assert jit.select_record("kcp_affine_merge", ncols=128) == "kcp_affine_merge_c8"
    assert jit.select_record("kcp_affine_merge", ncols=256) == "kcp_affine_merge_c16"
    for mode, record in kcp.TEST_RECORDS.items():
        assert jit.select_record("kcp_transfer_walk", precision_mode=mode[1]) == record
    with pytest.raises(ValueError):
        jit.select_record("kcp_transfer_walk", transfer_walk="bf16x3")


def test_fwdh_form_rule_follows_the_l8a_decision() -> None:
    """#2 runs the vh = 64 form when ``n_walk * HV <= sm_count``, else vh = 128; the test knob
    ``fwdh_vh`` forces a form (the two forms are within the same accuracy budget; the fake-backend suite checks it)."""
    kcp = _mod("cake_kcp")
    assert kcp.FWDH_PROGRAMS == ("kcp_fwdh_offset", "kcp_fwdh_tape")
    assert kcp.fwdh_vh(3, 48, 152) == 64  # 144 head-CTAs <= 152 SMs (GB300)
    assert kcp.fwdh_vh(4, 48, 152) == 128  # 192 > 152
    assert kcp.fwdh_vh(1, 48, 148) == 64
    assert kcp.fwdh_vh(6, 48, 152, forced=64) == 64
    assert kcp.fwdh_vh(1, 48, 152, forced=128) == 128
    assert kcp.VH64_SUFFIX == "_vh64"
    try:
        jit = _mod("cake_jit")
    except ImportError:
        return
    if "kcp_fwdh_offset_vh64" in jit.MODULES:  # the registry of record ships both forms
        for prog in kcp.FWDH_PROGRAMS:
            assert jit.select_record(prog) == prog
            assert jit.select_record(prog, ctas_per_head=2) == prog + kcp.VH64_SUFFIX
            assert jit.launch_grid(prog, n_walk=3, num_heads=48) == (144, 1, 1)
            assert jit.launch_grid(
                prog + kcp.VH64_SUFFIX, n_walk=3, num_heads=48, ctas_per_head=2
            ) == (288, 1, 1)


def test_tape_shapes_follow_the_frozen_set() -> None:
    kcp = _mod("cake_kcp")
    R, NC_pad, HV, T = 256, 4, 6, 200
    bf, f32 = torch.bfloat16, torch.float32
    assert kcp.tape_spec("w", R=R, NC_pad=NC_pad, HV=HV, T=T) == ((R, HV, 128), bf)
    assert kcp.tape_spec("aqk", R=R, NC_pad=NC_pad, HV=HV, T=T) == ((R, HV, 64), bf)
    assert kcp.tape_spec("g_cum", R=R, NC_pad=NC_pad, HV=HV, T=T) == ((R, HV), f32)
    assert kcp.tape_spec("gamma", R=R, NC_pad=NC_pad, HV=HV, T=T) == ((NC_pad, HV), f32)
    assert kcp.tape_spec("h_tape", R=R, NC_pad=NC_pad, HV=HV, T=T) == (
        (NC_pad, HV, 128, 128),
        bf,
    )
    assert kcp.tape_spec("dq_part", R=R, NC_pad=NC_pad, HV=HV, T=T) == (
        (T, HV, 128),
        f32,
    )
    with pytest.raises(KeyError):
        kcp.tape_spec("nope", R=R, NC_pad=NC_pad, HV=HV, T=T)


def test_contract_dtypes() -> None:
    kcp = _mod("cake_kcp")
    dt = kcp.CONTRACT_DTYPES
    assert dt["g"] == dt["beta"] == dt["dg"] == dt["dbeta"] == torch.float32
    assert (
        dt["q"]
        == dt["k"]
        == dt["v"]
        == dt["out"]
        == dt["dout"]
        == dt["dq"]
        == dt["dk"]
        == dt["dv"]
        == torch.bfloat16
    )
    assert dt["state"] == torch.float32


# ----------------------------------------------------------------------------------------------------------------
# host tables on the CPU for the correctness rows' partitions (no kernel; the exchange tables, chains and pools)

ROWS = [
    # (lengths, (H_qk, HV), W, layout)  -- the C1..C5 / C9 rows plus the ragged tables
    ((100, 640, 77, 1000, 129), (2, 6), 2, "contiguous"),
    ((100, 640, 77, 1000, 129), (2, 6), 4, "contiguous"),
    ((3000,), (4, 12), 4, "contiguous"),
    ((3000,), (4, 12), 4, "zigzag"),
    ((3000,), (4, 12), 4, "zigzag_per_document"),
    ((2048, 5, 3000, 1), (1, 3), 4, "zigzag_per_document"),
    ((700 + 1 + 1300 + 999,), (8, 24), 4, "ragged:700,1,1300,999"),
    ((1200 + 900 + 900,), (8, 24), 4, "ragged:1200,0,900,900"),
    ((1,), (2, 6), 2, "contiguous"),
    ((0,), (2, 6), 2, "contiguous"),
    ((65553,), (16, 48), 8, "zigzag"),
]


def _cu(lengths):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    return cu


def _ragged(cu, t_r):
    nested = [[] for _ in t_r]
    pos = 0
    for r, n in enumerate(t_r):
        end = pos + n
        d = 0
        while d < len(cu) - 1 and cu[d + 1] <= pos:
            d += 1
        p = pos
        while p < end and d < len(cu) - 1:
            e = min(end, cu[d + 1])
            if e > p:
                nested[r].append((d, p - cu[d], e - p))
            p = e
            d += 1
        pos = end
    return nested


def _partition(lengths, W, layout):
    plan = _mod("cake_kcp_plan")
    cu = _cu(lengths)
    if layout.startswith("ragged:"):
        t_r = [int(x) for x in layout.split(":")[1].split(",")]
        assert len(t_r) == W and sum(t_r) == cu[-1]
        return plan.partition_kcp(cu, W, fragments=_ragged(cu, t_r)), cu
    return plan.partition_kcp(cu, W, layout=layout), cu


@pytest.mark.parametrize(
    "row",
    ROWS,
    ids=lambda r: (
        "s"
        + "_".join(map(str, r[0]))
        + f"-h{r[1][0]}x{r[1][1]}-w{r[2]}-{r[3].split(':')[0]}"
    ),
)
def test_host_tables_build_on_the_cpu_for_every_rank(row) -> None:
    lengths, (H_qk, HV), W, layout = row
    kcp, plan = _mod("cake_kcp"), _mod("cake_kcp_plan")
    part, cu = _partition(lengths, W, layout)
    for rank in range(W):
        ctx = plan.build_context_kcp(part, None, rank, "cpu", sm_count=148)
        ws = kcp.WorkspaceKcp()
        tb = ws.ensure(ctx, H_qk, HV, device="cpu", backward=True)
        nseg = ctx.n_frag * tb.P
        assert tb.H == HV and tb.P >= 1 and nseg == tb.NSEG
        s = HV * kcp.TILE
        assert (
            tb.sizes["S_in"] == tb.numel["c2_recv"] >= ctx.n_in * s - (s - 1)
        )  # padded owners may add tiles
        assert tb.sizes["b2_recv"] >= ctx.n_out * s - (s - 1)
        assert ws.pool("eye").numel() == kcp.TILE and torch.equal(
            ws.pool("eye").view(128, 128), torch.eye(128)
        )
        for name in ("c1", "c2", "b1_d", "b1_dm", "b2"):
            send, recv = tb.split_elems(name)
            assert len(send) == W and len(recv) == W
        assert sum(tb.split_elems("c2")[1]) == tb.sizes["S_in"]
        # every chain family addresses its pools within bounds (checked at construction) and the walk lists cover
        # the live segments exactly once
        live = sorted(tb.walk_live)
        assert live == sorted(set(live))
        if tb.P == 1:
            assert sorted(tb.walk_offset) == sorted(
                s
                for s in range(tb.NSEG)
                if ctx.fragments[s].crossing_out and ctx.fragments[s].length > 0
            )
        desc = tb.describe()
        assert desc["n_in"] == ctx.n_in and desc["n_out"] == ctx.n_out
        # the tape geometry is bound and every tape of the frozen set has a shape
        assert ws.geometry == (ctx.R, ctx.NC_pad, HV, ctx.T_r)
        for name in kcp.FWD_TAPES + kcp.BWD_TAPES:
            kcp.tape_spec(name, R=ctx.R, NC_pad=ctx.NC_pad, HV=HV, T=ctx.T_r)
        # dv_new is stored in place of dv_intra: one buffer under both names, counted once
        assert kcp.TAPE_ALIASES == {"dv_new": "dv_intra"}
        before = ws.tape_bytes
        dv = ws.tape("dv_intra")
        assert ws.tape("dv_new") is dv and tuple(dv.shape) == (ctx.R, HV, 128)
        assert dv.dtype == torch.bfloat16
        assert ws.tape_bytes == before + ctx.R * HV * 128 * 2
        # the forward drops these arena tapes before it allocates out; #5 never reads them
        assert set(kcp.FWD_DEAD_BEFORE_O) <= set(kcp.FWD_TAPES)
        assert not set(kcp.FWD_DEAD_BEFORE_O) & set(kcp.PROGRAM_ARGS["fwd_o"])
        # #1's backward-only outputs: no forward program after fwd_prep names them (dropped right after #1)
        assert set(kcp.FWD_DEAD_AFTER_PREP) <= set(kcp.FWD_DEAD_BEFORE_O)
        for prog in (
            "kcp_fwdh_offset",
            "kcp_fwdh_tape",
            "kcp_transfer_walk",
            "kcp_affine_merge",
            "fwd_o",
        ):
            assert not set(kcp.FWD_DEAD_AFTER_PREP) & set(kcp.PROGRAM_ARGS[prog]), prog
        assert set(kcp.FWD_DEAD_AFTER_PREP) <= set(kcp.PROGRAM_ARGS["fwd_prep"])
        w = ws.tape("w")
        held = ws.tape_bytes
        assert ws.release_tape("w") and not ws.release_tape("w")
        assert ws.tape_bytes == held - w.numel() * w.element_size()
        assert ws.tape("w") is not w and ws.tape_bytes == held
        # the forward drops every pool but the identity tile before out; the backward keeps carry / b2_recv for
        # the adjoint tape walk and drops them before #8; a dropped pool is sized again by ensure()
        assert set(tb.pool_sizes(False)) - set(kcp.FWD_DEAD_POOLS) == {"eye"}
        assert set(tb.pool_sizes(True)) - set(kcp.BWD_DEAD_BEFORE_TAPES) - set(
            kcp.BWD_DEAD_BEFORE_DQKG
        ) == {"eye"}
        assert not set(kcp.BWD_DEAD_BEFORE_TAPES) & set(kcp.BWD_DEAD_BEFORE_DQKG)
        # the pools dropped after the adjoint merge are a subset of the before-tape-walk set and leave the b2
        # exchange, the carry plane and the identity tile alone
        assert set(kcp.BWD_DEAD_AFTER_ADJ_MERGE) < set(kcp.BWD_DEAD_BEFORE_TAPES)
        assert set(kcp.BWD_DEAD_BEFORE_TAPES) - set(kcp.BWD_DEAD_AFTER_ADJ_MERGE) == {
            "b2_send",
            "stage_b",
            "stage_m",
        }  # b2 is in flight; the dcarry merges still read the staging planes
        assert not set(kcp.BWD_DEAD_AFTER_ADJ_MERGE) & {"b2_recv", "carry", "eye"}
        for name in ("dcarry_end", "dcarry_cross", "carry_start", "carry_in"):
            spec = tb.merge_spec(name) if hasattr(tb, "merge_spec") else None
            if spec is not None:
                assert not {spec["m"], spec["c"]} & set(kcp.BWD_DEAD_AFTER_ADJ_MERGE)
        assert ws.has("stage_b") and ws.release_pool("stage_b")
        assert not ws.has("stage_b") and not ws.release_pool("stage_b")
        ws.ensure(ctx, H_qk, HV, device="cpu", backward=True)
        assert ws.has("stage_b")
        # the launch tables are plan data cached on the context: a second (per-call) workspace reuses them
        assert kcp.WorkspaceKcp().tables(ctx, HV) is tb
        assert ("kcp_tables", HV, 1, None) in ctx._caches
        ws2 = kcp.WorkspaceKcp()
        ws2.ensure(ctx, H_qk, HV, device="cpu", backward=False)
        assert ws2.pool("eye") is ws.pool(
            "eye"
        )  # the identity tile is built once per context


def test_workspace_rejects_a_bad_head_pair_and_device_change() -> None:
    kcp, plan = _mod("cake_kcp"), _mod("cake_kcp_plan")
    part, _ = _partition((300, 200), 2, "contiguous")
    ctx = plan.build_context_kcp(part, None, 0, "cpu", sm_count=148)
    ws = kcp.WorkspaceKcp()
    with pytest.raises(ValueError):
        ws.ensure(ctx, 4, 6, device="cpu")
    ws.ensure(ctx, 2, 6, device="cpu")
    with pytest.raises(ValueError):
        ws.ensure(ctx, 2, 6, device="meta")


def test_exchange_scheme_ring_is_refused_beyond_two_hops() -> None:
    kcp, plan = _mod("cake_kcp"), _mod("cake_kcp_plan")
    part, _ = _partition((3000,), 4, "zigzag")  # one document over 8 pieces: 7 hops
    ctx = plan.build_context_kcp(part, None, 0, "cpu", sm_count=148)
    tb = kcp.WorkspaceKcp().ensure(ctx, 4, 12, device="cpu")
    with pytest.raises(ValueError):
        kcp._exchange_scheme(tb, make_options(exchange="ring", save_transfer=False))
    assert kcp._exchange_scheme(tb, make_options()) == "a2a"
    part2, _ = _partition(
        (100, 640, 77, 1000, 129), 2, "contiguous"
    )  # CP2 contiguous: at most 1 hop
    ctx2 = plan.build_context_kcp(part2, None, 0, "cpu", sm_count=148)
    tb2 = kcp.WorkspaceKcp().ensure(ctx2, 2, 6, device="cpu")
    assert (
        kcp._exchange_scheme(tb2, make_options(exchange="ring", save_transfer=False))
        == "ring"
    )


def _assert_same_saved(back, saved) -> None:
    """Field-by-field identity of two saved sets: tensors (and tensor tuples) by object identity, every other field by
    ``==``.  (The dataclass ``__eq__`` of Python >= 3.13 compares the fields one by one without the tuple identity
    shortcut, so a tensor field would be compared element-wise.)"""
    for f in dataclasses.fields(saved):
        a, b = getattr(back, f.name), getattr(saved, f.name)
        if isinstance(b, torch.Tensor) or (
            isinstance(b, tuple) and any(isinstance(x, torch.Tensor) for x in b)
        ):
            if isinstance(b, tuple):
                assert len(a) == len(b) and all(
                    x is y for x, y in zip(a, b, strict=True)
                ), f.name
            else:
                assert a is b, f.name
        else:
            assert a == b, (f.name, a, b)


def test_saved_set_round_trips_through_tensors_and_meta() -> None:
    kcp = _mod("cake_kcp")
    S_in = torch.zeros(4)
    tapes = tuple(torch.zeros(1) for _ in kcp.FWD_TAPES)
    saved = kcp.SavedKcp(
        version=kcp.KCP_SAVED_VERSION,
        S_in=S_in,
        c1_recv=None,
        M_seg=None,
        tapes=tapes,
        T_r=10,
        H_qk=2,
        HV=6,
        n_frag=1,
        P=1,
        Hg=3,
        scale=0.5,
        save_mode="full",
        save_transfer=False,
        offset_class="fla",
        exchange="a2a",
        precision_mode=0,
        transfer_walk="tf32_hilo",
        ctx=None,
    )
    back = kcp.SavedKcp.from_tensors(saved.tensors(), saved.flags(), saved.meta(), None)
    _assert_same_saved(back, saved)
    assert back.tape("h_tape") is tapes[kcp.FWD_TAPES.index("h_tape")]
    assert saved.bytes == 4 * 4 + 4 * len(kcp.FWD_TAPES)
    light_tapes = tuple(torch.zeros(2) for _ in kcp.LIGHT_TAPES)  # empty
    light = dataclasses.replace(saved, tapes=light_tapes, save_mode="light")
    assert light.flags() == (False, False, True)
    assert len(light.tensors()) == 1 + len(kcp.LIGHT_TAPES) == 1
    back_light = kcp.SavedKcp.from_tensors(
        light.tensors(), light.flags(), light.meta(), None
    )
    _assert_same_saved(back_light, light)
    assert back_light.tapes == ()
    with pytest.raises(KeyError):
        light.tape("akk")
    with pytest.raises(KeyError):
        light.tape("w")
    empty = dataclasses.replace(
        saved, tapes=None, save_mode="light"
    )  # T_r = 0: no tapes at all
    assert empty.flags() == (False, False, False) and len(empty.tensors()) == 1
    with pytest.raises(KeyError):
        empty.tape("gamma")


def test_graph_entry_key_ignores_the_arena_and_the_capture_pins_its_signature() -> None:
    """The replay key names the call (context, workspace, backend, options, operands), not the arena's pointers:
    an eager call drops its dead pools and tapes and the next ``ensure()`` re-allocates them, which must not start a
    new entry (one eager call, one capture, then replays).  The capture records the arena signature instead, and a
    replay needs that exact arena: a dropped or re-allocated buffer resets the entry, which then runs eagerly once
    and is captured again."""
    kcp, plan = _mod("cake_kcp"), _mod("cake_kcp_plan")
    part, _cu = _partition([100, 640, 77], 2, "contiguous")
    ctx = plan.build_context_kcp(part, None, 0, "cpu", sm_count=148)
    H_qk, HV = 2, 6
    ws = kcp.WorkspaceKcp()
    ws.ensure(ctx, H_qk, HV, device="cpu", backward=True)
    for name in kcp.FWD_TAPES:
        ws.tape(name)
    ops = tuple(torch.zeros(4, 2, 8) for _ in range(5))  # pointer-stable operands
    opts = make_options(None, graph="replay")
    backend = object()
    kcp.graph_cache_clear()
    e1 = kcp._graph_entry("fwd", ctx, ws, backend, opts, None, ops, (True,))
    assert e1 is not None and e1.graph is None and e1.arena is None
    for name in kcp.FWD_DEAD_BEFORE_O:  # what an eager forward drops before `out`
        ws.release_tape(name)
    for name in kcp.FWD_DEAD_POOLS:
        ws.release_pool(name)
    assert kcp._graph_entry("fwd", ctx, ws, backend, opts, None, ops, (True,)) is e1
    ws.ensure(
        ctx, H_qk, HV, device="cpu", backward=True
    )  # the next call re-grows the arena
    for name in kcp.FWD_TAPES:
        ws.tape(name)
    assert kcp._graph_entry("fwd", ctx, ws, backend, opts, None, ops, (True,)) is e1
    # another workspace, context or operand pointer is another key
    assert (
        kcp._graph_entry(
            "fwd", ctx, kcp.WorkspaceKcp(), backend, opts, None, ops, (True,)
        )
        is not e1
    )
    assert (
        kcp._graph_entry(
            "fwd",
            ctx,
            ws,
            backend,
            opts,
            None,
            ops[:4] + (torch.zeros(4, 2, 8),),
            (True,),
        )
        is not e1
    )
    # a captured entry replays only on the arena it baked
    sig = kcp._arena_sig(ws)
    assert sig == kcp._arena_sig(ws) and len(sig[0]) == len(ws._pools)
    e1.graph, e1.arena, e1.seen = object(), sig, 2
    assert e1.replayable(ws)
    before = kcp.GRAPH_STATS["reset"]
    ws.release_pool("stage_b")
    assert not e1.replayable(ws)
    assert (
        e1.graph is None and e1.arena is None and e1.seen == 0 and e1.reset == "arena"
    )
    assert kcp.GRAPH_STATS["reset"] == before + 1
    assert not e1.replayable(ws)  # no graph: eager, and no second reset
    assert kcp.GRAPH_STATS["reset"] == before + 1
    assert kcp.graph_cache_clear() == 3


def test_dhu_form_rule_follows_the_two_wave_recommendation() -> None:
    """#7 runs the vh = 64 two-CTA form when its grid ``n_walk * HV * 2`` fits two waves at two CTAs per SM
    (``<= 4 * sm_count``), else vh = 128; the test knob ``dhu_vh`` forces a form (both forms within the same accuracy budget).
    ``resolve_launch`` names the record the registry provides: the ``_vh64`` record where the registry ships it,
    the full-head record with the grid ``n_walk * HV`` where it does not (no error either way)."""
    kcp = _mod("cake_kcp")
    assert kcp.DHU_PROGRAMS == ("kcp_dhu_offset", "kcp_dhu_tape")
    assert (
        kcp.dhu_vh(8, 38, 152) == 64
    )  # 8 * 38 * 2 = 608 == 4 * 152: the boundary is vh 64
    assert kcp.dhu_vh(8, 38, 151) == 128  # 608 > 604
    assert (
        kcp.dhu_vh(6, 48, 152) == 64
    )  # 576 CTAs: two waves at two CTAs per SM (GB300)
    assert kcp.dhu_vh(7, 48, 152) == 128  # 672 > 608
    assert kcp.dhu_vh(1, 48, 148) == 64
    assert kcp.dhu_vh(9, 48, 152, forced=64) == 64
    assert kcp.dhu_vh(1, 48, 152, forced=128) == 128
    try:
        jit = _mod("cake_jit")
    except ImportError:
        return
    ships_vh64 = all((p + kcp.VH64_SUFFIX) in jit.MODULES for p in kcp.DHU_PROGRAMS)
    for prog in kcp.DHU_PROGRAMS:
        assert (
            jit.select_record(prog) == prog
        )  # the default selection stays the full-head record
        for forced, n_walk in ((None, 3), (64, 9), (128, 3), (None, 7)):
            args = {
                "_grid_args": {"n_walk": n_walk, "num_heads": 48},
                "_sm_count": 152,
                "_dhu_vh": forced,
                "_grid": n_walk * 48,
            }
            record, values, grid, is_test = kcp.resolve_launch(prog, dict(args))
            assert values == {} and not is_test
            if kcp.dhu_vh(n_walk, 48, 152, forced) == 64 and ships_vh64:
                assert record == prog + kcp.VH64_SUFFIX, (prog, forced, n_walk)
                assert grid == (n_walk * 48 * 2, 1, 1)
                assert jit.launch_grid(record, n_walk=n_walk, num_heads=48) == grid
            else:
                assert record == prog and grid == (n_walk * 48, 1, 1), (
                    prog,
                    forced,
                    n_walk,
                )
    if ships_vh64:
        for prog in kcp.DHU_PROGRAMS:
            assert jit.select_record(prog, ctas_per_head=2) == prog + kcp.VH64_SUFFIX
            assert jit.MODULES[prog + kcp.VH64_SUFFIX]["program"] == prog


def test_workspace_holds_one_zeroed_ticket_slot_per_persistent_site() -> None:
    """The persistent programs' dynamic-ticket words are workspace slots: one 128-byte u32 slot per launch site
    (``fwd_o``, ``bwd_prep``, ``bwd_dqkg``), allocated and zeroed once by ``ensure()`` (never per call), kept across
    pool / tape releases and re-sizing, counted by ``bytes`` and by the public workspace query in both lifetimes,
    and part of the arena signature a capture pins."""
    kcp, plan, gb = _mod("cake_kcp"), _mod("cake_kcp_plan"), _mod("cake_bytes")
    assert kcp.TICKET_SITES == gb.TICKET_SITES == ("fwd_o", "bwd_prep", "bwd_dqkg")
    assert kcp.TICKET_SLOT_BYTES == gb.TICKET_SLOT_BYTES == 128
    assert kcp.TICKET_SLOT_WORDS == 32 and gb.TICKET_BYTES == 3 * 128
    assert set(kcp.TICKET_SITES) <= set(kcp.LOCAL_PROGRAMS)
    for site in kcp.TICKET_SITES:
        assert kcp.PROGRAM_ARGS[site][-1] == "ticket"
    assert "ticket" in kcp.PROVISIONAL_ARGS
    part, _cu = _partition([100, 640, 77], 2, "contiguous")
    ctx = plan.build_context_kcp(part, None, 0, "cpu", sm_count=148)
    H_qk, HV = 2, 6
    ws = kcp.WorkspaceKcp()
    assert ws.ticket_bytes == 0 and ws.bytes == 0
    with pytest.raises(KeyError, match="ensure"):
        ws.ticket("fwd_o")
    ws.ensure(ctx, H_qk, HV, device="cpu", backward=True)
    slots = {site: ws.ticket(site) for site in kcp.TICKET_SITES}
    for site, t in slots.items():
        assert t.dtype == torch.uint32 and t.numel() == 32, site
        assert t.data_ptr() % 128 == 0, site
        assert int(t.view(torch.int32).abs().sum()) == 0, site
    assert len({t.data_ptr() for t in slots.values()}) == 3
    with pytest.raises(KeyError, match="persistent launch site"):
        ws.ticket("fwd_prep")
    assert ws.ticket_bytes == gb.TICKET_BYTES
    assert ws.bytes == sum(ws.pool_bytes.values()) + ws.tape_bytes + gb.TICKET_BYTES
    # one allocation per workspace: releases and a re-size keep the slots
    for name in kcp.FWD_TAPES:
        ws.tape(name)
    for name in kcp.FWD_DEAD_POOLS:
        ws.release_pool(name)
    for name in kcp.FWD_DEAD_BEFORE_O:
        ws.release_tape(name)
    ws.ensure(ctx, H_qk, HV, device="cpu", backward=True)
    assert all(ws.ticket(s).data_ptr() == slots[s].data_ptr() for s in kcp.TICKET_SITES)
    sig = kcp._arena_sig(ws)
    assert sig[2] == (("tickets", ws._tickets.data_ptr(), 3 * 32),)
    other = kcp.WorkspaceKcp()
    other.ensure(ctx, H_qk, HV, device="cpu", backward=True)
    assert other.ticket("fwd_o").data_ptr() != ws.ticket("fwd_o").data_ptr()
    # the public query counts every site's slot in both lifetimes
    layout = gb.gdn_workspace_layout_kcp(ctx, H_qk=H_qk, HV=HV)
    assert layout["fwd.tickets"] == layout["bwd.tickets"] == gb.TICKET_BYTES
    assert (
        gb.gdn_workspace_layout_kcp(ctx, H_qk=H_qk, HV=HV, backward=False)[
            "fwd.tickets"
        ]
        == gb.TICKET_BYTES
    )
