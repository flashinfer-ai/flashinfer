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
"""

# The names of the non-deterministic Cake dense GEMMs (flashinfer-ai/flashinfer#8928): the public module
# ``flashinfer.cake_dense_gemm_nondet_train`` carries the mode word in every public name, no entry point takes a
# ``deterministic=`` keyword (nor any ``**kwargs``), the projection planner's keyword-only knobs are the deterministic
# planner's public ones plus ``splits`` and ``chains``, the former names and import paths are gone (no alias, no
# read-through to the deterministic package), and the module imports its backend only on first use.  CPU only.

import importlib
import importlib.util
import inspect
import re
import subprocess
import sys
import warnings
from types import ModuleType

import pytest
import torch

MODULE = "flashinfer.cake_dense_gemm_nondet_train"
BACKEND = "flashinfer.experimental.cake.cake_dense_gemm_nondet_train"
DET_MODULE = "flashinfer.cake_dense_gemm_det_train"
DET_BACKEND = "flashinfer.experimental.cake.cake_dense_gemm_det_train"
OP = "dense_gemm"
MODE = "nondet"
# the 20 public names of the module
PUBLIC = {
    "ARCHES",
    "is_cake_dense_gemm_nondet_train_supported",
    "cake_dense_gemm_nondet_train_projection",
    "cake_dense_gemm_nondet_train_projection_forward",
    "cake_dense_gemm_nondet_train_projection_dgrad",
    "cake_dense_gemm_nondet_train_projection_wgrad",
    "cake_dense_gemm_nondet_train_projection_plan",
    "cake_dense_gemm_nondet_train_projection_prepare",
    "cake_dense_gemm_nondet_train_projection_wgrad_prepare",
    "CakeDenseGemmNondetProjectionPlan",
    "CakeDenseGemmNondetProjectionLaunch",
    "cake_dense_gemm_nondet_train_router",
    "cake_dense_gemm_nondet_train_router_forward",
    "cake_dense_gemm_nondet_train_router_dgrad",
    "cake_dense_gemm_nondet_train_router_wgrad",
    "cake_dense_gemm_nondet_train_router_plan",
    "cake_dense_gemm_nondet_train_router_prepare",
    "CakeDenseGemmNondetRouterPlan",
    "CakeDenseGemmNondetRouterLaunch",
    "cake_dense_gemm_nondet_train_convenience_cache_clear",
}
# The callables among them (functions and classes; the module constants are not) and the functions.
CALLABLES = sorted(n for n in PUBLIC if not re.fullmatch(r"[A-Z][A-Z0-9_]*", n))
FUNCTIONS = sorted(n for n in CALLABLES if not n[0].isupper())
# The former import paths: the package and its modules.
FORMER_MODULES = (
    "flashinfer.experimental.dense_projection_gemm.cake_backend",
    "flashinfer.experimental.dense_projection_gemm.cake_jit",
    "flashinfer.experimental.dense_projection_gemm.cake_build",
)
FORMER_PACKAGE = "flashinfer.experimental.dense_projection_gemm"
# Removed backend names: the 18 former public names, the former-name map, the wrappers that only returned or ran a
# deterministic object, the mode resolvers and the forwarding of deterministic templates and programs.
REMOVED = {
    "cake_backend": (
        "dense_projection_gemm",
        "plan_dense_projection_gemm",
        "prepare_dense_projection_gemm",
        "projection_forward",
        "projection_dgrad",
        "prepare_projection_wgrad",
        "projection_wgrad",
        "plan_router_fp32_gemm",
        "prepare_router_fp32_gemm",
        "router_fp32_gemm",
        "router_forward",
        "router_dgrad",
        "router_wgrad",
        "clear_convenience_caches",
        "AtomicGemmPlan",
        "PreparedAtomicGemm",
        "AtomicRouterPlan",
        "PreparedAtomicRouterGemm",
        "DETERMINISTIC_NAMES",
        "instance_symbol",
        "split_fp32_to_bf16x3_",
        "_DET_TYPES",
        "_DET_SPELLED",
        "_det_entry",
        "_resolve_deterministic",
        "_require_false",
        "_det_package_name",
    ),
    "cake_jit": (
        "is_det_template",
        "is_det_program",
        "DET_TEMPLATE_PREFIXES",
        "DET_PROGRAM_PREFIX",
    ),
}
# Every public name the backend module defined before #8046 (FlashInfer c472a430: functions, classes and constants).
# The backend defines five of them itself (helpers and constants of its own programs); every other one is absent: no
# alias and no read-through to the deterministic package.
# fmt: off
FORMER_BACKEND_NAMES = (
    "ARCH_RULE_DEFAULTS", "BLOCK_K", "BLOCK_M", "BLOCK_N_CHOICES", "B_SWZ_CHOICES", "CH_T_COLS",
    "CONVENIENCE_CACHE_SIZE", "CTA_GROUP", "CTA_ROWS_CHOICES", "DeviceFacts", "EPI_MODES", "EPI_WARPS",
    "EXACT_WAVE_BLOCK_N", "GROUP_M", "GemmPlan", "K_TMA_BF16", "K_TWO_SLOTS", "L2_HINTS", "L2_PROMOS",
    "PANEL_BYTES", "PD_CHOICES", "PreparedGemm", "PreparedRouterGemm", "ROUTER_BLOCK_K", "ROUTER_BLOCK_M",
    "ROUTER_BLOCK_N", "ROUTER_SPLITS", "ROW_RULES", "RULE_PARENT", "RouterPlan", "SK_COUNTERS_MIN", "SK_DUMMY_BASE",
    "SK_MIN_ITERS", "SK_SLAB_CHOICES", "SLOT_BYTES", "SMEM_OPT_IN", "SMEM_OVERSIZED", "SPLIT_BLOCK_ELEMS",
    "SPLIT_TEMPLATE", "SUPPORTED_COMPUTE_CAPABILITIES", "W8_FORMS", "WORK_STAGES", "a_halves_of", "arch_for",
    "as_batched", "b_stage_bytes", "bind_launch", "box_rows_of", "check_stream_k_plan", "clear_convenience_caches",
    "convenience_key", "default_block_n", "default_cta_rows", "default_group_m", "default_hints",
    "default_hints_cgrp", "default_pf", "default_promo", "default_stages", "dense_projection_gemm", "device_facts",
    "device_l2_bytes", "device_sm_count", "epi_cols", "epi_mode", "epi_slots", "exact_waves", "flat_alias",
    "fused_split_module", "generated_program_available", "instance_key", "instance_symbol", "operand_view",
    "plan_dense_projection_gemm", "plan_router_fp32_gemm", "prepare_dense_projection_gemm",
    "prepare_projection_wgrad", "prepare_router_fp32_gemm", "projection_dgrad", "projection_forward",
    "projection_wgrad", "require_arch", "router_dgrad", "router_forward", "router_fp32_gemm", "router_layout_class",
    "router_symbol", "router_wgrad", "row_rule", "sk_exact_plan", "sk_parts_plan", "sk_sync_plan", "sm_pairs",
    "sm_units", "smem_limit_for", "split_fp32_to_bf16x3_", "staging_bytes", "stream_k_plan", "stt_bytes",
    "swap_small_m", "wave_working_set", "wave_working_set_cgrp", "wgrad_swapped", "wgrad_views",
)
# fmt: on
OWN_FORMER_NAMES = frozenset(
    {
        "ROUTER_BLOCK_K",
        "ROUTER_BLOCK_M",
        "ROUTER_BLOCK_N",
        "bind_launch",
        "router_symbol",
    }
)
# Names of the deterministic backend (former and #8928 spellings) and two names no module has: none is reachable here.
UNREACHABLE = (
    "projection_det_forward",
    "DetGemmPlan",
    "router_fp32_det_gemm",
    "ROW_RULES",
    "stream_k_plan",
    "wgrad_views",
    "cake_dense_gemm_det_train_projection_forward",
    "CakeDenseGemmDetProjectionPlan",
    "__wrapped__",
    "no_such_name",
)
# The deterministic planners' private knobs, which this module does not pass on.
DET_PRIVATE_KNOBS = {"_fallback", "_allow_swap", "_use_rules"}
# This package's own knobs of the projection planner.
OWN_KNOBS = {"splits", "chains"}


def mode_word_violations(name: str, *, op: str, mode: str) -> list[str]:
    """Why the public name ``name`` of the module ``flashinfer.cake_<op>_<mode>_train`` breaks the #8928 naming rules;
    empty when it follows them.  Self-contained (it reads no state of this test module), so another family can copy
    it as is.

    * A module constant (``^[A-Z][A-Z0-9_]*$``, e.g. ``ARCHES``) is exempt: the module path carries the mode.
    * The check is by word: ``name`` is split on ``_`` and on CamelCase boundaries, lower-cased, and must hold the
      mode word ``mode`` and no other mode word (``det`` and ``nondet`` are distinct words, so ``"det" in name`` is
      never the test), nor the verbs ``fwd`` / ``bwd``.
    * A function is ``cake_<op>_<mode>_train_<...>`` or the support query ``is_cake_<op>_<mode>_train_supported``;
      a class is ``Cake<Op><Mode><Upper...>`` (``<Op>`` and ``<Mode>`` in CamelCase).
    """
    if re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
        return []
    words = [
        w.lower()
        for part in name.split("_")
        for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z0-9]+", part)
    ]
    problems = []
    if mode not in words:
        problems.append(f"no mode word {mode!r} in {words}")
    others = {"det", "nondet"} - {mode}
    if others & set(words):
        problems.append(f"another mode word {sorted(others & set(words))} in {words}")
    if {"fwd", "bwd"} & set(words):
        problems.append(f"the verbs fwd / bwd are spelled forward / backward: {words}")
    prefix = f"cake_{op}_{mode}_train"
    if name[0].isupper():
        camel = "Cake" + "".join(w.capitalize() for w in f"{op}_{mode}".split("_"))
        if not re.fullmatch(re.escape(camel) + r"[A-Z]\w*", name):
            problems.append(f"a class is {camel}<Name>")
    elif not (name.startswith(prefix + "_") or name == f"is_{prefix}_supported"):
        problems.append(f"a function is {prefix}_<name> or is_{prefix}_supported")
    return problems


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )


def _keyword_only(fn) -> set[str]:
    return {
        n
        for n, p in inspect.signature(fn).parameters.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    }


def test_the_shared_mode_name_checks():
    """The shared #8928 checks (tests/cake/test_helpers/cake_mode_names.py): mode word and stem, no ``deterministic``
    keyword, one name per object across the module, the backend modules and the flashinfer top level, and the
    former paths and names failing at import in a fresh interpreter."""
    import flashinfer

    from tests.cake.test_helpers.cake_mode_names import assert_cake_mode_module

    module = importlib.import_module(MODULE)
    assert_cake_mode_module(
        module,
        MODE,
        namespaces=[
            flashinfer,
            importlib.import_module(f"{BACKEND}.cake_backend"),
            importlib.import_module(f"{BACKEND}.cake_jit"),
        ],
        old_modules=[FORMER_PACKAGE, *FORMER_MODULES],
        old_names=[
            (f"{BACKEND}.{submodule}", name)
            for submodule, names in sorted(REMOVED.items())
            for name in names
        ],
    )


def test_all_lists_the_public_names():
    module = importlib.import_module(MODULE)
    assert set(module.__all__) == PUBLIC
    assert len(module.__all__) == len(PUBLIC)
    assert sorted(module.__all__) == module.__all__


@pytest.mark.parametrize("name", sorted(PUBLIC))
def test_every_public_name_carries_the_mode_word(name):
    assert mode_word_violations(name, op=OP, mode=MODE) == []


def test_the_mode_word_check_reads_words():
    """The check splits words: ``nondet`` is not ``det``, a CamelCase class is split, constants are exempt."""
    check = mode_word_violations
    assert check("cake_dense_gemm_nondet_train_router", op=OP, mode="nondet") == []
    assert check("cake_dense_gemm_nondet_train_router", op=OP, mode="det")
    assert check("cake_dense_gemm_det_train_router", op=OP, mode="nondet")
    assert check("CakeDenseGemmNondetRouterPlan", op=OP, mode="nondet") == []
    assert check("CakeDenseGemmNondetRouterPlan", op=OP, mode="det")
    assert check("CakeDenseGemmDetRouterPlan", op=OP, mode="det") == []
    assert check("RouterPlan", op=OP, mode="nondet")
    assert check("cake_dense_gemm_nondet_train_projection_fwd", op=OP, mode="nondet")
    assert (
        check("is_cake_dense_gemm_nondet_train_supported", op=OP, mode="nondet") == []
    )
    assert check("router_wgrad", op=OP, mode="nondet")
    assert check("ARCHES", op=OP, mode="nondet") == []
    assert check("Arches", op=OP, mode="nondet")


@pytest.mark.parametrize("name", CALLABLES)
def test_no_entry_point_takes_a_mode_keyword(name):
    """``deterministic=`` does not bind to any callable of the module (a class: its constructor), and none takes
    ``**kwargs``."""
    obj = getattr(importlib.import_module(MODULE), name)
    assert callable(obj)
    signature = inspect.signature(obj)
    with pytest.raises(TypeError):
        signature.bind_partial(deterministic=False)
    assert not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    if isinstance(obj, type) and issubclass(obj, torch.autograd.Function):
        with pytest.raises(TypeError):
            inspect.signature(obj.forward).bind_partial(None, deterministic=False)


@pytest.mark.parametrize("value", [False, True, None])
@pytest.mark.parametrize("name", FUNCTIONS)
def test_a_mode_keyword_raises_type_error_before_anything_runs(
    monkeypatch, name, value
):
    """A call that passes ``deterministic=`` (any value, ``False`` included) raises ``TypeError`` before anything is
    planned, allocated or launched: the deterministic package and this package's paths are never reached."""
    backend = importlib.import_module(f"{BACKEND}.cake_backend")
    reached = []

    def forbidden(*args, **kwargs):
        reached.append((args, kwargs))
        raise AssertionError("a call with deterministic= ran")

    monkeypatch.setattr(backend, "deterministic_backend", forbidden)
    for path in ("_atomic_k1", "_atomic_k2", "_det_k1", "_router_gemm", "_splitk_k2"):
        monkeypatch.setattr(backend, path, forbidden)
    for path in ("_k1_cached_launch", "_k2_entry", "_drop_convenience_entries"):
        monkeypatch.setattr(backend, path, forbidden)
    fn = getattr(importlib.import_module(MODULE), name)
    required = [
        p
        for p in inspect.signature(fn).parameters.values()
        if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        and p.default is inspect.Parameter.empty
    ]
    with pytest.raises(TypeError, match="deterministic"):
        fn(*(object() for _ in required), deterministic=value)
    assert not reached


@pytest.mark.parametrize(
    "helper",
    ["router_symbol", "bind_launch", "plan_atomic_gemm", "plan_atomic_router"],
)
def test_the_backend_helpers_take_no_mode_keyword(helper):
    backend = importlib.import_module(f"{BACKEND}.cake_backend")
    with pytest.raises(TypeError):
        inspect.signature(getattr(backend, helper)).bind_partial(deterministic=False)


@pytest.mark.parametrize(
    "entry, det_entry",
    [
        ("projection_plan", "projection_plan"),
        ("projection_prepare", "projection_prepare"),
    ],
)
def test_the_projection_planner_knobs_are_the_deterministic_planners(entry, det_entry):
    """The keyword-only knobs of the projection plan / prepare entries are the deterministic entry's public ones plus
    ``splits`` and ``chains`` (a change of the deterministic signature shows here), each defaulting to one private
    sentinel: a knob the caller does not give is not passed on."""
    fn = getattr(
        importlib.import_module(MODULE), f"cake_dense_gemm_nondet_train_{entry}"
    )
    det_fn = getattr(
        importlib.import_module(DET_MODULE), f"cake_dense_gemm_det_train_{det_entry}"
    )
    det_knobs = _keyword_only(det_fn)
    assert det_knobs >= DET_PRIVATE_KNOBS or entry == "projection_prepare"
    public = {k for k in det_knobs if not k.startswith("_")}
    assert _keyword_only(fn) == public | OWN_KNOBS
    defaults = {
        p.default
        for p in inspect.signature(fn).parameters.values()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert len(defaults) == 1 and None not in defaults


def test_only_the_knobs_a_caller_gives_reach_the_deterministic_planner(monkeypatch):
    """On a layout routed to the deterministic program the projection plan / prepare entries call the deterministic
    planner with exactly the caller's knobs (none when the caller gives none)."""
    backend = importlib.import_module(f"{BACKEND}.cake_backend")
    det = importlib.import_module(f"{DET_BACKEND}.cake_backend")
    calls = []

    def recorder(*args, **kwargs):
        calls.append(kwargs)
        return "det-object"

    monkeypatch.setattr(det, "cake_dense_gemm_det_train_projection_plan", recorder)
    monkeypatch.setattr(det, "cake_dense_gemm_det_train_projection_prepare", recorder)
    monkeypatch.setattr(backend, "deterministic_backend", lambda: det)
    A, B, out = torch.zeros(8, 32), torch.zeros(16, 32).t(), torch.zeros(8, 16)  # kk
    m = importlib.import_module(MODULE)
    assert m.cake_dense_gemm_nondet_train_projection_plan(A, B, out) == "det-object"
    assert (
        m.cake_dense_gemm_nondet_train_projection_plan(
            A, B, out, sm_count=148, l2_bytes=1 << 20, sk=False
        )
        == "det-object"
    )
    assert m.cake_dense_gemm_nondet_train_projection_prepare(A, B, out) == "det-object"
    assert (
        m.cake_dense_gemm_nondet_train_projection_prepare(A, B, out, enable_pdl=True)
        == "det-object"
    )
    assert calls == [
        {},
        {"sm_count": 148, "l2_bytes": 1 << 20, "sk": False},
        {},
        {"enable_pdl": True},
    ]


@pytest.mark.parametrize("name", sorted(PUBLIC))
def test_the_module_returns_the_backend_object(name):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no ExperimentalWarning: a stable module
        obj = getattr(importlib.import_module(MODULE), name)
    submodule = "cake_jit" if name == "ARCHES" else "cake_backend"
    assert obj is getattr(importlib.import_module(f"{BACKEND}.{submodule}"), name)
    if inspect.isfunction(obj) or inspect.isclass(obj):
        assert obj.__name__ == name
        assert inspect.unwrap(obj).__name__ == name
        assert obj.__module__ == f"{BACKEND}.cake_backend"


def test_an_unknown_name_raises_attribute_error():
    module = importlib.import_module(MODULE)
    with pytest.raises(AttributeError):
        module.router_fp32_gemm  # noqa: B018
    # the deterministic backend's setting (it bounds both packages), not a name of this module
    with pytest.raises(AttributeError):
        module.CONVENIENCE_CACHE_SIZE  # noqa: B018


@pytest.mark.parametrize("submodule, names", sorted(REMOVED.items()))
def test_the_former_names_are_removed(submodule, names):
    backend = importlib.import_module(f"{BACKEND}.{submodule}")
    assert [n for n in names if hasattr(backend, n)] == []


@pytest.mark.parametrize("name", FORMER_BACKEND_NAMES)
def test_no_backend_name_from_before_8046_reads_through(name):
    """A name the backend had before #8046 is this package's own object (defined in its module) or absent."""
    backend = importlib.import_module(f"{BACKEND}.cake_backend")
    if name in OWN_FORMER_NAMES:
        obj = vars(backend)[name]
        assert not callable(obj) or obj.__module__ == backend.__name__
    else:
        assert name not in vars(backend) and name not in dir(backend)
        with pytest.raises(AttributeError):
            getattr(backend, name)


@pytest.mark.parametrize("name", UNREACHABLE)
def test_deterministic_names_are_not_reachable(name):
    backend = importlib.import_module(f"{BACKEND}.cake_backend")
    with pytest.raises(AttributeError):
        getattr(backend, name)


def test_no_read_through_to_the_deterministic_package():
    """The backend has no module ``__getattr__`` (no read-through), and no attribute of its modules is an object of
    the deterministic package (the deterministic package is reached only through ``deterministic_backend()``); a
    public name of the deterministic module is not an attribute here."""
    backend = importlib.import_module(f"{BACKEND}.cake_backend")
    jit = importlib.import_module(f"{BACKEND}.cake_jit")
    package = importlib.import_module(BACKEND)
    assert "__getattr__" not in vars(backend) and "__getattr__" not in vars(jit)
    det = importlib.import_module(f"{DET_BACKEND}.cake_backend")
    det_jit = importlib.import_module(f"{DET_BACKEND}.cake_jit")
    det_objects = {
        id(obj)
        for module in (det, det_jit)
        for obj in vars(module).values()
        if (inspect.isfunction(obj) or inspect.isclass(obj))
        and obj.__module__.startswith(DET_BACKEND)
    }
    for module in (backend, jit, package):
        leaked = sorted(
            n
            for n, obj in vars(module).items()
            if id(obj) in det_objects
            or (isinstance(obj, ModuleType) and obj.__name__.startswith(DET_BACKEND))
        )
        assert not leaked, (module.__name__, leaked)
    det_public = importlib.import_module(DET_MODULE).__all__
    for name in det_public:
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            assert not hasattr(backend, name), name


@pytest.mark.parametrize("name", FORMER_MODULES)
def test_the_former_modules_fail_at_import(name):
    result = _run(f"import importlib; importlib.import_module({name!r})")
    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr


def test_the_former_package_is_gone():
    # A checkout may keep an untracked ``aot/`` or ``__pycache__/`` in the former directory, which Python
    # imports as a namespace package (no ``origin``); it holds no module.
    result = _run(
        "import importlib.util; "
        f"spec = importlib.util.find_spec({FORMER_PACKAGE!r}); "
        "print(spec is None or spec.origin is None)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


def test_imports_are_lazy():
    code = f"""
import sys
import warnings
import flashinfer
loaded = sorted(m for m in sys.modules if "cake_dense_gemm_" in m)
assert loaded == [], loaded
for name in {sorted(PUBLIC)!r}:
    assert name not in vars(flashinfer), name
rows = getattr(flashinfer, "_LAZY_EXPORTS", {{}})
dense = {{k: v for k, v in rows.items() if "cake_dense_gemm" in str(v[0])}}
assert all(v[1] is None for v in dense.values()), dense
import {MODULE} as m
loaded = sorted(n for n in sys.modules if n.startswith("{BACKEND}"))
assert loaded == [], loaded
assert not [n for n in sys.modules if n.startswith("{DET_BACKEND}")]
# the first access loads the backend: no ExperimentalWarning (a stable module), no other warning
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    m.cake_dense_gemm_nondet_train_router_wgrad
assert "{BACKEND}.cake_backend" in sys.modules
assert not caught, [(w.category.__name__, str(w.message)) for w in caught]
print("ok")
"""
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_the_support_query():
    module = importlib.import_module(MODULE)
    det = importlib.import_module(DET_MODULE)
    with pytest.raises(TypeError):
        module.is_cake_dense_gemm_nondet_train_supported()  # the device is required
    assert (
        module.is_cake_dense_gemm_nondet_train_supported(torch.device("cpu")) is False
    )
    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
        assert module.is_cake_dense_gemm_nondet_train_supported(
            device
        ) == det.is_cake_dense_gemm_det_train_supported(device)
