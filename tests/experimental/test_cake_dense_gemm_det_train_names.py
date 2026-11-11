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

# The names of the deterministic Cake dense GEMMs (flashinfer-ai/flashinfer#8928): the public module
# ``flashinfer.cake_dense_gemm_det_train`` carries the mode word in every public name, no entry point takes a
# ``deterministic=`` keyword, the former names and import paths are gone (no alias), the module imports its backend
# only on first use, the deterministic entry points never consult the PyTorch deterministic-algorithms setting,
# and the generated registry, manifest and sources spell only their own names.  CPU only, apart from the one
# device test at the end.

import importlib
import importlib.util
import inspect
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch

MODULE = "flashinfer.cake_dense_gemm_det_train"
BACKEND = "flashinfer.experimental.cake.cake_dense_gemm_det_train"
MODE = "det"
# the 21 public names of the module
PUBLIC = {
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "is_cake_dense_gemm_det_train_supported",
    "cake_dense_gemm_det_train_projection",
    "cake_dense_gemm_det_train_projection_forward",
    "cake_dense_gemm_det_train_projection_dgrad",
    "cake_dense_gemm_det_train_projection_wgrad",
    "cake_dense_gemm_det_train_projection_plan",
    "cake_dense_gemm_det_train_projection_prepare",
    "cake_dense_gemm_det_train_projection_wgrad_prepare",
    "CakeDenseGemmDetProjectionPlan",
    "CakeDenseGemmDetProjectionLaunch",
    "cake_dense_gemm_det_train_router",
    "cake_dense_gemm_det_train_router_forward",
    "cake_dense_gemm_det_train_router_dgrad",
    "cake_dense_gemm_det_train_router_wgrad",
    "cake_dense_gemm_det_train_router_plan",
    "cake_dense_gemm_det_train_router_prepare",
    "CakeDenseGemmDetRouterPlan",
    "CakeDenseGemmDetRouterLaunch",
    "cake_dense_gemm_det_train_convenience_cache_clear",
}
# The former import paths: the package and its modules.
FORMER_MODULES = (
    "flashinfer.experimental.dense_projection_det_gemm.cake_backend",
    "flashinfer.experimental.dense_projection_det_gemm.cake_jit",
    "flashinfer.experimental.dense_projection_det_gemm.cake_build",
)
FORMER_PACKAGE = "flashinfer.experimental.dense_projection_det_gemm"
# Removed backend names: the former public names and the aliases of the names before the det rename.
REMOVED = {
    "cake_backend": (
        "DetGemmPlan",
        "PreparedDetGemm",
        "RouterDetPlan",
        "PreparedRouterDetGemm",
        "plan_dense_projection_det_gemm",
        "prepare_dense_projection_det_gemm",
        "dense_projection_det_gemm",
        "projection_det_forward",
        "projection_det_dgrad",
        "prepare_projection_det_wgrad",
        "projection_det_wgrad",
        "plan_router_fp32_det_gemm",
        "prepare_router_fp32_det_gemm",
        "router_fp32_det_gemm",
        "router_det_forward",
        "router_det_dgrad",
        "router_det_wgrad",
        "clear_convenience_caches",
        "GemmPlan",
        "PreparedGemm",
        "PreparedRouterGemm",
        "RouterPlan",
        "dense_projection_gemm",
        "plan_dense_projection_gemm",
        "plan_router_fp32_gemm",
        "prepare_dense_projection_gemm",
        "prepare_projection_wgrad",
        "prepare_router_fp32_gemm",
        "projection_dgrad",
        "projection_forward",
        "projection_wgrad",
        "router_dgrad",
        "router_forward",
        "router_fp32_gemm",
        "router_wgrad",
        "split_fp32_to_bf16x3_",
    ),
    "cake_jit": ("load_cake_dense_projection_gemm_module",),
}
# Spellings of the non-deterministic package that the deterministic registry, manifest and sources never contain.
FORMER_TOKENS = (
    "dense_projection_gemm",
    "dense_proj_gemm",
    "router_fp32_gemm",
    "split_fp32_bf16x3",
    "DENSE_PROJ_GEMM",
)
CONSTANT = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _tokens(name: str) -> list[str]:
    """The words of ``name``: split on ``_`` and on CamelCase boundaries, lower-cased."""
    words = []
    for part in name.split("_"):
        words += re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z0-9]+", part)
    return [w.lower() for w in words if w]


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )


def test_the_shared_mode_name_checks():
    """The shared #8928 checks (tests/cake/test_helpers/cake_mode_names.py): mode word and stem, no ``deterministic``
    keyword, one name per object across the module, the backend modules and the flashinfer top level, and the
    former paths and names failing at import in a fresh interpreter."""
    import flashinfer

    from tests.cake.test_helpers.cake_mode_names import assert_cake_mode_module

    module = importlib.import_module(MODULE)
    assert_cake_mode_module(
        module,
        "det",
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


@pytest.mark.parametrize("name", sorted(PUBLIC))
def test_every_public_name_carries_the_mode_word(name):
    if CONSTANT.match(name):
        return  # module constants: the module path carries the mode
    tokens = _tokens(name)
    assert MODE in tokens and "nondet" not in tokens, (name, tokens)
    if name[0].isupper():
        assert name.startswith("CakeDenseGemmDet") and name[16].isupper()
    else:
        assert name.startswith("cake_dense_gemm_det_train_") or (
            name == "is_cake_dense_gemm_det_train_supported"
        )
    assert not {"fwd", "bwd"} & set(tokens)


@pytest.mark.parametrize("name", sorted(n for n in PUBLIC if not CONSTANT.match(n)))
def test_no_entry_point_takes_a_mode_keyword(name):
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


@pytest.mark.parametrize("name", sorted(PUBLIC))
def test_the_module_returns_the_backend_object(name):
    obj = getattr(importlib.import_module(MODULE), name)
    submodule = "cake_jit" if name == "ARCHES" else "cake_backend"
    assert obj is getattr(importlib.import_module(f"{BACKEND}.{submodule}"), name)
    if inspect.isfunction(obj) or inspect.isclass(obj):
        assert obj.__name__ == name
        assert inspect.unwrap(obj).__name__ == name
        assert obj.__module__ == f"{BACKEND}.cake_backend"


def test_first_access_issues_no_experimental_warning():
    """A stable module: the first access of every name, in a fresh process, issues no ExperimentalWarning."""
    code = f"""
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import {MODULE} as m
    for name in m.__all__:
        getattr(m, name)
print(sorted({{type(w.message).__name__ for w in caught if "Experimental" in type(w.message).__name__}}))
"""
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"


def test_an_unknown_name_raises_attribute_error():
    module = importlib.import_module(MODULE)
    with pytest.raises(AttributeError):
        module.dense_projection_det_gemm  # noqa: B018


@pytest.mark.parametrize("submodule, names", sorted(REMOVED.items()))
def test_the_former_names_are_removed(submodule, names):
    backend = importlib.import_module(f"{BACKEND}.{submodule}")
    assert [n for n in names if hasattr(backend, n)] == []


@pytest.mark.parametrize("name", FORMER_MODULES)
def test_the_former_modules_fail_at_import(name):
    code = f"""
import importlib
try:
    importlib.import_module({name!r})
except ModuleNotFoundError as exc:
    print("missing", exc.name)
else:
    print("imported")
"""
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] in (
        f"missing {name}",
        f"missing {FORMER_PACKAGE}",
    )


def test_the_former_package_is_gone():
    # A checkout may keep an untracked ``aot/`` or ``__pycache__/`` in the former directory, which Python
    # imports as a namespace package (no ``origin``); it holds no module.
    spec = importlib.util.find_spec(FORMER_PACKAGE)
    assert spec is None or spec.origin is None


def test_imports_are_lazy():
    code = f"""
import sys
import flashinfer
loaded = sorted(m for m in sys.modules if m.startswith("flashinfer.experimental.cake.cake_dense_gemm"))
assert loaded == [], loaded
for name in {sorted(PUBLIC)!r}:
    assert name not in vars(flashinfer), name
rows = getattr(flashinfer, "_LAZY_EXPORTS", {{}})
dense = {{k: v for k, v in rows.items() if "cake_dense_gemm" in str(v[0])}}
assert all(v[1] is None for v in dense.values()), dense
import {MODULE} as m
loaded = sorted(n for n in sys.modules if n.startswith("{BACKEND}"))
assert loaded == [], loaded
m.cake_dense_gemm_det_train_projection
assert "{BACKEND}.cake_backend" in sys.modules
print("ok")
"""
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_the_backend_never_reads_the_torch_deterministic_setting():
    """The module is the mode: the deterministic entry points neither alert nor change path under
    ``torch.use_deterministic_algorithms`` (strict or ``warn_only``)."""
    root = Path(importlib.import_module(f"{BACKEND}.cake_jit").__file__).parent
    for path in sorted(root.glob("*.py")):
        text = path.read_text()
        for token in (
            "are_deterministic_algorithms_enabled",
            "is_deterministic_algorithms_warn_only_enabled",
            "_alert_not_deterministic",
            "use_deterministic_algorithms",
        ):
            assert token not in text, (path.name, token)


# the build leaf of a several-chain weight-gradient program's registry record: the shared module's path
_CHAIN_BUILD_LEAF = re.compile(
    r'"build": \{"module": "[\w.]+", "kwargs": \{[^{}]*"chains": [2-9][^{}]*\}\}'
)


def _former_lines(text: str) -> list[str]:
    """The lines of ``text`` spelling a former token, the build leaf of a several-chain program's record
    (``_CHAIN_BUILD_LEAF``) aside."""
    out = []
    for line in text.split("\n"):
        if '"template": "chain_det_gemm_' in line:
            line = _CHAIN_BUILD_LEAF.sub('"build": {}', line)
        if any(t in line for t in FORMER_TOKENS):
            out.append(line)
    return out


def test_the_registry_spells_only_canonical_names():
    jit = importlib.import_module(f"{BACKEND}.cake_jit")
    backend = importlib.import_module(f"{BACKEND}.cake_backend")
    assert jit.MODULES and jit.KERNELS
    # The generated names (library stems, program prefixes, kernel namespaces, templates) are build identities
    # and keep their spelling across the module rename.
    for name, record in jit.MODULES.items():
        assert (
            name.startswith("cake_dense_projection_det_gemm_")
            and record["library"] == name
        )
        assert record["kernel"] in (f"kernel_{name}",) or record["kernel"].startswith(
            "cake::dense_projection_det_gemm::"
        )
        assert record["template"].startswith(
            (
                "dense_proj_det_gemm_",
                "router_fp32_det_gemm_",
                "split_fp32_det_bf16x3",
                "swap_ab_",
                "chain_det_gemm_",
            )
        )
        if record["template"].startswith("chain_det_gemm_"):
            # the several-chain weight-gradient programs build the weight-gradient module both packages share: its
            # path keeps the former spelling, its program names do not
            assert record["build"]["kwargs"]["chains"] > 1
        else:
            assert (
                "build" not in record
                or ".dense_projection_gemm" not in record["build"]["module"]
            )
    templates = {t for table in jit.KERNELS.values() for t in table}
    assert backend.SPLIT_TEMPLATE in templates
    assert {
        backend.router_symbol(a, b, t)
        for a, b, t in ((False, False, False), (False, True, False), (True, True, True))
    } <= templates
    assert backend._TEMPLATE_PREFIX == "dense_proj_det_gemm_"
    assert all(
        t.startswith(backend._TEMPLATE_PREFIX)
        for t in templates
        if t.startswith("dense_proj")
    )
    assert _former_lines(json.dumps([jit.MODULES, jit.KERNELS])) == []


def test_no_former_spelling_in_the_package():
    root = Path(importlib.import_module(f"{BACKEND}.cake_jit").__file__).parent
    manifest = json.loads((root / "build.json").read_text())
    assert manifest["package"] == BACKEND
    for entry in manifest["libraries"]:
        assert entry["name"].startswith("cake_dense_projection_det_gemm_")
        assert all(
            s.startswith(
                ("csrc/cake_dense_projection_det_gemm/", "csrc/cake_device_helpers/")
            )
            for s in entry["sources"]
        )
    files = [
        root / "build.json",
        *sorted(root.glob("*.py")),
        *sorted((root / "csrc").rglob("*")),
    ]
    found = {
        str(path.relative_to(root)): lines
        for path in files
        if path.is_file() and (lines := _former_lines(path.read_text(errors="replace")))
    }
    assert found == {}
    assert not [
        p for p in (root / "csrc").rglob("*") if any(t in p.name for t in FORMER_TOKENS)
    ]


def test_the_support_query():
    module = importlib.import_module(MODULE)
    assert module.is_cake_dense_gemm_det_train_supported(torch.device("cpu")) is False
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        assert module.is_cake_dense_gemm_det_train_supported(
            torch.device("cuda", torch.cuda.current_device())
        ) == ((major, minor) in module.SUPPORTED_COMPUTE_CAPABILITIES)


# entry point -> its positional arguments on CPU tensors (T, K, N) = (257, 6144, 576), the kv_a projection, and the
# 256-expert router
def _cpu_calls():
    T, K, N, R = 257, 6144, 576, 256
    bf, f32 = torch.bfloat16, torch.float32
    X, W, G = (
        torch.zeros(T, K, dtype=bf),
        torch.zeros(N, K, dtype=bf),
        torch.zeros(T, N, dtype=bf),
    )
    Xr, Wr, Gr = torch.zeros(T, K), torch.zeros(R, K), torch.zeros(T, R)
    plan = dict(
        arch="sm_100a", sm_count=148, l2_bytes=126 * 2**20
    )  # the B200 device facts
    return {
        "cake_dense_gemm_det_train_projection": (
            (X, W.t(), torch.zeros(T, N, dtype=bf)),
            {},
        ),
        "cake_dense_gemm_det_train_projection_forward": ((X, W), {}),
        "cake_dense_gemm_det_train_projection_dgrad": ((G, W), {}),
        "cake_dense_gemm_det_train_projection_wgrad": ((G, X), {}),
        "cake_dense_gemm_det_train_projection_plan": (
            (X, W.t(), torch.zeros(T, N, dtype=bf)),
            plan,
        ),
        "cake_dense_gemm_det_train_projection_prepare": (
            (X, W.t(), torch.zeros(T, N, dtype=bf)),
            {},
        ),
        "cake_dense_gemm_det_train_projection_wgrad_prepare": (
            (G, X, torch.zeros(N, K, dtype=f32)),
            {},
        ),
        "cake_dense_gemm_det_train_router": ((Xr, Wr.t(), torch.zeros(T, R)), {}),
        "cake_dense_gemm_det_train_router_forward": ((Xr, Wr), {}),
        "cake_dense_gemm_det_train_router_dgrad": ((Gr, Wr), {}),
        "cake_dense_gemm_det_train_router_wgrad": ((Gr, Xr), {}),
        "cake_dense_gemm_det_train_router_plan": ((Xr, Wr.t(), torch.zeros(T, R)), {}),
        "cake_dense_gemm_det_train_router_prepare": (
            (Xr, Wr.t(), torch.zeros(T, R)),
            {},
        ),
        "cake_dense_gemm_det_train_convenience_cache_clear": ((), {}),
    }


def test_every_entry_point_is_covered_by_the_mode_test():
    callables = {n for n in PUBLIC if not CONSTANT.match(n) and not n[0].isupper()}
    assert set(_cpu_calls()) == callables - {"is_cake_dense_gemm_det_train_supported"}


@pytest.mark.parametrize("warn_only", [False, True])
@pytest.mark.parametrize("name", sorted(_cpu_calls()))
def test_no_entry_point_alerts_under_torch_deterministic_mode(name, warn_only):
    """Under ``torch.use_deterministic_algorithms(True)`` (strict or ``warn_only``) a deterministic entry point raises
    no deterministic-mode RuntimeError and issues no deterministic-mode warning.  On CPU tensors the launches stop at
    the device check (a ValueError); the planner runs (``..._projection_plan`` returns its plan)."""
    m = importlib.import_module(MODULE)
    args, kwargs = _cpu_calls()[name]
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    error = None
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = getattr(m, name)(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                error = exc
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
    assert [w for w in caught if "determinis" in str(w.message)] == []
    assert not (isinstance(error, RuntimeError) and "determinis" in str(error)), error
    if name in (
        "cake_dense_gemm_det_train_projection_plan",
        "cake_dense_gemm_det_train_router_plan",
    ):
        assert error is None and result is not None  # the planners need no device
