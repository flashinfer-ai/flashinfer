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

flashinfer public-API tests of the Cake deterministic GDN chunked-training facade ``flashinfer.cake_gdn_det_train``
over the backend package ``flashinfer.experimental.cake.cake_gdn_det_train``.

The backend suite ``test_cake_gdn_det_train.py`` imports only the package.  This file covers what needs flashinfer
itself: the facade's names (the mode-word names only), a stable API (no experimental marking, no
``ExperimentalWarning``), the package signature and every argument it forwards, no ``deterministic`` keyword and the
support query, without a device; and on SM100 / SM103 /
SM107 the facade's results against the package's (bitwise) and under ``torch.use_deterministic_algorithms(True)``
(no alert, bitwise the same).
"""

import ast
import inspect
import math
import warnings
from pathlib import Path

import pytest
import torch

import flashinfer.cake_gdn_det_train as facade
from flashinfer.api_logging import ExperimentalWarning
from flashinfer.experimental.cake import cake_gdn_det_train as pkg
from flashinfer.experimental.cake.cake_gdn_det_train import cake_backend
from tests.cake.test_helpers.cake_mode_names import assert_mode_names

FACADE_NAMES = (
    "CakeGDNDetFunction",
    "cake_gdn_det",
    "is_cake_gdn_det_train_supported",
)
# Package names the facade does not export: the low-level chain takes the package's chunk table and tape.
PACKAGE_ONLY_NAMES = ("cake_gdn_det_train_backward", "cake_gdn_det_train_forward")
FORMER_NAMES = (
    "ChunkGatedDeltaRuleDetFunction",
    "ChunkGatedDeltaRuleFunction",
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_det",
    "gdn_chunk_backward",
    "gdn_chunk_det_backward",
    "gdn_chunk_det_forward",
    "gdn_chunk_forward",
)


def _call(fn, *args, **kwargs):
    # a stable API: an ExperimentalWarning would raise
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        return fn(*args, **kwargs)


def test_the_facade_exports_the_mode_word_names_only() -> None:
    assert sorted(facade.__all__) == sorted(FACADE_NAMES)
    assert_mode_names(facade, "det")
    assert facade.CakeGDNDetFunction is pkg.CakeGDNDetFunction
    assert facade.CakeGDNDetFunction.__name__ == "CakeGDNDetFunction"
    assert facade.cake_gdn_det.__name__ == "cake_gdn_det"
    for name in FORMER_NAMES + PACKAGE_ONLY_NAMES:
        assert not hasattr(facade, name), name
        assert name not in dir(facade), name
    assert set(FACADE_NAMES) <= set(dir(facade))


def test_the_entry_point_is_a_stable_api(monkeypatch: pytest.MonkeyPatch) -> None:
    fn = facade.cake_gdn_det
    assert getattr(fn, "is_experimental", False) is False
    assert not hasattr(fn, "experimental_feature")
    assert not hasattr(facade, "_FEATURE")
    monkeypatch.setattr(pkg, "cake_gdn_det", lambda *args, **kwargs: "cake_gdn_det")
    assert _call(fn, *(None,) * 5) == "cake_gdn_det"


def test_every_facade_function_is_a_flashinfer_api_def() -> None:
    # the public functions of the facade module are real defs decorated with ``@flashinfer_api`` (the support query
    # too); the decorator returns the function unchanged while logging is off, so the check reads the source
    tree = ast.parse(Path(facade.__file__).read_text())
    decorators = {
        node.name: [ast.unparse(d) for d in node.decorator_list]
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }
    assert sorted(decorators) == sorted(n for n in FACADE_NAMES if n[0].islower())
    assert all(d == ["flashinfer_api"] for d in decorators.values()), decorators


def test_the_entry_point_takes_the_package_signature() -> None:
    want = inspect.signature(pkg.cake_gdn_det).parameters
    got = inspect.signature(facade.cake_gdn_det).parameters
    assert list(got) == list(want)
    for p in want:
        assert got[p].kind == want[p].kind, p
        assert got[p].default == want[p].default, p


def test_the_entry_point_forwards_every_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {}

    def fake(*args, **kwargs):
        calls["bound"] = inspect.signature(pkg.cake_chain.cake_gdn_det).bind(
            *args, **kwargs
        )
        return "cake_gdn_det"

    monkeypatch.setattr(pkg, "cake_gdn_det", fake)
    params = inspect.signature(facade.cake_gdn_det).parameters
    values = {p: object() for p in params}  # a distinct sentinel per parameter
    positional = [
        values[p]
        for p, v in params.items()
        if v.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    keywords = {
        p: values[p]
        for p, v in params.items()
        if v.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert _call(facade.cake_gdn_det, *positional, **keywords) == "cake_gdn_det"
    bound = calls["bound"].arguments
    assert set(bound) == set(values)
    assert all(bound[p] is values[p] for p in values)


@pytest.mark.parametrize("value", [True, False, None])
def test_no_entry_takes_a_deterministic_keyword(value) -> None:
    for module in (facade, pkg):
        for name in module.__all__:
            obj = getattr(module, name)
            target = obj.forward if inspect.isclass(obj) else obj
            assert "deterministic" not in inspect.signature(target).parameters, (
                module.__name__,
                name,
            )
    with pytest.raises(TypeError, match="deterministic"):
        _call(facade.cake_gdn_det, *(None,) * 5, deterministic=value)
    with pytest.raises(TypeError, match="deterministic"):
        facade.is_cake_gdn_det_train_supported(None, deterministic=value)


def test_the_support_query_returns_a_bool() -> None:
    assert facade.is_cake_gdn_det_train_supported("cpu") is False
    assert facade.is_cake_gdn_det_train_supported(torch.device("cpu")) is False
    result = facade.is_cake_gdn_det_train_supported()
    assert isinstance(result, bool)
    if not torch.cuda.is_available():
        assert result is False
    else:
        cap = tuple(torch.cuda.get_device_capability(torch.cuda.current_device()))
        assert result is (cake_backend.arch_for_capability(cap) is not None)


# --------------------------------------------------------------------------- device


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if not facade.is_cake_gdn_det_train_supported():
        pytest.skip("not an exported compute capability")
    return torch.device("cuda", torch.cuda.current_device())


def _inputs(seq_lens, heads, device, seed):
    num_heads, num_v_heads = heads
    total = sum(seq_lens)
    gen = torch.Generator().manual_seed(seed)
    out = {
        "q": torch.randn(1, total, num_heads, 128, generator=gen).to(torch.bfloat16),
        "k": torch.randn(1, total, num_heads, 128, generator=gen).to(torch.bfloat16),
        "v": torch.randn(1, total, num_v_heads, 128, generator=gen).to(torch.bfloat16),
        "g": torch.rand(1, total, num_v_heads, generator=gen) * -0.5,
        "beta": torch.rand(1, total, num_v_heads, generator=gen),
        "do": torch.randn(1, total, num_v_heads, 128, generator=gen).to(torch.bfloat16),
    }
    return {name: t.to(device).contiguous() for name, t in out.items()}


def _run(entry, inputs, seq_lens):
    leaves = {
        n: inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    }
    cu = [0]
    for n in seq_lens:
        cu.append(cu[-1] + n)
    o, _ = entry(
        leaves["q"],
        leaves["k"],
        leaves["v"],
        leaves["g"],
        leaves["beta"],
        scale=1.0 / math.sqrt(128),
        cu_seqlens=torch.tensor(cu, dtype=torch.int64, device=inputs["q"].device),
        cu_seqlens_cpu=torch.tensor(cu, dtype=torch.int64),
        use_qk_l2norm_in_kernel=True,
    )
    (o.float() * inputs["do"].float()).sum().backward()
    torch.cuda.synchronize()
    return {"o": o.detach(), **{"d" + n: leaves[n].grad for n in leaves}}


def test_facade_matches_the_package_bitwise() -> None:
    device = _device_or_skip()
    seq_lens = (700, 77, 300)
    inputs = _inputs(seq_lens, (2, 6), device, seed=3)
    got = _call(_run, facade.cake_gdn_det, inputs, seq_lens)
    want = _run(pkg.cake_gdn_det, inputs, seq_lens)
    for name in want:
        assert torch.equal(got[name], want[name]), name


def test_torch_deterministic_algorithms_raise_no_alert_on_the_facade() -> None:
    # strict mode: an alert of a non-deterministic torch op would raise; the det entry runs and gives the same bits
    device = _device_or_skip()
    seq_lens = (513, 2048, 100)
    inputs = _inputs(seq_lens, (2, 6), device, seed=11)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        strict = _call(_run, facade.cake_gdn_det, inputs, seq_lens)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    default = _call(_run, facade.cake_gdn_det, inputs, seq_lens)
    for name in default:
        assert torch.equal(strict[name], default[name]), name
