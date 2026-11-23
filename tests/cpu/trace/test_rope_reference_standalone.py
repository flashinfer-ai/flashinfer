"""Exported RoPE references must run standalone (GitHub issue #9830).

fi_trace copies each reference into the dumped JSON. A consumer runs it with
exec() in an empty namespace, so every module-level helper the reference
reaches must be inlined through ``_trace_reference_dependencies``. In-module
calls still resolve those helpers via module globals, so only an exec() of the
rendered string catches a missing one.
"""

import ast
import builtins
import copy
import dis
import json
import types
from pathlib import Path

import pytest
import torch

from flashinfer.trace.template import TraceTemplate, _render_reference_source
from flashinfer.trace.templates import rope

FI_TRACE_OUT = Path(__file__).resolve().parents[2] / "trace/fi_trace_out"

_TEMPLATES = sorted(
    (name, obj) for name, obj in vars(rope).items() if isinstance(obj, TraceTemplate)
)
_IDS = [name for name, _ in _TEMPLATES]
_BY_REFERENCE = {template.reference.__name__: template for _, template in _TEMPLATES}


def _entry_point(source):
    """Return the last top-level function, which the renderer emits last."""
    return [n.name for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)][
        -1
    ]


def _committed_rope_jsons():
    paths = []
    for path in sorted(FI_TRACE_OUT.glob("*rope*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if _entry_point(doc["reference"]) in _BY_REFERENCE:
            paths.append(path)
    return paths


_COMMITTED_JSONS = _committed_rope_jsons()


def _exec_in_fresh_namespace(source):
    namespace = {}
    exec(compile(source, "<exported-reference>", "exec"), namespace)  # noqa: S102
    return namespace


def _unbound_globals(source, namespace):
    """Return global names that the exported functions load but never define."""
    pending = [compile(source, "<exported-reference>", "exec")]
    missing = set()
    while pending:
        code = pending.pop()
        pending.extend(c for c in code.co_consts if isinstance(c, types.CodeType))
        for instr in dis.get_instructions(code):
            name = instr.argval
            if (
                instr.opname == "LOAD_GLOBAL"
                and name not in namespace
                and not hasattr(builtins, name)
            ):
                missing.add(name)
    return sorted(missing)


def _same(actual, expected):
    if isinstance(expected, torch.Tensor):
        if expected.element_size() == 1:  # compare FP8 outputs bit-for-bit
            actual, expected = actual.view(torch.uint8), expected.view(torch.uint8)
        return torch.equal(actual, expected)
    return len(actual) == len(expected) and all(
        _same(a, e) for a, e in zip(actual, expected, strict=True)
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("name,template", _TEMPLATES, ids=_IDS)
def test_rendered_reference_has_no_unbound_globals(name, template):
    source = _render_reference_source(template.reference)
    namespace = _exec_in_fresh_namespace(source)
    assert _unbound_globals(source, namespace) == []


@pytest.mark.cpu_only
@pytest.mark.parametrize("name,template", _TEMPLATES, ids=_IDS)
def test_rendered_reference_matches_module_reference(name, template):
    source = _render_reference_source(template.reference)
    exported = _exec_in_fresh_namespace(source)[template.reference.__name__]
    inputs = template.init(nnz=8, device="cpu")
    expected = template.reference(**copy.deepcopy(inputs))
    assert _same(exported(**copy.deepcopy(inputs)), expected)


@pytest.mark.cpu_only
@pytest.mark.parametrize("path", _COMMITTED_JSONS, ids=lambda path: path.stem)
def test_committed_json_runs_standalone(path):
    """The committed artifact is what consumers exec(); it must not go stale."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    entry = _entry_point(doc["reference"])
    namespace = _exec_in_fresh_namespace(doc["reference"])
    assert _unbound_globals(doc["reference"], namespace) == []

    template = _BY_REFERENCE[entry]
    if doc.get("init"):
        init = _exec_in_fresh_namespace(doc["init"])[_entry_point(doc["init"])]
    else:
        init = template.init
    inputs = init(nnz=8, device="cpu")
    expected = template.reference(**copy.deepcopy(inputs))
    assert _same(namespace[entry](**copy.deepcopy(inputs)), expected)


@pytest.mark.cpu_only
@pytest.mark.parametrize("is_neox", [True, False])
def test_exported_cos_sin_cache_reference_zero_rotation_is_identity(is_neox):
    template = rope.apply_rope_with_cos_sin_cache_trace
    source = _render_reference_source(template.reference)
    exported = _exec_in_fresh_namespace(source)[template.reference.__name__]
    query = torch.randn(2, 4096, dtype=torch.bfloat16)
    key = torch.randn(2, 1024, dtype=torch.bfloat16)
    positions = torch.tensor([1, 0], dtype=torch.int32)
    cos_sin_cache = torch.cat([torch.ones(2, 64), torch.zeros(2, 64)], dim=1)
    query_out, key_out = exported(positions, query, key, 128, cos_sin_cache, is_neox)
    assert torch.equal(query_out, query)
    assert torch.equal(key_out, key)
