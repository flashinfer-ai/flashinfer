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

"""CPU checks of the shared TF32 kernel bodies of the KDA prefill modules.

Twenty-four TF32 modules compile one of nine shared kernel bodies. A body
selects a module's text through ``#if`` blocks on ``CAKE_KDA_<KNOB>`` compile
definitions whose values are the module's factory kwargs. The sidecar pins, per
module, the sha256 of the kernel source the fold replaced: the module's own
kernel file as it stood right before the fold, with the device helpers already
shared through ``cake_kda_device.cuh``.
"""

import pytest

import hashlib
import json
import re
from pathlib import Path

from flashinfer.jit import cake_kda_tf32

_DIRECTIVE = re.compile(r"\s*#\s*(ifndef|if|elif|else|endif|error)\b\s*(.*)")
_FOLD_MACRO = re.compile(r"\bCAKE_KDA_[A-Z0-9_]+\b")


def _folded():
    return {name: cake_kda_tf32.MODULES[name] for name in cake_kda_tf32._TF32_FOLD_OF}


def _defines(record):
    defines = {}
    for flag in record["compile_flags"]:
        if flag.startswith("-D"):
            name, _, value = flag[2:].partition("=")
            defines[name] = value
    return defines


def _factory_rows(name):
    # Every factory row that names the module (its rows differ only in kwargs
    # the body does not depend on); all of them belong to one family.
    rows = []
    for arch in cake_kda_tf32.MODULES[name]["arches"]:
        for family, variants in cake_kda_tf32.FACTORIES[arch].items():
            for (_args, items), module in variants.items():
                if module == name:
                    rows.append((family, dict(items)))
    assert rows and len({family for family, _ in rows}) == 1, name
    return [kwargs for _, kwargs in rows]


def _holds(condition, values):
    # The folds use terms, && within a group and || between groups only. A term
    # is CAKE_KDA_X (non-zero), !CAKE_KDA_X (zero) or CAKE_KDA_X == <int>.
    def term(text):
        text = text.strip()
        if "==" in text:
            name, _, literal = text.partition("==")
            return int(values[name.strip()]) == int(literal)
        return (int(values[text.lstrip("!")]) != 0) != text.startswith("!")

    return any(
        all(term(part) for part in group.strip().strip("()").split("&&"))
        for group in condition.split("||")
    )


def _expand(text, defines):
    """Expand the fold's own #if blocks; keep every other preprocessor line."""
    values = {k: v for k, v in defines.items() if k != "CAKE_KDA_KERNEL_SYMBOL"}
    out = []
    stack = []  # [owned by the fold, a branch was taken, this branch is active]
    for line in text.split("\n"):
        active = all(entry[2] for entry in stack)
        match = _DIRECTIVE.match(line)
        keyword, rest = match.groups() if match else (None, None)
        owned = keyword is not None and bool(_FOLD_MACRO.search(rest))
        if keyword in ("if", "ifndef"):
            if not owned:
                stack.append([False, True, True])
            else:
                if keyword == "ifndef":
                    taken = rest.strip() not in defines
                else:
                    taken = _holds(rest, values)
                stack.append([True, taken, taken])
                continue
        elif keyword == "elif" and stack[-1][0]:
            entry = stack[-1]
            taken = not entry[1] and _holds(rest, values)
            entry[1], entry[2] = entry[1] or taken, taken
            continue
        elif keyword == "else" and stack[-1][0]:
            entry = stack[-1]
            entry[1], entry[2] = True, not entry[1]
            continue
        elif keyword == "endif":
            if stack.pop()[0]:
                continue
        elif keyword == "error" and active and stack and stack[-1][0]:
            raise AssertionError(f"fold #error reached: {rest}")
        if active:
            out.append(
                line.replace(
                    "CAKE_KDA_KERNEL_SYMBOL", defines["CAKE_KDA_KERNEL_SYMBOL"]
                )
            )
    assert not stack
    return "\n".join(out)


@pytest.mark.cpu_only
def test_fold_table_routes_tf32_modules_onto_shared_bodies():
    folded = _folded()
    assert len(folded) == 24
    assert len(cake_kda_tf32._TF32_FOLDS) == 9
    assert not set(folded) & set(cake_kda_tf32._M128_FOLD_OF)
    for body, (knobs, members) in cake_kda_tf32._TF32_FOLDS.items():
        body_path = f"csrc/kda/tf32/cake_kda_tf32_{body}_kernel.cu"
        assert cake_kda_tf32._source_path(body_path).is_file(), body_path
        assert len(members) >= 2
        assert any(cake_kda_tf32._MODULE_ROWS[name][1] == body for name in members), (
            body
        )
        assert list(knobs) == sorted(set(knobs))
        for name, values in members.items():
            assert cake_kda_tf32._MODULE_ROWS[name][0] == "tf32", name
            assert list(values) == list(knobs), name
            assert folded[name]["sources"] == [body_path], name


@pytest.mark.cpu_only
def test_fold_defines_match_the_factory_rows_and_the_body_guards():
    for name, record in _folded().items():
        body, knobs, values = cake_kda_tf32._TF32_FOLD_OF[name]
        assert record["compile_flags"] == [
            "--use_fast_math",
            f"-DCAKE_KDA_KERNEL_SYMBOL=kernel_{record['cache_name']}",
            *(f"-DCAKE_KDA_{knob.upper()}={values[knob]}" for knob in knobs),
        ]
        for kwargs in _factory_rows(name):
            for knob in knobs:
                assert kwargs[knob] == values[knob], (name, knob)
        text = cake_kda_tf32._source_path(record["sources"][0]).read_text(
            encoding="utf-8"
        )
        guarded = re.findall(r"^#ifndef (CAKE_KDA_\w+)$", text, re.MULTILINE)
        assert guarded == [
            "CAKE_KDA_KERNEL_SYMBOL",
            *(f"CAKE_KDA_{knob.upper()}" for knob in knobs),
        ]
        assert text.count("CAKE_KDA_KERNEL_SYMBOL(") == 1
        assert "kernel_cake_kda_tf32_" not in text
        used = set(_FOLD_MACRO.findall(text)) - {"CAKE_KDA_KERNEL_SYMBOL"}
        assert used == {f"CAKE_KDA_{knob.upper()}" for knob in knobs}
    # Every module keeps its own kernel symbol (all modules load into one process).
    symbols = {_defines(r)["CAKE_KDA_KERNEL_SYMBOL"] for r in _folded().values()}
    assert len(symbols) == len(_folded())


@pytest.mark.cpu_only
def test_folded_bodies_expand_to_the_former_kernel_sources():
    # The sidecar pins, per module, the sha256 of the kernel source the fold
    # replaced (the module's own kernel file right before the fold).
    sidecar = Path(__file__).with_name("cake_kda_tf32_fold_sha256.json")
    pinned = json.loads(sidecar.read_text(encoding="utf-8"))
    folded = _folded()
    assert sorted(pinned) == sorted(folded)
    for name, record in folded.items():
        text = cake_kda_tf32._source_path(record["sources"][0]).read_text(
            encoding="utf-8"
        )
        expanded = _expand(text, _defines(record))
        assert "CAKE_KDA_" not in expanded, name
        assert f"kernel_{record['cache_name']}(" in expanded, name
        digest = hashlib.sha256(expanded.encode("utf-8")).hexdigest()
        assert digest == pinned[name]["sha256"], name
