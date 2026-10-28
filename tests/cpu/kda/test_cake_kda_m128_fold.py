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

"""CPU checks of the shared fused-M128 kernel bodies of the KDA prefill modules."""

import pytest

import hashlib
import json
import re
from pathlib import Path

from flashinfer.jit import cake_kda_tf32

FAMILY = "compiled_bf16_fused_m128"
_DIRECTIVE = re.compile(r"\s*#\s*(ifndef|if|elif|else|endif|error)\b\s*(.*)")
_FOLD_MACRO = re.compile(r"\bCAKE_KDA_[A-Z0-9_]+\b")
_CUDA_ARCH = {"sm_100a": 1000, "sm_103a": 1030}


def _folded():
    return {name: cake_kda_tf32.MODULES[name] for name in cake_kda_tf32._M128_FOLD_OF}


def _defines(record):
    defines = {}
    for flag in record["compile_flags"]:
        if flag.startswith("-D"):
            name, _, value = flag[2:].partition("=")
            defines[name] = value
    return defines


def _factory_kwargs(name):
    kwargs = None
    for arch in cake_kda_tf32.MODULES[name]["arches"]:
        keys = [
            key
            for key, module in cake_kda_tf32.FACTORIES[arch][FAMILY].items()
            if module == name
        ]
        assert len(keys) == 1, (name, arch)
        found = dict(keys[0][1])
        assert kwargs in (None, found), name
        kwargs = found
    return kwargs


def _holds(condition, values):
    # The folds use literals, && within a term and || between terms only.
    def literal(token):
        token = token.strip()
        return (int(values[token.lstrip("!")]) != 0) != token.startswith("!")

    return any(
        all(literal(part) for part in term.strip().strip("()").split("&&"))
        for term in condition.split("||")
    )


def _expand(text, defines, cuda_arch=None):
    """Expand the fold's own #if blocks; keep every other preprocessor line.

    With ``cuda_arch`` the pre-existing ``#if __CUDA_ARCH__ == N`` guard is
    evaluated as well (single-target modules).
    """
    values = {k: v for k, v in defines.items() if k != "CAKE_KDA_KERNEL_SYMBOL"}
    out = []
    stack = []  # [owned by the fold, a branch was taken, this branch is active]
    for line in text.split("\n"):
        active = all(entry[2] for entry in stack)
        match = _DIRECTIVE.match(line)
        keyword, rest = match.groups() if match else (None, None)
        owned = keyword is not None and (
            bool(_FOLD_MACRO.search(rest))
            or (cuda_arch is not None and rest.startswith("__CUDA_ARCH__"))
        )
        if keyword in ("if", "ifndef"):
            if not owned:
                stack.append([False, True, True])
            else:
                if keyword == "ifndef":
                    taken = rest.strip() not in defines
                elif rest.startswith("__CUDA_ARCH__"):
                    taken = int(rest.split("==")[1]) == cuda_arch
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
def test_fold_table_routes_fused_m128_modules_onto_shared_bodies():
    folded = _folded()
    assert len(folded) == 32
    assert len(cake_kda_tf32._M128_FOLDS) == 8
    for body, (flags, members) in cake_kda_tf32._M128_FOLDS.items():
        body_path = f"csrc/kda/bf16/cake_kda_bf16_{body}_kernel.cu"
        assert cake_kda_tf32._source_path(body_path).is_file(), body_path
        assert len(members) >= 2
        assert any(cake_kda_tf32._MODULE_ROWS[name][1] == body for name in members), (
            body
        )
        assert list(flags) == sorted(set(flags))
        for name, enabled in members.items():
            assert set(enabled) <= set(flags), name
            assert folded[name]["sources"] == [body_path], name
    for name in cake_kda_tf32._MODULE_ROWS:
        record = cake_kda_tf32.MODULES[name]
        if name in folded or name in cake_kda_tf32._TF32_FOLD_OF:
            continue
        directory, digest = cake_kda_tf32._MODULE_ROWS[name][:2]
        assert record["sources"] == [
            f"csrc/kda/{directory}/cake_kda_{directory}_{digest}_kernel.cu"
        ]
        assert record["compile_flags"] == ["--use_fast_math"]


@pytest.mark.cpu_only
def test_fold_defines_match_the_factory_rows_and_the_body_guards():
    for name, record in _folded().items():
        body, flags, enabled = cake_kda_tf32._M128_FOLD_OF[name]
        assert record["compile_flags"] == [
            "--use_fast_math",
            f"-DCAKE_KDA_KERNEL_SYMBOL=kernel_{record['cache_name']}",
            *(f"-DCAKE_KDA_{flag}={int(flag in enabled)}" for flag in flags),
        ]
        kwargs = _factory_kwargs(name)
        for flag in flags:
            assert kwargs[flag.lower()] is (flag in enabled), (name, flag)
        text = cake_kda_tf32._source_path(record["sources"][0]).read_text(
            encoding="utf-8"
        )
        guarded = re.findall(r"^#ifndef (CAKE_KDA_\w+)$", text, re.MULTILINE)
        assert guarded == ["CAKE_KDA_KERNEL_SYMBOL", *(f"CAKE_KDA_{f}" for f in flags)]
        assert text.count("CAKE_KDA_KERNEL_SYMBOL(") == 1
        assert "kernel_cake_kda_bf16_" not in text
        used = set(_FOLD_MACRO.findall(text)) - {"CAKE_KDA_KERNEL_SYMBOL"}
        assert used == {f"CAKE_KDA_{flag}" for flag in flags}
    # Every module keeps its own kernel symbol (all modules load into one process).
    symbols = {_defines(r)["CAKE_KDA_KERNEL_SYMBOL"] for r in _folded().values()}
    assert len(symbols) == len(_folded())


@pytest.mark.cpu_only
def test_folded_bodies_expand_to_the_former_kernel_sources():
    # The sidecar pins, per module, the sha256 of the kernel source the fold
    # replaced (for the single-target module after evaluating its
    # ``__CUDA_ARCH__`` guard, which the former source did not carry).
    sidecar = Path(__file__).with_name("cake_kda_m128_fold_sha256.json")
    pinned = json.loads(sidecar.read_text(encoding="utf-8"))
    folded = _folded()
    assert sorted(pinned) == sorted(folded)
    for name, record in folded.items():
        text = cake_kda_tf32._source_path(record["sources"][0]).read_text(
            encoding="utf-8"
        )
        cuda_arch = pinned[name].get("cuda_arch")
        if cuda_arch is not None:
            assert [_CUDA_ARCH[arch] for arch in record["arches"]] == [cuda_arch]
        expanded = _expand(text, _defines(record), cuda_arch)
        assert "CAKE_KDA_" not in expanded, name
        assert f"kernel_{record['cache_name']}(" in expanded, name
        digest = hashlib.sha256(expanded.encode("utf-8")).hexdigest()
        assert digest == pinned[name]["sha256"], name
