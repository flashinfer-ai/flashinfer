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

Public entry points of ``flashinfer.cake_dsa_sparse_attention_det_train`` over the backend package
``flashinfer.experimental.cake.cake_dsa_sparse_attention_det_train``: the environment variables the facade resolves
into the backend's keywords, the parity keywords, the README's per-output statements, and the four entries' gradients
repeating bit for bit (with repeated destination rows and a nonzero ``dkv_acc``) next to a resident side kernel.
"""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import flashinfer.cake_dsa_sparse_attention_det_train as public_api
from flashinfer.experimental.cake.cake_dsa_sparse_attention_det_train import (
    cake_backend,
    cake_jit,
)
from tests.experimental.test_helpers.cake_dsa_sparse_attention_nondet_train_reference import (
    make_inputs,
)

PACKAGE_DIR = Path(cake_backend.__file__).resolve().parent
SEED = 20261108
REPEATS = 5


def _bits(t):
    view = {2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
    return t.detach().contiguous().view(view)


def _same_bits(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _require_backward(program=cake_jit.PROGRAM):
    if not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)
        in cake_backend.SUPPORTED_COMPUTE_CAPABILITIES
    ):
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    if not cake_backend.generated_program_available(
        torch.device("cuda"), backward=True, program=program
    ):
        pytest.skip("backward stages not registered for this device")


# ---------------------------------------------------------------------------
# Facade (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [(None, False), ("", False), ("0", False), ("1", True), ("x", True)],
)
def test_check_dst_map_environment(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(public_api.CHECK_DST_MAP_ENV, raising=False)
    else:
        monkeypatch.setenv(public_api.CHECK_DST_MAP_ENV, value)
    assert public_api._check_dst_map_from_environment() is expected


@pytest.mark.parametrize("value", ["0", "-3", "many"])
def test_binding_cache_capacity_environment_rejects_bad_values(value):
    code = "import flashinfer.cake_dsa_sparse_attention_det_train"
    env = {
        **__import__("os").environ,
        public_api.BINDING_CACHE_CAPACITY_ENV: value,
    }
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert public_api.BINDING_CACHE_CAPACITY_ENV in result.stderr


def test_binding_cache_is_this_packages_own():
    nondet = importlib.import_module(
        "flashinfer.experimental.cake.cake_dsa_sparse_attention_nondet_train.cake_backend"
    )
    assert cake_backend.BINDING_CACHE is not nondet.BINDING_CACHE


def test_readme_states_every_output():
    """The README's table names every output of the package (A7: the per-output statements)."""
    text = (PACKAGE_DIR / "README.md").read_text()
    for name in (
        "`out`, `lse`, `o_lo`",
        "derived `topk_length`",
        "`dq_latent`, `dq_rope`",
        "`dkv_latent`, `dk_rope`",
        "NoPE `dkv`",
        "`dkv_acc` columns `0:576` (NoPE `0:512`)",
        "varlen global rows and lengths",
    ):
        assert name in text, name
    assert "repeats bit for bit" in text
    assert "use_deterministic_algorithms(True)" in text


# ---------------------------------------------------------------------------
# Public entries (GPU)
# ---------------------------------------------------------------------------


def _entry_call(entry, inp, mode, dkv_map):
    """One autograd step of a public entry: the returned gradients, and ``dkv_acc`` when the mode accumulates."""
    nope = "nope" in entry
    leaves = [
        t.detach().clone().requires_grad_(True)
        for t in (
            (inp.q_latent, inp.kv_latent)
            if nope
            else (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
        )
    ]
    cols = 512 if nope else 576
    kw = {}
    acc = None
    if mode != "fresh":
        rows = int(dkv_map.shape[0]) // 2 + 1 if mode == "acc_map" else inp.total_k
        gen = torch.Generator(device="cuda").manual_seed(SEED)
        acc = torch.randn(rows, cols, device="cuda", generator=gen)
        acc[0, :4] = -0.0
        kw["dkv_acc"] = acc
        if mode == "acc_map":
            kw["dkv_dst_map"] = dkv_map
    fn = getattr(public_api, entry)
    if entry.endswith("varlen"):
        out = fn(
            *leaves,
            inp.idx_local,
            inp.cu_seqlens_q,
            inp.cu_seqlens_k,
            **kw,
        )
    else:
        out = fn(*leaves, inp.idx_global, **kw)
    out.backward(inp.dout)
    return [out] + [leaf.grad for leaf in leaves] + [acc]


ENTRIES = (
    "cake_dsa_sparse_attention_det",
    "cake_dsa_sparse_attention_det_varlen",
    "cake_dsa_sparse_attention_det_nope",
    "cake_dsa_sparse_attention_det_nope_varlen",
)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("mode", ["fresh", "acc", "acc_map"])
@pytest.mark.parametrize("side", [False, True])
def test_public_entries_repeat_bit_for_bit(entry, mode, side):
    """Five autograd steps of each public entry give the same bits for ``out`` and every gradient -- fresh, added
    into a nonzero ``dkv_acc`` (a planted ``-0.0`` included), and through a map with repeated destination rows --
    with or without a kernel resident on a side stream while the step runs."""
    _require_backward(cake_jit.NOPE_PROGRAM if "nope" in entry else cake_jit.PROGRAM)
    inp = make_inputs([200, 120], [600, 360], seed=SEED, topk=256)
    S = inp.total_k
    dkv_map = (
        torch.arange(S, device="cuda", dtype=torch.int32) % (S // 2)
    ).contiguous()
    side_stream = torch.cuda.Stream()
    first = None
    for _ in range(REPEATS):
        if side:
            with torch.cuda.stream(side_stream):
                torch.cuda._sleep(20_000_000)
        got = _entry_call(entry, inp, mode, dkv_map)
        torch.cuda.synchronize()
        if first is None:
            first = got
            continue
        for a, b in zip(got, first, strict=True):
            assert _same_bits(a, b)
