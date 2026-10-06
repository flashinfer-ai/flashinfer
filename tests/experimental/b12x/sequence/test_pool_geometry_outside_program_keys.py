"""Compiled programs of the PLE state layer and GDN decode ignore the state pool's size.

The state slot count sizes the caller's pool and differs between starts
whenever the available memory differs. The kernels receive it as a runtime
scalar, so two declarations that differ only in ``max_state_slots`` plan the
same program key set and a cached start recompiles nothing when the pool grows.

Planning runs in a child process configured like a compiler worker: it
describes every program a family would compile without a GPU, and the worker
configuration replaces this process's CUDA runtime, so it must not run here.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from b12x.preparation import DeviceIdentity
from b12x.sequence.gdn_decode._tuning import GdnQuery, TUNING as GDN_DECODE
from b12x.sequence.ple._tuning import PleQuery, TUNING as PLE

IDENTITY = DeviceIdentity("nvidia", (12, 0), 170, "NVIDIA GeForce RTX 5090")
SLOT_COUNTS = (7, 26_130)

_CHILD = """
import json, multiprocessing, sys
from dataclasses import replace
from types import SimpleNamespace
from b12x._lib import compile_pool
compile_pool._initialize_worker(
    0, (12, 0), "48d28d14-08f3-f3d1-cb99-20c3fa5eca41", "NVIDIA GeForce RTX 5090",
    170, 232448, 232448, multiprocessing.get_context("spawn").Array("q", (0, 0)),
)
import torch
from b12x._lib.compile_pool import describe_compilation
from b12x.preparation import FrozenMapping
from b12x.sequence import gdn_decode, ple

def plan_for(family, query):
    device = torch.device("cuda", 0)
    if family == "ple":
        from b12x.sequence.ple._preparation import make_plan
        caps = ple.Caps(
            device=device, mode=query["mode"], max_tokens=query["max_tokens"],
            max_seqs=query["max_seqs"], max_state_slots=query["max_state_slots"],
            max_speculative_tokens=query["max_speculative_tokens"], streams=query["streams"],
            hidden_size=query["hidden_size"], kernel_size=query["kernel_size"],
            dilation=query["dilation"],
        )
        return make_plan(caps, invocation={}, override=None)
    from b12x.sequence.gdn_decode._preparation import plan
    caps = gdn_decode.Caps(
        device=device, max_tokens=query["max_tokens"], max_seqs=query["max_seqs"],
        max_state_slots=query["max_state_slots"], key_heads=query["key_heads"],
        value_heads=query["value_heads"], state_index_columns=query["state_index_columns"],
        state_dtype=getattr(torch, query["state_dtype"]), gate_activation=query["gate_activation"],
        qk_l2norm=query["qk_l2norm"], null_state_index=query["null_state_index"],
    )
    return plan(caps, invocation=FrozenMapping({"dt_bias_dtype": query["dt_bias_dtype"]}))

results = []
for family, query, config in json.load(sys.stdin):
    plan = plan_for(family, query)
    config = plan.contract.decode_config(FrozenMapping(config))
    programs = set()
    for job in plan._compile_jobs(config, SimpleNamespace(ordinal=0)):
        programs.update(
            (program.dialect, program.key, program.name)
            for program in describe_compilation(job).programs
        )
    results.append(sorted(programs))
json.dump(results, sys.stdout)
"""


def _program_keys(requests):
    """Return the planned program key set of each (family, query, config) request."""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    payload = json.dumps([
        (family, query, config) for family, query, config in requests
    ])
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD], input=payload, capture_output=True, text=True,
        env=environment, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return [set(map(tuple, programs)) for programs in json.loads(completed.stdout)]


def _ple_query(mode):
    return PleQuery(
        mode=mode, dtype="bfloat16", max_tokens=4, max_seqs=2, max_speculative_tokens=3,
        streams=4, hidden_size=2560, kernel_size=4, dilation=3, max_state_slots=SLOT_COUNTS[0],
    )


def _gdn_query(recipe):
    if recipe == "qwen":
        return GdnQuery(
            gate_activation="sigmoid", qk_l2norm=True, state_dtype="float32", key_heads=8,
            value_heads=24, max_seqs=1, max_tokens=4, state_index_columns=4,
            max_state_slots=SLOT_COUNTS[0], dt_bias_dtype="bfloat16",
        )
    if recipe == "qwen-hashed":
        # More than 32 state index cells selects the hash-table validators.
        return GdnQuery(
            gate_activation="sigmoid", qk_l2norm=True, state_dtype="float32", key_heads=8,
            value_heads=24, max_seqs=16, max_tokens=64, state_index_columns=4,
            max_state_slots=SLOT_COUNTS[0], dt_bias_dtype="bfloat16",
        )
    return GdnQuery(
        gate_activation="sigmoid", qk_l2norm=True, state_dtype="float32", key_heads=8,
        value_heads=8, max_seqs=4, max_tokens=16, state_index_columns=4,
        max_state_slots=SLOT_COUNTS[0], null_state_index=0, dt_bias_dtype="float32",
    )


def _ple_payload(query):
    return {name: getattr(query, name) for name in (
        "mode", "max_tokens", "max_seqs", "max_state_slots", "max_speculative_tokens",
        "streams", "hidden_size", "kernel_size", "dilation",
    )}


def _gdn_payload(query):
    return {name: getattr(query, name) for name in (
        "max_tokens", "max_seqs", "max_state_slots", "key_heads", "value_heads",
        "state_index_columns", "state_dtype", "gate_activation", "qk_l2norm",
        "null_state_index", "dt_bias_dtype",
    )}


def _assert_pool_independent(family, payload, query, config, expected_count):
    """Slot count changes nothing; a sequence capacity change still changes the programs."""
    larger = replace(query, max_state_slots=SLOT_COUNTS[1])
    wider = replace(query, max_seqs=query.max_seqs + 1, max_tokens=query.max_tokens + 1)
    small, large, wide = _program_keys([
        (family, payload(query), config),
        (family, payload(larger), config),
        (family, payload(wider), config),
    ])
    assert len(small) == expected_count
    assert small == large
    assert small != wide


@pytest.mark.parametrize("mode", ("decode", "prefill", "mixed"))
def test_ple_programs_ignore_the_state_slot_count(mode):
    # The PLE contract is fixed: its only configuration is the Triton backend.
    query = _ple_query(mode)
    config = PLE.encode_config(PLE.default_config(query, IDENTITY))
    assert config == {"backend": "triton"}
    _assert_pool_independent("ple", _ple_payload, query, config, 7)


@pytest.mark.parametrize("recurrent_block_v", (32, 16))
@pytest.mark.parametrize("recipe, expected_count", (("qwen", 3), ("qwen-hashed", 5), ("kda", 5)))
def test_gdn_decode_programs_ignore_the_state_slot_count(recipe, expected_count, recurrent_block_v):
    query = _gdn_query(recipe)
    default = GDN_DECODE.default_config(query, IDENTITY)
    assert default.recurrent_block_v == 32
    config = GDN_DECODE.encode_config(replace(default, recurrent_block_v=recurrent_block_v))
    _assert_pool_independent("gdn_decode", _gdn_payload, query, config, expected_count)
