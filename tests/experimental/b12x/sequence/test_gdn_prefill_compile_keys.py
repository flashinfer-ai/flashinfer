"""Delta-prefill programs are keyed without the state pool's slot count.

The slot count sizes the caller's pool and differs between starts whenever
the available memory differs. No kernel of the family consumes it, so two
declarations that differ only in ``max_state_slots`` plan the same program
keys for every configuration and a cached start recompiles nothing.
The planning runs in a subprocess because the offline compile target shims
the process-global CUDA runtime the way a compiler worker does.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("cutlass")

PROBE = """
import json, os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import multiprocessing
from dataclasses import replace

from b12x._lib import compile_pool
compile_pool._initialize_worker(
    0, (12, 0), "synthetic-uuid", "synthetic SM120", 148, 232448, 232448,
    multiprocessing.get_context("spawn").Array("q", (0, 0)),
)
from b12x._lib.compile_pool import CompileJob, describe_compilation
from b12x.preparation import DeviceIdentity

IDENTITY = DeviceIdentity("nvidia", (12, 0), 148, "synthetic SM120")
COMPONENT = {component!r}
if COMPONENT == "gdn_prefill":
    from b12x.sequence.gdn_prefill._tuning import TUNING, GdnPrefillQuery as Query, tiles_capacity
    query = Query(
        key_heads=8, value_heads=24, head_dim=128, model_dtype="bfloat16", state_dtype="float32",
        qk_l2norm=True, checkpoint_export=True, max_tokens=1024, max_seqs=4, max_state_slots=7,
        null_state_index=0, dt_bias_dtype="bfloat16",
    )
else:
    from b12x.sequence.kda_prefill._tuning import TUNING, KdaPrefillQuery as Query
    query = Query(
        heads=6, head_dim=128, model_dtype="bfloat16", state_dtype="float32", qk_l2norm=True,
        checkpoint_export=True, max_tokens=1024, max_seqs=4, max_state_slots=7, null_state_index=0,
    )
default = TUNING.default_config(query, IDENTITY)
configs = {{
    "default": default,
    "v32_k2_s2": replace(default, v_split=32, k_split=2, stages=2),
    "w8_s2": replace(default, window_tiles=8, stages=2),
}}
if COMPONENT == "gdn_prefill":
    segments = (query.max_tokens + 255) // 256 + query.max_seqs
    configs["chunk_parallel"] = replace(
        default, algorithm="chunk_parallel", segment_tokens=256, stages=2,
        window_tiles=tiles_capacity(query.max_tokens, segments),
    )


def programs(declared, config):
    TUNING.validate_config(declared, config, IDENTITY)
    job = CompileJob.create(
        "b12x.sequence._shared.delta_prefill.preparation:compile_prefill",
        COMPONENT, declared.to_dict(), TUNING.encode_config(config), 0,
    )
    return sorted([p.dialect, p.name, p.key] for p in describe_compilation(job).programs)


report = {{}}
for label, config in configs.items():
    report[label] = {{
        str(slots): programs(replace(query, max_state_slots=slots), config)
        for slots in (7, 26_130)
    }}
print(json.dumps(report))
"""


def _plan_programs(component: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", PROBE.format(component=component)],
        capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, f"probe failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("component", ("gdn_prefill", "kda_prefill"))
def test_program_keys_ignore_the_state_slot_count(component):
    report = _plan_programs(component)
    assert len(report) >= 3
    for label, by_slots in report.items():
        small, large = by_slots["7"], by_slots["26130"]
        assert small, label
        assert small == large, (label, small, large)
        assert {name for _, name, _ in small} >= {
            "sequence.delta_prefill.prologue",
            "sequence.delta_prefill.prepare",
            "sequence.delta_prefill.recurrence",
        }


def test_kda_prefill_selection_key_ignores_the_state_slot_count():
    from dataclasses import replace

    from b12x.preparation import DeviceIdentity
    from b12x.sequence.kda_prefill._tuning import TUNING, KdaPrefillQuery

    identity = DeviceIdentity("nvidia", (12, 0), 148, "synthetic SM120")
    query = KdaPrefillQuery(
        heads=6, head_dim=128, model_dtype="bfloat16", state_dtype="float32", qk_l2norm=True,
        checkpoint_export=True, max_tokens=64, max_seqs=2, max_state_slots=7, null_state_index=0,
    )

    def encoded(declared):
        return TUNING.configure(declared, device=identity, override=None).encoded_query

    assert encoded(query) == encoded(replace(query, max_state_slots=26_130))
    assert "max_state_slots" not in TUNING.query_fields
    assert query.to_dict()["max_state_slots"] == 7
