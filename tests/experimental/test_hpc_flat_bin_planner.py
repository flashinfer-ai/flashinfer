"""Compare actual original GPU records and FI adapters against unmodified HPC CPU oracle."""

import json
import random
import subprocess
from pathlib import Path
import pytest
import torch
from flashinfer.hpc_decode import create_hpc_flat_bin_decode_wrapper


def assert_adapted_map(wrapper, lengths, heads):
    mapped, raw = wrapper.schedule(), wrapper.original_schedule()
    assert raw[0] - 1 == mapped["capacity_tiles"]
    original_counts = raw[12 * (1 + raw[0] * wrapper.pool) :]
    ptr = mapped["state_indptr"]
    for b in range(len(lengths)):
        for h in range(heads):
            pair = b * heads + h
            assert ptr[pair + 1] - ptr[pair] == original_counts[h * len(lengths) + b]
            intervals = sorted(
                (t[2], t[3]) for t in mapped["tasks"] if t[0] == b and t[1] == h
            )
            assert intervals[0][0] == 0 and intervals[-1][1] == lengths[b]
            assert all(
                a[1] == c[0] for a, c in zip(intervals, intervals[1:], strict=False)
            )
    assert sorted(t[4] for t in mapped["tasks"]) == list(range(ptr[-1]))
    for cta in range(wrapper.pool):
        lo, hi = mapped["cta_indptr"][cta : cta + 2]
        for slot, (b, h, start, stop, state) in enumerate(mapped["tasks"][lo:hi]):
            rec = raw[12 * (1 + cta * raw[0] + slot) :]
            assert rec[:5] == [h, b, state - ptr[b * heads + h], start, stop - start]
    pad12 = lambda n: (n + 11) // 12 * 12
    flags = 12 * (1 + raw[0] * wrapper.pool) + pad12(len(lengths) * heads)
    assert raw[flags : flags + pad12(wrapper.pool)] == [0] * pad12(wrapper.pool)


def test_replan_reuses_workspace_with_changing_layout():
    wrapper = create_hpc_flat_bin_decode_wrapper(min_process_len=128)
    for repeat in range(5):
        for lengths in (
            [65537, 1, 129],
            [1, 65, 127],
            [512] * 17,
            [128, 129, 1025],
            [8193, 1, 32769],
        ):
            H = 1 if repeat % 2 else 8
            pages = [(n + 63) // 64 for n in lengths]
            ptr = torch.tensor(
                [0] + torch.tensor(pages).cumsum(0).tolist(), dtype=torch.int32
            )
            ids = torch.zeros(int(ptr[-1]), device="cuda", dtype=torch.int32)
            tail = torch.tensor(
                [(n - 1) % 64 + 1 for n in lengths], device="cuda", dtype=torch.int32
            )
            wrapper.plan(ptr, ids, tail, 4 * H, H, 128, 64)
            assert_adapted_map(wrapper, lengths, H)


def test_parallel_planner_bounds():
    with pytest.raises(ValueError, match="128"):
        create_hpc_flat_bin_decode_wrapper(min_process_len=64)
    wrapper = create_hpc_flat_bin_decode_wrapper()
    with pytest.raises(RuntimeError, match="plan"):
        wrapper.assign_gpu()
    with pytest.raises(ValueError, match="KV length"):
        wrapper.plan(
            torch.tensor([0, 2**25], dtype=torch.int32), None, None, 4, 1, 128, 64
        )
    with pytest.raises(ValueError, match="Partial-state"):
        wrapper.plan(
            torch.tensor([0, 1], dtype=torch.int32),
            None,
            None,
            4 * 2**22,
            2**22,
            128,
            64,
        )
    wrapper.pool = wrapper.max_pool + 1
    with pytest.raises(ValueError, match="residency"):
        wrapper.plan(torch.tensor([0, 1], dtype=torch.int32), None, None, 4, 1, 128, 64)
    wrapper.pool = 1
    with pytest.raises(ValueError, match="2048"):
        wrapper.plan(torch.arange(2050, dtype=torch.int32), None, None, 4, 1, 128, 64)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="Requires SM100",
)


def test_original_parallel_map_matches_cpu(tmp_path):
    executable = tmp_path / "hpc_cpu_oracle"
    subprocess.run(
        [
            "g++",
            "-O2",
            "-std=c++17",
            str(Path(__file__).with_name("hpc_flat_bin_cpu_oracle.cpp")),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=60,
    )
    rng = random.Random(20261013)
    sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    max_pool = create_hpc_flat_bin_decode_wrapper().max_pool
    pools = sorted({min(p, max_pool) for p in (1, 4, sms, 2 * sms, 4 * sms)})
    cases = [([512, 512, 512, 6656], 1, 4, 512)]
    for lengths in (
        [1],
        [63, 64, 65],
        [127, 128, 129],
        [65536],
        [512] * 295 + [65536],
        [1] * 2048,
    ):
        for heads in (1, 8):
            cases.append((lengths, heads, sms, 512))
    for _ in range(100):
        B = rng.choice([1, 2, 3, 17, 31, 64])
        lengths = [rng.randint(1, 8193) for _ in range(B)]
        cases.append(
            (
                lengths,
                rng.choice([1, 2, 8, 16]),
                rng.choice(pools),
                rng.choice([128, 256, 512, 1024]),
            )
        )

    payload = "".join(
        f"{len(ns)} {h} {p} {m}\n" + " ".join(map(str, ns)) + "\n"
        for ns, h, p, m in cases
    )
    oracle = subprocess.run(
        [str(executable)], input=payload, text=True, capture_output=True, check=True
    )
    expected = [json.loads(line) for line in oracle.stdout.splitlines()]
    assert len(expected) == len(cases)
    checked_tasks = 0
    for case_id, ((ns, H, pool, minimum), ref) in enumerate(
        zip(cases, expected, strict=True)
    ):
        wrapper = create_hpc_flat_bin_decode_wrapper(min_process_len=minimum)
        # Override pool only for a diagnostic of the rule, before any allocations.
        wrapper.pool = pool
        pages = [(n + 63) // 64 for n in ns]
        ptr = torch.tensor(
            [0] + torch.tensor(pages).cumsum(0).tolist(), dtype=torch.int32
        )
        indices = torch.zeros(int(ptr[-1]), device="cuda", dtype=torch.int32)
        tail = torch.tensor(
            [(n - 1) % 64 + 1 for n in ns], device="cuda", dtype=torch.int32
        )
        wrapper.plan(ptr, indices, tail, H * 4, H, 128, 64)
        actual = wrapper.schedule()
        raw = wrapper.original_schedule()
        records = []
        for cta in range(pool):
            for slot in range(raw[0]):
                t = raw[12 * (1 + cta * raw[0] + slot) : 12 * (2 + cta * raw[0] + slot)]
                if t[0] < 0:
                    break
                records.append(t[:9])
        assert records == ref["records"], (
            f"case {case_id}: original 9-field map mismatch"
        )
        normalized = []
        for cta in range(pool):
            begin, end = actual["cta_indptr"][cta : cta + 2]
            for b, h, start, stop, state in actual["tasks"][begin:end]:
                chunk = state - actual["state_indptr"][b * H + h]
                normalized.append([cta, b, h, start, stop, chunk])
        assert actual["capacity_tiles"] == ref["capacity"], case_id
        assert normalized == ref["tasks"], (
            f"case {case_id}: GPU policy differs from original CPU source"
        )
        checked_tasks += len(normalized)
        if case_id == 0:
            print("Toy example tasks:", normalized)
    print(
        json.dumps(
            {
                "cases_passed": len(cases),
                "tasks_compared": checked_tasks,
                "compared_fields": [
                    "CTA",
                    "batch",
                    "KV head",
                    "token begin",
                    "token end",
                    "chunk",
                ],
                "scope": "positive KV lengths, one Q token, tile128; original C++ CPU function vs parallel GPU map; all 9 fields",
            }
        )
    )
