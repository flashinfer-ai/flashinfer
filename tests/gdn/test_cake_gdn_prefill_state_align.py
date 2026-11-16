# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Cake GDN non-CP prefill on element-aligned state that is not 16-byte aligned.

The non-CP route accepts any ``initial_state`` / ``output_state`` whose rows are dense
``[H, 128, 128]`` (strides ``(stride(0), 16384, 128, 1)``) and any contiguous
``state_checkpoints``.  A data pointer or a slot stride ``stride(0)`` that is a whole
number of elements but not of 16 bytes (a sliced state tensor, a padded or offset state
pool) is legal.  The kernels move state with vector accesses where the tile's absolute
state address is aligned to the width of those accesses (2 to 16 bytes) and with scalar
accesses elsewhere; every stored value is the same.

Every prefill record compiles in two flavors (``CAKE_GDN_STATE_VEC_ALIGNED=1|0``): the
host runs the unguarded body on 16-byte-aligned state and the guarded body otherwise.
Each case compares runs on such views bit for bit with the run on 16-byte-aligned buffers
holding the same data: token output, final state, checkpoints, the unchanged initial state
and the unused slots of an indexed pool.  Every state buffer sits inside a guard region
filled with a byte pattern; no byte outside the logical state may change.

A misaligned vector access raises a sticky CUDA error ("misaligned address") that poisons
the whole process, so every case runs in a child process (this file run as a script).  The
child reports each run ("arm") before it launches it; the parent names the arm that did
not finish and fails cleanly.

* ``test_public_cake_gdn_prefill_state_off_16_bytes_matches_aligned`` drives the public
  ``chunk_gated_delta_rule(backend="cake_gdn", use_cp=False)`` route per (contract,
  layout): FP32 / BF16 / FP16 / FP8 e4m3 / FP8 e5m2 state, DV-split and full-DV,
  FP16 I/O, checkpoints, indexed in-place pools, padded slot strides and zero-length
  sequences, at element offsets 1, 8 bytes and 16 bytes minus one element.  The aligned
  call must load the aligned flavor and every other call the guarded flavor.
* ``test_cake_gdn_prefill_unit_records_state_off_16_bytes_match_aligned`` sweeps the
  manifest: for each prefill CUDA unit, every stateful record built for the live
  architecture is launched through its raw TVM-FFI entry with inputs derived from the
  record's specializations and host binding: the aligned flavor on 16-byte-aligned
  buffers (the reference), the guarded flavor on the same buffers and on every
  misaligned layout.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def _register_repo_tests_package():
    """Make ``tests`` this checkout's test package in a child run as a script.

    The child keeps the parent's ``sys.path`` (``PYTHONPATH``) so ``flashinfer`` resolves
    to the same installation; only the ``tests`` package is bound to this checkout.
    """

    if "tests" in sys.modules:
        return
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "tests", root / "__init__.py", submodule_search_locations=[str(root)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["tests"] = module
    spec.loader.exec_module(module)


if __name__ == "__main__":
    _register_repo_tests_package()

import pytest  # noqa: E402
import torch  # noqa: E402

import flashinfer  # noqa: E402
from flashinfer.jit import cake_gdn  # noqa: E402
from tests.gdn.test_cake_gdn_prefill_gpu import _launch, _make_case  # noqa: E402

_CHILD_FLAG = "--cake-gdn-state-align"
_EVENT_PREFIX = "CAKE_GDN_STATE_ALIGN "
# A public case runs every offset of one layout in one child; a unit case compiles and runs
# every stateful record of one CUDA unit (cold nvcc builds dominate).
_PUBLIC_CHILD_TIMEOUT_S = 1200
_UNIT_CHILD_TIMEOUT_S = 5400
_GUARD_BYTES = 256
_GUARD_FILL = 0xA5
_BITS_DTYPE = {1: torch.int8, 2: torch.int16, 4: torch.int32}


@pytest.fixture(autouse=True)
def _require_cake_gdn_arch():
    if not torch.cuda.is_available():
        pytest.skip("Cake GDN requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    try:
        cake_gdn.arch_for_compute_capability(major, minor)
    except cake_gdn.CakeGDNUnsupportedError as error:
        pytest.skip(str(error))


def _live_arch():
    major, minor = torch.cuda.get_device_capability()
    return cake_gdn.arch_for_compute_capability(major, minor)


def _emit(event, **fields):
    print(
        _EVENT_PREFIX + json.dumps(dict(fields, event=event), sort_keys=True),
        flush=True,
    )


# ---------------------------------------------------------------------------------------
# Guarded state buffers


class _GuardedState:
    """A ``[slots, heads, 128, 128]`` state view inside a byte buffer with guard bytes.

    The view starts ``elem_offset`` elements past a 16-byte boundary and its slots are
    ``heads * 16384 + pad`` elements apart; everything outside the logical elements
    (leading and trailing guards, the padding between slots) holds ``_GUARD_FILL``.
    """

    def __init__(self, dtype, slots, heads, elem_offset=0, pad=0):
        esize = dtype.itemsize
        row = heads * 16384
        slot_stride = row + pad
        span = (slots - 1) * slot_stride + row
        nbytes = 2 * _GUARD_BYTES + (elem_offset + span) * esize
        nbytes = -(-nbytes // 16) * 16
        self.raw = torch.full((nbytes,), _GUARD_FILL, dtype=torch.uint8, device="cuda")
        assert self.raw.data_ptr() % 16 == 0
        start = _GUARD_BYTES // esize + elem_offset
        size = (slots, heads, 128, 128)
        stride = (slot_stride, 16384, 128, 1)
        self.view = self.raw.view(dtype).as_strided(size, stride, start)
        self.bits_view = self.raw.view(_BITS_DTYPE[esize]).as_strided(
            size, stride, start
        )
        probe = torch.zeros_like(self.raw)
        probe.view(_BITS_DTYPE[esize]).as_strided(size, stride, start).fill_(-1)
        self.footprint = probe.ne(0)
        aligned = (elem_offset * esize) % 16 == 0
        assert (self.view.data_ptr() % 16 == 0) == aligned, (elem_offset, esize)

    def bits(self):
        return self.bits_view.clone()

    def guard_violations(self):
        return int(self.raw[~self.footprint].ne(_GUARD_FILL).sum())


def _offsets(lanes):
    """Element offsets off a 16-byte boundary: one element, 8 bytes, 16 bytes less one element."""

    return sorted({1, lanes // 2, lanes - 1})


def _placement(layout, offset, lanes):
    """(element offset, slot padding) per state buffer for one layout at one offset."""

    full = {
        "initial": (offset, offset),
        "output": (lanes - offset, offset),
        "checkpoints": lanes - offset,
    }
    return {
        "initial": {"initial": (offset, 0)},
        "output": {"output": (offset, 0)},
        "padded_stride": {"initial": (0, offset), "output": (0, lanes - offset)},
        "checkpoints": {"checkpoints": offset},
        "all": full,
        "empty_seqs": full,
    }[layout]


def _plan(case, *, has_initial, has_final, in_place):
    """What one launch of ``case`` reads and writes (shared by the public and raw runs)."""

    heads = max(case["num_q_heads"], case["num_v_heads"])
    initial_src = case["initial_state"].clone() if has_initial else None
    state_dtype = case["state_dtype"]
    esize = state_dtype.itemsize
    return {
        "heads": heads,
        "state_dtype": state_dtype,
        "lanes": 16 // esize,
        "seq_lens": tuple(case["seq_lens"]),
        "state_slots": (
            int(case["initial_state"].shape[0]) if in_place else len(case["seq_lens"])
        ),
        "state_indices": (
            None
            if case["state_indices"] is None
            else [int(slot) for slot in case["state_indices"].tolist()]
        ),
        "has_initial": has_initial,
        "has_final": has_final,
        "in_place": in_place,
        "initial_src": initial_src,
        "initial_bits": (
            None
            if initial_src is None
            else initial_src.view(_BITS_DTYPE[esize]).clone()
        ),
        "num_checkpoints": (
            int(case["state_checkpoints"].shape[0]) if case["checkpoint_every"] else 0
        ),
        "tokens_shape": tuple(case["output"].shape),
        "io_dtype": case["output"].dtype,
    }


def _alloc_arm(plan, placement):
    dtype, heads, slots = plan["state_dtype"], plan["heads"], plan["state_slots"]
    buffers = {}
    if plan["has_initial"]:
        offset, pad = placement.get("initial", (0, 0))
        initial = _GuardedState(dtype, slots, heads, offset, pad)
        initial.view.copy_(plan["initial_src"])
        buffers["initial"] = initial
    if plan["has_final"]:
        if plan["in_place"]:
            buffers["output"] = buffers["initial"]
        else:
            offset, pad = placement.get("output", (0, 0))
            output = _GuardedState(dtype, slots, heads, offset, pad)
            output.bits_view.fill_(-1)
            buffers["output"] = output
    if plan["num_checkpoints"]:
        checkpoints = _GuardedState(
            dtype, plan["num_checkpoints"], heads, placement.get("checkpoints", 0), 0
        )
        checkpoints.bits_view.fill_(-1)
        buffers["checkpoints"] = checkpoints
    tokens = torch.empty(plan["tokens_shape"], dtype=plan["io_dtype"], device="cuda")
    tokens.view(torch.int16).fill_(-1)
    return buffers, tokens


def _difference(what, got, want):
    if torch.equal(got, want):
        return None
    differs = got.ne(want)
    return {
        "what": what,
        "count": int(differs.sum()),
        "first": differs.nonzero()[0].tolist(),
    }


def _check_arm(plan, buffers, tokens, reference):
    """Snapshot of one finished run and its problems (``reference`` is the aligned run)."""

    problems = []
    for name, buffer in buffers.items():
        if name == "output" and plan["in_place"]:
            continue
        violations = buffer.guard_violations()
        if violations:
            problems.append(
                {
                    "what": f"{violations} guard bytes outside the logical {name} state changed",
                    "count": violations,
                }
            )
    snapshot = {"output": tokens.view(torch.int16).clone()}
    if "output" in buffers:
        snapshot["output_state"] = buffers["output"].bits()
    if "checkpoints" in buffers:
        snapshot["state_checkpoints"] = buffers["checkpoints"].bits()
    initial_bits = plan["initial_bits"]
    if plan["has_initial"] and not plan["in_place"]:
        problems.append(
            _difference(
                "initial_state was written", buffers["initial"].bits(), initial_bits
            )
        )
    indices = plan["state_indices"]
    if plan["in_place"] and indices is not None:
        for slot in sorted(set(range(plan["state_slots"])) - set(indices)):
            problems.append(
                _difference(
                    f"unused pool slot {slot} was written",
                    snapshot["output_state"][slot],
                    initial_bits[slot],
                )
            )
    if reference is None:
        # The 16-byte-aligned run itself: every token row written, every final state
        # written, and a zero-length sequence's slot holds its initial (or zero) state or
        # is left untouched.
        if not torch.isfinite(tokens.float()).all():
            problems.append({"what": "aligned run left non-finite token output"})
        if "output_state" in snapshot:
            state = buffers["output"].view
            for seq, length in enumerate(plan["seq_lens"]):
                slot = seq if indices is None else indices[seq]
                if length:
                    if not torch.isfinite(state[slot].float()).all():
                        problems.append(
                            {
                                "what": f"aligned run left a non-finite final state in slot {slot}"
                            }
                        )
                    continue
                got = snapshot["output_state"][slot]
                kept = (
                    initial_bits[slot]
                    if initial_bits is not None
                    else torch.zeros_like(got)
                )
                if not (torch.equal(got, kept) or bool(got.eq(-1).all())):
                    problems.append(
                        {
                            "what": f"zero-length sequence {seq} left slot {slot} neither "
                            "copied from its initial state nor untouched"
                        }
                    )
    else:
        for key, value in snapshot.items():
            problems.append(
                _difference(
                    f"{key} differs from the 16-byte-aligned run", value, reference[key]
                )
            )
    return snapshot, [problem for problem in problems if problem is not None]


def _run_arms(plan, arms, launch, tag):
    """Run each (name, placement) arm in order; the first is the 16-byte-aligned reference.

    ``launch(buffers, tokens, placement)`` runs one call; a raw launch also reads its grid
    from the placement.
    """

    reference = None
    failures = 0
    for arm, placement in arms:
        _emit("arm", arm=arm, placement=placement, **tag)
        buffers, tokens = _alloc_arm(plan, placement)
        launch(buffers, tokens, placement)
        torch.cuda.synchronize()
        snapshot, problems = _check_arm(plan, buffers, tokens, reference)
        for problem in problems:
            _emit("mismatch", arm=arm, **tag, **problem)
        failures += len(problems)
        _emit("arm_done", arm=arm, **tag)
        if reference is None:
            reference = snapshot
        del buffers, tokens
    return failures


# ---------------------------------------------------------------------------------------
# Child process protocol


def _child_env():
    env = dict(os.environ)
    paths = [path or os.getcwd() for path in sys.path]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    return env


def _text(output):
    if output is None:
        return ""
    return output.decode(errors="replace") if isinstance(output, bytes) else output


def _events(stdout):
    return [
        json.loads(line[len(_EVENT_PREFIX) :])
        for line in stdout.splitlines()
        if line.startswith(_EVENT_PREFIX)
    ]


def _run_isolated(spec, *, timeout):
    """Run one case in a child process; fail with the arm that faulted or mismatched."""

    spec = dict(spec, flashinfer_file=str(Path(flashinfer.__file__).resolve()))
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        _CHILD_FLAG,
        json.dumps(spec),
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, env=_child_env()
        )
        stdout, stderr, status = (
            completed.stdout,
            completed.stderr,
            completed.returncode,
        )
    except subprocess.TimeoutExpired as expired:
        stdout, stderr, status = _text(expired.stdout), _text(expired.stderr), None
    events = _events(stdout)
    mismatches = [event for event in events if event["event"] == "mismatch"]
    done = any(event["event"] == "done" for event in events)
    if status == 0 and done and not mismatches:
        return events
    finished = {
        (event.get("record"), event["arm"])
        for event in events
        if event["event"] == "arm_done"
    }
    unfinished = [
        event
        for event in events
        if event["event"] == "arm"
        and (event.get("record"), event["arm"]) not in finished
    ]
    lines = [
        f"child exited with status {status}"
        if status is not None
        else f"child timed out after {timeout} s",
    ]
    lines += [
        f"{event['event']}: {json.dumps(event)}"
        for event in events
        if event["event"] in ("route", "record")
    ]
    if unfinished:
        lines.append(f"arm that did not finish: {json.dumps(unfinished[-1])}")
    if "misaligned address" in stderr:
        lines.append("the device reported 'misaligned address'")
    lines += [f"mismatch: {json.dumps(event)}" for event in mismatches[:32]]
    if len(mismatches) > 32:
        lines.append(f"... {len(mismatches) - 32} more mismatches")
    lines.append("child stderr (tail):\n" + stderr[-6000:])
    pytest.fail("\n".join(lines), pytrace=False)


# ---------------------------------------------------------------------------------------
# Public route cases

# Per contract: ``_make_case`` arguments, the batch, and the batch with zero-length
# sequences.  Each contract resolves on sm_100a, sm_103a and sm_107a; the
# low-precision states need the DV-split schedule (2 * num_seqs * max(Hq, Hv) within
# the active clusters).
_PUBLIC_CONTRACTS = {
    "fp32_dvsplit": {
        "make": dict(
            state_dtype=torch.float32, num_q_heads=8, num_k_heads=8, num_v_heads=16
        ),
        "seq_lens": (300, 512),
        "empty_seq_lens": (0, 300, 0, 512),
    },
    # several short full-DV tiles per CTA: the gate-stream twins (and the initial-state
    # L2 prefetch twin on sm_107a)
    "fp32_full_dv_short_tiles": {
        "make": dict(
            state_dtype=torch.float32, num_q_heads=4, num_k_heads=4, num_v_heads=16
        ),
        "seq_lens": (128,) * 32,
        "empty_seq_lens": (0, 128) * 16,
    },
    "f16io_fp32_dvsplit": {
        "make": dict(
            io_dtype=torch.float16,
            state_dtype=torch.float32,
            num_q_heads=2,
            num_k_heads=2,
            num_v_heads=4,
        ),
        "seq_lens": (128, 200),
        "empty_seq_lens": (0, 128, 0, 200),
    },
    "bf16_dvsplit": {
        "make": dict(
            state_dtype=torch.bfloat16, num_q_heads=8, num_k_heads=8, num_v_heads=16
        ),
        "seq_lens": (512, 300),
        "empty_seq_lens": (512, 0, 300, 0),
    },
    "bf16_full_dv": {
        "make": dict(
            state_dtype=torch.bfloat16, num_q_heads=4, num_k_heads=4, num_v_heads=16
        ),
        "seq_lens": (512,) * 8,
        "empty_seq_lens": (512, 0) * 4,
    },
    "bf16_full_dv_short_tiles": {
        "make": dict(
            state_dtype=torch.bfloat16, num_q_heads=4, num_k_heads=4, num_v_heads=16
        ),
        "seq_lens": (128,) * 32,
        "empty_seq_lens": (0, 128) * 16,
    },
    **{
        f"{name}_dvsplit": {
            "make": dict(
                state_dtype=dtype, num_q_heads=16, num_k_heads=16, num_v_heads=32
            ),
            "seq_lens": (300, 129),
            "empty_seq_lens": (0, 300),
        }
        for name, dtype in (
            ("f16", torch.float16),
            ("e4m3", torch.float8_e4m3fn),
            ("e5m2", torch.float8_e5m2),
        )
    },
    # FP16 I/O, FP32 state without an initial state, checkpoints every 64 tokens
    "f16io_fp32_checkpoints": {
        "make": dict(
            io_dtype=torch.float16,
            state_dtype=torch.float32,
            num_q_heads=2,
            num_k_heads=2,
            num_v_heads=4,
            checkpoint_every=64,
        ),
        "seq_lens": (128, 200),
        "empty_seq_lens": (0, 128, 0, 200),
    },
    # Indexed BF16 pools updated in place (``output_state is initial_state``)
    "bf16_indexed_dvsplit": {
        "make": dict(
            state_dtype=torch.bfloat16,
            indexed=True,
            num_q_heads=16,
            num_k_heads=16,
            num_v_heads=16,
            state_pool_padding=0,
        ),
        "seq_lens": (128, 77),
        "empty_seq_lens": (128, 0, 77),
    },
    "bf16_indexed_full_dv": {
        "make": dict(
            state_dtype=torch.bfloat16,
            indexed=True,
            num_q_heads=32,
            num_k_heads=32,
            num_v_heads=32,
            state_pool_padding=0,
        ),
        "seq_lens": (128, 192, 64, 96),
        "empty_seq_lens": (128, 0, 192, 64, 96),
    },
    # The SGLang TP4 indexed checkpoint contract (DV-split)
    "bf16_indexed_checkpoints": {
        "make": dict(
            state_dtype=torch.bfloat16,
            indexed=True,
            checkpoint_every=64,
            num_q_heads=4,
            num_k_heads=4,
            num_v_heads=8,
            state_pool_padding=0,
        ),
        "seq_lens": (52, 93, 15, 107, 72, 61, 21),
        "empty_seq_lens": (52, 0, 93, 15, 107, 72, 61, 21),
    },
}


def _public_layouts(contract):
    make = _PUBLIC_CONTRACTS[contract]["make"]
    if make.get("indexed"):
        layouts = ("initial", "padded_stride", "all", "empty_seqs")
        return layouts + (("checkpoints",) if make.get("checkpoint_every") else ())
    if make.get("checkpoint_every"):
        return ("output", "checkpoints", "all", "empty_seqs")
    return ("initial", "output", "padded_stride", "empty_seqs")


_PUBLIC_CASES = [
    pytest.param(contract, layout, id=f"{contract}-{layout}")
    for contract in _PUBLIC_CONTRACTS
    for layout in _public_layouts(contract)
]


def _child_public(spec):
    contract = _PUBLIC_CONTRACTS[spec["contract"]]
    layout = spec["layout"]
    seq_lens = contract["empty_seq_lens" if layout == "empty_seqs" else "seq_lens"]
    case = _make_case(seq_lens=seq_lens, **contract["make"])
    plan = _plan(
        case,
        has_initial=case["initial_state"] is not None,
        has_final=True,
        in_place=case["state_indices"] is not None,
    )
    load = cake_gdn.load_cake_gdn_kernel
    loaded = set()

    def loading(name, arch, **kw):
        if name not in loaded:
            loaded.add(name)
            unit = cake_gdn._cuda_record(cake_gdn._kernel_record(name), arch)["path"]
            _emit("route", variant=name, unit=unit)
        _emit("flavor", variant=name, aligned=kw.get("state_vec_aligned", True))
        return load(name, arch, **kw)

    # the public launch resolves the record and loads it through this module attribute
    cake_gdn.load_cake_gdn_kernel = loading

    def launch(buffers, tokens, placement):
        output_state = buffers["output"].view
        launched = dict(
            case,
            initial_state=buffers["initial"].view if "initial" in buffers else None,
            output_state=output_state,
            state_checkpoints=(
                buffers["checkpoints"].view
                if "checkpoints" in buffers
                else case["state_checkpoints"]
            ),
            output=tokens,
        )
        out, state = _launch(launched)
        assert out is tokens and state is output_state

    lanes = plan["lanes"]
    arms = [("aligned", {})] + [
        (f"{layout}_off{offset}", _placement(layout, offset, lanes))
        for offset in _offsets(lanes)
    ]
    tag = {"contract": spec["contract"], "layout": layout}
    return _run_arms(plan, arms, launch, tag)


@pytest.mark.parametrize(("contract", "layout"), _PUBLIC_CASES)
def test_public_cake_gdn_prefill_state_off_16_bytes_matches_aligned(contract, layout):
    # Per layout, the state buffers of one call sit 1 element, 8 bytes and 16 bytes less
    # one element past a 16-byte boundary (``padded_stride``: slot strides padded by as
    # many elements, so the slots of one pool alternate between aligned and misaligned
    # tiles).  Every output equals the 16-byte-aligned call's bit for bit and no byte
    # outside the logical state changes.
    events = _run_isolated(
        {"kind": "public", "contract": contract, "layout": layout},
        timeout=_PUBLIC_CHILD_TIMEOUT_S,
    )
    routes = [event for event in events if event["event"] == "route"]
    # Without a loaded Cake GDN record the comparisons would test some other route.
    assert routes, f"{contract}-{layout} loaded no Cake GDN prefill record"
    # The 16-byte-aligned call loads the aligned flavor, every misaligned (offset or
    # padded-stride) call the guarded one: a host rule that wrongly picked the aligned
    # flavor would fault instead.
    flavors = {}
    arm = None
    for event in events:
        if event["event"] == "arm":
            arm = event["arm"]
        elif event["event"] == "flavor":
            flavors.setdefault(arm, set()).add(event["aligned"])
    arms = [event["arm"] for event in events if event["event"] == "arm"]
    assert sorted(flavors) == sorted(arms), (flavors, arms)
    for arm, aligned in flavors.items():
        assert aligned == {arm == "aligned"}, (contract, layout, arm, aligned)
    for event in routes:
        print(f"{contract}-{layout}: {event['variant']} -> {event['unit']}")


# ---------------------------------------------------------------------------------------
# Manifest record sweep

# The batch of every swept record: a zero-length sequence (initial -> final state copy),
# a single token, partial and whole 64-token chunks.
_SWEEP_SEQ_LENS = (130, 0, 257, 1, 64)
_STATE_DL_DTYPES = {
    (2, 32): torch.float32,
    (4, 16): torch.bfloat16,
    (2, 16): torch.float16,
    (10, 8): torch.float8_e4m3fn,
    (12, 8): torch.float8_e5m2,
}
_IO_DL_DTYPES = {2: torch.float16, 4: torch.bfloat16}


def _prefill_records():
    return [
        record
        for record in cake_gdn._manifest()["variants"]
        if record["domain"] == "prefill"
    ]


def _prefill_unit_params():
    try:
        records = _prefill_records()
    except (FileNotFoundError, RuntimeError) as error:
        return [
            pytest.param(
                None, marks=pytest.mark.skip(reason=f"no Cake GDN manifest: {error}")
            )
        ]
    units = sorted(
        {output["path"] for record in records for output in record["outputs"]}
    )
    return [pytest.param(unit, id=Path(unit).stem) for unit in units]


def _is_stateful(record):
    specializations = record["specializations"]
    return any(
        specializations[key]
        for key in ("USE_INITIAL_STATE", "STORE_FINAL_STATE", "ENABLE_CHECKPOINTS")
    )


def _unit_records(unit, arch):
    return [
        record
        for record in _prefill_records()
        if cake_gdn._serves_arch(record, arch)
        and cake_gdn._cuda_record(record, arch)["path"] == unit
    ]


def _log2_or_minus_one(value):
    return value.bit_length() - 1 if value & (value - 1) == 0 else -1


def _record_heads(record):
    """(Hq, Hv) of a record: its host parameters, else the smallest pair its geometry keys admit."""

    params = record["host_binding"]["params"]
    if "HQ" in params and "HV" in params:
        return int(params["HQ"]), int(params["HV"])
    specializations = record["specializations"]
    wanted = tuple(
        int(specializations[key])
        for key in ("HEAD_GROUP_LOG2", "IS_GQA", "NUM_O_HEADS_LOG2")
    )
    for num_o_heads in range(1, 129):
        for min_heads in range(1, num_o_heads + 1):
            if num_o_heads % min_heads:
                continue
            for hq, hv in ((num_o_heads, min_heads), (min_heads, num_o_heads)):
                key = (
                    _log2_or_minus_one(num_o_heads // min_heads),
                    int(hq >= hv),
                    _log2_or_minus_one(num_o_heads),
                )
                if key == wanted:
                    return hq, hv
    raise RuntimeError(f"no head pair for {record['name']} {wanted}")


def _record_dtypes(record):
    """(I/O dtype, state dtype) the record's rendered host shim checks."""

    shim = cake_gdn.render_cake_gdn_host_shim(record)
    io = re.search(r'CheckDtype\(arg_Q, "Q", (\d+), 16, 1\)', shim)
    state = re.search(
        r'CheckDtype\(arg_output_state, "output_state", (\d+), (\d+), 1\)', shim
    )
    if io is None or state is None:
        raise RuntimeError(f"cannot read the dtypes of {record['name']}")
    return (
        _IO_DL_DTYPES[int(io.group(1))],
        _STATE_DL_DTYPES[(int(state.group(1)), int(state.group(2)))],
    )


def _sweep_record(record, arch, unit):
    specializations = record["specializations"]
    schedule = record["source_schedule"].rsplit(":", 1)[-1]
    hq, hv = _record_heads(record)
    io_dtype, state_dtype = _record_dtypes(record)
    heads = max(hq, hv)
    tiles_per_seq = heads * (2 if "_dvsplit" in schedule else 1)
    # The full-grid body is routed only for 128 tiles with one CTA per tile; every other
    # body runs a persistent grid of half the tiles, so each CTA also seeds (and, on the
    # prefetch twins, prefetches) a successor tile.
    full_grid = "fullgrid" in schedule
    if full_grid:
        if 128 % tiles_per_seq:
            raise RuntimeError(f"{record['name']}: {tiles_per_seq} tiles per sequence")
        num_seqs = 128 // tiles_per_seq
    else:
        num_seqs = len(_SWEEP_SEQ_LENS)
    seq_lens = tuple(_SWEEP_SEQ_LENS[i % len(_SWEEP_SEQ_LENS)] for i in range(num_seqs))
    indexed = bool(specializations["USE_STATE_INDICES"])
    checkpoint_every = 64 if specializations["ENABLE_CHECKPOINTS"] else 0
    case = _make_case(
        seq_lens=seq_lens,
        io_dtype=io_dtype,
        state_dtype=state_dtype,
        indexed=indexed,
        checkpoint_every=checkpoint_every,
        num_q_heads=hq,
        num_k_heads=min(hq, hv),
        num_v_heads=hv,
        state_pool_padding=0,
    )
    has_initial = bool(specializations["USE_INITIAL_STATE"])
    has_final = bool(specializations["STORE_FINAL_STATE"])
    plan = _plan(case, has_initial=has_initial, has_final=has_final, in_place=indexed)
    total_tiles = num_seqs * tiles_per_seq
    grid_x = total_tiles if full_grid else (total_tiles + 1) // 2
    tag = {"record": record["name"]}
    _emit(
        "record",
        unit=unit,
        schedule=schedule,
        heads=[hq, hv],
        io_dtype=str(io_dtype),
        state_dtype=str(state_dtype),
        seq_lens=list(seq_lens),
        total_tiles=total_tiles,
        grid_x=grid_x,
        **tag,
    )
    entries = {
        "aligned": cake_gdn.load_cake_gdn_kernel(record["name"], arch),
        "guarded": cake_gdn.load_cake_gdn_kernel(
            record["name"], arch, state_vec_aligned=False
        ),
    }
    # Both flavors load as distinct modules in one process.
    if entries["aligned"] is entries["guarded"]:
        raise RuntimeError(f"{record['name']}: one entry serves both flavors")
    # Sized for the one-CTA-per-tile arm below.
    workspace = torch.empty(total_tiles * 512, dtype=torch.uint8, device="cuda")
    empty_i32 = torch.empty(1, dtype=torch.int32, device="cuda")
    empty_state = torch.empty(1, dtype=state_dtype, device="cuda")
    total_tokens = sum(seq_lens)

    def launch(buffers, tokens, placement):
        initial = buffers["initial"].view if "initial" in buffers else empty_state
        output = buffers["output"].view if "output" in buffers else empty_state
        checkpoints = (
            buffers["checkpoints"].view if "checkpoints" in buffers else empty_state
        )
        entries[placement.get("flavor", "aligned")](
            case["q"],
            case["k"],
            case["v"],
            tokens,
            case["g"],
            case["beta"],
            case["cu_seqlens"],
            case["state_indices"] if indexed else empty_i32,
            initial,
            output,
            checkpoints,
            case["checkpoint_cu_starts"] if checkpoint_every else empty_i32,
            workspace,
            int(initial.stride(0)) if has_initial else heads * 16384,
            int(output.stride(0)) if has_final else heads * 16384,
            checkpoint_every,
            case["scale"],
            num_seqs,
            total_tokens,
            hq,
            hv,
            total_tiles,
            placement.get("grid_x", grid_x),
            1,
            1,
        )

    lanes = plan["lanes"]
    # The reference runs the aligned flavor; the guarded flavor on the same aligned
    # buffers must give the same bits.
    arms = [("aligned", {}), ("guarded_aligned", {})]
    arms += [
        (f"all_off{offset}", _placement("all", offset, lanes))
        for offset in _offsets(lanes)
    ]
    # One buffer off at a time: a guard keyed to the wrong buffer passes when all are off.
    if has_initial:
        arms.append(("initial_off1", {"initial": (1, 0)}))
    if has_final and not indexed:
        arms.append(("output_off1", {"output": (1, 0)}))
    if checkpoint_every:
        arms.append(("checkpoints_off1", {"checkpoints": 1}))
    arms.append(("slot_stride_pad1", {"initial": (0, 1), "output": (0, 1)}))
    if not full_grid:
        # One CTA per tile, the DV-split launch grid: every tile is its CTA's first.
        arms.append(
            (
                "all_off1_one_cta_per_tile",
                dict(_placement("all", 1, lanes), grid_x=total_tiles),
            )
        )
    # Every arm but the reference runs the guarded flavor.
    arms = arms[:1] + [
        (arm, dict(placement, flavor="guarded")) for arm, placement in arms[1:]
    ]
    failures = _run_arms(plan, arms, launch, tag)
    _emit("record_done", failures=failures, **tag)
    return failures


def _child_unit(spec):
    arch = _live_arch()
    failures = 0
    for name in spec["records"]:
        failures += _sweep_record(cake_gdn._kernel_record(name), arch, spec["unit"])
    return failures


@pytest.mark.parametrize("unit", _prefill_unit_params())
def test_cake_gdn_prefill_unit_records_state_off_16_bytes_match_aligned(unit):
    # Every stateful prefill record the unit builds for the live architecture, launched
    # through its raw entry (the aligned flavor on 16-byte-aligned buffers, the guarded
    # flavor on the same buffers and on every other arm): all state buffers off by 1
    # element, 8 bytes and 16 bytes less one element with padded slot strides, then the
    # initial state, the final state, the checkpoints and the slot stride off one at a
    # time, then (except on the full-grid body) all off by one element with one CTA per
    # tile.  Each run equals the 16-byte-aligned run bit for bit and leaves every byte
    # outside the logical state alone.
    arch = _live_arch()
    records = _unit_records(unit, arch)
    if not records:
        pytest.skip(f"{unit} is not built for {arch}")
    stateful = [record["name"] for record in records if _is_stateful(record)]
    if not stateful:
        pytest.skip(f"no record of {unit} reads or writes state on {arch}")
    events = _run_isolated(
        {"kind": "unit", "unit": unit, "records": stateful},
        timeout=_UNIT_CHILD_TIMEOUT_S,
    )
    swept = [event["record"] for event in events if event["event"] == "record_done"]
    assert sorted(swept) == sorted(stateful), (swept, stateful)
    print(f"{unit} on {arch}: {len(swept)} records swept: {swept}")


_CHILD_RUNNERS = {"public": _child_public, "unit": _child_unit}


def _child_main(argv):
    spec = json.loads(argv[2])
    loaded_from = str(Path(flashinfer.__file__).resolve())
    if loaded_from != spec["flashinfer_file"]:
        raise RuntimeError(
            f"child imported flashinfer from {loaded_from}, parent from {spec['flashinfer_file']}"
        )
    failures = _CHILD_RUNNERS[spec["kind"]](spec)
    _emit("done", failures=failures)
    return 1 if failures else 0


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == _CHILD_FLAG:
    sys.exit(_child_main(sys.argv))
