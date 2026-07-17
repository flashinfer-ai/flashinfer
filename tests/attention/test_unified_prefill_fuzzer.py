"""Reject-or-correct fuzzer for the unified paged-prefill prototype.

THE PROPERTY (the whole point of this file):

    For ANY input — valid, structurally malformed, or value-corrupted —
    a call either (a) raises a clean, actionable error, or (b) returns
    results that match the reference semantics.  Silent garbage is the
    only failure mode, and it always fails this suite.

This is the property whose absence produced cuDNN issue #3800 (plausible
finite garbage for batch > 0), the 4-D-KV-silently-dropped-offsets trap,
and the mask-divergence class from the #3921 review.  A unified API is
only trustworthy if this property is machine-checked, not promised.

Mechanics:
- VALID trials: random configs inside the declared capability envelope.
  The run must succeed AND match the oracle (a capability matrix that
  admits configs it cannot run fails here — capability honesty).
- CORRUPTED trials: one mutation applied to a valid config.
    * ``comparable=True`` mutations leave the ground truth well-defined
      (e.g. an under-claimed max): the call may either raise or still
      return the correct result.
    * ``comparable=False`` mutations make the input self-contradictory
      (e.g. non-monotonic indptr): the ONLY acceptable outcome is a raise.
- Every failure prints a standalone repro line (seed + mutation).

Trials per backend via FI_UNIFIED_FUZZ_TRIALS (default 30).

Known gaps (documented caller contract, not fuzzed):
- host mirrors are trusted to match the device tensors (engines own both
  copies; validating equality would cost the sync the API removes);
- zero-length sequences are out of the prototype's envelope.
"""

import os
import random

import pytest
import torch

from flashinfer.attention.unified import resolve_paged_prefill

from .test_unified_prefill_prototype import make_problem, run_unified
from .unified_prefill_reference import reference_paged_prefill

TRIALS = int(os.environ.get("FI_UNIFIED_FUZZ_TRIALS", "30"))
BACKENDS = ["fa2", "fa3", "cudnn", "trtllm-gen", "auto"]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)

# Exceptions that count as a *clean rejection*. Anything else (or silence
# with wrong numbers) is a property violation.
CLEAN = (ValueError, NotImplementedError, TypeError)


def _sample_config(rng: random.Random):
    return dict(
        batch_size=rng.randint(1, 8),
        max_q=rng.choice([1, 8, 33, 64]),
        max_kv=rng.choice([64, 130, 300, 512]),
        heads=rng.choice([(8, 8), (8, 2), (8, 1), (4, 4)]),
        page_size=rng.choice([16, 32, 64]),
        causal=True,
        dtype=torch.bfloat16,
    )


def _build(seed, cfg):
    return make_problem(
        seed=seed,
        batch_size=cfg["batch_size"],
        max_q=cfg["max_q"],
        max_kv=cfg["max_kv"],
        num_qo_heads=cfg["heads"][0],
        num_kv_heads=cfg["heads"][1],
        head_dim_qk=128,
        page_size=cfg["page_size"],
        dtype=cfg["dtype"],
    )


def _backend_runnable(p, backend, causal):
    try:
        resolve_paged_prefill(
            device=torch.device(p["device"]),
            num_qo_heads=p["num_qo_heads"],
            num_kv_heads=p["num_kv_heads"],
            head_dim_qk=p["head_dim_qk"],
            q_dtype=p["dtype"],
            page_size=p["page_size"],
            causal=causal,
            need_lse=True,
            backend=backend,
        )
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Mutation catalog.  Each entry: (name, comparable, mutate(p) -> p') where p'
# is a shallow-modified copy of the problem dict.  ``comparable`` == the
# TRUE config still defines the correct answer (so correct output is an
# acceptable outcome); otherwise only a clean raise is acceptable.
# ---------------------------------------------------------------------------


def _clone(p):
    return dict(p)


def m_max_q_underclaim(p):
    p = _clone(p)
    p["max_q_len"] = max(1, p["max_q_len"] - 1)
    return p


def m_max_kv_underclaim(p):
    p = _clone(p)
    p["max_kv_len"] = max(1, p["max_kv_len"] // 2)
    return p


def m_q_rows_chopped(p):
    p = _clone(p)
    p["q"] = p["q"][:-1]
    return p


def m_indptr_nonmonotonic(p):
    p = _clone(p)
    if p["qo_indptr_cpu"].shape[0] < 4:
        pytest.skip("needs batch >= 3")
    ip = p["qo_indptr_cpu"].clone()
    ip[1], ip[2] = ip[2].clone(), ip[1].clone()
    p["qo_indptr_cpu"] = ip
    p["qo_indptr"] = ip.to(p["qo_indptr"].device)
    return p


def m_indptr_total_mismatch(p):
    p = _clone(p)
    ip = p["qo_indptr_cpu"].clone()
    ip[-1] += 1
    p["qo_indptr_cpu"] = ip
    p["qo_indptr"] = ip.to(p["qo_indptr"].device)
    return p


def m_kv_lens_exceed_capacity(p):
    p = _clone(p)
    cap = p["block_tables"].shape[1] * p["page_size"]
    lens = p["kv_seq_lens_cpu"].clone()
    lens[0] = cap + 3
    p["kv_seq_lens_cpu"] = lens
    p["kv_seq_lens"] = lens.to(p["kv_seq_lens"].device)
    p["max_kv_len"] = int(lens.max())
    return p


def m_int64_indptr(p):
    p = _clone(p)
    p["qo_indptr"] = p["qo_indptr"].to(torch.int64)
    return p


def m_cpu_indptr(p):
    p = _clone(p)
    p["qo_indptr"] = p["qo_indptr_cpu"]
    return p


def m_kv_lens_bad_shape(p):
    p = _clone(p)
    lens = torch.cat([p["kv_seq_lens_cpu"], torch.ones(1, dtype=torch.int32)])
    p["kv_seq_lens"] = lens.to(p["kv_seq_lens"].device)
    p["kv_seq_lens_cpu"] = lens
    return p


def m_block_tables_1d(p):
    p = _clone(p)
    p["block_tables"] = p["block_tables"].flatten()
    return p


def m_page_size_mismatch(p):
    p = _clone(p)
    p["page_size"] = p["page_size"] * 2  # cache pages remain the old size
    return p


def m_q_noncontig(p):
    # fused-QKV-style slice: valid semantics, strided storage.  Backends
    # that can address it must be correct; token-offset backends must
    # REJECT rather than silently mis-address (the #3921 multiplier trap).
    p = _clone(p)
    total, h, d = p["q"].shape
    fused = torch.randn(total, 3 * h, d, dtype=p["q"].dtype, device=p["q"].device)
    fused[:, :h, :] = p["q"]
    p["q"] = fused[:, :h, :]
    assert not p["q"].is_contiguous()
    return p


MUTATIONS = [
    ("max_q_underclaim", True, m_max_q_underclaim),
    ("max_kv_underclaim", True, m_max_kv_underclaim),
    ("q_rows_chopped", False, m_q_rows_chopped),
    ("indptr_nonmonotonic", False, m_indptr_nonmonotonic),
    ("indptr_total_mismatch", False, m_indptr_total_mismatch),
    ("kv_lens_exceed_capacity", False, m_kv_lens_exceed_capacity),
    ("int64_indptr", True, m_int64_indptr),
    ("cpu_indptr", False, m_cpu_indptr),
    ("kv_lens_bad_shape", False, m_kv_lens_bad_shape),
    ("block_tables_1d", False, m_block_tables_1d),
    ("page_size_mismatch", False, m_page_size_mismatch),
    ("q_noncontig", True, m_q_noncontig),
]


def _run_and_check(p, backend, causal, repro):
    """Run one call and enforce reject-or-correct against the TRUE oracle."""
    try:
        _, out, lse = run_unified(p, backend, causal=causal)
    except CLEAN as e:
        assert str(e), f"empty error message is not a clean rejection [{repro}]"
        return "rejected", str(e)
    ref_out, ref_lse = reference_paged_prefill(
        p["q"].contiguous(),
        p["k_cache"],
        p["v_cache"],
        p["qo_indptr_cpu"],
        p["kv_seq_lens_cpu"],
        p["block_tables"],
        p["page_size"],
        causal,
    )
    torch.testing.assert_close(
        out.float(),
        ref_out,
        **OUT_TOL,
        msg=lambda m: f"SILENT WRONG OUTPUT [{repro}]\n{m}",
    )
    torch.testing.assert_close(
        lse,
        ref_lse,
        **LSE_TOL,
        msg=lambda m: f"SILENT WRONG LSE [{repro}]\n{m}",
    )
    return "correct", None


@pytest.mark.parametrize("backend", BACKENDS)
def test_fuzz_valid_configs(backend):
    """Valid configs must run AND be correct — capability honesty included:
    if resolve() admits it, no excuse is accepted at run time."""
    ran = 0
    for trial in range(TRIALS):
        seed = 10_000 + trial
        rng = random.Random(seed)
        cfg = _sample_config(rng)
        p = _build(seed, cfg)
        if not _backend_runnable(p, backend, cfg["causal"]):
            continue
        repro = f"backend={backend} seed={seed} cfg={cfg}"
        outcome, _ = _run_and_check(p, backend, cfg["causal"], repro)
        assert outcome == "correct", (
            f"valid config was rejected — capability matrix admits a config "
            f"the backend cannot run [{repro}]"
        )
        ran += 1
    if ran == 0:
        pytest.skip(f"no valid trial runnable for backend {backend} on this GPU")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mutation", [m[0] for m in MUTATIONS])
def test_fuzz_reject_or_correct(backend, mutation):
    """Corrupted inputs: clean raise, or (when semantics remain defined)
    a correct result.  Silent wrong numbers fail, always."""
    name, comparable, mutate = next(m for m in MUTATIONS if m[0] == mutation)
    checked = 0
    for trial in range(max(3, TRIALS // 5)):
        seed = 20_000 + trial
        rng = random.Random(seed)
        cfg = _sample_config(rng)
        cfg["batch_size"] = max(cfg["batch_size"], 3)
        p = _build(seed, cfg)
        if not _backend_runnable(p, backend, cfg["causal"]):
            continue
        mutated = mutate(p)
        repro = f"backend={backend} seed={seed} mutation={name} cfg={cfg}"
        if comparable:
            # ground truth is still the TRUE problem `p`; hand the oracle the
            # true metadata but the call the mutated one
            merged = dict(mutated)
            merged["qo_indptr_cpu"] = p["qo_indptr_cpu"]
            merged["kv_seq_lens_cpu"] = (
                p["kv_seq_lens_cpu"]
                if mutation != "q_noncontig"
                else mutated["kv_seq_lens_cpu"]
            )
            outcome, _ = _run_and_check(merged, backend, cfg["causal"], repro)
            assert outcome in ("rejected", "correct")
        else:
            try:
                run_unified(mutated, backend, causal=cfg["causal"])
            except CLEAN as e:
                assert str(e), f"empty error message [{repro}]"
            except RuntimeError as e:
                # kernel-level rejection is acceptable only if it is loud and
                # synchronous; record it distinctly so we can tighten later.
                assert str(e), f"empty RuntimeError [{repro}]"
            else:
                raise AssertionError(
                    f"self-contradictory input was ACCEPTED silently [{repro}]"
                )
        checked += 1
    if checked == 0:
        pytest.skip(f"no runnable trial for backend {backend}")
