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

The accuracy protocol of the raw forget-gate mode of the KDA chunked training
backend (``use_gate_in_kernel=True``): the raw path is no less accurate than the
explicit gate paths, declared before running.

Arms, each against the exact FP64 recurrence whose gate is activated from the
same BF16 raw gate and FP32 parameters in FP64
(:func:`kda_exact_raw_with_grads`):

* ``R``: the raw path, ``chunk_kda_det(raw, use_gate_in_kernel=True, A_log=...,
  dt_bias=..., lower_bound=-5.0)``;
* ``E_T``: the explicit path with a torch FP32 gate, autograd on the BF16 raw
  gate and the FP32 parameters, then ``chunk_kda_det(g FP32)``;
* ``E_F``: the explicit path with flash-linear-attention's gate (where it
  imports on the node; the arm is left out otherwise, never the test).

Metric: per tensor (``out``, ``dq``, ``dk``, ``dv``, ``dbeta``, the raw gate's
``dg``, ``dA_log``, ``ddt_bias``) the relative RMS error against FP64.
Statistic: for N per-seed log-ratios, ``U = mean + t_{0.975, N-1} sd / sqrt(N)``.
Critical value: per (tensor, row) the ``1 - alpha / m`` quantile of ``U`` over
``B_BOOT`` bootstrap resamples of the A/A log-ratios (``E_T`` against ``E_T``
with the activated gate moved by a seeded +-1 ulp, seeds 2000-2031),
``alpha = 0.05`` and ``m`` the Bonferroni count of the family (8 tensors x the
rows of the device's family x the explicit arms present).  Test (128 seeds
3000-3127, ``t_{0.975, 127}``): for every explicit arm, tensor and row,
``U(log(err_R / err_X)) <= c``.  Sanity: ``R`` has no non-finite element and
every element within ``1e-2 + 1e-2 |ref|``, except the parameter gradients
``dA_log`` / ``ddt_bias`` (sums over every token of a head, whose elementwise
error is the explicit arms' order too): there ``R``'s worst normalized error per
seed is at most ``max(1, 2 x`` the explicit arms' worst``)``.  Rows A-F run
on ``sm_100a``; rows A, B, D and E elsewhere.  Every error array, every
``U``, every ``c``, ``alpha``, ``m``, the seeds, the host and the job are
recorded as JSON.

The protocol takes minutes per row, so the default suite skips it.  Run it on
a compute node::

    FLASHINFER_CAKE_KDA_TRAIN_RAW_GATE_ACCURACY=1 \\
    FLASHINFER_CAKE_KDA_TRAIN_RAW_GATE_ACCURACY_JSON=/path/raw_gate_accuracy.json \\
        pytest tests/experimental/test_cake_kda_det_train_raw_gate_accuracy.py

    python tests/experimental/test_cake_kda_det_train_raw_gate_accuracy.py --json /path/raw_gate_accuracy.json [--rows A B]

``FLASHINFER_CAKE_KDA_TRAIN_RAW_GATE_ACCURACY_ROWS=A,B`` (``--rows``) runs a
subset of the family (``m`` stays the family's: the rows are independent, so
jobs may split them); ``FLASHINFER_CAKE_KDA_TRAIN_RAW_GATE_ACCURACY_REQUIRE_FLA=1``
(``--require-fla``) fails instead of leaving the ``E_F`` arm out, so split jobs
declare the same ``m`` (and fails when ``FLA_COMMIT`` names another commit than
the imported checkout's).  A row outside the device's family is an error, not a
skip.  The record's ``pass`` is ``pass_rows_run`` (every row run passed: the
test and the elementwise sanity gate of ``R``) and ``family_complete``; a split
job's record carries ``rows_run`` for combining.  The code under test is
identified by the commit, the dirty flag, the sha256 of ``git diff HEAD`` and of
each file in :data:`CODE_FILES`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
import torch

if __name__ == "__main__":  # script mode: the repository root on the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.test_helpers.cake_kda_det_train_reference import (  # noqa: E402
    LOWER_BOUND,
    GateFn,
    check_fla_gate,
    elementwise_gate,
    error_metrics,
    fla_kda_gate,
    kda_exact_raw_with_grads,
    make_raw_inputs,
    raw_gate,
    t_quantile,
)
from tests.test_helpers.cake_package import cake_package  # noqa: E402

cake_backend = cake_package("cake_kda_det_train", "cake_backend")
cake_jit = cake_package("cake_kda_det_train", "cake_jit")

ENV_RUN = "FLASHINFER_CAKE_KDA_TRAIN_RAW_GATE_ACCURACY"
ENV_JSON = "FLASHINFER_CAKE_KDA_TRAIN_RAW_GATE_ACCURACY_JSON"
ENV_ROWS = "FLASHINFER_CAKE_KDA_TRAIN_RAW_GATE_ACCURACY_ROWS"
ENV_REQUIRE_FLA = "FLASHINFER_CAKE_KDA_TRAIN_RAW_GATE_ACCURACY_REQUIRE_FLA"


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() not in ("", "0", "false", "no")


RUN = _flag(ENV_RUN)

# ---- the protocol, declared before the first run ----------------------------------------------------------------
ALPHA = 0.05
# the test seeds: 128, disjoint from the A/A seeds and from the 32 pilot seeds 1000-1031, whose U intervals were too
# wide to decide every row; the A/A calibration (32 seeds), c and every margin are the same for both
TEST_SEEDS = tuple(range(3000, 3128))
AA_SEEDS = tuple(range(2000, 2032))
N_SEEDS = len(TEST_SEEDS)
N_AA = len(AA_SEEDS)
B_BOOT = 100_000
BOOT_SEED = 7816
# the +-1 ulp signs of the A/A arm: a CPU generator seeded PERTURB_SEED_BASE + seed
PERTURB_SEED_BASE = 78160000
QUANTILE_METHOD = "linear"
TENSORS = ("out", "dq", "dk", "dv", "dbeta", "dg", "dA_log", "ddt_bias")
EXPLICIT_ARMS = ("E_T", "E_F")
ELEMENTWISE_ATOL = ELEMENTWISE_RTOL = 1e-2
# the parameter gradients are sums over every token of a head: their elementwise error is the explicit arms' order
# too, so R's sanity gate there is its worst normalized error per seed against max(1, 2 x the explicit arms' worst)
PARAM_TENSORS = ("dA_log", "ddt_bias")
PARAM_SANITY_FACTOR = 2.0
# rows: (lengths, heads, dt_bias shift); default gate: mean activated gate about -2.5; D: s -> 1 (dt_bias ~ +6 +
# 0.5 N, the cancellation case of the explicit sigma'); E: slow decay (dt_bias ~ -6 + 0.5 N, g ~ -0.012)
ROWS: dict[str, tuple[tuple[int, ...], int, float]] = {
    "A": ((100, 640, 77, 1000, 129), 2, 0.0),
    "B": ((63, 1, 65), 4, 0.0),
    "C": ((5000, 7001, 4383), 4, 0.0),
    "D": ((4096,), 4, 6.0),
    "E": ((1500, 2596), 4, -6.0),
    "F": ((1000, 1100, 900, 1096), 64, 0.0),
}
FULL_ARCHES = ("sm_100a",)
PARTIAL_ROWS = ("A", "B", "D", "E")
assert sum(ROWS["C"][0]) == 16384 and sum(ROWS["D"][0]) == sum(ROWS["E"][0]) == 4096
assert sum(ROWS["F"][0]) == 4096


def family_rows(arch: str) -> tuple[str, ...]:
    """The rows of ``arch``'s family: all six on ``sm_100a``, rows A, B, D and E elsewhere."""
    return tuple(ROWS) if arch in FULL_ARCHES else PARTIAL_ROWS


def family_size(arch: str, explicit_arms: Sequence[str]) -> int:
    """``m``: the comparisons of the family (8 tensors x rows x explicit arms present)."""
    return len(TENSORS) * len(family_rows(arch)) * len(explicit_arms)


# ---- the arms ---------------------------------------------------------------------------------------------------
RAW_LEAVES = ("q", "k", "v", "g", "beta", "A_log", "dt_bias")
LEAF_RESULTS = ("dq", "dk", "dv", "dg", "dbeta", "dA_log", "ddt_bias")


def _leaves(inputs: dict) -> list[torch.Tensor]:
    return [inputs[n].detach().clone().requires_grad_(True) for n in RAW_LEAVES]


def _results(out: torch.Tensor, leaves: Sequence[torch.Tensor]) -> dict:
    res = {"out": out.detach()}
    for name, leaf in zip(LEAF_RESULTS, leaves, strict=True):
        res[name] = leaf.grad.detach()
    return res


def arm_raw(inputs: dict) -> dict:
    """``R``: the raw path through the autograd entry."""
    leaves = _leaves(inputs)
    q, k, v, raw, beta, a_log, dt_bias = leaves
    out = cake_backend.chunk_kda_det(
        q,
        k,
        v,
        raw,
        beta,
        inputs["cu_seqlens"],
        use_gate_in_kernel=True,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
    )
    out.backward(inputs["dout"])
    return _results(out, leaves)


def arm_explicit(inputs: dict, gate_fn: GateFn) -> dict:
    """An explicit arm: ``gate_fn`` activates the gate (FP32, differentiable in the raw gate and the parameters),
    then ``chunk_kda_det(g)``; autograd carries ``dg`` back through the gate."""
    leaves = _leaves(inputs)
    q, k, v, raw, beta, a_log, dt_bias = leaves
    g = gate_fn(raw, a_log, dt_bias)
    assert g.dtype == torch.float32 and g.shape == raw.shape, (g.dtype, g.shape)
    out = cake_backend.chunk_kda_det(
        q, k, v, g.contiguous(), beta, inputs["cu_seqlens"]
    )
    out.backward(inputs["dout"])
    return _results(out, leaves)


def torch_gate(raw: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor):
    """``E_T``'s gate: torch FP32 on the BF16 raw gate and the FP32 parameters."""
    return raw_gate(raw, a_log, dt_bias, LOWER_BOUND, torch.float32)


def perturbed_torch_gate(seed: int) -> GateFn:
    """The A/A arm's gate: ``E_T``'s, every element moved by one ulp up or down (seeded signs) in the forward value
    only (``g + (nextafter(g, +-inf) - g).detach()``): an equally accurate gate evaluation."""

    def gate(raw, a_log, dt_bias):
        g = torch_gate(raw, a_log, dt_bias)
        gen = torch.Generator(device="cpu").manual_seed(PERTURB_SEED_BASE + int(seed))
        up = (torch.rand(tuple(g.shape), generator=gen) < 0.5).to(g.device)
        target = torch.where(
            up,
            torch.full_like(g, math.inf),
            torch.full_like(g, -math.inf),
        )
        return g + (torch.nextafter(g.detach(), target) - g.detach())

    return gate


# ---- statistics -------------------------------------------------------------------------------------------------
def u_stat(ell: Sequence[float], tq: float) -> float:
    n = len(ell)
    mean = sum(ell) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in ell) / (n - 1))
    return mean + tq * sd / math.sqrt(n)


def bootstrap_critical_value(ell_aa: Sequence[float], tq: float, m: int) -> float:
    """``c``: the ``1 - ALPHA / m`` quantile of ``U`` over ``B_BOOT`` resamples (with replacement, size N) of the
    A/A log-ratios, a CPU generator seeded ``BOOT_SEED`` per (tensor, row)."""
    x = torch.tensor(list(ell_aa), dtype=torch.float64)
    n = x.numel()
    gen = torch.Generator(device="cpu").manual_seed(BOOT_SEED)
    idx = torch.randint(0, n, (B_BOOT, n), generator=gen)
    samples = x[idx]
    u = samples.mean(dim=1) + tq * samples.std(dim=1, unbiased=True) / math.sqrt(n)
    level = 1.0 - ALPHA / m
    return float(torch.quantile(u, level, interpolation=QUANTILE_METHOD).item())


def _max_normalized_error(cand: torch.Tensor, ref: torch.Tensor) -> float:
    """``max |cand - ref| / (atol + rtol |ref|)`` in FP64 (``inf`` for a non-finite element)."""
    c = cand.detach().to(torch.float64)
    r = ref.detach().to(torch.float64)
    norm = (c - r).abs() / (ELEMENTWISE_ATOL + ELEMENTWISE_RTOL * r.abs())
    norm = torch.where(torch.isfinite(c), norm, torch.full_like(norm, math.inf))
    return float(norm.max().item())


def _sanity(elementwise: dict, explicit_arms: Sequence[str]) -> dict:
    """``R``'s elementwise sanity verdict per tensor and seed: no non-finite element; every element within
    ``atol + rtol |ref|``, or for :data:`PARAM_TENSORS` a worst normalized error at most ``max(1, factor x`` the
    explicit arms' worst on that seed ``)``."""
    out: dict = {}
    for t in TENSORS:
        verdicts = []
        for i, r in enumerate(elementwise["R"][t]):
            if r["nonfinite"]:
                verdicts.append(False)
            elif t in PARAM_TENSORS:
                worst = max(
                    (elementwise[a][t][i]["max_norm"] for a in explicit_arms),
                    default=0.0,
                )
                verdicts.append(r["max_norm"] <= max(1.0, PARAM_SANITY_FACTOR * worst))
            else:
                verdicts.append(r["mismatch"] == 0)
        out[t] = verdicts
    return out


def _log_ratio(num: float, den: float) -> float:
    floor = 1e-300
    return math.log(max(num, floor) / max(den, floor))


# ---- the run ----------------------------------------------------------------------------------------------------
# the files whose bytes identify the code under test (the commit alone does not when the tree is dirty)
CODE_FILES = (
    "flashinfer/experimental/cake_kda_det_train/cake_backend.py",
    "flashinfer/experimental/cake_kda_det_train/cake_jit.py",
    "flashinfer/cake_kda_det_train.py",
    "tests/experimental/test_cake_kda_det_train_raw_gate_accuracy.py",
    "tests/test_helpers/cake_kda_det_train_reference.py",
)


def _git(root: Path, *args: str) -> str | None:
    try:
        res = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            timeout=60,
            check=False,
        )
    except Exception:
        return None
    return res.stdout.decode(errors="replace") if res.returncode == 0 else None


def _code_identity(root: Path) -> dict:
    """The commit, whether the tree is dirty, the sha256 of ``git diff HEAD`` and of each file in
    :data:`CODE_FILES`."""
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain")
    diff = _git(root, "diff", "HEAD")
    files = {}
    for rel in CODE_FILES:
        path = root / rel
        files[rel] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
    return {
        "repo_commit": commit.strip() if commit else None,
        "repo_dirty": None if status is None else bool(status.strip()),
        "repo_status": None if status is None else status.splitlines(),
        "repo_diff_head_sha256": (
            None if diff is None else hashlib.sha256(diff.encode()).hexdigest()
        ),
        "code_sha256": files,
    }


def _environment(device: torch.device, arch: str) -> dict:
    root = Path(__file__).resolve().parents[2]
    props = torch.cuda.get_device_properties(device)
    return {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": props.name,
        "sm_count": props.multi_processor_count,
        "arch": arch,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        **_code_identity(root),
        "argv": sys.argv,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _errors(cand: dict, ref: dict) -> dict:
    return {name: error_metrics(cand[name], ref[name])["rel_rms"] for name in TENSORS}


def _reference(inputs: dict) -> dict:
    q, k, v, raw, beta = (inputs[n] for n in ("q", "k", "v", "g", "beta"))
    ref_out, ref_grads = kda_exact_raw_with_grads(
        q,
        k,
        v,
        raw,
        beta,
        inputs["A_log"],
        inputs["dt_bias"],
        inputs["dout"],
        inputs["cu_seqlens"],
        cake_backend.SCALE,
        LOWER_BOUND,
    )
    return {"out": ref_out, **ref_grads}


def run_row(
    row: str,
    device: torch.device,
    *,
    arms: dict[str, GateFn],
    m: int,
    tq_aa: float,
    tq_test: float,
    log: Callable[[str], None] = print,
) -> dict:
    """The A/A calibration and the test of one row: every error array, the log-ratios, ``c`` and ``U`` per tensor
    and arm, the verdicts and the elementwise sanity gate of ``R``."""
    lengths, heads, shift = ROWS[row]
    t0 = time.time()
    rec: dict = {
        "row": row,
        "lengths": list(lengths),
        "heads": heads,
        "dt_bias_mean": shift,
        "explicit_arms": list(arms),
    }

    def inputs_of(seed: int) -> dict:
        return make_raw_inputs(lengths, heads, device, seed=seed, dt_bias_mean=shift)

    # A/A calibration: E_T against E_T with the gate moved by +-1 ulp
    aa_err = {"E_T": {t: [] for t in TENSORS}, "E_T_ulp": {t: [] for t in TENSORS}}
    gate_stats = []
    for seed in AA_SEEDS:
        inputs = inputs_of(seed)
        with torch.no_grad():
            g = torch_gate(inputs["g"], inputs["A_log"], inputs["dt_bias"])
            gate_stats.append((g.mean().item(), g.min().item(), g.max().item()))
        ref = _reference(inputs)
        for arm, fn in (("E_T", torch_gate), ("E_T_ulp", perturbed_torch_gate(seed))):
            err = _errors(arm_explicit(inputs, fn), ref)
            for t in TENSORS:
                aa_err[arm][t].append(err[t])
        del inputs, ref
    rec["gate_mean_min_max_aa"] = gate_stats
    ell_aa = {
        t: [
            _log_ratio(b, a)
            for a, b in zip(aa_err["E_T"][t], aa_err["E_T_ulp"][t], strict=True)
        ]
        for t in TENSORS
    }
    c = {t: bootstrap_critical_value(ell_aa[t], tq_aa, m) for t in TENSORS}
    rec["aa"] = {
        "seeds": list(AA_SEEDS),
        "err": aa_err,
        "ell": ell_aa,
        "U": {t: u_stat(ell_aa[t], tq_aa) for t in TENSORS},
        "c": c,
    }
    log(f"row {row}: A/A done in {time.time() - t0:.0f} s; c = {c}")
    # the test
    test_err: dict = {arm: {t: [] for t in TENSORS} for arm in ("R", *arms)}
    elementwise: dict = {arm: {t: [] for t in TENSORS} for arm in ("R", *arms)}

    def record(arm: str, res: dict, ref: dict) -> None:
        err = _errors(res, ref)
        for t in TENSORS:
            test_err[arm][t].append(err[t])
            gate_res = elementwise_gate(
                res[t], ref[t], ELEMENTWISE_ATOL, ELEMENTWISE_RTOL
            )
            entry = {k: gate_res[k] for k in ("mismatch", "nonfinite", "numel")}
            entry["max_norm"] = _max_normalized_error(res[t], ref[t])
            elementwise[arm][t].append(entry)

    for seed in TEST_SEEDS:
        inputs = inputs_of(seed)
        ref = _reference(inputs)
        record("R", arm_raw(inputs), ref)
        for arm, fn in arms.items():
            record(arm, arm_explicit(inputs, fn), ref)
        del inputs, ref
    sanity = _sanity(elementwise, tuple(arms))
    ell = {
        arm: {
            t: [
                _log_ratio(r, x)
                for r, x in zip(test_err["R"][t], test_err[arm][t], strict=True)
            ]
            for t in TENSORS
        }
        for arm in arms
    }
    u = {arm: {t: u_stat(ell[arm][t], tq_test) for t in TENSORS} for arm in arms}
    verdict = {arm: {t: bool(u[arm][t] <= c[t]) for t in TENSORS} for arm in arms}
    nonfinite = {
        arm: [t for t in TENSORS if not all(math.isfinite(e) for e in test_err[arm][t])]
        for arm in test_err
    }
    rec["test"] = {
        "seeds": list(TEST_SEEDS),
        "err": test_err,
        "ell": ell,
        "U": u,
        "pass": verdict,
        "nonfinite_err": nonfinite,
        "elementwise": elementwise,
        "elementwise_R": sanity,
        "elementwise_R_pass": all(all(ok) for ok in sanity.values()),
    }
    rec["pass"] = (
        all(all(v.values()) for v in verdict.values())
        and not any(nonfinite.values())
        and rec["test"]["elementwise_R_pass"]
    )
    rec["wall_s"] = time.time() - t0
    log(
        f"row {row}: {'PASS' if rec['pass'] else 'FAIL'} in {rec['wall_s']:.0f} s; "
        + "; ".join(
            f"{arm}: "
            + ", ".join(f"{t} U={u[arm][t]:+.3f}/c={c[t]:+.3f}" for t in TENSORS)
            for arm in arms
        )
    )
    return rec


class Protocol:
    """One run of the protocol on the current device: the family, the arms present, ``m``, the record."""

    def __init__(self, json_path: str | None, require_fla: bool) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("the raw gate accuracy protocol needs a CUDA device")
        self.device = torch.device("cuda", torch.cuda.current_device())
        capability = tuple(torch.cuda.get_device_capability(self.device.index))
        self.arch = cake_jit.arch_for_capability(capability)
        if self.arch is None:
            raise RuntimeError(
                f"compute capability {capability} is not an exported target"
            )
        self.arms: dict[str, GateFn] = {"E_T": torch_gate}
        fla, fla_info = fla_kda_gate(LOWER_BOUND)
        if fla is not None:
            probe = make_raw_inputs((65,), 2, self.device, seed=1)
            reason = check_fla_gate(fla, probe, LOWER_BOUND)
            if reason is None:
                self.arms["E_F"] = fla
            else:
                fla_info["unavailable"] = reason
        if require_fla:
            if "E_F" not in self.arms:
                raise RuntimeError(f"E_F required but unavailable: {fla_info}")
            env, found = fla_info.get("commit_env"), fla_info.get("git_commit")
            if not env:
                fla_info["commit_warning"] = (
                    "FLA_COMMIT unset: the flash-linear-attention commit under test is not declared"
                )
                print(f"warning: {fla_info['commit_warning']}", file=sys.stderr)
            elif found and not found.startswith(env):
                raise RuntimeError(
                    f"E_F required: FLA_COMMIT={env} but the imported checkout is at {found} ({fla_info})"
                )
        self.m = family_size(self.arch, tuple(self.arms))
        self.tq_aa = t_quantile(0.975, N_AA - 1)
        self.tq = t_quantile(0.975, N_SEEDS - 1)
        assert abs(self.tq_aa - 2.0395134464) < 1e-8, self.tq_aa
        # the declared constant, against a tabulated value of t_{0.975, 31}
        assert abs(self.tq - 1.9788195347) < 1e-8, self.tq
        self.json_path = json_path
        self.record: dict = {
            "protocol": {
                "alpha": ALPHA,
                "m": self.m,
                "family_rows": list(family_rows(self.arch)),
                "tensors": list(TENSORS),
                "explicit_arms": list(self.arms),
                "aa_seeds": list(AA_SEEDS),
                "test_seeds": list(TEST_SEEDS),
                "n_seeds": N_SEEDS,
                "b_boot": B_BOOT,
                "boot_seed": BOOT_SEED,
                "boot_rng": "torch CPU Generator seeded boot_seed per (tensor, row); torch.randint(0, N, (B, N))",
                "quantile": f"torch.quantile(U, 1 - alpha / m, interpolation={QUANTILE_METHOD!r})",
                "t_0975_df": [self.tq, N_SEEDS - 1],
                "t_0975_df_aa": [self.tq_aa, N_AA - 1],
                "revision": 2,
                "param_tensors_sanity": [list(PARAM_TENSORS), PARAM_SANITY_FACTOR],
                "perturb_seed_base": PERTURB_SEED_BASE,
                "lower_bound": LOWER_BOUND,
                "metric": "rel_rms = rms(cand - ref) / rms(ref), FP64 reference",
                "elementwise_gate": [ELEMENTWISE_ATOL, ELEMENTWISE_RTOL],
            },
            "fla": fla_info,
            "environment": _environment(self.device, self.arch),
            "rows_requested": None,
            "rows_run": [],
            "family_complete": False,
            "pass_rows_run": False,
            "pass": False,
            "rows": {},
        }

    def rows(self, requested: Sequence[str] | None) -> tuple[str, ...]:
        """The rows of this run: the family, or the requested subset of it (``ValueError`` for a row outside the
        family of the device, so a mis-targeted job selects nothing silently)."""
        family = family_rows(self.arch)
        self.record["rows_requested"] = list(requested) if requested else None
        if not requested:
            return family
        outside = [r for r in requested if r not in family]
        if outside:
            raise ValueError(
                f"rows {outside} are not in the {self.arch} family {family} (the rows are {tuple(ROWS)})"
            )
        return tuple(dict.fromkeys(requested))

    def run(self, row: str, log: Callable[[str], None] = print) -> dict:
        rec = run_row(
            row,
            self.device,
            arms=self.arms,
            m=self.m,
            tq_aa=self.tq_aa,
            tq_test=self.tq,
            log=log,
        )
        self.record["rows"][row] = rec
        rows_run = list(self.record["rows"])
        self.record["rows_run"] = rows_run
        self.record["family_complete"] = set(rows_run) == set(family_rows(self.arch))
        # pass_rows_run: every row of this run passed; pass: and the run covers the whole family (a split job's
        # record is one part of the family's evidence: combine the parts' rows)
        self.record["pass_rows_run"] = all(
            r["pass"] for r in self.record["rows"].values()
        )
        self.record["pass"] = (
            self.record["pass_rows_run"] and self.record["family_complete"]
        )
        self.write()
        return rec

    def write(self) -> None:
        if not self.json_path:
            return
        path = Path(self.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.record, indent=1))
        tmp.replace(path)


_PROTOCOL: Protocol | None = None


def _protocol() -> Protocol:
    global _PROTOCOL
    if _PROTOCOL is None:
        _PROTOCOL = Protocol(
            os.environ.get(ENV_JSON) or None, require_fla=_flag(ENV_REQUIRE_FLA)
        )
    return _PROTOCOL


def _requested_rows() -> list[str]:
    raw = os.environ.get(ENV_ROWS, "")
    return [r.strip() for r in raw.split(",") if r.strip()]


@pytest.mark.skipif(
    not RUN, reason=f"raw gate accuracy protocol: set {ENV_RUN}=1 (minutes per row)"
)
@pytest.mark.parametrize("row", tuple(ROWS))
def test_raw_gate_is_no_less_accurate_than_the_explicit_gates(row: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    protocol = _protocol()
    rows = protocol.rows(
        _requested_rows()
    )  # ValueError (a failure) for a row outside the family
    if not rows:
        pytest.fail(f"{ENV_ROWS} selects no row of the {protocol.arch} family")
    if row not in rows:
        pytest.skip(
            f"row {row} is not in this run ({protocol.arch} family / {ENV_ROWS})"
        )
    rec = protocol.run(row)
    assert rec["test"]["elementwise_R_pass"], rec["test"]["elementwise_R"]
    failed = [
        (arm, t, rec["test"]["U"][arm][t], rec["aa"]["c"][t])
        for arm, by_t in rec["test"]["pass"].items()
        for t, ok in by_t.items()
        if not ok
    ]
    assert not failed and not any(rec["test"]["nonfinite_err"].values()), (
        failed,
        rec["test"]["nonfinite_err"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="the raw gate mode accuracy protocol")
    parser.add_argument("--json", default=os.environ.get(ENV_JSON), help="record path")
    parser.add_argument("--rows", nargs="*", default=_requested_rows() or None)
    parser.add_argument(
        "--require-fla", action="store_true", default=_flag(ENV_REQUIRE_FLA)
    )
    args = parser.parse_args()
    protocol = Protocol(args.json, require_fla=args.require_fla)
    print(
        f"arch {protocol.arch}, arms R vs {list(protocol.arms)}, m = {protocol.m}, "
        f"t = {protocol.tq:.10f}; fla: {protocol.record['fla']}",
        flush=True,
    )
    try:
        rows = protocol.rows(args.rows)
    except ValueError as exc:
        print(f"RAW_GATE_ACCURACY FAIL: {exc}", flush=True)
        return 1
    if not rows:
        print("RAW_GATE_ACCURACY FAIL: no row selected", flush=True)
        return 1
    ok = True
    for row in rows:
        rec = protocol.run(row, log=lambda s: print(s, flush=True))
        ok &= rec["pass"]
    family = family_rows(protocol.arch)
    complete = set(rows) == set(family)
    print(
        "RAW_GATE_ACCURACY",
        "PASS" if ok else "FAIL",
        f"rows={','.join(rows)}",
        f"family={','.join(family)}",
        f"family_complete={complete}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
