# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The canonical hierarchical-reduction engine (docs/generalization.md).

A :class:`Reduction` is an op's *definition*: an index set written as mixed-radix
digits in combine order, a leaf, and per-digit combine rules. Its canonical term is
``Reduction.canonical(fp)`` and knows nothing about hardware.

A :class:`Placement` is a *config*: each digit is factored into sub-digits assigned
to physical levels (``reg`` < ``lane`` < ``warp`` < ``cta`` < ``grid``), plus free
structure the engine does not care about.

``plan(red, pl)`` turns the pair into a :class:`Plan`: an ordered list of steps,
each combining one factor at one level, in canonical combine order. The same
:class:`Plan` drives

* :func:`trace`  - a lock-step symbolic simulation of every thread (L2 check), and
* the kernel     - ``init_state`` + ``apply_step`` executed once per thread with a
  hardware :class:`Peers` (CuTe), emitting the same combine sequence over real values.

Because both consume the same plan, the kernel cannot build a different tree than
the tracer. What remains unverified by the tracer is plumbing (loads, shuffles,
barriers, predicates), which the GPU tests cover.

Absent leaves (``None``) are dropped from the tree; a fold with no present input is
absent. A kernel may instead feed the combine's identity element for an absent
leaf when the digit's ``identity_note`` justifies that this is bit-exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

LEVELS = ("reg", "lane", "warp", "cta", "grid")
LANES = 32


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _pairwise(vals: Sequence, comb: Callable):
    """Adjacent pairwise tree, low index first; ``None`` entries are dropped."""
    vals = list(vals)
    assert is_pow2(len(vals)), len(vals)
    while len(vals) > 1:
        nxt = []
        for i in range(0, len(vals), 2):
            a, b = vals[i], vals[i + 1]
            nxt.append(b if a is None else a if b is None else comb(a, b))
        vals = nxt
    return vals[0]


# ---------------------------------------------------------------------------
# Spec side
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Digit:
    name: str
    radix: int
    shape: (
        str  # 'pairwise' (bitwise-commutative combine) | 'fold' (sequential, ascending)
    )
    combine: Optional[Callable] = None  # pairwise: fn(fp, a, b)
    init: Optional[Callable] = None  # fold: fn(fp) -> value
    step: Optional[Callable] = None  # fold: fn(fp, acc, x) -> value
    identity_note: str = (
        ""  # why a kernel may substitute the identity for absent leaves
    )
    indexed: bool = False  # fold: the step also takes the digit value, fn(fp, acc, x, i) (a per-index coefficient such as a block scale)

    def __post_init__(self):
        assert self.shape in ("pairwise", "fold"), self.shape
        if self.shape == "pairwise":
            assert self.combine is not None and is_pow2(self.radix), (
                self.name,
                self.radix,
            )
        else:
            assert self.init is not None and self.step is not None

    def reduce(self, fp, vals: Sequence):
        """Reduce ``vals`` (ordered by digit value; None = absent)."""
        if self.shape == "fold":
            acc = None
            for i, v in enumerate(vals):
                if v is None:
                    continue
                prev = self.init(fp) if acc is None else acc
                acc = (
                    self.step(fp, prev, v, i)
                    if self.indexed
                    else self.step(fp, prev, v)
                )
            return acc
        return _pairwise(vals, lambda a, b: self.combine(fp, a, b))


@dataclass(frozen=True)
class Reduction:
    name: str
    digits: Tuple[Digit, ...]  # combine order: digits[0] is combined first (innermost)
    leaf: Callable[[Any, Dict[str, int]], Any]  # fn(fp, {digit: value}) -> value | None

    def digit(self, name: str) -> Digit:
        for d in self.digits:
            if d.name == name:
                return d
        raise KeyError(name)

    def canonical(self, fp):
        """The canonical term: reduce digit 0 innermost, digit n-1 outermost."""

        def rec(j: int, fixed: Dict[str, int]):
            if j < 0:
                return self.leaf(fp, fixed)
            d = self.digits[j]
            return d.reduce(
                fp, [rec(j - 1, {**fixed, d.name: v}) for v in range(d.radix)]
            )

        return rec(len(self.digits) - 1, {})


# ---------------------------------------------------------------------------
# Config side
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Factor:
    radix: int
    level: str

    def __post_init__(self):
        assert self.level in LEVELS, self.level
        assert self.radix >= 1


@dataclass(frozen=True)
class Placement:
    """``factors[digit]`` lists that digit's factors innermost (least significant) first."""

    factors: Tuple[Tuple[str, Tuple[Factor, ...]], ...]
    free: Tuple[Tuple[str, Any], ...] = ()

    def of(self, name: str) -> Tuple[Factor, ...]:
        for n, fs in self.factors:
            if n == name:
                return fs
        raise KeyError(name)

    @staticmethod
    def make(factors: Mapping[str, Sequence[Tuple[int, str]]], **free) -> "Placement":
        return Placement(
            tuple(
                (n, tuple(Factor(r, l) for r, l in fs if r != 1))
                for n, fs in factors.items()
            ),
            tuple(sorted(free.items())),
        )


@dataclass(frozen=True)
class Step:
    digit: str
    pos: int  # factor position within the digit
    radix: int
    level: str
    shape: str
    reg_index: int = -1  # reg steps: index into the reg key tuple
    lane_mask: int = 0  # lane micro-steps: xor mask of this bit
    warp_stride: int = 1  # warp/cta steps: stride of this factor in the unit id
    warp_total: int = 1

    def describe(self) -> str:
        extra = f" mask={self.lane_mask}" if self.level == "lane" else ""
        return (
            f"{self.level}:{self.digit}[{self.pos}]x{self.radix}({self.shape}){extra}"
        )


class Plan:
    """Steps in canonical combine order plus the ownership map (who holds which leaf)."""

    def __init__(self, red: Reduction, pl: Placement):
        self.red, self.pl = red, pl
        self.steps: List[Step] = []
        self.reg_factors: List[
            Tuple[str, int, int]
        ] = []  # (digit, pos, radix) in plan order
        self.lane_bits: Dict[Tuple[str, int], int] = {}
        self.unit_strides: Dict[str, Dict[Tuple[str, int], int]] = {
            "warp": {},
            "cta": {},
            "grid": {},
        }
        self.units: Dict[str, int] = {"lane": 1, "warp": 1, "cta": 1, "grid": 1}
        self._build()

    # -- construction -----------------------------------------------------------
    def _build(self):
        red, pl = self.red, self.pl
        lane_bit = 0
        seen_grid = False
        for d in red.digits:
            fs = pl.of(d.name)
            if d.shape == "fold" and not fs and d.radix == 1:
                fs = (
                    Factor(1, "reg"),
                )  # a one-step fold still needs its (trivial) register step
            assert _prod(f.radix for f in fs) == d.radix, (
                f"{d.name}: factors {fs} != radix {d.radix}"
            )
            if d.shape == "fold":
                assert len(fs) == 1 and fs[0].level == "reg", (
                    f"fold digit {d.name} must be one reg factor"
                )
            for pos, f in enumerate(fs):
                if d.shape == "pairwise":
                    assert is_pow2(f.radix), (d.name, f)
                if seen_grid:
                    raise ValueError("grid-level factors must be outermost")
                if f.level == "reg":
                    self.reg_factors.append((d.name, pos, f.radix))
                    self.steps.append(
                        Step(
                            d.name,
                            pos,
                            f.radix,
                            "reg",
                            d.shape,
                            reg_index=len(self.reg_factors) - 1,
                        )
                    )
                elif f.level == "lane":
                    self.lane_bits[(d.name, pos)] = lane_bit
                    for b in range(f.radix.bit_length() - 1):
                        self.steps.append(
                            Step(
                                d.name,
                                pos,
                                f.radix,
                                "lane",
                                d.shape,
                                lane_mask=1 << (lane_bit + b),
                            )
                        )
                    lane_bit += f.radix.bit_length() - 1
                    self.units["lane"] <<= f.radix.bit_length() - 1
                    assert self.units["lane"] <= LANES, (
                        "more than 32 lanes of lane-level factors"
                    )
                else:
                    stride = self.units[f.level]
                    self.unit_strides[f.level][(d.name, pos)] = stride
                    self.units[f.level] *= f.radix
                    self.steps.append(
                        Step(d.name, pos, f.radix, f.level, d.shape, warp_stride=stride)
                    )
                    if f.level == "grid":
                        seen_grid = True
        # fill in totals
        self.steps = [
            Step(**{**s.__dict__, "warp_total": self.units[s.level]})
            if s.level in ("warp", "cta", "grid")
            else s
            for s in self.steps
        ]
        self.n_reg = len(self.reg_factors)
        self.final_key = (None,) * self.n_reg

    # -- queries ----------------------------------------------------------------
    def reg_index(self, digit: str, pos: int = 0) -> Optional[int]:
        for i, (d, p, _) in enumerate(self.reg_factors):
            if d == digit and p == pos:
                return i
        return None

    def reg_keys(self, exclude: Optional[int] = None) -> List[tuple]:
        """All reg keys, earliest reg factor most significant. ``exclude`` sets that
        position to None (used to enumerate the outer keys of a fused first step)."""
        keys: List[tuple] = [()]
        for i, (_, _, r) in enumerate(self.reg_factors):
            keys = [
                k + ((v,) if i != exclude else (None,))
                for k in keys
                for v in (range(r) if i != exclude else (0,))
            ]
        return keys

    def keys_at(self, step_i: int) -> List[tuple]:
        """Static set of state keys just before step ``step_i``."""
        keys = self.reg_keys()
        for s in self.steps[:step_i]:
            if s.level == "reg":
                keys = list(
                    dict.fromkeys(
                        k[: s.reg_index] + (None,) + k[s.reg_index + 1 :] for k in keys
                    )
                )
        return keys

    def digit_value(self, digit: str, coords: Dict[str, Any], key: tuple):
        """Value of ``digit`` for the leaf a unit holds under reg key ``key``.
        ``coords`` gives the unit's lane / warp / cta / grid ids (ints or dynamic)."""
        fs = self.pl.of(digit)
        val, stride = 0, 1
        for pos, f in enumerate(fs):
            if f.level == "reg":
                fv = key[self.reg_index(digit, pos)]
            elif f.level == "lane":
                lo = self.lane_bits[(digit, pos)]
                fv = (
                    (coords["lane"] >> lo) & (f.radix - 1)
                    if lo or f.radix != self.units["lane"]
                    else coords["lane"]
                )
            else:
                ustride = self.unit_strides[f.level][(digit, pos)]
                fv = coords[f.level]
                if ustride != 1:
                    fv = fv // ustride
                if f.radix != self.units[f.level] // ustride:
                    fv = fv % f.radix
            val = val + fv * stride if stride != 1 else fv
            stride *= f.radix
        return val

    def leaf_index(self, coords: Dict[str, Any], key: tuple) -> Dict[str, Any]:
        return {d.name: self.digit_value(d.name, coords, key) for d in self.red.digits}

    def describe(self) -> str:
        return " -> ".join(s.describe() for s in self.steps) + f"  units={self.units}"

    def cost(self) -> Dict[str, int]:
        """Shuffle / exchange multiplicity (how many entries are live at each non-reg step)."""
        out = {"shuffles": 0, "exchanged_floats_per_thread": 0}
        for i, s in enumerate(self.steps):
            n = len(self.keys_at(i))
            if s.level == "lane":
                out["shuffles"] += n
            elif s.level in ("warp", "cta"):
                out["exchanged_floats_per_thread"] += n * s.radix
        return out


def plan(red: Reduction, pl: Placement) -> Plan:
    return Plan(red, pl)


def _prod(xs) -> int:
    p = 1
    for x in xs:
        p *= x
    return p


# ---------------------------------------------------------------------------
# Execution (shared by tracer and kernel)
# ---------------------------------------------------------------------------
class Peers:
    """How a unit reaches its peers. Implemented by the tracer and by the CuTe kernel."""

    def shuffle_xor(self, key: tuple, value, mask: int):  # value from lane ^ mask
        raise NotImplementedError

    def exchange(self, step: Step, state: Dict[tuple, Any]) -> Dict[tuple, List[Any]]:
        """For every key, the ``step.radix`` sibling values (incl. own) in factor order."""
        raise NotImplementedError


def init_state(
    p: Plan, fp, loader: Callable[[tuple], Any]
) -> Tuple[Dict[tuple, Any], int]:
    """Materialise the per-thread state. If the first step is a reg step it is fused
    with loading (``loader`` is called for all its values consecutively, so a kernel
    can stream one fragment at a time). Returns ``(state, next_step_index)``."""
    if p.steps and p.steps[0].level == "reg":
        s0 = p.steps[0]
        d = p.red.digit(s0.digit)
        state = {}
        for outer in p.reg_keys(exclude=s0.reg_index):
            vals = [
                loader(outer[: s0.reg_index] + (v,) + outer[s0.reg_index + 1 :])
                for v in range(s0.radix)
            ]
            state[outer] = _reduce_factor(d, s0, fp, vals)
        return state, 1
    return {k: loader(k) for k in p.reg_keys()}, 0


def _reduce_factor(d: Digit, s: Step, fp, vals):
    if d.shape == "fold":
        return d.reduce(fp, vals)  # a fold digit is exactly one factor
    return _pairwise(vals, lambda a, b: d.combine(fp, a, b))


def apply_step(
    p: Plan, s: Step, state: Dict[tuple, Any], peers: Peers, fp
) -> Dict[tuple, Any]:
    d = p.red.digit(s.digit)
    if s.level == "reg":
        groups: Dict[tuple, Dict[int, Any]] = {}
        for key, v in state.items():
            g = key[: s.reg_index] + (None,) + key[s.reg_index + 1 :]
            groups.setdefault(g, {})[key[s.reg_index]] = v
        return {
            g: _reduce_factor(d, s, fp, [vals.get(i) for i in range(s.radix)])
            for g, vals in groups.items()
        }
    if s.level == "lane":
        new = {}
        for key, v in state.items():
            # Absent values are dropped from the tree: the result is whichever side is
            # present. (In a kernel an absent entry is uniformly absent across lanes, and
            # its Peers must return None without emitting a shuffle.)
            partner = peers.shuffle_xor(key, v, s.lane_mask)
            new[key] = (
                partner
                if v is None
                else v
                if partner is None
                else d.combine(fp, v, partner)
            )
        return new
    sib = peers.exchange(s, state)
    return {key: _pairwise(sib[key], lambda a, b: d.combine(fp, a, b)) for key in state}


def run_steps(
    p: Plan,
    state: Dict[tuple, Any],
    start: int,
    peers: Peers,
    fp,
    stop_before: Optional[str] = None,
):
    """Apply the plan's steps from index ``start`` up to (not including) the first step of digit
    ``stop_before`` (or to the end). Returns ``(state, next_step_index)``. A kernel that needs an inner
    digit's complete partials stops here, reads
    them from the state (one entry per remaining register key) and resumes with :func:`run`."""
    i = start
    while i < len(p.steps):
        s = p.steps[i]
        if stop_before is not None and s.digit == stop_before:
            break
        state = apply_step(p, s, state, peers, fp)
        i += 1
    return state, i


def run(p: Plan, state: Dict[tuple, Any], start: int, peers: Peers, fp):
    state, _ = run_steps(p, state, start, peers, fp)
    assert list(state) == [p.final_key], state.keys()
    return state[p.final_key]


# ---------------------------------------------------------------------------
# Symbolic tracer
# ---------------------------------------------------------------------------
class _TracePeers(Peers):
    def __init__(self, snapshot, unit, plan_):
        self.snap, self.unit, self.plan = snapshot, unit, plan_

    def shuffle_xor(self, key, value, mask):
        g, w, l = self.unit
        return self.snap[(g, w, l ^ mask)][key]

    def exchange(self, step, state):
        g, w, l = self.unit
        level_id = {"warp": w, "cta": g}[step.level]
        my = (level_id // step.warp_stride) % step.radix
        out = {}
        for key in state:
            vals = []
            for sv in range(step.radix):
                sid = level_id + (sv - my) * step.warp_stride
                u = (sid, w, l) if step.level == "cta" else (g, sid, l)
                vals.append(self.snap[u][key])
            out[key] = vals
        return out


def trace(
    p: Plan, fp, leaf_of: Optional[Callable] = None, stop_before: Optional[str] = None
):
    """Lock-step simulation of all units of one output group. Returns the final
    value per unit ``(cta, warp, lane)``; all of them must be identical. With ``stop_before``
    (a digit name) the simulation stops before that digit's first step, as :func:`run_steps` does,
    and returns every unit's state dict (the inner digits' complete partials, keyed by the remaining
    register keys)."""
    if any(s.level == "grid" for s in p.steps):
        raise NotImplementedError("grid-level steps")
    leaf_of = leaf_of or p.red.leaf
    units = [
        (c, w, l)
        for c in range(p.units["cta"])
        for w in range(p.units["warp"])
        for l in range(p.units["lane"])
    ]
    states, starts = {}, set()
    for u in units:
        coords = {"cta": u[0], "warp": u[1], "lane": u[2], "grid": 0}

        def loader(key: tuple, c: Dict[str, int] = coords) -> Any:
            return leaf_of(fp, p.leaf_index(c, key))

        st, start = init_state(p, fp, loader)
        states[u] = st
        starts.add(start)
    (start,) = starts
    for s in p.steps[start:]:
        if stop_before is not None and s.digit == stop_before:
            return {u: dict(st) for u, st in states.items()}
        snap = {u: dict(st) for u, st in states.items()}
        for u in units:
            states[u] = apply_step(p, s, states[u], _TracePeers(snap, u, p), fp)
    if stop_before is not None:
        return {u: dict(st) for u, st in states.items()}
    for u in units:
        assert list(states[u]) == [p.final_key]
    return {u: st[p.final_key] for u, st in states.items()}


def verify(p: Plan, fp) -> Any:
    """Assert every unit's traced value *is* the canonical term. Returns it."""
    ref = p.red.canonical(fp)
    got = trace(p, fp)
    bad = [u for u, v in got.items() if v is not ref]
    assert not bad, (
        f"{p.red.name}: {len(bad)}/{len(got)} units differ from canonical under {p.describe()}"
    )
    return ref


# ---------------------------------------------------------------------------
# Blocked lowering of a fold whose trip count is a runtime value
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BlockedFold:
    """How a kernel issues a ``fold`` digit whose step count is a *runtime* value.

    A kernel cannot unroll a fold over a runtime number of steps, so the plan's single
    register-level fold step is lowered to a loop nest: the steps are issued in blocks of
    ``inner`` (a static radix; for the GEMM, the K-steps one smem stage holds), blocks
    ascending, steps ascending inside a block; the fold's very first step carries the digit's
    ``init`` (the accumulator reset) and every later step accumulates. The kernel supplies the
    loop idiom over blocks (its trip count is runtime) and the step op; this object supplies
    everything that decides the order and the reset: the issue order inside a block
    (:meth:`inner_steps`), the digit value of each issue (:meth:`step_index`) and the
    accumulate predicate (:meth:`accumulate`). :meth:`trace` simulates the whole nest for a
    concrete block count, so the term a kernel builds is conformed node for node against the
    reduction's canonical term (``GemmKernelPlan.verify``) and against the Lean loop model
    (``GEMMConfig.kblockLoop``, generated cases in ``Cases.lean``).
    """

    digit: str
    inner: int

    def __post_init__(self):
        assert self.inner >= 1, self.inner

    def inner_steps(self) -> range:
        """Issue order inside a block: ascending."""
        return range(self.inner)

    def step_index(self, blk, j: int):
        """Digit value of issue ``(blk, j)``; ``blk`` may be a runtime value."""
        return blk * self.inner + j

    def accumulate(self, blk, j: int):
        """Accumulate predicate of issue ``(blk, j)``: false exactly on the fold's first step
        (block 0, step 0), which carries the init. ``blk`` may be a runtime value, in which
        case the predicate is the runtime comparison ``blk != 0``."""
        if j > 0:
            return True
        return blk != 0

    def blocks_for(self, steps: int) -> int:
        """Blocks needed for ``steps`` fold steps (the last block may be padded with identity steps)."""
        return -(-steps // self.inner)

    def trace(
        self, red: Reduction, fp, n_blocks: int, fixed: Optional[Dict[str, int]] = None
    ):
        """The term the nest builds over ``n_blocks`` blocks (``n_blocks * inner`` steps) for the
        leaves of ``red`` with its other digits held at ``fixed``."""
        d = red.digit(self.digit)
        assert d.shape == "fold", d
        fixed = dict(fixed or {})
        acc = None
        for blk in range(n_blocks):
            for j in self.inner_steps():
                i = self.step_index(blk, j)
                x = red.leaf(fp, {**fixed, self.digit: i})
                assert x is not None, "blocked folds have no absent leaves"
                prev = acc if self.accumulate(blk, j) else d.init(fp)
                # an indexed digit's step takes the digit value as well (`Digit.reduce`)
                acc = d.step(fp, prev, x, i) if d.indexed else d.step(fp, prev, x)
        return acc
