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

Options of the context-parallel (KCP) packed GDN training route
(``flashinfer.experimental.cake_gdn_train``): every behaviour switch of the
entry points is a keyword-only field of :class:`KcpOptions`, validated once
per call by :meth:`KcpOptions.check`.  No field is read from the environment;
the package reads ``os.environ`` nowhere.

Public fields (defaults are the shipped behaviour)::

    exchange        "a2a" | "ring"          cross-rank exchange scheme (ring: n_d_max - 1 <= 2 and save_transfer=False)
    save_transfer   True                    the merge owner keeps its c1 receive tensor (and M_seg at P > 1) for the backward
    save_mode       "light" | "full"        saved set: S_in / c1_recv / M_seg + the cheap gate rows and WY inverse (the
                                            backward replays #1 and the tape walk), or additionally every forward tape
                                            ("full" is opt-in only; never the default)
    boundary_mode   "saved" | "replay"      keep the saved set, or keep nothing and re-run the whole module forward
                                            (summaries, exchange, merge) inside the backward (checkpoint replay)
    precision_mode  0 | 1 | 2               transfer-walk build record (template numbering): 0 = shipped TF32 hi/lo (2-MMA,
                                            bitwise the 3-MMA class); 1 = the _three_mma twin and 2 = _plain_tf32 are
                                            TEST-ONLY: refused unless test_knobs names the same value
    transfer_walk   "tf32_hilo" | "bf16x3"  state products of the transfer walk; "bf16x3" stays in the vocabulary but is
                                            NOT BUILT in this package: the validator refuses it unconditionally
    offset_class    "fla" | "fp32"          B_f from the #2 offset walk from zero with the local rounding points (default),
                                            or from the #3 transfer walk with init_mode = 1 / use_u = 1 (FP32 class)
    layout          "tma" | "copy"          strided packed-QKV views pass through as TMA row strides, or are copied to
                                            contiguous operands inside the call (the copy is part of the timed call)
    group_reduce    "fused" | "host"        the FP32 reduction of the HV / H_qk value heads' dq / dk contributions inside
                                            the dqkg program, or FP32 partials reduced on the host in a fixed order
    graph           "off" | "replay"        keyed CUDA-graph capture / replay of a call
    reuse_recompute_tapes  True | False | "auto"   under torch.utils.checkpoint(use_reentrant=False) the light
                                            backward binds the forward tapes of torch's recompute of the same backward
                                            pass instead of running its own replay of #1 and the tape walk.  "auto"
                                            (the shipped default) = only when the GDN node is the last operation of
                                            the checkpointed region that saves a tensor (the same frame proof as
                                            recompute_output="auto"): then its backward is the region's first and the
                                            kept tapes die before any other backward runs; with a later saver (an
                                            output projection, an MLP) the kept tapes would live through those
                                            operations' backward and raise the region's backward peak by the full
                                            tape set (+2323 MiB on a 261,987-token (2, 6) region with a 768-wide
                                            projection and MLP: 9251 vs 6927 MiB, step 21.3 vs 22.7 ms on one GB300),
                                            so "auto" takes the replay form there; True = always; False = the replay
                                            form.
                                            None = the default, which is False with graph="replay", the one mode that
                                            excludes the kept tapes; an explicit True / "auto" with graph="replay" is
                                            refused; a forward outside a backward pass, the direct autograd path and
                                            every other mode are unchanged; one kept set is live at a time, so in a
                                            segment of several layers only the last one skips the replay
    recompute_output  "auto" | "always"    under torch.utils.checkpoint(use_reentrant=False) torch re-runs the forward
                                            inside .backward() only to re-save its tensors and discards that run's
                                            return value; when this module's autograd node is the last saver of the
                                            checkpointed region, "auto" (the shipped default) does not launch the
                                            output program (#5) of that re-run -- the proof reads torch's checkpoint
                                            frame on the verified torch builds only (the module constant
                                            RECOMPUTE_OUTPUT_TORCH_VERSIONS) and launches as before whenever any
                                            step of it is unreachable, the node is not executable, a later saver
                                            follows or early stop is off (LAST_CALL["fwd"]["fwd_o_reason"] says
                                            why); never skipped with graph="replay" or outside the autograd node;
                                            "always" = the unconditional launch
    backend         "kernels" | "fake"      the exported programs, or the torch emulation of the same ABI (tests)
    segments        None | int              requested segments per fragment (P); must equal the context's plan
    test_knobs      None | dict             private levers (schedule, fused_walks, cols_rule, debug_sequence,
                                            record_states, p_rule, fwdh_vh, dhu_vh) and the names of the test-only values above;
                                            keyword argument only, never an environment variable
"""

from __future__ import annotations

import dataclasses
from typing import Any

EXCHANGES = ("a2a", "ring")
EXCHANGE_DEFAULT = "a2a"
RING_MAX_HOPS = 2  # exchange="ring" needs every document chain within this many hops (n_d_max - 1 <= 2)
SAVE_TRANSFER_DEFAULT = True
SAVE_MODES = ("light", "full")
SAVE_MODE_DEFAULT = "light"  # "full" is opt-in only
BOUNDARY_MODES = ("saved", "replay")
BOUNDARY_MODE_DEFAULT = "saved"
# Transfer-walk build records (template numbering shared with the KDA route): 0 = shipped TF32 hi/lo, 1 = _three_mma
# (test twin), 2 = _plain_tf32 (test / negative control).  Test values are never selected by default.
PRECISION_MODES = (0, 1, 2)
PRECISION_MODE_DEFAULT = 0
PRECISION_MODES_TEST_ONLY = (1, 2)
TRANSFER_WALKS = ("tf32_hilo", "bf16x3")
TRANSFER_WALK_DEFAULT = "tf32_hilo"
TRANSFER_WALKS_UNBUILT = (
    "bf16x3",
)  # no GDN bf16x3 module is built in this release; the name is kept, the value refused
OFFSET_CLASSES = ("fla", "fp32")
OFFSET_CLASS_DEFAULT = "fla"
LAYOUTS = ("tma", "copy")
LAYOUT_DEFAULT = "tma"
GROUP_REDUCES = ("fused", "host")
GROUP_REDUCE_DEFAULT = "fused"
GRAPH_MODES = ("off", "replay")
GRAPH_DEFAULT = "off"
REUSE_RECOMPUTE_TAPES = (True, False, "auto")
# "auto" (the shipped default) = the checkpoint-aware light backward keeps the recompute's tapes only when the node
# closes the checkpointed region; True = always; False = the replay form (the default with graph="replay")
REUSE_RECOMPUTE_TAPES_DEFAULT: bool | str = "auto"
RECOMPUTE_OUTPUTS = ("auto", "always")
RECOMPUTE_OUTPUT_DEFAULT = "auto"  # the recompute's dead output program is skipped when torch's checkpoint frame proves it dead
BACKENDS = ("kernels", "fake")
BACKEND_DEFAULT = "kernels"
MAX_SEGMENTS = 64
# Private levers of record (test knobs, keyword-only): schedule of the launch sequences, the fused tape walks, the merge
# column rule, the collective issue-order assertion, the state copies for the oracles, the plan's P rule (checked
# against the context) and the names of the test-only build records (a test-only ``precision_mode`` /
# ``transfer_walk`` value is accepted only when ``test_knobs`` names the same value).
SCHEDULES = ("wait_first", "overlap")
SCHEDULE_DEFAULT = "wait_first"
COLS_RULES = ("fixed", "by_chains")
COLS_RULE_DEFAULT = "fixed"
TEST_KNOBS: dict[str, tuple[Any, ...] | None] = {
    "schedule": SCHEDULES,
    "fused_walks": (0, 1),
    "cols_rule": COLS_RULES,
    "debug_sequence": (False, True),
    "record_states": (
        False,
        True,
    ),  # keep S_in (forward) / dS_out (backward) copies in LAST_CALL for the oracles
    "p_rule": None,  # any string; validated against the context's plan rule at call time
    "precision_mode": PRECISION_MODES_TEST_ONLY,  # names the test-only build record the call selects
    "fwdh_vh": (
        64,
        128,
    ),  # force the #2 form (K1 L8A_DECISION rule otherwise: 64 iff n_walk * HV <= sm_count)
    "dhu_vh": (
        64,
        128,
    ),  # force the #7 form (rule otherwise: 64 iff n_walk * HV * 2 <= 4 * sm_count)
}
TEST_KNOB_DEFAULTS: dict[str, Any] = {
    "schedule": SCHEDULE_DEFAULT,
    "fused_walks": 1,
    "cols_rule": COLS_RULE_DEFAULT,
    "debug_sequence": False,
    "record_states": False,
    "p_rule": None,
    "precision_mode": PRECISION_MODE_DEFAULT,
    "fwdh_vh": None,
    "dhu_vh": None,
}


def validate_heads(H_qk: int, HV: int) -> int:
    """``r = HV / H_qk`` of a legal head pair (``HV % H_qk == 0``, both >= 1); ``ValueError`` otherwise."""
    H_qk, HV = int(H_qk), int(HV)
    if H_qk < 1 or HV < 1:
        raise ValueError(f"head counts must be >= 1, got H_qk={H_qk}, HV={HV}")
    if HV % H_qk:
        raise ValueError(
            f"HV={HV} is not a multiple of H_qk={H_qk}: every Q/K head serves the same number of value heads"
        )
    return HV // H_qk


def validate_ring(n_d_max: int) -> None:
    """``exchange="ring"`` applies only when every document chain has at most ``RING_MAX_HOPS`` hops."""
    hops = max(int(n_d_max) - 1, 0)
    if hops > RING_MAX_HOPS:
        raise ValueError(
            f"exchange='ring' needs n_d_max - 1 <= {RING_MAX_HOPS}, this partition has {hops}"
        )


@dataclasses.dataclass(kw_only=True)
class KcpOptions:
    """The keyword-only options of ``chunk_gdn_kcp`` / ``chunk_gdn_kcp_fwd`` / ``chunk_gdn_kcp_bwd`` (module docstring).

    :meth:`check` validates the field values and the cross-field rules (ring needs ``save_transfer=False``;
    ``reuse_recompute_tapes`` needs ``graph="off"``; a
    test-only ``precision_mode`` / ``transfer_walk`` needs the same value in ``test_knobs``; a known ``save_mode`` /
    ``boundary_mode`` / ``offset_class`` / ``graph`` / ``layout`` / ``group_reduce`` / ``recompute_output`` /
    ``backend``; ``test_knobs``
    with known keys and values).  Rules that need the context (``exchange="ring"`` vs ``n_d_max``) or the operands
    (``HV % H_qk``) are checked at call time through :func:`validate_ring` / :func:`validate_heads`.
    """

    exchange: str = EXCHANGE_DEFAULT
    save_transfer: bool = SAVE_TRANSFER_DEFAULT
    save_mode: str = SAVE_MODE_DEFAULT
    boundary_mode: str = BOUNDARY_MODE_DEFAULT
    precision_mode: int = PRECISION_MODE_DEFAULT
    transfer_walk: str = TRANSFER_WALK_DEFAULT
    offset_class: str = OFFSET_CLASS_DEFAULT
    layout: str = LAYOUT_DEFAULT
    group_reduce: str = GROUP_REDUCE_DEFAULT
    graph: str = GRAPH_DEFAULT
    # None = the shipped default, resolved in __post_init__ (REUSE_RECOMPUTE_TAPES_DEFAULT; False under graph="replay")
    reuse_recompute_tapes: bool | str | None = None
    recompute_output: str = RECOMPUTE_OUTPUT_DEFAULT
    backend: str = BACKEND_DEFAULT
    segments: int | None = None
    test_knobs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.reuse_recompute_tapes is None:
            # the shipped default: the kept-tape backward, except under CUDA-graph replay (a captured graph bakes the
            # arena addresses of its tapes while the kept tapes are fresh tensors), where the default is the replay form
            self.reuse_recompute_tapes = (
                REUSE_RECOMPUTE_TAPES_DEFAULT if self.graph == GRAPH_DEFAULT else False
            )

    def check(self) -> KcpOptions:
        if self.exchange not in EXCHANGES:
            raise ValueError(
                f"exchange must be one of {EXCHANGES}, got {self.exchange!r}"
            )
        if not isinstance(self.save_transfer, bool):
            raise TypeError(
                f"save_transfer must be a bool, got {type(self.save_transfer).__name__}"
            )
        if self.exchange == "ring" and self.save_transfer:
            raise ValueError(
                "exchange='ring' receives no c1 tensor: it needs save_transfer=False (there is no ring form of save_transfer=True)"
            )
        if self.save_mode not in SAVE_MODES:
            raise ValueError(
                f"save_mode must be one of {SAVE_MODES}, got {self.save_mode!r}"
            )
        if self.boundary_mode not in BOUNDARY_MODES:
            raise ValueError(
                f"boundary_mode must be one of {BOUNDARY_MODES}, got {self.boundary_mode!r}"
            )
        if (
            isinstance(self.precision_mode, bool)
            or self.precision_mode not in PRECISION_MODES
        ):
            raise ValueError(
                f"precision_mode must be one of {PRECISION_MODES} (0 = shipped TF32 hi/lo; 1 = _three_mma and "
                f"2 = _plain_tf32 are test-only), got {self.precision_mode!r}"
            )
        if self.transfer_walk not in TRANSFER_WALKS:
            raise ValueError(
                f"transfer_walk must be one of {TRANSFER_WALKS}, got {self.transfer_walk!r}"
            )
        if self.transfer_walk in TRANSFER_WALKS_UNBUILT:
            raise ValueError(
                f"transfer_walk={self.transfer_walk!r} is not built in this package (no GDN BF16x3 transfer "
                f"walk ships or is test-rendered in this release); the shipped walk is {TRANSFER_WALK_DEFAULT!r}"
            )
        if self.offset_class not in OFFSET_CLASSES:
            raise ValueError(
                f"offset_class must be one of {OFFSET_CLASSES}, got {self.offset_class!r}"
            )
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {self.layout!r}")
        if self.group_reduce not in GROUP_REDUCES:
            raise ValueError(
                f"group_reduce must be one of {GROUP_REDUCES}, got {self.group_reduce!r}"
            )
        if self.graph not in GRAPH_MODES:
            raise ValueError(f"graph must be one of {GRAPH_MODES}, got {self.graph!r}")
        if not (
            isinstance(self.reuse_recompute_tapes, bool)
            or self.reuse_recompute_tapes == "auto"
        ):
            raise TypeError(
                f"reuse_recompute_tapes must be a bool or 'auto', got {self.reuse_recompute_tapes!r}"
            )
        if self.reuse_recompute_tapes and self.graph != "off":
            raise ValueError(
                f"reuse_recompute_tapes={self.reuse_recompute_tapes!r} needs graph='off': a captured graph bakes the "
                "arena addresses of its tapes, while the kept tapes are fresh tensors handed from the recompute "
                "forward to the backward"
            )
        if self.recompute_output not in RECOMPUTE_OUTPUTS:
            raise ValueError(
                f"recompute_output must be one of {RECOMPUTE_OUTPUTS}, got {self.recompute_output!r}"
            )
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.segments is not None:
            if isinstance(self.segments, bool) or not isinstance(self.segments, int):
                raise TypeError(
                    f"segments must be an int or None, got {type(self.segments).__name__}"
                )
            if not 1 <= self.segments <= MAX_SEGMENTS:
                raise ValueError(
                    f"segments must be in [1, {MAX_SEGMENTS}], got {self.segments}"
                )
        if self.test_knobs is not None:
            if not isinstance(self.test_knobs, dict):
                raise TypeError(
                    f"test_knobs must be a dict or None, got {type(self.test_knobs).__name__}"
                )
            unknown = sorted(set(self.test_knobs) - set(TEST_KNOBS))
            if unknown:
                raise ValueError(
                    f"test_knobs: unknown knob(s) {unknown}; known: {sorted(TEST_KNOBS)}"
                )
            for key, value in self.test_knobs.items():
                allowed = TEST_KNOBS[key]
                if allowed is None:
                    if not isinstance(value, str):
                        raise ValueError(
                            f"test_knobs[{key!r}] must be a str, got {value!r}"
                        )
                elif (
                    isinstance(value, bool) != isinstance(allowed[0], bool)
                    or value not in allowed
                ):
                    raise ValueError(
                        f"test_knobs[{key!r}] must be one of {allowed}, got {value!r}"
                    )
        # test-only build records: the public field may carry them only when test_knobs names the
        # same value -- the public API alone cannot select a test program
        if self.precision_mode != PRECISION_MODE_DEFAULT and (
            self.test_knobs is None
            or self.test_knobs.get("precision_mode") != self.precision_mode
        ):
            raise ValueError(
                f"precision_mode={self.precision_mode!r} is a test-only build record: name it in "
                f"test_knobs={{'precision_mode': {self.precision_mode!r}}} (the shipped value is {PRECISION_MODE_DEFAULT!r})"
            )
        return self

    def knob(self, name: str) -> Any:
        """The value of test knob ``name`` (its default when ``test_knobs`` does not set it)."""
        if name not in TEST_KNOBS:
            raise KeyError(name)
        if self.test_knobs is not None and name in self.test_knobs:
            return self.test_knobs[name]
        return TEST_KNOB_DEFAULTS[name]

    def key(self) -> tuple:
        """A hashable view of every field (the CUDA-graph validity key)."""
        knobs = (
            None
            if self.test_knobs is None
            else tuple(sorted((str(k), v) for k, v in self.test_knobs.items()))
        )
        return tuple(
            getattr(self, f.name) if f.name != "test_knobs" else knobs
            for f in dataclasses.fields(self)
        )


OPTION_NAMES = tuple(f.name for f in dataclasses.fields(KcpOptions))


def make_options(options: Any = None, /, **kwargs: Any) -> KcpOptions:
    """A validated :class:`KcpOptions` from an instance, a mapping, or keyword arguments (unknown names are a
    ``TypeError``: no option is ever read from anywhere but the call)."""
    if isinstance(options, KcpOptions):
        if kwargs:
            options = dataclasses.replace(options, **kwargs)
        return options.check()
    merged: dict[str, Any] = {}
    if options is not None:
        if not isinstance(options, dict):
            raise TypeError(
                f"options must be a KcpOptions, a dict or None, got {type(options).__name__}"
            )
        merged.update(options)
    merged.update(kwargs)
    unknown = sorted(set(merged) - set(OPTION_NAMES))
    if unknown:
        raise TypeError(
            f"unexpected option(s) {unknown}; known options: {list(OPTION_NAMES)}"
        )
    return KcpOptions(**merged).check()


__all__ = [
    "BACKENDS",
    "BACKEND_DEFAULT",
    "BOUNDARY_MODES",
    "BOUNDARY_MODE_DEFAULT",
    "COLS_RULES",
    "COLS_RULE_DEFAULT",
    "EXCHANGES",
    "EXCHANGE_DEFAULT",
    "GRAPH_DEFAULT",
    "GRAPH_MODES",
    "GROUP_REDUCES",
    "GROUP_REDUCE_DEFAULT",
    "LAYOUTS",
    "LAYOUT_DEFAULT",
    "MAX_SEGMENTS",
    "OFFSET_CLASSES",
    "OFFSET_CLASS_DEFAULT",
    "OPTION_NAMES",
    "PRECISION_MODES",
    "PRECISION_MODES_TEST_ONLY",
    "PRECISION_MODE_DEFAULT",
    "RECOMPUTE_OUTPUTS",
    "RECOMPUTE_OUTPUT_DEFAULT",
    "REUSE_RECOMPUTE_TAPES_DEFAULT",
    "RING_MAX_HOPS",
    "SAVE_MODES",
    "SAVE_MODE_DEFAULT",
    "SAVE_TRANSFER_DEFAULT",
    "SCHEDULES",
    "SCHEDULE_DEFAULT",
    "TEST_KNOBS",
    "TEST_KNOB_DEFAULTS",
    "TRANSFER_WALKS",
    "TRANSFER_WALKS_UNBUILT",
    "TRANSFER_WALK_DEFAULT",
    "KcpOptions",
    "make_options",
    "validate_heads",
    "validate_ring",
]
