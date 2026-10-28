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

Registry and loader glue of the Cake packed KDA chunked training kernels
(SM100 / SM103 / SM107).

Every record of ``MODULES`` is one mechanically exported program: a CUDA
device translation unit plus a tvm-ffi binding, compiled once per
architecture.  A record carries the host plan only: the program's ``library``
stem (its ``build.json`` entries are ``<library>_<arch>``), the architectures it
is delivered for, the ``stage`` of the launch order it fills and its
``variant``, the ``select_rule`` that picks one variant of a stage from the
host-known problem scalars, the ``grid_rule`` that shapes its launch, the
kernel-parameter -> logical-argument map and the argument plan of its ``run``
entry.  What to compile -- sources, digests, nvcc flags -- is recorded once in
``build.json`` next to this file; :mod:`.cake_build` serves the library from a
prebuilt ``aot/`` file when a builder compiled it and otherwise through
FlashInfer's JIT from the same sources and flags.

The rules are data, evaluated here (:func:`select_record`, :func:`launch_grid`)
from the host-known scalars: the padded chunk bound ``num_chunks_pad``, the
document count, the head count, the device's SM count and architecture, and
the ``path`` of the call (``PATHS``: the forward with and without a saved set,
the three backward paths from a saved set and the checkpoint replay).  No rule
reads a tensor.  Adding a
program is one record; a stage with one record has the ``always`` rule unless
the stage is optional (``OPTIONAL_STAGES``: a stage some paths or routes do not
launch selects no record there).

Two records may share one library (``fwdh_v3`` and ``fwdh_p3``: the segmented
recurrence's first pass and its unfused third pass are launches of one kernel
with different grids and flags; ``wy_bwd`` and ``wy_fwd_rows``: the backward's
WY program on the saved ``Akk`` token rows, which the saving forward launches
too on its overlap rows and its paired serial rows); every other library has
one record.  Four records
launch with the programmatic-stream-serialization attribute
(``launch["use_pdl"]``: ``fwdh_p3_o_ov`` / ``fwdh_p3_o_ov_id`` and their
token-row twins ``fwdh_p3_o_ov_rows`` / ``fwdh_p3_o_ov_id_rows``, the pass 2 of
the forward's carry || pass-2 overlap pair with ``seg_carry_ov`` on the rows
whose pass-2 grid is at least 3.0 waves of the device on ``sm_100a`` /
``sm_103a`` and 2.3 waves on ``sm_107a``; the pass-2 CTAs start while the
carry runs and spin on the carry's per-segment release flags, ``flags`` /
``epoch``); every other record is a plain stream-ordered launch.

The forward has two paths.  ``fwd`` saves the set for the backward (the
token-row ``Aqk`` / ``Akk`` are written) and its consumers read those rows
where a program for them ships -- the fused serial recurrence ``fwdh_wy_rows``,
on the overlap rows ``wy_fwd_rows`` and ``fwdh_p3_o_ov_rows`` /
``fwdh_p3_o_ov_id_rows``, and on the paired serial rows (the 32-column form
below the fused threshold) ``wy_fwd_rows`` and ``fwdh_o_v2_rows`` -- so the
intra program writes no internal copies there; ``fwd_nosave`` (no saved set)
launches the internal-row forms ``fwdh_wy`` / ``wy_fwd`` / ``fwdh_p3_o_ov`` /
``fwdh_p3_o_ov_id`` / ``fwdh_o_v2`` and the intra program writes the internal
copies they read (the 64-column serial form keeps them on both paths on
``sm_100a`` / ``sm_103a``; on ``sm_107a`` its saving forward reads the saved rows
too: ``fwdh_o_v1_rows``).

The state recurrences have two routes.  With more than ``SEGMENTS["max_heads"]``
heads -- except the head counts of ``SEGMENTS["plan_heads_by_arch"]``: the
16-head rows of ``sm_103a`` / ``sm_107a`` take the segmented route too -- the
serial walk over each document's chunk chain runs, in two forms
(the ``slots_*`` rules: four CTAs of 32 value columns per document and head
when that one-wave form fits the device, else two CTAs of 64 columns).  The
forward's serial walk also forms the output (``fwdh_o_*``) and, from 32 heads,
the WY representation (``fwdh_wy`` / ``fwdh_wy_rows``: no ``wy_fwd`` launch); the backward's serial
recompute of the 64-column form from 64 heads forms the WY representation too
(``fwdh_wy_bwd*``: no ``wy`` launch, the ``w^T`` tile goes to the ``w`` buffer and
the matching ``dhu_v1_wt*`` reads it).  The serial adjoint recurrence
accumulates ``dv`` itself from the attention rows (``dhu_v2t*`` / ``dhu_v1*``),
so the serial ``dav`` program computes ``dAqk`` alone (``dav_serial*``) and from
16 heads the fused gradient hosts even that (``dqkg_davq`` / ``dqkg_int_davq``:
no ``dav`` launch; on ``sm_107a`` from 16 heads their ``_nohint`` twins, the
same kernel built with its L2 prefetch and evict-first hints off).  With at
most ``max_heads`` heads the segmented route runs:
every document is cut into ``segments`` chunk-aligned segments
(:func:`segment_plan`; the block table, with per-document proportional segment
counts for at most ``prop_max_chains`` (document, head) chains, is written up
to 256 documents together with the chunk tables by the intra program
``fwd_intra_v14`` on the forward and the replay -- no table launch there -- and
by ``table_v3`` where a saved-set path rebuilds them, and by ``seg_table``
after ``table`` above), the ``fwdh_v3`` launch walks every
segment in parallel (pass 1: the
per-segment transfer and offset), ``seg_carry`` chains the segments' states
within each document (with at most two heads the two-level carry
``seg_carry_v8`` where its host rule accepts,
``carry_two_level``: the groups of segments in parallel, their products handed
over through the release-flag words under a fresh ``epoch``), and the
``fwdh_p3`` stage walks again from the carried states (pass 2; the forward's ``fwdh_p3_o`` forms the output, the saved-state
backward of the rows with at most 32 (document, head) chains ``fwdh_p3_wy``
the WY representation, every other path ``fwdh_p3``).  The segmented adjoint
recurrence is two passes of one record around ``seg_carry`` in reverse, every
form accumulating ``dv`` itself: the few-chain rows ``dhu_v3s_f*`` (two CTAs
per segment and head), the other rows ``dhu_v3_f*`` / ``dhu_v3r_f*`` by the
walk's wave count (one CTA; on ``sm_103a`` below four waves ``dhu_v3p_f*``,
the form prefetching the next chunk's tiles; the ``_wt`` forms behind
``fwdh_p3_wy``, the ``_replay`` forms on the replay); so the fused gradient hosts ``dAqk`` on every
segmented row (no ``dav`` launch) and ``dav_serial*`` runs on the serial rows
below 16 heads alone.  ``seg_carry`` is launched by the segmented recurrences between their
passes; it is a stage of this registry for its record, not a step of the
launch orders.

The backward prep has four forms: ``bwd_prep_g8`` on the ``tape`` path (that
set carries no ``qg``), ``bwd_prep_g8_replay`` on the replay (it copies ``dout``
to the internal rows too) and, on the saved-set paths ``light`` /
``light_states`` whose version-4 set carries the forward's ``qg``,
``bwd_prep_g8_noqg`` (no ``q`` read, no ``qg`` write) or -- on the rows whose
set carries the WY representation's ``w`` / ``u`` too: the paired serial
16-head rows of ``sm_100a`` and the segmented 16-head rows of ``sm_103a`` /
``sm_107a`` above 32 (document, head) chains -- ``bwd_prep_g8_noqg_nokb`` (no
``kb`` write either; those rows launch no ``wy`` program on either light
path; the serial rows' form is gated to the ``light`` path, as the serial
route never saves the states).  The GLM intra programs of ``sm_103a`` and
``sm_107a`` are ``glm_intra_v14`` and ``glm_intra_v18``, the gate epilogue
fused into their drains (``glm_intra_v18`` is the same program with every
named-barrier slot an immediate, so two of its CTAs share an SM on ``sm_107a``);
the ``gate_epilogue`` stage is optional and launches on ``sm_100a`` alone,
after ``glm_intra``.

The derived leaves of a record (``kernel_symbol``, ``launch``, ``arg_plan``,
``logical``) are rewritten by the exporter from the rendered binding; the
other fields are the host plan and never change.
"""

from __future__ import annotations

import functools
from typing import Any

import tvm_ffi

from . import cake_build

# One verified build per architecture: B200 (``sm_100a``), B300 / GB300
# (``sm_103a``) and R200 (``sm_107a``).  Every architecture is compiled with its
# exact ``-gencode`` flag set (``build.json`` ``arch_flags``); no family fallback.
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}

# The paths a call takes (the ``path_in`` rules read them): ``fwd`` the
# forward that saves the set for the backward (its consumers read the saved
# token rows where a program for them ships); ``fwd_nosave`` the forward
# without one (the internal-row forms); ``light`` the backward from a version-4 saved set without the
# segment-boundary states (every serial problem, and a segmented problem whose
# pack is too short to save them): the full recurrence recompute; ``light_states``
# the backward from a version-4 set carrying them (the segmented recurrence as
# pass 2 alone); ``tape`` the backward from a version-2 set (no recurrence
# recompute); ``replay`` the checkpoint replay (the forward's internal rows
# recomputed first).
PATHS = ("fwd", "fwd_nosave", "light", "light_states", "tape", "replay")

# The launch orders.  The forward launches the table (the serial route; the
# segmented route above 256 documents, where ``seg_table`` follows it; up to
# 256 documents the segmented route launches no table program: its intra
# program ``fwd_intra_v14`` computes the chunk layout from ``cu_seqlens`` and
# writes the chunk tables and the block table itself), the intra-chunk program
# (which also writes the GLM prep's outputs: the gated keys / queries, the gate
# row and the ``v_new`` pads; no separate prep launch) and then either the
# serial recurrence (with ``wy_fwd`` before it unless the recurrence forms the
# WY representation itself) or the segmented pass 1, ``seg_carry`` and pass 2
# (``fwdh_p3``).  The backward launches the prep (on the replay path the
# intra-chunk program too: before the prep where it writes the tables the prep
# reads, after it otherwise), ``wy`` and the recurrence recompute where the path and the
# row take them, ``dav`` where the fused gradient does not host it, the adjoint
# recurrence (segmented: pass 1, ``seg_carry``, pass 2), the fused gradient,
# the GLM intra program and, on ``sm_100a``, the gate epilogue (``sm_103a`` /
# ``sm_107a`` launch ``glm_intra_v14`` / ``glm_intra_v18``, which write the gate
# gradients themselves).  A stage with several variants
# launches the record its ``select_rule`` admits; an optional stage launches
# nothing when no record admits.  The WY representation is two stages:
# ``wy_fwd`` (the forward's program, on the user's ``v`` / ``beta`` token rows;
# on the saving forward's overlap rows and paired serial rows ``wy_fwd_rows``,
# the backward's program on the saved ``Akk`` rows) and ``wy`` (the backward's,
# on the saved or recomputed ``Akk`` rows).
STAGES_FWD = (
    "table",
    "seg_table",
    "fwd_intra",
    "wy_fwd",
    "fwdh",
    "fwdh_p3",
)
STAGES_BWD = (
    "bwd_prep",
    "wy",
    "fwdh",
    "fwdh_p3",
    "dav",
    "dhu",
    "dqkg",
    "glm_intra",
    "gate_epilogue",
)
STAGES = (
    "table",
    "seg_table",
    "fwd_intra",
    "wy_fwd",
    "wy",
    "fwdh",
    "seg_carry",
    "fwdh_p3",
    "bwd_prep",
    "dav",
    "dhu",
    "dqkg",
    "glm_intra",
    "gate_epilogue",
)
# The stages some routes or paths do not launch: :func:`select_record` returns
# ``None`` for them when no record admits the problem (``table`` on the forward
# and the replay of the segmented route up to 256 documents, where the intra
# program writes the tables; the segmented route's own stages on the serial
# route; ``wy_fwd`` / ``wy`` / ``fwdh`` / ``dav`` where a fused program or the
# saved set replaces them; ``gate_epilogue`` where ``glm_intra_v14`` fuses it).
OPTIONAL_STAGES = (
    "table",
    "seg_table",
    "seg_carry",
    "fwdh_p3",
    "wy_fwd",
    "wy",
    "fwdh",
    "dav",
    "gate_epilogue",
)

# The segmented route (the ``route_in`` rules read the route
# :func:`segment_plan` derives from these constants): with at most
# ``max_heads`` heads, or with a head count listed for the device's
# architecture in ``plan_heads_by_arch`` (the 16-head rows of ``sm_103a`` /
# ``sm_107a``: their serial walk is the slower form there), every document is cut into
# ``clamp(ceil(slot_factor * sm_count / (num_seqs * num_heads)), 1,
# max_segments)`` segments, so the segment walks fill about ``slot_factor / 2``
# waves of the device's SMs at two CTAs per SM, and capped at ``sm_count //
# num_heads`` so one document's walk CTAs fit one wave of the device (pass 2
# at one CTA per SM, pass 1 at two; 152 SMs: 38 segments at 4 heads, 19 at
# 8).  With at most ``small_max_chains`` (document, head) chains the count is
# capped at ``small_segments`` (``small_segments_by_arch`` names the
# architectures with a count of their own: 37 on ``sm_100a``): the carry chain
# then takes 37 (36) steps instead of 63, which outweighs the lost parallelism
# of the segment walks.  With at
# most ``prop_max_chains`` chains the block table gives every document its
# proportional share of the ``num_seqs * segments`` segments (``seg_table``'s
# ``proportional`` scalar; the walk programs then take ``-num_seqs`` as their
# ``seg_arg`` and read each chain's bounds from the table).
SEGMENTS: dict[str, Any] = {
    "max_heads": 8,
    "slot_factor": 4,
    "max_segments": 64,
    "small_segments": 38,
    "small_segments_by_arch": {"sm_100a": 37},
    "small_max_chains": 4,
    "prop_max_chains": 4,
    "plan_heads_by_arch": {"sm_103a": [16], "sm_107a": [16]},
}
# The two-level segment carry (``seg_carry_v8``: the segments of each
# document's chain in groups of ``gsize``, the groups' products formed in
# parallel and chained, the CTAs handing over through release-flag words
# stamped with the launch's ``epoch``) touches ``CARRY_FLAG_WORDS_PER_CTA``
# flag words per CTA (:func:`flag_words`); its host rule is
# :func:`carry_choice`.
CARRY_FLAG_WORDS_PER_CTA = 3

# Logical argument names the host backend binds; every record maps its kernel
# parameter names onto this vocabulary (``record["logical"]``).  ``aqk_src`` /
# ``akk_src`` are the saved ``Aqk`` / ``Akk`` token rows; ``aqk_int`` / ``akk_int``
# the forward's internal-layout copies (the replay path's sources); ``do_int``
# the replay prep's internal copy of ``dout``; ``h2d`` / ``dh2d`` the
# ``[NC * H * 128, 128]`` views of the ``h`` / ``dh`` tapes the TMA-described
# programs read; ``w_t`` / ``w_tk`` the ``w`` buffer viewed as the transposed
# per-chunk tile the fused recomputes write (``[NC, H, 128, 64]`` on the serial
# route, ``[NC * 128, H, 64]`` on the segmented one).  The segmented route adds
# the block table ``seg_table`` (``[num_segments + 2 * num_seqs + 1]``: the
# chains' first segments, then per chain its segment starts and end), the fp32
# ``[128, 128]`` state blocks per (segment, head): ``seg_state`` (``[2,
# num_segments, heads, 128, 128]``: the forward's segment offsets then
# transfers), ``seg_m`` (its transfer half as ``[num_segments * heads, 128,
# 128]``, both TMA operands of ``seg_carry``), ``hseg`` / ``dhseg`` (the carried
# forward / adjoint states) and ``dseg`` (the backward's segment offsets);
# ``carry_b`` / ``carry_out`` are ``seg_carry``'s offset input and carried-state
# output (``seg_state[0]`` -> ``hseg`` forward, ``dseg`` -> ``dhseg`` reverse);
# ``carry_out_tiles`` is ``carry_out`` viewed as ``[num_segments * heads, 128,
# 128]`` tiles (the two-level carry's TMA-described operand for the groups'
# products); ``eye`` is the fp32 identity the forward's transfer walk starts
# from and ``flags`` its words viewed as the release flags.  The
# backward's fused gradient and GLM intra kernel read ``q`` / ``k`` / ``v`` on
# the token rows (no chunk-layout copies); ``num_pairs`` is the chunk-pair
# count ``num_chunks_pad / 2`` the persistent pair kernels walk.
LOGICAL_TENSORS = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "cu_seqlens",
    "dout",
    "chunk_bos",
    "chunk_len",
    "seq_chunk_start",
    "seq_order",
    "gk",
    "kb",
    "kg",
    "qg",
    "v_new",
    "aqk_int",
    "akk_int",
    "aqk_ext",
    "akk_ext",
    "aqk_src",
    "akk_src",
    "u",
    "w",
    "w_t",
    "w_tk",
    "h",
    "h2d",
    "out",
    "do_int",
    "beta_int",
    "dv1",
    "dv2",
    "dAqk",
    "dAkk",
    "dh",
    "dh2d",
    "dq_f",
    "dk_f",
    "dg_f",
    "db_f",
    "dg_int",
    "db_int",
    "dq",
    "dk",
    "dv",
    "dg",
    "dbeta",
    "seg_table",
    "seg_state",
    "seg_m",
    "hseg",
    "dhseg",
    "dseg",
    "carry_b",
    "carry_out",
    "carry_out_tiles",
    "eye",
    "flags",
)
# ``segments`` is the per-document segment count ``seg_table`` expands;
# ``seg_arg`` the walk programs' segment scalar (``segments``, or
# ``-num_seqs`` under a proportional table); ``proportional`` that switch;
# ``write_prep`` the forward intra program's switch for the GLM prep's outputs
# (1 on the forward, 0 on the replay, whose prep writes them); ``epoch`` the
# release-flag epoch of the carry || pass-2 pair and of the two-level carry;
# ``gsize`` the two-level carry's segment group size (:func:`carry_choice`).
LOGICAL_SCALARS = (
    "num_seqs",
    "num_chunks_pad",
    "num_heads",
    "num_pairs",
    "group",
    "write_ext",
    "write_int",
    "write_prep",
    "scale",
    "num_segments",
    "segments",
    "seg_arg",
    "proportional",
    "reverse",
    "use_h0",
    "use_dht",
    "write_tapes",
    "write_ht",
    "epoch",
    "gsize",
)

# Filled mechanically by the exporter (the derived leaves ``kernel_symbol``, ``launch``, ``arg_plan``,
# ``logical``); the host plan (``stage``, ``variant``, ``select_rule``, ``grid_rule``, ``library``, ``arches``)
# is written by hand.  Every value is a literal: FlashInfer's manifest checker and the exporter read this
# file with ``ast.literal_eval``.  Record names are the stages, suffixed with the variant where a stage has
# several programs (``bwd_prep_g8`` keeps its variant name: the backward prep ships its ``g8`` form alone).
MODULES: dict[str, dict[str, Any]] = {
    "table": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "cu_seqlens"],
            ["parameter", "num_seqs"],
            ["parameter", "num_chunks_max"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "seq_chunk_start"],
            ["buffer", "seq_order"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "fixed", "ctas": [1, 1, 1]},
        "kernel_symbol": "kernel_cake_kda_train_d7656e2b58b9bab49495",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 1152,
            "use_pdl": False,
        },
        "library": "cake_kda_train_d7656e2b58b9bab49495",
        "logical": {
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "cu_seqlens": "cu_seqlens",
            "num_chunks_max": "num_chunks_pad",
            "num_seqs": "num_seqs",
            "seq_chunk_start": "seq_chunk_start",
            "seq_order": "seq_order",
        },
        "select_rule": {"kind": "docs_above", "min": 256},
        "stage": "table",
        "variant": None,
    },
    "table_v2": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "cu_seqlens"],
            ["parameter", "num_seqs"],
            ["parameter", "num_chunks_max"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "seq_chunk_start"],
            ["buffer", "seq_order"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "chunk_blocks", "per_cta": 256},
        "kernel_symbol": "kernel_cake_kda_train_8d99ea1be1c1fef432a5",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 1152,
            "use_pdl": False,
        },
        "library": "cake_kda_train_8d99ea1be1c1fef432a5",
        "logical": {
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "cu_seqlens": "cu_seqlens",
            "num_chunks_max": "num_chunks_pad",
            "num_seqs": "num_seqs",
            "seq_chunk_start": "seq_chunk_start",
            "seq_order": "seq_order",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "route_in", "routes": ["serial"]},
                {"kind": "docs_at_most", "max": 256},
            ],
        },
        "stage": "table",
        "variant": "v2",
    },
    "table_v3": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "cu_seqlens"],
            ["parameter", "num_seqs"],
            ["parameter", "num_chunks_max"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "seq_chunk_start"],
            ["buffer", "seq_order"],
            ["parameter", "segments"],
            ["parameter", "proportional"],
            ["buffer", "seg_table"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "chunk_blocks", "per_cta": 256},
        "kernel_symbol": "kernel_cake_kda_train_9f69ca35a631ae5e1b4c",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 5376,
            "use_pdl": False,
        },
        "library": "cake_kda_train_9f69ca35a631ae5e1b4c",
        "logical": {
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "cu_seqlens": "cu_seqlens",
            "num_chunks_max": "num_chunks_pad",
            "num_seqs": "num_seqs",
            "proportional": "proportional",
            "seg_table": "seg_table",
            "segments": "segments",
            "seq_chunk_start": "seq_chunk_start",
            "seq_order": "seq_order",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "route_in", "routes": ["seg"]},
                {"kind": "docs_at_most", "max": 256},
                {"kind": "path_in", "paths": ["light", "light_states", "tape"]},
            ],
        },
        "stage": "table",
        "variant": "v3",
    },
    "seg_table": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "seq_chunk_start"],
            ["buffer", "seq_order"],
            ["parameter", "num_seqs"],
            ["parameter", "segments"],
            ["parameter", "proportional"],
            ["buffer", "seg_table"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "fixed", "ctas": [1, 1, 1]},
        "kernel_symbol": "kernel_cake_kda_train_fd4c92787fcc18446b14",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 3200,
            "use_pdl": False,
        },
        "library": "cake_kda_train_fd4c92787fcc18446b14",
        "logical": {
            "num_seqs": "num_seqs",
            "proportional": "proportional",
            "seg_table": "seg_table",
            "segments": "segments",
            "seq_chunk_start": "seq_chunk_start",
            "seq_order": "seq_order",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "route_in", "routes": ["seg"]},
                {"kind": "docs_above", "min": 256},
            ],
        },
        "stage": "seg_table",
        "variant": None,
    },
    "fwd_intra": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "g"],
            ["buffer", "beta"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "aqk_ext"],
            ["buffer", "akk_ext"],
            ["buffer", "aqk_int"],
            ["buffer", "akk_int"],
            ["parameter", "num_heads"],
            ["parameter", "num_qk_heads"],
            ["parameter", "group"],
            ["parameter", "write_ext"],
            ["parameter", "write_int"],
            ["parameter", "scale"],
            ["buffer", "gk_out"],
            ["buffer", "kb_out"],
            ["buffer", "kg_out"],
            ["buffer", "qg_out"],
            ["buffer", "v_new"],
            ["parameter", "write_prep"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "chunks", "x_factor": 1},
        "kernel_symbol": "kernel_cake_kda_train_c9395e21a813579c4bfc",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 76160,
            "use_pdl": False,
        },
        "library": "cake_kda_train_c9395e21a813579c4bfc",
        "logical": {
            "akk_ext": "akk_ext",
            "akk_int": "akk_int",
            "aqk_ext": "aqk_ext",
            "aqk_int": "aqk_int",
            "beta": "beta",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "g": "g",
            "gk_out": "gk",
            "group": "group",
            "k": "k",
            "kb_out": "kb",
            "kg_out": "kg",
            "num_heads": "num_heads",
            "num_qk_heads": "num_heads",
            "q": "q",
            "qg_out": "qg",
            "scale": "scale",
            "v_new": "v_new",
            "write_ext": "write_ext",
            "write_int": "write_int",
            "write_prep": "write_prep",
        },
        "select_rule": {
            "kind": "any_of",
            "rules": [
                {"kind": "route_in", "routes": ["serial"]},
                {"kind": "docs_above", "min": 256},
            ],
        },
        "stage": "fwd_intra",
        "variant": None,
    },
    "fwd_intra_v14": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "g"],
            ["buffer", "beta"],
            ["buffer", "cu_seqlens"],
            ["parameter", "num_seqs"],
            ["buffer", "aqk_ext"],
            ["buffer", "akk_ext"],
            ["buffer", "aqk_int"],
            ["buffer", "akk_int"],
            ["parameter", "num_heads"],
            ["parameter", "num_qk_heads"],
            ["parameter", "group"],
            ["parameter", "write_ext"],
            ["parameter", "write_int"],
            ["parameter", "scale"],
            ["buffer", "gk_out"],
            ["buffer", "kb_out"],
            ["buffer", "kg_out"],
            ["buffer", "qg_out"],
            ["buffer", "v_new"],
            ["parameter", "write_prep"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "seq_chunk_start"],
            ["buffer", "seq_order"],
            ["parameter", "segments"],
            ["parameter", "proportional"],
            ["buffer", "seg_table"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "chunks", "x_factor": 1},
        "kernel_symbol": "kernel_cake_kda_train_94c7d1cc6c908d53e8c1",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 76160,
            "use_pdl": False,
        },
        "library": "cake_kda_train_94c7d1cc6c908d53e8c1",
        "logical": {
            "akk_ext": "akk_ext",
            "akk_int": "akk_int",
            "aqk_ext": "aqk_ext",
            "aqk_int": "aqk_int",
            "beta": "beta",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "cu_seqlens": "cu_seqlens",
            "g": "g",
            "gk_out": "gk",
            "group": "group",
            "k": "k",
            "kb_out": "kb",
            "kg_out": "kg",
            "num_heads": "num_heads",
            "num_qk_heads": "num_heads",
            "num_seqs": "num_seqs",
            "proportional": "proportional",
            "q": "q",
            "qg_out": "qg",
            "scale": "scale",
            "seg_table": "seg_table",
            "segments": "segments",
            "seq_chunk_start": "seq_chunk_start",
            "seq_order": "seq_order",
            "v_new": "v_new",
            "write_ext": "write_ext",
            "write_int": "write_int",
            "write_prep": "write_prep",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "route_in", "routes": ["seg"]},
                {"kind": "docs_at_most", "max": 256},
            ],
        },
        "stage": "fwd_intra",
        "variant": "v14",
    },
    "wy_fwd": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "akk_tma"],
            ["buffer", "v_tma"],
            ["buffer", "kb_tma"],
            ["buffer", "beta"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "u_out"],
            ["buffer", "w_out"],
            ["parameter", "num_heads"],
            ["parameter", "num_pairs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "pairs_bounded", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_kda_train_16034f9855aa6bf70440",
        "launch": {
            "block": (448, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197632,
            "use_pdl": False,
        },
        "library": "cake_kda_train_16034f9855aa6bf70440",
        "logical": {
            "akk_tma": "akk_int",
            "beta": "beta",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "kb_tma": "kb",
            "num_heads": "num_heads",
            "num_pairs": "num_pairs",
            "u_out": "u",
            "v_tma": "v",
            "w_out": "w",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {
                    "kind": "any_of",
                    "rules": [
                        {"kind": "heads_below", "max": 32},
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "arch_in", "arches": ["sm_107a"]},
                                {"kind": "heads_above", "min": 32},
                            ],
                        },
                    ],
                },
                {
                    "kind": "any_of",
                    "rules": [
                        {
                            "kind": "path_in",
                            "paths": [
                                "fwd_nosave",
                                "light",
                                "light_states",
                                "tape",
                                "replay",
                            ],
                        },
                        {
                            "kind": "any_of",
                            "rules": [
                                {
                                    "kind": "all_of",
                                    "rules": [
                                        {
                                            "kind": "arch_in",
                                            "arches": ["sm_100a", "sm_103a"],
                                        },
                                        {"kind": "walk_ctas_below", "waves": 3.0},
                                    ],
                                },
                                {
                                    "kind": "all_of",
                                    "rules": [
                                        {"kind": "arch_in", "arches": ["sm_107a"]},
                                        {"kind": "walk_ctas_below", "waves": 2.3},
                                    ],
                                },
                            ],
                        },
                        {"kind": "segments_heads_above", "min": 8192},
                    ],
                },
                {
                    "kind": "any_of",
                    "rules": [
                        {"kind": "path_in", "paths": ["fwd_nosave"]},
                        {"kind": "route_in", "routes": ["seg"]},
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "slots_above", "slices": 4, "ctas_per_sm": 4},
                                {
                                    "kind": "any_of",
                                    "rules": [
                                        {"kind": "path_in", "paths": ["fwd_nosave"]},
                                        {
                                            "kind": "arch_in",
                                            "arches": ["sm_100a", "sm_103a"],
                                        },
                                    ],
                                },
                            ],
                        },
                    ],
                },
            ],
        },
        "stage": "wy_fwd",
        "variant": None,
    },
    "wy_fwd_rows": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "akk_tma"],
            ["buffer", "v_tma"],
            ["buffer", "kb_tma"],
            ["buffer", "beta"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "u_out"],
            ["buffer", "w_out"],
            ["parameter", "num_heads"],
            ["parameter", "num_pairs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "pairs_bounded", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_kda_train_9f65ab7adc0a39a322ca",
        "launch": {
            "block": (448, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197632,
            "use_pdl": False,
        },
        "library": "cake_kda_train_9f65ab7adc0a39a322ca",
        "logical": {
            "akk_tma": "akk_ext",
            "beta": "beta",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "kb_tma": "kb",
            "num_heads": "num_heads",
            "num_pairs": "num_pairs",
            "u_out": "u",
            "v_tma": "v",
            "w_out": "w",
        },
        "select_rule": {
            "kind": "any_of",
            "rules": [
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "route_in", "routes": ["seg"]},
                        {"kind": "path_in", "paths": ["fwd"]},
                        {
                            "kind": "any_of",
                            "rules": [
                                {
                                    "kind": "all_of",
                                    "rules": [
                                        {
                                            "kind": "arch_in",
                                            "arches": ["sm_100a", "sm_103a"],
                                        },
                                        {"kind": "walk_ctas_at_least", "waves": 3.0},
                                    ],
                                },
                                {
                                    "kind": "all_of",
                                    "rules": [
                                        {"kind": "arch_in", "arches": ["sm_107a"]},
                                        {"kind": "walk_ctas_at_least", "waves": 2.3},
                                    ],
                                },
                            ],
                        },
                        {"kind": "segments_heads_at_most", "max": 8192},
                    ],
                },
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "path_in", "paths": ["fwd"]},
                        {"kind": "route_in", "routes": ["serial"]},
                        {"kind": "slots_at_most", "slices": 4, "ctas_per_sm": 4},
                        {
                            "kind": "any_of",
                            "rules": [
                                {"kind": "heads_below", "max": 32},
                                {
                                    "kind": "all_of",
                                    "rules": [
                                        {"kind": "arch_in", "arches": ["sm_107a"]},
                                        {"kind": "heads_above", "min": 32},
                                    ],
                                },
                            ],
                        },
                    ],
                },
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "path_in", "paths": ["fwd"]},
                        {"kind": "route_in", "routes": ["serial"]},
                        {"kind": "slots_above", "slices": 4, "ctas_per_sm": 4},
                        {"kind": "arch_in", "arches": ["sm_107a"]},
                        {
                            "kind": "any_of",
                            "rules": [
                                {"kind": "heads_below", "max": 32},
                                {
                                    "kind": "all_of",
                                    "rules": [
                                        {"kind": "arch_in", "arches": ["sm_107a"]},
                                        {"kind": "heads_above", "min": 32},
                                    ],
                                },
                            ],
                        },
                    ],
                },
            ],
        },
        "stage": "wy_fwd",
        "variant": "rows",
    },
    "wy_bwd": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "akk_tma"],
            ["buffer", "v_tma"],
            ["buffer", "kb_tma"],
            ["buffer", "beta"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "u_out"],
            ["buffer", "w_out"],
            ["parameter", "num_heads"],
            ["parameter", "num_pairs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "pairs_bounded", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_kda_train_9f65ab7adc0a39a322ca",
        "launch": {
            "block": (448, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197632,
            "use_pdl": False,
        },
        "library": "cake_kda_train_9f65ab7adc0a39a322ca",
        "logical": {
            "akk_tma": "akk_src",
            "beta": "beta",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "kb_tma": "kb",
            "num_heads": "num_heads",
            "num_pairs": "num_pairs",
            "u_out": "u",
            "v_tma": "v",
            "w_out": "w",
        },
        "select_rule": {
            "kind": "any_of",
            "rules": [
                {"kind": "path_in", "paths": ["tape"]},
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "path_in", "paths": ["light"]},
                        {
                            "kind": "any_of",
                            "rules": [
                                {"kind": "heads_below", "max": 64},
                                {
                                    "kind": "slots_at_most",
                                    "slices": 4,
                                    "ctas_per_sm": 4,
                                },
                            ],
                        },
                        {
                            "kind": "any_of",
                            "rules": [
                                {"kind": "route_in", "routes": ["seg"]},
                                {"kind": "heads_below", "max": 16},
                                {"kind": "heads_above", "min": 16},
                                {"kind": "slots_above", "slices": 4, "ctas_per_sm": 4},
                                {"kind": "arch_in", "arches": ["sm_103a", "sm_107a"]},
                            ],
                        },
                        {
                            "kind": "any_of",
                            "rules": [
                                {"kind": "route_in", "routes": ["serial"]},
                                {"kind": "heads_below", "max": 16},
                                {"kind": "heads_above", "min": 16},
                                {"kind": "chains_at_most", "max": 32},
                                {"kind": "arch_in", "arches": ["sm_100a"]},
                            ],
                        },
                    ],
                },
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "path_in", "paths": ["light_states"]},
                        {"kind": "chains_above", "min": 32},
                        {
                            "kind": "any_of",
                            "rules": [
                                {"kind": "route_in", "routes": ["serial"]},
                                {"kind": "heads_below", "max": 16},
                                {"kind": "heads_above", "min": 16},
                                {"kind": "chains_at_most", "max": 32},
                                {"kind": "arch_in", "arches": ["sm_100a"]},
                            ],
                        },
                    ],
                },
            ],
        },
        "stage": "wy",
        "variant": "bwd",
    },
    "wy_bwd_replay": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "akk_tma"],
            ["buffer", "v_tma"],
            ["buffer", "kb_tma"],
            ["buffer", "beta"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "u_out"],
            ["buffer", "w_out"],
            ["parameter", "num_heads"],
            ["parameter", "num_pairs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "pairs_bounded", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_kda_train_18bd735cf970f8dfa1b4",
        "launch": {
            "block": (448, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197632,
            "use_pdl": False,
        },
        "library": "cake_kda_train_18bd735cf970f8dfa1b4",
        "logical": {
            "akk_tma": "akk_int",
            "beta": "beta",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "kb_tma": "kb",
            "num_heads": "num_heads",
            "num_pairs": "num_pairs",
            "u_out": "u",
            "v_tma": "v",
            "w_out": "w",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["replay"]},
                {
                    "kind": "any_of",
                    "rules": [
                        {"kind": "heads_below", "max": 64},
                        {"kind": "slots_at_most", "slices": 4, "ctas_per_sm": 4},
                    ],
                },
            ],
        },
        "stage": "wy",
        "variant": "bwd_replay",
    },
    "fwdh_o_v1": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["buffer", "seq_order"],
            ["buffer", "qg_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "o_out"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["parameter", "scale"],
            ["parameter", "write_tapes"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_doc_head", "slices": 2},
        "kernel_symbol": "kernel_cake_kda_train_5fcc76904e5e2dd891de",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_kda_train_5fcc76904e5e2dd891de",
        "logical": {
            "aqk_tma": "aqk_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "gk": "gk",
            "h_out": "h",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "o_out": "out",
            "qg_tma": "qg",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "seq_order": "seq_order",
            "u": "u",
            "v_new": "v_new",
            "w_tma": "w",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {
                    "kind": "any_of",
                    "rules": [
                        {"kind": "path_in", "paths": ["fwd_nosave"]},
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "path_in", "paths": ["fwd"]},
                                {"kind": "arch_in", "arches": ["sm_100a", "sm_103a"]},
                            ],
                        },
                    ],
                },
                {"kind": "route_in", "routes": ["serial"]},
                {"kind": "slots_above", "slices": 4, "ctas_per_sm": 4},
                {
                    "kind": "any_of",
                    "rules": [
                        {"kind": "heads_below", "max": 32},
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "arch_in", "arches": ["sm_107a"]},
                                {"kind": "heads_above", "min": 32},
                            ],
                        },
                    ],
                },
            ],
        },
        "stage": "fwdh",
        "variant": "o_v1",
    },
    "fwdh_o_v1_rows": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["buffer", "seq_order"],
            ["buffer", "qg_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "o_out"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["parameter", "scale"],
            ["parameter", "write_tapes"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_doc_head", "slices": 2},
        "kernel_symbol": "kernel_cake_kda_train_f011cd9add8f3c7c3eee",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_kda_train_f011cd9add8f3c7c3eee",
        "logical": {
            "aqk_tma": "aqk_ext",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "gk": "gk",
            "h_out": "h",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "o_out": "out",
            "qg_tma": "qg",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "seq_order": "seq_order",
            "u": "u",
            "v_new": "v_new",
            "w_tma": "w",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["fwd"]},
                {"kind": "route_in", "routes": ["serial"]},
                {"kind": "slots_above", "slices": 4, "ctas_per_sm": 4},
                {"kind": "arch_in", "arches": ["sm_107a"]},
                {
                    "kind": "any_of",
                    "rules": [
                        {"kind": "heads_below", "max": 32},
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "arch_in", "arches": ["sm_107a"]},
                                {"kind": "heads_above", "min": 32},
                            ],
                        },
                    ],
                },
            ],
        },
        "stage": "fwdh",
        "variant": "o_v1_rows",
    },
    "fwdh_o_v2": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["buffer", "seq_order"],
            ["buffer", "qg_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "o_out"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["parameter", "scale"],
            ["parameter", "write_tapes"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_doc_head", "slices": 4},
        "kernel_symbol": "kernel_cake_kda_train_4cce3cc5bc6210c95ce3",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 103424,
            "use_pdl": False,
        },
        "library": "cake_kda_train_4cce3cc5bc6210c95ce3",
        "logical": {
            "aqk_tma": "aqk_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "gk": "gk",
            "h_out": "h",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "o_out": "out",
            "qg_tma": "qg",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "seq_order": "seq_order",
            "u": "u",
            "v_new": "v_new",
            "w_tma": "w",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["fwd_nosave"]},
                {"kind": "route_in", "routes": ["serial"]},
                {"kind": "slots_at_most", "slices": 4, "ctas_per_sm": 4},
                {
                    "kind": "any_of",
                    "rules": [
                        {"kind": "heads_below", "max": 32},
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "arch_in", "arches": ["sm_107a"]},
                                {"kind": "heads_above", "min": 32},
                            ],
                        },
                    ],
                },
            ],
        },
        "stage": "fwdh",
        "variant": "o_v2",
    },
    "fwdh_o_v2_rows": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["buffer", "seq_order"],
            ["buffer", "qg_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "o_out"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["parameter", "scale"],
            ["parameter", "write_tapes"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_doc_head", "slices": 4},
        "kernel_symbol": "kernel_cake_kda_train_3f4b037a0a1f8dcc314b",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 103424,
            "use_pdl": False,
        },
        "library": "cake_kda_train_3f4b037a0a1f8dcc314b",
        "logical": {
            "aqk_tma": "aqk_ext",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "gk": "gk",
            "h_out": "h",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "o_out": "out",
            "qg_tma": "qg",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "seq_order": "seq_order",
            "u": "u",
            "v_new": "v_new",
            "w_tma": "w",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["fwd"]},
                {"kind": "route_in", "routes": ["serial"]},
                {"kind": "slots_at_most", "slices": 4, "ctas_per_sm": 4},
                {
                    "kind": "any_of",
                    "rules": [
                        {"kind": "heads_below", "max": 32},
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "arch_in", "arches": ["sm_107a"]},
                                {"kind": "heads_above", "min": 32},
                            ],
                        },
                    ],
                },
            ],
        },
        "stage": "fwdh",
        "variant": "o_v2_rows",
    },
    "fwdh_wy": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "akk_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "kb_tma"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["buffer", "seq_order"],
            ["buffer", "qg_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "o_out"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["parameter", "scale"],
            ["parameter", "write_tapes"],
            ["buffer", "v_tma"],
            ["buffer", "beta"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_doc_head", "slices": 2},
        "kernel_symbol": "kernel_cake_kda_train_a557058a8aeca49a6402",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 181248,
            "use_pdl": False,
        },
        "library": "cake_kda_train_a557058a8aeca49a6402",
        "logical": {
            "akk_tma": "akk_int",
            "aqk_tma": "aqk_int",
            "beta": "beta",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "gk": "gk",
            "h_out": "h",
            "kb_tma": "kb",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "o_out": "out",
            "qg_tma": "qg",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "seq_order": "seq_order",
            "v_new": "v_new",
            "v_tma": "v",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["fwd_nosave"]},
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "heads_at_least", "min": 32},
                        {
                            "kind": "any_of",
                            "rules": [
                                {"kind": "arch_in", "arches": ["sm_100a", "sm_103a"]},
                                {"kind": "heads_at_most", "max": 32},
                            ],
                        },
                    ],
                },
            ],
        },
        "stage": "fwdh",
        "variant": "wy",
    },
    "fwdh_wy_rows": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "akk_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "kb_tma"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["buffer", "seq_order"],
            ["buffer", "qg_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "o_out"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["parameter", "scale"],
            ["parameter", "write_tapes"],
            ["buffer", "v_tma"],
            ["buffer", "beta"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_doc_head", "slices": 2},
        "kernel_symbol": "kernel_cake_kda_train_898a1eb972fbaf55789a",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 181248,
            "use_pdl": False,
        },
        "library": "cake_kda_train_898a1eb972fbaf55789a",
        "logical": {
            "akk_tma": "akk_ext",
            "aqk_tma": "aqk_ext",
            "beta": "beta",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "gk": "gk",
            "h_out": "h",
            "kb_tma": "kb",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "o_out": "out",
            "qg_tma": "qg",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "seq_order": "seq_order",
            "v_new": "v_new",
            "v_tma": "v",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["fwd"]},
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "heads_at_least", "min": 32},
                        {
                            "kind": "any_of",
                            "rules": [
                                {"kind": "arch_in", "arches": ["sm_100a", "sm_103a"]},
                                {"kind": "heads_at_most", "max": 32},
                            ],
                        },
                    ],
                },
            ],
        },
        "stage": "fwdh",
        "variant": "wy_rows",
    },
    "fwdh_v3": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "segments"],
            ["buffer", "h0"],
            ["buffer", "eye"],
            ["buffer", "ht_out"],
            ["parameter", "use_h0"],
            ["parameter", "write_tapes"],
            ["parameter", "write_ht"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "segments_heads", "passes": [2]},
        "kernel_symbol": "kernel_cake_kda_train_0e5fa2fe9ae292c3e3aa",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_kda_train_0e5fa2fe9ae292c3e3aa",
        "logical": {
            "eye": "eye",
            "gk": "gk",
            "h0": "hseg",
            "h_out": "h",
            "ht_out": "seg_state",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_segments",
            "segments": "seg_arg",
            "seq_chunk_start": "seg_table",
            "u": "u",
            "use_h0": "use_h0",
            "v_new": "v_new",
            "w_tma": "w",
            "write_ht": "write_ht",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "route_in", "routes": ["seg"]},
                {"kind": "path_in", "paths": ["fwd", "fwd_nosave", "light", "replay"]},
            ],
        },
        "stage": "fwdh",
        "variant": "v3",
    },
    "fwdh_v1": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["buffer", "seq_order"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_doc_head", "slices": 2},
        "kernel_symbol": "kernel_cake_kda_train_90153af20fdef34c6cb3",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 91136,
            "use_pdl": False,
        },
        "library": "cake_kda_train_90153af20fdef34c6cb3",
        "logical": {
            "gk": "gk",
            "h_out": "h",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "seq_chunk_start": "seq_chunk_start",
            "seq_order": "seq_order",
            "u": "u",
            "v_new": "v_new",
            "w_tma": "w",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["light", "replay"]},
                {"kind": "route_in", "routes": ["serial"]},
                {"kind": "heads_below", "max": 64},
                {"kind": "slots_above", "slices": 4, "ctas_per_sm": 4},
                {"kind": "arch_in", "arches": ["sm_100a", "sm_103a"]},
            ],
        },
        "stage": "fwdh",
        "variant": "v1",
    },
    "fwdh_v1_sm107": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["buffer", "seq_order"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_doc_head", "slices": 2},
        "kernel_symbol": "kernel_cake_kda_train_612492edb453070b0d6a",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 91136,
            "use_pdl": False,
        },
        "library": "cake_kda_train_612492edb453070b0d6a",
        "logical": {
            "gk": "gk",
            "h_out": "h",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "seq_chunk_start": "seq_chunk_start",
            "seq_order": "seq_order",
            "u": "u",
            "v_new": "v_new",
            "w_tma": "w",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["light", "replay"]},
                {"kind": "route_in", "routes": ["serial"]},
                {"kind": "heads_below", "max": 64},
                {"kind": "slots_above", "slices": 4, "ctas_per_sm": 4},
                {"kind": "arch_in", "arches": ["sm_107a"]},
            ],
        },
        "stage": "fwdh",
        "variant": "v1_sm107",
    },
    "fwdh_v2": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["buffer", "seq_order"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_doc_head", "slices": 4},
        "kernel_symbol": "kernel_cake_kda_train_133d787e8f202007cd37",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 78848,
            "use_pdl": False,
        },
        "library": "cake_kda_train_133d787e8f202007cd37",
        "logical": {
            "gk": "gk",
            "h_out": "h",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "seq_chunk_start": "seq_chunk_start",
            "seq_order": "seq_order",
            "u": "u",
            "v_new": "v_new",
            "w_tma": "w",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["light", "replay"]},
                {"kind": "route_in", "routes": ["serial"]},
                {"kind": "slots_at_most", "slices": 4, "ctas_per_sm": 4},
                {
                    "kind": "any_of",
                    "rules": [
                        {"kind": "heads_below", "max": 16},
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "arch_in", "arches": ["sm_107a"]},
                                {"kind": "heads_above", "min": 16},
                            ],
                        },
                    ],
                },
            ],
        },
        "stage": "fwdh",
        "variant": "v2",
    },
    "fwdh_v2h": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["buffer", "seq_order"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_doc_head", "slices": 4},
        "kernel_symbol": "kernel_cake_kda_train_827dd9ceb9a75ea0330c",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 78848,
            "use_pdl": False,
        },
        "library": "cake_kda_train_827dd9ceb9a75ea0330c",
        "logical": {
            "gk": "gk",
            "h_out": "h",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "seq_chunk_start": "seq_chunk_start",
            "seq_order": "seq_order",
            "u": "u",
            "v_new": "v_new",
            "w_tma": "w",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["light", "replay"]},
                {"kind": "route_in", "routes": ["serial"]},
                {"kind": "slots_at_most", "slices": 4, "ctas_per_sm": 4},
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "heads_at_least", "min": 16},
                        {
                            "kind": "any_of",
                            "rules": [
                                {"kind": "arch_in", "arches": ["sm_100a", "sm_103a"]},
                                {"kind": "heads_at_most", "max": 16},
                            ],
                        },
                    ],
                },
            ],
        },
        "stage": "fwdh",
        "variant": "v2h",
    },
    "fwdh_wy_bwd": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "akk_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "kb_tma"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["buffer", "seq_order"],
            ["buffer", "v_tma"],
            ["buffer", "beta"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "w_out"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_doc_head", "slices": 2},
        "kernel_symbol": "kernel_cake_kda_train_cab3e194b7e9e7557edf",
        "launch": {
            "block": (352, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 107520,
            "use_pdl": False,
        },
        "library": "cake_kda_train_cab3e194b7e9e7557edf",
        "logical": {
            "akk_tma": "akk_src",
            "beta": "beta",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "gk": "gk",
            "h_out": "h",
            "kb_tma": "kb",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "seq_chunk_start": "seq_chunk_start",
            "seq_order": "seq_order",
            "v_new": "v_new",
            "v_tma": "v",
            "w_out": "w_t",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["light"]},
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "heads_at_least", "min": 64},
                        {"kind": "slots_above", "slices": 4, "ctas_per_sm": 4},
                    ],
                },
            ],
        },
        "stage": "fwdh",
        "variant": "wy_bwd",
    },
    "fwdh_wy_bwd_replay": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "akk_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "kb_tma"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["buffer", "seq_order"],
            ["buffer", "v_tma"],
            ["buffer", "beta"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "w_out"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_doc_head", "slices": 2},
        "kernel_symbol": "kernel_cake_kda_train_b2e877c538b6ab7317e9",
        "launch": {
            "block": (352, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 107520,
            "use_pdl": False,
        },
        "library": "cake_kda_train_b2e877c538b6ab7317e9",
        "logical": {
            "akk_tma": "akk_int",
            "beta": "beta",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "gk": "gk",
            "h_out": "h",
            "kb_tma": "kb",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "seq_chunk_start": "seq_chunk_start",
            "seq_order": "seq_order",
            "v_new": "v_new",
            "v_tma": "v",
            "w_out": "w_t",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["replay"]},
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "heads_at_least", "min": 64},
                        {"kind": "slots_above", "slices": 4, "ctas_per_sm": 4},
                    ],
                },
            ],
        },
        "stage": "fwdh",
        "variant": "wy_bwd_replay",
    },
    "seg_carry": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "m_k_tma"],
            ["buffer", "m_mn_tma"],
            ["buffer", "b"],
            ["buffer", "h_out"],
            ["parameter", "num_heads"],
            ["buffer", "chain_seg_start"],
            ["parameter", "reverse"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "docs_heads", "ctas_per_chain": 2},
        "kernel_symbol": "kernel_cake_kda_train_184d5ccfffea932ff1d8",
        "launch": {
            "block": (576, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197632,
            "use_pdl": False,
        },
        "library": "cake_kda_train_184d5ccfffea932ff1d8",
        "logical": {
            "b": "carry_b",
            "chain_seg_start": "seg_table",
            "h_out": "carry_out",
            "m_k_tma": "seg_m",
            "m_mn_tma": "seg_m",
            "num_heads": "num_heads",
            "reverse": "reverse",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "route_in", "routes": ["seg"]},
                {
                    "kind": "any_of",
                    "rules": [
                        {
                            "kind": "path_in",
                            "paths": ["light", "light_states", "tape", "replay"],
                        },
                        {
                            "kind": "any_of",
                            "rules": [
                                {
                                    "kind": "all_of",
                                    "rules": [
                                        {
                                            "kind": "arch_in",
                                            "arches": ["sm_100a", "sm_103a"],
                                        },
                                        {"kind": "walk_ctas_below", "waves": 3.0},
                                    ],
                                },
                                {
                                    "kind": "all_of",
                                    "rules": [
                                        {"kind": "arch_in", "arches": ["sm_107a"]},
                                        {"kind": "walk_ctas_below", "waves": 2.3},
                                    ],
                                },
                            ],
                        },
                        {"kind": "segments_heads_above", "min": 8192},
                    ],
                },
                {
                    "kind": "any_of",
                    "rules": [
                        {"kind": "heads_above", "min": 2},
                        {"kind": "carry_chain", "s_hops": 14},
                    ],
                },
            ],
        },
        "stage": "seg_carry",
        "variant": None,
    },
    "seg_carry_ov": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "m_k_tma"],
            ["buffer", "m_mn_tma"],
            ["buffer", "b"],
            ["buffer", "h_out"],
            ["parameter", "num_heads"],
            ["buffer", "chain_seg_start"],
            ["parameter", "reverse"],
            ["buffer", "flags"],
            ["parameter", "epoch"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "docs_heads", "ctas_per_chain": 2},
        "kernel_symbol": "kernel_cake_kda_train_0fb763e9a81e9066d776",
        "launch": {
            "block": (608, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197632,
            "use_pdl": False,
        },
        "library": "cake_kda_train_0fb763e9a81e9066d776",
        "logical": {
            "b": "carry_b",
            "chain_seg_start": "seg_table",
            "epoch": "epoch",
            "flags": "flags",
            "h_out": "carry_out",
            "m_k_tma": "seg_m",
            "m_mn_tma": "seg_m",
            "num_heads": "num_heads",
            "reverse": "reverse",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "route_in", "routes": ["seg"]},
                {"kind": "path_in", "paths": ["fwd", "fwd_nosave"]},
                {
                    "kind": "any_of",
                    "rules": [
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "arch_in", "arches": ["sm_100a", "sm_103a"]},
                                {"kind": "walk_ctas_at_least", "waves": 3.0},
                            ],
                        },
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "arch_in", "arches": ["sm_107a"]},
                                {"kind": "walk_ctas_at_least", "waves": 2.3},
                            ],
                        },
                    ],
                },
                {"kind": "segments_heads_at_most", "max": 8192},
            ],
        },
        "stage": "seg_carry",
        "variant": "ov",
    },
    "seg_carry_v8": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "m_k_tma"],
            ["buffer", "m_mn_tma"],
            ["buffer", "b"],
            ["buffer", "h_out"],
            ["buffer", "pi_k_tma"],
            ["parameter", "num_heads"],
            ["buffer", "chain_seg_start"],
            ["parameter", "reverse"],
            ["parameter", "num_chains"],
            ["parameter", "gsize"],
            ["buffer", "flags"],
            ["parameter", "epoch"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "carry_groups", "s_hops": 14},
        "kernel_symbol": "kernel_cake_kda_train_e244f5d7d0e0569a62c1",
        "launch": {
            "block": (608, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197632,
            "use_pdl": False,
        },
        "library": "cake_kda_train_e244f5d7d0e0569a62c1",
        "logical": {
            "b": "carry_b",
            "chain_seg_start": "seg_table",
            "epoch": "epoch",
            "flags": "flags",
            "gsize": "gsize",
            "h_out": "carry_out",
            "m_k_tma": "seg_m",
            "m_mn_tma": "seg_m",
            "num_chains": "num_seqs",
            "num_heads": "num_heads",
            "pi_k_tma": "carry_out_tiles",
            "reverse": "reverse",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "route_in", "routes": ["seg"]},
                {
                    "kind": "any_of",
                    "rules": [
                        {
                            "kind": "path_in",
                            "paths": ["light", "light_states", "tape", "replay"],
                        },
                        {
                            "kind": "any_of",
                            "rules": [
                                {
                                    "kind": "all_of",
                                    "rules": [
                                        {
                                            "kind": "arch_in",
                                            "arches": ["sm_100a", "sm_103a"],
                                        },
                                        {"kind": "walk_ctas_below", "waves": 3.0},
                                    ],
                                },
                                {
                                    "kind": "all_of",
                                    "rules": [
                                        {"kind": "arch_in", "arches": ["sm_107a"]},
                                        {"kind": "walk_ctas_below", "waves": 2.3},
                                    ],
                                },
                            ],
                        },
                        {"kind": "segments_heads_above", "min": 8192},
                    ],
                },
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "heads_at_most", "max": 2},
                        {"kind": "carry_two_level", "s_hops": 14},
                    ],
                },
            ],
        },
        "stage": "seg_carry",
        "variant": "v8",
    },
    "fwdh_p3_o": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "segments"],
            ["buffer", "h0"],
            ["buffer", "eye"],
            ["buffer", "ht_out"],
            ["parameter", "use_h0"],
            ["parameter", "write_tapes"],
            ["parameter", "write_ht"],
            ["buffer", "qg_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "o_out"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["parameter", "scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "segments_heads", "passes": [1]},
        "kernel_symbol": "kernel_cake_kda_train_9931cdff5dfd86c687c7",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 140288,
            "use_pdl": False,
        },
        "library": "cake_kda_train_9931cdff5dfd86c687c7",
        "logical": {
            "aqk_tma": "aqk_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "eye": "eye",
            "gk": "gk",
            "h0": "hseg",
            "h_out": "h",
            "ht_out": "seg_state",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_segments",
            "o_out": "out",
            "qg_tma": "qg",
            "scale": "scale",
            "segments": "seg_arg",
            "seq_chunk_start": "seg_table",
            "u": "u",
            "use_h0": "use_h0",
            "v_new": "v_new",
            "w_tma": "w",
            "write_ht": "write_ht",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "route_in", "routes": ["seg"]},
                {"kind": "path_in", "paths": ["fwd", "fwd_nosave"]},
                {
                    "kind": "any_of",
                    "rules": [
                        {
                            "kind": "path_in",
                            "paths": ["light", "light_states", "tape", "replay"],
                        },
                        {
                            "kind": "any_of",
                            "rules": [
                                {
                                    "kind": "all_of",
                                    "rules": [
                                        {
                                            "kind": "arch_in",
                                            "arches": ["sm_100a", "sm_103a"],
                                        },
                                        {"kind": "walk_ctas_below", "waves": 3.0},
                                    ],
                                },
                                {
                                    "kind": "all_of",
                                    "rules": [
                                        {"kind": "arch_in", "arches": ["sm_107a"]},
                                        {"kind": "walk_ctas_below", "waves": 2.3},
                                    ],
                                },
                            ],
                        },
                        {"kind": "segments_heads_above", "min": 8192},
                    ],
                },
            ],
        },
        "stage": "fwdh_p3",
        "variant": "o",
    },
    "fwdh_p3_o_ov": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "segments"],
            ["buffer", "h0"],
            ["buffer", "flags"],
            ["buffer", "ht_out"],
            ["parameter", "use_h0"],
            ["parameter", "write_tapes"],
            ["parameter", "write_ht"],
            ["buffer", "qg_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "o_out"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["parameter", "scale"],
            ["parameter", "epoch"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "segments_heads", "passes": [1]},
        "kernel_symbol": "kernel_cake_kda_train_8ab0929b8154c90b5eff",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 140288,
            "use_pdl": True,
        },
        "library": "cake_kda_train_8ab0929b8154c90b5eff",
        "logical": {
            "aqk_tma": "aqk_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "epoch": "epoch",
            "flags": "flags",
            "gk": "gk",
            "h0": "hseg",
            "h_out": "h",
            "ht_out": "seg_state",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_segments",
            "o_out": "out",
            "qg_tma": "qg",
            "scale": "scale",
            "segments": "seg_arg",
            "seq_chunk_start": "seg_table",
            "u": "u",
            "use_h0": "use_h0",
            "v_new": "v_new",
            "w_tma": "w",
            "write_ht": "write_ht",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "route_in", "routes": ["seg"]},
                        {"kind": "path_in", "paths": ["fwd_nosave"]},
                        {
                            "kind": "any_of",
                            "rules": [
                                {
                                    "kind": "all_of",
                                    "rules": [
                                        {
                                            "kind": "arch_in",
                                            "arches": ["sm_100a", "sm_103a"],
                                        },
                                        {"kind": "walk_ctas_at_least", "waves": 3.0},
                                    ],
                                },
                                {
                                    "kind": "all_of",
                                    "rules": [
                                        {"kind": "arch_in", "arches": ["sm_107a"]},
                                        {"kind": "walk_ctas_at_least", "waves": 2.3},
                                    ],
                                },
                            ],
                        },
                        {"kind": "segments_heads_at_most", "max": 8192},
                    ],
                },
                {"kind": "heads_at_most", "max": 2},
            ],
        },
        "stage": "fwdh_p3",
        "variant": "o_ov",
    },
    "fwdh_p3_o_ov_id": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "segments"],
            ["buffer", "h0"],
            ["buffer", "flags"],
            ["buffer", "ht_out"],
            ["parameter", "use_h0"],
            ["parameter", "write_tapes"],
            ["parameter", "write_ht"],
            ["buffer", "qg_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "o_out"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["parameter", "scale"],
            ["parameter", "epoch"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "segments_heads", "passes": [1]},
        "kernel_symbol": "kernel_cake_kda_train_22812d87c26b53a1ba10",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 140288,
            "use_pdl": True,
        },
        "library": "cake_kda_train_22812d87c26b53a1ba10",
        "logical": {
            "aqk_tma": "aqk_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "epoch": "epoch",
            "flags": "flags",
            "gk": "gk",
            "h0": "hseg",
            "h_out": "h",
            "ht_out": "seg_state",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_segments",
            "o_out": "out",
            "qg_tma": "qg",
            "scale": "scale",
            "segments": "seg_arg",
            "seq_chunk_start": "seg_table",
            "u": "u",
            "use_h0": "use_h0",
            "v_new": "v_new",
            "w_tma": "w",
            "write_ht": "write_ht",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "route_in", "routes": ["seg"]},
                        {"kind": "path_in", "paths": ["fwd_nosave"]},
                        {
                            "kind": "any_of",
                            "rules": [
                                {
                                    "kind": "all_of",
                                    "rules": [
                                        {
                                            "kind": "arch_in",
                                            "arches": ["sm_100a", "sm_103a"],
                                        },
                                        {"kind": "walk_ctas_at_least", "waves": 3.0},
                                    ],
                                },
                                {
                                    "kind": "all_of",
                                    "rules": [
                                        {"kind": "arch_in", "arches": ["sm_107a"]},
                                        {"kind": "walk_ctas_at_least", "waves": 2.3},
                                    ],
                                },
                            ],
                        },
                        {"kind": "segments_heads_at_most", "max": 8192},
                    ],
                },
                {"kind": "heads_above", "min": 2},
            ],
        },
        "stage": "fwdh_p3",
        "variant": "o_ov_id",
    },
    "fwdh_p3_o_ov_rows": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "segments"],
            ["buffer", "h0"],
            ["buffer", "flags"],
            ["buffer", "ht_out"],
            ["parameter", "use_h0"],
            ["parameter", "write_tapes"],
            ["parameter", "write_ht"],
            ["buffer", "qg_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "o_out"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["parameter", "scale"],
            ["parameter", "epoch"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "segments_heads", "passes": [1]},
        "kernel_symbol": "kernel_cake_kda_train_7eca60b36e233943dae6",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 140288,
            "use_pdl": True,
        },
        "library": "cake_kda_train_7eca60b36e233943dae6",
        "logical": {
            "aqk_tma": "aqk_ext",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "epoch": "epoch",
            "flags": "flags",
            "gk": "gk",
            "h0": "hseg",
            "h_out": "h",
            "ht_out": "seg_state",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_segments",
            "o_out": "out",
            "qg_tma": "qg",
            "scale": "scale",
            "segments": "seg_arg",
            "seq_chunk_start": "seg_table",
            "u": "u",
            "use_h0": "use_h0",
            "v_new": "v_new",
            "w_tma": "w",
            "write_ht": "write_ht",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "route_in", "routes": ["seg"]},
                        {"kind": "path_in", "paths": ["fwd"]},
                        {
                            "kind": "any_of",
                            "rules": [
                                {
                                    "kind": "all_of",
                                    "rules": [
                                        {
                                            "kind": "arch_in",
                                            "arches": ["sm_100a", "sm_103a"],
                                        },
                                        {"kind": "walk_ctas_at_least", "waves": 3.0},
                                    ],
                                },
                                {
                                    "kind": "all_of",
                                    "rules": [
                                        {"kind": "arch_in", "arches": ["sm_107a"]},
                                        {"kind": "walk_ctas_at_least", "waves": 2.3},
                                    ],
                                },
                            ],
                        },
                        {"kind": "segments_heads_at_most", "max": 8192},
                    ],
                },
                {"kind": "heads_at_most", "max": 2},
            ],
        },
        "stage": "fwdh_p3",
        "variant": "o_ov_rows",
    },
    "fwdh_p3_o_ov_id_rows": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "segments"],
            ["buffer", "h0"],
            ["buffer", "flags"],
            ["buffer", "ht_out"],
            ["parameter", "use_h0"],
            ["parameter", "write_tapes"],
            ["parameter", "write_ht"],
            ["buffer", "qg_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "o_out"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["parameter", "scale"],
            ["parameter", "epoch"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "segments_heads", "passes": [1]},
        "kernel_symbol": "kernel_cake_kda_train_5f5a5b3715675e083823",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 140288,
            "use_pdl": True,
        },
        "library": "cake_kda_train_5f5a5b3715675e083823",
        "logical": {
            "aqk_tma": "aqk_ext",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "epoch": "epoch",
            "flags": "flags",
            "gk": "gk",
            "h0": "hseg",
            "h_out": "h",
            "ht_out": "seg_state",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_segments",
            "o_out": "out",
            "qg_tma": "qg",
            "scale": "scale",
            "segments": "seg_arg",
            "seq_chunk_start": "seg_table",
            "u": "u",
            "use_h0": "use_h0",
            "v_new": "v_new",
            "w_tma": "w",
            "write_ht": "write_ht",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "route_in", "routes": ["seg"]},
                        {"kind": "path_in", "paths": ["fwd"]},
                        {
                            "kind": "any_of",
                            "rules": [
                                {
                                    "kind": "all_of",
                                    "rules": [
                                        {
                                            "kind": "arch_in",
                                            "arches": ["sm_100a", "sm_103a"],
                                        },
                                        {"kind": "walk_ctas_at_least", "waves": 3.0},
                                    ],
                                },
                                {
                                    "kind": "all_of",
                                    "rules": [
                                        {"kind": "arch_in", "arches": ["sm_107a"]},
                                        {"kind": "walk_ctas_at_least", "waves": 2.3},
                                    ],
                                },
                            ],
                        },
                        {"kind": "segments_heads_at_most", "max": 8192},
                    ],
                },
                {"kind": "heads_above", "min": 2},
            ],
        },
        "stage": "fwdh_p3",
        "variant": "o_ov_id_rows",
    },
    "fwdh_p3_wy": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "akk_tma"],
            ["buffer", "kb_tma"],
            ["buffer", "v_tma"],
            ["buffer", "beta"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "kg_tma"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["buffer", "w_t_out"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "segments"],
            ["buffer", "h0"],
            ["buffer", "ht_out"],
            ["parameter", "use_h0"],
            ["parameter", "write_tapes"],
            ["parameter", "write_ht"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "segments_heads", "passes": [1]},
        "kernel_symbol": "kernel_cake_kda_train_945a87fa84a265fc9f94",
        "launch": {
            "block": (448, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 173056,
            "use_pdl": False,
        },
        "library": "cake_kda_train_945a87fa84a265fc9f94",
        "logical": {
            "akk_tma": "akk_src",
            "beta": "beta",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "gk": "gk",
            "h0": "hseg",
            "h_out": "h",
            "ht_out": "seg_state",
            "kb_tma": "kb",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_segments",
            "segments": "seg_arg",
            "seq_chunk_start": "seg_table",
            "use_h0": "use_h0",
            "v_new": "v_new",
            "v_tma": "v",
            "w_t_out": "w_tk",
            "write_ht": "write_ht",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "route_in", "routes": ["seg"]},
                {"kind": "path_in", "paths": ["light_states"]},
                {"kind": "chains_at_most", "max": 32},
            ],
        },
        "stage": "fwdh_p3",
        "variant": "wy",
    },
    "fwdh_p3": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "segments"],
            ["buffer", "h0"],
            ["buffer", "eye"],
            ["buffer", "ht_out"],
            ["parameter", "use_h0"],
            ["parameter", "write_tapes"],
            ["parameter", "write_ht"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "segments_heads", "passes": [1]},
        "kernel_symbol": "kernel_cake_kda_train_0e5fa2fe9ae292c3e3aa",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_kda_train_0e5fa2fe9ae292c3e3aa",
        "logical": {
            "eye": "eye",
            "gk": "gk",
            "h0": "hseg",
            "h_out": "h",
            "ht_out": "seg_state",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_segments",
            "segments": "seg_arg",
            "seq_chunk_start": "seg_table",
            "u": "u",
            "use_h0": "use_h0",
            "v_new": "v_new",
            "w_tma": "w",
            "write_ht": "write_ht",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "route_in", "routes": ["seg"]},
                {
                    "kind": "any_of",
                    "rules": [
                        {"kind": "path_in", "paths": ["light", "replay"]},
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "path_in", "paths": ["light_states"]},
                                {"kind": "chains_above", "min": 32},
                            ],
                        },
                    ],
                },
            ],
        },
        "stage": "fwdh_p3",
        "variant": None,
    },
    "bwd_prep_g8": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "g"],
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "beta"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "gk_out"],
            ["buffer", "kb_out"],
            ["buffer", "qg_out"],
            ["buffer", "kg_out"],
            ["buffer", "beta_int"],
            ["buffer", "v_new"],
            ["buffer", "dv2"],
            ["parameter", "num_qk_heads"],
            ["parameter", "num_heads"],
            ["parameter", "group"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "chunks", "x_factor": 1},
        "kernel_symbol": "kernel_cake_kda_train_98e6ba1f97af72fa2e32",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 36864,
            "use_pdl": False,
        },
        "library": "cake_kda_train_98e6ba1f97af72fa2e32",
        "logical": {
            "beta": "beta",
            "beta_int": "beta_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dv2": "dv2",
            "g": "g",
            "gk_out": "gk",
            "group": "group",
            "k": "k",
            "kb_out": "kb",
            "kg_out": "kg",
            "num_heads": "num_heads",
            "num_qk_heads": "num_heads",
            "q": "q",
            "qg_out": "qg",
            "v_new": "v_new",
        },
        "select_rule": {"kind": "path_in", "paths": ["tape"]},
        "stage": "bwd_prep",
        "variant": "g8",
    },
    "bwd_prep_g8_replay": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "g"],
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "beta"],
            ["buffer", "do"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "gk_out"],
            ["buffer", "kb_out"],
            ["buffer", "qg_out"],
            ["buffer", "kg_out"],
            ["buffer", "do_int"],
            ["buffer", "beta_int"],
            ["buffer", "v_new"],
            ["buffer", "dv2"],
            ["parameter", "num_qk_heads"],
            ["parameter", "num_heads"],
            ["parameter", "group"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "chunks", "x_factor": 1},
        "kernel_symbol": "kernel_cake_kda_train_86df3294f866784ca973",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 36864,
            "use_pdl": False,
        },
        "library": "cake_kda_train_86df3294f866784ca973",
        "logical": {
            "beta": "beta",
            "beta_int": "beta_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "do": "dout",
            "do_int": "do_int",
            "dv2": "dv2",
            "g": "g",
            "gk_out": "gk",
            "group": "group",
            "k": "k",
            "kb_out": "kb",
            "kg_out": "kg",
            "num_heads": "num_heads",
            "num_qk_heads": "num_heads",
            "q": "q",
            "qg_out": "qg",
            "v_new": "v_new",
        },
        "select_rule": {"kind": "path_in", "paths": ["replay"]},
        "stage": "bwd_prep",
        "variant": "g8_replay",
    },
    "bwd_prep_g8_noqg": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "g"],
            ["buffer", "k"],
            ["buffer", "beta"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "gk_out"],
            ["buffer", "kb_out"],
            ["buffer", "kg_out"],
            ["buffer", "beta_int"],
            ["buffer", "v_new"],
            ["buffer", "dv2"],
            ["parameter", "num_qk_heads"],
            ["parameter", "num_heads"],
            ["parameter", "group"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "chunks", "x_factor": 1},
        "kernel_symbol": "kernel_cake_kda_train_e1bf4eadfa51d1646ef7",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 36864,
            "use_pdl": False,
        },
        "library": "cake_kda_train_e1bf4eadfa51d1646ef7",
        "logical": {
            "beta": "beta",
            "beta_int": "beta_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dv2": "dv2",
            "g": "g",
            "gk_out": "gk",
            "group": "group",
            "k": "k",
            "kb_out": "kb",
            "kg_out": "kg",
            "num_heads": "num_heads",
            "num_qk_heads": "num_heads",
            "v_new": "v_new",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["light", "light_states"]},
                {
                    "kind": "any_of",
                    "rules": [
                        {"kind": "path_in", "paths": ["light_states"]},
                        {"kind": "route_in", "routes": ["seg"]},
                        {"kind": "heads_below", "max": 16},
                        {"kind": "heads_above", "min": 16},
                        {"kind": "slots_above", "slices": 4, "ctas_per_sm": 4},
                        {"kind": "arch_in", "arches": ["sm_103a", "sm_107a"]},
                    ],
                },
                {
                    "kind": "any_of",
                    "rules": [
                        {"kind": "route_in", "routes": ["serial"]},
                        {"kind": "heads_below", "max": 16},
                        {"kind": "heads_above", "min": 16},
                        {"kind": "chains_at_most", "max": 32},
                        {"kind": "arch_in", "arches": ["sm_100a"]},
                    ],
                },
            ],
        },
        "stage": "bwd_prep",
        "variant": "g8_noqg",
    },
    "bwd_prep_g8_noqg_nokb": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "g"],
            ["buffer", "k"],
            ["buffer", "beta"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "gk_out"],
            ["buffer", "kg_out"],
            ["buffer", "beta_int"],
            ["buffer", "v_new"],
            ["buffer", "dv2"],
            ["parameter", "num_qk_heads"],
            ["parameter", "num_heads"],
            ["parameter", "group"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "chunks", "x_factor": 1},
        "kernel_symbol": "kernel_cake_kda_train_0e7c693f766175f4406c",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 36864,
            "use_pdl": False,
        },
        "library": "cake_kda_train_0e7c693f766175f4406c",
        "logical": {
            "beta": "beta",
            "beta_int": "beta_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dv2": "dv2",
            "g": "g",
            "gk_out": "gk",
            "group": "group",
            "k": "k",
            "kg_out": "kg",
            "num_heads": "num_heads",
            "num_qk_heads": "num_heads",
            "v_new": "v_new",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["light", "light_states"]},
                {
                    "kind": "any_of",
                    "rules": [
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "path_in", "paths": ["light"]},
                                {"kind": "route_in", "routes": ["serial"]},
                                {
                                    "kind": "all_of",
                                    "rules": [
                                        {"kind": "heads_at_least", "min": 16},
                                        {"kind": "heads_at_most", "max": 16},
                                    ],
                                },
                                {
                                    "kind": "slots_at_most",
                                    "slices": 4,
                                    "ctas_per_sm": 4,
                                },
                                {"kind": "arch_in", "arches": ["sm_100a"]},
                            ],
                        },
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "route_in", "routes": ["seg"]},
                                {
                                    "kind": "all_of",
                                    "rules": [
                                        {"kind": "heads_at_least", "min": 16},
                                        {"kind": "heads_at_most", "max": 16},
                                    ],
                                },
                                {"kind": "chains_above", "min": 32},
                                {"kind": "arch_in", "arches": ["sm_103a", "sm_107a"]},
                            ],
                        },
                    ],
                },
            ],
        },
        "stage": "bwd_prep",
        "variant": "g8_noqg_nokb",
    },
    "dav_serial": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "do_tma"],
            ["buffer", "vnew_tma"],
            ["buffer", "dAqk"],
            ["parameter", "num_heads"],
            ["parameter", "scale"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["parameter", "num_pairs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "pairs_bounded", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_kda_train_021592c077b54de470a5",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 132096,
            "use_pdl": False,
        },
        "library": "cake_kda_train_021592c077b54de470a5",
        "logical": {
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dAqk": "dAqk",
            "do_tma": "dout",
            "num_heads": "num_heads",
            "num_pairs": "num_pairs",
            "scale": "scale",
            "vnew_tma": "v_new",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["light", "light_states", "tape"]},
                {"kind": "route_in", "routes": ["serial"]},
                {"kind": "heads_below", "max": 16},
            ],
        },
        "stage": "dav",
        "variant": "serial",
    },
    "dav_serial_replay": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "do_tma"],
            ["buffer", "vnew_tma"],
            ["buffer", "dAqk"],
            ["parameter", "num_heads"],
            ["parameter", "scale"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["parameter", "num_pairs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "pairs_bounded", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_kda_train_cace58c99e770d8a5a04",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 132096,
            "use_pdl": False,
        },
        "library": "cake_kda_train_cace58c99e770d8a5a04",
        "logical": {
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dAqk": "dAqk",
            "do_tma": "do_int",
            "num_heads": "num_heads",
            "num_pairs": "num_pairs",
            "scale": "scale",
            "vnew_tma": "v_new",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["replay"]},
                {"kind": "route_in", "routes": ["serial"]},
                {"kind": "heads_below", "max": 16},
            ],
        },
        "stage": "dav",
        "variant": "serial_replay",
    },
    "dhu_v2t": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "gk"],
            ["buffer", "dh_out"],
            ["buffer", "dv2"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "scale"],
            ["buffer", "seq_order"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_doc_head", "slices": 4},
        "kernel_symbol": "kernel_cake_kda_train_8eb4e9b8a9eae4e077d3",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 99328,
            "use_pdl": False,
        },
        "library": "cake_kda_train_8eb4e9b8a9eae4e077d3",
        "logical": {
            "aqk_tma": "aqk_src",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dh_out": "dh",
            "do_tma": "dout",
            "dv2": "dv2",
            "gk": "gk",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "qg_tma": "qg",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "seq_order": "seq_order",
            "w_tma": "w",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["light", "light_states", "tape"]},
                {"kind": "route_in", "routes": ["serial"]},
                {"kind": "slots_at_most", "slices": 4, "ctas_per_sm": 4},
            ],
        },
        "stage": "dhu",
        "variant": "v2t",
    },
    "dhu_v2t_replay": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "gk"],
            ["buffer", "dh_out"],
            ["buffer", "dv2"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "scale"],
            ["buffer", "seq_order"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_doc_head", "slices": 4},
        "kernel_symbol": "kernel_cake_kda_train_f1777a88a21b832b29ee",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 99328,
            "use_pdl": False,
        },
        "library": "cake_kda_train_f1777a88a21b832b29ee",
        "logical": {
            "aqk_tma": "aqk_int",
            "dh_out": "dh",
            "do_tma": "do_int",
            "dv2": "dv2",
            "gk": "gk",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "qg_tma": "qg",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "seq_order": "seq_order",
            "w_tma": "w",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["replay"]},
                {"kind": "route_in", "routes": ["serial"]},
                {"kind": "slots_at_most", "slices": 4, "ctas_per_sm": 4},
            ],
        },
        "stage": "dhu",
        "variant": "v2t_replay",
    },
    "dhu_v1": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "gk"],
            ["buffer", "dh_out"],
            ["buffer", "dv2"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "scale"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_doc_head", "slices": 2},
        "kernel_symbol": "kernel_cake_kda_train_d32455627b87ba0a1d87",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 107520,
            "use_pdl": False,
        },
        "library": "cake_kda_train_d32455627b87ba0a1d87",
        "logical": {
            "aqk_tma": "aqk_src",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dh_out": "dh",
            "do_tma": "dout",
            "dv2": "dv2",
            "gk": "gk",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "qg_tma": "qg",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "w_tma": "w",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "route_in", "routes": ["serial"]},
                {"kind": "slots_above", "slices": 4, "ctas_per_sm": 4},
                {
                    "kind": "any_of",
                    "rules": [
                        {"kind": "path_in", "paths": ["tape"]},
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "path_in", "paths": ["light"]},
                                {"kind": "heads_below", "max": 64},
                            ],
                        },
                    ],
                },
            ],
        },
        "stage": "dhu",
        "variant": "v1",
    },
    "dhu_v1_replay": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "gk"],
            ["buffer", "dh_out"],
            ["buffer", "dv2"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_doc_head", "slices": 2},
        "kernel_symbol": "kernel_cake_kda_train_4734a5d4f63d815ec086",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 107520,
            "use_pdl": False,
        },
        "library": "cake_kda_train_4734a5d4f63d815ec086",
        "logical": {
            "aqk_tma": "aqk_int",
            "dh_out": "dh",
            "do_tma": "do_int",
            "dv2": "dv2",
            "gk": "gk",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "qg_tma": "qg",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "w_tma": "w",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["replay"]},
                {"kind": "route_in", "routes": ["serial"]},
                {"kind": "heads_below", "max": 64},
                {"kind": "slots_above", "slices": 4, "ctas_per_sm": 4},
            ],
        },
        "stage": "dhu",
        "variant": "v1_replay",
    },
    "dhu_v1_wt": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "gk"],
            ["buffer", "dh_out"],
            ["buffer", "dv2"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "scale"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_doc_head", "slices": 2},
        "kernel_symbol": "kernel_cake_kda_train_22a2f04d41df18554b84",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 107520,
            "use_pdl": False,
        },
        "library": "cake_kda_train_22a2f04d41df18554b84",
        "logical": {
            "aqk_tma": "aqk_src",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dh_out": "dh",
            "do_tma": "dout",
            "dv2": "dv2",
            "gk": "gk",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "qg_tma": "qg",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "w_tma": "w_t",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["light"]},
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "heads_at_least", "min": 64},
                        {"kind": "slots_above", "slices": 4, "ctas_per_sm": 4},
                    ],
                },
            ],
        },
        "stage": "dhu",
        "variant": "v1_wt",
    },
    "dhu_v1_wt_replay": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "gk"],
            ["buffer", "dh_out"],
            ["buffer", "dv2"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_doc_head", "slices": 2},
        "kernel_symbol": "kernel_cake_kda_train_036e1c7e1dfc95a88aae",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 107520,
            "use_pdl": False,
        },
        "library": "cake_kda_train_036e1c7e1dfc95a88aae",
        "logical": {
            "aqk_tma": "aqk_int",
            "dh_out": "dh",
            "do_tma": "do_int",
            "dv2": "dv2",
            "gk": "gk",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "qg_tma": "qg",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "w_tma": "w_t",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["replay"]},
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "heads_at_least", "min": 64},
                        {"kind": "slots_above", "slices": 4, "ctas_per_sm": 4},
                    ],
                },
            ],
        },
        "stage": "dhu",
        "variant": "v1_wt_replay",
    },
    "dhu_v3_f": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "gk"],
            ["buffer", "dh_out"],
            ["buffer", "dv2"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "segments"],
            ["parameter", "scale"],
            ["buffer", "dht"],
            ["buffer", "ht_out"],
            ["parameter", "use_dht"],
            ["parameter", "write_tapes"],
            ["parameter", "write_ht"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "segments_heads", "passes": [1, 1]},
        "kernel_symbol": "kernel_cake_kda_train_74e19f83f8d97e669d46",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_kda_train_74e19f83f8d97e669d46",
        "logical": {
            "aqk_tma": "aqk_src",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dh_out": "dh",
            "dht": "dhseg",
            "do_tma": "dout",
            "dv2": "dv2",
            "gk": "gk",
            "ht_out": "dseg",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_segments",
            "qg_tma": "qg",
            "scale": "scale",
            "segments": "seg_arg",
            "seq_chunk_start": "seg_table",
            "use_dht": "use_dht",
            "w_tma": "w",
            "write_ht": "write_ht",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "route_in", "routes": ["seg"]},
                        {"kind": "chains_above", "min": 4},
                    ],
                },
                {
                    "kind": "any_of",
                    "rules": [
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "arch_in", "arches": ["sm_100a", "sm_107a"]},
                                {
                                    "kind": "any_of",
                                    "rules": [
                                        {"kind": "walk_ctas_below", "waves": 4},
                                        {
                                            "kind": "all_of",
                                            "rules": [
                                                {"kind": "heads_at_most", "max": 4},
                                                {
                                                    "kind": "walk_ctas_at_most",
                                                    "waves": 4,
                                                },
                                            ],
                                        },
                                    ],
                                },
                            ],
                        },
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "arch_in", "arches": ["sm_103a"]},
                                {"kind": "heads_at_most", "max": 4},
                                {"kind": "walk_ctas_at_least", "waves": 4},
                                {"kind": "walk_ctas_at_most", "waves": 4},
                            ],
                        },
                    ],
                },
                {
                    "kind": "any_of",
                    "rules": [
                        {"kind": "path_in", "paths": ["light", "tape"]},
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "path_in", "paths": ["light_states"]},
                                {"kind": "chains_above", "min": 32},
                            ],
                        },
                    ],
                },
            ],
        },
        "stage": "dhu",
        "variant": "v3_f",
    },
    "dhu_v3_f_wt": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "gk"],
            ["buffer", "dh_out"],
            ["buffer", "dv2"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "segments"],
            ["parameter", "scale"],
            ["buffer", "dht"],
            ["buffer", "ht_out"],
            ["parameter", "use_dht"],
            ["parameter", "write_tapes"],
            ["parameter", "write_ht"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "segments_heads", "passes": [1, 1]},
        "kernel_symbol": "kernel_cake_kda_train_157990bf8c73b3dbc4fe",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_kda_train_157990bf8c73b3dbc4fe",
        "logical": {
            "aqk_tma": "aqk_src",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dh_out": "dh",
            "dht": "dhseg",
            "do_tma": "dout",
            "dv2": "dv2",
            "gk": "gk",
            "ht_out": "dseg",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_segments",
            "qg_tma": "qg",
            "scale": "scale",
            "segments": "seg_arg",
            "seq_chunk_start": "seg_table",
            "use_dht": "use_dht",
            "w_tma": "w_tk",
            "write_ht": "write_ht",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "route_in", "routes": ["seg"]},
                        {"kind": "chains_above", "min": 4},
                    ],
                },
                {
                    "kind": "any_of",
                    "rules": [
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "arch_in", "arches": ["sm_100a", "sm_107a"]},
                                {
                                    "kind": "any_of",
                                    "rules": [
                                        {"kind": "walk_ctas_below", "waves": 4},
                                        {
                                            "kind": "all_of",
                                            "rules": [
                                                {"kind": "heads_at_most", "max": 4},
                                                {
                                                    "kind": "walk_ctas_at_most",
                                                    "waves": 4,
                                                },
                                            ],
                                        },
                                    ],
                                },
                            ],
                        },
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "arch_in", "arches": ["sm_103a"]},
                                {"kind": "heads_at_most", "max": 4},
                                {"kind": "walk_ctas_at_least", "waves": 4},
                                {"kind": "walk_ctas_at_most", "waves": 4},
                            ],
                        },
                    ],
                },
                {"kind": "path_in", "paths": ["light_states"]},
                {"kind": "chains_at_most", "max": 32},
            ],
        },
        "stage": "dhu",
        "variant": "v3_f_wt",
    },
    "dhu_v3_f_replay": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "gk"],
            ["buffer", "dh_out"],
            ["buffer", "dv2"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "segments"],
            ["parameter", "scale"],
            ["buffer", "dht"],
            ["buffer", "ht_out"],
            ["parameter", "use_dht"],
            ["parameter", "write_tapes"],
            ["parameter", "write_ht"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "segments_heads", "passes": [1, 1]},
        "kernel_symbol": "kernel_cake_kda_train_251cc1717dfc8e3ba24e",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_kda_train_251cc1717dfc8e3ba24e",
        "logical": {
            "aqk_tma": "aqk_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dh_out": "dh",
            "dht": "dhseg",
            "do_tma": "do_int",
            "dv2": "dv2",
            "gk": "gk",
            "ht_out": "dseg",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_segments",
            "qg_tma": "qg",
            "scale": "scale",
            "segments": "seg_arg",
            "seq_chunk_start": "seg_table",
            "use_dht": "use_dht",
            "w_tma": "w",
            "write_ht": "write_ht",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "route_in", "routes": ["seg"]},
                        {"kind": "chains_above", "min": 4},
                    ],
                },
                {
                    "kind": "any_of",
                    "rules": [
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "arch_in", "arches": ["sm_100a", "sm_107a"]},
                                {
                                    "kind": "any_of",
                                    "rules": [
                                        {"kind": "walk_ctas_below", "waves": 4},
                                        {
                                            "kind": "all_of",
                                            "rules": [
                                                {"kind": "heads_at_most", "max": 4},
                                                {
                                                    "kind": "walk_ctas_at_most",
                                                    "waves": 4,
                                                },
                                            ],
                                        },
                                    ],
                                },
                            ],
                        },
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "arch_in", "arches": ["sm_103a"]},
                                {"kind": "heads_at_most", "max": 4},
                                {"kind": "walk_ctas_at_least", "waves": 4},
                                {"kind": "walk_ctas_at_most", "waves": 4},
                            ],
                        },
                    ],
                },
                {"kind": "path_in", "paths": ["replay"]},
            ],
        },
        "stage": "dhu",
        "variant": "v3_f_replay",
    },
    "dhu_v3p_f": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "gk"],
            ["buffer", "dh_out"],
            ["buffer", "dv2"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "segments"],
            ["parameter", "scale"],
            ["buffer", "dht"],
            ["buffer", "ht_out"],
            ["parameter", "use_dht"],
            ["parameter", "write_tapes"],
            ["parameter", "write_ht"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "segments_heads", "passes": [1, 1]},
        "kernel_symbol": "kernel_cake_kda_train_b4a1408642c39b6e6ccd",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_kda_train_b4a1408642c39b6e6ccd",
        "logical": {
            "aqk_tma": "aqk_src",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dh_out": "dh",
            "dht": "dhseg",
            "do_tma": "dout",
            "dv2": "dv2",
            "gk": "gk",
            "ht_out": "dseg",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_segments",
            "qg_tma": "qg",
            "scale": "scale",
            "segments": "seg_arg",
            "seq_chunk_start": "seg_table",
            "use_dht": "use_dht",
            "w_tma": "w",
            "write_ht": "write_ht",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "route_in", "routes": ["seg"]},
                        {"kind": "chains_above", "min": 4},
                    ],
                },
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "arch_in", "arches": ["sm_103a"]},
                        {"kind": "walk_ctas_below", "waves": 4},
                    ],
                },
                {
                    "kind": "any_of",
                    "rules": [
                        {"kind": "path_in", "paths": ["light", "tape"]},
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "path_in", "paths": ["light_states"]},
                                {"kind": "chains_above", "min": 32},
                            ],
                        },
                    ],
                },
            ],
        },
        "stage": "dhu",
        "variant": "v3p_f",
    },
    "dhu_v3p_f_wt": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "gk"],
            ["buffer", "dh_out"],
            ["buffer", "dv2"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "segments"],
            ["parameter", "scale"],
            ["buffer", "dht"],
            ["buffer", "ht_out"],
            ["parameter", "use_dht"],
            ["parameter", "write_tapes"],
            ["parameter", "write_ht"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "segments_heads", "passes": [1, 1]},
        "kernel_symbol": "kernel_cake_kda_train_91d59405d9b349c0709c",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_kda_train_91d59405d9b349c0709c",
        "logical": {
            "aqk_tma": "aqk_src",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dh_out": "dh",
            "dht": "dhseg",
            "do_tma": "dout",
            "dv2": "dv2",
            "gk": "gk",
            "ht_out": "dseg",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_segments",
            "qg_tma": "qg",
            "scale": "scale",
            "segments": "seg_arg",
            "seq_chunk_start": "seg_table",
            "use_dht": "use_dht",
            "w_tma": "w_tk",
            "write_ht": "write_ht",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "route_in", "routes": ["seg"]},
                        {"kind": "chains_above", "min": 4},
                    ],
                },
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "arch_in", "arches": ["sm_103a"]},
                        {"kind": "walk_ctas_below", "waves": 4},
                    ],
                },
                {"kind": "path_in", "paths": ["light_states"]},
                {"kind": "chains_at_most", "max": 32},
            ],
        },
        "stage": "dhu",
        "variant": "v3p_f_wt",
    },
    "dhu_v3p_f_replay": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "gk"],
            ["buffer", "dh_out"],
            ["buffer", "dv2"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "segments"],
            ["parameter", "scale"],
            ["buffer", "dht"],
            ["buffer", "ht_out"],
            ["parameter", "use_dht"],
            ["parameter", "write_tapes"],
            ["parameter", "write_ht"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "segments_heads", "passes": [1, 1]},
        "kernel_symbol": "kernel_cake_kda_train_e0b6246cba153e5c917b",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_kda_train_e0b6246cba153e5c917b",
        "logical": {
            "aqk_tma": "aqk_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dh_out": "dh",
            "dht": "dhseg",
            "do_tma": "do_int",
            "dv2": "dv2",
            "gk": "gk",
            "ht_out": "dseg",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_segments",
            "qg_tma": "qg",
            "scale": "scale",
            "segments": "seg_arg",
            "seq_chunk_start": "seg_table",
            "use_dht": "use_dht",
            "w_tma": "w",
            "write_ht": "write_ht",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "route_in", "routes": ["seg"]},
                        {"kind": "chains_above", "min": 4},
                    ],
                },
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "arch_in", "arches": ["sm_103a"]},
                        {"kind": "walk_ctas_below", "waves": 4},
                    ],
                },
                {"kind": "path_in", "paths": ["replay"]},
            ],
        },
        "stage": "dhu",
        "variant": "v3p_f_replay",
    },
    "dhu_v3r_f": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "gk"],
            ["buffer", "dh_out"],
            ["buffer", "dv2"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "segments"],
            ["parameter", "scale"],
            ["buffer", "dht"],
            ["buffer", "ht_out"],
            ["parameter", "use_dht"],
            ["parameter", "write_tapes"],
            ["parameter", "write_ht"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "segments_heads", "passes": [1, 1]},
        "kernel_symbol": "kernel_cake_kda_train_243fb0b23aea1b296ab6",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_kda_train_243fb0b23aea1b296ab6",
        "logical": {
            "aqk_tma": "aqk_src",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dh_out": "dh",
            "dht": "dhseg",
            "do_tma": "dout",
            "dv2": "dv2",
            "gk": "gk",
            "ht_out": "dseg",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_segments",
            "qg_tma": "qg",
            "scale": "scale",
            "segments": "seg_arg",
            "seq_chunk_start": "seg_table",
            "use_dht": "use_dht",
            "w_tma": "w",
            "write_ht": "write_ht",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "route_in", "routes": ["seg"]},
                        {"kind": "chains_above", "min": 4},
                    ],
                },
                {
                    "kind": "any_of",
                    "rules": [
                        {"kind": "walk_ctas_above", "waves": 4},
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "heads_above", "min": 4},
                                {"kind": "walk_ctas_at_least", "waves": 4},
                            ],
                        },
                    ],
                },
                {
                    "kind": "any_of",
                    "rules": [
                        {"kind": "path_in", "paths": ["light", "tape"]},
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "path_in", "paths": ["light_states"]},
                                {"kind": "chains_above", "min": 32},
                            ],
                        },
                    ],
                },
            ],
        },
        "stage": "dhu",
        "variant": "v3r_f",
    },
    "dhu_v3r_f_wt": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "gk"],
            ["buffer", "dh_out"],
            ["buffer", "dv2"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "segments"],
            ["parameter", "scale"],
            ["buffer", "dht"],
            ["buffer", "ht_out"],
            ["parameter", "use_dht"],
            ["parameter", "write_tapes"],
            ["parameter", "write_ht"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "segments_heads", "passes": [1, 1]},
        "kernel_symbol": "kernel_cake_kda_train_dec885b973bd44f88416",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_kda_train_dec885b973bd44f88416",
        "logical": {
            "aqk_tma": "aqk_src",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dh_out": "dh",
            "dht": "dhseg",
            "do_tma": "dout",
            "dv2": "dv2",
            "gk": "gk",
            "ht_out": "dseg",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_segments",
            "qg_tma": "qg",
            "scale": "scale",
            "segments": "seg_arg",
            "seq_chunk_start": "seg_table",
            "use_dht": "use_dht",
            "w_tma": "w_tk",
            "write_ht": "write_ht",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "route_in", "routes": ["seg"]},
                        {"kind": "chains_above", "min": 4},
                    ],
                },
                {
                    "kind": "any_of",
                    "rules": [
                        {"kind": "walk_ctas_above", "waves": 4},
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "heads_above", "min": 4},
                                {"kind": "walk_ctas_at_least", "waves": 4},
                            ],
                        },
                    ],
                },
                {"kind": "path_in", "paths": ["light_states"]},
                {"kind": "chains_at_most", "max": 32},
            ],
        },
        "stage": "dhu",
        "variant": "v3r_f_wt",
    },
    "dhu_v3r_f_replay": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "gk"],
            ["buffer", "dh_out"],
            ["buffer", "dv2"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "segments"],
            ["parameter", "scale"],
            ["buffer", "dht"],
            ["buffer", "ht_out"],
            ["parameter", "use_dht"],
            ["parameter", "write_tapes"],
            ["parameter", "write_ht"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "segments_heads", "passes": [1, 1]},
        "kernel_symbol": "kernel_cake_kda_train_909a0aa57efb3f353972",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_kda_train_909a0aa57efb3f353972",
        "logical": {
            "aqk_tma": "aqk_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dh_out": "dh",
            "dht": "dhseg",
            "do_tma": "do_int",
            "dv2": "dv2",
            "gk": "gk",
            "ht_out": "dseg",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_segments",
            "qg_tma": "qg",
            "scale": "scale",
            "segments": "seg_arg",
            "seq_chunk_start": "seg_table",
            "use_dht": "use_dht",
            "w_tma": "w",
            "write_ht": "write_ht",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "route_in", "routes": ["seg"]},
                        {"kind": "chains_above", "min": 4},
                    ],
                },
                {
                    "kind": "any_of",
                    "rules": [
                        {"kind": "walk_ctas_above", "waves": 4},
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "heads_above", "min": 4},
                                {"kind": "walk_ctas_at_least", "waves": 4},
                            ],
                        },
                    ],
                },
                {"kind": "path_in", "paths": ["replay"]},
            ],
        },
        "stage": "dhu",
        "variant": "v3r_f_replay",
    },
    "dhu_v3s_f": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "gk"],
            ["buffer", "dh_out"],
            ["buffer", "dv2"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "segments"],
            ["parameter", "scale"],
            ["buffer", "dht"],
            ["buffer", "ht_out"],
            ["parameter", "use_dht"],
            ["parameter", "write_tapes"],
            ["parameter", "write_ht"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "segments_heads", "passes": [2, 2]},
        "kernel_symbol": "kernel_cake_kda_train_3f7ffe0b7421d90ad674",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 107520,
            "use_pdl": False,
        },
        "library": "cake_kda_train_3f7ffe0b7421d90ad674",
        "logical": {
            "aqk_tma": "aqk_src",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dh_out": "dh",
            "dht": "dhseg",
            "do_tma": "dout",
            "dv2": "dv2",
            "gk": "gk",
            "ht_out": "dseg",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_segments",
            "qg_tma": "qg",
            "scale": "scale",
            "segments": "seg_arg",
            "seq_chunk_start": "seg_table",
            "use_dht": "use_dht",
            "w_tma": "w",
            "write_ht": "write_ht",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "route_in", "routes": ["seg"]},
                        {"kind": "chains_at_most", "max": 4},
                    ],
                },
                {"kind": "path_in", "paths": ["light", "tape"]},
            ],
        },
        "stage": "dhu",
        "variant": "v3s_f",
    },
    "dhu_v3s_f_wt": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "gk"],
            ["buffer", "dh_out"],
            ["buffer", "dv2"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "segments"],
            ["parameter", "scale"],
            ["buffer", "dht"],
            ["buffer", "ht_out"],
            ["parameter", "use_dht"],
            ["parameter", "write_tapes"],
            ["parameter", "write_ht"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "segments_heads", "passes": [2, 2]},
        "kernel_symbol": "kernel_cake_kda_train_2247b401b29d959d8133",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 107520,
            "use_pdl": False,
        },
        "library": "cake_kda_train_2247b401b29d959d8133",
        "logical": {
            "aqk_tma": "aqk_src",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dh_out": "dh",
            "dht": "dhseg",
            "do_tma": "dout",
            "dv2": "dv2",
            "gk": "gk",
            "ht_out": "dseg",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_segments",
            "qg_tma": "qg",
            "scale": "scale",
            "segments": "seg_arg",
            "seq_chunk_start": "seg_table",
            "use_dht": "use_dht",
            "w_tma": "w_tk",
            "write_ht": "write_ht",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "route_in", "routes": ["seg"]},
                        {"kind": "chains_at_most", "max": 4},
                    ],
                },
                {"kind": "path_in", "paths": ["light_states"]},
            ],
        },
        "stage": "dhu",
        "variant": "v3s_f_wt",
    },
    "dhu_v3s_f_replay": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "gk"],
            ["buffer", "dh_out"],
            ["buffer", "dv2"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "segments"],
            ["parameter", "scale"],
            ["buffer", "dht"],
            ["buffer", "ht_out"],
            ["parameter", "use_dht"],
            ["parameter", "write_tapes"],
            ["parameter", "write_ht"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "segments_heads", "passes": [2, 2]},
        "kernel_symbol": "kernel_cake_kda_train_de4f911d8beaf8ac8d79",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 107520,
            "use_pdl": False,
        },
        "library": "cake_kda_train_de4f911d8beaf8ac8d79",
        "logical": {
            "aqk_tma": "aqk_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dh_out": "dh",
            "dht": "dhseg",
            "do_tma": "do_int",
            "dv2": "dv2",
            "gk": "gk",
            "ht_out": "dseg",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_segments",
            "qg_tma": "qg",
            "scale": "scale",
            "segments": "seg_arg",
            "seq_chunk_start": "seg_table",
            "use_dht": "use_dht",
            "w_tma": "w",
            "write_ht": "write_ht",
            "write_tapes": "write_tapes",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "route_in", "routes": ["seg"]},
                        {"kind": "chains_at_most", "max": 4},
                    ],
                },
                {"kind": "path_in", "paths": ["replay"]},
            ],
        },
        "stage": "dhu",
        "variant": "v3s_f_replay",
    },
    "dqkg": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "do_tma"],
            ["buffer", "vn_tma"],
            ["buffer", "dv2_tma"],
            ["buffer", "v_tma"],
            ["buffer", "h_tma"],
            ["buffer", "dh_tma"],
            ["buffer", "akk_tma"],
            ["buffer", "gk_tma"],
            ["buffer", "k_tma"],
            ["buffer", "q_tma"],
            ["buffer", "beta"],
            ["buffer", "dq_out"],
            ["buffer", "dk_out"],
            ["buffer", "dg_out"],
            ["buffer", "db_out"],
            ["buffer", "dAkk_out"],
            ["buffer", "dv_out"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["parameter", "scale"],
            ["buffer", "dAqk_out"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "sm_bounded", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_kda_train_ecf750512be27b9043fa",
        "launch": {
            "block": (576, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 224384,
            "use_pdl": False,
        },
        "library": "cake_kda_train_ecf750512be27b9043fa",
        "logical": {
            "akk_tma": "akk_src",
            "beta": "beta_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dAkk_out": "dAkk",
            "dAqk_out": "dAqk",
            "db_out": "db_f",
            "dg_out": "dg_f",
            "dh_tma": "dh2d",
            "dk_out": "dk_f",
            "do_tma": "dout",
            "dq_out": "dq_f",
            "dv2_tma": "dv2",
            "dv_out": "dv",
            "gk_tma": "gk",
            "h_tma": "h2d",
            "k_tma": "k",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "q_tma": "q",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "v_tma": "v",
            "vn_tma": "v_new",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["light", "light_states", "tape"]},
                {"kind": "route_in", "routes": ["serial"]},
                {"kind": "heads_below", "max": 16},
            ],
        },
        "stage": "dqkg",
        "variant": None,
    },
    "dqkg_int": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "do_tma"],
            ["buffer", "vn_tma"],
            ["buffer", "dv2_tma"],
            ["buffer", "v_tma"],
            ["buffer", "h_tma"],
            ["buffer", "dh_tma"],
            ["buffer", "akk_tma"],
            ["buffer", "gk_tma"],
            ["buffer", "k_tma"],
            ["buffer", "q_tma"],
            ["buffer", "beta"],
            ["buffer", "dq_out"],
            ["buffer", "dk_out"],
            ["buffer", "dg_out"],
            ["buffer", "db_out"],
            ["buffer", "dAkk_out"],
            ["buffer", "dv_out"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["parameter", "scale"],
            ["buffer", "dAqk_out"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "sm_bounded", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_kda_train_6df24a76cec0e85a8c75",
        "launch": {
            "block": (576, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 224384,
            "use_pdl": False,
        },
        "library": "cake_kda_train_6df24a76cec0e85a8c75",
        "logical": {
            "akk_tma": "akk_int",
            "beta": "beta_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dAkk_out": "dAkk",
            "dAqk_out": "dAqk",
            "db_out": "db_f",
            "dg_out": "dg_f",
            "dh_tma": "dh2d",
            "dk_out": "dk_f",
            "do_tma": "do_int",
            "dq_out": "dq_f",
            "dv2_tma": "dv2",
            "dv_out": "dv",
            "gk_tma": "gk",
            "h_tma": "h2d",
            "k_tma": "k",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "q_tma": "q",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "v_tma": "v",
            "vn_tma": "v_new",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["replay"]},
                {"kind": "route_in", "routes": ["serial"]},
                {"kind": "heads_below", "max": 16},
            ],
        },
        "stage": "dqkg",
        "variant": "int",
    },
    "dqkg_davq": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "do_tma"],
            ["buffer", "vn_tma"],
            ["buffer", "dv2_tma"],
            ["buffer", "v_tma"],
            ["buffer", "h_tma"],
            ["buffer", "dh_tma"],
            ["buffer", "akk_tma"],
            ["buffer", "gk_tma"],
            ["buffer", "k_tma"],
            ["buffer", "q_tma"],
            ["buffer", "beta"],
            ["buffer", "dq_out"],
            ["buffer", "dk_out"],
            ["buffer", "dg_out"],
            ["buffer", "db_out"],
            ["buffer", "dAkk_out"],
            ["buffer", "dv_out"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["parameter", "scale"],
            ["buffer", "dAqk_out"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "sm_bounded", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_kda_train_cfa2684db4d92e37448d",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 224384,
            "use_pdl": False,
        },
        "library": "cake_kda_train_cfa2684db4d92e37448d",
        "logical": {
            "akk_tma": "akk_src",
            "beta": "beta_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dAkk_out": "dAkk",
            "dAqk_out": "dAqk",
            "db_out": "db_f",
            "dg_out": "dg_f",
            "dh_tma": "dh2d",
            "dk_out": "dk_f",
            "do_tma": "dout",
            "dq_out": "dq_f",
            "dv2_tma": "dv2",
            "dv_out": "dv",
            "gk_tma": "gk",
            "h_tma": "h2d",
            "k_tma": "k",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "q_tma": "q",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "v_tma": "v",
            "vn_tma": "v_new",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["light", "light_states", "tape"]},
                {
                    "kind": "any_of",
                    "rules": [
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "route_in", "routes": ["serial"]},
                                {"kind": "heads_at_least", "min": 16},
                                {
                                    "kind": "any_of",
                                    "rules": [
                                        {
                                            "kind": "arch_in",
                                            "arches": ["sm_100a", "sm_103a"],
                                        },
                                        {"kind": "heads_below", "max": 16},
                                    ],
                                },
                            ],
                        },
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "route_in", "routes": ["seg"]},
                                {
                                    "kind": "any_of",
                                    "rules": [
                                        {
                                            "kind": "arch_in",
                                            "arches": ["sm_100a", "sm_103a"],
                                        },
                                        {"kind": "heads_below", "max": 16},
                                    ],
                                },
                            ],
                        },
                    ],
                },
            ],
        },
        "stage": "dqkg",
        "variant": "davq",
    },
    "dqkg_int_davq": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "do_tma"],
            ["buffer", "vn_tma"],
            ["buffer", "dv2_tma"],
            ["buffer", "v_tma"],
            ["buffer", "h_tma"],
            ["buffer", "dh_tma"],
            ["buffer", "akk_tma"],
            ["buffer", "gk_tma"],
            ["buffer", "k_tma"],
            ["buffer", "q_tma"],
            ["buffer", "beta"],
            ["buffer", "dq_out"],
            ["buffer", "dk_out"],
            ["buffer", "dg_out"],
            ["buffer", "db_out"],
            ["buffer", "dAkk_out"],
            ["buffer", "dv_out"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["parameter", "scale"],
            ["buffer", "dAqk_out"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "sm_bounded", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_kda_train_d098004ffe4a23d58ff4",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 224384,
            "use_pdl": False,
        },
        "library": "cake_kda_train_d098004ffe4a23d58ff4",
        "logical": {
            "akk_tma": "akk_int",
            "beta": "beta_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dAkk_out": "dAkk",
            "dAqk_out": "dAqk",
            "db_out": "db_f",
            "dg_out": "dg_f",
            "dh_tma": "dh2d",
            "dk_out": "dk_f",
            "do_tma": "do_int",
            "dq_out": "dq_f",
            "dv2_tma": "dv2",
            "dv_out": "dv",
            "gk_tma": "gk",
            "h_tma": "h2d",
            "k_tma": "k",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "q_tma": "q",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "v_tma": "v",
            "vn_tma": "v_new",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["replay"]},
                {
                    "kind": "any_of",
                    "rules": [
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "route_in", "routes": ["serial"]},
                                {"kind": "heads_at_least", "min": 16},
                                {
                                    "kind": "any_of",
                                    "rules": [
                                        {
                                            "kind": "arch_in",
                                            "arches": ["sm_100a", "sm_103a"],
                                        },
                                        {"kind": "heads_below", "max": 16},
                                    ],
                                },
                            ],
                        },
                        {
                            "kind": "all_of",
                            "rules": [
                                {"kind": "route_in", "routes": ["seg"]},
                                {
                                    "kind": "any_of",
                                    "rules": [
                                        {
                                            "kind": "arch_in",
                                            "arches": ["sm_100a", "sm_103a"],
                                        },
                                        {"kind": "heads_below", "max": 16},
                                    ],
                                },
                            ],
                        },
                    ],
                },
            ],
        },
        "stage": "dqkg",
        "variant": "int_davq",
    },
    "dqkg_davq_nohint": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "do_tma"],
            ["buffer", "vn_tma"],
            ["buffer", "dv2_tma"],
            ["buffer", "v_tma"],
            ["buffer", "h_tma"],
            ["buffer", "dh_tma"],
            ["buffer", "akk_tma"],
            ["buffer", "gk_tma"],
            ["buffer", "k_tma"],
            ["buffer", "q_tma"],
            ["buffer", "beta"],
            ["buffer", "dq_out"],
            ["buffer", "dk_out"],
            ["buffer", "dg_out"],
            ["buffer", "db_out"],
            ["buffer", "dAkk_out"],
            ["buffer", "dv_out"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["parameter", "scale"],
            ["buffer", "dAqk_out"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "sm_bounded", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_kda_train_72ed2d17bb80595090a8",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 224384,
            "use_pdl": False,
        },
        "library": "cake_kda_train_72ed2d17bb80595090a8",
        "logical": {
            "akk_tma": "akk_src",
            "beta": "beta_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dAkk_out": "dAkk",
            "dAqk_out": "dAqk",
            "db_out": "db_f",
            "dg_out": "dg_f",
            "dh_tma": "dh2d",
            "dk_out": "dk_f",
            "do_tma": "dout",
            "dq_out": "dq_f",
            "dv2_tma": "dv2",
            "dv_out": "dv",
            "gk_tma": "gk",
            "h_tma": "h2d",
            "k_tma": "k",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "q_tma": "q",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "v_tma": "v",
            "vn_tma": "v_new",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["light", "light_states", "tape"]},
                {"kind": "arch_in", "arches": ["sm_107a"]},
                {"kind": "heads_at_least", "min": 16},
            ],
        },
        "stage": "dqkg",
        "variant": "davq_nohint",
    },
    "dqkg_int_davq_nohint": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "do_tma"],
            ["buffer", "vn_tma"],
            ["buffer", "dv2_tma"],
            ["buffer", "v_tma"],
            ["buffer", "h_tma"],
            ["buffer", "dh_tma"],
            ["buffer", "akk_tma"],
            ["buffer", "gk_tma"],
            ["buffer", "k_tma"],
            ["buffer", "q_tma"],
            ["buffer", "beta"],
            ["buffer", "dq_out"],
            ["buffer", "dk_out"],
            ["buffer", "dg_out"],
            ["buffer", "db_out"],
            ["buffer", "dAkk_out"],
            ["buffer", "dv_out"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["parameter", "scale"],
            ["buffer", "dAqk_out"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "sm_bounded", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_kda_train_f80ac356149d8d2c42e6",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 224384,
            "use_pdl": False,
        },
        "library": "cake_kda_train_f80ac356149d8d2c42e6",
        "logical": {
            "akk_tma": "akk_int",
            "beta": "beta_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dAkk_out": "dAkk",
            "dAqk_out": "dAqk",
            "db_out": "db_f",
            "dg_out": "dg_f",
            "dh_tma": "dh2d",
            "dk_out": "dk_f",
            "do_tma": "do_int",
            "dq_out": "dq_f",
            "dv2_tma": "dv2",
            "dv_out": "dv",
            "gk_tma": "gk",
            "h_tma": "h2d",
            "k_tma": "k",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "q_tma": "q",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "v_tma": "v",
            "vn_tma": "v_new",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": ["replay"]},
                {"kind": "arch_in", "arches": ["sm_107a"]},
                {"kind": "heads_at_least", "min": 16},
            ],
        },
        "stage": "dqkg",
        "variant": "int_davq_nohint",
    },
    "glm_intra": {
        "arches": ["sm_100a"],
        "arg_plan": [
            ["buffer", "dAqk"],
            ["buffer", "dAkk"],
            ["buffer", "gk"],
            ["buffer", "k_e"],
            ["buffer", "q_e"],
            ["buffer", "dq_f_t"],
            ["buffer", "dk_f_t"],
            ["buffer", "dg_f_t"],
            ["buffer", "beta"],
            ["buffer", "dq_f"],
            ["buffer", "dk_f"],
            ["buffer", "dg_f"],
            ["buffer", "db_f"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "dq_ext"],
            ["buffer", "dk_ext"],
            ["buffer", "dg_out"],
            ["buffer", "db_out"],
            ["parameter", "num_heads"],
            ["parameter", "num_qk_heads"],
            ["parameter", "group"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "chunks", "x_factor": 1},
        "kernel_symbol": "kernel_cake_kda_train_54786ffd01bd9b6bb416",
        "launch": {
            "block": (288, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115200,
            "use_pdl": False,
        },
        "library": "cake_kda_train_54786ffd01bd9b6bb416",
        "logical": {
            "beta": "beta_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dAkk": "dAkk",
            "dAqk": "dAqk",
            "db_f": "db_f",
            "db_out": "db_int",
            "dg_f": "dg_f",
            "dg_f_t": "dg_f",
            "dg_out": "dg_int",
            "dk_ext": "dk",
            "dk_f": "dk_f",
            "dk_f_t": "dk_f",
            "dq_ext": "dq",
            "dq_f": "dq_f",
            "dq_f_t": "dq_f",
            "gk": "gk",
            "group": "group",
            "k_e": "k",
            "num_heads": "num_heads",
            "num_qk_heads": "num_heads",
            "q_e": "q",
        },
        "select_rule": {"kind": "arch_in", "arches": ["sm_100a"]},
        "stage": "glm_intra",
        "variant": None,
    },
    "glm_intra_v14": {
        "arches": ["sm_103a"],
        "arg_plan": [
            ["buffer", "dAqk"],
            ["buffer", "dAkk"],
            ["buffer", "gk"],
            ["buffer", "k_e"],
            ["buffer", "q_e"],
            ["buffer", "dq_f_t"],
            ["buffer", "dk_f_t"],
            ["buffer", "dg_f_t"],
            ["buffer", "beta"],
            ["buffer", "dq_f"],
            ["buffer", "dk_f"],
            ["buffer", "dg_f"],
            ["buffer", "db_f"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "dq_ext"],
            ["buffer", "dk_ext"],
            ["buffer", "dg_ext"],
            ["buffer", "dbeta_ext"],
            ["parameter", "num_heads"],
            ["parameter", "num_qk_heads"],
            ["parameter", "group"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "chunks", "x_factor": 1},
        "kernel_symbol": "kernel_cake_kda_train_c27b09eb5f5881be2969",
        "launch": {
            "block": (288, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115200,
            "use_pdl": False,
        },
        "library": "cake_kda_train_c27b09eb5f5881be2969",
        "logical": {
            "beta": "beta_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dAkk": "dAkk",
            "dAqk": "dAqk",
            "db_f": "db_f",
            "dbeta_ext": "dbeta",
            "dg_ext": "dg",
            "dg_f": "dg_f",
            "dg_f_t": "dg_f",
            "dk_ext": "dk",
            "dk_f": "dk_f",
            "dk_f_t": "dk_f",
            "dq_ext": "dq",
            "dq_f": "dq_f",
            "dq_f_t": "dq_f",
            "gk": "gk",
            "group": "group",
            "k_e": "k",
            "num_heads": "num_heads",
            "num_qk_heads": "num_heads",
            "q_e": "q",
        },
        "select_rule": {"kind": "arch_in", "arches": ["sm_103a"]},
        "stage": "glm_intra",
        "variant": "v14",
    },
    "glm_intra_v18": {
        "arches": ["sm_107a"],
        "arg_plan": [
            ["buffer", "dAqk"],
            ["buffer", "dAkk"],
            ["buffer", "gk"],
            ["buffer", "k_e"],
            ["buffer", "q_e"],
            ["buffer", "dq_f_t"],
            ["buffer", "dk_f_t"],
            ["buffer", "dg_f_t"],
            ["buffer", "beta"],
            ["buffer", "dq_f"],
            ["buffer", "dk_f"],
            ["buffer", "dg_f"],
            ["buffer", "db_f"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "dq_ext"],
            ["buffer", "dk_ext"],
            ["buffer", "dg_ext"],
            ["buffer", "dbeta_ext"],
            ["parameter", "num_heads"],
            ["parameter", "num_qk_heads"],
            ["parameter", "group"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "chunks", "x_factor": 1},
        "kernel_symbol": "kernel_cake_kda_train_aa6b5c23c97e1cb9880a",
        "launch": {
            "block": (288, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115200,
            "use_pdl": False,
        },
        "library": "cake_kda_train_aa6b5c23c97e1cb9880a",
        "logical": {
            "beta": "beta_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dAkk": "dAkk",
            "dAqk": "dAqk",
            "db_f": "db_f",
            "dbeta_ext": "dbeta",
            "dg_ext": "dg",
            "dg_f": "dg_f",
            "dg_f_t": "dg_f",
            "dk_ext": "dk",
            "dk_f": "dk_f",
            "dk_f_t": "dk_f",
            "dq_ext": "dq",
            "dq_f": "dq_f",
            "dq_f_t": "dq_f",
            "gk": "gk",
            "group": "group",
            "k_e": "k",
            "num_heads": "num_heads",
            "num_qk_heads": "num_heads",
            "q_e": "q",
        },
        "select_rule": {"kind": "arch_in", "arches": ["sm_107a"]},
        "stage": "glm_intra",
        "variant": "v18",
    },
    "gate_epilogue": {
        "arches": ["sm_100a"],
        "arg_plan": [
            ["buffer", "dg_in"],
            ["buffer", "db_in"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "dg_out"],
            ["buffer", "dbeta_out"],
            ["parameter", "num_heads"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "chunks", "x_factor": 1},
        "kernel_symbol": "kernel_cake_kda_train_47a9c058e6d6c78c166d",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 4096,
            "use_pdl": False,
        },
        "library": "cake_kda_train_47a9c058e6d6c78c166d",
        "logical": {
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "db_in": "db_int",
            "dbeta_out": "dbeta",
            "dg_in": "dg_int",
            "dg_out": "dg",
            "num_heads": "num_heads",
        },
        "select_rule": {"kind": "arch_in", "arches": ["sm_100a"]},
        "stage": "gate_epilogue",
        "variant": None,
    },
}
PAIR_ROWS = 128
CHUNK = 64
HEAD_DIM = 128


def records_of_stage(stage: str) -> tuple[str, ...]:
    """The record names of ``stage`` in registry order (the ``otherwise`` record
    last, so a scan picks a positive rule before the fallback)."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; the stages are {STAGES}")
    names = [name for name, rec in MODULES.items() if rec["stage"] == stage]
    names.sort(key=lambda name: MODULES[name]["select_rule"]["kind"] == "otherwise")
    return tuple(names)


def _admits(
    rule: dict[str, Any],
    *,
    ctas: int,
    num_seqs: int,
    num_heads: int,
    sm_count: int,
    num_segments: int,
    arch: str,
    path: str,
) -> bool:
    kind = rule["kind"]
    if kind in ("always", "otherwise"):
        return True
    if kind in ("all_of", "any_of"):
        join = all if kind == "all_of" else any
        return join(
            _admits(
                sub,
                ctas=ctas,
                num_seqs=num_seqs,
                num_heads=num_heads,
                sm_count=sm_count,
                num_segments=num_segments,
                arch=arch,
                path=path,
            )
            for sub in rule["rules"]
        )
    if kind == "ctas_at_least":
        return ctas >= int(rule["min"])
    if kind == "ctas_below":
        return ctas < int(rule["max"])
    if kind == "ctas_between":
        return int(rule["min"]) <= ctas < int(rule["max"])
    if kind == "docs_at_most":
        return num_seqs <= int(rule["max"])
    if kind == "docs_above":
        return num_seqs > int(rule["min"])
    if kind == "heads_at_most":
        return num_heads <= int(rule["max"])
    if kind == "heads_above":
        return num_heads > int(rule["min"])
    if kind == "heads_at_least":
        return num_heads >= int(rule["min"])
    if kind == "heads_below":
        return num_heads < int(rule["max"])
    if kind == "chains_at_most":
        return num_seqs * num_heads <= int(rule["max"])
    if kind == "chains_above":
        return num_seqs * num_heads > int(rule["min"])
    if kind in ("slots_at_most", "slots_above"):
        slots = num_seqs * num_heads * int(rule["slices"])
        bound = sm_count * int(rule["ctas_per_sm"])
        return slots <= bound if kind == "slots_at_most" else slots > bound
    if kind in (
        "walk_ctas_below",
        "walk_ctas_at_most",
        "walk_ctas_at_least",
        "walk_ctas_above",
    ):
        walk = num_segments * num_heads
        limit = float(rule["waves"]) * sm_count  # waves may be fractional
        if kind == "walk_ctas_below":
            return walk < limit
        if kind == "walk_ctas_at_most":
            return walk <= limit
        if kind == "walk_ctas_at_least":
            return walk >= limit
        return walk > limit
    if kind == "segments_heads_at_most":
        return num_segments * num_heads <= int(rule["max"])
    if kind == "segments_heads_above":
        return num_segments * num_heads > int(rule["min"])
    if kind == "route_in":
        return ("seg" if num_segments > 0 else "serial") in rule["routes"]
    if kind in ("carry_two_level", "carry_chain"):
        choice = carry_choice(
            num_seqs, num_heads, sm_count, num_segments, int(rule["s_hops"])
        )
        return (choice is not None) if kind == "carry_two_level" else choice is None
    if kind == "arch_in":
        return arch in rule["arches"]
    if kind == "path_in":
        return path in rule["paths"]
    raise ValueError(f"unknown Cake KDA training selection rule: {kind!r}")


def segment_plan(
    num_seqs: int, num_heads: int, sm_count: int, arch: str | None = None
) -> tuple[str, int, int]:
    """``(route, segments, num_segments)`` of the state recurrences: ``("seg", P,
    num_seqs * P)`` with ``P = clamp(ceil(slot_factor * sm_count / (num_seqs *
    num_heads)), 1, max_segments)``, at most ``sm_count // num_heads`` (and at
    least one: one document's walk CTAs fit one wave), capped at the few-chain
    count when ``num_seqs * num_heads <= small_max_chains`` (``arch``'s entry of
    ``small_segments_by_arch``, else ``small_segments``; ``arch=None`` is the
    architecture-independent count), when ``num_heads <= SEGMENTS["max_heads"]``
    or ``num_heads`` is in ``arch``'s entry of ``SEGMENTS["plan_heads_by_arch"]``
    (the 16-head rows of ``sm_103a`` / ``sm_107a``; ``arch=None`` has no entry),
    else the serial walk ``("serial", 1, 0)``.  A pure function of the scalars:
    the route the ``route_in`` rules of the records read.
    """
    n, h, sm = int(num_seqs), int(num_heads), int(sm_count)
    if n < 1 or h < 1 or sm < 1:
        raise ValueError(
            f"segment_plan: num_seqs={num_seqs}, num_heads={num_heads}, "
            f"sm_count={sm_count} must all be >= 1"
        )
    plan_heads = SEGMENTS["plan_heads_by_arch"].get(arch, ()) if arch else ()
    if h > int(SEGMENTS["max_heads"]) and h not in plan_heads:
        return ("serial", 1, 0)
    segments = -(-int(SEGMENTS["slot_factor"]) * sm // (n * h))
    segments = min(int(SEGMENTS["max_segments"]), max(1, segments))
    # one walk CTA per SM per document at most
    segments = max(1, min(segments, sm // h))
    if n * h <= int(SEGMENTS["small_max_chains"]):
        small = SEGMENTS["small_segments_by_arch"].get(arch, SEGMENTS["small_segments"])
        segments = min(segments, int(small))
    return ("seg", segments, n * segments)


def segment_args(
    num_seqs: int, num_heads: int, sm_count: int, arch: str | None = None
) -> tuple[int, int]:
    """``(proportional, seg_arg)`` of the segmented route's programs: with at most
    ``SEGMENTS["prop_max_chains"]`` (document, head) chains ``seg_table`` gives the
    documents proportional segment counts (``proportional = 1``) and the walk
    programs take ``seg_arg = -num_seqs`` (the chains' bounds come from the
    table); otherwise ``(0, segments)``, every document ``segments`` long.  On
    the serial route ``(0, 1)`` (no program reads them).  ``arch`` selects the
    few-chain segment count (:func:`segment_plan`)."""
    route, segments, _num_segments = segment_plan(num_seqs, num_heads, sm_count, arch)
    proportional = int(
        route == "seg"
        and int(num_seqs) * int(num_heads) <= int(SEGMENTS["prop_max_chains"])
    )
    return proportional, (-int(num_seqs) if proportional else segments)


def carry_choice(
    num_seqs: int, num_heads: int, sm_count: int, num_segments: int, s_hops: int
) -> tuple[int, int] | None:
    """The two-level segment carry's host rule: ``(gsize, grid)`` when the
    segmented problem takes ``seg_carry_v8``, else ``None`` (the chain carry
    ``seg_carry``).  With ``P = num_segments / num_seqs`` segments per document
    the group size ``g`` in ``[2, P]`` minimising the chain's critical path
    ``critical_hops(P, g)`` (``P - 1`` for a single group ``G = ceil(P / g) = 1``,
    else ``max(2 g + G - 4, g + G - 3 + n_last)`` with ``n_last = P - (G - 1) g``;
    ties: the smaller grid) subject to its grid fitting the device, one CTA per
    SM: ``num_seqs * num_heads * ctas`` CTAs with ``ctas = 2`` for a single group
    and ``4 G - 4`` otherwise, or on a proportional block table (at most
    ``SEGMENTS["prop_max_chains"]`` chains: the host never reads the per-document
    counts) the bound ``num_heads * (4 * (num_segments // g) + 2 * num_seqs)``.
    The two-level carry is taken when it gains over the chain by more than the
    two level handoffs, ``critical_hops + s_hops < P - 1`` (``s_hops`` is the
    rule's, in hop units).  ``None`` on the serial route (``num_segments = 0``)
    and below two segments per document.  A pure function of the scalars.
    """
    n, h, sm, nseg = int(num_seqs), int(num_heads), int(sm_count), int(num_segments)
    if n < 1 or nseg <= 0 or nseg % n:
        return None
    p = nseg // n
    if p < 2:
        return None
    proportional = n * h <= int(SEGMENTS["prop_max_chains"])
    best: tuple[tuple[int, int], int, int] | None = None
    for g in range(2, p + 1):
        groups = -(-p // g)
        if groups == 1:
            ctas, hops = 2, p - 1
        else:
            ctas = 4 * groups - 4
            hops = max(2 * g + groups - 4, g + groups - 3 + p - (groups - 1) * g)
        grid = h * (4 * (nseg // g) + 2 * n) if proportional else n * h * ctas
        if grid > sm:
            continue
        key = (hops, grid)
        if best is None or key < best[0]:
            best = (key, g, grid)
    if best is None:
        return None
    (hops, _grid), g, grid = best
    if not hops + int(s_hops) < p - 1:
        return None
    return g, grid


def flag_words(grid: int) -> int:
    """An upper bound of the release-flag words a two-level carry launch of
    ``grid`` CTAs touches (``CARRY_FLAG_WORDS_PER_CTA`` per CTA); the host checks
    it against the ``flags`` buffer."""
    return CARRY_FLAG_WORDS_PER_CTA * int(grid)


def carry_grid(num_seqs: int, num_heads: int, num_segments: int, gsize: int) -> int:
    """The two-level carry's grid at a given group size ``gsize`` on the uniform
    block table (``P = num_segments / num_seqs`` segments per document): the CTAs
    of every document's segment groups and head, :func:`carry_choice`'s count
    (two for a single group ``G = ceil(P / gsize)``, else ``4 G - 4``).  The grid
    of a document group's launch under the whole problem's group size
    (:meth:`cake_backend.Plan.batch_plan`); it never exceeds the whole problem's
    admitted grid (a group has at most the problem's documents, and on a
    proportional block table the whole problem's bound is at or above the
    uniform count).
    """
    n, h, nseg, g = int(num_seqs), int(num_heads), int(num_segments), int(gsize)
    if n < 1 or nseg <= 0 or nseg % n or g < 2:
        raise ValueError(
            f"carry_grid: num_seqs={n}, num_segments={nseg}, gsize={g} is not a "
            "segmented problem with a group size of at least two"
        )
    groups = -(-(nseg // n) // g)
    return n * h * (2 if groups == 1 else 4 * groups - 4)


def select_record(
    stage: str,
    *,
    num_chunks_pad: int,
    num_seqs: int,
    num_heads: int,
    sm_count: int,
    arch: str,
    path: str,
) -> str | None:
    """The one record of ``stage`` the host scalars select (``None`` for an
    optional stage no record of which admits the problem).

    ``ctas = num_chunks_pad * num_heads`` is the chunk-shaped CTA count the
    preparation stages switch on (``ctas_at_least`` / ``ctas_below`` /
    ``ctas_between``: ``min <= ctas < max``); ``docs_at_most`` / ``docs_above`` read the
    document count; the recurrence stages switch on the head count
    (``heads_at_most`` / ``heads_above`` / ``heads_at_least`` / ``heads_below``),
    on the route (``route_in``: ``"seg"`` where :func:`segment_plan` cuts the
    documents into segments, else ``"serial"``), on the (document, head) chain count (``chains_at_most`` /
    ``chains_above``: ``num_seqs * num_heads``), on whether their one-wave form
    fits the device (``slots_at_most`` / ``slots_above``: ``num_seqs * num_heads
    * slices`` against ``sm_count * ctas_per_sm``) and on the segmented walk's
    CTA count (``walk_ctas_below`` / ``_at_most`` / ``_at_least`` / ``_above``:
    ``num_segments * num_heads`` against ``waves * sm_count``, ``num_segments``
    from :func:`segment_plan`) and on whether the two-level segment carry's host
    rule accepts the problem (``carry_two_level`` / ``carry_chain``:
    :func:`carry_choice` with the rule's ``s_hops``); ``arch_in`` reads the device architecture and
    ``path_in`` the call's path (``PATHS``); ``all_of`` conjoins rules and
    ``any_of`` disjoins them; ``otherwise`` is the fallback of its stage and
    ``always`` the rule of a single-program stage.  Exactly one record admits
    every problem (none, for an optional stage the path or route does not
    launch), or the registry is inconsistent.
    """
    if arch not in ARCHES:
        raise ValueError(f"arch {arch!r} is not one of {ARCHES}")
    if path not in PATHS:
        raise ValueError(f"path {path!r} is not one of {PATHS}")
    num_segments = segment_plan(num_seqs, num_heads, sm_count, arch)[2]

    def admits(name: str) -> bool:
        return _admits(
            MODULES[name]["select_rule"],
            ctas=int(num_chunks_pad) * int(num_heads),
            num_seqs=int(num_seqs),
            num_heads=int(num_heads),
            sm_count=int(sm_count),
            num_segments=num_segments,
            arch=arch,
            path=path,
        )

    admitted = [
        name
        for name in records_of_stage(stage)
        if MODULES[name]["select_rule"]["kind"] != "otherwise" and admits(name)
    ]
    if len(admitted) > 1:
        raise ValueError(
            f"stage {stage!r}: records {admitted} all admit num_chunks_pad="
            f"{num_chunks_pad}, num_seqs={num_seqs}, num_heads={num_heads}, "
            f"sm_count={sm_count}, arch={arch}, path={path}"
        )
    if admitted:
        return admitted[0]
    fallback = [
        name
        for name in records_of_stage(stage)
        if MODULES[name]["select_rule"]["kind"] == "otherwise"
    ]
    if not fallback and stage in OPTIONAL_STAGES:
        return None
    if len(fallback) != 1:
        raise ValueError(
            f"stage {stage!r}: no record admits num_chunks_pad={num_chunks_pad}, "
            f"num_seqs={num_seqs}, num_heads={num_heads}, sm_count={sm_count}, "
            f"arch={arch}, path={path} and the stage has {len(fallback)} fallback "
            "records"
        )
    return fallback[0]


def launches(name: str) -> int:
    """How many times its stage launches record ``name``: the passes of a
    ``segments_heads`` grid (the two passes of the segmented adjoint recurrence
    around ``seg_carry``; the forward recurrence's passes are the ``fwdh`` and
    ``fwdh_p3`` stages), else one."""
    rule = MODULES[name]["grid_rule"]
    return len(rule["passes"]) if rule["kind"] == "segments_heads" else 1


def launch_grid(
    name: str,
    *,
    num_chunks_pad: int,
    num_seqs: int,
    num_heads: int,
    sm_count: int,
    num_segments: int = 0,
    launch: int = 0,
    gsize: int = 0,
) -> tuple[int, int, int]:
    """The launch grid of record ``name`` under its recorded grid rule (of its
    ``launch``-th launch, for a record its stage launches more than once;
    ``gsize`` a two-level carry group size already chosen for the whole problem,
    for the launch of one document group: :func:`carry_grid`).

    ``fixed`` launches the recorded CTAs; ``chunks`` launches
    ``(x_factor * num_chunks_pad, num_heads, 1)`` (one or two CTAs per chunk
    and head, pad chunks included: the kernels treat ``chunk_len == 0`` as a
    no-op); ``chunk_blocks`` launches ``(ceil(num_chunks_pad / per_cta), 1, 1)``
    (the layout table's chunk-parallel program); ``pairs_bounded`` is
    the persistent pair grid ``min(num_chunks_pad / 2 * num_heads, ctas_per_sm *
    sm_count)``, at least one (the CTAs walk the (pair, head) items;
    ``num_chunks_pad`` is even); ``per_doc_head``
    launches ``(num_seqs * num_heads * slices, 1, 1)`` (the recurrence over the
    chunk chain of one document and head, in ``slices`` value-column slices);
    ``sm_bounded`` is the persistent grid ``min(num_chunks_pad * num_heads,
    ctas_per_sm * sm_count)``, at least one; ``docs_heads`` launches
    ``(ctas_per_chain * num_seqs * num_heads, 1, 1)`` (the carry over one
    document's segments and head, split over ``ctas_per_chain`` CTAs);
    ``carry_groups`` launches the two-level carry's grid (:func:`carry_choice`
    with the rule's ``s_hops``: the CTAs of every document's segment groups and
    head, at most ``sm_count``; with ``gsize`` the uniform-table count at that
    group size, :func:`carry_grid`); ``segments_heads`` launches ``(passes[launch] * num_segments *
    num_heads, 1, 1)`` (CTAs per segment and head: two for the forward's first
    pass, the offset walk and the transfer walk, two for the few-chain adjoint
    recurrence's value-column slices, else one; ``num_segments`` is the
    segmented route's :func:`segment_plan` count, at least one).
    """
    rule = MODULES[name]["grid_rule"]
    kind = rule["kind"]
    nc, n, h = int(num_chunks_pad), int(num_seqs), int(num_heads)
    if launch < 0 or launch >= launches(name):
        raise ValueError(f"record {name}: launch {launch} of {launches(name)} launches")
    if kind == "fixed":
        x, y, z = (int(c) for c in rule["ctas"])
        return (x, y, z)
    if kind == "chunks":
        return (int(rule["x_factor"]) * nc, h, 1)
    if kind == "chunk_blocks":
        return (max(1, -(-nc // int(rule["per_cta"]))), 1, 1)
    if kind == "pairs_bounded":
        if nc % 2:
            raise ValueError(f"record {name}: num_chunks_pad={nc} is not even")
        pairs = nc * CHUNK // PAIR_ROWS * h
        return (max(1, min(pairs, int(rule["ctas_per_sm"]) * int(sm_count))), 1, 1)
    if kind == "per_doc_head":
        return (n * h * int(rule["slices"]), 1, 1)
    if kind == "sm_bounded":
        return (max(1, min(nc * h, int(rule["ctas_per_sm"]) * int(sm_count))), 1, 1)
    if kind == "docs_heads":
        return (int(rule["ctas_per_chain"]) * n * h, 1, 1)
    if kind == "carry_groups":
        if int(gsize) > 0:
            return (carry_grid(n, h, num_segments, gsize), 1, 1)
        choice = carry_choice(n, h, sm_count, num_segments, int(rule["s_hops"]))
        if choice is None:
            raise ValueError(
                f"record {name}: the two-level carry's host rule rejects "
                f"num_seqs={n}, num_heads={h}, sm_count={sm_count}, "
                f"num_segments={num_segments}"
            )
        return (int(choice[1]), 1, 1)
    if kind == "segments_heads":
        if int(num_segments) < 1:
            raise ValueError(
                f"record {name}: num_segments={num_segments} is not a segmented "
                "route (it needs at least one segment)"
            )
        return (int(rule["passes"][launch]) * int(num_segments) * h, 1, 1)
    raise ValueError(f"unknown Cake KDA training grid rule: {kind!r}")


def arch_for_capability(capability: tuple[int, int]) -> str | None:
    """The exported architecture of one device capability, if any."""
    values = tuple(int(value) for value in capability)
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when the device
    is not an exported target)."""
    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported "
            f"Cake KDA training target; the exported architectures are {ARCHES}"
        )
    return arch


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""
    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake KDA training record {name} is not delivered for {arch}; "
            f"its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``: the
    prebuilt ``aot/`` library when a builder compiled it, else FlashInfer's JIT
    build of the same sources and flags (``cake_build.load_library``)."""
    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up through FlashInfer's ``build_jit_specs``)."""
    return cake_build.jit_specs(arch)


@functools.cache
def _argument_plan(name: str) -> tuple[tuple[str | None, int], ...]:
    """The record's plan as ``(logical name, -1)`` per kernel parameter and
    ``(None, axis)`` per grid argument, in ``run`` order; computed once."""
    record = MODULES[name]
    logical = record["logical"]
    plan: list[tuple[str | None, int]] = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            plan.append((None, "xyz".index(key[-1])))
        else:
            if key not in logical:
                raise ValueError(
                    f"record {name}: kernel parameter {key!r} has no logical name"
                )
            plan.append((logical[key], -1))
    return tuple(plan)


def run_module(
    name: str,
    values: dict[str, Any],
    *,
    grid: tuple[int, int, int],
    device_index: int,
) -> None:
    """Launch record ``name`` with logical argument ``values`` on ``grid``.

    ``values`` maps the names of ``LOGICAL_TENSORS`` / ``LOGICAL_SCALARS`` to
    tensors and Python scalars; the record maps its kernel parameters onto that
    vocabulary and ``grid`` is the plan's launch grid (:func:`launch_grid`).
    The kernel runs on the current torch stream of the device.
    """
    record = MODULES[name]
    args = []
    for logical_name, axis in _argument_plan(name):
        if logical_name is None:
            args.append(int(grid[axis]))
        else:
            try:
                args.append(values[logical_name])
            except KeyError:
                raise ValueError(
                    f"record {name} needs logical argument {logical_name!r}"
                ) from None
    module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*args)


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "CARRY_FLAG_WORDS_PER_CTA",
    "CHUNK",
    "HEAD_DIM",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "OPTIONAL_STAGES",
    "PAIR_ROWS",
    "PATHS",
    "SEGMENTS",
    "STAGES",
    "STAGES_BWD",
    "STAGES_FWD",
    "arch_for_capability",
    "carry_choice",
    "carry_grid",
    "device_arch",
    "flag_words",
    "launch_grid",
    "launches",
    "library_name",
    "load",
    "records_of_stage",
    "run_module",
    "segment_args",
    "segment_plan",
    "select_record",
    "sm_count",
    "specs_for_arch",
]
