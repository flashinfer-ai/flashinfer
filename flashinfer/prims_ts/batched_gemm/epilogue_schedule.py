# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""Compile-time schedules for Rubin's native BatchM epilogues.

The plan describes existing lowerings; it is not a runtime interpreter. Storage
ownership determines synchronization, so callers cannot independently disable
barriers. Explicit steps also support interleaved gated NVFP4 output.
Legacy and overlap-window epilogues keep their existing schedules.
"""

from dataclasses import asdict, dataclass, replace
from enum import IntEnum


class SmemStaging(IntEnum):
    AUTO = 0
    FRAGMENT_RING = 1
    FULL_TILE = 2


class TmemRelease(IntEnum):
    AUTO = 0
    AFTER_PRELOAD = 1
    AFTER_OUTPUT = 2


@dataclass(frozen=True)
class PreloadEpiloguePlan:
    """Resolved data lifetime and ownership, shared by preload and store tasks.

    Fragments are logical N64 slices: four per M128 half, eight per CTA.
    Eight warps read N32 halves cooperatively. Four plain-store warps use
    mixed N64/N32 reads to bound live registers; four finalizer warps read N64.
    Converted output, rather than FP32 accumulators, occupies registers/SMEM.
    """

    warps: int
    staging: SmemStaging
    load_width: int  # 0 denotes the existing mixed N64/N32 lowering.
    load_ahead: int  # At most one pending read; the register-heavy tail is serial.
    prefix_fragments: int
    retained_words: int
    release: TmemRelease
    finalize: bool
    scratch_stages: int

    @property
    def full_tile(self):
        return self.staging == SmemStaging.FULL_TILE

    @property
    def thread_owns_scratch(self):
        return self.full_tile and self.finalize

    @property
    def barrier_threads(self):
        # Plain TMA stores also require a writer rendezvous: their elected
        # issuer reads rows written by other threads, even with four warps.
        return 0 if self.thread_owns_scratch else self.warps * 32

    @property
    def store_fragment_begin(self):
        return self.prefix_fragments

    @property
    def release_after_output(self):
        return self.release == TmemRelease.AFTER_OUTPUT

    @property
    def smem_layout(self):
        return "padded_rows" if self.finalize else "swizzled_fragments"

    @property
    def output_operation(self):
        return "bulk_reduce_bf16" if self.finalize else "tma_store"

    @property
    def fragments_before_release(self):
        return 8 if self.release_after_output else self.prefix_fragments

    @property
    def tmem_reads(self):
        """(Subtile index, FP32 values/thread) in accumulator address order."""
        if self.load_width:
            return tuple((i, self.load_width) for i in range(8))
        return tuple((i, 64) for i in range(6)) + tuple((i, 32) for i in range(12, 16))

    @property
    def preload_destinations(self):
        """Location of each logical N64 fragment when preload finishes.

        ``output`` means the output operation was issued; its completion is
        governed by the store queue, not by this descriptive location.
        """
        prefix_location = "smem" if self.finalize else "output"
        return (prefix_location,) * self.prefix_fragments + ("registers",) * (
            8 - self.prefix_fragments
        )


def resolve_preload_epilogue(cfg):
    """Resolve AUTO from legacy fields and reject unsupported cross-products.

    Existing fields retain their meaning. In particular, the eight-warp
    prefix flag still accepts only 0 and 2. New controls are accepted only
    for the plain-output preload paths that actually implement them.
    """
    if cfg.epilogue_steps:
        return compile_epilogue_steps(cfg)
    try:
        staging = SmemStaging(cfg.epilogue_smem_staging)
        release = TmemRelease(cfg.epilogue_tmem_release)
    except ValueError as exc:
        raise ValueError("invalid epilogue staging or TMEM release policy") from exc
    width, ahead = cfg.epilogue_tmem_load_width, cfg.epilogue_tmem_load_ahead
    if width not in (0, 32, 64):
        raise ValueError("epilogue_tmem_load_width must be 0 (auto), 32 or 64")
    if ahead not in (-1, 0, 1):
        raise ValueError("epilogue_tmem_load_ahead must be -1 (auto), 0 or 1")
    if not cfg.dense_preload_epilogue:
        if staging or release or width or ahead != -1:
            raise ValueError(
                "epilogue schedule controls require dense_preload_epilogue=1"
            )
        return None

    warps = cfg.dense_epilogue_warps
    if warps not in (4, 8):
        raise ValueError("preload epilogues require four or eight warps")
    prefix = cfg.dense_preload_store_prefix_fragments
    if prefix not in (0, 2) or (prefix and warps != 8):
        raise ValueError(
            "dense_preload_store_prefix_fragments requires 0 or eight-warp 2"
        )
    full_tile = bool(cfg.fuse_moe_finalize and warps == 4)
    expected_staging = SmemStaging.FULL_TILE if full_tile else SmemStaging.FRAGMENT_RING
    if staging == SmemStaging.AUTO:
        staging = expected_staging
    if staging != expected_staging:
        raise ValueError(
            "epilogue_smem_staging is incompatible with this row ownership"
        )

    expected_width = 32 if warps == 8 else (64 if full_tile else 0)
    if width and width != expected_width:
        raise ValueError(
            "epilogue_tmem_load_width is incompatible with this fragment mapping"
        )
    width = expected_width
    default_ahead = int(not full_tile and not prefix)
    if ahead == -1:
        ahead = default_ahead
    # The plain four-warp lowering has a hand-pipelined mixed-width tail.
    # Full-tile and prefix paths currently lower serial reads. Eight-warp
    # prefix-zero supports either its existing pipeline or serial reads.
    if (warps == 4 and ahead != default_ahead) or (prefix and ahead):
        raise ValueError(
            "epilogue_tmem_load_ahead is unsupported for this preload schedule"
        )
    if release == TmemRelease.AUTO:
        release = TmemRelease.AFTER_PRELOAD
    if full_tile:
        prefix, retained = 8, 0
    elif warps == 4:
        prefix, retained = 1, 224
    else:
        retained = (8 - prefix) * 16
    minimum_stages = 2 if warps == 4 else 1
    if cfg.num_stages_c_smem < minimum_stages:
        raise ValueError(
            f"this epilogue requires at least {minimum_stages} C scratch stages"
        )
    return PreloadEpiloguePlan(
        warps,
        staging,
        width,
        ahead,
        prefix,
        retained,
        release,
        bool(cfg.fuse_moe_finalize),
        cfg.num_stages_c_smem,
    )


@dataclass(frozen=True)
class PreloadEpiloguePreset:
    """A reproducible epilogue preset; operand queues/CGA remain separate.

    Use ``dataclasses.replace(cfg, **preset.overrides())`` then validate the
    complete GEMM, including its shared-memory and register budgets.
    """

    warps: int
    finalize: bool
    prefix: int
    scratch_stages: int
    registers: int
    load_width: int
    load_ahead: int
    staging: SmemStaging

    def overrides(self):
        return dict(
            dense_preload_epilogue=1,
            dense_epilogue_warps=self.warps,
            fuse_moe_finalize=int(self.finalize),
            dense_preload_store_prefix_fragments=self.prefix,
            num_stages_c_smem=self.scratch_stages,
            epilogue_regs=self.registers,
            epilogue_tmem_load_width=self.load_width,
            epilogue_tmem_load_ahead=self.load_ahead,
            epilogue_smem_staging=int(self.staging),
            epilogue_tmem_release=int(TmemRelease.AFTER_PRELOAD),
        )


PLAIN_PRELOAD_4W = PreloadEpiloguePreset(
    4, False, 0, 2, 248, 0, 1, SmemStaging.FRAGMENT_RING
)
PLAIN_PRELOAD_8W = PreloadEpiloguePreset(
    8, False, 0, 1, 168, 32, 1, SmemStaging.FRAGMENT_RING
)
PLAIN_PREFIX_8W = PreloadEpiloguePreset(
    8, False, 2, 1, 168, 32, 0, SmemStaging.FRAGMENT_RING
)
FINALIZE_PRELOAD_8W = PreloadEpiloguePreset(
    8, True, 0, 1, 168, 32, 1, SmemStaging.FRAGMENT_RING
)
FINALIZE_PREFIX_8W = PreloadEpiloguePreset(
    8, True, 2, 1, 168, 32, 0, SmemStaging.FRAGMENT_RING
)
FINALIZE_FULL_TILE_4W = PreloadEpiloguePreset(
    4, True, 0, 2, 248, 64, 0, SmemStaging.FULL_TILE
)


@dataclass(frozen=True)
class EpilogueStep:
    """Compile-time FIFO transfers of logical M128 x N64 fragments.

    A load converts each read batch to packed output registers. A combined load/stage
    finishes all its loads before staging the oldest retained fragments. Emit
    and release are separate steps. Read-ahead permits one pending TMEM read
    batch within this step; the register-heavy tail may be serialized.
    For FC1 each four-warp group owns complete N64 accumulator fragments:
    four fragments per group with eight warps, eight with four warps. Processing
    produces N32 FP4 values and writes their two scale factors directly to GMEM.
    Two processed fragments fill one N64 FP4 TMA box.
    Adjacent FC1 fragments within a load step are read/processed together when
    their live payload fits. Separate one-fragment steps keep scalar batching.
    """

    load_fragments: int = 0
    stage_fragments: int = 0
    emit_fragments: int = 0
    release_tmem: bool = False
    read_ahead: int = 0


def normalize_epilogue_steps(steps):
    """Accept immutable Python steps or their JSON dictionary representation."""
    result = []
    for entry in steps:
        try:
            step = entry if isinstance(entry, EpilogueStep) else EpilogueStep(**entry)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid epilogue step: {entry!r}") from exc
        for name in ("load_fragments", "stage_fragments", "emit_fragments"):
            value = getattr(step, name)
            if type(value) is not int or not 0 <= value <= 8:
                raise ValueError(f"{name} must be an integer from 0 through 8")
        if type(step.release_tmem) is not bool or type(step.read_ahead) is not int:
            raise ValueError("release_tmem must be bool; read_ahead must be int")
        if step.read_ahead not in (0, 1) or (
            step.read_ahead and not step.load_fragments
        ):
            raise ValueError("read_ahead requires a load and accepts only 0 or 1")
        transfers = step.load_fragments + step.stage_fragments
        if step.release_tmem and (transfers or step.emit_fragments):
            raise ValueError("TMEM release must be a separate step")
        if step.emit_fragments and transfers:
            raise ValueError("emit must be a separate step")
        if not (transfers or step.emit_fragments or step.release_tmem):
            raise ValueError("empty epilogue step")
        result.append(step)
    return tuple(result)


class EpilogueOp(IntEnum):
    READ = 0
    READ_WAIT = 1
    CONVERT = 2
    STAGE = 3
    SHARED_FENCE = 4
    BARRIER = 5
    EMIT = 6
    COMMIT = 7
    SOURCE_WAIT = 8
    RELEASE = 9


@dataclass(frozen=True)
class EpilogueAction:
    op: EpilogueOp
    fragment: int = 0
    slot: int = 0
    width: int = 0
    offset: int = 0
    read_id: int = 0
    pending: int = 0


@dataclass(frozen=True)
class EpilogueStepPlan:
    steps: tuple[EpilogueStep, ...]
    actions: tuple[EpilogueAction, ...]
    warps: int
    finalize: bool
    scratch_stages: int
    retained_fragments: tuple[int, ...]
    peak_payload_words: int
    gated: bool = False

    @property
    def fragment_count(self):
        return 32 // self.warps if self.gated else 8

    @property
    def output_unit(self):
        return 2 if self.gated else (4 if self.finalize else 1)

    def fc1_fragment_index(self, fragment, warpgroup):
        """Folded accumulator index shared by TMEM, scale and output addressing."""
        return (fragment // 2 * (self.warps // 4) + warpgroup) * 2 + fragment % 2

    @property
    def thread_owns_scratch(self):
        return self.finalize and self.warps == 4

    @property
    def barrier_threads(self):
        if self.gated:
            return 128  # Independent FP4 output ring per four-warp group.
        return 0 if self.thread_owns_scratch else self.warps * 32

    @property
    def words_per_fragment(self):
        if self.gated:
            return (
                4  # 32 post-gate FP4 values; scale bytes are stored during processing.
            )
        return 32 if self.warps == 4 else 16

    @property
    def retained_words(self):
        return len(self.retained_fragments) * self.words_per_fragment

    @property
    def full_tile(self):
        # Explicit plans use their own lowering, including one-slot four-warp
        # schedules. This legacy selector must not choose the full-tile body.
        return False

    @property
    def release_after_output(self):
        # Pre/post actions already partition at the explicit release step.
        return False

    @property
    def before_release(self):
        return self.actions[
            : next(i for i, a in enumerate(self.actions) if a.op == EpilogueOp.RELEASE)
        ]

    @property
    def after_release(self):
        return self.actions[
            next(i for i, a in enumerate(self.actions) if a.op == EpilogueOp.RELEASE)
            + 1 :
        ]

    def signature(self):
        # Steps with identical lowered instructions share a compilation.
        return tuple(tuple(asdict(a).values()) for a in self.actions)

    @property
    def fc1_rolled_actions(self):
        """One output-unit body when the validated schedule repeats exactly.

        Preserve the legacy rolled FC1 loop without bypassing the planner's
        actions or waits. Grouped emissions and early release retain their
        general lowering. Fragment/slot addresses advance at runtime.
        """
        if not self.gated or self.after_release:
            return ()
        blocks, current = [], []
        for action in self.before_release:
            current.append(action)
            if action.op == EpilogueOp.COMMIT:
                blocks.append(tuple(current))
                current = []
        if current or len(blocks) != self.fragment_count // self.output_unit:
            return ()
        normalized = []
        for index, block in enumerate(blocks):
            base = index * self.output_unit
            reads = [a for a in block if a.op == EpilogueOp.READ]
            if not reads:
                return ()
            first_read = reads[0].read_id
            normalized.append(
                tuple(
                    replace(
                        a,
                        fragment=a.fragment - base
                        if a.op
                        in (
                            EpilogueOp.READ,
                            EpilogueOp.CONVERT,
                            EpilogueOp.STAGE,
                            EpilogueOp.EMIT,
                        )
                        else a.fragment,
                        slot=0
                        if a.op in (EpilogueOp.STAGE, EpilogueOp.EMIT)
                        else a.slot,
                        read_id=a.read_id - first_read
                        if a.op in (EpilogueOp.READ, EpilogueOp.CONVERT)
                        else a.read_id,
                    )
                    for a in block
                )
            )
        return normalized[0] if all(b == normalized[0] for b in normalized) else ()


def compile_epilogue_steps(cfg):
    """Validate transfers and derive queue waits, fences and writer barriers.

    A bulk group can read several slots. Its source-read wait is computed from
    the actual queue position, never from the number of allocated slots. Every
    issuer commits even for predicated-off output, keeping queue ordinals
    uniform. We compile a cyclic work-tile body: waits for previous-tile reads
    are harmless on the first tile, where the hardware queues are empty.
    """
    steps = normalize_epilogue_steps(cfg.epilogue_steps)
    if not cfg.dense_preload_epilogue:
        raise ValueError("epilogue_steps requires dense_preload_epilogue=1")
    if (
        not cfg.use_tma_store
        or cfg.epi_tile_m != 128
        or (cfg.has_fused_act and not cfg.has_gated_epilogue)
        or cfg.has_eltwise_act
        or cfg.uses_tmem_overlap_window
        or cfg.has_bias_m
        or cfg.has_per_token_sf_a
        or cfg.has_per_token_sf_b
        or cfg.aliases_c_scratch_with_ab
    ):
        raise ValueError(
            "epilogue_steps requires M128 fragments with independent TMA scratch; elementwise activation, bias, overlap and per-token scaling are unsupported"
        )
    gated = cfg.has_gated_epilogue
    if gated and not (
        cfg.uses_non_swap_fp4_gated_tma_c_scratch
        and cfg.gated_weight_group_size == 1
        and cfg.output_sf_block_size_c == 16
        and cfg.epi_tile_n == 128
        and cfg.non_swap_tmem_load_num_regs == 64
        and not cfg.fuse_moe_finalize
    ):
        raise ValueError(
            "FC1 epilogue_steps requires interleaved group1 NVFP4 output and N64 TMEM fragments"
        )
    if (
        cfg.dense_preload_store_prefix_fragments
        or cfg.epilogue_tmem_load_width
        or cfg.epilogue_tmem_load_ahead != -1
        or cfg.epilogue_smem_staging
        or cfg.epilogue_tmem_release
    ):
        raise ValueError(
            "epilogue_steps cannot be combined with legacy schedule controls"
        )
    warps, stages = cfg.dense_epilogue_warps, cfg.num_stages_c_smem
    if warps not in (4, 8) or stages not in (1, 2):
        raise ValueError("epilogue_steps supports four/eight warps and one/two C slots")
    finalize = bool(cfg.fuse_moe_finalize)
    cooperative = not (finalize and warps == 4)
    fragments = 32 // warps if gated else 8
    unit = 2 if gated else (4 if finalize else 1)
    words = 4 if gated else (32 if warps == 4 else 16)
    # Address/scale/scheduler state is not payload. These reservations match
    # the established mappings; compiler register usage still needs checking.
    # The retained FC1 processes two N64 accumulators with 144 registers.
    # Reserve its 16 non-payload words instead of excluding that known-valid
    # 128-register batch. Full task admission and compiler resource checks
    # remain separate from this payload estimate.
    budget = cfg.epilogue_regs - (16 if gated else (8 if warps == 4 else 24))
    loaded = staged = emitted = 0
    released = False
    retained = ()
    peak = read_id = 0
    logical = []
    for step in steps:
        if step.load_fragments:
            if released or loaded + step.load_fragments > fragments:
                raise ValueError(
                    f"TMEM loads must cover {fragments} fragments before release"
                )
            reads = []
            packed = (loaded - staged) * words
            fragment = loaded
            while fragment < loaded + step.load_fragments:
                if gated:
                    width = (
                        128
                        if fragment % 2 == 0
                        and fragment + 2 <= loaded + step.load_fragments
                        and packed + 128 <= budget
                        else 64
                    )
                else:
                    width = 32 if warps == 8 or packed + 64 > budget else 64
                for offset in range(0, width if gated else words * 2, width):
                    if packed + width > budget:
                        raise ValueError(
                            "epilogue schedule exceeds the payload register budget"
                        )
                    reads.append((fragment, offset, width, read_id))
                    read_id += 1
                    packed += width // 16 if gated else width // 2
                fragment += width // 64 if gated else 1
            packed = (loaded - staged) * words
            pending = False
            for i, (fragment, offset, width, rid) in enumerate(reads):
                if not pending:
                    logical.append(
                        EpilogueAction(
                            EpilogueOp.READ,
                            fragment,
                            width=width,
                            offset=offset,
                            read_id=rid,
                        )
                    )
                    logical.append(EpilogueAction(EpilogueOp.READ_WAIT))
                peak = max(peak, packed + width)
                pending = bool(
                    step.read_ahead
                    and i + 1 < len(reads)
                    and packed + width + reads[i + 1][2] <= budget
                )
                if pending:
                    nf, no, nw, nr = reads[i + 1]
                    logical.append(
                        EpilogueAction(
                            EpilogueOp.READ, nf, width=nw, offset=no, read_id=nr
                        )
                    )
                    peak = max(peak, packed + width + nw)
                logical.append(
                    EpilogueAction(
                        EpilogueOp.CONVERT,
                        fragment,
                        width=width,
                        offset=offset,
                        read_id=rid,
                    )
                )
                packed += width // 16 if gated else width // 2
                if pending:
                    logical.append(EpilogueAction(EpilogueOp.READ_WAIT))
            loaded += step.load_fragments
        if staged + step.stage_fragments > loaded:
            raise ValueError("cannot stage fragments that are not in registers")
        for fragment in range(staged, staged + step.stage_fragments):
            logical.append(
                EpilogueAction(
                    EpilogueOp.STAGE, fragment, slot=(fragment // unit) % stages
                )
            )
        staged += step.stage_fragments
        if step.emit_fragments:
            if step.emit_fragments % unit or emitted + step.emit_fragments > staged:
                raise ValueError("emit requires complete staged output units")
            logical.append(EpilogueAction(EpilogueOp.SHARED_FENCE))
            if cooperative:
                logical.append(EpilogueAction(EpilogueOp.BARRIER))
            for fragment in range(emitted, emitted + step.emit_fragments, unit):
                logical.append(
                    EpilogueAction(
                        EpilogueOp.EMIT, fragment, slot=(fragment // unit) % stages
                    )
                )
            logical.append(EpilogueAction(EpilogueOp.COMMIT))
            emitted += step.emit_fragments
        if step.release_tmem:
            if released or loaded != fragments:
                raise ValueError(
                    f"release TMEM exactly once after all {fragments} fragments are read"
                )
            retained = tuple(range(staged, loaded))
            if cooperative and not gated:
                # FC1's independent warpgroup readers already wait for their
                # TMEM loads. The TS empty barrier counts every reader across
                # both groups/CTAs, so no extra CTA rendezvous is needed.
                logical.append(EpilogueAction(EpilogueOp.BARRIER))
            logical.append(EpilogueAction(EpilogueOp.RELEASE))
            released = True
    if not released or (loaded, staged, emitted) != (fragments,) * 3:
        raise ValueError(
            f"schedule must load, stage and emit all {fragments} fragments and release TMEM"
        )

    # Queue entries are sets of slots read by each committed group. Slots hold
    # (tile, output-unit) identities until emission; overwrite before emission
    # is a semantic error, whereas overwrite after emission needs a wait.
    queue = []
    resident = {}
    output = []
    for tile in range(2):
        output = []
        issuing = set()
        for action in logical:
            if action.op == EpilogueOp.STAGE:
                identity = (tile, action.fragment // unit)
                if action.slot in resident and resident[action.slot] != identity:
                    raise ValueError(
                        "C slot capacity exceeded before its previous contents were emitted"
                    )
                if action.slot not in resident:
                    conflicts = [
                        i for i, slots in enumerate(queue) if action.slot in slots
                    ]
                    if conflicts:
                        last = max(conflicts)
                        output.append(
                            EpilogueAction(
                                EpilogueOp.SOURCE_WAIT, pending=len(queue) - last - 1
                            )
                        )
                        if cooperative:
                            output.append(EpilogueAction(EpilogueOp.BARRIER))
                        queue = queue[last + 1 :]
                    resident[action.slot] = identity
            elif action.op == EpilogueOp.EMIT:
                if resident.pop(action.slot, None) != (tile, action.fragment // unit):
                    raise ValueError(
                        "output unit is not resident in its assigned C slot"
                    )
                issuing.add(action.slot)
            elif action.op == EpilogueOp.COMMIT:
                queue.append(issuing)
                issuing = set()
                if len(queue) > 8:
                    raise ValueError("too many outstanding bulk groups")
            output.append(action)
    return EpilogueStepPlan(
        steps, tuple(output), warps, finalize, stages, retained, peak, gated
    )


def fc1_streaming_steps(*, warps=8, batch_fragments=2):
    """Express the retained FC1's per-group output loop through normal steps.

    Two N64 reads are processed together, staged and emitted before the next
    pair. Single-fragment batches are also useful for comparison. Slot waits
    and the rolled-loop lowering are derived from these same validated actions.
    """
    if warps not in (4, 8) or batch_fragments not in (1, 2):
        raise ValueError(
            "FC1 streaming supports four/eight warps and one/two-fragment batches"
        )
    body = (
        EpilogueStep(load_fragments=batch_fragments, stage_fragments=batch_fragments),
    ) * (2 // batch_fragments)
    return (body + (EpilogueStep(emit_fragments=2),)) * (16 // warps) + (
        EpilogueStep(release_tmem=True),
    )


def epilogue_steps_for_prefix(
    prefix=0, *, finalize=True, read_ahead=0, gated=False, warps=8
):
    """Build an ordered schedule with a configurable pre-release SMEM prefix.

    This is a search-space constructor, not a validity promise: the compiler
    checks scratch capacity and registers for the complete configuration.
    """
    if warps not in (4, 8) or (gated and finalize):
        raise ValueError("use four/eight warps and distinct gated/finalize modes")
    fragments = 32 // warps if gated else 8
    if type(prefix) is not int or not 0 <= prefix <= fragments:
        raise ValueError(f"prefix must be from zero through {fragments}")
    result = [EpilogueStep(load_fragments=1, stage_fragments=1) for _ in range(prefix)]
    if prefix < fragments:
        result.append(
            EpilogueStep(load_fragments=fragments - prefix, read_ahead=read_ahead)
        )
    result.append(EpilogueStep(release_tmem=True))
    unit = 2 if gated else (4 if finalize else 1)
    staged = prefix
    for emitted in range(0, fragments, unit):
        needed = max(0, emitted + unit - staged)
        if needed:
            result.append(EpilogueStep(stage_fragments=needed))
            staged += needed
        result.append(EpilogueStep(emit_fragments=unit))
    return tuple(result)
