"""Exception-word index for expansion into MMA-packed NVFP4 W4A16 scales.

Packed CSF planes retain row bases and four-bit offsets in the native
MMA-packed order. The index folds in the projection's scale-normalization
table and records complete replacement words for exceptions. Expansion writes
caller-owned native scale scratch before W4A16 or A4 prefill compute.

Each record covers four k-groups of one row slab. Its 32-byte header indexes
replacement words, optionally embedding a prefix in the record. MoE weight
preparation uses header-only records and a separate replacement-word stream.
The standalone storage builder also accepts embedded prefixes controlled by
``B12X_NVFP4_CSF_INLINE_WORDS``; compute kernels do not consume this storage.

W4A16's value table re-biases E4M3 exponents: it adds one constant to every
normal scale byte. Stored row bases include that constant, so a word is its
four rows' bases plus four codes. Words with a byte the table does not map by
that constant (zero and subnormal scales), or with a sum past the byte range,
are replacement words as well.

Slabs hold ``R = 128`` rows, or ``R = 64`` rows when the projection's row
count is 64 mod 128 (2048/TP6 = 352 channels: 704 gate/up rows). The W4A16
packed order permutes rows within 64-row groups, so a 64-row slab is one half
of a 128-row slab. A trailing atom of a k-group count that is 2 mod 4 holds
two k-groups; its other masks are zero.

Storage, in bytes:

- ``[0, 1024)``: the 256-byte value table of the plane, the folded constant
  (one byte), then zeros. Masked copies read from this header.
- one block per expert, ``S x (R + R/2 x C) + S x A x B`` bytes: the expert's
  fixed stream (``S`` slabs of ``R`` biased row bases and ``C`` k-groups of
  4-bit codes), then its exception records, ``A = ceil(C / 4)`` per slab, of
  ``B`` bytes each (``record_bytes(R, inline)``). The two code bytes of word j hold
  rows 4j and 4j + 2, then rows 4j + 1 and 4j + 3, low nibble first, so
  ``(h | h << 12) & 0x0F0F0F0F`` spreads a halfword ``h`` to the word's four
  bytes. Record words: 0, a base index such that the atom's replacement word
  ``i`` past its inline words is storage word ``base + i``; 1, the
  exception-word counts before each k-group (one byte per k-group); 2, the
  atom's replacement-word count; 4-7, one mask per k-group, where bit j marks
  word j, which holds rows 4j to 4j + 3 in packed order; from byte 32, the
  atom's first ``(B - 32) / 4`` replacement words in (k-group, word) order.
- replacement words past each atom's inline words (W4A16 scale bytes in
  packed order; the words of consecutive atoms of a slab are consecutive),
  followed by ``TAIL_BYTES`` zero bytes.

Every offset is a function of the geometry and the expert index, never of the
expert count, and records address replacement words from the storage start.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Int32, Int64, Uint32

HEADER_BYTES = 1024
RECORD_HEADER_BYTES = 32
TAIL_BYTES = 512
_EXPERT_CHUNK = 16
INLINE_WORDS_ENV = "B12X_NVFP4_CSF_INLINE_WORDS"
MAX_INLINE_WORDS = 64


def configured_inline_words() -> int:
    """Inline replacement words of 128-row records from ``B12X_NVFP4_CSF_INLINE_WORDS``."""
    import os

    value = os.environ.get(INLINE_WORDS_ENV, "0")
    try:
        words = int(value)
    except ValueError:
        words = -1
    if words < 0 or words > MAX_INLINE_WORDS or words % 4:
        raise ValueError(
            f"{INLINE_WORDS_ENV} must be a multiple of 4 from 0 to "
            f"{MAX_INLINE_WORDS}, got {value!r}"
        )
    return words


def slab_rows(rows: int) -> int:
    """Storage slab height of a plane with ``rows`` rows."""
    return 128 if int(rows) % 128 == 0 else 64


def inline_words(rows_per_slab: int, words: int) -> int:
    """Inline replacement words of a record of ``rows_per_slab`` rows.

    ``words`` is the 128-row value; a 64-row atom holds half the words, and
    its record five eighths of them, rounded up to four (32 -> 20).
    """
    words = int(words)
    if int(rows_per_slab) == 128:
        return words
    return -(-(5 * words) // 32) * 4


def record_bytes(rows_per_slab: int, words: int) -> int:
    """Record size of a slab height with ``words`` inline words per 128-row record.

    In the kernels' stage reads a word past a record's inline words comes
    from the stage's replacement-word window, or, past that, from global
    memory. A 128-row atom of 128 words has 4-5 replacement words in NVFP4
    checkpoints and about 15 with 3% out-of-window scale bytes.
    """
    return RECORD_HEADER_BYTES + 4 * inline_words(rows_per_slab, words)


@dataclass(frozen=True)
class PackedCsfScales:
    """Compressed packed scales and their exception-word index."""

    storage: torch.Tensor
    rows: int
    columns: int
    num_experts: int
    max_atom_words: int
    inline_words: int = 0

    @property
    def slab_rows(self) -> int:
        return slab_rows(self.rows)

    @property
    def slabs(self) -> int:
        return self.rows // self.slab_rows

    @property
    def atoms(self) -> int:
        return (self.columns + 3) // 4

    @property
    def record_bytes(self) -> int:
        return record_bytes(self.slab_rows, self.inline_words)

    @property
    def slab_bytes(self) -> int:
        return self.slab_rows + (self.slab_rows // 2) * self.columns

    @property
    def expert_bytes(self) -> int:
        return self.slabs * (self.slab_bytes + self.atoms * self.record_bytes)

    @property
    def words_offset(self) -> int:
        return HEADER_BYTES + self.num_experts * self.expert_bytes


def packed_slab_position(row: torch.Tensor) -> torch.Tensor:
    """Packed byte position of logical slab rows (W4A16's 64-row scale transpose)."""
    half, rr = row // 64, row % 64
    within = (rr % 8) * 8 + rr // 8
    within = (within & -4) | ((within & 1) << 1) | ((within & 2) >> 1)
    return half * 64 + within


def _as_int32_bits(values: torch.Tensor) -> torch.Tensor:
    """Unsigned 32-bit values in int64 as the int32 tensor with the same bits."""
    return torch.where(values >= 1 << 31, values - (1 << 32), values).to(torch.int32)


def _predicted(fixed: torch.Tensor, columns: int) -> torch.Tensor:
    """Bytes a layout-1 fixed stream yields, ``[E, S, C, 128]`` in packed row order."""
    bases = fixed[..., :128]
    codes = fixed[..., 128:].reshape(*fixed.shape[:2], columns, 64)
    nibbles = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(
        *fixed.shape[:2], columns, 128
    )
    return bases.unsqueeze(2) + nibbles


def _table_offset(lut: torch.Tensor) -> int:
    """The constant W4A16's value table adds to positive normal E4M3 bytes."""
    normal = torch.arange(8, 127, device=lut.device)
    mapped = lut[normal].to(torch.int64)
    delta = (mapped - normal)[mapped != 0] % 256
    return int(torch.mode(delta).values) if delta.numel() else 0


def _stored_fixed(fixed: torch.Tensor, columns: int, offset: int):
    """Biased bases ``[E, S, 128]`` and word-spreadable code bytes ``[E, S, C, 64]``."""
    bases = ((fixed[..., :128].to(torch.int16) + offset) & 255).to(torch.uint8)
    # Layout-1 byte b of word j holds rows 4j + 2b (low) and 4j + 2b + 1.
    codes = fixed[..., 128:].reshape(*fixed.shape[:2], columns, 32, 2)
    low, high = codes & 15, codes >> 4
    spread = torch.stack(
        (low[..., 0] | (low[..., 1] << 4), high[..., 0] | (high[..., 1] << 4)), -1
    )
    return bases, spread.reshape(*fixed.shape[:2], columns, 64)


def _to_slabs(tensor: torch.Tensor, rows: int, rows_per_slab: int) -> torch.Tensor:
    """``[E, S128, ..., 128 x k]`` packed-row tensors as ``[E, S, ..., R x k]`` slabs.

    The last axis holds the slab's packed rows (``k`` values per row); a 64-row
    slab is one half of it. Slabs past ``rows`` are dropped.
    """
    if rows_per_slab == 128:
        return tensor[:, : rows // 128]
    e, s = tensor.shape[:2]
    middle = tensor.shape[2:-1]
    halves = tensor.reshape(e, s, *middle, 2, tensor.shape[-1] // 2)
    halves = halves.movedim(-2, 2).reshape(e, 2 * s, *middle, tensor.shape[-1] // 2)
    return halves[:, : rows // 64]


def build_packed_csf_scales(batch, inline=None) -> PackedCsfScales:
    """Index a layout-1 ``Nvfp4CsfBatch`` for stage-wise expansion in W4A16.

    Padded batches (``logical_rows`` / ``logical_columns``) keep their logical
    geometry: 64-row slabs for 64 mod 128 rows, and a two-group last atom for
    a k-group count that is 2 mod 4. ``inline`` (default: the environment's
    configured count) is the inline replacement words of a 128-row record.
    """
    configured = configured_inline_words() if inline is None else int(inline)
    if configured < 0 or configured > MAX_INLINE_WORDS or configured % 4:
        raise ValueError("Packed CSF inline words must be a multiple of 4 up to 64")
    if batch.layout != 1 or batch.codec != 0 or batch.value_lut is None:
        raise ValueError("Packed CSF indexing requires MMA-packed byte-window planes")
    device = batch.fixed.device
    experts, padded_rows, padded_columns = batch.num_experts, batch.rows, batch.columns
    rows = getattr(batch, "logical_rows", None) or padded_rows
    columns = getattr(batch, "logical_columns", None) or padded_columns
    if rows % 64 or columns % 2 or rows > padded_rows or columns > padded_columns:
        raise ValueError(
            "Packed CSF indexing requires 64-row slabs and K32 k-group pairs"
        )
    r = slab_rows(rows)
    slabs, atoms = rows // r, (columns + 3) // 4
    words_per_group = r // 4
    inline = inline_words(r, configured)
    lut = batch.value_lut.to(device)
    offset = _table_offset(lut)
    raw = batch.exceptions
    raw = (
        raw.contiguous().view(torch.int32)
        if raw.numel()
        else raw.new_empty(0, dtype=torch.int32)
    )
    exceptions = raw.to(torch.int64) & 0xFFFFFFFF
    offsets = batch.task_offsets
    padded_slabs = padded_rows // 128
    masks_all, counts_all, words_all, fixed_all = [], [], [], []
    for first in range(0, experts, _EXPERT_CHUNK):
        last = min(experts, first + _EXPERT_CHUNK)
        n = last - first
        predicted = _predicted(batch.fixed[first:last], padded_columns)
        actual = predicted.clone()
        begin, end = int(offsets[first, 0]), int(offsets[last - 1, padded_slabs])
        entries = exceptions[begin:end]
        if entries.numel():
            per_expert = (
                offsets[first:last, padded_slabs] - offsets[first:last, 0]
            ).to(torch.int64)
            expert = torch.repeat_interleave(torch.arange(n, device=device), per_expert)
            position = entries & 0xFFFFFF
            value = (entries >> 24).to(torch.uint8)
            row, column = position // padded_columns, position % padded_columns
            actual[expert, row // 128, column, packed_slab_position(row % 128)] = value
        # A stored word is its biased bases plus codes; every other word is a
        # replacement word. Word j of a k-group holds packed rows 4j to 4j + 3.
        stored = predicted.to(torch.int16) + offset
        target = lut[actual.to(torch.int64)]
        flagged = ((stored > 255) | (stored != target))[:, :, :columns]
        flagged = flagged.reshape(n, padded_slabs, columns, 32, 4).any(-1)
        flagged = _to_slabs(flagged, rows, r)  # [n, S, C, R/4]
        words = target[:, :, :columns].reshape(n, padded_slabs, columns, 32, 4)
        words = _to_slabs(words.reshape(n, padded_slabs, columns, 128), rows, r)
        words = words.reshape(n, slabs, columns, words_per_group, 4)
        # Whole atoms: a trailing half atom's other k-groups have no words.
        if atoms * 4 != columns:
            pad = flagged.new_zeros(n, slabs, atoms * 4 - columns, words_per_group)
            flagged = torch.cat((flagged, pad), 2)
            words = torch.cat(
                (
                    words,
                    words.new_zeros(n, slabs, atoms * 4 - columns, words_per_group, 4),
                ),
                2,
            )
        bits = flagged.to(torch.int64) << torch.arange(words_per_group, device=device)
        masks_all.append(bits.sum(-1).reshape(n, slabs, atoms, 4))
        counts_all.append(flagged.sum(-1).reshape(n, slabs, atoms, 4))
        words_all.append(words[flagged].view(torch.int32).reshape(-1))
        bases, codes = _stored_fixed(batch.fixed[first:last], padded_columns, offset)
        bases = _to_slabs(bases, rows, r)
        codes = _to_slabs(codes[:, :, :columns], rows, r)
        fixed_all.append(
            torch.cat((bases, codes.reshape(n, slabs, -1)), -1).reshape(n, -1)
        )
    masks = torch.cat(masks_all)
    counts = torch.cat(counts_all)
    words = torch.cat(words_all)
    fixed = torch.cat(fixed_all)

    rb = record_bytes(r, configured)
    slab_bytes = r + (r // 2) * columns
    expert_bytes = slabs * (slab_bytes + atoms * rb)
    words_offset = HEADER_BYTES + experts * expert_bytes
    per_atom = counts.sum(-1).reshape(-1)
    first_word = torch.cumsum(per_atom, 0) - per_atom  # into ``words``
    overflow = (per_atom - inline).clamp(min=0)
    overflow_start = torch.cumsum(overflow, 0) - overflow + words_offset // 4
    prefixes = torch.cumsum(counts, -1) - counts
    if per_atom.numel() and (
        int(prefixes.max()) > 255 or int((overflow_start + overflow).max()) >= 1 << 31
    ):
        raise ValueError("Packed CSF exception counts exceed the record fields")
    prefix_word = (prefixes << (8 * torch.arange(4, device=device))).sum(-1).reshape(-1)
    records = torch.zeros(per_atom.numel(), rb // 4, dtype=torch.int64, device=device)
    records[:, 0] = overflow_start - inline
    records[:, 1] = prefix_word
    records[:, 2] = per_atom
    records[:, 4:8] = masks.reshape(-1, 4)
    words64 = words.to(torch.int64) & 0xFFFFFFFF
    if words.numel() and inline:
        slot = torch.arange(inline, device=device)
        index = first_word[:, None] + slot
        valid = slot < per_atom[:, None]
        records[:, 8:] = torch.where(
            valid, words64[index.clamp(max=words.numel() - 1)], 0
        )
    if words.numel():
        atom_of_word = torch.repeat_interleave(
            torch.arange(per_atom.numel(), device=device), per_atom
        )
        rank = torch.arange(words.numel(), device=device) - first_word[atom_of_word]
        spilled = words[rank >= inline]
    else:
        spilled = words
    records = _as_int32_bits(records).view(torch.uint8).reshape(experts, -1)

    total = words_offset + spilled.numel() * 4 + TAIL_BYTES
    storage = torch.zeros(total, dtype=torch.uint8, device=device)
    storage[:256] = lut
    storage[256] = offset
    blocks = storage[HEADER_BYTES:words_offset].view(experts, expert_bytes)
    blocks[:, : slabs * slab_bytes] = fixed
    blocks[:, slabs * slab_bytes :] = records
    if spilled.numel():
        storage[words_offset : words_offset + spilled.numel() * 4] = spilled.view(
            torch.uint8
        )
    return PackedCsfScales(
        storage=storage,
        rows=rows,
        columns=columns,
        num_experts=experts,
        max_atom_words=int(per_atom.max()) if per_atom.numel() else 0,
        inline_words=configured,
    )


def expand_packed_csf_scales(scales: PackedCsfScales) -> torch.Tensor:
    """Reference expansion to ``[E, C, R]`` W4A16 scale bytes."""
    s = scales.storage
    experts, rows, columns = scales.num_experts, scales.rows, scales.columns
    r, slabs, atoms = scales.slab_rows, scales.slabs, scales.atoms
    words_per_group = r // 4
    inline = inline_words(r, scales.inline_words)
    blocks = s[HEADER_BYTES : scales.words_offset].view(experts, scales.expert_bytes)
    fixed = blocks[:, : slabs * scales.slab_bytes].reshape(experts, slabs, -1)
    bases = (
        fixed[..., :r].to(torch.int16).reshape(experts, slabs, 1, words_per_group, 4)
    )
    codes = fixed[..., r:].reshape(experts, slabs, columns, words_per_group, 2)
    # Bytes 0 and 1 of a word are the low nibbles of its two code bytes,
    # bytes 2 and 3 their high nibbles.
    nibbles = torch.cat((codes & 15, codes >> 4), -1)
    raw = ((bases + nibbles.to(torch.int16)) & 255).to(torch.uint8)
    flat = (
        raw.contiguous()
        .view(torch.int32)
        .reshape(experts, slabs, columns, words_per_group)
    )
    records = blocks[:, slabs * scales.slab_bytes :].contiguous().view(torch.int32)
    records = records.to(torch.int64).reshape(experts, slabs, atoms, -1) & 0xFFFFFFFF
    storage_words = (
        s[: s.numel() // 4 * 4].view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    )
    masks = records[..., 4:8]
    jw = torch.arange(words_per_group, device=s.device)
    flagged = ((masks.unsqueeze(-1) >> jw) & 1).bool()  # [E, S, A, 4, W]
    before = torch.cumsum(flagged.to(torch.int64), -1) - flagged.to(torch.int64)
    prefix = (records[..., 1:2] >> (8 * torch.arange(4, device=s.device))) & 255
    index = prefix.unsqueeze(-1) + before  # [E, S, A, 4, W]
    if inline:
        inline_values = records[..., 8:]  # [E, S, A, inline]
        from_record = torch.gather(
            inline_values,
            -1,
            index.clamp(max=inline - 1).reshape(experts, slabs, atoms, -1),
        ).reshape(index.shape)
    else:
        from_record = index
    from_storage = storage_words[
        (records[..., 0:1, None] + index).clamp(max=storage_words.numel() - 1)
    ]
    replaced = torch.where(index < inline, from_record, from_storage)
    replaced = _as_int32_bits(replaced).reshape(
        experts, slabs, atoms * 4, words_per_group
    )
    mask = flagged.reshape(experts, slabs, atoms * 4, words_per_group)[:, :, :columns]
    flat = torch.where(mask, replaced[:, :, :columns], flat)
    out = flat.contiguous().view(torch.uint8).reshape(experts, slabs, columns, r)
    return out.permute(0, 2, 1, 3).reshape(experts, columns, rows).contiguous()


# ---------------------------------------------------------------------------
# Whole-expert expansion for calls too large to rebuild scales per stage.
# ---------------------------------------------------------------------------

PACKED_STORAGE_LAYOUT = 2


@dataclass(frozen=True)
class PackedCsfPlane:
    """Stage-readable storage presented to the routed NVFP4-CSF expansion pair."""

    fixed: torch.Tensor  # the whole storage
    exceptions: torch.Tensor  # unused placeholder
    task_offsets: torch.Tensor  # unused placeholder
    value_lut: torch.Tensor
    rows: int
    columns: int
    num_experts: int
    # Stage-readable storage has no codec choice: this slot of the geometry
    # carries the inline replacement words of its 128-row records.
    codec: int = 0
    layout: int = PACKED_STORAGE_LAYOUT

    @classmethod
    def of(cls, scales: PackedCsfScales) -> PackedCsfPlane:
        storage = scales.storage
        placeholder = storage[:16]
        return cls(
            fixed=storage,
            exceptions=placeholder,
            task_offsets=storage[:16].view(torch.int64),
            value_lut=storage[:256],
            rows=scales.rows,
            columns=scales.columns,
            num_experts=scales.num_experts,
            codec=scales.inline_words,
        )

    @property
    def geometry(self):
        return self.rows, self.columns, self.codec, self.layout

    def validate(self):
        if self.rows % 64 or self.columns % 2 or self.fixed.device.type != "cuda":
            raise ValueError(
                "Packed CSF storage needs 64-row slabs and k-group pairs on CUDA"
            )


def _load_at(pointer, offset: Int64):
    return cute.make_tensor(pointer + offset, cute.make_layout(1))[0]


class PackedStoragePlane:
    """Expand one slab into native packed scales per 256-thread CTA."""

    def __init__(self, geometry):
        self.rows, self.columns, configured, _ = map(int, geometry)
        self.slab_rows = slab_rows(self.rows)
        self.tasks = self.rows // self.slab_rows
        self.words_per_group = self.slab_rows // 4
        self.slab_bytes = self.slab_rows + (self.slab_rows // 2) * self.columns
        self.atoms = (self.columns + 3) // 4
        self.record_bytes = record_bytes(self.slab_rows, configured)
        self.inline_words = inline_words(self.slab_rows, configured)
        self.expert_bytes = self.tasks * (
            self.slab_bytes + self.atoms * self.record_bytes
        )

    @cute.jit
    def tensors(
        self, fixed, exceptions, offsets, output, lut, experts, exception_bytes
    ):
        return fixed, output

    @cute.jit
    def decode(self, tensors, expert, task, tid):
        storage, output = tensors
        bytes16 = cute.recast_ptr(storage, dtype=cutlass.Uint16)
        words = cute.recast_ptr(storage, dtype=cutlass.Uint32)
        out = cute.recast_ptr(output, dtype=cutlass.Uint32)
        block = Int64(HEADER_BYTES) + Int64(expert) * Int64(self.expert_bytes)
        slab = block + Int64(task) * Int64(self.slab_bytes)
        records = (
            block
            + Int64(self.tasks * self.slab_bytes)
            + Int64(task) * Int64(self.atoms * self.record_bytes)
        )
        destination = Int64(expert) * Int64(self.rows * self.columns) + Int64(
            task
        ) * Int64(self.slab_rows)
        word = tid
        while word < Int32(self.columns * self.words_per_group):
            group = word // Int32(self.words_per_group)
            lane = word % Int32(self.words_per_group)
            codes = _load_at(
                bytes16,
                (
                    slab
                    + Int64(self.slab_rows + 2 * lane)
                    + Int64(group) * Int64(self.slab_rows // 2)
                )
                // Int64(2),
            ).to(Uint32)
            codes = (codes | (codes << Uint32(12))) & Uint32(0x0F0F0F0F)
            value = (codes + _load_at(words, (slab + Int64(4 * lane)) // Int64(4))).to(
                Uint32
            )
            record = (
                records + Int64(group // Int32(4)) * Int64(self.record_bytes)
            ) // Int64(4)
            mask = _load_at(words, record + Int64(4 + group % Int32(4)))
            bit = Uint32(1) << lane.to(Uint32)
            if (mask & bit) != Uint32(0):
                prefixes = _load_at(words, record + Int64(1))
                index = (
                    (prefixes >> ((group % Int32(4)).to(Uint32) * Uint32(8)))
                    & Uint32(255)
                ) + cute.arch.popc(mask & (bit - Uint32(1))).to(Uint32)
                if index < Uint32(self.inline_words):
                    value = _load_at(words, record + Int64(8) + index.to(Int64)).to(
                        Uint32
                    )
                else:
                    value = _load_at(
                        words, _load_at(words, record).to(Int64) + index.to(Int64)
                    ).to(Uint32)
            out[
                (destination + Int64(group) * Int64(self.rows) + Int64(4 * lane))
                // Int64(4)
            ] = value
            word += Int32(256)
