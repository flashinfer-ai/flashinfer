"""Stage-readable index of MMA-packed NVFP4-CSF scale planes for W4A16.

W4A16 reads expert block scales one pipeline stage at a time: G k-groups of
one or more 128-row slabs, in its MMA-packed byte order. Packed CSF planes
(layout 1) store each slab's row bases and 4-bit codes in that order. This
storage keeps them with the plane's value table folded in and adds, per atom
of four k-groups of one slab (128 four-byte words), an exception record and the
complete replacement words. A kernel rebuilds any stage in shared memory with
one add per word instead of expanding whole experts into scratch before every
MoE layer.

W4A16's value table re-biases E4M3 exponents: it adds one constant to every
normal scale byte. Stored row bases include that constant, so a word is its
four rows' bases plus four codes. Words with a byte the table does not map by
that constant (zero and subnormal scales), or with a sum past the byte range,
are replacement words as well.

Storage, in bytes:

- ``[0, 1024)``: the 256-byte value table of the plane, the folded constant
  (one byte), then zeros. Masked copies read from this header.
- one block per expert, ``S x (128 + 64 C) + S x A x 32`` bytes: the expert's
  fixed stream (``S`` slabs of 128 biased row bases and ``C`` k-groups of 4-bit
  codes), then its exception records, ``A = C / 4`` per slab. The two code
  bytes of word j hold rows 4j and 4j + 2, then rows 4j + 1 and 4j + 3, low
  nibble first, so ``(h | h << 12) & 0x0F0F0F0F`` spreads a halfword ``h`` to
  the word's four bytes. Record words 0-2 hold the storage word index of the
  atom's first replacement word, the exception-word counts before each k-group
  (one byte per k-group) and the index after its last replacement word. Words
  4-7 are one mask per k-group; bit j marks word j, which holds rows 4j to
  4j + 3 in packed order.
- replacement words (W4A16 scale bytes in packed order), followed by
  ``TAIL_BYTES`` zero bytes so fixed-size stage copies stay in bounds.

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
RECORD_BYTES = 32
TAIL_BYTES = 512
_EXPERT_CHUNK = 16


@dataclass(frozen=True)
class PackedCsfScales:
    """Inline-readable packed CSF scales of one projection."""

    storage: torch.Tensor
    rows: int
    columns: int
    num_experts: int
    max_atom_words: int

    @property
    def slabs(self) -> int:
        return self.rows // 128

    @property
    def atoms(self) -> int:
        return self.columns // 4

    @property
    def slab_bytes(self) -> int:
        return 128 + 64 * self.columns

    @property
    def expert_bytes(self) -> int:
        return self.slabs * (self.slab_bytes + self.atoms * RECORD_BYTES)

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


def _stored_fixed(fixed: torch.Tensor, columns: int, offset: int) -> torch.Tensor:
    """Layout-1 fixed streams with biased bases and word-spreadable code bytes."""
    bases = ((fixed[..., :128].to(torch.int16) + offset) & 255).to(torch.uint8)
    # Layout-1 byte b of word j holds rows 4j + 2b (low) and 4j + 2b + 1.
    codes = fixed[..., 128:].reshape(*fixed.shape[:2], columns, 32, 2)
    low, high = codes & 15, codes >> 4
    spread = torch.stack(
        (low[..., 0] | (low[..., 1] << 4), high[..., 0] | (high[..., 1] << 4)), -1
    )
    return torch.cat((bases, spread.reshape(*fixed.shape[:2], -1)), -1)


def _stored_values(fixed: torch.Tensor, columns: int) -> torch.Tensor:
    """Sums a stored fixed stream yields, ``[E, S, C, 128]`` in packed row order."""
    bases = fixed[..., :128].to(torch.int16)
    codes = fixed[..., 128:].reshape(*fixed.shape[:2], columns, 32, 2).to(torch.int16)
    nibbles = torch.cat((codes & 15, codes >> 4), -1)
    return bases.unsqueeze(2) + nibbles.reshape(*fixed.shape[:2], columns, 128)


def build_packed_csf_scales(batch) -> PackedCsfScales:
    """Index a layout-1 ``Nvfp4CsfBatch`` for stage-wise expansion in W4A16."""
    if batch.layout != 1 or batch.codec != 0 or batch.value_lut is None:
        raise ValueError("Packed CSF indexing requires MMA-packed byte-window planes")
    if batch.columns % 4:
        raise ValueError("Packed CSF indexing requires whole four-k-group atoms")
    device = batch.fixed.device
    experts, rows, columns = batch.num_experts, batch.rows, batch.columns
    slabs, atoms = rows // 128, columns // 4
    lut = batch.value_lut.to(device)
    offset = _table_offset(lut)
    raw = batch.exceptions
    raw = raw.contiguous().view(torch.int32) if raw.numel() else raw.new_empty(0, dtype=torch.int32)
    exceptions = raw.to(torch.int64) & 0xFFFFFFFF
    offsets = batch.task_offsets
    masks_all, counts_all, words_all = [], [], []
    for first in range(0, experts, _EXPERT_CHUNK):
        last = min(experts, first + _EXPERT_CHUNK)
        predicted = _predicted(batch.fixed[first:last], columns)
        actual = predicted.clone()
        begin, end = int(offsets[first, 0]), int(offsets[last - 1, slabs])
        entries = exceptions[begin:end]
        if entries.numel():
            per_expert = (offsets[first:last, slabs] - offsets[first:last, 0]).to(
                torch.int64
            )
            expert = torch.repeat_interleave(
                torch.arange(last - first, device=device), per_expert
            )
            position = entries & 0xFFFFFF
            value = (entries >> 24).to(torch.uint8)
            row, column = position // columns, position % columns
            actual[expert, row // 128, column, packed_slab_position(row % 128)] = value
        # A stored word is its biased bases plus codes; every other word is a
        # replacement word. Word j of a k-group holds packed rows 4j to 4j + 3.
        stored = predicted.to(torch.int16) + offset
        target = lut[actual.to(torch.int64)]
        flagged = (
            ((stored > 255) | (stored != target))
            .reshape(last - first, slabs, columns, 32, 4)
            .any(-1)
        )
        bits = flagged.to(torch.int64) << torch.arange(32, device=device)
        masks_all.append(bits.sum(-1).reshape(last - first, slabs, atoms, 4))
        counts_all.append(flagged.sum(-1).reshape(last - first, slabs, atoms, 4))
        words = target.reshape(last - first, slabs, columns, 32, 4)
        words_all.append(words[flagged].view(torch.int32).reshape(-1))
    masks = torch.cat(masks_all)
    counts = torch.cat(counts_all)
    words = torch.cat(words_all)

    slab_bytes = 128 + 64 * columns
    expert_bytes = slabs * (slab_bytes + atoms * RECORD_BYTES)
    words_offset = HEADER_BYTES + experts * expert_bytes
    per_atom = counts.sum(-1).reshape(-1)
    start = torch.cumsum(per_atom, 0) - per_atom + words_offset // 4
    prefixes = torch.cumsum(counts, -1) - counts
    if per_atom.numel() and (
        int(prefixes.max()) > 255 or int((start + per_atom).max()) >= 1 << 31
    ):
        raise ValueError("Packed CSF exception counts exceed the record fields")
    prefix_word = (prefixes << (8 * torch.arange(4, device=device))).sum(-1).reshape(-1)
    records = torch.zeros(start.numel(), 8, dtype=torch.int64, device=device)
    records[:, 0] = start
    records[:, 1] = prefix_word
    records[:, 2] = start + per_atom
    records[:, 4:] = masks.reshape(-1, 4)
    records = _as_int32_bits(records).view(torch.uint8).reshape(experts, -1)

    total = words_offset + words.numel() * 4 + TAIL_BYTES
    storage = torch.zeros(total, dtype=torch.uint8, device=device)
    storage[:256] = lut
    storage[256] = offset
    blocks = storage[HEADER_BYTES:words_offset].view(experts, expert_bytes)
    blocks[:, : slabs * slab_bytes] = _stored_fixed(batch.fixed, columns, offset).reshape(
        experts, -1
    )
    blocks[:, slabs * slab_bytes :] = records
    if words.numel():
        storage[words_offset : words_offset + words.numel() * 4] = words.view(
            torch.uint8
        )
    return PackedCsfScales(
        storage=storage,
        rows=rows,
        columns=columns,
        num_experts=experts,
        max_atom_words=int(per_atom.max()) if per_atom.numel() else 0,
    )


def expand_packed_csf_scales(scales: PackedCsfScales) -> torch.Tensor:
    """Reference expansion to ``[E, C, R]`` W4A16 scale bytes."""
    s = scales.storage
    experts, rows, columns = scales.num_experts, scales.rows, scales.columns
    slabs, atoms = scales.slabs, scales.atoms
    blocks = s[HEADER_BYTES : scales.words_offset].view(experts, scales.expert_bytes)
    fixed = blocks[:, : slabs * scales.slab_bytes].reshape(experts, slabs, -1)
    raw = (_stored_values(fixed, columns) & 255).to(torch.uint8)
    records = blocks[:, slabs * scales.slab_bytes :].contiguous().view(torch.int32)
    records = records.to(torch.int64).reshape(experts, slabs, atoms, 8) & 0xFFFFFFFF
    storage_words = s[: s.numel() // 4 * 4].view(torch.int32)
    masks = records[..., 4:]
    jw = torch.arange(32, device=s.device)
    flagged = ((masks.unsqueeze(-1) >> jw) & 1).bool()
    before = torch.cumsum(flagged.to(torch.int64), -1) - flagged.to(torch.int64)
    prefix = (records[..., 1:2] >> (8 * torch.arange(4, device=s.device))) & 255
    index = records[..., 0:1, None] + prefix.unsqueeze(-1) + before
    replaced = raw.reshape(experts, slabs, atoms, 4, 32, 4).contiguous()
    flat = replaced.view(torch.int32).reshape(experts, slabs, atoms, 4, 32)
    flat[flagged] = storage_words[index[flagged]]
    out = replaced.reshape(experts, slabs, columns, 128)
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
        )

    @property
    def geometry(self):
        return self.rows, self.columns, self.codec, self.layout

    def validate(self):
        if self.rows % 128 or self.columns % 4 or self.fixed.device.type != "cuda":
            raise ValueError("Packed CSF storage needs 128-row slabs and whole atoms on CUDA")


def _load_at(pointer, offset: Int64):
    return cute.make_tensor(pointer + offset, cute.make_layout(1))[0]


class PackedStoragePlane:
    """Expands one 128-row slab of stage-readable storage per CTA (256 threads)."""

    def __init__(self, geometry):
        self.rows, self.columns, _, _ = map(int, geometry)
        self.tasks = self.rows // 128
        self.slab_bytes = 128 + 64 * self.columns
        self.atoms = self.columns // 4
        self.expert_bytes = self.tasks * (self.slab_bytes + self.atoms * RECORD_BYTES)

    @cute.jit
    def tensors(self, fixed, exceptions, offsets, output, lut, experts, exception_bytes):
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
            + Int64(task) * Int64(self.atoms * RECORD_BYTES)
        )
        destination = (
            Int64(expert) * Int64(self.rows * self.columns) + Int64(task) * Int64(128)
        )
        word = tid
        while word < Int32(self.columns * 32):
            group = word // Int32(32)
            lane = word % Int32(32)
            codes = _load_at(
                bytes16, (slab + Int64(128 + 2 * lane) + Int64(group) * Int64(64)) // Int64(2)
            ).to(Uint32)
            codes = (codes | (codes << Uint32(12))) & Uint32(0x0F0F0F0F)
            value = (codes + _load_at(words, (slab + Int64(4 * lane)) // Int64(4))).to(Uint32)
            record = (records + Int64(group // Int32(4)) * Int64(RECORD_BYTES)) // Int64(4)
            mask = _load_at(words, record + Int64(4 + group % Int32(4)))
            bit = Uint32(1) << lane.to(Uint32)
            if (mask & bit) != Uint32(0):
                prefixes = _load_at(words, record + Int64(1))
                index = (
                    _load_at(words, record)
                    + ((prefixes >> ((group % Int32(4)).to(Uint32) * Uint32(8))) & Uint32(255))
                    + cute.arch.popc(mask & (bit - Uint32(1))).to(Uint32)
                )
                value = _load_at(words, index.to(Int64)).to(Uint32)
            out[
                (destination + Int64(group) * Int64(self.rows) + Int64(4 * lane)) // Int64(4)
            ] = value
            word += Int32(256)
