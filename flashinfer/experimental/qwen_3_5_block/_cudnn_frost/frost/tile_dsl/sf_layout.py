# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""cuDNN's F8_128x4 block-scale scale-factor layout: the atom byte rule, ONE spelling for host and device.

Every block-scale consumer in this tree -- the MXFP8 SDPA's SF TMA descriptors, the block-scale
GEMM / conv templates, the mixed MXFP8 x FP4 projection -- reads its scale factors in cuDNN's
F8_128x4 order.  The rule (``test/python/sdpa/mxfp8_quant.py::_swizzle_128x4``):

    atoms of 128 rows x 4 scale columns (512 B) are laid out row-major over the padded
    ``(rows/128) x (cols/4)`` grid; inside an atom, scale ``(r, c)`` lives at byte
    ``(r % 32) * 16 + (r // 32) * 4 + c``.

A producer that WRITES scale factors (a quantize kernel) addresses individual bytes, so it needs
the rule as arithmetic rather than as a descriptor.  The two helpers below are plain Python on
purpose: the same function body traces on ``cutlass.Int32`` inside a ``@cute.kernel`` (the DSL
overloads ``//`` / ``%`` / ``*`` / ``+`` and folds the Python-int constants) AND evaluates on host
ints for the oracle-side twins and the tests, so there is exactly one place the formula is
spelled.  Constants are Python ints, not ``cutlass.Int32(...)``, so the host path works; the
sm_107a cubins of ``quantize_mxfp8.py`` (both arms) are byte-identical to the pre-hoist inlines --
the PTX only reorders two independent ops, ptxas emits the same SASS.

Who computes the ATOM index is the caller's business -- it differs per consumer (the SDPA's Q/K
SF is per-``(b, h, s_tile)`` contiguous, its V SF is D-plane-major, the GEMM's blob is a padded
``rows x K/block`` matrix), which is why only the two atom-local pieces live here.
"""

SF_ATOM_ROWS = 128  # rows of one F8_128x4 atom (== the SDPA's Q / KV tile height)
SF_ATOM_COLS = 4  # scale columns of one atom
SF_ATOM_BYTES = SF_ATOM_ROWS * SF_ATOM_COLS  # 512: one byte per scale
SF_ATOM_LINE_ROWS = 32  # rows per 16-byte line inside an atom: (r % 32) picks the line, (r // 32) the 4-byte quarter
SF_ATOM_LINE_BYTES = SF_ATOM_BYTES // SF_ATOM_LINE_ROWS  # 16


def sf_atom_byte(r, c4, *, base=0):
    """Byte of scale ``(r, c4)`` INSIDE one atom: ``base + (r % 32) * 16 + (r // 32) * 4 + c4``.

    ``r`` is the row within its 128-row atom (``0 <= r < 128``: the caller supplies ``s % 128`` -- no hidden
    modulo, so a kernel whose ``r`` is already tile-local pays no extra op); ``c4`` the scale column within the
    atom (``0 <= c4 < 4``); ``base`` the byte where that atom starts (the caller's atom-index arithmetic --
    per-tile for the SDPA's Q/K SF, D-plane-major for its V SF, ``(c // 4) * 512`` for a row of atoms).

    ``base`` is the FIRST term, not added afterwards, on purpose: ``base + A + B + c4`` associates left to right
    exactly as the pre-hoist inlines in ``quantize_mxfp8.py`` did, and LLVM does not re-associate integer
    adds, so both arms' sm_107a cubins are byte-identical before and after the hoist (``base + (A + B + c4)``
    moved one LEA.HI).  Host ints or traced ``cutlass.Int32`` alike."""
    return base + (r % SF_ATOM_LINE_ROWS) * SF_ATOM_LINE_BYTES + (r // SF_ATOM_LINE_ROWS) * SF_ATOM_COLS + c4


def sf_atom_offset(r, c):
    """Byte of scale ``(r, c)`` in a ROW of atoms: ``(c // 4) * 512 + (r % 32) * 16 + (r // 32) * 4 + c % 4``.

    ``c`` is the scale column across the whole row (``c = k // block``, any ``c >= 0``); the atoms of one
    128-row band are contiguous, so this is the offset from the band's first atom.  This is the device-side
    inline the rowwise ``quantize_mxfp8`` arm used to spell out, hoisted so the FP4 quantizer shares it; the
    host twin is ``quantize_mxfp8.sf_byte_rowwise`` (which adds the per-``(b, h, s_tile)`` tile base)."""
    return sf_atom_byte(r, c % SF_ATOM_COLS, base=(c // SF_ATOM_COLS) * SF_ATOM_BYTES)
