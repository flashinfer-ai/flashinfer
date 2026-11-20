"""Reserved control-region layout of the p2p_split (xnuma) transport.

Generalizes the T32/EP8 constants of the qualified V10 experiment
(``m1_xnuma.py``) to any shape. The region lives at the same offset in every
rank's segment 0 (symmetric), right after the P2P arena allocations:

  inbox [n_cross][record]   records received from each cross-NUMA source
  txrec [n_cross][record]   records this rank builds for each cross owner
  dsig  [world] u64         dispatch commit signal per source (epoch)
  done  [world] u64         return-complete signal per owner (epoch)
  txd   [world] u64         dispatch commit source words
  txf   [n_cross][fsplit] u64  return commit source words (one per forwarder CTA)
  k2cnt [experts] i32       K2 per-expert stored-token progress counters
  k2tot [experts] i32       K1 snapshot of per-expert totals for the epoch
  k2meta 64 B               snapshot epoch flag
  fscr  [n_cross][fsplit][64] i32  forwarder CTA scratch (per-expert "newly done" flags)
  sscr  [n_cross][experts][128] i32 sender scratch (per-thread route counts / write cursors)
  ret   [n_cross][fsplit][max_routes][slot]  packed return rows (receive)
  stage [n_cross][fsplit][max_routes][slot]  packed return rows (send)

record = i32 counts[experts] + i32 lists[experts][tokens_per_rank], where a list
entry is ``token * topk + slot``. slot = 16 B header (int64 entry) + one BF16
hidden row. Each rank sends one record and one input span per cross owner and
receives at most ``tokens_per_rank * topk`` rows back per cross owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


SENDER_THREADS = 128


def _up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


@dataclass(frozen=True)
class XnumaLayout:
    world: int
    n_cross: int
    experts_per_rank: int
    tokens_per_rank: int
    topk: int
    hidden: int
    fsplit: int = 2

    def __post_init__(self):
        if self.fsplit not in (1, 2):
            raise ValueError("fsplit must be 1 or 2")
        if self.n_cross <= 0 or self.n_cross >= self.world:
            raise ValueError("xnuma needs 0 < n_cross < world")
        if self.experts_per_rank > 64:
            raise ValueError("xnuma forwarder tracks completed experts in a 64-bit mask (experts_per_rank <= 64)")

    @property
    def record_i32(self) -> int:
        return self.experts_per_rank * (1 + self.tokens_per_rank)

    @property
    def record_bytes(self) -> int:
        return _up(self.record_i32 * 4, 256)

    @property
    def row_bytes(self) -> int:
        return self.hidden * 2  # BF16 combine row

    @property
    def slot_bytes(self) -> int:
        return 16 + self.row_bytes

    @property
    def max_routes(self) -> int:
        return self.tokens_per_rank * self.topk

    @property
    def scratch_bytes(self) -> int:
        return _up((self.experts_per_rank + 2) * 4, 256)

    @property
    def offsets(self) -> Dict[str, int]:
        o = {}
        o["inbox"] = 0
        o["txrec"] = o["inbox"] + self.n_cross * self.record_bytes
        o["dsig"] = o["txrec"] + self.n_cross * self.record_bytes
        o["done"] = o["dsig"] + _up(self.world * 8, 64)
        o["txd"] = o["done"] + _up(self.world * 8, 64)
        o["txf"] = o["txd"] + _up(self.world * 8, 64)
        o["k2cnt"] = o["txf"] + _up(self.n_cross * self.fsplit * 8, 64)
        o["k2tot"] = o["k2cnt"] + _up(self.experts_per_rank * 4, 256)
        o["k2meta"] = o["k2tot"] + _up(self.experts_per_rank * 4, 256)
        o["fscr"] = o["k2meta"] + 64
        o["sscr"] = o["fscr"] + self.n_cross * self.fsplit * self.scratch_bytes
        o["ret"] = _up(o["sscr"] + self.n_cross * self.experts_per_rank * SENDER_THREADS * 4, 4096)
        ret_bytes = self.n_cross * self.fsplit * self.max_routes * self.slot_bytes
        o["stage"] = o["ret"] + _up(ret_bytes, 4096)
        o["end"] = o["stage"] + _up(ret_bytes, 4096)
        return o

    @property
    def reserved_bytes(self) -> int:
        return _up(self.offsets["end"], 4096)


__all__ = ["XnumaLayout"]
