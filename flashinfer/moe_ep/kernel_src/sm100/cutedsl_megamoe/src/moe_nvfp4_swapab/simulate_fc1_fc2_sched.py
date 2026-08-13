#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Pure-Python preview of the fused fc1+fc2 static scheduler decode."""

import sys
from enum import IntEnum
from typing import IO, Iterator, List, Optional, Tuple


class BlockPhase(IntEnum):
    None_ = 0
    Linear1 = 1
    Linear2 = 2


def ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


class FusedFc12Simulator:
    def __init__(
        self,
        offsets: List[int],
        cta_tile_token: int,
        num_fc1_n_blocks: int,
        num_fc2_n_blocks: int,
        group_hint: int,
    ):
        if any(off < 0 for off in offsets):
            raise ValueError("offsets entries must be non-negative")
        if any(b < a for a, b in zip(offsets, offsets[1:], strict=False)):
            raise ValueError("offsets must be non-decreasing (it's a cumsum)")
        if cta_tile_token <= 0 or num_fc1_n_blocks <= 0 or num_fc2_n_blocks <= 0:
            raise ValueError("tile / block counts must be positive")
        if group_hint <= 0:
            raise ValueError("group_hint must be positive")

        self.offsets = list(offsets)
        self.expert_count = len(offsets)
        self.cta_tile_token = cta_tile_token
        self.num_fc1_n_blocks = num_fc1_n_blocks
        self.num_fc2_n_blocks = num_fc2_n_blocks
        self.group_hint = group_hint
        self._reset_state()

    def _reset_state(self) -> None:
        self.current_group_idx = -1
        self.current_group_first_expert = 0
        self.current_group_last_expert_exclusive = 0
        self.current_phase = BlockPhase.None_
        self.current_expert_idx = -1
        self.current_expert_tile_start = 0
        self.current_expert_tile_end = 0
        self.current_group_fc1_subphase_end = 0
        self.current_group_end = 0
        self.cumulative_fc1_tiles_at_group_end = 0
        self.cumulative_fc2_tiles_at_group_end = 0
        self.current_token_block_count = 0
        self.current_token_offset = 0
        self.current_this_expert_token_cnt = 0
        self.exhausted = False

    def _get_token_count(self, expert_idx: int) -> int:
        if expert_idx == 0:
            return self.offsets[0]
        return self.offsets[expert_idx] - self.offsets[expert_idx - 1]

    def _get_token_block_count(self, expert_idx: int) -> int:
        return ceil_div(self._get_token_count(expert_idx), self.cta_tile_token)

    def _find_next_group_end(
        self,
        group_first_expert: int,
        base_fc1: int,
        base_fc2: int,
    ) -> Tuple[int, int, int]:
        threshold = base_fc1 + self.group_hint
        cumulative_fc1 = base_fc1
        cumulative_fc2 = base_fc2
        cursor = group_first_expert
        while cursor < self.expert_count:
            token_block_count = self._get_token_block_count(cursor)
            cumulative_fc1 += token_block_count * self.num_fc1_n_blocks
            cumulative_fc2 += token_block_count * self.num_fc2_n_blocks
            cursor += 1
            if cumulative_fc1 >= threshold:
                break
        return cursor, cumulative_fc1, cumulative_fc2

    def _advance_expert_within_phase(self) -> None:
        self.current_expert_idx += 1
        if self.current_expert_idx >= self.current_group_last_expert_exclusive:
            return
        self.current_token_offset = (
            0
            if self.current_expert_idx == 0
            else self.offsets[self.current_expert_idx - 1]
        )
        self.current_this_expert_token_cnt = self._get_token_count(
            self.current_expert_idx
        )
        self.current_token_block_count = ceil_div(
            self.current_this_expert_token_cnt, self.cta_tile_token
        )
        self.current_expert_tile_start = self.current_expert_tile_end
        if self.current_phase == BlockPhase.Linear1:
            self.current_expert_tile_end += (
                self.current_token_block_count * self.num_fc1_n_blocks
            )
        else:
            self.current_expert_tile_end += (
                self.current_token_block_count * self.num_fc2_n_blocks
            )

    def _switch_to_fc2(self) -> None:
        self.current_phase = BlockPhase.Linear2
        self.current_expert_idx = self.current_group_first_expert - 1
        self.current_expert_tile_end = self.current_group_fc1_subphase_end
        self._advance_expert_within_phase()

    def _advance_group(self) -> None:
        if self.current_group_last_expert_exclusive >= self.expert_count:
            self.exhausted = True
            return
        base_fc1 = self.cumulative_fc1_tiles_at_group_end
        base_fc2 = self.cumulative_fc2_tiles_at_group_end
        self.current_group_idx += 1
        self.current_group_first_expert = self.current_group_last_expert_exclusive
        next_end, new_fc1, new_fc2 = self._find_next_group_end(
            self.current_group_first_expert,
            base_fc1,
            base_fc2,
        )
        self.current_group_last_expert_exclusive = next_end
        self.cumulative_fc1_tiles_at_group_end = new_fc1
        self.cumulative_fc2_tiles_at_group_end = new_fc2

        group_total_fc1 = new_fc1 - base_fc1
        group_total_fc2 = new_fc2 - base_fc2
        group_start_tile = self.current_group_end
        self.current_group_fc1_subphase_end = group_start_tile + group_total_fc1
        self.current_group_end = self.current_group_fc1_subphase_end + group_total_fc2

        # Empty group (all experts inside have 0 tokens) ⇒ fc1 / fc2 sub-segments
        # are 0 wide; the outer driver will re-enter advance_group via the
        # `while linear_idx >= current_group_end` loop.
        self.current_phase = BlockPhase.Linear1
        self.current_expert_idx = self.current_group_first_expert - 1
        self.current_expert_tile_end = group_start_tile
        self._advance_expert_within_phase()

    def _decode_inside_expert(self, linear_idx: int) -> dict:
        local_id = linear_idx - self.current_expert_tile_start
        if self.current_phase == BlockPhase.Linear1:
            token_block_idx = local_id // self.num_fc1_n_blocks
            n_block_idx = local_id % self.num_fc1_n_blocks
        else:
            if self.current_token_block_count <= self.num_fc2_n_blocks:
                token_block_idx = local_id % self.current_token_block_count
                n_block_idx = local_id // self.current_token_block_count
            else:
                token_block_idx = local_id // self.num_fc2_n_blocks
                n_block_idx = local_id % self.num_fc2_n_blocks
        return {
            "linear_idx": linear_idx,
            "group_idx": self.current_group_idx,
            "expert_idx": self.current_expert_idx,
            "phase": self.current_phase,
            "token_block_idx": token_block_idx,
            "intermediate_or_hidden_block_idx": n_block_idx,
        }

    def gen_next_work(self, linear_idx: int) -> Optional[dict]:
        while linear_idx >= self.current_group_end:
            self._advance_group()
            if self.exhausted:
                return None
        if (
            self.current_phase == BlockPhase.Linear1
            and linear_idx >= self.current_group_fc1_subphase_end
        ):
            self._switch_to_fc2()
        while linear_idx >= self.current_expert_tile_end:
            self._advance_expert_within_phase()
            if self.current_expert_idx >= self.current_group_last_expert_exclusive:
                return None
        return self._decode_inside_expert(linear_idx)

    def simulate(self) -> Iterator[dict]:
        self._reset_state()
        linear_idx = 0
        while True:
            tile = self.gen_next_work(linear_idx)
            if tile is None:
                break
            yield tile
            linear_idx += 1

    def enumerate_groups(self) -> Iterator[dict]:
        """Walk the group cutting state machine and yield one dict per group.
        Mutates state; ``simulate()`` calls ``_reset_state()`` so reuse is OK."""
        self._reset_state()
        while True:
            prev_fc1 = self.cumulative_fc1_tiles_at_group_end
            prev_fc2 = self.cumulative_fc2_tiles_at_group_end
            prev_end = self.current_group_end
            self._advance_group()
            if self.exhausted:
                break
            yield {
                "group_idx": self.current_group_idx,
                "first_expert": self.current_group_first_expert,
                "last_expert_exclusive": self.current_group_last_expert_exclusive,
                "fc1_start": prev_end,
                "fc1_end_exclusive": self.current_group_fc1_subphase_end,
                "fc2_start": self.current_group_fc1_subphase_end,
                "fc2_end_exclusive": self.current_group_end,
                "group_total_fc1": (self.cumulative_fc1_tiles_at_group_end - prev_fc1),
                "group_total_fc2": (self.cumulative_fc2_tiles_at_group_end - prev_fc2),
            }

    # ------------------------------------------------------------------
    # Print helpers (shared by ``main()`` and external callers such as
    # ``moe_nvfp4_swapab.runner_fc12``).  Each accepts ``prefix`` so
    # shell-comment style (``"# "``) and plain-text style coexist.
    # ------------------------------------------------------------------

    def print_header(self, *, prefix: str = "", file: Optional[IO[str]] = None) -> None:
        """Print a one-line summary of the simulator parameters."""
        if file is None:
            file = sys.stdout
        print(
            f"{prefix}scheduler preview: "
            f"cta_tile_token={self.cta_tile_token} "
            f"num_fc1_n_blocks={self.num_fc1_n_blocks} "
            f"num_fc2_n_blocks={self.num_fc2_n_blocks} "
            f"group_hint={self.group_hint} "
            f"expert_count={self.expert_count}",
            file=file,
        )

    def print_per_expert_layout(
        self, *, prefix: str = "", file: Optional[IO[str]] = None
    ) -> Tuple[int, int]:
        """Print per-expert tile counts. Returns ``(total_fc1, total_fc2)``."""
        if file is None:
            file = sys.stdout
        print(f"{prefix}per-expert layout:", file=file)
        print(
            f"{prefix}  {'expert':>6} | {'tokens':>6} | {'m_blks':>6} | "
            f"{'fc1_tiles':>10} | {'fc2_tiles':>10}",
            file=file,
        )
        total_fc1 = 0
        total_fc2 = 0
        for expert_idx in range(self.expert_count):
            token_count = self._get_token_count(expert_idx)
            m_block_count = self._get_token_block_count(expert_idx)
            fc1_tiles = m_block_count * self.num_fc1_n_blocks
            fc2_tiles = m_block_count * self.num_fc2_n_blocks
            total_fc1 += fc1_tiles
            total_fc2 += fc2_tiles
            print(
                f"{prefix}  {'e' + str(expert_idx):>6} | "
                f"{token_count:>6d} | {m_block_count:>6d} | "
                f"{fc1_tiles:>10d} | {fc2_tiles:>10d}",
                file=file,
            )
        last_offset = self.offsets[-1] if self.offsets else 0
        print(
            f"{prefix}  {'total':>6} | {last_offset:>6d} | "
            f"{'':>6} | {total_fc1:>10d} | {total_fc2:>10d}",
            file=file,
        )
        return total_fc1, total_fc2

    def print_group_layout(
        self, *, prefix: str = "", file: Optional[IO[str]] = None
    ) -> None:
        """Print the group-cut table from ``enumerate_groups``."""
        if file is None:
            file = sys.stdout
        print(
            f"{prefix}group layout (greedy cut at cumulative fc1 >= group_hint):",
            file=file,
        )
        print(
            f"{prefix}  {'group':>5} | {'experts':>12} | "
            f"{'fc1 work_ids':>22} | {'fc2 work_ids':>22} | "
            f"{'group_total':>11}",
            file=file,
        )
        for g in self.enumerate_groups():
            experts_str = f"[e{g['first_expert']}..e{g['last_expert_exclusive'] - 1}]"
            fc1_str = f"[{g['fc1_start']}..{g['fc1_end_exclusive'] - 1}]"
            fc2_str = f"[{g['fc2_start']}..{g['fc2_end_exclusive'] - 1}]"
            total = g["group_total_fc1"] + g["group_total_fc2"]
            print(
                f"{prefix}  {'g' + str(g['group_idx']):>5} | "
                f"{experts_str:>12} | {fc1_str:>22} | "
                f"{fc2_str:>22} | {total:>11}",
                file=file,
            )

    def print_layout(self, *, prefix: str = "", file: Optional[IO[str]] = None) -> None:
        """Convenience: header + per-expert layout + group layout + total tiles.

        This is the entry point external callers (e.g. ``runner_fc12``)
        invoke to dump everything except the per-tile dispatch sequence.
        """
        if file is None:
            file = sys.stdout
        self.print_header(prefix=prefix, file=file)
        total_fc1, total_fc2 = self.print_per_expert_layout(prefix=prefix, file=file)
        self.print_group_layout(prefix=prefix, file=file)
        print(f"{prefix}total tiles = {total_fc1 + total_fc2}", file=file)


def _format_tile(tile: dict) -> str:
    phase = "fc1" if tile["phase"] == BlockPhase.Linear1 else "fc2"
    return (
        f"(work_id, expert_idx, token_tile_idx, i_or_h_tile_idx, phase): "
        f"({tile['linear_idx']}, {tile['expert_idx']}, "
        f"{tile['token_block_idx']}, {tile['intermediate_or_hidden_block_idx']}, {phase})"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--offsets",
        nargs="+",
        type=int,
        required=True,
        help="cumsum of token counts per expert (each diff must be multiple of cta-tile-token)",
    )
    parser.add_argument("--cta-tile-token", type=int, required=True)
    parser.add_argument("--num-fc1-n-blocks", type=int, required=True)
    parser.add_argument("--num-fc2-n-blocks", type=int, required=True)
    parser.add_argument("--group-hint", type=int, required=True)
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=0,
        help="if >0, also print the per-cluster sub-stream "
        "(cluster c gets linear_idx in {c, c+C, c+2C, ...})",
    )
    args = parser.parse_args()

    sim = FusedFc12Simulator(
        offsets=args.offsets,
        cta_tile_token=args.cta_tile_token,
        num_fc1_n_blocks=args.num_fc1_n_blocks,
        num_fc2_n_blocks=args.num_fc2_n_blocks,
        group_hint=args.group_hint,
    )

    sim.print_layout(prefix="# ")
    print()

    tiles = list(sim.simulate())
    for tile in tiles:
        print(_format_tile(tile))

    if args.num_clusters > 0:
        print()
        num_clusters = args.num_clusters
        print(
            f"# per-cluster dispatch (cluster c gets linear_idx in "
            f"{{c, c+{num_clusters}, c+{2 * num_clusters}, ...}}):"
        )
        for cluster_idx in range(num_clusters):
            print(f"# --- cluster {cluster_idx} ---")
            for tile in tiles[cluster_idx::num_clusters]:
                print("  " + _format_tile(tile))


if __name__ == "__main__":
    main()
