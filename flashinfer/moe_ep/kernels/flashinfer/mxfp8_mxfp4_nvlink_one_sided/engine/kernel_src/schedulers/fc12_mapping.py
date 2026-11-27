"""FC12 work-tile ABI and grouped or phase-interleaved task mapping."""

import dataclasses
from enum import IntEnum
from typing import List, Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.cutlass_dsl import (
    Boolean,
    Int32,
    extract_mlir_values,
    new_from_mlir_values,
)

from ...helpers.iket_compat import iket
from ...helpers.utils import balanced_tile_rows, expert_pool_rows, token_slot_rows
from .base import SchedulerWorkTileBase


phase_bits = 16
phase_mask = (1 << phase_bits) - 1
peek_ready_bit = 1 << phase_bits


def _expert_pool_rows(mapping_state, token_count, padding_block: int):
    """One expert's row span in a pool, under the token tile's slot layout.

    Must agree bit for bit with the bases the communication component publishes,
    which use the same helper with the same token tile.
    """
    tile_n = mapping_state.mapping_cluster_tile_m
    return expert_pool_rows(
        token_count, Int32(padding_block), tile_n, token_slot_rows(tile_n)
    )


@cute.jit
def _valid_tokens_in_token_tile(
    token_count: Int32, token_block_idx: Int32, tile_n: int
) -> Int32:
    """Routes of an expert's ``token_block_idx``-th token tile.

    Dense slots fill every tile but the last; strided slots spread the expert's
    routes evenly over its tiles (``balanced_tile_rows``), as the communication
    component lays them out.
    """
    tile_rows = Int32(tile_n)
    if cutlass.const_expr(token_slot_rows(tile_n) != tile_n):
        tile_rows = balanced_tile_rows(token_count, tile_n)
    remaining_tokens = cutlass.max(token_count - token_block_idx * tile_rows, Int32(0))
    return cutlass.min(remaining_tokens, tile_rows)


class Fc12WorkTileState(IntEnum):
    """Sentinel values carried in the expert index field."""

    Done = -1


class BlockPhase(IntEnum):
    """FC1/FC2 phase encoded in one fused work tile."""

    None_ = 0
    Linear1 = 1
    Linear2 = 2


@dataclasses.dataclass(frozen=True)
class SwapAbFc12WorkTileInfo(SchedulerWorkTileBase):
    """Eight-field work tile for the swap-AB FC12 orientation."""

    storage_field_count = 8

    expert_idx: Int32
    tile_m_idx: Int32
    tile_n_idx: Int32
    cumulative_data_physical_row: Int32
    cumulative_sf_physical_row: Int32
    cumulative_token_block_count: Int32
    valid_tokens_in_cta_tile: Int32
    phase_and_flags: Int32

    @property
    def is_valid_tile(self):
        return self.expert_idx >= Int32(0)

    @property
    def phase(self) -> Int32:
        return self.phase_and_flags & Int32(phase_mask)

    @property
    def peek_ready(self):
        return self.phase_and_flags >= Int32(peek_ready_bit)

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for field in (
            self.expert_idx,
            self.tile_m_idx,
            self.tile_n_idx,
            self.cumulative_data_physical_row,
            self.cumulative_sf_physical_row,
            self.cumulative_token_block_count,
            self.valid_tokens_in_cta_tile,
            self.phase_and_flags,
        ):
            values.extend(extract_mlir_values(field))
        return values

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "SwapAbFc12WorkTileInfo":
        if len(values) != self.storage_field_count:
            raise ValueError(
                f"SwapAbFc12WorkTileInfo expects {self.storage_field_count} MLIR values, got {len(values)}."
            )
        fields = (
            self.expert_idx,
            self.tile_m_idx,
            self.tile_n_idx,
            self.cumulative_data_physical_row,
            self.cumulative_sf_physical_row,
            self.cumulative_token_block_count,
            self.valid_tokens_in_cta_tile,
            self.phase_and_flags,
        )
        rebuilt = [
            new_from_mlir_values(field, [value])
            for field, value in zip(fields, values, strict=False)
        ]
        return type(self)(*rebuilt)

    def to_rmem(self) -> cute.Tensor:
        registers = cute.make_rmem_tensor((self.storage_field_count,), cutlass.Int32)
        registers[0] = self.expert_idx
        registers[1] = self.tile_m_idx
        registers[2] = self.tile_n_idx
        registers[3] = self.cumulative_data_physical_row
        registers[4] = self.cumulative_sf_physical_row
        registers[5] = self.cumulative_token_block_count
        registers[6] = self.valid_tokens_in_cta_tile
        registers[7] = self.phase_and_flags
        return registers

    @classmethod
    def from_rmem(cls, registers: cute.Tensor) -> "SwapAbFc12WorkTileInfo":
        return cls(
            expert_idx=registers[0],
            tile_m_idx=registers[1],
            tile_n_idx=registers[2],
            cumulative_data_physical_row=registers[3],
            cumulative_sf_physical_row=registers[4],
            cumulative_token_block_count=registers[5],
            valid_tokens_in_cta_tile=registers[6],
            phase_and_flags=registers[7],
        )


@dataclasses.dataclass(frozen=True)
class NonSwapAbFc12WorkTileInfo(SchedulerWorkTileBase):
    """Eight-field work tile for the non-swap-AB FC12 orientation."""

    storage_field_count = 8

    expert_idx: Int32
    tile_m_idx: Int32
    tile_n_idx: Int32
    cumulative_data_physical_row: Int32
    cumulative_sf_physical_row: Int32
    cumulative_token_block_count: Int32
    valid_tokens_in_cta_cluster_tile: Int32
    phase_and_flags: Int32

    @property
    def is_valid_tile(self):
        return self.expert_idx >= Int32(0)

    @property
    def phase(self) -> Int32:
        return self.phase_and_flags & Int32(phase_mask)

    @property
    def peek_ready(self):
        return self.phase_and_flags >= Int32(peek_ready_bit)

    @property
    def valid_tokens_in_cta_tile(self) -> Int32:
        return self.valid_tokens_in_cta_cluster_tile >> Int32(16)

    @property
    def valid_tokens_in_cluster_tile(self) -> Int32:
        return self.valid_tokens_in_cta_cluster_tile & Int32(0xFFFF)

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for field in (
            self.expert_idx,
            self.tile_m_idx,
            self.tile_n_idx,
            self.cumulative_data_physical_row,
            self.cumulative_sf_physical_row,
            self.cumulative_token_block_count,
            self.valid_tokens_in_cta_cluster_tile,
            self.phase_and_flags,
        ):
            values.extend(extract_mlir_values(field))
        return values

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "NonSwapAbFc12WorkTileInfo":
        if len(values) != self.storage_field_count:
            raise ValueError(
                f"NonSwapAbFc12WorkTileInfo expects {self.storage_field_count} MLIR values, got {len(values)}."
            )
        fields = (
            self.expert_idx,
            self.tile_m_idx,
            self.tile_n_idx,
            self.cumulative_data_physical_row,
            self.cumulative_sf_physical_row,
            self.cumulative_token_block_count,
            self.valid_tokens_in_cta_cluster_tile,
            self.phase_and_flags,
        )
        rebuilt = [
            new_from_mlir_values(field, [value])
            for field, value in zip(fields, values, strict=False)
        ]
        return type(self)(*rebuilt)

    def to_rmem(self) -> cute.Tensor:
        registers = cute.make_rmem_tensor((self.storage_field_count,), cutlass.Int32)
        registers[0] = self.expert_idx
        registers[1] = self.tile_m_idx
        registers[2] = self.tile_n_idx
        registers[3] = self.cumulative_data_physical_row
        registers[4] = self.cumulative_sf_physical_row
        registers[5] = self.cumulative_token_block_count
        registers[6] = self.valid_tokens_in_cta_cluster_tile
        registers[7] = self.phase_and_flags
        return registers

    @classmethod
    def from_rmem(cls, registers: cute.Tensor) -> "NonSwapAbFc12WorkTileInfo":
        return cls(
            expert_idx=registers[0],
            tile_m_idx=registers[1],
            tile_n_idx=registers[2],
            cumulative_data_physical_row=registers[3],
            cumulative_sf_physical_row=registers[4],
            cumulative_token_block_count=registers[5],
            valid_tokens_in_cta_cluster_tile=registers[6],
            phase_and_flags=registers[7],
        )


class _Fc12TaskCursorState:
    """Register-resident cursor for the FC12 group/phase/expert state machine."""

    def __init__(
        self,
        current_group_first_expert: Int32,
        current_group_last_expert_exclusive: Int32,
        current_phase: Int32,
        current_expert_idx: Int32,
        current_expert_tile_start: Int32,
        current_expert_tile_end: Int32,
        current_group_fc1_subphase_end: Int32,
        current_group_end: Int32,
        cumulative_fc1_tiles_at_group_end: Int32,
        cumulative_fc2_tiles_at_group_end: Int32,
        current_data_cumulative: Int32,
        current_sf_cumulative: Int32,
        current_token_block_cumulative: Int32,
        group_start_data_cumulative: Int32,
        group_start_sf_cumulative: Int32,
        group_start_token_block_cumulative: Int32,
        current_token_block_count: Int32,
        current_expert_token_count: Int32,
    ) -> None:
        self.current_group_first_expert = current_group_first_expert
        self.current_group_last_expert_exclusive = current_group_last_expert_exclusive
        self.current_phase = current_phase
        self.current_expert_idx = current_expert_idx
        self.current_expert_tile_start = current_expert_tile_start
        self.current_expert_tile_end = current_expert_tile_end
        self.current_group_fc1_subphase_end = current_group_fc1_subphase_end
        self.current_group_end = current_group_end
        self.cumulative_fc1_tiles_at_group_end = cumulative_fc1_tiles_at_group_end
        self.cumulative_fc2_tiles_at_group_end = cumulative_fc2_tiles_at_group_end
        self.current_data_cumulative = current_data_cumulative
        self.current_sf_cumulative = current_sf_cumulative
        self.current_token_block_cumulative = current_token_block_cumulative
        self.group_start_data_cumulative = group_start_data_cumulative
        self.group_start_sf_cumulative = group_start_sf_cumulative
        self.group_start_token_block_cumulative = group_start_token_block_cumulative
        self.current_token_block_count = current_token_block_count
        self.current_expert_token_count = current_expert_token_count

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for field in self._fields():
            values.extend(extract_mlir_values(field))
        return values

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "_Fc12TaskCursorState":
        value_index = 0
        rebuilt = []
        for field in self._fields():
            field_value_count = len(extract_mlir_values(field))
            rebuilt.append(
                new_from_mlir_values(
                    field, values[value_index : value_index + field_value_count]
                )
            )
            value_index += field_value_count
        if value_index != len(values):
            raise ValueError(
                f"_Fc12TaskCursorState MLIR value count mismatch: consumed {value_index}, got {len(values)}."
            )
        return type(self)(*rebuilt)

    def _fields(self) -> Tuple:
        return (
            self.current_group_first_expert,
            self.current_group_last_expert_exclusive,
            self.current_phase,
            self.current_expert_idx,
            self.current_expert_tile_start,
            self.current_expert_tile_end,
            self.current_group_fc1_subphase_end,
            self.current_group_end,
            self.cumulative_fc1_tiles_at_group_end,
            self.cumulative_fc2_tiles_at_group_end,
            self.current_data_cumulative,
            self.current_sf_cumulative,
            self.current_token_block_cumulative,
            self.group_start_data_cumulative,
            self.group_start_sf_cumulative,
            self.group_start_token_block_cumulative,
            self.current_token_block_count,
            self.current_expert_token_count,
        )


class _Fc12PipelineCursorState:
    """Register-resident cursor for the pipelined FC12 order (``map_fc12_pipelined_work_id``).

    Segment ``k`` of the order holds the FC1 tiles of schedule group ``k`` and the
    FC2 tiles of group ``k - 1``. Tile ranges count tiles of one phase over every
    expert in processing order: ``fc1_group_*`` is group ``k``'s FC1 range,
    ``fc2_lag_*`` group ``k - 1``'s FC2 range, and ``fc2_group_end`` the end of
    group ``k``'s FC2 range, which lags into segment ``k + 1``. Each phase keeps
    its own expert cursor, since its tiles advance through the experts at their
    own place in the order.
    """

    def __init__(
        self,
        group_first_expert: Int32,
        group_last_expert_exclusive: Int32,
        fc1_group_begin: Int32,
        fc1_group_end: Int32,
        fc2_lag_begin: Int32,
        fc2_lag_end: Int32,
        fc2_group_end: Int32,
        segment_start: Int32,
        segment_end: Int32,
        fc1_cursor: "_PhaseFc12CursorState",
        fc2_cursor: "_PhaseFc12CursorState",
    ) -> None:
        self.group_first_expert = group_first_expert
        self.group_last_expert_exclusive = group_last_expert_exclusive
        self.fc1_group_begin = fc1_group_begin
        self.fc1_group_end = fc1_group_end
        self.fc2_lag_begin = fc2_lag_begin
        self.fc2_lag_end = fc2_lag_end
        self.fc2_group_end = fc2_group_end
        self.segment_start = segment_start
        self.segment_end = segment_end
        self.fc1_cursor = fc1_cursor
        self.fc2_cursor = fc2_cursor

    def _scalar_fields(self) -> Tuple:
        return (
            self.group_first_expert,
            self.group_last_expert_exclusive,
            self.fc1_group_begin,
            self.fc1_group_end,
            self.fc2_lag_begin,
            self.fc2_lag_end,
            self.fc2_group_end,
            self.segment_start,
            self.segment_end,
        )

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for field in (*self._scalar_fields(), self.fc1_cursor, self.fc2_cursor):
            values.extend(extract_mlir_values(field))
        return values

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "_Fc12PipelineCursorState":
        value_index = 0
        rebuilt = []
        for field in (*self._scalar_fields(), self.fc1_cursor, self.fc2_cursor):
            field_value_count = len(extract_mlir_values(field))
            rebuilt.append(
                new_from_mlir_values(
                    field, values[value_index : value_index + field_value_count]
                )
            )
            value_index += field_value_count
        if value_index != len(values):
            raise ValueError(
                f"_Fc12PipelineCursorState MLIR value count mismatch: consumed {value_index}, got {len(values)}."
            )
        return type(self)(*rebuilt)


class Fc12TaskMappingState:
    """Runtime inputs and cursor for monotonic FC12 linear-ID mapping.

    ``token_count_cache``, when present, holds the per-expert token counts in the
    mapping warp's registers: entry ``b`` of lane ``l`` is the count of expert
    ``32 * b + l`` (zero past the expert count). Seeks then read no global memory.

    ``cursor_state`` is a ``_Fc12TaskCursorState`` for the grouped order and a
    ``_Fc12PipelineCursorState`` for the pipelined order.
    """

    def __init__(
        self,
        expert_count,
        mapping_cta_tile_shape_mnk: Tuple[int, int, int],
        mapping_cluster_shape_mn: Tuple[int, int],
        group_hint: int,
        token_padding_block: int,
        sf_padding_block: int,
        is_swap_ab: bool,
        expert_token_sizes: Optional[cute.Tensor],
        expert_token_prefix_sum: Optional[cute.Tensor],
        cursor_state: _Fc12TaskCursorState,
        num_fc1_intermediate_blocks,
        num_fc2_hidden_blocks,
        token_count_cache: Optional[Tuple[Int32, ...]] = None,
    ) -> None:
        self.expert_count = expert_count
        self.mapping_cta_tile_shape_mnk = mapping_cta_tile_shape_mnk
        self.mapping_cluster_shape_mn = mapping_cluster_shape_mn
        self.group_hint = group_hint
        self.token_padding_block = token_padding_block
        self.sf_padding_block = sf_padding_block
        self.is_swap_ab = is_swap_ab
        self.expert_token_sizes = expert_token_sizes
        self.expert_token_prefix_sum = expert_token_prefix_sum
        self.cursor_state = cursor_state
        self.num_fc1_intermediate_blocks = num_fc1_intermediate_blocks
        self.num_fc2_hidden_blocks = num_fc2_hidden_blocks
        self.token_count_cache = token_count_cache

    @property
    def mapping_cluster_tile_m(self) -> int:
        return self.mapping_cta_tile_shape_mnk[0] * self.mapping_cluster_shape_mn[0]

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        if isinstance(self.expert_count, Int32):
            values.extend(extract_mlir_values(self.expert_count))
        token_counts = (
            self.expert_token_sizes
            if self.expert_token_sizes is not None
            else self.expert_token_prefix_sum
        )
        values.extend(extract_mlir_values(token_counts))
        values.extend(extract_mlir_values(self.cursor_state))
        if isinstance(self.num_fc1_intermediate_blocks, Int32):
            values.extend(extract_mlir_values(self.num_fc1_intermediate_blocks))
        if isinstance(self.num_fc2_hidden_blocks, Int32):
            values.extend(extract_mlir_values(self.num_fc2_hidden_blocks))
        if self.token_count_cache is not None:
            for entry in self.token_count_cache:
                values.extend(extract_mlir_values(entry))
        return values

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "Fc12TaskMappingState":
        value_index = 0

        def rebuild(field):
            nonlocal value_index
            field_value_count = len(extract_mlir_values(field))
            result = new_from_mlir_values(
                field, values[value_index : value_index + field_value_count]
            )
            value_index += field_value_count
            return result

        expert_count = (
            rebuild(self.expert_count)
            if isinstance(self.expert_count, Int32)
            else self.expert_count
        )
        if self.expert_token_sizes is not None:
            expert_token_sizes = rebuild(self.expert_token_sizes)
            expert_token_prefix_sum = None
        else:
            expert_token_sizes = None
            expert_token_prefix_sum = rebuild(self.expert_token_prefix_sum)
        result = type(self)(
            expert_count=expert_count,
            mapping_cta_tile_shape_mnk=self.mapping_cta_tile_shape_mnk,
            mapping_cluster_shape_mn=self.mapping_cluster_shape_mn,
            group_hint=self.group_hint,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            is_swap_ab=self.is_swap_ab,
            expert_token_sizes=expert_token_sizes,
            expert_token_prefix_sum=expert_token_prefix_sum,
            cursor_state=rebuild(self.cursor_state),
            num_fc1_intermediate_blocks=(
                rebuild(self.num_fc1_intermediate_blocks)
                if isinstance(self.num_fc1_intermediate_blocks, Int32)
                else self.num_fc1_intermediate_blocks
            ),
            num_fc2_hidden_blocks=(
                rebuild(self.num_fc2_hidden_blocks)
                if isinstance(self.num_fc2_hidden_blocks, Int32)
                else self.num_fc2_hidden_blocks
            ),
            token_count_cache=(
                None
                if self.token_count_cache is None
                else tuple(rebuild(entry) for entry in self.token_count_cache)
            ),
        )
        if value_index != len(values):
            raise ValueError(
                f"Fc12TaskMappingState MLIR value count mismatch: consumed {value_index}, got {len(values)}."
            )
        return result


@cute.jit
def create_fc12_task_mapping_state(
    *,
    expert_count,
    intermediate_gateup_size,
    hidden_size,
    mapping_cta_tile_shape_mnk: Tuple[int, int, int],
    mapping_cluster_shape_mn: Tuple[int, int],
    group_hint: int,
    token_padding_block: int,
    sf_padding_block: int,
    is_swap_ab: bool,
    expert_token_sizes: Optional[cute.Tensor],
    expert_token_prefix_sum: Optional[cute.Tensor],
    cache_token_counts: bool = False,
    pipelined_groups: bool = False,
) -> Fc12TaskMappingState:
    """Create the register-resident state for one CTA's FC12 mapper.

    With ``cache_token_counts`` the state carries a register cache of the token
    counts, zero until ``load_fc12_token_count_cache`` fills it. With
    ``pipelined_groups`` it carries the cursor of ``map_fc12_pipelined_work_id``
    instead of ``map_fc12_linear_work_id``'s.
    """
    token_count_cache = None
    if cutlass.const_expr(cache_token_counts):
        if cutlass.const_expr(
            not isinstance(expert_count, int) or expert_token_sizes is None
        ):
            raise ValueError(
                "The token count cache needs a static expert count and per-expert token sizes."
            )
        token_count_cache = (Int32(0),) * ((expert_count + 31) // 32)
    mapping_cluster_tile_n = mapping_cluster_shape_mn[1] * mapping_cta_tile_shape_mnk[1]
    num_fc1_intermediate_blocks = (
        intermediate_gateup_size + mapping_cluster_tile_n - 1
    ) // mapping_cluster_tile_n
    num_fc2_hidden_blocks = (
        hidden_size + mapping_cluster_tile_n - 1
    ) // mapping_cluster_tile_n
    if cutlass.const_expr(pipelined_groups):
        if cutlass.const_expr(
            not isinstance(intermediate_gateup_size, int)
            or not isinstance(hidden_size, int)
        ):
            raise ValueError("The pipelined FC12 order needs static expert dimensions.")
        return Fc12TaskMappingState(
            expert_count=expert_count,
            mapping_cta_tile_shape_mnk=mapping_cta_tile_shape_mnk,
            mapping_cluster_shape_mn=mapping_cluster_shape_mn,
            group_hint=group_hint,
            token_padding_block=token_padding_block,
            sf_padding_block=sf_padding_block,
            is_swap_ab=is_swap_ab,
            expert_token_sizes=expert_token_sizes,
            expert_token_prefix_sum=expert_token_prefix_sum,
            cursor_state=_Fc12PipelineCursorState(
                group_first_expert=Int32(0),
                group_last_expert_exclusive=Int32(0),
                fc1_group_begin=Int32(0),
                fc1_group_end=Int32(0),
                fc2_lag_begin=Int32(0),
                fc2_lag_end=Int32(0),
                fc2_group_end=Int32(0),
                segment_start=Int32(0),
                segment_end=Int32(0),
                fc1_cursor=_make_phase_cursor(num_fc1_intermediate_blocks),
                fc2_cursor=_make_phase_cursor(num_fc2_hidden_blocks),
            ),
            num_fc1_intermediate_blocks=num_fc1_intermediate_blocks,
            num_fc2_hidden_blocks=num_fc2_hidden_blocks,
            token_count_cache=token_count_cache,
        )
    cursor_state = _Fc12TaskCursorState(
        current_group_first_expert=Int32(0),
        current_group_last_expert_exclusive=Int32(0),
        current_phase=Int32(BlockPhase.Linear1),
        current_expert_idx=Int32(-1),
        current_expert_tile_start=Int32(0),
        current_expert_tile_end=Int32(0),
        current_group_fc1_subphase_end=Int32(0),
        current_group_end=Int32(0),
        cumulative_fc1_tiles_at_group_end=Int32(0),
        cumulative_fc2_tiles_at_group_end=Int32(0),
        current_data_cumulative=Int32(0),
        current_sf_cumulative=Int32(0),
        current_token_block_cumulative=Int32(0),
        group_start_data_cumulative=Int32(0),
        group_start_sf_cumulative=Int32(0),
        group_start_token_block_cumulative=Int32(0),
        current_token_block_count=Int32(0),
        current_expert_token_count=Int32(0),
    )
    return Fc12TaskMappingState(
        expert_count=expert_count,
        mapping_cta_tile_shape_mnk=mapping_cta_tile_shape_mnk,
        mapping_cluster_shape_mn=mapping_cluster_shape_mn,
        group_hint=group_hint,
        token_padding_block=token_padding_block,
        sf_padding_block=sf_padding_block,
        is_swap_ab=is_swap_ab,
        expert_token_sizes=expert_token_sizes,
        expert_token_prefix_sum=expert_token_prefix_sum,
        cursor_state=cursor_state,
        num_fc1_intermediate_blocks=num_fc1_intermediate_blocks,
        num_fc2_hidden_blocks=num_fc2_hidden_blocks,
        token_count_cache=token_count_cache,
    )


@cute.jit
def load_fc12_token_count_cache(
    mapping_state: Fc12TaskMappingState,
) -> Fc12TaskMappingState:
    """Fill the token count cache from ``expert_token_sizes``; call once the sizes are final."""
    lane_idx = Int32(cute.arch.lane_idx())
    entries = []
    for batch in cutlass.range_constexpr(len(mapping_state.token_count_cache)):
        expert_idx = Int32(batch * 32) + lane_idx
        token_count = Int32(0)
        if expert_idx < Int32(mapping_state.expert_count):
            token_count = mapping_state.expert_token_sizes[expert_idx]
        entries.append(token_count)
    mapping_state.token_count_cache = tuple(entries)
    return mapping_state


@cute.jit
def count_fc12_tiles(mapping_state: Fc12TaskMappingState) -> Tuple[Int32, Int32]:
    """FC1 and FC2 cluster tiles over every expert, from the token count cache; uniform across the warp."""
    lane_idx = Int32(cute.arch.lane_idx())
    fc1_tile_count = Int32(0)
    fc2_tile_count = Int32(0)
    for batch in cutlass.range_constexpr(len(mapping_state.token_count_cache)):
        (_, _, _, _, _, lane_fc1_tiles, lane_fc2_tiles) = _load_expert_batch_metrics(
            mapping_state,
            Int32(batch * 32),
            Int32(0),
            Int32(mapping_state.expert_count),
            lane_idx,
        )
        fc1_tile_count = fc1_tile_count + Int32(
            cute.arch.warp_redux_sync(lane_fc1_tiles, "add")
        )
        fc2_tile_count = fc2_tile_count + Int32(
            cute.arch.warp_redux_sync(lane_fc2_tiles, "add")
        )
    return fc1_tile_count, fc2_tile_count


@cute.jit
def count_fc12_tiles_from_sizes(
    mapping_state: Fc12TaskMappingState,
) -> Tuple[Int32, Int32]:
    """FC1 and FC2 cluster tiles over every expert, read from the token sizes (or their cache); uniform across the warp."""
    lane_idx = Int32(cute.arch.lane_idx())
    fc1_tile_count = Int32(0)
    fc2_tile_count = Int32(0)
    for batch in cutlass.range_constexpr((mapping_state.expert_count + 31) // 32):
        (_, _, _, _, _, lane_fc1_tiles, lane_fc2_tiles) = _load_expert_batch_metrics(
            mapping_state,
            Int32(batch * 32),
            Int32(0),
            Int32(mapping_state.expert_count),
            lane_idx,
        )
        fc1_tile_count = fc1_tile_count + Int32(
            cute.arch.warp_redux_sync(lane_fc1_tiles, "add")
        )
        fc2_tile_count = fc2_tile_count + Int32(
            cute.arch.warp_redux_sync(lane_fc2_tiles, "add")
        )
    return fc1_tile_count, fc2_tile_count


@dataclasses.dataclass(frozen=True)
class SharedExpertTiles:
    """Where a shared expert's work tiles sit in the FC12 work-ID space and in the pools (all static).

    The shared expert runs every token of the rank's own staged input
    (``tokens``, the capacity), so its tiles need no plan: they are one more
    weight stream through the same grouped GEMMs. Its token tiles follow the
    plan's token tile N and slot layout (``tile_rows`` routes in every tile but
    the last, one slot of ``slot_rows`` pool rows each), at pool rows past the
    routed worst case: data rows from ``data_base`` (FC1 output, routing
    records, top-k scores), scale-factor rows from ``sf_base`` and FC2 gate
    slots from ``block_base``. Its tiles carry expert index ``expert_marker``,
    the local expert count, which no routed tile holds.

    In every claim stream the shared FC1 tiles take the front of the work-ID
    space, token tile major and output block inner as the routed order, and
    the shared FC2 tiles follow them where ``fc2_early`` (where all of them fit
    one wave of clusters, unless placed otherwise); otherwise they follow the
    routed tiles. An FC2 tile thus reads only FC1 tiles of lower work IDs.

    With ``entry_head`` the shared FC1 tiles, and with ``fc2_early`` the shared
    FC2 tiles behind them, leave the claim streams for a head of their own
    work-ID counter, which clusters claim from kernel entry, before the plan
    exists; its tiles span every output block of a token tile whatever the
    weights' die placement. The claim streams then start with the routed
    tiles, and the shared FC2 tiles not in the head follow them as before.

    ``head_token_tiles`` (None: all) caps the entry head at the FC1 tiles of
    that many leading token tiles. The FC1 tiles of the others take every
    ``q + 1``-th place of each claim stream, ``q`` the stream's routed tiles
    per such shared tile, until they run out (``streamed_token_tiles``); the
    routed tiles keep their order around them and the shared FC2 tiles follow
    them all, so an FC2 tile still reads only FC1 tiles of lower work IDs.
    """

    expert_marker: int
    tokens: int
    tile_rows: int
    token_tiles: int
    slot_rows: int
    data_base: int
    sf_base: int
    block_base: int
    fc2_early: bool
    entry_head: bool = False
    head_token_tiles: Optional[int] = None

    @property
    def streamed_token_tiles(self) -> int:
        """Token tiles whose FC1 tiles join the claim streams instead of the entry head."""
        if not self.entry_head or self.head_token_tiles is None:
            return 0
        return self.token_tiles - self.head_token_tiles

    def valid_tokens(self, token_tile: int) -> int:
        return min(self.tile_rows, self.tokens - token_tile * self.tile_rows)

    def slot_row(self, token: int) -> int:
        """Pool row of a token, relative to the bases."""
        return token // self.tile_rows * self.slot_rows + token % self.tile_rows


@cute.jit
def map_shared_expert_work_id(
    shared_id: Int32,
    phase: int,
    cta_id_in_mapping_cluster: cute.Coord,
    mapping_state: Fc12TaskMappingState,
    shared: SharedExpertTiles,
    blocks_per_token_block: Optional[int] = None,
) -> SwapAbFc12WorkTileInfo:
    """The swap-AB work tile of the ``shared_id``-th shared tile of one phase: token tile major, output blocks inner.

    ``blocks_per_token_block`` (None: the claim stream's own, from
    ``mapping_state``) is the phase's output blocks per token tile.
    """
    if cutlass.const_expr(blocks_per_token_block is None):
        blocks_per_token_block = mapping_state.num_fc1_intermediate_blocks
        if cutlass.const_expr(phase == BlockPhase.Linear2):
            blocks_per_token_block = mapping_state.num_fc2_hidden_blocks
    cluster_token_block_idx = shared_id // Int32(blocks_per_token_block)
    cluster_output_block_idx = shared_id - cluster_token_block_idx * Int32(
        blocks_per_token_block
    )
    cta_token_block_idx = (
        cluster_token_block_idx * Int32(mapping_state.mapping_cluster_shape_mn[0])
        + cta_id_in_mapping_cluster[0]
    )
    cta_output_block_idx = (
        cluster_output_block_idx * Int32(mapping_state.mapping_cluster_shape_mn[1])
        + cta_id_in_mapping_cluster[1]
    )
    valid_tokens_in_cta_tile = cutlass.min(
        Int32(shared.tile_rows),
        Int32(shared.tokens) - cta_token_block_idx * Int32(shared.tile_rows),
    )
    return SwapAbFc12WorkTileInfo(
        expert_idx=Int32(shared.expert_marker),
        tile_m_idx=cta_output_block_idx,
        tile_n_idx=cta_token_block_idx,
        cumulative_data_physical_row=Int32(shared.data_base),
        cumulative_sf_physical_row=Int32(shared.sf_base),
        cumulative_token_block_count=Int32(shared.block_base),
        valid_tokens_in_cta_tile=valid_tokens_in_cta_tile,
        phase_and_flags=Int32(phase),
    )


@cute.jit
def _cached_token_count(
    mapping_state: Fc12TaskMappingState, batch_base: Int32
) -> Int32:
    """This lane's count in the batch of 32 experts that starts at ``batch_base``."""
    token_count = mapping_state.token_count_cache[0]
    for batch in cutlass.range_constexpr(1, len(mapping_state.token_count_cache)):
        if batch_base == Int32(batch * 32):
            token_count = mapping_state.token_count_cache[batch]
    return token_count


@cute.jit
def _warp_inclusive_sum(value: Int32, lane_idx: Int32) -> Int32:
    inclusive = value
    for step_log in cutlass.range_constexpr(5):
        step = Int32(1 << step_log)
        previous = Int32(cute.arch.shuffle_sync(inclusive, lane_idx - step))
        if lane_idx >= step:
            inclusive = inclusive + previous
    return inclusive


@cute.jit
def _first_matching_lane(predicate) -> Int32:
    mask = Int32(cute.arch.vote_ballot_sync(predicate))
    first_lane = Int32(-1)
    if mask != Int32(0):
        lowbit = mask & (-mask)
        first_lane = Int32(cute.arch.popc(lowbit - Int32(1)))
    return first_lane


@cute.jit
def _load_expert_batch_metrics(
    mapping_state: Fc12TaskMappingState,
    batch_base: Int32,
    active_begin: Int32,
    active_end: Int32,
    lane_idx: Int32,
) -> Tuple[Int32, Int32, Int32, Int32, Int32, Int32, Int32]:
    expert_idx = batch_base + lane_idx
    token_count = Int32(0)
    if cutlass.const_expr(mapping_state.token_count_cache is not None):
        token_count = _cached_token_count(mapping_state, batch_base)
    elif cutlass.const_expr(mapping_state.expert_token_sizes is not None):
        if expert_idx < mapping_state.expert_count:
            token_count = mapping_state.expert_token_sizes[expert_idx]
    else:
        prefix_end = Int32(0)
        if expert_idx < mapping_state.expert_count:
            prefix_end = mapping_state.expert_token_prefix_sum[expert_idx]
        prefix_begin = Int32(cute.arch.shuffle_sync(prefix_end, lane_idx - Int32(1)))
        if lane_idx == Int32(0):
            prefix_begin = Int32(0)
            if batch_base > Int32(0):
                prefix_begin = mapping_state.expert_token_prefix_sum[
                    batch_base - Int32(1)
                ]
        token_count = prefix_end - prefix_begin
    if (expert_idx < active_begin) | (expert_idx >= active_end):
        token_count = Int32(0)
    token_blocks = (
        token_count + Int32(mapping_state.mapping_cluster_tile_m - 1)
    ) // Int32(mapping_state.mapping_cluster_tile_m)
    data_rows = _expert_pool_rows(
        mapping_state, token_count, mapping_state.token_padding_block
    )
    sf_rows = _expert_pool_rows(
        mapping_state, token_count, mapping_state.sf_padding_block
    )
    fc1_tiles = token_blocks * mapping_state.num_fc1_intermediate_blocks
    fc2_tiles = token_blocks * mapping_state.num_fc2_hidden_blocks
    return (
        expert_idx,
        token_count,
        token_blocks,
        data_rows,
        sf_rows,
        fc1_tiles,
        fc2_tiles,
    )


def make_fc12_done_tile(is_swap_ab: bool) -> SchedulerWorkTileBase:
    """Build the terminal work tile for either FC12 orientation."""
    if cutlass.const_expr(is_swap_ab):
        return SwapAbFc12WorkTileInfo(
            expert_idx=Int32(Fc12WorkTileState.Done),
            tile_m_idx=Int32(0),
            tile_n_idx=Int32(0),
            cumulative_data_physical_row=Int32(0),
            cumulative_sf_physical_row=Int32(0),
            cumulative_token_block_count=Int32(0),
            valid_tokens_in_cta_tile=Int32(0),
            phase_and_flags=Int32(BlockPhase.None_),
        )
    return NonSwapAbFc12WorkTileInfo(
        expert_idx=Int32(Fc12WorkTileState.Done),
        tile_m_idx=Int32(0),
        tile_n_idx=Int32(0),
        cumulative_data_physical_row=Int32(0),
        cumulative_sf_physical_row=Int32(0),
        cumulative_token_block_count=Int32(0),
        valid_tokens_in_cta_cluster_tile=Int32(0),
        phase_and_flags=Int32(BlockPhase.None_),
    )


@cute.jit
def _switch_to_fc2(mapping_state: Fc12TaskMappingState) -> _Fc12TaskCursorState:
    cursor = mapping_state.cursor_state
    cursor.current_phase = Int32(BlockPhase.Linear2)
    cursor.current_expert_idx = cursor.current_group_first_expert - Int32(1)
    cursor.current_expert_tile_start = cursor.current_group_fc1_subphase_end
    cursor.current_expert_tile_end = cursor.current_group_fc1_subphase_end
    cursor.current_expert_token_count = Int32(0)
    cursor.current_token_block_count = Int32(0)
    cursor.current_data_cumulative = cursor.group_start_data_cumulative
    cursor.current_sf_cumulative = cursor.group_start_sf_cumulative
    cursor.current_token_block_cumulative = cursor.group_start_token_block_cumulative
    mapping_state.cursor_state = cursor
    return cursor


@cute.jit
def _sum_expert_range(
    mapping_state: Fc12TaskMappingState, expert_begin: Int32, expert_end: Int32
) -> Tuple[Int32, Int32, Int32]:
    lane_idx = Int32(cute.arch.lane_idx())
    data_rows = Int32(0)
    sf_rows = Int32(0)
    token_blocks = Int32(0)
    batch_base = (expert_begin // Int32(32)) * Int32(32)
    while batch_base < expert_end:
        (_, _, lane_token_blocks, lane_data_rows, lane_sf_rows, _, _) = (
            _load_expert_batch_metrics(
                mapping_state, batch_base, expert_begin, expert_end, lane_idx
            )
        )
        data_rows = data_rows + Int32(cute.arch.warp_redux_sync(lane_data_rows, "add"))
        sf_rows = sf_rows + Int32(cute.arch.warp_redux_sync(lane_sf_rows, "add"))
        token_blocks = token_blocks + Int32(
            cute.arch.warp_redux_sync(lane_token_blocks, "add")
        )
        batch_base = batch_base + Int32(32)
    return data_rows, sf_rows, token_blocks


@cute.jit
def _build_group_range(
    mapping_state: Fc12TaskMappingState,
    group_first_expert: Int32,
    base_fc1_tiles: Int32,
    base_fc2_tiles: Int32,
) -> Tuple[Int32, Int32, Int32]:
    lane_idx = Int32(cute.arch.lane_idx())
    group_threshold = base_fc1_tiles + Int32(mapping_state.group_hint)
    group_last_expert = group_first_expert
    cumulative_fc1_tiles = base_fc1_tiles
    cumulative_fc2_tiles = base_fc2_tiles
    batch_base = (group_first_expert // Int32(32)) * Int32(32)

    while (
        batch_base < mapping_state.expert_count
        and cumulative_fc1_tiles < group_threshold
    ):
        (lane_expert_idx, _, _, _, _, lane_fc1_tiles, lane_fc2_tiles) = (
            _load_expert_batch_metrics(
                mapping_state,
                batch_base,
                group_first_expert,
                Int32(mapping_state.expert_count),
                lane_idx,
            )
        )
        fc1_prefix = _warp_inclusive_sum(lane_fc1_tiles, lane_idx)
        reaches_threshold = (
            (lane_expert_idx >= group_first_expert)
            & (lane_expert_idx < mapping_state.expert_count)
        ) & (cumulative_fc1_tiles + fc1_prefix >= group_threshold)
        selected_lane = _first_matching_lane(reaches_threshold)
        included_fc2_tiles = lane_fc2_tiles
        if selected_lane >= Int32(0):
            if lane_idx > selected_lane:
                included_fc2_tiles = Int32(0)
        fc2_batch_tiles = Int32(cute.arch.warp_redux_sync(included_fc2_tiles, "add"))
        if selected_lane >= Int32(0):
            cumulative_fc1_tiles = cumulative_fc1_tiles + Int32(
                cute.arch.shuffle_sync(fc1_prefix, selected_lane)
            )
            group_last_expert = batch_base + selected_lane + Int32(1)
        else:
            cumulative_fc1_tiles = cumulative_fc1_tiles + Int32(
                cute.arch.shuffle_sync(fc1_prefix, Int32(31))
            )
            batch_base = batch_base + Int32(32)
            group_last_expert = cutlass.min(
                batch_base, Int32(mapping_state.expert_count)
            )
        cumulative_fc2_tiles = cumulative_fc2_tiles + fc2_batch_tiles

    return group_last_expert, cumulative_fc1_tiles, cumulative_fc2_tiles


@cute.jit
def _advance_group(mapping_state: Fc12TaskMappingState) -> _Fc12TaskCursorState:
    cursor = mapping_state.cursor_state

    residual_begin = cutlass.max(
        cursor.current_expert_idx, cursor.current_group_first_expert
    )
    residual_data_rows = Int32(0)
    residual_sf_rows = Int32(0)
    residual_token_blocks = Int32(0)
    if residual_begin < cursor.current_group_last_expert_exclusive:
        iket.range_push("scheduler.residual_scan")
        residual_data_rows, residual_sf_rows, residual_token_blocks = _sum_expert_range(
            mapping_state, residual_begin, cursor.current_group_last_expert_exclusive
        )
        iket.range_pop()
    cursor.current_data_cumulative = cursor.current_data_cumulative + residual_data_rows
    cursor.current_sf_cumulative = cursor.current_sf_cumulative + residual_sf_rows
    cursor.current_token_block_cumulative = (
        cursor.current_token_block_cumulative + residual_token_blocks
    )

    cursor.group_start_data_cumulative = cursor.current_data_cumulative
    cursor.group_start_sf_cumulative = cursor.current_sf_cumulative
    cursor.group_start_token_block_cumulative = cursor.current_token_block_cumulative

    base_fc1_tiles = cursor.cumulative_fc1_tiles_at_group_end
    base_fc2_tiles = cursor.cumulative_fc2_tiles_at_group_end
    cursor.current_group_first_expert = cursor.current_group_last_expert_exclusive

    iket.range_push("scheduler.group_scan")
    (
        cursor.current_group_last_expert_exclusive,
        cumulative_fc1_tiles,
        cumulative_fc2_tiles,
    ) = _build_group_range(
        mapping_state, cursor.current_group_first_expert, base_fc1_tiles, base_fc2_tiles
    )
    iket.range_pop()
    cursor.cumulative_fc1_tiles_at_group_end = cumulative_fc1_tiles
    cursor.cumulative_fc2_tiles_at_group_end = cumulative_fc2_tiles
    group_start_tile = cursor.current_group_end
    cursor.current_group_fc1_subphase_end = (
        group_start_tile + cumulative_fc1_tiles - base_fc1_tiles
    )
    cursor.current_group_end = (
        cursor.current_group_fc1_subphase_end + cumulative_fc2_tiles - base_fc2_tiles
    )

    cursor.current_phase = Int32(BlockPhase.Linear1)
    cursor.current_expert_idx = cursor.current_group_first_expert - Int32(1)
    cursor.current_expert_tile_start = group_start_tile
    cursor.current_expert_tile_end = group_start_tile
    cursor.current_expert_token_count = Int32(0)
    cursor.current_token_block_count = Int32(0)
    mapping_state.cursor_state = cursor
    return cursor


@cute.jit
def _seek_expert_for_work_id(
    linear_work_id: Int32, mapping_state: Fc12TaskMappingState
) -> _Fc12TaskCursorState:
    cursor = mapping_state.cursor_state

    base_tile_end = cursor.current_expert_tile_end
    base_data_cumulative = cursor.current_data_cumulative
    base_sf_cumulative = cursor.current_sf_cumulative
    base_token_block_cumulative = cursor.current_token_block_cumulative
    if cursor.current_expert_idx >= cursor.current_group_first_expert:
        current_token_count = cursor.current_expert_token_count
        base_data_cumulative = base_data_cumulative + _expert_pool_rows(
            mapping_state, current_token_count, mapping_state.token_padding_block
        )
        base_sf_cumulative = base_sf_cumulative + _expert_pool_rows(
            mapping_state, current_token_count, mapping_state.sf_padding_block
        )
        base_token_block_cumulative = (
            base_token_block_cumulative + cursor.current_token_block_count
        )

    search_begin = cutlass.max(
        cursor.current_expert_idx + Int32(1), cursor.current_group_first_expert
    )
    batch_base = (search_begin // Int32(32)) * Int32(32)
    selected_expert = Int32(-1)
    selected_token_count = Int32(0)
    selected_token_blocks = Int32(0)
    selected_tile_start = Int32(0)
    selected_tile_end = Int32(0)
    selected_data_cumulative = Int32(0)
    selected_sf_cumulative = Int32(0)
    selected_token_block_cumulative = Int32(0)
    lane_idx = Int32(cute.arch.lane_idx())

    iket.range_push("scheduler.expert_scan")
    while (
        selected_expert < Int32(0)
        and batch_base < cursor.current_group_last_expert_exclusive
    ):
        (
            lane_expert_idx,
            lane_token_count,
            lane_token_blocks,
            lane_data_rows,
            lane_sf_rows,
            lane_fc1_tiles,
            lane_fc2_tiles,
        ) = _load_expert_batch_metrics(
            mapping_state,
            batch_base,
            search_begin,
            cursor.current_group_last_expert_exclusive,
            lane_idx,
        )
        lane_phase_tiles = lane_fc1_tiles
        if cursor.current_phase == Int32(BlockPhase.Linear2):
            lane_phase_tiles = lane_fc2_tiles
        tile_prefix = _warp_inclusive_sum(lane_phase_tiles, lane_idx)
        candidate_tile_end = base_tile_end + tile_prefix
        contains_work = (
            (lane_expert_idx >= search_begin)
            & (lane_expert_idx < cursor.current_group_last_expert_exclusive)
            & (linear_work_id < candidate_tile_end)
        )
        selected_lane = _first_matching_lane(contains_work)

        included_data_rows = lane_data_rows
        included_sf_rows = lane_sf_rows
        included_token_blocks = lane_token_blocks
        if selected_lane >= Int32(0):
            if lane_idx > selected_lane:
                included_data_rows = Int32(0)
                included_sf_rows = Int32(0)
                included_token_blocks = Int32(0)
        batch_data_rows = Int32(cute.arch.warp_redux_sync(included_data_rows, "add"))
        batch_sf_rows = Int32(cute.arch.warp_redux_sync(included_sf_rows, "add"))
        batch_token_blocks = Int32(
            cute.arch.warp_redux_sync(included_token_blocks, "add")
        )
        if selected_lane >= Int32(0):
            selected_expert = batch_base + selected_lane
            selected_token_count = Int32(
                cute.arch.shuffle_sync(lane_token_count, selected_lane)
            )
            selected_token_blocks = Int32(
                cute.arch.shuffle_sync(lane_token_blocks, selected_lane)
            )
            selected_phase_tiles = Int32(
                cute.arch.shuffle_sync(lane_phase_tiles, selected_lane)
            )
            selected_tile_end = base_tile_end + Int32(
                cute.arch.shuffle_sync(tile_prefix, selected_lane)
            )
            selected_tile_start = selected_tile_end - selected_phase_tiles
            selected_data_rows = Int32(
                cute.arch.shuffle_sync(lane_data_rows, selected_lane)
            )
            selected_sf_rows = Int32(
                cute.arch.shuffle_sync(lane_sf_rows, selected_lane)
            )
            selected_data_cumulative = (
                base_data_cumulative + batch_data_rows - selected_data_rows
            )
            selected_sf_cumulative = (
                base_sf_cumulative + batch_sf_rows - selected_sf_rows
            )
            selected_token_block_cumulative = (
                base_token_block_cumulative + batch_token_blocks - selected_token_blocks
            )
        else:
            base_tile_end = base_tile_end + Int32(
                cute.arch.shuffle_sync(tile_prefix, Int32(31))
            )
            base_data_cumulative = base_data_cumulative + batch_data_rows
            base_sf_cumulative = base_sf_cumulative + batch_sf_rows
            base_token_block_cumulative = (
                base_token_block_cumulative + batch_token_blocks
            )
            batch_base = batch_base + Int32(32)
            search_begin = batch_base
    iket.range_pop()

    cursor.current_expert_idx = selected_expert
    cursor.current_expert_token_count = selected_token_count
    cursor.current_token_block_count = selected_token_blocks
    cursor.current_expert_tile_start = selected_tile_start
    cursor.current_expert_tile_end = selected_tile_end
    cursor.current_data_cumulative = selected_data_cumulative
    cursor.current_sf_cumulative = selected_sf_cumulative
    cursor.current_token_block_cumulative = selected_token_block_cumulative
    return cursor


@cute.jit
def _decode_inside_expert(
    linear_work_id: Int32,
    cta_id_in_mapping_cluster: cute.Coord,
    mapping_state: Fc12TaskMappingState,
) -> SchedulerWorkTileBase:
    cursor = mapping_state.cursor_state
    cta_tile_m = mapping_state.mapping_cta_tile_shape_mnk[0]
    local_work_id = linear_work_id - cursor.current_expert_tile_start

    cluster_token_block_idx = Int32(0)
    cluster_output_block_idx = Int32(0)
    if cursor.current_phase == Int32(BlockPhase.Linear1):
        cluster_token_block_idx = (
            local_work_id // mapping_state.num_fc1_intermediate_blocks
        )
        cluster_output_block_idx = (
            local_work_id
            - cluster_token_block_idx * mapping_state.num_fc1_intermediate_blocks
        )
    else:
        cluster_token_block_idx = local_work_id // mapping_state.num_fc2_hidden_blocks
        cluster_output_block_idx = (
            local_work_id
            - cluster_token_block_idx * mapping_state.num_fc2_hidden_blocks
        )

    cta_token_block_idx = (
        cluster_token_block_idx * mapping_state.mapping_cluster_shape_mn[0]
        + cta_id_in_mapping_cluster[0]
    )
    cta_output_block_idx = (
        cluster_output_block_idx * mapping_state.mapping_cluster_shape_mn[1]
        + cta_id_in_mapping_cluster[1]
    )
    valid_tokens_in_cta_tile = _valid_tokens_in_token_tile(
        cursor.current_expert_token_count, cta_token_block_idx, cta_tile_m
    )

    if cutlass.const_expr(mapping_state.is_swap_ab):
        return SwapAbFc12WorkTileInfo(
            expert_idx=cursor.current_expert_idx,
            tile_m_idx=cta_output_block_idx,
            tile_n_idx=cta_token_block_idx,
            cumulative_data_physical_row=cursor.current_data_cumulative,
            cumulative_sf_physical_row=cursor.current_sf_cumulative,
            cumulative_token_block_count=(cursor.current_token_block_cumulative),
            valid_tokens_in_cta_tile=valid_tokens_in_cta_tile,
            phase_and_flags=cursor.current_phase,
        )

    cluster_tile_m = mapping_state.mapping_cluster_shape_mn[0] * cta_tile_m
    cluster_token_start = cluster_token_block_idx * Int32(cluster_tile_m)
    remaining_cluster_tokens = cutlass.max(
        cursor.current_expert_token_count - cluster_token_start, Int32(0)
    )
    valid_tokens_in_cluster_tile = cutlass.min(
        remaining_cluster_tokens, Int32(cluster_tile_m)
    )
    valid_tokens_in_cta_cluster_tile = (
        valid_tokens_in_cta_tile << Int32(16)
    ) | valid_tokens_in_cluster_tile
    return NonSwapAbFc12WorkTileInfo(
        expert_idx=cursor.current_expert_idx,
        tile_m_idx=cta_token_block_idx,
        tile_n_idx=cta_output_block_idx,
        cumulative_data_physical_row=cursor.current_data_cumulative,
        cumulative_sf_physical_row=cursor.current_sf_cumulative,
        cumulative_token_block_count=(cursor.current_token_block_cumulative),
        valid_tokens_in_cta_cluster_tile=(valid_tokens_in_cta_cluster_tile),
        phase_and_flags=cursor.current_phase,
    )


@cute.jit
def map_fc12_linear_work_id(
    linear_work_id: Int32,
    cta_id_in_mapping_cluster: cute.Coord,
    mapping_state: Fc12TaskMappingState,
) -> Tuple[SchedulerWorkTileBase, Fc12TaskMappingState]:
    """Map one monotonically increasing scalar ID to an FC12 work tile."""
    cursor = mapping_state.cursor_state
    work_tile = make_fc12_done_tile(mapping_state.is_swap_ab)

    outer_group_end = cursor.current_group_end
    outer_expert_end = cursor.current_group_last_expert_exclusive
    while (
        linear_work_id >= outer_group_end
        and outer_expert_end < mapping_state.expert_count
    ):
        mapping_state.cursor_state = _advance_group(mapping_state)
        cursor = mapping_state.cursor_state
        outer_group_end = cursor.current_group_end
        outer_expert_end = cursor.current_group_last_expert_exclusive
    cursor = mapping_state.cursor_state

    if linear_work_id < cursor.current_group_end:
        if (
            cursor.current_phase == Int32(BlockPhase.Linear1)
            and linear_work_id >= cursor.current_group_fc1_subphase_end
        ):
            mapping_state.cursor_state = _switch_to_fc2(mapping_state)
        else:
            mapping_state.cursor_state = mapping_state.cursor_state
        cursor = mapping_state.cursor_state

        if linear_work_id >= cursor.current_expert_tile_end:
            mapping_state.cursor_state = _seek_expert_for_work_id(
                linear_work_id, mapping_state
            )
        else:
            mapping_state.cursor_state = mapping_state.cursor_state
        cursor = mapping_state.cursor_state
        work_tile = _decode_inside_expert(
            linear_work_id, cta_id_in_mapping_cluster, mapping_state
        )
    else:
        mapping_state.cursor_state = mapping_state.cursor_state
    return work_tile, mapping_state


@cute.jit
def _advance_pipeline_segment(
    mapping_state: Fc12TaskMappingState,
) -> _Fc12PipelineCursorState:
    """Move to the next segment: the next schedule group's FC1 tiles and the current group's FC2 tiles."""
    cursor = mapping_state.cursor_state
    cursor.fc2_lag_begin = cursor.fc2_lag_end
    cursor.fc2_lag_end = cursor.fc2_group_end
    cursor.fc1_group_begin = cursor.fc1_group_end
    cursor.group_first_expert = cursor.group_last_expert_exclusive
    iket.range_push("scheduler.group_scan")
    (cursor.group_last_expert_exclusive, cursor.fc1_group_end, cursor.fc2_group_end) = (
        _build_group_range(
            mapping_state,
            cursor.group_first_expert,
            cursor.fc1_group_begin,
            cursor.fc2_lag_end,
        )
    )
    iket.range_pop()
    cursor.segment_start = cursor.segment_end
    cursor.segment_end = (
        cursor.segment_start
        + (cursor.fc1_group_end - cursor.fc1_group_begin)
        + (cursor.fc2_lag_end - cursor.fc2_lag_begin)
    )
    mapping_state.cursor_state = cursor
    return cursor


@cute.jit
def _seek_pipeline_phase_cursor(
    phase_tile: Int32,
    cursor: "_PhaseFc12CursorState",
    use_fc2_tiles: bool,
    mapping_state: Fc12TaskMappingState,
) -> "_PhaseFc12CursorState":
    """Advance one phase's expert cursor to the expert that holds ``phase_tile``.

    ``phase_tile`` counts that phase's tiles over every expert in processing order
    and must lie past the cursor's current expert. The scan reads 32 experts per
    step and carries the pool-row and token-block bases along.
    """
    base_tile_end = cursor.expert_tile_end
    base_data_cumulative = cursor.data_cumulative
    base_sf_cumulative = cursor.sf_cumulative
    base_token_block_cumulative = cursor.token_block_cumulative
    if cursor.expert_idx >= Int32(0):
        current_token_count = cursor.current_expert_token_count
        base_data_cumulative = base_data_cumulative + _expert_pool_rows(
            mapping_state, current_token_count, mapping_state.token_padding_block
        )
        base_sf_cumulative = base_sf_cumulative + _expert_pool_rows(
            mapping_state, current_token_count, mapping_state.sf_padding_block
        )
        base_token_block_cumulative = (
            base_token_block_cumulative + cursor.current_token_block_count
        )

    expert_end = Int32(mapping_state.expert_count)
    search_begin = cursor.expert_idx + Int32(1)
    batch_base = (search_begin // Int32(32)) * Int32(32)
    selected_expert = Int32(-1)
    selected_token_count = Int32(0)
    selected_token_blocks = Int32(0)
    selected_tile_start = Int32(0)
    selected_tile_end = Int32(0)
    selected_data_cumulative = Int32(0)
    selected_sf_cumulative = Int32(0)
    selected_token_block_cumulative = Int32(0)
    lane_idx = Int32(cute.arch.lane_idx())

    iket.range_push("scheduler.expert_scan")
    while selected_expert < Int32(0) and batch_base < expert_end:
        (
            lane_expert_idx,
            lane_token_count,
            lane_token_blocks,
            lane_data_rows,
            lane_sf_rows,
            lane_fc1_tiles,
            lane_fc2_tiles,
        ) = _load_expert_batch_metrics(
            mapping_state, batch_base, search_begin, expert_end, lane_idx
        )
        lane_phase_tiles = lane_fc1_tiles
        if cutlass.const_expr(use_fc2_tiles):
            lane_phase_tiles = lane_fc2_tiles
        tile_prefix = _warp_inclusive_sum(lane_phase_tiles, lane_idx)
        contains_work = (
            (lane_expert_idx >= search_begin)
            & (lane_expert_idx < expert_end)
            & (phase_tile < base_tile_end + tile_prefix)
        )
        selected_lane = _first_matching_lane(contains_work)

        included_data_rows = lane_data_rows
        included_sf_rows = lane_sf_rows
        included_token_blocks = lane_token_blocks
        if selected_lane >= Int32(0):
            if lane_idx > selected_lane:
                included_data_rows = Int32(0)
                included_sf_rows = Int32(0)
                included_token_blocks = Int32(0)
        batch_data_rows = Int32(cute.arch.warp_redux_sync(included_data_rows, "add"))
        batch_sf_rows = Int32(cute.arch.warp_redux_sync(included_sf_rows, "add"))
        batch_token_blocks = Int32(
            cute.arch.warp_redux_sync(included_token_blocks, "add")
        )
        if selected_lane >= Int32(0):
            selected_expert = batch_base + selected_lane
            selected_token_count = Int32(
                cute.arch.shuffle_sync(lane_token_count, selected_lane)
            )
            selected_token_blocks = Int32(
                cute.arch.shuffle_sync(lane_token_blocks, selected_lane)
            )
            selected_phase_tiles = Int32(
                cute.arch.shuffle_sync(lane_phase_tiles, selected_lane)
            )
            selected_tile_end = base_tile_end + Int32(
                cute.arch.shuffle_sync(tile_prefix, selected_lane)
            )
            selected_tile_start = selected_tile_end - selected_phase_tiles
            selected_data_rows = Int32(
                cute.arch.shuffle_sync(lane_data_rows, selected_lane)
            )
            selected_sf_rows = Int32(
                cute.arch.shuffle_sync(lane_sf_rows, selected_lane)
            )
            selected_data_cumulative = (
                base_data_cumulative + batch_data_rows - selected_data_rows
            )
            selected_sf_cumulative = (
                base_sf_cumulative + batch_sf_rows - selected_sf_rows
            )
            selected_token_block_cumulative = (
                base_token_block_cumulative + batch_token_blocks - selected_token_blocks
            )
        else:
            base_tile_end = base_tile_end + Int32(
                cute.arch.shuffle_sync(tile_prefix, Int32(31))
            )
            base_data_cumulative = base_data_cumulative + batch_data_rows
            base_sf_cumulative = base_sf_cumulative + batch_sf_rows
            base_token_block_cumulative = (
                base_token_block_cumulative + batch_token_blocks
            )
            batch_base = batch_base + Int32(32)
            search_begin = batch_base
    iket.range_pop()

    cursor.expert_idx = selected_expert
    cursor.current_expert_token_count = selected_token_count
    cursor.current_token_block_count = selected_token_blocks
    cursor.expert_tile_start = selected_tile_start
    cursor.expert_tile_end = selected_tile_end
    cursor.data_cumulative = selected_data_cumulative
    cursor.sf_cumulative = selected_sf_cumulative
    cursor.token_block_cumulative = selected_token_block_cumulative
    return cursor


@cute.jit
def _decode_pipeline_phase_tile(
    phase_tile: Int32,
    phase: int,
    cta_id_in_mapping_cluster: cute.Coord,
    cursor: "_PhaseFc12CursorState",
    mapping_state: Fc12TaskMappingState,
) -> SchedulerWorkTileBase:
    """The work tile of ``phase_tile`` inside the cursor's expert: token-block major, output blocks inner."""
    cta_tile_m = mapping_state.mapping_cta_tile_shape_mnk[0]
    local_work_id = phase_tile - cursor.expert_tile_start
    blocks_per_token_block = Int32(cursor.blocks_per_token_block)
    cluster_token_block_idx = local_work_id // blocks_per_token_block
    cluster_output_block_idx = (
        local_work_id - cluster_token_block_idx * blocks_per_token_block
    )

    cta_token_block_idx = (
        cluster_token_block_idx * mapping_state.mapping_cluster_shape_mn[0]
        + cta_id_in_mapping_cluster[0]
    )
    cta_output_block_idx = (
        cluster_output_block_idx * mapping_state.mapping_cluster_shape_mn[1]
        + cta_id_in_mapping_cluster[1]
    )
    valid_tokens_in_cta_tile = _valid_tokens_in_token_tile(
        cursor.current_expert_token_count, cta_token_block_idx, cta_tile_m
    )

    if cutlass.const_expr(mapping_state.is_swap_ab):
        return SwapAbFc12WorkTileInfo(
            expert_idx=cursor.expert_idx,
            tile_m_idx=cta_output_block_idx,
            tile_n_idx=cta_token_block_idx,
            cumulative_data_physical_row=cursor.data_cumulative,
            cumulative_sf_physical_row=cursor.sf_cumulative,
            cumulative_token_block_count=cursor.token_block_cumulative,
            valid_tokens_in_cta_tile=valid_tokens_in_cta_tile,
            phase_and_flags=Int32(phase),
        )

    cluster_tile_m = mapping_state.mapping_cluster_shape_mn[0] * cta_tile_m
    cluster_token_start = cluster_token_block_idx * Int32(cluster_tile_m)
    remaining_cluster_tokens = cutlass.max(
        cursor.current_expert_token_count - cluster_token_start, Int32(0)
    )
    valid_tokens_in_cluster_tile = cutlass.min(
        remaining_cluster_tokens, Int32(cluster_tile_m)
    )
    return NonSwapAbFc12WorkTileInfo(
        expert_idx=cursor.expert_idx,
        tile_m_idx=cta_token_block_idx,
        tile_n_idx=cta_output_block_idx,
        cumulative_data_physical_row=cursor.data_cumulative,
        cumulative_sf_physical_row=cursor.sf_cumulative,
        cumulative_token_block_count=cursor.token_block_cumulative,
        valid_tokens_in_cta_cluster_tile=(valid_tokens_in_cta_tile << Int32(16))
        | valid_tokens_in_cluster_tile,
        phase_and_flags=Int32(phase),
    )


@cute.jit
def map_fc12_pipelined_work_id(
    linear_work_id: Int32,
    cta_id_in_mapping_cluster: cute.Coord,
    mapping_state: Fc12TaskMappingState,
) -> Tuple[SchedulerWorkTileBase, Fc12TaskMappingState]:
    """Map one monotonically increasing scalar ID to an FC12 work tile in the pipelined order.

    Schedule groups are built as in ``map_fc12_linear_work_id`` (whole experts, about
    ``group_hint`` FC1 tiles each), but a group's FC2 tiles are claimed together with
    the next group's FC1 tiles rather than right after its own: segment ``k`` of the
    order mixes the FC1 tiles of group ``k`` with the FC2 tiles of group ``k - 1``,
    the FC2 tiles spaced evenly among the segment's positions. The combine pushes
    of the FC2 tiles thus spread over the FC1 work instead of leaving in one burst
    per group, and an FC2 tile's FC1 tiles were all claimed at least one segment
    earlier. Every FC2 tile still follows every FC1 tile it reads in the order, so
    clusters that process their claims in order cannot deadlock. Within a phase,
    tiles keep the grouped order: experts in processing order, token blocks major,
    output blocks inner. A single group yields the grouped order.
    """
    cursor = mapping_state.cursor_state
    work_tile = make_fc12_done_tile(mapping_state.is_swap_ab)

    segment_end = cursor.segment_end
    more_segments = (
        cursor.group_last_expert_exclusive < Int32(mapping_state.expert_count)
    ) | (cursor.fc2_group_end > cursor.fc2_lag_end)
    while linear_work_id >= segment_end and more_segments:
        mapping_state.cursor_state = _advance_pipeline_segment(mapping_state)
        cursor = mapping_state.cursor_state
        segment_end = cursor.segment_end
        more_segments = (
            cursor.group_last_expert_exclusive < Int32(mapping_state.expert_count)
        ) | (cursor.fc2_group_end > cursor.fc2_lag_end)
    cursor = mapping_state.cursor_state

    if linear_work_id < cursor.segment_end:
        position = linear_work_id - cursor.segment_start
        fc2_tiles = cursor.fc2_lag_end - cursor.fc2_lag_begin
        segment_tiles = fc2_tiles + cursor.fc1_group_end - cursor.fc1_group_begin
        fc2_before = position * fc2_tiles // segment_tiles
        fc2_through = (position + Int32(1)) * fc2_tiles // segment_tiles
        fc1_cursor = cursor.fc1_cursor
        fc2_cursor = cursor.fc2_cursor
        if fc2_through > fc2_before:
            phase_tile = cursor.fc2_lag_begin + fc2_before
            if phase_tile >= fc2_cursor.expert_tile_end:
                fc2_cursor = _seek_pipeline_phase_cursor(
                    phase_tile, fc2_cursor, True, mapping_state
                )
            work_tile = _decode_pipeline_phase_tile(
                phase_tile,
                BlockPhase.Linear2,
                cta_id_in_mapping_cluster,
                fc2_cursor,
                mapping_state,
            )
        else:
            phase_tile = cursor.fc1_group_begin + position - fc2_before
            if phase_tile >= fc1_cursor.expert_tile_end:
                fc1_cursor = _seek_pipeline_phase_cursor(
                    phase_tile, fc1_cursor, False, mapping_state
                )
            work_tile = _decode_pipeline_phase_tile(
                phase_tile,
                BlockPhase.Linear1,
                cta_id_in_mapping_cluster,
                fc1_cursor,
                mapping_state,
            )
        cursor.fc1_cursor = fc1_cursor
        cursor.fc2_cursor = fc2_cursor
        mapping_state.cursor_state = cursor
    else:
        mapping_state.cursor_state = mapping_state.cursor_state
    return work_tile, mapping_state


class _PhaseFc12CursorState:
    """Monotonic expert cursor for one phase-local FC12 work-ID stream."""

    def __init__(
        self,
        expert_idx: Int32,
        expert_tile_start: Int32,
        expert_tile_end: Int32,
        current_expert_token_count: Int32,
        current_token_block_count: Int32,
        data_cumulative: Int32,
        sf_cumulative: Int32,
        token_block_cumulative: Int32,
        blocks_per_token_block: int,
    ) -> None:
        self.expert_idx = expert_idx
        self.expert_tile_start = expert_tile_start
        self.expert_tile_end = expert_tile_end
        self.current_expert_token_count = current_expert_token_count
        self.current_token_block_count = current_token_block_count
        self.data_cumulative = data_cumulative
        self.sf_cumulative = sf_cumulative
        self.token_block_cumulative = token_block_cumulative
        self.blocks_per_token_block = blocks_per_token_block

    def _runtime_fields(self) -> Tuple:
        return (
            self.expert_idx,
            self.expert_tile_start,
            self.expert_tile_end,
            self.current_expert_token_count,
            self.current_token_block_count,
            self.data_cumulative,
            self.sf_cumulative,
            self.token_block_cumulative,
        )

    @cute.jit
    def clone(self) -> "_PhaseFc12CursorState":
        """Fork this cursor while sharing its immutable initial SSA values."""
        return type(self)(
            *self._runtime_fields(), blocks_per_token_block=self.blocks_per_token_block
        )

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for field in self._runtime_fields():
            values.extend(extract_mlir_values(field))
        return values

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "_PhaseFc12CursorState":
        value_index = 0
        rebuilt_fields = []
        for field in self._runtime_fields():
            field_value_count = len(extract_mlir_values(field))
            rebuilt_fields.append(
                new_from_mlir_values(
                    field, values[value_index : value_index + field_value_count]
                )
            )
            value_index += field_value_count
        if value_index != len(values):
            raise ValueError(
                f"_PhaseFc12CursorState MLIR value count mismatch: consumed {value_index}, got {len(values)}."
            )
        return type(self)(
            *rebuilt_fields, blocks_per_token_block=self.blocks_per_token_block
        )


class PhaseInterleavedFc12MappingState:
    """Runtime inputs and independent FC1/FC2 cursors for phase-local IDs."""

    def __init__(
        self,
        expert_count: int,
        mapping_cta_tile_shape_mnk: Tuple[int, int, int],
        mapping_cluster_shape_mn: Tuple[int, int],
        token_padding_block: int,
        sf_padding_block: int,
        is_swap_ab: bool,
        expert_token_sizes: Optional[cute.Tensor],
        expert_token_prefix_sum: Optional[cute.Tensor],
        fc1_cursor: _PhaseFc12CursorState,
        fc2_cursor: _PhaseFc12CursorState,
        num_fc1_intermediate_blocks: int,
        num_fc2_hidden_blocks: int,
        dfc1_m_group: int = 1,
    ) -> None:
        if type(dfc1_m_group) is not int or dfc1_m_group not in (1, 16):
            raise ValueError("dfc1_m_group must be 1 or 16")
        self.dfc1_m_group = dfc1_m_group
        self.expert_count = expert_count
        self.mapping_cta_tile_shape_mnk = mapping_cta_tile_shape_mnk
        self.mapping_cluster_shape_mn = mapping_cluster_shape_mn
        self.token_padding_block = token_padding_block
        self.sf_padding_block = sf_padding_block
        self.is_swap_ab = is_swap_ab
        self.expert_token_sizes = expert_token_sizes
        self.expert_token_prefix_sum = expert_token_prefix_sum
        self.fc1_cursor = fc1_cursor
        self.fc2_cursor = fc2_cursor
        self.num_fc1_intermediate_blocks = num_fc1_intermediate_blocks
        self.num_fc2_hidden_blocks = num_fc2_hidden_blocks
        self.fc2_claims_per_token_block = num_fc2_hidden_blocks
        if fc2_cursor.blocks_per_token_block != self.fc2_claims_per_token_block:
            raise ValueError(
                "FC2 cursor blocks_per_token_block does not match the selected "
                "work-ID domain."
            )

    @property
    def mapping_cluster_tile_m(self) -> int:
        return self.mapping_cta_tile_shape_mnk[0] * self.mapping_cluster_shape_mn[0]

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        token_counts = (
            self.expert_token_sizes
            if self.expert_token_sizes is not None
            else self.expert_token_prefix_sum
        )
        values.extend(extract_mlir_values(token_counts))
        for field in (self.fc1_cursor, self.fc2_cursor):
            values.extend(extract_mlir_values(field))
        return values

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "PhaseInterleavedFc12MappingState":
        value_index = 0

        def rebuild(field):
            nonlocal value_index
            field_value_count = len(extract_mlir_values(field))
            result = new_from_mlir_values(
                field, values[value_index : value_index + field_value_count]
            )
            value_index += field_value_count
            return result

        if self.expert_token_sizes is not None:
            expert_token_sizes = rebuild(self.expert_token_sizes)
            expert_token_prefix_sum = None
        else:
            expert_token_sizes = None
            expert_token_prefix_sum = rebuild(self.expert_token_prefix_sum)
        result = type(self)(
            expert_count=self.expert_count,
            mapping_cta_tile_shape_mnk=self.mapping_cta_tile_shape_mnk,
            mapping_cluster_shape_mn=self.mapping_cluster_shape_mn,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            is_swap_ab=self.is_swap_ab,
            expert_token_sizes=expert_token_sizes,
            expert_token_prefix_sum=expert_token_prefix_sum,
            fc1_cursor=rebuild(self.fc1_cursor),
            fc2_cursor=rebuild(self.fc2_cursor),
            num_fc1_intermediate_blocks=self.num_fc1_intermediate_blocks,
            num_fc2_hidden_blocks=self.num_fc2_hidden_blocks,
            dfc1_m_group=self.dfc1_m_group,
        )
        if value_index != len(values):
            raise ValueError(
                f"PhaseInterleavedFc12MappingState MLIR value count mismatch: "
                f"consumed {value_index}, got {len(values)}."
            )
        return result


def _make_phase_cursor(blocks_per_token_block: int) -> _PhaseFc12CursorState:
    return _PhaseFc12CursorState(
        expert_idx=Int32(-1),
        expert_tile_start=Int32(0),
        expert_tile_end=Int32(0),
        current_expert_token_count=Int32(0),
        current_token_block_count=Int32(0),
        data_cumulative=Int32(0),
        sf_cumulative=Int32(0),
        token_block_cumulative=Int32(0),
        blocks_per_token_block=blocks_per_token_block,
    )


@cute.jit
def create_phase_interleaved_fc12_mapping_state(
    *,
    expert_count: int,
    intermediate_gateup_size: int,
    hidden_size: int,
    mapping_cta_tile_shape_mnk: Tuple[int, int, int],
    mapping_cluster_shape_mn: Tuple[int, int],
    token_padding_block: int,
    sf_padding_block: int,
    is_swap_ab: bool,
    expert_token_sizes: Optional[cute.Tensor],
    expert_token_prefix_sum: Optional[cute.Tensor],
    dfc1_m_group: int = 1,
) -> PhaseInterleavedFc12MappingState:
    """Create independent monotonic mapping cursors for the FC1 and FC2 streams."""
    mapping_cluster_tile_n = mapping_cluster_shape_mn[1] * mapping_cta_tile_shape_mnk[1]
    num_fc1_intermediate_blocks = (
        intermediate_gateup_size + mapping_cluster_tile_n - 1
    ) // mapping_cluster_tile_n
    num_fc2_hidden_blocks = (
        hidden_size + mapping_cluster_tile_n - 1
    ) // mapping_cluster_tile_n
    fc2_claims_per_token_block = num_fc2_hidden_blocks
    return PhaseInterleavedFc12MappingState(
        expert_count=expert_count,
        mapping_cta_tile_shape_mnk=mapping_cta_tile_shape_mnk,
        mapping_cluster_shape_mn=mapping_cluster_shape_mn,
        token_padding_block=token_padding_block,
        sf_padding_block=sf_padding_block,
        is_swap_ab=is_swap_ab,
        expert_token_sizes=expert_token_sizes,
        expert_token_prefix_sum=expert_token_prefix_sum,
        fc1_cursor=_make_phase_cursor(num_fc1_intermediate_blocks),
        fc2_cursor=_make_phase_cursor(fc2_claims_per_token_block),
        num_fc1_intermediate_blocks=num_fc1_intermediate_blocks,
        num_fc2_hidden_blocks=num_fc2_hidden_blocks,
        dfc1_m_group=dfc1_m_group,
    )


@cute.jit
def _advance_phase_cursor(
    cursor: _PhaseFc12CursorState, mapping_state: PhaseInterleavedFc12MappingState
) -> _PhaseFc12CursorState:
    previous_token_count = cursor.current_expert_token_count
    cursor.data_cumulative = cursor.data_cumulative + _expert_pool_rows(
        mapping_state, previous_token_count, mapping_state.token_padding_block
    )
    cursor.sf_cumulative = cursor.sf_cumulative + _expert_pool_rows(
        mapping_state, previous_token_count, mapping_state.sf_padding_block
    )
    cursor.token_block_cumulative = (
        cursor.token_block_cumulative + cursor.current_token_block_count
    )

    cursor.expert_idx = cursor.expert_idx + Int32(1)
    token_count = Int32(0)
    if cutlass.const_expr(mapping_state.expert_token_sizes is not None):
        token_count = mapping_state.expert_token_sizes[cursor.expert_idx]
    else:
        prefix_end = mapping_state.expert_token_prefix_sum[cursor.expert_idx]
        prefix_begin = Int32(0)
        if cursor.expert_idx > Int32(0):
            prefix_begin = mapping_state.expert_token_prefix_sum[
                cursor.expert_idx - Int32(1)
            ]
        token_count = prefix_end - prefix_begin

    cursor.current_expert_token_count = token_count
    cursor.current_token_block_count = (
        token_count + Int32(mapping_state.mapping_cluster_tile_m - 1)
    ) // Int32(mapping_state.mapping_cluster_tile_m)
    cursor.expert_tile_start = cursor.expert_tile_end
    cursor.expert_tile_end = (
        cursor.expert_tile_start
        + cursor.current_token_block_count * Int32(cursor.blocks_per_token_block)
    )
    return cursor


@cute.jit
def _seek_phase_cursor(
    linear_work_id: Int32,
    cursor: _PhaseFc12CursorState,
    mapping_state: PhaseInterleavedFc12MappingState,
) -> _PhaseFc12CursorState:
    expert_tile_end = cursor.expert_tile_end
    next_expert_idx = cursor.expert_idx + Int32(1)
    while linear_work_id >= expert_tile_end and next_expert_idx < Int32(
        mapping_state.expert_count
    ):
        cursor = _advance_phase_cursor(cursor, mapping_state)
        expert_tile_end = cursor.expert_tile_end
        next_expert_idx = cursor.expert_idx + Int32(1)
    return cursor


@cute.jit
def resolve_phase_interleaved_fc1_claim_target(
    minimum_claim_count: Int32, mapping_state: PhaseInterleavedFc12MappingState
) -> Tuple[Int32, Boolean]:
    """Resolve the static FC1 watermark or the smaller runtime stream extent."""
    target_work_id = minimum_claim_count - Int32(1)
    probe_cursor = _seek_phase_cursor(
        target_work_id, mapping_state.fc1_cursor.clone(), mapping_state
    )
    stream_ends_before_target = target_work_id >= probe_cursor.expert_tile_end
    claim_target = minimum_claim_count
    if stream_ends_before_target:
        claim_target = probe_cursor.expert_tile_end
    return claim_target, stream_ends_before_target


@cute.jit
def _decode_phase_work_id(
    linear_work_id: Int32,
    phase: Int32,
    cta_id_in_mapping_cluster: cute.Coord,
    cursor: _PhaseFc12CursorState,
    mapping_state: PhaseInterleavedFc12MappingState,
) -> SchedulerWorkTileBase:
    local_work_id = linear_work_id - cursor.expert_tile_start
    cluster_token_block_idx = local_work_id // Int32(cursor.blocks_per_token_block)
    cluster_output_block_idx = local_work_id - cluster_token_block_idx * Int32(
        cursor.blocks_per_token_block
    )
    if cutlass.const_expr(mapping_state.dfc1_m_group > 1):
        if phase == Int32(BlockPhase.Linear2):
            # Within each small M strip, visit its rows for a fixed N weight
            # tile before advancing N. Only the per-expert rectangle is
            # permuted: expert ranges, real row/ready indices, and task counts
            # remain unchanged. A short final strip uses its actual row count
            # and therefore adds neither padding work nor duplicate tiles.
            strip_span = Int32(
                mapping_state.dfc1_m_group * cursor.blocks_per_token_block
            )
            strip = local_work_id // strip_span
            strip_m_begin = strip * Int32(mapping_state.dfc1_m_group)
            strip_rows = cutlass.max(
                Int32(1),
                cutlass.min(
                    Int32(mapping_state.dfc1_m_group),
                    cursor.current_token_block_count - strip_m_begin,
                ),
            )
            in_strip = local_work_id - strip * strip_span
            cluster_output_block_idx = in_strip // strip_rows
            cluster_token_block_idx = (
                strip_m_begin + in_strip - cluster_output_block_idx * strip_rows
            )

    cta_token_block_idx = (
        cluster_token_block_idx * Int32(mapping_state.mapping_cluster_shape_mn[0])
        + cta_id_in_mapping_cluster[0]
    )
    cta_output_block_idx = (
        cluster_output_block_idx * Int32(mapping_state.mapping_cluster_shape_mn[1])
        + cta_id_in_mapping_cluster[1]
    )

    cta_tile_m = mapping_state.mapping_cta_tile_shape_mnk[0]
    valid_tokens_in_cta_tile = _valid_tokens_in_token_tile(
        cursor.current_expert_token_count, cta_token_block_idx, cta_tile_m
    )

    if cutlass.const_expr(mapping_state.is_swap_ab):
        return SwapAbFc12WorkTileInfo(
            expert_idx=cursor.expert_idx,
            tile_m_idx=cta_output_block_idx,
            tile_n_idx=cta_token_block_idx,
            cumulative_data_physical_row=cursor.data_cumulative,
            cumulative_sf_physical_row=cursor.sf_cumulative,
            cumulative_token_block_count=cursor.token_block_cumulative,
            valid_tokens_in_cta_tile=valid_tokens_in_cta_tile,
            phase_and_flags=phase,
        )

    cluster_tile_m = mapping_state.mapping_cluster_shape_mn[0] * cta_tile_m
    cluster_token_start = cluster_token_block_idx * Int32(cluster_tile_m)
    remaining_cluster_tokens = cutlass.max(
        cursor.current_expert_token_count - cluster_token_start, Int32(0)
    )
    valid_tokens_in_cluster_tile = cutlass.min(
        remaining_cluster_tokens, Int32(cluster_tile_m)
    )
    return NonSwapAbFc12WorkTileInfo(
        expert_idx=cursor.expert_idx,
        tile_m_idx=cta_token_block_idx,
        tile_n_idx=cta_output_block_idx,
        cumulative_data_physical_row=cursor.data_cumulative,
        cumulative_sf_physical_row=cursor.sf_cumulative,
        cumulative_token_block_count=cursor.token_block_cumulative,
        valid_tokens_in_cta_cluster_tile=(valid_tokens_in_cta_tile << Int32(16))
        | valid_tokens_in_cluster_tile,
        phase_and_flags=phase,
    )


@cute.jit
def map_phase_interleaved_fc12_work_id(
    linear_work_id: Int32,
    phase: Int32,
    cta_id_in_mapping_cluster: cute.Coord,
    mapping_state: PhaseInterleavedFc12MappingState,
) -> Tuple[SchedulerWorkTileBase, Boolean, PhaseInterleavedFc12MappingState]:
    """Map one phase-local ID and report whether the selected stream contains it."""
    work_tile = make_fc12_done_tile(mapping_state.is_swap_ab)
    stream_has_work = Boolean(False)
    fc1_cursor = mapping_state.fc1_cursor
    fc2_cursor = mapping_state.fc2_cursor

    if phase == Int32(BlockPhase.Linear1):
        fc1_cursor = _seek_phase_cursor(linear_work_id, fc1_cursor, mapping_state)
        if linear_work_id < fc1_cursor.expert_tile_end:
            work_tile = _decode_phase_work_id(
                linear_work_id,
                phase,
                cta_id_in_mapping_cluster,
                fc1_cursor,
                mapping_state,
            )
            stream_has_work = Boolean(True)
    else:
        fc2_cursor = _seek_phase_cursor(linear_work_id, fc2_cursor, mapping_state)
        if linear_work_id < fc2_cursor.expert_tile_end:
            work_tile = _decode_phase_work_id(
                linear_work_id,
                phase,
                cta_id_in_mapping_cluster,
                fc2_cursor,
                mapping_state,
            )
            stream_has_work = Boolean(True)

    mapping_state.fc1_cursor = fc1_cursor
    mapping_state.fc2_cursor = fc2_cursor
    return work_tile, stream_has_work, mapping_state


__all__ = [
    "BlockPhase",
    "Fc12TaskMappingState",
    "Fc12WorkTileState",
    "NonSwapAbFc12WorkTileInfo",
    "PhaseInterleavedFc12MappingState",
    "SharedExpertTiles",
    "SwapAbFc12WorkTileInfo",
    "count_fc12_tiles",
    "count_fc12_tiles_from_sizes",
    "create_fc12_task_mapping_state",
    "create_phase_interleaved_fc12_mapping_state",
    "load_fc12_token_count_cache",
    "make_fc12_done_tile",
    "map_fc12_linear_work_id",
    "map_phase_interleaved_fc12_work_id",
    "map_shared_expert_work_id",
    "peek_ready_bit",
    "resolve_phase_interleaved_fc1_claim_target",
]
