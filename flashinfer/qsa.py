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
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch

from .qsa_attention import QSAAttention
from .qsa_selection import QSASelection, selection_route_width
from .topk import WORKSPACE_ALIGNMENT


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


@dataclass(frozen=True)
class QSAConfig:
    """Everything about a deployment's QSA that a cache does not decide.

    One object because the two halves of a QSA step are sized and built
    together and share a buffer, and because a caller holding two descriptions
    can let them disagree. Deliberately absent are ``num_slots`` and
    ``page_size``: both are what a cache has, and a cache does not exist when
    this is first asked -- the page size in particular is settled when the
    cache is allocated, which is after this has been answered.

    Attributes
    ----------
    num_qo_heads, num_kv_heads, head_dim : int
        The attention's query and cache heads, and the dimension of both.
    max_rows : int
        The widest batch a step may bring. The plan ladder's top rung is this,
        and a batch past it has no plan.
    q_data_type, kv_data_type, o_data_type : torch.dtype
        Query, cache and output dtypes. A packed NVFP4 cache is ``uint8`` here
        and says so through ``kv_cache_format``.
    kv_cache_format : str
        ``dense``, ``fp8_e4m3`` or ``nvfp4``. What the cache's bytes mean.
    kv_layout : str
        ``NHD`` or ``HND``, as the cache is laid out.
    max_columns : int
        The widest score row the selection can reach, which the deployment's
        context length and compression decide.
    compress_ratio, token_topk : int
        How many tokens one compressed key stands for, and how many tokens a
        query keeps. Together they fix the route's width.
    index_num_heads, index_head_dim : int
        The index side's query shape, which is not the attention's.
    backend : str
        Which block-sparse backend to plan for.
    max_plans : int
        Bounds how many rungs the plan ladder may have.
    """

    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    max_rows: int
    q_data_type: torch.dtype
    kv_data_type: torch.dtype
    o_data_type: torch.dtype
    kv_cache_format: str
    max_columns: int
    compress_ratio: int
    token_topk: int
    index_num_heads: int
    index_head_dim: int
    kv_layout: str = "NHD"
    backend: str = "auto"
    max_plans: int = 16

    @property
    def route_width(self) -> int:
        """Columns the route carries, which the top-k and compression fix."""
        return selection_route_width(self.token_topk, self.compress_ratio)


class QSAWorkspaceRequirements(NamedTuple):
    """What a caller has to hand over, and nothing about how it is used.

    Two numbers because the bytes have two lifetimes. What is persistent is
    read back on later calls and so needs memory nobody else writes; what is
    transient is rewritten before it is read and may come out of whatever
    scratch the caller shares between the consumers of a step.

    Attributes
    ----------
    persistent_bytes : int
        One contiguous ``uint8`` buffer this object keeps for its lifetime.
    transient_bytes : int
        One contiguous ``uint8`` buffer a call may share with anything else.
    alignment : int
        What the first byte of each has to start on.
    """

    persistent_bytes: int
    transient_bytes: int
    alignment: int


class QSA:
    """One deployment's QSA, selection and attention together.

    The caller provides two contiguous buffers and this cuts everything out of
    them. Where each piece sits is not the caller's business; which buffer a
    piece comes out of is decided by whether a later call reads it back:

    * the plan arena and the row pointers are written once and read by every
      call, so they live in the persistent buffer,
    * the padded query and output, the physical route and its mask, the
      selection's scratch and the float workspace are rewritten before they are
      read, so they live in the transient one.

    What the caller keeps for itself is the *logical* route -- the token
    indices the selection writes and the attention reads. That one is per
    layer, because a caller reusing a route across speculative steps reuses
    that layer's, so it cannot live in a buffer this object shares.

    The order is fixed and it is the whole lifecycle:

    1. :meth:`workspace_requirements` while the caller's scratch can still grow,
    2. construction against the persistent buffer it allocated,
    3. :meth:`bind_transient_workspace` once its scratch has stopped moving,
    4. :meth:`plan_cache` once the cache exists, again if it is replaced,
    5. :meth:`run_selection` and :meth:`run_attention`, after which the plans
       are frozen because a graph may be replaying them.
    """

    def __init__(self, config: QSAConfig, persistent: torch.Tensor) -> None:
        self.config = config
        self.device = persistent.device
        self.route_width = config.route_width
        self._attention = QSAAttention(
            persistent,
            max_rows=config.max_rows,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            route_width=config.route_width,
            q_data_type=config.q_data_type,
            kv_data_type=config.kv_data_type,
            o_data_type=config.o_data_type,
            kv_cache_format=config.kv_cache_format,
            kv_layout=config.kv_layout,
            backend=config.backend,
            max_plans=config.max_plans,
        )
        self._selection = QSASelection(
            max_rows=config.max_rows,
            max_columns=config.max_columns,
            compress_ratio=config.compress_ratio,
            token_topk=config.token_topk,
            num_heads=config.index_num_heads,
            head_dim=config.index_head_dim,
            device=persistent.device,
        )

    def bind_transient_workspace(self, transient: torch.Tensor) -> None:
        """Take the scratch both halves rewrite before they read.

        Handed over once the caller's scratch has stopped moving. A workspace
        that grows by reallocating frees what earlier views point into, so a
        view taken before it settles is a view into memory somebody else now
        owns.
        """
        attention_bytes, _selection_bytes = QSA._transient_split(
            self.config, transient.device
        )
        self._attention.bind_transient_workspace(transient[:attention_bytes])
        self._selection.bind_workspace(transient[attention_bytes:])

    # -- sizing ------------------------------------------------------------

    @staticmethod
    def _attention_bytes(config: QSAConfig, device: torch.device):
        return QSAAttention.workspace_bytes(
            device=device,
            max_rows=config.max_rows,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            route_width=config.route_width,
            q_data_type=config.q_data_type,
            kv_data_type=config.kv_data_type,
            o_data_type=config.o_data_type,
            kv_cache_format=config.kv_cache_format,
            kv_layout=config.kv_layout,
            backend=config.backend,
            max_plans=config.max_plans,
        )

    @staticmethod
    def _selection_bytes(config: QSAConfig, device: torch.device) -> int:
        return QSASelection.plan_workspace_size(
            device=device,
            max_rows=config.max_rows,
            max_columns=config.max_columns,
            compress_ratio=config.compress_ratio,
            token_topk=config.token_topk,
            num_heads=config.index_num_heads,
            head_dim=config.index_head_dim,
        )

    @staticmethod
    def _transient_split(config: QSAConfig, device: torch.device):
        _persistent, attention = QSA._attention_bytes(config, device)
        selection = QSA._selection_bytes(config, device)
        return (
            _round_up(attention, WORKSPACE_ALIGNMENT),
            _round_up(selection, WORKSPACE_ALIGNMENT),
        )

    @staticmethod
    def workspace_requirements(
        config: QSAConfig, *, device: torch.device
    ) -> QSAWorkspaceRequirements:
        """How much memory of each lifetime this configuration runs out of.

        Asked before the caller's scratch is locked, which is before there is a
        cache, so nothing is built to answer: both halves compute their sizes
        from the geometry. A caller that built one to ask would pay the float
        workspace and every buffer at startup, for a question.
        """
        persistent, _attention = QSA._attention_bytes(config, device)
        attention, selection = QSA._transient_split(config, device)
        return QSAWorkspaceRequirements(
            persistent_bytes=_round_up(persistent, WORKSPACE_ALIGNMENT),
            transient_bytes=attention + selection,
            alignment=WORKSPACE_ALIGNMENT,
        )

    # -- planning ----------------------------------------------------------

    def plan_cache(self, num_slots: int, page_size: int) -> None:
        """Plan for a cache of this many slots, in the workspace already held.

        Called once the cache exists, and again if it is replaced -- a caller
        that binds a minimal cache for a memory profile and then the real one
        calls this twice. The workspace does not change and nothing is
        allocated by the second call. Asking for the cache it already has does
        nothing, so every layer of a rank may call it and only the first does.

        Refused once a run has happened: the plans keep byte offsets that a
        captured graph replays, so replacing one under a capture that holds it
        would be reading a schedule that is no longer there.
        """
        self._attention.plan_cache(num_slots, page_size)

    @property
    def num_slots(self) -> Optional[int]:
        """The cache size the plans are for, or ``None`` before there is one."""
        return self._attention.num_slots

    # -- execution ---------------------------------------------------------

    def run_selection(
        self,
        q: torch.Tensor,
        k_compressed: torch.Tensor,
        page_table: torch.Tensor,
        token_to_req: torch.Tensor,
        query_positions: torch.Tensor,
        sequence_lengths: torch.Tensor,
        *,
        out_route: torch.Tensor,
    ) -> torch.Tensor:
        """Choose this batch's tokens, writing them into the caller's route.

        ``out_route`` is the caller's, not this object's: a caller reusing a
        route across speculative steps reuses the one belonging to that layer,
        and this object is shared between layers.
        """
        return self._selection.run(
            q,
            k_compressed,
            page_table,
            token_to_req,
            query_positions,
            sequence_lengths,
            out_route=out_route,
        )

    def run_attention(
        self,
        q: torch.Tensor,
        k_data: torch.Tensor,
        v_data: torch.Tensor,
        *,
        route: torch.Tensor,
        block_table: torch.Tensor,
        token_to_req: torch.Tensor,
        output_gate: torch.Tensor,
        k_sf: Optional[torch.Tensor] = None,
        v_sf: Optional[torch.Tensor] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attend over the route the selection wrote, gate it, and return it.

        The cache planes arrive per call rather than being held: a caller whose
        cache is replaced replans and keeps running, and nothing here outlives
        a tensor the caller owns.
        """
        return self._attention.run(
            q,
            k_data,
            v_data,
            route=route,
            block_table=block_table,
            token_to_req=token_to_req,
            output_gate=output_gate,
            k_sf=k_sf,
            v_sf=v_sf,
            k_scale=k_scale,
            v_scale=v_scale,
            out=out,
        )
