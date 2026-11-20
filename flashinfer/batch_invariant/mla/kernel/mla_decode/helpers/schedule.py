# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Captured-schedule helper functions for MLA decode TS examples."""


def captured_loop_bounds(task_kwargs, default_start: int, default_step: int = 1):
    """Consume Task loop kwargs and return captured schedule loop bounds."""
    loop_start = task_kwargs.pop("domain_start", default_start)
    loop_step = task_kwargs.pop("step", default_step)
    loop_end = task_kwargs.pop("domain", None)
    if loop_end is None:
        loop_end = 1
    return loop_start, loop_end, loop_step


def work_queue_tail(work_queue, *, advance_label: str = "get_and_advance_work_tile"):
    """Advance the persistent work tile at the end of a task body."""
    if work_queue is None:
        return
    work_queue.wait()
    getattr(work_queue, advance_label)()
    work_queue.release()


def staged_kv_tma_load(
    smem_kv,
    iterations: int,
    cached_k_pages,
    cached_v_pages,
    cached_next_v_pages,
    *,
    is_v: bool,
    use_next_v_pages: bool = False,
):
    """Produce staged K or delayed-V TMA work for one logical k-tile."""
    for subtile_idx in range(iterations):
        smem_kv.acquire()
        smem_kv.tma_load(
            cached_k_pages=cached_k_pages,
            cached_v_pages=cached_v_pages,
            cached_next_v_pages=cached_next_v_pages,
            is_v=is_v,
            subtile_idx=subtile_idx,
            use_next_v_pages=use_next_v_pages,
        )
        smem_kv.commit()


def staged_qk_mma(
    smem_kv,
    tmem_s,
    iterations=None,
    *,
    head_dim_stages=None,
    consumer_label=None,
    include_acquire=True,
    release_k=True,
    whole_kv=False,
):
    """Consume staged K descriptors and produce one QK MMA score tile."""
    if consumer_label is None:
        for k_subtile_idx in range(iterations):
            smem_kv.wait()
            desc_k_base = smem_kv.k_desc(k_subtile_idx=k_subtile_idx)
            if k_subtile_idx == 0:
                tmem_s.acquire()
            tmem_s.qk_mma(desc_k_base=desc_k_base, k_subtile_idx=k_subtile_idx)
            smem_kv.release()
        tmem_s.commit()
        return

    if include_acquire:
        tmem_s.acquire()
    if whole_kv:
        smem_kv.wait()
    for k_subtile_idx in range(head_dim_stages):
        if not whole_kv:
            smem_kv.wait()
        kv_desc = getattr(smem_kv, consumer_label)(k_subtile_idx=k_subtile_idx)
        tmem_s.qk_mma(kv_desc=kv_desc, k_subtile_idx=k_subtile_idx)
        if release_k and not whole_kv:
            smem_kv.release()
    tmem_s.commit()
    if whole_kv and release_k:
        smem_kv.release()


def staged_pv_mma(
    smem_kv,
    smem_p,
    tmem_o,
    iterations=None,
    *,
    head_dim_stages=None,
    consumer_label=None,
    producer_label=None,
    is_tail: bool = False,
    reuse_k=False,
    retained_k_stages=0,
    whole_kv=False,
):
    """Consume staged V descriptors and produce one PV MMA output tile."""
    smem_p.wait()
    if producer_label is None:
        desc_p_base = smem_p.p_desc()
        tmem_o.acquire()
        for v_subtile_idx in range(iterations):
            smem_kv.wait()
            desc_v_base = smem_kv.v_desc(v_subtile_idx=v_subtile_idx)
            tmem_o.pv_mma(
                desc_p_base=desc_p_base,
                desc_v_base=desc_v_base,
                v_subtile_idx=v_subtile_idx,
                is_tail=is_tail,
            )
            smem_kv.release()
        tmem_o.commit()
        smem_p.release()
        return

    smem_p.p_desc()
    tmem_o.acquire()
    is_tail = is_tail or (
        producer_label is not None and producer_label.find("tail") >= 0
    )
    for v_subtile_idx in range(head_dim_stages):
        if not reuse_k:
            smem_kv.wait()
        if producer_label.endswith("_0"):
            v_desc_0 = getattr(smem_kv, consumer_label)(v_subtile_idx=v_subtile_idx)
            getattr(tmem_o, producer_label)(
                v_desc_0=v_desc_0,
                v_subtile_idx=v_subtile_idx,
                is_tail=is_tail,
            )
        else:
            v_desc_1 = getattr(smem_kv, consumer_label)(v_subtile_idx=v_subtile_idx)
            getattr(tmem_o, producer_label)(
                v_desc_1=v_desc_1,
                v_subtile_idx=v_subtile_idx,
                is_tail=is_tail,
            )
        if not whole_kv:
            smem_kv.release()
    if whole_kv:
        # QK and every PV feature slice consume the same complete latent tile.
        smem_kv.release()
    elif reuse_k:
        # Retained K stages this PV does not read (QK-only RoPE stages, or a
        # V split's other owners' slices) were consumed by QK and must retire
        # before the next tile's V slices.
        for _ in range(retained_k_stages - head_dim_stages):
            smem_kv.release()
    tmem_o.commit()
    smem_p.release()


def staged_qk_mma_k_tile(smem_k, tmem_s, iterations: int):
    """Consume one whole K stage and produce one QK score tile."""
    smem_k.wait()
    tmem_s.acquire()
    for k_subtile_idx in range(iterations):
        desc_k_base = smem_k.k_desc(k_subtile_idx=k_subtile_idx)
        tmem_s.qk_mma(desc_k_base=desc_k_base, k_subtile_idx=k_subtile_idx)
    tmem_s.commit()
    smem_k.release()


def staged_pv_mma_v_tile(smem_v, smem_p, tmem_o, iterations: int):
    """Consume one whole V stage plus one P stage and produce one O tile."""
    smem_p.wait()
    desc_p_base = smem_p.p_desc()
    smem_v.wait()
    tmem_o.acquire()
    for v_subtile_idx in range(iterations):
        desc_v_base = smem_v.v_desc(v_subtile_idx=v_subtile_idx)
        tmem_o.pv_mma(
            desc_p_base=desc_p_base,
            desc_v_base=desc_v_base,
            v_subtile_idx=v_subtile_idx,
        )
    tmem_o.commit()
    smem_v.release()
    smem_p.release()


def staged_pv_mma_v_tile_per_n(
    smem_v,
    smem_p,
    tmem_o,
    *,
    iterations_pv_k: int,
    iterations_pv_n: int,
):
    """Consume P/V and publish one O pipeline token per PV N-slice."""
    smem_p.wait()
    desc_p_base = smem_p.p_desc()
    smem_v.wait()
    for pv_n_idx in range(iterations_pv_n):
        tmem_o.acquire()
        for pv_k_idx in range(iterations_pv_k):
            desc_v_base = smem_v.v_desc_n_major(pv_n_idx=pv_n_idx, pv_k_idx=pv_k_idx)
            tmem_o.pv_mma_n_major(
                desc_p_base=desc_p_base,
                desc_v_base=desc_v_base,
                pv_n_idx=pv_n_idx,
                pv_k_idx=pv_k_idx,
            )
        tmem_o.commit()
    smem_v.release()
    smem_p.release()
