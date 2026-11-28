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

"""Correction tasks of the throughput 2CTA MLA decode kernel: O rescale, normalization and partial output."""

from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task
from ...device.schedule import captured_loop_bounds
from ..resources import GmemOResource, TmemCorrResource, TmemOResource
from .common import MlaTask, capture_clc_work_tile_body
from ..work_queue import MlaWorkQueue


def create_correction_task(
    tmem_corr: TmemCorrResource,
    tmem_o: TmemOResource,
    gmem_o: GmemOResource,
    iterations_pv_n: int = 1,
    per_n_o_pipeline: bool = False,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the Correction task (warps 4-7, 4 warps, 208 regs).

    domain_start=1: HEAD handles initial correction (no O yet), LOOP
    handles correction+O pairs, TAIL handles final O + epilogue.

    TMEM visibility is ensured by kernel-level named barrier sync before
    task_manager.run(), so o_init pipeline is no longer needed here.

    HEAD: consume Corr[0] (initial max/sum, no O rescaling).
    LOOP: consume Corr[n] + O[n-1], rescale accumulated O.
    TAIL: consume O[last], final epilogue store O + LSE to GMEM.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 1)
    use_clc_dynamic = bool(work_queue is not None and work_queue.use_clc_dynamic)
    use_tma_output = bool(gmem_o.cfg.use_tma_output)

    def correction_prelude(tmem_corr):
        """Create correction and epilogue register state before skip guards."""
        return tmem_corr.init_load_state()

    def correction_body(tmem_corr, tmem_o, gmem_o, correction_state):
        """Apply online-softmax correction and store the final O/LSE result."""
        row_sum, row_max, correction_factor, no_correction = correction_state

        # HEAD: consume the first correction factors. There is no O tile yet.
        tmem_corr.wait()
        row_sum, row_max, correction_factor, no_correction = tmem_corr.load_corr()
        tmem_corr.release()

        with domain_loop(loop_start, loop_end, loop_step):
            # LOOP: consume Corr[n] and rescale O[n-1].
            tmem_corr.wait()
            row_sum, row_max, correction_factor, no_correction = tmem_corr.load_corr()
            tmem_corr.release()
            if per_n_o_pipeline:
                for iter_n in range(iterations_pv_n):
                    tmem_o.wait()
                    tmem_o.rescale_o_slice(
                        correction_factor=correction_factor,
                        no_correction=no_correction,
                        iter_n=iter_n,
                    )
                    tmem_o.release()
            else:
                tmem_o.wait()
                tmem_o.rescale_o(
                    correction_factor=correction_factor,
                    no_correction=no_correction,
                )
                tmem_o.release()

        # TAIL: consume final O. Do not call rescale_o here; the correction was
        # already applied in LOOP and the last loop correction value would be
        # stale for a second application. epilogue_store also writes LSE.
        if per_n_o_pipeline:
            epilogue_row_sum, epilogue_row_max = (
                tmem_corr.prepare_epilogue_slice_store()
            )
            for iter_n in range(iterations_pv_n):
                if use_tma_output:
                    gmem_o.acquire()
                tmem_o.wait()
                gmem_o.epilogue_store_slice(
                    row_sum=epilogue_row_sum,
                    row_max=epilogue_row_max,
                    iter_n=iter_n,
                )
                if use_tma_output:
                    gmem_o.commit()
                tmem_o.release()
        else:
            tmem_o.wait()
            gmem_o.epilogue_store()
            tmem_o.release()

    @schedule
    def correction_schedule(tmem_corr, tmem_o, gmem_o, work_queue=None):
        """Capture one active correction tile and unconditional queue progress."""

        capture_clc_work_tile_body(
            work_queue,
            lambda correction_state: correction_body(
                tmem_corr,
                tmem_o,
                gmem_o,
                correction_state,
            ),
            lambda: correction_prelude(tmem_corr),
            use_clc_dynamic=use_clc_dynamic,
        )

    captured_schedule = (
        correction_schedule(tmem_corr, tmem_o, gmem_o)
        if work_queue is None
        else correction_schedule(tmem_corr, tmem_o, gmem_o, work_queue)
    )

    src = [tmem_corr, tmem_o]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[gmem_o],
        warp_idx=4,
        num_warps=4,
        schedule=captured_schedule,
        name="CorrectionTask",
        **task_kwargs,
    )
