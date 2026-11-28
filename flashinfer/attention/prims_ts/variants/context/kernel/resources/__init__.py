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

"""Resource definitions for the TS FMHA kernel.

Maps the FMHA data-flow onto ``MemoryResource`` subclasses:

Each resource owns the work attached to one live buffer: producer methods fill
the buffer, consumer methods drain it, and the pipeline state records when the
next task may use the data. Task files only order these resource work calls.

Schedule phase terms follow TS schedule-builder naming. HEAD is the one-time
schedule before the repeated K/V tile loop, LOOP is the repeated K/V tile body,
and TAIL is the one-time cleanup and drain after LOOP exits.

VC-Attention is the low-bit attention recipe of Li et al. 2026, arXiv
2609.15810, value smoothing with a softmax cast. Names with the ``vc`` prefix
belong to its recipes, QK16 with bf16 Q and K and QK8 with centred, Hadamard
rotated E4M3 Q and K under block scales, both with E4M3 V stored as residuals
around each 128-token tile mean. The kernel restores the means inside the
online softmax as ``O += sum_t rowsum(P_t) * mu_t``, one bf16 K=16 UMMA step
per group of sixteen tiles, and casts the probabilities straight to E4M3
codes. ``smem_mu`` holds the mean operand ring; ``device.expcast`` the cast helpers.

Each resource below lives in its own module of this package.

SMEM resources (TMA pipelines)
------------------------------
- SmemQResource   : SMEM Q buffer, TmaUmma pipeline. Load -> MMA.
- SmemKVResource  : SMEM K/V buffer, TmaUmma pipeline. Load -> MMA. The D256
                    depth is derived from the public SM100 SMEM capacity. One
                    instance holds both K and V when they share a dtype; a
                    QK/PV dtype mismatch instantiates one K-only and one V-only
                    instance, each sized from its own dtype.
- SmemOResource   : One SMEM O buffer per Q/O instance, AsyncAsync pipeline.
                    Correction -> Epilogue.

TMEM resources (split from former TmemComputeResource)
------------------------------------------------------
- TmemSPResource  : S/P buffer, UmmaAsync. D256 uses a two-stage S/P ring and
                    an independent TmemPResource readiness handoff.
                    MMA writes S (Q*K scores). Softmax reads S, computes P,
                    and writes P back into the same slot for BMM2.
                    Self-edge in dependency graph enables ping-pong validation.
- TmemStatsResource : Correction statistics, AsyncAsync pipeline.
                    Softmax writes [old_max, new_max, row_sum], Correction reads.
- TmemOResource   : O accumulation, UmmaAsync pipeline.
                    MMA writes P*V -> O. Correction waits for O, rescales it
                    in-place, then releases the stage for the interleaved MMA.

Sequencing resources
--------------------
- S0S1SequenceResource : PipelineAsync (1 stage), Softmax0 → Softmax1.
                         Paces the two softmax groups: Softmax1 waits for
                         Softmax0's token each iteration. The FP8 cadences
                         (D128 FP8 softmax, FP8 P in SMEM with the early
                         token) hand the token over before ``exp2_p``, so the
                         groups overlap P computation and the token does not
                         order P stores. Each group's own P handoff orders its
                         P stores before the PV MMA reads them:
                         - P in TMEM (written over S in the TmemSPResource
                           slot): the S/P release after the P store, or the
                           TmemPResource commit with split S/P (D256). With
                           ``pv_half_overlap``, TmemPPrefixReadyResource lets
                           the MMA read the first 64 keys of P before the S/P
                           release.
                         - P in SMEM: softmax releases the S/P stage before
                           ``exp2_p`` (S is already in registers);
                           SmemPResource acquire/commit publishes the P tile
                           and the MMA waits on it.
                         See the softmax schedule in ``tasks/softmax.py``. Operations
                         are inlined in TmemSPResource.consumer_work.

GMEM resources (no pipeline)
-----------------------------
- GmemQKVResource : TMA descriptors + per-tile coordinate resolution.
- GmemOResource   : TMA descriptor for O stores.
"""
