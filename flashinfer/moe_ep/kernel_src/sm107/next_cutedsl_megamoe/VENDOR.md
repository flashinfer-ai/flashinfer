# Vendoring record: next_cutedsl_megamoe

This directory contains one pinned upstream kernel snapshot plus the explicit
local patches recorded below.
[SKILL.md](SKILL.md) describes updates; [TUNING.md](TUNING.md) covers measurements.

## Upstream

- **Source**: the kernel team's `cutedsl_megamoe` repository; see the shared
  [acknowledgements](../../sm100/cutedsl_megamoe/ACKNOWLEDGEMENT.md).
- **Vendored commit**:
  `1667b47a3c911ecade464ab524baf192a0bf5962`,
  committed 2026-09-16, the upstream `main` snapshot selected for this refresh.
- **Last synced**: 2026-09-16, using the upstream exporter from that same commit.
  Previous drop: `92dd334af2eeedb36087834354b58ace08e880c6` (2026-08-15), inherited
  from FlashInfer PR #4601. The older `47881ad2` reference described equivalent
  inference files in that previous drop; it is not the current source pin.
- **Exporter**: `next/export_src.py` at the vendored commit,
  SHA-256 `a1deca5d171a4ba92ee56b5ee82f064d5dc8f44cdf52b9ce49c88d46f10fb523`.
- **Selected kernels**: `RubinInferenceMegaMoE` and `RubinInferenceGenphaseMegaMoE`.
  Their dependency closure is exported to `src/sources/`: 32 implementation
  modules and nine generated package initializers, 41 files total.
- **Verification**: two independent exports at the pinned commit produced
  identical file lists and bytes before local patches. The exceptions are listed below.
  The exporter validates internal import closure and rejects
  unresolved imports and dependencies outside its external-module allowlist.

The export command, from a checkout at the pinned SHA, is:

```bash
python next/export_src.py \
    --kernels RubinInferenceMegaMoE RubinInferenceGenphaseMegaMoE \
    --dst-dir "${FI_EXPORT_STAGE:?}/sources"
```

The destination must be absent or empty and its parent must exist. Preserve the
export manifest/log and file hashes with the integration evidence.

## Policy

- `src/` starts from a **verbatim copy of the pinned upstream export output**;
  the explicit local exceptions below must be reapplied and qualified. Do not
  hand-edit, format or inject files into it. Compare it with a regenerated
  export, rather than expecting every generated file to match a raw upstream path.
- All FlashInfer adaptation lives in `shim/`, re-exported through `__init__.py`;
  backends import the package API, never raw `sources` modules.
- Device-kernel and exporter fixes go upstream first, then re-export. Any
  unavoidable local exception must be recorded below until upstream absorbs it.
  See the shared [vendoring rules](../../README.md).

## Scope of this drop

The generic inference entry point is `RubinInferenceMegaMoE`
(`BlockScaledSwapAbMegaMoeKernel`). FlashInfer's NVFP4, MXFP4, MXFP8 E4M3/E5M2, and MXFP4-weight/MXFP8-activation
backends expose SwiGLU and SiTU with BF16 output. BF16 combine supports separate
and in-kernel reduction; `combine_dtype="nvfp4"` and `combine_dtype="mxfp8"`
require separate reduction. They accept canonical prequantized weights; NVFP4
also supports non-unit normalization and per-expert correction tensors.
Shared helpers, schedulers, communication and workspace code come from the
same pinned snapshot.

`RubinInferenceGenphaseMegaMoE` (`BlockScaledSwapAbGenphaseMoeKernel`) is selected
with `kernel_variant="genphase"`. It requires `fc2_use_bulk=True`, at most
1024 tokens per rank, and uniform clusters (no mixed-CGA fallback). The
inherited tile-N 128/256 configurations use `(4, 1)` clusters. The local
`(256, 64, 128)` decode tile additionally admits MXFP8 E4M3 with `(2, 1)` or
`(4, 1)` clusters, and MXFP4 weights/MXFP8 E4M3 activations with `(4, 1)`
clusters. At EP greater than one, the N64 path requires grouped scheduling:
an N64 phase-interleave EP2 dense/replay case stalls despite EP1 and EP4
coverage. The shim rejects that combination before allocation or compilation,
and `is_valid_sm107` excludes it from the geometry-specific tuning set. EP1
phase-interleave and the inherited N128/N256 admission remain unchanged.
Other N64 combinations remain rejected. Combine remains BF16.
Tile K is 256 for NVFP4/MXFP4 activations and 128 for MXFP8 activations.
SiTU requires both beta parameters, matching the exported kernel.

`combine_dtype="nvfp4"` selects the existing quantized FC2 return path.
It sends packed E2M1 values with a BF16 amax per 16 elements.
`combine_dtype="mxfp8"` sends E4M3 values with an E8M0 scale per 32 elements.
Both formats use the generic inference kernel. Training kernels and the local
fused-routing kernel are not selected; the latter supplies one dependency through an export marker.
No upstream tester, build metadata or repository scaffolding is vendored.

## Compiler contract and validation

The shim requires native `sm_107a` and `cutlass.utils.rubin_helpers`, available
in public `nvidia-cutlass-dsl==4.8.0.dev0` and compatible newer builds. Install the
`sm107` extra and set `CUTE_DSL_ARCH=sm_107a` before Python imports the DSL.
The shim checks capabilities and the target captured at import.

The older `92dd334` payload passed integration correctness at FlashInfer revision
`9a414e73c8f4246746b281f98db2217a515819cb`, including EP2/4/8. That evidence is for
the older drop. At FlashInfer `0f710df8`, this export passed 187 host/CUDA checks,
72 single-GPU cases, and 34 distributed cases per rank at EP2, EP4, and EP8,
with no skips. See the
[qualification guide](../../../../../docs/design_docs/moe_ep_sm107_qualification.md)
for coverage and the separate reference-decoder regression.
Tuning-cache entries use revision `sm107-block-scaled-1667b47a-runtime-options-v1`.
The cache identity includes the kernel variant and combine format, along with
activation, beta parameters, and clamp settings.

## Export transformations

Upstream's exporter materializes three `COPY_FROM_IMPORT` markers at the pinned commit:

| Exported file under `src/sources/kernel_src/` | Upstream source under `next/sources/kernel_src/` |
| --- | --- |
| `rubin/inference/mega/topk_reduce.py` | `blackwell/inference/mega/topk_reduce.py` |
| `rubin/custom_mix_cga_helpers.py` | `blackwell/custom_mix_cga_helpers.py` |
| `rubin/inference/mega/tma_gather.py` | `rubin/inference/local_mega/tma_gather.py` |

It also generates nine `__init__.py` files and rewrites one import statement in
`block_scaled_swap_ab_fc12_epilogue.py` from scheduler package re-exports to the
implementation modules. The generated root exposes the two selected aliases;
intermediate package initializers are empty. These are upstream exporter
transformations, not manual FlashInfer patches.

## Local differences

The existing comment above `pusher_cta_count` in the GenPhase kernel describes
residency without device-specific counts. In addition, FlashInfer PR
[#6447](https://github.com/flashinfer-ai/flashinfer/pull/6447) carries the following
local device patches. They are not part of upstream `1667b47a` and must not be
silently dropped by a fresh export. Paths below are relative to `src/sources/`.

| File | Local exception |
| --- | --- |
| `helpers/ptx_helpers.py` | Counted, unaligned CTA barrier helper for the actual participating warps. |
| `kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_mainloop.py` | Use that barrier for the Generic MMA/accumulator TMEM allocation rendezvous. |
| `kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_mainloop_gen_specialized.py` | N64 SFB/TMEM parity, valid-row TMA-gather extent and transaction accounting, and two gather warps for CGA4. |
| `kernel_src/rubin/inference/mega/block_scaled_swap_ab_mega_moe_kernel.py` | Pass routing IDs to separate BF16, NVFP4, and MXFP8 reductions, advertise masked-slot handling, and configure small-prefix CTA0 reset ownership. |
| `kernel_src/rubin/inference/mega/block_scaled_swap_ab_mega_moe_kernel_gen_specialized.py` | Propagate masked reduction and admitted N64 geometry; support a one-row allocation in the gather descriptor, explicit router/all-role quiescence before reset, and small-prefix CTA0 reset ownership. |
| `kernel_src/rubin/inference/mega/token_comm_gen_specialized.py` | Bound route metadata by the live extent, fence rewritten shared indices before async bulk publication, reuse the acquired extent in the sort body, skip invalid histogram/scatter work, and reuse the existing expert prefix for scale-factor offsets. |
| `kernel_src/rubin/inference/mega/topk_reduce.py` | Predicate BF16, NVFP4, and MXFP8 combine loads by routing validity, yielding exact zero for rows with no valid routes. |
| `communication/nvlink_domain/token_comm.py` | Skip invalid-route histogram/scatter work, protect partial dump reads, fence generic shared-memory writes before async bulk size publication, and let CTA0 reset prefixes up to 16 KiB without a second grid join. |

The Generic router shortcut is retained with an async-proxy publication fence
before the shared-memory histogram/size totals are consumed by bulk copy and
bulk reduction. A CTA barrier alone is insufficient: the sparse multirank graph
regression fails with only an extra barrier and passes with the proxy fence.
The regression covers early/late weights, changing routes, and full-capacity
output checks; the multi-CTA Generic path is included at capacity 2048 and top-k 10.

The Generic tail keeps the producer-drain grid join and both NVLink peer
barriers. For EP greater than one and at most 16 KiB of combined local/shared
reset prefixes, CTA0 clears all original reset bytes and skips the second grid
prologue. CTA0 joins after the drain's thread-zero peer wait/phase increment,
so no reset writer or next-phase reader can overtake that wait. The publication
barrier orders all CTA0 reset writers before peer signaling. Larger prefixes,
EP1, and `MEGA_USE_NCU=1` keep the original distributed reset. The selected
ownership is part of the generated kernel name.

GenPhase uses the same size/EP/NCU eligibility rule, with its own tail protocol:
drain the preceding router with `griddepcontrol_wait`, join every local warp
role, then retain the local grid join and single NVLink publication barrier.
These drains are required for both reset ownership choices: empty work can
finish the epilogue while gather roles still acquire ready flags or the router
still advances its refinement counter. A seven-pusher changing-route stress
test covers early/late weights without changing the default 27-pusher geometry. CTA0 clears every original local/shared reset prefix, then uses the
barrier's existing CTA rendezvous, system fence, peer notification and acquire.
Only the post-reset grid join is omitted. Persistent grid/phase/signal counters
are outside the reset prefix. GenPhase's reset bound depends on both top-k and
local expert count; large-prefix constructor checks must stay within its
1024-token capacity limit. EP16 construction is a layout test, not EP16 execution.

One more FlashInfer-owned kernel fix, pending upstream (the pinned commit and
upstream `main` up to `413da003` have the same code):

- `src/sources/kernel_src/schedulers/fc12_scheduler.py`,
  `PhaseInterleavedFc12Scheduler._wait_for_fc1_prologue_claims`: the
  phase-interleave scheduler no longer sets `fc1_exhausted` from its CTA-local
  peek of the global FC1 work-ID counter. Every CTA of a cluster runs the claim
  state machine itself and interprets the leader's broadcast work ID against
  its own stream choice, so that state must only change on cluster-uniform
  inputs. When the runtime FC1 stream is shorter than the static FC1 watermark
  (tiny per-rank work, e.g. DP dummy batches), the last terminal FC1 claims of
  other clusters can push the counter across the watermark between the peeks
  of two CTAs of one cluster (late-starting clusters make this likely). The
  CTAs then disagreed on `fc1_exhausted`, mapped the next broadcast ID to
  different phases, and the cluster deadlocked (CTAs leaving the claim loop
  while the leader keeps claiming, or 2-CTA MMA / multicast pairs on different
  tiles): one rank 2 CTAs short of the tail grid sync, its peers waiting in
  the NVLink drain barrier. The peek now only shortens this CTA's wait; FC1
  exhaustion is learned from a claimed ID like any other stream end, at the
  cost of at most one extra terminal FC1 claim per cluster. Re-apply this fix
  when the drop is re-exported until upstream carries it.

The NVFP4 and MXFP8 reducers also accept route IDs and skip masked slots before
loading payloads, scales or scores. The standard kernel advertises masked-slot
handling for all separate reducers. These quantized-reduction changes are not
part of the pinned upstream snapshot.

FlashInfer-owned `shim/block_scaled.py` admits the geometries listed above and
omits combine-slot preclear when the kernel advertises masked reduction.
In-kernel reduction retains its accumulation reset.
This shim change is outside the exported payload. The local patch inventory
above is separate from the historical upstream qualification evidence; run the
registered strict single/multirank suites when updating or rebasing these patches.
All remaining exported implementation files retain the pinned export, apart
from the previously documented comment and the scheduler fix above.

## NVFP4 batch-invariant integration patch

Pending upstream, both kernel entry points pass an optional `batch_invariant`
problem field to their mainloop, shared FC12 epilogue and BF16 reducer. The
opt-in mainloops pin instruction N and the corresponding two-CTA input split;
GenPhase also pins the Gather4 index split. The shared epilogue interprets
FlashInfer's ordered NVFP4 activation, block16 requantization and FC2
conversion chains. The reducer starts at positive zero, retains every top-k
slot in order and masks both operands of invalid routes. These branches leave
ordinary execution unchanged. They are additional FlashInfer-owned exceptions
to the pinned export and must be retained on regeneration until upstreamed.
The shim admits only the certified NVFP4 geometry and arithmetic modes.

## MXFP4 batch-invariant integration patch

Pure MXFP4 and mixed MXFP4-weight/MXFP8-E4M3-activation inference additionally
support the opt-in batch-invariant problem field. Their shim dispatch uses
a separate arithmetic policy and receipt; NVFP4 retains its existing identity.
Both fix instruction N128 and use cluster `(2, 1)` for standard inference or
`(4, 1)` for GenPhase. Mixed inference uses canonical tile `(256, 128, 128)`
and native K64 MMA; pure MXFP4 uses `(256, 128, 256)` and native K128 MMA. The latter's
actual Rubin opcode is `kind::mxf4nvf4.block_scale.block32`. Neither MX format
has per-expert alpha or normalization operands.

The additional shared-epilogue patch fixes block32 amax as ordered block16
folds and a logical low-half-first merge on both participating warps. Explicit
UE8M0 RP conversion and round-trip, the existing capped/masked approximate
reciprocal recipe, and RN data scaling interpret the shared MXFP4
specification. The existing fixed-N/Gather4 mapping, SwiGLU, BF16 conversion
and ordered masked reduction are reused. Ordinary execution and NVFP4
branches remain unchanged. These are additional FlashInfer-owned exceptions
to retain when regenerating the pinned export until upstreamed.

Standard inference can regroup the unchanged native instruction sequence
into a larger K stage: K512 for NVFP4/pure MXFP4 and K256 for mixed MXFP4/MXFP8.
Admission requires the H and post-activation I padding extents to match the
canonical stage exactly. M256, N128 and the two-CTA instruction remain fixed;
GenPhase keeps its Gather4 stage. This changes host admission and lowering
partitions, not the native arithmetic atom or canonical receipt.

## Activation variant integration patch

The standard FC2 epilogue also forwards its BI flag to the shared canonical
BF16-to-quantized combine encoder. The reducer uses one zero-seeded, ordered
masked slot fold for quantized BI wires, preserving the ordinary quantizer
and reducer branches. Reapply until upstream absorbs this patch.

The FC12 epilogue accepts an opt-in BI ordered activation recipe from
`ProblemDesc.batch_invariant_activation`. A missing recipe keeps the existing
default helper; ordinary execution retains its original activation branch.
Reapply until upstream absorbs this integration hook.

## MXFP8 batch-invariant integration patch

The E4M3/E5M2 W8A8 paths additionally use the opt-in batch-invariant problem
field with a separate typed quantizer and arithmetic receipt. They retain
native K64 MMA, fixed `(256, 128, 128)` tiles, cluster `(2, 1)` for standard
inference or `(4, 1)` for GenPhase, and no per-expert alpha or normalization
operands. The shared epilogue dispatches block32 requantization to the MXFP8
specification with the data format's reciprocal limit. Its ordered block16
amax folds, canonical half merge, explicit UE8M0 round-trip, capped/masked
reciprocal and RN scaling retain the next-generation arithmetic chain.
Existing fixed-N/Gather4 handling, SwiGLU, BF16 conversion and ordered masked
finalization are reused. NVFP4, MXFP4 and ordinary branches keep their existing
arithmetic. Retain this additional FlashInfer-owned exception until upstreamed.

## Batch-invariant compiled identity patch

Both standard and GenPhase kernel `name()` methods append the BI mode and,
when present, a hash of the ordered activation signature's effective FP32
coefficient bits. Ordinary names remain byte-for-byte unchanged. Reapply this
host metadata patch together with the BI arithmetic hooks after regenerating
the export until upstream absorbs it.

## Related trees

The FlashInfer host shim prepares kernel metadata, routing indices, scalar
buffers, local output and scratch before entering the symmetric heap. All EP
ranks exchange local allocation failures through their process group first.
Tuning winner installation likewise prepares replacement local storage before
freeing any existing symmetric owner; trial staging and compilation agree on
failure before collective execution. These checks run only at collective
preparation boundaries, without adding host collectives to steady-state or
captured launches. Retain this shim integration when refreshing the export.

`sm100/cutedsl_megamoe` and the SM90 drop are separate snapshots. This Rubin
drop comes from upstream's `next/` generation, with relative imports and the
`api.py` component model; it does not replace the older SM100 packages.

## Consumers

- `backends/mega/kernel/sm107/nvfp4_nvfp4_bf16_cutedsl/`
- `backends/mega/kernel/sm107/mxfp8_mxfp8_bf16_cutedsl/`
- `backends/mega/kernel/sm107/mxfp8_mxfp4_bf16_cutedsl/`
