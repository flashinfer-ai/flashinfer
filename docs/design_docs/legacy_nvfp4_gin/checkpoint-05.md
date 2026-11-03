# Blackwell checkpoint: all off gates pass with controlled worker placement

The final off matrix passes all 24 predeclared 5% performance gates. All 48
source/case/cache/scope groups have 100 samples; exact outputs, per-rank payload
identity and all profile event counts pass. EP1 uses job3578010; unresolved
EP2/EP4 cases use GPU-local worker placement in3578087/3578088. No next provider
is selected. Each source pair runs on the same exclusive GB300 allocation.

The EP4 baseline itself provides a controlled placement check. On theia0095,
job3578088 interleaves unchanged legacy F_BASE with two placements over five
rounds. Workers share CPUs0-31 in one placement; the other uses disjoint local
sets0-7,8-15,72-79,80-87. Both worker layouts cover32 distinct physical CPUs;
the allocation and parent-process resources are held constant. Bindings precede
Torch/CUDA context creation and NVML/actual affinity sidecars confirm the sets.

| Fixed F_BASE case | Shared32 raw median / round CV | GPU-local raw median / round CV |
| --- | ---: | ---: |
| T64 warm | 151.62 us / 6.31% | 129.73 us / 2.42% |
| T64 cold | 140.13 us / 16.85% | 130.90 us / 3.09% |
| T1024 warm | 207.44 us / 88.38% | 184.62 us / 0.97% |
| T1024 cold | 205.71 us / 40.03% | 186.27 us / 2.01% |

This supports worker placement as a causal contributor to instability under
this launcher. It does not isolate NUMA locality from per-rank CPU partitioning
or prove a particular driver-internal allocation defect. Old same-host traces
also show GPU starts separated by6.3 ms with ends separated by0.1 us: those
profiled kernel spans include waiting for a late peer. Their duration is not
an isolated arithmetic cost. Earlier noisy samples remain preserved.

The code corrections remain separate from placement: compact CPU agreement,
reused session-bound launch arguments, merged mandatory validation, and the
fixed donor's existing lean non-GIN compile entry. Persistent finalization and
FC1 per-token scaling retain their original entry. No device arithmetic,
communication protocol, support check, tolerance or tuning policy was weakened.

Final host candidate38f650e63 already passes15 strong/public tests per rank in
3578050. Its unbound standard public full-call median is1513.24 us versus a
native donor adapter's387.31 us; the adapters perform different validation.
The paired earlier host correction measured6155.76→2208.60 us in3577960.
Before/after device PTX and cubins in that experiment match on all four ranks.
These full-call improvements are not claims of a corresponding kernel speedup.

Final N2R2 GPU-local confirmation3578114 is still running/pending; its performance
conclusion is not yet available. Existing unbound raw medians changed direction
across allocations, so no unconditional intrinsic-kernel regression was established.

Evidence root:
`/lustre/fsw/coreai_devtech_all/qiangf/ncclGIN/port/migration-evidence/blackwell-fix-20261102/`.
Placement sidecars: `affinity-final/ep4/{shared32,pinned}/affinity/`; primary
samples/manifests and deferred traces remain alongside. Full off statistics
are currently generated under `off-final-v1/ep12/summary.json` with explicit
EP2/EP4 root overrides. The final report will bind the accepted roots/hashes.
