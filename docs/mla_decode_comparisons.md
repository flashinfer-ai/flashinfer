# MLA decode comparison scopes

Kernel comparisons should identify the implementation, adapter, dispatch route,
and timed boundary. The following definitions describe PR #6177 at
`4090e370e4d1a82750ef42ad3d21e172f3a1e44f` and the follow-up in this branch.

| Name | What it refers to |
| --- | --- |
| Existing FI monolithic | The pre-existing M128 MLA family, with its own split and reduction choices. It remains available alongside the new MTP family. |
| #6177 MTP | The Rubin M256 preferred/fallback pipeline. Its original integration admits H128 with Q2/Q4. |
| Adapted MTP baseline | The new MTP kernel plus an adapter for H96/Q8: pad heads to H128, fold Q8 into two Q4 requests, and include packing/unpacking in timing. |
| FI with #6177 | The library with both families and its actual routing rules. This is distinct from forcing the adapted MTP kernel. |
| This follow-up | Internal native H96/H128, Q2/Q4/Q8 candidates, bounded reduction options, and a monolithic correction fix. Native candidates are not registered with public auto/autotune. |

In [LightSeek's MLA 2 article](https://lightseek.org/blog/tokenspeed-mla-2),
the FlashInfer comparison uses the **adapted MTP baseline**. It does not use the
existing FI monolithic kernel as that baseline or measure FI selecting among
all available implementations. The article's separate **MLA1** control is
TokenSpeed's previous implementation recompiled for Rubin, not FI monolithic.
A result against adapted MTP therefore does not establish a result against FI's
existing default route. Different outcomes for those comparisons are compatible.

#6177 supplies the pipeline on which both TS MLA2 and this follow-up build.
A grid outside its original H128 contract does not directly measure the benefit
of that original contract. Native row packing here follows the direction
described by LightSeek, with implementation changes in FI's preferred/fallback
paths, causal coordinates, workspace bounds and reducers.

For a before/after comparison, use separate pinned snapshots of FI before
#6177, FI with #6177, and FI with the follow-up. Identify whether each result
uses the functional entrypoint, planned wrapper, or explicitly selected kernel.
The functional entrypoint and planned wrapper can differ in supported output
dtypes. Preserve each snapshot's defaults when making a default-route claim;
forcing persistent scheduling or an explicit split count changes that scope.
Report manually selected candidates separately from automatic dispatch.

This branch exposes the native factory only as an internal candidate. Its
geometry is bounded to H96/H128, Q2/Q4/Q8 and at most 32 splits. Validation and
workspace preparation precede execution; execution binds the live tensors.
Valid page IDs, live lengths and non-overlapping storage remain caller
contracts. The FP16-softmax research option is unqualified. Production dispatch
admission requires further workload and supported target-sanitizer coverage.
No benchmark numbers or claim about production routing are supplied here.
