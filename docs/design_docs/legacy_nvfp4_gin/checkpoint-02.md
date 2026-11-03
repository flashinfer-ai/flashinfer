# Checkpoint R2: completed fixed-scope migration

2026-11-02. Implementation commits: `fb033d522`, `79a519e20`, branch
`port/legacy-nvfp4-nccl-gin`. Plan A remains in effect: legacy NVFP4 GIN only,
default-off, fixed N2R2 geometry; FI off behavior retained.

Final whole-job passes: 3576874 (14 tests/rank and example), 3576878
(finalizers/cache/graphs). Every rank has 24 exact protocol calls, 96 total
message/L1 byte checks pass, and independent math max relative L2 is
0.0026485608 under the unchanged 0.02 limit. No selected GPU case was skipped.
Off EP1/EP2/EP4, BF16/MXFP8 compatibility and public error/lifetime results
are detailed in [validation](validation.md).

The generation race and local dtype-validation escape were corrected from
measured failures. Earlier failed jobs remain visible in
[investigation](investigation.md). Source/log hashes and per-rank manifests
are in [qualification-results.json](evidence/qualification-results.json).

The user explicitly waived Rubin/local performance validation. No timing
claim is made. Original checkouts are unchanged; nothing was pushed, merged
or published. Final review material is in [review-notes.md](review-notes.md).
