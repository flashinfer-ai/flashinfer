# GIN migration checkpoint 1

Snapshot of 2026-11-02; the [investigation](investigation.md) tracks later changes.

The live GIN runtime now passes actual N2R2 all-peer device reads and three
create/close cycles on every rank. The corrected mapper preserves NCCL's
self-address alias instead of requiring the original VA. F_BASE and the
migrated off path each pass the same two EP4 smoke cases, including per-token
scale, on all four ranks (job 3576310).

The first FI GIN launch failed (3576298). Source comparison found an omitted
donor argument, `publish_relay_counts_to_gemm=True`, which controls scheduled
counts and merged segment layout. After restoring that contract, job 3576341
passed 12 balanced and 12 local/remote/mixed/partial/empty-rank forwards per
rank. Fixed donor local-only routing also passed exact stage and final checks
(3576340). A later FI rejection test failed because unsupported per-token
scale was rejected too late for the constructor-level test; that validation
is being moved into the collective backend initialization preflight.

These are bounded results: the FI relaunch cases have checked shape and finite
output so far. Independent fixed-FC2-term protocol equality and Torch math
oracles remain pending, as do broader EP1/EP2/EP4/shared-caller regressions and
performance qualification. No migration-complete or performance claim is made.

Raw launchers and full-rank logs are under `port/migration-evidence/`.
