# MoE provider adapters

Adapters are grouped by SM capability, then implementation. They translate the
framework interface into provider calls and own tensor preparation, temporary
and symmetric workspaces, compilation, tuning and graph lifetime handling.
Kernel sources live under `../kernels/` and retain their independent update
records.

Blackwell uses `sm100/cutedsl_megamoe/sources/` for standard and GenPhase
and `sm100/cutedsl_megamoe/pre_unification/` for retained NVFP4 and BF16.
Rubin uses `sm107/cutedsl_megamoe/`; Hopper uses `sm90/cutedsl_megamoe/`.
SM120's branch implementations use sibling `split_cutedsl/` and
`swapab_cutedsl/` adapters. Implementation-specific helpers stay with their
adapter; backends retain SM, datatype and provider selection.

Backends import adapter package exports. Provider imports and any source-path setup stay inside the adapter. The old `kernel_src/` and `cute_dsl/megamoe/`
packages have been removed. Direct imports and `python -m` commands must use the
paths listed in `docs/design_docs/moe_ep_architecture.md`.
