# MoE kernel sources

This directory holds kernel implementations and their vendoring records.
Hardware, dtype and implementation selection remain under
`backends/mega/kernel/<sm>/<dtype_backend>/`. Adapters that wrap a separate
kernel API live under `shims/<sm>/<implementation>/`.

```text
kernels/
├── cutedsl_megamoe/
│   ├── sources/
│   │   ├── helpers/
│   │   ├── communication/
│   │   └── kernel_src/        # Blackwell and Rubin standard/GenPhase kernels
│   └── pre_unification/
│       ├── common/
│       ├── comm/
│       ├── moe_bf16_glu/
│       ├── moe_mxfp8_bf16_glu/
│       ├── moe_nvfp4_swapab/
│       ├── moe_hopper_bf16/
│       └── moe_hopper_fp8/
├── kimi_k3/
├── cake/
│   ├── bf16_megamoe_sm90/
│   └── mxfp4_megamoe_sm120/
└── flashinfer/
    ├── bf16_nvfp4/
    ├── bf16_rank_major/
    ├── push_style_megamoe/
    ├── sm120_split/
    └── sm120_swapab/
```

The provider's `sources/` and `pre_unification/` trees remain independent.
Blackwell and Rubin share the exported `sources/` dependencies.
Hopper uses the same canonical `pre_unification/` tree, including shared
legacy communication and reducer modules. The two separately maintained
SM120 drops have their own pins.
The SM120 drops stay under `flashinfer/`; their records preserve their source
revisions and attribution. Directory placement does not imply support for every
SM in an architecture family; each backend checks its supported capabilities.

For SM100, the registered backend selects the following implementation. Paths
in the last two columns are relative to `shims/sm100/` and `kernels/`.

| Implementation | Adapter | Kernel source |
|---|---|---|
| Block-scaled standard and GenPhase | `cutedsl_megamoe/sources/` | `cutedsl_megamoe/sources/kernel_src/blackwell/` |
| Retained NVFP4 implementation (`_legacy`) | `cutedsl_megamoe/pre_unification/` | `cutedsl_megamoe/pre_unification/moe_nvfp4_swapab/` |
| BF16/BF16 and BF16/MXFP8 | `cutedsl_megamoe/pre_unification/` | `cutedsl_megamoe/pre_unification/moe_{bf16,mxfp8_bf16}_glu/` |
| BF16/NVFP4 | Backend calls the implementation's frontend | `flashinfer/bf16_nvfp4/` |
| BF16 rank-major CUDA | Backend calls the implementation's session | `flashinfer/bf16_rank_major/` |
| Kimi K3 | `kimi_k3/` | `kimi_k3/` |
| DeepGEMM | Backend calls the installed package | Installed `deep_gemm` package |

Independent adapters are siblings under their SM directory. A backend that
already calls its implementation directly does not need another wrapper layer.
The `_cutedsl` suffix identifies the implementation language, not source
ownership; vendoring records identify the source of each drop.

Each provider export keeps its vendoring record beside the sources. Preserve the
recorded source bytes, generated initializers, manifests and release transforms.
Make kernel changes at their source and regenerate the affected export. Do not
format provider code or edit a generated kernel independently of its manifest.
An export update must include dependencies used directly by its FI shim as well
as those imported by the provider entry points.

The SM120 drops retain their raw package names, including the communication
package named `src`, and their existing process-isolation checks. The
combined Blackwell, Rubin and Hopper export uses package-relative imports.
Its shared VENDOR.md and export_manifest.json record component revisions
and shared dependencies; tuning notes live beside each SM adapter.

DeepGEMM remains an installed dependency; its adapter stays in the backend tree.
The `csrc/` build resources and experimental kernel packages are unchanged.
