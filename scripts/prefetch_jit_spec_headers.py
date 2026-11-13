#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Warm pinned TRT-LLM spec headers before parallel provider generation.

Only manifests and headers are fetched; no kernel binaries are downloaded and
no CUDA work is launched. Use the same artifact hashes and cache as generators.
"""


def prefetch_group(prefix: str, manifest_hash: str, headers: list[str]) -> int:
    from flashinfer.jit.cubin_loader import get_artifact, get_meta_hash

    manifest_path = f"{prefix.rstrip('/')}/checksums.txt"
    manifest = get_artifact(manifest_path, manifest_hash)
    if not manifest:
        raise RuntimeError(f"Failed to prefetch {manifest_path}")
    for header in headers:
        path = f"{prefix.rstrip('/')}/include/{header}"
        if not get_artifact(path, get_meta_hash(manifest, header)):
            raise RuntimeError(f"Failed to prefetch {path}")
    return len(headers)


def main() -> None:
    from flashinfer.artifacts import ArtifactPath, CheckSumHash
    from flashinfer.jit.fused_moe import BMM_EXPORT_HEADERS
    from flashinfer.jit.gemm.core import GEMM_EXPORT_HEADERS

    groups = [
        (
            ArtifactPath.TRTLLM_GEN_FMHA,
            CheckSumHash.TRTLLM_GEN_FMHA,
            ["flashInferMetaInfo.h"],
        ),
        (
            ArtifactPath.TRTLLM_GEN_GEMM,
            CheckSumHash.TRTLLM_GEN_GEMM,
            ["flashinferMetaInfo.h"]
            + [f"trtllmGen_gemm_export/{h}" for h in GEMM_EXPORT_HEADERS],
        ),
        (
            ArtifactPath.TRTLLM_GEN_BMM,
            CheckSumHash.TRTLLM_GEN_BMM,
            ["flashinferMetaInfo.h"]
            + [f"trtllmGen_bmm_export/{h}" for h in BMM_EXPORT_HEADERS],
        ),
    ]
    count = sum(prefetch_group(*group) for group in groups)
    print(f"Prefetched {count} provider spec headers")


if __name__ == "__main__":
    main()
