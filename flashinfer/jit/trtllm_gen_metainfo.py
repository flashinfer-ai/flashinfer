"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Architecture filtering for the trtllm-gen ``flashinferMetaInfo.h`` manifests.

The trtllm-gen BMM/GEMM packages are single-package multi-architecture: one
publish carries the Blackwell (sm100a/sm100f/sm103a) and the Rubin (sm107a)
kernels, and the corresponding ``flashinferMetaInfo.h`` lists every one of them
as a ``BatchedGemmConfig`` / ``GemmConfig`` aggregate initializer.

``BatchedGemmConfig`` embeds ``BatchedGemmOptions``, which has non-trivial
members (``std::string``, ``std::vector<int>``), so the manifest array cannot be
constant-folded into ``.rodata``: the compiler emits a dynamic initializer for
every entry.  That makes compile time and peak compiler RSS scale (superlinearly)
with the number of entries, and the multi-arch manifest doubled it -- ~157 s and
~3.8 GB for a single TU including ``BatchedGemmInterface.h``, versus ~60 s and
~2.0 GB for a single-architecture manifest.

The runtime never dispatches across architecture families
(``isArchCompatible()`` in ``csrc/trtllm_batched_gemm_runner.cu`` and
``csrc/trtllm_gemm_runner.cu``), so the entries for the other family are dead
weight in a given module.  We therefore strip them before compiling.

The filter is keyed on the *module variant* (Blackwell vs Rubin), not on the
GPU visible at build time.  AOT builds (``flashinfer-jit-cache``) generate both
variants on one machine, and a device-derived filter would silently bake a
Blackwell manifest into the Rubin module.
"""

import pathlib
import re
from typing import Collection, Tuple

# Cubin architectures the *Blackwell* trtllm-gen module can dispatch to.
# Mirrors isArchCompatible() for smVersion in {100, 103}.
BLACKWELL_CUBIN_ARCHS: Tuple[str, ...] = ("Sm100a", "Sm100f", "Sm103a")

# Cubin architectures the *Rubin* trtllm-gen module can dispatch to.
# Mirrors isArchCompatible() for smVersion == 107, which accepts only Sm107a.
# That is a dispatch policy, not a hardware limit: sm100f cubins do load and
# execute on Rubin (cuModuleLoadData succeeds for sm_100f on cc 10.7 and fails
# with CUDA_ERROR_NO_BINARY_FOR_GPU for sm_100a/sm_103a).  Keep this tuple in
# sync with isArchCompatible() -- listing an arch the runtime filter then
# discards yields a module carrying cubins it will never dispatch to.
# trtllm-gen FMHA makes the opposite policy choice: isSMCompatible() in
# include/flashinfer/trtllm/fmha/fmhaKernels.cuh does accept kSM_100f on
# kSM_107.
RUBIN_CUBIN_ARCHS: Tuple[str, ...] = ("Sm107a",)

# An entry starts at column 0 with the four leading POD members of
# BatchedGemmConfig/GemmConfig (mData, mSize, mSharedMemSize, ...).
_ENTRY_START = re.compile(r"^\{nullptr, 0, ")

# ... and ends with the trailing mSm member, which names the cubin architecture.
_ENTRY_END = re.compile(r"^\s*\}, gemm::SmVersion::(Sm\w+)\},\s*$")

# The declared length must be kept in sync with the array we emit.
_LIST_LEN = re.compile(r"^(\s*static constexpr size_t \w*ListLen = )\d+(;.*)$")

# Start of the manifest array; entries before this line are not filtered.
_LIST_START = re.compile(
    r"^static const \S+ (tllmGenBatchedGemmList|tllmGenGemmList)\[\] = \{"
)


class MetaInfoFilterError(RuntimeError):
    """Raised when flashinferMetaInfo.h does not have the expected shape."""


def filter_metainfo(source: str, keep_archs: Collection[str]) -> Tuple[str, int, int]:
    """Drop manifest entries whose ``mSm`` is not in *keep_archs*.

    Returns ``(filtered_source, num_kept, num_dropped)``.

    Raises :class:`MetaInfoFilterError` if the manifest does not parse as
    expected, so that a future package layout change fails loudly instead of
    silently emitting an empty or truncated kernel list.
    """
    keep = set(keep_archs)
    out = []
    entry: list = []
    in_list = False
    kept = dropped = 0
    seen_archs = set()

    for line in source.split("\n"):
        if entry:
            entry.append(line)
            match = _ENTRY_END.match(line)
            if match:
                arch = match.group(1)
                seen_archs.add(arch)
                if arch in keep:
                    out.extend(entry)
                    kept += 1
                else:
                    dropped += 1
                entry = []
            continue

        if in_list and _ENTRY_START.match(line):
            entry = [line]
            continue

        out.append(line)
        if _LIST_START.match(line):
            in_list = True

    if entry:
        raise MetaInfoFilterError(
            "flashinferMetaInfo.h has an unterminated kernel entry; "
            "the manifest layout changed and the arch filter needs updating."
        )
    if not in_list:
        raise MetaInfoFilterError(
            "flashinferMetaInfo.h has no recognizable kernel-list declaration; "
            "the manifest layout changed and the arch filter needs updating."
        )
    if kept == 0:
        raise MetaInfoFilterError(
            f"flashinferMetaInfo.h arch filter kept 0 of {dropped} kernels "
            f"(wanted {sorted(keep)}, manifest has {sorted(seen_archs)})."
        )

    # Rewrite the declared length so it matches the array we just emitted.
    fixed_len = False
    for i, line in enumerate(out):
        match = _LIST_LEN.match(line)
        if match:
            out[i] = f"{match.group(1)}{kept}{match.group(2)}"
            fixed_len = True
            break
    if not fixed_len:
        raise MetaInfoFilterError(
            "flashinferMetaInfo.h has no ...ListLen declaration to update; "
            "the manifest layout changed and the arch filter needs updating."
        )

    return "\n".join(out), kept, dropped


def write_filtered_metainfo(
    dest: pathlib.Path,
    source: bytes,
    keep_archs: Collection[str],
) -> Tuple[int, int]:
    """Filter *source* and write it to *dest* if the content changed.

    Writing only on change keeps ninja from rebuilding the module on every
    ``gen_*`` call.
    """
    from .core import logger
    from .utils import write_if_different

    filtered, kept, dropped = filter_metainfo(source.decode("utf-8"), keep_archs)
    write_if_different(dest, filtered)
    logger.debug(
        f"flashinferMetaInfo.h arch filter: kept {kept}, dropped {dropped} "
        f"(keeping {sorted(set(keep_archs))}) -> {dest}"
    )
    return kept, dropped
