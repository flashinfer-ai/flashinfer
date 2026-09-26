# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Instantiate frozen Frost source templates without importing cuDNN.

Geometry stays compile-time constant. The exporter checks AST equality against
Frost's concrete source, including its geometry-unrolled TMA store epilogue.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import tempfile
from pathlib import Path

_CONSTANTS = "# @@FROST_GEOMETRY@@"
_STORE = "# @@FROST_TMA_STORE@@"
_SYMBOL = "frost_template_kernel"


def _constant_expression(expression: str) -> None:
    node = ast.parse(expression, mode="eval").body
    try:
        ast.literal_eval(node)
        return
    except (ValueError, TypeError):
        pass
    # Dtype and enum constants emitted by Frost, never arbitrary calls/code.
    while isinstance(node, ast.Attribute) and not node.attr.startswith("_"):
        node = node.value
    if isinstance(node, ast.Name) and node.id in ("cutlass", "nvvm", "_tma"):
        return
    raise ValueError(f"unsupported Frost template constant: {expression!r}")


def _bf16_tma_store(constants: dict[str, str], swap: bool) -> str:
    """Reproduce Frost's single BF16 output staging, with immediate offsets.

    This is host-side source rendering, not a helper called inside a CuTe
    kernel. Preserve the producer's unrolling and barrier placement exactly.
    """

    # Adapted from cuDNN Frontend's sm100/compiler.py TMA staging renderer:
    # Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. SPDX-License-Identifier: MIT.
    def value(name):
        return ast.literal_eval(constants[name])

    n = value("epi_n")
    rows = value("epi_tile_mn")[0]
    # Block-scale templates use full epilogue warps and omit these dense-only
    # constants. extract_template still verifies the reconstructed source AST.
    packed = ast.literal_eval(constants.get("epi_packed_lanes", "False"))
    dp22 = ast.literal_eval(constants.get("epi_dp22", "False"))
    row_elems = value("epi_row_elems")
    if constants["epi_store_dtype"] != "cutlass.BFloat16":
        raise ValueError("TMA source templates currently support BF16 output only")
    desc = (
        "tma_c_descs[0].get_ptr()"
        if value("moe_aligned_offsets")
        else "d_desc_ptr_list[0]"
    )
    lines = [
        "epi_stage_idx = (epi_stage_idx + 1) % EPI_SMEM_STAGES",
        f"_tsv_0 = cutlass.Array(base=smem_d_ptr.data_ptr(epi_stage_idx * epi_subtile_elems), shape={value('epi_stage_rows') * row_elems}, dtype=cutlass.BFloat16)",
    ]

    def issue(ptr, coord):
        lines.extend(
            [
                "if warp_idx == 0:",
                "    if elect_one:",
                "        nvvm.cp_async_bulk_tensor_global_shared_cta(",
                f"            {desc},",
                f"            {ptr},",
                f"            {coord},",
                "        )",
            ]
        )

    slot = "(row - coord_m)" if packed else "tidx"
    if swap:
        lines += [f"_mrow_0 = {slot} % 64", f"_mblk_0 = ({slot} // 64) * {64 * n}"]
        lines += [f"_mx{x}_0 = _mrow_0 ^ {x * 8}" for x in range(8)]
        for column in range(n):
            statement = f"_tsv_0.data_ptr(_mblk_0 + {column * 64} + _mx{column % 8}_0).store(vec_out[{column} : {column + 1}], alignment=2)"
            lines.append(f"if row_active:\n    {statement}" if packed else statement)
    else:
        row_bytes = row_elems * 2
        swizzle = {32: 1, 64: 2, 128: 3}[row_bytes]
        statement = f"_tsv_0.data_ptr({slot} * {row_elems}).store_swizzled(vec_out, alignment={row_bytes}, swizzle=cutlass.Swizzle({swizzle}, 4, 3))"
        lines.append(f"if row_active:\n    {statement}" if packed else statement)
    lines += [
        "cute.arch.fence_view_async_shared()",
        "nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)",
    ]
    if swap:
        for block in range(rows // 64):
            issue(
                f"_tsv_0.data_ptr({block * 64 * n})", f"(coord_m + {block * 64}, col)"
            )
        if dp22:
            issue(f"_tsv_0.data_ptr({64 * n})", "(coord_m, col + epi_cols_per_mma_m)")
    else:
        issue("_tsv_0.data_ptr()", "(col, coord_m)")
        if dp22:
            issue(
                f"_tsv_0.data_ptr({rows * row_elems})",
                "(col + epi_cols_per_mma_m, coord_m)",
            )
    lines += [
        "    if elect_one:",
        "        nvvm.cp_async_bulk_commit_group()",
        "    nvvm.cp_async_bulk_wait_group(EPI_SMEM_STAGES - 1, read=True)",
        "nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)",
    ]
    return "\n".join(lines)


def render_source(template: str, parameters: dict) -> str:
    if parameters.get("version") != 1 or template.count(_CONSTANTS) != 1:
        raise ValueError("unsupported Frost source template")
    assignments = parameters["constants"]
    lines = []
    for name, expression in assignments:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("invalid Frost template constant name")
        _constant_expression(expression)
        lines.append(f"{name} = {expression}")
    source = template.replace(_CONSTANTS, "\n".join(lines))
    symbol = parameters["kernel_symbol"]
    if not isinstance(symbol, str) or not symbol.isidentifier():
        raise ValueError("invalid Frost kernel symbol")
    source = re.sub(rf"\b{_SYMBOL}\b", symbol, source)
    source = source.replace("FROST_EPI_N", dict(assignments)["epi_n"])
    if "FROST_EPI_ROW_ELEMS" in source:
        row_elems = ast.literal_eval(dict(assignments)["epi_row_elems"])
        source = source.replace("FROST_EPI_ROW_ELEMS", str(row_elems))
        source = source.replace(
            "FROST_EPI_SWIZZLE", f"_tma.TensorMapSwizzle.s{row_elems * 2}b"
        )
    if _STORE in source:
        store = _bf16_tma_store(dict(assignments), parameters["swap_ab"])
        source = re.sub(
            r"(?m)^([ \t]*)" + re.escape(_STORE) + r"$",
            lambda match: "\n".join(match[1] + line for line in store.splitlines()),
            source,
        )
    return source


def extract_template(source: str, *, swap_ab: bool) -> tuple[str, dict]:
    """Factor a concrete source; refuse any non-equivalent reconstruction."""
    header = (
        "# Block-scale config:"
        if "# Block-scale config:" in source
        else "# Tile config:"
    )
    begin = source.index(header)
    end = source.find("# Tensormap workspace slots", begin)
    if end < 0:
        # The block-scale swap-AB template starts its workspace formulas
        # directly, without the dense/normal template's section comment.
        end = source.index("moe_desc_slots =", begin)
    constants = []
    for statement in ast.parse(source[begin:end]).body:
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            raise ValueError("unexpected Frost tile constant statement")
        constants.append([statement.targets[0].id, ast.unparse(statement.value)])
    template = source[:begin] + _CONSTANTS + "\n\n" + source[end:]
    symbols = set(re.findall(r"\bfrost_sm100_\w+(?=\(|\.set_name_prefix)", template))
    if len(symbols) != 1:
        raise ValueError("expected one Frost grouped GEMM kernel symbol")
    symbol = symbols.pop()
    template = re.sub(rf"\b{symbol}\b", _SYMBOL, template)
    if swap_ab:
        template = re.sub(
            r"box_dims=\[64, \d+\]", "box_dims=[64, FROST_EPI_N]", template
        )
    else:
        template = re.sub(
            r"box_dims=\[\d+, epi_tile_mn\[0\]\],\n([ \t]*)"
            r"swizzle=_tma\.TensorMapSwizzle\.s(?:32|64|128)b,",
            r"box_dims=[FROST_EPI_ROW_ELEMS, epi_tile_mn[0]],\n\1"
            r"swizzle=FROST_EPI_SWIZZLE,",
            template,
        )
    # The generated code has one output and therefore one TMA staging block.
    pattern = (
        r"(?m)^([ \t]*)epi_stage_idx = \(epi_stage_idx \+ 1\) % EPI_SMEM_STAGES\n"
        r".*?^\1    nvvm\.cp_async_bulk_wait_group\(EPI_SMEM_STAGES - 1, read=True\)\n"
        r"^\1nvvm\.barrier_cta_sync\(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps \* 32\)"
    )
    template, count = re.subn(pattern, rf"\g<1>{_STORE}", template, flags=re.S)
    if count != int(dict(constants)["n_tma_outputs"]):
        raise ValueError("unsupported Frost TMA store layout")
    parameters = dict(
        version=1, constants=constants, kernel_symbol=symbol, swap_ab=swap_ab
    )
    reconstructed = render_source(template, parameters)
    if ast.dump(ast.parse(reconstructed)) != ast.dump(ast.parse(source)):
        raise ValueError("Frost template reconstruction changed the source AST")
    return template, parameters


def materialize_source(template_path: Path, parameters: dict) -> tuple[Path, str]:
    """Write only to FlashInfer's generated-source cache, never the package."""
    from ...jit.env import FLASHINFER_GEN_SRC_DIR

    source = render_source(template_path.read_text(), parameters)
    digest = hashlib.sha256(source.encode()).hexdigest()
    directory = FLASHINFER_GEN_SRC_DIR / "cudnn_frost_templates" / digest
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "kernel.py"
    if not path.exists() or path.read_text() != source:
        fd, temporary = tempfile.mkstemp(dir=directory, suffix=".py.tmp")
        try:
            with os.fdopen(fd, "w") as file:
                file.write(source)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return path, digest
