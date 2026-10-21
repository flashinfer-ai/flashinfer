#!/usr/bin/env python3
"""Encode FP4 expert scale bytes as csf_32x2_v1 using NumPy and safetensors.

Usage: python encode_csf.py SOURCE OUTPUT --quantization mxfp4 --workers 8

Packed weights and other tensors are copied verbatim. Each encoded matrix is
decoded and compared with its source; written tensors are reread and compared.
The output directory must not exist and appears only after all shards pass.
No PyTorch, CUDA, FlashInfer, or external compiler is required.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json
import mmap
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
import time

import numpy as np
from safetensors import safe_open

ENCODING = "csf_32x2_v1"
COMPONENTS = ("bases", "codes", "page_bases", "offsets", "exceptions")
SCALE_PATTERN = r"\.experts\.\d+\.[^.]+\.(?:weight_scale|scale)$"


@lru_cache(maxsize=16)
def _geometry(rows, columns):
    if rows <= 0 or rows % 32 or columns <= 0:
        raise ValueError("CSF requires positive 32-row extents and positive columns")
    grid = np.arange(rows // 32 * ((columns + 1) // 2)).reshape(rows // 32, -1)
    groups = [
        grid[r : r + 4, c : c + 2].T.ravel()
        for r in range(0, grid.shape[0], 4)
        for c in range(0, grid.shape[1], 2)
    ]
    blocks = np.concatenate(groups)
    first = (blocks // grid.shape[1]) * 32 * columns + blocks % grid.shape[1] * 2
    width = np.minimum(2, columns - first % columns)
    local = np.arange(64)
    positions = first[:, None] + local // 2 * columns + local % 2
    order = positions[local % 2 < width[:, None]]
    starts = np.concatenate(([0], np.cumsum(32 * width)))
    group_offsets = np.concatenate(([0], np.cumsum([len(g) for g in groups])))
    pages = np.concatenate((group_offsets[:-1:32], group_offsets[-1:]))
    return order, starts, pages, first


def encode_scales(source, quantization):
    """Compress a logical uint8 scale matrix, choosing the best row palette."""
    if source.dtype != np.uint8 or source.ndim != 2:
        raise ValueError("Scales must be a two-dimensional uint8 array")
    bits = {"mxfp4": 1, "nvfp4": 4}[quantization]
    rows, columns = source.shape
    order, starts, pages, _ = _geometry(rows, columns)
    indexed = (np.arange(rows, dtype=np.int64)[:, None] << 8) + source
    histogram = np.bincount(indexed.ravel(), minlength=rows * 256).reshape(rows, 256)
    width = 1 << bits
    cumulative = np.pad(np.cumsum(histogram, axis=1), ((0, 0), (1, 0)))
    bases = np.argmax(cumulative[:, width:] - cumulative[:, :-width], axis=1)
    delta = source.astype(np.int16) - bases[:, None]
    outside = (delta < 0) | (delta >= width)
    delta[outside] = 0
    fields = delta.ravel()[order].astype(np.uint8)
    codes = (
        np.packbits(fields, bitorder="little")
        if bits == 1
        else fields[::2] | (fields[1::2] << 4)
    )
    exceptional = outside.ravel()[order]
    positions = order[exceptional]
    records = (positions // columns % 32 * 2 + positions % columns % 2).astype(
        np.uint16
    ) | (source.ravel()[positions].astype(np.uint16) << 8)
    counts = np.add.reduceat(exceptional, starts[:-1], dtype=np.int64)
    prefix = np.concatenate(([0], np.cumsum(counts)))
    if records.nbytes >= 1 << 32:
        raise ValueError("CSF exception arena must be smaller than 4 GiB")
    result = dict(
        bases=bases.astype(np.uint8),
        codes=codes,
        page_bases=(2 * prefix[pages[:-1]]).astype("<u4"),
        offsets=np.concatenate(
            [
                prefix[a : b + 1] - prefix[a]
                for a, b in zip(pages[:-1], pages[1:], strict=True)
            ]
        ).astype("<u2"),
        exceptions=records.astype("<u2"),
    )
    np.testing.assert_array_equal(
        decode_scales(result, source.shape, quantization), source
    )
    return result


def decode_scales(parts, shape, quantization):
    """Reconstruct logical scale bytes from the five checkpoint components."""
    rows, columns = shape
    order, _, pages, first = _geometry(rows, columns)
    codes = parts["codes"]
    if quantization == "mxfp4":
        fields = np.unpackbits(codes, bitorder="little")
    else:
        fields = np.stack((codes & 15, codes >> 4), axis=1).ravel()
    output = np.empty(rows * columns, dtype=np.uint8)
    output[order] = parts["bases"][order // columns] + fields
    for page, (a, b) in enumerate(zip(pages[:-1], pages[1:], strict=True)):
        offsets = parts["offsets"][a + page : b + page + 1].astype(np.int64)
        begin = int(parts["page_bases"][page]) // 2
        records = parts["exceptions"][begin : begin + offsets[-1]]
        blocks = np.repeat(first[a:b], np.diff(offsets))
        local = (records & 255).astype(np.int64)
        output[blocks + local // 2 * columns + local % 2] = records >> 8
    return output.reshape(shape)


def _header(path):
    with open(path, "rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        return json.loads(stream.read(size)), size + 8


def _array(mapping, origin, info):
    return np.ndarray(
        info["shape"],
        dtype={"U8": "u1", "U16": "<u2", "U32": "<u4", "F8_E4M3": "u1"}[info["dtype"]],
        buffer=mapping,
        offset=origin + info["data_offsets"][0],
    )


def _encode_shard(source, destination, filename, selected, quantization):
    original, origin = _header(source / filename)
    arrays, output_header = {}, {}
    scale_bytes = compressed_bytes = 0
    with (
        open(source / filename, "rb") as stream,
        mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapping,
    ):
        for name in selected:
            values = _array(mapping, origin, original[name])
            parts = encode_scales(values, quantization)
            scale_bytes += values.nbytes
            compressed_bytes += sum(p.nbytes for p in parts.values())
            for component, value in parts.items():
                key = f"{name}.csf32_{component}"
                arrays[key] = value
                output_header[key] = dict(
                    dtype={1: "U8", 2: "U16", 4: "U32"}[value.itemsize],
                    shape=list(value.shape),
                    data_offsets=[0, value.nbytes],
                )
        values = None
    output_header.update(
        (name, info)
        for name, info in original.items()
        if name != "__metadata__" and name not in selected
    )
    widths = {
        "F64": 8,
        "I64": 8,
        "U64": 8,
        "F32": 4,
        "I32": 4,
        "U32": 4,
        "F16": 2,
        "BF16": 2,
        "I16": 2,
        "U16": 2,
    }
    names = sorted(
        output_header, key=lambda n: (-widths.get(output_header[n]["dtype"], 1), n)
    )
    ordered = (
        {"__metadata__": original["__metadata__"]} if "__metadata__" in original else {}
    )
    cursor = 0
    for name in names:
        info = output_header[name]
        size = info["data_offsets"][1] - info["data_offsets"][0]
        ordered[name] = dict(info, data_offsets=[cursor, cursor + size])
        cursor += size
    encoded = json.dumps(ordered, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    target = destination / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(source / filename, "rb") as src, open(target, "wb") as dst:
        dst.write(struct.pack("<Q", len(encoded)))
        dst.write(encoded)
        dst.flush()
        for name in names:
            if name in arrays:
                dst.write(arrays[name].tobytes())
                dst.flush()
            else:
                start, end = original[name]["data_offsets"]
                while start < end:
                    count = os.sendfile(
                        dst.fileno(), src.fileno(), origin + start, end - start
                    )
                    if count <= 0:
                        raise OSError(f"Incomplete tensor copy: {filename}/{name}")
                    start += count
        os.fsync(dst.fileno())
    with safe_open(target, framework="numpy") as tensors:
        if set(tensors.keys()) != set(names):
            raise ValueError(f"Output tensor names differ: {filename}")
        for name, expected in arrays.items():
            np.testing.assert_array_equal(
                tensors.get_tensor(name), expected, err_msg=name
            )
    with open(source / filename, "rb") as src, open(target, "rb") as dst:
        for name in names:
            if name in arrays:
                continue
            start, end = original[name]["data_offsets"]
            src.seek(origin + start)
            dst.seek(8 + len(encoded) + ordered[name]["data_offsets"][0])
            remaining = end - start
            while remaining:
                size = min(remaining, 16 << 20)
                expected, actual = src.read(size), dst.read(size)
                if len(expected) != size or actual != expected:
                    raise ValueError(f"Copied tensor bytes differ: {filename}/{name}")
                remaining -= size
    return filename, names, cursor, len(selected), scale_bytes, compressed_bytes


def _configure(value):
    if isinstance(value, dict):
        result = {k: _configure(v) for k, v in value.items()}
        if isinstance(result.get("quantization_config"), dict):
            result["quantization_config"]["weight_scale_encoding"] = ENCODING
        return result
    if isinstance(value, list):
        return [_configure(v) for v in value]
    return value


def convert(source, output, *, quantization, workers=4, scale_pattern=SCALE_PATTERN):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output checkpoint already exists: {output}")
    if source in output.parents:
        raise ValueError("Output checkpoint must be outside the source directory")
    if workers < 1:
        raise ValueError("workers must be positive")
    index_name = "model.safetensors.index.json"
    index = json.loads((source / index_name).read_text())
    config = json.loads((source / "config.json").read_text())
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    if any(Path(p).is_absolute() or ".." in Path(p).parts for p in shards):
        raise ValueError("Shards must be checkpoint-relative paths")
    descriptors = {}
    for filename in shards:
        info, _ = _header(source / filename)
        with safe_open(source / filename, framework="numpy") as tensors:
            if set(tensors.keys()) != {
                n for n, f in weight_map.items() if f == filename
            }:
                raise ValueError(f"Source shard and index disagree: {filename}")
        descriptors.update((n, v) for n, v in info.items() if n != "__metadata__")
    pattern = re.compile(scale_pattern)
    selected = {name for name in descriptors if pattern.search(name)}
    if not selected:
        raise ValueError("No matching dense expert scale tensors")
    group = {"mxfp4": 32, "nvfp4": 16}[quantization]
    for name in selected:
        scale = descriptors[name]
        prefix = name.rsplit(".", 1)[0]
        weights = [
            descriptors[prefix + suffix]
            for suffix in (".weight", ".weight_packed")
            if prefix + suffix in descriptors
        ]
        if len(weights) != 1:
            raise ValueError(f"Expected one packed weight tensor for {name}")
        weight = weights[0]
        if (
            scale["dtype"]
            not in (("U8",) if quantization == "mxfp4" else ("U8", "F8_E4M3"))
            or weight["dtype"] not in ("U8", "I8")
            or len(weight["shape"]) != 2
            or weight["shape"][1] * 2 % group
            or scale["shape"] != [weight["shape"][0], weight["shape"][1] * 2 // group]
        ):
            raise ValueError(f"FP4 weight/scale geometry or dtype mismatch: {name}")
        _geometry(*scale["shape"])
        if any(f"{name}.csf32_{c}" in descriptors for c in COMPONENTS):
            raise ValueError(f"CSF output name already exists: {name}")
    input_bytes = sum((source / f).stat().st_size for f in shards)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < input_bytes:
        raise OSError("Insufficient free space for an independent checkpoint copy")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent)
    )
    started = time.monotonic()
    print(
        f"Encoding {len(selected):,} matrices in {len(shards)} shards; staging {temporary}",
        flush=True,
    )
    results = []
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = [
                pool.submit(
                    _encode_shard,
                    source,
                    temporary,
                    f,
                    selected.intersection(n for n, p in weight_map.items() if p == f),
                    quantization,
                )
                for f in shards
            ]
            for future in as_completed(jobs):
                result = future.result()
                results.append(result)
                print(
                    f"{len(results)}/{len(shards)} {result[0]}: {result[3]:,} matrices, "
                    f"scales {result[4]:,} -> {result[5]:,} bytes; "
                    f"elapsed {time.monotonic() - started:.1f}s",
                    flush=True,
                )
        for item in source.iterdir():
            if item.name in ("config.json", index_name) or item.name in {
                Path(p).parts[0] for p in shards
            }:
                continue
            if item.is_dir():
                shutil.copytree(item, temporary / item.name)
            else:
                shutil.copy2(item, temporary / item.name)
        updated = dict(
            index, weight_map={name: r[0] for r in sorted(results) for name in r[1]}
        )
        updated["metadata"] = dict(
            index.get("metadata", {}), total_size=sum(r[2] for r in results)
        )
        (temporary / index_name).write_text(json.dumps(updated, indent=2) + "\n")
        (temporary / "config.json").write_text(
            json.dumps(_configure(config), indent=2) + "\n"
        )
        with (temporary / "README.md").open("a") as stream:
            stream.write(
                f"\n\n## Compressed expert scales\n\nRouted-expert {quantization.upper()} "
                f"scales use `{ENCODING}`. Each scale tensor is replaced by "
                "`.csf32_bases`, `.csf32_codes`, `.csf32_page_bases`, "
                "`.csf32_offsets`, and `.csf32_exceptions` components. "
                "Packed weights and all other tensors retain their original bytes. "
                "Inference requires a loader supporting this scale encoding.\n"
            )
        temporary.chmod(0o755)
        temporary.rename(output)
    except BaseException:
        print(f"Incomplete output retained at {temporary}", flush=True)
        raise
    print(
        f"Complete: {output}; {sum(r[3] for r in results):,} matrices; "
        f"scale bytes {sum(r[4] for r in results):,} -> {sum(r[5] for r in results):,}; "
        f"elapsed {time.monotonic() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quantization", choices=("mxfp4", "nvfp4"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scale-pattern", default=SCALE_PATTERN)
    args = parser.parse_args()
    convert(
        args.source,
        args.output,
        quantization=args.quantization,
        workers=args.workers,
        scale_pattern=args.scale_pattern,
    )
