#!/usr/bin/env python3
"""Rewrite slab CSF scales as lossless 32x2 blocks in standard safetensors shards.

The output directory must not exist. Source files are read-only. Every scale
plane is compared byte-for-byte after conversion. Packed weights, calibration,
and other tensors are copied without numerical conversion. Configuration and
the standard safetensors index describe only the block encoding.
"""

from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import argparse
import json
import mmap
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import tempfile

import numpy as np
from safetensors import safe_open

from b12x.loader._csf_blocks import export_csf_blocks, logical_csf_components

ENCODING = "csf_32x2_v1"
COMPONENTS = ("bases", "codes", "page_bases", "offsets", "exceptions")
SUFFIXES = (".nvfp4_csf_fixed", ".mxfp4_csf_fixed")


def header(path):
    with open(path, "rb") as stream:
        count = struct.unpack("<Q", stream.read(8))[0]
        return json.loads(stream.read(count)), count + 8


class TensorFiles(ExitStack):
    def __init__(self, root, index):
        super().__init__()
        self.root, self.index = Path(root), index
        self.headers, self.maps = {}, {}

    def describe(self, name):
        filename = self.index[name]
        if filename not in self.headers:
            self.headers[filename] = header(self.root / filename)
        info, origin = self.headers[filename]
        return filename, info[name], origin

    def array(self, name):
        filename, info, origin = self.describe(name)
        if filename not in self.maps:
            with open(self.root / filename, "rb") as stream:
                self.maps[filename] = self.enter_context(
                    mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
                )
        dtype = {"U8": "u1", "U16": "<u2", "U32": "<u4"}[info["dtype"]]
        return np.ndarray(
            info["shape"],
            dtype=dtype,
            buffer=self.maps[filename],
            offset=origin + info["data_offsets"][0],
        )


def _logical_source(fixed, records, rows, columns, bits):
    row_bytes = (columns * bits + 7) // 8
    codes = fixed[:, 16:].reshape(rows, row_bytes)
    bit_columns = np.arange(columns) * bits
    result = fixed[:, :16].reshape(rows, 1) + (
        (codes[:, bit_columns // 8] >> (bit_columns % 8)) & ((1 << bits) - 1)
    )
    result = np.ascontiguousarray(result, dtype=np.uint8)
    result.ravel()[records & 0xFFFFFF] = records >> 24
    return result


def _convert_shard(arguments):
    source, destination, filename, index = arguments
    source, destination = Path(source), Path(destination)
    original, origin = header(source / filename)
    arrays, output_header, cursor, planes = {}, {}, 0, 0
    if "__metadata__" in original:
        output_header["__metadata__"] = original["__metadata__"]
    with TensorFiles(source, index) as reader:
        for name, info in original.items():
            if name == "__metadata__":
                continue
            if name.endswith((".nvfp4_csf_exceptions", ".mxfp4_csf_exceptions")):
                continue
            suffix = next((s for s in SUFFIXES if name.endswith(s)), None)
            if suffix is None:
                size = info["data_offsets"][1] - info["data_offsets"][0]
                output_header[name] = dict(info, data_offsets=[cursor, cursor + size])
                cursor += size
                continue
            scale_name = name.removesuffix(suffix)
            projection, component = scale_name.rsplit(".", 1)
            if component not in ("weight_scale", "scale"):
                raise ValueError(f"Unrecognized FP4 scale tensor name: {scale_name}")
            weights = [
                f"{projection}.{field}"
                for field in ("weight", "weight_packed")
                if f"{projection}.{field}" in index
            ]
            if len(weights) != 1:
                raise ValueError(
                    f"CSF scale needs one packed weight tensor: {scale_name}"
                )
            weight_name = weights[0]
            _, weight, _ = reader.describe(weight_name)
            rows, packed_columns = weight["shape"]
            quantization = "nvfp4" if suffix == SUFFIXES[0] else "mxfp4"
            bits, group = (4, 16) if quantization == "nvfp4" else (1, 32)
            if packed_columns * 2 % group:
                raise ValueError(
                    f"FP4 block scales require complete quantization groups: {name}"
                )
            columns = packed_columns * 2 // group
            fixed = reader.array(name)
            records = reader.array(scale_name + suffix.replace("fixed", "exceptions"))
            plane = export_csf_blocks(
                fixed, records, rows=rows, columns=columns, quantization=quantization
            )
            plane.validate()
            base, fields, positions, values = logical_csf_components(plane)
            decoded = base[:, None] + fields
            decoded.ravel()[positions] = values
            np.testing.assert_array_equal(
                decoded,
                _logical_source(fixed, records, rows, columns, bits),
                err_msg=name,
            )
            for component in COMPONENTS:
                value = getattr(plane, component)
                key = f"{scale_name}.csf32_{component}"
                arrays[key] = value
                output_header[key] = dict(
                    dtype={1: "U8", 2: "U16", 4: "U32"}[value.dtype.itemsize],
                    shape=list(value.shape),
                    data_offsets=[cursor, cursor + value.nbytes],
                )
                cursor += value.nbytes
            planes += 1
        fixed = records = None
    sizes = {
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
        (k for k in output_header if k != "__metadata__"),
        key=lambda k: (-sizes.get(output_header[k]["dtype"], 1), k),
    )
    cursor = 0
    ordered = (
        {"__metadata__": output_header["__metadata__"]}
        if "__metadata__" in output_header
        else {}
    )
    for name in names:
        info = output_header[name]
        count = info["data_offsets"][1] - info["data_offsets"][0]
        ordered[name] = dict(info, data_offsets=[cursor, cursor + count])
        cursor += count
    encoded = json.dumps(ordered, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    target = destination / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    with (
        open(source / filename, "rb", buffering=0) as src,
        open(target, "wb", buffering=0) as dst,
    ):
        dst.write(struct.pack("<Q", len(encoded)))
        dst.write(encoded)
        for name in names:
            if name in arrays:
                dst.write(memoryview(arrays[name]).cast("B"))
            else:
                start, end = original[name]["data_offsets"]
                while start < end:
                    copied = os.sendfile(
                        dst.fileno(), src.fileno(), origin + start, end - start
                    )
                    if copied <= 0:
                        raise OSError(f"Incomplete tensor copy: {filename}/{name}")
                    start += copied
        os.fsync(dst.fileno())
    with safe_open(target, framework="numpy") as tensors:
        if set(tensors.keys()) != set(names):
            raise ValueError(f"Safetensors output names differ: {filename}")
        for name, expected in arrays.items():
            np.testing.assert_array_equal(
                tensors.get_tensor(name), expected, err_msg=name
            )
    return dict(
        filename=filename, names=names, tensor_bytes=cursor, scale_planes=planes
    )


def _configure(value):
    if isinstance(value, dict):
        return {
            k: ENCODING
            if k == "weight_scale_encoding" and v == "csf"
            else _configure(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_configure(v) for v in value]
    return value


def _readme(text, count):
    contract = f'''## Storage contract

Scale recipes declare `weight_scale_encoding: "{ENCODING}"`. The {count:,}
compressed scale matrices use 32-row by two-column blocks grouped by logical
128-row by four-column tiles. NVFP4 stores four-bit offsets; MXFP4 stores
one-bit offsets. Row bases and exact-byte exceptions reconstruct every logical
scale without rounding. Partial tiles contain only their logical fields.

Each scale tensor has five components: `.csf32_bases` (uint8), `.csf32_codes`
(uint8), `.csf32_page_bases` (uint32), `.csf32_offsets` (uint16), and
`.csf32_exceptions` (uint16). Pages contain 32 complete tile groups. Exception
records contain a block-local position in the low byte and the exact scale
byte in the high byte. The standard `model.safetensors.index.json` maps each
component to its shard.

Loaders slice blocks for the requested tensor-parallel partition and prepare
compressed codes and exception indexes for the selected native scale layout.
Routed expansion writes caller-owned, reusable scale buffers directly in that
layout before ordinary FP4 compute kernels execute.

'''
    text = re.sub(
        r"## Storage contract\n.*?(?=## Runtime requirements)",
        lambda _: contract,
        text,
        flags=re.S,
    )
    text = text.replace(
        'weight_scale_encoding: "csf"', f'weight_scale_encoding: "{ENCODING}"'
    )
    text = re.sub(r"## Validation\n.*?(?=## License)", "", text, flags=re.S)
    return text


def convert(source, output, *, workers=4):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output checkpoint already exists: {output}")
    index_file = source / "model.safetensors.index.json"
    index = json.loads(index_file.read_text())
    mapping = index["weight_map"]
    shards = sorted(set(mapping.values()))
    if any(Path(p).is_absolute() or ".." in Path(p).parts for p in shards):
        raise ValueError("Safetensors shards must be checkpoint-relative files")
    if not any(name.endswith(SUFFIXES) for name in mapping):
        raise ValueError("The source has no slab CSF scale tensors")
    for name in mapping:
        for suffix in SUFFIXES:
            exception_suffix = suffix.replace("fixed", "exceptions")
            if (
                name.endswith(suffix)
                and name.removesuffix(suffix) + exception_suffix not in mapping
            ):
                raise ValueError(f"CSF fixed stream has no exception tensor: {name}")
            if (
                name.endswith(exception_suffix)
                and name.removesuffix(exception_suffix) + suffix not in mapping
            ):
                raise ValueError(f"CSF exception tensor has no fixed stream: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent)
    )
    jobs = tuple((source, temporary, p, mapping) for p in shards)
    results = []
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(_convert_shard, jobs):
                results.append(result)
                print(
                    f"{result['filename']}: {result['scale_planes']} scale planes, {result['tensor_bytes']} tensor bytes",
                    flush=True,
                )
        for item in source.iterdir():
            if item.name in (
                "config.json",
                "README.md",
                index_file.name,
            ) or item.name in {Path(p).parts[0] for p in shards}:
                continue
            if item.is_dir():
                shutil.copytree(item, temporary / item.name)
            else:
                shutil.copy2(item, temporary / item.name)
        updated = dict(
            index,
            weight_map={
                name: result["filename"]
                for result in results
                for name in result["names"]
            },
        )
        updated["metadata"] = dict(
            index.get("metadata", {}),
            total_size=sum(r["tensor_bytes"] for r in results),
        )
        (temporary / index_file.name).write_text(json.dumps(updated, indent=2) + "\n")
        config = _configure(json.loads((source / "config.json").read_text()))
        (temporary / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        if (source / "README.md").exists():
            (temporary / "README.md").write_text(
                _readme(
                    (source / "README.md").read_text(),
                    sum(r["scale_planes"] for r in results),
                )
            )
        temporary.chmod(stat.S_IMODE(source.stat().st_mode))
        temporary.rename(output)
    except BaseException:
        print(
            f"Incomplete output remains at {temporary}; source checkpoint is unchanged",
            flush=True,
        )
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("workers must be positive")
    print(convert(args.source, args.output, workers=args.workers), flush=True)


if __name__ == "__main__":
    main()
