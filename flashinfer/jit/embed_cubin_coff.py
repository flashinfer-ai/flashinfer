"""Write a cubin into a COFF object for the MSVC linker.

On Linux, ``tvm_ffi.utils.embed_cubin`` embeds a cubin with GNU ``ld -r -b binary``
and ``objcopy``. The MSVC toolchain has neither tool, so on Windows the JIT build
writes the object itself: one read-only section holds the cubin bytes, with
``__tvm_ffi__cubin_<name>`` at offset 0 and ``__tvm_ffi__cubin_<name>_end`` at the
end of the bytes, the two symbols that ``TVM_FFI_EMBED_CUBIN`` declares.

This file uses only the standard library: ninja runs it as a script, without
importing the ``flashinfer`` package.
"""

import argparse
import struct
from pathlib import Path

IMAGE_FILE_MACHINE = {
    "arm64": 0xAA64,
    "x64": 0x8664,
}

_IMAGE_SCN_CNT_INITIALIZED_DATA = 0x00000040
_IMAGE_SCN_ALIGN_64BYTES = 0x00700000
_IMAGE_SCN_MEM_READ = 0x40000000
_IMAGE_SYM_CLASS_EXTERNAL = 2
_IMAGE_SYM_CLASS_STATIC = 3

_FILE_HEADER = struct.Struct("<HHIIIHH")
_SECTION_HEADER = struct.Struct("<8sIIIIIIHHI")
_SYMBOL = struct.Struct("<8sIhHBB")
_SECTION_AUX = struct.Struct("<IHHIHB3x")


def coff_cubin_object(cubin: bytes, name: str, machine: str) -> bytes:
    """Return a COFF object that defines the two embedded-cubin symbols."""
    if machine not in IMAGE_FILE_MACHINE:
        raise ValueError(f"unsupported COFF machine: {machine!r}")
    section_name = b".rdata"
    raw_data_offset = _FILE_HEADER.size + _SECTION_HEADER.size
    symbol_table_offset = raw_data_offset + len(cubin)

    strings = bytearray()

    def symbol_name(text: str) -> bytes:
        encoded = text.encode("ascii")
        if len(encoded) <= 8:
            return encoded.ljust(8, b"\0")
        offset = 4 + len(strings)
        strings.extend(encoded + b"\0")
        return struct.pack("<II", 0, offset)

    symbols = [
        _SYMBOL.pack(section_name.ljust(8, b"\0"), 0, 1, 0, _IMAGE_SYM_CLASS_STATIC, 1),
        _SECTION_AUX.pack(len(cubin), 0, 0, 0, 0, 0),
        _SYMBOL.pack(
            symbol_name(f"__tvm_ffi__cubin_{name}"),
            0,
            1,
            0,
            _IMAGE_SYM_CLASS_EXTERNAL,
            0,
        ),
        _SYMBOL.pack(
            symbol_name(f"__tvm_ffi__cubin_{name}_end"),
            len(cubin),
            1,
            0,
            _IMAGE_SYM_CLASS_EXTERNAL,
            0,
        ),
    ]

    file_header = _FILE_HEADER.pack(
        IMAGE_FILE_MACHINE[machine], 1, 0, symbol_table_offset, len(symbols), 0, 0
    )
    section_header = _SECTION_HEADER.pack(
        section_name.ljust(8, b"\0"),
        0,
        0,
        len(cubin),
        raw_data_offset,
        0,
        0,
        0,
        0,
        _IMAGE_SCN_CNT_INITIALIZED_DATA
        | _IMAGE_SCN_ALIGN_64BYTES
        | _IMAGE_SCN_MEM_READ,
    )
    string_table = struct.pack("<I", 4 + len(strings)) + bytes(strings)
    return file_header + section_header + cubin + b"".join(symbols) + string_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cubin", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--machine", choices=sorted(IMAGE_FILE_MACHINE), required=True)
    parser.add_argument("--output-obj", type=Path, required=True)
    args = parser.parse_args()
    obj = coff_cubin_object(args.cubin.read_bytes(), args.name, args.machine)
    args.output_obj.write_bytes(obj)


if __name__ == "__main__":
    main()
