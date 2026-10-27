# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Inline-friendly copy of NCCL's device bitcode for the SM90 GIN kernel.

nccl4py's CuTe device API links ``libnccl_device.bc`` into the kernel.  As
shipped (nccl4py 0.6.0 / NCCL 2.32.3), three things keep NCCL code out of line:

* the DSL declares the C entry points (``ncclGinPut_v2``, ``ncclGinSignal_v2``,
  ``ncclGinWaitSignal``, ...) with ``ncclTeam`` / ``ncclCoopAny`` passed as
  literal structs, while clang's definitions take ``ptr byval(...)``; the
  linked call has a mismatched function type and is emitted as an indirect
  call that LLVM never inlines;
* ``ncclCoopAny`` dispatches ``thread_rank`` / ``size`` / ``sync`` through a
  vtable, and the DSL does not fold the vtable loads after linking;
* assertions call the external ``__assertfail``.

Any call left in a kernel makes ptxas serialize every ``wgmma`` (C7509/C7510)
and ignore ``setmaxnreg`` (C7506): on H100 the internode kernel's FC1/FC2
mainloops ran 2-3x slower than the identical intranode kernel.

:func:`use_inline_nccl_device_bitcode` rewrites the bitcode once per NCCL build
(cached by content hash) so that everything inlines -- a wrapper with the DSL's
by-value signature per byval entry point, a compare-and-direct-call chain over
the address-taken functions in place of each indirect call, every definition
``alwaysinline``, and a trap body for ``__assertfail`` -- and points nccl4py's
extern stubs at the copy.  It needs ``llvmlite`` built on an LLVM that can read
the NCCL bitcode (>= its producer) and is no newer than the CuTe DSL's LLVM
(the cache key records the writer's LLVM).  The redirect is process-wide, and
a failed NCCL device assert traps without its message.  Set
``FLASHINFER_MOE_EP_NCCL_INLINE_BITCODE=0`` to link the stock bitcode.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import os
import re
import sys
import tempfile
import warnings
from typing import Dict, List, Tuple

# Bump when the rewrite changes so cached copies are regenerated.
_REWRITE_VERSION = 1

_INTRINSIC = re.compile(r"^@llvm\.")


def _split_top_level(s: str) -> List[str]:
    """Split ``s`` on commas that are not nested in (), [], {} or <>."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
            continue
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def _literal_struct(name: str, types: Dict[str, str]) -> str:
    """Expand a named struct type into the literal struct the DSL passes."""
    body = types[name].strip()
    assert body.startswith("{") and body.endswith("}"), (name, body)
    fields = _split_top_level(body[1:-1])

    def expand(t: str) -> str:
        t = t.strip()
        if t in types:
            return _literal_struct(t, types)
        m = re.fullmatch(r"\[(\d+) x (.+)\]", t)
        if m:
            return f"[{m.group(1)} x {expand(m.group(2))}]"
        return t

    return "{ " + ", ".join(expand(f) for f in fields) + " }"


def _rewrite(txt: str) -> str:
    """Rewrite textual LLVM IR of ``libnccl_device.bc`` (see module docstring)."""
    types = {
        m.group(1): m.group(2)
        for m in re.finditer(r"^(%[\w.\"$:]+) = type (\{.*\})$", txt, re.M)
    }
    groups = {
        int(m.group(1)): m.group(2)
        for m in re.finditer(r"^attributes #(\d+) = \{ (.*) \}$", txt, re.M)
    }
    nxt = max(groups) + 1

    def new_group(attrs: str) -> int:
        nonlocal nxt
        groups[nxt] = attrs
        nxt += 1
        return nxt - 1

    # 1. Every definition always inlines.
    inline_of: Dict[int, int] = {}

    def to_inline(gid: int) -> int:
        if gid not in inline_of:
            keep = [
                a
                for a in groups[gid].split()
                if a not in ("noinline", "optnone", "inlinehint", "alwaysinline")
            ]
            inline_of[gid] = new_group(" ".join(["alwaysinline"] + keep))
        return inline_of[gid]

    def fix_define(m: "re.Match[str]") -> str:
        line = m.group(0)
        g = re.search(r"#(\d+)(?=[^#]*\{$)", line)
        if not g:
            return line
        return line[: g.start(1)] + str(to_inline(int(g.group(1)))) + line[g.end(1) :]

    txt = re.sub(r"^define [^\n]*\{$", fix_define, txt, flags=re.M)

    # 2. By-value wrappers for the C entry points taking byval structs.
    wrap_group = new_group("alwaysinline convergent nounwind")
    wrappers = []
    for m in list(
        re.finditer(
            r"^define dso_local ([^@\n]*?) ?@([A-Za-z]\w*)\((.*)\) [^\n]*\{$", txt, re.M
        )
    ):
        ret, name, sig = m.group(1).split()[-1], m.group(2), m.group(3)
        params = _split_top_level(sig)
        if not any("byval(" in p for p in params):
            continue
        w_params, body, args = [], [], []
        for i, p in enumerate(params):
            ty = p.split()[0]
            bv = re.search(r"byval\((%[\w.\"$:]+)\)", p)
            if bv:
                lit = _literal_struct(bv.group(1), types)
                w_params.append(f"{lit} %p{i}")
                body.append(f"  %m{i} = alloca {bv.group(1)}, align 8")
                body.append(f"  store {lit} %p{i}, ptr %m{i}, align 4")
                args.append(f"ptr byval({bv.group(1)}) %m{i}")
            else:
                ext = (
                    " zeroext"
                    if "zeroext" in p.split()
                    else (" signext" if "signext" in p.split() else "")
                )
                w_params.append(f"{ty} %p{i}")
                args.append(f"{ty}{ext} %p{i}")
        call = f"call {ret} @{name}.impl({', '.join(args)})"
        tail = (
            f"  %r = {call}\n  ret {ret} %r"
            if ret != "void"
            else f"  {call}\n  ret void"
        )
        wrappers.append(
            f"define dso_local {ret} @{name}({', '.join(w_params)}) #{wrap_group} {{\n"
            + "\n".join(body)
            + f"\n{tail}\n}}\n"
        )
        txt = re.sub(r"@" + name + r"(?![\w.])", f"@{name}.impl", txt)
        txt = re.sub(
            r"^define dso_local (?=[^@\n]*@" + re.escape(name) + r"\.impl\()",
            "define internal ",
            txt,
            count=1,
            flags=re.M,
        )

    # 3. Indirect calls -> compare-and-direct-call chains over the functions
    # whose address is taken (the ncclCoopAny vtables).
    taken = set(
        re.findall(r"ptr (@[\w.$]+)", "\n".join(re.findall(r"^@.*$", txt, re.M)))
    )
    sig_of: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
    for m in re.finditer(
        r"^define [^@\n]*?(\S+) (@[\w.$]+)\((.*)\)[^\n]*\{$", txt, re.M
    ):
        if m.group(2) in taken:
            sig_of[m.group(2)] = (
                m.group(1),
                tuple(p.split()[0] for p in _split_top_level(m.group(3))),
            )
    conv = new_group("convergent nounwind")
    disp = new_group("alwaysinline convergent nounwind")
    dispatchers: Dict[Tuple[str, Tuple[str, ...]], str] = {}

    def dispatcher(ret: str, ptys: Tuple[str, ...]) -> str:
        key = (ret, ptys)
        if key in dispatchers:
            return dispatchers[key]
        cands = sorted(f for f, s in sig_of.items() if s == key)
        if not cands:
            raise RuntimeError(
                f"indirect call {ret}({', '.join(ptys)}) has no candidate target"
            )
        name = f"@nccl.dispatch.{len(dispatchers)}"
        dispatchers[key] = name
        formals = ", ".join(f"{t} %a{i}" for i, t in enumerate(ptys))
        actuals = ", ".join(f"{t} %a{i}" for i, t in enumerate(ptys))
        lines = [
            f"define internal {ret} {name}(ptr %f, {formals}) #{disp} {{",
            "entry:",
        ]
        for i, fn in enumerate(cands):
            lines += [
                f"  %e{i} = icmp eq ptr %f, {fn}",
                f"  br i1 %e{i}, label %c{i}, label %n{i}",
                f"c{i}:",
            ]
            if ret == "void":
                lines += [f"  call void {fn}({actuals}) #{conv}", "  ret void"]
            else:
                lines += [
                    f"  %r{i} = call {ret} {fn}({actuals}) #{conv}",
                    f"  ret {ret} %r{i}",
                ]
            lines.append(f"n{i}:")
        lines += ["  call void @llvm.trap()", "  unreachable", "}"]
        wrappers.append("\n".join(lines) + "\n")
        return name

    def fix_indirect(m: "re.Match[str]") -> str:
        ret, callee, args = m.group(1), m.group(2), _split_top_level(m.group(3))
        ptys = tuple(a.split()[0] for a in args)
        actual = ", ".join(
            f"{t} {a.split()[-1]}" for t, a in zip(ptys, args, strict=True)
        )
        return f"call {ret} {dispatcher(ret, ptys)}(ptr {callee}, {actual})"

    txt = re.sub(
        r"call (?:noundef |signext |zeroext )*(void|i\d+|ptr|float|double) (%[\w.]+)"
        r"\(((?:[^()\n]|\([^()\n]*\))*)\)",
        fix_indirect,
        txt,
    )
    left = re.findall(r"call [^@\n]*?%[\w.]+\(", txt)
    if left:
        raise RuntimeError(f"{len(left)} indirect calls left, e.g. {left[0]!r}")

    # 4. External declarations: only __assertfail may remain; give it a trap body.
    externs = [
        m.group(1)
        for m in re.finditer(r"^declare [^@\n]*(@[\w.$]+)\(", txt, re.M)
        if not _INTRINSIC.match(m.group(1))
    ]
    for ext in externs:
        if ext != "@__assertfail":
            raise RuntimeError(f"unexpected external function {ext} in NCCL bitcode")
        m = re.search(r"^declare [^\n]*@__assertfail\(([^)]*)\)[^\n]*$", txt, re.M)
        formals = ", ".join(
            f"{p.split()[0]} %a{i}" for i, p in enumerate(_split_top_level(m.group(1)))
        )
        trap = new_group("alwaysinline cold noreturn nounwind")
        txt = (
            txt[: m.start()]
            + f"define internal void @__assertfail({formals}) #{trap} {{\n"
            "  call void @llvm.trap()\n  unreachable\n}\n" + txt[m.end() :]
        )
    if "declare void @llvm.trap()" not in txt:
        txt += "\ndeclare void @llvm.trap() cold noreturn nounwind\n"

    txt = re.sub(r"^attributes #\d+ = \{ .* \}\n?", "", txt, flags=re.M)
    txt += "\n" + "\n".join(wrappers)
    txt += (
        "\n"
        + "\n".join(f"attributes #{k} = {{ {v} }}" for k, v in sorted(groups.items()))
        + "\n"
    )
    return txt


def rewrite_nccl_device_bitcode(data: bytes) -> bytes:
    """Return the inline-friendly rewrite of ``libnccl_device.bc`` bytes."""
    import llvmlite.binding as llvm

    mod = llvm.parse_assembly(_rewrite(str(llvm.parse_bitcode(data))))
    mod.verify()
    return mod.as_bitcode()


def _cache_dir() -> str:
    from flashinfer.jit.env import FLASHINFER_CACHE_DIR

    return os.path.join(str(FLASHINFER_CACHE_DIR), "nccl_device_bitcode")


@functools.cache
def use_inline_nccl_device_bitcode() -> bool:
    """Point nccl4py's CuTe extern stubs at the inline-friendly bitcode.

    Must run before the first GIN kernel is traced.  Returns whether the
    rewritten bitcode is in use; on any failure it warns and leaves nccl4py on
    the stock bitcode (correct, but ptxas serializes the kernel's wgmma).
    """
    if os.environ.get("FLASHINFER_MOE_EP_NCCL_INLINE_BITCODE", "1") == "0":
        return False
    src = "nccl4py's libnccl_device.bc"
    try:
        import llvmlite.binding as llvm
        from cutlass.cute import BitCode
        from nccl.core.device.cute import _bindings

        src = _bindings._BC.path
        with open(src, "rb") as f:
            data = f.read()
        writer = ".".join(str(v) for v in llvm.llvm_version_info)
        key = hashlib.sha256(
            data + f"v{_REWRITE_VERSION}-llvm{writer}".encode()
        ).hexdigest()[:16]
        dst = os.path.join(_cache_dir(), f"libnccl_device_inline_{key}.bc")
        if not os.access(dst, os.R_OK):
            out = rewrite_nccl_device_bitcode(data)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dst), prefix=".nccl_bc.")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(out)
                os.chmod(tmp, 0o644)
                os.replace(tmp, dst)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise

        old, new = _bindings._BC, BitCode(dst)
        patched = 0
        for name, mod in list(sys.modules.items()):
            if not name.startswith("nccl.core.device.cute") or mod is None:
                continue
            for obj in vars(mod).values():
                if getattr(obj, "source", None) != old:
                    continue
                obj.source = new
                # Signatures traced before this call cached an FFI with the
                # stock source; linking both would keep NCCL code out of line.
                for ffi in getattr(obj, "ffis", {}).values():
                    ffi.source = new
                patched += 1
        if patched == 0:
            raise RuntimeError("no nccl4py extern stub references the device bitcode")
        _bindings._BC = new
    except Exception as exc:  # noqa: BLE001 -- any failure falls back to stock bitcode
        warnings.warn(
            f"[moe_ep] could not use an inline NCCL device bitcode for {src!r} "
            f"({type(exc).__name__}: {exc}); linking it as shipped, so ptxas "
            "serializes the GIN kernel's wgmma (install llvmlite to fix).",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True
