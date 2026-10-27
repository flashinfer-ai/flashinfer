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
"""

import glob
import re
from pathlib import Path

import pytest
import torch

from flashinfer import NVFP44Over6Config, SfLayout, nvfp4_quantize
from flashinfer.experimental.cake_nvfp4_per_token import cake_backend as cb
from flashinfer.experimental.cake_nvfp4_per_token.cake_jit import (
    ARCHES,
    ARG_PLANS,
    DEFINITIONS,
    KERNELS,
    MODULES,
    gen_cake_nvfp4_per_token_module,
    kernel_definitions,
    kernel_module_name,
)
from tests.test_helpers.cake_nvfp4_per_token import (
    VALIDATED_F16_INPUT_ROWS,
    VALIDATED_OFF_MATRIX_QUANTIZE_K,
    VALIDATED_QUANTIZE_K,
    required_kernel_keys,
)

FLT_MAX = 3.4028234663852886e38
GLOBAL_SCALE_INV = 1.0 / (448.0 * 6.0)


# ---------------------------------------------------------------------------
# Host rules (CPU)
# ---------------------------------------------------------------------------


def test_padding_rules():
    assert cb.padded_rows(1) == 128 and cb.padded_rows(128) == 128
    assert cb.padded_rows(129) == 256 and cb.padded_rows(4097) == 4224
    assert cb.padded_sf_cols(7168) == 448 and cb.padded_sf_cols(2112) == 132
    assert cb.padded_sf_cols(16) == 4


def test_cta_config_rules():
    # Few CTAs: widest CTA, one CTA per SM for a single row, 256 threads with
    # two blocks per thread for K <= 8192, two CTAs per SM above 128 rows; a
    # single wave of rows (128 < M <= 148) keeps the widest CTA, more than one
    # wave with at most four blocks per 256-thread lane takes 256 x 2.
    assert cb.cta_config(7168, 1) == (512, 1)
    assert cb.cta_config(7168, 8) == (256, 3)
    assert cb.cta_config(7168, 130) == (256, 3)
    assert cb.cta_config(16384, 32) == (512, 1)
    assert cb.cta_config(16384, 130) == (512, 2)
    assert cb.cta_config(16384, 148) == (512, 2)
    assert cb.cta_config(16384, 149) == (256, 2)
    assert cb.cta_config(16384, 257) == (256, 2)
    assert cb.cta_config(18432, 257) == (512, 2)
    assert cb.cta_config(28672, 257) == (512, 2)
    # Many CTAs: narrow CTAs holding 4-8 blocks per thread.
    assert cb.cta_config(7168, 512) == (256, 4)
    assert cb.cta_config(16384, 2048) == (256, 4)
    assert cb.cta_config(28672, 1000) == (256, 3)
    assert cb.cta_config(7168, 8192) == (128, 8)
    assert cb.cta_config(16384, 8192) == (128, 5)
    assert cb.cta_config(28672, 8192) == (256, 3)


def _registered(plan, arch):
    """The plan's kernel resolves to a generated program of ``arch``."""
    assert plan.kernel_key in KERNELS[arch], plan.kernel_key
    return plan


def test_quantizer_takes_the_static_k_instance_below_the_row_limit():
    """Contiguous row sets of fewer than STATIC_M_LIMIT tokens at a row width of
    the validated set take the instance with the row width compiled in; a larger
    row set, another row width or a row-strided view takes the runtime-geometry
    instance of the same CTA shape.  A single row has no stride, so M == 1 takes
    the compiled-in instance even through a strided view.  The widths of one CTA
    shape share one program and differ only in its K_STATIC definition, and the
    tier spans two CTA shapes per width: the small-row shape and the one
    cta_config picks for the whole LARGE_ROW_MIN_M .. STATIC_M_LIMIT range."""
    for k in cb.STATIC_K:
        for m in (1, 8, 16, 130, cb.LARGE_ROW_MIN_M, 2048, cb.STATIC_M_LIMIT - 1):
            plan = _registered(
                cb.quant_plan(m, k, True, True, "sm_100a", 148), "sm_100a"
            )
            assert plan.k_static == k and plan.kernel_key.endswith(f"_k{k}")
            assert DEFINITIONS[plan.kernel_key] == {"K_STATIC": k}
            runtime_key = plan.kernel_key.removesuffix(f"_k{k}")
            assert runtime_key not in DEFINITIONS
            for arch in ARCHES:
                # one delivered source per CTA shape: every width of the shape names the
                # same module and differs only in its K_STATIC definition
                family = {
                    key: module
                    for key, module in KERNELS[arch].items()
                    if key.startswith(f"{runtime_key}_k")
                }
                assert plan.kernel_key in family
                assert len(set(family.values())) == 1, family
                # the runtime-geometry program of the same CTA shape, where the matrix
                # reaches one, is a different source: it takes the width as an argument
                if runtime_key in KERNELS[arch]:
                    assert KERNELS[arch][runtime_key] not in set(family.values())
            # A strided view cannot take the compiled-in instance (the program assumes
            # contiguous rows); the runtime-geometry sibling it falls back to exists only
            # where the validated matrix reaches that CTA shape -- see
            # test_row_strided_activation_at_a_new_width_names_its_missing_program.
            strided = cb.quant_plan(
                m, k, True, True, "sm_100a", 148, x_row_stride=k + 16
            )
            if m == 1:
                assert strided == plan
            else:
                assert strided.kernel_key == runtime_key and strided.k_static is None
        # the row counts of the tier span exactly two CTA shapes per width
        shapes = {
            cb.cta_config(k, m) for m in (1, 8, 16, 130, 511, cb.LARGE_ROW_MIN_M, 2048)
        }
        assert len(shapes) <= 3, (k, shapes)
    # the widths of one CTA shape are one program (the tier's widths span two shapes)
    by_shape: dict[tuple[int, int, int], set[str]] = {}
    for k in cb.STATIC_K:
        single = cb.quant_plan(1, k, True, True, "sm_100a", 148)
        shape = (single.threads, single.blocks_per_thread, single.min_blocks)
        by_shape.setdefault(shape, set()).add(KERNELS["sm_100a"][single.kernel_key])
    assert len(by_shape) == 2 and all(len(p) == 1 for p in by_shape.values()), by_shape
    big = _registered(
        cb.quant_plan(cb.STATIC_M_LIMIT, 7168, True, False, "sm_100a", 148), "sm_100a"
    )
    assert big.k_static is None and "_k" not in big.kernel_key
    other = _registered(
        cb.quant_plan(1, 7168 + 16 * 32, True, False, "sm_100a", 148), "sm_100a"
    )
    assert other.k_static is None


def test_row_strided_activation_at_a_new_width_names_its_missing_program():
    """A row-strided activation cannot take the compiled-in instance -- the program assumes
    contiguous rows -- so it falls back to the runtime-geometry key of the same CTA shape.
    The CTA shapes that only the new widths reach (K = 2688 / 4096 at or above
    LARGE_ROW_MIN_M) have no runtime-geometry program in this package, and the dispatcher
    names the missing key instead of mis-dispatching.  Generating the runtime-geometry
    sibling of those shapes is a recorded follow-up."""
    named = 0
    for k in (2688, 4096):
        for m in (cb.LARGE_ROW_MIN_M, 2048, cb.STATIC_M_LIMIT - 1):
            plan = cb.quant_plan(m, k, True, False, "sm_100a", 148, x_row_stride=k + 16)
            assert plan.k_static is None
            if plan.kernel_key in KERNELS["sm_100a"]:
                continue
            named += 1
            with pytest.raises(
                NotImplementedError, match=r"quant:t\d+_b\d+_mb\d+_bf16"
            ):
                kernel_module_name("sm_100a", plan.kernel_key)
    assert (
        named > 0
    )  # recorded follow-up: runtime-geometry sibling of the new CTA shapes


def test_quant_plan_grid_and_keys():
    plan = _registered(cb.quant_plan(257, 7168, True, False, "sm_100a", 148), "sm_100a")
    assert plan.grid == 384 and plan.padded_rows == 384 and plan.padded_cols == 448
    assert plan.threads == 256 and plan.blocks_per_thread == 2
    # fp16 input with the per-token scale folded into the GEMM alpha: the plan
    # resolves, but no fp16-input folded quantizer is generated (bf16 fold and
    # fp16 unfolded programs exist); the dispatcher names the missing program.
    fold = cb.quant_plan(257, 7168, False, True, "sm_100a", 148)
    assert fold.kernel_key != plan.kernel_key
    assert "_f16_fold" in fold.kernel_key
    with pytest.raises(NotImplementedError, match=r"quant:.*_f16_fold"):
        kernel_module_name("sm_100a", fold.kernel_key)
    wide = _registered(
        cb.quant_plan(8192, 7168, True, False, "sm_103a", 152), "sm_103a"
    )
    # One CTA per padded row regardless of the SM count.
    assert wide.threads == 128 and wide.min_blocks == 8 and wide.grid == 8192
    assert (
        _registered(cb.quant_plan(1, 2688, True, False, "sm_100a", 148), "sm_100a").grid
        == 128
    )
    # Short launches of long bf16 rows split every row over a cluster of four CTAs, each
    # with the CTA shape of a quarter row: one cluster per row, then the padding clusters
    # (each CTA zeroes 1024 16-byte chunks of the last scale tile's 8 x padded_cols).
    split = _registered(cb.quant_plan(1, 28672, True, False, "sm_100a", 148), "sm_100a")
    assert split.cluster_split == 4 and split.kernel_key == "quant:t512_b1_mb1_bf16_cs4"
    assert split.grid == (1 + 4) * 4 and split.padded_rows == 128
    split = _registered(cb.quant_plan(32, 18432, True, True, "sm_103a", 152), "sm_103a")
    assert (split.threads, split.blocks_per_thread, split.min_blocks) == (256, 2, 3)
    assert split.kernel_key == "quant:t256_b2_mb3_bf16_fold_cs4"
    assert split.grid == (32 + 3) * 4
    static = _registered(cb.quant_plan(1, 7168, True, False, "sm_103a", 152), "sm_103a")
    assert static.kernel_key == "quant:t128_b1_mb1_bf16_cs4_k7168"
    assert cb.quant_plan(8, 7168, True, False, "sm_100a", 148).cluster_split == 1
    assert cb.quant_plan(33, 28672, True, False, "sm_100a", 148).cluster_split == 1
    assert cb.quant_plan(8, 28672, False, False, "sm_100a", 148).cluster_split == 1
    assert cb.quant_plan(8, 12288, True, False, "sm_100a", 148).cluster_split == 1
    # K = 8192: one CTA per row for 2..32 tokens (sm_100a) and 2..16 tokens (sm_103a),
    # with or without the folded out_scale.  One token keeps the four-CTA split, and so
    # does sm_103a from 17 tokens up to its limits (17 with the out_scale folded).
    for arch, sm_count, high in (("sm_100a", 148, 32), ("sm_103a", 152, 16)):
        for fold in (False, True):
            splits = [
                cb.quant_plan(m, 8192, True, fold, arch, sm_count).cluster_split
                for m in (1, 2, high)
            ]
            assert splits == [4, 1, 1], (arch, fold, splits)
    assert cb.quant_plan(17, 8192, True, True, "sm_103a", 152).cluster_split == 4
    assert cb.quant_plan(18, 8192, True, True, "sm_103a", 152).cluster_split == 1
    assert cb.quant_plan(32, 8192, True, False, "sm_103a", 152).cluster_split == 4
    assert cb.quant_plan(32, 8192, True, True, "sm_100a", 148).cluster_split == 1
    # One program per CTA shape (threads, register blocks per thread, occupancy):
    # K = 2688 and 4096 resolve to one shape, 7168 and 8192 to another.  The widths of
    # the static tier share that shape's program and differ in K_STATIC; a width outside
    # the tier takes the shape's runtime-geometry program, which is a second source.
    by_shape: dict[tuple[int, int, int], set[str]] = {}
    for k in (2688, 4096, 7168, 8192):
        plan = _registered(
            cb.quant_plan(130, k, True, False, "sm_100a", 148), "sm_100a"
        )
        shape = (plan.threads, plan.blocks_per_thread, plan.min_blocks)
        by_shape.setdefault(shape, set()).add(plan.kernel_key)
    assert len(by_shape) == 2, by_shape
    for shape, keys in by_shape.items():
        static = {KERNELS["sm_100a"][key] for key in keys if re.search(r"_k\d+$", key)}
        runtime = {
            KERNELS["sm_100a"][key] for key in keys if not re.search(r"_k\d+$", key)
        }
        assert len(static) <= 1 and len(runtime) <= 1, (shape, keys)
        assert not (static & runtime), (shape, keys)
    with pytest.raises(ValueError):
        cb.quant_plan(8, 100, True, False, "sm_100a", 148)


# Row lengths of the validated matrix and dense token counts around every
# dispatch boundary (single row, one wave, two CTAs per SM, the many-CTA split).
REACHABLE_K = VALIDATED_QUANTIZE_K + VALIDATED_OFF_MATRIX_QUANTIZE_K
REACHABLE_M = tuple(range(1, 33)) + (
    *range(48, 160, 4),
    *range(160, 600, 16),
    *(1000, 1024, 2047, 2048, 2049, 4096, 4097, 8191, 8192, 8193, 16384, 65536),
)


@pytest.mark.parametrize(
    "arch,sm_count", [("sm_100a", 148), ("sm_103a", 152), ("sm_100a", 132)]
)
def test_every_reachable_bf16_quantizer_shape_has_a_program(arch, sm_count):
    """The dispatch is total for bf16 activations: every CTA shape it selects over
    the validated row lengths and any token count resolves to a generated program
    (K and the row stride are kernel arguments, so the set is the CTA shapes)."""
    shapes = set()
    for k in REACHABLE_K:
        for m in REACHABLE_M:
            for fold in (False, True):
                plan = _registered(
                    cb.quant_plan(m, k, True, fold, arch, sm_count), arch
                )
                shapes.add(plan.kernel_key)
    assert shapes <= set(KERNELS[arch])


def test_quantizer_alone_cluster_split_rows():
    """The quantizer alone (``quant_plan(..., alone=True)``, the public per-token quantize
    entries) splits bf16 rows over the clusters of QUANT_ALONE_CLUSTER_SPLIT at the measured
    SM count of each architecture, with a generated program; everywhere else (other token
    counts and row lengths, the other SM count, fp16 activations) it takes the chain's
    plan, :func:`quant_cluster_split`."""
    other = {"sm_100a": 152, "sm_103a": 148}
    for arch, table in cb.QUANT_ALONE_CLUSTER_SPLIT.items():
        sm_count = cb.MEASURED_SM_COUNT[arch]
        served = set()
        for k in sorted(set(REACHABLE_K) | set(table)):
            for m in REACHABLE_M:
                for fold in (False, True):
                    chain = cb.quant_plan(m, k, True, fold, arch, sm_count)
                    alone = _registered(
                        cb.quant_plan(m, k, True, fold, arch, sm_count, alone=True),
                        arch,
                    )
                    want = next(
                        (c for lo, hi, c in table.get(k, ()) if lo <= m <= hi), None
                    )
                    if want is None:
                        assert alone == chain, (arch, k, m, fold)
                        continue
                    served.add((k, m))
                    assert alone.cluster_split == want, (arch, k, m, fold)
                    assert alone.kernel_key.endswith(f"_cs{want}"), alone.kernel_key
                    assert alone.grid % want == 0 and alone.grid >= m * want
                    sms = other[arch]
                    assert cb.quant_plan(
                        m, k, True, fold, arch, sms, alone=True
                    ) == cb.quant_plan(m, k, True, fold, arch, sms), (arch, sms, k, m)
                    assert cb.quant_plan(
                        m, k, False, fold, arch, sm_count, alone=True
                    ) == cb.quant_plan(m, k, False, fold, arch, sm_count)
        listed = {
            (k, m)
            for k, ranges in table.items()
            for lo, hi, _ in ranges
            for m in REACHABLE_M
            if lo <= m <= hi
        }
        assert served == listed, (arch, sorted(listed - served))


def test_f16_quantizer_shapes_match_the_validated_f16_rows():
    """fp16 activations are generated for the validated fp16 rows (K = 7168 at
    256 threads, scale not folded); every other fp16 shape names its missing
    program instead of mis-dispatching."""
    for m in VALIDATED_F16_INPUT_ROWS:
        _registered(cb.quant_plan(m, 7168, False, False, "sm_100a", 148), "sm_100a")
    unregistered = 0
    for k in REACHABLE_K:
        for m in REACHABLE_M:
            for fold in (False, True):
                plan = cb.quant_plan(m, k, False, fold, "sm_100a", 148)
                if plan.kernel_key in KERNELS["sm_100a"]:
                    continue
                unregistered += 1
                with pytest.raises(
                    NotImplementedError, match=r"quant:t\d+_b\d+_mb\d+_f16"
                ):
                    kernel_module_name("sm_100a", plan.kernel_key)
    assert unregistered > 0  # recorded follow-up: fp16 input beyond the validated rows


def test_static_k_programs_are_built_with_their_compile_line_definition():
    """A program reached through a static-K key must never be compiled without the
    definition: every build of it has to come from a kernel key whose definitions the
    registry resolves (its own source either carries the name value-free behind an
    #error guard or defaults it to the runtime form).  The JIT spec of each such key
    must carry -DK_STATIC=<K> and name the value, so the two widths of one program do
    not share a build directory."""
    specialized = {
        name
        for name in MODULES
        if any(
            "K_STATIC" in dict(kernel_definitions(key))
            for arch in ARCHES
            for key, module in KERNELS.get(arch, {}).items()
            if module == name
        )
    }
    assert specialized, "no program declares a compile-line definition"
    for arch in ARCHES:
        for key, module in KERNELS[arch].items():
            definitions = kernel_definitions(key)
            width = re.search(r"_k(\d+)$", key)
            if width is None:
                assert not definitions, (key, definitions)
                continue
            assert dict(definitions) == {"K_STATIC": int(width.group(1))}, (
                key,
                definitions,
            )
            spec = gen_cake_nvfp4_per_token_module(module, arch, definitions)
            flags = " ".join(spec.extra_cuda_cflags)
            assert f"-DK_STATIC={width.group(1)}" in flags, (key, flags)
            assert f"k_static{width.group(1)}" in spec.name, (key, spec.name)
            plain = gen_cake_nvfp4_per_token_module(module, arch)
            assert plain.name != spec.name, (key, spec.name)
    # every specialized program is reached only through keys that define the name
    for arch in ARCHES:
        for key, module in KERNELS[arch].items():
            if module in specialized:
                assert "K_STATIC" in dict(kernel_definitions(key)), (arch, key)


@pytest.mark.parametrize("arch", cb.ARCHES)
def test_required_kernel_keys_are_registered_when_programs_exist(arch):
    required = required_kernel_keys(arch)
    assert any(key.startswith("quant:") for key in required)
    assert any(key.startswith("gemm:") for key in required)
    if arch in KERNELS:
        missing = sorted(set(required) - set(KERNELS[arch]))
        assert not missing, f"{arch} lacks {missing}"
        for module_name in KERNELS[arch].values():
            record = MODULES[module_name]
            assert record["arg_plan"] in ARG_PLANS, (module_name, record["arg_plan"])
            assert arch in record["closure_sha256"], (module_name, arch)


_PROGRAM_KERNEL_FLAG = "-DCAKE_PROGRAM_KERNEL="
_PROGRAM_KERNEL_DEFAULT = re.compile(r"^#define CAKE_PROGRAM_KERNEL (\w+)$", re.M)
_KNOB_DEFAULT = re.compile(r"^#define (CAKE_[A-Z0-9_]+) (\S+)$", re.M)
_TMA_POLICY = {"L": "0x14F0000000000000ULL", "F": "0x12F0000000000000ULL", None: "0"}


def _knobs_of_key(key):
    """Knob values the shared sources must see for ``key`` (its name grammar)."""
    m = re.match(r"quant:t(\d+)_b(\d+)_mb(\d+)_(bf16|f16)(_fold)?(_sl)?", key)
    if m:
        threads, blocks, min_blocks = (int(g) for g in m.groups()[:3])
        return {
            "CAKE_Q_THREADS": str(threads),
            "CAKE_Q_MIN_BLOCKS": str(min_blocks),
            "CAKE_Q_BPT": str(blocks),
            "CAKE_Q_WORDS": str(8 * blocks),
            "CAKE_Q_WARPS": str(threads // 32),
            "CAKE_Q_BLOCK": f"{threads}u",
            "CAKE_FOLD": "1" if m.group(5) else "0",
            "CAKE_STORE_EVICT_LAST": "1" if m.group(6) else "0",
        }
    m = re.match(
        r"gemm:.*?_(bf16|f16)(?:_a([LF]))?(?:_b([LF]))?(?:_l2\d+b)?(?:_g(\d+))?", key
    )
    assert m, key
    return {
        "CAKE_OUT_F16": "1" if m.group(1) == "f16" else "0",
        "CAKE_A_POLICY": _TMA_POLICY[m.group(2)],
        "CAKE_B_POLICY": _TMA_POLICY[m.group(3)] if m.group(3) else None,
        "CAKE_RASTER_GROUP": m.group(4) or "0",
    }


def test_shared_kernel_sources_keep_distinct_symbols_and_knobs():
    # Knob twins build from one kernel translation unit: the record whose source
    # it is defines the symbol through the file's ``CAKE_PROGRAM_KERNEL`` default,
    # every other record names its own symbol on the compile line.  All programs
    # of one architecture load into one process, so every record must still
    # produce ``kernel_<record>``; and the knob values the record's compile line
    # and the source's defaults select must be the ones its kernel keys name.
    # The kernel source is read where the loader's JIT spec resolves it.
    symbols = {}
    for name, record in MODULES.items():
        kernel_source = record["sources"][0]
        assert kernel_source.endswith("_kernel.cu"), (name, kernel_source)
        spec = gen_cake_nvfp4_per_token_module(name, record["arches"][0])
        text = spec.sources[0].read_text()
        flags = [
            f for f in record["compile_flags"] if f.startswith(_PROGRAM_KERNEL_FLAG)
        ]
        defaults = _PROGRAM_KERNEL_DEFAULT.findall(text)
        assert len(flags) <= 1 and len(defaults) <= 1, (name, flags, defaults)
        if flags:
            symbol = flags[0][len(_PROGRAM_KERNEL_FLAG) :]
        elif defaults:
            symbol = defaults[0]
        else:
            symbol = f"kernel_{name}"
        if name not in kernel_source:
            assert flags and defaults, (name, kernel_source)
        assert f"{symbol}(" in text or "CAKE_PROGRAM_KERNEL(" in text, name
        assert symbol == f"kernel_{name}", (name, symbol)
        symbols[name] = symbol
        declared = {
            knob: value
            for knob, value in _KNOB_DEFAULT.findall(text)
            if knob not in ("CAKE_PROGRAM_KERNEL", "CAKE_INF")
        }
        given = dict(
            f[2:].split("=", 1)
            for f in record["compile_flags"]
            if f.startswith("-DCAKE_") and not f.startswith("-DCAKE_PROGRAM_")
        )
        # CAKE_Q_BLOCK is the launch-block knob of the shared bf16 quantizer
        # binding, not of the kernel source; its value must be the kernel's
        # thread count with the unsigned suffix (pinned by _knobs_of_key).
        assert set(given) <= set(declared) | {"CAKE_Q_BLOCK"}, (name, given, declared)
        effective = {**declared, **given}
        keys = {k for table in KERNELS.values() for k, n in table.items() if n == name}
        for key in keys if effective else ():
            expected = _knobs_of_key(key)
            for knob, value in effective.items():
                assert expected.get(knob) == value, (name, key, knob, value)
    assert len(set(symbols.values())) == len(MODULES)


def test_generated_sources_are_declared_as_package_data():
    """A non-editable install carries the sources the loader resolves next to
    cake_jit.py: every kernel and binding named by MODULES and the two family
    headers they include.  pyproject.toml sets include-package-data = false, so
    only the package-data globs of this package put them into a wheel."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("pyproject.toml is only available in source-tree test runs")
    key = '"flashinfer.experimental.cake_nvfp4_per_token" = ['
    text = pyproject.read_text(encoding="utf-8")
    assert key in text, "pyproject.toml has no package-data entry for the package"
    globs = re.findall(r'"([^"]*)"', text.split(key, 1)[1].split("]", 1)[0])
    package_root = Path(cb.__file__).resolve().parent
    shipped = {
        path
        for pattern in globs
        for path in glob.glob(pattern, root_dir=package_root, recursive=True)
        if (package_root / path).is_file()
    }
    needed = {f"csrc/{rel}" for record in MODULES.values() for rel in record["sources"]}
    needed |= {
        "csrc/cake_nvfp4_per_token/cake_nvfp4_per_token_device_common.cuh",
        "csrc/cake_nvfp4_per_token/cake_nvfp4_per_token_host_common.cuh",
    }
    assert needed <= shipped, sorted(needed - shipped)


_PROGRAM_SHIM_FLAG = "-DCAKE_PROGRAM_SHIM="
_PROGRAM_SHIM_DEFAULT = re.compile(r"^#define CAKE_PROGRAM_SHIM (\w+)$", re.M)
_PROGRAM_EXPORT_LINE = "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, CAKE_PROGRAM_SHIM::Run);"
_BINDING_NAMESPACE = re.compile(r"^namespace (\w+) \{$", re.M)


def _check_conditions(text):
    # The condition of every TVM_FFI_CHECK( in ``text`` as the preprocessor
    # stringizes it: balanced parentheses, string literals skipped, multi-line.
    conditions = []
    for match in re.finditer(r"\bTVM_FFI_CHECK\(", text):
        start = match.end()
        depth, quoted, i = 1, False, start
        while depth and i < len(text):
            char = text[i]
            if quoted:
                if char == "\\":
                    i += 1
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            i += 1
        assert depth == 0, text[start : start + 80]
        conditions.append(text[start : i - 1])
    return conditions


def test_shared_binding_sources_keep_distinct_namespaces():
    # Every record exports ``run`` from a host-shim namespace no other record uses,
    # declares and launches its own kernel symbol, and gives the binding only knob
    # values its kernel source declares (plus the launch-block knob of the shared
    # bf16 quantizer binding, which must spell the kernel's thread count).  A
    # binding is in one of two forms:
    # - shared form: the text spells the kernel and the namespace through
    #   ``CAKE_PROGRAM_KERNEL`` / ``CAKE_PROGRAM_SHIM``; the record whose source it
    #   is selects both through the file's defaults, every other record names its
    #   own on the compile line.  Every binding a second record compiles is in this
    #   form.
    # - own form (a rendered program): the text names the record's kernel and its
    #   own namespace ``cake_binding_kernel_<record>`` literally, only that record
    #   compiles it, and its compile line names neither macro.
    csrc = Path(cb.__file__).resolve().parent / "csrc"
    users = {}
    for name, record in MODULES.items():
        users.setdefault(record["sources"][1], []).append(name)
    namespaces = {}
    for name, record in MODULES.items():
        binding_source = record["sources"][1]
        assert binding_source.endswith("_binding.cu"), (name, binding_source)
        text = (csrc / binding_source).read_text()
        opened = _BINDING_NAMESPACE.findall(text)
        assert len(opened) == 1, (name, opened)
        # TVM_FFI_CHECK stringizes its condition unexpanded: a kernel symbol in
        # a check condition would be spelt as the macro name in the message.
        for condition in _check_conditions(text):
            assert "CAKE_PROGRAM_" not in condition, (name, condition)
        shim_flags = [
            f for f in record["compile_flags"] if f.startswith(_PROGRAM_SHIM_FLAG)
        ]
        kernel_flags = [
            f for f in record["compile_flags"] if f.startswith(_PROGRAM_KERNEL_FLAG)
        ]
        assert len(shim_flags) <= 1 and len(kernel_flags) <= 1, (name, record)
        if opened == ["CAKE_PROGRAM_SHIM"]:
            assert 'extern "C" __global__ void CAKE_PROGRAM_KERNEL(' in text, name
            assert _PROGRAM_EXPORT_LINE in text, name
            shim_defaults = _PROGRAM_SHIM_DEFAULT.findall(text)
            assert len(shim_defaults) == 1, (name, shim_defaults)
            namespace = shim_flags[0][len(_PROGRAM_SHIM_FLAG) :] if shim_flags else None
            if name in binding_source:
                assert namespace in (None, shim_defaults[0]), (name, namespace)
                namespace = shim_defaults[0]
            else:
                assert namespace is not None, (name, record["compile_flags"])
                assert kernel_flags == [f"{_PROGRAM_KERNEL_FLAG}kernel_{name}"], name
            assert re.fullmatch(r"cake_host_shim_[0-9a-f]{16}", namespace), name
        else:
            namespace = opened[0]
            assert users[binding_source] == [name], (name, users[binding_source])
            assert name in binding_source, (name, binding_source)
            assert not shim_flags and not kernel_flags, (name, record["compile_flags"])
            assert namespace == f"cake_binding_kernel_{name}", (name, namespace)
            assert f'extern "C" __global__ void kernel_{name}(' in text, name
            export_line = f"TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, {namespace}::Run);"
            assert export_line in text, name
            assert "CAKE_PROGRAM_" not in text, name
        namespaces[name] = namespace
        kernel_text = (csrc / record["sources"][0]).read_text()
        binding_knobs = {
            knob: value
            for knob, value in _KNOB_DEFAULT.findall(text)
            if not knob.startswith("CAKE_PROGRAM_")
        }
        kernel_knobs = dict(_KNOB_DEFAULT.findall(kernel_text))
        assert set(binding_knobs) <= set(kernel_knobs) | {"CAKE_Q_BLOCK"}, name
        if "CAKE_Q_BLOCK" in binding_knobs:
            given = dict(
                f[2:].split("=", 1)
                for f in record["compile_flags"]
                if f.startswith("-DCAKE_Q_")
            )
            threads = given.get("CAKE_Q_THREADS", kernel_knobs["CAKE_Q_THREADS"])
            block = given.get("CAKE_Q_BLOCK", binding_knobs["CAKE_Q_BLOCK"])
            assert block == f"{threads}u", (name, block, threads)
    assert len(set(namespaces.values())) == len(MODULES)


# ---------------------------------------------------------------------------
# GPU correctness
# ---------------------------------------------------------------------------


def _require_program():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    device = torch.device("cuda", 0)
    if not cb.generated_program_available(device):
        pytest.skip("no generated per-token NVFP4 program registered for this GPU")
    return device


def _cute_dsl_peer():
    try:
        from flashinfer.cute_dsl import is_cute_dsl_available

        if not is_cute_dsl_available():
            return None
        from flashinfer.quantization.kernels.nvfp4_quantize import (
            nvfp4_quantize_per_token_cute_dsl,
        )
    except Exception:  # noqa: BLE001 - the peer is optional evidence
        return None
    return nvfp4_quantize_per_token_cute_dsl


def _reference(x, gs_inv, out_scale=None):
    """FP32 recipe (rounding-mode agnostic): scaled values, E4M3, per-token scales."""
    xf = x.float()
    m, k = xf.shape
    row_amax = xf.abs().amax(dim=1)
    token_scale = row_amax * gs_inv.float().reshape(())
    encode = torch.where(
        row_amax == 0, torch.full_like(token_scale, FLT_MAX), 1.0 / token_scale
    )
    token_scale = torch.where(row_amax == 0, torch.zeros_like(token_scale), token_scale)
    blocks = xf.view(m, k // 16, 16)
    block_max = blocks.abs().amax(dim=2)
    sf = encode[:, None] * (block_max / 6.0)
    sf_e4m3 = sf.to(torch.float8_e4m3fn)
    sf_dec = sf_e4m3.float()
    output_scale = torch.where(
        sf_dec == 0, torch.zeros_like(sf_dec), 1.0 / (sf_dec / encode[:, None])
    )
    scaled = blocks * output_scale[:, :, None]
    if out_scale is not None:
        token_scale = token_scale * out_scale.float().reshape(())
    return scaled.reshape(m, k), sf_e4m3, token_scale


def _check_against_reference(x, gs_inv, out_scale, fp4, sf, scale):
    """Recipe check that is agnostic to the kernels' FP32 rounding.

    Both the cake and the CuTe-DSL quantizer form the per-token encode scale with
    ``rcp.approx.ftz`` and multiply in a different association than the FP32
    reference, so a block scale that sits within one FP32 ulp of an E4M3 rounding
    tie may land one E4M3 step away from the reference (about 1e-6 of the blocks
    on random data).  The kernel is held to the reference within one E4M3 ulp on
    the block scales and the FP4 codes are checked against the values the
    kernel's own block scales imply; bitwise agreement with the CuTe-DSL peer is
    asserted separately by the caller.
    """
    m, k = x.shape
    scaled_ref, sf_ref, scale_ref = _reference(x, gs_inv, out_scale)
    torch.testing.assert_close(scale, scale_ref, atol=1e-2, rtol=1e-2)
    logical = cb.sf_logical_offsets(cb.padded_rows(m), cb.padded_sf_cols(k), x.device)
    sf_logical = sf.reshape(-1)[logical]
    assert int(sf_logical[m:].sum()) == 0, "padding rows are not zero"
    assert int(sf_logical[:, k // 16 :].sum()) == 0, "padding columns are not zero"
    sf_kernel = sf_logical[:m, : k // 16].view(torch.float8_e4m3fn).float()
    sf_ref = sf_ref.float()
    e4m3_ulp = torch.exp2(torch.floor(torch.log2(sf_ref.clamp_min(2.0**-9))) - 3)
    sf_gap = (sf_kernel - sf_ref).abs()
    assert bool((sf_gap <= e4m3_ulp * 1.001).all()), (
        f"{int((sf_gap > e4m3_ulp * 1.001).sum())} block scales differ from the "
        "reference by more than one E4M3 ulp"
    )
    off_tie = int((sf_gap > 0).sum())
    assert off_tie <= max(8, m * (k // 16) // 10000), (
        f"{off_tie} block scales differ from the reference (one E4M3 step each)"
    )
    # FP4 codes: the kernel's own block scales define the quantisation grid.
    xf = x.float()
    row_amax = xf.abs().amax(dim=1)
    token_scale = row_amax * gs_inv.float().reshape(())
    encode = torch.where(
        row_amax == 0, torch.full_like(token_scale, FLT_MAX), 1.0 / token_scale
    )
    out_sc = torch.where(
        sf_kernel == 0, torch.zeros_like(sf_kernel), 1.0 / (sf_kernel / encode[:, None])
    )
    ref_codes = (xf.view(m, k // 16, 16) * out_sc[:, :, None]).view(m, k).clamp(-6, 6)
    nib = torch.stack([fp4 & 0xF, fp4 >> 4], dim=-1).reshape(m, k).long()
    e2m1 = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=x.device,
    )
    decoded = e2m1[nib]
    quantum = torch.where(
        ref_codes.abs() < 2, 0.5, torch.where(ref_codes.abs() < 4, 1.0, 2.0)
    )
    assert bool(((decoded - ref_codes).abs() <= quantum * 1.001).all())


# The validated matrix (validated_problems in tests/test_helpers/
# cake_nvfp4_per_token.py): bf16 activations on
# every K x M x fold; fp16 activations on the validated fp16 rows of K=7168 only.
_QUANT_CASES = (
    [
        (m, k, torch.bfloat16, fold)
        for fold in (False, True)
        for k in (7168, 16384)
        for m in (1, 17, 130, 257, 4097)
    ]
    + [(m, 7168, torch.float16, False) for m in VALIDATED_F16_INPUT_ROWS]
    # Row lengths outside the measured families and the token counts between tiles.
    + [
        (m, k, torch.bfloat16, False)
        for k in (2688, 4096)
        for m in (1, 9, 16, 130, 2048)
    ]
)


@pytest.mark.parametrize("m,k,dtype,fold", _QUANT_CASES)
def test_quantize_matches_reference_and_cute_dsl(m, k, dtype, fold):
    device = _require_program()
    g = torch.Generator(device=device).manual_seed(1000 + m + k)
    x = torch.randn(m, k, device=device, dtype=dtype, generator=g)
    gs_inv = torch.tensor([GLOBAL_SCALE_INV], dtype=torch.float32, device=device)
    out_scale = (
        torch.tensor([0.37], dtype=torch.float32, device=device) if fold else None
    )
    fp4, sf, scale = nvfp4_quantize(
        x,
        gs_inv,
        sfLayout=SfLayout.layout_128x4,
        per_token_activation=True,
        backend="cake",
        out_scale=out_scale,
    )
    torch.cuda.synchronize()
    assert fp4.shape == (m, k // 2) and fp4.dtype == torch.uint8
    assert sf.shape == (cb.padded_rows(m), cb.padded_sf_cols(k))
    assert sf.dtype == torch.uint8
    assert scale.shape == (m,) and scale.dtype == torch.float32
    _check_against_reference(x, gs_inv, out_scale, fp4, sf, scale)
    peer = _cute_dsl_peer()
    if peer is not None:
        p_fp4, p_sf, p_scale = peer(x, gs_inv, 0, None, out_scale)
        torch.cuda.synchronize()
        assert torch.equal(p_fp4, fp4)
        assert torch.equal(p_sf.reshape(-1), sf.reshape(-1))
        assert torch.equal(p_scale, scale)


# The quantizer alone's cluster-split cells (QUANT_ALONE_CLUSTER_SPLIT) of every architecture,
# with and without the folded out_scale.
_ALONE_SPLIT_CASES = sorted(
    {
        (k, m, fold)
        for table in cb.QUANT_ALONE_CLUSTER_SPLIT.values()
        for k, ranges in table.items()
        for lo, hi, _ in ranges
        for m in sorted({lo, hi})
        for fold in (False, True)
    }
)


@pytest.mark.parametrize("k,m,fold", _ALONE_SPLIT_CASES)
def test_quantizer_alone_split_matches_the_chain_quantizer(k, m, fold):
    """The public quantizer's cluster split writes the same codes, block scales and
    per-token scales as the quantize + GEMM chain's quantizer, bit for bit."""
    device = _require_program()
    g = torch.Generator(device=device).manual_seed(3000 + m + k)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16, generator=g)
    gs_inv = torch.tensor([GLOBAL_SCALE_INV], dtype=torch.float32, device=device)
    out_scale = (
        torch.tensor([0.37], dtype=torch.float32, device=device) if fold else None
    )
    got = cb.allocate_nvfp4_per_token_quantize_outputs(m, k, device)
    want = cb.allocate_nvfp4_per_token_quantize_outputs(m, k, device)
    for t in (*got, *want):
        t.view(torch.uint8).fill_(0xFF)
    cb.prepare_nvfp4_per_token_quantize(x, gs_inv, got, out_scale=out_scale)()
    cb._prepare_quantize(x, gs_inv, want, out_scale=out_scale, alone=False)()
    torch.cuda.synchronize()
    for a, b in zip(got, want, strict=True):
        assert torch.equal(a.reshape(-1), b.reshape(-1))
    _check_against_reference(x, gs_inv, out_scale, *got)


# K = 2688 / 4096 are covered at M = 1 only: a single row has no stride, so it keeps the
# compiled-in instance.  Multi-row strided sets at those widths fall back to a runtime-geometry
# program the package does not carry -- see
# test_row_strided_activation_at_a_new_width_names_its_missing_program.
@pytest.mark.parametrize("k,m", [(7168, 130), (7168, 8), (2688, 1), (4096, 1)])
def test_quantize_reads_row_strided_activations_in_place(k, m):
    device = _require_program()
    g = torch.Generator(device=device).manual_seed(500 + m + k)
    full = torch.randn(m, k + 64, device=device, dtype=torch.bfloat16, generator=g)
    x = full[:, :k]  # row stride k + 64 elements
    assert m == 1 or not x.is_contiguous()  # a single row is contiguous at any stride
    gs_inv = torch.tensor([GLOBAL_SCALE_INV], dtype=torch.float32, device=device)
    want = nvfp4_quantize(
        x.contiguous(), gs_inv, per_token_activation=True, backend="cake"
    )
    torch.cuda.synchronize()
    data_ptr = full.data_ptr()
    got = nvfp4_quantize(x, gs_inv, per_token_activation=True, backend="cake")
    torch.cuda.synchronize()
    assert full.data_ptr() == data_ptr
    for a, b in zip(got, want, strict=True):
        assert torch.equal(a.reshape(-1), b.reshape(-1))
    _check_against_reference(x, gs_inv, None, *got)


def test_quantize_repeated_call_allocates_only_its_outputs():
    device = _require_program()
    m, k = 257, 7168
    g = torch.Generator(device=device).manual_seed(11)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16, generator=g)
    call = lambda: nvfp4_quantize(  # noqa: E731
        x, GLOBAL_SCALE_INV, per_token_activation=True, backend="cake"
    )
    call()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    outputs = call()
    torch.cuda.synchronize()
    # fp4 codes, block scales and per-token scales: nothing else per call.
    assert torch.cuda.memory_stats()["allocation.all.allocated"] - before == 3
    gs_inv = torch.tensor([GLOBAL_SCALE_INV], dtype=torch.float32, device=device)
    _check_against_reference(x, gs_inv, None, *outputs)


def test_prepared_runner_graph_replay_and_no_allocation():
    device = _require_program()
    m, k = 130, 7168
    g = torch.Generator(device=device).manual_seed(7)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16, generator=g)
    gs_inv = torch.tensor([GLOBAL_SCALE_INV], dtype=torch.float32, device=device)
    outputs = cb.allocate_nvfp4_per_token_quantize_outputs(m, k, device)
    runner = cb.prepare_nvfp4_per_token_quantize(x, gs_inv, outputs)
    assert runner.launch_count == 1
    runner()
    torch.cuda.synchronize()
    eager = tuple(t.clone() for t in outputs)
    before = torch.cuda.memory_stats()
    runner()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] - before["allocation.all.allocated"] == 0
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            runner()
    torch.cuda.synchronize()
    for _round_index in range(2):
        x.copy_(torch.randn(m, k, device=device, dtype=torch.bfloat16, generator=g))
        for t in outputs:
            t.view(torch.uint8).fill_(0xFF)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        expected = nvfp4_quantize(x, gs_inv, per_token_activation=True, backend="cake")
        torch.cuda.synchronize()
        for got, want in zip(outputs, expected, strict=True):
            assert torch.equal(got.reshape(-1), want.reshape(-1))
    del eager


def test_rejections():
    device = _require_program()
    x = torch.randn(8, 7168, device=device, dtype=torch.bfloat16)
    gs_inv = torch.tensor([GLOBAL_SCALE_INV], dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="per_token_activation=True only"):
        nvfp4_quantize(x, gs_inv, backend="cake")
    with pytest.raises(ValueError, match="128x4"):
        nvfp4_quantize(
            x,
            gs_inv,
            sfLayout=SfLayout.layout_8x4,
            per_token_activation=True,
            backend="cake",
        )
    with pytest.raises(ValueError, match="128x4"):
        nvfp4_quantize(
            x, gs_inv, do_shuffle=True, per_token_activation=True, backend="cake"
        )
    with pytest.raises(ValueError, match="dependent launch"):
        nvfp4_quantize(
            x, gs_inv, per_token_activation=True, backend="cake", enable_pdl=False
        )
    with pytest.raises(ValueError, match="4over6"):
        nvfp4_quantize(
            x,
            gs_inv,
            per_token_activation=True,
            backend="cake",
            nvfp4_4over6=NVFP44Over6Config(),
        )
    outputs = cb.allocate_nvfp4_per_token_quantize_outputs(4, 7168, device)
    with pytest.raises(ValueError, match="fp4"):
        cb.prepare_nvfp4_per_token_quantize(x, gs_inv, outputs)
