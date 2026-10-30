"""Source tests (no GPU) of the sm_107a wrap-plan programs
``exactpre_{bf16,f32}_varlen_wrap`` and their route (:func:`wrap_long_selected`)."""

import hashlib
import importlib
import inspect
import math
import re
from pathlib import Path

import pytest
import torch

STATES = {"bf16": torch.bfloat16, "f32": torch.float32}


def _module():
    return importlib.import_module("flashinfer.mamba.cake_ssd_combined")


def _device_dir():
    return (
        Path(_module().__file__).resolve().parents[2]
        / "csrc"
        / "cake_mamba_ssd_combined"
        / "generated"
        / "device"
    )


@pytest.mark.cpu_only
def test_source_wrap_programs_are_sm107_only_tailone_pools():
    """Two programs (bf16 / f32): sm_107a-only table entries, the one-CTA pair
    program's SMEM pool, no cluster-dims entry, PDL main launch."""

    module = _module()
    for state, dtype in STATES.items():
        name = f"exactpre_{state}_varlen_wrap"
        assert name in module._SM107_MODULES and name in module._SM107_ONLY_PROGRAMS
        program = module._PROGRAMS[name]
        assert program.family == "exactpre" and program.variant == "wrap"
        assert program.main_smem_bytes == module._EXACTPRE_TAILONE_SMEM_BYTES == 232448
        assert program.cluster_dims == (1, 1, 1)
        module_name = module._SM107_MODULES[name]
        assert re.fullmatch(
            rf"mamba_ssd_q_tmem_alias_{state}_varlen_precomputed_wrap_[0-9a-f]{{10}}",
            module_name,
        )
        assert program.main.kernel == (
            f"kernel_mamba_ssd_q_tmem_alias_{state}_varlen_precomputed_wrap"
        )
        built = module._arch_program(name, module._SM107_ARCH)
        assert built.main.module == module_name
        assert built.main_smem_bytes == 232448
        assert module._main_pdl(name, module._SM107_ARCH)
        assert module._program_name("exactpre", dtype, True, "wrap") == name
    assert "exactpre_f16_varlen_wrap" not in module._PROGRAMS


@pytest.mark.cpu_only
def test_source_wrap_units_match_their_identity():
    """Each unit exists, its sha256 prefix is its module identity, it defines
    its kernel symbol once with the 1-CTA cluster attribute, and every CTA
    draws its slot from status word 2 (no pair ticket words)."""

    module = _module()
    for state in STATES:
        module_name = module._SM107_MODULES[f"exactpre_{state}_varlen_wrap"]
        data = (_device_dir() / f"{module_name}.cu").read_bytes()
        assert hashlib.sha256(data).hexdigest()[:10] == module_name[-10:]
        text = data.decode()
        symbol = f"kernel_mamba_ssd_q_tmem_alias_{state}_varlen_precomputed_wrap("
        assert text.count("\n" + symbol) == 1
        assert text.count("__cluster_dims__(1,1,1)") == 1
        assert "&preprocess_status[2]" in text
        assert "2 + num_bids / 2" not in text


# Host-known geometry of 212-SM rows (label, items, host chunks per item,
# has z, admitted by the wrap rule): gate rows, long-item rows and the rows
# the rule rejects (makespan, 8 or 15 chunks per item, z).
ROWS = (
    ("perf_varlen_32x32", 256, 32, False, True),
    ("perf_varlen_32x8", 256, 8, False, False),
    ("nem_v1024x2", 256, 8, True, False),
    ("perf_varlen_64x8", 512, 8, False, False),
    ("perf_varlen_64x32", 512, 32, False, False),
    ("perf_varlen_128x32", 1024, 32, False, False),
    ("nem_v128p896", 256, 4, True, False),
    ("perf_varlen_16x8", 128, 8, False, False),
    ("nem_v2048", 128, 16, True, False),
    ("perf_varlen_64x1", 512, 1, False, False),
    ("nem_2x2048_z", 256, 16, True, False),
    ("nem_2x2048", 256, 16, False, True),
    ("pv_c16_s32", 256, 16, False, True),
    ("pv_c32_s40", 320, 32, False, True),
    ("pv_c32_s53", 424, 32, False, False),
    ("mix_16x1_16x15", 256, 8, False, False),
    ("h8_n27_t2000", 216, 16, False, True),
    ("h16_n15_t3000", 240, 24, False, True),
    ("h32_n8_t4000", 256, 32, False, True),
    ("h64_n4_t6100", 256, 48, False, True),
    ("h128_n2_t8100", 256, 64, False, True),
    ("h8_n56_t1950", 448, 16, False, True),
    ("h16_n20_t4096", 320, 32, False, True),
    ("h32_n17_t3050", 544, 24, False, True),
    ("h64_n10_t2040", 640, 16, False, True),
    ("h128_n7_t3000", 896, 24, False, True),
    ("h8_n137_t3000", 1096, 24, False, True),
    ("h16_n28_t6100", 448, 48, False, True),
    ("h64_n5_t4000", 320, 32, False, True),
    ("h32_n7_t8150", 224, 64, False, True),
    ("rand_h8_n30", 240, 38, False, True),
    ("rand_h32_n11", 352, 20, False, True),
    ("h16_n14_t2047", 224, 16, False, True),
    ("h128_n2_t4000", 256, 32, False, True),
    ("cpi15", 256, 15, False, False),
    ("z", 320, 32, True, False),
    ("makespan", 296, 32, False, False),
)


def _tail(module, items, chunks, has_z, sm_count=212):
    return module.tail_selected(
        total_work=items,
        chunks_per_item=chunks,
        sm_count=sm_count,
        mode_varlen=True,
        full_grid_rule=module.tail_full_grid_rule((10, 7)),
        has_z=has_z,
    )


@pytest.mark.cpu_only
def test_source_wrap_rule_on_the_rows():
    """On 212 SMs the rule admits exactly the rows above marked so; the
    15-chunk and z neighbours are makespan-rule rows the long-item and z
    conditions reject."""

    module = _module()
    assert module._WRAP_MIN_CHUNKS == 8 and module._WRAP_MAX_MAKESPAN_RATIO == 0.9
    assert module._WRAP_LONG_CHUNKS == 16
    for label, items, chunks, has_z, admitted in ROWS:
        tail = _tail(module, items, chunks, has_z)
        for dtype in STATES.values():
            got = module.wrap_long_selected(
                capability=(10, 7),
                state_dtype=dtype,
                mode_varlen=True,
                total_work=items,
                chunks_per_item=chunks,
                sm_count=212,
                tail=tail,
                has_z=has_z,
            )
            assert got == admitted, (label, dtype)
    for items, chunks, has_z in ((256, 15, False), (320, 32, True)):
        assert module.wrap_selected(
            capability=(10, 7),
            state_dtype=torch.bfloat16,
            mode_varlen=True,
            total_work=items,
            chunks_per_item=chunks,
            sm_count=212,
            tail=_tail(module, items, chunks, has_z),
        )


@pytest.mark.cpu_only
def test_source_wrap_rule_is_the_formula():
    """``wrap_selected`` is the makespan formula and ``wrap_long_selected``
    adds >= 16 host chunks per item and no z, over a grid of host inputs."""

    module = _module()
    for sm_count in (2, 3, 132, 148, 160, 211, 212):
        for items in (1, 2, 128, 212, 213, 216, 256, 296, 320, 384, 424, 512, 1024):
            for chunks in (1, 2, 7, 8, 9, 15, 16, 17, 24, 32, 64):
                for tail in (False, True):
                    ok = items > sm_count and chunks >= 8
                    if ok:
                        wrap = math.ceil(items * chunks / sm_count)
                        if tail:
                            cur = math.ceil(items / (sm_count // 2)) / 2 * chunks
                        else:
                            cur = math.ceil(items / sm_count) * chunks
                        ok = wrap <= 0.9 * cur
                    for dtype in (torch.bfloat16, torch.float32, torch.float16):
                        kw = dict(
                            capability=(10, 7),
                            state_dtype=dtype,
                            mode_varlen=True,
                            total_work=items,
                            chunks_per_item=chunks,
                            sm_count=sm_count,
                            tail=tail,
                        )
                        want = ok and dtype != torch.float16
                        assert module.wrap_selected(**kw) == want, kw
                        for has_z in (False, True):
                            assert module.wrap_long_selected(**kw, has_z=has_z) == (
                                want and chunks >= 16 and not has_z
                            ), (kw, has_z)
    kw = dict(
        state_dtype=torch.bfloat16,
        total_work=256,
        chunks_per_item=32,
        sm_count=212,
        tail=True,
        has_z=False,
    )
    assert module.wrap_long_selected(capability=(10, 7), mode_varlen=True, **kw)
    assert not module.wrap_long_selected(capability=(10, 7), mode_varlen=False, **kw)
    for cap in (None, (10, 0), (10, 3), (9, 0), (12, 0)):
        assert not module.wrap_long_selected(capability=cap, mode_varlen=True, **kw)


@pytest.mark.cpu_only
def test_source_wrap_route_and_grid_placement():
    """The route sits after the claim / ibulk rules and the pair rule and
    before the one-CTA pair form, reads only the exactpre ring / pair / ibulk
    choice; the wrap programs launch one CTA per SM."""

    module = _module()
    assert module._WRAP_FROM_VARIANTS == ("", "tail", "ibulk")
    src = inspect.getsource(module.CakeSSDCombined._materialize_workspace)
    i_claim = src.index('program_family, variant = "exactpre", "claim"')
    i_ibulk = src.index("and ibulk_selected(")
    i_wrap = src.index("and wrap_long_selected(")
    i_one = src.index('if variant == "tail" and tail_one_cta_selected(')
    i_unit = src.index("variant = unit_cluster_variant(")
    assert i_claim < i_ibulk < i_wrap < i_one < i_unit
    assert src.count("wrap_long_selected(") == 1
    assert src.count('variant = "wrap"') == 1
    launch = inspect.getsource(module.CakeSSDCombined)
    i_grid = launch.index('elif program.variant == "wrap":')
    assert launch.index("grid = _wrap_grid_size(sm_count=sm_count)") > i_grid
    assert i_grid < launch.index('elif program.variant in ("tail", "tailone"):')
    for sm_count in (2, 3, 148, 212):
        assert module._wrap_grid_size(sm_count=sm_count) == sm_count


@pytest.mark.cpu_only
def test_source_wrap_status_words_fit_the_runner_buffer():
    """Words of a wrap launch on grid G = sm_count: 0..2 and the boundary bits
    up to word 3 + (G - 1) // 32; the runner allocates 2 + pairs +
    ceil(pairs / 32) words (112 on 212 SMs), enough for every SM count >= 2."""

    module = _module()
    for sm_count in range(2, 1025):
        pairs = module._tail_grid_size(sm_count=sm_count) // 2
        words = 2 + pairs + (pairs + 31) // 32
        grid = module._wrap_grid_size(sm_count=sm_count)
        assert 3 + (grid - 1) // 32 < words, sm_count
    pairs = module._tail_grid_size(sm_count=212) // 2
    assert 2 + pairs + (pairs + 31) // 32 == 112
    assert 3 + (212 - 1) // 32 == 9  # words 2..9 on 212 SMs
