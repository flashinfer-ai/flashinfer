"""Shared DeepGEMM-family infrastructure: catalog reader, route lookup, random-input helpers."""

import json
from pathlib import Path

import pytest

from flashinfer.experimental.deepgemm_common import (
    ARCHES,
    Catalog,
    UnsupportedDevice,
    jit_spec_name,
    load_catalog,
)

_EXPERIMENTAL = Path(__file__).resolve().parents[2] / "flashinfer" / "experimental"
_FAMILIES = ("deepgemm_", "mega_moe_v3")


def _per_arch_catalog():
    return {
        "schema": "unit.v2",
        "arches": {
            "sm_100a": {
                "programs": {
                    "k_sm100a_aa": {"sources": ["csrc/x/a.cu"], "compile_flags": []},
                },
                "routes": {
                    "1:2:148": {"config": {"num_sms": 148}, "program": "k_sm100a_aa"},
                },
            },
            "sm_103a": {
                "programs": {
                    "k_sm103a_bb": {"sources": ["csrc/x/b.cu"], "compile_flags": []},
                },
                "routes": {
                    "1:2:152": {"num_sms": 152, "program": "k_sm103a_bb"},
                    "1:4:152": {"num_sms": 152, "program": "k_sm103a_bb"},
                },
            },
        },
    }


def _merged_catalog():
    return {
        "schema": "unit.merged.v1",
        "programs": {
            "k_shared": {
                "sources": ["csrc/x/shared.cu"],
                "compile_flags": [],
                "arches": ["sm_100a", "sm_103a"],
            },
            "k_only_103": {
                "sources": ["csrc/x/only.cu"],
                "compile_flags": [],
                "arches": ["sm_103a"],
            },
        },
        "routes": {
            "sm_100a": {"a:sm148": {"program": "k_shared"}},
            "sm_103a": {
                "a:sm152": {"program": "k_shared"},
                "b:sm152": {"program": "k_only_103"},
            },
        },
    }


def _merged_catalog_missing_arch_routes():
    """Merged catalog where a program exports sm_100a but the arch-keyed
    routes table carries no sm_100a entry at all (malformed catalog)."""
    return {
        "schema": "unit.merged.v1",
        "programs": {
            "k_shared": {
                "sources": ["csrc/x/shared.cu"],
                "compile_flags": [],
                "arches": ["sm_100a", "sm_103a"],
            },
        },
        "routes": {
            "sm_103a": {"a:sm152": {"program": "k_shared"}},
        },
    }


def _merged_catalog_shared_routes():
    """Merged catalog whose routes are one flat table shared by every arch
    (not keyed by architecture at all)."""
    return {
        "schema": "unit.merged.v2",
        "programs": {
            "k_shared": {
                "sources": ["csrc/x/shared.cu"],
                "compile_flags": [],
                "arches": ["sm_100a", "sm_103a"],
            },
        },
        "routes": {
            "a:sm148": {"program": "k_shared"},
        },
    }


def test_arches_cover_only_the_exported_capabilities():
    assert ARCHES == {(10, 0): "sm_100a", (10, 3): "sm_103a"}


def test_per_arch_layout_programs_routes_and_sm_counts():
    catalog = Catalog(_per_arch_catalog(), label="Unit")
    assert not catalog.merged
    assert catalog.arches == ("sm_100a", "sm_103a")
    assert list(catalog.programs("sm_100a")) == ["k_sm100a_aa"]
    assert catalog.program("sm_103a", "k_sm103a_bb")["sources"] == ["csrc/x/b.cu"]
    # ``config.num_sms`` and top-level ``num_sms`` are both read.
    assert catalog.supported_num_sms("sm_100a") == (148,)
    assert catalog.supported_num_sms("sm_103a") == (152,)
    assert catalog.route("sm_103a", "1:4:152")["program"] == "k_sm103a_bb"


def test_merged_layout_filters_programs_by_arch():
    catalog = Catalog(
        _merged_catalog(),
        label="Unit",
        route_num_sms=lambda key, route: int(key.rsplit(":sm", 1)[1]),
    )
    assert catalog.merged
    assert catalog.arches == ("sm_100a", "sm_103a")
    assert list(catalog.programs("sm_100a")) == ["k_shared"]
    assert sorted(catalog.programs("sm_103a")) == ["k_only_103", "k_shared"]
    assert catalog.supported_num_sms("sm_100a") == (148,)
    assert catalog.supported_num_sms("sm_103a") == (152,)
    with pytest.raises(KeyError, match="no program 'k_only_103' for sm_100a"):
        catalog.program("sm_100a", "k_only_103")


def test_key_encoded_sm_count_needs_an_explicit_reader():
    catalog = Catalog(_merged_catalog(), label="Unit")
    with pytest.raises(ValueError, match="carries no SM count"):
        catalog.supported_num_sms("sm_100a")


def test_merged_layout_keyed_routes_missing_an_arch_raises():
    # sm_100a is exported (a program lists it in "arches"), but the
    # arch-keyed routes table has no sm_100a key at all: this must raise
    # rather than silently hand back the outer {arch: routes} mapping as if
    # it were itself a route table keyed by route key.
    catalog = Catalog(_merged_catalog_missing_arch_routes(), label="Unit")
    assert catalog.arches == ("sm_100a", "sm_103a")
    with pytest.raises(
        ValueError,
        match="lists sm_100a as exported but its arch-keyed routes carry no entry",
    ):
        catalog.routes("sm_100a")
    # The present arch still resolves to its own, correctly-scoped routes.
    assert catalog.routes("sm_103a") == {"a:sm152": {"program": "k_shared"}}


def test_merged_layout_shared_routes_are_returned_verbatim_for_every_arch():
    # Top-level route keys that are not architecture names mean every
    # architecture shares the same flat route table.
    catalog = Catalog(_merged_catalog_shared_routes(), label="Unit")
    shared = {"a:sm148": {"program": "k_shared"}}
    assert catalog.routes("sm_100a") == shared
    assert catalog.routes("sm_103a") == shared


def _merged_catalog_shared_routes_per_arch_sm_counts():
    return {
        "schema": "unit.v3",
        "programs": {
            "k_shared": {
                "sources": ["csrc/x/shared.cu"],
                "compile_flags": {"sm_100a": ["-a"], "sm_103a": ["-a", "-b"]},
                "definitions": ["NUM_CTAS"],
                "arches": ["sm_100a", "sm_103a"],
            },
            "k_plain": {
                "sources": ["csrc/x/plain.cu"],
                "compile_flags": ["-c"],
                "arches": ["sm_100a", "sm_103a"],
            },
        },
        "routes": {
            "a": {
                "num_sms": {"sm_100a": 148, "sm_103a": 152},
                "program": "k_shared",
            },
        },
    }


def test_shared_routes_carry_one_sm_count_per_arch_and_per_arch_flags():
    catalog = Catalog(_merged_catalog_shared_routes_per_arch_sm_counts(), label="Unit")
    assert catalog.supported_num_sms("sm_100a") == (148,)
    assert catalog.supported_num_sms("sm_103a") == (152,)
    route = catalog.route("sm_103a", "a")
    assert catalog.route_num_sms("sm_103a", "a", route) == 152
    with pytest.raises(ValueError, match="carries no SM count for sm_90a"):
        catalog.route_num_sms("sm_90a", "a", route)
    shared = catalog.program("sm_100a", "k_shared")
    assert Catalog.compile_flags(shared, "sm_100a") == ["-a"]
    assert Catalog.compile_flags(shared, "sm_103a") == ["-a", "-b"]
    assert Catalog.compile_flags(catalog.program("sm_100a", "k_plain"), "sm_100a") == [
        "-c"
    ]


def test_definitions_are_exactly_the_declared_subset_and_name_the_spec():
    catalog = Catalog(_merged_catalog_shared_routes_per_arch_sm_counts(), label="Unit")
    assert catalog.definitions(
        "sm_100a", "k_shared", {"NUM_CTAS": 148, "OTHER": 1}
    ) == {"NUM_CTAS": 148}
    assert catalog.definitions("sm_100a", "k_plain", {"NUM_CTAS": 148}) == {}
    with pytest.raises(ValueError, match=r"cannot supply: \['NUM_CTAS'\]"):
        catalog.definitions("sm_100a", "k_shared", {})
    pytest.importorskip("torch")
    spec = catalog.jit_spec("sm_103a", "k_shared", {"NUM_CTAS": 152})
    assert spec.name == "k_shared_sm_103a_num_ctas152"
    flags = spec.extra_cuda_cflags
    assert "-DNUM_CTAS=152" in flags and "-a" in flags and "-b" in flags
    plain = catalog.jit_spec("sm_100a", "k_plain")
    assert plain.name == "k_plain_sm_100a" and "-c" in plain.extra_cuda_cflags


def test_missing_route_raises_unsupported_device_naming_the_options():
    catalog = Catalog(_per_arch_catalog(), label="Unit GEMM")
    options = {"M": 1, "N": 2, "num_sms": 132}
    with pytest.raises(UnsupportedDevice) as info:
        catalog.route("sm_100a", "1:2:132", options=options)
    message = str(info.value)
    assert "Unit GEMM" in message and "132" in message and "[148]" in message
    assert isinstance(info.value, NotImplementedError)


def test_unknown_arch_and_capability_errors_name_the_catalog():
    catalog = Catalog(_per_arch_catalog(), label="Unit GEMM")
    with pytest.raises(KeyError, match="no architecture 'sm_90a'"):
        catalog.routes("sm_90a")
    assert catalog.arch_for_capability((10, 3)) == "sm_103a"
    with pytest.raises(
        RuntimeError,
        match=r"Unit GEMM has no exported programs for compute capability \(12, 0\)",
    ):
        catalog.arch_for_capability((12, 0))
    with pytest.raises(ValueError, match="no 'schema'"):
        Catalog({"arches": {}}, label="Unit")


def test_jit_spec_name_keeps_per_arch_names_and_suffixes_neutral_ones():
    assert (
        jit_spec_name("deepgemm_x_sm100a_0123", "sm_100a") == "deepgemm_x_sm100a_0123"
    )
    assert (
        jit_spec_name("deepgemm_x_sm103a_0123", "sm_103a") == "deepgemm_x_sm103a_0123"
    )
    assert jit_spec_name("deepgemm_x_0123", "sm_103a") == "deepgemm_x_0123_sm_103a"


def test_load_catalog_is_cached_per_path(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(_per_arch_catalog()))
    first = load_catalog(path, label="Unit")
    assert first is load_catalog(path, label="Unit")
    assert first.path == path and first.schema == "unit.v2"


def _tree_catalogs():
    found = []
    for family in sorted(_EXPERIMENTAL.iterdir()):
        if not family.is_dir() or not family.name.startswith(_FAMILIES):
            continue
        for path in sorted(family.glob("*catalog*.json")):
            data = json.loads(path.read_text())
            if "schema" in data and ("arches" in data or "programs" in data):
                found.append(path)
    return found


@pytest.mark.parametrize("path", _tree_catalogs(), ids=lambda p: p.parent.name)
def test_every_family_catalog_in_the_tree_loads(path):
    catalog = Catalog(json.loads(path.read_text()), label=path.parent.name, path=path)
    assert set(catalog.arches) <= set(ARCHES.values())
    for arch in catalog.arches:
        programs = catalog.programs(arch)
        assert programs, f"{path.parent.name} has no {arch} programs"
        for name, record in programs.items():
            assert record["sources"], f"{name} lists no sources"
            assert all(src.startswith("csrc/") for src in record["sources"]), name
            assert "compile_flags" in record, name
            assert jit_spec_name(name, arch) == name or catalog.merged
        assert catalog.routes(arch), f"{path.parent.name} has no {arch} routes"


class TestRandomInputs:
    @pytest.fixture(autouse=True)
    def _torch(self):
        self.torch = pytest.importorskip("torch")
        from tests.experimental import deepgemm_common as helpers

        self.h = helpers
        self.gen = helpers.seeded_generator(7, "cpu")

    def test_e2m1_codes_roundtrip_in_both_nibbles(self):
        torch = self.torch
        values = torch.tensor(self.h.E2M1_VALUES, dtype=torch.float32)
        packed = self.h.e2m1_pack(values)
        assert packed.dtype == torch.uint8 and packed.shape == (8,)
        unpacked = self.h.e2m1_unpack(packed)
        assert torch.equal(unpacked, values)
        assert torch.equal(torch.signbit(unpacked), torch.signbit(values))
        # The analytic constants of the family tests: 0x22 is (1.0, 1.0), 0xAA is (-1.0, -1.0).
        assert torch.equal(
            self.h.e2m1_unpack(torch.tensor([0x22, 0xAA], dtype=torch.uint8)),
            torch.tensor([1.0, 1.0, -1.0, -1.0]),
        )
        with pytest.raises(ValueError, match="not E2M1-representable"):
            self.h.e2m1_pack(torch.tensor([0.7, 1.0]))

    def test_ue8m0_words_roundtrip_little_endian(self):
        torch = self.torch
        exponents = torch.randint(0, 256, (8, 3), generator=self.gen)
        words = self.h.ue8m0_pack_words(exponents)
        assert words.dtype == torch.int32 and words.shape == (2, 3)
        assert torch.equal(self.h.ue8m0_unpack_words(words), exponents)
        unit = self.h.ue8m0_pack_words(torch.full((4, 1), 127))
        assert unit.item() == 0x7F7F7F7F

    def test_fp4_operand_dequantizes_through_its_scales(self):
        torch = self.torch
        packed, words, dequantized = self.h.random_fp4_operand(
            16, 256, generator=self.gen, device="cpu"
        )
        assert (
            packed.shape == (16, 128)
            and words.shape == (2, 16)
            and dequantized.shape == (16, 256)
        )
        exponents = self.h.ue8m0_unpack_words(words)  # [8, 16]
        scale = torch.exp2(exponents.float() - 127).T.repeat_interleave(32, dim=1)
        assert torch.equal(self.h.e2m1_unpack(packed) * scale, dequantized)
        assert dequantized.ne(0).float().mean() > 0.5

    def test_fp8_blockwise_operand_is_exact_under_its_scales(self):
        torch = self.torch
        fp8, scales, dequantized = self.h.random_fp8_blockwise(
            8, 512, generator=self.gen, device="cpu"
        )
        assert fp8.dtype == torch.float8_e4m3fn and scales.shape == (8, 4)
        rebuilt = fp8.float().reshape(8, 4, 128) * scales.unsqueeze(-1)
        assert torch.equal(rebuilt.reshape(8, 512), dequantized)

    def test_fp8_ue8m0_operand_shapes_follow_the_granularity(self):
        fp8, words, dequantized = self.h.random_fp8_ue8m0(
            4, 512, generator=self.gen, device="cpu", gran_k=128
        )
        assert (
            words.shape == (1, 4)
            and fp8.shape == (4, 512)
            and dequantized.shape == (4, 512)
        )
        fp8, words, _ = self.h.random_fp8_ue8m0(
            4, 512, generator=self.gen, device="cpu", gran_k=32
        )
        assert words.shape == (4, 4)

    def test_reference_gemm_matches_the_analytic_family_constants(self):
        torch = self.torch
        k = 256
        ones = self.h.e2m1_unpack(torch.full((4, k // 2), 0x22, dtype=torch.uint8))
        out = self.h.reference_gemm(ones, ones, alpha=-0.75)
        assert torch.equal(out, torch.full((4, 4), -0.75 * k))
        self.h.assert_close(out.to(torch.bfloat16), out)
        with pytest.raises(KeyError, match="no DeepGEMM-family tolerance"):
            self.h.assert_close(out.to(torch.float16), out)


def _exported_device(describe):
    """Architecture of the current device as the family's runtime resolves it
    (``describe`` returns it or raises), or a skip naming the gap. The FP4 GEMM
    and batched projection families select their programs from runtime shapes
    and device facts; they ship no per-shape catalog file."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        return describe(torch)
    except (RuntimeError, UnsupportedDevice) as error:
        pytest.skip(str(error))


class TestRandomInputReferenceChecks:
    """Random operands against a float64 reference on the generated kernels."""

    @pytest.mark.parametrize(
        "m,n,k,alpha", [(256, 128, 256, 1.0), (128, 4608, 5120, -0.75)]
    )
    def test_fp4_gemm_matches_reference_on_random_operands(self, m, n, k, alpha):
        torch = pytest.importorskip("torch")
        from tests.experimental import deepgemm_common as helpers
        from flashinfer.experimental.deepgemm_fp4_gemm import fp4_gemm as runtime
        from flashinfer.fp4_gemm import prepare_fp4_gemm

        _exported_device(
            lambda torch: runtime.device_facts(torch.cuda.current_device())
        )
        gen = helpers.seeded_generator(2026, "cuda")
        a, sfa, a_ref = helpers.random_fp4_operand(m, k, generator=gen, device="cuda")
        b, sfb, b_ref = helpers.random_fp4_operand(n, k, generator=gen, device="cuda")
        try:
            plan = prepare_fp4_gemm(a, b, sfa, sfb, m=m, alpha=alpha)
        except NotImplementedError as error:
            pytest.skip(str(error))
        plan.run()
        torch.cuda.synchronize()
        expected = helpers.reference_gemm(a_ref, b_ref, alpha=alpha)
        helpers.assert_close(plan.output, expected)

    @pytest.mark.parametrize("tokens,alpha", [(4, None), (128, 0.5)])
    def test_fp8_batched_projection_matches_reference_on_random_operands(
        self, tokens, alpha
    ):
        torch = pytest.importorskip("torch")
        from tests.experimental import deepgemm_common as helpers
        from flashinfer.experimental.deepgemm_batched_gemm import (
            batched_gemm as runtime,
        )
        from flashinfer.fp8_batched_gemm import prepare_fp8_batched_gemm

        def describe(torch):
            arch = runtime.device_arch("cuda")
            sms = torch.cuda.get_device_properties(0).multi_processor_count
            if sms not in runtime.supported_num_sms(arch):
                raise UnsupportedDevice(
                    f"Batched FP8 projection schedules are pinned to "
                    f"{runtime.supported_num_sms(arch)} SMs, this device has {sms}"
                )
            return arch

        _exported_device(describe)
        heads, inner, width = 8, 4096, 1024
        gen = helpers.seeded_generator(2026, "cuda")
        aq, asf, a_ref, bq, bsf, b_ref = [], [], [], [], [], []
        for _ in range(heads):
            q, s, ref = helpers.random_fp8_blockwise(
                tokens, inner, generator=gen, device="cuda"
            )
            aq.append(q), asf.append(s), a_ref.append(ref)
            q, s, ref = helpers.random_fp8_block2d(
                width, inner, generator=gen, device="cuda"
            )
            bq.append(q), bsf.append(s), b_ref.append(ref)
        aq = torch.stack(aq, dim=1).contiguous()  # (tokens, heads, inner)
        asf = torch.stack(asf, dim=1).contiguous()  # (tokens, heads, inner // 128)
        bq = torch.stack(bq).contiguous()  # (heads, width, inner)
        bsf = torch.stack(bsf).contiguous()  # (heads, width // 128, inner // 128)
        try:
            plan = prepare_fp8_batched_gemm(
                (aq, asf), (bq, bsf), output_fp8=False, alpha=alpha
            )
        except NotImplementedError as error:
            pytest.skip(str(error))
        plan.run()
        torch.cuda.synchronize()
        expected = torch.stack(
            [
                helpers.reference_gemm(
                    a_ref[h], b_ref[h], alpha=1.0 if alpha is None else alpha
                )
                for h in range(heads)
            ],
            dim=1,
        )  # (tokens, heads, width)
        helpers.assert_close(plan.values, expected)
