"""Tests for flashinfer.grouped_mm.grouped_mm_mxfp8.

The tests do not reset FLASHINFER_CUDNN_EXCLUDE_BAD_ENGINES. With it set to 0 on an
affected device, the scale-row tests fail and show the misread. Tests that need a
specific value set it with monkeypatch.
"""

import pytest
import torch
import torch.nn.functional as F

from flashinfer.fp8_quantization import mxfp8_quantize
from flashinfer.grouped_mm import grouped_mm_mxfp8
from flashinfer.grouped_mm.cudnn import _cudnn_moe_block_scale_min_version

from .conftest import (
    ref_grouped_mm,
    requires_cudnn_moe_block_scale,
    requires_grouped_mm_mxfp8_cc,
)


@requires_grouped_mm_mxfp8_cc
class TestGroupedMmMxfp8:
    """Tests for grouped_mm_mxfp8 (block-scaled MXFP8 grouped GEMM)."""

    MIN_COS_SIM = 0.9
    # Segment lists in which some segments do not start on a multiple of 128 rows,
    # as real routing produces. cuDNN reads token scales padded to 128 rows per group, so these
    # exercise the re-layout in _segment_token_descale_per_group.
    UNALIGNED_SEGMENTS = [
        [64, 64],
        [120, 136],
        [16, 48, 64],
        [128, 64, 64],
        [192, 64],
        [13, 83, 0, 67, 77],
    ]
    MIN_SEGMENT_COS_SIM = 0.97

    @staticmethod
    def _quantize_a(a_bf16):
        """Quantize 2D token activations (cum_m, k) → mxfp8 + scale."""
        _cum_m, k = a_bf16.shape
        a_mxfp8, a_scale = mxfp8_quantize(a_bf16, is_sf_swizzled_layout=True)
        a_scale = a_scale.reshape(-1, k // 32)
        return a_mxfp8, a_scale

    @staticmethod
    def _quantize_b(b_bf16):
        """Quantize 3D expert weights (num_experts, n, k) → mxfp8 + scale."""
        num_experts, _n, k = b_bf16.shape
        b_mxfp8, b_scale = mxfp8_quantize(b_bf16, is_sf_swizzled_layout=True)
        b_scale = b_scale.reshape(num_experts, -1, k // 32)
        return b_mxfp8, b_scale

    @requires_cudnn_moe_block_scale
    @pytest.mark.parametrize("num_experts", [1, 4, 8])
    @pytest.mark.parametrize("tokens_per_expert", [128, 256])
    @pytest.mark.parametrize("k", [128, 256, 512])
    @pytest.mark.parametrize("n", [128, 256])
    def test_uniform_distribution(self, num_experts, tokens_per_expert, k, n):
        """Each expert gets the same number of tokens."""
        torch.manual_seed(42)
        cum_m = num_experts * tokens_per_expert

        a_bf16 = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
        m_indptr = (
            torch.arange(num_experts + 1, device="cuda") * tokens_per_expert
        ).to(torch.int32)

        a_mxfp8, a_scale = self._quantize_a(a_bf16)
        b_mxfp8, b_scale = self._quantize_b(b_bf16)

        out = grouped_mm_mxfp8(
            a_mxfp8,
            b_mxfp8,
            a_scale,
            b_scale,
            m_indptr,
            out_dtype=torch.bfloat16,
        )
        ref = ref_grouped_mm(a_bf16, b_bf16, m_indptr, out_dtype=torch.bfloat16)

        cos_sim = F.cosine_similarity(
            ref.reshape(-1).float(),
            out.reshape(-1).float(),
            dim=0,
        )
        assert cos_sim > self.MIN_COS_SIM, (
            f"Cosine similarity {cos_sim:.4f} too low (expected > {self.MIN_COS_SIM})"
        )

    @requires_cudnn_moe_block_scale
    def test_non_uniform_distribution(self):
        """Experts get different numbers of tokens."""
        torch.manual_seed(0)
        num_experts = 4
        seg_lens = [256, 128, 384, 256]
        cum_m = sum(seg_lens)
        k, n = 512, 256

        a_bf16 = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
        m_indptr = torch.tensor(
            [0, *torch.tensor(seg_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )

        a_mxfp8, a_scale = self._quantize_a(a_bf16)
        b_mxfp8, b_scale = self._quantize_b(b_bf16)

        out = grouped_mm_mxfp8(
            a_mxfp8,
            b_mxfp8,
            a_scale,
            b_scale,
            m_indptr,
            out_dtype=torch.bfloat16,
        )
        ref = ref_grouped_mm(a_bf16, b_bf16, m_indptr, out_dtype=torch.bfloat16)

        cos_sim = F.cosine_similarity(
            ref.reshape(-1).float(),
            out.reshape(-1).float(),
            dim=0,
        )
        assert cos_sim > self.MIN_COS_SIM

    @requires_cudnn_moe_block_scale
    def test_empty_experts(self):
        """Some experts receive zero tokens."""
        torch.manual_seed(1)
        num_experts = 4
        seg_lens = [256, 0, 512, 0]
        cum_m = sum(seg_lens)
        k, n = 256, 256

        a_bf16 = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
        m_indptr = torch.tensor(
            [0, *torch.tensor(seg_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )

        a_mxfp8, a_scale = self._quantize_a(a_bf16)
        b_mxfp8, b_scale = self._quantize_b(b_bf16)

        out = grouped_mm_mxfp8(
            a_mxfp8,
            b_mxfp8,
            a_scale,
            b_scale,
            m_indptr,
            out_dtype=torch.bfloat16,
        )
        ref = ref_grouped_mm(a_bf16, b_bf16, m_indptr, out_dtype=torch.bfloat16)

        cos_sim = F.cosine_similarity(
            ref.reshape(-1).float(),
            out.reshape(-1).float(),
            dim=0,
        )
        assert cos_sim > self.MIN_COS_SIM

    @requires_cudnn_moe_block_scale
    @pytest.mark.parametrize("seg_lens", UNALIGNED_SEGMENTS)
    def test_unaligned_segments(self, seg_lens):
        """Each non-empty segment, aligned or not, has cosine similarity above MIN_SEGMENT_COS_SIM."""
        torch.manual_seed(3)
        num_experts = len(seg_lens)
        cum_m = sum(seg_lens)
        k, n = 512, 256

        a_bf16 = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
        m_indptr = torch.tensor(
            [0, *torch.tensor(seg_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )

        a_mxfp8, a_scale = self._quantize_a(a_bf16)
        b_mxfp8, b_scale = self._quantize_b(b_bf16)

        out = grouped_mm_mxfp8(
            a_mxfp8,
            b_mxfp8,
            a_scale,
            b_scale,
            m_indptr,
            out_dtype=torch.bfloat16,
        )
        ref = ref_grouped_mm(a_bf16, b_bf16, m_indptr, out_dtype=torch.bfloat16)

        assert torch.isfinite(out).all()
        for e in range(num_experts):
            start, end = m_indptr[e].item(), m_indptr[e + 1].item()
            if start == end:
                continue
            cos_sim = F.cosine_similarity(
                ref[start:end].reshape(-1).float(),
                out[start:end].reshape(-1).float(),
                dim=0,
            )
            assert cos_sim > self.MIN_SEGMENT_COS_SIM, (
                f"segment {e} rows [{start}, {end}): cosine similarity {cos_sim:.4f}"
            )

    @requires_cudnn_moe_block_scale
    @pytest.mark.parametrize("seg_lens", [[64, 64], [16, 48, 64]])
    def test_token_scale_rows_read_per_group(self, seg_lens):
        """Each output row uses its own token scale, also for groups after the first.

        The probe gives each token row its own scale (see _row_scale_probe), so
        out[i] / k must equal row i's scale exactly. A kernel that reads another
        row's scale gives a wrong value on each affected row. A cosine-similarity
        check barely catches it: on the affected segment it drops only to about
        0.96, against the 0.97 bar (cudnn-frontend#2081).
        """
        args = self._row_scale_probe(seg_lens)
        out = grouped_mm_mxfp8(*args[:5], out_dtype=torch.float32)
        wrong = self._rows_reading_another_scale(out, *args[5:])
        assert not wrong, (
            f"{len(wrong)} of {out.shape[0]} rows read another row's token scale; "
            f"first wrong rows {wrong[:8]} (segments {seg_lens})"
        )

    @staticmethod
    def _row_scale_probe(seg_lens):
        """Inputs whose exact output names the token-scale row each row used.

        A = 1 and B = 1 with unit weight scales, and token row i carries one e8m0
        scale 2^(exps[i] - 127) in all its k-blocks, so out[i] / k must equal that
        scale exactly. Returns (a, b, a_scale, b_scale, m_indptr, k, exps).
        """
        cum_m = sum(seg_lens)
        assert cum_m == 128, "one 128-row scale block keeps the layout simple"
        k, n = 512, 256
        kb = k // 32
        m_indptr = torch.tensor(
            [0, *torch.tensor(seg_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        a = torch.ones(cum_m, k, device="cuda").to(torch.float8_e4m3fn)
        b = torch.ones(len(seg_lens), n, k, device="cuda").to(torch.float8_e4m3fn)
        b_scale = torch.full(
            (len(seg_lens), n, kb), 127, dtype=torch.uint8, device="cuda"
        )
        # 128 distinct e8m0 exponents, 2^-67 .. 2^60; k * scale stays finite in fp32
        exps = [60 + i for i in range(cum_m)]
        # token scales in the F8_128x4 layout the quantizer produces: row r of the
        # block, k-block kk sits at byte (kk // 4) * 512 + (r % 32) * 16 + (r // 32) * 4 + kk % 4
        flat = torch.empty(cum_m * kb, dtype=torch.uint8)
        for r in range(cum_m):
            for kk in range(kb):
                flat[(kk // 4) * 512 + (r % 32) * 16 + (r // 32) * 4 + kk % 4] = exps[r]
        a_scale = flat.cuda().view(cum_m, kb)
        return a, b, a_scale, b_scale, m_indptr, k, exps

    @staticmethod
    def _rows_reading_another_scale(out, k, exps):
        """Rows of the probe's output that did not use their own token scale."""
        assert torch.isfinite(out).all()
        assert (out.max(dim=1).values == out.min(dim=1).values).all(), (
            "a row must use one scale for every column"
        )
        want = torch.tensor([2.0 ** (e - 127) for e in exps], device="cuda")
        return (out[:, 0] / k != want).nonzero().flatten().tolist()

    @requires_cudnn_moe_block_scale
    @pytest.mark.parametrize("seg_lens", [[64, 64], [16, 48, 64]])
    def test_token_scale_rows_read_per_group_every_tactic(self, seg_lens):
        """Every tactic the plan-count query reports runs and reads its own rows.

        Covers the indexed path the autotuner takes. Tactics index only the
        executable plans, so engines left out by _BAD_CUDNN_ENGINES are neither
        counted nor run.
        """
        from flashinfer.grouped_mm.cudnn import core

        a, b, a_scale, b_scale, m_indptr, k, exps = self._row_scale_probe(seg_lens)
        count = core._cudnn_moe_block_scale_grouped_gemm_plan_count(
            a, b, a_scale, b_scale, m_indptr, out_dtype=torch.float32
        )
        assert count > 0
        for tactic in range(count):
            out = core._run_cudnn_moe_block_scale_grouped_gemm_mxfp8(
                a, b, a_scale, b_scale, m_indptr, out_dtype=torch.float32, tactic=tactic
            )
            assert out is not None, f"tactic {tactic} of {count} did not run"
            wrong = self._rows_reading_another_scale(out, k, exps)
            assert not wrong, (
                f"tactic {tactic}: {len(wrong)} rows read another row's token scale "
                f"(segments {seg_lens})"
            )
        for tactic in (count, -2):
            out = core._run_cudnn_moe_block_scale_grouped_gemm_mxfp8(
                a, b, a_scale, b_scale, m_indptr, out_dtype=torch.float32, tactic=tactic
            )
            assert out is None

    @requires_cudnn_moe_block_scale
    def test_listed_engine_is_left_out(self, monkeypatch):
        """A listed engine is neither counted nor run; the other plans still run.

        Adds one of this device's plan names to the table. The exclusion path then
        runs on any GPU, not only on the setups the real entries cover.
        """
        from flashinfer.grouped_mm.cudnn import core
        from flashinfer.utils import get_compute_capability

        a, b, a_scale, b_scale, m_indptr, k, exps = self._row_scale_probe([16, 48, 64])
        monkeypatch.delenv(core._CUDNN_BAD_ENGINES_ENV, raising=False)
        names_seen = []
        real = core._executable_plan_indices

        def record(plan_names, excluded):
            names_seen.append(list(plan_names))
            return real(plan_names, excluded)

        def count():
            return core._cudnn_moe_block_scale_grouped_gemm_plan_count(
                a, b, a_scale, b_scale, m_indptr, out_dtype=torch.float32
            )

        def run(tactic):
            return core._run_cudnn_moe_block_scale_grouped_gemm_mxfp8(
                a, b, a_scale, b_scale, m_indptr, out_dtype=torch.float32, tactic=tactic
            )

        monkeypatch.setattr(core, "_executable_plan_indices", record)
        core._build_cudnn_moe_block_scale_grouped_gemm_graph.cache_clear()
        try:
            count()
            names = names_seen[-1]
            # The real entries stay listed. Add one more plan name that they do not
            # cover: the one matching the fewest plans (e.g. one knob setting).
            device = torch.device("cuda")
            already = core._block_scale_excluded_engines(device, token_is_fp4=False)
            candidates = [n for n in names if not any(e in n for e in already)]
            if not candidates:
                pytest.skip("the real entries already exclude every plan here")
            engine = min(candidates, key=lambda n: sum(n in name for name in names))
            expected = len(real(names, (*already, engine)))
            if expected == 0:
                pytest.skip(f"no plan is left once {engine!r} is listed too")
            entry = core._BadCudnnEngine(
                graph="moe_block_scale_mxfp8",
                engine=engine,
                sm_majors=(get_compute_capability(device)[0],),
                cudnn_first_bad=0,
                cudnn_first_fixed=None,
                issue="https://example.invalid/test",
            )
            monkeypatch.setattr(
                core, "_BAD_CUDNN_ENGINES", (*core._BAD_CUDNN_ENGINES, entry)
            )
            core._build_cudnn_moe_block_scale_grouped_gemm_graph.cache_clear()
            assert count() == expected
            for tactic in [-1, *range(expected)]:
                out = run(tactic)
                assert out is not None, f"tactic {tactic} did not run"
                assert not self._rows_reading_another_scale(out, k, exps)
            assert run(expected) is None
        finally:
            core._build_cudnn_moe_block_scale_grouped_gemm_graph.cache_clear()

    @requires_cudnn_moe_block_scale
    def test_every_plan_listed_raises(self, monkeypatch):
        """With every offered plan listed, the call raises and names the override."""
        from flashinfer.grouped_mm.cudnn import core
        from flashinfer.utils import get_compute_capability

        a, b, a_scale, b_scale, m_indptr, _, _ = self._row_scale_probe([64, 64])
        monkeypatch.setenv(core._CUDNN_BAD_ENGINES_ENV, "0")
        names_seen = []
        real = core._executable_plan_indices

        def record(plan_names, excluded):
            names_seen.append(list(plan_names))
            return real(plan_names, excluded)

        monkeypatch.setattr(core, "_executable_plan_indices", record)
        core._build_cudnn_moe_block_scale_grouped_gemm_graph.cache_clear()
        core._cudnn_moe_block_scale_grouped_gemm_plan_count(
            a, b, a_scale, b_scale, m_indptr, out_dtype=torch.float32
        )
        monkeypatch.delenv(core._CUDNN_BAD_ENGINES_ENV)
        # one entry per offered plan name, so every plan is listed
        sm_major = get_compute_capability(torch.device("cuda"))[0]
        entries = tuple(
            core._BadCudnnEngine(
                graph="moe_block_scale_mxfp8",
                engine=name,
                sm_majors=(sm_major,),
                cudnn_first_bad=0,
                cudnn_first_fixed=None,
                issue="https://example.invalid/test",
            )
            for name in names_seen[-1]
        )
        monkeypatch.setattr(core, "_BAD_CUDNN_ENGINES", entries)
        core._build_cudnn_moe_block_scale_grouped_gemm_graph.cache_clear()
        try:
            with pytest.raises(RuntimeError, match=core._CUDNN_BAD_ENGINES_ENV):
                grouped_mm_mxfp8(
                    a, b, a_scale, b_scale, m_indptr, out_dtype=torch.float32
                )
        finally:
            core._build_cudnn_moe_block_scale_grouped_gemm_graph.cache_clear()

    @requires_cudnn_moe_block_scale
    def test_bad_engine_table_is_current(self, monkeypatch):
        """Listed engines still misread here, and no unlisted engine does.

        Runs every plan of the MXFP8 graph with FLASHINFER_CUDNN_EXCLUDE_BAD_ENGINES=0.
        It fails when a listed engine reads every row correctly on a setup the
        table covers. That means a fix shipped: record cudnn_first_fixed in the
        entry and keep the entry. It also fails when an unlisted plan misreads,
        which means a new bad engine needs an entry. The table may list nothing
        for this setup, or cuDNN may not offer the listed engines. Then it checks
        only that every plan reads its own rows.
        """
        from flashinfer.grouped_mm.cudnn import core
        from flashinfer.utils import get_compute_capability

        # what the table lists here, whatever the override is set to in this process
        listed = core._bad_engines_for(
            "moe_block_scale_mxfp8",
            get_compute_capability(torch.device("cuda")),
            core.cudnn.backend_version(),
            None,
        )
        monkeypatch.setenv(core._CUDNN_BAD_ENGINES_ENV, "0")
        names_seen = []
        real = core._executable_plan_indices

        def record(plan_names, excluded):
            names_seen.append(list(plan_names))
            return real(plan_names, excluded)

        monkeypatch.setattr(core, "_executable_plan_indices", record)
        core._build_cudnn_moe_block_scale_grouped_gemm_graph.cache_clear()
        try:
            a, b, a_scale, b_scale, m_indptr, k, exps = self._row_scale_probe(
                [16, 48, 64]
            )
            count = core._cudnn_moe_block_scale_grouped_gemm_plan_count(
                a, b, a_scale, b_scale, m_indptr, out_dtype=torch.float32
            )
            names = names_seen[-1]
            assert len(names) == count
            misread = []
            for tactic in range(count):
                out = core._run_cudnn_moe_block_scale_grouped_gemm_mxfp8(
                    a,
                    b,
                    a_scale,
                    b_scale,
                    m_indptr,
                    out_dtype=torch.float32,
                    tactic=tactic,
                )
                if self._rows_reading_another_scale(out, k, exps):
                    misread.append(names[tactic])
        finally:
            core._build_cudnn_moe_block_scale_grouped_gemm_graph.cache_clear()

        unlisted = [nm for nm in misread if not any(e in nm for e in listed)]
        assert not unlisted, (
            f"plans {unlisted} misread token-scale rows on compute capability "
            f"{get_compute_capability(torch.device('cuda'))}, cuDNN "
            f"{core.cudnn.backend_version()}; add an entry to _BAD_CUDNN_ENGINES"
        )
        offered = [e for e in listed if any(e in nm for nm in names)]
        if not offered:
            return  # nothing listed or offered here; every plan read its own rows
        fixed = [e for e in offered if not any(e in nm for nm in misread)]
        assert not fixed, (
            f"{fixed} read every row correctly on cuDNN "
            f"{core.cudnn.backend_version()}: if NVIDIA confirms the fix, set "
            "cudnn_first_fixed in its _BAD_CUDNN_ENGINES entry (keep the entry)"
        )

    @requires_cudnn_moe_block_scale
    @pytest.mark.parametrize("starts", [[0], [0, 64]])
    def test_rows_past_last_offset_belong_to_last_expert(self, starts):
        """cuDNN runs the last expert to cum_m: ending m_indptr at 200 gives the output of ending it at cum_m."""
        torch.manual_seed(4)
        cum_m, k, n = 256, 256, 128
        experts = len(starts)
        a_bf16 = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(experts, n, k, dtype=torch.bfloat16, device="cuda")
        a_mxfp8, a_scale = self._quantize_a(a_bf16)
        b_mxfp8, b_scale = self._quantize_b(b_bf16)

        def run(last_end):
            m_indptr = torch.tensor(
                starts + [last_end], dtype=torch.int32, device="cuda"
            )
            return grouped_mm_mxfp8(a_mxfp8, b_mxfp8, a_scale, b_scale, m_indptr)

        assert torch.equal(run(200), run(cum_m))

    @requires_cudnn_moe_block_scale
    def test_scales_of_the_wrong_size_are_refused(self):
        """a_descale that is not the padded 128x4 swizzle of cum_m rows is refused."""
        cum_m, k, n = 240, 256, 128
        a_mxfp8 = torch.zeros(cum_m, k, device="cuda").to(torch.float8_e4m3fn)
        b_mxfp8 = torch.zeros(1, n, k, device="cuda").to(torch.float8_e4m3fn)
        unpadded = torch.full((cum_m, k // 32), 127, dtype=torch.uint8, device="cuda")
        b_scale = torch.full((1, n, k // 32), 127, dtype=torch.uint8, device="cuda")
        m_indptr = torch.tensor([0, cum_m], dtype=torch.int32, device="cuda")
        with pytest.raises(ValueError, match="128x4-swizzled scales"):
            grouped_mm_mxfp8(a_mxfp8, b_mxfp8, unpadded, b_scale, m_indptr)

    @requires_cudnn_moe_block_scale
    def test_single_expert(self):
        """Degenerate case: only one expert (equivalent to dense mm)."""
        torch.manual_seed(2)
        cum_m, k, n = 256, 512, 256

        a_bf16 = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(1, n, k, dtype=torch.bfloat16, device="cuda")
        m_indptr = torch.tensor([0, cum_m], dtype=torch.int32, device="cuda")

        a_mxfp8, a_scale = self._quantize_a(a_bf16)
        b_mxfp8, b_scale = self._quantize_b(b_bf16)

        out = grouped_mm_mxfp8(
            a_mxfp8,
            b_mxfp8,
            a_scale,
            b_scale,
            m_indptr,
            out_dtype=torch.bfloat16,
        )
        ref = (a_bf16.float() @ b_bf16[0].float().T).to(torch.bfloat16)

        cos_sim = F.cosine_similarity(
            ref.reshape(-1).float(),
            out.reshape(-1).float(),
            dim=0,
        )
        assert cos_sim > self.MIN_COS_SIM

    @requires_cudnn_moe_block_scale
    def test_preallocated_output(self):
        """Pass a pre-allocated output tensor."""
        torch.manual_seed(3)
        num_experts, tpe, k, n = 4, 128, 256, 256
        cum_m = num_experts * tpe

        a_bf16 = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
        m_indptr = (torch.arange(num_experts + 1, device="cuda") * tpe).to(torch.int32)

        a_mxfp8, a_scale = self._quantize_a(a_bf16)
        b_mxfp8, b_scale = self._quantize_b(b_bf16)

        out = torch.empty(cum_m, n, dtype=torch.bfloat16, device="cuda")
        result = grouped_mm_mxfp8(
            a_mxfp8,
            b_mxfp8,
            a_scale,
            b_scale,
            m_indptr,
            out=out,
        )
        ref = ref_grouped_mm(a_bf16, b_bf16, m_indptr, out_dtype=torch.bfloat16)

        assert result.data_ptr() == out.data_ptr()
        cos_sim = F.cosine_similarity(
            ref.reshape(-1).float(),
            result.reshape(-1).float(),
            dim=0,
        )
        assert cos_sim > self.MIN_COS_SIM

    @requires_cudnn_moe_block_scale
    @pytest.mark.parametrize(
        "out_dtype", [torch.float16, torch.bfloat16, torch.float32]
    )
    def test_out_dtype(self, out_dtype):
        """Verify different output dtypes work correctly."""
        torch.manual_seed(4)
        num_experts, tpe, k, n = 2, 128, 256, 256
        cum_m = num_experts * tpe

        a_bf16 = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
        m_indptr = (torch.arange(num_experts + 1, device="cuda") * tpe).to(torch.int32)

        a_mxfp8, a_scale = self._quantize_a(a_bf16)
        b_mxfp8, b_scale = self._quantize_b(b_bf16)

        out = grouped_mm_mxfp8(
            a_mxfp8,
            b_mxfp8,
            a_scale,
            b_scale,
            m_indptr,
            out_dtype=out_dtype,
        )
        assert out.dtype == out_dtype

        ref = ref_grouped_mm(a_bf16, b_bf16, m_indptr, out_dtype=out_dtype)
        cos_sim = F.cosine_similarity(
            ref.reshape(-1).float(),
            out.reshape(-1).float(),
            dim=0,
        )
        assert cos_sim > self.MIN_COS_SIM

    @requires_cudnn_moe_block_scale
    def test_graph_cache_reuse(self):
        """Calling twice with identical shapes should reuse the cached graph."""
        torch.manual_seed(5)
        num_experts, tpe, k, n = 4, 128, 256, 256
        cum_m = num_experts * tpe

        a_bf16 = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
        m_indptr = (torch.arange(num_experts + 1, device="cuda") * tpe).to(torch.int32)

        a_mxfp8, a_scale = self._quantize_a(a_bf16)
        b_mxfp8, b_scale = self._quantize_b(b_bf16)

        out1 = grouped_mm_mxfp8(a_mxfp8, b_mxfp8, a_scale, b_scale, m_indptr)
        out2 = grouped_mm_mxfp8(a_mxfp8, b_mxfp8, a_scale, b_scale, m_indptr)
        torch.testing.assert_close(out1, out2)

    @requires_cudnn_moe_block_scale
    def test_two_captures_of_one_call_replay(self):
        """Two graphs captured from one unaligned call replay correctly, also after one is freed."""
        a, b, a_scale, b_scale, m_indptr, k, exps = self._row_scale_probe([16, 48, 64])

        def call():
            return grouped_mm_mxfp8(
                a, b, a_scale, b_scale, m_indptr, out_dtype=torch.float32
            )

        call()  # builds the plan outside capture
        torch.cuda.synchronize()
        graphs, outs = [], []
        for _ in range(2):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outs.append(call())
            graphs.append(graph)
        for graph, out in [(graphs[1], outs[1]), (graphs[0], outs[0])]:
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert not self._rows_reading_another_scale(out, k, exps)
        first, second = graphs
        out = outs[1]
        del (
            graphs,
            outs,
            first,
            graph,
        )  # nothing references the first graph or its output
        torch.cuda.synchronize()
        out.fill_(float("nan"))
        second.replay()
        torch.cuda.synchronize()
        assert not self._rows_reading_another_scale(out, k, exps)

    @requires_cudnn_moe_block_scale
    def test_quantize_dtypes(self):
        """Verify quantized data and scale tensor dtypes."""
        torch.manual_seed(9)
        num_experts, tpe, k, n = 4, 128, 256, 256
        cum_m = num_experts * tpe

        a_bf16 = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")

        a_mxfp8, a_scale = self._quantize_a(a_bf16)
        b_mxfp8, b_scale = self._quantize_b(b_bf16)

        assert a_mxfp8.dtype == torch.float8_e4m3fn
        assert b_mxfp8.dtype == torch.float8_e4m3fn
        assert a_scale.dtype == torch.uint8
        assert b_scale.dtype == torch.uint8
        assert a_scale.ndim == 2
        assert b_scale.ndim == 3


@requires_cudnn_moe_block_scale
@requires_grouped_mm_mxfp8_cc
class TestGroupedMmMxfp8Validation:
    def test_sm12x_requires_block_scale_cudnn(self, monkeypatch):
        """SM120 / SM121 reject a cuDNN without their block-scaled engine before
        any graph is built."""
        import cudnn

        import flashinfer.grouped_mm.core as grouped_mm_core

        a = torch.zeros(128, 128, dtype=torch.float8_e4m3fn, device="cuda")
        b = torch.zeros(1, 128, 128, dtype=torch.float8_e4m3fn, device="cuda")
        a_descale = torch.zeros(128, 4, dtype=torch.uint8, device="cuda")
        b_descale = torch.zeros(1, 128, 4, dtype=torch.uint8, device="cuda")
        m_indptr = torch.tensor([0, 128], dtype=torch.int32, device="cuda")
        required = _cudnn_moe_block_scale_min_version(120)
        monkeypatch.setattr(
            grouped_mm_core, "get_compute_capability", lambda _: (12, 0)
        )
        monkeypatch.setattr(cudnn, "backend_version", lambda: required - 100)
        with pytest.raises(
            RuntimeError, match=f"requires backend version >= {required}"
        ):
            grouped_mm_mxfp8(a, b, a_descale, b_descale, m_indptr)

    def test_wrong_input_dtype(self):
        a = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(2, 64, 128, dtype=torch.bfloat16, device="cuda")
        a_descale = torch.zeros(64, 4, dtype=torch.uint8, device="cuda")
        b_descale = torch.zeros(2, 64, 4, dtype=torch.uint8, device="cuda")
        m_indptr = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_mxfp8(a, b, a_descale, b_descale, m_indptr)

    def test_wrong_descale_dtype(self):
        a = torch.zeros(64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        b = torch.zeros(2, 64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        a_descale = torch.zeros(64, 4, dtype=torch.float32, device="cuda")
        b_descale = torch.zeros(2, 64, 4, dtype=torch.uint8, device="cuda")
        m_indptr = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_mxfp8(a, b, a_descale, b_descale, m_indptr)

    def test_wrong_m_indptr_dtype(self):
        a = torch.zeros(64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        b = torch.zeros(2, 64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        a_descale = torch.zeros(64, 4, dtype=torch.uint8, device="cuda")
        b_descale = torch.zeros(2, 64, 4, dtype=torch.uint8, device="cuda")
        m_indptr = torch.tensor([0, 32, 64], dtype=torch.int64, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_mxfp8(a, b, a_descale, b_descale, m_indptr)

    def test_wrong_m_indptr_length(self):
        a = torch.zeros(64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        b = torch.zeros(2, 64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        a_descale = torch.zeros(64, 4, dtype=torch.uint8, device="cuda")
        b_descale = torch.zeros(2, 64, 4, dtype=torch.uint8, device="cuda")
        m_indptr = torch.tensor([0, 32, 48, 64], dtype=torch.int32, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_mxfp8(a, b, a_descale, b_descale, m_indptr)

    def test_k_mismatch(self):
        a = torch.zeros(64, 128, dtype=torch.float8_e4m3fn, device="cuda")
        b = torch.zeros(2, 64, 256, dtype=torch.float8_e4m3fn, device="cuda")
        a_descale = torch.zeros(64, 4, dtype=torch.uint8, device="cuda")
        b_descale = torch.zeros(2, 64, 8, dtype=torch.uint8, device="cuda")
        m_indptr = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            grouped_mm_mxfp8(a, b, a_descale, b_descale, m_indptr)
