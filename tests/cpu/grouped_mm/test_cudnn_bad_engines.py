"""Unit tests for the cuDNN bad-engine table of the grouped GEMM (no GPU needed)."""

import pytest

from flashinfer.grouped_mm.cudnn import core

GRAPH = "moe_block_scale_mxfp8"


def _bad(cc, version, env=None, graph=GRAPH):
    return core._bad_engines_for(graph, cc, version, env)


@pytest.fixture
def table(monkeypatch):
    """Return a function that replaces ``_BAD_CUDNN_ENGINES`` for this test."""

    def set_entries(*entries):
        monkeypatch.setattr(core, "_BAD_CUDNN_ENGINES", tuple(entries))

    return set_entries


def _entry(**kw):
    fields = dict(
        graph=GRAPH,
        engine="eng0",
        sm_majors=(10,),
        cudnn_first_bad=92600,
        cudnn_first_fixed=None,
        issue="https://example.invalid/1",
    )
    fields.update(kw)
    return core._BadCudnnEngine(**fields)


class TestBadEngineTable:
    def test_entries_are_well_formed(self):
        for entry in core._BAD_CUDNN_ENGINES:
            assert entry.graph in ("moe_block_scale_mxfp8", "moe_block_scale_fp4")
            assert entry.engine and entry.sm_majors
            assert entry.issue.startswith("https://")
            if entry.cudnn_first_fixed is not None:
                assert entry.cudnn_first_bad < entry.cudnn_first_fixed

    def test_cudnn_frontend_2081_entry(self):
        """The cudnn-frontend#2081 entry: compute capability 10.x, cuDNN >= _CUDNN_MOE_MIN_VERSION, MXFP8."""
        for cc in [(10, 0), (10, 3), (10, 7)]:
            first = core._CUDNN_MOE_MIN_VERSION
            for version in [first, 92400, 92600, 92700, 92801, 92900, 100000]:
                assert _bad(cc, version) == ("eng0",), (cc, version)
        assert _bad((10, 0), core._CUDNN_MOE_MIN_VERSION - 1) == ()
        for cc in [(9, 0), (11, 0), (12, 0), (12, 1)]:
            assert _bad(cc, 92700) == (), cc
        assert _bad((10, 0), 92700, graph="moe_block_scale_fp4") == ()


class TestBadEngineRange:
    def test_closed_open_cudnn_range(self, table):
        table(_entry(cudnn_first_bad=92600, cudnn_first_fixed=92900))
        assert _bad((10, 0), 92599) == ()
        assert _bad((10, 0), 92600) == ("eng0",)
        assert _bad((10, 0), 92899) == ("eng0",)
        assert _bad((10, 0), 92900) == ()  # the fixed release
        assert _bad((10, 0), 93000) == ()

    def test_open_upper_bound(self, table):
        table(_entry(cudnn_first_fixed=None))
        assert _bad((10, 0), 999999) == ("eng0",)

    def test_several_entries(self, table):
        table(
            _entry(cudnn_first_bad=92600, cudnn_first_fixed=92800),
            _entry(cudnn_first_bad=93000),  # same engine broken again later
            _entry(engine="eng7", sm_majors=(12,), cudnn_first_bad=92200),
            _entry(engine="eng3", graph="moe_block_scale_fp4"),
        )
        assert _bad((10, 0), 92700) == ("eng0",)
        assert _bad((10, 0), 92900) == ()
        assert _bad((10, 0), 93000) == ("eng0",)
        assert _bad((12, 0), 92300) == ("eng7",)
        assert _bad((10, 0), 92700, graph="moe_block_scale_fp4") == ("eng3",)

    def test_engine_listed_once(self, table):
        table(_entry(), _entry(cudnn_first_bad=92000))
        assert _bad((10, 0), 92700) == ("eng0",)


class TestOverride:
    @pytest.mark.parametrize("env", [None, "", "auto", "yes", "2"])
    def test_unset_or_unknown_applies_the_table(self, env):
        assert _bad((10, 0), 92700, env=env) == ("eng0",)
        assert _bad((12, 0), 92700, env=env) == ()

    def test_zero_excludes_nothing(self):
        assert _bad((10, 0), 92700, env="0") == ()

    def test_one_excludes_listed_engines_everywhere(self):
        assert _bad((12, 0), 92200, env="1") == ("eng0",)
        assert _bad((9, 0), 1, env="1") == ("eng0",)
        # but only engines listed for this graph
        assert _bad((10, 0), 92700, env="1", graph="moe_block_scale_fp4") == ()


class TestExecutablePlanIndices:
    NAMES = ["eng0_k24=0", "eng9_k24=132", "eng9_k24=0", "eng9", "eng0", "eng9_k24=12"]

    def test_nothing_excluded(self):
        assert core._executable_plan_indices(self.NAMES, ()) == tuple(range(6))

    def test_eng0_excluded(self):
        # the eng0 plans at indices 0 and 4 are dropped; eng9_k24=0 is kept
        assert core._executable_plan_indices(self.NAMES, ("eng0",)) == (1, 2, 3, 5)

    def test_knob_value_zero_is_not_engine_zero(self):
        assert "eng0" not in "eng9_k24=0"
        assert core._executable_plan_indices(["eng9_k24=0"], ("eng0",)) == (0,)

    def test_matches_like_the_frontend(self):
        """As in the frontend's deselect_engines, a name containing the engine is excluded."""
        names = ["eng0", "eng01_k1=2", "xeng0", "eng10"]
        kept = core._executable_plan_indices(names, ("eng0",))
        assert kept == (3,)
        assert kept == tuple(
            i for i, n in enumerate(names) if not any(e in n for e in ("eng0",))
        )

    def test_all_excluded(self):
        assert core._executable_plan_indices(["eng0", "eng0_k24=0"], ("eng0",)) == ()
