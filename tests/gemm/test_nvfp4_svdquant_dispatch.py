"""Capability-dispatch unit tests for the NVFP4 SVDQuant Python layer.

These tests exercise the pure host-side routing logic (backend selection by compute
capability and the SM120 rank-32 gate) and run on any machine: no GPU, no JIT build.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from flashinfer.gemm import gemm_svdquant

from flashinfer.gemm import svdquant_sm120_cutlass


def test_backend_for_capability_routing():
    assert svdquant_sm120_cutlass._svdquant_backend_for_capability(10, 0) == "sm100"
    assert svdquant_sm120_cutlass._svdquant_backend_for_capability(10, 3) == "sm100"
    assert svdquant_sm120_cutlass._svdquant_backend_for_capability(12, 0) == "sm120"


@pytest.mark.parametrize(
    "major,minor", [(9, 0), (8, 9), (10, 1), (10, 2), (11, 0), (12, 1)]
)
def test_backend_for_capability_rejects_unsupported(major, minor):
    with pytest.raises(ValueError, match="not supported"):
        svdquant_sm120_cutlass._svdquant_backend_for_capability(major, minor)


class _FakeTensor:
    """Minimal stand-in for the tensor surface the public entries actually read.

    These tests grew SimpleNamespace fakes and then chased AttributeErrors one
    attribute at a time as the entry points came to read more of the tensor
    surface -- dtype, then numel, then size. A small class states the surface
    once, so a new read fails loudly here instead of somewhere downstream.
    """

    def __init__(self, shape, dtype=None, device="cuda:0", contiguous=True, **kwargs):
        # torch.empty takes either a size tuple or bare ints.
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype if dtype is not None else torch.float32
        self.device = device
        self.is_cuda = str(device).startswith("cuda")
        self.kwargs = kwargs
        self._contiguous = contiguous

    def numel(self):
        return math.prod(self.shape) if self.shape else 0

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def is_contiguous(self):
        return self._contiguous

    def __repr__(self):
        return f"_FakeTensor(shape={self.shape}, dtype={self.dtype})"


def _fake_tensor(shape, dtype, is_cuda=True, contiguous=True):
    """Duck-typed stand-in for what _check_mm_nvfp4_svdquant_problem reads."""
    return _FakeTensor(
        shape, dtype=dtype, device="cuda:0" if is_cuda else "cpu", contiguous=contiguous
    )


def _problem(m=256, n=256, k=256, rank=32):
    a = _fake_tensor((m, k // 2), torch.uint8)
    b = _fake_tensor((n, k // 2), torch.uint8)
    a_sf = _fake_tensor((m * k // 16,), torch.uint8)
    b_sf = _fake_tensor((n * k // 16,), torch.uint8)
    alpha = _fake_tensor((1,), torch.float32)
    d = _fake_tensor((m, rank), torch.bfloat16)
    l1 = _fake_tensor((n, rank), torch.bfloat16)
    return a, b, a_sf, b_sf, alpha, d, l1


@pytest.mark.parametrize("rank", [64, 96, 128])
def test_ranks_above_32_clear_the_shared_problem_check(monkeypatch, rank):
    """The shared check admits any positive multiple of 32; the rank gate is elsewhere.

    This used to assert that _check_mm_nvfp4_svdquant_problem refuses rank > 32
    on SM120, matching "rank 32 only". It never did -- that message comes from
    svdquant_linear in this backend, while the shared check only enforces the
    granularity -- and the premise has since stopped being true anyway: the
    collective takes its LoRA rank as a build parameter, so a module compiled
    for rank 64 serves it. What decides a rank now is which module you built
    and which entry point you call, not the architecture.
    """
    monkeypatch.setattr(
        svdquant_sm120_cutlass, "get_compute_capability", lambda device: (12, 0)
    )
    assert gemm_svdquant._check_mm_nvfp4_svdquant_problem(*_problem(rank=rank)) is True


@pytest.mark.parametrize("rank", [8, 48, 33])
def test_ranks_off_the_granularity_are_refused(monkeypatch, rank):
    """The granularity itself is still a hard boundary, on every architecture."""
    monkeypatch.setattr(
        svdquant_sm120_cutlass, "get_compute_capability", lambda device: (12, 0)
    )
    with pytest.raises(ValueError, match="multiple of 32"):
        gemm_svdquant._check_mm_nvfp4_svdquant_problem(*_problem(rank=rank))


def test_fused_linear_admission_is_rank_32_only(monkeypatch):
    """The fused K12 route is the part that is still rank-32 only.

    Its producer carries its own constexpr rank, so unlike the GEMM half it did
    not become a build parameter. This is the boundary the deleted test was
    reaching for.
    """
    for rank in (64, 96, 128):
        assert not svdquant_sm120_cutlass._sm120_fused_linear_supported(64, 3072, rank)
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(64, 3072, 32)


def test_sm120_accepts_rank_32(monkeypatch):
    monkeypatch.setattr(
        svdquant_sm120_cutlass, "get_compute_capability", lambda device: (12, 0)
    )
    assert gemm_svdquant._check_mm_nvfp4_svdquant_problem(*_problem(rank=32)) is True


def test_sm120_fused_route_declines_rank_64_but_the_entry_still_serves_it():
    """Rank 64 is served on SM120; only the *fused* prefix declines it.

    This replaces a test that asserted svdquant_linear raises "rank 32 only" for
    rank > 32 on SM120. That rejection was never implemented -- the string never
    existed outside a docstring -- and it is now the wrong contract: the LoRA
    rank is a build parameter of the GEMM half, so rank 64 is a supported rank,
    verified end to end on SM120 (4096x3072x3072 and 512x5120x5120 at 52.6 and
    52.7 dB SQNR through the public entry).

    What *is* true is narrower: the fused K12 prefix carries its own
    compile-time rank of 32, so it must not offer itself for anything else. The
    shape then takes the unfused route rather than being refused.
    """
    m, k = 64, 3072
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(m, k, 32)
    for rank in (64, 96, 128):
        assert not svdquant_sm120_cutlass._sm120_fused_linear_supported(m, k, rank), (
            f"the fused prefix stages rank 32 only, but offered itself for {rank}"
        )
    # Declining the fused route is not rejecting the problem: the public check
    # accepts any positive multiple of 32.
    assert gemm_svdquant._check_mm_nvfp4_svdquant_problem(*_problem(rank=64)) is True


@pytest.mark.parametrize("rank", [32, 64, 128])
def test_sm100_keeps_multiple_of_32_ranks(monkeypatch, rank):
    monkeypatch.setattr(
        svdquant_sm120_cutlass, "get_compute_capability", lambda device: (10, 0)
    )
    assert gemm_svdquant._check_mm_nvfp4_svdquant_problem(*_problem(rank=rank)) is True


def test_rank_not_multiple_of_32_rejected_everywhere(monkeypatch):
    monkeypatch.setattr(
        svdquant_sm120_cutlass, "get_compute_capability", lambda device: (10, 0)
    )
    with pytest.raises(ValueError, match="positive multiple"):
        gemm_svdquant._check_mm_nvfp4_svdquant_problem(*_problem(rank=48))


def test_module_getter_routes_by_capability(monkeypatch):
    calls = []
    # Patch where the name is looked up, not where it is defined:
    # svdquant_sm120_cutlass imported get_nvfp4_svdquant_module into its own
    # namespace, so patching gemm_svdquant leaves the bound name untouched and
    # the real getter runs -- which then tries to build an SM100 module on
    # whatever card the tests are on.
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "get_nvfp4_svdquant_module",
        lambda: calls.append("sm100") or "M100",
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "get_nvfp4_svdquant_sm120_module",
        lambda lora_rank=32: calls.append("sm120") or "M120",
    )

    monkeypatch.setattr(
        svdquant_sm120_cutlass, "get_compute_capability", lambda device: (10, 3)
    )
    assert svdquant_sm120_cutlass._get_nvfp4_svdquant_module_for_device("dev") == "M100"

    monkeypatch.setattr(
        svdquant_sm120_cutlass, "get_compute_capability", lambda device: (12, 0)
    )
    assert svdquant_sm120_cutlass._get_nvfp4_svdquant_module_for_device("dev") == "M120"

    monkeypatch.setattr(
        svdquant_sm120_cutlass, "get_compute_capability", lambda device: (9, 0)
    )
    with pytest.raises(ValueError, match="not supported"):
        svdquant_sm120_cutlass._get_nvfp4_svdquant_module_for_device("dev")

    assert calls == ["sm100", "sm120"]


class _FakeRunner:
    def get_valid_tactics(self, inputs, profile):
        return [0, 1]

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        return None


def _fake_runner():
    from flashinfer.autotuner import TunableRunner

    class FakeRunner(_FakeRunner, TunableRunner):
        pass

    return FakeRunner()


# Ten shapes matching the real input list (a, b, a_sf, b_sf, alpha, d, l1, bias,
# out, workspace): the tuning config's dynamic/constraint specs index into them.
_CACHE_TEST_SHAPES = (
    torch.Size([128, 32]),
    torch.Size([64, 32]),
    torch.Size([1024]),
    torch.Size([1024]),
    torch.Size([1]),
    torch.Size([128, 32]),
    torch.Size([64, 32]),
    torch.Size([64]),
    torch.Size([128, 64]),
    torch.Size([1024]),
)


def _seed_cache(tuner, op_name, runner, shapes, config, tactic):
    from flashinfer.autotuner import AutoTuner

    _, _, _, profile = tuner.search_cache(op_name, [runner], shapes, config)
    key = AutoTuner._get_cache_key(op_name, runner, shapes, config, ())
    # (tactic, profile). The runner id used to lead this tuple; AutoTuner reads
    # it as a pair now, so seeding the old shape raises on lookup rather than
    # missing, which is why this read as a cache-crossing failure.
    tuner.profiling_cache[key] = (tactic, profile)
    return key


def test_autotune_cache_namespaces_do_not_cross():
    """A tactic cached under the SM100 op name must not satisfy an SM120 lookup."""
    from flashinfer.autotuner import AutoTuner

    tuner = AutoTuner.get()
    runner = _fake_runner()
    config_sm100 = svdquant_sm120_cutlass._svdquant_tuning_config("sm100")
    config_sm120 = svdquant_sm120_cutlass._svdquant_tuning_config("sm120")
    key_sm100 = _seed_cache(
        tuner,
        svdquant_sm120_cutlass._svdquant_op_name("sm100"),
        runner,
        _CACHE_TEST_SHAPES,
        config_sm100,
        7,
    )
    try:
        hit_sm100, _, tactic_sm100, _ = tuner.search_cache(
            svdquant_sm120_cutlass._svdquant_op_name("sm100"),
            [runner],
            _CACHE_TEST_SHAPES,
            config_sm100,
        )
        assert hit_sm100 and tactic_sm100 == 7

        hit_sm120, _, tactic_sm120, _ = tuner.search_cache(
            svdquant_sm120_cutlass._svdquant_op_name("sm120"),
            [runner],
            _CACHE_TEST_SHAPES,
            config_sm120,
        )
        assert not hit_sm120 or tactic_sm120 != 7, (
            "an SM100-cached tactic leaked into the SM120 lookup"
        )
    finally:
        tuner.profiling_cache.pop(key_sm100, None)


def test_sm120_op_name_is_tactic_abi_versioned():
    """The SM120 namespace embeds the tactic-table ABI version."""
    name = svdquant_sm120_cutlass._svdquant_op_name("sm120")
    assert name == (
        f"nvfp4_svdquant_gemm_sm120_v{svdquant_sm120_cutlass._SM120_TACTIC_ABI_VERSION}"
    )
    assert name != "nvfp4_svdquant_gemm_sm120", (
        "v1 (pre-table) namespace must not be reused"
    )


def test_stale_v1_namespace_cache_never_replays():
    """Tactic ids cached under the pre-table v1 namespace must miss under v2."""
    tuner = _get_tuner()
    runner = _fake_runner()
    config = svdquant_sm120_cutlass._svdquant_tuning_config("sm120")
    key_v1 = _seed_cache(
        tuner, "nvfp4_svdquant_gemm_sm120", runner, _CACHE_TEST_SHAPES, config, 9
    )
    try:
        hit, _, tactic, _ = tuner.search_cache(
            svdquant_sm120_cutlass._svdquant_op_name("sm120"),
            [runner],
            _CACHE_TEST_SHAPES,
            config,
        )
        assert not hit or tactic != 9, "a v1-cached tactic id replayed under v2"
    finally:
        tuner.profiling_cache.pop(key_v1, None)


def test_same_process_stale_cache_invalidated_by_version_bump(monkeypatch):
    """An in-memory entry from the current version misses after an ABI bump."""
    tuner = _get_tuner()
    runner = _fake_runner()
    config = svdquant_sm120_cutlass._svdquant_tuning_config("sm120")
    key_cur = _seed_cache(
        tuner,
        svdquant_sm120_cutlass._svdquant_op_name("sm120"),
        runner,
        _CACHE_TEST_SHAPES,
        config,
        5,
    )
    try:
        monkeypatch.setattr(
            svdquant_sm120_cutlass,
            "_SM120_TACTIC_ABI_VERSION",
            svdquant_sm120_cutlass._SM120_TACTIC_ABI_VERSION + 1,
        )
        hit, _, tactic, _ = tuner.search_cache(
            svdquant_sm120_cutlass._svdquant_op_name("sm120"),
            [runner],
            _CACHE_TEST_SHAPES,
            config,
        )
        assert not hit or tactic != 5, (
            "a stale same-process tactic id survived the ABI version bump"
        )
    finally:
        tuner.profiling_cache.pop(key_cur, None)


def _get_tuner():
    from flashinfer.autotuner import AutoTuner

    return AutoTuner.get()


def test_sm120_exact_shape_keys_distinguish_m():
    """Two m values that share a hybrid bucket must get distinct exact keys."""
    from flashinfer.autotuner import AutoTuner

    runner = _fake_runner()
    config = svdquant_sm120_cutlass._svdquant_tuning_config("sm120")

    def shapes_for(m):
        return (
            torch.Size([m, 32]),
            torch.Size([64, 32]),
            torch.Size([1024]),
            torch.Size([1024]),
            torch.Size([1]),
            torch.Size([m, 32]),
            torch.Size([64, 32]),
            torch.Size([64]),
            torch.Size([m, 64]),
            torch.Size([1024]),
        )

    op = svdquant_sm120_cutlass._svdquant_op_name("sm120")
    key_a = AutoTuner._get_cache_key(op, runner, shapes_for(6889), config, ())
    key_b = AutoTuner._get_cache_key(op, runner, shapes_for(6890), config, ())
    assert key_a != key_b, "exact-shape tuning keys collided across different m"


def test_get_valid_tactics_filters_by_can_implement(monkeypatch):
    """The runner only offers tactics the backend says are implementable."""
    fake_module = SimpleNamespace(
        nvfp4_svdquant_gemm_tactic_num=lambda: 6,
        nvfp4_svdquant_gemm_can_implement=lambda m, n, k, rank, t: t % 2 == 0,
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "_get_nvfp4_svdquant_module_for_device",
        lambda device, lora_rank=32: fake_module,
    )
    # The two-argument runner is this backend's; gemm_svdquant keeps upstream's
    # one-argument version, and the module split left these pointing at it.
    runner = svdquant_sm120_cutlass._nvfp4_svdquant_gemm_runner(False, "dev")
    inputs = [
        SimpleNamespace(shape=(64, 1536)),  # a [m, k/2]
        SimpleNamespace(shape=(3072, 1536)),  # b [n, k/2]
        None,
        None,
        None,
        SimpleNamespace(shape=(64, 32)),  # d [m, r]
    ]
    assert runner.get_valid_tactics(inputs, profile=None) == [0, 2, 4]


def test_get_valid_tactics_unfiltered_without_query(monkeypatch):
    """Backends without the feasibility query keep the historical behavior."""
    fake_module = SimpleNamespace(nvfp4_svdquant_gemm_tactic_num=lambda: 4)
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "_get_nvfp4_svdquant_module_for_device",
        lambda device, lora_rank=32: fake_module,
    )
    # The two-argument runner is this backend's; gemm_svdquant keeps upstream's
    # one-argument version, and the module split left these pointing at it.
    runner = svdquant_sm120_cutlass._nvfp4_svdquant_gemm_runner(False, "dev")
    assert runner.get_valid_tactics([None] * 6, profile=None) == [0, 1, 2, 3]


def test_sm120_candidate_set_carries_no_shape_fitted_pruning(monkeypatch):
    """Beyond can_implement, the candidate set must not vary with the shape.

    What stood here pinned a filter that pruned tactics by fitted thresholds --
    `m <= 512` for the swap and 64-row driver rows, `k >= 8192` plus an output
    tile count for Split-K, `m <= 8192` for raster/swizzle, a one-wave envelope
    for the static-scheduler rows. Every one of those numbers was measured on a
    single SM120 card and then applied to every SM120 card, and the filter's own
    docstring priced its benefit as mixed in sign and inside the measurement
    floor. It bought scan time, and scan time is not a cost we
    are paying for.

    The bound that replaced it reads the card, not the shape. So on a card that
    can back every request, the candidate set is exactly the can_implement set
    -- on the very shapes the old thresholds singled out.
    """
    fake_module = SimpleNamespace(
        nvfp4_svdquant_gemm_tactic_num=lambda: 8,
        nvfp4_svdquant_gemm_can_implement=lambda m, n, k, rank, t: True,
        nvfp4_svdquant_gemm_workspace_size=lambda m, n, k, t: 1 << 20,
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "_sm120_workspace_budget_bytes",
        lambda device_index: 1 << 40,
    )
    every = list(range(8))
    # The shapes the deleted thresholds treated differently from one another.
    for m, n, k in [
        (64, 3072, 3072),
        (512, 3072, 3072),
        (1024, 3072, 3072),
        (7800, 3072, 3072),
        (64, 3072, 12288),
        (4096, 3072, 12288),
        (27280, 3072, 14336),
        (82752, 28672, 5376),
    ]:
        assert (
            svdquant_sm120_cutlass._nvfp4_svdquant_valid_tactics(
                fake_module, m, n, k, 32, SimpleNamespace(index=0)
            )
            == every
        ), (m, n, k)


# No _pack_row helper here any more: the candidate bound reads a tactic's
# workspace size, never its packed kernel id, raster or swizzle fields.


def test_sm120_tuned_linear_route_bypasses_repeated_autotuner_lookup():
    """A tuned exact-shape winner should dispatch directly after the first lookup."""

    class FakeTuner:
        is_tuning_mode = True

        def __init__(self):
            self.calls = 0

        def choose_one(self, op_name, runners, tuning_config, inputs):
            self.calls += 1
            return runners[1], 43

    tuner = FakeTuner()
    runners = [object(), object()]
    svdquant_sm120_cutlass._SM120_LINEAR_DISPATCH_CACHE.clear()
    try:
        first = svdquant_sm120_cutlass._choose_sm120_linear_runner(
            tuner,
            runners,
            object(),
            [],
            device="cuda:0",
            m=256,
            n=3072,
            k=12288,
            rank=32,
            enable_pdl=False,
            has_bias=True,
        )
        tuner.is_tuning_mode = False
        second = svdquant_sm120_cutlass._choose_sm120_linear_runner(
            tuner,
            runners,
            object(),
            [],
            device="cuda:0",
            m=256,
            n=3072,
            k=12288,
            rank=32,
            enable_pdl=False,
            has_bias=True,
        )
        third = svdquant_sm120_cutlass._choose_sm120_linear_runner(
            tuner,
            runners,
            object(),
            [],
            device="cuda:0",
            m=256,
            n=3072,
            k=12288,
            rank=32,
            enable_pdl=False,
            has_bias=False,
        )

        assert first == second == third == (runners[1], 43)
        assert tuner.calls == 2, "bias/no-bias profiles must not share a dispatch entry"
    finally:
        svdquant_sm120_cutlass._SM120_LINEAR_DISPATCH_CACHE.clear()


def _sm120_linear_op_name_in_use(m, k, rank, n=5376, enable_pdl=False):
    """Drive the production chooser and return the op name it actually used.

    Reads the name off both seams at once: what `choose_one` is keyed by (the
    persistent autotune cache) and what the in-process dispatch cache is keyed
    by. They must be one computed value, or a shape could profile under one name
    and replay under another.
    """

    class RecordingTuner:
        is_tuning_mode = True

        def __init__(self):
            self.op_names = []

        def choose_one(self, op_name, runners, tuning_config, inputs):
            self.op_names.append(op_name)
            return runners[0], 9

    tuner = RecordingTuner()
    runners = [object()]
    svdquant_sm120_cutlass._SM120_LINEAR_DISPATCH_CACHE.clear()
    try:
        svdquant_sm120_cutlass._choose_sm120_linear_runner(
            tuner,
            runners,
            object(),
            [],
            device="cuda:0",
            m=m,
            n=n,
            k=k,
            rank=rank,
            enable_pdl=enable_pdl,
            has_bias=False,
        )
        (op_name,) = tuner.op_names
        (cache_key,) = svdquant_sm120_cutlass._SM120_LINEAR_DISPATCH_CACHE
        assert cache_key[0] == op_name, (
            "the dispatch cache and the tuner key must share one computed name"
        )
        return op_name
    finally:
        svdquant_sm120_cutlass._SM120_LINEAR_DISPATCH_CACHE.clear()


# Representative row-52 route used by the PDL key tests below.
_SM120_ROUTE_V6_MKR = (73984, 14336, 32)


@pytest.mark.parametrize(
    "m,k,rank,route_version",
    [
        (*_SM120_ROUTE_V6_MKR, 6),
        (73984, 7168, 32, 6),
        (82752, 5376, 32, 8),
    ],
)
def test_sm120_linear_op_name_uses_current_route_abi(m, k, rank, route_version):
    """Every admitted runner-set change invalidates its older route record."""
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(m, k, rank)
    assert _sm120_linear_op_name_in_use(m, k, rank).endswith(
        f"routes_v{route_version}_tactics_v3"
    )


@pytest.mark.parametrize(
    "m,k,rank",
    [
        (256, 12288, 32),  # ordinary non-fused shape
        (512, 3072, 32),  # fused route that predates the admission
        (27280, 14336, 32),  # fused route on the same K as row 52
        (73984, 14336, 16),  # right (m, k), rank the fused route never admitted
    ],
)
def test_sm120_linear_op_name_keeps_v5_for_every_other_shape(m, k, rank):
    """Shapes whose runner set did not change must reach their retained winners.

    A global bump made every unrelated shape miss its v5 record and re-profile;
    one such re-tune picked a slower tactic and regressed the shape.
    """
    assert _sm120_linear_op_name_in_use(m, k, rank).endswith("routes_v5_tactics_v3")


@pytest.mark.parametrize(
    "m,k,rank,expected_without_pdl",
    [
        (256, 12288, 32, "svdquant_linear_sm120_routes_v5_tactics_v3"),
        (*_SM120_ROUTE_V6_MKR, "svdquant_linear_sm120_routes_v6_tactics_v3"),
    ],
)
def test_sm120_linear_op_name_separates_pdl_in_the_persistent_key(
    m, k, rank, expected_without_pdl
):
    """PDL must key its own autotune record, not replay the non-PDL winner.

    PDL changes what every runner costs, so a profile taken without it is not a
    valid answer for a call that has it. The in-process dispatch key always
    separated the two; the persistent name did not, so a PDL=False winner could
    be replayed for PDL=True. The names below are asserted exactly, not by
    suffix: PDL=False must stay byte-identical to the historical name (the
    frozen/default benchmark path's retained winners all live under it), and
    PDL=True must be that same name plus one stable `_pdl` suffix -- extending
    the name, never re-versioning the routes, so no PDL=False record anywhere is
    invalidated.

    The helper drives the real chooser and asserts, on each call, that the name
    `choose_one` was keyed by is the same computed value the in-process dispatch
    cache was keyed by -- so this covers both seams in both PDL modes.
    """
    without_pdl = _sm120_linear_op_name_in_use(m, k, rank, enable_pdl=False)
    with_pdl = _sm120_linear_op_name_in_use(m, k, rank, enable_pdl=True)

    assert without_pdl == expected_without_pdl
    assert with_pdl == f"{expected_without_pdl}_pdl"
    assert with_pdl != without_pdl, (
        "otherwise-identical PDL modes must not share one persistent record"
    )


# Removed: test_sm120_multibackend_tuning_avoids_cutedsl_cloned_buffer_tail_fault
# exercised _svdquant_tuning_config_for_runners, which guarded the CuTeDSL
# runner sharing a tuning record with this one. That runner is deliberately
# not part of this backend, so the hazard it covered cannot arise here.


def test_lora_ladder_gen_rungs_are_guarded_variants():
    """Ladder rungs are separate whole-module variants and refuse to combine
    with the other variant flags (they measure the byte-exact path only)."""
    from flashinfer.jit.gemm.svdquant_sm120 import (
        gen_gemm_sm120_module_cutlass_nvfp4_svdquant as gen,
    )

    with pytest.raises(ValueError):
        gen(lora_ladder="bogus")
    with pytest.raises(ValueError):
        gen(lora_dedicated=True, lora_ladder="residual")
    with pytest.raises(ValueError):
        gen(lora_every_split=True, lora_ladder="transfer")
    # the smem-neutral barrier is a third data-path policy: it excludes the
    # other variants but composes with the epilogue-overlap fold
    with pytest.raises(ValueError):
        gen(lora_neutral_barrier=True, lora_dedicated=True)
    with pytest.raises(ValueError):
        gen(lora_neutral_barrier=True, lora_every_split=True)
    with pytest.raises(ValueError):
        gen(lora_neutral_barrier=True, lora_ladder="residual")

    for rung in ("residual", "transfer"):
        spec = gen(lora_ladder=rung)
        assert spec.name.endswith(f"_lora_ladder_{rung}")

    # the epilogue-overlap fold is the production default; the serial-tail
    # fallback is its own build variant
    assert not gen().name.endswith("_lora_serial_tail")
    assert gen(lora_epi_overlap=False).name.endswith("_lora_serial_tail")
    assert gen(lora_neutral_barrier=True).name.endswith("_lora_neutral_barrier")
    assert gen(lora_neutral_barrier=True, lora_epi_overlap=False).name.endswith(
        "_lora_neutral_barrier_serial_tail"
    )


def test_lora_smem_swizzle_gen_is_guarded_variant():
    """The bank-mapping diagnostic is an isolated byte-exact-path build."""
    from flashinfer.jit.gemm.svdquant_sm120 import (
        gen_gemm_sm120_module_cutlass_nvfp4_svdquant as gen,
    )

    default_spec = gen()
    disabled_spec = gen(lora_smem_swizzle=False)
    swizzled_spec = gen(lora_smem_swizzle=True)
    assert disabled_spec.name == default_spec.name
    assert "-DSVDQ_SM120_LORA_SMEM_SWIZZLE" not in default_spec.extra_cuda_cflags
    assert swizzled_spec.name.endswith("_lora_smem_swizzle")
    assert "-DSVDQ_SM120_LORA_SMEM_SWIZZLE" in swizzled_spec.extra_cuda_cflags
    assert gen(
        lora_smem_swizzle=True,
        lora_epi_overlap=False,
    ).name.endswith("_lora_serial_tail_lora_smem_swizzle")
    for incompatible in (
        {"lora_dedicated": True},
        {"lora_every_split": True},
        {"lora_neutral_barrier": True},
        {"lora_ladder": "residual"},
    ):
        with pytest.raises(ValueError):
            gen(lora_smem_swizzle=True, **incompatible)


def test_row80_stage_override_gen_is_guarded_variant():
    """Only row80 may opt into the measurement-only 4/5-stage builds."""
    from flashinfer.jit.gemm.svdquant_sm120 import (
        gen_gemm_sm120_module_cutlass_nvfp4_svdquant as gen,
    )

    default_spec = gen()
    assert "SVDQ_SM120_ROW80_STAGES" not in " ".join(default_spec.extra_cuda_cflags)
    for stages in (4, 5):
        spec = gen(row80_stages=stages)
        assert spec.name.endswith(f"_row80_stages{stages}")
        assert f"-DSVDQ_SM120_ROW80_STAGES={stages}" in spec.extra_cuda_cflags
    combined = gen(lora_smem_swizzle=True, row80_stages=5)
    assert combined.name.endswith("_lora_smem_swizzle_row80_stages5")
    assert "-DSVDQ_SM120_LORA_SMEM_SWIZZLE" in combined.extra_cuda_cflags
    assert "-DSVDQ_SM120_ROW80_STAGES=5" in combined.extra_cuda_cflags
    for invalid in (-1, 1, 2, 3, 6):
        with pytest.raises(ValueError):
            gen(row80_stages=invalid)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1935, 5376, 7168),  # was pinned to tactic 82
        (537, 5376, 14336),  # was pinned to tactic 81, route ABI v9
    ],
)
def test_sm120_linear_always_reaches_the_tuner(monkeypatch, m, n, k):
    """No shape short-circuits selection any more.

    Both of these used to be dispatched straight to nvfp4_svdquant_linear_sm120
    with a tactic read out of a source table, so the tuner never saw them and
    nothing ever re-checked the value. One such value measured off the best
    candidate in its own field. The
    contract now is that every SM120 linear shape is selected, and the only
    thing that may produce a tactic is the tuner.
    """
    allocated = []

    def fake_empty(shape, **kwargs):
        # The public entry validates dtype and sizes before choosing a route,
        # so a stand-in has to answer those too.
        # torch.empty accepts either a size tuple or bare ints, and the code
        # under test uses both spellings; tuple() on a bare int raises.
        tensor = _FakeTensor(shape, **kwargs)
        allocated.append(tensor)
        return tensor

    module = SimpleNamespace(
        nvfp4_svdquant_gemm_workspace_size=lambda *args: 0,
        nvfp4_svdquant_linear_sm120=lambda *args: pytest.fail(
            "a shape reached the fused FFI without going through the tuner"
        ),
    )

    launched = []

    class Recorder:
        def __call__(self, inputs, tactic):
            launched.append((inputs, tactic))

    recorder = Recorder()

    class Tuner:
        is_tuning_mode = False
        seen = []

        def choose_one(self, op_name, runners, tuning_config, inputs):
            Tuner.seen.append(op_name)
            return runners[0], 7

    monkeypatch.setattr(
        svdquant_sm120_cutlass, "get_compute_capability", lambda device: (12, 0)
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "get_nvfp4_svdquant_sm120_module",
        lambda lora_rank=32: module,
    )
    monkeypatch.setattr(gemm_svdquant.torch, "empty", fake_empty)
    monkeypatch.setattr(
        gemm_svdquant, "_get_cache_buf", lambda *args, **kwargs: "workspace"
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass, "_sm120_max_workspace_bytes", lambda *args, **kwargs: 0
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "_cached_sm120_linear_runners",
        lambda *args, **kwargs: [recorder],
    )
    monkeypatch.setattr(
        gemm_svdquant.AutoTuner, "get", classmethod(lambda cls: Tuner())
    )
    svdquant_sm120_cutlass._SM120_LINEAR_DISPATCH_CACHE.clear()

    x = SimpleNamespace(ndim=2, shape=(m, k), device="cuda:0")
    weight = SimpleNamespace(ndim=2, shape=(n, k // 2))
    l2t = SimpleNamespace(ndim=2, shape=(k, 32))
    out = gemm_svdquant.svdquant_linear(
        x,
        weight,
        "weight_sf",
        "alpha",
        "pre_quant_scale",
        l2t,
        "l1_scaled",
        "global_scale",
        bias="bias",
        enable_pdl=False,
        backend="cutlass-sm120",
    )

    # Identify the output by its shape, not by being the last allocation: the
    # workspace buffer is allocated too, and which of them comes last is not
    # part of the contract this test is about.
    outputs = [t for t in allocated if t.shape == (m, n)]
    assert outputs and out is outputs[-1]
    assert launched and launched[0][1] == 7, "the tactic must be the tuner's answer"
    assert Tuner.seen, "the tuner was never consulted"
    svdquant_sm120_cutlass._SM120_LINEAR_DISPATCH_CACHE.clear()


def test_sm120_linear_route_pool(monkeypatch):
    """Which routes a shape offers the tuner, now that there are only two.

    A shape the fused prefix admits offers it alongside CUTLASS and the tuner
    picks; every other shape offers CUTLASS alone. Nothing else may appear --
    a third route would mean the CuTeDSL fallback came back.
    """
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "_sm120_fused_linear_runner",
        lambda enable_pdl, device: "fused",
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "_sm120_cutlass_linear_runner",
        lambda enable_pdl, device: "cutlass",
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "_sm120_fused_linear_supported",
        lambda m, k, rank: m == 4096,
    )
    pool = svdquant_sm120_cutlass._sm120_linear_runners(
        False, "dev", 4096, 3072, 3072, 32
    )
    assert pool == ["fused", "cutlass"]
    pool = svdquant_sm120_cutlass._sm120_linear_runners(
        False, "dev", 4097, 3072, 3072, 32
    )
    assert pool == ["cutlass"]
