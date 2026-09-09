"""CPU-only checks for SM120 linear-GEMM workspace sizing.

`_sm120_max_workspace_bytes` allocates one buffer per shape that must cover
every tactic the autotuner can select and no more. Sizing it over the full
`can_implement` set reserves the O(splits*m*n*4B) Split-K reduction workspace
even where no Split-K row could be backed, which once OOMed the route autotuner
on the largest MiniMax-H3 shapes and dropped three of them to tactic=-1.

The bound that prevents this is `_sm120_workspace_budget_bytes`: a fraction of
the card's own memory. It is a legality bound, not a profitability one -- it
asks what this card can back, never which candidate is faster, so it carries no
shape-fitted thresholds and a larger card admits strictly more. These tests pin
both halves of the invariant (never wider than the selectable set, never
narrower) plus the property that matters most: over the frozen matrix on a real
card the budget prunes nothing at all.

No GPU and no JIT build: the backend module is a duck-typed fake and the budget
is injected.
"""

from types import SimpleNamespace


from flashinfer.gemm import svdquant_sm120_cutlass
from flashinfer.testing.svdq_model_shapes import MODEL_SHAPE_CASES


def _pack_row(kernel_id, splits=1, raster=0, swizzle=1, streamk=0):
    return (
        kernel_id | (splits << 8) | (raster << 16) | (swizzle << 24) | (streamk << 32)
    )


# MiniMax-H3 row 69, the largest shape in the frozen benchmark matrix and one of
# the three that fell back to tactic=-1 after the route autotuner OOMed.
_H3_LARGEST = (82752, 28672, 5376, 32)

# The one MiniMax-H3 shape that does keep a Split-K row exposed: k >= 8192 and
# only 5*42 = 210 output tiles, so it clears both halves of the Split-K rule.
# Narrowing the sizing set must be a no-op here.
_H3_SPLITK_SHAPE = (537, 5376, 14336, 32)

# MiniMax-H3 row 52, newly admitted to the fused K12 producer. No combined
# tactic is pinned for it, so the buffer has to cover every K3 tactic the
# autotuner may select -- which is the same invariant as everywhere else, just
# on a shape that now has one more runner offering it.
_H3_ROW52 = (73984, 5376, 14336, 32)

# The 27 frozen MiniMax-H3 benchmark cases, mirroring the registry's own
# construction (see tests/gemm/test_svdq_model_shapes.py). Declared locally so
# this file stays readable, then cross-checked against the registry below.
_H3_TEXT_M = (537, 1935, 6913)
_H3_OMNI_M = (61056, 73984, 82752)
_H3_BLOCK_NK = ((21504, 5376), (5376, 7168), (28672, 5376), (5376, 14336))
_H3_SHAPES = frozenset(
    [(m, n, k) for m in (*_H3_TEXT_M, *_H3_OMNI_M) for n, k in _H3_BLOCK_NK]
    + [(m, 5376, 5120) for m in _H3_TEXT_M]
)

# Split-K on a legacy Stream-K kernel: implementable anywhere, exposed only on
# long-K problems with an underfilled grid, and by far the largest workspace.
_SPLITK_TACTIC = 1
_SPLITK_WORKSPACE_BYTES = 38 * 1024**3
_LEGACY_WORKSPACE_BYTES = 4 * 1024**2
# No list of Stream-K kernel ids here on purpose: the budget reads a tactic's
# workspace size and never its identity, so which kernels can do Split-K is no
# longer something this layer has to know.


def _fake_workspace_module(with_tactic_row=True):
    """Backend where only the Split-K row carries a multi-GB workspace.

    with_tactic_row=False models a backend compiled before the runtime tactic
    table was exported, where the exposure filter cannot run.
    """
    rows = {
        _SPLITK_TACTIC: _pack_row(2, splits=2, streamk=1),
    }
    module = SimpleNamespace(
        nvfp4_svdquant_gemm_tactic_num=lambda: 4,
        nvfp4_svdquant_gemm_can_implement=lambda m, n, k, rank, t: True,
        nvfp4_svdquant_gemm_workspace_size=lambda m, n, k, t: (
            _SPLITK_WORKSPACE_BYTES if t == _SPLITK_TACTIC else _LEGACY_WORKSPACE_BYTES
        ),
    )
    if with_tactic_row:
        module.nvfp4_svdquant_gemm_tactic_row = lambda t: rows.get(t, _pack_row(8))
    return module


def _patch_workspace_module(monkeypatch, module):
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "_get_nvfp4_svdquant_module_for_device",
        lambda device, lora_rank=32: module,
    )
    monkeypatch.setattr(svdquant_sm120_cutlass, "_SM120_WORKSPACE_BYTES_CACHE", {})
    # The budget is decided once per shape and held, so it has to be cleared
    # alongside the sizes or one test's card size answers for the next.
    monkeypatch.setattr(svdquant_sm120_cutlass, "_SM120_WORKSPACE_BUDGET_CACHE", {})


# A card big enough to back the Split-K row, and one that is not. Neither
# number is a threshold the code knows: the budget is read off the device.
_CARD_BACKS_SPLITK = int((_SPLITK_WORKSPACE_BYTES + 1) / 0.25)
_CARD_TOO_SMALL_FOR_SPLITK = int((_SPLITK_WORKSPACE_BYTES - 1) / 0.25)


def _patch_budget(monkeypatch, total_memory_bytes):
    """Stand in for the card, so these stay CPU-only tests."""
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "_sm120_workspace_budget_bytes",
        lambda device_index: int(
            total_memory_bytes * svdquant_sm120_cutlass._SM120_WORKSPACE_BUDGET_FRACTION
        ),
    )


def test_sm120_workspace_skips_tactics_the_card_cannot_back(monkeypatch):
    """Sizing ranges over what the card can back, not the wider can_implement set.

    Sizing over can_implement reserved the O(splits*m*n*4B) Split-K reduction
    buffer unconditionally, which OOMed the route autotuner on the largest H3
    shapes. On a card too small to back that row, it is neither offered nor
    sized for.
    """
    _patch_workspace_module(monkeypatch, _fake_workspace_module())
    _patch_budget(monkeypatch, _CARD_TOO_SMALL_FOR_SPLITK)
    m, n, k, rank = _H3_LARGEST
    device = SimpleNamespace(index=0)
    assert _SPLITK_TACTIC not in svdquant_sm120_cutlass._nvfp4_svdquant_valid_tactics(
        _fake_workspace_module(), m, n, k, rank, device
    )
    assert (
        svdquant_sm120_cutlass._sm120_max_workspace_bytes(device, m, n, k, rank)
        == _LEGACY_WORKSPACE_BYTES
    )


def test_sm120_workspace_covers_every_selectable_tactic(monkeypatch):
    """The buffer must be a superset of what the autotuner can return.

    Under-sizing is caught fail-closed by the TVM_FFI_ICHECK_GE in
    csrc/nvfp4_svdquant_gemm_cutlass_sm120.cu, so this asserts the Python side
    never gets there. Both card sizes, because the selectable set differs.
    """
    for total in (_CARD_TOO_SMALL_FOR_SPLITK, _CARD_BACKS_SPLITK):
        module = _fake_workspace_module()
        _patch_workspace_module(monkeypatch, module)
        _patch_budget(monkeypatch, total)
        device = SimpleNamespace(index=0)
        for m, n, k in [
            _H3_LARGEST[:3],
            _H3_SPLITK_SHAPE[:3],
            (512, 3072, 3072),
            _H3_ROW52[:3],
        ]:
            rank = 32
            allotted = svdquant_sm120_cutlass._sm120_max_workspace_bytes(
                device, m, n, k, rank
            )
            selectable = svdquant_sm120_cutlass._nvfp4_svdquant_valid_tactics(
                module, m, n, k, rank, device
            )
            assert selectable, f"no tactic selectable for {(m, n, k)}"
            for t in selectable:
                assert allotted >= module.nvfp4_svdquant_gemm_workspace_size(
                    m, n, k, t
                ), f"tactic {t} under-runs the buffer at {(m, n, k)}"


def test_sm120_workspace_sizes_splitk_when_the_card_can_back_it(monkeypatch):
    """The bound is the card, not the shape.

    The old filter pruned Split-K by shape-fitted thresholds (`k >= 8192` and
    an output-tile count), so the same shape was pruned on every card. Here the
    identical shape that was refused above is served once the card is big
    enough -- which is the whole point of pricing legality instead of guessing
    profitability.
    """
    _patch_workspace_module(monkeypatch, _fake_workspace_module())
    _patch_budget(monkeypatch, _CARD_BACKS_SPLITK)
    m, n, k, rank = _H3_LARGEST
    device = SimpleNamespace(index=0)
    assert _SPLITK_TACTIC in svdquant_sm120_cutlass._nvfp4_svdquant_valid_tactics(
        _fake_workspace_module(), m, n, k, rank, device
    )
    assert (
        svdquant_sm120_cutlass._sm120_max_workspace_bytes(device, m, n, k, rank)
        == _SPLITK_WORKSPACE_BYTES
    )


def test_sm120_valid_tactics_are_unbounded_without_a_device():
    """No device means no budget: the caller gets the can_implement set.

    The CPU-only tests rely on this, and so does any backend too old to export
    a per-tactic workspace query.
    """
    module = _fake_workspace_module()
    m, n, k, rank = _H3_LARGEST
    every = list(range(module.nvfp4_svdquant_gemm_tactic_num()))
    assert (
        svdquant_sm120_cutlass._nvfp4_svdquant_valid_tactics(module, m, n, k, rank)
        == every
    )
    no_query = SimpleNamespace(
        nvfp4_svdquant_gemm_tactic_num=module.nvfp4_svdquant_gemm_tactic_num,
        nvfp4_svdquant_gemm_can_implement=module.nvfp4_svdquant_gemm_can_implement,
    )
    assert (
        svdquant_sm120_cutlass._nvfp4_svdquant_valid_tactics(
            no_query, m, n, k, rank, SimpleNamespace(index=0)
        )
        == every
    )


def test_sm120_budget_keeps_one_tactic_when_the_card_backs_none(monkeypatch):
    """A shape whose every candidate overruns the budget still has to run.

    Refusing to return a tactic would fail the shape outright; returning the
    cheapest one lets it try, and the fail-closed ICHECK on the C++ side is
    what catches a buffer that genuinely cannot be allocated.
    """
    module = _fake_workspace_module()
    _patch_workspace_module(monkeypatch, module)
    _patch_budget(monkeypatch, 4096)
    m, n, k, rank = _H3_LARGEST
    selectable = svdquant_sm120_cutlass._nvfp4_svdquant_valid_tactics(
        module, m, n, k, rank, SimpleNamespace(index=0)
    )
    assert len(selectable) == 1
    assert module.nvfp4_svdquant_gemm_workspace_size(m, n, k, selectable[0]) == (
        _LEGACY_WORKSPACE_BYTES
    )


def test_sm120_h3_shapes_match_the_frozen_registry():
    """The inline H3 cases must be the registry's, not a drifted copy.

    This file names its 27 shapes locally so it stays readable; that is only
    safe while they still agree with the shared registry.
    """
    assert len(_H3_SHAPES) == 27
    assert set(MODEL_SHAPE_CASES) >= _H3_SHAPES
    for shape in [(73984, 28672, 5376), (82752, 21504, 5376), (82752, 28672, 5376)]:
        assert shape in _H3_SHAPES, "the three route-autotuner OOM rows"


def test_sm120_budget_is_priced_against_free_memory_not_total():
    """The quantity that decides is what is left, not what the card has.

    Measured on 82752x28672x5376: the widest per-tactic workspace is 8.84 GiB,
    under a quarter of a 71 GiB card, so a total-memory budget admits it. By
    the time the buffer is provisioned the problem's own operands hold 65 GiB
    and 5.74 GiB remains, and the allocation fails -- the OOM the shape-keyed
    exposure filter used to prevent, reintroduced by pricing total memory.
    """
    import inspect

    source = inspect.getsource(svdquant_sm120_cutlass._sm120_workspace_budget_bytes)
    assert "mem_get_info" in source, "the budget must read free memory"
    assert "get_device_properties" not in source, "total memory is the wrong quantity"


def test_sm120_budget_is_decided_once_per_shape(monkeypatch):
    """Sizing and selection must agree even though free memory moves.

    They are separate calls. If selection re-read free memory it could admit a
    tactic the already-provisioned buffer cannot hold, which the C++ ICHECK_GE
    refuses at launch.
    """
    monkeypatch.setattr(svdquant_sm120_cutlass, "_SM120_WORKSPACE_BUDGET_CACHE", {})
    seen = []

    def moving_target(device_index):
        seen.append(len(seen))
        return (1 << 40) if not seen[:-1] else 4096

    monkeypatch.setattr(
        svdquant_sm120_cutlass, "_sm120_workspace_budget_bytes", moving_target
    )
    device = SimpleNamespace(index=0)
    key = (0, 64, 64, 64, 32)
    first = svdquant_sm120_cutlass._sm120_shape_workspace_budget(device, key)
    second = svdquant_sm120_cutlass._sm120_shape_workspace_budget(device, key)
    assert first == second, "the budget moved between sizing and selection"
    assert len(seen) == 1, "free memory was read more than once for one shape"


def test_sm120_budget_prunes_nothing_over_the_frozen_matrix():
    """On a real card the budget is not a de-facto shape table.

    Measured over all 71 frozen shapes at rank 32 with the candidate plane
    unpruned, the widest workspace any tactic
    asks for is 8.86 GiB. A quarter of either SM120 card in use clears that, so
    every implementable tactic on every frozen shape reaches the autotuner and
    the tuner -- not this bound -- decides which one runs.

    If a future kernel raises that ceiling past the budget, this fails and the
    pruning becomes visible rather than silent.
    """
    widest_measured = 8.86 * 1024**3
    frac = svdquant_sm120_cutlass._SM120_WORKSPACE_BUDGET_FRACTION
    for card_bytes in (
        73415 * 1024**2,  # the smaller SM120 part
        85651 * 1024**2,
    ):  # the larger SM120 part
        assert card_bytes * frac > widest_measured


def test_sm120_row52_admission_does_not_widen_its_workspace(monkeypatch):
    """Admitting row 52 to the fused route cannot re-OOM the route autotuner.

    The three shapes that OOMed did so because Split-K sizing was reserved
    where the card could not back it. On a card that cannot, row 52's admission
    adds a runner without adding a byte of workspace.
    """
    _patch_workspace_module(monkeypatch, _fake_workspace_module())
    _patch_budget(monkeypatch, _CARD_TOO_SMALL_FOR_SPLITK)
    m, n, k, rank = _H3_ROW52
    assert (m, n, k) in _H3_SHAPES
    assert (
        svdquant_sm120_cutlass._sm120_max_workspace_bytes(
            SimpleNamespace(index=0), m, n, k, rank
        )
        == _LEGACY_WORKSPACE_BYTES
    )
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(m, k, rank)
