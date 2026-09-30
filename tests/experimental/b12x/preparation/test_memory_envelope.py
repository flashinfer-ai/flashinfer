"""Block-scaled configurations fit the caller's workspace cap."""

import pytest

from b12x.preparation import DetectedDevice, DeviceIdentity

IDENTITY = DeviceIdentity("nvidia", (12, 0), 148, "synthetic SM120")
DEVICE = DetectedDevice(ordinal=0, identity=IDENTITY)
WORKSPACE_CAP = 2_000_000_000
BUILTIN = [{
    "stage": "weights", "unit": "MXFP8", "request": "builtin.vision.fc",
    "autotune": False, "component": "gemm.blockscaled_precision",
    "query": {
        "recipe": "mxfp8", "num_tokens": 65_536, "in_features": 4_352,
        "padded_in_features": 4_352, "out_features": 3_456, "activation_mode": "a16",
        "source_contiguous": True, "source_aligned": True,
        "workspace_form": "provided", "workspace_nbytes": WORKSPACE_CAP,
    },
    "invocation": {},
}]


def _blockscaled_plan(record):
    from b12x.gemm.blockscaled import _tuning
    from b12x.gemm.blockscaled._preparation import plan

    fields = {name: value for name, value in record["query"].items() if name != "codegen"}
    query = _tuning.BlockscaledQuery(**fields)
    return plan(query), query


@pytest.mark.parametrize("record", BUILTIN,
                         ids=lambda record: record["request"])
def test_blockscaled_candidates_fit_the_workspace_cap(record):
    from b12x.gemm.blockscaled._preparation import _owned_bytes, _workspace_bytes
    from b12x.gemm.blockscaled._tuning import TUNING, effective_a16_config
    from b12x.preparation.types import _plan_scope

    declaration, query = _blockscaled_plan(record)
    configuration = TUNING.configure(query, device=IDENTITY, override=None)
    configs = [configuration.default, *(config for _, config in TUNING.iterate(configuration))]
    cap = query.workspace_nbytes
    for config in configs:
        needed = _workspace_bytes(query, config)
        assert cap is None or needed <= cap, (config, needed)
        with _plan_scope(declaration):
            requirements = declaration._memory_requirements(config, DEVICE)
        if query.workspace_form == "provided":
            assert requirements.scratch_nbytes <= (cap or requirements.scratch_nbytes) + 255
        else:
            assert _owned_bytes(query, config) <= (cap or 0) + 2 * query.num_tokens * query.padded_in_features + 8 << 20
    if record["request"] == "builtin.vision.fc":
        splits = {effective_a16_config(query, config)[2] for config in configs if config.mode == "a16"}
        assert splits == {1, 2}
        assert max(_workspace_bytes(query, config) for config in configs) == 1_811_939_328
