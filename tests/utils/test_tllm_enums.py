from flashinfer.tllm_enums import ActivationType, is_gated_activation


def test_geglu_tanh_is_gated_activation():
    assert is_gated_activation(ActivationType.GegluTanh)
    assert is_gated_activation(ActivationType.GegluTanh.value)
    assert ActivationType.GegluTanh.is_gated


# ---------------------------------------------------------------------------
# RoutingMethodType: Python enum <-> C++ enum <-> serializer contract.
#
# The kernel-side enum lives in include/flashinfer/trtllm/fused_moe/runner.h and
# is only kept in sync by convention. New members are appended *after*
# ``Unspecified`` (e.g. ``SqrtSoftplus = 11``), so nothing may treat
# ``Unspecified`` as the last / maximum value.
# ---------------------------------------------------------------------------
import re
from pathlib import Path

from flashinfer.tllm_enums import RoutingMethodType

_RUNNER_H = (
    Path(__file__).resolve().parents[2]
    / "include"
    / "flashinfer"
    / "trtllm"
    / "fused_moe"
    / "runner.h"
)


def _cpp_routing_method_enum():
    src = _RUNNER_H.read_text()
    block = src[src.index("enum class RoutingMethodType") :]
    block = block[: block.index("};")]
    return {
        name: int(value)
        for name, value in re.findall(r"^\s*([A-Za-z0-9_]+)\s*=\s*(\d+),", block, re.M)
    }


def _cpp_serializer_cases():
    src = _RUNNER_H.read_text()
    body = src[src.index("serializeMoeRoutingMethodType(") :]
    body = body[: body.index("default:")]
    return set(re.findall(r"case RoutingMethodType::([A-Za-z0-9_]+):", body))


def test_routing_method_type_matches_cpp_enum():
    assert {m.name: int(m) for m in RoutingMethodType} == _cpp_routing_method_enum()


def test_routing_method_type_serializer_covers_every_concrete_member():
    cases = _cpp_serializer_cases()
    missing = {
        m.name for m in RoutingMethodType if m is not RoutingMethodType.Unspecified
    } - cases
    assert not missing, f"serializeMoeRoutingMethodType lacks {sorted(missing)}"
    assert "Unspecified" not in cases  # falls through to the invalid-method default


def test_routing_method_type_int_round_trip_does_not_alias_unspecified():
    for member in RoutingMethodType:
        assert RoutingMethodType(int(member)) is member
        assert eval(repr(member)) is member
    # Appended-after-Unspecified member: numerically above the sentinel, still concrete.
    assert int(RoutingMethodType.SqrtSoftplus) == 11
    assert RoutingMethodType.SqrtSoftplus > RoutingMethodType.Unspecified
    assert RoutingMethodType(11) is RoutingMethodType.SqrtSoftplus
    assert RoutingMethodType(11) is not RoutingMethodType.Unspecified
    assert max(RoutingMethodType) is RoutingMethodType.SqrtSoftplus
