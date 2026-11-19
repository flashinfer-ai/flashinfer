"""CPU-only tests for the Cake MoE finalize backend selector."""

import re

import pytest
import torch

from flashinfer.comm import AllReduceFusionPattern, TRTLLMAllReduceFusionWorkspace
from flashinfer.comm import allreduce as allreduce_module
from flashinfer.comm import trtllm_ar
from flashinfer.comm.workspace_base import AllReduceFusionWorkspace
from flashinfer.jit import cake_moe_finalize_comm


def _arguments() -> dict[str, object]:
    state = torch.empty((1, 7168), dtype=torch.float16)
    return {
        "allreduce_in": torch.empty((8, 7168), dtype=torch.float16),
        "residual_in": state,
        "norm_weight": torch.empty((7168,), dtype=torch.float16),
        "expanded_idx_to_permuted_idx": torch.empty((1, 8), dtype=torch.int32),
        "norm_out": torch.empty_like(state),
        "residual_out": torch.empty_like(state),
        "quant_out": None,
        "scale_out": None,
        "workspace_ptrs": torch.empty((7,), dtype=torch.int64),
        "launch_with_pdl": False,
        "world_rank": 0,
        "world_size": 2,
        "eps": 1e-5,
        "shared_expert_output": None,
        "expert_scale_factor": torch.empty((1, 8), dtype=torch.float16),
        "routed_scaling_factor": None,
    }


def _fake_workspace() -> TRTLLMAllReduceFusionWorkspace:
    workspace = object.__new__(TRTLLMAllReduceFusionWorkspace)
    AllReduceFusionWorkspace.__init__(workspace, world_size=2, rank=0)
    workspace.mem_handles = []
    workspace.workspace_tensor = torch.empty((7,), dtype=torch.int64)
    workspace.metadata = {}
    workspace._destroyed = True
    return workspace


@pytest.mark.cpu_only
def test_cake_backend_routes_to_the_cake_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        cake_moe_finalize_comm, "run_cake_moe_finalize", lambda **kw: calls.append(kw)
    )
    monkeypatch.setattr(
        trtllm_ar,
        "get_trtllm_comm_module",
        lambda: pytest.fail("Cake dispatch loaded the TRT-LLM module"),
    )
    arguments = _arguments()

    trtllm_ar.trtllm_moe_finalize_allreduce_fusion(**arguments, backend="cake")

    assert len(calls) == 1
    assert set(calls[0]) == set(arguments) | {"weight_bias"}
    assert calls[0]["weight_bias"] is None
    for name, expected in arguments.items():
        if isinstance(expected, torch.Tensor):
            assert calls[0][name] is expected
        else:
            assert calls[0][name] == expected


@pytest.mark.cpu_only
def test_unknown_backend_fails_before_loading_a_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trtllm_ar,
        "get_trtllm_comm_module",
        lambda: pytest.fail("invalid backend loaded the TRT-LLM module"),
    )
    with pytest.raises(ValueError, match="backend must be"):
        trtllm_ar.trtllm_moe_finalize_allreduce_fusion(
            **_arguments(), backend="unknown"
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("backend_kwargs", "expected_backend"),
    [({}, "trtllm"), ({"moe_finalize_backend": "cake"}, "cake")],
)
def test_unified_finalize_forwards_selected_backend(
    monkeypatch: pytest.MonkeyPatch,
    backend_kwargs: dict[str, str],
    expected_backend: str,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        allreduce_module,
        "trtllm_moe_finalize_allreduce_fusion",
        lambda **kwargs: calls.append(kwargs),
    )
    arguments = _arguments()
    norm_out = arguments["norm_out"]

    result = allreduce_module.allreduce_fusion(
        input=arguments["allreduce_in"],
        workspace=_fake_workspace(),
        pattern=AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
        residual_in=arguments["residual_in"],
        rms_gamma=arguments["norm_weight"],
        expanded_idx_to_permuted_idx=arguments["expanded_idx_to_permuted_idx"],
        norm_out=norm_out,
        residual_out=arguments["residual_out"],
        expert_scale_factor=arguments["expert_scale_factor"],
        **backend_kwargs,
    )

    assert result is norm_out
    assert len(calls) == 1
    assert calls[0]["backend"] == expected_backend


@pytest.mark.cpu_only
def test_unified_selector_is_not_forwarded_to_other_patterns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        allreduce_module,
        "trtllm_allreduce_fusion",
        lambda **kwargs: calls.append(kwargs),
    )
    input = torch.empty((1, 16), dtype=torch.float16)
    output = torch.empty_like(input)

    result = allreduce_module.allreduce_fusion(
        input=input,
        workspace=_fake_workspace(),
        pattern=AllReduceFusionPattern.kAllReduce,
        output=output,
        moe_finalize_backend="cake",
    )

    assert result is output
    assert len(calls) == 1
    assert "backend" not in calls[0]


@pytest.mark.cpu_only
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_small_token_max_reaches_the_binding(
    monkeypatch: pytest.MonkeyPatch, world_size: int
) -> None:
    calls: list[tuple[object, ...]] = []
    arches: list[str] = []

    class _Module:
        def cake_moe_finalize_allreduce_fusion(self, *args: object) -> None:
            calls.append(args)

    def _get_module(arch: str) -> _Module:
        arches.append(arch)
        return _Module()

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(cake_moe_finalize_comm, "target_arch", lambda index: "sm_103a")
    monkeypatch.setattr(
        cake_moe_finalize_comm, "get_cake_moe_finalize_module", _get_module
    )
    arguments = _arguments()
    arguments["world_size"] = world_size

    cake_moe_finalize_comm.run_cake_moe_finalize(**arguments, weight_bias=None)

    assert arches == ["sm_103a"]
    assert len(calls) == 1
    # Positional binding arguments: world_rank, world_size, small_token_max.
    assert calls[0][11:14] == (0, world_size, cake_moe_finalize_comm.SMALL_TOKEN_MAX)


@pytest.mark.cpu_only
def test_module_records_match_the_binding() -> None:
    source_dir = cake_moe_finalize_comm._source_dir()
    binding = (
        source_dir / f"{cake_moe_finalize_comm.SOURCE_PACKAGE}_binding.cu"
    ).read_text()
    declared = set(
        re.findall(r"^CAKE_MOE_FINALIZE_KERNEL\([^,]+, (\w+)\)$", binding, re.M)
    )
    assert cake_moe_finalize_comm.MODULES
    for name, record in cake_moe_finalize_comm.MODULES.items():
        assert record["kernel"] == f"kernel_{name}"
        assert record["kernel"] in declared
        assert set(record["arches"]) <= {"sm_100a", "sm_103a"}
        for source in record["sources"]:
            assert (source_dir / source).is_file(), source


def _program_table(binding: str, name: str) -> list[str | None]:
    """Entries of one [dtype][world size index][quant epilogue] table of the binding, in
    order (float16 then bfloat16; world sizes 2, 4, 8; o110 then o111)."""
    start = binding.index(f"const void* const {name}[2][3][2] = {{")
    body = binding[start : binding.index("};", start)]
    entries = re.findall(r"reinterpret_cast<const void\*>\((\w+)\)|(nullptr)", body)
    assert len(entries) == 12, (name, entries)
    return [kernel or None for kernel, _ in entries]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("table", "suffix", "world_sizes", "empty"),
    [
        ("kSmallKernels", "_small", (2, 4, 8), ()),
        ("kKernels", "", (2, 4), ()),
        ("kLargeTokenKernels", "_lt", (2,), ()),
        ("kLargeTokenSharedKernels", "_lts", (2,), ()),
        ("kTwoShotKernels", "_ts", (4, 8), ()),
        ("kTwoShotDeferKernels", "_tsd", (4, 8), ()),
        # The float16 o110 deferred shared program of world size 8 is not built.
        ("kTwoShotDeferSharedKernels", "_tsds", (4, 8), (("float16", 8, "110"),)),
        ("kTwoShotEarlyKernels", "_tse", (4, 8), ()),
        ("kTwoShotSharedKernels", "_tss", (4, 8), ()),
        ("kTwoShotEarlySharedKernels", "_tses", (4,), ()),
        ("kW2QuadKernels", "_quad", (2,), ()),
    ],
)
def test_program_tables_list_one_program_kind(
    table: str,
    suffix: str,
    world_sizes: tuple[int, ...],
    empty: tuple[tuple[str, int, str], ...],
) -> None:
    source_dir = cake_moe_finalize_comm._source_dir()
    binding = (
        source_dir / f"{cake_moe_finalize_comm.SOURCE_PACKAGE}_binding.cu"
    ).read_text()
    entries = iter(_program_table(binding, table))
    for dtype in ("float16", "bfloat16"):
        for world_size in (2, 4, 8):
            for output_profile in ("110", "111"):
                expected = (
                    f"kernel_cake_trtllm_moe_finalize_{dtype}_ws{world_size}"
                    f"_o{output_profile}{suffix}"
                    if world_size in world_sizes
                    and (dtype, world_size, output_profile) not in empty
                    else None
                )
                assert next(entries) == expected


@pytest.mark.cpu_only
def test_program_tables_cover_the_module_records() -> None:
    source_dir = cake_moe_finalize_comm._source_dir()
    binding = (
        source_dir / f"{cake_moe_finalize_comm.SOURCE_PACKAGE}_binding.cu"
    ).read_text()
    tabled = [
        kernel
        for table in (
            "kSmallKernels",
            "kKernels",
            "kLargeTokenKernels",
            "kLargeTokenSharedKernels",
            "kTwoShotKernels",
            "kTwoShotDeferKernels",
            "kTwoShotDeferSharedKernels",
            "kTwoShotEarlyKernels",
            "kTwoShotSharedKernels",
            "kTwoShotEarlySharedKernels",
            "kW2QuadKernels",
        )
        for kernel in _program_table(binding, table)
        if kernel is not None
    ]
    assert len(tabled) == len(set(tabled))
    assert set(tabled) == {
        record["kernel"] for record in cake_moe_finalize_comm.MODULES.values()
    }


def _small_token_test(binding: str) -> tuple[str, str]:
    """The binding's small_token_extended class term and the condition of the branch
    that runs the small-token program, as Python expressions over the call's world_size,
    top_k, v_has_shared_expert, tokens and small_token_max, the binding's constant
    kSmallTokenExtendedMax and the device's small-token cluster capacity (capacity)."""
    extended = re.search(r"const bool small_token_extended =\s*([^;]+);", binding)
    branch = re.search(
        r"\} else if \((tokens <= small_token_max \|\|.*?)\) \{", binding, re.S
    )
    assert extended is not None and branch is not None
    expressions = tuple(
        " ".join(match.group(1).split())
        .replace("cluster_capacity(small_token)", "capacity")
        .replace("&&", "and")
        .replace("||", "or")
        for match in (extended, branch)
    )
    names = {
        "world_size",
        "top_k",
        "v_has_shared_expert",
        "tokens",
        "small_token_max",
        "small_token_extended",
        "kSmallTokenExtendedMax",
        "capacity",
        "and",
        "or",
    }
    for expression in expressions:
        assert set(re.findall(r"[A-Za-z_]\w*", expression)) <= names, expression
        assert set(re.sub(r"[\w\s]|==|!=|<=|\(|\)", "", expression)) == set()
    return expressions


@pytest.mark.cpu_only
@pytest.mark.parametrize("small_token_max", [16, 64])
def test_small_token_extension_covers_every_class_at_world_size_4(
    small_token_max: int,
) -> None:
    """Above small_token_max, up to kSmallTokenExtendedMax tokens within the small-token
    program's cluster capacity run the small-token program at world size 4 for every
    class, at world size 2 for every class but top_k 4 without a shared expert, and
    never at world size 8."""
    source_dir = cake_moe_finalize_comm._source_dir()
    binding = (
        source_dir / f"{cake_moe_finalize_comm.SOURCE_PACKAGE}_binding.cu"
    ).read_text()
    extended_max = re.findall(
        r"^constexpr int64_t kSmallTokenExtendedMax = (\d+);$", binding, re.M
    )
    assert extended_max == ["71"]
    extended_expression, branch_expression = _small_token_test(binding)
    extended_code = compile(extended_expression, "small_token_extended", "eval")
    branch_code = compile(branch_expression, "small_token_branch", "eval")
    for world_size in (2, 4, 8):
        for top_k in (4, 8):
            for use_shared in (False, True):
                scope: dict[str, object] = {
                    "world_size": world_size,
                    "top_k": top_k,
                    "v_has_shared_expert": int(use_shared),
                    "small_token_max": small_token_max,
                    "kSmallTokenExtendedMax": 71,
                }
                extended = world_size == 4 or (
                    world_size == 2 and (top_k != 4 or use_shared)
                )
                scope["small_token_extended"] = eval(
                    extended_code, {"__builtins__": {}}, scope
                )
                assert scope["small_token_extended"] == extended
                for capacity in (0, small_token_max + 1, 70, 71, 148):
                    for tokens in range(1, 80):
                        scope.update(tokens=tokens, capacity=capacity)
                        expected = tokens <= small_token_max or (
                            extended and tokens <= 71 and tokens <= capacity
                        )
                        assert (
                            eval(branch_code, {"__builtins__": {}}, scope) == expected
                        ), (world_size, top_k, use_shared, capacity, tokens)


def _as_python(expression: str) -> str:
    """A selection expression of the binding as Python: its (world_size == 8 ? A : B)
    choices, an outer (C) ? A : B, the null tests and the logical operators."""
    expression = " ".join(expression.split())
    expression = re.sub(
        r"\(world_size == 8 \? (\w+) : (\w+)\)",
        r"(\1 if world_size == 8 else \2)",
        expression,
    )
    outer = re.fullmatch(r"\((.*)\) \? (\w+) : (\w+)", expression)
    if outer is not None:
        expression = f"({outer.group(2)} if ({outer.group(1)}) else {outer.group(3)})"
    return (
        expression.replace(" != nullptr", " is not None")
        .replace("&&", "and")
        .replace("||", "or")
        .replace("!(", "not (")
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("world_size", [4, 8])
def test_two_shot_early_program_runs_up_to_128_tokens(world_size: int) -> None:
    """At small_token_max 16, the binding runs a two-shot batch of up to 128 tokens on
    the two-shot early program and one of 129 tokens on the two-shot program, at world
    sizes 4 and 8 (and the world size 8 batches above small_token_max that the
    small-token program does not take, from 17 tokens); the choice reads the token count
    and the world size alone."""
    source_dir = cake_moe_finalize_comm._source_dir()
    binding = (
        source_dir / f"{cake_moe_finalize_comm.SOURCE_PACKAGE}_binding.cu"
    ).read_text()
    constants = {
        name: int(value)
        for name, value in re.findall(
            r"^constexpr int64_t (kTwoShot\w+) = (\d+);$", binding, re.M
        )
    }
    assert constants["kTwoShotEarlyMaxTokensWorldSize4"] == 128
    assert constants["kTwoShotEarlyMaxTokensWorldSize8"] == 128
    early_now = re.search(r"const void\* two_shot_now =(.*?);", binding, re.S)
    first = re.search(r"if \((two_shot != nullptr &&.*?)\) \{", binding, re.S)
    above_small = re.search(
        r"\} else if \((two_shot != nullptr && world_size == 8)\) \{", binding
    )
    assert early_now is not None and first is not None and above_small is not None
    two_shot_kernel = re.findall(
        r"kernel = two_shot_deferred \? .*? : two_shot_now\);", binding
    )
    assert len(two_shot_kernel) == 2
    _, branch_expression = _small_token_test(binding)
    codes = {
        name: compile(_as_python(expression), name, "eval")
        for name, expression in (
            ("two_shot_now", early_now.group(1)),
            ("two_shot_first", first.group(1)),
            ("two_shot_above_small", above_small.group(1)),
            ("small_token", branch_expression),
        )
    }

    def program(tokens: int) -> str:
        scope: dict[str, object] = {
            **constants,
            "world_size": world_size,
            "top_k": 8,
            "v_has_shared_expert": 1,
            "tokens": tokens,
            "small_token_max": 16,
            "small_token_extended": world_size == 4,
            "kSmallTokenExtendedMax": 71,
            "capacity": 0,
            "two_shot": "_ts",
            "two_shot_early": "_tse",
        }
        if eval(codes["two_shot_first"], {"__builtins__": {}}, scope):
            return eval(codes["two_shot_now"], {"__builtins__": {}}, scope)
        if eval(codes["small_token"], {"__builtins__": {}}, scope):
            return "_small"
        if eval(codes["two_shot_above_small"], {"__builtins__": {}}, scope):
            return eval(codes["two_shot_now"], {"__builtins__": {}}, scope)
        return "persistent"

    assert program(128) == "_tse"
    assert program(129) == "_ts"
    assert program(16) == "_small"
    assert program(17) == ("_tse" if world_size == 8 else "persistent")


@pytest.mark.cpu_only
@pytest.mark.parametrize("small_token_max", [16, 64])
def test_world_size_2_quad_program_runs_up_to_142_tokens(small_token_max: int) -> None:
    """At world size 2, a batch the small-token program does not take runs the quad
    program up to 142 tokens and the persistent program from 143 tokens, below the
    large-token program's range; the choice reads the token count and the world size
    alone. World sizes 4 and 8 have no quad program."""
    source_dir = cake_moe_finalize_comm._source_dir()
    binding = (
        source_dir / f"{cake_moe_finalize_comm.SOURCE_PACKAGE}_binding.cu"
    ).read_text()
    constants = {
        name: int(value)
        for name, value in re.findall(
            r"^constexpr int64_t (k\w+) = (\d+);$", binding, re.M
        )
    }
    assert constants["kW2QuadMaxTokens"] == 142
    quad = re.search(
        r"\} else if \((w2_quad != nullptr && tokens <= kW2QuadMaxTokens)\) \{", binding
    )
    large = re.search(
        r"\} else if \((large_token != nullptr &&.*?)\) \{", binding, re.S
    )
    assert quad is not None and large is not None
    # The quad test follows the large-token test and precedes the persistent default.
    assert large.start() < quad.start()
    assert re.search(
        r"kernel = w2_quad;\s*\} else \{\s*kernel = kKernels\[", binding[quad.start() :]
    )
    _, branch_expression = _small_token_test(binding)
    shared_choice = re.compile(r"\(v_has_shared_expert != 0 \? (\w+) : (\w+)\)")
    codes = {
        name: compile(
            _as_python(
                shared_choice.sub(
                    r"(\1 if v_has_shared_expert != 0 else \2)",
                    " ".join(expression.split()),
                )
            ),
            name,
            "eval",
        )
        for name, expression in (
            ("quad", quad.group(1)),
            ("large_token", large.group(1)),
            ("small_token", branch_expression),
        )
    }

    def program(world_size: int, tokens: int, top_k: int, use_shared: bool) -> str:
        scope: dict[str, object] = {
            **constants,
            "world_size": world_size,
            "top_k": top_k,
            "v_has_shared_expert": int(use_shared),
            "tokens": tokens,
            "small_token_max": small_token_max,
            "small_token_extended": world_size == 4
            or (world_size == 2 and (top_k != 4 or use_shared)),
            "capacity": 71,
            "large_token": "_lt" if world_size == 2 else None,
            "w2_quad": "_quad" if world_size == 2 else None,
        }
        if eval(codes["small_token"], {"__builtins__": {}}, scope):
            return "_small"
        if eval(codes["large_token"], {"__builtins__": {}}, scope):
            return "_lt"
        if eval(codes["quad"], {"__builtins__": {}}, scope):
            return "_quad"
        return "persistent"

    for top_k in (4, 8):
        for use_shared in (False, True):
            extended = top_k != 4 or use_shared
            for tokens in (64, 65, 71, 72):
                expected = (
                    "_small"
                    if tokens <= small_token_max or (extended and tokens <= 71)
                    else "_quad"
                )
                assert program(2, tokens, top_k, use_shared) == expected, (
                    top_k,
                    use_shared,
                    tokens,
                )
            assert program(2, 141, top_k, use_shared) == "_quad"
            assert program(2, 142, top_k, use_shared) == "_quad"
            assert program(2, 143, top_k, use_shared) == "persistent"
            assert program(2, small_token_max + 1, top_k, use_shared) == (
                "_small" if extended and small_token_max < 71 else "_quad"
            )
            for world_size in (4, 8):
                assert program(world_size, 100, top_k, use_shared) != "_quad"


def _two_shot_select(binding: str) -> str:
    """The binding's choice among the deferred two-shot programs,
    two_shot_now_shared and two_shot_now, at both two-shot call sites, as a Python
    expression."""
    selects = re.findall(r"kernel = (two_shot_deferred \? .*?);", binding)
    assert len(selects) == 2 and selects[0] == selects[1], selects
    select = re.fullmatch(
        r"two_shot_deferred \? \(\((.*?)\) \? (\w+) : (\w+)\)"
        r" : \(\((.*)\) \? (\w+) : (\w+)\)",
        selects[0],
    )
    assert select is not None, selects[0]
    shared, shared_program, defer_program, now_shared, now_program, now = (
        select.groups()
    )
    assert (shared_program, defer_program, now_program, now) == (
        "defer_shared",
        "two_shot_defer",
        "two_shot_now_shared",
        "two_shot_now",
    )
    return (
        f"(({shared_program} if ({_as_python(shared)}) else {defer_program})"
        f" if two_shot_deferred else"
        f" ({now_program} if ({_as_python(now_shared)}) else {now}))"
    )


def _program_suffix(kernel: str | None) -> str | None:
    """The program suffix of a table entry (None for an empty slot)."""
    if kernel is None:
        return None
    match = re.fullmatch(r"kernel_cake_trtllm_moe_finalize_\w+?_o11[01](\w*)", kernel)
    assert match is not None, kernel
    return match.group(1)


def _table_slot(dtype: str, world_size: int, quantize: bool) -> int:
    """The index of a (dtype, world size, epilogue) slot in _program_table's order."""
    return (
        ("float16", "bfloat16").index(dtype) * 3 + (2, 4, 8).index(world_size)
    ) * 2 + int(quantize)


def _two_shot_model(small_token_max: int):
    """The binding's program choice for a call, as a function of (world_size, tokens,
    top_k, use_shared, quantize, dtype) returning the program suffix ("_small" for the
    small-token program, "one_shot" for any other one-shot program), evaluated from the
    binding text: the two-shot tests, the small-token test, the deferred and early
    switches, the shared forms' tables and the two-shot select at both call sites."""
    source_dir = cake_moe_finalize_comm._source_dir()
    binding = (
        source_dir / f"{cake_moe_finalize_comm.SOURCE_PACKAGE}_binding.cu"
    ).read_text()
    constants = {
        name: int(value)
        for name, value in re.findall(
            r"^constexpr int64_t (k\w+) = (\d+);$", binding, re.M
        )
    }
    assert constants["kTwoShotDeferMinTokensWorldSize4"] == 1536
    assert constants["kTwoShotDeferMinTokensWorldSize8"] == 768
    expressions = {
        name: re.search(rf"const (?:bool|void\*) {name} =\s*([^;]+);", binding)
        for name in (
            "defer_excluded",
            "two_shot_deferred",
            "two_shot_now",
            "two_shot_early_class",
        )
    }
    now_shared = re.search(
        r"const void\* two_shot_now_shared =\s*(\w+) \? (\w+)\[dtype_index\]"
        r"\[ws_index\]\[quant_index\]\s*: (\w+)\[dtype_index\]\[ws_index\]"
        r"\[quant_index\];",
        binding,
    )
    assert now_shared is not None
    assert now_shared.groups() == (
        "two_shot_early_class",
        "kTwoShotEarlySharedKernels",
        "kTwoShotSharedKernels",
    )
    first = re.search(r"if \((two_shot != nullptr &&.*?)\) \{", binding, re.S)
    above_small = re.search(
        r"\} else if \((two_shot != nullptr && world_size == 8)\) \{", binding
    )
    assert first is not None and above_small is not None
    assert all(match is not None for match in expressions.values())
    _, branch_expression = _small_token_test(binding)
    sources = {name: match.group(1) for name, match in expressions.items()}
    sources.update(
        two_shot_first=first.group(1),
        two_shot_above_small=above_small.group(1),
        small_token=branch_expression,
    )
    codes = {
        name: compile(re.sub(r"!(?!=)", "not ", _as_python(source)), name, "eval")
        for name, source in sources.items()
    }
    select = compile(_two_shot_select(binding), "two_shot_select", "eval")
    shared_table = _program_table(binding, "kTwoShotDeferSharedKernels")
    now_table = _program_table(binding, "kTwoShotSharedKernels")
    early_table = _program_table(binding, "kTwoShotEarlySharedKernels")

    def program(
        world_size: int,
        tokens: int,
        top_k: int,
        use_shared: bool,
        quantize: bool,
        dtype: str,
    ) -> str:
        two_shot_world = world_size in (4, 8)
        slot = _table_slot(dtype, world_size, quantize)
        shared_kernel = shared_table[slot]
        scope: dict[str, object] = {
            **constants,
            "world_size": world_size,
            "top_k": top_k,
            "v_has_shared_expert": int(use_shared),
            "quantize": quantize,
            "tokens": tokens,
            "small_token_max": small_token_max,
            "small_token_extended": world_size == 4
            or (world_size == 2 and (top_k != 4 or use_shared)),
            "capacity": 71,
            "two_shot": "_ts" if two_shot_world else None,
            "two_shot_defer": "_tsd" if two_shot_world else None,
            "defer_shared": _program_suffix(shared_kernel),
            "two_shot_early": "_tse" if two_shot_world else None,
        }
        for name in (
            "defer_excluded",
            "two_shot_deferred",
            "two_shot_now",
            "two_shot_early_class",
        ):
            scope[name] = eval(codes[name], {"__builtins__": {}}, scope)
        scope["two_shot_now_shared"] = _program_suffix(
            (early_table if scope["two_shot_early_class"] else now_table)[slot]
        )
        if eval(codes["two_shot_first"], {"__builtins__": {}}, scope):
            return eval(select, {"__builtins__": {}}, scope)
        if eval(codes["small_token"], {"__builtins__": {}}, scope):
            return "_small"
        if eval(codes["two_shot_above_small"], {"__builtins__": {}}, scope):
            return eval(select, {"__builtins__": {}}, scope)
        return "one_shot"

    return program


@pytest.mark.cpu_only
@pytest.mark.parametrize("small_token_max", [16, 64])
def test_shared_expert_runs_the_deferred_shared_program(
    small_token_max: int,
) -> None:
    """From the deferred two-shot threshold (kTwoShotDeferMinTokensWorldSize4 = 1536
    tokens at world size 4, kTwoShotDeferMinTokensWorldSize8 = 768 at world size 8), a
    rank with a shared expert runs the deferred program that also prefetches the
    shared-expert row (_tsds): at world size 4 for every top_k, at world size 8 for
    top_k 8 in every cell but float16 o110, whose slot is empty. Every other deferred call
    runs the deferred program (_tsd); below the threshold a rank with a shared expert
    at top_k 4 runs the shared form of the two-shot program (_tss) or, at world size 4,
    of the two-shot early program (_tses).
    Whether a call runs a deferred form reads the token count, the world size and the
    C11 defer_excluded class alone, so ranks with and without a shared expert, or with
    different top_k, run forms of the same exchange. World size 2 runs no two-shot form."""
    source_dir = cake_moe_finalize_comm._source_dir()
    binding = (
        source_dir / f"{cake_moe_finalize_comm.SOURCE_PACKAGE}_binding.cu"
    ).read_text()
    constants = {
        name: int(value)
        for name, value in re.findall(
            r"^constexpr int64_t (k\w+) = (\d+);$", binding, re.M
        )
    }
    assert constants["kTwoShotDeferMinTokensWorldSize4"] == 1536
    assert constants["kTwoShotDeferMinTokensWorldSize8"] == 768
    program = _two_shot_model(small_token_max)

    def now_shared(form: str, top_k: int, use_shared: bool) -> str:
        return form + "s" if use_shared and top_k == 4 else form

    def world_size_8(
        tokens: int, top_k: int, use_shared: bool, quantize: bool, dtype: str
    ) -> str:
        if tokens < 768 or (top_k == 4 and not use_shared and not quantize):
            return now_shared("_ts", top_k, use_shared)
        built = dtype == "bfloat16" or quantize
        return "_tsds" if use_shared and top_k == 8 and built else "_tsd"

    for dtype in ("float16", "bfloat16"):
        for quantize in (False, True):
            for top_k in (4, 8):
                for use_shared in (False, True):
                    case = (top_k, use_shared, quantize, dtype)
                    now_ts = now_shared("_ts", top_k, use_shared)
                    now_tse = now_shared("_tse", top_k, use_shared)
                    for tokens in (1535, 1536, 1537, 2048, 4096):
                        deferred = "_tsds" if use_shared else "_tsd"
                        assert program(
                            4, tokens, top_k, use_shared, quantize, dtype
                        ) == (deferred if tokens >= 1536 else now_ts), (tokens, *case)
                    for tokens in (72, 128):
                        assert (
                            program(4, tokens, top_k, use_shared, quantize, dtype)
                            == now_tse
                        )
                    for tokens in (767, 768, 769, 1536, 2048, 4096):
                        assert program(
                            8, tokens, top_k, use_shared, quantize, dtype
                        ) == world_size_8(tokens, *case), (tokens, *case)
                    for tokens in (1535, 1536, 2048):
                        assert program(
                            2, tokens, top_k, use_shared, quantize, dtype
                        ) not in ("_ts", "_tsd", "_tsds", "_tse", "_tss", "_tses")
                # Mixed shared-expert presence among the four ranks of one call.
                for present in ({0}, {0, 1}, {2, 3}, {1, 2, 3}, {0, 2}):
                    for tokens in (1535, 1536, 2048):
                        programs = [
                            program(4, tokens, top_k, rank in present, quantize, dtype)
                            for rank in range(4)
                        ]
                        expected = [
                            ("_tsds" if rank in present else "_tsd")
                            if tokens >= 1536
                            else now_shared("_ts", top_k, rank in present)
                            for rank in range(4)
                        ]
                        assert programs == expected, (present, tokens, top_k, quantize)
            # Mixed shared-expert presence and top_k among the eight ranks of one call.
            for present in (
                {0},
                {0, 1, 2, 3},
                {4, 5, 6, 7},
                set(range(1, 8)),
                {0, 2, 4, 6},
            ):
                for top_ks in ((8,) * 8, (4,) * 4 + (8,) * 4, (4, 8) * 4):
                    for tokens in (767, 768, 2048):
                        programs = [
                            program(
                                8,
                                tokens,
                                top_ks[rank],
                                rank in present,
                                quantize,
                                dtype,
                            )
                            for rank in range(8)
                        ]
                        expected = [
                            world_size_8(
                                tokens, top_ks[rank], rank in present, quantize, dtype
                            )
                            for rank in range(8)
                        ]
                        assert programs == expected, (present, top_ks, tokens, quantize)


@pytest.mark.cpu_only
@pytest.mark.parametrize("small_token_max", [16, 64])
def test_world_size_2_shared_expert_runs_the_large_token_shared_program(
    small_token_max: int,
) -> None:
    """At world size 2, a top_k 4 batch the large-token program takes (above
    kLargeTokenMinTokens = 512 tokens, kLargeTokenMinTokensSharedExpert = 895 with a
    shared expert) runs the large-token program that also prefetches the shared-expert row
    (_lts) on a rank with a shared expert and the large-token program (_lt) on a rank
    without one; the shared form runs the same exchange, so the ranks of one call may mix
    them. The large-token class itself is unchanged; world sizes 4 and 8 run neither."""
    source_dir = cake_moe_finalize_comm._source_dir()
    binding = (
        source_dir / f"{cake_moe_finalize_comm.SOURCE_PACKAGE}_binding.cu"
    ).read_text()
    constants = {
        name: int(value)
        for name, value in re.findall(
            r"^constexpr int64_t (k\w+) = (\d+);$", binding, re.M
        )
    }
    assert constants["kLargeTokenMinTokens"] == 512
    assert constants["kLargeTokenMinTokensSharedExpert"] == 895
    large = re.search(
        r"\} else if \((large_token != nullptr &&.*?)\) \{\s*kernel = (.*?);",
        binding,
        re.S,
    )
    assert large is not None
    shared_choice = re.compile(r"\(v_has_shared_expert != 0 \? (\w+) : (\w+)\)")
    condition = compile(
        _as_python(
            shared_choice.sub(
                r"(\1 if v_has_shared_expert != 0 else \2)",
                " ".join(large.group(1).split()),
            )
        ),
        "large_token",
        "eval",
    )
    choice = compile(_as_python(large.group(2)), "large_token_choice", "eval")
    lt_table = _program_table(binding, "kLargeTokenKernels")
    lts_table = _program_table(binding, "kLargeTokenSharedKernels")
    _, branch_expression = _small_token_test(binding)
    small_token = compile(branch_expression, "small_token", "eval")

    def program(
        world_size: int,
        tokens: int,
        top_k: int,
        use_shared: bool,
        quantize: bool,
        dtype: str,
    ) -> str | None:
        slot = _table_slot(dtype, world_size, quantize)
        scope: dict[str, object] = {
            **constants,
            "world_size": world_size,
            "top_k": top_k,
            "v_has_shared_expert": int(use_shared),
            "tokens": tokens,
            "small_token_max": small_token_max,
            "small_token_extended": world_size == 4
            or (world_size == 2 and (top_k != 4 or use_shared)),
            "capacity": 71,
            "large_token": _program_suffix(lt_table[slot]),
            "large_token_shared": _program_suffix(lts_table[slot]),
        }
        if eval(small_token, {"__builtins__": {}}, scope):
            return "_small"
        if eval(condition, {"__builtins__": {}}, scope):
            return eval(choice, {"__builtins__": {}}, scope)
        return None

    for dtype in ("float16", "bfloat16"):
        for quantize in (False, True):
            for use_shared in (False, True):
                case = (use_shared, quantize, dtype)
                large_min = 895 if use_shared else 512
                for tokens in (512, 513, 895, 896, 1280, 1281, 2048, 4096):
                    expected = (
                        ("_lts" if use_shared else "_lt")
                        if tokens > large_min
                        else None
                    )
                    assert program(2, tokens, 4, use_shared, quantize, dtype) == (
                        expected
                    ), (tokens, *case)
                    assert program(2, tokens, 8, use_shared, quantize, dtype) is None
                    for world_size in (4, 8):
                        assert (
                            program(world_size, tokens, 4, use_shared, quantize, dtype)
                            is None
                        ), (world_size, tokens, *case)
            # Mixed shared-expert presence among the two ranks of one call.
            for present in ({0}, {1}):
                for tokens in (896, 1281, 2048):
                    programs = [
                        program(2, tokens, 4, rank in present, quantize, dtype)
                        for rank in range(2)
                    ]
                    assert programs == [
                        "_lts" if rank in present else "_lt" for rank in range(2)
                    ], (present, tokens, quantize, dtype)


@pytest.mark.cpu_only
@pytest.mark.parametrize("small_token_max", [16, 64])
def test_shared_expert_at_top_k_4_runs_the_two_shot_shared_programs(
    small_token_max: int,
) -> None:
    """Below the deferred two-shot threshold, a rank with a shared expert at top_k 4
    runs the two-shot program that also prefetches the shared-expert row (_tss) in
    place of the two-shot program (_ts) at world sizes 4 and 8, and the two-shot early
    shared program (_tses) in place of the two-shot early program (_tse) at world size
    4, in every cell. World size 8 has no two-shot early shared program: its two-shot
    early batches run _tse for every rank. Every other call (no shared expert, top_k 8,
    the deferred range, world size 2) runs what it ran before. The two-shot test, the
    deferred and the early switches still read the token count and the world size
    alone, so ranks with and without a shared expert, or with different top_k, run
    forms of the same exchange: checked on both sides of every two-shot class edge and
    for mixed ranks."""
    program = _two_shot_model(small_token_max)
    edges = {4: (71, 72, 128, 129, 1535, 1536), 8: (55, 56, 128, 129, 767, 768)}
    first = {4: 72, 8: min(56, small_token_max + 1)}

    def expected(
        world_size: int, tokens: int, top_k: int, use_shared: bool, quantize: bool
    ) -> str:
        if tokens < first[world_size]:
            return "_small"
        deferred_from = 1536 if world_size == 4 else 768
        excluded = world_size == 8 and top_k == 4 and not use_shared and not quantize
        if tokens >= deferred_from and not excluded:
            return "deferred"
        form = "_tse" if tokens <= 128 else "_ts"
        if use_shared and top_k == 4 and not (world_size == 8 and form == "_tse"):
            return form + "s"
        return form

    for dtype in ("float16", "bfloat16"):
        for quantize in (False, True):
            for world_size, tokens_list in edges.items():
                for tokens in tokens_list:
                    for top_k in (4, 8):
                        for use_shared in (False, True):
                            got = program(
                                world_size, tokens, top_k, use_shared, quantize, dtype
                            )
                            want = expected(
                                world_size, tokens, top_k, use_shared, quantize
                            )
                            case = (world_size, tokens, top_k, use_shared, quantize)
                            if want == "deferred":
                                assert got in ("_tsd", "_tsds"), (*case, dtype, got)
                            else:
                                assert got == want, (*case, dtype, got)
                    # Mixed shared-expert presence and top_k among one call's ranks.
                    ranks = range(world_size)
                    for present in ({0}, set(ranks) - {0}, set(ranks[::2])):
                        for top_ks in (
                            (4,) * world_size,
                            (4, 8) * (world_size // 2),
                        ):
                            programs = [
                                program(
                                    world_size,
                                    tokens,
                                    top_ks[rank],
                                    rank in present,
                                    quantize,
                                    dtype,
                                )
                                for rank in ranks
                            ]
                            for rank, got in zip(ranks, programs, strict=True):
                                want = expected(
                                    world_size,
                                    tokens,
                                    top_ks[rank],
                                    rank in present,
                                    quantize,
                                )
                                if want in ("deferred", "_small"):
                                    continue
                                assert got == want, (present, top_ks, tokens, rank)
            for tokens in (72, 129, 1535):
                for top_k in (4, 8):
                    for use_shared in (False, True):
                        assert program(
                            2, tokens, top_k, use_shared, quantize, dtype
                        ) not in ("_ts", "_tse", "_tss", "_tses")
