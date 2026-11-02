"""Pool geometry does not affect compiled program identity in offline integration cases.

Compile planning runs in a spawned compiler worker without a GPU. Explicit KDA,
paged-attention, and sparse-MLA declarations are checked at different pool sizes.
Structural checks cover the pool fields excluded from each family's selection key.
"""

from __future__ import annotations

import json
import multiprocessing
from collections.abc import Callable
from typing import NamedTuple

import pytest

from b12x.preparation import DeviceIdentity, FrozenMapping
from b12x.preparation.device import DetectedDevice

IDENTITY = DeviceIdentity("nvidia", (12, 0), 170, "NVIDIA GeForce RTX 5090")
DEVICE = DetectedDevice(
    ordinal=0,
    identity=IDENTITY,
    uuid="synthetic-sm120",
    max_shared_memory_per_block=232448,
    max_shared_memory_per_multiprocessor=233472,
)
PROBE_TIMEOUT_SECONDS = 900


def _grown(value: int) -> int:
    """A pool about two percent larger, as a cached restart observes."""
    return value + max(1, value // 50)


def _realigned(value: int) -> int:
    """A larger pool whose count changes its 16-divisibility and 1-ness.

    Triton specializes integer arguments on equality with one and on
    divisibility by sixteen, so a key that survives ``_grown`` may still
    change when the pool count crosses one of those classes.
    """
    grown = _grown(value)
    if value % 16 == 0:
        return grown | 1
    return (grown + 16) & ~15


PERTURBATIONS = {"grown": _grown, "realigned": _realigned}


# --- Family declarations; every callable below runs inside the compiler worker. ---


def _cuda(ordinal=0):
    import torch

    return torch.device("cuda", ordinal)


def _declare_gdn_decode(spec, pool):
    from b12x.sequence.gdn_decode import _preparation as m
    from b12x.sequence.gdn_decode._tuning import GdnQuery

    query = GdnQuery(**{**spec["query"], **pool})
    return m.plan(m._caps(query, 0), invocation=FrozenMapping(spec["invocation"]))


def _declare_kda_decode(spec, pool):
    from b12x.sequence.gdn_decode import _impl, _preparation as m
    from b12x.sequence.gdn_decode._tuning import _OPERANDS, _default_kda_strides

    heads = spec["heads"]
    caps = _impl.Caps(
        device=_cuda(),
        max_tokens=spec["max_tokens"],
        max_seqs=spec["max_seqs"],
        max_state_slots=pool["max_state_slots"],
        key_heads=heads,
        value_heads=heads,
        state_index_columns=spec["state_index_columns"],
        gate_activation="sigmoid",
        qk_l2norm=True,
        null_state_index=0,
    )
    invocation = FrozenMapping(
        {
            "a_log_dtype": "float32",
            "dt_bias_dtype": "float32",
            "norm_weight_dtype": "bfloat16",
            "state_indices_dtype": "int32",
            "kda_strides": _default_kda_strides(heads, heads),
            "pointer_alignments": {name: 16 for name in _OPERANDS},
        }
    )
    return m.plan(caps, invocation=invocation)


def _declare_gdn_prefill(spec, pool):
    from b12x.sequence._shared.delta_prefill.preparation import _caps_from_query
    from b12x.sequence.gdn_prefill import _impl, _preparation as m
    from b12x.sequence.gdn_prefill._tuning import GdnPrefillQuery

    query = GdnPrefillQuery(**{**spec["query"], **pool})
    caps = _caps_from_query(_impl, query, 0, is_gdn=True)
    return m.make_plan(
        caps, invocation=FrozenMapping(spec["invocation"]), override=None
    )


def _declare_kda_prefill(spec, pool):
    from b12x.sequence.kda_prefill import _impl, _preparation as m

    caps = _impl.Caps(
        device=_cuda(),
        max_tokens=spec["max_tokens"],
        max_seqs=spec["max_seqs"],
        max_state_slots=pool["max_state_slots"],
        heads=spec["heads"],
        qk_l2norm=True,
        checkpoint_export=True,
        null_state_index=0,
    )
    invocation = FrozenMapping(
        {
            "a_log_dtype": "float32",
            "dt_bias_dtype": "float32",
            "state_indices_dtype": "int32",
        }
    )
    return m.make_plan(caps, invocation=invocation, override=None)


def _declare_ple(spec, pool):
    import torch
    from b12x.sequence.ple import _preparation as m
    from b12x.sequence.ple._contracts import LayerCaps

    query = {**spec["query"], **pool}
    caps = LayerCaps(
        device=_cuda(),
        mode=query["mode"],
        max_tokens=query["max_tokens"],
        max_seqs=query["max_seqs"],
        max_state_slots=query["max_state_slots"],
        max_speculative_tokens=query["max_speculative_tokens"],
        streams=query["streams"],
        hidden_size=query["hidden_size"],
        kernel_size=query["kernel_size"],
        dilation=query["dilation"],
        dtype=getattr(torch, query["dtype"]),
    )
    return m.make_plan(
        caps, invocation=FrozenMapping(spec["invocation"]), override=None
    )


def _declare_qsa(spec, pool):
    from b12x.attention.qsa import _contract as m
    from b12x.attention.qsa._tuning import QsaQuery

    query = QsaQuery(**{**spec["query"], **pool})
    return m.plan(
        m._caps_from_query(query, ordinal=0),
        invocation=FrozenMapping(spec["invocation"]),
    )


def _declare_paged(spec, pool):
    """The paged GQA declaration of vLLM's b12x attention backend for one KV pool."""
    import torch
    from b12x.attention import paged

    device = _cuda()
    dtype, kv_dtype = torch.bfloat16, getattr(torch, spec["kv_dtype"])
    heads, kv_heads, head_dim = spec["q_heads"], spec["kv_heads"], spec["head_dim"]
    page_size, batch, mode = spec["page_size"], spec["batch"], spec["mode"]
    width = spec["max_page_table_width"]
    total_q = batch if mode == "decode" else spec["total_q"]
    geometry = dict(
        device=device,
        q_dtype=dtype,
        kv_dtype=kv_dtype,
        num_q_heads=heads,
        num_kv_heads=kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        page_size=page_size,
        batch=batch,
        max_cache_page_count=width,
        window_left=-1,
    )
    if mode == "decode":
        capacity = paged.decode_graph_capacity(**geometry)
        max_work_items, max_partial_rows, copy_metadata = (
            capacity.max_work_items,
            capacity.max_partial_rows,
            True,
        )
    else:
        capacity = paged.extend_graph_capacity(**geometry, total_q_capacity=total_q)
        max_work_items, max_partial_rows, copy_metadata = (
            capacity.max_work_items,
            0,
            False,
        )
    caps = paged.Caps(
        device=device,
        mode=mode,
        dtype=dtype,
        kv_dtype=kv_dtype,
        num_q_heads=heads,
        num_kv_heads=kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        page_size=page_size,
        max_total_q=total_q,
        max_batch=batch,
        max_page_table_width=width,
        max_work_items=max_work_items,
        max_partial_rows=max_partial_rows,
        num_cache_pages=pool["num_cache_pages"],
        use_cuda_graph=spec.get("use_cuda_graph", True),
        copy_runtime_metadata=copy_metadata,
    )

    def descriptor(shape, strides, dtype):
        return {
            "shape": tuple(shape),
            "strides": tuple(strides),
            "dtype": str(dtype).removeprefix("torch."),
            "alignment": 16,
        }

    cache_shape = (pool["num_cache_pages"], page_size, kv_heads, head_dim)
    cache_strides = (page_size * kv_heads * head_dim, kv_heads * head_dim, head_dim, 1)
    operands = {
        "q": descriptor(
            (total_q, heads, head_dim), (heads * head_dim, head_dim, 1), dtype
        ),
        "k_cache": descriptor(cache_shape, cache_strides, kv_dtype),
        "v_cache": descriptor(cache_shape, cache_strides, kv_dtype),
        "output": descriptor(
            (total_q, heads, head_dim), (heads * head_dim, head_dim, 1), dtype
        ),
        "page_table": descriptor((batch, width), (width, 1), torch.int32),
        "cache_seqlens": descriptor((batch,), (1,), torch.int32),
        "cu_seqlens_q": descriptor((batch + 1,), (1,), torch.int32),
        "q2k_indices": None,
        "k_descale": None,
        "v_descale": None,
        "attention_sink_bias": None,
        "relative_attention_bias": None,
    }
    invocation = dict(paged.invocation_from_descriptors(caps, operands=operands))
    # The native route is specialized on the window, so the declaration carries
    # the same window its bindings request.
    invocation["window_left"] = geometry["window_left"]
    return paged.plan(caps, invocation=invocation)


def _declare_glm_cache_writer(spec, pool):
    """The GLM_NEXT cache-writer declaration of vLLM's sparse-MLA backend.

    The integration declares it from the bound KV pool and a probe of
    max_tokens latent rows; FakeTensor metadata stands in for those tensors.
    """
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from b12x.attention.sparse_mla import _preparation as m

    device = _cuda()
    with FakeTensorMode():
        kv_c = torch.empty(
            (spec["max_tokens"], 512), dtype=torch.bfloat16, device=device
        )
        kv_cache = torch.empty(
            (pool["num_cache_blocks"], spec["page_size"], spec["record_bytes"]),
            dtype=torch.uint8,
            device=device,
        )
        slots = torch.empty((spec["max_tokens"],), dtype=torch.int64, device=device)
        return m.plan_cache_writer(kv_c, kv_cache, slots)


def _paged_geometry(plan):
    """Pool geometry of a paged declaration: the caps payload and the KV cache ABI."""
    return {
        "caps.num_cache_pages": plan.invocation["caps"]["num_cache_pages"],
        "abi.k_cache.shape[0]": plan.query.abi["k_cache"]["shape"][0],
        "abi.v_cache.shape[0]": plan.query.abi["v_cache"]["shape"][0],
    }


class _Family(NamedTuple):
    declare: Callable[[dict, dict], object]
    contract_module: str
    query_type: str
    # Query fields that carry pool geometry; empty when it lives in the invocation.
    pool_fields: tuple[str, ...]
    geometry: Callable[[object], dict] | None = None

    def observed_geometry(self, plan, fields=None):
        if fields is None and self.geometry is not None:
            return self.geometry(plan)
        return {
            name: getattr(plan.query, name) for name in (fields or self.pool_fields)
        }


FAMILIES = {
    "attention.gdn": _Family(
        _declare_gdn_decode,
        "b12x.sequence.gdn_decode._tuning",
        "GdnQuery",
        ("max_state_slots",),
    ),
    "attention.gdn/kda": _Family(
        _declare_kda_decode,
        "b12x.sequence.gdn_decode._tuning",
        "GdnQuery",
        ("max_state_slots",),
    ),
    "sequence.gdn_prefill": _Family(
        _declare_gdn_prefill,
        "b12x.sequence.gdn_prefill._tuning",
        "GdnPrefillQuery",
        ("max_state_slots",),
    ),
    "sequence.kda_prefill": _Family(
        _declare_kda_prefill,
        "b12x.sequence.kda_prefill._tuning",
        "KdaPrefillQuery",
        ("max_state_slots",),
    ),
    "sequence.ple": _Family(
        _declare_ple, "b12x.sequence.ple._tuning", "PleQuery", ("max_state_slots",)
    ),
    "attention.qsa": _Family(
        _declare_qsa,
        "b12x.attention.qsa._tuning",
        "QsaQuery",
        ("num_main_cache_pages", "num_compressed_cache_pages"),
    ),
    "attention.gqa": _Family(
        _declare_paged, "b12x.attention.paged._tuning", "GqaQuery", (), _paged_geometry
    ),
    "attention.sparse_mla/cache_writer": _Family(
        _declare_glm_cache_writer,
        "b12x.attention.sparse_mla._tuning",
        "SparseMlaQuery",
        ("num_cache_blocks", "max_page_table_width", "max_physical_records"),
    ),
}

# Families whose declared queries carry no pool geometry, with the reason.
POOL_FREE_FAMILIES = {
    "attention.dsa_indexer": (
        "max_page_table_width and max_k_rows derive from max_model_len and the "
        "kernel page size (vllm/v1/attention/backends/mla/b12x_indexer.py), "
        "never from the KV pool"
    ),
    "attention.sparse_mla/attention": (
        "the GLM decode/extend declarations (vllm/v1/attention/backends/mla/"
        "b12x_mla_sparse.py _declare_plan) derive max_page_table_width from the "
        "selection width; num_cache_blocks and max_physical_records stay at their "
        "zero defaults, so only the cache-writer plan carries pool geometry"
    ),
    "attention.mla_compress": (
        "max_states is declared from max_num_seqs (vllm/models/deepseek_v4_1/"
        "compressor.py), not from a pool; the compile key does include it"
    ),
    "attention.compressed_sparse_mla": (
        "declared only by the DeepSeek v4.1 integration, outside the Qwen and GLM "
        "scope; its swa/indexed cache shapes come from the pool tensors and are "
        "not covered here"
    ),
    "attention.mla": (
        "no vLLM integration declares dense MLA; its query carries num_cache_pages, "
        "but its compile factory (b12x/attention/dense_mla/_preparation.py "
        "compile_dense_mla) slices CUDA-labelled FakeTensors, which initializes "
        "CUDA, so it cannot be planned in an offline compiler worker"
    ),
    "gemm.mla_query_projection": (
        "the MLA query-projection plan (vllm/model_executor/layers/attention/"
        "mla_attention.py) declares heads and row capacity only"
    ),
}


def _integration_cases():
    yield {
        "id": "attention.gdn/kda[kimi decode]",
        "family": "attention.gdn/kda",
        "spec": {"heads": 8, "max_tokens": 8, "max_seqs": 4, "state_index_columns": 2},
        "pool": {"max_state_slots": 4096},
    }
    yield {
        "id": "sequence.kda_prefill[kimi prefill m128]",
        "family": "sequence.kda_prefill",
        "spec": {"heads": 8, "max_tokens": 128, "max_seqs": 4},
        "pool": {"max_state_slots": 4096},
    }
    for mode, use_cuda_graph in (("decode", True), ("extend", True), ("extend", False)):
        yield {
            "id": f"attention.gqa[glm {mode} p64 b8{'' if use_cuda_graph else ' eager'}]",
            "family": "attention.gqa",
            "spec": {
                "mode": mode,
                "kv_dtype": "bfloat16",
                "q_heads": 16,
                "kv_heads": 2,
                "head_dim": 128,
                "page_size": 64,
                "batch": 8,
                "total_q": 128,
                "max_page_table_width": 64,
                "use_cuda_graph": use_cuda_graph,
            },
            "pool": {"num_cache_pages": 1000},
        }
    for record_bytes in (528, 304):
        yield {
            "id": f"attention.sparse_mla/cache_writer[glm record{record_bytes}]",
            "family": "attention.sparse_mla/cache_writer",
            "spec": {"max_tokens": 64, "page_size": 64, "record_bytes": record_bytes},
            "pool": {"num_cache_blocks": 1000},
        }


CASES = list(_integration_cases())


# --- Worker-side probing. ---

_PLANNED_FACTS: dict[str, object] = {}


def _record_planned_facts():
    """Keep the compile-spec facts behind every planned program for diagnostics."""
    import hashlib
    from b12x._lib import compile_plan, compiler

    if getattr(compiler._compile_disk_cache_payload, "_records_facts", False):
        return
    cute_payload = compiler._compile_disk_cache_payload

    def payload_recorded(compile_callable, func, args, kwargs, compile_spec=None):
        payload = cute_payload(compile_callable, func, args, kwargs, compile_spec)
        key = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
        _PLANNED_FACTS[key] = (
            json.loads(compile_spec.json_key)
            if compile_spec is not None
            else {"structural": repr(payload[5:7])}
        )
        return payload

    payload_recorded._records_facts = True
    compiler._compile_disk_cache_payload = payload_recorded
    triton_program = compile_plan._triton_program

    def triton_recorded(source, target=None, options=None, _env_vars=None):
        program = triton_program(source, target, options, _env_vars)
        names = source.fn.arg_names

        def named(mapping):
            return {
                ".".join(
                    names[index] if position == 0 else str(index)
                    for position, index in enumerate(path)
                )
                if isinstance(path, tuple)
                else str(path): repr(value)
                for path, value in dict(mapping or {}).items()
            }

        _PLANNED_FACTS[program.key] = {
            "constants": named(source.constants),
            "signature": named(source.signature),
            "attrs": named(source.attrs),
        }
        return program

    compile_plan._triton_program = triton_recorded


def _configurations(plan):
    configuration = plan.contract.configure(plan.query, device=IDENTITY, override=None)
    configs = {"default": configuration.default}
    for _, config in plan.contract.iterate(configuration):
        if config != configuration.default:
            configs["alternate"] = config
            break
    return configs


def _probe(case):
    """Plan every declaration of one case; returns program keys per configuration.

    Returns ``{"unplannable": reason}`` when the family's compile factory
    cannot run in an offline compiler worker.
    """
    import torch
    from b12x._lib.compile_pool import describe_compilation
    from b12x._lib.program_cache import PreparationProgramCache

    _record_planned_facts()
    family = FAMILIES[case["family"]]
    pools = {
        "base": case["pool"],
        **{
            label: {name: perturb(value) for name, value in case["pool"].items()}
            for label, perturb in PERTURBATIONS.items()
        },
    }
    programs = {}
    queries = {}
    for label, pool in pools.items():
        plan = family.declare(case["spec"], pool)
        queries[label] = family.observed_geometry(plan, case.get("observe"))
        for config_name, config in _configurations(plan).items():
            keys = set()
            for job in plan._compile_jobs(config, DEVICE):
                try:
                    scope = PreparationProgramCache()
                    with scope.activate():
                        first = describe_compilation(job)
                        misses = sum(cache._misses for cache in scope._caches.values())
                        repeated = describe_compilation(job)
                        assert first.programs == repeated.programs
                        assert (
                            sum(cache._misses for cache in scope._caches.values())
                            == misses
                        )
                    scope.clear()
                    keys.update(first.programs)
                except torch.AcceleratorError as error:
                    frame = error.__traceback__
                    while frame.tb_next is not None:
                        frame = frame.tb_next
                    location = f"{frame.tb_frame.f_code.co_filename}:{frame.tb_lineno}"
                    return {
                        "unplannable": f"{str(error).splitlines()[0]} at {location}"
                    }
            programs[(label, config_name)] = tuple(
                sorted((program.dialect, program.name, program.key) for program in keys)
            )
    facts = {
        key: _PLANNED_FACTS.get(key, {})
        for row in programs.values()
        for _, _, key in row
    }
    return {"programs": programs, "facts": facts, "queries": queries}


@pytest.fixture(scope="module")
def compiler_worker():
    """One spawned compiler child configured as a production CompilePool worker."""
    from b12x._lib import compile_pool

    context = multiprocessing.get_context("spawn")
    activity = context.Array("q", (0, 0))
    with compile_pool._offline_compiler_spawn_environment():
        pool = context.Pool(
            processes=1,
            initializer=compile_pool._initialize_worker,
            initargs=(
                DEVICE.ordinal,
                IDENTITY.compute_capability,
                DEVICE.uuid,
                IDENTITY.product_name,
                IDENTITY.sm_count,
                DEVICE.max_shared_memory_per_block,
                DEVICE.max_shared_memory_per_multiprocessor,
                activity,
            ),
        )
    try:
        yield pool
    finally:
        pool.terminate()
        pool.join()


def _fact_differences(before, after, path=""):
    """The smallest containers that hold a differing leaf, as `path: before != after`."""
    if type(before) is not type(after) or not isinstance(before, (dict, list)):
        return [f"{path or '.'}: {before!r} != {after!r}"]
    items = (
        ((key, before[key], after.get(key)) for key in before)
        if isinstance(before, dict)
        else (
            (index, left, right)
            for index, (left, right) in enumerate(zip(before, after, strict=False))
        )
    )
    lines = []
    for key, left, right in items:
        if left == right:
            continue
        nested = _fact_differences(left, right, f"{path}/{key}")
        if not isinstance(left, (dict, list)) or len(nested) > 1:
            lines.append(f"{path}/{key}: {left!r:.160} != {right!r:.160}")
        else:
            lines.extend(nested)
    if isinstance(before, list) and len(before) != len(after):
        lines.append(f"{path}: {len(before)} entries != {len(after)} entries")
    return lines


def _describe_difference(name, base, grown, facts):
    """Name the programs whose keys differ and the compile facts behind them."""
    base_keys, grown_keys = {row[2] for row in base}, {row[2] for row in grown}
    lines = []
    for dialect, program, key in base:
        if key in grown_keys:
            continue
        counterpart = next(
            (
                row
                for row in grown
                if row[0] == dialect and row[1] == program and row[2] not in base_keys
            ),
            None,
        )
        lines.append(
            f"  {dialect}:{program} {key[:12]} -> "
            + (f"{counterpart[2][:12]}" if counterpart else "absent")
        )
        if counterpart:
            for line in _fact_differences(
                facts.get(key, {}), facts.get(counterpart[2], {})
            ):
                lines.append(f"      {line}")
    for dialect, program, key in grown:
        if key not in base_keys and not any(
            row[0] == dialect and row[1] == program for row in base
        ):
            lines.append(f"  {dialect}:{program} absent -> {key[:12]}")
    return f"{name} configuration:\n" + "\n".join(lines)


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_compile_program_keys_ignore_pool_geometry(compiler_worker, case):
    """Two declarations that differ only in pool geometry plan identical programs."""
    result = compiler_worker.apply_async(_probe, (case,)).get(
        timeout=PROBE_TIMEOUT_SECONDS
    )
    if "unplannable" in result:
        pytest.skip(
            f"{case['family']} cannot be planned in an offline compiler worker: {result['unplannable']}"
        )
    programs, facts, queries = result["programs"], result["facts"], result["queries"]
    failures = []
    for label in PERTURBATIONS:
        assert queries["base"] != queries[label], "the perturbation changed nothing"
        for config_name in dict.fromkeys(name for _, name in programs):
            base, perturbed = (
                programs[("base", config_name)],
                programs[(label, config_name)],
            )
            assert base, f"{case['family']} planned no programs"
            if base != perturbed:
                failures.append(
                    f"{case['family']} programs depend on {queries['base']} -> {queries[label]}\n"
                    + _describe_difference(config_name, base, perturbed, facts)
                )
    if failures:
        pytest.fail("\n".join(failures))


@pytest.mark.parametrize("family", sorted(POOL_FREE_FAMILIES))
def test_families_without_pool_geometry_are_documented(family):
    pytest.skip(f"{family}: {POOL_FREE_FAMILIES[family]}")


def test_pool_fields_cover_every_field_outside_the_selection_key():
    """Every field a contract keeps out of its selection key is a perturbed pool field."""
    import importlib

    for name, family in FAMILIES.items():
        module = importlib.import_module(family.contract_module)
        query_fields = set(getattr(module, family.query_type).__dataclass_fields__)
        outside_key = query_fields - module.TUNING.query_fields - {"device"}
        assert outside_key <= set(family.pool_fields), (name, outside_key)
        assert set(family.pool_fields) <= query_fields, (name, family.pool_fields)


def _selection_key_parts(case):
    """The query and invocation as the selection cache keys them, per pool."""
    family = FAMILIES[case["family"]]
    pools = {
        "base": case["pool"],
        **{
            label: {name: perturb(value) for name, value in case["pool"].items()}
            for label, perturb in PERTURBATIONS.items()
        },
    }
    parts = {}
    for label, pool in pools.items():
        plan = family.declare(case["spec"], pool)
        contract = plan.contract
        encoded = contract.configure(
            plan.query, device=IDENTITY, override=None
        ).encoded_query
        parts[label] = {
            "fixed": all(
                knob.values is not None and len(knob.values) == 1
                for knob in contract.knobs
            ),
            "query": encoded.to_dict(),
            "invocation": plan.invocation.to_dict(),
        }
    return parts


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_selection_keys_ignore_pool_geometry(compiler_worker, case):
    """Two declarations that differ only in pool geometry share one selection key.

    The key digests the encoded query and the encoded invocation; a fixed
    contract has nothing to select, so its key may carry the pool.
    """
    if case["family"] == "attention.gqa":
        pytest.skip(
            "the paged configuration carries pool-derived capacities, so its key includes the pool"
        )
    parts = compiler_worker.apply_async(_selection_key_parts, (case,)).get(
        timeout=PROBE_TIMEOUT_SECONDS
    )
    if parts["base"]["fixed"]:
        pytest.skip(f"{case['family']} is a fixed contract and never races")
    for label in PERTURBATIONS:
        for part in ("query", "invocation"):
            assert parts[label][part] == parts["base"][part], (
                f"{case['family']} selection {part} depends on pool geometry ({label})"
            )
