"""Independent complete post-preparation performance and numerical checks."""

from __future__ import annotations

import math
import traceback
from contextlib import ExitStack
from dataclasses import replace

GEOMETRIC_REGRET_LIMIT = 0.005
WORST_REGRET_LIMIT = 0.02


def _flatten(value):
    import torch

    if isinstance(value, torch.Tensor):
        return [value.detach().float().reshape(-1)]
    if isinstance(value, dict):
        return [tensor for name in sorted(value) for tensor in _flatten(value[name])]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _flatten(item)]
    raise TypeError(f"cosine result must contain tensors, got {type(value).__name__}")


def _cosine(actual, expected):
    import torch

    left, right = _flatten(actual), _flatten(expected)
    if (
        not left
        or len(left) != len(right)
        or any(a.shape != b.shape for a, b in zip(left, right, strict=True))
    ):
        raise ValueError("cosine expected output layout differs from production output")
    values = []
    for a, b in zip(left, right, strict=True):
        if not a.numel() or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("cosine requires nonempty finite outputs")
        values.append(float(torch.nn.functional.cosine_similarity(a, b, dim=0)))
    return min(values)


def aggregate_quality(reports, *, cached):
    """Aggregate per-query ratios, excluding fixed choices from timing races."""
    rows = [row for report in reports for row in report["requests"].values()]
    failures = [failure for report in reports for failure in report["failures"]]
    measured_rows = [
        row for row in rows if row.get("requires_race", row["effective_count"] > 1)
    ]
    ratios = [
        row["latency_ratio"]
        for row in measured_rows
        if row.get("latency_ratio") is not None
    ]
    geometric = (
        math.exp(math.fsum(math.log(ratio) for ratio in ratios) / len(ratios)) - 1
        if ratios
        else None
    )
    worst = max(ratios) - 1 if ratios else None
    if not rows:
        failures.append("numerical qualification contains no requests")
    if cached and any(row["measured_count"] for row in rows):
        failures.append("cached qualification performed candidate measurements")
    if not cached:
        if len(ratios) != len(measured_rows):
            failures.append("independent performance coverage is incomplete")
        if any(
            row["measured_count"] != row["effective_count"] for row in measured_rows
        ):
            failures.append(
                "independent race did not measure every effective candidate"
            )
        if any(
            row["measured_count"]
            for row in rows
            if not row.get("requires_race", row["effective_count"] > 1)
        ):
            failures.append("fixed declarations must not claim measured search winners")
        if geometric is not None and geometric > GEOMETRIC_REGRET_LIMIT:
            failures.append(f"geometric mean regret {geometric:.8%} exceeds 0.5%")
        if worst is not None and worst > WORST_REGRET_LIMIT:
            failures.append(f"worst regret {worst:.8%} exceeds 2%")
    effective = sum(row["effective_count"] for row in rows)
    checked = sum(row["cosine_checked_count"] for row in rows)
    passed = sum(row["cosine_passed_count"] for row in rows)
    expected_checks = len(rows) if cached else effective
    if checked != expected_checks or passed != expected_checks:
        failures.append(
            f"cosine coverage/passes {checked}/{passed} differs from required {expected_checks}"
        )
    return {
        "request_count": len(rows),
        "effective_count": effective,
        "measured_count": sum(row["measured_count"] for row in rows),
        "cosine_checked_count": checked,
        "cosine_passed_count": passed,
        "cosine_expected_count": expected_checks,
        "geometric_mean_regret": geometric,
        "worst_regret": worst,
        "geometric_mean_regret_limit": GEOMETRIC_REGRET_LIMIT,
        "worst_regret_limit": WORST_REGRET_LIMIT,
        "failures": failures,
        "passed": not failures,
    }


def flatten_requests(requests, owners, device):
    """Expand capacity declarations into the same leaf identities as the session.

    A composite plan is not itself expanded by the session's public surface;
    the session synthesizes one child request per token count internally,
    named ``f"{request.name}/m{count}"``. This mirrors that naming so the
    independent quality recheck below addresses the same leaf identities the
    real preparation run selected and measured.
    """
    del device
    from b12x.preparation.types import _CompositePlan

    leaves, leaf_owners, groups = {}, {}, {}

    def expand(request, owner):
        if request.name in leaves or request.name in groups:
            raise ValueError(f"duplicate quality request {request.name!r}")
        if not isinstance(request.plan, _CompositePlan):
            leaves[request.name] = request
            leaf_owners[request.name] = owner
            return (request.name,)
        plan = request.plan
        children = {
            count: child.request(
                name=f"{request.name}/m{count}",
                prepare_call=request.prepare_call[count],
                benchmark_call=None
                if request.benchmark_call is None
                else request.benchmark_call[count],
                dependencies=request.dependencies,
                collective=request.collective,
                retain_benchmark_call=request.retain_benchmark_call,
            )
            for count, child in plan.variants.items()
        }
        names = tuple(
            name
            for count in plan.token_counts
            for name in expand(children[count], owner)
        )
        groups[request.name] = names
        return names

    for request in requests:
        expand(request, owners[request.name])
    for name, request in tuple(leaves.items()):
        dependencies = tuple(
            dict.fromkeys(
                leaf
                for dependency in request.dependencies
                for leaf in groups.get(dependency, (dependency,))
            )
        )
        leaves[name] = replace(request, dependencies=dependencies)
    return leaves, leaf_owners


def _numerical_check(call, owner, *, device_ordinal, cosine, prohibit_compilation):
    import torch
    from b12x.preparation.types import _prime

    with prohibit_compilation():
        _prime(call)
        torch.cuda.synchronize(device_ordinal)
    expected = owner.test_expected(call)
    actual = owner.test_actual(call) if hasattr(owner, "test_actual") else call.output
    score = _cosine(actual, expected)
    threshold = (
        max(cosine, 0.999999)
        if isinstance(actual, torch.Tensor) and actual.dtype == torch.bool
        else cosine
    )
    if score < threshold:
        raise AssertionError(f"cosine={score:.8f} < {threshold}")
    return score


def _new_row(result, name):
    return {
        "cartesian_count": 0,
        "legal_count": 0,
        "equivalent_count": 0,
        "effective_count": 0,
        "prepared_count": 0,
        "measured_count": 0,
        "cosine_checked_count": 0,
        "cosine_passed_count": 0,
        "minimum_cosine": None,
        "latency_ratio": None,
        "selected_choice": None,
        "best_choice": None,
        "requires_race": False,
        "startup_coverage": dict(result.coverage.get(name, {})),
    }


def _pinned_request(request, config, *, name, retain):
    factory = request.benchmark_call or request.prepare_call
    if not callable(factory):
        raise TypeError("quality checks require a scalar benchmark call factory")
    return replace(
        request,
        name=name,
        plan=replace(request.plan, override=config),
        prepare_call=factory,
        benchmark_call=factory if retain else None,
        retain_benchmark_call=retain,
    )


def _ancestor_requests(request, requests, result):
    ordered, visiting, visited = [], set(), set()

    def visit(name):
        if name in visiting:
            raise ValueError("quality dependency cycle")
        if name in visited:
            return
        dependency = requests[name]
        visiting.add(name)
        for parent in dependency.dependencies:
            visit(parent)
        visiting.remove(name)
        visited.add(name)
        ordered.append(
            _pinned_request(
                dependency,
                result.selections[name].config,
                name=name,
                retain=False,
            )
        )

    for name in request.dependencies:
        visit(name)
    return ordered


def check_quality(
    requests,
    owners,
    result,
    *,
    device,
    device_ordinal,
    cosine,
    cached,
    prohibit_compilation,
    samples=8,
    rounds=9,
    prepare_collective_batch=None,
):
    """Recheck real selected calls, with cache-only pinned sessions for competitors.

    Candidate construction uses the same public preparation engine. Its unique
    resource keys isolate trial owners from the retained startup winner. Fixed
    choices run numerical qualification without a fictitious timing contest.
    """
    import torch
    from b12x.preparation import DetectedDevice, PreparationSession
    from b12x.testing.benchmark import measure_calls
    from b12x.preparation.types import _prime

    detected = (
        device
        if isinstance(device, DetectedDevice)
        else DetectedDevice(device_ordinal, device)
    )
    requests, owners = flatten_requests(requests, owners, detected)
    report = {"requests": {}, "failures": [], "performance_recheck": not cached}

    def fail(name, stage, error):
        report["failures"].append(
            {"request": name, "stage": stage, "error": str(error)}
        )

    for name, request in requests.items():
        row = _new_row(result, name)
        report["requests"][name] = row
        selected = result.benchmark_calls.get(name)
        race = None
        resources = ExitStack()
        try:
            if selected is None:
                raise ValueError("selected benchmark call was not explicitly retained")
            selection = result.selections[name]
            contract = request.plan.contract
            selected_payload = contract.config_payload(selection.config)
            row["selected_choice"] = selected_payload.to_dict()
            row["selected_source"] = selection.source
            if cached:
                row["effective_count"] = max(
                    1, row["startup_coverage"].get("effective_count", 0)
                )
                if result.benchmarked_candidates:
                    raise AssertionError(
                        "cached startup performed candidate measurements"
                    )
                row["cosine_checked_count"] = 1
                row["minimum_cosine"] = _numerical_check(
                    selected,
                    owners[name],
                    device_ordinal=device_ordinal,
                    cosine=cosine,
                    prohibit_compilation=prohibit_compilation,
                )
                row["cosine_passed_count"] = 1
                continue

            configured = contract.configure(
                request.plan.query,
                device=detected.identity,
                override=request.plan.override,
            )
            if configured.pinned is not None:
                candidates = ((None, configured.pinned),)
                cartesian = legal = 1
            else:
                iterator = contract.iterate(configured)
                candidates = tuple(iterator)
                cartesian, legal = iterator.cartesian_count, iterator.legal_count
            if not candidates:
                raise ValueError("quality query has no legal candidates")
            row.update(
                cartesian_count=cartesian,
                legal_count=legal,
                equivalent_count=legal - len(candidates),
                effective_count=len(candidates),
                requires_race=len(candidates) > 1,
            )
            matching = [
                index
                for index, (_, config) in enumerate(candidates)
                if contract.config_payload(config) == selected_payload
            ]
            if len(matching) != 1:
                raise ValueError(
                    "selected config is absent or ambiguous in the complete eligible set"
                )
            selected_index = matching[0]
            calls = [selected]
            if row["requires_race"]:
                if selection.source not in ("tuned", "cached"):
                    raise ValueError(
                        "default/stopped preparation is not exhaustive selection acceptance"
                    )
                coverage = row["startup_coverage"]
                for field in ("cartesian_count", "legal_count", "effective_count"):
                    if coverage.get(field) != row[field]:
                        fail(
                            name,
                            "startup_coverage",
                            f"{field}: startup={coverage.get(field)} independent={row[field]}",
                        )
                if selection.source == "tuned" and coverage.get(
                    "measured_count"
                ) != len(candidates):
                    fail(
                        name,
                        "startup_coverage",
                        "startup did not measure the complete candidate set",
                    )
                if request.collective is not None:
                    raise ValueError(
                        "collective quality declarations must be fixed or pinned"
                    )
                candidate_names = [
                    f"quality.{name}.candidate{index}"
                    for index in range(len(candidates))
                ]
                batch = _ancestor_requests(request, requests, result)
                batch.extend(
                    _pinned_request(
                        request, config, name=candidate_names[index], retain=True
                    )
                    for index, (_, config) in enumerate(candidates)
                )
                session = resources.enter_context(
                    PreparationSession(
                        device=detected,
                        autotune=False,
                        cache_only=True,
                        compile_workers=1,
                    )
                )
                with prohibit_compilation():
                    if any(item.collective is not None for item in batch):
                        if prepare_collective_batch is None:
                            raise ValueError(
                                "collective dependencies require a complete-world driver"
                            )
                        prepared = prepare_collective_batch(session, batch)
                    else:
                        prepared = session.prepare(batch)
                    independent = resources.enter_context(prepared)
                calls = [
                    independent.benchmark_calls[candidate_name]
                    for candidate_name in candidate_names
                ]
                row["prepared_count"] = len(calls)
                with prohibit_compilation():
                    measured = measure_calls(
                        dict(zip(candidate_names, calls, strict=True)),
                        device=device_ordinal,
                        warmup=1,
                        samples=samples,
                        rounds=rounds,
                        collective=any(item.collective is not None for item in batch),
                    )
                latencies = tuple(measured.latencies_us.values())
                row["timing_method"] = measured.method
                row["raw_samples_us"] = {
                    name: measured.raw_samples(name) for name in candidate_names
                }
                if len(latencies) != len(candidates) or any(
                    not math.isfinite(value) or value <= 0 for value in latencies
                ):
                    raise ValueError(
                        "independent race lacks complete uncontaminated positive timings"
                    )
                best = min(range(len(latencies)), key=latencies.__getitem__)
                row.update(
                    measured_count=len(latencies),
                    selected_latency_us=latencies[selected_index],
                    best_latency_us=latencies[best],
                    latency_ratio=latencies[selected_index] / latencies[best],
                    regret=latencies[selected_index] / latencies[best] - 1,
                    best_choice=contract.config_payload(candidates[best][1]).to_dict(),
                )
            else:
                row["prepared_count"] = 1
            for call in calls:
                row["cosine_checked_count"] += 1
                score = _numerical_check(
                    call,
                    owners[name],
                    device_ordinal=device_ordinal,
                    cosine=cosine,
                    prohibit_compilation=prohibit_compilation,
                )
                row["minimum_cosine"] = (
                    score
                    if row["minimum_cosine"] is None
                    else min(row["minimum_cosine"], score)
                )
                row["cosine_passed_count"] += 1
        except Exception as error:
            fail(name, "quality", "".join(traceback.format_exception(error)))
        finally:
            if race is not None:
                try:
                    race.close()
                except Exception as error:
                    fail(name, "race_close", error)
            try:
                resources.close()
            except Exception as error:
                fail(name, "candidate_close", error)
            # Dependency-producing fixtures restore the exact retained winner,
            # not a newly materialized object with a different resource owner.
            if selected is not None:
                try:
                    with prohibit_compilation():
                        _prime(selected)
                        torch.cuda.synchronize(device_ordinal)
                except Exception as error:
                    fail(name, "restore_selected", error)
    report["aggregate"] = aggregate_quality([report], cached=cached)
    return report
