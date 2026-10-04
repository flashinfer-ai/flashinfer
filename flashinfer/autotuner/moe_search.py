# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional factorized search using the ordinary autotuner's profiler."""

import hashlib
import json
import logging

import torch

logger = logging.getLogger(__name__)


def _digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def iter_moe_tactic_results(
    runner, inputs, valid_tactics, profile, *, normalize, process_group=None
):
    """Yield a factorized winner, or ordinary results on unsupported metadata.

    ``profile`` owns error accounting and the per-tactic timing reduction. It
    returns the same finite time or infinity on every rank. All participants
    must enable the same strategy and enter this function in the same order,
    as required for ordinary distributed tuning. Admission is agreed before
    any adaptive profiling; failed attempts are memoized across fallback.
    """
    proposal = dict(valid=None, space=None, capable=False, oom=False, aborted=False)
    space = search = None
    aborted = None
    try:
        factory = getattr(runner, "get_factorized_tactic_space", None)
        proposal["capable"] = callable(factory)
        identities = [normalize(tactic) for tactic in valid_tactics]
        proposal["valid"] = _digest(identities)
        if callable(factory):
            # These are the native (tile, configuration) IDs, not arbitrary
            # runner handles. Preserve v2's additional default candidate and
            # filtered/incomplete spaces by falling back to ordinary search.
            if identities and all(
                isinstance(key, tuple)
                and len(key) == 2
                and all(type(part) is int for part in key)
                and key[0] > 0
                and key[1] >= 0
                for key in identities
            ):
                from flashinfer.fused_moe.da_tuner import (
                    FactorizedSearch,
                    FactorizedTacticSpace,
                )

                space = factory(inputs)
                if type(space) is FactorizedTacticSpace:
                    declared = space.all_tactics()
                    keys = [normalize(tactic.tactic) for tactic in declared]
                    if (
                        len(keys) == len(identities) == len(set(keys))
                        and set(keys) == set(identities)
                        and all(
                            type(part) is int
                            for tactic in declared
                            for part in (tactic.tile_n, tactic.fc1, tactic.fc2)
                        )
                    ):
                        metadata = [
                            (normalize(t.tactic), t.tile_n, t.fc1, t.fc2)
                            for t in declared
                        ]
                        anchors = [
                            (tile, normalize(space.anchor(tile).tactic))
                            for tile in space.tiles
                        ]
                        proposal["space"] = _digest((metadata, anchors))
                        search = FactorizedSearch(max_sweeps=2)
    except (torch.cuda.OutOfMemoryError, MemoryError):
        proposal["oom"] = True
    except Exception:
        # An unavailable factorization is not permission to omit legal tactics.
        proposal["space"] = None
    except BaseException as exc:
        # Peers must leave admission too if this rank is interrupted.
        proposal["aborted"] = True
        aborted = exc

    proposals = [proposal]
    if process_group is not None:
        import torch.distributed as dist

        proposals = [None] * dist.get_world_size(process_group)
        dist.all_gather_object(proposals, proposal, group=process_group)

    if any(p["aborted"] for p in proposals):
        if aborted is not None:
            raise aborted
        raise RuntimeError("A peer aborted factorized MoE search admission")
    if any(p["oom"] for p in proposals):
        # choose_one already handles this exception as a default-tactic
        # fallback. Raise on every rank, before any profiling collective.
        raise MemoryError("OOM during factorized MoE search admission")
    valid_digests = {p["valid"] for p in proposals if p["valid"] is not None}
    if (
        process_group is not None
        and any(p["capable"] for p in proposals)
        and (len(valid_digests) > 1 or any(p["valid"] is None for p in proposals))
    ):
        raise RuntimeError("MoE tuning ranks have different ordered legal tactics")
    use_factorized = proposal["space"] is not None and all(
        p["space"] == proposal["space"] for p in proposals
    )
    if not use_factorized:
        logger.debug(
            "Using exhaustive search for %s: no matching complete MoE factorization",
            type(runner).__name__,
        )
        for tactic in valid_tactics:
            yield tactic, profile(tactic)
        return

    observed = {}
    profile_failed = False

    def measure(tactic, decisive=False):
        nonlocal profile_failed
        try:
            key = normalize(tactic.tactic)
            if key not in observed:
                observed[key] = profile(tactic.tactic)
            return observed[key]
        except RuntimeError:
            # Unexpected errors in the profiler's own failure handling must
            # propagate. Retrying an unrecorded call could add a collective
            # on this rank while peers reuse their recorded result.
            profile_failed = True
            raise

    try:
        selected = search.search(space, measure)
    except (torch.cuda.OutOfMemoryError, MemoryError):
        raise
    except RuntimeError:
        if profile_failed:
            raise
        # Nonfinite times disqualify a candidate on every rank. Reuse all
        # attempted measurements, including failures, in the ordinary pass.
        logger.debug(
            "Factorized MoE search rejected a measurement; using exhaustive fallback"
        )
        for tactic in valid_tactics:
            key = normalize(tactic)
            if key not in observed:
                observed[key] = profile(tactic)
            yield tactic, observed[key]
        return
    logger.debug("Factorized MoE search profiled %d unique tactics", len(observed))
    yield selected.tactic, observed[normalize(selected.tactic)]
