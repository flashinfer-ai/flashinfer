# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-id namespace shared by every engine the frontend can dispatch to.

Execution engines live in ONE flat integer id space, split into segments by
provider. An engine's id is its stable IDENTITY and the key the plan walk looks
it up by; a plan's POSITION in the ranked list is its rank. The two are
independent — any engine may sit at any position in the list.

    [0,      10_000)   cuDNN backend engines. Ids assigned by the backend
                       (small: 0..~100 today), read back per graph with
                       get_engine_and_knobs_at_index().
    [10_000, 20_000)   C++-side OSS engines (plans.h OSS_*_ENGINE_CANDIDATE).
                       Reserved, not populated: those are retiring in favour of
                       the python engines.
    [20_000, ...)      Python engines, one sub-range per family.

Each engine owns its id the way a cuDNN engine does, so ids never shift with
discovery order and an autotune result (engine_id, knobs) replays with
create_execution_plan(engine_id, knobs).
"""

BACKEND_ENGINE_ID_BASE = 0
CPP_OSS_ENGINE_ID_BASE = 10_000
# Pre-release renumber: this space arrived with the Router MR and has never
# shipped, and no API could persist a python engine id before create_execution_plan()
# learned about them in this change — so there is nothing recorded to migrate.
PYTHON_ENGINE_ID_BASE = 20_000

# One block per FAMILY, FAMILY_BLOCK ids wide. A family is a kind of graph (see
# engines/manifest.py); every engine that serves that kind draws its id from
# that family's block, taking the next free slot. A family never opens a second
# block, so engine_id alone identifies the family and the two cannot disagree.
FAMILY_BLOCK = 100
GDN_ID_BASE = PYTHON_ENGINE_ID_BASE + 100  # 20_100..20_199
KDA_ID_BASE = PYTHON_ENGINE_ID_BASE + 200  # 20_200..20_299
GDN2_ID_BASE = PYTHON_ENGINE_ID_BASE + 300  # 20_300..20_399
FROST_GEMM_ID_BASE = PYTHON_ENGINE_ID_BASE + 400  # 20_400..20_499
FROST_SDPA_FWD_ID_BASE = PYTHON_ENGINE_ID_BASE + 500  # 20_500..20_599
FROST_SDPA_BWD_ID_BASE = PYTHON_ENGINE_ID_BASE + 600  # 20_600..20_699
GDP_ID_BASE = PYTHON_ENGINE_ID_BASE + 700  # 20_700..20_799
FROST_CONV_ID_BASE = PYTHON_ENGINE_ID_BASE + 800  # 20_800..20_899
MAMBA2_ID_BASE = PYTHON_ENGINE_ID_BASE + 900  # 20_900..20_999

# The delegating entry: the backend picks among candidates it holds but does not
# expose as plans (heur_mode.OPENSOURCE). It has no C++ plan index and no
# (engine_id, knobs) pair to replay — C++ builds and runs it itself.
BACKEND_HEURISTIC_ENGINE_ID = -1

# The placement marker a family hook may put in its proposal list: "the backend's
# own ranked block for this mode goes here". Expanded by
# ``engines/heuristics._assemble``; it never reaches ``graph.plans``.
BACKEND_BLOCK_ENGINE_ID = -2


def is_python_engine(engine_id: int) -> bool:
    """True iff ``engine_id`` names a python engine (vs a backend engine)."""
    return engine_id >= PYTHON_ENGINE_ID_BASE


def is_backend_engine(engine_id: int) -> bool:
    """True iff ``engine_id`` names a cuDNN backend engine."""
    return BACKEND_ENGINE_ID_BASE <= engine_id < CPP_OSS_ENGINE_ID_BASE
