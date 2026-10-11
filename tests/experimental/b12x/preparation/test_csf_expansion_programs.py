"""Compact scale programs retain both route ABIs through frozen preparation."""

from dataclasses import asdict

import pytest
import torch

from b12x._lib.compile_plan import (
    compiled_program_available,
    load_programs,
    program_keys,
)
from b12x._lib.compile_pool import CompileJob, compile_in_process, describe_compilation
from b12x._lib.quant.csf_blocks_tuning import (
    CsfExpansionConfig,
    CsfExpansionQuery,
    compile_expansion,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA compilation")
@pytest.mark.parametrize("bits,destination", [(4, 0), (4, 1), (1, 1), (1, 2), (1, 3)])
def test_scale_factory_declares_and_retains_all_route_abis(bits, destination):
    ordinal = torch.cuda.current_device()
    geometry = (256, 4, bits, 9, 256, 4, 8, destination, 1)
    query = asdict(CsfExpansionQuery(geometry, geometry, 5, "int32"))
    config = asdict(CsfExpansionConfig(groups_per_task=2))
    job = CompileJob.create(
        "b12x._lib.quant.csf_blocks_tuning:compile_expansion", query, config, ordinal
    )
    description = describe_compilation(job)
    assert len(description.programs) == 4
    compile_in_process((description,))
    assert all(compiled_program_available(key) for key in description.programs)
    programs = compile_expansion(query, config, ordinal)
    assert set(program_keys(programs)) == set(description.programs)
    load_programs(programs)
