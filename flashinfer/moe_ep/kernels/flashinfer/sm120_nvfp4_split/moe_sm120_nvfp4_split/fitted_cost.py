"""ABI27 native service model shared by offline fitting and runtime selection.

Logical requests, estimates and measured hardware counters remain distinct.
Parameters use physical work and candidate resources; no shape-name switches.
"""

_qualified_dual_PARAMETERS = {
    "k1_mem": (8400.0, 4000.0, 16000.0),
    "k2_mem": (14500.0, 7000.0, 30000.0),
    "math": (6500000.0, 2000000.0, 20000000.0),
    "k2_math": (6500000.0, 2000000.0, 20000000.0),
    "k2_tile_curve": (1.0, 0.5, 2.0),
    "reuse": (0.45, 0.05, 1.0),
    "stage1": (0.25, 0.0, 0.6),
    "stage2": (0.035, 0.0, 0.3),
    "queue": (0.025, 0.001, 0.5),
    "dispatch_bw": (25000.0, 10000.0, 100000.0),
    "cache_fixed": (8.0, 0.0, 40.0),
    "overlap": (0.35, 0.05, 1.0),
    "dispatch_control": (0.005, 0.0, 0.03),
    "token_back_bw": (35000.0, 10000.0, 100000.0),
    "tail_grain": (0.5, 0.0, 2.0),
    "rlc_fixed": (8.0, 0.0, 100.0),
    "critical": (0.35, 0.05, 1.0),
    "shared_bw": (1100000.0, 900000.0, 1344000.0),
    "k1_mixed": (0.3, 0.0, 2.0),
    "k2_mixed": (0.3, 0.0, 2.0),
    "dual_mem_gain": (1.5, 1.0, 4.0),
    "dual_math_eff": (6 / 6.5, 0.4, 3.0),
    "rlc_tile": (0.025, 0.0, 0.3),
    "fc1_epilogue_row": (0.01, 0.0, 0.1),
    "fc2_epilogue_row": (0.01, 0.0, 0.1),
    "fc2_task": (0.1, 0.0, 2.0),
    "stage_n_knee": (32.0, 16.0, 128.0),
    "stage_k_knee": (32.0, 8.0, 64.0),
    "consumer_sync_step": (0.02, 0.0, 0.5),
}


def _qualified_dual__evaluate_v13(shape, c, f, p):
    n1, n2 = (c.k1_tile[1], c.k2_tile[1])
    unique1 = f.unique_weight_bytes * 2 / 3
    unique2 = f.unique_weight_bytes / 3
    requests1 = min(f.fc1_weight_request_bytes, unique1) + p["reuse"] * max(
        0.0, f.fc1_weight_request_bytes - unique1
    )
    requests2 = min(f.fc2_weight_request_bytes, unique2) + p["reuse"] * max(
        0.0, f.fc2_weight_request_bytes - unique2
    )
    k1_mem = requests1 / (c.k1_sms * p["k1_mem"])
    k2_mem = requests2 / (
        c.k2_sms * p["k2_mem"] * (p["dual_mem_gain"] if c.k2_warps == 12 else 1.0)
    )
    k1_math = f.padded_fc1_flops / (c.k1_sms * p["math"] * (0.35 + 0.65 * n1 / 128.0))
    k2_eff = (
        p["dual_math_eff"]
        if c.k2_warps == 12
        else 0.35 + 0.65 * (n2 / 128.0) ** p["k2_tile_curve"]
    )
    k2_math = f.padded_fc2_flops / (c.k2_sms * p["k2_math"] * k2_eff)
    stage_gain = 1.0 - p["stage2"] * (c.k2_stages - 2) * min(
        f.descriptor_k_steps / 16.0, 1.0
    )
    k1_steps = (shape.hidden + 127) // 128
    depth_gain = (
        p["stage1"]
        * ((c.k1_stages or 2) - 2)
        * n1
        / (n1 + p["stage_n_knee"])
        * k1_steps
        / (k1_steps + p["stage_k_knee"])
    )
    producer = (max(k1_mem, k1_math) + f.k1_waves * (0.3 + n1 * 0.012)) * (
        1.0 - depth_gain
    )
    producer += f.k1_waves * k1_steps * p["consumer_sync_step"]
    consumer = (max(k2_mem, k2_math) + f.k2_waves * (0.22 + n2 * 0.002)) * stage_gain
    queue = f.descriptors * p["queue"] / max(c.k2_sms, 1)
    dispatch = (
        f.dispatch_bytes / (p["dispatch_bw"] * c.dispatch_warps)
        + shape.tokens_per_rank
        * shape.num_topk
        * p["dispatch_control"]
        * c.dispatch_warps
    )
    if c.dispatch_rank_cache:
        dispatch += p["cache_fixed"] + shape.tokens_per_rank * 0.0008
    if c.dispatch_compute_overlap:
        dispatch = max(4.0, dispatch * p["overlap"]) + 5.0
    consumer += f.token_back_bytes / p["token_back_bw"]
    grain = min(c.ready_queue_bundle, (shape.hidden + 63) // 64) * f.descriptor_k_steps
    tail = consumer * f.grain_relative_to_worker_work * p["tail_grain"] + consumer * (
        0.015 if c.k2_tail_reclaim else 0.03
    )
    if c.k2_tail_reclaim:
        tail += 7.0
    if c.k2_warps == 12:
        tail += 5.0
    k3 = 8.0 + shape.tokens_per_rank * shape.hidden * 2 / 150000.0
    if c.rank_local_combine:
        k3 += p["rlc_fixed"] + shape.tokens_per_rank * (
            0.005 if f.reducer == "owner_vector" else 0.025
        )
        k3 += p["rlc_tile"] * f.fc2_tiles * ((shape.hidden + 63) // 64)
    stage_scale = 1.0
    roof = (requests1 + requests2) / (p["shared_bw"] * stage_scale)
    physical_floor = (requests1 + requests2) / 1344000.0
    if c.k2_tail_reclaim:
        completion = (
            producer + max(0.0, consumer - producer) * c.k2_sms / shape.num_sms * 1.5
        )
        critical = roof + p["critical"] * max(0.0, completion - roof)
    else:
        critical = max(producer, consumer, roof)
    exposed1 = (
        p["k1_mixed"]
        * k1_mem
        * k1_math
        / max(k1_mem + k1_math, 1e-30)
        * (1.0 - depth_gain)
    )
    exposed2 = (
        p["k2_mixed"] * k2_mem * k2_math / max(k2_mem + k2_math, 1e-30) * stage_gain
    )
    epilogue1 = p["fc1_epilogue_row"] * f.k1_waves * n1
    epilogue2 = f.k2_waves * (p["fc2_task"] + p["fc2_epilogue_row"] * n2)
    critical = max(
        physical_floor, critical + exposed1 + exposed2 + epilogue1 + epilogue2
    )
    return (
        critical + dispatch + tail + k3 + queue + 15.0,
        producer,
        consumer,
        dispatch,
        tail,
        k3,
        queue,
    )


def _qualified_dual_describe_rlc_path(shape, c, f):
    """Logical requests and actual default reducer branch, never DRAM counters."""
    if not c.rank_local_combine:
        return dict(
            applicable=False,
            local_route_write_bytes=0.0,
            local_route_read_bytes=0.0,
            join_cta_waves=0.0,
            owner_partial_rows=0.0,
            reducer="ordinary_topk",
        )
    groups = (
        4
        if shape.hidden >= 2048
        and shape.hidden % (4 * max(128, c.k2_tile[0])) == 0
        and (c.k2_tile[1] >= 4)
        else 1
    )
    chunk = shape.hidden // groups
    vector = 1 <= shape.num_topk <= 8 and shape.hidden % 2048 == 0
    reducer = (
        "owner_warp32"
        if vector and chunk % 1024 == 0
        else "owner_vector16" if vector else "owner_scalar16"
    )
    route_bytes = float(shape.tokens_per_rank * shape.num_topk * shape.hidden * 2)
    return dict(
        applicable=True,
        local_route_write_bytes=route_bytes,
        local_route_read_bytes=route_bytes,
        join_cta_waves=f.k2_waves,
        owner_partial_rows=shape.tokens_per_rank
        * shape.expert_parallel_size
        * f.owner_probability,
        publication_groups=groups,
        reducer=reducer,
        default_packed_store=False,
        interpretation="logical staging requests; exposed path coefficient also covers the default unpacked store and reduction execution; not separately identified hardware bandwidth",
    )


def _qualified_dual_evaluate(shape, c, f, p):
    if not c.rank_local_combine:
        return _qualified_dual__evaluate_v13(shape, c, f, p)
    path = _qualified_dual_describe_rlc_path(shape, c, f)
    route_work = (
        f.token_back_bytes
        + path["local_route_write_bytes"]
        + path["local_route_read_bytes"]
    )
    effective_return_bw = (
        p["rlc_path_bytes_per_us"] * f.token_back_bytes / max(route_work, 1.0)
    )
    q = dict(
        p, rlc_fixed=0.0, rlc_tile=0.0, token_back_bw=max(effective_return_bw, 1.0)
    )
    base = list(_qualified_dual__evaluate_v13(shape, c, f, q))
    narrow = max(128.0 / c.k2_tile[1] - 1.0, 0.0)
    exposed = p["rlc_launch_us"] + p["rlc_narrow_wave_us"] * f.k2_waves * narrow
    base[0] += exposed
    base[2] += exposed
    return tuple(base)


_qualified_dual_COEFFICIENTS = {
    "k1_mem": 9451.371025090719,
    "k2_mem": 13240.374524469747,
    "math": 7682787.930879342,
    "k2_math": 6379796.865950565,
    "k2_tile_curve": 0.9999999999646394,
    "reuse": 0.7208063062676725,
    "stage1": 0.5135729602215287,
    "stage2": 0.22854572840671886,
    "queue": 0.0010186740994451936,
    "dispatch_bw": 99999.99999999999,
    "cache_fixed": 1.9796060610523915e-11,
    "overlap": 0.6571048635208797,
    "dispatch_control": 0.011592740950369957,
    "token_back_bw": 67913.65016351754,
    "tail_grain": 0.3578166202972688,
    "rlc_fixed": 3.482155654190642,
    "critical": 0.7688243533305082,
    "shared_bw": 1222422.8665584486,
    "k1_mixed": 0.49510982682047017,
    "k2_mixed": 2.549475727637911e-19,
    "dual_mem_gain": 1.2003691689501919,
    "dual_math_eff": 0.9230769231454712,
    "rlc_tile": 1.9000085438635815e-14,
    "fc1_epilogue_row": 3.3207151643418236e-17,
    "fc2_epilogue_row": 0.00512413820216883,
    "fc2_task": 5.4033116299426714e-18,
    "stage_n_knee": 16.000000000000004,
    "stage_k_knee": 8.000000000000002,
    "consumer_sync_step": 3.34346440342216e-11,
    "rlc_launch_us": 42.62977255922283,
    "rlc_narrow_wave_us": 0.43895750598061734,
    "rlc_path_bytes_per_us": 352629.8884072991,
}

"""ABI27 native service model shared by offline fitting and runtime selection.

Logical requests, estimates and measured hardware counters remain distinct.
Parameters use physical work and candidate resources; no shape-name switches.
"""
PARAMETERS = {
    "k1_mem": (8400.0, 4000.0, 16000.0),
    "k2_mem": (14500.0, 7000.0, 30000.0),
    "math": (6.5e6, 2e6, 20e6),
    "k2_math": (6.5e6, 2e6, 20e6),
    "k2_tile_curve": (1.0, 0.5, 2.0),
    "reuse": (0.45, 0.05, 1.0),
    "stage1": (0.25, 0.0, 0.6),
    "stage2": (0.035, 0.0, 0.30),
    "queue": (0.025, 0.001, 0.50),
    "dispatch_bw": (25000.0, 10000.0, 100000.0),
    "cache_fixed": (8.0, 0.0, 40.0),
    "overlap": (0.35, 0.05, 1.0),
    "dispatch_control": (0.005, 0.0, 0.03),
    "token_back_bw": (35000.0, 10000.0, 100000.0),
    "tail_grain": (0.5, 0.0, 2.0),
    "rlc_fixed": (8.0, 0.0, 100.0),
    "critical": (0.35, 0.05, 1.0),
    "shared_bw": (1100000.0, 900000.0, 1344000.0),
    "k1_mixed": (0.3, 0.0, 2.0),
    "k2_mixed": (0.3, 0.0, 2.0),
    "dual_mem_gain": (1.5, 1.0, 4.0),
    "dual_math_eff": (6 / 6.5, 0.4, 3.0),
    "rlc_tile": (0.025, 0.0, 0.3),
    "fc1_epilogue_row": (0.01, 0.0, 0.1),
    "fc2_epilogue_row": (0.01, 0.0, 0.1),
    "fc2_task": (0.1, 0.0, 2.0),
    "stage_n_knee": (32.0, 16.0, 128.0),
    "stage_k_knee": (32.0, 8.0, 64.0),
    "consumer_sync_step": (0.02, 0.0, 0.5),
}


def _evaluate_v13(shape, c, f, p):
    n1, n2 = c.k1_tile[1], c.k2_tile[1]
    unique1 = f.unique_weight_bytes * 2 / 3
    unique2 = f.unique_weight_bytes / 3
    requests1 = min(f.fc1_weight_request_bytes, unique1) + p["reuse"] * max(
        0.0, f.fc1_weight_request_bytes - unique1
    )
    requests2 = min(f.fc2_weight_request_bytes, unique2) + p["reuse"] * max(
        0.0, f.fc2_weight_request_bytes - unique2
    )
    k1_mem = requests1 / (c.k1_sms * p["k1_mem"])
    k2_mem = requests2 / (
        c.k2_sms * p["k2_mem"] * (p["dual_mem_gain"] if c.k2_warps == 12 else 1.0)
    )
    k1_math = f.padded_fc1_flops / (c.k1_sms * p["math"] * (0.35 + 0.65 * n1 / 128.0))
    # FC1 gated epilogue and FC2 token-back have different execution throughput.
    k2_eff = (
        p["dual_math_eff"]
        if c.k2_warps == 12
        else 0.35 + 0.65 * (n2 / 128.0) ** p["k2_tile_curve"]
    )
    k2_math = f.padded_fc2_flops / (c.k2_sms * p["k2_math"] * k2_eff)
    # First-read service can overlap deeper loads. Repeated token tiles share
    # the weight set; the extra buffering cost still applies to every tile.
    # L2 is fixed by the qualified hardware, not inferred from a model name.
    l2 = 100663296.0
    pressure = max(0.0, (unique2 - l2) / (unique2 + l2))
    first_read_fraction = min(1.0, unique2 / max(requests2, 1.0))
    stage_gain = 1.0 - p["stage2"] * (c.k2_stages - 2) * min(
        f.descriptor_k_steps / 16.0, 1.0
    ) * first_read_fraction * (1.0 + p["cache_stage_pressure"] * pressure)
    stage_gain = max(0.25, stage_gain)
    k1_steps = (shape.hidden + 127) // 128
    depth_gain = (
        p["stage1"]
        * ((c.k1_stages or 2) - 2)
        * n1
        / (n1 + p["stage_n_knee"])
        * k1_steps
        / (k1_steps + p["stage_k_knee"])
    )
    producer = (max(k1_mem, k1_math) + f.k1_waves * (0.30 + n1 * 0.012)) * (
        1.0 - depth_gain
    )
    producer_depth_exposed = (
        f.k1_waves
        * (n1 / 128.0)
        * max((c.k1_stages or 2) - 2, 0)
        * p["stage1_wave_overhead"]
    )
    # Each FC1 work wave consumes H/K shared stages. The repaired implementation
    # orders shared reads and rendezvous before stage reuse; deeper buffers do not
    # remove those operations. This term uses candidate work, not model labels.
    producer += f.k1_waves * k1_steps * p["consumer_sync_step"]
    consumer = (max(k2_mem, k2_math) + f.k2_waves * (0.22 + n2 * 0.002)) * stage_gain
    # Stage service multipliers model overlap/stalls, not a reduction of FLOPs.
    # ABI27 native accumulation has no software compensation work.
    consumer += (
        f.k2_waves * (n2 / 128.0) * max(c.k2_stages - 2, 0) * p["stage2_wave_overhead"]
    )
    queue = f.descriptors * p["queue"] / max(c.k2_sms, 1)
    dispatch = (
        f.dispatch_bytes / (p["dispatch_bw"] * c.dispatch_warps)
        + shape.tokens_per_rank
        * shape.num_topk
        * p["dispatch_control"]
        * c.dispatch_warps
    )
    dispatch += f.k1_waves * max(c.dispatch_warps - 1, 0) * p["dispatch_warp_wave"]
    if c.dispatch_rank_cache:
        dispatch += p["cache_fixed"] + shape.tokens_per_rank * 0.0008
    if c.dispatch_compute_overlap:
        dispatch = max(4.0, dispatch * p["overlap"]) + 5.0
    consumer += f.token_back_bytes / p["token_back_bw"]
    grain = min(c.ready_queue_bundle, (shape.hidden + 63) // 64) * f.descriptor_k_steps
    # A descriptor's share of a worker's work estimates scheduling imbalance.
    # Unlike a constant per Q, this shrinks as more independent tiles fill waves.
    tail = consumer * f.grain_relative_to_worker_work * p["tail_grain"] + consumer * (
        0.015 if c.k2_tail_reclaim else 0.03
    )
    if c.k2_tail_reclaim:
        tail += 7.0
    if c.k2_warps == 12:
        tail += 5.0
    k3 = 8.0 + shape.tokens_per_rank * shape.hidden * 2 / 150000.0
    if c.rank_local_combine:
        k3 += p["rlc_fixed"] + shape.tokens_per_rank * (
            0.005 if f.reducer == "owner_vector" else 0.025
        )
        # Group construction/metadata occurs at each FC2 task, before completion.
        # It is a tile cost even when little owner traffic remains to be returned.
        k3 += p["rlc_tile"] * f.fc2_tiles * ((shape.hidden + 63) // 64)
    # shared_bw is calibrated achieved bandwidth, not a physical lower bound.
    # Pipeline depth affects the achieved rate as well as each producer estimate.
    stage_scale = 1.0
    roof = (requests1 + requests2) / (p["shared_bw"] * stage_scale)
    physical_floor = (requests1 + requests2) / 1344000.0
    if c.k2_tail_reclaim:
        completion = (
            producer + max(0.0, consumer - producer) * c.k2_sms / shape.num_sms * 1.5
        )
        critical = roof + p["critical"] * max(0.0, completion - roof)
    else:
        # Without tail reclamation, neither kernel's completion can be discounted
        # by averaging it with a bandwidth estimate.
        critical = max(producer, consumer, roof)
    # Mixed memory/compute execution is not a perfectly overlapped roofline.
    # The harmonic term is monotone in both costs, vanishes when either is zero,
    # and represents calibrated exposed work. It uses candidate work/resources,
    # not a model identifier or a token threshold.
    exposed1 = (
        p["k1_mixed"]
        * k1_mem
        * k1_math
        / max(k1_mem + k1_math, 1e-30)
        * (1.0 - depth_gain)
    )
    exposed2 = (
        p["k2_mixed"] * k2_mem * k2_math / max(k2_mem + k2_math, 1e-30) * stage_gain
    )
    # Per-task epilogues scale with output rows and actual worker waves,
    # not the GEMM reduction dimension. They cannot be hidden just by a DRAM max.
    epilogue1 = p["fc1_epilogue_row"] * f.k1_waves * n1
    epilogue2 = f.k2_waves * (p["fc2_task"] + p["fc2_epilogue_row"] * n2)
    # Pipeline depth improves producer service only. Ordinary token-back can
    # dominate consumer completion; a K1 stage change must not discount it.
    # Deeper producer buffering may expose shared-resource/control cost even
    # when the consumer is the long-running role. Do not erase this term in max.
    critical = max(
        physical_floor,
        critical + exposed1 + exposed2 + epilogue1 + epilogue2 + producer_depth_exposed,
    )

    return (
        critical + dispatch + tail + k3 + queue + 15.0,
        producer,
        consumer,
        dispatch,
        tail,
        k3,
        queue,
    )


def describe_rlc_path(shape, c, f):
    """Logical requests and actual default reducer branch, never DRAM counters."""
    if not c.rank_local_combine:
        return dict(
            applicable=False,
            local_route_write_bytes=0.0,
            local_route_read_bytes=0.0,
            join_cta_waves=0.0,
            owner_partial_rows=0.0,
            reducer="ordinary_topk",
        )
    groups = (
        4
        if shape.hidden >= 2048
        and shape.hidden % (4 * max(128, c.k2_tile[0])) == 0
        and c.k2_tile[1] >= 4
        else 1
    )
    chunk = shape.hidden // groups
    vector = 1 <= shape.num_topk <= 8 and shape.hidden % 2048 == 0
    reducer = (
        "owner_warp32"
        if vector and chunk % 1024 == 0
        else "owner_vector16" if vector else "owner_scalar16"
    )
    route_bytes = float(shape.tokens_per_rank * shape.num_topk * shape.hidden * 2)
    return dict(
        applicable=True,
        local_route_write_bytes=route_bytes,
        local_route_read_bytes=route_bytes,
        join_cta_waves=f.k2_waves,
        owner_partial_rows=shape.tokens_per_rank
        * shape.expert_parallel_size
        * f.owner_probability,
        publication_groups=groups,
        reducer=reducer,
        default_packed_store=False,
        interpretation="logical staging requests; exposed path coefficient also covers the default unpacked store and reduction execution; not separately identified hardware bandwidth",
    )


def evaluate(shape, c, f, p):
    # Generic and dual-group kernels have different physical implementations.
    # Fit their service coefficients separately using native ABI27 measurements.
    # This branch depends on kernel mode, not a model name or token threshold.
    if c.k2_warps == 12:
        return _qualified_dual_evaluate(shape, c, f, _qualified_dual_COEFFICIENTS)
    if not c.rank_local_combine:
        return _evaluate_v13(shape, c, f, p)
    path = describe_rlc_path(shape, c, f)
    route_work = (
        f.token_back_bytes
        + path["local_route_write_bytes"]
        + path["local_route_read_bytes"]
    )
    # This is a calibrated exposed service rate for the whole RLC data path,
    # not a claim of measured DRAM or PCIe bandwidth. Preserve the existing
    # producer/consumer overlap model while accounting for local staging work.
    effective_return_bw = (
        p["rlc_path_bytes_per_us"] * f.token_back_bytes / max(route_work, 1.0)
    )
    q = dict(
        p, rlc_fixed=0.0, rlc_tile=0.0, token_back_bw=max(effective_return_bw, 1.0)
    )
    base = list(_evaluate_v13(shape, c, f, q))
    # Candidate geometry term: calibrate the narrower FC2 output layout's
    # exposed cost. It is not an exact count of executed barriers/instructions.
    narrow = max(128.0 / c.k2_tile[1] - 1.0, 0.0)
    exposed = p["rlc_launch_us"] + p["rlc_narrow_wave_us"] * f.k2_waves * narrow
    base[0] += exposed
    base[2] += exposed
    return tuple(base)


COEFFICIENTS = {
    "k1_mem": 11289.074110747335,
    "k2_mem": 16487.95468635183,
    "math": 6484776.496710679,
    "k2_math": 8115864.899326129,
    "k2_tile_curve": 0.9999999999646394,
    "reuse": 0.36308710848564985,
    "stage1": 0.47118524458229916,
    "stage2": 0.01460472233169305,
    "queue": 0.009324241119978184,
    "dispatch_bw": 93356.19052633381,
    "cache_fixed": 1.9796060610523915e-11,
    "overlap": 0.6571048635208797,
    "dispatch_control": 0.0043425911789876535,
    "token_back_bw": 67913.65016351754,
    "tail_grain": 1.4041937582770558,
    "rlc_fixed": 3.482155654190642,
    "critical": 0.7688243533305082,
    "shared_bw": 1126462.1917638215,
    "k1_mixed": 0.6411938142892919,
    "k2_mixed": 0.13444397484752488,
    "dual_mem_gain": 1.2003691689501919,
    "dual_math_eff": 0.9230769231454712,
    "rlc_tile": 1.9000085438635815e-14,
    "fc1_epilogue_row": 3.3207151643418236e-17,
    "fc2_epilogue_row": 0.00512413820216883,
    "fc2_task": 5.4033116299426714e-18,
    "stage_n_knee": 16.000000000000004,
    "stage_k_knee": 8.000000000000002,
    "consumer_sync_step": 3.34346440342216e-11,
    "rlc_launch_us": 34.42738337081887,
    "rlc_narrow_wave_us": 0.43895750598061734,
    "rlc_path_bytes_per_us": 673971.2293134194,
    "cache_stage_pressure": 1.9957163741835786,
    "dispatch_warp_wave": 0.07885948435680203,
    "compensation_wave_step": 0.0,
    "stage1_wave_overhead": 0.5791403514333352,
    "stage2_wave_overhead": 0.1970405612394204,
}


def describe_service(shape, c, f):
    if c.k2_warps == 12:
        return dict(
            applicable=True,
            implementation="qualified_dual_group",
            generic_compensation_work_units=0.0,
            interpretation="Dual-group native implementation uses separately fitted ABI27 service coefficients.",
        )
    units = 0.0
    return dict(
        applicable=True,
        generic_compensation_work_units=units,
        compensation_service_estimate_us=units * COEFFICIENTS["compensation_wave_step"],
        k1_depth_service_estimate_us=f.k1_waves
        * (c.k1_tile[1] / 128.0)
        * max((c.k1_stages or 2) - 2, 0)
        * COEFFICIENTS["stage1_wave_overhead"],
        extra_depth_worker_waves=f.k2_waves * max(c.k2_stages - 2, 0),
        depth_service_estimate_us=f.k2_waves
        * (c.k2_tile[1] / 128.0)
        * max(c.k2_stages - 2, 0)
        * COEFFICIENTS["stage2_wave_overhead"],
        dispatch_extra_warp_waves=f.k1_waves * max(c.dispatch_warps - 1, 0),
        interpretation="Analytic work and fitted service estimates; not measured instructions, bandwidth or isolated kernel latency.",
    )


CALIBRATION_METADATA = {
    "abi": 27,
    "training_configurations": 107,
    "training_sha256": "a65b05b40534aac9ac56f79e057eea67913c230cb9a0e6c45612a6ccdedfd3d7",
    "heldout_labels_used": False,
    "calibrated_token_range": [1, 24576],
    "status": "fitted_training_only; validation recorded separately",
}
