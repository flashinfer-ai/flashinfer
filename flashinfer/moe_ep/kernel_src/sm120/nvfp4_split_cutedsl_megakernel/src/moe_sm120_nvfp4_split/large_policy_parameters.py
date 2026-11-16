PARAMETERS = {
    "models": {
        "balanced": [
            [-0.017494052674048074, -0.018531115468068144],
            [-0.004833609737700763, -0.004053408872880447],
            [0.020880986729061544, 0.02168465858615267],
        ],
        "power_law": [
            [-0.0040195227045004, -0.004099473574249786],
            [-7.678184699816267e-05, 0.0001092308892507745],
            [-0.003951781541549089, -0.0030301938617634493],
        ],
        "zero_count_peer": [
            [0.004028265729136707, 0.004458693157308941],
            [-0.0016251717657858415, -0.0012521857797230071],
            [-0.0031204770937322372, -0.002295154238794959],
        ],
    },
    "routes": ["balanced", "power_law", "zero_count_peer"],
    "alpha": 0.05,
    "mean_log_gain_margin": 0.005,
    "worst_log_p50_limit": 0.005,
    "worst_log_p90_limit": 0.01,
    "density_min": 576.0,
    "density_max": 2731.0,
    "training_data_sha256": "0be14c38a3a72a484f0b005968090d8886edea7c746e3210cb013ecb0f756461",
    "features": ["1", "log2(expected_rows_per_expert/768)", "log(topk/6)/log(8/6)"],
    "notes": "Small paired-change ridge model; routes are offline scenarios only. Limits are conservative policy gates, not confidence bounds. Two target geometries; no universal-model claim.",
}
