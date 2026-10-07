"""Read vLLM prompt-source counters for exact cold-prefill qualification."""

import urllib.request

_SOURCES = ("external_kv_transfer", "local_compute", "local_cache_hit")


def metrics(base_url: str) -> dict[str, float]:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as response:
        text = response.read().decode()
    metric_name = "vllm:prompt_tokens_by_source_total"
    values = {source: 0.0 for source in _SOURCES}
    for line in text.splitlines():
        if not line.startswith(metric_name + "{"):
            continue
        for source in _SOURCES:
            if f'source="{source}"' in line:
                values[source] = float(line.rsplit(" ", 1)[1])
    return values


def delta(before: dict[str, float], after: dict[str, float]) -> dict[str, int]:
    return {key: round(after[key] - before[key]) for key in _SOURCES}
