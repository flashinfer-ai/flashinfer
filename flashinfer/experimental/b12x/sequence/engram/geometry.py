"""DeepSeek V4.1 Engram tokenizer and immutable hash geometry."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ple_hash.reference import is_prime_64


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Use raw backend decoding, retaining malformed UTF-8 token spellings."""
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(),
        normalizers.Lowercase(), normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel), normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    backend = tokenizer.backend_tokenizer
    keys: dict[str, int] = {}
    lookup = []
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        key = (backend.id_to_token(token_id) if "\ufffd" in text
               else normalizer.normalize_str(text) or text)
        if key not in keys:
            keys[key] = len(keys)
        lookup.append(keys[key])
    return lookup, len(keys)


@dataclass(frozen=True)
class Geometry:
    """Host-only immutable geometry; all device geometry uses signed int64."""
    layer_ids: tuple[int, ...]
    primes: tuple[tuple[int, ...], ...]
    offsets: tuple[tuple[int, ...], ...]
    multipliers: tuple[tuple[int, ...], ...]
    num_embeddings: tuple[int, ...]
    compressed_vocab_size: int


def build_geometry(*, layer_ids=(1, 14), base_table_size=16_000_000,
                   compressed_vocab_size=99_092) -> Geometry:
    """Generate 2/3/4-gram, eight-head geometry with globally unreused primes."""
    layer_ids = tuple(layer_ids)
    if (not layer_ids or len(set(layer_ids)) != len(layer_ids)
            or any(i < 0 for i in layer_ids)):
        raise ValueError("layer_ids must be unique nonnegative actual layer IDs")
    if base_table_size < 2 or not 0 < compressed_vocab_size < (1 << 62):
        raise ValueError("invalid table or compressed vocabulary size")
    seen: set[int] = set()
    primes, offsets, multipliers, totals = [], [], [], []
    bound = max(1, (((1 << 63) - 1) // compressed_vocab_size) // 2)
    for layer in layer_ids:
        sizes = []
        for _ in range(3):
            current = base_table_size - 1
            for _ in range(8):
                current += 1
                while current in seen or not is_prime_64(current):
                    current += 1
                seen.add(current)
                sizes.append(current)
        cumulative, total = [], 0
        for size in sizes:
            cumulative.append(total)
            total += size
        if total * 256 > (1 << 63) - 1:
            raise ValueError("table extent exceeds signed int64 indexing")
        rng = np.random.Generator(np.random.PCG64(10007 * layer))
        values = rng.integers(0, bound, size=4, dtype=np.int64) * 2 + 1
        primes.append(tuple(sizes))
        offsets.append(tuple(cumulative))
        multipliers.append(tuple(int(v) for v in values))
        totals.append(total)
    return Geometry(layer_ids, tuple(primes), tuple(offsets), tuple(multipliers),
                    tuple(totals), compressed_vocab_size)
