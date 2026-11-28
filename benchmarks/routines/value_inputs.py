"""Serving-shaped input values for value-dependent routines.

Most kernels do the same work for any input values, so random tensors time the
served work. Three families do not:

* MoE: the work follows how many tokens each expert receives.
* Sampling: the work follows how peaked each probability row is.
* Attention: the work follows each request's length.

Random router logits, flat probability rows and equal lengths describe a
workload serving does not run. The builders here keep the default (random)
inputs byte-for-byte and add shaped ones behind a spec string:

    --routing_dist        iid | uniform | zipf:<s> | counts:<list-or-file>
    --input_distribution  uniform | normal[:<std>] | zipf:<s> | rows:<file>
    --seq_lens_kv/_q      <list-or-file>

``<list-or-file>`` is either comma-separated numbers or a path. Files may be
.json / .txt (numbers), .pt, .npy or .safetensors; append ``#key`` to pick one
entry of a dict-like file and ``@i`` to pick row ``i`` of a 2-D array.
"""

import json
import math
import os
import re
from typing import List, Optional, Tuple

import torch

# Fixed seed for shaped inputs, so a testlist line builds the same routing,
# rows and popularity order on every run and every GPU.
_SEED = 20261120


def _generator(device, salt: int = 0) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(_SEED + salt)
    return gen


def split_spec(spec: str) -> Tuple[str, str]:
    """``"zipf:1.2"`` -> ``("zipf", "1.2")``; ``"iid"`` -> ``("iid", "")``."""
    kind, _, arg = spec.partition(":")
    return kind.strip().lower(), arg.strip()


def load_values(source: str) -> torch.Tensor:
    """Read numbers from a comma-separated literal or a file (see module doc)."""
    row = None
    match = re.search(r"@(\d+)$", source)
    if match:
        row = int(match.group(1))
        source = source[: match.start()]
    path, _, key = source.partition("#")
    if not os.path.exists(path):
        try:
            values = torch.tensor([float(x) for x in source.split(",") if x.strip()])
        except ValueError:
            raise ValueError(
                f"{source!r} is neither a file nor comma-separated numbers"
            ) from None
    else:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pt":
            data = torch.load(path, map_location="cpu", weights_only=True)
        elif ext == ".npy":
            import numpy as np

            data = torch.from_numpy(np.load(path))
        elif ext == ".safetensors":
            from safetensors.torch import load_file

            data = load_file(path)
        elif ext == ".json":
            with open(path) as f:
                data = json.load(f)
        else:
            with open(path) as f:
                data = [float(x) for x in re.split(r"[,\s]+", f.read()) if x]
        if isinstance(data, dict):
            if not key:
                if len(data) != 1:
                    raise ValueError(
                        f"{path} holds {sorted(data)}; pick one with '{path}#<key>'"
                    )
                key = next(iter(data))
            data = data[key]
        values = torch.as_tensor(data, dtype=torch.float64)
    if row is not None:
        if values.dim() != 2:
            raise ValueError(
                f"'@{row}' needs a 2-D array, got shape {tuple(values.shape)}"
            )
        values = values[row]
    return values


# ---------------------------------------------------------------------------
# MoE routing
# ---------------------------------------------------------------------------


def _ids_to_logits(ids: torch.Tensor, like: torch.Tensor):
    """Logits whose top-k (under any monotone score) is exactly ``ids``."""
    logits = torch.rand(like.shape, device=like.device, dtype=torch.float32)
    boost = 8.0 + torch.rand(ids.shape, device=like.device, dtype=torch.float32)
    logits.scatter_(1, ids.long(), boost)
    return logits.to(like.dtype)


def shape_routing_logits(
    routing_logits: torch.Tensor, top_k: int, spec: Optional[str]
) -> torch.Tensor:
    """Reshape random router logits so the selected experts follow ``spec``.

    The result is still a ``[num_tokens, num_experts]`` logits tensor, so it
    feeds both the in-kernel routing and the torch reference that produces
    precomputed expert ids. Every routing method ranks experts by a monotone
    function of the logit, so the top-k set is the same for all of them.
    (DeepSeekV3 with ``n_group > 1`` first keeps the best groups, which can
    drop some of the requested experts; the row's active_experts shows it.)

    * ``iid``: unchanged. Every expert gets about the same share.
    * ``uniform``: round-robin ids, an exactly even load.
    * ``zipf:<s>``: expert popularity ~ rank^-s, in a fixed shuffled order.
    * ``counts:<list-or-file>``: popularity proportional to tokens-per-expert
      counts captured from serving (one number per expert).

    Popularity is applied by adding log-popularity to Gumbel noise: the top-k
    of that is a draw of k distinct experts with those odds.
    """
    kind, arg = split_spec(spec or "iid")
    if kind == "iid":
        return routing_logits
    num_tokens, num_experts = routing_logits.shape
    device = routing_logits.device
    if kind == "uniform":
        ids = (
            torch.arange(num_tokens, device=device).unsqueeze(1) * top_k
            + torch.arange(top_k, device=device).unsqueeze(0)
        ) % num_experts
        return _ids_to_logits(ids, routing_logits)
    if kind == "zipf":
        exponent = float(arg or 1.0)
        ranks = torch.arange(1, num_experts + 1, dtype=torch.float64)
        popularity = ranks.pow(-exponent)
        order = torch.randperm(num_experts, generator=_generator("cpu"))
        popularity = popularity[torch.argsort(order)]
    elif kind == "counts":
        popularity = load_values(arg).flatten().double()
        if popularity.numel() != num_experts:
            raise ValueError(
                f"--routing_dist counts has {popularity.numel()} entries, "
                f"--num_experts is {num_experts}"
            )
        if (popularity < 0).any() or popularity.sum() <= 0:
            raise ValueError("--routing_dist counts must be non-negative, not all zero")
        if int((popularity > 0).sum()) < top_k:
            raise ValueError(
                f"--routing_dist counts has fewer than top_k={top_k} non-zero experts"
            )
    else:
        raise ValueError(
            f"Unknown --routing_dist {spec!r}: use iid, uniform, zipf:<s> or counts:<...>"
        )
    # Experts with zero popularity sit 30 below the rest: never picked, and
    # finite in every logits dtype.
    log_pop = torch.log(popularity / popularity.sum()).clamp_min(-30.0)
    log_pop = torch.where(popularity > 0, log_pop, log_pop.min() - 30.0)
    uniform = torch.rand(
        (num_tokens, num_experts), generator=_generator(device, 1), device=device
    ).clamp_(1e-12, 1.0 - 1e-7)
    gumbel = -torch.log(-torch.log(uniform))
    logits = gumbel + log_pop.to(device=device, dtype=torch.float32)
    return logits.to(routing_logits.dtype)


def add_routing_args(parser):
    parser.add_argument(
        "--routing_dist",
        type=str,
        default="",
        help=(
            "How tokens spread over experts: iid (default, also when unset; random router logits, "
            "an even share per expert), uniform (exactly even), zipf:<s> (a few "
            "hot experts), or counts:<list-or-file> (tokens-per-expert counts "
            "captured from serving). Serving leaves experts idle at small token "
            "counts and overloads a few at large ones; iid does neither."
        ),
    )


# ---------------------------------------------------------------------------
# Sampling probabilities
# ---------------------------------------------------------------------------

PROB_DIST_HELP = (
    "Probability inputs: uniform (default; normalized uniform noise, almost "
    "flat), normal[:<std>] (softmax of normal logits; a larger std is more "
    "peaked), zipf:<s> (row sorted mass ~ rank^-s), or rows:<file> (captured "
    "rows of leading probabilities; the rest of the vocabulary shares the "
    "remaining mass). Serving rows are peaked; flat rows make top-k/top-p "
    "kernels do far more work."
)


def validate_prob_dist(spec: str) -> str:
    kind, arg = split_spec(spec)
    if kind in ("uniform",) and not arg:
        return spec
    if kind in ("normal", "zipf"):
        if arg:
            float(arg)
        elif kind == "zipf":
            raise ValueError("zipf needs an exponent, e.g. zipf:1.1")
        return spec
    if kind == "rows" and arg:
        return spec
    raise ValueError(f"Unknown --input_distribution {spec!r}")


def is_default_prob_dist(spec: str) -> bool:
    return split_spec(spec)[0] == "uniform"


def _scatter_rows(sorted_probs: torch.Tensor, vocab_size: int, device) -> torch.Tensor:
    """Place each row's probabilities at random vocabulary positions."""
    batch_size = sorted_probs.shape[0]
    positions = torch.rand(
        (batch_size, vocab_size), generator=_generator(device, 2), device=device
    ).argsort(dim=-1)
    out = torch.empty((batch_size, vocab_size), device=device, dtype=torch.float32)
    out.scatter_(1, positions, sorted_probs.to(device=device, dtype=torch.float32))
    return out


def make_probs(
    spec: str, batch_size: int, vocab_size: int, device, dtype=None
) -> torch.Tensor:
    """Probability rows ``[batch_size, vocab_size]`` following ``spec``."""
    kind, arg = split_spec(spec)
    if kind == "uniform":
        # Normalize in float32: a float16 row sum overflows for large vocabularies.
        pre_norm = torch.rand(batch_size, vocab_size, device=device)
        probs = pre_norm / pre_norm.sum(dim=-1, keepdim=True)
    elif kind == "normal":
        logits = torch.randn(batch_size, vocab_size, device=device)
        if arg:
            logits = logits * float(arg)
        probs = torch.softmax(logits, dim=-1)
    elif kind == "zipf":
        ranks = torch.arange(1, vocab_size + 1, dtype=torch.float64)
        row = ranks.pow(-float(arg))
        row = (row / row.sum()).float()
        probs = _scatter_rows(row.expand(batch_size, vocab_size), vocab_size, device)
    elif kind == "rows":
        rows = load_values(arg)
        if rows.dim() == 1:
            rows = rows.unsqueeze(0)
        if rows.dim() != 2 or rows.shape[1] > vocab_size:
            raise ValueError(
                f"rows file must be [num_rows, n <= vocab_size], got {tuple(rows.shape)}"
            )
        if (rows < 0).any():
            raise ValueError("rows file holds negative probabilities")
        head = rows[torch.arange(batch_size) % rows.shape[0]]
        head_mass = head.sum(dim=-1, keepdim=True)
        if (head_mass > 1.0 + 1e-3).any():
            raise ValueError("a row in the rows file sums to more than 1")
        tail_len = vocab_size - head.shape[1]
        full = torch.zeros((batch_size, vocab_size), dtype=torch.float64)
        full[:, : head.shape[1]] = head
        if tail_len > 0:
            full[:, head.shape[1] :] = (1.0 - head_mass).clamp_min(0.0) / tail_len
        full = full / full.sum(dim=-1, keepdim=True)
        probs = _scatter_rows(full.float(), vocab_size, device)
    else:
        raise ValueError(f"Unknown --input_distribution {spec!r}")
    return probs if dtype is None else probs.to(dtype)


def make_logits(
    spec: str, batch_size: int, vocab_size: int, device, dtype
) -> torch.Tensor:
    """Logit rows: standard normal for the default specs, else log of make_probs."""
    kind, arg = split_spec(spec)
    if kind == "uniform" or (kind == "normal" and not arg):
        return torch.randn(batch_size, vocab_size, dtype=dtype, device=device)
    probs = make_probs(spec, batch_size, vocab_size, device)
    return torch.log(probs.clamp_min(1e-30)).to(dtype)


# ---------------------------------------------------------------------------
# Sequence lengths
# ---------------------------------------------------------------------------


def explicit_seq_lens(
    source: Optional[str], name: str, batch_size: int, max_len: int, device
) -> Optional[torch.Tensor]:
    """Per-request lengths from ``--seq_lens_*`` as ``[batch, 1, 1, 1]`` int32.

    Returns None when the flag is unset. Lengths must be one per request and
    within ``[1, max_len]``: the caches are sized from --s_qo / --s_kv.
    """
    if not source:
        return None
    values = load_values(source).flatten()
    lens: List[int] = [int(v) for v in values.tolist()]
    if any(not math.isfinite(v) or v != int(v) for v in values.tolist()):
        raise ValueError(f"--{name} must hold integers")
    if len(lens) != batch_size:
        raise ValueError(
            f"--{name} has {len(lens)} lengths, --batch_size is {batch_size}"
        )
    if min(lens) < 1 or max(lens) > max_len:
        raise ValueError(
            f"--{name} lengths must be in [1, {max_len}] (the routine's maximum "
            f"length flag); got min {min(lens)}, max {max(lens)}"
        )
    return torch.tensor(lens, dtype=torch.int32, device=device).view(-1, 1, 1, 1)


def add_seq_len_args(parser):
    parser.add_argument(
        "--seq_lens_kv",
        type=str,
        default="",
        help=(
            "Per-request KV lengths, one per request: comma-separated or a file. "
            "Each must be <= --s_kv, which still sizes the cache. Overrides "
            "--random_actual_seq_len. Serving batches mix short and long "
            "requests, and the longest sets the tail; equal lengths hide that."
        ),
    )
    parser.add_argument(
        "--seq_lens_q",
        type=str,
        default="",
        help=(
            "Prefill only: per-request query lengths, one per request, each "
            "<= --s_qo. KV lengths are raised to at least the query length."
        ),
    )
    parser.add_argument(
        "--decode_max_seq_len",
        type=str,
        default="",
        help=(
            "trtllm-gen decode only: the max_seq_len the kernel is told (it "
            "picks how each request's KV is split across blocks). Default: "
            "--s_kv. Serving passes its context-window cap, not the batch's "
            "longest request."
        ),
    )
