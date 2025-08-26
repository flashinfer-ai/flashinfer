import pytest
import torch

from flashinfer.sampling import _to_tensor_scalar_tuple, get_sampling_module


def opcheck(fn):
    def wrapper(*args, **kwargs):
        return torch.library.opcheck(fn, args, kwargs)

    return wrapper


module = get_sampling_module()
chain_speculative_sampling = opcheck(module.chain_speculative_sampling)
min_p_sampling_from_probs = opcheck(module.min_p_sampling_from_probs)
top_p_renorm_probs = opcheck(module.top_p_renorm_probs)
top_p_sampling_from_probs = opcheck(module.top_p_sampling_from_probs)
top_k_renorm_probs = opcheck(module.top_k_renorm_probs)
top_k_sampling_from_probs = opcheck(module.top_k_sampling_from_probs)
top_k_top_p_sampling_from_probs = opcheck(module.top_k_top_p_sampling_from_probs)
top_k_mask_logits = opcheck(module.top_k_mask_logits)
softmax = opcheck(module.softmax)
sampling_from_probs = opcheck(module.sampling_from_probs)
sampling_from_logits = opcheck(module.sampling_from_logits)


def normal_distribution(std):
    def normal_noise(shape, device):
        return torch.randn(shape, device=device) * std

    normal_noise.__name__ = f"normal_distribution(std={std})"
    return normal_noise


def gumbel_distribution(beta):
    def gumbel_noise(shape, device):
        U = torch.rand(shape, device=device)
        eps = 1e-20
        return torch.log(-torch.log(U + eps) + eps) / beta

    gumbel_noise.__name__ = f"gumbel_distribution(beta={beta})"
    return gumbel_noise


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize(
    "distribution",
    [
        normal_distribution(1),
        normal_distribution(5),
        gumbel_distribution(0.1),
    ],
)
@pytest.mark.parametrize("temperature", [1.0, 0.5, 0.1])
@pytest.mark.parametrize("temperature_arr", [True, False])
@pytest.mark.parametrize("neg_inf_input", [True, False])
def test_softmax(
    batch_size, vocab_size, distribution, temperature, temperature_arr, neg_inf_input
):
    torch.manual_seed(42)
    logits = distribution((batch_size, vocab_size), "cuda:0")
    if neg_inf_input:
        # assign random logits to -inf
        num_inf = torch.randint(0, logits.numel() - 1, (), device=logits.device).item()
        inf_idx = torch.randperm(logits.numel(), device=logits.device)[:num_inf]
        logits.view(-1).index_fill_(0, inf_idx, float("-inf"))

    workspace_buffer = torch.empty(1024 * 1024, device=logits.device)

    if temperature_arr:
        temperature_arr = torch.full((batch_size,), temperature, device="cuda:0")
        softmax(
            workspace_buffer, logits, *_to_tensor_scalar_tuple(temperature_arr), False
        )
    else:
        softmax(workspace_buffer, logits, *_to_tensor_scalar_tuple(temperature), False)


# Reduce number of inputs
@pytest.mark.parametrize(
    "vocab_size",
    [
        111,
    ],
)
@pytest.mark.parametrize(
    "distribution",
    [
        normal_distribution(1),
        gumbel_distribution(0.1),
    ],
)
@pytest.mark.parametrize("zero_ratio", [0.0, 0.5])
def test_sampling_freq(vocab_size, distribution, zero_ratio):
    torch.manual_seed(42)
    num_trials = 5000000
    logits = distribution((1, vocab_size), "cuda:0")
    zero_indices = torch.randperm(vocab_size)[: int(vocab_size * zero_ratio)]
    logits[:, zero_indices] = -float("inf")
    probs = torch.softmax(logits, dim=-1)

    sampling_from_probs(
        probs,
        indices=torch.zeros(num_trials, dtype=torch.int32, device=logits.device),
        deterministic=True,
        generator=None,
    )


@pytest.mark.parametrize(
    "vocab_size",
    [
        111,
    ],
)
@pytest.mark.parametrize(
    "distribution",
    [
        normal_distribution(1),
        gumbel_distribution(0.1),
    ],
)
@pytest.mark.parametrize("p", [0.1, 0.5])
def test_top_p_sampling_freq(vocab_size, distribution, p):
    # use torch profiler to check the performance of the code
    torch.manual_seed(42)
    logits = distribution((1, vocab_size), "cuda:0")
    probs = torch.softmax(logits, dim=-1)
    sorted_prob, indices = torch.sort(probs, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros(1, vocab_size, dtype=torch.int32, device=logits.device)
    mask.scatter_add_(1, indices, (cdf > (1 - p)).int())

    top_p_renorm_probs(probs, *_to_tensor_scalar_tuple(p))

    num_trials = 5000000
    top_p_sampling_from_probs(
        probs,
        torch.zeros(num_trials, dtype=torch.int32, device=logits.device),  # indices
        *_to_tensor_scalar_tuple(p),
        True,  # deterministic,
        None,  # generator
    )


@pytest.mark.parametrize(
    "vocab_size",
    [
        111,
    ],
)
@pytest.mark.parametrize(
    "distribution",
    [
        normal_distribution(1),
        gumbel_distribution(0.1),
    ],
)
@pytest.mark.parametrize("k", [10, 100])
def test_top_k_sampling_freq(vocab_size, distribution, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    logits = distribution((1, vocab_size), "cuda:0")
    probs = torch.softmax(logits, dim=-1)

    top_k_renorm_probs(probs, *_to_tensor_scalar_tuple(k))

    num_trials = 5000000
    top_k_sampling_from_probs(
        probs,
        torch.zeros(num_trials, dtype=torch.int32, device=logits.device),  # indices
        *_to_tensor_scalar_tuple(k),  # (maybe_top_k_array, top_k_val)
        False,  # deterministic
        None,  # generator
    )


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize(
    "vocab_size",
    [
        111,
    ],
)
def test_sampling(batch_size, vocab_size):
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sampling_from_probs(
        normalized_prob,
        None,  # indices
        True,  # deterministic
        None,  # generator
    )


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize(
    "vocab_size",
    [
        111,
    ],
)
def test_sampling_from_logits(batch_size, vocab_size):
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda:0")
    sampling_from_logits(
        logits,
        None,  # indices
        True,  # deterministic
        None,  # generator
    )


@pytest.mark.parametrize(
    "vocab_size",
    [
        111,
    ],
)
@pytest.mark.parametrize(
    "distribution",
    [
        normal_distribution(1),
        gumbel_distribution(0.1),
    ],
)
def test_sampling_from_logits_freq(vocab_size, distribution):
    torch.manual_seed(42)
    num_trials = 5000000
    logits = distribution((1, vocab_size), "cuda:0")
    sampling_from_logits(
        logits,
        torch.zeros(num_trials, dtype=torch.int32, device=logits.device),
        True,  # deterministic
        None,  # generator
    )


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize(
    "vocab_size",
    [
        111,
    ],
)
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_top_p_sampling(batch_size, vocab_size, p):
    torch.manual_seed(42)
    eps = 1e-4
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device="cuda:0")
    mask.scatter_add_(1, indices, (cdf > (1 - p) - eps).int())
    top_p_sampling_from_probs(
        normalized_prob,
        None,  # indices
        *_to_tensor_scalar_tuple(p),
        True,  # deterministic,
        None,  # generator
    )


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_sampling(batch_size, vocab_size, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    top_k_sampling_from_probs(
        normalized_prob,
        None,  # indices
        *_to_tensor_scalar_tuple(k),  # (maybe_top_k_array, top_k_val)
        False,  # deterministic
        None,  # generator
    )


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.05, 0.1, 0.2, 0.7, 1])
def test_min_p_sampling(batch_size, vocab_size, p):
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    # scale min-p
    top_probs = sorted_prob[:, -1].unsqueeze(-1)
    scaled_p = p * top_probs
    # min-p mask
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device="cuda:0")
    mask.scatter_add_(1, indices, (sorted_prob >= scaled_p).int())
    min_p_tensor = torch.full((batch_size,), p, device="cuda:0")

    min_p_sampling_from_probs(
        normalized_prob,
        None,  # indices,
        *_to_tensor_scalar_tuple(min_p_tensor),
        True,  # deterministic
        None,  # generator
    )


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize(
    "vocab_size",
    [
        111,
    ],
)
@pytest.mark.parametrize("p", [0.1, 0.5])
def test_top_k_top_p_joint_sampling_from_probs(batch_size, vocab_size, p):
    torch.manual_seed(42)
    if p == 0.1:
        k = int(vocab_size * 0.5)
    elif p == 0.5:
        k = int(vocab_size * 0.1)
    else:
        raise ValueError("p not recognized")

    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    top_p_tensor = torch.full((batch_size,), p, device="cuda:0")
    top_k_tensor = torch.full((batch_size,), k, device="cuda:0")

    top_k_top_p_sampling_from_probs(
        normalized_prob,
        None,  # indices
        *_to_tensor_scalar_tuple(top_k_tensor),
        *_to_tensor_scalar_tuple(top_p_tensor),
        True,  # deterministic,
        None,  # generator
    )


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize(
    "vocab_size",
    [
        111,
    ],
)
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_top_p_renorm_probs(batch_size, vocab_size, p):
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    top_p_renorm_probs(normalized_prob, *_to_tensor_scalar_tuple(p))


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize(
    "vocab_size",
    [
        111,
    ],
)
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_renorm_probs(batch_size, vocab_size, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    top_k_renorm_probs(normalized_prob, *_to_tensor_scalar_tuple(k))


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize(
    "vocab_size",
    [
        111,
    ],
)
@pytest.mark.parametrize("k", [10, 100, 500])
@pytest.mark.parametrize("neginf_input", [False, True])
def test_top_k_mask_logits(batch_size, vocab_size, k, neginf_input):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda:0") * 5
    if neginf_input:
        num_neginf = torch.randint(1, vocab_size * batch_size, (1,)).item()
        idxs = torch.randperm(batch_size * vocab_size, device="cuda:0")[:num_neginf]
        logits[idxs // vocab_size, idxs % vocab_size] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    top_k_mask_logits(logits, *_to_tensor_scalar_tuple(k))
    top_k_renorm_probs(probs, *_to_tensor_scalar_tuple(k))


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize(
    "vocab_size",
    [
        111,
    ],
)
@pytest.mark.parametrize("num_speculate_tokens", [1, 3, 5, 7])
@pytest.mark.parametrize("onehot_target", [False, True])
def test_chain_speculative_sampling(
    batch_size,
    vocab_size,
    num_speculate_tokens,
    onehot_target,
):
    pre_norm_draft_prob = torch.rand(
        batch_size, num_speculate_tokens, vocab_size, device="cuda:0"
    )
    normalized_draft_prob = pre_norm_draft_prob / pre_norm_draft_prob.sum(
        dim=-1, keepdim=True
    )
    draft_token_ids = torch.randint(
        vocab_size, (batch_size, num_speculate_tokens), device="cuda:0"
    )
    if not onehot_target:
        pre_norm_target_prob = torch.rand(
            batch_size, num_speculate_tokens + 1, vocab_size, device="cuda:0"
        )
        target_onehot_prob = pre_norm_target_prob / pre_norm_target_prob.sum(
            dim=-1, keepdim=True
        )
    else:
        target_token_ids = torch.randint(
            vocab_size, (batch_size, num_speculate_tokens + 1), device="cuda:0"
        )
        target_token_ids[..., :num_speculate_tokens] = draft_token_ids
        target_onehot_prob = torch.zeros(
            (batch_size, num_speculate_tokens + 1, vocab_size), device="cuda:0"
        )
        target_onehot_prob.scatter_(2, target_token_ids.unsqueeze(-1), 1)

    accepted_num = torch.zeros(batch_size, dtype=torch.int32, device="cuda:0")
    emitted_num = torch.zeros(batch_size, dtype=torch.int32, device="cuda:0")
    chain_speculative_sampling(
        normalized_draft_prob,
        draft_token_ids,
        target_onehot_prob,
        accepted_num,
        emitted_num,
        True,  # deterministic
        None,  # generator
    )
