import numpy as np
import pytest
import torch

import flashinfer
from flashinfer.logits_processor import (
    LogitsPipe,
    MinP,
    Sample,
    Softmax,
    Temperature,
    TensorType,
    TopK,
    TopP,
)


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


def set_random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_generators():
    gen1 = torch.Generator("cuda:0")
    gen1.manual_seed(42)
    gen2 = gen1.clone_state()
    return gen1, gen2


class TestLogitsPipeCompilation:
    """Test LogitsPipe with compile=True vs compile=False"""

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
    def test_temperature_softmax(
        self, batch_size, vocab_size, distribution, temperature
    ):
        set_random_seed(42)
        logits = distribution((batch_size, vocab_size), "cuda:0")

        pipe_compiled = LogitsPipe([Temperature(), Softmax()], compile=True)
        pipe_no_compile = LogitsPipe([Temperature(), Softmax()], compile=False)

        probs_compiled = pipe_compiled(logits, temperature=temperature)
        probs_no_compile = pipe_no_compile(logits, temperature=temperature)

        assert torch.allclose(probs_compiled, probs_no_compile, atol=1e-5)

    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize(
        "distribution",
        [
            normal_distribution(1),
            normal_distribution(5),
            gumbel_distribution(0.1),
        ],
    )
    @pytest.mark.parametrize("zero_ratio", [0.0, 0.5, 0.9])
    def test_probs_sample_freq(self, vocab_size, distribution, zero_ratio):
        set_random_seed(42)
        num_trials = 5000000

        logits = distribution((1, vocab_size), "cuda:0")
        zero_indices = torch.randperm(vocab_size)[: int(vocab_size * zero_ratio)]
        logits[:, zero_indices] = -float("inf")
        probs = torch.softmax(logits, dim=-1)

        pipe_compiled = LogitsPipe(
            [Sample()], compile=True, input_type=TensorType.PROBS
        )
        counter_compiled = torch.zeros(vocab_size, dtype=torch.int32, device="cuda:0")

        samples_compiled = pipe_compiled(
            probs, indices=torch.zeros(num_trials, dtype=torch.int32, device="cuda:0")
        )
        counter_compiled.scatter_add_(
            0, samples_compiled.long(), torch.ones_like(samples_compiled)
        )
        freq_compiled = counter_compiled.float() / num_trials

        pipe_no_compile = LogitsPipe(
            [Sample()], compile=False, input_type=TensorType.PROBS
        )
        counter_no_compile = torch.zeros(vocab_size, dtype=torch.int32, device="cuda:0")
        samples_no_compile = pipe_no_compile(
            probs, indices=torch.zeros(num_trials, dtype=torch.int32, device="cuda:0")
        )
        counter_no_compile.scatter_add_(
            0, samples_no_compile.long(), torch.ones_like(samples_no_compile)
        )
        freq_no_compile = counter_no_compile.float() / num_trials

        # check if the zero indices are never sampled
        assert torch.all(counter_compiled[zero_indices] == 0) and torch.all(
            counter_no_compile[zero_indices] == 0
        )

        # check if sampled results follow given distribution
        similarity_compiled = torch.cosine_similarity(freq_compiled, probs)
        similarity_no_compile = torch.cosine_similarity(freq_no_compile, probs)
        assert similarity_compiled > 0.99, f"Compiled similarity: {similarity_compiled}"
        assert similarity_no_compile > 0.99, (
            f"Non-compiled similarity: {similarity_no_compile}"
        )

        # check if compiled and non-compiled results are similar
        freq_similarity = torch.cosine_similarity(freq_compiled, freq_no_compile, dim=0)
        assert freq_similarity > 0.99, (
            f"Compiled vs non-compiled similarity: {freq_similarity}"
        )

    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize(
        "distribution",
        [
            normal_distribution(1),
            normal_distribution(5),
            gumbel_distribution(0.1),
        ],
    )
    def test_logits_sample_freq(self, vocab_size, distribution):
        set_random_seed(42)
        num_trials = 5000000

        logits = distribution((1, vocab_size), "cuda:0")
        probs = torch.softmax(logits, dim=-1)

        pipe_compiled = LogitsPipe(
            [Sample()], compile=True, input_type=TensorType.LOGITS
        )
        counter_compiled = torch.zeros(vocab_size, dtype=torch.int32, device="cuda:0")

        samples_compiled = pipe_compiled(
            logits, indices=torch.zeros(num_trials, dtype=torch.int32, device="cuda:0")
        )
        counter_compiled.scatter_add_(
            0, samples_compiled.long(), torch.ones_like(samples_compiled)
        )
        freq_compiled = counter_compiled.float() / num_trials

        pipe_no_compile = LogitsPipe(
            [Sample()], compile=False, input_type=TensorType.LOGITS
        )
        counter_no_compile = torch.zeros(vocab_size, dtype=torch.int32, device="cuda:0")
        samples_no_compile = pipe_no_compile(
            logits, indices=torch.zeros(num_trials, dtype=torch.int32, device="cuda:0")
        )
        counter_no_compile.scatter_add_(
            0, samples_no_compile.long(), torch.ones_like(samples_no_compile)
        )
        freq_no_compile = counter_no_compile.float() / num_trials

        # check if sampled results follow given distribution
        similarity_compiled = torch.cosine_similarity(freq_compiled, probs)
        similarity_no_compile = torch.cosine_similarity(freq_no_compile, probs)
        assert similarity_compiled > 0.99, f"Compiled similarity: {similarity_compiled}"
        assert similarity_no_compile > 0.99, (
            f"Non-compiled similarity: {similarity_no_compile}"
        )

        # check if compiled and non-compiled results are similar
        freq_similarity = torch.cosine_similarity(freq_compiled, freq_no_compile, dim=0)
        assert freq_similarity > 0.99, (
            f"Compiled vs non-compiled similarity: {freq_similarity}"
        )

    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize(
        "distribution",
        [
            normal_distribution(1),
            normal_distribution(5),
            gumbel_distribution(0.1),
        ],
    )
    @pytest.mark.parametrize("k", [10, 100, 500])
    def test_probs_top_k_sample_freq(self, vocab_size, distribution, k):
        if k > vocab_size:
            pytest.skip("k should be less than vocab_size")

        set_random_seed(42)
        num_trials = 5000000

        logits = distribution((1, vocab_size), "cuda:0")
        probs = torch.softmax(logits, dim=-1)

        sorted_prob, _ = torch.sort(probs, descending=True)
        pivot = sorted_prob[:, k - 1]
        mask = (probs >= pivot.unsqueeze(-1)).int()
        masked_probs = probs.clone()
        masked_probs[mask == 0] = 0

        pipe_compiled = LogitsPipe(
            [TopK(), Sample()], compile=True, input_type=TensorType.PROBS
        )
        counter_compiled = torch.zeros(vocab_size, dtype=torch.int32, device="cuda:0")

        samples_compiled = pipe_compiled(
            probs,
            indices=torch.zeros(num_trials, dtype=torch.int32, device="cuda:0"),
            top_k=k,
        )
        counter_compiled.scatter_add_(
            0, samples_compiled.long(), torch.ones_like(samples_compiled)
        )
        freq_compiled = counter_compiled.float() / num_trials

        pipe_no_compile = LogitsPipe(
            [TopK(), Sample()], compile=False, input_type=TensorType.PROBS
        )
        counter_no_compile = torch.zeros(vocab_size, dtype=torch.int32, device="cuda:0")
        samples_no_compile = pipe_no_compile(
            probs,
            indices=torch.zeros(num_trials, dtype=torch.int32, device="cuda:0"),
            top_k=k,
        )
        counter_no_compile.scatter_add_(
            0, samples_no_compile.long(), torch.ones_like(samples_no_compile)
        )
        freq_no_compile = counter_no_compile.float() / num_trials

        # check if the top-k thresholding is properly applied
        assert torch.all(mask[torch.arange(1), samples_compiled] == 1)
        assert torch.all(mask[torch.arange(1), samples_no_compile] == 1)

        similarity_compiled = torch.cosine_similarity(freq_compiled, masked_probs)
        similarity_no_compile = torch.cosine_similarity(freq_no_compile, masked_probs)
        # check if the sampled results follow given distribution
        assert similarity_compiled > 0.99, f"Compiled similarity: {similarity_compiled}"
        assert similarity_no_compile > 0.99, (
            f"Non-compiled similarity: {similarity_no_compile}"
        )

        # check if compiled and non-compiled results are similar
        freq_similarity = torch.cosine_similarity(freq_compiled, freq_no_compile, dim=0)
        assert freq_similarity > 0.99, (
            f"Compiled vs non-compiled similarity: {freq_similarity}"
        )

    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize(
        "distribution",
        [
            normal_distribution(1),
            normal_distribution(5),
            gumbel_distribution(0.1),
        ],
    )
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    def test_probs_top_p_sample_freq(self, vocab_size, distribution, p):
        set_random_seed(42)
        num_trials = 5000000
        eps = 1e-4

        logits = distribution((1, vocab_size), "cuda:0")
        probs = torch.softmax(logits, dim=-1)

        sorted_prob, indices = torch.sort(probs, descending=False)
        cdf = torch.cumsum(sorted_prob, dim=-1)
        mask = torch.zeros(1, vocab_size, dtype=torch.int32, device="cuda:0")
        mask.scatter_add_(1, indices, (cdf > (1 - p) - eps).int())
        masked_probs = probs.clone()
        masked_probs[mask == 0] = 0

        pipe_compiled = LogitsPipe(
            [TopP(), Sample()],
            compile=True,
        )
        counter_compiled = torch.zeros(vocab_size, dtype=torch.int32, device="cuda:0")
        samples_compiled = pipe_compiled(
            probs,
            indices=torch.zeros(num_trials, dtype=torch.int32, device="cuda:0"),
            top_p=p,
        )
        counter_compiled.scatter_add_(
            0, samples_compiled.long(), torch.ones_like(samples_compiled)
        )
        freq_compiled = counter_compiled.float() / num_trials

        pipe_no_compile = LogitsPipe(
            [TopP(), Sample()], compile=False, input_type=TensorType.PROBS
        )
        counter_no_compile = torch.zeros(vocab_size, dtype=torch.int32, device="cuda:0")
        samples_no_compile = pipe_no_compile(
            probs,
            indices=torch.zeros(num_trials, dtype=torch.int32, device="cuda:0"),
            top_p=p,
        )
        counter_no_compile.scatter_add_(
            0, samples_no_compile.long(), torch.ones_like(samples_no_compile)
        )
        freq_no_compile = counter_no_compile.float() / num_trials

        # check if the top-p thresholding is properly applied
        assert torch.all(mask[torch.arange(1), samples_compiled] == 1)
        assert torch.all(mask[torch.arange(1), samples_no_compile] == 1)

        # check if the sampled results follow given distribution
        similarity_compiled = torch.cosine_similarity(freq_compiled, masked_probs)
        similarity_no_compile = torch.cosine_similarity(freq_no_compile, masked_probs)
        assert similarity_compiled > 0.99, f"Compiled similarity: {similarity_compiled}"
        assert similarity_no_compile > 0.99, (
            f"Non-compiled similarity: {similarity_no_compile}"
        )

        # check if compiled and non-compiled results are similar
        freq_similarity = torch.cosine_similarity(freq_compiled, freq_no_compile, dim=0)
        assert freq_similarity > 0.99, (
            f"Compiled vs non-compiled similarity: {freq_similarity}"
        )

    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize(
        "distribution",
        [
            normal_distribution(1),
            normal_distribution(5),
            gumbel_distribution(0.1),
        ],
    )
    @pytest.mark.parametrize("p", [0.05, 0.1, 0.2, 0.7, 1])
    def test_probs_min_p_sample_freq(self, vocab_size, distribution, p):
        set_random_seed(42)
        num_trials = 5000000

        logits = distribution((1, vocab_size), "cuda:0")
        probs = torch.softmax(logits, dim=-1)

        sorted_prob, indices = torch.sort(probs, descending=False)
        top_probs = sorted_prob[:, -1].unsqueeze(-1)
        scaled_p = p * top_probs

        mask = torch.zeros(1, vocab_size, dtype=torch.int32, device="cuda:0")
        mask.scatter_add_(1, indices, (sorted_prob >= scaled_p).int())
        masked_probs = probs.clone()
        masked_probs[mask == 0] = 0

        pipe_compiled = LogitsPipe(
            [MinP(), Sample()],
            compile=True,
        )
        counter_compiled = torch.zeros(vocab_size, dtype=torch.int32, device="cuda:0")
        samples_compiled = pipe_compiled(
            probs,
            indices=torch.zeros(num_trials, dtype=torch.int32, device="cuda:0"),
            min_p=p,
        )
        counter_compiled.scatter_add_(
            0, samples_compiled.long(), torch.ones_like(samples_compiled)
        )
        freq_compiled = counter_compiled.float() / num_trials

        pipe_no_compile = LogitsPipe(
            [MinP(), Sample()],
            compile=False,
        )
        counter_no_compile = torch.zeros(vocab_size, dtype=torch.int32, device="cuda:0")
        samples_no_compile = pipe_no_compile(
            probs,
            indices=torch.zeros(num_trials, dtype=torch.int32, device="cuda:0"),
            min_p=p,
        )
        counter_no_compile.scatter_add_(
            0, samples_no_compile.long(), torch.ones_like(samples_no_compile)
        )
        freq_no_compile = counter_no_compile.float() / num_trials

        # check if the min-p thresholding is properly applied
        assert torch.all(mask[torch.arange(1), samples_compiled] == 1)
        assert torch.all(mask[torch.arange(1), samples_no_compile] == 1)

        # check if the sampled results follow given distribution
        similarity_compiled = torch.cosine_similarity(freq_compiled, masked_probs)
        similarity_no_compile = torch.cosine_similarity(freq_no_compile, masked_probs)
        assert similarity_compiled > 0.99, f"Compiled similarity: {similarity_compiled}"
        assert similarity_no_compile > 0.99, (
            f"Non-compiled similarity: {similarity_no_compile}"
        )

        # check if compiled and non-compiled results are similar
        freq_similarity = torch.cosine_similarity(freq_compiled, freq_no_compile, dim=0)
        assert freq_similarity > 0.99, (
            f"Compiled vs non-compiled similarity: {freq_similarity}"
        )

    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize(
        "distribution",
        [
            normal_distribution(1),
            normal_distribution(5),
            gumbel_distribution(0.1),
        ],
    )
    @pytest.mark.parametrize("p", [0.1, 0.5])
    def test_probs_top_k_top_p_joint_sample_freq(self, vocab_size, distribution, p):
        set_random_seed(42)
        num_trials = 5000000
        eps = 1e-4

        if p == 0.1:
            k = int(vocab_size * 0.5)
        elif p == 0.5:
            k = int(vocab_size * 0.1)
        else:
            raise ValueError("p not recognized")

        logits = distribution((1, vocab_size), "cuda:0")
        probs = torch.softmax(logits, dim=-1)

        sorted_prob_asc, idx_asc = torch.sort(probs, descending=False)
        cdf = torch.cumsum(sorted_prob_asc, dim=-1)
        mask_top_p = torch.zeros(1, vocab_size, dtype=torch.int32, device="cuda:0")
        mask_top_p.scatter_add_(1, idx_asc, (cdf > (1 - p) - eps).int())

        sorted_prob_desc, _ = torch.sort(probs, descending=True)
        pivot = sorted_prob_desc[:, k - 1]
        mask_top_k = (probs >= pivot.unsqueeze(-1)).int()

        mask = torch.minimum(mask_top_k, mask_top_p)
        masked_probs = probs.clone()
        masked_probs[mask == 0] = 0

        pipe_compiled = LogitsPipe(
            [
                TopK(joint_topk_topp=True),
                TopP(),
                Sample(),
            ],
            compile=True,
            input_type=TensorType.PROBS,
        )
        counter_compiled = torch.zeros(vocab_size, dtype=torch.int32, device="cuda:0")
        samples_compiled = pipe_compiled(
            probs,
            indices=torch.zeros(num_trials, dtype=torch.int32, device="cuda:0"),
            top_k=k,
            top_p=p,
        )
        counter_compiled.scatter_add_(
            0, samples_compiled.long(), torch.ones_like(samples_compiled)
        )
        freq_compiled = counter_compiled.float() / num_trials

        pipe_no_compile = LogitsPipe(
            [
                TopK(),
                TopP(),
                Sample(),
            ],
            compile=False,
            input_type=TensorType.PROBS,
        )
        samples_no_compile = pipe_no_compile(
            probs,
            indices=torch.zeros(num_trials, dtype=torch.int32, device="cuda:0"),
            top_k=k,
            top_p=p,
        )

        # check if the top-k-top-p thresholding is properly applied
        assert torch.all(mask[torch.arange(1), samples_compiled] == 1)
        assert torch.all(mask[torch.arange(1), samples_no_compile] == 1)

        # check if the sampled results follow given distribution
        # we don't check the non-compiled results because joint topk-topp yeilds different results from topk then topp
        # same for the compile-non-compile similarity as well
        similarity_compiled = torch.cosine_similarity(freq_compiled, masked_probs)
        assert similarity_compiled > 0.99, f"Compiled similarity: {similarity_compiled}"

    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize(
        "distribution",
        [
            normal_distribution(1),
            normal_distribution(5),
            gumbel_distribution(0.1),
        ],
    )
    @pytest.mark.parametrize("p", [0.1, 0.5])
    def test_logits_top_k_top_p_joint_sample_freq(self, vocab_size, distribution, p):
        set_random_seed(42)
        num_trials = 5000000
        eps = 1e-4

        if p == 0.1:
            k = int(vocab_size * 0.5)
        elif p == 0.5:
            k = int(vocab_size * 0.1)
        else:
            raise ValueError("p not recognized")

        logits = distribution((1, vocab_size), "cuda:0")
        probs = torch.softmax(logits, dim=-1)

        sorted_prob_asc, idx_asc = torch.sort(probs, descending=False)
        cdf = torch.cumsum(sorted_prob_asc, dim=-1)
        mask_top_p = torch.zeros(1, vocab_size, dtype=torch.int32, device="cuda:0")
        mask_top_p.scatter_add_(1, idx_asc, (cdf > (1 - p) - eps).int())

        sorted_prob_desc, _ = torch.sort(probs, descending=True)
        pivot = sorted_prob_desc[:, k - 1]
        mask_top_k = (probs >= pivot.unsqueeze(-1)).int()

        mask = torch.minimum(mask_top_k, mask_top_p)
        masked_probs = probs.clone()
        masked_probs[mask == 0] = 0

        pipe_compiled = LogitsPipe(
            [
                Softmax(),
                TopK(joint_topk_topp=True),
                TopP(),
                Sample(),
            ],
            compile=True,
            input_type=TensorType.LOGITS,
        )
        counter_compiled = torch.zeros(vocab_size, dtype=torch.int32, device="cuda:0")
        samples_compiled = pipe_compiled(
            logits,
            indices=torch.zeros(num_trials, dtype=torch.int32, device="cuda:0"),
            top_k=k,
            top_p=p,
        )
        counter_compiled.scatter_add_(
            0, samples_compiled.long(), torch.ones_like(samples_compiled)
        )
        freq_compiled = counter_compiled.float() / num_trials

        pipe_no_compile = LogitsPipe(
            [
                Softmax(),
                TopK(),
                TopP(),
                Sample(),
            ],
            compile=False,
            input_type=TensorType.LOGITS,
        )
        samples_no_compile = pipe_no_compile(
            logits,
            indices=torch.zeros(num_trials, dtype=torch.int32, device="cuda:0"),
            top_k=k,
            top_p=p,
        )

        # check if the top-k-top-p thresholding is properly applied
        assert torch.all(mask[torch.arange(1), samples_compiled] == 1)
        assert torch.all(mask[torch.arange(1), samples_no_compile] == 1)

        # check if the sampled results follow given distribution
        # we don't check the non-compiled results because joint topk-topp yeilds different results from topk then topp
        # same for the compile-non-compile similarity as well
        similarity_compiled = torch.cosine_similarity(freq_compiled, masked_probs)
        assert similarity_compiled > 0.99, f"Compiled similarity: {similarity_compiled}"


class TestLogitsPipeVsSamplingOps:
    """Test LogitsPipe implementations against direct sampling operations"""

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize("temperature", [1.0, 0.5, 0.1])
    @pytest.mark.parametrize("temperature_arr", [True, False])
    def test_temperature_softmax(
        self, batch_size, vocab_size, temperature, temperature_arr
    ):
        set_random_seed(42)

        logits = torch.randn(batch_size, vocab_size, device="cuda:0")

        if temperature_arr:
            temperature = torch.rand(batch_size, device="cuda:0")

        samples_direct = flashinfer.sampling.softmax(
            logits=logits, temperature=temperature
        )

        pipe = LogitsPipe([Temperature(), Softmax()])
        samples_pipe = pipe(logits, temperature=temperature)

        assert torch.allclose(samples_pipe, samples_direct, atol=1e-5)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    def test_topp(self, batch_size, vocab_size, p):
        set_random_seed(42)

        probs = torch.rand(batch_size, vocab_size, device="cuda:0")
        probs = probs / probs.sum(dim=-1, keepdim=True)

        samples_direct = flashinfer.sampling.top_p_renorm_probs(probs, p)

        pipe = LogitsPipe([TopP()])
        samples_pipe = pipe(probs, top_p=p)

        assert torch.all(samples_pipe == samples_direct)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize("k", [10, 100, 500])
    def test_probs_topk(self, batch_size, vocab_size, k):
        set_random_seed(42)

        probs = torch.rand(batch_size, vocab_size, device="cuda:0")
        probs = probs / probs.sum(dim=-1, keepdim=True)

        samples_direct = flashinfer.sampling.top_k_renorm_probs(probs, k)

        pipe = LogitsPipe([TopK()], input_type=TensorType.PROBS)
        samples_pipe = pipe(probs, top_k=k)

        assert torch.all(samples_pipe == samples_direct)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize("k", [10, 100, 500])
    @pytest.mark.parametrize("neginf_input", [False, True])
    def test_logits_topk(self, batch_size, vocab_size, k, neginf_input):
        if k > vocab_size:
            pytest.skip("k should be less than vocab_size")

        set_random_seed(42)

        logits = torch.randn(batch_size, vocab_size, device="cuda:0")

        if neginf_input:
            num_neginf = torch.randint(1, vocab_size * batch_size, (1,)).item()
            idxs = torch.randperm(batch_size * vocab_size, device="cuda:0")[:num_neginf]
            logits[idxs // vocab_size, idxs % vocab_size] = -float("inf")

        samples_direct = flashinfer.sampling.top_k_mask_logits(logits, k)

        pipe = LogitsPipe([TopK()], input_type=TensorType.LOGITS)
        samples_pipe = pipe(logits, top_k=k)

        assert torch.all(samples_pipe == samples_direct)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    def test_probs_sample(self, batch_size, vocab_size):
        set_random_seed(42)

        probs = torch.rand(batch_size, vocab_size, device="cuda:0")
        probs = probs / probs.sum(dim=-1, keepdim=True)

        gen1, gen2 = get_generators()

        samples_direct = flashinfer.sampling.sampling_from_probs(probs, generator=gen1)

        pipe = LogitsPipe([Sample()], input_type=TensorType.PROBS)
        samples_pipe = pipe(probs, generator=gen2)

        assert torch.all(samples_pipe == samples_direct)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    def test_logits_sample(self, batch_size, vocab_size):
        set_random_seed(42)

        logits = torch.randn(batch_size, vocab_size, device="cuda:0")

        gen1, gen2 = get_generators()

        samples_direct = flashinfer.sampling.sampling_from_logits(
            logits, generator=gen1
        )

        pipe = LogitsPipe([Sample()], input_type=TensorType.LOGITS)
        samples_pipe = pipe(logits, generator=gen2)

        assert torch.all(samples_pipe == samples_direct)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize("k", [10, 100, 500])
    def test_probs_topk_sample(self, batch_size, vocab_size, k):
        if k > vocab_size:
            pytest.skip("k should be less than vocab_size")

        set_random_seed(42)

        probs = torch.rand(batch_size, vocab_size, device="cuda:0")
        probs = probs / probs.sum(dim=-1, keepdim=True)

        gen1, gen2 = get_generators()

        samples_direct = flashinfer.sampling.top_k_sampling_from_probs(
            probs, k, generator=gen1
        )

        pipe = LogitsPipe([TopK(), Sample()], input_type=TensorType.PROBS)
        samples_pipe = pipe(probs, top_k=k, generator=gen2)

        assert torch.all(samples_pipe == samples_direct)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    def test_probs_topp_sample(self, batch_size, vocab_size, p):
        set_random_seed(42)

        probs = torch.rand(batch_size, vocab_size, device="cuda:0")
        probs = probs / probs.sum(dim=-1, keepdim=True)

        gen1, gen2 = get_generators()

        samples_direct = flashinfer.sampling.top_p_sampling_from_probs(
            probs, p, generator=gen1
        )

        pipe = LogitsPipe([TopP(), Sample()])
        samples_pipe = pipe(probs, top_p=p, generator=gen2)

        assert torch.all(samples_pipe == samples_direct)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize("p", [0.05, 0.1, 0.2, 0.7, 1])
    def test_probs_minp_sample(self, batch_size, vocab_size, p):
        set_random_seed(42)

        probs = torch.rand(batch_size, vocab_size, device="cuda:0")
        probs = probs / probs.sum(dim=-1, keepdim=True)

        gen1, gen2 = get_generators()

        samples_direct = flashinfer.sampling.min_p_sampling_from_probs(
            probs, p, generator=gen1
        )

        pipe = LogitsPipe([MinP(), Sample()])
        samples_pipe = pipe(probs, min_p=p, generator=gen2)

        assert torch.all(samples_pipe == samples_direct)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize("p", [0.1, 0.5])
    def test_joint_probs_topk_topp_sample(self, batch_size, vocab_size, p):
        set_random_seed(42)

        if p == 0.1:
            k = int(vocab_size * 0.5)
        elif p == 0.5:
            k = int(vocab_size * 0.1)
        else:
            raise ValueError("p not recognized")

        probs = torch.rand(batch_size, vocab_size, device="cuda:0")
        probs = probs / probs.sum(dim=-1, keepdim=True)

        gen1, gen2 = get_generators()

        samples_direct = flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs, k, p, filter_apply_order="joint", generator=gen1
        )

        pipe = LogitsPipe(
            [TopK(joint_topk_topp=True), TopP(), Sample()], input_type=TensorType.PROBS
        )

        samples_pipe = pipe(probs, top_k=k, top_p=p, generator=gen2)

        assert torch.all(samples_pipe == samples_direct)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize("p", [0.1, 0.5])
    def test_sequential_probs_topk_topp_sample(self, batch_size, vocab_size, p):
        set_random_seed(42)

        if p == 0.1:
            k = int(vocab_size * 0.5)
        elif p == 0.5:
            k = int(vocab_size * 0.1)
        else:
            raise ValueError("p not recognized")

        probs = torch.rand(batch_size, vocab_size, device="cuda:0")
        probs = probs / probs.sum(dim=-1, keepdim=True)

        gen1, gen2 = get_generators()

        samples_direct = flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs, k, p, filter_apply_order="top_k_first", generator=gen1
        )

        pipe = LogitsPipe([TopK(), TopP(), Sample()], input_type=TensorType.PROBS)
        samples_pipe = pipe(probs, top_k=k, top_p=p, generator=gen2)

        assert torch.all(samples_pipe == samples_direct)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize("p", [0.1, 0.5])
    def test_joint_logits_topk_topp_sample(self, batch_size, vocab_size, p):
        set_random_seed(42)

        if p == 0.1:
            k = int(vocab_size * 0.5)
        elif p == 0.5:
            k = int(vocab_size * 0.1)
        else:
            raise ValueError("p not recognized")

        logits = torch.randn(batch_size, vocab_size, device="cuda:0")

        gen1, gen2 = get_generators()

        samples_direct = flashinfer.sampling.top_k_top_p_sampling_from_logits(
            logits, k, p, filter_apply_order="joint", generator=gen1
        )

        pipe = LogitsPipe(
            [Softmax(), TopK(joint_topk_topp=True), TopP(), Sample()],
            input_type=TensorType.LOGITS,
        )
        samples_pipe = pipe(logits, top_k=k, top_p=p, generator=gen2)

        assert torch.all(samples_pipe == samples_direct)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize("p", [0.1, 0.5])
    def test_sequential_logits_topk_topp_sample(self, batch_size, vocab_size, p):
        set_random_seed(42)

        if p == 0.1:
            k = int(vocab_size * 0.5)
        elif p == 0.5:
            k = int(vocab_size * 0.1)
        else:
            raise ValueError("p not recognized")

        logits = torch.randn(batch_size, vocab_size, device="cuda:0")

        gen1, gen2 = get_generators()

        samples_direct = flashinfer.sampling.top_k_top_p_sampling_from_logits(
            logits, k, p, filter_apply_order="top_k_first", generator=gen1
        )

        topk_mask_pipe = LogitsPipe([TopK()], input_type=TensorType.LOGITS)
        topp_pipe = LogitsPipe([Softmax(), TopP(), Sample()])

        samples_pipe = topp_pipe(
            topk_mask_pipe(logits, top_k=k), top_p=p, generator=gen2
        )

        assert torch.all(samples_pipe == samples_direct)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
