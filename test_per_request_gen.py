"""Test per-request generator support for sampling_from_probs"""
import torch
import flashinfer

def test_per_request_generators():
    """Test that per-request generators work correctly"""
    device = torch.device("cuda:0")
    batch_size = 8
    vocab_size = 1000

    # Create test probabilities
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device=device)
    probs = torch.softmax(logits, dim=-1)

    # Create per-request generators
    seed_arr = torch.randint(0, 2**32, (batch_size,), dtype=torch.int64, device=device)
    offset_arr = torch.zeros(batch_size, dtype=torch.int64, device=device)

    print("Testing per-request generators...")
    print(f"Initial offset_arr: {offset_arr.tolist()}")

    # Sample with per-request generators
    samples_1 = flashinfer.sampling.sampling_from_probs(
        probs,
        generator=(seed_arr, offset_arr)
    )

    print(f"After sampling offset_arr: {offset_arr.tolist()}")
    print(f"Samples shape: {samples_1.shape}")
    print(f"Samples: {samples_1.tolist()}")

    # Verify offsets were updated
    assert torch.all(offset_arr == 4), f"Expected all offsets to be 4, got {offset_arr.tolist()}"
    print("✓ Offsets correctly updated to 4")

    # Test reproducibility: same seeds should give same samples
    seed_arr_copy = seed_arr.clone()
    offset_arr_copy = torch.zeros(batch_size, dtype=torch.int64, device=device)

    samples_2 = flashinfer.sampling.sampling_from_probs(
        probs,
        generator=(seed_arr_copy, offset_arr_copy)
    )

    assert torch.all(samples_1 == samples_2), "Same seeds should produce same samples"
    print("✓ Reproducibility verified: same seeds → same samples")

    # Test that different seeds give different samples
    seed_arr_diff = torch.randint(1000, 2000, (batch_size,), dtype=torch.int64, device=device)
    offset_arr_diff = torch.zeros(batch_size, dtype=torch.int64, device=device)

    samples_3 = flashinfer.sampling.sampling_from_probs(
        probs,
        generator=(seed_arr_diff, offset_arr_diff)
    )

    # At least some samples should differ (very high probability)
    assert not torch.all(samples_1 == samples_3), "Different seeds should produce different samples"
    print("✓ Different seeds → different samples")

    # Test backward compatibility: traditional generator still works
    gen = torch.Generator(device=device)
    gen.manual_seed(42)

    samples_trad = flashinfer.sampling.sampling_from_probs(
        probs,
        generator=gen
    )

    print(f"Traditional generator samples shape: {samples_trad.shape}")
    print("✓ Backward compatibility: traditional torch.Generator works")

    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_per_request_generators()
