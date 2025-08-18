import torch


# ref: DeepGEMM
def enumerate_m_grouped_masked():
    max_m = 4096
    for num_groups, m in (
        # GB200 cases
        (6, 1024),
        # DeepGEMM default cases
        (1, 1024), (2, 512), (4, 256),
    ):
        for n, k in ((4096, 7168), (7168, 2048),):
            yield dict(num_groups=num_groups, max_m=max_m, m=m, n=n, k=k)


def bench_one(num_groups, max_m, m, n, k):
    TODO


if __name__ == "__main__":
    torch.manual_seed(42)
    for config in enumerate_m_grouped_masked():
        bench_one(**config)
