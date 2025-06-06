import flashinfer.comm as comm


def test_nvshmem():
    comm.get_nvshmem_module()


if __name__ == "__main__":
    test_nvshmem()
