import flashinfer.comm.nvshmem as nvshmem


def test_nvshmem_1_gpu() -> None:
    uid = nvshmem.get_unique_id()
    nvshmem.init(uid, 0, 1)
    assert nvshmem.my_pe() == 0
    assert nvshmem.n_pes() == 1
    nvshmem.finalize()


def test_nvshmem():
    nvshmem.get_nvshmem_module()


if __name__ == "__main__":
    test_nvshmem()
