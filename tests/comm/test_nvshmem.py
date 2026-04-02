import flashinfer.comm.nvshmem as fi_nvshmem


def test_nvshmem_1_gpu() -> None:
    uid = fi_nvshmem.get_unique_id()
    fi_nvshmem.init(uid, 0, 1)
    assert fi_nvshmem.my_pe() == 0
    assert fi_nvshmem.n_pes() == 1
    fi_nvshmem.finalize()


def test_nvshmem():
    """Verify that nvshmem4py can be imported and basic APIs are accessible."""
    import nvshmem.core

    uid = nvshmem.core.get_unique_id(empty=True)
    assert uid is not None


if __name__ == "__main__":
    test_nvshmem()
