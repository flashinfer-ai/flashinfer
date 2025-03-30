// adapted from https://github.com/efeslab/Nanoflow/blob/d6b381e58110a8b5d08cfabd4a55c0d5d0ebef57/pipeline/src/comm_test.cu
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "netWrapper.cuh"
#include "tensorLogger.cuh"

void test_NetAllGather(std::shared_ptr<mscclpp::Communicator> comm,
                       std::vector<std::shared_ptr<mscclpp::Connection>>& connections,
                       const int rank, const int nranks,
                       const size_t buff_size, const bool inplace, const bool columnwise) {
    // Intialize host and device buffers
    std::vector<cutlass::half_t> host_buff(buff_size / sizeof(cutlass::half_t));
    for (size_t i = 0; i < host_buff.size(); ++i) host_buff[i] = cutlass::half_t(int((i * rank) % 101));
    void* input_buff;
    CUDA_CHECK(cudaMalloc(&input_buff, buff_size));
    CUDA_CHECK(cudaMemcpy(input_buff, host_buff.data(), buff_size, cudaMemcpyHostToDevice));
    void* output_buff;
    if (inplace) {
        output_buff = input_buff;
    } else {
        CUDA_CHECK(cudaMalloc(&output_buff, buff_size));
    }

    // Initialize NetWrapper
    NetAllGather wrapper;
    int dim1, input_dim2, output_dim2;
    if (columnwise) {
        const size_t alignment = sizeof(int4) / sizeof(cutlass::half_t) * nranks;
        dim1 = ((int) sqrt(buff_size / sizeof(cutlass::half_t))) / alignment * alignment;
        input_dim2 = dim1 / nranks;
        output_dim2 = dim1;
    } else {
        dim1 = buff_size / sizeof(cutlass::half_t);
        input_dim2 = 1;
        output_dim2 = 1;
    }
	wrapper.init(comm,
				 connections,
				 rank,
				 nranks,
				 pllmTensor<cutlass::half_t>{
                    (cutlass::half_t*)input_buff, dim1, input_dim2, PllmLayout::ROW_MAJOR},
				 pllmTensor<cutlass::half_t>{
                    (cutlass::half_t*)output_buff, dim1, output_dim2, PllmLayout::ROW_MAJOR});

	MPI_Barrier(MPI_COMM_WORLD);
    wrapper.setColumnwise(columnwise);
    wrapper(0, 32, 1024, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check allgather correctness
    const size_t nelem_per_shard = host_buff.size() / nranks;
    CUDA_CHECK(cudaMemcpy(host_buff.data(), output_buff, buff_size, cudaMemcpyDeviceToHost));
    if (columnwise) {
        bool correct = true;
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < output_dim2; ++j) {
                const int remoteRank = (int) (j / input_dim2);
                cutlass::half_t expected = cutlass::half_t(int(((i * input_dim2 + j % input_dim2) * remoteRank) % 101));
                if (host_buff[i * output_dim2 + j] != expected) {
                    std::cerr << "Rank " << rank << " received incorrect data from rank " << remoteRank
                              << " at index (" << i << "," << j << ")" << std::endl;
                    correct = false;
                    break;
                }
            }
            if (!correct) break;
        }
    } else {
        for (size_t i = 0; i < host_buff.size(); ++i) {
            const int remoteRank = i / nelem_per_shard;
            cutlass::half_t expected = cutlass::half_t(int((i * remoteRank) % 101));
            if (host_buff[i] != expected) {
                std::cerr << "Rank " << rank << " received incorrect data from rank " << remoteRank << " at index " << i << std::endl;
                break;
            }
        }
    }

    CUDA_CHECK(cudaFree(input_buff));
    if (input_buff != output_buff) CUDA_CHECK(cudaFree(output_buff));
    std::cout << "Rank " << rank << " NetAllGather test ("
              << "buff_size=" << buff_size << ",inplace=" << inplace
              << ",columnwise=" << columnwise << ") finished" << std::endl;
}

void test_NetReduceScatter(std::shared_ptr<mscclpp::Communicator> comm,
                           std::vector<std::shared_ptr<mscclpp::Connection>>& connections,
                           const int rank, const int nranks,
                           const size_t buff_size, const bool inplace) {
    // Intialize host and device buffers
    std::vector<cutlass::half_t> host_buff(buff_size / sizeof(cutlass::half_t));
    for (size_t i = 0; i < host_buff.size(); ++i) host_buff[i] = cutlass::half_t(int((i * rank) % 101));
    void* input_buff;
    CUDA_CHECK(cudaMalloc(&input_buff, buff_size));
    CUDA_CHECK(cudaMemcpy(input_buff, host_buff.data(), buff_size, cudaMemcpyHostToDevice));
    void* output_buff;
    if (inplace) {
        output_buff = input_buff;
    } else {
        CUDA_CHECK(cudaMalloc(&output_buff, buff_size));
    }

    // Initialize NetWrapper
    NetReduceScatter wrapper;
    wrapper.init(
        comm, connections, rank, nranks, 
        (cutlass::half_t*) input_buff, (cutlass::half_t*) output_buff, 
        (int) (buff_size / sizeof(cutlass::half_t)), (int) (buff_size / sizeof(cutlass::half_t)));

    MPI_Barrier(MPI_COMM_WORLD);

    wrapper(0, 32, 1024, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check reduce-scatter correctness
    const size_t nelem_per_shard = host_buff.size() / nranks;
    CUDA_CHECK(cudaMemcpy(host_buff.data(), output_buff, buff_size, cudaMemcpyDeviceToHost));
    for (size_t i = rank * nelem_per_shard; i < (rank + 1) * nelem_per_shard; ++i) {
        cutlass::half_t expected = cutlass::half_t(0);
        for (int j = 0; j < nranks; ++j) expected += cutlass::half_t(int((i * j) % 101));
        if (host_buff[i] != expected) {
            std::cerr << "Rank " << rank << " incorrect data at index " << i << std::endl;
            break;
        }
    }

    CUDA_CHECK(cudaFree(input_buff));
    if (input_buff != output_buff) CUDA_CHECK(cudaFree(output_buff));
    std::cout << "Rank " << rank << " NetReduceScatter test ("
              << "buff_size=" << buff_size << ",inplace=" << inplace
              << ") finished" << std::endl;
}

void test_NetAllReduce(std::shared_ptr<mscclpp::Communicator> comm,
                       std::vector<std::shared_ptr<mscclpp::Connection>>& connections,
                       const int rank, const int nranks,
                       const size_t buff_size, const bool inplace) {
    // Intialize host and device buffers
    std::vector<cutlass::half_t> host_buff(buff_size / sizeof(cutlass::half_t));
    for (size_t i = 0; i < host_buff.size(); ++i) host_buff[i] = cutlass::half_t(int((i * rank) % 101));
    void* input_buff;
    CUDA_CHECK(cudaMalloc(&input_buff, buff_size));
    CUDA_CHECK(cudaMemcpy(input_buff, host_buff.data(), buff_size, cudaMemcpyHostToDevice));
    void* output_buff;
    if (inplace) {
        output_buff = input_buff;
    } else {
        CUDA_CHECK(cudaMalloc(&output_buff, buff_size));
    }

    // Initialize NetWrapper
    NetAllReduce wrapper;
	wrapper.init(comm,
				 connections,
				 rank,
				 nranks,
				 pllmTensor<cutlass::half_t>((cutlass::half_t*)input_buff,
											buff_size / sizeof(cutlass::half_t), 1, PllmLayout::ROW_MAJOR),
				 pllmTensor<cutlass::half_t>((cutlass::half_t*)output_buff,
											buff_size / sizeof(cutlass::half_t), 1, PllmLayout::ROW_MAJOR));

	MPI_Barrier(MPI_COMM_WORLD);

    wrapper(0, 32, 1024, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check allreduce correctness
    CUDA_CHECK(cudaMemcpy(host_buff.data(), output_buff, buff_size, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < host_buff.size(); ++i) {
        cutlass::half_t expected = cutlass::half_t(0);
        for (int j = 0; j < nranks; ++j) expected += cutlass::half_t(int((i * j) % 101));
        if (host_buff[i] != expected) {
            std::cerr << "Rank " << rank << " incorrect data at index " << i << " with value " << host_buff[i] << std::endl;
            break;
        }
    }

    CUDA_CHECK(cudaFree(input_buff));
    if (input_buff != output_buff) CUDA_CHECK(cudaFree(output_buff));
    std::cout << "Rank " << rank << " NetAllReduce test ("
              << "buff_size=" << buff_size << ",inplace=" << inplace
              << ") finished" << std::endl;
}

void test_NetAllGatherAsync(std::shared_ptr<mscclpp::Communicator> comm,
                            std::vector<std::shared_ptr<mscclpp::Connection>>& connections,
                            const int rank, const int nranks,
                            const size_t buff_size, const bool inplace) {
    // Intialize host and device buffers
    std::vector<cutlass::half_t> host_buff(buff_size / sizeof(cutlass::half_t));
    for (size_t i = 0; i < host_buff.size(); ++i) host_buff[i] = cutlass::half_t(int((i * rank) % 101));
    void* input_buff;
    CUDA_CHECK(cudaMalloc(&input_buff, buff_size));
    CUDA_CHECK(cudaMemcpy(input_buff, host_buff.data(), buff_size, cudaMemcpyHostToDevice));
    void* output_buff;
    if (inplace) {
        output_buff = input_buff;
    } else {
        CUDA_CHECK(cudaMalloc(&output_buff, buff_size));
    }

    // Initialize NetWrapper
    NetAllGatherAsync wrapper;
    wrapper.init(
        comm, connections, rank, nranks, 
        (cutlass::half_t*) input_buff, (cutlass::half_t*) output_buff, 
        (int) (buff_size / sizeof(cutlass::half_t)), (int) (buff_size / sizeof(cutlass::half_t)));

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    wrapper.start(0, 1, nranks);
    wrapper.finish(0, 1, nranks);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check allgather correctness
    const size_t nelem_per_shard = host_buff.size() / nranks;
    CUDA_CHECK(cudaMemcpy(host_buff.data(), output_buff, buff_size, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < host_buff.size(); ++i) {
        const int remoteRank = i / nelem_per_shard;
        cutlass::half_t expected = cutlass::half_t(int((i * remoteRank) % 101));
        if (host_buff[i] != expected) {
            std::cerr << "Rank " << rank << " received incorrect data from rank " << remoteRank << " at index " << i << std::endl;
            break;
        }
    }

    CUDA_CHECK(cudaFree(input_buff));
    if (input_buff != output_buff) CUDA_CHECK(cudaFree(output_buff));
    std::cout << "Rank " << rank << " NetAllGatherAsync test ("
              << "buff_size=" << buff_size << ",inplace=" << inplace
              << ") finished" << std::endl;
}

void test_NetReduceScatterAsync(std::shared_ptr<mscclpp::Communicator> comm,
                                std::vector<std::shared_ptr<mscclpp::Connection>>& connections,
                                const int rank, const int nranks,
                                const size_t buff_size, const bool inplace) {
    // Intialize host and device buffers
    std::vector<cutlass::half_t> host_buff(buff_size / sizeof(cutlass::half_t));
    for (size_t i = 0; i < host_buff.size(); ++i) host_buff[i] = cutlass::half_t(int((i * rank) % 101));
    void* input_buff;
    CUDA_CHECK(cudaMalloc(&input_buff, buff_size));
    CUDA_CHECK(cudaMemcpy(input_buff, host_buff.data(), buff_size, cudaMemcpyHostToDevice));
    void* output_buff;
    if (inplace) {
        output_buff = input_buff;
    } else {
        CUDA_CHECK(cudaMalloc(&output_buff, buff_size));
    }

    // Initialize NetWrapper
    NetReduceScatterAsync wrapper;
    wrapper.init(
        comm, connections, rank, nranks, 
        (cutlass::half_t*) input_buff, (cutlass::half_t*) output_buff, 
        (int) (buff_size / sizeof(cutlass::half_t)), (int) (buff_size / sizeof(cutlass::half_t)));

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    wrapper.start(0, 1, nranks);
    wrapper.finish(0, 32, 1024);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check reduce-scatter correctness
    const size_t nelem_per_shard = host_buff.size() / nranks;
    CUDA_CHECK(cudaMemcpy(host_buff.data(), output_buff, buff_size, cudaMemcpyDeviceToHost));
    for (size_t i = rank * nelem_per_shard; i < (rank + 1) * nelem_per_shard; ++i) {
        cutlass::half_t expected = cutlass::half_t(0);
        for (int j = 0; j < nranks; ++j) expected += cutlass::half_t(int((i * j) % 101));
        if (host_buff[i] != expected) {
            std::cerr << "Rank " << rank << " incorrect data at index " << i << std::endl;
            break;
        }
    }

    CUDA_CHECK(cudaFree(input_buff));
    if (input_buff != output_buff) CUDA_CHECK(cudaFree(output_buff));
    std::cout << "Rank " << rank << " NetReduceScatterAsync test ("
              << "buff_size=" << buff_size << ",inplace=" << inplace
              << ") finished" << std::endl;
}

int main(int argc, char* argv[]) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int nranks;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    CUDA_CHECK(cudaSetDevice(rank));

    // Print off a hello world message
    std::cout << "Hello world from rank " << rank << " out of " << nranks << " ranks" << std::endl;

    // Initialize Communicator
    auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, nranks);
    mscclpp::UniqueId uniqueId;
    if (rank == 0) uniqueId = bootstrap->createUniqueId();
    MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    bootstrap->initialize(uniqueId);
    auto comm = std::make_shared<mscclpp::Communicator>(bootstrap);

    // Initialize Connections
    std::vector<std::shared_ptr<mscclpp::Connection>> connections;
    std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionFutures;
    for (int r = 0; r < nranks; ++r) {
        if (r == rank) continue;
        mscclpp::Transport transport = mscclpp::Transport::CudaIpc;
        connectionFutures.push_back(comm->connectOnSetup(r, 0, transport));
    }
    comm->setup();
    std::transform(
        connectionFutures.begin(), connectionFutures.end(), std::back_inserter(connections),
        [](const mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>& future) { return future.get(); });

    MPI_Barrier(MPI_COMM_WORLD);

    // Tests
    constexpr size_t buff_size = 16 * 1024 * 1024 + 4 * 8;
    test_NetAllGather(comm, connections, rank, nranks, buff_size, true, false);
    test_NetAllGather(comm, connections, rank, nranks, buff_size, false, false);
    test_NetAllGather(comm, connections, rank, nranks, buff_size, false, true); // columnwise cannot inplace
    test_NetReduceScatter(comm, connections, rank, nranks, buff_size, true);
    test_NetReduceScatter(comm, connections, rank, nranks, buff_size, false);
    test_NetAllReduce(comm, connections, rank, nranks, buff_size, true);
    test_NetAllReduce(comm, connections, rank, nranks, buff_size, false);

    test_NetAllGatherAsync(comm, connections, rank, nranks, buff_size, true);
    test_NetAllGatherAsync(comm, connections, rank, nranks, buff_size, false);
    test_NetReduceScatterAsync(comm, connections, rank, nranks, buff_size, true);
    test_NetReduceScatterAsync(comm, connections, rank, nranks, buff_size, false);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    

    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}