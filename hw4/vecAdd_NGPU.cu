#include <cstdio>
#include <cstdlib>
#include <omp.h>          // header for OpenMP
#include <cuda_runtime.h>

void RandomInit(float *data, int n);

__global__ void VecAdd(const float *A, const float *B, float *C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = 1.0 / A[i] + 1.0 / B[i];

    __syncthreads();
}

int main() {

    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Size (MB): %d\n", prop.totalGlobalMem / 1024 / 1024);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  Max Threads Dim: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1],
               prop.maxThreadsDim[2]);
        printf("  Max Grid Size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1],
               prop.maxGridSize[2]);
        printf(" Max Shared Memory Per Block: %d \n\n", prop.sharedMemPerBlock);

    }

    float *A_h;
    float *B_h;
    float *C_h;
    float *C_ref;

    int N;
    printf("Input the size of vector:");
    scanf("%d", &N);

    int nGPU;
    printf("Input the number of gpus:");
    scanf("%d", &nGPU);

    int threadsPerBlock;
    printf("Input the number of threads per block:");
    scanf("%d", &threadsPerBlock);
    int blocksPerGrid = (N + threadsPerBlock * nGPU - 1) / (threadsPerBlock * nGPU);

    int size = N * sizeof(float);
    A_h = (float *) malloc(size);
    B_h = (float *) malloc(size);
    C_h = (float *) malloc(size);

    RandomInit(A_h, N);
    RandomInit(B_h, N);

    float InTime, OutTime, PTime, CPUTime;

#pragma omp parallel default(none) \
    shared(N, nGPU, size, A_h, B_h, C_h, blocksPerGrid, threadsPerBlock, InTime, OutTime, PTime) \
    num_threads(nGPU)
    {
        int id = omp_get_thread_num();
        cudaError_t isate = cudaSetDevice(id);
        printf("Setting Device: %d (cudaError_t: %5d)\n", id, isate);
        float *A, *B, *C;
        cudaEvent_t start, stop;
        if (id == 0) {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
        }
        cudaMalloc((void **) &A, size / nGPU);
        cudaMalloc((void **) &B, size / nGPU);
        cudaMalloc((void **) &C, size / nGPU);

        cudaMemcpy(A, A_h + N / nGPU * id, size / nGPU, cudaMemcpyHostToDevice);
        cudaMemcpy(B, B_h + N / nGPU * id, size / nGPU, cudaMemcpyHostToDevice);

#pragma omp barrier

        if (id == 0) {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&InTime, start, stop);
            printf("Data input time for GPU: %f (ms) \n", InTime);
            cudaEventRecord(start, 0);
        }

        VecAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N / nGPU);
        cudaDeviceSynchronize();

        if (id == 0) {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&PTime, start, stop);
            printf("Processing time for GPU: %f (ms) \n", PTime);
            cudaEventRecord(start, 0);
        }

        cudaMemcpy(C_h + N / nGPU * id, C, size / nGPU, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        if (id == 0) {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&OutTime, start, stop);
            printf("Data output time for GPU: %f (ms) \n", OutTime);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        cudaFree(A);
        cudaFree(B);
        cudaFree(C);

    }

    cudaEvent_t Cstart, Cstop;
    cudaEventCreate(&Cstart);
    cudaEventCreate(&Cstop);
    cudaEventRecord(Cstart, 0);
    C_ref = (float *) malloc(size);
    for (int i = 0; i < N; i++) {
        C_ref[i] = 1.0 / A_h[i] + 1.0 / B_h[i];
    }
    cudaEventRecord(Cstop, 0);
    cudaEventSynchronize(Cstop);
    cudaEventElapsedTime(&CPUTime, Cstart, Cstop);
    printf("Processing time for CPU: %f (ms) \n", CPUTime);

    double error = 0.0;
    for (int i = 0; i < N; i++) {
        //printf("%.4e\n", C_h[i]);
        error += abs(C_ref[i] - C_h[i]);
    }
    error = sqrt(error);
    printf("norm(h_C - h_D)= %20.15e \n", error);
    return 0;
}

void RandomInit(float *data, int n) {
    for (int i = 0; i < n; ++i)
        data[i] = rand() / (float) RAND_MAX;
}
