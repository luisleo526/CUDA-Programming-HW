#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <float.h>

long NUM = 81920000;

void init_data(double *x, double *max_value);


__global__ void hist_gmem(double *data, const long N, unsigned int *hist, const double binsize) {

    long i = threadIdx.x + blockIdx.x * blockDim.x;
    long stride = blockDim.x * gridDim.x;

    while (i < N) {
        int index = (int) (data[i] / binsize);
        atomicAdd(&hist[index], 1);
        i += stride;
    }

    __syncthreads();

}

__global__ void hist_shmem(double *data, const long N, unsigned int *hist, const double binsize) {

    extern __shared__  unsigned int temp[];
    temp[threadIdx.x] = 0;
    __syncthreads();

    long i = threadIdx.x + blockIdx.x * blockDim.x;
    long stride = blockDim.x * gridDim.x;

    while (i < N) {
        int index = (int) (data[i] / binsize);
//        atomicAdd(&hist[index], 1);
        atomicAdd(&temp[index], 1);
        i += stride;
    }

    __syncthreads();
    atomicAdd(&(hist[threadIdx.x]), temp[threadIdx.x]);

}


int main() {

    double *dat, *dat_d;
    unsigned int *hist, *hist_d, *hist_ref;
    double max_value;

    int memtype;
    printf("Global / Shared memory (0/1): \n");
    scanf("%d", &memtype);

    int threadsPerBlock;
    printf("Enter the number of threads per block: \n");
    scanf("%d", &threadsPerBlock);

    int blocksPerGrid = ((int) NUM + threadsPerBlock - 1) / threadsPerBlock;

//    int blocksPerGrid;
//    printf("Enter the number of blocks per grid: ");
//    scanf("%d", &blocksPerGrid);

    int num_bins;
    if (memtype == 0) {
        printf("Input the number of bins: \n");
        scanf("%d", &num_bins);
    } else {
        num_bins = threadsPerBlock;
    }

    if (blocksPerGrid > 2147483647) {
        printf("The number of blocks must be less than 2147483647 ! \n");
        exit(0);
    }

    dat = (double *) malloc(NUM * sizeof(double));
    hist = (unsigned int *) malloc(num_bins * sizeof(unsigned int));
    hist_ref = (unsigned int *) malloc(num_bins * sizeof(unsigned int));

    init_data(dat, &max_value);
    double bin_size = max_value / (double) num_bins;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (long i = 0; i < NUM; i++) {
        long index = (long) (dat[i] / bin_size);
        hist[index] += 1;
    }

    printf("---------------------------------------\n");
    printf("Threads per block: %d\n", threadsPerBlock);
    printf("Blocks per grid: %d\n", blocksPerGrid);
    if (memtype == 0) {
        printf("Memory type: Global memory\n");
    } else {
        printf("Memory type: Shared memory\n");
    }
    printf("---------------------------------------\n");

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Processing time of CPU: %f (ms)\n", elapsedTime);

    cudaMalloc((void **) &dat_d, NUM * sizeof(double));
    cudaMalloc((void **) &hist_d, num_bins * sizeof(unsigned int));
    cudaMemcpy(dat_d, dat, NUM * sizeof(double), cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);

    if (memtype == 0) {
        hist_gmem<<<blocksPerGrid, threadsPerBlock >>>(dat_d, NUM, hist_d, bin_size);
    } else {
        int shm_size = threadsPerBlock * sizeof(unsigned int);
        hist_shmem<<<blocksPerGrid, threadsPerBlock, shm_size>>>(dat_d, NUM, hist_d, bin_size);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Processing time of GPU: %f (ms)\n", elapsedTime);


    cudaMemcpy(hist_ref, hist_d, num_bins * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    FILE *out1;
    out1 = fopen("histogram.plt", "w");
    fprintf(out1, "Variables = \"x\" \"cpu\" \"gpu\" \"theory\"\n");
    for (int i = 0; i < num_bins; i++) {
        double x = (i + 0.5) * bin_size;
        double ref = exp(-x);
        fprintf(out1, "%f %f %f %f\n", x, (double) hist[i] / (double) NUM, (double) hist_ref[i] / (double) NUM, ref);
    }
    fclose(out1);
}

void init_data(double *x, double *max_value) {

    for (long i = 0; i < NUM; i++) {
        double entry = rand() / (double) RAND_MAX;
        x[i] = -log(1.0 - entry);
        *max_value = (x[i] < *max_value) ? *max_value : x[i];
    }

}
