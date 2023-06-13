#include <cstdio>
#include <cstdlib>
#include <omp.h>          // header for OpenMP
#include <cuda_runtime.h>

#define OUTPUT "solution_%d.tec"

int GRIDS = 1024;
double TOL = 1e-10;
int ITER_MAX = 10000000;

void plot(float *phi, int id);

void merge(float **phis, float *phi, int nGx, int nGy);

__global__ void laplacian(float *phi_old, float *phi_new, float *buffer,
                          float *phi_R, float *phi_L, float *phi_F, float *phi_B) {

    extern __shared__ float cache[];

    float r, l, f, b;
    float diff;

    int Nx = blockDim.x * gridDim.x;
    int Ny = blockDim.y * gridDim.y;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    int cacheIndex = threadIdx.x + threadIdx.y * blockDim.x;

    int site = i + j * Nx;

    if ((i == 0) || (i == Nx - 1) || (j == 0) || (j == Ny - 1)) {
        bool flag = false;
        // Check X
        if (i == 0) {
            if (phi_L != NULL) {
                r = phi_old[site + 1];
                l = phi_L[Nx - 1 + j * Nx];
            } else {
                flag = true;
            }
        } else if (i == Nx - 1) {
            if (phi_R != NULL) {
                r = phi_R[0 + j * Nx];
                l = phi_old[site - 1];
            } else {
                flag = true;
            }
        } else {
            r = phi_old[site + 1];
            l = phi_old[site - 1];
        }
        // Check Y
        if (j == 0) {
            if (phi_B != NULL) {
                f = phi_old[site + Nx];
                b = phi_B[i + (Ny - 1) * Nx];
            } else {
                flag = true;
            }
        } else if (j == Ny - 1) {
            if (phi_F != NULL) {
                f = phi_F[i];
                b = phi_old[site - Nx];
            } else {
                flag = true;
            }
        } else {
            f = phi_old[site + Nx];
            b = phi_old[site - Nx];
        }

        if (flag) {
            diff = 0.0;
        } else {
            phi_new[site] = (r + l + f + b) / 4.0;
//            phi_new[site] = 1.5 * phi_new[site] - phi_old[site];

            diff = phi_new[site] - phi_old[site];
        }

    } else {

        r = phi_old[site + 1];
        l = phi_old[site - 1];
        f = phi_old[site + Nx];
        b = phi_old[site - Nx];

        phi_new[site] = (r + l + f + b) / 4.0;
//        phi_new[site] = 1.5 * phi_new[site] - phi_old[site];

        diff = phi_new[site] - phi_old[site];
    }

    cache[cacheIndex] = diff * diff;
    __syncthreads();

    int ib = blockDim.x * blockDim.y / 2;
    while (ib != 0) {
        if (cacheIndex < ib)
            cache[cacheIndex] += cache[cacheIndex + ib];
        __syncthreads();
        ib /= 2;
    }
    int blockIndex = blockIdx.x + gridDim.x * blockIdx.y;
    if (cacheIndex == 0) buffer[blockIndex] = cache[0];
    phi_old[site] = phi_new[site];

}

__global__ void laplacian_tex(float *phi_old, float *phi_new, float *buffer, cudaTextureObject_t textCache,
                              cudaTextureObject_t phi_R, cudaTextureObject_t phi_L,
                              cudaTextureObject_t phi_F, cudaTextureObject_t phi_B) {

    extern __shared__ float cache[];

    float r, l, f, b;
    float diff;

    int Nx = blockDim.x * gridDim.x;
    int Ny = blockDim.y * gridDim.y;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    int cacheIndex = threadIdx.x + threadIdx.y * blockDim.x;

    int site = i + j * Nx;

    if ((i == 0) || (i == Nx - 1) || (j == 0) || (j == Ny - 1)) {
        bool flag = false;
        // Check X
        if (i == 0) {
            if (phi_L != 0) {
//                r = phi_old[site + 1];
//                l = phi_L[Nx - 1 + j * Nx];
                r = tex1Dfetch<float>(textCache, site + 1);
                l = tex1Dfetch<float>(phi_L, Nx - 1 + j * Nx);

            } else {
                flag = true;
            }
        } else if (i == Nx - 1) {
            if (phi_R != 0) {
//                r = phi_R[0 + j * Nx];
//                l = phi_old[site - 1];
                r = tex1Dfetch<float>(phi_R, 0 + j * Nx);
                l = tex1Dfetch<float>(textCache, site - 1);
            } else {
                flag = true;
            }
        } else {
//            r = phi_old[site + 1];
//            l = phi_old[site - 1];
            r = tex1Dfetch<float>(textCache, site + 1);
            l = tex1Dfetch<float>(textCache, site - 1);
        }
        // Check Y
        if (j == 0) {
            if (phi_B != 0) {
//                f = phi_old[site + Nx];
//                b = phi_B[i + (Ny - 1) * Nx];
                f = tex1Dfetch<float>(textCache, site + Nx);
                b = tex1Dfetch<float>(phi_B, i + (Ny - 1) * Nx);
            } else {
                flag = true;
            }
        } else if (j == Ny - 1) {
            if (phi_F != 0) {
//                f = phi_F[i];
//                b = phi_old[site - Nx];
                f = tex1Dfetch<float>(phi_F, i);
                b = tex1Dfetch<float>(textCache, site - Nx);
            } else {
                flag = true;
            }
        } else {
//            f = phi_old[site + Nx];
//            b = phi_old[site - Nx];
            f = tex1Dfetch<float>(textCache, site + Nx);
            b = tex1Dfetch<float>(textCache, site - Nx);
        }

        if (flag) {
            diff = 0.0;
        } else {
            phi_new[site] = (r + l + f + b) / 4.0;
//            phi_new[site] = 1.5 * phi_new[site] - phi_old[site];

            diff = phi_new[site] - phi_old[site];
        }

    } else {

        r = tex1Dfetch<float>(textCache, site + 1);
        l = tex1Dfetch<float>(textCache, site - 1);
        f = tex1Dfetch<float>(textCache, site + Nx);
        b = tex1Dfetch<float>(textCache, site - Nx);

        phi_new[site] = (r + l + f + b) / 4.0;
//        phi_new[site] = 1.5 * phi_new[site] - phi_old[site];

        diff = phi_new[site] - phi_old[site];
    }

    cache[cacheIndex] = diff * diff;
    __syncthreads();

    int ib = blockDim.x * blockDim.y / 2;
    while (ib != 0) {
        if (cacheIndex < ib)
            cache[cacheIndex] += cache[cacheIndex + ib];
        __syncthreads();
        ib /= 2;
    }
    int blockIndex = blockIdx.x + gridDim.x * blockIdx.y;
    if (cacheIndex == 0) buffer[blockIndex] = cache[0];
    phi_old[site] = phi_new[site];

}


int main() {

    int nDevices;
    int nGPU, nGPUx, nGPUy;
    int N;
    int threadsPerBlock, blocksPerGrid;

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

    printf("Input the number of gpus (Gx, Gy): ");
    scanf("%d %d", &nGPUx, &nGPUy);
    nGPU = nGPUx * nGPUy;

    printf("Input the number of threads per block: ");
    scanf("%d", &threadsPerBlock);
    blocksPerGrid = GRIDS / threadsPerBlock;

    int useTexture;
    printf("Use texture cache ? (0 / 1)");
    scanf("%d", &useTexture);

    if (blocksPerGrid * threadsPerBlock != GRIDS || blocksPerGrid % nGPUx != 0 || blocksPerGrid % nGPUy != 0) {
        printf("The block size is incorrect\n");
    }

    dim3 griddim = dim3(blocksPerGrid / nGPUx, blocksPerGrid / nGPUy, 1);
    dim3 blockdim = dim3(threadsPerBlock, threadsPerBlock, 1);

    float **phi_host, **buffer_host, *phi_global;
    N = GRIDS * GRIDS;
    int size = N * sizeof(float);
    int buffer_size = blocksPerGrid * blocksPerGrid * sizeof(float);

    phi_host = (float **) malloc(nGPU * sizeof(float *));
    phi_global = (float *) malloc(size);
    buffer_host = (float **) malloc(nGPU * sizeof(float *));
    for (int i = 0; i < nGPU; i++) {
        phi_host[i] = (float *) malloc(size / nGPU);
        buffer_host[i] = (float *) malloc(buffer_size / nGPU);
        for (int j = 0; j < N; j++) {
            phi_host[i][j] = 273.0;
        }
    }

    // -------------------------------------------------------------------------------------
    // Set Boundary Conditions
    for (int gx = 0; gx < nGPUx; gx++) {
        int gy;
        // Top B.C.
        gy = nGPUy - 1;
        for (int i = 0; i < GRIDS / nGPUx; i++) {
            phi_host[gx + gy * nGPUx][i + (GRIDS / nGPUy - 1) * GRIDS / nGPUx] = 400.0;
        }
        // Bottom B.C.
        gy = 0;
        for (int i = 0; i < GRIDS / nGPUx; i++) {
            phi_host[gx + gy * nGPUx][i + 0 * GRIDS / nGPUx] = 273.0;
        }
    }

    for (int gy = 0; gy < nGPUy; gy++) {
        int gx;
        // Right B.C.
        gx = nGPUx - 1;
        for (int j = 0; j < GRIDS / nGPUy; j++) {
            phi_host[gx + gy * nGPUx][(GRIDS / nGPUx - 1) + j * GRIDS / nGPUx] = 273.0;
        }
        // Left B.C.
        gx = 0;
        for (int j = 0; j < GRIDS / nGPUy; j++) {
            phi_host[gx + gy * nGPUx][0 + j * GRIDS / nGPUx] = 273.0;
        }
    }
    // -------------------------------------------------------------------------------------

    // -------------------------------------------------------------------------------------
    // Collect all subdomains to a global domain, and plot
    merge(phi_host, phi_global, nGPUx, nGPUy);
    plot(phi_global, 0);
    // -------------------------------------------------------------------------------------

    cudaEvent_t start, stop;

    printf("Grid Dimension: (%d, %d, %d)\n", griddim.x, griddim.y, griddim.z);
    printf("Block Dimension: (%d, %d, %d)\n", blockdim.x, blockdim.y, blockdim.z);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float InTime, OutTime, PTime, CPUTime;
    float **phi, **phio, **buffer;

    phi = (float **) malloc(nGPU * sizeof(float *));
    phio = (float **) malloc(nGPU * sizeof(float *));
    buffer = (float **) malloc(nGPU * sizeof(float *));
    int buffer_dim = blocksPerGrid * blocksPerGrid / nGPUx / nGPUy;

    // Use texture cache only for phi_old
    cudaTextureObject_t *tex_cache;
    tex_cache = (cudaTextureObject_t *) malloc(nGPU * sizeof(cudaTextureObject_t));

#pragma omp parallel num_threads(nGPU) default(none) shared(phi, phio, buffer, phi_host, buffer_host) \
shared(nGPU, nGPUx, nGPUy, size, buffer_size, buffer_dim, griddim, blockdim, TOL, ITER_MAX, useTexture, tex_cache)
    {
        int id = omp_get_thread_num();
        cudaError_t isate = cudaSetDevice(id);
        printf("Setting Device: %d (cudaError_t: %5d)\n", id, isate);

        cudaMalloc((void **) &phi[id], size / nGPU);
        cudaMalloc((void **) &phio[id], size / nGPU);
        cudaMalloc((void **) &buffer[id], buffer_size / nGPU);

        cudaMemcpy(phi[id], phi_host[id], size / nGPU, cudaMemcpyHostToDevice);
        cudaMemcpy(phio[id], phi_host[id], size / nGPU, cudaMemcpyHostToDevice);

        if (useTexture != 0) {

            struct cudaResourceDesc resDesc;
            struct cudaTextureDesc texDesc;

            memset(&texDesc, 0, sizeof(texDesc));
            memset(&resDesc, 0, sizeof(resDesc));
            resDesc.resType = cudaResourceTypeLinear;
            resDesc.res.linear.devPtr = phio[id];
            resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
            resDesc.res.linear.sizeInBytes = size;
            cudaCreateTextureObject(&tex_cache[id], &resDesc, &texDesc, NULL);

        }

        int gx = id % nGPUx;
        int gy = id / nGPUx;

        if (gx > 0) {
            printf("Setting P2P Access for (%d <--> %d), State: %d\n", id, id - 1,
                   cudaDeviceEnablePeerAccess(id - 1, 0));
        }
        if (gx < nGPUx - 1) {
            printf("Setting P2P Access for (%d <--> %d), State: %d\n", id, id + 1,
                   cudaDeviceEnablePeerAccess(id + 1, 0));
        }
        if (gy > 0) {
            printf("Setting P2P Access for (%d <--> %d), State: %d\n", id, id - nGPUx,
                   cudaDeviceEnablePeerAccess(id - nGPUx, 0));
        }
        if (gy < nGPUy - 1) {
            printf("Setting P2P Access for (%d <--> %d), State: %d\n", id, id + nGPUx,
                   cudaDeviceEnablePeerAccess(id + nGPUx, 0));
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&InTime, start, stop);
    printf("Data input time for GPU: %f (ms) \n", InTime);
    cudaEventRecord(start, 0);

    double error = 10 * TOL;
    int iter = 0;
    int shm_size = threadsPerBlock * threadsPerBlock / nGPUx / nGPUy * sizeof(float);

    while ((error > TOL) && (iter < ITER_MAX)) {

#pragma omp parallel num_threads(nGPU) default(none) \
shared(phi, phio, buffer, buffer_host, nGPUx, nGPUy, nGPU, griddim, blockdim, buffer_size, shm_size, \
        size, useTexture, tex_cache)
        {
            int id = omp_get_thread_num();
            cudaSetDevice(id);

            int gx = id % nGPUx;
            int gy = id / nGPUx;

            if (useTexture == 0) {
                float *phi_r = gx == nGPUx - 1 ? NULL : phio[id + 1];
                float *phi_l = gx == 0 ? NULL : phio[id - 1];
                float *phi_t = gy == nGPUy - 1 ? NULL : phio[id + nGPUx];
                float *phi_b = gy == 0 ? NULL : phio[id - nGPUx];

                laplacian<<<griddim, blockdim, shm_size>>>(phi[id], phio[id], buffer[id],
                                                           phi_r, phi_l, phi_t, phi_b);
            } else {
                cudaTextureObject_t phi_r = gx == nGPUx - 1 ? 0 : tex_cache[id + 1];
                cudaTextureObject_t phi_l = gx == 0 ? 0 : tex_cache[id - 1];
                cudaTextureObject_t phi_f = gy == nGPUy - 1 ? 0 : tex_cache[id + nGPUx];
                cudaTextureObject_t phi_b = gy == 0 ? 0 : tex_cache[id - nGPUx];

                laplacian_tex<<<griddim, blockdim, shm_size>>>(phi[id], phio[id],
                                                               buffer[id], tex_cache[id],
                                                               phi_r, phi_l, phi_f, phi_b);
            }

            cudaMemcpy(buffer_host[id], buffer[id], buffer_size / nGPU, cudaMemcpyDeviceToHost);
//            cudaMemcpy(phio[id], phi[id], size / nGPU, cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();

        }

        error = 0.0;
        for (int id = 0; id < nGPU; id++) {
            for (int i = 0; i < buffer_dim; i++) {
                error += buffer_host[id][i];
            }
        }
        error = sqrt(error);

        if (iter++ % 1000 == 0) {
            printf("Iteration: %d, Error: %20.15e \n", iter, error);
        }

    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&PTime, start, stop);
    printf("Processing time for GPU: %f (ms) \n", PTime);
    cudaEventRecord(start, 0);

#pragma omp parallel for num_threads(nGPU) default(none) shared(phi, phi_host, nGPU, size)
    for (int id = 0; id < nGPU; id++) {
        cudaSetDevice(id);
        cudaMemcpy(phi_host[id], phi[id], size / nGPU, cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&OutTime, start, stop);
    printf("Data output time for GPU: %f (ms) \n", OutTime);
    cudaEventRecord(start, 0);

    merge(phi_host, phi_global, nGPUx, nGPUy);
    plot(phi_global, 1);
}

void merge(float **phis, float *phi, int nGx, int nGy) {

    for (int gx = 0; gx < nGx; gx++) {
        for (int gy = 0; gy < nGy; gy++) {
            for (int i = 0; i < GRIDS / nGx; i++) {
                for (int j = 0; j < GRIDS / nGy; j++) {
                    int site = (i + gx * GRIDS / nGx) + (j + gy * GRIDS / nGy) * GRIDS;
                    phi[site] = phis[gx + nGx * gy][i + j * GRIDS / nGx];
                }
            }
        }
    }
}


void plot(float *phi, int id) {

    char filename[100];
    sprintf(filename, OUTPUT, id);

    float h = 1.0 / (float) GRIDS;
    FILE *out1;
    out1 = fopen(filename, "w");
    fprintf(out1, "TITLE = \"Numerical Solution\"\n");
    fprintf(out1, "VARIABLES =\"X\" \"Y\" \"Phi\" \n");
    fprintf(out1, "ZONE T=\"Numerical\" I=%d, J=%d, F=POINT \n", GRIDS, GRIDS);
    for (int i = 0; i < GRIDS * GRIDS; i++) {
        fprintf(out1, "%f %f %f \n", (i % GRIDS) * h, (i / GRIDS) * h, phi[i]);
    }
    fclose(out1);
}