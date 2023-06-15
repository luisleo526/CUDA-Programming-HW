#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define OUTPUT "solution_%d.plt"
#define PI acos(-1.0)

int MAX = 10000000;     // maximum iterations
double eps = 1.0e-10;      // stopping criterion
float threshold = 1.5;

void plot(float *phi, int Nx, int Ny, int Nz, float dx, float dy, float dz, int id);

__global__ void laplacian(float *phi_old, float *phi_new, float *buffer,
                          float dx, float dy, float dz, bool flag) {

    extern __shared__ float cache[];
    float r, l, u, d, f, b;
    float diff;
    float ax, ay, az, src;
    int ip, im, jp, jm, kp, km;

    int Nx = blockDim.x * gridDim.x;
    int Ny = blockDim.y * gridDim.y;
    int Nz = blockDim.z * gridDim.z;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int cacheIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    int site = i + j * Nx + k * Nx * Ny;

    ax = 1.0 / (dx * dx);
    ay = 1.0 / (dy * dy);
    az = 1.0 / (dz * dz);

    if ((i == 0) || (i == Nx - 1) || (j == 0) || (j == Ny - 1) || (k == 0) || (k == Nz - 1)) {
        diff = 0.0;
    } else {

        if ((i == Nx / 2) && (j == Ny / 2) && (k == Nz / 2)) {
            src = 1.0 / dx / dy / dz;
        } else {
            src = 0.0;
        }

        ip = site + 1;
        im = site - 1;
        jp = site + Nx;
        jm = site - Nx;
        kp = site + Nx * Ny;
        km = site - Nx * Ny;

        if (flag) {
            r = phi_old[ip];
            l = phi_old[im];
            f = phi_old[jp];
            b = phi_old[jm];
            u = phi_old[kp];
            d = phi_old[km];

            phi_new[site] = (ax * (r + l) + ay * (f + b) + az * (u + d) - src) / (ax + ay + az) / 2;
//            phi_new[site] = 1.5 * phi_new[site] - 0.5 * phi_old[site];
//            phi_new[site] = (r + l + f + b + u + d) / 6.0;

        } else {

            r = phi_new[ip];
            l = phi_new[im];
            f = phi_new[jp];
            b = phi_new[jm];
            u = phi_new[kp];
            d = phi_new[km];

            phi_old[site] = (ax * (r + l) + ay * (f + b) + az * (u + d) - src) / (ax + ay + az) / 2;
//            phi_old[site] = 1.5 * phi_old[site] - 0.5 * phi_new[site];
//            phi_old[site] = (r + l + f + b + u + d) / 6.0;
        }

        diff = phi_new[site] - phi_old[site];
    }

    cache[cacheIndex] = diff * diff;
    __syncthreads();

    int ib = blockDim.x * blockDim.y * blockDim.z / 2;
    while (ib != 0) {
        if (cacheIndex < ib)
            cache[cacheIndex] += cache[cacheIndex + ib];
        __syncthreads();
        ib /= 2;
    }
    int blockIndex = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;
    if (cacheIndex == 0) buffer[blockIndex] = cache[0];
}

int main(void) {

    cudaSetDevice(0);

    float *phi;
    float *phi_h;
    float *phi_old;
    float *src;
    float *src_h;
    float *buffer;
    float *buffer_h;
    int L, NN, T;
    int Nx, Ny, Nz;
    int tx, ty, tz;
    int Lx, Ly, Lz;

    // Set GPU device
    cudaError_t err = cudaSuccess;
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printf("!!! Cannot select GPU with device ID = 0");
        exit(1);
    }

    printf("Enter the domain size (LxLxL) of the 3D lattice domain: \n");
    scanf("%d", &L);

    printf("Enter the number of grids (N x N x N) of the 3D lattice domain: \n");
    scanf("%d", &NN);

    printf("Enter the number of threads (t, t, t) per block: \n");
    scanf("%d", &T);

    // Structured Grids
    Nx = NN;
    Ny = NN;
    Nz = NN;
    Lx = L;
    Ly = L;
    Lz = L;
    tx = T;
    ty = T;
    tz = T;

//    if (tx * ty * tz > 1024) {
//        printf("The number of threads per block must be less than 1024 ! \n");
//        exit(0);
//    }

    int bx = Nx / tx;
    int by = Ny / ty;
    int bz = Nz / tz;

    if (bx * tx != Nx) {
        printf("The block size in x is incorrect\n");
        exit(0);
    }

    if (by * ty != Ny) {
        printf("The block size in y is incorrect\n");
        exit(0);
    }

    if (bz * tz != Nz) {
        printf("The block size in z is incorrect\n");
        exit(0);
    }

    if ((bx > 65535) || (by > 65535) || (bz > 65535)) {
        printf("The grid size exceeds the limit ! \n");
        exit(0);
    }

    dim3 blockdim(tx, ty, tz);
    dim3 griddim(bx, by, bz);

    float dx = (float) Lx / (Nx - 1);
    float dy = (float) Ly / (Ny - 1);
    float dz = (float) Lz / (Nz - 1);

    int N = Nx * Ny * Nz;
    int size = N * sizeof(float);

    phi_h = (float *) malloc(size);
    src_h = (float *) malloc(size);
    memset(phi_h, 0, size);
    memset(src_h, 0, size);

//    threshold = threshold * dx;
//    float r, x, y, z;
//    // Source term
//    for (int i = 0; i < Nx; ++i) {
//        for (int j = 0; j < Ny; ++j) {
//            for (int k = 0; k < Nz; k++) {
//                x = i * dx;
//                y = j * dy;
//                z = k * dz;
//                r = sqrt(pow(x - (float) Lx / 2.0, 2) + pow(y - (float) Ly / 2.0, 2) + pow(z - (float) Lz / 2.0, 2));
//                if (r < threshold) {
//                    src_h[i + j * Nx + k * Nx * Ny] = 0.5 * (1.0 + r / threshold + sin(PI * r / threshold) / PI);
//                }
//            }
//        }
//    }

    int sm = tx * ty * tz * sizeof(float);
    int sb = bx * by * bz * sizeof(float);

    cudaMalloc((void **) &phi, size);
    cudaMalloc((void **) &phi_old, size);
    cudaMalloc((void **) &src, size);
    cudaMalloc((void **) &buffer, sb);

    buffer_h = (float *) malloc(sb);

    cudaMemcpy(phi, phi_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(phi_old, phi_h, size, cudaMemcpyHostToDevice);

    printf("Block Dim: (%d, %d, %d)\n", tx, ty, tz);
    printf(" Grid Dim: (%d, %d, %d)\n", bx, by, bz);
    printf("Mesh Size: (%.4e, %.4e, %.4e)\n", dx, dy, dz);
    printf("   Domain: %d x %d x %d (%d)\n", Nx, Ny, Nz, N);

    double error = 10 * eps;

    int iter = 0;
    while ((error > eps) && (iter++ < MAX)) {

        laplacian<<<griddim, blockdim, sm>>>(phi_old, phi, buffer, dx, dy, dz, true);
        laplacian<<<griddim, blockdim, sm>>>(phi_old, phi, buffer, dx, dy, dz, false);

        cudaMemcpy(buffer_h, buffer, sb, cudaMemcpyDeviceToHost);

        error = 0.0;
        for (int i = 0; i < bx * by * bz; ++i) {
            error += buffer_h[i];
        }
        error = sqrt(error);

        // if (iter % 200 == 0) {
        //     printf("Iter: %d, error: %.4e\n", iter, error);
        //     fflush(stdout);
        // }

    }

    cudaMemcpy(phi_h, phi, size, cudaMemcpyDeviceToHost);
    plot(phi_h, Nx, Ny, Nz, dx, dy, dz, (int)L);

}

void plot(float *phi, int Nx, int Ny, int Nz, float dx, float dy, float dz, int id) {

    char filename[100];
    sprintf(filename, OUTPUT, id);

    FILE *out1;
    out1 = fopen(filename, "w");
    fprintf(out1, "VARIABLES =\"r\" \"Phi\" \n");
    int idx = Nx / 2 + Nx * Ny / 2 + Nz * Ny * Nx / 2;
    for (int i = 0; i < Nx / 2; i++) {
        fprintf(out1, "%.4e %.4e \n", (float) i * dx, phi[idx + i]);
    }

    fclose(out1);
}
