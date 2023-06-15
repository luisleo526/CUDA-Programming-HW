#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <float.h>


int ipow(int x, int n);

void mean_std(const double *x, int n, double *mean, double *std);

__global__ void importance_sampling(double *I, double C, int dim, int num) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    curandState state;
    curand_init((unsigned long long) clock() + i, 0, 0, &state);

    double w, w_old, y, Iold;
    int iter = 0;

    // Only one block
    while (i < num) {
        I[i] = 1.0;
        w = 1.0;
        for (int j = 0; j < dim; j++) {
            double x = curand_uniform_double(&state);
            y = -log(1.0 - x / C);
            I[i] += y * y;
            w = w * C * exp(-y);
        }
        I[i] = 1.0 / I[i] / w;

        if (iter > 0) {
            if (w > w_old) {
                Iold = I[i];
                w_old = w;
            } else {
                if (curand_uniform_double(&state) < w / w_old) {
                    Iold = I[i];
                    w_old = w;
                }
            }
        } else {
            Iold = I[i];
            w_old = w;
        }
        I[i] = Iold;
        iter++;
        i += blockDim.x;
    }

}

__global__ void simple_sampling(double *I, int dim, int num) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    curandState state;
    curand_init((unsigned long long) clock() + i, 0, 0, &state);

    if (i < num) {
        I[i] = 1.0;
        for (int j = 0; j < dim; j++) {
            double x = curand_uniform_double(&state);
            I[i] += x * x;
        }
        I[i] = 1.0 / I[i];
    }

}

int main() {

    int power_max = 16;
    int dim = 10;
    int num = ipow(2, power_max);
    double C = 1.0 / (1.0 - exp(-1.0));

    double *d_I, *d_Im, *d_w, *d_wold, *buffer, *buffer2;

    int threadsPerBlock;
    printf("Input threads per block: ");
    scanf("%d", &threadsPerBlock);
    int blocksPerGrid = (num + threadsPerBlock - 1) / threadsPerBlock;

    buffer = (double *) malloc(num * sizeof(double));
    buffer2 = (double *) malloc(num * sizeof(double));

    cudaMalloc((void **) &d_I, num * sizeof(double));
    cudaMalloc((void **) &d_Im, num * sizeof(double));
    cudaMalloc((void **) &d_w, num * sizeof(double));
    cudaMalloc((void **) &d_wold, num * sizeof(double));

    // Simple Sampling
    double *I = (double *) malloc(num * sizeof(double));
    // Importance Sampling
    double *Im = (double *) malloc(num * sizeof(double));

    double *x = (double *) malloc(dim * sizeof(double));
    double *xold = (double *) malloc(dim * sizeof(double));
    double wx, wx_old;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float cpu_sim, cpu_imp, gpu_sim, gpu_imp;

    cudaEventRecord(start, 0);

    double w;
    for (int i = 0; i < num; i++) {
        for (int j = 0; j < dim; j++) {
            x[j] = rand() / (double) RAND_MAX;
        }
        I[i] = 1.0;
        for (int j = 0; j < dim; j++) {
            // Simple Sampling
            I[i] += x[j] * x[j];
        }
        I[i] = 1.0 / I[i];
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_sim, start, stop);

    cudaEventRecord(start, 0);

    // Importance Sampling
    for (int i = 0; i < num; i++) {

        wx = 1.0;
        for (int j = 0; j < dim; j++) {
            x[j] = rand() / (double) RAND_MAX;
            double y = -log(1.0 - x[j] / C);
            wx *= C * exp(-y);
        }

        if (i > 0) {
            if (wx > wx_old) {
                for (int j = 0; j < dim; j++)xold[j] = x[j];
            } else {
                if (rand() / (double) RAND_MAX < wx / wx_old) {
                    for (int j = 0; j < dim; j++)xold[j] = x[j];
                }
            }
        } else {
            for (int j = 0; j < dim; j++)xold[j] = x[j];
            wx_old = wx;
        }

        Im[i] = 1.0;
        w = 1.0;
        for (int j = 0; j < dim; j++) {
            double y = -log(1.0 - xold[j] / C);
            Im[i] += y * y;
            w *= C * exp(-y);
        }
        Im[i] = 1.0 / Im[i] / w;

    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_imp, start, stop);

    cudaEventRecord(start, 0);
    simple_sampling<<<blocksPerGrid, threadsPerBlock >>>(d_I, dim, num);;
    cudaMemcpy(buffer, d_I, num * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_sim, start, stop);

    cudaEventRecord(start, 0);
    importance_sampling<<<1, threadsPerBlock >>>(d_I, C, dim, num);
    cudaMemcpy(buffer2, d_I, num * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_imp, start, stop);

    FILE *f1, *f2, *f3, *f4;

    f1 = fopen("simple_sampling_cpu.txt", "w");
    f2 = fopen("simple_sampling_gpu.txt", "w");
    f3 = fopen("importance_sampling_cpu.txt", "w");
    f4 = fopen("importance_sampling_gpu.txt", "w");

    double mean, std;
    for (int i = 6; i < power_max + 1; i++) {
        num = ipow(2, i);
        printf("N = %d \n", num);
        mean_std(I, num, &mean, &std);
        printf("    Simple Sampling(CPU): %f (+/-) %.6e, Processing Time: %f\n", mean, std, cpu_sim);
        fprintf(f1, "%d, %f, %f\n", num, mean, std);

        mean_std(buffer, num, &mean, &std);
        printf("    Simple Sampling(GPU): %f (+/-) %.6e, Processing Time: %f\n", mean, std, gpu_sim);
        fprintf(f2, "%d, %f, %f\n", num, mean, std);

        mean_std(Im, num, &mean, &std);
        printf("Importance Sampling(CPU): %f (+/-) %.6e, Processing Time: %f\n", mean, std, cpu_imp);
        fprintf(f3, "%d, %f, %f\n", num, mean, std);

        mean_std(buffer2, num, &mean, &std);
        printf("Importance Sampling(GPU): %f (+/-) %.6e, Processing Time: %f\n", mean, std, gpu_imp);
        fprintf(f4, "%d, %f, %f\n", num, mean, std);
        printf("-----------------------------\n");
    }
    fclose(f1);
    fclose(f2);
    fclose(f3);
    fclose(f4);


}

int ipow(int x, int n) {
    int result = 1;
    int iter = 0;
    while (iter < n) {
        result *= x;
        iter++;
    }
    return result;
}

void mean_std(const double *x, int n, double *mean, double *std) {

    double l1, l2;
    l1 = 0.0;
    l2 = 0.0;
    for (int i = 0; i < n; i++) {
        l1 = l1 + x[i];
        l2 = l2 + x[i] * x[i];
    }

    *mean = l1 / (double) n;
    *std = sqrt((l2 / (double) n - (*mean) * (*mean)) / (double) n);

}
