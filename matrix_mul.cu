#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

#define BLOCK_SIZE 16  // Block size (16x16 threads per block)

// CUDA Kernel for Matrix Multiplication
__global__ void matrixMul(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void initializeMatrix(float* mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    int N = 2048; // Matrix size (N x N)
    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    srand(time(0));
    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy matrices to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    std::cout << "Performing matrix multiplication on GPU..." << std::endl;
    matrixMul<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify result (optional for large matrices)
    bool success = true;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float expected = 0.0f;
            for (int k = 0; k < N; k++) {
                expected += h_A[i * N + k] * h_B[k * N + j];
            }
            if (fabs(h_C[i * N + j] - expected) > 1e-3) {
                success = false;
                break;
            }
        }
    }

    std::cout << (success ? "Test PASSED!" : "Test FAILED!") << std::endl;

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
