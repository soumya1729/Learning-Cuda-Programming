#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

#define WIDTH 3840  // 4K Resolution for aesthetics
#define HEIGHT 2160
#define MAX_ITER 2000  // More iterations for sharper fractals
#define BLOCK_SIZE 16  // Optimal block size for GPU

// Color mapping kernel
__device__ void colorMap(int iter, int maxIter, unsigned char& r, unsigned char& g, unsigned char& b) {
    float t = iter / (float)maxIter;
    r = (unsigned char)(9 * (1 - t) * t * t * t * 255);
    g = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * 255);
    b = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
}

// Mandelbrot Kernel
__global__ void mandelbrotKernel(unsigned char* image, int width, int height, float xMin, float xMax, float yMin, float yMax, int maxIter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float real = xMin + (x / (float)width) * (xMax - xMin);
    float imag = yMin + (y / (float)height) * (yMax - yMin);

    float zr = 0.0, zi = 0.0;
    int iter = 0;

    while (zr * zr + zi * zi <= 4.0f && iter < maxIter) {
        float temp = zr * zr - zi * zi + real;
        zi = 2.0f * zr * zi + imag;
        zr = temp;
        iter++;
    }

    int index = (y * width + x) * 3;
    unsigned char r, g, b;
    colorMap(iter, maxIter, r, g, b);

    image[index] = r;
    image[index + 1] = g;
    image[index + 2] = b;
}

// Save Image in PPM format
void saveImage(const char* filename, unsigned char* image, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<char*>(image), width * height * 3);
    file.close();
}

// CUDA Error Checking
void checkCudaError(cudaError_t err, const char* message) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(err) << "\n";
        exit(EXIT_FAILURE);
    }
}

int main() {
    unsigned char* d_image;
    unsigned char* h_image = new unsigned char[WIDTH * HEIGHT * 3];

    checkCudaError(cudaMalloc(&d_image, WIDTH * HEIGHT * 3 * sizeof(unsigned char)), "Allocating device memory");

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);

    std::cout << "Generating Mandelbrot fractal on GPU...\n";
    mandelbrotKernel<<<numBlocks, threadsPerBlock>>>(d_image, WIDTH, HEIGHT, -2.0f, 1.0f, -1.2f, 1.2f, MAX_ITER);
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution failed");

    checkCudaError(cudaMemcpy(h_image, d_image, WIDTH * HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost), "Copying image to host");

    cudaFree(d_image);

    saveImage("mandelbrot_optimized.ppm", h_image, WIDTH, HEIGHT);
    delete[] h_image;

    std::cout << "Mandelbrot fractal saved as 'mandelbrot_optimized.ppm'\n";
    return 0;
}
