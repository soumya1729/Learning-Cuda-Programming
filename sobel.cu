#include <iostream>
#include <opencv2/opencv.hpp>

#define BLOCK_SIZE 16  // Block size for tiling

// Sobel Kernel for edge detection
__global__ void sobelKernel(const unsigned char* d_input, unsigned char* d_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Sobel kernels
        int Gx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
        int Gy[3][3] = { {-1, -2, -1}, { 0, 0, 0}, { 1, 2, 1} };

        int sumX = 0, sumY = 0;

        // Apply Sobel kernel
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int pixelX = min(max(x + j, 0), width - 1);
                int pixelY = min(max(y + i, 0), height - 1);

                unsigned char pixel = d_input[pixelY * width + pixelX];
                sumX += pixel * Gx[i + 1][j + 1];
                sumY += pixel * Gy[i + 1][j + 1];
            }
        }

        // Compute magnitude
        int magnitude = min(sqrtf(sumX * sumX + sumY * sumY), 255.0f);
        d_output[y * width + x] = (unsigned char)magnitude;
    }
}

int main() {
    // Load input image using OpenCV
    cv::Mat inputImage = cv::imread("input_image.jpg", cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    // Allocate memory for input and output images
    unsigned char *d_input, *d_output;
    unsigned char *h_input = inputImage.data;
    unsigned char *h_output = new unsigned char[width * height];

    // Allocate memory on device
    cudaMalloc((void**)&d_input, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&d_output, width * height * sizeof(unsigned char));

    // Copy input image to device
    cudaMemcpy(d_input, h_input, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch Sobel filter kernel
    sobelKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_output, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Create output image
    cv::Mat outputImage(height, width, CV_8UC1, h_output);

    // Save the output image
    cv::imwrite("output_image.jpg", outputImage);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_output;

    std::cout << "Edge detection complete. Output saved as 'output_image.jpg'" << std::endl;
    return 0;
}
