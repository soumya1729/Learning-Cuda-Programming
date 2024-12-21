#include <iostream>
#include <cuda_runtime.h>
#include <SFML/Graphics.hpp> // For window rendering

#define WIDTH 1920
#define HEIGHT 1080
#define MAX_ITER 1000

// CUDA kernel for Mandelbrot computation
__global__ void mandelbrotKernel(unsigned char* image, int width, int height, int max_iter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float jx = (x - width / 2.0f) * 4.0f / width;
    float jy = (y - height / 2.0f) * 4.0f / height;

    float zx = 0.0f, zy = 0.0f;
    int iter = 0;
    while (zx * zx + zy * zy < 4.0f && iter < max_iter) {
        float temp = zx * zx - zy * zy + jx;
        zy = 2.0f * zx * zy + jy;
        zx = temp;
        iter++;
    }

    int index = (y * width + x) * 4; // RGBA format
    float t = iter / (float)max_iter;
    image[index] = (unsigned char)(9 * (1 - t) * t * t * t * 255);       // Red
    image[index + 1] = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * 255); // Green
    image[index + 2] = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255); // Blue
    image[index + 3] = 255; // Alpha
}

int main() {
    const int imageSize = WIDTH * HEIGHT * 4;
    unsigned char* h_image = new unsigned char[imageSize];
    unsigned char* d_image;

    cudaMalloc((void**)&d_image, imageSize);

    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

    // Launch CUDA kernel
    mandelbrotKernel<<<grid, block>>>(d_image, WIDTH, HEIGHT, MAX_ITER);
    cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost);

    // SFML for visualization
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Mandelbrot Set (CUDA Rendered)");
    sf::Texture texture;
    texture.create(WIDTH, HEIGHT);
    sf::Sprite sprite(texture);

    texture.update(h_image);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }
        window.clear();
        window.draw(sprite);
        window.display();
    }

    cudaFree(d_image);
    delete[] h_image;

    return 0;
}
