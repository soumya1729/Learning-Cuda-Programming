#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <SFML/Graphics.hpp>

#define NUM_PARTICLES 1000
#define WIDTH 800
#define HEIGHT 600
#define GRAVITY 0.1f
#define PARTICLE_SIZE 2
#define MAX_VELOCITY 5

// Particle structure
struct Particle {
    float x, y;
    float vx, vy;
    sf::Color color;
};

// Kernel to update particle positions
__global__ void updateParticles(Particle* particles, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        Particle& p = particles[idx];

        // Apply gravity
        p.vy += GRAVITY;

        // Update position
        p.x += p.vx;
        p.y += p.vy;

        // Check boundaries and bounce off walls
        if (p.x < 0 || p.x >= width) p.vx = -p.vx;
        if (p.y < 0 || p.y >= height) p.vy = -p.vy;

        // Add some color effect based on position
        p.color.r = (int)(abs(sin(p.x / width * 3.14) * 255));
        p.color.g = (int)(abs(cos(p.y / height * 3.14) * 255));
        p.color.b = (int)(abs(sin((p.x + p.y) / (width + height) * 3.14) * 255));
    }
}

int main() {
    // Set up window with SFML
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "CUDA Particle Simulation");

    // Allocate memory for particles
    Particle* h_particles = new Particle[NUM_PARTICLES];
    Particle* d_particles;
    cudaMalloc(&d_particles, NUM_PARTICLES * sizeof(Particle));

    // Initialize particles
    for (int i = 0; i < NUM_PARTICLES; i++) {
        h_particles[i].x = rand() % WIDTH;
        h_particles[i].y = rand() % HEIGHT;
        h_particles[i].vx = (rand() % 10 - 5) / 10.0f * MAX_VELOCITY;
        h_particles[i].vy = (rand() % 10 - 5) / 10.0f * MAX_VELOCITY;
        h_particles[i].color = sf::Color(rand() % 255, rand() % 255, rand() % 255);
    }

    // Copy particles to device
    cudaMemcpy(d_particles, h_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);

    // Set up SFML shapes for particles
    sf::CircleShape* shapes = new sf::CircleShape[NUM_PARTICLES];
    for (int i = 0; i < NUM_PARTICLES; i++) {
        shapes[i].setRadius(PARTICLE_SIZE);
    }

    // Main loop
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // Update particles on the device
        updateParticles<<<(NUM_PARTICLES + 255) / 256, 256>>>(d_particles, WIDTH, HEIGHT);
        cudaMemcpy(h_particles, d_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);

        // Draw particles on SFML window
        window.clear();
        for (int i = 0; i < NUM_PARTICLES; i++) {
            shapes[i].setPosition(h_particles[i].x, h_particles[i].y);
            shapes[i].setFillColor(h_particles[i].color);
            window.draw(shapes[i]);
        }
        window.display();
    }

    // Free memory
    cudaFree(d_particles);
    delete[] h_particles;
    delete[] shapes;

    return 0;
}
