// smoke_sim.cu
#include <GL/glew.h>      // Modern OpenGL extension wrangler
#include <GL/freeglut.h>  // FreeGLUT library
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>

// --- PARAMETERS ---
#define WIDTH 512
#define HEIGHT 512
#define DT 0.1f
#define DIFF 0.0001f
#define VISC 0.0001f

float *d_density, *d_velocityX, *d_velocityY;
GLuint pbo; // Pixel Buffer Object
struct cudaGraphicsResource *cuda_pbo_resource;

// --- CUDA Kernels ---

__global__ void applyForceKernel(float *velocityX, float *velocityY, int mouseX, int mouseY, float force) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x == mouseX && y == mouseY) {
        velocityX[y * WIDTH + x] += force;
        velocityY[y * WIDTH + x] += force;
    }
}

__global__ void advectKernel(float *density, float *velocityX, float *velocityY, float *newDensity) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * WIDTH + x;
    float x0 = x - velocityX[idx] * DT;
    float y0 = y - velocityY[idx] * DT;

    x0 = fmaxf(0.0f, fminf(x0, WIDTH - 1));
    y0 = fmaxf(0.0f, fminf(y0, HEIGHT - 1));

    int x0_int = (int)x0, y0_int = (int)y0;
    newDensity[idx] = density[y0_int * WIDTH + x0_int];
}

// --- OpenGL Display ---

void render() {
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    float *devPtr;
    size_t size;
    cudaGraphicsResourceGetMappedPointer((void **)&devPtr, &size, cuda_pbo_resource);

    dim3 blockDim(16, 16);
    dim3 gridDim(WIDTH / 16, HEIGHT / 16);

    float *d_newDensity;
    cudaMalloc(&d_newDensity, WIDTH * HEIGHT * sizeof(float));
    advectKernel<<<gridDim, blockDim>>>(d_density, d_velocityX, d_velocityY, d_newDensity);
    cudaMemcpy(d_density, d_newDensity, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(d_newDensity);

    cudaMemcpy(devPtr, d_density, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);

    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(WIDTH, HEIGHT, GL_LUMINANCE, GL_FLOAT, 0);
    glutSwapBuffers();
}

// --- OpenGL Initialization ---

void initOpenGL() {
    glewExperimental = GL_TRUE;  // Ensure modern OpenGL features are enabled
    GLenum glewStatus = glewInit();
    if (glewStatus != GLEW_OK) {
        std::cerr << "GLEW Initialization Failed: " << glewGetErrorString(glewStatus) << std::endl;
        exit(EXIT_FAILURE);
    }

    glViewport(0, 0, WIDTH, HEIGHT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, WIDTH, 0, HEIGHT);

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * sizeof(float), 0, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}


// --- CUDA Initialization ---

void initCUDA() {
    cudaMalloc(&d_density, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc(&d_velocityX, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc(&d_velocityY, WIDTH * HEIGHT * sizeof(float));
    cudaMemset(d_density, 0, WIDTH * HEIGHT * sizeof(float));
    cudaMemset(d_velocityX, 0, WIDTH * HEIGHT * sizeof(float));
    cudaMemset(d_velocityY, 0, WIDTH * HEIGHT * sizeof(float));
}

// --- Update Function ---

void updateSimulation() {
    dim3 blockDim(16, 16);
    dim3 gridDim(WIDTH / 16, HEIGHT / 16);

    // Apply a small force in the center
    applyForceKernel<<<gridDim, blockDim>>>(d_velocityX, d_velocityY, WIDTH / 2, HEIGHT / 2, 0.1f);

    glutPostRedisplay();
}

// --- Cleanup ---

void cleanup() {
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    cudaFree(d_density);
    cudaFree(d_velocityX);
    cudaFree(d_velocityY);
    glDeleteBuffers(1, &pbo);
}


// --- Main ---

int main(int argc, char **argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("CUDA Smoke Fluid Simulation");

    initOpenGL();
    initCUDA();

    glutDisplayFunc(render);
    glutIdleFunc(updateSimulation);
    glutCloseFunc(cleanup);

    glutMainLoop();
    return 0;
}
