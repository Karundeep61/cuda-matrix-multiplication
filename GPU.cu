#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector>
#include <cassert>
#include <iostream>
#include <chrono>
#include <cmath>

// ==========================================================
// Matrix struct (row-major)
// ==========================================================
struct Matrix {
    int rows;
    int cols;
    std::vector<float> data;

    Matrix(int r, int c) : rows(r), cols(c), data(r* c, 0.0f) {}

    inline float& operator()(int i, int j) {
        return data[i * cols + j];
    }

    inline const float& operator()(int i, int j) const {
        return data[i * cols + j];
    }
};

// ==========================================================
// CPU matrix multiplication (baseline)
// ==========================================================
void cpu_matmul(const Matrix& A, const Matrix& B, Matrix& C) {
    assert(A.cols == B.rows);

    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A.cols; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
}

// ==========================================================
// Naive CUDA kernel (one thread per output element)
// ==========================================================
#define TILE 16

__global__ void gpu_matmul_tiled(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles along K dimension
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {

        // Load tile of A into shared memory
        if (row < M && (t * TILE + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] =
            A[row * K + (t * TILE + threadIdx.x)];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B into shared memory
        if (col < N && (t * TILE + threadIdx.y) < K)
            Bs[threadIdx.y][threadIdx.x] =
            B[(t * TILE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();  // 🔴 critical barrier

        // Compute partial dot product
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();  // prepare for next tile
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ==========================================================
// Main
// ==========================================================
int main() {
    const int M = 512;
    const int K = 512;
    const int N = 512;

    Matrix A(M, K);
    Matrix B(K, N);
    Matrix C_cpu(M, N);
    Matrix C_gpu(M, N);

    // Initialize matrices
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            A(i, k) = 1.0f;

    for (int k = 0; k < K; ++k)
        for (int j = 0; j < N; ++j)
            B(k, j) = 1.0f;

    // ---------------- CPU timing ----------------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_matmul(A, B, C_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = cpu_end - cpu_start;

    std::cout << "CPU MatMul Time: " << cpu_time.count() << " seconds\n";

    // ---------------- GPU memory ----------------
    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    // ---------------- GPU timing ----------------
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gpu_matmul_tiled << <grid, block >> > (d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    std::cout << "GPU Kernel Time: " << gpu_ms / 1000.0f << " seconds\n";

    cudaMemcpy(C_gpu.data.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // ---------------- Correctness check ----------------
    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        max_error = std::max(max_error, std::abs(C_cpu.data[i] - C_gpu.data[i]));
    }

    std::cout << "Max error: " << max_error << "\n";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
