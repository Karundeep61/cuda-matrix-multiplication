CUDA Matrix Multiplication: CPU vs GPU Performance
Overview

This project implements dense matrix multiplication from scratch on both the CPU and an NVIDIA GPU using CUDA. The objective is to understand how performance differs between sequential CPU execution and massively parallel GPU execution, and to explore how memory access patterns and optimization strategies affect real-world performance.

Rather than relying on optimized libraries, all implementations were written manually to expose the underlying execution and memory behavior of matrix multiplication on modern hardware.

Problem Description

Matrix multiplication is a core operation in machine learning, scientific computing, and physics simulations. It is computationally intensive and highly parallelizable, making it an ideal workload for studying GPU acceleration.

The project compares three implementations:

A CPU baseline

An optimized CUDA GPU kernel using shared memory tiling

All implementations produce identical numerical results and are validated for correctness.

Implementation Details
Matrix Representation

Matrices are stored in contiguous memory using row-major layout. This layout is shared across CPU and GPU implementations to ensure consistent indexing and fair performance comparison. Each matrix element is accessed using calculated offsets rather than higher-level abstractions.

CPU Implementation

The CPU version uses a straightforward triple-nested loop approach. This implementation serves as a correctness reference and performance baseline. Execution time is measured using high-resolution timing utilities.

While simple and easy to reason about, this approach is limited by:

sequential execution

cache inefficiencies

limited parallelism

Naive CUDA GPU Implementation

Shared Memory Tiled CUDA Implementation

To reduce redundant global memory access, a second CUDA kernel was implemented using shared memory tiling. In this version, each thread block cooperatively loads small tiles of the input matrices into fast shared memory.

Threads within the block reuse these values to compute partial results before moving to the next tile. Synchronization is used to ensure correctness during shared memory access.

This optimization reflects techniques used in production GPU libraries. For moderate matrix sizes, the added synchronization and control overhead can outweigh the benefits, demonstrating that performance optimizations are workload-dependent.

Performance Results

Typical benchmark results show:

CPU execution times on the order of hundreds of milliseconds

GPU kernel execution times on the order of microseconds

Speedups of several hundred times compared to the CPU baseline

Zero numerical error between CPU and GPU results

Repeated kernel executions show decreasing runtime due to CUDA context initialization, GPU clock boosting, and improved cache behavior.

Key Observations

GPUs are exceptionally well-suited for dense linear algebra workloads.

Correctness validation is essential when optimizing low-level GPU code.

Shared memory optimizations are powerful but must be applied judiciously.

Performance engineering requires empirical measurement rather than assumptions.

Technologies Used

C++, CUDA C/C++, NVIDIA CUDA Runtime, Visual Studio, performance timing utilities.

How to Run

Open the project in Visual Studio with CUDA support enabled, build the solution, and run the executable. Matrix sizes can be adjusted to observe how performance scales with problem size.

Notes

This project is intended as a systems-level learning exercise. For production workloads, optimized vendor libraries such as cuBLAS should be used instead.
