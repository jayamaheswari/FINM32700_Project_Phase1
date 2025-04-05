#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include "linalg.hpp"

using namespace linalg;

static const int NUM_RUNS = 10;


// Fill a matrix or vector with random values in [-1.0, 1.0]
void fill_random(double* data, int size, unsigned seed = 0) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < size; ++i) {
        data[i] = dist(rng);
    }
}

void aligned_free(void* ptr) {
#ifdef _MSC_VER  // Windows MSVC
    _aligned_free(ptr);
#else  // POSIX
    free(ptr);
#endif
}




// Naive matrix-matrix multiplication: C = A * B
void multiply_mm_naive(double* A, int rowsA, int colsA, double* B, int rowsB, int colsB, double* C) {
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            C[i * colsB + j] = 0.0;
            for (int k = 0; k < colsA; ++k) {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
}

// ¿çÆ½Ì¨µÄ 64 ×Ö½Ú¶ÔÆë·ÖÅäº¯Êý
double* aligned_alloc64(size_t size) {
#if defined(_MSC_VER)
    return static_cast<double*>(_aligned_malloc(size * sizeof(double), 64));
#else
    void* ptr = nullptr;
    posix_memalign(&ptr, 64, size * sizeof(double));
    return static_cast<double*>(ptr);
#endif
}

void aligned_free64(void* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// Benchmark helper structure
struct BenchmarkResult {
    double mean;
    double stddev;
};

// Timing for matrix-vector multiplication (row-major)
BenchmarkResult time_mv_row_major(int rows, int cols, bool use_aligned) {
    double* A = use_aligned ? aligned_alloc64(rows * cols) : new double[rows * cols];
    double* x = use_aligned ? aligned_alloc64(cols) : new double[cols];
    double* result = use_aligned ? aligned_alloc64(rows) : new double[rows];

    fill_random(A, rows * cols);
    fill_random(x, cols);

    std::vector<double> timings;
    timings.reserve(NUM_RUNS);

    for (int run = 0; run < NUM_RUNS; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        multiply_mv_row_major(A, rows, cols, x, result);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        timings.push_back(elapsed);
    }

    if (use_aligned) {
        aligned_free(A);
        aligned_free(x);
        aligned_free(result);
    }
    else {
        delete[] A;
        delete[] x;
        delete[] result;
    }

    double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
    double mean = sum / NUM_RUNS;

    double variance = 0.0;
    for (double t : timings) variance += (t - mean) * (t - mean);
    double stddev = std::sqrt(variance / NUM_RUNS);

    return { mean, stddev };
}

// Timing for matrix-vector multiplication (column-major)
BenchmarkResult time_mv_col_major(int rows, int cols, bool use_aligned) {
    double* A = use_aligned ? aligned_alloc64(rows * cols) : new double[rows * cols];
    double* x = use_aligned ? aligned_alloc64(cols) : new double[cols];
    double* result = use_aligned ? aligned_alloc64(rows) : new double[rows];

    fill_random(A, rows * cols);
    fill_random(x, cols);

    std::vector<double> timings;
    timings.reserve(NUM_RUNS);

    for (int run = 0; run < NUM_RUNS; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        multiply_mv_col_major(A, rows, cols, x, result);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        timings.push_back(elapsed);
    }

    if (use_aligned) {
        aligned_free(A);
        aligned_free(x);
        aligned_free(result);
    }
    else {
        delete[] A;
        delete[] x;
        delete[] result;
    }

    double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
    double mean = sum / NUM_RUNS;

    double variance = 0.0;
    for (double t : timings) variance += (t - mean) * (t - mean);
    double stddev = std::sqrt(variance / NUM_RUNS);

    return { mean, stddev };
}

// Timing for naive matrix-matrix multiplication
BenchmarkResult time_mm_naive(int rowsA, int colsA, int rowsB, int colsB, bool use_aligned) {
    double* A = use_aligned ? aligned_alloc64(rowsA * colsA) : new double[rowsA * colsA];
    double* B = use_aligned ? aligned_alloc64(rowsB * colsB) : new double[rowsB * colsB];
    double* C = use_aligned ? aligned_alloc64(rowsA * colsB) : new double[rowsA * colsB];

    fill_random(A, rowsA * colsA);
    fill_random(B, rowsB * colsB);

    std::vector<double> timings;
    timings.reserve(NUM_RUNS);

    for (int run = 0; run < NUM_RUNS; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        multiply_mm_naive(A, rowsA, colsA, B, rowsB, colsB, C);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        timings.push_back(elapsed);
    }

    if (use_aligned) {
        aligned_free(A);
        aligned_free(B);
        aligned_free(C);
    }
    else {
        delete[] A;
        delete[] B;
        delete[] C;
    }

    double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
    double mean = sum / NUM_RUNS;

    double variance = 0.0;
    for (double t : timings) variance += (t - mean) * (t - mean);
    double stddev = std::sqrt(variance / NUM_RUNS);

    return { mean, stddev };
}

// Timing for matrix-matrix multiplication with transposed B
BenchmarkResult time_mm_transposed(int rowsA, int colsA, int rowsB, int colsB, bool use_aligned) {
    double* A = use_aligned ? aligned_alloc64(rowsA * colsA) : new double[rowsA * colsA];
    double* B = use_aligned ? aligned_alloc64(rowsB * colsB) : new double[rowsB * colsB];
    double* B_t = use_aligned ? aligned_alloc64(rowsB * colsB) : new double[rowsB * colsB];
    double* C = use_aligned ? aligned_alloc64(rowsA * colsB) : new double[rowsA * colsB];

    fill_random(A, rowsA * colsA);
    fill_random(B, rowsB * colsB);

    // Transpose B into B_t
    for (int i = 0; i < rowsB; ++i) {
        for (int j = 0; j < colsB; ++j) {
            B_t[j * rowsB + i] = B[i * colsB + j];
        }
    }

    std::vector<double> timings;
    timings.reserve(NUM_RUNS);

    for (int run = 0; run < NUM_RUNS; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        multiply_mm_transposed_b(A, rowsA, colsA, B_t, rowsB, colsB, C);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        timings.push_back(elapsed);
    }

    if (use_aligned) {
        aligned_free(A);
        aligned_free(B);
        aligned_free(B_t);
        aligned_free(C);
    }
    else {
        delete[] A;
        delete[] B;
        delete[] B_t;
        delete[] C;
    }

    double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
    double mean = sum / NUM_RUNS;

    double variance = 0.0;
    for (double t : timings) variance += (t - mean) * (t - mean);
    double stddev = std::sqrt(variance / NUM_RUNS);

    return { mean, stddev };
}

int main() {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "================= Benchmarking Linear Algebra Kernels =================\n";

    std::vector<std::pair<int, int>> mv_sizes = {
        {100, 100}, {500, 500}, {1000, 1000}
    };

    struct MMSize { int rA; int cA; int rB; int cB; };
    std::vector<MMSize> mm_sizes = {
        {100, 100, 100, 100}, {500, 500, 500, 500}, {1000, 1000, 1000, 1000}
    };

    std::cout << "\nMATRIX-VECTOR MULTIPLICATION BENCHMARKS\n";
    std::cout << "Rows Cols   Layout       Align     Mean(ms)  StdDev(ms)\n";
    for (auto& sz : mv_sizes) {
        int rows = sz.first;
        int cols = sz.second;

        BenchmarkResult row_unaligned = time_mv_row_major(rows, cols, false);
        BenchmarkResult row_aligned = time_mv_row_major(rows, cols, true);
        std::cout << rows << "   " << cols << "   Row-Major    Unaligned   "
            << row_unaligned.mean << "    " << row_unaligned.stddev << "\n";
        std::cout << rows << "   " << cols << "   Row-Major    Aligned     "
            << row_aligned.mean << "    " << row_aligned.stddev << "\n";

        BenchmarkResult col_unaligned = time_mv_col_major(rows, cols, false);
        BenchmarkResult col_aligned = time_mv_col_major(rows, cols, true);
        std::cout << rows << "   " << cols << "   Column-Major Unaligned   "
            << col_unaligned.mean << "    " << col_unaligned.stddev << "\n";
        std::cout << rows << "   " << cols << "   Column-Major Aligned     "
            << col_aligned.mean << "    " << col_aligned.stddev << "\n";
    }

    std::cout << "\nMATRIX-MATRIX MULTIPLICATION BENCHMARKS\n";
    std::cout << "RowsA ColsA RowsB ColsB   Type            Align      Mean(ms)  StdDev(ms)\n";
    for (auto& sz : mm_sizes) {
        BenchmarkResult naive_unaligned = time_mm_naive(sz.rA, sz.cA, sz.rB, sz.cB, false);
        BenchmarkResult naive_aligned = time_mm_naive(sz.rA, sz.cA, sz.rB, sz.cB, true);
        std::cout << sz.rA << "    " << sz.cA << "    " << sz.rB << "    " << sz.cB
            << "   Naive           Unaligned  " << naive_unaligned.mean << "    " << naive_unaligned.stddev << "\n";
        std::cout << sz.rA << "    " << sz.cA << "    " << sz.rB << "    " << sz.cB
            << "   Naive           Aligned    " << naive_aligned.mean << "    " << naive_aligned.stddev << "\n";

        BenchmarkResult trans_unaligned = time_mm_transposed(sz.rA, sz.cA, sz.rB, sz.cB, false);
        BenchmarkResult trans_aligned = time_mm_transposed(sz.rA, sz.cA, sz.rB, sz.cB, true);
        std::cout << sz.rA << "    " << sz.cA << "    " << sz.rB << "    " << sz.cB
            << "   Transposed-B    Unaligned  " << trans_unaligned.mean << "    " << trans_unaligned.stddev << "\n";
        std::cout << sz.rA << "    " << sz.cA << "    " << sz.rB << "    " << sz.cB
            << "   Transposed-B    Aligned    " << trans_aligned.mean << "    " << trans_aligned.stddev << "\n";
    }

    return 0;
}
