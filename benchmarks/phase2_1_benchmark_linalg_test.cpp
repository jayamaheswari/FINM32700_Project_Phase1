#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include "linalg.hpp"

static const int NUM_RUNS = 100;

struct BenchmarkResult {
    double mean;
    double stddev;
};

std::unique_ptr<double[]> allocate_random_matrix(int rows, int cols, unsigned seed=0) {
    auto mat = std::make_unique<double[]>(rows * cols);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = dist(rng);
    }
    return mat;
}

std::unique_ptr<double[]> allocate_random_vector(int size, unsigned seed=1) {
    auto vec = std::make_unique<double[]>(size);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int i = 0; i < size; ++i) {
        vec[i] = dist(rng);
    }
    return vec;
}

BenchmarkResult benchmark_mv_row_major(int rows, int cols) {
    using namespace linalg;

    auto A = allocate_random_matrix(rows, cols, 0);
    auto x = allocate_random_vector(cols, 1);
    std::unique_ptr<double[]> result = std::make_unique<double[]>(rows);

    multiply_mv_row_major(A.get(), rows, cols, x.get(), result.get());

    std::vector<double> timings;
    timings.reserve(NUM_RUNS);

    for (int run = 0; run < NUM_RUNS; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        multiply_mv_row_major(A.get(), rows, cols, x.get(), result.get());
        auto end   = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        timings.push_back(elapsed_ms);
    }

    double sum = 0.0;
    for (auto t : timings) sum += t;
    double mean = sum / NUM_RUNS;

    double variance = 0.0;
    for (auto t : timings) {
        double diff = t - mean;
        variance += diff * diff;
    }
    variance /= NUM_RUNS;
    double stddev = std::sqrt(variance);

    return { mean, stddev };
}

BenchmarkResult benchmark_mv_col_major(int rows, int cols) {
    using namespace linalg;

    auto A = allocate_random_matrix(rows, cols, 2);
    auto x = allocate_random_vector(cols, 3);
    std::unique_ptr<double[]> result = std::make_unique<double[]>(rows);

    multiply_mv_col_major(A.get(), rows, cols, x.get(), result.get());

    std::vector<double> timings;
    timings.reserve(NUM_RUNS);

    for (int run = 0; run < NUM_RUNS; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        multiply_mv_col_major(A.get(), rows, cols, x.get(), result.get());
        auto end   = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        timings.push_back(elapsed_ms);
    }

    double sum = 0.0;
    for (auto t : timings) sum += t;
    double mean = sum / NUM_RUNS;

    double variance = 0.0;
    for (auto t : timings) {
        double diff = t - mean;
        variance += diff * diff;
    }
    variance /= NUM_RUNS;
    double stddev = std::sqrt(variance);

    return { mean, stddev };
}

BenchmarkResult benchmark_mm_naive(int rowsA, int colsA, int rowsB, int colsB) {
    using namespace linalg;

    if (colsA != rowsB) {
        throw std::invalid_argument("Dimension mismatch in benchmark_mm_naive.");
    }

    auto A = allocate_random_matrix(rowsA, colsA, 4);
    auto B = allocate_random_matrix(rowsB, colsB, 5);
    auto C = std::make_unique<double[]>(rowsA * colsB);

    multiply_mm_naive(A.get(), rowsA, colsA, B.get(), rowsB, colsB, C.get());

    std::vector<double> timings;
    timings.reserve(NUM_RUNS);

    for (int run = 0; run < NUM_RUNS; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        multiply_mm_naive(A.get(), rowsA, colsA, B.get(), rowsB, colsB, C.get());
        auto end   = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        timings.push_back(elapsed_ms);
    }

    double sum = 0.0;
    for (auto t : timings) sum += t;
    double mean = sum / NUM_RUNS;

    double variance = 0.0;
    for (auto t : timings) {
        double diff = t - mean;
        variance += diff * diff;
    }
    variance /= NUM_RUNS;
    double stddev = std::sqrt(variance);

    return { mean, stddev };
}

BenchmarkResult benchmark_mm_transposed(int rowsA, int colsA, int rowsB, int colsB) {
    using namespace linalg;

    if (colsA != rowsB) {
        throw std::invalid_argument("Dimension mismatch in benchmark_mm_transposed.");
    }

    auto A  = allocate_random_matrix(rowsA, colsA, 6);
    auto B  = allocate_random_matrix(rowsB, colsB, 7);
    auto Bt = std::make_unique<double[]>(rowsB * colsB);
    auto C  = std::make_unique<double[]>(rowsA * colsB);

    for (int i = 0; i < rowsB; ++i) {
        for (int j = 0; j < colsB; ++j) {
            Bt[j * rowsB + i] = B[i * colsB + j];
        }
    }

    multiply_mm_transposed_b(A.get(), rowsA, colsA, Bt.get(), rowsB, colsB, C.get());

    std::vector<double> timings;
    timings.reserve(NUM_RUNS);

    for (int run = 0; run < NUM_RUNS; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        multiply_mm_transposed_b(A.get(), rowsA, colsA, Bt.get(), rowsB, colsB, C.get());
        auto end   = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        timings.push_back(elapsed_ms);
    }

    double sum = 0.0;
    for (auto t : timings) sum += t;
    double mean = sum / NUM_RUNS;

    double variance = 0.0;
    for (auto t : timings) {
        double diff = t - mean;
        variance += diff * diff;
    }
    variance /= NUM_RUNS;
    double stddev = std::sqrt(variance);

    return { mean, stddev };
}

int main() {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "================= Benchmarking Linear Algebra Kernels =================\n";

    std::vector<std::pair<int,int>> mv_sizes = {
        {100, 100},
        {500, 500},
        {1000, 1000}
    };

    struct MMSize { int rA; int cA; int rB; int cB; };
    std::vector<MMSize> mm_sizes = {
        {100, 100, 100, 100},
        {500, 500, 500, 500},
        {1000, 1000, 1000, 1000}
    };

    std::cout << "\n--- MATRIX-VECTOR MULTIPLICATION BENCHMARKS ---\n";
    std::cout << "Rows  Cols  Layout         Mean(ms)   StdDev(ms)\n";
    for (auto& sz : mv_sizes) {
        int rows = sz.first;
        int cols = sz.second;

        // Row-major
        BenchmarkResult rowRes = benchmark_mv_row_major(rows, cols);
        std::cout << rows << "    " << cols
                  << "    Row-major       "
                  << rowRes.mean << "        "
                  << rowRes.stddev << "\n";

        // Col-major
        BenchmarkResult colRes = benchmark_mv_col_major(rows, cols);
        std::cout << rows << "    " << cols
                  << "    Column-major    "
                  << colRes.mean << "        "
                  << colRes.stddev << "\n";
    }

    // Benchmark MM
    std::cout << "\n--- MATRIX-MATRIX MULTIPLICATION BENCHMARKS ---\n";
    std::cout << "RowsA  ColsA  RowsB  ColsB   Type            Mean(ms)   StdDev(ms)\n";

    for (auto& sz : mm_sizes) {
        // naive
        BenchmarkResult naiveRes = benchmark_mm_naive(sz.rA, sz.cA, sz.rB, sz.cB);
        std::cout << sz.rA << "     " << sz.cA
                  << "     " << sz.rB << "     " << sz.cB
                  << "   Naive         "
                  << naiveRes.mean << "       "
                  << naiveRes.stddev << "\n";

        // transposed
        BenchmarkResult transRes = benchmark_mm_transposed(sz.rA, sz.cA, sz.rB, sz.cB);
        std::cout << sz.rA << "     " << sz.cA
                  << "     " << sz.rB << "     " << sz.cB
                  << "   Transposed-B  "
                  << transRes.mean << "       "
                  << transRes.stddev << "\n";
    }

    return 0;
}
