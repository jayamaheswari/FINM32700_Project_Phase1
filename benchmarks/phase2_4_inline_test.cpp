#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include "linalg.hpp"
#include "linalg_inline.hpp"

static const int NUM_RUNS = 10;
static const bool INLINE_FLAG = false;
struct BenchmarkResult {
    double mean;
    double stddev;
};

double* allocate_random_matrix(int rows, int cols) {
    double* mat = new double[rows * cols];
    static std::mt19937_64 rng(0);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = dist(rng);
    }
    return mat;
}

double* allocate_random_vector(int size) {
    double* vec = new double[size];
    static std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < size; ++i) {
        vec[i] = dist(rng);
    }
    return vec;
}

BenchmarkResult time_mv_row_major(int rows, int cols) {
    double* A = allocate_random_matrix(rows, cols);
    double* x = allocate_random_vector(cols);
    double* result = new double[rows];

    std::vector<double> timings;
    timings.reserve(NUM_RUNS);

    for (int run = 0; run < NUM_RUNS; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        if (INLINE_FLAG) {
            multiply_mv_row_major_inline(A, rows, cols, x, result);
        } else {
            linalg::multiply_mv_row_major(A, rows, cols, x, result);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        timings.push_back(elapsed);
    }

    delete[] A;
    delete[] x;
    delete[] result;

    double sum = 0.0;
    for (double t : timings) sum += t;
    double mean = sum / NUM_RUNS;

    double variance = 0.0;
    for (double t : timings) {
        variance += (t - mean) * (t - mean);
    }
    variance /= NUM_RUNS;
    double stddev = std::sqrt(variance);

    return { mean, stddev };
}

BenchmarkResult time_mv_col_major(int rows, int cols) {
    double* A = allocate_random_matrix(rows, cols);
    double* x = allocate_random_vector(cols);
    double* result = new double[rows];
    std::vector<double> timings;
    timings.reserve(NUM_RUNS);

    for (int run = 0; run < NUM_RUNS; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        if (INLINE_FLAG) {
            multiply_mv_col_major_inline(A, rows, cols, x, result);
        } else {
            linalg::multiply_mv_col_major(A, rows, cols, x, result);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        timings.push_back(elapsed);
    }

    delete[] A;
    delete[] x;
    delete[] result;

    double sum = 0.0;
    for (double t : timings) sum += t;
    double mean = sum / NUM_RUNS;

    double variance = 0.0;
    for (double t : timings) {
        variance += (t - mean) * (t - mean);
    }
    variance /= NUM_RUNS;
    double stddev = std::sqrt(variance);

    return { mean, stddev };
}

BenchmarkResult time_mm_naive(int rowsA, int colsA, int rowsB, int colsB) {
    double* A = allocate_random_matrix(rowsA, colsA);
    double* B = allocate_random_matrix(rowsB, colsB);
    double* C = new double[rowsA * colsB];
    std::vector<double> timings;
    timings.reserve(NUM_RUNS);

    for (int run = 0; run < NUM_RUNS; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        if (INLINE_FLAG) {
            multiply_mm_naive_inline(A, rowsA, colsA, B, rowsB, colsB, C);
        } else {
            linalg::multiply_mm_naive(A, rowsA, colsA, B, rowsB, colsB, C);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        timings.push_back(elapsed);
    }

    delete[] A;
    delete[] B;
    delete[] C;

    double sum = 0.0;
    for (double t : timings) sum += t;
    double mean = sum / NUM_RUNS;

    double variance = 0.0;
    for (double t : timings) {
        variance += (t - mean) * (t - mean);
    }
    variance /= NUM_RUNS;
    double stddev = std::sqrt(variance);

    return { mean, stddev };
}

BenchmarkResult time_mm_transposed(int rowsA, int colsA, int rowsB, int colsB) {
    double* A = allocate_random_matrix(rowsA, colsA);
    double* B = allocate_random_matrix(rowsB, colsB);
    double* B_t = new double[rowsB * colsB];
    double* C = new double[rowsA * colsB];

    for (int i = 0; i < rowsB; ++i) {
        for (int j = 0; j < colsB; ++j) {
            B_t[j * rowsB + i] = B[i * colsB + j];
        }
    }

    std::vector<double> timings;
    timings.reserve(NUM_RUNS);

    for (int run = 0; run < NUM_RUNS; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        if (INLINE_FLAG) {
            multiply_mm_transposed_b_inline(A, rowsA, colsA, B_t, rowsB, colsB, C);
        } else {
            linalg::multiply_mm_transposed_b(A, rowsA, colsA, B_t, rowsB, colsB, C);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        timings.push_back(elapsed);
    }

    delete[] A;
    delete[] B;
    delete[] B_t;
    delete[] C;

    double sum = 0.0;
    for (double t : timings) sum += t;
    double mean = sum / NUM_RUNS;

    double variance = 0.0;
    for (double t : timings) {
        variance += (t - mean) * (t - mean);
    }
    variance /= NUM_RUNS;
    double stddev = std::sqrt(variance);

    return { mean, stddev };
}

int main() {
    std::cout << "Inline flag: " << INLINE_FLAG << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "================= Benchmarking Linear Algebra Kernels =================\n";

    std::vector<std::pair<int,int>> mv_sizes = {
        {100, 100},   // small
        {1000, 1000},   // medium
        {1000000, 1000000}  // large
    };

    struct MMSize { int rA; int cA; int rB; int cB; };
    std::vector<MMSize> mm_sizes = {
        {100, 100, 100, 100},   // small
        {500, 500, 500, 500},   // medium
        {1000, 1000, 1000, 1000} // large
    };

    std::cout << "\nMATRIX-VECTOR MULTIPLICATION BENCHMARKS \n";
    std::cout << "Rows Cols Type Mean(ms) StdDev(ms)\n";
    for (auto& sz : mv_sizes) {
        int rows = sz.first;
        int cols = sz.second;

        BenchmarkResult rowRes = time_mv_row_major(rows, cols);
        std::cout << "   " << rows << "   " << cols << "   Row-Major       "
                  << rowRes.mean << "       " << rowRes.stddev << "\n";

        BenchmarkResult colRes = time_mv_col_major(rows, cols);
        std::cout << "   " << rows << "   " << cols << "   Column-Major    "
                  << colRes.mean << "       " << colRes.stddev << "\n";
    }

    std::cout << "\nMATRIX-MATRIX MULTIPLICATION BENCHMARKS\n";
    std::cout << " RowsA  ColsA  RowsB  ColsB   Type mean(ms)  StdDev(ms)\n";
    for (auto& sz : mm_sizes) {
        BenchmarkResult naiveRes = time_mm_naive(sz.rA, sz.cA, sz.rB, sz.cB);
        std::cout << "   " << sz.rA << "     " << sz.cA << "     " << sz.rB << "     " << sz.cB
                  << "   Naive              " << naiveRes.mean << "       " << naiveRes.stddev << "\n";

        BenchmarkResult transRes = time_mm_transposed(sz.rA, sz.cA, sz.rB, sz.cB);
        std::cout << "   " << sz.rA << "     " << sz.cA << "     " << sz.rB << "     " << sz.cB
                  << "   Transposed-B       " << transRes.mean << "       " << transRes.stddev << "\n";
    }
    return 0;
}
