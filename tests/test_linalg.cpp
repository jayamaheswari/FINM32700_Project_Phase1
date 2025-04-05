#include <iostream>
#include <cstdlib>
#include <cmath>
#include "linalg.hpp"

static bool double_equals(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) < eps;
}

int main() {
    using namespace linalg;

    int rowsA = 2, colsA = 3, rowsB = 3, colsB = 2;

    double* A = new double[rowsA * colsA];
    double* B = new double[rowsB * colsB];
    double* B_t = new double[rowsB * colsB];
    double* v = new double[colsA];
    double* r_mv_row = new double[rowsA];
    double* r_mv_col = new double[rowsA];
    double* r_mm_naive = new double[rowsA * colsB];
    double* r_mm_transposed = new double[rowsA * colsB];

    for (int i = 0; i < rowsA * colsA; i++) {
        A[i] = i + 1.0;
    }
    for (int i = 0; i < rowsB * colsB; i++) {
        B[i] = (i + 1.0) * 0.1;
    }
    for (int i = 0; i < colsA; i++) {
        v[i] = i + 1.0;
    }
    for (int i = 0; i < rowsB * colsB; i++) {
        B_t[i] = 0.0;
    }
    for (int i = 0; i < rowsA; i++) {
        r_mv_row[i] = r_mv_col[i] = 0.0;
    }
    for (int i = 0; i < rowsA * colsB; i++) {
        r_mm_naive[i] = r_mm_transposed[i] = 0.0;
    }

    for (int i = 0; i < rowsB; i++) {
        for (int j = 0; j < colsB; j++) {
            B_t[j * rowsB + i] = B[i * colsB + j];
        }
    }

    std::cout << "Array A:" << std::endl;
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsA; j++) {
            std::cout << A[i * colsA + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Array B:" << std::endl;
    for (int i = 0; i < rowsB; i++) {
        for (int j = 0; j < colsB; j++) {
            std::cout << B[i * colsB + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Array v:" << std::endl;
    for (int i = 0; i < colsA; i++) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Array B_t (transposed):" << std::endl;
    for (int i = 0; i < colsB; i++) {
        for (int j = 0; j < rowsB; j++) {
            std::cout << B_t[i * rowsB + j] << " ";
        }
        std::cout << std::endl;
    }



    multiply_mv_row_major(A, rowsA, colsA, v, r_mv_row);
    multiply_mv_col_major(A, rowsA, colsA, v, r_mv_col);
    multiply_mm_naive(A, rowsA, colsA, B, rowsB, colsB, r_mm_naive);
    multiply_mm_transposed_b(A, rowsA, colsA, B_t, rowsB, colsB, r_mm_transposed);

    std::cout << "Row-Major MV:\n";
    for (int i = 0; i < rowsA; i++)
        std::cout << r_mv_row[i] << " ";
    std::cout << "\nColumn-Major MV:\n";
    for (int i = 0; i < rowsA; i++)
        std::cout << r_mv_col[i] << " ";
    std::cout << "\nNaive MM:\n";
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            std::cout << r_mm_naive[i * colsB + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "Transposed-B MM:\n";
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            std::cout << r_mm_transposed[i * colsB + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "Row-Major MV:\n";
    for (int i = 0; i < rowsA; i++) {
        std::cout << r_mv_row[i] << " ";
    }
    std::cout << "\n\nColumn-Major MV:\n";
    for (int i = 0; i < rowsA; i++) {
        std::cout << r_mv_col[i] << " ";
    }
    std::cout << "\n\nNaive MM:\n";
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            std::cout << r_mm_naive[i * colsB + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\nTransposed-B MM:\n";
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            std::cout << r_mm_transposed[i * colsB + j] << " ";
        }
        std::cout << "\n";
    }

    delete[] A;
    delete[] B;
    delete[] B_t;
    delete[] v;
    delete[] r_mv_row;
    delete[] r_mv_col;
    delete[] r_mm_naive;
    delete[] r_mm_transposed;

    return 0;
}
