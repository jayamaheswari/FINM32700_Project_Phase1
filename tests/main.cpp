#include <iostream>
#include <cstdlib>
#include "linalg.h"

int main() {
    int rowsA = 2, colsA = 3, rowsB = 3, colsB = 2;
    double* A = new double[rowsA * colsA];
    double* B = new double[rowsB * colsB];
    double* B_t = new double[colsB * rowsB];
    double* v = new double[colsA];
    double* r_mv_row = new double[rowsA];
    double* r_mv_col = new double[rowsA];
    double* r_mm_naive = new double[rowsA * colsB];
    double* r_mm_transposed = new double[rowsA * colsB];

    for (int i = 0; i < rowsA * colsA; i++)
        A[i] = i + 1.0;
    for (int i = 0; i < rowsB * colsB; i++)
        B[i] = (i + 1.0) * 0.1;
    for (int i = 0; i < colsA; i++)
        v[i] = i + 1.0;
    for (int i = 0; i < colsB * rowsB; i++)
        B_t[i] = 0.0;
    for (int i = 0; i < rowsA; i++)
        r_mv_row[i] = r_mv_col[i] = 0.0;
    for (int i = 0; i < rowsA * colsB; i++)
        r_mm_naive[i] = r_mm_transposed[i] = 0.0;

    for (int i = 0; i < rowsB; i++) {
        for (int j = 0; j < colsB; j++) {
            B_t[j * rowsB + i] = B[i * colsB + j];
        }
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
