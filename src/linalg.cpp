#include "linalg.hpp"
#include <stdexcept>

namespace linalg {

void multiply_mv_row_major(const double* matrix,  int rows, int cols, const double* vector, double* result) {
    if (!matrix || !vector || !result || rows < 1 || cols < 1) {
        throw std::invalid_argument("Invalid arguments in multiply_mv_row_major.");
    }

    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += matrix[i * cols + j] * vector[j];
        }
        result[i] = sum;
    }
}

void multiply_mv_col_major(const double* matrix,  int rows, int cols, const double* vector, double* result) {
    if (!matrix || !vector || !result || rows < 1 || cols < 1) {
        throw std::invalid_argument("Invalid arguments in multiply_mv_col_major.");
    }
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += matrix[j * rows + i] * vector[j];
        }
        result[i] = sum;
    }
}

void multiply_mm_naive(const double* matrixA,  int rowsA, int colsA, const double* matrixB, int rowsB, int colsB, double* result)  {
    if (!matrixA || !matrixB || !result ||
        rowsA < 1 || colsA < 1 || rowsB < 1 || colsB < 1) {
        throw std::invalid_argument("Invalid arguments in multiply_mm_naive.");
    }
    if (colsA != rowsB) {
        throw std::invalid_argument("Dimension mismatch in multiply_mm_naive: colsA != rowsB.");
    }

    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            double sum = 0.0;
            for (int k = 0; k < colsA; k++) {
                sum += matrixA[i * colsA + k] * matrixB[k * colsB + j];
            }
            result[i * colsB + j] = sum;
        }
    }
}

void multiply_mm_transposed_b(const double* matrixA,  int rowsA, int colsA, const double* matrixB_transposed, int rowsB, int colsB, double* result)  {
    if (!matrixA || !matrixB_transposed || !result ||
        rowsA < 1 || colsA < 1 || rowsB < 1 || colsB < 1) {
        throw std::invalid_argument("Invalid arguments in multiply_mm_transposed_b.");
    }
    if (colsA != rowsB) {
        throw std::invalid_argument("Dimension mismatch in multiply_mm_transposed_b: colsA != rowsB.");
    }

    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            double sum = 0.0;
            for (int k = 0; k < colsA; k++) {
                sum += matrixA[i * colsA + k] * matrixB_transposed[j * rowsB + k];
            }
            result[i * colsB + j] = sum;
        }
    }
}

}
