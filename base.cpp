#include <iostream>
#include <stdexcept>
#include <chrono>

using namespace std;
using std::chrono::microseconds;
using std::invalid_argument;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

void print_vector(const double* vec, int size) {
    for (int i = 0; i < size; ++i) {
        cout << vec[i] << " ";
    }
    cout << endl;
}

void print_matrix(const double* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << matrix[i * cols + j] << " ";
        }
        cout << endl;
    }
}

// Matrix-vector multiplication (row-major order)
void multiply_mv_row_major(const double* matrix, int rows, int cols, const double* vector, double* result) {
    if (matrix == nullptr || vector == nullptr || result == nullptr) {
        throw invalid_argument("Null pointer passed to the function.");
    }
    if (cols <= 0 || rows <= 0) {
        throw invalid_argument("Invalid matrix dimensions.");
    }
    std::fill(result, result + rows, 0.0); 
    for (int i = 0; i < rows; ++i) {
        // result[i] = 0.0;
        for (int j = 0; j < cols; ++j) {
            result[i] += matrix[i * cols + j] * vector[j];  // Row-major access pattern
        }
    }
}
// Matrix-vector multiplication (column-major order)
void multiply_mv_col_major(const double* matrix, int rows, int cols, const double* vector, double* result) {
    if (matrix == nullptr || vector == nullptr || result == nullptr) {
        throw invalid_argument("Null pointer passed to the function.");
    }
    if (cols <= 0 || rows <= 0) {
        throw invalid_argument("Invalid matrix dimensions.");
    }
    std::fill(result, result + rows, 0.0); 
    for (int j = 0; j < cols; j++) {        
        for (int i = 0; i < rows; i++) {
            result[i] += matrix[j * rows + i] * vector[j];
        }
    }
}
int main() {
    long long rowsA = 1024, colsA = 1024;

    double* matrixA = new double[rowsA * colsA];
    double* result = new double[rowsA];
    double* vector = new double[colsA];

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsA; ++j) {
            matrixA[i * colsA + j] = (i + 1) * (j + 1);
        }
    }

    for (int i = 0; i < colsA; ++i) {
        vector[i] = i + 1;
    }

    // Measure time for Matrix-Vector multiplication (Row-Major)
    auto start = std::chrono::high_resolution_clock::now();
    multiply_mv_row_major(matrixA, rowsA, colsA, vector, result);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    cout << "Matrix-Vector multiplication (Row-Major) took: " << duration.count() << " milliseconds.\n";

    // Measure time for Matrix-Vector multiplication (Column-Major)
    start = std::chrono::high_resolution_clock::now();
    multiply_mv_col_major(matrixA, rowsA, colsA, vector, result);
    stop = std::chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    cout << "Matrix-Vector multiplication (Column-Major) took: " << duration.count() << " milliseconds.\n";

    // Clean up dynamically allocated memory
    delete[] matrixA;
    delete[] result;
    delete[] vector;
    return 0;
}
