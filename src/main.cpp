#include <iostream>
#include <omp.h>
#include <vector>
#include <chrono>
#include <random>

using Matrix = std::vector<std::vector<int>>;

// Função para transpor uma matriz
Matrix transposeMatrix(const Matrix& mat) {
    int rows = mat.size();
    int cols = mat[0].size();
    Matrix transposed(cols, std::vector<int>(rows));
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transposed[j][i] = mat[i][j];
        }
    }
    return transposed;
}

// Multiplicação sequencial otimizada
void multiplyMatrices(const Matrix& mat1, const Matrix& mat2T, Matrix& result, double& result_time) {
    auto start = std::chrono::high_resolution_clock::now();
    
    int rows1 = mat1.size();
    int cols1 = mat1[0].size();
    int cols2 = mat2T.size();
    
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            int sum = 0;
            for (int k = 0; k < cols1; k++) {
                sum += mat1[i][k] * mat2T[j][k];
            }
            result[i][j] = sum;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    result_time = std::chrono::duration<double>(end - start).count();
}


void parallelMultiplyMatrices(const Matrix& mat1, const Matrix& mat2T, Matrix& result, double& result_time) {
    auto start = std::chrono::high_resolution_clock::now();
    
    int rows1 = mat1.size();
    int cols1 = mat1[0].size();
    int cols2 = mat2T.size();
    
    #pragma omp parallel for collapse(2) schedule(dynamic, 16)
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            int sum = 0;
            for (int k = 0; k < cols1; k++) {
                sum += mat1[i][k] * mat2T[j][k];
            }
            result[i][j] = sum;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    result_time = std::chrono::duration<double>(end - start).count();
}

// Função para preencher matriz com valores aleatórios
void fillRandomMatrix(Matrix& mat) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 9);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[0].size(); j++) {
            mat[i][j] = distrib(gen);
        }
    }
}

int main() {
    const int rows1 = 1000, cols1 = 1000, cols2 = 1000;
    
    // Alocação dinâmica com vetores
    Matrix mat1(rows1, std::vector<int>(cols1));
    Matrix mat2(cols1, std::vector<int>(cols2));
    Matrix result(rows1, std::vector<int>(cols2));
    
    // Preencher matrizes com valores aleatórios
    fillRandomMatrix(mat1);
    fillRandomMatrix(mat2);
    
    // Transpor a segunda matriz
    Matrix mat2T = transposeMatrix(mat2);
    
    double sequential_time, parallel_time;
    
    // Teste sequencial
    multiplyMatrices(mat1, mat2T, result, sequential_time);
    std::cout << "Tempo sequencial: " << sequential_time << " s\n";
    
    // Teste paralelo
    omp_set_num_threads(omp_get_num_procs());
    parallelMultiplyMatrices(mat1, mat2T, result, parallel_time);
    std::cout << "Tempo paralelo: " << parallel_time << " s\n";
    
    // Cálculo do speedup
    std::cout << "Speedup: " << (sequential_time / parallel_time) << "\n";
    
    return 0;
}