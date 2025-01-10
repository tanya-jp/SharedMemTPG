#ifndef sharedMemoryEigen_h
#define sharedMemoryEigen_h

#include <Eigen/Dense>
#include <vector>
#include "MemoryEigen.h"

class sharedMemoryEigen
{
private:
    /* data */
public:

    // Vector of MemoryEigen objects
    std::vector<MemoryEigen *> team_scalar_memory_;
    std::vector<MemoryEigen *> team_vector_memory_;
    std::vector<MemoryEigen *> team_matrix_memory_;

    static const size_t NA_TYPE = 3;
    static const size_t SHARED_MEM_TYPE = 4;
    static const int NUM_SHARED_MEM_TYPES = 1;

    size_t rows_;
    size_t cols_;

    sharedMemoryEigen(int rows=10, int cols=8){
        rows_ = rows;
        cols_ = cols;

        // // Add scalar, vector, and matrix MemoryEigen
        // MemoryEigen scalarMemory(MemoryEigen::kScalarType_, 1, 1);
        // MemoryEigen vectorMemory(MemoryEigen::kVectorType_, 3, 3); // 3 indices, 3x1 vectors
        // MemoryEigen matrixMemory(MemoryEigen::kMatrixType_, 2, 2); // 2 indices, 2x2 matrices

        // team_scalar_memory_.push_back(scalarMemory);
        

        // Add to memories vectors
        for (int i=0; i<= 8; i++){

            team_scalar_memory_.push_back(
                new MemoryEigen (MemoryEigen::kScalarType_, 1, 1));
            team_vector_memory_.push_back(
                new MemoryEigen (MemoryEigen::kVectorType_, 3, 3));
            team_matrix_memory_.push_back(
                new MemoryEigen (MemoryEigen::kMatrixType_, 2, 2));
        }
    }


    // ~sharedMemoryEigen();

    // // Add a new MemoryEigen object to the vector
    // void AddMemory(const MemoryEigen& memory) {
    //     team_memory_.push_back(memory);
    // }

    // // Clear all memories
    // void ClearMemories() {
    //     team_scalar_memory_.clear();
    //     team_vector_memory_.clear();
    //     team_matirx_memory_.clear();
    // }

    // // Get a reference to a specific MemoryEigen object
    // MemoryEigen& GetMemory(size_t index) {
    //     if (index >= team_memory_.size()) {
    //         throw std::out_of_range("Index out of range in GetMemory.");
    //     }
    //     return team_memory_[index];
    // }

    // // Print details of all MemoryEigen objects
    // void PrintMemories() const {
    //     for (size_t i = 0; i < team_memory_.size(); ++i) {
    //         std::cout << "Memory " << i << ":\n";
    //         std::cout << team_memory_[i].ToString(0) << "\n"; // Assuming prog_id = 0 for demo
    //     }
    // }
};

#endif // sharedMemoryEigen_h