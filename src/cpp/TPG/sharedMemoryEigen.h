#ifndef sharedMemoryEigen_h
#define sharedMemoryEigen_h

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "MemoryEigen.h"

class sharedMemoryEigen {
public:
    std::vector<std::unique_ptr<MemoryEigen>> team_scalar_memory_;
    std::vector<std::unique_ptr<MemoryEigen>> team_vector_memory_;
    std::vector<std::unique_ptr<MemoryEigen>> team_matrix_memory_;

    static const size_t NA_TYPE = 3;
    static const size_t SHARED_MEM_TYPE = 4;
    static const int NUM_SHARED_MEM_TYPES = 1;

    // void printVectorMemories();


    size_t rows_;
    size_t cols_;

    bool reset = false;

    sharedMemoryEigen(int n_memories = 8, int memory_size = 2) 
        : rows_(n_memories), cols_(n_memories) {
        // cols_ = 4;
        for (int i = 0; i <= static_cast<int>(cols_); i++) {
            team_scalar_memory_.emplace_back(std::make_unique<MemoryEigen>(MemoryEigen::kScalarType_, n_memories, 1, false));
            team_vector_memory_.emplace_back(std::make_unique<MemoryEigen>(MemoryEigen::kVectorType_, n_memories, memory_size, false));
            team_matrix_memory_.emplace_back(std::make_unique<MemoryEigen>(MemoryEigen::kMatrixType_, n_memories, memory_size, false));
        }
        // printMemory();
    }

    void printMemory() {
        std::cout << "Vector Memories:\n";
        for (const auto& mem : team_vector_memory_) {
            if (mem) {
                mem->printVectorMemory();  // Ensure `MemoryEigen` has this method
            }
        }
    }

    ~sharedMemoryEigen() = default;

    // Disable copy semantics for the class
    sharedMemoryEigen(const sharedMemoryEigen&) = delete;
    sharedMemoryEigen& operator=(const sharedMemoryEigen&) = delete;

    // Enable move semantics for efficiency
    sharedMemoryEigen(sharedMemoryEigen&&) = default;
    sharedMemoryEigen& operator=(sharedMemoryEigen&&) = default;
};

// #ifndef sharedMemoryEigen_h
// #define sharedMemoryEigen_h

// #include <Eigen/Dense>
// #include <vector>
// #include "MemoryEigen.h"

// class sharedMemoryEigen
// {
// private:
//     /* data */
// public:

//     // Vector of MemoryEigen objects
//     std::vector<MemoryEigen *> team_scalar_memory_;
//     std::vector<MemoryEigen *> team_vector_memory_;
//     std::vector<MemoryEigen *> team_matrix_memory_;

//     static const size_t NA_TYPE = 3;
//     static const size_t SHARED_MEM_TYPE = 4;
//     static const int NUM_SHARED_MEM_TYPES = 1;

//     size_t rows_;
//     size_t cols_;

//     sharedMemoryEigen(int n_memories=8, int memory_size=2){

//         rows_ = n_memories;
//         cols_ = n_memories;

//         // // Add scalar, vector, and matrix MemoryEigen
//         // MemoryEigen scalarMemory(MemoryEigen::kScalarType_, 1, 1);
//         // MemoryEigen vectorMemory(MemoryEigen::kVectorType_, 3, 3); // 3 indices, 3x1 vectors
//         // MemoryEigen matrixMemory(MemoryEigen::kMatrixType_, 2, 2); // 2 indices, 2x2 matrices

//         // team_scalar_memory_.push_back(scalarMemory);
        

//         // Add to memories vectors
//         for (int i=0; i<= static_cast<int>(cols_); i++){

//             // std::any_cast<int>(params["n_memories"])
//             team_scalar_memory_.push_back(
//                 new MemoryEigen (MemoryEigen::kScalarType_, n_memories, 1));
//             team_vector_memory_.push_back(
//                 new MemoryEigen (MemoryEigen::kVectorType_, n_memories, memory_size));
//             team_matrix_memory_.push_back(
//                 new MemoryEigen (MemoryEigen::kMatrixType_, n_memories, memory_size));
//         }
//     }


//     // ~sharedMemoryEigen();

//     // // Add a new MemoryEigen object to the vector
//     // void AddMemory(const MemoryEigen& memory) {
//     //     team_memory_.push_back(memory);
//     // }

//     // // Clear all memories
//     // void ClearMemories() {
//     //     team_scalar_memory_.clear();
//     //     team_vector_memory_.clear();
//     //     team_matirx_memory_.clear();
//     // }

//     // // Get a reference to a specific MemoryEigen object
//     // MemoryEigen& GetMemory(size_t index) {
//     //     if (index >= team_memory_.size()) {
//     //         throw std::out_of_range("Index out of range in GetMemory.");
//     //     }
//     //     return team_memory_[index];
//     // }

//     // // Print details of all MemoryEigen objects
//     // void PrintMemories() const {
//     //     for (size_t i = 0; i < team_memory_.size(); ++i) {
//     //         std::cout << "Memory " << i << ":\n";
//     //         std::cout << team_memory_[i].ToString(0) << "\n"; // Assuming prog_id = 0 for demo
//     //     }
//     // }
// };

#endif // sharedMemoryEigen_h
