#ifndef sharedMemoryEigen_h
#define sharedMemoryEigen_h


#include <Eigen/Dense>
#include <cmath>
#include <iostream> 
#include <iomanip>  

class sharedMemoryEigen {
public:
    // Constructor
    sharedMemoryEigen(int rows=30, int cols=8){
        rows_ = rows;
        cols_ = cols;
        team_memory_.resize(rows, cols); 
    }
    
    // Destructor
    // ~sharedMemoryEigen(){}

    // Function to print the entire team_memory_ matrix
    void printTeamMemory() const {
        std::cout << "Team Memory Matrix (" << rows_ << "x" << cols_ << "):\n";
        for (int i = 0; i < team_memory_.rows(); ++i) {
            for (int j = 0; j < team_memory_.cols(); ++j) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(2)
                          << team_memory_(i, j) << " ";
            }
            std::cout << "\n";
        }
    }
    
    // Function to set values in the Eigen matrix (optional, depending on needs)
    void setMatrixValue(int row, int col, double value) {
        if (row >= 0 && row < static_cast<int>(rows_) && 
            col >= 0 && col < static_cast<int>(cols_)) {
            team_memory_(row, col) = value;
        }
        else{
            cout << "ERROR: col " << col << "row " << row << endl;
        }
    }

    // Probability functions
    double memWriteProb_def(int i) const {
        return 0.25 - std::pow(0.01 * i, 2);
    }

    double memWriteProb_cauchy1(int i) const {
        return 1.0 / (M_PI * (std::pow(i, 2) + 1));
    }

    double memWriteProb_cauchyHalf(int i) const {
        return 0.25 / (0.5 * M_PI * (std::pow(i, 2) + 0.25));
    }
    
    Eigen::MatrixXd team_memory_; // Pointer to Eigen Matrix to manage memory
    static const size_t NA_TYPE = 3;
    static const size_t SHARED_MEM_TYPE = 4;
    static const int NUM_SHARED_MEM_TYPES = 1;

    size_t rows_;
    size_t cols_;
};

#endif // sharedMemoryEigen_h
