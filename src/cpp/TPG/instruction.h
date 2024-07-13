#ifndef instruction_h
#define instruction_h

#include <bitset>
#include <iomanip>
#include <random>
#include <vector>

#include "memoryEigen.h"

class instruction {
 public:
  // Operations
  static const int SCALAR_SUM_OP_ = 0;
  static const int SCALAR_DIFF_OP_ = 1;
  static const int SCALAR_PRODUCT_OP_ = 2;
  static const int SCALAR_DIVISION_OP_ = 3;
  static const int SCALAR_ABS_OP_ = 4;
  static const int SCALAR_RECIPROCAL_OP_ = 5;
  static const int SCALAR_SIN_OP_ = 6;
  static const int SCALAR_COS_OP_ = 7;
  static const int SCALAR_TAN_OP_ = 8;
  static const int SCALAR_ARCSIN_OP_ = 9;
  static const int SCALAR_ARCCOS_OP_ = 10;
  static const int SCALAR_ARCTAN_OP_ = 11;
  static const int SCALAR_EXP_OP_ = 12;
  static const int SCALAR_LOG_OP_ = 13;
  static const int SCALAR_HEAVYSIDE_OP_ = 14;
  static const int VECTOR_HEAVYSIDE_OP_ = 15;
  static const int MATRIX_HEAVYSIDE_OP_ = 16;
  static const int SCALAR_VECTOR_PRODUCT_OP_ = 17;
  static const int SCALAR_BROADCAST_OP_ = 18;
  static const int VECTOR_RECIPROCAL_OP_ = 19;
  static const int VECTOR_NORM_OP_ = 20;
  static const int VECTOR_ABS_OP_ = 21;
  static const int VECTOR_SUM_OP_ = 22;
  static const int VECTOR_DIFF_OP_ = 23;
  static const int VECTOR_PRODUCT_OP_ = 24;
  static const int VECTOR_DIVISION_OP_ = 25;
  static const int VECTOR_INNER_PRODUCT_OP_ = 26;
  static const int VECTOR_OUTER_PRODUCT_OP_ = 27;
  static const int SCALAR_MATRIX_PRODUCT_OP_ = 28;
  static const int MATRIX_RECIPROCAL_OP_ = 29;
  static const int MATRIX_VECTOR_PRODUCT_OP_ = 30;
  static const int VECTOR_COLUMN_BROADCAST_OP_ = 31;
  static const int VECTOR_ROW_BROADCAST_OP_ = 32;
  static const int MATRIX_NORM_OP_ = 33;
  static const int MATRIX_COLUMN_NORM_OP_ = 34;
  static const int MATRIX_ROW_NORM_OP_ = 35;
  static const int MATRIX_TRANSPOSE_OP_ = 36;
  static const int MATRIX_ABS_OP_ = 37;
  static const int MATRIX_SUM_OP_ = 38;
  static const int MATRIX_DIFF_OP_ = 39;
  static const int MATRIX_PRODUCT_OP_ = 40;
  static const int MATRIX_DIVISION_OP_ = 41;
  static const int MATRIX_MATRIX_PRODUCT_OP_ = 42;
  static const int SCALAR_MIN_OP_ = 43;
  static const int VECTOR_MIN_OP_ = 44;
  static const int MATRIX_MIN_OP_ = 45;
  static const int SCALAR_MAX_OP_ = 46;
  static const int VECTOR_MAX_OP_ = 47;
  static const int MATRIX_MAX_OP_ = 48;
  static const int VECTOR_MEAN_OP_ = 49;
  static const int MATRIX_MEAN_OP_ = 50;
  static const int MATRIX_ROW_MEAN_OP_ = 51;
  static const int MATRIX_ROW_ST_DEV_OP_ = 52;
  static const int VECTOR_ST_DEV_OP_ = 53;
  static const int MATRIX_ST_DEV_OP_ = 54;
  static const int SCALAR_CONST_SET_OP_ = 55;
  static const int VECTOR_CONST_SET_OP_ = 56;
  static const int MATRIX_CONST_SET_OP_ = 57;
  static const int SCALAR_UNIFORM_SET_OP_ = 58;
  static const int VECTOR_UNIFORM_SET_OP_ = 59;
  static const int MATRIX_UNIFORM_SET_OP_ = 60;
  static const int SCALAR_GAUSSIAN_SET_OP_ = 61;
  static const int VECTOR_GAUSSIAN_SET_OP_ = 62;
  static const int MATRIX_GAUSSIAN_SET_OP_ = 63;
  static const int SCALAR_COND_A_OP_ = 64;
  static const int SCALAR_COND_B_OP_ = 65;
  static const int SCALAR_POW_OP_ = 66;
  static const int SCALAR_SQR_OP_ = 67;
  static const int SCALAR_CUBE_OP_ = 68;
  static const int SCALAR_TANH_OP_ = 69;
  static const int SCALAR_SQRT_OP_ = 70;
  static const int SCALAR_VECTOR_ASSIGN_OP_ = 71;
  static const int SCALAR_MATRIX_ASSIGN_OP_ = 72;

  static const int NUM_OP = 73;

  static const vector<double> constants_;
  mt19937 rng_;

  // Mutable instruction parameters

  // Whether in1 is a memory or input reference
  // 0: memory ref
  // 1: input ref
  int in1Src_ = 0;

  // Whether in2 is a memory or input reference
  // 0: memory ref
  // 1: input ref
  int in2Src_ = 0;

  // Which memory index does this instruction write to
  int outIdx_ = 0;

  // Which operation does this instrcution execute
  int op_ = 0;

  // TODO(skelly): update comment
  // in1/in2 index to memory or input buffer.
  // For memories, this index specifies which memory to use.
  // For input, this index specifies which timestep in the buffer to use, where
  // 0 is the current observation, 1 is the previous observation, etc.
  int in1Idx_ = 0;
  int in2Idx_ = 0;
  int in3Idx_ = 0;

  // The number of indices of each memory type (scalar, vector, matrix)
  int memIndices_ = 1;

  // Pointers to i/o for this instruction
  memoryEigen* out_;
  memoryEigen* in1_;
  memoryEigen* in2_;

  // Dimensionality of vector and martrix memory
  // Vector memories will be size memory_size_ x 1
  // Matrices will be size memory_size_ x memory_size_
  int memory_size_ = 0;

  // Maps operations to memory types for {out, in1, in2}
  // Each operation requires unique i/o memory types. For example:
  //  op_mem_types_[SCALAR_SUM_OP_] = {SCALAR_TYPE, SCALAR_TYPE, SCALAR_TYPE};
  //  op_mem_types_[SCALAR_COS_OP_] = {SCALAR_TYPE, SCALAR_TYPE, NA_TYPE};
  // NA_TYPE is a place holder for ops which only require one input (in1).
  static vector<vector<size_t> > op_mem_types_;

  string checkpoint();

  // Constructor
  instruction(std::unordered_map<string, std::any>&, mt19937&);

  // Copy COnstructor
  instruction(instruction&);

  // op_list_ maps each operation to a function pointer for its execution
  typedef void (instruction::*operation)(bool);
  static vector<operation> op_list_;

  // Execute this instruction
  inline void exec(bool dbg) {
    (this->*op_list_[op_])(dbg);

    // Change infinite values to 0.0 in output memory
    // Will get a lot of nan without this
    out_->working_memory_[outIdx_].array() =
        out_->working_memory_[outIdx_].array().unaryExpr(
            [](double v) { return std::isfinite(v) ? v : 0.0; });

    // TODO(skelly): tmp debugging output
    if (dbg) cerr << out_->working_memory_[outIdx_](0, 0) << endl;
  }

  inline int GetInIdx(int i) const {
    if (i == 0)
      return in1Idx_;
    else if (i == 2)
      return in2Idx_;
    else
      return in3Idx_;
  }
  inline void SeInIdx(int i, int idx) {
    if (i == 0)
      in1Idx_ = idx;
    else if (i == 1)
      in2Idx_ = idx;
    else
      in3Idx_ = idx;
  }
  inline memoryEigen* GetInMem(int i) const { return i == 0 ? in1_ : in2_; }
  inline void SetInMem(int i, memoryEigen* m) { (i == 0 ? in1_ : in2_) = m; }
  inline size_t GetInType(int i) const { return op_mem_types_[op_][i + 1]; }
  inline bool IsInput(int i) const {
    return (i == 0 ? in1Src_ == 1 : in2Src_ == 1) &&
           GetInType(i) != memoryEigen::NA_TYPE;
  }
  inline bool IsMemoryRef(int i) const {
    return !IsInput(i) && GetInType(i) != memoryEigen::NA_TYPE;
  }
  void Mutate(bool, vector<bool>&, mt19937&);
  inline size_t GetOutType() const { return op_mem_types_[op_][0]; }
  // inline void rng(mt19937& r) { rng_ = r; }
  static void SetupOps();

  /* Operation implementations ************************************************/

  inline void ExecuteScalarSumOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) = 
        in1_->working_memory_[in1Idx_](0, 0) +
         in2_->working_memory_[in2Idx_](0, 0);
    if (dbg) {
      cerr << std::setprecision(std::numeric_limits<double>::digits10 + 1)
           << std::fixed << "s" << outIdx_ << " = s" << in1Idx_ << " + " << "s"
           << in2Idx_ << " | ";
      cerr << std::fixed << in1_->working_memory_[in1Idx_](0, 0) << " + "
           << in2_->working_memory_[in2Idx_](0, 0) << " = "
           << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarDiffOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        in1_->working_memory_[in1Idx_](0, 0) -
        in2_->working_memory_[in2Idx_](0, 0);
    if (dbg) {
      cerr << std::setprecision(std::numeric_limits<double>::digits10 + 1)
           << std::fixed << "s" << outIdx_ << " = s" << in1Idx_ << " - " << "s"
           << in2Idx_ << " | ";
      cerr << std::fixed << in1_->working_memory_[in1Idx_](0, 0) << " - "
           << in2_->working_memory_[in2Idx_](0, 0) << " = "
           << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarProductOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        in1_->working_memory_[in1Idx_](0, 0) *
        in2_->working_memory_[in2Idx_](0, 0);
    if (dbg) {
      cerr << std::setprecision(std::numeric_limits<double>::digits10 + 1)
           << std::fixed << "s" << outIdx_ << " = s" << in1Idx_ << " * " << "s"
           << in2Idx_ << " | ";
      cerr << std::fixed << in1_->working_memory_[in1Idx_](0, 0) << " * "
           << in2_->working_memory_[in2Idx_](0, 0) << " = "
           << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarDivisionOp(bool dbg) {
    // Protected division
    if (isEqual(in2_->working_memory_[in2Idx_](0, 0), 0.0)) {
      out_->working_memory_[outIdx_](0, 0) = 0;
    } else {
      out_->working_memory_[outIdx_](0, 0) =
          in1_->working_memory_[in1Idx_](0, 0) /
          in2_->working_memory_[in2Idx_](0, 0);
    }
    if (dbg) {
      cerr << std::setprecision(std::numeric_limits<double>::digits10 + 1)
           << std::fixed << "s" << outIdx_ << " = s" << in1Idx_ << " / " << "s"
           << in2Idx_ << " | ";
      cerr << std::fixed << in1_->working_memory_[in1Idx_](0, 0) << " / "
           << in2_->working_memory_[in2Idx_](0, 0) << " = "
           << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarReciprocalOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        1.0 / in1_->working_memory_[in1Idx_](0, 0);
    if (dbg) {
    }
  }

  inline void ExecuteScalarAbsOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::abs(in1_->working_memory_[in1Idx_](0, 0));
    if (dbg) {
      cerr << "s" << outIdx_ << " = abs(s" << in1Idx_ << ") | ";
      cerr << "abs(" << in1_->working_memory_[in1Idx_](0, 0)
           << ") = " << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarSinOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::sin(in1_->working_memory_[in1Idx_](0, 0));
    if (dbg) {
      cerr << "s" << outIdx_ << " = sin(s" << in1Idx_ << ") | ";
      cerr << "sin(" << in1_->working_memory_[in1Idx_](0, 0)
           << ") = " << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarCosOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::cos(in1_->working_memory_[in1Idx_](0, 0));
    if (dbg) {
      cerr << "s" << outIdx_ << " = cos(s" << in1Idx_ << ") | ";
      cerr << "cos(" << in1_->working_memory_[in1Idx_](0, 0)
           << ") = " << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarTanOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::tan(in1_->working_memory_[in1Idx_](0, 0));
    if (dbg) {
      cerr << "s" << outIdx_ << " = tan(s" << in1Idx_ << ") | ";
      cerr << "tan(" << in1_->working_memory_[in1Idx_](0, 0)
           << ") = " << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarExpOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::exp(in1_->working_memory_[in1Idx_](0, 0));
    if (dbg) {
      cerr << "s" << outIdx_ << " = exp(s" << in1Idx_ << ") | ";
      cerr << "exp(" << in1_->working_memory_[in1Idx_](0, 0)
           << ") = " << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarLogOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::log(std::abs(in1_->working_memory_[in1Idx_](0, 0)));
    if (dbg) {
      cerr << "s" << outIdx_ << " = log(s" << in1Idx_ << ") | ";
      cerr << "log(" << in1_->working_memory_[in1Idx_](0, 0)
           << ") = " << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarArcSinOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::asin(in1_->working_memory_[in1Idx_](0, 0));
    if (dbg) {
      cerr << "s" << outIdx_ << " = asin(s" << in1Idx_ << ") | ";
      cerr << "asin(" << in1_->working_memory_[in1Idx_](0, 0)
           << ") = " << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarArcCosOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::acos(in1_->working_memory_[in1Idx_](0, 0));
    if (dbg) {
      cerr << "s" << outIdx_ << " = acos(s" << in1Idx_ << ") | ";
      cerr << "acos(" << in1_->working_memory_[in1Idx_](0, 0)
           << ") = " << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarArcTanOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::atan(in1_->working_memory_[in1Idx_](0, 0));
    if (dbg) {
      cerr << "s" << outIdx_ << " = atan(s" << in1Idx_ << ") | ";
      cerr << "atan(" << in1_->working_memory_[in1Idx_](0, 0)
           << ") = " << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarHeavisideOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        in1_->working_memory_[in1Idx_](0, 0) >= 0.0 ? 1.0 : 0.0;
    if (dbg) {
    }
  }

  inline void ExecuteVectorHeavisideOp(bool dbg) {
    const double* in = out_->working_memory_[in1Idx_].data();
    const double* in_end = in + memory_size_;
    double* out = out_->working_memory_[outIdx_].data();
    while (in != in_end) {
      *out = *in > 0.0 ? 1.0 : 0.0;
      ++out;
      ++in;
    }
    if (dbg) {
    }
  }

  inline void ExecuteMatrixHeavisideOp(bool dbg) {
    const double* ind = in1_->working_memory_[in1Idx_].data();
    const double* ind_end =
        ind + memory_size_ * memory_size_;  // all matices will be same size
    double* outd = out_->working_memory_[outIdx_].data();
    while (ind != ind_end) {
      *outd = *ind > 0.0 ? 1.0 : 0.0;
      ++outd;
      ++ind;
    }
    if (dbg) {
    }
  }

  inline void ExecuteScalarVectorProductOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1Idx_](0, 0) * in2_->working_memory_[in2Idx_];
    if (dbg) {
    }
  }

  inline void ExecuteScalarBroadcastOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1Idx_](0, 0) * MatrixXd::Ones(memory_size_, 1);
    if (dbg) {
    }
  }

  inline void ExecuteVectorReciprocalOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        1.0 / in1_->working_memory_[in1Idx_](0, 0);
    if (dbg) {
    }
  }

  inline void ExecuteVectorNormOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        in1_->working_memory_[in1Idx_].norm();
    if (dbg) {
    }
  }

  inline void ExecuteVectorAbsOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        (in1_->working_memory_[in1Idx_].array().abs()).matrix();
    if (dbg) {
    }
  }

  inline void ExecuteVectorSumOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1Idx_] + in2_->working_memory_[in2Idx_];
    if (dbg) {
    }
  }

  inline void ExecuteVectorDiffOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1Idx_] - in2_->working_memory_[in2Idx_];
    if (dbg) {
    }
  }

  inline void ExecuteVectorProductOp(bool dbg) {
    out_->working_memory_[outIdx_] = in1_->working_memory_[in1Idx_].array() *
                                     in2_->working_memory_[in2Idx_].array();
    if (dbg) {
    }
  }

  inline void ExecuteVectorDivisionOp(bool dbg) {
    out_->working_memory_[outIdx_] = in1_->working_memory_[in1Idx_].array() /
                                     in2_->working_memory_[in2Idx_].array();
    if (dbg) {
    }
  }

  inline void ExecuteVectorInnerProductOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        in1_->working_memory_[in1Idx_].col(0).dot(
            in2_->working_memory_[in2Idx_].col(0));
    if (dbg) {
    }
  }

  inline void ExecuteVectorOuterProductOp(bool dbg) {
    out_->working_memory_[outIdx_] = in1_->working_memory_[in1Idx_] *
                                     in2_->working_memory_[in2Idx_].transpose();
    if (dbg) {
    }
  }

  inline void ExecuteScalarMatrixProductOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1Idx_](0, 0) * in2_->working_memory_[in2Idx_];
    if (dbg) {
    }
  }

  inline void ExecuteMatrixReciprocalOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        (1.0 / in1_->working_memory_[in1Idx_].array()).matrix();
    if (dbg) {
    }
  }

  inline void ExecuteMatrixVectorProductOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1Idx_] * in2_->working_memory_[in2Idx_];
    if (dbg) {
    }
  }

  inline void ExecuteVectorColumnBroadcastOp(bool dbg) {
    out_->working_memory_[outIdx_] = in1_->working_memory_[in1Idx_].replicate(
        1, in1_->working_memory_[in1Idx_].rows());
    if (dbg) {
    }
  }

  inline void ExecuteVectorRowBroadcastOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1Idx_]
            .replicate(1, in1_->working_memory_[in1Idx_].rows())
            .transpose();
    if (dbg) {
    }
  }

  inline void ExecuteMatrixNormOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        in1_->working_memory_[in1Idx_].norm();
    if (dbg) {
    }
  }

  inline void ExecuteMatrixColumnNormOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1Idx_]
            .colwise()
            .norm()
            .transpose();  // automl-zero doesn't transpose?
    if (dbg) {
    }
  }

  inline void ExecuteMatrixRowNormOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1Idx_].rowwise().norm();
    if (dbg) {
    }
  }

  inline void ExecuteMatrixTransposeOp(bool dbg) {
    if (outIdx_ == in1Idx_)
      out_->working_memory_[outIdx_].transposeInPlace();
    else
      out_->working_memory_[outIdx_] =
          in1_->working_memory_[in1Idx_].transpose();
    if (dbg) {
    }
  }

  inline void ExecuteMatrixAbsOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1Idx_].array().abs().matrix();
    if (dbg) {
    }
  }

  inline void ExecuteMatrixSumOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1Idx_] + in2_->working_memory_[in2Idx_];
    if (dbg) {
    }
  }

  inline void ExecuteMatrixDiffOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1Idx_] - in2_->working_memory_[in2Idx_];
    if (dbg) {
    }
  }

  inline void ExecuteMatrixProductOp(bool dbg) {
    out_->working_memory_[outIdx_] = (in1_->working_memory_[in1Idx_].array() *
                                      in2_->working_memory_[in2Idx_].array())
                                         .matrix();
    if (dbg) {
    }
  }

  inline void ExecuteMatrixDivisionOp(bool dbg) {
    out_->working_memory_[outIdx_] = (in1_->working_memory_[in1Idx_].array() /
                                      in2_->working_memory_[in2Idx_].array())
                                         .matrix();
    if (dbg) {
    }
  }

  inline void ExecuteMatrixMatrixProductOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1Idx_] * in2_->working_memory_[in2Idx_];
    if (dbg) {
    }
  }

  inline void ExecuteScalarMinOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        min(in1_->working_memory_[in1Idx_](0, 0),
            in2_->working_memory_[in2Idx_](0, 0));
    if (dbg) {
    }
  }

  inline void ExecuteVectorMinOp(bool dbg) {
    out_->working_memory_[outIdx_] = in1_->working_memory_[in1Idx_].array().min(
        in2_->working_memory_[in2Idx_].array());
    if (dbg) {
    }
  }

  inline void ExecuteMatrixMinOp(bool dbg) {
    const double* in1d = in1_->working_memory_[in1Idx_].data();
    const double* in2d = in2_->working_memory_[in2Idx_].data();
    const double* in1_end = in1d + in2_->working_memory_[in2Idx_].rows() *
                                       in2_->working_memory_[in2Idx_].cols();
    double* outd = out_->working_memory_[outIdx_].data();
    while (in1d != in1_end) {
      const double in1v = *in1d;
      const double in2v = *in2d;
      *outd = in1v < in2v ? in1v : in2v;
      ++outd;
      ++in1d;
      ++in2d;
    }
    if (dbg) {
    }
  }

  inline void ExecuteScalarMaxOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        max(out_->working_memory_[outIdx_](0, 0),
            in1_->working_memory_[in1Idx_](0, 0));
    if (dbg) {
    }
  }

  inline void ExecuteVectorMaxOp(bool dbg) {
    out_->working_memory_[outIdx_] = in1_->working_memory_[in1Idx_].array().max(
        in2_->working_memory_[in2Idx_].array());
    if (dbg) {
    }
  }

  inline void ExecuteMatrixMaxOp(bool dbg) {
    const double* in1d = in1_->working_memory_[in1Idx_].data();
    const double* in2d = in2_->working_memory_[in2Idx_].data();
    const double* in1_end = in1d + in2_->working_memory_[in2Idx_].rows() *
                                       in2_->working_memory_[in2Idx_].cols();
    double* outd = out_->working_memory_[outIdx_].data();
    while (in1d != in1_end) {
      const double in1v = *in1d;
      const double in2v = *in2d;
      *outd = in1v > in2v ? in1v : in2v;
      ++outd;
      ++in1d;
      ++in2d;
    }
    if (dbg) {
    }
  }

  inline void ExecuteVectorMeanOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        in1_->working_memory_[in1Idx_].mean();
    if (dbg) {
    }
  }

  inline void ExecuteMatrixMeanOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        in1_->working_memory_[in1Idx_].mean();
    if (dbg) {
    }
  }

  inline void ExecuteMatrixRowMeanOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1Idx_].rowwise().mean();
    if (dbg) {
    }
  }

  inline void ExecuteMatrixRowStDevOp(bool dbg) {
    for (int row = 0; row < out_->working_memory_[outIdx_].rows(); ++row) {
      const MatrixXd values = in1_->working_memory_[in1Idx_].row(row);
      const double mean = values.mean();
      const double stdev =
          sqrt((values.array() * values.array()).sum() /
                   static_cast<double>(in1_->working_memory_[in1Idx_].rows()) -
               mean * mean);
      out_->working_memory_[outIdx_](row, 0) = stdev;
    }
    if (dbg) {
    }
  }

  inline void ExecuteVectorStDevOp(bool dbg) {
    const double mean = in1_->working_memory_[in1Idx_].mean();
    out_->working_memory_[outIdx_](0, 0) =
        sqrt(in1_->working_memory_[in1Idx_].col(0).dot(
                 in1_->working_memory_[in1Idx_].col(0)) /
                 in1_->working_memory_[in1Idx_].rows() -
             mean * mean);
    if (dbg) {
    }
  }

  inline void ExecuteMatrixStDevOp(bool dbg) {
    const MatrixXd values = in1_->working_memory_[in1Idx_];
    const double mean = values.mean();
    out_->working_memory_[outIdx_](0, 0) =
        sqrt((values.array() * values.array()).sum() /
                 static_cast<double>(out_->working_memory_[outIdx_].rows() *
                                     out_->working_memory_[outIdx_].cols()) -
             mean * mean);
    if (dbg) {
    }
  }

  inline void ExecuteScalarConstSetOp(bool dbg) {
    out_->working_memory_[outIdx_] = in1_->const_memory_[in1Idx_];
    if (dbg) {
    }
  }

  inline void ExecuteVectorConstSetOp(bool dbg) {
    out_->working_memory_[outIdx_] = in1_->const_memory_[in1Idx_];
    // out_->working_memory_[outIdx_] = constants_[in1Idx_ % constants_.size()]
    // *
    //  MatrixXd::Ones(memory_size_, 1);
    if (dbg) {
    }
  }

  inline void ExecuteMatrixConstSetOp(bool dbg) {
    out_->working_memory_[outIdx_] = in1_->const_memory_[in1Idx_];
    // out_->working_memory_[outIdx_] = constants_[in1Idx_ % constants_.size()]
    // *
    //  MatrixXd::Ones(memory_size_, memory_size_);
    if (dbg) {
    }
  }

  inline void ExecuteScalarUniformSetOp(bool dbg) {
    uniform_real_distribution<double> dis(in1_->working_memory_[in1Idx_](0, 0),
                                          in2_->working_memory_[in2Idx_](0, 0));
    out_->working_memory_[outIdx_](0, 0) = dis(rng_);
    if (dbg) {
    }
  }

  inline void ExecuteVectorUniformSetOp(bool dbg) {
    uniform_real_distribution<double> dis(in1_->working_memory_[in1Idx_](0, 0),
                                          in2_->working_memory_[in2Idx_](0, 0));
    for (int i = 0; i < out_->working_memory_[outIdx_].rows(); i++)
      out_->working_memory_[outIdx_](i, 0) = dis(rng_);
    if (dbg) {
    }
  }

  inline void ExecuteMatrixUniformSetOp(bool dbg) {
    uniform_real_distribution<double> dis(in1_->working_memory_[in1Idx_](0, 0),
                                          in2_->working_memory_[in2Idx_](0, 0));
    for (int row = 0; row < out_->working_memory_[outIdx_].rows(); row++)
      for (int col = 0; col < out_->working_memory_[outIdx_].cols(); col++)
        out_->working_memory_[outIdx_](row, col) = dis(rng_);
    if (dbg) {
    }
  }

  inline void ExecuteScalarGaussianSetOp(bool dbg) {
    normal_distribution<double> dis(in1_->working_memory_[in1Idx_](0, 0),
                                    in2_->working_memory_[in2Idx_](0, 0));
    out_->working_memory_[outIdx_](0, 0) = dis(rng_);
    if (dbg) {
    }
  }

  inline void ExecuteVectorGaussianSetOp(bool dbg) {
    normal_distribution<double> dis(in1_->working_memory_[in1Idx_](0, 0),
                                    in2_->working_memory_[in2Idx_](0, 0));
    for (int i = 0; i < out_->working_memory_[outIdx_].rows(); i++)
      out_->working_memory_[outIdx_](i, 0) = dis(rng_);
    if (dbg) {
    }
  }

  inline void ExecuteMatrixGaussianSetOp(bool dbg) {
    normal_distribution<double> dis(in1_->working_memory_[in1Idx_](0, 0),
                                    in2_->working_memory_[in2Idx_](0, 0));
    for (int row = 0; row < out_->working_memory_[outIdx_].rows(); row++)
      for (int col = 0; col < out_->working_memory_[outIdx_].cols(); col++)
        out_->working_memory_[outIdx_](row, col) = dis(rng_);
    if (dbg) {
    }
  }

  inline void ExecuteScalarCondAOp(bool dbg) {
    if (in1_->working_memory_[in1Idx_](0, 0) <
        in2_->working_memory_[in2Idx_](0, 0))
      out_->working_memory_[outIdx_](0, 0) =
          -(out_->working_memory_[outIdx_](0, 0));

    if (dbg) {
      cerr << " IF s" << in1Idx_ << " < s" << in2Idx_ << " THEN s" << outIdx_
           << " = -s" << outIdx_ << " | ";
      cerr << " in1 " << in1_->working_memory_[in1Idx_](0, 0) << " in2 "
           << in2_->working_memory_[outIdx_](0, 0) << " : "
           << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarCondBOp(bool dbg) {
    if (in1_->working_memory_[in1Idx_](0, 0) >=
        in2_->working_memory_[in2Idx_](0, 0))
      out_->working_memory_[outIdx_](0, 0) =
          -(out_->working_memory_[outIdx_](0, 0));
    if (dbg) {
    }
  }

  inline void ExecuteScalarPowOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::pow(in1_->working_memory_[in1Idx_](0, 0),
                 in2_->working_memory_[in2Idx_](0, 0));
    if (dbg) {
      cerr << "s" << outIdx_ << " = pow(s" << in1Idx_ << ", s" << in2Idx_
           << ") | ";
      cerr << "pow(" << in1_->working_memory_[in1Idx_](0, 0) << ", "
           << in2_->working_memory_[in2Idx_](0, 0) << " = "
           << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarSqrOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::pow(in1_->working_memory_[in1Idx_](0, 0), 2);
    if (dbg) {
    }
  }

  inline void ExecuteScalarCubeOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::pow(in1_->working_memory_[in1Idx_](0, 0), 3);
    if (dbg) {
    }
  }

  inline void ExecuteScalarTanhOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::tanh(in1_->working_memory_[in1Idx_](0, 0));
    if (dbg) {
    }
  }

  inline void ExecuteScalarSqrtOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::sqrt(in1_->working_memory_[in1Idx_](0, 0));
    if (dbg) {
    }
  }

  inline void ExecuteScalarVectorAssignOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        in1_->working_memory_[in1Idx_](in2Idx_, 0);
    if (dbg) {
    }
  }

  inline void ExecuteScalarMatrixAssignOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        in1_->working_memory_[in1Idx_](in2Idx_, in3Idx_);
    if (dbg) {
    }
  }
};
#endif
