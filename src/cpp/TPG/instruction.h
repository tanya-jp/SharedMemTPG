#ifndef instruction_h
#define instruction_h

#include <bitset>
#include <iomanip>
#include <random>
#include <vector>

#include "memoryEigen.h"

class instruction {
 public:
  /* Operations. */
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

  static const int NUM_OP = 71;

  static const vector<double> constants_;
  mt19937 rng_;

  // Mutable instruction parameters
  int in1Src_ = 0;  // 0,1,2 - private, shared, input (2024-04-29 limit to
                    // private or input)
  int in2Src_ = 0;  // 0,1,2 - private, shared, input (2024-04-29 limit to
                    // private or input)
  int outSrc_ = 0;  // 0,1 - private, shared (2024-04-29 limit to private)
  int outIdx_ = 0;  // index to memory
  int op_ = 0;      // operation
  // Index to memory or feature
  // Range: (0 -> memIndices - 1) sometimes moded by obs size
  int in1Idx_ =
      0;  // index to memory or feature (range: memory_size * memory_size )
  int in1IdxE_ = 0;  // idx for features stored in tmp memoryEign* TODO(skelly)
                     // what is this?
  int in2Idx_ = 0;   // index to memory or feature
  int in2IdxE_ = 0;  // idx for features stored in tmp memoryEign* TODO(skelly)
                     // what is this?

  int memIndices_ = 1;
  memoryEigen* out_;
  memoryEigen* in1_;
  memoryEigen* in2_;
  int memory_size_ = 0;

  // maps operations to memory types for out, in1, in2
  static vector<vector<size_t> > op_mem_types_;

  string checkpoint();
  instruction(std::unordered_map<string, std::any>&, mt19937&);
  instruction(instruction&);

  typedef void (instruction::*operation)(bool);
  static vector<operation> op_list_;
  inline void exec(bool dbg) {
    (this->*op_list_[op_])(dbg);

    // will get a lot of nan without this
    out_->working_memory_[outIdx_].array() =
        out_->working_memory_[outIdx_].array().unaryExpr(
            [](double v) { return std::isfinite(v) ? v : 0.0; });

    // TODO(skelly): remove this
    out_->working_memory_[outIdx_].array() =
        out_->working_memory_[outIdx_].array().unaryExpr(
            [](double v) { return isEqual(v, 0.0) ? 0.0 : v; });

    if (dbg) cerr << out_->working_memory_[outIdx_](0, 0) << endl;
  }

  inline int inIdx(int i) const { return i == 0 ? in1Idx_ : in2Idx_; }
  inline void inIdx(int i, int idx) { (i == 0 ? in1Idx_ : in2Idx_) = idx; }
  inline int inIdxE(int i) const { return i == 0 ? in1IdxE_ : in2IdxE_; }
  inline void inIdxE(int i, int idx) { (i == 0 ? in1IdxE_ : in2IdxE_) = idx; }
  inline memoryEigen* inMem(int i) const { return i == 0 ? in1_ : in2_; }
  inline void inMem(int i, memoryEigen* m) { (i == 0 ? in1_ : in2_) = m; }
  inline size_t inType(int i) const { return op_mem_types_[op_][i + 1]; }
  inline bool IsInput(int i) const {
    return (i == 0 ? in1Src_ == 2 : in2Src_ == 2) &&
           inType(i) != memoryEigen::NA_TYPE;
  }
  inline bool IsMemoryRef(int i) const {
    return !IsInput(i) && inType(i) != memoryEigen::NA_TYPE;
  }
  inline int memIndices() const { return memIndices_; }
  inline void memIndices(int i) { memIndices_ = i; }
  void mutate(bool, vector<bool>&, mt19937&);
  inline size_t outType() const { return op_mem_types_[op_][0]; }
  inline void rng(mt19937& r) { rng_ = r; }
  static void SetupOps();

  /* op implementations *******************************************************/

  inline void ExecuteScalarSumOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        (in1_->working_memory_[in1IdxE_](0, 0) +
         in2_->working_memory_[in2IdxE_](0, 0));
    if (dbg) {
      cerr << std::setprecision(std::numeric_limits<double>::digits10 + 1)
           << std::fixed << "s" << outIdx_ << " = s" << in1IdxE_ << " + " << "s"
           << in2IdxE_ << " | ";
      cerr << std::fixed << in1_->working_memory_[in1IdxE_](0, 0) << " + "
           << in2_->working_memory_[in2IdxE_](0, 0) << " = "
           << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarDiffOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        in1_->working_memory_[in1IdxE_](0, 0) -
        in2_->working_memory_[in2IdxE_](0, 0);
    if (dbg) {
      cerr << std::setprecision(std::numeric_limits<double>::digits10 + 1)
           << std::fixed << "s" << outIdx_ << " = s" << in1IdxE_ << " - " << "s"
           << in2IdxE_ << " | ";
      cerr << std::fixed << in1_->working_memory_[in1IdxE_](0, 0) << " - "
           << in2_->working_memory_[in2IdxE_](0, 0) << " = "
           << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarProductOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        in1_->working_memory_[in1IdxE_](0, 0) *
        in2_->working_memory_[in2IdxE_](0, 0);
    if (dbg) {
      cerr << std::setprecision(std::numeric_limits<double>::digits10 + 1)
           << std::fixed << "s" << outIdx_ << " = s" << in1IdxE_ << " * " << "s"
           << in2IdxE_ << " | ";
      cerr << std::fixed << in1_->working_memory_[in1IdxE_](0, 0) << " * "
           << in2_->working_memory_[in2IdxE_](0, 0) << " = "
           << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarDivisionOp(bool dbg) {
    // Protected division
    if (isEqual(in2_->working_memory_[in2IdxE_](0, 0), 0.0)) {
      out_->working_memory_[outIdx_](0, 0) = 0;
    } else {
      out_->working_memory_[outIdx_](0, 0) =
          in1_->working_memory_[in1IdxE_](0, 0) /
          in2_->working_memory_[in2IdxE_](0, 0);
    }
    if (dbg) {
      cerr << std::setprecision(std::numeric_limits<double>::digits10 + 1)
           << std::fixed << "s" << outIdx_ << " = s" << in1IdxE_ << " / " << "s"
           << in2IdxE_ << " | ";
      cerr << std::fixed << in1_->working_memory_[in1IdxE_](0, 0) << " / "
           << in2_->working_memory_[in2IdxE_](0, 0) << " = "
           << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarReciprocalOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        1.0 / in1_->working_memory_[in1IdxE_](0, 0);
    if (dbg) {
    }
  }

  inline void ExecuteScalarAbsOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::abs(in1_->working_memory_[in1IdxE_](0, 0));
    if (dbg) {
      cerr << "s" << outIdx_ << " = abs(s" << in1IdxE_ << ") | ";
      cerr << "abs(" << in1_->working_memory_[in1IdxE_](0, 0)
           << ") = " << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarSinOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::sin(in1_->working_memory_[in1IdxE_](0, 0));
    if (dbg) {
      cerr << "s" << outIdx_ << " = sin(s" << in1IdxE_ << ") | ";
      cerr << "sin(" << in1_->working_memory_[in1IdxE_](0, 0)
           << ") = " << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarCosOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::cos(in1_->working_memory_[in1IdxE_](0, 0));
    if (dbg) {
      cerr << "s" << outIdx_ << " = cos(s" << in1IdxE_ << ") | ";
      cerr << "cos(" << in1_->working_memory_[in1IdxE_](0, 0)
           << ") = " << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarTanOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::tan(in1_->working_memory_[in1IdxE_](0, 0));
    if (dbg) {
      cerr << "s" << outIdx_ << " = tan(s" << in1IdxE_ << ") | ";
      cerr << "tan(" << in1_->working_memory_[in1IdxE_](0, 0)
           << ") = " << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarExpOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::exp(in1_->working_memory_[in1IdxE_](0, 0));
    if (dbg) {
      cerr << "s" << outIdx_ << " = exp(s" << in1IdxE_ << ") | ";
      cerr << "exp(" << in1_->working_memory_[in1IdxE_](0, 0)
           << ") = " << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarLogOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::log(std::abs(in1_->working_memory_[in1IdxE_](0, 0)));
    if (dbg) {
      cerr << "s" << outIdx_ << " = log(s" << in1IdxE_ << ") | ";
      cerr << "log(" << in1_->working_memory_[in1IdxE_](0, 0)
           << ") = " << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarArcSinOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::asin(in1_->working_memory_[in1IdxE_](0, 0));
    if (dbg) {
      cerr << "s" << outIdx_ << " = asin(s" << in1IdxE_ << ") | ";
      cerr << "asin(" << in1_->working_memory_[in1IdxE_](0, 0)
           << ") = " << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarArcCosOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::acos(in1_->working_memory_[in1IdxE_](0, 0));
    if (dbg) {
      cerr << "s" << outIdx_ << " = acos(s" << in1IdxE_ << ") | ";
      cerr << "acos(" << in1_->working_memory_[in1IdxE_](0, 0)
           << ") = " << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarArcTanOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::atan(in1_->working_memory_[in1IdxE_](0, 0));
    if (dbg) {
      cerr << "s" << outIdx_ << " = atan(s" << in1IdxE_ << ") | ";
      cerr << "atan(" << in1_->working_memory_[in1IdxE_](0, 0)
           << ") = " << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarHeavisideOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        in1_->working_memory_[in1IdxE_](0, 0) >= 0.0 ? 1.0 : 0.0;
    if (dbg) {
    }
  }

  inline void ExecuteVectorHeavisideOp(bool dbg) {
    const double* in = out_->working_memory_[in1IdxE_].data();
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
    const double* ind = in1_->working_memory_[in1IdxE_].data();
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
        in1_->working_memory_[in1IdxE_](0, 0) * in2_->working_memory_[in2IdxE_];
    if (dbg) {
    }
  }

  inline void ExecuteScalarBroadcastOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1IdxE_](0, 0) * MatrixXd::Ones(memory_size_, 1);
    if (dbg) {
    }
  }

  inline void ExecuteVectorReciprocalOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        1.0 / in1_->working_memory_[in1IdxE_](0, 0);
    if (dbg) {
    }
  }

  inline void ExecuteVectorNormOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        in1_->working_memory_[in1IdxE_].norm();
    if (dbg) {
    }
  }

  inline void ExecuteVectorAbsOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        (in1_->working_memory_[in1IdxE_].array().abs()).matrix();
    if (dbg) {
    }
  }

  inline void ExecuteVectorSumOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1IdxE_] + in2_->working_memory_[in2IdxE_];
    if (dbg) {
    }
  }

  inline void ExecuteVectorDiffOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1IdxE_] - in2_->working_memory_[in2IdxE_];
    if (dbg) {
    }
  }

  inline void ExecuteVectorProductOp(bool dbg) {
    out_->working_memory_[outIdx_] = in1_->working_memory_[in1IdxE_].array() *
                                     in2_->working_memory_[in2IdxE_].array();
    if (dbg) {
    }
  }

  inline void ExecuteVectorDivisionOp(bool dbg) {
    out_->working_memory_[outIdx_] = in1_->working_memory_[in1IdxE_].array() /
                                     in2_->working_memory_[in2IdxE_].array();
    if (dbg) {
    }
  }

  inline void ExecuteVectorInnerProductOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        in1_->working_memory_[in1IdxE_].col(0).dot(
            in2_->working_memory_[in2IdxE_].col(0));
    if (dbg) {
    }
  }

  inline void ExecuteVectorOuterProductOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1IdxE_] *
        in2_->working_memory_[in2IdxE_].transpose();
    if (dbg) {
    }
  }

  inline void ExecuteScalarMatrixProductOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1IdxE_](0, 0) * in2_->working_memory_[in2IdxE_];
    if (dbg) {
    }
  }

  inline void ExecuteMatrixReciprocalOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        (1.0 / in1_->working_memory_[in1IdxE_].array()).matrix();
    if (dbg) {
    }
  }

  inline void ExecuteMatrixVectorProductOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1IdxE_] * in2_->working_memory_[in2IdxE_];
    if (dbg) {
    }
  }

  inline void ExecuteVectorColumnBroadcastOp(bool dbg) {
    out_->working_memory_[outIdx_] = in1_->working_memory_[in1IdxE_].replicate(
        1, in1_->working_memory_[in1IdxE_].rows());
    if (dbg) {
    }
  }

  inline void ExecuteVectorRowBroadcastOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1IdxE_]
            .replicate(1, in1_->working_memory_[in1IdxE_].rows())
            .transpose();
    if (dbg) {
    }
  }

  inline void ExecuteMatrixNormOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        in1_->working_memory_[in1IdxE_].norm();
    if (dbg) {
    }
  }

  inline void ExecuteMatrixColumnNormOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1IdxE_]
            .colwise()
            .norm()
            .transpose();  // automl-zero doesn't transpose?
    if (dbg) {
    }
  }

  inline void ExecuteMatrixRowNormOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1IdxE_].rowwise().norm();
    if (dbg) {
    }
  }

  inline void ExecuteMatrixTransposeOp(bool dbg) {
    if (outIdx_ == in1IdxE_)
      out_->working_memory_[outIdx_].transposeInPlace();
    else
      out_->working_memory_[outIdx_] =
          in1_->working_memory_[in1IdxE_].transpose();
    if (dbg) {
    }
  }

  inline void ExecuteMatrixAbsOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1IdxE_].array().abs().matrix();
    if (dbg) {
    }
  }

  inline void ExecuteMatrixSumOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1IdxE_] + in2_->working_memory_[in2IdxE_];
    if (dbg) {
    }
  }

  inline void ExecuteMatrixDiffOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1IdxE_] - in2_->working_memory_[in2IdxE_];
    if (dbg) {
    }
  }

  inline void ExecuteMatrixProductOp(bool dbg) {
    out_->working_memory_[outIdx_] = (in1_->working_memory_[in1IdxE_].array() *
                                      in2_->working_memory_[in2IdxE_].array())
                                         .matrix();
    if (dbg) {
    }
  }

  inline void ExecuteMatrixDivisionOp(bool dbg) {
    out_->working_memory_[outIdx_] = (in1_->working_memory_[in1IdxE_].array() /
                                      in2_->working_memory_[in2IdxE_].array())
                                         .matrix();
    if (dbg) {
    }
  }

  inline void ExecuteMatrixMatrixProductOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1IdxE_] * in2_->working_memory_[in2IdxE_];
    if (dbg) {
    }
  }

  inline void ExecuteScalarMinOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        min(in1_->working_memory_[in1IdxE_](0, 0),
            in2_->working_memory_[in2IdxE_](0, 0));
    if (dbg) {
    }
  }

  inline void ExecuteVectorMinOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1IdxE_].array().min(
            in2_->working_memory_[in2IdxE_].array());
    if (dbg) {
    }
  }

  inline void ExecuteMatrixMinOp(bool dbg) {
    const double* in1d = in1_->working_memory_[in1IdxE_].data();
    const double* in2d = in2_->working_memory_[in2IdxE_].data();
    const double* in1_end = in1d + in2_->working_memory_[in2IdxE_].rows() *
                                       in2_->working_memory_[in2IdxE_].cols();
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
            in1_->working_memory_[in1IdxE_](0, 0));
    if (dbg) {
    }
  }

  inline void ExecuteVectorMaxOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1IdxE_].array().max(
            in2_->working_memory_[in2IdxE_].array());
    if (dbg) {
    }
  }

  inline void ExecuteMatrixMaxOp(bool dbg) {
    const double* in1d = in1_->working_memory_[in1IdxE_].data();
    const double* in2d = in2_->working_memory_[in2IdxE_].data();
    const double* in1_end = in1d + in2_->working_memory_[in2IdxE_].rows() *
                                       in2_->working_memory_[in2IdxE_].cols();
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
        in1_->working_memory_[in1IdxE_].mean();
    if (dbg) {
    }
  }

  inline void ExecuteMatrixMeanOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        in1_->working_memory_[in1IdxE_].mean();
    if (dbg) {
    }
  }

  inline void ExecuteMatrixRowMeanOp(bool dbg) {
    out_->working_memory_[outIdx_] =
        in1_->working_memory_[in1IdxE_].rowwise().mean();
    if (dbg) {
    }
  }

  inline void ExecuteMatrixRowStDevOp(bool dbg) {
    for (int row = 0; row < out_->working_memory_[outIdx_].rows(); ++row) {
      const MatrixXd values = in1_->working_memory_[in1IdxE_].row(row);
      const double mean = values.mean();
      const double stdev =
          sqrt((values.array() * values.array()).sum() /
                   static_cast<double>(in1_->working_memory_[in1IdxE_].rows()) -
               mean * mean);
      out_->working_memory_[outIdx_](row, 0) = stdev;
    }
    if (dbg) {
    }
  }

  inline void ExecuteVectorStDevOp(bool dbg) {
    const double mean = in1_->working_memory_[in1IdxE_].mean();
    out_->working_memory_[outIdx_](0, 0) =
        sqrt(in1_->working_memory_[in1IdxE_].col(0).dot(
                 in1_->working_memory_[in1IdxE_].col(0)) /
                 in1_->working_memory_[in1IdxE_].rows() -
             mean * mean);
    if (dbg) {
    }
  }

  inline void ExecuteMatrixStDevOp(bool dbg) {
    const MatrixXd values = in1_->working_memory_[in1IdxE_];
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
    out_->working_memory_[outIdx_] = in1_->const_memory_[in1IdxE_];
    if (dbg) {
    }
  }

  inline void ExecuteVectorConstSetOp(bool dbg) {
    out_->working_memory_[outIdx_] = in1_->const_memory_[in1IdxE_];
    // out_->working_memory_[outIdx_] = constants_[in1Idx_ % constants_.size()]
    // *
    //  MatrixXd::Ones(memory_size_, 1);
    if (dbg) {
    }
  }

  inline void ExecuteMatrixConstSetOp(bool dbg) {
    out_->working_memory_[outIdx_] = in1_->const_memory_[in1IdxE_];
    // out_->working_memory_[outIdx_] = constants_[in1Idx_ % constants_.size()]
    // *
    //  MatrixXd::Ones(memory_size_, memory_size_);
    if (dbg) {
    }
  }

  inline void ExecuteScalarUniformSetOp(bool dbg) {
    uniform_real_distribution<double> dis(
        in1_->working_memory_[in1IdxE_](0, 0),
        in2_->working_memory_[in2IdxE_](0, 0));
    out_->working_memory_[outIdx_](0, 0) = dis(rng_);
    if (dbg) {
    }
  }

  inline void ExecuteVectorUniformSetOp(bool dbg) {
    uniform_real_distribution<double> dis(
        in1_->working_memory_[in1IdxE_](0, 0),
        in2_->working_memory_[in2IdxE_](0, 0));
    for (int i = 0; i < out_->working_memory_[outIdx_].rows(); i++)
      out_->working_memory_[outIdx_](i, 0) = dis(rng_);
    if (dbg) {
    }
  }

  inline void ExecuteMatrixUniformSetOp(bool dbg) {
    uniform_real_distribution<double> dis(
        in1_->working_memory_[in1IdxE_](0, 0),
        in2_->working_memory_[in2IdxE_](0, 0));
    for (int row = 0; row < out_->working_memory_[outIdx_].rows(); row++)
      for (int col = 0; col < out_->working_memory_[outIdx_].cols(); col++)
        out_->working_memory_[outIdx_](row, col) = dis(rng_);
    if (dbg) {
    }
  }

  inline void ExecuteScalarGaussianSetOp(bool dbg) {
    normal_distribution<double> dis(in1_->working_memory_[in1IdxE_](0, 0),
                                    in2_->working_memory_[in2IdxE_](0, 0));
    out_->working_memory_[outIdx_](0, 0) = dis(rng_);
    if (dbg) {
    }
  }

  inline void ExecuteVectorGaussianSetOp(bool dbg) {
    normal_distribution<double> dis(in1_->working_memory_[in1IdxE_](0, 0),
                                    in2_->working_memory_[in2IdxE_](0, 0));
    for (int i = 0; i < out_->working_memory_[outIdx_].rows(); i++)
      out_->working_memory_[outIdx_](i, 0) = dis(rng_);
    if (dbg) {
    }
  }

  inline void ExecuteMatrixGaussianSetOp(bool dbg) {
    normal_distribution<double> dis(in1_->working_memory_[in1IdxE_](0, 0),
                                    in2_->working_memory_[in2IdxE_](0, 0));
    for (int row = 0; row < out_->working_memory_[outIdx_].rows(); row++)
      for (int col = 0; col < out_->working_memory_[outIdx_].cols(); col++)
        out_->working_memory_[outIdx_](row, col) = dis(rng_);
    if (dbg) {
    }
  }

  inline void ExecuteScalarCondAOp(bool dbg) {
    if (in1_->working_memory_[in1IdxE_](0, 0) <
        in2_->working_memory_[in2IdxE_](0, 0))
      out_->working_memory_[outIdx_](0, 0) =
          -(out_->working_memory_[outIdx_](0, 0));

    if (dbg) {
      cerr << " IF s" << in1IdxE_ << " < s" << in2IdxE_ << " THEN s" << outIdx_
           << " = -s" << outIdx_ << " | ";
      cerr << " in1 " << in1_->working_memory_[in1IdxE_](0, 0) << " in2 "
           << in2_->working_memory_[outIdx_](0, 0) << " : "
           << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarCondBOp(bool dbg) {
    if (in1_->working_memory_[in1IdxE_](0, 0) >=
        in2_->working_memory_[in2IdxE_](0, 0))
      out_->working_memory_[outIdx_](0, 0) =
          -(out_->working_memory_[outIdx_](0, 0));
    if (dbg) {
    }
  }

  inline void ExecuteScalarPowOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::pow(in1_->working_memory_[in1IdxE_](0, 0),
                 in2_->working_memory_[in2IdxE_](0, 0));
    if (dbg) {
      cerr << "s" << outIdx_ << " = pow(s" << in1IdxE_ << ", s" << in2IdxE_
           << ") | ";
      cerr << "pow(" << in1_->working_memory_[in1IdxE_](0, 0) << ", "
           << in2_->working_memory_[in2IdxE_](0, 0) << " = "
           << out_->working_memory_[outIdx_](0, 0) << endl;
    }
  }

  inline void ExecuteScalarSqrOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::pow(in1_->working_memory_[in1IdxE_](0, 0), 2);
    if (dbg) {
    }
  }

  inline void ExecuteScalarCubeOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::pow(in1_->working_memory_[in1IdxE_](0, 0), 3);
    if (dbg) {
    }
  }

  inline void ExecuteScalarTanhOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::tanh(in1_->working_memory_[in1IdxE_](0, 0));
    if (dbg) {
    }
  }

  inline void ExecuteScalarSqrtOp(bool dbg) {
    out_->working_memory_[outIdx_](0, 0) =
        std::sqrt(in1_->working_memory_[in1IdxE_](0, 0));
    if (dbg) {
    }
  }
};
#endif
