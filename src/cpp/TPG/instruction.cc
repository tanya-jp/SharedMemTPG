#include "instruction.h"

const std::vector<double> instruction::constants_ = {
    -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
    0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9};

vector<vector<size_t> > instruction::op_mem_types_(NUM_OP);
vector<instruction::operation> instruction::op_list_(NUM_OP);

string instruction::checkpoint() {
  ostringstream oss;
  oss << in1Src_ << "_";
  oss << in2Src_ << "_";
  oss << outSrc_ << "_";
  oss << outIdx_ << "_";
  oss << op_ << "_";
  oss << in1Idx_ << "_";
  oss << in1IdxE_ << "_";
  oss << in2Idx_ << "_";
  oss << in2IdxE_;
  return oss.str();
}

void instruction::SetupOps() {
  op_mem_types_[SCALAR_SUM_OP_] = {memoryEigen::SCALAR_TYPE,
                                   memoryEigen::SCALAR_TYPE,
                                   memoryEigen::SCALAR_TYPE};
  op_list_[SCALAR_SUM_OP_] = (&instruction::ExecuteScalarSumOp);

  op_mem_types_[SCALAR_DIFF_OP_] = {memoryEigen::SCALAR_TYPE,
                                    memoryEigen::SCALAR_TYPE,
                                    memoryEigen::SCALAR_TYPE};
  op_list_[SCALAR_DIFF_OP_] = (&instruction::ExecuteScalarDiffOp);

  op_mem_types_[SCALAR_PRODUCT_OP_] = {memoryEigen::SCALAR_TYPE,
                                       memoryEigen::SCALAR_TYPE,
                                       memoryEigen::SCALAR_TYPE};
  op_list_[SCALAR_PRODUCT_OP_] = (&instruction::ExecuteScalarProductOp);

  op_mem_types_[SCALAR_DIVISION_OP_] = {memoryEigen::SCALAR_TYPE,
                                        memoryEigen::SCALAR_TYPE,
                                        memoryEigen::SCALAR_TYPE};
  op_list_[SCALAR_DIVISION_OP_] = (&instruction::ExecuteScalarDivisionOp);

  op_mem_types_[SCALAR_ABS_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::SCALAR_TYPE, memoryEigen::NA_TYPE};
  op_list_[SCALAR_ABS_OP_] = (&instruction::ExecuteScalarAbsOp);

  op_mem_types_[SCALAR_SIN_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::SCALAR_TYPE, memoryEigen::NA_TYPE};
  op_list_[SCALAR_SIN_OP_] = (&instruction::ExecuteScalarSinOp);

  op_mem_types_[SCALAR_COS_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::SCALAR_TYPE, memoryEigen::NA_TYPE};
  op_list_[SCALAR_COS_OP_] = (&instruction::ExecuteScalarCosOp);

  op_mem_types_[SCALAR_TAN_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::SCALAR_TYPE, memoryEigen::NA_TYPE};
  op_list_[SCALAR_TAN_OP_] = (&instruction::ExecuteScalarTanOp);

  op_mem_types_[SCALAR_EXP_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::SCALAR_TYPE, memoryEigen::NA_TYPE};
  op_list_[SCALAR_EXP_OP_] = (&instruction::ExecuteScalarExpOp);

  op_mem_types_[SCALAR_LOG_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::SCALAR_TYPE, memoryEigen::NA_TYPE};
  op_list_[SCALAR_LOG_OP_] = (&instruction::ExecuteScalarLogOp);

  op_mem_types_[SCALAR_RECIPROCAL_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::SCALAR_TYPE, memoryEigen::NA_TYPE};
  op_list_[SCALAR_RECIPROCAL_OP_] = (&instruction::ExecuteScalarReciprocalOp);

  op_mem_types_[SCALAR_SIN_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::SCALAR_TYPE, memoryEigen::NA_TYPE};
  op_list_[SCALAR_SIN_OP_] = (&instruction::ExecuteScalarSinOp);

  op_mem_types_[SCALAR_COS_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::SCALAR_TYPE, memoryEigen::NA_TYPE};
  op_list_[SCALAR_COS_OP_] = (&instruction::ExecuteScalarCosOp);

  op_mem_types_[SCALAR_ARCSIN_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::SCALAR_TYPE, memoryEigen::NA_TYPE};
  op_list_[SCALAR_ARCSIN_OP_] = (&instruction::ExecuteScalarArcSinOp);

  op_mem_types_[SCALAR_ARCCOS_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::SCALAR_TYPE, memoryEigen::NA_TYPE};
  op_list_[SCALAR_ARCCOS_OP_] = (&instruction::ExecuteScalarArcCosOp);

  op_mem_types_[SCALAR_ARCTAN_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::SCALAR_TYPE, memoryEigen::NA_TYPE};
  op_list_[SCALAR_ARCTAN_OP_] = (&instruction::ExecuteScalarArcTanOp);

  op_mem_types_[SCALAR_HEAVYSIDE_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::SCALAR_TYPE, memoryEigen::NA_TYPE};
  op_list_[SCALAR_HEAVYSIDE_OP_] = (&instruction::ExecuteScalarHeavisideOp);

  op_mem_types_[VECTOR_HEAVYSIDE_OP_] = {
      memoryEigen::VECTOR_TYPE, memoryEigen::VECTOR_TYPE, memoryEigen::NA_TYPE};
  op_list_[VECTOR_HEAVYSIDE_OP_] = (&instruction::ExecuteVectorHeavisideOp);

  op_mem_types_[MATRIX_HEAVYSIDE_OP_] = {
      memoryEigen::MATRIX_TYPE, memoryEigen::MATRIX_TYPE, memoryEigen::NA_TYPE};
  op_list_[MATRIX_HEAVYSIDE_OP_] = (&instruction::ExecuteMatrixHeavisideOp);

  op_mem_types_[SCALAR_VECTOR_PRODUCT_OP_] = {memoryEigen::VECTOR_TYPE,
                                              memoryEigen::SCALAR_TYPE,
                                              memoryEigen::VECTOR_TYPE};
  op_list_[SCALAR_VECTOR_PRODUCT_OP_] =
      (&instruction::ExecuteScalarVectorProductOp);

  op_mem_types_[SCALAR_BROADCAST_OP_] = {
      memoryEigen::VECTOR_TYPE, memoryEigen::SCALAR_TYPE, memoryEigen::NA_TYPE};
  op_list_[SCALAR_BROADCAST_OP_] = (&instruction::ExecuteScalarBroadcastOp);

  op_mem_types_[VECTOR_RECIPROCAL_OP_] = {
      memoryEigen::VECTOR_TYPE, memoryEigen::VECTOR_TYPE, memoryEigen::NA_TYPE};
  op_list_[VECTOR_RECIPROCAL_OP_] = (&instruction::ExecuteVectorReciprocalOp);

  op_mem_types_[VECTOR_NORM_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::VECTOR_TYPE, memoryEigen::NA_TYPE};
  op_list_[VECTOR_NORM_OP_] = (&instruction::ExecuteVectorNormOp);

  op_mem_types_[VECTOR_ABS_OP_] = {
      memoryEigen::VECTOR_TYPE, memoryEigen::VECTOR_TYPE, memoryEigen::NA_TYPE};
  op_list_[VECTOR_ABS_OP_] = (&instruction::ExecuteVectorAbsOp);

  op_mem_types_[VECTOR_SUM_OP_] = {memoryEigen::VECTOR_TYPE,
                                   memoryEigen::VECTOR_TYPE,
                                   memoryEigen::VECTOR_TYPE};
  op_list_[VECTOR_SUM_OP_] = (&instruction::ExecuteVectorSumOp);

  op_mem_types_[VECTOR_DIFF_OP_] = {memoryEigen::VECTOR_TYPE,
                                    memoryEigen::VECTOR_TYPE,
                                    memoryEigen::VECTOR_TYPE};
  op_list_[VECTOR_DIFF_OP_] = (&instruction::ExecuteVectorDiffOp);

  op_mem_types_[VECTOR_PRODUCT_OP_] = {memoryEigen::VECTOR_TYPE,
                                       memoryEigen::VECTOR_TYPE,
                                       memoryEigen::VECTOR_TYPE};
  op_list_[VECTOR_PRODUCT_OP_] = (&instruction::ExecuteVectorProductOp);

  op_mem_types_[VECTOR_DIVISION_OP_] = {memoryEigen::VECTOR_TYPE,
                                        memoryEigen::VECTOR_TYPE,
                                        memoryEigen::VECTOR_TYPE};
  op_list_[VECTOR_DIVISION_OP_] = (&instruction::ExecuteVectorDivisionOp);

  op_mem_types_[VECTOR_INNER_PRODUCT_OP_] = {memoryEigen::SCALAR_TYPE,
                                             memoryEigen::VECTOR_TYPE,
                                             memoryEigen::VECTOR_TYPE};
  op_list_[VECTOR_INNER_PRODUCT_OP_] =
      (&instruction::ExecuteVectorInnerProductOp);

  op_mem_types_[VECTOR_OUTER_PRODUCT_OP_] = {memoryEigen::MATRIX_TYPE,
                                             memoryEigen::VECTOR_TYPE,
                                             memoryEigen::VECTOR_TYPE};
  op_list_[VECTOR_OUTER_PRODUCT_OP_] =
      (&instruction::ExecuteVectorOuterProductOp);

  op_mem_types_[SCALAR_MATRIX_PRODUCT_OP_] = {memoryEigen::MATRIX_TYPE,
                                              memoryEigen::SCALAR_TYPE,
                                              memoryEigen::MATRIX_TYPE};
  op_list_[SCALAR_MATRIX_PRODUCT_OP_] =
      (&instruction::ExecuteScalarMatrixProductOp);

  op_mem_types_[MATRIX_RECIPROCAL_OP_] = {
      memoryEigen::MATRIX_TYPE, memoryEigen::MATRIX_TYPE, memoryEigen::NA_TYPE};
  op_list_[MATRIX_RECIPROCAL_OP_] = (&instruction::ExecuteMatrixReciprocalOp);

  op_mem_types_[MATRIX_VECTOR_PRODUCT_OP_] = {memoryEigen::VECTOR_TYPE,
                                              memoryEigen::MATRIX_TYPE,
                                              memoryEigen::VECTOR_TYPE};
  op_list_[MATRIX_VECTOR_PRODUCT_OP_] =
      (&instruction::ExecuteMatrixVectorProductOp);

  op_mem_types_[VECTOR_COLUMN_BROADCAST_OP_] = {
      memoryEigen::MATRIX_TYPE, memoryEigen::VECTOR_TYPE, memoryEigen::NA_TYPE};
  op_list_[VECTOR_COLUMN_BROADCAST_OP_] =
      (&instruction::ExecuteVectorColumnBroadcastOp);

  op_mem_types_[VECTOR_ROW_BROADCAST_OP_] = {
      memoryEigen::MATRIX_TYPE, memoryEigen::VECTOR_TYPE, memoryEigen::NA_TYPE};
  op_list_[VECTOR_ROW_BROADCAST_OP_] =
      (&instruction::ExecuteVectorRowBroadcastOp);

  op_mem_types_[MATRIX_NORM_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::MATRIX_TYPE, memoryEigen::NA_TYPE};
  op_list_[MATRIX_NORM_OP_] = (&instruction::ExecuteMatrixNormOp);

  op_mem_types_[MATRIX_COLUMN_NORM_OP_] = {
      memoryEigen::VECTOR_TYPE, memoryEigen::MATRIX_TYPE, memoryEigen::NA_TYPE};
  op_list_[MATRIX_COLUMN_NORM_OP_] = (&instruction::ExecuteMatrixColumnNormOp);

  op_mem_types_[MATRIX_ROW_NORM_OP_] = {
      memoryEigen::VECTOR_TYPE, memoryEigen::MATRIX_TYPE, memoryEigen::NA_TYPE};
  op_list_[MATRIX_ROW_NORM_OP_] = (&instruction::ExecuteMatrixRowNormOp);

  op_mem_types_[MATRIX_TRANSPOSE_OP_] = {
      memoryEigen::MATRIX_TYPE, memoryEigen::MATRIX_TYPE, memoryEigen::NA_TYPE};
  op_list_[MATRIX_TRANSPOSE_OP_] = (&instruction::ExecuteMatrixTransposeOp);

  op_mem_types_[MATRIX_ABS_OP_] = {
      memoryEigen::MATRIX_TYPE, memoryEigen::MATRIX_TYPE, memoryEigen::NA_TYPE};
  op_list_[MATRIX_ABS_OP_] = (&instruction::ExecuteMatrixAbsOp);

  op_mem_types_[MATRIX_SUM_OP_] = {memoryEigen::MATRIX_TYPE,
                                   memoryEigen::MATRIX_TYPE,
                                   memoryEigen::MATRIX_TYPE};
  op_list_[MATRIX_SUM_OP_] = (&instruction::ExecuteMatrixSumOp);

  op_mem_types_[MATRIX_DIFF_OP_] = {memoryEigen::MATRIX_TYPE,
                                    memoryEigen::MATRIX_TYPE,
                                    memoryEigen::MATRIX_TYPE};
  op_list_[MATRIX_DIFF_OP_] = (&instruction::ExecuteMatrixDiffOp);

  op_mem_types_[MATRIX_PRODUCT_OP_] = {memoryEigen::MATRIX_TYPE,
                                       memoryEigen::MATRIX_TYPE,
                                       memoryEigen::MATRIX_TYPE};
  op_list_[MATRIX_PRODUCT_OP_] = (&instruction::ExecuteMatrixProductOp);

  op_mem_types_[MATRIX_DIVISION_OP_] = {memoryEigen::MATRIX_TYPE,
                                        memoryEigen::MATRIX_TYPE,
                                        memoryEigen::MATRIX_TYPE};
  op_list_[MATRIX_DIVISION_OP_] = (&instruction::ExecuteMatrixDivisionOp);

  op_mem_types_[MATRIX_MATRIX_PRODUCT_OP_] = {memoryEigen::MATRIX_TYPE,
                                              memoryEigen::MATRIX_TYPE,
                                              memoryEigen::MATRIX_TYPE};
  op_list_[MATRIX_MATRIX_PRODUCT_OP_] =
      (&instruction::ExecuteMatrixMatrixProductOp);

  op_mem_types_[SCALAR_MIN_OP_] = {memoryEigen::SCALAR_TYPE,
                                   memoryEigen::SCALAR_TYPE,
                                   memoryEigen::SCALAR_TYPE};
  op_list_[SCALAR_MIN_OP_] = (&instruction::ExecuteScalarMinOp);

  op_mem_types_[VECTOR_MIN_OP_] = {memoryEigen::VECTOR_TYPE,
                                   memoryEigen::VECTOR_TYPE,
                                   memoryEigen::VECTOR_TYPE};
  op_list_[VECTOR_MIN_OP_] = (&instruction::ExecuteVectorMinOp);

  op_mem_types_[MATRIX_MIN_OP_] = {memoryEigen::MATRIX_TYPE,
                                   memoryEigen::MATRIX_TYPE,
                                   memoryEigen::MATRIX_TYPE};
  op_list_[MATRIX_MIN_OP_] = (&instruction::ExecuteMatrixMinOp);

  op_mem_types_[SCALAR_MAX_OP_] = {memoryEigen::SCALAR_TYPE,
                                   memoryEigen::SCALAR_TYPE,
                                   memoryEigen::SCALAR_TYPE};
  op_list_[SCALAR_MAX_OP_] = (&instruction::ExecuteScalarMaxOp);

  op_mem_types_[VECTOR_MAX_OP_] = {memoryEigen::VECTOR_TYPE,
                                   memoryEigen::VECTOR_TYPE,
                                   memoryEigen::VECTOR_TYPE};
  op_list_[VECTOR_MAX_OP_] = (&instruction::ExecuteVectorMaxOp);

  op_mem_types_[MATRIX_MAX_OP_] = {memoryEigen::MATRIX_TYPE,
                                   memoryEigen::MATRIX_TYPE,
                                   memoryEigen::MATRIX_TYPE};
  op_list_[MATRIX_MAX_OP_] = (&instruction::ExecuteMatrixMaxOp);

  op_mem_types_[VECTOR_MEAN_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::VECTOR_TYPE, memoryEigen::NA_TYPE};
  op_list_[VECTOR_MEAN_OP_] = (&instruction::ExecuteVectorMeanOp);

  op_mem_types_[MATRIX_MEAN_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::MATRIX_TYPE, memoryEigen::NA_TYPE};
  op_list_[MATRIX_MEAN_OP_] = (&instruction::ExecuteMatrixMeanOp);

  op_mem_types_[MATRIX_ROW_MEAN_OP_] = {
      memoryEigen::VECTOR_TYPE, memoryEigen::MATRIX_TYPE, memoryEigen::NA_TYPE};
  op_list_[MATRIX_ROW_MEAN_OP_] = (&instruction::ExecuteMatrixRowMeanOp);

  op_mem_types_[MATRIX_ROW_ST_DEV_OP_] = {
      memoryEigen::VECTOR_TYPE, memoryEigen::MATRIX_TYPE, memoryEigen::NA_TYPE};
  op_list_[MATRIX_ROW_ST_DEV_OP_] = (&instruction::ExecuteMatrixRowStDevOp);

  op_mem_types_[VECTOR_ST_DEV_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::MATRIX_TYPE, memoryEigen::NA_TYPE};
  op_list_[VECTOR_ST_DEV_OP_] = (&instruction::ExecuteVectorStDevOp);

  op_mem_types_[MATRIX_ST_DEV_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::MATRIX_TYPE, memoryEigen::NA_TYPE};
  op_list_[MATRIX_ST_DEV_OP_] = (&instruction::ExecuteMatrixStDevOp);

  op_mem_types_[SCALAR_CONST_SET_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::SCALAR_TYPE, memoryEigen::NA_TYPE};
  op_list_[SCALAR_CONST_SET_OP_] = (&instruction::ExecuteScalarConstSetOp);

  op_mem_types_[VECTOR_CONST_SET_OP_] = {
      memoryEigen::VECTOR_TYPE, memoryEigen::VECTOR_TYPE, memoryEigen::NA_TYPE};
  op_list_[VECTOR_CONST_SET_OP_] = (&instruction::ExecuteVectorConstSetOp);

  op_mem_types_[MATRIX_CONST_SET_OP_] = {
      memoryEigen::MATRIX_TYPE, memoryEigen::MATRIX_TYPE, memoryEigen::NA_TYPE};
  op_list_[MATRIX_CONST_SET_OP_] = (&instruction::ExecuteMatrixConstSetOp);

  op_mem_types_[SCALAR_UNIFORM_SET_OP_] = {memoryEigen::SCALAR_TYPE,
                                           memoryEigen::SCALAR_TYPE,
                                           memoryEigen::SCALAR_TYPE};
  op_list_[SCALAR_UNIFORM_SET_OP_] = (&instruction::ExecuteScalarUniformSetOp);

  op_mem_types_[VECTOR_UNIFORM_SET_OP_] = {memoryEigen::VECTOR_TYPE,
                                           memoryEigen::SCALAR_TYPE,
                                           memoryEigen::SCALAR_TYPE};
  op_list_[VECTOR_UNIFORM_SET_OP_] = (&instruction::ExecuteVectorUniformSetOp);

  op_mem_types_[MATRIX_UNIFORM_SET_OP_] = {memoryEigen::MATRIX_TYPE,
                                           memoryEigen::SCALAR_TYPE,
                                           memoryEigen::SCALAR_TYPE};
  op_list_[MATRIX_UNIFORM_SET_OP_] = (&instruction::ExecuteMatrixUniformSetOp);

  op_mem_types_[SCALAR_GAUSSIAN_SET_OP_] = {memoryEigen::SCALAR_TYPE,
                                            memoryEigen::SCALAR_TYPE,
                                            memoryEigen::SCALAR_TYPE};
  op_list_[SCALAR_GAUSSIAN_SET_OP_] =
      (&instruction::ExecuteScalarGaussianSetOp);

  op_mem_types_[VECTOR_GAUSSIAN_SET_OP_] = {memoryEigen::VECTOR_TYPE,
                                            memoryEigen::SCALAR_TYPE,
                                            memoryEigen::SCALAR_TYPE};
  op_list_[VECTOR_GAUSSIAN_SET_OP_] =
      (&instruction::ExecuteVectorGaussianSetOp);

  op_mem_types_[MATRIX_GAUSSIAN_SET_OP_] = {memoryEigen::MATRIX_TYPE,
                                            memoryEigen::SCALAR_TYPE,
                                            memoryEigen::SCALAR_TYPE};
  op_list_[MATRIX_GAUSSIAN_SET_OP_] =
      (&instruction::ExecuteMatrixGaussianSetOp);

  op_mem_types_[SCALAR_COND_A_OP_] = {memoryEigen::SCALAR_TYPE,
                                      memoryEigen::SCALAR_TYPE,
                                      memoryEigen::SCALAR_TYPE};
  op_list_[SCALAR_COND_A_OP_] = (&instruction::ExecuteScalarCondAOp);

  op_mem_types_[SCALAR_COND_B_OP_] = {memoryEigen::SCALAR_TYPE,
                                      memoryEigen::SCALAR_TYPE,
                                      memoryEigen::SCALAR_TYPE};
  op_list_[SCALAR_COND_B_OP_] = (&instruction::ExecuteScalarCondBOp);

  op_mem_types_[SCALAR_POW_OP_] = {memoryEigen::SCALAR_TYPE,
                                   memoryEigen::SCALAR_TYPE,
                                   memoryEigen::SCALAR_TYPE};
  op_list_[SCALAR_POW_OP_] = (&instruction::ExecuteScalarPowOp);

  op_mem_types_[SCALAR_SQR_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::SCALAR_TYPE, memoryEigen::NA_TYPE};
  op_list_[SCALAR_SQR_OP_] = (&instruction::ExecuteScalarSqrOp);

  op_mem_types_[SCALAR_CUBE_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::SCALAR_TYPE, memoryEigen::NA_TYPE};
  op_list_[SCALAR_CUBE_OP_] = (&instruction::ExecuteScalarCubeOp);

  op_mem_types_[SCALAR_TANH_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::SCALAR_TYPE, memoryEigen::NA_TYPE};
  op_list_[SCALAR_TANH_OP_] = (&instruction::ExecuteScalarTanhOp);

  op_mem_types_[SCALAR_SQRT_OP_] = {
      memoryEigen::SCALAR_TYPE, memoryEigen::SCALAR_TYPE, memoryEigen::NA_TYPE};
  op_list_[SCALAR_SQRT_OP_] = (&instruction::ExecuteScalarSqrtOp);
}

// constructor
instruction::instruction(std::unordered_map<string, std::any> &params,
                         mt19937 &rng) {
  memory_size_ = std::any_cast<int>(params["memory_size"]);
  memIndices_ = std::any_cast<int>(params["memory_indices"]);
  rng_ = rng;
}

// copy construction
instruction::instruction(instruction &i) {
  memory_size_ = i.memory_size_;
  memory_size_ = i.memory_size_;
  memIndices_ = i.memIndices_;
  out_ = i.out_;
  in1_ = i.in1_;
  in2_ = i.in2_;
  op_mem_types_ = i.op_mem_types_;
  op_list_ = i.op_list_;

  in1Src_ = i.in1Src_;
  in2Src_ = i.in2Src_;
  outSrc_ = i.outSrc_;
  outIdx_ = i.outIdx_;
  op_ = i.op_;
  in1Idx_ = i.in1Idx_;
  in1IdxE_ = i.in1IdxE_;
  in2Idx_ = i.in2Idx_;
  in2IdxE_ = i.in2IdxE_;

  rng_ = i.rng_;
}

void instruction::mutate(bool uniform, vector<bool> &legal_ops, mt19937 &rng) {
  auto nOp = std::count(legal_ops.begin(), legal_ops.end(), true);

  if (uniform) {  // randomly set each part of this instruction
    //std::uniform_int_distribution<> dis(0, 2);
    std::uniform_int_distribution<> dis(0, 1);
    in1Src_ = dis(rng) == 0 ? 0 : 2; // private memory or input
    in2Src_ = dis(rng) == 0 ? 0 : 2; //private memory or input
    // dis = std::uniform_int_distribution<>(0, 1);
    outSrc_ = 0; //dis(rng);  // only write to private
    dis = std::uniform_int_distribution<>(0, memIndices_ - 1);
    outIdx_ = dis(rng);
    dis = std::uniform_int_distribution<>(0, legal_ops.size() - 1);
    do {
      op_ = dis(rng);
    } while (!legal_ops[op_]);
    dis = std::uniform_int_distribution<>(
        0, memIndices_ - 1);
    in1Idx_ = in1IdxE_ = dis(rng);
    in2Idx_ = in2IdxE_ = dis(rng);
  } else {  // randomly change one part of this instruction
    int prev;
    // select which part to change
    // std::uniform_int_distribution<> dis(0, 6);
    std::uniform_int_distribution<> dis(0, 5);
    int i = dis(rng);
    switch (i) {
      case 0:  // change in1 src to one of: private memory, shared memory, input
        prev = in1Src_;
        // dis = std::uniform_int_distribution<>(0, 2);
        dis = std::uniform_int_distribution<>(0, 1);
        do {
          in1Src_ = dis(rng) == 0 ? 0 : 2;  // private memory or input
        } while (in1Src_ == prev);
        // switching from input to memory ref
        if (prev == 2) in1Idx_ = in1IdxE_ = in1Idx_ % memIndices_;
        break;
      case 1:  // change in2 src to one of: private memory, shared memory, input
        prev = in2Src_;
        // dis = std::uniform_int_distribution<>(0, 2);
        dis = std::uniform_int_distribution<>(0, 1);
        do {
          in2Src_ = dis(rng) == 0 ? 0 : 2;  // private memory or input
        } while (in2Src_ == prev);
        // switching from input to memory ref
        if (prev == 2) in2Idx_ = in2IdxE_ = in2Idx_ % memIndices_;
        break;
      case 2:  // change out index
        prev = outIdx_;
        dis = std::uniform_int_distribution<>(0, memIndices_ - 1);
        do {
          outIdx_ = dis(rng);
        } while (outIdx_ == prev);
        break;
      case 3:  // change op
        prev = op_;
        dis = std::uniform_int_distribution<>(0, legal_ops.size() - 1);
        do {
          op_ = dis(rng);
        } while ((nOp > 1 && op_ == prev) || !legal_ops[op_]);
        break;
      case 4:  // change in1 index
        // if (num_input_ < 2) break;
        prev = in1Idx_;
        dis = std::uniform_int_distribution<>(
            0, memIndices_ - 1);  
        do {
          in1Idx_ = in1IdxE_ = dis(rng);
        } while (in1Idx_ == prev);
        break;
      case 5:  // change in2 index
        // if (num_input_ < 2) break;
        prev = in2Idx_;
        dis = std::uniform_int_distribution<>(0, memIndices_ - 1);
        do {
          in2Idx_ = in2IdxE_ = dis(rng);
        } while (in2Idx_ == prev);
        break;
    }
  }
}
