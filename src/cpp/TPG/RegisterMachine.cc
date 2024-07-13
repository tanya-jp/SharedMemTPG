#include "RegisterMachine.h"

/******************************************************************************/
string RegisterMachine::checkpoint(bool all) {
  ostringstream oss;

  oss << "RegisterMachine:" << id_ << ":" << gtime_ << ":" << action_ << ":"
      << stateful_ << ":" << nrefs_;
  // for (int mem_t = 0; mem_t < memoryEigen::NUM_MEMORY_TYPES; mem_t++) {
  //   oss << ":" << sharedMemory_[mem_t]->id();
  // }

  if (all)
    for (size_t i = 0; i < bid_.size(); i++)
      oss << ":" << bid_[i]->checkpoint();
  else
    for (size_t i = 0; i < bidEffective_.size(); i++)
      oss << ":" << bidEffective_[i]->checkpoint();
  oss << endl;

  return oss.str();
}

/******************************************************************************
 * Create arbitrary RegisterMachine
 */
RegisterMachine::RegisterMachine(
    long gtime, long action, std::unordered_map<std::string, std::any> &params,
    long id, mt19937 &rng, std::vector<bool> &legalOps) {
  action_ = action;
  stateful_ = std::any_cast<int>(params["stateful"]);
  gtime_ = gtime;
  id_ = id;
  key_ = 0;
  nrefs_ = 0;

  skipIntrons_ = false;

  instruction *in;

  uniform_real_distribution<double> disR(0.0, 1.0);
  uniform_int_distribution<int> disP(
      1, std::any_cast<int>(params["max_initial_prog_size"]));
  int progSize = disP(rng);

  for (int i = 0; i < progSize; i++) {
    in = new instruction(params, rng);
    in->Mutate(true, legalOps, rng);
    bid_.push_back(in);
  }
  op_counts_.resize(instruction::NUM_OP);
  SetupMemory(std::any_cast<int>(params["memory_indices"]), std::any_cast<int>(params["memory_size"]));
}

/******************************************************************************
 * Create RegisterMachine from another RegisterMachine
 */
RegisterMachine::RegisterMachine(
    long gtime, RegisterMachine &plr,
    std::unordered_map<std::string, std::any> &params, long id) {
  action_ = plr.action();
  gtime_ = gtime;
  id_ = id;
  key_ = plr.key();
  bid_val_ = -(numeric_limits<double>::max());
  nrefs_ = 0;

  skipIntrons_ = false;
  stateful_ = plr.stateful_;

  for (auto initer = plr.bid_.begin(); initer != plr.bid_.end(); initer++)
    bid_.push_back(new instruction(**initer));

  op_counts_.resize(instruction::NUM_OP);
  SetupMemory(std::any_cast<int>(params["memory_indices"]), std::any_cast<int>(params["memory_size"]));
}
/******************************************************************************
 * Create RegisterMachine from checkpoint file
 */
RegisterMachine::RegisterMachine(
    long gtime, long action, int stateful,
    std::unordered_map<std::string, std::any> &params, long id, long nrefs,
    std::vector<instruction *> bid) {
  action_ = action;
  bid_ = bid;
  gtime_ = gtime;
  id_ = id;
  key_ = 0;
  nrefs_ = nrefs;

  stateful_ = stateful > 0 ? true : false;
  skipIntrons_ = false;

  op_counts_.resize(instruction::NUM_OP);

  SetupMemory(std::any_cast<int>(params["memory_indices"]),
              std::any_cast<int>(params["memory_size"]));
}

/******************************************************************************/
RegisterMachine::~RegisterMachine() {
  for (auto instr : bid_) delete instr;
  bid_.clear();
  for (auto memory : privateMemory_) delete memory;
  privateMemory_.clear();

  for (auto memory : input_memory_buff_) delete memory;
  input_memory_buff_.clear();
}

void RegisterMachine::MarkFeatures(instruction *istr, int in) {
  features_.clear();
  if (istr->GetInType(in) == memoryEigen::SCALAR_TYPE) {
    features_.insert(istr->GetInIdx(in));
  } else if (istr->GetInType(in) == memoryEigen::VECTOR_TYPE) {
    for (size_t f = istr->GetInIdx(in), row = 0;
         row < istr->GetInMem(in)->memory_size_; row++) {
      // features_.insert(f++ % num_input_);  // toroidal
      features_.insert(f++);
    }
  } else if (istr->GetInType(in) == memoryEigen::MATRIX_TYPE) {
    for (size_t f = istr->GetInIdx(in), row = 0;
         row < istr->GetInMem(in)->memory_size_; row++) {
      for (size_t col = 0; col < istr->GetInMem(in)->memory_size_; col++) {
        // features_.insert(f++ % num_input_);  // toroidal
        features_.insert(f++);
      }
    }
  }
}

/******************************************************************************/
void RegisterMachine::MarkIntrons(
    std::unordered_map<std::string, std::any> &params) {
  fill(op_counts_.begin(), op_counts_.end(), 0);  // count occurance of each op

  map<int, vector<bool> > Meff;  // maps [memory type][index]->true/false
  Meff[memoryEigen::SCALAR_TYPE] =
      vector<bool>(std::any_cast<int>(params["memory_indices"]), false);
  Meff[memoryEigen::VECTOR_TYPE] =
      vector<bool>(std::any_cast<int>(params["memory_indices"]), false);
  Meff[memoryEigen::MATRIX_TYPE] =
      vector<bool>(std::any_cast<int>(params["memory_indices"]), false);

  Meff[memoryEigen::SCALAR_TYPE][0] = true;  // mark bid output memory

  if (std::any_cast<int>(params["continuous_output"])) {
    Meff[memoryEigen::SCALAR_TYPE][1] = true;  // mark continuous output memory
  }
  for (auto istr : bid_) {
    for (int in = 0; in < 2; in++) {
      if (istr->IsMemoryRef(in)) Meff[istr->GetInType(in)][istr->GetInIdx(in)] = true;
    }
  }

  bidEffective_.clear();

  for (auto istr : bid_) {
    if (!skipIntrons_ || Meff[istr->GetOutType()][istr->outIdx_]) {
      bidEffective_.push_back(istr);
      op_counts_[istr->op_]++;
      // Setup IO memory
      istr->out_ = privateMemory_[istr->GetOutType()];
      // inputs
      for (int in = 0; in < 2; in++) {  // add in arity?
        if (istr->IsInput(in)) {
          istr->SetInMem(in, input_memory_buff_[istr->GetInType(in)]); 
          MarkFeatures(istr, in);
        } 
        else if (istr->IsMemoryRef(in)) {
          istr->SetInMem(in, privateMemory_[istr->GetInType(in)]);
        }
      }
    }
  }
}

/******************************************************************************/
void RegisterMachine::MuBid(std::unordered_map<std::string, std::any> &params,
                            mt19937 &rng, uniform_real_distribution<> &disR,
                            vector<bool> &legalOps) {
  bool changed = false;

  while (!changed) {
    /* Remove random instruction. */
    if (bid_.size() > 1 &&
        disR(rng) < std::any_cast<double>(params["p_bid_delete"])) {
      uniform_int_distribution<int> disBid(0, bid_.size() - 1);
      int i = disBid(rng);
      delete *(bid_.begin() + i);
      bid_.erase(bid_.begin() + i);
      changed = true;
    }

    /* Insert random instruction. */
    if ((int)bid_.size() < std::any_cast<int>(params["max_prog_size"]) &&
        disR(rng) < std::any_cast<double>(params["p_bid_add"])) {
      instruction *instr = new instruction(params, rng);
      instr->Mutate(true, legalOps, rng);
      uniform_int_distribution<int> disBid(0, bid_.size());
      int i = disBid(rng);
      bid_.insert(bid_.begin() + i, instr);
      changed = true;
    }

    /* Mutate a random instruction. */
    if (disR(rng) < std::any_cast<double>(params["p_bid_mutate"])) {
      uniform_int_distribution<int> disBid(0, bid_.size() - 1);
      int i = disBid(rng);
      bid_[i]->Mutate(false, legalOps, rng);
      changed = true;
    }

    // /* Add noise to constants */
    // if (params.find("p_bid_mu_const") != params.end() &&
    //     disR(rng) < std::any_cast<double>(params["p_bid_mu_const"])) {
    //   for (auto m : sharedMemory_) {
    //     m->NoiseToConst(rng,
    //                     std::any_cast<double>(params["bid_mu_const_stddev"]));
    //   }
    // }

    /* Swap positions of two instructions. */
    if (bid_.size() > 1 &&
        disR(rng) < std::any_cast<double>(params["p_bid_swap"])) {
      uniform_int_distribution<int> disBid(0, bid_.size() - 1);
      int i = disBid(rng);
      int j;
      do {
        j = disBid(rng);
      } while (i == j);
      std::swap(bid_[i], bid_[j]);
      changed = true;
    }
  }
}

// void RegisterMachine::CopyInputToMemory(instruction *istr, state *obs,
//                                         size_t in) {
//   // In this case GetInMem(in) will be inputMemory_ and we use index 0
//   // Indices to input memory are mod by obs->dim_ to support environments
//   // with different number of scalar observation variables
//   size_t idx = 0;
//   if (istr->GetInType(in) == memoryEigen::SCALAR_TYPE) {
//     istr->GetInMem(in)->working_memory_[idx](0, 0) =
//         obs->stateValueAtIndex(istr->GetInIdx(in) % obs->dim_);
//   } else if (istr->GetInType(in) == memoryEigen::VECTOR_TYPE) {
//     for (size_t f = istr->GetInIdx(in), row = 0;
//          row < istr->GetInMem(in)->memory_size_; row++) {
//       istr->GetInMem(in)->working_memory_[idx](row, 0) =
//           obs->stateValueAtIndex(f++ % obs->dim_);
//     }
//   } else if (istr->GetInType(in) == memoryEigen::MATRIX_TYPE) {
//     for (size_t f = istr->GetInIdx(in), row = 0;
//          row < istr->GetInMem(in)->memory_size_; row++) {
//       for (size_t col = 0; col < istr->GetInMem(in)->memory_size_; col++) {
//         istr->GetInMem(in)->working_memory_[idx](row, col) =
//             obs->stateValueAtIndex(f++ % obs->dim_);
//       }
//     }
//   }
//   istr->SetInIdxE(in, idx);  // reset inIdxE to zero for input ref
// }

// TODO(skelly): This functions currently assumes obs is a vector of state vars
void RegisterMachine::CopyInputToMemoryBuff(state *obs) {
  // Memory size is the same for SCALAR, VECTOR, MATRIX
  auto memory_size = input_memory_buff_[memoryEigen::SCALAR_TYPE]->memory_size_;

  // Copy obs to scalar memory
  Matrix<double, Dynamic, Dynamic> scalar_mat(1, 1);
  scalar_mat(0, 0) = obs->stateValueAtIndex(0);
  AddToInputMemoryBuff(scalar_mat, memoryEigen::SCALAR_TYPE);

  // Copy obs to vector memory
  Matrix<double, Dynamic, Dynamic> vector_mat(memory_size, 1);
  for (size_t row = 0; row < memory_size; row++)
    vector_mat(row, 0) = obs->stateValueAtIndex(row);
  AddToInputMemoryBuff(vector_mat, memoryEigen::VECTOR_TYPE);

  // Copy obs to matrix memory
  Matrix<double, Dynamic, Dynamic> matrix_mat(memory_size, memory_size);
  for (size_t row = 0; row < memory_size; row++)
    for (size_t col = 0; col < memory_size; col++)
      matrix_mat(row, col) = obs->stateValueAtIndex(col % obs->dim_);
  AddToInputMemoryBuff(matrix_mat, memoryEigen::MATRIX_TYPE);
}

/******************************************************************************/
double RegisterMachine::Run(state *obs, int &time_step,
                            const size_t &graph_depth, bool &verbose) {
  CopyInputToMemoryBuff(obs);

  // Clear working memory prior to execution, making this program stateless
  if (!stateful_) ClearWorking();

  for (auto istr : bidEffective_) {
    istr->exec(verbose);  // Execute instruction

    // Track memory read times
    for (size_t in = 0; in < 2; in++) {
      if (istr->GetInType(in) != memoryEigen::NA_TYPE) {
        // If input is a memory ref then track read time for temporal memory
        if (!(istr->IsInput(in))) {  
          istr->GetInMem(in)->getReadTimeE()(istr->GetInIdx(in), 0) =
              time_step + (graph_depth / MAX_GRAPH_DEPTH);
        }
      }
    }
    // Track write times for temporal memory
    istr->out_->getWriteTimeE()(istr->outIdx_, 0) =
        time_step + (graph_depth / MAX_GRAPH_DEPTH);
  }
  // Return bid value
  return privateMemory_[memoryEigen::SCALAR_TYPE]->working_memory_[0](0, 0);
}

/******************************************************************************/
void RegisterMachine::SetupMemory(size_t memoryIndices, size_t memory_size) {
  for (int mem_t = 0; mem_t < memoryEigen::NUM_MEMORY_TYPES; mem_t++) {
    privateMemory_.push_back(
        new memoryEigen(-1, mem_t, memoryIndices, memory_size));
    input_memory_buff_.push_back(
        new memoryEigen(-1, mem_t, memoryIndices, memory_size));
  }
}
