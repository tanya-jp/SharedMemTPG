#include "RegisterMachine.h"

/******************************************************************************/
string RegisterMachine::checkpoint(bool all) {
  ostringstream oss;

  oss << "RegisterMachine:" << id_ << ":" << gtime_ << ":" << action_ << ":"
      << stateful_ << ":" << num_input_ << ":" << nrefs_;
  for (int mem_t = 0; mem_t < memoryEigen::NUM_MEMORY_TYPES; mem_t++) {
    oss << ":" << sharedMemory_[mem_t]->id();
  }

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
  num_input_ = std::any_cast<int>(params["n_input"]);
  memoryRows_ = std::any_cast<int>(params["memory_rows"]);
  memoryCols_ = std::any_cast<int>(params["memory_cols"]);
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
    in->mutate(true, legalOps, rng);
    bid_.push_back(in);
  }
  op_counts_.resize(instruction::NUM_OP);
  SetupMemory(std::any_cast<int>(params["memory_indices"]), memoryRows_,
              memoryCols_);
}

/******************************************************************************
 * Create RegisterMachine from another RegisterMachine
 */
RegisterMachine::RegisterMachine(
    long gtime, RegisterMachine &plr,
    std::unordered_map<std::string, std::any> &params, long id) {
  action_ = plr.action();
  num_input_ = plr.num_input_;
  memoryRows_ = plr.memoryRows_;
  memoryCols_ = plr.memoryCols_;
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
  SetupMemory(std::any_cast<int>(params["memory_indices"]), memoryRows_,
              memoryCols_);
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
  num_input_ = std::any_cast<int>(params["n_input"]);
  gtime_ = gtime;
  id_ = id;
  key_ = 0;
  nrefs_ = nrefs;

  stateful_ = stateful > 0 ? true : false;
  skipIntrons_ = false;

  op_counts_.resize(instruction::NUM_OP);

  SetupMemory(std::any_cast<int>(params["memory_indices"]),
              std::any_cast<int>(params["memory_rows"]),
              std::any_cast<int>(params["memory_cols"]));
}

/******************************************************************************/
RegisterMachine::~RegisterMachine() {
  for (auto biditer = bid_.begin(); biditer != bid_.end(); biditer++)
    delete *biditer;
  bid_.clear();
  for (auto meiter = privateMemory_.begin(); meiter != privateMemory_.end();
       meiter++)
    delete *meiter;
  privateMemory_.clear();
  for (size_t mp = 0; mp < inputMemoryPointers_.size(); mp++) {
    for (auto meiter = inputMemoryPointers_[mp].begin();
         meiter != inputMemoryPointers_[mp].end(); meiter++)
      delete *meiter;
    inputMemoryPointers_[mp].clear();
  }
  inputMemoryPointers_.clear();
}

void RegisterMachine::MarkFeatures(instruction *istr, int in) {
  features_.clear();
  if (istr->inType(in) == memoryEigen::SCALAR_TYPE) {
    features_.insert(istr->inIdx(in));
  } else if (istr->inType(in) == memoryEigen::VECTOR_TYPE) {
    for (size_t f = istr->inIdx(in), row = 0;
         row < istr->inMem(in)->memoryRows(); row++) {
      features_.insert(f++ % num_input_);  // toroidal
    }
  } else if (istr->inType(in) == memoryEigen::MATRIX_TYPE) {
    for (size_t f = istr->inIdx(in), row = 0;
         row < istr->inMem(in)->memoryRows(); row++) {
      for (size_t col = 0; col < istr->inMem(in)->memoryCols(); col++) {
        features_.insert(f++ % num_input_);  // toroidal
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
      if (istr->IsMemoryRef(in)) Meff[istr->inType(in)][istr->inIdx(in)] = true;
    }
  }

  bidEffective_.clear();

  for (auto istr : bid_) {
    if (!skipIntrons_ || Meff[istr->outType()][istr->outIdx_]) {
      bidEffective_.push_back(istr);
      op_counts_[istr->op_]++;
      // TODO(spkelly) this is always true now, move elsewhere
      istr->out_ = privateMemory_[istr->outType()];
      // inputs
      for (int in = 0; in < 2; in++) {  // add in arity?
        if (istr->IsInput(in)) {
          istr->inMem(in, inputMemoryPointers_[in][istr->inType(in)]);
          MarkFeatures(istr, in);
        } else if (istr->IsMemoryRef(in)) {
          istr->inMem(in, privateMemory_[istr->inType(in)]);
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
      instr->mutate(true, legalOps, rng);
      uniform_int_distribution<int> disBid(0, bid_.size());
      int i = disBid(rng);
      bid_.insert(bid_.begin() + i, instr);
      changed = true;
    }

    /* Mutate a random instruction. */
    if (disR(rng) < std::any_cast<double>(params["p_bid_mutate"])) {
      uniform_int_distribution<int> disBid(0, bid_.size() - 1);
      int i = disBid(rng);
      bid_[i]->mutate(false, legalOps, rng);
      changed = true;
    }

    /* Add noise to constants */
    if (params.find("p_bid_mu_const") != params.end() &&
        disR(rng) < std::any_cast<double>(params["p_bid_mu_const"])) {
      for (auto m : sharedMemory_) {
        m->NoiseToConst(rng,
                        std::any_cast<double>(params["bid_mu_const_stddev"]));
      }
    }

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

void RegisterMachine::CopyInputToMemory(instruction *istr, state *obs,
                                        size_t in) {
  // in this case inMem(in) will be inputMemory_ and we use index 0
  size_t idx = 0;
  if (istr->inType(in) == memoryEigen::SCALAR_TYPE) {
    istr->inMem(in)->working_memory_[idx](0, 0) =
        obs->stateValueAtIndex(istr->inIdx(in));
  } else if (istr->inType(in) == memoryEigen::VECTOR_TYPE) {
    for (size_t f = istr->inIdx(in), row = 0;
         row < istr->inMem(in)->memoryRows(); row++) {
      istr->inMem(in)->working_memory_[idx](row, 0) =
          obs->stateValueAtIndex(f++ % num_input_);
    }
  } else if (istr->inType(in) == memoryEigen::MATRIX_TYPE) {
    for (size_t f = istr->inIdx(in), row = 0;
         row < istr->inMem(in)->memoryRows(); row++) {
      for (size_t col = 0; col < istr->inMem(in)->memoryCols(); col++) {
        istr->inMem(in)->working_memory_[idx](row, col) =
            obs->stateValueAtIndex(f++ % num_input_);
      }
    }
  }
  istr->inIdxE(in, idx);  // reset inIdxE to zero for input ref
}

/******************************************************************************/
double RegisterMachine::Run(state *obs, int &time_step,
                            const size_t &graph_depth, bool& verbose) {
  bool dbg = verbose;
  // reset memory
  if (!stateful_) CopySharedConstToWorking();

  // Reset the output registers.
  // privateMemory_[memoryEigen::SCALAR_TYPE]->working_memory_[0].setZero();
  // privateMemory_[memoryEigen::SCALAR_TYPE]->working_memory_[1].setZero();

  int i = 0;
  for (auto istr : bidEffective_) {
    // read inputs
    for (size_t in = 0; in < 2; in++) {
      if (istr->inType(in) != memoryEigen::NA_TYPE) {
        if (istr->IsInput(in)) {
          CopyInputToMemory(istr, obs, in);
        } else {  // this input is a memory ref
                  // track read time for temporal memory
          istr->inMem(in)->getReadTimeE()(istr->inIdx(in), 0) =
              time_step + (graph_depth / MAX_GRAPH_DEPTH);
        }
      }
    }
    // track write times for temporal memory
    istr->out_->getWriteTimeE()(istr->outIdx_, 0) =
        time_step + (graph_depth / MAX_GRAPH_DEPTH);

    if (dbg) {
      cerr << "dbg pid " << id_ << " i " << i++ << " ";
    }
    istr->exec(dbg);
  }
  // return bid value
  return privateMemory_[memoryEigen::SCALAR_TYPE]->working_memory_[0](0, 0);
}

/******************************************************************************/
void RegisterMachine::SetupMemory(size_t memoryIndices, size_t memoryRows,
                                  size_t memoryCols) {
  inputMemoryPointers_.resize(2);  // for in1 and in2
  for (int mem_t = 0; mem_t < memoryEigen::NUM_MEMORY_TYPES; mem_t++) {
    privateMemory_.push_back(
        new memoryEigen(-1, mem_t, memoryIndices, memoryRows, memoryCols));
    inputMemoryPointers_[0].push_back(
        new memoryEigen(-1, mem_t, memoryIndices, memoryRows, memoryCols));
    inputMemoryPointers_[1].push_back(
        new memoryEigen(-1, mem_t, memoryIndices, memoryRows, memoryCols));
  }
  sharedMemory_.resize(memoryEigen::NUM_MEMORY_TYPES);
}
