#include "linearM.h"

/******************************************************************************/
string linearM::checkpoint(bool all) {
  ostringstream oss;

  oss << "linearM:" << id_ << ":" << gtime_ << ":" << action_ << ":"
      << stateful_ << ":" << num_input_ << ":" << nrefs_;
  for (size_t mem_t = 0; mem_t < memoryEigen::NUM_MEMORY_TYPES; mem_t++)
    oss << ":" << sharedMemoryPointers_[mem_t]->id();

  if (all)
    for (size_t i = 0; i < bid_.size(); i++)
      oss << ":" << bid_[i]->checkpoint();
  else
    for (size_t i = 0; i < bidEffective_.size(); i++)
      oss << ":" << bidEffective_[i]->checkpoint();
  oss << endl;

  return oss.str();
}

/******************************************************************************/
linearM::linearM(long gtime, long action,
                 std::unordered_map<std::string, std::any> &params, long id,
                 mt19937 &rng, std::vector<bool> &legalOps) {
  action_ = action;
  num_input_ = std::any_cast<int>(params["n_input"]);
  memoryRows_ = std::any_cast<int>(params["memory_rows"]);
  memoryCols_ = std::any_cast<int>(params["memory_cols"]);
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
  setupMemory(std::any_cast<int>(params["memory_indices"]), memoryRows_,
              memoryCols_);
}

/******************************************************************************/
linearM::linearM(long gtime, linearM &plr,
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
  stateful_ = plr.stateful();

  for (auto initer = plr.bid_.begin(); initer != plr.bid_.end(); initer++)
    bid_.push_back(new instruction(**initer));

  op_counts_.resize(instruction::NUM_OP);
  setupMemory(std::any_cast<int>(params["memory_indices"]), memoryRows_,
              memoryCols_);
}

/******************************************************************************/
// Create linearM from checkpoint file
linearM::linearM(long gtime, long action, int stateful,
                 std::unordered_map<std::string, std::any> &params, long id,
                 long nrefs, std::vector<instruction *> bid) {
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

  setupMemory(std::any_cast<int>(params["memory_indices"]),
              std::any_cast<int>(params["memory_rows"]),
              std::any_cast<int>(params["memory_cols"]));
}

/******************************************************************************/
linearM::~linearM() {
  for (auto biditer = bid_.begin(); biditer != bid_.end(); biditer++)
    delete *biditer;
  bid_.clear();
  for (auto meiter = privateMemoryPointers_.begin();
       meiter != privateMemoryPointers_.end(); meiter++)
    delete *meiter;
  privateMemoryPointers_.clear();
  for (size_t mp = 0; mp < tmpMemoryPointers_.size(); mp++) {
    for (auto meiter = tmpMemoryPointers_[mp].begin();
         meiter != tmpMemoryPointers_[mp].end(); meiter++)
      delete *meiter;
    tmpMemoryPointers_[mp].clear();
  }
  tmpMemoryPointers_.clear();
}

/******************************************************************************/
void linearM::markIntrons(bool continuousOutput) {
  fill(op_counts_.begin(), op_counts_.end(), 0);

  map<int, vector<bool> > targets;  //[memory type][index]->true/false
  for (size_t mem_t = 0; mem_t < memoryEigen::NUM_MEMORY_TYPES; mem_t++)
    targets[mem_t] = vector<bool>(
        privateMemoryPointers_[memoryEigen::SCALAR_TYPE]->indexSize(), false);

  features_.clear();
  bidEffective_.clear();

  targets[memoryEigen::SCALAR_TYPE][0] = true;  // mark the bid output register
  if (continuousOutput)
    sharedMemoryPointers_[memoryEigen::SCALAR_TYPE]->getActiveE()(0, 0) =
        true;  // return value for continuous output environments

  // From last to first instruction.
  vector<instruction *>::reverse_iterator riter;
  for (riter = bid_.rbegin(); riter != bid_.rend(); riter++) {
    // Intruction is effective if:
    if (!skipIntrons_ ||  // not skipping introns
        targets[(*riter)->outType()]
               [(*riter)->outIdx_] ||  // destination is a memory index used
                                       // later in this program
        (*riter)
            ->outShared())  // destination is an index to shared stateful memory
    {
      bidEffective_.insert(bidEffective_.begin(), *riter);

      op_counts_[(*riter)->op_]++;

      // output
      (*riter)->out_ = (*riter)->outShared()
                           ? sharedMemoryPointers_[(*riter)->outType()]
                           : privateMemoryPointers_[(*riter)->outType()];

      // inputs
      for (int in = 0; in < 2; in++) {
        if ((*riter)->inType(in) !=
            memoryEigen::NA_TYPE) {  // this input is actually used for this op
          if ((*riter)->isInput(in)) {  // this input is a feature ref
            // set memory pointer
            (*riter)->inMem(in, tmpMemoryPointers_[in][(*riter)->inType(in)]);
            // mark features
            if ((*riter)->inType(in) == memoryEigen::SCALAR_TYPE)
              features_.insert((*riter)->inIdx(in));
            else if ((*riter)->inType(in) == memoryEigen::VECTOR_TYPE)
              for (size_t f = (*riter)->inIdx(in), row = 0;
                   row < (*riter)->inMem(in)->memoryRows(); row++)
                features_.insert(f++ % num_input_);  // toroidal
            else if ((*riter)->inType(in) == memoryEigen::MATRIX_TYPE)
              for (size_t f = (*riter)->inIdx(in), row = 0;
                   row < (*riter)->inMem(in)->memoryRows(); row++)
                for (size_t col = 0; col < (*riter)->inMem(in)->memoryCols();
                     col++)
                  features_.insert(f++ % num_input_);  // toroidal
          } else if ((*riter)->inShared(
                         in)) {  // this input is a shared memory ref
            (*riter)->inMem(in, sharedMemoryPointers_[(*riter)->inType(in)]);
            (*riter)->inMem(in)->getActiveE()((*riter)->inIdx(in), 0) = true;
            targets[(*riter)->inType(in)][(*riter)->inIdx(in)] = true;
          } else {  // this input is an internal memory ref
            (*riter)->inMem(in, privateMemoryPointers_[(*riter)->inType(in)]);
            targets[(*riter)->inType(in)][(*riter)->inIdx(in)] = true;
          }
        }
      }
    }
  }
}

/******************************************************************************/
void linearM::MuBid(std::unordered_map<std::string, std::any> &params,
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

    /* Flip single bit of random instruction. */
    if (disR(rng) < std::any_cast<double>(params["p_bid_mutate"])) {
      uniform_int_distribution<int> disBid(0, bid_.size() - 1);
      int i = disBid(rng);
      bid_[i]->mutate(false, legalOps, rng);
      changed = true;
    }

    /* Add noise to constants */
    if (params.find("p_bid_mu_const") != params.end() &&
        disR(rng) < std::any_cast<double>(params["p_bid_mu_const"])) {
      for (auto m : privateMemoryPointers_)
        m->NoiseToConst(rng,
                        std::any_cast<double>(params["bid_mu_const_stddev"]));
    }

    /* Swap positions of two instructions. */
    if (bid_.size() > 1 &&
        disR(rng) < std::any_cast<double>(params["p_bid_swap"])) {
      uniform_int_distribution<int> disBid(0, bid_.size() - 1);
      int i = disBid(rng);
      int j;
      do { j = disBid(rng); } while (i == j);
      std::swap(bid_[i], bid_[j]);
      changed = true;
    }
  }
}

/******************************************************************************/
double linearM::run(state *obs, int timeStep, int graphDepth, mt19937 &rng) {
  (void)rng;
  bool dbg = false;

  if (!stateful_) {
    for (size_t mem_t = 0; mem_t < memoryEigen::NUM_MEMORY_TYPES; mem_t++) {
      sharedMemoryPointers_[mem_t]->CopyConstToWorking();
    }
    CopySharedConstToWorking();
  }

  for (auto initer = bidEffective_.begin(); initer != bidEffective_.end();
       initer++) {
    // read inputs
    size_t idx = 0;
    for (size_t in = 0; in < 2; in++)
      if ((*initer)->inType(in) !=
          memoryEigen::NA_TYPE) {  // this input is actually used for this op
        if ((*initer)->isInput(in)) {  // this input is a feature ref
          if ((*initer)->inType(in) == memoryEigen::SCALAR_TYPE)
            (*initer)->inMem(in)->working_memory_[idx](0, 0) =
                obs->stateValueAtIndex((*initer)->inIdx(in));
          else if ((*initer)->inType(in) == memoryEigen::VECTOR_TYPE)
            for (size_t f = (*initer)->inIdx(in), row = 0;
                 row < (*initer)->inMem(in)->memoryRows(); row++)
              (*initer)->inMem(in)->working_memory_[idx](row, 0) =
                  obs->stateValueAtIndex(
                      f++ % num_input_);  //(*feature)[f++ % num_input_];
          else if ((*initer)->inType(in) == memoryEigen::MATRIX_TYPE)
            for (size_t f = (*initer)->inIdx(in), row = 0;
                 row < (*initer)->inMem(in)->memoryRows(); row++)
              for (size_t col = 0; col < (*initer)->inMem(in)->memoryCols();
                   col++)
                (*initer)->inMem(in)->working_memory_[idx](row, col) =
                    obs->stateValueAtIndex(f++ % num_input_);
          (*initer)->inIdxE(in, idx);  // reset inIdxE to zero for input ref
        } else {                       // this input is a memory ref
          // track read time for temporal memory
          if ((*initer)->inShared(in))
            (*initer)->inMem(in)->getReadTimeE()((*initer)->inIdx(in), 0) =
                timeStep + (graphDepth / MAX_GRAPH_DEPTH);
        }
      }
    // track write times for temporal memory
    if ((*initer)->outShared())
      (*initer)->out_->getWriteTimeE()((*initer)->outIdx_, 0) =
          timeStep + (graphDepth / MAX_GRAPH_DEPTH);

    (*initer)->exec(dbg);
  }
  return privateMemoryPointers_[memoryEigen::SCALAR_TYPE]->working_memory_[0](
      0, 0);
}

/******************************************************************************/
void linearM::setupMemory(size_t memoryIndices, size_t memoryRows,
                          size_t memoryCols) {
  tmpMemoryPointers_.resize(2);  // for in1 and in2
  for (size_t mem_t = 0; mem_t < memoryEigen::NUM_MEMORY_TYPES; mem_t++) {
    privateMemoryPointers_.push_back(
        new memoryEigen(-1, mem_t, memoryIndices, memoryRows, memoryCols));
    tmpMemoryPointers_[0].push_back(
        new memoryEigen(-1, mem_t, memoryIndices, memoryRows, memoryCols));
    tmpMemoryPointers_[1].push_back(
        new memoryEigen(-1, mem_t, memoryIndices, memoryRows, memoryCols));
  }
  sharedMemoryPointers_.resize(memoryEigen::NUM_MEMORY_TYPES);
}
