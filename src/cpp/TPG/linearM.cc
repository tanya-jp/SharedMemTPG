#include "linearM.h"

/******************************************************************************/
string linearM::checkpoint(bool all) {
  ostringstream oss;

  oss << "linearM:" << id_ << ":" << gtime_ << ":" << action_ << ":"
      << stateful_ << ":" << num_input_ << ":" << nrefs_;
  for (int mem_t = 0; mem_t < memoryEigen::NUM_MEMORY_TYPES; mem_t++) {
    oss << ":" << sharedMemoryPointers_[mem_t]->id();
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

/******************************************************************************/
linearM::linearM(long gtime, long action,
                 std::unordered_map<std::string, std::any> &params, long id,
                 mt19937 &rng, std::vector<bool> &legalOps) {
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
  stateful_ = plr.stateful_;

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
  for (size_t mp = 0; mp < inputMemoryPointers_.size(); mp++) {
    for (auto meiter = inputMemoryPointers_[mp].begin();
         meiter != inputMemoryPointers_[mp].end(); meiter++)
      delete *meiter;
    inputMemoryPointers_[mp].clear();
  }
  inputMemoryPointers_.clear();
}

// /******************************************************************************
//  * Markus F. Brameier and Wolfgang Banzhaf. 2010.
//  * Linear Genetic Programming (1st. ed.). Springer Publishing Company, Inc.
//  * Algorithm 3.1 (detection of structural introns)
//  */
// void linearM::MarkIntrons(std::unordered_map<std::string, std::any> &params)
// {
//   fill(op_counts_.begin(), op_counts_.end(), 0);  // count occurance of each
//   op

//   // keep track of which memories are effective with a map
//   map<int, vector<bool> > Reff;  // maps [memory type][index]->true/false

//   for (int mem_t = 0; mem_t < memoryEigen::NUM_MEMORY_TYPES; mem_t++)
//     Reff[mem_t] =
//         vector<bool>(std::any_cast<int>(params["memory_indices"]), false);

//   Reff[memoryEigen::SCALAR_TYPE][0] = true;  // mark bid output memory

//   if (std::any_cast<int>(params["continuous_output"]))
//     Reff[memoryEigen::SCALAR_TYPE][1] = true;  // mark continuous output
//     memory

//   features_.clear();
//   bidEffective_.clear();

//   // From last to first instruction.
//   // vector<instruction *>::reverse_iterator riter;
//   for (auto riter = bid_.rbegin(); riter != bid_.rend(); riter++) {
//     if (!skipIntrons_ || Reff[(*riter)->outType()][(*riter)->outIdx_]) {
//       bidEffective_.insert(bidEffective_.begin(), *riter);
//       op_counts_[(*riter)->op_]++;
//       // output TODO(spkelly) this is always true now
//       (*riter)->out_ = privateMemoryPointers_[(*riter)->outType()];
//       // inputs
//       for (int in = 0; in < 2; in++) {
//         // if this input is actually used for this op
//         if ((*riter)->inType(in) != memoryEigen::NA_TYPE) {
//           if ((*riter)->isInput(in)) {  // this input is a feature ref
//             (*riter)->inMem(in,
//             inputMemoryPointers_[in][(*riter)->inType(in)]);
//             // mark features (accounting, should have no effect behaviour)
//             if ((*riter)->inType(in) == memoryEigen::SCALAR_TYPE) {
//               features_.insert((*riter)->inIdx(in));
//             } else if ((*riter)->inType(in) == memoryEigen::VECTOR_TYPE) {
//               for (size_t f = (*riter)->inIdx(in), row = 0;
//                    row < (*riter)->inMem(in)->memoryRows(); row++) {
//                 features_.insert(f++ % num_input_);  // toroidal
//               }
//             } else if ((*riter)->inType(in) == memoryEigen::MATRIX_TYPE) {
//               for (size_t f = (*riter)->inIdx(in), row = 0;
//                    row < (*riter)->inMem(in)->memoryRows(); row++) {
//                 for (size_t col = 0; col < (*riter)->inMem(in)->memoryCols();
//                      col++) {
//                   features_.insert(f++ % num_input_);  // toroidal
//                 }
//               }
//             }
//           } else {  // this input is a memory ref
//             (*riter)->inMem(in,
//             privateMemoryPointers_[(*riter)->inType(in)]);
//             Reff[(*riter)->inType(in)][(*riter)->inIdx(in)] = true;
//           }
//         }
//       }
//     }
//   }
// }

/******************************************************************************/
void linearM::MarkIntrons(std::unordered_map<std::string, std::any> &params) {
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
  for (auto ins : bid_)
    for (int in = 0; in < 2; in++)
      if (!(ins->isInput(in)) && ins->inType(in) != memoryEigen::NA_TYPE) Meff[ins->inType(in)][ins->inIdx(in)] = true;

  features_.clear();
  bidEffective_.clear();

  for (auto riter : bid_) {
    if (!skipIntrons_ || Meff[riter->outType()][riter->outIdx_]) {
      bidEffective_.push_back(riter);
      op_counts_[riter->op_]++;
      // output TODO(spkelly) this is always true now
      riter->out_ = privateMemoryPointers_[riter->outType()];
      // inputs
      for (int in = 0; in < 2; in++) {  // add in arity?
        if (riter->isInput(in)) {       // this input is a feature ref
          riter->inMem(in, inputMemoryPointers_[in][riter->inType(in)]);
          // // mark features (accounting, should have no effect behaviour)
          // if ((*riter)->inType(in) == memoryEigen::SCALAR_TYPE) {
          //   features_.insert((*riter)->inIdx(in));
          // } else if ((*riter)->inType(in) == memoryEigen::VECTOR_TYPE) {
          //   for (size_t f = (*riter)->inIdx(in), row = 0;
          //        row < (*riter)->inMem(in)->memoryRows(); row++) {
          //     features_.insert(f++ % num_input_);  // toroidal
          //   }
          // } else if (riter->inType(in) == memoryEigen::MATRIX_TYPE) {
          //   for (size_t f = (*riter)->inIdx(in), row = 0;
          //        row < (*riter)->inMem(in)->memoryRows(); row++) {
          //     for (size_t col = 0; col < (*riter)->inMem(in)->memoryCols();
          //          col++) {
          //       features_.insert(f++ % num_input_);  // toroidal
          //     }
          //   }
          // }
        } else if (riter->inType(in) !=
                   memoryEigen::NA_TYPE) {  // this input is a memory ref
          riter->inMem(in, privateMemoryPointers_[riter->inType(in)]);
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
      do {
        j = disBid(rng);
      } while (i == j);
      std::swap(bid_[i], bid_[j]);
      changed = true;
    }
  }
}

/******************************************************************************/
double linearM::run(state *obs, int timeStep, int graphDepth, mt19937 &rng) {
  (void)rng;
  bool dbg = false;
  if (dbg) cerr << "id: " << id_ << " run:" << endl;

  // reset memory, intron removal is only consistent if we set memory here
  if (!stateful_) CopySharedConstToWorking();

  inputMemoryPointers_[0][memoryEigen::SCALAR_TYPE]->ClearWorking();
  inputMemoryPointers_[0][memoryEigen::VECTOR_TYPE]->ClearWorking();
  inputMemoryPointers_[0][memoryEigen::MATRIX_TYPE]->ClearWorking();
  inputMemoryPointers_[1][memoryEigen::SCALAR_TYPE]->ClearWorking();
  inputMemoryPointers_[1][memoryEigen::VECTOR_TYPE]->ClearWorking();
  inputMemoryPointers_[1][memoryEigen::MATRIX_TYPE]->ClearWorking();

  privateMemoryPointers_[memoryEigen::SCALAR_TYPE]
      ->working_memory_[0]
      .setZero();
  privateMemoryPointers_[memoryEigen::SCALAR_TYPE]
      ->working_memory_[1]
      .setZero();
  // privateMemoryPointers_[memoryEigen::SCALAR_TYPE]->ClearWorking();

  for (auto i : bidEffective_) {
    // read inputs
    size_t idx = 0;
    for (size_t in = 0; in < 2; in++) {
      if (i->inType(in) != memoryEigen::NA_TYPE) {
        if (i->isInput(in)) {  // this input is a feature ref
          if (dbg) cerr << "in" << in << " fRef ";
          // in this case inMem(in) will be inputMemory_ and we use index 0
          if (i->inType(in) == memoryEigen::SCALAR_TYPE)
            i->inMem(in)->working_memory_[idx](0, 0) =
                obs->stateValueAtIndex(i->inIdx(in));
          else if (i->inType(in) == memoryEigen::VECTOR_TYPE)
            for (size_t f = i->inIdx(in), row = 0;
                 row < i->inMem(in)->memoryRows(); row++)
              i->inMem(in)->working_memory_[idx](row, 0) =
                  obs->stateValueAtIndex(
                      f++ % num_input_);  //(*feature)[f++ % num_input_];
          else if (i->inType(in) == memoryEigen::MATRIX_TYPE)
            for (size_t f = i->inIdx(in), row = 0;
                 row < i->inMem(in)->memoryRows(); row++)
              for (size_t col = 0; col < i->inMem(in)->memoryCols(); col++)
                i->inMem(in)->working_memory_[idx](row, col) =
                    obs->stateValueAtIndex(f++ % num_input_);
          i->inIdxE(in, idx);  // reset inIdxE to zero for input ref
        } else {               // this input is a memory ref
          if (dbg) cerr << "in" << in << " mRef ";
          // // track read time for temporal memory
          // if ((*initer)->inShared(in))
          //   (*initer)->inMem(in)->getReadTimeE()((*initer)->inIdx(in), 0) =
          //       timeStep + (graphDepth / MAX_GRAPH_DEPTH);
        }
      }
    }
    if (dbg) cerr << endl;
    // // track write times for temporal memory
    // if ((*initer)->outShared())
    //   (*initer)->out_->getWriteTimeE()((*initer)->outIdx_, 0) =
    //       timeStep + (graphDepth / MAX_GRAPH_DEPTH);

    i->exec(dbg);
  }
  if (dbg) {
    cerr << "id: " << id_ << " outs ";
    cerr << privateMemoryPointers_[memoryEigen::SCALAR_TYPE]
                ->working_memory_[0](0, 0);
    cerr << " ";
    cerr << privateMemoryPointers_[memoryEigen::SCALAR_TYPE]
                ->working_memory_[1](0, 0);
    cerr << endl;
  }
  return privateMemoryPointers_[memoryEigen::SCALAR_TYPE]->working_memory_[0](
      0, 0);
}

/******************************************************************************/
void linearM::setupMemory(size_t memoryIndices, size_t memoryRows,
                          size_t memoryCols) {
  inputMemoryPointers_.resize(2);  // for in1 and in2
  for (int mem_t = 0; mem_t < memoryEigen::NUM_MEMORY_TYPES; mem_t++) {
    privateMemoryPointers_.push_back(
        new memoryEigen(-1, mem_t, memoryIndices, memoryRows, memoryCols));
    inputMemoryPointers_[0].push_back(
        new memoryEigen(-1, mem_t, memoryIndices, memoryRows, memoryCols));
    inputMemoryPointers_[1].push_back(
        new memoryEigen(-1, mem_t, memoryIndices, memoryRows, memoryCols));
  }
  sharedMemoryPointers_.resize(memoryEigen::NUM_MEMORY_TYPES);
}
