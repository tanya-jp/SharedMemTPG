#ifndef linearM2h
#define linearM_h
#include <any>
#include <bitset>
#include <random>

#include "instruction.h"
#include "memoryEigen.h"
#include "program.h"

struct instructionDecoded;

class linearM : public program {
 public:
  static const int registers_;
  static const int registersRW_;
  static const int registersRO_;

  static const int num_constants_;

  // Bid program, a list of instructions
  std::vector<instruction *> bid_;  
  std::vector<instruction *> bidEffective_;

  void markIntrons(bool);
  double run(state *, int, int, mt19937 &rng);
  std::string checkpoint(bool);
  inline void getBid(std::vector<instruction *> &b) { b = bid_; }
  // Create arbitrary linearM
  linearM(long, long, std::unordered_map<std::string, std::any> &, long,
          mt19937 &, std::vector<bool> &); 
  // Create linearM from another linearM        
  linearM(long, linearM &, std::unordered_map<std::string, std::any> &,
          long); 
  // Create linearM from checkpoint file        
  linearM(
      long, long, int, std::unordered_map<std::string, std::any> &, long, long,
      std::vector<instruction *>); 
  ~linearM();
  // Mutate bid
  void MuBid(std::unordered_map<std::string, std::any> &, mt19937 &,
             uniform_real_distribution<> &,
             std::vector<bool>
                 &); 
  void setupMemory(size_t, size_t, size_t);
  inline int size() { return bid_.size(); }
  inline int esize() { return bidEffective_.size(); }
};

#endif
