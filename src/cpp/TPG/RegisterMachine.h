#ifndef RegisterMachine_h
#define RegisterMachine_h
#include <any>
#include <bitset>
#include <random>

#include "instruction.h"
#include "memoryEigen.h"
#include "program.h"

struct instructionDecoded;

class RegisterMachine : public program {
 public:
  // Bid program, a list of instructions
  std::vector<instruction *> bid_;
  std::vector<instruction *> bidEffective_;
  inline void AddToInputMemoryBuff(Matrix<double, Dynamic, Dynamic> &mat,
                                   int mem_t) {
    input_memory_buff_[mem_t]->working_memory_.push_front(mat);
    input_memory_buff_[mem_t]->working_memory_.pop_back();
  }
  // void CopyInputToMemory(instruction *istr, state *obs, size_t in);
  void CopyInputToMemoryBuff(state *obs);
  void MarkFeatures(instruction *istr, int in);
  void MarkIntrons(std::unordered_map<std::string, std::any> &params_);
  double Run(state *, int &time_step, const size_t &graph_depth, bool &verbose);
  std::string checkpoint(bool);
  // Create arbitrary RegisterMachine
  RegisterMachine(long, long, std::unordered_map<std::string, std::any> &, long,
                  mt19937 &, std::vector<bool> &);
  // Create RegisterMachine from another RegisterMachine
  RegisterMachine(long, RegisterMachine &,
                  std::unordered_map<std::string, std::any> &, long);
  // Create RegisterMachine from checkpoint file
  RegisterMachine(long, long, int, std::unordered_map<std::string, std::any> &,
                  long, long, std::vector<instruction *>);
  ~RegisterMachine();
  // Mutate bid
  void MuBid(std::unordered_map<std::string, std::any> &, mt19937 &,
             uniform_real_distribution<> &, std::vector<bool> &);
  void SetupMemory(size_t, size_t, size_t);
  inline int Size() { return bid_.size(); }
  inline int SizeEffective() { return bidEffective_.size(); }
};

#endif
