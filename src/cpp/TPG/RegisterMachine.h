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
    observation_memory_buff_[mem_t]->working_memory_.push_front(mat);
    observation_memory_buff_[mem_t]->working_memory_.pop_back();
  }
  void CopyObservationToMemoryBuff(state *obs);
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
                  long, long, int, std::vector<instruction *>);
  ~RegisterMachine();
  // Mutate bid
  void Mutate(std::unordered_map<std::string, std::any> &, mt19937 &, std::vector<bool> &);
  void SetupMemory(size_t memoryIndices, int observation_buff_size, size_t memory_size);
  void MutateObsBuffSize(size_t max_observation_buff_size, mt19937& rng);
  inline int Size() { return bid_.size(); }
  inline int SizeEffective() { return bidEffective_.size(); }
};

#endif
