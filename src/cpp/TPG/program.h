#ifndef program_h
#define program_h
#include <memoryEigen.h>
#include <state.h>

#include <random>

#define MAX_GRAPH_DEPTH 1000.0

class program {
 public:
  int action_;         // Action index
  double bid_val_;     // Most recent bid value
  static long count_;  // Next id to use
  size_t memory_size_;
  // vector<double>* feature;

  // Features indexed by non-introns in this program, determined in
  // MarkIntrons().
  set<long> features_;
  // Features indexed by non-introns that write to memoryEigen, determined in
  // MarkIntrons().
  set<long> featuresMem_;
  long gtime_;
  long id_;
  double key_;
  int lastCompareFactor_;

  vector<memoryEigen *> sharedMemory_;
  vector<memoryEigen *> privateMemory_;

  // read inputs into these at runtime
  vector<vector<memoryEigen *> > inputMemoryPointers_;

  int nrefs_;               //  Number of references by teams
  vector<int> op_counts_;   // count for each operator over _bidEffective
  vector<double> profile_;  // Bid profile
  bool skipIntrons_;
  bool stateful_;
  // Set to true in MarkIntrons if this program writes to stateful memoryEigen.
  bool targetMem_;

  inline int action() { return action_; }
  inline void action(int a) { action_ = a; }
  virtual double Run(state *s, int &time_step, const size_t &graph_depth,
                     bool &verbose) = 0;
  inline double bidVal() { return bid_val_; }
  inline void bidVal(double b) { bid_val_ = b; }
  virtual string checkpoint(bool) = 0;
  inline void features(set<long> &f) { f = features_; }
  inline void featuresMem(set<long> &f) { f = featuresMem_; }
  inline void getProfile(vector<double> &p) { p = profile_; }
  inline long gtime() { return gtime_; }
  inline long id() { return id_; }
  inline void id(long i) { id_ = i; }
  inline double key() { return key_; }
  inline void key(double key) { key_ = key; }
  inline int lastCompareFactor() { return lastCompareFactor_; }
  inline void lastCompareFactor(int c) { lastCompareFactor_ = c; }
  virtual ~program(){};

  virtual void MarkIntrons(std::unordered_map<std::string, std::any> &) = 0;

  inline void MemGet(size_t type, memoryEigen *&m) { m = sharedMemory_[type]; }

  inline void MemSet(uint8_t type, memoryEigen *m) {
    sharedMemory_[type] = m;
    m->refInc();
  }

  inline memoryEigen *MemGet(uint8_t type) { return sharedMemory_[type]; }

  inline void CopySharedConstToWorking() {
    for (size_t i = 0; i < sharedMemory_.size(); i++) {
      privateMemory_[i]->working_memory_ = sharedMemory_[i]->const_memory_;
    }
  }

  inline void ClearWorking() {
    for (size_t i = 0; i < privateMemory_.size(); i++) {
      privateMemory_[i]->ClearWorking();
    }
  }

  // Mutate action, return true if the action was actually changed
  inline bool muAction(long action) {
    long a = action_;
    action_ = action;
    return a != action;
  }
  // Mutate bid, return true if any changes occured
  virtual void MuBid(std::unordered_map<std::string, std::any> &, mt19937 &,
                     uniform_real_distribution<> &, vector<bool> &) = 0;
  // Not counting introns
  inline long numFeatures() { return features_.size(); }
  inline void op_counts(vector<int> &v) { v = op_counts_; }
  inline void setId(long id) { id_ = id; }
  inline void setProfile(vector<double> &p) { profile_ = p; }
  virtual int Size() = 0;
  virtual int SizeEffective() = 0;
  inline bool stateful() { return stateful_; }
  inline void stateful(bool s) { stateful_ = s; }
  inline bool targetMem() { return targetMem_; }
};

struct programIdComp {
  bool operator()(program *l1, program *l2) const {
    return l1->id() < l2->id();
  }
};

struct programIdEQ {
  bool operator()(program *l1, program *l2) const {
    return l1->id() == l2->id();
  }
};

struct ProgramBidLexicalCompare {
  bool operator()(program *l1, program *l2) const {
    // most recent bid, higher is better
    if (l1->bidVal() != l2->bidVal()) {
      // l1->lastCompareFactor(0);
      // l2->lastCompareFactor(0);
      return l1->bidVal() > l2->bidVal();
    }
    ////program size post intron removal, smaller is better (assumes MarkIntrons
    /// is up to date)
    // else if (l1->esize() != l2->esize()) {
    //    l1->lastCompareFactor(1);
    //    l2->lastCompareFactor(1);
    //    return l1->esize() < l2->esize();
    // }
    ////number of references, less is better
    // else if (l1->refs() != l2->refs()) {
    //    l1->lastCompareFactor(2);
    //    l2->lastCompareFactor(2);
    //    return l1->refs() < l2->refs();
    // }
    ////number of features indexed, less is better
    // else if (l1->numFeatures() != l2->numFeatures()) {
    //    l1->lastCompareFactor(3);
    //    l2->lastCompareFactor(3);
    //    return l1->numFeatures() < l2->numFeatures();
    // }
    ////age, younger is better
    // else if (l1->gtime() != l2->gtime()) {
    //    l1->lastCompareFactor(4);
    //    l2->lastCompareFactor(4);
    //    return l1->gtime() > l2->gtime();
    // }
    // correlated to age but technically arbirary,
    //  id is guaranteed to be unique and thus ensures deterministic comparison
    else {
      // l1->lastCompareFactor(6);
      // l2->lastCompareFactor(6);
      return l1->id() < l2->id();
    }
  }
};

#endif
