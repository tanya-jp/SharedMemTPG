#ifndef team_h
#define team_h

#include <list>
#include <map>
#include <queue>
#include <set>
#include <unordered_set>

#include "misc.h"
#include "point.h"
#include "program.h"
#include "state.h"

#define MEMBERS_RUN_ENTROPY_INDEX 3
#define NUM_TEAM_DISTANCE_MEASURES 3
// arbitrarily large, greater than max ALE frames/episode of 18000
#define MAX_NCD_PROFILE_SIZE 20000

using namespace std;

struct teamIdComp;

class team {
 public:
  inline void activeMembers(set<program *, programIdComp> *m) const {
    m->clear();
    m->insert(active_.begin(), active_.end());
  }
  inline void addAncestorId(long aid) { ancestorIds_.push_back(aid); }
  inline void addEvalSeed(int s) { evalSeeds_.push_back(s); }
  inline void clearEvalSeeds() { evalSeeds_.clear(); }
  void clearMemory(map<long, team *> &);
  void InitMemory(map<long, team *> &);
  inline void getAncestorIds(vector<long> &a) { a = ancestorIds_; }
  inline void setAncestorIds(vector<long> &a) { ancestorIds_ = a; }
  inline int numAncestorIds() { return ancestorIds_.size(); }
  inline bool elite(int phase) const { return elite_[phase]; }
  inline void elite(int phase, bool e) { elite_[phase] = e; }
  inline void getActiveMembersByRef(set<program *, programIdComp> &m) const {
    m = active_;
  }
  inline string getBehaviourString(int seed, int phase) {
    string s = "";
    for (size_t task = 0; task < outcomes_.size(); task++)
      if (outcomes_[task][phase].find(seed) != outcomes_[task][phase].end())
        s += outcomes_[task][phase][seed]->getBehaviourString();
    return s;
  }
  bool hasPointDesc(string, int, int);
  double getPointDescScore(string, int, int, int);
  // The number of active programs in this team
  inline int asize() const { return active_.size(); }
  void cleanup(map<long, team *> &, deque<program *> &);
  void features(set<long> &) const;
  inline string fitnessBin() const { return (fitnessBins_.rbegin())->second; }
  inline string fitnessBin(long t) const {
    auto search = fitnessBins_.find(t);
    if (search != fitnessBins_.end())
      return search->second;
    else
      return "NA";  // code for not found
  }
  inline void fitnessBin(long t, string task) { fitnessBins_[t] = task; }
  inline void fitnessBins(map<long, string> &fb) {
    fitnessBins_.clear();
    fitnessBins_.insert(fb.begin(), fb.end());
  }
  inline map<long, string> fitnessBins() { return fitnessBins_; }
  void getAllMemories(map<long, team *> &, set<team *, teamIdComp> &,
                      set<memoryEigen *, memoryEigenIdComp> &, bool) const;
  void getAllNodes(map<long, team *> &teamMap, set<team *, teamIdComp> &, long,
                   bool) const;
  void getAllNodes(map<long, team *> &teamMap, set<team *, teamIdComp> &,
                   set<program *, programIdComp> &) const;
  void getAllNodes(map<long, team *> &teamMap, set<team *, teamIdComp> &,
                   set<program *, programIdComp> &,
                   set<memoryEigen *, memoryEigenIdComp> &, bool) const;
  void getBehaviourSequence(vector<int> &, int);
  double getMeanOutcome(int, int, int, bool, bool);
  double getMeanOutcome(int, int, int, int, long, bool, bool);
  inline bool hasOutcome(int task, int phase, int seed) {
    return outcomes_[task][phase].find(seed) != outcomes_[task][phase].end();
  }
  inline int inDeg() const { return incomingPrograms_.size(); }
  inline void addIncomingProgram(long id) { incomingPrograms_.insert(id); }
  inline void removeIncomingProgram(long id) { incomingPrograms_.erase(id); }
  inline void incomingPrograms(set<long> &icp) const {
    icp = incomingPrograms_;
  }
  inline void members(set<program *, programIdComp> *m) const {
    m->clear();
    copy(members_.begin(), members_.end(), inserter(*m, m->end()));
  }
  inline void members(list<program *> *m) const {
    m->assign(members_.begin(), members_.end());
  }
  inline void members(set<program *, programIdComp> &m) const {
    m.clear();
    copy(members_.begin(), members_.end(), inserter(m, m.end()));
  }
  // double membersRunEntropy(){
  //    double e = 0;
  //    for ( auto it = membersRun_Tally.begin(); it != membersRun_Tally.end();
  //    it++ ){
  //       if (it->second > 0){
  //          it->second /= _visitedCount;
  //          e += it->second * log2(it->second);
  //       }
  //    }
  //    return -e;
  // }
  inline void members(vector<program *> &m) const {
    m.assign(members_.begin(), members_.end());
  }
  inline void getMembersRef(list<program *> *&m) { m = &members_; }
  inline int numEffectiveInstructions() const {
    return numEffectiveInstructions_;
  }
  double novelty(int, int) const;
  // Number of outcomes
  int numOutcomes(int phase, int task);
  // Get all outcomes from a particular phase
  // void outcomes(map<point*, double, pointLexicalLessThan>&, int);
  // Get all outcomes from a particular phase
  inline void outcomes(map<int, map<int, map<int, point *>>> &o) {
    o = outcomes_;
  }
  // void outcomes(int, int, vector < double >&); /* Get all outcome values of a
  // particular type and from a particular phase.*/
  inline void policyFeaturesGet(set<long> &f, bool active) const {
    if (active)
      f = policyFeaturesActive_;
    else
      f = policyFeatures_;
  }
  inline void policyFeaturesSet(set<long> &f, bool active) {
    if (active)
      policyFeaturesActive_ = f;
    else
      policyFeatures_ = f;
  }
  inline set<long> policyRootIds() const { return policyRootIds_; }
  inline void policyRootIds(set<long> &prids) const { prids = policyRootIds_; }
  inline void resetPolicyRootIds() { policyRootIds_.clear(); }
  inline void addPolicyRootIds(set<long> ptids) {
    policyRootIds_.insert(ptids.begin(), ptids.end());
  }
  inline void addPolicyRootId(long rootId) { policyRootIds_.insert(rootId); }
  void updatePolicyRoot(map<long, team *> &teamMap, set<team *, teamIdComp> &,
                        long &);
  int policyFeatures(
      map<long, team *> &teamMap, set<team *, teamIdComp> &, set<long> &,
      bool) const;  // populates last set with features and returns number of
                    // nodes(programs) in policy
  void policyInstructions(map<long, team *> &, set<team *, teamIdComp> &,
                          vector<int> &, vector<int> &) const;
  void prunePrograms(deque<program *> &);
  bool removeProgram(program *);
  void resetOutcomes(int); /* Delete all outcomes from phase. */
  inline bool root() const { return incomingPrograms_.size() == 0; }
  inline double runTimeComplexityIns() const { return runTimeComplexityIns_; }
  inline void runTimeComplexityIns(double rtc) { runTimeComplexityIns_ = rtc; }
  inline double runTimeComplexityTms() const { return runTimeComplexityTms_; }
  inline void runTimeComplexityTms(double rtc) { runTimeComplexityTms_ = rtc; }
  inline void setActive(program *l) { active_.insert(l); }
  inline bool hasQuickMean(int task, int fitMode, int phase) {
    return quickMeans_.count(task) > 0 &&
           quickMeans_[task].count(fitMode) > 0 &&
           quickMeans_[task][fitMode].count(phase) > 0;
  }
  inline bool hasQuickSum(int task, int fitMode, int phase) {
    return quickSums_.count(task) > 0 && quickSums_[task].count(fitMode) > 0 &&
           quickSums_[task][fitMode].count(phase) > 0;
  }
  inline void setQuickMean(int task, int fitMode, int phase, double m) {
    quickMeans_[task][fitMode][phase] = m;
  }
  inline double getQuickMean(int task, int fitMode, int phase) {
    return quickMeans_[task][fitMode][phase];
  }
  inline void unSetActive(program *l) { active_.erase(l); }
  void setOutcome(point *);
  inline int size() const { return members_.size(); }
  double symbiontUtilityDistance(team *) const;
  double symbiontUtilityDistance(vector<long> &) const;
  inline void swapProgramOrder(int i, int j) {
    auto leiter_i = members_.begin();
    auto leiter_j = members_.begin();
    advance(leiter_i, i);
    advance(leiter_j, j);
    swap(*leiter_i, *leiter_j);
  }
  void swapOutcomePhase(int, int, int, long);
  inline void muProgramOrder(int source, int destination) {
    auto leiter_source = members_.begin();
    auto leiter_destination = members_.begin();
    advance(leiter_source, source);
    advance(leiter_destination, destination);
    members_.splice(leiter_destination, members_, leiter_source);
  }

  void updateComplexityRecord(map<long, team *> &, int);
  void updateComplexityRecord(map<long, team *> &, int, int, long, int);

  string taskCode() const { return task_code_; }
  void taskCode(string tc) { task_code_ = tc; }

  // this constructor used for initialization and checkpointing
  team(long gtime, long id) {
    cloneId_ = -1;
    clones_ = 0;
    //_depthSum = 0;
    for (int i = 0; i < _NUM_PHASE; i++) elite_.push_back(false);
    //_visitedCount = 0;
    domBy_ = -1;
    domOf_ = -1;
    fit_ = 0.0;
    gtime_ = gtime;
    id_ = id;
    key_ = 0;
    lastCompareFactor_ = -1;
    numAtomic_ = 0;
    _n_eval = 0;
    numLinearM_ = 0;
    root_ = true;
    runTimeComplexityIns_ = 0;
    runTimeComplexityTms_ = 0;
    // ancestorIds_.reserve(200);
  };

  // Affects program refs, unlike addProgram() and removeProgram()
  ~team(){};

  inline void addDistance(int type, double d) {
    if (type == 0)
      distances_0_.insert(d);
    else if (type == 1)
      distances_1_.insert(d);
    else if (type == 2)
      distances_2_.insert(d);
  }
  bool addProgram(program *, int i = -1);
  bool addProgramActive(program *);
  string checkpoint(bool, long id = -1) const;
  void clone(map<long, phyloRecord> &, team **);
  inline void clearDistances() {
    distances_0_.clear();
    distances_1_.clear();
    distances_2_.clear();
  }
  // void deleteOutcome(point *); /* Delete outcome. */
  program *getAction(state *, map<long, team *> &, bool,
                     set<team *, teamIdComp> &, long &, int, vector<team *> &,
                     mt19937 &);
  program *getAction(state *, map<long, team *> &, bool,
                     set<team *, teamIdComp> &, long &, int,
                     vector<program *> &, vector<program *> &,
                     vector<set<long>> &,
                     vector<set<memoryEigen *, memoryEigenIdComp>> &,
                     vector<team *> &, mt19937 &);
  // double ncdBehaviouralDistance(team*, int);
  void shuff(mt19937 &rng) {
    vector<program *> vec(members_.begin(), members_.end());
    shuffle(vec.begin(), vec.end(), rng);
    list<program *> shuffled_list{vec.begin(), vec.end()};
    members_.swap(shuffled_list);
  }
  void updateActiveMembersFromIds(vector<long> &);

  // protected:

  set<program *, programIdComp>
      active_; /* Active member programs, a subset of members_, activated in
                  getAction(). */
  vector<long> ancestorIds_;
  long cloneId_;
  long clones_;
  vector<bool> elite_;
  multiset<double> distances_0_;
  multiset<double> distances_1_;
  multiset<double> distances_2_;
  int domBy_; /* Number of other teams dominating this team. */
  int domOf_; /* Number of other teams that this team dominates. */
  vector<int> evalSeeds_;
  // map of t -> task# keeps track of which fitness bin this
  // team was protected by in each gen (-1 for multi-task)
  map<long, string> fitnessBins_;
  double fit_; /* Fitness value used for selection */
  double fitProb_;
  long gtime_; /* Time step at which generated. */
  long id_;    /* Unique id of team. */
  set<long> incomingPrograms_;
  double key_; /* For sorting. */
  int lastCompareFactor_;
  list<program *> members_;
  vector<program *> membersRun_;
  // map <long, double > membersRun_Tally;//keep count of wins for each program
  int numAtomic_;
  int _n_eval;
  int numLinearM_;
  int numActiveTeams_;
  int numActivePrograms_;
  int numEffectiveInstructions_;
  int numActiveFeatures_;
  // Maps point[task][phase][envSeed] -> outcome
  map<int, map<int, map<int, point *>>> outcomes_;
  set<long> policyFeatures_;
  set<long> policyFeaturesActive_;
  set<long> policyRootIds_;
  map<int, map<int, map<int, double>>> quickMeans_;
  map<int, map<int, map<int, double>>> quickSums_;
  bool root_;
  double runTimeComplexityIns_;
  double runTimeComplexityTms_;
  string task_code_;
};

struct teamIdComp {
  bool operator()(team *t1, team *t2) const { return t1->id_ > t2->id_; }
};

struct teamFitnessCompare {
  bool operator()(team *t1, team *t2) const {
    // double d1, d2;
    if (!isEqual(t1->fit_, t2->fit_)) {
      return t1->fit_ > t2->fit_;
    } else {
      return t1->id_ > t2->id_;
    }
  }
};

struct teamFitnessLexicalCompare {
  bool operator()(team *t1, team *t2) const {
    // double d1, d2;
    if (!isEqual(t1->fit_, t2->fit_)) {
      t1->lastCompareFactor_ = 1;
      t2->lastCompareFactor_ = 1;
      return t1->fit_ > t2->fit_;
    } else {
      t1->lastCompareFactor_ = 7;
      t2->lastCompareFactor_ = 7;
      // cout << "teamLexComp lcf 7 " << t1->id_ << " (>) " << t2->id_ << endl;
      return t1->id_ > t2->id_;
    }
  }
};

struct teamFitComplexLexCompare {
  bool operator()(team *t1, team *t2) const {
    if (!isEqual(t1->fit_, t2->fit_)) {
      t1->lastCompareFactor_ = 1;
      t2->lastCompareFactor_ = 1;
      // cout <<"teamFitComplexLexCompare 0:" << t1->fit_ << " " << t2->fit_ <<
      // endl;
      return t1->fit_ > t2->fit_;
    } else if (!isnan(t1->runTimeComplexityIns()) &&
               !isnan(t2->runTimeComplexityIns()) &&
               !isEqual(t1->runTimeComplexityIns(),
                        t2->runTimeComplexityIns())) {
      // cout <<"teamFitComplexLexCompare 1:" << t1->runTimeComplexityIns()  <<
      // " " << t2->runTimeComplexityIns() << endl;
      return t1->runTimeComplexityIns() < t2->runTimeComplexityIns();
    } else {
      // cout <<"teamFitComplexLexCompare 2:" << t1->id_ << " " << t2->id_ <<
      // endl;
      return t1->id_ > t2->id_;
    }
  }
};

struct teamInDegCompare {
  bool operator()(team *t1, team *t2) { return t1->inDeg() < t2->inDeg(); }
};

struct teamPair {
  set<team *, teamIdComp> teams;
  teamPair(team *t1, team *t2) {
    teams.insert(t1);
    teams.insert(t2);
  }
};

inline bool operator<(const teamPair &tmp1, const teamPair &tmp2) {
  if (tmp1.teams == tmp2.teams) return false;
  auto t1 = tmp1.teams.begin();
  auto t2 = tmp2.teams.begin();
  if ((*t1)->id_ != (*t2)->id_) return (*t1)->id_ < (*t2)->id_;
  advance(t1, 1);
  advance(t2, 1);
  return (*t1)->id_ < (*t2)->id_;
}

#endif
