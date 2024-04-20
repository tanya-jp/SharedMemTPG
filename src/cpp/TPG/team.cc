#include "team.h"

#include <algorithm>
#include <limits>

/******************************************************************************/
bool team::addProgram(program *lr, int i) {
  if (find(members_.begin(), members_.end(), lr) ==
      members_.end()) {  // could have duplicates since ordermaters
    if (i < 0)
      members_.push_back(lr);
    else {
      auto it = members_.begin();
      advance(it, i);
      members_.insert(it, lr);
    }
    if (lr->action() < 0) numAtomic_++;
    membersRun_.resize(members_.size());
    // membersRun_Tally[lr->id_] = 0;
    return true;
  }
  return false;
}

/******************************************************************************/
bool team::addProgramActive(program *lr) {
  if (find(active_.begin(), active_.end(), lr) == active_.end()) {
    active_.insert(lr);
    return true;
  }
  return false;
}

/******************************************************************************/
string team::checkpoint(bool fitnessBins, long id) const {
  long idToWrite = id > -1 ? id : id_;
  ostringstream oss;
  if (fitnessBins) {
    oss << "fBin:" << idToWrite;
    for (auto iter = fitnessBins_.begin(); iter != fitnessBins_.end(); iter++)
      oss << ":" << (*iter).first << "-" << (*iter).second;
    oss << endl;
  } else {
    oss << "team:" << idToWrite << ":" << gtime_ << ":" << _n_eval;
    //      for (int s = 0; s < _evalSeeds.size(); s++)
    //	     oss << ":" << _evalSeeds[s];
    // oss << ":" << task_code_;
    for (auto leiter = members_.begin(); leiter != members_.end(); leiter++)
      oss << ":" << (*leiter)->id_;
    for (auto leiter = active_.begin(); leiter != active_.end(); leiter++)
      oss << ":" << (*leiter)->id_;
    oss << endl;
    if (incomingPrograms_.size() > 0 && id == -1) {
      oss << "teamIncoming:" << idToWrite;
      for (auto it = incomingPrograms_.begin(); it != incomingPrograms_.end();
           it++)
        oss << ":" << *it;
      oss << endl;
    }
    oss << endl;
  }
  return oss.str();
}

/******************************************************************************/
void team::clearMemory(map<long, team *> &teamMap) {
  set<team *, teamIdComp> teams;
  set<memoryEigen *, memoryEigenIdComp> memories;
  getAllMemories(teamMap, teams, memories, false);
  for (auto it = memories.begin(); it != memories.end(); it++) {
    (*it)->ClearWorking();
    (*it)->ClearReadTime();   // needed?
    (*it)->ClearWriteTime();  // needed?
  }
}

/******************************************************************************/
void team::InitMemory(map<long, team *> &teamMap) {
  set<team *, teamIdComp> teams;
  set<memoryEigen *, memoryEigenIdComp> memories;
  getAllMemories(teamMap, teams, memories, false);
  for (auto it = memories.begin(); it != memories.end(); it++) {
    (*it)->CopyConstToWorking();
    (*it)->ClearReadTime();   // needed?
    (*it)->ClearWriteTime();  // needed?
  }
}

/******************************************************************************/
void team::clone(map<long, phyloRecord> &phyloGraph, team **tm) {
  phyloGraph[(*tm)->id_].ancestorIds.insert(id_);
  for (auto leiter = members_.begin(); leiter != members_.end(); leiter++) {
    (*tm)->addProgram(*leiter);
    (*leiter)->refInc();
  }
  for (auto leiter = active_.begin(); leiter != active_.end(); leiter++)
    (*tm)->addProgramActive(*leiter);
  (*tm)->fitnessBins(fitnessBins_);
  (*tm)->cloneId_ = id_;
  clones_++;
}

///****************************************************************************/
////this version creates all new teams anprograms with the same structure
// void team::cloneNew(team **tm, vector <team*> &teamClones, vector<program*>
// &programClones) {
//    //team* tm = new team(t, id);
//    (*tm)->addAncestorId(id_);
//    for(auto leiter = members_.begin(); leiter != members_.end(); leiter++){
//       (*tm)->addProgram(*leiter);
//       (*leiter)->refInc();
//    }
//    for(auto leiter = active_.begin(); leiter != active_.end(); leiter++)
//       (*tm)->addProgramActive(*leiter);
//    (*tm)->fitnessBins(fitnessBins_);
//    (*tm)->clone(true);
// }

///****************************************************************************/
// void team::deleteOutcome(point *pt) {
//    map < point *, double, pointLexicalLessThan > :: iterator ouiter;
//
//    if((ouiter = outcomes_.find(pt)) == outcomes_.end())
//       die(__FILE__, __FUNCTION__, __LINE__, "should not delete outcome that
//       is not set");
//    delete ouiter->first;
//    outcomes_.erase(ouiter);
// }

/******************************************************************************/
void team::features(set<long> &F) const {
  if (F.empty() == false)
    die(__FILE__, __FUNCTION__, __LINE__, "feature set not empty");

  for (auto leiter = members_.begin(); leiter != members_.end(); leiter++)
    (*leiter)->features(F);
}

/******************************************************************************/
double team::novelty(int type, int kNN) const {
  multiset<double>::iterator it;
  double nov = 0;
  int i = 0;
  if (type == 0) {
    for (it = distances_0_.begin(); it != distances_0_.end() && i <= kNN;
         ++it, i++)
      nov += *it;
    return nov / i;
  } else if (type == 1) {
    for (it = distances_1_.begin(); it != distances_1_.end() && i <= kNN;
         ++it, i++)
      nov += *it;
    return nov / i;
  } else if (type == 2) {
    for (it = distances_2_.begin(); it != distances_2_.end() && i <= kNN;
         ++it, i++)
      nov += *it;
    return nov / i;
  } else
    return -1;
}

/******************************************************************************/
double team::symbiontUtilityDistance(team *t) const {
  vector<int> symbiontIntersection;
  vector<int> symbiontUnion;
  vector<int>::iterator it;
  int symIntersection;
  int symUnion;
  vector<long> team1Ids;
  vector<long> team2Ids;
  set<program *, programIdComp> activeMembers;
  t->activeMembers(&activeMembers);
  /* if either team has no active members then return 0 */
  if (active_.size() < 1 || activeMembers.size() < 1) return 0.0;
  for (auto leiter = active_.begin(); leiter != active_.end(); leiter++)
    team1Ids.push_back((*leiter)->id_);
  for (auto leiter = activeMembers.begin(); leiter != activeMembers.end();
       leiter++)
    team2Ids.push_back((*leiter)->id_);
  sort(team1Ids.begin(), team1Ids.end());
  sort(team2Ids.begin(), team2Ids.end());
  set_intersection(team1Ids.begin(), team1Ids.end(), team2Ids.begin(),
                   team2Ids.end(), back_inserter(symbiontIntersection));
  symIntersection = symbiontIntersection.size();
  set_union(team1Ids.begin(), team1Ids.end(), team2Ids.begin(), team2Ids.end(),
            back_inserter(symbiontUnion));
  symUnion = symbiontUnion.size();
#ifdef MYDEBUG
  cout << "genoDiffa t1Size " << active_.size() << " t1Ids "
       << vecToStr(team1Ids);
  cout << " allMembersSize " << members_.size();
  cout << " t2Size " << t->asize() << " t2Ids ";
  cout << vecToStr(team2Ids) << " symIntersection "
       << vecToStr(symbiontIntersection) << " symIntersectionSize "
       << symIntersection;
  cout << " symUnion " << vecToStr(symbiontUnion) << " symUnionSize "
       << symUnion << " diff "
       << 1.0 - ((double)symIntersection / (double)symUnion) << endl;
#endif
  return 1.0 - ((double)symIntersection / (double)symUnion);
}

/******************************************************************************/
// this version compares with a vector of Ids *assumed sorted*
double team::symbiontUtilityDistance(vector<long> &compareWithThese) const {
  vector<int> symbiontIntersection;
  vector<int> symbiontUnion;
  vector<int>::iterator it;
  int symIntersection;
  int symUnion;
  vector<long> team1Ids;
  /* if either team has no active members then return 0 */
  if (active_.size() < 1 || compareWithThese.size() < 1) return 0.0;
  for (auto leiter = active_.begin(); leiter != active_.end(); leiter++)
    team1Ids.push_back((*leiter)->id_);
  sort(team1Ids.begin(), team1Ids.end());
  set_intersection(team1Ids.begin(), team1Ids.end(), compareWithThese.begin(),
                   compareWithThese.end(), back_inserter(symbiontIntersection));
  symIntersection = symbiontIntersection.size();
  set_union(team1Ids.begin(), team1Ids.end(), compareWithThese.begin(),
            compareWithThese.end(), back_inserter(symbiontUnion));
  symUnion = symbiontUnion.size();
#ifdef MYDEBUG
  cout << "genoDiffb t1Size " << active_.size() << " t1Ids "
       << vecToStr(team1Ids);
  cout << " allMembersSize " << members_.size();
  cout << " t2Size " << compareWithThese.size() << " t2Ids ";
  cout << vecToStr(compareWithThese) << " symIntersection "
       << vecToStr(symbiontIntersection) << " symIntersectionSize "
       << symIntersection;
  cout << " symUnion " << vecToStr(symbiontUnion) << " symUnionSize "
       << symUnion << " diff "
       << 1.0 - ((double)symIntersection / (double)symUnion) << endl;
#endif
  return 1.0 - ((double)symIntersection / (double)symUnion);
}

/******************************************************************************/
void team::updateComplexityRecord(map<long, team *> &teamMap, int rtcIndex) {
  (void)teamMap;
  // set <team *, teamIdComp> teams;
  // set <program *, programIdComp> programs;
  // set <memoryEigen *, memoryEigenIdComp> memories;
  // getAllNodes(teamMap, teams, programs, memories, false);//not just active
  // programs _numActiveTeams = teams.size(); _numActivePrograms =
  //programs.size(); _numEffectiveInstructions = 0; _numActiveFeatures = 0; for
  // (auto leiter = programs.begin(); leiter != programs.end(); leiter++){
  //    _numEffectiveInstructions += (*leiter)->esize();
  //    _numActiveFeatures += (*leiter)->numFeatures();
  // }
  runTimeComplexityIns_ =
      getMeanOutcome(_TRAIN_PHASE, 0, rtcIndex, false, true);
  runTimeComplexityTms_ =
      getMeanOutcome(_TRAIN_PHASE, 0, rtcIndex - 1, false, true);
}

/******************************************************************************/
void team::updateComplexityRecord(map<long, team *> &teamMap, int rtcIndex,
                                  int auxInt, long auxIntMatch, int phase) {
  (void)teamMap;
  // set <team *, teamIdComp> teams;
  // set <program *, programIdComp> programs;
  // set <memoryEigen *, memoryEigenIdComp> memories;
  // getAllNodes(teamMap, teams, programs, memories, false);//not just active
  // programs _numActiveTeams = teams.size(); _numActivePrograms =
  //programs.size(); _numEffectiveInstructions = 0; _numActiveFeatures = 0; for
  // (auto leiter = programs.begin(); leiter != programs.end(); leiter++){
  //    _numEffectiveInstructions += (*leiter)->esize();
  //    _numActiveFeatures += (*leiter)->numFeatures();
  // }
  runTimeComplexityIns_ =
      getMeanOutcome(phase, 0, rtcIndex, auxInt, auxIntMatch, false, true);
  runTimeComplexityTms_ =
      getMeanOutcome(phase, 0, rtcIndex - 1, auxInt, auxIntMatch, false, true);
}

/******************************************************************************/
void team::getAllMemories(map<long, team *> &teamMap,
                          set<team *, teamIdComp> &visitedTeams,
                          set<memoryEigen *, memoryEigenIdComp> &memories,
                          bool activePrograms) const {
  visitedTeams.insert(teamMap[id_]);

  for (auto leiter = members_.begin(); leiter != members_.end(); leiter++)
    if (!activePrograms ||
        (activePrograms && active_.find(*leiter) != active_.end())) {
      for (int memType = 0; memType < memoryEigen::NUM_MEMORY_TYPES;
           memType++)
        memories.insert((*leiter)->memGet(memType));
      if ((*leiter)->action() >= 0 &&
          find(visitedTeams.begin(), visitedTeams.end(),
               teamMap[(*leiter)->action()]) == visitedTeams.end())
        teamMap[(*leiter)->action()]->getAllMemories(teamMap, visitedTeams,
                                                     memories, activePrograms);
    }
}

/******************************************************************************/
// this version returns partial graph up to team tm
void team::getAllNodes(map<long, team *> &teamMap,
                       set<team *, teamIdComp> &visitedTeams, long stopId,
                       bool skipRoot) const {
  if (!skipRoot || !root_) visitedTeams.insert(teamMap[id_]);

  for (auto leiter = members_.begin(); leiter != members_.end(); leiter++)
    if ((*leiter)->action() >= 0 &&
        find(visitedTeams.begin(), visitedTeams.end(),
             teamMap[(*leiter)->action()]) == visitedTeams.end() &&
        (*leiter)->action() != stopId)
      teamMap[(*leiter)->action()]->getAllNodes(teamMap, visitedTeams, stopId,
                                                skipRoot);
}

/******************************************************************************/
void team::getAllNodes(map<long, team *> &teamMap,
                       set<team *, teamIdComp> &visitedTeams,
                       set<program *, programIdComp> &programs) const {
  visitedTeams.insert(teamMap[id_]);

  for (auto leiter = members_.begin(); leiter != members_.end(); leiter++) {
    programs.insert(*leiter);
    if ((*leiter)->action() >= 0 &&
        find(visitedTeams.begin(), visitedTeams.end(),
             teamMap[(*leiter)->action()]) == visitedTeams.end())
      teamMap[(*leiter)->action()]->getAllNodes(teamMap, visitedTeams,
                                                programs);
  }
}

/******************************************************************************/
void team::getAllNodes(map<long, team *> &teamMap,
                       set<team *, teamIdComp> &visitedTeams,
                       set<program *, programIdComp> &programs,
                       set<memoryEigen *, memoryEigenIdComp> &memories,
                       bool activePrograms) const {
  visitedTeams.insert(teamMap[id_]);

  for (auto leiter = members_.begin(); leiter != members_.end(); leiter++)
    if (!activePrograms ||
        (activePrograms && active_.find(*leiter) != active_.end())) {
      programs.insert(*leiter);
      for (int memType = 0; memType < memoryEigen::NUM_MEMORY_TYPES;
           memType++)
        memories.insert((*leiter)->memGet(memType));
      if ((*leiter)->action() >= 0 &&
          find(visitedTeams.begin(), visitedTeams.end(),
               teamMap[(*leiter)->action()]) == visitedTeams.end())
        teamMap[(*leiter)->action()]->getAllNodes(
            teamMap, visitedTeams, programs, memories, activePrograms);
    }
}

/******************************************************************************/
void team::updatePolicyRoot(map<long, team *> &teamMap,
                            set<team *, teamIdComp> &visitedTeams,
                            long &rootId) {
  visitedTeams.insert(teamMap[id_]);
  addPolicyRootId(rootId);  // add even if this is the root

  for (auto leiter = members_.begin(); leiter != members_.end(); leiter++)
    if ((*leiter)->action() >= 0 &&
        find(visitedTeams.begin(), visitedTeams.end(),
             teamMap[(*leiter)->action()]) == visitedTeams.end())
      teamMap[(*leiter)->action()]->updatePolicyRoot(teamMap, visitedTeams,
                                                     rootId);
}

/******************************************************************************/
// Fill F with every feature indexed by every program in this policy (tree).If
// we ever build massive policy tress, this should be changed to a more
// efficient traversal. For now just look at every node.
int team::policyFeatures(map<long, team *> &teamMap,
                         set<team *, teamIdComp> &visitedTeams, set<long> &F,
                         bool active) const {
  visitedTeams.insert(teamMap[id_]);

  set<long> featuresSingle;
  int numProgramsInPolicy = 0;
  for (auto leiter = members_.begin(); leiter != members_.end(); leiter++) {
    numProgramsInPolicy++;
    (*leiter)->features(featuresSingle);
    F.insert(featuresSingle.begin(), featuresSingle.end());
    if ((*leiter)->action() >= 0 &&
        find(visitedTeams.begin(), visitedTeams.end(),
             teamMap[(*leiter)->action()]) == visitedTeams.end())
      numProgramsInPolicy += teamMap[(*leiter)->action()]->policyFeatures(
          teamMap, visitedTeams, F, active);
  }
  return numProgramsInPolicy;
}

/******************************************************************************/
void team::policyInstructions(
    map<long, team *> &teamMap, set<team *, teamIdComp> &visitedTeams,
    vector<int> &programInstructionCounts,
    vector<int> &effectiveProgramInstructionCounts) const {
  visitedTeams.insert(teamMap[id_]);

  for (auto leiter = members_.begin(); leiter != members_.end(); leiter++) {
    programInstructionCounts.push_back((*leiter)->size());
    effectiveProgramInstructionCounts.push_back((*leiter)->esize());

    if ((*leiter)->action() >= 0 &&
        find(visitedTeams.begin(), visitedTeams.end(),
             teamMap[(*leiter)->action()]) == visitedTeams.end())
      teamMap[(*leiter)->action()]->policyInstructions(
          teamMap, visitedTeams, programInstructionCounts,
          effectiveProgramInstructionCounts);
  }
}

///****************************************************************************/
// void team::getBehaviourSequence(vector<int>&s, int phase) {
//    vector < behaviourType > singleEpisodeBehaviour;
//    map < point *, double >::reverse_iterator rit;
//    for (rit=outcomes_.rbegin(); rit!=outcomes_.rend(); rit++){
//       if ((rit->first)->phase() == phase){
//          (rit->first)->getBehaviour(singleEpisodeBehaviour);
//          if (s.size() + singleEpisodeBehaviour.size() <=
//          MAX_NCD_PROFILE_SIZE)
//             s.insert(s.end(),singleEpisodeBehaviour.begin(),singleEpisodeBehaviour.end());
//             //will cast to int (only discrete values used)
//          else
//             break;
//       }
//    }
// }

/******************************************************************************/
double team::getMeanOutcome(int phase, int task, int auxDouble, bool allPhase,
                            bool allTask) {
  vector<double> outcomes;

  // for(auto ouiter = outcomes_[task][phase].begin(); ouiter !=
  // outcomes_[task][phase].end(); ouiter++){
  //    if (((allPhase || (ouiter->second)->phase() == phase) &&
  //             (allTask || (ouiter->second)->task() == task)))
  //       outcomes.push_back((ouiter->second)->auxDouble(auxDouble));

  //}

  for (auto ouiter1 = outcomes_.begin(); ouiter1 != outcomes_.end();
       ouiter1++) {  // task
    if (ouiter1->first != task && !allTask) continue;
    for (auto ouiter2 = ouiter1->second.begin();
         ouiter2 != ouiter1->second.end(); ouiter2++) {  // phase
      if (ouiter2->first != phase && !allPhase) continue;
      for (auto ouiter3 = ouiter2->second.begin();
           ouiter3 != ouiter2->second.end(); ouiter3++)  // points
        outcomes.push_back(ouiter3->second->auxDouble(auxDouble));
    }
  }

  if (outcomes.size() == 0)
    die(__FILE__, __FUNCTION__, __LINE__,
        "trying to get meanOutcome with no outcomes");
  return accumulate(outcomes.begin(), outcomes.end(), 0.0) /
         outcomes
             .size(); 
}

/******************************************************************************/
double team::getMeanOutcome(int phase, int task, int auxDouble, int auxInt,
                            long auxIntMatch, bool allPhase, bool allTask) {
  vector<double> outcomes;

  // for(auto ouiter = outcomes_[task][phase].begin(); ouiter !=
  // outcomes_[task][phase].end(); ouiter++){
  //    if ((allPhase || (ouiter->second)->phase() == phase) && (allTask ||
  //    (ouiter->second)->task() == task) && (ouiter->second)->auxInt(auxInt) ==
  //    auxIntMatch)
  //       outcomes.push_back((ouiter->second)->auxDouble(auxDouble));
  // }

  for (auto ouiter1 = outcomes_.begin(); ouiter1 != outcomes_.end();
       ouiter1++) {  // task
    if (ouiter1->first != task && !allTask) continue;
    for (auto ouiter2 = ouiter1->second.begin();
         ouiter2 != ouiter1->second.end(); ouiter2++) {  // phase
      if (ouiter2->first != phase && !allPhase) continue;
      for (auto ouiter3 = ouiter2->second.begin();
           ouiter3 != ouiter2->second.end(); ouiter3++)  // points
        if (ouiter3->second->auxInt(auxInt) == auxIntMatch)
          outcomes.push_back(ouiter3->second->auxDouble(auxDouble));
    }
  }

  if (outcomes.size() == 0)
    die(__FILE__, __FUNCTION__, __LINE__,
        "trying to get meanOutcome with no outcomes");
  return accumulate(outcomes.begin(), outcomes.end(), 0.0) /
         outcomes
             .size();  //+(int)(topPortion*outcomes.size()),0.0)/(int)(topPortion*outcomes.size());
}

///****************************************************************************/
// double team::getMeanOutcome(int phase, int task, int auxDouble, double
// &rValue1, double &rValue2, double minVal) {
//    vector < double > outcomes;
//    rValue2 = 0;
//    for(auto ouiter = outcomes_[task][phase].begin(); ouiter !=
//    outcomes_[task][phase].end(); ouiter++){
//       if (((phase == -1 || (ouiter->first)->phase() == phase) &&
//                (task == -1 || (ouiter->first)->task() == task) &&
//                (ouiter->first)->auxDouble(auxDouble) > minVal)){
//          outcomes.push_back((ouiter->first)->auxDouble(auxDouble));
//          if((ouiter->first)->auxDouble(auxDouble) >= 50)
//             rValue2++;
//       }
//
//    }
//    if (outcomes.size() == 0)
//       return 0.0;//die(__FILE__, __FUNCTION__, __LINE__, "trying to get
//       meanOutcome with no outcomes");
//    rValue1 = accumulate(outcomes.begin(),outcomes.end(), 0.0) /
//    outcomes.size();
//    //+(int)(topPortion*outcomes.size()),0.0)/(int)(topPortion*outcomes.size());
//    return rValue2 + rValue1/3000;
// }

/******************************************************************************/
bool team::hasPointDesc(string d, int task, int phase) {
  for (auto ouiter = outcomes_[task][phase].begin();
       ouiter != outcomes_[task][phase].end(); ouiter++)
    if (d.compare((ouiter->second)->desc()) == 0) return true;
  return false;
}

/******************************************************************************/
double team::getPointDescScore(string d, int task, int phase, int auxDouble) {
  for (auto ouiter = outcomes_[task][phase].begin();
       ouiter != outcomes_[task][phase].end(); ouiter++)
    if (d.compare((ouiter->second)->desc()) == 0)
      return (ouiter->second)->auxDouble(auxDouble);
  die(__FILE__, __FUNCTION__, __LINE__,
      "trying to get score from point that doesn't exist");
  return 0;
}

///****************************************************************************/
// bool team::getOutcome(point *pt, double *out) {
//    map < point *, double, pointLexicalLessThan > :: iterator ouiter;
//
//    if((ouiter = outcomes_.find(pt)) == outcomes_.end())
//       return false;
//
//    *out = ouiter->second;
//
//    return true;
// }

///****************************************************************************/
// double team::getRMSOutcome(int phase, int auxDouble) {
//    double rms = 0;
//    for(auto ouiter = outcomes_.begin(); ouiter != outcomes_.end(); ouiter++)
//       if ((ouiter->first)->phase() == phase)
//          rms += (ouiter->first)->auxDouble(auxDouble);
//    return isfinite(rms) ? -(sqrt(rms / outcomes_.size())) :
//    -numeric_limits<double>::max();
// }

///****************************************************************************/
// bool team::hasOutcome(point *pt) {
//    map < point *, double, pointLexicalLessThan > :: iterator ouiter;
//
//    if((ouiter = outcomes_.find(pt)) == outcomes_.end())
//       return false;
//
//    return true;
// }

///****************************************************************************/
///* Calculate normalized compression distance w.r.t another team. */
// double team::ncdBehaviouralDistance(team * t, int phase) {
//    ostringstream oss;
//    vector <int> theirBehaviourSequence;
//    t->getBehaviourSequence(theirBehaviourSequence, phase);
//    vector <int> myBehaviourSequence;
//    getBehaviourSequence(myBehaviourSequence, phase);
//    if (myBehaviourSequence.size() == 0 || theirBehaviourSequence.size() == 0)
//       return -1;
//    return
//    normalizedCompressionDistance(myBehaviourSequence,theirBehaviourSequence);
// }

/******************************************************************************/
int team::numOutcomes(int phase, int task) {
  int numOut = 0;
  // for(auto ouiter = outcomes_.begin(); ouiter != outcomes_.end(); ouiter++)
  //    if (ouiter->first->phase() == phase && (task == -1 ||
  //    ouiter->first->task() == task))
  //       numOut++;
  if (task < 0)
    for (auto ouiter1 = outcomes_.begin(); ouiter1 != outcomes_.end();
         ouiter1++)
      numOut += ouiter1->second[phase].size();
  else
    numOut = outcomes_[task][phase].size();
  return numOut;
}

///****************************************************************************/
// void team::outcomes(int i, int phase, vector < double > &outcomes) {
//    map < point *, double, pointLexicalLessThan > :: iterator ouiter;
//    for(ouiter = outcomes_.begin(); ouiter != outcomes_.end(); ouiter++)
//       if ((ouiter->first)->phase() == phase)
//          outcomes.push_back((ouiter->first)->auxDouble(i));
// }

///****************************************************************************/
// void team::outcomes(map < point*, double, pointLexicalLessThan >
// &outcomes,int phase) {
//    for(auto ouiter = outcomes_.begin(); ouiter != outcomes_.end(); ouiter++)
//       if ((ouiter->first)->phase() == phase)
//          outcomes.insert(*ouiter);
// }

/******************************************************************************/
// must run markEffectiveCode first
void team::prunePrograms(deque<program *> &programsWithNoRefs) {
  // if (active_.size() == 0)
  //    die(__FILE__, __FUNCTION__, __LINE__, "should not prune when no programs
  //    are marked active");
  if (active_.size() > 0) {
    for (auto leiter = members_.begin(); leiter != members_.end();) {
      if (((*leiter)->action() < 0 &&
           numAtomic_ < 2) ||  // don't remove the only atomic
          active_.find(*leiter) !=
              active_.end())  // don't remove active programs
        leiter++;
      else {
        (*leiter)->refDec();
        if ((*leiter)->refs() == 0 &&
            find(programsWithNoRefs.begin(), programsWithNoRefs.end(),
                 *leiter) == programsWithNoRefs.end())
          programsWithNoRefs.push_back(*leiter);
        if ((*leiter)->action() < 0) numAtomic_--;
        // membersRun_Tally.erase((*leiter)->id_);
        members_.erase(leiter++);
      }
    }
    membersRun_.resize(members_.size());
  }
}

///****************************************************************************/
// void team::cleanup(set <program *> &learnersWithNoRefs) {
//    //decrement program refs
//    for(auto leiter = members_.begin(); leiter != members_.end(); leiter++){
//       (*leiter)->refDec();
//       if ((*leiter)->refs() == 0)
//          learnersWithNoRefs.insert(*leiter);
//    }
//    for (auto ouiter1 = outcomes_.begin(); ouiter1 != outcomes_.end();
//    ouiter1++)
//       for (auto ouiter2 = ouiter1->second.begin(); ouiter2 !=
//       ouiter1->second.end(); ouiter2++)
//          for (auto ouiter3 = ouiter2->second.begin(); ouiter3 !=
//          ouiter2->second.end();){
//             delete ouiter3->first;
//             ouiter2->second.erase(ouiter3++);
//          }
// }

/******************************************************************************/

void team::cleanup(map<long, team *> &teamMap, deque<program *> &p) {
  // decrement program refs
  for (auto leiter = members_.begin(); leiter != members_.end(); leiter++) {
    (*leiter)->refDec();
    if ((*leiter)->refs() == 0)  // && find(p.begin(), p.end(), *leiter) ==
                                 // p.end())//could prob skip the find/check
      p.push_back(*leiter);
  }
  for (auto ouiter1 = outcomes_.begin(); ouiter1 != outcomes_.end(); ouiter1++)
    for (auto ouiter2 = ouiter1->second.begin();
         ouiter2 != ouiter1->second.end(); ouiter2++)
      for (auto ouiter3 = ouiter2->second.begin();
           ouiter3 != ouiter2->second.end();) {
        delete ouiter3->second;
        ouiter2->second.erase(ouiter3++);
      }
  if (teamMap.find(cloneId_) != teamMap.end()) teamMap[cloneId_]->clones_--;
}

/******************************************************************************/
bool team::removeProgram(program *lr) {
  list<program *>::iterator leiter;
  set<program *, programIdComp>::iterator aiter;

  if ((leiter = find(members_.begin(), members_.end(), lr)) == members_.end())
    return false;  // die(__FILE__, __FUNCTION__, __LINE__, "should not remove
                   // program that is not there");

  members_.erase(leiter);
  membersRun_.resize(members_.size());
  // membersRun_Tally.erase((*leiter)->id_);

  if ((aiter = active_.find(lr)) != active_.end()) active_.erase(aiter);

  if (lr->action() < 0) numAtomic_--;
  return true;
}

/******************************************************************************/
void team::resetOutcomes(int phase) {
  // map < point *, double, pointLexicalLessThan > :: iterator ouiter;
  // for (ouiter = outcomes_.begin(); ouiter != outcomes_.end();)
  //{
  //    if ((ouiter->first)->phase() == phase || phase < 0){
  //       delete ouiter->first;
  //       outcomes_.erase(ouiter++);
  //    }
  //    else
  //       ouiter++;
  // }
  //

  for (auto ouiter1 = outcomes_.begin(); ouiter1 != outcomes_.end();
       ouiter1++)  // task
    for (auto ouiter2 = ouiter1->second.begin();
         ouiter2 != ouiter1->second.end(); ouiter2++)  // phase
      if (ouiter2->first == phase || phase == -1) {
        for (auto ouiter3 = ouiter2->second.begin();
             ouiter3 != ouiter2->second.end();) {
          delete ouiter3->second;
          ouiter2->second.erase(ouiter3++);
        }
      }
  quickSums_.clear();
  quickMeans_.clear();
  runTimeComplexityIns_ = 0;
  runTimeComplexityTms_ = 0;
}

/******************************************************************************/
void team::swapOutcomePhase(int phaseFrom, int phaseTo, int auxInt,
                            long auxIntMatch) {
  resetOutcomes(phaseTo);
  resetOutcomes(_TEST_PHASE);
  for (auto ouiter1 = outcomes_.begin(); ouiter1 != outcomes_.end();
       ouiter1++) {  // task
    for (auto ouiter2 = outcomes_[ouiter1->first][phaseFrom].begin();
         ouiter2 != outcomes_[ouiter1->first][phaseFrom].end();
         ouiter2++)  // points
      if ((ouiter2->second)->auxInt(auxInt) == auxIntMatch) {
        ouiter2->second->phase(phaseTo);
        setOutcome(ouiter2->second);
      }
    outcomes_[ouiter1->first][phaseFrom].clear();
  }
}

/******************************************************************************/
void team::setOutcome(point *pt) {
  // if((outcomes_[pt->task()][pt->phase()].insert(map <
  // pt->auxInt(POINT_AUX_INT_ENVSEED), point
  // *>::value_type(pt->auxInt(POINT_AUX_INT_ENVSEED),pt))).second == false)
  // die(__FILE__, __FUNCTION__, __LINE__, "could not set outcome, duplicate
  // point?");
  outcomes_[pt->task()][pt->phase()][pt->auxInt(POINT_AUX_INT_ENVSEED)] = pt;
  if (hasQuickSum(pt->task(), pt->key(), pt->phase()))
    quickSums_[pt->task()][pt->key()][pt->phase()] += pt->auxDouble(pt->key());
  else
    quickSums_[pt->task()][pt->key()][pt->phase()] = pt->auxDouble(pt->key());
  quickMeans_[pt->task()][pt->key()][pt->phase()] =
      quickSums_[pt->task()][pt->key()][pt->phase()] /
      numOutcomes(pt->phase(), pt->task());
}

/******************************************************************************/
void team::updateActiveMembersFromIds(vector<long> &activeMemberIds) {
  sort(activeMemberIds.begin(), activeMemberIds.end());  // for binary search
  for (auto leiter = members_.begin(); leiter != members_.end(); leiter++)
    if (binary_search(activeMemberIds.begin(), activeMemberIds.end(),
                      (*leiter)->id_))
      active_.insert(*leiter);
}

/******************************************************************************/
program *team::getAction(state *s, map<long, team *> &teamMap,
                         bool updateActive,
                         set<team *, teamIdComp> &visitedTeams,
                         long &decisionInstructions, int timeStep,
                         vector<team *> &teamPath, mt19937 &rng) {
  //_depthSum += visitedTeams.size(); _visitedCount++;
  visitedTeams.insert(teamMap[id_]);
  teamPath.push_back(teamMap[id_]);

  int l = 0;
  for (auto leiter = members_.begin(); leiter != members_.end(); leiter++) {
    (*leiter)->bidVal((*leiter)->run(s, timeStep, visitedTeams.size(), rng));
    membersRun_[l++] = *leiter;
    decisionInstructions += (*leiter)->esize();
  }

  sort(membersRun_.begin(), membersRun_.end(), ProgramBidLexicalCompare());
  long teamIdToFollow = 0;
  for (size_t i = 0; i < membersRun_.size(); i++) {
    if (membersRun_[i]->action() < 0) {
      if (root_ && updateActive) active_.insert(membersRun_[i]);
      // membersRun_Tally[membersRun_[i]->id_]++;
      return membersRun_[i];  //->action();
    } else if (find(visitedTeams.begin(), visitedTeams.end(),
                    teamMap[membersRun_[i]->action()]) == visitedTeams.end()) {                 
      teamIdToFollow = membersRun_[i]->action();
      if (root_ && updateActive) active_.insert(membersRun_[i]);
      // membersRun_Tally[membersRun_[i]->id_]++;
      break;
    }
  }
  return teamMap[teamIdToFollow]->getAction(s, teamMap, updateActive,
                                            visitedTeams, decisionInstructions,
                                            timeStep, teamPath, rng);
}

/******************************************************************************/
program *team::getAction(
    state *s, map<long, team *> &teamMap, bool updateActive,
    set<team *, teamIdComp> &visitedTeams, long &decisionInstructions,
    int timeStep, vector<program *> &allPrograms,
    vector<program *> &winningPrograms, vector<set<long> > &decisionFeatures,
    vector<set<memoryEigen *, memoryEigenIdComp> > &decisionMemories,
    vector<team *> &teamPath, mt19937 &rng) {
  //_depthSum += visitedTeams.size(); _visitedCount++;
  visitedTeams.insert(teamMap[id_]);
  teamPath.push_back(teamMap[id_]);

  set<long> features;
  set<long> featuresSingle;
  set<memoryEigen *, memoryEigenIdComp> memories;
  set<memoryEigen *, memoryEigenIdComp> memoriesSingle;

  int l = 0;
  for (auto leiter = members_.begin(); leiter != members_.end(); leiter++) {
    (*leiter)->bidVal((*leiter)->run(s, timeStep, visitedTeams.size(), rng));
    allPrograms.push_back(*leiter);
    membersRun_[l++] = *leiter;
    decisionInstructions += (*leiter)->esize();

    (*leiter)->features(featuresSingle);
    features.insert(featuresSingle.begin(), featuresSingle.end());
    for (int memType = 0; memType < memoryEigen::NUM_MEMORY_TYPES; memType++)
      memories.insert((*leiter)->memGet(memType));
  }
  decisionFeatures.push_back(features);
  decisionMemories.push_back(memories);

  sort(membersRun_.begin(), membersRun_.end(), ProgramBidLexicalCompare());
  long teamIdToFollow = 0;
  for (size_t i = 0; i < membersRun_.size(); i++) {
    if (membersRun_[i]->action() < 0) {  // atomic
      if (root_ && updateActive) active_.insert(membersRun_[i]);
      winningPrograms.push_back(membersRun_[i]);
      membersRun_[i]->featuresMem(featuresSingle);
      decisionFeatures.push_back(featuresSingle);
      // membersRun_Tally[membersRun_[i]->id_]++;
      return membersRun_[i];
    } else if (find(visitedTeams.begin(), visitedTeams.end(),
                    teamMap[membersRun_[i]->action()]) == visitedTeams.end()) {
      teamIdToFollow = membersRun_[i]->action();
      if (root_ && updateActive) active_.insert(membersRun_[i]);
      winningPrograms.push_back(membersRun_[i]);
      membersRun_[i]->featuresMem(featuresSingle);
      decisionFeatures.push_back(featuresSingle);
      // membersRun_Tally[membersRun_[i]->id_]++;
      break;
    }
  }
  return teamMap[teamIdToFollow]->getAction(
      s, teamMap, updateActive, visitedTeams, decisionInstructions, timeStep,
      allPrograms, winningPrograms, decisionFeatures, decisionMemories,
      teamPath, rng);
}
