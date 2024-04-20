#include "TPG.h"

/******************************************************************************/
TPG::TPG() {
  instruction::setupOps();
  real_distribution_ = uniform_real_distribution<>(0.0, 1.0);
  _Memids.resize(memoryEigen::NUM_MEMORY_TYPES);
  _Memory.resize(memoryEigen::NUM_MEMORY_TYPES);
  for (size_t i = 0; i < _NUM_PHASE; i++) _numEliteTeamsCurrent.push_back(0);
  _numStoredOutcomesPerHost.resize(_NUM_PHASE);
  _ops.resize(instruction::NUM_OP);
  fill(_ops.begin(), _ops.end(), false);
  _rngs.resize(NUM_RNG);
  _seeds.resize(NUM_RNG);
}

/******************************************************************************/
TPG::~TPG() {}

/******************************************************************************/
void TPG::addProgram(program *p) {
  _L[p->id_] = p;
  _Lids.push_back(p->id_);
}

/******************************************************************************/
void TPG::removeProgram(program *p, bool updateLids) {
  if (updateLids) {
    auto it = find(_Lids.begin(), _Lids.end(), p->id_);
    if (it == _Lids.end())
      die(__FILE__, __FUNCTION__, __LINE__, "failed to remove program");
    swap(_Lids[distance(_Lids.begin(), it)], _Lids.back());
    _Lids.pop_back();
  }
  _L.erase(p->id_);
}

/******************************************************************************/
void TPG::addTeam(team *tm) {
  _M.insert(tm);
  _teamMap[tm->id_] = tm;
  _Mids.push_back(tm->id_);
}

/******************************************************************************/
void TPG::removeTeam(team *tm, bool updateMids) {
  if (updateMids) {
    auto it = find(_Mids.begin(), _Mids.end(), tm->id_);
    if (it == _Mids.end())
      die(__FILE__, __FUNCTION__, __LINE__, "failed to remove team");
    swap(_Mids[it - _Mids.begin()], _Mids.back());
    _Mids.pop_back();
  }
  _teamMap.erase(tm->id_);
  _M.erase(tm);
}

/******************************************************************************/
void TPG::addMemory(memoryEigen *m) {
  _Memory[m->type()][m->id_] = m;
  _Memids[m->type()].push_back(m->id_);
}

/******************************************************************************/
void TPG::removeMemory(memoryEigen *m) {
  auto it = find(_Memids[m->type()].begin(), _Memids[m->type()].end(), m->id_);
  if (it == _Memids[m->type()].end())
    die(__FILE__, __FUNCTION__, __LINE__, "failed to remove memoryEigen");
  swap(_Memids[m->type()][it - _Memids[m->type()].begin()],
       _Memids[m->type()].back());
  _Memids[m->type()].pop_back();
  _Memory[m->type()].erase(m->id_);
}

/******************************************************************************/
void TPG::cloneProgramLinearM(linearM *p1, linearM **c1) {
  // clone programs
  //*c1 = new linearM(_tCurrent, *(dynamic_cast<linearM*>(p1)),
  //_memoryIndices, _memoryRows, _memoryCols, _programCount++);
  *c1 = new linearM(GetState("t_current"), *(dynamic_cast<linearM *>(p1)),
                    params_, state_["program_count"]++);
  if ((*c1)->action() >= 0)
    _teamMap[(*c1)->action()]->addIncomingProgram((*c1)->id_);
  for (size_t memType = 0; memType < memoryEigen::NUM_MEMORY_TYPES; memType++) {
    (*c1)->memSet(memType, p1->memGet(memType));  // copy memoryEigen
                                                  // reference
    (*c1)->memGet(memType)->refInc();
  }
}

/******************************************************************************/
team *TPG::getTeamByID(long id) {
  for (auto teiter = _M.begin(); teiter != _M.end(); teiter++)
    if ((*teiter)->id_ == id) return *teiter;
  return *(_M.begin());
}

/******************************************************************************/
bool TPG::haveEliteTeam(string taskset, int fitMode, int phase) {
  return _eliteTeamPS[taskset][fitMode].find(phase) !=
         _eliteTeamPS[taskset][fitMode].end();
}

/******************************************************************************/
void TPG::seed(size_t i, int s) {
  _seeds[i] = s;
  _rngs[i].seed(s);
}

/******************************************************************************/
void TPG::clearMemory() {
  for (size_t memType = 0; memType < memoryEigen::NUM_MEMORY_TYPES; memType++)
    for (auto meiter = _Memory[memType].begin();
         meiter != _Memory[memType].end(); meiter++) {
      meiter->second->ClearWorking();
      meiter->second->ClearReadTime();
      meiter->second->ClearWriteTime();
    }
}

/******************************************************************************/
program *TPG::getAction(team *tm, state *s, bool updateActive,
                        set<team *, teamIdComp> &visitedTeams,
                        long &decisionInstructions, int timeStep,
                        vector<team *> &teamPath, mt19937 &rng) {
  visitedTeams.clear();
  decisionInstructions = 0;
  teamPath.clear();
  return tm->getAction(s, _teamMap, updateActive, visitedTeams,
                       decisionInstructions, timeStep, teamPath, rng);
}

/******************************************************************************/
program *TPG::getAction(
    team *tm, state *s, bool updateActive,
    set<team *, teamIdComp> &visitedTeams, long &decisionInstructions,
    int timeStep, vector<program *> &allPrograms,
    vector<program *> &winningPrograms, vector<set<long>> &decisionFeatures,
    vector<set<memoryEigen *, memoryEigenIdComp>> &decisionMemories,
    vector<team *> &teamPath, mt19937 &rng) {
  allPrograms.clear();
  winningPrograms.clear();
  decisionInstructions = 0;
  decisionFeatures.clear();
  decisionMemories.clear();
  // visitedTeams.clear();
  teamPath.clear();
  return tm->getAction(s, _teamMap, updateActive, visitedTeams,
                       decisionInstructions, timeStep, allPrograms,
                       winningPrograms, decisionFeatures, decisionMemories,
                       teamPath, rng);
}

/******************************************************************************/
void TPG::getAllNodes(team *tm, set<team *, teamIdComp> &teams,
                      set<program *, programIdComp> &programs) {
  teams.clear();
  programs.clear();
  tm->getAllNodes(_teamMap, teams, programs);
}

/******************************************************************************/
void TPG::getAllNodes(team *tm, set<team *, teamIdComp> &teams,
                      set<program *, programIdComp> &programs,
                      set<memoryEigen *, memoryEigenIdComp> &memories) {
  teams.clear();
  programs.clear();
  memories.clear();
  tm->getAllNodes(_teamMap, teams, programs, memories, false);
}

/******************************************************************************/
void TPG::getTeams(vector<team *> &t, bool roots) const {
  if (roots)
    t.assign(_Mroot.begin(), _Mroot.end());
  else
    t.assign(_M.begin(), _M.end());
}

/******************************************************************************/
void TPG::getTeams(map<long, team *> &t, bool roots) const {
  t.clear();
  if (roots)
    for (auto teiter = _Mroot.begin(); teiter != _Mroot.end(); teiter++)
      t[(*teiter)->id_] = *teiter;
  else
    for (auto teiter = _M.begin(); teiter != _M.end(); teiter++)
      t[(*teiter)->id_] = *teiter;
}

/******************************************************************************/
map<long, team *> TPG::GetTeams(bool roots) const {
  map<long, team *> t;
  if (roots)
    for (auto teiter = _Mroot.begin(); teiter != _Mroot.end(); teiter++)
      t[(*teiter)->id_] = *teiter;
  else
    for (auto teiter = _M.begin(); teiter != _M.end(); teiter++)
      t[(*teiter)->id_] = *teiter;
  return t;
}

/******************************************************************************/
bool TPG::isElitePS(team *tm, int phase) {
  for (auto itr1 = _eliteTeamPS.begin(); itr1 != _eliteTeamPS.end();
       itr1++) {  // taskset
    for (auto itr2 = itr1->second.begin(); itr2 != itr1->second.end();
         itr2++) {  // fitmode
                    // for (auto itr3 = itr2->second.begin(); itr3 !=
                    // itr2->second.end(); itr3++)//phase
      if (itr2->second.find(phase) != itr2->second.end() &&
          itr2->second[phase]->id_ == tm->id_)
        return true;
    }
  }
  return false;
}

/******************************************************************************/
void TPG::markEffectiveCode(team *tm) {
  set<team *, teamIdComp> teams;
  set<program *, programIdComp> programs;
  set<memoryEigen *, memoryEigenIdComp> memories;
  tm->getAllNodes(_teamMap, teams, programs, memories, false);

  for (auto meiter = memories.begin(); meiter != memories.end(); meiter++) {
    (*meiter)->ClearActive();
    (*meiter)->refsPolicy(0);
  }

  for (auto leiter = programs.begin(); leiter != programs.end(); leiter++) {
    (*leiter)->skipIntrons(GetParam<int>("skip_introns"));
    (*leiter)->markIntrons(GetParam<int>("continuous_output"));
    for (size_t memType = 0; memType < memoryEigen::NUM_MEMORY_TYPES; memType++)
      ((*leiter)->memGet(memType))->refsPolicyInc();
  }

  // update active programs
  list<program *> rootMembers;
  tm->members(&rootMembers);
  for (auto leiter = rootMembers.begin(); leiter != rootMembers.end();
       leiter++) {
    bool active = false;
    for (size_t memType = 0; memType < memoryEigen::NUM_MEMORY_TYPES; memType++)
      if (((*leiter)->memGet(memType))->refsPolicy() > 1) active = true;
    if ((*leiter)->esize() > 0 && active) tm->setActive(*leiter);
  }
}

/******************************************************************************/
void TPG::printOss() {
  cout << oss.str();
  oss.str("");
}

/******************************************************************************/
void TPG::printOss(ostringstream &o) {
  oss << o.str();
  o.str("");
}

/******************************************************************************/
void TPG::ReadParameters(string file_name,
                         std::unordered_map<string, std::any> &params) {
  std::ifstream infile(file_name);
  string oneline;
  vector<string> outcome_fields;
  while (std::getline(infile, oneline)) {
    if (oneline.find('#') != std::string::npos || oneline.size() == 0)
      continue;  // skip comments and empty lines
    splitString(oneline, ' ', outcome_fields);

    if (outcome_fields[0] == "SCALAR_SUM_OP")
      _ops[instruction::SCALAR_SUM_OP_] = true;
    if (outcome_fields[0] == "SCALAR_DIFF_OP")
      _ops[instruction::SCALAR_DIFF_OP_] = true;
    if (outcome_fields[0] == "SCALAR_PRODUCT_OP")
      _ops[instruction::SCALAR_PRODUCT_OP_] = true;
    if (outcome_fields[0] == "SCALAR_DIVISION_OP")
      _ops[instruction::SCALAR_DIVISION_OP_] = true;
    if (outcome_fields[0] == "SCALAR_ABS_OP")
      _ops[instruction::SCALAR_ABS_OP_] = true;
    if (outcome_fields[0] == "SCALAR_RECIPROCAL_OP")
      _ops[instruction::SCALAR_RECIPROCAL_OP_] = true;
    if (outcome_fields[0] == "SCALAR_SIN_OP")
      _ops[instruction::SCALAR_SIN_OP_] = true;
    if (outcome_fields[0] == "SCALAR_COS_OP")
      _ops[instruction::SCALAR_COS_OP_] = true;
    if (outcome_fields[0] == "SCALAR_TAN_OP")
      _ops[instruction::SCALAR_TAN_OP_] = true;
    if (outcome_fields[0] == "SCALAR_ARCSIN_OP")
      _ops[instruction::SCALAR_ARCSIN_OP_] = true;
    if (outcome_fields[0] == "SCALAR_ARCCOS_OP")
      _ops[instruction::SCALAR_ARCCOS_OP_] = true;
    if (outcome_fields[0] == "SCALAR_ARCTAN_OP")
      _ops[instruction::SCALAR_ARCTAN_OP_] = true;
    if (outcome_fields[0] == "SCALAR_EXP_OP")
      _ops[instruction::SCALAR_EXP_OP_] = true;
    if (outcome_fields[0] == "SCALAR_LOG_OP")
      _ops[instruction::SCALAR_LOG_OP_] = true;
    if (outcome_fields[0] == "SCALAR_HEAVYSIDE_OP")
      _ops[instruction::SCALAR_HEAVYSIDE_OP_] = true;
    if (outcome_fields[0] == "VECTOR_HEAVYSIDE_OP")
      _ops[instruction::VECTOR_HEAVYSIDE_OP_] = true;
    if (outcome_fields[0] == "MATRIX_HEAVYSIDE_OP")
      _ops[instruction::MATRIX_HEAVYSIDE_OP_] = true;
    if (outcome_fields[0] == "SCALAR_VECTOR_PRODUCT_OP")
      _ops[instruction::SCALAR_VECTOR_PRODUCT_OP_] = true;
    if (outcome_fields[0] == "SCALAR_BROADCAST_OP")
      _ops[instruction::SCALAR_BROADCAST_OP_] = true;
    if (outcome_fields[0] == "VECTOR_RECIPROCAL_OP")
      _ops[instruction::VECTOR_RECIPROCAL_OP_] = true;
    if (outcome_fields[0] == "VECTOR_NORM_OP")
      _ops[instruction::VECTOR_NORM_OP_] = true;
    if (outcome_fields[0] == "VECTOR_ABS_OP")
      _ops[instruction::VECTOR_ABS_OP_] = true;
    if (outcome_fields[0] == "VECTOR_SUM_OP")
      _ops[instruction::VECTOR_SUM_OP_] = true;
    if (outcome_fields[0] == "VECTOR_DIFF_OP")
      _ops[instruction::VECTOR_DIFF_OP_] = true;
    if (outcome_fields[0] == "VECTOR_PRODUCT_OP")
      _ops[instruction::VECTOR_PRODUCT_OP_] = true;
    if (outcome_fields[0] == "VECTOR_DIVISION_OP")
      _ops[instruction::VECTOR_DIVISION_OP_] = true;
    if (outcome_fields[0] == "VECTOR_INNER_PRODUCT_OP")
      _ops[instruction::VECTOR_INNER_PRODUCT_OP_] = true;
    if (outcome_fields[0] == "VECTOR_OUTER_PRODUCT_OP")
      _ops[instruction::VECTOR_OUTER_PRODUCT_OP_] = true;
    if (outcome_fields[0] == "SCALAR_MATRIX_PRODUCT_OP")
      _ops[instruction::SCALAR_MATRIX_PRODUCT_OP_] = true;
    if (outcome_fields[0] == "MATRIX_RECIPROCAL_OP")
      _ops[instruction::MATRIX_RECIPROCAL_OP_] = true;
    if (outcome_fields[0] == "MATRIX_VECTOR_PRODUCT_OP")
      _ops[instruction::MATRIX_VECTOR_PRODUCT_OP_] = true;
    if (outcome_fields[0] == "VECTOR_COLUMN_BROADCAST_OP")
      _ops[instruction::VECTOR_COLUMN_BROADCAST_OP_] = true;
    if (outcome_fields[0] == "VECTOR_ROW_BROADCAST_OP")
      _ops[instruction::VECTOR_ROW_BROADCAST_OP_] = true;
    if (outcome_fields[0] == "MATRIX_NORM_OP")
      _ops[instruction::MATRIX_NORM_OP_] = true;
    if (outcome_fields[0] == "MATRIX_COLUMN_NORM_OP")
      _ops[instruction::MATRIX_COLUMN_NORM_OP_] = true;
    if (outcome_fields[0] == "MATRIX_ROW_NORM_OP")
      _ops[instruction::MATRIX_ROW_NORM_OP_] = true;
    if (outcome_fields[0] == "MATRIX_TRANSPOSE_OP")
      _ops[instruction::MATRIX_TRANSPOSE_OP_] = true;
    if (outcome_fields[0] == "MATRIX_ABS_OP")
      _ops[instruction::MATRIX_ABS_OP_] = true;
    if (outcome_fields[0] == "MATRIX_SUM_OP")
      _ops[instruction::MATRIX_SUM_OP_] = true;
    if (outcome_fields[0] == "MATRIX_DIFF_OP")
      _ops[instruction::MATRIX_DIFF_OP_] = true;
    if (outcome_fields[0] == "MATRIX_PRODUCT_OP")
      _ops[instruction::MATRIX_PRODUCT_OP_] = true;
    if (outcome_fields[0] == "MATRIX_DIVISION_OP")
      _ops[instruction::MATRIX_DIVISION_OP_] = true;
    if (outcome_fields[0] == "MATRIX_MATRIX_PRODUCT_OP")
      _ops[instruction::MATRIX_MATRIX_PRODUCT_OP_] = true;
    if (outcome_fields[0] == "SCALAR_MIN_OP")
      _ops[instruction::SCALAR_MIN_OP_] = true;
    if (outcome_fields[0] == "VECTOR_MIN_OP")
      _ops[instruction::VECTOR_MIN_OP_] = true;
    if (outcome_fields[0] == "MATRIX_MIN_OP")
      _ops[instruction::MATRIX_MIN_OP_] = true;
    if (outcome_fields[0] == "SCALAR_MAX_OP")
      _ops[instruction::SCALAR_MAX_OP_] = true;
    if (outcome_fields[0] == "VECTOR_MAX_OP")
      _ops[instruction::VECTOR_MAX_OP_] = true;
    if (outcome_fields[0] == "MATRIX_MAX_OP")
      _ops[instruction::MATRIX_MAX_OP_] = true;
    if (outcome_fields[0] == "VECTOR_MEAN_OP")
      _ops[instruction::VECTOR_MEAN_OP_] = true;
    if (outcome_fields[0] == "MATRIX_MEAN_OP")
      _ops[instruction::MATRIX_MEAN_OP_] = true;
    if (outcome_fields[0] == "MATRIX_ROW_ST_DEV_OP")
      _ops[instruction::MATRIX_ROW_ST_DEV_OP_] = true;
    if (outcome_fields[0] == "VECTOR_ST_DEV_OP")
      _ops[instruction::VECTOR_ST_DEV_OP_] = true;
    if (outcome_fields[0] == "MATRIX_ST_DEV_OP")
      _ops[instruction::MATRIX_ST_DEV_OP_] = true;
    if (outcome_fields[0] == "SCALAR_CONST_SET_OP")
      _ops[instruction::SCALAR_CONST_SET_OP_] = true;
    if (outcome_fields[0] == "VECTOR_CONST_SET_OP")
      _ops[instruction::VECTOR_CONST_SET_OP_] = true;
    if (outcome_fields[0] == "MATRIX_CONST_SET_OP")
      _ops[instruction::MATRIX_CONST_SET_OP_] = true;
    if (outcome_fields[0] == "SCALAR_UNIFORM_SET_OP")
      _ops[instruction::SCALAR_UNIFORM_SET_OP_] = true;
    if (outcome_fields[0] == "VECTOR_UNIFORM_SET_OP")
      _ops[instruction::VECTOR_UNIFORM_SET_OP_] = true;
    if (outcome_fields[0] == "MATRIX_UNIFORM_SET_OP")
      _ops[instruction::MATRIX_UNIFORM_SET_OP_] = true;
    if (outcome_fields[0] == "SCALAR_GAUSSIAN_SET_OP")
      _ops[instruction::SCALAR_GAUSSIAN_SET_OP_] = true;
    if (outcome_fields[0] == "VECTOR_GAUSSIAN_SET_OP")
      _ops[instruction::VECTOR_GAUSSIAN_SET_OP_] = true;
    if (outcome_fields[0] == "MATRIX_GAUSSIAN_SET_OP")
      _ops[instruction::MATRIX_GAUSSIAN_SET_OP_] = true;
    if (outcome_fields[0] == "SCALAR_COND_A_OP")
      _ops[instruction::SCALAR_COND_A_OP_] = true;
    if (outcome_fields[0] == "SCALAR_COND_B_OP")
      _ops[instruction::SCALAR_COND_B_OP_] = true;
    if (outcome_fields[0] == "SCALAR_POW_OP")
      _ops[instruction::SCALAR_POW_OP_] = true;
    if (outcome_fields[0] == "SCALAR_SQR_OP")
      _ops[instruction::SCALAR_SQR_OP_] = true;
    if (outcome_fields[0] == "SCALAR_CUBE_OP")
      _ops[instruction::SCALAR_CUBE_OP_] = true;
    if (outcome_fields[0] == "SCALAR_TANH_OP")
      _ops[instruction::SCALAR_TANH_OP_] = true;
    if (outcome_fields[0] == "SCALAR_SQRT_OP")
      _ops[instruction::SCALAR_SQRT_OP_] = true;
      
    if (outcome_fields[0] == "active_tasks")
      params[outcome_fields[0]] = outcome_fields[1];
    else if (outcome_fields[1].find('.') !=
        std::string::npos)  // found double parameter
      params[outcome_fields[0]] = stringToDouble(outcome_fields[1]);
    else  // found int parameter
      params[outcome_fields[0]] = stringToInt(outcome_fields[1]);
  }
}

/******************************************************************************/
void TPG::resetOutcomes(int phase, bool roots) {
  if (roots)
    for (auto teiter = _Mroot.begin(); teiter != _Mroot.end(); teiter++)
      (*teiter)->resetOutcomes(phase);
  else
    for (auto teiter = _M.begin(); teiter != _M.end(); teiter++)
      (*teiter)->resetOutcomes(phase);
}

/******************************************************************************/
void TPG::setOutcome(team *tm, string behav, vector<double> &rewards,
                     vector<int> &ints, long gtime) {
  point *p = new point(gtime, state_["point_count"]++, behav, rewards, ints);
  p->key(GetParam<int>("fit_mode"));
  tm->setOutcome(p);
}

/******************************************************************************/
void TPG::finalize() {
  // TODO: remove clears that are not required
  _allComponentsA.clear();
  _allComponentsAt.clear();
  _eliteTeams.clear();
  _eliteTeamPS.clear();
  _eliteTestScoresMQ.clear();
  _Mroot.clear();
  _teamMap.clear();
  _Lids.clear();
  _Mids.clear();
  _Memids.clear();
  _Memids.resize(memoryEigen::NUM_MEMORY_TYPES);
  state_["memory_eigen_count"] = 0;
  _numEliteTeamsCurrent.clear();
  for (size_t i = 0; i < _NUM_PHASE; i++) _numEliteTeamsCurrent.push_back(0);
  _persistenceFilterA.clear();
  _persistenceFilterA_t.clear();
  _persistenceFilterAllTime.clear();
  state_["point_count"] = 0;
  state_["program_count"] = 0;
  _taskSetMap.clear();
  state_["team_count"] = 0;
  _teamPairsToCompair.clear();

  for (auto teiter = _M.begin(); teiter != _M.end(); teiter++) {
    (*teiter)->resetOutcomes(-1);
    delete *teiter;
  }
  _M.clear();

  for (auto leiter = _L.begin(); leiter != _L.end(); leiter++)
    delete leiter->second;
  _L.clear();

  for (size_t memType = 0; memType < memoryEigen::NUM_MEMORY_TYPES; memType++) {
    for (auto meiter = _Memory[memType].begin();
         meiter != _Memory[memType].end(); meiter++)
      delete meiter->second;
    _Memory[memType].clear();
  }
  _Memory.clear();
  _Memory.resize(memoryEigen::NUM_MEMORY_TYPES);

  _phyloGraph.clear();
}

/******************************************************************************/
void TPG::genTeams() {
  team *pm2 = *_Mroot.begin();
  team *cm;

  size_t n_new_teams = 0;

  int n_nonempty_sets = 0;
  for (auto &set : powerSet(GetParam<int>("n_task")))
    n_nonempty_sets += set.size() == 0 ? 0 : 1;

  for (auto &set : powerSet(GetParam<int>("n_task"))) {
    if (_taskSetMap[vecToStrNoSpace(set)].size() == 0) continue;
    vector<team *> parent = _taskSetMap[vecToStrNoSpace(set)];
    uniform_int_distribution<int> disP(0, parent.size() - 1);

    for (int i = 0; i < GetParam<int>("n_elite") / n_nonempty_sets; i++) {
      auto pm1 = parent[disP(_rngs[TPG_SEED_INDEX])];
      bool crossover =
          real_distribution_(_rngs[TPG_SEED_INDEX]) < GetParam<double>("pmx")
              ? true
              : false;
      if (crossover) {
        do {
          pm2 = parent[disP(_rngs[TPG_SEED_INDEX])];
        } while (pm1->id_ == pm2->id_);
      }
      genTeams(pm1, pm2, crossover, &cm, n_new_teams);

      _phyloGraph[pm1->id_].adj.push_back(cm->id_);
      if (crossover) _phyloGraph[pm2->id_].adj.push_back(cm->id_);
      _phyloGraph.insert(pair<long, phyloRecord>(cm->id_, phyloRecord()));
      _phyloGraph[cm->id_].gtime = GetState("t_current");
      _phyloGraph[cm->id_].root = true;
      addTeam(cm);
      _Mroot.insert(cm);
      n_new_teams++;
    }
  }
  oss << "genTms t " << GetState("t_current") << " Msz " << _M.size() << " Lsz "
      << _L.size() << " rSz " << _Mroot.size() << " mSz";
  for (size_t memType = 0; memType < memoryEigen::NUM_MEMORY_TYPES; memType++)
    oss << " " << _Memory[memType].size();
  oss << _Memory.size() << " eLSz " << _numEliteTeamsCurrent[GetState("phase")];
  oss << " nNTms " << n_new_teams << endl;
}

/******************************************************************************/
void TPG::genTeams(team *pm1, team *pm2, bool crossover, team **cm,
                   size_t &numNewTeams) {
  list<program *> *p1programs;
  list<program *> *p2programs;
  vector<program *> cprograms;

  double b;

  program *lr;

  bool changedL;
  bool changedM;

  uniform_int_distribution<int> disL(0, _L.size() - 1);

  pm1->getMembersRef(p1programs);
  auto p1liter = p1programs->begin();
  *cm = new team(GetState("t_current"), state_["team_count"]++);

  _phyloGraph[(*cm)->id_].ancestorIds.insert(pm1->id_);
  (*cm)->addAncestorId(pm1->id_);

  bool linearCrossover = false;
  team *cm2 = (*cm);

  // linear crossover with two parent programs (i.e. two teams with one program
  // each)
  if (crossover && GetParam<double>("p_atomic") == 1.0 && pm1->size() == 1 &&
      pm2->size() == 1 &&
      real_distribution_(_rngs[TPG_SEED_INDEX]) <
          GetParam<double>("p_bid_xover")) {
    _phyloGraph[(*cm)->id_].ancestorIds.insert(pm2->id_);
    (*cm)->addAncestorId(pm2->id_);

    linearCrossover = true;
    vector<program *> pm1Programs;
    pm1->members(pm1Programs);
    vector<program *> pm2Programs;
    pm1->members(pm2Programs);
    linearM *c1;  // = NULL;
    linearM *c2;  // = NULL;
    programCrossover(dynamic_cast<linearM *>(pm1Programs[0]),
                     dynamic_cast<linearM *>(pm2Programs[0]), &c1, &c2,
                     _rngs[TPG_SEED_INDEX]);

    cm2 = new team(GetState("t_current"), state_["team_count"]++);
    numNewTeams++;

    _phyloGraph[(cm2)->id_].ancestorIds.insert(pm1->id_);
    (cm2)->addAncestorId(pm1->id_);
    _phyloGraph[(cm2)->id_].ancestorIds.insert(pm2->id_);
    (cm2)->addAncestorId(pm2->id_);

    if (real_distribution_(_rngs[TPG_SEED_INDEX]) < 0.5) {
      (*cm)->addProgram(c1);
      (cm2)->addProgram(c2);
    } else {
      (*cm)->addProgram(c2);
      (cm2)->addProgram(c1);
    }

    addTeam(cm2);
    _Mroot.insert(cm2);
    _phyloGraph.insert(pair<long, phyloRecord>(cm2->id_, phyloRecord()));
    _phyloGraph[cm2->id_].gtime = GetState("t_current");
    _phyloGraph[cm2->id_].root = true;
  }

  // team crossover
  else if (crossover && (pm1->size() > 1 || pm2->size() > 1)) {
    _phyloGraph[(*cm)->id_].ancestorIds.insert(pm2->id_);
    (*cm)->addAncestorId(pm2->id_);

    pm2->getMembersRef(p2programs);
    auto p2liter = p2programs->begin();
    while (p1liter != p1programs->end() || p2liter != p2programs->end()) {
      if (p1liter != p1programs->end() &&
          (int)(*cm)->size() < GetParam<int>("max_team_size") &&
          (((*p1liter)->action() < 0 && (*cm)->numAtomic_ < 1) ||
           find(p2programs->begin(), p2programs->end(), *p1liter) !=
               p2programs->end()))
        (*cm)->addProgram(*p1liter);
      else if ((int)(*cm)->size() < GetParam<int>("max_team_size") &&
               p1liter != p1programs->end() &&
               real_distribution_(_rngs[TPG_SEED_INDEX]) < 0.5)
        (*cm)->addProgram(*p1liter);
      if ((int)(*cm)->size() < GetParam<int>("max_team_size") &&
          p2liter != p2programs->end() &&
          real_distribution_(_rngs[TPG_SEED_INDEX]) < 0.5)
        (*cm)->addProgram(*p2liter);
      if (p1liter != p1programs->end()) p1liter++;
      if (p2liter != p2programs->end()) p2liter++;
    }
    if ((*cm)->numAtomic_ < 1)
      die(__FILE__, __FUNCTION__, __LINE__,
          "Crossover must leave the fail-safe atomic program!");
  } else
    for (p1liter = p1programs->begin(); p1liter != p1programs->end(); p1liter++)
      (*cm)->addProgram(*p1liter);

  (*cm)->members(cprograms);

  uniform_int_distribution<int> disM(0, _M.size() - 1);

  // Remove programs.
  program *p;
  uniform_int_distribution<int> disCprograms(0, cprograms.size() - 1);
  for (b = 1.0;
       real_distribution_(_rngs[TPG_SEED_INDEX]) < b && ((*cm)->size() > 1);
       b = b * GetParam<double>("pmd")) {
    do {
      p = cprograms[disCprograms(_rngs[TPG_SEED_INDEX])];
    } while ((p->action() < 0 &&
              (*cm)->numAtomic_ <
                  2) ||  // keep at least one program with an atomic action
             !((*cm)->removeProgram(p)));
  }

  // Add programs.
  for (b = 1.0; real_distribution_(_rngs[TPG_SEED_INDEX]) < b &&
                (int)(*cm)->size() < GetParam<int>("max_team_size");
       b = b * GetParam<double>("pma")) {
    uniform_int_distribution<int> disTmSize(0, (*cm)->size() - 1);
    do {
      p = _L[_Lids[disL(_rngs[TPG_SEED_INDEX])]];
    } while (!((*cm)->addProgram(p, disTmSize(_rngs[TPG_SEED_INDEX]))));
  }

  // Change program order.
  if ((*cm)->size() > 1 &&
      real_distribution_(_rngs[TPG_SEED_INDEX]) < GetParam<double>("pmw")) {
    int i, j;
    uniform_int_distribution<int> disMemberList(0, (*cm)->size() - 1);
    do {
      i = disMemberList(_rngs[TPG_SEED_INDEX]);
      j = disMemberList(_rngs[TPG_SEED_INDEX]);
    } while (i == j);
    (*cm)->muProgramOrder(i, j);
  }

  // Mutate programs.
  deque<program *> programsWithNoRefs;
  for (size_t tm = 0; tm < (linearCrossover ? 2 : 1); tm++) {
    team *teamToMutate = tm == 0 ? (*cm) : cm2;

    changedM = false;

    set<program *, programIdComp> cprogramsCopy;
    teamToMutate->members(cprogramsCopy);  // need to copy for cloning/removing

    while (GetParam<double>("pmm") > 0 && changedM == false) {
      for (auto ccliter = cprogramsCopy.begin(); ccliter != cprogramsCopy.end();
           ++ccliter)   
        if (real_distribution_(_rngs[TPG_SEED_INDEX]) <
            GetParam<double>("pmm")) {
          changedM = true;
          teamToMutate->removeProgram(*ccliter);

          // clone program
          lr = new linearM(GetState("t_current"),
                           *(dynamic_cast<linearM *>(*ccliter)), params_,
                           state_["program_count"]++);
          if (lr->action() >= 0)
            _teamMap[lr->action()]->addIncomingProgram(lr->id_);
          for (size_t memType = 0; memType < memoryEigen::NUM_MEMORY_TYPES;
               memType++) {
            lr->memSet(memType, (*ccliter)->memGet(
                                    memType));  // copy memoryEigen reference
            lr->memGet(memType)->refInc();
          }
          if (linearCrossover) delete *ccliter;
          // modify program
          do {
            changedL = lr->muBid(params_, _rngs[TPG_SEED_INDEX],
                                 real_distribution_, _ops);
          } while (changedL == false);
          // change memory pointer
          if (real_distribution_(_rngs[TPG_SEED_INDEX]) <
              GetParam<double>("pms")) {
            uniform_int_distribution<int> disMemory(0, _Memory.size() - 1);
            memoryEigen *memNew;
            do {
              memNew = _Memory[memoryEigen::SCALAR_TYPE]
                              [_Memids[memoryEigen::SCALAR_TYPE]
                                      [disMemory(_rngs[TPG_SEED_INDEX])]];
            } while (lr->memGet(memoryEigen::SCALAR_TYPE)->id_ == memNew->id_);
            lr->memGet(memoryEigen::SCALAR_TYPE)->refDec();
            lr->memSet(memoryEigen::SCALAR_TYPE, memNew);
            lr->memGet(memoryEigen::SCALAR_TYPE)->refInc();
          }
          // change action pointer
          uniform_int_distribution<int> disAct(
              0, GetParam<int>("n_discrete_action") - 1);
          if (real_distribution_(_rngs[TPG_SEED_INDEX]) <
              GetParam<double>("pmn")) {
            long a;
            // atomic
            if (GetState("t_current") == 1 ||  // first generation is atomic
                (lr->action() < 0 &&
                 teamToMutate->numAtomic_ <
                     2) ||  // always mutate the fail-safe atomic program to
                            // another atomic
                real_distribution_(_rngs[TPG_SEED_INDEX]) <
                    GetParam<double>("p_atomic")) {
              if (GetParam<int>("n_discrete_action") > 1) {
                do {
                  a = -1 -
                      disAct(_rngs[TPG_SEED_INDEX]);  // atomic actions are
                                                      // negatives: -1 down to
                                                      // -numAtomicActions()
                } while (lr->action() == a);
                if (lr->action() >= 0)
                  _teamMap[lr->action()]->removeIncomingProgram(lr->id_);
                lr->muAction(a);
              }
            }
            // path
            else {
              team *tm;
              do {
                tm = _teamMap[_Mids[disM(_rngs[TPG_SEED_INDEX])]];
              } while ((tm->gtime_ == GetState("t_current") ||
                        tm->clones_ > 0 || lr->action() == tm->id_));

              if (lr->action() >= 0)
                _teamMap[lr->action()]->removeIncomingProgram(lr->id_);
              if (!tm->root()) {  // already subsumed, don't clone
                lr->muAction(tm->id_);
                tm->addIncomingProgram(lr->id_);
              } else {  // clone when subsumed
                tm->prunePrograms(programsWithNoRefs);
                team *sub =
                    new team(GetState("t_current"), state_["team_count"]++);
                tm->clone(_phyloGraph, &sub);
                lr->muAction(sub->id_);
                sub->addIncomingProgram(lr->id_);
                _phyloGraph[tm->id_].adj.push_back(sub->id_);
                _phyloGraph.insert(
                    pair<long, phyloRecord>(sub->id_, phyloRecord()));
                _phyloGraph[sub->id_].gtime = GetState("t_current");
                _phyloGraph[sub->id_].root = false;
                addTeam(sub);
                numNewTeams++;
              }
            }
          }
          teamToMutate->addProgram(lr);
          addProgram(lr);
        }
    }
  }

  for (size_t tm = 0; tm < (linearCrossover ? 2 : 1); tm++) {
    team *teamToMutate = tm == 0 ? (*cm) : cm2;
    list<program *> *cp;
    teamToMutate->getMembersRef(cp);
    for (auto cliter = cp->begin(); cliter != cp->end(); cliter++)
      (*cliter)->refInc();
  }
  cleanupProgramsWithNoRefs(GetState("t_current"), programsWithNoRefs, true);
}

/******************************************************************************/
// team* TPG::genTeamsInternal(long t,
//       mt19937 &rng,
//       set < team*, teamIdComp >  &internalReplacements, //new candidates
//       map < long, team* > &associatedRoots){
//
//    //get pool of internal teams
//    vector < team *> nonRootTeams;
//    for (auto it = _M.begin(); it != _M.end(); it++)
//       if (!(*it)->root())
//          nonRootTeams.push_back(*it);
//    uniform_int_distribution<int> disNonRootTeams(0,nonRootTeams.size()-1);
//    map <long, team*> parentPool;
//    int N1 = 10;
//    for (int i = 0; i < N1; i++){
//       team * tm = nonRootTeams[disNonRootTeams(rng)];
//       parentPool[tm->id_] = tm;
//       tm->resetPolicyRootIds();
//    }
//
//    //update associated root nodes for all internal teams
//    set <team*, teamIdComp> visitedTeams;
//    for (auto riter = _Mroot.begin(); riter != _Mroot.end(); riter++){
//       visitedTeams.clear();
//       long id = (*riter)->id_;
//       (*riter)->updatePolicyRoot(_teamMap, visitedTeams, id);
//    }
//
//    //count number of associated roots for each internal team
//    map <long, size_t> aRootSzMap;
//    for (auto ti = parentPool.begin(); ti != parentPool.end(); ti++){
//       //get associated roots
//       set < long > tmIds;
//       ti->second->policyRootIds(tmIds);
//       aRootSzMap[ti->second->id_] = tmIds.size();
//    }
//
//    //pick internal team with fewest associated roots
//    auto it = min_element(aRootSzMap.begin(), aRootSzMap.end(), [](const auto&
//    l, const auto& r) { return l.second < r.second; }); team * pm1 =
//    parentPool[it->first]; set < long > tmIds;
//    pm1->policyRootIds(tmIds);//this could just be done once
//
//    for (auto it = tmIds.begin(); it != tmIds.end(); it++){
//       associatedRoots[*it] = _teamMap[*it];
//       _teamMap[*it]->resetOutcomes(_VALIDATION_PHASE);
//    }
//
//    //parent2 for possible crossover
//    team *pm2 = _teamMap[_Mids[0]];
//    uniform_int_distribution<int> dis2(0,_Mids.size()-1);
//
//    //produce N children of pm1 (maybe pm2)
//    team *cm;
//    size_t numNewTeams = 0;
//    size_t N2 = 10;
//    while (numNewTeams < N2){
//       bool crossover = real_distribution_(rng) < _pmx ? true : false;
//       if (crossover){
//          do { pm2 = _teamMap[_Mids[dis2(rng)]]; }
//          while (pm1->id_ == pm2->id_);
//       }
//       genTeams(t, pm1, pm2, crossover, &cm, numNewTeams);
//       if (crossover)
//          _phyloGraph[pm2->id_].adj.push_back(cm->id_);
//       _phyloGraph[pm1->id_].adj.push_back(cm->id_);
//       _phyloGraph.insert(pair<long, phyloRecord>(cm->id_, phyloRecord()));
//       _phyloGraph[cm->id_].gtime = GetState("t_current");
//       _phyloGraph[cm->id_].root = false;
//       addTeam(cm);
//       internalReplacements.insert(cm);
//       numNewTeams++;
//    }
//    return pm1;
// }

/******************************************************************************/
team *TPG::getBestTeam() {
  set<team *, teamFitnessLexicalCompare> teams;

  for (auto teiter = _Mroot.begin(); teiter != _Mroot.end(); teiter++)
    teams.insert(*teiter);

  return *teams.begin();
}

/******************************************************************************/
void TPG::setEliteTeams(int t, int phase, int fitMode, bool verbose) {
  vector<vector<double>> minScoresST, maxScoresST;
  minScoresST.resize(1);
  maxScoresST.resize(1);
  vector<team *> teamsRankedVec;
  teamsRankedVec.reserve(_Mroot.size() * GetParam<int>("n_task"));

  // Find the elite single-task program graphs
  _numEliteTeamsCurrent[phase] = 0;

  minScoresST[fitMode].resize(GetParam<int>("n_task"));
  maxScoresST[fitMode].resize(GetParam<int>("n_task"));
  for (int o = 0; o < GetParam<int>("n_task"); o++) {
    teamsRankedVec.clear();
    for (auto teiter = _Mroot.begin(); teiter != _Mroot.end(); teiter++) {
      (*teiter)->elite(phase, false);
      if ((phase == _TRAIN_PHASE && (*teiter)->numOutcomes(phase, o) > 0) ||
          ((*teiter)->numOutcomes(phase, o) >= _numStoredOutcomesPerHost[phase])) {
        // if ((*teiter)->runTimeComplexityIns() == 0)
        //    (*teiter)->updateComplexityRecord(_teamMap, _numPointAuxDouble -
        //    2);
        (*teiter)->fit_ = (*teiter)->getQuickMean(o, fitMode, phase);
        teamsRankedVec.push_back(*teiter);
        if (phase == _TEST_PHASE) {
          markEffectiveCode(*teiter);
          if ((*teiter)->runTimeComplexityIns() == 0)
            (*teiter)->updateComplexityRecord(
                _teamMap, GetParam<int>("n_point_aux_double") - 2);
          _phyloGraph[(*teiter)->id_].fitness = (*teiter)->fit_;
          _phyloGraph[(*teiter)->id_].numActiveFeatures =
              (*teiter)->numActiveFeatures_;
          _phyloGraph[(*teiter)->id_].numActivePrograms =
              (*teiter)->numActivePrograms_;
          _phyloGraph[(*teiter)->id_].numActiveTeams =
              (*teiter)->numActiveTeams_;
          _phyloGraph[(*teiter)->id_].numEffectiveInstructions =
              (*teiter)->numEffectiveInstructions();
        }
      } else
        (*teiter)->elite(true);  // mark elite to protect until eval in all
                                 // tasks
    }
    if (teamsRankedVec.size() > 0) {
      sort(teamsRankedVec.begin(), teamsRankedVec.end(),
           teamFitnessLexicalCompare());
      maxScoresST[fitMode][o] = (*(teamsRankedVec.begin()))->fit_;
      minScoresST[fitMode][o] = (*(teamsRankedVec.rbegin()))->fit_;
    }
  }

  // find multitask elites
  int n_nonempty_sets = 0;
  for (auto &set : powerSet(GetParam<int>("n_task")))
    n_nonempty_sets += set.size() == 0 ? 0 : 1;
  auto PS = powerSet(GetParam<int>("n_task"));

  for (auto &set : PS) {
    if (phase == _TRAIN_PHASE) _taskSetMap[vecToStrNoSpace(set)].clear();
    teamsRankedVec.clear();
    vector<double> normalizedScores;
    double rawMeanScore;
    for (auto teiter = _Mroot.begin(); teiter != _Mroot.end(); teiter++) {
      normalizedScores.clear();
      for (size_t o = 0; o < set.size(); o++) {
        if (((phase == _TRAIN_PHASE &&
              (*teiter)->numOutcomes(phase, set[o]) > 0) ||
             ((*teiter)->numOutcomes(phase, set[o]) >=
                                          _numStoredOutcomesPerHost[phase]))) {
          rawMeanScore = (*teiter)->getQuickMean(set[o], fitMode, phase);
          if (!isEqual(minScoresST[fitMode][set[o]],
                       maxScoresST[fitMode][set[o]]))
            normalizedScores.push_back(
                (rawMeanScore - minScoresST[fitMode][set[o]]) /
                (maxScoresST[fitMode][set[o]] - minScoresST[fitMode][set[o]]));
          else
            normalizedScores.push_back(
                isfinite(rawMeanScore) ? rawMeanScore : 0);  // be careful
        }
      }
      if (normalizedScores.size() == set.size()) {
        (*teiter)->fit_ = *min_element(
            normalizedScores.begin(),
            normalizedScores.end());  // if PS.size() == 1 then
                                      // normalizedScores.size() == 1
        teamsRankedVec.push_back(*teiter);
      }
    }
    if (teamsRankedVec.size() > 0) {
      sort(teamsRankedVec.begin(), teamsRankedVec.end(),
           teamFitnessLexicalCompare());
      size_t ne = GetParam<int>("n_elite") / n_nonempty_sets;
      size_t c = 0;
      bool gotPSElite = false;
      for (auto teiter = teamsRankedVec.begin();
           teiter != teamsRankedVec.end() && c < ne; teiter++) {
        if (!gotPSElite) {
          _eliteTeamPS[vecToStrNoSpace(set)][fitMode][phase] =
              *(teiter);  // keep track of single elite for each taskset
          gotPSElite = true;
        }
        if (!(*teiter)->elite(phase)) {
          c++;
          _numEliteTeamsCurrent[phase]++;
          (*teiter)->elite(phase, true);
          if (phase == _TRAIN_PHASE) {
            (*teiter)->fitnessBin(t, vecToStrNoSpace(set));
            _phyloGraph[(*teiter)->id_].fitnessBin = (*teiter)->fitnessBin();
            _phyloGraph[(*teiter)->id_].fitness = (*teiter)->fit_;
          }
          if (phase == _TRAIN_PHASE)
            _taskSetMap[vecToStrNoSpace(set)].push_back(*teiter);
        }
      }
      _eliteTeamPS[vecToStrNoSpace(set)][fitMode][phase] =
          *(teamsRankedVec.begin());
    }

    if (verbose && set.size() == 1 &&
        haveEliteTeam(vecToStrNoSpace(set), fitMode, phase)) {
      oss << "setElTmsST eLSz " << _numEliteTeamsCurrent[GetState("phase")]
          << " ss " << vecToStrNoSpace(set) << " fm " << fitMode << " minThr "
          << _eliteTeamPS[vecToStrNoSpace(set)][fitMode][phase]->fit_ << " ";
      printTeamInfo(t, phase, false,
                    _eliteTeamPS[vecToStrNoSpace(set)][fitMode][phase]->id_);
    }
    if (verbose && set.size() == (size_t)GetParam<int>("n_task") &&
        GetParam<int>("n_task") > 1 &&
        haveEliteTeam(vecToStrNoSpace(set), fitMode, phase)) {
      oss << "setElTmsMTA eLSz " << _numEliteTeamsCurrent[GetState("phase")]
          << " ss " << vecToStrNoSpace(set) << " fm " << fitMode << " minThr "
          << _eliteTeamPS[vecToStrNoSpace(set)][fitMode][phase]->fit_ << " ";
      printTeamInfo(t, phase, false,
                    _eliteTeamPS[vecToStrNoSpace(set)][fitMode][phase]->id_);
    }

    // if (phase == _TRAIN_PHASE)
    //    if (haveEliteTeam(vecToStrNoSpace(PS[ss]), fitMode, _TEST_PHASE))
    //       getEliteTeam(vecToStrNoSpace(PS[ss]), fitMode,
    //       _TEST_PHASE)->elite(phase,true);
  }
}

/******************************************************************************/
// void TPG::internalReplacementPareto(
//       int phase,
//       int fitMode,
//       int complexityAuxDouble,
//       team * tm,
//       map < long, team* > &associatedRoots,
//       set < team*, teamIdComp > &internalReplacements,
//       mt19937 &rng){
//    (void)rng;
//    vector <double> minScoresST, maxScoresST;
//    minScoresST.resize(GetParam<int>("n_task"));
//    maxScoresST.resize(GetParam<int>("n_task"));
//
//    vector <team *> teamsRankedVec;
//    teamsRankedVec.reserve(internalReplacements.size() *
//    GetParam<int>("n_task"));
//
//    vector <double> internalTeamMeanReward;
//    internalTeamMeanReward.reserve(associatedRoots.size());
//    internalTeamMeanReward.resize(associatedRoots.size());
//    vector <double> internalTeamMeanComplexity;
//    internalTeamMeanComplexity.reserve(associatedRoots.size());
//    internalTeamMeanComplexity.resize(associatedRoots.size());
//    for (size_t o = 0; o < GetParam<int>("n_task"); o++){
//       teamsRankedVec.clear();
//       for(auto teiterInternal = internalReplacements.begin(); teiterInternal
//       != internalReplacements.end(); teiterInternal++){
//          int r = 0;
//          for (auto teiterRoots = associatedRoots.begin(); teiterRoots !=
//          associatedRoots.end(); teiterRoots++){
//             internalTeamMeanReward[r] =
//             teiterRoots->second->getMeanOutcome(phase, o, fitMode,
//             POINT_AUX_INT_internalTestNodeId, (*teiterInternal)->id_, false,
//             false); internalTeamMeanComplexity[r++] =
//             teiterRoots->second->getMeanOutcome(phase, o,
//             complexityAuxDouble, POINT_AUX_INT_internalTestNodeId,
//             (*teiterInternal)->id_, false, false);
//          }
//          (*teiterInternal)->setQuickMean(o, fitMode, phase,
//          vecMedian(internalTeamMeanReward));
//          (*teiterInternal)->fit((*teiterInternal)->getQuickMean(o, fitMode,
//          phase));
//          (*teiterInternal)->runTimeComplexityIns(vecMedian(internalTeamMeanComplexity));
//          teamsRankedVec.push_back(*teiterInternal);
//       }
//       if (teamsRankedVec.size() > 0){
//          sort(teamsRankedVec.begin(), teamsRankedVec.end(),
//          teamFitnessCompare()); maxScoresST[o] =
//          (*(teamsRankedVec.begin()))->fit(); minScoresST[o] =
//          (*(teamsRankedVec.rbegin()))->fit();
//       }
//    }
//
//    //re-map fitness score to minimum normalized score over all objectives
//    teamsRankedVec.clear();
//    vector <double> normalizedScores;
//    double rawMeanScore;
//    for(auto teiterInternal = internalReplacements.begin(); teiterInternal !=
//    internalReplacements.end(); teiterInternal++){
//       normalizedScores.clear();
//       for (size_t o = 0; o < GetParam<int>("n_task"); o++){
//          rawMeanScore = (*teiterInternal)->getQuickMean(o, fitMode, phase);
//          if (!isEqual(minScoresST[o], maxScoresST[0]))
//             normalizedScores.push_back((rawMeanScore -
//             minScoresST[o])/(maxScoresST[o] - minScoresST[o]));
//          else
//             normalizedScores.push_back(rawMeanScore / minScoresST[o]);
//          //watch out for nan here in case of zero scores...
//       }
//       (*teiterInternal)->fit(*min_element(normalizedScores.begin(),
//       normalizedScores.end())); teamsRankedVec.push_back(*teiterInternal);
//    }
//
//    ////find pareto front wrt fitness & complexity
//    //vector < team* > paretoFrontTeams;
//    //double minComplexity = numeric_limits<double>::max();
//    //sort(teamsRankedVec.begin(), teamsRankedVec.end(),
//    teamFitnessLexicalCompare());
//    //for (auto itA = teamsRankedVec.begin(); itA != teamsRankedVec.end();
//    itA++){
//    //addFront:
//    //   minComplexity = min(minComplexity, (*itA)->runTimeComplexityIns());
//    //   paretoFrontTeams.push_back(*itA);
//    //   auto itB = next(itA, 1);
//    //   while (itB != teamsRankedVec.end()){
//    //      if ((*itB)->runTimeComplexityIns() < minComplexity){
//    //         itA = itB;
//    //         goto addFront;
//    //      }
//    //      itB++;
//    //   }
//    //   break;
//    //}
//    ////DEBUG PARETO BY PLOTTING
//    //cout << " pFront size " << paretoFrontTeams.size() << endl;
//    //for (auto it = paretoFrontTeams.begin(); it != paretoFrontTeams.end();
//    it ++)
//    //   cout << "pfront t " << GetState("t_current") << " " << (*it)->id_ <<
//    " " << (*it)->fit() << " " << -1*(*it)->runTimeComplexityIns() << endl;
//
//    //uniform_int_distribution<int> disP(0, paretoFrontTeams.size() - 1);
//    //team* paretoWinner = paretoFrontTeams[disP(rng)];
//
//    sort(teamsRankedVec.begin(), teamsRankedVec.end(),
//    teamFitComplexLexCompare()); team* paretoWinner =
//    *(teamsRankedVec.begin());
//
//    //if (isEqual(paretoWinner->fit(), tm->fit()))
//    //   paretoWinner = tm;
//
//    //update incoming program pointers
//    set < long > incomingProgramIds;
//    tm->incomingPrograms(incomingProgramIds);
//    for (auto piditer = incomingProgramIds.begin(); piditer !=
//    incomingProgramIds.end(); piditer++){
//       tm->removeIncomingProgram(*piditer);
//       _L[*piditer]->action(paretoWinner->id_);
//       paretoWinner->addIncomingProgram(*piditer);
//    }
//
//    //delete the unused candidates
//    internalReplacements.erase(paretoWinner);
//    deque < program* > programsWithNoRefs;
//    for (auto teiter = internalReplacements.begin(); teiter !=
//    internalReplacements.end(); teiter++){
//       _phyloGraph[(*teiter)->id_].dtime = GetState("t_current");
//       (*teiter)->cleanup(_teamMap, programsWithNoRefs);
//       _teamMap.erase((*teiter)->id_);
//       removeTeam(*teiter, true);//_M.erase(*teiter);
//       _Mroot.erase(*teiter);
//       delete *teiter;
//    }
//    cleanupProgramsWithNoRefs(GetState("t_current"), programsWithNoRefs,
//    true);
//
//    //update score wrt paretoWinner
//    for (auto teiterRoots = associatedRoots.begin(); teiterRoots !=
//    associatedRoots.end(); teiterRoots++)
//       teiterRoots->second->swapOutcomePhase(phase, _TRAIN_PHASE,
//       POINT_AUX_INT_internalTestNodeId, paretoWinner->id_);
// }

/******************************************************************************/
// Helper struct for distance comparisons
struct distanceInstance {
  double distance;
  bool fromArchive;
  distanceInstance(double d, bool a) : distance(d), fromArchive(a) {}
};

/******************************************************************************/
bool compareByDistance(const distanceInstance &a, const distanceInstance &b) {
  return a.distance < b.distance;
}

/******************************************************************************/
void TPG::initTeams() {
  auto n_init = GetParam<int>("n_elite");
  uniform_int_distribution<int> disA(0, GetParam<int>("n_discrete_action") - 1);
  for (int tc = 0; tc < n_init; tc++) {
    auto m = new team(GetState("t_current"), state_["team_count"]++);
    for (int n = 0; n < GetParam<int>("initial_team_size"); n++) {
      long discrete_action =
          -1 -
          disA(_rngs[TPG_SEED_INDEX]);  // discrete atomic actions are negatives
                                        // -1 to -numAtomicActions()
      auto l =
          new linearM(GetState("t_current"), discrete_action, params_,
                      state_["program_count"]++, _rngs[TPG_SEED_INDEX], _ops);
      for (size_t memType = 0; memType < memoryEigen::NUM_MEMORY_TYPES;
           memType++) {
        auto *m =
            new memoryEigen(state_["memory_eigen_count"]++, memType, params_);
        addMemory(m);
        l->memSet(memType, m);
        l->memGet(memType)->refInc();
      }
      l->stateful(GetParam<int>("stateful"));
      m->addProgram(l);
      l->refInc();
      addProgram(l);
    }
    addTeam(m);
    _Mroot.insert(m);

    _phyloGraph.insert(pair<long, phyloRecord>(m->id_, phyloRecord()));
    _phyloGraph[m->id_].gtime = 0;
  }
  oss << "initTms Msz " << _M.size() << " Lsz " << _L.size() << " rSz "
      << _Mroot.size() << " mSz";
  for (size_t memType = 0; memType < memoryEigen::NUM_MEMORY_TYPES; memType++)
    oss << " " << _Memory[memType].size();
  oss << _Memory.size() << " eLSz " << _numEliteTeamsCurrent[GetState("phase")]
      << endl;
}

/******************************************************************************/
void TPG::policyFeatures(int hostId, set<long> &features, bool active) {
  features.clear();
  set<team *, teamIdComp> visitedTeams;
  if (hostId == -1) {
    for (auto it = _Mroot.begin(); it != _Mroot.end(); it++) {
      visitedTeams.clear();
      (*it)->policyFeatures(_teamMap, visitedTeams, features, active);
    }
  } else
    _teamMap[hostId]->policyFeatures(_teamMap, visitedTeams, features, active);
}

/******************************************************************************/
// Print graph defined by <rootTeam> in DOT format for GraphViz
void TPG::printGraphDot(
    team *rootTeam, size_t frame, int episode, int step, size_t depth,
    vector<program *> allPrograms, vector<program *> winningPrograms,
    vector<set<long>> decisionFeatures,
    vector<set<memoryEigen *, memoryEigenIdComp>> decisionMemories,
    vector<team *> teamPath, bool drawPath,
    set<team *, teamIdComp> visitedTeamsAllTasks) {
  // unused arguments
  (void)decisionFeatures;
  (void)decisionMemories;
  (void)allPrograms;

  // just use winning programs up to a specific graph depth
  // vector<program*> winningProgramsDepth(winningPrograms.begin(),
  // winningPrograms.begin()+depth);

  vector<program *> winningProgramsDepth(winningPrograms.begin(),
                                         winningPrograms.end());

  double nodeWidth = 2.0;
  double edgeWidth_1 = 1;  // 5;
  double edgeWidth_2 = 30;
  double arrowSize_1 = 1;  // 0.1;
  double arrowSize_2 = 1;  // 0.2;

  char outputFilename[80];
  ofstream ofs;

  set<team *, teamIdComp> teams;
  set<program *, programIdComp> programs;
  set<memoryEigen *, memoryEigenIdComp> memories;

  // rootTeam->getAllNodes(_teamMap, teams, programs, memories, false);
  //(void)visitedTeamsAllTasks;
  for (auto it = visitedTeamsAllTasks.begin(); it != visitedTeamsAllTasks.end();
       it++) {
    set<program *, programIdComp> p;
    (*it)->members(p);
    programs.insert(p.begin(), p.end());
  }
  for (auto it = programs.begin(); it != programs.end(); it++) {
    for (size_t memType = 0; memType < memoryEigen::NUM_MEMORY_TYPES; memType++)
      memories.insert((*it)->memGet(memType));
  }

  sprintf(outputFilename, "replay/graphs/gv_%d_%05d_%03d_%05d_%05d%s",
          (int)rootTeam->id_, (int)frame, episode, step, (int)depth, ".dot");
  ofs.open(outputFilename, ios::out);
  if (!ofs) die(__FILE__, __FUNCTION__, __LINE__, "Can't open file.");

  ofs << "digraph G {" << endl;
  ofs << "ratio=1" << endl;
  ofs << "root=t_" << rootTeam->id_ << endl;

  ////atomic actions
  // for(auto leiter = programs.begin(); leiter != programs.end(); leiter++)
  //    if ((*leiter)->action() < 0)
  //       ofs << " a_" << ((*leiter)->action()*-1)-1 << "_" << (*leiter)->id_
  //       << " [shape=point, label=\"\", regular=1, width=0.1]" << endl;

  // programs
  for (auto leiter = programs.begin(); leiter != programs.end(); leiter++) {
    if (step > 0 &&
        find(winningProgramsDepth.begin(), winningProgramsDepth.end(),
             *leiter) != winningProgramsDepth.end()) {
      // ofs << " p_" << (*leiter)->id_ << " [shape=box, style=filled,
      // color=black, label=\"\", fontsize=200, regular=1, width=" << nodeWidth
      // << "]" << endl;
      ofs << " p_" << (*leiter)->id_
          << " [shape=box, style=filled, color=green, label=\"\", "
             "fontsize=200, regular=1, width="
          << nodeWidth << "]" << endl;
    } else if (step > 0 && find(allPrograms.begin(), allPrograms.end(),
                                *leiter) != allPrograms.end())
      ofs << " p_" << (*leiter)->id_
          << " [shape=box, style=filled, color=black, label=\"\", "
             "fontsize=200, regular=1, width="
          << nodeWidth << "]" << endl;
    else if (teamPath.size() > 0)
      ofs << " p_" << (*leiter)->id_
          << " [shape=box, style=filled, color=grey90, label=\"\", "
             "fontsize=200, regular=1, width="
          << nodeWidth << "]" << endl;
    else
      ofs << " p_" << (*leiter)->id_
          << " [shape=box, style=filled, color=grey70, label=\"\", "
             "fontsize=200, regular=1, width="
          << nodeWidth << "]" << endl;
  }

  // teams
  int label = 0;
  // for(auto teiter = teams.begin(); teiter != teams.end(); teiter++)
  for (auto teiter = visitedTeamsAllTasks.begin();
       teiter != visitedTeamsAllTasks.end(); teiter++)
    if ((*teiter)->id_ == rootTeam->id_ ||
        (step > 0 &&
         find(teamPath.begin(), teamPath.end(), *teiter) != teamPath.end()))
      ofs << " t_" << (*teiter)->id_
          << " [shape=circle, style=filled, fillcolor=black, label=\"t"
          << label++ << "\", fontsize=84, regular=1, width=" << nodeWidth * 2
          << "]" << endl;
    else
      // ofs << " t_" << (*teiter)->id_ << " [shape=circle, style=filled,
      // fillcolor=grey90, label=\"\", fontsize=84, regular=1, width=" <<
      // nodeWidth*2 << "]" << endl;
      ofs << " t_" << (*teiter)->id_
          << " [shape=circle, style=filled, fillcolor=deepskyblue, label=\"\", "
             "fontsize=84, regular=1, width="
          << nodeWidth * 2 << "]" << endl;

  ////memoryEigen registers
  // for(auto meiter = memories.begin(); meiter != memories.end(); meiter++)
  //    ofs << " m_" << (*meiter)->id_ << " [shape=invhouse, style=filled,
  //    fillcolor=grey, label=\"\", regular=1, width=" << nodeWidth << "]" <<
  //    endl;

  // program -> team edges
  for (auto leiter = programs.begin(); leiter != programs.end(); leiter++) {
    if ((*leiter)->action() >= 0 &&
        find(visitedTeamsAllTasks.begin(), visitedTeamsAllTasks.end(),
             _teamMap[(*leiter)->action()]) != visitedTeamsAllTasks.end()) {
      double w = find(winningProgramsDepth.begin(), winningProgramsDepth.end(),
                      *leiter) == winningProgramsDepth.end() ||
                         !drawPath
                     ? edgeWidth_1
                     : edgeWidth_2;
      if (teamPath.size() < 1) w = 5;
      double as = find(winningProgramsDepth.begin(), winningProgramsDepth.end(),
                       *leiter) == winningProgramsDepth.end() ||
                          !drawPath
                      ? arrowSize_1
                      : arrowSize_2;
      string col =
          find(winningProgramsDepth.begin(), winningProgramsDepth.end(),
               *leiter) == winningProgramsDepth.end()
              ? "black"
              : "green";
      ofs << " p_" << (*leiter)->id_ << "->"
          << "t_" << (*leiter)->action() << " [arrowsize=" << as
          << ", penwidth=" << w << " color=" << col.c_str() << "];" << endl;
    }
  }

  ////program -> memoryEigen edges
  // for(auto leiter = programs.begin(); leiter != programs.end(); leiter++){
  //    double w = find(winningProgramsDepth.begin(),
  //    winningProgramsDepth.end(), *leiter) == winningProgramsDepth.end() ?
  //    edgeWidth_1 : edgeWidth_2; double as =
  //    find(winningProgramsDepth.begin(), winningProgramsDepth.end(), *leiter)
  //    == winningProgramsDepth.end() ? arrowSize_1 : arrowSize_2; ofs << " p_"
  //    << (*leiter)->id_ << "->" << "m_"<< (*leiter)->memGet()->id_;
  //       ofs << " [dir=both, arrowsize=" << as << ", penwidth=" << w << "];"
  //       << endl;
  // }

  // team -> program edges
  // for(auto teiter = teams.begin(); teiter != teams.end(); teiter++){
  for (auto teiter = visitedTeamsAllTasks.begin();
       teiter != visitedTeamsAllTasks.end(); teiter++) {
    list<program *> mem;
    (*teiter)->members(&mem);
    for (auto leiter = mem.begin(); leiter != mem.end(); leiter++) {
      double w =
          find(teamPath.begin(), teamPath.end(), *teiter) == teamPath.end() ||
                  !drawPath
              ? edgeWidth_1
              : edgeWidth_2;
      if (teamPath.size() < 1) w = 5;
      double as =
          find(teamPath.begin(), teamPath.end(), *teiter) == teamPath.end() ||
                  !drawPath
              ? arrowSize_1
              : arrowSize_2;
      string col =
          find(winningProgramsDepth.begin(), winningProgramsDepth.end(),
               *leiter) == winningProgramsDepth.end()
              ? "black"
              : "green";
      ofs << " t_" << (*teiter)->id_ << "->p_" << (*leiter)->id_
          << " [arrowsize=" << as << ", penwidth=" << w
          << " color=" << col.c_str() << "];" << endl;
    }
  }
  ofs << "}" << endl;
  ofs.close();
}

/******************************************************************************/
void TPG::printGraphDotGPEM(long rootTeamId, map<long, string> &teamColMap,
                            set<team *, teamIdComp> &visitedTeamsAllTasks,
                            vector<map<long, double>> &teamUseMapPerTask) {
  team *rootTeam = _teamMap[rootTeamId];

  (void)teamColMap;
  vector<string> taskCol;
  taskCol.push_back("#7fc97f");
  taskCol.push_back("#beaed4");
  taskCol.push_back("#fdc086");
  taskCol.push_back("#ffff99");
  taskCol.push_back("#386cb0");
  taskCol.push_back("#f0027f");
  map<long, string> nodeLabMap;

  nodeLabMap[1594815] = "1";  //   29000"
  nodeLabMap[624659] = "2";   //  149"
  nodeLabMap[1302870] = "3";  //   28373"
  nodeLabMap[623346] = "4";   //  580"
  nodeLabMap[493990] = "5";   //  149"
  nodeLabMap[836830] = "6";   //  28019"
  nodeLabMap[548151] = "7";   //  832"
  nodeLabMap[126871] = "8";   //  1373"
  nodeLabMap[425177] = "9";   //  774"
  nodeLabMap[602173] = "10";  //   1826"
  nodeLabMap[42314] = "11";   //  9"
  nodeLabMap[26879] = "12";   //  7"
  nodeLabMap[200127] = "13";  //   4"
  nodeLabMap[470578] = "14";  //   1826"
  nodeLabMap[5964] = "15";    // 7"
  nodeLabMap[23266] = "16";   //  2"
  nodeLabMap[226807] = "17";  //   394"
  nodeLabMap[180005] = "18";  //   953"

  double nodeWidth = 2.0;
  double arrowSize_1 = 3;  // 0.1;

  char outputFilename[80];
  ofstream ofs;

  set<team *, teamIdComp> teams;
  set<program *, programIdComp> programs;
  set<memoryEigen *, memoryEigenIdComp> memories;

  for (auto it = visitedTeamsAllTasks.begin(); it != visitedTeamsAllTasks.end();
       it++) {
    set<program *, programIdComp> p;
    (*it)->members(p);
    programs.insert(p.begin(), p.end());
  }
  for (auto it = programs.begin(); it != programs.end(); it++) {
    for (size_t memType = 0; memType < memoryEigen::NUM_MEMORY_TYPES; memType++)
      memories.insert((*it)->memGet(memType));
  }

  sprintf(outputFilename,
          "replay/gv_taskDecomposition_%d_%05d_%03d_%05d_%05d%s",
          (int)rootTeam->id_, 0, 0, 0, 0, ".dot");
  ofs.open(outputFilename, ios::out);
  if (!ofs) die(__FILE__, __FUNCTION__, __LINE__, "Can't open file.");

  ofs << "strict digraph G {" << endl;
  ofs << "ratio=0.7" << endl;
  ofs << "root=t_" << rootTeam->id_ << endl;

  ////programs
  // for(auto leiter = programs.begin(); leiter != programs.end(); leiter++){
  //    ofs << " p_" << (*leiter)->id_ << " [shape=box, style=filled,
  //    color=grey70, label=\"\", fontsize=200, regular=1, width=" << nodeWidth
  //    << "]" << endl;
  // }

  // teams
  for (auto teiter = visitedTeamsAllTasks.begin();
       teiter != visitedTeamsAllTasks.end(); teiter++) {
    string col = "";
    // if (teamColMap.find((*teiter)->id_) != teamColMap.end())
    //    col = teamColMap[(*teiter)->id_];
    // else
    //    col = "white";

    ofs << " t_" << (*teiter)->id_
        << " [shape=circle, style=wedged, fillcolor=\"";
    double sumUse = 0;
    for (int tsk = 0; tsk < 6; tsk++)
      sumUse += teamUseMapPerTask[tsk][(*teiter)->id_];
    for (int tsk = 0; tsk < 6; tsk++) {
      ofs << taskCol[tsk] << ";"
          << (teamUseMapPerTask[tsk].find((*teiter)->id_) !=
                      teamUseMapPerTask[tsk].end()
                  ? teamUseMapPerTask[tsk][(*teiter)->id_] / sumUse
                  : 0);
      if (tsk < 5) ofs << ":";
    }
    // ofs << ":" << taskCol[1] << ";"<<
    // teamUseMapPerTask[0].find((*teiter)->id_) != teamUseMapPerTask[0].end()
    // ? teamUseMapPerTask[0][(*teiter)->id_]/6: "0"; ofs << ":" << taskCol[2]
    // << ";"<< teamUseMapPerTask[0].find((*teiter)->id_) !=
    // teamUseMapPerTask[0].end() ? teamUseMapPerTask[0][(*teiter)->id_]/6:
    // "0"; ofs << ":" << taskCol[3] << ";"<<
    // teamUseMapPerTask[0].find((*teiter)->id_) != teamUseMapPerTask[0].end()
    // ? teamUseMapPerTask[0][(*teiter)->id_]/6: "0"; ofs << ":" << taskCol[4]
    // << ";"<< teamUseMapPerTask[0].find((*teiter)->id_) !=
    // teamUseMapPerTask[0].end() ? teamUseMapPerTask[0][(*teiter)->id_]/6:
    // "0"; ofs << ":" << taskCol[5] << ";"<<
    // teamUseMapPerTask[0].find((*teiter)->id_) != teamUseMapPerTask[0].end()
    // ? teamUseMapPerTask[0][(*teiter)->id_]/6: "0";
    ofs << "\"";

    ofs << ", label=\""
        << (nodeLabMap.find((*teiter)->id_) == nodeLabMap.end()
                ? ""
                : nodeLabMap[(*teiter)->id_])
        << "\", fontsize=84, regular=1, width=" << nodeWidth * 2
        << ",penwidth=0]" << endl;
    // ofs << " t_" << (*teiter)->id_ << " [shape=circle, style=wedged,
    // fillcolor=\"" << col << "\", label=\"\", fontsize=84, regular=1, width="
    // << nodeWidth*2 << "]" << endl;
  }

  ////program -> team edges
  // for(auto leiter = programs.begin(); leiter != programs.end(); leiter++){
  //    if ((*leiter)->action() >= 0 && find(visitedTeamsAllTasks.begin(),
  //    visitedTeamsAllTasks.end(), _teamMap[(*leiter)->action()]) !=
  //    visitedTeamsAllTasks.end()){
  //       ofs << " p_" << (*leiter)->id_ << "->" << "t_"<< (*leiter)->action()
  //       << " [arrowsize=" << arrowSize_1 << ", penwidth=" << "1" << " color="
  //       << "black" << "];" << endl;
  //    }
  // }

  ////team -> program edges
  // for(auto teiter = visitedTeamsAllTasks.begin(); teiter !=
  // visitedTeamsAllTasks.end(); teiter++){
  //    list < program * > mem;
  //    (*teiter)->members(&mem);
  //    for(auto leiter = mem.begin(); leiter != mem.end(); leiter++){
  //       ofs << " t_" << (*teiter)->id_ << "->p_" << (*leiter)->id_ << "
  //       [arrowsize=" << arrowSize_1  << ", penwidth=" << "1" << " color=" <<
  //       "black" << "];" << endl;
  //    }
  // }

  // team -> team edges
  for (auto teiter = visitedTeamsAllTasks.begin();
       teiter != visitedTeamsAllTasks.end(); teiter++) {
    list<program *> mem;
    (*teiter)->members(&mem);
    for (auto leiter = mem.begin(); leiter != mem.end(); leiter++) {
      // ofs << " t_" << (*teiter)->id_ << "->p_" << (*leiter)->id_ << "
      // [arrowsize=" << arrowSize_1  << ", penwidth=" << "1" << " color=" <<
      // "black" << "];" << endl;
      if ((*leiter)->action() >= 0 &&
          find(visitedTeamsAllTasks.begin(), visitedTeamsAllTasks.end(),
               _teamMap[(*leiter)->action()]) != visitedTeamsAllTasks.end()) {
        ofs << " t_" << (*teiter)->id_ << "->t_" << (*leiter)->action()
            << " [arrowsize=" << arrowSize_1 << ", penwidth="
            << "1"
            << " color="
            << "black"
            << "];" << endl;
      }
    }
  }

  ////legend
  // ofs << "subgraph {" << endl;
  // ofs << "ratio=1" << endl;
  // ofs << "rank=sink" << endl;
  // ofs << "node [shape=plaintext]" << endl;
  // ofs << "legend [colorscheme=set18," << endl;
  // ofs << "label=<" << endl;
  //
  // ofs << "<table border=\"0\" cellborder=\"1\" cellspacing=\"0\">" << endl;
  // ofs << "<tr><td bgcolor=\"" << taskCol[0] << "\">" << "CartPole" <<
  // "</td></tr>" << endl; ofs << "<tr><td bgcolor=\"" << taskCol[1] << "\">" <<
  // "Acrobot" << "</td></tr>" << endl; ofs << "<tr><td bgcolor=\"" <<
  // taskCol[2] << "\">" << "CartCentering" << "</td></tr>" << endl; ofs <<
  // "<tr><td bgcolor=\"" << taskCol[3] << "\">" << "Pendulum" << "</td></tr>"
  // << endl; ofs << "<tr><td bgcolor=\"" << taskCol[4] << "\">" <<
  // "MountainCar" << "</td></tr>" << endl; ofs << "<tr><td bgcolor=\"" <<
  // taskCol[5] << "\">" << "MountainCarC." << "</td></tr>" << endl;
  //
  // ofs << "</table>>" << endl;
  // ofs << ", fontsize=84, regular=1];" << endl;
  // ofs << "}" << endl;
  ///////////////////////////////////////////////////////////

  ofs << "}" << endl;
  ofs.close();
}

/******************************************************************************/
void TPG::printGraphDotGPEMAnimate(
    long rootTeamId, size_t frame, int episode, int step, size_t depth,
    vector<program *> allPrograms, vector<program *> winningPrograms,
    set<team *, teamIdComp> &visitedTeamsAllTasks,
    vector<map<long, double>> &teamUseMapPerTask, vector<team *> teamPath) {
  team *rootTeam = _teamMap[rootTeamId];

  vector<string> taskCol;
  taskCol.push_back("#7fc97f");
  taskCol.push_back("#beaed4");
  taskCol.push_back("#fdc086");
  taskCol.push_back("#ffff99");
  taskCol.push_back("#386cb0");
  taskCol.push_back("#f0027f");
  map<long, string> nodeLabMap;

  nodeLabMap[1594815] = "1";  //   29000"
  nodeLabMap[624659] = "2";   //  149"
  nodeLabMap[1302870] = "3";  //   28373"
  nodeLabMap[623346] = "4";   //  580"
  nodeLabMap[493990] = "5";   //  149"
  nodeLabMap[836830] = "6";   //  28019"
  nodeLabMap[548151] = "7";   //  832"
  nodeLabMap[126871] = "8";   //  1373"
  nodeLabMap[425177] = "9";   //  774"
  nodeLabMap[602173] = "10";  //   1826"
  nodeLabMap[42314] = "11";   //  9"
  nodeLabMap[26879] = "12";   //  7"
  nodeLabMap[200127] = "13";  //   4"
  nodeLabMap[470578] = "14";  //   1826"
  nodeLabMap[5964] = "15";    // 7"
  nodeLabMap[23266] = "16";   //  2"
  nodeLabMap[226807] = "17";  //   394"
  nodeLabMap[180005] = "18";  //   953"

  bool drawPath = true;
  double nodeWidth = 2.0;
  double arrowSize_1 = 3;
  double arrowSize_2 = 3;
  double edgeWidth_2 = 10;
  double edgeWidth_1 = 1;

  char outputFilename[80];
  ofstream ofs;

  set<team *, teamIdComp> teams;
  set<program *, programIdComp> programs;
  set<memoryEigen *, memoryEigenIdComp> memories;
  vector<program *> winningProgramsDepth(winningPrograms.begin(),
                                         winningPrograms.end());

  for (auto it = visitedTeamsAllTasks.begin(); it != visitedTeamsAllTasks.end();
       it++) {
    set<program *, programIdComp> p;
    (*it)->members(p);
    programs.insert(p.begin(), p.end());
  }
  for (auto it = programs.begin(); it != programs.end(); it++) {
    for (size_t memType = 0; memType < memoryEigen::NUM_MEMORY_TYPES; memType++)
      memories.insert((*it)->memGet(memType));
  }

  sprintf(outputFilename, "replay/graphs/gv_%d_%05d_%03d_%05d_%05d%s",
          (int)rootTeam->id_, (int)frame, episode, step, (int)depth, ".dot");
  ofs.open(outputFilename, ios::out);
  if (!ofs) die(__FILE__, __FUNCTION__, __LINE__, "Can't open file.");

  ofs << "strict digraph G {" << endl;
  ofs << "ratio=0.7" << endl;
  ofs << "root=t_" << rootTeam->id_ << endl;

  // programs
  for (auto leiter = programs.begin(); leiter != programs.end(); leiter++) {
    if (step > 0 &&
        find(winningProgramsDepth.begin(), winningProgramsDepth.end(),
             *leiter) != winningProgramsDepth.end()) {
      // ofs << " p_" << (*leiter)->id_ << " [shape=box, style=filled,
      // color=black, label=\"\", fontsize=200, regular=1, width=" << nodeWidth
      // << "]" << endl;
      ofs << " p_" << (*leiter)->id_
          << " [shape=box, style=filled, color=green, label=\"\", "
             "fontsize=200, regular=1, width="
          << nodeWidth << "]" << endl;
    } else if (step > 0 && find(allPrograms.begin(), allPrograms.end(),
                                *leiter) != allPrograms.end())
      ofs << " p_" << (*leiter)->id_
          << " [shape=box, style=filled, color=black, label=\"\", "
             "fontsize=200, regular=1, width="
          << nodeWidth << "]" << endl;
    else if (teamPath.size() > 0)
      ofs << " p_" << (*leiter)->id_
          << " [shape=box, style=filled, color=grey90, label=\"\", "
             "fontsize=200, regular=1, width="
          << nodeWidth << "]" << endl;
    else
      ofs << " p_" << (*leiter)->id_
          << " [shape=box, style=filled, color=grey70, label=\"\", "
             "fontsize=200, regular=1, width="
          << nodeWidth << "]" << endl;
  }

  // teams
  for (auto teiter = visitedTeamsAllTasks.begin();
       teiter != visitedTeamsAllTasks.end(); teiter++) {
    string col = "";

    ofs << " t_" << (*teiter)->id_
        << " [shape=circle, style=wedged, fillcolor=\"";
    double sumUse = 0;
    for (int tsk = 0; tsk < 6; tsk++) {
      if (teamUseMapPerTask[tsk].find((*teiter)->id_) !=
          teamUseMapPerTask[tsk].end())
        sumUse += teamUseMapPerTask[tsk][(*teiter)->id_];
    }
    for (int tsk = 0; tsk < 6; tsk++) {
      ofs << taskCol[tsk] << ";"
          << (teamUseMapPerTask[tsk].find((*teiter)->id_) !=
                      teamUseMapPerTask[tsk].end()
                  ? teamUseMapPerTask[tsk][(*teiter)->id_] / sumUse
                  : 0);
      if (tsk < 5) ofs << ":";
    }
    ofs << "\"";

    ofs << ", label=\""
        << (nodeLabMap.find((*teiter)->id_) == nodeLabMap.end()
                ? ""
                : nodeLabMap[(*teiter)->id_])
        << "\", fontsize=84, regular=1, width=" << nodeWidth * 2
        << ",penwidth=0]" << endl;
  }

  ////team -> team edges
  // for(auto teiter = visitedTeamsAllTasks.begin(); teiter !=
  // visitedTeamsAllTasks.end(); teiter++){
  //    list < program * > mem;
  //    (*teiter)->members(&mem);
  //    for(auto leiter = mem.begin(); leiter != mem.end(); leiter++){
  //       if ((*leiter)->action() >= 0 && find(visitedTeamsAllTasks.begin(),
  //       visitedTeamsAllTasks.end(), _teamMap[(*leiter)->action()]) !=
  //       visitedTeamsAllTasks.end()){
  //          ofs << " t_" << (*teiter)->id_ << "->t_" << (*leiter)->action()
  //          << " [arrowsize=" << arrowSize_1  << ", penwidth=" << "1" << "
  //          color=" << "black" << "];" << endl;
  //       }
  //    }
  // }

  ////team -> team edges path
  // if (step > 0){
  //    for(size_t t = 0; t < teamPath.size()-1; t++){
  //       list < program * > mem;
  //       teamPath[t]->members(&mem);
  //       for(auto leiter = mem.begin(); leiter != mem.end(); leiter++){
  //          if ((*leiter)->action() >= 0 && teamPath[t+1]->id_ ==
  //          _teamMap[(*leiter)->action()]->id_){
  //             ofs << " t_" << teamPath[t]->id_ << "->t_" <<
  //             (*leiter)->action() << " [arrowsize=" << arrowSize_2  << ",
  //             penwidth=" << edgeWidth_2 << " color=" << "black" << "];" <<
  //             endl;
  //          }
  //       }
  //    }
  // }

  // program -> team edges
  for (auto leiter = programs.begin(); leiter != programs.end(); leiter++) {
    if ((*leiter)->action() >= 0 &&
        find(visitedTeamsAllTasks.begin(), visitedTeamsAllTasks.end(),
             _teamMap[(*leiter)->action()]) != visitedTeamsAllTasks.end()) {
      double w = find(winningProgramsDepth.begin(), winningProgramsDepth.end(),
                      *leiter) == winningProgramsDepth.end() ||
                         !drawPath
                     ? edgeWidth_1
                     : edgeWidth_2;
      if (teamPath.size() < 1) w = 5;
      double as = find(winningProgramsDepth.begin(), winningProgramsDepth.end(),
                       *leiter) == winningProgramsDepth.end() ||
                          !drawPath
                      ? arrowSize_1
                      : arrowSize_2;
      string col =
          find(winningProgramsDepth.begin(), winningProgramsDepth.end(),
               *leiter) == winningProgramsDepth.end()
              ? "black"
              : "green";
      ofs << " p_" << (*leiter)->id_ << "->"
          << "t_" << (*leiter)->action() << " [arrowsize=" << as
          << ", penwidth=" << w << " color=" << col.c_str() << "];" << endl;
    }
  }

  // team -> program edges
  for (auto teiter = visitedTeamsAllTasks.begin();
       teiter != visitedTeamsAllTasks.end(); teiter++) {
    list<program *> mem;
    (*teiter)->members(&mem);
    for (auto leiter = mem.begin(); leiter != mem.end(); leiter++) {
      double w =
          find(teamPath.begin(), teamPath.end(), *teiter) == teamPath.end() ||
                  !drawPath
              ? edgeWidth_1
              : edgeWidth_2;
      if (teamPath.size() < 1) w = 5;
      double as =
          find(teamPath.begin(), teamPath.end(), *teiter) == teamPath.end() ||
                  !drawPath
              ? arrowSize_1
              : arrowSize_2;
      string col =
          find(winningProgramsDepth.begin(), winningProgramsDepth.end(),
               *leiter) == winningProgramsDepth.end()
              ? "black"
              : "green";
      ofs << " t_" << (*teiter)->id_ << "->p_" << (*leiter)->id_
          << " [arrowsize=" << as << ", penwidth=" << w
          << " color=" << col.c_str() << "];" << endl;
    }
  }

  ofs << "}" << endl;
  ofs.close();
}

/******************************************************************************/
void TPG::printPhyloGraphDot(team *tm) {
  vector<long> teamIds;
  tm->getAncestorIds(teamIds);

  double nodeWidth = 10.0;
  double edgeWidth_1 = 20.0;  // 5;
                              // double edgeWidth_2 = 30;
                              // double arrowSize_1 = 2;//0.1;
                              // double arrowSize_2 = 1;//0.2;

  char outputFilename[80];
  ofstream ofs;

  sprintf(outputFilename, "phyloGraphs/phylo-t%05d-s%d%s",
          (int)GetState("t_current"), _seeds[TPG_SEED_INDEX], ".dot");
  ofs.open(outputFilename, ios::out);
  if (!ofs) die(__FILE__, __FUNCTION__, __LINE__, "Can't open file.");

  ofs << "digraph G {" << endl;
  ofs << "ratio=0.5" << endl;
  ofs << "rankdir=\"LR\"" << endl;

  for (auto it = teamIds.begin(); it != teamIds.end(); it++) {
    ofs << "subgraph {" << endl;

    string col = "";
    if (_phyloGraph[*it].fitnessBin.compare("012") == 0)
      col = "1";
    else if (_phyloGraph[*it].fitnessBin.compare("12") == 0)
      col = "2";
    else if (_phyloGraph[*it].fitnessBin.compare("02") == 0)
      col = "3";
    else if (_phyloGraph[*it].fitnessBin.compare("2") == 0)
      col = "4";
    else if (_phyloGraph[*it].fitnessBin.compare("01") == 0)
      col = "5";
    else if (_phyloGraph[*it].fitnessBin.compare("1") == 0)
      col = "6";
    else if (_phyloGraph[*it].fitnessBin.compare("0") == 0)
      col = "8";

    ofs << " t_" << *it
        << " [shape=circle, style=filled, colorscheme=set18, color="
        << col.c_str()
        << ", label=\"\", fontsize=84, regular=1, width=" << nodeWidth << "]"
        << endl;

    if (_phyloGraph[*it].adj.size() > 0)
      for (size_t i = 0; i < _phyloGraph[*it].adj.size(); i++)
        if (find(teamIds.begin(), teamIds.end(), _phyloGraph[*it].adj[i]) !=
            teamIds.end())
          ofs << " t_" << *it << "->"
              << "t_" << _phyloGraph[*it].adj[i] << " [penwidth=" << edgeWidth_1
              << " color="
              << "black"
              << "];" << endl;
    ofs << "}" << endl;
  }

  ////legend
  // vector <int> S;
  // for (int tsk = 0; tsk < (int)GetParam<int>("n_task"); tsk++)
  //    S.push_back(tsk);
  // vector <int> tmpSet;
  // vector < vector < int > > PS;
  // findPowerSet(S, tmpSet, PS, GetParam<int>("n_task"), 1);
  // for (size_t ss = 0; ss < PS.size(); ss++)
  //    sort(PS[ss].begin(), PS[ss].end());
  // string col = "";
  // ofs << "subgraph {" << endl;
  // ofs << "ratio=1" << endl;
  // ofs << "node [shape=plaintext]" << endl;
  // ofs << "legend [colorscheme=set18," << endl;
  // ofs << "label=<" << endl;
  // ofs << "<table border=\"0\" cellborder=\"1\" cellspacing=\"0\">" << endl;
  // string taskSetString = "";
  // for (size_t ss = 0; ss < PS.size(); ss++){
  //    if (vecToStrNoSpace(PS[ss]) == "012"){
  //       col = "1";
  //       taskSetString = "Acrobot + Cart Centering + Mountain Car Continuous";
  //    }
  //    else if (vecToStrNoSpace(PS[ss]) == "12"){
  //       col = "2";
  //       taskSetString = "Cart Centering + Mountain Car Continuous";
  //    }
  //    else if (vecToStrNoSpace(PS[ss]) == "02"){
  //       col = "3";
  //       taskSetString = "Acrobot + Mountain Car Continuous";
  //    }
  //    else if (vecToStrNoSpace(PS[ss]) == "2"){
  //       col = "4";
  //       taskSetString = "Mountain Car Continuous";
  //    }
  //    else if (vecToStrNoSpace(PS[ss]) == "01"){
  //       col = "5";
  //       taskSetString = "Acrobot + Cart Centering";
  //    }
  //    else if (vecToStrNoSpace(PS[ss]) == "1"){
  //       col = "6";
  //       taskSetString = "Cart Centering";
  //    }
  //    else if (vecToStrNoSpace(PS[ss]) == "0"){
  //       col = "8";
  //       taskSetString = "Acrobot";
  //    }
  //    ofs << "<tr><td bgcolor=\"" << col << "\">" << taskSetString <<
  //    "</td></tr>" << endl;
  // }
  // ofs << "</table>>" << endl;
  // ofs << ", fontsize=84, regular=1];" << endl;
  // ofs << "}" << endl;
  ofs << "}" << endl;
  ofs.close();
}

/******************************************************************************/
void TPG::printTeamInfo(long t, int phase, bool singleBest, long teamId) {
  team *bestTeam = *(_Mroot.begin());
  if (singleBest && teamId == -1) bestTeam = getBestTeam();
  ostringstream tmposs;
  map<point *, double, pointLexicalLessThan> allOutcomes;
  // map < point *, double > :: iterator myoiter;
  vector<int> behaviourSequence;
  set<team *, teamIdComp> visitedTeams;
  for (auto teiter = _M.begin(); teiter != _M.end(); teiter++) {
    if ((!singleBest && (*teiter)->root() && teamId == -1) ||  // all root teams
        (!singleBest && (*teiter)->id_ == teamId) ||           // specific team
        (singleBest &&
         (*teiter)->id_ == bestTeam->id_))  // singleBest root team
    {
      oss << "tminfo t " << t << " id " << (*teiter)->id_ << " gtm "
          << (*teiter)->gtime_ << " phs " << phase;
      oss << " root " << ((*teiter)->root() ? 1 : 0);
      oss << " sz " << (*teiter)->size();
      oss << " asz " << (*teiter)->asize();
      oss << " age " << t - (*teiter)->gtime_;
      // oss << " compl";
      // oss << " " << (*teiter)->numActiveTeams_;
      // oss << " " << (*teiter)->numActivePrograms_;
      // oss << " " << (*teiter)->numEffectiveInstructions();
      // oss << " " << (*teiter)->numActiveFeatures_;
      oss << " nOut " << (*teiter)->numOutcomes(_TRAIN_PHASE, -1) << " "
          << (*teiter)->numOutcomes(_TEST_PHASE, -1);

      oss << setprecision(5) << fixed;

      oss << " mnOut";
      for (int phs : {0,1,2}) {
        bool allPhase = false;
        bool allTask = false;  // for genomic, set to true

        // multi-task
        // statistics stored in aux doubles
        for (int task = 0; task < GetParam<int>("n_task"); task++)
          for (int i = 0; i < GetParam<int>("n_point_aux_double"); i++) {
            if ((*teiter)->numOutcomes(phs, task) > 0) {
              oss << " p" << phs << "t" << task << "a" << i << " "
                  << (*teiter)->getMeanOutcome(phs, task, i, allPhase, allTask);
            } else
              oss << " p" << phs << "t" << task << "a" << i << " x";
          }
      }

      oss << " fit " << (*teiter)->fit_;

      oss << setprecision(2) << fixed;

      visitedTeams.clear();
      vector<int> programInstructionCounts, effectiveProgramInstructionCounts;
      (*teiter)->policyInstructions(_teamMap, visitedTeams,
                                    programInstructionCounts,
                                    effectiveProgramInstructionCounts);

      oss << " pIns "
          << accumulate(programInstructionCounts.begin(),
                        programInstructionCounts.end(), 0);
      oss << " mnProgIns " << vecMean(programInstructionCounts);

      oss << " ePIns "
          << accumulate(effectiveProgramInstructionCounts.begin(),
                        effectiveProgramInstructionCounts.end(), 0);
      oss << " mnEProgIns " << vecMean(effectiveProgramInstructionCounts);
      set<program *, programIdComp> programs;
      set<memoryEigen *, memoryEigenIdComp> memories;
      set<team *, teamIdComp> visitedTeams2;
      (*teiter)->getAllNodes(_teamMap, visitedTeams2, programs, memories,
                             false);
      oss << " nP " << programs.size();
      oss << " nT " << visitedTeams2.size();
      oss << " nM " << memories.size();

      visitedTeams.clear();
      set<long> pF;
      (*teiter)->policyFeatures(_teamMap, visitedTeams, pF, true);
      oss << " pF " << (double)pF.size() / GetParam<int>("n_input");

      vector<int> op_countsSingle;
      vector<int> op_countsTally;
      op_countsTally.resize(instruction::NUM_OP);

      fill(op_countsTally.begin(), op_countsTally.end(), 0);
      for (auto it = programs.begin(); it != programs.end(); it++) {
        (*it)->op_counts(op_countsSingle);
        for (size_t i = 0; i < op_countsSingle.size(); i++)
          op_countsTally[i] += op_countsSingle[i];
      }
      oss << " nOp " << vecToStr(op_countsTally);

      vector<int> tmSizesRoot, tmSizesSub;
      tmSizesRoot.push_back((*teiter)->size());
      for (auto teiter2 = visitedTeams2.begin(); teiter2 != visitedTeams2.end();
           teiter2++)
        if ((*teiter2)->id_ != (*teiter)->id_)  // not the root of this policy
          tmSizesSub.push_back((*teiter2)->size());
      oss << " mnTmSzR " << vecMean(tmSizesRoot) << " mnTmSzS "
          << (tmSizesSub.size() > 0 ? vecMean(tmSizesSub) : 0);

      ////map < int, map< int, map <point *, double, pointLexicalLessThan > > >
      /// outs;
      // map < int, map< int, map <int, point *> > > outs;
      //(*teiter)->outcomes(outs);

      // double maxVisitedTeams = 0;
      // for (auto ouiter = outs.begin(); ouiter != outs.end();
      // ouiter++){//tasks
      //   for (auto ouiter2 = ouiter->second[2].begin(); ouiter2 !=
      //   ouiter->second[2].end(); ouiter2++){//test outcomes, loop over
      //   envSeed
      //      maxVisitedTeams = max(maxVisitedTeams,
      //      ouiter2->second->auxDouble(2));
      //   }
      // }
      // oss << " maxVisitedTeams " << maxVisitedTeams << endl;

      // oss << " outcomes";
      // int nSolved = 0;
      // for(auto ouiter = outs[0][0].begin(); ouiter != outs[0][0].end();
      // ouiter++){
      //    if ((ouiter->first)->auxDouble(0) > 1)
      //       oss << " " << (ouiter->first)->auxDouble(0);
      //    if ((ouiter->first)->auxDouble(0) >= 50)
      //       nSolved++;
      // }
      // oss << " outcomesDesc";
      // for(auto ouiter = outs[0][0].begin(); ouiter != outs[0][0].end();
      // ouiter++){
      //    if ((ouiter->first)->auxDouble(0) >= 50)
      //       oss<< " " << (ouiter->first)->desc().c_str();
      // }
      // oss << " nSol " << nSolved;

      // oss << " fBins";
      // map <long, int> bins = (*teiter)->fitnessBins();
      // for (auto iter = bins.begin(); iter != bins.end(); iter++)
      //    oss << ":" << (*iter).first << "-" << (*iter).second;
      oss << endl;
    }
  }
}

/******************************************************************************/
// Algorithm 5.1 (linear crossover)
void TPG::programCrossover(linearM *p1, linearM *p2, linearM **c1, linearM **c2,
                           mt19937 &rng) {
  int dcMax = min(p1->size(), p2->size());
  int dsMax = dcMax;
  int lsMax = dcMax;

  cloneProgramLinearM(p1, c1);
  cloneProgramLinearM(p2, c2);

  int pos1, pos2;

  vector<program *> parents{p1, p2};
  vector<int> segLengths{1, 1};

  if (p1->size() > p2->size()) swap(parents[0], parents[1]);

  // 1
  uniform_int_distribution<> dis1(0, parents[0]->size() - 1);
  pos1 = dis1(rng);
  uniform_int_distribution<> dis2(0, parents[1]->size() - 1);
  do {
    pos2 = dis2(rng);
  } while (abs(pos1 - pos2) > min(parents[0]->size() - 1, dcMax));

  // 2,3
  uniform_int_distribution<> dis3(1, min(parents[0]->size() - pos1, lsMax));
  segLengths[0] = dis3(rng);
  uniform_int_distribution<> dis4(1, min(parents[1]->size() - pos2, lsMax));
  do {
    segLengths[1] = dis4(rng);
  } while (abs(segLengths[0] - segLengths[1]) > dsMax);

  // 4
  if (segLengths[0] > segLengths[1]) swap(segLengths[0], segLengths[1]);

  // 5
  if (p1->size() - (segLengths[1] - segLengths[0]) < 1 ||
      p2->size() + (segLengths[1] - segLengths[0]) >
          GetParam<int>("max_prog_size")) {
    if (real_distribution_(rng) < 0.5)
      segLengths[1] = segLengths[0];
    else
      segLengths[0] = segLengths[1];

    if (pos1 + segLengths[0] > p1->size())
      segLengths[0] = segLengths[1] = p1->size() - pos1;
  }

  vector<instruction *> parentProg1;
  p1->getBid(parentProg1);
  vector<instruction *> parentProg2;
  p2->getBid(parentProg2);

  vector<instruction *> childProg1;
  p1->getBid(childProg1);
  vector<instruction *> childProg2;
  p2->getBid(childProg2);

  // exchange seg1 in p1 by seg2 in p2
  childProg1.clear();
  auto start = parentProg1.begin();
  auto end = parentProg1.begin() + pos1;
  copy(start, end, back_inserter(childProg1));
  start = parentProg2.begin() + pos2;
  end = parentProg2.begin() + pos2 + segLengths[1];
  copy(start, end, back_inserter(childProg1));
  start = parentProg1.begin() + pos1 + segLengths[0];
  end = parentProg1.end();
  copy(start, end, back_inserter(childProg1));

  // exchange seg2 in p2 by seg1 in p1
  childProg2.clear();
  start = parentProg2.begin();
  end = parentProg2.begin() + pos2;
  copy(start, end, back_inserter(childProg2));
  start = parentProg1.begin() + pos1;
  end = parentProg1.begin() + pos1 + segLengths[0];
  copy(start, end, back_inserter(childProg2));
  start = parentProg2.begin() + pos2 + segLengths[1];
  end = parentProg2.end();
  copy(start, end, back_inserter(childProg2));
}

/******************************************************************************/
// Read in populations from a checkpoint file.
// (This whole process needs a rewrite sometime.)
void TPG::readCheckpoint(long t, int phase, int chkpID, bool fromString,
                         const string &inString) {
  finalize();  // clear populations

  string str;

  if (fromString) {
    str = inString;
  } else {
    char filename[80];
    sprintf(filename, "%s/%s.%ld.%d.%d.%d.rslt", "checkpoints", "cp", t, chkpID,
            _seeds[TPG_SEED_INDEX], phase);
    ifstream t(filename);
    t.seekg(0, ios::end);
    str.reserve(t.tellg());
    t.seekg(0, ios::beg);
    str.assign((istreambuf_iterator<char>(t)), istreambuf_iterator<char>());
  }

  istringstream iss(str);

  string oneline;
  char delim = ':';
  long memberId = 0;
  long max_teamCount = -1;
  long max_programCount = -1;
  int f;

  vector<string> outcomeFields;

  while (getline(iss, oneline)) {
    outcomeFields.clear();

    splitString(oneline, delim, outcomeFields);

    if (outcomeFields[0].compare("teamPair") == 0) {
      long id1 = atoi(outcomeFields[1].c_str());
      long id2 = atoi(outcomeFields[2].c_str());
      teamPair tp(_teamMap[id1], _teamMap[id2]);
      _teamPairsToCompair.push_back(tp);
    } else if (outcomeFields[0].compare("seed_tpg") == 0)
      seed(TPG_SEED_INDEX, atoi(outcomeFields[1].c_str()));
    else if (outcomeFields[0].compare("seed_aux") == 0)
      seed(AUX_SEED_INDEX, atoi(outcomeFields[1].c_str()));
    else if (outcomeFields[0].compare("t") == 0)
      state_["t_current"] = atoi(outcomeFields[1].c_str());
    else if (outcomeFields[0].compare("active_task") == 0)
      state_["active_task"] = atoi(outcomeFields[1].c_str());
    else if (outcomeFields[0].compare("internalTestNodeId") == 0)
      state_["internal_test_node_id"] = atoi(outcomeFields[1].c_str());
    else if (outcomeFields[0].compare("fitMode") == 0)
      params_["fit_mode"] = atoi(outcomeFields[1].c_str());
    else if (outcomeFields[0].compare("phase") == 0)
      state_["phase"] = atoi(outcomeFields[1].c_str());

    else if (outcomeFields[0].compare("memoryEigen") == 0) {
      size_t i = 1;
      long id = atoi(outcomeFields[i++].c_str());
      int type = atoi(outcomeFields[i++].c_str());
      int memoryIndices = atoi(outcomeFields[i++].c_str());
      int memoryRows = atoi(outcomeFields[i++].c_str());
      int memoryCols = atoi(outcomeFields[i++].c_str());
      int nrefs = atoi(outcomeFields[i++].c_str());
      memoryEigen *mem = new memoryEigen(id, type, memoryIndices, memoryRows,
                                         memoryCols, nrefs);
      // read in evolved constants
      for (int idx = 0; idx < memoryIndices; idx++) {
        if (type == memoryEigen::SCALAR_TYPE) {
          mem->const_memory_[idx](0, 0) = stod(outcomeFields[i++].c_str());
        } else if (type == memoryEigen::VECTOR_TYPE) {
          for (int r = 0; r < memoryRows; r++)
            mem->const_memory_[idx](r, 0) = stod(outcomeFields[i++].c_str());
        } else if (type == memoryEigen::MATRIX_TYPE) {
          for (int r = 0; r < memoryRows; r++)
            for (int c = 0; c < memoryCols; c++)
              mem->const_memory_[idx](r, c) = stod(outcomeFields[i++].c_str());
        }
      }
      addMemory(mem);
    }

    else if (outcomeFields[0].compare("linearM") == 0) {
      vector<int> memTypeIds;
      memTypeIds.resize(memoryEigen::NUM_MEMORY_TYPES);
      program *l;
      f = 1;
      long id = atoi(outcomeFields[f++].c_str());
      if (id > max_programCount) max_programCount = id;
      long gtime = atoi(outcomeFields[f++].c_str());
      long action = atoi(outcomeFields[f++].c_str());
      int stateful = atoi(outcomeFields[f++].c_str());
      long dim = atoi(outcomeFields[f++].c_str());
      (void)dim;
      int nrefs = atoi(outcomeFields[f++].c_str());
      for (size_t memType = 0; memType < memoryEigen::NUM_MEMORY_TYPES;
           memType++)
        memTypeIds[memType] = atoi(outcomeFields[f++].c_str());

      vector<instruction *> bid;
      for (size_t ii = f; ii < outcomeFields.size(); ii++) {
        vector<string> instructionString;
        splitString(outcomeFields[ii], '_', instructionString);
        instruction *in = new instruction(params_, _rngs[TPG_SEED_INDEX]);
        in->in1Src_ = stringToInt(instructionString[0]);
        in->in2Src_ = stringToInt(instructionString[1]);
        in->outSrc_ = stringToInt(instructionString[2]);
        in->outIdx_ = stringToInt(instructionString[3]);
        in->op_ = stringToInt(instructionString[4]);
        in->in1Idx_ = stringToInt(instructionString[5]);
        in->in1IdxE_ = stringToInt(instructionString[6]);
        in->in2Idx_ = stringToInt(instructionString[7]);
        in->in2IdxE_ = stringToInt(instructionString[8]);
        bid.push_back(in);
      }
      // l = new linearM(gtime, action, stateful, dim, _memoryIndices,
      // _memoryRows, _memoryCols, id, nrefs, bid);
      l = new linearM(gtime, action, stateful, params_, id, nrefs, bid);

      for (size_t memType = 0; memType < memoryEigen::NUM_MEMORY_TYPES;
           memType++) {
        l->memSet(memType, _Memory[memType][memTypeIds[memType]]);
        l->memGet(memType)->refInc();
      }

      addProgram(l);
    } else if (outcomeFields[0].compare("team") == 0) {
      team *m;
      f = 1;
      long id = atoi(outcomeFields[f++].c_str());
      if (id > max_teamCount) max_teamCount = id;
      long gtime = atoi(outcomeFields[f++].c_str());
      m = new team(gtime, id);
      // cerr << "new0 " << id << endl;
      m->_n_eval = atoi(outcomeFields[f++].c_str());
      // m->clearEvalSeeds();
      // for (size_t es = 0; es < m->numEval(); es++)
      //    m->addEvalSeed(atoi(outcomeFields[f++].c_str()));
      // m->taskCode(outcomeFields[f++]);
      // add programs in order
      for (size_t ii = f; ii < outcomeFields.size(); ii++) {
        memberId = atoi(outcomeFields[ii].c_str());
        if (m->addProgram(_L[memberId]) == false)
          m->addProgramActive(_L[memberId]);
      }
      addTeam(m);
      _Mroot.insert(m);
    } else if (outcomeFields[0].compare("teamIncoming") == 0) {
      f = 1;
      long id = atoi(outcomeFields[f++].c_str());
      for (size_t ii = f; ii < outcomeFields.size(); ii++) {
        long incomingId = atoi(outcomeFields[ii].c_str());
        _teamMap[id]->addIncomingProgram(incomingId);
      }
      _Mroot.erase(_teamMap[id]);
    } else if (outcomeFields[0].compare("fBin") == 0) {
      long id = atoi(outcomeFields[1].c_str());
      team *tm = _teamMap[id];
      for (size_t ii = 2; ii < outcomeFields.size(); ii++) {
        vector<string> fb;
        splitString(outcomeFields[ii].c_str(), '-', fb);
        tm->fitnessBin(atoi(fb[0].c_str()), fb[1]);
      }
    } else if (!fromString && outcomeFields[0].compare("phyloNode") == 0 &&
               find(outcomeFields.begin(), outcomeFields.end(), "gtime") ==
                   outcomeFields.end()) {
      f = 1;
      long id = atoi(outcomeFields[f++].c_str());
      _phyloGraph.insert(pair<long, phyloRecord>(id, phyloRecord()));
      _phyloGraph[id].gtime = atoi(outcomeFields[f++].c_str());
      _phyloGraph[id].dtime = atoi(outcomeFields[f++].c_str());
      _phyloGraph[id].fitnessBin = atoi(outcomeFields[f++].c_str());
      _phyloGraph[id].fitness = atof(outcomeFields[f++].c_str());
      _phyloGraph[id].root =
          atoi(outcomeFields[f++].c_str()) > 0 ? true : false;
    } else if (!fromString && outcomeFields[0].compare("phyloLink") == 0 &&
               find(outcomeFields.begin(), outcomeFields.end(), "from") ==
                   outcomeFields.end())
      _phyloGraph[atoi(outcomeFields[1].c_str())].adj.push_back(
          atoi(outcomeFields[2].c_str()));
    else if (!fromString && outcomeFields[0].compare("ancestorIds") == 0) {
      f = 1;
      long id = atoi(outcomeFields[f++].c_str());
      for (size_t ii = f; ii < outcomeFields.size(); ii++) {
        long aid = atoi(outcomeFields[ii].c_str());
        _phyloGraph[id].ancestorIds.insert(aid);
      }
    }
  }

  state_["program_count"] = max_programCount + 1;
  state_["team_count"] = max_teamCount + 1;

  // put this in a "sanity check" function
  int sumTeamSizes = 0;
  int nrefs = 0;
  int sumNumOutcomes = 0;

  if (!fromString)
    oss << "TPG::readCheckpoint "
        << " Msize " << _M.size() << " Lsize " << _L.size() << " MrooSize "
        << _Mroot.size();

  for (auto teiter = _M.begin(); teiter != _M.end(); teiter++) {
    sumTeamSizes += (*teiter)->size();
    sumNumOutcomes += (*teiter)->numOutcomes(_TRAIN_PHASE, -1);
  }

  if (fromString) recalculateProgramRefs();

  for (auto leiter = _L.begin(); leiter != _L.end(); leiter++)
    nrefs += leiter->second->refs();

  if (!fromString)
    oss << " sumTeamSizes " << sumTeamSizes << " nrefs " << nrefs
        << " sumNumOutcomes " << sumNumOutcomes << endl;

  // if(sumTeamSizes != nrefs)
  //    die(__FILE__, __FUNCTION__, __LINE__, "something messed up during
  //    readCheckpoint");
}

/******************************************************************************/
void TPG::recalculateProgramRefs() {
  for (auto leiter = _L.begin(); leiter != _L.end(); leiter++)
    leiter->second->setNrefs(0);

  set<program *, programIdComp> mem;
  for (auto teiter = _M.begin(); teiter != _M.end(); teiter++) {
    (*teiter)->members(&mem);
    for (auto leiter = mem.begin(); leiter != mem.end(); leiter++)
      (*leiter)->refInc();
    mem.clear();
  }
}

///****************************************************************************/
// void TPG::selSampleSets()
//{
//    size_t numDeleted = 0;
//    auto ss = _samplesets.begin();
//    while(ss != _samplesets.end()) {
//       if(!(*ss)->_elite) {
//          delete *ss;
//          ss = _samplesets.erase(ss);
//          numDeleted++;
//       }
//       else ss++;
//    }
//    oss << "selSmp t " << GetState("t_current") << " nD " << numDeleted << "
//    ssSz " << _samplesets.size() << endl;
// }
/******************************************************************************/
void TPG::selTeams(long t, bool verbose, int genTime) {
  (void)verbose;
  (void)genTime;

  set<team *, teamFitnessLexicalCompare> teams;
  int numOldDeleted = 0;
  int numDeleted = 0;

  deque<program *> programsWithNoRefs;

  vector<long> deletedIds;
  for (auto teiter = _Mroot.begin(); teiter != _Mroot.end();) {
    if (!(*teiter)->elite(GetState("phase")) &&
        !isElitePS(*teiter, GetState("phase"))) {
      if ((*teiter)->gtime_ < t) numOldDeleted++;
      _phyloGraph[(*teiter)->id_].dtime = t;
      (*teiter)->cleanup(_teamMap, programsWithNoRefs);
      removeTeam(*teiter, false);
      deletedIds.push_back((*teiter)->id_);
      // cerr << "del " << (*teiter)->id_ << endl;
      delete *teiter;
      teiter = _Mroot.erase(teiter);
      numDeleted++;
    } else
      teiter++;
  }

  sort(deletedIds.begin(), deletedIds.end());
  sort(_Mids.begin(), _Mids.end());
  vector<long> diff;
  set_difference(_Mids.begin(), _Mids.end(), deletedIds.begin(),
                 deletedIds.end(), inserter(diff, diff.begin()));
  _Mids = diff;

  cleanupProgramsWithNoRefs(t, programsWithNoRefs, false);

  memoryEigen *m;
  for (size_t memType = 0; memType < memoryEigen::NUM_MEMORY_TYPES; memType++) {
    while (_Memory[memType].size() < _M.size()) {
      m = new memoryEigen(state_["memory_eigen_count"]++, memType, params_);
      addMemory(m);
    }
  }

  oss << "selTms t " << t << " Msz " << _M.size() << " Lsz " << _L.size()
      << " mrSz " << _Mroot.size() << " mSz";
  for (size_t memType = 0; memType < memoryEigen::NUM_MEMORY_TYPES; memType++)
    oss << " " << _Memory[memType].size();
  oss << " eLSz " << _numEliteTeamsCurrent[GetState("phase")] << " nDel "
      << numDeleted << " nOldDel " << numOldDeleted << " nOldDelPr "
      << (double)numOldDeleted / numDeleted;
  oss << endl;
}

/******************************************************************************/
void TPG::cleanupProgramsWithNoRefs(long t,
                                    deque<program *> &programsWithNoRefs,
                                    bool updateLidsImmediately) {
  vector<long> deletedIds;
  while (programsWithNoRefs.size() > 0) {
    auto leiter = programsWithNoRefs.front();
    if (leiter->refs() != 0) {
      programsWithNoRefs.pop_front();
      continue;
    }
    if (leiter->action() >= 0) {
      if (_teamMap[leiter->action()]->inDeg() == 1) {
        _Mroot.insert(_teamMap[leiter->action()]);
        _phyloGraph[_teamMap[leiter->action()]->id_].root = true;
      }
      _teamMap[leiter->action()]->removeIncomingProgram(leiter->id_);
      // if team was a subsumed root clone that has now become a root itself,
      // just delete it
      if (_teamMap[leiter->action()]->root() &&
          _teamMap[leiter->action()]->cloneId_ != -1) {
        auto it = _teamMap.find(_teamMap[leiter->action()]->cloneId_);
        if (it != _teamMap.end()) it->second->clones_ = it->second->clones_ - 1;
        team *tm = _teamMap[leiter->action()];
        _phyloGraph[tm->id_].dtime = t;
        _Mroot.erase(tm);
        removeTeam(tm, true);
        tm->cleanup(_teamMap, programsWithNoRefs);
        // cerr << "del " << tm->id_ << endl;
        delete tm;
      }
    }
    for (size_t memType = 0; memType < memoryEigen::NUM_MEMORY_TYPES;
         memType++) {
      leiter->memGet(memType)->refDec();
      if (leiter->memGet(memType)->refs() == 0) {
        removeMemory(leiter->memGet(memType));
        delete leiter->memGet(memType);
      }
    }
    removeProgram(leiter, updateLidsImmediately);
    if (!updateLidsImmediately) deletedIds.push_back(leiter->id_);
    delete leiter;
    programsWithNoRefs.pop_front();
  }
  if (!updateLidsImmediately) {
    sort(deletedIds.begin(), deletedIds.end());
    sort(_Lids.begin(), _Lids.end());
    vector<long> diff;
    set_difference(_Lids.begin(), _Lids.end(), deletedIds.begin(),
                   deletedIds.end(), inserter(diff, diff.begin()));
    _Lids = diff;
  }
}

/******************************************************************************/
void TPG::setParams() {
  ReadParameters("parameters.txt", params_);
  _numStoredOutcomesPerHost[_TRAIN_PHASE] =
      GetParam<int>("n_stored_outcomes_TRAIN");
  _numStoredOutcomesPerHost[_VALIDATION_PHASE] =
      GetParam<int>("n_stored_outcomes_VALIDATION");
  _numStoredOutcomesPerHost[_TEST_PHASE] =
      GetParam<int>("n_stored_outcomes_TEST");
  params_["n_task"] = to_string(GetState("active_task")).length();
}

/******************************************************************************/
void TPG::teamTaskRank(int phase, const vector<int> &objectives) {
  oss << "TPG::teamTaskRank <team:avgRank>";
  for (auto teiterA = _M.begin(); teiterA != _M.end(); teiterA++) {
    if (!(*teiterA)->root()) continue;
    vector<int> ranks;
    for (size_t i = 0; i < objectives.size(); i++) ranks.push_back(1);
    for (size_t o = 0; o < objectives.size(); o++) {
      for (auto teiterB = _M.begin(); teiterB != _M.end(); teiterB++) {
        if (!(*teiterB)->root()) continue;
        if ((*teiterB)->getMeanOutcome(
                phase, objectives[o], GetParam<int>("fit_mode"), false, false) >
            (*teiterA)->getMeanOutcome(phase, objectives[o],
                                       GetParam<int>("fit_mode"), false, false))
          ranks[o]++;
      }
    }

    double rankSum = 0;
    for (size_t i = 0; i < ranks.size(); i++) rankSum += ranks[i];
    (*teiterA)->fit_ = 1 / (rankSum / ranks.size());
    oss << " " << (*teiterA)->id_ << ":" << (*teiterA)->fit_;
  }
  oss << endl;
}

/******************************************************************************/
void TPG::updateMODESFilters(bool roots) {
  (void)roots;
  vector<long> symbiontIntersection;
  symbiontIntersection.reserve(100);
  vector<long> symbiontUnion;
  symbiontUnion.reserve(100);

  if (GetState("t_current") != GetParam<int>("t_start")) {
    _persistenceFilterA.clear();
    for (auto teiter = _Mroot.begin(); teiter != _Mroot.end(); teiter++) {
      vector<long> ancestorIds;
      (*teiter)->getAncestorIds(ancestorIds);
      for (auto pr = _allComponentsA.begin(); pr != _allComponentsA.end(); pr++)
        if (find(ancestorIds.begin(), ancestorIds.end(), (*pr).first) !=
                ancestorIds.end() &&
            (*teiter)->hasOutcome(0, _TRAIN_PHASE, 0)) {
          // found an ancestor of *teiter in _allComponentsA, add *teiter to
          // _persistenceFilterA
          _persistenceFilterA.insert(
              pair<long, modesRecord>((*teiter)->id_, modesRecord()));
          // store active programs for novelty metric
          set<team *, teamIdComp> teams;
          set<program *, programIdComp> programs;
          set<memoryEigen *, memoryEigenIdComp> memories;
          (*teiter)->getAllNodes(_teamMap, teams, programs, memories, true);
          for (auto leiter = programs.begin(); leiter != programs.end();
               leiter++) {
            _persistenceFilterA[(*teiter)->id_].activeProgramIds.insert(
                (*leiter)->id_);
            _persistenceFilterA[(*teiter)->id_].effectiveInstructionsTotal +=
                (*leiter)->esize();
          }
          for (auto teiter2 = teams.begin(); teiter2 != teams.end(); teiter2++)
            _persistenceFilterA[(*teiter)->id_].activeTeamIds.insert(
                (*teiter2)->id_);
          _persistenceFilterA[(*teiter)->id_].runTimeComplexityIns =
              (*teiter)->runTimeComplexityIns();
          _persistenceFilterA[(*teiter)->id_].behaviourString =
              (*teiter)->getBehaviourString(0, _TRAIN_PHASE);
          break;
        }
    }

    // do analysis here
    int change = 0;
    // double minNCD = 0;
    double novelty = 0;
    int complexityRTC = 0;
    int complexityTeams = 0;
    int complexityPrograms = 0;
    int complexityInstructions = 0;
    double ecology = 0;
    for (auto prA = _persistenceFilterA.begin();
         prA != _persistenceFilterA.end(); prA++) {
      // change
      if (_persistenceFilterA_t.find((*prA).first) ==
          _persistenceFilterA_t.end())
        change++;
      // novelty
      bool found = false;
      for (auto prAll = _persistenceFilterAllTime.begin();
           prAll != _persistenceFilterAllTime.end(); prAll++) {
        // double ncd =
        // normalizedCompressionDistance((*prAll).second.behaviourString,
        // (*prA).second.behaviourString);
        if ((*prAll).second.behaviourString.compare(
                (*prA).second.behaviourString) == 0) {
          found = true;
          break;
        }
      }
      if (!found) novelty++;

      // complexity
      complexityRTC =
          max(complexityRTC, (int)(*prA).second.runTimeComplexityIns);
      complexityTeams =
          max(complexityTeams,
              (int)(*prA).second.activeTeamIds.size());  // transitions ?
      complexityPrograms =
          max(complexityPrograms, (int)(*prA).second.activeProgramIds.size());
      complexityInstructions =
          max(complexityInstructions,
              (int)(*prA).second.effectiveInstructionsTotal);

      // ecology
      double Pc = 0;
      for (auto prA2 = _persistenceFilterA.begin();
           prA2 != _persistenceFilterA.end(); prA2++)
        if ((*prA2).first != (*prA).first &&
            (*prA2).second.behaviourString.compare(
                (*prA).second.behaviourString) == 0)
          Pc++;
      if (Pc > 0) {
        Pc = Pc / _persistenceFilterA.size();
        ecology += Pc * log2(Pc);
      }
    }
    ecology *= -1;

    oss << "TPG::MODES t " << GetState("t_current") << " pfASize "
        << _persistenceFilterA.size() << " pfA_tSize "
        << _persistenceFilterA_t.size();
    oss << " change " << change << " novelty " << novelty;
    oss << " complexityRTC " << complexityRTC << " complexityTeams "
        << complexityTeams << " complexityPrograms " << complexityPrograms;
    oss << " complexityInstructions " << complexityInstructions << " ecology "
        << ecology << endl;

    _persistenceFilterA_t.clear();
    _persistenceFilterA_t.insert(_persistenceFilterA.begin(),
                                 _persistenceFilterA.end());
    _persistenceFilterAllTime.insert(_persistenceFilterA.begin(),
                                     _persistenceFilterA.end());
  }

  // for next t
  _allComponentsA.clear();
  // for(auto teiter = _M.begin(); teiter != _M.end(); teiter++)
  //    if (!roots || (roots && (*teiter)->root()))
  for (auto teiter = _Mroot.begin(); teiter != _Mroot.end(); teiter++)
    _allComponentsA.insert(
        pair<long, modesRecord>((*teiter)->id_, modesRecord()));
}

/******************************************************************************/
void TPG::writeCheckpoint(long t, bool elite) {
  ofstream ofs;
  char filename[80];
  sprintf(filename, "%s/%s.%ld.%d.%d.%d.rslt", "checkpoints", "cp", t,
          GetParam<int>("id"), _seeds[TPG_SEED_INDEX], GetState("phase"));

  if (fileExists(filename)) {
    if (remove(filename) != 0) cerr << "error deleting " << filename << endl;
  }
  ofs.open(filename, ios::out);
  if (!ofs.is_open() || ofs.fail()) {
    cerr << "open failed for file: " << filename << " error:" << strerror(errno)
         << '\n';
    die(__FILE__, __FUNCTION__, __LINE__, "Can't open file.");
  }

  ofs << "seed_tpg:" << _seeds[TPG_SEED_INDEX] << endl;
  ofs << "seed_aux:" << _seeds[AUX_SEED_INDEX] << endl;
  ofs << "t:" << GetState("t_current") << endl;
  ofs << "active_task:" << GetState("active_task") << endl;
  ofs << "fitMode:" << GetParam<int>("fit_mode") << endl;

  if (elite) {
    set<program *, programIdComp> programs;
    set<team *, teamIdComp> teams, teamsAll;
    set<memoryEigen *, memoryEigenIdComp> memories;

    for (auto itr1 = _eliteTeamPS.begin(); itr1 != _eliteTeamPS.end();
         itr1++)  // taskset
      for (auto itr2 = itr1->second.begin(); itr2 != itr1->second.end();
           itr2++)  // fitmode
      // for (auto itr3 = itr2->second.begin(); itr3 != itr2->second.end();
      //  itr3++) // phase
      {
        auto tm = itr2->second[2];
        teams.clear();
        // itr3->second->getAllNodes(_teamMap, teams, programs, memories,
        // false);
        tm->getAllNodes(_teamMap, teams, programs, memories, false);
        teamsAll.insert(teams.begin(), teams.end());
      }

    for (auto meiter = memories.begin(); meiter != memories.end(); meiter++)
      ofs << (*meiter)->checkpoint();
    for (auto leiter = programs.begin(); leiter != programs.end(); leiter++)
      ofs << (*leiter)->checkpoint(true);  // all instructions
    for (auto teiter = teamsAll.begin(); teiter != teamsAll.end(); teiter++)
      ofs << (*teiter)->checkpoint(false);
    for (auto teiter = teamsAll.begin(); teiter != teamsAll.end(); teiter++)
      ofs << (*teiter)->checkpoint(true);
  } else {
    for (size_t memType = 0; memType < memoryEigen::NUM_MEMORY_TYPES; memType++)
      for (auto meiter = _Memory[memType].begin();
           meiter != _Memory[memType].end(); meiter++)
        ofs << meiter->second->checkpoint();
    for (auto leiter = _L.begin(); leiter != _L.end(); leiter++)
      ofs << leiter->second->checkpoint(true);  // all instructions
    for (auto teiter = _M.begin(); teiter != _M.end(); teiter++)
      ofs << (*teiter)->checkpoint(false);
    for (auto teiter = _M.begin(); teiter != _M.end(); teiter++)
      ofs << (*teiter)->checkpoint(true);

    ofs << "phyloNode:id:gtime:dtime:fitness:root" << endl;
    ofs << "phyloLink:from,to" << endl;
    for (auto it = _phyloGraph.begin(); it != _phyloGraph.end(); it++) {
      ofs << "phyloNode:" << (*it).first << ":" << (*it).second.gtime << ":"
          << (*it).second.dtime << ":" << (*it).second.fitnessBin << ":"
          << (*it).second.fitness << ":" << (*it).second.root << endl;
      if ((*it).second.adj.size() > 0)
        for (size_t i = 0; i < (*it).second.adj.size(); i++)
          ofs << "phyloLink:" << (*it).first << ":" << (*it).second.adj[i]
              << endl;
      if ((*it).second.ancestorIds.size() > 0) {
        ofs << "ancestorIds:" << (*it).first;
        for (auto it2 = (*it).second.ancestorIds.begin();
             it2 != (*it).second.ancestorIds.end(); it2++)
          ofs << ":" << *it2;
        ofs << endl;
      }
    }
  }

  ofs << "end" << endl;
  ofs.close();

  // if (compress){
  //    //compress this checkpoint
  //    oss.str("");
  //    oss << "tar czpf " << filename << ".tgz -C  checkpoints cp." << t << "."
  //    << _id << "." << _seed << "." << GetState("phase") << ".rslt"; if
  //    (system (oss.str().c_str()) != 0)
  //       cerr << "error with system call" << endl;
  //    if( remove(filename) != 0 )
  //       cerr << "error deleting " << filename << endl;
  //    oss.str("");
  // }
  ////delete previous (compressed) checkpoint
  // if (previousT >= 0){
  //    sprintf(filename, "%s/%s.%ld.%d.%d.%d.rslt%s", "checkpoints", "cp",
  //    previousT, _id,_seed, GetState("phase"), compress ? ".tgz": ""); if
  //    (ifstream(filename))
  //       if( remove(filename) != 0 )
  //          cerr << "error deleting " << filename << endl;
  // }

  oss << "TPG::writeCheckpoint "
      << " Msize " << _M.size() << " Lsize " << _L.size() << " MrooSize "
      << _Mroot.size() << endl;
}

/******************************************************************************/
void TPG::writeCheckpoint(string &s, vector<team *> &rootTeams) {
  set<program *, programIdComp> programs;
  set<team *, teamIdComp> teams;
  set<memoryEigen *, memoryEigenIdComp> memories;

  for (auto teiter = rootTeams.begin(); teiter != rootTeams.end(); teiter++)
    (*teiter)->getAllNodes(_teamMap, teams, programs, memories, false);

  stringstream ss;

  ss << "seed_tpg:" << _seeds[TPG_SEED_INDEX] << endl;
  ss << "seed_aux:" << _seeds[AUX_SEED_INDEX] << endl;
  ss << "t:" << GetState("t_current") << endl;
  ss << "active_task:" << GetState("active_task") << endl;
  ss << "internalTestNodeId:-1" << endl;
  ss << "fitMode:" << GetParam<int>("fit_mode") << endl;
  ss << "phase:" << GetState("phase") << endl;
  for (auto meiter = memories.begin(); meiter != memories.end(); meiter++)
    ss << (*meiter)->checkpoint();
  for (auto leiter = programs.begin(); leiter != programs.end(); leiter++)
    ss << (*leiter)->checkpoint(true);  // all instructions
  for (auto teiter = teams.begin(); teiter != teams.end(); teiter++)
    ss << (*teiter)->checkpoint(false);
  // ss << "miniBatchIndexes";
  // for (size_t i = 0; i < _miniBatchIndexes.size(); i++)
  //    ss << ":" << _miniBatchIndexes[i];
  ss << endl;

  s = ss.str();
}

/******************************************************************************/
void TPG::writeCheckpoint(string &s, vector<team *> &rootTeams,
                          vector<teamPair> &teamPairs) {
  set<program *, programIdComp> programs;
  set<team *, teamIdComp> teams;
  set<memoryEigen *, memoryEigenIdComp> memories;

  for (auto teiter = rootTeams.begin(); teiter != rootTeams.end(); teiter++)
    (*teiter)->getAllNodes(_teamMap, teams, programs, memories, false);

  stringstream ss;

  ss << "seed_tpg:" << _seeds[TPG_SEED_INDEX] << endl;
  ss << "seed_aux:" << _seeds[AUX_SEED_INDEX] << endl;
  ss << "t:" << GetState("t_current") << endl;
  ss << "active_task:" << GetState("active_task") << endl;
  ss << "internalTestNodeId:-1" << endl;
  ss << "fitMode:" << GetParam<int>("fit_mode") << endl;
  ss << "phase:" << GetState("phase") << endl;
  for (auto meiter = memories.begin(); meiter != memories.end(); meiter++)
    ss << (*meiter)->checkpoint();
  for (auto leiter = programs.begin(); leiter != programs.end(); leiter++)
    ss << (*leiter)->checkpoint(true);  // all instructions
  for (auto teiter = teams.begin(); teiter != teams.end(); teiter++)
    ss << (*teiter)->checkpoint(false);
  // this has to come after all teams
  for (auto it1 = teamPairs.begin(); it1 != teamPairs.end(); it1++) {
    ss << "teamPair";
    for (auto it2 = (*it1).teams.begin(); it2 != (*it1).teams.end(); it2++)
      ss << ":" << (*it2)->id_;
    ss << endl;
  }
  ss << endl;

  s = ss.str();
}

/******************************************************************************/
void TPG::writeCheckpoint(string &s, vector<team *> &rootTeams,
                          team *originalTeam, team *replacementTeam) {
  set<program *, programIdComp> programs;
  set<team *, teamIdComp> teams;
  set<memoryEigen *, memoryEigenIdComp> memories;

  for (auto teiter = rootTeams.begin(); teiter != rootTeams.end(); teiter++)
    (*teiter)->getAllNodes(_teamMap, teams, programs, memories, false);
  originalTeam->getAllNodes(_teamMap, teams, programs, memories, false);
  replacementTeam->getAllNodes(_teamMap, teams, programs, memories, false);

  stringstream ss;

  ss << "seed_tpg:" << _seeds[TPG_SEED_INDEX] << endl;
  ss << "seed_aux:" << _seeds[AUX_SEED_INDEX] << endl;
  ss << "t:" << GetState("t_current") << endl;
  ss << "active_task:" << GetState("active_task") << endl;
  ss << "internalTestNodeId:" << state_["internal_test_node_id"] << endl;
  ss << "fitMode:" << GetParam<int>("fit_mode") << endl;
  ss << "phase:" << GetState("phase") << endl;
  for (auto meiter = memories.begin(); meiter != memories.end(); meiter++)
    ss << (*meiter)->checkpoint();
  for (auto leiter = programs.begin(); leiter != programs.end(); leiter++)
    ss << (*leiter)->checkpoint(true);  // all instructions
  for (auto teiter = teams.begin(); teiter != teams.end(); teiter++) {
    if ((*teiter)->id_ == originalTeam->id_ &&
        originalTeam->id_ != replacementTeam->id_)
      ss << replacementTeam->checkpoint(false, originalTeam->id_);
    else
      ss << (*teiter)->checkpoint(false);
  }
  ss << endl;

  s = ss.str();
}
