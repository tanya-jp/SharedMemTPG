#include <TPG.h>
#include <acrobot.h>
#include <cartCentering.h>
#include <cartPole.h>
#include <classicRLEnv.h>
#include <mountainCar.h>
#include <mountainCarContinuous.h>
#include <pendulum.h>

#include <boost/mpi.hpp>
#include <chrono>
#include <thread>

#if !defined(CCANADA) && !defined(HPCC)
#include <GL/gl.h>
#include <GL/glut.h>
#endif

#define REWARD_IDX 0
#define VISITED_TEAMS_IDX 1
#define INSTRUCTIONS_IDX 2
#define MEMBERS_RUN_ENTROPY_IDX 3

namespace mpi = boost::mpi;

vector<team *> GetTeamsToEval(TPG &tpg) {
  auto root_teams = tpg.GetTeams(true);
  vector<team *> teams;
  for (auto it : root_teams) {
    it.second->numEval_ = tpg._numStoredOutcomesPerHost[tpg.GetState("phase")] -
                          it.second->numOutcomes(tpg.GetState("phase"),
                                                 tpg.GetState("active_task"));
    if (it.second->numEval_ > 0) {
      teams.push_back(it.second);
    }
  }
  return teams;
}

void AssignTeamsToEvaluators(TPG &tpg, mpi::communicator &world,
                             vector<team *> &teams_to_eval,
                             int world_size_per_task, int& evaluator) {
  auto teams_per_evaluator = teams_to_eval.size() / world_size_per_task;
  auto remainder = teams_to_eval.size() % world_size_per_task;
  vector<team *> teams;
  for (auto it = teams_to_eval.begin(); it != teams_to_eval.end(); it++) {
    teams.push_back(*it);
    if ((remainder > 0 && teams.size() == teams_per_evaluator + 1) ||
        (remainder == 0 && teams.size() == teams_per_evaluator) ||
        next(it) == teams_to_eval.end()) {
      string s = "";
      tpg.writeCheckpoint(s, teams);
      world.send(evaluator, 0, s);
      evaluator++;
      teams.clear();
      if (remainder > 0) remainder--;
    }
  }
}

/*
 * 1. Assign agents to evaluator procs
 * 2. Wait for evals to finish
 * 3. Collect results
 */
void evaluate_main(TPG &tpg, ostringstream &os, mpi::communicator &world,
                   vector<classicRLEnv *> &tasks, vector<int> &taskSet) {
  (void)tasks;
  (void)os;
  string my_string = "MAIN";
  vector<team *> teams_this_eval;
  vector<string> all_strings;
  vector<string> splitStr;
  string resultLine;

  int world_size_per_task = (world.size() - 1) / taskSet.size();
  // assign agents to evaluators
  int evaluator = 1;
  for (size_t task = 0; task < taskSet.size(); task++) {
    tpg.state_["active_task"] = taskSet[task];
    auto teams_to_eval = GetTeamsToEval(tpg);
    AssignTeamsToEvaluators(tpg, world, teams_to_eval, world_size_per_task, evaluator);
  }
  // let the rest of the procs know they are not needed this round
  while (evaluator <= (world.size() - 1)) {
    world.send(evaluator++, 0, "x");
  }

  // collect evaluation result from each evaluator
  auto root_teams = tpg.GetTeams(true);
  all_strings.clear();
  gather(world, my_string, all_strings, 0);

  for (int proc = 1; proc < world.size(); proc++) {
    if (!all_strings[proc].empty()) {
      istringstream f(all_strings[proc]);
      while (getline(f, resultLine)) {
        vector<long> active;
        vector<double> r_runTimeStats;
        vector<int> r_runTimeInts;
        splitString(resultLine, ':', splitStr);
        size_t s = 0;
        long rslt_id = atol(splitStr[s++].c_str());
        string behavSeq = splitStr[s++].c_str();
        tpg.params_["active_task"] = atoi(splitStr[s++].c_str());
        for (int i = 0; i < tpg.GetParam<int>("n_point_aux_double"); i++)
          r_runTimeStats.push_back(atof(splitStr[s++].c_str()));
        for (int i = 0; i < tpg.GetParam<int>("n_point_aux_int"); i++)
          r_runTimeInts.push_back(atoi(splitStr[s++].c_str()));
        while (s < splitStr.size())
          active.push_back(atol(splitStr[s++].c_str()));
        root_teams[rslt_id]->updateActiveMembersFromIds(active);
        tpg.setOutcome(root_teams[rslt_id], behavSeq, r_runTimeStats,
                       r_runTimeInts, tpg.GetState("t_current"));              
      }
    }
  }
}

/*******************************************************************************
 * Assign program graphs to evaluator procs, wait for evals to finish, collect
 * eval results This version is used for internal node replacement
 */
void evaluate_main_internals(TPG &tpg, ostringstream &os,
                             mpi::communicator &world,
                             vector<classicRLEnv *> &tasks,
                             vector<int> &taskSet, team *internalTeam,
                             set<team *, teamIdComp> &internalReplacements,
                             map<long, team *> &associatedRoots) {
  (void)tasks;
  string my_string = "MAIN";
  map<long, team *> root_teams = associatedRoots;
  vector<team *> teamsToEval;
  vector<team *> teams_this_eval;
  vector<string> all_strings;
  vector<string> splitStr;
  string resultLine;
  string s;
  vector<long> active;
  vector<double> r_runTimeStats;
  vector<int> r_runTimeInts;

  size_t totalTeamsToEval = 0;
  int worldSizePerTask = (world.size() - 1) / taskSet.size();
  for (auto internaliter = internalReplacements.begin();
       internaliter != internalReplacements.end(); internaliter++) {
    int evaluator = 1;
    size_t numTeamsToEval = 0;
    tpg.state_["internal_test_node_id"] = (*internaliter)->id_;
    for (size_t task = 0; task < taskSet.size(); task++) {
      tpg.state_["active_task"] = taskSet[task];
      // assign policies to evaluators
      teamsToEval.clear();
      teams_this_eval.clear();
      for (auto it = associatedRoots.begin(); it != associatedRoots.end();
           it++) {
        it->second->numEval_ = tpg.GetParam<int>("n_stored_outcomes_TRAIN");
        teamsToEval.push_back(it->second);
        numTeamsToEval++;
      }
      size_t teamsPerEvaluator = teamsToEval.size() / worldSizePerTask;
      size_t remainder = teamsToEval.size() % worldSizePerTask;

      for (auto it = teamsToEval.begin(); it != teamsToEval.end(); it++) {
        teams_this_eval.push_back(*it);
        if ((remainder > 0 &&
             teams_this_eval.size() == teamsPerEvaluator + 1) ||
            (remainder == 0 && teams_this_eval.size() == teamsPerEvaluator) ||
            next(it) == teamsToEval.end()) {
          s = "";
          tpg.writeCheckpoint(s, teams_this_eval, internalTeam,
                              *internaliter);  // with replacement
          world.send(evaluator++, 0, s);
          teams_this_eval.clear();
          if (remainder > 0) remainder--;
        }
      }
    }
    totalTeamsToEval += numTeamsToEval;
    // no more teams to evaluate, so let the rest of the procs know they are
    // not needed this round
    while (evaluator <= (world.size() - 1)) {
      s = "x";
      world.send(evaluator++, 0, "x");
    }

    // collect evaluation result from each evaluator
    all_strings.clear();
    gather(world, my_string, all_strings, 0);
    for (int proc = 1; proc < world.size(); proc++) {
      if (!all_strings[proc].empty()) {
        istringstream f(all_strings[proc]);
        while (getline(f, resultLine)) {
          splitString(resultLine, ':', splitStr);
          size_t s = 0;
          long rslt_id = atol(splitStr[s++].c_str());
          string behavSeq = splitStr[s++].c_str();
          tpg.params_["active_task"] = atoi(splitStr[s++].c_str());
          for (int i = 0; i < tpg.GetParam<int>("n_point_aux_double"); i++)
            r_runTimeStats.push_back(atof(splitStr[s++].c_str()));
          for (int i = 0; i < tpg.GetParam<int>("n_point_aux_int"); i++)
            r_runTimeInts.push_back(atoi(splitStr[s++].c_str()));
          while (s < splitStr.size())
            active.push_back(atol(splitStr[s++].c_str()));
          root_teams[rslt_id]->updateActiveMembersFromIds(active);
          tpg.setOutcome(root_teams[rslt_id], behavSeq, r_runTimeStats,
                         r_runTimeInts, tpg.GetState("t_current"));
          active.clear();
          r_runTimeStats.clear();
          r_runTimeInts.clear();
        }
      }
    }
  }
  os << "t " << tpg.GetState("t_current") << " phs " << tpg.GetState("phase")
     << " tToEval " << totalTeamsToEval << endl;
}

/*******************************************************************************
 * Receive program graphs from main MPI proc, eval in environment, return
 * results
 */
void evaluate_sub(TPG &tpg, mpi::communicator &world,
                  vector<classicRLEnv *> &tasks) {
  //  (void)tasks;
  classicRLEnv *game = tasks[0];  // gets updated prior to eval below
  state *worldState = new state(tpg.GetParam<int>("n_input"));
  vector<double> runTimeStats;
  runTimeStats.reserve(tpg.GetParam<int>("n_point_aux_double"));
  runTimeStats.resize(tpg.GetParam<int>("n_point_aux_double"));
  vector<int> runTimeInts;
  runTimeInts.reserve(tpg.GetParam<int>("n_point_aux_int"));
  runTimeInts.resize(tpg.GetParam<int>("n_point_aux_int"));
  vector<int> behavSeq;

  vector<team *> teamPath;
  long decisionInstructions;
  set<team *, teamIdComp> visitedTeams;
  vector<team *> teams;
  string evalResult = "";
  string checkpointString = "";
  bool partially_observable = tpg.GetParam<int>("partially_observable");
  while (checkpointString.compare("done") != 0) {
    world.recv(0, 0, checkpointString);
    evalResult = "";
    set<program *, programIdComp> active;
    if (checkpointString.compare("x") != 0 &&
        checkpointString.compare("done") != 0) {
      tpg.readCheckpoint(-1, _TRAIN_PHASE, -1, true, checkpointString);

      tpg.getTeams(teams, true);
      game = tasks[tpg.GetState("active_task")];
      behavSeq.reserve(game->maxStep() +
                       game->maxStep() * tpg.GetParam<int>("n_input"));
      map<long, team *> teamMap;
      tpg.teamMap(teamMap);
      // cout << "DEBUG " << teams.size() << endl;
      for (size_t i = 0; i < teams.size(); i++) {
        program *atomicProgram = tpg._L.begin()->second;
        tpg.markEffectiveCode(teams[i]);
        // teams[i]->clearMembersRunTally();

        int eStart =
            0;  // teams[i]->numOutcomes(tpg.phase(),tpg.GetState("active_task"));
        if (tpg.GetState("phase") == _TEST_PHASE)
          eStart = tpg.GetParam<int>("n_stored_outcomes_TRAIN");
        else if (tpg.GetState("phase") == _VALIDATION_PHASE)
          eStart = tpg.GetState("internal_test_node_id");  // double check this
        for (size_t e = 0; e < teams[i]->numEval_; e++) {
          // if (tpg._seeds[AUX_SEED_INDEX] < 1)
          tpg._rngs[AUX_SEED_INDEX].seed(e + eStart);
          fill(runTimeStats.begin(), runTimeStats.end(), 0);
          teams[i]->clearMemory(teamMap);
          behavSeq.clear();
          game->reset(tpg._rngs[AUX_SEED_INDEX]);
          worldState->setState(game->getStateVec(partially_observable));
          while (!game->terminal()) {
            atomicProgram = tpg.getAction(
                teams[i], worldState, true, visitedTeams, decisionInstructions,
                game->getStep(), teamPath, tpg._rngs[AUX_SEED_INDEX]);

            if (game->discreteActions())
              behavSeq.push_back(atomicProgram->action());
            else
              behavSeq.push_back(
                  -1 -
                  discretize(
                      bound((atomicProgram->memGet(memoryEigen::SCALAR_TYPE))
                                ->mem_[0](0, 0),
                            game->minActionContinuous(),
                            game->maxActionContinuous()),
                      game->minActionContinuous(), game->maxActionContinuous(),
                      3));
            behavSeq.push_back(
                discretize(worldState->getStateVarDouble(0), 0, 1, 3));
            behavSeq.push_back(
                discretize(worldState->getStateVarDouble(1), 0, 1, 3));

            runTimeStats[REWARD_IDX] += game->update(
                (atomicProgram->action() * -1) - 1,
                (atomicProgram->memGet(memoryEigen::SCALAR_TYPE))
                    ->mem_[0](0, 0),
                tpg._rngs[AUX_SEED_INDEX]);  // atomic actions are represented
                                             // as negatives in tpg
            worldState->setState(game->getStateVec(partially_observable));

            runTimeStats[VISITED_TEAMS_IDX] += visitedTeams.size();
            runTimeStats[INSTRUCTIONS_IDX] += decisionInstructions;
          }
          runTimeStats[VISITED_TEAMS_IDX] =
              runTimeStats[VISITED_TEAMS_IDX] / game->getStep();
          runTimeStats[INSTRUCTIONS_IDX] =
              runTimeStats[INSTRUCTIONS_IDX] / game->getStep();
          runTimeStats[MEMBERS_RUN_ENTROPY_IDX] =
              0;  // teams[i]->membersRunEntropy(); this might be broken
          runTimeInts[POINT_AUX_INT_TASK] = tpg.GetState("active_task");
          runTimeInts[POINT_AUX_INT_PHASE] = tpg.GetState("phase");
          runTimeInts[POINT_AUX_INT_ENVSEED] = e + eStart;
          runTimeInts[POINT_AUX_INT_internalTestNodeId] =
              tpg.GetState("internal_test_node_id");

          if (isnan(runTimeStats[REWARD_IDX])) {
            cerr << "WTF " << (atomicProgram->action() * -1) - 1 << " "
                 << (atomicProgram->memGet(memoryEigen::SCALAR_TYPE))
                        ->mem_[0](0, 0)
                 << endl;
            die(__FILE__, __FUNCTION__, __LINE__, "WTF nana");
          }
          evalResult += to_string(teams[i]->id_);
          evalResult += ":4" + vecToStrNoSpace(behavSeq);
          evalResult += ":" + to_string(tpg.GetState("active_task"));
          for (size_t r = 0; r < runTimeStats.size(); r++)
            evalResult += ":" + to_string(runTimeStats[r]);
          for (size_t r = 0; r < runTimeInts.size(); r++)
            evalResult += ":" + to_string(runTimeInts[r]);
          teams[i]->getActiveMembersByRef(active);
          for (auto leiter = active.begin(); leiter != active.end(); leiter++)
            evalResult += ":" + to_string((*leiter)->id_);
          evalResult += "\n";
        }
      }
      tpg.finalize();  ////////////////////////////////////////////
    }
    if (tpg.GetParam<int>("replay")) break;
    gather(world, evalResult, 0);
  }
  delete worldState;
  tpg.finalize();  //////////////////////////////////////////////////////
}

/*******************************************************************************
 * Receive program graphs from main MPI proc, eval in environment, return
 * results This version is used for animations and logging of many stats
 * during replay
 */
void evaluate_sub_replay(TPG &tpg, mpi::communicator &world,
                         vector<classicRLEnv *> &tasks,
                         set<team *, teamIdComp> &visitedTeamsAll,
                         set<team *, teamIdComp> &visitedTeamsAllTasks,
                         map<long, double> &teamUseMap,
                         vector<map<long, double>> &teamUseMapPerTask) {
  classicRLEnv *game = tasks[0];  // gets updated prior to eval below
  size_t saveFrame = 0;
#if !defined(CCANADA) && !defined(HPCC)
  /* opengl setup for visual mode
   * *********************************************/
  if (tpg.GetParam<int>("visual") == 1) {///////////////////////////////////////////////////////////////////////
    double _width = 1200;
    double _height = 1200;
    int argc = 1;
    char *argv[1] = {(char *)"null"};
    glutInit(&argc, argv);
    // glutInitDisplayMode(GLUT_SINGLE |GLUT_RGB);
    glutInitWindowSize(_width, _height);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("classicRL");
    // glutHideWindow();
    glScalef(0.5, 0.5, 0.0);
  }
  /****************************************************************************/
#endif

  state *worldState = new state(tpg.GetParam<int>("n_input"));
  vector<double> runTimeStats;
  runTimeStats.reserve(tpg.GetParam<int>("n_point_aux_double"));
  runTimeStats.resize(tpg.GetParam<int>("n_point_aux_double"));
  vector<int> runTimeInts;
  runTimeInts.reserve(tpg.GetParam<int>("n_point_aux_int"));
  runTimeInts.resize(tpg.GetParam<int>("n_point_aux_int"));
  vector<int> behavSeq;
  // for graph animations
  long totalReplaySteps = 0;
  vector<program *> allPrograms;
  vector<program *> winningPrograms;
  vector<program *> winningProgramsAll;
  // vector <team*> visitedTeamsAll;
  vector<set<long>> decisionFeatures;
  vector<set<memoryEigen *, memoryEigenIdComp>> decisionMemories;
  vector<team *> teamPath;

  long decisionInstructions;
  set<team *, teamIdComp> visitedTeams;
  vector<team *> teams;
  string evalResult = "";
  string checkpointString = "";
  bool partially_observable = tpg.GetParam<int>("partially_observable");
  while (checkpointString.compare("done") != 0) {
    // receive from p0
    if (!tpg.GetParam<int>("replay")) world.recv(0, 0, checkpointString);

    evalResult = "";
    set<program *, programIdComp> active;
    if (tpg.GetParam<int>("replay") == 1 ||
        (checkpointString.compare("x") != 0 &&
         checkpointString.compare("done") != 0)) {
      if (tpg.GetParam<int>("replay") == 0)
        tpg.readCheckpoint(-1, _TRAIN_PHASE, -1, true,
                           checkpointString);  // setRoots
      tpg.getTeams(teams, tpg.GetParam<int>("replay") == 1 ? false : true);
      game = tasks[tpg.GetState("active_task")];
      behavSeq.reserve(game->maxStep() +
                       game->maxStep() * tpg.GetParam<int>("n_input"));
      map<long, team *> teamMap;
      tpg.teamMap(teamMap);
      for (size_t i = 0; i < teams.size(); i++) {
        program *atomicProgram = tpg._L.begin()->second;
        if (tpg.GetParam<int>("replay") &&
            teams[i]->id_ != tpg.GetParam<int>("host_to_replay")) { continue; }
        tpg.markEffectiveCode(teams[i]);
        // if (tpg.replay()) teams[i]->prunePrograms();
        // teams[i]->clearMembersRunTally();

        if (tpg.GetParam<int>("replay"))
          teams[i]->numEval_ = tpg._numStoredOutcomesPerHost[_TEST_PHASE];
        int eStart = 0;
        // if (tpg.phase() == _TRAIN_PHASE || tpg.phase() == _TEST_PHASE)
        if (tpg.GetState("phase") == _TEST_PHASE &&
            !tpg.GetParam<int>("replay"))
          eStart = tpg.GetParam<int>(
              "n_stored_outcomes_TRAIN");  //(tpg.GetState("t_current") -
                                           // teams[i]->gtime()) *
                                           // tasks[tpg.GetState("active_task")]->minEval();
        else if (tpg.GetState("phase") == _VALIDATION_PHASE)
          eStart = tpg.GetState("internal_test_node_id");
        for (size_t e = 0; e < teams[i]->numEval_; e++) {
          if (tpg._seeds[AUX_SEED_INDEX] < 1)
            tpg._rngs[AUX_SEED_INDEX].seed(
                e + eStart);  // + (tpg.phase()*1000));//ensure trainand test
                              // phases have different seed set
          fill(runTimeStats.begin(), runTimeStats.end(), 0);
          teams[i]->clearMemory(teamMap);
          behavSeq.clear();
          game->reset(tpg._rngs[AUX_SEED_INDEX]);
          worldState->setState(game->getStateVec(partially_observable));

          // memoryEigen replay header with ids
          vector<double> vel;
          set<team *, teamIdComp> iTeams;
          set<program *, programIdComp> iPrograms;
          set<memoryEigen *, memoryEigenIdComp> iMemories;
          tpg.getAllNodes(teams[i], iTeams, iPrograms, iMemories);
          cout << "TPG::replay iMem sz " << iMemories.size() << " tsk "
               << tpg.GetState("active_task") << " step " << game->getStep()
               << " |";
          cout << "0 0";  // cout << vecToStr(vel);//first 2 col will be
                          // velocity
          for (auto it = iMemories.begin(); it != iMemories.end(); it++) {
            // bool active = false;
            // for (int m = 0; m < MEMORY_COLUMNS; m++)
            //    if ((*it)->getActive()[m])
            //       active = true;
            //    if(!active)
            //     continue;
            for (int m = 0; m < tpg.GetParam<int>("memory_cols"); m++) {
              // if ((*it)->getActive()[m])
              cout << " " << (*it)->id_ << "_" << m;
              // else
              //    cout << " " << 0;
            }
          }
          cout << endl;
          while (!game->terminal()) {
            double v0 = 0;
            double v1 = 0;
            double v2 = 0;
            double v3 = 0;

            if (tpg.GetParam<int>("replay") &&
                tpg.GetState("active_task") == 0 && game->getStep() > 9) {
              if (e == 0 && game->getStep() == 10)
                cout << "behaviour v0 v1 v2 v3 id actionC actionD terminal "
                        "depth"
                     << endl;
              v0 = game->getStateVar(0, false);
              v1 = game->getStateVar(1, false);
              v2 = game->getStateVar(2, false);
              v3 = game->getStateVar(3, false);
              // cout << "behaviour " << fixed << game->getStateVar(0,false)
              // << " " << game->getStateVar(1,false) << " " <<
              // game->getStateVar(2,false); if (tpg.GetParam<int>("animate"))
              //    cout << "bPendulum " << fixed <<
              //    dynamic_cast<pendulum*>(game)->theta() << " " <<
              //    dynamic_cast<pendulum*>(game)->thetaDot() << endl;
            }
            if (tpg.GetParam<int>("replay")) {
              visitedTeams.clear();
              // visitedTeams.insert(teamMap[23266]);
              // visitedTeams.insert(teamMap[200127]);
              // visitedTeams.insert(teamMap[42314]);
              atomicProgram = tpg.getAction(
                  teams[i], worldState, true, visitedTeams,
                  decisionInstructions, game->getStep(), allPrograms,
                  winningPrograms, decisionFeatures, decisionMemories, teamPath,
                  tpg._rngs[AUX_SEED_INDEX]);
              winningProgramsAll.insert(winningProgramsAll.begin(),
                                        winningPrograms.begin(),
                                        winningPrograms.end());
              visitedTeamsAll.insert(visitedTeams.begin(), visitedTeams.end());

              for (auto vt = visitedTeams.begin(); vt != visitedTeams.end();
                   vt++) {
                if (teamUseMap.find((*vt)->id_) == teamUseMap.end())
                  teamUseMap[(*vt)->id_] = 0;
                teamUseMap[(*vt)->id_]++;
              }
              // pendulum-specific
              if (tpg.GetState("active_task") == 0 && game->getStep() > 9) {
                // cout << fixed << " " << teamPath[teamPath.size()-1]->id_ <<
                // " " <<
                // bound((atomicProgram->memGet(memoryEigen::SCALAR_TYPE))->getMem()[0],-2,2)
                // << endl;
                for (size_t tp = 0; tp < teamPath.size(); tp++) {
                  cout << fixed << "behaviour " << v0 << " " << v1 << " " << v2
                       << " " << v3 << " " << teamPath[tp]->id_ << " "
                       << bound(
                              (atomicProgram->memGet(memoryEigen::SCALAR_TYPE))
                                  ->mem_[0](0, 0),
                              -2, 2)
                       << " " << (atomicProgram->action() * -1) - 1;
                  if (teamPath[tp]->id_ == teamPath[teamPath.size() - 1]->id_)
                    cout << " " << 1 << " " << tp << endl;
                  else
                    cout << " " << 0 << " " << tp << endl;
                }
              }

              // print state, velocities first
              if (game->id() == 1 || game->id() == 2)
                cout << fixed << "state step " << game->getStep() << " ss "
                     << v2 << " " << v3 << " " << v0 << " " << v1 << endl;
              else if (game->id() == 3 || game->id() == 5 || game->id() == 6)
                cout << fixed << "state step " << game->getStep() << " ss "
                     << v1 << " " << 0 << " " << v0 << " " << 0 << endl;
              else if (game->id() == 4)
                cout << fixed << "state step " << game->getStep() << " ss "
                     << dynamic_cast<pendulum *>(game)->thetaDot() << " " << 0
                     << " " << v0 << " " << 0 << endl;
            } else
              atomicProgram =
                  tpg.getAction(teams[i], worldState, true, visitedTeams,
                                decisionInstructions, game->getStep(), teamPath,
                                tpg._rngs[AUX_SEED_INDEX]);

            if (game->discreteActions())
              behavSeq.push_back(atomicProgram->action());  // discrete
            else
              behavSeq.push_back(
                  -1 -
                  discretize(
                      bound((atomicProgram->memGet(memoryEigen::SCALAR_TYPE))
                                ->mem_[0](0, 0),
                            game->minActionContinuous(),
                            game->maxActionContinuous()),
                      game->minActionContinuous(), game->maxActionContinuous(),
                      3));
            behavSeq.push_back(
                discretize(worldState->getStateVarDouble(0), 0, 1, 3));
            behavSeq.push_back(
                discretize(worldState->getStateVarDouble(0), 0, 1, 3));

            // teamUseMapPerTask[tpg.GetState("active_task")] =
            // teamUseMap;//update on the fly
            if (tpg.GetParam<int>("replay")) {
              size_t repeats = game->getStep() == 0 ? 40 : 1;
              for (size_t r = 1; r <= repeats; r++) {
                for (size_t d = teamPath.size(); d <= teamPath.size(); d++) {
                  // tpg.printGraphDot(teams[i], saveFrame, e, repeats > 1 &&
                  // r < repeats ? 0 : game->getStep(), d, allPrograms,
                  // winningPrograms, decisionFeatures, decisionMemories,
                  // teamPath, r==repeats ? true: false,
                  // visitedTeamsAllTasks);
                  tpg.printGraphDotGPEMAnimate(
                      tpg.GetParam<int>("host_to_replay"), saveFrame, e,
                      repeats > 1 && r < repeats ? 0 : game->getStep(), d,
                      allPrograms, winningPrograms, visitedTeamsAllTasks,
                      teamUseMapPerTask, teamPath);

#if !defined(CCANADA) && !defined(HPCC)
                  if (tpg.GetParam<int>("visual") == 1)
                    game->display_function(
                        e,
                        game->getStep() == 0 && r < repeats
                            ? 0
                            : (atomicProgram->action() * -1) - 1,
                        game->getStep() == 0 && r < repeats
                            ? 0
                            : (atomicProgram->memGet(memoryEigen::SCALAR_TYPE))
                                  ->mem_[0](0, 0));
#endif
                  char filename[80];
                  sprintf(filename, "%s_%05d_%03d_%05d_%05d_%05d.tga",
                          "replay/frames/gl", (int)saveFrame, (int)e,
                          game->getStep(), (int)d, 0);
#if !defined(CCANADA) && !defined(HPCC)
                  game->saveScreenshotToFile(filename, 1200, 1200);
#endif
                  saveFrame++;
                }
                this_thread::sleep_for(std::chrono::milliseconds(10));
              }
            }

            ////random actions
            // uniform_real_distribution<>disAc(-2.0, 2.0);
            // uniform_int_distribution<>disAi(0, tpg.numAtomicActions()-1);
            // double ac = disAc(rng);
            // int ai = disAi(rng);
            // runTimeStats[REWARD_IDX] += game->update(ai, ac, rng);

            ////optimal cart centering policy
            // int ai = 0;
            // double ac = 0;
            // if ((-1 * v0) > (v1 * abs(v1)))
            //   ai = 2;
            // else
            //   ai = 0;
            // runTimeStats[REWARD_IDX] += game->update(ai, ac, rng);

            runTimeStats[REWARD_IDX] += game->update(
                (atomicProgram->action() * -1) - 1,
                (atomicProgram->memGet(memoryEigen::SCALAR_TYPE))
                    ->mem_[0](0, 0),
                tpg._rngs[AUX_SEED_INDEX]);  // atomic actions are represented
                                             // as negatives in tpg
            worldState->setState(game->getStateVec(partially_observable));
            runTimeStats[VISITED_TEAMS_IDX] += visitedTeams.size();
            // runTimeStats[VISITED_TEAMS_IDX] =
            // max(runTimeStats[VISITED_TEAMS_IDX],
            // (double)visitedTeams.size());
            runTimeStats[INSTRUCTIONS_IDX] += decisionInstructions;

            ////step-wise replay stats for
            /// plotting///////////////////////////////////////////////////////////////////////////////
            // if (tpg.replay()){
            //    cout << "TPG::replay id " << teams[i]->id_;
            //    cout << " visitedTeams " << visitedTeams.size();
            //    //cout << " activeNodes";
            //    //for (auto it = visitedTeams.begin(); it !=
            //    visitedTeams.end(); it++)
            //    //   cout << " h" << (*it)->id_;
            //    //for (auto it = winningLearners.begin(); it !=
            //    winningLearners.end(); it++)
            //    //   cout << " s" << (*it)->id_;
            //    cout << " decisionInstructions " << decisionInstructions <<
            //    " decisionFeatures"; for (auto it =
            //    decisionFeatures.begin(); it
            //    != decisionFeatures.end(); it++)
            //       cout << " " << (*it).size();
            //    cout << " decisionMemories";

            //   //get all memories used this decision
            //   set < memoryEigen* > mems;
            //   for (auto it = decisionMemories.begin(); it !=
            //   decisionMemories.end(); it++)
            //      mems.insert((*it).begin(), (*it).end());

            //   //get uniq read/write times as int (ignoring graph depth)
            //   set <int> memActiveReadT;
            //   set <int> memActiveWriteT;
            //   set <int> memAges;
            //   vector <double> memReadTimes;
            //   vector <double> memWriteTimes;
            //   for (auto it = mems.begin(); it != mems.end(); it++){
            //      memReadTimes.clear();
            //      memWriteTimes.clear();
            //      for (size_t mr = 0; mr < tpg.memoryCols(); mr++){
            //         memReadTimes.push_back(-1);
            //         memWriteTimes.push_back(-1);
            //      }

            //      (*it)->getActiveReadTime(memReadTimes);
            //      (*it)->getActiveWriteTime(memWriteTimes);

            //      //for (auto rwit = memReadTimes.begin(); rwit !=
            //      memReadTimes.end(); rwit++)
            //      //   if (((int)(*rwit)) >= 0)//ignore untouched registers
            //      (-1)
            //      //      memActiveReadT.insert((int)(*rwit));
            //      //for (auto rwit = memWriteTimes.begin(); rwit !=
            //      memWriteTimes.end(); rwit++)
            //      //   if (((int)(*rwit)) >= 0)//ignore untouched registers
            //      (-1)
            //      //      memActiveWriteT.insert((int)(*rwit));

            //      for (size_t reg = 0; reg < memReadTimes.size(); reg++){
            //         //if we read a value other than default
            //         if (((int)memReadTimes[reg]) == game->getStep()-1 &&
            //         (int)memWriteTimes[reg] >= 0){
            //            memActiveWriteT.insert((int)memWriteTimes[reg]);
            //            memActiveReadT.insert((int)memReadTimes[reg]);
            //         }
            //      }
            //   }
            //   cout << " step " << game->getStep();
            //   //cout << " Dsize " << memActiveReadT.size();

            //   //vector<int> v;
            //   //for (auto it = memActiveReadT.begin(); it !=
            //   memActiveReadT.end(); it++)
            //   //   v.push_back(game->getStep() - (*it));

            //   //cout << " D " << vecToStr(v);

            //   //cout << " Dmax_med_min ";
            //   //cout << *(max_element(v.begin(), v.end()));
            //   //cout << " " << vecMedian(v);
            //   //cout << " " << *(min_element(v.begin(), v.end()));

            //   cout << " readTimes";
            //   for (auto it = memActiveReadT.begin(); it !=
            //   memActiveReadT.end(); it++)
            //      cout << " " << (*it);
            //   cout << " writeTimes";
            //   for (auto it = memActiveWriteT.begin(); it !=
            //   memActiveWriteT.end(); it++)
            //      cout << " " << (*it);
            //   //cout << " teamPath";
            //   //for(size_t i = 0; i < teamPath.size(); i++)
            //   //   cout << " h" << teamPath[i]->id_;
            //   cout << endl;

            //   //all memoryEigen

            //   //get velocities after update
            //   v0 = game->getStateVar(0,false);
            //   v1 = game->getStateVar(1,false);
            //   v2 = game->getStateVar(2,false);
            //   v3 = game->getStateVar(3,false);
            //   vel.clear();
            //   if (game->id() == 1 || game->id() == 2){
            //      vel.push_back(v2);
            //      vel.push_back(v3);
            //   }
            //   else if (game->id() == 3 || game->id() == 5 || game->id() ==
            //   6){
            //      vel.push_back(v1);
            //      vel.push_back(0);
            //   }
            //   else if (game->id() == 4){
            //      vel.push_back(dynamic_cast<pendulum*>(game)->thetaDot());
            //      vel.push_back(0);
            //   }

            //   iTeams.clear();
            //   iPrograms.clear();
            //   iMemories.clear();
            //   tpg.getAllNodes(teams[i], iTeams, iPrograms, iMemories);
            //   cout << "TPG::replay iMem sz " << iMemories.size() << " tsk "
            //   << tpg.GetState("active_task") << " step " << game->getStep()
            //   << " |"; cout << vecToStr(vel);//first 2 col will be velocity
            //   for (auto it = iMemories.begin(); it != iMemories.end();
            //   it++){
            //      if ((*it)->type() != memoryEigen::SCALAR_TYPE)
            //         continue;
            //      //bool active = false;
            //      //for (int m = 0; m < MEMORY_COLUMNS; m++)
            //      //   if ((*it)->getActive()[m])
            //      //      active = true;
            //      //   if(!active)
            //      //    continue;
            //      for (size_t m = 0; m < (*it)->indexSize(); m++){
            //         //if ((*it)->getActive()[m])
            //         cout << " " << (*it)->getMemE(m)(0,0);
            //         //else
            //         //   cout << " " << 0;
            //      }
            //   }
            //   cout << endl;

            //}//replay
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
          }
          if (tpg.GetParam<int>("animate")) {
            size_t repeats = 40;
            for (size_t r = 1; r <= repeats; r++) {
              for (size_t d = teamPath.size(); d <= teamPath.size(); d++) {
                // tpg.printGraphDot(teams[i], saveFrame, e, game->getStep(),
                // d, allPrograms, winningPrograms, decisionFeatures,
                // decisionMemories, teamPath, true, visitedTeamsAllTasks);
                tpg.printGraphDotGPEMAnimate(
                    tpg.GetParam<int>("host_to_replay"), saveFrame, e,
                    game->getStep(), d, allPrograms, winningPrograms,
                    visitedTeamsAllTasks, teamUseMapPerTask, teamPath);
#if !defined(CCANADA) && !defined(HPCC)
                if (tpg.GetParam<int>("visual") == 1)
                  game->display_function(
                      e, (atomicProgram->action() * -1) - 1,
                      (atomicProgram->memGet(memoryEigen::SCALAR_TYPE))
                          ->mem_[0](0, 0));
#endif
                char filename[80];
                sprintf(filename, "%s_%05d_%03d_%05d_%05d.tga",
                        "replay/frames/gl", (int)saveFrame, (int)e,
                        game->getStep(), (int)d);
#if !defined(CCANADA) && !defined(HPCC)
                game->saveScreenshotToFile(filename, 1200, 1200);
#endif
                saveFrame++;
              }
              this_thread::sleep_for(std::chrono::milliseconds(10));
            }
          }
          if (tpg.GetParam<int>("replay")) {
            totalReplaySteps += game->getStep();
            // for (auto tu = teamUseMap.begin(); tu != teamUseMap.end();
            // tu++) tu->second = tu->second/game->getStep();
          }
          runTimeStats[VISITED_TEAMS_IDX] =
              runTimeStats[VISITED_TEAMS_IDX] / game->getStep();
          runTimeStats[INSTRUCTIONS_IDX] =
              runTimeStats[INSTRUCTIONS_IDX] / game->getStep();
          runTimeStats[MEMBERS_RUN_ENTROPY_IDX] =
              0;  // teams[i]->membersRunEntropy(); this might be broken
          runTimeInts[POINT_AUX_INT_TASK] = tpg.GetState("active_task");
          runTimeInts[POINT_AUX_INT_PHASE] = tpg.GetState("phase");
          runTimeInts[POINT_AUX_INT_ENVSEED] = e + eStart;
          runTimeInts[POINT_AUX_INT_internalTestNodeId] =
              tpg.GetState("internal_test_node_id");
          evalResult += to_string(teams[i]->id_);
          evalResult += ":4" + vecToStrNoSpace(behavSeq);
          evalResult += ":" + to_string(tpg.GetState("active_task"));
          for (size_t r = 0; r < runTimeStats.size(); r++)
            evalResult += ":" + to_string(runTimeStats[r]);
          for (size_t r = 0; r < runTimeInts.size(); r++)
            evalResult += ":" + to_string(runTimeInts[r]);
          teams[i]->getActiveMembersByRef(active);
          for (auto leiter = active.begin(); leiter != active.end(); leiter++)
            evalResult += ":" + to_string((*leiter)->id_);
          evalResult += "\n";
          if (tpg.GetParam<int>("replay")) {
            tpg.setOutcome(teams[i], "", runTimeStats, runTimeInts,
                           tpg.GetParam<int>("t_pickup"));
          }
        }
        // graph use
        if (tpg.GetParam<int>("replay")) {
          for (auto tu = teamUseMap.begin(); tu != teamUseMap.end(); tu++)
            tu->second = tu->second / totalReplaySteps;

          // cout << "visitedTeamsAll";
          // for (auto vt = visitedTeamsAll.begin(); vt !=
          // visitedTeamsAll.end(); vt++)
          //    cout << " " << (*vt)->id_;
          // cout << endl;
          // tpg.printGraphDot(teams[i], -1, -1, -1, teamPath.size(),
          // allPrograms, winningProgramsAll, decisionFeatures,
          // decisionMemories, visitedTeamsAll); winningProgramsAll.clear();
          // allPrograms.clear();
          // teamPath.clear();
          // tpg.printGraphDot(teams[i], -2, -2, -2, teamPath.size(),
          // allPrograms, winningProgramsAll, decisionFeatures,
          // decisionMemories, teamPath);
        }
      }
    }
    if (tpg.GetParam<int>("replay")) break;
    gather(world, evalResult, 0);
  }
  delete worldState;
}

/******************************************************************************/

void replay(TPG &tpg, vector<classicRLEnv *> &tasks, mpi::communicator &world) {
  set<team *, teamIdComp> visitedTeamsAll;
  vector<set<team *, teamIdComp>> visitedTeamsAllPerTask;
  set<team *, teamIdComp> visitedTeamsAllTasks;
  map<long, double>
      teamUseMap;  // maps team to frequency of use for a particular task;
  vector<map<long, double>> teamUseMapPerTask;
  teamUseMapPerTask.reserve(tasks.size());
  teamUseMapPerTask.resize(tasks.size());

  long tmpSeed = tpg._seeds[AUX_SEED_INDEX];
  tpg.readCheckpoint(tpg.GetParam<int>("t_pickup"),
                     tpg.GetParam<int>("checkpoint_in_phase"), -1, false,
                     "");  // setRoots
  tpg.seed(AUX_SEED_INDEX, tmpSeed);
  tpg.state_["active_task"] = 0;
  tpg.params_["phase"] = _TEST_PHASE;
  if (tpg.GetParam<int>("animate")) {
    tpg._numStoredOutcomesPerHost[_TEST_PHASE] = 1;
  }
  evaluate_sub_replay(tpg, world, tasks, visitedTeamsAll, visitedTeamsAllTasks,
                      teamUseMap, teamUseMapPerTask);
  tpg.printTeamInfo(tpg.GetParam<int>("t_pickup"),
                    tpg.GetParam<int>("checkpoint_in_phase"), false,
                    tpg.GetParam<int>("host_to_replay"));
  tpg.printOss();

  
  ////GPEM graphs
  // if (tpg.replay()){

  //   map <long, string> teamColMap;
  //   vector <string> elevenColors;
  //   elevenColors.push_back("#a6cee3");
  //   elevenColors.push_back("#1f78b4");
  //   elevenColors.push_back("#b2df8a");
  //   elevenColors.push_back("#33a02c");
  //   elevenColors.push_back("#fb9a99");
  //   elevenColors.push_back("#e31a1c");
  //   elevenColors.push_back("#fdbf6f");
  //   elevenColors.push_back("#ff7f00");
  //   elevenColors.push_back("#cab2d6");
  //   elevenColors.push_back("#6a3d9a");
  //   elevenColors.push_back("#ffff99");

  //   tpg.readCheckpoint(tpg.GetParam<int>("t_pickup"),
  //   tpg.checkpointInPhase(), -1, false, ""); //setRoots

  //   vector<map<long,double>> teamUseMapPerTaskCopy;// = teamUseMapPerTask;
  //   //replay and record teams visited for each task
  //   for (size_t tsk = 0; tsk < tasks.size(); tsk++){
  //      tpg.ActiveTask(tsk);
  //      tpg.phase(_TEST_PHASE);
  //      tpg.animate(false);
  //      tpg.visual(false);
  //      if (tpg.animate())
  //         tpg.numStoredOutcomesPerHost(_TEST_PHASE, 1);
  //      visitedTeamsAll.clear();
  //      teamUseMap.clear();
  //      //if (tsk == 3)//pendulum only
  //       evaluate_sub(tpg, world, tasks, rngEnv, visitedTeamsAll,
  //       visitedTeamsAllTasks, teamUseMap, teamUseMapPerTask);
  //      visitedTeamsAllPerTask.push_back(visitedTeamsAll);
  //      teamUseMapPerTaskCopy.push_back(teamUseMap);//this gets updated on
  //      the fly visitedTeamsAllTasks.insert(visitedTeamsAll.begin(),
  //      visitedTeamsAll.end());
  //   }
  //   tpg.printGraphDotGPEM(tpg.hostToReplay(), teamColMap,
  //   visitedTeamsAllTasks, teamUseMapPerTaskCopy);
  //
  //   //vector<map<long,double>> teamUseMapPerTaskCopy = teamUseMapPerTask;
  //   //animate
  //   for (size_t tsk = 0; tsk < tasks.size(); tsk++)
  //      teamUseMapPerTask[tsk].clear();
  //   teamUseMap.clear();
  //   tpg.numStoredOutcomesPerHost(_TEST_PHASE, 1);
  //   tpg.animate(true);
  //   tpg.visual(true);
  //   tpg.ActiveTask(5);
  //   rngEnv.seed(tpg.seed2());
  //   evaluate_sub(tpg, world, tasks, rngEnv, visitedTeamsAll,
  //   visitedTeamsAllTasks, teamUseMap, teamUseMapPerTaskCopy);

  //   //int col = 0;
  //   //map<long, vector<int>> teamTaskMap;
  //   //set <string> ActiveTaskSets;
  //   //map<string, string> ActiveTaskSetColMap;
  //   //for (auto vta = visitedTeamsAllTasks.begin(); vta !=
  //   visitedTeamsAllTasks.end(); vta++){
  //   //   vector <int> tskSet;
  //   //   for (size_t tsk = 0; tsk < tasks.size(); tsk++)
  //   //      if (find(visitedTeamsAllPerTask[tsk].begin(),
  //   visitedTeamsAllPerTask[tsk].end(), (*vta)) !=
  //   visitedTeamsAllPerTask[tsk].end())
  //   //         tskSet.push_back(tsk);
  //   //   teamTaskMap[(*vta)->id()] = tskSet;
  //   //   ActiveTaskSets.insert(vecToStrNoSpace(tskSet));
  //   //   if (ActiveTaskSetColMap.find(vecToStrNoSpace(tskSet)) ==
  //   ActiveTaskSetColMap.end())
  //   //      ActiveTaskSetColMap[vecToStrNoSpace(tskSet)] =
  //   elevenColors[col++];
  //   //   teamColMap[(*vta)->id()] =
  //   ActiveTaskSetColMap[vecToStrNoSpace(tskSet)];
  //   //}

  //   //cout << "ActiveTaskSets " << ActiveTaskSets.size() << endl;
  //   //for (auto it = teamTaskMap.begin(); it != teamTaskMap.end(); it++)
  //   //  cout << "teamTaskMap tm " << it->first << " tsks " <<
  //   vecToStrNoSpace(it->second) << " col-" <<
  //   ActiveTaskSetColMap[vecToStrNoSpace(it->second)] << "-" <<
  //   teamColMap[it->first]  << endl;

  //   tpg.printTeamInfo(tpg.GetParam<int>("t_pickup"),
  //   tpg.checkpointInPhase(), false, tpg.hostToReplay()); tpg.printOss();
  //   //tpg.printGraphDotGPEM(tpg.hostToReplay(), teamColMap,
  //   visitedTeamsAllTasks, teamUseMapPerTaskCopy);

  //   ////replay visual
  //   //tpg.ActiveTask(0);
  //   //tpg.phase(_TEST_PHASE);
  //   //tpg.animate(true);
  //   //tpg.visual(true);
  //   //tpg.numStoredOutcomesPerHost(_TEST_PHASE, 1);
  //   //evaluate_sub(tpg, world, tasks, rngEnv, visitedTeamsAll,
  //   visitedTeamsAllTasks);
  //}
}