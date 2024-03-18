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

struct EvalStruct {
  int episode;
  int saveFrame = 0;
  state *obs;
  vector<double> runTimeStats;
  vector<int> runTimeInts;
  vector<int> behavSeq;
  vector<team *> teamPath;
  long decisionInstructions;
  set<team *, teamIdComp> visitedTeams;
  vector<team *> teams;
  team *tm;
  string evalResult;
  string checkpointString;
  set<program *, programIdComp> active;
  classicRLEnv *game;
  program *leafProgram;
  bool animate;
  bool partially_observable;
  EvalStruct(TPG &tpg) {
    obs = new state(tpg.GetParam<int>("n_input"));
    runTimeStats.reserve(tpg.GetParam<int>("n_point_aux_double"));
    runTimeStats.resize(tpg.GetParam<int>("n_point_aux_double"));
    runTimeInts.reserve(tpg.GetParam<int>("n_point_aux_int"));
    runTimeInts.resize(tpg.GetParam<int>("n_point_aux_int"));
    animate = tpg.GetParam<int>("animate") == 1;
    partially_observable = tpg.GetParam<int>("partially_observable") == 1;
  }
  ~EvalStruct() { delete obs; }
};

// TPG represents discrete actions as negative ints starting at -1
// Map them to positive ints starting at 0
int WrapDiscreteAction(EvalStruct &eval) {
  return (eval.leafProgram->action() * -1) - 1;
}

double WrapContinuousAction(EvalStruct &eval) {
  return (eval.leafProgram->memGet(memoryEigen::SCALAR_TYPE))->mem_[0](0, 0);
}

vector<team *> GetTeamsToEval(TPG &tpg) {
  auto root_teams = tpg.GetTeams(true);
  vector<team *> teams;
  if (tpg.GetParam<int>("replay") == 1) {
    teams.push_back(tpg._teamMap[tpg.GetParam<int>("host_to_replay")]);
    if (tpg.GetParam<int>("animate"))
      teams[0]->_n_eval = 1;
    else
      teams[0]->_n_eval = tpg._numStoredOutcomesPerHost[tpg.GetState("phase")];
  } else {
    for (auto it : root_teams) {
      it.second->_n_eval =
          tpg._numStoredOutcomesPerHost[tpg.GetState("phase")] -
          it.second->numOutcomes(tpg.GetState("phase"),
                                 tpg.GetState("active_task"));
      if (it.second->_n_eval > 0) {
        teams.push_back(it.second);
      }
    }
  }
  return teams;
}

void AssignTeamsToEvaluators(TPG &tpg, mpi::communicator &world,
                             vector<team *> &teams_to_eval,
                             int world_size_per_task, int &evaluator) {
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

void MaybeStartAnimation(TPG &tpg) {
#if !defined(CCANADA) && !defined(HPCC)
  if (tpg.GetParam<int>("animate")) {
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
#endif
}

void MaybeAnimateStep(EvalStruct &eval) {
#if !defined(CCANADA)
  if (eval.animate) {
    eval.game->display_function(eval.episode, WrapDiscreteAction(eval),
                                WrapContinuousAction(eval));
    char filename[80];
    sprintf(filename, "%s_%05d_%03d_%05d_%05d_%05d.tga", "replay/frames/gl",
            eval.saveFrame++, eval.episode, eval.game->getStep(), 0, 0);
    eval.game->saveScreenshotToFile(filename, 1200, 1200);
    this_thread::sleep_for(std::chrono::milliseconds(10));
  }
#endif
}

void AccumulateStepStats(EvalStruct &eval) {
  if (eval.game->getStep() == 1) {
    fill(eval.runTimeStats.begin(), eval.runTimeStats.end(), 0);
    eval.behavSeq.clear();
  }
  if (eval.game->discreteActions())
    eval.behavSeq.push_back(eval.leafProgram->action());
  else
    eval.behavSeq.push_back(-1 -
                            discretize(bound(WrapContinuousAction(eval),
                                             eval.game->minActionContinuous(),
                                             eval.game->maxActionContinuous()),
                                       eval.game->minActionContinuous(),
                                       eval.game->maxActionContinuous(), 3));
  eval.behavSeq.push_back(discretize(eval.obs->getStateVarDouble(0), 0, 1, 3));
  eval.behavSeq.push_back(discretize(eval.obs->getStateVarDouble(1), 0, 1, 3));
  eval.runTimeStats[VISITED_TEAMS_IDX] += eval.visitedTeams.size();
  eval.runTimeStats[INSTRUCTIONS_IDX] += eval.decisionInstructions;
  eval.obs->Set(eval.game->getStateVec(eval.partially_observable));
}

void FinalizeStepStats(TPG &tpg, EvalStruct &eval) {
  eval.runTimeStats[VISITED_TEAMS_IDX] /= eval.game->getStep();
  eval.runTimeStats[INSTRUCTIONS_IDX] /= eval.game->getStep();
  eval.runTimeStats[MEMBERS_RUN_ENTROPY_IDX] = 0;  // place holder
  eval.runTimeInts[POINT_AUX_INT_TASK] = tpg.GetState("active_task");
  eval.runTimeInts[POINT_AUX_INT_PHASE] = tpg.GetState("phase");
  eval.runTimeInts[POINT_AUX_INT_ENVSEED] = eval.episode;
  eval.runTimeInts[POINT_AUX_INT_internalTestNodeId] =
      tpg.GetState("internal_test_node_id");
  eval.evalResult += to_string(eval.tm->id_);
  eval.evalResult += ":4" + vecToStrNoSpace(eval.behavSeq);  // 4?
  eval.evalResult += ":" + to_string(tpg.GetState("active_task"));
  for (size_t r = 0; r < eval.runTimeStats.size(); r++)
    eval.evalResult += ":" + to_string(eval.runTimeStats[r]);
  for (size_t r = 0; r < eval.runTimeInts.size(); r++)
    eval.evalResult += ":" + to_string(eval.runTimeInts[r]);
  eval.tm->getActiveMembersByRef(eval.active);
  for (auto leiter = eval.active.begin(); leiter != eval.active.end(); leiter++)
    eval.evalResult += ":" + to_string((*leiter)->id_);
  eval.evalResult += "\n";
}

bool NotDoneAndActive(EvalStruct &eval) {
  return eval.checkpointString.compare("x") != 0 &&
         eval.checkpointString.compare("done") != 0;
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
    AssignTeamsToEvaluators(tpg, world, teams_to_eval, world_size_per_task,
                            evaluator);
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
 * Receive agents from main MPI proc, eval in environment, return results
 */
void evaluator(TPG &tpg, mpi::communicator &world,
               vector<classicRLEnv *> &tasks) {
  MaybeStartAnimation(tpg);
  EvalStruct eval(tpg);
  while (NotDoneAndActive(eval)) {
    world.recv(0, 0, eval.checkpointString);
    if (NotDoneAndActive(eval)) {
      if (tpg.GetParam<int>("replay")) {
        tpg.readCheckpoint(tpg.GetParam<int>("t_pickup"),
                     tpg.GetParam<int>("checkpoint_in_phase"), -1, false,
                     "");  // setRoots
      } else {
      tpg.readCheckpoint(-1, _TRAIN_PHASE, -1, true, eval.checkpointString);
      }   
      tpg.getTeams(eval.teams, true);
      eval.game = tasks[tpg.GetState("active_task")];
      eval.evalResult = "";
      for (auto tm : eval.teams) {
        eval.tm = tm;
        tpg.markEffectiveCode(eval.tm);
        for (eval.episode = 0; eval.episode < eval.tm->_n_eval;
             eval.episode++) {
          tpg._rngs[AUX_SEED_INDEX].seed(eval.episode);
          eval.tm->clearMemory(tpg._teamMap);
          eval.game->reset(tpg._rngs[AUX_SEED_INDEX]);
          eval.obs->Set(eval.game->getStateVec(eval.partially_observable));
          while (!eval.game->terminal()) {
            eval.leafProgram =
                tpg.getAction(eval.tm, eval.obs, true, eval.visitedTeams,
                              eval.decisionInstructions, eval.game->getStep(),
                              eval.teamPath, tpg._rngs[AUX_SEED_INDEX]);
            MaybeAnimateStep(eval);
            eval.runTimeStats[REWARD_IDX] += eval.game->update(
                WrapDiscreteAction(eval), WrapContinuousAction(eval),
                tpg._rngs[AUX_SEED_INDEX]);
            AccumulateStepStats(eval);
          }
          FinalizeStepStats(tpg, eval);
        }
      }
    gather(world, eval.evalResult, 0);
    }
  }
  tpg.finalize();
}