#include <Acrobot.h>
#include <CartCentering.h>
#include <CartPole.h>
#include <MountainCar.h>
#include <MountainCarContinuous.h>
#include <Pendulum.h>
#include <TPG.h>
#include <TaskEnv.h>

#include <boost/mpi.hpp>
#include <chrono>
#include <thread>

#if !defined(CCANADA) && !defined(HPCC)
#include <GL/gl.h>
#include <GL/glut.h>
#endif

#define REWARD1_IDX 0
#define VISITED_TEAMS_IDX 1
#define INSTRUCTIONS_IDX 2
#define REWARD2_IDX 3

namespace mpi = boost::mpi;

struct EvalStruct;
typedef void (*EvaluatorFunction)(TPG &, EvalStruct &);

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
  TaskEnv *game;
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
  return eval.leafProgram->privateMemory_[memoryEigen::SCALAR_TYPE]
      ->working_memory_[1](0, 0);
}
double WrapContinuousActionSigmoid(EvalStruct &eval) {
  double p = eval.leafProgram->privateMemory_[memoryEigen::SCALAR_TYPE]
                 ->working_memory_[1](0, 0);
  return 1 / (1 + exp(-p));
}

vector<team *> GetTeamsToEval(TPG &tpg) {
  auto root_teams = tpg.GetTeamsInVec(true);
  vector<team *> teams_to_eval;
  // train and validate all teams
  if (tpg.GetState("phase") != _TEST_PHASE) {
    for (auto tm : root_teams) {
      tm->_n_eval =
          tpg._numStoredOutcomesPerHost[tpg.GetState("active_task")][tpg.GetState("phase")] -
          tm->numOutcomes(tpg.GetState("phase"), tpg.GetState("active_task"));    
      if (tm->_n_eval > 0) {
        teams_to_eval.push_back(tm);
      }
    }
  } else {
    // // only test the validation champions (set fitmode later)
    // // auto PS = PowerSet(tpg.GetParam<int>("n_task"));
    // // for (auto &set : PS) {
    // // team *tm =
    // //
    // tpg._eliteTeamPS[vecToStrNoSpace(set)][tpg.GetParam<int>("fit_mode")][_VALIDATION_PHASE];
    // team *tm =
    //     tpg._eliteTeamPS[to_string(tpg.GetState("active_task"))]
    //                     [tpg.GetParam<int>("fit_mode")][_VALIDATION_PHASE];
    // tm->_n_eval =
    //     tpg._numStoredOutcomesPerHost[tpg.GetState("active_task")][tpg.GetState("phase")] -
    //     tm->numOutcomes(tpg.GetState("phase"), tpg.GetState("active_task"));
    // teams_to_eval.push_back(tm);
    // // }

    // TODO(skelly): for now test every root team
    for (auto tm : root_teams) {
      tm->_n_eval =
          tpg._numStoredOutcomesPerHost[tpg.GetState("active_task")][tpg.GetState("phase")] -
          tm->numOutcomes(tpg.GetState("phase"), tpg.GetState("active_task"));
      if (tm->_n_eval > 0) {
        teams_to_eval.push_back(tm);
      }
    }
  }
  return teams_to_eval;
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
  (void)tpg;
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
  (void)eval;
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
}

void FinalizeStepStats(TPG &tpg, EvalStruct &eval) {
  if (eval.game->eval_type_ == "RecursiveForecast") {
    eval.runTimeStats[REWARD1_IDX] /= eval.game->getStep();
    eval.runTimeStats[REWARD2_IDX] /= eval.game->getStep();
    if (!isfinite(eval.runTimeStats[REWARD1_IDX]))
      eval.runTimeStats[REWARD1_IDX] = eval.game->min_reward_;
    if (!isfinite(eval.runTimeStats[REWARD2_IDX]))
      eval.runTimeStats[REWARD2_IDX] = eval.game->min_reward_;
  }
  eval.runTimeStats[VISITED_TEAMS_IDX] /= eval.game->getStep();
  eval.runTimeStats[INSTRUCTIONS_IDX] /= eval.game->getStep();
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
  eval.evalResult += "\n";
}

bool NotDoneAndActive(EvalStruct &eval) {
  return eval.checkpointString.compare("x") != 0 &&
         eval.checkpointString.compare("done") != 0;
}

/*******************************************************************************
 * 1. Assign agents to evaluator procs
 * 2. Wait for evals to finish
 * 3. Collect results
 ******************************************************************************/
void evaluate_main(TPG &tpg, mpi::communicator &world, vector<int> &taskSet) {
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
  auto root_teams_map = tpg.GetTeamsInMap(true);
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
        tpg.setOutcome(root_teams_map[rslt_id], behavSeq, r_runTimeStats,
                       r_runTimeInts, tpg.GetState("t_current"));
      }
    }
  }
}

/******************************************************************************/
void EvalControl(TPG &tpg, EvalStruct &eval) {
  bool verbose = false;  // tpg.GetState("phase") == _TEST_PHASE ? true : false;
  eval.game->reset(tpg.rngs_[AUX_SEED]);
  eval.obs->Set(eval.game->GetObsVec(eval.partially_observable));
  while (!eval.game->terminal()) {
    eval.leafProgram = tpg.getAction(
        eval.tm, eval.obs, true, eval.visitedTeams, eval.decisionInstructions,
        eval.game->getStep(), eval.teamPath, tpg.rngs_[AUX_SEED], verbose);
    MaybeAnimateStep(eval);
    TaskEnv::Results r =
        eval.game->update(WrapDiscreteAction(eval), WrapContinuousAction(eval),
                          tpg.rngs_[AUX_SEED]);
    eval.runTimeStats[REWARD1_IDX] += r.r1;
    AccumulateStepStats(eval);
    eval.obs->Set(eval.game->GetObsVec(eval.partially_observable));
  }
}

/******************************************************************************/
void EvalRecursiveForecast(TPG &tpg, EvalStruct &eval) {
  RecursiveUnivar *game = dynamic_cast<RecursiveUnivar *>(eval.game);
  game->reset(tpg.rngs_[AUX_SEED]);
  bool verbose = false;  // tpg.GetState("phase") == _TEST_PHASE ? true : false;
  list<double> obs_list(tpg.GetParam<int>("n_input"), 0.0);
  vector<double> obs(tpg.GetParam<int>("n_input"), 0.0);
  // prime
  int sample = game->t_start[tpg.GetState("phase")][eval.episode];
  for (int i = 0; i < game->num_samples_prime_ - 1; i++) {
    obs_list.push_back(game->data[sample][0]);
    obs_list.pop_front();
    // TODO(skelly): make this more efficient ?
    std::copy(obs_list.begin(), obs_list.end(), obs.begin());

    eval.obs->Set(obs);
    eval.leafProgram = tpg.getAction(
        eval.tm, eval.obs, true, eval.visitedTeams, eval.decisionInstructions,
        eval.game->getStep(), eval.teamPath, tpg.rngs_[AUX_SEED], verbose);
    sample++;
  }
  // predict
  for (int i = 0; i < game->num_samples_predict_[tpg.GetState("phase")]; i++) {
    obs_list.push_back(WrapContinuousActionSigmoid(eval));
    obs_list.pop_front();
    // TODO(skelly): make this more efficient ?
    std::copy(obs_list.begin(), obs_list.end(), obs.begin());
    eval.obs->Set(obs);
    eval.leafProgram = tpg.getAction(
        eval.tm, eval.obs, true, eval.visitedTeams, eval.decisionInstructions,
        game->getStep(), eval.teamPath, tpg.rngs_[AUX_SEED], verbose);
    TaskEnv::Results r = game->update(
        sample++, WrapContinuousActionSigmoid(eval), tpg.rngs_[AUX_SEED]);
    eval.runTimeStats[REWARD1_IDX] += r.r1;  // MSE
    eval.runTimeStats[REWARD2_IDX] += r.r2;  // MAE
    AccumulateStepStats(eval);
  }
}

/*******************************************************************************
 * Receive agents from main MPI proc, eval in environment, return results
 */
void evaluator(TPG &tpg, mpi::communicator &world, vector<TaskEnv *> &tasks) {
  unordered_map<string, EvaluatorFunction> evaluator_map;
  evaluator_map["Control"] = &EvalControl;
  evaluator_map["RecursiveForecast"] = &EvalRecursiveForecast;
  MaybeStartAnimation(tpg);
  EvalStruct eval(tpg);
  while (NotDoneAndActive(eval)) {
    world.recv(0, 0, eval.checkpointString);
    if (NotDoneAndActive(eval)) {
      tpg.readCheckpoint(-1, _TRAIN_PHASE, -1, true, eval.checkpointString);
      tpg.getTeams(eval.teams, true);
      eval.game = tasks[tpg.GetState("active_task")];
      eval.evalResult = "";
      for (auto tm : eval.teams) {
        eval.tm = tm;
        tpg.MarkEffectiveCode(eval.tm);

        for (eval.episode = 0; eval.episode < eval.tm->_n_eval;
             eval.episode++) {
          tpg.rngs_[AUX_SEED].seed(eval.episode);
          eval.tm->InitMemory(tpg._teamMap, tpg.HaveParam("p_bid_mu_const"));
          // tpg.InitMemory();
          evaluator_map[eval.game->eval_type_](tpg, eval);
          FinalizeStepStats(tpg, eval);
        }
      }
      gather(world, eval.evalResult, 0);
    }
  }
  tpg.finalize();
}

/******************************************************************************/
void replayer(TPG &tpg, vector<TaskEnv *> &tasks) {
  MaybeStartAnimation(tpg);
  EvalStruct eval(tpg);
  tpg.getTeams(eval.teams, true);
  eval.game = tasks[tpg.GetState("active_task")];
  eval.evalResult = "";
  for (auto tm : eval.teams) {
    eval.tm = tm;
    if (eval.animate) eval.tm->_n_eval = 1;
    tpg.MarkEffectiveCode(eval.tm);
    for (eval.episode = 0; eval.episode < eval.tm->_n_eval; eval.episode++) {
      tpg.rngs_[AUX_SEED].seed(eval.episode);
      eval.tm->InitMemory(tpg._teamMap, tpg.HaveParam("p_bid_mu_const"));
      EvalControl(tpg, eval);
      FinalizeStepStats(tpg, eval);
    }
  }
}

/******************************************************************************/
void EvalControlViz(TPG &tpg, EvalStruct &eval,
                    vector<map<long, double>> &teamUseMapPerTask,
                    set<team *, teamIdComp> &visitedTeamsAllTasks, int &steps) {
  bool verbose = false;  // tpg.GetState("phase") == _TEST_PHASE ? true : false;
  eval.game->reset(tpg.rngs_[AUX_SEED]);
  eval.obs->Set(eval.game->GetObsVec(eval.partially_observable));
  while (!eval.game->terminal()) {
    eval.leafProgram = tpg.getAction(
        eval.tm, eval.obs, true, eval.visitedTeams, eval.decisionInstructions,
        eval.game->getStep(), eval.teamPath, tpg.rngs_[AUX_SEED], verbose);

    for (auto tm : eval.visitedTeams) {
      if (teamUseMapPerTask[tpg.state_["active_task"]].find(tm->id_) ==
          teamUseMapPerTask[tpg.state_["active_task"]].end()) {
        teamUseMapPerTask[tpg.state_["active_task"]][tm->id_] = 1.0;
      } else {
        teamUseMapPerTask[tpg.state_["active_task"]][tm->id_] += 1.0;
      }
    }
    steps++;
    // teamUseMapPerTask[tpg.state_["active_task"]][eval.tm->id_] += 1.0;
    visitedTeamsAllTasks.insert(eval.visitedTeams.begin(),
                                eval.visitedTeams.end());

    MaybeAnimateStep(eval);
    TaskEnv::Results r =
        eval.game->update(WrapDiscreteAction(eval), WrapContinuousAction(eval),
                          tpg.rngs_[AUX_SEED]);
    eval.runTimeStats[REWARD1_IDX] += r.r1;
    AccumulateStepStats(eval);
    eval.obs->Set(eval.game->GetObsVec(eval.partially_observable));
  }
  for (auto p : teamUseMapPerTask[tpg.state_["active_task"]]) {
    p.second = p.second / eval.game->step;
  }
}

/******************************************************************************/
void EvalRecursiveForecastViz(TPG &tpg, EvalStruct &eval,
                              vector<map<long, double>> &teamUseMapPerTask,
                              set<team *, teamIdComp> &visitedTeamsAllTasks,
                              int &steps) {
  cerr << "";  // TODO(skelly): why do we need this?
  RecursiveUnivar *game = dynamic_cast<RecursiveUnivar *>(eval.game);
  game->reset(tpg.rngs_[AUX_SEED]);
  bool verbose = false;  // tpg.GetState("phase") == _TEST_PHASE ? true : false;
  list<double> obs_list(tpg.GetParam<int>("n_input"), 0.0);
  vector<double> obs(tpg.GetParam<int>("n_input"), 0.0);
  // prime
  int sample =
      game->t_start[tpg.GetParam<int>("checkpoint_in_phase")][eval.episode];   
  for (int i = 0; i < game->num_samples_prime_ - 1; i++) {
    obs_list.push_back(game->data[sample][0]);
    obs_list.pop_front();  //  FIFO
    // TODO(skelly): make this more efficient ?
    std::copy(obs_list.begin(), obs_list.end(), obs.begin());
    eval.obs->Set(obs);
    eval.leafProgram = tpg.getAction(
        eval.tm, eval.obs, true, eval.visitedTeams, eval.decisionInstructions,
        eval.game->getStep(), eval.teamPath, tpg.rngs_[AUX_SEED], verbose);

    for (auto tm : eval.visitedTeams) {
      if (teamUseMapPerTask[tpg.state_["active_task"]].find(tm->id_) ==
          teamUseMapPerTask[tpg.state_["active_task"]].end()) {
        teamUseMapPerTask[tpg.state_["active_task"]][tm->id_] = 1.0;
      } else {
        teamUseMapPerTask[tpg.state_["active_task"]][tm->id_] += 1.0;
      }
    }
    steps++;
    // teamUseMapPerTask[tpg.state_["active_task"]][eval.tm->id_] += 1.0;
    visitedTeamsAllTasks.insert(eval.visitedTeams.begin(),
                                eval.visitedTeams.end());

    sample++;
  }
  // predict
  for (int i = 0;
       i < game->num_samples_predict_[tpg.GetParam<int>("checkpoint_in_phase")];
       i++) {
    obs_list.push_back(WrapContinuousAction(eval));
    obs_list.pop_front();  //  FIFO
    // TODO(skelly): make this more efficient ?
    std::copy(obs_list.begin(), obs_list.end(), obs.begin());
    eval.obs->Set(obs);
    eval.leafProgram = tpg.getAction(
        eval.tm, eval.obs, true, eval.visitedTeams, eval.decisionInstructions,
        game->getStep(), eval.teamPath, tpg.rngs_[AUX_SEED], verbose);

    for (auto tm : eval.visitedTeams) {
      if (teamUseMapPerTask[tpg.state_["active_task"]].find(tm->id_) ==
          teamUseMapPerTask[tpg.state_["active_task"]].end()) {
        teamUseMapPerTask[tpg.state_["active_task"]][tm->id_] = 1.0;
      } else {
        teamUseMapPerTask[tpg.state_["active_task"]][tm->id_] += 1.0;
      }
    }
    steps++;
    // teamUseMapPerTask[tpg.state_["active_task"]][eval.tm->id_] += 1.0;
    visitedTeamsAllTasks.insert(eval.visitedTeams.begin(),
                                eval.visitedTeams.end());
    TaskEnv::Results r = game->update(
        sample++, WrapContinuousAction(eval), tpg.rngs_[AUX_SEED]);
    cerr << std::fixed << "test tm " << eval.tm->id_ << " t,p "
         << game->data[sample][0] << "," << WrapContinuousAction(eval)
         << endl;
    eval.runTimeStats[REWARD1_IDX] += r.r1;  // MSE
    eval.runTimeStats[REWARD2_IDX] += r.r2;  // MAE
    AccumulateStepStats(eval);
  }
}

/******************************************************************************/
// For each task, map team id to the proportion of steps in which this team was
// used vector<map<long, double>> teamUseMapPerTask;

void replayer_viz(TPG &tpg, vector<TaskEnv *> &tasks) {
  MaybeStartAnimation(tpg);
  EvalStruct eval(tpg);

  vector<map<long, double>> teamUseMapPerTask;
  teamUseMapPerTask.resize(tpg.GetParam<int>("n_task"));
  std::set<team *, teamIdComp> visitedTeamsAllTasks;

  tpg.getTeams(eval.teams, true);
  eval.evalResult = "";
  for (auto tm : eval.teams) {
    if (tm->id_ != tpg.GetParam<int>("host_to_replay")) continue;
    eval.tm = tm;
    if (eval.animate) eval.tm->_n_eval = 1;
    eval.tm->_n_eval =
        tpg._numStoredOutcomesPerHost[tpg.GetState("active_task")][tpg.GetParam<int>("checkpoint_in_phase")];
    tpg.MarkEffectiveCode(eval.tm);
    vector<int> steps_per_task(tpg.GetParam<int>("n_task"), 0);
    for (int task = 0; task < tpg.GetParam<int>("n_task"); task++) {
      tpg.state_["active_task"] = task;
      eval.game = tasks[tpg.GetState("active_task")];

      for (eval.episode = 0; eval.episode < eval.tm->_n_eval; eval.episode++) {
        tpg.rngs_[AUX_SEED].seed(eval.episode);
        eval.tm->InitMemory(tpg._teamMap, tpg.HaveParam("p_bid_mu_const"));

        if (eval.game->eval_type_ == "RecursiveForecast") {
          EvalRecursiveForecastViz(tpg, eval, teamUseMapPerTask,
                                   visitedTeamsAllTasks, steps_per_task[task]);
        } else {
          EvalControlViz(tpg, eval, teamUseMapPerTask, visitedTeamsAllTasks,
                         steps_per_task[task]);
        }
        FinalizeStepStats(tpg, eval);
      }
    }
    tpg.printGraphDotGPTPXXI(eval.tm->id_, visitedTeamsAllTasks,
                             teamUseMapPerTask, steps_per_task);
  }
}
