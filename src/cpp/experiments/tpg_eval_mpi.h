#ifndef tpg_eval_mpi_h
#define tpg_eval_mpi_h

#include <Acrobot.h>
#include <CartCentering.h>
#include <CartPole.h>
#include <MountainCar.h>
#include <MountainCarContinuous.h>
#include <Pendulum.h>
#include <TPG.h>
#include <TaskEnv.h>
#include <sequence_comparisons.h>

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

// TODO(spkelly): put this in TPG.h
struct EvalStruct {
  int episode;
  int sample;
  int n_prediction;
  int saveFrame = 0;

  vector<double> runTimeStats;
  vector<int> runTimeInts;
  vector<int> behavSeq;
  vector<team *> teamPath;
  long decision_instructions;
  set<team *, teamIdComp> visitedTeams;
  vector<team *> teams;
  team *tm;
  string evalResult;
  string checkpointString;
  set<program *, programIdComp> active;
  TaskEnv *task;
  program *leafProgram;
  bool animate;
  bool partially_observable;

  // only for recursive forecasting
  vector<double> sequence_targ;
  vector<double> sequence_pred;

  state *obs;  // = new state(tpg.n_input_[tpg.GetState("active_task")]);
  list<double> obs_list;   //(tpg.n_input_[tpg.GetState("active_task")], 1.0);
  vector<double> obs_vec;  //(tpg.n_input_[tpg.GetState("active_task")], 1.0);

  //////////////////////////////////////////////////////////////////////////////

  EvalStruct(TPG &tpg) {
    // runTimeStats.reserve(tpg.GetParam<int>("n_point_aux_double"));
    runTimeStats.resize(tpg.GetParam<int>("n_point_aux_double"));
    // runTimeInts.reserve(tpg.GetParam<int>("n_point_aux_int"));
    runTimeInts.resize(tpg.GetParam<int>("n_point_aux_int"));
    animate = tpg.GetParam<int>("animate") == 1;
    partially_observable = tpg.GetParam<int>("partially_observable") == 1;
    n_prediction = 0;
  }
  ~EvalStruct() {}
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

vector<double> WrapVectorActionSigmoid(EvalStruct &eval) {
  auto mat = eval.leafProgram->privateMemory_[memoryEigen::VECTOR_TYPE]
                 ->working_memory_[0];
  vector<double> vec(mat.data(), mat.data() + mat.rows() * mat.cols());
  for (auto &v : vec) v = sigmoid(v);  // TODO(skelly): better/faster way?
  return vec;
}

vector<team *> GetTeamsToEval(TPG &tpg, TaskEnv *task) {
  auto root_teams = tpg.GetTeamsInVec(true);
  vector<team *> teams_to_eval;
  // train and validate all teams
  if (tpg.GetState("phase") != _TEST_PHASE) {
    for (auto tm : root_teams) {
      tm->_n_eval =
          task->GetNumEval(tpg.GetState("phase")) -
          tm->numOutcomes(tpg.GetState("phase"), tpg.GetState("active_task"));
      if (tm->_n_eval > 0) {
        teams_to_eval.push_back(tm);
      }
    }
  } else {
    // test the single validation champion for each subset
    auto PS = PowerSet(tpg.GetState("n_task"));
    for (auto &set : PS) {
      team *tm =
          tpg._eliteTeamPS[vecToStrNoSpace(set)][tpg.GetParam<int>("fit_mode")]
                          [_VALIDATION_PHASE];
      tm->_n_eval =
          task->GetNumEval(tpg.GetState("phase")) -
          tm->numOutcomes(tpg.GetState("phase"), tpg.GetState("active_task"));
      teams_to_eval.push_back(tm);
    }
  }
  return teams_to_eval;
}

/**
 * Parameters:
 * - tpg: TPG instance
 * - world: MPI communicator object, which represents a group of processes that
 * can communicate with each other
 * - teams_to_eval: teams to evaluate
 * - world_size_per_task: number of processors available to evaluate on this
 * task
 * - evaluator: keeps track of current processor
 */
void AssignTeamsToEvaluators(TPG &tpg, mpi::communicator &world,
                             vector<team *> &teams_to_eval,
                             int world_size_per_task, int &evaluator) {
  auto teams_per_evaluator = teams_to_eval.size() / world_size_per_task;
  auto remainder = teams_to_eval.size() % world_size_per_task;
  vector<team *> teams;

  for (auto it = teams_to_eval.begin(); it != teams_to_eval.end(); it++) {
    // Assign teams_per_evaluator teams to each of world_size_per_task
    // processors
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
    eval.task->display_function(eval.episode, WrapDiscreteAction(eval),
                                WrapContinuousAction(eval));
    char filename[80];
    sprintf(filename, "%s_%05d_%03d_%05d_%05d_%05d.tga", "replay/frames/gl",
            eval.saveFrame++, eval.episode, eval.task->step, 0, 0);
    eval.task->saveScreenshotToFile(filename, 1200, 1200);
    this_thread::sleep_for(std::chrono::milliseconds(10));
  }
#endif
}

void AccumulateStepStats(EvalStruct &eval) {
  // TODO(spkelly): re-enable behavSeq with obs outide of EvalStruct
  if (eval.n_prediction == 1) {
    fill(eval.runTimeStats.begin(), eval.runTimeStats.end(), 0);
    // eval.behavSeq.clear();
  }
  // if (eval.task->discreteActions())
  //   eval.behavSeq.push_back(eval.leafProgram->action());
  // else
  //   eval.behavSeq.push_back(-1 -
  //                           discretize(bound(WrapContinuousAction(eval),
  //                                            eval.task->minActionContinuous(),
  //                                            eval.task->maxActionContinuous()),
  //                                      eval.task->minActionContinuous(),
  //                                      eval.task->maxActionContinuous(), 3));
  // eval.behavSeq.push_back(discretize(eval.obs->getStateVarDouble(0), 0, 1,
  // 3)); eval.behavSeq.push_back(discretize(eval.obs->getStateVarDouble(1), 0,
  // 1, 3));
  eval.runTimeStats[VISITED_TEAMS_IDX] += eval.visitedTeams.size();
  eval.runTimeStats[INSTRUCTIONS_IDX] += eval.decision_instructions;
}

void FinalizeStepStats(TPG &tpg, EvalStruct &eval) {
  if (eval.task->eval_type_ == "RecursiveForecast") {
    if (tpg.GetParam<string>("forecast_fitness") == "mse") {
      auto err = MeanSquaredError(eval.sequence_targ, eval.sequence_pred);
      eval.runTimeStats[REWARD1_IDX] = -err;
    } else if (tpg.GetParam<string>("forecast_fitness") == "correlation") {
      auto corr = Correlation(eval.sequence_targ, eval.sequence_pred);
      eval.runTimeStats[REWARD1_IDX] = corr;
    } else if (tpg.GetParam<string>("forecast_fitness") == "pearson") {
      auto corr = PearsonCorrelation(eval.sequence_targ, eval.sequence_pred);
      eval.runTimeStats[REWARD1_IDX] = corr;
    } else if (tpg.GetParam<string>("forecast_fitness") == "theils") {
      auto err = TheilsStatistic(eval.sequence_targ, eval.sequence_pred);
      eval.runTimeStats[REWARD1_IDX] = -err;
    } else if (tpg.GetParam<string>("forecast_fitness") == "mse_multivar") {
      auto err = calculateMSE_Multi(eval.sequence_targ, eval.sequence_pred);
      eval.runTimeStats[REWARD1_IDX] = -err;
    } else if (tpg.GetParam<string>("forecast_fitness") == "theils_multivar") {
      auto err = calculateTheils_Multi(eval.sequence_targ, eval.sequence_pred);
      eval.runTimeStats[REWARD1_IDX] = -err;
    } else if (tpg.GetParam<string>("forecast_fitness") == "pearson_multivar") {
      auto corr =
          calculatePearson_Multi(eval.sequence_targ, eval.sequence_pred);
      eval.runTimeStats[REWARD1_IDX] = corr;
      if (!isfinite(eval.runTimeStats[REWARD1_IDX]))
        eval.runTimeStats[REWARD1_IDX] = 0;
    } else {
      die(__FILE__, __FUNCTION__, __LINE__,
          "Unsupported forecast fitness function");
    }
  }
  eval.runTimeStats[VISITED_TEAMS_IDX] /= eval.n_prediction;
  eval.runTimeStats[INSTRUCTIONS_IDX] /= eval.n_prediction;
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

/**
 * 1. Assign agents to evaluator procs
 *  a. Partition available processes into groups for each task
 *  b. Each process in a group evaluates a subset of agents on the task
 * 2. Wait for evals to finish
 * 3. Collect results
 *
 * @param tpg The TPG instance with all the teams
 * @param world The MPI communicator object
 * @param tasks The set of all tasks in the TPG
 * @param evalTasks The indices of the tasks to evaluate
 */
void evaluate_main(TPG &tpg, mpi::communicator &world, vector<TaskEnv *> &tasks,
                   vector<int> evalTasks) {
  string my_string = "MAIN";
  vector<team *> teams_this_eval;
  vector<string> all_strings;
  vector<string> splitStr;
  string resultLine;

  int world_size_per_task = (world.size() - 1) / tasks.size();
  // assign agents to evaluators
  int evaluator = 1;
  for (int task : evalTasks) {
    tpg.state_["active_task"] = task;
    auto teams_to_eval = GetTeamsToEval(tpg, tasks[task]);

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

/// Returns the fitness of a team on a given task using its phylogeny
double estimate_fitness(TPG &tpg, team *tm, int task) {
  std::vector<long> visited = {tm->id_};
  list<long> queue = {tm->id_};

  // Breadth-first search through phylogeny
  while (!queue.empty()) {
    long currId = queue.front();
    queue.pop_front();

    // If the team has been evaluated on the task, return its fitness
    vector<double> taskFitnesses = tpg._phyloGraph[currId].taskFitnesses;
    int sizeInt = static_cast<int>(taskFitnesses.size());
    if (task < sizeInt) {
      return taskFitnesses[task];
    }

    for (long ancId : tpg._phyloGraph[currId].ancestorIds) {
      if (std::find(visited.begin(), visited.end(), ancId) == visited.end()) {
        visited.push_back(ancId);
        queue.push_back(ancId);
      }
    }
  }

  cerr << "Reached unexpected point in estimate_fitness function" << endl;
  return 0;
}

/// @brief Estimates the fitness of all teams on a given set of tasks
/// @param tpg The TPG instance with all the teams
/// @param tasks The set of all tasks in the TPG
/// @param estTasks Task indices to estimate fitness on
void estimate_main(TPG &tpg, vector<TaskEnv *> &tasks, vector<int> estTasks) {
  // Loop through tasks
  for (int task : estTasks) {
    tpg.state_["active_task"] = task;
    auto teams_to_eval = GetTeamsToEval(tpg, tasks[task]);

    // Loop through teams
    for (auto tm : teams_to_eval) {
      // Estimate fitness of team
      double est_fit = estimate_fitness(tpg, tm, task);

      string behavSeq = "";
      vector<double> r_runTimeStats(4);
      r_runTimeStats[0] = est_fit;
      vector<int> r_runTimeInts(4);
      r_runTimeInts[POINT_AUX_INT_TASK] = task;
      r_runTimeInts[POINT_AUX_INT_PHASE] = tpg.GetState("phase");

      for (int i = 0; i < tasks[task]->GetNumEval(tpg.GetState("phase")); i++) {
        r_runTimeInts[POINT_AUX_INT_ENVSEED] = i;
        tpg.setOutcome(tm, behavSeq, r_runTimeStats, r_runTimeInts,
                       tpg.GetState("t_current"));
      }
    }
  }
}

/******************************************************************************/
void EvalControl(TPG &tpg, EvalStruct &eval) {
  eval.task->reset(tpg.rngs_[AUX_SEED]);
  eval.n_prediction = 0;
  state *obs = new state(tpg.n_input_[tpg.GetState("active_task")]);
  obs->Set(eval.task->GetObsVec(eval.partially_observable));
  while (!eval.task->terminal()) {
    eval.leafProgram = tpg.getAction(
        eval.tm, obs, true, eval.visitedTeams, eval.decision_instructions,
        eval.task->step, eval.teamPath, tpg.rngs_[AUX_SEED], false);
    eval.n_prediction++;
    MaybeAnimateStep(eval);
    TaskEnv::Results r =
        eval.task->update(WrapDiscreteAction(eval), WrapContinuousAction(eval),
                          tpg.rngs_[AUX_SEED]);
    eval.runTimeStats[REWARD1_IDX] += r.r1;
    AccumulateStepStats(eval);
    obs->Set(eval.task->GetObsVec(eval.partially_observable));
  }
  delete obs;
}

void SaveRecursiveForecast(TPG &tpg, EvalStruct &eval) {
  RecursiveForecast *task = dynamic_cast<RecursiveForecast *>(eval.task);
  bool discrete_actions = tpg.GetParam<int>("forecast_discrete");
  if (tpg.GetParam<int>("forecast_univar")) {
    eval.sequence_targ[eval.n_prediction] = task->data[eval.sample + 1][0];
    if (discrete_actions) {
      int action = WrapDiscreteAction(eval);
      eval.sequence_pred[eval.n_prediction] =
          task->uniq_discrete_univars_[action];
    } else
      eval.sequence_pred[eval.n_prediction] = WrapContinuousActionSigmoid(eval);
  } else {
    if (tpg.GetParam<string>("action_dim") == "1x3") {
      auto targ = task->data[eval.sample + 1];
      auto act = WrapVectorActionSigmoid(eval);
      for (size_t var = 0; var < act.size(); var++) {
        eval.sequence_targ[eval.n_prediction * act.size() + var] = targ[var];
        eval.sequence_pred[eval.n_prediction * act.size() + var] = targ[var];
      }
    } else {  // if (tpg.GetParam<string>("action_dim") == "1x1"){
      auto targ = task->data[eval.sample + 1];
      vector<double> pred;
      size_t y = size_t(tpg.GetParam<int>("predict_var"));
      for (size_t var = 0; var < targ.size(); var++) {
        eval.sequence_targ[eval.n_prediction * targ.size() + var] = targ[var];
        if (var == y) {
          eval.sequence_pred[eval.n_prediction * targ.size() + var] =
              WrapContinuousActionSigmoid(eval);
           pred.push_back(WrapContinuousActionSigmoid(eval));   
        } else {
          eval.sequence_pred[eval.n_prediction * targ.size() + var] = targ[var];
          pred.push_back(targ[var]);
        }
      }
      cerr << "t:" << vecToStr(targ) << " p:" << vecToStr(pred) << endl;
    }
  }
}

void InitRecusiveForecastObs(TPG &tpg, EvalStruct &eval) {
  eval.obs = new state(tpg.n_input_[tpg.GetState("active_task")]);
  eval.obs_list.assign(tpg.n_input_[tpg.GetState("active_task")], 1.0);
  eval.obs_vec.assign(tpg.n_input_[tpg.GetState("active_task")], 1.0);
}

void PrepareRecusiveForecastObs(TPG &tpg, EvalStruct &eval, bool prime) {
  RecursiveForecast *task = dynamic_cast<RecursiveForecast *>(eval.task);
  bool discrete_actions = tpg.GetParam<int>("forecast_discrete");
  if (prime) {  // prime
    if (tpg.GetParam<int>("forecast_univar")) {
      eval.obs_list.push_back(task->data[eval.sample][0]);
      eval.obs_list.pop_front();
      std::copy(eval.obs_list.begin(), eval.obs_list.end(),
                eval.obs_vec.begin());
      eval.obs->Set(eval.obs_vec);
    } else {
      eval.obs->Set(task->data[eval.sample]);
    }
  } else {  // predict
    if (tpg.GetParam<int>("forecast_univar")) {
      if (discrete_actions) {
        int action = WrapDiscreteAction(eval);
        eval.obs_list.push_back(
            task->uniq_discrete_univars_[action]);  // Prev action
      } else
        eval.obs_list.push_back(
            WrapContinuousActionSigmoid(eval));  // Prev action
      eval.obs_list.pop_front();
      std::copy(eval.obs_list.begin(), eval.obs_list.end(),
                eval.obs_vec.begin());
      eval.obs->Set(eval.obs_vec);
    } else {
      std::vector<double> v;
      if (tpg.GetParam<string>("action_dim") == "1x3") {
        v = WrapVectorActionSigmoid(eval);  // Prev action
      } else {  // if (tpg.GetParam<string>("action_dim") == "1x1"){
        v = eval.n_prediction == 0 ? task->data[eval.sample - 1]
                                   : task->data[eval.sample];
        v[tpg.GetParam<int>("predict_var")] =
            WrapContinuousActionSigmoid(eval);  // Prev action
      }
      eval.obs->Set(v);
    }
  }
}

/******************************************************************************/
void EvalRecursiveForecast(TPG &tpg, EvalStruct &eval) {
  RecursiveForecast *task = dynamic_cast<RecursiveForecast *>(eval.task);
  eval.n_prediction = 0;
  InitRecusiveForecastObs(tpg, eval);

  // Prime
  eval.sample = task->t_start[tpg.GetState("phase")][eval.episode];
  for (int i = 0; i < task->n_prime_ - 1; i++) {
    PrepareRecusiveForecastObs(tpg, eval, true);
    // Execute graph
    eval.leafProgram = tpg.getAction(
        eval.tm, eval.obs, true, eval.visitedTeams, eval.decision_instructions,
        eval.task->step, eval.teamPath, tpg.rngs_[AUX_SEED], false);
    eval.sample++;
  }
  // Predict
  for (eval.n_prediction = 0;
       eval.n_prediction < task->n_predict_[tpg.GetState("phase")];
       eval.n_prediction++) {
    PrepareRecusiveForecastObs(tpg, eval, false);
    // Execute graph
    eval.leafProgram = tpg.getAction(eval.tm, eval.obs, true, eval.visitedTeams,
                                     eval.decision_instructions, task->step,
                                     eval.teamPath, tpg.rngs_[AUX_SEED], false);

    SaveRecursiveForecast(tpg, eval);
    eval.sample++;
    AccumulateStepStats(eval);
  }
  delete eval.obs;
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
      eval.task = tasks[tpg.GetState("active_task")];

      if (eval.task->eval_type_ == "RecursiveForecast") {
        RecursiveForecast *task = dynamic_cast<RecursiveForecast *>(eval.task);
        if (tpg.GetParam<int>("forecast_univar")) {
          eval.sequence_targ.resize(task->n_predict_[tpg.GetState("phase")]);
          eval.sequence_pred.resize(task->n_predict_[tpg.GetState("phase")]);
        } else {
          eval.sequence_targ.resize(task->n_predict_[tpg.GetState("phase")] *
                                    tpg.n_input_[tpg.GetState("active_task")]);
          eval.sequence_pred.resize(task->n_predict_[tpg.GetState("phase")] *
                                    tpg.n_input_[tpg.GetState("active_task")]);
        }
      }
      eval.evalResult = "";
      for (auto tm : eval.teams) {
        eval.tm = tm;
        tpg.MarkEffectiveCode(eval.tm);
        for (eval.episode = 0; eval.episode < eval.tm->_n_eval;
             eval.episode++) {
          tpg.rngs_[AUX_SEED].seed(eval.episode);
          eval.tm->InitMemory(tpg._teamMap, tpg.HaveParam("p_bid_mu_const"));
          evaluator_map[eval.task->eval_type_](tpg, eval);
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
  eval.task = tasks[tpg.GetState("active_task")];
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
  eval.task->reset(tpg.rngs_[AUX_SEED]);
  state *obs = new state(tpg.n_input_[tpg.GetState("active_task")]);
  obs->Set(eval.task->GetObsVec(eval.partially_observable));
  while (!eval.task->terminal()) {
    eval.leafProgram = tpg.getAction(
        eval.tm, obs, true, eval.visitedTeams, eval.decision_instructions,
        eval.task->step, eval.teamPath, tpg.rngs_[AUX_SEED], false);
    eval.n_prediction++;
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
        eval.task->update(WrapDiscreteAction(eval), WrapContinuousAction(eval),
                          tpg.rngs_[AUX_SEED]);
    eval.runTimeStats[REWARD1_IDX] += r.r1;
    AccumulateStepStats(eval);
    obs->Set(eval.task->GetObsVec(eval.partially_observable));
  }
  for (auto p : teamUseMapPerTask[tpg.state_["active_task"]]) {
    p.second = p.second / eval.task->step;
  }
  delete obs;
}

void PrintRecursizeForecast(string filename, int t_start, int n_var,
                            vector<double> prime_samples,
                            vector<double> targets,
                            vector<double> predictions) {
  // Print csv format for quick plotting
  ofstream test_file;
  test_file.open(filename);
  // Print header
  test_file << "Time";
  for (int i = 0; i < n_var; i++) test_file << ",x" << to_string(i);
  for (int i = 0; i < n_var; i++) test_file << ",y" << to_string(i);
  test_file << endl;

  for (size_t i = 0; i < prime_samples.size() / n_var; i++) {
    test_file << t_start++;
    for (int var = 0; var < n_var; var++) {
      test_file << "," << prime_samples[i * n_var + var];
    }
    test_file << endl;
  }
  for (size_t i = 0; i < targets.size() / n_var; i++) {
    test_file << t_start++;
    for (int var = 0; var < n_var; var++) {
      test_file << std::fixed << "," << targets[i * n_var + var];
    }
    for (int var = 0; var < n_var; var++) {
      test_file << std::fixed << "," << predictions[i * n_var + var];
    }
    test_file << endl;
  }
  test_file.close();
}

/******************************************************************************/
void EvalRecursiveForecastViz(TPG &tpg, EvalStruct &eval,
                              vector<map<long, double>> &teamUseMapPerTask,
                              set<team *, teamIdComp> &visitedTeamsAllTasks,
                              int &steps) {
  RecursiveForecast *task = dynamic_cast<RecursiveForecast *>(eval.task);
  eval.n_prediction = 0;
  if (tpg.GetParam<int>("forecast_univar")) {
    eval.sequence_targ.resize(task->n_predict_[tpg.GetState("phase")]);
    eval.sequence_pred.resize(task->n_predict_[tpg.GetState("phase")]);
  } else {
    eval.sequence_targ.resize(task->n_predict_[tpg.GetState("phase")] *
                              tpg.n_input_[tpg.GetState("active_task")]);
    eval.sequence_pred.resize(task->n_predict_[tpg.GetState("phase")] *
                              tpg.n_input_[tpg.GetState("active_task")]);
  }

  InitRecusiveForecastObs(tpg, eval);

  // Prime
  vector<double> prime_samples_plot;
  eval.sample =
      task->t_start[tpg.GetParam<int>("checkpoint_in_phase")][eval.episode];
  for (int i = 0; i < task->n_prime_ - 1; i++) {
    PrepareRecusiveForecastObs(tpg, eval, true);

    if (tpg.GetParam<int>("forecast_univar")) {
      prime_samples_plot.push_back(task->data[eval.sample][0]);
    } else {
      prime_samples_plot.insert(prime_samples_plot.end(),
                                task->data[eval.sample].begin(),
                                task->data[eval.sample].end());
    }

    // Execute graph
    eval.leafProgram = tpg.getAction(
        eval.tm, eval.obs, true, eval.visitedTeams, eval.decision_instructions,
        eval.task->step, eval.teamPath, tpg.rngs_[AUX_SEED], false);

    // Team user per task stats TODO(skelly): move to accumulator?
    for (auto tm : eval.visitedTeams) {
      if (teamUseMapPerTask[tpg.state_["active_task"]].find(tm->id_) ==
          teamUseMapPerTask[tpg.state_["active_task"]].end()) {
        teamUseMapPerTask[tpg.state_["active_task"]][tm->id_] = 1.0;
      } else {
        teamUseMapPerTask[tpg.state_["active_task"]][tm->id_] += 1.0;
      }
    }
    // teamUseMapPerTask[tpg.state_["active_task"]][eval.tm->id_] += 1.0;
    visitedTeamsAllTasks.insert(eval.visitedTeams.begin(),
                                eval.visitedTeams.end());
    eval.sample++;
    steps++;
  }
  // Predict
  for (eval.n_prediction = 0;
       eval.n_prediction <
       task->n_predict_[tpg.GetParam<int>("checkpoint_in_phase")];
       eval.n_prediction++) {
    PrepareRecusiveForecastObs(tpg, eval, false);

    // Execute graph
    eval.leafProgram = tpg.getAction(eval.tm, eval.obs, true, eval.visitedTeams,
                                     eval.decision_instructions, task->step,
                                     eval.teamPath, tpg.rngs_[AUX_SEED], false);
    // Team user per task stats TODO(skelly): move to accumulator?
    for (auto tm : eval.visitedTeams) {
      if (teamUseMapPerTask[tpg.state_["active_task"]].find(tm->id_) ==
          teamUseMapPerTask[tpg.state_["active_task"]].end()) {
        teamUseMapPerTask[tpg.state_["active_task"]][tm->id_] = 1.0;
      } else {
        teamUseMapPerTask[tpg.state_["active_task"]][tm->id_] += 1.0;
      }
    }
    // teamUseMapPerTask[tpg.state_["active_task"]][eval.tm->id_] += 1.0;
    visitedTeamsAllTasks.insert(eval.visitedTeams.begin(),
                                eval.visitedTeams.end());

    SaveRecursiveForecast(tpg, eval);
    eval.sample++;
    steps++;
    AccumulateStepStats(eval);
  }
  delete eval.obs;

  // TODO(spkelly): fix hard coding
  int n_var = tpg.GetParam<int>("forecast_univar") ? 1 : 3;  
  PrintRecursizeForecast(
      "tpg_" + to_string(tpg.seeds_[TPG_SEED]) + "_test_t" +
          to_string(task->t_start[tpg.GetParam<int>("checkpoint_in_phase")]
                                 [eval.episode]) +
          ".csv",
      task->t_start[tpg.GetParam<int>("checkpoint_in_phase")][eval.episode],
      n_var, prime_samples_plot,
      eval.sequence_targ, eval.sequence_pred);
}

/******************************************************************************/
// For each task, map team id to the proportion of steps in which this team was
// used vector<map<long, double>> teamUseMapPerTask;

void replayer_viz(TPG &tpg, vector<TaskEnv *> &tasks) {
  MaybeStartAnimation(tpg);
  EvalStruct eval(tpg);

  vector<map<long, double>> teamUseMapPerTask;
  teamUseMapPerTask.resize(tpg.GetState("n_task"));
  std::set<team *, teamIdComp> visitedTeamsAllTasks;

  tpg.getTeams(eval.teams, true);
  eval.evalResult = "";
  for (auto tm : eval.teams) {
    if (tm->id_ != tpg.GetParam<int>("host_to_replay")) continue;
    eval.tm = tm;
    
    tpg.MarkEffectiveCode(eval.tm);
    vector<int> steps_per_task(tpg.GetState("n_task"), 0);
    // TODO(skelly): clean up
    // for (int task = 0; task < tpg.GetState("n_task"); task++) {
    // tpg.state_["active_task"] = task;
    eval.task = tasks[tpg.GetState("active_task")];
    if (eval.animate) eval.tm->_n_eval = 1;
    else {
    eval.tm->_n_eval =
        eval.task->GetNumEval(tpg.GetParam<int>("checkpoint_in_phase"));
    }
    for (eval.episode = 0; eval.episode < eval.tm->_n_eval; eval.episode++) {
      tpg.rngs_[AUX_SEED].seed(eval.episode);
      eval.tm->InitMemory(tpg._teamMap, tpg.HaveParam("p_bid_mu_const"));

      if (eval.task->eval_type_ == "RecursiveForecast") {
        EvalRecursiveForecastViz(tpg, eval, teamUseMapPerTask,
                                 visitedTeamsAllTasks,
                                 steps_per_task[tpg.GetState("active_task")]);
      } else {
        EvalControlViz(tpg, eval, teamUseMapPerTask, visitedTeamsAllTasks,
                       steps_per_task[tpg.GetState("active_task")]);
      }
      FinalizeStepStats(tpg, eval);
    }
    // }
    tpg.printGraphDotGPTPXXI(eval.tm->id_, visitedTeamsAllTasks,
                             teamUseMapPerTask, steps_per_task);
  }
}

#endif