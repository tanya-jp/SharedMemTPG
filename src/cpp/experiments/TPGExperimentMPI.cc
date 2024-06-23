#include <Acrobot.h>
#include <CartCentering.h>
#include <CartPole.h>
#include <MountainCar.h>
#include <MountainCarContinuous.h>
#include <Pendulum.h>
#include <RecursiveForecast.h>
#include <TPG.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <chrono>

#include "tpg_arg_parse.h"
#include "tpg_eval_mpi.h"
#define CHECKPOINT_MOD 1000000
#define PRINT_MOD 1
// rawfitness,  mean visitedTeams, decisionInstructions
#define NUM_POINT_AUX_DOUBLE 3
// task, phase, environment seed, internal test node id
#define NUM_POINT_AUX_INT 4
#define MODES_T 1000000000

int main(int argc, char **argv) {
  mpi::environment env(argc, argv);
  mpi::communicator world;
  TPG tpg;
  tpg.params_["id"] = -1;  // remove later
  tpg.setParams();
  tpg_arg_parse(tpg, argc, argv);

  ostringstream os;  // logging

  /****************************************************************************/
  // Read task sets from parameters and create environments.
  vector<TaskEnv *> tasks;
  stringstream ss(tpg.GetParam<string>("active_tasks"));
  while (ss.good()) {
    string substr;
    getline(ss, substr, ',');
    if (substr == "Cartpole")
      tasks.push_back(new CartPole());
    else if (substr == "Acrobot")
      tasks.push_back(new Acrobot());
    else if (substr == "CartCentering")
      tasks.push_back(new CartCentering());
    else if (substr == "Pendulum")
      tasks.push_back(new Pendulum());
    else if (substr == "Mountaincar")
      tasks.push_back(new MountainCar());
    else if (substr == "MountainCarContinuous")
      tasks.push_back(new MountainCarContinuous());
    else if (substr == "Sunspots")
      tasks.push_back(new RecursiveForecast("Sunspots"));
    else if (substr == "Mackey")
      tasks.push_back(new RecursiveForecast("Mackey"));
    else if (substr == "Laser")
      tasks.push_back(new RecursiveForecast("Laser"));
    else if (substr == "Offset")
      tasks.push_back(new RecursiveForecast("Offset"));
    else if (substr == "Duration")
      tasks.push_back(new RecursiveForecast("Duration"));
    else if (substr == "Pitch")
      tasks.push_back(new RecursiveForecast("Pitch"));
    else if (substr == "PitchBach")
      tasks.push_back(new RecursiveForecast("PitchBach"));  
    else {
      cout << "Unrecognised task:" << substr << endl;
      exit(1);
    }
    if (tasks[tasks.size() - 1]->eval_type_ == "RecursiveForecast") {
      RecursiveForecast *task =
          dynamic_cast<RecursiveForecast *>(tasks[tasks.size() - 1]);
      task->n_prime_ = tpg.GetParam<int>("forecast_prime_steps");
      task->n_predict_[0] =
          tpg.GetParam<int>("forecast_horizon_train");
      task->n_predict_[1] = tpg.GetParam<int>("forecast_horizon_val");
      task->n_predict_[2] =
          tpg.GetParam<int>("forecast_horizon_test");
      task->n_eval_train_ = tpg.GetParam<int>("forecast_n_eval_train");
      task->n_eval_val_ = tpg.GetParam<int>("forecast_n_eval_val");    
      task->PrepareData(tpg.rngs_[TPG_SEED]);
      if (tpg.GetParam<int>("forecast_normalize_data")) {
        task->Normalize();
      }
    }
  }
  // Read number of inputs per task from parameters
  ss.clear();
  ss.str(tpg.GetParam<string>("n_input"));
  while (ss.good()) {
    string substr;
    getline(ss, substr, ',');
    tpg.n_input_.push_back(std::stoi(substr));
  }

  string allTaskString = "";
  for (size_t i = 0; i < tasks.size(); i++) allTaskString += to_string(i);

  tpg.state_["n_task"] = (int)tasks.size();
  tpg.state_["active_task"] = 0;
  tpg.params_["n_point_aux_double"] = NUM_POINT_AUX_DOUBLE;
  tpg.params_["n_point_aux_int"] = NUM_POINT_AUX_INT;
  tpg.state_["phase"] = _TRAIN_PHASE;
  if (world.rank() == 0) {
    os << "world_size " << world.size() << endl;
    os << "n_task " << tpg.GetState("n_task") << endl;
  }

  // placeholders for logging stats only
  set<team *, teamIdComp> visitedTeamsAll;
  vector<set<team *, teamIdComp>> visitedTeamsAllPerTask;
  set<team *, teamIdComp> visitedTeamsAllTasks;
  map<long, double>
      teamUseMap;  // maps team to frequency of use for a particular task;
  vector<map<long, double>> teamUseMapPerTask;
  teamUseMapPerTask.reserve(tasks.size());
  teamUseMapPerTask.resize(tasks.size());

  if (world.rank() == 0) {  // Master Process
    string my_string = "MAIN";

    // time logging
    auto startGen = chrono::system_clock::now();
    chrono::duration<double> endGen = chrono::system_clock::now() - startGen;
    auto startGenTeams = chrono::system_clock::now();
    chrono::duration<double> endGenTeams =
        chrono::system_clock::now() - startGenTeams;
    auto startSetEliteTeams = chrono::system_clock::now();
    chrono::duration<double> endSetEliteTeams;
    auto startSelTeams = chrono::system_clock::now();
    chrono::duration<double> endSelTeams;
    auto startEval = chrono::system_clock::now();
    chrono::duration<double> endEval;
    auto startChkp = chrono::system_clock::now();
    chrono::duration<double> endChkp;
    auto startMODES = chrono::system_clock::now();
    chrono::duration<double> endMODES;
    auto startReport = chrono::system_clock::now();
    chrono::duration<double> endReport;

    // initialization
    if (tpg.GetParam<int>("checkpoint")) {
      tpg.readCheckpoint(tpg.GetParam<int>("t_pickup"),
                         tpg.GetParam<int>("checkpoint_in_phase"), -1, false,
                         "");
    } else {
      tpg.InitTeams();
    }

    // Main training loop.
    tpg.params_["t_start"] = 0;
    tpg.state_["t_current"] = 0;  // tpg.GetParam<int>("t_start");
    tpg.state_["phase"] = _TRAIN_PHASE;
    if (tpg.GetParam<int>("replay")) {
      tpg.state_["phase"] = _TEST_PHASE;
      tpg.state_["active_task"] = tpg.state_["replay_task"];
      replayer_viz(tpg, tasks);
    } else {
      while (tpg.GetState("t_current") <= tpg.GetParam<int>("n_generations")) {
        /* replacement *******************************************************/
        if (tpg.GetState("t_current") > tpg.GetParam<int>("t_start")) {
          startGenTeams = chrono::system_clock::now();
          tpg.GenerateNewTeams();
          endGenTeams = chrono::system_clock::now() - startGenTeams;
        }

        /* evaluation ********************************************************/
        startEval = chrono::system_clock::now();
        // evaluate on all tasks
        evaluate_main(tpg, world, tasks);
        endEval = chrono::system_clock::now() - startEval;

        /* selection *********************************************************/
        startSetEliteTeams = chrono::system_clock::now();
        tpg.SetEliteTeams(tasks);
        endSetEliteTeams = chrono::system_clock::now() - startSetEliteTeams;
        startSelTeams = chrono::system_clock::now();
        tpg.SelectTeams();
        endSelTeams = chrono::system_clock::now() - startSelTeams;

        /* accounting and reporting ******************************************/
        startReport = chrono::system_clock::now();
        if (tpg.GetState("t_current") % tpg.GetParam<int>("test_mod") == 0) {
          // validation
          tpg.state_["phase"] = _VALIDATION_PHASE;
          evaluate_main(tpg, world, tasks);
          tpg.SetEliteTeams(tasks);

          // test
          tpg.state_["phase"] = _TEST_PHASE;
          evaluate_main(tpg, world, tasks);
          tpg.SetEliteTeams(tasks);

          tpg.state_["phase"] = _TRAIN_PHASE;
        }
        endReport = chrono::system_clock::now() - startReport;

        /* MODES *************************************************************/
        startMODES = chrono::system_clock::now();
        if (tpg.GetState("t_current") == tpg.GetParam<int>("t_start") ||
            tpg.GetState("t_current") % MODES_T == 0)
          tpg.updateMODESFilters(true);
        endMODES = chrono::system_clock::now() - startMODES;

        /* checkpoint ********************************************************/
        startChkp = chrono::system_clock::now();
        if (tpg.GetParam<int>("write_train_checkpoints") &&
            tpg.GetState("t_current") % CHECKPOINT_MOD == 0) {
          tpg.writeCheckpoint(tpg.GetState("t_current"),
                              false);  // checkpoint entire pop
        }
        if (tpg.GetParam<int>("write_phylogeny")) {
          tpg.printPhyloGraphDot(tpg.getBestTeam());
        }
        endChkp = chrono::system_clock::now() - startChkp;
        endGen = chrono::system_clock::now() - startGen;

        /* print generation timing *******************************************/
        os << setprecision(5) << fixed;
        os << "gTime t " << tpg.GetState("t_current");
        os << " sec " << endGen.count();
        os << " evl " << endEval.count();
        os << " gTms " << endGenTeams.count();
        os << " elTms " << endSetEliteTeams.count();
        os << " sTms " << endSelTeams.count();
        os << " chkp " << endChkp.count();
        os << " rprt " << endReport.count();
        os << " MDS " << endMODES.count();
        os << " lost ";
        os << endGen.count() - (endEval.count() + endGenTeams.count() +
                                endSetEliteTeams.count() + endSelTeams.count() +
                                endChkp.count() + endReport.count());
        os << endl;
        tpg.printOss(os);

        startGen = chrono::system_clock::now();
        if (tpg.GetState("t_current") % PRINT_MOD == 0) tpg.printOss();
        tpg.SanityCheck();
        tpg.state_["t_current"]++;
      }
    }
    for (int ev = 1; ev <= world.size() - 1; ev++) {
      string d = "done";
      world.send(ev, 0, d);
    }
    tpg.printOss();
    cout << "Goodbye cruel world:" << world.rank() << endl;
  } else {  // Evaluator Process
    evaluator(tpg, world, tasks);
  }
  tpg.finalize();
  for (size_t tsk = 0; tsk < tasks.size(); tsk++) delete tasks[tsk];
  tasks.clear();
  return 0;
}
