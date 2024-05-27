#include <Acrobot.h>
#include <CartCentering.h>
#include <CartPole.h>
#include <MountainCar.h>
#include <MountainCarContinuous.h>
#include <Pendulum.h>
#include <RecursiveUnivar.h>
#include <TPG.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <chrono>

#include "tpg_arg_parse.h"
#include "tpg_eval_mpi.h"
#define CHECKPOINT_MOD 1000000
#define PRINT_MOD 1
// rawfitness,  mean visitedTeams, decisionInstructions, meanAbsoluteError
#define NUM_POINT_AUX_DOUBLE 4
#define NUM_POINT_AUX_INT 4
#define MODES_T 1000000000

int main(int argc, char** argv) {
  mpi::environment env(argc, argv);
  mpi::communicator world;
  TPG tpg;
  tpg.params_["id"] = -1;  // remove later
  tpg.setParams();
  tpg_arg_parse(tpg, argc, argv);
  tpg.params_["memory_rows"] = tpg.GetParam<int>("n_input");
  tpg.params_["memory_cols"] = tpg.GetParam<int>("n_input");

  ostringstream os;  // logging

  /* task sets ***************************************************************/
  vector<TaskEnv*> tasks;
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
      tasks.push_back(new RecursiveUnivar("Sunspots"));
    else if (substr == "Mackey")
      tasks.push_back(new RecursiveUnivar("Mackey"));
    else if (substr == "Laser")
      tasks.push_back(new RecursiveUnivar("Laser"));
    else if (substr == "Offset")
      tasks.push_back(new RecursiveUnivar("Offset"));
    else if (substr == "Duration")
      tasks.push_back(new RecursiveUnivar("Duration"));
    else if (substr == "Pitch")
      tasks.push_back(new RecursiveUnivar("Pitch"));
    else {
      cout << "Unrecognised task:" << substr << endl;
      exit(1);
    }
  }

  string allTaskString = "";
  for (size_t i = 0; i < tasks.size(); i++) allTaskString += to_string(i);

  tpg.params_["n_task"] = (int)tasks.size();
  tpg.state_["active_task"] = 0;
  tpg.params_["n_point_aux_double"] = NUM_POINT_AUX_DOUBLE;
  tpg.params_["n_point_aux_int"] = NUM_POINT_AUX_INT;
  tpg.state_["phase"] = _TRAIN_PHASE;
  if (world.rank() == 0) {
    os << "world_size " << world.size() << endl;
    os << "n_task " << tpg.GetParam<int>("n_task") << endl;
  }

  // placeholders for logging stats only
  set<team*, teamIdComp> visitedTeamsAll;
  vector<set<team*, teamIdComp>> visitedTeamsAllPerTask;
  set<team*, teamIdComp> visitedTeamsAllTasks;
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
    vector<int> taskSet;
    for (int tsk = 0; tsk < (int)tasks.size(); tsk++) {
      taskSet.push_back(tsk);
    }
    if (tpg.GetParam<int>("replay")) {
      tpg.state_["active_task"] = tpg.state_["replay_task"];
      // replayer(tpg, tasks);
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
        evaluate_main(tpg, world, taskSet);
        endEval = chrono::system_clock::now() - startEval;

        /* selection *********************************************************/
        startSetEliteTeams = chrono::system_clock::now();
        tpg.SetEliteTeams(true);
        endSetEliteTeams = chrono::system_clock::now() - startSetEliteTeams;
        startSelTeams = chrono::system_clock::now();
        tpg.selTeams(
            tpg.GetState("t_current"), true,
            tpg.GetState("t_current") > 1
                ? floor(chrono::duration_cast<chrono::milliseconds>(endGen)
                            .count())
                : 0);  // also does some reporting
        endSelTeams = chrono::system_clock::now() - startSelTeams;

        /* accounting and reporting ******************************************/
        startReport = chrono::system_clock::now();
        if (tpg.GetState("t_current") % tpg.GetParam<int>("test_mod") == 0) {
          
          // validation
          tpg.state_["phase"] = _VALIDATION_PHASE;
          evaluate_main(tpg, world, taskSet);
          tpg.SetEliteTeams(true);

          // test
          tpg.state_["phase"] = _TEST_PHASE;
          evaluate_main(tpg, world, taskSet);
          tpg.SetEliteTeams(true);

          if (tpg.GetParam<int>("write_checkpoints")) {
            // checkpoint single elite program graph in each dimension
            tpg.writeCheckpoint(tpg.GetState("t_current"), true);
          }
          tpg.state_["phase"] = _TRAIN_PHASE;
          if (std::any_cast<int>(tpg.params_["write_checkpoints"])) {
            tpg.printPhyloGraphDot(tpg.getBestTeam());
          }
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
        if (tpg.GetParam<int>("write_checkpoints") &&
            tpg.GetState("t_current") % CHECKPOINT_MOD == 0) {
          tpg.writeCheckpoint(tpg.GetState("t_current"),
                              false);  // checkpoint entire pop
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
        os << " tskS " << vecToStrNoSpace(taskSet);
        os << endl;
        tpg.printOss(os);

        startGen = chrono::system_clock::now();
        if (tpg.GetState("t_current") % PRINT_MOD == 0) tpg.printOss();
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
