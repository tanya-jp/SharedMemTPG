#include <TPG.h>
#include <acrobot.h>
#include <cartCentering.h>
#include <cartPole.h>
#include <mountainCar.h>
#include <mountainCarContinuous.h>
#include <pendulum.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <chrono>

#include "tpg_arg_parse.h"
#include "tpg_eval_rl_mpi.h"
#define CHECKPOINT_MOD 1000000
#define PRINT_MOD 1
// rawfitness,  mean visitedTeams, decisionInstructions, membersRunEntropy
#define NUM_POINT_AUX_DOUBLE 4
#define NUM_POINT_AUX_INT 4
#define MODES_T 50

int main(int argc, char** argv) {
  mpi::environment env(argc, argv);
  mpi::communicator world;
  TPG tpg;
  tpg.params_["id"] = -1;  // remove later
  tpg_arg_parse(tpg, argc, argv);
  tpg.setParams();
  ostringstream os;  // logging

  /* task sets ***************************************************************/
  vector<classicRLEnv*> tasks;
  string taskString = to_string(tpg.GetParam<int>("active_task"));
  if (taskString.find_first_of("1") != std::string::npos)
    tasks.push_back(new cartPole());
  if (taskString.find_first_of("2") != std::string::npos)
    tasks.push_back(new acrobot());
  if (taskString.find_first_of("3") != std::string::npos)
    tasks.push_back(new cartCentering());
  if (taskString.find_first_of("4") != std::string::npos)
    tasks.push_back(new pendulum());
  if (taskString.find_first_of("5") != std::string::npos)
    tasks.push_back(new mountainCar());
  if (taskString.find_first_of("6") != std::string::npos)
    tasks.push_back(new mountainCarContinuous());
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

  if (tpg.GetParam<int>("replay"))
    replay(tpg, tasks, world);
  else if (world.rank() == 0) {  // Master Process
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
    if (tpg.GetParam<int>("checkpoint"))
      tpg.readCheckpoint(tpg.GetParam<int>("t_pickup"),
                         tpg.GetParam<int>("checkpoint_in_phase"), -1, false,
                         "");
    else {
      tpg.initTeams();
    }

    // Main training loop.
    tpg.params_["t_start"] = 0;
    tpg.state_["t_current"] = 0;  // tpg.GetParam<int>("t_start");
    tpg.state_["phase"] = _TRAIN_PHASE;
    vector<int> taskSet;
    vector<int> S;
    for (int tsk = 0; tsk < (int)tasks.size(); tsk++) {
      taskSet.push_back(tsk);
      S.push_back(tsk);
    }
    while (tpg.GetState("t_current") <= tpg.GetParam<int>("n_generations")) {
      /* replacement *******************************************************/
      if (tpg.GetState("t_current") > tpg.GetParam<int>("t_start")) {
        startGenTeams = chrono::system_clock::now();
        tpg.genTeams();
        endGenTeams = chrono::system_clock::now() - startGenTeams;
      }

      /* evaluation ********************************************************/
      startEval = chrono::system_clock::now();
      // evaluate on all tasks
      evaluate_main(tpg, os, world, tasks, taskSet);
      endEval = chrono::system_clock::now() - startEval;

      /* selection *********************************************************/
      startSetEliteTeams = chrono::system_clock::now();
      tpg.setEliteTeams(tpg.GetState("t_current"), tpg.GetState("phase"), 0,
                        true);
      endSetEliteTeams = chrono::system_clock::now() - startSetEliteTeams;
      startSelTeams = chrono::system_clock::now();
      tpg.selTeams(
          tpg.GetState("t_current"), true,
          tpg.GetState("t_current") > 1
              ? floor(
                    chrono::duration_cast<chrono::milliseconds>(endGen).count())
              : 0);  // also does some reporting
      endSelTeams = chrono::system_clock::now() - startSelTeams;

      /* accounting and reporting ******************************************/
      startReport = chrono::system_clock::now();
      if (tpg.GetState("t_current") % tpg.GetParam<int>("test_mod") == 0) {
        tpg.state_["phase"] = _TEST_PHASE;
        evaluate_main(tpg, os, world, tasks, S);
        tpg.setEliteTeams(tpg.GetState("t_current"), _TEST_PHASE,
                          tpg.GetParam<int>("fit_mode"), true);
        if (tpg.GetParam<int>("write_checkpoints")) {
          // checkpoint single elite program graph in each dimension
          tpg.writeCheckpoint(tpg.GetState("t_current"), true);
        }
        tpg.state_["phase"] = _TRAIN_PHASE;
        // if (std::any_cast<int>(tpg.params_["write_checkpoints"])) {
        //   tpg.printPhyloGraphDot(tpg.getEliteTeam(
        //       allTaskString, tpg.hostFitnessMode(), _TEST_PHASE));
        // }
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
      os << endGen.count() -
                (endEval.count() + endGenTeams.count() +
                 endSetEliteTeams.count() + endSelTeams.count() +
                 endChkp.count() + endReport.count());
      os << " tskS " << vecToStrNoSpace(taskSet);
      os << endl;
      tpg.printOss(os);

      startGen = chrono::system_clock::now();
      if (tpg.GetState("t_current") % PRINT_MOD == 0) tpg.printOss();
      tpg.state_["t_current"]++;
    }
    for (int ev = 1; ev <= world.size() - 1; ev++) {
      string d = "done";
      world.send(ev, 0, d);
    }
    tpg.printOss();
    cout << "Goodbye cruel world." << world.rank() << endl;
  } else {  // Evaluator Process
    evaluate_sub(tpg, world, tasks);
  }
  tpg.finalize();
  for (size_t tsk = 0; tsk < tasks.size(); tsk++) delete tasks[tsk];
  tasks.clear();
  return 0;
}
