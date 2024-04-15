#include <TPG.h>
#include "tpg_arg_parse.h"
#include "tpg_eval_tsmt_mpi.h"
#define TEST_MOD 10
#define PRINT_MOD 1
#define DIM 1
#define NUM_ATOMIC 2 //min
#define HORIZON 1
//  #define HURDLE_GEN 1000
//  #define HURDLE_THRESHOLD -0.02
//  #define HURDLE_SEED_INC 100

int main(int argc, char** argv) {
   mpi::environment env(argc, argv);
   mpi::communicator world;
   TPG tpg(-1);
   tpg_arg_parse_001(tpg, argc, argv);
   ostringstream os; //logging

   /* allProc: task sets ********************************************************************************/
   int activeTask , prevActiveTask;
   vector < int > tasks;
   if (tpg.replay() || tpg.activeTask() == -1) tasks = {0,1,2};
   else if (tpg.activeTask() == 0) tasks = {0};
   else if (tpg.activeTask() == 1) tasks = {1};
   else if (tpg.activeTask() == 2) tasks = {2}; 
   else if (tpg.activeTask() == 3) tasks = {0,1};
   else if (tpg.activeTask() == 4) tasks = {0,2};
   else if (tpg.activeTask() == 5) tasks = {1,2};
   tpg.numTask(tasks.size());
   tpg.numFitMode(1);
   tpg.dim(DIM);
   tpg.numPointAuxDouble(tpg.numFitMode() + 2); //fitness + mean visitedTeams, decisionInstructions);
   tpg.numAtomicActions(NUM_ATOMIC);
   tpg.phase(_TRAIN_PHASE);
   tpg.setParams(world.rank() == 0);

   /* allProc: data import ********************************************************************************/
   vector < vector < vector < double > > > data;
   CSVReader *reader;
   for (size_t o = 0; o < tasks.size(); o++){
      if (tasks[o] == 0) reader = new CSVReader("../datasets/SN_ms_tot_V2.0_Nov1834-June1926.csv", DIM);
      else if (tasks[o] == 1) reader = new CSVReader("../datasets/Mackey-1100.csv", DIM);
      else reader = new CSVReader("../datasets/Laser-10000-1000-2100.csv", DIM);
      data.push_back(reader->getData());
      delete reader;
   }
   /* allProc: data normalize *****************************************************************************/
   for (size_t d = 0; d < data.size(); d++){
      double maxFeature = numeric_limits<double>::lowest();
      double minFeature = numeric_limits<double>::max();
      for (size_t sample = 0; sample < data[d].size(); sample++){
         maxFeature = max(maxFeature, *(max_element(data[d][sample].begin(), data[d][sample].end())));
         minFeature = min(minFeature, *(min_element(data[d][sample].begin(), data[d][sample].end())));
      }
      for (size_t sample = 0; sample < data[d].size(); sample++)
         for (size_t feature = 0; feature < data[d][sample].size(); feature++)
            data[d][sample][feature] = (data[d][sample][feature] - minFeature) / (maxFeature - minFeature);
   }
   /* allProc: evaluation prime starting points ****************************************************************************/
   vector< vector <int> > pis {
      {0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900}, //train
         {50, 150, 250, 350, 450, 550, 650, 750, 850}, //valid
         {950} //test
   };

   /* master process *****************************************************************************/
   if (world.rank() == 0) {
      mt19937 rng(tpg.seed());
      string my_string = "MAIN";
      uniform_int_distribution<int> disTS(0, tpg.numTask() - 1);
      activeTask = prevActiveTask = disTS(rng);

      //time logging
      auto startGen = chrono::system_clock::now(); chrono::duration<double> endGen = chrono::system_clock::now()-startGen;;
      auto startInit = chrono::system_clock::now(); chrono::duration<double> endInit;
      auto startGenTeams = chrono::system_clock::now(); chrono::duration<double> endGenTeams;
      auto startSelTeams = chrono::system_clock::now(); chrono::duration<double> endSelTeams;
      auto startEval = chrono::system_clock::now(); chrono::duration<double> endEval;
      auto startChkp = chrono::system_clock::now(); chrono::duration<double> endChkp;
      auto startReport = chrono::system_clock::now(); chrono::duration<double> endReport;

      set < team*, teamIdComp > eliteTeams;
      //long prevEliteTeamID = -1;
      map <long, team*> rootTeams; vector < team* > teamsToEval; vector < team* > teamsThisEval;
      vector < string > all_strings; vector <string> splitString; string resultLine; string s;
      vector <long> active; vector <behaviourType> tmpBehavSeq; vector <double> r_rewards;
      tmpBehavSeq.push_back(0.0);
      //initialization
      if (tpg.checkpoint()) 
         tpg.readCheckpoint(tpg.tPickup(), tpg.checkpointInPhase(), -1, false, "");
      else{ 
         startInit = chrono::system_clock::now(); tpg.initTeams(rng); endInit = chrono::system_clock::now()-startInit; }
      if (tpg.replay()){
         tpg.phase(_TEST_PHASE);
         for (size_t o = 0; o < tasks.size(); o++){
            tpg.hostFitnessMode(o);
            eval_tsmt_mpi_main(tpg.tPickup(), tpg, os, world);
         }
      }
      else{
         team *eliteTeamMT = tpg.getMrootBegin();
         team *prevEliteTeamMT = tpg.getMrootBegin();
         /* main training loop. *******************************************************************/
         long t = tpg.tStart();
         while (t <= tpg.tMain()){
            /* task switching *********************************************************************/
            if (tpg.numTask() > 1 && t % tpg.taskSwitchMod() == 0){
               do { activeTask = disTS(rng); } while (activeTask == prevActiveTask);
               prevActiveTask = activeTask;
            }
            /* replacement ************************************************************************/
            startGenTeams = chrono::system_clock::now();
            if (tpg.numRoot() < tpg.rSize()) tpg.genTeams(t, rng);
            endGenTeams = chrono::system_clock::now()-startGenTeams;
            /* evaluation *************************************************************************/
            startEval = chrono::system_clock::now();
            //if (t % TEST_MOD == 0 && chrono::duration_cast<chrono::milliseconds>(endGen).count() <= 3 * tpg.maxGenMillis()){
            if (t % TEST_MOD == 0){
               tpg.phase(_VALIDATION_PHASE);
               eval_tsmt_mpi_main(t, tpg, os, world); 
               tpg.phase(_TEST_PHASE);
               eval_tsmt_mpi_main(t, tpg, os, world); 
               tpg.phase(_TRAIN_PHASE);
            }
            eval_tsmt_mpi_main(t, tpg, os, world);
            tpg.activeTask(activeTask);
            endEval = chrono::system_clock::now()-startEval;
            /* selection **************************************************************************/
            startSelTeams = chrono::system_clock::now();
            tpg.selTeams(t, true, t > 1 ? floor(chrono::duration_cast<chrono::milliseconds>(endGen).count()) : 0);//also does some reporting
            prevEliteTeamMT = eliteTeamMT;
            eliteTeamMT = tpg.getEliteTeamMT();
            endSelTeams = chrono::system_clock::now()-startSelTeams;
            ///* hurdle ****************************************************************************/
            //if (t == HURDLE_GEN && (tpg.getEliteTeam(0))->meanReward(0) < HURDLE_THRESHOLD){
            //   os << "hurdle restart seed " << tpg.seed() << " t " << t << " elFit " << (tpg.getEliteTeam(0))->meanReward(0) << " thr " << HURDLE_THRESHOLD << endl;
            //   tpg.printOss(os);
            //   tpg.finalize();
            //   tpg.seed(tpg.seed() + HURDLE_SEED_INC);
            //   rng.seed(tpg.seed());
            //   tpg.initTeams(rng);
            //   t = tpg.tStart();
            //   continue;
            //}
            //else if (t == HURDLE_GEN){
            //   tpg.writeCheckpoint(t, _TRAIN_PHASE, false, -1);
            //   for (int ev = 1; ev <= world.size() - 1; ev++){
            //      s = "done";
            //      world.send(ev, 0, s);
            //   }
            //   tpg.finalize();
            //   tpg.printOss();
            //   cout << "Goodbye cruel world." << endl;
            //   return 0;
            //}
            /* reporting **************************************************************************/
            startReport = chrono::system_clock::now();
            if (t == tpg.tStart() || t % tpg.rSize() == 0)
               tpg.updateMODESFilters(t, t == tpg.tStart(), false);
            endReport = chrono::system_clock::now()-startReport;
            /* checkpont **************************************************************************/
            startChkp = chrono::system_clock::now();
            if (tpg.writeCheckpoints() && eliteTeamMT->id() != prevEliteTeamMT->id())
               tpg.writeCheckpoint(t, tpg.phase(), false, -1);//pprevCheckpointT);
            endChkp = chrono::system_clock::now()-startChkp;

            endGen = chrono::system_clock::now()-startGen;

            /* print gen time ************************************************************************/

            os << setprecision(5) << fixed << "gTime t " << t << " sec " << endGen.count() << " evl " << endEval.count(); 
            os << " gTms " << endGenTeams.count() << " iTms " << endInit.count() << " sTms " << endSelTeams.count();
            os << " chkp " << endChkp.count() << " rprt " << endReport.count() << " lost " << endGen.count() -
               (endEval.count() + endGenTeams.count() + endInit.count() + endSelTeams.count() + endChkp.count() + endReport.count()) << endl;
            tpg.printOss(os);

            startGen = chrono::system_clock::now();
            if (t % PRINT_MOD == 0) tpg.printOss();
            t++;
         }
         }
         for (int ev = 1; ev <= world.size() - 1; ev++){
            s = "done";
            world.send(ev, 0, s);
         }
         tpg.finalize();
         tpg.printOss();
         cout << "Goodbye cruel world." << endl;
      }
      /* evaluator process ***************************************************************************/
      else { 
         eval_tsmt_mpi_sub(tpg, world, data, pis, HORIZON);
         tpg.finalize();
      }
      return 0;
   }

