#ifndef tpg_eval_tsmt_mpi_h
#define tpg_eval_tsmt_mpi_h
#include <boost/mpi.hpp>
#include <boost/algorithm/string.hpp>
namespace mpi = boost::mpi;

void eval_tsmt_mpi_main(long t, TPG &tpg, ostringstream &os, mpi::communicator &world){
   string my_string = "MAIN";
   map <long, team*> rootTeams; vector < team* > teamsToEval; vector < team* > teamsThisEval;
   vector < string > all_strings; vector <string> splitString; string resultLine; string s;
   vector <long> active; vector <behaviourType> tmpBehavSeq; vector <double> r_runTimeStats;
   set < team*, teamIdComp > eliteTeams;

   int worldSizePerTask = (world.size() - 1)/tpg.numTask();
   int evaluator = 1;
   size_t numTeamsToEval = 0;
   for (size_t task = 0; task < tpg.numTask(); task++){
      tpg.activeTask(task);
      //assign policies to evaluators
      tpg.getTeams(rootTeams,true);
      teamsToEval.clear(); teamsThisEval.clear();

      if (tpg.phase() == _TEST_PHASE){
         tpg.setEliteTeams(t, _VALIDATION_PHASE, false);
         tpg.getEliteTeams(eliteTeams);
         for (auto it = eliteTeams.begin(); it != eliteTeams.end(); it++)
            if ((*it)->numOutcomes(tpg.phase(), tpg.activeTask()) == 0){
               teamsToEval.push_back(*it);
               numTeamsToEval++;
            }
      }
      else{
         for (auto it = rootTeams.begin(); it != rootTeams.end(); it++)
            if (it->second->numOutcomes(tpg.phase(),tpg.activeTask()) == 0){
               teamsToEval.push_back(it->second);
               numTeamsToEval++;
            }
      }

      size_t teamsPerEvaluator = teamsToEval.size()/worldSizePerTask;
      size_t remainder = teamsToEval.size() % worldSizePerTask;

      for (auto it = teamsToEval.begin(); it != teamsToEval.end(); it++){
         teamsThisEval.push_back(*it);
         if ((remainder > 0 && teamsThisEval.size() == teamsPerEvaluator+1) ||
               (remainder == 0 && teamsThisEval.size() == teamsPerEvaluator) ||
               next(it) == teamsToEval.end()){
            s = "";
            tpg.writeCheckpoint(s, teamsThisEval);
            world.send(evaluator++, 0, s);
            teamsThisEval.clear();
            if (remainder > 0) remainder--;
         }
      }
   }
   os << "t " << t << " phs " << tpg.phase() << " tToEvl " << numTeamsToEval << endl;
   //no more teams to evaluate, so let the rest of the procs know they are not needed this round
   while (evaluator <= (world.size() - 1)){
      s = "x";
      world.send(evaluator++, 0, "x");
   }

   //collect evaluation result from each evaluator
   all_strings.clear();
   gather(world, my_string, all_strings, 0);
   for (int proc = 1; proc < world.size(); proc++){
      if (!all_strings[proc].empty()){
         istringstream f(all_strings[proc]);
         while(getline(f, resultLine)){
            split(resultLine, ':', splitString);
            //parse the string
            size_t s = 0;
            long rslt_id = atol(splitString[s++].c_str());
            tpg.activeTask(atoi(splitString[s++].c_str()));
            for (size_t i = 0; i < tpg.numPointAuxDouble(); i++)
               r_runTimeStats.push_back(atof(splitString[s++].c_str()));
            while (s < splitString.size())
               active.push_back(atol(splitString[s++].c_str()));
            rootTeams[rslt_id]->updateActiveMembersFromIds(active);
            tpg.setOutcome(rootTeams[rslt_id], tmpBehavSeq, r_runTimeStats, tpg.phase(), t);
            active.clear();
            r_runTimeStats.clear();
         }
      }
   }
}


void eval_tsmt_mpi_sub(TPG &tpg, 
      mpi::communicator &world, 
      vector < vector < vector < double > > > &data, 
      vector< vector<int> > & pis,
      int horizon) { 
   state *worldState = new state(tpg.dim());
   vector <int> *pisEval;
   deque<double> recursizeSeriesQ;
   vector<double> runTimeStats; runTimeStats.reserve(tpg.numPointAuxDouble()); runTimeStats.resize(tpg.numPointAuxDouble());
   vector <learner*> winningLearners;
   long decisionInstructions;
   vector < set <long> > decisionFeatures;
   vector < set <memory*, memoryIdComp> > decisionMemories;

   int atomicAction = 0;
   learner * atomicLearner;
   set <team*, teamIdComp> visitedTeams;
   vector <team*> teamPath;
   vector <team*> teams;
   string evalResult = "";

   string checkpointString = "";
   while (checkpointString.compare("done") != 0){
      //receive from p0
      //checkpointString = "";
      if (!tpg.replay())
         world.recv(0, 0, checkpointString);
      evalResult = "";
      set < learner*, learnerIdComp > active;
      if (tpg.replay() || (checkpointString.compare("x") != 0 && checkpointString.compare("done") != 0)){
         if (!tpg.replay())
            tpg.readCheckpoint(-1, _TRAIN_PHASE, -1, true, checkpointString);
         pisEval = &pis[tpg.phase()];
         tpg.getTeams(teams,true);
         for (size_t i = 0; i < teams.size(); i++){
            tpg.markEffectiveCode(teams[i], true);
            fill(runTimeStats.begin(), runTimeStats.end(), 0);
            
            int numScoredPrediction = 0;
            for (size_t tI = 0;  tI < pisEval->size(); tI++){
               tpg.clearMemory();
               recursizeSeriesQ.clear();
               int sample = 0;
               int step = 0;

               //pack embedding
               while (sample < tpg.dim()){
                  recursizeSeriesQ.push_back(data[tpg.activeTask()][(*pisEval)[tI] + sample][0]);
                  sample++;
               }

               //prime
               int numSamplesPrime = 50;
               while (sample < numSamplesPrime - horizon){
                  worldState->setState(recursizeSeriesQ);
                  atomicLearner = tpg.getAction(teams[i], worldState, true, visitedTeams, decisionInstructions, step);
                  recursizeSeriesQ.push_back(data[tpg.activeTask()][(*pisEval)[tI] + sample][0]);
                  recursizeSeriesQ.pop_front();
                  sample++;
                  step++;
               }

               //recursize series prediction
               int numSamplesEval = tpg.phase() == _TEST_PHASE || tpg.phase() == _VALIDATION_PHASE ? 100 : 50;
               while (sample < numSamplesPrime + numSamplesEval - horizon){
                  worldState->setState(recursizeSeriesQ);
                  if (tpg.replay())
                     atomicLearner = tpg.getAction(teams[i], worldState, true, visitedTeams, decisionInstructions, step, winningLearners, decisionFeatures, decisionMemories, teamPath);
                  else
                     atomicLearner = tpg.getAction(teams[i], worldState, true, visitedTeams, decisionInstructions, step);
                  recursizeSeriesQ.push_back((atomicLearner->memGet())->getMem()[0]);
                  recursizeSeriesQ.pop_front();
                  if (tpg.phase() != _VALIDATION_PHASE || (tpg.phase() == _VALIDATION_PHASE && sample >= (numSamplesPrime  * 2) - horizon)){
                     runTimeStats[tpg.hostFitnessMode()] += pow((data[tpg.activeTask()][(*pisEval)[tI] + sample + horizon][0] - (atomicLearner->memGet())->getMem()[0]),2);//SSE
                     runTimeStats[tpg.numFitMode()] += visitedTeams.size();
                     runTimeStats[tpg.numFitMode() + 1] += decisionInstructions;
                     numScoredPrediction++;
                     if (tpg.replay()){
                        cout << "TPG::test id " << teams[i]->id() << " sample " << sample << " input " << data[tpg.activeTask()][(*pisEval)[tI] + sample][0];
                        cout << " target " << data[tpg.activeTask()][(*pisEval)[tI] + sample + horizon][0] << " prediction " << (atomicLearner->memGet())->getMem()[0];
                        cout << " visitedTeams " << visitedTeams.size();
                        cout << " activeNodes";
                        for (auto it = visitedTeams.begin(); it != visitedTeams.end(); it++)
                           cout << " h" << (*it)->id();
                        for (auto it = winningLearners.begin(); it != winningLearners.end(); it++)
                           cout << " s" << (*it)->id();
                        cout << " decisionInstructions " << decisionInstructions << " decisionFeatures";
                        for (auto it = decisionFeatures.begin(); it != decisionFeatures.end(); it++)
                           cout << " " << (*it).size();
                        cout << " decisionMemories";
                        //get all memories used this decision
                        set < memory* > mems;
                        for (auto it = decisionMemories.begin(); it != decisionMemories.end(); it++)
                           mems.insert((*it).begin(), (*it).end());

                        //get uniq read/write times as int (ignoring graph depth)
                        set <int> memActiveReadT;
                        set <int> memActiveWriteT;
                        vector <double> memReadTimes; 
                        vector <double> memWriteTimes;
                        for (int mr = 0; mr < MEMORY_REGISTERS; mr++){
                           memReadTimes.push_back(-1); 
                           memWriteTimes.push_back(-1); 
                        }
                        for (auto it = mems.begin(); it != mems.end(); it++){
                           (*it)->getActiveReadTime(memReadTimes);
                           (*it)->getActiveWriteTime(memWriteTimes);

                           for (auto rwit = memReadTimes.begin(); rwit != memReadTimes.end(); rwit++)
                              if (((int)(*rwit)) >= 0)//ignore untouched registers (-1)
                                 memActiveReadT.insert((int)(*rwit)); 
                           for (auto rwit = memWriteTimes.begin(); rwit != memWriteTimes.end(); rwit++)
                              if (((int)(*rwit)) >= 0)//ignore untouched registers (-1)
                                 memActiveWriteT.insert((int)(*rwit));
                        }
                        cout << " step " << step;
                        cout << " Dsize " << memActiveReadT.size();
                        vector<int> v;
                        for (auto it = memActiveReadT.begin(); it != memActiveReadT.end(); it++)
                           v.push_back(step- (*it));

                        cout << " D " << vecToStr(v);

                        cout << " Dmax_med_min ";
                        cout << *(max_element(v.begin(), v.end()));
                        cout << " " << vecMedian(v);
                        cout << " " << *(min_element(v.begin(), v.end()));
                        cout << " v " << vecToStr(v);
                        //cout << " readTimes";
                        //for (auto it = memActiveReadT.begin(); it != memActiveReadT.end(); it++)
                        //   cout << " " << step - (*it);
                        //cout << " writeTimes";
                        //for (auto it = memActiveWriteT.begin(); it != memActiveWriteT.end(); it++)
                        //   cout << " " << step - (*it);
                        cout << " teamPath"; 
                        for(size_t i = 0; i < teamPath.size(); i++)
                           cout << " h" << teamPath[i]->id();
                        cout << endl;
                     }//replay
                  }
                  sample++;
                  step++;
               }
            }
            if (isfinite(runTimeStats[tpg.hostFitnessMode()]))
               runTimeStats[tpg.hostFitnessMode()] = -1 * (runTimeStats[tpg.hostFitnessMode()] / numScoredPrediction);
            else{
               runTimeStats[tpg.hostFitnessMode()] = numeric_limits<double>::lowest();
               cerr << "TPG::eval infinite SSE " << " team " << teams[i]->id()  << " atomicAction " << atomicAction << endl;
            }
            runTimeStats[tpg.numFitMode()] = runTimeStats[tpg.numFitMode()] / numScoredPrediction; //mean visited teams
            runTimeStats[tpg.numFitMode() + 1] = runTimeStats[tpg.numFitMode() + 1] / numScoredPrediction; //mean instruction count (RTC)
            //if (tpg.phase() == _TEST_PHASE)
            //   runTimeStats[tpg.hostFitnessMode()] = abs(runTimeStats[tpg.hostFitnessMode()]);
            evalResult += to_string(teams[i]->id());
            evalResult += ":" + to_string(tpg.activeTask());
            for (size_t r = 0; r < runTimeStats.size(); r++)
               evalResult += ":" + to_string(runTimeStats[r]);
            teams[i]->getActiveMembersByRef(active);
            for (auto leiter = active.begin(); leiter != active.end(); leiter++)
               evalResult += ":" + to_string((*leiter)->id());
            evalResult += "\n";
            if (tpg.replay()){
               vector <behaviourType> tmpBehavSeq;
               tpg.setOutcome(teams[i], tmpBehavSeq, runTimeStats, tpg.phase(), tpg.tPickup());
            }
         }
      }
      if (tpg.replay()) break;
      mpi::gather(world, evalResult, 0);
   }
   delete worldState;
}
#endif
