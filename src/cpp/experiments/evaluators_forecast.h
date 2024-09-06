#ifndef evaluators_forecast_h
#define evaluators_forecast_h

#include <TPG.h>

#include "EvalData.h"

void SaveRecursiveForecast(TPG &tpg, EvalData &eval) {
    RecursiveForecast *task = dynamic_cast<RecursiveForecast *>(eval.task);
    bool discrete_actions = tpg.GetParam<int>("forecast_discrete");
    if (tpg.GetParam<int>("forecast_univar")) {
        eval.sequence_targ[eval.n_prediction] = task->data[eval.sample + 1][0];
        if (discrete_actions) {
            int action = WrapDiscreteAction(eval);
            eval.sequence_pred[eval.n_prediction] =
                task->uniq_discrete_univars_[action];
        } else
            eval.sequence_pred[eval.n_prediction] =
                WrapContinuousActionSigmoid(eval);
    } else {
        if (tpg.GetParam<string>("action_dim") == "1x3") {
            auto targ = task->data[eval.sample + 1];
            auto act = WrapVectorActionSigmoid(eval);
            for (size_t var = 0; var < act.size(); var++) {
                eval.sequence_targ[eval.n_prediction * act.size() + var] =
                    targ[var];
                eval.sequence_pred[eval.n_prediction * act.size() + var] =
                    targ[var];
            }
        } else {  // if (tpg.GetParam<string>("action_dim") == "1x1"){
            auto targ = task->data[eval.sample + 1];
            vector<double> pred;
            size_t y = size_t(tpg.GetParam<int>("predict_var"));
            for (size_t var = 0; var < targ.size(); var++) {
                eval.sequence_targ[eval.n_prediction * targ.size() + var] =
                    targ[var];
                if (var == y) {
                    eval.sequence_pred[eval.n_prediction * targ.size() + var] =
                        WrapContinuousActionSigmoid(eval);
                    pred.push_back(WrapContinuousActionSigmoid(eval));
                } else {
                    eval.sequence_pred[eval.n_prediction * targ.size() + var] =
                        targ[var];
                    pred.push_back(targ[var]);
                }
            }
            cerr << "t:" << vecToStr(targ) << " p:" << vecToStr(pred) << endl;
        }
    }
}

void InitRecusiveForecastObs(TPG &tpg, EvalData &eval) {
    eval.obs = new state(tpg.n_input_[tpg.GetState("active_task")]);
    eval.obs_list.assign(tpg.n_input_[tpg.GetState("active_task")], 1.0);
    eval.obs_vec.assign(tpg.n_input_[tpg.GetState("active_task")], 1.0);
}

void PrepareRecursiveForecastObs(TPG &tpg, EvalData &eval, bool prime) {
    RecursiveForecast *task = dynamic_cast<RecursiveForecast *>(eval.task);
    bool discrete_actions = tpg.GetParam<int>("forecast_discrete");
    if (prime) {  // prime
        if (tpg.GetParam<int>("forecast_univar")) {
            eval.obs_list.push_back(task->data[eval.sample][0]);
            eval.obs_list.pop_front();
            std::copy(eval.obs_list.begin(), eval.obs_list.end(),
                      eval.obs_vec.begin());

            // // TODO(skelly): debugging obs
            // double c = 1;
            // for (size_t ov = 0; ov < eval.obs_vec.size(); ov++) {
            //   eval.obs_vec[ov] = c;
            //   c += 1.0;
            // }
            // cerr << "obs " << vecToStr(eval.obs_vec) << endl;

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

            // // TODO(skelly): debugging obs
            // double c = 1;
            // for (size_t ov = 0; ov < eval.obs_vec.size(); ov++) {
            //   eval.obs_vec[ov] = c;
            //   c += 1.0;
            // }
            // cerr << "obs " << vecToStr(eval.obs_vec) << endl;

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
void EvalRecursiveForecast(TPG &tpg, EvalData &eval) {
    RecursiveForecast *task = dynamic_cast<RecursiveForecast *>(eval.task);
    InitRecusiveForecastObs(tpg, eval);

    // Prime
    eval.sample = task->t_start[tpg.GetState("phase")][eval.episode];
    for (int i = 0; i < task->n_prime_ - 1; i++) {
        PrepareRecursiveForecastObs(tpg, eval, true);
        // Execute graph
        eval.program_out = tpg.getAction(
            eval.tm, eval.obs, true, eval.teams_visited, eval.instruction_count,
            eval.task->step, eval.team_path, tpg.rngs_[AUX_SEED], false);
        eval.sample++;
    }
    // Predict
    for (eval.n_prediction = 0;
         eval.n_prediction < task->n_predict_[tpg.GetState("phase")];
         eval.n_prediction++) {
        PrepareRecursiveForecastObs(tpg, eval, false);
        // Execute graph
        eval.program_out = tpg.getAction(
            eval.tm, eval.obs, true, eval.teams_visited, eval.instruction_count,
            task->step, eval.team_path, tpg.rngs_[AUX_SEED], false);
        SaveRecursiveForecast(tpg, eval);
        eval.sample++;
        eval.AccumulateStepData();
    }
    delete eval.obs;
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
void EvalRecursiveForecastViz(TPG &tpg, EvalData &eval,
                              vector<map<long, double>> &teamUseMapPerTask,
                              set<team *, teamIdComp> &teams_visitedAllTasks,
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
        PrepareRecursiveForecastObs(tpg, eval, true);

        if (tpg.GetParam<int>("forecast_univar")) {
            prime_samples_plot.push_back(task->data[eval.sample][0]);
        } else {
            prime_samples_plot.insert(prime_samples_plot.end(),
                                      task->data[eval.sample].begin(),
                                      task->data[eval.sample].end());
        }

        // Execute graph
        eval.program_out = tpg.getAction(
            eval.tm, eval.obs, true, eval.teams_visited, eval.instruction_count,
            eval.task->step, eval.team_path, tpg.rngs_[AUX_SEED], false);

        // Team user per task stats TODO(skelly): move to accumulator?
        for (auto tm : eval.teams_visited) {
            if (teamUseMapPerTask[tpg.state_["active_task"]].find(tm->id_) ==
                teamUseMapPerTask[tpg.state_["active_task"]].end()) {
                teamUseMapPerTask[tpg.state_["active_task"]][tm->id_] = 1.0;
            } else {
                teamUseMapPerTask[tpg.state_["active_task"]][tm->id_] += 1.0;
            }
        }
        // teamUseMapPerTask[tpg.state_["active_task"]][eval.tm->id_] += 1.0;
        teams_visitedAllTasks.insert(eval.teams_visited.begin(),
                                     eval.teams_visited.end());
        eval.sample++;
        steps++;
    }
    // Predict
    for (eval.n_prediction = 0;
         eval.n_prediction <
         task->n_predict_[tpg.GetParam<int>("checkpoint_in_phase")];
         eval.n_prediction++) {
        PrepareRecursiveForecastObs(tpg, eval, false);

        // Execute graph
        eval.program_out = tpg.getAction(
            eval.tm, eval.obs, true, eval.teams_visited, eval.instruction_count,
            task->step, eval.team_path, tpg.rngs_[AUX_SEED], false);
        // Team user per task stats TODO(skelly): move to accumulator?
        for (auto tm : eval.teams_visited) {
            if (teamUseMapPerTask[tpg.state_["active_task"]].find(tm->id_) ==
                teamUseMapPerTask[tpg.state_["active_task"]].end()) {
                teamUseMapPerTask[tpg.state_["active_task"]][tm->id_] = 1.0;
            } else {
                teamUseMapPerTask[tpg.state_["active_task"]][tm->id_] += 1.0;
            }
        }
        // teamUseMapPerTask[tpg.state_["active_task"]][eval.tm->id_] += 1.0;
        teams_visitedAllTasks.insert(eval.teams_visited.begin(),
                                     eval.teams_visited.end());

        SaveRecursiveForecast(tpg, eval);
        eval.sample++;
        steps++;
        eval.AccumulateStepData();
    }
    delete eval.obs;

    // TODO(spkelly): fix hard coding
    int n_var = tpg.GetParam<int>("forecast_univar") ? 1 : 3;
    PrintRecursizeForecast(
        "tpg_seed_" + to_string(tpg.seeds_[TPG_SEED]) + "_task_" +
            to_string(tpg.state_["active_task"]) + "_test_t" +
            to_string(task->t_start[tpg.GetParam<int>("checkpoint_in_phase")]
                                   [eval.episode]) +
            ".csv",
        task->t_start[tpg.GetParam<int>("checkpoint_in_phase")][eval.episode],
        n_var, prime_samples_plot, eval.sequence_targ, eval.sequence_pred);
}

#endif