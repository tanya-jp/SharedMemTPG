// TODO(skelly): clean this clean up file!
#ifndef TaskEnv_h
#define TaskEnv_h

#include <cstring>
#include <deque>
#include <random>

using namespace std;

class TaskEnv {
   public:
    string eval_type_;
    vector<double> state;     // state variables TODO(skelly): rename to obs...
    vector<double> state_po;  // state variables (partially observable)
    vector<double> actionsDiscrete;  // discrete actions map to an index into
                                     // actionDiscrete
    int previousActionDiscrete;
    double previousActionContinuous;
    double reward;
    uniform_real_distribution<> disReset;
    uniform_real_distribution<> disNoise;
    int step;
    int max_step;
    bool terminalState;
    // A min reward is useful for filtering very bad or infinitely bad rewards
    // in time series tasks
    double min_reward_;
    // bool continous_action_;
    vector<deque<double> > actionTrace;

    int n_eval_train_;
    int n_eval_validation_;
    int n_eval_test_;

    struct Results {
        double r1;
        double r2;
    };

    TaskEnv() {
        disNoise = uniform_real_distribution<>(-M_PI, M_PI);
        terminalState = false;
        actionTrace.reserve(3);
        actionTrace.resize(3);
        for (size_t i = 0; i < 200; i++) {
            actionTrace[0].push_back(0);
            actionTrace[1].push_back(0);
            actionTrace[2].push_back(0);
        }
    }
    double bound(double x, double m, double M) { return min(max(x, m), M); }
    virtual void display_function(int, int, double) {};
    vector<double> &GetObsVec(bool po) { return po ? state_po : state; }
    double GetObsVar(int var, bool po) {
        return po ? state_po[var] : state[var];
    }
    void setStateVar(int var, double v) {
        state_po[var] = v;
        state[var] = v;
    }
    void setStep(int s) { step = s; }
    inline string EvalType() const { return eval_type_; }
    virtual bool discreteActions() const { return true; }
    virtual double minActionContinuous() const { return 0.0; }
    virtual double maxActionContinuous() const { return 0.0; }
    virtual void reset(mt19937 &) { step = 0; }
    int maxStep() { return max_step; }
    void maxStep(int i) { max_step = i; }
    virtual Results update(int, double, mt19937 &) { return {0.0, 0.0}; };
    Results sim_step(std::vector<double> &action) { return {0.0, 0.0}; }
    int getStep() { return step; }
    virtual bool terminal() { return false; }
    virtual ~TaskEnv() {}
    int GetNumEval(int phase) {
        if (phase == 0)
            return n_eval_train_;
        else if (phase == 1)
            return n_eval_validation_;
        else
            return n_eval_test_;
    }
};
#endif
