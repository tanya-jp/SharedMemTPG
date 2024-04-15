#ifndef RecursiveUnivar_h
#define RecursiveUnivar_h

#include <TaskEnv.h>
#include <math.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#if !defined(CCANADA) && !defined(HPCC)
#include <GL/gl.h>
#include <GL/glut.h>
#endif

#define DIM 1

using namespace std;

class RecursiveUnivar : public TaskEnv {
 public:
  // [sample_t][variables_t]
  vector<vector<double>> data;
  vector<vector<int>> t_start;

  int num_samples_prime_ = 50;
  int num_samples_predict_[3] = {50, 100, 100};
  class CSVReader {
    string fileName;
    string delimiter;
    int dim;

   public:
    CSVReader(string fname, int d, string delm = " ") {
      fileName = fname;
      delimiter = delm;
      dim = d;
    }
    std::vector<std::vector<double>> getData() {
      ifstream file(fileName);
      std::vector<std::vector<double>> dataVec;
      string line = "";
      while (getline(file, line)) {
        std::vector<double> doubleValues(dim);  // features
        doubleValues[0] = stod(line.c_str());
        dataVec.push_back(doubleValues);
      }
      file.close();
      return dataVec;
    }
  };

  RecursiveUnivar(string task) {
    eval_type_ = "RecursiveForecast";
    max_step = 300;
    state.reserve(DIM);
    state.resize(DIM);
    state_po.reserve(DIM);
    state_po.resize(DIM);
    PrepareData(task);
  }

  ~RecursiveUnivar() {
    state.clear();
    state_po.clear();
    actionTrace.clear();
  }

  void PrepareData(string task) {
    // allProc: data import
    CSVReader *reader;
    if (task == "Sunspots")
      reader =
          new CSVReader("../datasets/SN_ms_tot_V2.0_Nov1834-June1926.csv", DIM);
    else if (task == "Mackey")
      reader = new CSVReader("../datasets/Mackey-1100.csv", DIM);
    else  // if (task == "Laser")
      reader = new CSVReader("../datasets/Laser-10000-1000-2100.csv", DIM);
    data = reader->getData();
    delete reader;

    // allProc: data normalize
    double maxFeature = numeric_limits<double>::lowest();
    double minFeature = numeric_limits<double>::max();
    for (size_t sample = 0; sample < data.size(); sample++) {
      maxFeature = max(
          maxFeature, *(max_element(data[sample].begin(), data[sample].end())));
      minFeature = min(
          minFeature, *(min_element(data[sample].begin(), data[sample].end())));
    }
    for (size_t sample = 0; sample < data.size(); sample++)
      for (size_t feature = 0; feature < data[sample].size(); feature++)
        data[sample][feature] =
            (data[sample][feature] - minFeature) / (maxFeature - minFeature);

    // allProc: evaluation prime starting points
    t_start.resize(3);  // train, validate, test
    // train
    t_start[0].insert(t_start[0].begin(),
                      {0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550,
                       600, 650, 700, 750, 800, 850, 900});
    // validation
    t_start[1].insert(t_start[1].begin(),
                      {50, 150, 250, 350, 450, 550, 650, 750, 850});
    // test
    t_start[2].insert(t_start[2].begin(), {950});
  }

  void normalizeState(bool po) { (void)po; }

  void reset(mt19937 &rng) {
    (void)rng;
    step = 0;
  }

  bool terminal() { return false; }

  double update(int sample, double prediction, mt19937 &rng) {  // fix hack
    (void)rng;
    step++;
    prediction = 1 / (1 + exp(-prediction));  // sigmoid
    double error = abs(prediction - data[sample][0]);
    return -error;
  }

  // opengl
  void display_function(int episode, int actionD, double actionC) {
    (void)actionC;
    (void)episode;
    (void)actionD;
    (void)actionC;
#if !defined(CCANADA) && !defined(HPCC)
#endif
  }
};

#endif
