#ifndef RecursiveUnivar_h
#define RecursiveUnivar_h

#include <TaskEnv.h>

#define DIM 1

class RecursiveUnivar : public TaskEnv {
 public:
  // [sample_t][variables_t]
  vector<vector<double>> data;
  // starting points to slice datasets
  vector<vector<int>> t_start;

  const int num_samples_prime_ = 50;
  // number of samples for training, validation, test
  const int num_samples_predict_[3] = {50, 100, 100};
  class CSVReader {
    const string filename;
    const int dim;
    const string delim;

   public:
    CSVReader(string f, int d, string dlm = " ")
        : filename(f), dim(d), delim(dlm) {}
    std::vector<std::vector<double>> ReadData() {
      ifstream file(filename);
      std::vector<std::vector<double>> data_vec;
      string line = "";
      while (getline(file, line)) {
        std::vector<double> values(dim);
        values[0] = stod(line.c_str());
        data_vec.push_back(values);
      }
      file.close();
      return data_vec;
    }
  };

  RecursiveUnivar(string task) {
    eval_type_ = "RecursiveForecast";
    state.reserve(DIM);
    state.resize(DIM);
    PrepareData(task);
  }

  ~RecursiveUnivar() {}

  void PrepareData(string task) {
    // import data
    CSVReader *reader;
    if (task == "Sunspots")
      reader =
          new CSVReader("../datasets/SN_ms_tot_V2.0_Nov1834-June1926.csv", DIM);
    else if (task == "Mackey")
      reader = new CSVReader("../datasets/Mackey-1100.csv", DIM);
    else  // task == "Laser"
      reader = new CSVReader("../datasets/Laser-10000-1000-2100.csv", DIM);
    data = reader->ReadData();
    delete reader;

    // normalize data in [0,1]
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

    t_start.resize(3);  // train, validate, test

    // train
    t_start[0].insert(t_start[0].begin(),
                      {0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900});
    // validate
    t_start[1].insert(t_start[1].begin(),
                      {50, 150, 250, 350, 450, 550, 650, 750, 850});
    // test
    t_start[2].insert(t_start[2].begin(), {950});
  }

  void reset(mt19937 & /*rng*/) { step = 0; }

  Results update(int sample, double prediction, mt19937 & /*rng*/) {
    step++;
    prediction = 1 / (1 + exp(-prediction));  // sigmoid
    double se = pow(prediction - data[sample + 1][0], 2);
    double ae = abs(prediction - data[sample + 1][0]);
    // cerr << "env s " << sample << " pred " << prediction << " target " << data[sample + 1][0] << " se " << se << " ae " << ae << endl;
    return {-se, -ae};
  }
};
#endif
