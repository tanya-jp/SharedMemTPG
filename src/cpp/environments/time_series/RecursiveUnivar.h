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

  int num_samples_prime_;
  // number of samples for training, validation, test
  int num_samples_predict_[3];

  string task_;

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
    task_ = task;
  }

  ~RecursiveUnivar() {}

  int GetNumEval(int phase) { return static_cast<int>(t_start[phase].size()); }

  void PrepareData() {
    // import data
    CSVReader *reader;
    if (task_ == "Sunspots")
      reader =
          new CSVReader("./datasets/SN_ms_tot_V2.0_Nov1834-June1926.csv", DIM);
    else if (task_ == "Mackey")
      reader = new CSVReader("./datasets/Mackey-1100.csv", DIM);
    else if (task_ == "Laser")
      reader = new CSVReader("./datasets/Laser-10000-1000-2100.csv", DIM);
    else if (task_ == "Audio")
      reader = new CSVReader("./datasets/2024-05-22-ali-10sec.dat", DIM);
    else if (task_ == "Offset")
      reader = new CSVReader("./datasets/ali_offset_diff.csv", DIM);
    else if (task_ == "Duration")
      reader = new CSVReader("./datasets/ali_duration.csv", DIM);
    else {  // task_ == "Pitch"
      reader = new CSVReader("./datasets/ali_pitch.csv", DIM);
    }
    data = reader->ReadData();
    delete reader;

    // start steps for priming, for each phase (train, validate, test)
    t_start.resize(3);

    if (task_ == "Audio") {
      // train
      t_start[0].insert(t_start[0].begin(), {0, 1000, 2000, 3000, 4000, 5000,
                                             6000, 7000, 8000, 9000});
      // validate
      t_start[1].insert(t_start[1].begin(), {500, 1500, 2500, 3500, 4500, 5500,
                                             6500, 7500, 8500, 9500});
      // test
      t_start[2].insert(t_start[2].begin(), {1000});
    } else if (task_ == "Sunspots" || task_ == "Mackey" || task_ == "Laser") {
      // train (original, 19 start points)
      for (int s = 0; s <= 900; s += 50) t_start[0].push_back(s);

      // validation (original, 9 start points)
      for (int s = 50; s <= 850; s += 100) t_start[1].push_back(s);

      // test (original single start point)
      t_start[2].insert(t_start[2].begin(), {950});

    } else if (task_ == "Offset" || task_ == "Duration" || task_ == "Pitch") {
      // num_samples_prime_ = 10;
      // num_samples_predict_[0] = 10;    // train
      // num_samples_predict_[1] = 20;   // validate
      // num_samples_predict_[2] = 20;  // test

      // // train
      // for (int s = 0; s <= 800; s += 100) t_start[0].push_back(s);
      // // extra train slices
      // for (int s = 25; s <= 825; s += 100) t_start[0].push_back(s);

      // // validate
      // for (int s = 0; s <= 750; s += 150) t_start[1].push_back(s);

      // // test
      // for (int s = 50; s <= 800; s += 150) t_start[2].push_back(s);

      // train
      for (size_t s = 0;
           s < data.size() - (num_samples_prime_ + num_samples_predict_[0]);
           s += 20)
        t_start[0].push_back(s);

      // validate
      for (size_t s = 0;
           s < data.size() - (num_samples_prime_ + num_samples_predict_[1]);
           s += 20)
        t_start[1].push_back(s);  

      // test
      for (size_t s = 0;
           s < data.size() - (num_samples_prime_ + num_samples_predict_[2]);
           s += 50)
        t_start[2].push_back(s);   
    }
    cout << "time series train slices: " << t_start[0].size() << endl;
    cout << "time series validation slices: " << t_start[1].size() << endl;
    cout << "time series test slices: " << t_start[2].size() << endl;
  }

  void Normalize() {
    // normalize data in [0,1]
    double maxFeature = numeric_limits<double>::lowest();
    double minFeature = numeric_limits<double>::max();
    for (size_t sample = 0; sample < data.size(); sample++) {
      maxFeature = max(
          maxFeature, *(max_element(data[sample].begin(), data[sample].end())));
      minFeature = min(
          minFeature, *(min_element(data[sample].begin(), data[sample].end())));
    }
    for (size_t sample = 0; sample < data.size(); sample++) {
      for (size_t feature = 0; feature < data[sample].size(); feature++) {
        data[sample][feature] =
            (data[sample][feature] - minFeature) / (maxFeature - minFeature);
      }
    }
  }

  void reset(mt19937 & /*rng*/) { step = 0; }

  Results update(int sample, double prediction, mt19937 & /*rng*/) {
    step++;
    // just return the target and prediction
    return {data[sample + 1][0], prediction};  // target, prediction
  }
};
#endif
