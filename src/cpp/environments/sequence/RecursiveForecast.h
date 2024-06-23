#ifndef RecursiveForecast_h
#define RecursiveForecast_h

#include <TaskEnv.h>
#define DIM 1

class RecursiveForecast : public TaskEnv {
 public:
  // Data. One vec per timestep. Supports multivariate
  vector<vector<double>> data;
  // Starting points to slice datasets. One vec each for train, val, test
  vector<vector<int>> t_start;

  int n_eval_train_;
  int n_eval_val_;
  int n_prime_;
  // Number of samples (horizon) for training, validation, test
  int n_predict_[3];

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

  RecursiveForecast(string task) {
    eval_type_ = "RecursiveForecast";
    task_ = task;
  }

  ~RecursiveForecast() {}

  int GetNumEval(int phase) { return static_cast<int>(t_start[phase].size()); }

  void PrepareData(mt19937& rng) {
    // import data
    CSVReader* reader;
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
    else if (task_ == "Pitch")
      reader = new CSVReader("./datasets/ali_pitch.csv", DIM);
    else if (task_ == "PitchBach")
      reader = new CSVReader("./datasets/ali_bach_pitch.csv", DIM);
    else {
      std::cerr << "Unrecognised RecursiveForecast task" << std::endl;
      exit(1);
    }
    data = reader->ReadData();
    delete reader;

    // start steps for priming, for each phase (train, validate, test)
    t_start.resize(3);

    if (task_ == "Sunspots" || task_ == "Mackey" || task_ == "Laser") {
      // train (original, 19 start points)
      for (int s = 0; s <= 900; s += 50) t_start[0].push_back(s);

      // validation (original, 9 start points)
      for (int s = 50; s <= 850; s += 100) t_start[1].push_back(s);

      // test (original single start point)
      t_start[2].insert(t_start[2].begin(), {950});

    } else if (task_ == "Offset" || task_ == "Duration" || task_ == "Pitch" ||
               task_ == "PitchBach") {
      // // train
      // for (size_t s = 0; s < data.size() - (n_prime_ + n_predict_[0]); s +=
      // 40)
      //   t_start[0].push_back(s);

      // // validate
      // for (size_t s = 0; s < data.size() - (n_prime_ + n_predict_[1]); s +=
      // 40)
      //   t_start[1].push_back(s);

      // // test
      // for (size_t s = 0; s < data.size() - (n_prime_ + n_predict_[2]); s +=
      // 50)
      //   t_start[2].push_back(s);

      uniform_int_distribution<int> DisTrain(
          0, data.size() - (n_prime_ + n_predict_[0]));
      uniform_int_distribution<int> DisVal(
          0, data.size() - (n_prime_ + n_predict_[1]));

      for (int i = 0; i < n_eval_train_; i++)
        t_start[0].push_back(DisTrain(rng));

      for (int i = 0; i < n_eval_val_; i++) t_start[1].push_back(DisVal(rng));

      t_start[2].insert(t_start[2].begin(),
                        {50, 150, 250, 350, 450, 550, 650, 750, 850});
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

  double GetSampleUnivar(int sample) { return data[sample + 1][0]; }
};
#endif
