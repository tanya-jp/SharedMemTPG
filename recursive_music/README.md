# Recursive Forecasting Midi Music
This code reproduces results from the paper: 

## Quick Start
This code is designed to be used in Linux. If you use Windows, you can use Windows Subsystem for Linux (WSL). You can work with WSL in Visual Studio Code by following [this tutorial](https://code.visualstudio.com/docs/remote/wsl-tutorial).

### Install required software
From the tpg directory run:
```
sudo xargs --arg-file requirements.txt apt install
```

### Set environment variables
In order to easily access tpg scripts, we must add appropriate folders to the $PATH environment variable.
To do so, add the following to *~/.profile*
```
export TPG_PATH=<YOUR_PATH_HERE>/tpg
export PATH=$PATH:$TPG_PATH/scripts/plot
export PATH=$PATH:$TPG_PATH/scripts/run
```
Then run:
```
source ~/.profile
```

### Compile
From the tpg directory run:
```
scons --opt
```

### Run an experiment
To run an experiment using 4 parallel MPI processes, make tpg/recursive_music your working directory and run:
```
tpg-run-mpi.sh -n 4
```

### Plot results
Generate a pdf with various statistics from training (for test stats use "-p 2"):
```
tpg-plot-stats.sh -p 0
```

### Test the best policy
This will reload the single best test policy from the repeat with seed 42 and run it on the test data.
A .csv file containing targets and predictions for each test will be produced.
```
tpg-run-mpi.sh -s 42 -m 1 -r 0
```

### Visualize test results
Plot the test data with this command, which should produce a .png plot for each .csv file in the experiment folder.
```
for f in $(ls *csv); do echo $f; python $TPG_PATH/scripts/plot/tpg-plot-horizon.py $f; done
```

### 7. Cleanup
Delete all checkpoints and output files:
```
tpg-cleanup.sh
```
