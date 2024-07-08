# Recursive Forecasting Midi Music
This code reproduces results from the paper: "Towards Evolving Creative Algorithms: Musical Time Series Forecasting with Tangled Program Graphs" [pdf](../2024-07-05-ALife-Data/Towards_Evolving_Creative_Algorithms.pdf)

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

### Run 3 experiements with uniqe random seeds
To run 3 experiments, each using 5 parallel MPI processes, make tpg/recursive_music your working directory and run:
```
for seed in `seq 1 3`; do tpg-run-mpi.sh -n 5 -s $seed; done
```
The experiments will run until they finish or you kill them with:
```
tpg-kill.sh
```

### Plot results
Generate a pdf with various statistics from training (for test stats use "-p 2"):
```
tpg-plot-stats.sh -p 0
```

### Test the best policy
This script will find the signle repeat with the best test fitness, replay that agent, and plot the result. 10 tests will be run, each generating a unique csv file. Only one test (starting at timestep 200) is potted.
```
ALife2024_Figure3.sh
```

### Visualize all tests
You may plot all tests for the best agent with this command, which should produce a .pdf plot for each .csv file in the experiment folder.
```
for f in $(ls *csv); do echo $f; python $TPG_PATH/scripts/plot/tpg-plot-horizon.py $f; done
```

### Get test MSE of single best agent from each repeat (used to generate Figure 2 scatter plot)
```
ALife2024_Figure2_get_data.sh
```


### Cleanup
Delete all checkpoints and output files:
```
tpg-cleanup.sh
```

## Replay best agent with longer horizons

