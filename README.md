# Tangled Program Graphs (TPG)

## Quick Start

This code is designed to be used in Linux

### 1. Install required software
From the tpg directory run:
```
sudo xargs --arg-file requirements.txt apt install
```

### 2. Set environment variables.
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

### 3. Compile tpg
From the tpg directory run:
```
scons --opt
```

### 4. Run an experiment
The folder tpg/classic_control_example contains scripts to evolve policies for classic control tasks. It's possible to reproduce all experiments from this [paper](https://rdcu.be/czd3s). Parameters are set in parameters.txt. The default settings will evolve a policy for the [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) task.

To run an experiment using 4 parallel MPI processes, make *tpg/classic_control_example* your working directory and run:
```
tpg-run-mpi.sh -n 4
```

### 5. Plot results
```
tpg-plot-stats.sh
```
This should produce a pdf named *classic_control_p0.pdf* with various statistics. The first page will be a training curve looking something like this:

<img src="./classic_control_example/images/cartpole-example.png" height="400" />


