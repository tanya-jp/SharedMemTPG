# Tangled Program Graphs (TPG)

## Quick Start

This code is designed to be used in Linux

### Install requirements
```
sudo xargs --arg-file requirements.txt apt install
```

### Set environment variables in ~/.profile
```
export TPG_PATH=<YOUR_PATH_HERE>/tpg
export PATH=$PATH:$TPG_PATH/scripts/plot
export PATH=$PATH:$TPG_PATH/scripts/run
```

### Run an experiment
The folder tpg/classic_control contains scripts to evolve policies for classic control tasks. It's possible to reproduce all experiments from this [paper](https://rdcu.be/czd3s). Parameters are set in parameters.txt. The default settings will evolve a policy for the [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) task.

To run an experiment using 4 parallel MPI processes, first ensure alert(tpg/classic_control_example) is your working directory then run:
```
tpg-run-mpi.sh -n 4
```
(Note this assumes tpg/script/run is in your $PATH environment variable).

To plot results run:
```
tpg-plot-stats.sh
```
This should produce a pdf named aler(classic_control_p0.pdf) with various statistics. The first page will be a training curve looking something like this:
<img src="./images/cartpole-example.png" height="200" />


