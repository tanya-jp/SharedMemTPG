# Tangled Program Graphs (TPG)
This code reproduces results from the paper: 

Stephen Kelly, Tatiana Voegerl, Wolfgang Banzhaf, and Cedric Gondro. Evolving Hierarchical Memory-Prediction Machines in Multi-Task Reinforcement Learning. Genetic Programming and Evolvable Machines, 2021. [pdf](https://rdcu.be/czd3s)

## Quick Start

### 1. Install required software
This code is designed to be used in Linux. From the tpg directory run:
```
sudo xargs --arg-file requirements.txt apt install
```

### 2. Set environment variables
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

### 3. Compile
From the tpg directory run:
```
scons --opt
```

### 4. Run an experiment
The folder tpg/classic_control_example contains scripts to evolve policies for classic control tasks. Parameters are set in parameters.txt. The default settings will evolve a policy for the [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) task.

To run an experiment using 4 parallel MPI processes, make tpg/classic_control_example your working directory and run:
```
tpg-run-mpi.sh -n 4
```

Note that as of right now, the number of assigned processes must be greater than the number of active tasks.

### 5. Plot results
Generate classic_control_example_p0.pdf with various statistics:
```
tpg-plot-stats.sh
```
The first page will be a training curve looking something like the plot below. A fitness of 300 indicates the agent balances the pole for 300 timesteps, thus solving the task.

<img src="./classic_control_example/images/cartpole-example.png" height="300" />

### 6. Visualize the best policy's behaviour
Display an OpenGL animation of the single best policy interacting with the environment:
```
tpg-run-mpi.sh -m 1
```
 
### 7. Cleanup
Delete all checkpoints and output files:
```
tpg-cleanup.sh
```

## Experiments on the Digital Research Alliance of Canada
The Digital Research Alliance of Canada (aka "The Alliance") provide High Performance Parallel Compute (HPPC) reseources to Canada's research community. This includes servers with many parallel CPUs, GPUs, FPGAs, and more. We primarily use many CPUs.

### Resources
[Technical Documentation](https://docs.alliancecan.ca/wiki/Technical_documentation)

### Quick Start
Pick a [compute cluster](https://docs.alliancecan.ca/wiki/National_systems#Compute_clusters) to use and login via ssh. We'll use [narval](https://docs.alliancecan.ca/wiki/Narval):
```
ssh <user>@narval.alliancecan.ca
```

Set up your environment variables to automatically load when you login by adding the following to the end of your `.bash_profile` file:
```
export TPG_PATH=/home/$HOME/scratch/tpg
export PATH=$PATH:$TPG_PATH/scripts/plot
export PATH=$PATH:$TPG_PATH/scripts/run

module load \
  StdEnv/2023 scipy-stack/2023b python/3.10 \
  arrow/15.0.1 gcc/12.3 opencv/4.9.0 cmake/3.27.7 \
  eigen/3.4.0 boost-mpi/1.82.0
```

After editing `.bash_profile`, run:
```
source ~/.bash_profile
```

Move to your [scratch filesystem](https://docs.alliancecan.ca/wiki/Storage_and_file_management):
```
cd $SCRATCH
```

3. Clone this repo and cd to its root directory:
```
git clone https://gitlab.cas.mcmaster.ca/kellys32/tpg.git
cd tpg
```

4. `tpg/scripts/run/tpg-run-slurm.sh` is the [job script](https://docs.alliancecan.ca/wiki/Running_jobs) which sets parameters such as how many nodes and cpus you need and which [time limit queue](https://docs.alliancecan.ca/wiki/Job_scheduling_policies#Time_limits) you want to place your job in. In general, shorter jobs that use less resources will run sooner. See [scheduling policies](https://docs.alliancecan.ca/wiki/Job_scheduling_policies) for complete details. 

In our example, each job (experiment repeat) will use 64 cpus and we want them all on the same node, so we use an entire 64-cpu node. The default time limit is 3 hours. Our script looks like this:
```
#!/bin/bash 
#SBATCH --account=def-skelly
# single node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=0
#SBATCH --time=0-3:00  # time (DD-HH:MM)

#defaults
mode=0 #Train:0, Replay:1, Debug:2
seed=1
while getopts m:s: flag
do
   case "${flag}" in
      m) mode=${OPTARG};;
      s) seed=${OPTARG};;
   esac
done
if [ $mode -eq 0 ]; then
  srun ../build/release/cpp/experiments/TPGExperimentMPI -s $seed \
  1> tpg.$seed.$$.std 2> tpg.$seed.$$.err
fi
```

For each unique experiment, best practice is to copy the experiment directory and append a unique date, time, and git revision like this:
```
cp -r control_and_forecast/ control_and_forecast-`date +%Y-%m-%d-%H-%M-%S`-`git rev-parse --short HEAD`
```

From inside the newly created experiment directory, we run serveral experiments at once using unique seeds. Here's an example command using a for loop:
```
for i in `seq 1 3`; do sbatch ../scripts/run/tpg-run-slurm.sh -s $i; done
```

To [monitor](https://docs.alliancecan.ca/wiki/Running_jobs#Monitoring_jobs) your job use:
```
squeue -u <user>
```

To cancel a job:
```
scancel <jobid>
```

To cancel all your running jobs:
```
scancel -u <user>


**Copying data from clusters to you local computer**

You can use `scp` to copy data from the cluster to you local computer. Here's an example command to run locally:
```
scp -r skelly@narval.alliancecan.ca:/home/skelly/scratch/tpg/control_and_forecast-2024-05-21-21-27-09-294294c ./
```
The `-r` flag indicates you want to copy the directory and all its contents recursively.

