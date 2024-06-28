#!/bin/bash 
#SBATCH --account=def-skelly

# cpus anywhere
#SBATCH --ntasks=41               
#SBATCH --mem-per-cpu=2G      
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
