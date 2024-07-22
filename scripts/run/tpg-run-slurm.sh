#!/bin/bash 
#SBATCH --account=def-skelly

# # single node
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=5
# #SBATCH --mem=0
# #SBATCH --time=0-0:30  # time (DD-HH:MM)

# cpus anywhere
#SBATCH --ntasks=21               
#SBATCH --mem-per-cpu=4G      
#SBATCH --time=0-12:00  # time (DD-HH:MM)

#defaults
mode=0 #Train:0, Replay:1, Debug:2
seed=1
experiment_name=""

while getopts m:s:e: flag
do
   case "${flag}" in
      m) mode=${OPTARG};;
      s) seed=${OPTARG};;
      e) experiment_name=${OPTARG};;
   esac
done

if [ $mode -eq 0 ]; then
   if [ -n "$experiment_name" ]; then
      # Access internet for experiment tracking
      module load httpproxy

      # Create experiment
      experiment_key=$($TPG_PATH/scripts/run/create-experiment.sh $experiment_name)

      srun ../build/release/cpp/experiments/TPGExperimentMPI -s $seed -k $experiment_key \
      1> tpg.$seed.$$.std 2> tpg.$seed.$$.err

      wait

      # Upload files
      $TPG_PATH/scripts/run/upload-asset.sh $experiment_key tpg.$seed.$$.std
      $TPG_PATH/scripts/run/upload-asset.sh $experiment_key tpg.$seed.$$.err

      module unload httpproxy
   else
      srun ../build/release/cpp/experiments/TPGExperimentMPI -s $seed \
      1> tpg.$seed.$$.std 2> tpg.$seed.$$.err
   fi
fi

##pickup from checkpoint file
#t=$(grep -iRl end checkpoints/cp.*.-1.$seed.0.rslt | cut -d '.' -f 2 | sort -n | tail -n 2 | head -n 1)
#pid=$(ls tpg.crl.$seed.*.std | cut -d '.' -f 4 | tail -n 1)
#echo "pid $pid"
#echo "Starting run seed $seed t $t"
#srun ../build/release/cpp/exp/tpgExpClassicRL_MPI -C 0 -t $t -w -p -S -T 1000000 -s $seed -a $activeTask 1>> tpg.crl.$seed.$pid.std 2>> tpg.crl.$seed.$pid.err 
