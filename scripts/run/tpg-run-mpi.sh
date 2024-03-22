#!/bin/bash

# start tpg ###################################################

#defaults
mode=0 #Train:0, Replay:1, Debug:2
numMPIProc=2
seed=42

while getopts m:n:s: flag
do
   case "${flag}" in
      m) mode=${OPTARG};;
      n) numMPIProc=${OPTARG};;
      s) seed=${OPTARG};;
   esac
done

# Evolve
if [ $mode -eq 0 ]; then
   echo "Starting run $seedTPG..."
   #run from scratch (in background)
   mpirun --oversubscribe -np $numMPIProc $TPG_PATH/build/release/cpp/exp/tpgExpClassicRL_MPI -s $seed 1> tpg.$seed.$$.std 2> tpg.$seed.$$.err &
fi

# Replay best
if [ $mode -eq 1 ]; then
   #replay
   phs=2
   #seedEnv=0
   if ls replay/frames/* 1> /dev/null 2>&1; then rm replay/frames/*; fi
   if ls replay/graphs/* 1> /dev/null 2>&1; then rm replay/graphs/*; fi
   bestScore=$(grep setElTmsST  tpg.${seed}.*.std | grep " fm 0 " | grep " phs $phs " | awk -F"mnOut" '{print $2}' | awk -F "p${phs}t0a0 " '{print $2}' | awk '{print $1}' | sort -n | uniq | tail -n 1)
   t=$(grep setElTmsST tpg.${seed}.*.std | grep " fm 0 " | grep "p${phs}t0a0 ${bestScore} " tpg.${seed}.*.std | grep " phs $phs " | head -n 1 | awk -F" t " '{print $2}' | awk '{print $1}')
   tm=$(grep setElTmsST tpg.${seed}.*.std | grep " fm 0 " |  grep "p${phs}t0a0 ${bestScore} " tpg.${seed}.*.std | grep " phs $phs " | grep " t $t " | head -n 1 | awk -F" id " '{print $2}' | awk '{print $1}')
   echo "Fitness:$bestScore Generation:$t Team:$tm"
   mpirun --oversubscribe -np 2 $TPG_PATH/build/release/cpp/exp/tpgExpClassicRL_MPI -a -R $tm -C $phs -t $t -s $seed -g $seed \
      1> tpg.$seed.replay.std 2> tpg.$seed.replay.err &
   # mpirun --oversubscribe -np 2 xterm -hold -e gdb -ex run --args $TPG_PATH/build/release/cpp/exp/tpgExpClassicRL_MPI -a -R $tm -C $phs -t $t -s $seed -g $seed \
   #    1> tpg.$seed.replay.std 2> tpg.$seed.replay.err &
fi

# Debug
if [ $mode -eq 2 ]; then
   #debug
   mpirun --oversubscribe -np $numMPIProc xterm -hold -e gdb -ex run --args ../build/release/cpp/exp/tpgExpClassicRL_MPI -s $seed 1> tpg.$seed.$$.std 2> tpg.$seed.$$.err &
fi

# # Check for memoy leaks
# if [ $mode -eq 3 ]; then
# ##view profile with: google-pprof ../build/release/cpp/exp/tpgExpBlocks_MPI ./tpg.out_27134   
# ##google-pprof --gv --focus=genTeams ../build/release/cpp/exp/tpgExpClassicRL_MPI tpg.out_441771
# #LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libprofiler.so CPUPROFILE=tpg.out \
# #mpirun --oversubscribe -np $numMPIProc ../build/release/cpp/exp/tpgExpClassicRL_MPI -w -F $numFitMode -D $dim -p -S -T $tMax -s $seedTPG -g $seedEnv -a $activeTasks -d $taskSwitchMod -f $fitMode 1> tpg.$seedTPG.$$.std 2> tpg.$seedTPG.$$.err &

# ##valgrind
# mpirun --oversubscribe -np $numMPIProc valgrind --leak-check=yes --show-reachable=yes --log-file=vg.%p --suppressions=/usr/share/openmpi/openmpi-valgrind.supp \
# ../build/release/cpp/exp/tpgExpClassicRL_MPI -s $seed 1> tpg.$seed.$$.std 2> tpg.$seed.$$.err &
# fi

# #if [ $mode -eq 4 ]; then
# ##pickup from checkpoint file
# #t=$(grep -iRl end checkpoints/cp.*.-1.$seedTPG.0.rslt | cut -d '.' -f 2 | sort -n | tail -n 2 | head -n 1)
# ##sed  -i "1,/genTime\ t\ $t/!d" tpg.$seedTPG.std
# #pid=$(ls tpg.$seedTPG.*.std | cut -d '.' -f 4 | tail -n 1)
# #echo "pid $pid"
# #echo "Starting run seedTPG $seedTPG t $t"
# #LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libprofiler.so CPUPROFILE=tpg.out \
# #mpirun --oversubscribe -np $numMPIProc ../build/release/cpp/exp/tpgExpClassicRL_MPI -s $seedTPG 1>> tpg.$seedTPG.$pid.std 2>> tpg.$seedTPG.$pid.err &
# ##mpirun --oversubscribe -np $numMPIProc xterm -hold -e gdb -ex run --args ../build/release/cpp/exp/tpgExpClassicRL_MPI -C 0 -t $t -w -F $numFitMode -D $dim -p -S -T $tMax -s $seedTPG -g $seedEnv -d $taskSwitchMod -a $activeTasks 1>> tpg.$seedTPG.$pid.std 2>> tpg.$seedTPG.$pid.err &
# #fi
