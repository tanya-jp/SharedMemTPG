#!/bin/bash

#defaults
numAux=4
fitMode=0
maxt=0
modes=0
name="crl"
phs=0
numTask=6
winSize=1
wd=$(echo $PWD | rev | cut -d '/' -f 1 | rev)

while getopts a:f:g:m:n:p:T:t:w: flag
do
   case "${flag}" in
      a) numAux=${OPTARG};;
      f) fitMode=${OPTARG};;
      m) modes=${OPTARG};;
      n) name=${OPTARG};;
      p) phs=${OPTARG};;
      T) maxt=${OPTARG};;
      t) numTask=${OPTARG};;
      w) winSize=${OPTARG};;
   esac
done

c=1
#if ls *p${phs}*rslt 1> /dev/null 2>&1; then rm *p${phs}.rslt; fi
if ls *.rslt 1> /dev/null 2>&1; then rm *.rslt; fi
maxT=$maxt
if [ $maxt -eq 0 ] 
then
   maxT=$(grep "gTime t " tpg*.std | awk -F" t " '{print $2}' | awk '{print $1}' | sort -n | tail -n 1 | tr -d '\n')


elif [ $maxt -eq -1 ]
then
   files=$(ls tpg*.std | grep -v replay)
   mt=""
   for f in $files; do
      mt="$(grep "gTime t " $f | awk -F" t " '{print $2}' | awk '{print $1}' | sort -n | tail -n 1 | tr -d '\n') $mt"
   done
   maxT=$(echo $mt | tr ' ' '\n' | sort -n | head -n 1)
fi

files=$(ls tpg*.std | sort -n -t '.' -k 3 | grep -v replay)

for f in $files
do
   echo "Processing $f ..."

   #run time stats
   for aux in `seq 0 $(echo "$numAux-1" | bc)`; do
      for task in `seq 0 $(echo "$numTask-1" | bc)`; do
         echo $(tac $f | sed '/restart/q' | tac | grep setElTmsST | grep " fm ${fitMode} " | grep " ss $task " | grep " phs $phs " | head -n $maxT | awk -F"mnOut " '{print $2}' | \
            awk -F"p${phs}t${task}a${aux} " '{print $2}' | awk '{print $1}' | tr '\n' ' ')  >> ${wd}_aux_${aux}_ST_${task}_p${phs}_${name}.rslt;
                     if [ $fitMode -eq 0 ]; then
                        echo $(tac $f | sed '/restart/q' | tac | grep setElTmsMTA | grep " fm ${fitMode} " | grep " phs $phs " | head -n $maxT | awk -F"mnOut " '{print $2}' | \
                           awk -F"p${phs}t${task}a${aux} " '{print $2}' | awk '{print $1}' | tr '\n' ' ')  >> ${wd}_aux_${aux}_MTA_${task}_p${phs}_${name}.rslt;
                     fi 
                  done
               done

               echo $(tac $f | sed '/restart/q' | tac | grep "setElTmsMTA " | grep " phs $phs " | head -n $maxT | awk -F"minThr" '{print $2}' | \
                  awk '{print $1}' | tr '\n' ' ')  >> tpg-auxDouble_MTA-minThresh-$name.rslt

   #wall time
   echo $(tac $f | sed '/restart/q' | tac | grep "gTime " | head -n $maxT | awk -F" sec " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-genTimeCurve-sec-$name.rslt
   echo $(tac $f | sed '/restart/q' | tac | grep "gTime " | head -n $maxT | awk -F" evl " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-genTimeCurve-eval-$name.rslt
   #echo $(tac $f | sed '/restart/q' | tac | grep "gTime " | head -n $maxT | awk -F" gTms " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-genTimeCurve-genTeams-$name.rslt
   #echo $(tac $f | sed '/restart/q' | tac | grep "gTime " | head -n $maxT | awk -F" elTms " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-genTimeCurve-setEliteTeams-$name.rslt
   #echo $(tac $f | sed '/restart/q' | tac | grep "gTime " | head -n $maxT | awk -F" sTms " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-genTimeCurve-selTeams-$name.rslt
   #echo $(tac $f | sed '/restart/q' | tac | grep "gTime " | head -n $maxT | awk -F" rprt " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-genTimeCurve-rprt-$name.rslt
   #
   #   #cumulative state
   #   echo $(tac $f | sed '/restart/q' | tac | grep "cpAFts " | head -n $maxT | awk -F"cpAFts " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-cpAFts-$name.rslt
   #
   #echo $(grep "tToEvl " $f | head -n $maxT | awk -F" tToEvl " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-genTimeCurve-tToEvl-$name.rslt

#   for task in `seq 0 $(echo "$numTask-1" | bc)`; do
#      echo $(tac $f | sed '/restart/q' | tac | grep setElTmsST | grep " fm ${fitMode} " | grep "ss $task " | grep " phs $phs " | head -n $maxT | awk -F " nP " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-pCount-${name}-st-${task}.rslt
#      echo $(tac $f | sed '/restart/q' | tac | grep setElTmsST | grep "fm ${fitMode} " | grep "ss $task " | grep " phs $phs " | head -n $maxT | awk -F " nT " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-tCount-${name}-st-${task}.rslt
#      echo $(tac $f | sed '/restart/q' | tac | grep setElTmsST | grep "fm ${fitMode} " | grep "ss $task " | grep " phs $phs " | head -n $maxT | awk -F "age" '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-age-${name}-st-${task}.rslt
#      echo $(tac $f | sed '/restart/q' | tac | grep setElTmsST | grep "fm ${fitMode} " | grep "ss $task " | grep " phs $phs " | head -n $maxT | awk -F "fit" '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-fit-${name}-st-${task}.rslt
#   done

   #echo $(tac $f | sed '/restart/q' | tac | grep "setElTmsMTA fm ${fitMode} " | grep " phs $phs " | head -n $maxT | awk -F " nP " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-pCount-$name-mt.rslt
   #echo $(tac $f | sed '/restart/q' | tac | grep "setElTmsMTA fm ${fitMode} " | grep " phs $phs " | head -n $maxT | awk -F " nT " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-tCount-$name-mt.rslt
   #echo $(tac $f | sed '/restart/q' | tac | grep setElTmsMTA | grep " fm ${fitMode} "  | grep " phs $phs " | head -n $maxT | awk -F "age" '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-age-$name-mt.rslt   

   #echo $(tac $f | sed '/restart/q' | tac | grep "setElTmsMTA fm ${fitMode} " | grep " phs $phs " | head -n $maxT | awk -F " mnProgIns " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-meanPIns-$name.rslt
   #echo $(tac $f | sed '/restart/q' | tac | grep "setElTmsMTA fm ${fitMode} " | grep " phs $phs " | head -n $maxT | awk -F " mnEProgIns " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-meanEPIns-$name.rslt

#echo $(tac $f | sed '/restart/q' | tac | grep "setElTmsMTA fm ${fitMode} " | grep " phs $phs " | head -n $maxT | awk -F " pF " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-policyFeatures-$name.rslt
#echo $(tac $f | sed '/restart/q' | tac | grep "setElTmsMTA fm ${fitMode} " | grep " phs $phs " | head -n $maxT | awk -F " pF " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-policyFeatures-$name.rslt

#echo $(tac $f | sed '/restart/q' | tac | grep "setElTmsMTA fm ${fitMode} " | grep " phs $phs " | head -n $maxT | awk -F " lTypePr " '{print $2}' | awk '{print $1}') >> tpg-propType0-$name.rslt
#echo $(tac $f | sed '/restart/q' | tac | grep setElTmsMTA | grep " fm ${fitMode} " | grep " phs $phs " | head -n $maxT | awk -F " mnTmSzR " '{print $2}' | awk '{print $1}') >> tpg-tmSizeRoot-$name.rslt
#echo $(tac $f | sed '/restart/q' | tac | grep setElTmsMTA | grep " fm ${fitMode} " | grep " phs $phs " | head -n $maxT | awk -F " mnTmSzS " '{print $2}' | awk '{print $1}') >> tpg-tmSizeSub-$name.rslt

echo $(tac $f | sed '/restart/q' | tac | grep genTms | head -n $maxT | awk -F" Msz " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-gt-Msize-$name.rslt
echo $(tac $f | sed '/restart/q' | tac | grep genTms | head -n $maxT | awk -F" Lsz " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-gt-Lsize-$name.rslt
echo $(tac $f | sed '/restart/q' | tac | grep genTms | head -n $maxT | awk -F" mSz " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-gt-MemSize-$name.rslt
echo $(tac $f | sed '/restart/q' | tac | grep genTms | head -n $maxT | awk -F" rSz " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-gt-Rsize-$name.rslt
echo $(tac $f | sed '/restart/q' | tac | grep genTms | head -n $maxT | awk -F" eLSz " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-gt-eLSz-$name.rslt

#echo $(tac $f | sed '/restart/q' | tac | grep selTms | head -n $maxT | awk -F" Msz " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-st-Msize-$name.rslt
#echo $(tac $f | sed '/restart/q' | tac | grep selTms | head -n $maxT | awk -F" Lsz " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-st-Lsize-$name.rslt
#echo $(tac $f | sed '/restart/q' | tac | grep selTms | head -n $maxT | awk -F" mSz " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-st-MemSize-$name.rslt
#echo $(tac $f | sed '/restart/q' | tac | grep selTms | head -n $maxT | awk -F" rSz " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-st-Rsize-$name.rslt
#echo $(tac $f | sed '/restart/q' | tac | grep selTms | head -n $maxT | awk -F" mrSz " '{print $2}' | awk '{print $1}' | tr '\n' ' ') >> tpg-st-mRsize-$name.rslt
#echo $(tac $f | sed '/restart/q' | tac | grep selTms | head -n $maxT | awk -F" nOldDelPr " '{print $2}' | awk '{print $1}') >> tpg-osr-$name.rslt
#echo $(tac $f | sed '/restart/q' | tac | grep selTms | head -n $maxT | awk -F" sRTC " '{print $2}' | awk '{print $1}') >> tpg-sRTC-$name.rslt
#echo $(tac $f | sed '/restart/q' | tac | grep selTms | head -n $maxT | awk -F" sGsz " '{print $2}' | awk '{print $1}') >> tpg-sGsz-$name.rslt
echo $(tac $f | sed '/restart/q' | tac | grep selTms | head -n $maxT | awk -F" nDel " '{print $2}' | awk '{print $1}') >> tpg-nDel-$name.rslt

   #MODES
   if [ $modes -gt 0 ]; then
      echo $(tac $f | sed '/restart/q' | tac | grep TPG::MODES | awk -F"change" '{print $2}' | awk '{print $1}') >> ${wd}-MODES-change-$name.rslt
      echo $(tac $f | sed '/restart/q' | tac | grep TPG::MODES | awk -F"novelty" '{print $2}' | awk '{print $1}') >> ${wd}-MODES-novelty-$name.rslt
      echo $(tac $f | sed '/restart/q' | tac | grep TPG::MODES | awk -F"complexityRTC" '{print $2}' | awk '{print $1}') >> ${wd}-MODES-complexityRTC-$name.rslt
      echo $(tac $f | sed '/restart/q' | tac | grep TPG::MODES | awk -F"complexityTeams" '{print $2}' | awk '{print $1}') >> ${wd}-MODES-complexityTeams-$name.rslt
      echo $(tac $f | sed '/restart/q' | tac | grep TPG::MODES | awk -F"complexityPrograms" '{print $2}' | awk '{print $1}') >> ${wd}-MODES-complexityPrograms-$name.rslt
      echo $(tac $f | sed '/restart/q' | tac | grep TPG::MODES | awk -F"complexityInstructions" '{print $2}' | awk '{print $1}') >> ${wd}-MODES-complexityInstructions-$name.rslt
      echo $(tac $f | sed '/restart/q' | tac | grep TPG::MODES | awk -F"ecology" '{print $2}' | awk '{print $1}') >> ${wd}-MODES-ecology-$name.rslt
   fi
done

echo $(tac $f | sed '/restart/q' | tac | grep archive | head -n $maxT | awk -F" sz " '{print $2}' | awk '{print $1}') >> tpg-archive-size-$name.rslt


# plot #########################################################

i=0
if ls *p${phs}.pdf 1> /dev/null 2>&1; then rm *p${phs}.pdf; fi

##combined train+test
#if [ $phs -eq 2 ]; then
#Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves-paired.R ${wd}_aux_0_ST_0_p0_${name}.rslt ${wd}_aux_0_ST_0_p2_${name}.rslt "Mean Fitness" "$winSize" "$(printf "%03d" $i)_${task}-${aux}-${name}" 0
#i=$((i+1))
#fi

for aux in `seq 0 $(echo "$numAux-1" | bc)`; do
   for task in `seq 0 $(echo "$numTask-1" | bc)`; do
      i=$((i+1))
      Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R ${wd}_aux_${aux}_ST_${task}_p${phs}_${name}.rslt "Aux ${aux} (single-task $task)" "$winSize" "$(printf "%03d" $i)_${task}-${aux}-${name}" 0
   done
done

if [ $fitMode -eq 0 ]; then
   for aux in `seq 0 $(echo "$numAux-1" | bc)`; do
      for task in `seq 0 $(echo "$numTask-1" | bc)`; do
         i=$((i+1))
         Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R ${wd}_aux_${aux}_MTA_${task}_p${phs}_${name}.rslt "Aux ${aux} (multi-task $task)" "$winSize" "$(printf "%03d" $i)_${task}-${aux}-${name}" 0
      done
   done
fi

#  for task in `seq 0 $(echo "$numTask-1" | bc)`; do
#     i=$((i+1))
#     Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-tCount-$name-st-${task}.rslt "Teams per Graph (best single-task ${task})" "$winSize" "$(printf "%03d" $i)_$name-st-${task}" 0
#  done
#  
#  for task in `seq 0 $(echo "$numTask-1" | bc)`; do
#     i=$((i+1))
#     Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-pCount-$name-st-${task}.rslt "Programs per Graph (best single-task ${task})" "$winSize" "$(printf "%03d" $i)_$name-st-${task}" 0
#  done
#  
#  for task in `seq 0 $(echo "$numTask-1" | bc)`; do
#     i=$((i+1))
#     Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-age-$name-st-${task}.rslt "Age of Graph (best single-task ${task})" "$winSize" "$(printf "%03d" $i)_$name-st-${task}" 0
#  done

#for task in `seq 0 $(echo "$numTask-1" | bc)`; do
#     i=$((i+1))
#     Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-fit-$name-st-${task}.rslt "Fitness of Graph (best single-task ${task})" "$winSize" "$(printf "%03d" $i)_$name-st-${task}" 0
#done

#i=$((i+1))
#Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-tCount-$name-mt.rslt "Teams per Graph (best multi-task ${task})" "$winSize" "$(printf "%03d" $i)_$name" 0
#i=$((i+1))
#Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-pCount-$name-mt.rslt "Programs per Graph (best multi-task ${task})" "$winSize" "$(printf "%03d" $i)_$name" 0
#i=$((i+1))
#Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-age-$name-mt.rslt "Age of Graph (best multi-task ${task})" "$winSize" "$(printf "%03d" $i)_$name" 0
#i=$((i+1))
#Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-auxDouble_MTA-minThresh-$name.rslt "minThreshold" "$winSize" "$(printf "%03d" $i)_$name" 0
#i=$((i+1))
#Rscript  $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-sRTC-$name.rslt "Population-wide Instructions Executed" "$winSize" "$(printf "%03d" $i)_$name" 0
#i=$((i+1))
#Rscript  $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-sGsz-$name.rslt "Population-wide Teams Executed" "$winSize" "$(printf "%03d" $i)_$name" 0
#i=$((i+1))
#Rscript  $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-nDel-$name.rslt "# teams deleted" "$winSize" "$(printf "%03d" $i)_$name" 0
i=$((i+1))
Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-genTimeCurve-sec-$name.rslt "Seconds Total" "$winSize" "$(printf "%03d" $i)_$name" 0 
i=$((i+1))
Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-genTimeCurve-eval-$name.rslt "Seconds Evaluation " "$winSize" "$(printf "%03d" $i)_$name" 0 
#i=$((i+1))
#Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-genTimeCurve-genTeams-$name.rslt "Seconds Replacement" "$winSize" "$(printf "%03d" $i)_$name" 0 
#i=$((i+1))
#Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-genTimeCurve-setEliteTeams-$name.rslt "Seconds Set Elite Teams" "$winSize" "$(printf "%03d" $i)_$name" 0
#i=$((i+1))
#Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-genTimeCurve-selTeams-$name.rslt "Seconds Selection" "$winSize" "$(printf "%03d" $i)_$name" 0 
#i=$((i+1))
#Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-genTimeCurve-rprt-$name.rslt "Seconds Accounting & Reporting" "$winSize" "$(printf "%03d" $i)_$name" 0
#i=$((i+1))
#Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-genTimeCurve-tToEvl-$name.rslt "Teams to Evaluate" "$winSize" "$(printf "%03d" $i)_$name" 0
#  #i=$((i+1))
#  #Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-propType0-$name.rslt "Proportion of Action-Value Programs (best graph)" "$winSize" "$(printf "%03d" $i)_$name" 1 
#i=$((i+1))
#Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-osr-$name.rslt "Offspring Survival Rate (numOldDeleted/numDeleted)" "$winSize" "$(printf "%03d" $i)_$name" 0 
#  #i=$((i+1))
#  #Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-avp-$name.rslt "Proportion Action-Value Programs in Population" "$winSize" "$(printf "%03d" $i)_$name" 0

if [ $modes -gt 0 ]; then
   i=$((i+1))
   Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R ${wd}-MODES-change-$name.rslt "MODES - Change" 5 "$(printf "%03d" $i)_$name" 0 
   i=$((i+1))
   Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R ${wd}-MODES-novelty-$name.rslt "MODES - Novelty" 5 "$(printf "%03d" $i)_$name" 0 
   i=$((i+1))
   Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R ${wd}-MODES-complexityRTC-$name.rslt "MODES - Complexity RTC" 5 "$(printf "%03d" $i)_$name" 0 
   i=$((i+1))
   Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R ${wd}-MODES-complexityTeams-$name.rslt "MODES - Complexity Teams" 5 "$(printf "%03d" $i)_$name" 0 
   i=$((i+1))
   Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R ${wd}-MODES-complexityPrograms-$name.rslt "MODES - Complexity Programs" 5 "$(printf "%03d" $i)_$name" 0
   i=$((i+1))
   Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R ${wd}-MODES-complexityInstructions-$name.rslt "MODES - Complexity Instruction" 5 "$(printf "%03d" $i)_$name" 0
   i=$((i+1))
   Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R ${wd}-MODES-ecology-$name.rslt "MODES - Ecology" 5 "$(printf "%03d" $i)_$name" 0 
fi

#i=$((i+1))
#Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-meanPIns-$name.rslt "Mean Instructions per Program (best graph)" "$winSize" "$(printf "%03d" $i)_$name" 0 
#i=$((i+1))
#Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-meanEPIns-$name.rslt "Mean Effective Instructions per Program (best graph)" "$winSize" "$(printf "%03d" $i)_$name" 0 
#i=$((i+1))
#Rscript $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-policyFeatures-$name.rslt "Features (best graph)" "$winSize" "$(printf "%03d" $i)_$name" 0
#i=$((i+1))
#Rscript  $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-gt-Msize-$name.rslt "Team Population Size" "$winSize" "$(printf "%03d" $i)_$name" 0 
#i=$((i+1))
#Rscript  $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-gt-Lsize-$name.rslt "Program Population Size" "$winSize" "$(printf "%03d" $i)_$name" 0
#i=$((i+1))
#Rscript  $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-gt-MemSize-$name.rslt "Memory Population Size" "$winSize" "$(printf "%03d" $i)_$name" 0
#i=$((i+1))
#Rscript  $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-gt-Rsize-$name.rslt "Root Population Size" "$winSize" "$(printf "%03d" $i)_$name" 0
#i=$((i+1))
#Rscript  $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-gt-eLSz-$name.rslt "Elite Teams " "$winSize" "$(printf "%03d" $i)_$name" 0
#i=$((i+1))
#Rscript  $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-st-Msize-$name.rslt "st Team population size" "$winSize" "$(printf "%03d" $i)_$name" 0
#i=$((i+1))
#Rscript  $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-st-Lsize-$name.rslt "st Program population size" "$winSize" "$(printf "%03d" $i)_$name" 0
#i=$((i+1))
#Rscript  $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-st-Rsize-$name.rslt "st Rsize" "$winSize" "$(printf "%03d" $i)_$name" 0
#i=$((i+1))
#Rscript  $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-st-mRsize-$name.rslt "st mRoot population size" "$winSize" "$(printf "%03d" $i)_$name" 0
#i=$((i+1))
#Rscript  $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-st-MemSize-$name.rslt "st Memory population size" "$winSize" "$(printf "%03d" $i)_$name" 0
#i=$((i+1))
#Rscript  $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-tmSizeRoot-$name.rslt "Mean Root Team Size" "$winSize" "$(printf "%03d" $i)_$name" 0 
#i=$((i+1))
#Rscript  $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-tmSizeSub-$name.rslt "Mean Sub Team Size" "$winSize" "$(printf "%03d" $i)_$name" 0 
#i=$((i+1))
#Rscript  $TPG_PATH/scripts/plot/plot-tpg-trainingCurves.R tpg-archive-size-$name.rslt "Archive Size " "$winSize" "$(printf "%03d" $i)_$name" 0


#img2pdf --out ${name}_${wd}_p${phs}.pdf *.png
#convert "*.{png,jpeg}" -quality 100 ${name}_${wd}_p${phs}.pdf
pdfunite 0*$name*.pdf ${wd}_p${phs}.pdf
rm 0*$name*.pdf
#rm 0*$name*.png
rm *rslt

