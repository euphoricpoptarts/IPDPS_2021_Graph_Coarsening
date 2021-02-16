#!/bin/bash
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 executable csr_filename metrics_filename trials"
    exit
fi
executable=$1
testFile=$2
metricsFile=$3
trials=$4

./$executable $testFile $metricsFile 0 0 0 $trials > /dev/null 2>&1

totalT=0
coarsenT=0
refineT=0
line=0
while IFS=',' read -r total coarsen refine
do
    totalT=$(echo $totalT + $total | bc)
    coarsenT=$(echo $coarsenT + $coarsen | bc)
    refineT=$(echo $refineT + $refine | bc)
    line=$((line + 1))
done < "$metricsFile"
averageTotal=$(echo "scale=5; $totalT/$line" | bc)
averageCoarsen=$(echo "scale=5; $coarsenT/$line" | bc)
averageRefine=$(echo "scale=5; $refineT/$line" | bc)
echo "average time = $averageTotal, average coarsen = $averageCoarsen, average refine = $averageRefine, total lines = $line"