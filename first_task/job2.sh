#!/bin/bash 
#PBS -l walltime=00:01:00,nodes=5:ppn=1 
#PBS -N example_job 
#PBS -q batch 
for (( i = 0; i < 30; i++ )); do 
    a=($(mpirun -n 5 ./main $i)) 
    echo $a, $i >> 1.txt 
done
