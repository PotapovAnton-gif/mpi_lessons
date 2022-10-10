#!/bin/bash

#PBS -l walltime=00:01:00,nodes=1:ppn=2
#PBS -N example_job
#PBS -q batch

for (( i = 0; i < 15; i++ )); do
    a=($(mpirun -n 2 ./main $i))
    echo $a, $i
done