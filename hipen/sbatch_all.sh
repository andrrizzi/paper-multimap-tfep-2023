#!/bin/bash

# Good molecules, unbiased dynamics.
for m in 00077329 00079729 00086442 00107550 00133435 00138607 00140610 00164361 00167648 00169358 01867000 06568023
do
for i in {1..10}
do
   sbatch --export=ALL,name=${m},rand1=${i} dyna_mm.slurm
done
done


# Bad/ugly molecules, unbiased dynamics.
for m in 00061095 00087557 00095858 01755198 03127671 04344392 33381936 00107778 00123162 04363792
do
for i in {1..10}
do
   sbatch --export=ALL,name=${m},rand1=${i} dyna_mm.slurm
done
done


# Production OPES dynamics.
for m in 00061095 00095858
do
   sbatch --export=ALL,name=${m},rand1=11 dyna_mm.slurm
done


# Test OPES dynamics. Before running these, you need to set the nstlim option of the amber_opes/amber_mm_plumed.in file
# to 15000000. These are only used to test the effectiveness of the sampling as described in the paper.
#for m in 00061095 00095858
#do
#for i in {1..10}
#do
#   sbatch --export=ALL,name=${m},rand1=${i} dyna_mm.slurm
#done
#done
