#!/usr/local_rwth/bin/zsh

#SBATCH --time=0-08:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5120M

#SBATCH --job-name=dynamm
#SBATCH --output=dynamm.%A.out

# TODO: Load CHARMM in your environment.

# Create temporary dir to use as working directory for the calculations and copy the relevant files.
TMPDIR="tmp/zinc_${name}-${rand1}-mm/"

mkdir -p ${TMPDIR}
cp 3ob-mol/3ob-${name}.str ${TMPDIR}
cp 3ob-mol/sccdftb-${name}.dat ${TMPDIR}sccdftb.dat
cp cgenff/{top_all36_cgenff.rtf,par_all36_cgenff.prm} ${TMPDIR}
cp cgenff-mol/zinc_${name}.str ${TMPDIR}
cp coors/zinc_${name}.{crd,psf} ${TMPDIR}
cp charmm_inp/mdpars-{mm,3ob}.str ${TMPDIR}
cp corr_dihe/corr-dihe-${name}.str ${TMPDIR}
cp randomize/randomize-${name}.str ${TMPDIR}
cd ${TMPDIR}

# Create subdirectories where the CHARMM input scripts save files.
mkdir restart traj dihe energy

# Directories are relative paths to TMPDIR.
DIR="../.."
INPDIR="${DIR}/charmm_inp"
OUTDIR="${DIR}/out"

date

# Run the MM dynamics.
srun charmm name=${name} rand1=${rand1} -i ${INPDIR}/dyna_mm.inp -o dyna-${name}-${rand1}-mm.out
tail dyna-${name}-${rand1}-mm.out
date

# Analyze dihedrals.
srun charmm name=${name} rand1=${rand1} lot=mm -i ${INPDIR}/eval_dihe.inp -o eval_mm.dihe.${name}-${rand1}.out
tail eval_mm.dihe.${name}-${rand1}.out
date

# Compute energies at the MM level.
srun charmm name=${name} rand1=${rand1} lot=mm -i ${INPDIR}/eval_mm.inp -o eval_mm.mm.${name}-${rand1}.out
tail eval_mm.mm.${name}-${rand1}.out
date

# We don't need to compute energies at the 3ob level since we can compute them through the python script in ../tfep/.
#srun charmm name=${name} rand1=${rand1} lot=mm qmreg=MOL -i ${INPDIR}/eval_3ob.inp -o eval_mm.3ob.${name}-${rand1}.out
#tail eval_mm.3ob.${name}-${rand1}.out
#date

# Compress output.
tar -cvzf restarts-${name}-${rand1}-mm.tar.gz restart/*
tar -cvzf outs-${name}-${rand1}-mm.tar.gz *.out

# Copy output files from working directory.
mkdir -p ${OUTDIR}/{energy,traj,dihe,restart,logs}
cp energy/*.${rand1}.ene ${OUTDIR}/energy/
cp traj/dyna.${name}.mm.${rand1}.dcd ${OUTDIR}/traj/
cp dihe/dihe.${name}.mm.${rand1}.dat ${OUTDIR}/dihe/
cp restarts-${name}-${rand1}-mm.tar.gz ${OUTDIR}/restart/
cp outs-${name}-${rand1}-mm.tar.gz ${OUTDIR}/logs/

date
