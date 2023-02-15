#!/bin/bash -x
#SBATCH --time=1-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
##SBATCH --gres=mem192

#SBATCH --job-name=opes
#SBATCH --output=opes.%j.out

#SBATCH --partition=batch


# TODO: Load CHARMM, AMBER, PLUMED and python environment with ParmEd and MDAnalysis.

# Example input variables
#name=00061095
#rand1=1

# Create temporary dir to use as working directory for the calculations and copy the relevant files.
TMPDIR="tmp/zinc_${name}-${rand1}-opes/"

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
mkdir traj dihe # We don't use the subfolders restart and energy for the OPES runs with amber.

# Directories are relative paths to TMPDIR.
DIR="../.."
INPDIR="${DIR}/charmm_inp"
OUTDIR="${DIR}/out"
OUTDIROPES="${DIR}/out-opes"

# Convert first frame of CHARMM trajectory from dcd to crd format.
# This will be the starting configuration for the OPES simulation.
# This way both the unbiased and biased simulations will start from the same dihedrals.
if [[ ${rand1} -gt 10 ]]
then
  # We have generated random dihedrals only for the first ten replicates.
  cp ${OUTDIR}/traj/dyna.${name}.mm.1.dcd unbiased-trajectory.dcd
else
  cp ${OUTDIR}/traj/dyna.${name}.mm.${rand1}.dcd unbiased-trajectory.dcd
fi
cat > dcdtocrd.inp << EOF
read rtf card name top_all36_cgenff.rtf
read para card flex name par_all36_cgenff.prm
stream zinc_@name.str
read psf card name zinc_@name.psf

open unit 12 read file name unbiased-trajectory.dcd
read coor file unit 12 ifile 1
close unit 12
write coor card name frame1.crd
* frame #1 from unbiased-trajectory.dcd
*
EOF
charmm name=${name} rand1=${rand1} -i dcdtocrd.inp -o dcdtocrd.out

# Convert parameters and coordinates from CHARMM to AMBER formats.
conda activate parmed
cat > parmed.in << EOF
chamber -top top_all36_cgenff.rtf -param par_all36_cgenff.prm -str zinc_${name}.str -psf zinc_${name}.psf -crd frame1.crd -nocmap
outparm zinc_${name}.prmtop zinc_${name}.inpcrd
EOF
parmed -i parmed.in

# Copy also AMBER + PLUMED input files.
cp ${DIR}/amber_opes/{amber_mm_plumed.in,amber_netcdf_to_dcd.py} .
cp ${DIR}/amber_opes/plumed-${name}.dat plumed.dat

date

# Run the OPES dynamics.
sander -O -i amber_mm_plumed.in -o amber.out -p zinc_${name}.prmtop -c zinc_${name}.inpcrd -x mm_plumed.ncdf -r mm_plumed.rst -frc mm_plumed.frc -inf mm_plumed.info

# Convert AMBER NetCDF trajectory to traj/dyna.@name.mm.@rand1.dcd, which is needed by the dihedral analysis.
python amber_netcdf_to_dcd.py --molid=${name} --rand1=${rand1}

# Analyze dihedrals. Optional since the PLUMED colvar file will also contain all the dihedrals.
#charmm name=${name} rand1=${rand1} lot=mm -i ${INPDIR}/eval_dihe.inp -o eval_mm.dihe.${name}-${rand1}.out
#tail eval_mm.dihe.${name}-${rand1}.out
#date

# Copy output files from working directory.
mkdir -p ${OUTDIROPES}/{colvar,traj,dihe}
cp colvar.data ${OUTDIROPES}/colvar/colvar.${name}.${rand1}.data
cp traj/dyna.${name}.mm.${rand1}.dcd ${OUTDIROPES}/traj/
# We don't really need to copy also the ncdf file and the dihedral analysis since we have the dcd trajectory and colvar files.
#cp mm_plumed.ncdf ${OUTDIROPES}/traj/dyna.${name}.mm.${rand1}.ncdf
#cp dihe/dihe.${name}.mm.${rand1}.dat ${OUTDIROPES}/dihe/

date
