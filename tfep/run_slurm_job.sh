#!/bin/bash -x
#SBATCH --time=1-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48

#SBATCH --job-name=tfep
#SBATCH --output=tfep.%j.out

#SBATCH --partition=batch


# TODO: Load AMBER and Python environment.

# Select molecule ID.
molidx=${molidx:-'6'}  # If molidx env variable not defined, run for 00140610.
MOLIDS=(00077329 00079729 00086442 00107550 00133435 00138607 00140610 00164361 00167648 00169358 01867000 06568023 00061095 00087557 00095858 01755198 03127671 04344392 33381936 00107778 00123162 04363792)
MOLID=${MOLIDS[${molidx}]}


date

# Run the TFEP analysis on the unbiased data.
python run_tfep.py --molid=${MOLID} --flow=${flow} --subdir=${flow}6-b${batch} --batch=${batch}
python run_tfep.py --molid=${molid} --subdir=${flow}6-b${batch} --df  # bootstrap analysis.

date


# De-comment these lines to run the TFEP analysis on the OPES production data.
# -----------------------------------------------------------------------------
#python run_tfep.py --molid=${MOLID} --flow=${flow} --subdir=${flow}6-b${batch}-r11-opes --batch=${batch} --repeat=${repeat} --opes
#python run_tfep.py --molid=${molid} --subdir=${flow}6-b${batch}-r11-opes --repeat=${repeat} --opes --df  # bootstrap analysis

date


# De-comment these lines to run the standard FEP analysis.
# ---------------------------------------------------------

# Evaluate the potential energies to evaluate standard FEP.
#python run_tfep.py --molid=${MOLID} --subdir=unbiased --fep  # unbiased data.
#python run_tfep.py --molid=${MOLID} --subdir=opes --repeat=11 --opes --fep  # production OPES data.

date

# Run bootstrap analysis.
#python run_tfep.py --molid=${molid} --subdir=unbiased --fep --df  # unbiased data.
#python run_tfep.py --molid=${molid} --subdir=opes --repeat=11 --fep --df --opes  # production OPES data.

date
