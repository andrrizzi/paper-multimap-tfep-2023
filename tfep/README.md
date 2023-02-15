## Manifest

- ``run_tfep.py``: The main script running the TFEP analysis on HiPen.
- ``analysis.py``: Main analysis script used to generate the figures of the paper.
- ``modules/``: Contains utility classes to create and train normalizing flows used in the paper.
- ``run_slurm_job.sh``: Example SLURM script to launch the TFEP analysis on a molecule.

## Requirements

Running the script requires an AMBER installation as well as a Python environment with the [tfep library](https://github.com/andrrizzi/tfep)
installed. The [ASE](https://wiki.fysik.dtu.dk/ase/) and [bgflow](https://github.com/noegroup/bgflow) must also be
installed.

## Usage

Before using this test, it is necessary to run ``../hipen/sbatch_all.sh``, which generate the reference trajectories.
Then you can run the ``run_tfep.py`` script to run TFEP. The following example runs TFEP on molecule 00140610 using
zmatrix coordinates with a spline transformer and a batch size of 48.
```
# The first command train the map and generates the potential energy samples.
python run_tfep.py --molid=00140610 --flow=zmatrix-spline --batch=48 --subdir=zmatrix-spline-b48
# This second command (with the --df flag) reads the potential and runs a bootstrap analysis on the free energy prediction.
python run_tfep.py --molid=00140610 --subdir=zmatrix-spline-b48 --df
```
To run a standard FEP, use the ``--fep`` flag. For example
```
python run_tfep.py --molid=00140610 --subdir=unbiased --fep
python run_tfep.py --molid=00140610 --subdir=unbiased --fep --df
```
To run the analysis (both standard FEP and multimap TFEP) on the OPES data use the ``--opes`` flag. For example, for
repeat 11 (i.e., ran with ``rand=11``) which in this work was used as the production run:
```
python run_tfep.py --molid=00140610 --flow=zmatrix-spline --batch=48 --repeat=11 --subdir=zmatrix-spline-b48-r11-opes --opes
python run_tfep.py --molid=00140610 --repeat=11 --subdir=zmatrix-spline-b48-r11-opes --opes --df
```
You can get more info on the options with
```
python run_tfep.py --help
```
