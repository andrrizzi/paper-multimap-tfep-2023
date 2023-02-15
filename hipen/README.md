HiPen dataset [1] for benchmarking methods computing MM -> DFTB3 free energy differences.

Almost all of these files are taken from <doi.org/10.5281/zenodo.2328951> and modified for the purpose of testing the
performance of targeted free energy perturbation against FEP.

## Manifest

- ``3ob-3-1/``: 3ob-3-1 parameters for the DFTB potential. This directory must be downloaded from https://dftb.org/parameters/download/3ob/3ob-3-1-cc.
- ``3ob-mol/``: Molecule-specific options and parameters (Hubbard derivatives and zeta parameters) for SCCDFTB. Some of
                the files included in this directory necessary to run the tests must first be created with ``create_sccdftb_dat.py``.
- ``amber_opes/``: AMBER input files to run an OPES calculation on molecules 00061095 and 00095858.
- ``cgenff/``: CGenFF parameters.
- ``cgenff-mol/``: Molecule-specific CGenFF parameters/
- ``charmm_inp/``: CHARMM input scripts to run the calculations.
- ``coors/``: Starting coordinates and topology files for CHARMM in ``crd`` and ``psf`` formats.
- ``corr_dihe/``: CHARMM scripts to analyze the molecule-specific dihedrals trajectories.
- ``randomize/``: CHARMM scripts to randomize molecule-specific dihedrals.
- ``scripts/create_sccdftb_dat.py``: Helper script to create the ``3ob-mol/sccdftb-X.dat`` files.
- ``reference.json``: Reference result for a subset of the compounds in the HiPen dataset taken from [1-2].
- ``dyna_mm.slurm``: Example of SLURM script that runs the unbiased dynamics at the MM level and evaluates energies at
                     the MM potential. This was modified from the ``dyna_mm.slu`` in the original HiPen files to skip
                     the MNDO calculations.
- ``dyna_mm-opes.slurm``: Example of SLURM script that runs the OPES dynamics at the MM level and evaluates energies at
                          the MM potential.
- ``sbatch_all.sh``: Bash script to launch all calculation on a SLURM system.

These files are generated by running the calculations

- ``out/``: Contain trajectories, energies, dihedrals, etc. files from running the unbiased simulations.
- ``out-opes/``: Contain trajectories, energies, colvar, etc. files from running the OPES simulations.

## Reference results

The results in ``reference.json`` are taken from the calculations in [1-2]. For the CRO estimates, we used the results
in Table S6 in the supporting information PDF of [2] for the "good" set and molecule 06568023 (number 21). This because
one of the conclusions of [2] is that less and longer trajectories seem to give more robust estimates. For all other
compounds and BAR estimates (which are not present in [2]) we used instead the results in Table 2-3 in [1].

The schema of the JSON file follows this
```json
{
  ZINC_ID: {  # e.g., ZINC_ID="00140610"
    "num": INTEGER  # integer id assigned in [1]. E.g., 12 for molecule 00140610.
    "DA offset": FLOAT  # offset as in Table 1 in [1] in kcal/mol.
    "CRO": {  # Crook's fluctuation theorem non-equilibrium calculations.
      "DA": FLOAT  # free energy difference (including offset) in kcal/mol.
      "sigma DA": FLOAT  # standard deviation of DA from block averaging as in Table 3 in [1] in kcal/mol.
      "doi": STRING  # the DOI of the publication from where the free energy result is taken.
    }
    "BAR": {  # BAR calculations
      # Same as "CRO"
    }
  }
}
```

## Requirements

Running the unbiased dynamics requires only an installation of CHARMM. The script running OPES calculations instead
requires also AMBER, PLUMED, and a python environment with MDAnalysis and ParmEd.

## Usage

First, download the 3ob-3-1 parameters for DFTB3 from https://dftb.org/parameters/download/3ob/3ob-3-1-cc. You should
have here a ``3ob-3-1/`` folder including many ``.skf`` files.

Second you need to generate an ``sccdftb.dat`` file for each molecule. To do this, run
```
cd scripts
python create_sccdftb_dat.py
```
This will result in the generation of one ``3ob-mol/sccdftb-X.dat` file for each molecule.

Finally, run all the calculations on slurm using
```
bash sbatch_all.sh
```
The script launches the SLURM scripts for each molecule and repeat. E.g. for the third repeat of the 00061095 molecule
```
sbatch --export=ALL,name='00061095',rand1=3 dyna_mm-claix.slurm
```
The output is generated inside an ``out/`` directory for the unbiased calculations and ``out-opes`` for the biased ones.
The potential energies of the MM samples evaluated at the MM and (optionally) DFTB3 level are in the
``out/energy/energy.00061095.mm.mm.3.ene`` and ``out/energy/energy.00061095.mm.3ob.3.ene`` files, respectively.

## Differences with original HiPen files

Many of the files in here were taken from the HiPen dataset published at doi.org/10.5281/zenodo.2328951. Some of these
files were modified for the specific purpose of benchmarking TFEP. These are the modifications:
- The equilibration time specified in ``charmm_inp/mdpars-mm.str`` was increased from 1 ps to 5 ns for all molecules as
  this was found to improve the results in [2].
- Molecule 1 (00061095): Constrained double bond dihedral during the initial randomization to avoid isomerization. The
  file ``corr-dihe-00061095.str`` was also modified to analyze the dihedral controlling the double bond (called X5).
- Molecule 2 (00077329): Randomized also X2 as defined in [2]. Note that X2 was not randomized in [2], but it was later
  recognized as an important dihedral for sampling.
- Molecule 6 (00095858): Swapped the X3 and X4 dihedrals to reflect nomenclature in [1] and constrained X2 to 180 degrees
  during the initial dihedral randomization to avoid isomerization of the double bond.
- Molecule 7 (00107550): Randomized only X1 and X2 as defined in [2]. Note that X2 was not randomized in [2], but it was
  later recognized as an important dihedral for sampling.
- Molecule 8 (00107778): Modified ``corr-dihe-00107778.str`` to analyze the dihedral controlling the N-O bond (called X3).
- Molecule 9 (00123162): Constrained X4 to 0 degrees during the initial dihedral randomization to avoid isomerization of
  the double bond.
- Molecule 12 (00140610): The ``3ob-mol/3ob-00140610.str`` file was modified to use parameters for ``S`` rather than ``CL``
  (which is not in the molecule).
- Molecule 16 (01755198): Swapped X4 and X5 to reflect nomenclature in [1].
- Molecule 20 (04363792): Constrained double bond dihedral during the initial randomization to avoid isomerization. The
  file ``corr-dihedral-04363792.str`` was also modified to analyze the double bond (called X3).
- Molecule 21 (06568023): Randomized dihedral X2 according to its definition in [2] rather than in [1].
- Molecule 22 (33381936): Constrained X3 dihedral controlling isomerization of a double bond during the initial randomization.
- Note that also the dihedrals of 13 and 14 changed in [2] according to Figure 1, but for this experiment they were kept
  as in [1] since they do not seem to affect the result.
- All CHARMM input scripts take only a ``name`` (e.g., 00061095) and ``rand1`` argument. The ``stream`` argument was removed.
- All ``3ob-mol/3ob-X.str`` where modified to take a ``qmreg`` parameter instead of ``name``.
- ``dyna_mm.inp`` was modified to save restarts every 1 ns instead of 100 fs.
- ``eval_mm.inp`` and ``eval_3ob.inp` has been modify to not include the energy of the positional harmonic restraint.
- Deleted input files for molecule ``01036618``, which was present in the dataset but was not actually included in the
  paper [1].


## References

[1] Kearns FL, Warrensford L, Boresch S, Woodcock HL. The good, the bad, and the ugly: “HiPen”, a new dataset for validating
    (S)QM/MM free energy simulations. Molecules. 2019 Jan;24(4):681.
[2] Schöller A, Kearns F, Woodcock HL, Boresch S. Optimizing the Calculation of Free Energy Differences in
    Nonequilibrium Work SQM/MM Switching Simulations. The Journal of Physical Chemistry B. 2022 Apr 11.