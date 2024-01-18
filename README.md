# Multimap targeted free energy estimation

This is a repository including code and input files to reproduce the work of the paper:

Andrea Rizzi, Paolo Carloni, Michele Parrinello. *Free energies at QM accuracy from force fields via multimap targeted estimation*. PNAS [DOI: 10.1073/pnas.2304308120](https://www.pnas.org/doi/10.1073/pnas.2304308120).

preprint version:

Andrea Rizzi, Paolo Carloni, Michele Parrinello. *Multimap targeted free energy estimation.* [arXiv preprint arXiv:2302.07683](http://arxiv.org/abs/2302.07683).

## Manifest

- ``amber/``: Scripts to compare the AMBER and CHARMM implementation of the potentials used in this work.
- ``hipen/``: Curated HiPen dataset with only the input files necessary for benchmarking (T)FEP. This includes only the
              input files to run the MM and evaluate energies at the DFTB level of theory (i.e., no BAR/JAR/CO).
- ``tfep/``:  Scripts to run the targeted free energy perturbation analysis.
- ``conda_environment.yml``: The conda environment I used to run the calculations, included for reproducibility. Not
                             all the packages in this environment might be actually required to run the calculations.

## HiPen dataset

The HiPen dataset files in this repository are a subset and slightly modified version of the original files
published at [doi.org/10.5281/zenodo.2328951](doi.org/10.5281/zenodo.2328951) under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode).
See the README in ``hipen/`` for a list of differences with the original files.

If you use these files, please acknowledge also the original authors of the HiPen dataset by including this citation:

Kearns FL, Warrensford L, Boresch S, Woodcock HL. *The good, the bad, and the ugly: “HiPen”, a new dataset for validating
(S)QM/MM free energy simulations*. Molecules. 2019; 24(4):681.
