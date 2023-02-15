#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
A script to convert AMBER's NetCDF trajectories to dcd format that CHARMM can read.

The script assumes that the NetCDF trajectory is saved in a file called "mm_plumed.ncdf"
in the same directory of the script. The DCD trajectory will be saved as
"traj/dyna.@name.mm.@rand1.dcd", which is what the "eval_dihe.inp" CHARMM script
expects.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import MDAnalysis
from MDAnalysis.coordinates.DCD import DCDWriter


# =============================================================================
# MAIN
# =============================================================================

def amber_netcdf_to_dcd(mol_id, rand1):
    """Convert a trajectory in AMBER's NetCDF format to dcd.

    See also module docstrings.
    """
    psf_file_path = 'zinc_' + mol_id + '.psf'
    in_traj_file_path = 'mm_plumed.ncdf'
    out_traj_file_path = f'traj/dyna.' + mol_id + '.mm.' + rand1 + '.dcd'

    # Read in trajectory.
    universe = MDAnalysis.Universe(psf_file_path, in_traj_file_path, format='NCDF')
    with DCDWriter(
            filename=out_traj_file_path,
            n_atoms=universe.atoms.n_atoms,
            dt=0.1,  # Time between frames in ps.
            nsavc=100,  # Number of integration time steps between frames.
            istart=5000100,  # Number of equilibration time steps + nsavc.
    ) as writer:
        for ts in universe.trajectory:
            writer.write(universe)


if __name__ == '__main__':
    # The arguments passed through the command line overwrite the SLURM variables.
    import argparse
    parser = argparse.ArgumentParser()

    # Options for both standard and targeted reweighting.
    parser.add_argument('--molid', dest='mol_id', help='The ID of the molecule (e.g., 00140610).')
    parser.add_argument('--rand1', dest='rand1', help='The repeat number.')
    args = parser.parse_args()

    amber_netcdf_to_dcd(**vars(args))
