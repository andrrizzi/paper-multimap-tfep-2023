#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test that the AMBER DFTB implementation of AMBER and CHARMM using 3ob-3-1 parameters are equivalent.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import glob
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np


# =============================================================================
# CONSTANTS
# =============================================================================

HIPEN_DIR = os.path.abspath('../hipen')
CHARMM_TOP_DIR = os.path.join(HIPEN_DIR, 'coors')
CGENFF_DIR = os.path.join(HIPEN_DIR, 'cgenff')
CGENFF_MOL_DIR = os.path.join(HIPEN_DIR, 'cgenff-mol')
AMBER_TOP_DIR = 'prmtop'

EV_TO_KCALMOL = 23.060541945


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def charmm_to_amber():
    """Convert all psf files to prmtop."""
    # Find all psf files.
    psf_file_paths = list(glob.glob(os.path.join(CHARMM_TOP_DIR, '*.psf')))

    # Create output directory.
    os.makedirs(AMBER_TOP_DIR, exist_ok=True)

    # Convert psf/crd to prmtop/inpcrd.
    for psf_file_path in psf_file_paths:
        # Derive crd, prmtop, and inpcrd file paths.
        file_name = os.path.splitext(os.path.basename(psf_file_path))[0]

        prmtop_file_path = os.path.join(AMBER_TOP_DIR, file_name+'.prmtop')
        inpcrd_file_path = os.path.join(AMBER_TOP_DIR, file_name+'.inpcrd')

        # Do not re-generate the output at each execution.
        if os.path.isfile(prmtop_file_path):
            continue

        crd_file_path = os.path.join(CHARMM_TOP_DIR, file_name+'.crd')
        rtf_file_path = os.path.join(CGENFF_DIR, 'top_all36_cgenff.rtf')
        prm_file_path = os.path.join(CGENFF_DIR, 'par_all36_cgenff.prm')
        str_file_path = os.path.join(CGENFF_MOL_DIR, file_name+'.str')

        # Write the parmed script to run.
        with tempfile.NamedTemporaryFile('w') as f:
            f.write(f'chamber -top {rtf_file_path} -param {prm_file_path} -str {str_file_path} -psf {psf_file_path} -crd {crd_file_path} -nocmap\n')
            f.write(f'outparm {prmtop_file_path}\n')
            # We don't need to output inpcrd since the coordinates that
            # we'll test will be taken from the MM trajectory.
            # f.write(f'outparm {prmtop_file_path} {inpcrd_file_path}\n')
            f.flush()

            # Execute parmed.
            subprocess.run(['parmed', '-i', f.name])


def read_charmm_energies(mol_id, out_dir_path, pot='3ob', rand1='1'):
    """Read all energies saved in the energy.000X.mm.pot.rand1.ene file.

    Parameters
    ----------
    mol_id : str
        The ID of the molecule (e.g., 00061095).

    Returns
    -------
    energies_charmm : numpy.ndarray
        Shape (n_frames,). The list of potential energies evaluated by
        ``eval_3ob.inp`` in CHARMM units (i.e., kcal/mol).

    """
    file_name = 'energy.' + mol_id + '.mm.' + pot + '.' + rand1 + '.ene'
    file_path = os.path.join(out_dir_path, 'energy', file_name)

    # Read CHARMM energies.
    energies_charmm = []
    with open(file_path, 'r') as f:
        for line_idx, line in enumerate(f):
            # Skip first line.
            if line_idx == 0:
                continue

            # Add the energy.
            energy = float(line.split()[1])
            energies_charmm.append(energy)

    return np.array(energies_charmm)


def compare_energies(pot, mol_ids=None):
    """Raise an error CHARMM and AMBER DFTB3 energies are different.

    The function selects 100 frames from the MM trajectory obtained for each
    molecule with rand1=1, computes the DFTB3 potential energy with AMBER and
    compares with the CHARMM energy. An error is raised if these are not
    sufficiently similar (using ``numpy.allclose()``).

    Parameters
    ----------
    pot : str
        Either 'mm' or '3ob'.
    mol_ids : set, optional
        A set of molecules ID (e.g., '04363792') to test. If not passed, all
        molecules will be tested.

    """
    import MDAnalysis
    from ase import Atoms
    from ase.calculators.amber import Amber

    # Use absolute paths since we'll cd into a different working directory.
    out_dir_path = os.path.join(HIPEN_DIR, 'out')
    traj_dir_path = os.path.join(out_dir_path, 'traj')
    amber_dir_path = os.path.abspath(AMBER_TOP_DIR)
    amber_in_file_path = os.path.abspath('amber_' + pot + '.in')
    offsets_json_file_path = os.path.abspath('offsets_' + pot + '.json')

    # Find all trajectory files. We analyze only a few samples of the first replicate.
    dcd_file_paths = list(glob.glob(os.path.abspath(os.path.join(traj_dir_path, '*.1.dcd'))))

    # Create a subdirectory where to execute amber. We don't create a
    # tempfile.TemporaryDirectory so that we can look for eventual errors.
    os.makedirs('tmp', exist_ok=True)
    os.chdir('tmp')

    # Create a directory where to save AMBER energies for debugging.
    amber_energy_dir_path = os.path.join('..', 'energy_amber', pot)
    os.makedirs(amber_energy_dir_path, exist_ok=True)

    # We can use the offsets saved in the 'offsets.json' file to check which
    # molecules have been already checked and implement resuming.
    try:
        with open(offsets_json_file_path, 'r') as f:
            offsets = json.load(f)
    except FileNotFoundError:
        offsets = {}

    # Analyze all molecules.
    for dcd_file_path in dcd_file_paths:
        # dcd file name is something like dyna.00140610.mm.1.dcd
        mol_id = os.path.basename(dcd_file_path)[5:13]
        file_name = 'zinc_' + mol_id

        # Check if we have already checked this molecule.
        if mol_id in offsets:
            continue

        # Check if we need to skip it.
        if (mol_ids is not None) and mol_id not in mol_ids:
            continue

        # The path is relative to the temporary directory.
        psf_file_path = os.path.join(CHARMM_TOP_DIR, file_name+'.psf')

        # Get the atom symbols and total charge.
        universe = MDAnalysis.Universe(psf_file_path, dcd_file_path)
        symbols = []
        tot_charge = 0.0
        for atom in universe.atoms:
            # atom.name is something like "C12" so we need to remove the numbers.
            name = ''.join([c for c in atom.name if not c.isdigit()])
            symbols.append(name.capitalize())
            tot_charge += atom.charge
        tot_charge = round(tot_charge)

        # Copy the AMBER input file in the temporary directory.
        # If this is a QM run, we also need to add the qmcharge.
        if pot == 'mm':
            shutil.copy(amber_in_file_path, 'amber.in')
        else:
            print('Fixing the QM charge of', mol_id, 'to', tot_charge)
            with open(amber_in_file_path, 'r') as fin, open('amber.in', 'w') as fout:
                for line in fin:
                    if 'qmcharge=0' in line:
                        line = line.replace('qmcharge=0', 'qmcharge='+str(tot_charge))
                    fout.write(line)

        # Take 100 equally spaced frames.
        step = universe.trajectory.n_frames // 100
        traj = universe.trajectory[::step]
        print('Evaluating', mol_id, 'trajectory (', universe.trajectory.n_frames, 'frames ) every', step, 'steps')

        # Read CHARMM energies.
        energies_charmm = read_charmm_energies(mol_id, out_dir_path=out_dir_path, pot=pot)
        assert len(energies_charmm) == universe.trajectory.n_frames
        energies_charmm = energies_charmm[::step]

        # Create AMBER calculator.
        calc = Amber(
            amber_exe='sander -O',
            infile='amber.in',
            outfile='amber.out',
            topologyfile=os.path.join(amber_dir_path, file_name+'.prmtop'),
            incoordfile='coord.crd',
        )

        # Compute the energies of all frames with AMBER.
        energies_amber = []
        for frame_idx, ts in enumerate(traj):
            atoms = Atoms(
                symbols=symbols,
                positions=ts.positions,
                calculator=calc,
            )
            energies_amber.append(atoms.get_potential_energy())

        # Convert eV to kcal/mol.
        energies_amber = np.array(energies_amber) * EV_TO_KCALMOL

        # Energies are for some reason off by a constant factor.
        offset = (energies_charmm - energies_amber).mean()
        energies_amber = energies_amber + offset
        max_offset = np.max(np.abs(energies_charmm - energies_amber))
        print('Offset for', mol_id, 'is', offset, 'with maximum deviation', max_offset)

        # Saving energies in case we need to inspect them after an error.
        np.save(os.path.join(amber_energy_dir_path, file_name+'.npy'), energies_amber)

        # Update offsets.
        offsets[mol_id] = offset
        with open(offsets_json_file_path, 'w') as f:
            json.dump(offsets, f)

        # Compare energies.
        assert np.allclose(energies_amber, energies_charmm), 'Assertion failed for ' + mol_id

    # Restore working directory.
    os.chdir('..')


# =============================================================================
# MAIN
# =============================================================================

def test_amber():
    """Convert CHARMM topologies into AMBER and compare DFTB3 potential energies."""
    # Convert input files from CHARMM to AMBER.
    charmm_to_amber()

    # Compare energies at the DFTB/3ob level.
    compare_energies(pot='3ob')

    # In the paper we ran MD at the MM level with AMBER only for 00061095 and
    # 00095858 so we test only the energies for this one. Other molecules might
    # have problems depending on how ParmED convert the input files.
    compare_energies(pot='mm', mol_ids={'00061095', '00095858'})


if __name__ == '__main__':
    test_amber()
