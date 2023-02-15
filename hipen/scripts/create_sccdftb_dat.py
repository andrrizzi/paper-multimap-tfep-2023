#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Build the sccdftb.dat file for each molecule.

The sccdftb.dat must point to the files corresponding to the atom types of the
specific molecule, which are defined in the ``3ob-mol/3ob-000X.str`` files.

The script creates one ``3ob-mol/sccdftb-000X.dat`` file for each molecule.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import glob
import os


# =============================================================================
# CONSTANTS
# =============================================================================

HUBBARD_DERIVATIVES = {
    'Br': -0.0573,
    'C': -0.1492,
    'Ca': -0.0340,
    'Cl': -0.0697,
    'F': -0.1623,
    'H': -0.1857,
    'I': -0.0433,
    'K': -0.0339,
    'Mg': -0.02,
    'N': -0.1535,
    'Na': -0.0454,
    'O': -0.1575,
    'P': -0.14,
    'S': -0.11,
    'Zn': -0.03,
}

# =============================================================================
# MAIN
# =============================================================================

def main():
    # We need to parse all the 3ob-000X files that contain the definition of atom types.
    atom_type_files = list(glob.glob('../3ob-mol/3ob-*.str'))
    for atom_type_file in atom_type_files:
        print('Processing', atom_type_file)

        # Id of the molecule (e.g., 00061095).
        mol_id = os.path.basename(atom_type_file)[4:-4]

        # Extract all atom types for this molecule.
        atom_types = []
        with open(atom_type_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('scalar wmain'):
                    line = line.split()
                    # Remove the asterisk after the type name (e.g., "CL*").
                    atom_type = line[6][:-1]
                    # Make sure only the first letter is uppercase (e.g., CL -> Cl).
                    atom_types.append(atom_type.capitalize())

        # Create sccdftb.dat file.
        sccdftb_dat_file_path = '../3ob-mol/sccdftb-' + mol_id + '.dat'
        with open(sccdftb_dat_file_path, 'w') as f:
            # Create the paths to the parameter files.
            for at1 in atom_types:
                for at2 in atom_types:
                    f.write(f"'../../3ob-3-1/{at1}-{at2}.skf'\n")

            # Add Hubbard derivatives.
            for at in atom_types:
                hubbard_derivative = HUBBARD_DERIVATIVES[at]
                f.write(f"'{at}' {hubbard_derivative}\n")

            # Add zeta parameter.
            f.write('4.00')


if __name__ == '__main__':
    main()
