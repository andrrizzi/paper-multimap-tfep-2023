#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Simple MAF on Z-matrix coordinates.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import torch

from tfep.nn.transformers.spline import NeuralSplineTransformer

from . import base


# =============================================================================
# TFEP MAP
# =============================================================================

class TFEPMap(base.TFEPMap):
    """Simple MAF operating on Z-matrix coordinates.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        An MDAnalysis ``Universe`` object encapsulating both the topology and
        the trajectory. This information is used
    n_maf_layers : int, optional
        The number of MAF layers of the transformation flow.
    n_maf_layers_ic : int, optional
        The number of MAF layers used for the change of coordinate flow.
    **kwargs
        Other keyword arguments for the MAF layers.

    """

    def _get_maf_degrees_in(self):
        """Return the degrees of each DOF."""
        # Here we return the input order since the conversion to Z-matrix already
        # reorder the atoms according to the Z-matrix.
        return np.arange(self.n_dofs)

    def _get_periodic(self):
        """Return periodic_indices and periodic_limits arguments for MAF."""
        # The first five DOFs of a Z-matrix are (non-periodic) bonds and angles.
        # Then a (periodic) dihedral every 3 DOFs.
        periodic_indices = list(range(5, self.n_dofs, 3))

        # The periodic limits are 0 to 1 if normalize_angles=True in cart2ic_flow
        # (which is the default).
        periodic_limits = [0, 1]
        return periodic_indices, periodic_limits

    def _get_transformer(self):
        """Return the transformer for MAF."""
        # Use a neural spline since there are periodic DOFs.
        periodic_indices, _ = self._get_periodic()

        # Initialize using the limits for the (normalized) angles.
        x0 = torch.zeros(self.n_dofs)
        xf = torch.ones(self.n_dofs)

        # Now fill the limits for the bonds (in Angstrom).
        bonds_indices = [0, 1] + list(range(3, self.n_dofs, 3))
        x0[bonds_indices] = 0.5
        xf[bonds_indices] = 3.0

        return NeuralSplineTransformer(
            x0=x0,
            xf=xf,
            n_bins=5,
            circular=periodic_indices,
        )

    def _build_cart2ic_flow(self):
        """Initialize the layer transforming cartesian to internal coordinates (optional)."""
        cart2ic_flow = _CartesianToInternalCoordinateTransformer(self.z_matrix)
        return cart2ic_flow

    def _build_partial_flow(self, flow, restore=True):
        """Encapsulate the flow in layers that map only a subset of the DOFs."""
        # This should never be called.
        assert False


# =============================================================================
# UTILITY CLASSES
# =============================================================================

class _CartesianToInternalCoordinateTransformer(torch.nn.Module):
    r"""A transformer to convert cartesian to internal coordinates.

    Wraps the GlobalInternalCoordinateTransformation class in bgflow to provide
    a Flow API that converts Cartesian coordinates into internal coordinates.

    Parameters
    ----------
    z_matrix : numpy.ndarray
        Shape ``(n_atoms, 4)``. A Z-matrix. E.g., ``z_matrix[i] == [7, 2, 4, 8]``
        means that the distance, angle, and dihedral for atom ``7`` must be
        computed between atoms ``7-2``, ``7-2-4``, and ``7-2-4-8`` respectively.
        Empty element of the Z-matrix are indicated with ``-1``. E.g.
        ``z_matrix = [[3, -1, -1, -1], [0, 3, -1, -1], [7, 0, 3, -1], ...]``
        will use atoms 3, 0, and 7 to determine the frame of reference.
    **kwargs
        Keywords arguments for ``bgflow.nn.flow.crd_transform.ic.GlobalInternalCoordinateTransformation``.

    """

    def __init__(self, z_matrix, **kwargs):
        super().__init__()
        from bgflow.nn.flow.crd_transform.ic import GlobalInternalCoordinateTransformation
        self.ic = GlobalInternalCoordinateTransformation(z_matrix=z_matrix, **kwargs)

    def forward(self, x):
        r"""Forward transformation.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, n_atoms*3). Cartesian coordinates.

        Returns
        -------
        ics : torch.Tensor
            Shape ``(batch_size, n_atoms*3-6)``. The internal coordinates.
        log_det_J : torch.Tensor
            Shape ``(batch_size,)``. The log det Jacobian of the transformation.
        x0: torch.Tensor
            Shape ``(batch_size, 1, 3)``. The systems origin point. This is
            required for inversion.
        R: torch.Tensor
            Shape ``(batch_size, 3)``. Global rotation of the system - 3-vector
            of Euler angles. This is required for inversion.
        """
        batch_size = x.shape[0]

        # Apply the IC transformation.
        bonds, angles, dihedrals, x0, R, log_det_J = self.ic.forward(x)

        # Re-arrange by atom.
        ics = torch.stack([bonds[:, 2:], angles[:, 1:], dihedrals], dim=2)

        # From (batch, 1) to (batch,)
        log_det_J = log_det_J.squeeze(-1)

        # From (batch, n_atoms-3, 3) to (batch, (n_atoms-3)*3).
        ics = ics.reshape(batch_size, -1)

        # Add coordinates of reference system atoms.
        ics = torch.cat([bonds[:, :2], angles[:, :1], ics], dim=1)
        return ics, log_det_J, x0, R

    def inverse(self, ics, x0, R):
        """Inverse transformation.

        Parameters
        ----------
        ics : torch.Tensor
            Shape (batch_size, n_atoms*3 - 6). Internal coordinates.
        x0: torch.Tensor
            Shape ``(batch_size, 1, 3)``. The systems origin point.
        R: torch.Tensor
            Shape ``(batch_size, 3)``. Global rotation of the system - 3-vector
            of Euler angles.

        Returns
        -------
        x : torch.Tensor
            Shape (batch_size, n_atoms*3). Cartesian coordinates.
        log_det_J : torch.Tensor
            Shape ``(batch_size,)``. The log det Jacobian of the transformation.
        """
        batch_size = ics.shape[0]

        # Separate the DOFs from the reference system atoms.
        ics_ref = ics[:, :3]
        ics = ics[:, 3:]

        # Separate bonds, angles, and dihedral
        bonds = torch.cat([ics_ref[:, :2], ics[:, ::3]], dim=1)
        angles = torch.cat([ics_ref[:, 2:3], ics[:, 1::3]], dim=1)
        dihedrals = ics[:, 2::3]

        # Apply the ic transformation.
        x, log_det_J = self.ic.forward(bonds, angles, dihedrals, x0, R, inverse=True)

        # From (batch, 1) to (batch,)
        log_det_J = log_det_J.squeeze(-1)

        return x, log_det_J.squeeze(-1)
