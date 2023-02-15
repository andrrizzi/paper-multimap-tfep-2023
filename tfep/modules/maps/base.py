#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Simple MAF on cartesian coordinates.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import MDAnalysis
import numpy as np
import torch

import tfep.io
from tfep.nn import flows
from tfep.utils.misc import atom_to_flattened_indices


# =============================================================================
# TFEP MAP
# =============================================================================

class TFEPMap(torch.nn.Module):
    """Simple Masked Autoregressive Flow map in Cartesian coordinates.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        An MDAnalysis ``Universe`` object encapsulating both the topology and
        the trajectory. This information is used to obtain the topology of the
        molecule and build a Z-matrix.
    n_maf_layers : int, optional
        The number of MAF layers of the transformation flow.
    n_maf_layers_ic : int, optional
        The number of MAF layers used for the change of coordinate flow.
    **kwargs
        Other keyword arguments for the MAF layers.

    """

    def __init__(
            self,
            universe,
            n_maf_layers=6,
            n_maf_layers_ic=0,
            **kwargs
    ):
        super().__init__()

        # Determine a Z-matrix for the molecule.
        self.universe = universe
        self.z_matrix = self._build_z_matrix()

        # Build the MAF flow.
        flow = self._build_maf_flow(n_maf_layers, **kwargs)

        # Change of coordinate layer.
        if n_maf_layers_ic > 0:
            coord_flow = self._build_maf_flow(n_maf_layers_ic, **kwargs)
        else:
            coord_flow = None

        # We need to reduce the input dimension from 3N to 3N-6. This is done by
        # either cart2ic_flow or by the Oriented/CenteredCentroidFlow.
        cart2ic_flow = self._build_cart2ic_flow()
        if cart2ic_flow is not None:
            flow = ChangeOfCoordinatesFlow(flow, cart2ic_flow=cart2ic_flow, coord_flow=coord_flow)
        else:
            # THe transformation is performed in cartesian space.
            if coord_flow is not None:
                flow = ChangeOfCoordinatesFlow(flow, coord_flow=coord_flow)

            # Fix translational/rotational DOFs.
            flow = self._build_partial_flow(flow)

        self.flow = flow

    @property
    def n_dofs(self):
        """Number of degrees of freedom excluding translational/rotational DOFs."""
        return self.universe.atoms.n_atoms * 3 - 6

    @property
    def atom_order(self):
        """The order of the atom followed by the Z-matrix."""
        return self.z_matrix[:, 0]

    def eval(self):
        super().eval()
        self.flow.regularization = False  # For continuous flows.

    def forward(self, x):
        """Return the mapped configuration and log absolute Jacobian."""
        return self.flow(x)

    def inverse(self, y):
        """Run the flow in the reversed direction."""
        return self.flow.inverse(y)

    def _get_maf_degrees_in(self):
        """Return the degrees of each DOF."""
        # Here we return the degrees based on the order in the Z-matrix for a
        # fairer comparison with the Z-matrix based coordinates.
        dofs_order = atom_to_flattened_indices(self.atom_order).tolist()

        # Remove the translational/rotational DOFs.
        excluded_dofs = atom_to_flattened_indices(self.atom_order[:3]).tolist()
        excluded_dofs = set(excluded_dofs[:3] + excluded_dofs[4:6] + excluded_dofs[8:9])
        dofs_order = np.array([x for x in dofs_order if x not in excluded_dofs])

        # Convert DOFs order to MAF degree.
        degrees_in = np.argsort(dofs_order)
        return degrees_in

    def _get_periodic(self):
        """Return periodic_indices and periodic_limits arguments for MAF."""
        # No periodic degrees of freedom.
        return None, None

    def _get_transformer(self):
        """Return the transformer for MAF."""
        # Default transformer (affine).
        return None

    def _get_atom_zmatrix_priorities(self, atom, graph_distances, added_atoms, is_h, bond_atom=None):
        """Build priority list for this atom.

        Atoms are prioritized with the following criterias (in this order).
        1) Closest to ``atom``.
        2) Closest to the ``bond_atom`` if passed as an argument (for angle and torsion atoms).
        3) Prioritize heavy atoms if ``atom`` is not a hydrogen.

        """
        # priorities[i][0] is the atom index.
        # priorities[i][1] is the distance (in number of edges) from atom.
        # priorities[i][2] is the distance (in number of edges) from bond_atom.
        # priorities[i][3] is 1 if it is a hydrogen and atom_idx isn't, else 0.
        # This way we can select the reference atoms simply by sorting a list.
        priorities = []
        for prev_atom, dist in graph_distances[atom].items():
            # atom_idx cannot depend on itself and on atoms that are not already in the Z-matrix.
            if (prev_atom.index not in added_atoms) or (atom.index == prev_atom.index):
                continue

            if bond_atom is None:
                # Set all bond distances to the same value to avoid prioritization based on this criteria.
                bond_atom_dist = 0
            elif prev_atom.index == bond_atom.index:
                # Do not add bond_atom twice.
                continue
            elif prev_atom not in graph_distances[bond_atom]:
                # prev_atom needs to be close to the bond atom as well.
                continue
            else:
                bond_atom_dist = graph_distances[bond_atom][prev_atom]

            priorities.append([
                prev_atom.index,
                dist,
                bond_atom_dist,
                float(not is_h and _is_hydrogen(prev_atom)),
            ])

        # The minus sign of the atom order is because we want to prioritize atoms that have just been added.
        priorities.sort(key=lambda k: (k[1], k[2], k[3]))
        return priorities

    def _build_z_matrix(self):
        """Build a Z-matrix for the system."""
        import networkx as nx

        # Build a graph representing the topology.
        graph = nx.Graph()
        graph.add_nodes_from(self.universe.atoms)
        graph.add_edges_from(self.universe.bonds)

        if not nx.is_connected(graph):
            raise ValueError('Only connected graphs (i.e., single molecules) are supported.')

        # graph_distances[i, j] is the distance (in number of edges) between atoms i and j.
        # We don't need paths longer than 3 edges as we'll select torsion atoms prioritizing
        # closer atoms.
        graph_distances = dict(nx.all_pairs_shortest_path_length(graph, cutoff=3))

        # Select the first atom as the graph center.
        center_atom = nx.center(graph)[0]
        z_matrix = [[center_atom.index, -1, -1, -1]]

        # atom_order[atom_idx] is the row index of the Z-matrix defining its coords.
        added_atoms = set([center_atom.index])

        # We traverse the graph breadth first.
        for _, added_atom in nx.bfs_edges(graph, source=center_atom):
            z_matrix_row = [added_atom.index]

            # Find bond atom.
            is_h = _is_hydrogen(added_atom)
            priorities = self._get_atom_zmatrix_priorities(added_atom, graph_distances, added_atoms, is_h)
            z_matrix_row.append(priorities[0][0])

            # For angle and torsion atoms, adds also the distance to the bond atom
            # in the priorities. This reduces the chance of collinear torsions.
            bond_atom = self.universe.atoms[z_matrix_row[-1]]
            priorities = self._get_atom_zmatrix_priorities(added_atom, graph_distances, added_atoms, is_h, bond_atom)
            z_matrix_row.extend([p[0] for p in priorities[:2]])

            # The first two added atoms are the reference atoms.
            if len(z_matrix_row) < 4:
                assert len(z_matrix) < 4
                z_matrix_row = z_matrix_row + [-1] * (4-len(z_matrix_row))

            # Add entry to Z-matrix.
            z_matrix.append(z_matrix_row)

            # Add this atom to those added to the Z-matrix.
            added_atoms.add(added_atom.index)

        return np.array(z_matrix)

    def _build_maf_flow(self, n_maf_layers, **maf_kwargs):
        """Build MAF layers."""
        # Determine the degrees of each DOF based on atom order.
        degrees_in = self._get_maf_degrees_in()

        # Determine the transformer and the periodic degrees of freedom.
        transformer = self._get_transformer()
        periodic_indices, periodic_limits = self._get_periodic()

        # Build layers.
        maf_layers = []
        for layer_idx in range(n_maf_layers):
            even_layer = layer_idx%2 == 0
            maf_layers.append(flows.MAF(
                dimension_in=self.n_dofs,
                periodic_indices=periodic_indices,
                periodic_limits=periodic_limits,
                degrees_in=degrees_in if even_layer else np.flip(degrees_in),
                transformer=transformer,
                **maf_kwargs
            ))
        return flows.SequentialFlow(*maf_layers)

    def _build_cart2ic_flow(self):
        """Initialize the layer transforming cartesian to internal coordinates (optional)."""
        # By default, work in cartesian coordinates.
        return None

    def _build_partial_flow(self, flow, restore=True):
        """Encapsulate the flow in layers that map only a subset of the DOFs."""
        translated_atom_idx, axis_atom_idx, plane_atom_idx = self.atom_order[:3]

        # Add rotational invariance.
        flow = flows.OrientedFlow(
            flow,
            axis_point_idx=axis_atom_idx,
            plane_point_idx=plane_atom_idx,
            rotate_back=restore,
            return_partial=not restore,
        )

        # Add translational invariance.
        flow = flows.CenteredCentroidFlow(
            flow,
            space_dimension=3,
            subset_point_indices=[translated_atom_idx],
            translate_back=restore,
            return_partial=not restore,
        )

        return flow

    def _get_min_max_dofs(self):
        """Compute the minimum and maximum value of each DOF in the trajectory.

        These are the coordinate after the first partial flow (built with
        ``self._build_partial_flow`` has been executed.

        This can be used to calculate appropriate values for the left/rightmost
        nodes of a neural spline transformer.

        Returns
        -------
        min_dofs : torch.Tensor
            ``min_dofs[i]`` is the minimum value of the ``i``-th degree of freedom.
        max_dofs : torch.Tensor
            ``max_dofs[i]`` is the maximum value of the ``i``-th degree of freedom.

        """
        # We need to calculate the minimum and maximum dof AFTER it has gone
        # through the partial flow as this is the input that will be passed to
        # the transformers.
        identity_flow = lambda x_: (x_, torch.zeros_like(x_[:, 0]))
        flow = self._build_partial_flow(identity_flow, restore=False)

        # Initialize returned values.
        min_dofs = torch.full((self.n_dofs,), float('inf'))
        max_dofs = torch.full((self.n_dofs,), -float('inf'))

        # Read the trajectory in batches.
        dataset = tfep.io.TrajectoryDataset(
            universe=self.universe,
            return_dataset_sample_index=False,
            return_trajectory_sample_index=False
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1024, shuffle=False, drop_last=False
        )
        for batch_data in data_loader:
            # Go through partial flow.
            dofs, _ = flow(batch_data['positions'])

            # Take the min/max across the batch.
            batch_min = torch.min(dofs, dim=0).values
            batch_max = torch.max(dofs, dim=0).values

            # Update current min/max.
            min_dofs = torch.minimum(min_dofs, batch_min)
            max_dofs = torch.maximum(max_dofs, batch_max)

        return min_dofs, max_dofs


# =============================================================================
# CHANGE OF COORDINATES FLOW
# =============================================================================

class ChangeOfCoordinatesFlow(torch.nn.Module):
    """Wraps a flow and execute it in a different coordinate space.

    The final flow is given by.

    cart2ic -> coord_flow -> flow -> coord_flow^-1 -> cart2ic^-1.

    """
    def __init__(self, flow, cart2ic_flow=None, coord_flow=None):
        super().__init__()
        self.flow = flow
        self.cart2ic_flow = cart2ic_flow
        self.coord_flow = coord_flow

    def forward(self, x):
        return self._transform(x, inverse=False)

    def inverse(self, x):
        return self._transform(x, inverse=True)

    def _transform(self, x, inverse):
        cumulative_log_det_J = torch.zeros_like(x[:, 0])

        # Change the coordinate system.
        if self.cart2ic_flow is not None:
            x, log_det_J, x0, R = self.cart2ic_flow(x)
            cumulative_log_det_J = cumulative_log_det_J + log_det_J
        if self.coord_flow is not None:
            x, log_det_J = self.coord_flow(x)
            cumulative_log_det_J = cumulative_log_det_J + log_det_J

        # Run the flow in the transformed coordinate.
        if inverse:
            y, log_det_J = self.flow.inverse(x)
        else:
            y, log_det_J = self.flow(x)
        cumulative_log_det_J = cumulative_log_det_J + log_det_J

        # Go back to original coordinate system.
        if self.coord_flow is not None:
            y, log_det_J = self.coord_flow.inverse(y)
            cumulative_log_det_J = cumulative_log_det_J + log_det_J
        if self.cart2ic_flow is not None:
            y, log_det_J = self.cart2ic_flow.inverse(y, x0, R)
            cumulative_log_det_J = cumulative_log_det_J + log_det_J

        return y, cumulative_log_det_J


# =============================================================================
# UTILS
# =============================================================================

def _get_atom_element(atom, uppercase=True):
    """Get the element of the atom.

    If atom.element does not have element information, it is inferred from the name.

    """
    # The Universe doesn't always have element information.
    try:
        element = atom.element
    except MDAnalysis.exceptions.NoDataError:
        # a.name is something like 'C1', 'H3', 'CL12'.
        element = ''.join([x for x in atom.name if not x.isdigit()])
    if uppercase:
        return element.upper()
    return element


def _is_hydrogen(atom):
    return _get_atom_element(atom) == 'H'
