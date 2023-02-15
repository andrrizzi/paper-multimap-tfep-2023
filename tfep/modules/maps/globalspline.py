#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
MAF on cartesian coordinates using a neural spline transformer.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from tfep.nn.transformers.spline import NeuralSplineTransformer

from . import base


# =============================================================================
# TFEP MAP
# =============================================================================

class TFEPMap(base.TFEPMap):
    """MAF on cartesian coordinates using a neural spline transformer.

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

    def _get_transformer(self):
        """Build a neural spline transformer."""
        # Get minimum and maximum values of each coordinate.
        min_dofs, max_dofs = self._get_min_max_dofs()

        # Allow for an extra modification of coordinates by 1.5 angstrom.
        x0 = min_dofs - 1.5
        xf = max_dofs + 1.5

        return NeuralSplineTransformer(x0=x0, xf=xf, n_bins=5)
