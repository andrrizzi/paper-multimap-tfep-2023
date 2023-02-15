#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Simple volume-preserving MAF on Z-matrix coordinates.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch

from tfep.nn.transformers.affine import VolumePreservingShiftTransformer

from . import zmatrix


# =============================================================================
# TFEP MAP
# =============================================================================

class TFEPMap(zmatrix.TFEPMap):
    """Simple volume-preserving MAF operating on Z-matrix coordinates.

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
        """Return the transformer for MAF."""
        periodic_indices, periodic_limits = self._get_periodic()
        return VolumePreservingShiftTransformer(
            periodic_indices=torch.tensor(periodic_indices),
            periodic_limits=torch.tensor(periodic_limits),
        )
