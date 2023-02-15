#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Simple volume-preserving MAF on cartesian coordinates.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from tfep.nn.transformers.affine import VolumePreservingShiftTransformer

from . import base


# =============================================================================
# TFEP MAP
# =============================================================================

class TFEPMap(base.TFEPMap):
    """Simple volume-preserving MAF on cartesian coordinates.

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
        return VolumePreservingShiftTransformer()
