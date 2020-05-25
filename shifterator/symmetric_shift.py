"""
symmetric_shift.py

TODO:
- Change the axis / title labels for shifts
"""
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

import shifterator.shifterator as shifterator
from .helper import *

# ------------------------------------------------------------------------------
# -------------------------- Symmetric shift classes ---------------------------
# ------------------------------------------------------------------------------
class ProportionShift(shifterator.Shift):
    """
    Shift object for calculating differences in proportions of types across two
    systems.

    Parameters
    __________
    system_1, system_2: dict
        keys are types of a system and values are frequencies of those types
    stop_lens: list
        currently not implemented, but left for later updates
    """
    def __init__(self, system_1, system_2, stop_lens=None):
        # Set relative frequency to 0 for types that don't appear
        types = set(system_1.keys()).union(system_2.keys())
        for t in types:
            if t not in system_1:
                system_1[t] = 0
            elif t not in system_2:
                system_2[t] = 0
        # Initialize shift object
        shifterator.Shift.__init__(self, system_1=system_1, system_2=system_2,
                                   type2score_1=None, type2score_2=None,
                                   reference_value=0, stop_lens=stop_lens,)

    def get_shift_graph(self, top_n=50, normalize=False, text_size_inset=True,
                        cumulative_inset=False, show_plot=True, filename=None,
                        detailed=False, **kwargs):
        shifterator.Shift.get_shift_graph(self, top_n=top_n, normalize=normalize,
                                          text_size_inset=text_size_inset,
                                          cumulative_inset=cumulative_inset,
                                          show_plot=show_plot, filename=filename,
                                          detailed=detailed, **kwargs)


class JSDivergenceShift(shifterator.Shift):
    """
    Shift object for calculating the Jensen-Shannon divergence (JSD) between two
    systems

    Parameters
    __________
    system_1, system_2: dict
        keys are types of a system and values are frequencies of those types
    base: int
        the base for the logarithm when computing entropy for the JSD
    weight_1, weight_2: float
        relative weights of system_1 and system_2 when constructing their mixed
        distribution. Should sum to 1
    alpha: float
        currently not implemented, but left for later updates
    """
    def __init__(self, system_1, system_2, base=2, weight_1=0.5, weight_2=0.5,
                 alpha=1, stop_lens=None):
        # Check weights
        if weight_1 + weight_2 != 1:
            raise ValueError('weight_1 and weight_2 do not sum to 1')
        # Get JSD scores
        type2p,type2q,type2m,type2score_1,type2score_2 = get_jsd_scores(system_1, system_2,
                                                                        weight_1=weight_1,
                                                                        weight_2=weight_2,
                                                                        base=base, alpha=alpha,)
        # Initialize shift object
        shifterator.Shift.__init__(self, system_1=system_1, system_2=system_2,
                                   type2score_1=type2score_1,
                                   type2score_2=type2score_2,
                                   reference_value=0, stop_lens=stop_lens)
        self.type2p_1 = type2p
        self.type2p_2 = type2q
        self.type2p_mixed = type2m

    def get_shift_graph(self, top_n=50, normalize=True, text_size_inset=True,
                        cumulative_inset=True, show_plot=True, filename=None,
                        detailed=False, show_total=False, **kwargs):
        shifterator.Shift.get_shift_graph(self, top_n=top_n, normalize=normalize,
                                          text_size_inset=text_size_inset,
                                          cumulative_inset=cumulative_inset,
                                          show_plot=show_plot, filename=filename,
                                          detailed=detailed, show_total=show_total,
                                          all_pos_contributions=True, **kwargs)
