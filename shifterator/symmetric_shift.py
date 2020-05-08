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
# same reason here

# ------------------------------------------------------------------------------
# -------------------------- Symmetric shift classes ---------------------------
# ------------------------------------------------------------------------------
class ProportionShift(shifterator.Shift):
    """

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
                                   reference_value=0, step_lens=stop_lens,)


class JSDivergenceShift(shifterator.Shift):
    """
    Extra parameters: alpha of entropy
    """
    def __init__(self, system_1, system_2, base=2, weight_1=0.5, weight_2=0.5,
                 alpha=1, stop_lens=None):
        # Get JSD scores
        type2score_1, type2score_2 = get_jsd_scores(system_1, system_2,
                                                    weight_1=weight_1,
                                                    weight_2=weight_2,
                                                    base=base, alpha=alpha,)
        # Initialize shift object
        shifterator.Shift.__init__(self, system_1=system_1, system_2=system_2,
                                   type2score_1=type2score_1,
                                   type2score_2=type2score_2,
                                   reference_value=0, stop_lens=stop_lens)
