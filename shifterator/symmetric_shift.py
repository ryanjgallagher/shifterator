"""
symmetric_shift.py

TODO:
- Capitalize shift classes  
- Define options for reference value of JSD shift
- Decide how to handle missing types in JSD shift
- Change the axis / title labels for shifts
"""
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

import shifterator.shifterator as shifterator
from shifterator.helper import *

# ------------------------------------------------------------------------------
# -------------------------- Symmetric shift classes ---------------------------
# ------------------------------------------------------------------------------
class js_divergence_shift(shifterator.Shift):
    """
    Extra parameters: type of divergence (?), and alpha of entropy
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
