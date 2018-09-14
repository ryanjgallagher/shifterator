"""
symmetric_shift.py

Author: Ryan J. Gallagher, Network Science Institute, Northeastern University
Last updated: June 13th, 2018

TODO:
- Define symmetric shift class
- Define divergence shift class
"""
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

import shifterator

# ------------------------------------------------------------------------------
# -------------------------- SYMMETRIC SHIFT CLASSES ---------------------------
# ------------------------------------------------------------------------------
class js_divergence_shift(shifterator.shift):
    """
    Extra parameters: type of divergence (?), and alpha of entropy
    """
    def __init__(self, system_1, system_2, type2score_1=None, type2score_2=None,
                 base=2, weight_1=0.5, weight_2=0.5, alpha=1, stop_lens=None):
        # Normalize reference and comparison frequencies
        n_ref = sum(reference.values())
        type2p = {t:s/n_ref for t,s in reference.items()}
        n_comp = sum(comparison_text.values())
        type2q = {t:s/n_comp for t,s in comparison.items()}
        # Get mixed distribution
        type2m = get_mixed_distribution(type2p, type2q, p=weight_1, q=weight_2)
        # Get surprisal of each type
        type2surprisal_p = get_type_surprisals(type2p, base=base, alpha=alpha)
        type2surprisal_q = get_type_surprisals(type2q, base=base, alpha=alpha)
        type2surprisal_m = get_type_surprisals(type2m, base=base, alpha=alpha)
        # Get scores (handle missing types)
        type2score_1 =
        type2score_2 =
        # Initialize shift object
        shift.__init__(system_1, system_2, type2score_1, type2score_2,
                       stop_lens=stop_lens, delimiter=delimiter)



def get_type_surprisals(type2p, base=2, alpha=alpha):
    """

    """
    pass

def get_mixed_distribution(type2p, type2q, p=0.5, q=0.5):
    """

    """
    types = set(type2p.keys()).union(set(type2q.keys()))
    type2m = dict()
    for t in types:
        if t in type2p and t in type2q:
            type2m[t] = p*type2p[t] + q*type2q[t]
        elif t in type2p:
            type2m[t] = p*type2p[t]
        else:
            type2m[t] = q*type2q[t]
    return type2m
