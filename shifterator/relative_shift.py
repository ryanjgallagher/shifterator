"""
relative_shift.py

TODO:
- Add option to get simple contributions for entropy, KLD, JSD (no breakdown)
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
# --------------------------- Relative shift classes ---------------------------
# ------------------------------------------------------------------------------
class RelativeShift(shifterator.Shift):
    def __init__(self, reference, comparison, type2score_ref=None,
                 type2score_comp=None, stop_lens=None, reference_value=None):
        """
        Shift object for calculating the relative shift of a comparison system
        from a reference system

        Parameters
        ----------
        reference, comparison: dict
            keys are types of a system and values are frequencies of those types
        type2score_ref, type2score_comp: dict or str, optional
            if dict, types are keys and values are "scores" associated with each
            type (e.g., sentiment). If str, the name of a score dict. If None
            and other type2score is None, defaults to uniform scores across
            types. Otherwise defaults to the other type2score dict
        stop_lens: iterable of 2-tuples, optional
            denotes intervals that should be excluded when calculating shift
            scores
        reference_value: float, optional
            the reference score from which to calculate the deviation. If None,
            defaults to the weighted score of reference
        """
        shifterator.Shift.__init__(self, system_1=reference, system_2=comparison,
                                   type2score_1=type2score_ref,
                                   type2score_2=type2score_comp,
                                   stop_lens=stop_lens,
                                   reference_value=reference_value)
        # Set new names for interpretability (takes up memory...)
        self.type2freq_ref = self.type2freq_1
        self.type2freq_comp = self.type2freq_2
        self.type2score_ref = self.type2score_1
        self.type2score_comp = self.type2score_2


class SentimentShift(RelativeShift):
    def __init__(self, reference, comparison, sent_dict_ref='labMT_english',
                 sent_dict_comp=None, stop_lens=None, reference_value=None):
        """
        Shift object for calculating the relative shift in sentiment of a
        comparison text from a reference text

        Parameters
        ----------
        reference, comparison: dict
            keys are word types of a text and values are frequencies of those
            types
        type2score_ref, type2score_comp: dict or str, optional
            if dict, word types are keys and values are sentiment scores
            associated with each type. If str, the name of a sentiment
            dict. If None and other type2score is None, defaults to uniform
            sentment across types, i.e. shift is in terms of just frequency,
            not sentiment. Otherwise defaults to the other type2score dict
        stop_lens: iterable of 2-tuples, optional
            denotes intervals that should be excluded when calculating shift
            scores. Defaults to stop lens for labMT sentiment dictionary
        reference_value: float, optional
            the reference score from which to calculate the deviation. If None,
            defaults to the average sentiment of reference
        """
        RelativeShift.__init__(self, reference, comparison, sent_dict_ref,
                               sent_dict_comp, stop_lens, reference_value)

class EntropyShift(RelativeShift):
    """
    Shift object for calculating the relative shift in entropy of a comparison
    system from a reference system

    Parameters
    ----------
    reference, comparison: dict
        keys are types of a system and values are frequencies of those types
    base: float, optional
        base of the logarithm for calculating entropy
    stop_lens: iterable of 2-tuples, optional
        denotes intervals that should be excluded when calculating shift
        scores
    reference_value: float, optional
        the reference score from which to calculate the deviation. If None,
        defaults to the entropy of reference
    """
    def __init__(self, reference, comparison, base=2, stop_lens=None):
        # Get surprisal scores
        type2p_ref,type2p_comp,type2s_ref,type2s_comp = get_surprisal_scores(reference,
                                                                             comparison,
                                                                             base=2, alpha=1)
        # Set zero surprisal scores for types that do not appear
        types = set(type2s_ref.keys()).union(set(type2s_comp.keys()))
        for t in types:
            if t not in type2s_ref:
                type2s_ref[0] = 0
            elif t not in type2s_comp:
                type2s_comp[t] = 0
        # Initialize shift
        RelativeShift.__init__(self, reference, comparison, type2s_ref,
                               type2s_comp, stop_lens, reference_value=0)
        self.type2p_ref = type2p_ref
        self.type2p_comp = type2p_comp

    def get_shift_graph(self, top_n=50, normalize=True, text_size_inset=True,
                        cumulative_inset=True, show_plot=True, filename=None,
                        detailed=False, **kwargs):
        RelativeShift.get_shift_graph(self, top_n=top_n, normalize=normalize,
                                      text_size_inset=text_size_inset,
                                      cumulative_inset=cumulative_inset,
                                      show_plot=show_plot, filename=filename,
                                      detailed=detailed, **kwargs)

class KLDivergenceShift(RelativeShift):
    """
    Shift object for calculating the KL Divergence between two systems

    Parameters
    ----------
    reference, comparison: dict
        keys are types of a system and values are frequencies of those types.
        For KL divergence to be well defined, all types must have nonzero
        frequencies in both reference and comparison
    base: float, optional
        base of the logarithm for calculating entropy
    stop_lens: iterable of 2-tuples, optional
        denotes intervals that should be excluded when calculating shift
        scores
    """
    def __init__(self, reference, comparison, base=2, stop_lens=None):
        # Check that KLD is well defined
        reference_types = set(reference.keys())
        comparison_types = set(comparison.keys())
        if len(reference_types.symmetric_difference(comparison_types)) > 0:
            warning = 'There are types that appear in either the reference or'\
                      +'comparison but not the other: KL divergence is not'\
                      +'well defined'
            warnings.warn(warning, Warning)
            return
        # Get surprisal scores
        type2p_ref,type2p_comp,type2s_ref,type2s_comp = get_surprisal_scores(reference,
                                                                             comparison,
                                                                             base=2, alpha=1)
        # Initialize shift
        RelativeShift.__init__(self, comparison, comparison, type2s_ref,
                               type2s_comp, stop_lens, reference_value=0)
        self.type2p_ref = type2p_ref
        self.type2p_comp = type2p_comp

    def get_shift_graph(self, top_n=50, normalize=True, text_size_inset=True,
                        cumulative_inset=True, show_plot=True, filename=None,
                        detailed=False, **kwargs):
        RelativeShift.get_shift_graph(self, top_n=top_n, normalize=normalize,
                                      text_size_inset=text_size_inset,
                                      cumulative_inset=cumulative_inset,
                                      show_plot=show_plot, filename=filename,
                                      detailed=detailed, **kwargs)
