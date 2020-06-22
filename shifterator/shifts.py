"""
shifts.py
"""
import warnings
import numpy as np

import shifterator.shifterator as shifterator
from .helper import *

class WeightedAvgShift(shifterator.Shift):
    """
    Shift object for calculating differences in weighted averages between two
    systems
    """
    def __init__(self, type2freq_1, type2freq_2, type2score_1=None,
                 type2score_2=None,reference_value=None, stop_lens=None,
                 normalization='variation', encoding='utf-8'):
        shifterator.Shift.__init__(self,
                                   type2freq_1=type2freq_1,
                                   type2freq_2=type2freq_2,
                                   type2score_1=type2score_1,
                                   type2score_2=type2score_2,
                                   reference_value=reference_value,
                                   stop_lens=None,
                                   normalization=normalization,
                                   encoding=encoding)

class ProportionShift(shifterator.Shift):
    """
    Shift object for calculating differences in proportions of types across two
    systems

    Parameters
    __________
    type2freq_1, type2freq_2: dict
        keys are types of a system and values are frequencies of those types
    stop_lens: list
        currently not implemented, but left for later updates
    """
    def __init__(self, type2freq_1, type2freq_2, stop_lens=None,
                 reference_value=0, normalization='variation'):
        # Set relative frequency to 0 for types that don't appear
        type2freq_1 = type2freq_1.copy()
        type2freq_2 = type2freq_2.copy()
        types = set(type2freq_1.keys()).union(type2freq_2.keys())
        for t in types:
            if t not in type2freq_1:
                type2freq_1[t] = 0
            elif t not in type2freq_2:
                type2freq_2[t] = 0
        # Initialize shift object
        shifterator.Shift.__init__(self,
                                   type2freq_1=type2freq_1,
                                   type2freq_2=type2freq_2,
                                   type2score_1=None,
                                   type2score_2=None,
                                   reference_value=reference_value,
                                   stop_lens=stop_lens,
                                   normalization=normalization)

    def get_shift_graph(self, top_n=50, show_plot=True, detailed=False,
                        text_size_inset=True, cumulative_inset=True,
                        filename=None, **kwargs):
        shifterator.Shift.get_shift_graph(self,
                                          top_n=top_n,
                                          text_size_inset=text_size_inset,
                                          cumulative_inset=cumulative_inset,
                                          detailed=detailed,
                                          show_plot=show_plot,
                                          filename=filename,
                                          show_total=False,
                                          **kwargs)

class EntropyShift(shifterator.Shift):
    """
    Shift object for calculating the shift in entropy between two systems

    Parameters
    ----------
    type2freq_1, type2freq_2: dict
        keys are types of a system and values are frequencies of those types
    base: float, optional
        base of the logarithm for calculating entropy
    stop_lens: iterable of 2-tuples, optional
        denotes intervals that should be excluded when calculating shift
        scores
    reference_value: float, optional
        the reference score from which to calculate the deviation. If None,
        defaults to the entropy according to type2freq_1
    """
    def __init__(self, type2freq_1, type2freq_2, base=2, stop_lens=None,
                 reference_value=None, normalization='variation'):
        # Get surprisal scores
        type2freq_1 = type2freq_1.copy()
        type2freq_2 = type2freq_2.copy()
        type2p_1,type2p_2,type2s_1,type2s_2 = get_surprisal_scores(type2freq_1,
                                                                   type2freq_2,
                                                                   base=2,
                                                                   alpha=1)
        # Set zero surprisal scores for types that do not appear
        types = set(type2s_1.keys()).union(set(type2s_2.keys()))
        for t in types:
            if t not in type2s_1:
                type2s_1[0] = 0
            elif t not in type2s_2:
                type2s_2[t] = 0
        # Initialize shift
        shifterator.Shift.__init__(self,
                                   type2freq_1=type2freq_1,
                                   type2freq_2=type2freq_2,
                                   type2score_1=type2s_1,
                                   type2score_2=type2s_2,
                                   stop_lens=stop_lens,
                                   reference_value=reference_value,
                                   normalization=normalization)
        self.type2p_1 = type2p_1
        self.type2p_2 = type2p_2

    def get_shift_graph(self, top_n=50, show_plot=True, detailed=False,
                        text_size_inset=True, cumulative_inset=True,
                        filename=None, **kwargs):
        shifterator.Shift.get_shift_graph(self,
                                          top_n=top_n,
                                          text_size_inset=text_size_inset,
                                          cumulative_inset=cumulative_inset,
                                          detailed=detailed,
                                          show_plot=show_plot,
                                          filename=filename,
                                          **kwargs)

class KLDivergenceShift(shifterator.Shift):
    """
    Shift object for calculating the Kullback-Leibler divergence (KLD) between
    two systems

    Parameters
    ----------
    type2freq_1, type2freq_2: dict
        keys are types of a system and values are frequencies of those types.
        The KLD will be computed with respect type2freq_1, i.e. D(T2 || T1).
        For the KLD to be well defined, all types must have nonzero frequencies
        in both type2freq_1 and type2_freq2
    base: float, optional
        base of the logarithm for calculating entropy
    stop_lens: iterable of 2-tuples, optional
        denotes intervals that should be excluded when calculating shift
        scores
    """
    def __init__(self, type2freq_1, type2freq_2, base=2, stop_lens=None,
                 reference_value=None, normalization='variation'):
        # Check that KLD is well defined
        types_1 = set(type2freq_1.keys())
        types_2 = set(type2freq_2.keys())
        if len(types_1.symmetric_difference(types_2)) > 0:
            err = 'There are types that appear in either type2freq_1 or '\
                  + 'type2freq_2 but not the other: KL divergence is not '\
                  + 'well defined'
            raise ValueError(err)
        # Get surprisal scores
        type2freq_1 = type2freq_1.copy()
        type2freq_2 = type2freq_2.copy()
        type2p_1,type2p_2,type2s_1,type2s_2 = get_surprisal_scores(type2freq_1,
                                                                   type2freq_2,
                                                                   base=2,
                                                                   alpha=1)
        # Initialize shift
        shifterator.Shift.__init__(self,
                                   type2freq_1=type2freq_2,
                                   type2freq_2=type2freq_2,
                                   type2score_1=type2s_1,
                                   type2score_2=type2s_2,
                                   stop_lens=stop_lens,
                                   reference_value=reference_value,
                                   normalization=normalization)
        self.type2p_1 = type2p_1
        self.type2p_2 = type2p_2

    def get_shift_graph(self, top_n=50, show_plot=True, detailed=False,
                        text_size_inset=True, cumulative_inset=True,
                        filename=None, **kwargs):
        shifterator.Shift.get_shift_graph(self,
                                          top_n=top_n,
                                          text_size_inset=text_size_inset,
                                          cumulative_inset=cumulative_inset,
                                          detailed=detailed,
                                          show_plot=show_plot,
                                          filename=filename,
                                          **kwargs)

class JSDivergenceShift(shifterator.Shift):
    """
    Shift object for calculating the Jensen-Shannon divergence (JSD) between two
    systems

    Parameters
    __________
    type2freq_1, type2freq_2: dict
        keys are types of a system and values are frequencies of those types
    base: int
        the base for the logarithm when computing entropy for the JSD
    weight_1, weight_2: float
        relative weights of type2freq_1 and type2frq_2 when constructing their
        mixed distribution. Should sum to 1
    reference_value: float, optional
        the reference score from which to calculate the deviation. If None,
        defaults to the entropy of the mixed distribution
    alpha: float
        currently not implemented, but left for later updates
    """
    def __init__(self, type2freq_1, type2freq_2, base=2, weight_1=0.5,
                 weight_2=0.5, alpha=1, stop_lens=None, reference_value=None,
                 normalization='variation'):
        # Check weights
        if weight_1 + weight_2 != 1:
            raise ValueError('weight_1 and weight_2 do not sum to 1')
        # Get JSD scores
        type2freq_1 = type2freq_1.copy()
        type2freq_2 = type2freq_2.copy()
        type2p,type2q,type2m,type2score_1,type2score_2 = get_jsd_scores(type2freq_1,
                                                                        type2freq_2,
                                                                        weight_1=weight_1,
                                                                        weight_2=weight_2,
                                                                        base=base,
                                                                        alpha=alpha)
        # Get entropy of mixture distribution
        if reference_value is None:
            type2score_m = get_type_surprisals(type2m, base=base, alpha=alpha)
            reference_value = sum([m * type2score_m[t] for t,m in type2m.items()])

        # Initialize shift object
        shifterator.Shift.__init__(self,
                                   type2freq_1=type2freq_1,
                                   type2freq_2=type2freq_2,
                                   type2score_1=type2score_1,
                                   type2score_2=type2score_2,
                                   reference_value=reference_value,
                                   normalization=normalization,
                                   stop_lens=stop_lens)
        self.type2p_1 = type2p
        self.type2p_2 = type2q
        self.type2p_mixed = type2m

    def get_shift_graph(self, top_n=50, show_plot=True, detailed=False,
                        text_size_inset=True, cumulative_inset=True,
                        filename=None, show_total=False, **kwargs):
        shifterator.Shift.get_shift_graph(self,
                                          top_n=top_n,
                                          text_size_inset=text_size_inset,
                                          cumulative_inset=cumulative_inset,
                                          detailed=detailed,
                                          show_plot=show_plot,
                                          filename=filename,
                                          show_total=show_total,
                                          all_pos_contributions=True,
                                          **kwargs)
