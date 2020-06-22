"""
shifts.py
"""
import warnings
import numpy as np

import shifterator.shifterator as shifterator
from .entropy import *

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
                        title=None, filename=None, **kwargs):
        if title is None:
            title = ''
        shifterator.Shift.get_shift_graph(self,
                                          top_n=top_n,
                                          text_size_inset=text_size_inset,
                                          cumulative_inset=cumulative_inset,
                                          detailed=detailed,
                                          show_plot=show_plot,
                                          filename=filename,
                                          show_total=False,
                                          title=title,
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
        the reference score from which to calculate the deviation
    """
    def __init__(self, type2freq_1, type2freq_2, base=2, alpha=1, stop_lens=None,
                 reference_value=0, normalization='variation'):
        # Get relative frequencies
        type2freq_1 = type2freq_1.copy()
        type2freq_2 = type2freq_2.copy()
        type2p_1 = get_relative_freqs(type2freq_1)
        type2p_2 = get_relative_freqs(type2freq_2)
        # Get entropy scores
        type2s_1,type2s_2 = get_entropy_scores(type2p_1, type2p_2, base, alpha)

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
        self.alpha = alpha

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
                 reference_value=0, normalization='variation'):
        # Check that KLD is well defined
        types_1 = set(type2freq_1.keys())
        types_2 = set(type2freq_2.keys())
        if len(types_1.symmetric_difference(types_2)) > 0:
            err = 'There are types that appear in either type2freq_1 or '\
                  + 'type2freq_2 but not the other: the KL divergence is not '\
                  + 'well defined'
            raise ValueError(err)

        # Get relative frequencies
        type2freq_1 = type2freq_1.copy()
        type2freq_2 = type2freq_2.copy()
        type2p_1 = get_relative_freqs(type2freq_1)
        type2p_2 = get_relative_freqs(type2freq_2)
        # Get surprisal scores
        type2s_1 = {t:p * -1 * log(p, base) for t,p in type2p_1.items()}
        type2s_2 = {t:p * -1 * log(p, base) for t,p in type2p_2.items()}

        # Initialize shift
        shifterator.Shift.__init__(self,
                                   type2freq_1=type2p_2,
                                   type2freq_2=type2p_2,
                                   type2score_1=type2s_1,
                                   type2score_2=type2s_2,
                                   stop_lens=stop_lens,
                                   reference_value=reference_value,
                                   normalization=normalization)
        self.type2p_1 = type2p_1
        self.type2p_2 = type2p_2

    def get_shift_graph(self, top_n=50, show_plot=True, detailed=False,
                        text_size_inset=True, cumulative_inset=True,
                        title=None, filename=None, **kwargs):
        if title is None:
            title = ''
        shifterator.Shift.get_shift_graph(self,
                                          top_n=top_n,
                                          text_size_inset=text_size_inset,
                                          cumulative_inset=cumulative_inset,
                                          detailed=detailed,
                                          show_plot=show_plot,
                                          title=title,
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
    weight_1, weight_2: float
        relative weights of type2freq_1 and type2frq_2 when constructing their
        mixed distribution. Should sum to 1
    base: int
        the base for the logarithm when computing entropy for the JSD
    alpha: float
        currently not implemented, but left for later updates
    reference_value: float, optional
        the reference score from which to calculate the score deviation
    """
    def __init__(self, type2freq_1, type2freq_2, base=2, weight_1=0.5,
                 weight_2=0.5, alpha=1, stop_lens=None, reference_value=0,
                 normalization='variation'):
        # Check weights
        if weight_1 + weight_2 != 1:
            raise ValueError('weight_1 and weight_2 do not sum to 1')

        # Get relative frequencies
        type2freq_1 = type2freq_1.copy()
        type2freq_2 = type2freq_2.copy()
        type2p_1 = get_relative_freqs(type2freq_1)
        type2p_2 = get_relative_freqs(type2freq_2)
        # Get shift scores
        type2m,type2s_1,type2s_2 = get_jsd_scores(type2p_1,
                                                  type2p_2,
                                                  weight_1=weight_1,
                                                  weight_2=weight_2,
                                                  base=base,
                                                  alpha=alpha)

        # Initialize shift object
        shifterator.Shift.__init__(self,
                                   type2freq_1=type2freq_1,
                                   type2freq_2=type2freq_2,
                                   type2score_1=type2s_1,
                                   type2score_2=type2s_2,
                                   reference_value=reference_value,
                                   normalization=normalization,
                                   stop_lens=stop_lens)
        self.type2p_1 = type2p_1
        self.type2p_2 = type2p_2
        self.type2m = type2m
        self.alpha = alpha

    def get_shift_graph(self, top_n=50, show_plot=True, detailed=False,
                        text_size_inset=True, cumulative_inset=True,
                        title=None, filename=None, **kwargs):
        if self.alpha == 1 and self.reference_value == 0:
            all_pos_contributions = True
        else:
            all_pos_contributions = False
        if title is None:
            title = ''
        shifterator.Shift.get_shift_graph(self,
                                          top_n=top_n,
                                          text_size_inset=text_size_inset,
                                          cumulative_inset=cumulative_inset,
                                          detailed=detailed,
                                          show_plot=show_plot,
                                          filename=filename,
                                          title=title,
                                          all_pos_contributions=all_pos_contributions,
                                          **kwargs)
