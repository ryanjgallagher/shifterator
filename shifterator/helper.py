"""
helper.py

Author: Ryan J. Gallagher, Network Science Institute, Northeastern University
"""
import os
import pkgutil
import collections
import numpy as np
from math import log

# ------------------------------------------------------------------------------
# -------------------------- Relative Frequency Funcs --------------------------
# ------------------------------------------------------------------------------

def get_relative_freqs(type2freq):
    """
    Calculates the relative frequency (proportion) of each type in a system

    Parameters
    ----------
    type2freq: dict
        keys are types of a system and values are frequencies of those types
    """
    n = sum(type2freq.values())
    type2p = {t:s/n for t,s in type2freq.items()}
    return type2p

def get_mixed_distribution(type2p, type2q, p=0.5, q=0.5):
    """
    Calculates the additive mixed distribution of two other distributions

    Parameters
    ----------
    type2, type2q: dict
        keys are types of a system and values are relative freqs of those types
    p, q: float
        relative weights of each distribution in the mixed distribution. Should
        sum to 1.
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

# ------------------------------------------------------------------------------
# -------------------------------- Score Funcs ---------------------------------
# ------------------------------------------------------------------------------
def filter_by_scores(type2freq, type2score, stop_lens):
    """
    Loads a dictionary of type scores

    Parameters
    ----------
    type2freq: dict
        keys are types, values are frequencies of those types
    type2score: dict
        keys are types, values are scores associated with those types
    stop_lens: iteratble of 2-tuples
        denotes intervals that should be excluded when calculating shift scores

    Returns
    -------
    type2freq_new,type2score_new: dict,dict
        Frequency and score dicts filtered of words whose score fall within stop
        window
    """
    type2freq_new = dict()
    type2score_new = dict()
    stop_words = set()
    for lower_stop,upper_stop in stop_lens:
        for t in type2score:
            if ((type2score[t] < lower_stop) or (type2score[t] > upper_stop))\
            and t not in stop_words:
                try:
                    type2freq_new[t] = type2freq[t]
                except KeyError:
                    pass
                type2score_new[t] = type2score[t]
            else:
                stop_words.add(t)

    return (type2freq_new, type2score_new, stop_words)

def get_score_dictionary(scores, encoding='utf-8'):
    """
    Loads a dictionary of type scores

    Parameters
    ----------
    scores: dict or str
        if dict, then returns the dict automatically. If str, then it is either
        the name of a lexicon included in Shifterator

    Returns
    -------
    type2score, dict
        dictionary where keys are types and values are scores of those types
    """
    if isinstance(scores, collections.Mapping):
        return scores.copy()

    # Else, load scores from predefined score file in shifterator
    try:
        lexicon = scores.split('_')[0]
        score_f = 'lexicons/{}/{}.tsv'.format(lexicon, scores)
        all_scores = pkgutil.get_data(__name__, score_f).decode(encoding)
    except FileNotFoundError:
        raise FileNotFoundError('Lexicon does not exit in Shifterator: {}'.format(scores))
    # Parse scores from all_scores, which is just a long str
    # Score files are line delimited with two tab-spaced columns: type and score
    type_scores = all_scores.split('\n')
    type2score = dict()
    for t_s in type_scores:
        if len(t_s) == 0:
            continue
        t,s = t_s.split('\t')
        type2score[t] = float(s)

    return type2score

def get_missing_scores(type2score_1, type2score_2):
    """
    Get missing scores between systems by setting the score in one system with
    the score in the other system

    Parameters
    ----------
    type2score_1, type2score_2: dict
        keys are types and values are scores

    Output
    ------
    type2score_1, type2score_2: dict
        keys are types and values are scores, updated to have scores across all
        types between the two score dictionaries
    """
    missing_types = set()
    types = set(type2score_1.keys()).union(set(type2score_2.keys()))
    for t in types:
        if t not in type2score_1:
            type2score_1[t] = type2score_2[t]
            missing_types.add(t)
        elif t not in type2score_2:
            type2score_2[t] = type2score_1[t]
            missing_types.add(t)
    return (type2score_1, type2score_2, missing_types)

# ------------------------------------------------------------------------------
# -------------------------------- Entropy Funcs -------------------------------
# ------------------------------------------------------------------------------
def get_type_surprisals(type2p, base=2, alpha=1):
    """
    Calculates the surprisal of each type in a system, i.e. log(1/p_i). Does not
    catch types with 0 relative frequency, assumed to be handled upstream

    Parameters
    ----------
    type2p: dict
        keys are types of a system and values are relative freqs of those types
    base: int
        the base of the logarithm
    alpha: float
        currently not implemented, but left for later updates
    """
    type2surprise = {t : -1*log(p, base) for t,p in type2p.items()}
    return type2surprise

def get_type_logs(type2p, base=2, alpha=1, force_zero=False):
    """
    Calculates the logarithm of each type in a system, i.e. log(p_i)

    Parameters
    ----------
    type2p: dict
        keys are types of a system and values are relative freqs of those types
    base: int
        the base of the logarithm
    alpha: float
        currently not implemented, but left for later updates
    force_zero: boolean
        if True, force any type with 0 probability to have log(p_i) = log(0) = 0
        This is mathematically invalid, but a useful workaround for calculating
        the JSD, where even though we calculate log(p) here individually, it is
        recombined with other scores to get p * log(p), which should be 0 if
        both p and log(p) are zero.
    """
    type2log = dict()
    for t,p in type2p.items():
        try:
            type2log[t] = log(p, base)
        except ValueError:
            if force_zero:
                type2log[t] = 0
            else:
                raise
    return type2log

def get_surprisal_scores(system_1, system_2, base=2, alpha=1):
    """
    Gets all surprisal scores necessary for calculating probability divergence
    measures like the KLD or JSD

    Parameters
    ----------
    system_1, system_2: dict
        keys are types of a system and values are frequencies of those types
    base: int
        the base for the logarithm when computing entropy for the JSD
    alpha: float
        currently not implemented, but left for later updates
    """
    # Normalize reference and comparison frequencies
    type2p_1 = get_relative_freqs(system_1)
    type2p_2 = get_relative_freqs(system_2)
    # Get surprisal of each type
    type2surprisal_1 = get_type_surprisals(type2p_1, base=base, alpha=alpha)
    type2surprisal_2 = get_type_surprisals(type2p_2, base=base, alpha=alpha)
    return type2p_1, type2p_2, type2surprisal_1, type2surprisal_2

def get_jsd_scores(type2freq_1, type2freq_2, base=2, alpha=1, weight_1=0.5,
                   weight_2=0.5):
    """
    Calculates the contribution of the types in two systems to the Jensen-Shannon
    divergence (JSD) between those systems

    Parameters
    ----------
    type2freq_1, type2freq_2: dict
        keys are types of a system and values are frequencies of those types
    base: int
        the base for the logarithm when computing entropy for the JSD
    weight_1, weight_2: float
        relative weights of system_1 and system_2 when constructing their mixed
        distribution. Should sum to 1
    alpha: float
        currently not implemented, but left for later updates
    """
    # Normalize reference and comparison frequencies
    type2p = get_relative_freqs(type2freq_1)
    type2q = get_relative_freqs(type2freq_2)
    # Get mixed distribution
    type2m = get_mixed_distribution(type2p, type2q, p=weight_1, q=weight_2)
    # Get surprisal of each type
    # Forcing zero should be OK, by formula anything that has a 0 should be 0
    #   in the end when multiplied against its 0 frequency, i.e. 0 * log 0 = 0
    type2log_p = get_type_logs(type2p, base=base, alpha=alpha, force_zero=True)
    type2log_q = get_type_logs(type2q, base=base, alpha=alpha, force_zero=True)
    type2log_m = get_type_logs(type2m, base=base, alpha=alpha)
    # Get scores (handle missing types)
    type2score_1 = {t : 0.5*(type2log_m[t] - type2log_p[t])
                    if t in type2log_p else 0 for t in type2log_m}
    type2score_2 = {t : 0.5*(type2log_q[t] - type2log_m[t])
                    if t in type2log_q else 0 for t in type2log_m}
    return type2p,type2q,type2m,type2score_1,type2score_2
