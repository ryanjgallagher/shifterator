"""
helper.py

Author: Ryan J. Gallagher, Network Science Institute, Northeastern University

TODO:
- Allow different order entropies to be specified using alpha
"""
import os
import numpy as np
from math import log

# ------------------------------------------------------------------------------
# -------------------------- Relative Frequency Funcs --------------------------
# ------------------------------------------------------------------------------

def get_relative_freqs(type2freq):
    """

    """
    n = sum(type2freq.values())
    type2p = {t:s/n for t,s in type2freq.items()}
    return type2p

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

def get_score_dictionary(scores):
    """
    Loads a dictionary of type scores

    Parameters
    ----------
    scores: dict or str
        if dict, then returns the dict automatically. If str, then it is either
        the name of a shifterator dictionary to load, or file path of dictionary
        to load. File should be two columns of types and scores on each line,
        separated by delimiter
            Options: 'labMT_english'
    stop_lens: iteratble of 2-tuples
        denotes intervals that should be excluded when calculating shift scores

    Returns
    -------
    type2score, dict
        dictionary where keys are types and values are scores of those types
    """
    if type(scores) is dict:
        return scores.copy()
    # Check if dictionary name is in shifterator data
    score_dicts = os.listdir('data')
    if scores in score_dicts:
        dict_file = 'data/'+scores
    elif  scores+'.csv' in score_dicts:
        dict_file = 'data/'+scores+'.csv'
    else: # Assume file path
        dict_file = scores
    # Load score dictionary
    type2score = {}
    with open(dict_file, 'r') as f:
        for line in f:
            t,score = line.strip().split('\t')
            type2score[t] = score

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

    TODO: set order of entropy using alpha
    """
    type2surprise = {t : -1*log(p, base) for t,p in type2p.items()}
    return type2surprise

def get_type_logs(type2p, base=2, alpha=1, force_zero=False):
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
    # Normalize reference and comparison frequencies
    type2p_1 = get_relative_freqs(system_1)
    type2p_2 = get_relative_freqs(system_2)
    # Get surprisal of each type
    type2surprisal_1 = get_type_surprisals(type2p_1, base=base, alpha=alpha)
    type2surprisal_2 = get_type_surprisals(type2p_2, base=base, alpha=alpha)
    return type2p_1, type2p_2, type2surprisal_1, type2surprisal_2

def get_jsd_scores(type2freq_1, type2freq_2, base=2, alpha=1, weight_1=0.5,
                   weight_2=0.5):
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
