"""
relative_shift.py

Author: Ryan J. Gallagher, Network Science Institute, Northeastern University
Last updated: June 13th, 2018
"""
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

import shifterator

# ------------------------------------------------------------------------------
# --------------------------- RELATIVE SHIFT CLASSES ---------------------------
# ------------------------------------------------------------------------------
class relative_shift(shifterator.shift):
    def __init__(self, reference, comparison, filenames=False,
                 type2score_ref=None, type2score_comp=None, stop_lens=None,
                 delimiter=','):
        """
        Shift object for calculating the relative shift of a comparison system
        from a reference system

        Parameters
        ----------
        reference, comparison: dict or str
            if dict, then keys are types of a system and values are frequencies
            of those types. if str and filenames=False, then the types are
            assumed to be tokens separated by white space. If str and
            filenames=True, then types are assumed to be tokens and text is read
            in line by line from the designated file and split on white space
        filenames: bool, optional
            True if reference and comparison are filenames of files with text to
            parse
        type2score_ref, type2score_comp: dict or str, optional
            if dict, types are keys and values are "scores" associated with each
            type (e.g., sentiment). If str, either the name of a score dict or
            file path to a score dict, where types and scores are given on each
            line, separated by commas. If None and other type2score is None,
            defaults to uniform scores across types. Otherwise defaults to the
            other type2score dict
        stop_lens: iterable of 2-tuples, optional
            denotes intervals that should be excluded when calculating shift
            scores
        """
        shift.__init__(reference, comparison, filenames=filenames,
                       type2score_1=type2score_ref,type2score_2=type2score_comp,
                       stop_lens=stop_lens, delimiter=delimiter)

class sentiment_shift(relative_shift):
    def __init__(self, reference_text, comparison_text, filenames=False,
                 sent_dict_ref='labMT_english', sent_dict_comp=None,
                 stop_lens=[(4,6)], delimiter=','):
        """
        Shift object for calculating the relative shift of a comparison system
        from a reference system

        Parameters
        ----------
        reference_text, comparison_text: dict or str
            if dict, then keys are word types of a system and values are
            frequencies of those types. if str and filenames=False, then the
            types are assumed to be tokens separated by white space. If str and
            filenames=True, then types are assumed to be tokens and text is read
            in line by line from the designated file and split on white space
        filenames: bool, optional
            True if reference_text and comparison_text are filenames of files
            with text to parse
        type2score_ref, type2score_comp: dict or str, optional
            if dict, word types are keys and values are sentiment scores
            associated with each type. If str, either the name of a sentiment
            dict or file path to a score dict, where types and scores are
            given on each line, separated by commas. If None and other
            type2score is None, defaults to uniform sentment across types, i.e.
            shift is in terms of just frequency, not sentiment.
            Otherwise defaults to the other type2score dict
        stop_lens: iterable of 2-tuples, optional
            denotes intervals that should be excluded when calculating shift
            scores
        """
        relative_shift.__init__(reference_text, comparison_text,
                                filenames=filenames, stop_lens=stop_lens,
                                delimiter=delimiter, type2score_ref=sent_dict_ref,
                                type2score_comp=sent_dict_comp)
