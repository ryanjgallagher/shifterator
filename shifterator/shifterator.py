"""
wordshift.py

Author: Ryan J. Gallagher, Network Science Institute, Northeastern University
Last updated: June 12th, 2018
"""
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class shift:
    def __init__(self, sys_1, sys_2, ref, filenames=False, type_dict_1=None,
                 type_dict_2=None, stop_radius=0.0, middle_score=None,
                 delimiter=','):
    """
    Shift object for calculating weighted scores of two systems of types, and
    the relative shift between them

    sys1: str or dict, if str and filenames=False, then the input is assumed
          to be text and it is read in directliy and split on white space. If
          str and filenames=True, then text is read in line by line from the
          designated file and split on white space. If dict, then should
          be of the form where keys are types and values are frequencies of
          those types
    sys2: str or dict, of the same type as sys1.
    ref: str or dict, of the same type as sys1 and sys2. Shift scores will be in
          terms of how the systems 1 and 2 deviate from the reference system.
    filenames: bool, True if sys_1 and sys_2 are filenames of files with text to
                parse
    type_dict_1: str or dict, If dict, types are keys and values are "scores"
                 associated with each type (e.g., sentiment). If str, either the
                 name of a score dict or file path to a score dict, where types
                 and scores are given on each line, separated by commas. If
                 None and type_dict_2 is None, defaults to uniform scores across
                 types. Otherwise defaults to type_dict_2 scores. Similarly, for
                 type_dict_2
    type_dict_ref: str or dict, If dict, types are keys and values are "scores"
                   associated with each type (e.g., sentiment). If str, either
                   the name of a score dict or file path to a score dict, where
                   types and scores are given on each line, separated by commas.
                   If None, defaults to uniform scores across
    stop_radius: float, types that have scores within stop_radius of the middle
                 score will be excluded
                 Stop window = middle_score +- stop_radius
    middle_score: float, middle, "neutral" (not average) score denoting the
                  center of the stop window
    delimiter: str, delimiter separating types from scores if loading from file
    """
    # Load type2freq dictionaries
    if isinstance(ref, dict) and isinstance(comp, dict):
        self.type2freq_1 = sys_1
        self.type2freq_2 = sys_2
    elif isinstance(ref, basestring) and isinstance(comp, basestring):
        if filenames is True:
            self.type2freq_1 = get_freqs_from_file(sys_1)
            self.type2freq_2 = get_freqs_from_file(sys_2)
        elif filename is False:
            self.type2freq_1 = dict(Counter(sys_1.split()))
            self.type2freq_2 = dict(Counter(sys_2.split()))
    else:
        warning = 'Shift object was not given text, a file to parse, or '+\
                  'frequency dictionaries. Check input.'
        warnings.warn(warning, Warning)
        self.type2freq_1 = dict()
        self.type2freq_2 = dict()
    # Set observed types
    # TODO: maybe get rid of these and just calculate in funcs
    self.types = (set(self.type2freq_1.keys()).union(set(self.typefreq_2.keys())))
    self.types_1 = set(self.type2freq_1.keys())
    self.types_2 = set(self.type2freq_2.keys())
    # Load type2score dictionaries
    self.stop_radius = stop_radius
    self.middle_score = middle_score
    if type_dict_1 is not None and type_dict_2 is not None:
        self.type2score_1 = get_score_dictionary(type_dict_1, stop_radius,
                                                 middle_score, delimiter)
        self.type2score_2 = get_score_dictionary(type_dict_2, stop_radius,
                                                 middle_score, delimiter)
    elif type_dict_1 is not None:
        self.type2score_1 = get_score_dictionary(type_dict_1, stop_radius,
                                                 middle_score, delimiter)
        self.type2score_2 = self.word2score_1
    elif type_dict_2 is not None:
        self.type2score_2 = get_score_dictionary(type_dict_2, stop_radius,
                                                 middle_score, delimiter)
        self.type2score_1 = self.word2score_2
    else:
        self.type2score_1 = {t : 1 for t in self.types_1}
        self.type2score_2 = {t : 1 for t in self.types_2}
    # Update types according to score types: (sys1 \cup sys2) \cap score_words
    # TODO: maybe get rid of these and just calculate in funcs
    self.types_1 = self.types_1.intersection(set(self.type2score_1))
    self.types_2 = self.types_2.intersection(set(self.type2score_2))
    self.types = self.types_1.union(self.types_2)
    if len(vocab) == 0:
        warning = 'No words in input texts are in score dictionary'
        warnings.warn(warning, Warning)

    # TODO: add functions that allow you to easily update the type2freq dicts

    def get_weighted_score(self, type2freq, type2score):
        """
        Calculate the average score of the system specified by the frequencies
        and scores of the types in that system

        INPUT
        -----
        type2freq: dict, keys are types and values are frequencies
        type2score: dict, keys are types and values are scores

        OUTPUT
        ------
        s_avg: float, avg. weighted score of sys according to freqs and scores
        """
        # Check we have a vocabulary to work with
        types = set(type2freq.keys()).intersection(set(type2score.keys()))
        if len(types) == 0:
            warning = 'No types in the frequency dict appear in the score dict'
            warnings.warn(warning, Warning)
            return
        # Get weighted score and total frequency
        f_total = sum([freq for t,freq in type2freq.items() if t in types])
        s_weighted = sum([type2score[t]*freq for t,freq in type2freq.items()
                          if t in types])
        s_avg = s_weighted / f_total
        return s_avg

    def get_shift_scores(self, type2freq_1=None, type2score_1=None,
                         type2freq_2=None, type2score_2=None, type2freq_ref=None,
                         type2score_ref=None, normalize=True, details=False):
        """
        Calculates the type shift scores between two systems

        INPUT
        -----
        type2freq: dict, keys are types and values are frequencies. If None,
                   defaults to the sys_1 and sys_2 type freqs respectively
        type2score: dict, keys are types and values are scores. If None,
                    defaults to the sys_1 and sys_2 type scores respectively
        normalize: bool, if True normalizes shift scores so they sum to 1
        details: bool, if True returns each component of the shift score and
                 the final normalized shift score. If false, only returns the
                 normalized shift scores

        OUTPUT
        ------
        type2p_diff: dict, if details is True, returns dict where words are keys
                     and values are the difference in relatively frequency
                     between the comparison text and frequency text
        type2s_diff: dict, if details is True, returns dict where words are keys
                     and values are the relative differences of each word's
                     sentiment from the reference text's average sentiment
        type2shift_Score: dict, words are keys and values are shift scores,
                          p_diff*s_diff, normalized to be between 0 and 1
        """
        # Check input
        if type2freq_1 is None or type2score_1 is None:
            type2freq_1 = self.type2freq_1
            type2score_1 = self.type2score_1
        if type2freq_2 is None or type2score_2 is None:
            type2freq_2 = self.type2freq_2
            type2score_2 = self.type2score_2
        if type2freq_ref is None or type2score_ref is None:
            type2freq_ref = self.type2freq_ref
            type2freq_score = self.type2freq_self
        # Get observed types that are also in score dicts
        types_1 = set(type2freq_1.keys()).intersection(set(type2score_1.keys()))
        types_2 = set(type2freq_2.keys()).intersection(set(type2score_2.keys()))
        types = types_1.union(types_2)
        # Get total frequencies, and average score of reference
        total_freq_1 = sum([freq for t,freq in type2freq_1.items() if t in types])
        total_freq_2 = sum([freq for t,freq in type2freq_2.items() if t in types])
        s_avg_ref = self.get_weighted_score(type2freq_ref, type2score_ref)
        # Get relative frequency of words in reference and comparison
        type2p_1 = {t:type2freq_1[t]/total_freq_1 if t in type2freq_1 else 0
                    for t in types}
        type2p_2 = {t:type2freq_2[t]/total_freq_2 if t in type2freq_2 else 0
                    for t in types}
        # Calculate relative diffs in freq
        type2p_diff = {t:type2p_2[t]-type2p_1[t] for t in types}
        # Calculate relative diffs in score and relative diff from ref, where defined
        # TODO: this needs to be handled more elegantly
        type2s_diff = {}
        type2s_ref_diff = {}
        for t in types:
            if t in types_1 and types_2:
                type2s_diff[t] = type2score_2[t]-type2score_1[t]
                type2s_ref_diff[t] = 0.5*(type2score_2[t]+type2score_1[t])-s_avg_ref
            else:
                type2s_diff[t] = None
                type2s_ref_diff[t] = None
        # Calculate total shift scores
        type2shift_score = {t : type2p_diff[t]*type2s_ref_diff[t]+0.5*type2s_diff[t]\
                            *(type2p_2[t]+type2p_1[t]) for t in types
                            if t in types_1 and types_2}
        # Normalize the total shift scores
        if normalize:
            total_diff = abs(sum(type2shift_score.values()))
            type2shift_score = {t : shift_score/total_diff for t,shift_score
                                in type2shift_score.items()}
        # Set results in sentiment shift object
        self.type2p_diff = type2p_diff
        self.type2s_diff = type2s_diff
        self.type2s_ref_diff = type2s_ref_diff
        self.type2shift_score = type2shift_score
        # Return shift scores
        if details:
            return type2p_diff,type2s_diff,type2s_ref_diff,type2shift_score
        else:
            return type2shift_score

    def get_shift_graph(self, top_n=50, bar_colors=('#ffff80','#3377ff'),
                        bar_type_space=0.5, width_scaling=1.4, xlabel=None,
                        ylabel=None, title=None, xlabel_fontsize=18,
                        ylabel_fontsize=18, title_fontsize=14, detailed=False,
                        show_plot=True, tight=True):
        """
        Plot the shift graph between two systems of types

        INPUT
        -----
        top_n: int, display the top_n types as sorted by their absolute
               contribution to the difference between systems
        bar_colors: tuple, colors to use for bars where first and second entries
                    are the colors for types that have positive and negative
                    relative score differences relatively
        bar_type_space: float, space between the end of each bar and the
                        corresponding label
        width_scaling: float, parameter controls the width of the x-axis. If
                       types overlap with the y-axis then increase the scaling
        detailed: bool, whether to return detailed (advanced) shift graph
        show_plot: bool, whether to show plot on finish
        tight: bool, whether to call plt.tight_layout() on the plot
        """
        if not detailed:
            return self.get_shift_graph_simple(top_n, bar_colors, bar_type_space,
                                               width_scaling, xlabel, ylabel,
                                               title, xlabel_fontsize,
                                               ylabel_fontsize, title_fontsize,
                                               show_plot, tight)
        else:
            return self.get_shift_graph_detailed(top_n, bar_colors,
                                                 bar_type_space, width_scaling,
                                                 xlabel, ylabel, title,
                                                 xlabel_fontsize, ylabel_fontsize,
                                                 title_fontsize, show_plot, tight)

    def get_shift_graph_simple(self, top_n=50, bar_colors=('#ffff80','#3377ff'),
                               bar_type_space=0.5, width_scaling=1.4,
                               xlabel=None, ylabel=None, title=None,
                               xlabel_fontsize=18, ylabel_fontsize=18,
                               title_fontsize=14, show_plot=True, tight=True):
        """
        Plot the simple shift graph between two systems of types

        INPUT
        -----
        top_n: int, display the top_n types as sorted by their absolute
               contribution to the difference between systems
        bar_colors: tuple, colors to use for bars where first and second entries
                    are the colors for types that have positive and negative
                    relative score differences relatively
        bar_type_space: float, space between the end of each bar and the
                        corresponding label
        width_scaling: float, parameter controls the width of the x-axis. If
                       types overlap with the y-axis then increase the scaling
        show_plot: bool, whether to show plot on finish
        tight: bool, whether to call plt.tight_layout() on the plot
        """
        if self.type2shift_score is None:
            self.get_shift_scores(details=False)
        # Sort type scores and take top_n. Reverse for plotting
        type_scores = [(t, self.type2s_diff[t], self.type2p_diff[t],
                        self.type2shift_score[t]) for t in self.types]
        # reverse?
        type_scores = sorted(type_scores, key=labmda x:abs(x[3]))[:top_n]
        type_diffs = [100*score for (t,s_diff,p_diff,score) in type_scores]
        # Get bar colors
        bar_colors = [bar_colors[0] if s_diff>0 else bar_colors[1]\
                      for (word,s_diff,p_diff,score) in word_scores]
        # Plot scores, height:width ratio = 2.5:1
        f,ax = plt.subplots(figsize=(6,15))
        ax.margins(y=0.01)
        # Plot the skeleton of the word shift
        # edgecolor thing is a workaround for a bug in matplotlib
        bars = ax.barh(range(1,len(type_scores)+1), word_diffs, .8, linewidth=1
                       align='center', color=bar_colors, edgecolor=['black']*top_n)
        # Add center dividing line
        ax.plot([0,0],[1,top_n], '-', color='black', linewidth=0.7)
        # Make sure there's the same amount of space on either side of y-axis,
        # and add space for word labels using 'width_scaling'
        # TODO: can we automate selection of width_scaling?
        x_min,x_max = ax.get_xlim()
        x_sym = width_scaling*max([abs(x_min),abs(x_max)])
        ax.set_xlim((-1*x_sym, x_sym))
        # Flip y-axis tick labels and make sure every 5th tick is labeled
        y_ticks = list(range(1,top_n,5))+[top_n]
        y_tick_labels = [str(n) for n in (list(range(top_n,1,-5))+['1'])]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        # Format word labels with up/down arrows and +/-
        type_labels = _get_shift_type_labels(type_scores)
        # Add word labels to bars
        ax = _set_bar_labels(bars, type_labels, bar_type_space=bar_type_space)
        # Set axis labels and title
        if xlabel is None:
            xlabel = 'Per type average score shift $\delta s_{avg,r}$ (%)'
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
        if ylabel is None:
            ylabel = 'Type rank $r$'
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        if title is None:
            s_avg_1 = self.get_average_sentiment(self.type2freq_1,self.type2score_1)
            s_avg_2 = self.get_average_sentiment(self.type2freq_2,self.type2score_2)
            title = '$\Phi_{\Omega^{(2)}}$: $s_{avg}^{(ref)}=$'+'{0:.2f}'.format(s_avg_ref)+'\n'\
                    +'$\Phi_{\Omega^{(1)}}$: $s_{avg}^{(comp)}=$'+'{0:.2f}'.format(s_avg_comp)
        ax.set_title(title_str, fontsize=14)
        # Show and return plot
        if tight:
            plt.tight_layout()
        if show_plot:
            plt.show()
        return ax

    def get_shift_graph_detailed(self, top_n=50, bar_colors=('#ffff80','#3377ff'),
                                 bar_type_space=0.5, width_scaling=1.4,
                                 xlabel=None, ylabel=None, title=None,
                                 xlabel_fontsize=18, ylabel_fontsize=18,
                                 title_fontsize=14, show_plot=True, tight=True):
        """
        Plot the detailed shift graph between two systems of types

        INPUT
        -----
        top_n: int, display the top_n types as sorted by their absolute
               contribution to the difference between systems
        bar_colors: tuple, colors to use for bars where first and second entries
                    are the colors for types that have positive and negative
                    relative score differences relatively
        bar_type_space: float, space between the end of each bar and the
                        corresponding label
        width_scaling: float, parameter controls the width of the x-axis. If
                       types overlap with the y-axis then increase the scaling
        show_plot: bool, whether to show plot on finish
        tight: bool, whether to call plt.tight_layout() on the plot
        """
        # TODO: implement, can probably make a func that's shared between the
        # simple and detailed shift that creates the fundamental layout
        pass

# ------------------------------------------------------------------------------
__init__(self, sys_1, sys_2, ref, filenames=False, type_dict_1=None,
             type_dict_2=None, stop_radius=0.0, middle_score=None,
             delimiter=',')

class relative_shift(shift):
    def __init__(self, reference, comparison, filenames=False, ref_dict=None,
                 comp_dict=None, stop_radius=0.0, middle_score=None,
                 delimiter=','):
        shift.__init__(reference, comparison, reference, filenames=filenames,
                       type_dict_1=ref_dict, type_dict_2=comp_dict,
                       stop_radius=stop_radius, middle_score=middle_score,
                       delimiter=delimiter)

class sentiment_shift(relative_shift):
    def __init__(self, reference_text, comparison_text, filenames=False,
                 ref_sent_dict='labMT_english', comp_sent_dict=None,
                 stop_radius=1.0, middle_score=None, delimiter=','):
        relative_shift.__init__(reference_text, comparison_text, filenames=filenames,
                       ref_dict=ref_sent_dict, comp_dict=comp_sent_dict,
                       stop_radius=stop_radius, middle_score=middle_score,
                       delimiter=delimiter)

# ------------------------------------------------------------------------------
class symmetric_shift(shift):
    pass

class divergence_shift(shift):
    pass

# ------------------------------------------------------------------------------


class sentiment_shift(score_shift):
    def __init__(self, ref_text, comp_text, filenames=False,
                 dictionary_ref='labMT_english', dictionary_comp = None,
                 stop_radius=1.0, middle_score=5.0):
        """
        Word shift object for calculating weighted scores of texts based on a
        sentiment dictionary

        ref_text: str or dict, if str and filenames=False, then the text is read
                  in directliy and split on white space. If str and
                  filenames=True, then text is read in line by line from the
                  designated file and split on white space. If dict, then should
                  be of the form where keys are words and values are frequencies
                  of those words
        comp_text: str or dict, of the same type as reference_text. Word shift
                   scores will be in terms of how the comparison text differs
                   from the reference text
        filenames: bool, True if reference_text and comparison_text are
                   filenames of files with text to parse
        dictionary: str, name of dictionary to load, or file path of dictionary
                    to load. Options: 'labMT_english',
        stop_radius: float, words that have sentiment within stop_radius of the
                     middle sentiment score will be excluded.
                     Stop window = middle_score +- stop_radius
        middle_score: float, middle, neutral score of sentiment (not average)
                      denoting the center of the stop window
        """
        score_shift.__init__(self, ref_text, comp_text, filenames=False,
                             dictionary_ref=dictionary_ref, stop_radius=1.0
                             dictionary_comp=dictionary_comp, middle_score=5.0)
        # Set sentiment shift specific attributes for convenience
        self.word2sentiment = self.word2score
        # Initialize word shift score components to None
        self.word2p_diff = None
        self.word2s_diff = None
        self.word2s_rel_diff = None
        self.word2shift_score = None

    def get_weighted_sentiment(self, text='reference'):
        """
        Calculate the average sentiment of the comparison or reference text

        INPUT
        -----
        text: str, whether to calculate average for 'comparison' or 'reference'

        OUTPUT
        ------
        average_sentiment: float, average sentiment of comparison or reference
        """
        self.get_weighted_score(self, text=text)

    def get_word_shift_scores(self, normalize=True, details=False):
        """
        Calculates the sentiment word shift scores between a reference and
        comparison text

        INPUT
        -----
        details: bool, if True returns each component of the shift score and
                 the final normalized shift score. If false, only returns the
                 normalized word shift scores
        normalize: bool, if True normalizes word shift scores so they sum to 1

        OUTPUT
        ------
        word2p_diff: dict, if details is True, returns dict where words are keys
                     and values are the difference in relatively frequency
                     between the comparison text and frequency text
        word2s_diff: dict, if details is True, returns dict where words are keys
                     and values are the relative differences of each word's
                     sentiment from the reference text's average sentiment
        word2shift_Score: dict, words are keys and values are shift scores,
                          p_diff*s_diff, normalized to be between 0 and 1
        """
        # Get total frequencies, and average sentiment of reference
        total_freq_ref = sum([freq for word,freq in self.word2freq_ref.items()
                              if word in self.vocab])
        total_freq_comp = sum([freq for word,freq in self.word2freq_comp.items()
                               if word in self.vocab])
        average_sentiment_ref = get_weighted_score(self.word2freq_ref,
                                                   self.word2sentiment)
        # Get relative frequency of words in reference and comparison
        word2p_ref = {word:word2freq_ref[word]/total_freq_ref if word
                      in self.word2freq_ref else 0 for word in self.vocab}
        word2p_comp = {word:word2freq_comp[word]/total_freq_comp if word
                       in self.word2freq_comp else 0 for word in self.vocab}
        # Calculate relative diffs of freq and sentiment, and total shift scores
        word2p_diff = {word:word2p_comp[word]-word2p_ref[word] for word in self.vocab}
        word2s_diff = {word:word2sent[word]-average_sentiment_ref for word in self.vocab}
        word2shift_score = {word:word2s_diff[word]*word2p_diff[word] for word in self.vocab}
        # Normalize the total shift scores
        if normalize:
            total_diff = abs(sum(word2shift_score.values()))
            word2shift_score = {word:shift_score/total_diff for word,shift_score
                                in word2shift_score.items()}
        # Set results in sentiment shift object
        self.word2p_diff = word2p_diff
        self.word2s_diff = word2s_diff
        self.word2shift_score = word2shift_score
        # Return shift scores
        if details:
            return word2p_diff,word2s_diff,word2shift_score
        else:
            return word2shift_score

class divergence_shift(word_shift):
    def __init__(self, ref_text, comp_text, filenames=False, divergence='jsd',
                 alpha=1.5):
        """
        Word shift object for calculating the information-theoretic divergence
        between two texts.

        ref_text: str or dict, if str and filenames=False, then the text is read
                in directliy and split on white space. If str and filenames=True,
                then text is read in line by line from the designated file and
                split on white space. If dict, then should be of the form where
                keys are words and values are frequencies of those words. If
                divergence='jsd', ref_text and comp_text are interchangeable
        comp_text: str or dict, of the same type as ref_text. If divergence='jsd'
                   ref_text and comp_text are interchangeable
        filenames: bool, True if ref_text and comp_text are filenames of files
                   with text to parse
        divergence: str, type of divergence to calculate. Options: 'jsd','kld'
        alpha: float, (0,2], order of generalized divergence
        """
        word_shift.__init__(self, ref_text, comp_text, filenames)
        self.divergence = divergence

def get_score_dictionary(score_dict, stop_radius=0.0, middle_score=5.0,
                         delimiter=','):
    """
    Loads a dictionary of type scores.

    INPUT
    -----
    score_dict: dict or str, dict where keys are types and values are scores (in
                which case the dict is returned automatically), orname of
                dictionary to load, or file path of dictionary to load.
                Options: 'labMT_english',
    delimiter: str, delimiter used in the dictionary file

    OUTPUT:
    ------
    """
    if type(score_dict) is dict:
        return score_dict
    # Check if CSV of dictionary exists, otherwise use direct file path
    dicts = os.listdir('data')
    if  score_dict+'.csv' in dicts:
        dict_file = 'data/'+score_dict+'.csv'
    elif score_dict in dicts:
        dict_file = 'data/'+score_dict
    else:
        dict_file = score_dict
    # Load score dictionary
    typescore = {}
    with open(dictionary_file, 'r') as f:
        for line in f:
            t,score = line.strip().split(delimiter)
            type2score[word] = score
    # Filter dictionary of words outside of stop range
    if stop_radius > 0 or middle_score is None:
        lower_stop = middle_score - stop_radius
        upper_stop = middle_score + stop_radius
        type2score = {t:score for t,score in type2score.items()
                      if score <= lower_stop or score >= upper_stop}
        return type2score
    else:
        return type2score

def get_freqs_from_file(filename):
    """
    Parses text of a file line by line, splitting across white space

    INPUT
    -----
    filename: str, file to load text from

    OUTPUT
    ------
    type2freq: dict, keys are words and values are frequencies of those words
    """
    type2freq = Counter()
    with open(filename, 'r') as f:
        for line in f:
            types = line.strip().split()
            type2freq.update(types)
    return dict(type2freq)

def _get_shift_type_labels(type_scores):
    """

    """
    type_labels = []
    for (t,s_diff,p_diff,total_diff) in type_scores:
        type_label = t
        if total_diff < 0:
            if p_diff < 0:
                type_label = u'\u2193'+type_label
            else:
                type_label = u'\u2191'+type_label
            if s_diff < 0:
                type_label = '-'+type_label
            else:
                type_label = '+'+type_label
        else:
            if s_diff < 0:
                type_label = type_label+'-'
            else:
                type_label = type_label+'+'
            if p_diff < 0:
                type_label = type_label+u'\u2193'
            else:
                type_label = type_label+u'\u2191'
        type_labels.append(type_label)
    return type_labels

def _set_bar_labels(bars, word_labels, bar_word_space=1.4):
    for bar_n,bar in enumerate(bars):
        y = bar.get_y()
        height = bar.get_height()
        width = bar.get_width()
        if word_diffs[bar_n] < 0:
            ha='right'
            space = -1*bar_word_space
        else:
            ha='left'
            space = bar_word_space
        ax.text(width+space, bar_n+1, word_labels[bar_n],
                ha=ha, va='center',fontsize=13)
    return ax
