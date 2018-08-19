"""
shifterator.py

Author: Ryan J. Gallagher, Network Science Institute, Northeastern University
Last updated: June 13th, 2018

Requires: Python 3

TODO:
- Add funcs to shift class that allow for easy updating of type2freq dicts
- Make it easy to remove / reset the filter. This will involve having to hold
  onto stop words, their freqs, and their values (discarded as of now)
- Make it so you can specify words as stop words instead of just a filter window
- Clean up class docstrings to fit standards of where things should be described
  (whether it's in init or under class, and listing what funcs are available)
"""

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ------------------------------------------------------------------------------
# ---------------------------- GENERAL SHIFT CLASS -----------------------------
# ------------------------------------------------------------------------------
class Shift:
    def __init__(self, system_1, system_2, reference_value=None,
                 filenames=True, type2score_1=None, type2score_2=None,
                 stop_lens=None, delimiter=','):
        """
        Shift object for calculating weighted scores of two systems of types,
        and the shift between them

        Parameters
        ----------
        system_1, system_2: dict or str
            if dict, then keys are types of a system and values are frequencies
            of those types. if str and filenames=False, then the types are
            assumed to be tokens separated by white space. If str and
            filenames=True, then types are assumed to be tokens and text is read
            in line by line from the designated file and split on white space
        reference_value: float, optional
            the reference score from which to calculate the deviation. If None,
            defaults to the weighted score of system_1
        filenames: bool, optional
            True if system_1 and system_2 are filenames of files with text to
            parse
        type2score_1, type2score_2: dict or str, optional
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
        # Load type2freq dictionaries
        if isinstance(system_1, dict) and isinstance(system_2, dict):
            self.type2freq_1 = system_1
            self.type2freq_2 = system_2
        elif isinstance(system_1, str) and isinstance(system_2, str):
            if filenames is True:
                self.type2freq_1 = get_freqs_from_file(system_1)
                self.type2freq_2 = get_freqs_from_file(system_2)
            elif filenames is False:
                self.type2freq_1 = dict(Counter(system_1.split()))
                self.type2freq_2 = dict(Counter(system_2.split()))
        else:
            warning = 'Shift object was not given text, a file to parse, or '+\
                      'frequency dictionaries. Check input.'
            warnings.warn(warning, Warning)
            self.type2freq_1 = dict()
            self.type2freq_2 = dict()
        # Load type2score dictionaries
        if type2score_1 is not None and type2score_2 is not None:
            self.type2score_1 = get_score_dictionary(type2score_1, delimiter)
            self.type2score_2 = get_score_dictionary(type2score_2, delimiter)
        elif type2score_1 is not None:
            self.type2score_1 = get_score_dictionary(type2score_1, delimiter)
            self.type2score_2 = self.type2score_1
        elif type2score_2 is not None:
            self.type2score_2 = get_score_dictionary(type2score_2, delimiter)
            self.type2score_1 = self.type2score_2
        else:
            self.type2score_1 = {t : 1 for t in self.type2freq_1}
            self.type2score_2 = {t : 1 for t in self.type2freq_2}
        # Filter type dictionaries by stop lense
        self.stop_lens = stop_lens
        if stop_lens is not None:
            self.type2freq_1,self.type2score_1,stop_words = filter_by_scores(self.type2freq_1,
                                                                             self.type2score_1,
                                                                             stop_lens)
            self.type2freq_2,self.type2score_2,stop_words = filter_by_scores(self.type2freq_2,
                                                                             self.type2score_2,
                                                                             stop_lens)
            self.stop_words = stop_words
        # Set reference value
        if reference_value is not None:
            self.reference_value = reference_value
        else:
            self.reference_value = self.get_weighted_score(self.type2freq_1,
                                                           self.type2score_1)
        # Set default score shift values
        self.diff = None
        self.type2p_diff = None
        self.type2s_diff = None
        self.type2p_avg = None
        self.type2s_ref_diff = None
        self.type2shift_score = None

    def get_types(self, type2freq_1, type2score_1, type2freq_2, type2score_2):
        """
        Returns the common "vocabulary" between the types of both systems and
        the types in the dictionaries

        Parameters
        ----------
        type2freq: dict
            keys are types and values are frequencies
        type2score: dict
            keys are types and values are scores
        """
        # Enforce common score vocabulary
        if len(set(type2score_1.keys()).difference(type2score_2.keys())) != 0:
            warning = 'Score dictionaries do not have a common vocabulary. '\
                      +'Shift is not well-defined.'
            warnings.warn(warning, Warning)
            #return
        # Get observed types that are also in score dicts
        types_1 = set(type2freq_1.keys()).intersection(set(type2score_1.keys()))
        types_2 = set(type2freq_2.keys()).intersection(set(type2score_2.keys()))
        types = types_1.union(types_2)
        return types

    def get_weighted_score(self, type2freq, type2score):
        """
        Calculate the average score of the system specified by the frequencies
        and scores of the types in that system

        Parameters
        ----------
        type2freq: dict
            keys are types and values are frequencies
        type2score: dict
            keys are types and values are scores

        Returns
        -------
        s_avg: float
            Average weighted score of system
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
                         type2freq_2=None, type2score_2=None,
                         reference_value=None, normalize=True, details=False):
        """
        Calculates the type shift scores between two systems

        Parameters
        ----------
        type2freq: dict
            keys are types and values are frequencies. If None, defaults to the
            system_1 and system_2 type2freq dicts respectively
        type2score: dict
            keys are types and values are scores. If None, defaults to the
            system_1 and system_2 type2score dicts respectively
        reference_value: float
            the reference score from which to calculate the deviation. If None,
            defaults to the weighted score given by type2freq_1 and type2score_1
        normalize: bool
            if True normalizes shift scores so they sum to 1 or -1
        details: bool,
            if True returns each component of the shift score and the final
            normalized shift score. If false, only returns the normalized shift
            scores

        Returns
        -------
        type2p_diff: dict
            if details is True, returns dict where keys are types and values are
            the difference in relatively frequency, i.e. p_i,2 - p_i,1 for type i
        type2s_diff: dict,
            if details is True, returns dict where keys are types and values are
            the relative differences in score, i.e. s_i,2 - s_i,1 for type i
        type2p_avg: dict,
            if details is True, returns dict where keys are types and values are
            the average relative frequencies, i.e. 0.5*(p_i,1+p_i,2) for type i
        type2s_ref_diff: dict
            if details is True, returns dict where keys are types and values are
            relative deviation from reference score, i.e. 0.5*(s_i,2+s_i,1)-s_ref
            for type i
        type2shift_score: dict
            words are keys and values are shift scores
        """
        # Check input of type2freq and type2score dicts
        if type2freq_1 is None:
            type2freq_1 = self.type2freq_1
        if type2score_1 is None:
            type2score_1 = self.type2score_1
        if type2freq_2 is None:
            type2freq_2 = self.type2freq_2
        if type2score_2 is None:
            type2score_2 = self.type2score_2

        # TODO: get rid of this hack!
        type2score_1 = {t:s for t,s in type2score_1.items() if t in type2score_2}
        type2score_2 = {t:s for t,s in type2score_2.items() if t in type2score_1}

        # Enforce common score vocabulary
        types = self.get_types(type2freq_1, type2score_1,
                               type2freq_2, type2score_2)

        # Get total frequencies
        total_freq_1 = sum([freq for t,freq in type2freq_1.items() if t in types])
        total_freq_2 = sum([freq for t,freq in type2freq_2.items() if t in types])
        # Get relative frequency of types in both systems
        type2p_1 = {t:type2freq_1[t]/total_freq_1 if t in type2freq_1 else 0
                    for t in types}
        type2p_2 = {t:type2freq_2[t]/total_freq_2 if t in type2freq_2 else 0
                    for t in types}
        # Get average relative frequency of types
        type2p_avg = {t:0.5*(type2p_1[t]+type2p_2[t]) for t in types}

        # Check input of reference value
        if reference_value is None:
            s_avg_ref = self.get_weighted_score(type2freq_1, type2score_1)
        # Calculate relative diffs in freq
        type2p_diff = {t:type2p_2[t]-type2p_1[t] for t in types}
        # Calculate relative diffs in score and relative diff from ref
        type2s_diff = {}
        type2s_ref_diff = {}
        for t in types:
            type2s_diff[t] = type2score_2[t]-type2score_1[t]
            type2s_ref_diff[t] = 0.5*(type2score_2[t]+type2score_1[t])-s_avg_ref

        # Calculate total shift scores
        type2shift_score = {t : type2p_diff[t]*type2s_ref_diff[t]\
                                +type2s_diff[t]*type2p_avg[t]
                                for t in types if t in types}

        # Normalize the total shift scores
        if normalize:
            total_diff = sum(type2shift_score.values())
            self.diff = total_diff
            type2shift_score = {t : shift_score/abs(total_diff) for t,shift_score
                                in type2shift_score.items()}


        # Set results in shift object
        self.type2p_diff = type2p_diff
        self.type2s_diff = type2s_diff
        self.type2p_avg = type2p_avg
        self.type2s_ref_diff = type2s_ref_diff
        self.type2shift_score = type2shift_score
        # Return shift scores
        if details:
            return type2p_diff,type2s_diff,type2p_avg,type2s_ref_diff,type2shift_score
        else:
            return type2shift_score

    def get_shift_component_sums(self, type2freq_1=None, type2score_1=None,
                                 type2freq_2=None, type2score_2=None,
                                 reference_value=None, normalize=True):
        """

        """
        # Check input of type2freq and type2score dicts
        if type2freq_1 is None:
            type2freq_1 = self.type2freq_1
        if type2score_1 is None:
            type2score_1 = self.type2score_1
        if type2freq_2 is None:
            type2freq_2 = self.type2freq_2
        if type2score_2 is None:
            type2score_2 = self.type2score_2
        # Get shift scores
        if self.type2shift_score is None:
            shift_scores = self.get_shift_scores(type2freq_1=type2freq_1,
                                                 type2score_1=type2score_1,
                                                 type2freq_2=type2freq_2,
                                                 type2score_2=type2score_2,
                                                 reference_value=reference_value,
                                                 normalize=normalize,
                                                 details=True)
        else:
            shift_scores = [(t, self.type2s_diff[t], self.type2p_diff[t],
                             self.type2p_avg[t], self.type2s_ref_diff[t],
                             self.type2shift_score[t]) for t in self.type2s_diff]

        # Sum up components of shift score
        pos_freq_pos_score = 0
        pos_freq_neg_score = 0
        neg_freq_pos_score = 0
        neg_freq_neg_score = 0
        pos_s_diff = 0
        neg_s_diff = 0
        for t,s_diff,p_diff,p_avg,s_ref_diff, _ in shift_scores:
            # Get contribution of p_diff*s_ref_diff term
            if p_diff > 0:
                if s_ref_diff > 0:
                    pos_freq_pos_score += p_diff*s_ref_diff
                else:
                    pos_freq_neg_score += p_diff*s_ref_diff
            else:
                if s_ref_diff > 0:
                    neg_freq_pos_score += p_diff*s_ref_diff
                else:
                    neg_freq_neg_score += p_diff*s_ref_diff
            # Get contribution of s_diff term
            if s_diff > 0:
                pos_s_diff += p_avg*s_diff
            else:
                neg_s_diff += p_avg*s_diff

        return (pos_freq_pos_score, pos_freq_neg_score,
                neg_freq_pos_score, neg_freq_neg_score,
                pos_s_diff, neg_s_diff)

    def get_shift_graph(self, top_n=50, width=6, height=15, inset=True,
                        score_colors=('#ffff80','#FDFFD2','#3377ff', '#C4CAFC',
                                      '#9E75B7', '#FECC5D'),
                        width_scaling=1.4, bar_type_space_scaling=0.05,
                        xlabel=None, ylabel=None, title=None,
                        xlabel_fontsize=18, ylabel_fontsize=18, title_fontsize=14,
                        show_plot=True, tight=True):
        # TODO: **kwargs
        """
        Plot the simple shift graph between two systems of types

        Parameters
        ----------
        top_n: int
            display the top_n types as sorted by their absolute contribution to
            the difference between systems
        bar_colors: 4-tuple
            colors to use for bars where first and second entries are the colors
            for types that have positive and negative relative score differences
            relatively
        bar_type_space_scaling: float
            parameter between 0 and 1 that controls the space between the end of
            each bar and its text label. Increase if more space is desired
        width_scaling: float
            parameter between 0 and 1 that controls the width of the x-axis.
            If types overlap with the y-axis then increase the scaling
        insets: bool
            whether to show insets showing the cumulative contribution to the
            shift by ranked words, and the relative sizes of each system
        show_plot: bool
            whether to show plot on finish
        tight: bool
            whether to call plt.tight_layout() on the plot

        Returns
        -------
        ax
            matplotlib ax of shift graph. Displays shift graph if show_plot=True
        """
        # TODO: wrap the parts into functions (basic bars, contributions, handling
        #       the labels, etc)
        if self.type2shift_score is None:
            self.get_shift_scores(details=False)
        # Get type score components
        type_scores = [(t, self.type2s_diff[t], self.type2p_diff[t],
                        self.type2p_avg[t], self.type2s_ref_diff[t],
                        self.type2shift_score[t]) for t in self.type2s_diff]
        # Reverse sorting to get highest scores, then reverse top n for plotting order
        type_scores = sorted(type_scores, key=lambda x:abs(x[-1]), reverse=True)[:top_n]
        type_scores.reverse()

        # Plot scores, height:width ratio = 2.5:1
        f,ax = plt.subplots(figsize=(width,height))
        ax.margins(y=0.01)
        # Get bar heights
        heights_comp1, heights_comp2, bottoms, bar_ends = _get_bar_heights(type_scores,
                                                                           abs(self.diff))
        # Get bar colors
        bar_colors_comp1,bar_colors_comp2 = _get_bar_colors(type_scores, score_colors)
        # Plot the skeleton of the word shift
        ax.barh(range(1,len(type_scores)+1), heights_comp1, 0.8, linewidth=1,
                       align='center', color=bar_colors_comp1, edgecolor=['black']*top_n)
        ax.barh(range(1, len(type_scores)+1), heights_comp2, 0.8, left=bottoms,
                       linewidth=1, align='center', color=bar_colors_comp2, edgecolor=['black']*top_n)

        # Get total contribution component bars
        # +freq+score, +freq-score, -freq+score, -freq-score, +s_diff, -s_diff
        total_comp_sums = self.get_shift_component_sums()
        total_comp_sums = [100*s for s in total_comp_sums]
        ys = [top_n+2,top_n+3.5,top_n+3.5,top_n+5,top_n+5,top_n+6.5,top_n+6.5]
        comp_colors = ['#707070', score_colors[5], score_colors[4], score_colors[3],
                       score_colors[2], score_colors[1], score_colors[0]]
        comp_bars = [sum(total_comp_sums)] + list(reversed(total_comp_sums))
        ax.barh(ys, comp_bars, 0.8, linewidth=1, align='center',
                            color=comp_colors, edgecolor=['black']*top_n)
        # TODO: add symbols to ends of component bars

        # Estimate bar_type_space as a fraction of largest xlim
        x_width = 2*abs(max(ax.get_xlim(), key=lambda x: abs(x)))
        bar_type_space = bar_type_space_scaling*x_width
        # Format word labels with up/down arrows and +/-
        type_labels = _get_shift_type_labels(type_scores)
        # Add word labels to bars
        ax,text_objs = _set_bar_labels(ax, bar_ends, type_labels,
                                       bar_type_space=bar_type_space)
        # Adjust for width of word labels and make x-axis symmetric
        ax = _adjust_axes_for_labels(f, ax, bar_ends, comp_bars, text_objs,
                                     bar_type_space=bar_type_space,
                                     width_scaling=width_scaling)
        # Make x-axis tick labels bigger
        for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(12)

        # Flip y-axis tick labels and make sure every 5th tick is labeled
        y_ticks = list(range(1,top_n,5))+[top_n]
        y_tick_labels = [str(n) for n in (list(range(top_n,1,-5))+['1'])]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels, fontsize=14)

        # Add center dividing line
        y_min,y_max = ax.get_ylim()
        ax.plot([0,0],[1,y_max], '-', color='black', linewidth=0.7)
        # Add dividing line between words and component bars
        x_min,x_max = ax.get_xlim()
        ax.plot([x_min,x_max], [top_n+1,top_n+1], '-', color='black', linewidth=0.7)
        ax.plot([x_min,x_max], [top_n+2.75, top_n+2.75], '-', color='black', linewidth=0.5)

        if inset:
            # Get cumulative diff
            scores = sorted([100*s for s in self.type2shift_score.values()],
                             key=lambda x:abs(x), reverse=True)
            cum_scores = np.cumsum(scores)
            # Add inset axes
            left, bottom, width, height = [0.2, 0.18, 0.125, 0.17]
            in_ax = f.add_axes([left, bottom, width, height])
            # Plot cumulative diff
            in_ax.semilogy(cum_scores, range(len(cum_scores)), '-o', color='black',
                           linewidth=0.5, markersize=1.0)
            in_ax.set_xlabel('$\sum_i^r \delta s_{avg,i}$')
            # Set view line
            in_x_min,in_x_max = in_ax.get_xlim()
            in_ax.plot([in_x_min,in_x_max], [top_n,top_n], '-', color='black', linewidth=0.5)
            # Clean up axes
            in_y_min,in_y_max = in_ax.get_ylim()
            in_ax.set_ylim((in_y_max, in_y_min))
            in_ax.margins(x=0)
            in_ax.margins(y=0)

        # Set axis labels and title
        if xlabel is None:
            xlabel = 'Per type average score shift $\delta s_{avg,r}$ (%)'
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
        if ylabel is None:
            ylabel = 'Type rank $r$'
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        if title is None:
            s_avg_1 = self.get_weighted_score(self.type2freq_1,self.type2score_1)
            s_avg_2 = self.get_weighted_score(self.type2freq_2,self.type2score_2)
            title = '$\Phi_{\Omega^{(2)}}$: $s_{avg}^{(1)}=$'+'{0:.2f}'\
                    .format(s_avg_1)+'\n'\
                    +'$\Phi_{\Omega^{(1)}}$: $s_{avg}^{(2)}=$'+'{0:.2f}'\
                    .format(s_avg_2)
        ax.set_title(title, fontsize=14)

        # Show and return plot
        if tight:
            plt.tight_layout()
        if show_plot:
            plt.show()
        return ax


# ------------------------------------------------------------------------------
# ------------------------------ HELPER FUNCTIONS ------------------------------
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
        Frequency and score dicts filtered of words whose score fall within stop window
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

def get_score_dictionary(scores, delimiter=','):
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
    delimiter: str
        delimiter used in the dictionary file

    Returns
    -------
    type2score, dict
        dictionary where keys are types and values are scores of those types
    """
    if type(scores) is dict:
        return scores
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
            t,score = line.strip().split(delimiter)
            type2score[t] = score

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
    for (t,s_diff,p_diff,p_avg,s_ref_diff,total_diff) in type_scores:
        type_label = t
        if total_diff < 0:
            if p_diff < 0:
                type_label = u'\u2193'+type_label
            else:
                type_label = u'\u2191'+type_label
            if s_ref_diff < 0:
                type_label = '-'+type_label
            else:
                type_label = '+'+type_label
        else:
            if s_ref_diff < 0:
                type_label = type_label+'-'
            else:
                type_label = type_label+'+'
            if p_diff < 0:
                type_label = type_label+u'\u2193'
            else:
                type_label = type_label+u'\u2191'
        type_labels.append(type_label)
    return type_labels

def _get_bar_colors(type_scores, score_colors):
    """

    """
    bar_colors_comp1 = []
    bar_colors_comp2 = []
    for (_,s_diff,p_diff,p_avg,s_ref_diff,_) in type_scores:
        # Get first p_diff/s_ref_diff comp colors
        if s_ref_diff > 0:
            if p_diff > 0:
                bar_colors_comp1.append(score_colors[0])
            else:
                bar_colors_comp1.append(score_colors[1])
        else:
            if p_diff > 0:
                bar_colors_comp1.append(score_colors[2])
            else:
                bar_colors_comp1.append(score_colors[3])
        # Get s_diff comp colors
        if s_diff > 0:
            bar_colors_comp2.append(score_colors[4])
        else:
            bar_colors_comp2.append(score_colors[5])
    return (bar_colors_comp1, bar_colors_comp2)

def _get_bar_heights(type_scores, normalizer):
    """
    tuple: (bar 1 height, bar 2 bottom, bar 2 height)
    """
    heights_comp1 = []
    heights_comp2 = []
    bottoms = []
    bar_ends = []
    for (_,s_diff,p_diff,p_avg,s_ref_diff,_) in type_scores:
        heights_comp1.append(100*p_diff*s_ref_diff/normalizer)
        heights_comp2.append(100*p_avg*s_diff/normalizer)
        # Determine if direction of comp bars are congruent
        if np.sign(s_ref_diff*p_diff)*np.sign(s_diff) == 1:
            contribution = 100*(p_diff*s_ref_diff+p_avg*s_diff)/normalizer
            bar_ends.append(contribution)
            if np.sign(s_diff) == 1:
                bottoms.append(100*p_diff*s_ref_diff/normalizer)
            else:
                bottoms.append(contribution/normalizer)
        else:
            bottoms.append(0)
            if abs(s_ref_diff*p_diff) > abs(p_avg*s_diff):
                bar_ends.append(100*s_ref_diff*p_diff/normalizer)
            else:
                bar_ends.append(100*p_avg*s_diff/normalizer)

    return (heights_comp1, heights_comp2, bottoms, bar_ends)

def _set_bar_labels(ax, bar_ends, type_labels, bar_type_space=1.4):
    text_objs = []
    for bar_n,height in enumerate(range(len(bar_ends))):
        #height = bar.get_height()
        width = bar_ends[bar_n]
        if bar_ends[bar_n] < 0:
            ha='right'
            space = -1*bar_type_space
        else:
            ha='left'
            space = bar_type_space
        t = ax.text(width+space, bar_n+1, type_labels[bar_n],
                    ha=ha, va='center',fontsize=13)
        text_objs.append(t)
    return (ax, text_objs)

def _adjust_axes_for_labels(f, ax, bar_ends, comp_bars, text_objs,
                            bar_type_space, width_scaling):
    # Get the max length
    lengths = []
    for bar_n,bar_end in enumerate(bar_ends):
        bar_length = bar_end
        bbox = text_objs[bar_n].get_window_extent(renderer=f.canvas.get_renderer())
        bbox = ax.transData.inverted().transform(bbox)
        text_length = abs(bbox[0][0]-bbox[1][0])
        if bar_length > 0:
            lengths.append(bar_length+text_length+bar_type_space)
        else:
            lengths.append(bar_length-text_length-bar_type_space)
    # Add the top component bars to the lengths to check
    comp_bars = [abs(b) for b in comp_bars]
    lengths += comp_bars
    # Get max length
    max_length = width_scaling*abs(sorted(lengths, key=lambda x: abs(x), reverse=True)[0])
    # Symmetrize the axis around that max length
    ax.set_xlim((-1*max_length, max_length))
    return ax
