"""
wordshift.py

Author: Ryan J. Gallagher, Network Science Institute, Northeastern University
Last updated: May 15th, 2018
"""
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# TODO: use inheritance to clean up code
# 1) word_shift -> (score_shift -> sentiment_shift) and (divergence_shift)
#    divergence_shift doesn't have methods like get_avg_score
# 2) Can word shift could be put into general word_shift class, or is the sentiment
#    one too specialized?

class word_shift:
    def __init__(self, ref_text, comp_text, filenames=False):
        """
        ref_text: str or dict, if str and filenames=False, then the text is read
                  in directliy and split on white space. If str and
                  filenames=True, then text is read in line by line from the
                  designated file and split on white space. If dict, then should
                  be of the form where keys are words and values are frequencies
                  of those words
        comp_text: str or dict, of the same type as ref_text
        filenames: bool, True if ref_text and comp_text are
                   filenames of files with text to parse
        """
        # Load text into word2freq dictionaries
        if isinstance(ref_text, dict) and isinstance(comp_text, dict):
            self.word2freq_ref = text_ref
            self.word2freq_comp = text_comp
        elif isinstance(ref_text, basestring) and isinstance(comp_text, basestring):
            if filenames is True:
                self.word2freq_ref = get_word_freqs_from_file(ref_text)
                self.word2freq_comp = get_word_freqs_from_file(comp_text)
            elif filename is False:
                self.word2freq_ref = dict(Counter(ref_text.split()))
                self.word2freq_comp = dict(Counter(comp_text.split()))
        else:
            warning = 'Shift object was not given text, a file to parse, or '+\
                      'word frequency dictionaries. Check input texts.'
            warnings.warn(warning, Warning)
            self.word2freq_ref = dict()
            self.word2freq_comp = dict()
        # Set vocab
        self.vocab = (set(self.word2freq_ref.keys())\
                      .union(set(self.word2freq_comp.keys())))

        # TODO: add functions that allow you to easily update the word2freq
        #       dictionaries. What input should be accepted for that?

class divergence_shift(word_shift):
    def __init__(self, ref_text, comp_text, filenames=False, divergence='jsd',
                 alpha=1.5):
        """
        ref_text: str or dict, if str and filenames=False, then the text is read
                in directliy and split on white space. If str and filenames=True,
                then text is read in line by line from the designated file and
                split on white space. If dict, then should be of the form where
                keys are words and values are frequencies of those words. If
                divergence='jsd', ref_text and comp_text are interchangeable
        comp_text: str or dict, of the same type as ref_text. If divergence='jsd'
                   ref_text and comp_text are interchangeable
        filenames: bool, True if ref_text and comp_text are
                   filenames of files with text to parse
        divergence: str, type of divergence to calculate. Options: 'jsd','kld'
        alpha: float, (0,2], order of generalized divergence
        """
        word_shift.__init__(self, ref_text, comp_text, filenames)
        self.divergence = divergence

class sentiment_shift(word_shift):
    def __init__(self, ref, comp_text, filenames=False, dictionary='labMT_english',
                 stop_radius=1.0, middle_score=5.0):
        """
        reference_text: str or dict, if str and filenames=False, then the text
                        is read in directliy and split on white space. If str
                        and filenames=True, then text is read in line by line
                        from the designated file and split on white space. If
                        dict, then should be of the form where keys are words
                        and values are frequencies of those words
        comparision_text: str or dict, of the same type as reference_text. Word
                          shift scores will be in terms of how the comparison
                          text differs from the reference text
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
        # Note, in word_shift object reference_text is ref_text, comparison_text
        # is comp_text
        word_shift.__init__(reference_text, comparison_text, filenames)
        # Load sentiment dictionary
        self.stop_radius = stop_radius
        self.middle_score = middle_score
        self.word2sentiment = get_score_dictionary(dictionary, stop_radius,
                                                   middle_score)
        # Set vocabulary from loaded words: (ref \cup comp) \cap sent_words
        self.vocab = self.vocab.intersection(set(self.word2sentiment.keys()))
        if len(vocab) == 0:
            warning = 'No words in input texts are in score dictionary'
            warnings.warn(warning, Warning)
        # Initialize word shift score components to None
        self.word2p_diff = None
        self.word2s_diff = None
        self.word2shift_score = None

    def get_average_sentiment(self, text='reference'):
        """
        Calculate the average sentiment of the comparison or reference text

        INPUT
        -----
        text: str, whether to calculate average for 'comparison' or 'reference'

        OUTPUT
        ------
        average_sentiment: float, average sentiment of comparison or reference
        """
        # Throw warning if word2freq dicts haven't been initialized
        if (text=='reference' and self.word2freq_ref) is None\
            or (text=='comparison' and self.word2freq_comp is None):
            warning = 'Text has not been specified in word shift object. Please'\
                      +'initialize object.'
                warnings.warn(warning, Warn)
                return
        # Get average sentiment
        if text == 'reference':
            return self.get_weighted_score(self.word2freq_ref,self.word2sentiment)
        elif text == 'comparison':
            return self.get_weighted_score(self.word2freq_comp,self.word2sentiment)
        else:
            warning = "Please specify either text='reference' or 'comparison'."
            warnings.warn(warning, Warning)
            return

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
            total_diff = np.abs(sum(word2shift_score.values()))
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

    def get_word_shift(self, top_n=50, bar_colors=('#ffff80','#3377ff'),
                       bar_word_space=0.5, width_scaling=1.4, show_plot=True
                       tight=True, xlabel=None, ylabel=None, title=None,
                       xlabel_fontsize=18, ylabel_fontsize=18, title_fontsize=14):
        """
        Plot the word shift of the comparison text with respect to the reference
        text

        INPUT
        -----
        top_n: int, display the top_n words as sorted by their absolute
               contribution to the difference
        bar_colors: tuple, colors to use for bars where first entry is the color
                    for words that have a positive relative sentiment increase
                    and the second is for those that have a negative relative
                    decrease
        bar_word_space: float, space between the end of each bar and the
                        corresponding label
        width_scaling: float, parameter controls the width of the x-axis. If
                       words overlap with the y-axis then increase the scaling
        show_plot: bool, whether to show plot on finish
        tight: bool, whether to call plt.tight_layout() on the plot
        """
        if self.word2shift_score is None:
            self.get_word_shift_scores(details=False)
        # Sort word scores and take top_n. Reverse for plotting
        word_scores = [(word, self.word2s_diff[word], self.word2p_diff[word],
                        self.word2shift_score[word])]
        word_scores = sorted(word_scores, key=labmda x:np.abs(x[3]))[:top_n]
        word_diffs = [100*score for (word,s_diff,p_diff,score) in word_scores]
        # Get bar colors
        bar_colors = [bar_colors[0] if s_diff>0 else bar_colors[1]\
                      for (word,s_diff,p_diff,score) in word_scores]
        # Plot scores, height:width=2.5:1
        f,ax = plt.subplots(figsize=(6,15))
        ax.margins(y=0.01)
        # Plot the skeleton of the word shift
        # edgecolor thing is a workaround for a bug in matplotlib
        bars = ax.barh(range(1,len(word_scores)+1), word_diffs, .8, linewidth=1
                       align='center', color=bar_colors, edgecolor=['black']*top_n)
        # Add center line dividing line
        ax.plot([0,0],[1,top_n], '-', color='black', linewidth=0.7)
        # Make sure there's the same amount of space on either side of y-axis,
        # and add space for word labels using 'width_scaling' (can we automate?)
        x_min,x_max = ax.get_xlim()
        x_sym = width_scaling*max([np.abs(x_min),np.abs(x_max)])
        ax.set_xlim((-1*x_sym, x_sym))
        # Flip y-axis tick labels and make sure every 5th tick is labeled
        y_ticks = list(range(1,top_n,5))+[top_n]
        y_tick_labels = [str(n) for n in (list(range(top_n,1,-5))+['1'])]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        # Format word labels with up/down arrows and +/-
        word_labels = _get_word_shift_word_labels(word_scores)
        # Add word labels to bars
        ax = _set_bar_labels(bars, word_labels, bar_word_space=bar_word_space)
        # Set axis labels and title
        if xlabel is None:
            xlabel = 'Per word average sentiment shift $\delta s_{avg,r}$ (%)'
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
        if ylabel is None:
            ylabel = 'Word rank $r$'
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        if title is None:
            s_avg_ref = self.get_average_sentiment(text='reference')
            s_avg_comp = self.get_average_sentiment(text='comparison'_)
            title = '$T_{ref}$: '+'$s_{avg}^{(ref)}=$'+'{0:.2f}'.format(s_avg_ref)+'\n'\
                    +'$T_{comp}$: '+'$s_{avg}^{(comp)}=$'+'{0:.2f}'.format(s_avg_comp)
        ax.set_title(title_str, fontsize=14)
        # Show and return plot
        if tight:
            plt.tight_layout()
        if show_plot:
            plt.show()
        return ax

def get_weighted_score(word2freq, word2score):
    """
    Should make more flexible so it's easier to use

    Let user enter text like they would for word shift object
    Let user pick dictionary here
    """
    # Get vocabulary of words that are in our text and in our dictionary
    vocab = set(word2freq.keys()).intersection(set(word2score.keys()))
    if len(vocab) == 0:
        warning = 'No words in input text are in score dictionary'
        warnings.warn(warning, Warning)
        return
    # Get weighted score and total frequency
    f_total = sum([freq for word,freq in word2freq.items() if word in vocab])
    s_weighted = sum([word2score[word]*freq for word,freq in word2freq.items()
                      if word in vocab])
    s_avg = s_weighted / f_total
    return s_avg

def get_score_dictionary(dictionary, stop_radius=0.0, middle_score=5.0,
                         delimiter=','):
    """
    Loads a dictionary of word scores.

    INPUT
    -----
    dictionary: str, name of dictionary to load, or file path of dictionary to
                     load. Options: 'labMT_english',
    delimiter: str, delimiter used in the dictionary file

    OUTPUT:
    ------
    """
    # Check if CSV of dictionary exists, otherwise use direct file path
    dictionaries = os.listdir('data')
    if dictionary+'.csv' in dictionaries:
        dictionary_file = 'data/'+dictionary+'.csv'
    else:
        dictionary_file = dictionary
    # Load sentiment dictionary
    word2score = {}
    with open(dictionary_file, 'r') as f:
        for line in f:
            word,score = line.strip().split(delimiter)
            word2score[word] = score
    # Filter dictionary of words outside of stop range
    if stop_radius > 0:
        lower_stop = middle_score - stop_radius
        upper_stop = middle_score + stop_radius
        word2score = {word:score for word,score in word2score.items()
                      if score <= lower_stop or score >= upper_stop}
        return word2score
    else:
        return word2score

def get_word_freqs_from_file(filename):
    """
    Parses text of a file line by line, splitting across white space

    INPUT
    -----
    filename: str, file to load text from

    OUTPUT
    ------
    word2freq: dict, keys are words and values are frequencies of those words
    """
    word2freq = Counter()
    with open(filename, 'r') as f:
        for line in f:
            words = line.strip().split()
            Counter.update(words)
    return dict(word2freq)

def _get_word_shift_word_labels(word_scores):
    """

    """
    word_labels = []
    for (word,s_diff,p_diff,total_diff) in word_scores:
        word_label = word
        if total_diff < 0:
            if p_diff < 0:
                word_label = u'\u2193'+word_label
            else:
                word_label = u'\u2191'+word_label
            if s_diff < 0:
                word_label = '-'+word_label
            else:
                word_label = '+'+word_label
        else:
            if s_diff < 0:
                word_label = word_label+'-'
            else:
                word_label = word_label+'+'
            if p_diff < 0:
                word_label = word_label+u'\u2193'
            else:
                word_label = word_label+u'\u2191'
        word_labels.append(word_label)
    return word_labels

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
