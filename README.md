# Shifterator

The Shifterator package provides functionality for constructing **word shift graphs**, vertical bart charts that quantify *which* words contribute to a pairwise difference between two texts and *how* they contribute. By allowing you to look at changes in how words are used, word shifts help you to conduct analyses of sentiment, entropy, and divergence that are fundamentally more interpretable.

<p align="center">
  <img src ="https://github.com/ryanjgallagher/shifterator/blob/master/figures/presidential-speeches_smaller.png" width="400"/>
</p>

This code is still under development. Please open an issue on Github if you find any errors.

## Install

Python code to produce shift graphs can be downloaded via pip.

`pip install shifterator`

## Producing Word Shift Graphs  

### Relative Word Shifts

Word shift graphs can be constructed to show *relative* differences in sentiment and other dictionary-based scores. For relative word shifts, you specify a *reference* text and a *comparison* text by providing two dictionaries, where each has word types as keys and frequencies as values. The word shift will be interpreted in terms of how the comparison text differs from the reference text (see below for details on interpretation).

For sentiment (or any other dictionary-based) analysis, one or two dictionaries can be provided where keys are word types and values are scores. If one dictionary is provided, then that dictionary will be used to measure both texts.

```python
from shifterator import relative_shift as rs

# Get a sentiment word shift
sentiment_shift = rs.SentimentShift(reference=word2freq_ref,
                                    comparison=word2freq_comp
                                    sent_dict_ref=word2score_ref,
                                    sent_dict_comp=word2score_comp)
sentiment_shift.get_shift_graph()

```

### Interpreting Word Shift Graphs

Word shifts are quantify how each word contributes to the difference between two texts:  

![Contribution equation](https://github.com/ryanjgallagher/shifterator/blob/master/figures/contribution.png)  

The contribution depends on the change in relative frequency of a word, the relative difference between the average score of the word and the reference text's score, and the difference in scores (which is zero if working with a single score dictionary). The main types of contributes depend on how the signs of the contribution components:
1. (+ &#8593;): A relatively positive word (+) is used more (&#8593;)
2. (- &#8595;): A relatively negative word (-) is used less (&#8595;)
3. (+ &#8595;): A relatively positive word (+) is used less (&#8595;)
4. (- &#8593;): A relatively negative word (-) is used more (&#8593;)
5. (&#9651;): A word's score increases (&#9651;)
6. (&#9661;): A word's score decreases (&#9661;)

The first four types of contributions can stack with the later two types to yield 8 qualitatively different ways that a word can contribute in a word shift graph. In some cases, the direction of a word's score change may differ from the direction of the rest of its contribution, in which case we shade the bars to indicate the cancelling of the contributions.


<p align="center">
  <img src ="https://github.com/ryanjgallagher/shifterator/blob/master/figures/shift-components_smaller.png" width="400"/>
</p>


Please see ["Temporal Patterns of Happiness and Information in a Global Social Network: Hedonometrics and Twitter"](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0026752) by Dodds et al. (2011) for a more detailed discussion of interpreting word shift graphs.


## Other Shift Graphs

Relative word shifts can also be constructed from Shannon's entropy and the Kullback-Leibler divergence. A *symmetric* word shift can be produced from the Jensen-Shannon divergence.

### Entropy and Kullback-Leibler Divergence Shifts

For entropy shifts and Kullback-Leibler divergence shifts, only word frequencies need to be provided to Shifterator.

**Note**, the Kullback-Leibler divergence is only well-defined if both texts have *exactly* all the same words. If this is not the case, then you should consider using a Jensen-Shannon divergence shift.

```python
# Get an entropy shift
entropy_shift = rs.EntropyShift(reference=word2freq_ref,
                                comparison=word2freq_comp,
                                base=2
entropy_shift.get_shift_graph()

# Get a Kullback-Leibler divergence shift
# Note: only well-defined if both texts have all the same words
kld_shift = rs.KLDivergenceShift(reference=word2freq_ref,
                                 comparison=word2freq_comp,
                                 base=2)
kld_shift.get_shift_graph()
```

### Jensen-Shannon Divergence Shifts

The Jensen-Shannon divergence symmetrizes the Kullback-Leibler divergence by measuring the average divergence of each text from another text representing their average. The measure is symmetric, meaning there is no order in how the texts are specified.

```python
# Get a Jensen-Shannon divergence shift
from shifterator import symmetric_shift as ss
jsd_shift = ss.JSDivergenceShift(system_1=word2freq_1,
                                 system_2=word2freq_2,
                                 base=2)
jsd_shift.get_shift_graph()
```

### General Shift Graphs

If needed, there is a general shift object that allows for particular specifications.

```python
from shifterator import shifterator as sh

# Construct a general shift object
shift = sh.Shift(system_1=type2freq_1,
                 system_2=type2freq_2,
                 type2score_1=type2score_1,
                 type2score_2=type2score_2,
                 reference_val=None,
                 stop_lens=None)
```

## Functionality

### Calculating Weighted Scores

Given a Shift object, a weighted score can be quickly calculated via the `get_weighted_score()` function. If you only need a weighted score, you do not need to specify the word frequencies and dictionary ahead of time.

```python
# Get a weighted average using a Shift object
shift = sh.Shift()
weighted_avg = shift.get_weighted_score(word2freq, word2score)
```

### Word Shift Scores and Shift Components

Word shift scores can be calculated by calling the `get_shift_scores()` function.

```python
# Get shift scores of each word as a dictionary
type2shift_scores = shift.get_shift_scores(details=False)
```
The components of the shift score are stored in the Shift object as `type2p_avg`, `type2s_diff`, `type2p_diff`, `type2s_ref_diff`. If `details=True` when calculating shift scores, then all of those components are returned with the overall shift scores.  

```python
# Get the components of the shift score for each word
type2p_diff,type2s_diff,type2p_avg,type2s_ref_diff,type2shift_score = shift.get_shift_scores()
```

The sum of each type of contribution can be retrieved by calling `get_shift_component_sums()`.

```python
# Get the total sum of each type of contribution
shift_components = shift.get_shift_component_sums()
```


### Stop Lens

There may be times when you want to exclude particular words based on their scores to better understand the dynamics of a particular range of scores. A stop lens can be specified as a list of tuples when initializing a Shift object. The object will then automatically exclude words within the stop lens for all following calculations.

```python
# Set a stop lens on a Shift object
sentiment_shift = rs.SentimentShift(reference=word2freq1,
                                    comparison=word2freq2,
                                    sent_dict_ref=word2score,
                                    stop_lens=[(4,6), (0,1), (8,9)])
```

### Reference Values

For relative shifts, the weighted average of the reference text is automatically used as the reference value. If you would like to override this choice, or if you want to set the reference value for a symmetric shift, you can specify the reference value when initializing a Shift object.

```python
# Manually set reference value on a Shift object
jsd_shift = ss.JSDivergenceShift(system_1=word2freq_1,
                                 system_2=word2freq_2)
```

### Plotting Parameters

There are a number of plotting parameters that can be passed to `get_shift_graph()` when constructing a word shift graph. See [`get_plot_params()`](https://github.com/ryanjgallagher/shifterator/blob/master/shifterator/plotting.py#L17) for the parameters that can currently altered in a word shift graph.


## Contributing

If you run into any issues, please feel free to open an issue on Github or submit a pull request.  

Are you proficient in R? We're looking for help developing an R package to produce word shift graphs! Get in touch with us if you are interested.
