# Shifterator 

The Shifterator package provides functionality for constructing **word shift graphs**. Word shift graphs are vertical bart charts that quantify *which* words contribute to the difference between two texts and *how* they contribute. This allows for more interpretable analysis of sentiment, entropy, and divergence of texts.

This code is still under development. Please open an issue on Github if you find any errors.

## Install

Python code to produce shift graphs can be downloaded by cloning the repository through either the "Clone or download" button on Github or the command line.  

`git clone https://github.com/ryanjgallagher/shifterator.git`

## Producing Word Shift Graphs

Word shift graphs can be constructed to show *relative* differences in sentiment, entropy, and the Kullback-Leibler divergence. They can also show *symmetric* differences via the Jensen-Shannon divergence.  

### Dictionary-Based Shifts

For a dictionary-based analysis, such as sentiment analysis, you must specify a *reference* text and a *comparison* text, by providing two dictionaries, where each has word types as keys and frequencies as values. The word shift will be interpreted in terms of how the comparison text differs from the reference text (see below for details on interpretation).

For sentiment (or any other dictionary-based) analysis, one or two dictionaries can be provided where keys are word types and values are scores. If one dictionary is provided, then that dictionary will be used to measure both texts. If no dictionaries are provided for sentiment analysis, then Shifterator will default to the labMT sentiment dictionary.

```python
from shifterator import relative_shift as rs

# Get a sentiment word shift
sentiment_shift = rs.sentiment_shift(reference=word2freq_ref, 
                                     comparison=word2freq_comp
                                     sent_dict_ref=word2score_ref, 
                                     sent_dict_comp=word2score_comp)
sentiment_shift.get_shift_graph()

```

### Interpreting Word Shift Graphs


## Other Shift Graphs

### Entropy and Kullback-Leibler Shifts

For entropy shifts and Kullback-Leibler divergence shifts, only word frequencies need to be provided to Shifterator. 

**Note**, the Kullback-Leibler divergence is only well-defined if both texts have *exactly* all the same words. If this is not the case, then you should consider using a Jensen-Shannon divergence shift.

```python
# Get an entropy shift
entropy_shift = rs.entropy_shift(reference=type2freq_ref, 
                                 comparison=type2freq_comp,
                                 base=2
entropy_shift.get_shift_graph()

# Get a Kullback-Leibler divergence shift
# Note: only well-defined if both texts have all the same words
kld_shift = rs.kl_divergence_shift(reference=word2freq_ref,
                                   comparison=word2freq_comp,
                                   base=2)
kld_shift.get_shift_graph()
```

### Jensen-Shannon Divergence Shifts

The Jensen-Shannon divergence symmetrizes the Kullback-Leibler divergence by measuring the average divergence of each text from another text representing their average. The measure is symmetric, meaning there is no order in how the texts are specified.

```python
# Get a Jensen-Shannon divergence shift
from shifterator import symmetric_shift as ss
jsd_shift = ss.js_divergence_shift(system_1=word2freq_1, 
                                   system_2=word2freq_2,
                                   base=2)
jsd_shift.get_shift_graph()
```

### General Shift Graphs

If needed, there is a general shift object for more specifications of how the shift is constructed.

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

### Plotting Parameters

### Stop Lens

### Reference Values

### Shift Components

