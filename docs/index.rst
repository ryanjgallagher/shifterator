Shifterator
===========

Shifterator is a Python package for visualizing pairwise comparisons between texts through *word shifts*, a general method for extracting *which* words contribute to a difference between two texts and, importantly, *how* they do so. These contributions are visualized through *word shift graphs*, detailed and interpretable horizontal bar charts that display the interacting components of word shifts. Shifterator can be used for direct text comparisons, sentiment analysis, or as a scientifically sound alternative to a word cloud.

.. image:: figs/shift_sentiment_detailed_full.png
    :width: 450
    :alt: Example of a word shift graph
    :align: center


Features
--------

Shifterator:

- Provides interpretable tools for working with text as data and mapping the complexities of how two texts are similar or different.
- Implements common text comparison measurse, including relative frequency, Shannon entropy, Tsallis entropy, the Kullback-Leibler divergence, and the Jensen-Shannon divergence.
- Unpacks weighted averages calculated from any dictionary-based sentiment analysis method.
- Diagnoses data artificats and measurement errors early in the research process.
- Produces publication-ready visualizations of word shift graphs that provide a detailed summaries of text comparison measures.
- Removes the need to ever make a word cloud for a scientific publication.

Computational social scientists, digital humanists, and other text analysis practitioners can all use Shifterator to construct reliable, robust, and interpretable stories from text data.

.. toctree::
    :maxdepth: 2

    installation
    cookbook/index
    shifts
    plotting
    citations

Search:

* :ref:`search`
