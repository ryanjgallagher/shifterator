Sentiment Analysis
==================

Word Shifts from Dictionary Scores
----------------------------------

Weighted word shift graphs can be used for dictionary-based sentiment analysis, or any other analysis where words are weighted according to scores provided by a lexicon.

If you have a scores from a lexicon, they can be loaded into :code:`type2score_1` and :code:`type2score_2` when constructing the word shift object. If you only have one lexicon, only :code:`type2score_1` needs to be specified.

.. code-block:: python

    weighted_shift = sh.WeightedAvgShift(type2freq_1=type2freq_1,
                                         type2freq_2=type2freq_2,
                                         type2score_1=type2score_1
                                         type2score_2=type2score_2)

.. note::
    Word shift graphs work best with *continuous* scores, which is what we recommend using. Discrete or binary sentiment scores can also be used if necessary.

.. warning::
    Dictionary-based sentiment analysis and word shift graphs should only be used on long, aggregate texts, and not short texts, like individual tweets. There are many easily constructed counterexamples to show how dictionary methods can fail on short texts. For example, the sarcastic exasperation, "Oh great, the birthday party is cancelled," has many positive words but is negative. These contextual issues are mitigated when considering longer sets of text because there is generally more consistency in how positive and negative words are used. Word shift graphs compare weighted averages, not averages of weighted averages, as is often done with short texts, unfortunately.

Lexicons in Shifterator
-----------------------

To make it easier to do sentiment analysis directly within :code:`shifterator`, there are several sentiment lexicons that are wrapped into the package. Any of these lexicons can be loaded by just passing the `name of the lexicon <https://github.com/ryanjgallagher/shifterator/tree/master/shifterator/lexicons>`_ to the shift object.

.. code-block:: python

    sentiment_shift = sh.WeightedAvgShift(type2freq_1=type2freq_1,
                                          type2freq_1=type2freq_2,
                                          type2score_1='labMT_English',
                                          stop_lens=[(4,6)])

.. note::

    Here we have used a :code:`stop_lens` to exclude words with a sentiment score between 4 and 6. For the labMT dictionary, whose scale goes from 1 to 9, this masks emotionally neutral words.

    If you use any of these lexicons, please cite the appropriate sources.

labMT
~~~~~

The labMT lexicon is a general purpose sentiment dictionary assembled by the `Computational Story Lab <https://www.uvm.edu/storylab/>`_ at the University of Vermont. It was constructed by taking the 5,000 most frequently used words from Twitter, New York Times, Google Books, and music lyrics. In total, there are 10,022 words. Words were rated on a continuous scale from 1 to 9 by crowd workers on Amazon Mechanical Turk (MT), where 1 is the least happy and 9 is the most. The labMT dictionary is available in `multiple languages <https://github.com/ryanjgallagher/shifterator/tree/master/shifterator/lexicons/labMT>`_.

For word shift graphs, :code:`shifterator` defaults to a reference value of 5. The labMT dictionary can be called by :code:`labMT_{language}`, where the language is the desired language.

.. code-block:: python

    sentiment_shift = sh.WeightedAvgShift(type2freq_1,
                                          type2freq_2,
                                          'labMT_English')

If you use the English labMT lexicon, cite the following paper:

    Dodds, Peter Sheridan, Kameron Decker Harris, Isabel M. Kloumann, Catherine A. Bliss, and Christopher M. Danforth. "Temporal patterns of happiness and information in a global social network: Hedonometrics and Twitter." *PLoS ONE* 6, no. 12 (2011).

If you use any of the non-English labMT lexicons, cite the following paper:

    Dodds, Peter Sheridan, Eric M. Clark, Suma Desu, Morgan R. Frank, Andrew J. Reagan, Jake Ryland Williams, Lewis Mitchell et al. "Human language reveals a universal positivity bias." *Proceedings of the National Academy of Sciences* 112, no. 8 (2015): 2389-2394.

NRC
~~~

The NRC affect lexicons are assembled and curated by `Saif Mohammad <http://saifmohammad.com/WebPages/lexicons.html>`_ and colleagues at the National Research Council in Canada. Along with the ones included in :code:`shifterator`, they maintain `several other widely used lexicons <http://saifmohammad.com/WebPages/lexicons.html>`_.

Emotion
>>>>>>>

The `NRC emotion intensity <http://saifmohammad.com/WebPages/AffectIntensity.htm lexicon>`_ is a set of affect dictionaries based on Plutchik's theory of emotions, which is based on the eight core emotions: anger, anticipation, disgust, fear, joy, sadness, surprise, and trust. Words were chosen according to those that were already in the `NRC emotion lexicon <http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm>`_ and in the NRC Hashtag Emotion Corpus. There are 1,000-2,000 words per dictionary depending on the emotion and language. Words were rated using best-worst scaling by crowd workers on CrowdFlower. The best-worst ratings were translated to continuous scale from 0 to 1, where 0 indicates the word is lowly associated with the emotion and 1 indicates it is highly associated with the emotion. The NRC emotion intensity dictionaries are available in `multiple languages <https://github.com/ryanjgallagher/shifterator/tree/master/shifterator/lexicons/NRC-emotion>`_.

For word shift graphs, :code:`shifterator` defaults to a reference value of 0.5. The NRC dictionaries can be called by :code:`NRC-emotion_{emotion}_{language}`, where the emotion is one of Plautchik's core emotions and the language is the desired language.

.. code-block:: python

    sentiment_shift = sh.WeightedAvgShift(type2freq_1,
                                          type2freq_2,
                                          'NRC-emotion_anger_Nepali')

If you use any of the NRC emotion intensity dictionaries, please cite the following paper:

    Word Affect Intensities. Saif M. Mohammad. In *Proceedings of the 11th Edition of the Language Resources and Evaluation Conference (LREC-2018)*, May 2018, Miyazaki, Japan.

VAD
>>>

The `NRC valence, arousal, and dominance lexicon <http://saifmohammad.com/WebPages/nrc-vad.html>`_ is a set of affect dictionaries based on the valence, arousal, and dominance theory of affect. Words were chosen according to those that were already in the `NRC emotion lexicon <http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm>`_ and several other sentiment lexicons. In total, there are 20,007 words. Words were rated using best-worst scaling by crowd workers on CrowdFlower. The best-worst ratings were translated to continuous scale from 0 to 1, where 0 indicates the word is lowly associated with the affect dimension and 1 indicates it is highly associated. The NRC valence, arousal, and dominance dictionaries are available in `multiple languages <https://github.com/ryanjgallagher/shifterator/tree/master/shifterator/lexicons/NRC-VAD>`_.

For word shift graphs, :code:`shifterator` defaults to a reference value of 0.5. The NRC dictionaries can be called by :code:`NRC-VAD_{dimension}_{language}`, where the dimension is one of valence, arousal, or dominance, and the language is the desired language.

.. code-block:: python

    sentiment_shift = sh.WeightedAvgShift(type2freq_1,
                                          type2freq_2,
                                          'NRC-VAD_valence_Ukranian')

If you use any of the NRC valence, arousal, and dominance dictionaries, please cite the following paper:

    Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words. Saif M. Mohammad. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics*, Melbourne, Australia, July 2018.


SocialSent
~~~~~~~~~~

The `SocialSent sentiment lexicons <https://nlp.stanford.edu/projects/socialsent>`_ are assembled and curated by the `Stanford NLP Group <https://nlp.stanford.edu/>`.

Historical
>>>>>>>>>>

The `SocialSent historical lexicons <https://nlp.stanford.edu/projects/socialsent>`_ are a set of historical sentiment dictionaries, one for every decade from 1850 to 2000. It was constructed by taking the 5,000 most frequently used non-stop words used in each decade of the Corpus of Historical American English. The words were rated for sentiment using a semi-supervised machine learning approach based on embeddings constructed from the same historical corpus. They were then mapped to a scale with 0 mean and unit variance.

For word shift graphs, :code:`shifterator` defaults to a reference value of 0. The SocialSent historical dictionaries can be called by :code:`SocialSent-historical_{year}`, where the year is the `decade of interest <https://github.com/ryanjgallagher/shifterator/tree/master/shifterator/lexicons/SocialSent-historical>`_.

.. code-block:: python

    sentiment_shift = sh.WeightedAvgShift(type2freq_1,
                                          type2freq_2,
                                          'SocialSent-historical_1920')

If you use any of the SocialSent historical lexicons, please cite the following paper:

    William L. Hamilton, Kevin Clark, Jure Leskovec, and Dan Jurafsky. Inducing Domain-Specific Sentiment Lexicons from Unlabeled Corpora. *Proceedings of EMNLP*, 2016

Reddit
>>>>>>

The `SocialSent subreddit community lexicons <https://nlp.stanford.edu/projects/socialsent>`_ are a set of community-specific sentiment dictionaries for the most popular subreddits on Reddit. It was constructed by taking the 5,000 most frequently used non-stop words used in each of the 250 most popular subreddits in 2014 (according to number of comments). The words were rated for sentiment using a semi-supervised machine learning approach based on embeddings constructed from all public comments on each subreddit in 2014. They were then mapped to a scale with 0 mean and unit variance.

For word shift graphs, :code:`shifterator` defaults to a reference value of 0. The SocialSent historical dictionaries can be called by :code:`SocialSent-Reddit_{subreddit}`, where the subreddit is the `subreddit of interest <https://github.com/ryanjgallagher/shifterator/tree/master/shifterator/lexicons/SocialSent-Reddit>`_.

.. code-block:: python

    sentiment_shift = sh.WeightedAvgShift(type2freq_1,
                                          type2freq_2,
                                          'SocialSent-Reddit_Frozen')

If you use any of the SocialSent subreddit community lexicons, please cite the following paper:

    William L. Hamilton, Kevin Clark, Jure Leskovec, and Dan Jurafsky. Inducing Domain-Specific Sentiment Lexicons from Unlabeled Corpora. *Proceedings of EMNLP*, 2016
