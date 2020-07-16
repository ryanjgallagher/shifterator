Getting Started
===============

Introduction
------------

Word shift graphs are interpretable horizontal bar charts for visualizing how any two texts compare according to a given measure. They can be used any time a measure can be written as a sum of individual word contributions.

All of the shifts implemented in :code:`shifterator` can be easily loaded into Python.

.. code-block:: python

    import shifterator as sh

We assume that :code:`shifterator` has been imported in this way for the rest of the tutorial.

Case Study
----------

Throughout this tutorial, we compare the speeches of two U.S. presidents, Lyndon B. Johnson (1963-1969) and George W. Bush (2001-2009). The speech transcripts are from the University of Virgina's `Miller Center <https://millercenter.org/the-presidency/presidential-speeches>`_.

Necessary Data Structures
-------------------------
We load the parsed text into two dictionaries: :code:`type2freq_1` (for Lyndon B. Johnson) and :code:`type2freq_2` (for George W. Bush). These are dictionaries where keys are word types and valules are their frequencies in each text. For many word shifts, this is the only input that is required.
