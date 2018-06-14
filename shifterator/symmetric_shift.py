"""
symmetric_shift.py

Author: Ryan J. Gallagher, Network Science Institute, Northeastern University
Last updated: June 13th, 2018

TODO:
- Define symmetric shift class
- Define divergence shift class
"""
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

import shifterator

# ------------------------------------------------------------------------------
# -------------------------- SYMMETRIC SHIFT CLASSES ---------------------------
# ------------------------------------------------------------------------------
class symmetric_shift(shifterator.shift):
    pass

class divergence_shift(symmetric_shift):
    """
    Extra parameters: type of divergence (?), and alpha of entropy
    """
    pass
