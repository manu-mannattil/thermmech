# -*- coding: utf-8 -*-
"""Free-energy landscape of the singular trimer."""

import numpy as np
import matplotlib.pyplot as plt
from utils import *

# Number of bins for the histogram.
# Choose an odd number of bins so that 0 falls at bins/2.
bins = 51

cv = np.linspace(-np.pi, np.pi, bins)

data = np.loadtxt("data/trimer_met.dat")
free = pdf(data, cv)

free = -np.log(free / free[int(bins/2)])
np.savetxt("data/trimer_free.dat", np.array([cv, free]).T)
