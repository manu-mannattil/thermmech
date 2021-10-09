# -*- coding: utf-8 -*-
#
#   index     fold
#   -----     ----
#     0        3-5
#     1        4-5
#     2        1-5
#     3        6-5
#     4        1-6
#     5        2-6
#     6        3-6
#

import sys
from utils import *
import numpy as np
import matplotlib.pyplot as plt

# Number of bins for the histogram.
# Choose an odd number of bins so that 0 falls at bins/2.
bins = 181

cv = np.linspace(-180, 180, bins)
free = np.zeros(bins)

for name in sys.argv[1:]:
    a = np.loadtxt(name)[:, 3]
    free += pdf(a, cv)
    print("Processed {}".format(name), file=sys.stderr)

free = -np.log(free / free[-1])
np.savetxt("free_numer.dat", np.array([cv, free]).T)
