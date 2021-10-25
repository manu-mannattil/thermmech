import sys
from utils import *
import numpy as np

# Number of bins for the histogram.
# Choose an odd number of bins so that 0 falls at bins/2.
bins = 271

cv = np.linspace(-np.pi, np.pi, bins)

data = np.loadtxt("data/4bar_met_2.dat")[:, 0]
free = pdf(data, cv)

free = -np.log(free / free[int(bins/2)])
np.savetxt("data/4bar_free_2.dat", np.array([cv, free]).T)
