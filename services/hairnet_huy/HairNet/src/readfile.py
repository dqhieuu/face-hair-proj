# Read and print out numpy file:
import sys

import numpy as np

convdata = np.load("../data/strands00009_00026_00000_v0.vismap")
print(convdata.shape)
np.set_printoptions(threshold=sys.maxsize)
print(convdata)