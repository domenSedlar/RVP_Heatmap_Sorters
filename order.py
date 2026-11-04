
from Heuristics.hclust import Hclust, hclust_with_olo_return_ordering
from Heuristics.bae import BAE
from Heuristics.rand_swap import rand_swaps, rand_block_swaps, randomly_mirror, RandomSorter
from Heuristics.tsp_heur import TSP_LK

from Heuristics.real_bae import real_BAE as BEA
from matplotlib import pyplot as plt
import numpy as np
import random

with open("Data/I2000/data.txt") as f:
    content = f.read()

with open("Data/I2000/poz_neg_lables.txt") as f:
    ys = f.read()

ys = ys.strip().split('\n')
ys = list(map(float, ys))
y = np.array([[1 if x < 0 else 0 for x in ys]])
raw_rows = content.strip().split('\n\n')

data = []
for row in raw_rows:
    if row.strip():
        values = list(map(float, row.split()))
        data.append(values)
        
data = np.array(data)

H = data.copy()
bea = BEA()
rows, cols = hclust_with_olo_return_ordering(H)

c = list(range(len(ys)))
random.shuffle(c)
c = np.array(c)
print(c)

plt.subplot(221)
plt.imshow(data[:,c])
plt.subplot(222)
plt.imshow(H[rows][:, cols])
plt.subplot(223)
plt.imshow(y[:, c])
plt.subplot(224)
plt.imshow(y[:, cols])
plt.show()