from matplotlib.collections import PatchCollection
import matplotlib
import matplotlib.pyplot as plt
import random

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
plt.xlim([0, 1001])
plt.ylim([0, 1001])
n=10000
patches = []
for i in range(0,n):
    x = random.uniform(1, 1000)
    y = random.uniform(1, 1000)
    patches.append(matplotlib.patches.Rectangle((x, y),1,1,))
ax.add_collection(PatchCollection(patches))
plt.show()