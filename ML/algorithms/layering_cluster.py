import numpy as np
from scipy.cluster.hierarchy import linkage,dendrogram
import matplotlib.pylab as plt

data=[[1,3],[2,1],[1,2],[4,4],[4,6],[6,7]]
dara=np.array(data)
Z=linkage(data,method='single')
dendrogram(Z)
plt.show()
