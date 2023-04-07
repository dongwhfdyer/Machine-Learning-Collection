# K值的确定
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文注释
plt.rcParams['axes.unicode_minus'] = False  # 显示正负号

cluster1 = np.random.uniform(0.5, 1.5, (2, 5))
cluster2 = np.random.uniform(3.5, 4.5, (2, 5))
X = np.hstack((cluster1, cluster2)).T
# print(X)
K = range(1, 6)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    # print(kmeans)
    kmeans.fit(X)
    # print(X.shape[0])
    # 找中"心位置，找出每个点到中心点的最小距离，求和，在求平均
    ###euclidean,欧式距离，固定的写法
    meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    print("#" * 40 + "第" + str(k) + "次测试")
    print(cdist(X, kmeans.cluster_centers_, 'euclidean'))
print("#" * 40 + "最小值")
print(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1))
print(meandistortions)
plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
# plt.ylabel('平均畸变程度',fontproperties=font)
plt.ylabel('Ave Distor')
# plt.title('用肘部法则来确定最佳的K值',fontproperties=font);
plt.title('Elbow method value K')
plt.scatter(K, meandistortions)
plt.show()
