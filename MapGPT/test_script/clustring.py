from read_datasets import load_result

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from dtw import dtw


class Clastring:
    def __init__(self, data):
        self.data = data
        self.distance_matrix = None
        self.silhouette_scores = []
        self.best_n_clusters = None
        self.best_cluster = None

    # 计算 DTW 距离矩阵
    def compute_dtw_distance_matrix(self):
        n = len(self.data)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                distance = dtw(
                    self.data[i],
                    self.data[j],
                    dist_method=lambda x, y: np.linalg.norm(x - y),
                ).distance
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        self.distance_matrix = distance_matrix
        return distance_matrix

    def find_best_clusters(self):
        # 尝试不同的聚类数量，计算轮廓系数
        best_score = -1
        best_n_clusters = 2  # 至少需要 2 个聚类
        best_cluster = None
        # for n_clusters in range(
        #     2, min(100, len(self.data))
        # ):  # 尝试 2 到 min(10, 数据量) 个聚类
        for n_clusters in range(4, 5):
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, metric="precomputed", linkage="average"
            )
            labels = clustering.fit_predict(self.distance_matrix)
            score = silhouette_score(self.distance_matrix, labels, metric="precomputed")
            self.silhouette_scores.append(score)
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_cluster = labels
        self.best_n_clusters, self.best_cluster = best_n_clusters, best_cluster
        return best_n_clusters, best_cluster

    def run(self):
        self.compute_dtw_distance_matrix()
        best_n_clusters, best_cluster = self.find_best_clusters()
        return best_n_clusters, best_cluster



def test_clustering(data):
    data_trajs = []
    for i, traj in enumerate(data):
        uncertainty = traj["uncertainty"]
        data_traj = []
        for k, v in uncertainty.items():
            data_traj.append([v])
        data_traj = np.array(data_traj)
        data_trajs.append(data_traj)
    # data_trajs = np.array(data_trajs)
    clastring = Clastring(data_trajs)
    clastring.run()
    # 输出结果
    print(
        "Silhouette scores for different numbers of clusters:",
        clastring.silhouette_scores,
    )
    print("Best number of clusters:", clastring.best_n_clusters)
    print("Cluster labels for each data point:", clastring.best_cluster)


if __name__ == "__main__":
    dir_name = "./datasets/exprs_map/test/preds"
    result = load_result(dir_name)
    test_clustering(result)