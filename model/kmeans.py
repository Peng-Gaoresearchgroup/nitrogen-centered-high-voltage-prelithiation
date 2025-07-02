from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import extract_xy
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE


class kmeans:
    def __init__(self, n_clusters=3, init='k-means++', max_iter=300, random_state=42):
        """
        - n_clusters: int.
        - init: str, initializs method ('k-means++', 'random').
        - max_iter: int.
        - random_state: int.
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, data):
        """
        - data: DataFrame or ndarray.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.model.fit(data)
        self.cluster_centers_ = self.model.cluster_centers_
        self.labels_ = self.model.labels_
        self.inertia_ = self.model.inertia_
        self.X=data
    def predict(self, data):
        """
        return:
        - labels: ndarray
        """
        if self.model is None:
            raise ValueError("train model first.")
        if isinstance(data, pd.DataFrame):
            data = data.values
        return self.model.predict(data)

    def plot_scatters(self, data,save):
        """
        - data: DataFrame or ndarray
        """
        if self.labels_ is None:
            raise ValueError("train model first.")
        
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        if data.shape[1] != 2:
            raise ValueError("Demension error")

        plt.figure(figsize=(8, 6))
        for i in range(self.n_clusters):
            cluster_points = data[self.labels_ == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}")
        
        plt.scatter(
            self.cluster_centers_[:, 0],
            self.cluster_centers_[:, 1],
            s=200,
            c='red',
            marker='+',
            label='Centroids'
        )
        plt.title("KMeans Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.savefig(save,dpi=300,format='png')
        # plt.show()

    def get_cluster_centers(self):
        """
        - cluster_centers: ndarray.
        """
        if self.cluster_centers_ is None:
            raise ValueError("train model first.")
        return self.cluster_centers_

    def get_plot_information(self, data,save):
        if self.labels_ is None:
            print("Model has not been fitted yet.")
            return None


        data = np.array(data)

        cluster_data = []
        for idx, label in enumerate(self.labels_):
            x,y=extract_xy(str(data[idx]))
            cluster_data.append({ "Molecule": idx,"Cluster": label, "x": x,'y':y})
        info=pd.DataFrame(cluster_data) 
        info.to_csv(save,index=False)
        grouped = info.groupby('Cluster')['Molecule'] \
               .agg(lambda x: ', '.join(x.astype(str))) \
               .sort_index()
        summary_str = '\n'.join(
        [f"Cluster{group}: {samples}" 
         for group, samples in grouped.items()]
        )
        print('kmeans_information:\n',summary_str)
        return info

    def get_labels(self):
        """
        return:
        - labels: ndarray.
        """
        if self.labels_ is None:
            raise ValueError("train model first.")
        return self.labels_

    def get_inertia(self):
        """
        return:
        - inertia: float    
        """
        if self.inertia_ is None:
            raise ValueError("train model first.")
        return self.inertia_
    

    def get_fit_info(self,data,save):
        if self.labels_ is None:
            print("Model has not been fitted yet.")
            return None
        # np.set_printoptions(suppress=True)
        np.set_printoptions(precision=16)
        np.set_printoptions(linewidth=np.inf)
        data = np.array(data)
        cluster_data = []
        for idx, label in enumerate(self.labels_):
            # x,y=extract_xy(str(data[idx]))
            cluster_data.append({ "Molecule": idx,"Cluster": label,'data':data[idx],'Cluster_center':self.cluster_centers_[label],'Distance_to_center':np.linalg.norm(data[idx] - self.cluster_centers_[label])})
        info=pd.DataFrame(cluster_data)
        info.to_csv(save,index=False)
        return info
    
    def get_silhouette_scores(self):
        return silhouette_score(self.X, self.labels_)
    
    def get_wcss(self):
        if self.inertia_ ==None:
            print('Model not fit yet')
        else:
            return self.inertia_
        
    def get_tss(self,data):
        mean = np.mean(data, axis=0)
        tss = np.sum(np.linalg.norm(data - mean, axis=1) ** 2)
        return tss
    
    def get_t_sne(self,data, sample_ids=None, perplexity=30, random_state=0):

        if self.labels_ is None:
            raise ValueError("Fit first!")

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        tsne_result = tsne.fit_transform(data)

        n_samples = data.shape[0]
        if sample_ids is None:
            sample_ids = list(range(n_samples))

        df = pd.DataFrame({
            'id': sample_ids,
            'cluster': self.labels_,
            'X': tsne_result[:, 0],
            'Y': tsne_result[:, 1]
        })
        df = df.sort_values(by='cluster')
        return df
    
    def get_heatmap(self,data):
        n_clusters = len(np.unique(self.labels_))
        cluster_means = np.zeros((n_clusters, data.shape[1]))
        
        for i in range(n_clusters):
            cluster_data = data[self.labels_ == i]
            cluster_means[i] = np.mean(cluster_data, axis=0)

        df = pd.DataFrame(
            cluster_means,
            index=[f"Cluster {i}" for i in range(n_clusters)],
            columns=[f"PCA feature {j}" for j in range(data.shape[1])]
        )
        return df

