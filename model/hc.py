import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram,fcluster,cut_tree
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from collections import defaultdict

class hierarchical_clustering:
    def __init__(self, method="ward",distance=0.5,partition_line=0.2):
        """
        Initialize the hierarchical clustering model.
        :param method: "single", "complete", "average", "ward", 等
        """
        self.method = method 
        self.linkage_matrix = None 
        self.distance=distance
        self.model=None
        self.partition_line=partition_line
        
    def _fit(self, X):

        if isinstance(X, pd.DataFrame):
            X = X.values
        self.p=len(X)

        self.model = AgglomerativeClustering(
            distance_threshold=self.distance,
            n_clusters=None
        )
        self.model.fit(X)

        self._build_valid_linkage() 
    
    def plot_dendrogram(self, figsize=(10, 7), dpi=400, treelw=0.5, borderlw=0.25, fontname="Arial", fontsize=10,save='./output/hc_dendro.png'):
        if self.linkage_matrix is None:
            raise ValueError("Model has not been fitted. Please call 'fit' before plotting.")
        plt.figure(figsize=figsize)
        
        hierarchy.set_link_color_palette(['#DC143C', '#FF8C00', '#2E8B57', '#1A82E6', '#9400D3'])
        
        dendro_info =dendrogram(
            self.linkage_matrix,
            no_plot=True, p=self.p,
            truncate_mode='lastp', show_contracted=True,color_threshold=self.partition_line, above_threshold_color='grey')
        
        for x, y, color in zip(dendro_info['icoord'], dendro_info['dcoord'], dendro_info['color_list']):
            plt.plot(y, x, lw=treelw,color=color,alpha=1,antialiased=False)
            plt.xlim(0, self.linkage_matrix[:, 2].max() * 1.05)
            plt.ylim(0, 10 * len(dendro_info['ivl']))
        plt.yticks(ticks=[5 + 10 * i for i in range(len(dendro_info['ivl']))],
               labels=[int(node) for node in dendro_info['ivl']],
               fontsize=fontsize)
            
        plt.ylabel("Smaples",fontname=fontname, fontsize=fontsize,weight='bold')
        plt.xlabel("Distance",fontname=fontname, fontsize=fontsize,weight='bold')
        ax = plt.gca()
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_label_position('left')
        ax.tick_params(axis='y', labelsize=fontsize,width=borderlw)
        ax.tick_params(axis='x', labelsize=fontsize,width=borderlw)
        
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_linewidth(borderlw)
        plt.axvline(x=self.partition_line, color='#000000', linewidth=treelw,antialiased=False)
        plt.savefig(save,format='png',dpi=dpi,bbox_inches='tight',transparent=False)
        # plt.show()

    def group_mapping(self, save='./outputs/hc_infor.csv'):
        if self.linkage_matrix is None:
            raise RuntimeError("fit before group_mapping")
        if not hasattr(self, 'partition_line'):
            raise ValueError("set partition_line before group_mapping")
        labels = cut_tree(
            self.linkage_matrix, 
            height=self.partition_line
        ).flatten()
        
        unique_labels = np.unique(labels, return_inverse=True)[1]
        mapping = pd.DataFrame({"Molecule": range(len(unique_labels)), "Cluster": unique_labels})
        mapping.to_csv(save, index=False)
        grouped = mapping.groupby('Cluster')['Molecule'] \
               .agg(lambda x: ', '.join(x.astype(str))) \
               .sort_index()
    
        summary_str = '\n'.join(
        [f"Cluster{group}: {samples}" 
         for group, samples in grouped.items()]
        )
        print('hc_information:\n',summary_str)
        return mapping
    
    
    def _build_valid_linkage(self):
        children = self.model.children_
        distances = self.model.distances_
        n_leaves = self.model.n_leaves_  
        
        node_counts = np.zeros(children.shape[0] + n_leaves, dtype=int)
        node_counts[:n_leaves] = 1  
        
        for i, (left, right) in enumerate(children):
            current_node = n_leaves + i
            node_counts[current_node] = node_counts[left] + node_counts[right]
        
        self.linkage_matrix = np.column_stack([
            children,
            distances,
            node_counts[n_leaves:] 
        ]).astype(float)
