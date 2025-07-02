import sys,os
import pandas as pd
from model import hc,kmeans
import utils,yaml

class Test():
    def __init__(self):
        with open("./conf/conf.yaml") as f:
            conf = yaml.safe_load(f)
        self.conf=conf
        self.data=pd.read_csv('./outputs/pca_features.csv')
    def get_hc_model(self):
        h=hc.hierarchical_clustering(distance=self.conf['hc']['distance'],partition_line=self.conf['hc']['partition_line'])
        self.hc_model=h
    def get_kmeans_model(self):
        km=kmeans.kmeans(n_clusters=self.conf['kmeans']['n_cluster'],max_iter=self.conf['kmeans']['max_iter'])
        self.kmeans_model=km


def compare_silhouette_score(test):
    test.get_kmeans_model()
    for i in range(5,13):
        test.kmeans_model.n_clusters=i
        test.kmeans_model.fit(data=test.data)
        # test1.kmeans_model.get_fit_info(data=test1.data,save='./experiment/1.csv')
        print(i,',',test.kmeans_model.get_silhouette_scores())

def k_partitionline(test,partition):
    test.get_hc_model()
    test.hc_model._fit(X=test.data)
    test.hc_model.partition_line=partition
    test.hc_model.group_mapping(save=False)


def compare_wcss(test):
    test.get_kmeans_model()
    for i in range(1,50):
        test.kmeans_model.n_clusters=i
        test.kmeans_model.fit(data=test.data)
        print(i,',',test.kmeans_model.get_tss(data=test.data))
    
def compare_tsne(test):
    test.get_kmeans_model()
    for i in [8,10,12]:
        test.kmeans_model.n_clusters=i
        test.kmeans_model.fit(data=test.data)
        df=test.kmeans_model.get_t_sne(data=test.data)
        df.to_csv(f'./experiment/t_sne_k_{i}.csv',index=False)

def compare_representative_mols(test):
    test.get_kmeans_model()
    for i in [8,10,12]:
        test.kmeans_model.n_clusters=i
        test.kmeans_model.fit(data=test.data)

        fit_info=test.kmeans_model.get_fit_info(data=test.data,save=f'./experiment/kmeans_info_k_{i}.csv')
        represent=utils.get_representative_mol(fit_info,save=f'./experiment/kmeans_repre_k_{i}.csv')
def compare_heatmap(test):
    test.get_kmeans_model()
    for i in [8,10,12]:
        test.kmeans_model.n_clusters=i
        test.kmeans_model.fit(data=test.data)
        df=test.kmeans_model.get_heatmap(data=test.data)
        df.to_csv(f'./experiment/heatmap_k_{i}.csv',index=False)

if __name__=="__main__":
    test=Test()
    # k_partitionline(test=test,partition=9)
    # compare_silhouette_score(test=test)
    # compare_wcss(test)
    # compare_tsne(test)
    # df=pd.read_csv('./data/onlyNLi.csv')
    # df=pd.read_csv('./experiment/t_sne_k_10.csv')
    # print(df[df['id']==11])
    # print(df.loc[342])
    # compare_representative_mols(test=test)
    compare_heatmap(test)



