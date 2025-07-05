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

def get_optima_mol(k):
    if k !=12:
        rank_real,rank_normalize=utils.get_rank_score(DFT_df=pd.read_csv(f'./data/result_k{k}_m062x_def2svp.csv'))
        rank_real.to_csv(f'./experiment/rank_absolute_k{k}.csv',index=False)
        rank_normalize.to_csv(f'./experiment/rank_normalized_k{k}.csv',index=False)
    else:
        rank_real,rank_normalize=utils.get_rank_score(DFT_df=pd.read_csv(f'./data/result_all_m062x_def2svp.csv'))
        rank_real.to_csv(f'./outputs/rank_absolute.csv',index=False)
        rank_normalize.to_csv(f'./outputs/rank_normalized.csv',index=False)

def rank_process_al(k):
    def rank_al(al):
            return 1 if 3.7 <= float(al) <= 4.3 else 0
    if k!=12:
        df=pd.read_csv(f'./experiment/rank_absolute_k{k}.csv')
        df['anode_limit']=df['anode_limit'].apply(rank_al)
        df[['precursor_scscore','scscore','anode_limit','element_friendliness','capacity(mAh/g)']] = df[['precursor_scscore','scscore','anode_limit','element_friendliness','capacity(mAh/g)']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        df['total_score']=df['precursor_scscore']+df['scscore']+df['anode_limit']+df['element_friendliness']+df['capacity(mAh/g)']
        df=df.sort_values(by='total_score',ascending=False)
        df.to_csv(f'./experiment/rank_normalized_k{k}.csv',index=False)
    if k==12:
        df=pd.read_csv(f'./outputs/rank_absolute.csv')
        df['anode_limit']=df['anode_limit'].apply(rank_al)
        df[['precursor_scscore','scscore','anode_limit','element_friendliness','capacity(mAh/g)']] = df[['precursor_scscore','scscore','anode_limit','element_friendliness','capacity(mAh/g)']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        df['total_score']=df['precursor_scscore']+df['scscore']+df['anode_limit']+df['element_friendliness']+df['capacity(mAh/g)']
        df=df.sort_values(by='total_score',ascending=False)
        df.to_csv(f'./outputs/rank_normalized.csv',index=False)

if __name__=="__main__":
    k = int(sys.argv[1])
    # print(k,type(k))
    # test=Test()
    # k_partitionline(test=test,partition=9)
    # compare_silhouette_score(test=test)
    # compare_wcss(test)
    # compare_tsne(test)
    # df=pd.read_csv('./data/onlyNLi.csv')
    # df=pd.read_csv('./experiment/t_sne_k_10.csv')
    # print(df[df['id']==11])
    # print(df.loc[342])
    # compare_representative_mols(test=test)
    # compare_heatmap(test)
    get_optima_mol(k)
    rank_process_al(k)



