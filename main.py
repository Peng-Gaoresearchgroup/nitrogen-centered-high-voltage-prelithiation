import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky")
import yaml
from utils import SafeDict,smiles2descirptors,get_valid_smiles,only_NH,NH2NLi,get_representative_mol,get_sctter_info_for_origin,get_rank_score,get_confusion_matrix
from model import hc, kmeans,pca
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#-----------------------------------------------------
# Load conf
#-----------------------------------------------------
with open("./conf/conf.yaml") as f:
    conf = SafeDict(yaml.safe_load(f))

#-----------------------------------------------------
# data wash
#-----------------------------------------------------
if conf.data_washed == False:
    df=pd.read_csv('./data/raw_data.csv')
    valid_df=get_valid_smiles(df)
    onlyNH=pd.DataFrame({'canonicalsmiles':[i for i in valid_df['canonicalsmiles'] if only_NH(i)]})
    onlyNH.to_csv('./data/onlyNH.csv',index=False)
    onlyNLi=pd.DataFrame({'canonicalsmiles':[NH2NLi(i) for i in onlyNH['canonicalsmiles']]})
    onlyNLi.to_csv('./data/onlyNLi.csv',index=False)
    
#-----------------------------------------------------
# Get original desritors
#-----------------------------------------------------
if conf.features_calculated == False:
    # print('Start')
    df=pd.read_csv('./data/onlyNLi.csv')
    descriptors= df['canonicalsmiles'].apply(lambda x: smiles2descirptors(smiles=x,desdic=conf.hc,seed=conf.seed))
    descriptors.dropna(how='any', inplace=True)
    descriptors=pd.DataFrame(descriptors)
    descriptors.to_csv('./outputs/data_descriptors.csv',index=False)
else:
    df=pd.read_csv('./data/onlyNLi.csv')
    descriptors=pd.read_csv('./outputs/data_descriptors.csv')
    
#-----------------------------------------------------
# Get nromalized desritors
#-----------------------------------------------------
scaler = MinMaxScaler()
normal_des = pd.DataFrame(scaler.fit_transform(descriptors), columns=descriptors.columns)
normal_des.to_csv('./outputs/data_normalized_descriptors.csv',index=False)

#-----------------------------------------------------
# PCA dimensionality reduction for descritors
#-----------------------------------------------------
pca_ = pca.PCA(n_components=6)
pca2_ =pca.PCA(n_components=2)
features = pca_.fit_transform(X=normal_des)

features.to_csv('./outputs/pca_features.csv',index=False)
features_2d=pca2_.fit_transform(X=normal_des)

#-----------------------------------------------------
# Clustering
#-----------------------------------------------------
hc_=hc.hierarchical_clustering(distance=conf.hc.distance,partition_line=conf.hc.partition_line)
hc_._fit(X=features)
hc_.plot_dendrogram(
    dpi=conf.hc.dendro.dpi,
    figsize=tuple((conf.hc.dendro.fig_size_x,conf.hc.dendro.fig_size_y)),
    fontname=conf.hc.dendro.font,
    fontsize=conf.hc.dendro.font_size,
    treelw=conf.hc.dendro.treelw,
    borderlw=conf.hc.dendro.borderlw,
    save=conf.hc.dendro.save)
mapping = hc_.group_mapping(save=conf.hc.save_infor)

k=kmeans.kmeans(n_clusters=conf.kmeans.n_cluster,max_iter=conf.kmeans.max_iter)
k.fit(data=features)
fit_info=k.get_fit_info(data=features,save=conf.kmeans.fit_info_save)
represent=get_representative_mol(fit_info,save=conf.kmeans.represent_save)
k.plot_scatters(data=features_2d,save=conf.kmeans.plot_save)
info=k.get_plot_information(data=features_2d,save=conf.kmeans.plot_info_save)
reslut=get_sctter_info_for_origin(info)
reslut.to_csv('./outputs/kmeans_draw.csv',index=False)

#-----------------------------------------------------
# Gat confusion matrix and Rand index
#-----------------------------------------------------
cm=get_confusion_matrix()
cm.to_csv('./outputs/confusion_matrix.csv',index=False)

# -----------------------------------------------------
# Rank
# -----------------------------------------------------
if conf.DFT_done == True:
    rank_real,rank_normalize=get_rank_score(DFT_df=pd.read_csv('./data/DFT_result.csv'))
    rank_real.to_csv('./outputs/rank_real_value.csv',index=False)
    rank_normalize.to_csv('./outputs/rank_normalized.csv',index=False)