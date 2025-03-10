# Nitrogen Centered High Voltage Prelithiation

### Introduction
Scripts from the manuscript Z. Kang, et al.

### Contents
The project is as follows：
```
├── Readme.md
├── main.py 
├── utils.py
├── conf/
│   └── conf.yaml    # global configuration information, such as file storage paths and model hyperparameters
├── data/
│   ├── raw_data.csv    # molecule library download from PubMed
│   ├── onlyNH.csv    # screen the molecules in raw_data.csv to ensure that the N atoms are all secondary amine structures
│   ├── only_NLi.csv    # the lithiation structure corresponding to the molecules in onlyNH.csv, is the input samples for hierarchical clustering and KMeans clustering
│   └── DFT_result.csv    # After clustering is complete, DFT calculations are performed on the molecules of interest, and the results are used for the next molecular scoring
├── model/
│   ├── hc.py    # hierarchical clustering
│   ├── kmeans.py    # kmeans clustering
│   └── pca.py    # principal component analysis
├── outputs/    # the following file is generated after running main.py
│   ├── confusion_matrix.csv    # confusion matrix comparing similarity of hierarchical and kmeans clustering results
│   ├── data_descriptors.csv    # results for descriptor of molecules in /data/olyNLi.csv
│   ├── data_normalized_descriptors.csv    # normalized result of data_descriptors.csv
│   ├── hc_dendro.png    # hierarchical clustering dendrogram, matplotlib
│   ├── hc_information.csv    # hierarchical clustering grouping information
│   ├── kmeans_draw.csv    # the coordinates in kmeans_plot_information.csv have been organized to make it easier for Origin.exe
│   ├── kmeans_fit_information.csv    # recording high-dimensional data for KMeans clustering
│   ├── kmeans_plot_information.csv    # coordinates in kmeans_scatter.png
│   ├── kmeans_representative_molecules.csv    # molecule closest to the center of the cluster
│   ├── kmeans_scatter.png    # KMeans clustering scatter plot, reduced to 2D with PCA, matplotlib
│   ├── pca_features.csv    # PCA downscaling of ./outputs/data_normalized_descriptors.csv for clustering
│   ├── rank_normalized.csv    # normalized rank of rank_real_value.csv, you can know the final rank
│   └── rank_real_value.csv    # five-dimension data of recommended molecules
└── requirements.txt
```

### System requirements
In order to run source code file in the Data folder, the following requirements need to be met:
- Windows, Mac, Linux
- Python and the required modules. See the [Instructions for use](#Instructions-for-use) for versions.

### Installation
You can download the package in zip format directly from this github site,or use git in the terminal：
```
git clone https://github.com/Peng-Gaoresearchgroup/nitrogen-centered-high-voltage-prelithiation.git
```

### Instructions for use
- Environment
```
# create environment, conda is recommended
conda create -n yourname -c conda-forge rdkit=2024.9.4 python=3.11.8

# install python modules
pip install -r ./requirments.txt

# switch to it
conda activate yourname
```
- Quick test
```
python ./main.py
```
- Reproduce the paper

Firstly, this project will pause to perform DFT calculations based on the results of the clustering, which will be used as input for the next ranking section. If you want to reproduce this project from scratch, follow the steps below.

1. Download original data from PubMed. See the paper for details. Rename it to "raw_data.csv", put it into ./data/. The format references existing [raw_data.csv](./data/raw_data.csv)
2. In [conf](./conf/conf.yaml), modify these key and values:
    ```
    data_washed : False
    features_calculated : False
    DFT_done : False
    ``` 
    run [main.py](./main.py), data will be clean and clustering will be performed.
3. Analyze clusterign results(see article), get recommended molecules, and their DFT calculated value, name it to "DFT_result.csv", put it into ./data/, make sure the column names match [existing files](./data/DFT_result.csv).
4. In [conf](./conf/conf.yaml), modify these key and values:
    ```
    data_washed : True
    features_calculated : True
    DFT_done : True
    ``` 
    run [main.py](./main.py) again, and the program will score the recommended molecules, generating [rank_real_value.csv](./outputs/rank_real_value.csv) and [rank_normalized.csv](./outputs/rank_normalized.csv).


### Contributions
Y. Gao, Z. Kang and G. Wu developed a workflow. G. Wu wrote the program.

### License
This project uses the [MIT LICENSE](LICENSE).

### Disclaimer
This code is intended for educational and research purposes only. Please ensure that you comply with relevant laws and regulations as well as the terms of service of the target website when using this code. The author is not responsible for any legal liabilities or other issues arising from the use of this code.

### Contact
If you have any questions, you can contact us at: yuegao@fudan.edu.cn
