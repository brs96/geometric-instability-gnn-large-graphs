# Geometric instability of graph neural networks on large graphs

To train NC models:
* cd into `\node-classification` and run `python3 nc_train.py --dataset=$DATASET --model=$MODEL`

To train LP models:
* cd into `\link-prediction` and run `python3 lp_train.py --dataset=$DATASET --model=$MODEL`

To compute metrics:
* cd into either `\node-classification` or `\link-prediction` and run `python3 ../metrics.py --dataset=$DATASET --model=$MODEL --metric=$METRIC`

To draw box plots:
* cd into either `\node-classification` or `\link-prediction` and run `python3 ../plots.py --dataset=$DATASET -metric=$METRIC`

Model can be GraphSAGE, GCN, GATv1(NC only), GIN(NC only) and dataset can be Cora(NC), ogbn-arxiv(NC), ogbl-citation2(LP)
Metric can be jaccard, 2ndcos, gram


# Environment spec
Package            Version
------------------ --------
attrs              22.1.0
certifi            2023.5.7
charset-normalizer 3.0.1
colorama           0.4.6
contourpy          1.0.7
cycler             0.11.0
exceptiongroup     1.0.4
filelock           3.12.2
fonttools          4.38.0
idna               3.4
iniconfig          1.1.1
Jinja2             3.1.2
joblib             1.2.0
kiwisolver         1.4.4
littleutils        0.2.2
MarkupSafe         2.1.2
matplotlib         3.7.0
mpmath             1.3.0
mycolorpy          1.5.1
networkx           3.0
numpy              1.24.2
ogb                1.3.6
outdated           0.2.2
packaging          23.0
pandas             1.5.3
Pillow             9.4.0
pip                22.3.1
pluggy             1.0.0
psutil             5.9.4
pyparsing          3.0.9
pytest             7.3.1
python-dateutil    2.8.2
pytz               2022.7.1
requests           2.28.2
scikit-learn       1.2.1
scipy              1.10.1
seaborn            0.12.2
setuptools         65.6.3
six                1.16.0
sympy              1.12
threadpoolctl      3.1.0
tomli              2.0.1
torch              1.13.0
torch-cluster      1.6.0
torch-geometric    2.2.0
torch-scatter      2.1.0
torch-sparse       0.6.16
tqdm               4.64.1
typing_extensions  4.5.0
urllib3            1.26.14
wheel              0.38.4
