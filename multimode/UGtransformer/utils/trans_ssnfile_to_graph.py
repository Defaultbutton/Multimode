import numpy as np
from UGtransformer.utils.labels_dict import *
import networkx as nx
from torch_geometric.data import Data
import os


file_path = "../../SSN-master/target/brca/"  #"ssn_TCGA-A8-A075-01A-11R-A084-07.txt"
filenames = os.listdir(file_path)

g_list = []
label_dict = {}
feat_dict = []


for filename in filenames:
    df = pd.read_csv(file_path + filename, sep="\t")

    nodes = list(set(df['Gene1'].tolist() + df['Gene2'].tolist()))
    num_nodes = len(nodes)

    sample = pd.read_csv(
        file_path,
        sep="\t",
        usecols=[0, 1]
    )

    G = nx.from_pandas_edgelist(sample, "Gene1", "Gene2")
    g_list.append(G)

    # 节点特征矩阵[num_nodes, num_node_features]
    normal = pd.read_csv(
        '../../DataProcess/normal_matrix.csv',
        header=0
    )
    tumor = pd.read_csv(
        '../../DataProcess/tumor_matrix.csv',
        header=0
    )

    # 构建特征矩阵feature_matrix
    feature_matrix = torch.zeros(G.number_of_nodes(), 1, dtype=torch.float)
    node_list = list(G.nodes())
    sample_id = file_path.split("_")[1].split(".txt")[0]
    # 节点特征 = 样本counts-mean(正常样本counts)
    for node in node_list:
        normal_rows = normal.loc[normal.iloc[:, 0] == node]
        sample_count = tumor.loc[tumor.iloc[:, 0] == node, tumor.columns == sample_id].values[0, 0]
        normal_count_mean = normal_rows.iloc[:, 1:].mean(axis=1).values[0]
        feature_matrix[node_list.index(node), 0] = sample_count - normal_count_mean

        feat_dict.append(feature_matrix)


    # 转换为pyg的图
    x = torch.eye(G.number_of_nodes(), dtype=torch.float)

    adj = nx.to_scipy_sparse_matrix(G).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)

    label_dict = get_lables_dict("brca")

    y = torch.tensor([label_mapping(label_dict[sample_id])])

    data = Data(x=x, edge_index=edge_index, y=y)



