import config
import pickle
import random
from sklearn import metrics
from scipy.sparse import csr_matrix, find
from config import *
from pyrwr.rwr import RWR
import multiprocessing as mp
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork
from torch_geometric.datasets import Coauthor, Amazon, CitationFull, LINKXDataset
import torch_geometric.transforms as T
import torch
import numpy as np
from torch_geometric.utils import  from_scipy_sparse_matrix
import scipy.io as sio
def download(data_name):
    if data_name in ["Pubmed", "Cora", "Citeseer"]:
        datasets = Planetoid(root='./datasets/', name=data_name, transform=T.NormalizeFeatures())
    elif data_name in ["computers", "photo"]:
        datasets = Amazon(root='./datasets/', name=data_name, transform=T.NormalizeFeatures())
    elif data_name in ["CS", "Physics"]:
        datasets = Coauthor(root='./datasets/', name=data_name, transform=T.NormalizeFeatures())
    elif data_name in ["Actor"]:
        datasets = Actor(root='./datasets/', transform=T.NormalizeFeatures())
    elif data_name in ["Cornell", "Texas", "Wisconsin"]:
        datasets = WebKB(root='./datasets/', name=data_name)
    elif data_name in ["chameleon", "squirrel"]:
        datasets = WikipediaNetwork(
            root='./datasets/', name=data_name, geom_gcn_preprocess=True)
    elif data_name in ["DBLP"]:
        datasets = CitationFull(root='./datasets/', name=data_name)
    elif data_name in ["penn94", "cornell5"]:
        datasets = LINKXDataset(root='./datasets/', name=data_name)
    dataset = datasets[0]
    node_num = dataset.num_nodes
    if data_name in ["squirrel", "chameleon"]:
        preProcDs = WikipediaNetwork(
            root='../data/', name=data_name, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
        edge_index = np.array(preProcDs[0].edge_index)
    else:
        edge_index = np.array(dataset.edge_index)
    return edge_index, node_num

def load_mat_data(data_name):
    # read .mat format files
    data_dir = "./datasets/" + data_name + '/' + data_name + '.mat'
    print('Load data from: ' + data_dir)
    import scipy.io as sio
    net = sio.loadmat(data_dir)
    edge_index, _ = from_scipy_sparse_matrix(net['net'])
    node_num = torch.max(edge_index) + 1
    edge_index = np.array(edge_index)
    return edge_index, node_num

def split_data(edge_index, node_num, data_file, directed=True, weighted=False):
    edge_weight = [1 for i in range(edge_index.shape[1])]
    A = csr_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(node_num, node_num))
    if directed:
        A = (A + A.T)
    A = A.A
    A = np.triu(A, 1)
    edge_list = []
    I, J, K = find(A)
    for i in range(len(I)):
        edge_list.append((I[i], J[i]))
    random.shuffle(edge_list)
    edge_num = len(edge_list)
    train_index = int(edge_num * config.training_ratio)
    valid_index = int(edge_num * (config.training_ratio + config.valid_ratio))
    train_edge_list = edge_list[0:train_index]
    valid_edge_list = edge_list[train_index: valid_index]
    test_edge_list = edge_list[valid_index:]
    valid_edge_num = len(valid_edge_list)
    test_edge_num = len(test_edge_list)
    print(len(train_edge_list), valid_edge_num, test_edge_num)
    print("Sample Negative edges...")
    neg_candidate_set = set([(random.randint(0, node_num - 1), random.randint(0, node_num - 1)) for i in range(5 * node_num)])

    neg_set = neg_candidate_set - set(edge_list)
    neg_valid_edges = list(neg_set)[0:valid_edge_num]
    neg_test_edges = list(neg_set)[valid_edge_num: valid_edge_num + test_edge_num]
    data = [node_num, train_edge_list, valid_edge_list, neg_valid_edges, test_edge_list, neg_test_edges]
    data_file = open(data_file, 'wb')
    pickle.dump(data, data_file)
    data_file.close()

def preprocess(data_file):
    data_file = open(data_file, 'rb')
    data = pickle.load(data_file)
    data_file.close()
    train_edge_list = data[1]
    node_num = data[0]
    new_I = []
    new_J = []
    for i in range(len(train_edge_list)):
        new_I.append(train_edge_list[i][0])
        new_J.append(train_edge_list[i][1])
    weight = [1 for i in range(len(new_I))]
    A = csr_matrix((weight, (new_I, new_J)), shape=(node_num, node_num))
    A = (A + A.T)
    return A

def cal_rwr(A):
    rwr = RWR()
    rwr.read_csr_matrix(A)
    rwr_graph_tuple = []
    print("Start calculating....")
    for j in range(A.shape[0]):
        rwr_scores = list(rwr.compute(j))[0]
        rwr_tuple = []
        for k in range(len(rwr_scores)):
            rwr_tuple.append((j, k, rwr_scores[k]))
        rwr_tuple = sorted(rwr_tuple, key=lambda k: k[2], reverse=True)
        rwr_graph_tuple.append(rwr_tuple)
    print("Finish calculating....")
    return rwr_graph_tuple

def within_layer_NE_sampling(tuple_list, threshold, neg_num = 40):
    index_num = 1
    total_nodes_num = len(tuple_list) - 1
    for i in range(1, total_nodes_num):
        if tuple_list[i][2] < threshold:
            index_num = i
            break
    pos_samples = tuple_list[1:index_num]
    sample_tuples = []
    for sample in pos_samples:
        neg_rand_int = np.random.randint(index_num, total_nodes_num, size=neg_num)
        for neg_sample_index in neg_rand_int:
            sample_tuples.append([sample, tuple_list[neg_sample_index]])
    return sample_tuples

def within_layer_Ranking_sampling(tuple_list, pos_num=3):
    sample_tuples = []
    for i in range(1, pos_num):
        for j in range(i+1, pos_num):
            sample_tuples.append([tuple_list[i], tuple_list[j]])
    return sample_tuples

def pool_within_layer_Ranking_sampling(graph_tuple, pool_num=5):
    pool = mp.Pool(pool_num)
    sample_tuples = pool.map(within_layer_Ranking_sampling, graph_tuple)
    pool.close()
    pool.join()
    return sample_tuples

###Below is original version for most datasets.
def pool_within_layer_NE_sampling(graph_tuple, threshold, pool_num=5):
    pool = mp.Pool(pool_num)
    threshold_l = [threshold for i in range(len(graph_tuple))]
    zip_args = list(zip(graph_tuple, threshold_l))
    sample_tuples = pool.starmap(within_layer_NE_sampling, zip_args)
    #sample_tuples = pool.map(within_layer_NE_sampling, graph_tuple, threshold)
    pool.close()
    pool.join()
    return sample_tuples

###Maybe used for baselines.
def eval_link_prediction(emd, pos_edge, neg_edge, threshold=0, bl_flag=True):
    if bl_flag:
        norm_value = np.linalg.norm(emd, axis=1)
        emd = (emd.T / norm_value).T
    pos_index1 = [pos_edge[i][0] for i in range(len(pos_edge))]
    pos_index2 = [pos_edge[i][1] for i in range(len(pos_edge))]
    neg_index1 = [neg_edge[i][0] for i in range(len(neg_edge))]
    neg_index2 = [neg_edge[i][1] for i in range(len(neg_edge))]
    pos_emd1 = emd[pos_index1]
    pos_emd2 = emd[pos_index2]
    neg_emd1 = emd[neg_index1]
    neg_emd2 = emd[neg_index2]
    #print(pos_emd1.shape, pos_emd2.shape)
    pos_pred = np.sum(pos_emd1*pos_emd2, axis=1)
    pos_correct = np.sum(pos_pred >= threshold)
    neg_pred = np.sum(neg_emd1*neg_emd2, axis=1)
    neg_correct = np.sum(neg_pred < threshold)
    #print(pos_pred, neg_pred)
    #print(pos_correct, neg_correct)
    acc = (pos_correct + neg_correct) / (len(pos_edge) + len(neg_edge))
    TP = pos_correct
    FN = len(pos_edge) - TP
    TN = neg_correct
    FP = len(neg_edge) - TN
    F1_score = 2*TP/(2*TP+FN+FP)
    y = []
    for i in range(len(pos_edge)):
        y.append(1)
    for i in range(len(neg_edge)):
        y.append(0)
    y = np.array(y)
    pred = np.concatenate((pos_pred, neg_pred), axis=None)
    fpr, tpr, thr = metrics.roc_curve(y, pred, pos_label=1)
    AP = metrics.average_precision_score(y, pred)
    AUC = metrics.auc(fpr, tpr)
    return [acc, F1_score, AUC, AP]


#Maybe used for baselines
def eval_recommendation(emd, pos_edge, bl_flag=True, top_k=(1, 5, 10, 30, 50, 100)):
    if bl_flag:
        norm_value = np.linalg.norm(emd, axis=1)
        emd = (emd.T / norm_value).T
    test_L = [pair[0] for pair in pos_edge]
    Lvec = emd[test_L]
    sim1 = - np.dot(Lvec, emd.T)
    Lmap = {}
    Rmap = {}
    for pair in pos_edge:
        [e1, e2] = pair
        Lmap[e1] = e2
        Rmap[e2] = e1
    top_lr = [0] * len(top_k)
    L_mrr = 0
    for i in range(Lvec.shape[0]):
        rank1 = sim1[i, :].argsort()
        rank_index1 = np.where(rank1 == Lmap[pos_edge[i][0]])[0][0]
        L_mrr += 1 / (rank_index1 + 1)
        for j in range(len(top_k)):
            if rank_index1 < top_k[j]:
                top_lr[j] += 1
    L_mrr = L_mrr / len(pos_edge)
    result = []
    for i in range(len(top_lr)):
        result.append(int(10000*top_lr[i] / len(pos_edge))/10000)
    result.append(int(10000*L_mrr)/10000)
    return result



































