import config
from model import *
import torch
import pickle
import random
import numpy as np
from preprocess import *
import os

def train(G_rwr_tuple, data_file, top_k=(1, 5, 10, 30, 50, 100)):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    data_file_pkl = open(data_file, 'rb')
    [node_num, train_edge_list, valid_edge_list, neg_valid_edges, test_edge_list, neg_test_edges] = pickle.load(data_file_pkl)
    data_file_pkl.close()
    node_index = [i for i in range(node_num)]
    LP_loss = Pro_single_loss().cuda()
    NR_loss = Ranking_single_loss().cuda()
    model = Model(config.dim, node_num).cuda()
    optimizer1 = torch.optim.Adam(model.parameters(), lr=config.lr1)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=config.lr2)
    max_valid_lp = [0, 0, 0, 0]
    max_test_lp = [0, 0, 0, 0]
    max_valid_nr = [0 for i in range(7)]
    max_test_nr = [0 for i in range(7)]
    step1_test_lp = [0, 0, 0, 0]
    step1_valid_lp = [0, 0, 0, 0]
    step1_valid_nr = [0 for i in range(7)]
    step1_test_nr = [0 for i in range(7)]
    lp_batch_size = int(node_num/config.batch_num)
    graph_data = [i for i in range(node_num)]
    print("Start Training...")
    np_emd = np.zeros([node_num, config.dim])
    for i in range(config.epoch_num1):
        print("Epoch: " + str(i) + " ******")
        random.shuffle(node_index)
        for j in range(config.batch_num):
            sample_node = node_index[j * lp_batch_size:(j + 1) * lp_batch_size]
            rwr_tuple = [G_rwr_tuple[sample_node[index]] for index in range(len(sample_node))]
            lp_left = []
            lp_pos_right = []
            lp_neg_right = []
            #Below is the correct version
            sample = pool_within_layer_NE_sampling(rwr_tuple, config.threshold)
            for l in range(len(sample)):
                for sample_pair in sample[l]:
                    lp_left.append(sample_pair[0][0])
                    lp_pos_right.append(sample_pair[0][1])
                    lp_neg_right.append(sample_pair[1][1])
            emd = model(graph_data)
            loss = LP_loss(emd, lp_left, lp_pos_right, lp_neg_right)
            loss.backward()
            optimizer1.step()
            np_emd = emd.detach().cpu().numpy()
        if i%1 == 0:
            valid_lp = eval_link_prediction(np_emd, valid_edge_list, neg_valid_edges)
            test_lp = eval_link_prediction(np_emd, test_edge_list, neg_test_edges)
            valid_nr = eval_recommendation(np_emd, valid_edge_list)
            test_nr = eval_recommendation(np_emd, test_edge_list)
            #valid_nr = [0, 0, 0, 0, 0, 0, 0]
            #test_nr = [0, 0, 0, 0, 0, 0, 0]
            log = 'Epoch1:Valid_acc : {:04f}, Valid_F1_score: {:04f}, Valid_AUC: {:04f}, Valid_AP {:04f}'
            print(log.format(i, *valid_lp))
            log = 'Test_acc : {:04f}, Test_F1_score: {:04f}, Test_AUC: {:04f}, Test_AP {:04f}'
            print(log.format(*test_lp))
            print("Node recommendation...")
            print('Validation recommendation:')
            print(valid_nr[0:7])
            print('Test recommendation:')
            print(test_nr[0:7])
            for j in range(len(valid_lp)):
                if valid_lp[3] >= step1_valid_lp[3]:
                    step1_test_lp[j] = test_lp[j]
            for j in range(len(valid_nr)):
                if valid_lp[3] >= step1_valid_lp[3]:
                    step1_test_nr[j] = test_nr[j]
    for i in range(config.epoch_num2):
        print("Epoch2: " + str(i+config.epoch_num1) + " ******")
        random.shuffle(node_index)
        for j in range(config.batch_num):
            sample_node = node_index[j * lp_batch_size:(j + 1) * lp_batch_size]
            rwr_tuple = [G_rwr_tuple[sample_node[index]] for index in range(len(sample_node))]
            lp_left = []
            lp_pos_right = []
            lp_neg_right = []
            sample = pool_within_layer_Ranking_sampling(rwr_tuple)
            for l in range(len(sample)):
                for sample_pair in sample[l]:
                    lp_left.append(sample_pair[0][0])
                    lp_pos_right.append(sample_pair[0][1])
                    lp_neg_right.append(sample_pair[1][1])
            emd = model(graph_data)
            loss = NR_loss(emd, lp_left, lp_pos_right, lp_neg_right, config.gamma)
            loss.backward()
            optimizer2.step()
            np_emd = emd.detach().cpu().numpy()
        if i % 1 == 0:
            valid_lp = eval_link_prediction(np_emd, valid_edge_list, neg_valid_edges)
            test_lp = eval_link_prediction(np_emd, test_edge_list, neg_test_edges)
            valid_nr = eval_recommendation(np_emd, valid_edge_list)
            test_nr = eval_recommendation(np_emd, test_edge_list)
            #valid_nr = [0, 0, 0, 0, 0, 0, 0]
            #test_nr = [0, 0, 0, 0, 0, 0, 0]
            log = 'Valid_acc : {:04f}, Valid_F1_score: {:04f}, Valid_AUC: {:04f}, Valid_AP {:04f}'
            print(log.format(*valid_lp))
            log = 'Test_acc : {:04f}, Test_F1_score: {:04f}, Test_AUC: {:04f}, Test_AP {:04f}'
            print(log.format(*test_lp))
            print("Node recommendation...")
            print('Validation recommendation:')
            print(valid_nr[0:7])
            print('Test recommendation:')
            print(test_nr[0:7])
            for j in range(len(valid_lp)):
                if valid_nr[2] > max_valid_nr[2]:
                    max_test_lp[j] = test_lp[j]
            for j in range(len(valid_nr)):
                if valid_nr[2] > max_valid_nr[2]:
                    max_test_nr[j] = test_nr[j]
    return step1_test_lp, step1_test_nr, max_test_lp, max_test_nr