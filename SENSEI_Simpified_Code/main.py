import config
from preprocess import *
from train import train
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
if __name__ == '__main__':
    step1_lp_list = [[], [], [], []]
    max_lp_list = [[], [], [], []]
    step1_nr_list = [[] for i in range(7)]
    max_nr_list = [[] for i in range(7)]
    for i in range(5):
        data_file = config.data_file + '-' + str(i) + '.pkl'
        #edge_index, node_num = load_mat_data(config.dataset)
        edge_index, node_num = download(config.dataset)
        split_data(edge_index, node_num, data_file)
    for i in range(5):
        data_file = config.data_file + '-' + str(i) + '.pkl'
        adj_matrix = preprocess(data_file)
        G_rwr_tuple = cal_rwr(adj_matrix)
        step1_test_lp, step1_test_nr, max_test_lp, max_test_nr = train(G_rwr_tuple, data_file)
        for j in range(len(step1_lp_list)):
            step1_lp_list[j].append(step1_test_lp[j])
            max_lp_list[j].append(max_test_lp[j])
        for j in range(len(step1_nr_list)):
            step1_nr_list[j].append(step1_test_nr[j])
            max_nr_list[j].append(max_test_nr[j])
    res_file = open(config.res_file, 'wb')
    pickle.dump([step1_lp_list, step1_nr_list, max_lp_list, max_nr_list], res_file)
    res_file.close()
    print(config.dataset)
    print("Step 1 LP: ACC: {:.4f}, std: {:.4f},".format(np.array(step1_lp_list)[0].mean(axis=0), np.array(step1_lp_list)[0].std(axis=0)))
    print("Step 1 LP: F1: {:.4f}, std: {:.4f},".format(np.array(step1_lp_list)[1].mean(axis=0), np.array(step1_lp_list)[1].std(axis=0)))
    print("Step 1 LP: AUC: {:.4f}, std: {:.4f},".format(np.array(step1_lp_list)[2].mean(axis=0), np.array(step1_lp_list)[2].std(axis=0)))
    print("Step 1 LP: AP: {:.4f}, std: {:.4f},".format(np.array(step1_lp_list)[3].mean(axis=0),
                                                        np.array(step1_lp_list)[3].std(axis=0)))
    print("Step 2 LP: ACC: {:.4f}, std: {:.4f},".format(np.array(max_lp_list)[0].mean(axis=0), np.array(max_lp_list)[0].std(axis=0)))
    print("Step 2 LP: F1: {:.4f}, std: {:.4f},".format(np.array(max_lp_list)[1].mean(axis=0), np.array(max_lp_list)[1].std(axis=0)))
    print("Step 2 LP: AUC: {:.4f}, std: {:.4f},".format(np.array(max_lp_list)[2].mean(axis=0), np.array(max_lp_list)[2].std(axis=0)))
    print("Step 2 LP: AP: {:.4f}, std: {:.4f},".format(np.array(max_lp_list)[3].mean(axis=0),
                                                        np.array(max_lp_list)[3].std(axis=0)))
    cal_step1_nr = []
    cal_max_nr = []
    for i in range(7):
        cal_step1_nr.append([round(np.array(step1_nr_list[i]).mean(axis=0), 4), round(np.array(step1_nr_list[i]).std(axis=0), 4)])
        cal_max_nr.append([round(np.array(max_nr_list[i]).mean(axis=0), 4), round(np.array(max_nr_list[i]).std(axis=0), 4)])

    print(cal_step1_nr)
    print(cal_max_nr)






