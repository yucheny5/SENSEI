###sensei
#Cora, Citeseer, C.ele, NS
"""
dataset = "NS"
training_ratio = 0.7
valid_ratio = 0.1
data_file = "./datasets/" + dataset + '/' + str(training_ratio) + '-' + str(valid_ratio) +'.pkl'
batch_num = 1
dim = 128
lr1 = 0.2
lr2 = 0.1
threshold = 0.05
epoch_num1 = 20
epoch_num2 = 20
gamma = 0.1

###ve
dataset = "Citeseer"
training_ratio = 0.7
valid_ratio = 0.1
data_file = "./datasets/" + dataset + '/' + str(training_ratio) + '-' + str(valid_ratio) +'.pkl'
batch_num = 1
dim = 128
lr1 = 0.1
lr2 = 0.005
threshold = 0.005
epoch_num1 = 20
epoch_num2 = 50
gamma = 0.0001

###red
dataset = "Celegans"
training_ratio = 0.7
valid_ratio = 0.1
data_file = "./datasets/" + dataset + '/' + str(training_ratio) + '-' + str(valid_ratio) +'.pkl'
batch_num = 1
dim = 128
lr1 = 0.02
lr2 = 0.01
threshold = 0.008
epoch_num1 = 40
epoch_num2 = 40
gamma = 0.05

###baselines
dataset = "Power"
training_ratio = 0.7
valid_ratio = 0.1
data_file = "./datasets/" + dataset + '/' + str(training_ratio) + '-' + str(valid_ratio) +'.pkl'
batch_num = 1
dim = 128
lr1 = 0.1
lr2 = 0.1
threshold = 0.01
epoch_num1 = 100
epoch_num2 = 100
gamma = 0.01


###yyc_vgnn
dataset = "Power"
training_ratio = 0.7
valid_ratio = 0.1
data_file = "./datasets/" + dataset + '/' + str(training_ratio) + '-' + str(valid_ratio) +'.pkl'
batch_num = 1
dim = 128
lr1 = 0.1
lr2 = 0.01
threshold = 0.02
epoch_num1 = 100
epoch_num2 = 100
gamma = 0.01
"""
dataset = "Cora"
###node_num 41,554
#edge_num 1,362,229
training_ratio = 0.7
valid_ratio = 0.1
data_file = "./datasets/" + dataset + '/' + str(training_ratio) + '-' + str(valid_ratio) +'.pkl'
batch_num = 1
dim = 128
lr1 = 0.1
lr2 = 0.005
threshold = 0.01
epoch_num1 = 100
epoch_num2 = 100
gamma = 0.0001

res_file = "./datasets/" + dataset + '/' + str(training_ratio) + '-' + str(valid_ratio) +'res.pkl'