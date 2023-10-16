import torch
import torch.nn.functional as F

###network embedding loss in single network
class Pro_single_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, left, pos_right, neg_right):
        left_emd = out[left]
        pos_right_emd = out[pos_right]
        neg_right_emd = out[neg_right]
        pos_res = torch.sum(left_emd * pos_right_emd)
        neg_res = torch.sum(left_emd * neg_right_emd)

        return (neg_res - pos_res)/len(left)

###ranking loss in single network
class Ranking_single_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, left, pos_right, neg_right, single_gamma):
        left_emd = out[left]
        pos_right_emd = out[pos_right]
        neg_right_emd = out[neg_right]
        pos_res = torch.sum(left_emd * pos_right_emd, 1)
        neg_res = torch.sum(left_emd * neg_right_emd, 1)
        L1 = torch.sum(F.relu(neg_res - pos_res + single_gamma))
        return L1/len(left)


class Node2Vec(torch.nn.Module):
    def __init__(self, nodes_size=20000, dim=300):
        super(Node2Vec, self).__init__()
        self.nodes_size = nodes_size
        self.dim = dim
        self.vector = torch.nn.Embedding(self.nodes_size, self.dim)
        self.vector.weight = torch.nn.Parameter(torch.FloatTensor(self.nodes_size, self.dim).uniform_(0, 1))
        self.vector.requires_weight = True

    def forward(self, data):
        v = torch.LongTensor(data)
        v = v.cuda() if self.vector.weight.is_cuda else v
        return F.normalize(self.vector(v), p=2, dim=1)
        #return self.vector(v)

class Model(torch.nn.Module):
    def __init__(self, dim, node_num):
        super(Model, self).__init__()
        self.node2vec = Node2Vec(node_num, dim)

    def forward(self, graph_data):
        emd = self.node2vec(graph_data)
        return emd
