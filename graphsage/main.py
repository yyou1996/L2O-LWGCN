import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator

from graphsage.feeder import feeder as feed
import graphsage.net as net

import yaml

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def load_pubmed():
    # hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists


def run(dataset_load_func, config_file):

    # dataset load
    feat_data, labels, adj_lists = dataset_load_func()

    # config parameter load
    with open('./config/' + config_file) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    print(args)

    num_nodes = args['node_num']

    # train, val and test set index generate
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:args['testset_size']]
    val = rand_indices[args['testset_size']:(args['testset_size'] + args['valset_size'])]
    train = list(rand_indices[(args['testset_size'] + args['valset_size']):])
    train.sort()

    # feature generate
    feat_train = torch.FloatTensor(feat_data)[train, :]
    feat_test = torch.FloatTensor(feat_data)

    # Adj matrix generate
    Adj = torch.eye(num_nodes)
    for i in range(num_nodes):
        for j in adj_lists[i]:
            Adj[i, j] = 1
            Adj[j, i] = 1
    Adj_eye = torch.eye(num_nodes)

    Adj_train = Adj[train, :][:, train]
    Adj_train = ( Adj_train / Adj_train.sum(dim=0) ).t() + Adj_eye[train, :][:, train]

    Adj_test = Adj
    Adj_test = ( Adj_test / Adj_test.sum(dim=0) ).t() + Adj_eye



    feat_0 = torch.FloatTensor(feat_data)
    Adj_0 = ( Adj / Adj.sum(dim=0) ).t() + Adj_eye
    feat_train = torch.FloatTensor(feat_data)[train, :]
    Adj_train = Adj[train, :][:, train]
    Adj_train = ( Adj_train / Adj_train.sum(dim=0) ).t() + Adj_eye[train, :][:, train]

    '''
    Adj_test = Adj
    Adj_test = ( Adj_test / Adj_test.sum(dim=0) ).t()
    Adj_test = Adj_test + Adj_eye

    __feat = features.weight
    __feat = Adj_test.mm(__feat)
    _feat = __feat[torch.LongTensor(train)]
    '''

    __feat = feat_train
    __feat = Adj_train.mm(__feat)
    _feat = __feat

    times = []

    # print(train)
#    _feat = features(torch.LongTensor(train))
#    _feat = Adj_train.mm(_feat)
    __label = labels[np.array(train)]
    _feeder = feed(_feat, __label)
    _dataset = torch.utils.data.DataLoader(
        dataset=_feeder, batch_size=1024)

    net1 = net.net1(500, 3)
    optimizer = torch.optim.SGD(net1.parameters(), lr=1)
 
    loss_func = nn.CrossEntropyLoss()
    batch = 0
    flag = 0

    for _ in range(200):
        for x, _label in _dataset:
            start_time = time.time()
            optimizer.zero_grad()
            output = net1(x)
            loss = loss_func(output, _label.view(-1))
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time-start_time)
            batch = batch + 1
            print(batch, loss.data)
            print("Training F1:", f1_score(_label, output.data.numpy().argmax(axis=1), average="micro"))
            if batch == 50:
                flag = 1
                break
        if flag == 1:
            break

    relu = nn.ReLU(inplace=True)
    w1, w2, w3 = net1.get_w()
    w1.requires_grad = False
    # _feat = relu(Adj_train.mm(_feat).mm(w1))

    '''
    __feat = Adj_test.mm(relu(__feat.mm(w1)))
    _feat = __feat[torch.LongTensor(train)]
    '''

    __feat = Adj_train.mm(relu(__feat.mm(w1)))
    _feat = __feat
 
    print(_feat.size(), _label.size())

    _feeder = feed(_feat, __label)
    _dataset = torch.utils.data.DataLoader(
        dataset=_feeder, batch_size=1024)

    net2 = net.net2(w2, w3)
    optimizer = torch.optim.SGD(net2.parameters(), lr=1)
    batch = 0
    flag = 0

    for _ in range(2000):
        for x, _label in _dataset:
            start_time = time.time()
            optimizer.zero_grad()
            output = net2(x)
            loss = loss_func(output, _label.view(-1))
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time-start_time)
            batch = batch + 1
            print(batch, loss.data)
            print("Training F1:", f1_score(_label, output.data.numpy().argmax(axis=1), average="micro"))
            if batch == 200:
                flag = 1
                break
        if flag == 1:
            break

    '''
    Adj_test = Adj
    Adj_test = ( Adj_test / Adj_test.sum(dim=0) ).t()
    Adj_test = Adj_test + Adj_eye
    '''

    w2, w3 = net2.get_w()
    # net_test = net.net_test(w1, w2, w3)

    '''
    with torch.no_grad():
        val_output = net_test(_feat, Adj_test)[train, :]
    '''

    __feat = Adj_0.mm(relu(Adj_0.mm(feat_0).mm(w1)))
    with torch.no_grad():
        val_output = relu(__feat.mm(w2)).mm(w3)[torch.LongTensor(test)]

    print("Validation F1:", f1_score(labels[test], val_output.data.numpy().argmax(axis=1), average="micro"))
    # print("Validation F1:", f1_score(__label, val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    setup_seed(50)

    # run_cora_ltf()
    run_pubmed_ltf()
