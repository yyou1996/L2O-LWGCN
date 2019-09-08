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

def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(loss)
        print(batch, loss.data)

    val_output = graphsage.forward(val) 
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))

def load_pubmed():
    #hardcoded for simplicity...
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

def run_pubmed():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 19717
    feat_data, labels, adj_lists = load_pubmed()
    # print(labels.shape)
    # print(adj_lists[3], 1)
    features = nn.Embedding(19717, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=True)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 25

    graphsage = SupervisedGraphSage(3, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(200):
        batch_nodes = train[:1024]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.data)

    val_output = graphsage.forward(val) 
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))

def run_cora_ltf():
    
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=True)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 25

    graphsage = SupervisedGraphSage(3, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])
    train.sort()

    Adj = torch.eye(2708)
    for i in range(2708):
        for j in adj_lists[i]:
            Adj[i, j] = 1
            Adj[j, i] = 1
    Adj_eye = torch.eye(2708)

    '''
    Adj_train = Adj[train, :][:, train]
    Adj_train = ( Adj_train / Adj_train.sum(dim=0) ).t()
    Adj_train = Adj_train + Adj_eye[train, :][:, train]
    '''

    Adj_test = Adj
    Adj_test = ( Adj_test / Adj_test.sum(dim=0) ).t()
    Adj_test = Adj_test + Adj_eye

    __feat = features.weight
    __feat = Adj_test.mm(__feat)
    _feat = __feat[torch.LongTensor(train)]

    times = []

    # print(train)
#    _feat = features(torch.LongTensor(train))
#    _feat = Adj_train.mm(_feat)
    __label = labels[np.array(train)]
    _feeder = feed(_feat, __label)
    _dataset = torch.utils.data.DataLoader(
        dataset=_feeder, batch_size=256)

    net1 = net.net1(1433, 7)
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
            if batch == 80:
                flag = 1
                break
        if flag == 1:
            break

    relu = nn.ReLU(inplace=True)
    w1, w2, w3 = net1.get_w()
    w1.requires_grad = False
    # _feat = relu(Adj_train.mm(_feat).mm(w1))
    __feat = Adj_test.mm(relu(__feat.mm(w1)))
    _feat = __feat[torch.LongTensor(train)]

    print(_feat.size(), _label.size())

    _feeder = feed(_feat, __label)
    _dataset = torch.utils.data.DataLoader(
        dataset=_feeder, batch_size=256)

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
            if batch == 170:
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

    with torch.no_grad():
        val_output = relu(__feat.mm(w2)).mm(w3)[torch.LongTensor(test)]

    print("Validation F1:", f1_score(labels[test], val_output.data.numpy().argmax(axis=1), average="micro"))
    # print("Validation F1:", f1_score(__label, val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))

def run_pubmed_ltf():
    
    num_nodes = 19717
    feat_data, labels, adj_lists = load_pubmed()
    features = nn.Embedding(19717, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=True)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 25

    graphsage = SupervisedGraphSage(3, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])
    train.sort()

    Adj = torch.eye(19717)
    for i in range(19717):
        for j in adj_lists[i]:
            Adj[i, j] = 1
            Adj[j, i] = 1
    Adj_eye = torch.eye(19717)

    '''
    Adj_train = Adj[train, :][:, train]
    Adj_train = ( Adj_train / Adj_train.sum(dim=0) ).t()
    Adj_train = Adj_train + Adj_eye[train, :][:, train]
    '''

    Adj_test = Adj
    Adj_test = ( Adj_test / Adj_test.sum(dim=0) ).t()
    Adj_test = Adj_test + Adj_eye

    __feat = features.weight
    __feat = Adj_test.mm(__feat)
    _feat = __feat[torch.LongTensor(train)]

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
    __feat = Adj_test.mm(relu(__feat.mm(w1)))
    _feat = __feat[torch.LongTensor(train)]

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

    run_cora_ltf()
    # run_pubmed_ltf()
