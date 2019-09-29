import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from lt_gcn.feeder import feeder as feed
import lt_gcn.net as net

import yaml

from l2l_utils.meta_optimizer import MetaModel, MetaOptimizer, FastMetaOptimizer

"""
layered training, forked from graphsage
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
        args = yaml.load(f)
    print(args)

    num_nodes = args['node_num']

    # train, val and test set index generate
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:args['testset_size']]
    val = rand_indices[args['testset_size']:(args['testset_size'] + args['valset_size'])]
    train = list(rand_indices[(args['testset_size'] + args['valset_size']):])
    train.sort()

    # feature and label generate
    feat_train = torch.FloatTensor(feat_data)[train, :]
    label_train = labels[np.array(train)]
    feat_test = torch.FloatTensor(feat_data)

    # Adj matrix generate
    Adj = torch.eye(num_nodes)
    for i in range(num_nodes):
        for j in adj_lists[i]:
            Adj[i, j] = 1
            Adj[j, i] = 1
    Adj_eye = torch.eye(num_nodes)

    Adj_train = Adj[train, :][:, train]
    Adj_train = ( Adj_train / Adj_train.sum(dim=0) ).t()
    Adj_train = Adj_train + Adj_eye[train, :][:, train]

    Adj_test = Adj
    Adj_test = ( Adj_test / Adj_test.sum(dim=0) ).t()
    Adj_test = Adj_test + Adj_eye

    # layered training
    times = []
    weight_list = nn.ParameterList()

    loss_func = nn.CrossEntropyLoss()
    relu = nn.ReLU(inplace=True)

    for l in range(args['layer_num']):

        feat_train = torch.mm(Adj_train, feat_train) # sparse.mm
        feeder_train = feed(feat_train, label_train)
        dataset_train = torch.utils.data.DataLoader(dataset=feeder_train, batch_size=args['batch_size'], shuffle=True)

        if l == 0:
            in_channel = args['feat_dim']
        else:
            in_channel = args['layer_output_dim'][l-1]
        hidden_channel = args['layer_output_dim'][l]
        out_channel = args['class_num']

        net_train = net.net_train(in_channel, hidden_channel, out_channel).to(torch.device(args['device']))
        optimizer = torch.optim.SGD(net_train.parameters(), lr=args['learning_rate'])

        batch = 0
        flag = 0
        while True:
            for x, x_label in dataset_train:

                x = x.to(torch.device(args['device']))
                x_label = x_label.to(torch.device(args['device']))

                start_time = time.time()
                optimizer.zero_grad()
                output = net_train(x)
                loss = loss_func(output, x_label.view(-1))
                loss.backward()
                optimizer.step()
                end_time = time.time()
                times.append(end_time - start_time)
                batch = batch + 1
                print('batch', batch, 'loss:', loss.data)
                # print("Acc in val:", f1_score(x_label, output.data.numpy().argmax(axis=1), average="micro"))
                if batch == args['layer_train_batch'][l]:
                    flag = 1
                    break

            if flag == 1:
                w = net_train.get_w()
                w.requires_grad = False
                feat_train = torch.mm(feat_train, w.to(torch.device('cpu')))
                feat_train = relu(feat_train)
                weight_list.append(w)
                if l == args['layer_num'] - 1:
                    classifier = net_train.get_c()
                    classifier.requires_grad = False
                break

    weight_list = weight_list.to(torch.device('cpu'))
    classifier = classifier.to(torch.device('cpu'))
    # test
    net_test = net.net_test()
    with torch.no_grad():
        output = net_test(feat_test, Adj_test, weight_list, classifier)[test, :]

    print("Acc in test:", f1_score(labels[test], output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))

def run_l2l(dataset_load_func, config_file):

    # dataset load
    feat_data, labels, adj_lists = dataset_load_func()

    # config parameter load
    with open('./config/' + config_file) as f:
        args = yaml.load(f)
    print(args)

    num_nodes = args['node_num']

    # train, val and test set index generate
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:args['testset_size']]
    val = rand_indices[args['testset_size']:(args['testset_size'] + args['valset_size'])]
    train = list(rand_indices[(args['testset_size'] + args['valset_size']):])
    train.sort()

    # feature and label generate
    feat_train = torch.FloatTensor(feat_data)[train, :]
    label_train = labels[np.array(train)]
    feat_test = torch.FloatTensor(feat_data)

    # Adj matrix generate
    Adj = torch.eye(num_nodes)
    for i in range(num_nodes):
        for j in adj_lists[i]:
            Adj[i, j] = 1
            Adj[j, i] = 1
    Adj_eye = torch.eye(num_nodes)

    Adj_train = Adj[train, :][:, train]
    Adj_train = ( Adj_train / Adj_train.sum(dim=0) ).t()
    Adj_train = Adj_train + Adj_eye[train, :][:, train]

    Adj_test = Adj
    Adj_test = ( Adj_test / Adj_test.sum(dim=0) ).t()
    Adj_test = Adj_test + Adj_eye

    # layered training
    times = []
    weight_list = nn.ParameterList()

    loss_func = nn.CrossEntropyLoss()
    relu = nn.ReLU(inplace=True)

    # create meta optimizer
    # the '1' parameters do not matter
    meta_net_train = net.net_train(1, 1, 1).to(torch.device(args['device']))
    meta_optimizer = FastMetaOptimizer(MetaModel(meta_net_train), 1, 1).to(torch.device(args['device']))
    optimizer =  torch.optim.Adam(meta_optimizer.parameters(), lr=args['meta_learning_rate'])

    for l in range(args['layer_num']):

        feat_train = torch.mm(Adj_train, feat_train) # sparse.mm
        feeder_train = feed(feat_train, label_train)
        dataset_train = torch.utils.data.DataLoader(dataset=feeder_train, batch_size=args['batch_size'], shuffle=True)

        if l == 0:
            in_channel = args['feat_dim']
        else:
            in_channel = args['layer_output_dim'][l-1]
        hidden_channel = args['layer_output_dim'][l]
        out_channel = args['class_num']

        net_train = net.net_train(in_channel, hidden_channel, out_channel).to(torch.device(args['device']))
        meta_net_train = net.net_train(in_channel, hidden_channel, out_channel).to(torch.device(args['device']))
        meta_optimizer.meta_model = MetaModel(meta_net_train)

        batch = 0
        flag = 0
        while True:
            for x, x_label in dataset_train:

                x = x.to(torch.device(args['device']))
                x_label = x_label.to(torch.device(args['device']))

                meta_optimizer.reset_lstm(keep_states=(batch > 0), model=net_train, use_cuda=(args['device'] == 'cuda'))
                loss_sum = 0
                prev_loss = torch.zeros(1).to(torch.device(args['device']))

                start_time = time.time()

                # meta learning
                for bs in range(args['bptt_step']):

                    # compute the gradient of the original net
                    net_train.zero_grad()
                    output = net_train(x)
                    loss = loss_func(output, x_label.view(-1))
                    loss.backward()

                    # meta learning
                    meta_model = meta_optimizer.meta_update(net_train, loss.data)
                    output = meta_model(x)
                    loss = loss_func(output, x_label.view(-1))

                    loss_sum += (loss - Variable(prev_loss))
                    prev_loss = loss.data

                meta_optimizer.zero_grad()
                loss_sum.backward()
                for param in meta_optimizer.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

                end_time = time.time()
                times.append(end_time - start_time)
                batch = batch + 1
                print('batch', batch, 'loss:', loss.data)
                # print("Acc in val:", f1_score(x_label, output.data.numpy().argmax(axis=1), average="micro"))
                if batch == args['layer_train_batch'][l]:
                    flag = 1
                    break

            if flag == 1:
                w = net_train.get_w()
                w.requires_grad = False
                feat_train = torch.mm(feat_train, w.to(torch.device('cpu')))
                feat_train = relu(feat_train)
                weight_list.append(w)
                if l == args['layer_num'] - 1:
                    classifier = net_train.get_c()
                    classifier.requires_grad = False
                break

    weight_list = weight_list.to(torch.device('cpu'))
    classifier = classifier.to(torch.device('cpu'))
    # test
    net_test = net.net_test()
    with torch.no_grad():
        output = net_test(feat_test, Adj_test, weight_list, classifier)[test, :]

    print("Acc in test:", f1_score(labels[test], output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    setup_seed(50)
    run_l2l(load_pubmed, 'pubmed.yaml')
