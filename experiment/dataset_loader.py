import json
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import StandardScaler

import experiment.utils_graphsage as utils_graphsage


# return:
#   1. feature
#   2. label
#   3. edge list
#   4. dataset split: [train, val, test]

def amazon_3m_loader():

    node_num = 2460406
    edge_num = -1
    class_num = 38

    dataset_dir = '/scratch/user/yuning.you/dataset/amazon_3m/amazon_3m'
    # id_map_file = dataset_dir + '/ppi-id_map.json'
    # node_file = dataset_dir + '/ppi-G.json'
    A_file = dataset_dir + '/Adj_hat.npz'
    feat_file = dataset_dir + '/feat_truncatedSVD.npy'
    label_file = dataset_dir + '/label.npy'
    dataset_split_file = dataset_dir + '/dataset_split.json'

    # load feature
    feat = np.load(feat_file)
    scaler = StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)

    '''
    # load id map
    with open(id_map_file, 'r') as f:
        id_map = json.load(f)
    '''

    # load label
    label = np.load(label_file)
    label = label.astype(np.int64)

    # load Adj hat matrix
    Adj_hat = sps.load_npz(A_file)

    # load dataset split: [train, val, test]
    with open(dataset_split_file, 'r') as f:
        dataset_split = json.load(f)

    '''
    with open(node_file, 'r') as f:
        data_node = json.load(f)['nodes']
    dataset_split = {'train': [], 'val': [], 'test': []}
    for dn in data_node:
        if dn['val']:
            dataset_split['val'].append(dn['id'])
        elif dn['test']:
            dataset_split['test'].append(dn['id'])
        else:
            dataset_split['train'].append(dn['id'])
    '''

    return feat, label, Adj_hat, dataset_split


def amazon_670k_loader():

    node_num = 643474
    edge_num = -1
    class_num = 32

    dataset_dir = '/scratch/user/yuning.you/dataset/amazon_670k/amazon_670k'
    # id_map_file = dataset_dir + '/ppi-id_map.json'
    # node_file = dataset_dir + '/ppi-G.json'
    A_file = dataset_dir + '/Adj_hat.npz'
    feat_file = dataset_dir + '/feat_truncatedSVD.npy'
    label_file = dataset_dir + '/label.npy'
    dataset_split_file = dataset_dir + '/dataset_split.json'

    # load feature
    feat = np.load(feat_file)
    scaler = StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)

    '''
    # load id map
    with open(id_map_file, 'r') as f:
        id_map = json.load(f)
    '''

    # load label
    label = np.load(label_file)
    label = label.astype(np.int64)

    # load Adj hat matrix
    Adj_hat = sps.load_npz(A_file)

    # load dataset split: [train, val, test]
    with open(dataset_split_file, 'r') as f:
        dataset_split = json.load(f)

    '''
    with open(node_file, 'r') as f:
        data_node = json.load(f)['nodes']
    dataset_split = {'train': [], 'val': [], 'test': []}
    for dn in data_node:
        if dn['val']:
            dataset_split['val'].append(dn['id'])
        elif dn['test']:
            dataset_split['test'].append(dn['id'])
        else:
            dataset_split['train'].append(dn['id'])
    '''

    return feat, label, Adj_hat, dataset_split


def ppi_loader():

    node_num = 56944
    edge_num = 818716
    class_num = 121

    dataset_dir = '/scratch/user/yuning.you/dataset/ppi/ppi'
    id_map_file = dataset_dir + '/ppi-id_map.json'
    node_file = dataset_dir + '/ppi-G.json'
    A_file = dataset_dir + '/A_hat.npz'
    feat_file = dataset_dir + '/ppi-feats.npy'
    label_file = dataset_dir + '/label.npy'

    # load feature
    feat = np.load(feat_file)
    scaler = StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)

    # load id map
    with open(id_map_file, 'r') as f:
        id_map = json.load(f)

    # load label
    label = np.load(label_file)
    label = label.astype(np.float32)

    # load Adj hat matrix
    Adj_hat = sps.load_npz(A_file)

    # load dataset split: [train, val, test]
    with open(node_file, 'r') as f:
        data_node = json.load(f)['nodes']
    dataset_split = {'train': [], 'val': [], 'test': []}
    for dn in data_node:
        if dn['val']:
            dataset_split['val'].append(dn['id'])
        elif dn['test']:
            dataset_split['test'].append(dn['id'])
        else:
            dataset_split['train'].append(dn['id'])

    return feat, label, Adj_hat, dataset_split

def reddit_loader():

    node_num = 232965
    edge_num = 11606919

    dataset_dir = '/scratch/user/yuning.you/dataset/reddit/reddit'
    id_map_file = dataset_dir + '/reddit-id_map.json'
    node_file = dataset_dir + '/reddit-G.json'
    A_file = dataset_dir + '/Adj_hat.npz'
    feat_file = dataset_dir + '/reddit-feats.npy'
    label_file = dataset_dir + '/reddit-class_map.json'

    # _, feat, _, _, _ = utils_graphsage.load_data('/scratch/user/yuning.you/dataset/reddit/reddit/reddit')
    print('finish loading feature')

    # load feature
    feat = np.load(feat_file)

    # load id map
    with open(id_map_file, 'r') as f:
        id_map = json.load(f)
    print('finish loading id map')

    # load label
    with open(label_file, 'r') as f:
        data_label = json.load(f)
    label = np.ones(node_num, dtype=np.int64) * -1
    for dl in data_label.items():
        label[id_map[dl[0]]] = dl[1]
    print('finish loading label')

    # load Adj hat matrix
    Adj_hat = sps.load_npz(A_file)
    print('finish loading Adj')

    # load dataset split: [train, val, test]
    with open(node_file, 'r') as f:
        data_node = json.load(f)['nodes']
    dataset_split = {'train': [], 'val': [], 'test': []}
    for dn in data_node:
        if dn['val']:
            dataset_split['val'].append(id_map[dn['id']])
        elif dn['test']:
            dataset_split['test'].append(id_map[dn['id']])
        else:
            dataset_split['train'].append(id_map[dn['id']])
    print('finish loading dataset split')

    scaler = StandardScaler()
    scaler.fit(feat[dataset_split['train']])
    feat = scaler.transform(feat)

    n = 5
    print(feat[0:n])
    print(label[0:n])

    return feat, label, Adj_hat, dataset_split

# reddit_loader()
