import torch
import torch.nn as nn


class net_train(nn.Module):

    def __init__(self, in_channel, hidden_channel, out_channel):
        super().__init__()

        self.weight = nn.Parameter(
                torch.zeros((in_channel, hidden_channel), dtype=torch.float))
        nn.init.xavier_uniform(self.weight)

        self.classifier = nn.Parameter(
                torch.zeros((hidden_channel, out_channel), dtype=torch.float))
        nn.init.xavier_uniform(self.classifier)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.mm(x, self.weight)
        x = self.relu(x)
        x = torch.mm(x, self.classifier)
        return x

    def get_w(self):
        return self.weight

    def get_c(self):
        return self.classifier

class net_test(nn.Module):

    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, Adj, weight_list, classifier):

        for w in weight_list:
            x = torch.sparse.mm(Adj, x)
            x = torch.mm(x, w)
            x = self.relu(x)

        x = torch.mm(x, classifier)

        return x


'''

class net1(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.weight1 = nn.Parameter(
                torch.zeros((in_channel, 128), dtype=torch.float))
        nn.init.xavier_uniform(self.weight1)
        self.bn1 = nn.BatchNorm1d(128)

        self.weight2 = nn.Parameter(
                torch.zeros((128, 128), dtype=torch.float))
        nn.init.xavier_uniform(self.weight2)
        self.bn2 = nn.BatchNorm1d(128)

        self.weight3 = nn.Parameter(
                torch.zeros((128, out_channel), dtype=torch.float))
        nn.init.xavier_uniform(self.weight3)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.relu(torch.mm(x, self.weight1))
        x = self.relu(torch.mm(x, self.weight2))
        x = torch.mm(x, self.weight3)

        return x

    def get_w(self):
        return self.weight1, self.weight2, self.weight3

class net2(nn.Module):

    def __init__(self, w2, w3):
        super().__init__()

        self.weight2 = w2
        self.weight3 = w3

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.relu(torch.mm(x, self.weight2))
        x = torch.mm(x, self.weight3)

        return x

    def get_w(self):
        return self.weight2, self.weight3
'''

