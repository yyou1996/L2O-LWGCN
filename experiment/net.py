import torch
import torch.nn as nn
import scipy.sparse as sps


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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.mm(x, self.weight)
        x = self.relu(x)
        x = torch.mm(x, self.classifier)
        x = self.sigmoid(x)
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
            x = Adj.dot(x.numpy())
            x = torch.FloatTensor(x)
            x = torch.mm(x, w)
            x = self.relu(x)

        x = torch.mm(x, classifier)
        x[x>0] = 1
        x[x<=0] = 0

        return x


