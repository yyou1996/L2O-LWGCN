import torch
import torch.utils.data as d


class feeder(d.Dataset):

    def __init__(self, _feat, _label):

        self.feat = _feat
        self.label = _label

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        data = self.feat[index]
        label = self.label[index]

        return data, label
