from torch import optim
import matplotlib.pyplot as plt

from feeder.feeder import Feeder
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data

from net.mlp import *


class Processor:
    def __init__(self, train_data_path, train_label_path, test_data_path, test_label_path):
        self.acc = []
        self.loader = {}
        self.loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(train_data_path, train_label_path),
                batch_size=128,
                shuffle=True,
                num_workers=4,
                drop_last=True)
        self.loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(test_data_path, test_label_path),
            batch_size=128,
            shuffle=True,
            num_workers=4,
            drop_last=True)
        self.loss = nn.CrossEntropyLoss()
        # self.dev = torch.device('cpu')
        self.dev = torch.device('cuda:1')
        self.model = MLP2().to(self.dev)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.1,
            momentum=0.9,
            nesterov=True,
            weight_decay=0.0001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

    def train(self, epoch=50):
        self.model.train()
        loss_res = []

        pbar = tqdm(range(epoch))
        for i in pbar:
            if i%5 == 0:
                self.test()
            # local_loss = []
            for data, label in self.loader['train']:
                # get data
                data = data.float().to(self.dev)
                label = label.long().to(self.dev)

                y = self.model(data)
                loss = self.loss(y, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # local_loss.append(loss.item())
            self.scheduler.step()
            # loss_res.append(np.mean(local_loss))
        print(loss_res)

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.acc.append(accuracy)
        print('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def test(self, evaluation=True):
        self.model.eval()
        loader = self.loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in loader:
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            with torch.no_grad():
                output = self.model(data)

            result_frag.append(output.cpu().numpy())
            if evaluation:
                with torch.no_grad():
                    loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            # show top-k accuracy
            # 可能性在前k位均视为正确
            for k in [1]:
                self.show_topk(k)



train_data_path = './data/train_feature.pkl'
train_label_path = './data/train_label3.pkl'
test_data_path = './data/test_feature.pkl'
test_label_path = './data/test_label3.pkl'
p = Processor(train_data_path, train_label_path, test_data_path, test_label_path)
p.train(61)

print(p.acc)