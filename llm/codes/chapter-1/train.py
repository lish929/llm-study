# -*- coding: utf-8 -*-
# @Time    : 2024/9/22 19:24
# @Author  : Lee
# @Project ：chapter-1 
# @File    : train.py


from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



class TorchDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        result = self.net(x)
        return result

class Trainer(object):
    def __init__(self,train_dataset,test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = DataLoader(dataset=self.train_dataset,batch_size=4,shuffle=True,drop_last=True)
        self.test_dataloader = DataLoader(dataset=self.test_dataset,batch_size=4,shuffle=True,drop_last=True)
        self.model = TorchModel().to("cuda")
        self.loss = nn.BCEWithLogitsLoss()
        self.opt = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def train(self):
        self.model.train()
        for idx, (x, y) in enumerate(self.train_dataloader):
            x, y = x.to("cuda"), y.to("cuda")

            result = self.model(x)
            loss = self.loss(result, y.view(-1, 1))

            self.model.zero_grad()
            loss.backward()
            self.opt.step()

            if idx % 10 == 0:
                loss, current = loss.item(), idx * len(x)
                print(f"loss: {loss:>7f}")

    def test(self):
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for x, y in self.test_dataloader:
                x, y = x.to("cuda"), y.to("cuda")
                result = self.model(x)
                test_loss += self.loss(result, y.view(-1, 1)).item()
                pre = F.sigmoid(result)
                output = (pre > 0.5).float()
                correct += (output == y.view(-1, 1)).sum().item()
        print(f"Accuracy: {(100 * correct/len(self.train_dataloader)):>0.1f}")

if __name__ == '__main__':
    x,y = make_circles(n_samples=1000, noise=0.05, factor=0.5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    train_dataset = TorchDataset(X_train, y_train)
    test_dataset = TorchDataset(X_test, y_test)
    trainer = Trainer(train_dataset,test_dataset)

    for epoch in range(3):
        trainer.train()
    trainer.test()
