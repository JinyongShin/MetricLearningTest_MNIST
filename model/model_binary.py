import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        
        self.embedding = nn.Sequential(
            nn.Linear(784,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
        )
        
        self.classification = nn.Sequential(
            nn.Linear(10,2),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self,x):
        x = x.view(-1,784)
        out = self.embedding(x)
        out = self.classification(out)
        return out