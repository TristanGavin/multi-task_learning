from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from make_targets import getParity, get_data
from model_training import train_model, test_model

################################################################################################################################
# DATASETS

class BinaryDataset(Dataset):

    def __init__(self):
        # data loading
        df = pd.read_csv('./targets.csv')
        X = []
        for i in range(256):
            X.append([int(i) for i in str(format(i, '08b'))])

        self.x = torch.FloatTensor(X)
        self.y1 = torch.transpose(torch.FloatTensor([df['1'].to_numpy()]), 0, 1)
        self.y2 = torch.transpose(torch.FloatTensor([df['2'].to_numpy()]), 0, 1)
        self.y3 = torch.transpose(torch.FloatTensor([df['3'].to_numpy()]), 0, 1)
        self.y4 = torch.transpose(torch.FloatTensor([df['4'].to_numpy()]), 0, 1) 
        self.num_samples = self.x.shape[0]


    def __getitem__(self, index):
        return self.x[index], self.y1[index], self.y2[index], self.y3[index], self.y4[index]

    def __len__(self):
        return self.num_samples

################################################################################################################################
# MODEL CLASSES

class MultiTask(nn.Module):

    def __init__(self, num_targets):
        super(MultiTask, self).__init__()
        self.hidden1 = nn.Linear(8, 100)  # 8 input units 160 hidden units
        self.hidden2 = nn.Linear(100, 20)
        self.output = nn.Linear(20, num_targets)    # 1 output
    
    def forward(self, x):
        x = F.relu(self.hidden1(x))         # relu activation function for hidden layers
        x = F.relu(self.hidden2(x)) 
        x = torch.sigmoid(self.output(x))   # sigmoid returns probability of being 1
        return x  

################################################################################################################################
# SINGLE TASK MODELS

# format and shuffle data
dataset = BinaryDataset()
dataloader = DataLoader(dataset=dataset, batch_size=6, shuffle=True)

# initialize models
single_task1 = MultiTask(1)
# single_task2 = SingleTask()
# single_task3 = SingleTask()
# single_task4 = SingleTask()
       
#train individual models
# train_1loss = train_model(single_task1, dataloader, [1])
# train_2loss = train_model(single_task2, X_train, [y2_train])
# train_3loss = train_model(single_task3, X_train, y3_train)
# train_4loss = train_model(single_task4, X_train, y4_train)

################################################################################################################################
# MULTI-TASK MODELS

# initialize models
multitask1 = MultiTask(2)

# train models
train_loss = train_model(multitask1, dataloader, [1, 2])

################################################################################################################################
# TEST MODEL

test_loss = test_model(single_task1, X_test, [y1_test])
multi_testloss = test_model(multitask1, X_test, [y1_test, y2_test])
print(multi_testloss)
print(test_loss)
