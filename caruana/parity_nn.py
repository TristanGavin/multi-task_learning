from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from get_parity import getParity, get_data
from model_training import train_model, test_model

################################################################################################################################
# DATASETS

class BinaryDataset(Dataset):

    def __init__(self):
        # data loading

    def __getitem__(self, index):

################################################################################################################################
# MODEL CLASSES

class SingleTask(nn.Module):

    def __init__(self):
        super(SingleTask, self).__init__()
        self.hidden1 = nn.Linear(8, 100)  # 8 input units 160 hidden units
        self.hidden2 = nn.Linear(100, 20)
        self.output = nn.Linear(20, 1)    # 1 output
    
    def forward(self, x):
        x = F.relu(self.hidden1(x))     # relu activation function for hidden layers
        x = F.relu(self.hidden2(x)) 
        x = torch.sigmoid(self.output(x))   # sigmoid returns probability of being 1
        return x

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
data = get_data() # returns dict of task y vals (task1: [y1, y2, ...], task2: ...) 
X = []
for i in range(256):
    X.append([int(i) for i in str(format(i, '08b'))])

X, y1, y2, y3, y4 = shuffle(X, data[1], data[2], data[3], data[4], random_state=0)
X_train, X_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test, y4_train, y4_test = train_test_split(X, y1, y2, y3, y4, test_size=0.6, random_state=42)

# convert everything to tensor
X_train = torch.FloatTensor(X_train)
y1_train = torch.FloatTensor(y1_train)
y2_train = torch.FloatTensor(y2_train)
y3_train = torch.FloatTensor(y3_train)
y4_train = torch.FloatTensor(y4_train)
X_test = torch.FloatTensor(X_test)

 
# initialize models
single_task1 = SingleTask()
single_task2 = SingleTask()
single_task3 = SingleTask()
single_task4 = SingleTask()
       
#train individual models
train_1loss = train_model(single_task1, X_train, [y1_train])
train_2loss = train_model(single_task2, X_train, [y2_train])
# train_3loss = train_model(single_task3, X_train, y3_train)
# train_4loss = train_model(single_task4, X_train, y4_train)

################################################################################################################################
# MULTI-TASK MODELS

# initialize models
multitask1 = MultiTask(2)

# train models
train_loss = train_model(multitask1, X_train, [y1_train, y2_train])


################################################################################################################################
# TEST MODEL

test_loss = test_model(single_task1, X_test, [y1_test])
multi_testloss = test_model(multitask1, X_test, [y1_test, y2_test])
print(multi_testloss)
print(test_loss)
