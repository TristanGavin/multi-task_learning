from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Task 1 = B1      OR  Parity(B2-B8)
# Task 2 = not(B1) OR  Parity(B2-B8)
# Task 3 = B1      AND Parity(B2-B8)
# Task 4 = not(B1) AND Parity(B2-B8)

# build the dataset
def getParity(n):
    parity = 0
    while n:
        parity = ~parity
        n = n & (n - 1)
    return parity
    
def get_data():
    tasks = {
            1: [],
            2: [],
            3: [],
            4: []
            }

    mask = (1 << 8) - 1

    for byte in range(256):
        binary = format(byte, '08b')
        little = byte & mask # last 8 bits

        for key, value in tasks.items():
            if key == 1:
                # Task 1
                if (byte & (1 << (8-1))) or (getParity(little) == -1):
                    tasks[1].append(1)
                else:
                    tasks[1].append(0)
            if key == 2:
                # Task 2 
                if not(byte & (1 << (8-1))) or (getParity(little) == -1):
                    tasks[2].append(1)
                else:
                    tasks[2].append(0)
            if key == 3:
                # Task 3
                if (byte & (1 << (8-1))) and (getParity(little) == -1):
                    tasks[3].append(1)
                else:
                    tasks[3].append(0)
            if key == 4:
                # Task 4
                if not(byte & (1 << (8-1))) and (getParity(little) == -1):
                    tasks[4].append(1)
                else:
                    tasks[4].append(0)
    
    return tasks


def train_model(model, X, label):
    # optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    loss_func = torch.nn.BCELoss()  # Binary cross entropy - (label, probability of being 1)

    # fig, ax = plt.subplots(figsize=(12,7))

    max_iters = 500
    #train network
    for epoch in range(max_iters):

        pred= model(X) # make prediction

        loss = loss_func(torch.squeeze(pred), label) 
        
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # take gradient step.

        if epoch % 50 == 0:    # print every 2000 mini-batches
            print(f'training loss: {loss}')

    return loss

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

    def __init__(self):
        super(SingleTask, self).__init__()
        self.hidden1 = nn.Linear(8, 100)  # 8 input units 160 hidden units
        self.hidden2 = nn.Linear(100, 20)
        self.output = nn.Linear(20, 8)    # 1 output
    
    def forward(self, x):
        x = F.relu(self.hidden1(x))         # relu activation function for hidden layers
        x = F.relu(self.hidden2(x)) 
        x = torch.sigmoid(self.output(x))   # sigmoid returns probability of being 1
        return x  #returning 8 predictions


################################################################################################################################


# format and shuffle data
data = get_data() # returns dict of task y vals (task1: [y1, y2, ...], task2: ...) 
X = []
for i in range(256):
    X.append([int(i) for i in str(format(i, '08b'))])

X, y1, y2, y3, y4 = shuffle(X, data[1], data[2], data[3], data[4], random_state=0)
X_train, X_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test, y4_train, y4_test = train_test_split(X, y1, y2, y3, y4, test_size=0.6, random_state=42)

# initialize models
single_task1 = SingleTask()
single_task2 = SingleTask()
single_task3 = SingleTask()
single_task4 = SingleTask()

# convert everything to tensor
X_train = torch.FloatTensor(X_train)
y1_train = torch.FloatTensor(y1_train)
y2_train = torch.FloatTensor(y2_train)
y3_train = torch.FloatTensor(y3_train)
y4_train = torch.FloatTensor(y4_train)

#train individual models
train_1loss = train_model(single_task1, X_train, y1_train)
train_2loss = train_model(single_task2, X_train, y2_train)
train_3loss = train_model(single_task3, X_train, y3_train)
train_4loss = train_model(single_task4, X_train, y4_train)















