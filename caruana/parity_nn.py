from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from make_targets import getParity, get_data
from model_training import train_model, test_model
import os

################################################################################################################################
# DATASETS

class BinaryDataset(Dataset):

    def __init__(self, filename):
        # data loading
        df = pd.read_csv(filename) # bits 3, 4 don't matter
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
# MAKE GRAPHS FUNCITON

def make_graphs(train_acc, test_acc, target_acc, taskname):
            
    x = [i for i in range(len(test_acc))]
    x = x[::10]

    if len(target_acc) > 1: 
        for idx, target in enumerate(target_acc):


            plt.figure(figsize=(8, 8), dpi=80)
            plt.title(f"task {target}")
            plt.plot(x, target_acc[target][0][::10], label="train accuracy", color='dodgerblue', linewidth=1.5)
            plt.plot(x, target_acc[target][1][::10], label="test accuracy", color='red', linewidth=1.5)
            plt.legend(frameon=False, prop={'size': 22})
            plt.xlabel('Epochs')
            plt.ylabel('% correct') 
            plt.legend(frameon=False)
            plt.tick_params(axis='both', which='major', labelsize=17)
            # plt.savefig('./graphs/loss_test.png')
            plt.show()


    # target_acc[target][train/test][value] 

    plt.figure(figsize=(8, 8), dpi=80)
    plt.title(taskname)
    plt.plot(x, target1_acc[1][0][::10], label="train accuracy", color='dodgerblue', linewidth=1.5)
    plt.plot(x, test_acc[::10], label="test accuracy", color='red', linewidth=1.5)
    plt.legend(frameon=False, prop={'size': 22})
    plt.xlabel('Epochs')
    plt.ylabel('% correct') 
    plt.legend(frameon=False)
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.show()

    return

################################################################################################################################
# IMPORT DATA

# format and shuffle data
dataset = BinaryDataset('./targets2.csv')
dataset_size = len(dataset)
valid_split = 0.5
random_seed = 46

# Creating data indices for training and validation splits:
indices = list(range(dataset_size))
split = int(np.floor(valid_split * dataset_size))
np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=len(train_sampler), 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=len(valid_sampler),
                                                sampler=valid_sampler)
dataloaders = [train_loader, validation_loader]

# initialize models
single_task1 = MultiTask(1) # task 1 
single_task2 = MultiTask(1) # task 2
single_task3 = MultiTask(1) # task 3 
single_task4 = MultiTask(1) # task 4
    
# train individual models
print("training task 1 (single task learning): ")
print("----------------------------------------------------")
train1_acc, test1_acc, target1_acc = train_model(single_task1, dataloaders, [1])
# make_graphs(train1_acc, test1_acc, target1_acc, "task1_STL")

# print("training task 2 (single task learning): ")
# print("----------------------------------------------------")
# train2_acc, test2_acc, target2_acc = train_model(single_task2, dataloaders, [2])
# make_graphs(train2_acc, test2_acc, target2_acc, "task2_STL")
# 
# print("training task 3 (single task learning): ")
# print("----------------------------------------------------")
# train3_acc, test3_acc, target3_acc = train_model(single_task3, dataloaders, [3])
# make_graphs(train3_acc, test3_acc, target3_acc, "task3_STL")
# 
# print("training task 4 (single task learning): ")
# print("----------------------------------------------------")
# train4_acc, test4_acc, target4_acc = train_model(single_task4, dataloaders, [4])
# make_graphs(train4_acc, test4_acc, target4_acc, "task4_STL")
# 
# # save models torch.save(model.state_dict()
# torch.save(single_task1.state_dict(), "./models/single_task1.pth") 
# torch.save(single_task2.state_dict(), "./models/single_task2.pth") 
# torch.save(single_task3.state_dict(), "./models/single_task3.pth") 
# torch.save(single_task4.state_dict(), "./models/single_task4.pth") 

################################################################################################################################
# MULTI-TASK MODELS

print()
# initialize models 
multitask1 = MultiTask(2)
multitask2 = MultiTask(2)
multitask4 = MultiTask(4)

# train models
print("training task 1 and 2 (Multi-task learning): ")
print("----------------------------------------------------")
train_acc, test_acc, target_acc = train_model(multitask1, dataloaders, [1, 2])


print("training task 1-4 (Multi-task learning): ")
print("----------------------------------------------------")
mtltrain_acc, mtltest_acc, mtltarget_acc = train_model(multitask4, dataloaders, [1, 2, 3, 4])

print("training task 1 and 3 (Multi-task learning): ")
print("----------------------------------------------------")
mtltrain3_acc, mtltest3_acc, mtltarget3_acc = train_model(multitask1, dataloaders, [1, 3])


# make_graphs(train_acc, test_acc, target_acc, "task1_task2_MTL") 
# make_graphs(mtltrain_acc, mtltest_acc, mtltarget_acc, "MTL")


################################################################################################################################
# TEST MODEL

print("----------------------------------------------------")
print(f"max accuracy target 1 (STL): {max(test1_acc)}")
print(f"max accuracy target 1 (MTL 1 and 2) {max(target_acc[1][1])}")
print(f"max accuracy target 1 (MTL 1-4): {max(mtltarget_acc[1][1])}") 

x = [i for i in range(len(test_acc))]
x = x[::10]

print(mtltarget_acc)
input('-------')

np.save('./models1/STL1(4)', target1_acc, allow_pickle=True)
np.save('./models1/MTL12(4)', target_acc, allow_pickle=True)
np.save('./models1/MTL1234(4)', mtltarget_acc, allow_pickle=True)
np.save('./models1/MTL13(4)', mtltarget3_acc, allow_pickle=True)


plt.figure(figsize=(8, 8), dpi=80)
plt.title("compare accuracy")
plt.plot(x, test1_acc[::10], label="STL", color="orange", linewidth=1.5)
plt.plot(x, target_acc[1][1][::10], label="task_1_and_2", color='dodgerblue', linewidth=1.5)
plt.plot(x, mtltarget_acc[1][1][::10], label="task_1-4", color='red', linewidth=1.5)
plt.plot(x, mtltarget3_acc[1][1][::10], label="task_1_and_3", color='purple', linewidth=1.5)
plt.legend(frameon=False, prop={'size': 22})
plt.xlabel('Epochs')
plt.ylabel('% correct') 
plt.legend(frameon=False)
plt.tick_params(axis='both', which='major', labelsize=17)
plt.show()


################################################################################################################################
# SAVE RESULTS



