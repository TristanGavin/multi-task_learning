import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def train_model(model, X, labels):
    num_targets = len(labels)

    # optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    # loss_func = torch.nn.MSELoss()
    loss_func = torch.nn.BCELoss()  # Binary cross entropy - (label, probability of being 1)
    max_iters = 500

    # initialize dict to store loss
    train_loss = {}
    for target in range(num_targets):
        train_loss[target] = []

    #train network
    for epoch in range(max_iters):
        for target in range(num_targets):

            # forward pass (batch gds)
            pred= model(X) # make prediction

            if num_targets > 1:
                pred = pred[:, target]

            loss = loss_func(torch.squeeze(pred), labels[target]) 
            
            # backward pass
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # take gradient step.

            train_loss[target].append(loss.item())

            # if epoch % 50 == 0:    # print every 2000 mini-batches
            #     print(f'training loss: {loss}')

        
    #TODO make graphs
    return train_loss

def test_model(model, X, labels):
    num_targets = len(labels)

    loss_func = torch.nn.BCELoss()  # Binary cross entropy - (label, probability of being 1)

    accuracy = {}

    with torch.no_grad():

        for target in range(num_targets):

            pred= model(X) # make prediction

            if num_targets > 1:
                pred = pred[:, target]

            pred = pred.numpy()
            pred = [1 if x > .5 else 0 for x in pred]

            wrong = 0
            for idx, y in enumerate(pred):
                if pred[idx] != labels[target][idx]:
                    wrong += 1 

            accuracy[target] = (len(pred)-wrong) / len(pred)

    #TODO make graphs
    return accuracy 
