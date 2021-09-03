import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def train_model(model, dataloader, targets):

    # optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    # loss_func = torch.nn.MSELoss()
    loss_func = torch.nn.BCELoss()  # Binary cross entropy - (label, probability of being 1)
    max_iters = 500

    #train network
    for epoch in range(max_iters):
        for i, (inputs, y1, y2, y3, y4) in enumerate(dataloader):
            data = (inputs, y1, y2, y3, y4)

            # forward pass (batch gds)
            pred = model(inputs) # make prediction
            
            # reshape data for multi targets
            if len(targets) > 1:
                labels = [data[idx] for idx in targets]
                labels = torch.cat(labels, 1)
            else:
                labels = data[targets[0]]
                
            # calculate loss
            loss = loss_func(pred, labels) 
            
            # backward pass
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # take gradient step.
        
    #TODO display the loss
    # check model against test every so often or something.
    return loss

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
