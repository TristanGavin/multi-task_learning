import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

def train_model(model, dataloaders, targets):

    # optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # loss_func = torch.nn.MSELoss()
    loss_func = torch.nn.BCELoss()  # Binary cross entropy - (label, probability of being 1)
    max_iters = 5000

    train_loader = dataloaders[0]
    test_loader = dataloaders[1]

    train_acc  = []
    test_acc = []
    stop = 0 # for early stopping criteria
    max_acc = 0


    # train network
    for epoch in range(max_iters):
        for i, (inputs, y1, y2, y3, y4) in enumerate(train_loader):
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


        # evaluate model on train
        model.eval()
        with torch.no_grad():
            for i, (inputs, y1, y2, y3, y4) in enumerate(train_loader):
                data = (inputs, y1, y2, y3, y4)

                # forward pass (batch gds)
                y_pred = model(inputs)      
                y_pred_cls = y_pred.round() # round all targets to 1 or 0
                
                # reshape data for multi targets
                if len(targets) > 1:
                    labels = [data[idx] for idx in targets]
                    labels = torch.cat(labels, 1)
                    # calculate acc
                    tr_acc = y_pred_cls.eq(labels).sum() / (float(labels.shape[0]) * 2)
                    train_acc.append(tr_acc)
                else:
                    labels = data[targets[0]]
                    tr_acc = y_pred_cls.eq(labels).sum() / float(labels.shape[0]) 
                    train_acc.append(tr_acc)


        # evaluate model on test
        model.eval()
        with torch.no_grad():
            for i, (inputs, y1, y2, y3, y4) in enumerate(test_loader):
                data = (inputs, y1, y2, y3, y4)

                y_pred = model(inputs)      
                y_pred_cls = y_pred.round() # round all targets to 1 or 0

                #TODO seperate accuracy for each target
                # reshape data for multi targets
                if len(targets) > 1:
                    labels = [data[idx] for idx in targets]
                    labels = torch.cat(labels, 1)
                    te_acc = y_pred_cls.eq(labels).sum() / (float(labels.shape[0]) * 2)
                    test_acc.append(te_acc)
                else:
                    labels = data[targets[0]]
                    te_acc = y_pred_cls.eq(labels).sum() / float(labels.shape[0]) 
                    test_acc.append(te_acc)

        # early stopping criteria
        if te_acc <= max_acc:
            stop += 1
        else:
            max_acc = te_acc
            stop = 0
        
        if (epoch > 500) and (stop == 500):
            break
           
        
    #TODO display the loss
    plt.figure(figsize=(8, 8), dpi=80)
    plt.plot(train_acc, label="train accuracy", color='dodgerblue', linewidth=1.5)
    plt.plot(test_acc, label="test accuracy", color='red', linewidth=1.5)
    plt.legend(frameon=False, prop={'size': 22})
    plt.xlabel('Epochs')
    plt.ylabel('% correct') 
    plt.legend(frameon=False)
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.savefig('./graphs/loss_test.png')
    plt.show()

    return test_acc

def test_model(model, X, labels):

    loss_func = torch.nn.BCELoss()  # Binary cross entropy - (label, probability of being 1)
    accuracy = {}
    with torch.no_grad():
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
