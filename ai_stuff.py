import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils  import make_grid
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

transform = transforms.Compose([
    transforms.Resize((176,176)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder("train",transform = transform)
test_data = datasets.ImageFolder("test",transform = transform)

torch.manual_seed(42)

train_loader = DataLoader(train_data,batch_size=10,shuffle=True)
test_loader = DataLoader(test_data,batch_size=10)

class_names = train_data.classes

def big_function(a1,b1,c,d1):
    class ConvolutionalNetwork(nn.Module):
        
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3,a1,3,1)
            self.conv2 = nn.Conv2d(a1,b1,3,1)
            self.fc1 = nn.Linear(42*42*b1,c)
            self.fc2 = nn.Linear(c,d1)
            self.fc3 = nn.Linear(d1,2)
            
        def forward(self,X):
            X = F.relu(self.conv1(X))
            X = F.max_pool2d(X,2,2)
            X = F.relu(self.conv2(X))
            X = F.max_pool2d(X,2,2)
            X = X.view(-1,42*42*b1)
            X = F.relu(self.fc1(X))
            X = F.relu(self.fc2(X))
            X = self.fc3(X)
            
            return F.log_softmax(X,dim = 1)

    torch.manual_seed(101)
    CNNmodel = ConvolutionalNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(CNNmodel.parameters(),lr=0.001)

    import time
    start_time = time.time()

    epochs = 10

    max_trn_batch = 800
    max_tst_batch = 300

    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []

    for i in range(epochs):
        trn_corr = 0
        tst_corr = 0
        
        # Run the training batches
        for d1, (X_train, y_train) in enumerate(train_loader):
            
            # Limit the number of batches
            d1+=1
            
            # Apply the model
            y_pred = CNNmodel(X_train)
            loss = criterion(y_pred, y_train)
    
            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr
            
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print interim results
            if d1%10 == 0:
                print(f'epoch: {i:2}  batch: {d1:4} [{10*d1:6}/8000]  loss: {loss.item():10.8f}  \
    accuracy: {trn_corr.item()*100/(10*d1):7.3f}%')

        train_losses.append(loss)
        train_correct.append(trn_corr)

        # Run the testing batches
        with torch.no_grad():
            for d1, (X_test, y_test) in enumerate(test_loader):
                # Limit the number of batches

                # Apply the model
                y_val = CNNmodel(X_test)

                # Tally the number of correct predictions
                predicted = torch.max(y_val.data, 1)[1] 
                tst_corr += (predicted == y_test).sum()

        loss = criterion(y_val, y_test)
        test_losses.append(loss)
        test_correct.append(tst_corr)

    return float(sum(list(test_losses))/10)