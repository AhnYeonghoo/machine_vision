# -*- coding: utf-8 -*-
"""10_advance_multi_label_classification

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XksLXeAmt7AfTGUhVcNPjRxBKlHKmeyq
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np

train_rawdata = datasets.MNIST(root = 'dataset',
                               train=True,
                               download=True,
                               transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root = 'dataset',
                              train=False,
                              download=True,
                              transform=transforms.ToTensor())
print(len(train_rawdata))
print(len(test_dataset))

validation_rate = 0.2
train_indices, val_indices, _, _ = train_test_split(
    range(len(train_rawdata)),
    train_rawdata.targets,
    stratify=train_rawdata.targets,
    test_size=validation_rate
)

train_dataset = Subset(train_rawdata, train_indices)
validation_dataset = Subset(train_rawdata, val_indices)

print(len(train_dataset))
print(len(validation_dataset))
print(len(test_dataset))

batch_size = 128
train_batches = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_batches = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
test_batches = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

x_train, y_train = next(iter(train_batches))
print(x_train.shape, y_train.shape)

index = 1
x_train[index, :, :, :].shape
x_train[index, :, :, :].numpy().reshape(28, 28).shape

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

plt.figure(figsize=(10, 12))

for index in range(100):
  plt.subplot(10, 10, index + 1)
  plt.axis('off')
  plt.imshow(x_train[index, :, :, :].numpy().reshape(28, 28), cmap='gray')
  plt.title("Class: " + str(y_train[index].item()))

x_train, y_train = next(iter(train_batches))
print(x_train.shape, y_train.shape)
print(x_train.size(0))
print(x_train.view(x_train.size(0), -1).shape)

class FunModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.linear_layers = nn.Sequential (
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, output_dim),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, x):
        y = self.linear_layers(x)
        return y

minibatch_size = 128 # Mini-batch 사이즈는 128 로 설정
input_dim = 28 * 28 # 784
output_dim = 10
model = FunModel(input_dim, output_dim)  

loss_func = nn.NLLLoss() # log softmax 는 NLLLoss() 로 진행해야 함
optimizer = torch.optim.Adam(model.parameters()) # Adam, learning rate 필요없음

from copy import deepcopy

def train_model(model, early_stop, n_epochs, progress_interval):
    
    train_losses, valid_losses, lowest_loss = list(), list(), np.inf

    for epoch in range(n_epochs):
        
        train_loss, valid_loss = 0, 0
        
        # train the model
        model.train() # prep model for training
        for x_minibatch, y_minibatch in train_batches:
            y_minibatch_pred = model(x_minibatch.view(x_minibatch.size(0), -1))
            loss = loss_func(y_minibatch_pred, y_minibatch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_batches)
        train_losses.append(train_loss)      
        
        # validate the model
        model.eval()
        with torch.no_grad():
            for x_minibatch, y_minibatch in val_batches:
                y_minibatch_pred = model(x_minibatch.view(x_minibatch.size(0), -1))
                loss = loss_func(y_minibatch_pred, y_minibatch)
                valid_loss += loss.item()
                
        valid_loss = valid_loss / len(val_batches)
        valid_losses.append(valid_loss)

        if valid_losses[-1] < lowest_loss:
            lowest_loss = valid_losses[-1]
            lowest_epoch = epoch
            best_model = deepcopy(model.state_dict())
        else:
            if (early_stop > 0) and lowest_epoch + early_stop < epoch:
                print ("Early Stopped", epoch, "epochs")
                break
                
        if (epoch % progress_interval) == 0:
            print (train_losses[-1], valid_losses[-1], lowest_loss, lowest_epoch, epoch)
            
    model.load_state_dict(best_model)        
    return model, lowest_loss, train_losses, valid_losses

nb_epochs = 30 
progress_interval = 3
early_stop = 10

model, lowest_loss, train_losses, valid_losses = train_model(model, early_stop, nb_epochs, progress_interval)

valid_losses

test_loss = 0
correct = 0
wrong_samples, wrong_preds, actual_preds = list(), list(), list()

model.eval()
with torch.no_grad():
    for x_minibatch, y_minibatch in test_batches:
        y_test_pred = model(x_minibatch.view(x_minibatch.size(0), -1))
        test_loss += loss_func(y_test_pred, y_minibatch)  
        pred = torch.argmax(y_test_pred, dim=1)
        correct += pred.eq(y_minibatch).sum().item()
        
        wrong_idx = pred.ne(y_minibatch).nonzero()[:, 0].numpy().tolist()
        for index in wrong_idx:
            wrong_samples.append(x_minibatch[index])
            wrong_preds.append(pred[index])
            actual_preds.append(y_minibatch[index])
            
test_loss /= len(test_batches.dataset)
print('Average Test Loss: {:.4f}'.format( test_loss ))
print('Accuracy: {}/{} ({:.2f}%)'.format( correct, len(test_batches.dataset), 100 * correct / len(test_batches.dataset) ))

plt.figure(figsize=(18 , 20))

for index in range(100):
    plt.subplot(10, 10, index + 1)
    plt.axis('off')
    plt.imshow(wrong_samples[index].numpy( ).reshape(28,28), cmap = "gray")
    plt.title("Pred" + str(wrong_preds[index].item()) + "(" + str(actual_preds[index].item()) + ")", color='red')

