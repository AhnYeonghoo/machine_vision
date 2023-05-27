# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

df_train.head(10)

print(df_train.shape, df_test.shape)

df_train.head()

df_test.head()

df_train.corr()["SalePrice"]

feature_list = df_train.corr()['SalePrice'][(df_train.corr(
)['SalePrice'] > 0.5) | (df_train.corr()['SalePrice'] < -0.5)].index

df_train[feature_list].isnull().sum()

features = list(feature_list[:-1])
features

for feature in features:
    df_test[feature].fillna((df_test[feature].mean()), inplace=True)

X_train = df_train[features]
Y_train = df_train[['SalePrice']].values
X_test = df_test[features]

type(Y_train)

std_scaler = StandardScaler()
std_scaler.fit(X_train)
X_train_tensor = torch.from_numpy(std_scaler.transform(X_train)).float()
X_test_tensor = torch.from_numpy(std_scaler.transform(X_test)).float()
y_train_tensor = torch.from_numpy(Y_train).float()

print(X_train_tensor.shape, X_test_tensor.shape, y_train_tensor.shape)

epochs = 10000
batch_size = 256


class FunModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 6),
            nn.LeakyReLU(),
            nn.Linear(6, output_dim)
        )

    def forward(self, x):
        y = self.linear_layers(x)
        return y


input_dim = X_train_tensor.size(-1)
output_dim = y_train_tensor.size(-1)
print(input_dim, output_dim)
model = FunModel(input_dim, output_dim)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

indices = torch.randperm(X_train_tensor.size(0))
print(indices)
x_batch_list = torch.index_select(X_train_tensor, 0, index=indices)
y_batch_list = torch.index_select(y_train_tensor, 0, index=indices)
x_batch_list = x_batch_list.split(batch_size, 0)
y_batch_list = y_batch_list.split(batch_size, 0)

for index in range(epochs):
    indices = torch.randperm(X_train_tensor.size(0))

    x_batch_list = torch.index_select(X_train_tensor, 0, index=indices)
    y_batch_list = torch.index_select(y_train_tensor, 0, index=indices)
    x_batch_list = x_batch_list.split(batch_size, 0)
    y_batch_list = y_batch_list.split(batch_size, 0)

    epoch_loss = list()
    for x_batch, y_batch in zip(x_batch_list, y_batch_list):
        y_batch_pred = model(x_batch)

        loss = torch.sqrt(loss_function(y_batch_pred, y_batch))
        epoch_loss.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (index % 100) == 0:
        print(index, sum(epoch_loss) / len(epoch_loss))

print(loss)
