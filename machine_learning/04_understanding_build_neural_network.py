# -*- coding: utf-8 -*-
"""04.Understanding_Build_Neural_Network

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19CrBwnxt9y-pmrmTBjvekFD88kAjdnmp
"""

import torch
import torch.nn as nn

x = torch.FloatTensor(4)
my_linear = nn.Linear(4, 3)
y = my_linear(x)
print(y, y.shape)

for param in my_linear.parameters():
  print(param)

class NeuralNetwork(nn.Module):
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.linear = nn.Linear(input_dim, output_dim)
    
  
  def forward(self, x):
    y = self.linear(x)
    return y

my_linear = NeuralNetwork(4, 3)
y = my_linear(x)
print(y, y.shape)

for param in my_linear.parameters():
  print(param)

