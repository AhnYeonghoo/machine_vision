# -*- coding: utf-8 -*-
"""05.Gradient_Descent_Autograd

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VJ9P-IzzwyJGynsTg4cYiw02Z4OEJ0IJ
"""

import torch
import torch.nn as nn

x = torch.rand(1, requires_grad=True)
y = torch.rand(1)
y.requires_grad=True
loss = y - x
print(loss)

loss.backward()
print(x.grad, y.grad)

x = torch.ones(4)
y = torch.zeros(3)
W = torch.rand(4, 3, requires_grad=True)
b = torch.rand(3, requires_grad=True)
z = torch.matmul(x,W) + b
z

import torch.nn.functional as F

loss = F.mse_loss(z, y)
loss.backward()
print(loss, W.grad, b.grad)

threshold = 0.1
learning_rate = 0.1
iteration_num = 0

while loss > threshold:
  iteration_num += 1
  W = W - learning_rate * W.grad
  b = b - learning_rate * b.grad
  print(iteration_num, loss, z, y)
  
  W.detach_().requires_grad_(True)
  b.detach_().requires_grad_(True)
  
  z = torch.matmul(x, W) + b
  loss = F.mse_loss(z, y)
  loss.backward()

print(iteration_num + 1, loss, z, y)

w = torch.tensor(4.0, requires_grad=True)
z = 2 * w
z.backward()
print(w.grad)

z = 2 * w
z.backward()
print(w.grad)

z = 2 * w
z.backward()
print(w.grad)

x = torch.ones(4)
y = torch.zeros(3)
W = torch.rand(4, 3, requires_grad=True)
b = torch.rand(3, requires_grad=True)
learning_rate = 0.01
optimizer = torch.optim.SGD([W,b], lr=learning_rate)

nb_epochs = 300
for epoch in range(nb_epochs + 1):
  z = torch.matmul(x, W) + b
  loss = F.mse_loss(z, y)
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  if epoch % 100 == 0:
    print(epoch, nb_epochs, W, b, loss)

class LinearRegressionModel(nn.Module):
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.linear = nn.Linear(input_dim, output_dim)
    
  def forward(self, x):
    return self.linear(x)
  

model = LinearRegressionModel(4, 3)

x = torch.ones(4)
y = torch.zeros(3)

learning_rate = 0.01
nb_epochs = 1000
opimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(nb_epochs + 1):
  
  pred = model(x)
  loss = F.mse_loss(pred, y)
  
  opimizer.zero_grad()
  loss.backward()
  opimizer.step()

print(loss)
for param in model.parameters():
  print(param)

