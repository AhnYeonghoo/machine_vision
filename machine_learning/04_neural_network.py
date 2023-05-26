import torch
import numpy as np
import pandas as pd


print(torch.__version__)
print(np.__version__)
print(pd.__version__)

x = torch.FloatTensor(4) # 입력
W = torch.FloatTensor(4, 3) # 가중치
b = torch.FloatTensor(3) # 편향

# linear_layout function
def linear_function(x, W, b):
    y = torch.matmul(x, W) + b
    return y

print(x.shape)
print(W.shape)
print(b.shape)

y = linear_function(x, W, b)
print(y.shape)

import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim)) # __init__()에서 초기화 진행 weight
        self.b = nn.Parameter(torch.FloatTensor(output_dim)) # bias
        
    def forward(self, x):
        y = torch.matmul(x, self.W) + self.b 
        return y
    
  
x = torch.FloatTensor(5)
my_linear = NeuralNetwork(5, 3)
# forward에 넣을 인자값으로 호출하면, 내부적으로 forward() 메서드를 자동으로 호출함
# 내부 처리중 forward() 전처리/후처리도 수행함

y = my_linear(x) # 알아서 포워드 진행함
print(y, y.shape)

x = torch.FloatTensor(4)
my_linear = NeuralNetwork(4, 3)
y = my_linear(x)
print(y, y.shape)

for param in my_linear.parameters():
    print(param)
