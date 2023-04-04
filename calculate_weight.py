import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        return self.linear(x)
    
    def set_weight_bias(self, weight, bias):
        self.linear.weight = weight
        self.linear.bias = bias
