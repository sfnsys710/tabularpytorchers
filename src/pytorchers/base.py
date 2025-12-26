import torch
from torch import nn

class BaseNNRegressor(nn.Module):
    def __init__(self, input_size, layers=[32, 32, 8], output_size=1):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.layers = layers
        self.output_size = output_size
        for i in range(len(layers)):
            in_features = input_size if i == 0 else layers[i-1]
            out_features = layers[i]
            setattr(self, f"fc{i}", nn.Linear(in_features, out_features))
        self.final_linear = nn.Linear(layers[-1], output_size)
    
    def forward(self, x):
        for i in range(len(self.layers)):
            x = getattr(self, f"fc{i}")(x)
            x = nn.ReLU()(x)
        x = self.final_linear(x)
        return x