import torch
from torch import nn
"""
layers = [
    layer1 = {
        {'type': 'conv2d', 'in_channels': 3, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'type': 'activation', 'activation': 'relu'},
    }
}
"""

class Student(nn.Module):
    def __init__(self, layers: list[dict[dict]], teacher_dim: int, num_classes: int):
        super(Student, self).__init__()
        self.teacher_dim = teacher_dim
        self.num_classes = num_classes

        for layer in layers:
            for name, attribute in layer.items():
                setattr(self, name, nn.Sequential())
                layer_sequential = getattr(self,f'layer_{name}')
                
                if attribute['type'] == 'conv2d':
                    layer_sequential.append(nn.Conv2d(in_channels=attribute['in_channels'],
                                                      out_channels=attribute['out_channels'],
                                                      kernel_size=attribute['kernel_size'],
                                                      stride=attribute['stride'],
                                                      padding=attribute['padding']))
                    
                    layer_sequential.append(nn.BatchNorm2d(num_features=attribute['out_channels']))

                elif attribute['type'] == 'activation':
                    if attribute['activation'] == 'relu':
                        layer_sequential.append(nn.ReLU())
                    elif attribute['activation'] == 'gelu':
                        layer_sequential.append(nn.GELU())
                    elif attribute['activation'] == 'tanh':
                        layer_sequential.append(nn.Tanh())

                elif attribute['type'] == 'MaxPool2d':
                    layer_sequential.append(nn.MaxPool2d(kernel_size=attribute['kernel_size'],
                                                         stride=attribute['stride'],
                                                         padding=attribute['padding']))
                
                elif attribute['type'] == 'Dropout':
                    layer_sequential.append(nn.Dropout(p=attribute['p']))
                
                elif attribute['type'] == 'linear':
                    layer_sequential.append(nn.Linear(in_features=attribute['in_features'],
                                                      out_features=attribute['out_features']))
                    
                elif attribute['type'] == 'flatten':
                    layer_sequential.append(nn.Flatten())
                    
                elif attribute['type'] == 'AdaptiveAvgPool2d':
                    layer_sequential.append(nn.AdaptiveAvgPool2d(output_size=attribute['output_size']))
    
        def encode(self, x):
            for name, _ in self._modules.items():
                layer_sequential = getattr(self, f'layer_{name}')
                if isinstance(layer_sequential, nn.Sequential) and name not in ['classifier', 'project']:
                    x = layer_sequential(x)
            return x
    
        def project(self, x):
            layer = getattr(self, 'layer_project')
            return layer(self.encode(x))
        
        def forward(self, x):
            layer = getattr(self, 'layer_classifier')
            return layer(self.project(x))
    


        