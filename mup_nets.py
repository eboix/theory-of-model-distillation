import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

# Setting the ReLUResNet Maximal Update Parametrization manually, without using the Yang-Hu code.

class ReLUResNetMUP_manual(torch.nn.Module):
    def __init__(self,input_length,num_layers,width,output_width, weight_std=0.5, bias_std=0.5):
        super(ReLUResNetMUP_manual, self).__init__()
        self.num_layers = num_layers
        layer_width = []
        layer_width.append(input_length)
        for i in range(num_layers-1):
            layer_width.append(width)
        layer_width.append(1)
        
        linear_list = [nn.Linear(layer_width[i], layer_width[i+1],bias=True) for i in range(num_layers-1)]
        linear_list.append(nn.Linear(layer_width[-2], layer_width[-1]))
        self.linears = nn.ModuleList(linear_list)
        
        
        # Initialize
        for i, l in enumerate(self.linears):
            if i == 0:
                bwi = 0.5
                bbi = 0.5
            elif i == len(self.linears) - 1:
                bwi = 0.5
                bbi = 0
            else:
                bwi = 0.5
                bbi = 0.5
            
            if i > 0:
                torch.nn.init.normal_(l.weight, std=weight_std * math.pow(width, -bwi))
                torch.nn.init.normal_(l.bias, std=bias_std * math.pow(width, -bbi))
            else:
                # If first layer, adjust the size of the weight initialization
                torch.nn.init.normal_(l.weight, std=weight_std * math.pow(width, -bwi) * math.pow(input_length,-0.5))
                torch.nn.init.normal_(l.bias, std=bias_std * math.pow(width, -bbi))
            
        self.width = width
        self.num_layers = num_layers
        
        
    
    def forward(self, x, return_activations=None):
        
#         x = torch.reshape(x, (x.shape[0],784))
        
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            
            if i == 0:
                awi = -0.5
                abi = -0.5
            elif i == len(self.linears) - 1:
                awi = 0.5
                abi = 0
            else:
                awi = 0
                abi = -0.5
                
            prevx = x
            x = F.linear(x, math.pow(self.width, -awi) * l.weight, math.pow(self.width,-abi) * l.bias)
            
            
            if not return_activations is None:
                return_activations[('pre',i)] = x.detach()
            
            if i < self.num_layers - 1:
                x = F.relu(x)
                if i > 0:
                    x = x + prevx

            if not return_activations is None:
                return_activations[('post',i)] = x.detach()
#         print(x.shape)
#         x = F.log_softmax(x, dim=1)
        
        return x