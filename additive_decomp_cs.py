import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np

class new_Module(nn.Module):
    def __init__(self):
        super().__init__()
        
        #self.sparse1 = True
        
        self.__name__ = "new_Module"

def num_parameter(net):
    val = 0
    if hasattr(net, 'dense_weight'):
        val += net.dense_weight.reshape(-1).shape[0]
    for module in net.children():
        val += num_parameter(module)
    return val
'''
def sparse_one(net):
    if hasattr(net, 'sparse1'):
        net.sparse1 = True
    for module in net.children():
        sparse_one(module)

def sparse_two(net):
    if hasattr(net, 'sparse1'):
        net.sparse1 = False
    for module in net.children():
        sparse_two(module)
'''
def get_norm_dense(net):
    val = 0
    if hasattr(net, 'dense_weight'):
        val += torch.norm(net.dense_weight, 1)
    for module in net.children():
        val += get_norm_dense(module)
    return val

def weight_compressed_sensing(x, prob=None, var=1):
    if prob is not None:
        matrix = torch.where(torch.rand(x.shape) < prob, torch.normal(1, var, x.shape), torch.ones(x.shape)).to(x.device)
        x = matrix * x
        
    return x

class compressed_sensing_weight(new_Module):
    def __init__(self, shape, value=None):
        super().__init__()
        
        if value is None: # for weight
            self.dense_weight = nn.Parameter(torch.randn(shape))
            nn.init.kaiming_normal_(self.dense_weight, mode='fan_out', nonlinearity='relu')
            
        else: # for bias or initialization of Norm
            self.dense_weight = nn.Parameter(value*torch.ones(shape))
        
    def forward(self, prob, var):
        weight = weight_compressed_sensing(self.dense_weight, prob, var)
            
        return weight

class add_conv2d(new_Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        
        self.weight = compressed_sensing_weight((out_channels, in_channels, kernel_size, kernel_size), None) # initialize with random
        
        self.stride = stride
        self.padding = padding
        
        if bias:
            self.bias = compressed_sensing_weight((out_channels), 1e-6) # initialize with 0
        else:
            self.bias = None    
    
    def forward(self, x_prob):
        x, prob, var = x_prob
        weight = self.weight.forward(prob, var)
        
        if self.bias is not None:
            bias = self.bias.forward(prob, var)
            x = F.conv2d(x, weight, bias=bias, stride=self.stride, padding=self.padding)
        else:
            x = F.conv2d(x, weight, bias=None, stride=self.stride, padding=self.padding)
        
        return (x, prob, var)

class add_linear(new_Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        
        self.weight = compressed_sensing_weight((out_channels, in_channels), None) # initialize with random
        
        if bias:
            self.bias = compressed_sensing_weight((out_channels), 1e-6) # initialize with 0
        else:
            self.bias = None    
    
    def forward(self, x_prob):
        x, prob, var = x_prob
        weight = self.weight.forward(prob, var)
        
        if self.bias is not None:
            bias = self.bias.forward(prob, var)
            x = F.linear(x, weight, bias=bias)
        else:
            x = F.linear(x, weight, bias=None)
        
        return (x, prob, var)

class add_BN_2d(new_Module):
    def __init__(self, channels):
        super().__init__()
        
        self.weight = compressed_sensing_weight((1,channels,1,1), 1) # initialize with 1
        self.bias = compressed_sensing_weight((1,channels,1,1), 1e-6) # initialize with 0
        
        self.bn = nn.BatchNorm2d(channels, affine=False)
        
    def forward(self, x_prob):
        x, prob, var = x_prob
        weight = self.weight.forward(prob, var)
        bias = self.bias.forward(prob, var)
        
        x = weight * self.bn(x) + bias
        
        return (x, prob, var)


class add_gelu(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.gelu = nn.GELU()
        
    def forward(self, x_prob):
        x, prob, var = x_prob
        x = self.gelu(x)
        return (x, prob, var)


class add_maxpool(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        
    def forward(self, x_prob):
        x, prob, var = x_prob
        x = self.pool(x)
        return (x, prob, var)


class add_upsample(nn.Module):
    def __init__(self, size, mode):
        super().__init__()
        
        self.upsample = nn.Upsample(size=size, mode=mode)
        
    def forward(self, x_prob):
        x, prob, var = x_prob
        x = self.upsample(x)
        return (x, prob, var)

'''
class net1(new_Module):
    def __init__(self, scale=0.01):
        super().__init__()
        
        self.conv = add_conv2d(3,4,3,1,1,False,scale)
        self.fc = add_linear(64,5,True,scale)
        self.bn = add_BN_2d(4,scale)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        
        return x
    
if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    a = torch.randn((2,3,4,4)).to(device)
    b = net1().to(device)
    
    b.train()
    sparse_one(b)
    is_labeled(b)
    
    c = b(a)
    print(c.shape)
    print(c)
    print(b.bn.bn.running_mean)
    
    b.eval()
    sparse_two(b)
    is_unlabeled(b)
    
    d = b(a)
    print(d.shape)
    print(d)
    print(b.bn.bn.running_mean)
    
    
    print("="*10)
    e = additive_decomposed_weight((5), None, 1)
    e.train()
    sparse_one(e)
    is_labeled(e)
    print(e.forward())
    
    e.eval()
    sparse_two(e)
    is_unlabeled(e)
    print(e.forward())
    
'''

def create_upconv(in_channels, out_channels, size=None):
    return nn.Sequential(
        add_upsample(size=size, mode='nearest')
        , add_conv2d(in_channels,out_channels,3,1,1,False)
        , add_BN_2d(out_channels)
        , add_gelu()
        , add_conv2d(out_channels,out_channels,3,1,1,False)
        , add_BN_2d(out_channels)
        , add_gelu()
        )

class Unet(new_Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.conv_l1 = nn.Sequential(
            add_conv2d(in_channels,filters[0],3,1,1,False)
            , add_BN_2d(filters[0])
            , add_gelu()
            , add_conv2d(filters[0],filters[0],3,1,1,False)
            , add_BN_2d(filters[0])
            , add_gelu()
            )

        self.maxpool1 = add_maxpool(kernel_size=2, stride=2)

        self.conv_l2 = nn.Sequential(
            add_conv2d(filters[0],filters[1],3,1,1,False)
            , add_BN_2d(filters[1])
            , add_gelu()
            , add_conv2d(filters[1],filters[1],3,1,1,False)
            , add_BN_2d(filters[1])
            , add_gelu()
            )

        self.maxpool2 = add_maxpool(kernel_size=2, stride=2)

        self.conv_l3 = nn.Sequential(
            add_conv2d(filters[1],filters[2],3,1,1,False)
            , add_BN_2d(filters[2])
            , add_gelu()
            , add_conv2d(filters[2],filters[2],3,1,1,False)
            , add_BN_2d(filters[2])
            , add_gelu()
            )

        self.maxpool3 = add_maxpool(kernel_size=2, stride=2)

        self.conv_l4 = nn.Sequential(
            add_conv2d(filters[2],filters[3],3,1,1,False)
            , add_BN_2d(filters[3])
            , add_gelu()
            , add_conv2d(filters[3],filters[3],3,1,1,False)
            , add_BN_2d(filters[3])
            , add_gelu()
            )

        self.maxpool4 = add_maxpool(kernel_size=2, stride=2)

        self.conv_l5 = nn.Sequential(
            add_conv2d(filters[3],filters[4],3,1,1,False)
            , add_BN_2d(filters[4])
            , add_gelu()
            , add_conv2d(filters[4],filters[4],3,1,1,False)
            , add_BN_2d(filters[4])
            , add_gelu()
            )

        self.deconv_u4 = create_upconv(in_channels=filters[4], out_channels=filters[3], size=(32,32))

        self.conv_u4 = nn.Sequential(
            add_conv2d(filters[4],filters[3],3,1,1,False)
            , add_BN_2d(filters[3])
            , add_gelu()
            , add_conv2d(filters[3],filters[3],3,1,1,False)
            , add_BN_2d(filters[3])
            , add_gelu()
            )

        self.deconv_u3 = create_upconv(in_channels=filters[3], out_channels=filters[2], size=(64,64))

        self.conv_u3 = nn.Sequential(
            add_conv2d(filters[3],filters[2],3,1,1,False)
            , add_BN_2d(filters[2])
            , add_gelu()
            , add_conv2d(filters[2],filters[2],3,1,1,False)
            , add_BN_2d(filters[2])
            , add_gelu()
            )

        self.deconv_u2 = create_upconv(in_channels=filters[2], out_channels=filters[1], size=(128,128))

        self.conv_u2 = nn.Sequential(
            add_conv2d(filters[2],filters[1],3,1,1,False)
            , add_BN_2d(filters[1])
            , add_gelu()
            , add_conv2d(filters[1],filters[1],3,1,1,False)
            , add_BN_2d(filters[1])
            , add_gelu()
            )

        self.deconv_u1 = create_upconv(in_channels=filters[1], out_channels=filters[0], size=(256,256))

        self.conv_u1 = nn.Sequential(
            add_conv2d(filters[1],filters[0],3,1,1,False)
            , add_BN_2d(filters[0])
            , add_gelu()
            , add_conv2d(filters[0],filters[0],3,1,1,False)
            , add_BN_2d(filters[0])
            , add_gelu()
            )

        self.out = add_conv2d(filters[0],out_channels,1,1,0,True)
        
        if out_channels == 1:
            self.smout = nn.Sigmoid()
        else:
            self.smout = None
        
    def forward(self, x_prob):
        _, prob, var = x_prob

        output1 = self.conv_l1(x_prob)
        input2 = self.maxpool1(output1)
        
        output2 = self.conv_l2(input2)
        input3 = self.maxpool2(output2)
        
        output3 = self.conv_l3(input3)
        input4 = self.maxpool3(output3)
        
        output4 = self.conv_l4(input4)
        input5 = self.maxpool4(output4)
        
        output5 = self.conv_l5(input5)
        input6 = self.deconv_u4(output5)
        
        output6 = self.conv_u4((torch.cat((input6[0], output4[0]), dim=1), prob, var))
        input7 = self.deconv_u3(output6)
        
        output7 = self.conv_u3((torch.cat((input7[0], output3[0]), dim=1), prob, var))
        input8 = self.deconv_u2(output7)
        
        output8 = self.conv_u2((torch.cat((input8[0], output2[0]), dim=1), prob, var))
        input9 = self.deconv_u1(output8)
        
        output9 = self.conv_u1((torch.cat((input9[0], output1[0]), dim=1), prob, var))
        out, _, _ = self.out(output9)
        
        if self.smout is not None:
            out = self.smout(out)
        
        return out


    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    a = torch.randn((2,1,256,256)).to(device)
    b = Unet(1,1).to(device)    
    
    c = b((a, 0.2, 1))
    print(c.shape)

    
    
    