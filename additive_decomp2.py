import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


class new_Module(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.sparse1 = True
        self.labeled = True
        
        self.__name__ = "new_Module"

def num_parameter(net):
    val = 0
    if hasattr(net, 'dense_weight'):
        val += net.dense_weight.reshape(-1).shape[0]
    for module in net.children():
        val += num_parameter(module)
    return val

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
def is_labeled(net):
    if hasattr(net, 'labeled'):
        net.labeled = True
    for module in net.children():
        is_labeled(module)

def is_unlabeled(net):
    if hasattr(net, 'labeled'):
        net.labeled = False
    for module in net.children():
        is_unlabeled(module)
'''
def get_norm_sparse1(net):
    val = 0
    if hasattr(net, 'sparse_weight1'):
        val += torch.norm(net.sparse_weight1, 1)# torch.norm(net.sparse_weight1, 1) + torch.norm(net.sparse_weight1, 2)
    for module in net.children():
        val += get_norm_sparse1(module)
    return val

def get_norm_sparse2(net):
    val = 0
    if hasattr(net, 'sparse_weight2'):
        val += torch.norm(net.sparse_weight2, 1) #torch.norm(net.sparse_weight2, 1) + torch.norm(net.sparse_weight2, 2)
    for module in net.children():
        val += get_norm_sparse2(module)
    return val

def get_norm_dense(net):
    val = 0
    if hasattr(net, 'dense_weight'):
        val += torch.norm(net.dense_weight, 2) #torch.norm(net.dense_weight, 2)
    for module in net.children():
        val += get_norm_dense(module)
    return val

class additive_decomposed_weight(new_Module):
    def __init__(self, shape, value=None, scale=1):
        super().__init__()
        
        if value is None: # for weight
            self.dense_weight = nn.Parameter(torch.randn(shape))
            self.sparse_weight1 = nn.Parameter(scale*torch.zeros(shape))
            self.sparse_weight2 = nn.Parameter(scale*torch.randn(shape)) 
            
        else: # for bias or initialization of Norm
            self.dense_weight = nn.Parameter(value*torch.ones(shape))
            self.sparse_weight1 = nn.Parameter(value*torch.zeros(shape))
            self.sparse_weight2 = nn.Parameter(value*torch.zeros(shape)) 
        
        self.dropout = nn.Dropout(0.05)
        
    def forward(self):
        if self.sparse1:
            sparse_weight = self.sparse_weight1
        else:
            sparse_weight = self.sparse_weight2
        
        dense = self.dropout(self.dense_weight)
        sparse = sparse_weight
        weight = dense + sparse
            
        return weight

class add_conv2d(new_Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, scale=1):
        super().__init__()
        
        self.weight = additive_decomposed_weight((out_channels, in_channels, kernel_size, kernel_size), None, scale) # initialize with random
        
        self.stride = stride
        self.padding = padding
        
        if bias:
            self.bias = additive_decomposed_weight((out_channels), 0, scale) # initialize with 0
        else:
            self.bias = None    
    
    def forward(self, x):
        weight = self.weight.forward()
        
        if self.bias is not None:
            bias = self.bias.forward()
            x = F.conv2d(x, weight, bias=bias, stride=self.stride, padding=self.padding)
        else:
            x = F.conv2d(x, weight, bias=None, stride=self.stride, padding=self.padding)
        
        return x

class add_linear(new_Module):
    def __init__(self, in_channels, out_channels, bias=True, scale=1):
        super().__init__()
        
        self.weight = additive_decomposed_weight((out_channels, in_channels), None, scale) # initialize with random
        
        if bias:
            self.bias = additive_decomposed_weight((out_channels), 0, scale) # initialize with 0
        else:
            self.bias = None    
    
    def forward(self, x):
        weight = self.weight.forward()
        
        if self.bias is not None:
            bias = self.bias.forward()
            x = F.linear(x, weight, bias=bias)
        else:
            x = F.linear(x, weight, bias=None)
        
        return x

class add_BN_2d(new_Module):
    def __init__(self, channels, scale=1):
        super().__init__()
        
        self.weight = additive_decomposed_weight((1,channels,1,1), 1, scale) # initialize with 1
        self.bias = additive_decomposed_weight((1,channels,1,1), 0, scale) # initialize with 0
        
        self.bn = nn.BatchNorm2d(channels, affine=False)
        
    def forward(self, x):
        weight = self.weight.forward()
        bias = self.bias.forward()
        
        x = weight * self.bn(x) + bias
        
        return x




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

def create_upconv(in_channels, out_channels, scale, size=None):
    return nn.Sequential(
        nn.Upsample(size=size, mode='nearest')
        , add_conv2d(in_channels,out_channels,3,1,1,False,scale)
        , add_BN_2d(out_channels, scale)
        , nn.GELU()
        , add_conv2d(out_channels,out_channels,3,1,1,False,scale)
        , add_BN_2d(out_channels, scale)
        , nn.GELU()
        )

class Unet(new_Module):
    def __init__(self, in_channels, out_channels, scale=1):
        super().__init__()
        
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.conv_l1 = nn.Sequential(
            add_conv2d(in_channels,filters[0],3,1,1,False,scale)
            , add_BN_2d(filters[0], scale)
            , nn.GELU()
            , add_conv2d(filters[0],filters[0],3,1,1,False,scale)
            , add_BN_2d(filters[0], scale)
            , nn.GELU()
            )

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l2 = nn.Sequential(
            add_conv2d(filters[0],filters[1],3,1,1,False,scale)
            , add_BN_2d(filters[1], scale)
            , nn.GELU()
            , add_conv2d(filters[1],filters[1],3,1,1,False,scale)
            , add_BN_2d(filters[1], scale)
            , nn.GELU()
            )

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l3 = nn.Sequential(
            add_conv2d(filters[1],filters[2],3,1,1,False,scale)
            , add_BN_2d(filters[2], scale)
            , nn.GELU()
            , add_conv2d(filters[2],filters[2],3,1,1,False,scale)
            , add_BN_2d(filters[2], scale)
            , nn.GELU()
            )

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l4 = nn.Sequential(
            add_conv2d(filters[2],filters[3],3,1,1,False,scale)
            , add_BN_2d(filters[3], scale)
            , nn.GELU()
            , add_conv2d(filters[3],filters[3],3,1,1,False,scale)
            , add_BN_2d(filters[3], scale)
            , nn.GELU()
            )

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l5 = nn.Sequential(
            add_conv2d(filters[3],filters[4],3,1,1,False,scale)
            , add_BN_2d(filters[4], scale)
            , nn.GELU()
            , add_conv2d(filters[4],filters[4],3,1,1,False,scale)
            , add_BN_2d(filters[4], scale)
            , nn.GELU()
            )

        self.deconv_u4 = create_upconv(in_channels=filters[4], out_channels=filters[3], scale=scale, size=(32,32))

        self.conv_u4 = nn.Sequential(
            add_conv2d(filters[4],filters[3],3,1,1,False,scale)
            , add_BN_2d(filters[3], scale)
            , nn.GELU()
            , add_conv2d(filters[3],filters[3],3,1,1,False,scale)
            , add_BN_2d(filters[3], scale)
            , nn.GELU()
            )

        self.deconv_u3 = create_upconv(in_channels=filters[3], out_channels=filters[2], scale=scale, size=(64,64))

        self.conv_u3 = nn.Sequential(
            add_conv2d(filters[3],filters[2],3,1,1,False,scale)
            , add_BN_2d(filters[2], scale)
            , nn.GELU()
            , add_conv2d(filters[2],filters[2],3,1,1,False,scale)
            , add_BN_2d(filters[2], scale)
            , nn.GELU()
            )

        self.deconv_u2 = create_upconv(in_channels=filters[2], out_channels=filters[1], scale=scale, size=(128,128))

        self.conv_u2 = nn.Sequential(
            add_conv2d(filters[2],filters[1],3,1,1,False,scale)
            , add_BN_2d(filters[1], scale)
            , nn.GELU()
            , add_conv2d(filters[1],filters[1],3,1,1,False,scale)
            , add_BN_2d(filters[1], scale)
            , nn.GELU()
            )

        self.deconv_u1 = create_upconv(in_channels=filters[1], out_channels=filters[0], scale=scale, size=(256,256))

        self.conv_u1 = nn.Sequential(
            add_conv2d(filters[1],filters[0],3,1,1,False,scale)
            , add_BN_2d(filters[0], scale)
            , nn.GELU()
            , add_conv2d(filters[0],filters[0],3,1,1,False,scale)
            , add_BN_2d(filters[0], scale)
            , nn.GELU()
            )

        self.out = add_conv2d(filters[0],out_channels,1,1,0,True,scale)
        
        if out_channels == 1:
            self.smout = nn.Sigmoid()
        else:
            self.smout = None
        
    def forward(self, x):

        output1 = self.conv_l1(x)
        input2 = self.maxpool1(output1)
        
        output2 = self.conv_l2(input2)
        input3 = self.maxpool2(output2)
        
        output3 = self.conv_l3(input3)
        input4 = self.maxpool3(output3)
        
        output4 = self.conv_l4(input4)
        input5 = self.maxpool4(output4)
        
        output5 = self.conv_l5(input5)
        input6 = self.deconv_u4(output5)
        
        output6 = self.conv_u4(torch.cat((input6, output4), dim=1))
        input7 = self.deconv_u3(output6)
        
        output7 = self.conv_u3(torch.cat((input7, output3), dim=1))
        input8 = self.deconv_u2(output7)
        
        output8 = self.conv_u2(torch.cat((input8, output2), dim=1))
        input9 = self.deconv_u1(output8)
        
        output9 = self.conv_u1(torch.cat((input9, output1), dim=1))
        out = self.out(output9)
        
        if self.smout is not None:
            out = self.smout(out)
        
        return out


    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    a = torch.randn((2,1,256,256)).to(device)
    b = Unet(1,1,0.01).to(device)
    print(num_sparse(b))
    
    b.train()
    sparse_one(b)
    is_labeled(b)
    
    c = b(a)
    print(c.shape)
    
    b.eval()
    sparse_two(b)
    is_unlabeled(b)
    
    d = b(a)
    print(d.shape)

    
    
    