import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import config

class conv_block1(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, input_nc, output_nc):
        super(conv_block1, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2,inplace=True)
            )

    def forward(self, x):

        x = self.conv(x)
        return x

class conv_block2(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, input_nc, output_nc):
        super(conv_block2, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(output_nc),
            nn.LeakyReLU(0.2,True)
        )

    def forward(self, x):

        x = self.conv(x)
        return x

class conv_block3(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, input_nc, output_nc):
        super(conv_block3, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def forward(self, x):

        x = self.conv(x)
        return x





class Filter_net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, input_nc, output_nc, num_downs, nhf=64, norm_type='none', use_dropout=True, output_function='sigmoid'):
        super(Filter_net, self).__init__()

        

        self.Conv1 = conv_block1(input_nc, 96)
        self.Conv2 = conv_block2(96, 96)
        self.Conv3 = conv_block3(96, 96)

        self.Conv = nn.Conv2d(96, output_nc, kernel_size=1, stride=1, padding=0)
        self.active = torch.nn.Sigmoid()


    def forward(self, x):
        # print(x.shape)
        e1 = self.Conv1(x)
        
        e2 = self.Conv2(e1)

        e3 = self.Conv3(e2)

        out = self.Conv(e3)

        # out = self.active(out)
        return out