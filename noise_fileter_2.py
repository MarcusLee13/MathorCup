import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import config

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, input_nc, output_nc):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(output_nc),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(output_nc),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, input_nc, output_nc):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(output_nc),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

    def forward(self, x):
        x = self.up(x)
        return x


class noise_filters(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, input_nc, output_nc, use_dropout=True):
        super(noise_filters, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(input_nc, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_nc, kernel_size=1, stride=1, padding=0)

        self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)
        
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        

        d2 = self.Up2(e2)
        # print(e1.shape,d2.shape)
        d2 = torch.cat((e1, d2), dim=1)
        # print(d2.shape)
        d2 = self.Up_conv2(d2)
        # print(d2.shape)

        out = self.Conv(d2)

        d1 = self.active(out)
        # print(d1.shape)
        return d1