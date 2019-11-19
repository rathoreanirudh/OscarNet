import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):

    
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.conv1 = self.__conv(2, name='conv1', in_channels=3, out_channels=96, kernel_size=(11, 11), stride=(4, 4), groups=1, bias=True)
        self.conv2 = self.__conv(2, name='conv2', in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(1, 1), groups=2, bias=True)
        self.conv3 = self.__conv(2, name='conv3', in_channels=256, out_channels=384, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv4 = self.__conv(2, name='conv4', in_channels=384, out_channels=384, kernel_size=(3, 3), stride=(1, 1), groups=2, bias=True)
        self.conv5 = self.__conv(2, name='conv5', in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=2, bias=True)
        self.fc6_gb_conv = self.__conv(2, name='fc6_gb_conv', in_channels=256, out_channels=512, kernel_size=(6, 6), stride=(2, 2), groups=1, bias=True)
        self.fc7_gb_conv = self.__conv(2, name='fc7_gb_conv', in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.fc8_gb_conv = self.__conv(2, name='fc8_gb_conv', in_channels=256, out_channels=2, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)

    def forward(self, x):
        conv1_pad       = F.pad(x, (0, 1, 0, 1))
        conv1           = self.conv1(conv1_pad)
        relu1           = F.relu(conv1)
        pool1_pad       = F.pad(relu1, (0, 1, 0, 1), value=float('-inf'))
        pool1           = F.max_pool2d(pool1_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        conv2_pad       = F.pad(pool1, (2, 2, 2, 2))
        conv2           = self.conv2(conv2_pad)
        relu2           = F.relu(conv2)
        pool2_pad       = F.pad(relu2, (0, 1, 0, 1), value=float('-inf'))
        pool2           = F.max_pool2d(pool2_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        conv3_pad       = F.pad(pool2, (1, 1, 1, 1))
        conv3           = self.conv3(conv3_pad)
        relu3           = F.relu(conv3)
        conv4_pad       = F.pad(relu3, (1, 1, 1, 1))
        conv4           = self.conv4(conv4_pad)
        relu4           = F.relu(conv4)
        conv5_pad       = F.pad(relu4, (1, 1, 1, 1))
        conv5           = self.conv5(conv5_pad)
        relu5           = F.relu(conv5)
        pool5_pad       = F.pad(relu5, (0, 1, 0, 1), value=float('-inf'))
        pool5           = F.max_pool2d(pool5_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        fc6_gb_conv_pad = F.pad(pool5, (0, 1, 0, 1))
        fc6_gb_conv     = self.fc6_gb_conv(fc6_gb_conv_pad)
        relu6           = F.relu(fc6_gb_conv)
        drop6           = F.dropout(input = relu6, p = 0.5, training = self.training, inplace = True)
        fc7_gb_conv     = self.fc7_gb_conv(drop6)
        relu7           = F.relu(fc7_gb_conv)
        drop7           = F.dropout(input = relu7, p = 0.5, training = self.training, inplace = True)
        fc8_gb_conv     = self.fc8_gb_conv(drop7)
        prob            = F.softmax(fc8_gb_conv)
        return prob


    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

