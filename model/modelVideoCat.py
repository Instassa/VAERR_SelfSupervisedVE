import math

import numpy as np
import torch
import torch.nn as nn
from model.gru import GRU
from model.resnet import ResNet, BasicBlock


def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch*s_time, n_channels, sx, sy)

class CategoricalOnlyEmotionModelVideo(nn.Module):
    
    def __init__(
        self,
        inputDim: int = 512,
        hiddenDim: int = 256,
        n_classes: int = 6,
        fine_tuning: str = "FT",
        every_frame: bool = False,
        device=None,
        relu_type: str = "relu",
        gamma_zero: bool = False,
        avg_pool_downsample: bool = False,
    ):

        super().__init__()
        self.relu_type = relu_type
        print('INPUT: ', inputDim)
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.n_classes = n_classes
        self.every_frame = every_frame
        self.device = device
        self.fine_tuning = fine_tuning
    

        self.frontend_nout = 64
        self.backend_out = 512
        
        self.trunk = ResNet(
            BasicBlock,
            [2, 2, 2, 2],
            num_classes=self.inputDim,
            relu_type=relu_type,
            gamma_zero=gamma_zero,
            avg_pool_downsample=avg_pool_downsample,
        )
        if self.fine_tuning == 'FullyFrozen':
            print('FULLY_FROZEN')
            for parameter in self.trunk.parameters():
                parameter.requires_grad = False

        frontend_relu = nn.PReLU(num_parameters=self.frontend_nout) if relu_type == 'prelu' else nn.ReLU()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d( kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        
        if 'Frozen' in self.fine_tuning:
            print('FROZEN 3D conv layer')   
            for parameter in self.frontend3D.parameters():
                parameter.requires_grad = False
                 

        print('EVERY FRAME: ', self.every_frame)
        self.n_layers = 2
        self.backend_out = 512
        self.gru = GRU(
            self.backend_out,
            self.hiddenDim,
            self.n_layers,
            self.n_classes,
            self.every_frame,
            device=self.device,
        )

        self._initialize_weights_randomly()

    def forward(self, x, lengths):
        
        x = x.float()
        B, C, T, H = x.size()
        x = x.view(B, 1, C, T, H)
        x = self.frontend3D(x)
        Tnew = x.shape[2] 
        
        x = threeD_to_2D_tensor( x )
        x = self.trunk(x)
        x = x.view(B, Tnew, x.size(1))
        
        x_cat = self.gru(x, lengths)
        # print(x_cat.size())
        if self.every_frame == False:
            x = torch.softmax(x_cat, dim=1)
        else:
            x = torch.softmax(x_cat, dim=2)
        return x

    def _initialize_weights_randomly(self):

        use_sqrt = True

        if use_sqrt:

            def f(n):
                return math.sqrt(2 / n)

        else:

            def f(n):
                return 2 / n

        for m in self.modules():
            if (
                isinstance(m, nn.Conv3d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv1d)
            ):
                n = np.prod(m.kernel_size) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif (
                isinstance(m, nn.BatchNorm3d)
                or isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.BatchNorm1d)
            ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))