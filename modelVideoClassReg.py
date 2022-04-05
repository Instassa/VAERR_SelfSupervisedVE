import math

import numpy as np
import torch
import torch.nn as nn
from gru import GRU
from resnet import ResNet, BasicBlock


def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch*s_time, n_channels, sx, sy)

class EmotionModelVideo(nn.Module):
    
    def __init__(
        self,
        inputDim: int = 512,
        hiddenDim: int = 256,
        n_classes: int = 500,
        every_frame: bool = True,
        device=None,
        relu_type: str = "relu",
        gamma_zero: bool = False,
        avg_pool_downsample: bool = False,
    ):

        super().__init__()
        self.relu_type = relu_type

        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.n_classes = n_classes
        self.every_frame = every_frame
        self.device = device
        self.bins=20


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
        # for parameter in self.trunk.parameters():
        #     parameter.requires_grad = False
        #     print('FROZEN 2')

        frontend_relu = nn.PReLU(num_parameters=self.frontend_nout) if relu_type == 'prelu' else nn.ReLU()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d( kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        
        for parameter in self.frontend3D.parameters():
            parameter.requires_grad = False
            print('FROZEN 1')        

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
        self.gru2 = GRU(
            self.backend_out,
            self.hiddenDim,
            self.n_layers,
            self.n_classes,
            self.every_frame,
            device=self.device,
        )
        self._initialize_weights_randomly()

    def forward(self, x, lengths):
        # print('output_shape -3:', np.shape(x))
        x = x.float()
        B, C, T, H = x.size()
        x = x.view(B, 1, C, T, H)
        x = self.frontend3D(x)
        Tnew = x.shape[2]    # outpu should be B x C2 x Tnew x H x W
        x = threeD_to_2D_tensor( x )
        x = self.trunk(x)
        x = x.view(B, Tnew, x.size(1))
        # print('output_shape -1:', np.shape(x))
        # B, T = x.size()

        # x = self.trunk(x[:, None, :])
        # x = x.transpose(1, 2)
        # lengths = [i // 320 for i in lengths]  # sewa
        # print('params:',             self.backend_out,
        #     self.hiddenDim,
        #     self.n_layers,
        #     self.n_classes,
        #     self.every_frame,)
        x = self.gru(x, lengths)
        # x2 = self.gru2(x, lengths)
        # print('output_shape 0:', np.shape(x))
        # print('output_shape 1:', np.shape((torch.split(x, 7, dim=2)[0])))
        x1, x1_reg, x2, x2_reg = torch.split(x, [self.bins, 1, self.bins, 1], dim=2)
        # print('output_shape 1:', np.shape(x1))
        x1 = torch.softmax(x1, dim=2)
        x_reg = torch.cat((x1_reg, x2_reg), dim=2)
        x1_reg, x2_reg = torch.split(torch.sigmoid(x_reg) * 2 - 1.0, [1,1], dim=2)
        # x1_reg = torch.sigmoid(x1_reg) * 2 - 1.0
        # print(np.shape(x1_reg))
        x2 = torch.softmax(x2, dim=2)
        # x2_reg = torch.sigmoid(x2_reg) * 2 - 1.0
        # print('output_shape 2:', x1[0, 0, :])
        x = torch.cat((x1, x1_reg, x2, x2_reg), dim=2)
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
                
    # def forward_alternative(self, x, lengths):
    #     # x = x.float()
    #     # print('1:', x.size())
    #     # try:
    #     #     B, C, T, H = x.size()
    #     #     x = x.view(B, 1, C, T, H)
    #     # except:
    #     #     B, C, T, H, W = x.size()
    #     B, C, T, H, W = x.size()
    #     # print('1.5:', x.size())
    #     x = self.frontend3D(x)
    #     # print('2:', x.size())
    #     Tnew = x.shape[2]    # outpu should be B x C2 x Tnew x H x W
    #     x = threeD_to_2D_tensor( x )
    #     # print('3:', x.size())
    #     x = self.trunk(x)
    #     # print('4:', x.size())
    #     x = x.view(B, Tnew, x.size(1))
    #     # B, T = x.size()
    #     # print('5:', x.size())

    #     # x = self.trunk(x[:, None, :])
    #     # x = x.transpose(1, 2)
    #     # lengths = [i // 320 for i in lengths]  # sewa

    #     lengths = [i * 200 for i in lengths]
    #     x = self.gru(x, lengths)
    #     # print('6:', x.size())
    #     return torch.sigmoid(x) * 2 - 1.0