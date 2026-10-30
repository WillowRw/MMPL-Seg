import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention
from thop import clever_format
from thop import profile
import os
import random
class BERTProject(nn.Module):
    def __init__(self, text_dim, project_dim):
        super(BERTProject, self).__init__()
        self.project_dim = project_dim
        self.project_head = nn.Sequential(
            nn.Linear(text_dim, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Linear(project_dim, project_dim)
        )

    def forward(self, x):
        # new_shape = torch.sqrt(self.project_dim).totenso()
        embed = self.project_head(x.clone()).clone().reshape(-1, 10, 14, 14)
        return embed



def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels//2,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class Dy_Double_UNet(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 0, lora_dropout=0.1, n_channels=3, n_classes=1):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = 64
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        # self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, 502, nb_Conv=2)  # 512-20=492 # 512-26=486
        self.up4 = UpBlock(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpBlock(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpBlock(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpBlock(in_channels * 2, in_channels, nb_Conv=2)
        self.outx9 = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1))

        self.bert_project = BERTProject(256, 196)  # 768->256

        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

        self.outc = nn.Conv2d(65, 1, kernel_size=(1, 1))

        self.precls_conv = nn.Sequential(
            nn.GroupNorm(32, 64),
            nn.ReLU(),
            nn.Conv2d(64, 8, kernel_size=(1, 1))
        )

        self.GAP = nn.Sequential(
            nn.GroupNorm(32, 512),  # nn.GroupNorm(16, 256),
            nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        # self.controller = nn.Conv2d(256 + 6, 162, kernel_size=1, stride=1, padding=0)  #### change the channel
        self.controller = nn.Conv2d(512+18, 162, kernel_size=1, stride=1, padding=0)
        torch.autograd.set_detect_anomaly(True)


        self.reconstruction_para_1 = nn.Parameter(torch.empty(1, 512, 1, 1))
        nn.init.kaiming_normal_(self.reconstruction_para_1, mode='fan_in', nonlinearity='relu')

        self.reconstruction_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(502, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Softplus(),
        )

        self.reconstruction_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(502, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, 18),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Softplus(),
        )

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 2, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 2)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            # if i < n_layers - 1:
            #     x = F.relu(x)
                # x = nn.Sigmoid()(x)
        return x


    def forward(self,x, text):
        b = x.size(0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # no_use_text = torch.randn(8, 20, 256)
        # language = self.alignmodel(x, no_use_text, mode="test").view(-1, 26, 14, 14)

        language14 = self.bert_project(text[:, 0:10, :])
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5_1 = torch.cat((x5, language14), dim=1)
        mode = "one"
        if mode == "one":
            recon_feature_1 = self.reconstruction_1(x5).unsqueeze(-1).unsqueeze(-1).expand_as(x5_1)
            # torch.Size([4, 512, 14, 14]) torch.Size([4, 512, 14, 14]) torch.Size([1, 512, 1, 1])
            x5_1 = x5_1 * recon_feature_1 * self.reconstruction_para_1  # torch.Size([4, 512, 14, 14])
        x_6 = self.up4(x5_1, x4)
        x_7 = self.up3(x_6, x3)
        x_8 = self.up2(x_7, x2)
        x_9 = self.up1(x_8, x1)  # torch.Size([batch, 64, 224, 224])
        if self.last_activation is not None:
            logits = self.last_activation(self.outx9(x_9))
        else:
            logits = self.outx9(x_9)
        # result1 = x_9
        # # result2 = (self.lora_dropout(text.view(-1, 2560)) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0,
        #                                                                                                         1)).view(
        #     -1, 1, 224, 224) * self.scaling
        # result = nn.Sigmoid()(self.outc(torch.concat((result1, result2), dim=1)))
        return logits, x_9  # x_9 contains the pre-activation UNet features.


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(4, 3, 224, 224).to(device)
    text = torch.randn(4, 18, 256).to(device)
    net = Dy_Double_UNet(in_features=2560, out_features=224*224, r=512, n_channels=3, n_classes=1).to(device)
    out = net(x, text)
    print(out[0].size())
    # flops, params = profile(net, inputs=(x, text))
    # print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    # print("params=", str(params / 1e6) + '{}'.format("M"))


