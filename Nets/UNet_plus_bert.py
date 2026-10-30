import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention
from thop import clever_format
from thop import profile
import os
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
        # print(x.size())
        # new_shape = torch.sqrt(self.project_dim).totenso()
        embed = self.project_head(x).view(-1, 10, 14, 14)
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

class UNetText(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        # self.alignmodel = AlignModel()
        # for param in self.alignmodel.parameters():
        #     param.requires_grad = False
        # weights = 'params/Best_Align_5120.pth'
        # if os.path.exists(weights):
        #     self.alignmodel.load_state_dict(torch.load(weights))
            # print('Weight load successfully')
        # else:
            # print('No Weights, train')

        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = 64
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        # self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, 502, nb_Conv=2)  # 512-20=492 # 512-26=486
        self.up4 = UpBlock(in_channels*16, in_channels*4, nb_Conv=2)
        self.up3 = UpBlock(in_channels*8, in_channels*2, nb_Conv=2)
        self.up2 = UpBlock(in_channels*4, in_channels, nb_Conv=2)
        self.up1 = UpBlock(in_channels*2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1))

        self.bert_project = BERTProject(256, 196)  # 768->256

        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self,x, text):
        # print(x.size(), text.size())
        # no_use_text = torch.randn(8, 20, 256)
        # language = self.alignmodel(x, no_use_text, mode="test").view(-1, 26, 14, 14)

        language = self.bert_project(text)
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x5.size(), language.size())
        O1 = self.up4(torch.cat((x5, language), dim=1), x4)
        # print(O1.size())
        # O1 = self.up4(x5, x4)
        O2 = self.up3(O1, x3)
        O3 = self.up2(O2, x2)
        O4 = self.up1(O3, x1)
        # print(x.size(), x1.size(), x2.size(), x3.size(), x4.size(), x5.size(), )
        # print(O1.size(), O2.size(), O3.size(), O4.size(), )
        if self.last_activation is not None:
            logits = self.last_activation(self.outc(O4))
        else:
            logits = self.outc(O4)
        # print(logits.size())
        return x5, logits


if __name__ == '__main__':
    x = torch.randn(4, 3, 224, 224)
    text = torch.randn(4, 18, 256)
    # text = torch.randn(4, 20, 512)
    net = UNetText(n_channels=3, n_classes=1)
    print(net(x, text)[0].shape, net(x, text)[1].shape)
    flops, params = profile(net, inputs=(x, text,))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))


