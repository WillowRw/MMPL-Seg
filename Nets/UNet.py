import torch
from torch import nn
from torch.nn import functional as F
from thop import clever_format
from thop import profile

# class Conv_Block(nn.Module):
#     def __init__(self,in_channel,out_channel):
#         super(Conv_Block, self).__init__()
#         self.layer=nn.Sequential(
#             nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
#             nn.BatchNorm2d(out_channel),
#             nn.Dropout2d(0.3),
#             nn.LeakyReLU(),
#             nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
#             nn.BatchNorm2d(out_channel),
#             nn.Dropout2d(0.3),
#             nn.LeakyReLU()
#         )
#     def forward(self,x):
#         return self.layer(x)
#
#
# class DownSample(nn.Module):
#     def __init__(self,channel):
#         super(DownSample, self).__init__()
#         self.layer=nn.Sequential(
#             nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
#             nn.BatchNorm2d(channel),
#             nn.LeakyReLU()
#         )
#     def forward(self,x):
#         return self.layer(x)
#
#
# class UpSample(nn.Module):
#     def __init__(self,channel):
#         super(UpSample, self).__init__()
#         self.layer=nn.Conv2d(channel,channel//2,1,1)
#     def forward(self,x,feature_map):
#         up=F.interpolate(x,scale_factor=2,mode='nearest')
#         out=self.layer(up)
#         return torch.cat((out,feature_map),dim=1)
#
#
# class UNet(nn.Module):
#     def __init__(self,num_classes):
#         super(UNet, self).__init__()
#         self.c1=Conv_Block(3,64)
#         self.d1=DownSample(64)
#         self.c2=Conv_Block(64,128)
#         self.d2=DownSample(128)
#         self.c3=Conv_Block(128,256)
#         self.d3=DownSample(256)
#         self.c4=Conv_Block(256,512)
#         self.d4=DownSample(512)
#         self.c5=Conv_Block(512,1024)
#         self.u1=UpSample(1024)
#         self.c6=Conv_Block(1024,512)
#         self.u2 = UpSample(512)
#         self.c7 = Conv_Block(512, 256)
#         self.u3 = UpSample(256)
#         self.c8 = Conv_Block(256, 128)
#         self.u4 = UpSample(128)
#         self.c9 = Conv_Block(128, 64)
#         self.out=nn.Conv2d(64,num_classes,3,1,1)
#
#     def forward(self,x):
#         R1=self.c1(x)
#         R2=self.c2(self.d1(R1))
#         R3 = self.c3(self.d2(R2))
#         R4 = self.c4(self.d3(R3))
#         R5 = self.c5(self.d4(R4))
#         O1=self.c6(self.u1(R5,R4))
#         O2 = self.c7(self.u2(O1, R3))
#         O3 = self.c8(self.u3(O2, R2))
#         O4 = self.c9(self.u4(O3, R1))
#         out = self.out(O4)
#         return R5, out
#         return out
# UNet building blocks.

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

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
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
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)
        self.up4 = UpBlock(in_channels*16, in_channels*4, nb_Conv=2)
        self.up3 = UpBlock(in_channels*8, in_channels*2, nb_Conv=2)
        self.up2 = UpBlock(in_channels*4, in_channels, nb_Conv=2)
        self.up1 = UpBlock(in_channels*2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1))
        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        # print("model UNet from ViLT")
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_6 = self.up4(x5, x4)
        x_7 = self.up3(x_6, x3)
        x_8 = self.up2(x_7, x2)
        x_9 = self.up1(x_8, x1)  # torch.Size([batch, 64, 224, 224])
        # print("!!", x_9.shape)
        if self.last_activation is not None:
            logits = self.last_activation(self.outc(x_9))
        else:
            logits = self.outc(x_9)
        return x_9, logits


if __name__ == '__main__':
    x = torch.randn(4, 3, 224, 224)
    net = UNet(n_channels=3, n_classes=1)  # LViT
    print(net(x)[1].shape)
    flops, params = profile(net, inputs=(x,))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))  # 100.740562944G
    print("params=", str(params / 1e6) + '{}'.format("M"))  # 14.751873M

