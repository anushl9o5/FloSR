import torch.nn as nn

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))

class BasicBlock(nn.Module):
    expansion = 2
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()
        
        # self.conv1 = nn.Sequential(nn.convbn(inplanes, planes, 3, stride, pad, dilation),
        #                            nn.ReLU(inplace=True))
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(in_channels=inplanes,
                               out_channels=planes,
                               kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(in_channels=planes,
                                out_channels=planes,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(in_channels=planes,
                                out_channels=planes*self.expansion,
                                kernel_size=1,
                                bias=False)

        
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        
        
    def forward(self, x):
        # print(f"x.shape : {tuple(x.shape)}")
        residual = x
        out = self.bn1(x)
        out = self.leaky_relu(out)

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.leaky_relu(out)
        # print(f"conv1(x).shape {tuple(out.shape)}")

        out = self.conv2(out)
        out = self.bn3(out)
        out = self.leaky_relu(out)
        # print(f"conv2(x).shape {tuple(out.shape)}")


        out = self.conv3(out)
        # print(f"conv3(x).shape {tuple(out.shape)}")

        if self.downsample is not None:
            residual = self.downsample(residual)

        # print(f"residual.shape : {tuple(residual.shape)}")

        out += residual
        # print('*'*20)
        return out
