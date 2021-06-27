import torch
import torch.nn as nn
from math import ceil,sqrt
from .mobile_v2 import _make_divisible
from math import ceil, sqrt

__all__ = ['Bottleneck', 'ResNet']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, groups=1, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, 
        groups=groups, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, groups=1, base_width=64, std_ch_width=16,
                 norm_layer=None, boost_groups=1, split_groups=1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        group_list = [x * boost_groups for x in [split_groups, 1, split_groups]]
        width = int(planes  * (base_width / std_ch_width) )  * groups 
        width = _make_divisible(width, boost_groups * groups * split_groups)
        self.conv1 = conv1x1(inplanes, width * split_groups, group_list[0])
        self.bn1 = norm_layer(width * split_groups)
        self.conv2 = conv3x3(width, width, stride, groups if groups > 1 else group_list[1])
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width * split_groups, planes * self.expansion, group_list[2])
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, group_list[-1], stride),
                nn.BatchNorm2d(planes * self.expansion),
            )    
        else:
            self.downsample = None        

    def forward(self, x):
        identity = x

        out = self.conv1(x.view(-1, self.conv1.in_channels, x.size(2), x.size(3)))
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out.view(-1, self.conv2.in_channels, out.size(2), out.size(3)))
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out.view(-1, self.conv3.in_channels, out.size(2), out.size(3)))
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, layers, num_classes=1000, groups=1, width_per_group=64, 
                 boost_groups=[1]*5, split_groups=[1]*5, wide_factor={i:sqrt(i) for i in range(6)}, 
                norm_layer=None, imagenet=False,  zero_init_residual=False, ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # self._norm_layer = norm_layer
        self.num_classes = num_classes
        self.groups = groups
        imagenet_archi = num_classes > 100
        # imagnet architecture has 4 stages
        if imagenet_archi:
            ch_list = [64, 128, 256, 512]
        else:
        # resnext architecture has wider channels
            ch_list = [64, 128, 256]

        std_ch_width = 64.
        # if not imagenet_archi and groups ==1:
        #     std_ch_width = 16.

        stride_list = [1, 2, 2, 2]
        self.total_groups = [x * y for x,y in zip(boost_groups, split_groups)]
        wide_factor_list = [wide_factor[i] for i in boost_groups[1:]]
        ch_list = [x * y for x,y in zip(ch_list, wide_factor_list)]
        # make channel width to be divisible by next stage
        ch_list = [ _make_divisible(ch, self.total_groups[i+2]) for i, ch in enumerate(ch_list)]
        in_channels = ch_list[0]
        if imagenet_archi:
            conv1 = nn.Conv2d(3, in_channels, kernel_size=7, stride=2, padding=3, bias=False)
            pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            conv1 = conv3x3(3, in_channels)
            pool1 = nn.Identity()

        features: List[nn.Module] = [conv1, norm_layer(in_channels), nn.ReLU(True), pool1]
        for i, ch_width in enumerate(ch_list):
            out_channels = ch_width
            for j in range(layers[i]):
                stride = stride_list[i] if j == 0 else 1
                features.append(Bottleneck(in_channels, out_channels, stride, groups, width_per_group, std_ch_width, norm_layer,
                    boost_groups[i+1], split_groups[i+1]))
                in_channels = out_channels * Bottleneck.expansion

        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * Bottleneck.expansion, num_classes*self.total_groups[-1],
                kernel_size=1, bias=True, groups=self.total_groups[-1])
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        
        x = x.view(x.size(0), self.total_groups[-1], self.num_classes)
        sum_x = x.clone()
        for i in range(1, self.total_groups[-1]):
            sum_x[:,i] += sum_x[:,i-1].detach()

        return x, sum_x

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnext50_32x4d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnext101_32x4d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnext101_32x8d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def wide_resnet50_2(**kwargs):
    kwargs['width_per_group'] = 64 * 2
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def boost_conv3x3(in_planes, out_planes, stride=1, groups=1, boost_groups=1, dilation=1):
    return nn.ModuleList([nn.Conv2d(in_planes//boost_groups, out_planes //boost_groups, 
        kernel_size=3, groups=groups, padding=dilation, stride=stride, bias=False) for i in range(boost_groups)])
