from torch import nn
from torch import Tensor
# from torchvision.models.utils import load_state_dict_from_url
from typing import Callable, Any, Optional, List, Dict
from torch.autograd.function import Function

__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.in_channels = in_planes
        self.out_channels = out_planes


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation

class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        groups: Optional[int] = 1,
        split_groups: Optional[int] = 1,
        fuse_conv: Optional[bool] = False,
        scale_gradient: Optional[bool] = False,
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        self.split_groups = split_groups
        self.scale_gradient = scale_gradient

        group_list = [x * groups for x in [split_groups, 1, split_groups]]
        # layers: List[nn.Module] = []
        self.fuse_conv = fuse_conv
        if fuse_conv:
            self.conv1 = ConvBNReLU(inp, hidden_dim, stride=stride, norm_layer=norm_layer, groups=group_list[0])
        else:
            if expand_ratio * split_groups != 1:
                # pw
                self.conv1 = ConvBNReLU(inp, hidden_dim * split_groups, kernel_size=1, norm_layer=norm_layer, groups=group_list[0])
            else:
                self.conv1 = nn.Identity()
            self.conv2 = ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer)
            
        self.conv3 = nn.Conv2d(hidden_dim * split_groups, oup, 1, 1, 0, bias=False, groups=group_list[2])
        self.norm_layer = norm_layer(oup)

    def forward(self, x: Tensor) -> Tensor:
        if self.fuse_conv:
            out = self.conv1(x)
        else:
            if isinstance(self.conv1, nn.Identity):
                out = x
            else:
                out = x.view(-1, self.conv1.in_channels, x.size(2), x.size(3))
            out = self.conv1(out)
            out = _ScaleGradient.apply(out, self.split_groups if self.scale_gradient else 1.)
            out = self.conv2(out.view(-1, self.conv2.in_channels, out.size(2), out.size(3)))
            out = _ScaleGradient.apply(out, 1. / self.split_groups if self.scale_gradient else 1.)
        
        out = self.conv3(out.view(-1, self.conv3.in_channels, out.size(2), out.size(3)))
        out = self.norm_layer(out)

        if self.use_res_connect:
            return x + out
        else:
            return out

class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        boost_groups: Optional[List[int]] = [1]*8,
        split_groups: Optional[List[int]] = [1]*8,
        wide_factor: Optional[Dict] = {},
        scale_gradient: Optional[bool] = False,
        fuse_stages: Optional[int] = 0,
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()
        self.num_classes = num_classes
        self.groups = [x * y for x,y in zip(boost_groups, split_groups)]
        wide_factor_list = [wide_factor[i] for i in boost_groups[1:]]
        imagenet_archi = num_classes > 100

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280 * wide_factor_list[-2]

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2 if imagenet_archi else 1],
                [6, 32, 3, 2 ],
                [6, 64, 4, 2 if imagenet_archi else 1],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, self.groups[1] * round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), self.groups[-1]  * round_nearest)
        features: List[nn.Module] = [ConvBNReLU(3, input_channel, stride=2 if imagenet_archi else 1, norm_layer=norm_layer)]
        # features: List[nn.Module] = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for stage, (t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = _make_divisible(c * width_mult * wide_factor_list[stage+1], self.groups[stage + 2] * round_nearest)
            # print(f'out_c: {output_channel}')
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer,
                    groups=boost_groups[stage + 1], split_groups=split_groups[stage + 1], fuse_conv=stage < fuse_stages,
                    scale_gradient=scale_gradient))
                input_channel = output_channel

        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer,
            groups=self.groups[-2]))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier =  nn.Conv2d(self.last_channel, num_classes * self.groups[-1], 1, groups=self.groups[-1])

        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.last_channel, num_classes),
        # )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = self.classifier(x).reshape(-1, self.groups[-1], self.num_classes)
        sum_x = x.clone()
        for i in range(1, self.groups[-1]):
            sum_x[:,i] += sum_x[:,i-1].detach()

        return x, sum_x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

# def mobilenet_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
#     """
#     Constructs a MobileNetV2 architecture from
#     `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     model = MobileNetV2(**kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model