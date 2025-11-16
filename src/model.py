from torch import nn
from torchvision.models.segmentation import fcn_resnet50


class FCNResNet50Binary(nn.Module):
    """
    torchvision 自带的 fcn_resnet50，改成 1 通道二分类输出
    """
    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            self.net = fcn_resnet50(weights="DEFAULT")
        else:
            self.net = fcn_resnet50(weights=None)

        # self.net.classifier 通常是一个 nn.Sequential，比如：
        # [0] Conv2d, [1] BatchNorm2d, [2] ReLU, [3] Dropout, [4] Conv2d
        # 我们只需要把“最后一个 Conv2d”改成输出 1 通道
        if isinstance(self.net.classifier, nn.Sequential):
            last_conv = self.net.classifier[-1]
            if not isinstance(last_conv, nn.Conv2d):
                raise RuntimeError(f"最后一层不是 Conv2d，而是 {type(last_conv)}")

            in_ch = last_conv.in_channels
            self.net.classifier[-1] = nn.Conv2d(in_ch, 1, kernel_size=1)
        else:
            # 极少数版本，classifier 可能是别的结构，这里做个防御
            raise RuntimeError(f"未知的 classifier 类型: {type(self.net.classifier)}")

        # 不用 aux 分支就直接关掉
        self.net.aux_classifier = None

    def forward(self, x):
        out = self.net(x)["out"]  # B x 1 x H x W
        return out