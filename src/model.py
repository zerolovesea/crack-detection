from torch import nn
from torchvision.models.segmentation import fcn_resnet50


class FCNResNet50(nn.Module):
    """
    torchvision fcn_resnet50，多类别像素级输出
    num_classes 包含背景类 (class 0)
    """
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()

        if pretrained:
            self.net = fcn_resnet50(weights="DEFAULT")
        else:
            self.net = fcn_resnet50(weights=None)

        if isinstance(self.net.classifier, nn.Sequential):
            last_conv = self.net.classifier[-1]
            if not isinstance(last_conv, nn.Conv2d):
                raise RuntimeError(f"最后一层不是 Conv2d，而是 {type(last_conv)}")

            in_ch = last_conv.in_channels
            self.net.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
        else:
            raise RuntimeError(f"未知的 classifier 类型: {type(self.net.classifier)}")

        self.net.aux_classifier = None

    def forward(self, x):
        out = self.net(x)["out"]  # B x C x H x W
        return out
