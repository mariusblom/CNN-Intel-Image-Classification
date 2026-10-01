"""
Eigene Implementierung eines ResNet-18 (He et al., 2015: "Deep Residual
Learning for Image Recognition").

Die Modul- und Parameternamen (conv1, bn1, layer1..layer4, avgpool, fc,
innerhalb der Blöcke conv1/bn1/conv2/bn2/downsample) sind bewusst
identisch zu torchvision.models.resnet18 gewählt. Dadurch lässt sich der
state_dict der offiziellen, auf ImageNet vortrainierten Gewichte direkt
in dieses Modell laden (siehe model_factory.py).

Architektur:
    Stem:    7x7-Conv (64 Filter, stride 2) -> BN -> ReLU -> 3x3-MaxPool (stride 2)
    Stage 1: 2x BasicBlock,  64 Kanäle, stride 1
    Stage 2: 2x BasicBlock, 128 Kanäle, stride 2 (erster Block)
    Stage 3: 2x BasicBlock, 256 Kanäle, stride 2 (erster Block)
    Stage 4: 2x BasicBlock, 512 Kanäle, stride 2 (erster Block)
    Head:    Global Average Pooling -> Fully Connected (num_classes)

Parameteranzahl bei num_classes=1000: 11.689.512 (identisch zu torchvision)
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Residual-Block mit zwei 3x3-Convolutions, wie in ResNet-18/34 verwendet."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: nn.Module = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity  # Skip-Connection
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 1000, in_channels: int = 3):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_stage(64, blocks=2, stride=1)
        self.layer2 = self._make_stage(128, blocks=2, stride=2)
        self.layer3 = self._make_stage(256, blocks=2, stride=2)
        self.layer4 = self._make_stage(512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self._initialize_weights()

    def _make_stage(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = ResNet18(num_classes=1000)
    print(f"Gesamtparameter: {count_parameters(model):,}")
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("Output-Shape:", tuple(y.shape))
