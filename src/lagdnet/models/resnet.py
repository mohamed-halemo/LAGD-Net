"""ResNet-18/34/50/101/152 with one gate per residual block.

The stem is a 3x3 convolution with stride 1 on CIFAR-10 and stride 2 on Tiny-ImageNet, without
max pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lagdnet.layers import TAU, BlockGate

STEM_STRIDE = {'cifar': 1, 'tiny_imagenet': 2}


def _shortcut(in_planes, out_planes, stride):
    if stride == 1 and in_planes == out_planes:
        return nn.Identity()
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                         nn.BatchNorm2d(out_planes))


class GatedBlock(nn.Module):
    """y = relu(gate * F(x) + S(x)). Once the threshold has been applied (snapped), a block whose
    gate is below tau keeps only its shortcut, y = S(x)."""
    expansion = 1

    def __init__(self, in_planes, planes, stride, temperature):
        super().__init__()
        self.shortcut = _shortcut(in_planes, planes * self.expansion, stride)
        self.gate = BlockGate(temperature)
        self.snapped = False

    def branch(self, x):
        raise NotImplementedError

    def forward(self, x):
        if self.snapped and self.gate.value() < TAU:
            return self.shortcut(x)
        return F.relu(self.gate(self.branch(x)) + self.shortcut(x))


class BasicBlock(GatedBlock):
    expansion = 1

    def __init__(self, in_planes, planes, stride, temperature):
        super().__init__(in_planes, planes, stride, temperature)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def branch(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(out))


class Bottleneck(GatedBlock):
    expansion = 4

    def __init__(self, in_planes, planes, stride, temperature):
        super().__init__(in_planes, planes, stride, temperature)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

    def branch(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return self.bn3(self.conv3(out))


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dataset='cifar', start_temp=5.0):
        super().__init__()
        self.in_planes = 64
        self.start_temp = start_temp
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=STEM_STRIDE[dataset], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(block(self.in_planes, planes, s, self.start_temp))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = torch.flatten(self.avgpool(out), 1)
        return self.linear(out)

    def set_temperature(self, temperature):
        for m in self.modules():
            if isinstance(m, BlockGate):
                m.temperature = temperature


def ResNet18(dataset='cifar', num_classes=10, start_temp=5.0):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, dataset, start_temp)


def ResNet34(dataset='cifar', num_classes=10, start_temp=5.0):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, dataset, start_temp)


def ResNet50(dataset='cifar', num_classes=10, start_temp=5.0):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, dataset, start_temp)


def ResNet101(dataset='cifar', num_classes=10, start_temp=5.0):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, dataset, start_temp)


def ResNet152(dataset='cifar', num_classes=10, start_temp=5.0):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, dataset, start_temp)
