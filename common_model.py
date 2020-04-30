import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.backends.cudnn as cudnn

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'l2norm']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class a(nn.Module):
    def __init__(self):
        super(a, self).__init__()
        self.a =torch.nn.Parameter(torch.Tensor([0.8]))  # input and output is 1 dimension

    def forward(self, x1,x2):
        out = self.a*x1+(1-self.a)*x2
        return out

def conv3x3(in_planes, out_planes, stride=1):  # 3*3的卷积模板
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block,layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.embed = nn.Embedding(70, 16, padding_idx=0, max_norm=None,
                                  norm_type=2, scale_grad_by_freq=False, sparse=False)
        self.conv01 = nn.Conv1d(
            in_channels=16,
            out_channels=224,
            kernel_size=3,
            padding=1
        )
        self.conv02 = nn.Conv1d(
            in_channels=224,
            out_channels=448,
            kernel_size=3,
            padding=1
        )
        self.conv0 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(14, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        # self.fc1 = nn.Linear(512 * block.expansion, 512)
        # self.fc2 = nn.Linear(512, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))


        return nn.Sequential(*layers)

    def process_txt(self, x):  # x是(4,448)
   #     print('x1',x.size())
        x = self.embed(x)  # (4,448,16)
        x=x.transpose(2,1)#转置([4, 16, 448])
        x=self.conv01(x)#[4,224,448]
        x=self.relu(x)
        x=self.conv02(x)#[4, 448, 448]
        x=x.view(-1,1,448,448)#[4, 1, 448, 448]
        x=self.conv0(x)#[4, 3, 448, 448]
        return x

    def forward_share(self, x):
        # 输入x[16, 3, 448, 448]
        x = self.conv1(x)  # [16, 64, 224, 224]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 输出x[16, 64, 112, 112]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print('xxx',x.size())
        x = self.fc1(x)
        # '''全连接修改'''
        # x=self.fc2(x)
        return x

    def forward_txt(self, x):
        x = self.process_txt(x)
        x = self.forward_share(x)
        return x

    def forward(self, x1, x2, x3, x):
        x = self.process_txt(x)  # [4, 3, 448, 448]
        #print('xxx',x.size())
        x = torch.cat((x1, x2, x3, x), dim=0)
        x = self.forward_share(x)
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck,[3, 4, 6, 3], **kwargs)
    model=model#.cuda()
    if pretrained:
         model.load_state_dict(model_zoo.load_url(model_urls['resnet50']),strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class l2norm(nn.Module):
    def __init__(self):
        super(l2norm, self).__init__()

    def forward(self, input, epsilon=1e-7):
        assert len(input.size()) == 2, "Input dimension requires 2,but get {}".format(len(input.size()))

        norm = torch.norm(input, p=2, dim=1, keepdim=True)
        output = torch.div(input, norm + epsilon)
        return output
