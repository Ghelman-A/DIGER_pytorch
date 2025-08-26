import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Tuple


class Darknet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = self.create_model()
        self.desired_levels = ['layer2', 'layer3', 'layer4', 'layer5', 'layer6',]  # Layers to use for knowledge distillation

    def create_model(self):
        base = self.conv_norm_relu(3, 32, kernel_size=3)

        layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            self.conv_norm_relu(32, 64, kernel_size=3)
        )

        layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            self.conv_norm_relu(64, 128, kernel_size=3),
            self.conv_norm_relu(128, 64, kernel_size=1, padding=0),
            self.conv_norm_relu(64, 128, kernel_size=3)
        )

        layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            self.conv_norm_relu(128, 256, kernel_size=3),
            self.conv_norm_relu(256, 128, kernel_size=1, padding=0),
            self.conv_norm_relu(128, 256, kernel_size=3)
        )

        layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            self.conv_norm_relu(256, 512, kernel_size=3),
            self.conv_norm_relu(512, 256, kernel_size=1, padding=0),
            self.conv_norm_relu(256, 512, kernel_size=3),
            self.conv_norm_relu(512, 256, kernel_size=1, padding=0),
            self.conv_norm_relu(256, 512, kernel_size=3)
        )

        layer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            self.conv_norm_relu(512, 1024, kernel_size=3),
            self.conv_norm_relu(1024, 512, kernel_size=1, padding=0),
            self.conv_norm_relu(512, 1024, kernel_size=3),
            self.conv_norm_relu(1024, 512, kernel_size=1, padding=0),
            self.conv_norm_relu(512, 1024, kernel_size=3)
        )

        # Layer 6 is the main difference between Darknet-19 backbone and the full YOLOv2 model
        layer6 = nn.Sequential(OrderedDict([
            ('conv0', self.conv_norm_relu(1024, 1024, kernel_size=3)),
            ('conv1', self.conv_norm_relu(1024, 1024, kernel_size=3)),

            ('conv2', self.conv_norm_relu(512, 64, kernel_size=1, padding=0)),
            ('reorg', Reorg()),
            
            ('conv3', self.conv_norm_relu(1280, 1024, kernel_size=3)),
            ('conv4', nn.Conv2d(1024, 425, kernel_size=(1, 1), stride=(1, 1))),
        ]))

        model = nn.Sequential(OrderedDict([
            ('base', base),
            ('layer1', layer1),
            ('layer2', layer2),
            ('layer3', layer3),
            ('layer4', layer4),
            ('layer5', layer5),
            ('layer6', layer6),
        ]))
        return model

    def conv_norm_relu(self, in_size: int, out_size: int, kernel_size: int, stride: int=1, padding: int=1) -> nn.Sequential:

        k = kernel_size
        s = stride
        p = padding
        unit = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=(k, k), stride=(s, s), padding=(p, p), bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(inplace=True)
        )
        return unit

    def forward(self, input):
        out = self.model.base(input)
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        out = self.model.layer3(out)
        out = self.model.layer4(out)
        route1 = out.clone()

        out = self.model.layer5(out)
        out = self.model.layer6.conv0(out)
        out = self.model.layer6.conv1(out)
        
        # Route 1, for multi-scaling of YOLO
        route1 = self.model.layer6.conv2(route1)
        route1 = self.model.layer6.reorg(route1)

        # Route 2, for multi-scaling of YOLO
        out = self.model.layer6.conv3(torch.cat([route1, out], dim=1))
        out = self.model.layer6.conv4(out)

        return out

    def listed_output(self, x: torch.Tensor, num_blocks: int=1) -> Tuple[torch.Tensor]:
        out_dict = dict()
        desired_layers = self.desired_levels[-num_blocks:]

        out = self.model.base(x)
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        if 'layer2' in desired_layers: out_dict[5] = out

        out = self.model.layer3(out)
        if 'layer3' in desired_layers: out_dict[4] = out

        out = self.model.layer4(out)
        route1 = out.clone()
        if 'layer4' in desired_layers: out_dict[3] = out

        out = self.model.layer5(out)
        if 'layer5' in desired_layers: out_dict[2] = out

        out = self.model.layer6.conv0(out)
        out = self.model.layer6.conv1(out)
        
        # Route 1, for multi-scaling of YOLO
        route1 = self.model.layer6.conv2(route1)
        route1 = self.model.layer6.reorg(route1)

        # Route 2, for multi-scaling of YOLO
        out = self.model.layer6.conv3(torch.cat([route1, out], dim=1))
        out = self.model.layer6.conv4(out)

        if 'layer6' in desired_layers: out_dict[1] = out

        return out, out_dict
    
    def load_pretrained_model(self, checkpoint_dir: str, freeze: bool=True) -> None:
        #-----------------------------------------------------------------------------------#
        # Loading the pretrained model. There is a difference if the model is loaded from a #
        # chackpoint or if it is a pretrained ImageNet model.                               #
        #-----------------------------------------------------------------------------------#
        saved_model = torch.load(checkpoint_dir)
        saved_model= saved_model['state_dict'] if 'state_dict' in saved_model.keys() else saved_model
        
        #-----------------------------------------------------------------------------------#
        # Replacing the backbone with the loaded weights and making sure that the gradient  #
        # is properly set.                                                                  #
        #-----------------------------------------------------------------------------------#
        self.load_state_dict(saved_model, strict=False)
        layers = ['conv3', 'conv4']
        for name, param in self.named_parameters():
            if not freeze:
                param.requires_grad = True
            else:
                if any(layer in name for layer in layers):
                    param.requires_grad = True
                else:
                    param.requires_grad = False


class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride
    
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H//hs, hs, W//ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, H//hs*W//ws, hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H//hs, W//ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H//hs, W//ws)
        return x

