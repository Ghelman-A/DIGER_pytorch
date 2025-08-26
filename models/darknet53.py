from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_norm_relu(in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=1) -> nn.Sequential:

    k = kernel_size
    s = stride
    p = padding
    unit = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_channels, momentum=0.9),
        nn.LeakyReLU(inplace=True, negative_slope=0.1)
    )
    return unit

class DarkResBlock(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        
        mid_channels = int(in_channels / 2)
        self.conv1 = conv_norm_relu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = conv_norm_relu(mid_channels, in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        
        return out + x
        
class Darknet53(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self):
        super(Darknet53, self).__init__()        
        
        self.conv1 = conv_norm_relu(3, 32, kernel_size=3, stride=2)
        
        self.layer1 = self.create_darknet_layer(num_blocks=1, in_channels=32)
        self.layer2 = self.create_darknet_layer(num_blocks=2, in_channels=64)
        self.layer3 = self.create_darknet_layer(num_blocks=8, in_channels=128)
        self.layer4 = self.create_darknet_layer(num_blocks=8, in_channels=256)
        self.layer5 = self.create_darknet_layer(num_blocks=4, in_channels=512)
        
        # Layer 6 is the main difference between Darknet-53 backbone and the full YOLOv3 model
        self.layer6 = self.create_yolo_layer(in_channels=1024, mid_channels=512)
        self.layer7 = self.create_yolo_layer(in_channels=768, mid_channels=256)
        self.layer8 = self.create_yolo_layer(in_channels=384, mid_channels=128, out_channels=256)
        self.conv_last = nn.Conv2d(256, 255, kernel_size=1)

    def create_darknet_layer(self, num_blocks: int, in_channels: int):
        layer = []
        out_channels = 2 * in_channels
        layer.append(conv_norm_relu(in_channels, out_channels, kernel_size=3, stride=2))
        
        for _ in range(num_blocks):
            layer.append(DarkResBlock(out_channels))

        return nn.Sequential(*layer)

    def create_yolo_layer(self, in_channels: int, mid_channels: int, out_channels: int=None):
        mid = mid_channels
        mid_2 = 2 * mid

        # out channel is only for the last layer differences
        out = int(mid / 2) if out_channels is None else out_channels
        out_kernel = 1 if out_channels is None else 3
        out_padding = 0 if out_channels is None else 1
        out_stride = 1 if out_channels is None else 2

        layer = nn.Sequential(
            conv_norm_relu(in_channels, mid, kernel_size=1, padding=0),
            conv_norm_relu(mid, mid_2, kernel_size=3),
            conv_norm_relu(mid_2, mid, kernel_size=1, padding=0),
            conv_norm_relu(mid, mid_2, kernel_size=3),
            conv_norm_relu(mid_2, mid, kernel_size=1, padding=0),
            conv_norm_relu(mid, out, kernel_size=out_kernel, stride=out_stride, padding=out_padding),
        )

        return layer

    def forward(self, input):
        out = self.conv1(input)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        route1 = out.clone()
        
        out = self.layer4(out)
        route2 = out.clone()
        
        out = self.layer5(out)        
        out = self.layer6(out)
        
        up_sample_size = (route2.shape[-2], route2.shape[-1])       # output spatial size
        out = torch.cat([F.interpolate(out, size=up_sample_size, mode='nearest'), route2], dim=1)
        out = self.layer7(out)

        up_sample_size = (route1.shape[-2], route1.shape[-1])
        out = torch.cat([F.interpolate(out, size=up_sample_size, mode='nearest'), route1], dim=1)

        out = self.layer8(out)
        out = self.conv_last(out)
        return out

    def load_pretrained_model(self, checkpoint_dir: str, freeze: bool=True) -> None:
        saved_model = torch.load(checkpoint_dir)
        
        #-----------------------------------------------------------------------------------#
        # Replacing the backbone with the loaded weights and making sure that the gradient  #
        # is properly set.                                                                  #
        #-----------------------------------------------------------------------------------#
        self.load_state_dict(saved_model, strict=True)
        layers = ['conv_last', 'layer8.5']
        for name, param in self.named_parameters():
            if not freeze:
                param.requires_grad = True
            else:
                if any(layer in name for layer in layers):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

