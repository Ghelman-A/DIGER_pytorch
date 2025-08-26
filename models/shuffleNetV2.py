'''ShuffleNetV2 in PyTorch.

See the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" for more details.
'''

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Tuple, List, Dict


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1,1,1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )

def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )

def channel_shuffle(x, groups):
    '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
    batchsize, num_channels, depth, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, depth, height, width)
    #permute
    x = x.permute(0,2,1,3,4,5).contiguous()
    # flatten
    x = x.view(batchsize, num_channels, depth, height, width)
    return x
    
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, keep_tem_res: bool=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2

        if self.stride == 1:
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv3d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm3d(oup_inc),
                # pw-linear
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            layers = [
                # dw
                nn.Conv3d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm3d(inp),
                # pw-linear
                nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
            ]
            if keep_tem_res: layers[0] = nn.Conv3d(inp, inp, 3, (1, stride, stride), 1, groups=inp, bias=False)
            self.banch1 = nn.Sequential(*layers)

            layers2 = [
                # pw
                nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv3d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm3d(oup_inc),
                # pw-linear
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
            ]

            if keep_tem_res: layers2[3] = nn.Conv3d(oup_inc, oup_inc, 3, (1, stride, stride), 1, groups=oup_inc, bias=False)
            self.banch2 = nn.Sequential(*layers2)
          
    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)        

    def forward(self, x):
        if self.stride == 1:
            x1 = x[:, :(x.shape[1]//2), :, :, :]
            x2 = x[:, (x.shape[1]//2):, :, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif self.stride == 2:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, distill_layers: int, num_classes=600, width_mult=1.):
        super(ShuffleNetV2, self).__init__()
        self.dst_layers = distill_layers
        self.width_mult = width_mult
        
        self.stage_repeats = [4, 8, 4]
        self.desired_levels = [3, 11, 15, -1]       # Used to get the output of each stage
        self.stage_out_channels = self.width_mult_channels(width_mult)
        
        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, stride=(1,2,2))
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # building inverted residual blocks
        self.features, input_channel = self.build_res_blocks(input_channel, width_mult)
        self.features = nn.Sequential(*self.features)

        # building last several layers
        self.conv_last      = conv_1x1x1_bn(input_channel, self.stage_out_channels[-1])
        if width_mult == 2.0:
            self.project_conv = self.build_project_covs(self.dst_layers)

        if width_mult == 0.25:
            self.lateral_conv = nn.Sequential(      # Necessary to be able to pass to CFAM
                nn.Conv3d(256, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm3d(2048), nn.ReLU(inplace=True)
            )

        ### building classifier, (original shuffleNetV2, not used in YOWO)
        # self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.stage_out_channels[-1], num_classes))
    
        self.avgpool = nn.AvgPool3d((2, 1, 1), stride=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.features(out)
        out = self.conv_last(out)

        if self.width_mult == 0.25: out = self.lateral_conv(out)
        
        #-------------------------------------------------------------------------------#
        # building classifier, From the original shuffleNetV2 work, not used in YOWO    #
        # due to CFAM module.                                                           #
        #-------------------------------------------------------------------------------#
        # out = F.avg_pool3d(out, out.data.size()[-3:])
        # out = out.view(out.size(0), -1)
        # out = self.classifier(out)
        #-------------------------------------------------------------------------------#

        if out.size(2) == 2:
            out = self.avgpool(out)

        return out

    def listed_output(self, x) -> Tuple[torch.Tensor]:
        """This method is an alternative to the forward method for getting the output of the
        intermediate levels in addition to the normal model output.

        Args:
            x (_type_): The model input.

        Returns:
            Tuple[torch.Tensor]: normal model output and the flattened and concatenated output of the intermediate layers.
        """
        out_dict = dict()
        
        out = self.conv1(x)
        out = self.maxpool(out)
        if self.dst_layers == 5: out_dict[5] = out

        for i in range(sum(self.stage_repeats)):
            out = self.features[i](out)
            
            if i == 3 and self.dst_layers >= 4:
                out_dict[4] = self.project_conv[2](out) if self.width_mult == 2.0 else out
            elif i == 11 and self.dst_layers >= 3:
                out_dict[3] = self.project_conv[1](out) if self.width_mult == 2.0 else out
            elif i == 15 and self.dst_layers >= 2:
                out_dict[2] = self.project_conv[0](out) if self.width_mult == 2.0 else out
        
        out = self.conv_last(out)
        if self.width_mult == 0.25: out = self.lateral_conv(out)

        if out.size(2) == 2:
            out = self.avgpool(out)

        # The output size (N, 2048, 1, 7, 7) is averaged over the (7, 7) portion to get to (N, 2048)
        if self.dst_layers >= 1: out_dict[1] = out

        return out, out_dict
    
    def load_pretrained_model(self, checkpoint_dir, freeze: bool=True) -> None:
        #-----------------------------------------------------------------------------------#
        # Loading the pretrained model and modifying the checkpoint keys.                   #
        #-----------------------------------------------------------------------------------#
        saved_model = torch.load(checkpoint_dir)['state_dict']
        saved_model_new_keys = OrderedDict()
        for key, item in saved_model.items():
            saved_model_new_keys[key.replace('module.', '')] = item
        
        #-----------------------------------------------------------------------------------#
        # Replacing the backbone with the loaded weights and making sure that the gradient  #
        # is properly set.                                                                  #
        #-----------------------------------------------------------------------------------#
        if self.width_mult == 0.25:
            saved_model_new_keys = {k: v for k, v in saved_model_new_keys.items() if 'conv_last' not in k}
        self.load_state_dict(saved_model_new_keys, strict=False)
        for name, param in self.named_parameters():
            if not freeze:
                param.requires_grad = True
            else:
                param.requires_grad = 'conv_last' in name

    def build_res_blocks(self, input_channel:int, width_mult: float) -> List:
        features = []

        for idx in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idx]
            output_channel = self.stage_out_channels[idx+2]
            
            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride)) #, keep_tem_res=(width_mult == 0.25)))
                input_channel = output_channel
        
        return features, input_channel

    @staticmethod
    def build_project_covs(distill_layers: int) -> nn.ModuleList:
        projection_layers = nn.ModuleList()
        proj_cfg = {        # In reverse order (from the last layer to start)
            1: {'in': 976, 'out': 128, 'k': (1, 1, 1), 's': (1, 1, 1)},
            2: {'in': 488, 'out': 64, 'k': (1, 1, 1), 's': (1, 1, 1)},
            3: {'in': 224, 'out': 32, 'k': (1, 1, 1), 's': (1, 1, 1)},
        }

        for i in range(1, distill_layers):
            projection_layers.append(
                nn.Sequential(nn.Conv3d(proj_cfg[i]['in'], proj_cfg[i]['out'],
                                        proj_cfg[i]['k'], proj_cfg[i]['s'], 0, bias=False), nn.ReLU(inplace=True))
            )
        
        return projection_layers

    @staticmethod
    def width_mult_channels(width_mult: float) -> List:
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.25:
            stage_out_channels = [-1, 24,  32,  64, 128, 256]      # The original last layer = 1024
        elif width_mult == 0.5:
            stage_out_channels = [-1, 24,  48,  96, 192, 1024]
        elif width_mult == 1.0:
            stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(f"width_mult = {width_mult} is not supported for 1x1 Grouped Convolutions")

        return stage_out_channels

    