import torch
import torch.nn as nn
from torchvision import models

from models.yowo_cfam import CFAMBlock
from models import darknet, darknet53
from models.shuffleNetV2 import ShuffleNetV2
from models.mlp_projection import MLP
from collections import OrderedDict
from typing import Tuple, Dict, List

"""
DIGER model used in spatiotemporal action localization
"""


class DIGER(nn.Module):

    def __init__(self, cfg):
        super(DIGER, self).__init__()
        self.cfg = cfg.arch_cfg
        self.tg_type = cfg.arch_cfg.tg_cfg.type
        self.aux_cfg = self.cfg.tg_cfg.aux_distill

        #---------------------   2D and 3D Backbone      -------------------#
        self.backbone_2d, num_ch_2d = self.load_2d_backbone(self.cfg.backbone_2d)
        self.backbone_3d, num_ch_3d = self.load_3d_backbone(self.cfg.backbone_3d)
        if self.tg_type == 'concat': num_ch_3d *= 2

        #---------------------   2D and 3D AUX Backbone   ------------------#
        if self.tg_type == 'aux_distill':
            self.backbone_3d_aux, _ = self.load_3d_backbone(self.aux_cfg.aux_3d)
            self.projector = self.load_3d_projections(cfg)

            if not self.aux_cfg.only_3d:
                self.backbone_2d_aux, _ = self.load_2d_backbone(self.aux_cfg.aux_2d)
                self.projector_2d = self.load_2d_projections(cfg)
        
        #--------------------   Attention & Final Conv    ------------------#
        self.cfam = CFAMBlock(num_ch_2d+num_ch_3d, 1024)
        self.conv_final = nn.Conv2d(1024, 5*(3+4+1), kernel_size=1, bias=False)

    def forward(self, input):
        if type(input) in [tuple, list]:        # With TG input
            rgb_input, tg_input = input

            x_3d = rgb_input
            x_2d = rgb_input[:, :, -1, :, :]            # Last frame of the clip

            if self.tg_type == 'distill':
                x_2d = self.backbone_2d(x_2d)
                x_3d, distill_rgb = self.backbone_3d.listed_output(x_3d, self.cfg.tg_cfg.distill_layers)
                
                _, distill_tg = self.backbone_3d.listed_output(tg_input, self.cfg.tg_cfg.distill_layers)
                
                x_3d = torch.squeeze(x_3d, dim=2)
                x = torch.cat((x_3d, x_2d), dim=1)
                x = self.cfam(x)
                out = self.conv_final(x)

                return out, torch.stack([distill_rgb, distill_tg.detach()], dim=-1)
            
            elif self.tg_type == 'aux_distill':
                
                if self.aux_cfg.only_3d:
                    #-------------- Pass RGB through YOWO ---------------#
                    x_2d = self.backbone_2d(x_2d)
                    x_3d, distill_3d_rgb = self.backbone_3d.listed_output(x_3d)
                    out1 = self.yowo_cfam(x_3d, x_2d)

                    #-------------- Pass TG through 3D CNN and CFAM (No update for 2D CNN) ---------------#
                    x_3d_tg, distill_3d_tg = self.backbone_3d_aux.listed_output(tg_input)
                    out2 = self.yowo_cfam(x_3d_tg, x_2d.detach())
                    
                    distill_out = self.distill_out(self.projector, distill_3d_rgb, distill_3d_tg)
                
                else:
                    tg_2d = tg_input[:, :, -1, :, :]

                    #-------------- Pass RGB through YOWO ---------------#
                    x_2d, distill_2d_rgb = self.backbone_2d.listed_output(x_2d, self.cfg.tg_cfg.distill_layers)
                    x_3d, distill_3d_rgb = self.backbone_3d.listed_output(x_3d)
                    out1 = self.yowo_cfam(x_3d, x_2d)

                    #-------------- Pass TG through YOWO ---------------#
                    x_2d_tg, distill_2d_tg = self.backbone_2d_aux.listed_output(tg_2d, self.cfg.tg_cfg.distill_layers)
                    x_3d_tg, distill_3d_tg = self.backbone_3d_aux.listed_output(tg_input)
                    out2 = self.yowo_cfam(x_3d_tg, x_2d.detach()), self.yowo_cfam(x_3d.detach(), x_2d_tg)
                    
                    distill_out = list(zip(self.distill_out(self.projector, distill_3d_rgb, distill_3d_tg),
                                           self.distill_out(self.projector_2d, distill_2d_rgb, distill_2d_tg)))

                return out1, distill_out, out2

            elif self.tg_type in ['concat', 'aggregate']:
                x_2d = self.backbone_2d(x_2d)
                x_3d = self.backbone_3d(x_3d)
                x_3d_tg = self.backbone_3d(tg_input)
                
                x_3d = torch.cat([x_3d_tg, x_3d], dim=1) if self.tg_type == 'concat' else (x_3d + x_3d_tg) / 2

                x_3d = torch.squeeze(x_3d, dim=2)
                x = torch.cat((x_3d, x_2d), dim=1)
                x = self.cfam(x)
                out = self.conv_final(x)

                return out
            
            else:
                raise ValueError('Wrong TG integration type selected!')

        else:   # Without TG input
            rgb_input = input
        
            x_3d = rgb_input
            x_2d = rgb_input[:, :, -1, :, :]            # Last frame of the clip
            x_2d = self.backbone_2d(x_2d)
            x_3d = self.backbone_3d(x_3d)

            x_3d = torch.squeeze(x_3d, dim=2)

            x = torch.cat((x_3d, x_2d), dim=1)
            x = self.cfam(x)
            out = self.conv_final(x)

            return out

    @staticmethod
    def load_2d_backbone(cfg: Dict) -> Tuple[nn.Module, int]:
        """Loading the main and auxiliary 2D backbones

        Args:
            cfg (Dict): _description_

        Raises:
            ValueError: Raises error if wrong backbone is specified.

        Returns:
            Tuple[nn.Module, int]: Returns the model instance the number of 2D channels needed to configure the CFAM module.
        """

        if cfg.model == "darknet":
            backbone_2d = darknet.Darknet()
            num_ch_2d = 425                                         # Number of output channels
        elif cfg.model == "darknet53":
            backbone_2d = darknet53.Darknet53()
            num_ch_2d = 255
        else:
            raise ValueError("Wrong backbone_2d model is requested!")
            
        if cfg.pretrained:                     #  load pretrained weights on COCO dataset
            backbone_2d.load_pretrained_model(cfg.load_dir, cfg.freeze_backbone)
        
        return backbone_2d, num_ch_2d

    def load_3d_backbone(self, cfg: Dict) -> Tuple[nn.Module, int]:
        """Loading the main and auxiliary 3D backbones

        Args:
            cfg (Dict): _description_

        Raises:
            ValueError: Raises error if wrong backbone is specified.

        Returns:
            Tuple[nn.Module, int]: Returns the model instance the number of 3D channels needed to configure the CFAM module.
        """

        if cfg.model == "resnet18":
            backbone_3d = models.video.r3d_18(pretrained=cfg.pretrained)
            num_ch_3d = 512                                         # Number of output channels

        elif cfg.model == "shuffleNetV2":            
            backbone_3d = ShuffleNetV2(self.cfg.tg_cfg.distill_layers, width_mult=cfg.model_width)
            num_ch_3d = backbone_3d.stage_out_channels[-1]                                        # Number of output channels
            if cfg.pretrained: backbone_3d.load_pretrained_model(cfg.load_dir, cfg.freeze_backbone)

        else:
            raise ValueError("Wrong backbone_3d model is requested!")        

        return backbone_3d, num_ch_3d

    @staticmethod
    def load_3d_projections(cfg: Dict) -> nn.ModuleList:
        proj_layers = nn.ModuleList()
        dst_layers = cfg.arch_cfg.tg_cfg.distill_layers
        map_h = cfg.localization_cfg.bbone_h
        map_w = cfg.localization_cfg.bbone_w
        
        for i in range(dst_layers):
            if i == 0:
                proj_layers.append(MLP(128, (2048, 2, map_h, map_w)))
            elif i == 1:
                proj_layers.append(MLP(128, (128, 2, map_h, map_w), init_conv=False))
            elif i == 2:
                proj_layers.append(MLP(32, (64, 2, map_h * 2, map_w * 2)))
            elif i == 3:
                proj_layers.append(MLP(16, (32, 4, map_h * 4, map_w * 4))) 
            elif i == 4:
                proj_layers.append(MLP(8, (24, 8, map_h * 8, map_w * 8)))                
        
        return proj_layers

    @staticmethod
    def load_2d_projections(cfg: Dict) -> nn.ModuleList:
        proj_layers = nn.ModuleList()
        dst_layers = cfg.arch_cfg.tg_cfg.distill_layers
        
        for i in range(dst_layers):
            if i == 0:
                proj_layers.append(MLP(128, (425, 2, 7, 7), use_conv_3d=False))
            elif i == 1:
                proj_layers.append(MLP(128, (1024, 2, 7, 7), use_conv_3d=False))
            elif i == 2:
                proj_layers.append(MLP(32, (512, 2, 14, 14), use_conv_3d=False))
            elif i == 3:
                proj_layers.append(MLP(16, (256, 4, 28, 28), use_conv_3d=False)) 
            elif i == 4:
                proj_layers.append(MLP(8, (128, 8, 56, 56), use_conv_3d=False))                
        
        return proj_layers

    def distill_out(self, projector, rgb_out: Dict, tg_out: Dict) -> List:
        """This method applies the MLP projections on the output of the intermediate layers of the networks.
        The lateral convolution are already implemented in the backbone models.

        Args:
            projector (_type_): The MLP projection of the selected distillation layer.
            rgb_out (Dict): _description_
            tg_out (Dict): _description_

        Returns:
            List: List of projections of all layers. For each layer, the RGB and TG projections are stacked together.
        """
        distill_out = []
        
        for i in range(self.cfg.tg_cfg.distill_layers):
            distill_out.append(torch.stack([projector[i](rgb_out[i+1]), projector[i](tg_out[i+1].detach())], dim=-1))
            
        return distill_out

    def yowo_cfam(self, x_3d: torch.Tensor, x_2d: torch.Tensor) -> torch.Tensor:
        """A short method just to make the forward method is less crowded.

        Args:
            x_3d (torch.Tensor): Output of the 3D CNN backbone.
            x_2d (torch.Tensor): Output of the 2D CNN backbone.

        Returns:
            torch.Tensor: Final yowo output.
        """
        x_3d = torch.squeeze(x_3d, dim=2)
        x = torch.cat((x_3d, x_2d), dim=1)
        x = self.cfam(x)
        return self.conv_final(x)
    
    def load_pretrained_model(self, checkpoint_dir: str) -> None:
        saved_model = torch.load(checkpoint_dir)['state_dict']
        new_model = OrderedDict()
        for name, param in saved_model.items():
            new_model[name.replace('model.', '', 1)] = param        # Only replace the first 'model.'
        
        self.load_state_dict(new_model, strict=True)
