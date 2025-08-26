import torch
from torchvision import models
import torch.nn as nn
from collections import OrderedDict


class CustomResNet(nn.Module):
    """
        This class loads the 3D ResNet model and replaces the FC layers with a convnet
        for producing the localization and classification outputs. Furthermore, the 
        forward functionality is also implemented in here.
    """
    def __init__(self, cfg):
        super(CustomResNet, self).__init__()
        self.cfg = cfg
        self.train_mode = cfg.train_mode
        self.model_cfg = cfg[self.train_mode]
        base_model = models.video.r3d_18(pretrained=cfg.pre_trained_resnet)

        self.backbone = torch.nn.Sequential(OrderedDict([
            ('stem', base_model.stem),
            ('layer1', base_model.layer1),
            ('layer2', base_model.layer2),
            ('layer3', base_model.layer3),
            ('layer4', base_model.layer4),
            ('avgpool', nn.AdaptiveAvgPool3d((1, cfg.localization_cfg.bbone_h, cfg.localization_cfg.bbone_w)))
        ]))

        num_anc = cfg.localization_cfg.num_anchors
        num_classes = 3
        self.conv_final = nn.Conv2d(512, num_anc * (4 + 1 + num_classes), kernel_size=1, bias=False)
    
    def forward(self, x):
        x = self.backbone(x)
        return self.conv_final(x.squeeze(dim=2))

    def load_pretrained_model(self, checkpoint_dir) -> None:
        #-----------------------------------------------------------------------------------#
        # Loading the pretrained model and modifying the checkpoint keys except the last fc #
        # layer and avgpool layer which will be dropped.                                    #
        #-----------------------------------------------------------------------------------#
        saved_model = torch.load(checkpoint_dir)['state_dict']
        saved_model_new_keys = OrderedDict()
        for key, item in saved_model.items():
            saved_model_new_keys[key.replace('DIGER_model.', '', 1)] = item
        
        #-----------------------------------------------------------------------------------#
        # Replacing the backbone with the loaded weights and making sure that the gradient  #
        # is only set for the case of fine-tuning.                                          #
        #-----------------------------------------------------------------------------------#
        self.load_state_dict(saved_model_new_keys, strict=True)
        for _, param in self.named_parameters():
            param.requires_grad = True

