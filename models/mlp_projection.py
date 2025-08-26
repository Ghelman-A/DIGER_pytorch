import torch
import torch.nn as nn
from typing import  Dict


class MLP(nn.Module):
    def __init__(self, num_channels: int, conv_shape: tuple, use_conv_3d: bool=True, init_conv: bool=True) -> None:
        super().__init__()
        in_ch, k, h, w = conv_shape
        self.init_conv = init_conv
        
        if use_conv_3d:
            if init_conv: self.conv = nn.Conv3d(in_ch, num_channels, kernel_size=(k-1, 1, 1), stride=(2, 1, 1), padding=0, bias=False)
        else:
            self.conv = nn.Conv2d(in_ch, num_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.projector = nn.Sequential(
            nn.Linear(num_channels * h * w, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The projection used to map the output of the two backbones to a lower dimension for knowledge distillation.

        Args:
            x (torch.Tensor): The flattened output of the 3D CNN backbone

        Returns:
            torch.Tensor: The projected output tensor
        """
        if self.init_conv: x = self.conv(x)
        x = torch.flatten(x.squeeze(dim=2), start_dim=1)
        return self.projector(x)
    
