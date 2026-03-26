import math
from typing import Optional, Union, List

import torch
import torch.nn.functional as F

from base import SegmentationModel,SegmentationModel1, SegmentationHead, SegmentationModel_v2
import torch.nn as nn
from encoders import resnet50
from encoders import get_encoder
from decoders import UnetDecoder
from base.init_func import *


class ProjectUNet(SegmentationModel):
    def __init__(
            self,
            encoder_name:str='resnet50',
            encoder_weight:object='imagenet',
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            in_channels: int = 1,
            classes: int = 1,
            **kwargs: object,
    ) -> object:
        super(SegmentationModel, self).__init__()

        self.encoder = get_encoder(encoder_name,in_channels=in_channels,weights=encoder_weight,**kwargs)
        self.decoder = UnetDecoder(
            decoder_channels=decoder_channels,
            encoder_channels=self.encoder.get_out_channels()
        )
        self.projector = nn.Sequential(
            nn.Conv2d(decoder_channels[1], decoder_channels[1], kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(decoder_channels[1], 64, kernel_size=1)
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=None,
            kernel_size=3,
        )
        self.initialize()
        
class ProjectUNet1(SegmentationModel1):
    def __init__(
            self,
            encoder_name:str='resnet50',
            encoder_weight:object='imagenet',
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            in_channels: int = 1,
            classes: int = 1,
            **kwargs: object,
    ) -> object:
        super(SegmentationModel1, self).__init__()

        self.encoder = get_encoder(encoder_name,in_channels=in_channels,weights=encoder_weight,**kwargs)
        self.decoder = UnetDecoder(
            decoder_channels=decoder_channels,
            encoder_channels=self.encoder.get_out_channels()
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=None,
            kernel_size=3,
        )
        self.initialize()
        
# class ProjectUNet_Vx(SegmentationModel):
#     def __init__(
#             self,
#             decoder_channels: List[int] = (256, 128, 64, 32, 16),
#             in_channels: int = 1,
#             classes: int = 1,
#             weights=False,
#             **kwargs: object,
#     ) -> object:
#         super(SegmentationModel, self).__init__()

#         self.encoder = Fusion_Encoder(weights=weights)
#         self.decoder = UnetDecoder(
#             decoder_channels=decoder_channels,
#             encoder_channels=self.encoder.get_out_channels()
#         )
#         self.projector = nn.Sequential(
#             nn.Conv2d(decoder_channels[1], decoder_channels[1], kernel_size=1),
#             nn.PReLU(),
#             nn.Conv2d(decoder_channels[1], 64, kernel_size=1)
#         )
#         self.segmentation_head = SegmentationHead(
#             in_channels=decoder_channels[-1],
#             out_channels=classes,
#             activation=None,
#             kernel_size=3,
#         )
#         self.initialize()

class ProjectUNet_v2(SegmentationModel_v2):
    def __init__(
            self,
            encoder_name: str = 'resnet50',
            encoder_weight: object = 'imagenet',
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            in_channels: int = 1,
            classes: int = 1,
            **kwargs: object,
    ) -> object:
        super(SegmentationModel_v2, self).__init__()

        self.encoder = get_encoder(encoder_name, in_channels=in_channels, weights=encoder_weight, **kwargs)
        self.decoder = UnetDecoder(
            decoder_channels=decoder_channels,
            encoder_channels=self.encoder.get_out_channels()
        )
        self.projector_0 = nn.Sequential(
            nn.Conv2d(decoder_channels[0], decoder_channels[0], kernel_size=1),
            nn.PReLU(),
        )
        self.projector_1 = nn.Sequential(
            nn.Conv2d(decoder_channels[1], decoder_channels[1], kernel_size=1),
            nn.PReLU(),
        )
        self.projector_2 = nn.Sequential(
            nn.Conv2d(decoder_channels[2], decoder_channels[2], kernel_size=1),
            nn.PReLU(),
        )
        self.projector_3 = nn.Sequential(
            nn.Conv2d(decoder_channels[3], decoder_channels[3], kernel_size=1),
            nn.PReLU(),
        )
        self.projector_4 = nn.Sequential(
            nn.Conv2d(decoder_channels[4], decoder_channels[4], kernel_size=1),
            nn.PReLU(),
        )
        self.segmentation_head_0 = SegmentationHead(
            in_channels=decoder_channels[0],
            out_channels=classes,
            activation=None,
            kernel_size=3,
        )
        self.segmentation_head_1 = SegmentationHead(
            in_channels=decoder_channels[1],
            out_channels=classes,
            activation=None,
            kernel_size=3,
        )
        self.segmentation_head_2 = SegmentationHead(
            in_channels=decoder_channels[2],
            out_channels=classes,
            activation=None,
            kernel_size=3,
        )
        self.segmentation_head_3 = SegmentationHead(
            in_channels=decoder_channels[3],
            out_channels=classes,
            activation=None,
            kernel_size=3,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=None,
            kernel_size=3,
        )
        self.initialize()

if __name__ == '__main__':
    x = torch.rand(2,1,256,256).cuda()
    model = ProjectUNet_Vx(classes=4).cuda()
    _, *proj_list = model(x)
    for feat in proj_list:
        print(feat.shape)
    params = sum([p.numel() for p in model.parameters()])
    print(params)
