import torch.nn as nn
from typing import Optional, Union, List

from smp.get_encoder import *
from smp.decoder import UnetDualDecoder
from smp.heads import SegmentationHead



def initialize_decoder(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)



def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)



class RGBTail(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(RGBTail, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace = True)

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
        out += residual
        out = self.relu(out)
        return out




class CADDN(nn.Module):
    def __init__(        
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = None,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        imgage_decoder_dropout: float = 0,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()
    
        # Ecoder
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # Image decoder
        self.decoderI = UnetDualDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            skip_dropout=imgage_decoder_dropout,
            n_blocks=encoder_depth,
            use_dual_skip=False,
            return_decoded_feat=True,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        self.reconstruction_head = RGBTail(inplanes=decoder_channels[-1], planes=8, stride=1)
        self.out = nn.Sequential(nn.Conv2d(decoder_channels[-1], in_channels, 3, 1, 1, bias=False), 
                                 nn.Tanh())
        
        # Shadow decoder
        self.decoderS = UnetDualDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            skip_dropout=0,
            n_blocks=encoder_depth,
            use_dual_skip=True,
            return_decoded_feat=True,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        initialize_decoder(self.decoderI)
        initialize_decoder(self.decoderS)
        initialize_head(self.segmentation_head)
        initialize_head(self.reconstruction_head)
        initialize_head(self.out)


    def forward(self,x):
        features = self.encoder(x)
        decoderI_output, skips2 = self.decoderI(features)
        image = self.reconstruction_head(decoderI_output)
        image = self.out(image)
        decoderS_output, _ = self.decoderS(features, skips2)
        masks = self.segmentation_head(decoderS_output)
        return (masks, image)
