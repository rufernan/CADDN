
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax

from smp import modules as md


def INF(B,H,W, GPU=0):
     return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1).to(torch.device('cuda:'+str(GPU)))


class CrossAttention(nn.Module):

    def __init__(self, in_dim):
        super(CrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        return self.gamma*(out_H + out_W) + x





class TransferBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TransferBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x





class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        skip2_channels=0,
        skip_dropout=0,
        use_batchnorm=True,
        attention_type=None,
    ):
        
        self.in_channels=in_channels
        self.skip_channels=skip_channels
        self.out_channels=out_channels
        self.skip2_channels=skip2_channels
        self.skip_dropout=skip_dropout
        
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels + skip2_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels + skip2_channels)
        
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)
        
        self.P = TransferBlock(in_ch=in_channels + skip_channels, out_ch=in_channels + skip_channels)
        self.att_skip2post = CrossAttention(in_channels + skip_channels + skip2_channels) 
        self.drop = nn.Dropout(p=skip_dropout, inplace=False)
        
        

    def forward(self, x, skip=None, skip2=None):
        
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        
        x_ini = x
        
        if skip is not None:

            if self.skip_dropout>0:
                skip = self.drop(skip)

            x = torch.cat([x, skip], dim=1)
            
            if skip2 is not None:

                x = self.P(x)
                x = torch.cat([x, skip2], dim=1)
                x = self.att_skip2post(x)

        x = self.attention1(x)
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.attention2(x)

        return x





class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)



class UnetDualDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        skip_dropout=0,
        use_dual_skip=False,
        return_decoded_feat=False,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        self.return_decoded_feat = return_decoded_feat

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]
        
        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels
        
        
        if use_dual_skip:
            skip2_channels =  list(out_channels[:-1]) + [0]
        else:
            skip2_channels = [0 for i in range(len(out_channels))]
            

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, skip2_ch, skip_dropout, **kwargs)
            for in_ch, skip_ch, out_ch, skip2_ch in zip(in_channels, skip_channels, out_channels, skip2_channels)
        ]
        self.blocks = nn.ModuleList(blocks)



    def forward(self, features, skips2=None):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from the encoder head

        head = features[0]
        skips = features[1:]
        new_skips = []
        
        if skips2 is None:
            skips2 = [None for i in range(len(skips))]
            
        x = self.center(head)

        for i, decoder_block in enumerate(self.blocks):

            skip = skips[i] if i < len(skips) else None
            skip2 = skips2[i] if i < len(skips2) else None

            x = decoder_block(x, skip, skip2)           
            new_skips.append(x)

        if self.return_decoded_feat:
            return x, new_skips[:-1]
        else:
            return x
