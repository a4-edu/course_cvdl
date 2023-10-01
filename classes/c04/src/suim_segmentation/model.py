import torch
from torch import nn
from einops import rearrange, reduce
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights


class ShuffleNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.md = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
    
    def forward(self, x):
        c1 = self.md.conv1(x) # 1/2
        c2 = self.md.maxpool(c1) # 1/4
        c3 = self.md.stage2(c2)# 1/8
        c4 = self.md.stage3(c3)# 1/16
        c5 = self.md.stage4(c4)# 1/32
        return c3, c4, c5


class ConvBNReLU(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 use_normalization: bool = True,
                 use_activation: bool = True,
                 conv_kwargs: dict = {},
                 bn_kwargs: dict = {}
                 ):

        super().__init__()
        self.seq = nn.Sequential()
        self.seq.add_module("conv1", nn.Conv2d(in_channels,
                                              out_channels,
                                              kernel_size=1,
                                              padding=0,
                                              **conv_kwargs))
        if kernel_size != 1:
            self.seq.add_module("convK", nn.Conv2d(out_channels,
                                              out_channels,
                                              kernel_size=kernel_size,
                                              padding=padding,
                                              groups=out_channels,
                                              **conv_kwargs))
        if use_normalization:
            self.seq.add_module("bn", nn.BatchNorm2d(out_channels, **bn_kwargs))
        if use_activation:
            self.seq.add_module("relu", nn.ReLU())

    def forward(self, x):
        return self.seq(x)


def upscale(x: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """
    Увеличение в s раз через билинейную интерполяцию
    [B C H W] -> [B C sH sW]
    """
    if scale == 1:
        return x
    return torch.nn.functional.interpolate(x, scale_factor=scale, mode='bilinear')


def downscale(x: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """
    Уменьшение в s раз через MeanPool
    [B C H W] -> [B C H/s W/s] 
    """
    if scale == 1:
        return x
    return reduce(x, 'b c (h s1) (w s2) -> b c h w', 'mean', s1=scale, s2=scale)
    

class SPPBranch(nn.Module):
    def __init__(self, in_channels: int, out_channels:int , scale: int):
        super().__init__()
        self.scale = scale
        self.conv =  ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        x = downscale(x, self.scale)
        x = self.conv(x)
        x = upscale(x, self.scale)
        return x


class SimplePyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = SPPBranch(in_channels=in_channels, out_channels=out_channels, scale=1)        
        self.branch2 = SPPBranch(in_channels=in_channels, out_channels=out_channels, scale=2)
        self.branch4 = SPPBranch(in_channels=in_channels, out_channels=out_channels, scale=4)

        self.conv_final = ConvBNReLU(in_channels=out_channels, out_channels=out_channels)
    
    def forward(self, x):
        return self.conv_final(
            self.branch1(x) + self.branch2(x) + self.branch4(x)
        )


class UnifiedAttentionModule(nn.Module):
    """
    UAFM как на картинке
    """
    def __init__(self, attn: nn.Module):
        super().__init__()
        self.attn = attn
        
    def forward(self, x, x_up):
        assert x.size(2) == x_up.size(2) * 2, (x.shape, x_up.shape)
        assert x.size(3) == x_up.size(3) * 2, (x.shape, x_up.shape)
        
        x_up = upscale(x_up)
        alpha = self.attn(x, x_up)
        return x * alpha + (1 - alpha) * x_up
    

class UnifiedAttentionModuleWithProjections(UnifiedAttentionModule):
    """
    UAFM с предварительной проекцией в одинаковое количество каналов
    """
    def __init__(self, in_channels: int, in_channels_up: int, out_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.conv = ConvBNReLU(in_channels=in_channels, out_channels=out_channels)
        self.conv_up = ConvBNReLU(in_channels=in_channels_up, out_channels=out_channels)
    
    def forward(self, x, x_up):
        x = self.conv(x)
        x_up = self.conv_up(x_up)
        return super().forward(x, x_up)


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=4 * in_channels, out_channels=in_channels, kernel_size=3, padding=1)

    def forward(self, x_low, x_high):
        p_xlow_max = reduce(x_low, 'b c h w -> b c 1 1', reduction='max')
        p_xlow_mean = reduce(x_low, 'b c h w -> b c 1 1', reduction='mean')
        p_xhigh_max = reduce(x_high, 'b c h w -> b c 1 1', reduction='max')
        p_xhigh_mean = reduce(x_high, 'b c h w -> b c 1 1', reduction='mean')     

        y = rearrange(
            [p_xlow_max, p_xlow_mean, p_xhigh_max, p_xhigh_mean],
            'N b c 1 1 -> b (N c) 1 1', N=4
        )
        y = self.conv(y)
        return y.sigmoid()


class LiteSeg(nn.Module):
    def __init__(self, 
            encoder: nn.Module, 
            num_classes: int, 
            encoder_channels = [116, 232, 464],
            decoder_channels = [128, 64, 32]
        ):
        super().__init__()

        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        
        self.encoder = encoder
        self.c6 = SimplePyramidPooling(in_channels=encoder_channels[-1], out_channels=decoder_channels[0])
        self.c7 = UnifiedAttentionModuleWithProjections(
            in_channels_up=decoder_channels[0],
            in_channels=encoder_channels[-2],
            out_channels=decoder_channels[1],
            attn=ChannelAttentionModule(in_channels=decoder_channels[1]),
        )
        self.c8 = UnifiedAttentionModuleWithProjections(
            in_channels_up=decoder_channels[1],
            in_channels=encoder_channels[-3],
            out_channels=decoder_channels[2],
            attn=ChannelAttentionModule(in_channels=decoder_channels[2]),
        )
        self.head = nn.Sequential( 
            nn.Conv2d(in_channels=decoder_channels[2], out_channels=num_classes, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        #1/8, 1/16, 1/32
        x_c3, x_c4, x_c5 = self.encoder(x)
        
        assert x_c3.shape[1] == self.encoder_channels[0], (x_c5.shape, self.encoder_channels)
        assert x_c4.shape[1] == self.encoder_channels[1], (x_c4.shape, self.encoder_channels)
        assert x_c5.shape[1] == self.encoder_channels[2], (x_c3.shape, self.encoder_channels)
        
        y_c6 = self.c6(x_c5)
        y_c7 = self.c7(x_c4, y_c6)
        y_c8 = self.c8(x_c3, y_c7)

        y = upscale(y_c8, scale=8) #1/4
        return self.head(y)


class SuimModel(LiteSeg):
    def __init__(self):
        super().__init__(ShuffleNetEncoder(), num_classes=8, encoder_channels=[48, 96, 192])
        

