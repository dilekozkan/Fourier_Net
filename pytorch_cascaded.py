import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
#         nn.Dropout(p=0.2),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self, in_channels, n_class, f_size):
        super().__init__()
                
        self.dconv_down1 = double_conv(in_channels, f_size)
        self.dconv_down2 = double_conv(f_size, f_size*2)
        self.dconv_down3 = double_conv(f_size*2, f_size*4)
        self.dconv_down4 = double_conv(f_size*4, f_size*8)  
        self.dconv_down5 = double_conv(f_size*8, f_size*16) 
        self.maxpool = nn.MaxPool2d(2)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.dconv_up4 = double_conv(f_size*8 + f_size*16, f_size*8)
        self.dconv_up3 = double_conv(f_size*4 + f_size*8, f_size*4)
        self.dconv_up2 = double_conv(f_size*2 + f_size*4, f_size*2)
        self.dconv_up1 = double_conv(f_size + f_size*2, f_size)
        self.conv_last = nn.Conv2d(f_size, n_class, 1)

              
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3) 
        
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4) 
        
        x_bottleneck = self.dconv_down5(x)
        
        x = self.upsample(x_bottleneck)        
        x = torch.cat([x, conv4], dim=1)
        
        x = self.dconv_up4(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)       

        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)   

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)         
        
        x = self.dconv_up1(x)
       
        out = self.conv_last(x)

        return out
    
    
class cascaded(nn.Module):
    def __init__(self, num_class, feature_map_size):
        super().__init__()
        
        self.unet1 = UNet(1,1, f_size=feature_map_size) #for binary, changed from 2 to 1
        self.unet2 = UNet(2,num_class, f_size=feature_map_size) #changed 
       
    def forward(self, x):
        unet1 = self.unet1(x)
        x = torch.cat([x, unet1], dim=1) #channel boyunca
        unet2 = self.unet2(x) 

        return unet1, unet2
