# U-Net. Taken from:
# https://github.com/chinakook/U-Net

import mxnet.gluon.nn as nn


def conv_block(channels, kernel_size):
    out = nn.HybridSequential()
    out.add(
        nn.Conv2D(channels, kernel_size, padding=1, use_bias=False),
        nn.BatchNorm(),
        nn.Activation('relu')
    )
    return out


def down_block(channels):
    out = nn.HybridSequential()
    out.add(
        conv_block(channels, 3),
        conv_block(channels, 3)
    )
    return out


class UpBlock(nn.HybridBlock):
    def __init__(self, channels, shrink=True, **kwargs):
        super(UpBlock, self).__init__(**kwargs)
        self.upsampler = nn.Conv2DTranspose(channels=channels, kernel_size=4, strides=2, 
                                            padding=1, use_bias=False)
        self.conv1 = conv_block(channels, 1)
        self.conv3_0 = conv_block(channels, 3)
        if shrink:
            self.conv3_1 = conv_block(int(channels/2), 3)
        else:
            self.conv3_1 = conv_block(channels, 3)

    def hybrid_forward(self, F, x, s):
        x = self.upsampler(x)
        x = self.conv1(x)
        x = F.relu(x)
        # shape_x = F.shape_array(x)
        # shape_s = F.shape_array(s)
        # x = F.slice(x, begin=(0, 0, (shape_x[2] - shape_s[2])/2, (shape_x[3] - shape_s[3])/2),
        #             end=(shape_x[0], shape_x[1], -(shape_x[2] - shape_s[2])/2, -(shape_x[3] - shape_s[3])/2))
        x = F.Crop(*[x, s], center_crop=True)
        x = s + x
        x = self.conv3_0(x)
        x = self.conv3_1(x)
        return x


class Unet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Unet, self).__init__(**kwargs)
        with self.name_scope():
            self.d0 = down_block(64)
            
            self.d1 = nn.HybridSequential()
            self.d1.add(nn.MaxPool2D(2,2,ceil_mode=True), down_block(128))
            
            self.d2 = nn.HybridSequential()
            self.d2.add(nn.MaxPool2D(2,2,ceil_mode=True), down_block(256))
            
            self.d3 = nn.HybridSequential()
            self.d3.add(nn.MaxPool2D(2,2,ceil_mode=True), down_block(512))
            
            self.d4 = nn.HybridSequential()
            self.d4.add(nn.MaxPool2D(2,2,ceil_mode=True), down_block(1024))
            
            self.u3 = UpBlock(512, shrink=True)
            self.u2 = UpBlock(256, shrink=True)
            self.u1 = UpBlock(128, shrink=True)
            self.u0 = UpBlock(64, shrink=False)
            
            self.conv = nn.Conv2D(2, 1)
            
    def hybrid_forward(self, F, x):
        x0 = self.d0(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)

        y3 = self.u3(x4,x3)
        y2 = self.u2(y3,x2)
        y1 = self.u1(y2,x1)
        y0 = self.u0(y1,x0)
        
        out = F.softmax(self.conv(y0), axis=1)
        
        return out
