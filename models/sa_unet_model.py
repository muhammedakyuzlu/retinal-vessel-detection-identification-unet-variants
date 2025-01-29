import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.encode = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.encode(x)
        p = self.pool(x)
        return x, p

class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True) 
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv(concat)
        scaled_x = self.sigmoid(attention)        
        return scaled_x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2)
        self.spatial_attention = SpatialAttentionModule(mid_channels)  # Initialize here with the correct number of channels
        self.decode = ConvBlock(mid_channels*2, out_channels)  # Note that it's mid_channels*2 because of concatenation

    def forward(self, x, skip):
        x = self.up(x)
        skip_attention = self.spatial_attention(skip) * skip  # Apply spatial attention
        x = torch.cat([x, skip_attention], dim=1)  # Concatenate along the channel dimension
        return self.decode(x)

class SAUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(SAUNet, self).__init__()

        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        self.dec1 = DecoderBlock(1024, 512, 256)
        self.dec2 = DecoderBlock(256, 256, 128)
        self.dec3 = DecoderBlock(128, 128, 64)
        self.dec4 = DecoderBlock(64, 64, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        enc1, x = self.enc1(x)
        enc2, x = self.enc2(x)
        enc3, x = self.enc3(x)
        enc4, x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec1(x, enc4)
        x = self.dec2(x, enc3)
        x = self.dec3(x, enc2)
        x = self.dec4(x, enc1)

        return self.final_conv(x)

if __name__ == '__main__':

    model = SAUNet(in_channels=3, num_classes=2)  # Example for 3 input channels and 2 classes
    input_tensor = torch.rand(1, 3, 512, 512)  # Example input
    output = model(input_tensor)
    print(output.shape)  # Check output shape