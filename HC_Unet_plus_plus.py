import torch
import torch.nn as nn
import torch.nn.functional as F

#ConvBlock 기본 블록
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


#DPFFB (Deep Parallel Feature Fusion Block)
class DPFFB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upper = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.lower = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        up = self.upper(x)
        low = self.lower(x)
        return torch.cat([up, low], dim=1)

#SEBlock (Squeeze-and-Excitation)
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.fc(self.pool(x))
        return x * weight

#BlurPool
class BlurPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]])
        kernel = kernel / kernel.sum()
        self.register_buffer('filter', kernel[None, None, :, :].repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.filter, stride=2, padding=1, groups=self.channels)

#HC-Unet++ 메인 구조 (Encoder + Decoder)
class HCUnetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, base=64, num_classes=1):
        super().__init__()
        ch = base

        # Encoder
        self.conv0_0 = ConvBlock(in_channels, ch)
        self.pool0 = BlurPool(ch)
        self.conv1_0 = ConvBlock(ch, ch*2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2_0 = ConvBlock(ch*2, ch*4)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3_0 = ConvBlock(ch*4, ch*8)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4_0 = DPFFB(ch*8)  # bottleneck with DPFFB

        # Decoder (nested skip)
        self.up3_1 = nn.ConvTranspose2d(ch*16, ch*8, 2, stride=2)
        self.conv3_1 = ConvBlock(ch*16, ch*8)

        self.up2_2 = nn.ConvTranspose2d(ch*8, ch*4, 2, stride=2)
        self.conv2_2 = ConvBlock(ch*8, ch*4)
        self.se2_2 = SEBlock(ch*4)

        self.up1_3 = nn.ConvTranspose2d(ch*4, ch*2, 2, stride=2)
        self.conv1_3 = ConvBlock(ch*4, ch*2)

        self.up0_4 = nn.ConvTranspose2d(ch*2, ch, 2, stride=2)
        self.conv0_4 = ConvBlock(ch*2, ch)

        self.final = nn.Conv2d(ch, num_classes, 1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool0(x0_0))
        x2_0 = self.conv2_0(self.pool1(x1_0))
        x3_0 = self.conv3_0(self.pool2(x2_0))
        x4_0 = self.conv4_0(self.pool3(x3_0))

        x3_1 = self.conv3_1(torch.cat([self.up3_1(x4_0), x3_0], dim=1))
        x2_2 = self.conv2_2(torch.cat([self.up2_2(x3_1), x2_0], dim=1))
        x2_2 = self.se2_2(x2_2)
        x1_3 = self.conv1_3(torch.cat([self.up1_3(x2_2), x1_0], dim=1))
        x0_4 = self.conv0_4(torch.cat([self.up0_4(x1_3), x0_0], dim=1))

        return self.final(x0_4)

'''
# 모델 디버깅용
if __name__ == '__main__':
    model = HCUnetPlusPlus()
    dummy = torch.randn(1, 3, 384, 640)  # 테스트 입력 (배치1, 3채널, 384x640)
    out = model(dummy)
    print("출력 shape:", out.shape)  # (1, 1, 384, 640)
'''