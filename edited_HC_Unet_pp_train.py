import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

# =================================================================================
# 1. 모델 아키텍처 및 관련 모듈 정의
# (바로 이전 답변에서 완성한 HCUnetPlusPlus와 관련 모듈/함수 전체)
# =================================================================================

# ConvBlock: 기본 블록
class ConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

# DPFFB (Atrous Convolution 적용 버전)
class DPFFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_channels = out_channels // 2
        self.upper = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        self.lower = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        up = self.upper(x)
        low = self.lower(x)
        return torch.cat([up, low], dim=1)

# SEBlock (Squeeze-and-Excitation)
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        weight = self.fc(self.pool(x))
        return x * weight

# BlurPool: Anti-aliasing 다운샘플링
class BlurPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]])
        kernel = kernel / kernel.sum()
        self.register_buffer('filter', kernel[None, None, :, :].repeat(channels, 1, 1, 1))
        self.channels = channels
    def forward(self, x):
        return F.conv2d(x, self.filter, stride=2, padding=1, groups=self.channels)

# 가중치 초기화 함수
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# HC-Unet++ 메인 아키텍처
class HCUnetPlusPlus(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super().__init__()

        filters = [64, 128, 256, 512, 1024]

        # Downsampling
        self.pool0 = BlurPool(filters[0])
        self.pool1 = BlurPool(filters[1])
        self.pool2 = BlurPool(filters[2])
        self.pool3 = BlurPool(filters[3])

        # Upsampling
        self.up1_0 = nn.ConvTranspose2d(filters[1], filters[1], 2, stride=2)
        self.up2_0 = nn.ConvTranspose2d(filters[2], filters[2], 2, stride=2)
        self.up3_0 = nn.ConvTranspose2d(filters[3], filters[3], 2, stride=2)
        self.up4_0 = nn.ConvTranspose2d(filters[4], filters[4], 2, stride=2)

        self.up1_1 = nn.ConvTranspose2d(filters[1], filters[1], 2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(filters[2], filters[2], 2, stride=2)
        self.up3_1 = nn.ConvTranspose2d(filters[3], filters[3], 2, stride=2)
        
        self.up1_2 = nn.ConvTranspose2d(filters[1], filters[1], 2, stride=2)
        self.up2_2 = nn.ConvTranspose2d(filters[2], filters[2], 2, stride=2)

        self.up1_3 = nn.ConvTranspose2d(filters[1], filters[1], 2, stride=2)
        
        # Encoder & Bottleneck
        self.conv0_0 = ConvBlock(input_channels, filters[0], filters[0])
        self.conv1_0 = ConvBlock(filters[0], filters[1], filters[1])
        self.conv2_0 = ConvBlock(filters[1], filters[2], filters[2])
        self.conv3_0 = ConvBlock(filters[2], filters[3], filters[3])
        self.conv4_0 = DPFFB(filters[3], filters[4])

        # Nested Skip Paths (채널 수 재계산 및 수정 완료)
        self.conv0_1 = ConvBlock(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = ConvBlock(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = ConvBlock(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = ConvBlock(filters[3] + filters[4], filters[3], filters[3])
       
        self.conv0_2 = ConvBlock(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = ConvBlock(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = ConvBlock(filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = ConvBlock(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = ConvBlock(filters[1]*3 + filters[2], filters[1], filters[1])

        self.conv0_4 = ConvBlock(filters[0]*4 + filters[1], filters[0], filters[0])
        
        # SEBlock
        self.se3_1 = SEBlock(filters[3])

        # Final Layer
        self.output = nn.Conv2d(filters[0], num_classes, kernel_size=1)

        # 가중치 초기화 적용
        self.apply(init_weights)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool0(x0_0))
        x2_0 = self.conv2_0(self.pool1(x1_0))
        x3_0 = self.conv3_0(self.pool2(x2_0))
        x4_0 = self.conv4_0(self.pool3(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], dim=1))
        
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], dim=1))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], dim=1))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], dim=1))
        x3_1 = self.se3_1(x3_1)

        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], dim=1))
        
        output = self.output(x0_4)
        return output

# =================================================================================
# 3. 데이터셋 및 손실 함수 정의
# =================================================================================

class CrackDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images = sorted([f for f in os.listdir(root_dir) if f.endswith('.jpg')])
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        mask_path = img_path.replace('.jpg', '.png')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = mask / 255.0
        mask = mask.astype('float32')
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)
        return image, mask

# Dice Loss 함수
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

# =================================================================================
# 4. 학습 루프 (최적화된 학습 전략 적용)
# =================================================================================

def train():
    # --- 하이퍼파라미터 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ROOT_PATH = r"/home/jovyan/workspace/chaeyoung/traincrop"  
    SAVE_PATH_DIR = r"/home/jovyan/workspace/chaeyoung/models" 
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4
    NUM_EPOCHS = 200 # 충분한 학습을 위해 에포크 증가
    NUM_WORKERS = 0 # 공유 메모리 오류 방지 = 0
    IMAGE_HEIGHT = 384
    IMAGE_WIDTH = 640

    os.makedirs(SAVE_PATH_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")
    print(f"Models will be saved to: {SAVE_PATH_DIR}")

    # --- 데이터 증강 (강화 버전) ---
    transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.MotionBlur(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet 통계치 사용
        ToTensorV2(),
    ])

    dataset = CrackDataset(root_dir=ROOT_PATH, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    # --- 모델, 손실함수, 옵티마이저, 스케줄러 ---
    model = HCUnetPlusPlus(num_classes=1, input_channels=3).to(DEVICE)
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # --- 학습 루프 ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        epoch_loss = 0
        
        for images, masks in loop:
            images = images.to(device=DEVICE)
            masks = masks.to(device=DEVICE)
            
            preds = model(images)
            
            # 손실 함수 조합 (BCE Loss + Dice Loss)
            loss = 0.5 * bce_loss(preds, masks) + 0.5 * dice_loss(preds, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        # 에포크마다 스케줄러 업데이트
        scheduler.step()

        avg_loss = epoch_loss / len(loader)
        print(f"----> Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # 10 에포크마다, 그리고 마지막 에포크에 모델 저장
        if (epoch + 1) % 10 == 0 or (epoch + 1) == NUM_EPOCHS:
            save_path = os.path.join(SAVE_PATH_DIR, f"hc_unetpp_edited_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model checkpoint saved to {save_path}")

if __name__ == '__main__':
    # Windows 환경에서 multiprocessing 사용 시 필요
    import multiprocessing
    multiprocessing.freeze_support()
    train()