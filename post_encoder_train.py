import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# =================================================================================
# 1. 모듈 정의
# =================================================================================
class DPFFB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upper = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.lower = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return torch.cat([self.upper(x), self.lower(x)], dim=1)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        weight = self.fc(self.pool(x))
        return x * weight

class BlurPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
        self.register_buffer('filter', kernel[None, None, :, :].repeat(channels, 1, 1, 1))
        self.channels = channels
    def forward(self, x):
        # 블러링 효과만 주도록 stride=1로 고정
        return F.conv2d(x, self.filter, stride=1, padding=1, groups=self.channels)

# =================================================================================
# 2. 데이터셋 및 손실 함수 정의
# =================================================================================
class CrackDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images = sorted([f for f in os.listdir(root_dir) if f.endswith('.jpg')])
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        mask_path = os.path.splitext(img_path)[0] + '.png'
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            h, w, _ = image.shape
            mask = np.zeros((h, w), dtype=np.uint8)
        mask = (mask > 127).astype('float32')
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)
        return image, mask

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs, targets = inputs.view(-1), targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

# =================================================================================
# 3. 모델 래퍼 정의 (오류 수정 최종 버전)
# =================================================================================
class HCUnetPlusPlusWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        # smp 모델의 주요 구성요소를 분리하여 저장
        self.base_encoder = base_model.encoder
        self.base_decoder = base_model.decoder
        self.segmentation_head = base_model.segmentation_head
        
        encoder_channels = self.base_encoder.out_channels[-1]
        
        # HC-Unet++ 핵심 모듈
        self.blur = BlurPool(encoder_channels)
        self.dpffb = DPFFB(encoder_channels)
        self.se = SEBlock(encoder_channels * 2)
        # 채널 수를 다시 원래대로 맞춰주는 브릿지 레이어
        self.bridge = nn.Conv2d(encoder_channels * 2, encoder_channels, kernel_size=1)

    def forward(self, x):
        features = self.base_encoder(x)
        f = features[-1]
        
        # 병목 구간에서 HC 모듈 순차 적용
        f = self.blur(f)
        f = self.dpffb(f)
        f = self.se(f)
        f = self.bridge(f) # 채널 수 복원
        
        features[-1] = f # 정제된 특징으로 교체
        
        decoder_output = self.base_decoder(features)
        
        # segmentation_head를 통과시켜 최종 출력 생성 (채널=1)
        masks = self.segmentation_head(decoder_output)
        
        return masks

# =================================================================================
# 4. 훈련 함수 (최적화 전략 적용)
# =================================================================================
def train():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # --- 하이퍼파라미터 ---
    DATASET_DIR = "/mnt/storage/chaeyoung/crack_seg_datasets"
    SAVE_PATH = "/mnt/storage/chaeyoung/saved_models/smp_hybrid_hc_unetpp_final.pth"
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4
    NUM_EPOCHS = 50
    NUM_WORKERS = 0  # 공유 메모리 오류 방지

    # --- 데이터 증강 및 로더 ---
    transform = A.Compose([
        A.Resize(384, 640),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    dataset = CrackDataset(root_dir=DATASET_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # --- 모델, 손실함수, 옵티마이저, 스케줄러 ---
    base_model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    model = HCUnetPlusPlusWrapper(base_model).to(DEVICE)
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # --- 학습 루프 ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for images, masks in loop:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            outputs = model(images)
            # BCE+Dice 손실 조합 사용
            loss = 0.5 * bce_loss(outputs, masks) + 0.5 * dice_loss(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(loader)
        print(f"----> [Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Training finished and model saved to {SAVE_PATH}")

if __name__ == "__main__":
    train()