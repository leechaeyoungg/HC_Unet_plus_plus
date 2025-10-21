# eval_crack_iou.py

#python eval_crack_iou.py --root "C:\Users\dromii\Downloads\crack_masks" --weights "C:\Users\dromii\Downloads\500_hc_unetpp_final_epoch_100.pth" --batch 4 --thr 0.5 --size 512


import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

# =========================
# 1) 모델 정의 (훈련 코드와 동일)
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class DPFFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_channels = out_channels // 2
        self.upper = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.lower = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        up = self.upper(x)
        low = self.lower(x)
        return torch.cat([up, low], dim=1)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weight = self.fc(self.pool(x))
        return x * weight

class BlurPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel = torch.tensor([[1., 2., 1.],
                               [2., 4., 2.],
                               [1., 2., 1.]])
        kernel = kernel / kernel.sum()
        self.register_buffer('filter', kernel[None, None, :, :].repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.filter, stride=2, padding=1, groups=self.channels)

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class HCUnetPlusPlus(nn.Module):
    def __init__(self, num_classes=1, input_channels=3):
        super().__init__()
        filters = [64, 128, 256, 512, 1024]

        # Down
        self.pool0 = BlurPool(filters[0])
        self.pool1 = BlurPool(filters[1])
        self.pool2 = BlurPool(filters[2])
        self.pool3 = BlurPool(filters[3])

        # Up
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

        # Nested
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

        self.se3_1 = SEBlock(filters[3])
        self.output = nn.Conv2d(filters[0], num_classes, 1)

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

        return self.output(x0_4)

# =========================
# 2) 데이터셋 & 전처리
# =========================
IMG_EXTS = ('.jpg', '.jpeg', '.JPG', '.JPEG')
MSK_EXT = '.png'

class CrackEvalDataset(Dataset):
    """
    root_dir 안에 동일 basename의 이미지(.JPG)와 마스크(.png)가 존재
    """
    def __init__(self, root_dir, size=512):
        self.root = root_dir
        # 마스크 목록 기준으로 페어링(대소문자 이미지 확장자 모두 허용)
        mask_files = [f for f in os.listdir(root_dir) if f.lower().endswith('.png')]
        pairs = []
        for m in mask_files:
            base = os.path.splitext(m)[0]
            img_path = None
            for ext in IMG_EXTS:
                cand = os.path.join(root_dir, base + ext)
                if os.path.exists(cand):
                    img_path = cand
                    break
            if img_path is not None:
                pairs.append((img_path, os.path.join(root_dir, m)))
        self.pairs = sorted(pairs, key=lambda x: os.path.basename(x[0]))

        self.transform = A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        self.size = size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # 이진화(안전)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = (mask / 255.0).astype('float32')  # 0/1

        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']                      # C,H,W float tensor
        mask = augmented['mask'].unsqueeze(0)           # 1,H,W
        return image, mask, os.path.basename(img_path)

# =========================
# 3) 지표 함수
# =========================
def binarize_logits(logits, thr=0.5):
    probs = torch.sigmoid(logits)
    return (probs >= thr).float()

@torch.no_grad()
def evaluate(model, loader, device, thr=0.5, save_csv='crack_eval_iou.csv'):
    model.eval()
    eps = 1e-7

    # 글로벌 누적(픽셀 총합 기반)
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    # per-image 로깅
    rows = []
    num_pos_gt_imgs = 0
    iou_pos_sum_posgt = 0.0  # GT에 크랙 있는 이미지에서의 IoU 평균용

    for images, masks, names in tqdm(loader, desc="Evaluating", ncols=100):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        preds = binarize_logits(logits, thr=thr)

        # 배치 단위 반복
        for i in range(images.size(0)):
            gt = masks[i, 0] > 0.5
            pr = preds[i, 0] > 0.5

            tp = torch.logical_and(pr, gt).sum().item()
            fp = torch.logical_and(pr, torch.logical_not(gt)).sum().item()
            fn = torch.logical_and(torch.logical_not(pr), gt).sum().item()
            tn = torch.logical_and(torch.logical_not(pr), torch.logical_not(gt)).sum().item()

            TP += tp; FP += fp; FN += fn; TN += tn

            iou_pos = (tp + eps) / (tp + fp + fn + eps)
            dice = (2*tp + eps) / (2*tp + fp + fn + eps)

            gt_pos_pixels = gt.sum().item()
            pr_pos_pixels = pr.sum().item()

            if gt_pos_pixels > 0:
                num_pos_gt_imgs += 1
                iou_pos_sum_posgt += iou_pos

            rows.append([
                names[i],
                float(iou_pos),
                float(dice),
                int(gt_pos_pixels),
                int(pr_pos_pixels)
            ])

    # 글로벌 IoU(양성/배경) 및 mIoU
    iou_crack = (TP + eps) / (TP + FP + FN + eps)                 # 양성(크랙) IoU
    iou_bg    = (TN + eps) / (TN + FP + FN + eps)                 # 배경 IoU
    mIoU      = 0.5 * (iou_crack + iou_bg)                        # 두 클래스 평균

    iou_pos_only_gt = (iou_pos_sum_posgt / max(1, num_pos_gt_imgs)) if num_pos_gt_imgs > 0 else 0.0

    # CSV 저장
    try:
        import csv
        with open(save_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['filename', 'iou_crack', 'dice', 'gt_crack_pixels', 'pred_crack_pixels'])
            w.writerows(rows)
        print(f"[INFO] Per-image metrics saved to: {os.path.abspath(save_csv)}")
    except Exception as e:
        print(f"[WARN] CSV save failed: {e}")

    # 요약 출력
    total_imgs = len(rows)
    print("\n========== Evaluation Summary ==========")
    print(f"Images evaluated          : {total_imgs}")
    #print(f"Global Crack IoU          : {iou_crack*100:.2f}%")
    #print(f"Global Background IoU     : {iou_bg*100:.2f}%")
    print(f"mIoU (Crack/BG mean)      : {mIoU*100:.2f}%")
    #print(f"IoU on GT-positive images : {iou_pos_only_gt*100:.2f}%  "
    #      f"(N={num_pos_gt_imgs} with crack)")
    print("========================================")

    return {
        "iou_crack": float(iou_crack),
        "iou_bg": float(iou_bg),
        "mIoU": float(mIoU),
        "iou_pos_only_gt": float(iou_pos_only_gt),
        "N_images": total_imgs,
        "N_gt_pos_images": int(num_pos_gt_imgs)
    }

# =========================
# 4) 실행부
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='폴더 경로 (JPG + PNG가 짝으로 있는 폴더)')
    parser.add_argument('--weights', type=str, required=True,
                        help='학습된 모델 .pth 경로')
    parser.add_argument('--size', type=int, default=512,
                        help='평가 입력 이미지 크기(정방향 리사이즈)')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--thr', type=float, default=0.5,
                        help='시그모이드 임계값')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--csv', type=str, default='crack_eval_iou.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    # Dataset / Loader
    ds = CrackEvalDataset(args.root, size=args.size)
    if len(ds) == 0:
        raise RuntimeError("평가할 페어(JPG+PNG)를 찾지 못했습니다. 경로/확장자를 확인하세요.")
    print(f"[INFO] Paired samples: {len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type=='cuda'))

    # Model
    model = HCUnetPlusPlus(num_classes=1, input_channels=3).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    print(f"[INFO] Weights loaded from: {args.weights}")

    # Eval
    evaluate(model, loader, device, thr=args.thr, save_csv=args.csv)

if __name__ == "__main__":
    main()
