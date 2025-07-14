# evaluate_accuracy.py

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 학습 스크립트에서 모델 아키텍처와 데이터셋 클래스를 가져옵니다.
# 파일 이름이 'HC_Unet_pp_deep_supervison_train.py'가 맞는지 확인하세요.
from HC_Unet_pp_deep_supervision_train import HCUnetPlusPlus, CrackDataset

# --- 성능 지표 계산 함수 ---
def iou_score(output, target):
    """IoU (Intersection over Union) 점수 계산"""
    with torch.no_grad():
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
        target = (target > 0.5).float()
        intersection = (output * target).sum()
        union = output.sum() + target.sum() - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

def dice_score(output, target):
    """Dice 점수 계산"""
    with torch.no_grad():
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
        target = (target > 0.5).float()
        intersection = (output * target).sum()
        dice = (2. * intersection + 1e-6) / (output.sum() + target.sum() + 1e-6)
    return dice.item()

def evaluate(model, loader, device):
    """테스트 데이터셋으로 모델의 성능을 평가하고 출력"""
    model.eval()
    total_iou = 0
    total_dice = 0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            preds = model(images)
            
            # Deep Supervision 모델의 경우, 가장 마지막 출력으로 평가
            if isinstance(preds, list):
                preds = preds[-1]
                
            total_iou += iou_score(preds, masks)
            total_dice += dice_score(preds, masks)
            
    avg_iou = total_iou / len(loader)
    avg_dice = total_dice / len(loader)
    
    print("\n" + "="*30)
    print("      Evaluation Results")
    print("="*30)
    print(f"  - Average IoU (mIoU)  : {avg_iou:.4f}")
    print(f"  - Average Dice Score  : {avg_dice:.4f}")
    print("="*30)

# --- 메인 실행부 ---
if __name__ == '__main__':
    # --- 설정 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = r"C:\Users\dromii\Downloads\hc_unetpp_ds_epoch_50.pth"  # <<< 평가할 모델 경로
    TEST_DATA_PATH = r"D:\datasets\UAV_crack_pothole_datasets\pavement crack datasets-20210103T153625Z-001\pavement crack datasets\CRACK500\traincrop\traincrop"                       # <<< 테스트 데이터셋 경로
    BATCH_SIZE = 4
    IMAGE_HEIGHT = 384
    IMAGE_WIDTH = 640

    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Loading test data from: {TEST_DATA_PATH}")

    # --- 데이터 로더 설정 ---
    eval_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), # 학습 시와 동일한 값 사용
        ToTensorV2(),
    ])
    test_dataset = CrackDataset(root_dir=TEST_DATA_PATH, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- 모델 로드 ---
    # 학습 시 설정과 동일하게 deep_supervision=True로 모델 로드
    model = HCUnetPlusPlus(in_channels=3, num_classes=1, deep_supervision=True).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # --- 평가 실행 ---
    evaluate(model, test_loader, DEVICE)