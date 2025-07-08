import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, jaccard_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# 동일한 Dataset 클래스
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
        mask = (mask > 127).astype('float32')

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)

        return image, mask

# HC-Unet++ 모델 클래스 (이미 정의된 클래스 그대로 import)

from hc_unet_pp_v2_train import HCUnetPlusPlus  # 기존 모델 정의 파일에서 import

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 평가용 데이터셋 경로
    root_path = "/mnt/storage/chaeyoung/crack_seg_datasets"

    # transform (학습 때와 동일해야 함)
    transform = A.Compose([
        A.Resize(384, 640),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    dataset = CrackDataset(root_dir=root_path, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 모델 불러오기
    model = HCUnetPlusPlus().to(device)
    model.load_state_dict(torch.load("/mnt/storage/chaeyoung/hc_unetpp_crack.pth", map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(np.uint8)
            targets = masks.numpy().astype(np.uint8)

            all_preds.append(preds.flatten())
            all_targets.append(targets.flatten ())

    # concat all
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    # 메트릭 계산
    iou = jaccard_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print(f"IoU: {iou:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

if __name__ == '__main__':
    evaluate()
