import torch
import cv2
import numpy as np
from albumentations import Resize, Normalize, Compose
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from hc_unet_pp_v2_train import HCUnetPlusPlus  # 모델 정의 파일 import

# ----------------------------
# 모델 로드 함수
# ----------------------------
def load_model(model_path, device):
    model = HCUnetPlusPlus()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    return model

# ----------------------------
# 슬라이딩 윈도우 예측 함수
# ----------------------------
def sliding_window_predict(image, model, window_size=(384, 640), stride=(256, 448), device='cuda'): #window_size=(384, 640), stride=(256, 448), device='cuda'):
    h, w, _ = image.shape
    out_mask = np.zeros((h, w), dtype=np.float32)
    count_mask = np.zeros((h, w), dtype=np.float32)

    transform = Compose([
        Resize(*window_size),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    for y in tqdm(range(0, h - window_size[0] + 1, stride[0])):
        for x in range(0, w - window_size[1] + 1, stride[1]):
            crop = image[y:y + window_size[0], x:x + window_size[1]]
            augmented = transform(image=crop)
            input_tensor = augmented['image'].unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(input_tensor)
                prob = torch.sigmoid(pred).cpu().numpy()[0, 0]

            out_mask[y:y + window_size[0], x:x + window_size[1]] += prob
            count_mask[y:y + window_size[0], x:x + window_size[1]] += 1

    final_mask = out_mask / np.maximum(count_mask, 1e-5)
    binary_mask = (final_mask > 0.5).astype(np.uint8)
    return binary_mask

# ----------------------------
# 시각화 함수
# ----------------------------
def overlay_mask(image, binary_mask, save_path):
    h, w = image.shape[:2]
    resized_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    red_mask = np.zeros_like(image)
    red_mask[:, :, 0] = 255  # Red only

    overlay = image.copy()
    mask_bool = resized_mask.astype(bool)
    overlay[mask_bool] = red_mask[mask_bool]

    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"결과 저장 완료: {save_path}")

# ----------------------------
# 실행부
# ----------------------------
if __name__ == "__main__":
    image_path = r"D:\datasets\UAV_crack_pothole_datasets\pavement crack datasets-20210103T153625Z-001\pavement crack datasets\CRACK500\traincrop\traincrop\20160222_081011_1_361.jpg"
    model_path = r"C:\Users\dromii\Downloads\hc_unetpp_crack_v2.pth"
    save_path = r"C:\Users\dromii\Downloads\X_00108_202406201329_hcunetpp_v2_detect.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 원본 이미지 로드
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 모델 로드 및 예측
    model = load_model(model_path, device)
    mask = sliding_window_predict(image, model, device=device)

    # 시각화 저장
    overlay_mask(image, mask, save_path)
