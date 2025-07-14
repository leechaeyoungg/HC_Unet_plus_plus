# predict_with_sliding_window.py

import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from albumentations import Resize, Normalize, Compose
from albumentations.pytorch import ToTensorV2

# 학습 스크립트에서 모델 아키텍처를 가져옵니다.
from HC_Unet_pp_deep_supervision_train import HCUnetPlusPlus

# predict_with_sliding_window.py
def get_gaussian_kernel(window_size, sigma_scale=1./8):
    # ... (내용 동일) ...
    tmp = np.zeros(window_size)
    center_y, center_x = (window_size[0] - 1) / 2., (window_size[1] - 1) / 2.
    sigma_y, sigma_x = window_size[0] * sigma_scale, window_size[1] * sigma_scale
    y, x = np.mgrid[0 - center_y:window_size[0] - center_y, 0 - center_x:window_size[1] - center_x]
    kernel = np.exp(-(y**2 / (2 * sigma_y**2) + x**2 / (2 * sigma_x**2)))
    kernel[kernel < np.finfo(float).eps * kernel.max()] = 0
    return kernel

def load_inference_model(model_path, device):
    # ... (내용 동일) ...
    model = HCUnetPlusPlus(in_channels=3, num_classes=1, deep_supervision=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# --- [수정된 부분] 슬라이딩 윈도우 예측 함수 ---
def predict_with_sliding_window(image, model, window_size, stride, device):
    """슬라이딩 윈도우 기법으로 예측. 입력 이미지가 윈도우보다 작으면 일반 예측 수행"""
    h, w, _ = image.shape
    
    # --- [추가] 입력 이미지 크기 체크 ---
    # 만약 이미지의 높이나 너비가 윈도우 크기보다 작거나 같으면, 슬라이딩 윈도우 없이 바로 예측
    if h <= window_size[0] or w <= window_size[1]:
        print("Image is smaller than window size, performing a single prediction.")
        transform = Compose([
            Resize(height=window_size[0], width=window_size[1]),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])
        augmented = transform(image=image)
        input_tensor = augmented['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(input_tensor)
            final_pred = preds[-1]
            prob = torch.sigmoid(final_pred).squeeze().cpu().numpy()
        
        # 원래 이미지 크기로 되돌리기
        resized_prob = cv2.resize(prob, (w, h))
        binary_mask = (resized_prob > 0.5).astype(np.uint8)
        return binary_mask
        
    # --- 기존 슬라이딩 윈도우 로직 (이미지가 윈도우보다 클 경우) ---
    out_mask = np.zeros((h, w), dtype=np.float32)
    count_mask = np.zeros((h, w), dtype=np.float32)

    transform = Compose([
        Resize(height=window_size[0], width=window_size[1]),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    
    gaussian_kernel = get_gaussian_kernel(window_size)

    # 경계 처리를 포함한 좌표 리스트 생성
    y_steps = list(range(0, h - window_size[0], stride[0])) + [h - window_size[0]]
    x_steps = list(range(0, w - window_size[1], stride[1])) + [w - window_size[1]]

    for y in tqdm(y_steps, desc="Sliding Window"):
        for x in x_steps:
            crop = image[y:y + window_size[0], x:x + window_size[1]]
            
            augmented = transform(image=crop)
            input_tensor = augmented['image'].unsqueeze(0).to(device)

            with torch.no_grad():
                preds = model(input_tensor)
                final_pred = preds[-1] 
                prob = torch.sigmoid(final_pred).squeeze().cpu().numpy()

            out_mask[y:y + window_size[0], x:x + window_size[1]] += prob * gaussian_kernel
            count_mask[y:y + window_size[0], x:x + window_size[1]] += gaussian_kernel

    final_mask = out_mask / np.maximum(count_mask, 1e-6)
    binary_mask = (final_mask > 0.5).astype(np.uint8)
    return binary_mask

# ... (overlay_mask 및 메인 실행부는 동일) ...
def overlay_mask(image, binary_mask, color=(255, 0, 0), alpha=0.4):
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[binary_mask == 1] = color
    overlayed_image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    return overlayed_image

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = r"C:\Users\dromii\Downloads\hc_unetpp_ds_epoch_50.pth"
    IMAGE_PATH = r"C:\Users\dromii\Downloads\image4.png"
    SAVE_PATH = r"C:\Users\dromii\Downloads\prediction_result2.png"

    WINDOW_SIZE = (384, 640)
    STRIDE = (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2)

    print(f"Using device: {DEVICE}")
    print(f"Loading model: {MODEL_PATH}")
    model = load_inference_model(MODEL_PATH, DEVICE)
    
    print(f"Loading image: {IMAGE_PATH}")
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("Error: Image not found.")
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predicted_mask = predict_with_sliding_window(image_rgb, model, WINDOW_SIZE, STRIDE, DEVICE)
        
        print("Overlaying mask on the image...")
        result_image = overlay_mask(image_rgb, predicted_mask)
        
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        cv2.imwrite(SAVE_PATH, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        print(f"Prediction result saved to: {SAVE_PATH}")