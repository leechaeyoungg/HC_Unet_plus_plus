import os
import cv2
from tqdm import tqdm

def check_dataset_integrity(dataset_path):
    """
    지정된 경로의 이미지와 마스크 파일 쌍의 무결성을 검사합니다.
    1. 모든 .jpg 파일에 대해 짝이 되는 .png 파일이 있는지 확인합니다.
    2. cv2.imread를 통해 파일이 정상적으로 읽히는지 확인합니다.
    """
    print("="*50)
    print(f"데이터셋 무결성 검사를 시작합니다.")
    print(f"대상 경로: {dataset_path}")
    print("="*50)

    if not os.path.isdir(dataset_path):
        print(f"오류: '{dataset_path}'는 유효한 디렉토리가 아닙니다.")
        return

    image_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.jpg')])
    
    if not image_files:
        print("경로에서 '.jpg' 이미지 파일을 찾을 수 없습니다.")
        return

    total_files = len(image_files)
    missing_masks = []
    unreadable_images = []
    unreadable_masks = []

    for image_name in tqdm(image_files, desc="파일 검사 중"):
        image_path = os.path.join(dataset_path, image_name)
        
        # .jpg를 .png로 바꾸는 가장 안정적인 방법 사용
        mask_name = os.path.splitext(image_name)[0] + '.png'
        mask_path = os.path.join(dataset_path, mask_name)

        # 1. 이미지 파일 읽기 검사
        img = cv2.imread(image_path)
        if img is None:
            unreadable_images.append(image_name)
            continue # 이미지 로드 실패 시 마스크 검사는 의미 없으므로 건너뜀

        # 2. 마스크 파일 존재 여부 검사
        if not os.path.exists(mask_path):
            missing_masks.append(image_name)
            continue # 마스크 파일이 없으면 읽기 검사는 의미 없으므로 건너뜀
        
        # 3. 마스크 파일 읽기 검사
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            unreadable_masks.append(mask_name)

    # --- 최종 결과 출력 ---
    print("\n" + "="*50)
    print("검사 결과 요약")
    print("="*50)
    print(f"총 이미지 파일 수 (.jpg): {total_files}")
    
    # 누락된 마스크 파일 출력
    if missing_masks:
        print(f"\n[오류] 짝이 되는 마스크(.png) 파일이 없는 이미지 ({len(missing_masks)}개):")
        for name in missing_masks:
            print(f"  - {name}")
    else:
        print("\n[성공] 모든 이미지에 짝이 되는 마스크 파일이 존재합니다.")
        
    # 읽을 수 없는 이미지 파일 출력
    if unreadable_images:
        print(f"\n[오류] 손상되었거나 읽을 수 없는 이미지(.jpg) 파일 ({len(unreadable_images)}개):")
        for name in unreadable_images:
            print(f"  - {name}")
    else:
        print("\n[성공] 모든 이미지 파일을 정상적으로 읽었습니다.")

    # 읽을 수 없는 마스크 파일 출력
    if unreadable_masks:
        print(f"\n[오류] 손상되었거나 읽을 수 없는 마스크(.png) 파일 ({len(unreadable_masks)}개):")
        for name in unreadable_masks:
            print(f"  - {name}")
    else:
        print("\n[성공] 모든 마스크 파일을 정상적으로 읽었습니다.")
        
    print("\n검사가 완료되었습니다.")
    print("="*50)


if __name__ == '__main__':
    # <<< 검사할 데이터셋의 경로를 여기에 입력하세요 >>>
    TARGET_DATASET_PATH = r"D:\datasets\UAV_crack_pothole_datasets\pavement crack datasets-20210103T153625Z-001\pavement crack datasets\CRACK500\traincrop\traincrop"
    
    check_dataset_integrity(TARGET_DATASET_PATH)