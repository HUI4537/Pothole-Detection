import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random

# Mask-RCNN 모델 불러오기 (COCO 기반으로 학습된 가중치)
from torchvision.models.detection import maskrcnn_resnet50_fpn

# 저장된 가중치를 불러와서 모델 로드
model = maskrcnn_resnet50_fpn(weights=None)  # 가중치 없음으로 초기화
model.load_state_dict(torch.load("pothole_detection_model.pth"))
model.eval()  # 평가 모드로 전환

# 색상 생성 함수
def random_color():
    return [random.randint(0, 255) for _ in range(3)]  # RGB 컬러 생성

# 이미지 마스킹 및 저장 함수
def mask_image(img_path, output_path, threshold=0.5):
    # 이미지 불러오기 및 변환
    img = Image.open(img_path).convert("RGB")
    img_tensor = F.to_tensor(img).unsqueeze(0)  # 배치를 추가해 (1, C, H, W) 형식으로 변환

    # 모델로 예측
    with torch.no_grad():
        predictions = model(img_tensor)

    # 포트홀 마스크 추출 (threshold 이상의 값만 마스크로 취급)
    masks = predictions[0]['masks']  # (N, 1, H, W) 형태
    masks = masks.squeeze(1)  # (N, H, W)로 차원 축소
    scores = predictions[0]['scores']  # 예측된 각 마스크의 신뢰도

    # 특정 threshold 이상인 마스크만 사용 (0.5는 보통 많이 사용됨)
    masks = masks[scores >= threshold]
    labels = predictions[0]['labels'][scores >= threshold]  # 레이블 추출

    # 이미지로 변환할 numpy 배열 생성
    img_np = np.array(img)

    # 마스크가 있으면 처리
    if masks.size(0) > 0:
        for i in range(masks.size(0)):
            mask = masks[i].numpy()  # 마스크를 NumPy 배열로 변환
            color = random_color()  # 랜덤 색상 선택
            img_np[mask > 0.5] = img_np[mask > 0.5] * 0.5 + np.array(color) * 0.5  # 색상 오버레이
    else:
        print("No pothole detected")

    # 결과 이미지를 저장
    result_img = Image.fromarray(img_np.astype(np.uint8))
    result_img.save(output_path)
    print(f"Image with masks saved to {output_path}")

# 메인 블록으로 감싸기
if __name__ == "__main__":
    input_image = "201754376_500.jpg"  # 예측할 이미지 경로 (업로드된 이미지 경로)
    output_mask = "pothole_detection_result.png"  # 저장할 마스크 파일 경로
    mask_image(input_image, output_mask)
