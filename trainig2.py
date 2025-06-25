import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import os
import numpy as np

# Mask-RCNN 모델 준비 (weights 파라미터 사용)
weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1  # COCO 데이터셋에서 학습된 가중치 사용
model = maskrcnn_resnet50_fpn(weights=weights)
model.train()

# XML 파일에서 바운딩 박스 정보 추출
def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []
    for obj in root.iter('object'):
        name = obj.find('name').text
        if name != 'pothole':  # 포트홀 클래스만 사용
            continue
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # 높이와 너비가 0이 아닌지 확인
        if xmax > xmin and ymax > ymin:
            boxes.append([xmin, ymin, xmax, ymax])
        else:
            print(f"Invalid box found in {xml_file}: [{xmin}, {ymin}, {xmax}, {ymax}]")
    
    return boxes

# 바운딩 박스를 바탕으로 마스크 생성
def create_mask(boxes, img_size):
    masks = []
    for box in boxes:
        # 바운딩 박스를 튜플 형식으로 변환 (xmin, ymin, xmax, ymax)
        box = tuple(box)
        # 이미지 크기와 동일한 빈 마스크 생성
        mask = Image.new("L", img_size, 0)
        draw = ImageDraw.Draw(mask)
        # 바운딩 박스를 채워서 마스크 생성 (box 좌표를 1로 채움)
        draw.rectangle(box, outline=1, fill=1)
        
        # PIL 이미지에서 NumPy 배열로 변환 후 텐서로 변환
        mask = torch.as_tensor(np.array(mask), dtype=torch.uint8)
        masks.append(mask)
    
    if len(masks) > 0:
        return torch.stack(masks)  # (num_objs, height, width) 형식으로 반환
    else:
        return torch.zeros((0, img_size[1], img_size[0]), dtype=torch.uint8)  # 객체가 없을 경우 빈 마스크
    

# 데이터셋 클래스 정의
class PotholeDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, annot_dir, transforms=None):
        self.img_dir = img_dir
        self.annot_dir = annot_dir
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(img_dir)))  # 이미지 파일 목록
        self.annots = list(sorted(os.listdir(annot_dir)))  # XML 파일 목록
        print("datasetclass init")

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        annot_path = os.path.join(self.annot_dir, self.annots[idx])
        img = Image.open(img_path).convert("RGB")

        boxes = parse_voc_annotation(annot_path)  # 바운딩 박스 추출
        num_objs = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # 포트홀 클래스 라벨 1
        
        # 마스크 생성 (이미지 크기에 맞는 마스크)
        masks = create_mask(boxes, img.size)

        # target 딕셔너리 생성
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks

        if self.transforms:
            img = self.transforms(img)
        
        return img, target

    def __len__(self):
        return len(self.imgs)

# 커스텀 collate_fn 정의
def collate_fn(batch):
    return tuple(zip(*batch))

# 메인 블록으로 코드 감싸기
if __name__ == "__main__":
    # DataLoader 생성
    print('Create dataloader...')
    dataset = PotholeDataset(img_dir='images', annot_dir='annotations', transforms=F.to_tensor)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    print('Complete 1/3')

    # 모델 학습 설정
    print('Set model')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    print('Complete 2/3')

    # Optimizer와 학습 루프 설정

    print('Set loop')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    print('Complete 3/3 Let\'s training')

    
    num_epochs = 10
    for epoch in range(num_epochs):
        num = 0
        for images, targets in data_loader:
            print("images to device...", end=" ")
            images = list(image.to(device) for image in images)
            print("done")

            print("targets to device...", end=" ")
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            print("done")

            print("forward pass...", end=" ")
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            print("done")

            print("backward pass...", end=" ")
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            print(f"Epoch: {epoch}, num: {num} Loss: {losses.item()},")
            num+=1

    print("Training complete!")

    torch.save(model.state_dict(), "pothole_detection_model.pth")
    print("Model weights saved.")
