import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import xml.etree.ElementTree as ET
from PIL import Image
import os

# Mask-RCNN 모델 준비
model = maskrcnn_resnet50_fpn(pretrained=True)
model.train()

# XML 파일에서 바운딩 박스 정보 추출
def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    boxes = []
    for obj in root.iter('object'):
        name = obj.find('name').text
        if name != 'pothole':  # 포트홀 클래스
            continue
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
    
    return boxes

# 데이터셋 클래스 정의
class PotholeDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, annot_dir, transforms=None):
        self.img_dir = img_dir
        self.annot_dir = annot_dir
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(img_dir)))  # 이미지 파일 목록
        self.annots = list(sorted(os.listdir(annot_dir)))  # XML 파일 목록

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        annot_path = os.path.join(self.annot_dir, self.annots[idx])
        img = Image.open(img_path).convert("RGB")

        boxes = parse_voc_annotation(annot_path)  # 바운딩 박스 추출
        num_objs = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # 포트홀 클래스 라벨 1
        masks = torch.zeros((num_objs,))  # 마스크(추후 마스크를 적용할 경우)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks  # 마스크 추가 가능

        if self.transforms:
            img = self.transforms(img)
        
        return img, target

    def __len__(self):
        return len(self.imgs)

# DataLoader 생성
dataset = PotholeDataset(img_dir='images', annot_dir='annotations', transforms=F.to_tensor)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

# 모델 학습 설정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Optimizer와 학습 루프 설정
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

num_epochs = 10
for epoch in range(num_epochs):
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)  # 모델에 입력 및 학습
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    print(f'Epoch: {epoch}, Loss: {losses.item()}')

print("Training complete!")
