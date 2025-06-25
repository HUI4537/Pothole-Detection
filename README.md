
## 📂 Folder Structure
- `images/` : Input images (e.g., `1.jpg`, `2.jpg`, ...)
- `annotations/` : Corresponding labels for each image (e.g., `1.txt`, `2.txt`, ...)
- `training/` : Used for training the model
- `eval/` : Used to test the trained model with new images
- Other folders/files are experimental and can be ignored.

**📝 Note:** Each image in the `images/` folder has a matching annotation file in the `annotations/` folder. For example, `1.jpg` corresponds to `1.txt`.

---

## 🚀 How to Use

1. **Train the Model**
   - Run the training script inside the `training/` directory.
   - After training completes, a file named like `p.._d..._model.pth` (the trained weights) will be generated.

2. **Evaluate the Model**
   - Go to the `eval/` folder.
   - Provide the image you want to test as input.
   - The script will load the trained weights and display the results.

3. 🎉 **See the Results**
   - The output with the model's predictions will be shown after evaluation.

---


## 📂 폴더 구조
- `images/` : 입력 이미지들 (`1.jpg`, `2.jpg`, ...)
- `annotations/` : 각 이미지에 해당하는 라벨 파일 (`1.txt`, `2.txt`, ...)
- `training/` : 모델 학습용 디렉토리
- `eval/` : 학습된 모델을 테스트하는 디렉토리
- 그 외 폴더/파일들은 테스트 중 생성된 것이므로 무시해도 됩니다.

**📝 참고:** `images/` 폴더의 각 이미지와 `annotations/` 폴더의 라벨 파일은 번호 순서대로 1:1로 매칭됩니다. 예: `1.jpg` ↔ `1.txt`

---

## 🚀 사용 방법

1. **모델 학습**
   - `training/` 디렉토리에서 학습 스크립트를 실행합니다.
   - 학습이 완료되면 `p.._d..._model.pth` 같은 이름의 가중치 파일이 생성됩니다.

2. **모델 평가**
   - `eval/` 디렉토리에서 평가 스크립트를 실행합니다.
   - 원하는 이미지를 입력으로 넣으면, 저장된 가중치를 이용해 결과를 예측합니다.

3. 🎉 **결과 확인**
   - 평가가 끝나면 이미지에 대한 예측 결과가 출력됩니다.
