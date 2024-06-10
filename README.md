##  딥러닝 기술을 활용한 유방암 영상 분류
> (CNN-Breast_Cancer_Classification)
---

### 1. 프로젝트 개요
   
의료 영상 분석 기술을 활용하여 환자의 이미지 데이터를 분석, 신체 / 정신 건강에 대한 이상 증세를 듣고 데이터를 수집해 수집한 데이터를 기반으로 **질병을 분류하는 모델**을 제작한 것이 본 프로젝트의 핵심이다.

이 모델은 유방암 환자의 초음파 사진 데이터로 훈련시켰으며 **클래스는 세 가지로 분류**된다 


<img width="116" alt="image" src="https://github.com/ShinBangHo/CNN-Breast_Cancer_Classification/assets/164139725/cd686558-4f3e-4fe7-a731-7194a9273358"><img width="155" alt="image" src="https://github.com/ShinBangHo/CNN-Breast_Cancer_Classification/assets/164139725/5397253c-47ac-46b6-b5ce-bb719da994a8"><img width="113" alt="image" src="https://github.com/ShinBangHo/CNN-Breast_Cancer_Classification/assets/164139725/149eb458-e2ea-4d13-a165-cb268f7049f0">

> benign : 상대적으로 성장 속도가 느리고 전이를 하지 않는 양성 종양 사진

<img width="115" alt="image" src="https://github.com/ShinBangHo/CNN-Breast_Cancer_Classification/assets/164139725/560dd063-0396-4eab-bd5f-92ecf6ef090d"><img width="124" alt="image" src="https://github.com/ShinBangHo/CNN-Breast_Cancer_Classification/assets/164139725/93cc4cb4-9abc-4a01-ab11-9935b24f1495"><img width="115" alt="image" src="https://github.com/ShinBangHo/CNN-Breast_Cancer_Classification/assets/164139725/bdc5003f-4c35-4aa5-bf9d-e6b66ecae8d9">

> malignant : 성장이 빠르고 주위 조직과 다른 신체 부위로 퍼져나가는 악성 종양 사진

<img width="119" alt="image" src="https://github.com/ShinBangHo/CNN-Breast_Cancer_Classification/assets/164139725/a25cd535-7b9e-4372-8726-e631c65ca4cd"><img width="110" alt="image" src="https://github.com/ShinBangHo/CNN-Breast_Cancer_Classification/assets/164139725/f7f5bc43-56bc-433a-8061-c0594c181ac1"><img width="113" alt="image" src="https://github.com/ShinBangHo/CNN-Breast_Cancer_Classification/assets/164139725/33518e0f-f298-4337-a0b5-ac1facf7450f">

> normal : 아무 이상 없는 사진

데이터는 kaggle 사이트에서 수집했으며

위 데이터들을 각각 Train / test set로 분할하고 훈련 및 테스트를 진행했다.

---

### 2. 필요 라이브러리 및 프로그램

아래 사진은 사용된 라이브러리이다.

![image](https://github.com/ShinBangHo/CNN-Breast_Cancer_Classification/assets/164139725/ad3c30ef-98e7-482f-ba76-101bc102fbae)

PyTorch 관련 라이브러리 및 영상 처리, 유틸리티 등의 라이브러리를 사용했다.  

---

### 3. 모델 설명

모델 : **resnet**

![image](https://github.com/ShinBangHo/CNN-Breast_Cancer_Classification/assets/164139725/198ccac9-b20a-4e2b-827a-4fdab74851ad)![image](https://github.com/ShinBangHo/CNN-Breast_Cancer_Classification/assets/164139725/441ab539-8d5d-4f9c-ad3d-34b7e22f1557)
![image](https://github.com/ShinBangHo/CNN-Breast_Cancer_Classification/assets/164139725/2310f4f8-7994-48f1-be7d-6c1d816d3094)

기존 모델은 Simple2DCNN을 사용했지만 성능을 개선해보려고 augmentaion 함수, Learning Rate scheduler 등을 사용해봐도 개선이 크게 되지 않아 모델을 resnet으로 변경했다.

네트워크의 깊이와 복잡성, 기울기 소실 (Gradient Vanishing) 현상 완화 등만 비교해봐도 성능의 개선이 크게 될 것으로 예상했기에 resnet을 기용했다. 

.

