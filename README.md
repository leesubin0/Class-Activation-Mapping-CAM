Paper:
* [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150)

Paper Review:

* [[논문 리뷰] CAM - Learning Deep Features for Discriminative Localization (2016)](https://velog.io/@keepbini366/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-CAM-Learning-Deep-Features-for-Discriminative-Localization-2016-twm3odzv)

GitHub:

* [Class-Activation-Mapping-CAM](https://github.com/leesubin0/Class-Activation-Mapping-CAM)

---

## CAM이란?

>**CAM**
: **Class Activation Mapping**
CNN이 특정 클래스를 예측할 때 이미지의 어느 영역을 중요하게 활용했는지 Heatmap 형태로 시각화하는 방법.

ex. 이미지 분류 모델이 입력 이미지를 `dog`라고 예측했다면,
CAM을 이용해 모델이 강아지의 얼굴, 몸통, 눈과 같은 영역 중 어디를 근거로 판단했는지 확인할 수 있음.

CAM 논문에서는 이미지 수준의 Class Label만으로 학습한 CNN이
객체의 위치까지 어느 정도 추정할 수 있음을 보였음.

> **= 학습 과정에서 Bounding Box나 Segmentation Mask를 제공하지 않았음에도
모델이 클래스 판별에 중요한 영역을 Feature Map 내부에 학습한다는 것.**

---

### CAM 모델 구조

CAM은 모든 CNN 모델에 바로 적용할 수 있는 방법은 아님.

CAM을 계산하려면 모델의 마지막 부분이 다음과 같은 구조를 가져야 함.

```text
Input Image
      │
      ▼
Convolution Layers
      │
      ▼
Last Convolution Feature Maps
      │
      ▼
Global Average Pooling
      │
      ▼
Linear Classifier
      │
      ▼
Class Score
```

일반적인 CNN에서는 Convolution Layer 뒤에 Feature Map을 Flatten한 후
여러 개의 Fully Connected Layer를 연결하는 경우가 많음.

```text
Feature Maps
      │
      ▼
Flatten
      │
      ▼
Fully Connected Layer
      │
      ▼
Fully Connected Layer
      │
      ▼
Class Score
```

**하지만 이렇게 Feature Map을 모두 펼치면 공간적 위치 정보가 사라짐.**

> **반면 CAM 구조에서는 마지막 Convolution Feature Map에
Global Average Pooling을 적용한 뒤 Linear Classifier와 직접 연결함.**
>
> ---
>**★★★이 구조 덕분에 마지막 Feature Map의 각 Channel과
Linear Classifier의 Weight를 대응시킬 수 있는 것.** 

---

### 프로젝트 환경 구축

이번 실습은 별도의 Conda 환경에서 진행함.

```bash
conda create -n cam_practice python=3.10 -y
conda activate cam_practice
```

환경 구성 후 필요한 라이브러리와 GPU 사용 여부를 확인함.

### i) 구현 환경

| 항목          | 환경                      |
| ----------- | ----------------------- |
| OS          | Windows                 |
| Python      | 3.10.20                 |
| PyTorch     | 2.6.0+cu124             |
| Torchvision | 0.21.0+cu124            |
| OpenCV      | 5.0.0                   |
| Matplotlib  | 3.10.9                  |
| GPU         | NVIDIA GeForce RTX 2070 |

GPU 사용 가능 여부는 다음 코드로 확인함.

```python
import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

CUDA가 정상적으로 인식되어 이후 학습은 GPU 기반으로 진행.

---

### ii) 프로젝트 폴더 구성

CAM 생성뿐 아니라 학습, 평가, 시각화, Localization 실험까지 진행하기 위해
프로젝트 구조를 다음과 같이 구성함.

```text
cam_practice/
│
├── data/
│
├── experiments/
│
├── models/
│   └── cam_cnn.py
│
├── utils/
│   ├── cam_utils.py
│   ├── localization.py
│   └── metrics.py
│
├── results/
│   ├── checkpoints/
│   ├── cams/
│   ├── predictions/
│   ├── bounding_boxes/
│   └── graphs/
│
├── train.py
├── dataset.py
├── evaluate.py
├── visualize_cam.py
├── test_model.py
├── check_environment.py
└── README.md
```

각 폴더와 파일의 역할은 다음과 같음.

| 경로                        | 역할                                 |
| ------------------------- | ---------------------------------- |
| `models/`                 | CAM을 적용할 CNN 모델 구현                 |
| `utils/`                  | CAM 계산, Localization, Metric 관련 함수 |
| `data/`                   | CIFAR-10 데이터셋 저장                   |
| `results/checkpoints/`    | 학습된 모델 Weight 저장                   |
| `results/cams/`           | 생성한 CAM Heatmap 저장                 |
| `results/predictions/`    | 모델의 Classification 결과 저장           |
| `results/bounding_boxes/` | CAM 기반 Bounding Box 결과 저장          |
| `results/graphs/`         | Loss, Accuracy 등 학습 그래프 저장         |
| `experiments/`            | GAP와 GMP 비교 등 추가 실험                |
| `train.py`                | 모델 학습                              |
| `evaluate.py`             | 모델 성능 평가                           |
| `visualize_cam.py`        | CAM 생성 및 시각화                       |
| `test_model.py`           | 모델 Shape 및 CAM 구조 조건 검증            |

---

### CAM 모델 구현

이번 실습에서는 CIFAR-10을 기준으로 입력 이미지 크기를 `32×32`로 설정하고,
마지막 Convolution Layer가 128개의 Feature Map을 출력하도록 구성함.

전체적인 모델 흐름은 다음과 같음.

```text
Input
[Batch, 3, 32, 32]

↓

Convolution Block 1
[Batch, 32, 32, 32]

↓

Max Pooling
[Batch, 32, 16, 16]

↓

Convolution Block 2
[Batch, 64, 16, 16]

↓

Max Pooling
[Batch, 64, 8, 8]

↓

Convolution Block 3
[Batch, 128, 8, 8]

↓

Global Average Pooling
[Batch, 128, 1, 1]

↓

Flatten
[Batch, 128]

↓

Linear Classifier
[Batch, 10]
```

모델은 `models/cam_cnn.py`에 구현함.

> #### models/cam_cnn.py

```python
import torch
import torch.nn as nn


class CAMCNN(nn.Module):
    """
    CNN architecture for Class Activation Mapping

    Conv Layers -> Global Average Pooling -> Linear Classifier
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Input: [B, 3, 32, 32]
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            # [B, 32, 16, 16]
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            # [B, 64, 8, 8]
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(
            in_features=128,
            out_features=num_classes,
        )

    def forward(self, x, return_features=False):
        feature_maps = self.features(x)

        pooled = self.gap(feature_maps)
        pooled = pooled.view(pooled.size(0), -1)

        logits = self.classifier(pooled)

        if return_features:
            return logits, feature_maps

        return logits
```

---

### Convolution Feature Map의 의미

모델의 마지막 Convolution Layer 출력 Shape은 다음과 같음.

```text
[Batch, 128, 8, 8]
```

여기서 `128×8×8`을 하나의 Feature Map으로 이해하면 안 됨.

실제로는 다음과 같이 `8×8` 크기의 Feature Map이 128개 존재하는 것.

```text
Feature Map 1   : 8×8
Feature Map 2   : 8×8
Feature Map 3   : 8×8
...
Feature Map 128 : 8×8
```

각 Feature Map은 입력 이미지에서 서로 다른 특징을 검출함.

예를 들어 학습이 진행된 이후에는 특정 Channel이 다음과 같은 특징에 반응할 수 있음.

* 객체의 윤곽
* 색상 변화
* Texture
* 귀나 눈과 같은 일부 객체 특징
* 차량의 바퀴나 창문
* 배경과 객체의 경계

다만 특정 Feature Map이 반드시 사람이 이해할 수 있는 하나의 의미만 표현하는 것은 아님.

여러 Feature Map이 함께 조합되면서 최종 클래스를 판별하게 됨.

---

### Global Average Pooling

CAM 구조에서 가장 중요한 요소는 **Global Average Pooling, GAP**임.

> **GAP**
> : 각 Feature Map의 모든 공간 값을 평균내어
Feature Map 하나를 하나의 Scalar 값으로 변환함.

즉, `8×8` Feature Map 하나가 하나의 값으로 변환됨.
현재 모델에는 Feature Map이 128개 있으므로, GAP 이후에는 128개의 값이 생성됨.

```text
Before GAP
[Batch, 128, 8, 8]

↓

After GAP
[Batch, 128, 1, 1]

↓

Flatten
[Batch, 128]
```

따라서 Linear Classifier의 입력 차원도 128이 되어야 함.

```python
self.classifier = nn.Linear(
    in_features=128,
    out_features=10,
)
```

---

### Max Pooling과 GAP의 차이

모델 내부에는 Max Pooling과 Global Average Pooling이 모두 사용됨.

처음에는 둘 다 Feature Map의 크기를 줄이는 연산이므로
역할이 비슷해 보일 수 있음.

하지만 모델 내 사용 목적은 다름.

### i) Max Pooling

Feature Map의 일부 영역에서 가장 큰 값을 선택하여
공간 해상도를 줄이는 역할을 함.

```text
32×32
↓
16×16
↓
8×8
```

Max Pooling 이후에도 공간 정보가 남아 있음.

```text
[Batch, Channel, Height, Width]
```

### ii) Global Average Pooling

Feature Map 전체 공간의 평균을 계산함.

```text
8×8
↓
1×1
```

즉, 공간 차원을 완전히 제거하고 Channel별 대표값만 남김.

```text
[Batch, Channel, 1, 1]
```


| 연산                     | 주요 목적                                |
| ---------------------- | ------------------------------------ |
| Max Pooling            | 공간 해상도 축소 및 강한 특징 유지                 |
| Global Average Pooling | Feature Map별 대표값 생성                  |
| Flatten                | Tensor Shape을 Linear Layer 입력 형태로 변경 |

---

### 일반 CNN VS CAM

일반적인 CNN에서는 마지막 Feature Map 전체를 Flatten한 뒤
여러 개의 Fully Connected Layer를 사용하는 경우가 많음.

예를 들어 현재 Feature Map을 그대로 Flatten하면 다음과 같음.

```text
128 × 8 × 8 = 8192
```

즉, `8192`개의 값을 Fully Connected Layer의 입력으로 사용하게 됨.

> **하지만 CAM에서는 이렇게 하지 않고, 각 Feature Map을 GAP로 평균낸 후
128개의 값만 Linear Classifier에 전달함.
**
>
```text
128 Feature Maps
↓
GAP
↓
128 Values
↓
Linear Classifier
```
>
이 구조를 사용해야 각 Feature Map과 Classifier Weight의 대응 관계가 유지됨.

---

### CAM 수식과 코드의 대응

CAM 논문의 주요 수식은 다음과 같음.

$$
M_c(x,y)=\sum_k w_k^c f_k(x,y)
$$

* $M_c(x,y)$
  클래스 $c$에 대한 CAM의 위치 $(x,y)$ 값

* $f_k(x,y)$
  마지막 Convolution Layer의 $k$번째 Feature Map

* $w_k^c$
  $k$번째 Feature Map과 클래스 $c$를 연결하는 Linear Classifier Weight

* $k$
  Feature Map Channel Index

현재 구현한 모델과 연결하면 다음과 같음.

| CAM 수식     | PyTorch 코드                   |
| ---------- | ---------------------------- |
| $f_k(x,y)$ | `feature_maps`               |
| $w_k^c$    | `model.classifier.weight[c]` |
| $c$        | CAM을 생성할 Class Index         |
| $M_c(x,y)$ | 최종 CAM Heatmap               |

예를 들어 CIFAR-10의 세 번째 클래스에 대한 Weight를 가져오면 다음과 같음.

```python
class_weights = model.classifier.weight[3]
```

Classifier Weight의 전체 Shape은 다음과 같음.

```text
[10, 128]
```

여기서 클래스 하나를 선택하면 다음 Shape이 됨.

```text
[128]
```

입력 이미지 한 장에 대한 마지막 Feature Map Shape은 다음과 같음.

```text
[128, 8, 8]
```

따라서 클래스 Weight와 Feature Map을 Channel별로 곱하고 모두 합하면
`8×8` 크기의 CAM을 생성할 수 있음.

```text
Class Weight
[128]

×

Feature Maps
[128, 8, 8]

↓

Channel-wise Weighted Sum

↓

CAM
[8, 8]
```

---

### 모델 구조 검증

모델을 작성한 뒤 바로 학습을 진행하지 않고,
Dummy Input을 이용해 모델의 Tensor Shape과 CAM 구조 조건을 검증함.

입력은 `CIFAR-10` 이미지 4장을 가정해서 생성함.

```python
dummy_input = torch.randn(
    4,
    3,
    32,
    32,
)
```

`test_model.py`에서는 다음 항목들을 확인함.

* 입력 이미지 Shape
* 마지막 Feature Map Shape
* GAP 출력 Shape
* Flatten 결과 Shape
* Classification Logits Shape
* Classifier Weight Shape
* 마지막 Feature Map Channel 수와 Classifier 입력 차원의 일치 여부

> #### test_model.py

```python
import torch

from models.cam_cnn import CAMCNN


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("CAM CNN Structure Test")
    print("=" * 60)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = CAMCNN(num_classes=10).to(device)
    model.eval()

    dummy_input = torch.randn(
        4,
        3,
        32,
        32,
        device=device,
    )

    with torch.no_grad():
        logits, feature_maps = model(
            dummy_input,
            return_features=True,
        )

        pooled = model.gap(feature_maps)
        flattened = torch.flatten(pooled, start_dim=1)

    print("\n[Tensor Shapes]")

    print(f"Input                 : {dummy_input.shape}")
    print(f"Last feature maps     : {feature_maps.shape}")
    print(f"GAP output            : {pooled.shape}")
    print(f"Flattened GAP output  : {flattened.shape}")
    print(f"Logits                : {logits.shape}")

    print("\n[Classifier Information]")

    print(
        f"Classifier input units  : "
        f"{model.classifier.in_features}"
    )

    print(
        f"Classifier output units : "
        f"{model.classifier.out_features}"
    )

    print(
        f"Classifier weight shape : "
        f"{model.classifier.weight.shape}"
    )

    print(
        f"Classifier bias shape   : "
        f"{model.classifier.bias.shape}"
    )

    feature_channels = feature_maps.shape[1]
    classifier_input_units = model.classifier.in_features

    assert feature_maps.ndim == 4

    assert feature_channels == classifier_input_units

    assert pooled.shape[2:] == (1, 1)

    assert flattened.shape == (
        dummy_input.shape[0],
        128,
    )

    assert logits.shape == (
        dummy_input.shape[0],
        10,
    )

    print("\n[CAM Structural Conditions]")

    print(f"Feature-map channels : {feature_channels}")
    print(f"Classifier inputs    : {classifier_input_units}")

    print(
        "Feature channels == "
        "Classifier input units: PASS"
    )

    print("GAP output spatial size == 1x1: PASS")
    print("Output class dimension == 10: PASS")

    print("\nAll structure tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

### 모델 구조 검증 결과


```bash
python test_model.py
```

출력 결과는 다음과 같았음.

```text
============================================================
CAM CNN Structure Test
============================================================
Device: cuda
GPU: NVIDIA GeForce RTX 2070

[Tensor Shapes]
Input                 : torch.Size([4, 3, 32, 32])
Last feature maps     : torch.Size([4, 128, 8, 8])
GAP output            : torch.Size([4, 128, 1, 1])
Flattened GAP output  : torch.Size([4, 128])
Logits                : torch.Size([4, 10])

[Classifier Information]
Classifier input units  : 128
Classifier output units : 10
Classifier weight shape : torch.Size([10, 128])
Classifier bias shape   : torch.Size([10])

[CAM Structural Conditions]
Feature-map channels : 128
Classifier inputs    : 128
Feature channels == Classifier input units: PASS
GAP output spatial size == 1x1: PASS
Output class dimension == 10: PASS

All structure tests passed.
============================================================
```

모든 조건이 정상적으로 통과했음.

---

### Shape 변화 정리

모델 내부 Tensor Shape은 다음과 같이 변화함.

```text
Input
[4, 3, 32, 32]

↓

Last Feature Maps
[4, 128, 8, 8]

↓

Global Average Pooling
[4, 128, 1, 1]

↓

Flatten
[4, 128]

↓

Linear Classifier
[4, 10]
```

각 숫자의 의미는 다음과 같음.

| 값       | 의미                      |
| ------- | ----------------------- |
| `4`     | Batch에 포함된 이미지 수        |
| `3`     | RGB Channel             |
| `32×32` | CIFAR-10 입력 이미지 해상도     |
| `128`   | 마지막 Feature Map 개수      |
| `8×8`   | 마지막 Feature Map의 공간 해상도 |
| `10`    | CIFAR-10 클래스 수          |

가장 중요한 부분은 다음 조건임.

```text
마지막 Feature Map Channel 수 = 128
Classifier 입력 차원 = 128
```

두 값이 동일하기 때문에 각 Feature Map Channel과
Linear Classifier Weight를 직접 대응시킬 수 있음.

따라서 현재 모델은 아직 학습되지 않았지만,
구조적으로는 CAM을 계산할 수 있는 상태임.

---

### CIFAR-10 데이터셋 구성

모델 학습과 평가에는 `CIFAR-10` 데이터셋을 사용함.

CIFAR-10은 `32×32` 크기의 RGB 이미지로 구성되어 있으며,
총 10개의 클래스를 포함하는 이미지 분류 데이터셋임.

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

데이터셋 구성은 다음과 같음.

| 구분 | 이미지 수 |
| --- | ---: |
| Training Dataset | 50,000 |
| Test Dataset | 10,000 |
| Class 수 | 10 |
| Class별 Training Image | 5,000 |
| Class별 Test Image | 1,000 |

Training Dataset 50,000장 중 일부를 Validation Dataset으로 분리해서 모델 학습 과정에서 일반화 성능을 확인함.

```text
CIFAR-10 Training Dataset
50,000 Images

↓

Training Dataset
45,000 Images

+

Validation Dataset
5,000 Images
```

최종 모델 성능은 학습에 사용하지 않은
Test Dataset 10,000장을 이용해 평가함.

---

### 데이터 전처리

`CIFAR-10` 이미지에 다음 전처리를 적용함.

```text
Training Dataset
    │
    ├── Random Crop
    ├── Random Horizontal Flip
    ├── ToTensor
    └── Normalize

Validation / Test Dataset
    │
    ├── ToTensor
    └── Normalize
```

Training Dataset에는 데이터 증강을 적용하고,
Validation 및 Test Dataset에는 정규화만 적용함.

**데이터 증강을 적용한 이유**
:모델이 학습 이미지의 위치나 방향에 지나치게 의존하는 것을 방지하기 위함임.

ex. 동일한 자동차 이미지라도 객체의 위치나 좌우 방향이 달라질 수 있으므로,
Random Crop과 Horizontal Flip을 통해 다양한 입력 형태를 학습하도록 함.

**Normalization**에는 `CIFAR-10`의 Channel별 평균과 표준편차를 사용함.

```python
CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616,
)
```

정규화는 다음 식으로 수행됨.

$$
x_{\text{normalized}}
=
\frac{x-\mu}{\sigma}
$$

* $x$  
  입력 Pixel 값

* $\mu$  
  해당 Channel의 평균

* $\sigma$  
  해당 Channel의 표준편차

---

### 모델 학습

모델은 총 `100 Epoch` 동안 학습함.

```bash
python train.py --epochs 100
```

학습 과정에서는 매 Epoch마다 다음 값을 계산함.

* Training Loss
* Training Accuracy
* Validation Loss
* Validation Accuracy

마지막 Epoch의 모델을 무조건 사용하는 것이 아니라,
Validation Accuracy가 가장 높게 측정된 시점의 Weight를 저장함.

```text
Epoch 1
    │
    ├── Training
    ├── Validation
    └── Validation Accuracy 확인

Epoch 2
    │
    ├── Training
    ├── Validation
    └── Validation Accuracy 확인

...

Epoch 100
    │
    ├── Training
    ├── Validation
    └── Validation Accuracy 확인
```

현재 Epoch의 Validation Accuracy가 기존 최고 성능보다 높으면
다음 경로에 모델 Checkpoint를 저장함.

```text
results/
└── checkpoints/
    └── best_model.pth
```

> **100 Epoch까지 학습하더라도 실제 평가에는 마지막 Epoch의 모델이 아니라
Validation Dataset에서 가장 높은 성능을 기록한 `best_model.pth`를 사용함.**

---

### 초기 학습과 최종 학습 비교

구현 검증 단계에서는 먼저 모델이 정상적으로 학습되는지 확인하기 위해
`2 Epoch`만 학습함.

초기 학습 결과는 다음과 같았음.

```text
Epoch 1
Train Accuracy : 약 49%
Validation Accuracy : 약 57%

Epoch 2
Train Accuracy : 약 62%
Validation Accuracy : 약 59%
```

이 모델을 Test Dataset에서 평가한 결과는 다음과 같았음.

```text
Test Accuracy : 59.86%
```

2 Epoch만으로도 학습 코드와 모델 구조가 정상적으로 동작한다는 것은 확인할 수 있었지만,
모델의 분류 성능이 충분하지 않아 생성된 CAM의 신뢰성도 제한적이었음.

따라서 최종 실험에서는 모델을 `100 Epoch`까지 학습함.

```text
2 Epoch Model
Test Accuracy : 59.86%

↓

100 Epoch Model
Test Accuracy : 84.98%
```

정확도가 약 `25.12%p` 향상됨.

> **CAM은 모델이 사용한 판단 근거를 보여주는 방법이므로,
분류 모델 자체의 성능이 낮으면 시각화된 판단 근거 역시 불안정할 수 있음.**
>
> **★★ 따라서 CAM을 해석하기 전에 먼저 충분한 분류 성능을 확보하는 과정이 필요함.**

---

### 모델 성능 평가

100 Epoch 학습 후 저장된 `best_model.pth`를 이용해
Test Dataset 10,000장에 대한 성능을 평가함.

```bash
python evaluate.py
```

평가 결과는 다음과 같았음.

```text
============================================================
CAM CNN Test Evaluation
============================================================
Device     : cuda
Checkpoint : ./results/checkpoints/best_model.pth
============================================================
Test Loss     : 0.4681
Test Accuracy : 84.98%
============================================================
```

최종 Test Accuracy는 `84.98%`로 측정됨.

---

### 클래스별 성능

클래스별 Precision, Recall, F1-score는 다음과 같았음.

| Class | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| airplane | 0.8418 | 0.8940 | 0.8671 |
| automobile | 0.9173 | 0.9090 | 0.9131 |
| bird | 0.8333 | 0.7800 | 0.8058 |
| cat | 0.7108 | 0.6980 | 0.7043 |
| dog | 0.7925 | 0.7870 | 0.7898 |
| frog | 0.8762 | 0.8990 | 0.8875 |
| horse | 0.8932 | 0.8700 | 0.8815 |
| ship | 0.9050 | 0.9140 | 0.9095 |
| truck | 0.9027 | 0.8910 | 0.8968 |

전체 평균 성능은 다음과 같음.

| Metric | Score |
| --- | ---: |
| Accuracy | 0.8498 |
| Macro Precision | 0.8497 |
| Macro Recall | 0.8498 |
| Macro F1-score | 0.8495 |

`automobile`, `ship`, `truck` 클래스에서 높은 성능을 기록한 반면,
`cat`, `dog`, `bird` 클래스는 상대적으로 낮은 성능을 보였음.

특히 `cat` 클래스의 F1-score는 `0.7043`으로 가장 낮았음.

이는 CIFAR-10의 낮은 해상도에서 고양이와 개처럼
형태 및 Texture가 유사한 동물 클래스를 구분하기 어렵기 때문으로 해석할 수 있음.

---

### +Precision, Recall, F1-score의 의미

#### i) Precision

모델이 특정 클래스라고 예측한 이미지 중
실제로 해당 클래스인 이미지의 비율임.
>$$
\text{Precision}
=
\frac{TP}{TP+FP}
>$$

예를 들어 모델이 `cat`이라고 예측한 이미지 중
실제로 고양이인 이미지가 얼마나 되는지를 의미함.

---

#### ii) Recall

실제 특정 클래스 이미지 중
모델이 해당 클래스로 올바르게 예측한 비율임.
>$$
\text{Recall}
=
\frac{TP}{TP+FN}
>$$

예를 들어 실제 고양이 이미지 전체 중
모델이 `cat`이라고 찾아낸 이미지의 비율을 의미함.

---

#### iii) F1-score

Precision과 Recall의 조화평균임.
>$$
F1
=
2
\times
\frac{\text{Precision}\times\text{Recall}}
{\text{Precision}+\text{Recall}}
>$$

Precision과 Recall 중 어느 한쪽만 높아도 좋은 점수를 얻는 것을 방지하며,
두 지표의 균형을 평가할 수 있음.

---

### 평가 결과 저장

모델 평가 결과는 다음 경로에 저장함.

```text
results/
└── evaluation/
    ├── classification_report.txt
    └── confusion_matrix.png
```

`classification_report.txt`에는 클래스별 Precision, Recall, F1-score가 저장되며,
`confusion_matrix.png`에는 실제 클래스와 예측 클래스 간의 혼동 관계가 저장됨.

Confusion Matrix를 통해 단순히 전체 Accuracy만 확인하는 것이 아니라,
어떤 클래스끼리 자주 혼동되는지를 분석할 수 있음.

---

### CAM 생성 과정

학습된 모델로부터 CAM을 생성하는 과정은 다음과 같음.

```text
Input Image
      │
      ▼
Forward Propagation
      │
      ├── Classification Logits
      │
      └── Last Convolution Feature Maps
                  │
                  ▼
         Target Class Weight 선택
                  │
                  ▼
       Channel-wise Weighted Sum
                  │
                  ▼
              ReLU
                  │
                  ▼
             Normalize
                  │
                  ▼
             Upsampling
                  │
                  ▼
           CAM Heatmap 생성
```

> CAM 계산에는 다음 두 가지 정보가 필요함.
>```text
1. 마지막 Convolution Layer의 Feature Maps
>
2. CAM을 생성할 클래스의 Linear Classifier Weight
>```

---

### Feature Map 추출

마지막 Convolution Layer의 Feature Map은
**Forward Hook**을 이용해 추출함.

> **Forward Hook**
: 모델의 Forward Propagation이 수행될 때
특정 Layer의 입력 또는 출력을 별도로 저장할 수 있게 해주는 기능임.
>
```python
last_conv_layer.register_forward_hook(
    save_feature_maps
)
```
>
Hook을 사용하면 모델의 기존 `forward()` 함수를 수정하지 않고도
중간 Feature Map을 가져올 수 있음.

현재 모델에서 추출되는 마지막 Feature Map Shape은 다음과 같음.

```text
[1, 128, 8, 8]
```

Batch에서 이미지 한 장을 선택하면 다음과 같음.

```text
[128, 8, 8]
```

---

### 클래스 Weight 추출

Linear Classifier의 Weight Shape은 다음과 같음.

```text
[10, 128]
```

10개의 클래스 각각에 대해 128개의 Feature Map Weight가 존재함.

특정 클래스 $c$에 대한 Weight는 다음과 같이 가져올 수 있음.

```python
class_weights = classifier_weights[class_index]
```

출력 Shape은 다음과 같음.

```text
[128]
```

---

### CAM 계산

CAM은 다음 코드로 계산함.

```python
cam = torch.sum(
    class_weights[:, None, None] * feature_maps,
    dim=0,
)
```

`class_weights`의 원래 Shape은 `[128]`이므로,
각 Feature Map에 Weight를 곱할 수 있도록 다음 Shape으로 확장함.

```text
[128]
↓
[128, 1, 1]
```

Feature Map Shape은 다음과 같음.

```text
[128, 8, 8]
```

따라서 Broadcasting을 통해 각 Channel에 해당 클래스 Weight가 곱해짐.

```text
Class Weights
[128, 1, 1]

×

Feature Maps
[128, 8, 8]

↓

Weighted Feature Maps
[128, 8, 8]
```

이후 Channel 차원인 `dim=0` 방향으로 모두 합산함.

```text
[128, 8, 8]

↓

Sum over Channel Dimension

↓

[8, 8]
```

이 결과가 클래스 $c$에 대한 CAM임.

---

### ReLU 적용

가중합 결과에는 양수와 음수가 모두 포함될 수 있음.

```python
cam = torch.relu(cam)
```

양수 값은 해당 영역이 클래스 점수를 높이는 방향으로 기여했다는 것을 의미하고,
음수 값은 해당 클래스 점수를 낮추는 방향으로 기여했음을 의미함.

이번 시각화에서는 해당 클래스를 지지하는 영역을 확인하기 위해
ReLU를 적용하여 양의 기여만 남김.

```text
Positive Contribution
→ Target Class를 지지하는 영역

Negative Contribution
→ Target Class 점수를 낮추는 영역
```

---

### CAM 정규화

CAM 값을 `0~1` 범위로 변환함.

```python
cam = (
    cam - cam.min()
) / (
    cam.max() - cam.min()
)
```

정규화 이후 값의 의미는 다음과 같음.

```text
0에 가까운 값
→ 해당 클래스 판단에 대한 활성도가 낮은 영역

1에 가까운 값
→ 해당 클래스 판단에 대한 활성도가 높은 영역
```

---

### CAM Upsampling

원본 CAM의 크기는 마지막 Feature Map과 동일한 `8×8`임.

하지만 입력 이미지 크기는 `32×32`이므로,
원본 이미지 위에 CAM을 겹쳐 보기 위해 CAM을 Upsampling함.

```python
cam = F.interpolate(
    cam.unsqueeze(0).unsqueeze(0),
    size=(32, 32),
    mode="bilinear",
    align_corners=False,
)
```

```text
Original CAM
[8, 8]

↓

Bilinear Interpolation

↓

Upsampled CAM
[32, 32]
```

> **Upsampling은 새로운 공간 정보를 생성하는 것이 아님.**
>
> 기존 `8×8` CAM을 확대하는 것이므로,
CAM의 실제 해상도는 여전히 마지막 Feature Map의 공간 해상도에 의해 제한됨.

이로 인해 CAM Heatmap의 경계가 다소 거칠게 나타날 수 있음.

---

### CAM 시각화

CAM 시각화는 다음 명령어로 실행함.

```bash
python visualize_cam.py --num-images 10 --target predicted
```

`--target predicted`는 모델이 실제로 예측한 클래스를 기준으로 CAM을 생성함.

```text
입력 이미지
      │
      ▼
모델 예측
      │
      ▼
예측 클래스 Weight 선택
      │
      ▼
Predicted CAM 생성
```

결과는 다음 경로에 저장함.

```text
results/
└── cam/
    └── predicted/
        ├── correct/
        └── wrong/
```

모델의 예측이 정답과 일치하면 `correct/`,
예측이 틀리면 `wrong/` 폴더에 자동으로 저장되도록 구현함.

---

### i) Predicted CAM

Predicted CAM은 모델이 최종적으로 선택한 클래스의 Weight를 사용함.

```python
target_class_index = predicted_label
```

Predicted CAM이 답하는 질문은 다음과 같음.

> **모델은 왜 이 클래스로 예측했는가?**

예를 들어 모델이 이미지를 `dog`라고 예측했다면,
Dog Classifier Weight를 이용해 CAM을 계산함.

```text
Feature Maps
      ×
Dog Classifier Weights
      ↓
Dog CAM
```

따라서 Predicted CAM은 모델의 실제 판단 근거를 분석할 때 사용함.

---

### ii) True CAM

정답 클래스를 기준으로 한 CAM도 생성함.

```bash
python visualize_cam.py --num-images 10 --target true
```

`--target true`는 데이터셋의 Ground Truth Label을 기준으로 CAM을 생성함.

```python
target_class_index = true_label
```

True CAM이 답하는 질문은 다음과 같음.

> **같은 Feature Map에 정답 클래스 Weight를 적용하면 어느 영역이 활성화되는가?**

예를 들어 실제 정답이 `automobile`이지만
모델이 `dog`라고 예측한 경우 다음 두 CAM을 생성할 수 있음.

```text
Predicted CAM
Feature Maps × Dog Weights

True CAM
Feature Maps × Automobile Weights
```

---

### Predicted CAM과 True CAM의 차이는?

Predicted CAM과 True CAM을 계산할 때
마지막 Convolution Feature Map은 동일함.

> 달라지는 것은 Linear Classifier에서 선택하는 클래스 Weight뿐.

```text
Same Feature Maps
[128, 8, 8]
```

>Predicted CAM은 다음과 같이 계산됨.
>
```text
Same Feature Maps
        ×
Predicted Class Weights
        ↓
Predicted CAM
```

>True CAM은 다음과 같이 계산됨.
>
```text
Same Feature Maps
        ×
True Class Weights
        ↓
True CAM
```

두 CAM은 서로 다른 이미지를 분석하거나
서로 다른 Feature Map을 사용하는 것이 아님.

> **동일한 Feature Map을 서로 다른 클래스 관점에서 가중합한 결과.**

---

### 정답 샘플에서 두 CAM이 동일한 이유

모델의 예측 클래스와 실제 정답 클래스가 같다면
Predicted CAM과 True CAM은 동일함.

예를 들어 다음과 같은 경우임.

```text
True Class      : cat
Predicted Class : cat
```

Predicted CAM과 True CAM 모두 `cat` 클래스 Weight를 사용함.

```text
Predicted CAM
Feature Maps × Cat Weights

True CAM
Feature Maps × Cat Weights
```

따라서 두 CAM의 결과가 동일하게 나타남.

초기 10개 샘플에서는 모두 분류에 성공했기 때문에
Predicted CAM과 True CAM 간의 차이를 확인하기 어려웠음.

이에 따라 오분류된 샘플만 자동으로 탐색하여
두 CAM을 비교하는 코드를 추가로 구현함.

---

### 오분류 샘플 CAM 비교

오분류된 Test Image를 자동으로 탐색하고,
동일한 이미지에 대해 Predicted CAM과 True CAM을 함께 생성함.

```bash
python visualize_misclassified_cam.py --num-images 20
```

실행 과정은 다음과 같음.

```text
Test Dataset
      │
      ▼
이미지 한 장씩 모델에 입력
      │
      ▼
Predicted Label 계산
      │
      ▼
Predicted Label == True Label?
      │
      ├── Yes → 건너뜀
      │
      └── No  → CAM 생성
                    │
                    ├── Predicted CAM
                    └── True CAM
```

총 146개의 Test Sample을 확인한 뒤
20개의 오분류 이미지를 저장함.

```text
Checked Samples : 146
Saved Samples   : 20
```

결과는 다음 경로에 저장함.

```text
results/
└── cam_misclassified/
    ├── sample_00024_true_dog_pred_deer.png
    ├── sample_00032_true_deer_pred_frog.png
    ├── sample_00035_true_bird_pred_cat.png
    └── ...
```

---

### 오분류 결과

탐색된 일부 오분류 결과는 다음과 같았음.

| Index | True Class | Predicted Class | P(Predicted) | P(True) |
| ---: | --- | --- | ---: | ---: |
| 24 | dog | deer | 95.73% | 4.10% |
| 32 | deer | frog | 63.51% | 24.82% |
| 35 | bird | cat | 87.10% | 1.10% |
| 52 | airplane | horse | 36.34% | 0.78% |
| 57 | horse | cat | 92.78% | 0.97% |
| 58 | deer | dog | 68.00% | 1.81% |
| 61 | cat | dog | 88.87% | 6.43% |
| 68 | cat | dog | 95.49% | 3.08% |
| 70 | bird | cat | 33.03% | 9.84% |
| 76 | truck | airplane | 62.55% | 37.24% |
| 78 | cat | dog | 74.89% | 24.58% |
| 85 | dog | horse | 42.85% | 17.23% |
| 86 | bird | horse | 86.62% | 7.70% |
| 106 | cat | dog | 48.42% | 3.59% |
| 112 | frog | bird | 93.02% | 6.84% |
| 127 | cat | bird | 46.69% | 39.29% |
| 128 | dog | cat | 51.28% | 25.78% |
| 129 | bird | airplane | 50.42% | 49.45% |
| 139 | truck | ship | 69.63% | 18.62% |
| 145 | horse | dog | 50.01% | 19.65% |

> 오분류 결과에서는 동물 클래스 간 혼동이 자주 나타났음.

```text
cat → dog
dog → cat
dog → deer
deer → dog
bird → cat
horse → cat
```

특히 `cat`과 `dog`처럼 시각적으로 유사한 클래스 간 오분류가 반복적으로 발생함.

---

### 확신도가 높은 오분류

일부 이미지에서는 모델이 매우 높은 확률로 잘못된 클래스를 선택함.

> ![](https://velog.velcdn.com/images/keepbini366/post/c427de7a-25ef-4a8e-b9dd-2fe53232672b/image.png)
>```text
>True : dog
>Pred : deer
>
>P(deer) : 95.73%
>P(dog)  : 4.10%
>```

> ![](https://velog.velcdn.com/images/keepbini366/post/a5feba66-1452-40ed-9d6a-da274df8c621/image.png)
>```text
>True : horse
>Pred : cat
>
>P(cat)   : 92.78%
>P(horse) : 0.97%
>```

> ![](https://velog.velcdn.com/images/keepbini366/post/997fa1bc-94e8-4ff6-9674-007b7e65cbd3/image.png)
>```text
>True : frog
>Pred : bird
>
>P(bird) : 93.02%
>P(frog) : 6.84%
>```

이러한 사례는 모델이 단순히 두 클래스 사이에서 고민한 것도 아니고,
잘못된 특징을 특정 클래스의 강한 근거로 사용했음을 의미.

> CAM을 통해 다음 내용을 확인할 필요가 있음.
>
* 모델이 객체 자체를 보고 있었는가?
* 객체의 특정 부분만 과도하게 사용했는가?
* 배경이나 주변 Context에 의존했는가?
* 정답 CAM과 예측 CAM의 활성 영역이 서로 다른가?

---

### 경계에 가까운 오분류

일부 샘플에서는 예측 클래스와 정답 클래스의 확률 차이가 매우 작았음.

> ![](https://velog.velcdn.com/images/keepbini366/post/09ad1dbb-75c1-4ab8-af1b-edc1eb9fd20e/image.png)
>```text
True : bird
Pred : airplane
>
P(airplane) : 50.42%
P(bird)     : 49.45%
>```

두 클래스 간 확률 차이는 `0.97%p`에 불과함.

이는 모델이 두 클래스를 거의 비슷한 수준으로 판단했으나,
Airplane Logit이 Bird Logit보다 근소하게 높아 최종적으로 `airplane`을 선택한 경우임.

---

### i) 위 오분류 사례(Bird → Airplane)를 좀 더 분석해 보자면,  
실제 정답이 `bird`이지만
모델이 `airplane`으로 예측한 사례임.

```text
True Class      : bird
Predicted Class : airplane

P(airplane) : 50.42%
P(bird)     : 49.45%
```

원본 이미지에는 나뭇가지 위에 앉아 있는 새가 존재함.

CIFAR-10 이미지 해상도는 `32×32`로 매우 낮기 때문에,
새의 몸통과 날개 형태가 비행기의 동체 및 날개와 유사한 패턴으로 표현될 수 있음.

---

### ii) Predicted CAM 분석

Predicted CAM은 `airplane` 클래스 Weight를 사용해 생성함.

```text
Feature Maps
      ×
Airplane Classifier Weights
      ↓
Predicted CAM
```

Predicted CAM에서는 이미지 오른쪽에 위치한
새의 몸통과 날개 부근이 강하게 활성화됨.

중요한 점은 모델이 하늘이나 배경만을 근거로
Airplane을 예측한 것이 아니라는 점임.

```text
낮은 활성도
→ 이미지 좌측 배경 및 하늘 일부

높은 활성도
→ 새의 몸통과 날개 부근
```

즉, 모델은 실제 객체가 존재하는 위치를 찾았지만,
해당 객체의 시각적 특징을 Airplane 클래스와 연결함.

---

### iii) True CAM 분석

True CAM은 `bird` 클래스 Weight를 사용해 생성함.

```text
Feature Maps
      ×
Bird Classifier Weights
      ↓
True CAM
```

True CAM 역시 새의 몸통과 날개가 위치한 영역에서 높은 활성도를 보임.

Predicted CAM과 True CAM 모두 대체로 동일한 객체 영역에 집중하고 있음.

```text
Predicted CAM
→ 새가 있는 위치를 Airplane의 근거로 사용

True CAM
→ 같은 위치를 Bird의 근거로 사용
```

---

### iv) 해석

해당 사례는 모델이 객체 위치를 잘못 찾은 경우와 구분해야 함.

모델은 새가 존재하는 영역을 다소 정상적으로 찾았음.

하지만 해당 영역에서 추출된 Feature가
Bird Classifier보다 Airplane Classifier에 근소하게 더 높은 점수를 제공함.

```text
CNN Feature Extractor
      │
      ▼
객체가 존재하는 위치의 Feature 추출
      │
      ├── Airplane Score : 50.42%
      └── Bird Score     : 49.45%
```

따라서 이 사례의 오분류 원인은 다음과 같이 정리할 수 있음.

> **객체 Localization 실패가 아니라,
올바르게 찾은 객체 특징에 대한 Class Discrimination 실패임.**

새의 형태를 비행기와 유사한 특징으로 해석한 것.

---

### 오분류와 잘못된 Localization은 다름

CAM을 분석하면서 확인한 중요한 점은 다음과 같음.

> **모델의 분류가 틀렸다고 해서 반드시 엉뚱한 위치를 본 것은 아님.**

오분류는 크게 다음과 같이 구분할 수 있음.

#### Case 1. 잘못된 영역을 본 경우

```text
실제 객체
→ 이미지 중앙

Predicted CAM
→ 배경이나 주변 Context 활성화
```

모델이 객체가 아닌 배경 특징에 의존해 잘못된 클래스를 예측한 경우임.

#### Case 2. 올바른 영역을 보고 잘못 해석한 경우

```text
실제 객체
→ 이미지 중앙

Predicted CAM
→ 실제 객체 영역 활성화

하지만
→ 잘못된 Class Weight에 더 높은 Score 부여
```

---

### CAM 시각화를 통해 확인한 점

이번 실습을 통해 CAM의 주요 동작을 코드와 결과로 확인할 수 있었음.

### i) Feature Map과 Classifier Weight의 직접적인 대응

CAM 구조에서는 마지막 Feature Map Channel 수와
Linear Classifier의 입력 차원이 동일함.

```text
Last Feature Maps
[128, 8, 8]

Classifier Weights
[10, 128]
```

이를 통해 클래스별 Weight와 각 Feature Map을 직접 대응시킬 수 있었음.

---

### ii) 동일한 Feature Map도 클래스에 따라 다르게 해석됨

Predicted CAM과 True CAM은 같은 Feature Map을 사용함.

```text
Same Feature Maps
      │
      ├── Predicted Class Weights
      │       ↓
      │   Predicted CAM
      │
      └── True Class Weights
              ↓
          True CAM
```

따라서 CNN이 추출한 Feature 자체가 바뀌는 게 아니라,
어떤 클래스 Weight를 적용하는지에 따라 활성 영역과 강도가 달라지는 것.

---

### iii) 높은 Confidence가 올바른 판단을 보장하ㄴ지는 않음

일부 오분류 이미지에서는 모델이 90% 이상의 확률로
잘못된 클래스를 예측함.

```text
dog → deer : 95.73%
horse → cat : 92.78%
frog → bird : 93.02%
```

Softmax Confidence가 높다는 사실만으로
모델의 판단이 올바르다고 볼 수 없음.

CAM은 이러한 High-confidence Wrong Prediction에서
모델이 어떤 특징을 강한 근거로 사용했는지 분석하는 데 활용할 수 있음.

---

### iv) CAM은 판단 근거를 보여주지만 인과관계를 완전히 증명하지는 않음

CAM에서 특정 영역이 강하게 활성화되었다는 것은
해당 영역의 Feature가 특정 클래스 점수에 크게 기여했음을 의미함.

하지만 이를 다음과 같이 과도하게 해석하면 안 됨.

```text
CAM이 빨간 영역을 표시함

≠

모델이 사람과 동일한 의미로
그 영역을 이해했다는 증거
```

예를 들어 새의 날개 부분이 활성화되었다고 해서
모델이 해당 영역을 인간처럼 명확하게 `날개`라는 개념으로 이해했다고 단정할 수 없음.

CAM은 모델 내부의 Class-discriminative Feature를
공간적으로 시각화한 결과로 이해해야 함.

---

### CAM의 한계

이번 구현과 시각화를 통해 CAM의 몇 가지 한계도 확인할 수 있었음.

### i) 특정 모델 구조가 필요함

CAM을 적용하려면 마지막 Convolution Layer 뒤에
GAP와 Linear Classifier가 직접 연결되어야 함.

```text
Last Conv Feature Maps
      ↓
GAP
      ↓
Linear Classifier
```

중간에 여러 Fully Connected Layer가 존재하거나
Feature Map을 Flatten하여 복잡한 분류기를 사용하는 모델에는
CAM을 직접 적용하기 어려움.

---

### ii) 마지막 Feature Map의 낮은 해상도

현재 CAM의 원본 해상도는 `8×8`임.

```text
Input Image
32×32

↓

Last Feature Maps
8×8

↓

CAM
8×8

↓

Upsampling
32×32
```

최종 시각화 결과는 `32×32`로 확대되지만,
실제 위치 정보는 `8×8` 수준에 머물러 있음.

따라서 작은 객체 영역이나 세밀한 경계를 정확하게 표현하기 어려우며,
Heatmap이 넓고 거칠게 나타날 수 있음.

---

### iii) 가장 판별적인 영역만 강조할 수 있음

CAM은 클래스 점수를 높이는 데 가장 강하게 기여한 영역을 강조함.

따라서 객체 전체를 나타내기보다
얼굴, 날개, 바퀴와 같은 일부 판별적 영역만 활성화할 수 있음.

```text
Object 전체 영역

≠

CAM 활성 영역
```

CAM은 Object Segmentation 결과가 아니라,
Class Discrimination에 중요했던 영역을 나타내는 결과임.

---

### iv) 시각화 결과의 정성적 해석

`CIFAR-10`에는 각 객체의 Bounding Box나 Segmentation Mask가 제공되지 않음.

따라서 이번 실습에서는 CAM이 실제 객체 영역과 얼마나 겹치는지를
IoU 등의 지표로 정확하게 평가하기 어려움.

>그렇기에 현재 결과는 주로 다음과 같은 정성적 분석에 기반함.
>
* Heatmap이 객체 위치에 형성되는가?
* 배경에 과도하게 반응하는가?
* 정답과 오답 샘플에서 활성 영역이 어떻게 달라지는가?
* Predicted CAM과 True CAM은 어떤 차이를 보이는가?

**정량적인 Localization 성능을 평가하려면
Bounding Box Annotation이 포함된 데이터셋을 사용해야 함.**

---

### 현재까지의 구현 결과

이번 실습에서 구현한 전체 과정은 다음과 같음.

```text
CAM 논문 구조 분석
      │
      ▼
CAM CNN 직접 구현
      │
      ▼
Dummy Input 기반 구조 검증
      │
      ▼
CIFAR-10 데이터셋 구성
      │
      ▼
100 Epoch 모델 학습
      │
      ▼
Test Dataset 성능 평가
      │
      ├── Accuracy
      ├── Precision
      ├── Recall
      ├── F1-score
      └── Confusion Matrix
      │
      ▼
마지막 Feature Map 추출
      │
      ▼
CAM 수식 직접 구현
      │
      ▼
Predicted CAM 생성
      │
      ▼
True CAM 생성
      │
      ▼
오분류 샘플 자동 탐색
      │
      ▼
Predicted CAM과 True CAM 비교
```

최종 Test Accuracy는 다음과 같았음.

```text
Test Loss     : 0.4681
Test Accuracy : 84.98%
```

> CAM 시각화를 통해 단순히 모델의 예측 결과만 확인하는 게 아니라, 
모델이 어떤 위치의 Feature를 각 클래스의 판단 근거로 사용했는지 확인할 수 있었음.
>
> ---
> **특히 모델이 오분류했다고 해서 항상 잘못된 위치를 본 것은 아니고,
올바른 객체 영역을 찾았더라도 추출된 특징을 잘못된 클래스로 해석할 수 있음.**

