import torch

from models.cam_cnn import CAMCNN


def main():
    # --------------------------------------------------
    # 1. Device 설정
    # --------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("CAM CNN Structure Test")
    print("=" * 60)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------
    # 2. 모델 생성
    # --------------------------------------------------
    model = CAMCNN(num_classes=10).to(device)

    # 아직 학습 전이므로 평가 모드로 설정
    model.eval()

    # CIFAR-10 이미지와 동일한 크기의 가상 입력
    dummy_input = torch.randn(
        4,
        3,
        32,
        32,
        device=device,
    )

    # --------------------------------------------------
    # 3. Forward propagation
    # --------------------------------------------------
    with torch.no_grad():
        logits, feature_maps = model(
            dummy_input,
            return_features=True,
        )

        pooled = model.gap(feature_maps)
        flattened = torch.flatten(pooled, start_dim=1)

    # --------------------------------------------------
    # 4. Shape 출력
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 5. CAM 구조 조건 검증
    # --------------------------------------------------
    feature_channels = feature_maps.shape[1]
    classifier_input_units = model.classifier.in_features

    assert feature_maps.ndim == 4, (
        "Feature maps must have shape [B, C, H, W]."
    )

    assert feature_channels == classifier_input_units, (
        "The number of final feature-map channels must match "
        "the classifier input dimension."
    )

    assert pooled.shape[2:] == (1, 1), (
        "GAP output spatial size must be 1x1."
    )

    assert flattened.shape == (dummy_input.shape[0], 128), (
        "Flattened GAP output must have shape [B, 128]."
    )

    assert logits.shape == (dummy_input.shape[0], 10), (
        "Logits must have shape [B, 10]."
    )

    print("\n[CAM Structural Conditions]")

    print(
        f"Feature-map channels : {feature_channels}"
    )

    print(
        f"Classifier inputs    : {classifier_input_units}"
    )

    print("Feature channels == Classifier input units: PASS")
    print("GAP output spatial size == 1x1: PASS")
    print("Output class dimension == 10: PASS")

    print("\nAll structure tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()