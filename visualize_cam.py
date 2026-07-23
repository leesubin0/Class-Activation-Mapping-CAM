import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10

try:
    from cam_cnn import CAMCNN
except ImportError:
    from models.cam_cnn import CAMCNN

from dataset import (
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
)


class FeatureMapExtractor:
    """
    마지막 Convolution Layer의 출력 Feature Map을 저장하는 Hook.

    CAM은 마지막 Convolution Feature Map과
    Linear Classifier Weight가 필요하다.
    """

    def __init__(self, model: nn.Module) -> None:
        self.feature_maps = None
        self.hook_handle = None

        last_conv_layer = self.find_last_conv_layer(model)

        self.hook_handle = last_conv_layer.register_forward_hook(
            self.save_feature_maps
        )

    @staticmethod
    def find_last_conv_layer(model: nn.Module) -> nn.Conv2d:
        """
        모델 안에서 마지막 nn.Conv2d Layer를 찾는다.
        """

        last_conv_layer = None

        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv_layer = module

        if last_conv_layer is None:
            raise ValueError(
                "모델에서 nn.Conv2d Layer를 찾을 수 없습니다."
            )

        return last_conv_layer

    def save_feature_maps(
        self,
        module: nn.Module,
        inputs,
        output: torch.Tensor,
    ) -> None:
        self.feature_maps = output.detach()

    def remove(self) -> None:
        if self.hook_handle is not None:
            self.hook_handle.remove()


def find_classifier_layer(model: nn.Module) -> nn.Linear:
    """
    CAM 계산에 사용할 마지막 Linear Layer를 찾는다.
    """

    last_linear_layer = None

    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear_layer = module

    if last_linear_layer is None:
        raise ValueError(
            "모델에서 nn.Linear Layer를 찾을 수 없습니다."
        )

    return last_linear_layer


def generate_cam(
    feature_maps: torch.Tensor,
    classifier_weights: torch.Tensor,
    class_index: int,
    output_size: tuple[int, int],
) -> np.ndarray:
    """
    CAM 논문의 핵심 수식 구현.

    M_c(x, y) = sum_k w_k^c * f_k(x, y)

    feature_maps:
        [1, K, H, W]

    classifier_weights:
        [number_of_classes, K]
    """

    if feature_maps.ndim != 4:
        raise ValueError(
            "Feature Map은 [B, C, H, W] 형태여야 합니다."
        )

    # Batch Size가 1이므로 첫 번째 이미지 선택
    feature_maps = feature_maps[0]

    # 선택한 Class의 Linear Weight
    class_weights = classifier_weights[class_index]

    if feature_maps.shape[0] != class_weights.shape[0]:
        raise ValueError(
            "Feature Map 채널 수와 Linear Layer 입력 차원이 "
            "일치하지 않습니다."
        )

    # 각 Feature Map에 Class Weight를 곱하고 모두 더함
    cam = torch.sum(
        class_weights[:, None, None] * feature_maps,
        dim=0,
    )

    # 양수 영역만 사용
    cam = torch.relu(cam)

    # 원본 이미지 크기로 확대
    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )

    cam = cam.squeeze()

    # 0~1 범위로 정규화
    cam_min = cam.min()
    cam_max = cam.max()

    if (cam_max - cam_min).item() > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)

    return cam.cpu().numpy()


def create_overlay(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    원본 이미지 위에 CAM Heatmap을 합성한다.
    """

    colormap = plt.get_cmap("jet")

    heatmap = colormap(cam)[..., :3]

    overlay = (
        (1.0 - alpha) * image
        + alpha * heatmap
    )

    return np.clip(
        overlay,
        0.0,
        1.0,
    )


def save_cam_result(
    original_image: np.ndarray,
    cam: np.ndarray,
    overlay: np.ndarray,
    true_class_name: str,
    predicted_class_name: str,
    target_class_name: str,
    confidence: float,
    save_path: Path,
) -> None:
    """
    원본, CAM, Overlay 이미지를 한 화면에 저장한다.
    """

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
    )
    axes[0].imshow(original_image)
    axes[0].set_title(
        f"Original\nTrue: {true_class_name}"
    )
    axes[0].axis("off")

    axes[1].imshow(
        cam,
        cmap="jet",
    )
    axes[1].set_title(
        f"CAM\nTarget Class: {target_class_name}"
    )
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay\n"
        f"Prediction: {predicted_class_name}\n"
        f"CAM Target: {target_class_name}\n"
        f"Confidence: {confidence:.2%}"
    )
    axes[2].axis("off")

    figure.tight_layout()

    figure.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize CAM on CIFAR-10"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./results/checkpoints/best_model.pth",
    )

    parser.add_argument(
        "--image-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--result-dir",
        type=str,
        default="./results/cam",
    )

    parser.add_argument(
        "--target",
        type=str,
        choices=["predicted", "true"],
        default="predicted",
        help=(
            "predicted: 모델이 예측한 Class의 CAM, "
            "true: 정답 Class의 CAM"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint를 찾을 수 없습니다: {checkpoint_path}"
        )

    # predicted 또는 true 기준으로 상위 폴더를 분리한다.
    result_dir = (
        Path(args.result_dir)
        / args.target
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("Class Activation Mapping Visualization")
    print("=" * 60)
    print(f"Device       : {device}")
    print(f"Checkpoint   : {checkpoint_path}")
    print(f"Start index  : {args.image_index}")
    print(f"Number       : {args.num_images}")
    print(f"CAM target   : {args.target}")
    print("=" * 60)

    normalized_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=CIFAR10_MEAN,
                std=CIFAR10_STD,
            ),
        ]
    )

    original_transform = transforms.ToTensor()

    normalized_dataset = CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=normalized_transform,
    )

    original_dataset = CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=original_transform,
    )

    model = CAMCNN(
        num_classes=10
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    feature_extractor = FeatureMapExtractor(model)
    classifier_layer = find_classifier_layer(model)

    classifier_weights = (
        classifier_layer.weight.detach()
    )

    end_index = min(
        args.image_index + args.num_images,
        len(normalized_dataset),
    )

    with torch.no_grad():
        for index in range(
            args.image_index,
            end_index,
        ):
            normalized_image, true_label = (
                normalized_dataset[index]
            )

            original_tensor, _ = (
                original_dataset[index]
            )

            input_tensor = (
                normalized_image
                .unsqueeze(0)
                .to(device)
            )

            logits = model(input_tensor)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predicted_label = (
                probabilities.argmax(dim=1).item()
            )

            confidence = probabilities[
                0,
                predicted_label,
            ].item()

            if args.target == "predicted":
                target_class_index = predicted_label
            else:
                target_class_index = true_label

            if feature_extractor.feature_maps is None:
                raise RuntimeError(
                    "Feature Map이 추출되지 않았습니다."
                )

            cam = generate_cam(
                feature_maps=feature_extractor.feature_maps,
                classifier_weights=classifier_weights,
                class_index=target_class_index,
                output_size=(32, 32),
            )

            original_image = (
                original_tensor
                .permute(1, 2, 0)
                .numpy()
            )

            overlay = create_overlay(
                image=original_image,
                cam=cam,
            )

            true_class_name = (
                CIFAR10_CLASSES[true_label]
            )

            predicted_class_name = (
                CIFAR10_CLASSES[predicted_label]
            )

            target_class_name = (
                CIFAR10_CLASSES[target_class_index]
            )

            correctness = (
                "correct"
                if predicted_label == true_label
                else "wrong"
            )

            save_name = (
                f"sample_{index:05d}_"
                f"true_{true_class_name}_"
                f"pred_{predicted_class_name}_"
                f"{correctness}_"
                f"cam_{target_class_name}.png"
            )

            # 정답 여부에 따라 correct 또는 wrong 폴더로 분리한다.
            classification_dir = (
                result_dir / correctness
            )

            classification_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            save_path = classification_dir / save_name

            save_cam_result(
                original_image=original_image,
                cam=cam,
                overlay=overlay,
                true_class_name=true_class_name,
                predicted_class_name=predicted_class_name,
                target_class_name=target_class_name,
                confidence=confidence,
                save_path=save_path,
            )

            relative_save_path = save_path.relative_to(
                Path(args.result_dir)
            )

            print(
                f"[{index:05d}] "
                f"True: {true_class_name:<10} | "
                f"Pred: {predicted_class_name:<10} | "
                f"Confidence: {confidence:.2%} | "
                f"Saved: {relative_save_path}"
            )

    feature_extractor.remove()

    print("=" * 60)
    print("CAM visualization completed")
    print(f"Result directory: {result_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()