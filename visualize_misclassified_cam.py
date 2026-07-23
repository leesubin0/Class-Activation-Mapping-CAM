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
    마지막 Convolution Layer의 출력 Feature Map을 저장한다.
    """

    def __init__(self, model: nn.Module) -> None:
        self.feature_maps = None

        last_conv_layer = self.find_last_conv_layer(model)

        self.hook_handle = last_conv_layer.register_forward_hook(
            self.save_feature_maps
        )

    @staticmethod
    def find_last_conv_layer(model: nn.Module) -> nn.Conv2d:
        last_conv_layer = None

        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv_layer = module

        if last_conv_layer is None:
            raise ValueError(
                "모델에서 Convolution Layer를 찾을 수 없습니다."
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
        self.hook_handle.remove()


def find_classifier_layer(model: nn.Module) -> nn.Linear:
    """
    모델에서 마지막 Linear Layer를 찾는다.
    """

    last_linear_layer = None

    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear_layer = module

    if last_linear_layer is None:
        raise ValueError(
            "모델에서 Linear Layer를 찾을 수 없습니다."
        )

    return last_linear_layer


def generate_cam(
    feature_maps: torch.Tensor,
    classifier_weights: torch.Tensor,
    class_index: int,
    output_size: tuple[int, int],
) -> np.ndarray:
    """
    특정 클래스에 대한 CAM을 계산한다.

    CAM_c(x, y) = sum_k w_k^c * f_k(x, y)
    """

    feature_maps = feature_maps[0]

    class_weights = classifier_weights[class_index]

    cam = torch.sum(
        class_weights[:, None, None] * feature_maps,
        dim=0,
    )

    cam = torch.relu(cam)

    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )

    cam = cam.squeeze()

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
    원본 이미지와 CAM Heatmap을 합성한다.
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


def save_comparison_figure(
    original_image: np.ndarray,
    predicted_cam: np.ndarray,
    true_cam: np.ndarray,
    predicted_overlay: np.ndarray,
    true_overlay: np.ndarray,
    true_class_name: str,
    predicted_class_name: str,
    predicted_confidence: float,
    true_class_probability: float,
    save_path: Path,
) -> None:
    """
    원본 이미지와 Predicted/True CAM을 하나의 그림으로 저장한다.
    """

    figure, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4),
    )

    axes[0].imshow(original_image)
    axes[0].set_title(
        f"Original\n"
        f"True: {true_class_name}\n"
        f"Pred: {predicted_class_name}"
    )
    axes[0].axis("off")

    axes[1].imshow(
        predicted_cam,
        cmap="jet",
    )
    axes[1].set_title(
        f"Predicted CAM\n"
        f"Target: {predicted_class_name}"
    )
    axes[1].axis("off")

    axes[2].imshow(predicted_overlay)
    axes[2].set_title(
        f"Predicted Overlay\n"
        f"P({predicted_class_name})="
        f"{predicted_confidence:.2%}"
    )
    axes[2].axis("off")

    axes[3].imshow(
        true_cam,
        cmap="jet",
    )
    axes[3].set_title(
        f"True CAM\n"
        f"Target: {true_class_name}"
    )
    axes[3].axis("off")

    axes[4].imshow(true_overlay)
    axes[4].set_title(
        f"True Overlay\n"
        f"P({true_class_name})="
        f"{true_class_probability:.2%}"
    )
    axes[4].axis("off")

    figure.tight_layout()

    figure.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find misclassified CIFAR-10 samples and "
            "compare predicted CAM with true CAM"
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./results/checkpoints/best_model.pth",
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=20,
        help="저장할 오분류 샘플 개수",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="테스트 데이터 탐색 시작 인덱스",
    )

    parser.add_argument(
        "--result-dir",
        type=str,
        default="./results/cam_misclassified",
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

    result_dir = Path(args.result_dir)

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("Misclassified CAM Comparison")
    print("=" * 70)
    print(f"Device            : {device}")
    print(f"Checkpoint        : {checkpoint_path}")
    print(f"Start index       : {args.start_index}")
    print(f"Target image count: {args.num_images}")
    print(f"Result directory  : {result_dir}")
    print("=" * 70)

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

    saved_count = 0
    checked_count = 0

    with torch.no_grad():
        for index in range(
            args.start_index,
            len(normalized_dataset),
        ):
            if saved_count >= args.num_images:
                break

            checked_count += 1

            normalized_image, true_label = (
                normalized_dataset[index]
            )

            original_tensor, _ = original_dataset[index]

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

            if predicted_label == true_label:
                continue

            if feature_extractor.feature_maps is None:
                raise RuntimeError(
                    "Feature Map이 추출되지 않았습니다."
                )

            predicted_confidence = probabilities[
                0,
                predicted_label,
            ].item()

            true_class_probability = probabilities[
                0,
                true_label,
            ].item()

            predicted_cam = generate_cam(
                feature_maps=feature_extractor.feature_maps,
                classifier_weights=classifier_weights,
                class_index=predicted_label,
                output_size=(32, 32),
            )

            true_cam = generate_cam(
                feature_maps=feature_extractor.feature_maps,
                classifier_weights=classifier_weights,
                class_index=true_label,
                output_size=(32, 32),
            )

            original_image = (
                original_tensor
                .permute(1, 2, 0)
                .numpy()
            )

            predicted_overlay = create_overlay(
                image=original_image,
                cam=predicted_cam,
            )

            true_overlay = create_overlay(
                image=original_image,
                cam=true_cam,
            )

            true_class_name = (
                CIFAR10_CLASSES[true_label]
            )

            predicted_class_name = (
                CIFAR10_CLASSES[predicted_label]
            )

            save_name = (
                f"sample_{index:05d}_"
                f"true_{true_class_name}_"
                f"pred_{predicted_class_name}.png"
            )

            save_path = result_dir / save_name

            save_comparison_figure(
                original_image=original_image,
                predicted_cam=predicted_cam,
                true_cam=true_cam,
                predicted_overlay=predicted_overlay,
                true_overlay=true_overlay,
                true_class_name=true_class_name,
                predicted_class_name=predicted_class_name,
                predicted_confidence=predicted_confidence,
                true_class_probability=true_class_probability,
                save_path=save_path,
            )

            saved_count += 1

            print(
                f"[{saved_count:02d}/{args.num_images:02d}] "
                f"Index: {index:05d} | "
                f"True: {true_class_name:<10} | "
                f"Pred: {predicted_class_name:<10} | "
                f"P(pred): {predicted_confidence:.2%} | "
                f"P(true): {true_class_probability:.2%}"
            )

    feature_extractor.remove()

    print("=" * 70)
    print("Misclassified CAM visualization completed")
    print(f"Checked samples : {checked_count}")
    print(f"Saved samples   : {saved_count}")
    print(f"Result directory: {result_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()