import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from tqdm import tqdm

try:
    from cam_cnn import CAMCNN
except ImportError:
    from models.cam_cnn import CAMCNN

from dataset import CIFAR10_CLASSES, get_dataloaders


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CAM CNN on CIFAR-10"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./results/checkpoints/best_model.pth",
    )

    parser.add_argument(
        "--result-dir",
        type=str,
        default="./results/evaluation",
    )

    return parser.parse_args()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    progress_bar = tqdm(
        test_loader,
        desc="Testing",
    )

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total += batch_size

        all_labels.extend(
            labels.cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return (
        average_loss,
        accuracy,
        np.array(all_labels),
        np.array(all_predictions),
    )


def save_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: Path,
) -> None:
    matrix = confusion_matrix(
        labels,
        predictions,
    )

    figure, axis = plt.subplots(
        figsize=(10, 10)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CIFAR10_CLASSES,
    )

    display.plot(
        ax=axis,
        xticks_rotation=45,
        values_format="d",
    )

    axis.set_title(
        "CIFAR-10 Confusion Matrix"
    )

    figure.tight_layout()

    figure.savefig(
        save_path,
        dpi=200,
    )

    plt.close(figure)


def main() -> None:
    args = parse_arguments()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("CAM CNN Test Evaluation")
    print("=" * 60)
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print("=" * 60)

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint를 찾을 수 없습니다: {checkpoint_path}"
        )

    _, _, test_loader = get_dataloaders()

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

    criterion = nn.CrossEntropyLoss()

    (
        test_loss,
        test_accuracy,
        labels,
        predictions,
    ) = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    result_dir = Path(args.result_dir)

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = classification_report(
        labels,
        predictions,
        target_names=CIFAR10_CLASSES,
        digits=4,
    )

    report_path = (
        result_dir / "classification_report.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    save_confusion_matrix(
        labels=labels,
        predictions=predictions,
        save_path=result_dir / "confusion_matrix.png",
    )

    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.2f}%")
    print()
    print(report)

    print("=" * 60)
    print("Evaluation completed")
    print(
        f"Classification Report : {report_path}"
    )
    print(
        "Confusion Matrix      : "
        f"{result_dir / 'confusion_matrix.png'}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()