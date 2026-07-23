import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# 모델 파일 위치에 따른 호환 처리
try:
    from cam_cnn import CAMCNN
except ImportError:
    from models.cam_cnn import CAMCNN

# dataset.py 함수 이름에 따른 호환 처리
try:
    from dataset import get_dataloaders
except ImportError as error:
    raise ImportError(
        "dataset.py에서 get_dataloaders 함수를 불러올 수 없습니다."
    ) from error


def set_seed(seed: int = 42) -> None:
    """실행할 때마다 최대한 비슷한 결과가 나오도록 Seed를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """Batch별 값을 누적해서 전체 평균을 계산한다."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, batch_size: int) -> None:
        self.total += value * batch_size
        self.count += batch_size

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0

        return self.total / self.count


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """한 Epoch 동안 모델을 학습한다."""

    model.train()

    loss_meter = AverageMeter()
    correct = 0
    total = 0

    progress_bar = tqdm(
        train_loader,
        desc="Training",
        leave=False,
    )

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        # 이전 Batch에서 계산된 Gradient 제거
        optimizer.zero_grad()

        # Forward
        logits = model(images)

        # Loss 계산
        loss = criterion(logits, labels)

        # Backpropagation
        loss.backward()

        # Weight 업데이트
        optimizer.step()

        batch_size = labels.size(0)
        loss_meter.update(loss.item(), batch_size)

        predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += batch_size

        current_accuracy = 100.0 * correct / total

        progress_bar.set_postfix(
            loss=f"{loss_meter.average:.4f}",
            accuracy=f"{current_accuracy:.2f}%",
        )

    epoch_accuracy = 100.0 * correct / total

    return loss_meter.average, epoch_accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    validation_loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validation Dataset으로 현재 모델 성능을 평가한다."""

    model.eval()

    loss_meter = AverageMeter()
    correct = 0
    total = 0

    progress_bar = tqdm(
        validation_loader,
        desc="Validation",
        leave=False,
    )

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        loss_meter.update(loss.item(), batch_size)

        predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += batch_size

    epoch_accuracy = 100.0 * correct / total

    return loss_meter.average, epoch_accuracy


def save_training_graphs(
    history: dict,
    result_dir: Path,
) -> None:
    """Loss와 Accuracy 그래프를 저장한다."""

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        history["validation_loss"],
        label="Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        result_dir / "loss_curve.png",
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Training Accuracy",
    )
    plt.plot(
        epochs,
        history["validation_accuracy"],
        label="Validation Accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        result_dir / "accuracy_curve.png",
        dpi=200,
    )
    plt.close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CIFAR-10 CAM CNN Training"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--result-dir",
        type=str,
        default="./results",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("CAM CNN Training")
    print("=" * 60)
    print(f"Device        : {device}")
    print(f"Epochs        : {args.epochs}")
    print(f"Learning rate : {args.learning_rate}")
    print("=" * 60)

    # dataset.py에서 DataLoader 불러오기
    loaders = get_dataloaders()

    # dataset.py 반환 형식 처리
    if len(loaders) == 3:
        train_loader, validation_loader, test_loader = loaders
    elif len(loaders) == 2:
        train_loader, validation_loader = loaders
        test_loader = None
    else:
        raise ValueError(
            "get_dataloaders()는 2개 또는 3개의 DataLoader를 반환해야 합니다."
        )

    model = CAMCNN(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    result_dir = Path(args.result_dir)
    checkpoint_dir = result_dir / "checkpoints"

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    best_validation_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_loss, validation_accuracy = validate(
            model=model,
            validation_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(validation_loss)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(
            validation_accuracy
        )

        current_learning_rate = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.2f}% | "
            f"LR: {current_learning_rate:.6f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_accuracy": validation_accuracy,
            "history": history,
        }

        # 가장 최근 모델
        torch.save(
            checkpoint,
            checkpoint_dir / "last_model.pth",
        )

        # Validation Accuracy가 가장 높은 모델
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

            torch.save(
                checkpoint,
                checkpoint_dir / "best_model.pth",
            )

            print(
                f"→ Best model saved: "
                f"{best_validation_accuracy:.2f}%"
            )

    history_path = result_dir / "training_history.json"

    with history_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=4,
        )

    save_training_graphs(
        history=history,
        result_dir=result_dir,
    )

    print("=" * 60)
    print("Training completed")
    print(
        f"Best Validation Accuracy : "
        f"{best_validation_accuracy:.2f}%"
    )
    print(
        f"Best Model Path          : "
        f"{checkpoint_dir / 'best_model.pth'}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()