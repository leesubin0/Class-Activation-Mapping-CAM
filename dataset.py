from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_train_transform() -> transforms.Compose:
    """
    학습 데이터용 전처리.

    RandomCrop과 RandomHorizontalFlip을 적용하여
    데이터 다양성을 높이고 과적합을 완화한다.
    """
    return transforms.Compose(
        [
            transforms.RandomCrop(
                size=32,
                padding=4,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=CIFAR10_MEAN,
                std=CIFAR10_STD,
            ),
        ]
    )


def get_test_transform() -> transforms.Compose:
    """
    Validation/Test 데이터용 전처리.

    평가 데이터에는 무작위 증강을 적용하지 않는다.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=CIFAR10_MEAN,
                std=CIFAR10_STD,
            ),
        ]
    )


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    CIFAR-10 DataLoader 생성.

    CIFAR-10 학습 데이터 50,000장:
        Training   45,000장
        Validation  5,000장

    CIFAR-10 테스트 데이터:
        Test       10,000장
    """

    # 학습용 증강이 적용된 전체 Training Dataset
    full_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_train_transform(),
    )

    # Validation에는 무작위 증강이 없어야 하므로
    # 동일한 CIFAR-10 학습 데이터를 별도로 불러온다.
    full_validation_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_test_transform(),
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=get_test_transform(),
    )

    train_size = 45_000
    validation_size = 5_000

    generator = torch.Generator().manual_seed(seed)

    # 동일한 Seed를 사용하여 Training과 Validation의 Index를 결정
    train_subset, validation_index_subset = random_split(
        range(len(full_train_dataset)),
        lengths=[train_size, validation_size],
        generator=generator,
    )

    train_indices = train_subset.indices
    validation_indices = validation_index_subset.indices

    train_dataset = torch.utils.data.Subset(
        full_train_dataset,
        train_indices,
    )

    validation_dataset = torch.utils.data.Subset(
        full_validation_dataset,
        validation_indices,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, validation_loader, test_loader


def print_dataset_information(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
) -> None:
    """Dataset과 첫 번째 Batch 정보를 출력한다."""

    print("=" * 60)
    print("CIFAR-10 Dataset Information")
    print("=" * 60)

    print(
        f"Training samples   : "
        f"{len(train_loader.dataset):,}"
    )
    print(
        f"Validation samples : "
        f"{len(validation_loader.dataset):,}"
    )
    print(
        f"Test samples       : "
        f"{len(test_loader.dataset):,}"
    )

    print("-" * 60)

    print(
        f"Training batches   : "
        f"{len(train_loader):,}"
    )
    print(
        f"Validation batches : "
        f"{len(validation_loader):,}"
    )
    print(
        f"Test batches       : "
        f"{len(test_loader):,}"
    )

    print("=" * 60)

    images, labels = next(iter(train_loader))

    print("\n[First Training Batch]")
    print(f"Image tensor shape : {images.shape}")
    print(f"Image dtype        : {images.dtype}")
    print(f"Label dtype        : {labels.dtype}")
    print(f"Minimum value      : {images.min().item():.4f}")
    print(f"Maximum value      : {images.max().item():.4f}")

    print("\n[Sample Labels]")

    for index in range(min(10, len(labels))):
        label_index = labels[index].item()
        class_name = CIFAR10_CLASSES[label_index]

        print(
            f"Sample {index:02d}: "
            f"{label_index} ({class_name})"
        )


def main() -> None:
    train_loader, validation_loader, test_loader = (
        get_dataloaders()
    )

    print_dataset_information(
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
    )


if __name__ == "__main__":
    main()