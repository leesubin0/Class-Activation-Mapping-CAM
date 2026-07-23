import torch
import torch.nn as nn


class CAMCNN(nn.Module):
    """
    CNN architecture for Class Activation Mapping (CAM)

    Conv -> GAP -> Linear
    """

    def __init__(self, num_classes=10):

        super().__init__()

        # -----------------------------
        # Feature Extractor
        # -----------------------------
        self.features = nn.Sequential(

            # Input : [B,3,32,32]

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),

            # [B,32,16,16]

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),

            # [B,64,8,8]

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()

            # Output
            # [B,128,8,8]
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        # Linear classifier
        self.classifier = nn.Linear(
            128,
            num_classes
        )

    def forward(self,x,return_features=False):

        feature_maps = self.features(x)

        pooled = self.gap(feature_maps)

        pooled = pooled.view(
            pooled.size(0),
            -1
        )

        logits = self.classifier(pooled)

        if return_features:
            return logits, feature_maps

        return logits